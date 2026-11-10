"""Custom kernels (SPEC §11.1, P2-CUSTOM): pattern matching into ScheduleItem(kind="custom"), fallback to composed ops,
and custom == composed for the NPY (numpy) and CPU (plain C) reference kernels, and for NEURON's NKI kernels (NKI-N3,
fusezap/codegen/nki_kernels.py) on the NKISHIM tier (always) and NKISIM (the real nki simulator, when FZ_TEST_TIERS names it).
NEURON also offers the "attention_mask" pattern, so is_causal and bool-mask attention become fz_attention_mask there.

Also a pytest plugin: `FZ_CUSTOM_CPU=1 PYTHONPATH=tests uv run pytest -p test_custom tests/test_ops.py` runs the op suite
with the reference kernels installed on the CPU device (NPY offers them by default).
"""
from __future__ import annotations
import os, subprocess, sys
import numpy as np
import pytest
import torch
from fusezap import Tensor, dtypes, helpers
from fusezap.device import Device
from fusezap.schedule import create_schedule, install_custom_kernels, REFERENCE_KERNELS, CustomCall

_ran = [0]
def pytest_configure(config):  # only active when loaded as a plugin with -p
  if not (os.getenv("FZ_CUSTOM_CPU") and Device.available("CPU")): return
  d = Device["CPU"]
  install_custom_kernels(d)
  run = d.run
  def counted(si):
    _ran[0] += si.kind == "custom"
    return run(si)
  counted._fz_custom = True
  d.run = counted
def pytest_terminal_summary(terminalreporter):
  if os.getenv("FZ_CUSTOM_CPU"): terminalreporter.write_line(f"custom items run: {_ran[0]}")

R = np.random.default_rng(11)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEURON_TIERS = ("NKISHIM", "NKISIM", "NEURON")
TIERS = [t for t in ("NPY", "CPU") if Device.available(t)] + \
        [t for t in NEURON_TIERS if (t == "NKISHIM" or t in os.getenv("FZ_TEST_TIERS", "")) and Device.available(t)]

@pytest.fixture(params=TIERS)
def dev(request, monkeypatch):
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.delenv("FZ_CUSTOM", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", request.param, raising=False)
  if request.param == "CPU":
    d = Device["CPU"]
    old = d.__dict__.copy()
    install_custom_kernels(d)
    yield request.param
    for k in ("custom_kernels", "run"):
      if k in old: setattr(d, k, old[k])
      else: d.__dict__.pop(k, None)
  else: yield request.param
  helpers.getenv.cache_clear()

def rnd(*shape) -> np.ndarray: return R.standard_normal(shape).astype(np.float32)
def kinds(t: Tensor) -> list[str]:
  return [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([t.lazydata]) if si.kind != "copy"]

def both(fn, monkeypatch, *arrs):
  """(custom, composed) outputs of fn on fresh tensors."""
  outs = []
  for flag in ("1", "0"):
    monkeypatch.setenv("FZ_CUSTOM", flag)
    outs.append(fn(*[Tensor(a) for a in arrs]).numpy())
  monkeypatch.delenv("FZ_CUSTOM")
  return outs

def assert_same(dev: str, c: np.ndarray, p: np.ndarray):
  # the reference kernels repeat the composed op order: CPU is bit-identical, NPY within SPEC §10 matmul tolerance
  if dev == "CPU": np.testing.assert_array_equal(c, p)
  else: np.testing.assert_allclose(c, p, rtol=1e-4, atol=1e-5)

# ------------------------------------------------------------------ matmul
MM = {
  "2d": (lambda a, b: a @ b, [(5, 7), (7, 3)]),
  "batched": (lambda a, b: a @ b, [(2, 3, 5, 4), (2, 3, 4, 6)]),
  "broadcast_b": (lambda a, b: a @ b, [(2, 3, 5, 4), (4, 6)]),
  "broadcast_batch": (lambda a, b: a @ b, [(2, 1, 5, 4), (3, 4, 6)]),  # M axes split by an N axis: falls back
  "transposed": (lambda a, b: a.T @ b.T, [(4, 5), (6, 4)]),
  "linear": (lambda a, b: a @ b.transpose(-1, -2) + 1, [(2, 3, 8), (5, 8)]),   # Tensor.linear's graph
  "vec_mat": (lambda a, b: a @ b, [(4,), (4, 6)]),
  "mat_vec": (lambda a, b: a @ b, [(6, 4), (4,)]),
  "k1": (lambda a, b: a @ b, [(5, 1), (1, 4)]),
  "outer_1x1": (lambda a, b: a @ b, [(1, 9), (9, 1)]),
  "strided": (lambda a, b: a[::2, 1:] @ b.permute(1, 0)[:5], [(7, 6), (4, 9)]),
  "chain_epilogue": (lambda a, b: ((a.exp() @ b) * 2 + 1) @ b.T, [(3, 4), (4, 4)]),
  "big": (lambda a, b: a @ b, [(65, 130), (130, 33)]),
}

@pytest.mark.parametrize("name", list(MM))
def test_matmul_custom_equals_composed(dev, monkeypatch, name):
  fn, shapes = MM[name]
  arrs = [rnd(*s) for s in shapes]
  k = kinds(fn(*[Tensor(a) for a in arrs]))
  assert ("fz_matmul" in k) == (name not in ("outer_1x1", "broadcast_batch")), k
  c, p = both(fn, monkeypatch, *arrs)
  assert_same(dev, c, p)
  ref = fn(*[torch.tensor(a) for a in arrs]).numpy()
  np.testing.assert_allclose(c, ref, rtol=1e-4, atol=1e-5)

def test_matmul_schedule_shape(dev):
  a, b = Tensor(rnd(4, 5)), Tensor(rnd(5, 6))
  s = [si for si in create_schedule([((a @ b) + 1).lazydata]) if si.kind != "copy"]
  assert [si.kind for si in s] == ["custom", "kernel"]           # the epilogue reads the custom output
  si = s[0]
  assert isinstance(si.ast[0], CustomCall) and si.ast[0].kernel is Device[dev].custom_kernels["matmul"] and si.var_vals == {}
  if dev not in NEURON_TIERS: assert si.ast[0].kernel is REFERENCE_KERNELS["matmul"]
  assert si.bufs[0].size == 24 and len(si.bufs) == 3 and si.n_outputs == 1
  assert dict(zip(si.ast[0].kernel.signature[3:7], si.ast[0].params[:4])) == {"Bt": 1, "M": 4, "N": 6, "K": 5}

def test_prologue_is_materialized(dev, monkeypatch):
  a, b = rnd(3, 4), rnd(4, 5)
  assert kinds(Tensor(a).exp() @ Tensor(b)) == ["kernel", "fz_matmul"]
  c, p = both(lambda x, y: x.exp() @ y, monkeypatch, a, b)
  assert_same(dev, c, p)

# ------------------------------------------------------------------ fallback to composed ops
FALLBACK = {
  "dot_sum": lambda a, b: (a * b).sum(1),          # nothing broadcast: stays a fused reduce
  "padded": lambda a, b: a.pad(((0, 0), (1, 0)))[:, :4] @ b,  # masked operand view
  "int": lambda a, b: (a * 4).cast(dtypes.int32) @ (b * 4).cast(dtypes.int32),
  "half": lambda a, b: a.half() @ b.half(),         # composed in f32 after a cast prologue; the casts are materialized
  "const_operand": lambda a, b: a @ Tensor.ones(4, 4),
}

@pytest.mark.parametrize("name", list(FALLBACK))
def test_fallback_composed(dev, monkeypatch, name):
  fn = FALLBACK[name]
  a, b = rnd(4, 4), rnd(4, 4)
  k = kinds(fn(Tensor(a), Tensor(b)))
  if name in ("dot_sum", "padded", "int", "const_operand"): assert "fz_matmul" not in k and "custom" not in k
  c, p = both(fn, monkeypatch, a, b)
  np.testing.assert_allclose(np.asarray(c, np.float32), np.asarray(p, np.float32), rtol=1e-3, atol=1e-3)

def test_disabled_by_env(dev, monkeypatch):
  monkeypatch.setenv("FZ_CUSTOM", "0")
  assert kinds(Tensor(rnd(3, 4)) @ Tensor(rnd(4, 5))) == ["kernel"]

def test_device_without_custom_kernels(monkeypatch):
  for name in ("CPU", "METAL"):
    if not Device.available(name) or getattr(Device[name], "custom_kernels", None): continue  # METAL offers its own (P2-MTL)
    monkeypatch.setattr(Device, "DEFAULT", name, raising=False)
    a, b = rnd(3, 4), rnd(4, 5)
    t = Tensor(a) @ Tensor(b)
    assert kinds(t) == ["kernel"]
    np.testing.assert_allclose((Tensor(a) @ Tensor(b)).numpy(), a @ b, rtol=1e-4, atol=1e-5)

# ------------------------------------------------------------------ attention
ATT = [((1, 4, 8), (1, 4, 8)), ((2, 4, 6, 8), (2, 4, 6, 8)), ((2, 2, 1, 4), (2, 2, 1, 4)), ((2, 3, 5, 16), (2, 3, 7, 16)),
       ((3, 9, 32), (3, 17, 32))]

@pytest.mark.parametrize("qs,ks", ATT, ids=str)
def test_attention_custom(dev, monkeypatch, qs, ks):
  q, k, v = rnd(*qs), rnd(*ks), rnd(*ks)
  fn = lambda q, k, v: q.scaled_dot_product_attention(k, v)
  assert kinds(fn(Tensor(q), Tensor(k), Tensor(v))) == ["fz_attention"]
  c, p = both(fn, monkeypatch, q, k, v)
  assert_same(dev, c, p)
  ref = torch.nn.functional.scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v)).numpy()
  np.testing.assert_allclose(c, ref, rtol=1e-4, atol=1e-5)

def test_attention_composed_forms(dev, monkeypatch):
  q, k, v = rnd(2, 5, 8), rnd(2, 5, 8), rnd(2, 5, 3)
  # explicit softmax(q k^T) v with no scale, and through a transposed key view
  fn = lambda q, k, v: (q @ k.transpose(-1, -2)).softmax(-1) @ v
  assert kinds(fn(Tensor(q), Tensor(k), Tensor(v))) == ["fz_attention"]
  c, p = both(fn, monkeypatch, q, k, v)
  assert_same(dev, c, p)
  ref = (torch.tensor(q) @ torch.tensor(k).transpose(-1, -2)).softmax(-1) @ torch.tensor(v)
  np.testing.assert_allclose(c, ref.numpy(), rtol=1e-4, atol=1e-5)

@pytest.mark.parametrize("kind", ["causal", "mask", "softmax_other_axis", "scores_used_twice"])
def test_attention_fallback(dev, monkeypatch, kind):
  q, k, v = rnd(2, 6, 8), rnd(2, 6, 8), rnd(2, 6, 8)
  m = R.uniform(size=(6, 6)) > 0.3
  np.fill_diagonal(m, True)
  fns = {
    "causal": lambda q, k, v: q.scaled_dot_product_attention(k, v, is_causal=True),
    "mask": lambda q, k, v: q.scaled_dot_product_attention(k, v, attn_mask=Tensor(m)),
    "softmax_other_axis": lambda q, k, v: (q @ k.transpose(-1, -2)).softmax(-2) @ v,
    "scores_used_twice": lambda q, k, v: (q @ k.transpose(-1, -2)).softmax(-1) @ v + (q @ k.transpose(-1, -2)).sum(-1, keepdim=True),
  }
  fn = fns[kind]
  ks = kinds(fn(Tensor(q), Tensor(k), Tensor(v)))
  if kind in ("causal", "mask") and dev in NEURON_TIERS: assert ks[-1] == "fz_attention_mask", ks   # NEURON's masked pattern
  elif kind != "scores_used_twice": assert "fz_attention" not in ks and "fz_matmul" in ks
  c, p = both(fn, monkeypatch, q, k, v)
  assert_same(dev, c, p)

def test_attention_masked_const_scale_falls_back(dev, monkeypatch):
  # a padded const reads 0 outside its mask: it is not a uniform scale and must not become fz_attention's scale_bits
  q, k, v = rnd(2, 4, 8), rnd(2, 4, 8), rnd(2, 4, 8)
  fn = lambda q, k, v: ((q @ k.transpose(-1, -2)) * Tensor.full((2, 4, 3), 0.5).pad(((0, 0), (0, 0), (0, 1)))).softmax(-1) @ v
  assert "fz_attention" not in kinds(fn(Tensor(q), Tensor(k), Tensor(v)))
  c, p = both(fn, monkeypatch, q, k, v)
  assert_same(dev, c, p)
  s = (q @ k.transpose(0, 2, 1)) * np.pad(np.full((2, 4, 3), 0.5, np.float32), ((0, 0), (0, 0), (0, 1)))
  e = np.exp(s - s.max(-1, keepdims=True))
  np.testing.assert_allclose(c, (e / e.sum(-1, keepdims=True)) @ v, rtol=1e-4, atol=1e-5)

@pytest.mark.parametrize("flip", [0, 1])
def test_attention_flipped_mask(dev, monkeypatch, flip):
  # a mask with flipped rows (negative row stride) stays composed on NEURON; flipped keys (negative key stride) stay fused
  q, k, v = rnd(2, 3, 6, 8), rnd(2, 3, 9, 8), rnd(2, 3, 9, 8)
  m = R.uniform(size=(6, 9)) > 0.3
  m[:, 0] = m[:, -1] = True
  fn = lambda q, k, v: q.scaled_dot_product_attention(k, v, attn_mask=Tensor(m).flip(flip))
  ks = kinds(fn(Tensor(q), Tensor(k), Tensor(v)))
  if dev in NEURON_TIERS: assert ("fz_attention_mask" in ks) == (flip == 1), ks
  c, p = both(fn, monkeypatch, q, k, v)
  assert_same(dev, c, p)
  ref = torch.nn.functional.scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v), attn_mask=torch.tensor(m).flip(flip))
  np.testing.assert_allclose(c, ref.numpy(), rtol=1e-4, atol=1e-5)

def test_replaced_kernel_recompiles():
  # a different CustomKernel under the same name on the same device must not reuse the old compiled program
  if not Device.available("CPU"): pytest.skip("CPU not available")
  from fusezap.schedule import CustomKernel, _C_MATMUL
  d = Device["CPU"]
  old = d.__dict__.copy()
  try:
    helpers.getenv.cache_clear()
    install_custom_kernels(d)
    def f():  # real buffers: CONST operands are not matched
      mk = lambda: Tensor(np.ones((3, 4), np.float32), device="CPU") @ Tensor(np.ones((4, 5), np.float32), device="CPU")
      assert kinds(mk()) == ["fz_matmul"]
      return mk().numpy()
    np.testing.assert_array_equal(f(), np.full((3, 5), 4.0, np.float32))
    m = REFERENCE_KERNELS["matmul"]
    k2 = CustomKernel(m.name, {"CPU": _C_MATMUL.replace("c[n*scn] = acc[n];", "c[n*scn] = 2.0f*acc[n];")}, m.signature, m.nbufs, m.dtypes)
    install_custom_kernels(d, {"matmul": k2})
    np.testing.assert_array_equal(f(), np.full((3, 5), 8.0, np.float32))
  finally:
    for k in ("custom_kernels", "run"):
      if k in old: setattr(d, k, old[k])
      else: d.__dict__.pop(k, None)

# ------------------------------------------------------------------ training goes through the custom kernels too
def test_backward_through_custom(dev):
  a, b, q, k, v = rnd(4, 6), rnd(6, 5), rnd(2, 4, 8), rnd(2, 4, 8), rnd(2, 4, 8)
  ts = [Tensor(x, requires_grad=True) for x in (a, b, q, k, v)]
  rs = [torch.tensor(x, requires_grad=True) for x in (a, b, q, k, v)]
  def f(x, y, q, k, v, sdpa): return (x @ y).tanh().sum() + (sdpa(q, k, v) ** 2).sum()
  f(*ts, lambda q, k, v: q.scaled_dot_product_attention(k, v)).backward()
  f(*rs, torch.nn.functional.scaled_dot_product_attention).backward()
  for t, r in zip(ts, rs): np.testing.assert_allclose(t.grad.numpy(), r.grad.numpy(), rtol=1e-4, atol=1e-5)

# ------------------------------------------------------------------ the op suite with the CPU reference kernels installed
def test_op_suite_cpu_custom():
  if not Device.available("CPU"): pytest.skip("CPU not available")
  env = dict(os.environ, FZ_CUSTOM_CPU="1", FZ_TEST_TIERS="CPU", PYTHONPATH=os.path.join(ROOT, "tests"))
  sel = "matmul or linear or sdpa or cross_entropy or one_hot or layernorm or backward_semantics"
  p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "test_custom", "-p", "no:cacheprovider", "tests/test_ops.py",
                      "-k", sel], cwd=ROOT, env=env, capture_output=True, text=True)
  assert p.returncode == 0, p.stdout[-4000:] + p.stderr[-2000:]
  n = int(p.stdout.split("custom items run: ")[1].split()[0])
  assert n > 50, p.stdout[-2000:]
