"""NKI-N1: the NEURON device skeleton (fusezap/device/neuron.py) and its NKISHIM test tier (docs/NKI_BACKEND.md §4, §6).

The op suite runs on the tier with `FZ_TEST_TIERS=NKISHIM uv run pytest tests/test_ops.py`; a kernel the renderer refuses
reports as xfailed with its NKIUnsupported reason (tests/conftest.py). These tests check the device itself.
"""
from __future__ import annotations
import os, subprocess, sys
import numpy as np
import pytest
from fusezap import Tensor, dtypes
from fusezap.codegen.nki import NKIUnsupported
from fusezap.device import Device

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@pytest.fixture
def shim(monkeypatch):
  from fusezap import helpers
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "NKISHIM", raising=False)
  yield Device["NKISHIM"]
  helpers.getenv.cache_clear()

def test_tiers_on_this_host():
  from fusezap.device.neuron import NEURONDevice, NKISHIMDevice, NKIShimProgram
  dev = Device["NKISHIM"]
  assert isinstance(dev, NKISHIMDevice) and isinstance(dev, NEURONDevice) and dev.runtime is NKIShimProgram
  assert set(dev.custom_kernels) == {"matmul", "attention", "attention_mask"}   # NKI-N3: TensorE kernels via the hook
  from fusezap.device import neuron
  for name, why in (("NEURON", "N4"), ("NKISIM", "no nki env")):  # no nrtpy / nki here: the tiers refuse to start
    if name == "NKISIM" and neuron.nki_python().exists(): continue   # a Linux host with .venv-nki (scripts/nki_env.sh)
    if name == "NEURON" and neuron.neuron_device_present(): continue  # an inf2/trn1 instance (N4 run 1 failed here)
    assert not Device.available(name)
    with pytest.raises(RuntimeError, match=why): getattr(neuron, f"{name}Device")(name)

@pytest.mark.parametrize("tier", ["NKISHIM", "NKISIM", "NEURON"])
def test_tier_is_live_when_requested(tier):
  """FZ_TEST_TIERS=NKISHIM / NKISIM must not quietly skip every op test."""
  if tier not in [t.strip() for t in os.getenv("FZ_TEST_TIERS", "").split(",")]: pytest.skip(f"{tier} tier not requested")
  assert Device.available(tier)

def test_elementwise_reduce_and_copies(shim):
  a = np.random.default_rng(0).uniform(-2, 2, (300, 1000)).astype(np.float32)
  b = np.random.default_rng(1).uniform(-2, 2, (1000,)).astype(np.float32)
  x, y = Tensor(a), Tensor(b)
  assert x.device == "NKISHIM"
  np.testing.assert_allclose((x * 2 + y).exp().numpy(), np.exp(a * 2 + b), rtol=1e-5)
  np.testing.assert_allclose(x.sum(axis=1).numpy(), a.sum(axis=1), rtol=1e-4, atol=1e-4)
  np.testing.assert_array_equal(x.max(axis=0).numpy(), a.max(axis=0))
  z = (x + 1).to("NPY")                                # copies through host memory, like every other device
  assert z.device == "NPY"
  np.testing.assert_allclose((z * 3).to("NKISHIM").numpy(), (a + 1) * 3, rtol=1e-6)
  np.testing.assert_array_equal(Tensor(np.arange(10, dtype=np.int32)).sum().numpy(), 45)

def test_program_cache_keys_on_bound_vars(shim):
  """Every Var value is compiled in (exception 2): one program per (AST, var_vals), reused on a repeat."""
  from fusezap.jit import bound
  from fusezap.shape.symbolic import Var
  shim._cache.clear()
  n = Var("n", 1, 16)
  a = np.arange(16, dtype=np.float32)
  for k in (3, 16, 3):
    with bound({n: k}): out = (Tensor(a).shrink(((0, n),)) * 2.0).sum().numpy()
    assert out == pytest.approx(2 * a[:k].sum())
  srcs = [p.prog.src for p in shim._cache.values() if "R=(1,)" in p.prog.src]
  assert len(srcs) == 2 and any("F=(3,)" in s for s in srcs) and any("F=(16,)" in s for s in srcs)

def test_source_is_the_renderers(shim):
  from fusezap.codegen.nki import NKIRenderer
  from fusezap.schedule import create_schedule
  t = Tensor(np.zeros((4, 6), np.float32)).mean(axis=1)
  (si,) = [si for si in create_schedule([t.lazydata]) if si.kind == "kernel"]
  assert shim.program(si).prog.src == NKIRenderer().render(si.ast, si.var_vals).src
  assert shim.program(si) is shim.program(si)

def test_unsupported_is_never_faked(shim):
  with pytest.raises(NKIUnsupported, match="IDIV.*rejects"): (Tensor(np.array([7, -7, 9], np.int32)) // 2).numpy()

def test_multi_view_reads_are_materialized(shim):
  """N5: a read that composes into several views (a reshape of a permute, of a shrink) is split by the scheduler into
  single-view kernels (NEURONDevice.single_view), so it runs instead of raising NKIUnsupported; the renderer still refuses
  a multi-view ShapeTracker it is handed directly."""
  from fusezap.codegen.nki import NKIRenderer
  from fusezap.schedule import create_schedule
  a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
  x = Tensor(a).permute(1, 0, 2).reshape(3, 8)[:, :6].reshape(3, 6)
  np.testing.assert_array_equal((x + 1).numpy(), a.transpose(1, 0, 2).reshape(3, 8)[:, :6] + 1)
  y = Tensor(a, device="CPU").permute(1, 0, 2).reshape(3, 8)[:, :6].reshape(3, 6) + 1
  (si,) = [si for si in create_schedule([y.lazydata]) if si.kind == "kernel"]   # CPU keeps the one multi-view kernel
  with pytest.raises(NKIUnsupported, match="multi-view"): NKIRenderer().render(si.ast, si.var_vals)

def test_unknown_custom_items_refused(shim):
  """A custom item NEURON has no NKI kernel for (e.g. another device's library call) is refused, never run elsewhere."""
  from fusezap.ir import ScheduleItem
  from fusezap.schedule import CustomCall, CustomKernel
  call = CustomCall(CustomKernel("fz_other", {"NKISHIM": "x"}, ("C",), 1), (1, 2))
  with pytest.raises(NKIUnsupported, match="no NKI kernel for custom item 'fz_other'"): shim.run(ScheduleItem((call,), (), {}, "custom", 1))  # type: ignore[arg-type]

def test_op_suite_reports_unsupported_as_xfail():
  """The conformance output names an unsupported op (xfailed + reason) instead of hiding or faking it."""
  env = dict(os.environ, FZ_TEST_TIERS="NKISHIM")
  p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", "-rx",
                      "tests/test_ops.py::test_int_floordiv_mod", "tests/test_ops.py::test_softmax"],
                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
  out = p.stdout + p.stderr
  assert p.returncode == 0, out
  assert "1 xfailed" in out and "NKIUnsupported" in out and "test_int_floordiv_mod[NKISHIM]" in out, out
  assert " passed" in out and "failed" not in out.replace("xfailed", ""), out

# ---------------------------------------------------------------- the composed path (FZ_CUSTOM=0; NKI-N2)
# test_custom.py runs the custom kernels on the NEURON tiers (NKI-N3). With FZ_CUSTOM=0 the same matmul and attention graphs
# still run composed (MUL+SUM, softmax) as plain kernels: each must match its numpy/torch reference on the NKI tier
# (NKISHIM here, NKISIM on a Linux pod).
NKI_CUSTOM_TIERS = ["NKISHIM"] + [t for t in ("NKISIM", "NEURON") if t in os.getenv("FZ_TEST_TIERS", "")]

@pytest.fixture(params=NKI_CUSTOM_TIERS)
def nki_dev(request, monkeypatch):
  from fusezap import helpers
  if not Device.available(request.param): pytest.skip(f"{request.param} not available")
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.setenv("FZ_CUSTOM", "0")
  monkeypatch.setattr(Device, "DEFAULT", request.param, raising=False)
  yield Device[request.param]
  helpers.getenv.cache_clear()

def _custom_cases():
  import importlib
  return importlib.import_module("test_custom")

@pytest.mark.parametrize("name", list(_custom_cases().MM))
def test_custom_matmul_cases_composed(nki_dev, name):
  import torch
  tc = _custom_cases()
  fn, shapes = tc.MM[name]
  arrs = [np.random.default_rng(3).standard_normal(s).astype(np.float32) for s in shapes]
  ks = tc.kinds(fn(*[Tensor(a) for a in arrs]))   # scheduling consumes the graph: inspect one copy, run another
  assert all(k == "kernel" for k in ks), ks          # composed: no custom item reaches the device
  ref = fn(*[torch.tensor(a) for a in arrs]).numpy()
  np.testing.assert_allclose(fn(*[Tensor(a) for a in arrs]).numpy(), ref, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("qs,ks", _custom_cases().ATT, ids=str)
@pytest.mark.parametrize("causal", [False, True])
def test_custom_attention_cases_composed(nki_dev, qs, ks, causal):
  import torch
  rng = np.random.default_rng(5)
  q, k, v = (rng.standard_normal(s).astype(np.float32) for s in (qs, ks, ks))
  if causal and qs[-2] != ks[-2]: pytest.skip("causal needs square scores here")
  out = Tensor(q).scaled_dot_product_attention(Tensor(k), Tensor(v), is_causal=causal).numpy()
  ref = torch.nn.functional.scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v), is_causal=causal).numpy()
  np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

def test_custom_backward_composed(nki_dev):
  import torch
  rng = np.random.default_rng(7)
  arrs = [rng.standard_normal(s).astype(np.float32) for s in ((4, 6), (6, 5), (2, 4, 8), (2, 4, 8), (2, 4, 8))]
  ts = [Tensor(x, requires_grad=True) for x in arrs]
  rs = [torch.tensor(x, requires_grad=True) for x in arrs]
  def f(x, y, q, k, v, sdpa): return (x @ y).tanh().sum() + (sdpa(q, k, v) ** 2).sum()
  f(*ts, lambda q, k, v: q.scaled_dot_product_attention(k, v)).backward()
  f(*rs, torch.nn.functional.scaled_dot_product_attention).backward()
  for t, r in zip(ts, rs): np.testing.assert_allclose(t.grad.numpy(), r.grad.numpy(), rtol=1e-4, atol=1e-5)

def test_compile_neff_without_a_device(tmp_path):
  """NKI-N2: a rendered kernel compiles to a NEFF for inf2 with neuronx-cc and no Neuron device (Linux, nki_env.sh --cc)."""
  from fusezap.codegen.nki import NKIRenderer
  from fusezap.device import neuron
  from fusezap.schedule import create_schedule
  if not neuron.nki_python().exists(): pytest.skip("no .venv-nki (Linux x86_64 only)")
  if not neuron.NKIWorker.get().call({"op": "version"}).get("neuronxcc"): pytest.skip("neuronx-cc not installed (scripts/nki_env.sh --cc)")
  t = Tensor(np.ones((4, 6), np.float32)) * 2 + 1
  (si,) = [si for si in create_schedule([t.lazydata]) if si.kind == "kernel"]
  prog = NKIRenderer().render(si.ast, si.var_vals)
  arrays = [np.zeros(b.size, np.float32) for b in si.bufs][:prog.n_bufs]
  neff, secs = neuron.compile_neff(prog, arrays, "inf2", str(tmp_path))
  assert len(neff) > 1000 and (tmp_path / "kernel.neff").exists() and secs > 0

# ---------------------------------------------------------------- NKI-N3: matmul and attention custom kernels
NEURON_TIERS = ["NKISHIM"] + [t for t in ("NKISIM", "NEURON") if t in os.getenv("FZ_TEST_TIERS", "")]

@pytest.fixture(params=NEURON_TIERS)
def ndev(request, monkeypatch):
  from fusezap import helpers
  if not Device.available(request.param): pytest.skip(f"{request.param} not available")
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.delenv("FZ_CUSTOM", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", request.param, raising=False)
  yield Device[request.param]
  helpers.getenv.cache_clear()

def _kinds(t: Tensor) -> list[str]:
  from fusezap.schedule import create_schedule
  return [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([t.lazydata]) if si.kind != "copy"]

def _attn_ref(q, k, v, vis):
  G = q.shape[-3] // k.shape[-3]
  kk, vv = np.repeat(k, G, -3).astype(np.float64), np.repeat(v, G, -3).astype(np.float64)
  s = q.astype(np.float64) @ np.swapaxes(kk, -1, -2) / np.sqrt(q.shape[-1])
  s = np.where(vis, s, -np.inf)
  e = np.exp(s - s.max(-1, keepdims=True))
  return (e / e.sum(-1, keepdims=True)) @ vv

def _llama_attention(q, K, V, start: int):
  """fusezap/serve/models/llama.py's composed attention (GQA reshape, position mask, WHERE -inf, softmax, @ V)."""
  B, Hq, T, hd = q.shape
  Hkv, S = K.shape[1], K.shape[2]
  pos = Tensor(np.arange(start + T + 1, dtype=np.int32))
  mask = pos.shrink(((0, S),)).reshape(1, S) <= pos.shrink(((start, start + T),)).reshape(T, 1)
  s = (Tensor(q).reshape(B, Hkv, Hq // Hkv, T, hd) @ Tensor(K).reshape(B, Hkv, 1, S, hd).transpose(-1, -2)) * (1 / np.sqrt(hd))
  return (mask.where(s, float("-inf")).softmax(-1) @ Tensor(V).reshape(B, Hkv, 1, S, hd)).reshape(B, Hq, T, hd)

def test_matches_are_taken(ndev):
  """Exception 3: on NEURON a missed matmul/attention match is a bug (composed MUL+SUM never reaches TensorE)."""
  rng = np.random.default_rng(0)
  r = lambda *s: Tensor(rng.standard_normal(s).astype(np.float32))
  assert _kinds(r(5, 7) @ r(7, 3)) == ["fz_matmul"]
  assert _kinds(r(2, 3, 8).linear(r(5, 8), r(5))) == ["fz_matmul", "kernel"]          # Tensor.linear; the bias is an epilogue
  assert _kinds(r(2, 4, 6, 8).scaled_dot_product_attention(r(2, 4, 6, 8), r(2, 4, 6, 8))) == ["fz_attention"]
  assert _kinds(r(2, 4, 6, 8).scaled_dot_product_attention(r(2, 4, 6, 8), r(2, 4, 6, 8), is_causal=True))[-1] == "fz_attention_mask"
  m = Tensor(np.tril(np.ones((6, 9), bool), 3))
  assert _kinds(r(2, 6, 8).scaled_dot_product_attention(r(2, 9, 8), r(2, 9, 8), attn_mask=m)) == ["fz_attention_mask"]
  for T, S, start in ((5, 5, 0), (1, 40, 39)):   # llama.py prefill and decode, Qwen-style GQA (Hq 14, Hkv 2)
    q, K, V = (rng.standard_normal(s).astype(np.float32) for s in ((1, 14, T, 16), (1, 2, S, 16), (1, 2, S, 16)))
    assert _kinds(_llama_attention(q, K, V, start))[-1] == "fz_attention_mask"

def test_masked_pattern_is_neuron_only(monkeypatch):
  """The other devices keep masked attention composed (they do not list "attention_mask"): behaviour there is unchanged."""
  from fusezap import helpers
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_CUSTOM", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "NPY", raising=False)
  q = Tensor(np.ones((2, 4, 8), np.float32))
  ks = _kinds(q.scaled_dot_product_attention(Tensor(np.ones((2, 4, 8), np.float32)), Tensor(np.ones((2, 4, 8), np.float32)), is_causal=True))
  assert "fz_attention_mask" not in ks and "fz_matmul" in ks

@pytest.mark.parametrize("T,S,start", [(5, 5, 0), (1, 40, 39), (3, 9, 6), (130, 130, 0)])
def test_llama_graph_gqa(ndev, T, S, start):
  rng = np.random.default_rng(T)
  q, K, V = (rng.standard_normal(s).astype(np.float32) for s in ((1, 14, T, 16), (1, 2, S, 16), (1, 2, S, 16)))
  vis = np.arange(S)[None, :] <= np.arange(start, start + T)[:, None]
  np.testing.assert_allclose(_llama_attention(q, K, V, start).numpy(), _attn_ref(q, K, V, vis), rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("Hq,Hkv,T,S,hd", [(4, 4, 7, 7, 8), (14, 2, 9, 9, 64), (28, 4, 3, 600, 128), (2, 1, 129, 140, 16)])
@pytest.mark.parametrize("mode", ["none", "causal", "pos"])
def test_direct_attention(ndev, Hq, Hkv, T, S, hd, mode):
  """neuron.attention: GQA, causal via affine_select (coff aligns the last query with the last key), decode/chunk masks from
  an int32 position tensor, key blocks of 512 and row tiles of 128 with ragged tails."""
  from fusezap.device import neuron
  rng = np.random.default_rng(S)
  q, K, V = (rng.standard_normal(s).astype(np.float32) for s in ((2, Hq, T, hd), (2, Hkv, S, hd), (2, Hkv, S, hd)))
  coff = S - T
  vis = np.ones((T, S), bool) if mode == "none" else np.arange(S)[None, :] <= np.arange(T)[:, None] + coff
  kw = {"causal": True, "coff": coff} if mode == "causal" else {"pos": Tensor(np.arange(coff, coff + T, dtype=np.int32))} if mode == "pos" else {}
  out = neuron.attention(Tensor(q), Tensor(K), Tensor(V), **kw).numpy()
  np.testing.assert_allclose(out, _attn_ref(q, K, V, vis), rtol=1e-5, atol=1e-5)

def test_direct_attention_causal_matches_torch(ndev):
  import torch
  from fusezap.device import neuron
  rng = np.random.default_rng(1)
  q, k, v = (rng.standard_normal((3, 600, 32)).astype(np.float32) for _ in range(3))
  ref = torch.nn.functional.scaled_dot_product_attention(*(torch.tensor(x) for x in (q, k, v)), is_causal=True).numpy()
  np.testing.assert_allclose(neuron.attention(Tensor(q), Tensor(k), Tensor(v), causal=True).numpy(), ref, rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize("M,K,N", [(1, 896, 1152), (128, 896, 1152), (3, 4864, 896), (130, 300, 1030)])
def test_matmul_qwen_shapes(ndev, M, K, N):
  """x.linear(w) and x @ kmajor(w) at Qwen2.5-0.5B linear shapes (and ragged tiles): the hook's fz_matmul, K-major or torch layout."""
  from fusezap.device import neuron
  rng = np.random.default_rng(M)
  x, w = rng.uniform(-1, 1, (M, K)).astype(np.float32), (rng.uniform(-1, 1, (N, K)) / np.sqrt(K)).astype(np.float32)
  ref = x.astype(np.float64) @ w.T.astype(np.float64)
  for t in (Tensor(x).linear(Tensor(w)), Tensor(x) @ neuron.kmajor(Tensor(w))):
    np.testing.assert_allclose(t.numpy(), ref, rtol=1e-5, atol=2e-6)
  ps = [si.ast[0].params for si in __import__("fusezap.schedule", fromlist=["x"]).create_schedule([(Tensor(x) @ neuron.kmajor(Tensor(w))).lazydata]) if si.kind == "custom"]
  assert len(ps) == 1 and ps[0][8:10] == (N, 1)    # K-major: moving rows are contiguous (sbk = N, sbn = 1)

def test_matmul_flipped_operand(ndev):
  """A negative partition step (here K reversed) becomes one DMA per row: neuronx-cc rejects it in an HBM access pattern."""
  rng = np.random.default_rng(2)
  a, b = rng.standard_normal((6, 200)).astype(np.float32), rng.standard_normal((200, 5)).astype(np.float32)
  t = Tensor(a).flip(1) @ Tensor(b)
  assert _kinds(Tensor(a).flip(1) @ Tensor(b)) == ["fz_matmul"]
  np.testing.assert_allclose(t.numpy(), a[:, ::-1] @ b, rtol=1e-5, atol=1e-5)

def test_shim_tensore_semantics():
  """The shim's nc_matmul / nc_transpose / iota / affine_select follow the 0.6 simulator, and refuse what the hardware cannot do."""
  from fusezap.device.nkishim import isa, ndarray, nl
  rng = np.random.default_rng(3)
  a = ndarray((128, 64), nl.float32); a.write_raw(rng.standard_normal((128, 64)).astype(np.float32))
  b = ndarray((128, 300), nl.float32); b.write_raw(rng.standard_normal((128, 300)).astype(np.float32))
  ps = ndarray((64, 300), nl.float32, nl.psum)
  isa.nc_matmul(ps, a, b, accumulate=False); isa.nc_matmul(ps, a, b, accumulate=True)
  np.testing.assert_allclose(ps.f32(), 2 * (a.f32().T @ b.f32()), rtol=1e-5, atol=1e-4)
  tp = ndarray((300 - 172, 64), nl.float32, nl.psum)
  isa.nc_transpose(tp, b[0:64, 172:300])
  np.testing.assert_array_equal(tp.f32(), b.f32()[:64, 172:].T)
  io = ndarray((4, 6), nl.int32)
  isa.iota(io, [[1, 6]], offset=10, channel_multiplier=100)
  np.testing.assert_array_equal(io.values(), 10 + 100 * np.arange(4)[:, None] + np.arange(6)[None, :])
  x = ndarray((4, 6), nl.float32); x.write_raw(np.ones((4, 6), np.float32))
  sel = ndarray((4, 6), nl.float32)
  isa.affine_select(sel, [[-1, 6]], 1, x, -np.inf, nl.greater_equal, offset=1)
  np.testing.assert_array_equal(np.isfinite(sel.f32()), np.arange(4)[:, None] + 1 - np.arange(6)[None, :] >= 0)
  with pytest.raises(AssertionError, match="N <= 512"): isa.nc_matmul(ndarray((64, 513), nl.float32, nl.psum), a, ndarray((128, 513), nl.float32), accumulate=False)
  with pytest.raises(AssertionError, match="reads SBUF and writes PSUM"): isa.nc_matmul(ndarray((64, 300), nl.float32), a, b, accumulate=False)
  with pytest.raises(AssertionError, match="accumulate=True/False"): isa.nc_matmul(ps, a, b)
  with pytest.raises(AssertionError, match="PSUM tile exceeds"): ndarray((128, 4097), nl.float32, nl.psum)

def test_custom_kernels_compile_to_neff(tmp_path):
  """NKI-N3: the matmul and attention kernels (each mask mode) compile to inf2 NEFFs (Linux with scripts/nki_env.sh --cc)."""
  from fusezap.codegen import nki_kernels as nk
  from fusezap.device import neuron
  if not neuron.nki_python().exists(): pytest.skip("no .venv-nki (Linux x86_64 only)")
  if not neuron.NKIWorker.get().call({"op": "version"}).get("neuronxcc"): pytest.skip("neuronx-cc not installed (scripts/nki_env.sh --cc)")
  progs = [(nk.render_matmul((1, 130, 600, 300, 0, 300, 1, 0, 600, 1, 0, 600, 1, 0, 0)), [130 * 600, 130 * 300, 300 * 600])]
  for mode, m in (("none", ()), ("causal", ()), ("pos", (0, 1)), ("tensor", (0, 0, 0, 40, 1, 0))):
    sp = nk.AttnSpec((2, 1), (7, 3) if mode != "causal" else (1, 3), 40, 64, 64, (7 * 3 * 64, 0, 64, 1, 0), (40 * 64, 0, 64, 1, 0),
                     (40 * 64, 0, 64, 1, 0), (7 * 3 * 64, 0, 64, 1, 0), 0.125, mode, m, 37)
    sp = sp if mode != "causal" else nk.AttnSpec((2, 7), (1, 3), 40, 64, 64, (7 * 3 * 64, 3 * 64, 64, 1, 0), (40 * 64, 0, 64, 1, 0),
                                                 (40 * 64, 0, 64, 1, 0), (7 * 3 * 64, 3 * 64, 64, 1, 0), 0.125, mode, m, 37)
    sizes = [2 * 7 * 3 * 64, 2 * 7 * 3 * 64, 2 * 40 * 64, 2 * 40 * 64] + ([3] if mode == "pos" else [3 * 40] if mode == "tensor" else [])
    progs.append((nk.render_attention(sp), sizes))
  for prog, sizes in progs:
    arrays = [np.zeros(n, np.int32 if prog.buf_dtypes[i] == dtypes.int32 else np.uint8 if prog.buf_dtypes[i] == dtypes.bool else np.float32)
              for i, n in enumerate(sizes)]
    neff, secs = neuron.compile_neff(prog, arrays, "inf2", str(tmp_path / prog.name))
    assert len(neff) > 1000 and secs > 0, prog.name
