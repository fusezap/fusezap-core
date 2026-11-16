"""I105 A2: sparse MoE decode on NEURON (FZ_MOE_SPARSE=1; fusezap/codegen/nki_moe.py, fusezap/serve/models/moe.py):

  * moe.route_keys on CPU: the chosen experts in increasing order are torch.topk's (ties to the lower index), the routing
    weights route()'s bits; the sparse numpy references over a CPU dense model's layer give its expert MLP output;
  * the two kernels on NKISHIM, every FZ_MOE_IDMA variant (each under its own shim model of the indirect address), against
    the float64 references and **bit for bit** against the dense kernels (MatmulBF16 / render_matmul over every expert, the
    unchosen ones weighted 0) with FZ_NKISHIM_MM=f64, at small shapes and at OLMoE's real layer shapes (D 2048, 64 experts
    of 1024 or the sliced 512, top 8; 1 and 4 rows);
  * the model on NKISHIM: decode logits with FZ_MOE_SPARSE == the dense model's bit for bit (float32 and bf16), prefill
    stays dense, the decode capture holds the sparse kernels and no dense expert matmul; LLMEngine tokens equal;
  * the sliced tensor-parallel layout (two NKISHIM processes): dense and sparse engine tokens equal, logits within float32
    rounding of one core's;
  * the chip job script (scripts/nki_olmoe_sparse_inf2.sh): bash, launch line, cost guard, phases.
"""
from __future__ import annotations
import os, re, subprocess
import numpy as np
import pytest
from fusezap.dtype import dtypes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _keys_np(p: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
  """(sel, key) for probability rows p (ties: lower index first), the numpy form of route_keys."""
  E = p.shape[-1]
  order = np.lexsort((np.arange(E)[None, :].repeat(p.shape[0], 0), -p), axis=-1)
  rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(E)[None, :].repeat(p.shape[0], 0), -1)
  sel = rank < K
  return sel, np.where(sel, np.cumsum(sel, -1) - 1, E).astype(np.float32)

def test_route_keys_are_sorted_topk():
  torch = pytest.importorskip("torch")
  from fusezap.tensor import Tensor
  from fusezap.serve.models.moe import route, route_keys, tie_order, upper_order
  from fusezap.codegen.nki_moe import picks_ref
  rng = np.random.default_rng(0)
  p = torch.softmax(torch.tensor(rng.standard_normal((6, 64)).astype(np.float32) * 2), -1).numpy()
  p[1, 20:26] = p[1, 5]   # ties
  T = lambda x: Tensor(x, device="CPU")
  tri, up = tie_order(64, "CPU"), upper_order(64, "CPU")
  for nt in (False, True):
    w, key = (t.numpy() for t in route_keys(T(p), tri, up, 8, nt))
    assert np.array_equal(w.view(np.int32), route(T(p), tri, 8, nt).numpy().view(np.int32))
    assert ((key < 8).sum(-1) == 8).all() and set(np.unique(key)) <= set(range(8)) | {64.0}
    _, kn = _keys_np(p, 8)
    assert np.array_equal(key, kn)
  ti = torch.topk(torch.tensor(p[[0, 2, 3]]), 8, -1).indices.numpy()
  assert np.array_equal(picks_ref(key[[0, 2, 3]], 8), np.sort(ti, -1))

def test_sparse_reference_equals_dense_layer_on_cpu(tmp_path):
  """A random OLMoE's layer 0 on CPU (float32): the dense masked expert MLP (gate_up over every expert, moe_act, down) ==
  the sparse references (route_keys -> the chosen experts' gate / up, silu * up, weighted down) within float32 rounding."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from test_olmoe import _olmoe_dir
  from fusezap.tensor import Tensor
  from fusezap.serve.models import load_model
  from fusezap.serve.models.moe import route_keys, router_probs, upper_order
  from fusezap.codegen.nki_moe import gate_up_ref, down_ref
  m = load_model(_olmoe_dir(tmp_path / "m", seed=6), dtype=dtypes.float32, device="CPU", max_len=64, prefill_chunk=32)
  c, l = m.cfg, m.layers[0]
  E, K, Iw = c.n_experts, c.top_k, c.intermediate_size // c.n_experts
  h = Tensor(np.random.default_rng(1).standard_normal((5, c.hidden_size)).astype(np.float32), device="CPU")
  gu = m.linear(h, l.wgu, None)
  a = m._act(gu.shrink((None, (0, c.intermediate_size)))) * gu.shrink((None, (c.intermediate_size, 2 * c.intermediate_size)))
  want = m.linear(m._moe(h, a, l), l.wd, None).numpy()
  w, key = (t.numpy() for t in route_keys(router_probs(m, h, l), m._moe_tri, upper_order(E, "CPU"), K, c.norm_topk))
  wgu, wd = l.wgu.numpy().T, l.wd.numpy().T   # CPU keeps the torch layout: (2 E Iw, D) and (D, E Iw)
  G = gate_up_ref(h.numpy(), key, wgu, E, Iw, K, False).astype(np.float32)
  sa = (G[..., :Iw] / (1 + np.exp(-G[..., :Iw])) * G[..., Iw:]).astype(np.float32)
  got = down_ref(sa, key, w, wd, Iw, K, False)
  np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
  assert np.abs(want).max() > 1e-3

# ---------------------------------------------------------------- the kernels on NKISHIM

SHIM_MODE = {"vec": "replace", "vecadd": "add", "scalar": "replace"}

def _case(rng, N, D, E, Iw, K, bf16):
  from fusezap.serve.weights import f32_to_bf16, bf16_to_f32
  h = rng.standard_normal((N, D)).astype(np.float32)
  p = rng.random((N, E)).astype(np.float32)
  sel, key = _keys_np(p, K)
  w = np.where(sel, p, 0).astype(np.float32)
  wgu = rng.standard_normal((D, 2 * E * Iw), dtype=np.float32); wgu *= 0.03
  wd = rng.standard_normal((E * Iw, D), dtype=np.float32); wd *= 0.03
  if bf16:   # in row chunks (the real shapes' 2048 x 131072 weight: f32_to_bf16's temporaries are ~4x the array)
    def bf(x):
      b = np.empty(x.shape, np.uint16)
      for i in range(0, x.shape[0], 128):
        b[i:i + 128] = f32_to_bf16(x[i:i + 128]); x[i:i + 128] = bf16_to_f32(b[i:i + 128])
      return b
    s_gu, s_d = bf(wgu), bf(wd)
  else: s_gu, s_d = wgu, wd
  return h, sel, key, w, wgu, wd, s_gu, s_d

def _dense(N, D, E, Iw, h, sel, w, s_gu, s_d, bf16):
  """The dense path's kernels: gate_up over every expert, silu * up, a * (w widened), down over every expert."""
  from fusezap.device import nkishim
  from fusezap.codegen.nki_bf16 import MatmulBF16
  from fusezap.codegen.nki_kernels import render_matmul
  mm = (lambda M, Nn, Kk: MatmulBF16(M, Nn, Kk).render()) if bf16 else \
       (lambda M, Nn, Kk: render_matmul((1, M, Nn, Kk, 0, Kk, 1, 0, Nn, 1, 0, Nn, 1, 0, 0)))
  gu = np.zeros(N * 2 * E * Iw, np.float32)
  nkishim.execute(mm(N, 2 * E * Iw, D), [gu, h.reshape(-1), s_gu.reshape(-1)])
  gu = gu.reshape(N, 2 * E * Iw)
  a = (gu[:, :E * Iw] * (1 / (1 + np.exp(-gu[:, :E * Iw])))).astype(np.float32) * gu[:, E * Iw:]
  aw = (a.reshape(N, E, Iw) * w[:, :, None]).astype(np.float32).reshape(N, E * Iw)
  y = np.zeros(N * D, np.float32)
  nkishim.execute(mm(N, D, E * Iw), [y, aw.reshape(-1), s_d.reshape(-1)])
  return gu, a, y.reshape(N, D)

def _sparse(N, D, E, Iw, K, var, h, key, w, s_gu, s_d, a_full, bf16):
  from fusezap.device import nkishim
  from fusezap.codegen.nki_moe import MoeGateUpSparse, MoeDownSparse, picks_ref
  G = np.zeros(N * K * 2 * Iw, np.float32)
  nkishim.execute(MoeGateUpSparse(N, D, E, Iw, K, bf16, var).render(), [G, h.reshape(-1), key.reshape(-1), s_gu.reshape(-1)])
  ex = picks_ref(key, K)
  A = np.stack([a_full.reshape(N, E, Iw)[n, ex[n]] for n in range(N)])   # the dense path's silu * up of the chosen experts
  Y = np.zeros(N * D, np.float32)
  nkishim.execute(MoeDownSparse(N, D, E, Iw, K, bf16, var).render(), [Y, A.reshape(-1), key.reshape(-1), w.reshape(-1), s_d.reshape(-1)])
  return G.reshape(N, K, 2 * Iw), A, Y.reshape(N, D), ex

def _check(N, D, E, Iw, K, var, bf16, monkeypatch, seed=0):
  from fusezap.codegen.nki_moe import gate_up_ref, down_ref
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  monkeypatch.setenv("FZ_NKISHIM_IDMA", SHIM_MODE[var])
  rng = np.random.default_rng(seed)
  h, sel, key, w, wgu, wd, s_gu, s_d = _case(rng, N, D, E, Iw, K, bf16)
  gu, a, y = _dense(N, D, E, Iw, h, sel, w, s_gu, s_d, bf16)
  G, A, Y, ex = _sparse(N, D, E, Iw, K, var, h, key, w, s_gu, s_d, a, bf16)
  # the sparse gate / up are the dense kernel's columns of the chosen experts, bit for bit
  for n in range(N):
    for j, e in enumerate(ex[n]):
      want = np.concatenate([gu[n, e * Iw:(e + 1) * Iw], gu[n, E * Iw + e * Iw:E * Iw + (e + 1) * Iw]])
      assert np.array_equal(G[n, j].view(np.int32), want.view(np.int32)), (n, j, e)
  np.testing.assert_allclose(G, gate_up_ref(h, key, wgu, E, Iw, K, bf16), rtol=1e-5, atol=1e-5)
  np.testing.assert_allclose(Y, down_ref(A, key, w, wd, Iw, K, bf16), rtol=1e-5, atol=1e-5)
  if Iw % 128 == 0: assert np.array_equal(Y.view(np.int32), y.view(np.int32)), np.abs(Y - y).max()   # the same PSUM sums
  else: np.testing.assert_allclose(Y, y, rtol=1e-5, atol=1e-6)
  return ex

@pytest.mark.parametrize("var", ["vec", "vecadd", "scalar"])
@pytest.mark.parametrize("bf16", [True, False])
def test_kernels_equal_dense_small(var, bf16, monkeypatch):
  for N in (1, 3):
    _check(N, 256, 16, 128, 4, var, bf16, monkeypatch, seed=N)
  _check(2, 200, 8, 48, 3, var, bf16, monkeypatch, seed=7)   # D and Iw not multiples of 128 (within rounding of dense)

@pytest.mark.parametrize("N,Iw", [(1, 1024), (4, 512)])
def test_kernels_equal_dense_real_shapes(N, Iw, monkeypatch):
  """OLMoE's layer: D 2048, 64 experts of 1024 (one core) or the sliced 512 (TP=2), top 8, bf16, the default variant."""
  ex = _check(N, 2048, 64, Iw, 8, "vec", True, monkeypatch, seed=11)
  assert (np.diff(ex, axis=-1) > 0).all()

def test_wrong_shim_model_is_detected(monkeypatch):
  """The variants are told apart: the vec kernel under the 'add' model of vector_offset gives other numbers (an error), so
  the chip self-test's comparison is what picks the address form."""
  monkeypatch.setenv("FZ_NKISHIM_IDMA", "add")
  from fusezap.device import nkishim
  from fusezap.codegen.nki_moe import MoeGateUpSparse, gate_up_ref
  rng = np.random.default_rng(2)
  h, sel, key, w, wgu, wd, s_gu, s_d = _case(rng, 2, 256, 16, 128, 4, True)
  G = np.zeros(2 * 4 * 256, np.float32)
  try:
    nkishim.execute(MoeGateUpSparse(2, 256, 16, 128, 4, True, "vec").render(), [G, h.reshape(-1), key.reshape(-1), s_gu.reshape(-1)])
  except IndexError: return   # out of bounds: also detected
  assert not np.allclose(G.reshape(2, 4, -1), gate_up_ref(h, key, wgu, 16, 128, 4, True), atol=1e-4)

# ---------------------------------------------------------------- the model on NKISHIM

TINY128 = {"intermediate_size": 128, "num_experts": 16, "num_experts_per_tok": 4}   # expert width 128: the bitwise case

def _load(path, dt, sparse: bool, monkeypatch, **kw):
  from fusezap.serve.models import load_model
  if sparse: monkeypatch.setenv("FZ_MOE_SPARSE", "1")
  else: monkeypatch.setenv("FZ_MOE_SPARSE", "0")
  if dt == "bfloat16": monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  else: monkeypatch.delenv("FZ_NEURON_DTYPE", raising=False)
  m = load_model(path, dtype=getattr(dtypes, dt), device="NKISHIM", **kw)
  assert getattr(m, "moe_sparse", False) == sparse
  return m

def _names(m) -> list[str]:
  out = []
  for jf in m._neuron_steps.jits.values():
    for it in (jf.captured.items if jf.captured is not None else []):
      sp = getattr(it.si.ast[0], "spec", None) if it.si.ast else None
      if sp is not None: out += [q[0] for q in sp.parts] if hasattr(sp, "parts") else [sp.name]
  return out

@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_model_decode_logits_bitwise(tmp_path, dt, monkeypatch):
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from test_olmoe import _olmoe_dir
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  p = _olmoe_dir(tmp_path / "m", seed=7, **TINY128)
  ids = [int(i) for i in np.random.default_rng(5).integers(3, 512, 24)]
  toks = [int(t) for t in np.random.default_rng(6).integers(3, 512, 5)]
  out = {}
  for sp in (False, True):
    m = _load(p, dt, sp, monkeypatch, max_len=64, prefill_chunk=32)
    m.reset()
    lg = [m.prefill([ids])] + [m.decode([[t]]) for t in toks]
    out[sp] = (lg, _names(m))
  for a, b in zip(out[False][0], out[True][0]): assert np.array_equal(a.view(np.int32), b.view(np.int32))
  dn, sn = out[False][1], out[True][1]
  assert not any("fz_moe_" in n for n in dn)
  assert sum(n.startswith("fz_moe_gu_scalar_1_") for n in sn) == 2 and sum(n.startswith("fz_moe_down_scalar_1_") for n in sn) == 2   # 2 layers, decode
  assert not any(n.startswith(("fz_moe_gu_scalar_32_", "fz_moe_down_scalar_32_")) for n in sn)   # the 32-row prefill stays dense

def test_engine_tokens_sparse_equal_dense(tmp_path, monkeypatch):
  """LLMEngine (decode buckets 1, 2, 4) on NKISHIM in bf16 at OLMoE's head layout (D 2048, 16 / 16 heads of 128), 16 experts
  of 128, top 4: the sparse decode's greedy tokens == the dense model's."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from test_olmoe import _olmoe_dir, HEADS, PROMPTS
  from fusezap.serve.engine import LLMEngine, SamplingParams
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  p = _olmoe_dir(tmp_path / "e", seed=8, num_hidden_layers=1, **TINY128, **HEADS)
  from fusezap.serve.models.llama import Llama
  calls: list[int] = []
  orig = Llama._moe_sparse
  monkeypatch.setattr(Llama, "_moe_sparse", lambda self, h, l: calls.append(int(h.shape[0])) or orig(self, h, l))
  got = {}
  for sp in (False, True):
    m = _load(p, "bfloat16", sp, monkeypatch, max_len=128, prefill_chunk=32)
    eng = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
    got[sp] = eng.generate(PROMPTS, SamplingParams(max_tokens=8, ignore_eos=True)) + eng.generate(PROMPTS[1:2], SamplingParams(max_tokens=6, ignore_eos=True))
  assert got[True] == got[False] and len({t for r in got[False] for t in r}) > 3
  assert set(calls) == {1, 4}, calls   # traced once per decode bucket used (4 rows, then 1), never for a prefill

# ---------------------------------------------------------------- the sliced TP layout (two processes)

def test_tp2_sliced_sparse(tmp_path, monkeypatch):
  """FZ_MOE_SPARSE=1 under TP=2 (two NKISHIM ranks): the sliced layout (each rank every expert's half of the width; one
  all-reduce per layer); greedy tokens == one core's dense, decode logits within float32 rounding of it."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from test_olmoe import _olmoe_dir
  from fusezap.serve import dist
  from fusezap.serve.models import load_model
  p = _olmoe_dir(tmp_path / "tp", seed=9, intermediate_size=256, num_experts=16, num_experts_per_tok=4)
  for k in ("FZ_NEURON_TP_AR", "FZ_NEURON_TP_HEAD", "FZ_NEURON_DTYPE", "FZ_MOE_TP"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_MOE_SPARSE", "0")
  ref = load_model(p, dtype=dtypes.float32, device="NKISHIM", max_len=128, prefill_chunk=32)
  monkeypatch.setenv("FZ_MOE_SPARSE", "1")
  g = dist.launch(2, "NKISHIM", timeout=120)
  try:
    g.post("load", {"model": p, "dtype": "float32", "max_len": 128, "prefill_chunk": 32})
    m = load_model(p, dtype=dtypes.float32, device=g.device, max_len=128, prefill_chunk=32, tp=g)
    g.barrier("load")
    assert m.ep and m.moe_slice and m.moe_sparse and tuple(m.layers[0].wgu.shape) == (128, 2 * 16 * 128)
    ids = [int(i) for i in np.random.default_rng(3).integers(3, 512, 30)]
    toks = [int(t) for t in np.random.default_rng(4).integers(3, 512, 5)]
    for mm in (ref, m): mm.reset()
    la = [ref.prefill([ids])] + [ref.decode([[t]]) for t in toks]
    lb = [m.prefill([ids])] + [m.decode([[t]]) for t in toks]
    for a, b in zip(la, lb):
      np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4)
      assert (a.argmax(-1) == b.argmax(-1)).all()
    names = _names(m)
    assert any(n.startswith("fz_moe_down_scalar_1_") for n in names)
    ra = list(ref.generate(ids[:16], 12, stop=[]))
    assert list(m.generate(ids[:16], 12, stop=[])) == ra
  finally: g.close()

# ---------------------------------------------------------------- the chip job

def test_sparse_job_script():
  fn = os.path.join(ROOT, "scripts", "nki_olmoe_sparse_inf2.sh")
  src = open(fn).read()
  assert subprocess.run(["bash", "-n", fn]).returncode == 0
  m = re.search(r'scripts/aws_remote\.sh job --type inf2\.8xlarge --disk (\d+) --max-hours ([\d.]+) --max-usd ([\d.]+) -- "bash scripts/nki_olmoe_sparse_inf2\.sh"', src)
  assert m, "launch command"
  hours, usd = float(m.group(2)), float(m.group(3))
  assert (hours + 10 / 60) * 1.996 <= usd + 1e-9 and usd <= 3.6 and hours <= 1.6, (hours, usd)
  assert "REV=b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e" in src and "--tp 2" in src
  for k in ("olmoe_sparse_selftest.py", "FZ_MOE_SPARSE=1", "fz_dense", "fz_sparse", "olmoe_sparse_compare.py", "fzhttp_sparse", "--decode-top1"):
    assert k in src, k
  assert "hf_" not in src.replace("hf_hub_download", "")
