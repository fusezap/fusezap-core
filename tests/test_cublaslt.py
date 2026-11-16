"""Prefill GEMMs on cuBLASLt (fusezap/device/cuda_kernels/cublaslt.py, speed run 4 round 12): a pinned algo per shape,
self-checked for bit identity across every registered M; linear() uses it only for registered shapes and M (prefill
sizes) and FuseZap's kernel otherwise; the values match a float reference; rows are bit-identical whatever the bucket
and the other rows. CUDA only (skips elsewhere)."""
import numpy as np
import pytest
from fusezap import Tensor, dtypes
from fusezap.device import Device

HAVE_CUDA = Device.available("CUDA")
gpu = pytest.mark.skipif(not HAVE_CUDA, reason="needs an NVIDIA GPU + cuda-bindings")

def test_plan_for_without_enable_is_none():
  from fusezap.device.cuda_kernels import cublaslt
  assert cublaslt.plan_for("NOPE", 256, 64, 64, "bf16", "bf16") is None

@gpu
@pytest.mark.parametrize("out", ["bf16", "f32"])
def test_cublaslt_prefill_linear(out, monkeypatch):
  from fusezap.device.cuda_kernels import cublaslt
  from fusezap.serve import cuda as C
  if not cublaslt.available(): pytest.skip("no libcublasLt")
  N, K, Ms = 384, 512, [256, 512, 1024]
  plans = cublaslt.enable("CUDA", [(N, K, "bf16", out)], Ms)
  pl = plans[(N, K, "bf16", out)]
  assert pl is not None and set(Ms) <= pl.Ms and pl.algo is not None
  rng = np.random.default_rng(0)
  X = rng.standard_normal((1024, K)).astype(np.float32)
  W = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
  w = Tensor(W, device="CUDA", dtype=dtypes.bfloat16).realize()
  odt = dtypes.float32 if out == "f32" else None
  def lin(rows):
    x = Tensor(X[rows], device="CUDA", dtype=dtypes.bfloat16).realize()
    return C.linear(x, w, None, odt).float().numpy()
  big = lin(np.arange(1024))
  ref = X.astype(np.float32) @ W.T
  np.testing.assert_allclose(big, ref, rtol=3e-2, atol=3e-2)
  perm = rng.permutation(1024)
  for M in Ms:   # the same rows at another position and in another bucket: the same bits
    rows = perm[:M]
    assert np.array_equal(lin(rows).view(np.uint32), big[rows].view(np.uint32)), M
  # an unregistered M (300) runs FuseZap's kernel: close, not necessarily the same bits
  assert cublaslt.plan_for("CUDA", 300, N, K, "bf16", out) is None
  np.testing.assert_allclose(lin(np.arange(300)), ref[:300], rtol=3e-2, atol=3e-2)
  # FZ_CUBLASLT=0 turns it off
  monkeypatch.setenv("FZ_CUBLASLT", "0")
  from fusezap import helpers
  helpers.getenv.cache_clear()
  assert not cublaslt.available()
  monkeypatch.delenv("FZ_CUBLASLT"); helpers.getenv.cache_clear()

@gpu
def test_cublaslt_decode_group(monkeypatch):
  """FZ_CUBLASLT_DECODE's plans: their own algo for the decode buckets 1 .. 32, rows bit-identical across buckets and
  positions; the prefill group's plan of the same shape is untouched."""
  from fusezap.device.cuda_kernels import cublaslt
  from fusezap.serve import cuda as C
  if not cublaslt.available(): pytest.skip("no libcublasLt")
  monkeypatch.setenv("FZ_CUBLASLT_DECODE", "o")
  N, K, Ms = 640, 512, [1, 2, 4, 8, 16, 32]
  pl = cublaslt.enable("CUDA", [(N, K, "bf16", "bf16")], Ms, group="decode")[(N, K, "bf16", "bf16")]
  if pl is None: pytest.skip("no cuBLASLt algo passed the bit-identity check over the decode buckets on this GPU")
  assert set(Ms) <= pl.Ms and cublaslt.plan_for("CUDA", 8, N, K, "bf16", "bf16") is pl and cublaslt.plan_for("CUDA", 3, N, K, "bf16", "bf16") is None
  rng = np.random.default_rng(1)
  X = rng.standard_normal((32, K)).astype(np.float32)
  w = Tensor((rng.standard_normal((N, K)) * 0.05).astype(np.float32), device="CUDA", dtype=dtypes.bfloat16).realize()
  lin = lambda rows: C.linear(Tensor(X[rows], device="CUDA", dtype=dtypes.bfloat16).realize(), w).float().numpy()
  big = lin(np.arange(32))
  for M in Ms:
    rows = rng.permutation(32)[:M]
    assert np.array_equal(lin(rows).view(np.uint32), big[rows].view(np.uint32)), M

@gpu
def test_cublaslt_check_speed_drops_or_keeps():
  """W15 FZ_CUBLASLT_CHECK: check_speed times the pinned plan and FuseZap's kernel at the smallest and largest M and
  plan_for follows the decision (None = FuseZap's kernel at every M)."""
  from fusezap.device.cuda_kernels import cublaslt
  if not cublaslt.available(): pytest.skip("no libcublasLt")
  N, K, Ms = 1536, 1024, [128, 256, 2048]
  cublaslt.enable("CUDA", [(N, K, "bf16", "bf16")], Ms)
  res = cublaslt.check_speed("CUDA", Ms)
  t_lt, t_fz, kept = res[(N, K, "bf16", "bf16")]
  assert set(t_lt) == set(t_fz) == {128, 2048} and kept == cublaslt.keep_rule(t_lt, t_fz)
  assert (cublaslt.plan_for("CUDA", 256, N, K, "bf16", "bf16") is not None) == kept

def test_per_m_algo_pick_with_fakes(monkeypatch):
  """W15 FZ_CUBLASLT_PER_M: below the largest M each size takes cuBLASLt's first own algo that runs and passes the bit
  check against the pinned algo's rows; a size whose top pick is the pinned algo, or where no candidate passes, keeps
  the pinned one. launch / _supported use plan.algo_for(M)."""
  import ctypes
  from fusezap.device.cuda_kernels import cublaslt as L
  mk = lambda v: (ctypes.c_uint64 * 8)(*([v] + [0] * 7))
  pinned, bad, good, other = mk(1), mk(2), mk(3), mk(4)
  plan = L.Plan(64, 32, "bf16", "bf16", pinned, {}, None, None, frozenset({128, 256, 512, 2048}))
  heur = {128: [bad, good], 256: [pinned, other], 512: [bad]}
  monkeypatch.setattr(L, "_heuristic", lambda M, *a, **k: heur[M])
  monkeypatch.setattr(L, "_LIB", [None, None, None])
  monkeypatch.setattr(L, "_supported", lambda lib, h, p, M: True)
  seen = []
  def chk(Ms):
    seen.append((Ms[0], bytes(plan.algo_for(Ms[0]))[0]))
    return None if bytes(plan.algo_for(Ms[0])) != bytes(bad) else "differs"
  L._per_m(plan, [128, 256, 512], chk)
  assert bytes(plan.algos[128]) == bytes(good) and 256 not in plan.algos and 512 not in plan.algos
  assert bytes(plan.algo_for(256)) == bytes(pinned) and bytes(plan.algo_for(2048)) == bytes(pinned)
  assert seen == [(128, 2), (128, 3), (512, 2)]
  assert L.keep_rule({128: 207.8, 2048: 1072.9}, {128: 126.4, 2048: 1159.3})   # pod A2's tie: kept within the margin

@gpu
def test_cublaslt_per_m_rows_bit_identical(monkeypatch):
  """FZ_CUBLASLT_PER_M=1 on the GPU: whatever own algos the small sizes got, every M's rows equal the largest M's."""
  from fusezap.device.cuda_kernels import cublaslt
  from fusezap.serve import cuda as C
  from fusezap import helpers
  if not cublaslt.available(): pytest.skip("no libcublasLt")
  monkeypatch.setenv("FZ_CUBLASLT_PER_M", "1"); helpers.getenv.cache_clear()
  try:
    N, K, Ms = 2560, 1024, [128, 256, 2048]
    pl = cublaslt.enable("CUDA", [(N, K, "bf16", "bf16")], Ms)[(N, K, "bf16", "bf16")]
    assert pl is not None
    print(f"per-M own algos at M {sorted(pl.algos)}")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((2048, K)).astype(np.float32)
    w = Tensor((rng.standard_normal((N, K)) * 0.05).astype(np.float32), device="CUDA", dtype=dtypes.bfloat16).realize()
    lin = lambda rows: C.linear(Tensor(X[rows], device="CUDA", dtype=dtypes.bfloat16).realize(), w).float().numpy()
    big = lin(np.arange(2048))
    perm = rng.permutation(2048)
    for M in Ms: assert np.array_equal(lin(perm[:M]).view(np.uint32), big[perm[:M]].view(np.uint32)), M
  finally: monkeypatch.delenv("FZ_CUBLASLT_PER_M"); helpers.getenv.cache_clear()
