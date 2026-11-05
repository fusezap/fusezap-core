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
