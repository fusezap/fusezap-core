"""I95 fixes for FLUX.2 [klein] on CUDA at 1024^2.

- 64-bit index math (fusezap/codegen/lower.py): the VAE's last upsampler conv (256 ch at 1024^2) materializes a
  2.4e9-element im2col on CUDA (the matmul hook takes it as an operand) and its store index gidx2 * 9437184 overflowed
  int32 -> CUDA_ERROR_ILLEGAL_ADDRESS. Rendered for C / CUDA / Metal: int64 only where a value can leave int32, every
  other kernel unchanged. A CPU run over a > 2**31-element buffer is the slow test.
- The CUDA SIMT attention hook declines a match whose shared memory exceeds the device's opt-in limit (the VAE's head
  dim 512 on an RTX 4090: 139392 > 101376 bytes), leaving the sdpa composed.
- FZ_CUDA_SYNC=1 names the faulting kernel.
"""
from __future__ import annotations
import types
import numpy as np
import pytest
from fusezap import Tensor, dtypes, schedule
from fusezap.codegen import lower as L
from fusezap.codegen.cstyle import CRenderer
from fusezap.codegen.cuda import CUDARenderer
from fusezap.codegen.metal import MetalRenderer
from fusezap.device import Device
from fusezap.ir import LazyOp, MemBuffer
from fusezap.ops import BufferOps, BinaryOps, UnaryOps
from fusezap.shape.view import ShapeTracker

def _ew(shape, st1=None):
  ld = LazyOp(BufferOps.LOAD, (), MemBuffer(1, dtypes.float32, st1 or ShapeTracker.from_shape(shape)))
  return (LazyOp(BufferOps.STORE, (LazyOp(UnaryOps.NEG, (ld,)),), MemBuffer(0, dtypes.float32, ShapeTracker.from_shape(shape))),)

WIDE = {CRenderer: "long long", CUDARenderer: "long long", MetalRenderer: "long "}

@pytest.mark.parametrize("R", list(WIDE))
def test_large_buffer_index_is_64bit(R):
  src = L.lower(_ew((256, 9, 1048576)), R()).src   # the 1024^2 im2col's shape: 2.4e9 elements
  assert WIDE[R] in src, src

@pytest.mark.parametrize("R", list(WIDE))
def test_int32_kernels_unchanged(R):
  for shape in ((256, 1024, 1024), (4096, 2304), (7, 5)):   # under 2**31 elements: no 64-bit math at all
    src = L.lower(_ew(shape), R()).src
    assert "long" not in src, (shape, src)

def _emulated_cuda_kernels(monkeypatch, accept=None):
  from fusezap.device import cuda_kernels as ck
  from fusezap.device.cuda_kernels import attention as at
  ks = {"matmul": ck._hook_kernel(), **at.custom_kernels()}
  if accept is not None: ks["attention"].accept = accept
  monkeypatch.setattr(schedule, "_device_kernels", lambda device: ks)
  return ks

def test_vae_upsampler_conv_im2col_store_is_64bit(monkeypatch):
  """The kernel that faulted on the H200: scheduled with CUDA's matmul hook (emulated on CPU, nothing runs)."""
  _emulated_cuda_kernels(monkeypatch)
  x = Tensor.empty(1, 256, 512, 512, device="CPU")
  y = x.interpolate(scale_factor=2).conv2d(Tensor.empty(256, 256, 3, 3, device="CPU"), Tensor.empty(256, device="CPU"), padding=1)
  big = [si for si in schedule.create_schedule([y.lazydata]) if si.kind == "kernel" and si.bufs[0].size > 2 ** 31]
  assert len(big) == 1 and big[0].bufs[0].size == 256 * 9 * 1024 * 1024
  src = L.lower(big[0].ast, CUDARenderer()).src
  import re
  idx = re.findall(r"data0\[(\w+)\] =", src)
  assert idx and all(f"long long {i} =" in src for i in idx), src

def _sdpa_custom_names(D: int) -> list[str]:
  q, k, v = (Tensor.empty(1, 1, 64, D, device="CPU") for _ in range(3))
  o = q.scaled_dot_product_attention(k, v)
  return [si.ast[0].kernel.name for si in schedule.create_schedule([o.lazydata]) if si.kind == "custom"]

def test_attention_hook_declines_when_it_does_not_fit(monkeypatch):
  from fusezap.device.cuda_kernels import attention as at
  dev = types.SimpleNamespace(_fz_max_smem=101376)   # RTX 4090 (sm_89) opt-in limit
  assert at.hook_fits(dev, (1, 64, 64, 128, 128)) and not at.hook_fits(dev, (1, 64, 64, 512, 512))
  assert at.hook_fits(types.SimpleNamespace(_fz_max_smem=232448), (1, 64, 64, 512, 512))   # H100 / H200
  _emulated_cuda_kernels(monkeypatch, accept=lambda p: at.hook_fits(dev, p))
  assert "fz_attention" in _sdpa_custom_names(128)
  names = _sdpa_custom_names(512)
  assert "fz_attention" not in names and "fz_matmul_tf32x3" in names   # composed: QK^T and PV still take the GEMM hook

def test_cuda_sync_names_the_faulting_kernel(monkeypatch):
  from fusezap import helpers
  from fusezap.device import cuda as C
  calls = []
  cu = types.SimpleNamespace(cuStreamSynchronize=lambda s: calls.append(s) or (700,))
  monkeypatch.setattr(C, "_check", lambda ret, what="": (_ for _ in ()).throw(RuntimeError(what)) if ret[0] else None)
  dev = types.SimpleNamespace(cu=cu, stream=5, capturing=False)
  helpers.getenv.cache_clear(); monkeypatch.delenv("FZ_CUDA_SYNC", raising=False)
  C.debug_sync(dev, "E_256_9_1048576"); assert calls == []
  monkeypatch.setenv("FZ_CUDA_SYNC", "1"); helpers.getenv.cache_clear()
  try:
    with pytest.raises(RuntimeError, match="E_256_9_1048576"): C.debug_sync(dev, "E_256_9_1048576", (4096, 9, 256), (256, 1, 1))
    dev.capturing = True; C.debug_sync(dev, "x")   # never inside a stream capture
    assert calls == [5]
  finally: helpers.getenv.cache_clear()

@pytest.mark.slow
@pytest.mark.skipif(not Device.available("CPU"), reason="CPU backend (clang) not available")
def test_cpu_kernel_over_2g_elements():
  """~4.3 GB: a uint8 (2, 2**30 + 16) elementwise kernel; the flat index of row 1 passes 2**31."""
  n = 2 ** 30 + 16
  y = (Tensor.full((2, n), 7, dtype=dtypes.uint8, device="CPU") + Tensor.full((2, n), 1, dtype=dtypes.uint8, device="CPU")).realize()
  a = y.numpy()
  assert a.shape == (2, n) and a[1, -16:].tolist() == [8] * 16 and a[0, :4].tolist() == [8] * 4 and int(a[1, ::2 ** 20].min()) == 8
