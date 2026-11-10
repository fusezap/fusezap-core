"""Cross-package seams found during Run A Phase 1 integration (A-INT). Each test runs on every tier."""
from __future__ import annotations
import numpy as np
import pytest
import torch
from fusezap import Tensor, dtypes
from fusezap.device import Buffer, Device
from fusezap.lazy import LazyBuffer
from fusezap.ops import BinaryOps, ReduceOps
from fusezap.realize import run_schedule
from fusezap.schedule import create_schedule
from fusezap.shape.symbolic import Var

def _dev(tier: str) -> str: return Device.DEFAULT if tier == "interp" else tier

def test_symbolic_dim_may_be_zero_end_to_end(device):
  # WP1 relaxed DivNode/ModNode to allow a Node divisor with vmin 0 (index math uses s+(s<1)); the
  # LtNode inside a divisor must lower and run on every backend, including the p=0 (empty) case.
  a = np.arange(32, dtype=np.float32).reshape(4, 8)
  buf = Buffer(_dev(device), 32, dtypes.float32).copyin(memoryview(a).cast("B"))
  lb = LazyBuffer.from_buffer(buf, (4, 8))
  p = Var("p", 0, 8)
  for v in (0, 1, 5, 8):
    s = lb.shrink(((0, 4), (0, p))).permute((1, 0))         # KV-cache style: (p,4) transposed window
    e = s.e(BinaryOps.ADD, s).contiguous().reshape((p * 4,)).contiguous()
    r = lb.shrink(((0, 4), (0, p))).r(ReduceOps.SUM, (1,)).contiguous()
    run_schedule(create_schedule([e, r], var_vals={p: v}))
    np.testing.assert_array_equal(e.base.realized.numpy()[:v * 4], (a[:, :v].T * 2).reshape(-1))
    np.testing.assert_array_equal(r.base.realized.numpy()[:4], a[:, :v].sum(1))

def test_int32_add_wraps(device):
  # WP2's min/argmin rely on two's-complement wrap of NEG/ADD at the dtype limit; CPU compiles with -fwrapv
  a = np.array([2**31 - 1, -2**31, 5], np.int32)
  x = Tensor(a)
  np.testing.assert_array_equal((x + 1).numpy(), (torch.tensor(a) + 1).numpy())
  np.testing.assert_array_equal((-x).numpy(), (-torch.tensor(a)).numpy())
  np.testing.assert_array_equal(x.min().numpy(), np.int32(-2**31))
  assert int(x.argmin().numpy()) == 1

def test_bitcast_byte_to_bool_normalizes(device):
  # a byte other than 0/1 reinterpreted as bool is "true" on every tier (CPU used __builtin_bit_cast -> invalid _Bool)
  a = np.array([0, 1, 2, 255], np.uint8)
  got = Tensor(a).bitcast(dtypes.bool).cast(dtypes.int32).numpy()
  np.testing.assert_array_equal(got, [0, 1, 1, 1])

def test_bf16_numpy_is_uint16_bits(device):
  x = Tensor(np.array([1.0, -2.5, 3.0], np.float32)).cast(dtypes.bfloat16)
  out = x.numpy()
  assert out.dtype == np.uint16
  np.testing.assert_array_equal(out, torch.tensor([1.0, -2.5, 3.0]).bfloat16().view(torch.int16).numpy().view(np.uint16))

@pytest.mark.skipif(not Device.available("CPU"), reason="CPU backend (clang) not available")
def test_cpu_disk_cache_key_includes_flags():
  from fusezap.device.cpu import ClangCompiler
  c = ClangCompiler()
  assert "-fwrapv" in c.args and "-fwrapv" in c.cache_key()
  c2 = ClangCompiler(cc="/nonexistent/clang")
  assert c.cache_key() != c2.cache_key()
