"""NKISHIM: a numpy model of the nki 0.6 subset that fusezap.codegen.nki emits (Phase C, N1; docs/NKI_BACKEND.md §6).

Not a device tier (that is N2/N4): it runs generated NKI kernel source on the Mac so the renderer can be tested
against the op suite. `execute(prog, arrays)` runs a rendered NKIProgram on flat numpy buffers in place, and
`run_item(si)` does the same for a ScheduleItem whose buffers live on the NPY device.

The generated source keeps its real `import nki` lines; `load(src)` executes it with an import hook that resolves
`nki`, `nki.language` and `nki.isa` to this module's namespaces, so the same text runs under the real toolchain.

What is modelled (from the nki 0.6.0 API stubs and its CPU simulator's documented behaviour):
  - tensors: one flat storage array plus (shape, strides, offset) views; `t[...]` slices (an int index keeps the
    partition dim of an on-chip tensor and drops an HBM dim), `.ap(pattern, offset)`, same-size `.view(dtype)`.
    bfloat16 storage is uint16 bits; reads decode to float32 and writes round to nearest even.
  - engines compute in float32 and cast the result to dst.dtype; float -> int casts round to nearest even and
    saturate (NaN -> int max). `tensor_tensor` uses native integer math (GpSimd, wrapping) when it names
    `engine=nisa.gpsimd_engine`, or by itself for add/subtract/multiply with every tile int32/uint32 (as nki 0.6's
    isa layer dispatches; maximum, compares, divide and fmod otherwise run in float32). GpSimd divide truncates.
  - `tensor_copy_predicated` needs a tile src (the real 0.6 function reads src.shape; its docstring's scalar src fails).
  - `tensor_reduce` computes in float32 for every dtype (int sums are inexact past 2**24, as on the hardware).
  - the ScalarE `sin` is the hardware cubic fit without range reduction (callers must reduce into [-pi, pi]).
  - checks: on-chip partition dim <= 128 and <= 192 KiB per partition, compute engines never touch HBM, operand
    shapes match, out-of-bounds HBM/SBUF access raises (the simulator's oob_mode.error), and fresh tiles are
    filled with garbage (NaN / 0xA5 bytes) so a read of an unwritten element shows up in the results.
NKI-N3 adds the TensorE subset of the matmul and attention kernels (fusezap/codegen/nki_kernels.py), modelled on the
0.6 simulator's numpy code: `nc_matmul` (dst PSUM float32 = stationary.T @ moving in float32, accumulate=True adds to the
PSUM tile, False overwrites; operands in SBUF, K <= 128 partitions, stationary free <= 128, moving free <= 512 = one
PSUM bank), `nc_transpose` into PSUM (TensorE: an exact transpose in the simulator, <= 128 x 128), `iota` and
`affine_select` (an affine function of partition and free index, compared with 0).
Not modelled: timing, engine queues/semaphores, PSUM bank allocation, DMA queues, the reduce registers of `activation`.
FZ_NKISHIM_MM=f64 (N7 step 4, tests only): `nc_matmul` as exact float64 products summed in float64 and rounded to float32
once per instruction, so its bits do not depend on the output row's position or on which operand is stationary (BLAS's
float32 kernels do: remainder rows of a microkernel, gemv vs gemm at one row); TensorE's PE array is row-invariant. Used to
check kernels that move rows or swap operands (LayerSpec early / weight-stationary) against the members bit for bit.
"""
from __future__ import annotations
import builtins, math, os, time, types
from dataclasses import dataclass
from typing import Any
import numpy as np

PMAX, SBUF_PARTITION_BYTES, PSUM_PARTITION_BYTES = 128, 192 * 1024, 16 * 1024   # PSUM: 8 banks x 2 KiB

# ---------------------------------------------------------------- dtypes and buffers

@dataclass(frozen=True)
class DType:
  name: str
  storage: Any        # numpy storage dtype
  def __repr__(self): return f"nl.{self.name}"
  @property
  def is_float(self) -> bool: return self.name in ("float32", "float16", "bfloat16")
  @property
  def itemsize(self) -> int: return np.dtype(self.storage).itemsize

DTYPES = {n: DType(n, s) for n, s in [("float32", np.float32), ("float16", np.float16), ("bfloat16", np.uint16),
                                      ("int32", np.int32), ("int16", np.int16), ("int8", np.int8), ("uint32", np.uint32),
                                      ("uint16", np.uint16), ("uint8", np.uint8), ("bool_", np.uint8)]}

class MemoryRegion(str):
  def __repr__(self): return f"nl.{self}"
SBUF, PSUM, HBM, SHARED_HBM, PRIVATE_HBM = (MemoryRegion(n) for n in ("sbuf", "psum", "hbm", "shared_hbm", "private_hbm"))
ON_CHIP = (SBUF, PSUM)

def _bf16_decode(bits: np.ndarray) -> np.ndarray: return (bits.astype(np.uint32) << 16).view(np.float32)

def _bf16_encode(f: np.ndarray) -> np.ndarray:
  f = np.ascontiguousarray(f, np.float32)
  u = f.view(np.uint32)
  r = ((u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)).astype(np.uint16)
  return np.where(np.isnan(f), ((u >> np.uint32(16)) | np.uint32(0x40)).astype(np.uint16), r)

def from_f32(x: np.ndarray, dt: DType) -> np.ndarray:
  """An engine's float32 result -> storage values of `dt` (RNE + saturation for ints, like the hardware)."""
  x = np.asarray(x, np.float32)
  if dt.name == "bfloat16": return _bf16_encode(x)
  if dt.is_float: return x.astype(dt.storage)
  info = np.iinfo(dt.storage)
  with np.errstate(invalid="ignore"):
    r = np.clip(np.rint(x).astype(np.float64), float(info.min), float(info.max))
    return np.where(np.isnan(x), float(info.max), r).astype(dt.storage)

class NkiTensor:
  """A view (shape, strides, offset in elements) of a flat storage array on a memory region."""
  def __init__(self, storage: np.ndarray, dtype: DType, buffer: str, shape: tuple[int, ...], strides: tuple[int, ...] | None = None,
               offset: int = 0, name: str = ""):
    if strides is None:
      acc, st = 1, []
      for s in reversed(shape): st.append(acc); acc *= s
      strides = tuple(reversed(st))
    self.storage, self.dtype, self.buffer, self.name = storage, dtype, buffer, name
    self.shape, self.strides, self.offset = tuple(int(s) for s in shape), tuple(int(s) for s in strides), int(offset)
  def __repr__(self): return f"<NkiTensor {self.name or '?'} {self.buffer} {self.shape} {self.dtype}>"
  @property
  def ndim(self) -> int: return len(self.shape)
  @property
  def size(self) -> int: return int(np.prod(self.shape)) if self.shape else 1
  def get_pattern(self) -> list[list[int]]: return [[st, s] for st, s in zip(self.strides, self.shape)]

  def _index(self) -> np.ndarray:
    idx = np.full(self.shape, self.offset, np.int64)
    for k, (s, st) in enumerate(zip(self.shape, self.strides)):
      idx = idx + (np.arange(s, dtype=np.int64) * st).reshape([-1 if j == k else 1 for j in range(self.ndim)])
    if idx.size and (idx.min() < 0 or idx.max() >= self.storage.size):
      raise IndexError(f"NKISHIM: out-of-bounds access on {self!r}: index range [{idx.min()}, {idx.max()}] "
                       f"outside storage of {self.storage.size} elements (pattern {self.get_pattern()}, offset {self.offset})")
    return idx
  def raw(self) -> np.ndarray: return self.storage[self._index()]
  def values(self) -> np.ndarray:
    """Element values: float32 for bfloat16, the storage dtype otherwise."""
    r = self.raw()
    return _bf16_decode(r) if self.dtype.name == "bfloat16" else r
  def f32(self) -> np.ndarray: return self.values().astype(np.float32, copy=False)
  def write_raw(self, r: np.ndarray) -> None:
    r = np.asarray(r)
    assert r.size == self.size, f"NKISHIM: writing {r.size} elements into {self!r}"
    self.storage[self._index()] = r.reshape(self.shape).astype(self.storage.dtype, copy=False)

  def __getitem__(self, key) -> "NkiTensor":
    key = key if isinstance(key, tuple) else (key,)
    if any(k is Ellipsis for k in key):
      i = key.index(Ellipsis)
      key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
    assert len(key) <= self.ndim, f"NKISHIM: too many indices {key} for {self!r}"
    key = key + (slice(None),) * (self.ndim - len(key))
    shape, strides, off = [], [], self.offset
    for d, (k, s, st) in enumerate(zip(key, self.shape, self.strides)):
      if isinstance(k, (int, np.integer)):
        k = int(k) + (s if k < 0 else 0)
        assert 0 <= k < s, f"NKISHIM: index {k} out of range for dim {d} of {self!r}"
        off += k * st
        if d == 0 and self.buffer in ON_CHIP: shape.append(1); strides.append(st)   # on-chip: the partition dim stays
        continue
      assert isinstance(k, slice), f"NKISHIM: unsupported index {k!r}"
      start, stop, step = k.indices(s)
      assert step >= 1 and 0 <= start < stop <= s, f"NKISHIM: invalid slice {k} for dim {d} (size {s}) of {self!r}"
      shape.append(len(range(start, stop, step))); strides.append(st * step); off += start * st
    return NkiTensor(self.storage, self.dtype, self.buffer, tuple(shape), tuple(strides), off, self.name)

  def ap(self, pattern: list[list[int]], offset: int | None = None, dtype: DType | None = None, **kw) -> "NkiTensor":
    assert not kw, f"NKISHIM: ap() options {sorted(kw)} are not modelled"
    assert dtype is None or dtype == self.dtype, "NKISHIM: ap(dtype=) reinterpretation is not modelled"
    assert all(int(n) >= 1 for _, n in pattern), f"NKISHIM: ap() counts must be >= 1: {pattern}"
    if self.buffer in ON_CHIP and len(pattern) > 1 and len(self.shape) > 1:
      free = int(np.prod(self.shape[1:]))
      assert pattern[0][0] == free, f"NKISHIM: on-chip ap() partition step {pattern[0][0]} must equal the free size {free}"
    return NkiTensor(self.storage, self.dtype, self.buffer, tuple(n for _, n in pattern), tuple(s for s, _ in pattern),
                     self.offset + (offset or 0), self.name)

  def view(self, dtype: DType) -> "NkiTensor":
    assert dtype.itemsize == self.dtype.itemsize, "NKISHIM: only same-size view(dtype) is modelled"
    return NkiTensor(self.storage.view(dtype.storage), dtype, self.buffer, self.shape, self.strides, self.offset, self.name)

def tensor(arr: np.ndarray, dtype: DType, buffer: str = SHARED_HBM, name: str = "") -> NkiTensor:
  """Wrap a flat numpy storage array (no copy) as an HBM tensor of `dtype`."""
  arr = np.asarray(arr).reshape(-1)
  if arr.dtype != np.dtype(dtype.storage): arr = arr.view(dtype.storage)
  return NkiTensor(arr, dtype, buffer, (arr.size,), (1,), 0, name)

def ndarray(shape, dtype: DType, buffer: str = SBUF, name: str = "", address=None) -> NkiTensor:
  shape = tuple(int(s) for s in shape)
  assert all(s >= 1 for s in shape), f"NKISHIM: ndarray shape {shape} must be positive"
  if buffer in ON_CHIP:
    assert len(shape) >= 2, f"NKISHIM: on-chip tensors are at least 2D, got {shape}"
    assert shape[0] <= PMAX, f"NKISHIM: partition dim {shape[0]} exceeds {PMAX}"
    per = int(np.prod(shape[1:])) * dtype.itemsize
    assert per <= SBUF_PARTITION_BYTES, f"NKISHIM: {per} bytes per partition exceeds the {SBUF_PARTITION_BYTES} of a partition"
    assert buffer != PSUM or per <= PSUM_PARTITION_BYTES, f"NKISHIM: a {per}-byte PSUM tile exceeds the {PSUM_PARTITION_BYTES} of a partition"
  return NkiTensor(_garbage(int(np.prod(shape)), dtype), dtype, buffer, shape, None, 0, name)

def _garbage(n: int, dt: DType) -> np.ndarray:
  """Fresh on-chip memory: NaN for floats, 0xA5 bytes for ints, so reading an unwritten element is visible."""
  st = np.dtype(dt.storage)
  if dt.name == "bfloat16": return np.full(n, 0x7FC1, np.uint16)
  if st.kind == "f": return np.full(n, np.nan, st)
  return np.full(n * st.itemsize, 0xA5, np.uint8).view(st)

def zeros(shape, dtype: DType, buffer: str = SBUF, name: str = "") -> NkiTensor:
  t = ndarray(shape, dtype, buffer, name)
  t.storage[:] = 0
  return t

# ---------------------------------------------------------------- ops

class _Op:
  def __init__(self, name: str, fn=None): self.name, self.fn = name, fn
  def __repr__(self): return f"nl.{self.name}"

def _hw_sin(x: np.ndarray) -> np.ndarray:
  """ScalarE sin: exact on [0, pi] (odd), the cubic Taylor polynomial about pi outside it (no range reduction)."""
  x = np.asarray(x, np.float32)
  a, t = np.abs(x), np.abs(x) - np.float32(math.pi)
  with np.errstate(invalid="ignore", over="ignore"):
    r = np.where(a >= np.float32(math.pi), -t + np.float32(1 / 6) * t ** 3, np.sin(a)).astype(np.float32)
    r = np.where(np.signbit(x), -r, r)
    return np.where(np.isfinite(x), r, np.float32("nan")).astype(np.float32)

ARITH = {n: _Op(n, f) for n, f in [("add", np.add), ("subtract", np.subtract), ("multiply", np.multiply), ("divide", np.divide),
                                   ("maximum", np.maximum), ("minimum", np.minimum), ("fmod", np.fmod), ("mod", np.mod)]}
COMPARE = {n: _Op(n, f) for n, f in [("equal", np.equal), ("not_equal", np.not_equal), ("less", np.less), ("less_equal", np.less_equal),
                                     ("greater", np.greater), ("greater_equal", np.greater_equal)]}
ACTIVATION = {n: _Op(n, f) for n, f in [("exp", np.exp), ("log", np.log), ("sqrt", np.sqrt), ("sin", _hw_sin), ("copy", lambda x: x),
                                        ("reciprocal", lambda x: np.float32(1) / x), ("square", np.square), ("abs", np.abs),
                                        ("sign", np.sign), ("tanh", np.tanh)]}
BITVEC = {n: _Op(n, f) for n, f in [("bitwise_and", np.bitwise_and), ("bitwise_or", np.bitwise_or), ("bitwise_xor", np.bitwise_xor)]}
OPS = {**ARITH, **COMPARE, **ACTIVATION, **BITVEC}
REDUCE = {"add": np.sum, "maximum": np.max, "minimum": np.min, "multiply": np.prod}
INT_AUTO_GPSIMD = {"add", "subtract", "multiply"}   # nki 0.6 dispatches these to GpSimd by itself when every tile is int32/uint32
# What neuronx-cc 2.27 rejects for inf2 although the 0.6 simulator runs it (scripts/nki_op_matrix.py, NKI-N2). The shim
# refuses these too, so a kernel that passes NKISHIM also compiles.
NO_COMPILE_OPS = {"divide", "fmod", "mod"}           # every engine, tensor_tensor and tensor_scalar
GPSIMD_COMPILES = {"add", "subtract", "multiply", "bitwise_and", "bitwise_or", "bitwise_xor"}   # int32 only

def _compiles(op: "_Op", where: str) -> None:
  assert op.name not in NO_COMPILE_OPS, f"NKISHIM: {where} op {op}: neuronx-cc 2.27 rejects it for inf2 (the simulator would run it)"

class engine:
  unknown, vector, scalar, gpsimd, tensor, sync = "unknown", "vector", "scalar", "gpsimd", "tensor", "sync"

def _pstart(t: NkiTensor) -> int:
  """First partition of an on-chip view (tiles are allocated row-major: one row per partition)."""
  return t.offset // t.strides[0] if t.strides and t.strides[0] > 0 else 0

def _onchip(*ts: Any) -> None:
  for t in ts:
    if isinstance(t, NkiTensor):
      assert t.buffer in ON_CHIP, f"NKISHIM: compute engines cannot access {t.buffer} ({t!r})"
      assert t.shape and t.shape[0] <= PMAX, f"NKISHIM: partition dim of {t!r} exceeds {PMAX}"
      # neuronx-cc (inf2): compute instructions start on a partition quadrant; DMAs may start anywhere (NKI-N2)
      assert _pstart(t) % 32 == 0, f"NKISHIM: {t!r} starts at partition {_pstart(t)}; compute engines need a multiple of 32"

def _same(a: NkiTensor, b: NkiTensor, what: str) -> None:
  assert a.shape == b.shape, f"NKISHIM: {what}: shape {a.shape} != {b.shape}"

def _operand(o: Any, data: NkiTensor) -> Any:
  """tensor_scalar/activation operand: a compile-time float or a (P, 1) on-chip vector."""
  if isinstance(o, NkiTensor):
    _onchip(o)
    assert o.shape[0] == data.shape[0] and o.size == o.shape[0], f"NKISHIM: operand {o!r} must be ({data.shape[0]}, 1)"
    return o.f32().reshape([o.shape[0]] + [1] * (data.ndim - 1))
  assert isinstance(o, (int, float, np.floating, np.integer)) and not isinstance(o, bool), f"NKISHIM: bad scalar operand {o!r}"
  return np.float32(o)

def _apply(op: _Op, a, b):
  with np.errstate(all="ignore"): return np.asarray(op.fn(a, b)).astype(np.float32)

class _ISA:
  """nki.isa: dst-first instructions."""
  engine = engine
  def dma_copy(self, dst: NkiTensor, src: NkiTensor, **kw) -> None:
    assert dst.size == src.size, f"NKISHIM: dma_copy element counts differ: {dst!r} <- {src!r}"
    for t in (dst, src):   # neuronx-cc (inf2) rejects a negative partition step in an HBM access pattern (NKI-N2)
      assert t.buffer in ON_CHIP or not (t.ndim > 1 and t.shape[0] > 1 and t.strides[0] < 0), f"NKISHIM: negative partition step in {t.get_pattern()}"
    assert dst.buffer not in ON_CHIP or dst.shape[0] <= PMAX
    if dst.dtype == src.dtype: dst.write_raw(src.raw())
    else: dst.write_raw(from_f32(src.f32(), dst.dtype))   # DMA casts through float32
  def memset(self, dst: NkiTensor, value, engine=engine.unknown, name=None) -> None:
    _onchip(dst)
    if dst.dtype.is_float: dst.write_raw(np.broadcast_to(from_f32(np.float32(value), dst.dtype), dst.shape))
    else:
      assert float(value) == int(value), f"NKISHIM: memset of {dst.dtype} with non-integer {value!r}"
      dst.write_raw(np.full(dst.shape, int(value)).astype(dst.dtype.storage))
  def tensor_copy(self, dst: NkiTensor, src: NkiTensor, engine=engine.unknown, name=None) -> None:
    _onchip(dst, src)
    assert dst.shape[0] == src.shape[0] and dst.size == src.size, f"NKISHIM: tensor_copy {dst!r} <- {src!r}"
    if dst.dtype == src.dtype: dst.write_raw(src.raw())                # bit-accurate when the dtypes agree
    else: dst.write_raw(from_f32(src.f32(), dst.dtype))
  def tensor_tensor(self, dst: NkiTensor, data1: NkiTensor, data2: NkiTensor, op: _Op, engine=engine.unknown, name=None) -> None:
    _onchip(dst, data1, data2); _same(data1, data2, f"tensor_tensor {op}"); _same(dst, data1, f"tensor_tensor {op} dst")
    assert op.name in ARITH or op.name in COMPARE or op.name in BITVEC, f"NKISHIM: tensor_tensor op {op} not modelled"
    _compiles(op, "tensor_tensor")
    ints = all(t.dtype.name in ("int32", "uint32") for t in (dst, data1, data2))
    if op.name in BITVEC:   # bit patterns, no float32 cast (integer tiles only)
      assert not any(t.dtype.is_float for t in (dst, data1, data2)), f"NKISHIM: {op} needs integer tiles"
      dst.write_raw(op.fn(data1.raw(), data2.raw()).astype(dst.dtype.storage))
      return
    if engine == "gpsimd":
      assert ints and op.name in GPSIMD_COMPILES, f"NKISHIM: GpSimd {op} on {data1.dtype}: neuronx-cc 2.27 rejects it for inf2"
    if engine == "gpsimd" or (engine == "unknown" and ints and op.name in INT_AUTO_GPSIMD):   # GpSimd: native integer ALU
      assert not (data1.dtype.is_float or data2.dtype.is_float), f"NKISHIM: GpSimd float tensor_tensor {op} not modelled"
      a, b = data1.values().astype(np.int64), data2.values().astype(np.int64)
      with np.errstate(all="ignore"):
        if op.name == "divide":   # true quotient truncated toward zero (the 0.6 simulator: float64 divide, then astype)
          r = np.where(b == 0, np.iinfo(np.int32).min, np.trunc(a / np.where(b == 0, 1, b)))
        else: r = op.fn(a, b)
      dst.write_raw(np.asarray(r).astype(np.int64).astype(dst.dtype.storage))   # integer results wrap
    else: dst.write_raw(from_f32(_apply(op, data1.f32(), data2.f32()), dst.dtype))
  def tensor_scalar(self, dst: NkiTensor, data: NkiTensor, op0: _Op, operand0, reverse0=False, op1: _Op | None = None, operand1=None,
                    reverse1=False, engine=engine.unknown, name=None) -> None:
    _onchip(dst, data); _same(dst, data, f"tensor_scalar {op0}")
    assert (op1 is None) == (operand1 is None), "NKISHIM: op1 and operand1 go together"
    x = data.f32()
    for op, o, rev in ((op0, operand0, reverse0), (op1, operand1, reverse1)):
      if op is None: continue
      assert op.name in ARITH or op.name in COMPARE, f"NKISHIM: tensor_scalar op {op} not modelled"
      _compiles(op, "tensor_scalar")
      assert not (op.name == "multiply" and data.dtype.name == "int32"), "NKISHIM: tensor_scalar multiply of int32 data fails neuronx-cc's MLIR verification"
      v = _operand(o, data)
      x = _apply(op, v, x) if rev else _apply(op, x, v)
    dst.write_raw(from_f32(x, dst.dtype))
  def activation(self, dst: NkiTensor, op: _Op, data: NkiTensor, bias=None, scale=1.0, reduce_op=None, reduce_res=None, reduce_cmd=None,
                 name=None) -> None:
    _onchip(dst, data); _same(dst, data, f"activation {op}")
    assert op.name in ACTIVATION, f"NKISHIM: {op} is not an activation function"
    assert reduce_op is None and reduce_res is None, "NKISHIM: activation reductions are not modelled"
    x = data.f32()
    with np.errstate(all="ignore"):
      s = _operand(scale, data)
      if not (np.ndim(s) == 0 and s == 1): x = (x * s).astype(np.float32)
      if bias is not None: x = (x + _operand(bias, data)).astype(np.float32)
      dst.write_raw(from_f32(np.asarray(ACTIVATION[op.name].fn(x), np.float32), dst.dtype))
  def reciprocal(self, dst: NkiTensor, data: NkiTensor, name=None) -> None:
    _onchip(dst, data); _same(dst, data, "reciprocal")
    with np.errstate(all="ignore"): dst.write_raw(from_f32(np.float32(1) / data.f32(), dst.dtype))
  def tensor_reduce(self, dst: NkiTensor, op: _Op, data: NkiTensor, axis, negate=False, keepdims=False, name=None) -> None:
    _onchip(dst, data)
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    assert 0 not in axes and axes == tuple(range(data.ndim - len(axes), data.ndim)), \
      f"NKISHIM: tensor_reduce axes {axes} must be the trailing free dims of {data.shape}"
    assert op.name in REDUCE, f"NKISHIM: reduce op {op} not modelled"
    r = REDUCE[op.name](data.f32(), axis=axes, keepdims=keepdims).astype(np.float32)
    if negate: r = -r
    assert dst.shape[0] == data.shape[0] and dst.size == r.size, f"NKISHIM: tensor_reduce dst {dst!r} for result {r.shape}"
    dst.write_raw(from_f32(r, dst.dtype))
  def tensor_copy_predicated(self, dst: NkiTensor, src: NkiTensor, predicate: NkiTensor, reverse_pred=False, name=None) -> None:
    # the 0.6 stub's docstring allows a scalar src, but the real isa function validates src.shape first and fails
    assert isinstance(src, NkiTensor), f"NKISHIM: tensor_copy_predicated src must be a tile, got {type(src).__name__} {src!r}"
    _onchip(dst, src, predicate); _same(dst, predicate, "tensor_copy_predicated"); _same(dst, src, "tensor_copy_predicated src")
    assert predicate.dtype.name in ("uint8", "uint16", "uint32"), f"NKISHIM: predicate dtype {predicate.dtype}"
    assert src.dtype == dst.dtype, f"NKISHIM: tensor_copy_predicated src {src.dtype} != dst {dst.dtype}"
    p = predicate.values() != 0
    if reverse_pred: p = ~p
    dst.write_raw(np.where(p, src.raw(), dst.raw()))

  def nc_matmul(self, dst: NkiTensor, stationary: NkiTensor, moving: NkiTensor, is_stationary_onezero=False, is_moving_onezero=False,
                is_transpose=False, accumulate=None, tile_position=(), tile_size=(), perf_mode=None, name=None) -> None:
    assert dst.buffer == PSUM and stationary.buffer == SBUF and moving.buffer == SBUF, \
      f"NKISHIM: nc_matmul reads SBUF and writes PSUM ({stationary!r}, {moving!r} -> {dst!r})"
    _onchip(dst, stationary, moving)
    assert not is_transpose and not tile_position and not tile_size, "NKISHIM: nc_matmul tiling/transpose modes are not modelled"
    assert accumulate in (True, False), "NKISHIM: pass accumulate=True/False explicitly (None infers per element on the device)"
    assert dst.dtype.name == "float32", f"NKISHIM: nc_matmul dst must be float32 on NeuronCore-v2, got {dst.dtype}"
    for t in (stationary, moving):
      assert t.dtype.is_float and t.ndim == 2, f"NKISHIM: nc_matmul operand {t!r} must be a 2D float tile"
    assert (stationary.dtype.name == "float32") == (moving.dtype.name == "float32"), "NKISHIM: a float32 operand needs a float32 partner"
    K, M = stationary.shape
    assert moving.shape[0] == K and M <= PMAX and moving.shape[1] <= 512, \
      f"NKISHIM: nc_matmul stationary {stationary.shape} (K <= 128, M <= 128) x moving {moving.shape} (N <= 512)"
    assert dst.shape == (M, moving.shape[1]), f"NKISHIM: nc_matmul dst {dst.shape} != {(M, moving.shape[1])}"
    if os.environ.get("FZ_NKISHIM_MM") == "f64": r = np.matmul(stationary.f32().T.astype(np.float64), moving.f32().astype(np.float64)).astype(np.float32)
    else: r = np.matmul(stationary.f32().T, moving.f32()).astype(np.float32)   # the 0.6 simulator: float32 np.matmul
    if accumulate: r = (dst.f32() + r).astype(np.float32)
    dst.write_raw(from_f32(r, dst.dtype))
  def nc_transpose(self, dst: NkiTensor, data: NkiTensor, engine=engine.unknown, name=None) -> None:
    _onchip(dst, data)
    assert dst.buffer == PSUM and data.buffer == SBUF and engine in ("unknown", "tensor"), \
      "NKISHIM: only the TensorE transpose (SBUF -> PSUM) is modelled"
    assert data.ndim == 2 and data.shape[0] <= PMAX and data.shape[1] <= PMAX, f"NKISHIM: nc_transpose of {data.shape} (TensorE: <= 128 x 128)"
    assert dst.shape == data.shape[::-1] and dst.dtype == data.dtype, f"NKISHIM: nc_transpose {data!r} -> {dst!r}"
    dst.write_raw(data.raw().T)   # exact in the simulator; on NeuronCore-v2 an identity matmul (NaN/Inf are not bit-accurate)
  def _affine(self, dst: NkiTensor, pattern, offset, channel_multiplier) -> np.ndarray:
    assert dst.ndim == 2, f"NKISHIM: iota/affine_select on a 2D tile, got {dst.shape}"
    terms = [(int(st), int(n)) for st, n in pattern if not (st == 0 and n == 1)]
    assert int(np.prod([n for _, n in terms])) == dst.shape[1], f"NKISHIM: pattern {pattern} does not cover {dst.shape}"
    v = np.zeros(1, np.int64)
    for st, n in terms: v = (v[:, None] + np.arange(n, dtype=np.int64)[None, :] * st).reshape(-1)
    return int(offset) + np.arange(dst.shape[0], dtype=np.int64)[:, None] * int(channel_multiplier) + v[None, :]
  def iota(self, dst: NkiTensor, pattern, offset=0, channel_multiplier=0, name=None) -> None:
    _onchip(dst)
    v = self._affine(dst, pattern, offset, channel_multiplier)
    dst.write_raw(from_f32(v.astype(np.float32), dst.dtype) if dst.dtype.is_float else v.astype(dst.dtype.storage))
  def affine_select(self, dst: NkiTensor, pattern, channel_multiplier, on_true_tile: NkiTensor, on_false_value, cmp_op: _Op = None,
                    offset=0, name=None) -> None:
    _onchip(dst, on_true_tile); _same(dst, on_true_tile, "affine_select")
    assert cmp_op is not None and cmp_op.name in COMPARE, f"NKISHIM: affine_select compare {cmp_op}"
    pred = cmp_op.fn(self._affine(dst, pattern, offset, channel_multiplier), 0)
    dst.write_raw(from_f32(np.where(pred, on_true_tile.f32(), np.float32(on_false_value)), dst.dtype))

def jit(fn=None, **kwargs):
  """nki.jit: the shim calls the traced function directly on NkiTensors."""
  if fn is None: return lambda f: jit(f, **kwargs)
  fn.__nki_jit__ = True
  return fn

# ---------------------------------------------------------------- module namespaces

nl = types.ModuleType("nki.language")
for _n, _d in DTYPES.items(): setattr(nl, _n, _d)
for _n, _o in OPS.items(): setattr(nl, _n, _o)
nl.sbuf, nl.psum, nl.hbm, nl.shared_hbm, nl.private_hbm = SBUF, PSUM, HBM, SHARED_HBM, PRIVATE_HBM
nl.ndarray, nl.zeros, nl.NkiTensor = ndarray, zeros, NkiTensor
nl.affine_range = nl.sequential_range = nl.static_range = range   # nki 0.6: every range is a fully unrolled python range
nl.tile_size = types.SimpleNamespace(pmax=PMAX)
isa = _ISA()
nisa = types.ModuleType("nki.isa")
for _n in ("dma_copy", "memset", "tensor_copy", "tensor_tensor", "tensor_scalar", "activation", "reciprocal", "tensor_reduce",
           "tensor_copy_predicated", "nc_matmul", "nc_transpose", "iota", "affine_select"): setattr(nisa, _n, getattr(isa, _n))
nisa.engine = engine
nisa.vector_engine, nisa.scalar_engine, nisa.gpsimd_engine, nisa.unknown_engine = engine.vector, engine.scalar, engine.gpsimd, engine.unknown
# N8 (I72): nki.collectives.all_reduce, the subset AllReduceSpec emits. The shim has no peer NeuronCore: the values go to
# COLLECTIVE[0](values float32, op name, replica groups) -> the reduced values, which fusezap.serve.neuron_tp installs in a
# tensor-parallel rank (the group's host all-reduce: rank 0 sums in rank order). Without it a collective raises.
COLLECTIVE: list = []

class ReplicaGroup:
  def __init__(self, _value): self._value = [list(g) for g in _value]
  def __repr__(self): return f"ReplicaGroup({self._value})"

def _all_reduce(srcs, dsts, replica_group, op, priority=None, name=None) -> None:
  assert len(srcs) == len(dsts) > 0 and isinstance(replica_group, ReplicaGroup), "NKISHIM: all_reduce(srcs=[...], dsts=[...], replica_group=ReplicaGroup)"
  assert op.name in ("add", "maximum", "minimum"), f"NKISHIM: all_reduce op {op}"
  assert len({t.buffer for t in [*srcs, *dsts]}) == 1 and PSUM not in {t.buffer for t in srcs}, "NKISHIM: all_reduce tensors share one buffer (not PSUM)"
  if not COLLECTIVE: raise RuntimeError("NKISHIM: a collective outside a tensor-parallel group (fusezap.serve.neuron_tp installs the peer exchange)")
  for s, d in zip(srcs, dsts):
    assert s.shape == d.shape and s.dtype == d.dtype, f"NKISHIM: all_reduce {s!r} -> {d!r}"
    r = np.asarray(COLLECTIVE[0](np.ascontiguousarray(s.f32()), op.name, replica_group._value), np.float32).reshape(d.shape)
    d.write_raw(from_f32(r, d.dtype))

ncc = types.ModuleType("nki.collectives")
ncc.all_reduce, ncc.ReplicaGroup = _all_reduce, ReplicaGroup
nki = types.ModuleType("nki")
nki.jit, nki.language, nki.isa, nki.collectives = jit, nl, nisa, ncc
_MODULES = {"nki": nki, "nki.language": nl, "nki.isa": nisa, "nki.collectives": ncc}

def _import(name, globals=None, locals=None, fromlist=(), level=0):
  if name in _MODULES: return _MODULES[name] if fromlist else nki
  return builtins.__import__(name, globals, locals, fromlist, level)

_CODE: dict[str, Any] = {}

def load(src: str, name: str):
  """Execute generated kernel source against the shim; returns the kernel function `name`."""
  if (code := _CODE.get(src)) is None: _CODE[src] = code = compile(src, f"<nki:{name}>", "exec")
  g: dict[str, Any] = {"__builtins__": {**builtins.__dict__, "__import__": _import}, "__name__": f"nki_kernel_{name}"}
  exec(code, g)
  return g[name]

def execute(prog: Any, arrays: list[np.ndarray]) -> float:
  """Run an NKIProgram on flat numpy storage arrays (in place; outputs are written into their arrays). Returns seconds."""
  from fusezap.codegen.nki import nl_dtype
  assert len(arrays) == prog.n_bufs, f"NKISHIM: {prog.name} takes {prog.n_bufs} buffers, got {len(arrays)}"
  ts = [tensor(a, getattr(nl, nl_dtype(prog.buf_dtypes[i])) if i in prog.buf_dtypes else nl.uint8, name=f"data{i}")
        for i, a in enumerate(arrays)]
  fn = load(prog.src, prog.name)
  st = time.perf_counter()
  ret = fn(*ts)
  et = time.perf_counter() - st
  ret = ret if isinstance(ret, tuple) else (ret,)
  assert [r.name for r in ret] == [f"data{i}" for i in prog.outs], f"NKISHIM: {prog.name} returned {ret}, expected outputs {prog.outs}"
  return et

def run_item(si: Any, renderer: Any = None) -> float:
  """Render and run one kind="kernel" ScheduleItem whose buffers are NPY buffers (numpy handles)."""
  from fusezap.codegen.nki import NKIRenderer
  from fusezap.dtype import to_np
  prog = (renderer or NKIRenderer()).render(si.ast, si.var_vals)
  arrays = [b.ensure_allocated().handle[:b.nbytes].view(to_np(b.dtype)) if b.size else np.zeros(1, to_np(b.dtype)) for b in si.bufs]
  arrays += [np.zeros(1, np.uint8)] * (prog.n_bufs - len(arrays))
  return execute(prog, arrays[:prog.n_bufs])
