"""User-facing Tensor (SPEC §9). Implemented by WP2. This stub **is** the API contract; WP6/WP7 code against it.

Semantics follow PyTorch where names coincide. `axis` == torch `dim` (both keywords accepted).
Broadcasting follows numpy rules. Dtype promotion follows fusezap.dtype.promote, with python scalars "weak".
All ops are lazy; `realize()`, `numpy()`, `item()` and `tolist()` force computation.

Weak scalars: a python scalar takes the tensor's dtype, except that a float scalar with an int/bool tensor
gives default_float and an int scalar with a bool tensor gives default_int.
float16/bfloat16 reductions, matmul, softmax-like ops and norms accumulate in float32 and round once at the end.
"""
from __future__ import annotations
import math
from typing import Any, Sequence
import numpy as np
from fusezap.dtype import DType, dtypes, promote, least_upper_float, to_np
from fusezap.helpers import prod, canonicalize_axis
from fusezap.ops import LoadOps
from fusezap.lazy import LazyBuffer
from fusezap.device import Device
from fusezap.shape.symbolic import Node
import fusezap.autograd as F

Shape = tuple[int, ...]

def _shape_arg(shape) -> tuple:
  """(*shape) varargs -> tuple: accepts ints or one tuple/list."""
  if len(shape) == 1 and isinstance(shape[0], (tuple, list)): return tuple(shape[0])
  return tuple(shape)

def _pick_axis(axis, dim):
  return dim if dim is not None else axis

_HALF = (dtypes.float16, dtypes.bfloat16)

class _Rng:
  gen: np.random.Generator = np.random.Generator(np.random.Philox(0))

def _np_dtype_for_list(a: np.ndarray) -> DType:
  if a.dtype.kind == "b": return dtypes.bool
  if a.dtype.kind in "iu": return dtypes.default_int
  if a.dtype.kind == "f": return dtypes.default_float
  raise TypeError(f"cannot make a Tensor from data of numpy dtype {a.dtype}")

def _host_array(a: np.ndarray, dtype: DType) -> np.ndarray:
  """numpy data -> host storage array of `dtype` (bfloat16 -> uint16 bits)."""
  if dtype == dtypes.bfloat16:
    from fusezap.interpret import _f32_to_bf16
    return _f32_to_bf16(np.asarray(a, np.float32))
  if dtype == dtypes.bool: return np.asarray(a).astype(bool)
  return np.asarray(a).astype(to_np(dtype))

class Tensor:
  no_grad: bool = False        # class-level flag; use `with Tensor.inference():` or fusezap.no_grad()
  training: bool = False       # enables dropout

  lazydata: LazyBuffer
  grad: "Tensor | None"
  requires_grad: bool | None

  def __init__(self, data: Any, device: str | None = None, dtype: DType | None = None,
               requires_grad: bool | None = None):
    """data: python scalar / nested list / np.ndarray / LazyBuffer / Tensor."""
    self.grad, self.requires_grad, self._ctx = None, requires_grad, None
    if isinstance(data, Tensor): data = data.lazydata
    if isinstance(data, LazyBuffer):
      lb = data
      if dtype is not None and dtype != lb.dtype: lb = lb.cast(dtype)
      if device is not None and Device.canonicalize(device) != lb.device: lb = lb.copy_to_device(Device.canonicalize(device))
      self.lazydata = lb
      return
    device = Device.canonicalize(device)
    if isinstance(data, (bool, int, float)) and not isinstance(data, np.generic):
      dt = dtype or dtypes.from_py(data)
      self.lazydata = LazyBuffer.loadop(LoadOps.CONST, (), dt, device, data)
      return
    if isinstance(data, Node):
      self.lazydata = LazyBuffer.loadop(LoadOps.CONST, (), dtype or dtypes.default_int, device, data)
      return
    if isinstance(data, (np.ndarray, np.generic)):
      a = np.asarray(data)
      dt = dtype or dtypes.from_np(a.dtype)
    elif isinstance(data, (list, tuple)):
      a = np.array(data)
      dt = dtype or (_np_dtype_for_list(a) if a.size else dtypes.default_float)
    elif isinstance(data, (bytes, memoryview)):
      dt = dtype or dtypes.uint8
      a = np.frombuffer(data, dtype=to_np(dt))
    else: raise TypeError(f"cannot make a Tensor from {type(data)}")
    host = a if dt == dtypes.bfloat16 and a.dtype == np.uint16 else _host_array(a, dt)  # uint16 + bfloat16: already bits
    self.lazydata = LazyBuffer.loadop(LoadOps.COPY, tuple(int(s) for s in a.shape), dt, device, np.ascontiguousarray(host).reshape(-1))

  # ---- properties
  @property
  def shape(self) -> Shape: return self.lazydata.shape
  @property
  def dtype(self) -> DType: return self.lazydata.dtype
  @property
  def device(self) -> str: return self.lazydata.device
  @property
  def ndim(self) -> int: return len(self.shape)
  def numel(self) -> int: return prod(self.shape)
  @property
  def T(self) -> "Tensor": return self.permute(*range(self.ndim - 1, -1, -1))
  def __len__(self) -> int:
    if not self.shape: raise TypeError("len() of a 0-d tensor")
    return self.shape[0]

  # ---- creation
  @staticmethod
  def _const(shape, val, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    dt = dtype or dtypes.from_py(val)
    return Tensor(LazyBuffer.loadop(LoadOps.CONST, tuple(shape), dt, Device.canonicalize(device), val), requires_grad=requires_grad)
  @staticmethod
  def empty(*shape, dtype: DType | None = None, device: str | None = None, **kw) -> "Tensor":
    return Tensor(LazyBuffer.loadop(LoadOps.EMPTY, _shape_arg(shape), dtype or dtypes.default_float, Device.canonicalize(device)), **kw)
  @staticmethod
  def zeros(*shape, dtype: DType | None = None, **kw) -> "Tensor": return Tensor.full(_shape_arg(shape), 0.0, dtype=dtype or dtypes.default_float, **kw)
  @staticmethod
  def ones(*shape, dtype: DType | None = None, **kw) -> "Tensor": return Tensor.full(_shape_arg(shape), 1.0, dtype=dtype or dtypes.default_float, **kw)
  @staticmethod
  def full(shape: Shape, fill_value, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
    return Tensor._const(shape, fill_value, dtype, device, requires_grad)
  @staticmethod
  def arange(start, stop=None, step=1, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    if stop is None: start, stop = 0, start
    dt = dtype or (dtypes.default_float if any(isinstance(x, float) for x in (start, stop, step)) else dtypes.default_int)
    n = max(math.ceil((stop - start) / step), 0)
    a = start + np.arange(n, dtype=np.float64 if dt.is_float else np.int64) * step
    return Tensor(a, dtype=dt, device=device, requires_grad=requires_grad)
  @staticmethod
  def eye(n: int, m: int | None = None, dtype: DType | None = None, **kw) -> "Tensor":
    m = n if m is None else m
    i = Tensor.arange(n, device=kw.get("device"))
    j = Tensor.arange(m, device=kw.get("device"))
    return Tensor((i.reshape(n, 1) == j.reshape(1, m)).cast(dtype or dtypes.default_float).lazydata, requires_grad=kw.get("requires_grad"))
  @staticmethod
  def _from_rng(a: np.ndarray, dtype: DType | None, device: str | None, requires_grad: bool | None, default: DType) -> "Tensor":
    t = Tensor(a, device=device)
    dt = dtype or default
    if dt != t.dtype: t = t.cast(dt)
    t.requires_grad = requires_grad
    return t
  @staticmethod
  def rand(*shape, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    """Uniform [0, 1). Host Philox stream seeded by fusezap.manual_seed."""
    a = _Rng.gen.random(_shape_arg(shape), dtype=np.float32)
    return Tensor._from_rng(a, dtype, device, requires_grad, dtypes.default_float)
  @staticmethod
  def randn(*shape, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    a = _Rng.gen.standard_normal(_shape_arg(shape), dtype=np.float32)
    return Tensor._from_rng(a, dtype, device, requires_grad, dtypes.default_float)
  @staticmethod
  def randint(*shape, low=0, high=10, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    a = _Rng.gen.integers(low, high, size=_shape_arg(shape), dtype=np.int32)
    return Tensor._from_rng(a, dtype, device, requires_grad, dtypes.default_int)
  @staticmethod
  def uniform(*shape, low=0.0, high=1.0, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    a = (_Rng.gen.random(_shape_arg(shape), dtype=np.float32) * np.float32(high - low) + np.float32(low)).astype(np.float32)
    return Tensor._from_rng(a, dtype, device, requires_grad, dtypes.default_float)
  @staticmethod
  def normal(*shape, mean=0.0, std=1.0, dtype: DType | None = None, device: str | None = None, requires_grad: bool | None = None) -> "Tensor":
    a = (_Rng.gen.standard_normal(_shape_arg(shape), dtype=np.float32) * np.float32(std) + np.float32(mean)).astype(np.float32)
    return Tensor._from_rng(a, dtype, device, requires_grad, dtypes.default_float)
  def zeros_like(self, **kw) -> "Tensor": return self.full_like(0, **kw)
  def ones_like(self, **kw) -> "Tensor": return self.full_like(1, **kw)
  def full_like(self, fill_value, **kw) -> "Tensor":
    return Tensor.full(self.shape, fill_value, dtype=kw.pop("dtype", self.dtype), device=kw.pop("device", self.device), **kw)

  # ---- realization / data movement
  def realize(self) -> "Tensor":
    from fusezap.realize import realize
    if not self.lazydata.is_unrealized_const() and not self.lazydata.is_realized(): realize([self.lazydata])
    return self
  def _realized_contiguous(self) -> LazyBuffer:
    """A realized base holding exactly this tensor's elements in row-major order."""
    lb = self.lazydata
    if lb.base is lb and lb.op is not LoadOps.CONST and lb.realized is not None: return lb
    if not (lb.base is lb and lb.op is not LoadOps.CONST): lb = lb.contiguous()
    from fusezap.realize import realize
    realize([lb])
    return lb
  def numpy(self) -> np.ndarray:
    assert all(isinstance(s, int) for s in self.shape), f"numpy() needs a concrete shape, got {self.shape}"
    lb = self._realized_contiguous()
    return lb.realized.numpy().reshape(self.shape)
  def item(self):
    assert self.numel() == 1, f"item() needs exactly one element, shape is {self.shape}"
    a = self.numpy().reshape(-1)
    if self.dtype == dtypes.bfloat16:
      from fusezap.interpret import _bf16_to_f32
      a = _bf16_to_f32(a)
    return a[0].item()
  def tolist(self):
    a = self.numpy()
    if self.dtype == dtypes.bfloat16:
      from fusezap.interpret import _bf16_to_f32
      a = _bf16_to_f32(a).reshape(self.shape)
    return a.tolist()
  def to(self, device: str) -> "Tensor":
    device = Device.canonicalize(device)
    if device == self.device: return self
    ret = F.Copy.apply(self, device=device)
    return ret
  def detach(self) -> "Tensor": return Tensor(self.lazydata, requires_grad=False)
  def contiguous(self) -> "Tensor": return F.Contiguous.apply(self)
  def assign(self, x: "Tensor") -> "Tensor":
    """In-place update (used by optimizers): self's storage takes x's value. Shapes/dtypes must match."""
    if not isinstance(x, Tensor): x = Tensor(x, device=self.device, dtype=self.dtype)
    assert tuple(x.shape) == tuple(self.shape), f"assign shape mismatch {self.shape} vs {x.shape}"
    assert x.device == self.device, f"assign device mismatch {self.device} vs {x.device}"
    if x.dtype != self.dtype: x = x.cast(self.dtype)
    self.lazydata = x.lazydata
    return self
  def cast(self, dtype: DType) -> "Tensor": return self if dtype == self.dtype else F.Cast.apply(self, dtype=dtype)
  def bitcast(self, dtype: DType) -> "Tensor":
    assert dtype.itemsize == self.dtype.itemsize, f"bitcast {self.dtype} -> {dtype} changes itemsize"
    return self if dtype == self.dtype else F.Cast.apply(self, dtype=dtype, bitcast=True)
  def float(self) -> "Tensor": return self.cast(dtypes.float32)
  def half(self) -> "Tensor": return self.cast(dtypes.float16)
  def int(self) -> "Tensor": return self.cast(dtypes.int32)
  def bool(self) -> "Tensor": return self.cast(dtypes.bool)

  # ---- movement
  def reshape(self, *shape) -> "Tensor":
    """Accepts ints or one tuple; a single -1 is inferred."""
    shape = _shape_arg(shape)
    if sum(1 for s in shape if isinstance(s, int) and s == -1) > 1: raise ValueError(f"only one -1 allowed in {shape}")
    if any(isinstance(s, int) and s == -1 for s in shape):
      rest = prod([s for s in shape if not (isinstance(s, int) and s == -1)])
      shape = tuple((self.numel() // rest if rest else 0) if isinstance(s, int) and s == -1 else s for s in shape)
    if all(isinstance(s, int) for s in shape + self.shape) and prod(shape) != self.numel():
      raise ValueError(f"cannot reshape {self.shape} to {shape}")
    return self if tuple(shape) == self.shape else F.Reshape.apply(self, shape=tuple(shape))
  def view(self, *shape) -> "Tensor": return self.reshape(*shape)
  def permute(self, *order) -> "Tensor":
    order = tuple(canonicalize_axis(o, self.ndim) for o in _shape_arg(order))
    assert sorted(order) == list(range(self.ndim)), f"bad permutation {order} for ndim {self.ndim}"
    return self if order == tuple(range(self.ndim)) else F.Permute.apply(self, order=order)
  def transpose(self, ax0: int = 1, ax1: int = 0) -> "Tensor":
    order = list(range(self.ndim))
    a, b = canonicalize_axis(ax0, self.ndim), canonicalize_axis(ax1, self.ndim)
    order[a], order[b] = order[b], order[a]
    return self.permute(order)
  def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
    if self.ndim == 0: return self.reshape(1)
    s, e = canonicalize_axis(start_dim, self.ndim), canonicalize_axis(end_dim, self.ndim)
    return self.reshape(self.shape[:s] + (prod(self.shape[s:e + 1]),) + self.shape[e + 1:])
  def unflatten(self, axis: int, sizes: Shape) -> "Tensor":
    axis = canonicalize_axis(axis, self.ndim)
    return self.reshape(self.shape[:axis] + tuple(sizes) + self.shape[axis + 1:])
  def squeeze(self, axis: int | None = None, *, dim: int | None = None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    if axis is None: return self.reshape(tuple(s for s in self.shape if s != 1))
    if self.ndim == 0: return self
    axis = canonicalize_axis(axis, self.ndim)
    return self if self.shape[axis] != 1 else self.reshape(self.shape[:axis] + self.shape[axis + 1:])
  def unsqueeze(self, axis: int | None = None, *, dim: int | None = None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    axis = canonicalize_axis(axis, self.ndim + 1)
    return self.reshape(self.shape[:axis] + (1,) + self.shape[axis:])
  def expand(self, *shape) -> "Tensor":
    """-1 keeps a dim; may add leading dims."""
    shape = _shape_arg(shape)
    assert len(shape) >= self.ndim, f"cannot expand {self.shape} to {shape}"
    x = self.reshape((1,) * (len(shape) - self.ndim) + self.shape) if len(shape) > self.ndim else self
    new = tuple(o if isinstance(s, int) and s == -1 else s for s, o in zip(shape, x.shape))
    for o, s in zip(x.shape, new):
      if not (o == s or o == 1): raise ValueError(f"cannot expand {self.shape} to {shape}")
    return x if new == x.shape else F.Expand.apply(x, shape=new)
  def pad(self, padding: Sequence[tuple[int, int]], value: float = 0.0) -> "Tensor":
    """One (before, after) per axis, ordered first axis to last (NOT torch's reversed order)."""
    padding = tuple((0, 0) if p is None else (p[0], p[1]) for p in padding)
    assert len(padding) == self.ndim, f"pad needs one (before, after) per axis, got {padding} for {self.shape}"
    if all(b == 0 and e == 0 for b, e in padding): return self
    if any(b < 0 or e < 0 for b, e in padding if isinstance(b, int) and isinstance(e, int)):  # negative pad = shrink
      x = self.shrink(tuple((-min(b, 0), s + min(e, 0)) for (b, e), s in zip(padding, self.shape)))
      return x.pad(tuple((max(b, 0), max(e, 0)) for b, e in padding), value)
    ret = F.Pad.apply(self, arg=padding)
    if value == 0: return ret
    mask = F.Pad.apply(Tensor._const(self.shape, True, dtypes.bool, self.device), arg=padding)
    return mask.where(ret, value)
  def shrink(self, arg: Sequence[tuple[int, int] | None]) -> "Tensor":
    arg = tuple((0, s) if a is None else (a[0], a[1]) for a, s in zip(arg, self.shape))
    assert len(arg) == self.ndim, f"shrink needs one (start, end) per axis, got {arg} for {self.shape}"
    if all(b == 0 and e == s for (b, e), s in zip(arg, self.shape)): return self
    return F.Shrink.apply(self, arg=arg)
  def flip(self, axis: int | Sequence[int], *, dims=None) -> "Tensor":
    axis = dims if dims is not None else axis
    axes = {canonicalize_axis(a, self.ndim) for a in ((axis,) if isinstance(axis, int) else axis)}
    if not axes: return self
    return F.Flip.apply(self, arg=tuple(i in axes for i in range(self.ndim)))

  def _slice_axis(self, ax: int, sl: slice) -> "Tensor":
    """Basic slicing of one axis (any step)."""
    n = self.shape[ax]
    start, stop, step = sl.indices(n)
    x = self
    if step < 0:
      x = x.flip(ax)
      start, stop, step = n - 1 - start, n - 1 - stop, -step
    length = max(0, -(-(stop - start) // step))
    if step == 1 or length <= 1:
      return x.shrink(tuple((start, start + length) if i == ax else None for i in range(x.ndim)))
    # strided: take [start, start + length*step), split into (length, step) and keep column 0
    end = start + length * step
    if end > n: x = x.pad(tuple((0, end - n) if i == ax else (0, 0) for i in range(x.ndim)))
    x = x.shrink(tuple((start, end) if i == ax else None for i in range(x.ndim)))
    x = x.reshape(x.shape[:ax] + (length, step) + x.shape[ax + 1:])
    x = x.shrink(tuple((0, 1) if i == ax + 1 else None for i in range(x.ndim)))
    return x.reshape(x.shape[:ax] + (length,) + x.shape[ax + 2:])

  def _index_select(self, ax: int, idx: "Tensor") -> "Tensor":
    """x[..., idx, ...] on axis ax: result shape x.shape[:ax] + idx.shape + x.shape[ax+1:] (one-hot where + sum)."""
    n = self.shape[ax]
    if idx.dtype == dtypes.bool: raise NotImplementedError("boolean mask indexing is not supported")
    idx = idx.cast(dtypes.default_int) if idx.dtype != dtypes.default_int else idx
    idx = (idx < 0).where(idx + n, idx)
    pre, post = self.shape[:ax], self.shape[ax + 1:]
    k = idx.ndim
    xt = self.permute(*range(ax), *range(ax + 1, self.ndim), ax)                  # pre, post, n
    xt = xt.reshape(pre + (1,) * k + post + (n,))
    oh = idx.reshape((1,) * len(pre) + idx.shape + (1,) * len(post) + (1,)) == Tensor.arange(n, device=self.device)
    out = oh.where(xt, Tensor._const((), 0, self.dtype, self.device))
    return out._sum_exact(-1)

  def _sum_exact(self, axis: int) -> "Tensor":
    """Sum over one axis keeping the dtype (used where at most one term is nonzero)."""
    if self.dtype == dtypes.bool: return self.cast(dtypes.default_int).sum(axis).cast(dtypes.bool)
    return self.sum(axis, dtype=self.dtype)

  def __getitem__(self, idx) -> "Tensor":
    """ints, slices (any step, incl. negative), None, Ellipsis, and int Tensors (gather on that axis)."""
    if not isinstance(idx, tuple): idx = (idx,)
    idx = tuple(_index_array(i, self.device) if isinstance(i, (list, np.ndarray)) else i for i in idx)
    n_real = sum(1 for i in idx if i is not None and i is not Ellipsis)
    if n_real > self.ndim: raise IndexError(f"too many indices for tensor of dimension {self.ndim}")
    if sum(1 for i in idx if i is Ellipsis) > 1: raise IndexError("only one Ellipsis allowed")
    if Ellipsis in [i for i in idx if not isinstance(i, Tensor)]:
      k = next(j for j, i in enumerate(idx) if i is Ellipsis)
      idx = idx[:k] + (slice(None),) * (self.ndim - n_real) + idx[k + 1:]
    else: idx = idx + (slice(None),) * (self.ndim - n_real)
    x, ax, final_shape, kinds, tensor_idx = self, 0, [], [], None
    for i in idx:
      if i is None:
        final_shape.append(1)
        continue
      if isinstance(i, Tensor):
        if tensor_idx is not None: raise NotImplementedError("indexing with more than one Tensor is not supported")
        tensor_idx = (ax, i)
        final_shape.extend(i.shape)
        kinds.append("t")
      elif isinstance(i, (int, np.integer)) and not isinstance(i, bool):
        n = x.shape[ax]
        j = int(i) + n if i < 0 else int(i)
        if not 0 <= j < n: raise IndexError(f"index {i} out of range for axis {ax} with size {n}")
        x = x.shrink(tuple((j, j + 1) if a == ax else None for a in range(x.ndim)))
        kinds.append("i")
      elif isinstance(i, slice):
        x = x._slice_axis(ax, i)
        final_shape.append(x.shape[ax])
        kinds.append("s")
      else: raise TypeError(f"unsupported index {i!r}")
      ax += 1
    if tensor_idx is not None:
      tax, t = tensor_idx
      x = x.reshape(tuple(s for s, k in zip(x.shape, kinds) if k != "i"))
      x = x._index_select(sum(1 for k in kinds[:tax] if k != "i"), t)
    return x.reshape(tuple(final_shape))

  def gather(self, axis: int, index: "Tensor", *, dim: int | None = None) -> "Tensor":
    axis = canonicalize_axis(_pick_axis(axis, dim), self.ndim)
    assert index.ndim == self.ndim, f"gather index must have ndim {self.ndim}, got {index.ndim}"
    n = self.shape[axis]
    x = self.shrink(tuple(None if a == axis else (0, s) for a, s in enumerate(index.shape)))
    xt = x.permute(*range(axis), *range(axis + 1, x.ndim), axis).unsqueeze(axis)   # shape with axis->1, n last
    xt = xt.expand(index.shape + (n,))
    idx = index.cast(dtypes.default_int) if index.dtype != dtypes.default_int else index
    idx = (idx < 0).where(idx + n, idx)
    oh = idx.unsqueeze(-1) == Tensor.arange(n, device=self.device)
    return oh.where(xt, Tensor._const((), 0, self.dtype, self.device))._sum_exact(-1)
  def cat(self, *others: "Tensor", axis: int = 0, dim: int | None = None) -> "Tensor":
    axis = canonicalize_axis(_pick_axis(axis, dim), self.ndim)
    ts = [self, *others]
    for t in ts:
      assert t.ndim == self.ndim and all(a == b for k, (a, b) in enumerate(zip(t.shape, self.shape)) if k != axis), \
        f"cat shape mismatch {self.shape} vs {t.shape}"
    dt = ts[0].dtype
    for t in ts[1:]: dt = promote(dt, t.dtype)
    ts = [t.cast(dt) for t in ts]
    total = sum(t.shape[axis] for t in ts)
    out, off = None, 0
    for t in ts:
      if t.shape[axis] == 0 and len(ts) > 1: continue
      p = t.pad(tuple((off, total - off - t.shape[axis]) if a == axis else (0, 0) for a in range(t.ndim)))
      out = p if out is None else (out.maximum(p) if dt == dtypes.bool else out + p)
      off += t.shape[axis]
    return out if out is not None else ts[0]
  @staticmethod
  def stack(tensors: Sequence["Tensor"], axis: int = 0, dim: int | None = None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    tensors = list(tensors)
    axis = canonicalize_axis(axis, tensors[0].ndim + 1)
    us = [t.unsqueeze(axis) for t in tensors]
    return us[0].cat(*us[1:], axis=axis)
  def split(self, sizes: int | Sequence[int], axis: int = 0, dim: int | None = None) -> list["Tensor"]:
    axis = canonicalize_axis(_pick_axis(axis, dim), self.ndim)
    n = self.shape[axis]
    if isinstance(sizes, int):
      sizes = [min(sizes, n - i) for i in range(0, n, max(sizes, 1))] or [0]
    assert sum(sizes) == n, f"split sizes {sizes} do not sum to {n}"
    out, off = [], 0
    for s in sizes:
      out.append(self.shrink(tuple((off, off + s) if a == axis else None for a in range(self.ndim))))
      off += s
    return out
  def chunk(self, n: int, axis: int = 0, dim: int | None = None) -> list["Tensor"]:
    axis = canonicalize_axis(_pick_axis(axis, dim), self.ndim)
    size = self.shape[axis]
    return self.split(max(math.ceil(size / n), 1), axis)
  def repeat(self, *repeats) -> "Tensor":
    repeats = _shape_arg(repeats)
    assert len(repeats) >= self.ndim, f"repeat needs at least {self.ndim} sizes"
    x = self.reshape((1,) * (len(repeats) - self.ndim) + self.shape)
    base = x.shape
    x = x.reshape(tuple(v for s in base for v in (1, s)))
    x = x.expand(tuple(v for r, s in zip(repeats, base) for v in (r, s)))
    return x.reshape(tuple(r * s for r, s in zip(repeats, base)))

  # ---- dtype helpers
  def _float(self) -> "Tensor": return self.cast(least_upper_float(self.dtype))
  def _upcast(self) -> "Tensor": return self.float() if self.dtype in _HALF else self._float()

  # ---- elementwise (unary)
  def neg(self) -> "Tensor": return self.logical_not() if self.dtype == dtypes.bool else F.Neg.apply(self)
  def exp(self) -> "Tensor":  # MUL by log2(e) then EXP2: half inputs run in f32 so the product is not rounded first
    x = self._float()
    return F.Exp.apply(x.float()).cast(x.dtype) if x.dtype in _HALF else F.Exp.apply(x)
  def exp2(self) -> "Tensor": return F.Exp2.apply(self._float())
  def log(self) -> "Tensor":
    x = self._float()
    return F.Log.apply(x.float()).cast(x.dtype) if x.dtype in _HALF else F.Log.apply(x)
  def log2(self) -> "Tensor": return F.Log2.apply(self._float())
  def sqrt(self) -> "Tensor": return F.Sqrt.apply(self._float())
  def rsqrt(self) -> "Tensor": return self.sqrt().reciprocal()
  def reciprocal(self) -> "Tensor": return F.Reciprocal.apply(self._float())
  def square(self) -> "Tensor": return self * self
  def sin(self) -> "Tensor": return F.Sin.apply(self._float())
  def cos(self) -> "Tensor":
    x = self._float()
    return (math.pi / 2 - x.float()).sin().cast(x.dtype) if x.dtype in _HALF else (math.pi / 2 - x).sin()
  def tanh(self) -> "Tensor":
    x = self._float()
    if x.dtype in _HALF: return (2.0 * (2.0 * x.float()).sigmoid() - 1.0).cast(x.dtype)
    return 2.0 * (2.0 * x).sigmoid() - 1.0
  def sigmoid(self) -> "Tensor":
    x = self._float()
    return F.Sigmoid.apply(x.float()).cast(x.dtype) if x.dtype in _HALF else F.Sigmoid.apply(x)
  def relu(self) -> "Tensor": return F.Relu.apply(self)
  def gelu(self) -> "Tensor":
    """tanh approximation (== torch gelu(approximate='tanh'))."""
    x = self._float()
    xf = x.float() if x.dtype in _HALF else x
    out = 0.5 * xf * (1 + (math.sqrt(2 / math.pi) * (xf + 0.044715 * xf * xf * xf)).tanh())
    return out.cast(x.dtype)
  def silu(self) -> "Tensor":
    x = self._float()
    xf = x.float() if x.dtype in _HALF else x
    return (xf * xf.sigmoid()).cast(x.dtype)
  def abs(self) -> "Tensor":
    if self.dtype == dtypes.bool: return self
    return self.relu() + (-self).relu()
  def sign(self) -> "Tensor":
    if self.dtype == dtypes.bool: return self
    return (self > 0).cast(self.dtype) - (self < 0).cast(self.dtype)
  def logical_not(self) -> "Tensor":
    x = self if self.dtype == dtypes.bool else self.cast(dtypes.bool)
    return F.Eq.apply(x, Tensor._const(x.shape, False, dtypes.bool, x.device))

  # ---- broadcasting
  def _scalar_dtype(self, y) -> DType:
    """Result dtype of self (op) python scalar y, with y weak."""
    if isinstance(y, bool): return self.dtype
    if isinstance(y, float) and not self.dtype.is_float: return dtypes.default_float
    if isinstance(y, int) and self.dtype == dtypes.bool: return dtypes.default_int
    return self.dtype
  def _broadcasted(self, y, reverse: bool = False) -> tuple["Tensor", "Tensor"]:
    x: Tensor = self
    if not isinstance(y, Tensor):
      if isinstance(y, np.generic): y = y.item()
      assert isinstance(y, (bool, int, float, Node)), f"unsupported operand {type(y)}"
      dt = x._scalar_dtype(y) if not isinstance(y, Node) else x.dtype
      x = x.cast(dt)
      y = Tensor._const((), y, dt, x.device)
    else:
      dt = promote(x.dtype, y.dtype)
      x, y = x.cast(dt), y.cast(dt)
    if reverse: x, y = y, x
    if x.shape == y.shape: return x, y
    shape = _broadcast_shape(x.shape, y.shape)
    return x._broadcast_to(shape), y._broadcast_to(shape)
  def _broadcast_to(self, shape) -> "Tensor":
    if self.shape == tuple(shape): return self
    x = self.reshape((1,) * (len(shape) - self.ndim) + self.shape) if len(shape) > self.ndim else self
    return x.expand(shape)

  # ---- elementwise (binary / ternary); `x` may be a Tensor or python scalar
  def add(self, x, reverse: bool = False) -> "Tensor":
    a, b = self._broadcasted(x, reverse)
    return a.maximum(b) if a.dtype == dtypes.bool else F.Add.apply(a, b)
  def sub(self, x, reverse: bool = False) -> "Tensor":
    a, b = self._broadcasted(x, reverse)
    return F.Sub.apply(a, b)
  def mul(self, x, reverse: bool = False) -> "Tensor":
    a, b = self._broadcasted(x, reverse)
    return a.minimum(b) if a.dtype == dtypes.bool else F.Mul.apply(a, b)
  def div(self, x, reverse: bool = False) -> "Tensor":
    """True division; ints promote to default_float."""
    a, b = self._broadcasted(x, reverse)
    if not a.dtype.is_float: a, b = a._float(), b._float()
    return F.Div.apply(a, b)
  def pow(self, x, reverse: bool = False) -> "Tensor":
    if not reverse and not isinstance(x, Tensor) and isinstance(x, (int, float)) and float(x).is_integer() and abs(x) <= 4:
      e = int(x)
      if e == 0: return self.ones_like() if self.dtype.is_float or not isinstance(x, float) else self.ones_like(dtype=dtypes.default_float)
      base = self if (self.dtype.is_float or isinstance(x, int)) and e > 0 else self._float()
      r = base
      for _ in range(abs(e) - 1): r = r * base
      return r if e > 0 else r.reciprocal()
    b, e = self._broadcasted(x, reverse)
    if not b.dtype.is_float: b, e = b._float(), e._float()
    half = b.dtype if b.dtype in _HALF else None
    if half: b, e = b.float(), e.float()
    r = F.Pow.apply(b, e, _pow_value(b.detach(), e.detach()))  # explicit backward: torch's rules at base 0
    return r.cast(half) if half else r
  def maximum(self, x) -> "Tensor":
    a, b = self._broadcasted(x)
    return F.Max.apply(a, b)
  def minimum(self, x) -> "Tensor":
    a, b = self._broadcasted(x)
    if a.dtype.is_float: return -F.Max.apply(-a, -b)
    return (a < b).where(a, b)
  def clip(self, min_=None, max_=None, *, min=None, max=None) -> "Tensor":
    min_ = min if min_ is None else min_
    max_ = max if max_ is None else max_
    x = self
    # python-scalar bounds: like torch.clamp, x on a bound gets zero gradient and NaN passes through.
    # Tensor bounds use maximum/minimum (a tie splits the gradient).
    if min_ is not None:
      if isinstance(min_, Tensor): x = x.maximum(min_)
      else:
        a, m = x._broadcasted(min_)
        x = (a <= m).where(m, a)
    if max_ is not None:
      if isinstance(max_, Tensor): x = x.minimum(max_)
      else:
        a, m = x._broadcasted(max_)
        x = (a >= m).where(m, a)
    return x
  clamp = clip
  def where(self, x, y) -> "Tensor":
    """self is the condition."""
    cond = self if self.dtype == dtypes.bool else self.cast(dtypes.bool)
    if isinstance(x, Tensor) and isinstance(y, Tensor): dt = promote(x.dtype, y.dtype)
    elif isinstance(x, Tensor): dt = x._scalar_dtype(y)
    elif isinstance(y, Tensor): dt = y._scalar_dtype(x)
    else: dt = promote(dtypes.from_py(x), dtypes.from_py(y))
    x = x.cast(dt) if isinstance(x, Tensor) else Tensor._const((), x, dt, cond.device)
    y = y.cast(dt) if isinstance(y, Tensor) else Tensor._const((), y, dt, cond.device)
    shape = _broadcast_shape(_broadcast_shape(cond.shape, x.shape), y.shape)
    return F.Where.apply(cond._broadcast_to(shape), x._broadcast_to(shape), y._broadcast_to(shape))
  def masked_fill(self, mask: "Tensor", value) -> "Tensor": return mask.where(value, self)
  def __add__(self, x): return self.add(x)
  def __radd__(self, x): return self.add(x, True)
  def __sub__(self, x): return self.sub(x)
  def __rsub__(self, x): return self.sub(x, True)
  def __mul__(self, x): return self.mul(x)
  def __rmul__(self, x): return self.mul(x, True)
  def __truediv__(self, x): return self.div(x)
  def __rtruediv__(self, x): return self.div(x, True)
  def __floordiv__(self, x):   # ints only, C semantics
    a, b = self._broadcasted(x)
    if not a.dtype.is_int: raise TypeError(f"// is ints only (C semantics), got {a.dtype}")
    return F.IDiv.apply(a, b)
  def __rfloordiv__(self, x):
    a, b = self._broadcasted(x, True)
    if not a.dtype.is_int: raise TypeError(f"// is ints only (C semantics), got {a.dtype}")
    return F.IDiv.apply(a, b)
  def __mod__(self, x):        # ints only, C semantics
    a, b = self._broadcasted(x)
    if not a.dtype.is_int: raise TypeError(f"% is ints only (C semantics), got {a.dtype}")
    return F.Mod.apply(a, b)
  def __pow__(self, x): return self.pow(x)
  def __rpow__(self, x): return self.pow(x, True)
  def __neg__(self): return self.neg()
  def __matmul__(self, x): return self.matmul(x)
  def __rmatmul__(self, x): return Tensor(x, device=self.device).matmul(self) if not isinstance(x, Tensor) else x.matmul(self)
  def __lt__(self, x): return F.Less.apply(*self._broadcasted(x))
  def __gt__(self, x): return F.Less.apply(*self._broadcasted(x, True))
  def __le__(self, x):
    a, b = self._broadcasted(x)
    return F.Less.apply(a, b).maximum(F.Eq.apply(a, b))
  def __ge__(self, x):
    a, b = self._broadcasted(x)
    return F.Less.apply(b, a).maximum(F.Eq.apply(a, b))
  def __eq__(self, x): return F.Eq.apply(*self._broadcasted(x))         # elementwise; Tensor is therefore unhashable by value
  def __ne__(self, x): return F.Eq.apply(*self._broadcasted(x)).logical_not()
  __hash__ = object.__hash__
  def __bool__(self):
    if self.numel() != 1: raise RuntimeError("the truth value of a Tensor with more than one element is ambiguous")
    return bool(self.item())

  # ---- reductions; axis: int | tuple | None
  def _axes(self, axis) -> tuple[int, ...]:
    if axis is None: return tuple(range(self.ndim))
    if isinstance(axis, (int, np.integer)): axis = (int(axis),)
    return tuple(sorted({canonicalize_axis(a, self.ndim) for a in axis}))
  def _reduce(self, fxn, axis, keepdim: bool) -> "Tensor":
    axes = self._axes(axis)
    ret = fxn.apply(self, axis=axes) if axes else self
    if keepdim: return ret
    return ret.reshape(tuple(s for i, s in enumerate(self.shape) if i not in axes))
  def sum(self, axis=None, keepdim: bool = False, dtype: DType | None = None, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    if dtype is not None: x, out = self.cast(dtype), None
    elif self.dtype in _HALF: x, out = self.float(), self.dtype
    elif self.dtype == dtypes.bool or (self.dtype.is_int and self.dtype.itemsize < dtypes.default_int.itemsize):
      x, out = self.cast(dtypes.default_int), None
    else: x, out = self, None
    if x.dtype == dtypes.bool: x = x.cast(dtypes.default_int)
    ret = x._reduce(F.Sum, axis, keepdim)
    return ret.cast(out) if out is not None else ret
  def mean(self, axis=None, keepdim: bool = False, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    x = self._upcast()
    n = prod([self.shape[a] for a in self._axes(axis)])
    s = x.sum(axis, keepdim)
    if isinstance(n, Node): out = s / Tensor(n, device=s.device).cast(s.dtype)   # symbolic count: runtime divide
    else: out = s * (1.0 / n) if n else s * float("nan")
    return out.cast(self.dtype) if self.dtype in _HALF else out
  def max(self, axis=None, keepdim: bool = False, *, dim=None) -> "Tensor":
    return self._reduce(F.MaxReduce, _pick_axis(axis, dim), keepdim)
  def min(self, axis=None, keepdim: bool = False, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    if self.dtype == dtypes.bool: return self.logical_not().max(axis, keepdim).logical_not()
    if self.dtype.is_unsigned: return (255 - self).max(axis, keepdim).__rsub__(255)
    if self.dtype.is_int: return (-1 - self).max(axis, keepdim).__rsub__(-1)   # ~x = -1-x: order-reversing, no overflow at the dtype min
    return -((-self).max(axis, keepdim))
  def var(self, axis=None, keepdim: bool = False, correction: int = 1, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    x = self._upcast()
    n = prod([self.shape[a] for a in self._axes(axis)])
    d = x - x.mean(axis, keepdim=True)
    ss = (d * d).sum(axis, keepdim)
    if isinstance(n, Node): out = ss / Tensor(n - correction, device=ss.device).cast(ss.dtype)
    else: out = ss * (1.0 / (n - correction) if n - correction > 0 else float("nan"))
    return out.cast(self.dtype) if self.dtype in _HALF else out
  def std(self, axis=None, keepdim: bool = False, correction: int = 1, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    x = self._upcast()
    out = x.var(axis, keepdim, correction).sqrt()
    return out.cast(self.dtype) if self.dtype in _HALF else out
  def argmax(self, axis=None, keepdim: bool = False, *, dim=None) -> "Tensor":
    """First index of the max; int32."""
    axis = _pick_axis(axis, dim)
    if axis is None:
      ret = self.flatten().argmax(0)
      return ret.reshape((1,) * self.ndim) if keepdim else ret
    axis = canonicalize_axis(axis, self.ndim)
    n = self.shape[axis]
    x = self.detach()
    is_max = x == x.max(axis, keepdim=True)
    rev = Tensor.arange(n, 0, -1, device=self.device).reshape((n,) + (1,) * (self.ndim - axis - 1))
    idx = is_max.where(rev, 0).max(axis, keepdim)
    return (n - idx).cast(dtypes.int32)
  def argmin(self, axis=None, keepdim: bool = False, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    if self.dtype == dtypes.bool: return self.logical_not().argmax(axis, keepdim)
    if self.dtype.is_unsigned: return (255 - self).argmax(axis, keepdim)
    if self.dtype.is_int: return (-1 - self).argmax(axis, keepdim)
    return (-self).argmax(axis, keepdim)
  def logsumexp(self, axis=None, keepdim: bool = False, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    x = self._upcast()
    m = x.max(axis, keepdim=True).detach()
    m = (m.abs() == float("inf")).where(0.0, m)
    out = (x - m).exp().sum(axis, keepdim=True).log() + m
    if not keepdim: out = out.reshape(tuple(s for i, s in enumerate(self.shape) if i not in self._axes(axis)))
    return out.cast(self.dtype) if self.dtype in _HALF else out
  def softmax(self, axis: int = -1, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    x = self._upcast()
    e = (x - x.max(axis, keepdim=True).detach()).exp()
    out = e / e.sum(axis, keepdim=True)
    return out.cast(self.dtype) if self.dtype in _HALF else out
  def log_softmax(self, axis: int = -1, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    x = self._upcast()
    s = x - x.max(axis, keepdim=True).detach()
    out = s - s.exp().sum(axis, keepdim=True).log()
    return out.cast(self.dtype) if self.dtype in _HALF else out
  def cumsum(self, axis: int = 0, *, dim=None) -> "Tensor":
    axis = _pick_axis(axis, dim)
    if self.ndim == 0: return self
    axis = canonicalize_axis(axis, self.ndim)
    n = self.shape[axis]
    x = self.transpose(axis, -1) if axis != self.ndim - 1 else self
    e = x.unsqueeze(-2).expand(x.shape[:-1] + (n, n))
    i = Tensor.arange(n, device=self.device)
    mask = i.reshape(n, 1) >= i.reshape(1, n)           # row i keeps cols j <= i
    small = self.dtype == dtypes.bool or (self.dtype.is_int and self.dtype.itemsize < dtypes.default_int.itemsize)
    acc = dtypes.default_int if small else self.dtype   # like sum(): bool/small ints accumulate in default_int
    r = mask.where(e.cast(acc), Tensor._const((), 0, acc, self.device)).sum(-1)
    r = r.cast(acc) if r.dtype != acc else r
    return r.transpose(axis, -1) if axis != self.ndim - 1 else r

  # ---- linear algebra / nn functional
  def matmul(self, x: "Tensor") -> "Tensor":
    """Batched with broadcasting, like torch.matmul (1-D operands supported)."""
    if not isinstance(x, Tensor): x = Tensor(x, device=self.device)
    a, b = self, x
    assert a.ndim > 0 and b.ndim > 0, "matmul needs operands with ndim >= 1"
    dt = promote(a.dtype, b.dtype)
    a, b = a.cast(dt), b.cast(dt)
    acc = dtypes.float32 if dt in _HALF else (dtypes.default_int if dt == dtypes.bool else dt)
    a, b = a.cast(acc), b.cast(acc)
    sq_a, sq_b = a.ndim == 1, b.ndim == 1
    if sq_a: a = a.reshape(1, a.shape[0])
    if sq_b: b = b.reshape(b.shape[0], 1)
    assert a.shape[-1] == b.shape[-2], f"matmul shape mismatch {self.shape} @ {x.shape}"
    out = (a.unsqueeze(-1) * b.unsqueeze(-3)).sum(-2, dtype=acc)
    if sq_a: out = out.squeeze(-2)
    if sq_b: out = out.squeeze(-1)
    return out.cast(dt) if dt != acc else out
  def dot(self, x: "Tensor") -> "Tensor": return self.matmul(x)
  def linear(self, weight: "Tensor", bias: "Tensor | None" = None) -> "Tensor":
    """self @ weight.T + bias; weight is (out, in) like torch."""
    out = self.matmul(weight.transpose(-1, -2) if weight.ndim >= 2 else weight)
    return out + bias if bias is not None else out
  def layernorm(self, axis: int = -1, eps: float = 1e-5) -> "Tensor":
    """Normalization only (no affine)."""
    x = self._upcast()
    d = x - x.mean(axis, keepdim=True)
    out = d * ((d * d).mean(axis, keepdim=True) + eps).rsqrt()
    return out.cast(self.dtype) if self.dtype in _HALF else out
  def conv2d(self, weight: "Tensor", bias: "Tensor | None" = None, stride=1, padding=0, dilation=1, groups: int = 1) -> "Tensor":
    """== torch F.conv2d for groups=1: self (N, C, H, W), weight (O, C, kh, kw), bias (O,). stride / padding / dilation
    are ints or (h, w) pairs (padding: also "valid" / "same" for stride 1). Composed from the primitives (I82 §2.2):
    im2col as kh*kw shifted SHRINK views of one PAD, stacked to (C*kh*kw, Ho*Wo), then one matmul (weight on the left,
    so the output is NCHW with no transpose and the GEMM sees plain strided operands). A 1x1 stride-1 conv is the
    matmul alone. Batches run one sample per matmul (N > 1 would otherwise give the GEMM a two-stride N group). The input
    is made contiguous first (a no-op for a plain or computed buffer), so a view input (interpolate's expand, a batch
    slice) is materialized once and the im2col is provably of one dense X: the device conv kernel (schedule._match_conv)
    applies, and the GEMM operands stay plain strided."""
    if groups != 1: raise NotImplementedError("conv2d: groups != 1 is not supported")
    assert self.ndim == 4 and weight.ndim == 4, f"conv2d needs (N, C, H, W) and (O, C, kh, kw), got {self.shape} and {weight.shape}"
    N, C, H, W = self.shape
    O, Cw, kh, kw = weight.shape
    assert C == Cw, f"conv2d: input has {C} channels, weight expects {Cw}"
    pair = lambda v: (v, v) if isinstance(v, int) else (int(v[0]), int(v[1]))
    (sh, sw), (dh, dw) = pair(stride), pair(dilation)
    if padding == "valid": padding = 0
    if padding == "same":
      assert (sh, sw) == (1, 1), "conv2d: padding='same' needs stride 1"
      th, tw = dh * (kh - 1), dw * (kw - 1)   # torch puts the odd extra element after
      pads = ((th // 2, th - th // 2), (tw // 2, tw - tw // 2))
    else:
      ph, pw = pair(padding)
      pads = ((ph, ph), (pw, pw))
    Ho = (H + pads[0][0] + pads[0][1] - dh * (kh - 1) - 1) // sh + 1
    Wo = (W + pads[1][0] + pads[1][1] - dw * (kw - 1) - 1) // sw + 1
    assert Ho > 0 and Wo > 0, f"conv2d: output size {Ho}x{Wo} is empty"
    if N > 1:
      return Tensor.cat(*[self[n:n + 1].conv2d(weight, bias, stride, padding, dilation) for n in range(N)], axis=0)
    xp = self.contiguous().pad(((0, 0), (0, 0)) + pads)
    if (kh, kw, sh, sw) == (1, 1, 1, 1): col = xp.reshape(C, Ho * Wo)
    else:
      cols = []
      for i in range(kh):
        for j in range(kw):
          v = xp.shrink(((0, 1), (0, C), (i * dh, i * dh + sh * (Ho - 1) + 1), (j * dw, j * dw + sw * (Wo - 1) + 1)))
          if sh > 1 or sw > 1: v = v[:, :, ::sh, ::sw]
          cols.append(v.reshape(C, Ho * Wo))
      col = Tensor.stack(cols, axis=1).reshape(C * kh * kw, Ho * Wo)   # row c*kh*kw + i*kw + j, like weight.reshape(O, -1)
    out = (weight.reshape(O, C * kh * kw) @ col).reshape(1, O, Ho, Wo)
    return out + bias.reshape(1, O, 1, 1) if bias is not None else out
  def group_norm(self, num_groups: int, weight: "Tensor | None" = None, bias: "Tensor | None" = None, eps: float = 1e-5) -> "Tensor":
    """== torch F.group_norm: self (N, C, *), statistics over each group of C/num_groups channels and all spatial
    positions (biased variance), then the per-channel affine. Half inputs compute in float32 and round once."""
    N, C = self.shape[0], self.shape[1]
    assert C % num_groups == 0, f"group_norm: {C} channels do not split into {num_groups} groups"
    x = self._upcast().reshape(N, num_groups, -1)
    n = x.shape[-1]
    m = x._sum_2level() * (1.0 / n)
    d = x - m
    y = (d * ((d * d)._sum_2level() * (1.0 / n) + eps).rsqrt()).reshape(self.shape)
    aff = (1, C) + (1,) * (self.ndim - 2)
    if weight is not None: y = y * weight.cast(y.dtype).reshape(aff)
    if bias is not None: y = y + bias.cast(y.dtype).reshape(aff)
    return y.cast(self.dtype) if self.dtype in _HALF else y
  def _sum_2level(self) -> "Tensor":
    """Sum over the last axis, keepdim, as two reductions when it is long: chunks of f ~ sqrt(n) first, then the
    partial sums. One f32 reduction over a VAE GroupNorm group at 512^2 (2^20 elements) was 2.7e-4 off on METAL; two
    levels bring it near torch's pairwise sum (and give the device n/f-way parallelism)."""
    n = self.shape[-1]
    if not isinstance(n, int) or n <= 4096: return self.sum(-1, keepdim=True)
    f = next(d for d in range(math.isqrt(n), 0, -1) if n % d == 0)
    if f < 16: return self.sum(-1, keepdim=True)
    return self.reshape(self.shape[:-1] + (n // f, f)).sum(-1).sum(-1, keepdim=True)
  def interpolate(self, size=None, scale_factor=None, mode: str = "nearest") -> "Tensor":
    """== torch F.interpolate(mode='nearest') over the trailing spatial axes of (N, C, *spatial): source index
    floor(dst * in / out). Integer upsampling is a reshape + EXPAND (exact, no gather); integer downsampling a strided
    slice; any other size a host-built index gather. scale_factor: number or one per spatial axis (output size
    floor(in * scale), and the index scale from the sizes, i.e. torch's recompute_scale_factor=True)."""
    if mode != "nearest": raise NotImplementedError(f"interpolate: mode {mode!r} is not supported (only 'nearest')")
    sp = self.shape[2:]
    assert sp, "interpolate needs (N, C, *spatial)"
    if size is not None:
      out = (size,) * len(sp) if isinstance(size, int) else tuple(int(s) for s in size)
    else:
      assert scale_factor is not None, "interpolate needs size or scale_factor"
      sf = (scale_factor,) * len(sp) if isinstance(scale_factor, (int, float)) else tuple(scale_factor)
      out = tuple(int(math.floor(s * f)) for s, f in zip(sp, sf))
    assert len(out) == len(sp), f"interpolate: size {out} does not match spatial dims {sp}"
    x = self
    for k, (n, m) in enumerate(zip(sp, out)):
      ax = 2 + k
      if m == n: continue
      if m % n == 0:
        r = m // n
        x = x.reshape(x.shape[:ax + 1] + (1,) + x.shape[ax + 1:]).expand(x.shape[:ax + 1] + (r,) + x.shape[ax + 1:])
        x = x.reshape(x.shape[:ax] + (m,) + x.shape[ax + 2:])
      elif n % m == 0: x = x._slice_axis(ax, slice(0, n, n // m))
      else:
        idx = np.minimum(np.floor(np.arange(m, dtype=np.float64) * (n / m)).astype(np.int64), n - 1).astype(np.int32)
        x = x._index_select(ax, Tensor(idx, device=x.device))
    return x
  def dropout(self, p: float = 0.5) -> "Tensor":
    """Identity unless Tensor.training."""
    if not Tensor.training or p == 0: return self
    if p >= 1: return self * 0
    keep = Tensor.rand(*self.shape, device=self.device) >= p
    return self * keep.where(1.0 / (1.0 - p), 0.0).cast(self.dtype if self.dtype.is_float else dtypes.default_float)
  def one_hot(self, num_classes: int) -> "Tensor":
    return (self.unsqueeze(-1) == Tensor.arange(num_classes, device=self.device)).cast(dtypes.default_int)
  def _tri_mask(self, diagonal: int, upper: bool) -> "Tensor":
    r, c = self.shape[-2], self.shape[-1]
    i = Tensor.arange(r, device=self.device).reshape(r, 1)
    j = Tensor.arange(c, device=self.device).reshape(1, c)
    return (j - i >= diagonal) if upper else (j - i <= diagonal)
  def triu(self, diagonal: int = 0) -> "Tensor": return self._tri_mask(diagonal, True).where(self, Tensor._const((), 0, self.dtype, self.device))
  def tril(self, diagonal: int = 0) -> "Tensor": return self._tri_mask(diagonal, False).where(self, Tensor._const((), 0, self.dtype, self.device))
  def cross_entropy(self, target: "Tensor", ignore_index: int = -1, reduction: str = "mean",
                    label_smoothing: float = 0.0) -> "Tensor":
    """self: logits (N, C) or (C,); target: int class indices. == torch F.cross_entropy, except
    ignore_index defaults to -1 here (torch's default is -100)."""
    assert reduction in ("mean", "sum", "none"), f"bad reduction {reduction}"
    c = self.shape[-1]
    lsm = self.log_softmax(-1)
    y = (target.unsqueeze(-1) == Tensor.arange(c, device=self.device))
    nll = -y.where(lsm, 0.0).sum(-1)
    loss = nll * (1 - label_smoothing) - lsm.sum(-1) * (label_smoothing / c) if label_smoothing else nll
    valid = target != ignore_index
    loss = valid.where(loss, 0.0)
    if reduction == "none": return loss
    if reduction == "sum": return loss.sum()
    return loss.sum() / valid.sum().cast(loss.dtype)
  def scaled_dot_product_attention(self, key: "Tensor", value: "Tensor", attn_mask: "Tensor | None" = None,
                                   dropout_p: float = 0.0, is_causal: bool = False) -> "Tensor":
    """self is the query (..., T, D). == torch F.scaled_dot_product_attention."""
    q = self
    scale = 1.0 / math.sqrt(q.shape[-1])
    qk = (q @ key.transpose(-1, -2)) * scale
    L, S = qk.shape[-2], qk.shape[-1]
    if is_causal: qk = Tensor.ones(L, S, dtype=dtypes.bool, device=q.device).tril(0).where(qk, float("-inf"))
    if attn_mask is not None:
      qk = attn_mask.where(qk, float("-inf")) if attn_mask.dtype == dtypes.bool else qk + attn_mask
    attn = qk.softmax(-1)
    if dropout_p: attn = attn.dropout(dropout_p)
    return attn @ value

  # ---- autograd
  def _deepwalk(self) -> list["Tensor"]:
    """Tensors with a _ctx reachable from self, in topological order (parents first). Iterative."""
    order, visited = [], set()
    stack: list[tuple[Tensor, bool]] = [(self, False)]
    while stack:
      t, done = stack.pop()
      if done:
        order.append(t)
        continue
      if id(t) in visited: continue
      visited.add(id(t))
      if t._ctx is None: continue
      stack.append((t, True))
      for p in t._ctx.parents:
        if id(p) not in visited: stack.append((p, False))
    return order
  def backward(self, gradient: "Tensor | None" = None) -> "Tensor":
    """Scalar self unless gradient is given. Accumulates into .grad of leaves with requires_grad."""
    if gradient is None:
      if self.numel() != 1: raise RuntimeError(f"backward() without a gradient needs a scalar output, shape is {self.shape}")
      g = Tensor._const(self.shape, 1.0, self.dtype if self.dtype.is_float else dtypes.default_float, self.device).lazydata
    else:
      if not isinstance(gradient, Tensor): gradient = Tensor(gradient, device=self.device)
      assert tuple(gradient.shape) == tuple(self.shape), f"gradient shape {gradient.shape} != {self.shape}"
      g = gradient.cast(self.dtype).lazydata if self.dtype.is_float else gradient.lazydata
    if self._ctx is None:
      if self.requires_grad: self._accum(g)
      return self
    grads: dict[int, LazyBuffer] = {id(self): g}
    for t in reversed(self._deepwalk()):
      tg = grads.pop(id(t), None)
      if tg is None: continue
      ctx = t._ctx
      gs = ctx.backward(tg)
      if not isinstance(gs, tuple): gs = (gs,)
      for p, pg, need in zip(ctx.parents, gs, ctx.needs_input_grad):
        if pg is None or not need: continue
        assert tuple(pg.shape) == tuple(p.shape), f"grad shape {pg.shape} != {p.shape} for {type(ctx).__name__}"
        if p._ctx is None:
          if p.requires_grad: p._accum(pg)
        else: grads[id(p)] = pg if id(p) not in grads else grads[id(p)].e(_ADD, pg)
    return self
  def _accum(self, g: LazyBuffer):
    if g.dtype != self.dtype and self.dtype.is_float: g = g.cast(self.dtype)
    self.grad = Tensor(g, requires_grad=False) if self.grad is None else Tensor(self.grad.lazydata.e(_ADD, g), requires_grad=False)
  def requires_grad_(self, flag: bool = True) -> "Tensor":
    self.requires_grad = flag
    return self

  @staticmethod
  def inference() -> "no_grad": return no_grad()

  def __repr__(self): return f"<Tensor {getattr(self, 'lazydata', None)!r}>"

from fusezap.ops import BinaryOps as _BO  # noqa: E402
_ADD = _BO.ADD

def _broadcast_shape(a: tuple, b: tuple) -> tuple:
  n = max(len(a), len(b))
  a, b = (1,) * (n - len(a)) + tuple(a), (1,) * (n - len(b)) + tuple(b)
  out = []
  for x, y in zip(a, b):
    if x == y or y == 1: out.append(x)
    elif x == 1: out.append(y)
    else: raise ValueError(f"shapes {a} and {b} cannot be broadcast")
  return tuple(out)

def _index_array(i, device) -> Tensor:
  """A list/ndarray index -> int Tensor. Boolean masks are rejected (torch gives mask semantics, not 0/1 positions)."""
  a = np.asarray(i)
  if a.dtype.kind == "b": raise NotImplementedError("boolean mask indexing is not supported")
  if a.size and a.dtype.kind not in "iu": raise IndexError(f"index arrays must be integers, got {a.dtype}")
  return Tensor(a.astype(np.int32), device=device)

def _pow_value(b: Tensor, e: Tensor) -> Tensor:
  """b ** e for float tensors of one shape (no autograd). Negative base: integer exponents keep a sign, others give NaN."""
  r = (e * b.abs().log2()).exp2()
  e_int = e == e.cast(dtypes.int32).cast(e.dtype)
  odd = (e.abs().cast(dtypes.int32) % 2) == 1
  neg_r = e_int.where(odd.where(-r, r), float("nan"))
  r = (b < 0).where(neg_r, r)
  return (e == 0).where(1.0, r)

def manual_seed(seed: int) -> None:
  _Rng.gen = np.random.Generator(np.random.Philox(seed))

class no_grad:
  """Context manager: disables graph recording for autograd."""
  def __enter__(self):
    self.prev, Tensor.no_grad = Tensor.no_grad, True
    return self
  def __exit__(self, *a): Tensor.no_grad = self.prev
