"""NPY reference device (SPEC §2). Implemented by WP3.

Executes a ScheduleItem's kernel AST directly with numpy (no codegen), so the scheduler and AST can
be tested independently of WP4/WP5. Buffers are numpy arrays (LRUAllocator). Must honor
ShapeTracker index semantics exactly as codegen would: use st.expr_idxs / expr_node evaluated with numpy.

Evaluation: nodes under the REDUCE run over the full shape F; the reduce and everything after it (the epilogue) run
over the STORE shape R, reading every epilogue LOAD/CONST with its reduced axes pinned to index 0 (they have
stride 0 there by the AST convention). Masked lanes of a LOAD or CONST read 0. Elementwise and reduce numerics
are the interpreter's (fusezap.interpret), so NPY == interp isolates scheduler/AST bugs.

Custom kernels (SPEC §11.1): `custom_kernels` offers the numpy reference matmul/attention of fusezap.schedule, which
repeat the composed numerics; kind="custom" items go to fusezap.schedule.run_custom.
"""
from __future__ import annotations
import functools, time
from typing import Any
import numpy as np
from fusezap.device import DeviceBase, LRUAllocator
from fusezap.dtype import to_np
from fusezap.helpers import prod
from fusezap.ir import LazyOp, ScheduleItem
from fusezap.ops import BufferOps, ReduceOps
from fusezap.shape.symbolic import Node, Var, sym_infer
from fusezap.shape.view import ShapeTracker

class NPYAllocator(LRUAllocator):
  """Handles are flat np.uint8 arrays."""
  def _alloc(self, nbytes: int) -> Any: return np.empty(nbytes, np.uint8)
  def copyin(self, handle: Any, src: memoryview) -> None:
    handle[:src.nbytes] = np.frombuffer(src.cast("B") if src.format != "B" or src.ndim != 1 else src, np.uint8)
  def copyout(self, dst: memoryview, handle: Any) -> None:
    np.frombuffer(dst.cast("B") if dst.format != "B" or dst.ndim != 1 else dst, np.uint8)[:] = handle[:dst.nbytes]

def _typed(buf) -> np.ndarray:
  if buf.size == 0: return np.zeros(0, to_np(buf.dtype))
  return buf.handle[:buf.nbytes].view(to_np(buf.dtype))

@functools.lru_cache(maxsize=4096)
def _index(st: ShapeTracker, shape: tuple[int, ...], pin: tuple[bool, ...]) -> tuple[np.ndarray, np.ndarray | None]:
  """(buffer index, valid or None) over the grid `shape`, with pinned axes held at 0 (their grid size is 1)."""
  out = tuple(1 if p else s for s, p in zip(shape, pin))
  idxs: list[Any] = [0 if (p or s == 1) else Var(f"_npy{k}", 0, s - 1) for k, (s, p) in enumerate(zip(shape, pin))]
  idx, valid = st.expr_idxs(idxs)
  vals = {v: np.arange(s, dtype=np.int64).reshape([-1 if j == k else 1 for j in range(len(shape))])
          for k, (v, s) in enumerate(zip(idxs, shape)) if isinstance(v, Var)}
  ev = lambda x: x.substitute(vals) if isinstance(x, Node) else x
  idx_a = np.broadcast_to(np.asarray(ev(idx), np.int64), out)
  if isinstance(valid, int) and valid == 1: return idx_a, None
  return idx_a, np.broadcast_to(np.asarray(ev(valid)).astype(bool), out)

class _Run:
  def __init__(self, si: ScheduleItem):
    from fusezap.interpret import from_host
    self.si, self.vv = si, si.var_vals
    self.flat = [from_host(_typed(b), b.dtype) for b in si.bufs]
    self.memo: dict = {}

  def st(self, st: ShapeTracker) -> ShapeTracker: return st.subs(self.vv) if st.vars() else st
  def shape(self, s) -> tuple[int, ...]: return tuple(int(sym_infer(x, self.vv)) for x in s)

  def leaf(self, n: LazyOp, full: tuple[int, ...], pin: tuple[bool, ...]) -> np.ndarray:
    from fusezap.interpret import _fix, _vdtype
    out = tuple(1 if p else s for s, p in zip(full, pin))
    st = self.st(n.arg.st)
    assert self.shape(st.shape) == full, f"NPY: leaf st shape {st.shape} != kernel shape {full}"
    if prod(out) == 0: return np.zeros(out, _vdtype(n.dtype))
    idx, valid = _index(st, full, pin)
    if n.op is BufferOps.CONST:
      v = n.arg.val
      if isinstance(v, Node): v = sym_infer(v, self.vv)
      a = _fix(np.full(out, v), n.dtype)
    else:
      flat = self.flat[n.arg.idx]
      if valid is None: return flat[idx]
      a = flat[np.clip(idx, 0, max(flat.size - 1, 0))] if flat.size else np.zeros(out, flat.dtype)
    return a if valid is None else np.where(valid, a, np.zeros((), a.dtype))

  def eval(self, n: LazyOp, full: tuple[int, ...], pin: tuple[bool, ...]) -> np.ndarray:
    from fusezap.interpret import _elementwise, _reduce
    key = (id(n), pin)
    if key in self.memo: return self.memo[key]
    if n.op in (BufferOps.LOAD, BufferOps.CONST): r = self.leaf(n, full, pin)
    elif isinstance(n.op, ReduceOps):
      x = self.eval(n.src[0], full, (False,) * len(full))
      r = _reduce(n.op, x, n.src[0].dtype, n.arg)
    else:
      out = tuple(1 if p else s for s, p in zip(full, pin))
      xs = [np.broadcast_to(self.eval(s, full, pin), out) for s in n.src]
      r = _elementwise(n.op, xs, [s.dtype for s in n.src], n.dtype, n.arg)
    self.memo[key] = r
    return r

  def store(self, root: LazyOp) -> None:
    from fusezap.interpret import to_host
    assert root.op is BufferOps.STORE
    ost = self.st(root.arg.st)
    R = self.shape(ost.shape)
    full = self._full_shape(root) or R
    assert len(full) == len(R), f"NPY: rank mismatch {full} vs {R}"
    pin = tuple(r == 1 and f != 1 for r, f in zip(R, full))
    val = np.broadcast_to(self.eval(root.src[0], full, pin), R)
    buf = self.si.bufs[root.arg.idx]
    dst = _typed(buf)
    if prod(R) == 0: return
    host = to_host(np.ascontiguousarray(val).reshape(-1), root.arg.dtype)
    idx, valid = _index(ost, R, (False,) * len(R))
    idx, host = idx.reshape(-1), host.reshape(-1)
    if valid is not None: idx, host = idx[valid.reshape(-1)], host[valid.reshape(-1)]
    dst[idx] = host

  def _full_shape(self, root: LazyOp) -> tuple[int, ...] | None:
    for n in root.walk():
      if n.op in (BufferOps.LOAD, BufferOps.CONST): return self.shape(n.arg.st.shape)
    return None

class NPYDevice(DeviceBase):
  def __init__(self, name: str):
    self.name = name
    self.allocator = NPYAllocator()

  @property
  def custom_kernels(self) -> dict:
    from fusezap.schedule import REFERENCE_KERNELS
    return REFERENCE_KERNELS

  def run(self, si: ScheduleItem) -> float | None:
    if si.kind == "custom":
      from fusezap.schedule import run_custom
      return run_custom(self, si)
    assert si.kind == "kernel" and si.ast is not None, f"NPY runs kernels only, got {si.kind}"
    st = time.perf_counter()
    r = _Run(si)
    for root in si.ast: r.store(root)
    return time.perf_counter() - st
