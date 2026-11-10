"""Kernel replay (SPEC §11.1). Implemented by P2-JIT.

  @jit
  def step(tok: Tensor, kc: Tensor, pos: Var) -> Tensor: ...
  step(tok, kc, bind(pos, 5))         # or: step(tok, kc, pos, var_vals={pos: 5})

Call 1 runs `fn` eagerly (weights and other lazy state get realized). Call 2 runs `fn` again while recording every
ScheduleItem that executes. Calls 3+ do not run `fn` at all: they replay the recorded kernels with the new input
buffers and the new symbolic var values, skipping graph building, scheduling and lowering.

Contract (as in tinygrad's TinyJit):
  - Tensor args (positional or keyword, also inside lists/tuples/dicts) are the inputs. They are realized before the
    call. Their shapes and dtypes must match the captured call; anything else must change through a symbolic Var.
  - Symbolic values are passed as `bind(var, val)` args (fn receives the Var) and/or `var_vals={var: val}`. A plain Var
    arg that an enclosing `bound`/jit call binds takes that value.
  - Tensors reached any other way (closures, model weights) are constants: replay keeps reading the buffers that were
    used during capture. Rebinding them (e.g. `w.assign(...)`) after capture is not seen by the replay.
  - State must flow through args and return values; side effects of `fn` (python mutations) do not replay.
  - The return value may be a Tensor or a (nested) list/tuple/dict of Tensors; each output gets a fresh buffer per call.
    Non-Tensor values in it are returned as captured.
  - `.numpy()`/`.item()` inside `fn` bake the captured value into the replay.
  - Nesting: a jitted fn called while another jit captures runs `fn` inline (no replay), so the outer capture records
    its kernels. Vars it binds that the outer jit does not receive are baked at their captured value.
  - If a symbolic buffer is empty at the capture call's var values (its kernels are skipped), that capture is dropped
    and the next call captures again.

Device graphs: when a capture has no symbolic vars, no copy items and every item on one device that offers
`capture_graph` (CUDA), replays launch it as one device graph (FZ_GRAPH=0 turns this off; FZ_DEBUG >= 1 too, since it
times launches). Graph nodes hold fixed addresses, so read-only inputs are copied into staging buffers (device to
device) and each output is copied out of a persistent buffer into its fresh one. Inputs the replay writes (in-place
state such as a KV pool) are used directly; if one arrives as a different buffer, the graph is rebuilt. Replay 1 runs
the items on the staging buffers (binding launchers to those addresses), replay 2 captures, and later replays only
launch the graph. If a capture fails, the jit falls back to launch-by-launch replay.

`bound(var_vals)` is a context manager that lets eager (non-jit) code realize graphs with symbolic shapes:
realize() inside it schedules with those var values.
"""
from __future__ import annotations
import contextlib
from typing import Any, Callable
import numpy as np
from fusezap.device import Buffer, Device, CompiledDevice
from fusezap.dtype import dtypes
from fusezap.helpers import getenv, prod
from fusezap.ir import ScheduleItem
from fusezap.lazy import LazyBuffer
from fusezap.ops import LoadOps
from fusezap.shape.symbolic import Node, Var, sym_infer
from fusezap.tensor import Tensor
import fusezap.realize as _realize_mod

class BoundVar:
  """A Var with a value for one call. The jitted fn receives the Var itself."""
  __slots__ = ("var", "val")
  def __init__(self, var: Var, val: int):
    assert isinstance(var, Var), f"bind needs a Var, got {var!r}"
    val = int(val)
    assert var.vmin <= val <= var.vmax, f"{var.name}={val} is outside [{var.vmin}, {var.vmax}]"
    self.var, self.val = var, val
  def __repr__(self): return f"BoundVar({self.var.name}={self.val})"

def bind(var: Var, val: int) -> BoundVar: return BoundVar(var, val)

# ---------------------------------------------------------------- realize hooks

_orig_realize = _realize_mod.realize
_state: dict[str, Any] = {"vv": None, "rec": None, "sizes": None}

def _hooked_realize(lbs: list[LazyBuffer]) -> None:
  vv, rec, sizes = _state["vv"], _state["rec"], _state["sizes"]
  if getenv("FZ_INTERP"):
    assert not vv, "symbolic var values are not supported with FZ_INTERP"
    return _orig_realize(lbs)
  from fusezap.schedule import create_schedule
  bases = _unrealized_bases(lbs) if sizes is not None else []
  sched = create_schedule(lbs, dict(vv or {}))
  if sizes is not None:
    sizes.setdefault("vv", {}).update(vv or {})
    for b in bases:
      if (rb := b.base.realized) is None: continue
      sizes[id(rb)] = sz = prod(b.shape)
      if rb.size == 0 and isinstance(sz, Node): sizes["empty"] = True  # its kernels were skipped: capture is incomplete
    for si in sched:  # a COPY of a view copies from a contiguous source the scheduler made: not in the walked graph
      if si.kind == "copy" and id(si.bufs[1]) not in sizes and id(si.bufs[0]) in sizes: sizes[id(si.bufs[1])] = sizes[id(si.bufs[0])]
  if rec is not None: rec.extend(sched)
  _realize_mod.run_schedule(sched)

def _unrealized_bases(lbs: list[LazyBuffer]) -> list[LazyBuffer]:
  out, seen, stack = [], set(), [lb.base for lb in lbs]
  while stack:
    b = stack.pop()
    if id(b) in seen or b.realized is not None: continue
    seen.add(id(b))
    out.append(b)
    stack.extend(s.base for s in b.srcs)
  return out

def current_var_vals() -> dict[Var, int]:
  """The symbolic values bound right now (by an enclosing fz.jit call or `bound`)."""
  return dict(_state["vv"] or {})

def run_item(si: ScheduleItem) -> float | None:
  """Public hook for out-of-tree kernels: run ScheduleItem `si` now and, if a fz.jit capture is recording, record it so
  every replay launches it too. A kind="custom" item whose ast[0] has `launch(dev, bufs, var_vals)` (an external call)
  is launched through that method, at capture and at replay, so no device needs patching."""
  if (rec := _state["rec"]) is not None: rec.append(si)
  return _run(Device[si.bufs[0].device], si)

def _run(dev: Any, si: ScheduleItem) -> float | None:
  if si.kind == "custom" and si.ast and hasattr(a := si.ast[0], "launch"): return a.launch(dev, si.bufs, si.var_vals)
  return dev.run(si)

@contextlib.contextmanager
def bound(var_vals: dict[Var, int], _rec: list | None = None, _sizes: dict | None = None):
  """Realize symbolic graphs eagerly with these var values (and optionally record the ScheduleItems that run).
  The lazy graph refers to the Var, not to a value: realize everything a step built before leaving its `bound` block,
  or a later realize evaluates it with the later value."""
  prev = dict(_state), _realize_mod.realize
  _state.update(vv={**(_state["vv"] or {}), **var_vals}, rec=_rec if _rec is not None else _state["rec"],
                sizes=_sizes if _sizes is not None else _state["sizes"])
  _realize_mod.realize = _hooked_realize
  try: yield
  finally:
    _state.update(prev[0])
    _realize_mod.realize = prev[1]

# ---------------------------------------------------------------- arg / output trees

def _flatten(x: Any, tensors: list, vv: dict) -> Any:
  """Collect the Tensors (inputs) and replace BoundVars by their Var (recording the value); returns the fn-facing args."""
  if isinstance(x, Tensor):
    tensors.append(x)
    return x
  if isinstance(x, BoundVar):
    vv[x.var] = x.val
    return x.var
  if isinstance(x, Var) and x in (cur := _state["vv"] or {}): vv.setdefault(x, cur[x])  # a Var bound by an enclosing call
  if isinstance(x, (list, tuple)): return type(x)(_flatten(a, tensors, vv) for a in x)
  if isinstance(x, dict): return {k: _flatten(v, tensors, vv) for k, v in x.items()}
  return x

def _out_tensors(x: Any, out: list) -> None:
  if isinstance(x, Tensor): out.append(x)
  elif isinstance(x, (list, tuple)):
    for a in x: _out_tensors(a, out)
  elif isinstance(x, dict):
    for a in x.values(): _out_tensors(a, out)

class _Slot: pass
_SLOT = _Slot()

def _wrap(buf: Buffer, shape: tuple, vv: dict[Var, int]) -> Tensor:
  return Tensor(LazyBuffer.from_buffer(buf, tuple(int(sym_infer(s, vv)) if isinstance(s, Node) else s for s in shape)))

def _rebuild(x: Any, it) -> Any:
  if isinstance(x, (Tensor, _Slot)): return next(it)
  if isinstance(x, (list, tuple)): return type(x)(_rebuild(a, it) for a in x)
  if isinstance(x, dict): return {k: _rebuild(v, it) for k, v in x.items()}
  return x

def _as_realized_base(t: Tensor) -> Buffer:
  """Make `t` a realized contiguous base (same values) and return its Buffer."""
  lb = t.lazydata
  if not (lb.base is lb and lb.op is not LoadOps.CONST and lb.realized is not None):
    if not (lb.base is lb and lb.op is not LoadOps.CONST): lb = lb.contiguous()
    t.lazydata = lb
    _realize_mod.realize([lb])
  return lb.realized

# ---------------------------------------------------------------- replay

class _Item:
  """One recorded ScheduleItem, prepared for replay. slots: per buffer, (kind, ref) with kind 0=fixed handle owner
  (a Buffer), 1=input index, 2=owned index."""
  __slots__ = ("si", "slots", "prg", "gs", "ls", "vars", "dev", "kind", "nb", "static", "const", "done", "nbytes", "nout", "fast")

def _programs() -> dict[tuple, tuple]:
  """(ast key, renderer type, device name) -> (ProgramSpec, Program) from the lowering cache. run_compiled's key may
  carry more fields (e.g. opt settings); any cached program for the AST is correct, the most recent one wins."""
  from fusezap.codegen import lower
  out: dict[tuple, tuple] = {}
  for k, ent in list(getattr(lower, "_cache", {}).items()):
    if isinstance(k, tuple) and len(k) >= 3: out[k[:3]] = ent
  return out

class _Captured:
  def __init__(self, sched: list[ScheduleItem], in_bufs: list[Buffer], out_bufs: list[Buffer], sizes: dict[int, Any],
               vars_: tuple[Var, ...], in_meta: list, out_shapes: list, ret_tree: Any, plan: bool = False):
    from fusezap.codegen.lower import _ast_key
    self.vars, self.in_meta, self.out_shapes, self.ret_tree = vars_, in_meta, out_shapes, ret_tree
    # vars bound during capture but not passed to this jit (e.g. by a nested jit's bind): baked at the captured value
    self.baked = {v: x for v, x in sizes.get("vv", {}).items() if v not in vars_}
    in_idx = {id(b): i for i, b in enumerate(in_bufs)}
    owned: dict[int, int] = {}       # id(Buffer written by a recorded item) -> owned slot
    self.owned_proto: list[Buffer] = []
    for si in sched:
      for b in si.bufs[:max(si.n_outputs, 1)]:
        if id(b) not in in_idx and id(b) not in owned:
          owned[id(b)] = len(self.owned_proto)
          self.owned_proto.append(b)
    self.n_owned = len(self.owned_proto)
    # outputs: (kind, ref) like slots
    def ref(b: Buffer):
      if id(b) in in_idx: return (1, in_idx[id(b)])
      if id(b) in owned: return (2, owned[id(b)])
      return (0, b)
    self.out_refs = [ref(b) for b in out_bufs]
    out_owned = {r[1] for r in self.out_refs if r[0] == 2}
    # owned buffers: outputs get a fresh buffer each call (exact size); intermediates are persistent (max size)
    self.owned_size: list[Any] = []
    for i, b in enumerate(self.owned_proto):
      sz = sizes.get(id(b), b.size)
      self.owned_size.append(sz if i in out_owned else (sz.vmax if isinstance(sz, Node) else int(sz)))
    progs = _programs()
    self.items: list[_Item] = []
    const: set[int] = set()   # owned slots whose content never changes between replays (computed from constants only)
    for si in sched:
      it = _Item()
      it.si, it.kind, it.slots, it.prg, it.dev, it.done = si, si.kind, tuple(ref(b) for b in si.bufs), None, Device[si.bufs[0].device], False
      it.fast = None   # a device's own replay launcher for the item (NEURON, N7: NEURONDevice.replay_launcher)
      it.nout = max(si.n_outputs, 1)
      if si.kind == "copy":  # bytes to move: the current logical size (persistent buffers are allocated at vmax)
        it.nbytes, it.si = sizes.get(id(si.bufs[0]), si.bufs[0].size) * si.bufs[0].dtype.itemsize, None
      if si.kind == "kernel" and isinstance(it.dev, CompiledDevice) and (ent := progs.get((_ast_key(si.ast), type(it.dev.renderer), it.dev.name))):
        spec, it.prg = ent
        it.nb, it.ls, it.vars, it.gs = spec.n_bufs, spec.local_size, tuple(spec.vars), tuple(spec.global_size)
        it.static = not any(isinstance(s, Node) for s in it.gs)
        it.si = None  # do not keep the capture's buffers alive
      elif it.si is not None and plan: it.si = ScheduleItem(si.ast, (), si.var_vals, si.kind, si.n_outputs)  # nor here
      if it.prg is None and si.kind != "copy" and (rl := getattr(it.dev, "replay_launcher", None)) is not None: it.fast = rl(si)
      outs_, ins_ = it.slots[:max(si.n_outputs, 1)], it.slots[max(si.n_outputs, 1):]
      # constant item: no var values, every input fixed or constant, every output a persistent intermediate
      it.const = not si.var_vals and all(r[0] == 0 or (r[0] == 2 and r[1] in const) for r in ins_) and \
                 all(r[0] == 2 and r[1] not in out_owned for r in outs_)
      if it.const: const.update(r[1] for r in outs_)
      self.items.append(it)
    self.persist: list[Buffer | None] = self._plan(out_owned, const) if plan else \
      [None if i in out_owned else Buffer(b.device, self.owned_size[i], b.dtype).ensure_allocated() for i, b in enumerate(self.owned_proto)]
    self.proto_meta = [(b.device, b.dtype) for b in self.owned_proto]
    del self.owned_proto
    devs = {id(it.dev): it.dev for it in self.items}
    self.gdev = next(iter(devs.values())) if len(devs) == 1 else None
    self.graph_ok = bool(getenv("FZ_GRAPH", 1)) and getenv("FZ_DEBUG", 0) < 1 and not vars_ and self.items and \
                    hasattr(self.gdev, "capture_graph") and all(it.kind != "copy" for it in self.items) and \
                    all(Device[b.device] is self.gdev for b in in_bufs)
    self.written_in = sorted({r[1] for it in self.items for r in it.slots[:it.nout] if r[0] == 1})
    self.graph, self.gstage, self.gowned, self.gwritten, self.gwarm = None, None, None, None, False
    self.alias: tuple[int, ...] = ()   # outputs a graph replay returns as its persistent buffer (Jit alias=)
    if getenv("FZ_DEBUG", 0) >= 1:
      print(f"jit: captured {len(self.items)} items, {sum(it.const for it in self.items)} constant (run once), "
            f"{sum(it.prg is None and it.kind == 'kernel' for it in self.items)} kernels without a cached program")

  def _plan(self, out_owned: set[int], const: set[int]) -> list[Buffer | None]:
    """Persistent intermediates that share memory: one replay runs its items in order on one queue, so a buffer is
    live from the item that first writes it to the last item that touches it, and buffers whose live ranges do not
    overlap share one allocation (greedy best fit by bytes, per device). Outputs, constant items' results (written
    once, read by every replay) and buffers read before their first write keep their own allocation."""
    first: dict[int, int] = {}
    last: dict[int, int] = {}
    own = {i for i in range(len(self.owned_proto)) if i not in out_owned and i not in const}
    for t, it in enumerate(self.items):
      for k, r in enumerate(it.slots):
        if r[0] != 2 or r[1] not in own: continue
        if r[1] not in first:
          if k >= it.nout: own.discard(r[1]); continue   # read before written: state carried between replays
          first[r[1]] = t
        last[r[1]] = t
    arenas: list[list] = []   # [device, bytes, free from item index]
    where: dict[int, int] = {}
    # largest first among buffers first written by the same item (N8): smaller ones taken first used to grab the big arenas
    # (a fused 7B layer + head: every layer added a new 19 MB gate_up arena)
    for i in sorted((i for i in own if i in first), key=lambda i: (first[i], -self.owned_size[i] * self.owned_proto[i].dtype.itemsize)):
      b, need = self.owned_proto[i], self.owned_size[i] * self.owned_proto[i].dtype.itemsize
      free = [a for a, (d, _, t) in enumerate(arenas) if d == b.device and t < first[i]]
      fit = [a for a in free if arenas[a][1] >= need]
      a = min(fit, key=lambda a: arenas[a][1]) if fit else (max(free, key=lambda a: arenas[a][1]) if free else None)
      if a is None: a = len(arenas); arenas.append([b.device, 0, 0])
      arenas[a][1], arenas[a][2], where[i] = max(arenas[a][1], need), last[i], a
    self.arenas = [Buffer(d, max(n, 1), dtypes.uint8).ensure_allocated() for d, n, _ in arenas]
    out: list[Buffer | None] = []
    for i, b in enumerate(self.owned_proto):
      if i in out_owned: out.append(None); continue
      nb = Buffer(b.device, self.owned_size[i], b.dtype)
      if i in where: nb._handle = self.arenas[where[i]].handle   # a view: the arena owns (and frees) the memory
      out.append(nb.ensure_allocated())
    return out

  def _fresh(self, i: int, vv: dict[Var, int]) -> Buffer:
    sz, (dev, dt) = self.owned_size[i], self.proto_meta[i]
    return Buffer(dev, int(sym_infer(sz, vv)) if isinstance(sz, Node) else int(sz), dt).ensure_allocated()

  def __call__(self, in_bufs: list[Buffer], vv: dict[Var, int]) -> list[Buffer]:
    if self.graph_ok and (outs := self._graph_call(in_bufs, vv)) is not None: return outs
    owned: list[Buffer] = [b if b is not None else self._fresh(i, vv) for i, b in enumerate(self.persist)]
    self._launch(in_bufs, owned, vv)
    return [self._res(r, in_bufs, owned) for r in self.out_refs]

  @staticmethod
  def _res(r: tuple, in_bufs: list[Buffer], owned: list[Buffer]) -> Buffer: return r[1] if r[0] == 0 else (in_bufs[r[1]] if r[0] == 1 else owned[r[1]])

  def _graph_call(self, in_bufs: list[Buffer], vv: dict[Var, int]) -> list[Buffer] | None:
    """Replay as one device graph (see the module docstring); None: replay launch by launch instead."""
    dev = self.gdev
    if self.gstage is None:   # staging copies of the read-only inputs, persistent buffers for the outputs
      self.gstage = [None if i in self.written_in else Buffer(b.device, b.size, b.dtype).ensure_allocated() for i, b in enumerate(in_bufs)]
      self.gowned = [b if b is not None else self._fresh(i, vv) for i, b in enumerate(self.persist)]
    if (w := [in_bufs[i].handle for i in self.written_in]) != self.gwritten: self.graph, self.gwarm, self.gwritten = None, False, w
    for s, b in zip(self.gstage, in_bufs):
      if s is not None: dev.copy_async(s.handle, b.handle, b.nbytes)
    ins = [b if s is None else s for s, b in zip(self.gstage, in_bufs)]
    if self.graph is None and self.gwarm:
      if (g := dev.capture_graph(lambda: self._launch(ins, self.gowned, vv))) is None: self.graph_ok = False
      self.graph = g
    if self.graph is not None: self.graph.launch()
    else:
      self._launch(ins, self.gowned, vv)
      self.gwarm = True
    outs = []
    for k, r in enumerate(self.out_refs):
      if r[0] == 2 and self.persist[r[1]] is None and k in self.alias: outs.append(self.gowned[r[1]])
      elif r[0] == 2 and self.persist[r[1]] is None:
        src = self.gowned[r[1]]
        outs.append(b := Buffer(src.device, src.size, src.dtype).ensure_allocated())
        dev.copy_async(b.handle, src.handle, src.nbytes)
      else: outs.append(self._res(r, in_bufs, self.gowned))
    return outs

  def _launch(self, in_bufs: list[Buffer], owned: list[Buffer], vv: dict[Var, int]) -> None:
    res = lambda r: self._res(r, in_bufs, owned)
    for it in self.items:
      if it.done: continue
      it.done = it.const
      if it.kind == "copy":
        if (n := int(sym_infer(it.nbytes, vv)) if isinstance(it.nbytes, Node) else it.nbytes):
          dst, src = res(it.slots[0]), res(it.slots[1])
          host = memoryview(np.empty(n, np.uint8))
          Device[src.device].allocator.copyout(host, src.handle)
          Device[dst.device].allocator.copyin(dst.handle, host)
        continue
      bufs = [res(r) for r in it.slots]
      if it.prg is not None:
        gs = it.gs if it.static else tuple(int(sym_infer(s, vv)) for s in it.gs)
        if any(s <= 0 for s in gs): continue
        it.prg(*[b.handle for b in bufs[:it.nb]], global_size=gs, local_size=it.ls, vals=tuple(vv[v] for v in it.vars))
      elif it.fast is not None: it.fast(bufs)
      else:
        si = it.si
        _run(it.dev, ScheduleItem(si.ast, tuple(bufs), {v: vv[v] for v in si.var_vals}, si.kind, si.n_outputs))

def _device_fuse(sched: list[ScheduleItem]) -> list[ScheduleItem]:
  """A capture whose items all run on one device that offers `fuse_schedule` (NEURON, N6: several kernels as one launch)
  is replayed as that device's rewrite of it; the capture call itself ran the items one by one."""
  devs = {si.bufs[0].device for si in sched if si.bufs}
  if len(devs) != 1: return sched
  dev = Device[next(iter(devs))]
  return dev.fuse_schedule(sched) if hasattr(dev, "fuse_schedule") else sched

# ---------------------------------------------------------------- the decorator

class Jit:
  def __init__(self, fn: Callable, plan: bool = False, alias: tuple[int, ...] = ()):
    """plan: the captured intermediates share memory where their live ranges do not overlap (see _Captured._plan).
    alias: indices of outputs that a graph replay returns as its own persistent buffer instead of a fresh copy; the
    caller must be done with such an output before the next call (each replay overwrites it)."""
    self.fn, self.cnt, self.captured, self.plan, self.alias = fn, 0, None, plan, tuple(alias)
    self.__wrapped__ = fn
    self.__doc__, self.__name__ = getattr(fn, "__doc__", None), getattr(fn, "__name__", "jit")

  def reset(self) -> None: self.cnt, self.captured = 0, None

  def __call__(self, *args, var_vals: dict[Var, int] | None = None, **kwargs):
    tensors: list[Tensor] = []
    vv: dict[Var, int] = {}
    fargs, fkw = _flatten(args, tensors, vv), _flatten(kwargs, tensors, vv)
    for k, v in (var_vals or {}).items():
      assert isinstance(k, Var), f"var_vals keys must be Vars, got {k!r}"
      assert k.vmin <= int(v) <= k.vmax, f"{k.name}={v} is outside [{k.vmin}, {k.vmax}]"
      vv[k] = int(v)
    # nested in a recording (an outer jit's capture): run fn inline so the outer capture records its kernels
    inline = _state["rec"] is not None
    with bound(vv): in_bufs = [_as_realized_base(t) for t in tensors]
    assert len({id(b) for b in in_bufs}) == len(in_bufs), "jit: the same buffer is passed as two inputs"
    if not inline and self.cnt >= 2 and self.captured is not None: return self._replay(tensors, in_bufs, vv)
    if not inline: self.cnt += 1
    rec: list[ScheduleItem] | None = [] if self.cnt == 2 and not inline else None
    sizes: dict[int, Any] | None = {} if rec is not None else None
    with bound(vv, rec, sizes):
      ret = self.fn(*fargs, **fkw)
      outs: list[Tensor] = []
      _out_tensors(ret, outs)
      lbs = []
      for t in outs:
        lb = t.lazydata
        # a reshape of a base this call computes is returned as that base's buffer (no copy kernel; N5: one launch per step)
        reshaped = lb.base is not lb and lb.base.realized is None and lb.st.contiguous and len(lb.st.views) == 1 and \
                   all(isinstance(x, int) for x in (*lb.shape, *lb.base.shape)) and prod(lb.shape) == prod(lb.base.shape)
        if not ((lb.base is lb or reshaped) and lb.base.op is not LoadOps.CONST): lb = lb.contiguous()
        lbs.append(lb)
      if (todo := [lb for lb in lbs if lb.realized is None]): _realize_mod.realize(todo)
    out_bufs = [lb.realized for lb in lbs]
    out_shapes = [(lb.shape, lb.dtype) for lb in lbs]
    for t, lb in zip(outs, lbs):  # outputs are realized bases; a symbolic shape becomes this call's concrete shape
      t.lazydata = _wrap(lb.realized, lb.shape, vv).lazydata if not inline and any(isinstance(x, Node) for x in lb.shape) else lb
    if rec is not None and (sizes or {}).get("empty"):
      self.cnt = 1  # a symbolic buffer was empty at these var values, so its kernels never ran: capture on the next call
    elif rec is not None:
      self.captured = _Captured(_device_fuse(rec), in_bufs, out_bufs, sizes or {}, tuple(vv), [(t.shape, t.dtype, t.device) for t in tensors],
                                out_shapes, _rebuild(ret, iter([_SLOT] * len(outs))), self.plan)
      self.captured.alias = self.alias
    return ret

  def _replay(self, tensors: list[Tensor], in_bufs: list[Buffer], vv: dict[Var, int]):
    c = self.captured
    assert len(tensors) == len(c.in_meta), f"jit: got {len(tensors)} tensor inputs, captured {len(c.in_meta)}"
    for t, (shp, dt, dev) in zip(tensors, c.in_meta):
      assert t.dtype == dt and t.device == dev and tuple(t.shape) == tuple(shp), \
        f"jit: input {t.shape} {t.dtype} {t.device} does not match the captured {shp} {dt} {dev}"
    assert set(vv) == set(c.vars), f"jit: var values for {sorted(v.name for v in vv)}, captured {sorted(v.name for v in c.vars)}"
    vv = {**c.baked, **vv}
    bufs = c(in_bufs, vv)
    return _rebuild(c.ret_tree, iter([_wrap(b, shp, vv) for b, (shp, _) in zip(bufs, c.out_shapes)]))

def jit(fn: Callable) -> Jit:
  """Decorator: capture `fn`'s kernels on the second call and replay them afterwards (see module docstring)."""
  return Jit(fn)
