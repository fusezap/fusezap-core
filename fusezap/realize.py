"""Realization entry point (SPEC §2). Implemented by WP2 (dispatch) and WP3 (`run_schedule`).

  realize(lbs)        compute every LazyBuffer in `lbs` and set `.base.realized`
                      FZ_INTERP=1 -> fusezap.interpret.realize(lbs)
                      otherwise   -> run_schedule(create_schedule(lbs))
  run_schedule(sched) allocate buffers and call Device[buf.device].run(si) for each item, in order;
                      FZ_DEBUG>=1 prints one line per kernel with its time
"""
from __future__ import annotations
from fusezap.helpers import getenv
from fusezap.lazy import LazyBuffer
from fusezap.ir import ScheduleItem

def realize(lbs: list[LazyBuffer]) -> None:
  if getenv("FZ_INTERP"):
    from fusezap.interpret import realize as interp_realize
    return interp_realize(lbs)
  from fusezap.schedule import create_schedule
  run_schedule(create_schedule(lbs))

def run_schedule(schedule: list[ScheduleItem]) -> None:
  import time
  from fusezap.device import Device
  dbg = getenv("FZ_DEBUG", 0)
  for i, si in enumerate(schedule):
    for b in si.bufs: b.ensure_allocated()
    st = time.perf_counter()
    if si.kind == "copy":
      dst, src = si.bufs[0], si.bufs[1]
      if dst.nbytes and src.device == "NPY": dst.copyin(memoryview(src.handle)[:dst.nbytes])   # host data: no staging copy
      elif dst.nbytes:
        import numpy as np
        host = np.empty(dst.nbytes, np.uint8)
        src.copyout(memoryview(host))
        dst.copyin(memoryview(host))
      et, name = None, f"copy {src.device} -> {dst.device}"
    else:
      dev = Device[si.bufs[0].device]
      et = dev.run(si)
      if dbg >= 1: dev.synchronize()
      name = f"kernel {si.bufs[0].device}"
    if dbg >= 1:
      t = et if et is not None else time.perf_counter() - st
      print(f"*** {i:4d} {name:24s} {si.bufs[0].size:9d} x {si.bufs[0].dtype.name:8s} {t*1e3:9.3f} ms")
