"""Opt-in per-step timeline of LLMEngine.step (docs/status/SPEED.md "Run 4"): FZ_STEP_PROF=<file.jsonl>.

Off by default. When FZ_STEP_PROF is unset the engine keeps `_prof = None` and every hook is one `is not None` check.
When set, each step appends one JSON line:
  t           wall clock (time.time()) at the step's start, to slice steps by a load window
  rows_d      decode rows launched in the step (real sequences; a decode forward runs its bucket's rows), 0 if none
  rows_p      prompt rows of the prefill forward, size_p its row count (a prefill size), 0 if none
  host ms     sched (Scheduler.schedule), d_prep / p_prep / s_prep (rows, step metadata, input upload or build), d_call /
              p_call / s_call (the jit call: staging copies + graph launch); s is the lookahead decode, enqueued before the
              previous decode's tokens are read; post (token read-back, which waits for that forward, and the finish
              checks), total
  fwd         (CUDA only) the forwards that completed by the end of the step, in order: [kind, GPU ms from just before to
              just after the jit call, GPU idle ms since the previous forward ended (null for the first)]. Events only:
              the profiler never waits for the GPU, so it does not change the overlap it measures.
bench/step_prof_summary.py aggregates a file.
"""
from __future__ import annotations
import collections, json, os, time
from typing import Any

def from_env(device: str) -> "StepProf | None":
  path = os.environ.get("FZ_STEP_PROF")
  return StepProf(path, device) if path else None

class StepProf:
  def __init__(self, path: str, device: str):
    self.f = open(path, "a", buffering=1)
    self.cu = None
    if device.split(":")[0] == "CUDA":
      from fusezap.device import Device
      self.dev = Device[device]
      self.cu = self.dev.cu
    self.inflight: collections.deque = collections.deque()   # (kind, start event, end event)
    self.prev_end: Any = None
    self.rec: dict[str, Any] = {}
    self._t = self._t0 = 0.0
    self._cur: tuple[str, Any] | None = None

  def _event(self) -> Any:
    self.dev.ctx()
    e = self.cu.cuEventCreate(0)[1]
    self.cu.cuEventRecord(e, self.dev.stream)
    return e

  def _ready(self, e: Any) -> bool:
    q = self.cu.cuEventQuery(e)
    q = q[0] if isinstance(q, tuple) else q
    return int(getattr(q, "value", q)) == 0

  def _ms(self, a: Any, b: Any) -> float:
    err, ms = self.cu.cuEventElapsedTime(a, b)
    return round(float(ms), 4) if int(getattr(err, "value", err)) == 0 else -1.0

  # ---- hooks called by LLMEngine.step / _Forward.__call__
  def begin(self) -> None:
    self.rec = {"t": round(time.time(), 6), "rows_d": 0, "rows_p": 0, "size_p": 0}
    self._t0 = self._t = time.perf_counter()

  def mark(self, name: str) -> None:
    t = time.perf_counter()
    self.rec[name] = round(self.rec.get(name, 0.0) + (t - self._t) * 1e3, 4)
    self._t = t

  def fwd(self, kind: str, which: int) -> None:
    """kind 'd' / 'p' / 's'; which 0 just before the jit call, 1 just after."""
    if self.cu is None: return
    if which == 0: self._cur = (kind, self._event())
    elif self._cur is not None:
      self.inflight.append((self._cur[0], self._cur[1], self._event()))
      self._cur = None

  def end(self, rows_d: int, rows_p: int, size_p: int) -> None:
    self.mark("post")
    r = self.rec
    r["rows_d"], r["rows_p"], r["size_p"] = rows_d, rows_p, size_p
    r["total"] = round((time.perf_counter() - self._t0) * 1e3, 4)
    if self.cu is not None:
      done = []
      while self.inflight and self._ready(self.inflight[0][2]):
        kind, s, e = self.inflight.popleft()
        done.append([kind, self._ms(s, e), self._ms(self.prev_end, s) if self.prev_end is not None else None])
        self.cu.cuEventDestroy(s)
        if self.prev_end is not None: self.cu.cuEventDestroy(self.prev_end)
        self.prev_end = e
      r["fwd"] = done
    self.f.write(json.dumps(r) + "\n")
