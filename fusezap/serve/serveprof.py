"""Opt-in timeline of `fusezap serve` (N7 serve): FZ_SERVE_PROFILE=<file.jsonl> ("1": fusezap_serve_profile.jsonl here).

Off by default: the engine loop and the handlers keep `prof = None` and every hook is one `is not None` check.
When set, the engine loop (AsyncEngine's thread, or the engine process) appends one line per engine step:
  k "step"; t wall clock (time.time) at the step's start; ms the step's wall (LLMEngine.step), cpu the engine thread's CPU
  time in it (time.thread_time: excludes waiting for the device and for the GIL), dev the wall inside runtime-worker
  round trips (NEURON: the device's time plus the pipe), gap the wall since the previous step ended (the loop between
  steps: requests added, outputs handed to the HTTP side, stats), out the hand-off of this step's outputs (ms, part of
  the next gap); kind "d" | "p" | "dp" (decode, prefill, both), tok the positions it computed (decode rows + prompt
  tokens), n the outputs it handed off (one token each); ctx the NEURON pool prefixes its forwards ran at and top the
  slots they needed (the highest block in use + 1, times the block size); run / wait the running and
  waiting sequences after it; adds the requests added before it.
and the HTTP side (in the server process) one line per request when it ends:
  k "req"; rid; n tokens; chunks outputs handled; lag the ms from the engine's hand-off of a step's output to the
  handler taking it (p50, p99, max: the event loop's delay); hms the ms the handler spent per output (detokenize, stop
  scan, chunk JSON, the SSE write up to the next await: mean, total); coal outputs merged into an earlier one.
scripts/serve_prof_summary.py aggregates a file (steps by kind, time outside the steps, the HTTP side, pool prefixes).
"""
from __future__ import annotations
import json, os, threading, time
from typing import Any

def path_from_env() -> str | None:
  p = os.environ.get("FZ_SERVE_PROFILE", "")
  if p in ("", "0"): return None
  return "fusezap_serve_profile.jsonl" if p == "1" else p

def from_env() -> "ServeProf | None":
  p = path_from_env()
  return ServeProf(p) if p else None

class ServeProf:
  def __init__(self, path: str):
    self.f = open(path, "a", buffering=1)
    self.lock = threading.Lock()
    self.prev_end: float | None = None
    self.adds = 0
    self._t = self._c = self._w = 0.0
    self._st: dict = {}
    self._eng: Any = None
    self._al: Any = None
  def write(self, rec: dict) -> None:
    with self.lock: self.f.write(json.dumps(rec) + "\n")

  # ---- engine loop (one thread)
  def _alloc(self, eng: Any) -> Any:
    if self._eng is not eng:
      self._eng, self._al = eng, None
      if getattr(eng, "neuron", False):
        from fusezap.device import Device
        self._al = getattr(Device[eng.model.device], "allocator", None)
    return self._al
  def begin(self, eng: Any) -> None:
    al = self._alloc(eng)
    self._st = dict(eng.stats)
    eng._ctx_seen = []
    self._w = getattr(al, "wait_s", 0.0) if al is not None else 0.0
    self._c, self._t = time.thread_time(), time.perf_counter()
  def end(self, eng: Any, wall: float) -> dict:
    """The step's record (written by `done`, after its outputs were handed off)."""
    t1, c1 = time.perf_counter(), time.thread_time()
    al, s0, s = self._al, self._st, eng.stats
    sch = eng.scheduler
    dd, dp = s["decode_steps"] - s0.get("decode_steps", 0), s["prefill_steps"] - s0.get("prefill_steps", 0)
    toks = s["tokens"] - s0.get("tokens", 0)
    rec = {"k": "step", "t": round(wall, 6), "ms": round(1e3 * (t1 - self._t), 4), "cpu": round(1e3 * (c1 - self._c), 4),
           "gap": round(1e3 * (self._t - self.prev_end), 4) if self.prev_end is not None else None,
           "kind": ("d" if dd else "") + ("p" if dp else ""), "tok": toks,
           "run": len(sch.running), "wait": len(sch.waiting), "adds": self.adds}
    if (cs := eng._ctx_seen): rec["ctx"], rec["top"] = sorted({c for c, _ in cs}), max(t for _, t in cs)
    if al is not None: rec["dev"] = round(1e3 * (getattr(al, "wait_s", 0.0) - self._w), 4)
    self.adds, self._te = 0, t1
    return rec
  def done(self, rec: dict, n_out: int) -> None:
    now = time.perf_counter()
    rec["out"], rec["n"] = round(1e3 * (now - self._te), 4), n_out
    self.prev_end = now
    self.write(rec)
  def idle(self) -> None: self.prev_end = None   # the loop waits for requests: no gap to count

  # ---- HTTP side (event loop)
  def request(self, rid: Any, n: int, lags: list[float], hms: list[float], coal: int) -> None:
    lags = sorted(lags)
    q = lambda p: round(lags[min(len(lags) - 1, int(p * len(lags)))], 4) if lags else None
    self.write({"k": "req", "t": round(time.time(), 6), "rid": rid, "n": n, "chunks": len(hms), "coal": coal,
                "lag": [q(0.5), q(0.99), round(lags[-1], 4) if lags else None],
                "hms": [round(sum(hms) / len(hms), 4) if hms else None, round(sum(hms), 4)]})
