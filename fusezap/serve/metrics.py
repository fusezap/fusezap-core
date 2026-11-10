"""Prometheus `/metrics` of `fusezap serve` (I55): vLLM's metric names, so Invarix and existing vLLM dashboards read it
unchanged. Text exposition format 0.0.4, written by hand (no prometheus_client dependency).

  vllm:num_requests_running        gauge    sequences in the scheduler's running set (prefilling or decoding)
  vllm:num_requests_waiting        gauge    queued, not admitted yet (also preempted sequences waiting to be readmitted)
  vllm:kv_cache_usage_perc         gauge    used fraction of the paged KV pool's blocks (0..1)
  vllm:num_preemptions_total       counter  preemptions since the server started (recompute)
  vllm:prompt_tokens_total         counter  prompt tokens of the requests the engine accepted
  vllm:generation_tokens_total     counter  tokens generated for clients
  process_resident_memory_bytes    gauge    this (HTTP server) process's current RSS; with --engine-process the engine
                                            process is not included (as vLLM's API-server process metric)

The loop that steps the engine (AsyncEngine's thread, or the engine process) owns an EngineStats and replaces its `snap`
after every step and every batch of commands (adds, aborts), so the gauges follow aborts without waiting for a step. A
reader takes the current `snap` (one immutable tuple: an attribute read, no lock) and never touches the engine.
"""
from __future__ import annotations
import ctypes, os, sys
from typing import NamedTuple

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

class Snap(NamedTuple):
  running: int = 0
  waiting: int = 0
  kv: float = 0.0
  preempt: int = 0
  prompt: int = 0
  gen: int = 0

class EngineStats:
  """Engine-side counters; update() reads the scheduler and the KV allocator (host bookkeeping only). An engine without
  a scheduler or paged cache (the tests' stand-ins) reports 0 for what it lacks."""
  def __init__(self, eng):
    self.eng, self.prompt, self.gen = eng, 0, 0
    self.sc, self.c = getattr(eng, "scheduler", None), getattr(eng, "cache", None)
    self.preempt0 = getattr(self.sc, "num_preemptions", 0)   # warm-up's are not the server's
    self.snap = Snap()
    self.update()
  def update(self) -> bool:
    """New snapshot; True if it changed."""
    sc, c = self.sc, self.c
    nb = getattr(c, "num_blocks", 0)
    s = Snap(len(sc.running) if sc else 0, len(sc.waiting) if sc else 0, round(1 - c.num_free_blocks / nb, 6) if nb else 0.0,
             getattr(sc, "num_preemptions", 0) - self.preempt0, self.prompt, self.gen)
    ch, self.snap = s != self.snap, s
    return ch

def _esc(v: str) -> str: return v.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
def _num(v: float) -> str: return repr(float(v)) if isinstance(v, float) else str(v)

def render(model: str, s: Snap, rss: int | None = None) -> str:
  lab = f'{{model_name="{_esc(model)}"}}'
  rows = [("vllm:num_requests_running", "gauge", "Number of requests in model execution batches.", "", s.running),
          ("vllm:num_requests_waiting", "gauge", "Number of requests waiting to be processed.", "", s.waiting),
          ("vllm:kv_cache_usage_perc", "gauge", "KV-cache usage. 1 means 100 percent usage.", "", s.kv),
          ("vllm:num_preemptions", "counter", "Cumulative number of preemptions from the engine.", "_total", s.preempt),
          ("vllm:prompt_tokens", "counter", "Number of prefill tokens processed.", "_total", s.prompt),
          ("vllm:generation_tokens", "counter", "Number of generation tokens processed.", "_total", s.gen)]
  out = []
  for name, kind, hlp, suf, v in rows: out += [f"# HELP {name}{suf} {hlp}", f"# TYPE {name}{suf} {kind}", f"{name}{suf}{lab} {_num(v)}"]
  if rss is not None:
    out += ["# HELP process_resident_memory_bytes Resident memory size in bytes.", "# TYPE process_resident_memory_bytes gauge",
            f"process_resident_memory_bytes {rss}"]
  return "\n".join(out) + "\n"

class _MachInfo(ctypes.Structure):   # mach_task_basic_info (MACH_TASK_BASIC_INFO = 20)
  _fields_ = [("virtual_size", ctypes.c_uint64), ("resident_size", ctypes.c_uint64), ("resident_size_max", ctypes.c_uint64),
              ("user_time", ctypes.c_int32 * 2), ("system_time", ctypes.c_int32 * 2), ("policy", ctypes.c_int32), ("suspend_count", ctypes.c_int32)]

_libc = None
def rss_bytes() -> int | None:
  """Current RSS of this process: /proc on Linux, mach task_info on macOS; else getrusage's peak RSS (an upper bound)."""
  global _libc
  try:
    if sys.platform.startswith("linux"):
      with open("/proc/self/statm") as f: return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    if sys.platform == "darwin":
      if _libc is None: _libc = ctypes.CDLL(None)
      info, cnt = _MachInfo(), ctypes.c_uint32(ctypes.sizeof(_MachInfo) // 4)
      task = ctypes.c_uint32.in_dll(_libc, "mach_task_self_")
      if _libc.task_info(task, 20, ctypes.byref(info), ctypes.byref(cnt)) == 0: return int(info.resident_size)
  except Exception: pass
  try:
    import resource
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)
  except Exception: return None
