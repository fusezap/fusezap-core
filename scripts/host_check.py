#!/usr/bin/env python3
"""A pod's host check before a big-model run (stdlib only: runs before uv sync). Prints one HOST_CHECK line (JSON):
the container's memory limit (cgroup v2 memory.max / v1 memory.limit_in_bytes), MemTotal / MemAvailable, CPUs, and the
disk's sequential write and cold read speed (a GB-sized file in DIR, fsynced, dropped from the page cache with
posix_fadvise DONTNEED, read back in 16 MB blocks, deleted). Exits 3 with HOST_CHECK_FAIL lines when the effective
memory (min of the limit and MemTotal) is under --min-mem-gb or the read speed under --min-read-mbs: the 26 Oct 2x H200
pod's ~120-130 GB container limit stalled the 70B TP=2 load (I57 M6), and slow network disks stall it too.

  python3 scripts/host_check.py --dir ~/.cache/huggingface --gb 4 --min-mem-gb 100 --min-read-mbs 300
"""
from __future__ import annotations
import argparse, json, os, sys, time

def read(fn: str) -> str:
  try:
    with open(fn) as f: return f.read().strip()
  except OSError: return ""

def mem() -> dict:
  out: dict = {}
  for line in read("/proc/meminfo").splitlines():
    k, _, v = line.partition(":")
    if k in ("MemTotal", "MemAvailable"): out[k] = int(v.split()[0]) * 1024
  for lim, cur in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                   ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
    if v := read(lim):
      out["cgroup_limit"] = None if v == "max" or int(v) >= 1 << 60 else int(v)
      if (u := read(cur)).isdigit(): out["cgroup_used"] = int(u)
      break
  return out

def disk(d: str, gb: float) -> dict:
  os.makedirs(d, exist_ok=True)
  fn, blk = os.path.join(d, f".fz_host_check_{os.getpid()}"), os.urandom(16 << 20)
  n = max(1, int(gb * 1e9) // len(blk))
  try:
    t = time.perf_counter()
    fd = os.open(fn, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
      for _ in range(n): os.write(fd, blk)
      os.fsync(fd)
    finally: os.close(fd)
    w = n * len(blk) / (time.perf_counter() - t)
    fd = os.open(fn, os.O_RDONLY)
    try:
      if hasattr(os, "posix_fadvise"): os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)   # a cold read, not the page cache
      t, got = time.perf_counter(), 0
      while (b := os.read(fd, len(blk))): got += len(b)
      r = got / (time.perf_counter() - t)
      if hasattr(os, "posix_fadvise"): os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally: os.close(fd)
    return {"disk_dir": d, "disk_gb": round(n * len(blk) / 1e9, 2), "write_mbs": round(w / 1e6), "read_mbs": round(r / 1e6)}
  except OSError as e: return {"disk_dir": d, "disk_error": str(e)}
  finally:
    try: os.unlink(fn)
    except OSError: pass

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("--dir", default=os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface")))
  ap.add_argument("--gb", type=float, default=4.0)
  ap.add_argument("--min-mem-gb", type=float, default=100.0)
  ap.add_argument("--min-read-mbs", type=float, default=300.0)
  a = ap.parse_args()
  r = {**mem(), "cpus": os.cpu_count(), **disk(a.dir, a.gb)}
  eff = min(x for x in (r.get("cgroup_limit"), r.get("MemTotal")) if x) if (r.get("cgroup_limit") or r.get("MemTotal")) else None
  r["mem_effective_gb"] = round(eff / 1e9, 1) if eff else None
  print("HOST_CHECK " + json.dumps(r), flush=True)
  fails = []
  if eff is not None and eff < a.min_mem_gb * 1e9:
    fails.append(f"memory {eff / 1e9:.0f} GB (container limit {r.get('cgroup_limit') and round(r['cgroup_limit'] / 1e9)} GB, "
                 f"MemTotal {r.get('MemTotal', 0) / 1e9:.0f} GB) < {a.min_mem_gb:.0f} GB")
  if "disk_error" in r: fails.append(f"disk test failed: {r['disk_error']}")
  elif r["read_mbs"] < a.min_read_mbs: fails.append(f"disk read {r['read_mbs']} MB/s < {a.min_read_mbs:.0f} MB/s ({a.dir})")
  for f in fails: print(f"HOST_CHECK_FAIL {f}: retry on another pod", flush=True)
  return 3 if fails else 0

if __name__ == "__main__": sys.exit(main())
