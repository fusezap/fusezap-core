"""Regression tests for bench/bench_ops.py timing helpers."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench"))
import bench_ops

def test_time_fz_synchronizes_before_clock_reads():
  ev = []
  class T:
    def realize(self): ev.append("run"); return self
  bench_ops.time_fz(lambda: T(), "CPU", iters=2, warmup=1, sync=lambda: ev.append("sync"))
  assert ev == ["run", "sync", "run", "run", "sync"]
