"""Prefill and decode throughput of fusezap.serve's Llama model at batch 1 (SPEC §11.3, P4-MODEL).

  uv run python bench/bench_decode.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--device METAL] [--dtypes float32,float16]
                                      [--prompt 128,512] [--decode 128] [--beam 4] [--max-len 1024]

For each dtype: prefill tokens/s for each prompt length (jit replay of the padded bucket, after two warm-up calls that
compile and capture), then decode tokens/s over `--decode` greedy steps after a 128-token prompt (host argmax and the
logits read-back included; p50 is the median step). Decode kernels are tuned with FZ_BEAM=--beam (winners are cached
in ~/.cache/fusezap, so only the first run pays for the search, a few minutes per dtype; --beam 0 uses the hand-coded
opts). Exits non-zero if a run fails.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fusezap import Device
from fusezap.dtype import dtypes
from fusezap.serve.models import load_model

def run(name: str, device: str, dt, prompts: list[int], n_decode: int, beam: int, max_len: int) -> dict:
  t0 = time.perf_counter()
  m = load_model(name, dtype=dt, device=device, max_len=max_len, prefill_chunk=512, beam=beam)
  dev = Device[m.device]
  load_s = time.perf_counter() - t0
  rng = np.random.default_rng(0)
  res: dict = {"load_s": load_s, "prefill": {}}
  for n in prompts:
    ids = rng.integers(1000, 30000, n).tolist()
    best = float("inf")
    for i in range(5):   # calls 1-2 compile and capture the bucket's jit; 3+ replay
      m.reset()
      dev.synchronize()
      st = time.perf_counter()
      m.prefill([ids])
      dev.synchronize()
      if i >= 2: best = min(best, time.perf_counter() - st)
    res["prefill"][n] = n / best
  m.reset()
  ids = rng.integers(1000, 30000, 128).tolist()
  lg = m.prefill([ids])
  tok = int(lg[0].argmax())
  t0 = time.perf_counter()
  for _ in range(3): tok = int(m.decode([[tok]])[0].argmax())   # compile + capture (+ BEAM search on a cold cache)
  res["compile_s"] = time.perf_counter() - t0
  dev.synchronize()
  st, steps = time.perf_counter(), []
  for _ in range(n_decode):
    t = time.perf_counter()
    tok = int(m.decode([[tok]])[0].argmax())   # reading the logits waits for the step's kernels
    steps.append(time.perf_counter() - t)
  dev.synchronize()
  el = time.perf_counter() - st
  res["decode"], res["decode_ms"], res["p50_ms"] = n_decode / el, el / n_decode * 1000, float(np.median(steps)) * 1000
  res["kernels"] = len(m._decode_jit.captured.items) if m._decode_jit.captured else 0
  del m
  return res

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
  ap.add_argument("--device", default="METAL")
  ap.add_argument("--dtypes", default="float32,float16")
  ap.add_argument("--prompt", default="128,512")
  ap.add_argument("--decode", type=int, default=128)
  ap.add_argument("--beam", type=int, default=4)
  ap.add_argument("--max-len", type=int, default=1024)
  a = ap.parse_args()
  if not Device.available(a.device):
    print(f"device {a.device} is not available")
    return 1
  prompts = [int(x) for x in a.prompt.split(",")]
  print(f"model {a.model}, device {a.device}, batch 1, max_len {a.max_len}, decode BEAM {a.beam}")
  print(f"{'dtype':10s}" + "".join(f"  prefill {n:4d} tok/s" for n in prompts) + "   decode tok/s   ms/token  p50 ms  kernels/step")
  for name in a.dtypes.split(","):
    dt = getattr(dtypes, name)
    r = run(a.model, a.device, dt, prompts, a.decode, a.beam, a.max_len)
    print(f"{name:10s}" + "".join(f"  {r['prefill'][n]:18.0f}" for n in prompts) +
          f"   {r['decode']:12.1f}   {r['decode_ms']:8.2f}  {r['p50_ms']:6.2f}  {r['kernels']:12d}   (load {r['load_s']:.1f}s, decode warm-up {r['compile_s']:.1f}s)")
    sys.stdout.flush()
  return 0

if __name__ == "__main__": sys.exit(main())
