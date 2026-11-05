"""Aggregate throughput of fusezap.serve's LLMEngine (paged KV cache + continuous batching) at batch 1, 8 and 32 on METAL.

  uv run python bench/bench_batch.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--device METAL] [--dtype float32]
                                     [--batches 1,8,32] [--prompt 128] [--gen 128] [--beam 4]

For each batch size B: B different random prompts of --prompt tokens are submitted at once, each generates --gen tokens
(greedy, EOS ignored). Reported: aggregate generated tokens/s over the whole run (prefill included), decode tokens/s
over the steps that only decoded, the median step time and the number of steps. A warm-up request first compiles and
captures both forwards (and runs the BEAM search for the decode kernels on a cold ~/.cache/fusezap: minutes, once).
The engine shape (32 decode rows, 256 prefill rows, 16-token blocks) matches tests/test_scheduler.py.
Exits non-zero if a run fails or if request 0 of the largest batch does not produce the batch-1 tokens.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fusezap import Device
from fusezap.dtype import dtypes
from fusezap.serve.models import load_model
from fusezap.serve.engine import LLMEngine, SamplingParams

def run(eng: LLMEngine, prompts: list[list[int]], gen: int) -> dict:
  dev = Device[eng.model.device]
  rids = [eng.add_request(p, SamplingParams(max_tokens=gen, ignore_eos=True)) for p in prompts]
  dev.synchronize()
  steps, dec_t, dec_tok = [], 0.0, 0
  t0 = time.perf_counter()
  while eng.has_unfinished():
    pre = eng.stats["prefill_steps"]
    t = time.perf_counter()
    outs = eng.step()
    dt = time.perf_counter() - t
    steps.append(dt)
    if eng.stats["prefill_steps"] == pre: dec_t, dec_tok = dec_t + dt, dec_tok + sum(len(o.new_token_ids) for o in outs)
  el = time.perf_counter() - t0
  toks = [list(eng.seqs[r].output) for r in rids]
  for r in rids: eng.release(r)
  assert all(len(t) == gen for t in toks)
  return {"agg": len(prompts) * gen / el, "decode": dec_tok / dec_t if dec_t else 0.0, "p50": float(np.median(steps)) * 1000,
          "steps": len(steps), "s": el, "toks": toks}

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
  ap.add_argument("--device", default=None)
  ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
  ap.add_argument("--batches", default="1,8,32")
  ap.add_argument("--prompt", type=int, default=128)
  ap.add_argument("--gen", type=int, default=128)
  ap.add_argument("--beam", type=int, default=4)
  ap.add_argument("--blocks", type=int, default=1024)
  a = ap.parse_args()
  dt = getattr(dtypes, a.dtype)
  m = load_model(a.model, dtype=dt, device=a.device, max_len=512, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=a.blocks, block_size=16, max_num_seqs=32, max_prefill_tokens=256, decode_beam=a.beam)
  rng = np.random.default_rng(0)
  t0 = time.perf_counter()
  eng.generate([rng.integers(1000, 30000, 300).tolist()], SamplingParams(max_tokens=4, ignore_eos=True))
  eng.generate([rng.integers(1000, 30000, 20).tolist()] * 2, SamplingParams(max_tokens=4, ignore_eos=True))
  print(f"model {a.model}, device {m.device}, {a.dtype}, prompt {a.prompt}, gen {a.gen}, decode BEAM {a.beam}; warm-up {time.perf_counter() - t0:.1f}s")
  prompts = [rng.integers(1000, 30000, a.prompt).tolist() for _ in range(max(int(b) for b in a.batches.split(",")))]
  print(f"{'batch':>5}  {'agg tok/s':>10}  {'decode tok/s':>12}  {'p50 step ms':>11}  {'steps':>5}  {'seconds':>7}")
  res = {}
  for b in (int(x) for x in a.batches.split(",")):
    r = res[b] = run(eng, prompts[:b], a.gen)
    print(f"{b:>5}  {r['agg']:>10.1f}  {r['decode']:>12.1f}  {r['p50']:>11.2f}  {r['steps']:>5}  {r['s']:>7.2f}", flush=True)
  lo, hi = min(res), max(res)
  ok = res[hi]["toks"][0] == res[lo]["toks"][0]
  print(f"request 0 at batch {hi} == batch {lo}: {'yes' if ok else 'NO'}")
  return 0 if ok else 1

if __name__ == "__main__": sys.exit(main())
