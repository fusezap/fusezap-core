"""Greedy-token identity between engine configurations on one GPU (docs/status/SPEED.md "Run 4"; a tool, not the harness).

  python bench/ab_tokens.py --model <path> --out new.json [--configs "default;FZ_LOOKAHEAD=0,FZ_DECODE_BUCKETS=0;FZ_PAGED_DECODE=0"]
  python bench/ab_tokens.py --compare new.json base.json

One model load, then for each configuration (env assignments, ';'-separated; read when an engine is built or a forward is
traced) a fresh LLMEngine (32 rows, 2048 context) runs 48 greedy requests together (24 x 512-token prompts and 24 of
1-1500 tokens, 48 new tokens each, ignore_eos) and 6 of them alone. Writes {config: {"batched": [...], "alone": [...]}}.
--compare prints, for every pair of configurations across the files, how many requests are identical and how many
tokens agree by position, and whether batched == alone inside each configuration.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

def workload() -> list[list[int]]:
  rng = np.random.default_rng(1234)
  return [rng.integers(1000, 30000, 512).tolist() for _ in range(24)] + [rng.integers(1000, 30000, int(n)).tolist() for n in rng.integers(1, 1500, 24)]

def run(model: str, configs: list[str]) -> dict:
  from fusezap import helpers
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  m = load_model(model, dtype=dtypes.bfloat16, device="CUDA", max_len=2048, prefill_chunk=128)
  prompts, out = workload(), {}
  for cfg in configs:
    env = dict(kv.split("=", 1) for kv in cfg.split(",") if "=" in kv)
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    helpers.getenv.cache_clear()
    try:
      eng = LLMEngine(m, num_blocks=6000, block_size=16, max_num_seqs=32, max_model_len=2048)
      sp = SamplingParams(max_tokens=48, ignore_eos=True)
      batched = eng.generate(prompts, sp)
      alone = [eng.generate([prompts[i]], sp)[0] for i in (0, 5, 24, 30, 40, 47)]
      out[cfg] = {"batched": batched, "alone": alone, "stats": dict(eng.stats)}
      print(f"{cfg}: done, {eng.stats}", flush=True)
      del eng
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      helpers.getenv.cache_clear()
  return out

def compare(files: list[str]) -> None:
  runs = {}
  for f in files:
    for cfg, r in json.load(open(f)).items(): runs[f"{os.path.basename(f)}:{cfg}"] = r
  for name, r in runs.items():
    same = sum(a == r["batched"][i] for a, i in zip(r["alone"], (0, 5, 24, 30, 40, 47)))
    print(f"{name}: batched == alone for {same}/6 requests")
  names = list(runs)
  for i, a in enumerate(names):
    for b in names[i + 1:]:
      A, B = runs[a]["batched"], runs[b]["batched"]
      ident = sum(x == y for x, y in zip(A, B))
      toks = sum(sum(p == q for p, q in zip(x, y)) for x, y in zip(A, B))
      al = sum(x == y for x, y in zip(runs[a]["alone"], runs[b]["alone"]))
      print(f"{a} vs {b}: {ident}/{len(A)} requests identical, {toks}/{sum(len(x) for x in A)} tokens equal by position; alone {al}/{len(runs[a]['alone'])} identical")

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
  ap.add_argument("--configs", default="default")
  ap.add_argument("--out", default="")
  ap.add_argument("--compare", nargs="+", default=None)
  ap.add_argument("--require-alone", action="store_true", help="exit 3 unless batched == alone (6/6) in every configuration (a gate)")
  a = ap.parse_args()
  if a.compare: compare(a.compare); return 0
  res = run(a.model, [c.strip() for c in a.configs.split(";") if c.strip()])
  if a.out:
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f: json.dump(res, f)
  bad = []
  for cfg, r in res.items():
    same = sum(x == r["batched"][i] for x, i in zip(r["alone"], (0, 5, 24, 30, 40, 47)))
    print(f"{cfg}: batched == alone for {same}/{len(r['alone'])} requests", flush=True)
    if same != len(r["alone"]): bad.append(cfg)
  if a.require_alone and bad: print(f"GATE FAILED: batched != alone in {bad}", flush=True); return 3
  return 0

if __name__ == "__main__": sys.exit(main())
