"""N8 (I71): where a NEURON engine's device memory goes, per phase, with and without planned intermediates (FZ_NEURON_PLAN).

I69 run 1 (Qwen2.5-7B bf16 on one inf2 NeuronCore) failed to allocate at 15.85 GiB of tensors against 13.17 GiB of weights.
This script runs an LLMEngine on a model with 7B's layer shapes but few layers (random weights; NKISHIM on the Mac, NEURON
on an instance) through the phases that allocate: warm-up (decode 4 rows + prefill 128 rows), a c4 burst, then the logprobs
scorer on one sequence (teacher-forced: Validate's accuracy check, and the in-process top-1). After each phase it reads
neuron_kv.hbm_report: the bytes each captured forward and the scorer keep between replays, the KV pool, the allocator's free
list and the rest (weights). From two layer counts it extrapolates per-layer bytes to 28 layers.

  FZ_DEVICE=NKISHIM uv run python scripts/nki_hbm_account.py --layers 1,2 --score-tokens 300 --json out.json      # Mac
  FZ_DEVICE=NKISHIM uv run python scripts/nki_hbm_account.py --small                                             # a quick dry run
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def log(*a) -> None: print(*a, flush=True)

SHAPES_7B = dict(hidden=3584, inter=18944, heads=28, kv=4, hd=128)
SHAPES_SMALL = dict(hidden=256, inter=640, heads=28, kv=4, hd=128)   # 7B's head layout, small widths
SHAPES_L8B_TP2 = dict(hidden=4096, inter=7168, heads=16, kv=4, hd=128)   # I13: one Llama-3.1-8B TP=2 rank's layer (--shapes l8b-tp2)

def run(layers: int, plan: bool, shapes: dict, score_tokens: int, slots: int, vocab: int) -> dict:
  os.environ["FZ_NEURON_PLAN"] = "1" if plan else "0"
  from fusezap.device import Device
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.logprobs import scorer
  from fusezap.serve.neuron_kv import hbm_report
  from nki_n7_layer_bench import qwen_shaped
  dev = os.environ.get("FZ_DEVICE", "NKISHIM")
  al = Device[dev].allocator
  al.free_cache()
  m = qwen_shaped(dev, 1, slots, layers=layers, vocab=vocab, **shapes)   # max_len = slots
  m.max_len = slots
  eng = LLMEngine(m, num_blocks=slots // 16, block_size=16, max_num_seqs=4, max_prefill_tokens=128, max_model_len=slots)
  out: dict = {}
  rng = np.random.default_rng(0)
  def phase(nm: str) -> None:
    out[nm] = r = hbm_report(eng)
    log(f"  layers {layers} plan {int(plan)} {nm:<16} forwards {sum(r['forwards'].values()) / 2**20:9.1f} MiB {r['forwards']}  scorer "
        f"{r['scorer'] / 2**20:8.1f} MiB  kv {r['kv'] / 2**20:6.1f} MiB  cached {r['cached'] / 2**20:8.1f} MiB  held {r['held'] / 2**20:9.1f} MiB")
  t0 = time.perf_counter()
  for n in (1, 4):   # warm-up: a prefill (128 rows) and the 4-row decode, twice each (fz.jit captures on the second call)
    rids = [eng.add_request(rng.integers(0, vocab, 40).tolist(), SamplingParams(max_tokens=4, ignore_eos=True)) for _ in range(n)]
    while eng.has_unfinished(): eng.step()
  phase("warm-up")
  rids = [eng.add_request(rng.integers(0, vocab, 100).tolist(), SamplingParams(max_tokens=6, ignore_eos=True)) for _ in range(4)]
  while eng.has_unfinished(): eng.step()
  phase("c4")
  if score_tokens:
    sc = scorer(eng)
    ids = rng.integers(0, vocab, score_tokens).tolist()
    for _ in range(2): sc.score(ids, score_tokens // 2, 5)   # twice: the jit captures on the second call
    phase("scored")
  out["seconds"] = round(time.perf_counter() - t0, 1)
  del eng, m
  return out

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--layers", default="1,2")
  ap.add_argument("--score-tokens", type=int, default=300)
  ap.add_argument("--slots", type=int, default=1024)
  ap.add_argument("--vocab", type=int, default=512)
  ap.add_argument("--small", action="store_true", help="7B's head layout at small widths (a quick run)")
  ap.add_argument("--shapes", default="", choices=["", "l8b-tp2"], help="l8b-tp2: a Llama-3.1-8B TP=2 rank's layer shapes (use --target-layers 32)")
  ap.add_argument("--target-layers", type=int, default=28)
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  os.environ.setdefault("FZ_DEVICE", "NKISHIM")
  os.environ.setdefault("FZ_NEURON_DTYPE", "bf16")
  os.environ.setdefault("FZ_NEURON_FUSE", "1")
  os.environ.setdefault("FZ_DECODE_BUCKETS", "0")
  os.environ.setdefault("FZ_NEURON_CTX", "0")
  shapes = SHAPES_L8B_TP2 if a.shapes == "l8b-tp2" else SHAPES_SMALL if a.small else SHAPES_7B
  Ls = [int(x) for x in a.layers.split(",")]
  res: dict = {"shapes": shapes, "runs": {}}
  for plan in (False, True):
    for L in Ls: res["runs"][f"L{L} plan{int(plan)}"] = run(L, plan, shapes, a.score_tokens, a.slots, a.vocab)
  if len(Ls) >= 2:
    lo, hi = min(Ls), max(Ls)
    log(f"--- extrapolated to {a.target_layers} layers (per-layer bytes from {lo} and {hi} layers; prediction, not a device measurement)")
    for plan in (False, True):
      for ph in ("warm-up", "c4", "scored"):
        a_, b_ = res["runs"][f"L{lo} plan{int(plan)}"].get(ph), res["runs"][f"L{hi} plan{int(plan)}"].get(ph)
        if not a_ or not b_: continue
        tot = lambda r: sum(r["forwards"].values()) + r["scorer"]
        per = (tot(b_) - tot(a_)) / (hi - lo)
        est = tot(a_) + per * (a.target_layers - lo)
        res.setdefault("extrapolated", {})[f"plan{int(plan)} {ph}"] = est
        log(f"plan {int(plan)} after {ph:<8}: captured forwards + scorer {est / 2**30:.3f} GiB at {a.target_layers} layers "
            f"({per / 2**20:.1f} MiB per layer)")
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1)

if __name__ == "__main__": main()
