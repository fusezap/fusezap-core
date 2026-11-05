"""Memory fit of one model on one CUDA GPU (H100-32B): weights, free memory, and the largest KV pool the engine serves with.

  python bench/mem_fit.py Qwen/Qwen2.5-32B-Instruct [--quant fp8] [--max-num-seqs 32] [--max-len 32768]

Loads the model as `fusezap serve` does and prints its device weight bytes and the free memory. Then it builds an
LLMEngine whose KV pool fills the free memory minus a reserve (reserves tried small to large), warms it up and runs
two loads that use the whole pool: max_num_seqs prompts at once, and one prompt as long as the pool (up to --max-len).
The first reserve that runs gives the KV capacity in tokens, the prefill budget, the longest context and the batch at
a few context lengths. A probe, not a benchmark: no timing claims.
"""
from __future__ import annotations
import argparse, gc, json, time
from fusezap.device import Device
from fusezap.dtype import dtypes
from fusezap.serve.models import load_model
from fusezap.serve.engine import LLMEngine, SamplingParams
from fusezap.serve.api import warmup
from fusezap.serve.quant import quantize_model, weight_bytes

GiB = 1 << 30

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("model")
  ap.add_argument("--quant", default=None)
  ap.add_argument("--max-num-seqs", type=int, default=32)
  ap.add_argument("--max-len", type=int, default=32768, help="longest context to try (Qwen2.5: max_position_embeddings 32768)")
  ap.add_argument("--block-size", type=int, default=16)
  ap.add_argument("--reserves", default="5,6.5,8,10,14", help="GiB left out of the KV pool, tried in order")
  a = ap.parse_args()
  dev = Device["CUDA"]
  t0 = time.perf_counter()
  m = load_model(a.model, dtype=dtypes.bfloat16, device="CUDA", max_len=a.max_len, prefill_chunk=128)
  if a.quant: quantize_model(m, a.quant)
  dev.allocator.free_cache()
  c, bs = m.cfg, a.block_size
  wb, free0 = weight_bytes(m), dev.mem_free()
  block_bytes = 2 * c.n_layers * c.n_kv_heads * c.head_dim * bs * m.dtype.itemsize
  out: dict = {"model": a.model, "quant": a.quant, "load_s": round(time.perf_counter() - t0, 1), "weights_GiB": round(wb["device_total"] / GiB, 2),
               "free_after_load_GiB": round(free0 / GiB, 2), "kv_bytes_per_token": block_bytes // bs}
  print(json.dumps(out), flush=True)
  for r in (float(x) for x in a.reserves.split(",")):
    blocks = int((dev.mem_free() - r * GiB) // block_bytes)
    cap = blocks * bs
    L = min(a.max_len, cap - 2 * bs)
    eng = None
    try:
      t0 = time.perf_counter()
      eng = LLMEngine(m, num_blocks=blocks, block_size=bs, max_num_seqs=a.max_num_seqs, max_model_len=L)
      warmup(eng)
      tw = time.perf_counter() - t0
      n = a.max_num_seqs   # every row at once, together filling ~95% of the pool
      per = min(L - 8, int(0.95 * cap) // n - 8)
      t0 = time.perf_counter()
      res = eng.generate([[1000 + (i * 31 + j) % 20000 for j in range(per)] for i in range(n)], SamplingParams(max_tokens=8, ignore_eos=True))
      tb = time.perf_counter() - t0
      assert all(len(x) == 8 for x in res)
      t0 = time.perf_counter()   # one sequence as long as the pool allows
      res = eng.generate([[1000 + j % 20000 for j in range(L - 8)]], SamplingParams(max_tokens=8, ignore_eos=True))
      tl = time.perf_counter() - t0
      assert len(res[0]) == 8
      ok = {"reserve_GiB": r, "kv_blocks": blocks, "kv_tokens": cap, "kv_GiB": round(blocks * block_bytes / GiB, 2), "prefill_budget": eng.prefill_sizes[-1],
            "warmup_s": round(tw, 1), "batch": f"{n} x {per + 8} tokens in {tb:.1f}s", "max_context": L, "long_s": round(tl, 1),
            "free_after_GiB": round(dev.mem_free() / GiB, 2), "seqs_at_ctx": {k: cap // k for k in (768, 2048, 4096, 8192, 32768)}}
      print(json.dumps({**out, **ok}), flush=True)
      return 0
    except Exception as e:   # out of memory (MemoryError, or a failed launch/capture under memory pressure): next reserve
      print(json.dumps({"reserve_GiB": r, "kv_blocks": blocks, "error": f"{type(e).__name__}: {str(e)[:200]}"}), flush=True)
    finally:
      del eng
      gc.collect()
      dev.allocator.free_cache()
  return 1

if __name__ == "__main__": raise SystemExit(main())
