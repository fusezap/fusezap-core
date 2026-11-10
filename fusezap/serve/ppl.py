"""WikiText-2 perplexity of any fusezap.serve model and weight format (bench/bench_ppl.py's protocol, any device/dtype).

  python -m fusezap.serve.ppl --model Qwen/Qwen2.5-7B-Instruct --device CUDA --dtype bfloat16 --modes bf16,int8,fp8 --out results/ppl.json

The protocol, the parquet reader and the NLL step are bench/bench_ppl.py's (imported from the repo checkout): the raw test
split joined with "\\n\\n", the first --tokens predictions, windows of --ctx every --stride, each window from position 0.
Mode "bf16"/"f16"/"f32" is the unquantized model in --dtype; anything else goes through fusezap.serve.quant.quantize_model.
Prints a markdown table and writes JSON (per mode: ppl, scored tokens, seconds, weight bytes)."""
from __future__ import annotations
import argparse, gc, importlib.util, json, os, sys, time
import numpy as np

def _bench_ppl():
  fn = os.path.join(os.path.dirname(__file__), "..", "..", "bench", "bench_ppl.py")
  spec = importlib.util.spec_from_file_location("fz_bench_ppl", fn)
  assert spec is not None and spec.loader is not None, f"needs the repo checkout ({fn})"
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod

def main(argv: list[str] | None = None) -> int:
  ap = argparse.ArgumentParser(prog="python -m fusezap.serve.ppl")
  ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
  ap.add_argument("--device", default=os.environ.get("FZ_DEVICE"))
  ap.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
  ap.add_argument("--modes", default="bf16,int8,fp8")
  ap.add_argument("--tokens", type=int, default=32768)
  ap.add_argument("--ctx", type=int, default=1024)
  ap.add_argument("--stride", type=int, default=512)
  ap.add_argument("--out", default=None)
  a = ap.parse_args(argv)
  bp = _bench_ppl()
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  from fusezap.serve.tokenizer import Tokenizer
  from fusezap.serve.weights import resolve_model
  from fusezap.serve.quant import quantize_model, weight_bytes
  path = resolve_model(a.model)
  ids = np.asarray(Tokenizer.from_pretrained(path).encode(bp.wikitext2_test()), np.int64)[:a.tokens + 1]
  dt = getattr(dtypes, a.dtype)
  rows: list[dict] = []
  for mode in a.modes.split(","):
    t0 = time.perf_counter()
    m = load_model(a.model, dtype=dt, device=a.device, max_len=a.ctx, prefill_chunk=16)
    if mode not in ("bf16", "f16", "f32"): quantize_model(m, mode)
    tq = time.perf_counter() - t0
    print(f"{mode}: loaded{' + quantized' if mode not in ('bf16', 'f16', 'f32') else ''} in {tq:.1f}s", flush=True)
    ppl, n, sec = bp.perplexity(m, ids, a.ctx, a.stride, log=lambda s: print(s, flush=True))
    wb = weight_bytes(m)
    rows.append(dict(mode=mode, ppl=ppl, tokens=n, seconds=sec, load_seconds=tq, **wb))
    print(f"{mode}: ppl {ppl:.4f} over {n} tokens in {sec:.1f}s; linear {wb['linear'] / 2**30:.2f} GiB, device {wb['device_total'] / 2**30:.2f} GiB", flush=True)
    del m
    gc.collect()
  base = rows[0]["ppl"]
  print(f"\n{a.model}, {m_dev(a.device)}, activations {a.dtype}, {rows[0]['tokens']} scored tokens, ctx {a.ctx}, stride {a.stride}")
  print("| weights | perplexity | Δ vs first | linear weights | device weights (total) | eval time |\n|---|---|---|---|---|---|")
  for r in rows:
    d = "—" if r is rows[0] else f"{(r['ppl'] / base - 1) * 100:+.2f}%"
    print(f"| {r['mode']} | {r['ppl']:.4f} | {d} | {r['linear'] / 2**30:.2f} GiB | {r['device_total'] / 2**30:.2f} GiB | {r['seconds']:.0f} s |")
  if a.out:
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f: json.dump(dict(model=a.model, device=m_dev(a.device), dtype=a.dtype, ctx=a.ctx, stride=a.stride, rows=rows), f, indent=1)
  return 0

def m_dev(device: str | None) -> str:
  from fusezap.device import Device
  d = Device.canonicalize(device)
  return f"{d} ({getattr(Device[d], 'gpu_name', d)})"

if __name__ == "__main__": sys.exit(main())
