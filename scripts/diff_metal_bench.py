"""I81 D8 timings on METAL: flash attention (render_attention_flash) and the implicit-GEMM conv (render_conv) at the FLUX.2
[klein] shapes, against the composed path (FZ_METAL_FLASH=0: the earlier attention kernels; FZ_METAL_CONV=0: im2col +
GEMM) and torch MPS (float32), each the best of --reps warm runs (wall time around a device sync).

  uv run python scripts/diff_metal_bench.py [--out docs/diff/d8_bench.json] [--reps 5] [--only attn|conv]

Each variant runs in its own subprocess (env flags are read when a kernel is first built). The composed conv is skipped
where its im2col buffer would exceed --max-im2col-gb (default 5). The old attention kernel is skipped at D=512 (its
per-row fallback takes minutes at 16384 tokens).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time

ATTN = [  # (name, B*heads, L=S, D, Dv)
  ("klein joint attention 512^2 (24 x 1536 x 128)", 24, 1536, 128, 128),
  ("klein joint attention 1024^2 (24 x 4608 x 128)", 24, 4608, 128, 128),
  ("VAE mid-block 512^2 (1 x 4096 x 512)", 1, 4096, 512, 512),
  ("VAE mid-block 1024^2 (1 x 16384 x 512)", 1, 16384, 512, 512),
]
CONV = [  # (name, C, O, H): 3x3 s1 p1 at the decoder's resolutions for a 1024^2 image
  ("conv_in 32->512 @128", 32, 512, 128),
  ("resnet 512->512 @128", 512, 512, 128),
  ("resnet 512->512 @256", 512, 512, 256),
  ("resnet 512->512 @512", 512, 512, 512),
  ("resnet 512->256 @512", 512, 256, 512),
  ("resnet 256->256 @1024", 256, 256, 1024),
  ("resnet 256->128 @1024", 256, 128, 1024),
  ("resnet 128->128 @1024", 128, 128, 1024),
  ("conv_out 128->3 @1024", 128, 3, 1024),
]

def _time(fn, sync, reps: int) -> float:
  fn(); sync()
  best = float("inf")
  for _ in range(reps):
    sync(); t0 = time.perf_counter(); fn(); sync(); best = min(best, time.perf_counter() - t0)
  return best

def run_variant(kind: str, variant: str, reps: int, max_gb: float) -> dict:
  import numpy as np
  out: dict = {}
  r = np.random.default_rng(0)
  if variant == "torch":
    import torch
    sync = torch.mps.synchronize
    if kind == "attn":
      for name, B, L, D, Dv in ATTN:
        q, k, v = [torch.tensor(r.standard_normal(s).astype(np.float32), device="mps") for s in ((B, L, D), (B, L, D), (B, L, Dv))]
        out[name] = _time(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v), sync, reps)
    else:
      for name, C, O, H in CONV:
        x = torch.tensor(r.standard_normal((1, C, H, H)).astype(np.float32), device="mps")
        w = torch.tensor((r.standard_normal((O, C, 3, 3)) * 0.05).astype(np.float32), device="mps")
        out[name] = _time(lambda: torch.nn.functional.conv2d(x, w, padding=1), sync, reps)
    return out
  from fusezap import Tensor
  from fusezap.device import Device
  dev = Device["METAL"]
  if kind == "attn":
    for name, B, L, D, Dv in ATTN:
      if variant == "old" and D > 128: continue
      q, k, v = [Tensor(r.standard_normal(s).astype(np.float32), device="METAL").realize() for s in ((B, L, D), (B, L, D), (B, L, Dv))]
      out[name] = _time(lambda: q.scaled_dot_product_attention(k, v).realize(), dev.synchronize, reps)
  else:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))
    from test_metal_flash import conv2d_composed
    for name, C, O, H in CONV:
      if variant == "composed" and C * 9 * H * H * 4 / 1e9 > max_gb: continue
      x = Tensor(r.standard_normal((1, C, H, H)).astype(np.float32), device="METAL").realize()
      w = Tensor((r.standard_normal((O, C, 3, 3)) * 0.05).astype(np.float32), device="METAL").realize()
      out[name] = _time(lambda: conv2d_composed(x, w, 1, 1).realize(), dev.synchronize, reps)
  return out

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="docs/diff/d8_bench.json")
  ap.add_argument("--reps", type=int, default=5)
  ap.add_argument("--only", default=None, choices=["attn", "conv"])
  ap.add_argument("--max-im2col-gb", type=float, default=5.0)
  ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
  ap.add_argument("--kind", default=None, help=argparse.SUPPRESS)
  a = ap.parse_args()
  if a.variant:
    print(json.dumps(run_variant(a.kind, a.variant, a.reps, a.max_im2col_gb)))
    return
  plans = {"attn": [("fz flash", "fz", {}), ("fz old kernels", "old", {"FZ_METAL_FLASH": "0"}), ("torch mps", "torch", {})],
           "conv": [("fz conv kernel", "fz", {}), ("fz composed (im2col + GEMM)", "composed", {"FZ_METAL_CONV": "0"}), ("torch mps", "torch", {})]}
  res: dict = {"date": time.strftime("%Y-%m-%d %H:%M"), "reps": a.reps, "unit": "seconds (best of reps)", "attn": {}, "conv": {}}
  import platform
  res["machine"] = platform.platform()
  for kind in ([a.only] if a.only else ["attn", "conv"]):
    for label, variant, env in plans[kind]:
      p = subprocess.run([sys.executable, __file__, "--variant", variant, "--kind", kind, "--reps", str(a.reps),
                          "--max-im2col-gb", str(a.max_im2col_gb)], env={**os.environ, **env}, capture_output=True, text=True)
      if p.returncode: print(p.stderr[-3000:], file=sys.stderr); raise SystemExit(f"{kind} {variant} failed")
      for name, t in json.loads(p.stdout.strip().splitlines()[-1]).items(): res[kind].setdefault(name, {})[label] = t
      print(f"[{time.strftime('%H:%M:%S')}] {kind} {label} done", flush=True)
  for kind, rows in (("attn", ATTN), ("conv", CONV)):
    for row in rows:
      name = row[0]
      if name not in res[kind]: continue
      fl = 2 * row[1] * row[2] * row[2] * (row[3] + row[4]) if kind == "attn" else 2 * row[1] * row[2] * 9 * row[3] * row[3]
      cells = "  ".join(f"{k}: {v * 1e3:8.1f} ms ({fl / v / 1e12:4.2f} TF/s)" for k, v in res[kind][name].items())
      print(f"{name:52s} {cells}")
  os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
  json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__": main()
