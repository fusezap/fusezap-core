"""I91 (I81 D9): the NKI kernels the FLUX.2 [klein] VAE decoder and the long attentions need, rendered for their real shapes
and (on Linux with .venv-nki) compiled for inf2 / trn1, one NEFF each, with seconds, NEFF bytes and failures.

  PYTHONPATH=. python scripts/i91_kernels.py --sizes 512,1024 --json out.json                       # Mac: the list + source sizes
  PYTHONPATH=. python scripts/i91_kernels.py --sizes 512 --compile --target inf2 --jobs 8 --json out.json   # pod / instance

Kernels (VAE: AutoencoderKLFlux2's decoder from klein's vae/config.json, block_out_channels 128/256/512/512, 2 + 1 resnets
per up block, latent 32 channels at size / 8; every conv stride 1):
  conv    nki_conv.render_conv per distinct (C -> O, k x k, H x W): post_quant_conv 1x1, conv_in, the mid block's and up blocks'
          resnet convs, the 1x1 shortcuts where channels change, the upsamplers' convs (at twice the resolution), conv_out;
  attn    nki_kernels.AttnSpec flash attention: the VAE mid block (1 head, S = L = (size / 8)^2, D = Dv = 512) and klein's
          transformer attention at that size (24 heads, S = L = (size / 16)^2 + 512, D = Dv = 128).
The conv kernels unroll their pixel tiles at trace time (nki_conv's module doc), so the big ones are big programs; the
compile runs smallest first (by unrolled matmul count, an estimate of the program size) so a timeout keeps the small ones.
"""
from __future__ import annotations
import argparse, json, math, os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

VAE = {"block_out_channels": [128, 256, 512, 512], "layers_per_block": 2, "latent_channels": 32, "out_channels": 3}

def vae_convs(size: int, cfg: dict = VAE) -> list[tuple[int, int, int, int]]:
  """Distinct (C, O, k, H) convs of the decoder at image size `size` (square; H = W), stride 1, padding k // 2."""
  ch = list(reversed(cfg["block_out_channels"]))
  h = size // 8
  out = [(cfg["latent_channels"], cfg["latent_channels"], 1, h), (cfg["latent_channels"], ch[0], 3, h), (ch[0], ch[0], 3, h)]   # post_quant, conv_in, mid
  cin = ch[0]
  for b, c in enumerate(ch):
    for r in range(cfg["layers_per_block"] + 1):
      out.append((cin, c, 3, h)); out.append((c, c, 3, h))
      if cin != c: out.append((cin, c, 1, h))
      cin = c
    if b < len(ch) - 1: h *= 2; out.append((c, c, 3, h))   # upsample (nearest 2x) then conv
  out.append((cin, cfg["out_channels"], 3, h))
  return sorted(set(out), key=out.index)

def conv_params(C: int, O: int, k: int, H: int) -> tuple[int, ...]:
  from fusezap.schedule import CONV_PARAMS
  p = dict(M=O, N=H * H, K=C * k * k, sam=C * k * k, sak=1, oa=0, so=H * H, sp=1, C=C, H=H, W=H, Wo=H, kh=k, kw=k, sh=1, sw=1, dh=1, dw=1,
           ph=k // 2, pw=k // 2)
  return tuple(p[n] for n in CONV_PARAMS)

def attn_spec(heads: int, S: int, D: int):
  from fusezap.codegen.nki_kernels import AttnSpec
  q = (heads * S * D, S * D, D, 1, 0)
  return AttnSpec((1, heads), (1, S), S, D, D, q, q, q, q, 1 / math.sqrt(D))

def kernels(sizes: list[int]) -> list[dict]:
  from fusezap.codegen.nki_conv import render_conv
  out = []
  for s in sizes:
    for C, O, k, H in vae_convs(s):
      p = conv_params(C, O, k, H)
      prog = render_conv(p)
      tiles = H * H // 512 if H <= 512 else H * -(-H // 512)
      work = -(-O // 128) * max(1, tiles) * k * k * -(-C // 128)   # nc_matmul instructions after unrolling (the compile's size)
      out.append({"size": s, "kind": "conv", "what": f"{C}->{O} {k}x{k} at {H}x{H}", "prog": prog, "matmuls": work,
                  "specs": [(O * H * H, "<f4"), (O * C * k * k, "<f4"), (C * H * H, "<f4")]})
    for what, heads, S, D in (("VAE mid-block attention", 1, (s // 8) ** 2, 512), ("klein transformer attention", 24, (s // 16) ** 2 + 512, 128)):
      sp = attn_spec(heads, S, D)
      n = heads * S * D
      out.append({"size": s, "kind": "attn", "what": f"{what}: {heads} x S {S} D {D}", "prog": sp.render(), "specs": [(n, "<f4")] * 4,
                  "matmuls": heads * -(-S // 128) * -(-S // 512) * (-(-D // 128) + 4 + 4 * -(-D // 512))})   # rough: QKᵀ, P transposes, P·V
  seen, uniq = set(), []
  for r in out:
    key = (r["prog"].name, r["prog"].src)
    if key not in seen: seen.add(key); uniq.append(r)
  for r in uniq: r["src_lines"] = r["prog"].src.count("\n")
  return uniq

def compile_all(recs: list[dict], target: str, jobs: int, log, save) -> None:
  from fusezap.device import neuron as N
  versions = N._nki_versions()
  q, lock = sorted(recs, key=lambda r: r["matmuls"]), threading.Lock()   # smallest unrolled program first
  def loop() -> None:
    w = N.compile_worker() if jobs > 1 else None
    if w is not None: N._TLS.worker = w
    try:
      while True:
        with lock:
          if not q: return
          r = q.pop(0)
        t0 = time.perf_counter()
        try:
          path, _ = N.cached_neff(r["prog"], r["specs"], target, versions)
          r.update(ok=True, seconds=round(time.perf_counter() - t0, 1), neff_bytes=path.stat().st_size)
        except Exception as e:
          r.update(ok=False, seconds=round(time.perf_counter() - t0, 1), error=f"{type(e).__name__}: {e}"[:400])
        log(f"NEFF {r['size']}^2 {r['kind']} {r['what']} ({r['prog'].name}, {r['src_lines']} lines): "
            + (f"ok {r['seconds']} s, {r['neff_bytes'] / 2**20:.1f} MiB" if r["ok"] else f"FAILED after {r['seconds']} s: {r['error'][:300]}"))
        with lock: save()
    finally:
      N._TLS.worker = None
      if w is not None:
        try: w.close()
        except Exception: pass
  ts = [threading.Thread(target=loop) for _ in range(max(1, jobs))]
  for t in ts: t.start()
  for t in ts: t.join()

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--sizes", default="512,1024")
  ap.add_argument("--compile", action="store_true")
  ap.add_argument("--target", default=os.environ.get("FZ_NEURON_TARGET", "inf2"))
  ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 4))
  ap.add_argument("--kinds", default="conv,attn")
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  log = lambda *m: print(*m, flush=True)
  recs = [r for r in kernels([int(s) for s in a.sizes.split(",")]) if r["kind"] in a.kinds.split(",")]
  def save() -> None:
    if a.json:
      with open(a.json, "w") as f:
        json.dump({"sizes": a.sizes, "target": a.target if a.compile else None,
                   "kernels": [{k: v for k, v in r.items() if k != "prog"} | {"name": r["prog"].name} for r in recs]}, f, indent=1)
  for r in recs: log(f"I91 kernel {r['size']}^2 {r['kind']:<4} {r['what']:<48} {r['prog'].name} (~{r['matmuls']} matmuls unrolled)")
  log(f"I91: {len(recs)} distinct kernels")
  if a.compile:
    t0 = time.perf_counter()
    compile_all(recs, a.target, a.jobs, log, save)
    ok = [r for r in recs if r.get("ok")]
    log(f"I91 compile {a.target}: {len(ok)}/{len(recs)} ok, {time.perf_counter() - t0:.0f} s wall at {a.jobs} jobs, "
        f"NEFFs {sum(r.get('neff_bytes', 0) for r in recs) / 2**20:.0f} MiB")
  save()

if __name__ == "__main__": main()
