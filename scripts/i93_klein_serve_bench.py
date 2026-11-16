"""I93: served FLUX.2 [klein] latency per image: `fusezap serve-image flux2-klein` as a subprocess, then sequential
requests (one at a time, closed loop c1) through POST /v1/images/generations.

  FZ_KLEIN_PATH=<klein dir> python scripts/i93_klein_serve_bench.py --device NEURON --dtype bfloat16 --sizes 512,1024 \
      --requests 3 --steps 4 --out results/x [--port 8011] [--startup-s 1500] [-- extra serve-image args]
  # batch 4 through the images API (n=4: one request, four images, one batch-4 step) next to n=1, then a 60-minute soak:
  ... --batch 4 --n-list 1,4 --soak-minutes 60

Per request: the client's wall latency (HTTP in to PNG out) and the server's per-image timings (fusezap_timings: queue,
encode, denoise, decode, total); the first image of each size is saved as a PNG. The server warms up every size first
(--warmup-sizes: loads the NEFFs, the encoder and the VAE), so the requests measure the steady state. The server's own
log goes to <out>/serve.log. Writes <out>/serve_bench.json.

--batch B (I93 batch 4): the server runs with --max-batch B and --batch-sizes = the n values and B, and FZ_KLEIN_NEURON_BATCH=B
(klein on NEURON caps the batch at 1 otherwise: each batch size is its own set of NEFFs). --n-list: images per request (the
images API's n); each n gets --requests requests per size; an n > 1 request runs as one batch-n step per denoising step
(ImageEngine buckets the rows of one request together). JSON: sizes["<s>"] for n = 1 (as before), sizes["<s>_n<n>"] else,
with images per second (n / wall).
--soak-minutes M (I93 soak): after the measured requests, closed-loop requests for M minutes, cycling over every (size in
--soak-sizes, n in --soak-n) with a fixed seed per pair, so each image must come back byte-identical to the pair's first
(PNG sha256: determinism over the run); per request the wall, the server's timings, the server process group's RSS (ps) and
errors; the summary compares the first and the last third (latency drift), counts failures and mismatches, and keeps the
server's /metrics. The soak stops early if the server dies (recorded).
"""
from __future__ import annotations
import argparse, base64, hashlib, json, os, signal, statistics, subprocess, sys, time, urllib.request

def log(*a) -> None: print(*a, flush=True)

def post(url: str, body: dict, timeout: float) -> dict:
  req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read())

def group_rss_mib(pgid: int) -> float | None:
  """Resident MiB of every process in the server's process group (serve-image, its NEURON runtime worker, ...)."""
  try: out = subprocess.run(["ps", "-A", "-o", "pgid=,rss="], capture_output=True, text=True, timeout=10).stdout
  except Exception: return None
  kb = sum(int(b) for a, b in (ln.split() for ln in out.splitlines() if len(ln.split()) == 2) if a == str(pgid))
  return round(kb / 1024, 1)

def ints(s: str) -> list[int]: return [int(x) for x in s.split(",") if x.strip()]

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default="NEURON")
  ap.add_argument("--dtype", default="bfloat16")
  ap.add_argument("--sizes", default="512,1024")
  ap.add_argument("--requests", type=int, default=3)
  ap.add_argument("--steps", type=int, default=4)
  ap.add_argument("--port", type=int, default=8011)
  ap.add_argument("--startup-s", type=float, default=1500, help="give up if /health is not up by then (load + warm-up)")
  ap.add_argument("--request-timeout", type=float, default=900)
  ap.add_argument("--out", default="results/i93_serve")
  ap.add_argument("--prompt", default="a red fox standing in fresh snow at dawn, detailed fur, soft light")
  ap.add_argument("--batch", type=int, default=1, help="the server's max batch (serve-image --max-batch; FZ_KLEIN_NEURON_BATCH)")
  ap.add_argument("--n-list", default="1", help="images per request (the images API's n), each <= --batch")
  ap.add_argument("--soak-minutes", type=float, default=0.0)
  ap.add_argument("--soak-sizes", default="", help="default: --sizes")
  ap.add_argument("--soak-n", default="", help="default: --n-list")
  ap.add_argument("--soak-deadline", type=float, default=0.0, help="epoch seconds: the soak ends by then whatever --soak-minutes says (a job's budget)")
  ap.add_argument("extra", nargs="*", help="more serve-image arguments (after --)")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  sizes, nl = ints(a.sizes), ints(a.n_list)
  assert nl and all(1 <= n <= a.batch for n in nl), f"--n-list {nl} must be in [1, --batch {a.batch}]"
  bsz = sorted(set(nl) | {a.batch})
  cmd = [sys.executable, "-m", "fusezap", "serve-image", "flux2-klein", "--device", a.device, "--dtype", a.dtype, "--max-batch", str(a.batch),
         "--batch-sizes", ",".join(map(str, bsz)), "--port", str(a.port), "--warmup-sizes", ",".join(f"{s}x{s}" for s in sizes), *a.extra]
  env = dict(os.environ, **({"FZ_KLEIN_NEURON_BATCH": str(a.batch)} if a.batch > 1 else {}))
  rep: dict = {"cmd": cmd, "env": {k: v for k, v in env.items() if k.startswith(("FZ_KLEIN", "FZ_NKI", "FZ_NEURON"))}, "batch": a.batch,
               "n_list": nl, "sizes": {}}
  def save() -> None:
    with open(os.path.join(a.out, "serve_bench.json"), "w") as f: json.dump(rep, f, indent=1)
  log("serve:", " ".join(cmd))
  t0 = time.time()
  srv = subprocess.Popen(cmd, stdout=open(os.path.join(a.out, "serve.log"), "w"), stderr=subprocess.STDOUT, start_new_session=True, env=env)
  base = f"http://127.0.0.1:{a.port}"
  def request(s: int, n: int, seed: int) -> tuple[dict, float]:
    st = time.perf_counter()
    j = post(base + "/v1/images/generations", {"prompt": a.prompt, "size": f"{s}x{s}", "seed": seed, "num_inference_steps": a.steps, "n": n,
                                               "output_format": "png"}, a.request_timeout)
    return j, time.perf_counter() - st
  try:
    while True:   # load + warm-up, then uvicorn answers /health
      if srv.poll() is not None: raise SystemExit(f"serve-image exited with {srv.returncode} (see {a.out}/serve.log)")
      try:
        with urllib.request.urlopen(base + "/health", timeout=5) as r:
          if r.status == 200: break
      except Exception: pass
      if time.time() - t0 > a.startup_s: raise SystemExit(f"serve-image not up after {a.startup_s} s")
      time.sleep(5)
    rep["startup_s"] = round(time.time() - t0, 1)
    rep["rss_mib_after_startup"] = group_rss_mib(srv.pid)
    log(f"server up in {rep['startup_s']} s (load + warm-up of {sizes}, batch sizes {bsz}); server RSS {rep['rss_mib_after_startup']} MiB")
    seed = 100
    for s in sizes:
      for n in nl:
        key = str(s) if n == 1 else f"{s}_n{n}"
        rows = []
        for i in range(a.requests):
          seed += n
          j, wall = request(s, n, seed)
          ds = j["data"]
          assert len(ds) == n, f"asked n={n}, got {len(ds)} images"
          tms = [d.get("fusezap_timings", {}) for d in ds]
          rows.append({"seed": seed, "n": n, "wall_s": round(wall, 3), "images_per_s": round(n / wall, 4),
                       **{k: round(v, 3) for k, v in tms[0].items()}, **({"timings": [{k: round(v, 3) for k, v in t.items()} for t in tms]} if n > 1 else {})})
          if i == 0:
            for k, d in enumerate(ds):
              with open(os.path.join(a.out, f"served_{s}.png" if n == 1 else f"served_{s}_n{n}_{k}.png"), "wb") as f: f.write(base64.b64decode(d["b64_json"]))
          log(f"{s}^2 n={n} request {i}: {wall:.2f} s wall ({n / wall * 60:.2f} images/min), timings {tms[0]}")
          rep["sizes"][key] = {"n": n, "requests": rows}
          save()
        walls = [r["wall_s"] for r in rows]
        summ = {"median_wall_s": round(statistics.median(walls), 3), "min_wall_s": min(walls), "median_s_per_image": round(statistics.median(walls) / n, 3),
                "images_per_min": round(n / statistics.median(walls) * 60, 3)}
        for k in ("encode_s", "denoise_s", "decode_s", "total_s"):
          if all(k in r for r in rows): summ[f"median_{k}"] = round(statistics.median(r[k] for r in rows), 3)
        if "median_denoise_s" in summ: summ["denoise_s_per_step"] = round(summ["median_denoise_s"] / a.steps, 3)
        rep["sizes"][key]["summary"] = summ
        log(f"{s}^2 n={n} served, {a.steps} steps: median {summ['median_wall_s']} s per request, {summ['median_s_per_image']} s per image (" +
            ", ".join(f"{k} {v}" for k, v in summ.items() if k.startswith("median_") and k not in ("median_wall_s", "median_s_per_image")) + ")")
        save()
    if a.soak_minutes > 0: soak(a, srv, request, rep, save)
    try:
      with urllib.request.urlopen(base + "/metrics", timeout=10) as r: rep["metrics"] = r.read().decode()[-4000:]
    except Exception as e: rep["metrics_error"] = str(e)
  finally:
    try: os.killpg(srv.pid, signal.SIGTERM); srv.wait(timeout=60)
    except Exception:
      try: os.killpg(srv.pid, signal.SIGKILL)
      except Exception: pass
    save()
  log("I93 klein served: " + "; ".join(f"{k} median {v['summary']['median_wall_s']} s/request" for k, v in rep["sizes"].items() if "summary" in v))
  if "soak" in rep: log("I93 klein soak: " + json.dumps(rep["soak"]["summary"]))

def soak(a, srv, request, rep: dict, save) -> None:
  """Closed-loop requests for a.soak_minutes over every (size, n) pair, each pair with its own fixed seed: every image must
  equal the pair's first (sha256 of the PNG). Rows and a summary in rep["soak"]."""
  sizes, ns = ints(a.soak_sizes or a.sizes), ints(a.soak_n or a.n_list)
  pairs = [(s, n) for s in sizes for n in ns]
  first: dict = {}
  rows: list = []
  rep["soak"] = {"minutes": a.soak_minutes, "pairs": pairs, "rows": rows}
  t0 = time.time()
  end = t0 + a.soak_minutes * 60
  if a.soak_deadline: end = min(end, a.soak_deadline)
  rep["soak"]["planned_minutes"] = round((end - t0) / 60, 1)
  k = 0
  log(f"soak: {rep['soak']['planned_minutes']} min (asked {a.soak_minutes}) over {pairs} (fixed seed per pair: images must repeat byte for byte)")
  while time.time() < end:
    if srv.poll() is not None:
      rep["soak"]["server_died"] = f"exit {srv.returncode} after {time.time() - t0:.0f} s"; log("soak: server died:", rep["soak"]["server_died"]); break
    s, n = pairs[k % len(pairs)]
    k += 1
    row: dict = {"t_s": round(time.time() - t0, 1), "size": s, "n": n}
    try:
      j, wall = request(s, n, 7000 + 10 * s + n)
      hs = [hashlib.sha256(base64.b64decode(d["b64_json"])).hexdigest() for d in j["data"]]
      if (s, n) not in first: first[(s, n)] = hs
      row.update(ok=len(hs) == n, wall_s=round(wall, 3), same=hs == first[(s, n)],
                 **{k_: round(v, 3) for k_, v in j["data"][0].get("fusezap_timings", {}).items()})
    except Exception as e:
      row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
    row["rss_mib"] = group_rss_mib(srv.pid)
    rows.append(row)
    log(f"soak {row['t_s']:7.1f} s {s}^2 n={n}: " + (f"{row['wall_s']} s, same images {row['same']}" if row.get("ok") else f"FAILED {row.get('error')}") +
        f", RSS {row['rss_mib']} MiB")
    if len(rows) % 4 == 0: save()
  ok = [r for r in rows if r.get("ok")]
  summ: dict = {"requests": len(rows), "images": sum(r["n"] for r in ok), "failures": len(rows) - len(ok),
                "mismatched": sum(1 for r in ok if not r.get("same")), "wall_minutes": round((time.time() - t0) / 60, 1)}
  for s, n in pairs:
    rs = [r for r in ok if r["size"] == s and r["n"] == n]
    if len(rs) >= 3:
      third = max(1, len(rs) // 3)
      f, l = statistics.median(r["wall_s"] for r in rs[:third]), statistics.median(r["wall_s"] for r in rs[-third:])
      summ[f"{s}_n{n}"] = {"requests": len(rs), "median_wall_s": round(statistics.median(r["wall_s"] for r in rs), 3),
                           "p95_wall_s": round(sorted(r["wall_s"] for r in rs)[min(len(rs) - 1, int(0.95 * len(rs)))], 3),
                           "first_third_median_s": round(f, 3), "last_third_median_s": round(l, 3), "drift_pct": round(100 * (l - f) / f, 2)}
  rss = [r["rss_mib"] for r in rows if r.get("rss_mib") is not None]
  if rss: summ["rss_mib"] = {"first": rss[0], "max": max(rss), "last": rss[-1]}
  if "server_died" in rep["soak"]: summ["server_died"] = rep["soak"]["server_died"]
  rep["soak"]["summary"] = summ
  save()

if __name__ == "__main__": main()
