"""Steady-state load windows for profiling a running server (docs/status/SPEED.md "Run 4"; a tool, not the harness).

  decode:  --conc N streams (512-token prompts, ignore_eos) are started; once every stream has its first token (all
           prefills done) and --settle seconds passed, --start-cmd runs (e.g. `nsys start`), the window lasts --window
           seconds, --stop-cmd runs, then the streams are dropped (both servers abort a disconnected request).
  prefill: --start-cmd, then --repeat rounds of --conc prompts sent together with max_tokens 1 (one prefill step of
           conc x prompt rows on FuseZap's 2048-row budget), then --stop-cmd.

Prompts are random token ids seeded by --label, so no window repeats another's prompt (no prefix-cache hits on vLLM).
One JSON line per window goes to --out: wall-clock bounds (time.time(), to slice FZ_STEP_PROF steps), streamed chunks
in the window (one per token on both servers), output tok/s, the per-stream inter-chunk gaps (p50 / p90 / p99), and
whether a stream finished inside the window (then --max-tokens was too small).

  python bench/prof_window.py --base-url http://127.0.0.1:18003 --model Qwen/Qwen2.5-7B-Instruct --label fz-c32 \\
      --mode decode --conc 32 --window 2 --start-cmd "nsys start ..." --stop-cmd "nsys stop ..." --out windows.jsonl
"""
from __future__ import annotations
import argparse, asyncio, json, sys, time, zlib
import numpy as np

async def _stream(http, url: str, body: dict, times: list[float], first: asyncio.Event | None, done: list[bool]) -> None:
  async with http.stream("POST", url, json=body) as resp:
    if resp.status_code != 200: raise RuntimeError(f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}")
    async for line in resp.aiter_lines():
      if not line.startswith("data:"): continue
      data = line[5:].strip()
      if data == "[DONE]": break
      ev = json.loads(data)
      if any(ch.get("text") for ch in ev.get("choices") or []):
        times.append(time.time())
        if first is not None and not first.is_set(): first.set()
  done[0] = True

async def _shell(cmd: str) -> float:
  if not cmd: return 0.0
  t = time.time()
  p = await asyncio.create_subprocess_shell(cmd)
  await p.wait()
  return time.time() - t

def _prompt(rng, n: int) -> list[int]: return rng.integers(1000, 30000, n).tolist()

async def run(a: argparse.Namespace) -> dict:
  import httpx
  rng = np.random.default_rng(zlib.crc32(a.label.encode()))
  url = a.base_url.rstrip("/") + "/v1/completions"
  rec: dict = {"label": a.label, "mode": a.mode, "conc": a.conc, "prompt": a.prompt}
  async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0), limits=httpx.Limits(max_connections=None)) as http:
    if a.mode == "prefill":
      rec["start_cmd_s"] = await _shell(a.start_cmd)
      rec["t0"] = time.time()
      rounds = []
      for _ in range(a.repeat):
        t = time.time()
        await asyncio.gather(*[_stream(http, url, {"model": a.model, "prompt": _prompt(rng, a.prompt), "max_tokens": 1, "temperature": 0.0,
                                                    "ignore_eos": True, "stream": True}, [], None, [False]) for _ in range(a.conc)])
        rounds.append(time.time() - t)
      rec["t1"] = time.time()
      rec["stop_cmd_s"] = await _shell(a.stop_cmd)
      rec["round_s"] = [round(x, 4) for x in rounds]
      rec["round_s_median"] = round(float(np.median(rounds)), 4)
      return rec
    times = [[] for _ in range(a.conc)]
    firsts = [asyncio.Event() for _ in range(a.conc)]
    dones = [[False] for _ in range(a.conc)]
    body = lambda: {"model": a.model, "prompt": _prompt(rng, a.prompt), "max_tokens": a.max_tokens, "temperature": 0.0, "ignore_eos": True, "stream": True}
    tasks = [asyncio.create_task(_stream(http, url, body(), times[i], firsts[i], dones[i])) for i in range(a.conc)]
    t_send = time.time()
    try:
      await asyncio.wait_for(asyncio.gather(*[f.wait() for f in firsts]), timeout=a.first_timeout)
    except asyncio.TimeoutError:
      rec["error"] = "not every stream got its first token"
    rec["all_first_s"] = round(time.time() - t_send, 4)
    await asyncio.sleep(a.settle)
    rec["start_cmd_s"] = await _shell(a.start_cmd)
    t0 = time.time()
    await asyncio.sleep(a.window)
    t1 = time.time()
    rec["stop_cmd_s"] = await _shell(a.stop_cmd)
    rec["finished_early"] = sum(1 for d, ts in zip(dones, times) if d[0] and ts and ts[-1] < t1)
    for t in tasks: t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
  n = sum(sum(1 for x in ts if t0 <= x <= t1) for ts in times)
  gaps = np.array([g for ts in times for g in np.diff([x for x in ts if t0 <= x <= t1])] or [np.nan]) * 1e3
  rec.update({"t0": round(t0, 6), "t1": round(t1, 6), "window_s": round(t1 - t0, 4), "chunks": n, "tok_s": round(n / (t1 - t0), 2),
              "itl_ms_p50": round(float(np.nanpercentile(gaps, 50)), 3), "itl_ms_p90": round(float(np.nanpercentile(gaps, 90)), 3),
              "itl_ms_p99": round(float(np.nanpercentile(gaps, 99)), 3)})
  await asyncio.sleep(a.drain)   # let the server abort the dropped streams before the next window
  return rec

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-url", required=True)
  ap.add_argument("--model", required=True, help="the served model name")
  ap.add_argument("--label", required=True)
  ap.add_argument("--mode", choices=("decode", "prefill"), default="decode")
  ap.add_argument("--conc", type=int, default=32)
  ap.add_argument("--prompt", type=int, default=512)
  ap.add_argument("--max-tokens", type=int, default=1200)
  ap.add_argument("--window", type=float, default=2.0)
  ap.add_argument("--settle", type=float, default=0.5)
  ap.add_argument("--repeat", type=int, default=5)
  ap.add_argument("--first-timeout", type=float, default=120.0)
  ap.add_argument("--drain", type=float, default=2.0)
  ap.add_argument("--start-cmd", default="")
  ap.add_argument("--stop-cmd", default="")
  ap.add_argument("--out", default="")
  a = ap.parse_args()
  rec = asyncio.run(run(a))
  line = json.dumps(rec)
  print(line, flush=True)
  if a.out:
    with open(a.out, "a") as f: f.write(line + "\n")
  return 1 if rec.get("error") else 0

if __name__ == "__main__": sys.exit(main())
