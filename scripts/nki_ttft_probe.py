#!/usr/bin/env python3
"""I75 prewarm check against a freshly started `fusezap serve` (before any other request): streamed /v1/completions with
token-id prompts of several lengths (default 100, 200, 400, 700 tokens: on a 1024-slot pool the prefill's pool prefixes
128, 256, 512 and the whole pool), each length twice in a row. The first request of a length is the first use of its
prefix bucket after start; with every prefill NEFF compiled and loaded at prewarm it must be within --ratio (default 2x)
of the second (steady) one, plus --slack ms (default 50: one decode step and host noise). Prints one line per request and
a JSON verdict; exits 0 iff every length passes (stdlib only).

  python3 scripts/nki_ttft_probe.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen2.5-7B-Instruct --json out.json
"""
from __future__ import annotations
import argparse, json, sys, time, urllib.request

def ttft_ms(base: str, model: str, n: int, salt: int, timeout: float) -> float:
  body = {"model": model, "prompt": [1000 + (salt * 7919 + i * 31) % 20000 for i in range(n)], "max_tokens": 4,
          "temperature": 0, "stream": True, "ignore_eos": True}
  req = urllib.request.Request(base.rstrip("/") + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
  t0 = time.perf_counter()
  with urllib.request.urlopen(req, timeout=timeout) as r:
    for line in r:
      line = line.strip()
      if not line.startswith(b"data:") or line == b"data: [DONE]": continue
      ch = json.loads(line[5:])
      if ch.get("choices"): return (time.perf_counter() - t0) * 1e3
  return (time.perf_counter() - t0) * 1e3

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-url", default="http://127.0.0.1:8000")
  ap.add_argument("--model", required=True)
  ap.add_argument("--lengths", default="100,200,400,700")
  ap.add_argument("--ratio", type=float, default=2.0)
  ap.add_argument("--slack", type=float, default=50.0)
  ap.add_argument("--timeout", type=float, default=900.0)
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  rows, ok = [], True
  for k, n in enumerate(int(x) for x in a.lengths.split(",") if x.strip()):
    try: first, steady = ttft_ms(a.base_url, a.model, n, 2 * k, a.timeout), ttft_ms(a.base_url, a.model, n, 2 * k + 1, a.timeout)
    except Exception as e:
      print(f"prewarm probe: {n} tokens: request failed: {e}"); rows.append({"tokens": n, "error": str(e)}); ok = False; continue
    good = first <= a.ratio * steady + a.slack
    ok &= good
    rows.append({"tokens": n, "first_ms": round(first, 1), "steady_ms": round(steady, 1), "ok": good})
    print(f"prewarm probe: {n:>4} tokens: first TTFT {first:8.1f} ms, steady {steady:7.1f} ms ({first / max(steady, 1e-9):.2f}x) {'ok' if good else 'NOT PREWARMED'}")
  out = {"ok": ok, "ratio": a.ratio, "slack_ms": a.slack, "rows": rows}
  print(f"prewarm probe: {'PASS' if ok else 'FAIL'}: every first request within {a.ratio}x its steady TTFT + {a.slack} ms" if ok else
        f"prewarm probe: FAIL: {[r['tokens'] for r in rows if not r.get('ok')]} token prompts paid a compile or load on first use")
  if a.json:
    with open(a.json, "w") as f: json.dump(out, f, indent=1)
  return 0 if ok else 1

if __name__ == "__main__": sys.exit(main())
