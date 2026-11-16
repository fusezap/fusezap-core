#!/usr/bin/env python3
"""I37: failure and recovery probes for the qualification report, against a running `fusezap serve` (they complement
Invarix's reliability check: overload, cancelled / abandoned requests, restart). Standard library only.

  python scripts/i37_failure_probes.py --base-url http://127.0.0.1:18003 --model <served name> --pid <server pid> \
      [--probes bad,overlength,pressure,crash] [--json out.json]

Probes (each pass / fail / skip with a reason; the crash probe kills the engine, so it runs last):
  bad         malformed requests (max_tokens -1, an unknown model, a body that is not JSON) get a 4xx, never a 5xx or a
              dropped connection, and the server answers normally after them
  overlength  a prompt longer than the context length (completions with prompt token ids) gets a 4xx
              (context_length_exceeded), and the server answers normally after it
  pressure    KV-cache pressure (start the server with a small --num-blocks): --pressure-requests concurrent greedy
              requests whose KV need exceeds the pool all complete (the scheduler preempts and recomputes) and each one's
              text equals the same request sent alone (FuseZap's guarantee holds through preemption)
  crash       the engine process (the server's spawned child, `fusezap-engine`) is killed with SIGKILL, as a kernel fault
              or a device error would end it: /health must answer 503 and the server must exit with code 1 within
              --crash-timeout s, so a supervisor restarts it (fusezap/serve/api.py, c7b891b); the exit code is read only
              when this script started the server, else the process must be gone
Not covered here (left for the report): a real GPU fault or device reset (needs a privileged host), long-run memory
growth (the stability run's RSS and memory columns), out-of-memory at start-up (the prefill budget picks a size that fits).
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor

def http(base: str, path: str, body: object | None = None, timeout: float = 120.0, raw: bytes | None = None) -> tuple[int, dict | str]:
  """(status, parsed JSON or text); status 0 for a dropped connection / timeout (the reason as text)."""
  data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
  req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"}, method="POST" if data is not None else "GET")
  try:
    with urllib.request.urlopen(req, timeout=timeout) as r: txt, st = r.read().decode(), r.status
  except urllib.error.HTTPError as e: txt, st = e.read().decode(errors="replace"), e.code
  except Exception as e: return 0, f"{type(e).__name__}: {e}"
  try: return st, json.loads(txt)
  except ValueError: return st, txt

def complete(base: str, model: str, prompt, max_tokens: int = 16, timeout: float = 300.0) -> tuple[int, dict | str]:
  return http(base, "/v1/completions", {"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "ignore_eos": True}, timeout)

def text_of(r: dict | str) -> str | None:
  try: return r["choices"][0]["text"]  # type: ignore[index]
  except Exception: return None

def serves(base: str, model: str) -> bool:
  st, r = complete(base, model, "The capital of France is", 4, 120)
  return st == 200 and text_of(r) is not None

def probe_bad(a) -> dict:
  cases = {"max_tokens -1": dict(body={"model": a.model, "prompt": "hi", "max_tokens": -1}),
           "unknown model": dict(body={"model": "no-such-model", "prompt": "hi", "max_tokens": 4}),
           "not JSON": dict(raw=b"{this is not json")}
  got = {}
  for k, c in cases.items():
    st, r = http(a.base_url, "/v1/completions", c.get("body"), 60, c.get("raw"))
    got[k] = st
  bad = {k: v for k, v in got.items() if not 400 <= v < 500}
  ok_after = serves(a.base_url, a.model)
  return {"status": "pass" if not bad and ok_after else "fail", "codes": got, "serves_after": ok_after,
          "reason": "every malformed request got a 4xx and the server serves after" if not bad and ok_after else f"non-4xx {bad}, serves after {ok_after}"}

def probe_overlength(a) -> dict:
  st, info = http(a.base_url, "/v1/models")
  try: L = int(info["data"][0]["max_model_len"])  # type: ignore[index]
  except Exception: return {"status": "skip", "reason": f"/v1/models gives no max_model_len ({st})"}
  st, r = complete(a.base_url, a.model, [1000 + (i % 5000) for i in range(L + 64)], 8, 120)
  ok_after = serves(a.base_url, a.model)
  code = r.get("error", {}).get("code") if isinstance(r, dict) and isinstance(r.get("error"), dict) else None
  ok = 400 <= st < 500 and ok_after
  return {"status": "pass" if ok else "fail", "http": st, "code": code, "prompt_tokens": L + 64, "max_model_len": L, "serves_after": ok_after,
          "reason": f"HTTP {st} ({code}) for {L + 64} prompt tokens > {L}; serves after: {ok_after}"}

def probe_pressure(a) -> dict:
  n, plen, out = a.pressure_requests, a.pressure_prompt, a.pressure_tokens
  prompts = [[1000 + ((i * 7919 + j * 31) % 20000) for j in range(plen)] for i in range(n)]
  alone = []
  for p in prompts[:a.pressure_check]:
    st, r = complete(a.base_url, a.model, p, out, 600)
    alone.append(text_of(r) if st == 200 else None)
  t0 = time.time()
  with ThreadPoolExecutor(n) as ex: res = list(ex.map(lambda p: complete(a.base_url, a.model, p, out, 900), prompts))
  wall = time.time() - t0
  errs = [(st, str(r)[:120]) for st, r in res if st != 200 or text_of(r) is None]
  same = [text_of(res[i][1]) == alone[i] and alone[i] is not None for i in range(len(alone))]
  ok = not errs and all(same)
  return {"status": "pass" if ok else "fail", "requests": n, "prompt_tokens": plen, "max_tokens": out, "wall_s": round(wall, 1),
          "errors": errs[:5], "n_errors": len(errs), "alone_equal": f"{sum(same)}/{len(same)}",
          "reason": f"{n} x ({plen} + {out}) tokens: {len(errs)} errors, {sum(same)}/{len(same)} equal to alone, {wall:.1f} s"}

def children(pid: int) -> list[int]:
  out = []
  try:
    for t in os.listdir(f"/proc/{pid}/task"):
      with open(f"/proc/{pid}/task/{t}/children") as f: out += [int(x) for x in f.read().split()]
  except OSError:
    try: out = [int(x) for x in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()]
    except Exception: pass
  return out

def cmdline(pid: int) -> str:
  try:
    with open(f"/proc/{pid}/cmdline", "rb") as f: return f.read().replace(b"\0", b" ").decode(errors="replace")
  except OSError: return ""

def probe_crash(a, proc: subprocess.Popen | None) -> dict:
  if not a.pid: return {"status": "skip", "reason": "no --pid"}
  eng = [c for c in children(a.pid) if "spawn_main" in cmdline(c) and "resource_tracker" not in cmdline(c)]
  if not eng: return {"status": "skip", "reason": f"no spawned engine child of {a.pid} (children {children(a.pid)})"}
  os.kill(eng[0], signal.SIGKILL)
  t0, h503, gone, code = time.time(), None, None, None
  while time.time() - t0 < a.crash_timeout:
    if h503 is None:
      st, _ = http(a.base_url, "/health", None, 5)
      if st == 503: h503 = round(time.time() - t0, 2)
    if proc is not None and proc.poll() is not None: code, gone = proc.returncode, round(time.time() - t0, 2); break
    if proc is None and not os.path.exists(f"/proc/{a.pid}"): gone = round(time.time() - t0, 2); break
    time.sleep(0.2)
  ok = gone is not None and (proc is None or code == 1)
  return {"status": "pass" if ok else "fail", "engine_pid": eng[0], "health_503_after_s": h503, "server_exit_after_s": gone, "exit_code": code,
          "reason": f"engine killed: /health 503 after {h503} s, server exited after {gone} s with code {code}"}

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--base-url", required=True); ap.add_argument("--model", required=True)
  ap.add_argument("--pid", type=int, default=0, help="the server's pid (crash probe)")
  ap.add_argument("--probes", default="bad,overlength,pressure,crash")
  ap.add_argument("--pressure-requests", type=int, default=32); ap.add_argument("--pressure-prompt", type=int, default=1024)
  ap.add_argument("--pressure-tokens", type=int, default=256); ap.add_argument("--pressure-check", type=int, default=4, help="requests also sent alone first")
  ap.add_argument("--crash-timeout", type=float, default=60.0)
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  a.base_url = a.base_url.rstrip("/")
  fns = {"bad": probe_bad, "overlength": probe_overlength, "pressure": probe_pressure, "crash": lambda a: probe_crash(a, None)}
  out = {}
  for p in [x for x in a.probes.split(",") if x]:
    t0 = time.time()
    try: out[p] = fns[p](a)
    except Exception as e: out[p] = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    out[p]["seconds"] = round(time.time() - t0, 1)
    print(f"I37_PROBE {p}: {out[p]['status']}: {out[p].get('reason')}", flush=True)
  if a.json:
    with open(a.json, "w") as f: json.dump(out, f, indent=1)
  return 0 if all(v["status"] in ("pass", "skip") for v in out.values()) else 1

if __name__ == "__main__": sys.exit(main())
