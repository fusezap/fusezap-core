"""I69: the --i69 job's choice, table and bar (scripts/nki_n8_prefill_inf2.sh --i69): the first-token gap, main's prefill
kernels vs the bitwise I69 forms (FZ_NEURON_PFK_ATTN_HB / _SILU / _EXPF, FZ_NEURON_EMBED_BF16).

  python3 scripts/nki_n8_i69_summary.py <out dir> [--json <file>]     the summary (SUMMARY.txt)
  python3 scripts/nki_n8_i69_summary.py <out dir> --kern-env           the kernel flags pfk_bench found bitwise and faster (env words)
  python3 scripts/nki_n8_i69_summary.py <out dir> --choose             the fastest bitwise in-process variant's env words (stdout)

Reads
  <out>/i69_variants.tsv                       name <TAB> env words, per in-process variant (main first)
  <out>/serve_i69_<name>.json                  in process (scripts/nki_n6_serve.py --tp 2): prefill_bench (each prefill forward
                                               at each pool prefix: wall to the token, the first logits row's hash), ttft_by_len
                                               (128-token prompts: c1 / c4 burst TTFT and the first tokens), c1 / c4 decode
  <out>/pfk_bench/pfk_bench.json               one NeuronCore, a 3-layer model of the shard (scripts/nki_pfk_bench.py): per config
                                               the layer NEFF's device ms and its logits bits vs the members (v1)
  <out>/fz_tp2_bf16_7b_i69_<lab>_{perf,acc}/   served (Invarix: short 128 / 128 at c1 and c4; accuracy auto), lab chosen / main
A variant is bitwise when every prefill forward main ran has the same logits hash in it and its 128-token c1 / c4 first
tokens and its greedy 24 x 64 tokens are main's. The choice: the bitwise variant with the lowest 128-row forward at the 128-slot prefix (Invarix short c1's
forward: lowest-first blocks), if it beats main there; else main. The bar (printed): chosen served short c1 TTFT p50 below
main's served p50, TPOT p50 within main + 1 ms, no errors, Invarix PROMOTE; AWS NxDI TP=2 c1 TTFT p50 50.6 ms for reference.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NXDI_C1_TTFT = 50.6        # AWS NxD Inference TP=2 offline, I73 run (same instance type, same workload)
I78_MAIN_C1 = (55.1, 56.3)  # I78 served short c1 TTFT p50 / p99, main's exact prefill (docs/status/nki-n8-pfk2-inf2)
FLAGS = ("hb", "silu", "xf")

def _load(p: str) -> dict:
  try:
    with open(p) as f: return json.load(f)
  except (OSError, ValueError): return {}

def _f(x) -> float | None:
  try: return float(x)
  except (TypeError, ValueError): return None

def variants(out: str) -> list[tuple[str, str]]:
  try:
    with open(os.path.join(out, "i69_variants.tsv")) as f: return [tuple((ln.rstrip("\n").split("\t") + [""])[:2]) for ln in f if ln.strip()]
  except OSError: return []

def wall(run: dict, key: str = "128@128") -> float | None:
  pb = run.get("prefill_bench") or {}
  return _f((pb.get(key) or {}).get("wall_ms_min")) if isinstance(pb.get(key), dict) else None

def bits(main: dict, run: dict) -> tuple[bool, str]:
  """(bitwise, why not): every prefill forward main ran has the same logits hash in run; the 128-token first tokens equal."""
  pm, pr = main.get("prefill_bench") or {}, run.get("prefill_bench") or {}
  if not pm or "error" in pm: return False, "main has no prefill bench"
  if not pr or "error" in pr: return False, f"no prefill bench ({str(pr.get('error', ''))[:120]})"
  for k, x in pm.items():
    if not isinstance(x, dict): continue
    y = pr.get(k)
    if not isinstance(y, dict): return False, f"forward {k} not run"
    if x.get("logits_sha") != y.get("logits_sha"): return False, f"forward {k}: logits {y.get('logits_sha')} != main's {x.get('logits_sha')}"
  tm, tr = (main.get("ttft_by_len") or {}).get("128") or {}, (run.get("ttft_by_len") or {}).get("128") or {}
  for k in ("c1_ids", "c4_ids"):
    if tm.get(k) and tm.get(k) != tr.get(k): return False, f"128-token {k} {tr.get(k)} != main's {tm.get(k)}"
  gm = {r.get("id"): r.get("ids") for r in ((main.get("greedy") or {}).get("rows") or [])}
  gr = {r.get("id"): r.get("ids") for r in ((run.get("greedy") or {}).get("rows") or [])}
  for k, ids in gm.items():
    if k in gr and gr[k] != ids: return False, f"greedy prompt {k}: tokens differ from main's"
  return True, ""

def choose(out: str) -> tuple[str, str, list[str]]:
  """(name, env words, log lines) of the fastest bitwise variant (main if none beats it)."""
  vs = variants(out)
  if not vs: return "main", "", ["no variants recorded: main"]
  runs = {n: _load(os.path.join(out, f"serve_i69_{n}.json")) for n, _ in vs}
  main_n, main_env = vs[0]
  base = wall(runs[main_n])
  best, log = (main_n, main_env, base), [f"{main_n}: 128@128 {base} ms (the reference)"]
  for n, env in vs[1:]:
    r = runs[n]
    if not r: log.append(f"{n}: not run"); continue
    ok, why = bits(runs[main_n], r)
    w = wall(r)
    log.append(f"{n}: 128@128 {w} ms, bitwise {ok}" + (f" ({why})" if why else ""))
    if ok and w is not None and (best[2] is None or w < best[2]): best = (n, env, w)
  log.append(f"chosen: {best[0]} ({best[1] or 'no env'})")
  return best[0], best[1], log

def kern_env(out: str) -> str:
  """The I69 kernel flags whose single-core layer (v2exact+<flag>:128:128) is bitwise vs v1 and faster than v2exact:128:128,
  as env words (with FZ_NEURON_EMBED_BF16=0: kernels only); '' if none."""
  from fusezap.codegen.nki_prefill import i69_env
  recs = {r.get("config"): r for r in (_load(os.path.join(out, "pfk_bench", "pfk_bench.json")).get("records") or [])}
  base = _f((recs.get("v2exact:128:128") or {}).get("layer_device_ms"))
  good = [f for f in FLAGS if (r := recs.get(f"v2exact+{f}:128:128")) and r.get("logits_bitwise_vs_v1") is True and base
          and (_f(r.get("layer_device_ms")) or 1e9) < base]
  if not good: return ""
  return " ".join(f"{k}={v}" for k, v in i69_env(good).items()) + " FZ_NEURON_EMBED_BF16=0"

def perf(out: str, lab: str) -> dict:
  res: dict = {}
  p = os.path.join(out, f"fz_tp2_bf16_7b_i69_{lab}_perf", "performance.csv")
  if os.path.exists(p):
    with open(p) as f:
      for r in csv.DictReader(f):
        res[(r["workload"], int(r["concurrency"]))] = {"ttft_p50": _f(r.get("ttft_p50_ms")), "ttft_p99": _f(r.get("ttft_p99_ms")),
                                                       "tpot_p50": _f(r.get("tpot_p50_ms")), "tok_s": _f(r.get("output_tok_s")),
                                                       "errors": int(_f(r.get("errors")) or 0)}
  return res

def decision(out: str, lab: str) -> str | None:
  for sub in ("acc", "perf"):
    rep = _load(os.path.join(out, f"fz_tp2_bf16_7b_i69_{lab}_{sub}", "report.json"))
    if rep and sub == "acc": return (rep.get("decision") or {}).get("label") or rep.get("verdict")
  return None

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--json", default="")
  ap.add_argument("--choose", action="store_true")
  ap.add_argument("--kern-env", action="store_true")
  a = ap.parse_args()
  if a.kern_env: print(kern_env(a.out)); return
  if a.choose:
    n, env, log = choose(a.out)
    for ln in log: print(ln, file=sys.stderr)
    print(f"{n}\t{env}")
    return
  S: dict = {}
  print("--- one NeuronCore, 3-layer model of the shard (scripts/nki_pfk_bench.py): the layer NEFF, device ms, bits vs the members")
  recs = (_load(os.path.join(a.out, "pfk_bench", "pfk_bench.json")).get("records") or [])
  base = {r.get("ctx"): _f(r.get("layer_device_ms")) for r in recs if str(r.get("path")) == "v2exact"}
  for r in recs:
    d, b = _f(r.get("layer_device_ms")), base.get(r.get("ctx"))
    gain = f"  {'saves' if b >= d else 'costs'} {28 * abs(b - d):.2f} ms per 28 layers vs v2exact" if d and b and r.get("path") not in ("v1", "v2exact") else ""
    print(f"  {r.get('config', '?'):<24} device {d}  execute {r.get('layer_exec_ms')}  bitwise vs v1 {r.get('logits_bitwise_vs_v1')}{gain}" +
          (f"  ERROR {str(r['error'])[:160]}" if "error" in r else ""))
  S["pfk_bench"] = [{k: r.get(k) for k in ("config", "layer_device_ms", "layer_exec_ms", "logits_bitwise_vs_v1", "error")} for r in recs]
  print(f"  kernel flags bitwise and faster: {kern_env(a.out) or 'none'}")
  print("--- in process, TP=2 (prefill forward wall to the token, min of the replays, ms; 128-token TTFT)")
  vs = variants(a.out)
  runs = {n: _load(os.path.join(a.out, f"serve_i69_{n}.json")) for n, _ in vs}
  S["in_process"] = {}
  for n, env in vs:
    r = runs[n]
    if not r: print(f"  {n:<6} not run ({env})"); continue
    pb = r.get("prefill_bench") or {}
    t = (r.get("ttft_by_len") or {}).get("128") or {}
    ok, why = bits(runs[vs[0][0]], r) if n != vs[0][0] else (True, "the reference")
    fw = "  ".join(f"{k} {x.get('wall_ms_min')}" for k, x in sorted(pb.items(), key=lambda kv: (len(kv[0]), kv[0])) if isinstance(x, dict))
    print(f"  {n:<6} {fw}  | TTFT c1 p50 {t.get('c1_p50')} c4 p99 {t.get('c4_p99')} | TPOT c1 {(r.get('c1') or {}).get('tpot_ms_median')} | bitwise {ok}{' (' + why + ')' if why and n != vs[0][0] else ''}")
    print(f"         env: {env or '(none)'}")
    pp = r.get("prefill_profile") or {}
    if pp and "error" not in pp: print(f"         one prefill step: {pp.get('step_ms')} ms wall, {pp.get('trips')} runtime-worker round trips, {pp.get('launches')} launches")
    S["in_process"][n] = {"env": env, "prefill_bench": {k: x.get("wall_ms_min") for k, x in pb.items() if isinstance(x, dict)},
                          "ttft_128": {k: t.get(k) for k in ("c1_p50", "c1_p99", "c4_p50", "c4_p99")}, "bitwise": ok, "why": why}
  n, env, log = choose(a.out)
  for ln in log: print("  " + ln)
  S["chosen"] = {"name": n, "env": env}
  print("--- served (fusezap serve --tp 2; Invarix short 128 / 128; ms)")
  P = {lab: perf(a.out, lab) for lab in ("chosen", "main")}
  print(f"  {'config':<8} {'conc':>4} {'TTFT p50':>9} {'TTFT p99':>9} {'TPOT p50':>9} {'tok/s':>7} {'errors':>6}")
  for lab, res in P.items():
    for (prof, c), x in sorted(res.items()):
      print(f"  {lab:<8} {c:>4} {x['ttft_p50']!s:>9} {x['ttft_p99']!s:>9} {x['tpot_p50']!s:>9} {x['tok_s']!s:>7} {x['errors']:>6}")
  dec = decision(a.out, "chosen")
  S["served"] = {lab: {f"{p}@c{c}": x for (p, c), x in res.items()} for lab, res in P.items()}
  S["decision"] = dec
  ch, mn = P["chosen"].get(("short", 1)) or {}, P["main"].get(("short", 1)) or {}
  ref = mn.get("ttft_p50") or I78_MAIN_C1[0]
  checks = {"chosen is not main": n != vs[0][0] if vs else False,
            f"short c1 TTFT p50 < main ({ref})": bool(ch.get("ttft_p50") is not None and ch["ttft_p50"] < ref),
            "TPOT p50 <= main + 1 ms": all((x.get("tpot_p50") or 0) <= ((P["main"].get(k) or {}).get("tpot_p50") or 1e9) + 1.0 for k, x in P["chosen"].items()),
            "no request errors": all(x["errors"] == 0 for x in P["chosen"].values()) and bool(P["chosen"]),
            "Invarix PROMOTE": dec == "PROMOTE"}
  S["checks"] = checks
  print("  checks: " + "; ".join(f"{k}: {v}" for k, v in checks.items()))
  if ch.get("ttft_p50") is not None:
    print(f"  chosen short c1 TTFT p50 {ch['ttft_p50']} ms vs main {mn.get('ttft_p50', 'not served')} (I78: {I78_MAIN_C1[0]}), AWS NxDI {NXDI_C1_TTFT} "
          f"(gap {ch['ttft_p50'] - NXDI_C1_TTFT:+.1f} ms)")
  ok = all(checks.values())
  S["recommend"] = ok
  print(f"I69 first token: {'RECOMMEND' if ok else 'DO NOT RECOMMEND'} {n} as the default" + ("" if ok else " (see the checks)"))
  if a.json:
    with open(a.json, "w") as f: json.dump(S, f, indent=1, default=str)

if __name__ == "__main__": main()
