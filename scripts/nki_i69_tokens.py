"""I69 follow-up: a served configuration's greedy text for fixed prompts, to compare configurations that must give the same
tokens (the host-path switches and FZ_NEURON_EMBED_DEV change no value; FZ_NEURON_PFK_EXP=act does).

  python scripts/nki_i69_tokens.py --base-url http://127.0.0.1:8000 --model M --reference REF --out tokens_<label>.json
  python scripts/nki_i69_tokens.py --compare tokens_a.json tokens_b.json        # exit 0 when every text is identical

Each of the reference's first --n prompts (token ids) continued greedily for --tokens tokens over /v1/completions, sent
4 at a time (the engine batches them as served traffic would); the texts and their sha256 go to --out. Plain urllib."""
from __future__ import annotations
import argparse, concurrent.futures as cf, hashlib, json, sys, urllib.request

def complete(base: str, model: str, ids: list[int], n: int, timeout: float) -> str:
  body = json.dumps({"model": model, "prompt": ids, "max_tokens": n, "temperature": 0.0, "top_p": 1.0, "ignore_eos": True}).encode()
  req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)["choices"][0]["text"]

def compare(a: str, b: str) -> tuple[bool, str]:
  x, y = (json.load(open(p)) for p in (a, b))
  tx, ty = x.get("texts") or {}, y.get("texts") or {}
  if not tx or set(tx) != set(ty): return False, f"different prompt sets ({len(tx)} vs {len(ty)})"
  diff = [k for k in tx if tx[k] != ty[k]]
  return (not diff), (f"{len(diff)} of {len(tx)} texts differ (first: prompt {diff[0]})" if diff else f"all {len(tx)} texts identical")

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--base-url"); ap.add_argument("--model"); ap.add_argument("--reference"); ap.add_argument("--out")
  ap.add_argument("--n", type=int, default=12)
  ap.add_argument("--tokens", type=int, default=64)
  ap.add_argument("--timeout", type=float, default=300.0)
  ap.add_argument("--compare", nargs=2, default=None)
  a = ap.parse_args()
  if a.compare:
    ok, why = compare(*a.compare)
    print(f"tokens {a.compare[0]} vs {a.compare[1]}: {why}")
    return 0 if ok else 1
  ref = json.load(open(a.reference))
  ps = [(str(s["id"]), list(s["prompt_ids"])) for s in (ref.get("samples") or ref["prompts"])[:a.n]]
  with cf.ThreadPoolExecutor(4) as ex:
    texts = dict(zip([k for k, _ in ps], ex.map(lambda p: complete(a.base_url, a.model, p[1], a.tokens, a.timeout), ps)))
  out = {"model": a.model, "tokens": a.tokens, "texts": texts,
         "sha256": hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()[:16]}
  with open(a.out, "w") as f: json.dump(out, f, indent=1)
  print(f"greedy texts of {len(texts)} prompts x {a.tokens} tokens: sha256 {out['sha256']} -> {a.out}")
  return 0

if __name__ == "__main__": sys.exit(main())
