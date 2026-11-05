"""NKI-N6b: the correctness gate in front of every performance measurement of scripts/nki_n6b_*.sh. N6b inf2 run 2 measured
AWS vLLM + NxD Inference at TPOT 11.7 ms while it returned gibberish after the first token (on-device sampling with
max_num_seqs > 1 on Inf2/Trn1, vllm-project/vllm-neuron#57: AwsNeuronTopK returns wrong indices for a reduction size of
151936, Qwen2.5's vocabulary, once batch > 1). No speed is published for a server that fails this gate.

The gate: --n reference prompts (those whose float32 greedy continuation has at least --tokens tokens) continued greedily
over /v1/completions (prompt as token ids, temperature 0, max_tokens --tokens), the text compared with the reference's
greedy ids decoded by the model's tokenizer. Pass = every prompt agrees on its first --min-exact tokens and at least
--need prompts agree on all --tokens (bfloat16 may leave the float32 path at a near tie later on). Sent one at a time and
then all together (the bug corrupted all but one batch row, so a serial-only check could pass).

  python scripts/nki_n6b_gate.py --base-url http://127.0.0.1:18000 --model Qwen/Qwen2.5-0.5B-Instruct \\
      --reference docs/validate/examples/refs/qwen2.5-0.5b-instruct-f32.json --tokenizer <dir with tokenizer.json> --out gate.json
Exit 0 = pass, 1 = output incorrect, 2 = the server could not be asked. Plain urllib + the tokenizers package.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, json, os, sys, urllib.request

def complete(base: str, model: str, ids: list[int], n: int, timeout: float) -> str:
  body = json.dumps({"model": model, "prompt": ids, "max_tokens": n, "temperature": 0.0, "top_p": 1.0}).encode()
  req = urllib.request.Request(base.rstrip("/") + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
  with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)["choices"][0]["text"]

def agree(tok, want_ids: list[int], got: str) -> int:
  """How many leading reference tokens the server's text reproduces (by decoded-prefix comparison)."""
  k = 0
  for i in range(1, len(want_ids) + 1):
    if got.startswith(tok.decode(want_ids[:i], skip_special_tokens=False)): k = i
    else: break
  return k

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--base-url", required=True)
  ap.add_argument("--model", required=True)
  ap.add_argument("--reference", required=True)
  ap.add_argument("--tokenizer", required=True, help="directory with tokenizer.json (the model snapshot)")
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=3)
  ap.add_argument("--tokens", type=int, default=16)
  ap.add_argument("--min-exact", type=int, default=4)
  ap.add_argument("--need", type=int, default=2)
  ap.add_argument("--timeout", type=float, default=300)
  a = ap.parse_args()
  from tokenizers import Tokenizer
  tok = Tokenizer.from_file(os.path.join(a.tokenizer, "tokenizer.json"))
  ref = json.load(open(a.reference))
  ps = [p for p in ref["prompts"] if len(p["greedy_ids"]) >= a.tokens][:a.n]
  rows, ok = [], True
  try:
    serial = [complete(a.base_url, a.model, p["prompt_ids"], a.tokens, a.timeout) for p in ps]
    with cf.ThreadPoolExecutor(len(ps)) as ex:
      together = list(ex.map(lambda p: complete(a.base_url, a.model, p["prompt_ids"], a.tokens, a.timeout), ps))
  except Exception as e:
    print(f"gate: could not query the server: {type(e).__name__}: {e}")
    json.dump({"pass": False, "error": f"{type(e).__name__}: {e}"}, open(a.out, "w"), indent=1)
    return 2
  for mode, texts in (("serial", serial), ("concurrent", together)):
    full = 0
    for p, got in zip(ps, texts):
      want = p["greedy_ids"][:a.tokens]
      k = agree(tok, want, got)
      full += k == len(want)
      rows.append({"mode": mode, "id": p["id"], "agree_tokens": k, "of": len(want), "got": got, "want": tok.decode(want, skip_special_tokens=False)})
      if k < a.min_exact: ok = False
    if full < min(a.need, len(ps)): ok = False
  for r in rows:
    print(f"gate {r['mode']:<10} {r['id']:<24} {r['agree_tokens']:>2}/{r['of']}  got {r['got'][:70]!r}" + ("" if r["agree_tokens"] == r["of"] else f"  want {r['want'][:70]!r}"))
  print(f"gate: {'PASS' if ok else 'FAIL: OUTPUT INCORRECT'} (rule: every prompt >= {a.min_exact} exact tokens, >= {a.need} of {len(ps)} all {a.tokens}, serial and concurrent)")
  json.dump({"pass": ok, "rule": {"tokens": a.tokens, "min_exact": a.min_exact, "need": a.need, "prompts": len(ps)}, "rows": rows}, open(a.out, "w"), indent=1)
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
