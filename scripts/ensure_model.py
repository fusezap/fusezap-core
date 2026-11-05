#!/usr/bin/env python3
"""Make sure the exact model a Validate reference was built from is on disk, and print its local path.

  python scripts/ensure_model.py --ref docs/validate/examples/refs/qwen2.5-0.5b-instruct-f32.json
  python scripts/ensure_model.py --model Qwen/Qwen2.5-0.5B-Instruct --revision <commit>

"Right model" means: the same repo id AND the same commit (revision) as the reference, not whatever copy of that repo
happens to be cached (a cached `main` can be an older or newer commit). The check:
  1. look in the HF cache for that repo at that exact commit, without network;
  2. verify it: the snapshot folder is that commit, config.json is there and is the expected architecture, every weight
     file the index lists is present, and each safetensors header parses (catches truncated downloads);
  3. if it is missing or fails verification, download that commit and verify again; fail loudly if it still does not.
Prints the snapshot path on the last line of stdout; serve both engines from that path with --served-model-name <repo>
so both read the identical files."""
import argparse, glob, json, os, struct, sys

PATTERNS = ["*.json", "*.safetensors", "*.model", "*.txt", "*.tiktoken", "*.jinja"]

def log(msg: str) -> None: print(f"ensure_model: {msg}", file=sys.stderr, flush=True)

def weight_files(path: str) -> list[str]:
  idx = os.path.join(path, "model.safetensors.index.json")
  if os.path.exists(idx):
    with open(idx) as f: return sorted(set(json.load(f)["weight_map"].values()))
  return ["model.safetensors"] if os.path.exists(os.path.join(path, "model.safetensors")) else []

def header_ok(fn: str) -> bool:
  """A safetensors file starts with an 8-byte header length and a JSON header; the data must reach the last tensor's end."""
  try:
    size = os.path.getsize(fn)
    with open(fn, "rb") as f:
      n = struct.unpack("<Q", f.read(8))[0]
      if n <= 0 or 8 + n > size: return False
      hdr = json.loads(f.read(n))
    end = max((v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"), default=0)
    return 8 + n + end == size
  except Exception: return False

def verify(path: str, revision: str, arch: str | None) -> list[str]:
  problems = []
  if os.path.basename(os.path.normpath(path)) != revision: problems.append(f"snapshot is {os.path.basename(path)}, want {revision}")
  cfg = os.path.join(path, "config.json")
  if not os.path.exists(cfg): problems.append("config.json missing")
  elif arch:
    with open(cfg) as f: archs = json.load(f).get("architectures") or []
    if arch not in archs: problems.append(f"architectures {archs}, want {arch}")
  ws = weight_files(path)
  if not ws: problems.append("no safetensors weights")
  for w in ws:
    fn = os.path.join(path, w)
    if not os.path.exists(fn): problems.append(f"{w} missing")
    elif not header_ok(fn): problems.append(f"{w} truncated or corrupt")
  if not os.path.exists(os.path.join(path, "tokenizer.json")) and not glob.glob(os.path.join(path, "tokenizer*")):
    problems.append("tokenizer files missing")
  return problems

def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--ref", help="a Validate reference JSON (supplies model and revision)")
  ap.add_argument("--model"); ap.add_argument("--revision")
  ap.add_argument("--arch", default=None, help="expected config.json architecture, e.g. Qwen2ForCausalLM")
  a = ap.parse_args()
  model, revision = a.model, a.revision
  if a.ref:
    with open(a.ref) as f: ref = json.load(f)
    model, revision = model or ref.get("model"), revision or ref.get("revision")
  if not model or not revision or len(revision) < 40:
    log(f"need a repo id and a full 40-character commit revision (got model={model!r}, revision={revision!r})"); return 2
  from huggingface_hub import snapshot_download
  try:
    path = snapshot_download(model, revision=revision, local_files_only=True, allow_patterns=PATTERNS)
    problems = verify(path, revision, a.arch)
    if not problems: log(f"using cached {model}@{revision[:12]} (verified)"); print(path); return 0
    wrong = [p for p in problems if p.startswith("architectures")]
    if wrong: log(f"{model}@{revision[:12]} is not the expected model: {wrong[0]}; not downloading"); return 1
    log(f"cached copy of {model}@{revision[:12]} failed verification: {'; '.join(problems)}; downloading it again")
    force = True
  except Exception as e:
    log(f"{model}@{revision[:12]} not in the cache ({type(e).__name__}); downloading"); force = False
  path = snapshot_download(model, revision=revision, allow_patterns=PATTERNS, force_download=force)
  problems = verify(path, revision, a.arch)
  if problems: log(f"downloaded {model}@{revision[:12]} still fails verification: {'; '.join(problems)}"); return 1
  log(f"downloaded {model}@{revision[:12]} (verified)"); print(path); return 0

if __name__ == "__main__": sys.exit(main())
