"""I72: the TP=2 job's per-shard layer choice from its A/B runs (scripts/nki_n8_7b_tp_inf2.sh: serve_<prefix>_halves.json and
serve_<prefix>_members.json, scripts/nki_n6_serve.py --tp 2 --quick).

  python3 scripts/nki_n8_tp_choose.py <out> [prefix]   # stdout: env words; stderr: the table the choice came from

Rule: the halves (LayerSpec's two halves around the all-reduces, FZ_NEURON_TP_LAYER=halves) if their c1 tokens equal the
members' (the same bits, as on NKISHIM) and their c1 TPOT is lower; else the members. No members run: the halves if they
ran; nothing ran: the members.
"""
from __future__ import annotations
import json, os, sys

def load(out: str, name: str) -> dict:
  try:
    with open(os.path.join(out, f"serve_{name}.json")) as f: return json.load(f)
  except (OSError, ValueError): return {}

def main() -> None:
  out, pre = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "tp2_ab")
  h, m = load(out, f"{pre}_halves"), load(out, f"{pre}_members")
  th, tm = (h.get("c1") or {}).get("tpot_ms_median"), (m.get("c1") or {}).get("tpot_ms_median")
  same = (h.get("c1") or {}).get("ids") == (m.get("c1") or {}).get("ids") if h and m else None
  print(f"halves c1 TPOT {th} ms, c4 TPOT {(h.get('c4') or {}).get('tpot_ms_median')} ms; members c1 TPOT {tm} ms, "
        f"c4 TPOT {(m.get('c4') or {}).get('tpot_ms_median')} ms; same c1 tokens: {same}", file=sys.stderr)
  if th is not None and (tm is None or (same and th < tm)):
    print("chosen: halves", file=sys.stderr); print("FZ_NEURON_TP_LAYER=halves")
  else:
    print("chosen: members", file=sys.stderr); print("FZ_NEURON_TP_LAYER=members")

if __name__ == "__main__": main()
