"""Compile rendered FuseZap kernels to NEFFs with no Neuron device (NKI-N2; docs/NKI_BACKEND.md §3, §8).

Needs a Linux x86_64 host with `scripts/nki_env.sh --cc` (nki 0.6.0 + neuronx-cc 2.27 in .venv-nki):
  FZ_DEVICE=NKISIM uv run python scripts/nki_neff.py [--target inf2] [--out results/neff]
Renders a few representative kernels, compiles each with neuronx-cc for the target and prints a table of compile time
and NEFF size. Before compiling, each kernel is also run in the simulator and checked against numpy.
"""
from __future__ import annotations
import argparse, os, time
import numpy as np
from fusezap import Tensor, dtypes
from fusezap.codegen.nki import NKIRenderer
from fusezap.device.neuron import NKISimError, NKISimProgram, NKIWorker, compile_neff
from fusezap.dtype import to_np
from fusezap.schedule import create_schedule

def cases() -> dict:
  r = np.random.default_rng(0)
  a, b = r.standard_normal((300, 1000)).astype(np.float32), r.standard_normal((1000,)).astype(np.float32)
  s = r.standard_normal((200, 1300)).astype(np.float32)
  i, j = r.integers(-2**30, 2**30, (128, 512), dtype=np.int32), r.integers(1, 1000, (128, 512), dtype=np.int32)
  x = r.uniform(-50, 50, (4096,)).astype(np.float32)
  e = np.exp(s - s.max(1, keepdims=True))
  return {
    "add_bcast_300x1000": (lambda: Tensor(a) + Tensor(b) * 2, a + b * 2),
    "row_max_300x1000": (lambda: Tensor(a).max(axis=1), a.max(1)),
    "softmax_200x1300": (lambda: Tensor(s).softmax(-1), e / e.sum(1, keepdims=True)),
    "int32_max_128x512": (lambda: Tensor(i).maximum(Tensor(j * 2**20)), np.maximum(i, j * 2**20)),   # exact hi/lo form
    "int32_rowmax_128x512": (lambda: Tensor(i).max(axis=1), i.max(1)),                               # int32 max tree
    "trunc_cast_300x1000": (lambda: (Tensor(a) * 1000).cast(dtypes.int32), np.trunc(a * np.float32(1000)).astype(np.int32)),
    "sin_4096": (lambda: Tensor(x).sin(), np.sin(x)),
  }

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--target", default="inf2")
  ap.add_argument("--out", default="results/neff")
  args = ap.parse_args()
  ver = NKIWorker.get().call({"op": "version"})
  print(f"nki {ver['nki']}, neuronx-cc {ver['neuronxcc']}, target {args.target}")
  rows = []
  for name, (mk, ref) in cases().items():
    for k, si in enumerate([si for si in create_schedule([mk().lazydata]) if si.kind == "kernel"]):
      prog = NKIRenderer().render(si.ast, si.var_vals)
      arrays = [b.ensure_allocated().handle[:b.nbytes].view(to_np(b.dtype)) if b.size else np.zeros(1, to_np(b.dtype)) for b in si.bufs]
      arrays = (arrays + [np.zeros(1, np.uint8)] * prog.n_bufs)[:prog.n_bufs]
      st = time.perf_counter(); NKISimProgram(prog, prog.src.encode())(*arrays); sim_s = time.perf_counter() - st
      d = os.path.abspath(os.path.join(args.out, args.target, f"{name}_{k}"))
      os.makedirs(d, exist_ok=True)
      try:
        neff, cc_s = compile_neff(prog, [a.copy() for a in arrays], args.target, d)
        rows.append((name, prog.name, len(prog.src.splitlines()), f"{sim_s:.2f}", f"{cc_s:.2f}", len(neff)))
      except NKISimError as e:   # reported, never hidden: the simulator accepts some ops neuronx-cc rejects
        msg = str(e).split("diagnostics:")[-1].strip().splitlines()[0][:120]
        rows.append((name, prog.name, len(prog.src.splitlines()), f"{sim_s:.2f}", "FAILED", msg))
    out = mk().numpy()
    np.testing.assert_allclose(out.astype(np.float64), ref.astype(np.float64), rtol=2e-5, atol=2e-5)
  print(f"| kernel | NKI name | source lines | simulate s | compile s | NEFF bytes |\n|---|---|---|---|---|---|")
  for r in rows: print("| " + " | ".join(str(x) for x in r) + " |")

if __name__ == "__main__": main()
