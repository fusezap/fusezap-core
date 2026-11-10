"""I69: how much device memory (HBM) one NeuronCore's runtime worker can allocate, measured, before any 7B load.

The question: Qwen2.5-7B-Instruct in bf16 on FuseZap NEURON needs ~14.1 GB of device weights (the linear layers 13.05 GB +
lm_head 1.09 GB; the embedding table stays on the host) + the KV pool (float32, 115 KB per slot over 28 layers) + scratch.
The Inferentia2 chip has 32 GiB of HBM and two NeuronCore-v2 (neuron-ls: "32 GB" per device); AWS's SageMaker / LMI guide
says "Each NeuronCore has a dedicated high bandwidth memory (HBM) of 16 GB" (docs/NKI_BACKEND.md, I69). If one core's
allocations stop near 16 GiB, 7B fits one core with ~2 GiB left; if the runtime lets one core address the whole device,
the margin is larger. This allocates zeroed device tensors of --chunk-mib each in the .venv-nrt runtime worker (the same
NrtpyTensor.from_numpy the NEURON allocator uses) until one fails or --max-gib is reached, then frees them.

  NEURON_RT_VISIBLE_CORES=0 .venv/bin/python scripts/nki_hbm_probe.py --json results/x/hbm_core0.json
Prints "hbm_probe: <cores> allocated <G> GiB in <n> chunks of <c> MiB; first failure: <error>".
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--chunk-mib", type=int, default=512)
  ap.add_argument("--max-gib", type=float, default=34.0)
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  from fusezap.device.neuron import NrtWorker
  w = NrtWorker.get()
  rep = w.call({"op": "version"})
  if not rep.get("ok"): raise SystemExit(f"hbm_probe: runtime worker failed: {rep}")
  chunk, ids, err, t0 = a.chunk_mib << 20, [], None, time.perf_counter()
  while (len(ids) + 1) * chunk <= a.max_gib * (1 << 30):
    i = 10 ** 8 + len(ids)
    r = w.call({"op": "stats", "new": {i: chunk}})
    if not r.get("ok"):
      err = r.get("err")
      break
    ids.append(i)
  got = len(ids) * chunk / (1 << 30)
  cores = os.environ.get("NEURON_RT_VISIBLE_CORES", "(all visible)")
  out = {"visible_cores": cores, "chunk_mib": a.chunk_mib, "chunks": len(ids), "allocated_gib": round(got, 2), "first_failure": err,
         "seconds": round(time.perf_counter() - t0, 1), "fits_7b_bf16_1core": got >= 15.0}
  print(f"hbm_probe: cores {cores} allocated {got:.2f} GiB in {len(ids)} chunks of {a.chunk_mib} MiB "
        f"({out['seconds']} s); first failure: {(err or 'none (cap reached)')[:300]}", flush=True)
  w.call({"op": "stats", "free": ids})
  if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w") as f: json.dump(out, f, indent=1)

if __name__ == "__main__": main()
