"""NKI-N4: the raw Neuron runtime (nrtpy) on the NEFFs scripts/nki_n4_probe.py compiled (docs/NKI_BACKEND.md §4 exception 4 and
risk 3). Runs in its own venv (.venv-nrt: numpy + nrtpy; the nrtpy and nki wheels conflict), so it imports nothing from fusezap.

  .venv-nrt/bin/python scripts/nki_n4_nrtpy.py --neff-dir results/nki_n4/neff --out results/nki_n4/nrtpy.json

API used (awsdocs-neuron .../neuron-runtime/nrtpy/tutorial-validate-benchmark.html, checked 2026-10-26):
  NrtpyModel.load_from_neff(path); model.input_tensors_info / output_tensors_info; NrtpyTensor.from_numpy(a, name=...);
  outputs = model(inputs={name: tensor}); outputs[name].numpy() (auto-allocated outputs may come back as a void dtype, so
  they are viewed as the expected dtype); model.benchmark(inputs=..., warmup_iter=, benchmark_iter=, mode="device"|"host").
Everything else (pre-allocated outputs, slicing/aliasing views) is probed by introspection and recorded, never assumed.

Per NEFF case (kernel.neff + io.npz with in<i> = argument i and out<i> = the simulator's result for output argument i):
  * the tensor names and shapes/dtypes the NEFF declares (how our data0..n arguments map to them);
  * load time, from_numpy (host->device copy) and .numpy() (device->host) cost by size, model() call cost warm (host loop)
    and the runtime's own benchmark (device and host mode): the per-NEFF launch overhead;
  * correctness against the simulator result;
  * exception 4: an input passed as an untyped uint8 byte view (does the runtime care about dtypes?), a pre-allocated output
    tensor (`outputs=`), and any slice/view API on NrtpyTensor.
"""
from __future__ import annotations
import argparse, glob, json, os, statistics, sys, time, traceback
import numpy as np

RES: dict = {"cases": {}}

def note(case: str, key: str, val) -> None:
  RES["cases"].setdefault(case, {})[key] = val
  print(json.dumps({"case": case, key: val}, default=str), flush=True)

def attempt(case: str, key: str, fn):
  try:
    v = fn(); note(case, key, v); return v
  except Exception as e:
    note(case, key, {"error": f"{type(e).__name__}: {str(e)[-500:]}", "tb": traceback.format_exc()[-1200:]}); return None

def info_dict(info) -> dict:
  out = {}
  try: items = info.items()
  except AttributeError: return {"repr": repr(info)[:2000]}
  for k, v in items:
    d = {a: getattr(v, a) for a in ("shape", "dtype", "size", "nbytes", "name") if hasattr(v, a)}
    out[k] = {a: (list(x) if isinstance(x, tuple) else str(x)) for a, x in d.items()} or repr(v)[:300]
  return out

def timed(fn, n: int) -> dict:
  ts = []
  for _ in range(n):
    st = time.perf_counter(); fn(); ts.append(time.perf_counter() - st)
  return {"n": n, "median_us": statistics.median(ts) * 1e6, "p10_us": float(np.percentile(ts, 10)) * 1e6,
          "p90_us": float(np.percentile(ts, 90)) * 1e6, "min_us": min(ts) * 1e6}

def as_dict(stats) -> dict:
  if isinstance(stats, dict): return stats
  return {a: getattr(stats, a) for a in ("mean_ms", "min_ms", "max_ms", "std_dev_ms", "iterations") if hasattr(stats, a)} or {"repr": repr(stats)}

def run_case(d: str, iters: int) -> None:
  from nrtpy import NrtpyModel, NrtpyTensor
  case = os.path.basename(d)
  io = np.load(os.path.join(d, "io.npz"))
  ins = [io[f"in{i}"] for i in range(sum(k.startswith("in") for k in io.files))]
  outs = {int(k[3:]): io[k] for k in io.files if k.startswith("out")}
  st = time.perf_counter(); model = NrtpyModel.load_from_neff(os.path.join(d, "kernel.neff")); note(case, "load_s", time.perf_counter() - st)
  iinfo, oinfo = info_dict(model.input_tensors_info), info_dict(model.output_tensors_info)
  note(case, "inputs_info", iinfo); note(case, "outputs_info", oinfo)
  in_names, out_names = list(iinfo), list(oinfo)
  # our kernels take every buffer as an argument (data0..n, outputs included) and return the outputs
  def name_of(i: int) -> str | None:
    # run 1 fix: argument i is declared as data<i>, or as data<i>.must_alias_input when the kernel also returns it (output
    # data<i>). Run 1 matched neither for the output argument, left the must-alias input out and got NRT_INVALID(2).
    # Every declared input must be passed; nrtpy then makes each must-alias output a slice of its input (alias_info).
    for n in in_names:
      if n in (f"data{i}", f"data{i}.must_alias_input"): return n
    return None
  mapping = {name_of(i): i for i in range(len(ins)) if name_of(i) is not None}
  note(case, "arg_mapping", {k: f"data{v}" for k, v in mapping.items()})
  def tensors(view_u8: bool = False) -> dict:
    return {n: NrtpyTensor.from_numpy(np.ascontiguousarray(ins[i]).view(np.uint8) if view_u8 else np.ascontiguousarray(ins[i]), name=n)
            for n, i in mapping.items()}
  inputs = attempt(case, "from_numpy_ok", lambda: bool(tensors())) and tensors()
  if not inputs: return
  def check(res) -> dict:
    got = {}
    for oi, (n, t) in enumerate(res.items()):
      arr = np.asarray(t.numpy())
      idx = int(n[4:]) if n.startswith("data") and n[4:].isdigit() else mapping.get(n, sorted(outs)[oi] if oi < len(outs) else None)
      if idx is None or idx not in outs: got[n] = {"dtype": str(arr.dtype), "nbytes": int(arr.nbytes), "matched": None}; continue
      exp = outs[idx]
      v = arr.reshape(-1).view(np.uint8)[:exp.nbytes].view(exp.dtype) if arr.nbytes >= exp.nbytes else None
      got[n] = {"dtype": str(arr.dtype), "nbytes": int(arr.nbytes), "as": f"data{idx}",
                "bitwise_equal_to_sim": None if v is None else float(np.mean(v == exp.reshape(-1))),
                "max_abs_vs_sim": None if v is None or exp.dtype.kind != "f" else float(np.max(np.abs(v.astype(np.float64) - exp.reshape(-1))))}
    return got
  res = attempt(case, "first_run", lambda: check(model(inputs=inputs)))
  attempt(case, "call_warm_host_loop", lambda: timed(lambda: model(inputs=inputs), iters))
  attempt(case, "call_and_readback_warm", lambda: timed(lambda: [t.numpy() for t in model(inputs=inputs).values()], max(10, iters // 4)))
  for mode in ("device", "host"):
    attempt(case, f"benchmark_{mode}", lambda mode=mode: as_dict(model.benchmark(inputs=inputs, warmup_iter=10, benchmark_iter=iters, mode=mode)))
  # exception 4: untyped byte buffers, pre-allocated outputs, views
  attempt(case, "untyped_uint8_inputs", lambda: check(model(inputs=tensors(view_u8=True))))
  def prealloc():
    o = {}
    for n in out_names:
      k = int(n[4:]) if n.startswith("data") and n[4:].isdigit() else None
      src = ins[k] if k is not None and k < len(ins) else np.zeros(1, np.float32)
      o[n] = NrtpyTensor.from_numpy(np.full_like(src, 7), name=n)
    model(inputs=inputs, outputs=o)
    return check(o)
  attempt(case, "preallocated_outputs", prealloc)
  attempt(case, "tensor_api", lambda: sorted(a for a in dir(NrtpyTensor) if not a.startswith("_")))
  attempt(case, "model_api", lambda: sorted(a for a in dir(model) if not a.startswith("_")))
  def view_probe():
    """A view of bytes [2048, 4096) of a 1024-float tensor: its first value must be 512.0 if views alias by byte offset."""
    t = NrtpyTensor.from_numpy(np.arange(1024, dtype=np.float32), name="v")
    errs = []
    for nm in ("slice", "slice_from_tensor", "view", "narrow"):
      if not hasattr(t, nm) and not hasattr(NrtpyTensor, nm): continue
      for call in ((lambda nm=nm: getattr(t, nm)(2048, 2048)), (lambda nm=nm: getattr(NrtpyTensor, nm)(t, 2048, 2048))):
        try:
          v = call()
          first = np.asarray(v.numpy()).reshape(-1).view(np.uint8)[:4].view(np.float32).tolist() if hasattr(v, "numpy") else None
          return {"api": nm, "repr": repr(v)[:200], "first_value": first, "aliases_by_offset": first == [512.0]}
        except Exception as e: errs.append(f"{nm}: {type(e).__name__}: {str(e)[:200]}")
    return {"api": None, "errors": errs}
  attempt(case, "view_api", view_probe)

def copy_costs(iters: int) -> None:
  from nrtpy import NrtpyTensor
  for nb in (4 << 10, 64 << 10, 1 << 20, 16 << 20, 64 << 20):
    a = np.ones(nb // 4, np.float32)
    t = NrtpyTensor.from_numpy(a, name="c")
    n = max(5, iters // (1 + nb // (1 << 20)))
    attempt("copies", f"from_numpy_{nb}B", lambda a=a, n=n: timed(lambda: NrtpyTensor.from_numpy(a, name="c"), n))
    attempt("copies", f"numpy_readback_{nb}B", lambda t=t, n=n: timed(lambda: t.numpy(), n))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--neff-dir", default="results/nki_n4/neff")
  ap.add_argument("--out", default="results/nki_n4/nrtpy.json")
  ap.add_argument("--iters", type=int, default=1000)
  a = ap.parse_args()
  try:
    import nrtpy
    from importlib.metadata import version
    try: RES["nrtpy"] = version("nrtpy")
    except Exception: RES["nrtpy"] = getattr(nrtpy, "__version__", "?")
    print(json.dumps({"nrtpy": RES["nrtpy"], "python": sys.version.split()[0]}), flush=True)
    RES["nrtpy_api"] = sorted(x for x in dir(nrtpy) if not x.startswith("_"))
  except Exception as e:
    RES["error"] = f"nrtpy import: {type(e).__name__}: {e}"; print(RES["error"]); json.dump(RES, open(a.out, "w"), indent=1); sys.exit(1)
  for d in sorted(glob.glob(os.path.join(a.neff_dir, "*"))):
    if not os.path.exists(os.path.join(d, "kernel.neff")): continue
    try: run_case(d, a.iters)
    except Exception as e: note(os.path.basename(d), "error", f"{type(e).__name__}: {str(e)[-500:]}\n{traceback.format_exc()[-1500:]}")
    json.dump(RES, open(a.out, "w"), indent=1, default=str)
  try: copy_costs(a.iters)
  except Exception as e: note("copies", "error", f"{type(e).__name__}: {e}")
  json.dump(RES, open(a.out, "w"), indent=1, default=str)
  print("\n==== nrtpy summary ====")
  for c, r in RES["cases"].items():
    for k in ("load_s", "first_run", "call_warm_host_loop", "benchmark_device", "benchmark_host", "untyped_uint8_inputs", "preallocated_outputs"):
      if k in r: print(f"{c:<22} {k:<22} {json.dumps(r[k], default=str)[:220]}")
    if c == "copies":
      for k, v in r.items(): print(f"{c:<22} {k:<22} {json.dumps(v, default=str)[:160]}")

if __name__ == "__main__": main()
