"""NKI-N4 probe on a real inf2/trn1 instance (docs/NKI_BACKEND.md §8 N4, docs/status/NKI-N0-N3.md "What N4 must check first").
Run by scripts/nki_n4_inf2.sh after `scripts/nki_env.sh --cc --nrt`; it uses the NEURON tier (fusezap/device/neuron.py).

  uv run python scripts/nki_n4_probe.py --out results/nki_n4 [--only nrt,neff,numerics] [--iters 200]
  FZ_NEURON_RT=jit uv run python scripts/nki_n4_probe.py --out results/nki_n4 --only launch --iters 5   (its own process:
            the nki.jit path opens the NeuronCore in the .venv-nki worker, which a live runtime worker would hold)

Sections (each one is independent: a failure is recorded in the JSON and the next section runs):
  nrt       N4b, the persistent path (default FZ_NEURON_RT=nrt): a tiny add and a 1 MiB add compiled once to a NEFF, loaded once
            into the .venv-nrt runtime worker and launched on device-resident buffers. Per size: the first launch (compile +
            load + execute), warm launches from fusezap (one `exec` request each: what a schedule item costs), the runtime's
            execute alone (timed in the worker), nrtpy's device-trace and host benchmarks, a dependent chain of 200 launches
            ping-ponging two device buffers (checked: a + 200), and host<->device copy costs (4 KiB, 1 MiB, 16 MiB) through
            the worker, against the bare runtime-worker round trip.
  launch    the nki.jit fallback (FZ_NEURON_RT=jit): the first call (compile + load + run) and warm calls of a tiny kernel and a
            1 MiB one, against the bare worker round trip (so the pipe cost can be subtracted). Run 1: ~1.31 s per call.
  neff      compiles the tiny add kernel and a 128x1024x512 fz_matmul to NEFFs (kernel.neff + io.npz with the inputs in argument
            order and the expected outputs) for scripts/nki_n4_nrtpy.py, which times the raw runtime (nrtpy) without nki.jit.
  numerics  the [unverified on hardware] numerics: exp/exp2/log2/sin/sqrt/reciprocal/tanh (N5: exact VectorE exp2/log2/sqrt by
            default; FZ_NKI_FAST_EXP=1 renders the ScalarE instructions, whose log2 the run-2 inputs broke), long fp32 sums (accumulation
            order), TensorE fp32 matmul and flash attention, int32 GpSimd add wraparound, int32 max/compare above 2**24, a
            partition-axis max with NaN/Inf (nc_transpose), bitcast. Each case runs on NEURON and on NKISHIM (bit-identical to
            the nki 0.6 simulator on the op suite, NKI-N2) and is compared with float64 numpy: bitwise-equal fraction against
            the simulator, max abs/rel error against float64 and the max ULP distance.
Writes <out>/probe.json and prints one JSON line per measurement plus a summary.
"""
from __future__ import annotations
import argparse, json, math, os, statistics, sys, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusezap import Tensor, dtypes
from fusezap.codegen.nki import NKIRenderer
from fusezap.device import Device, neuron
from fusezap.dtype import to_np
from fusezap.schedule import create_schedule

OUT: dict = {"sections": {}}
HW = os.getenv("FZ_N4_HW", "NEURON")   # NKISHIM: a dry run of this script on the Mac (no device; numbers meaningless)
R = np.random.default_rng(4)

def emit(sec: str, row: dict) -> None:
  OUT["sections"].setdefault(sec, []).append(row)
  print(json.dumps({"section": sec, **row}, default=str), flush=True)

def kernels_of(mk) -> list:
  """The kind="kernel" schedule items of a lazy graph built on NKISHIM (the AST is device-independent), rendered for NKI."""
  out = []
  for si in [si for si in create_schedule([mk().lazydata]) if si.kind == "kernel"]:
    prog = NKIRenderer().render(si.ast, si.var_vals)
    arrays = [b.ensure_allocated().handle[:b.nbytes].view(to_np(b.dtype)) if b.size else np.zeros(1, to_np(b.dtype)) for b in si.bufs]
    out.append((prog, (arrays + [np.zeros(1, np.uint8)] * prog.n_bufs)[:prog.n_bufs]))
  return out

def pct(xs: list[float], q: float) -> float: return float(np.percentile(np.asarray(xs), q))

# ---------------------------------------------------------------- launch overhead

def sec_launch(iters: int) -> None:
  rt = [0.0]
  if HW == "NEURON":
    w = neuron.NKIWorker.get(); rt = []
    for _ in range(iters):
      st = time.perf_counter(); w.call({"op": "version"}); rt.append(time.perf_counter() - st)
  emit("launch", {"what": "worker round trip (no device)", "n": iters, "median_us": statistics.median(rt) * 1e6, "p90_us": pct(rt, 90) * 1e6})
  for label, n in (("tiny add, 128 x f32", 128), ("add, 1 MiB f32", 1 << 18)):
    a = np.arange(n, dtype=np.float32)
    (prog, arrays), = kernels_of(lambda: Tensor(a, device="NKISHIM") + 1)
    for i in range(prog.n_bufs):
      if i not in prog.outs: arrays[i] = a.copy()   # the schedule's input copy is not run here
    prg = (neuron.NeuronHwProgram if HW == "NEURON" else neuron.NKIShimProgram)(prog, prog.src.encode())
    st = time.perf_counter(); dev_s = prg(*arrays, wait=True) or 0.0; first = time.perf_counter() - st
    ok = bool(np.array_equal(arrays[prog.outs[0]].view(np.float32)[:n], a + 1))
    host, dev = [], []
    st = time.perf_counter(); prg(*arrays); probe_s = time.perf_counter() - st
    n_it = iters if probe_s < 0.05 else min(iters, 20)   # a warm call that still costs > 50 ms (a re-trace or re-compile per call?)
    for _ in range(n_it):
      st = time.perf_counter(); d = prg(*arrays, wait=True); host.append(time.perf_counter() - st); dev.append(float(d or 0))
    emit("launch", {"what": label, "kernel": prog.name, "correct": ok, "first_call_s": first, "first_worker_s": dev_s, "second_call_s": probe_s, "n": n_it,
                    "warm_median_us": statistics.median(host) * 1e6, "warm_p10_us": pct(host, 10) * 1e6, "warm_p90_us": pct(host, 90) * 1e6,
                    "worker_median_us": statistics.median(dev) * 1e6, "out_modes": dict(neuron.NeuronHwProgram.out_modes)})

# ---------------------------------------------------------------- N4b: the persistent path

def _us(ts: list[float]) -> dict:
  return {"n": len(ts), "median_us": statistics.median(ts) * 1e6, "p10_us": pct(ts, 10) * 1e6, "p90_us": pct(ts, 90) * 1e6, "min_us": min(ts) * 1e6}

def sec_nrt(iters: int) -> None:
  from fusezap.device import Buffer
  dev = Device["NEURON"] if HW == "NEURON" else None
  if dev is None or not isinstance(dev.runtime, type) or dev.runtime is not neuron.NeuronNeffProgram:
    emit("nrt", {"error": f"the NEURON device is not on the persistent path (FZ_NEURON_RT={os.getenv('FZ_NEURON_RT', 'nrt')}, HW={HW})"}); return
  al = dev.allocator
  rt = []
  for _ in range(iters):
    st = time.perf_counter(); al.call({"op": "stats"}); rt.append(time.perf_counter() - st)
  emit("nrt", {"what": "runtime worker round trip (no device)", **_us(rt)})
  for label, n in (("tiny add, 128 x f32", 128), ("add, 1 MiB f32", 1 << 18)):
    a = np.arange(n, dtype=np.float32)
    x = Tensor(a, device="NEURON").realize()
    (si,) = [s for s in create_schedule([(x + 1).lazydata]) if s.kind == "kernel"]
    prg = dev.program(si)
    c0, s0 = neuron.NEFF_STATS["compiles"], neuron.NEFF_STATS["compile_s"]
    st = time.perf_counter(); dev.run_program(prg, si.bufs); first = time.perf_counter() - st
    row: dict = {"what": label, "kernel": prg.name, "first_launch_s": first, "compiled": neuron.NEFF_STATS["compiles"] - c0,
                 "compile_s": neuron.NEFF_STATS["compile_s"] - s0}
    row["correct"] = bool(np.array_equal(si.bufs[0].numpy()[:n], a + 1))
    host = []
    for _ in range(iters):
      st = time.perf_counter(); dev.run_program(prg, si.bufs); host.append(time.perf_counter() - st)
    row["launch_warm"] = _us(host)
    row["execute_in_worker"] = _us(dev.run_neff(prg, si.bufs, iters=iters))
    key, ins, outs, als = next(iter(prg.loaded.values()))
    hs = [b.handle for b in si.bufs]
    load = al.call({"op": "load", "key": key, "path": ""})
    row["load_s"] = load.get("seconds")
    row["neff_io"] = {"inputs": load["inputs"], "outputs": load["outputs"], "alias": load["alias"]}
    for mode in ("device", "host"):
      try:
        row[f"benchmark_{mode}"] = al.call({"op": "bench", "key": key, "in": {nm: hs[i].id for nm, i in ins.items()},
                                            "out": {nm: hs[i].id for nm, i in outs.items()}, "alias": als, "iters": iters, "mode": mode})["stats"]
      except Exception as e: row[f"benchmark_{mode}"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    # a dependent chain: out = in + 1 with the two buffers swapped each launch, so every launch reads the previous one's output
    o, i_ = si.bufs[0], si.bufs[1]
    o.copyin(memoryview(a.copy()).cast("B")); i_.copyin(memoryview(a.copy()).cast("B"))
    pairs, chain = ((o, i_), (i_, o)), []
    st = time.perf_counter()
    for k in range(200): dev.run_program(prg, pairs[k % 2])
    chain_s = time.perf_counter() - st
    row["chain_200_launches"] = {"total_s": chain_s, "per_launch_us": chain_s / 200 * 1e6,
                                 "correct": bool(np.array_equal(i_.numpy()[:n], a + 200) and np.array_equal(o.numpy()[:n], a + 199))}
    emit("nrt", row)
  for nb in (4 << 10, 1 << 20, 16 << 20):
    b = Buffer("NEURON", nb // 4, dtypes.float32).ensure_allocated()
    h = np.ones(nb // 4, np.float32)
    m = max(5, min(iters, (64 << 20) // nb))
    cin, cout = [], []
    for _ in range(m):
      st = time.perf_counter(); b.copyin(memoryview(h).cast("B")); cin.append(time.perf_counter() - st)
    for _ in range(m):
      st = time.perf_counter(); b.numpy(); cout.append(time.perf_counter() - st)
    emit("nrt", {"what": f"copy {nb} B", "copyin": _us(cin), "copyout": _us(cout), "copyin_GBps": nb / statistics.median(cin) / 1e9,
                 "copyout_GBps": nb / statistics.median(cout) / 1e9})
  emit("nrt", {"what": "worker stats", **al.call({"op": "stats"}), "allocator_calls": dict(al.calls), "neff_stats": dict(neuron.NEFF_STATS)})

# ---------------------------------------------------------------- NEFFs for the nrtpy probe

def sec_neff(out: str) -> None:
  base = os.path.abspath(os.path.join(out, "neff"))
  a = np.arange(128, dtype=np.float32)
  tp, ta = kernels_of(lambda: Tensor(a, device="NKISHIM") + 1)[0]
  cases = [("tiny_add", (tp, [x if i in tp.outs else a.copy() for i, x in enumerate(ta)]))]
  M, K, N = 128, 1024, 512
  x, wk = R.uniform(-1, 1, (M, K)).astype(np.float32), (R.uniform(-1, 1, (K, N)) / 32).astype(np.float32)
  dev = Device["NKISHIM"]
  prg = dev.custom_program("fz_matmul", (1, M, N, K, 0, K, 1, 0, N, 1, 0, N, 1, 0, 0))
  cases.append(("matmul_128x1024x512", (prg.prog, [np.zeros(M * N, np.float32), x.reshape(-1), wk.reshape(-1)])))
  for name, (prog, arrays) in cases:
    d = os.path.join(base, name); os.makedirs(d, exist_ok=True)
    inputs = [x.copy() for x in arrays]
    expect = [x.copy() for x in arrays]
    neuron.NKIShimProgram(prog, prog.src.encode())(*expect)   # the simulator-equivalent result
    try:
      neff, secs = neuron.compile_neff(prog, [x.copy() for x in inputs], "inf2", d)
      np.savez(os.path.join(d, "io.npz"), **{f"in{i}": v for i, v in enumerate(inputs)}, **{f"out{i}": expect[i] for i in prog.outs})
      with open(os.path.join(d, "kernel.py"), "w") as f: f.write(prog.src)
      emit("neff", {"case": name, "kernel": prog.name, "neff_bytes": len(neff), "compile_s": secs, "dir": d, "outs": list(prog.outs)})
    except Exception as e:
      emit("neff", {"case": name, "error": f"{type(e).__name__}: {str(e)[-400:]}"})

# ---------------------------------------------------------------- numerics vs the simulator and float64

def ulp_dist(a: np.ndarray, b: np.ndarray) -> int:
  if a.dtype != np.float32 or not a.size: return 0
  ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
  ia = np.where(ia < 0, -2**31 - ia, ia); ib = np.where(ib < 0, -2**31 - ib, ib)
  fin = np.isfinite(a) & np.isfinite(b)
  return int(np.max(np.abs(ia - ib)[fin])) if fin.any() else 0

def run_on(dev: str, fn, args: list[np.ndarray]) -> np.ndarray:
  return np.asarray(fn(*[Tensor(x, device=dev) for x in args]).numpy())

def numerics_cases() -> list:
  f = lambda *s, lo=-1.0, hi=1.0: R.uniform(lo, hi, s).astype(np.float32)
  i32 = lambda *s, lo, hi: R.integers(lo, hi, s, dtype=np.int64).astype(np.int32)
  big = i32(128, 512, lo=2**24, hi=2**30)
  nan_inf = f(128, 8); nan_inf[3, 1], nan_inf[5, 2], nan_inf[7, 3], nan_inf[9, 4] = np.nan, np.inf, -np.inf, np.nan
  q, k, v = f(1, 14, 64, 64), f(1, 2, 128, 64), f(1, 2, 128, 64)
  def attn(q, k, v): return neuron.attention(q, k, v, causal=True)   # the N3 flash kernel (GQA folded, affine_select mask)
  def attn_ref(q, k, v):
    q, k, v = (t.astype(np.float64) for t in (q, k, v)); k, v = np.repeat(k, 7, 1), np.repeat(v, 7, 1)
    s = q @ k.swapaxes(-1, -2) / math.sqrt(q.shape[-1]); s = np.where(np.tril(np.ones(s.shape[-2:], bool)), s, -np.inf)
    e = np.exp(s - s.max(-1, keepdims=True)); return (e / e.sum(-1, keepdims=True)) @ v
  return [
    ("exp", lambda x: x.exp(), np.exp, [f(128, 2048, lo=-80, hi=80)]),
    ("exp2", lambda x: x.exp2(), np.exp2, [f(128, 2048, lo=-120, hi=120)]),
    ("log2", lambda x: x.log2(), np.log2, [f(128, 2048, lo=1e-30, hi=1e30)]),
    # N5: the N4 run-2 failures. log2 by decade of the input (where ScalarE's log goes wrong; FZ_NKI_FAST_EXP=1 renders it), the
    # exact VectorE exp2/log2 over the whole float32 range with the special values, tanh (= 2*sigmoid(2x) - 1, through exp2)
    ("log2_near1", lambda x: x.log2(), np.log2, [f(128, 2048, lo=0.5, hi=2.0)]),
    ("log2_decades", lambda x: x.log2(), np.log2, [np.exp2(f(128, 2048, lo=-140, hi=128)).astype(np.float32)]),
    ("log2_special", lambda x: x.log2(), np.log2, [np.array([0, -0.0, 1, -1, np.inf, -np.inf, np.nan, 1e-40, 1.4e-45, 3.4e38, 2.0, 0.5] * 32, np.float32)]),
    ("exp2_wide", lambda x: x.exp2(), np.exp2, [f(128, 2048, lo=-160, hi=130)]),
    ("exp2_special", lambda x: x.exp2(), np.exp2, [np.array([0, -0.0, 1, -1, np.inf, -np.inf, np.nan, 128, -149, -126.5, 127.99, 0.5] * 32, np.float32)]),
    ("exp_wide", lambda x: x.exp(), np.exp, [f(128, 2048, lo=-100, hi=88)]),
    ("tanh", lambda x: x.tanh(), np.tanh, [f(128, 2048, lo=-9, hi=9)]),
    ("tanh_small", lambda x: x.tanh(), np.tanh, [f(128, 2048, lo=-0.01, hi=0.01)]),
    ("sqrt_wide", lambda x: x.sqrt(), np.sqrt, [np.exp2(f(128, 2048, lo=-126, hi=127)).astype(np.float32)]),
    ("rmsnorm_rows", lambda x: x * (x * x).mean(-1, keepdim=True).add(1e-6).rsqrt(), lambda x: x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-6),
     [f(64, 896, lo=-4, hi=4)]),
    ("sin_small", lambda x: x.sin(), np.sin, [f(128, 2048, lo=-6.3, hi=6.3)]),
    ("sin_large", lambda x: x.sin(), np.sin, [f(128, 2048, lo=-1e4, hi=1e4)]),
    ("sqrt", lambda x: x.sqrt(), np.sqrt, [f(128, 2048, lo=0, hi=1e6)]),
    ("reciprocal", lambda x: x.reciprocal(), np.reciprocal, [f(128, 2048, lo=0.001, hi=1e3)]),
    ("row_sum_65536", lambda x: x.sum(axis=1), lambda x: x.sum(1), [f(16, 65536)]),
    ("col_sum_partition", lambda x: x.sum(axis=0), lambda x: x.sum(0), [f(1024, 512)]),
    ("softmax_rows", lambda x: x.softmax(-1), lambda x: (lambda e: e / e.sum(-1, keepdims=True))(np.exp(x - x.max(-1, keepdims=True))), [f(256, 4096, lo=-20, hi=20)]),
    ("matmul_fp32_tensore", lambda a, b: a @ b, lambda a, b: a @ b, [f(128, 4096), f(4096, 512)]),
    ("attention_gqa_causal", attn, attn_ref, [q, k, v]),
    ("int32_add_wrap_gpsimd", lambda a, b: a + b, lambda a, b: (a.astype(np.int64) + b).astype(np.int64).astype(np.int32),
     [i32(128, 512, lo=2**30, hi=2**31 - 1), i32(128, 512, lo=2**30, hi=2**31 - 1)]),
    ("int32_max_above_2p24", lambda a, b: a.maximum(b), np.maximum, [big, big + i32(128, 512, lo=-3, hi=4)]),
    ("int32_cmplt_above_2p24", lambda a, b: (a < b).cast(dtypes.int32), lambda a, b: (a < b).astype(np.int32), [big, big + i32(128, 512, lo=-2, hi=3)]),
    ("max_partition_nan_inf", lambda x: x.max(axis=0), lambda x: x.max(0), [nan_inf]),
    ("bitcast_f32_i32", lambda x: x.bitcast(dtypes.int32) + 1, lambda x: x.view(np.int32) + 1, [f(128, 512)]),
  ]

def sec_numerics() -> None:
  for name, fn, ref, args in numerics_cases():
    row: dict = {"case": name}
    try:
      t0 = time.perf_counter()
      hw = run_on(HW, fn, args); row["neuron_s"] = round(time.perf_counter() - t0, 3)
      sim = run_on("NKISHIM", fn, args)
      with np.errstate(all="ignore"):
        r64 = np.asarray(ref(*[a.astype(np.float64) if a.dtype.kind == "f" and not name.startswith("bitcast") else a for a in args]))
      hw_, sim_ = hw.reshape(-1), sim.reshape(-1)
      same = (hw_.view(np.uint8).reshape(hw_.size, -1) == sim_.view(np.uint8).reshape(sim_.size, -1)).all(1) if hw_.size else np.ones(0, bool)
      if hw_.dtype.kind == "f": same |= np.isnan(hw_) & np.isnan(sim_)
      row.update(n=int(hw_.size), dtype=str(hw.dtype), bitwise_equal_to_sim=float(same.mean()) if same.size else 1.0,
                 max_ulp_vs_sim=ulp_dist(hw_, sim_), nan_mismatch=int((np.isnan(hw_) != np.isnan(r64.reshape(-1))).sum()) if hw_.dtype.kind == "f" else 0)
      if hw_.dtype.kind == "f":
        fin = np.isfinite(r64.reshape(-1)) & np.isfinite(hw_)
        err = np.abs(hw_.astype(np.float64) - r64.reshape(-1))[fin]
        sc = float(np.max(np.abs(r64.reshape(-1))[fin])) if fin.any() else 1.0
        sc = sc or 1.0   # normwise: max |error| / max |reference|, as scripts/nki_n3_gate.py
        e_sim = np.abs(sim_.astype(np.float64) - r64.reshape(-1))[fin]
        r_ = r64.reshape(-1)[fin]
        big = np.abs(r_) > 1e-30   # elementwise relative error (normwise hides the small values of a wide-range case)
        el = err[big] / np.abs(r_[big]) if big.any() else np.zeros(0)
        row["max_elem_rel_vs_f64"] = float(el.max()) if el.size else 0.0
        if el.size and len(args) == 1 and args[0].size == hw_.size:
          x_ = args[0].reshape(-1)[fin][big]
          w = np.argsort(el)[-4:][::-1]
          row["worst"] = [[float(x_[j]), float(hw_[fin][big][j]), float(r_[big][j])] for j in w]
          if name.startswith(("log2", "exp", "sqrt")):   # max elementwise rel error per binade group of the input (8 binades)
            with np.errstate(all="ignore"): g = np.floor(np.log2(np.abs(x_.astype(np.float64))) / 8).astype(np.int64) * 8 if name.startswith(("log2", "sqrt")) else np.floor(x_ / 16).astype(np.int64) * 16
            row["by_group"] = {int(k): float(el[g == k].max()) for k in np.unique(g)}
        row.update(max_abs_vs_f64=float(err.max()) if err.size else 0.0, max_rel_vs_f64=float(err.max()) / sc if err.size else 0.0,
                   sim_max_rel_vs_f64=float(e_sim.max()) / sc if e_sim.size else 0.0,
                   inf_mismatch=int((np.isinf(hw_) != np.isinf(r64.reshape(-1))).sum()))
      else:
        row["exact_vs_ref"] = bool(np.array_equal(hw_.astype(np.int64), r64.reshape(-1).astype(np.int64)))
        row["sim_exact_vs_ref"] = bool(np.array_equal(sim_.astype(np.int64), r64.reshape(-1).astype(np.int64)))
    except Exception as e:
      row["error"] = f"{type(e).__name__}: {str(e)[-500:]}"
      row["tb"] = traceback.format_exc()[-1500:]
    emit("numerics", row)
  emit("numerics", {"case": "out_modes", "out_modes": dict(neuron.NeuronHwProgram.out_modes)})

def _fmt(d: dict | None) -> str:
  if not isinstance(d, dict): return str(d)
  if "error" in d: return "ERROR " + str(d["error"])[:120]
  if "median_us" in d: return f"median {d['median_us']:.1f} us (p10 {d['p10_us']:.1f}, p90 {d['p90_us']:.1f})"
  if "mean_ms" in d: return f"mean {d['mean_ms'] * 1e3:.1f} us (min {d['min_ms'] * 1e3:.1f})" + (f", median {d['median_ms'] * 1e3:.1f} us" if "median_ms" in d else "")
  return json.dumps(d, default=str)[:160]

def summary() -> None:
  print("\n==== NKI-N4 probe summary ====")
  for r in OUT["sections"].get("nrt", []):
    w = r.get("what", "")
    if "error" in r and not w: print(f"nrt     ERROR {r['error'][:300]}")
    elif "first_launch_s" in r:
      print(f"nrt     {w:<24} first {r['first_launch_s']:.2f} s (compile {r['compile_s']:.2f} s, load {r.get('load_s') or 0:.3f} s), correct={r['correct']}")
      print(f"nrt     {'':<24} launch warm (fusezap -> worker -> execute): {_fmt(r['launch_warm'])}")
      print(f"nrt     {'':<24} execute in worker: {_fmt(r['execute_in_worker'])}")
      print(f"nrt     {'':<24} nrtpy benchmark device: {_fmt(r.get('benchmark_device'))}; host: {_fmt(r.get('benchmark_host'))}")
      c = r["chain_200_launches"]
      print(f"nrt     {'':<24} dependent chain: {c['per_launch_us']:.1f} us per launch, correct={c['correct']}")
    elif w.startswith("copy"):
      print(f"nrt     {w:<24} copyin {_fmt(r['copyin'])} = {r['copyin_GBps']:.2f} GB/s; copyout {_fmt(r['copyout'])} = {r['copyout_GBps']:.2f} GB/s")
    elif "median_us" in r: print(f"nrt     {w:<24} {_fmt(r)}")
    else: print(f"nrt     {json.dumps(r, default=str)[:300]}")
  for r in OUT["sections"].get("launch", []):
    if "warm_median_us" in r:
      print(f"launch  {r['what']:<24} first {r['first_call_s']:.2f} s, warm median {r['warm_median_us']:.0f} us "
            f"(p10 {r['warm_p10_us']:.0f}, p90 {r['warm_p90_us']:.0f}), correct={r['correct']}, out_modes={r['out_modes']}")
    elif "median_us" in r: print(f"launch  {r['what']:<24} median {r['median_us']:.0f} us (subtract from the above)")
    else: print(f"launch  {r}")
  for r in OUT["sections"].get("neff", []): print(f"neff    {r.get('case')}: {r.get('neff_bytes', r.get('error'))}")
  for r in OUT["sections"].get("numerics", []):
    if "error" in r: print(f"numerics {r['case']:<26} ERROR {r['error'][:160]}")
    elif "bitwise_equal_to_sim" in r:
      extra = (f"normwise rel err vs f64 {r.get('max_rel_vs_f64', 0):.2e} (sim {r.get('sim_max_rel_vs_f64', 0):.2e}), max ulp vs sim {r['max_ulp_vs_sim']}"
               if "max_rel_vs_f64" in r else f"exact {r['exact_vs_ref']} (sim {r['sim_exact_vs_ref']})")
      print(f"numerics {r['case']:<26} bitwise==sim {r['bitwise_equal_to_sim'] * 100:6.2f}%  {extra}  nan/inf mismatches "
            f"{r.get('nan_mismatch', 0)}/{r.get('inf_mismatch', 0)}" + (f"  elementwise rel {r['max_elem_rel_vs_f64']:.2e}" if "max_elem_rel_vs_f64" in r else ""))
      if r.get("worst") and r.get("max_elem_rel_vs_f64", 0) > 1e-6: print(f"numerics {'':<26} worst (x, hw, ref): {r['worst']}")
      if r.get("by_group") and r.get("max_elem_rel_vs_f64", 0) > 1e-6:
        print(f"numerics {'':<26} max rel by input group: " + ", ".join(f"{k}:{v:.1e}" for k, v in r["by_group"].items()))

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default="results/nki_n4")
  ap.add_argument("--only", default="nrt,neff,numerics")
  ap.add_argument("--iters", type=int, default=200)
  ap.add_argument("--json", default="probe.json", help="file name under --out")
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  OUT["neuron_available"], OUT["hw_tier"], OUT["neuron_rt"] = Device.available("NEURON"), HW, os.getenv("FZ_NEURON_RT", "nrt")
  try: OUT["worker"] = neuron.NKIWorker.get().call({"op": "version"})
  except Exception as e: OUT["worker"] = {"error": str(e)}
  print(json.dumps({"neuron_available": OUT["neuron_available"], "worker": OUT["worker"]}), flush=True)
  todo = a.only.split(",")
  for name, fn in (("nrt", lambda: sec_nrt(a.iters)), ("launch", lambda: sec_launch(a.iters)), ("neff", lambda: sec_neff(a.out)),
                   ("numerics", sec_numerics)):
    if name not in todo: continue
    try: fn()
    except Exception as e: emit(name, {"error": f"{type(e).__name__}: {str(e)[-600:]}", "tb": traceback.format_exc()[-2000:]})
    with open(os.path.join(a.out, a.json), "w") as f: json.dump(OUT, f, indent=1, default=str)
  summary()

if __name__ == "__main__": main()
