"""NKI-N3 gate: the NEURON matmul and attention kernels at the shapes Qwen2.5-0.5B and Qwen2.5-7B use (incl. GQA), on a
NEURON tier, against a float64 numpy reference, plus an inf2 NEFF compile of each kernel (docs/status/NKI-N3.md).

  FZ_DEVICE=NKISIM uv run python scripts/nki_n3_gate.py [--neff] [--quick] [--json results/nki_n3_gate.json]
  FZ_DEVICE=NKISHIM uv run python scripts/nki_n3_gate.py --quick          # the Mac shim (no NEFF)
  FZ_DEVICE=NEURON uv run python scripts/nki_n3_gate.py                    # inf2: the persistent path (N5 fix; run_s is a warm launch)

Matmul cases call the kernel the scheduler hook builds for x.linear(w) directly on numpy buffers (the big weights would
otherwise be copied several times on a 4 GB pod): K-major weights (`neuron.kmajor`, contiguous moving rows) for every
linear, the torch (N, K) layout (strided DMA) for two. Attention cases run the direct entry point `neuron.attention`
(causal by affine_select, decode by a position tensor, GQA folded) and the scheduler hook on llama.py's composed graph
(WHERE(position mask, scores, -inf) -> fz_attention_mask) and on plain / causal scaled_dot_product_attention.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusezap import Tensor
from fusezap.device import Buffer, Device, neuron
from fusezap.dtype import dtypes, to_np
from fusezap.schedule import create_schedule
from fusezap.codegen.nki_kernels import TN

QWEN = {"0.5B": dict(D=896, I=4864, Hq=14, Hkv=2, hd=64, V=151936), "7B": dict(D=3584, I=18944, Hq=28, Hkv=4, hd=128, V=152064)}
R = np.random.default_rng(0)
ROWS: list[dict] = []

def neff(prog, arrays) -> dict:
  try:
    b, s = neuron.compile_neff(prog, arrays, "inf2")
    return {"neff_bytes": len(b), "neff_s": round(s, 2)}
  except Exception as e: return {"neff_err": f"{type(e).__name__}: {str(e)[-300:]}"}

def record(row: dict, out: np.ndarray, ref: np.ndarray, tol: float, prog=None, arrays=None, do_neff=False) -> None:
  err = float(np.max(np.abs(out.astype(np.float64) - ref))) if out.size else 0.0
  scale = float(np.max(np.abs(ref))) or 1.0
  row.update(max_abs=err, rel=err / scale, tol=tol, ok=bool(np.isfinite(out).all() and err / scale <= tol))
  if do_neff and prog is not None: row.update(neff(prog, arrays))
  ROWS.append(row)
  print(json.dumps(row), flush=True)

SIM_BYTES = 256 << 20   # the nki 0.6 simulator peaks at ~7x its HBM inputs (224 MiB weight -> 1.8 GiB RSS): 4 GB pod

def mm_case(dev, model: str, name: str, M: int, K: int, N: int, layout: str, do_neff: bool) -> None:
  """Simulate x (M, K) @ w.T for w (N, K), on the largest N-slice whose weight fits SIM_BYTES; compile the full shape."""
  full = N
  if K * N * 4 > SIM_BYTES and Device.DEFAULT.split(":")[0] != "NEURON": N = SIM_BYTES // (4 * K) // TN * TN   # hardware: full shape
  x = R.uniform(-1, 1, (M, K)).astype(np.float32)
  w = (R.uniform(-1, 1, (N, K)) / math.sqrt(K)).astype(np.float32)
  def params(N: int) -> tuple[int, ...]:   # what _mm_spec gives x @ kmajor(w) / x.linear(w): (Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob)
    p = (1, M, N, K, 0, K, 1, 0, N, 1, 0, N, 1, 0, 0) if layout == "kmajor" else (1, M, N, K, 0, K, 1, 0, 1, K, 0, N, 1, 0, 0)
    return p[:5] + (0,) + p[6:11] + (0,) + p[12:] if M == 1 else p   # _canon: a size-1 axis has stride 0
  wb, p = (np.ascontiguousarray(w.T) if layout == "kmajor" else w), params(N)
  prg = dev.custom_program("fz_matmul", p)
  arrays = [np.zeros(M * N, np.float32), x.reshape(-1), wb.reshape(-1)]
  # device Buffers + run_program on every tier: the persistent NEURON path (NeuronNeffProgram) launches on device buffers only
  # (N4 run 2: calling the program on numpy arrays raised "NeuronNeffProgram runs on device buffers")
  bufs_ = [Buffer(dev.name, a.size, dtypes.float32).copyin(memoryview(a).cast("B")) for a in arrays]
  dev.run_program(prg, tuple(bufs_))   # first launch: compile (cached) + load on NEURON
  st = time.perf_counter(); dev.run_program(prg, tuple(bufs_)); dev.synchronize(); secs = time.perf_counter() - st
  arrays[0] = bufs_[0].numpy()
  ref = x.astype(np.float64) @ w.T.astype(np.float64)
  prog, bufs = prg.prog, arrays
  if N != full:   # the NEFF is the full-shape kernel; the worker allocates its buffers (untouched zero pages)
    prog = dev.custom_program("fz_matmul", params(full)).prog
    bufs = [(M * full, np.float32), (M * K, np.float32), (K * full, np.float32)]
  shape = f"M={M} K={K} N={full}" + (f" (simulated on N={N}; NEFF full)" if N != full else "")
  record({"kernel": "fz_matmul", "model": model, "case": f"{name} {layout}", "shape": shape, "run_s": round(secs, 5)},
         arrays[0].reshape(M, N), ref, 2e-6, prog, bufs, do_neff)

def attn_ref(q, k, v, vis) -> np.ndarray:
  G = q.shape[1] // k.shape[1]
  kk, vv = np.repeat(k, G, 1).astype(np.float64), np.repeat(v, G, 1).astype(np.float64)
  s = q.astype(np.float64) @ kk.transpose(0, 1, 3, 2) / math.sqrt(q.shape[-1])
  s = np.where(vis, s, -np.inf)
  e = np.exp(s - s.max(-1, keepdims=True))
  return (e / e.sum(-1, keepdims=True)) @ vv

LAUNCHED: list = []

def capture(dev) -> None:
  """Record each custom kernel launch (program, zero buffers of the same sizes) so it can be compiled to a NEFF afterwards."""
  run = dev.run_program
  def rec(prg, bufs):
    if prg.prog.name.startswith("fz_"): LAUNCHED.append((prg, [np.zeros(max(b.size, 1), to_np(b.dtype)) for b in bufs[:prg.prog.n_bufs]]))
    return run(prg, bufs)
  dev.run_program = rec

def attn_case(dev, model: str, kind: str, T: int, S: int, start: int, do_neff: bool) -> None:
  c = QWEN[model]
  Hq, Hkv, hd = c["Hq"], c["Hkv"], c["hd"]
  q = R.standard_normal((1, Hq, T, hd)).astype(np.float32)
  K, V = R.standard_normal((1, Hkv, S, hd)).astype(np.float32), R.standard_normal((1, Hkv, S, hd)).astype(np.float32)
  vis = np.arange(S)[None, :] <= np.arange(start, start + T)[:, None]
  LAUNCHED.clear()
  st = time.perf_counter()
  if kind == "causal": out, label = neuron.attention(Tensor(q), Tensor(K), Tensor(V), causal=True, coff=start).numpy(), "entry causal (affine_select)"
  elif kind == "pos":
    out = neuron.attention(Tensor(q), Tensor(K), Tensor(V), pos=Tensor(np.arange(start, start + T, dtype=np.int32))).numpy()
    label = "entry pos (iota vs position tensor)"
  elif kind == "llama":   # fusezap/serve/models/llama.py's composed attention: the hook must take it (exception 3)
    G = Hq // Hkv
    def graph():   # built afresh for each use: create_schedule marks a graph's buffers realized
      pos = Tensor(np.arange(start + T + 1, dtype=np.int32))
      mask = pos.shrink(((0, S),)).reshape(1, S) <= pos.shrink(((start, start + T),)).reshape(T, 1)
      s = (Tensor(q).reshape(1, Hkv, G, T, hd) @ Tensor(K).reshape(1, Hkv, 1, S, hd).transpose(-1, -2)) * (1.0 / math.sqrt(hd))
      return (mask.where(s, float("-inf")).softmax(-1) @ Tensor(V).reshape(1, Hkv, 1, S, hd)).reshape(1, Hq, T, hd)
    ks = [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([graph().lazydata]) if si.kind != "copy"]
    assert "fz_attention_mask" in ks, f"llama attention graph missed the hook: {ks}"
    st = time.perf_counter()
    o = graph()
    out, label = o.numpy(), "hook: llama.py graph -> fz_attention_mask"
  elif kind in ("sdpa", "sdpa_causal"):   # no GQA in torch's default sdpa: K/V repeated per q head
    Kr, Vr = np.repeat(K, Hq // Hkv, 1), np.repeat(V, Hq // Hkv, 1)
    f = lambda: Tensor(q).scaled_dot_product_attention(Tensor(Kr), Tensor(Vr), is_causal=kind == "sdpa_causal")
    ks = [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([f().lazydata]) if si.kind != "copy"]
    want = "fz_attention_mask" if kind == "sdpa_causal" else "fz_attention"
    assert want in ks, f"sdpa missed the hook: {ks}"
    st = time.perf_counter()
    out, label = f().numpy(), f"hook: sdpa{' is_causal' if kind == 'sdpa_causal' else ''} -> {want}"
    if kind == "sdpa": vis = np.ones((T, S), bool)
  else: raise ValueError(kind)
  secs = time.perf_counter() - st
  att = [(prg, arrs) for prg, arrs in LAUNCHED if prg.prog.name.startswith("fz_attention")]
  assert len(att) == 1, f"expected one attention launch, got {[p.prog.name for p, _ in LAUNCHED]}"
  prg, arrays = att[0]
  row = {"kernel": prg.prog.name.rsplit("_", 5)[0], "model": model, "case": label, "shape": f"Hq={Hq} Hkv={Hkv} hd={hd} T={T} S={S} start={start}",
         "run_s": round(secs, 2)}
  record(row, out, attn_ref(q, K, V, vis), 1e-5, prg.prog, arrays, do_neff)

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--neff", action="store_true", help="also compile each kernel to an inf2 NEFF (needs neuronx-cc in .venv-nki)")
  ap.add_argument("--quick", action="store_true", help="0.5B shapes and short sequences only")
  ap.add_argument("--json", default=None)
  ap.add_argument("--only", default="", help="mm or attn")
  a = ap.parse_args()
  dev = Device[Device.DEFAULT]
  assert isinstance(dev, neuron.NEURONDevice), f"run on a NEURON tier (FZ_DEVICE=NKISIM or NKISHIM), got {Device.DEFAULT}"
  capture(dev)
  print(f"tier {Device.DEFAULT}", flush=True)
  models = ["0.5B"] if a.quick else ["0.5B", "7B"]
  if a.only in ("", "mm"):
    for model in models:
      c = QWEN[model]
      lin = [("qkv", c["D"], c["D"] + 2 * c["Hkv"] * c["hd"]), ("o_proj", c["D"], c["D"]), ("gate_up", c["D"], 2 * c["I"]), ("down", c["I"], c["D"])]
      for M in ((1,) if a.quick else (1, 128)):
        for name, K, N in lin:
          if a.quick and name == "gate_up": continue
          mm_case(dev, model, name, M, K, N, "kmajor", a.neff)
      if not a.quick:
        for name, K, N in (lin[0], lin[3]): mm_case(dev, model, name, 128, K, N, "torch", a.neff)
        mm_case(dev, model, "lm_head", 1, c["D"], c["V"], "kmajor", a.neff)
  if a.only in ("", "attn"):
    for model in models:
      cases = [("causal", 128, 128, 0), ("pos", 1, 1024, 1000), ("llama", 1, 512, 511), ("llama", 64, 64, 0), ("sdpa", 64, 64, 0)]
      if not a.quick:
        cases += [("causal", 640, 640, 0), ("causal", 128, 640, 512), ("pos", 4, 300, 296), ("llama", 128, 128, 0),
                  ("sdpa_causal", 128, 128, 0)]
      for kind, T, S, start in cases: attn_case(dev, model, kind, T, S, start, a.neff)
  bad = [r for r in ROWS if not r["ok"] or "neff_err" in r]
  print(f"\n{len(ROWS)} cases, {len(bad)} failed", flush=True)
  if a.json:
    os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
    with open(a.json, "w") as f: json.dump(ROWS, f, indent=1)
  sys.exit(1 if bad else 0)

if __name__ == "__main__": main()
