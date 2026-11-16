#!/usr/bin/env python3
"""I105 A2: the sparse MoE decode kernels (fusezap/codegen/nki_moe.py) on the device, before any model runs on them.

The indirect-DMA address forms (FZ_MOE_IDMA: vec, vecadd, scalar) are written against this repo's model of nki 0.6's
`ap(vector_offset= / scalar_offset=)`, which is unverified on hardware. For each form in turn, at OLMoE's real layer shapes
(D 2048, 64 experts, top 8, bf16 weights; --width 512 = one rank's share under the sliced TP=2 layout, 1024 = one core),
1 and 4 rows: render, compile and launch MoeGateUpSparse and MoeDownSparse on random data, compare with the float64
references (and with the dense kernels, MatmulBF16 over every expert: bit for bit on NKISHIM), and time launches (wall per
launch, a round trip included) next to the dense gate_up / down matmuls. The first form that compiles and matches is
written to --json ("variant") and printed as `chosen variant: <form>`; the job exports it as FZ_MOE_IDMA. Exit 3 when no
form works (the job then skips the sparse phases: the dense path stays).

  FZ_DEVICE=NKISHIM .venv/bin/python scripts/olmoe_sparse_selftest.py --width 512 --json results/sparse_selftest.json   # Mac dry run
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
import numpy as np

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--width", type=int, default=512)
  ap.add_argument("--rows", default="1,4")
  ap.add_argument("--variants", default="vec,vecadd,scalar")
  ap.add_argument("--reps", type=int, default=10)
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  from fusezap.tensor import Tensor
  from fusezap.dtype import dtypes
  from fusezap.serve.weights import f32_to_bf16, bf16_to_f32
  from fusezap.serve.models.moe import sparse_gate_up, sparse_down
  from fusezap.serve.neuron_kv import matmul_bf16
  from fusezap.codegen.nki_moe import gate_up_ref, down_ref, picks_ref
  dev = os.environ.get("FZ_DEVICE", "NEURON")
  D, E, K, Iw = 2048, 64, 8, a.width
  rng = np.random.default_rng(0)
  def bf(shape):
    x = rng.standard_normal(shape, dtype=np.float32); x *= 0.03
    b = np.empty(shape, np.uint16)
    for i in range(0, shape[0], 128): b[i:i + 128] = f32_to_bf16(x[i:i + 128]); x[i:i + 128] = bf16_to_f32(b[i:i + 128])
    return x, b
  wgu, wgu_b = bf((D, 2 * E * Iw))
  wd, wd_b = bf((E * Iw, D))
  Wgu = Tensor(wgu_b, device=dev, dtype=dtypes.bfloat16).realize()
  Wd = Tensor(wd_b, device=dev, dtype=dtypes.bfloat16).realize()
  res: dict = {"device": dev, "D": D, "E": E, "K": K, "width": Iw, "variants": {}, "variant": None}
  def timed(fn, reps):
    fn().numpy()
    t = time.perf_counter()
    for _ in range(reps): out = fn()
    out.numpy()
    return round(1e3 * (time.perf_counter() - t) / reps, 3)
  def save():
    if a.json:
      os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
      with open(a.json, "w") as f: json.dump(res, f, indent=1)
  cases = {}
  for N in (int(x) for x in a.rows.split(",")):
    h = rng.standard_normal((N, D), dtype=np.float32)
    p = rng.random((N, E)).astype(np.float32)
    order = np.argsort(-p, -1, kind="stable")
    rank = np.empty_like(order); np.put_along_axis(rank, order, np.arange(E)[None].repeat(N, 0), -1)
    sel = rank < K
    key = np.where(sel, np.cumsum(sel, -1) - 1, E).astype(np.float32)
    w = np.where(sel, p, 0).astype(np.float32)
    cases[N] = (h, key, w, *(Tensor(x, device=dev).realize() for x in (h, key, w)))
  for var in a.variants.split(","):   # 1. correctness of each form, sparse kernels only (the dense comparison is extra, below)
    r: dict = {"rows": {}}
    res["variants"][var] = r
    try:
      ok = True
      for N, (h, key, w, H, KEY, Wt) in cases.items():
        t0 = time.perf_counter()
        G = sparse_gate_up(H, KEY, Wgu, E, K, var).numpy()
        A = (G[..., :Iw] / (1 + np.exp(-G[..., :Iw])) * G[..., Iw:]).astype(np.float32)
        Y = sparse_down(Tensor(A, device=dev).realize(), KEY, Wt, Wd, E, K, var).numpy()
        rg, ry = gate_up_ref(h, key, wgu, E, Iw, K, True), down_ref(A, key, w, wd, Iw, K, True)
        eg, ey = float(np.abs(G - rg).max()), float(np.abs(Y - ry).max())
        sg, sy = float(np.abs(rg).max()), float(np.abs(ry).max())
        good = bool(np.isfinite(G).all() and np.isfinite(Y).all() and eg <= 2e-3 * max(sg, 1e-6) + 1e-5 and ey <= 2e-3 * max(sy, 1e-6) + 1e-5)
        r["rows"][N] = {"first_call_s": round(time.perf_counter() - t0, 1), "gate_up_max_err": eg, "gate_up_scale": sg, "down_max_err": ey,
                        "down_scale": sy, "ok": good, "G": G, "A": A, "Y": Y}
        print(f"[{var}] rows {N}: ok {good}, gate_up max err {eg:.3g} (scale {sg:.3g}), down max err {ey:.3g} (scale {sy:.3g}), "
              f"first call (compiles) {r['rows'][N]['first_call_s']} s", flush=True)
        ok = ok and good
      r["ok"] = ok
    except Exception as e:
      r["ok"], r["error"] = False, f"{type(e).__name__}: {str(e)[:2000]}"
      print(f"[{var}] FAILED: {r['error'][:600]}", flush=True)
      traceback.print_exc()
    if r["ok"]:
      res["variant"] = var
      break   # the first working form; the others are not compiled
  keep = {N: {k: rec.pop(k, None) for k in ("G", "A", "Y")} for var in res["variants"] for N, rec in res["variants"][var]["rows"].items()}
  save()
  print(f"chosen variant: {res['variant']}", flush=True)
  # 2. extras with the chosen form: bit for bit vs the dense kernels on the same inputs, and ms per launch (a round trip each)
  var = res["variant"]
  if var:
    for N, (h, key, w, H, KEY, Wt) in cases.items():
      rec = res["variants"][var]["rows"][N]
      try:
        G, A, Y = keep[N]["G"], keep[N]["A"], keep[N]["Y"]
        gu = matmul_bf16(H, Wgu).numpy()
        ex = picks_ref(key, K)
        dense_g = np.stack([np.concatenate([gu[n, e * Iw:(e + 1) * Iw], gu[n, E * Iw + e * Iw:E * Iw + (e + 1) * Iw]]) for n in range(N) for e in ex[n]]).reshape(G.shape)
        afull = np.zeros((N, E, Iw), np.float32)
        for n in range(N): afull[n, ex[n]] = A[n] * w[n, ex[n]][:, None]
        AF = Tensor(afull.reshape(N, E * Iw), device=dev).realize()
        dense_y = matmul_bf16(AF, Wd).numpy()
        A_t = Tensor(A, device=dev).realize()
        rec.update(gate_up_bitwise_vs_dense=bool(np.array_equal(G.view(np.int32), dense_g.view(np.int32))),
                   down_bitwise_vs_dense=bool(np.array_equal(Y.view(np.int32), dense_y.view(np.int32))),
                   down_max_diff_vs_dense=float(np.abs(Y - dense_y).max()),
                   ms_per_launch={"sparse_gate_up": timed(lambda: sparse_gate_up(H, KEY, Wgu, E, K, var), a.reps),
                                  "sparse_down": timed(lambda: sparse_down(A_t, KEY, Wt, Wd, E, K, var), a.reps),
                                  "dense_gate_up": timed(lambda: matmul_bf16(H, Wgu), a.reps),
                                  "dense_down": timed(lambda: matmul_bf16(AF, Wd), a.reps)})
      except Exception as e: rec["extras_error"] = f"{type(e).__name__}: {str(e)[:600]}"
      print(f"[{var}] rows {N} vs dense: {json.dumps({k: v for k, v in rec.items() if k not in ('ok',)})}", flush=True)
      save()
  return 0 if res["variant"] else 3

if __name__ == "__main__": sys.exit(main())
