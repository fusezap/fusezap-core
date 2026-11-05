"""P3-GEMM: FuseZap CUDA tensor-core GEMMs vs torch (cuBLAS / torch._scaled_mm).

  uv run python bench/bench_cuda_gemm.py                 squares 1024-8192 (bf16, f16: nn = a @ b, nt = a @ w^T; fp8 nt),
                                                         Qwen2.5-7B projection shapes (M = 512, 4096) in bf16, f16 and fp8
  uv run python bench/bench_cuda_gemm.py --quick         4096 only + two projection shapes
  uv run python bench/bench_cuda_gemm.py --configs ...   time every candidate tile config of one case, each in a
                                                         subprocess with a timeout (a hung kernel cannot stall the run)

Both sides run on the same torch-allocated device buffers (FuseZap's kernels take raw device addresses through
fusezap.device.cuda_kernels.gemm, in torch's primary context, legacy default stream) and are timed the same way: CUDA
events around a burst of back-to-back launches (>= ~25 ms), best of interleaved rounds. FuseZap launches through a
pre-bound launcher (parameters built once); torch through torch.matmul(out=) / torch._scaled_mm. Tile configs are
tuned once per shape (FZ_BEAM=1, cached in ~/.cache/fusezap/cuda_gemm_tune.json) unless --no-tune.
Every result is checked against an f32 torch reference of the same (dequantized) inputs.
The first table times the raw kernels (ck.gemm). The gate is measured on FuseZap's Tensor matmul: bf16 4096^2 `a @ b` on CUDA tensors after
cuda_kernels.install(dev), JIT-replayed (the scheduler hook routes it to the 16-bit kernel; the f32 -> bf16 output cast is
a second kernel and is timed too), plus eager, nt (`a @ w.T`) and f16 rows. Each Tensor row records its schedule.
Exit: 1 if a result is wrong or the Tensor path did not reach the 16-bit kernel, or on an H100 (sm_90) if the gate row is
below 0.60x cuBLAS (the P3-GEMM gate).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
GATE = ("bf16", "nn", 4096, 4096, 4096)
QWEN7B = [("qkv", 4608, 3584), ("o", 3584, 3584), ("gate_up", 37888, 3584), ("down", 3584, 18944)]

def tdt(kind: str):
  import torch
  return {"bf16": torch.bfloat16, "f16": torch.float16, "e4m3": torch.float8_e4m3fn}[kind]

def make(kind: str, layout: str, M: int, N: int, K: int, seed: int = 0):
  """torch inputs: (a, b, scales or None); b is (K, N) for nn, (N, K) for nt."""
  import torch
  g = torch.Generator(device="cuda").manual_seed(seed)
  a = torch.randn(M, K, device="cuda", generator=g)
  b = torch.randn(*((K, N) if layout == "nn" else (N, K)), device="cuda", generator=g)
  if kind != "e4m3": return a.to(tdt(kind)), b.to(tdt(kind)), None
  sa, sb = a.abs().max() / 448, b.abs().max() / 448
  return (a / sa).to(torch.float8_e4m3fn), (b / sb).to(torch.float8_e4m3fn), (sa.float().reshape(()), sb.float().reshape(()))

def reference(kind, layout, a, b, sc):
  bf = b.float()
  ref = a.float() @ (bf if layout == "nn" else bf.t())
  return ref * (sc[0] * sc[1]) if sc is not None else ref

def fz_launcher(dev, kind, layout, a, b, sc, c, impl=None, cfg=None):
  import fusezap.device.cuda_kernels as ck
  M, K = a.shape
  N = b.shape[1] if layout == "nn" else b.shape[0]
  extra = dict(sa=sc[0].data_ptr(), sb=sc[1].data_ptr()) if sc is not None else {}
  args = (dev, c.data_ptr(), a.data_ptr(), b.data_ptr(), M, N, K)
  kw = dict(kind=kind, out="f16" if kind == "f16" else "bf16", nn=layout == "nn", impl=impl, cfg=cfg, **extra)
  ck.gemm(*args, **kw, wait=True)   # compile (+ tune when FZ_BEAM) outside the timing
  return ck.gemm(*args, **kw, bind=True)

def torch_fn(kind, layout, a, b, sc, c):
  import torch
  if kind == "e4m3": return lambda: torch._scaled_mm(a, b.t(), scale_a=sc[0], scale_b=sc[1], out_dtype=torch.bfloat16)
  bb = b if layout == "nn" else b.t()
  return lambda: torch.matmul(a, bb, out=c)

def burst(fn, min_ms: float = 25.0) -> float:
  """Seconds per call: CUDA events around n back-to-back calls, n sized to ~min_ms."""
  import torch
  torch.cuda.synchronize()
  e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  n = 1
  while True:
    e0.record()
    for _ in range(n): fn()
    e1.record()
    e1.synchronize()
    ms = e0.elapsed_time(e1)
    if ms >= min_ms or n >= 4096: return ms / n * 1e-3
    n = min(4096, max(n * 2, int(n * min_ms / max(ms, 1e-3)) + 1))

def run_case(dev, kind, layout, M, N, K, rounds=5, impl=None, cfg=None) -> dict:
  import torch
  a, b, sc = make(kind, layout, M, N, K)
  c_fz = torch.empty(M, N, device="cuda", dtype=torch.bfloat16 if kind == "e4m3" else tdt(kind))
  c_t = torch.empty_like(c_fz)
  fz = fz_launcher(dev, kind, layout, a, b, sc, c_fz, impl, cfg)
  th = torch_fn(kind, layout, a, b, sc, c_t)
  fz(); out_t = th(); torch.cuda.synchronize()
  if kind == "e4m3": c_t = out_t[0] if isinstance(out_t, tuple) else out_t
  ref = reference(kind, layout, a, b, sc)
  scale = ref.abs().max().item()
  err = (c_fz.float() - ref).abs().max().item() / scale
  err_t = (c_t.float() - ref).abs().max().item() / scale
  tf, tt = [], []
  for _ in range(rounds):
    tf.append(burst(lambda: fz())); tt.append(burst(th))
  fl = 2 * M * N * K
  r = dict(kind=kind, layout=layout, M=M, N=N, K=K, fz_tflops=fl / min(tf) / 1e12, torch_tflops=fl / min(tt) / 1e12,
           err=err, torch_err=err_t, ok=bool(err <= max(1e-2, 2 * err_t)))
  r["ratio"] = r["fz_tflops"] / r["torch_tflops"]
  return r

def tensor_case(dev, kind: str, layout: str, M: int, N: int, K: int, rounds: int = 5, use_jit: bool = True) -> dict:
  """The SPEC §11.2 gate path: FuseZap Tensor matmul (`a @ b`, or `a @ w.T` for nt) on CUDA tensors, through the scheduler
  hook that cuda_kernels.install(dev) offers, against torch.matmul on the same values. Records the scheduled kernels."""
  import numpy as np, torch
  import fusezap as fz
  import fusezap.device.cuda_kernels as ck
  from fusezap import Tensor, dtypes
  from fusezap.schedule import create_schedule
  assert ck.install(dev), "the CUDA matmul hook needs sm_80+"
  dt = {"bf16": dtypes.bfloat16, "f16": dtypes.float16}[kind]
  rng = np.random.default_rng(0)
  a = Tensor(rng.standard_normal((M, K), dtype=np.float32), device="CUDA").cast(dt).realize()
  b = Tensor(rng.standard_normal((K, N) if layout == "nn" else (N, K), dtype=np.float32), device="CUDA").cast(dt).realize()
  fn = (lambda x, y: x @ y) if layout == "nn" else (lambda x, y: x @ y.T)
  sched = [si.ast[0].kernel.name if si.kind == "custom" else si.kind for si in create_schedule([fn(a, b).lazydata]) if si.kind != "copy"]
  f = fz.jit(fn) if use_jit else (lambda x, y: fn(x, y).realize())
  for _ in range(3): out = f(a, b)
  ta, tb = (torch.from_numpy(t.cast(dtypes.float32).numpy()).cuda() for t in (a, b))
  ref = ta @ (tb if layout == "nn" else tb.t())
  ta, tb = ta.to(tdt(kind)), tb.to(tdt(kind))
  c_t = torch.empty(M, N, device="cuda", dtype=tdt(kind))
  th = torch_fn(kind, layout, ta, tb, None, c_t)
  th(); torch.cuda.synchronize()
  dev.synchronize()
  scale = ref.abs().max().item()
  err = (torch.from_numpy(out.cast(dtypes.float32).numpy()).cuda() - ref).abs().max().item() / scale
  err_t = (c_t.float() - ref).abs().max().item() / scale
  tf, tt = [], []
  for _ in range(rounds):
    tf.append(burst(lambda: f(a, b))); tt.append(burst(th))
  fl = 2 * M * N * K
  r = dict(kind=kind, layout=layout, M=M, N=N, K=K, path="Tensor " + ("jit" if use_jit else "eager"), schedule=sched,
           fz_tflops=fl / min(tf) / 1e12, torch_tflops=fl / min(tt) / 1e12, err=err, torch_err=err_t,
           ok=bool(err <= max(1e-2, 2 * err_t)) and "fz_matmul_mma16" in sched)
  r["ratio"] = r["fz_tflops"] / r["torch_tflops"]
  return r

def gpu_info() -> dict:
  import torch
  from fusezap.device import Device
  dev = Device["CUDA"]
  return dict(gpu=dev.gpu_name, cc=list(dev.cc), torch=torch.__version__, cuda=torch.version.cuda)

def sweep(args) -> int:
  """--configs: every candidate config of one case, each in its own subprocess."""
  import fusezap.device.cuda_kernels as ck
  from fusezap.device import Device
  dev = Device["CUDA"]
  kind, layout, (M, N, K) = args.kind, args.layout, args.shape
  impl = args.impl or ("wgmma" if tuple(dev.cc) == (9, 0) else "mma")
  cands = ck.wg_candidates(kind, dev) if impl == "wgmma" else ck.mma_candidates(dev, kind, layout == "nn")
  print(f"{gpu_info()}\nsweep {impl} {kind} {layout} {M}x{N}x{K}: {len(cands)} configs")
  rows = []
  for cfg in cands:
    cmd = [sys.executable, __file__, "--one", impl, kind, layout, str(M), str(N), str(K), json.dumps(list(cfg.__dict__.values()))]
    t0 = time.time()
    try: p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
      print(f"  {cfg}: TIMEOUT ({args.timeout}s)"); rows.append(dict(cfg=str(cfg), status="timeout")); continue
    line = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode or not line:
      print(f"  {cfg}: FAILED rc={p.returncode}\n" + "\n".join(("    " + l) for l in (p.stdout + p.stderr).strip().splitlines()[-25:]))
      rows.append(dict(cfg=str(cfg), status="failed")); continue
    r = json.loads(line[-1])
    r.update(cfg=str(cfg), status="ok", secs=round(time.time() - t0, 1))
    rows.append(r)
    print(f"  {cfg}: {r['fz_tflops']:7.1f} TFLOPS vs torch {r['torch_tflops']:7.1f} = {r['ratio']:.2f}x  err {r['err']:.1e}  "
          f"regs {r.get('regs')} spill {r.get('spills')} {'OK' if r['ok'] else 'WRONG'}")
  out = Path(args.json_out or ROOT / "results" / f"cuda_gemm_sweep_{impl}_{kind}_{layout}_{M}x{N}x{K}.json")
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(dict(info=gpu_info(), rows=rows), indent=1))
  return 0

def one(argv) -> int:
  import fusezap.device.cuda_kernels as ck
  from fusezap.device import Device
  from fusezap.device.cuda_kernels import mma, wgmma
  impl, kind, layout, M, N, K, cfgj = argv
  dev = Device["CUDA"]
  vals = json.loads(cfgj)
  cfg = wgmma.WgCfg(*vals) if impl == "wgmma" else mma.MmaCfg(*vals)
  r = run_case(dev, kind, layout, int(M), int(N), int(K), rounds=3, impl=impl, cfg=cfg)
  k = next(iter(v for n, v in dev._fz_gemm.items() if n.startswith(f"fz_{impl}")))
  r.update(regs=k.regs, spills=k.spills)
  print(json.dumps(r))
  return 0

def main() -> int:
  if len(sys.argv) > 1 and sys.argv[1] == "--one": return one(sys.argv[2:])
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--quick", action="store_true")
  ap.add_argument("--no-tune", action="store_true", help="default tile configs instead of FZ_BEAM tuning")
  ap.add_argument("--configs", action="store_true", help="sweep the candidate configs of one case (--kind/--layout/--shape)")
  ap.add_argument("--kind", default="bf16", choices=["bf16", "f16", "e4m3"])
  ap.add_argument("--layout", default="nn", choices=["nn", "nt"])
  ap.add_argument("--shape", type=int, nargs=3, default=[4096, 4096, 4096])
  ap.add_argument("--impl", choices=["wgmma", "mma"])
  ap.add_argument("--timeout", type=float, default=120)
  ap.add_argument("--json-out")
  args = ap.parse_args()
  if args.configs: return sweep(args)
  if not args.no_tune: os.environ.setdefault("FZ_BEAM", "1")
  import torch
  import fusezap.device.cuda_kernels as ck
  from fusezap.device import Device
  dev = Device["CUDA"]
  info = gpu_info()
  fp8 = tuple(dev.cc) >= (8, 9)
  sizes = [4096] if args.quick else [1024, 2048, 4096, 8192]
  cases = [(k, l, s, s, s) for s in sizes for k in ("bf16", "f16") for l in ("nn", "nt")]
  if fp8: cases += [("e4m3", "nt", s, s, s) for s in sizes]
  for M in ([512] if args.quick else [512, 4096]):
    for name, N, K in (QWEN7B[:2] if args.quick else QWEN7B):
      cases += [(k, "nt", M, N, K) for k in ("bf16", "f16") + (("e4m3",) if fp8 else ())]
  print(f"{info}  FZ_BEAM={os.environ.get('FZ_BEAM', '0')}")
  print(f"{'case':34s} {'fusezap TFLOPS':>14s} {'torch TFLOPS':>13s} {'ratio':>6s} {'err':>8s} {'torch err':>9s}")
  rows, bad = [], 0
  for kind, layout, M, N, K in cases:
    r = run_case(dev, kind, layout, M, N, K)
    rows.append(r)
    bad += not r["ok"]
    tag = " GATE" if (kind, layout, M, N, K) == GATE else ""
    print(f"{kind:5s} {layout} {M:5d}x{N:5d}x{K:5d}{'':9s} {r['fz_tflops']:14.1f} {r['torch_tflops']:13.1f} {r['ratio']:6.2f} "
          f"{r['err']:8.1e} {r['torch_err']:9.1e}{'' if r['ok'] else '  WRONG'}{tag}", flush=True)
  # the gate: FuseZap's own matmul (Tensor a @ b through the scheduler hook), not the raw ck.gemm rows above
  print(f"\nTensor matmul on CUDA (cuda_kernels.install(dev); jit = captured replay, eager = schedule + realize per call)")
  trows = []
  for kind, layout, use_jit in [("bf16", "nn", True), ("bf16", "nn", False), ("bf16", "nt", True), ("f16", "nn", True)]:
    _, _, M, N, K = GATE
    r = tensor_case(dev, kind, layout, M, N, K, use_jit=use_jit)
    trows.append(r)
    bad += not r["ok"]
    tag = " GATE" if (kind, layout, use_jit) == ("bf16", "nn", True) else ""
    print(f"{kind:5s} {layout} {M:5d}x{N:5d}x{K:5d} {r['path']:12s} {r['fz_tflops']:10.1f} {r['torch_tflops']:13.1f} {r['ratio']:6.2f} "
          f"{r['err']:8.1e} {r['torch_err']:9.1e}  {'+'.join(r['schedule'])}{'' if r['ok'] else '  WRONG'}{tag}", flush=True)
  gate = trows[0]
  h100 = tuple(dev.cc) == (9, 0)
  passed = gate["ratio"] >= 0.60 and gate["ok"]
  out = ROOT / "results" / "cuda_gemm.json"
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(dict(info=info, rows=rows, tensor_rows=trows, gate=dict(case=GATE, path=gate["path"], ratio=gate["ratio"],
                                                                                    h100=h100, passed=passed)), indent=1))
  print(f"gate: bf16 4096^2 Tensor a @ b ({gate['path']}, {'+'.join(gate['schedule'])}) at {gate['ratio']:.2f}x cuBLAS "
        f"(need >= 0.60 on H100): {('PASS' if passed else 'FAIL') if h100 else 'not gated on ' + info['gpu']}")
  if bad: print(f"{bad} wrong result(s)")
  return 1 if bad or (h100 and not passed) else 0

if __name__ == "__main__": sys.exit(main())
