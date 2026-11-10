"""P3-ATTN benchmark: FuseZap's CUDA flash attention vs torch scaled_dot_product_attention (FlashAttention-2 backend).

Shapes are Qwen2.5-7B's attention (28 query heads, 4 kv heads, head dim 128). Prefill is causal; both sides get the same
bf16 inputs on the same GPU, GQA native on both (torch `enable_gqa=True` under the FLASH_ATTENTION backend; if this torch
cannot do GQA in that backend, kv heads are repeated before timing and the row says so). Timing: CUDA events around a
burst of back-to-back launches on the default stream (~60 ms), best of 5 rounds, both sides alike. Every FuseZap output is
checked against torch's (rtol = atol = 2e-2 in bf16).

  uv run python bench/bench_cuda_attn.py                   # the gate: exits 1 if any gated prefill case < 0.50x torch
  uv run python bench/bench_cuda_attn.py --sweep           # also time other tensor-core tile configs (NWxBN)
  uv run python bench/bench_cuda_attn.py --out results/attn.json

Gate (docs/PLAN.md, P3-ATTN): every bf16 causal prefill row marked GATE must reach >= 0.50x torch SDPA's speed.
Decode rows (paged KV, fusezap/serve/kv_cache.py layout) are information: torch has no paged kernel, so they are compared
with torch SDPA on the same K/V laid out contiguously, and their effective K/V read bandwidth is printed.
"""
from __future__ import annotations
import argparse, json, math, os, sys
import numpy as np

HQ, HKV, D = 28, 4, 128
PREFILL = [(1, 512), (1, 1024), (1, 2048), (1, 4096), (1, 8192), (4, 2048), (16, 512)]   # (batch, tokens)
GATE = 0.50

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", default=None, help="write the results as JSON here")
  ap.add_argument("--sweep", action="store_true", help="also time other tile configs of the tensor-core kernel")
  ap.add_argument("--quick", action="store_true", help="fewer cases (smoke test; no gate)")
  args = ap.parse_args()
  import torch
  import torch.nn.functional as F
  from fusezap.dtype import dtypes
  from fusezap.device import Device, Buffer
  from fusezap.device.cuda_kernels import attention as A
  if not (torch.cuda.is_available() and Device.available("CUDA")):
    print("bench_cuda_attn: needs an NVIDIA GPU (torch CUDA + fusezap CUDA)")
    return 2
  dev = Device["CUDA"]
  torch.manual_seed(0)
  print(f"GPU: {dev.gpu_name} sm_{dev.cc[0]}{dev.cc[1]}, torch {torch.__version__}, NVRTC target {dev.compiler.arch}")
  try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    flash_ctx = lambda: sdpa_kernel([SDPBackend.FLASH_ATTENTION])
  except ImportError:
    flash_ctx = lambda: torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)

  def timeit(fn, target_ms: float = 60.0, rounds: int = 5) -> float:
    """Best per-call ms over `rounds` bursts of back-to-back calls, CUDA events on the default stream."""
    fn(); torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); e.synchronize()
    n = max(3, min(2000, int(target_ms / max(s.elapsed_time(e), 1e-3))))
    best = math.inf
    for _ in range(rounds):
      torch.cuda.synchronize()
      s.record()
      for _ in range(n): fn()
      e.record(); e.synchronize()
      best = min(best, s.elapsed_time(e) / n)
    return best

  def to_buf(t: "torch.Tensor", dt) -> Buffer:
    b = Buffer("CUDA", t.numel(), dt)
    b.copyin(memoryview(t.contiguous().view(torch.int16).cpu().numpy()).cast("B"))
    return b

  def from_buf(b: Buffer, shape) -> "torch.Tensor":
    return torch.from_numpy(b.numpy().view(np.int16).reshape(shape).copy()).view(torch.bfloat16)

  def torch_sdpa(q, k, v, causal: bool):
    """(callable, note) for torch's flash sdpa with native GQA, else with repeated kv heads."""
    try:
      with flash_ctx(): F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
      def run():
        with flash_ctx(): return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)
      return run, "gqa"
    except Exception:
      g = q.shape[1] // k.shape[1]
      ke, ve = k.repeat_interleave(g, 1).contiguous(), v.repeat_interleave(g, 1).contiguous()
      def run():
        with flash_ctx(): return F.scaled_dot_product_attention(q, ke, ve, is_causal=causal)
      return run, "kv repeated"

  rows, ok_all, gate_ok = [], True, True
  cases = PREFILL[:3] if args.quick else PREFILL
  extra = [] if args.quick else [(1, 4096, False)]
  print(f"\nprefill, Qwen2.5-7B heads (Hq={HQ}, Hkv={HKV}, D={D}), bfloat16")
  print(f"{'case':28s} {'fusezap ms':>10s} {'torch ms':>9s} {'fz TFLOPS':>9s} {'torch TFLOPS':>12s} {'ratio':>6s}  {'max err':>8s}  note")
  for B, L, *rest in [(b, l, True) for b, l in cases] + extra:
    causal = rest[0]
    q = torch.randn(B, HQ, L, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, HKV, L, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, HKV, L, D, device="cuda", dtype=torch.bfloat16)
    trun, note = torch_sdpa(q, k, v, causal)
    ref = trun().float()
    st = lambda H: (H * L * D, L * D, D, 1)
    bufs = (Buffer("CUDA", B * HQ * L * D, dtypes.bfloat16), to_buf(q, dtypes.bfloat16), to_buf(k, dtypes.bfloat16), to_buf(v, dtypes.bfloat16))
    def fz_time(cfg=None):
      call = A.dense_call(dev, dtypes.bfloat16, B, HQ, HQ // HKV, L, L, D, D, st(HQ), st(HKV), st(HKV), st(HQ), causal, 1 / math.sqrt(D), cfg=cfg)
      run = call.prepare(dev, bufs)
      run(); dev.synchronize()
      got = from_buf(bufs[0], (B, HQ, L, D)).float().to("cuda")
      err = float(((got - ref).abs() / (ref.abs() + 1)).max())
      return timeit(run), err, call.name
    fz_ms, err, kname = fz_time()
    t_ms = timeit(trun)
    flops = 4 * B * HQ * L * L * D * (0.5 if causal else 1.0)
    ratio = t_ms / fz_ms
    good = err <= 2e-2
    gated = causal and not args.quick
    ok_all &= good
    if gated and ratio < GATE: gate_ok = False
    name = f"B={B} L={L} {'causal' if causal else 'full'}"
    print(f"{name:28s} {fz_ms:10.3f} {t_ms:9.3f} {flops / fz_ms / 1e9:9.1f} {flops / t_ms / 1e9:12.1f} {ratio:6.2f}  {err:8.1e}  "
          f"{'GATE ' + ('ok' if ratio >= GATE else 'FAIL') if gated else ''}{'' if good else ' WRONG'} [{note}] {kname}")
    row = dict(kind="prefill", B=B, L=L, causal=causal, fz_ms=fz_ms, torch_ms=t_ms, fz_tflops=flops / fz_ms / 1e9,
               torch_tflops=flops / t_ms / 1e9, ratio=ratio, max_rel_err=err, gated=gated, torch_note=note, kernel=kname)
    if args.sweep:
      sw = {}
      for cfg in [(nw, bn, sp) for sp in (False, True) for nw, bn in ((4, 32), (4, 64), (4, 128), (8, 32), (8, 64), (8, 128))]:
        if (cfg[1] * D // 8) % (32 * cfg[0]) or A.flash_smem(D, cfg[1]) > A.max_smem(dev): continue
        key = f"{cfg[0]}x{cfg[1]}{'s' if cfg[2] else ''}"
        try:
          ms, e2, _ = fz_time(cfg)
          sw[key] = round(ms, 4) if e2 <= 2e-2 else f"WRONG {e2:.1e}"
        except Exception as ex: sw[key] = f"error: {str(ex).splitlines()[0][:80]}"
      print(f"{'':28s} sweep (NW x BN: ms): {sw}")
      row["sweep"] = sw
    rows.append(row)
    del q, k, v, ref, bufs
    torch.cuda.empty_cache()

  # ---- paged decode (information)
  from fusezap.serve.kv_cache import PagedKVCache, StepLayout
  from fusezap.tensor import Tensor
  print(f"\npaged decode (block size 16, Hq={HQ}, Hkv={HKV}, D={D}), bfloat16: one new token per sequence")
  print(f"{'case':28s} {'fusezap ms':>10s} {'torch ms':>9s} {'fz GB/s':>8s} {'ratio':>6s}  {'max err':>8s}")
  for Bs, ctx in ([(1, 1024), (32, 1024)] if args.quick else [(1, 1024), (1, 8192), (8, 2048), (32, 1024), (32, 4096)]):
    bsz = 16
    nblk = Bs * -(-ctx // bsz)
    cache = PagedKVCache(1, HKV, D, nblk + 1, bsz, dtypes.bfloat16, "CUDA", max_seq_len=ctx)
    for s in range(Bs): cache.ensure(s, ctx)
    kv = torch.randn(2, Bs, HKV, ctx, D, device="cuda", dtype=torch.bfloat16)
    pool = torch.zeros(2, nblk + 1, HKV, bsz, D, dtype=torch.bfloat16)
    kvc = kv.cpu()
    for s in range(Bs):
      for j, blk in enumerate(cache.tables[s]):
        n = min(bsz, ctx - j * bsz)
        pool[:, blk, :, :n] = kvc[:, s, :, j * bsz:j * bsz + n]
    pb = to_buf(pool, dtypes.bfloat16)
    meta = cache.step_meta(Bs, Bs, [(s, ctx - 1) for s in range(Bs)], [])
    L_ = StepLayout(Bs, Bs, cache.max_blocks)
    mb = Tensor(meta, device="CUDA", dtype=dtypes.int32)._realized_contiguous().realized
    q = torch.randn(Bs, HQ, D, device="cuda", dtype=torch.bfloat16)
    xb = to_buf(q, dtypes.bfloat16)
    out = Buffer("CUDA", Bs * HQ * D, dtypes.bfloat16)
    args_ = (dtypes.bfloat16, Bs, HKV, HQ // HKV, D, bsz, nblk + 1, L_.MB, HQ * D, L_.row, L_.pos, L_.bt, 1 / math.sqrt(D))
    if A.split_ok(D, HQ // HKV, HQ * D):
      a, c = A.split_calls(*args_)
      NP = a.params[-1]
      acc, ml = Buffer("CUDA", Bs * HQ * NP * D, dtypes.float32), Buffer("CUDA", Bs * HQ * NP * 2, dtypes.float32)
      r1, r2 = a.prepare(dev, (acc, ml, xb, pb, mb)), c.prepare(dev, (out, acc, ml, mb))
      def run(): r1(); r2()
      kname = a.name
    else:
      call = A.paged_call(*args_)
      run, kname = call.prepare(dev, (out, xb, pb, mb)), call.name
    run(); dev.synchronize()
    qt, kt, vt = q.unsqueeze(2), kv[0], kv[1]
    def trun():
      return F.scaled_dot_product_attention(qt, kt, vt, enable_gqa=True)
    try: ref = trun().float().squeeze(2)
    except Exception:
      kt, vt = kt.repeat_interleave(HQ // HKV, 1), vt.repeat_interleave(HQ // HKV, 1)
      ref = trun().float().squeeze(2)
    got = from_buf(out, (Bs, HQ, D)).float().to("cuda")
    err = float(((got - ref).abs() / (ref.abs() + 1)).max())
    fz_ms, t_ms = timeit(run), timeit(trun)
    gbs = 2 * Bs * HKV * ctx * D * 2 / fz_ms / 1e6
    good = err <= 2e-2
    ok_all &= good
    name = f"batch={Bs} ctx={ctx}"
    print(f"{name:28s} {fz_ms:10.3f} {t_ms:9.3f} {gbs:8.0f} {t_ms / fz_ms:6.2f}  {err:8.1e}{'' if good else '  WRONG'}  {kname}")
    rows.append(dict(kind="decode", B=Bs, ctx=ctx, fz_ms=fz_ms, torch_contiguous_ms=t_ms, fz_kv_gbps=gbs, ratio=t_ms / fz_ms, max_rel_err=err))
    del kv, pool, kvc, q
    torch.cuda.empty_cache()

  gated = [r for r in rows if r.get("gated")]
  if gated: print(f"\ngate: prefill >= {GATE:.2f}x torch SDPA (FlashAttention-2) on every causal row: "
                  f"min {min(r['ratio'] for r in gated):.2f}x -> {'PASS' if gate_ok else 'FAIL'}")
  if not ok_all: print("FAIL: a FuseZap result differs from torch")
  if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
      json.dump(dict(gpu=dev.gpu_name, cc=dev.cc, torch=torch.__version__, gate=GATE, gate_pass=gate_ok and ok_all, rows=rows), f, indent=1)
  return 0 if (ok_all and gate_ok) else 1

if __name__ == "__main__":
  sys.exit(main())
