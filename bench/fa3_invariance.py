"""I50 option (2), the cheap check first: can FlashAttention-3 (vLLM's vllm_flash_attn, sm_90) serve FuseZap's paged
decode attention without breaking "batched == alone", and how fast is it with a cold L2? Runs under vLLM's environment:

  ~/vllm-env/bin/python bench/fa3_invariance.py [--splits 0,1,2,4,8] [--batch 32] [--json out.json]

Qwen2.5-7B decode shapes (28 q heads, 4 kv heads, head dim 128, bf16), paged KV in 16-token blocks scattered over a pool
sized for --max-len, one query token per sequence, called as vLLM's FlashAttention backend calls it
(flash_attn_varlen_func with a block table and seqused_k, fa_version=3). Per num_splits value (0 = FA3's own heuristic,
which looks at the batch; n > 0 = fixed):
  invariance  each of --batch sequences (contexts 37 .. 1800, ragged) computed alone (batch 1) and inside the batch, and
              inside the batch in reversed order: the output rows must be bitwise equal (FuseZap's guarantee: a token's
              result does not depend on what else is in the step)
  cold L2     32 x 600 (the c32 gate case) with a 256 MB buffer written before every launch, CUDA events around the
              launch only: us and K/V GB/s (I50's gate: >= 1.5 TB/s, i.e. <= 26.2 us); warm back-to-back for reference
Prints one line per num_splits and writes --json. Licence note: FlashAttention-3 is BSD-3-Clause.
"""
from __future__ import annotations
import argparse, json, math

HQ, HKV, D, BS = 28, 4, 128, 16

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--splits", default="0,1,2,4,8")
  ap.add_argument("--batch", type=int, default=32)
  ap.add_argument("--max-len", type=int, default=2048)
  ap.add_argument("--reps", type=int, default=50)
  ap.add_argument("--json", default=None)
  a = ap.parse_args()
  import torch
  from vllm.vllm_flash_attn import flash_attn_varlen_func
  dev = torch.device("cuda")
  g = torch.Generator(device="cuda").manual_seed(0)
  mb = -(-a.max_len // BS)
  B = a.batch
  nblk = B * mb + 1
  kc = torch.randn(nblk, BS, HKV, D, device=dev, generator=g).to(torch.bfloat16)
  vc = torch.randn(nblk, BS, HKV, D, device=dev, generator=g).to(torch.bfloat16)
  tables = (torch.randperm(nblk - 1, device=dev, generator=g)[:B * mb].reshape(B, mb) + 1).to(torch.int32)
  q_all = torch.randn(B, HQ, D, device=dev, generator=g).to(torch.bfloat16)
  ctx = torch.tensor([37 + (i * 1763) // max(1, B - 1) for i in range(B)], device=dev, dtype=torch.int32)   # 37 .. 1800, ragged
  flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)
  name = torch.cuda.get_device_name()
  print(f"{name}: FA3 paged decode, Hq={HQ} Hkv={HKV} hd={D} bf16, block {BS}, pool for max_len {a.max_len}, batch {B}", flush=True)

  def call(q, tbl, seqk, maxk, ns, out):
    kw = dict(max_seqlen_q=1, cu_seqlens_q=torch.arange(q.shape[0] + 1, device=dev, dtype=torch.int32), max_seqlen_k=maxk,
              seqused_k=seqk, block_table=tbl, causal=True, softmax_scale=1 / math.sqrt(D), out=out, fa_version=3)
    if ns: kw["num_splits"] = ns   # 0: not passed (FA3 picks the split count itself)
    flash_attn_varlen_func(q, kc, vc, **kw)
    return out

  res = []
  for ns in [int(x) for x in a.splits.split(",") if x != ""]:
    r: dict = {"num_splits": ns}
    try:
      out_b = call(q_all, tables, ctx, int(ctx.max()), ns, torch.empty_like(q_all)).clone()
      rev = torch.arange(B - 1, -1, -1, device=dev)
      out_r = call(q_all[rev].contiguous(), tables[rev].contiguous(), ctx[rev].contiguous(), int(ctx.max()), ns, torch.empty_like(q_all))[rev].clone()
      alone_bad, rev_bad = 0, int((out_r.view(torch.int16) != out_b.view(torch.int16)).any(dim=(1, 2)).sum())
      for i in range(B):
        o1 = call(q_all[i:i + 1].contiguous(), tables[i:i + 1].contiguous(), ctx[i:i + 1].contiguous(), int(ctx[i]), ns, torch.empty_like(q_all[:1]))
        alone_bad += int(not torch.equal(o1.view(torch.int16), out_b[i:i + 1].view(torch.int16)))
      r.update(alone_vs_batched_differ=alone_bad, reversed_differ=rev_bad, invariant=alone_bad == 0 and rev_bad == 0)
      # cold L2: the gate case 32 x 600
      c600 = torch.full((min(B, 32),), 600, device=dev, dtype=torch.int32)
      q32, t32, o32 = q_all[:32].contiguous(), tables[:32].contiguous(), torch.empty_like(q_all[:32])
      call(q32, t32, c600, 600, ns, o32); torch.cuda.synchronize()
      ts = []
      for _ in range(a.reps):
        flush.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record(); call(q32, t32, c600, 600, ns, o32); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3)
      ts.sort()
      e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
      e0.record()
      for _ in range(200): call(q32, t32, c600, 600, ns, o32)
      e1.record(); torch.cuda.synchronize()
      kv = 2 * q32.shape[0] * HKV * 600 * D * 2
      cold, warm = ts[len(ts) // 2], e0.elapsed_time(e1) * 1e3 / 200
      r.update(cold_us=round(cold, 2), cold_GBps=round(kv / cold / 1e3, 0), warm_us=round(warm, 2), warm_GBps=round(kv / warm / 1e3, 0))
    except Exception as e: r["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    res.append(r)
    print(f"FA3_INV {json.dumps(r)}", flush=True)
  if a.json:
    with open(a.json, "w") as f: json.dump({"gpu": name, "batch": B, "contexts": ctx.tolist(), "results": res}, f, indent=1)
  return 0

if __name__ == "__main__": raise SystemExit(main())
