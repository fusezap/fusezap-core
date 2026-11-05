"""vLLM's flash-attention decode (FA3 on sm_90, else FA2) on bench/decode_attn.py's cases, for the round 14 comparison.
Runs under vLLM's environment (torch + vllm.vllm_flash_attn), not FuseZap's:

  ~/vllm-env/bin/python bench/decode_attn_fa.py [--cases 1x600,8x600,32x600,32x1024] [--max-len 2048] [--block 16]

Qwen2.5-7B shapes (28 q heads, 4 kv heads, hd 128, bf16), paged KV with --block-size pages scattered over a pool sized
for --max-len, one query token per sequence; called like vLLM's FlashAttention backend does (flash_attn_varlen_func
with a block table and seqused_k). Prints us per call and the K/V read bandwidth, best of 5 bursts of 200.
"""
from __future__ import annotations
import argparse, math

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--cases", default="1x600,4x600,8x600,16x600,32x600,32x1024,32x1800")
  ap.add_argument("--max-len", type=int, default=2048)
  ap.add_argument("--block", type=int, default=16)
  a = ap.parse_args()
  import torch
  from vllm.vllm_flash_attn import flash_attn_varlen_func
  HQ, HKV, D, bs = 28, 4, 128, a.block
  dev = torch.device("cuda")
  g = torch.Generator(device="cuda").manual_seed(0)
  print(f"{torch.cuda.get_device_name()}: vLLM flash-attn decode, Hq={HQ} Hkv={HKV} hd={D} bf16, block {bs}, pool for max_len {a.max_len}")
  for case in a.cases.split(","):
    B, ctx = (int(x) for x in case.split("x"))
    mb = -(-a.max_len // bs)
    nblk = B * mb + 1
    k = torch.randn(nblk, bs, HKV, D, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(nblk, bs, HKV, D, device=dev, generator=g).to(torch.bfloat16)
    perm = torch.randperm(nblk - 1, device=dev, generator=g)[:B * mb].reshape(B, mb).to(torch.int32) + 1
    q = torch.randn(B, HQ, D, device=dev, generator=g).to(torch.bfloat16)
    cuq = torch.arange(B + 1, device=dev, dtype=torch.int32)
    seqk = torch.full((B,), ctx, device=dev, dtype=torch.int32)
    out = torch.empty_like(q)
    res = []
    for ver in (3, 2):
      try:
        def run():
          flash_attn_varlen_func(q, k, v, max_seqlen_q=1, cu_seqlens_q=cuq, max_seqlen_k=ctx, seqused_k=seqk, block_table=perm,
                                 causal=True, softmax_scale=1 / math.sqrt(D), out=out, fa_version=ver)
        run(); torch.cuda.synchronize()
        best = 1e9
        for _ in range(5):
          e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
          e0.record()
          for _ in range(200): run()
          e1.record(); torch.cuda.synchronize()
          best = min(best, e0.elapsed_time(e1) * 1e3 / 200)
        kv = 2 * B * HKV * ctx * D * 2
        res.append(f"FA{ver} {best:7.1f} us ({kv / best / 1e3:5.0f} GB/s)")
      except Exception as e: res.append(f"FA{ver} failed: {str(e)[:120]}")
    print(f"  {B:3d} x {ctx:5d}: " + " | ".join(res), flush=True)
  return 0

if __name__ == "__main__": raise SystemExit(main())
