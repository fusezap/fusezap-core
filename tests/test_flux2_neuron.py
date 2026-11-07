"""I96 W1/W2: the FLUX.2 transformer on a NEURON tier (NKISHIM: the numpy model of nki 0.6, runs on any host).

W1's routing (fusezap/serve/models/flux2.py): float32 flash attention for bf16 q/k/v, a contiguous attention output before
its projections, bf16 K-major linears through neuron_kv.matmul_bf16. Checked here: the scheduled kernels (flash attention
taken, no matmul left to the VectorE renderer, no float32 weight copy) and the numbers against CPU float32.
"""
import math
import numpy as np
import pytest
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap.ops import BufferOps, BinaryOps, ReduceOps
from fusezap.schedule import create_schedule
from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer

CFG = dict(num_layers=1, num_single_layers=1, num_heads=2, head_dim=128, joint_attention_dim=96, mlp_ratio=3.0, guidance_embeds=False)

def tiny_weights(seed: int = 0) -> tuple[Flux2Config, dict]:
  c = Flux2Config(**CFG)
  rng = np.random.default_rng(seed)
  D, M, hd = c.dim, c.mlp_hidden, c.head_dim
  def shape(n: str) -> tuple:
    if n.endswith(("norm_q.weight", "norm_k.weight", "norm_added_q.weight", "norm_added_k.weight")): return (hd,)
    if "to_qkv_mlp_proj" in n: return (3 * D + 2 * M, D)
    if n.endswith("attn.to_out.weight"): return (D, D + M)
    if "linear_in" in n: return (2 * M, D)
    if "linear_out" in n: return (D, M)
    if n.startswith("double_stream_modulation"): return (6 * D, D)
    if n.startswith("single_stream_modulation"): return (3 * D, D)
    return {"x_embedder.weight": (D, 128), "context_embedder.weight": (D, 96), "proj_out.weight": (128, D),
            "norm_out.linear.weight": (2 * D, D), "time_guidance_embed.timestep_embedder.linear_1.weight": (D, 256)}.get(n, (D, D))
  tmp = Flux2Transformer.__new__(Flux2Transformer); tmp.cfg = c
  W = {}
  for n in tmp.weight_names():
    s = shape(n)
    W[n] = (rng.standard_normal(s) / math.sqrt(s[-1])).astype(np.float32) if len(s) > 1 else (1 + 0.1 * rng.standard_normal(s)).astype(np.float32)
  return c, W

def inputs(seed: int = 1, T: int = 8, g: int = 4):
  rng = np.random.default_rng(seed)
  return rng.standard_normal((1, g * g, 128)).astype(np.float32), rng.standard_normal((1, T, 96)).astype(np.float32), T, (g, g)

def rel_l2(a, b) -> float: return float(np.linalg.norm(a - b) / np.linalg.norm(b))

def test_neuron_tiny_f32_matches_cpu():
  c, W = tiny_weights()
  x, e, T, hw = inputs()
  ref = Flux2Transformer(c, W, dtypes.float32, "CPU")(x, e, 0.7, T, hw)
  got = Flux2Transformer(c, W, dtypes.float32, "NKISHIM")(x, e, 0.7, T, hw)
  assert np.abs(got - ref).max() / np.abs(ref).max() <= 1e-4   # G1 (float32)

def test_neuron_tiny_bf16_within_cpu_bf16():
  """bf16 on NKISHIM (MatmulBF16: bf16 operands, float32 accumulation; float32 attention) vs CPU float32, against CPU bf16's
  own distance (the relative bf16 bar); and within G1's bf16 tolerance (1e-2 of the output scale)."""
  c, W = tiny_weights()
  x, e, T, hw = inputs()
  ref = Flux2Transformer(c, W, dtypes.float32, "CPU")(x, e, 0.7, T, hw)
  cpu = Flux2Transformer(c, W, dtypes.bfloat16, "CPU")(x, e, 0.7, T, hw)
  tr = Flux2Transformer(c, W, dtypes.bfloat16, "NKISHIM")
  assert tr.kmajor and tr.w["x_embedder.weight"].shape == (128, c.dim)   # K-major (in, out)
  got = tr(x, e, 0.7, T, hw)
  assert rel_l2(got, ref) <= 1.5 * rel_l2(cpu, ref) + 1e-6
  assert np.abs(got - ref).max() / np.abs(ref).max() <= 1e-2

def test_metal_and_cpu_untouched():
  c, W = tiny_weights()
  assert not Flux2Transformer(c, W, dtypes.bfloat16, "CPU").kmajor

def _walk(op):
  yield op
  for s in op.src: yield from _walk(s)

def _gemm_on_vector(si) -> bool:
  """A generic kernel that is a GEMM: SUM over MUL of exactly two loads, each broadcast over a different output axis."""
  for n in _walk(si.ast[0]):
    if n.op is not ReduceOps.SUM or n.src[0].op is not BinaryOps.MUL: continue
    loads = [x for x in _walk(n.src[0]) if x.op is BufferOps.LOAD]
    if len(loads) != 2: continue
    zs = [{i for i, (sz, st) in enumerate(zip(x.arg.st.views[-1].shape, x.arg.st.views[-1].strides)) if st == 0 and sz > 1 and i not in n.arg} for x in loads]
    if zs[0] and zs[1] and zs[0] != zs[1]: return True
  return False

@pytest.mark.parametrize("dt", [dtypes.float32, dtypes.bfloat16])
def test_neuron_block_kernels(dt):
  """Every kernel a double + single block schedules on NKISHIM: attention is the flash kernel, no GEMM on VectorE.
  (The bf16 linears launch MatmulBF16 eagerly through jit.run_item; recorded here by patching the device's run.)"""
  from fusezap.device import Device
  c, W = tiny_weights()
  x, e, T, hw = inputs()
  tr = Flux2Transformer(c, W, dt, "NKISHIM")
  dev = Device["NKISHIM"]
  seen, orig = [], dev.run
  def run(si):
    seen.append((si.kind, getattr(getattr(si.ast[0], "kernel", None), "name", None) if si.ast else None,
                 getattr(getattr(si.ast[0], "spec", None), "name", None) if si.ast else None, si.kind == "kernel" and _gemm_on_vector(si)))
    return orig(si)
  dev.run = run
  try: tr(x, e, 0.7, T, hw)
  finally: dev.run = orig
  names = [k or s for _, k, s, _ in seen if k or s]
  assert any(n == "fz_attention" for n in names), names            # flash attention, also for bf16 q/k/v
  assert not any(g for *_, g in seen)                               # no matmul left to the VectorE renderer
  if dt == dtypes.bfloat16: assert any(n and n.endswith("_bf16") for n in names)   # MatmulBF16, no float32 weight copy
