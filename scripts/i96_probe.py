"""I96 probe: which kernels of FLUX.2 [klein] 4B the NEURON backend lowers today, at klein's real 1024^2 shapes.

  PYTHONPATH=. python scripts/i96_probe.py            # part A (shapes only, ~1 min, ~0 GB) + part B (tiny numerics)
  PYTHONPATH=. python scripts/i96_probe.py --json out.json

Part A builds each piece of the klein pipeline (transformer block parts, joint attention, RoPE, modulation, VAE conv /
GroupNorm / upsample / mid attention, the Euler step) as a lazy graph on the NKISHIM tier, with every input an
unallocated device Buffer (no memory, nothing runs), schedules it, and sends every ScheduleItem to what NEURONDevice
would use: kind="custom" items to nki_kernels.render_matmul / render_attention, kind="kernel" items to the generic NKI
renderer (fusezap.codegen.nki). Each item is classified:
  tensorE   a custom matmul / attention kernel (TensorE), rendered
  vector    a generic elementwise / reduce kernel, rendered (VectorE / ScalarE / GpSimd)
  gemm-vec  a generic kernel that is a GEMM the matmul pattern did not take (SUM of a MUL of two loads, one broadcast):
            correct, but VectorE work at matmul size (very slow at klein's sizes)
  FAIL      NKIUnsupported (the reason is printed): nothing falls back to another device
Part B runs a tiny klein-shaped transformer and VAE on NKISHIM (the numpy model of nki 0.6) against CPU float32, and
tries the Qwen3 text encoder on NKISHIM. Nothing here touches AWS or a chip.
"""
from __future__ import annotations
import argparse, json, math, sys, time, traceback
from collections import Counter
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap.device import Buffer, Device
from fusezap.lazy import LazyBuffer
from fusezap.ops import BufferOps, BinaryOps, ReduceOps, UnaryOps
from fusezap.schedule import create_schedule
from fusezap.codegen.nki import NKIUnsupported
from fusezap.codegen import nki_kernels as nk

DEV = "NKISHIM"

def empty(shape, dt=dtypes.float32) -> Tensor:
  """A device tensor over an unallocated Buffer: schedulable, never allocated."""
  return Tensor(LazyBuffer.from_buffer(Buffer(DEV, math.prod(shape), dt), tuple(shape)))

def _walk(op):
  yield op
  for s in op.src: yield from _walk(s)

def _gemm_like(ast) -> bool:
  for root in ast:
    for n in _walk(root):
      if n.op is not ReduceOps.SUM: continue
      m = n.src[0]
      while m.op in (UnaryOps.CAST,): m = m.src[0]
      if m.op is not BinaryOps.MUL: continue
      loads = [x for x in _walk(m) if x.op is BufferOps.LOAD]
      if len({x.arg.idx for x in loads}) < 2: continue
      red = set(n.arg)
      if len(loads) != 2: continue   # a GEMM is SUM(MUL(a, b)): exactly two loads, each broadcast over a different output axis
      zs = [{i for i, (sz, st) in enumerate(zip(x.arg.st.views[-1].shape, x.arg.st.views[-1].strides)) if st == 0 and sz > 1 and i not in red} for x in loads]
      if zs[0] and zs[1] and zs[0] != zs[1]: return True
  return False

def classify(si, dev) -> tuple[str, str]:
  if si.kind == "copy": return "copy", ""
  try:
    if si.kind == "custom":
      call = si.ast[0]
      name = call.kernel.name
      if name == "fz_matmul": nk.render_matmul(call.params)
      elif name in ("fz_attention", "fz_attention_mask"): nk.render_attention(nk.spec_from_params(call.params, name == "fz_attention_mask"))
      else: return "FAIL", f"no NKI kernel for custom {name}"
      return "tensorE", name
    dev.renderer.render(si.ast, si.var_vals)
    return ("gemm-vec" if _gemm_like(si.ast) else "vector"), ""
  except NKIUnsupported as e: return "FAIL", str(e)[:140]
  except Exception as e: return "FAIL", f"{type(e).__name__}: {str(e)[:120]}"

# neuron_kv.matmul_bf16 (flux2's NEURON bf16 linear, I96 W1) launches its kernel at once on realized buffers; here it is
# replaced by a stand-in that renders the same MatmulBF16 spec, keeps its input in the schedule and returns an
# unallocated (..., N) float32 tensor, so part A stays shapes-only.
_MM: list = []
def _matmul_bf16_standin(x: Tensor, w: Tensor) -> Tensor:
  from fusezap.codegen.nki_bf16 import MatmulBF16
  K, N = int(w.shape[0]), int(w.shape[1])
  if x.dtype != dtypes.float32: x = x.float()
  M = math.prod(int(d) for d in x.shape[:-1])
  sp = MatmulBF16(M, N, K)
  try: sp.render(); r = ("tensorE", sp.name)
  except NKIUnsupported as e: r = ("FAIL", f"{sp.name}: {str(e)[:120]}")
  _MM.append((x.contiguous(), r, -(-M // 128) * -(-N // 512) * -(-K // 128)))
  return empty(tuple(x.shape[:-1]) + (N,))

def probe(name: str, fn, where: str) -> dict:
  dev = Device[DEV]
  import fusezap.serve.neuron_kv as NKV
  real, NKV.matmul_bf16, _MM[:] = NKV.matmul_bf16, _matmul_bf16_standin, []
  t0 = time.perf_counter()
  try:
    outs = fn()
    outs = outs if isinstance(outs, (list, tuple)) else [outs]
    sched = create_schedule([o.lazydata for o in outs] + [m[0].lazydata for m in _MM])
  except Exception as e:
    return {"piece": name, "where": where, "error": f"{type(e).__name__}: {str(e)[:200]}"}
  finally: NKV.matmul_bf16 = real
  cls = [classify(si, dev) for si in sched] + [m[1] for m in _MM]
  cnt = Counter(c for c, _ in cls)
  from fusezap.codegen.lower import _ast_key
  keys, unroll = set(), 0
  for si in sched:
    if si.kind == "copy": continue
    if si.kind == "custom":
      call = si.ast[0]
      keys.add((call.kernel.name, tuple(call.params)))
      pr = [int(x) for x in call.params]
      if call.kernel.name == "fz_matmul":   # nc_matmul instructions nki 0.6 unrolls: Bt * M/128 * N/512 * K/128
        unroll = max(unroll, pr[0] * -(-pr[1] // 128) * -(-pr[2] // 512) * -(-pr[3] // 128))
      else:                                  # (query block, key block) pairs: Bt * L/128 * S/512
        unroll = max(unroll, pr[0] * -(-pr[1] // 128) * -(-pr[2] // 512))
    else: keys.add(_ast_key(si.ast))
  for _, (c, nm), tiles in _MM: keys.add(nm); unroll = max(unroll, tiles)
  fails = sorted({r for c, r in cls if c == "FAIL"})
  customs = sorted({r for c, r in cls if c == "tensorE"})
  return {"piece": name, "where": where, "items": len(sched) + len(_MM), **{k: cnt.get(k, 0) for k in ("tensorE", "vector", "gemm-vec", "FAIL", "copy")},
          "custom": customs, "fail_reasons": fails, "distinct": len(keys), "max_unrolled_tiles": unroll, "s": round(time.perf_counter() - t0, 2)}

# ---------------------------------------------------------------- part A: real klein shapes, 1024^2
def part_a(dt) -> list[dict]:
  from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer
  c = Flux2Config(num_layers=5, num_single_layers=20, num_heads=24, head_dim=128, joint_attention_dim=7680, guidance_embeds=False)
  D, M, hd, H, T, L, B = c.dim, c.mlp_hidden, c.head_dim, c.num_heads, 512, 64 * 64, 1
  S = T + L
  tr = Flux2Transformer.__new__(Flux2Transformer)   # no weights loaded: every weight an unallocated device buffer
  tr.cfg, tr.dtype, tr.device, tr.realize_blocks = c, dt, DEV, False
  tr.kmajor = dt == dtypes.bfloat16   # as Flux2Transformer.__init__ on a NEURON tier (FZ_KLEIN_NEURON_BF16MM=1)
  shapes = {"x_embedder.weight": (D, 128), "context_embedder.weight": (D, 7680), "proj_out.weight": (128, D), "norm_out.linear.weight": (2 * D, D),
            "time_guidance_embed.timestep_embedder.linear_1.weight": (D, 256), "time_guidance_embed.timestep_embedder.linear_2.weight": (D, D),
            "double_stream_modulation_img.linear.weight": (6 * D, D), "double_stream_modulation_txt.linear.weight": (6 * D, D),
            "single_stream_modulation.linear.weight": (3 * D, D)}
  p = "transformer_blocks.0."
  for n in "qkv": shapes[p + f"attn.to_{n}.weight"] = shapes[p + f"attn.add_{n}_proj.weight"] = (D, D)
  shapes[p + "attn.to_out.0.weight"] = shapes[p + "attn.to_add_out.weight"] = (D, D)
  for n in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"): shapes[p + f"attn.{n}.weight"] = (hd,)
  for f in ("ff", "ff_context"): shapes[p + f"{f}.linear_in.weight"], shapes[p + f"{f}.linear_out.weight"] = (2 * M, D), (D, M)
  q = "single_transformer_blocks.0.attn."
  shapes[q + "to_qkv_mlp_proj.weight"], shapes[q + "to_out.weight"] = (3 * D + 2 * M, D), (D, D + M)
  shapes[q + "norm_q.weight"] = shapes[q + "norm_k.weight"] = (hd,)
  for k in [k for k in shapes if k.startswith(("transformer_blocks.0.", "single_transformer_blocks.0."))]:   # every block: same shapes
    for i in range(1, c.num_layers if k.startswith("transformer") else c.num_single_layers):
      shapes[k.replace(".0.", f".{i}.", 1)] = shapes[k]
  tr.w = {k: empty(s[::-1] if tr.kmajor and len(s) == 2 else s, dt) for k, s in shapes.items()}
  tr._tfreq = empty((128,))
  F = "fusezap/serve/models/flux2.py"
  x, e, ts = empty((B, L, 128)), empty((B, T, 7680)), empty((B,))
  cos, sin = empty((S, hd)), empty((S, hd))
  h, hs = empty((B, L, D), dt), empty((B, S, D), dt)
  qh = empty((B, S, H, hd), dt)
  mod = [empty((B, 1, D), dt) for _ in range(6)]
  R = []
  with Tensor.inference():
    R.append(probe("temb + 3 shared modulations", lambda: [tr._lin(tr.temb(ts).silu(), "double_stream_modulation_img.linear.weight")], f"{F}:227-251"))
    R.append(probe("x/context embedders (linear)", lambda: [tr._lin(x.cast(dt), "x_embedder.weight"), tr._lin(e.cast(dt), "context_embedder.weight")], f"{F}:253-254"))
    R.append(probe("layernorm + modulate (adaLN)", lambda: (1 + mod[1]) * tr._ln(h) + mod[0], f"{F}:194,160"))
    R.append(probe("q/k/v linear 3072x3072", lambda: tr._lin(h, p + "attn.to_q.weight"), f"{F}:196,159"))
    R.append(probe("per-head q/k RMSNorm", lambda: tr._rms(qh, p + "attn.norm_q.weight"), f"{F}:161-164"))
    R.append(probe("4-axis interleaved RoPE", lambda: tr._rope(qh, cos, sin), f"{F}:168-176"))
    R.append(probe("joint attention 24h x 4608 x 128", lambda: tr._attn(qh, qh, qh), f"{F}:180-184,85-94"))
    R.append(probe("SwiGLU FF 3072->18432->3072", lambda: tr._lin(tr._swiglu(tr._lin(h, p + "ff.linear_in.weight")), p + "ff.linear_out.weight"), f"{F}:165-167,208"))
    R.append(probe("double block (whole)", lambda: list(tr._double(0, h, empty((B, T, D), dt), mod, mod, cos, sin)), f"{F}:190-212"))
    R.append(probe("single block (whole)", lambda: tr._single(0, hs, mod[:3], cos, sin), f"{F}:214-225"))
    R.append(probe("output norm + proj_out", lambda: tr._lin(tr._ln(h) * (1 + mod[0]) + mod[1], "proj_out.weight"), f"{F}:262-264"))
    R.append(probe("FULL STEP: 5 double + 20 single + head", lambda: tr._forward(x, e, ts, cos, sin), f"{F}:244-264"))
    R.append(probe("Euler step x += dt*v", lambda: empty((B, L, 128)) + empty((B, L, 128)) * 0.25, "fusezap/serve/models/flux2_klein.py:119"))
  return R

def part_a_vae(dt) -> list[dict]:
  from fusezap.serve.models.flux2 import sdpa
  V = "fusezap/serve/models/flux2_vae.py"
  R = []
  g, b = empty((128,), dt), empty((128,), dt)
  g5, b5 = empty((512,), dt), empty((512,), dt)
  with Tensor.inference():
    R.append(probe("conv3x3 512->512 @128^2 (mid)", lambda: empty((1, 512, 128, 128), dt).conv2d(empty((512, 512, 3, 3), dt), empty((512,), dt), padding=1), f"{V}:97 -> tensor.py:746"))
    R.append(probe("conv3x3 128->128 @1024^2 (last level)", lambda: empty((1, 128, 1024, 1024), dt).conv2d(empty((128, 128, 3, 3), dt), empty((128,), dt), padding=1), f"{V}:97 -> tensor.py:746"))
    R.append(probe("conv1x1 shortcut 256->128 @1024^2", lambda: empty((1, 256, 1024, 1024), dt).conv2d(empty((128, 256, 1, 1), dt), empty((128,), dt)), f"{V}:104"))
    R.append(probe("GroupNorm(32)+SiLU 128ch @1024^2", lambda: empty((1, 128, 1024, 1024), dt).group_norm(32, g, b, 1e-6).silu(), f"{V}:98 -> tensor.py:787"))
    R.append(probe("GroupNorm(32) 512ch @128^2", lambda: empty((1, 512, 128, 128), dt).group_norm(32, g5, b5, 1e-6), f"{V}:98"))
    R.append(probe("nearest 2x upsample 256ch @512^2", lambda: empty((1, 256, 512, 512), dt).interpolate(scale_factor=2), f"{V}:131 -> tensor.py:810"))
    R.append(probe("mid attention 1 head x 16384 x 512", lambda: sdpa(*(empty((1, 1, 128 * 128, 512), dt) for _ in range(3))), f"{V}:106-113"))
    R.append(probe("BN denormalize + unpatchify", lambda: empty((1, 128, 64, 64)) * empty((1, 128, 1, 1)) + empty((1, 128, 1, 1)), f"{V}:136-145"))
  return R

# ---------------------------------------------------------------- part B: tiny numerics on NKISHIM vs CPU
def part_b() -> list[dict]:
  from fusezap.serve.models.flux2 import Flux2Config, Flux2Transformer
  from fusezap.serve.models.flux2_vae import Flux2VAEConfig, Flux2VAEDecoder
  out = []
  c = Flux2Config(num_layers=1, num_single_layers=1, num_heads=2, head_dim=128, joint_attention_dim=96, mlp_ratio=3.0, guidance_embeds=False)
  rng = np.random.default_rng(0)
  tmp = Flux2Transformer.__new__(Flux2Transformer); tmp.cfg = c
  D, M, hd = c.dim, c.mlp_hidden, c.head_dim
  def shape(n):
    if n.endswith(("norm_q.weight", "norm_k.weight", "norm_added_q.weight", "norm_added_k.weight")): return (hd,)
    if "to_qkv_mlp_proj" in n: return (3 * D + 2 * M, D)
    if n.endswith("attn.to_out.weight"): return (D, D + M)
    if "linear_in" in n: return (2 * M, D)
    if "linear_out" in n: return (D, M)
    if n.startswith("double_stream_modulation"): return (6 * D, D)
    if n.startswith("single_stream_modulation"): return (3 * D, D)
    return {"x_embedder.weight": (D, 128), "context_embedder.weight": (D, 96), "proj_out.weight": (128, D), "norm_out.linear.weight": (2 * D, D),
            "time_guidance_embed.timestep_embedder.linear_1.weight": (D, 256)}.get(n, (D, D))
  W = {n: (rng.standard_normal(shape(n)) / math.sqrt(shape(n)[-1])).astype(np.float32) if len(shape(n)) > 1 else
          (1 + 0.1 * rng.standard_normal(shape(n))).astype(np.float32) for n in tmp.weight_names()}
  T, gh, gw = 8, 4, 4
  x = rng.standard_normal((1, gh * gw, 128)).astype(np.float32); e = rng.standard_normal((1, T, 96)).astype(np.float32)
  for dt in (dtypes.float32, dtypes.bfloat16):
    ref = Flux2Transformer(c, W, dt, "CPU")(x, e, 0.7, T, (gh, gw))
    t0 = time.perf_counter()
    try:
      got = Flux2Transformer(c, W, dt, DEV)(x, e, 0.7, T, (gh, gw))
      err = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30))
      out.append({"test": f"tiny transformer {dt.name} NKISHIM vs CPU", "max_rel_err": err, "s": round(time.perf_counter() - t0, 1)})
    except Exception as ex:
      out.append({"test": f"tiny transformer {dt.name} NKISHIM vs CPU", "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
  vc = Flux2VAEConfig(latent_channels=4, out_channels=3, block_out_channels=(32, 64), layers_per_block=1, norm_num_groups=8, patch_size=(2, 2))
  vae = Flux2VAEDecoder.__new__(Flux2VAEDecoder); vae.cfg = vc
  vw = {}
  top = vc.block_out_channels[-1]
  def conv(p, cin, cout, k):
    vw[p + ".weight"] = (rng.standard_normal((cout, cin, k, k)) / math.sqrt(cin * k * k)).astype(np.float32); vw[p + ".bias"] = (0.1 * rng.standard_normal(cout)).astype(np.float32)
  def gn(p, ch): vw[p + ".weight"] = (1 + 0.1 * rng.standard_normal(ch)).astype(np.float32); vw[p + ".bias"] = (0.1 * rng.standard_normal(ch)).astype(np.float32)
  def res(p, cin, cout):
    gn(p + "norm1", cin); conv(p + "conv1", cin, cout, 3); gn(p + "norm2", cout); conv(p + "conv2", cout, cout, 3)
    if cin != cout: conv(p + "conv_shortcut", cin, cout, 1)
  vw["bn.running_mean"] = (0.1 * rng.standard_normal(16)).astype(np.float32); vw["bn.running_var"] = (1 + 0.1 * rng.random(16)).astype(np.float32)
  conv("post_quant_conv", 4, 4, 1); conv("decoder.conv_in", 4, top, 3)
  res("decoder.mid_block.resnets.0.", top, top); res("decoder.mid_block.resnets.1.", top, top)
  a = "decoder.mid_block.attentions.0."
  gn(a + "group_norm", top)
  for n in ("to_q", "to_k", "to_v", "to_out.0"):
    vw[a + n + ".weight"] = (rng.standard_normal((top, top)) / math.sqrt(top)).astype(np.float32); vw[a + n + ".bias"] = (0.1 * rng.standard_normal(top)).astype(np.float32)
  for i, (cin, cout) in enumerate(Flux2VAEDecoder._up_channels(vae)):
    for j in range(vc.layers_per_block + 1): res(f"decoder.up_blocks.{i}.resnets.{j}.", cin if j == 0 else cout, cout)
    if i < len(vc.block_out_channels) - 1: conv(f"decoder.up_blocks.{i}.upsamplers.0.conv", cout, cout, 3)
  gn("decoder.conv_norm_out", vc.block_out_channels[0]); conv("decoder.conv_out", vc.block_out_channels[0], 3, 3)
  z = rng.standard_normal((1, gh * gw, 16)).astype(np.float32)   # packed (B, h*w, C*2*2) for h=w=4, C=4
  ref = Flux2VAEDecoder(vc, vw, dtypes.float32, "CPU").decode_packed(Tensor(z, device="CPU"), gh, gw).numpy()
  t0 = time.perf_counter()
  try:
    got = Flux2VAEDecoder(vc, vw, dtypes.float32, DEV).decode_packed(Tensor(z, device=DEV), gh, gw).numpy()
    out.append({"test": "tiny VAE decode float32 NKISHIM vs CPU", "max_rel_err": float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-30)), "s": round(time.perf_counter() - t0, 1)})
  except Exception as ex:
    out.append({"test": "tiny VAE decode float32 NKISHIM vs CPU", "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
  # the Qwen3 text encoder (q/k RMSNorm) on a NEURON tier
  try:
    from fusezap.serve.models.llama import Llama, LlamaConfig
    cfg = LlamaConfig.from_hf({"hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                               "head_dim": 128, "vocab_size": 300, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "model_type": "qwen3", "max_position_embeddings": 512,
                               "eos_token_id": 1, "tie_word_embeddings": True})
    out.append({"test": "Qwen3 encoder config qk_norm", "qk_norm": bool(cfg.qk_norm)})
    wts = {"model.embed_tokens.weight": rng.standard_normal((300, 256)).astype(np.float32), "model.norm.weight": np.ones(256, np.float32)}
    for i in range(2):
      pp = f"model.layers.{i}."
      wts.update({pp + "self_attn.q_proj.weight": rng.standard_normal((256, 256)).astype(np.float32) * .05, pp + "self_attn.k_proj.weight": rng.standard_normal((128, 256)).astype(np.float32) * .05,
                  pp + "self_attn.v_proj.weight": rng.standard_normal((128, 256)).astype(np.float32) * .05, pp + "self_attn.o_proj.weight": rng.standard_normal((256, 256)).astype(np.float32) * .05,
                  pp + "self_attn.q_norm.weight": np.ones(128, np.float32), pp + "self_attn.k_norm.weight": np.ones(128, np.float32),
                  pp + "mlp.gate_proj.weight": rng.standard_normal((512, 256)).astype(np.float32) * .05, pp + "mlp.up_proj.weight": rng.standard_normal((512, 256)).astype(np.float32) * .05,
                  pp + "mlp.down_proj.weight": rng.standard_normal((256, 512)).astype(np.float32) * .05,
                  pp + "input_layernorm.weight": np.ones(256, np.float32), pp + "post_attention_layernorm.weight": np.ones(256, np.float32)})
    Llama(cfg, wts, dtypes.float32, DEV, max_len=64, lm_head=False, load_layers=2)
    out.append({"test": "Qwen3 encoder on NKISHIM", "result": "constructed"})
  except Exception as ex:
    out.append({"test": "Qwen3 encoder on NKISHIM", "error": f"{type(ex).__name__}: {str(ex)[:200]}"})
  return out

def show(rows: list[dict], title: str) -> None:
  print(f"\n== {title}")
  print(f"{'piece':40s} {'items':>5s} {'tensorE':>7s} {'vector':>6s} {'gemm-vec':>8s} {'FAIL':>4s} {'distinct':>8s} {'max tiles':>9s}  custom kernels / failures")
  for r in rows:
    if "error" in r: print(f"{r['piece']:40s} ERROR {r['error']}"); continue
    note = ", ".join(r["custom"]) + ("; FAIL: " + " | ".join(r["fail_reasons"]) if r["fail_reasons"] else "")
    print(f"{r['piece']:40s} {r['items']:5d} {r['tensorE']:7d} {r['vector']:6d} {r['gemm-vec']:8d} {r['FAIL']:4d} {r['distinct']:8d} {r['max_unrolled_tiles']:9d}  {note}")

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", default=None)
  ap.add_argument("--skip-b", action="store_true")
  a = ap.parse_args()
  res = {}
  for dt in (dtypes.float32, dtypes.bfloat16):
    res[f"transformer_{dt.name}"] = part_a(dt); show(res[f"transformer_{dt.name}"], f"A: klein transformer pieces at 1024^2 (S=4608), {dt.name}, NEURON lowering")
    res[f"vae_{dt.name}"] = part_a_vae(dt); show(res[f"vae_{dt.name}"], f"A: VAE decoder pieces at 1024^2, {dt.name}, NEURON lowering")
  if not a.skip_b:
    res["numerics"] = part_b()
    print("\n== B: tiny numerics on NKISHIM (numpy model of nki 0.6) vs CPU float32")
    for r in res["numerics"]: print("  ", r)
  if a.json:
    with open(a.json, "w") as f: json.dump(res, f, indent=1, default=str)

if __name__ == "__main__": main()
