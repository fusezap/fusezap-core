"""Random-weight Llama-style models at a real checkpoint's exact shapes (benchmarks: no download, no 70 GB load).

  m = build(PRESETS["llama70b"], "CUDA", dtypes.bfloat16, max_len=8192, tp=g)   # this rank's TP slices (dist.tp_slice)
  m = build(PRESETS["llama70b-tp2"], "CUDA", dtypes.bfloat16, max_len=8192)    # one TP=2 rank's local shapes, one GPU

Values: one tiled random block (std 0.02, norms 1); they do not change the time of the decode kernels. Every layer gets
its own device buffers (unless share: layer 0's reused, for GPUs smaller than the weights); the host arrays are one per
shape, so the host holds a few GB at most (70B: ~3.6 GB with the 2.1 GB embedding table; a layer's slice copies and
q/k/v / gate/up concatenations are freed once uploaded: before I57 M6 every uploaded tensor's LazyBuffer kept its host
copy, ~60 GB per 70B TP=2 rank). A TP rank slices the full-shape arrays like a checkpoint's (views), so the same seed
gives every rank the slices of one model: the replicated lm_head's logits agree across ranks (lookahead).
bench/decode_kernels.py (one GPU) and bench/tp_step.py --random (every TP rank, dist.Worker.op_load) use it.
"""
from __future__ import annotations
from typing import Any
import numpy as np

PRESETS = {   # full shapes ("-tp2": the local per-rank shapes of a TP=2 split); head_dim 128 everywhere
  "llama70b": dict(hidden_size=8192, intermediate_size=28672, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=128256, bias=False, rope_theta=5e5),
  "qwen72b": dict(hidden_size=8192, intermediate_size=29568, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=152064, bias=True, rope_theta=1e6),
  "llama70b-tp2": dict(hidden_size=8192, intermediate_size=14336, n_layers=80, n_heads=32, n_kv_heads=4, vocab_size=128256, bias=False, rope_theta=5e5),
  "qwen72b-tp2": dict(hidden_size=8192, intermediate_size=14784, n_layers=80, n_heads=32, n_kv_heads=4, vocab_size=152064, bias=True, rope_theta=1e6),
  "llama70b-tp1": dict(hidden_size=8192, intermediate_size=28672, n_layers=80, n_heads=64, n_kv_heads=8, vocab_size=128256, bias=False, rope_theta=5e5),
  "qwen7b": dict(hidden_size=3584, intermediate_size=18944, n_layers=28, n_heads=28, n_kv_heads=4, vocab_size=152064, bias=True, rope_theta=1e6),
  "tiny": dict(hidden_size=256, intermediate_size=512, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=1000, bias=False, rope_theta=1e4),
}

def weights(sh: dict, dtype: Any, n_layers: int, seed: int = 0) -> dict[str, np.ndarray]:
  """HF tensor names -> host arrays (dtype's host format; bf16 as BF16-tagged uint16) for n_layers layers at shapes sh."""
  from fusezap.serve.weights import STArray, f32_to_bf16
  from fusezap.dtype import dtypes
  rng = np.random.default_rng(seed)
  blk = rng.standard_normal(1 << 22).astype(np.float32) * 0.02
  bf = dtype == dtypes.bfloat16
  bits = f32_to_bf16(blk) if bf else blk.astype(np.float16).view(np.uint16) if dtype == dtypes.float16 else blk
  one = np.uint16(0x3F80) if bf else np.float16(1.0).view(np.uint16) if dtype == dtypes.float16 else np.float32(1.0)
  cache: dict = {}
  def arr(shape: tuple, ones: bool = False) -> np.ndarray:
    if (shape, ones) not in cache:
      a = (np.full(shape, one, bits.dtype) if ones else np.resize(bits, shape)).view(STArray)
      if bf: a.st_dtype = "BF16"
      elif dtype == dtypes.float16: a = a.view(np.float16)
      cache[(shape, ones)] = a
    return cache[(shape, ones)]
  D, I, Hq, Hkv, hd, V = sh["hidden_size"], sh["intermediate_size"], sh["n_heads"], sh["n_kv_heads"], 128, sh["vocab_size"]
  w: dict = {"model.embed_tokens.weight": arr((V, D)), "lm_head.weight": arr((V, D)), "model.norm.weight": arr((D,), True)}
  for i in range(n_layers):
    p = f"model.layers.{i}."
    a, m = p + "self_attn.", p + "mlp."
    w.update({a + "q_proj.weight": arr((Hq * hd, D)), a + "k_proj.weight": arr((Hkv * hd, D)), a + "v_proj.weight": arr((Hkv * hd, D)),
              a + "o_proj.weight": arr((D, Hq * hd)), m + "gate_proj.weight": arr((I, D)), m + "up_proj.weight": arr((I, D)),
              m + "down_proj.weight": arr((D, I)), p + "input_layernorm.weight": arr((D,), True), p + "post_attention_layernorm.weight": arr((D,), True)})
    if sh.get("bias"): w.update({a + "q_proj.bias": arr((Hq * hd,)), a + "k_proj.bias": arr((Hkv * hd,)), a + "v_proj.bias": arr((Hkv * hd,))})
  return w

def config(sh: dict, n_layers: int, max_len: int) -> Any:
  from fusezap.serve.models.llama import LlamaConfig
  V = sh["vocab_size"]
  return LlamaConfig(sh["hidden_size"], sh["intermediate_size"], n_layers, sh["n_heads"], sh["n_kv_heads"], 128, V, 1e-5,
                     sh.get("rope_theta", 5e5), None, False, max(max_len, 8192), [V - 1], 1, "qwen2" if sh.get("bias") else "llama")

def build(sh: dict, device: str, dtype: Any, max_len: int, prefill_chunk: int = 512, share: bool = False, tp: Any = None,
          seed: int = 0, layers: int = 0) -> Any:
  """A Llama at shapes sh (layers: override the layer count). share: one layer's weights, reused by every layer (one
  GPU only). tp: a TPGroup; sh are then the full shapes and this rank keeps its slices."""
  from fusezap.serve.models.llama import Llama
  L = layers or sh["n_layers"]
  assert not (share and tp is not None), "share is for one device"
  m = Llama(config(sh, 1 if share else L, max_len), weights(sh, dtype, 1 if share else L, seed), dtype, device, max_len=max_len,
            max_batch=1, prefill_chunk=prefill_chunk, tp=tp)
  if share: m.layers, m.cfg.n_layers = m.layers * L, L
  return m
