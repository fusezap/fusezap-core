"""Sliding-window attention and the layer variants that come with it (tracker I105, part B), for the Llama-style decoder.

Two families, parsed from their HF configs (transformers 4.x published keys and 5.x `rope_parameters` per layer type):

  * Gemma 3 text (`gemma3_text`, e.g. google/gemma-3-1b-it): 5 sliding-window layers (window 512) per global layer,
    local layers RoPE theta 1e4 (`rope_local_base_freq`), global 1e6 (+ optional scaling); sandwich norms (input,
    post-attention, pre-feedforward, post-feedforward RMSNorm), RMSNorm weights as (1 + w), q/k RMSNorm per head,
    attention scale query_pre_attn_scalar^-0.5, embeddings times sqrt(hidden_size), GELU (tanh) MLP, tied lm_head.
    Gemma 2's attention / final logit softcapping is refused (Gemma 3 has none).
  * OLMo 3 (`olmo3`, e.g. allenai/Olmo-3-7B-Instruct): 3 sliding-window layers (window 4096) per full layer; post-norm
    only (RMSNorm on the attention and MLP outputs before the residual add, no pre-norm); q/k RMSNorm over all heads;
    full layers YaRN RoPE (cos / sin times its attention factor), sliding layers plain RoPE with the same theta.

A window W lets the query at position p see keys p - W + 1 .. p (HF: kv_idx > q_idx - W). Everything else follows the
LlamaConfig fields (`layer_windows`, `rope_local_theta`, `post_norm`, `pre_norm`, `norm_plus_one`, `embed_scale`, `act`,
`attn_scale`). Mistral's all-layer `sliding_window` maps onto `layer_windows` too.
"""
from __future__ import annotations
import math
import numpy as np

def _layer_types(c: dict) -> list[str]:
  n = c["num_hidden_layers"]
  if c.get("layer_types"): return list(c["layer_types"])
  pat = c.get("sliding_window_pattern") or c.get("_sliding_window_pattern")
  if c.get("model_type") == "olmo3": pat = pat or 4
  if not pat: return ["full_attention"] * n
  return ["full_attention" if (i + 1) % int(pat) == 0 else "sliding_attention" for i in range(n)]

def extras(c: dict) -> dict:
  """LlamaConfig overrides for a gemma3_text / olmo3 config (empty for other model types, except Mistral-style windows)."""
  mt = c.get("model_type")
  if mt == "gemma2": raise NotImplementedError("Gemma 2 (attention / final logit softcapping) is not implemented")
  if mt not in ("gemma3_text", "olmo3"): return {}
  if c.get("attn_logit_softcapping") or c.get("final_logit_softcapping"): raise NotImplementedError("logit softcapping is not implemented")
  types = _layer_types(c)
  W = int(c["sliding_window"])
  rp = c.get("rope_parameters") or {}
  full, slide = rp.get("full_attention") or {}, rp.get("sliding_attention") or {}
  out: dict = {"layer_windows": tuple(W if t == "sliding_attention" else 0 for t in types), "sliding_window": None}
  g_theta = full.get("rope_theta") or c.get("rope_theta") or 10000.0
  g_scaling = ({k: v for k, v in full.items() if k != "rope_theta"} if full.get("rope_type", "default") != "default" else None) or c.get("rope_scaling")
  out["rope_theta"], out["rope_scaling"] = float(g_theta), g_scaling
  out["rope_local_theta"] = float(slide.get("rope_theta") or c.get("rope_local_base_freq") or g_theta)
  assert slide.get("rope_type", "default") == "default", f"sliding-layer rope {slide}"
  if mt == "gemma3_text":
    out.update(norm_plus_one=True, post_norm=True, embed_scale=True, act=c.get("hidden_activation") or "gelu_pytorch_tanh",
               attn_scale=float(c.get("query_pre_attn_scalar") or c.get("head_dim", 256)) ** -0.5, qk_norm=True, qk_norm_full=False,
               tie_embeddings=bool(c.get("tie_word_embeddings", True)))
  else:
    out.update(post_norm=True, pre_norm=False, qk_norm=True, qk_norm_full=True)
  return out

def yarn_inv_freq(theta: float, d: int, rs: dict, max_position: int) -> tuple[np.ndarray, float]:
  """HF's _compute_yarn_parameters in float32: (inverse frequencies, attention factor that scales cos and sin)."""
  factor = rs.get("factor") or max_position / rs["original_max_position_embeddings"]
  af = rs.get("attention_factor")
  if af is None:
    ms = lambda s, m=1.0: 1.0 if s <= 1 else 0.1 * m * math.log(s) + 1.0
    af = ms(factor, rs["mscale"]) / ms(factor, rs["mscale_all_dim"]) if rs.get("mscale") and rs.get("mscale_all_dim") else ms(factor)
  bf, bs = rs.get("beta_fast") or 32, rs.get("beta_slow") or 1
  old = rs["original_max_position_embeddings"]
  dim_of = lambda r: (d * math.log(old / (r * 2 * math.pi))) / (2 * math.log(theta))
  lo, hi = dim_of(bf), dim_of(bs)
  if rs.get("truncate", True): lo, hi = math.floor(lo), math.ceil(hi)
  lo, hi = max(lo, 0), min(hi, d - 1)
  if lo == hi: hi += 0.001
  pos = (np.float32(theta) ** (np.arange(0, d, 2, dtype=np.int64).astype(np.float32) / np.float32(d))).astype(np.float32)
  extra, inter = (np.float32(1.0) / pos).astype(np.float32), (np.float32(1.0) / (np.float32(factor) * pos)).astype(np.float32)
  ramp = np.clip((np.arange(d // 2, dtype=np.float32) - np.float32(lo)) / np.float32(hi - lo), 0, 1).astype(np.float32)
  ef = (np.float32(1) - ramp).astype(np.float32)
  return (inter * (np.float32(1) - ef) + extra * ef).astype(np.float32), float(af)

def window_mask(kpos, qpos, win: int):
  """Key positions kpos (1, Sk) visible to query positions qpos (T, 1): causal, and inside the window when win > 0."""
  m = kpos <= qpos
  return m * (kpos > qpos - win) if win else m
