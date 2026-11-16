"""Llama-architecture decoder (SPEC §11.3), parametrized by an HF config.json. Covers Llama, Qwen2/Qwen2.5 (qkv biases),
Mistral and Qwen3 (per-head q/k RMSNorm before RoPE, no qkv bias): RMSNorm, RoPE (rope_theta, optional linear/llama3
scaling), GQA, SwiGLU, tied or untied embeddings.

  m = Llama.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtypes.float32, max_len=2048)
  logits = m.prefill([ids])           # (B, vocab) float32 numpy, logits of the last prompt position
  logits = m.decode([[tok]])          # one token per sequence; replays a captured fz.jit step
  hs = m.encode(ids, mask, (9, 18, 27))  # I81 D3 text encoder: HF hidden_states[k] for each k, no cache, no lm_head

KV cache interface (the model never touches cache storage directly, so a paged cache plugs in unchanged):
  cache.state() -> list[Tensor]               every tensor the step reads or rewrites (threaded through fz.jit as
                                              inputs; the new versions come back as outputs)
  cache.bind(state) -> KVCache                a view of the cache over `state` that a traced step mutates lazily
  cache.update(layer, k, v, start, T) -> (K, V)
                                              write k, v (B, Hkv, T, hd) at positions [start, start+T) and return
                                              keys/values (B, Hkv, Sk, hd) with Sk >= start+T, where row j holds
                                              position j for j < start+T. Rows past that must be finite; the model
                                              masks every key whose position is after the query's. start and T may
                                              be ints or Vars; Sk may be a concrete length (cheaper to reduce over).
  cache.inputs() -> list[Tensor]              extra per-step inputs (e.g. block tables); [] for the contiguous cache
  cache.load(state)                           adopt the step's output state
`ContiguousKVCache` keeps one (2, B, Hkv, L, hd) tensor per layer (k and v), rewritten with a select, and returns all
L rows (full_read=True: concrete reduce lengths, which the kernel optimizer can parallelize; a serial loop over a
symbolic length was ~45% of decode time at position 400) or the first start+T rows (full_read=False).
"""
from __future__ import annotations
import contextlib, math, os, time
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import DType, dtypes
from fusezap.device import Device
from fusezap.shape.symbolic import Var, Node
from fusezap.jit import Jit, bind
from fusezap.helpers import getenv
from fusezap.serve.weights import load_safetensors, resolve_model, load_config, to_dtype

sint = int | Node

@dataclass
class LlamaConfig:
  hidden_size: int
  intermediate_size: int
  n_layers: int
  n_heads: int
  n_kv_heads: int
  head_dim: int
  vocab_size: int
  rms_eps: float = 1e-6
  rope_theta: float = 10000.0
  rope_scaling: dict | None = None
  tie_embeddings: bool = False
  max_position: int = 4096
  eos_token_ids: list[int] = field(default_factory=list)
  bos_token_id: int | None = None
  model_type: str = "llama"
  sliding_window: int | None = None    # Mistral-style: every layer windowed (-> layer_windows)
  qk_norm: bool = False                # Qwen3: RMSNorm over head_dim on q and k (q_norm / k_norm weights), before RoPE
  qk_norm_full: bool = False           # OLMoE (I105): q_norm / k_norm over the whole projection (all heads at once)
  n_experts: int = 0                   # OLMoE (I105, models/moe.py): intermediate_size = n_experts * the expert width
  top_k: int = 0
  norm_topk: bool = False
  # I105 B (models/swa.py): per layer sliding window (0 = full attention), the sliding layers' RoPE theta, layer variants
  layer_windows: tuple | None = None
  rope_local_theta: float | None = None
  pre_norm: bool = True                # False: no input / pre-MLP RMSNorm (OLMo 3)
  post_norm: bool = False              # RMSNorm on the attention and MLP outputs before the residual add (Gemma 3, OLMo 3)
  norm_plus_one: bool = False          # RMSNorm weight (1 + w) (Gemma)
  embed_scale: bool = False            # embeddings times sqrt(hidden_size) in the model dtype (Gemma)
  act: str = "silu"                    # MLP activation: silu or gelu_pytorch_tanh (Gemma)
  attn_scale: float | None = None      # attention scale (Gemma: query_pre_attn_scalar ** -0.5); None: head_dim ** -0.5
  @property
  def plain(self) -> bool:
    """The layer shape the NEURON fast kernels (LayerSpec, prefill kernels v2) and CUDA fused paths are written for."""
    return not (self.n_experts or self.qk_norm or self.layer_windows or self.post_norm or not self.pre_norm or self.act != "silu"
                or self.rope_local_theta is not None or self.attn_scale is not None)
  @staticmethod
  def from_hf(c: dict) -> "LlamaConfig":
    act = c.get("hidden_activation") or c.get("hidden_act", "silu")
    assert act in ("silu", "gelu_pytorch_tanh"), f"unsupported activation {act}"
    moe = c.get("model_type") == "olmoe"   # I105: the experts as one dense SwiGLU of width E * Ie, masked (models/moe.py)
    if moe: assert c.get("clip_qkv") is None, "OLMoE clip_qkv is not implemented"
    E = int(c.get("num_experts") or 0) if moe else 0
    eos = c.get("eos_token_id")
    rp = c.get("rope_parameters") or {}   # transformers >= 5 keeps rope_theta and the scaling dict here
    theta = c.get("rope_theta") or rp.get("rope_theta") or 10000.0
    scaling = c.get("rope_scaling") or ({k: v for k, v in rp.items() if k != "rope_theta"} if rp.get("rope_type", "default") != "default" else None)
    cfg = LlamaConfig(c["hidden_size"], c["intermediate_size"] * max(E, 1), c["num_hidden_layers"], c["num_attention_heads"],
      c.get("num_key_value_heads") or c["num_attention_heads"], c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
      c["vocab_size"], c.get("rms_norm_eps", 1e-6), float(theta), scaling,
      bool(c.get("tie_word_embeddings", False)), c.get("max_position_embeddings", 4096),
      [eos] if isinstance(eos, int) else list(eos or []), c.get("bos_token_id"), c.get("model_type", "llama"),
      c.get("sliding_window") if c.get("use_sliding_window", c.get("model_type") == "mistral") else None,
      c.get("model_type") in ("qwen3", "olmoe"), moe, E, int(c.get("num_experts_per_tok") or 0) if moe else 0, bool(c.get("norm_topk_prob")) and moe)
    from fusezap.serve.models.swa import extras   # I105 B: Gemma 3 / OLMo 3 layer variants and sliding windows
    import dataclasses
    cfg = dataclasses.replace(cfg, **extras(c))
    if cfg.sliding_window: cfg.layer_windows = (int(cfg.sliding_window),) * cfg.n_layers   # Mistral-style: every layer
    return cfg

def rope_inv_freq(cfg: LlamaConfig) -> np.ndarray:
  """float32 inverse frequencies, computed like HF (`1 / theta ** (arange(0, d, 2) / d)` in float32)."""
  d = cfg.head_dim
  inv = (1.0 / (np.float32(cfg.rope_theta) ** (np.arange(0, d, 2, dtype=np.int64).astype(np.float32) / np.float32(d)))).astype(np.float32)
  rs = cfg.rope_scaling or {}
  kind = rs.get("rope_type", rs.get("type"))
  if kind == "linear": inv = (inv / np.float32(rs["factor"])).astype(np.float32)
  elif kind == "llama3":
    f, lo, hi, old = rs["factor"], rs.get("low_freq_factor", 1.0), rs.get("high_freq_factor", 4.0), rs["original_max_position_embeddings"]
    wl = 2 * math.pi / inv
    smooth = (old / wl - lo) / (hi - lo)
    out = np.where(wl > old / lo, inv / f, inv)
    mid = (wl <= old / lo) & (wl >= old / hi)
    inv = np.where(mid, (1 - smooth) * out / f + smooth * out, out).astype(np.float32)
  elif kind == "yarn":   # I105 B (OLMo 3's full layers): models/swa.py
    from fusezap.serve.models.swa import yarn_inv_freq
    inv = yarn_inv_freq(cfg.rope_theta, d, rs, cfg.max_position)[0]
  elif kind not in (None, "default"): raise NotImplementedError(f"rope_scaling {rs}")
  return inv

def rope_tables(cfg: LlamaConfig, max_len: int) -> tuple[np.ndarray, np.ndarray]:
  """(max_len, head_dim/2) cos and sin, float32 (HF: freqs = pos * inv_freq in float32, then cos/sin)."""
  fr = np.arange(max_len, dtype=np.float32)[:, None] * rope_inv_freq(cfg)[None, :]
  rs = cfg.rope_scaling or {}
  if rs.get("rope_type", rs.get("type")) == "yarn":   # HF: cos and sin times the attention factor
    from fusezap.serve.models.swa import yarn_inv_freq
    af = np.float32(yarn_inv_freq(cfg.rope_theta, cfg.head_dim, rs, cfg.max_position)[1])
    return (np.cos(fr) * af).astype(np.float32), (np.sin(fr) * af).astype(np.float32)
  return np.cos(fr).astype(np.float32), np.sin(fr).astype(np.float32)

def rope_local_tables(cfg: LlamaConfig, max_len: int) -> tuple[np.ndarray, np.ndarray] | None:
  """I105 B: the sliding layers' own (max_len, head_dim/2) tables (rope_local_theta, no scaling), None if they share the global ones."""
  if cfg.rope_local_theta is None: return None
  import dataclasses
  return rope_tables(dataclasses.replace(cfg, rope_theta=cfg.rope_local_theta, rope_scaling=None), max_len)

def rope_host(cfg: LlamaConfig, max_len: int, neuron: bool = False) -> tuple[np.ndarray, np.ndarray]:
  """The engine's host RoPE rows: (global | local) side by side when the sliding layers have their own (Llama.rope_pair
  splits them per layer); NEURON: each half doubled (neuron_kv.rope_rows)."""
  pairs = [rope_tables(cfg, max_len)] + ([lt] if (lt := rope_local_tables(cfg, max_len)) is not None else [])
  if neuron:
    from fusezap.serve.neuron_kv import rope_rows
    pairs = [rope_rows(*p) for p in pairs]
  return np.concatenate([p[0] for p in pairs], -1), np.concatenate([p[1] for p in pairs], -1)

# ---------------------------------------------------------------- KV cache

class KVCache:
  """Interface (see module docstring). Subclasses: ContiguousKVCache here; a paged cache in fusezap/serve/kv_cache.py."""
  def state(self) -> list[Tensor]: raise NotImplementedError
  def inputs(self) -> list[Tensor]: return []
  def bind(self, state: list[Tensor], inputs: list[Tensor] | None = None) -> "KVCache": raise NotImplementedError
  def load(self, state: list[Tensor]) -> None: raise NotImplementedError
  def update(self, layer: int, k: Tensor, v: Tensor, start: sint, T: sint) -> tuple[Tensor, Tensor]: raise NotImplementedError

class ContiguousKVCache(KVCache):
  """Per layer one (2, B, Hkv, max_len, hd) tensor holding k and v. update() rewrites it with a select (decode: the new
  row broadcast where arange == start; prefill: a one-hot scatter), which is exact and needs no in-place op. k and v
  share the tensor so one kernel writes both. Rows never written stay zero, rows of an earlier sequence stay finite."""
  def __init__(self, n_layers: int, B: int, n_kv: int, max_len: int, hd: int, dtype: DType, device: str,
               _state: list[Tensor] | None = None, _pos: Tensor | None = None, full_read: bool = True):
    self.full_read = full_read
    self.n_layers, self.B, self.n_kv, self.max_len, self.hd, self.dtype, self.device = n_layers, B, n_kv, max_len, hd, dtype, device
    self._pos = _pos if _pos is not None else Tensor(np.arange(max_len, dtype=np.int32), device=device).realize()
    self._kv = _state
  @property
  def kv(self) -> list[Tensor]:   # allocated on first use: a model served by LLMEngine (paged cache) never touches it
    if self._kv is None: self._kv = self._zeros()
    return self._kv
  @kv.setter
  def kv(self, v: list[Tensor]) -> None: self._kv = v
  def _zeros(self) -> list[Tensor]:
    return [Tensor.zeros(2, self.B, self.n_kv, self.max_len, self.hd, dtype=self.dtype, device=self.device).contiguous().realize()
            for _ in range(self.n_layers)]
  def state(self) -> list[Tensor]: return list(self.kv)
  def bind(self, state: list[Tensor], inputs: list[Tensor] | None = None) -> "ContiguousKVCache":
    return ContiguousKVCache(self.n_layers, self.B, self.n_kv, self.max_len, self.hd, self.dtype, self.device, list(state), self._pos, self.full_read)
  def load(self, state: list[Tensor]) -> None: self.kv = list(state)
  def reset(self) -> None: self._kv = None
  def update(self, layer: int, k: Tensor, v: Tensor, start: sint, T: sint) -> tuple[Tensor, Tensor]:
    B, H, L, hd = self.B, self.n_kv, self.max_len, self.hd
    pos, kv = self._pos, k.unsqueeze(0).cat(v.unsqueeze(0), axis=0)          # (2, B, H, T, hd)
    if isinstance(T, int) and T == 1:
      sel = (pos == Tensor(start, device=self.device)).reshape(1, 1, 1, L, 1)
      new = sel.where(kv.expand(2, B, H, L, hd), self.kv[layer])
    else:
      rel = pos - Tensor(start, device=self.device)                                    # (L,) row j holds position j
      valid = ((rel >= 0) * (rel < Tensor(T, device=self.device))).reshape(1, 1, 1, L, 1)
      oh = (rel.reshape(L, 1) == pos.shrink(((0, T),)).reshape(1, T)).cast(self.dtype)  # (L, T) one-hot scatter
      new = valid.where(oh @ kv, self.kv[layer])
    self.kv[layer] = new
    S = L if self.full_read else start + T
    k_, v_ = [new.shrink(((i, i + 1), None, None, (0, S), None)).reshape(B, H, S, hd) for i in (0, 1)]
    return k_, v_

# ---------------------------------------------------------------- model

class _Layer:
  wqkv: Tensor; bqkv: Tensor | None; wo: Tensor; bo: Tensor | None; wgu: Tensor; bgu: Tensor | None; wd: Tensor; bd: Tensor | None
  ln1: Tensor; ln2: Tensor
  qn: Tensor | None = None; kn: Tensor | None = None   # Qwen3 q_norm / k_norm (head_dim,); OLMoE (Hq*hd,) / (Hkv*hd,)
  wr: Tensor | None = None   # I105 MoE router (n_experts, D) float32 (K-major on NEURON)
  pn1: Tensor | None = None; pn2: Tensor | None = None   # I105 B post-attention / post-MLP RMSNorm (Gemma 3, OLMo 3)

class _NoCache(KVCache):
  """Llama.encode's cache: keeps nothing, returns this step's own keys and values (positions [0, T))."""
  def update(self, layer: int, k: Tensor, v: Tensor, start: sint, T: sint) -> tuple[Tensor, Tensor]: return k, v

class _LoadProgress:
  """The weight load's progress on this rank: with tensor parallelism and FZ_TP_DEBUG=1 one line at the start and every
  10 s (layer, GB on the device, GB/s, host memory: this rank's RSS, MemAvailable, the cgroup's limit and use) through
  dist.dbg, so a slow rank (or a container near its memory limit) shows in the server log while rank 0 waits at 'load'."""
  def __init__(self, tp: Any, n_layers: int):
    from fusezap.serve import dist
    self.on = tp is not None and dist.debug_on()
    self.dbg, self.n, self.nb, self.t0 = dist.dbg, n_layers, 0, time.perf_counter()
    self.t_log = self.t0
    if self.on: self._line("start")
  def _line(self, what: str) -> None:
    dt = time.perf_counter() - self.t0
    from fusezap.serve.weights import host_mem_str
    self.dbg(f"load: {what}, {self.nb / 1e9:.1f} GB on the device in {dt:.0f} s ({self.nb / 1e9 / max(dt, 1e-3):.2f} GB/s); host: {host_mem_str()}")
  def layer(self, i: int, nbytes: int) -> None:
    if self.on and time.perf_counter() - self.t_log > 10: self.t_log = time.perf_counter(); self._line(f"layer {i}/{self.n}")
    self.nb += nbytes
  def done(self, nbytes: int) -> None:
    self.nb += nbytes
    if self.on: self._line(f"{self.n} layers and lm_head")

def _cut(t: Tensor | None) -> Tensor | None:
  """t realized as a new graph root (no srcs), so the layer's intermediates can be freed (flux2.cut)."""
  if t is None: return None
  from fusezap.lazy import LazyBuffer
  lb = t._realized_contiguous()
  return Tensor(LazyBuffer.from_buffer(lb.realized, t.shape))

class Llama:
  def __init__(self, cfg: LlamaConfig, weights: dict[str, np.ndarray], dtype: DType = dtypes.float32, device: str | None = None,
               max_len: int = 2048, max_batch: int = 1, prefill_chunk: int = 512, beam: int | None = None, tp: Any = None,
               load_layers: int | None = None, lm_head: bool = True):
    """weights: HF tensor names -> host arrays (e.g. load_safetensors). max_len: context length the cache holds.
    max_batch: the cache's batch size (all sequences of a batch share positions). prefill_chunk: the largest prefill bucket.
    beam: FZ_BEAM width for the decode kernels (default: the FZ_BEAM env var; 0 = hand-coded opts, FZ_OPT=1). Decode
    linear layers run as composed kernels (FZ_CUSTOM=0 while compiling): at M = batch rows the optimized composed GEMV
    beats the simdgroup GEMM, whose 128x64 tiles leave most of the GPU idle. Prefill keeps the library GEMM.
    tp: a fusezap.serve.dist.TPGroup (tensor parallelism): this rank's slices of the weights (dist.tp_slice), self.cfg
    the local sizes (self.full_cfg the checkpoint's), an all-reduce after o_proj and down_proj, and prefill / decode /
    reset mirrored on the other ranks. None (default): one device, exactly as before.
    load_layers / lm_head=False (I81 D3, a text encoder that reads hidden_states[k] only): keep the first load_layers
    layers (cfg.n_layers becomes that) and no lm_head on the device; encode() works, prefill/decode/generate do not."""
    self.ckpt_layers = cfg.n_layers   # the checkpoint's layer count (encode: hidden_states[ckpt_layers] is the final norm's output)
    if load_layers is not None and load_layers < cfg.n_layers:
      import dataclasses
      cfg = dataclasses.replace(cfg, n_layers=load_layers)
    self.beam = int(os.environ.get("FZ_BEAM", "0") or 0) if beam is None else beam
    self.tp, self.full_cfg = tp, cfg
    # I105 expert parallelism (an MoE model with a group): each rank holds n_experts / world experts and everything else
    # whole (attention, norms, router, lm_head), so only the expert MLP's output is all-reduced (moe.py)
    self.ep = tp is not None and tp.world > 1 and cfg.n_experts > 0
    self.ep_lo, self.moe_slice = 0, False
    if tp is not None and tp.world > 1:
      from fusezap.serve.dist import local_cfg, tp_slice
      if self.ep:
        assert cfg.n_experts % tp.world == 0, f"{cfg.n_experts} experts over {tp.world} ranks"
        import dataclasses
        self.ep_lo = tp.rank * cfg.n_experts // tp.world
        from fusezap.serve.models.moe import tp_layout
        # I105 A2 FZ_MOE_TP=slice (the default with FZ_MOE_SPARSE): every expert's columns [r Ie/W, (r+1) Ie/W) per rank
        self.moe_slice = tp_layout() == "slice"
        if self.moe_slice: assert (cfg.intermediate_size // cfg.n_experts) % tp.world == 0, "expert width over the ranks"
        cfg = dataclasses.replace(cfg, intermediate_size=cfg.intermediate_size // tp.world)
      else: cfg = local_cfg(cfg, tp.world)
    else: self.tp = None
    # I68: bf16 on NEURON (fusezap/codegen/nki_bf16.py): the linear weights are bf16 (K-major, relaid on the host), the
    # activations, norms, softmax and logits float32 (self.dtype); --dtype bfloat16 or FZ_NEURON_DTYPE=bf16 select it
    from fusezap.serve.neuron_kv import is_neuron as _isn
    from fusezap.codegen.nki_bf16 import neuron_bf16_env
    # N8 (I72): also per tensor-parallel shard (TK relays the rank's slice); a TP group of one is self.tp None
    self.nbf16 = _isn(device or Device.DEFAULT) and (dtype == dtypes.bfloat16 or neuron_bf16_env())
    if self.nbf16: dtype = dtypes.float32
    self.wdtype = dtypes.bfloat16 if self.nbf16 else dtype
    self.cfg, self.dtype, self.device = cfg, dtype, Device.canonicalize(device)
    self.max_len, self.prefill_chunk = max_len, min(prefill_chunk, max_len)
    dev, c = self.device, cfg
    if not c.plain and (_cu_dev := Device.canonicalize(device or Device.DEFAULT).split(":")[0].upper()) == "CUDA" and c.layer_windows:
      raise NotImplementedError("sliding-window attention is not in the CUDA paged kernels (I105)")
    if c.qk_norm_full and self.tp is not None and not self.ep:   # the q/k norm spans every rank's heads (OLMo 3)
      raise NotImplementedError("q/k norm over all heads under tensor parallelism is not implemented (I105)")
    def W(name: str) -> np.ndarray: return weights[name] if self.tp is None or self.ep else tp_slice(name, weights[name], self.full_cfg, tp.rank, tp.world)
    def T(a: np.ndarray, host: bool = False) -> Tensor:   # host=True: `a` is already in dtype's host format
      return Tensor(a if host else to_dtype(a, dtype), device=dev, dtype=dtype).realize()
    def cat(names: list[str], axis: int = 0) -> Tensor | None:   # fused projections: one matmul instead of two or three
      if not all(n in weights for n in names): return None
      return T(np.concatenate([to_dtype(W(n), dtype) for n in names], axis), host=True)
    def TK(names: list[str], axis: int = 0, part: Any = None) -> Tensor | None:   # I68 NEURON bf16: a linear weight (or a fused one) bf16, K-major
      if not all(n in weights for n in names): return None   # axis 1: stacked along the input (I105: the experts' down_proj)
      R = (lambda n: W(n)) if part is None else part   # part(name): a slice of the tensor (I105 A2: the sliced expert layout)
      if not self.nbf16:
        if part is None: return cat(names, axis) if len(names) > 1 else T(W(names[0]))
        return T(np.concatenate([to_dtype(R(n), dtype) for n in names], axis), host=True)
      a = np.concatenate([to_dtype(R(n), dtypes.bfloat16) for n in names], axis)
      return Tensor(np.ascontiguousarray(a.T), device=dev, dtype=dtypes.bfloat16).realize()
    def NT(name: str, W=W) -> Tensor | None:   # an RMSNorm weight; Gemma's (1 + w) in float32 first (I105 B); None if absent
      if name not in weights: return None
      return T((to_dtype(W(name), dtypes.float32) + np.float32(1)) if c.norm_plus_one else W(name))
    self.layers: list[_Layer] = []
    from fusezap.serve.weights import prefetch, release, close_prefetch
    by_layer: dict[int, list[str]] = {}
    for k in weights:
      if k.startswith("model.layers."): by_layer.setdefault(int(k.split(".")[2]), []).append(k)
    # Host memory stays bounded by a few layers: the file ranges of `ahead` layers are read ahead (WILLNEED), and a
    # layer's pages are released (weights.release) `lag` layers after its upload (TP ranks read the same files in step,
    # so the lag keeps one rank from dropping pages its peer is about to read); the device copies keep no host data.
    ahead = int(os.environ.get("FZ_PREFETCH_LAYERS", "") or 2)
    lag = int(os.environ.get("FZ_RELEASE_LAG", "") or 4)
    lw = lambda i: [W(k) for k in by_layer.get(i, [])] if 0 <= i < c.n_layers else []
    for i in range(ahead): prefetch(lw(i))
    prog = _LoadProgress(self.tp, c.n_layers)
    for i in range(c.n_layers):
      prefetch(lw(i + ahead))
      prog.layer(i, sum(int(np.prod(W(k).shape)) * self.wdtype.itemsize for k in by_layer.get(i, [])))
      p, l = f"model.layers.{i}.", _Layer()
      a, m = p + "self_attn.", p + "mlp."
      l.wqkv, l.bqkv = TK([a + "q_proj.weight", a + "k_proj.weight", a + "v_proj.weight"]), cat([a + "q_proj.bias", a + "k_proj.bias", a + "v_proj.bias"])
      l.wo, l.bo = TK([a + "o_proj.weight"]), (T(W(a + "o_proj.bias")) if a + "o_proj.bias" in weights else None)
      l.wgu, l.bgu = TK([m + "gate_proj.weight", m + "up_proj.weight"]), cat([m + "gate_proj.bias", m + "up_proj.bias"])
      l.wd, l.bd = TK([m + "down_proj.weight"]), (T(W(m + "down_proj.bias")) if m + "down_proj.bias" in weights else None)
      if c.n_experts:   # I105: every expert's gate rows, then every up row; down's columns in the same expert order (moe.py)
        from fusezap.serve.models.moe import expert_names as ex
        if self.moe_slice:   # I105 A2: every expert, this rank's Iw = Ie / W of its columns (gate / up rows, down columns)
          Iw = c.intermediate_size // c.n_experts
          rs = slice(tp.rank * Iw, (tp.rank + 1) * Iw)
          l.wgu, l.bgu = TK(ex(m, c.n_experts, "gate") + ex(m, c.n_experts, "up"), part=lambda n: W(n)[rs]), None
          l.wd, l.bd = TK(ex(m, c.n_experts, "down"), axis=1, part=lambda n: W(n)[:, rs]), None
        else:
          ne = c.n_experts // (tp.world if self.ep else 1)   # expert parallelism: this rank's experts [ep_lo, ep_lo + ne)
          l.wgu, l.bgu = TK(ex(m, ne, "gate", self.ep_lo) + ex(m, ne, "up", self.ep_lo)), None
          l.wd, l.bd = TK(ex(m, ne, "down", self.ep_lo), axis=1), None
        l.wr = Tensor(to_dtype(W(m + "gate.weight"), dtypes.float32) if self.nbf16 else to_dtype(W(m + "gate.weight"), dtype), device=dev,
                      dtype=dtypes.float32 if self.nbf16 else dtype).realize()   # NEURON bf16: the router stays float32
      if not c.post_norm: l.ln1, l.ln2 = NT(p + "input_layernorm.weight"), NT(p + "post_attention_layernorm.weight")
      else:   # I105 B: Gemma 3 input / pre_feedforward (pre-norms), post_attention / post_feedforward on the outputs; OLMo 3 the latter two
        l.ln1, l.ln2 = NT(p + "input_layernorm.weight"), NT(p + "pre_feedforward_layernorm.weight")
        l.pn1, l.pn2 = NT(p + "post_attention_layernorm.weight"), NT(p + "post_feedforward_layernorm.weight")
      if c.qk_norm:   # (head_dim,): the same on every tensor-parallel rank
        l.qn, l.kn = NT(a + "q_norm.weight", W=lambda n: weights[n]), NT(a + "k_norm.weight", W=lambda n: weights[n])
      self.layers.append(l)
      release(lw(i - lag))
    for i in range(max(c.n_layers - lag, 0), c.n_layers): release(lw(i))
    if c.n_experts:
      from fusezap.serve.models.moe import tie_order
      self._moe_tri = tie_order(c.n_experts, dev)
      if _isn(dev):   # NEURON: the routing weights widened on TensorE (moe.mix; this rank's experts under EP)
        ne = c.n_experts // (tp.world if self.ep and not self.moe_slice else 1)
        from fusezap.serve.models.moe import repeat_matrix, pair_matrices, sparse_enabled, upper_order
        self._moe_rep = repeat_matrix(ne, c.intermediate_size // ne, dev)
        self._moe_pairs = pair_matrices(c.n_experts, dev)
        # I105 A2 (FZ_MOE_SPARSE): decode steps read only the chosen experts (one device, or the sliced layout under TP)
        self.moe_sparse = sparse_enabled() and (not self.ep or self.moe_slice)
        if self.moe_sparse: self._moe_upper = upper_order(c.n_experts, dev)
    self.norm = NT("model.norm.weight")
    self.embed_host = W("model.embed_tokens.weight")    # host (mmap) table: token rows are gathered on the host
    if not lm_head: self._init_encoder_only(c, max_len, max_batch)
    if not lm_head: return
    tied = c.tie_embeddings or "lm_head.weight" not in weights
    head = W("model.embed_tokens.weight") if tied else W("lm_head.weight")
    # N8 FZ_NEURON_TP_HEAD=vocab: this rank's V/W rows of lm_head; the logits are all-gathered (neuron_tp.gather_cols)
    from fusezap.serve import neuron_tp as _nt
    self.tp_vocab = self.tp is not None and _isn(device or Device.DEFAULT) and _nt.vocab_head() and c.vocab_size % self.tp.world == 0
    if self.tp_vocab:
      Vl = c.vocab_size // self.tp.world
      head = head[self.tp.rank * Vl:(self.tp.rank + 1) * Vl]
      _W = W
      W = lambda name: head if name in ("lm_head.weight", "model.embed_tokens.weight") else _W(name)   # TK reads the head through W
    prefetch([head])
    # I68 NEURON bf16: lm_head only K-major bf16 (shape (K, V)); no torch-layout device copy (7B: 1.09 GB saved)
    from fusezap.codegen.nki_bf16_n8 import bf16_head, bf16_head_w, head_tiles_bf16
    self.head_v, self.lm_head_t = int(head.shape[0]), None
    if self.nbf16 and bf16_head(self.tp is not None):   # N8 (I70): bf16 lm_head in head_tiles' layout only (HeadBF16; no K-major copy)
      self.lm_head = self.lm_head_t = Tensor(head_tiles_bf16(to_dtype(head, dtypes.bfloat16), bf16_head_w()), device=dev, dtype=dtypes.bfloat16).realize()
    else: self.lm_head = TK(["lm_head.weight" if not tied else "model.embed_tokens.weight"]) if self.nbf16 else T(head)
    if not tied: release([head])   # a tied table stays: embed gathers its rows on the host
    prog.done(int(np.prod(head.shape)) * self.wdtype.itemsize)
    close_prefetch()
    # prefill runs padded to a power-of-two bucket, so tables and the cache have `prefill_chunk` rows of slack past max_len
    self.table_len = max_len + self.prefill_chunk
    cos, sin = rope_tables(c, self.table_len)
    self.cos, self.sin = T(cos), T(sin)
    lt = rope_local_tables(c, self.table_len)   # I105 B: the sliding layers' RoPE (Gemma 3: theta 1e4)
    self.cos_l, self.sin_l = (T(lt[0]), T(lt[1])) if lt is not None else (self.cos, self.sin)
    self._pos = Tensor(np.arange(self.table_len, dtype=np.int32), device=dev).realize()
    self.max_batch = max_batch
    self.cache: KVCache = ContiguousKVCache(c.n_layers, max_batch, c.n_kv_heads, self.table_len, c.head_dim, dtype, dev, _pos=self._pos)
    self.start_v = Var("start_pos", 0, max_len - 1)
    # CUDA: f16/bf16 linears run on the tensor-core GEMM (fusezap/serve/cuda.py) in every step; prefill-style forwards
    # with concrete positions use P3-ATTN's flash attention. Other devices (and float32) keep the composed kernels.
    from fusezap.serve import cuda as _cu
    self.cuda16 = _cu.is_cuda(dev) and dtype in _cu.HALVES and _cu.cc(dev) >= (8, 0)
    self._decode_jit = Jit(self._decode_step)
    self._prefill_jits: dict[int, tuple[Jit, Var]] = {}
    self.pos = 0
    # NEURON (N5): every matmul is the TensorE custom kernel (exception 3), whose moving tile is one contiguous run per
    # partition only for a K-major weight: the linear weights are relaid (K, N) once here (neuron.kmajor), and linear()
    # computes x @ w, the same products and sums as x.linear(w). FZ_NEURON_KMAJOR=0 keeps the torch layout (strided DMA).
    from fusezap.serve.neuron_kv import is_neuron
    self.neuron = is_neuron(dev)
    self.kmajor = self.neuron and (getenv("FZ_NEURON_KMAJOR", 1) or self.nbf16)   # N8: per shard too (relaid after tp_slice)
    self.lm_head_k = None   # lm_head_t: set above (N8 I70 HeadBF16's tiles) or below (N7 HeadSpec)
    if self.nbf16: self.lm_head_k = self.lm_head if self.lm_head_t is None else None   # I68: bf16 K-major (TK); N8: HeadBF16's tiles
    elif self.kmajor:   # the layers' weights are replaced; lm_head stays as well (callers read it torch-layout), +V*D*4 bytes
      from fusezap.codegen.nki_kernels import head_tiles, head_v2
      from fusezap.device.neuron import kmajor
      for l in self.layers: l.wqkv, l.wo, l.wgu, l.wd = (kmajor(w) for w in (l.wqkv, l.wo, l.wgu, l.wd))
      if head_v2() and getattr(self.lm_head, "apply", None) is None and self.lm_head.dtype == dtypes.float32:
        # N7: lm_head for nki_kernels.HeadSpec, relaid on the host once (contiguous weight blocks per partition)
        self.lm_head_t = Tensor(head_tiles(self.lm_head.numpy()), device=dev).realize()
      else: self.lm_head_k = kmajor(self.lm_head)
    if self.kmajor:   # I105: the MoE router, float32 K-major (also with bf16 weights)
      from fusezap.device.neuron import kmajor as _km
      for l in self.layers:
        if l.wr is not None: l.wr = _km(l.wr)
    self._neuron_steps: Any = None   # NEURON: prefill/decode through the ragged forward (neuron_kv.NeuronSteps), built on first use

  def _init_encoder_only(self, c: LlamaConfig, max_len: int, max_batch: int) -> None:
    """__init__'s tail for lm_head=False: RoPE tables and flags for forward(); no lm_head, no KV cache, no decode jit."""
    from fusezap.serve.weights import close_prefetch
    from fusezap.serve import cuda as _cu
    close_prefetch()
    self.lm_head = self.lm_head_t = self.lm_head_k = None
    self.head_v, self.tp_vocab, self.neuron, self.kmajor, self.max_batch, self.pos = 0, False, False, False, max_batch, 0
    self.table_len = max_len
    cos, sin = rope_tables(c, max_len)
    assert c.rope_local_theta is None, "encode() with per-layer RoPE is not implemented"
    self.cos = Tensor(to_dtype(cos, self.dtype), device=self.device, dtype=self.dtype).realize()
    self.sin = Tensor(to_dtype(sin, self.dtype), device=self.device, dtype=self.dtype).realize()
    self.cos_l, self.sin_l = self.cos, self.sin
    self._pos = Tensor(np.arange(max_len, dtype=np.int32), device=self.device).realize()
    self.cuda16 = _cu.is_cuda(self.device) and self.dtype in _cu.HALVES and _cu.cc(self.device) >= (8, 0)
    self._encode_jits: dict[tuple[int, int, tuple[int, ...]], Jit] = {}
    self._neuron_steps = None

  @staticmethod
  def from_pretrained(name_or_path: str, dtype: DType = dtypes.float32, device: str | None = None, **kw) -> "Llama":
    path = resolve_model(name_or_path)
    cfg = LlamaConfig.from_hf(load_config(path))
    gen = os.path.join(path, "generation_config.json")
    if os.path.exists(gen):   # chat models stop on more ids than config.json lists (Qwen: <|im_end|> and <|endoftext|>)
      import json
      with open(gen) as f: eos = json.load(f).get("eos_token_id")
      cfg.eos_token_ids = sorted({*cfg.eos_token_ids, *([eos] if isinstance(eos, int) else (eos or []))})
    return Llama(cfg, load_safetensors(path), dtype, device, **kw)

  # ---- building blocks
  def linear(self, x: Tensor, w: Any, b: Tensor | None = None, redo: tuple[int, int] | None = None) -> Tensor:
    """x @ w^T + b. A quantized weight (fusezap.serve.quant.QuantLinear) picks its own kernel; on CUDA f16/bf16 a plain
    weight runs on the tensor-core GEMM (the lm_head writes float32 logits there: rounding logits to bf16 makes near-ties
    flip the top-1); everywhere else this is Tensor.linear."""
    if (ap := getattr(w, "apply", None)) is not None: return ap(x, b)
    if getattr(self, "kmajor", False):   # NEURON: w is (in, out), relaid at load
      if w is self.lm_head and self.lm_head_t is not None:   # N7 FZ_NEURON_HEAD=v2: HeadSpec on the relaid weight
        from fusezap.serve.neuron_kv import head_matmul
        return head_matmul(x, self.lm_head_t, self.head_v)
      if w.dtype == dtypes.bfloat16:   # I68: bf16 weights, float32 activations (nki_bf16.MatmulBF16)
        from fusezap.serve.neuron_kv import matmul_bf16
        y = matmul_bf16(x, w)
      else: y = x @ (self.lm_head_k if w is self.lm_head else w)
      return y + b if b is not None else y
    if self.cuda16 and x.shape[-1] % 8 == 0 and all(isinstance(s, int) for s in x.shape):
      from fusezap.serve.cuda import linear
      return linear(x, w, b, dtypes.float32 if w is self.lm_head else None, redo)
    return x.linear(w, b)
  def _rowpar(self, x: Tensor, w: Any, b: Tensor | None, redo: tuple[int, int] | None = None, fuse: bool = False, reduce: bool = True) -> Tensor:
    """A row-parallel projection (o_proj, down_proj): self.linear on one device; with tensor parallelism this rank's
    partial product, all-reduced over the ranks, then the bias once. fuse (I75, NEURON TP prefill): the all-reduce as a
    collective the fused member kernels may hold (dist.all_reduce)."""
    if self.tp is None or not reduce: return self.linear(x, w, b, redo)   # reduce=False: a replicated projection (I105 EP: o_proj)
    from fusezap.serve.dist import all_reduce
    y = all_reduce(self.linear(x, w, None, redo), self.tp, fuse=fuse)
    return y if b is None else y + b
  def _ar_add_norm(self, p: Tensor, x: Tensor, w: Tensor) -> tuple[Tensor, Tensor]:
    """_add_norm(x, all_reduce(p), w) as one kernel (dist.all_reduce_add_norm; forward_tokens checks dist.fuse_ok)."""
    from fusezap.serve.dist import all_reduce_add_norm
    return all_reduce_add_norm(p, x, w, self.cfg.rms_eps, self.tp)
  def _spmd(self, op: str, *args: Any):
    """Mirror this call on the other tensor-parallel ranks (dist.TPGroup.spmd); a no-op on one device."""
    return self.tp.spmd(op, *args) if self.tp is not None else contextlib.nullcontext()
  def qkv_rope_fused(self, h: Tensor, l: _Layer, cos: Tensor, sin: Tensor, redo: tuple[int, int] | None = None, kvw: Any = None) -> Tensor:
    """forward_tokens' fused qkv path (CUDA f16/bf16, plain weights, >= cuda.fuse_rows() rows, or any rows with kvw): h
    (N, D), cos / sin (N, head_dim/2) -> (N, Hq + 2 Hkv, head_dim), q and k rotated. With a qkv bias the GEMM writes its
    f32 accumulator and cuda.rope_qkv adds the bias before the one rounding (round(hW + b), as cuBLAS's bias epilogue; the
    same bits as Llama.linear's composed bias add); without one the GEMM rounds to the model dtype itself. kvw (a
    kv_cache.KVDest): the same kernel stores k and v into the paged pool (I60)."""
    from fusezap.serve import cuda as _cu
    c = self.cfg
    y = _cu.linear(h, l.wqkv, None, dtypes.float32 if l.bqkv is not None and not _cu.bias_round2() else None, redo)
    return _cu.rope_qkv(y, l.bqkv, cos, sin, c.n_heads + 2 * c.n_kv_heads, c.n_heads + c.n_kv_heads, c.head_dim, kvw)
  def _flash(self, start: sint, T: sint, Sk: sint) -> bool:
    return self.cuda16 and isinstance(start, int) and isinstance(T, int) and isinstance(Sk, int) and self.cfg.head_dim in (64, 128) and self.cfg.plain
  def _rmsnorm(self, x: Tensor, w: Tensor) -> Tensor:
    xf = x.float()
    y = xf * ((xf * xf).mean(-1, keepdim=True) + self.cfg.rms_eps).rsqrt()
    return w * y.cast(x.dtype)
  def _act(self, g: Tensor) -> Tensor: return g.gelu() if self.cfg.act == "gelu_pytorch_tanh" else g.silu()
  def _post(self, y: Tensor, w: Tensor | None) -> Tensor:
    """I105 B: a post-norm (Gemma 3 / OLMo 3: RMSNorm on a sublayer's output before the residual add); y itself without one."""
    return y if w is None else self._rmsnorm(y, w)
  def _add_norm(self, x: Tensor, d: Tensor | None, w: Tensor | None) -> tuple[Tensor, Tensor]:
    """(x + d, _rmsnorm(x + d, w)), or (x, _rmsnorm(x, w)) when d is None. CUDA f16/bf16: one fused row kernel
    (fusezap.serve.cuda.rmsnorm) that also writes the residual stream; the composed reduce re-read every earlier layer's
    residual terms, since the stream was never realized."""
    if w is None:   # I105 B: no pre-norm (OLMo 3): the sublayer reads the residual stream itself
      x = x if d is None else x + d
      return x, x
    if self.cuda16 and (d is None or d.dtype == x.dtype):
      from fusezap.serve import cuda as _cu
      if _cu.rmsnorm_ok(x): return _cu.rmsnorm(x, w, self.cfg.rms_eps, d)
    if getattr(self, "neuron", False) and x.dtype == dtypes.float32 and (d is None or d.dtype == x.dtype) and getenv("FZ_NEURON_NORM", 1) and \
       all(isinstance(s, int) for s in x.shape):   # NEURON (N5): one kernel that also stores the residual stream
      from fusezap.serve.neuron_kv import rmsnorm
      return rmsnorm(x, w, self.cfg.rms_eps, d)
    if d is not None: x = x + d
    return x, self._rmsnorm(x, w)
  def _rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    h = x.shape[-1] // 2
    x1, x2 = x.shrink((None, None, None, (0, h))), x.shrink((None, None, None, (h, 2 * h)))
    return (x1 * cos + (-x2) * sin).cat(x2 * cos + x1 * sin, axis=-1)
  def embed(self, ids: np.ndarray) -> np.ndarray:
    """(B, T) token ids -> (B, T, D) host rows in the model dtype's host format."""
    e = self.embed_host[np.asarray(ids, np.int64)]
    if self.cfg.embed_scale:   # Gemma: rows times sqrt(hidden_size) rounded to the model dtype, one rounding of the product
      from fusezap.serve.weights import bf16_to_f32, f32_to_bf16
      sc = np.array([math.sqrt(self.cfg.hidden_size)], np.float32)
      if self.dtype == dtypes.bfloat16: sc = bf16_to_f32(f32_to_bf16(sc))
      elif self.dtype == dtypes.float16: sc = sc.astype(np.float16).astype(np.float32)
      return to_dtype(to_dtype(e, dtypes.float32) * sc[0], self.dtype)
    return to_dtype(e, self.dtype)

  def forward(self, x: Tensor, cache: KVCache, start: sint, T: sint, last_only: bool = True, last: sint | None = None,
              hiddens: list[Tensor] | None = None, head: bool = True, key_mask: Tensor | None = None, upto: int | None = None,
              cut_layers: bool = False) -> Tensor:
    """x: (B, T, D) embeddings at positions [start, start+T). Returns logits (B, 1 or T, vocab) in the model dtype (float32 on CUDA f16/bf16):
    for row `last` (default T-1) if last_only, else for every row.
    hiddens (optional) collects HF's hidden_states: the input of every layer, then the final-norm output.
    head=False stops before the final norm: (B, 1 or T, D), the residual stream that `self.head` turns into logits
    (fusezap.serve.logprobs scores a long sequence's rows in chunks that way).
    key_mask (B, Sk) bool (optional): keys that may be attended to, on top of the causal mask (HF's attention_mask: a
    padded batch's pad keys are hidden from every query). upto: run only layers [0, upto) and return the residual
    stream after them (HF hidden_states[upto]), every row (encode)."""
    c, B = self.cfg, x.shape[0]
    Hq, Hkv, hd = c.n_heads, c.n_kv_heads, c.head_dim
    G, S = Hq // Hkv, start + T
    rows = lambda t: t.shrink(((start, S), None)).reshape(1, 1, T, hd // 2)
    cos, sin = rows(self.cos), rows(self.sin)
    cos_l, sin_l = (rows(self.cos_l), rows(self.sin_l)) if self.cos_l is not self.cos else (cos, sin)   # I105 B: sliding layers' RoPE
    qpos = self._pos.shrink(((start, S),)).reshape(T, 1)
    scale = c.attn_scale or 1.0 / math.sqrt(hd)
    wins = c.layer_windows or ()
    masks: dict[int, Tensor] = {}   # per window size (0: causal only)
    d: Tensor | None = None   # the residual term not yet added to x
    # N8 (I71): fusion boundaries per layer on NEURON here too (the logprobs scorer's trunk): without them a captured pass
    # fused up to FZ_NEURON_FUSE_MAX items across layers into one NEFF, and its intermediates, all inside one launch, could
    # not share memory (Jit plan)
    nfence = getattr(self, "neuron", False)
    if nfence: from fusezap.serve.neuron_kv import fence
    for i, l in enumerate(self.layers if upto is None else self.layers[:upto]):
      if nfence and i: d = fence(d, i, self.device)
      x, h = self._add_norm(x, d, l.ln1)
      if hiddens is not None: hiddens.append(x)
      qkv = self.linear(h, l.wqkv, l.bqkv)
      q = qkv.shrink((None, None, (0, Hq * hd))).reshape(B, T, Hq, hd)
      k = qkv.shrink((None, None, (Hq * hd, (Hq + Hkv) * hd))).reshape(B, T, Hkv, hd)
      v = qkv.shrink((None, None, ((Hq + Hkv) * hd, (Hq + 2 * Hkv) * hd))).reshape(B, T, Hkv, hd).transpose(1, 2)
      if l.qn is not None and c.qk_norm_full:   # OLMoE: over all heads at once, before RoPE
        q = self._rmsnorm(q.reshape(B, T, Hq * hd), l.qn).reshape(B, T, Hq, hd)
        k = self._rmsnorm(k.reshape(B, T, Hkv * hd), l.kn).reshape(B, T, Hkv, hd)
      elif l.qn is not None: q, k = self._rmsnorm(q, l.qn), self._rmsnorm(k, l.kn)   # Qwen3: per head, before RoPE
      win = wins[i] if wins else 0
      cs, sn = (cos_l, sin_l) if win else (cos, sin)
      q, k = self._rope(q.transpose(1, 2), cs, sn), self._rope(k.transpose(1, 2), cs, sn)
      K, V = cache.update(i, k, v, start, T)                                  # (B, Hkv, Sk, hd), row j = position j
      Sk = K.shape[2]
      if key_mask is None and self._flash(start, T, Sk):   # CUDA flash attention: key j visible to query row l iff j <= start + l
        from fusezap.device.cuda_kernels.attention import attention as flash
        o = flash(q, K, V, causal=True, scale=scale, coff=start).transpose(1, 2).reshape(B, T, Hq * hd)
      else:
        if (mask := masks.get(win)) is None:   # key position <= query position (and inside the window, I105 B)
          from fusezap.serve.models.swa import window_mask
          mask = masks[win] = window_mask(self._pos.shrink(((0, Sk),)).reshape(1, Sk), qpos, win)
          if key_mask is not None: mask = masks[win] = mask.reshape(1, 1, 1, T, Sk) * key_mask.reshape(B, 1, 1, 1, Sk)
        s = (q.reshape(B, Hkv, G, T, hd) @ K.reshape(B, Hkv, 1, Sk, hd).transpose(-1, -2)) * scale
        s = mask.where(s, float("-inf"))
        o = s.softmax(-1) @ V.reshape(B, Hkv, 1, Sk, hd)                        # (B, Hkv, G, T, hd)
        o = o.reshape(B, Hq, T, hd).transpose(1, 2).reshape(B, T, Hq * hd)
      # I105 B2: NEURON float32 (K-major weights): o_proj's input as one contiguous buffer. The transposed view has two
      # views, so `o @ wo` did not match the TensorE matmul and ran as a composed MUL+SUM kernel that reads wo one element per
      # DMA descriptor for every row (the scorer's 256-row bucket at Gemma 3's 1024 x 1152: 302 M descriptors in one NEFF).
      # bf16 weights already materialize it (matmul_bf16).
      if getattr(self, "kmajor", False) and not self.nbf16: o = o.contiguous()
      x, h = self._add_norm(x, self._post(self._rowpar(o, l.wo, l.bo, reduce=not self.ep), l.pn1), l.ln2)
      gu = self.linear(h, l.wgu, l.bgu)
      I = c.intermediate_size
      g, u = gu.shrink((None, None, (0, I))), gu.shrink((None, None, (I, 2 * I)))
      a = self._act(g) * u
      if l.wr is not None: a = self._moe(h, a, l)   # I105: zero for the experts outside the row's top-k (moe.py)
      d = self._post(self._rowpar(a, l.wd, l.bd), l.pn2)
      if cut_layers: x, d = _cut(x), _cut(d)   # eager encode: one layer's graph (and its temporaries) at a time
    if d is not None: x = x + d
    if upto is not None: return x
    if last_only:
      last = T - 1 if last is None else last
      x = x.shrink((None, (last, last + 1), None))
    if not head: return x
    h = self._add_norm(x, None, self.norm)[1]
    if hiddens is not None: hiddens.append(h)
    return self._vocab(self.linear(h, self.lm_head))
  def head(self, x: Tensor) -> Tensor:
    """forward's last two ops on rows of the residual stream (forward(..., head=False)): final RMSNorm, then lm_head ->
    logits (float32 on CUDA f16/bf16). Row-wise, so any subset of rows gives the same values as the full forward."""
    return self._vocab(self.linear(self._add_norm(x, None, self.norm)[1], self.lm_head))

  def forward_tokens(self, x: Tensor, cache: Any, cos: Tensor, sin: Tensor, select: bool = True, wide_linear: Any = None) -> Tensor:
    """Ragged step for continuous batching (fusezap/serve/engine.py): x (N, D) embeddings of N tokens from any number of
    sequences, each at its own position; cos/sin (N, head_dim/2) are the RoPE rows of those positions. The cache owns
    the token -> sequence/position/slot metadata: cache.attend(layer, qkv, n_heads, scale) writes k/v into the paged
    cache and returns (N, Hq*hd) attention over each token's own sequence (keys 0..position), and cache.select(x) picks
    the rows whose logits are wanted. Returns float logits (R or N, vocab) in the model dtype.
    Every row is computed independently of the others (row-wise kernels, attention per token), so a token's result does
    not depend on which other tokens share the step.
    wide_linear(x, w, b) (optional) computes the wide projections (gate/up and lm_head) instead of Tensor.linear, e.g.
    with the device's library GEMM, which beats the composed kernels there at a few dozen rows."""
    c, N = self.cfg, x.shape[0]
    wl = wide_linear or self.linear
    Hq, Hkv, hd = c.n_heads, c.n_kv_heads, c.head_dim
    h2 = hd // 2
    # CUDA f16/bf16 with plain weights: the qkv bias + RoPE and SwiGLU's silu(g) * u run as single bandwidth-bound kernels
    # on the GEMM output buffers, with the composed expressions' ops and roundings (bit-identical); the composed kernels
    # ran them at 0.3-1.2 TB/s on an H100. A quantized weight's output is still lazy (e.g. fp8's row scale and cast),
    # which the composed kernels fuse, so those keep the composed path. Below FUSE_ROWS rows (decode) the composed kernels'
    # one-element-per-thread grids were a little faster (H100, 32 rows: 3.4 vs 3.7 us per launch); both paths give the
    # same bits, so the choice never changes a token. I60: with a paged cache the RoPE kernel also stores k / v into the
    # pool (FZ_ROPE_KVW, default on, every row count: one launch per layer fewer than RoPE + kv_write); FZ_FUSE_ROWS moves
    # the silu_mul threshold (and the RoPE one without FZ_ROPE_KVW).
    from fusezap.serve import cuda as _cu
    ok16 = self.cuda16 and _cu.rope_qkv_ok(x, hd, cos)
    fuse = ok16 and N >= _cu.fuse_rows()
    kvw = ok16 and _cu.rope_kvw_on() and getattr(cache, "kv_dest", None) is not None and cache.kv_dest(0) is not None
    def plain(w: Any) -> bool: return fuse and getattr(w, "apply", None) is None
    def plain_qkv(w: Any) -> bool: return (fuse or kvw) and getattr(w, "apply", None) is None and c.plain
    sc = c.attn_scale or 1.0 / math.sqrt(hd)
    wins = c.layer_windows or ()
    # I105 B: with per-layer RoPE the rows are (global | local) side by side (rope_host); a sliding layer takes the right half
    if c.rope_local_theta is not None:
      wd_ = cos.shape[-1] // 2
      halves = [(t.shrink((None, (0, wd_))), t.shrink((None, (wd_, 2 * wd_)))) for t in (cos, sin)]
      ropes = {0: (halves[0][0], halves[1][0]), 1: (halves[0][1], halves[1][1])}
    else: ropes = {0: (cos, sin), 1: (cos, sin)}
    def rope_of(i: int, flat: bool) -> tuple[Tensor, Tensor]:
      cs, sn = ropes[1 if wins and wins[i] else 0]
      return (cs, sn) if flat or getattr(cache, "rope_in_attend", False) else (cs.reshape(N, 1, h2), sn.reshape(N, 1, h2))   # NEURON: (N, hd) rows
    lay = getattr(cache, "layout", None)
    mix = getattr(lay, "mix", 0)   # LLMEngine mixed forward: rows >= mix are decode rows
    rd = (mix, N - mix) if mix and getattr(lay, "mix_exact", True) else None   # throughput mode: the prefill GEMM's bits will do
    d: Tensor | None = None   # the residual term not yet added to x
    # tensor parallelism with the custom all-reduce (dist.fuse_ok, decode rows): a row-parallel product's all-reduce runs
    # fused with the residual add + RMSNorm after it (dist.all_reduce_add_norm: the same bits); p is the product of the
    # previous layer's down_proj, not reduced yet
    if self.tp is not None:
      from fusezap.serve.dist import fuse_ok, _stamps
      fz = fuse_ok(self.tp, x)
      if (st := _stamps()) is not None: st.stamp(x._realized_contiguous().realized, "begin")   # FZ_TP_AR_STAMP (tpprof)
    else: fz = False
    p: Tensor | None = None
    nfence = getattr(self, "neuron", False) and getattr(cache, "rope_in_attend", False)   # N6: fusion boundaries (neuron_kv.fence)
    if nfence: from fusezap.serve.neuron_kv import fence
    # N7 step 3: a layer as one LayerSpec (FZ_NEURON_LAYER). N8 TP (I72): a shard's layer is LayerSpec's two halves around
    # its two all-reduces (after o_proj and after down_proj); FZ_NEURON_TP_LAYER=members keeps the shard's member kernels
    lv2 = nfence and getattr(self, "kmajor", False) and hasattr(cache, "layer") and c.plain and \
          (self.tp is None or os.environ.get("FZ_NEURON_TP_LAYER", "halves") != "members")   # I105: other layer shapes: the members
    ar = None
    if self.tp is not None:
      from fusezap.serve.dist import all_reduce as _ar
      ar = lambda t: _ar(t, self.tp)
    # I75 (FZ_NEURON_TP_PREFILL=fused): a prefill's member layers record their all-reduces as collectives the fused kernels
    # hold, and the forward is grouped pk layers per kernel (neuron_tp.fused_prefill: 0 = the separate all-reduces)
    pk = 0
    if nfence and self.tp is not None and getattr(lay, "prefill", False):
      from fusezap.serve.neuron_tp import fused_prefill
      pk = fused_prefill(self.tp)
    # I77 (FZ_NEURON_PREFILL_KERNEL=v2): a prefill forward's layers on the prefill kernels (serve/neuron_prefill.py: the
    # members' bits in 5 kernels + the same all-reduces), and its lm_head split over the ranks (FZ_NEURON_PFK_HEAD)
    pfk, arf = None, None
    if nfence and getattr(lay, "prefill", False):
      from fusezap.serve import neuron_prefill as _pf
      if _pf.enabled() and c.plain: pfk = _pf
      if pfk is not None: pk = _pf.big_fuse(int(N), pk)   # a 512-row chunk: groups of FZ_NEURON_PFK_FUSE_BIG layers per NEFF
      if pfk is not None and self.tp is not None:
        from fusezap.serve.dist import all_reduce as _ar2
        arf = lambda t: _ar2(t, self.tp, fuse=bool(pk))
    # I105 A2: a decode forward (not a prefill chunk, not a mixed step) of at most moe.sparse_rows() rows on NEURON
    sparse_ok = getattr(self, "moe_sparse", False) and getattr(self, "kmajor", False) and not getattr(lay, "prefill", False) and not mix
    if sparse_ok:
      from fusezap.serve.models.moe import sparse_rows
      sparse_ok = isinstance(N, int) and N <= sparse_rows()
    for i, l in enumerate(self.layers):
      if nfence and i: d = fence(d, i, self.device, every=pk)
      cos2, sin2 = rope_of(i, True)
      cos, sin = rope_of(i, False)
      if pfk is not None and p is None and pfk.layer_ok(self, l, N, x):
        x, d = pfk.layer(cache, i, x, d, l, cos2, sin2, Hq, 1.0 / math.sqrt(hd), c.rms_eps, arf)
        continue
      if lv2 and (r := cache.layer(i, x, d, l, cos2, sin2, Hq, 1.0 / math.sqrt(hd), c.rms_eps, **({"all_reduce": ar, "tp": self.tp} if ar else {}))) is not None:
        x, d = r
        continue
      if p is not None: x, h = self._ar_add_norm(p, x, l.ln1)
      else: x, h = self._add_norm(x, d, l.ln1)
      dest = None
      if getattr(cache, "rope_in_attend", False):   # NEURON (serve/neuron_kv.py): RoPE without cat, k / v written by a one-hot matmul
        qkv = self._qk_full(self.linear(h, l.wqkv, l.bqkv, rd), l)
        if l.qn is not None and not c.qk_norm_full: qkv = self._qk_heads(qkv, l)   # Gemma 3 / Qwen3 on the members: per head
        o = cache.attend_rope(i, qkv, cos2, sin2, Hq, sc)
      elif plain_qkv(l.wqkv):
        dest = cache.kv_dest(i) if kvw else None
        qkv = self.qkv_rope_fused(h, l, cos2, sin2, rd, dest)
        if dest is None: o = cache.attend(i, qkv, Hq, sc)
      else:
        qkv = self._qk_full(self.linear(h, l.wqkv, l.bqkv, rd), l).reshape(N, Hq + 2 * Hkv, hd)
        qk, v = qkv.shrink((None, (0, Hq + Hkv), None)), qkv.shrink((None, (Hq + Hkv, Hq + 2 * Hkv), None))
        if l.qn is not None and not c.qk_norm_full:   # Qwen3: q_norm on the Hq query heads, k_norm on the Hkv key heads, before RoPE
          qk = self._rmsnorm(qk.shrink((None, (0, Hq), None)), l.qn).cat(self._rmsnorm(qk.shrink((None, (Hq, Hq + Hkv), None)), l.kn), axis=1)
        x1, x2 = qk.shrink((None, None, (0, h2))), qk.shrink((None, None, (h2, hd)))
        qkv = (x1 * cos + (-x2) * sin).cat(x2 * cos + x1 * sin, axis=-1).cat(v, axis=1)   # the same RoPE expression as _rope
        o = cache.attend(i, qkv, Hq, sc)
      if dest is not None: o = cache.attend(i, qkv, Hq, sc, written=True)
      if fz and l.bo is None and l.pn1 is None and not self.ep: x, h = self._ar_add_norm(self.linear(o, l.wo, None, rd), x, l.ln2)
      else: x, h = self._add_norm(x, self._post(self._rowpar(o, l.wo, l.bo, rd, fuse=bool(pk), reduce=not self.ep), l.pn1), l.ln2)
      if l.wr is not None and sparse_ok:   # I105 A2: the chosen experts only (moe.moe_sparse; this rank's share under the sliced layout)
        y = self._moe_sparse(h, l)
        if fz and l.pn2 is None and i + 1 < len(self.layers): p, d = y, None
        else:
          if self.tp is not None:
            from fusezap.serve.dist import all_reduce as _ar_y
            y = _ar_y(y, self.tp, fuse=bool(pk))
          p, d = None, self._post(y, l.pn2)
        continue
      gu = self.linear(h, l.wgu, l.bgu, rd) if wide_linear is None else wl(h, l.wgu, l.bgu)
      I = c.intermediate_size
      if plain(l.wgu) and l.bgu is None and c.act == "silu" and _cu.silu_mul_ok(gu): a = _cu.silu_mul(gu)
      else: a = self._act(gu.shrink((None, (0, I)))) * gu.shrink((None, (I, 2 * I)))
      if l.wr is not None: a = self._moe(h, a, l)   # I105
      if fz and l.bd is None and l.pn2 is None and i + 1 < len(self.layers): p, d = self.linear(a, l.wd, None, rd), None
      else: p, d = None, self._post(self._rowpar(a, l.wd, l.bd, rd, fuse=bool(pk)), l.pn2)
    if nfence: d = fence(d, len(self.layers), self.device, hard=True, every=pk)
    if getattr(self, "neuron", False) and not select: return self._vocab(wl(self._add_norm(x, d, self.norm)[1], self.lm_head, None))   # one norm kernel
    if d is not None: x = x + d
    if select: x = cache.select(x)
    if pfk is not None and wide_linear is None and not getattr(self, "tp_vocab", False) and self.lm_head_t is None and \
       self.lm_head.dtype == dtypes.bfloat16 and pfk.head_mode(self.tp is not None) != "off":   # I77: the prefill's head (FZ_NEURON_PFK_HEAD)
      return pfk.head(self, self._add_norm(x, None, self.norm)[1], arf)
    return self._vocab(wl(self._add_norm(x, None, self.norm)[1], self.lm_head, None))
  def _qk_heads(self, qkv: Tensor, l: _Layer) -> Tensor:
    """Per-head q_norm / k_norm (Qwen3, Gemma 3) on a flat (N, (Hq + 2 Hkv) hd) qkv row block, v unchanged."""
    c = self.cfg
    N, hd, Hq, Hkv = qkv.shape[0], c.head_dim, c.n_heads, c.n_kv_heads
    y = qkv.reshape(N, Hq + 2 * Hkv, hd)
    q, k, v = y.shrink((None, (0, Hq), None)), y.shrink((None, (Hq, Hq + Hkv), None)), y.shrink((None, (Hq + Hkv, Hq + 2 * Hkv), None))
    return self._rmsnorm(q, l.qn).cat(self._rmsnorm(k, l.kn), v, axis=1).reshape(N, (Hq + 2 * Hkv) * hd)
  def _qk_full(self, qkv: Tensor, l: _Layer) -> Tensor:
    """OLMoE's q_norm / k_norm (I105) on a (N, (Hq + 2 Hkv) hd) qkv row block: RMSNorm over all of q's columns, then over
    all of k's, before RoPE; v unchanged. Other models: qkv itself."""
    if l.qn is None or not self.cfg.qk_norm_full: return qkv
    c = self.cfg
    nq, nk = c.n_heads * c.head_dim, c.n_kv_heads * c.head_dim
    q, k, v = qkv.shrink((None, (0, nq))), qkv.shrink((None, (nq, nq + nk))), qkv.shrink((None, (nq + nk, nq + 2 * nk)))
    return self._rmsnorm(q, l.qn).cat(self._rmsnorm(k, l.kn), v, axis=-1)
  def _moe(self, h: Tensor, a: Tensor, l: _Layer) -> Tensor:
    from fusezap.serve.models.moe import moe_act
    return moe_act(self, h, a, l)
  def _moe_sparse(self, h: Tensor, l: _Layer) -> Tensor:
    from fusezap.serve.models.moe import moe_sparse
    return moe_sparse(self, h, l)
  def _vocab(self, y: Tensor) -> Tensor:
    """N8 FZ_NEURON_TP_HEAD=vocab: this rank's logits columns -> every rank's full logits (neuron_tp.gather_cols)."""
    if not getattr(self, "tp_vocab", False): return y
    from fusezap.serve.neuron_tp import gather_cols
    return gather_cols(y, self.tp, self.full_cfg.vocab_size)

  @contextlib.contextmanager
  def _opt_env(self, prefill: bool = False):
    """Kernel optimizer settings while a step compiles (lower.run_compiled reads them from os.environ per call).
    Decode (M = batch rows) uses composed GEMV kernels; prefill's concrete-M matmuls may use the device's library GEMM."""
    new = {"FZ_CUSTOM": "1" if prefill else "0", **({"FZ_BEAM": str(self.beam)} if self.beam > 0 and not prefill else {"FZ_OPT": "1", "FZ_BEAM": "0"})}
    if getattr(self, "neuron", False): new = {"FZ_CUSTOM": "1", "FZ_OPT": "1", "FZ_BEAM": "0"}   # NEURON: matmul/attention must hit TensorE
    old = {k: os.environ.get(k) for k in new}
    os.environ.update(new)
    getenv.cache_clear()   # fusezap.helpers.getenv is lru_cached; the scheduler reads FZ_CUSTOM through it
    try: yield
    finally:
      for k, v in old.items():
        if v is None: os.environ.pop(k, None)
        else: os.environ[k] = v
      getenv.cache_clear()

  # ---- jitted steps (state flows through args and return values)
  def _decode_step(self, x: Tensor, state: list[Tensor], extra: list[Tensor], start: Var):
    cache = self.cache.bind(state, extra)
    lg = self.forward(x, cache, start, 1)
    return lg.reshape(lg.shape[0], -1).float(), cache.state()
  def _prefill_step(self, x: Tensor, state: list[Tensor], extra: list[Tensor], start: Var, last: Var):
    """x: (B, bucket, D), real tokens first. Padding rows run too (they are causally after every real row, and the
    cache rows they write are rewritten before any later step reads them)."""
    cache = self.cache.bind(state, extra)
    lg = self.forward(x, cache, start, x.shape[1], last=last)
    return lg.reshape(lg.shape[0], -1).float(), cache.state()
  def _bucket(self, t: int) -> int: return min(self.prefill_chunk, max(16, 1 << (t - 1).bit_length()))

  def reset(self) -> None:
    with self._spmd("reset"):
      if hasattr(self.cache, "reset"): self.cache.reset()  # type: ignore[attr-defined]
      self.pos = 0

  def prefill(self, ids: Sequence[Sequence[int]] | np.ndarray, start: int | None = None, chunk: int | None = None) -> np.ndarray:
    """Run the prompt (B, T) through the model at positions [start, start+T) (default: continue after self.pos),
    filling the cache in chunks of `chunk` (<= prefill_chunk) tokens. Returns the last position's logits, (B, vocab)
    float32. Each chunk is padded to a power-of-two bucket (>= 16) and runs that bucket's jitted step."""
    ids = np.asarray(ids, np.int64)
    if ids.ndim == 1: ids = ids[None]
    start = self.pos if start is None else start
    with self._spmd("prefill", ids, start, chunk): return self._prefill(ids, start, chunk)
  def _steps(self) -> Any:
    if self._neuron_steps is None:
      from fusezap.serve.neuron_kv import NeuronSteps
      self._neuron_steps = NeuronSteps(self)
    return self._neuron_steps
  def _prefill(self, ids: np.ndarray, start: int, chunk: int | None) -> np.ndarray:
    chunk = min(chunk or self.prefill_chunk, self.prefill_chunk)
    B, n = ids.shape
    assert B == self.max_batch, f"batch {B} != the cache batch {self.max_batch}"
    assert start + n <= self.max_len, f"{start + n} tokens exceed max_len {self.max_len}"
    if getattr(self, "neuron", False):   # positions as data, one jitted forward per bucket (no symbolic start)
      for s in range(0, n, chunk):
        part = ids[:, s:s + chunk]
        t = part.shape[1]
        bk = 1 if t == 1 else self._bucket(t)
        lg = self._steps().run(np.pad(part, ((0, 0), (0, bk - t))), start + s, t)
      self.pos = start + n
      return lg
    lg = None
    for s in range(0, n, chunk):
      part = ids[:, s:s + chunk]
      t = part.shape[1]
      if t == 1:
        lg = self.decode(part, start + s)
        continue
      bk = self._bucket(t)
      if bk not in self._prefill_jits: self._prefill_jits[bk] = (Jit(self._prefill_step), Var(f"last{bk}", 0, bk - 1))
      jf, last_v = self._prefill_jits[bk]
      emb = self.embed(part)
      buf = np.zeros((B, bk, emb.shape[-1]), dtype=emb.dtype)
      buf[:, :t] = emb
      x = Tensor(buf, device=self.device, dtype=self.dtype)
      with self._opt_env(prefill=True):
        out, st = jf(x, self.cache.state(), self.cache.inputs(), bind(self.start_v, start + s), bind(last_v, t - 1))
      self.cache.load(st)
      lg = out
    self.pos = start + n
    return lg.numpy() if isinstance(lg, Tensor) else lg

  def decode(self, ids: Sequence[Sequence[int]] | np.ndarray, start: int | None = None) -> np.ndarray:
    """One token per sequence, (B, 1), at position `start` (default self.pos). Returns (B, vocab) float32 logits."""
    ids = np.asarray(ids, np.int64).reshape(self.max_batch, 1)
    start = self.pos if start is None else start
    with self._spmd("decode", ids, start): return self._decode(ids, start)
  def _decode(self, ids: np.ndarray, start: int) -> np.ndarray:
    assert start < self.max_len, f"position {start} >= max_len {self.max_len}"
    if getattr(self, "neuron", False):
      lg = self._steps().run(ids, start, 1)
      self.pos = start + 1
      return lg
    x = Tensor(self.embed(ids), device=self.device, dtype=self.dtype)
    with self._opt_env(): out, st = self._decode_jit(x, self.cache.state(), self.cache.inputs(), bind(self.start_v, start))
    self.cache.load(st)
    self.pos = start + 1
    return out.numpy()

  # ---- I81 D3: text encoder (hidden states, no cache, no lm_head)
  def _encode_step(self, x: Tensor, km: Tensor, layers: tuple[int, ...], cut_layers: bool = False) -> Tensor:
    hs: list[Tensor] = []
    top = self.forward(x, _NoCache(), 0, x.shape[1], key_mask=km, hiddens=hs, upto=max(layers), cut_layers=cut_layers)
    # hs[k]: HF hidden_states[k] (hs[0] the embeddings, hs[k] the residual after layer k-1; HF's last entry is normed)
    hs.append(self._add_norm(top, None, self.norm)[1] if max(layers) == self.ckpt_layers else top)
    return Tensor.stack([hs[k] for k in layers], dim=1).float().contiguous()
  def encode(self, ids: Sequence[Sequence[int]] | np.ndarray, mask: Sequence[Sequence[int]] | np.ndarray | None = None,
             layers: Sequence[int] = (9, 18, 27)) -> np.ndarray:
    """HF `model(ids, attention_mask=mask, output_hidden_states=True).hidden_states[k]` for k in layers, stacked:
    (B, len(layers), T, D) float32 (values in the model dtype). Positions are 0..T-1 for every row (right padding, like
    HF with a 0/1 mask); mask 0 hides that key from every query. Runs layers [0, max(layers)) only; nothing is cached.
    One fz.jit per (B, T, layers)."""
    ids = np.asarray(ids, np.int64)
    if ids.ndim == 1: ids = ids[None]
    B, T = ids.shape
    layers = tuple(int(k) for k in layers)
    assert layers and 0 <= min(layers) and 1 <= max(layers) <= self.cfg.n_layers, f"layers {layers} vs {self.cfg.n_layers} loaded"
    assert T <= self.table_len, f"{T} tokens exceed the RoPE table ({self.table_len})"
    m = np.ones((B, T), np.bool_) if mask is None else np.asarray(mask).astype(np.bool_).reshape(B, T)
    jits = self.__dict__.setdefault("_encode_jits", {})
    key = (B, T, layers)
    if key not in jits:   # jit_encode False (FZ_ENCODE_JIT=0): eager, realized layer by layer (a capture holds every
      # intermediate of the forward at once: the klein Qwen3-4B encoder in bf16 peaks +16 GB and keeps 9.5 GB that way)
      jits[key] = Jit(lambda x, km: self._encode_step(x, km, layers)) if getattr(self, "jit_encode", True) and getenv("FZ_ENCODE_JIT", 1) \
                  else (lambda x, km: self._encode_step(x, km, layers, cut_layers=True))
    x = Tensor(self.embed(ids), device=self.device, dtype=self.dtype)
    with self._opt_env(prefill=True): out = jits[key](x, Tensor(m, device=self.device))
    return out.numpy()

  def generate(self, ids: Sequence[int], max_new_tokens: int, sampler: Any = None, stop: Sequence[int] | None = None):
    """Yield new token ids for one prompt (batch 1). sampler: fusezap.serve.sampling.Sampler (default greedy)."""
    from fusezap.serve.sampling import Sampler
    sampler = sampler or Sampler()
    stop = set(self.cfg.eos_token_ids if stop is None else stop)
    self.reset()
    hist = list(ids)
    lg = self.prefill([hist])
    for _ in range(max_new_tokens):
      tok = int(sampler(lg[0], hist))
      yield tok
      hist.append(tok)
      if tok in stop or self.pos >= self.max_len: return
      lg = self.decode([[tok]])
