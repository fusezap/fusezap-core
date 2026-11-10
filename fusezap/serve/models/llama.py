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
  sliding_window: int | None = None    # only set when the checkpoint enables it (not implemented: see Llama.__init__)
  qk_norm: bool = False                # Qwen3: RMSNorm over head_dim on q and k (q_norm / k_norm weights), before RoPE
  @staticmethod
  def from_hf(c: dict) -> "LlamaConfig":
    assert c.get("hidden_act", "silu") == "silu", f"unsupported activation {c.get('hidden_act')}"
    eos = c.get("eos_token_id")
    rp = c.get("rope_parameters") or {}   # transformers >= 5 keeps rope_theta and the scaling dict here
    theta = c.get("rope_theta") or rp.get("rope_theta") or 10000.0
    scaling = c.get("rope_scaling") or ({k: v for k, v in rp.items() if k != "rope_theta"} if rp.get("rope_type", "default") != "default" else None)
    return LlamaConfig(c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["num_attention_heads"],
      c.get("num_key_value_heads") or c["num_attention_heads"], c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
      c["vocab_size"], c.get("rms_norm_eps", 1e-6), float(theta), scaling,
      bool(c.get("tie_word_embeddings", False)), c.get("max_position_embeddings", 4096),
      [eos] if isinstance(eos, int) else list(eos or []), c.get("bos_token_id"), c.get("model_type", "llama"),
      c.get("sliding_window") if c.get("use_sliding_window", c.get("model_type") == "mistral") else None,
      c.get("model_type") == "qwen3")

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
  elif kind not in (None, "default"): raise NotImplementedError(f"rope_scaling {rs}")
  return inv

def rope_tables(cfg: LlamaConfig, max_len: int) -> tuple[np.ndarray, np.ndarray]:
  """(max_len, head_dim/2) cos and sin, float32 (HF: freqs = pos * inv_freq in float32, then cos/sin)."""
  fr = np.arange(max_len, dtype=np.float32)[:, None] * rope_inv_freq(cfg)[None, :]
  return np.cos(fr).astype(np.float32), np.sin(fr).astype(np.float32)

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
  qn: Tensor | None = None; kn: Tensor | None = None   # Qwen3 q_norm / k_norm (head_dim,)

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
    if tp is not None and tp.world > 1:
      from fusezap.serve.dist import local_cfg, tp_slice
      cfg = local_cfg(cfg, tp.world)
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
    if c.sliding_window is not None and max_len > c.sliding_window:
      raise NotImplementedError(f"sliding-window attention (window {c.sliding_window}) is not implemented; use max_len <= window")
    if c.qk_norm and _isn(device or Device.DEFAULT):
      raise NotImplementedError("q/k RMSNorm (Qwen3) is not in the NEURON kernels yet (I81 D10)")
    def W(name: str) -> np.ndarray: return weights[name] if self.tp is None else tp_slice(name, weights[name], self.full_cfg, tp.rank, tp.world)
    def T(a: np.ndarray, host: bool = False) -> Tensor:   # host=True: `a` is already in dtype's host format
      return Tensor(a if host else to_dtype(a, dtype), device=dev, dtype=dtype).realize()
    def cat(names: list[str]) -> Tensor | None:          # fused projections: one matmul instead of two or three
      if not all(n in weights for n in names): return None
      return T(np.concatenate([to_dtype(W(n), dtype) for n in names], 0), host=True)
    def TK(names: list[str]) -> Tensor | None:           # I68 NEURON bf16: a linear weight (or a fused one) bf16, K-major
      if not all(n in weights for n in names): return None
      if not self.nbf16: return cat(names) if len(names) > 1 else T(W(names[0]))
      a = np.concatenate([to_dtype(W(n), dtypes.bfloat16) for n in names], 0)
      return Tensor(np.ascontiguousarray(a.T), device=dev, dtype=dtypes.bfloat16).realize()
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
      l.ln1, l.ln2 = T(W(p + "input_layernorm.weight")), T(W(p + "post_attention_layernorm.weight"))
      if c.qk_norm:   # (head_dim,): the same on every tensor-parallel rank
        l.qn, l.kn = T(weights[a + "q_norm.weight"]), T(weights[a + "k_norm.weight"])
      self.layers.append(l)
      release(lw(i - lag))
    for i in range(max(c.n_layers - lag, 0), c.n_layers): release(lw(i))
    self.norm = T(W("model.norm.weight"))
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
    self.cos = Tensor(to_dtype(cos, self.dtype), device=self.device, dtype=self.dtype).realize()
    self.sin = Tensor(to_dtype(sin, self.dtype), device=self.device, dtype=self.dtype).realize()
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
  def _rowpar(self, x: Tensor, w: Any, b: Tensor | None, redo: tuple[int, int] | None = None, fuse: bool = False) -> Tensor:
    """A row-parallel projection (o_proj, down_proj): self.linear on one device; with tensor parallelism this rank's
    partial product, all-reduced over the ranks, then the bias once. fuse (I75, NEURON TP prefill): the all-reduce as a
    collective the fused member kernels may hold (dist.all_reduce)."""
    if self.tp is None: return self.linear(x, w, b, redo)
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
    return self.cuda16 and isinstance(start, int) and isinstance(T, int) and isinstance(Sk, int) and self.cfg.head_dim in (64, 128)
  def _rmsnorm(self, x: Tensor, w: Tensor) -> Tensor:
    xf = x.float()
    y = xf * ((xf * xf).mean(-1, keepdim=True) + self.cfg.rms_eps).rsqrt()
    return w * y.cast(x.dtype)
  def _add_norm(self, x: Tensor, d: Tensor | None, w: Tensor) -> tuple[Tensor, Tensor]:
    """(x + d, _rmsnorm(x + d, w)), or (x, _rmsnorm(x, w)) when d is None. CUDA f16/bf16: one fused row kernel
    (fusezap.serve.cuda.rmsnorm) that also writes the residual stream; the composed reduce re-read every earlier layer's
    residual terms, since the stream was never realized."""
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
    return to_dtype(self.embed_host[np.asarray(ids, np.int64)], self.dtype)

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
    cos = self.cos.shrink(((start, S), None)).reshape(1, 1, T, hd // 2)
    sin = self.sin.shrink(((start, S), None)).reshape(1, 1, T, hd // 2)
    qpos = self._pos.shrink(((start, S),)).reshape(T, 1)
    scale = 1.0 / math.sqrt(hd)
    mask: Tensor | None = None
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
      if l.qn is not None: q, k = self._rmsnorm(q, l.qn), self._rmsnorm(k, l.kn)   # Qwen3: per head, before RoPE
      q, k = self._rope(q.transpose(1, 2), cos, sin), self._rope(k.transpose(1, 2), cos, sin)
      K, V = cache.update(i, k, v, start, T)                                  # (B, Hkv, Sk, hd), row j = position j
      Sk = K.shape[2]
      if key_mask is None and self._flash(start, T, Sk):   # CUDA flash attention: key j visible to query row l iff j <= start + l
        from fusezap.device.cuda_kernels.attention import attention as flash
        o = flash(q, K, V, causal=True, scale=scale, coff=start).transpose(1, 2).reshape(B, T, Hq * hd)
      else:
        if i == 0 or mask is None:
          mask = self._pos.shrink(((0, Sk),)).reshape(1, Sk) <= qpos   # key position <= query position
          if key_mask is not None: mask = mask.reshape(1, 1, 1, T, Sk) * key_mask.reshape(B, 1, 1, 1, Sk)
        s = (q.reshape(B, Hkv, G, T, hd) @ K.reshape(B, Hkv, 1, Sk, hd).transpose(-1, -2)) * scale
        s = mask.where(s, float("-inf"))
        o = s.softmax(-1) @ V.reshape(B, Hkv, 1, Sk, hd)                        # (B, Hkv, G, T, hd)
        o = o.reshape(B, Hq, T, hd).transpose(1, 2).reshape(B, T, Hq * hd)
      x, h = self._add_norm(x, self._rowpar(o, l.wo, l.bo), l.ln2)
      gu = self.linear(h, l.wgu, l.bgu)
      I = c.intermediate_size
      g, u = gu.shrink((None, None, (0, I))), gu.shrink((None, None, (I, 2 * I)))
      d = self._rowpar(g.silu() * u, l.wd, l.bd)
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
    def plain_qkv(w: Any) -> bool: return (fuse or kvw) and getattr(w, "apply", None) is None and not c.qk_norm
    cos2, sin2 = cos, sin
    if not getattr(cache, "rope_in_attend", False): cos, sin = cos.reshape(N, 1, h2), sin.reshape(N, 1, h2)   # NEURON: (N, hd) rows
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
    lv2 = nfence and getattr(self, "kmajor", False) and hasattr(cache, "layer") and \
          (self.tp is None or os.environ.get("FZ_NEURON_TP_LAYER", "halves") != "members")
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
      if _pf.enabled(): pfk = _pf
      if pfk is not None: pk = _pf.big_fuse(int(N), pk)   # a 512-row chunk: groups of FZ_NEURON_PFK_FUSE_BIG layers per NEFF
      if pfk is not None and self.tp is not None:
        from fusezap.serve.dist import all_reduce as _ar2
        arf = lambda t: _ar2(t, self.tp, fuse=bool(pk))
    for i, l in enumerate(self.layers):
      if nfence and i: d = fence(d, i, self.device, every=pk)
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
        o = cache.attend_rope(i, self.linear(h, l.wqkv, l.bqkv, rd), cos2, sin2, Hq, 1.0 / math.sqrt(hd))
      elif plain_qkv(l.wqkv):
        dest = cache.kv_dest(i) if kvw else None
        qkv = self.qkv_rope_fused(h, l, cos2, sin2, rd, dest)
        if dest is None: o = cache.attend(i, qkv, Hq, 1.0 / math.sqrt(hd))
      else:
        qkv = self.linear(h, l.wqkv, l.bqkv, rd).reshape(N, Hq + 2 * Hkv, hd)
        qk, v = qkv.shrink((None, (0, Hq + Hkv), None)), qkv.shrink((None, (Hq + Hkv, Hq + 2 * Hkv), None))
        if l.qn is not None:   # Qwen3: q_norm on the Hq query heads, k_norm on the Hkv key heads, before RoPE
          qk = self._rmsnorm(qk.shrink((None, (0, Hq), None)), l.qn).cat(self._rmsnorm(qk.shrink((None, (Hq, Hq + Hkv), None)), l.kn), axis=1)
        x1, x2 = qk.shrink((None, None, (0, h2))), qk.shrink((None, None, (h2, hd)))
        qkv = (x1 * cos + (-x2) * sin).cat(x2 * cos + x1 * sin, axis=-1).cat(v, axis=1)   # the same RoPE expression as _rope
        o = cache.attend(i, qkv, Hq, 1.0 / math.sqrt(hd))
      if dest is not None: o = cache.attend(i, qkv, Hq, 1.0 / math.sqrt(hd), written=True)
      if fz and l.bo is None: x, h = self._ar_add_norm(self.linear(o, l.wo, None, rd), x, l.ln2)
      else: x, h = self._add_norm(x, self._rowpar(o, l.wo, l.bo, rd, fuse=bool(pk)), l.ln2)
      gu = self.linear(h, l.wgu, l.bgu, rd) if wide_linear is None else wl(h, l.wgu, l.bgu)
      I = c.intermediate_size
      if plain(l.wgu) and l.bgu is None and _cu.silu_mul_ok(gu): a = _cu.silu_mul(gu)
      else: a = gu.shrink((None, (0, I))).silu() * gu.shrink((None, (I, 2 * I)))
      if fz and l.bd is None and i + 1 < len(self.layers): p, d = self.linear(a, l.wd, None, rd), None
      else: p, d = None, self._rowpar(a, l.wd, l.bd, rd, fuse=bool(pk))
    if nfence: d = fence(d, len(self.layers), self.device, hard=True, every=pk)
    if getattr(self, "neuron", False) and not select: return self._vocab(wl(self._add_norm(x, d, self.norm)[1], self.lm_head, None))   # one norm kernel
    if d is not None: x = x + d
    if select: x = cache.select(x)
    if pfk is not None and wide_linear is None and not getattr(self, "tp_vocab", False) and self.lm_head_t is None and \
       self.lm_head.dtype == dtypes.bfloat16 and pfk.head_mode(self.tp is not None) != "off":   # I77: the prefill's head (FZ_NEURON_PFK_HEAD)
      return pfk.head(self, self._add_norm(x, None, self.norm)[1], arf)
    return self._vocab(wl(self._add_norm(x, None, self.norm)[1], self.lm_head, None))
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
