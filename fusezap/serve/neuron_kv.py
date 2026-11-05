"""Serving on NEURON (N5, docs/NKI_BACKEND.md §3 and §4 exception 8): the paged KV cache's device side without indirect DMA.

LLMEngine keeps its scheduler, block allocator and block tables (PagedKVCache's bookkeeping, unchanged). What differs is
how a step touches the pool, because an NKI kernel has no runtime integers (exception 2) and this backend uses no indirect
DMA (unverified on inf2): every kernel below has static shapes, so each forward (decode bucket, prefill size) compiles to
a fixed set of NEFFs that every step reuses, whatever the positions, slots and block tables are.

Pool: per layer one (P, 2*C) float32 tensor, P = num_blocks * block_size slots, C = n_kv_heads * head_dim; slot row r
holds that token's k (C values) then v (C values). Per step the host appends to the int32 metadata (StepLayout's arrays):
  lo[t], hi[t] (N each)   token t sees the keys whose id is in [lo, hi]: its block-table row * KEY_STRIDE + [0, position]
                          (padding rows: an empty range, so they attend to nothing and give zeros);
  kid[p] (P)              each slot's key id: block-table row of its sequence * KEY_STRIDE + position, -1 if no sequence of
                          this step owns it. KEY_STRIDE = max_blocks * block_size (> any position).
Kernels of one layer (all composed by the scheduler and lowered by the NKI renderer, except the two custom ones):
  qkv = h @ Wqkv + b (fz_matmul), then one kernel: RoPE without `cat` (x * [cos, cos] + x.flip(half) * [-sin, sin], the same
  products and sums as Llama._rope, so the same bits) into X = [q | k | v | 1] (N, (Hq + 2*Hkv + 1)*hd);
  D = W @ X[:, q:] (fz_matmul on TensorE), W (P, N) the step's one-hot slot matrix (W[p, t] = 1 iff slot[t] == p, built once
  per step): row p of D is the k, v of the token written to slot p and a 1 in the last block, or all zeros;
  pool = where(D[:, 2C] > 0.5, D[:, :2C], pool), stored into the pool buffer itself (the kernel's output buffer is the pool:
  each element is read before it is written, in the same tile; store_into);
  o = attention (nki_kernels.render_attention, mask "keyrange", rows = (token, q head of the group) with two row strides),
  straight from X's q columns and the pool, written as (N, Hq*hd) for o_proj.
The one-hot products are exact: every output element is one finite value times 1 plus zeros (TensorE accumulates float32).
N7 step 3 (FZ_NEURON_LAYER=v2, default): a decode step's layer is one nki_kernels.LayerSpec kernel instead (NeuronKVCache.layer:
the same instructions, intermediates in SBUF; steps whose rows do not fit it, e.g. prefill buckets, keep the kernels above).
select(x) is a one-hot (R, N) matmul too. Costs grow with P (the KV write reads and writes the whole pool, attention scans
every slot), which is why `fusezap serve --device NEURON` wants a small --num-blocks (scripts/nki_n5_inf2.sh: 256 x 16).
"""
from __future__ import annotations
from typing import Any, Sequence
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import DType, dtypes
from fusezap.device import Buffer, Device
from fusezap.ir import LazyOp, MemBuffer, ScheduleItem
from fusezap.ops import BufferOps, LoadOps
from fusezap.lazy import LazyBuffer
from fusezap.serve.kv_cache import BlockAllocator, PagedKVCache, StepLayout

NEURON_TIERS = ("NEURON", "NKISHIM", "NKISIM")
FUSED_KV_WRITE = bool(int(__import__("os").environ.get("FZ_NEURON_KVW", "1") or 1))   # 0: the composed one-hot matmul + WHERE

def is_neuron(device: str) -> bool: return Device.canonicalize(device).split(":")[0].upper() in NEURON_TIERS

def _buf(t: Tensor) -> Buffer:
  """The buffer holding t's elements in row-major order; a reshape of a computed base is that base (no copy kernel)."""
  lb = t.lazydata
  if lb.base is not lb and lb.base.op is not LoadOps.CONST and len(lb.st.views) == 1 and lb.st.contiguous and \
     all(isinstance(x, int) for x in (*lb.shape, *lb.base.shape)) and int(np.prod(lb.shape)) == int(np.prod(lb.base.shape)):
    if lb.base.realized is None:
      from fusezap.realize import realize
      realize([lb.base])
    return lb.base.realized
  return t._realized_contiguous().realized
def _wrap(buf: Buffer, shape: tuple[int, ...]) -> Tensor: return Tensor(LazyBuffer.from_buffer(buf, shape))

def store_into(dst: Buffer, value: Tensor) -> None:
  """Compute `value` (dst's size and dtype) with its last kernel writing dst itself: the in-place pool update. Every
  earlier kernel runs (and is recorded by an fz.jit capture) as usual; the last item's output buffer is swapped for dst."""
  from fusezap import jit as J
  from fusezap.schedule import create_schedule
  lb = value.contiguous().lazydata
  assert lb.base is lb and lb.size == dst.size and lb.dtype == dst.dtype, f"store_into: {lb} into {dst}"
  sched = create_schedule([lb], J.current_var_vals())
  assert sched and sched[-1].kind == "kernel" and sched[-1].bufs[0] is lb.realized, "store_into: the value must end in a kernel"
  for si in sched[:-1]:
    for b in si.bufs: b.ensure_allocated()
    J.run_item(si)
  last = sched[-1]
  # the kernel reads dst through its own output argument (data0, which every NKI kernel takes as a must-alias input), so dst
  # is bound once: an input k that is dst becomes index 0 and the later inputs shift down
  keep = [i for i, b in enumerate(last.bufs) if i == 0 or b is not dst]
  remap = {old: (0 if last.bufs[old] is dst else keep.index(old)) for old in range(len(last.bufs))}
  def re(n: LazyOp) -> LazyOp:
    arg = n.arg
    if n.op is BufferOps.LOAD: arg = MemBuffer(remap[arg.idx], arg.dtype, arg.st)
    return LazyOp(n.op, tuple(re(x) for x in n.src), arg)
  ast = tuple(re(r) for r in last.ast)
  J.run_item(ScheduleItem(ast, (dst,) + tuple(last.bufs[i] for i in keep[1:]), last.var_vals, last.kind, last.n_outputs))

class NeuronKVCache(PagedKVCache):
  rope_in_attend = True   # Llama.forward_tokens hands attend_rope the qkv projection and the RoPE rows (no cat)

  def __init__(self, n_layers: int, n_kv: int, head_dim: int, num_blocks: int, block_size: int, dtype: DType, device: str | None = None,
               max_seq_len: int = 2048, _shared: "NeuronKVCache | None" = None, _state: list[Tensor] | None = None,
               _inputs: list[Tensor] | None = None):
    assert dtype == dtypes.float32, f"NEURON serving is float32 here (bf16 is later), got {dtype}"
    self.n_layers, self.n_kv, self.hd, self.num_blocks, self.bs, self.dtype = n_layers, n_kv, head_dim, num_blocks, block_size, dtype
    self.device = Device.canonicalize(device)
    self.max_seq_len, self.max_blocks = max_seq_len, -(-max_seq_len // block_size)
    self.P, self.C, self.key_stride = num_blocks * block_size, n_kv * head_dim, self.max_blocks * block_size
    assert self.P < 2 ** 24, "slot ids must stay exact in float32"
    if _shared is None:
      # lowest block ids first (FZ_NEURON_LOW_BLOCKS, default on; 0: LIFO as before): a step's pool prefix stays as small as
      # its sequences' blocks allow (N7 serve: after Validate's warm-up a lone request ran at 1024 slots instead of 512)
      low = __import__("os").environ.get("FZ_NEURON_LOW_BLOCKS", "1") not in ("", "0")
      self.allocator, self.tables, self._rows = BlockAllocator(num_blocks, lowest_first=low), {}, ()
      self.pools = [Tensor.zeros(self.P, 2 * self.C, dtype=dtype, device=self.device).contiguous().realize() for _ in range(n_layers)]
      self._inputs: list[Tensor] = []
      self.iota = Tensor(np.arange(self.P, dtype=np.int32), device=self.device).realize()   # slot ids, a device constant
      self._const: dict = {}
    else:
      self.allocator, self.tables, self._rows = _shared.allocator, _shared.tables, _shared._rows
      self.pools, self._inputs = list(_state or _shared.pools), list(_inputs if _inputs is not None else _shared._inputs)
      self.iota, self._const = _shared.iota, _shared._const
    self.layout: StepLayout | None = _shared.layout if _shared is not None else None
    self._step: dict = {}   # per traced step: the one-hot matrices and RoPE tables, built at layer 0
    # N6: the pool prefix a step's kernels touch (KV write rows, attention keys): every slot of the step's sequences is
    # below it (meta_ctx; the allocator hands out low block ids first). None: the whole pool (P).
    self.ctx: int | None = _shared.ctx if _shared is not None else None

  @staticmethod
  def for_model(model: Any, num_blocks: int, block_size: int = 16, max_seq_len: int | None = None) -> "NeuronKVCache":
    c = model.cfg
    return NeuronKVCache(c.n_layers, c.n_kv_heads, c.head_dim, num_blocks, block_size, model.dtype, model.device, max_seq_len or model.max_len)

  def bind(self, state: list[Tensor], inputs: list[Tensor] | None = None) -> "NeuronKVCache":
    return NeuronKVCache(self.n_layers, self.n_kv, self.hd, self.num_blocks, self.bs, self.dtype, self.device, self.max_seq_len,
                         _shared=self, _state=state, _inputs=inputs)
  def meta_ctx(self, meta: np.ndarray, N: int, R: int, prefill: bool = False) -> int:
    """The smallest of ctx_buckets(prefill) that holds every block in the step's block tables (meta: step_meta's array)."""
    L = StepLayout(N, R, self.max_blocks)
    bt = np.asarray(meta[L.bt:L.bt + R * L.MB])
    top = (int(bt.max()) + 1) * self.bs if bt.size and bt.max() >= 0 else 1
    self.last_top = top   # FZ_SERVE_PROFILE: the slots the step needs
    return next(c for c in self.ctx_buckets(prefill) if c >= top)
  PREFILL_CTX_TP = "128,256,512"   # I75: the prefill's pool prefixes under TP when FZ_NEURON_PREFILL_CTX is unset
  def ctx_buckets(self, prefill: bool = False) -> list[int]:
    """N6: the pool prefixes (slots) a step can run at, ascending: powers of two from FZ_NEURON_CTX_MIN (default 512) up to
    the pool, then the pool itself; FZ_NEURON_CTX=0: the pool only (N5: every step reads and writes all P slots).
    N7 step 3: FZ_NEURON_CTX=mid adds 3/2 of each power of two (768, 1536, ...: served c4 with 128-token prompts and
    outputs needs <= 768 slots in half its steps, 1024 in the rest); FZ_NEURON_CTX=512,768,1024 (a list) uses those
    (below the pool) and the pool. A prefix only changes how many padded key slots are scanned, and attention's last key
    block is narrower at 768 than at 1024, so its row sums are not guaranteed to round like 1024's: off by default.
    I75: prefill forwards (prefill=True) read FZ_NEURON_PREFILL_CTX instead when it is set (the same forms), so a job can
    keep decode at the whole pool (FZ_NEURON_CTX=0: one decode NEFF) while a prompt's attention scans only its prefix
    (7B TP=2 at 128 rows: attention is ~40% of a prefill layer, linear in the slots scanned). Under tensor parallelism
    (neuron.TP_WORLD > 1) an unset FZ_NEURON_PREFILL_CTX defaults to PREFILL_CTX_TP (128,256,512: the I75 inf2 run's choice),
    clamped to the pool (prefixes at or past it drop; the pool is always the last bucket); TP=1 follows FZ_NEURON_CTX."""
    import os
    from fusezap.device.neuron import TP_WORLD
    mode = os.environ.get("FZ_NEURON_CTX", "1").strip()
    if prefill:
      pm = os.environ.get("FZ_NEURON_PREFILL_CTX")
      if pm is None and TP_WORLD[0] > 1: pm = self.PREFILL_CTX_TP   # FZ_NEURON_PREFILL_CTX="": follow FZ_NEURON_CTX
      if pm is not None and pm.strip(): mode = pm.strip()
    if mode in ("", "0"): return [self.P]
    if "," in mode or (mode.isdigit() and int(mode) > 1):
      return sorted({int(x) for x in mode.split(",") if x.strip() and self.bs <= int(x) < self.P}) + [self.P]
    b, out = max(int(os.environ.get("FZ_NEURON_CTX_MIN", "") or 512), self.bs), []
    while b < self.P:
      out.append(b)
      if mode == "mid" and b * 3 // 2 < self.P and b * 3 // 2 % self.bs == 0: out.append(b * 3 // 2)
      b *= 2
    return out + [self.P]

  # ---- metadata: StepLayout's arrays, then lo (N), hi (N), kid (P)
  def _extend(self, m: np.ndarray, N: int, R: int) -> np.ndarray:
    assert R * self.key_stride < 2 ** 24, f"{R} rows x {self.key_stride} positions: key ids must stay exact in float32"
    L = StepLayout(N, R, self.max_blocks)
    row, pos = m[L.row:L.row + N].astype(np.int64), m[L.pos:L.pos + N].astype(np.int64)
    real = row >= 0
    lo = np.where(real, row * self.key_stride, 1)          # padding: lo > hi, nothing visible
    hi = np.where(real, row * self.key_stride + pos, 0)
    kid = np.full(self.P, -1, np.int64)
    bt = m[L.bt:L.bt + R * L.MB].reshape(R, L.MB).astype(np.int64)
    r_, j_ = np.nonzero(bt >= 0)
    if len(r_):
      blk = bt[r_, j_]
      rows = (blk[:, None] * self.bs + np.arange(self.bs)[None, :]).reshape(-1)
      kid[rows] = (r_[:, None] * self.key_stride + j_[:, None] * self.bs + np.arange(self.bs)[None, :]).reshape(-1)
    return np.concatenate([m, lo, hi, kid]).astype(np.int32)
  def step_meta(self, N: int, R: int, tokens: Sequence[tuple[int, int]], sel: Sequence[int]) -> np.ndarray:
    return self._extend(super().step_meta(N, R, tokens, sel), N, R)
  def step_meta_rows(self, N: int, R: int, sid: np.ndarray, pos: np.ndarray, sel: Sequence[int]) -> np.ndarray:
    return self._extend(super().step_meta_rows(N, R, sid, pos, sel), N, R)

  # ---- the step
  def _ints(self, off: int, n: int) -> Tensor:
    return self._inputs[0].shrink(((off, off + n),))
  def _onehot(self, idx: Tensor, n: int) -> Tensor:
    """(len(idx), n) float32: [i, j] = 1 iff idx[i] == j (n <= P: the first n slot ids)."""
    return (idx.reshape(-1, 1) == self.iota.shrink(((0, n),)).reshape(1, n)).cast(dtypes.float32)
  def _tables(self, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """(N, 1, 2, hd/2) RoPE factors for x (N, H, 2, hd/2): [cos, cos] and [-sin, sin]. LLMEngine uploads them in that form
    (rope_rows: (N, hd) rows of the doubled tables), so they are views; plain (N, hd/2) rows are doubled here once per step."""
    N, h2 = cos.shape[0], self.hd // 2
    if cos.shape[-1] == self.hd: return cos.reshape(N, 1, 2, h2), sin.reshape(N, 1, 2, h2)
    if "cs" not in self._step:
      c, s = cos.reshape(N, 1, h2), sin.reshape(N, 1, h2)
      self._step["cs"] = (c.cat(c, axis=1).reshape(N, 1, 2, h2).contiguous().realize(),
                          (-s).cat(s, axis=1).reshape(N, 1, 2, h2).contiguous().realize())
    return self._step["cs"]
  def attend_rope(self, layer: int, qkv: Tensor, cos: Tensor, sin: Tensor, n_heads: int, scale: float) -> Tensor:
    """qkv (N, (Hq + 2 Hkv) * hd) = the step's qkv projection (bias added, not rotated); cos / sin (N, hd/2) the RoPE rows.
    Writes the rotated k and v into the tokens' slots, returns attention (N, Hq*hd)."""
    from fusezap.codegen.nki_kernels import AttnSpec, KVWriteSpec
    from fusezap.device.neuron import NeuronCall
    from fusezap import jit as J
    L, Hkv, hd, C, P = self.layout, self.n_kv, self.hd, self.C, self.ctx or self.P   # P: the pool prefix this step touches
    assert L is not None and self._inputs, "set the step layout first"
    N, Hq, h2, G = qkv.shape[0], n_heads, self.hd // 2, n_heads // self.n_kv
    COS, SIN = self._tables(cos, sin)
    y = qkv.reshape(N, Hq + 2 * Hkv, 2, h2)
    rot = lambda x: x * COS + x.flip(2) * SIN      # first half x1*cos + x2*(-sin), second x2*cos + x1*sin: Llama._rope's ops
    ones = Tensor.ones(N, 1, 2, h2, device=self.device, dtype=dtypes.float32)
    # one kernel: X = [rotated q | rotated k | v | 1] (N, Hq + 2 Hkv + 1, 2, hd/2), 4-D so every read is one view
    x = rot(y.shrink((None, (0, Hq + Hkv), None, None))).cat(y.shrink((None, (Hq + Hkv, Hq + 2 * Hkv), None, None)), ones, axis=1).contiguous()
    Wx = (Hq + 2 * Hkv + 1) * hd
    if "w" not in self._step: self._step["w"] = self._onehot(self._ints(L.slot, N), P).contiguous().realize()   # (N, P): WT[t, p]
    pool = self.pools[layer]
    qb, pb, mb, wb = _buf(x), _buf(pool), _buf(self._inputs[0]), _buf(self._step["w"])
    if FUSED_KV_WRITE: J.run_item(ScheduleItem((NeuronCall(KVWriteSpec(P, N, 2 * C, Wx, Hq * hd)),), (pb, wb, qb), {}, "custom", 1))
    else:   # the composed form: a one-hot matmul, then a WHERE stored into the pool buffer itself
      d = self._step["w"].transpose(0, 1) @ x.reshape(N, Wx).shrink((None, (Hq * hd, Wx)))                                     # (P, 2C + hd)
      store_into(pb, (d.shrink((None, (2 * C, 2 * C + 1))) > 0.5).where(d.shrink((None, (0, 2 * C))), pool))
    out = Buffer(qb.device, N * Hq * hd, dtypes.float32)
    base = L.size
    sp = AttnSpec((Hkv, 1), (N, G), P, hd, hd, (G * hd, 0, hd, 1, 0), (hd, 0, 2 * C, 1, 0), (hd, 0, 2 * C, 1, C), (G * hd, 0, hd, 1, 0),
                  float(scale), "keyrange", (1, 0, base, base + N, base + 2 * N), rs=(Wx, Hq * hd), kv_one=True)
    J.run_item(ScheduleItem((NeuronCall(sp),), (out, qb, pb, mb), {}, "custom", 1))
    return _wrap(out, (N, Hq * hd))
  def layer(self, i: int, x: Tensor, d: Tensor | None, l: Any, cos: Tensor, sin: Tensor, n_heads: int, scale: float, eps: float,
            all_reduce: Any = None, tp: Any = None) -> tuple[Tensor, Tensor] | None:
    """N8 TP: with all_reduce (a tensor-parallel shard's sum over the ranks) the layer runs as LayerSpec's two halves
    (part attn, then part mlp) with that all-reduce after each; the result is (x2, the all-reduced down product).
    N8 I72 follow-up: FZ_NEURON_TP_LAYER=fused and the group (tp) on the NKI collective: one LayerSpec(part="tp") kernel with
    both all-reduces inside it (neuron_tp.fused_layer), so no host step sits between a layer's halves."""
    """N7 step 3 (FZ_NEURON_LAYER=v2, nki_kernels.LayerSpec): decoder layer i as one launch, (x + d + o_proj(attention),
    down(silu(gate) * up)) with the members' bits; l: the layer's K-major weights. None when the step (rows) or the layer
    (gate/up/down biases) is outside LayerSpec: the caller runs the members (v1)."""
    from fusezap.codegen.nki_kernels import LayerSpec, layer_mode, layer_ok, layer_variant
    from fusezap.device.neuron import NeuronCall
    from fusezap import jit as J
    Lo, N, D = self.layout, x.shape[0], x.shape[-1]
    if layer_mode() != "v2" or Lo is None or not self._inputs or not isinstance(N, int) or not isinstance(D, int) or x.dtype != dtypes.float32: return None
    v = layer_variant()
    if v["heads"] == "stack" and not layer_ok(N, n_heads, self.n_kv, self.hd, "stack"): v["heads"] = "each"   # N8: stack (the default)
    # needs every kv head's rows padded to 32 in one tile (Qwen2.5-0.5B: <= 9 rows); "each" covers up to 128 / group rows
    if not layer_ok(N, n_heads, self.n_kv, self.hd, v["heads"]) or l.bgu is not None or l.bd is not None: return None
    P, Hq = self.ctx or self.P, n_heads
    COS, SIN = self._tables(cos, sin)
    if "w" not in self._step: self._step["w"] = self._onehot(self._ints(Lo.slot, N), P).contiguous().realize()
    sp = LayerSpec(N, D, Hq, self.n_kv, self.hd, int(l.wd.shape[0]), P, float(eps), float(scale), Lo.size, d is not None,
                   l.bqkv is not None, l.bo is not None, v["heads"], v["rope"], v["mm"], v["early"], v["pf"])
    from fusezap.codegen.nki_bf16 import LayerBF16, bf16_attn, layer_fits
    # N8 (I71): the tiled layer when asked (FZ_NEURON_LAYER_TILE=1) or when the untiled rows do not fit (auto: Qwen2.5-7B);
    # it takes mm=mv and rope=slice (the defaults); otherwise, or when it does not fit either, the members (I69's path)
    if (v["tile"] == "1" or (v["tile"] == "auto" and not layer_fits(sp))) and v["mm"] == "mv" and v["rope"] == "slice" and self.hd and 512 % self.hd == 0:
      import dataclasses
      sp = dataclasses.replace(sp, tile=True, wcols=v["wcols"])
      while not layer_fits(sp) and sp.wcols >= 1024:   # I13: narrower weight DMAs before the members (the same bits at any width;
        sp = dataclasses.replace(sp, wcols=sp.wcols // 1024 * 512)   # Llama-3.1-8B's TP=2 shard at 1024 slots: 2048 -> 1024)
    if not layer_fits(sp): return None   # I69: rows too wide for SBUF (7B untiled, or 7B tiled at a prefix past ~1024 slots): the members
    if l.wqkv.dtype == dtypes.bfloat16 and v["mm"] != "mv": return None   # I68: mm=ws has no bf16 form: the members
    if l.wqkv.dtype == dtypes.bfloat16:   # I68: bf16 weight matmuls; N8 (I70): + FZ_NEURON_BF16_WN / _PF / _PFQO (layer_bf16w:
      from fusezap.codegen.nki_bf16_n8 import layer_bf16w   # a whole untiled layer only; tiled layers / TP halves unchanged)
      bf = lambda s: layer_bf16w(LayerBF16(s, bf16_attn()))
    else: bf = lambda s: s
    dev = self.device
    src = {"x": x, "d": d, "ln1": l.ln1, "wqkv": l.wqkv, "bqkv": l.bqkv, "cos": COS, "sin": SIN, "wt": self._step["w"], "meta": self._inputs[0],
           "wo": l.wo, "bo": l.bo, "ln2": l.ln2, "wgu": l.wgu, "wd": l.wd}
    def run(spec: Any, out: dict) -> None:
      ls = spec
      while not hasattr(ls, "outs"): ls = ls.base   # LayerBF16W -> LayerBF16 -> LayerSpec
      J.run_item(ScheduleItem((NeuronCall(spec),), tuple(out[a] if a in out else _buf(src[a]) for a in spec.args()), {}, "custom", len(ls.outs())))
    scratch = lambda: {"pool": _buf(self.pools[i]), "xs": Buffer(dev, N * (Hq + 2 * self.n_kv + 1) * self.hd, dtypes.float32),
                       "os": Buffer(dev, N * Hq * self.hd, dtypes.float32)}
    if all_reduce is None:
      out = {"xo": Buffer(dev, N * D, dtypes.float32), "dn": Buffer(dev, N * D, dtypes.float32), **scratch()}
      run(bf(sp), out)
      return _wrap(out["xo"], (N, D)), _wrap(out["dn"], (N, D))
    import dataclasses
    from fusezap.serve.neuron_tp import fused_layer
    if (W := fused_layer(tp)):   # N8 I72 follow-up: attn | all-reduce | mlp | all-reduce as one kernel (the halves' bits)
      out = {"xo": Buffer(dev, N * D, dtypes.float32), "dn": Buffer(dev, N * D, dtypes.float32), **scratch()}
      run(bf(dataclasses.replace(sp, part="tp", tpw=W)), out)
      return _wrap(out["xo"], (N, D)), _wrap(out["dn"], (N, D))
    # N8 TP (I72): the shard's layer as its two halves around the two all-reduces: attention .. o_proj's product (po), its
    # all-reduce; the residual add (+ bo) .. down's product, its all-reduce (the members' cut points, so the members' bits)
    a = {"xo": Buffer(dev, N * D, dtypes.float32), "po": Buffer(dev, N * D, dtypes.float32), **scratch()}
    run(bf(dataclasses.replace(sp, part="attn")), a)
    x1, o = _wrap(a["xo"], (N, D)), all_reduce(_wrap(a["po"], (N, D)))
    src.update(x=x1, o=o)
    m = {"xo": Buffer(dev, N * D, dtypes.float32), "dn": Buffer(dev, N * D, dtypes.float32)}
    run(bf(dataclasses.replace(sp, part="mlp")), m)
    return _wrap(m["xo"], (N, D)), all_reduce(_wrap(m["dn"], (N, D)))
  def attend(self, layer: int, qkv: Tensor, n_heads: int, scale: float, written: bool = False) -> Tensor:
    raise NotImplementedError("NeuronKVCache: Llama.forward_tokens calls attend_rope (rope_in_attend)")
  def select(self, x: Tensor) -> Tensor:
    """The step's logits rows of x (N, D) -> (R, D): a one-hot (R, N) matmul (exact), zeros for unused rows."""
    L = self.layout
    assert L is not None
    if "sel" not in self._step: self._step["sel"] = self._onehot(self._ints(L.sel, L.R), x.shape[0]).contiguous().realize()
    return self._step["sel"] @ x
  def update(self, layer: int, k: Tensor, v: Tensor, start: Any, T: Any) -> tuple[Tensor, Tensor]:
    raise NotImplementedError("NeuronKVCache serves LLMEngine's ragged steps; Llama.prefill/decode use the model's contiguous cache")

def jit_device_bytes(j: Any) -> int:
  """Device bytes a captured fz.jit keeps between replays: its planned arenas plus its unplanned persistent buffers (0 if not
  captured yet). Outputs are fresh per call and not counted."""
  c = getattr(j, "captured", None)
  if c is None: return 0
  arenas = list(getattr(c, "arenas", []) or [])
  ah = {id(a._handle) for a in arenas if a._handle is not None}
  return sum(a.nbytes for a in arenas) + sum(b.nbytes for b in c.persist if b is not None and b._handle is not None and id(b._handle) not in ah)

def hbm_report(eng: Any) -> dict:
  """N8 (I71): what holds a NEURON engine's device memory, in bytes. kv: the pools; forwards: each captured decode / prefill
  forward (all its pool prefixes); scorer: the logprobs pass's captured buckets and head chunks; cached: the allocator's
  free list; weights: the rest of what this process holds (weights, the RoPE / slot constants, fence and scratch buffers);
  neffs: NEFF files loaded (their code and DMA rings sit in HBM next to the tensors, outside the budget). The runtime worker's
  own count of the bytes it holds is `worker_bytes` (it should equal held)."""
  from fusezap.device import Device
  al = Device[eng.model.device].allocator
  fw: dict[str, int] = {}
  for kind, d in (("decode", getattr(eng, "_decodes", {})), ("prefill", getattr(eng, "_prefills", {}))):
    for n, f in d.items(): fw[f"{kind}_{n}"] = sum(jit_device_bytes(j) for j in [f.jit, *f.ctx_jits.values()])
  sc = getattr(eng, "_fz_scorer", None)
  scorer = sum(jit_device_bytes(j) for j in [*getattr(sc, "jits", {}).values(), *getattr(sc, "head_jits", {}).values()]) if sc is not None else 0
  kv = sum(int(np.prod(p.shape)) * p.dtype.itemsize for p in getattr(eng.cache, "pools", []))
  held, cached = int(getattr(al, "held", 0)), int(getattr(al, "cached", 0))
  out = {"held": held, "cached": cached, "peak": int(getattr(al, "peak", held)), "budget": int(getattr(al, "budget", 0)), "kv": kv,
         "forwards": fw, "scorer": scorer, "weights": held - cached - kv - sum(fw.values()) - scorer,
         "neffs": int(getattr(al, "neffs", 0)), "neff_bytes": int(getattr(al, "neff_bytes", 0)), "ooms": int(getattr(al, "ooms", 0))}
  if hasattr(al, "worker") and hasattr(al, "flush"):
    try: out["worker_bytes"] = int(al.call({"op": "stats"}).get("bytes", -1))
    except Exception: pass
  return out

def hbm_line(eng: Any, when: str = "") -> str:
  """hbm_report as one log line (GiB)."""
  r, g = hbm_report(eng), lambda b: f"{b / 2**30:.3f}"
  fw = ", ".join(f"{k} {g(v)}" for k, v in r["forwards"].items())
  return (f"fusezap: NEURON HBM{' ' + when if when else ''}: held {g(r['held'])} GiB of budget {g(r['budget'])} (peak {g(r['peak'])}): "
          f"weights+constants {g(r['weights'])}, KV pool {g(r['kv'])}, forwards [{fw}], scorer {g(r['scorer'])}, cached free "
          f"{g(r['cached'])}; NEFFs {r['neffs']} ({r['neff_bytes'] / 2**20:.1f} MiB, outside the budget)"
          + (f"; runtime worker holds {g(r['worker_bytes'])}" if "worker_bytes" in r else "") + (f"; {r['ooms']} refused" if r["ooms"] else ""))

def upload(a: np.ndarray, device: str, dtype: DType) -> Tensor:
  """N7: a host array as a realized device Tensor, written straight into a new buffer (one copyin, which NeuronAllocator
  queues with the step's launches) instead of Tensor(a).realize()'s schedule and copy item per step input."""
  from fusezap.dtype import to_np
  a = np.ascontiguousarray(a, dtype=to_np(dtype))
  b = Buffer(device, a.size, dtype).ensure_allocated()
  if a.size: b.copyin(memoryview(a).cast("B"))
  return Tensor(LazyBuffer.from_buffer(b, tuple(a.shape)))

def rope_rows(cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Host RoPE tables (L, hd/2) -> the doubled ones NeuronKVCache.attend_rope multiplies by: [cos, cos] and [-sin, sin] (L, hd)."""
  return np.concatenate([cos, cos], -1), np.concatenate([-sin, sin], -1)

def rmsnorm(x: Tensor, w: Tensor, eps: float, d: Tensor | None = None) -> tuple[Tensor, Tensor]:
  """(x + d, w * rmsnorm(x + d)) (d None: (x, w * rmsnorm(x))) in one NKI kernel (nki_kernels.NormSpec), float32, rows of
  the last dim: Llama._add_norm on NEURON. The residual stream x + d is stored, so the next layer reads one buffer."""
  from fusezap.codegen.nki_kernels import NormSpec
  from fusezap.device.neuron import NeuronCall
  from fusezap import jit as J
  D = int(x.shape[-1])
  N = int(np.prod(x.shape[:-1]))
  xb, wb = _buf(x), _buf(w)
  h = Buffer(xb.device, N * D, dtypes.float32)
  if d is None:
    J.run_item(ScheduleItem((NeuronCall(NormSpec(N, D, float(eps), False)),), (h, xb, wb), {}, "custom", 1))
    return x, _wrap(h, tuple(x.shape))
  xo = Buffer(xb.device, N * D, dtypes.float32)
  J.run_item(ScheduleItem((NeuronCall(NormSpec(N, D, float(eps), True)),), (h, xo, xb, _buf(d), wb), {}, "custom", 2))
  return _wrap(xo, tuple(x.shape)), _wrap(h, tuple(x.shape))

_FENCE_BUF: dict = {}
def fence(d: Tensor | None, layer: int, device: str, hard: bool = False, every: int = 0) -> Tensor | None:
  """N6: the fusion boundary before decoder layer `layer` (hard: before the head). The residual term d (the previous layer's
  down projection) is realized first, so a layer's items are [norm, qkv, RoPE, KV write, attention, o_proj, norm, gate_up,
  silu*mul, down] whatever comes next (the same items in the same order as without the fence), then a no-op fence item is
  recorded (NEURONDevice.fuse_schedule splits there: every middle layer is the same fused kernel). Nothing when
  FZ_NEURON_FUSE=0. every (I75): this forward's layers per fused kernel instead of FZ_NEURON_FUSE (a TP prefill)."""
  from fusezap.codegen.nki_kernels import FenceSpec
  from fusezap.device.neuron import NeuronCall, fuse_layers
  from fusezap import jit as J
  if not fuse_layers(): return d
  if d is not None: d = _wrap(_buf(d), tuple(d.shape))
  if (b := _FENCE_BUF.get(device)) is None: _FENCE_BUF[device] = b = Buffer(device, 1, dtypes.uint8).ensure_allocated()
  J.run_item(ScheduleItem((NeuronCall(FenceSpec(layer, hard, int(every))),), (b,), {}, "custom", 1))
  return d

def argmax_rows(x: Tensor) -> Tensor:
  """x (R, V) float32 -> (R,) int32 first index of the row max, as np.argmax (finite rows), with float32 index arithmetic
  (exact: V < 2**24) instead of Tensor.argmax's int32 MAX, which NEURON computes as an exact int tree. N7 FZ_NEURON_HEAD=v3
  (default) / v2: one kernel with the vocabulary on the partitions (nki_kernels.ArgmaxSpec); v1: the composed kernels below."""
  from fusezap.codegen.nki_kernels import ArgmaxSpec, head_argmax
  R, V = x.shape
  if head_argmax() and is_neuron(x.device) and x.dtype == dtypes.float32:
    from fusezap.device.neuron import NeuronCall
    from fusezap import jit as J
    xb = _buf(x)
    out = Buffer(xb.device, R, dtypes.int32)
    J.run_item(ScheduleItem((NeuronCall(ArgmaxSpec(int(R), int(V))),), (out, xb), {}, "custom", 1))
    return _wrap(out, (R,))
  rev = _rev(x.device, V)
  m = x.max(-1, keepdim=True)
  return (V - (x == m).where(rev.reshape(1, V), 0.0).max(-1)).cast(dtypes.int32)

def head_matmul(h: Tensor, tiles: Tensor, V: int, W: int | None = None, dmas: int | None = None, ws: bool | None = None) -> Tensor:
  """N7 FZ_NEURON_HEAD=v2: h (..., K) @ lm_headᵀ -> (..., V) float32 logits, one nki_kernels.HeadSpec launch over the weight
  in head_tiles' layout (tiles: that flat device array, relaid once at load with the same W; default FZ_NEURON_HEAD_W).
  ws (default: FZ_NEURON_HEAD=v4): the weight-stationary HeadSpec on the same layout."""
  from fusezap.codegen.nki_kernels import HEAD_DMAS, HEAD_W, HeadSpec, head_mode
  from fusezap.device.neuron import NeuronCall
  from fusezap import jit as J
  K = int(h.shape[-1])
  M = int(np.prod(h.shape[:-1]))
  hb = _buf(h)
  out = Buffer(hb.device, M * V, dtypes.float32)
  if tiles.dtype == dtypes.bfloat16:   # N8 (I70) FZ_NEURON_BF16_HEAD: bf16 tiles (nki_bf16_n8.head_tiles_bf16), MatmulBF16's bits
    from fusezap.codegen.nki_bf16_n8 import HeadBF16, bf16_head_w
    sp = HeadBF16(M, V, K, W or bf16_head_w())
  else: sp = HeadSpec(M, V, K, W or HEAD_W, dmas or HEAD_DMAS, head_mode() == "v4" if ws is None else ws)
  J.run_item(ScheduleItem((NeuronCall(sp),), (out, hb, _buf(tiles)), {}, "custom", 1))
  return _wrap(out, (*h.shape[:-1], V))

def matmul_bf16(x: Tensor, w: Tensor) -> Tensor:
  """I68: x (..., K) float32 @ w (K, N) bf16 K-major -> (..., N) float32, one nki_bf16.MatmulBF16 launch (bf16 operands on
  TensorE, float32 accumulation): Llama.linear on NEURON for a bf16 weight."""
  from fusezap.codegen.nki_bf16 import MatmulBF16
  from fusezap.device.neuron import NeuronCall
  from fusezap import jit as J
  K, Nn = int(w.shape[0]), int(w.shape[1])
  assert int(x.shape[-1]) == K and w.dtype == dtypes.bfloat16, f"matmul_bf16: {x.shape} @ {w.shape} {w.dtype}"
  if x.dtype != dtypes.float32: x = x.float()
  M = int(np.prod(x.shape[:-1]))
  xb = _buf(x)
  out = Buffer(xb.device, M * Nn, dtypes.float32)
  J.run_item(ScheduleItem((NeuronCall(MatmulBF16(M, Nn, K)),), (out, xb, _buf(w)), {}, "custom", 1))
  return _wrap(out, (*x.shape[:-1], Nn))

_REV: dict = {}
def _rev(device: str, V: int) -> Tensor:
  if (device, V) not in _REV: _REV[(device, V)] = Tensor(np.arange(V, 0, -1, dtype=np.float32), device=device).realize()
  return _REV[(device, V)]

class NeuronSteps:
  """Llama.prefill / decode on NEURON (batch rows sharing positions, the P4-MODEL interface) through the same ragged forward
  as LLMEngine: the contiguous cache's decode step binds the position as a symbolic start, and NEURON compiles var values
  into the kernel (exception 2), so every position would be a new set of NEFFs. Here batch row b is sequence b of a
  NeuronKVCache that owns max_len positions per row, a step is (ids, positions) as data, and the jitted forwards are keyed
  by their row counts only: decode (B rows) and one per prefill bucket (B * bucket rows, padding rows inert)."""
  def __init__(self, model: Any, block_size: int = 16):
    from fusezap.serve.models.llama import rope_tables
    from fusezap.serve.weights import to_dtype
    c, self.m, self.B, L = model.cfg, model, model.max_batch, model.max_len
    self.cache = NeuronKVCache(c.n_layers, c.n_kv_heads, c.head_dim, self.B * -(-L // block_size), block_size, model.dtype, model.device, L)
    for b in range(self.B): self.cache.ensure(b, L)
    cos, sin = rope_rows(*rope_tables(c, L))
    self.cos, self.sin = to_dtype(cos, model.dtype), to_dtype(sin, model.dtype)
    self.jits: dict[tuple[int, int], Any] = {}
  def jit(self, N: int, R: int) -> Any:
    from fusezap.jit import Jit
    if (N, R) not in self.jits:
      layout = StepLayout(N, R, self.cache.max_blocks, N != R)   # prefill: the rows are prompt positions (I75 reads it under TP)
      def step(x: Tensor, cos: Tensor, sin: Tensor, meta: Tensor, pools: list[Tensor]) -> Tensor:
        c = self.cache.bind(pools, [meta])
        c.layout = layout
        return self.m.forward_tokens(x, c, cos, sin, select=R != N).float()
      from fusezap.device.neuron import neuron_plan
      self.jits[(N, R)] = Jit(step, plan=neuron_plan())   # N8: intermediates share memory (FZ_NEURON_PLAN)
    return self.jits[(N, R)]
  def run(self, ids: np.ndarray, start: int, t: int) -> np.ndarray:
    """ids (B, T) at positions start.. (only the first t columns are real; the rest pad a prefill bucket) -> the logits of
    each row's last real position, (B, vocab) float32."""
    B, T = ids.shape
    N, R = B * T, B
    real = np.tile(np.arange(T) < t, B)
    sid = np.where(real, np.repeat(np.arange(B), T), -1)
    pos = np.where(real, start + np.tile(np.arange(T), B), 0)
    sel = [b * T + t - 1 for b in range(B)]
    meta = self.cache.step_meta_rows(N, R, sid, pos, sel)
    m = self.m
    x = Tensor(m.embed(ids.reshape(1, N))[0], device=m.device, dtype=m.dtype)
    cos = Tensor(self.cos[pos], device=m.device, dtype=m.dtype)
    sin = Tensor(self.sin[pos], device=m.device, dtype=m.dtype)
    mt = Tensor(meta, device=m.device, dtype=dtypes.int32)
    with m._opt_env(prefill=T > 1): out = self.jit(N, R)(x, cos, sin, mt, self.cache.state())
    return out.numpy()
