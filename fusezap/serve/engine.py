"""LLMEngine: continuous batching over a paged KV cache (SPEC §11.3).

  eng = LLMEngine(model, num_blocks=1024, block_size=16, max_num_seqs=32)
  rid = eng.add_request(prompt_ids, SamplingParams(max_tokens=64))
  while eng.has_unfinished():
    for out in eng.step():          # RequestOutput per request that got tokens (or finished) in this step
      print(out.request_id, out.new_token_ids, out.finished)
  eng.abort(rid)

A step runs at most two jitted forwards over the shared paged cache (`Llama.forward_tokens`):
  - decode: max_num_seqs rows, one token per running sequence past its prompt (padding rows are inert);
  - prefill: up to max_prefill_tokens rows of prompt chunks from up to max_num_seqs sequences, returning the logits of
    the chunks that end a prompt. It runs at the smallest of `prefill_sizes` that holds the step's prompt rows (CUDA:
    max_prefill_tokens, default prefill_budget(): 2048 if the free memory holds it, and its halves down to 256;
    elsewhere max_prefill_tokens alone, default 256).
Each forward has a fixed shape, so each is captured once and then replayed (no tracing, scheduling or lowering per
step), and their kernels never change with the batch composition (the prefill sizes share one split-K attention
partition count, the largest size's, since that count sets the order of a token's key sums). Every kernel computes a row independently of the other rows,
and every position of a sequence is always computed by the same kind of forward (prompt positions by prefill, generated
positions by decode, also when recomputed after a preemption). So a request's tokens are bit-identical whether it runs
alone or batched with any others, chunked or not, preempted or not; tests/test_scheduler.py checks exactly that.
Decode kernels are BEAM-tuned (decode_beam; the winners are cached on disk), prefill uses the library GEMM.
"""
from __future__ import annotations
import contextlib, itertools, os, sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence as Seq
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap.jit import Jit
from fusezap.helpers import getenv
from fusezap.serve.kv_cache import PagedKVCache, StepLayout, lib_linear
from fusezap.serve.sampling import Sampler
from fusezap.serve.scheduler import Scheduler, SchedulerConfig, Sequence, SeqStatus, SchedulerOutput
from fusezap.serve.weights import to_dtype
from fusezap.serve.models.llama import rope_tables

@dataclass
class SamplingParams:
  max_tokens: int = 16
  temperature: float = 0.0
  top_k: int = 0
  top_p: float = 1.0
  repetition_penalty: float = 1.0
  seed: int | None = None
  stop_token_ids: list[int] = field(default_factory=list)
  ignore_eos: bool = False
  constraint: Any = None   # constrain.Constraint: the output must match its grammar (masked logits; see _Logits.token)
  def sampler(self) -> Sampler: return Sampler(self.temperature, self.top_k, self.top_p, self.repetition_penalty, self.seed)

@dataclass
class RequestOutput:
  request_id: int
  new_token_ids: list[int]          # tokens produced in this step (streaming)
  token_ids: list[int]              # every token produced so far
  finished: bool = False
  finish_reason: str | None = None  # "stop" | "length" | "abort"
  num_prompt_tokens: int = 0

@contextlib.contextmanager
def _opt_env(model: Any, prefill: bool, beam: int):
  """Llama._opt_env with the engine's decode beam: decode rows use composed BEAM-tuned kernels, prefill the library GEMM."""
  old = model.beam
  model.beam = beam
  try:
    with model._opt_env(prefill=prefill): yield
  finally: model.beam = old

class _Rows:
  """A step's token rows as arrays: sid[i] the sequence of row i (-1: a padding row), pos[i] its position, ids[i] its
  token. Built per prompt chunk with numpy (the per-row tuples, and Sequence.tokens re-concatenated per row, were most
  of a 4096-row prefill step's 6.6 ms host prep)."""
  __slots__ = ("sid", "pos", "ids")
  def __init__(self, sid: np.ndarray, pos: np.ndarray, ids: np.ndarray): self.sid, self.pos, self.ids = sid, pos, ids
  def __len__(self) -> int: return len(self.sid)
  @staticmethod
  def build(chunks: Seq[tuple[Sequence, int, int]], pad_to: int = 0, decode: Seq[Sequence] = ()) -> "_Rows":
    """Prompt chunks (seq, start, n) in order, then padding rows up to pad_to (mixed forwards), then one row per decode
    sequence at its position num_computed."""
    sid, pos, ids = [], [], []
    for s, st, n in chunks:
      sid.append(np.full(n, s.seq_id, np.int64))
      pos.append(np.arange(st, st + n, dtype=np.int64))
      src = s.prompt if st + n <= len(s.prompt) else s.tokens
      ids.append(np.asarray(src[st:st + n], np.int64))
    k = sum(n for _, _, n in chunks)
    if pad_to > k:
      sid.append(np.full(pad_to - k, -1, np.int64)); pos.append(np.zeros(pad_to - k, np.int64)); ids.append(np.zeros(pad_to - k, np.int64))
    if decode:
      sid.append(np.array([s.seq_id for s in decode], np.int64))
      pos.append(np.array([s.num_computed for s in decode], np.int64))
      ids.append(np.array([s.tokens[s.num_computed] for s in decode], np.int64))
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0, np.int64)
    return _Rows(cat(sid), cat(pos), cat(ids))

class _Forward:
  """One fixed-shape jitted ragged forward over the paged cache: N token rows, R logits rows. Prefill always selects its
  R rows (sel), even when R == N; decode (prefill=False, R == N) returns every row. np_n: the row count the split-K
  attention partitions are sized for (StepLayout.np_n). plan: the capture's intermediates share memory (Jit plan)."""
  def __init__(self, eng: "LLMEngine", N: int, R: int, prefill: bool, np_n: int = 0, plan: bool = False, alias: bool = False, mix: int = 0,
               mix_exact: bool = True):
    self.eng, self.N, self.R, self.prefill = eng, N, R, prefill
    self.key = ("m", mix) if mix else ("p" if prefill else "d", N)   # names this forward for the tensor-parallel workers
    self.layout = StepLayout(N, R, eng.cache.max_blocks, prefill, np_n, mix, mix_exact)
    # alias: the replay returns its persistent logits buffer instead of a copy (19.5 MB per 32-row step); valid until the
    # next replay of this forward, and a step reads its logits before that (pending decodes are greedy: they read ids)
    self.jit = Jit(self._step, plan=plan, alias=(0,) if alias else ())
    # NEURON (N6): one jitted forward per pool prefix (NeuronKVCache.ctx_buckets: the KV write and attention touch only the
    # slots below it), chosen per step from the step's block tables; self.jit is the whole pool's
    self.ctx_jits: dict[int, Jit] = {}
  def jit_for(self, ctx: int | None) -> Jit:
    if ctx is None or ctx == getattr(self.eng.cache, "P", None): return self.jit
    if ctx not in self.ctx_jits:
      import functools
      self.ctx_jits[ctx] = Jit(functools.partial(self._step, ctx=ctx), plan=self.jit.plan, alias=self.jit.alias)
    return self.ctx_jits[ctx]
  def _step(self, x: Tensor, cos: Tensor, sin: Tensor, meta: Tensor, pools: list[Tensor], ctx: int | None = None) -> Tensor | tuple[Tensor, Tensor]:
    c = self.eng.cache.bind(pools, [meta])
    c.layout = self.layout
    if ctx is not None: c.ctx = ctx
    wide = lib_linear if not self.prefill and self.eng.lib_gemm else None
    lg = self.eng.model.forward_tokens(x, c, cos, sin, select=self.prefill or self.R != self.N, wide_linear=wide).float()
    if not self.eng.device_argmax: return lg
    if self.eng.neuron:
      from fusezap.codegen.nki_kernels import head_argmax
      from fusezap.serve.neuron_kv import _buf, _wrap, argmax_rows as neuron_argmax
      # N7 ArgmaxSpec reads the logits buffer itself: return that buffer, not a view of it (fz.jit would copy a view out)
      if head_argmax(): lg = _wrap(_buf(lg), tuple(lg.shape))
      return lg, neuron_argmax(lg)
    from fusezap.serve.cuda import argmax_rows
    return lg, argmax_rows(lg)
  def __call__(self, tokens: Seq[tuple[int, int, int]], sel: Seq[int], src: Seq[int] | None = None, prev: "_Logits | None" = None,
               kind: str = "") -> "_Logits":
    """tokens: (seq_id, position, token id) per row; sel: rows whose logits to return (all rows when R == N).
    src (decode only): per row, the row of `prev` (a launched decode whose tokens are not read yet) whose greedy token is
    this row's input, or -1 for the row's own token id. On CUDA with LLMEngine.dsteps the inputs are built on the device
    (one async upload, no wait for the GPU) and prev's device argmaxes feed the rows directly; elsewhere prev's tokens
    are read on the host first (same tokens, no overlap)."""
    e, m = self.eng, self.eng.model
    ids = np.zeros(self.N, np.int64)
    pos = np.zeros(self.N, np.int64)
    sv = np.full(self.N, -1, np.int64)
    if isinstance(tokens, _Rows):
      n, real = len(tokens), tokens.sid >= 0
      ids[:n], pos[:n] = np.where(real, tokens.ids, 0), np.where(real, tokens.pos, 0)
      meta = e.cache.step_meta_rows(self.N, self.R, tokens.sid, tokens.pos, sel)
    else:
      for i, (sq, p, t) in enumerate(tokens):
        if sq is not None: ids[i], pos[i] = t, p   # None: a padding row inside the step (mixed forwards)
      meta = e.cache.step_meta(self.N, self.R, [(s, p) for s, p, _ in tokens], sel)
    if src is not None: sv[:len(src)] = src
    if e.tp is not None and e.dsteps is None and (sv >= 0).any():   # tokens resolved here, so the workers get plain ids
      ids, sv = np.where(sv >= 0, [prev.host_id(int(j)) if j >= 0 else 0 for j in sv], ids), np.full(self.N, -1, np.int64)
    return self.run(ids, sv, pos, meta, prev, kind)
  def run(self, ids: np.ndarray, sv: np.ndarray, pos: np.ndarray, meta: np.ndarray, prev: "_Logits | None" = None, kind: str = "") -> "_Logits":
    """The forward on the step's host arrays (ids, sources, positions: N rows; meta: the cache metadata). With tensor
    parallelism rank 0 sends them to the workers first, which run this same forward on their shards (mirror_forward)."""
    tp = self.eng.tp
    if tp is None: return self._run(ids, sv, pos, meta, prev, kind)
    with tp.spmd("fwd", self.key, ids, sv, pos, meta, prev.seq if prev is not None else -1, kind): return self._run(ids, sv, pos, meta, prev, kind)
  def _run(self, ids: np.ndarray, sv: np.ndarray, pos: np.ndarray, meta: np.ndarray, prev: "_Logits | None", kind: str) -> "_Logits":
    e, m = self.eng, self.eng.model
    tp = e.tp
    pf, k = e._prof, kind or ("p" if self.prefill else "d")
    if (tpp := e._tpprof) is not None: tpp.begin()
    if (ds := e.dsteps) is not None:   # decode and prefill (a 2048-row host gather + upload took ~3.9 ms of GPU idle)
      x, cos, sin, mt = ds.inputs(ids, sv, pos, meta, prev.ids_buf() if prev is not None else None)
    elif e.upload:
      from fusezap.serve.neuron_kv import upload
      if (sv >= 0).any(): ids = np.where(sv >= 0, [prev.host_id(int(j)) if j >= 0 else 0 for j in sv], ids)
      x, cos, sin = (upload(a, m.device, m.dtype) for a in (m.embed(ids[None])[0], e._cos[pos], e._sin[pos]))
      mt = upload(meta, m.device, dtypes.int32)
    else:
      if (sv >= 0).any(): ids = np.where(sv >= 0, [prev.host_id(int(j)) if j >= 0 else 0 for j in sv], ids)
      x = Tensor(m.embed(ids[None])[0], device=m.device, dtype=m.dtype)
      cos = Tensor(e._cos[pos], device=m.device, dtype=m.dtype)
      sin = Tensor(e._sin[pos], device=m.device, dtype=m.dtype)
      mt = Tensor(meta, device=m.device, dtype=dtypes.int32)
      if pf is not None:
        for t in (x, cos, sin, mt): t.realize()   # the uploads count as prep, not as the jit call
    if pf is not None: pf.mark(k + "_prep"); pf.fwd(k, 0)
    with _opt_env(m, self.prefill, e.decode_beam):
      ctx = e.cache.meta_ctx(meta, self.N, self.R, self.prefill) if hasattr(e.cache, "meta_ctx") else None   # NEURON: this step's pool prefix
      out = self.jit_for(ctx)(x, cos, sin, mt, e.cache.state())
      if e._ctx_seen is not None and ctx is not None: e._ctx_seen.append((ctx, getattr(e.cache, "last_top", 0)))   # FZ_SERVE_PROFILE
    if pf is not None: pf.fwd(k, 1)
    lg = _Logits(*out) if isinstance(out, tuple) else _Logits(out)
    lg.seq, e._nfwd = e._nfwd, e._nfwd + 1
    if ds is not None and lg._ids is not None and (tp is None or tp.rank == 0): lg.rb = ds.readback(lg.ids_buf())   # async copy of the ids + an event
    if tpp is not None: tpp.end(lg.seq, "m" if self.key[0] == "m" else k, self.N)
    if pf is not None: pf.mark(k + "_call")
    return lg

@dataclass
class _Pending:
  """A launched decode whose tokens are not read yet: rows (sequence, produces a token, row) and its logits / ids."""
  rows: list
  lg: "_Logits"

class _Logits:
  """A step's (R, vocab) float32 logits on the device, read back only when a row is sampled on the host; `ids` (R,)
  int32 are the device argmaxes (CUDA), which greedy rows without a repetition penalty use instead (the same token as
  np.argmax of the row: first index of the maximum)."""
  def __init__(self, lg: Tensor, ids: Tensor | None = None):
    self.lg, self._ids, self._host, self._hids, self.rb, self.seq = lg, ids, None, None, None, -1
  def ids_buf(self) -> Any: return self._ids._realized_contiguous().realized
  def host_id(self, i: int) -> int:
    """Row i's greedy token (device argmax, or np.argmax of the host logits when there is none)."""
    if self._ids is None:
      if self._host is None: self._host = self.lg.numpy()
      return int(np.argmax(self._host[i]))
    if self._hids is None: self._hids = self.rb.get() if self.rb is not None else self._ids.numpy()
    return int(self._hids[i])
  def row(self, i: int) -> np.ndarray:
    """Row i of the logits on the host. CUDA: that row alone (one vocab-wide copy, not R of them); Metal: read from the
    shared buffer; elsewhere the whole host copy."""
    if self._host is not None: return self._host[i]
    buf = self.lg._realized_contiguous().realized
    V = int(self.lg.shape[-1])
    from fusezap.device import Device
    al = Device[buf.device].allocator
    if buf.dtype.itemsize == 4 and isinstance(buf._handle, int):   # CUDA: a device pointer
      out = np.empty(V, np.float32)
      al.copyout(memoryview(out).cast("B"), buf.handle + i * V * 4)
      return out
    if buf.dtype.itemsize == 4 and hasattr(al, "_view"):           # Metal: shared memory, after the queued work
      al.dev.synchronize()
      return np.frombuffer(al._view(buf.handle, buf.nbytes), np.float32, V, i * V * 4).copy()
    self._host = self.lg.numpy()
    return self._host[i]
  def token(self, i: int, s: Sequence) -> int:
    sp = s.sampler
    if s.cons is not None: return int(sp(s.cons.mask(self.row(i)), s.tokens))   # constrained: masked on the host (grammar state there)
    if self._ids is not None and sp.greedy and sp.repetition_penalty == 1.0: return self.host_id(i)
    if self._host is None: self._host = self.lg.numpy()
    return int(sp(self._host[i], s.tokens))

def prefill_sizes(device: str, max_prefill_tokens: int) -> list[int]:
  """Row counts of the prefill forwards, ascending. CUDA: max_prefill_tokens and its halves down to 256, so a big budget
  admits many prompts per step (every mixed step also pays a whole decode forward) while a short prompt still runs a
  small forward; a token's result does not depend on the row count (row-wise kernels, M-independent GEMM sums).
  Elsewhere one size, as before."""
  sizes, n = [max_prefill_tokens], max_prefill_tokens
  if device.split(":")[0].upper() in ("NEURON", "NKISHIM", "NKISIM"):   # I77: FZ_NEURON_PREFILL_SIZES (e.g. 128,512) below max_prefill_tokens
    from fusezap.serve.neuron_prefill import prefill_sizes as _nps
    return sorted({s for s in (_nps(max_prefill_tokens, device) or []) if s < max_prefill_tokens} | {max_prefill_tokens})
  if device.split(":")[0] == "CUDA":
    while n % 2 == 0 and n // 2 >= 256: n //= 2; sizes.append(n)
  return sorted(sizes)

def decode_buckets(device: str, max_num_seqs: int) -> list[int]:
  """Row counts of the decode forwards, ascending. CUDA: 1, 2, 4, ... below max_num_seqs, then max_num_seqs (a step runs
  the smallest that holds its sequences, so one sequence no longer pays for max_num_seqs rows of GEMM, attention and
  lm_head; NEURON too, N5: each bucket is one more set of NEFFs); FZ_DECODE_BUCKETS=0 keeps one size. Elsewhere one size (the decode kernels are BEAM-tuned per shape there).
  Every decode kernel computes a row independently of the row count (swap-AB GEMM sums do not depend on M, attention
  partitions follow np_n = max_num_seqs), so a token is bit-identical whichever bucket runs it."""
  if device.split(":")[0] not in ("CUDA", "NEURON", "NKISHIM", "NKISIM") or not getenv("FZ_DECODE_BUCKETS", 1): return [max_num_seqs]
  sizes, n = [], 1
  while n < max_num_seqs: sizes.append(n); n *= 2
  return sizes + [max_num_seqs]

def prefill_row_bytes(cfg: Any, itemsize: int, quant: str | None = None) -> int:
  """Conservative activation bytes per prefill row while a prefill forward is traced or captured (every intermediate of
  the forward is alive then): per layer about three intermediate_size rows (gate/up, SiLU*mul), ten hidden_size rows
  and two qkv rows. An fp8 (W8A8) linear also keeps its input's e4m3 row (K bytes) and the GEMM's float32 output
  (4 N bytes) next to the cast one: about twice the bf16 bytes, which 32B fp8 on an 80 GB card ran out of memory on."""
  H, I, QKV = cfg.hidden_size, cfg.intermediate_size, (cfg.n_heads + 2 * cfg.n_kv_heads) * cfg.head_dim
  fp8 = (3 * H + I) + 4 * (QKV + H + 2 * I + H) if quant == "fp8" else 0
  return cfg.n_layers * (itemsize * (3 * I + 10 * H + 2 * QKV) + fp8)

def prefill_budget(free: int, row_bytes: int, reserve: int = 1 << 30) -> int:
  """CUDA default max_prefill_tokens: the largest of 4096, 2048, 1024, 512 (at most FZ_PREFILL_MAX, default 4096) whose
  forward's activations fit the device memory left after the weights and the KV pool (minus `reserve` for the decode
  forwards and the smaller prefill sizes), else 256. An 80 GB card alone takes 4096 for 7B bf16 (2048 when vLLM holds
  45% of it, as in the Validate config); a 24 GB card 1024. A larger forward halves the mixed decode + prefill steps of
  a burst of prompts (each pays a whole decode forward); a token's result does not depend on the size (packing-invariant
  prefill), so this changes no token."""
  cap = getenv("FZ_PREFILL_MAX", 4096)
  return next((n for n in (4096, 2048, 1024, 512) if n <= cap and n * row_bytes <= free - reserve), 256)

def enable_cublaslt(eng: "LLMEngine") -> dict:
  """Prefill GEMMs on cuBLASLt (CUDA f16/bf16, FZ_CUBLASLT=1 default; FZ_CUBLASLT_SHAPES, default qkv,o,gate_up): one
  pinned, self-checked algo per shape for exactly the engine's prefill sizes (fusezap/device/cuda_kernels/cublaslt.py),
  so a prompt token's bits do not depend on the bucket or the other rows. down_proj keeps FuseZap's kernel (faster than
  cuBLAS in round 11's microbench); decode rows, mixed forwards and every other M keep FuseZap's kernels."""
  from fusezap.device.cuda_kernels import cublaslt
  from fusezap.serve import cuda as _cu
  m = eng.model
  if m.dtype not in (dtypes.float16, dtypes.bfloat16) or not getenv("FZ_CUBLASLT", 1) or not m.layers or not cublaslt.available(): return {}
  l, kind = m.layers[0], ("bf16" if m.dtype == dtypes.bfloat16 else "f16")
  want = set((os.environ.get("FZ_CUBLASLT_SHAPES") or "qkv,o,gate_up").split(","))
  shapes = []
  for nm, w, out in (("qkv", l.wqkv, "f32" if l.bqkv is not None and not _cu.bias_round2() else kind), ("o", l.wo, kind),
                     ("gate_up", l.wgu, kind), ("down", l.wd, kind)):
    if nm in want and getattr(w, "apply", None) is None and len(getattr(w, "shape", ())) == 2: shapes.append((int(w.shape[0]), int(w.shape[1]), kind, out))
  if not shapes: return {}
  log = lambda msg: print(msg, flush=True)
  plans = cublaslt.enable(m.device, shapes, list(eng.prefill_sizes), log=log)
  # decode rows (FZ_CUBLASLT_DECODE, opt-in, default off; e.g. "o,gate_up"): a second pinned algo per shape, self-checked
  # over the decode buckets (1 .. max_num_seqs), so a decode row's bits do not depend on the bucket or the batch. Not with
  # exact-mode mixed forwards, whose decode rows must carry the swap-AB kernel's bits (cuda.linear redo_rows).
  dec = set(filter(None, (os.environ.get("FZ_CUBLASLT_DECODE") or "").replace("+", ",").split(",")))   # "o,gate_up" or "o+gate_up"
  if dec and not (eng.mix_sizes and eng.mode == "exact"):
    names = {"qkv": l.wqkv, "o": l.wo, "gate_up": l.wgu, "down": l.wd}
    dshapes = [sh for nm, sh in zip(("qkv", "o", "gate_up", "down"), [(int(w.shape[0]), int(w.shape[1]), kind, "f32" if nm == "qkv" and l.bqkv is not None and not _cu.bias_round2() else kind)
               for nm, w in names.items()]) if nm in dec and getattr(names[nm], "apply", None) is None]
    if dshapes: cublaslt.enable(m.device, dshapes, list(eng.decode_sizes), log=log, group="decode")
  return plans

def autotune_gemms(eng: "LLMEngine") -> list:
  """The decode GEMMs' per-shape table (fusezap/device/cuda_kernels/gemm_table.py) at engine start, before any forward is
  traced or captured: CUDA sm_90 f16/bf16 (the swap-AB kernel), FZ_GEMM_AUTOTUNE=1 (default) and FZ_GEMM_TABLE=1; each
  linear shape of the model (qkv, o, gate_up, down, lm_head) without a disk entry is timed on the model's own weights at
  every decode bucket's BX. Every rank of a TP group tunes its own (same shapes); the entry written first is kept."""
  from fusezap.device import Device
  from fusezap.device.cuda_kernels import gemm_table as GT, ONE_K_ORDER
  from fusezap.serve import cuda as _cu
  m = eng.model
  if m.dtype not in (dtypes.float16, dtypes.bfloat16) or not getenv("FZ_GEMM_AUTOTUNE", 1) or not GT.enabled() or ONE_K_ORDER[0] or not m.layers: return []
  dev = Device[m.device]
  if tuple(dev.cc) != (9, 0): return []
  kind = "bf16" if m.dtype == dtypes.bfloat16 else "f16"
  qkv_out = "f32" if m.layers[0].bqkv is not None and not _cu.bias_round2() else kind
  shapes = []
  for nm, attr, out in (("qkv", "wqkv", qkv_out), ("o", "wo", kind), ("gate_up", "wgu", kind), ("down", "wd", kind)):
    ws = [getattr(l, attr) for l in m.layers]
    if any(getattr(w, "apply", None) is not None or len(getattr(w, "shape", ())) != 2 for w in ws): continue
    shapes.append((nm, int(ws[0].shape[0]), int(ws[0].shape[1]), out, [_cu._buf(w).handle for w in ws[:16]]))
  if getattr(m.lm_head, "apply", None) is None: shapes.append(("lm_head", int(m.lm_head.shape[0]), int(m.lm_head.shape[1]), "f32", [_cu._buf(m.lm_head).handle]))
  rank = getattr(getattr(m, "tp", None), "rank", 0)
  log = (lambda msg: print(msg, flush=True)) if rank == 0 else (lambda msg: print(msg.replace("fusezap:", f"fusezap (rank {rank}):", 1), file=sys.stderr, flush=True))
  return GT.autotune(dev, kind, shapes, list(eng.decode_sizes), log=log)

class LLMEngine:
  def __init__(self, model: Any, num_blocks: int = 1024, block_size: int = 16, max_num_seqs: int = 32,
               max_prefill_tokens: int | None = None, max_model_len: int | None = None, decode_beam: int | None = None,
               cache: PagedKVCache | None = None, prefill_buckets: Seq[int] | None = None, mode: str | None = None):
    """model: fusezap.serve.models.llama.Llama (its own contiguous cache is not used). num_blocks * block_size is the
    KV capacity in tokens, shared by all requests. max_prefill_tokens: prompt tokens per step (default on CUDA
    prefill_budget() of the free device memory: 2048 on a large card; 256 elsewhere); prefill_buckets: the prefill forwards' row counts (default prefill_sizes(); the largest must be
    max_prefill_tokens). decode_beam: FZ_BEAM width for the decode kernels (default: the model's beam, else 4: the
    hand-coded opts are far from tuned at max_num_seqs rows; 0 on CUDA).
    mode (default FZ_MODE, else "exact"): "exact" computes every position with one kind of forward, so a request's tokens
    are bit-identical whether it runs alone or batched. "throughput" (opt-in; CUDA sm_90 f16/bf16) lets the decode rows of
    a step ride inside its prefill forward on the prefill kernels, as vLLM's mixed batches do: faster when prompts and
    decodes overlap, but a request's near-tie tokens may then depend on what shared its steps. A request served alone
    never mixes, so it gets exactly the exact mode's tokens. `mode` is the mode in effect (throughput falls back to exact
    where its kernels do not apply)."""
    self.model = model
    self.tp = getattr(model, "tp", None)   # tensor parallelism (fusezap.serve.dist): rank 0 mirrors each forward on the workers
    from fusezap.serve.tpprof import from_env as _tpprof_env
    self._tpprof = _tpprof_env(self)       # FZ_TP_PROFILE=<dir>: a line per forward per rank (fusezap/serve/tpprof.py)
    self._nfwd, self._seen = 0, {}         # forwards run so far; a worker's recent logits by sequence number (lookahead)
    cuda = model.device.split(":")[0] == "CUDA"
    self.max_model_len = min(max_model_len or model.max_len, model.max_len)
    from fusezap.serve.neuron_kv import NeuronKVCache, is_neuron
    self.neuron = is_neuron(model.device)   # N5: the pool's device side without indirect DMA (fusezap/serve/neuron_kv.py)
    self.cache = cache or (NeuronKVCache if self.neuron else PagedKVCache).for_model(model, num_blocks, block_size, self.max_model_len)
    cos, sin = rope_tables(model.cfg, self.max_model_len)
    if self.neuron:   # the step's RoPE rows as NeuronKVCache multiplies them (no table kernels per step)
      from fusezap.serve.neuron_kv import rope_rows
      cos, sin = rope_rows(cos, sin)
    self._cos, self._sin = to_dtype(cos, model.dtype), to_dtype(sin, model.dtype)
    # lookahead (FZ_LOOKAHEAD, default on for CUDA): a decode-only step of greedy rows enqueues the next decode, fed by its
    # device argmaxes, before it reads its own tokens, so the host work of a step overlaps the GPU; CUDA builds decode
    # inputs on the device (DeviceSteps: embedding + RoPE tables there, about 1.1 GB for a 152k x 3584 bf16 table)
    self.lookahead, self.dsteps = bool(getenv("FZ_LOOKAHEAD", 1 if cuda else 0)), None
    if cuda and self.lookahead:
      from fusezap.device import Device
      from fusezap.serve.cuda import DeviceSteps
      E = model.embed_host
      need = int(np.prod(E.shape)) * model.dtype.itemsize + (4 << 30)
      if DeviceSteps.ok(int(E.shape[1]), model.cfg.head_dim, model.dtype) and Device[model.device].mem_free() > need:
        self.dsteps = DeviceSteps(model.device, to_dtype(E, model.dtype), self._cos, self._sin, model.dtype)
      else: self.lookahead = False
    self._pend: _Pending | None = None
    if max_prefill_tokens is None:
      from fusezap.device import Device
      from fusezap.serve.neuron_prefill import prefill_sizes as _nps
      max_prefill_tokens = prefill_budget(Device[model.device].mem_free(), prefill_row_bytes(model.cfg, model.dtype.itemsize, getattr(model, "quant", None))) if cuda else \
                           ((max(_nps(128, model.device) or [128]) if self.neuron else 256))   # NEURON: attention rows = tokens x q heads per kv
                                                                                 # head; I77: the largest FZ_NEURON_PREFILL_SIZES
    self.cfg = SchedulerConfig(max_num_seqs, max_prefill_tokens)
    self.scheduler = Scheduler(self.cfg, self.cache)
    self.lib_gemm = model.device.split(":")[0] == "METAL"   # decode's gate/up and lm_head on the simdgroup GEMM
    # greedy tokens on the device: R ints read back, not R * vocab logits (NEURON: FZ_NEURON_DEVICE_ARGMAX=0 reads the logits)
    self.device_argmax = cuda or (self.neuron and os.environ.get("FZ_NEURON_DEVICE_ARGMAX", "1") not in ("", "0"))
    from fusezap.device.neuron import host_fast
    self.upload = self.neuron and host_fast()   # N7: step inputs written straight into device buffers (no schedule per input)
    # CUDA: the linears are library GEMMs and BEAM search is compile-bound (NVRTC per candidate, ~30+ min cold for 0.5B),
    # so the composed rest keeps the hand-coded opts unless a beam is asked for
    self.decode_beam = decode_beam if decode_beam is not None else (model.beam or (0 if cuda else 4))
    self.stop_ids = set(model.cfg.eos_token_ids)
    self.decode_sizes = decode_buckets(model.device, max_num_seqs)
    from fusezap.device.neuron import neuron_plan
    plan = cuda or (self.neuron and neuron_plan())   # N8 (I71): NEURON plans its forwards' intermediates too
    self._decodes = {n: _Forward(self, n, n, False, max_num_seqs if cuda else 0, plan=plan, alias=cuda) for n in self.decode_sizes}
    self._decode = self._decodes[max_num_seqs]
    self.prefill_sizes = sorted(prefill_buckets) if prefill_buckets else prefill_sizes(model.device, max_prefill_tokens)
    if self.neuron and not prefill_buckets:   # I77: the prefill kernels' device self-test (FZ_NEURON_PREFILL_KERNEL=v2), before any capture;
      from fusezap.serve import neuron_prefill as _NP   # bigger prefill sizes (FZ_NEURON_PREFILL_SIZES) only on them: the members re-stream
      if (line := _NP.gate(self, rows=tuple(self.prefill_sizes), ctxs=tuple(self.cache.ctx_buckets(True)))): print(line, flush=True)
      if len(self.prefill_sizes) > 1 and not _NP.enabled():   # the weights per 128-row block (a 512-row member forward: ~4x, ~3.4 GB a prefix)
        print(f"fusezap: prefill sizes {self.prefill_sizes} need the prefill kernels (FZ_NEURON_PREFILL_KERNEL=v2, on and self-tested): "
              f"{self.prefill_sizes[0]} rows only", flush=True)
        self.prefill_sizes = self.prefill_sizes[:1]
        max_prefill_tokens = self.cfg.max_prefill_tokens = self.prefill_sizes[0]
    assert self.prefill_sizes[-1] == max_prefill_tokens, f"largest prefill bucket {self.prefill_sizes[-1]} != max_prefill_tokens {max_prefill_tokens}"
    # every prefill size sizes the split-K attention partitions for the largest one (a token's sum order must not
    # depend on which size its step ran), and on CUDA its intermediates share memory: unplanned, each size kept its whole
    # activation set alive and 7B bf16 with 2048 prefill tokens no longer fit a 24 GB card
    self._prefills = {n: _Forward(self, n, max_num_seqs, True, max_prefill_tokens, plan=plan, alias=cuda) for n in self.prefill_sizes}
    self._prefill = self._prefills[max_prefill_tokens]
    # mixed forwards (CUDA, f16/bf16 with the tensor-core prefill and decode attention; FZ_MIX=0 disables): while
    # sequences decode, a step's prompt chunk (at most FZ_MIX_TOKENS rows, default 512) and its decode rows run as ONE
    # forward: rows 0 .. C-1 the chunk (padded to C), rows C .. C+max_num_seqs-1 the decode rows. Decode rows take the
    # decode forwards' attention kernel and partitions; in the GEMMs the prefill tile sums like the decode swap-AB kernel
    # without split-K (bench/korder_check.py: bit-identical, sm_90), and the linears whose decode config splits K (7B
    # o_proj / down_proj) run again on the decode rows with the decode kernel (cuda.linear redo_rows). So a decode row's
    # result is the same in either forward and a request's tokens do not depend on what shares its steps. A decoding step
    # stays short (no separate 2048-row prefill forward to wait for) and most of its decode work rides in the prefill GEMMs.
    want = (mode or os.environ.get("FZ_MODE") or "exact").lower()
    assert want in ("exact", "throughput"), f"mode must be 'exact' or 'throughput', got {want!r}"
    thr = want == "throughput"
    self.mix_sizes: list[int] = []
    if not cuda and getenv("FZ_MIX_ANY", 0) and (thr or getenv("FZ_MIX", 0)):   # tests: the engine's mixed-step bookkeeping off CUDA (no bit-identity claim there)
      c = 64
      while c <= min(getenv("FZ_MIX_TOKENS", 512), max_prefill_tokens): self.mix_sizes.append(c); c *= 2
    elif cuda and model.dtype in (dtypes.float16, dtypes.bfloat16) and model.cfg.head_dim in (64, 128) and self.device_argmax and (thr or getenv("FZ_MIX", 0)):
      from fusezap.device import Device
      from fusezap.device.cuda_kernels import attention as _A
      dv = Device[model.device]
      G, xs = model.cfg.n_heads // model.cfg.n_kv_heads, (model.cfg.n_heads + 2 * model.cfg.n_kv_heads) * model.cfg.head_dim
      hd_ = model.cfg.head_dim
      dec_ok = _A.kpt_ok(model.dtype, hd_, G, xs, max_num_seqs * model.cfg.n_heads * -(-self.cache.max_blocks * self.cache.bs // _A.KPT_PART)) or \
               _A.mdec_ok(dv, model.dtype, hd_, G, xs)   # the decode kernels that take a row offset
      # exact mixing needs a decode kernel that takes a row offset; throughput mode can put decode rows on the prefill kernel
      if tuple(dv.cc) == (9, 0) and _A.use_paged_flash(dv, model.dtype, hd_, xs) and (dec_ok or thr):
        top = max(64, min(getenv("FZ_MIX_TOKENS", max_prefill_tokens if thr else 512), max_prefill_tokens))
        c = 64
        while c <= top: self.mix_sizes.append(c); c *= 2
    from fusezap.device import cuda_kernels as _ck
    if cuda: _ck.set_one_k_order(bool(getenv("FZ_ONE_K", 0)))   # FZ_ONE_K=1: decode GEMMs without split-K (no redo needed)
    # throughput mode: mixed forwards without the exact decode-row arithmetic (no split-K recompute)
    self.mode = "throughput" if thr and self.mix_sizes else "exact"
    self.cublaslt = enable_cublaslt(self) if cuda else {}
    self.gemm_tune = autotune_gemms(self) if cuda else []   # before any forward binds a GEMM config
    exact_mix = self.mode == "exact"
    self._mixes = {C: _Forward(self, C + max_num_seqs, max_num_seqs, True, max_num_seqs, plan=True, alias=True, mix=C, mix_exact=exact_mix)
                   for C in self.mix_sizes}
    if not self.mix_sizes:   # exact mode without mixed forwards: prefill-first scheduling and zero-padding sizing
      # NEURON (N7 serve): 4, so requests arriving together are prefilled before the decoders' next step instead of one
      # prompt per mixed step (each such step also runs the decode forward; Validate's c4 burst in a scheduler sim: TTFT p99 ~190 -> ~140 ms)
      self.cfg.prefill_first = max(0, getenv("FZ_PREFILL_FIRST", 8 if cuda else 4 if self.neuron else 0))
      if getenv("FZ_PREFILL_FIT", 1 if cuda or self.neuron else 0) and len(self.prefill_sizes) > 1:
        self.cfg.prefill_buckets, self.cfg.split_rows = tuple(self.prefill_sizes), getenv("FZ_PREFILL_SPLIT_ROWS", 256)
        if self.neuron:   # I77: by each size's estimated forward time (FZ_NEURON_PREFILL_COST_MS), not its padded rows
          from fusezap.serve.neuron_prefill import prefill_cost_ms
          self.cfg.prefill_cost = prefill_cost_ms(self.prefill_sizes)
          print(f"fusezap: prefill sizes {self.prefill_sizes} chosen by estimated forward ms {self.cfg.prefill_cost} (FZ_NEURON_PREFILL_COST_MS)", flush=True)
    # prefill pipeline (CUDA, FZ_PREFILL_PIPE): a prefill-only step's tokens are read after the next prefill-first step
    # is scheduled, built and launched, so that step's host prep overlaps this one's GPU work
    self.prefill_pipe = bool(cuda and self.cfg.prefill_first and getenv("FZ_PREFILL_PIPE", 1))
    self._pend_p: tuple | None = None
    if self.mix_sizes: self.cfg.mixed_prefill_tokens = self.mix_sizes[-1]
    elif getenv("FZ_DECODE_PREFILL_CAP", 0): self.cfg.mixed_prefill_tokens = getenv("FZ_DECODE_PREFILL_CAP", 0)   # sweeps (round 6)
    self.cfg.prefill_every = max(1, getenv("FZ_PREFILL_EVERY", 1))
    # I76: the serving loops gather a burst of short prompts into one prefill forward (scheduler.AdmitGate): off (0) by
    # default (I77 run 1 on NEURON, one repeat each: 128-token c4 p99 186 -> 141 ms but 40-token c4 p50 116 -> 132); 8 to try
    self.admit_wait_ms = float(os.environ.get("FZ_ADMIT_WAIT_MS") or 0)
    self._ids, self._arrival = itertools.count(), itertools.count()
    self.seqs: dict[int, Sequence] = {}
    self.stats = {"steps": 0, "decode_steps": 0, "prefill_steps": 0, "tokens": 0, "preemptions": 0, "lookahead_steps": 0, "mixed_steps": 0}
    from fusezap.serve.stepprof import from_env
    self._prof = from_env(model.device)   # FZ_STEP_PROF=<file.jsonl>: per-step host / GPU timeline (off: None)
    self._ctx_seen: list | None = None    # FZ_SERVE_PROFILE (fusezap/serve/serveprof.py): the step's (pool prefix, slots needed)
    self._by_key = {f.key: f for f in (*self._decodes.values(), *self._prefills.values(), *self._mixes.values())}

  # ---- tensor parallelism (fusezap/serve/dist.py)
  def tp_spec(self) -> dict:
    """The constructor arguments that give a worker rank this engine's forwards (sizes and mode as resolved here: the
    defaults depend on the device's free memory, which differs per rank)."""
    return {"num_blocks": self.cache.num_blocks, "block_size": self.cache.bs, "max_num_seqs": self.cfg.max_num_seqs,
            "max_prefill_tokens": self.prefill_sizes[-1], "max_model_len": self.max_model_len, "decode_beam": self.decode_beam,
            "prefill_buckets": list(self.prefill_sizes), "mode": self.mode}
  def tp_signature(self) -> tuple:
    """What every rank's engine must agree on: the forwards' shapes and kinds."""
    return (tuple(self.decode_sizes), tuple(self.prefill_sizes), tuple(self.mix_sizes), self.mode, self.device_argmax)
  def mirror_forward(self, key: tuple, ids: np.ndarray, sv: np.ndarray, pos: np.ndarray, meta: np.ndarray, pseq: int, kind: str) -> None:
    """A worker rank: run rank 0's forward `key` on this rank's shard (the logits are kept a few forwards, for lookahead
    rows whose input is this forward's device argmax)."""
    prev = self._seen.get(pseq) if pseq >= 0 else None
    assert pseq < 0 or prev is not None, f"tensor-parallel worker: logits of forward {pseq} are gone"
    lg = self._by_key[key].run(ids, sv, pos, meta, prev, kind)
    self._seen[lg.seq] = lg
    self._seen.pop(lg.seq - 8, None)

  def admit_gate(self, clock=None) -> "AdmitGate | None":
    """The serving loops' burst gate (scheduler.AdmitGate), or None when FZ_ADMIT_WAIT_MS / the device default is 0."""
    if self.admit_wait_ms <= 0: return None
    from fusezap.serve.scheduler import AdmitGate
    return AdmitGate(self.scheduler, self.admit_wait_ms, float(os.environ.get("FZ_ADMIT_RECENT_MS") or 250), clock)

  # ---- requests
  def add_request(self, prompt_ids: Seq[int], params: SamplingParams | None = None, request_id: int | None = None) -> int:
    params = params or SamplingParams()
    rid = next(self._ids) if request_id is None else request_id
    assert rid not in self.seqs, f"request id {rid} is in use"
    if len(prompt_ids) + params.max_tokens > self.max_model_len:
      raise ValueError(f"prompt ({len(prompt_ids)}) + max_tokens ({params.max_tokens}) exceeds max_model_len {self.max_model_len}")
    seq = Sequence(rid, [int(t) for t in prompt_ids], params, arrival=next(self._arrival))
    seq.sampler = params.sampler()
    if params.constraint is not None: seq.cons = params.constraint.matcher(self.model.cfg.vocab_size, sorted(self.stop_ids))   # ConstraintError: a ValueError (400)
    self.scheduler.add(seq, params.max_tokens)
    self.seqs[rid] = seq
    return rid
  def abort(self, request_id: int) -> RequestOutput | None:
    seq = self.seqs.get(request_id)
    if seq is None or seq.status == SeqStatus.FINISHED: return None
    self.scheduler.finish(seq, "abort")
    return self._out(seq, [])
  def has_unfinished(self) -> bool: return self.scheduler.has_unfinished()
  def release(self, request_id: int) -> None:
    """Forget a finished request (its outputs stay available until then)."""
    if (s := self.seqs.get(request_id)) is not None and s.status == SeqStatus.FINISHED: del self.seqs[request_id]

  def _out(self, seq: Sequence, new: list[int]) -> RequestOutput:
    return RequestOutput(seq.seq_id, new, list(seq.output), seq.status == SeqStatus.FINISHED, seq.finish_reason, len(seq.prompt))

  # ---- one engine step
  def step(self) -> list[RequestOutput]:
    pf = self._prof
    if pf is not None: pf.begin()
    outs: list[RequestOutput] = []
    rows_d = 0
    if (PP := self._pend_p) is not None:   # a prefill launched by the previous step, tokens not read yet
      self._pend_p = None
      nxt = self.scheduler.schedule_prefill([s for s, l, _ in PP[0] if l is not None])
      if nxt is not None:
        self.stats["steps"] += 1
        self.stats["preemptions"] = self.scheduler.num_preemptions
        if pf is not None: pf.mark("sched")
        done, lg, np_rows, np_size = self._launch_prefill(nxt)
        self.stats["tokens"] += nxt.num_tokens
        outs = self._emit_done(PP[0], check=True)
        if self._pipe_ok(done, lg): self._pend_p = (done, lg)
        else: outs += self._emit_done(done, check=True)
        if pf is not None: pf.end(0, np_rows, np_size)
        return outs
      outs = self._emit_done(PP[0], check=True)
    if (P := self._pend) is not None:   # a decode launched by the previous step, tokens not read yet
      self._pend = None
      Q = self._speculate(P)
      if Q is not None: rows_d += len(Q.rows)
      outs = self._complete(P)
      if Q is not None:
        self._pend = Q
        if pf is not None: pf.end(rows_d, 0, 0)
        return outs
    so = self.scheduler.schedule()
    self.stats["steps"] += 1
    self.stats["preemptions"] = self.scheduler.num_preemptions
    if so.empty:
      if pf is not None and outs: pf.end(rows_d, 0, 0)
      return outs
    if pf is not None: pf.mark("sched")
    np_rows, np_size = 0, 0
    done: list[tuple[Sequence, _Logits | None, int]] = []    # (seq, the step's logits if it produces a token, row)
    if so.decode and so.prefill and self.mix_sizes and sum(n for _, _, n in so.prefill) <= self.mix_sizes[-1]:
      np_rows = sum(n for _, _, n in so.prefill)
      C = next(c for c in self.mix_sizes if c >= np_rows)
      sel, who, k = [], [], 0
      for s, st, n in so.prefill:
        k += n
        if st + n == s.num_tokens: sel.append(k - 1); who.append(s)
      sel.extend(range(C, C + len(so.decode)))
      lg = self._mixes[C](_Rows.build(so.prefill, C, so.decode), sel)
      self.stats["mixed_steps"] += 1
      self.stats["decode_steps"] += 1
      self.stats["prefill_steps"] += 1
      rows_d, np_size = len(so.decode), C
      for i, s in enumerate(so.decode):
        s.num_computed += 1
        done.append((s, lg if s.num_computed == s.num_tokens else None, len(who) + i))
      for s, st, n in so.prefill:
        s.num_computed = st + n
        done.append((s, lg if s in who else None, who.index(s) if s in who else -1))
      self.stats["tokens"] += so.num_tokens
      for s, l, i in done:
        if l is None: continue
        outs.append(self._emit(s, l.token(i, s)))
      if pf is not None: pf.end(rows_d, np_rows, np_size)
      return outs
    if so.decode:
      rows = [(s.seq_id, s.num_computed, s.tokens[s.num_computed]) for s in so.decode]
      lg = self._decode_fwd(len(rows))(rows, list(range(len(rows))))
      self.stats["decode_steps"] += 1
      rows_d += len(rows)
      for i, s in enumerate(so.decode):
        s.num_computed += 1
        done.append((s, lg if s.num_computed == s.num_tokens else None, i))
      if not so.prefill and self.lookahead and (lg._ids is not None or self.dsteps is None) and all(self._greedy(s) for s in so.decode):
        P = _Pending([(s, l is not None, i) for s, l, i in done], lg)
        self.stats["tokens"] += so.num_tokens
        Q = self._speculate(P)
        if Q is not None: rows_d += len(Q.rows)
        outs += self._complete(P)
        self._pend = Q
        if pf is not None: pf.end(rows_d, 0, 0)
        return outs
    if so.prefill:
      dp, lgp, np_rows, np_size = self._launch_prefill(so)
      if not so.decode and self._pipe_ok(dp, lgp):   # read its tokens next step, after launching the next prefill
        self._pend_p = (dp, lgp)
        self.stats["tokens"] += so.num_tokens
        if pf is not None: pf.end(rows_d, np_rows, np_size)
        return outs
      done += dp
    self.stats["tokens"] += so.num_tokens
    for s, lg, i in done:
      if lg is None: continue
      outs.append(self._emit(s, lg.token(i, s)))
    if pf is not None: pf.end(rows_d, np_rows, np_size)
    return outs

  def _launch_prefill(self, so: SchedulerOutput) -> tuple[list, "_Logits", int, int]:
    """Launch so's prompt chunks as one prefill forward (the smallest prefill size that holds them); returns
    (done rows as in step(), logits, rows, size)."""
    sel, who, k = [], [], 0
    for s, st, n in so.prefill:
      k += n
      if st + n == s.num_tokens: sel.append(k - 1); who.append(s)
    np_size = next(n for n in self.prefill_sizes if n >= k)
    lg = self._prefills[np_size](_Rows.build(so.prefill), sel)
    self.stats["prefill_steps"] += 1
    done, w = [], {id(s): i for i, s in enumerate(who)}
    for s, st, n in so.prefill:
      s.num_computed = st + n
      done.append((s, lg if id(s) in w else None, w.get(id(s), -1)))
    return done, lg, k, np_size

  def _pipe_ok(self, done: list, lg: "_Logits") -> bool:
    """A prefill's tokens can be read after the next forward is launched: the prefill pipeline is on, the ids come
    back by an async copy enqueued right after the forward (the next replay may reuse the logits buffer: alias), every
    row producing a token is greedy (reads only its id) and more prompt work is pending."""
    return (self.prefill_pipe and lg.rb is not None and lg._ids is not None and self.scheduler._prompt_work()
            and all(self._greedy(s) for s, l, _ in done if l is not None))

  def _emit_done(self, done: list, check: bool = False) -> list[RequestOutput]:
    """Emit the tokens of done rows (seq, logits or None, row); check: skip sequences finished or aborted since."""
    return [self._emit(s, l.token(i, s)) for s, l, i in done if l is not None and (not check or s.status == SeqStatus.RUNNING)]

  def _emit(self, s: Sequence, tok: int) -> RequestOutput:
    s.output.append(tok)
    p = s.params
    if (c := s.cons) is not None and tok not in self.stop_ids: c.advance(tok)
    if (not p.ignore_eos and tok in self.stop_ids) or tok in p.stop_token_ids or (c is not None and c.done): self.scheduler.finish(s, "stop")
    elif len(s.output) >= p.max_tokens or s.num_tokens >= self.max_model_len: self.scheduler.finish(s, "length")
    return self._out(s, [tok])

  def _decode_fwd(self, n: int) -> _Forward: return self._decodes[next(b for b in self.decode_sizes if b >= n)]

  # ---- lookahead: the next decode is enqueued before the previous one's tokens are read
  @staticmethod
  def _greedy(s: Sequence) -> bool:
    """A row whose token is the device argmax: greedy, no repetition penalty, unconstrained (a constrained row's token
    needs its logits masked on the host, so lookahead and the prefill pipeline, which feed or read device argmaxes, stop
    while one runs; unconstrained rows of the same steps keep their device argmax)."""
    return s.sampler.greedy and s.sampler.repetition_penalty == 1.0 and s.cons is None

  def _speculate(self, P: "_Pending") -> "_Pending | None":
    """Launch the decode that follows P, before P's tokens are read: the same running sequences, one position further,
    each row fed P's token for it (by its row in P) or its known next token (teacher forcing after a preemption). None
    (nothing launched) when the next step needs the scheduler: waiting requests, prompts in flight, a sequence not in P,
    a non-greedy row, or a sequence that would need a block that is not free. A sequence that P finishes by length is
    left out; one that P finishes by a stop token gets a row whose result is dropped."""
    sch = self.scheduler
    if sch.waiting or (self.dsteps is not None and P.lg._ids is None): return None   # device inputs need device ids
    live = [(s, prod, i) for s, prod, i in P.rows if s.status == SeqStatus.RUNNING]
    if {id(s) for s in sch.running} != {id(s) for s, _, _ in live}: return None
    toks, src, rows = [], [], []
    for s, prod, i in live:
      if not self._greedy(s) or s.in_prompt: return None
      n_tok = s.num_tokens + (1 if prod else 0)          # tokens once P's is appended
      if prod and (len(s.output) + 1 >= s.params.max_tokens or n_tok >= self.max_model_len): continue   # P finishes it
      if s.num_computed >= n_tok: return None
      toks.append((s.seq_id, s.num_computed, 0 if prod and s.num_computed == n_tok - 1 else s.tokens[s.num_computed]))
      src.append(i if prod and s.num_computed == n_tok - 1 else -1)
      rows.append((s, s.num_computed + 1 == n_tok, len(rows)))
    if not rows or not all(self.cache.can_ensure(s.seq_id, s.num_computed + 1) for s, _, _ in rows): return None
    for s, _, _ in rows: self.cache.ensure(s.seq_id, s.num_computed + 1)
    sch.steps += 1
    self.stats["steps"] += 1
    self.stats["decode_steps"] += 1
    self.stats["lookahead_steps"] += 1
    self.stats["tokens"] += len(rows)
    lg = self._decode_fwd(len(toks))(toks, list(range(len(toks))), src, P.lg, kind="s")
    for s, _, _ in rows: s.num_computed += 1
    return _Pending(rows, lg)

  def _complete(self, P: "_Pending") -> list[RequestOutput]:
    """Read P's tokens (waits for P only) and emit them; rows of sequences finished or aborted since are dropped."""
    return [self._emit(s, P.lg.token(i, s)) for s, prod, i in P.rows if prod and s.status == SeqStatus.RUNNING]

  # ---- convenience
  def generate(self, prompts: Iterable[Seq[int]], params: SamplingParams | Seq[SamplingParams] | None = None,
               max_steps: int | None = None) -> list[list[int]]:
    """Run prompts to completion (all admitted up front); returns each one's tokens in prompt order."""
    prompts = list(prompts)
    ps = params if isinstance(params, (list, tuple)) else [params or SamplingParams()] * len(prompts)
    rids = [self.add_request(p, q) for p, q in zip(prompts, ps)]
    n = 0
    while any(self.seqs[r].status != SeqStatus.FINISHED for r in rids):
      self.step()
      n += 1
      if max_steps is not None and n > max_steps: raise RuntimeError(f"not finished after {max_steps} steps")
    res = [list(self.seqs[r].output) for r in rids]
    for r in rids: self.release(r)
    return res
