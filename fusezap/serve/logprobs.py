"""Log-probabilities for `fusezap serve` (OpenAI `logprobs` / `top_logprobs`, `echo` prompt logprobs).

  sc = scorer(engine)                       # one per LLMEngine (cached on it)
  r = sc.score(ids, start=len(prompt), k=5) # positions start..len(ids)-1: logprob of ids[i] given ids[:i], and the top k

They are computed from the model's own logits by a teacher-forced scoring pass: one full-sequence forward of the model
(`Llama.forward`, no KV cache: causal attention over the whole sequence) over prompt + generated tokens, then
log_softmax in float32 on the device. The pass is jitted per power-of-two length bucket (>= 16 rows; padding rows sit
causally after every real row, so they change nothing) and runs on the engine's thread or process between steps, so
the paged engine, its scheduler and its kernels are not touched. It is the same model and weights as generation, but
not the same kernels (the engine's decode rows are BEAM-tuned GEMVs), so a logprob can differ from the generation
step's own logit by rounding; at a near tie the top-1 shown may then differ from the token the engine sampled.

Memory: the bucket's forward stops before the final norm (`Llama.forward(head=False)`: (T, D)); the final norm,
lm_head and log_softmax (`Llama.head`) then run on fixed chunks of `chunk` rows (FZ_LOGPROBS_CHUNK, default 256; a
bucket below that is one chunk), each copied to the host and reduced to what the request needs (the scored token's
logprob and the top k) before the next one runs. Peak extra memory is chunk x vocab float32 (logits and log_softmax on
the device, one copy on the host) whatever the prompt length; a whole 2048-row bucket of a 152k vocabulary was 1.25 GB
per tensor. Rows are independent, so the values do not depend on the chunking; chunks holding no requested row are
skipped. On CUDA and NEURON (N8) the jitted steps' intermediates share memory (Jit plan, as the engine's prefill steps): unplanned,
every bucket a server had scored kept its whole activation set allocated for good. After an out-of-memory error
(another process may hold most of the GPU) the scorer drops its jitted steps, hands the allocator's cached blocks back
to the driver and retries once (a second failure releases again before raising), so a failed pass leaves nothing
allocated behind for the next request.
"""
from __future__ import annotations
import contextlib, gc, os
from typing import Any, Iterator, Sequence
import numpy as np

MAX_TOP = 20   # OpenAI's (and vLLM's default) cap on top_logprobs
CHUNK = 256    # rows per final norm + lm_head + log_softmax chunk (FZ_LOGPROBS_CHUNK)

class _NoCache:
  """KVCache interface without a cache: forward attends over the T rows it is given (bench/bench_ppl.py's NLLEval)."""
  def state(self): return []
  def inputs(self): return []
  def bind(self, state, inputs=None): return self
  def load(self, state): pass
  def update(self, layer, k, v, start, T): return k, v

class Scorer:
  def __init__(self, model: Any, max_len: int, chunk: int | None = None):
    from fusezap.serve.cuda import is_cuda
    self.m, self.max_len, self.cache = model, max_len, _NoCache()
    self.chunk = int(chunk or os.environ.get("FZ_LOGPROBS_CHUNK") or CHUNK)
    assert self.chunk >= 16 and self.chunk & (self.chunk - 1) == 0, f"logprobs chunk {self.chunk}: a power of two >= 16"
    self.jits: dict[int, Any] = {}        # bucket -> jitted trunk (Llama.forward without the head)
    self.head_jits: dict[int, Any] = {}   # chunk rows -> jitted final norm + lm_head + log_softmax
    from fusezap.serve.neuron_kv import is_neuron
    from fusezap.device.neuron import neuron_plan
    neuron = is_neuron(model.device)
    # jitted intermediates share memory (as the engine's prefill steps on CUDA; NEURON too since N8, FZ_NEURON_PLAN)
    self.plan = is_cuda(model.device) or (neuron and neuron_plan())
    # N8 (I71): NEURON keeps at most FZ_LOGPROBS_JITS trunk buckets captured (default 2; 0 = no limit, the default elsewhere):
    # each holds its planned arena on the device, and a 7B bucket of 512 rows is ~0.3 GB next to 13.2 GiB of weights
    self.max_jits = int(os.environ.get("FZ_LOGPROBS_JITS") or (2 if neuron else 0))
  def _bucket(self, n: int) -> int:
    b = max(16, 1 << (n - 1).bit_length())
    return min(b, getattr(self.m, "table_len", b))
  def _trunk(self, x: Any) -> Any:
    return self.m.forward(x, self.cache, 0, x.shape[1], last_only=False, head=False)[0]   # (T, D), before the final norm
  def _head(self, h: Any) -> Any:
    return self.m.head(h).float().log_softmax(-1)                                          # (rows, V) float32

  def release(self) -> None:
    """Drop the jitted steps (and the device memory their captures hold) and hand the allocator's cached free blocks
    back to the driver."""
    from fusezap.device import Device
    self.jits.clear()
    self.head_jits.clear()
    gc.collect()
    alloc = getattr(Device[self.m.device], "allocator", None)
    if hasattr(alloc, "free_cache"): alloc.free_cache()

  def _check(self, n: int) -> int:
    """The bucket for n tokens; ValueError if they cannot be scored."""
    if not 1 <= n <= self.max_len: raise ValueError(f"cannot score {n} tokens (max_model_len {self.max_len})")
    b = self._bucket(n)
    if b < n: raise ValueError(f"cannot score {n} tokens (the model's tables hold {b})")
    return b
  def chunks(self, ids: Sequence[int], r0: int = 0, r1: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (first row, rows) covering the log-probability rows [r0, r1) (default every row): rows is (m, vocab)
    float32 numpy, and row j of a sequence is log_softmax of the logits after ids[:j+1]. The device computes fixed
    `chunk`-row slices of the bucket, so a row's values never depend on which rows were asked for."""
    n = len(ids)
    b = self._check(n)
    done, r1 = r0, n if r1 is None else min(r1, n)
    for attempt in (0, 1):
      err = None
      try:
        for lo, rows in self._chunks(ids, b, done, r1):
          yield lo, rows
          done = lo + rows.shape[0]
        return
      except MemoryError as e:
        err = MemoryError(str(e))   # no traceback: the failed pass's frames (and their buffers) die with `e` here
      self.release()
      if attempt: raise err

  def _chunks(self, ids: Sequence[int], b: int, r0: int, r1: int) -> Iterator[tuple[int, np.ndarray]]:
    from fusezap.jit import Jit
    from fusezap.tensor import Tensor
    if r0 >= r1: return
    n, C = len(ids), min(self.chunk, b)
    if b in self.jits: self.jits[b] = self.jits.pop(b)   # most recently used last
    else:
      while self.max_jits and len(self.jits) >= self.max_jits: self.jits.pop(next(iter(self.jits)))   # drop the least recently used bucket
      self.jits[b] = Jit(self._trunk, plan=self.plan)
    if C not in self.head_jits: self.head_jits[C] = Jit(self._head, plan=self.plan)
    emb = self.m.embed(np.asarray(ids, np.int64)[None])
    buf = np.zeros((1, b, emb.shape[-1]), dtype=emb.dtype)
    buf[:, :n] = emb
    with self.m._opt_env(prefill=True):
      h = self.jits[b](Tensor(buf, device=self.m.device, dtype=self.m.dtype))
      del buf, emb
      for s in range(r0 // C * C, r1, C):
        rows = self.head_jits[C](h.shrink(((s, s + C), None))).numpy()
        lo, hi = max(r0, s), min(r1, s + C)
        yield lo, rows[lo - s:hi - s]
        del rows

  def logprobs(self, ids: Sequence[int]) -> np.ndarray:
    """(len(ids), vocab) float32: row j is log_softmax of the logits after ids[:j+1]. The whole matrix, for tools and
    tests (scripts/diag_top1_layers.py); serving uses `score`, which keeps only what a request needs."""
    return np.concatenate([rows for _, rows in self.chunks(ids)], 0)

  def score(self, ids: Sequence[int], start: int, k: int) -> dict:
    """Positions i in [max(start, 1), len(ids)): {"logprobs": [logprob of ids[i]], "top": [[(id, logprob)] * k]} (top
    sorted by logprob, ties by id). start=0 asks for every position; position 0 has no logprob (nothing before it), so
    the lists then start at position 1 and the caller puts None first."""
    tp = getattr(self.m, "tp", None)   # tensor parallelism: the workers run the same pass (its trunk all-reduces)
    if tp is not None: self._check(len(ids))   # refused here, before the workers start it
    with tp.spmd("score", list(ids), start, k) if tp is not None else contextlib.nullcontext(): return self._score(ids, start, k)
  def _score(self, ids: Sequence[int], start: int, k: int) -> dict:
    k = max(0, min(int(k), MAX_TOP))
    lo = max(start, 1)
    out_lp, top = [], []
    for r0, rows in self.chunks(ids, lo - 1, len(ids) - 1):
      for j, row in enumerate(rows):
        i = r0 + j + 1   # row i - 1 predicts token i
        out_lp.append(float(row[ids[i]]))
        if k:
          idx = np.argpartition(-row, k - 1)[:k] if k < row.shape[0] else np.arange(row.shape[0])
          idx = sorted(idx.tolist(), key=lambda t: (-float(row[t]), t))
          top.append([(int(t), float(row[t])) for t in idx])
        else: top.append([])
    return {"start": lo, "logprobs": out_lp, "top": top}

def scorer(engine: Any) -> Scorer:
  sc = getattr(engine, "_fz_scorer", None)
  if sc is None:
    sc = Scorer(engine.model, engine.max_model_len)
    engine._fz_scorer = sc
  return sc

# ---------------------------------------------------------------- response shapes

def token_str(tok: Any, tid: int, as_ids: bool) -> str:
  """A token's key in a logprobs response: its decoded text (vLLM / OpenAI), or "token_id:<id>" with vLLM's
  return_tokens_as_token_ids."""
  return f"token_id:{tid}" if as_ids else tok.decode([tid], skip_special_tokens=False)

def text_offsets(tok: Any, ids: Sequence[int], base: int = 0) -> list[int]:
  """Character offset of each token in the decoded text of ids (special tokens kept, as in the echoed prompt)."""
  out, prev = [], 0
  for j in range(len(ids)):
    out.append(base + prev)
    prev = len(tok.decode(list(ids[:j + 1]), skip_special_tokens=False)) if j + 1 < len(ids) else prev
  return out

def completion_logprobs(tok: Any, ids: Sequence[int], sc: dict | None, n_echo: int, as_ids: bool, text_base: int = 0) -> dict:
  """/v1/completions `logprobs` object for the tokens `ids` (the echoed prompt tokens first when n_echo > 0, then the
  generated ones). sc: Scorer.score over the full sequence; positions without a logprob (the first prompt token) get
  None, as in vLLM / OpenAI. The sampled token is always in its top_logprobs dict (so up to k + 1 entries)."""
  toks = [token_str(tok, t, as_ids) for t in ids]
  tlp: list[float | None] = []
  top: list[dict | None] = []
  first = len(ids) - len(sc["logprobs"]) if sc else len(ids)
  for j, t in enumerate(ids):
    if j < first: tlp.append(None); top.append(None); continue
    lp, tk = sc["logprobs"][j - first], sc["top"][j - first]
    d: dict[str, float] = {}
    for tid, v in tk:
      key = token_str(tok, tid, as_ids)
      if key not in d: d[key] = v
    d.setdefault(toks[j], lp)
    tlp.append(lp); top.append(d)
  return {"tokens": toks, "token_logprobs": tlp, "top_logprobs": top, "text_offset": text_offsets(tok, ids, text_base)}

def chat_logprobs(tok: Any, ids: Sequence[int], sc: dict, k: int) -> dict:
  """/v1/chat/completions `logprobs` object: {"content": [{token, logprob, bytes, top_logprobs: [...k]}]}."""
  def ent(tid: int, lp: float) -> dict:
    s = tok.decode([tid], skip_special_tokens=False)
    return {"token": s, "logprob": lp, "bytes": list(s.encode("utf-8"))}
  content = []
  for j, t in enumerate(ids):
    lp = sc["logprobs"][j]
    content.append({**ent(t, lp), "top_logprobs": [ent(tid, v) for tid, v in sc["top"][j][:k]]})
  return {"content": content}
