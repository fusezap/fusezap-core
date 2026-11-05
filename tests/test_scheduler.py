"""Continuous batching (P4-SERVE): the scheduler alone (simulated steps, no model), then LLMEngine on Qwen2.5-0.5B.

Gate:
  - 32 concurrent greedy requests with different prompts and lengths each produce exactly the tokens of running that
    request alone (and alone matches the contiguous-cache Llama.generate);
  - with a deliberately small block pool, preemption happens and every request still completes with the same tokens,
    each within a bounded number of steps.
The model tests run on FZ_DEVICE (default METAL) and skip if the weights are not in the HF cache.
"""
from __future__ import annotations
import os
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.device import Device
from fusezap.serve.kv_cache import PagedKVCache
from fusezap.serve.scheduler import Scheduler, SchedulerConfig, Sequence, SeqStatus

def _device() -> str: return os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")

# ---------------------------------------------------------------- scheduler simulation (no model)

class _P:
  def __init__(self, max_tokens: int): self.max_tokens = max_tokens

def _sim(prompts: list[int], max_tokens: list[int], num_blocks: int, bs: int = 4, cfg: SchedulerConfig | None = None,
         max_steps: int = 10_000, arrivals: dict[int, int] | None = None, check=None):
  """Drive a Scheduler like LLMEngine.step does (a generated token is a function of the sequence id and its length).
  Checks the block/budget invariants after every schedule() and returns (scheduler, seqs, per-step outputs).
  arrivals: request index -> step at which it is added (default: all at step 0)."""
  cfg = cfg or SchedulerConfig(max_num_seqs=4, max_prefill_tokens=8)
  cache = PagedKVCache(1, 1, 4, num_blocks, bs, dtypes.float32, "NPY", max_seq_len=1 << 12)
  sch = Scheduler(cfg, cache)
  seqs = [Sequence(i, [7] * n, _P(m), arrival=i) for i, (n, m) in enumerate(zip(prompts, max_tokens))]
  arrivals = arrivals or {}
  history = []
  for step in range(max_steps):
    for i, s in enumerate(seqs):
      if arrivals.get(i, 0) == step: sch.add(s, s.params.max_tokens)
    if not sch.has_unfinished() and step >= max(arrivals.values(), default=0): break
    so = sch.schedule()
    history.append(so)
    # invariants
    owned = [b for t in cache.tables.values() for b in t]
    assert len(owned) == len(set(owned)) and len(owned) + cache.num_free_blocks == num_blocks
    assert set(cache.tables) <= {s.seq_id for s in sch.running}
    assert len(sch.running) <= cfg.max_num_seqs
    assert so.num_tokens <= cfg.token_budget and sum(n for _, _, n in so.prefill) <= cfg.max_prefill_tokens
    ids = [s.seq_id for s in so.decode] + [s.seq_id for s, _, _ in so.prefill]
    assert len(ids) == len(set(ids)), "a sequence was scheduled twice in one step"
    for s in so.decode: cache.slot(s.seq_id, s.num_computed)
    for s, st, n in so.prefill:
      assert st == s.num_computed and st + n <= len(s.prompt)
      cache.slot(s.seq_id, st + n - 1)
    for v in so.preempted: assert v.status == SeqStatus.WAITING and v.seq_id not in cache.tables and v.num_computed == 0
    if check: check(sch, so)
    # "run" the step
    done = [(s, s.num_computed + 1) for s in so.decode] + [(s, st + n) for s, st, n in so.prefill]
    for s, nc in done:
      s.num_computed = nc
      if nc == s.num_tokens:
        s.output.append(1000 * s.seq_id + len(s.output))
        if len(s.output) >= s.params.max_tokens: sch.finish(s, "length")
  else: raise AssertionError(f"not finished after {max_steps} steps")
  return sch, seqs, history

def test_admission_budget_and_chunked_prefill():
  cfg = SchedulerConfig(max_num_seqs=3, max_prefill_tokens=8)
  sch, seqs, hist = _sim([20, 3, 5, 9, 2], [4, 6, 2, 3, 5], num_blocks=64, cfg=cfg)
  assert all(s.status == SeqStatus.FINISHED and len(s.output) == s.params.max_tokens for s in seqs)
  assert [s.output for s in seqs] == [[1000 * i + k for k in range(s.params.max_tokens)] for i, s in enumerate(seqs)]
  first = hist[0]
  assert [(s.seq_id, st, n) for s, st, n in first.prefill] == [(0, 0, 8)]       # the 20-token prompt is chunked 8+8+4
  assert [(s.seq_id, st, n) for s, st, n in hist[1].prefill] == [(0, 8, 8)]
  assert [(s.seq_id, st, n) for s, st, n in hist[2].prefill][:2] == [(0, 16, 4), (1, 0, 3)]  # leftover budget admits more
  assert any(so.decode and so.prefill for so in hist), "no step mixed prefill and decode"
  assert sch.num_preemptions == 0

def test_mixed_prefill_budget_while_decoding():
  """Speed run 4: with mixed_prefill_tokens set, a step that has decode rows takes at most that many prompt tokens (the
  engine's mixed forwards); a step without decode rows still takes the full prefill budget."""
  cfg = SchedulerConfig(max_num_seqs=4, max_prefill_tokens=32, mixed_prefill_tokens=8)
  seen = {"mixed": 0, "full": 0}
  def check(sch, so):
    n = sum(k for _, _, k in so.prefill)
    if so.decode: assert n <= 8; seen["mixed"] += bool(n)
    elif n > 8: seen["full"] += 1
  _sim([30, 30, 30, 30, 30], [6, 6, 6, 6, 6], 64, cfg=cfg, arrivals={0: 0, 1: 0, 2: 3, 3: 5, 4: 9}, check=check)
  assert seen["mixed"] > 0 and seen["full"] > 0, seen

def test_prefill_every_n_steps_while_decoding():
  """Speed run 4 round 6: with prefill_every = 3, a step that has decode rows takes prompt tokens only when the scheduler's
  step count is a multiple of 3; with nothing decoding every step prefills, and every request still finishes."""
  cfg = SchedulerConfig(max_num_seqs=4, max_prefill_tokens=16, prefill_every=3)
  seen = {"skipped": 0, "with": 0, "solo": 0}
  def check(sch, so):
    n = sum(k for _, _, k in so.prefill)
    if so.decode and sch.steps % 3: assert n == 0; seen["skipped"] += 1
    elif so.decode and n: seen["with"] += 1
    elif n: seen["solo"] += 1
  _sim([30, 12, 40, 7, 25], [9, 9, 9, 9, 9], 64, cfg=cfg, arrivals={0: 0, 1: 2, 2: 4, 3: 6, 4: 9}, check=check)
  assert all(seen.values()), seen

def test_fcfs_head_of_line_not_skipped():
  # the pool holds only one 30-token request at a time: request 1 (big) must run before the small request 2
  sch, seqs, hist = _sim([10, 30, 2], [2, 2, 2], num_blocks=8, bs=4, cfg=SchedulerConfig(max_num_seqs=4, max_prefill_tokens=40))
  assert seqs[1].first_step < seqs[2].first_step
  assert seqs[0].finish_step <= seqs[1].finish_step <= seqs[2].finish_step

def test_preemption_recompute_and_no_starvation():
  rng = np.random.default_rng(0)
  n = 24
  prompts, maxt = rng.integers(1, 30, n).tolist(), rng.integers(1, 25, n).tolist()
  seen_running = []
  def check(sch, so):
    for v in so.preempted:   # only a newer sequence than every one that keeps running ahead of it is preempted
      assert any(s.arrival < v.arrival for s in sch.running) or not sch.running
    seen_running.append(len(sch.running))
  cfg = SchedulerConfig(max_num_seqs=8, max_prefill_tokens=16)
  sch, seqs, hist = _sim(prompts, maxt, num_blocks=16, bs=4, cfg=cfg, check=check)   # 64 slots for 24 requests
  assert sch.num_preemptions > 0
  assert all(s.status == SeqStatus.FINISHED for s in seqs)
  assert [s.output for s in seqs] == [[1000 * i + k for k in range(m)] for i, m in enumerate(maxt)]
  # bounded: nobody waits longer than all the work queued ahead of it run serially, twice (recompute included)
  work = [-(-p // cfg.max_prefill_tokens) + m for p, m in zip(prompts, maxt)]
  for i, s in enumerate(seqs): assert s.finish_step <= 2 * sum(work[:i + 1]) + 2, (i, s.finish_step, s.num_preemptions)
  assert seqs[0].num_preemptions == 0   # the oldest request never loses its blocks

def test_late_arrivals_and_capacity():
  sch, seqs, _ = _sim([5, 6, 7, 8], [3, 3, 3, 3], num_blocks=8, bs=4, arrivals={0: 0, 1: 2, 2: 2, 3: 9})
  assert all(s.status == SeqStatus.FINISHED for s in seqs)
  cache = PagedKVCache(1, 1, 4, 4, 4, dtypes.float32, "NPY", max_seq_len=64)
  sch = Scheduler(SchedulerConfig(), cache)
  with pytest.raises(ValueError): sch.add(Sequence(0, [1] * 10, _P(7), 0), 7)    # 17 tokens > 16 slots
  with pytest.raises(ValueError): sch.add(Sequence(1, [], _P(1), 1), 1)

def test_prefill_sizes():
  from fusezap.serve.engine import prefill_sizes
  assert prefill_sizes("CUDA", 2048) == [256, 512, 1024, 2048] and prefill_sizes("CUDA:1", 768) == [384, 768]
  assert prefill_sizes("CUDA", 256) == [256] and prefill_sizes("CUDA", 128) == [128] and prefill_sizes("CUDA", 1000) == [500, 1000]
  for d in ("METAL", "CPU", "NPY"): assert prefill_sizes(d, 2048) == [2048] and prefill_sizes(d, 256) == [256]

def test_prefill_budget():
  from fusezap.serve.engine import prefill_budget, prefill_row_bytes
  from fusezap.serve.models.llama import LlamaConfig
  q7 = LlamaConfig(3584, 18944, 28, 28, 4, 128, 152064)   # Qwen2.5-7B
  rb = prefill_row_bytes(q7, 2)
  GiB = 1 << 30
  assert prefill_budget(64 * GiB, rb) == 4096     # H100 80 GB alone, after 15 GB of weights and the KV pool (speed run 4)
  assert prefill_budget(22 * GiB, rb) == 2048     # the same with vLLM holding 45% of it (the Validate config)
  assert prefill_budget(9 * GiB, rb) == 1024      # 24 GB card (review: 2048 ran out of memory at warmup)
  assert prefill_budget(4 * GiB, rb) == 512 and prefill_budget(2 * GiB, rb) == 256 and prefill_budget(0, rb) == 256
  q32 = LlamaConfig(5120, 27648, 64, 40, 8, 128, 152064)  # Qwen2.5-32B on an H100: fp8 rows cost about twice bf16's
  r16, r8 = prefill_row_bytes(q32, 2), prefill_row_bytes(q32, 2, "fp8")
  assert prefill_row_bytes(q7, 2, "int8") == rb and 2 * r16 < r8 < 2.5 * r16
  assert prefill_budget(40 * GiB, r16) == 2048 and prefill_budget(40 * GiB, r8) == 1024   # 2048 fp8 rows ran out of memory

def test_finish_frees_blocks():
  cache = PagedKVCache(1, 1, 4, 8, 4, dtypes.float32, "NPY", max_seq_len=64)
  sch = Scheduler(SchedulerConfig(max_num_seqs=2, max_prefill_tokens=16), cache)
  a, b = Sequence(0, [1] * 9, _P(4), 0), Sequence(1, [1] * 3, _P(4), 1)
  sch.add(a, 4); sch.add(b, 4)
  so = sch.schedule()
  assert len(so.prefill) == 2 and cache.num_free_blocks == 8 - 3 - 1
  sch.finish(a, "abort")
  assert cache.num_free_blocks == 8 - 1 and a not in sch.running and a.finish_reason == "abort"

# ---------------------------------------------------------------- LLMEngine on the real model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
def _model_path() -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(MODEL, allow_download=False)
  except Exception: return None
needs_model = pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")

# the engine shape shared with bench/bench_batch.py (same decode/prefill row counts, so the BEAM-tuned kernels are shared)
ENGINE = dict(block_size=16, max_num_seqs=32, max_prefill_tokens=256)

@pytest.fixture(scope="module")
def model():
  from fusezap.serve.models import load_model
  return load_model(MODEL, dtype=dtypes.float32, device=_device(), max_len=512, prefill_chunk=128)

@pytest.fixture(scope="module")
def engine(model):
  from fusezap.serve.engine import LLMEngine
  return LLMEngine(model, num_blocks=1024, **ENGINE)

@pytest.fixture(scope="module")
def workload():
  """32 different prompts, 1..300 tokens, and different output lengths."""
  rng = np.random.default_rng(42)
  lens = [1, 2, 7, 300, 16, 17, 31, 33, 64, 100, 129, 255, 3, 250, 5, 90] + rng.integers(1, 200, 16).tolist()
  prompts = [rng.integers(100, 30000, n).tolist() for n in lens]
  maxt = rng.integers(8, 40, 32).tolist()
  return prompts, maxt

@pytest.fixture(scope="module")
def alone(engine, workload):
  from fusezap.serve.engine import SamplingParams
  prompts, maxt = workload
  out = [engine.generate([p], SamplingParams(max_tokens=m, ignore_eos=True))[0] for p, m in zip(prompts, maxt)]
  assert [len(o) for o in out] == maxt
  return out

@needs_model
@pytest.mark.slow   # builds LLMEngines; minutes of BEAM tuning on a cold ~/.cache/fusezap
def test_alone_matches_contiguous_generate(model, workload, alone):
  from fusezap.serve.sampling import Sampler
  prompts, maxt = workload
  for i in (2, 4, 5):     # a paged run alone == the P4-MODEL contiguous path (which is token-identical to HF)
    model.reset()
    assert list(model.generate(prompts[i], maxt[i], Sampler(), stop=[])) == alone[i], f"request {i}"

@needs_model
@pytest.mark.slow   # builds LLMEngines; minutes of BEAM tuning on a cold ~/.cache/fusezap
def test_32_concurrent_greedy_match_alone(engine, workload, alone):
  from fusezap.serve.engine import SamplingParams
  prompts, maxt = workload
  steps0, dec0, pre0 = engine.stats["steps"], engine.stats["decode_steps"], engine.stats["prefill_steps"]
  rids = [engine.add_request(p, SamplingParams(max_tokens=m, ignore_eos=True)) for p, m in zip(prompts, maxt)]
  streamed: dict[int, list[int]] = {r: [] for r in rids}
  while engine.has_unfinished():
    outs = engine.step()
    for o in outs:
      streamed[o.request_id] += o.new_token_ids
      assert o.token_ids == streamed[o.request_id]
  got = [engine.seqs[r].output for r in rids]
  for i, (g, a) in enumerate(zip(got, alone)): assert g == a, f"request {i} (prompt {len(prompts[i])} tokens) differs from running alone"
  assert [streamed[r] for r in rids] == got
  assert all(engine.seqs[r].finish_reason == "length" for r in rids)
  steps = engine.stats["steps"] - steps0
  # mixing: fewer forward launches than steps*2 means steps ran decode and prefill together at least sometimes
  assert engine.stats["decode_steps"] - dec0 + engine.stats["prefill_steps"] - pre0 > steps
  assert steps < sum(maxt)       # batched, not serial
  for r in rids: engine.release(r)

@needs_model
@pytest.mark.slow   # builds LLMEngines; minutes of BEAM tuning on a cold ~/.cache/fusezap
def test_small_pool_preempts_and_completes(model, workload, alone):
  from fusezap.serve.engine import LLMEngine, SamplingParams
  prompts, maxt = workload
  eng = LLMEngine(model, num_blocks=24, **ENGINE)     # 384 token slots for 32 requests that need ~4,000 in total
  rids = [eng.add_request(p, SamplingParams(max_tokens=m, ignore_eos=True)) for p, m in zip(prompts, maxt)]
  n = 0
  while eng.has_unfinished():
    eng.step()
    n += 1
    assert n < 5000, "did not finish"
  seqs = [eng.seqs[r] for r in rids]
  assert eng.scheduler.num_preemptions > 0 and any(s.num_preemptions for s in seqs)
  for i, (s, a) in enumerate(zip(seqs, alone)):
    assert s.status == SeqStatus.FINISHED and s.output == a, f"request {i} (preempted {s.num_preemptions}x) differs from running alone"
  # bounded: request i finishes within the serial work of requests 0..i, twice over (recompute included)
  P = ENGINE["max_prefill_tokens"]
  work = [-(-len(p) // P) + m for p, m in zip(prompts, maxt)]
  for i, s in enumerate(seqs): assert s.finish_step <= 2 * sum(work[:i + 1]) + 2, (i, s.finish_step, s.num_preemptions)
  assert seqs[0].num_preemptions == 0

@needs_model
@pytest.mark.slow   # builds LLMEngines; minutes of BEAM tuning on a cold ~/.cache/fusezap
def test_abort_and_stop_tokens(engine, workload, alone):
  from fusezap.serve.engine import SamplingParams
  prompts, maxt = workload
  a = engine.add_request(prompts[3], SamplingParams(max_tokens=maxt[3], ignore_eos=True))
  b = engine.add_request(prompts[4], SamplingParams(max_tokens=maxt[4], ignore_eos=True))
  stop_tok = alone[5][3]
  c = engine.add_request(prompts[5], SamplingParams(max_tokens=maxt[5], stop_token_ids=[stop_tok]))
  for _ in range(3): engine.step()
  free0 = engine.cache.num_free_blocks
  out = engine.abort(a)
  assert out is not None and out.finished and out.finish_reason == "abort" and engine.cache.num_free_blocks > free0
  while engine.has_unfinished(): engine.step()
  assert engine.seqs[b].output == alone[4]
  k = alone[5].index(stop_tok)
  assert engine.seqs[c].output == alone[5][:k + 1] and engine.seqs[c].finish_reason == "stop"
  assert engine.abort(b) is None
  for r in (a, b, c): engine.release(r)

@needs_model
@pytest.mark.slow   # builds LLMEngines; minutes of BEAM tuning on a cold ~/.cache/fusezap
def test_seeded_sampling_batched_matches_alone(engine, workload):
  from fusezap.serve.engine import SamplingParams
  prompts, _ = workload
  ps = [SamplingParams(max_tokens=12, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1, seed=i, ignore_eos=True) for i in range(6)]
  solo = [engine.generate([prompts[i]], ps[i])[0] for i in range(6)]
  ps = [SamplingParams(max_tokens=12, temperature=0.8, top_k=40, top_p=0.9, repetition_penalty=1.1, seed=i, ignore_eos=True) for i in range(6)]
  assert engine.generate(prompts[:6], ps) == solo

@needs_model
@pytest.mark.slow   # builds LLMEngines; minutes of BEAM tuning on a cold ~/.cache/fusezap
def test_prefill_selects_rows_when_prefill_tokens_eq_num_seqs(model, workload, alone):
  # regression: with max_prefill_tokens == max_num_seqs the prefill forward skipped the logits-row selection and a
  # finishing prompt got the logits of token row 0, 1, ... instead of its last prompt token
  from fusezap.serve.engine import LLMEngine, SamplingParams
  prompts, maxt = workload
  eng = LLMEngine(model, num_blocks=64, block_size=16, max_num_seqs=8, max_prefill_tokens=8)
  idx = [14, 1, 2, 0]                 # 5, 2, 7, 1 prompt tokens: several prompts finish in one prefill step
  got = eng.generate([prompts[i] for i in idx], [SamplingParams(max_tokens=6, ignore_eos=True) for _ in idx])
  for i, g in zip(idx, got): assert g == alone[i][:6], f"request {i} (prompt {len(prompts[i])} tokens)"
