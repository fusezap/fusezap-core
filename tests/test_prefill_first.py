"""Speed run 4 round 11 (docs/status/speed-run4/c32-analysis.md fixes 2 and 3): prefill-first exact scheduling
(Scheduler.schedule / schedule_prefill, cfg.prefill_first), the prefill pipeline (LLMEngine._pend_p, CUDA), zero-padding
prefill sizing (Scheduler._fit, cfg.prefill_buckets) and the vectorized step inputs (_Rows, PagedKVCache.step_meta_rows).
All are scheduling or host code: a request's tokens must be exactly the plain engine's."""
import os
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve.kv_cache import PagedKVCache
from fusezap.serve.scheduler import SchedulerConfig, SeqStatus
from tests.test_scheduler import _sim
from tests.test_lookahead import _model_path, _device, _run, _script, MODEL

def test_step_meta_rows_matches_step_meta():
  rng = np.random.default_rng(0)
  c = PagedKVCache(1, 1, 4, 256, 4, dtypes.float32, "NPY", max_seq_len=512)
  lens = {3: 37, 5: 9, 8: 64, 11: 1}
  for s, n in lens.items(): c.ensure(s, n)
  for trial in range(20):
    toks = []
    for s, n in lens.items():
      st = int(rng.integers(0, n)); k = int(rng.integers(1, n - st + 1))
      toks += [(s, p) for p in range(st, st + k)]
      if rng.random() < 0.3: toks += [(None, 0)] * int(rng.integers(1, 4))   # padding rows inside the step
    rng.shuffle(toks) if trial % 2 else None
    N = len(toks) + int(rng.integers(0, 5))
    sel = sorted(rng.choice(len(toks), size=min(3, len(toks)), replace=False).tolist())
    ref = c.step_meta(N, 4, toks, sel)
    sid = np.array([-1 if s is None else s for s, _ in toks], np.int64)
    pos = np.array([p for _, p in toks], np.int64)
    np.testing.assert_array_equal(c.step_meta_rows(N, 4, sid, pos, sel), ref)

def test_prefill_first_bounded():
  """While prompts are pending, up to prefill_first consecutive steps run prompt chunks alone (decoders wait), then a
  step with the decode rows; every request finishes with the same tokens as without."""
  base = _sim([30, 30, 30, 30, 30, 30], [9] * 6, 128, cfg=SchedulerConfig(max_num_seqs=8, max_prefill_tokens=8), arrivals={0: 0, 1: 0, 2: 3, 3: 3, 4: 9, 5: 9})[1]
  seen = {"pf_only": 0, "longest_wait": 0}
  wait = {"n": 0}
  def check(sch, so):
    decoders = [s for s in sch.running if not s.in_prompt and not any(s is x for x, _, _ in so.prefill)]
    if so.prefill and not so.decode and decoders:
      seen["pf_only"] += 1; wait["n"] += 1
      assert wait["n"] <= 2, "decoders waited more than prefill_first steps"
    else: wait["n"] = 0
    seen["longest_wait"] = max(seen["longest_wait"], wait["n"])
  cfg = SchedulerConfig(max_num_seqs=8, max_prefill_tokens=8, prefill_first=2)
  sch, seqs, hist = _sim([30, 30, 30, 30, 30, 30], [9] * 6, 128, cfg=cfg, arrivals={0: 0, 1: 0, 2: 3, 3: 3, 4: 9, 5: 9}, check=check)
  assert seen["pf_only"] > 0 and seen["longest_wait"] == 2, seen
  assert [s.output for s in seqs] == [s.output for s in base] and all(s.status == SeqStatus.FINISHED for s in seqs)

def test_zero_padding_sizing():
  """2560 prompt rows with buckets 256..4096: the step runs 2048 (no padding) and the next 512, instead of one 4096-row
  forward with 1536 padding rows; 700 rows still run one 1024-row step (a split would cost more than the padding)."""
  def first(prompts, budget=4096):
    cfg = SchedulerConfig(max_num_seqs=32, max_prefill_tokens=budget, prefill_buckets=(256, 512, 1024, 2048, 4096))
    rows = []
    def check(sch, so):
      if len(rows) < 2: rows.append(sum(n for _, _, n in so.prefill))
    _sim(prompts, [1] * len(prompts), 4096, bs=16, cfg=cfg, check=check)
    return rows
  assert first([512] * 5) == [2048, 512]
  assert first([700])[0] == 700
  assert first([512] * 8) == [4096]
  assert first([300, 300])[0] == 600         # 512 + a 256-row forward + its fixed cost (256) is not below 1024

@pytest.mark.slow
@pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")
def test_prefill_first_tokens_identical():
  """The real engine under a script of arrivals, stops, an abort and a sampled request: prefill-first (and on CUDA the
  prefill pipeline and zero-padding sizing over 256 / 512 / 1024-row prefill forwards) gives exactly the tokens of the
  engine with all three off."""
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine
  cuda = _device().startswith("CUDA")
  m = load_model(MODEL, dtype=dtypes.bfloat16 if cuda else dtypes.float32, device=_device(), max_len=512, prefill_chunk=128)
  def eng(on: bool):
    e = LLMEngine(m, num_blocks=1024, block_size=16, max_num_seqs=8, max_prefill_tokens=1024 if cuda else 256, max_model_len=512)
    e.cfg.prefill_first = 3 if on else 0
    if not on: e.cfg.prefill_buckets = None
    e.prefill_pipe = bool(on and cuda and e.dsteps is not None)
    return e
  base = eng(False)
  ref, _ = _run(base, _script(np.random.default_rng(0), -1))
  stop_tok = ref[9][0][5]
  ref, _ = _run(eng(False), _script(np.random.default_rng(0), stop_tok, sampled=True))
  e = eng(True)
  got, streamed = _run(e, _script(np.random.default_rng(0), stop_tok, sampled=True))
  a, b = got.pop(4)[0], ref.pop(4)[0]          # the abort lands at a different step
  assert a[:len(b)] == b[:len(a)] and a and b
  assert got == ref and all(streamed[k] == got[k][0] for k in got)
  assert e._pend_p is None and not e.has_unfinished() and e.scheduler.num_prefill_first > 0
  assert base.scheduler.num_prefill_first == 0
