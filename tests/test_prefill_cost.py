"""I77 follow-up: the prefill size by prompt length (Scheduler._fit_cost, neuron_prefill.prefill_cost_ms). With several
prefill sizes (FZ_NEURON_PREFILL_SIZES=128,512) a step covers the oldest pending prompt by the cheapest sequence of
forwards (estimated ms per forward), not always the largest: 200 tokens as two 128-row forwards, 400+ as one 512; a burst
of short prompts is not bundled into a 512-row forward (I78).

  1. the scheduler alone (no model): the rows per step for prompts of several lengths and bursts, cost sizing off
     (None) keeps the old schedule;
  2. the cost table: defaults per exp mode, FZ_NEURON_PREFILL_COST_MS, interpolation, bad values;
  3. LLMEngine on NKISHIM (sizes 32 / 64, the prefill kernels exact): the forward sizes the steps ran, and the greedy tokens
     equal to the schedule without cost sizing (FZ_PREFILL_FIT=0) and to the members (v1, 32 only): scheduling only.
"""
from __future__ import annotations
import pytest
from fusezap.serve.scheduler import SchedulerConfig
from fusezap.serve import neuron_prefill as PF
from test_scheduler import _sim

COST = {128: 50.0, 512: 172.0}

def _rows(prompts: list[int], cost: dict | None, max_tokens: int = 2, arrivals: dict | None = None) -> list[int]:
  """Prompt rows per step that ran a prefill (cache large enough: no preemption)."""
  cfg = SchedulerConfig(max_num_seqs=4, max_prefill_tokens=512, prefill_buckets=(128, 512) if cost else None, prefill_cost=cost)
  _, _, hist = _sim(prompts, [max_tokens] * len(prompts), num_blocks=1024, bs=16, cfg=cfg, arrivals=arrivals)
  return [sum(n for _, _, n in so.prefill) for so in hist if so.prefill]

def test_cost_sizing_by_prompt_length():
  assert _rows([200], COST) == [128, 72]            # 2 x 128 rows ~100 ms < one 512 ~172
  assert _rows([200], None) == [200]                # off: one step (a 512-row forward)
  assert _rows([400], COST) == [400]                # 4 x 128 ~200 ms > one 512
  assert _rows([512], COST) == [512]
  assert _rows([100], COST) == [100]                # fits the smallest
  assert _rows([300], COST) == [128, 128, 44]       # 150 < 172
  assert _rows([300], {128: 43.0, 512: 130.0}) == [128, 128, 44]   # act's table: 3 x 43 = 129 < 130
  assert _rows([450], {128: 43.0, 512: 130.0}) == [450]

def test_cost_sizing_bursts_and_long_prompts():
  assert _rows([128] * 4, COST) == [128] * 4        # a burst of 4 x 128: one by one, never bundled into a 512-row forward (I78)
  assert _rows([100] * 5, COST) == [128] * 3 + [116]   # short prompts pack 128-row forwards
  assert _rows([400, 60, 30], COST) == [490]         # a long oldest prompt picks 512 rows; the others fill its spare rows
  assert _rows([40] * 4, COST) == [128, 32]         # 4 x 40 = 160: 128 + 32 rows (100 < 172)
  assert _rows([1000], COST) == [512, 488]          # 172 + 172 < 172 + 4 x 50
  assert _rows([600], COST) == [512, 88]           # 512 + 88: 222 = 128 + 472 (a tie: the bigger forward first)
  assert _rows([700], COST) == [512, 128, 60]
  assert _rows([200, 200], COST, arrivals={1: 1}) == [128, 128, 128, 16]   # the second arrives after the first step: 272, then 144 pending

def test_cost_table(monkeypatch):
  monkeypatch.delenv("FZ_NEURON_PREFILL_COST_MS", raising=False)
  assert PF.prefill_cost_ms([128, 512], "exact") == {128: 53.0, 512: 195.0}
  assert PF.prefill_cost_ms([128, 512], "act") == {128: 43.0, 512: 130.0}
  c = PF.prefill_cost_ms([128, 256, 1024], "act")   # interpolated / extrapolated linearly in the rows
  assert c[128] == 43.0 and c[256] == pytest.approx(43 + 87 / 3) and c[1024] == pytest.approx(130 + 87 / 384 * 512)
  monkeypatch.setenv("FZ_NEURON_PREFILL_COST_MS", "128:40,512:100")
  assert PF.prefill_cost_ms([128, 512], "exact") == {128: 40.0, 512: 100.0}
  monkeypatch.setenv("FZ_NEURON_PREFILL_COST_MS", "512:150")   # one size overridden, the other from the defaults
  assert PF.prefill_cost_ms([128, 512], "exact") == {128: 53.0, 512: 150.0}
  for bad in ("128-50", "128:x", "0:5", "128:-1"):
    monkeypatch.setenv("FZ_NEURON_PREFILL_COST_MS", bad)
    with pytest.raises(ValueError): PF.prefill_cost_ms([128, 512], "exact")

def test_engine_cost_sizing_nkishim_tokens_unchanged(tmp_path, monkeypatch):
  """LLMEngine with prefill forwards of 32 and 64 rows on NKISHIM (the prefill kernels, exact exp): with a cost table that
  makes two 32-row forwards cheaper than one 64-row one every step runs 32 rows; FZ_PREFILL_FIT=0 runs the 64-row forward
  for the steps that need it (the schedule before); the members (v1: 32 rows only) too. The greedy tokens are the same."""
  import numpy as np
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from test_nki_prefill import _model
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  for k in ("FZ_NEURON_PFK_HEAD", "FZ_NEURON_PFK_BANKS", "FZ_NEURON_PFK_WN", "FZ_NEURON_PFK_EVICT", "FZ_NEURON_PFK_SELFTEST", "FZ_PREFILL_FIT"):
    monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "32")
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "exact")
  monkeypatch.setenv("FZ_NEURON_PREFILL_COST_MS", "32:10,64:30")
  ps = [[int(t) for t in np.random.default_rng(40 + i).integers(0, 199, n)] for i, n in enumerate((40, 7, 50))]
  toks, sizes = {}, {}
  try:
    for run, mode, fit in (("cost", "v2", None), ("largest", "v2", "0"), ("members", "v1", None)):
      PF._STATE.update(ok=None, why="", override=None, knobs=None)
      monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", mode)
      if fit is None: monkeypatch.delenv("FZ_PREFILL_FIT", raising=False)
      else: monkeypatch.setenv("FZ_PREFILL_FIT", fit)
      m, _ = _model(tmp_path, seed=5, layers=1)
      e = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=64, decode_beam=0)
      used, orig = [], e._launch_prefill
      def launch(so, orig=orig, used=used):
        r = orig(so); used.append(r[3]); return r
      e._launch_prefill = launch
      if run == "cost": assert e.cfg.prefill_cost == {32: 10.0, 64: 30.0} and e.cfg.prefill_buckets == (32, 64)
      else: assert e.cfg.prefill_cost is None
      toks[run] = e.generate(ps, SamplingParams(max_tokens=5, ignore_eos=True))
      sizes[run] = used
  finally: PF._STATE.update(ok=None, why="", override=None, knobs=None)
  assert set(sizes["cost"]) == {32} and 64 in sizes["largest"] and set(sizes["members"]) == {32}, sizes
  assert toks["cost"] == toks["largest"] == toks["members"] and len({t for o in toks["cost"] for t in o}) > 3

def test_cost_sizing_only_where_set():
  """CUDA / CPU keep their schedule: the cost table is NEURON's (LLMEngine sets it only there); the scheduler default is off."""
  assert SchedulerConfig().prefill_cost is None
  assert _rows([300], None) == [300]

def test_chunk_check_in_process(tmp_path, monkeypatch):
  """scripts/nki_n6_serve.py chunk_check (the --pfk2 job's bitwise check): per prompt length the forwards each schedule ran
  (auto = cost sizing, small, large), and the last prompt token's logits bits: equal across the schedules and to the
  members' (v1, the smallest size only), stable over the reps; the engine's own schedule is restored after."""
  import os, sys
  import numpy as np
  from fusezap.serve.engine import LLMEngine
  from test_nki_prefill import _model
  sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
  import nki_n6_serve as S6
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  for k in ("FZ_NEURON_PFK_HEAD", "FZ_NEURON_PFK_BANKS", "FZ_NEURON_PFK_WN", "FZ_NEURON_PFK_EVICT", "FZ_NEURON_PFK_SELFTEST", "FZ_PREFILL_FIT"):
    monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "32")
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "exact")
  monkeypatch.setenv("FZ_NEURON_PREFILL_COST_MS", "32:10,64:30")
  ref = {"prompts": [{"prompt_ids": [int(t) for t in np.random.default_rng(3).integers(0, 199, 23)]}]}
  out = {}
  try:
    for mode in ("v2", "v1"):
      PF._STATE.update(ok=None, why="", override=None, knobs=None)
      monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", mode)
      m, _ = _model(tmp_path, seed=6, layers=1)
      e = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
      before = (e.cfg.prefill_cost, e.cfg.prefill_buckets)
      out[mode] = S6.chunk_check(e, ref, [40, 64], 3, reps=2)
      assert (e.cfg.prefill_cost, e.cfg.prefill_buckets) == before and e._launch_prefill.__func__ is LLMEngine._launch_prefill
  finally: PF._STATE.update(ok=None, why="", override=None, knobs=None)
  v2, v1 = out["v2"], out["v1"]
  assert sorted(v2["40"]) == ["auto", "large", "small"] and sorted(v1["40"]) == ["auto"]
  assert v2["40"]["auto"]["sizes"] == [32, 32] and v2["40"]["small"]["sizes"] == [32, 32] and v2["40"]["large"]["sizes"] == [64]
  assert v2["64"]["large"]["sizes"] == [64] and v1["64"]["auto"]["sizes"] == [32, 32]
  for L in ("40", "64"):
    shas = {r["logits_sha"] for r in v2[L].values()} | {v1[L]["auto"]["logits_sha"]}
    assert len(shas) == 1 and None not in shas, (L, shas)
    assert all(r["stable"] and len(r["ttft_ms"]) == 2 and len(r["ids"]) == 3 for r in list(v2[L].values()) + [v1[L]["auto"]])
    assert v2[L]["large"]["ids"] == v1[L]["auto"]["ids"]
