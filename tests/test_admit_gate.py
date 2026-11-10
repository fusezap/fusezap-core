"""I76: served TTFT at c4. The burst gate (scheduler.AdmitGate, FZ_ADMIT_WAIT_MS) and the off-chip model of the served
path (bench/serveq_sim.py: the real engine, scheduler and, in --http mode, the HTTP app + ProcEngine + Invarix's client,
with each forward a fixed latency measured on inf2).

  - the gate holds only while replacements are owed, nothing decodes and the prompts leave room in the prefill forward;
  - the virtual-time model reproduces the I75 chip run's served c4 staircase (four 128-token prompts: one 128-row prefill
    each, at pool prefixes 128 / 256 / 512 / 512) and the gate leaves it unchanged, while 40-token prompts share a forward;
  - through AsyncEngine on a real (tiny, CPU) engine the gate batches a burst and the tokens are the engine's own;
  - through the HTTP app and the engine process (real time, scaled-down latencies) the burst shares one forward."""
from __future__ import annotations
import asyncio, os, sys, time
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve.kv_cache import PagedKVCache
from fusezap.serve.scheduler import AdmitGate, Scheduler, SchedulerConfig, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))

class _P:
  def __init__(self, n): self.max_tokens = n

class _Clock:
  def __init__(self): self.t = 0.0
  def __call__(self): return self.t

def _sched(max_prefill: int = 128) -> Scheduler:
  cache = PagedKVCache(1, 1, 4, 64, 16, dtypes.float32, "NPY", max_seq_len=1024)
  return Scheduler(SchedulerConfig(max_num_seqs=4, max_prefill_tokens=max_prefill), cache)

def _add(sch: Scheduler, gate: AdmitGate, i: int, n: int) -> Sequence:
  s = Sequence(i, [5] * n, _P(8), arrival=i)
  sch.add(s, 8)
  gate.added()
  return s

def test_gate_holds_for_owed_short_prompts():
  sch, clk = _sched(), _Clock()
  g = AdmitGate(sch, 8.0, 250.0, clk)
  g.finished(4)                                    # a c4 wave ended: four replacements are owed
  _add(sch, g, 0, 40)
  assert g.hold() == pytest.approx(0.008)          # the window opens at the first add
  clk.t = 0.003; _add(sch, g, 1, 40)
  assert g.hold() == pytest.approx(0.005)          # still the first add's deadline
  clk.t = 0.005; _add(sch, g, 2, 40)
  assert g.hold() == pytest.approx(0.003)          # 120 rows: room left
  clk.t = 0.007; _add(sch, g, 3, 40)
  assert g.hold() == 0.0                           # 160 rows fill the 128-row forward (and nothing more is owed)

def test_gate_deadline_and_no_hold_cases():
  sch, clk = _sched(), _Clock()
  g = AdmitGate(sch, 8.0, 250.0, clk)
  g.finished(4); _add(sch, g, 0, 40)
  clk.t = 0.0081
  assert g.hold() == 0.0                           # the window is over
  # a prompt that fills the forward alone (Invarix short on NEURON: 128 tokens) never waits
  sch, clk = _sched(), _Clock(); g = AdmitGate(sch, 8.0, 250.0, clk)
  g.finished(4); _add(sch, g, 0, 128)
  assert g.hold() == 0.0
  # a lone client (c1): its one replacement is in, nothing more is owed
  sch, clk = _sched(), _Clock(); g = AdmitGate(sch, 8.0, 250.0, clk)
  g.finished(1); _add(sch, g, 0, 40)
  assert g.hold() == 0.0
  # nothing finished recently: no hold (a first request, or an open-loop arrival long after the last finish)
  sch, clk = _sched(), _Clock(); g = AdmitGate(sch, 8.0, 250.0, clk)
  g.finished(4); clk.t = 0.3; _add(sch, g, 0, 40)
  assert g.hold() == 0.0
  # a sequence is decoding: a hold would stall it
  sch, clk = _sched(), _Clock(); g = AdmitGate(sch, 8.0, 250.0, clk)
  s = _add(sch, g, 0, 20)
  so = sch.schedule(); s.num_computed = 20; s.output.append(1)   # its prompt is done: it decodes from now on
  g.finished(4); _add(sch, g, 1, 40)
  assert g.hold() == 0.0
  # off
  sch, clk = _sched(), _Clock(); g = AdmitGate(sch, 0.0, 250.0, clk)
  g.finished(4); _add(sch, g, 0, 40)
  assert g.hold() == 0.0

def test_engine_gate_defaults(monkeypatch):
  path = _tiny_dir()
  monkeypatch.delenv("FZ_ADMIT_WAIT_MS", raising=False)
  eng = _engine(path)
  assert eng.admit_wait_ms == 0 and eng.admit_gate() is None          # off by default (I77 run 1)
  monkeypatch.setenv("FZ_ADMIT_WAIT_MS", "5")
  eng = _engine(path)
  assert eng.admit_wait_ms == 5 and isinstance(eng.admit_gate(), AdmitGate)

_TINY: list = []
def _tiny_dir() -> str:
  if not _TINY:
    import tempfile, serveq_sim as S
    _TINY.append(S.tiny_model(tempfile.mkdtemp(prefix="fz_serveq_")))
  return _TINY[0]

def _engine(path: str, max_prefill: int = 128):
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine
  m = load_model(path, dtype=dtypes.float32, device="CPU", max_len=512, prefill_chunk=128)
  return LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=max_prefill, max_model_len=512, decode_beam=0)

# ---------------------------------------------------------------- the virtual-time model of the served path

def _virtual(monkeypatch, wait: str, in_len: int) -> dict:
  import serveq_sim as S
  monkeypatch.setenv("FZ_ADMIT_WAIT_MS", wait)
  monkeypatch.setenv("FZ_PREFILL_FIRST", "4")
  for k in ("FZ_SIM_PREFILL_MS", "FZ_SIM_DECODE_MS", "FZ_NEURON_PREFILL_CTX"): monkeypatch.delenv(k, raising=False)
  import fusezap.serve.engine as E
  saved = (E.LLMEngine.__init__, E._Forward.run, getattr(E, "_serveq_sim", None))
  clock = S.Clock(virtual=True)
  try:
    if hasattr(E, "_serveq_sim"): del E._serveq_sim
    S.install(clock)
    out = {}
    for c in (1, 4):
      eng = S.build_engine(_tiny_dir())
      out[c] = S.run_virtual(eng, clock, c, 8, in_len, 128, 1.8, 1.8, 2.5, 1.5)
    return out
  finally:
    E.LLMEngine.__init__, E._Forward.run = saved[0], saved[1]
    if saved[2] is None and hasattr(E, "_serveq_sim"): del E._serveq_sim

def test_virtual_model_reproduces_the_chip_staircase(monkeypatch):
  """I75 inf2 served (Invarix short, 128 / 128, c4): TTFTs ~96 / 186 / 282 / 376 ms, p50 234.6, p99 375.5, TPOT p50 39.75,
  96.9 tok/s; c1 TTFT 90.6. The model: one 128-row prefill per prompt, at prefixes 128 / 256 / 512 / 512."""
  r = _virtual(monkeypatch, "0", 128)
  assert r[1]["ttft"] == [90.6] * 8 and r[1]["tpot_p50"] == 38.5
  assert r[4]["ttft"][:4] == [90.6, 180.1, 274.1, 368.1]
  assert abs(r[4]["ttft_p99"] - 375.5) < 15 and abs(r[4]["ttft_p50"] - 234.6) < 15 and abs(r[4]["tpot_p50"] - 39.75) < 0.5
  g = _virtual(monkeypatch, "8", 128)
  assert g[4] == r[4] and g[1] == r[1]             # 128-token prompts fill a forward alone: the gate never holds

def test_virtual_gate_batches_short_prompts(monkeypatch):
  r, g = _virtual(monkeypatch, "0", 40), _virtual(monkeypatch, "8", 40)
  assert g[1] == r[1]                              # c1 unchanged
  assert r[4]["ttft_p50"] > 170 and g[4]["ttft_p50"] < 110   # three of four prompts share the first forward
  assert g[4]["ttft_p99"] < r[4]["ttft_p99"] + 12 and abs(g[4]["tok_s"] - r[4]["tok_s"]) < 1.0

# ---------------------------------------------------------------- AsyncEngine on a real tiny engine

def test_async_engine_gate_batches_a_burst_with_the_same_tokens(monkeypatch):
  from fusezap.serve.api import AsyncEngine
  from fusezap.serve.engine import SamplingParams
  path = _tiny_dir()
  prompts = [[(7 * i + j) % 90 + 3 for j in range(20 + 3 * i)] for i in range(3)]
  ref = _engine(path).generate(prompts[:1], SamplingParams(max_tokens=6, ignore_eos=True)) + \
        _engine(path).generate(prompts[1:2], SamplingParams(max_tokens=6, ignore_eos=True)) + \
        _engine(path).generate(prompts[2:], SamplingParams(max_tokens=6, ignore_eos=True))
  monkeypatch.setenv("FZ_ADMIT_WAIT_MS", "300")
  eng = _engine(path)
  ae = AsyncEngine(eng)
  ae.gate.finished(3)                               # as after a wave of three
  ae.start()
  async def go():
    hs = []
    for p in prompts:
      hs.append(ae.submit(p, SamplingParams(max_tokens=6, ignore_eos=True)))
      await asyncio.sleep(0.02)                     # the burst arrives 20 ms apart, inside the 300 ms window
    outs = []
    for h in hs:
      toks = []
      while True:
        o = await h.q.get()
        toks += o.new_token_ids
        if o.finished: break
      outs.append(toks)
    return outs
  try: got = asyncio.run(go())
  finally: ae.stop()
  assert got == ref
  assert eng.stats["prefill_steps"] == 1           # one prefill forward for the three prompts

# ---------------------------------------------------------------- the HTTP app + engine process, real time (scaled down)

def test_http_engine_process_gate(monkeypatch):
  import serveq_sim as S
  monkeypatch.setenv("FZ_SIM_PREFILL_MS", "128:40,256:40,512:40,1024:40")
  monkeypatch.setenv("FZ_SIM_DECODE_MS", "512:4,1024:4")
  monkeypatch.setenv("FZ_PREFILL_FIRST", "4")
  res = {}
  for w in ("0", "30"):
    monkeypatch.setenv("FZ_ADMIT_WAIT_MS", w)
    res[w] = S.run_http(_tiny_dir(), [4], 4, 30, 8, 4, 64, 18476 + int(w))["c4"]
  assert res["0"]["errors"] == 0 and res["30"]["errors"] == 0
  # without the gate the first prompt is prefilled alone and the others wait for its forward; with it the burst shares one
  assert sorted(res["0"]["ttft"])[1] > 70, res
  assert max(res["30"]["ttft"]) < 85, res

# ---------------------------------------------------------------- the chip job (--serveq) and its summary

def test_serveq_job_and_summary(tmp_path):
  import csv, subprocess
  sh = os.path.join(ROOT, "scripts", "nki_n8_prefill_inf2.sh")
  assert subprocess.run(["bash", "-n", sh]).returncode == 0
  src = open(sh).read()
  assert '"--serveq"' in src and "serveq_http serveq_old none FZ_ADMIT_WAIT_MS=0" in src and "serveq_http serveq auto" in src
  assert "--ttft-lens 0,128" in src and "short40" in src
  cols = ["workload", "concurrency", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "output_tok_s", "errors"]
  def write(lab, rows):
    d = tmp_path / f"fz_tp2_bf16_7b_{lab}_perf"; d.mkdir()
    with open(d / "performance.csv", "w", newline="") as f:
      w = csv.writer(f); w.writerow(cols); w.writerows(rows)
  write("serveq_old", [["short", 1, 90.6, 92.7, 38.6, 25.7, 0], ["short", 4, 234.6, 375.5, 39.75, 96.9, 0],
                       ["short40", 1, 90.5, 92.0, 38.6, 25.7, 0], ["short40", 4, 183.0, 185.2, 38.98, 99.8, 0]])
  write("serveq", [["short", 1, 90.8, 92.5, 38.6, 25.6, 0], ["short", 4, 233.0, 376.0, 39.8, 96.8, 0],
                   ["short40", 1, 90.9, 92.2, 38.6, 25.6, 0], ["short40", 4, 98.0, 186.7, 39.65, 99.7, 0]])
  run = lambda: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_n8_serveq_summary.py"), str(tmp_path)], capture_output=True, text=True).stdout
  out = run()
  assert "serveq perf bar: PROMOTE" in out, out
  write_bad = tmp_path / "fz_tp2_bf16_7b_serveq_perf" / "performance.csv"
  write_bad.write_text(write_bad.read_text().replace("98.0,186.7", "170.0,186.7"))
  assert "DO NOT PROMOTE" in run()
