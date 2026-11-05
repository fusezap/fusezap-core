"""N7 serve: FZ_SERVE_PROFILE (fusezap/serve/serveprof.py) and scripts/serve_prof_summary.py, on a tiny random checkpoint on
CPU: the engine loop writes one line per step (time in the step, between steps, kind, tokens), the handlers one per
request (lag behind the engine, time per output); off without the variable."""
from __future__ import annotations
import asyncio, importlib.util, json, os
import pytest
from fusezap.dtype import dtypes
from test_tp import tiny_model_dir

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class _Tok:   # the handlers' tokenizer: decode only
  def decode(self, ids, skip_special_tokens=False): return "".join(f"<{i}>" for i in ids)

def _summary_mod():
  spec = importlib.util.spec_from_file_location("serve_prof_summary", os.path.join(ROOT, "scripts", "serve_prof_summary.py"))
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  return m

def test_off_without_the_variable(monkeypatch):
  from fusezap.serve import serveprof as SP
  for v in (None, "", "0"):
    if v is None: monkeypatch.delenv("FZ_SERVE_PROFILE", raising=False)
    else: monkeypatch.setenv("FZ_SERVE_PROFILE", v)
    assert SP.from_env() is None
  monkeypatch.setenv("FZ_SERVE_PROFILE", "1")
  assert SP.path_from_env() == "fusezap_serve_profile.jsonl"

def test_serve_profile_lines_and_summary(tmp_path, monkeypatch):
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.api import Server
  path = tmp_path / "prof.jsonl"
  monkeypatch.setenv("FZ_SERVE_PROFILE", str(path))
  m = load_model(tiny_model_dir(str(tmp_path / "tiny")), dtype=dtypes.float32, device="CPU", max_len=128, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
  s = Server(eng, _Tok(), "tiny")
  assert s.prof is s.async_engine.prof is not None   # one profile file object for the engine loop and the handlers
  prompts = [[1, 2, 3, 4, 5], [7, 8, 9], list(range(20, 40))]
  async def one(p):
    toks = []
    async for _, fin, n, new in s._gen(p, SamplingParams(max_tokens=6, ignore_eos=True), []): toks += new
    return toks
  async def go(): return await asyncio.gather(*[one(p) for p in prompts])
  s.async_engine.start()
  try: got = asyncio.run(go())
  finally: s.async_engine.stop()
  assert [len(t) for t in got] == [6, 6, 6]
  assert got == eng.generate(prompts, SamplingParams(max_tokens=6, ignore_eos=True))   # profiling changes no token
  recs = [json.loads(l) for l in open(path)]
  steps, reqs = [r for r in recs if r["k"] == "step"], [r for r in recs if r["k"] == "req"]
  assert sum(r["n"] for r in steps) == 18 and len(reqs) == 3
  assert {r["kind"] for r in steps} >= {"d"} and any("p" in r["kind"] for r in steps)
  for r in steps:
    assert r["ms"] >= 0 and r["cpu"] >= 0 and r["out"] >= 0 and "ctx" not in r and "dev" not in r   # CPU: no pool prefix, no runtime worker
  assert any(r["gap"] is not None for r in steps)
  assert all(r["n"] == 6 and r["chunks"] >= 1 and r["lag"][0] is not None and r["hms"][0] is not None for r in reqs)
  S = _summary_mod()
  res = S.summarize(recs, idle=5.0)
  assert len(res["periods"]) == 1 and res["periods"][0]["outputs"] == 18 and res["requests"]["n"] == 3
  text = S.report(res, "t")
  assert "serve_prof t period 0:" in text and "serve_prof t http (whole file): 3 requests" in text
