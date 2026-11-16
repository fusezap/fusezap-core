"""I69 follow-up: the served first token's host-side switches (fusezap/serve/hostpath.py) on a tiny CPU engine.
  - FZ_SERVE_HTTP / FZ_SERVE_ACCESS_LOG map to uvicorn's options; `fast` without httptools / uvloop is an error;
  - FZ_SERVE_EARLY_HEADERS=1 streams the same tokens and text as the default; an engine failure before the first token is
    then the stream's error event (status 200) instead of an HTTP error;
  - FZ_SERVE_GC=freeze freezes the collector's objects (and is off by default);
  - FZ_REQ_TRACE's engine line carries first_out, and scripts/nki_i69_ttft_breakdown.py splits a TTFT into its parts.
"""
from __future__ import annotations
import gc, json, os, subprocess, sys
import numpy as np
import pytest
from fusezap.serve import hostpath as HP

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_uvicorn_opts(monkeypatch):
  for k in ("FZ_SERVE_HTTP", "FZ_SERVE_ACCESS_LOG", "FZ_SERVE_EARLY_HEADERS", "FZ_SERVE_GC"): monkeypatch.delenv(k, raising=False)
  assert HP.uvicorn_opts() == {} and not HP.early_headers() and HP.gc_mode() == "0"   # defaults: uvicorn's own choice, nothing changed
  monkeypatch.setenv("FZ_SERVE_HTTP", "h11"); monkeypatch.setenv("FZ_SERVE_ACCESS_LOG", "0")
  assert HP.uvicorn_opts() == {"http": "h11", "loop": "asyncio", "access_log": False}
  assert "http h11 + asyncio (FZ_SERVE_HTTP=h11), access log off" in HP.describe()
  monkeypatch.setenv("FZ_SERVE_HTTP", "fast")
  import importlib.util as U
  real = U.find_spec
  monkeypatch.setattr(U, "find_spec", lambda name, *a: None if name in ("httptools", "uvloop") else real(name, *a))
  with pytest.raises(RuntimeError, match="httptools and uvloop"): HP.uvicorn_opts()
  monkeypatch.setattr(U, "find_spec", lambda name, *a: object() if name in ("httptools", "uvloop") else real(name, *a))
  assert HP.uvicorn_opts() == {"http": "httptools", "loop": "uvloop", "access_log": False}
  for k, v in (("FZ_SERVE_HTTP", "turbo"), ("FZ_SERVE_GC", "off")):
    monkeypatch.setenv(k, v)
    with pytest.raises(ValueError): HP.http_mode() if k == "FZ_SERVE_HTTP" else HP.gc_mode()
    monkeypatch.delenv(k)

def test_gc_settle(monkeypatch):
  monkeypatch.delenv("FZ_SERVE_GC", raising=False)
  assert not HP.gc_settle()
  monkeypatch.setenv("FZ_SERVE_GC", "freeze")
  calls = []
  class G:   # a tensor-parallel group on rank 0: the freeze is mirrored on the workers through op "call"
    world, rank = 2, 0
    import contextlib
    @contextlib.contextmanager
    def spmd(self, op, *args): calls.append((op, *args)); yield
  try:
    assert HP.gc_settle(G()) and gc.get_freeze_count() > 0
    assert calls == [("call", "fusezap.serve.hostpath:_freeze")]
  finally: gc.unfreeze()

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
  sys.path.insert(0, ROOT)
  from bench.serveq_sim import tiny_model
  return tiny_model(str(tmp_path_factory.mktemp("hostpath_tiny")))

def _app(path: str):
  from fusezap.dtype import dtypes
  from fusezap.serve.api import create_app
  from fusezap.serve.engine import LLMEngine
  from fusezap.serve.models import load_model
  from fusezap.serve.tokenizer import Tokenizer
  m = load_model(path, dtype=dtypes.float32, device="CPU", max_len=256, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=128, max_model_len=256, decode_beam=0)
  return create_app(eng, Tokenizer.from_pretrained(path), "tiny"), eng

def _stream(c, body: dict) -> tuple[int, list]:
  with c.stream("POST", "/v1/completions", json=body) as r:
    if r.status_code != 200: return r.status_code, [json.loads(r.read())]
    return 200, [json.loads(l[5:]) for l in r.iter_lines() if l.startswith("data:") and l[5:].strip() != "[DONE]"]

@pytest.mark.parametrize("chat", [False, True], ids=["completions", "chat"])
def test_early_headers_same_stream(tiny, monkeypatch, chat):
  from starlette.testclient import TestClient
  outs = {}
  for flag in ("0", "1"):
    monkeypatch.setenv("FZ_SERVE_EARLY_HEADERS", flag)
    app, _ = _app(tiny)
    with TestClient(app) as c:
      if chat:
        with c.stream("POST", "/v1/chat/completions", json={"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                                                            "temperature": 0, "stream": True, "ignore_eos": True}) as r:
          assert r.status_code == 200
          evs = [json.loads(l[5:]) for l in r.iter_lines() if l.startswith("data:") and l[5:].strip() != "[DONE]"]
        outs[flag] = [[ch for e in evs for ch in e["choices"]], len(evs)]
      else:
        st, evs = _stream(c, {"model": "tiny", "prompt": [5, 6, 7, 8, 9] * 6, "max_tokens": 8, "temperature": 0, "stream": True, "ignore_eos": True,
                              "stream_options": {"include_usage": True}})
        assert st == 200
        outs[flag] = [[ch for e in evs for ch in e["choices"]], [e.get("usage") for e in evs if e.get("usage")]]
        assert outs[flag][1] == [{"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38}]
  # the same events (the tiny byte tokenizer decodes most random ids to no text; the chunks, finish reasons and usage still match)
  assert outs["0"] == outs["1"] and outs["0"][0], outs

def test_early_headers_engine_error_in_stream(tiny, monkeypatch):
  """An engine failure before the first token: an HTTP error by default, the stream's error event with early headers."""
  from starlette.testclient import TestClient
  got = {}
  for flag in ("0", "1"):
    monkeypatch.setenv("FZ_SERVE_EARLY_HEADERS", flag)
    app, eng = _app(tiny)
    def boom(*a, **k): raise RuntimeError("injected step failure")
    monkeypatch.setattr(eng, "step", boom)
    with TestClient(app) as c:
      got[flag] = _stream(c, {"model": "tiny", "prompt": [5, 6, 7], "max_tokens": 4, "stream": True})
  assert got["0"][0] >= 500 and "injected" in json.dumps(got["0"][1]), got["0"]
  assert got["1"][0] == 200 and "error" in got["1"][1][-1] and "injected" in got["1"][1][-1]["error"]["message"], got["1"]

def test_trace_first_out_and_breakdown(tiny, tmp_path, monkeypatch):
  """The engine process's trace line has first_out between sched and the API's first output, and the breakdown script
  splits each client TTFT into parts that add up to it (bench/serveq_sim.py: the engine process, the real app, Invarix's
  client in another process)."""
  pre = str(tmp_path / "tr")
  env = {**os.environ, "FZ_REQ_TRACE": pre, "FZ_SIM_PREFILL_MS": "128:8,1024:8", "FZ_SIM_DECODE_MS": "1024:2", "PYTHONPATH": ROOT}
  import socket
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "bench", "serveq_sim.py"), "--serve", "--port", str(port)], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  try:
    import httpx, time
    t0 = time.time()
    while True:
      try:
        if httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=1).status_code == 200: break
      except Exception: pass
      assert time.time() - t0 < 120 and srv.poll() is None, "serveq_sim --serve did not start"
      time.sleep(0.3)
    js = str(tmp_path / "cl.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "bench", "serveq_sim.py"), "--client", f"http://127.0.0.1:{port}", "--conc", "1", "-n", "6",
                        "--out-len", "4", "--json", js], env={**env, "FZ_REQ_TRACE": ""}, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
  finally:
    srv.terminate(); srv.wait(30)
  eng = [json.loads(l) for l in open(pre + ".engine.jsonl")]
  assert eng and all(d["sched"] <= d["first_out"] <= d["fin"] for d in eng if "sched" in d), eng[:2]
  sys.path.insert(0, os.path.join(ROOT, "scripts"))
  import nki_i69_ttft_breakdown as B
  recs, miss = B.load(js, pre, 1)
  assert len(recs) == 6 and miss == 0
  for d in recs:
    assert abs(sum(d[k] for k in B.PARTS) - d["ttft"]) < 0.2 and d["step"] >= 7.5, d   # the parts add up; the step holds the 8 ms prefill

def test_i69_ttft_job_mode_and_summary(tmp_path):
  """scripts/nki_n8_prefill_inf2.sh --i69-ttft: bash -n, its results dir, budget, launch guard (worst case <= $4), the served
  configurations in order; scripts/nki_i69_ttft_summary.py's choice on synthetic result files: the fastest bitwise
  configuration whose greedy texts equal base's (one that differs, one with a TPOT regression and one whose device gather
  fell back are not chosen; act is reported, never chosen)."""
  import csv
  sh = os.path.join(ROOT, "scripts", "nki_n8_prefill_inf2.sh")
  assert subprocess.run(["bash", "-n", sh]).returncode == 0
  src = open(sh).read()
  start = src.index('if [ "$I69T" = 1 ]; then')
  blk = src[start:src.index("\nfi\n", start)]
  assert "results/nki_i69_ttft" in src and "FZ_PF_BUDGET_MIN:-75" in src
  line = '--max-hours 1.4 --max-usd 3.2 --hf-token -- "bash scripts/nki_n8_prefill_inf2.sh --i69-ttft"'
  assert line in src and (1.4 + 0.17) * 1.996 <= 3.2 <= 4.0
  order = [blk.index(f"ttft_variant {v} ") for v in ("base", "http", "host", "embed", "act", "base2")]
  assert order == sorted(order) and "FZ_SERVE_HTTP=h11" in blk and "FZ_NEURON_EMBED_DEV=1" in blk and "FZ_NEURON_PFK_EXP=act" in blk
  assert "FZ_NEURON_PREFILL_SIZES=128" in blk and "FZ_REQ_TRACE=" in blk and "nki_i69_ttft_summary.py" in blk
  out = tmp_path / "o"; out.mkdir()
  V = [("base", "bitwise", "FZ_SERVE_HTTP=h11", 52.2, 38.9, "same"), ("http", "bitwise", "FZ_SERVE_HTTP=fast", 51.6, 38.8, "same"),
       ("host", "bitwise", "FZ_SERVE_HTTP=fast FZ_SERVE_EARLY_HEADERS=1", 50.9, 38.8, "same"),
       ("embed", "bitwise", "... FZ_NEURON_EMBED_DEV=1", 49.8, 38.8, "same"), ("slow", "bitwise", "x", 49.0, 39.9, "same"),
       ("odd", "bitwise", "y", 48.0, 38.8, "diff"), ("act", "act", "... FZ_NEURON_PFK_EXP=act", 47.0, 38.8, "diff")]
  with open(out / "ttft_variants.tsv", "w") as f:
    for lab, kind, env, *_ in V: f.write(f"{lab}\t{kind}\t{env}\n")
  for lab, kind, env, t1, tp, tok in V:
    d = out / f"fz_tp2_bf16_7b_ttft_{lab}_perf"; d.mkdir()
    with open(d / "performance.csv", "w") as f:
      w = csv.DictWriter(f, ["workload", "concurrency", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "output_tok_s", "errors"]); w.writeheader()
      w.writerow({"workload": "short", "concurrency": 1, "ttft_p50_ms": t1, "ttft_p99_ms": t1 + 3, "tpot_p50_ms": tp, "output_tok_s": 25.6, "errors": 0})
      w.writerow({"workload": "short", "concurrency": 4, "ttft_p50_ms": 140, "ttft_p99_ms": 220, "tpot_p50_ms": tp + 0.7, "output_tok_s": 99, "errors": 0})
    json.dump({"texts": {"1": "a" if tok == "same" else "b", "2": "c"}}, open(out / f"tokens_{lab}.json", "w"))
    if lab in ("embed", "act"):
      a = out / f"fz_tp2_bf16_7b_ttft_{lab}_acc"; a.mkdir()
      json.dump({"decision": {"label": "PROMOTE"}, "accuracy": {"top1_agreement": 0.9987}}, open(a / "report.json", "w"))
  (out / "server_tp2_bf16_7b_ttft_slow.log").write_text("fusezap: ready\n")
  r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_i69_ttft_summary.py"), str(out), "--json", str(out / "s.json")],
                     capture_output=True, text=True)
  assert r.returncode == 0, r.stderr
  S = json.load(open(out / "s.json"))
  assert S["chosen"] == "embed", r.stdout
  assert "CHOSEN embed" in r.stdout and "AHEAD" in r.stdout and "not bitwise, reported only: act" in r.stdout
  (out / "server_tp2_bf16_7b_ttft_embed.log").write_text("fusezap: FZ_NEURON_EMBED_DEV off: the device gather failed its check (x)\n")
  r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_i69_ttft_summary.py"), str(out), "--json", str(out / "s.json")],
                     capture_output=True, text=True)
  assert json.load(open(out / "s.json"))["chosen"] == "host" and "FELL BACK" in r.stdout, r.stdout
