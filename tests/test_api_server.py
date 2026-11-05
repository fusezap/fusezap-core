"""P4-API gate: the real server (uvicorn on a free port, Qwen2.5-0.5B-Instruct on METAL) driven by the official openai client.

  - chat completion (non-stream) returns the same text as LLMEngine greedy; streaming yields incremental deltas and finishes;
  - /v1/completions, /v1/models, /health; stop sequences, max_tokens, usage, seeded sampling, error format;
  - 8 concurrent streaming requests all complete with exactly their alone-greedy text.
Skips if the model is not in the HF cache or no METAL device (FZ_DEVICE overrides).
"""
from __future__ import annotations
from typing import Any
import asyncio, json, os, socket, threading, time
import httpx
import pytest
from fusezap.device import Device
from fusezap.dtype import dtypes
from fusezap.serve.api import create_app, stop_scan, warmup
from fusezap.serve.engine import LLMEngine, SamplingParams
from fusezap.serve.models import load_model
from fusezap.serve.tokenizer import Tokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
QUESTIONS = ["What is the capital of France?", "Write a haiku about the sea.", "Name three primary colors.", "Say hello in Spanish.",
             "What is 12 times 12?", "Give one tip for sleeping well.", "Who wrote Hamlet?", "Name a planet with rings."]

def test_stop_scan():
  assert stop_scan("hello", [], False) == ("hello", False, "")
  assert stop_scan("hello wor", ["world"], False) == ("hello ", False, "wor")
  assert stop_scan("hello world!", ["world"], False) == ("hello ", True, "")
  assert stop_scan("hello wor", ["world"], True) == ("hello wor", False, "")
  assert stop_scan("a\n\nb", ["\n\n", "b"], False) == ("a", True, "")

@pytest.fixture(scope="module")
def srv():
  dev = os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
  try: tok = Tokenizer.from_pretrained(MODEL); m = load_model(MODEL, dtype=dtypes.float32, device=dev, max_len=512, prefill_chunk=128)
  except Exception as e: pytest.skip(f"model unavailable: {e}")
  # BEAM-tuned decode kernels as `fusezap serve` uses them; CUDA's serve default is no beam (a cold BEAM tune there takes 30+ min)
  eng = LLMEngine(m, num_blocks=1024, block_size=16, max_num_seqs=32, max_prefill_tokens=256, decode_beam=0 if dev.split(":")[0].upper() == "CUDA" else 4)
  warmup(eng)
  def ref(ids: list[int], n: int) -> str:     # LLMEngine greedy, before the server owns the engine
    out = eng.generate([ids], SamplingParams(max_tokens=n))[0]
    return tok.decode([t for t in out if t not in eng.stop_ids], skip_special_tokens=True)
  chat = lambda q: tok.apply_chat_template([{"role": "user", "content": q}], True)
  refs = {q: ref(chat(q), 40) for q in QUESTIONS}
  refs["The capital of France is"] = ref(tok.encode("The capital of France is", add_special_tokens=False), 12)
  import uvicorn
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  server = uvicorn.Server(uvicorn.Config(create_app(eng, tok, MODEL), host="127.0.0.1", port=port, log_level="warning"))
  th = threading.Thread(target=server.run, daemon=True); th.start()
  t0 = time.time()
  while not server.started:
    assert time.time() - t0 < 60 and th.is_alive(), "server did not start"
    time.sleep(0.05)
  yield {"url": f"http://127.0.0.1:{port}", "refs": refs, "tok": tok, "eng": eng}
  server.should_exit = True; th.join(timeout=30)

@pytest.fixture(scope="module")
def client(srv):
  from openai import OpenAI
  return OpenAI(base_url=srv["url"] + "/v1", api_key="x", timeout=300)

def chat_kw(q: str, **kw): return dict(model=MODEL, messages=[{"role": "user", "content": q}], temperature=0, **kw)

@pytest.mark.slow
def test_health_and_models(srv, client):
  assert httpx.get(srv["url"] + "/health").json() == {"status": "ok"}
  ms = client.models.list().data
  assert [m.id for m in ms] == [MODEL]
  assert client.models.retrieve(MODEL).id == MODEL
  r = httpx.get(srv["url"] + "/v1/models")   # the serving mode (speed run 4): exact unless --mode throughput
  assert r.json()["data"][0]["fusezap_mode"] == "exact" and r.headers["x-fusezap-mode"] == "exact"
  assert httpx.get(srv["url"] + "/health").headers["x-fusezap-mode"] == "exact"

@pytest.mark.slow
def test_chat_nonstream_matches_engine(srv, client):
  q = QUESTIONS[0]
  r = client.chat.completions.create(**chat_kw(q, max_tokens=40))
  assert r.object == "chat.completion" and r.model == MODEL
  c = r.choices[0]
  assert c.message.role == "assistant" and c.message.content == srv["refs"][q]
  n = len(srv["tok"].apply_chat_template([{"role": "user", "content": q}], True))
  assert r.usage.prompt_tokens == n and 0 < r.usage.completion_tokens <= 40
  assert r.usage.total_tokens == n + r.usage.completion_tokens
  assert c.finish_reason == ("length" if r.usage.completion_tokens == 40 else "stop")

@pytest.mark.slow
def test_chat_stream_incremental(srv, client):
  q = QUESTIONS[1]
  chunks = list(client.chat.completions.create(**chat_kw(q, max_tokens=40, stream=True, stream_options={"include_usage": True})))
  assert chunks[0].choices[0].delta.role == "assistant"
  deltas = [c.choices[0].delta.content for c in chunks if c.choices and c.choices[0].delta.content]
  assert len(deltas) > 3 and "".join(deltas) == srv["refs"][q]
  fins = [c.choices[0].finish_reason for c in chunks if c.choices and c.choices[0].finish_reason]
  assert fins in (["stop"], ["length"])
  assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens > 3 and not chunks[-1].choices

@pytest.mark.slow
def test_sse_wire_format(srv):
  with httpx.stream("POST", srv["url"] + "/v1/chat/completions", json=chat_kw("Say hi.", max_tokens=5, stream=True), timeout=120) as r:
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [l for l in r.iter_lines() if l]
  assert all(l.startswith("data: ") for l in lines) and lines[-1] == "data: [DONE]"

@pytest.mark.slow
def test_completions(srv, client):
  p = "The capital of France is"
  r = client.completions.create(model=MODEL, prompt=p, max_tokens=12, temperature=0)
  assert r.choices[0].text == srv["refs"][p] and r.usage.completion_tokens <= 12
  parts = [c.choices[0].text for c in client.completions.create(model=MODEL, prompt=p, max_tokens=12, temperature=0, stream=True)]
  assert "".join(parts) == srv["refs"][p] and len([x for x in parts if x]) > 2
  r2 = client.completions.create(model=MODEL, prompt=srv["tok"].encode(p, add_special_tokens=False), max_tokens=12, temperature=0)
  assert r2.choices[0].text == srv["refs"][p]

@pytest.mark.slow
def test_max_tokens_and_stop(srv, client):
  r = client.completions.create(model=MODEL, prompt="Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,", max_tokens=3, temperature=0)
  assert r.usage.completion_tokens == 3 and r.choices[0].finish_reason == "length"
  full = client.completions.create(model=MODEL, prompt="Count: 1, 2, 3, 4,", max_tokens=24, temperature=0).choices[0].text
  assert "7" in full
  for stream in (False, True):
    r = client.completions.create(model=MODEL, prompt="Count: 1, 2, 3, 4,", max_tokens=24, temperature=0, stop=["7"], stream=stream)
    if stream: cs = list(r); text, fin = "".join(c.choices[0].text for c in cs), cs[-1].choices[0].finish_reason
    else: text, fin = r.choices[0].text, r.choices[0].finish_reason
    assert text == full[:full.index("7")] and fin == "stop"

@pytest.mark.slow
def test_ignore_eos(srv, client):
  kw = dict(model=MODEL, messages=[{"role": "user", "content": "Reply with just the word: hi"}], max_tokens=48, temperature=0)
  short = client.chat.completions.create(**kw)
  assert short.choices[0].finish_reason == "stop" and short.usage.completion_tokens < 48
  full = client.chat.completions.create(**kw, extra_body={"ignore_eos": True})
  assert full.usage.completion_tokens == 48 and full.choices[0].finish_reason == "length"
  import openai
  with pytest.raises(openai.BadRequestError):
    client.chat.completions.create(**kw, extra_body={"ignore_eos": "yes"})

@pytest.mark.slow
def test_seeded_sampling(srv, client):
  kw = dict(model=MODEL, prompt="Once upon a time", max_tokens=16, temperature=0.9, top_p=0.9, seed=1234)
  a, b = (client.completions.create(**kw).choices[0].text for _ in range(2))
  assert a == b and a
  assert client.completions.create(**{**kw, "seed": 99}).choices[0].text != a

@pytest.mark.slow
def test_errors(srv):
  u = srv["url"] + "/v1/chat/completions"
  r = httpx.post(u, json={"model": "nope", "messages": [{"role": "user", "content": "hi"}]})
  assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
  r = httpx.post(u, json={"model": MODEL, "messages": []})
  assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
  r = httpx.post(u, json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 100000})
  assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
  assert httpx.post(u, content=b"{nope").status_code == 400

@pytest.mark.slow
def test_concurrent_streams(srv):
  from openai import AsyncOpenAI
  async def one(c: AsyncOpenAI, q: str):
    parts, fin = [], None
    async for ch in await c.chat.completions.create(**chat_kw(q, max_tokens=40, stream=True)):
      if ch.choices[0].delta.content: parts.append(ch.choices[0].delta.content)
      fin = ch.choices[0].finish_reason or fin
    return "".join(parts), fin
  async def go():
    async with AsyncOpenAI(base_url=srv["url"] + "/v1", api_key="x", timeout=300) as c:
      return await asyncio.gather(*[one(c, q) for q in QUESTIONS])
  steps0 = srv["eng"].stats["decode_steps"]
  res = asyncio.run(go())
  for q, (text, fin) in zip(QUESTIONS, res):
    assert text == srv["refs"][q], q
    assert fin in ("stop", "length")
  assert srv["eng"].stats["decode_steps"] - steps0 < 8 * 40, "requests were not batched"

@pytest.mark.slow
def test_bad_token_ids_rejected_and_isolated(srv):
  u = srv["url"] + "/v1/completions"
  for bad in ([1, 2, 10000000], [1, -1, 2], [[1, 2], [3, -5]]):
    r = httpx.post(u, json={"model": MODEL, "prompt": bad, "max_tokens": 2})
    assert r.status_code == 400 and r.json()["error"]["param"] == "prompt", bad
  r = httpx.post(u, json={"model": MODEL, "prompt": [1, 2, 3], "max_tokens": 2})
  assert r.status_code == 200

@pytest.mark.slow
def test_top_p_zero_and_rep_penalty_zero(srv):
  u = srv["url"] + "/v1/completions"
  r = httpx.post(u, json={"model": MODEL, "prompt": "The capital of France is", "max_tokens": 12, "top_p": 0})
  assert r.status_code == 200 and r.json()["choices"][0]["text"] == srv["refs"]["The capital of France is"]
  r = httpx.post(u, json={"model": MODEL, "prompt": "hi", "max_tokens": 2, "repetition_penalty": 0})
  assert r.status_code == 400 and r.json()["error"]["param"] == "repetition_penalty"

@pytest.mark.slow
def test_usage_counts_prompt_once(srv):
  u = srv["url"] + "/v1/completions"
  r = httpx.post(u, json={"model": MODEL, "prompt": [1, 2, 3, 4, 5], "max_tokens": 2, "n": 3, "temperature": 0}).json()
  assert r["usage"]["prompt_tokens"] == 5 and r["usage"]["total_tokens"] == 5 + r["usage"]["completion_tokens"]
  evs = [json.loads(l[6:]) for l in httpx.post(u, json={"model": MODEL, "prompt": [1, 2, 3, 4, 5], "max_tokens": 2, "n": 3, "stream": True, "stream_options": {"include_usage": True}}).text.splitlines()
         if l.startswith("data: {")]
  assert evs[-1]["usage"]["prompt_tokens"] == 5

@pytest.mark.slow
def test_nonstream_disconnect_aborts(srv):
  eng = srv["eng"]
  with pytest.raises(httpx.TimeoutException):
    httpx.post(srv["url"] + "/v1/completions", json={"model": MODEL, "prompt": "Count up:", "max_tokens": 400, "min_tokens": 400, "temperature": 0}, timeout=1.0)
  t0 = time.time()
  while eng.seqs and time.time() - t0 < 5: time.sleep(0.1)
  assert not eng.seqs, "abandoned non-stream request kept its slot"

# ---------------------------------------------------------------- the engine in its own process (`fusezap serve --engine-process`)

def test_engine_process_auto(monkeypatch):
  from fusezap.serve.cli import build_parser, engine_process
  monkeypatch.delenv("FZ_DEVICE", raising=False)
  p = lambda *a: build_parser().parse_args(["serve", MODEL, *a])
  assert engine_process(p("--device", "CUDA")) and engine_process(p("--device", "cuda:1"))
  assert not engine_process(p("--device", "METAL")) and not engine_process(p("--device", "CPU"))
  assert engine_process(p("--device", "NEURON")) and not engine_process(p("--device", "NEURON", "--engine-process", "off"))
  assert engine_process(p("--device", "METAL", "--engine-process", "on")) and not engine_process(p("--device", "CUDA", "--engine-process", "off"))
  monkeypatch.setenv("FZ_DEVICE", "CUDA")
  assert engine_process(p())
  monkeypatch.setenv("FZ_DEVICE", "METAL")
  assert not engine_process(p())

def test_engine_process_start_failure():
  from fusezap.serve.api import ProcEngine
  from fusezap.serve.cli import build_parser
  pe = ProcEngine(build_parser().parse_args(["serve", "/nonexistent/model", "--device", "CPU"]))
  with pytest.raises(RuntimeError, match="engine process failed"): pe.start()
  assert pe.proc is None

def test_engine_process_death():
  """The engine process dies while serving (SPEED-R3-reviewfix): the request in flight fails, /health and new requests
  (stream or not) get 503 instead of 200 / a BrokenPipeError 500, no handle leaks, and on_dead runs once. A stop() is not a
  death. The child is a bare pipe end here (closing it is what the server sees when the process is killed)."""
  import multiprocessing as mp
  from fusezap.serve.api import ProcEngine
  def fake(on_dead) -> tuple[ProcEngine, Any]:
    pe = ProcEngine(None, on_dead=on_dead)
    pe.conn, child = mp.Pipe()
    pe.info = {"max_model_len": 64, "vocab": 100}
    pe.reader = threading.Thread(target=pe._read, daemon=True); pe.reader.start()
    return pe, child
  calls = []
  pe, child = fake(lambda: calls.append(1))
  async def go():
    h = pe.submit([1, 2], SamplingParams(max_tokens=4))
    assert child.recv()[0] == "add"
    child.close()
    out = await asyncio.wait_for(h.q.get(), 10)
    assert isinstance(out, RuntimeError) and "engine process exited" in str(out)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(pe, None, "m")), base_url="http://t") as c:
      r = await c.get("/health")
      assert r.status_code == 503 and r.json()["status"] == "error"
      for stream in (False, True):
        r = await c.post("/v1/completions", json={"prompt": [1, 2, 3], "max_tokens": 2, "stream": stream})
        assert r.status_code == 503 and r.json()["error"]["code"] == "engine_unavailable", r.text
  asyncio.run(go())
  pe.reader.join(5)
  assert pe.dead and not pe.handles and calls == [1]
  pe, child = fake(lambda: calls.append(2))   # a stop(): the reader sees the same EOF, but nothing is marked dead
  pe.stopping = True; child.close(); pe.reader.join(5)
  assert pe.dead is None and calls == [1]

@pytest.mark.slow
def test_engine_process_server(srv):
  """`fusezap serve --engine-process on` (the CUDA default): the same greedy text as the in-process engine for concurrent
  streams (a 16-block KV pool, so some are preempted and recomputed), completions, n > 1, and the error paths, including
  one only the engine process detects (a prompt over the KV pool's capacity)."""
  import subprocess, sys
  dev = os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  cmd = [sys.executable, "-m", "fusezap", "serve", MODEL, "--device", dev, "--dtype", "float32", "--port", str(port), "--max-model-len", "512",
         "--num-blocks", "16", "--max-prefill-tokens", "256", "--beam", str(srv["eng"].decode_beam), "--engine-process", "on", "--log-level", "warning"]
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
  url = f"http://127.0.0.1:{port}"
  try:
    t0 = time.time()
    while True:
      assert proc.poll() is None, proc.stdout.read()
      assert time.time() - t0 < 600, "server did not start"
      try:
        if httpx.get(url + "/health", timeout=1).status_code == 200: break
      except httpx.HTTPError: pass
      time.sleep(0.5)
    from openai import AsyncOpenAI, OpenAI
    async def one(c: AsyncOpenAI, q: str) -> str:
      parts = []
      async for ch in await c.chat.completions.create(**chat_kw(q, max_tokens=40, stream=True)):
        if ch.choices[0].delta.content: parts.append(ch.choices[0].delta.content)
      return "".join(parts)
    async def go():
      async with AsyncOpenAI(base_url=url + "/v1", api_key="x", timeout=300) as c: return await asyncio.gather(*[one(c, q) for q in QUESTIONS])
    for q, text in zip(QUESTIONS, asyncio.run(go())): assert text == srv["refs"][q], q
    c = OpenAI(base_url=url + "/v1", api_key="x", timeout=300)
    p = "The capital of France is"
    r = c.completions.create(model=MODEL, prompt=p, max_tokens=12, temperature=0)
    assert r.choices[0].text == srv["refs"][p] and 0 < r.usage.completion_tokens <= 12
    r = httpx.post(url + "/v1/completions", json={"model": MODEL, "prompt": p, "max_tokens": 12, "n": 3, "temperature": 0}, timeout=300).json()
    assert [ch["text"] for ch in r["choices"]] == [srv["refs"][p]] * 3
    r = httpx.post(url + "/v1/completions", json={"model": MODEL, "prompt": list(range(1000, 1300)), "max_tokens": 4}, timeout=60)
    assert r.status_code == 400 and "KV cache capacity" in r.json()["error"]["message"]
    r = httpx.post(url + "/v1/completions", json={"model": MODEL, "prompt": [1, 2, 10000000], "max_tokens": 2}, timeout=60)
    assert r.status_code == 400 and r.json()["error"]["param"] == "prompt"
    assert httpx.get(url + "/v1/models").json()["data"][0]["max_model_len"] == 512
    assert httpx.get(url + "/v1/models").json()["data"][0]["fusezap_mode"] == "exact"   # reported by the engine process
    assert c.completions.create(model=MODEL, prompt=p, max_tokens=12, temperature=0).choices[0].text == srv["refs"][p]   # still serving
    # the engine process dies (OOM kill, segfault): the server stops and exits 1, so a supervisor can restart it
    import signal
    kids = subprocess.run(["pgrep", "-P", str(proc.pid)], capture_output=True, text=True).stdout.split()
    cmds = {int(k): subprocess.run(["ps", "-o", "command=", "-p", k], capture_output=True, text=True).stdout for k in kids}
    (pid,) = [k for k, v in cmds.items() if "spawn_main" in v and "resource_tracker" not in v]
    os.kill(pid, signal.SIGKILL)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 1 and "the engine process exited" in out, out[-2000:]
  finally:
    proc.terminate()
    try: proc.communicate(timeout=60)
    except subprocess.TimeoutExpired: proc.kill(); proc.communicate()

def test_req_trace_file(tmp_path, monkeypatch):
  """FZ_REQ_TRACE=<prefix>: one JSON line per request in <prefix>.<side>.jsonl; off without the variable."""
  import json
  from fusezap.serve.api import ReqTrace
  monkeypatch.delenv("FZ_REQ_TRACE", raising=False)
  assert ReqTrace("api").f is None
  monkeypatch.setenv("FZ_REQ_TRACE", str(tmp_path / "t"))
  tr = ReqTrace("api")
  tr.write({"rid": 1, "recv": 1.0, "done": 2.0}); tr.write({"rid": 2, "recv": 3.0})
  tr.f.close()
  assert [json.loads(l)["rid"] for l in open(tmp_path / "t.api.jsonl")] == [1, 2]

def test_gen_coalesces_queued_outputs():
  """Outputs that queued up while a handler was busy go out as one chunk with the same tokens and text (round 12), the
  finishing output included; an error after tokens still surfaces after them."""
  import asyncio
  from fusezap.serve.api import Server, ReqTrace, _Handle, APIError
  from fusezap.serve.engine import RequestOutput, SamplingParams
  class Tok:
    def decode(self, ids, skip_special_tokens=True): return "".join(chr(97 + i % 26) for i in ids)
  class Eng:
    def __init__(self, items): self.items = items
    def submit(self, ids, sp):
      h = _Handle(asyncio.get_running_loop()); h.rid = 1
      for it in self.items: h.q.put_nowait(it)
      return h
    def abort(self, h): pass
  def run(items):
    s = Server.__new__(Server)
    s.tok, s.trace, s.async_engine, s.prof = Tok(), ReqTrace("none"), Eng(items), None
    async def go():
      out = []
      try:
        async for x in s._gen([1, 2], SamplingParams(), []): out.append(x)
      except APIError as e: out.append(("error", str(e)))
      return out
    return asyncio.run(go())
  outs = [RequestOutput(1, [i], [], i == 4, "length" if i == 4 else None) for i in range(5)]
  got = run(outs)
  assert got == [("abcde", "length", 5, [0, 1, 2, 3, 4])]
  got = run(outs[:3] + [RuntimeError("boom")])
  assert got[0] == ("abc", None, 3, [0, 1, 2]) and got[1][0] == "error" and "boom" in got[1][1]
