"""`fusezap serve` OpenAI-API extensions: logprobs (completions `logprobs` + `echo`, chat `logprobs` / `top_logprobs`,
`return_tokens_as_token_ids`), tool calling (`tools` / `tool_choice`, Hermes and Mistral output parsed into `tool_calls`)
and `response_format` handling (json_object / json_schema constrained: tests/test_serve_constrain.py; a 400 without llguidance).

  - fast: the parsers and request validation; the response shapes through the real HTTP app with a fake engine (canned
    tokens, a fake scorer) and the Qwen2.5 tokenizer (skips if it is not in the HF cache);
  - slow (FZ_SLOW=1): Qwen2.5-0.5B-Instruct served on METAL (or CPU): echo logprobs against HF transformers float32
    log_softmax at every position, chat logprobs = completions logprobs on the templated ids, tool calls with auto /
    named / required / none, buffered streams; and the engine-process path (`--engine-process on`) for a score request.
"""
from __future__ import annotations
import json, os, socket, subprocess, sys, threading, time
import numpy as np
import httpx
import pytest
from fusezap.serve import tools as TL, logprobs as LP

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CLOCK = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}

def _tok(name: str):
  from fusezap.serve.tokenizer import Tokenizer
  from fusezap.serve.weights import resolve_model
  try: return Tokenizer(resolve_model(name, allow_download=False))
  except Exception as e: pytest.skip(f"{name} tokenizer not in the HF cache: {e}")

# ---------------------------------------------------------------- parsers and validation (no model)

def test_hermes_parse():
  spec = TL.ToolSpec([WEATHER], "auto", None, "hermes")
  calls, content = TL.parse(spec, '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>')
  assert content is None and len(calls) == 1
  c = calls[0]
  assert c["type"] == "function" and c["id"].startswith("call_") and c["function"]["name"] == "get_weather"
  assert isinstance(c["function"]["arguments"], str) and json.loads(c["function"]["arguments"]) == {"city": "Paris"}
  # text before the call, two calls, a missing closing tag at the very end
  calls, content = TL.parse(spec, 'Let me check.\n<tool_call>{"name": "get_weather", "arguments": {"city": "Oslo"}}</tool_call>\n'
                                  '<tool_call>\n{"name": "get_time", "arguments": {}}')
  assert content == "Let me check." and [c["function"]["name"] for c in calls] == ["get_weather", "get_time"]
  assert TL.parse(spec, "Paris is sunny.") == ([], "Paris is sunny.")
  bad = '<tool_call>\n{"name": "get_weather", "arguments": {"city": }\n</tool_call>'
  assert TL.parse(spec, bad) == ([], bad)   # invalid JSON: all content, nothing half-parsed

def test_mistral_parse_and_prefixes():
  spec = TL.ToolSpec([WEATHER], "auto", None, "mistral")
  calls, content = TL.parse(spec, '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Rome"}}]')
  assert content is None and json.loads(calls[0]["function"]["arguments"]) == {"city": "Rome"}
  assert TL.parse(spec, "no call") == ([], "no call")
  named = TL.ToolSpec([WEATHER], "named", "get_weather", "hermes")
  assert named.forced_prefix == '<tool_call>\n{"name": "get_weather", "arguments": '
  calls, _ = TL.parse(named, named.forced_prefix + '{"city": "Lima"}}\n</tool_call>')
  assert calls[0]["function"]["name"] == "get_weather" and json.loads(calls[0]["function"]["arguments"]) == {"city": "Lima"}
  assert TL.ToolSpec([WEATHER], "required", None, "hermes").forced_prefix == '<tool_call>\n{"name": "'
  assert TL.ToolSpec([WEATHER], "auto", None, "hermes").forced_prefix == ""

def test_tool_spec_validation():
  tok = None   # tool_format falls back to hermes without a tokenizer
  assert not TL.tool_spec({}, tok).active
  assert not TL.tool_spec({"tools": [WEATHER], "tool_choice": "none"}, tok).active
  s = TL.tool_spec({"tools": [WEATHER, CLOCK]}, tok)
  assert s.mode == "auto" and s.active
  s = TL.tool_spec({"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}, tok)
  assert s.mode == "named" and s.name == "get_weather"
  for body, param in [({"tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "nope"}}}, "tool_choice"),
                      ({"tools": [{"type": "function"}]}, "tools"), ({"tools": "x"}, "tools"), ({"tool_choice": "required"}, "tool_choice"),
                      ({"tools": [WEATHER], "tool_choice": "sometimes"}, "tool_choice")]:
    with pytest.raises(TL.ToolError) as e: TL.tool_spec(body, tok)
    assert e.value.param == param, body
  m = TL.request_messages([{"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                                                                                   "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]}])
  assert m[0]["tool_calls"][0]["function"]["arguments"] == {"city": "Paris"}

def test_response_format_handling(monkeypatch):
  """text / none: no constraint; json_object / json_schema: a grammar constraint (tests/test_serve_constrain.py), or a 400
  naming the optional llguidance package when it is not installed."""
  from fusezap.serve import constrain as C
  assert TL.check_response_format({}) is None and TL.check_response_format({"response_format": {"type": "text"}}) is None
  def missing(): raise C.ConstraintError(f"grammar-constrained decoding needs the optional llguidance package ({C.INSTALL})")
  monkeypatch.setattr(C, "_llg", missing)
  for rf in ({"type": "json_object"}, {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}):
    with pytest.raises(TL.ToolError, match="grammar-constrained decoding needs the optional llguidance") as e: TL.check_response_format({"response_format": rf})
    assert e.value.param == "response_format"
  for rf in ("json", {"type": "xml"}, {"type": "json_schema", "json_schema": "x"}):
    with pytest.raises(TL.ToolError, match="response_format"): TL.check_response_format({"response_format": rf})

class _CharTok:
  """Each id decodes to one character (chr(97 + id % 26)), enough for the logprobs shapes."""
  def decode(self, ids, skip_special_tokens=False): return "".join(chr(97 + i % 26) for i in ids)

def test_logprob_shapes():
  tok = _CharTok()
  sc = {"start": 1, "logprobs": [-1.0, -0.5, -2.0], "top": [[(1, -1.0), (5, -1.5)], [(2, -0.5), (3, -0.7)], [(9, -0.1), (3, -2.0)]]}
  lp = LP.completion_logprobs(tok, [0, 1, 2, 3], sc, 2, as_ids=False)
  assert lp["tokens"] == ["a", "b", "c", "d"] and lp["token_logprobs"] == [None, -1.0, -0.5, -2.0]
  assert lp["top_logprobs"][0] is None and lp["top_logprobs"][3] == {"j": -0.1, "d": -2.0}   # the sampled token is always in it
  assert lp["text_offset"] == [0, 1, 2, 3]
  lp = LP.completion_logprobs(tok, [3], {"start": 3, "logprobs": [-2.0], "top": [[(9, -0.1)]]}, 0, as_ids=True)
  assert lp["tokens"] == ["token_id:3"] and lp["top_logprobs"] == [{"token_id:9": -0.1, "token_id:3": -2.0}]
  ch = LP.chat_logprobs(tok, [1, 2], {"start": 5, "logprobs": [-1.0, -0.5], "top": [[(1, -1.0), (5, -1.5)], [(2, -0.5), (3, -0.7)]]}, 1)
  assert ch["content"][0] == {"token": "b", "logprob": -1.0, "bytes": [98], "top_logprobs": [{"token": "b", "logprob": -1.0, "bytes": [98]}]}

# ---------------------------------------------------------------- the HTTP app with a fake engine (fast)

class _FakeModel:
  def __init__(self, vocab: int): self.embed_host = np.zeros((vocab, 1), np.float32)

class _FakeScorer:
  """logprob of ids[i] = -0.1 * (i % 7) - 0.01; top-k are ids (ids[i] + j) % vocab with descending logprobs."""
  def __init__(self, vocab: int): self.vocab, self.calls = vocab, []
  def score(self, ids, start, k):
    self.calls.append((list(ids), start, k))
    lo = max(start, 1)
    lps = [-0.1 * (i % 7) - 0.01 for i in range(lo, len(ids))]
    return {"start": lo, "logprobs": lps, "top": [[((ids[i] + j) % self.vocab, lps[i - lo] - 0.5 * j) for j in range(k)] for i in range(lo, len(ids))]}

class FakeEngine:
  """LLMEngine's interface: every request produces the token ids of `script` (then eos, unless max_tokens ends it first)."""
  def __init__(self, tok, script: str):
    from fusezap.serve.engine import RequestOutput
    self.RO, self.tok = RequestOutput, tok
    self.vocab, self.max_model_len = tok.vocab_size, 4096
    self.model = _FakeModel(self.vocab)
    self.script = tok.encode(script, add_special_tokens=False) + [tok.stop_token_ids[0]]
    self.seqs, self.n = {}, 0
    self._fz_scorer = _FakeScorer(self.vocab)
  def add_request(self, ids, params, request_id=None):
    self.n += 1
    self.seqs[self.n] = {"out": [], "p": params}
    return self.n
  def has_unfinished(self): return bool(self.seqs)
  def abort(self, rid): self.seqs.pop(rid, None)
  def release(self, rid): pass
  def step(self):
    outs = []
    for rid, s in list(self.seqs.items()):
      t = self.script[len(s["out"])]
      s["out"].append(t)
      fin = t in self.tok.stop_token_ids or len(s["out"]) >= s["p"].max_tokens
      outs.append(self.RO(rid, [t], list(s["out"]), fin, ("stop" if t in self.tok.stop_token_ids else "length") if fin else None))
      if fin: del self.seqs[rid]
    return outs

def _serve(app):
  import uvicorn
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
  th = threading.Thread(target=server.run, daemon=True); th.start()
  t0 = time.time()
  while not server.started:
    assert time.time() - t0 < 30 and th.is_alive(), "server did not start"
    time.sleep(0.02)
  return server, th, f"http://127.0.0.1:{port}"

@pytest.fixture
def fake():
  """(url, engine) of a server whose engine answers `fake.script` (set it before the request: a fresh engine per test)."""
  from fusezap.serve.api import create_app
  tok = _tok(QWEN)
  servers = []
  def make(script: str):
    eng = FakeEngine(tok, script)
    server, th, url = _serve(create_app(eng, tok, "m"))
    servers.append((server, th))
    return url, eng, tok
  yield make
  for server, th in servers: server.should_exit = True; th.join(10)

def _sse(r: httpx.Response) -> list[dict]:
  return [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]

def test_completions_logprobs_and_echo_shapes(fake):
  url, eng, tok = fake(" Paris.")
  prompt = tok.encode("The capital of France is", add_special_tokens=False)
  r = httpx.post(url + "/v1/completions", json={"prompt": prompt, "max_tokens": 8, "logprobs": 2, "echo": True, "return_tokens_as_token_ids": True}).json()
  ch, gen = r["choices"][0], eng.script
  lp = ch["logprobs"]
  assert ch["text"] == "The capital of France is Paris." and ch["finish_reason"] == "stop"
  assert lp["tokens"] == [f"token_id:{t}" for t in prompt + gen]
  assert lp["token_logprobs"][0] is None and lp["top_logprobs"][0] is None and len(lp["token_logprobs"]) == len(prompt) + len(gen)
  assert all(isinstance(d, dict) and 2 <= len(d) <= 3 for d in lp["top_logprobs"][1:])
  assert lp["text_offset"][0] == 0 and lp["text_offset"] == sorted(lp["text_offset"])
  assert eng._fz_scorer.calls[-1] == (prompt + gen, 0, 2)
  # without echo: only the generated tokens, string keys
  r = httpx.post(url + "/v1/completions", json={"prompt": "Hi", "max_tokens": 8, "logprobs": 1}).json()
  lp = r["choices"][0]["logprobs"]
  assert lp["tokens"] == [tok.decode([t]) for t in gen] and None not in lp["token_logprobs"] and r["choices"][0]["text"] == " Paris."
  # echo alone (no logprobs) and plain requests keep logprobs null
  r = httpx.post(url + "/v1/completions", json={"prompt": "Hi", "max_tokens": 8, "echo": True}).json()
  assert r["choices"][0]["text"] == "Hi Paris." and r["choices"][0]["logprobs"] is None
  for bad in ({"logprobs": 21}, {"logprobs": True}, {"echo": "yes"}, {"tools": [WEATHER]}, {"return_tokens_as_token_ids": 1}):
    rr = httpx.post(url + "/v1/completions", json={"prompt": "Hi", "max_tokens": 4, **bad})
    assert rr.status_code == 400 and rr.json()["error"]["param"] == next(iter(bad)), bad

def test_chat_logprobs_shapes_and_stream(fake):
  url, eng, tok = fake("Hello there")
  msgs = [{"role": "user", "content": "hi"}]
  r = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "logprobs": True, "top_logprobs": 3}).json()
  c = r["choices"][0]["logprobs"]["content"]
  assert [e["token"] for e in c] == [tok.decode([t]) for t in eng.script]
  assert all(set(e) == {"token", "logprob", "bytes", "top_logprobs"} and len(e["top_logprobs"]) == 3 for e in c)
  assert c[0]["bytes"] == list(c[0]["token"].encode())
  r = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "logprobs": True}).json()
  assert all(e["top_logprobs"] == [] for e in r["choices"][0]["logprobs"]["content"])
  assert httpx.post(url + "/v1/chat/completions", json={"messages": msgs}).json()["choices"][0]["logprobs"] is None
  # stream with logprobs: sent once generation is done, same content and logprobs as non-stream
  rs = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "logprobs": True, "top_logprobs": 3, "stream": True, "stream_options": {"include_usage": True}})
  ev = _sse(rs)
  text = "".join(e["choices"][0]["delta"].get("content", "") for e in ev if e["choices"])
  lps = [x for e in ev if e["choices"] and e["choices"][0].get("logprobs") for x in e["choices"][0]["logprobs"]["content"]]
  assert text == "Hello there" and [x["token"] for x in lps] == [x["token"] for x in c] and ev[-1]["usage"]["completion_tokens"] == len(eng.script)
  assert rs.text.rstrip().endswith("data: [DONE]")
  for bad, param in (({"top_logprobs": 2}, "top_logprobs"), ({"logprobs": True, "top_logprobs": 21}, "top_logprobs"), ({"logprobs": 1}, "logprobs"),
                     ({"echo": True}, "echo"), ({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "array", "uniqueItems": True}}}}, "response_format")):
    rr = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, **bad})
    assert rr.status_code == 400 and rr.json()["error"]["param"] == param, bad
  assert httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "response_format": {"type": "text"}}).status_code == 200

def test_chat_tool_calls(fake):
  url, eng, tok = fake('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>')
  msgs = [{"role": "user", "content": "Weather in Paris?"}]
  r = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER, CLOCK]}).json()
  ch = r["choices"][0]
  assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] is None
  call = ch["message"]["tool_calls"][0]
  assert call["function"]["name"] == "get_weather" and json.loads(call["function"]["arguments"]) == {"city": "Paris"}
  # streamed: one tool_calls delta with an index, then finish_reason tool_calls
  ev = _sse(httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER], "stream": True}))
  deltas = [e["choices"][0]["delta"] for e in ev]
  tc = [d["tool_calls"] for d in deltas if "tool_calls" in d]
  assert len(tc) == 1 and tc[0][0]["index"] == 0 and tc[0][0]["function"]["name"] == "get_weather" and ev[-1]["choices"][0]["finish_reason"] == "tool_calls"
  # tool_choice none: tools are not rendered and nothing is parsed
  r = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER], "tool_choice": "none"}).json()
  assert "tool_calls" not in r["choices"][0]["message"] and r["choices"][0]["finish_reason"] == "stop"
  # the conversation continues with the call and its result (arguments as a JSON string, as OpenAI clients send them)
  follow = msgs + [{"role": "assistant", "content": None, "tool_calls": [call]}, {"role": "tool", "tool_call_id": call["id"], "content": '{"temp_c": 18}'}]
  assert httpx.post(url + "/v1/chat/completions", json={"messages": follow, "tools": [WEATHER]}).status_code == 200
  rr = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER], "tool_choice": {"type": "function", "function": {"name": "x"}}})
  assert rr.status_code == 400 and rr.json()["error"]["param"] == "tool_choice"

def test_named_tool_choice_forces_the_call(fake):
  url, eng, tok = fake('{"city": "Oslo"}}\n</tool_call>')   # the model writes only the arguments after the forced prefix
  r = httpx.post(url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Oslo?"}], "tools": [WEATHER, CLOCK],
                                                     "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}).json()
  ch = r["choices"][0]
  assert ch["finish_reason"] == "stop" and ch["message"]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Oslo"}'}

def test_parallel_tool_calls_false_keeps_one_call(fake):
  two = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>\n<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'
  url, eng, tok = fake(two)
  msgs = [{"role": "user", "content": "Weather and time?"}]
  names = lambda r: [c["function"]["name"] for c in r.json()["choices"][0]["message"]["tool_calls"]]
  assert names(httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER, CLOCK]})) == ["get_weather", "get_time"]
  assert names(httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER, CLOCK], "parallel_tool_calls": True})) == ["get_weather", "get_time"]
  assert names(httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER, CLOCK], "parallel_tool_calls": False})) == ["get_weather"]
  ev = _sse(httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER, CLOCK], "parallel_tool_calls": False, "stream": True}))
  tc = [d["tool_calls"] for e in ev if "tool_calls" in (d := e["choices"][0]["delta"])]
  assert len(tc) == 1 and len(tc[0]) == 1 and tc[0][0]["function"]["name"] == "get_weather"
  rr = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "tools": [WEATHER], "parallel_tool_calls": "no"})
  assert rr.status_code == 400 and rr.json()["error"]["param"] == "parallel_tool_calls"

# a value of each unimplemented field that would change the output, and the neutral values clients send by default
REJECTED = [("presence_penalty", 0.5), ("presence_penalty", -1), ("presence_penalty", False), ("frequency_penalty", 1.2),
            ("logit_bias", {"15": -100}), ("best_of", 3), ("suffix", " end"), ("functions", [WEATHER["function"]]),
            ("function_call", {"name": "get_weather"}), ("modalities", ["text", "audio"]), ("audio", {"voice": "alloy", "format": "wav"}),
            ("web_search_options", {}), ("reasoning_effort", "high"), ("verbosity", "low"), ("min_p", 0.1), ("min_tokens", 5),
            ("stop_token_ids", [13]), ("use_beam_search", True), ("include_stop_str_in_output", True), ("skip_special_tokens", False),
            ("prompt_logprobs", 1), ("truncate_prompt_tokens", 8), ("guided_json", {"type": "object"}), ("guided_regex", "a+"),
            ("guided_choice", ["a", "b"]), ("guided_grammar", "root ::= \"a\""), ("chat_template", "{{ messages }}"),
            ("chat_template_kwargs", {"enable_thinking": False}), ("add_generation_prompt", False), ("continue_final_message", True)]
NEUTRAL = {"presence_penalty": 0, "frequency_penalty": 0.0, "logit_bias": {}, "best_of": 1, "suffix": None, "functions": [],
           "function_call": "auto", "modalities": ["text"], "audio": None, "web_search_options": None, "reasoning_effort": None,
           "verbosity": "medium", "min_p": 0.0, "min_tokens": 0, "stop_token_ids": [], "use_beam_search": False,
           "include_stop_str_in_output": False, "skip_special_tokens": True, "prompt_logprobs": None, "truncate_prompt_tokens": None,
           "guided_json": None, "chat_template_kwargs": {}, "add_generation_prompt": True, "continue_final_message": False,
           "top_k": -1, "parallel_tool_calls": True,
           # informational: accepted, no effect on the output
           "user": "u-1", "metadata": {"run": "a"}, "store": False, "service_tier": "auto", "prediction": {"type": "content", "content": "x"},
           "prompt_cache_key": "k", "safety_identifier": "s", "stream_options": {"include_usage": True, "include_obfuscation": False}}

def test_unsupported_parameters_rejected(fake):
  """A field that would change the output and is not implemented is a 400 unsupported_parameter naming it, on both
  endpoints and also for a stream (an HTTP error before any event); neutral values and informational fields pass."""
  url, eng, tok = fake(" Paris.")
  bodies = {"/v1/chat/completions": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 4},
            "/v1/completions": {"prompt": "The capital of France is", "max_tokens": 4}}
  for path, base in bodies.items():
    for stream in (False, True):
      r = httpx.post(url + path, json={**base, **NEUTRAL, "stream": stream})
      assert r.status_code == 200, (path, stream, r.text)
      if stream: assert r.text.rstrip().endswith("data: [DONE]")
      else: ch = r.json()["choices"][0]; assert (ch["message"]["content"] if "message" in ch else ch["text"]) == " Paris."
      n0 = eng.n
      for k, v in REJECTED:
        r = httpx.post(url + path, json={**base, k: v, "stream": stream})
        assert r.status_code == 400 and r.headers["content-type"].startswith("application/json"), (path, k, v, stream)
        e = r.json()["error"]
        assert e == {"message": e["message"], "type": "invalid_request_error", "param": k, "code": "unsupported_parameter"}, (path, k, e)
        assert e["message"].startswith(f"{k} is not supported by FuseZap Serve yet; "), e
      assert eng.n == n0   # nothing rejected reached the engine
  e = httpx.post(url + "/v1/chat/completions", json={**bodies["/v1/chat/completions"], "presence_penalty": 0.5}).json()["error"]
  assert e["message"] == "presence_penalty is not supported by FuseZap Serve yet; send 0 or leave it out"
  # best_of equal to n asks for nothing extra; min_tokens with ignore_eos (and no stop) is what ignore_eos already does
  c = bodies["/v1/completions"]
  assert httpx.post(url + "/v1/completions", json={**c, "n": 2, "best_of": 2}).status_code == 200
  assert httpx.post(url + "/v1/completions", json={**c, "min_tokens": 4, "ignore_eos": True}).status_code == 200
  assert httpx.post(url + "/v1/completions", json={**c, "min_tokens": 4, "ignore_eos": True, "stop": ["x"]}).status_code == 400

def test_qwen_template_renders_tools_and_mistral_format():
  tok = _tok(QWEN)
  assert TL.tool_format(tok) == "hermes"
  text = tok.render_chat([{"role": "user", "content": "hi"}], True, [WEATHER])
  assert "<tools>" in text and '"name": "get_weather"' in text
  from fusezap.serve.tokenizer import Tokenizer
  from fusezap.serve.weights import resolve_model
  try: mt = Tokenizer(resolve_model(MISTRAL, allow_download=False))
  except Exception: return
  assert TL.tool_format(mt) == "mistral"
  assert "[AVAILABLE_TOOLS]" in mt.render_chat([{"role": "user", "content": "hi"}], True, [WEATHER])

def test_scorer_oom_releases_and_retries(monkeypatch):
  """Scorer.chunks after an out-of-memory error: drops its jits and the allocator cache (release), retries once from the
  first row not yet delivered (no duplicates), and on a second failure releases again and raises MemoryError."""
  from types import SimpleNamespace
  V = 128
  sc = LP.Scorer(SimpleNamespace(device="CPU", table_len=4096), 4096, chunk=16)
  calls, released = [], []
  fail_at = {"n": 1}
  def fake_chunks(ids, b, r0, r1):
    calls.append((b, r0, r1))
    for s in range(r0 // 16 * 16, r1, 16):
      if fail_at["n"] and s >= 32: fail_at["n"] -= 1; raise MemoryError("CUDA out of memory: cuMemAlloc(1245708288)")
      lo, hi = max(r0, s), min(r1, s + 16)
      yield lo, np.tile(np.arange(lo, hi, dtype=np.float32)[:, None], (1, V))
  def fake_release():
    released.append(1); sc.jits.clear(); sc.head_jits.clear()
  monkeypatch.setattr(sc, "_chunks", fake_chunks)
  monkeypatch.setattr(sc, "release", fake_release)
  ids = list(range(100))
  rows = sc.logprobs(ids)
  assert rows.shape == (100, V) and rows[:, 0].tolist() == list(range(100))   # every row once, in order
  assert calls == [(128, 0, 100), (128, 32, 100)] and len(released) == 1
  r = sc.score(ids, 40, 2)                                                     # rows 39..98 predict tokens 40..99
  assert r["start"] == 40 and len(r["logprobs"]) == 60 and calls[-1] == (128, 39, 99)
  fail_at["n"], calls[:] = 2, []
  with pytest.raises(MemoryError): sc.logprobs(ids)
  assert len(released) == 3 and len(calls) == 2

# ---------------------------------------------------------------- a real model (slow)

@pytest.mark.slow
def test_real_scorer_chunks_bit_identical():
  """The chunked logprobs pass (trunk per bucket, then final norm + lm_head + log_softmax per row chunk) is bit-identical
  to the whole-bucket pass it replaced (Llama.forward over the bucket, log_softmax of the (T, V) logits), for chunks
  smaller than, equal to and larger than the bucket; score() over a sub-range is the same rows. bf16 on CUDA / METAL
  (FZ_DEVICE), float32 on CPU."""
  from fusezap.device import Device
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  from fusezap.tensor import Tensor
  _tok(QWEN)
  dev = os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
  m = load_model(QWEN, dtype=dtypes.float32 if dev == "CPU" else dtypes.bfloat16, device=dev, max_len=512, prefill_chunk=128)
  ids = np.random.default_rng(0).integers(100, 20000, 300).tolist()
  b = LP.Scorer(m, 512)._bucket(len(ids))
  emb = m.embed(np.asarray(ids, np.int64)[None])
  buf = np.zeros((1, b, emb.shape[-1]), dtype=emb.dtype)
  buf[:, :len(ids)] = emb
  with m._opt_env(prefill=True):
    ref = m.forward(Tensor(buf, device=m.device, dtype=m.dtype), LP._NoCache(), 0, b, last_only=False).float()[0].log_softmax(-1).numpy()[:len(ids)]
  for C in (16, 256, 1024):
    sc = LP.Scorer(m, 512, chunk=C)
    for _ in range(3):   # eager, capture, replay
      got = sc.logprobs(ids)
      np.testing.assert_array_equal(got.view(np.uint32), ref.view(np.uint32))
    assert set(sc.head_jits) == {min(C, b)}
    r = sc.score(ids, 250, 3)
    assert r["start"] == 250 and r["logprobs"] == [float(ref[i - 1, ids[i]]) for i in range(250, len(ids))]
    assert [t for t, _ in r["top"][0]] == np.argsort(-ref[249], kind="stable")[:3].tolist()
    sc.release()
    assert not sc.jits and not sc.head_jits

@pytest.fixture(scope="module")
def real():
  from fusezap.device import Device
  from fusezap.dtype import dtypes
  from fusezap.serve.api import create_app, warmup
  from fusezap.serve.engine import LLMEngine
  from fusezap.serve.models import load_model
  tok = _tok(QWEN)
  dev = os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
  m = load_model(QWEN, dtype=dtypes.float32, device=dev, max_len=512, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=256, block_size=16, max_num_seqs=8, max_prefill_tokens=256, decode_beam=0)
  warmup(eng)
  server, th, url = _serve(create_app(eng, tok, QWEN))
  yield url, tok, dev
  server.should_exit = True; th.join(30)

@pytest.mark.slow
def test_real_echo_logprobs_match_hf_float32(real):
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  from fusezap.serve.weights import resolve_model
  url, tok, _ = real
  prompt = tok.apply_chat_template([{"role": "user", "content": "Name three primary colors."}], True)
  r = httpx.post(url + "/v1/completions", json={"prompt": prompt, "max_tokens": 24, "temperature": 0, "logprobs": 5, "echo": True,
                                                "return_tokens_as_token_ids": True}, timeout=600).json()
  lp = r["choices"][0]["logprobs"]
  ids = [int(t.split(":")[1]) for t in lp["tokens"]]
  assert ids[:len(prompt)] == prompt and len(ids) > len(prompt)
  hfm = tr.AutoModelForCausalLM.from_pretrained(resolve_model(QWEN, allow_download=False), dtype=torch.float32).eval()
  with torch.no_grad(): ref = torch.log_softmax(hfm(torch.tensor([ids])).logits[0].float(), -1).numpy()
  got = np.array(lp["token_logprobs"][1:], np.float64)
  want = np.array([ref[i - 1, ids[i]] for i in range(1, len(ids))])
  np.testing.assert_allclose(got, want, atol=2e-3)
  for i in range(1, len(ids)):   # the top-5 ids and values (the sampled token may be added as a 6th)
    top = sorted(((int(k.split(":")[1]), v) for k, v in lp["top_logprobs"][i].items()), key=lambda kv: -kv[1])
    ref_top = np.argsort(-ref[i - 1])[:5]
    assert [t for t, _ in top[:5]] == ref_top.tolist() or abs(ref[i - 1, ref_top[4]] - ref[i - 1, ref_top[5]]) < 1e-3, i
    np.testing.assert_allclose([v for _, v in top[:5]], ref[i - 1, ref_top], atol=2e-3)
  # the generated tokens are greedy: each one is the reference's top-1 (no near ties in this prompt)
  assert all(ids[i] == int(np.argmax(ref[i - 1])) for i in range(len(prompt), len(ids)))

@pytest.mark.slow
def test_real_chat_logprobs_equal_completions(real):
  url, tok, _ = real
  msgs = [{"role": "user", "content": "Say hello in Spanish."}]
  c = httpx.post(url + "/v1/chat/completions", json={"messages": msgs, "max_tokens": 12, "temperature": 0, "logprobs": True, "top_logprobs": 3}, timeout=600).json()
  p = httpx.post(url + "/v1/completions", json={"prompt": tok.apply_chat_template(msgs, True), "max_tokens": 12, "temperature": 0, "logprobs": 3}, timeout=600).json()
  cc, pl = c["choices"][0]["logprobs"]["content"], p["choices"][0]["logprobs"]
  assert [e["token"] for e in cc] == pl["tokens"][:len(cc)]
  np.testing.assert_allclose([e["logprob"] for e in cc], pl["token_logprobs"][:len(cc)], atol=1e-5)

@pytest.mark.slow
def test_real_tool_calls(real):
  F = pytest.importorskip("fusezap.invarix.functional")   # FuseZap Invarix: not in the public export
  TOOLS, TOOL_TASKS, TOOL_SCHEMAS, schema_errors = F.TOOLS, F.TOOL_TASKS, F.TOOL_SCHEMAS, F.schema_errors
  url, tok, _ = real
  for t in TOOL_TASKS:
    for choice in ("auto", "required", {"type": "function", "function": {"name": t["expect"]}}):
      r = httpx.post(url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": t["prompt"]}], "tools": TOOLS, "tool_choice": choice,
                                                         "max_tokens": 128, "temperature": 0}, timeout=600).json()
      ch = r["choices"][0]
      call = ch["message"]["tool_calls"][0]["function"]
      assert call["name"] == t["expect"], (t, choice, ch)
      assert not schema_errors(json.loads(call["arguments"]), TOOL_SCHEMAS[call["name"]]), call
      assert ch["finish_reason"] == ("stop" if isinstance(choice, dict) else "tool_calls")
    r = httpx.post(url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": t["prompt"]}], "tools": TOOLS, "tool_choice": "none",
                                                       "max_tokens": 32, "temperature": 0}, timeout=600).json()
    assert "tool_calls" not in r["choices"][0]["message"]

@pytest.mark.slow
def test_real_engine_process_scores(real):
  """`--engine-process on` (the CUDA default): a score request crosses the pipe and gives the in-process logprobs."""
  url0, tok, dev = real
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  cmd = [sys.executable, "-m", "fusezap", "serve", QWEN, "--device", dev, "--dtype", "float32", "--port", str(port), "--max-model-len", "512",
         "--num-blocks", "64", "--max-prefill-tokens", "256", "--beam", "0", "--engine-process", "on", "--log-level", "warning"]
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
    body = {"prompt": tok.encode("The capital of France is", add_special_tokens=False), "max_tokens": 6, "temperature": 0, "logprobs": 3, "echo": True}
    ra = httpx.post(url + "/v1/completions", json=body, timeout=600).json()
    assert "choices" in ra, ra
    a = ra["choices"][0]
    b = httpx.post(url0 + "/v1/completions", json=body, timeout=600).json()["choices"][0]
    assert a["text"] == b["text"] and a["logprobs"]["tokens"] == b["logprobs"]["tokens"]
    np.testing.assert_allclose(a["logprobs"]["token_logprobs"][1:], b["logprobs"]["token_logprobs"][1:], atol=1e-5)
  finally:
    proc.terminate(); proc.wait(30)
