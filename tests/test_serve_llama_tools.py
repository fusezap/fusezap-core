"""`fusezap serve` tool calls for Llama 3.1+ models (I61, vLLM's `--tool-call-parser llama3_json`): format detection from
the tokenizer (<|python_tag|>) and chat template, prompt rendering through the model's own template (tools in the first
user message, tool results as ipython turns, parallel calls in a request split into one call per message), output
parsing (python_tag, `;` / newline / list parallel calls, "parameters" / "arguments", malformed JSON as content, named
tool_choice), live streams (content streamed, a call held and sent as tool_calls deltas), the CallConstraint hook for
grammar-constrained decoding, and Validate's tools check (fusezap/validate/structured.py, tools-v1) run end to end
against the real HTTP app whose fake model answers in Llama format.

Tokenizers: "mini" (built here: byte-level BPE + the Llama 3 special tokens and a cut-down Llama 3.1 tool template) always
runs; "real" (the Llama 3.3 / 3.1 / 3.2 tokenizer and chat template from the HF cache) skips when none is cached.
"""
from __future__ import annotations
import asyncio, json, os, socket, threading, time
import numpy as np
import httpx
import pytest
from fusezap.serve import tools as TL

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CLOCK = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}
SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>", "<|python_tag|>"]
REAL = ("meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/Llama-3.2-1B-Instruct")

# The tool part of the Llama 3.1 / 3.3 chat template (same structure and strings), without the date / builtin-tools parts.
MINI_TEMPLATE = """{{- bos_token }}
{%- if not tools is defined %}{%- set tools = none %}{%- endif %}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}
{%- else %}{%- set system_message = "" %}{%- endif %}
{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}
{%- if tools is not none %}{{- "Environment: ipython\\n" }}{%- endif %}
{{- system_message }}{{- "<|eot_id|>" }}
{%- if tools is not none %}
    {%- if messages | length != 0 %}{%- set first_user_message = messages[0]['content']|trim %}{%- set messages = messages[1:] %}
    {%- else %}{{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.\\n\\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.Do not use variables.\\n\\n' }}
    {%- for t in tools %}{{- t | tojson(indent=4) }}{{- "\\n\\n" }}{%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}
{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}{{- raise_exception("This model only supports single tool-calls at once!") }}{%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}{{- '"parameters": ' }}{{- tool_call.arguments | tojson }}{{- "}" }}{{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}
        {%- if message.content is mapping or message.content is iterable %}{{- message.content | tojson }}{%- else %}{{- message.content }}{%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}
"""

def _mini_dir(path, template: str = MINI_TEMPLATE, specials=SPECIALS) -> str:
  from tokenizers import Tokenizer, models, pre_tokenizers, decoders
  alpha = pre_tokenizers.ByteLevel.alphabet()
  t = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(sorted(alpha))}, merges=[]))
  t.pre_tokenizer, t.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
  t.add_special_tokens(list(specials))
  os.makedirs(path, exist_ok=True)
  t.save(os.path.join(path, "tokenizer.json"))
  json.dump({"bos_token": "<|begin_of_text|>", "eos_token": "<|eot_id|>", "chat_template": template}, open(os.path.join(path, "tokenizer_config.json"), "w"))
  ids = [t.token_to_id(s) for s in ("<|end_of_text|>", "<|eom_id|>", "<|eot_id|>") if t.token_to_id(s) is not None]
  json.dump({"eos_token_id": ids}, open(os.path.join(path, "generation_config.json"), "w"))
  return str(path)

def _real_dir() -> str | None:
  try: from huggingface_hub import try_to_load_from_cache
  except ImportError: return None
  for repo in REAL:
    f = [try_to_load_from_cache(repo, n) for n in ("tokenizer.json", "tokenizer_config.json")]
    if all(isinstance(x, str) for x in f): return os.path.dirname(f[0])
  return None

@pytest.fixture(params=["mini", "real"])
def ltok(request, tmp_path):
  from fusezap.serve.tokenizer import Tokenizer
  if request.param == "mini": return Tokenizer(_mini_dir(tmp_path / "mini"))
  d = _real_dir()
  if d is None: pytest.skip("no Llama 3.1+ tokenizer in the HF cache")
  return Tokenizer(d)

def _spec(mode="auto", name=None, tools=(WEATHER, CLOCK)): return TL.ToolSpec(list(tools), mode, name, "llama3_json")

# ---------------------------------------------------------------- detection

def test_detection(ltok, tmp_path):
  from fusezap.serve.tokenizer import Tokenizer
  assert TL.tool_format(ltok) == "llama3_json"
  s = TL.tool_spec({"tools": [WEATHER], "tool_choice": "required"}, ltok)
  assert s.fmt == "llama3_json" and s.forced_prefix == '{"name": "' and s.args_key == "parameters"
  # the python tag alone (a template without tools, as Llama 3.0's) or a tools template without the tag: no known format
  plain = "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n{{ m['content'] }}<|eot_id|>{% endfor %}"
  assert TL.tool_format(Tokenizer(_mini_dir(tmp_path / "plain", plain))) is None
  assert TL.tool_format(Tokenizer(_mini_dir(tmp_path / "notag", specials=[s for s in SPECIALS if s != "<|python_tag|>"]))) is None
  with pytest.raises(TL.ToolError, match="Llama 3.1"): TL.tool_spec({"tools": [WEATHER]}, Tokenizer(_mini_dir(tmp_path / "plain2", plain)))
  assert TL.llama_tools_template(MINI_TEMPLATE) and not TL.llama_tools_template(plain) and not TL.llama_tools_template(None)

@pytest.mark.parametrize("repo,fmt", [("unsloth/llama-3-8b-Instruct", None), ("unsloth/Llama-3.2-1B-Instruct", "llama3_json"),
                                      ("unsloth/Meta-Llama-3.1-8B-Instruct", "llama3_json"), ("meta-llama/Llama-3.3-70B-Instruct", "llama3_json"),
                                      ("Qwen/Qwen2.5-0.5B-Instruct", "hermes"), ("mistralai/Mistral-7B-Instruct-v0.3", "mistral")])
def test_detection_cached_tokenizers(repo, fmt):
  """The format of real tokenizers in the HF cache (Llama 3.0 has no python tag and no tools template: none)."""
  from huggingface_hub import try_to_load_from_cache
  from fusezap.serve.tokenizer import Tokenizer
  f = [try_to_load_from_cache(repo, n) for n in ("tokenizer.json", "tokenizer_config.json")]
  if not all(isinstance(x, str) for x in f): pytest.skip(f"{repo} tokenizer not in the HF cache")
  assert TL.tool_format(Tokenizer(os.path.dirname(f[0]))) == fmt

# ---------------------------------------------------------------- rendering (the model's own template)

def test_rendering(ltok):
  msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Weather in Paris?"}]
  text = ltok.render_chat(msgs, True, [WEATHER, CLOCK])
  assert "Environment: ipython" in text and '"parameters": dictionary of argument name and its value' in text
  u = text.index("<|start_header_id|>user<|end_header_id|>")
  assert text.index('"name": "get_weather"') > u and text.index('"name": "get_time"') > u and text.index("Weather in Paris?") > text.index('"name": "get_time"')
  assert text.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n") and "Be brief." in text[:u]
  assert "get_weather" not in ltok.render_chat(msgs, True, None)   # tool_choice none renders no tools
  # multi-turn: the assistant call as the template's JSON, the result as an ipython turn (arguments arrive as a JSON string)
  call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
  follow = msgs + [{"role": "assistant", "content": "", "tool_calls": [call]}, {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 18}'}]
  text = ltok.render_chat(TL.request_messages(follow, "llama3_json"), True, [WEATHER])
  assert '<|start_header_id|>assistant<|end_header_id|>\n\n{"name": "get_weather", "parameters": {"city": "Paris"}}<|eot_id|>' in text
  assert "<|start_header_id|>ipython<|end_header_id|>\n\n" in text and "temp_c" in text.split("ipython")[-1]
  # parallel calls in the history: split into one call per assistant message, each followed by its result (by id)
  two = [dict(call), {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Oslo"}}}]
  hist = msgs + [{"role": "assistant", "content": None, "tool_calls": two}, {"role": "tool", "tool_call_id": "c2", "content": "OSLO3"},
                 {"role": "tool", "tool_call_id": "c1", "content": "PARIS18"}]
  rm = TL.request_messages(hist, "llama3_json")
  assert [(m["role"], m.get("tool_call_id") or m["tool_calls"][0]["id"]) for m in rm[2:]] == [("assistant", "c1"), ("tool", "c1"), ("assistant", "c2"), ("tool", "c2")]
  text = ltok.render_chat(rm, True, [WEATHER])
  assert text.index('"Paris"') < text.index("PARIS18") < text.index('"Oslo"') < text.index("OSLO3")
  # an empty tool_calls on an assistant message is a plain turn (the template tests `'tool_calls' in message`)
  rm = TL.request_messages(msgs + [{"role": "assistant", "content": "Hi.", "tool_calls": []}, {"role": "user", "content": "x"}], "llama3_json")
  assert "tool_calls" not in rm[2] and "Hi.<|eot_id|>" in ltok.render_chat(rm, True, [WEATHER])
  # other formats keep the messages as they were
  assert len(TL.request_messages(hist)) == len(hist)

# ---------------------------------------------------------------- parsing

def _args(calls): return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in calls]

def test_parse():
  sp = _spec()
  for text in ('{"name": "get_weather", "parameters": {"city": "Paris"}}', '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}',
               '\n {"name": "get_weather", "arguments": {"city": "Paris"}} ', '{"name": "get_weather", "parameters": "{\\"city\\": \\"Paris\\"}"}',
               '[{"name": "get_weather", "parameters": {"city": "Paris"}}]'):
    calls, content = TL.parse(sp, text)
    assert _args(calls) == [("get_weather", {"city": "Paris"})] and content is None, text
    assert calls[0]["id"].startswith("call_") and calls[0]["type"] == "function" and isinstance(calls[0]["function"]["arguments"], str)
  par = [("get_weather", {"city": "Paris"}), ("get_time", {})]
  for text in ('{"name": "get_weather", "parameters": {"city": "Paris"}}; {"name": "get_time", "parameters": {}}',
               '{"name": "get_weather", "parameters": {"city": "Paris"}}\n{"name": "get_time"}',
               '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}};<|python_tag|>{"name": "get_time", "parameters": {}}',
               '[{"name": "get_weather", "parameters": {"city": "Paris"}}, {"name": "get_time", "parameters": {}}]'):
    calls, _ = TL.parse(sp, text)
    assert _args(calls) == par and len({c["id"] for c in calls}) == 2, text
  # not a call: returned unchanged as content
  for text in ('The weather is fine.', 'Sure: {"name": "get_weather", "parameters": {"city": "Paris"}}', '{"name": "get_weather", "parameters": {"city": }',
               '{"name": "get_weather", "parameters": {"city": "Paris"}', '{"answer": 42}', '{"name": "get_weather", "parameters": {"city": "Paris"}} and more text',
               '{"name": "get_weather", "parameters": "not json"}', '[1, 2]', '{"name": "", "parameters": {}}', ''):
    assert TL.parse(sp, text) == ([], text), text
  # named: the forced prefix is in the prompt; the first call only, and only when it names that function
  nm = _spec("named", "get_weather")
  assert nm.forced_prefix == '{"name": "get_weather", "parameters": '
  calls, content = TL.parse(nm, nm.forced_prefix + '{"city": "Lima"}}; {"name": "get_time", "parameters": {}}')
  assert _args(calls) == [("get_weather", {"city": "Lima"})] and content is None
  assert TL.parse(_spec("named", "get_time"), '{"name": "get_weather", "parameters": {}}')[0] == []
  rq = _spec("required")
  assert _args(TL.parse(rq, rq.forced_prefix + 'get_time", "parameters": {}}')[0]) == [("get_time", {})]

def test_stream_kind():
  sp = _spec()
  assert [TL.stream_kind(sp, t) for t in ("", "  \n", "<|py", "[", "[ \n")] == [None] * 5
  assert [TL.stream_kind(sp, t) for t in ("{", ' {"na', "<|python_tag|>", "[{", "[ {")] == ["tool"] * 5
  assert [TL.stream_kind(sp, t) for t in ("The", " Sure {", "[1", "<|eot")] == ["content"] * 4
  assert TL.stream_kind(sp, " ", final=True) == "content" and TL.stream_kind(_spec("required"), "") == "tool"
  assert TL.streams_live(sp) and not TL.streams_live(TL.ToolSpec([WEATHER], "auto", None, "hermes")) and not TL.streams_live(TL.ToolSpec())

def test_call_constraint_hook():
  """call_constraint: None for auto / none; for named / required the continuation it describes, appended to the forced
  prefix, parses as exactly that call in every format (the contract a grammar-constrained decoder relies on)."""
  assert TL.call_constraint(_spec()) is None and TL.call_constraint(TL.ToolSpec()) is None
  strict = {**WEATHER, "function": {**WEATHER["function"], "strict": True}}
  for fmt in ("llama3_json", "hermes", "mistral"):
    nm = TL.ToolSpec([strict, CLOCK], "named", "get_weather", fmt)
    c = TL.call_constraint(nm)
    assert c.mode == "named" and c.prefix == nm.forced_prefix and c.tools["get_weather"] == WEATHER["function"]["parameters"] and c.strict == {"get_weather"}
    assert c.call_schema()["properties"] == {"name": {"const": "get_weather"}, c.args_key: WEATHER["function"]["parameters"]}
    calls, _ = TL.parse(nm, c.prefix + json.dumps({"city": "Rome"}) + c.close)
    assert _args(calls) == [("get_weather", {"city": "Rome"})], fmt
    rq = TL.ToolSpec([strict, CLOCK], "required", None, fmt)
    c = TL.call_constraint(rq)
    assert c.mode == "required" and set(c.tools) == {"get_weather", "get_time"} and len(c.call_schema()["anyOf"]) == 2
    gen = 'get_time", "' + c.args_key + '": {}' + c.close
    if c.parallel: gen += c.separator + c.call_open + 'get_weather", "' + c.args_key + '": {"city": "Rome"}' + c.close
    calls, _ = TL.parse(rq, c.prefix + gen)
    assert _args(calls) == [("get_time", {})] + ([("get_weather", {"city": "Rome"})] if c.parallel else []), fmt
  assert TL.call_constraint(TL.ToolSpec([WEATHER], "named", "get_weather", "llama3_json")).args_key == "parameters"

# ---------------------------------------------------------------- the HTTP app with a fake Llama-format model

class _FakeModel:
  def __init__(self, vocab: int): self.embed_host = np.zeros((vocab, 1), np.float32)

class OracleEngine:
  """LLMEngine's interface; each request's output is answer(prompt text with special tokens) (then <|eot_id|>, unless
  max_tokens ends it first), emitted one token per step."""
  def __init__(self, tok, answer, delay: float = 0.0):
    from fusezap.serve.engine import RequestOutput
    self.RO, self.tok, self.answer = RequestOutput, tok, answer
    self.max_model_len, self.model = 1 << 17, _FakeModel(tok.vocab_size)
    self.eot = tok.tok.token_to_id("<|eot_id|>")
    self.seqs, self.n, self.prompts, self.delay = {}, 0, [], delay
  def add_request(self, ids, params, request_id=None):
    self.n += 1
    text = self.tok.decode(ids, skip_special_tokens=False)
    self.prompts.append(text)
    self.seqs[self.n] = {"out": [], "p": params, "script": self.tok.encode(self.answer(text), add_special_tokens=False) + [self.eot]}
    return self.n
  def has_unfinished(self): return bool(self.seqs)
  def abort(self, rid): self.seqs.pop(rid, None)
  def release(self, rid): pass
  def step(self):
    if self.delay: time.sleep(self.delay)
    outs = []
    for rid, s in list(self.seqs.items()):
      t = s["script"][len(s["out"])]
      s["out"].append(t)
      fin = t == self.eot or len(s["out"]) >= s["p"].max_tokens
      outs.append(self.RO(rid, [t], list(s["out"]), fin, ("stop" if t == self.eot else "length") if fin else None))
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
def serve(ltok):
  from fusezap.serve.api import create_app
  servers = []
  def make(answer, delay: float = 0.0):
    eng = OracleEngine(ltok, answer if callable(answer) else (lambda _p: answer), delay)
    server, th, url = _serve(create_app(eng, ltok, "m"))
    servers.append((server, th))
    return url, eng
  yield make
  for server, th in servers: server.should_exit = True; th.join(10)

def _sse(r: httpx.Response) -> list[dict]:
  assert r.status_code == 200 and r.text.rstrip().endswith("data: [DONE]"), r.text[:300]
  return [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]

def _post(url, **body): return httpx.post(url + "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Weather in Paris?"}], **body}, timeout=30)

def test_http_calls_content_and_choices(serve):
  url, eng = serve('<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}; {"name": "get_time", "parameters": {}}')
  r = _post(url, tools=[WEATHER, CLOCK]).json()
  ch = r["choices"][0]
  assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] is None
  assert _args(ch["message"]["tool_calls"]) == [("get_weather", {"city": "Paris"}), ("get_time", {})]
  assert '"name": "get_weather"' in eng.prompts[-1] and "Environment: ipython" in eng.prompts[-1]
  r = _post(url, tools=[WEATHER, CLOCK], tool_choice="none").json()
  assert "tool_calls" not in r["choices"][0]["message"] and r["choices"][0]["finish_reason"] == "stop" and "get_time" not in eng.prompts[-1]
  # content when the model answers in text; malformed JSON is content too (200, not a 500), for Validate to classify
  for text in ("It is sunny in Paris.", '{"name": "get_weather", "parameters": {"city": "Par'):
    url2, _ = serve(text)
    r = _post(url2, tools=[WEATHER])
    assert r.status_code == 200 and r.json()["choices"][0]["message"] == {"role": "assistant", "content": text} and r.json()["choices"][0]["finish_reason"] == "stop"
  # named: the prompt ends with the forced call start, the model writes the arguments; finish_reason stop (OpenAI, vLLM)
  url3, eng3 = serve('{"city": "Oslo"}}')
  r = _post(url3, tools=[WEATHER, CLOCK], tool_choice={"type": "function", "function": {"name": "get_weather"}}).json()
  assert eng3.prompts[-1].endswith('<|end_header_id|>\n\n{"name": "get_weather", "parameters": ')
  assert r["choices"][0]["finish_reason"] == "stop" and _args(r["choices"][0]["message"]["tool_calls"]) == [("get_weather", {"city": "Oslo"})]
  url4, eng4 = serve('get_time", "parameters": {}}')
  r = _post(url4, tools=[WEATHER, CLOCK], tool_choice="required").json()
  assert eng4.prompts[-1].endswith('\n\n{"name": "') and r["choices"][0]["finish_reason"] == "tool_calls" and _args(r["choices"][0]["message"]["tool_calls"]) == [("get_time", {})]
  rr = _post(url, tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "nope"}})
  assert rr.status_code == 400 and rr.json()["error"]["param"] == "tool_choice"
  # the conversation continues with the calls and their results (two calls: split per message by the server)
  call = ch["message"]["tool_calls"]
  follow = [{"role": "user", "content": "Weather in Paris?"}, {"role": "assistant", "content": None, "tool_calls": call},
            {"role": "tool", "tool_call_id": call[0]["id"], "content": '{"temp_c": 18}'}, {"role": "tool", "tool_call_id": call[1]["id"], "content": "12:00"}]
  assert httpx.post(url + "/v1/chat/completions", json={"messages": follow, "tools": [WEATHER, CLOCK]}).status_code == 200
  assert eng.prompts[-1].count("<|start_header_id|>ipython<|end_header_id|>") == 2

def test_http_streams(serve):
  # a call: held, then per call one delta with id / type / name and "" arguments, one with the arguments; finish tool_calls
  url, _ = serve('{"name": "get_weather", "parameters": {"city": "Paris"}}\n{"name": "get_time", "parameters": {}}')
  ev = _sse(_post(url, tools=[WEATHER, CLOCK], stream=True, stream_options={"include_usage": True}))
  ds = [e["choices"][0]["delta"] for e in ev if e["choices"]]
  assert ds[0] == {"role": "assistant", "content": ""} and not any(d.get("content") for d in ds)
  tc = [d["tool_calls"][0] for d in ds if "tool_calls" in d]
  assert [(x["index"], "id" in x, x["function"]) for x in tc] == [(0, True, {"name": "get_weather", "arguments": ""}), (0, False, {"arguments": '{"city": "Paris"}'}),
                                                                    (1, True, {"name": "get_time", "arguments": ""}), (1, False, {"arguments": "{}"})]
  assert tc[0]["type"] == "function" and tc[0]["id"] != tc[2]["id"]
  assert [e["choices"][0]["finish_reason"] for e in ev if e["choices"]][-1] == "tool_calls" and ev[-1]["usage"]["completion_tokens"] > 0
  # content: streamed live (several deltas; a held stream would send one), finish stop
  url, _ = serve("  It is sunny in Paris today.", delay=0.003)
  ev = _sse(_post(url, tools=[WEATHER], stream=True))
  parts = [e["choices"][0]["delta"].get("content") for e in ev[1:] if e["choices"][0]["delta"].get("content")]
  assert len(parts) >= 2 and "".join(parts) == "  It is sunny in Paris today." and ev[-1]["choices"][0]["finish_reason"] == "stop"
  assert not any("tool_calls" in e["choices"][0]["delta"] for e in ev)
  # malformed JSON: held as a call, then sent as content; length-cut call: content with finish length
  bad = '{"name": "get_weather", "parameters": {"city": '
  url, _ = serve(bad)
  ev = _sse(_post(url, tools=[WEATHER], stream=True))
  assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev) == bad and ev[-1]["choices"][0]["finish_reason"] == "stop"
  url, _ = serve('{"name": "get_weather", "parameters": {"city": "Paris"}}')
  ev = _sse(_post(url, tools=[WEATHER], stream=True, max_tokens=10))
  assert ev[-1]["choices"][0]["finish_reason"] == "length" and not any("tool_calls" in e["choices"][0]["delta"] for e in ev)
  # named, streamed: the arguments the model wrote after the forced prefix, finish stop; n=2 choices each held on their own
  url, _ = serve('{"city": "Oslo"}}')
  ev = _sse(_post(url, tools=[WEATHER, CLOCK], stream=True, n=2, tool_choice={"type": "function", "function": {"name": "get_weather"}}))
  for i in (0, 1):
    mine = [e["choices"][0] for e in ev if e["choices"][0]["index"] == i]
    tc = [c["delta"]["tool_calls"][0] for c in mine if "tool_calls" in c["delta"]]
    assert tc[0]["function"]["name"] == "get_weather" and json.loads(tc[1]["function"]["arguments"]) == {"city": "Oslo"} and mine[-1]["finish_reason"] == "stop"

# ---------------------------------------------------------------- Validate's tools check end to end

def _validate_oracle(tok):
  """The answer of a model that gets tools-v1 right in Llama format: the task is recognised by its rendered prompt (as
  fusezap serve renders it); auto: the expected calls as `{"name": ..., "parameters": ...}` (parallel calls joined by
  "; ", every other task after <|python_tag|>), no-tool tasks a text answer; named / required: the rest of the call
  after the forced prefix. JSON tasks get their expected object."""
  TT = pytest.importorskip("fusezap.validate.tooltasks")   # FuseZap Validate: not in the public export
  tasks, _ = TT.task_set("tools")
  jtasks, _ = TT.task_set("json")
  gp = "<|start_header_id|>assistant<|end_header_id|>\n\n"
  by_prompt = {}
  for k, t in enumerate(tasks):
    msgs = t.get("messages") or [{"role": "user", "content": t["prompt"]}]
    by_prompt[tok.render_chat(TL.request_messages(msgs, "llama3_json"), True, t["tools"])] = (k, t)
  def answer(prompt: str) -> str:
    head, _, tail = prompt.rpartition(gp)
    hit = by_prompt.get(head + gp)
    if hit is None:
      jt = next((x for x in jtasks if x["prompt"] in prompt), None)
      return json.dumps(TT.canonical(jt["expect"])) if jt else "OK"
    k, t = hit
    calls = [(c["name"], TT.canonical(c["args"])) for c in t["calls"]]
    if tail.startswith('{"name": "') and tail.endswith('"parameters": '):   # named
      name = tail[len('{"name": "'):].split('"')[0]
      return json.dumps(next(a for n, a in calls if n == name)) + "}"
    if not calls: return "Here you go. " + " ".join(t.get("answer_contains") or [])
    return ("<|python_tag|>" if k % 2 else "") + "; ".join(json.dumps({"name": n, "parameters": a}) for n, a in calls)
  return answer

def test_validate_tools_check_runs_on_llama_format(serve, ltok):
  """Validate's structured check against fusezap serve with a Llama 3.x tokenizer: the tools check runs in auto and
  named modes (not skipped as in M2's HTTP 400) and a model answering in Llama format scores 93/93, named 81/81,
  parallel 8/8; the same server with a model that writes malformed calls is scored by Validate (parser_error: the reply text holds an unparsed call; no-tool tasks wrong_value),
  never an HTTP error."""
  STR = pytest.importorskip("fusezap.validate.structured")
  url, eng = serve(_validate_oracle(ltok))
  out, ev, _ = asyncio.run(STR.run_structured(url, "m", "candidate", scfg={"concurrency": 16, "max_tokens": 1024}))   # the mini tokenizer: one token per byte
  tl = out["tools"]
  assert tl["modes"]["auto"]["status"] == "ok" and tl["modes"]["named"]["status"] == "ok", {m: v.get("reason") for m, v in tl["modes"].items()}
  assert tl["counts"]["valid_rate"] == {"k": 93, "n": 93} and tl["named_valid_rate"] == 1.0 and tl["parallel_valid_rate"] == 1.0, tl["by_error"]
  assert not [e for e in ev if e["check"].startswith("tools.")]
  url, _ = serve(lambda p: '{"name": "get_weather", "parameters": {"city": ')
  out, _, _ = asyncio.run(STR.run_structured(url, "m", "candidate", scfg={"concurrency": 16, "max_tokens": 1024}))   # the mini tokenizer: one token per byte
  be = out["tools"]["modes"]["auto"]["by_error"]
  assert out["tools"]["modes"]["auto"]["status"] == "ok" and "http_error" not in be and "parser_error" in be and set(be) <= {"parser_error", "no_tool_call", "unexpected_tool_call", "wrong_value"}, be
