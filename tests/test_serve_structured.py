"""Tool calling and structured output in `fusezap serve` (I14), beyond tests/test_serve_constrain.py (I62),
tests/test_serve_llama_tools.py (I61) and tests/test_serve_openai_ext.py:

  - fast: live Hermes (Qwen2.5) streams: content sent as it is generated up to `<tool_call>`, the calls held to the end
    of the choice, the reassembled stream equal to the non-streamed reply (calls, content, finish reason), malformed
    blocks and length cuts sent as content; the official `openai` Python client against the HTTP app (fake engine,
    Qwen2.5 tokenizer): tool calls, streamed tool calls accumulated as the client sees them, response_format;
  - fast, NEURON: grammar-constrained requests on the NEURON serving path (NKISHIM, and the persistent NEFF path with
    the fake runtime of tests/test_neuron_nrt.py): the mask is applied on the host to the step's logits, so the device
    forwards (fixed shapes, head fused into the last layers, device argmax for the other rows) are unchanged and the
    tokens equal CPU's;
  - slow (FZ_SLOW=1): Qwen2.5-0.5B-Instruct served on METAL (else CPU; FZ_DEVICE overrides) with the `openai` client: json_schema /
    json_object valid, strict named / required tool calls valid against the tool's schema, auto tool calls streamed
    = non-streamed; and the host-side mask's cost (per-token mask time, TPOT with and without a constraint).
Skips what needs llguidance / openai / a cached tokenizer when they are missing.
"""
from __future__ import annotations
import json, os, time
import numpy as np
import pytest
from fusezap.serve import tools as TL
from test_neuron_nrt import nrt, neuron_dev   # noqa: F401  (fixtures: the fake NEURON runtime)

QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
WORDS = ['{"', '": ', '", "', '"}', ', "', 'true', 'false', 'null', 'name', 'type', 'city', 'tags', '[', ']}']

def tiny_tokenizer(d: str) -> str:
  """A byte-level BPE tokenizer.json over printable ASCII + newline and a few JSON merges, `<eos>` last (its id is the vocab size - 1)."""
  from tokenizers import Tokenizer, models, AddedToken, pre_tokenizers, decoders
  B = lambda s: s.replace(" ", "Ġ").replace("\n", "Ċ")
  vocab = {B(chr(c)): i for i, c in enumerate([*range(32, 127), 10])}
  merges = []
  for w in WORDS:
    cur = B(w[0])
    for ch in w[1:]:
      nxt = cur + B(ch)
      if nxt not in vocab: vocab[nxt] = len(vocab); merges.append((cur, B(ch)))
      cur = nxt
  tk = Tokenizer(models.BPE(vocab=vocab, merges=merges))
  tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
  tk.decoder = decoders.ByteLevel()
  tk.add_special_tokens([AddedToken("<eos>", special=True)])
  os.makedirs(d, exist_ok=True)
  tk.save(os.path.join(d, "tokenizer.json"))
  return d

class TinyTok:
  """The part of serve.tokenizer.Tokenizer that constrain uses (path, tok) plus encode/decode."""
  def __init__(self, d: str):
    from tokenizers import Tokenizer
    self.path, self.tok = d, Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    self.eos = self.tok.token_to_id("<eos>")
  def encode(self, s: str, add_special_tokens: bool = False) -> list[int]: return self.tok.encode(s, add_special_tokens=False).ids
  def decode(self, ids, skip_special_tokens: bool = True) -> str: return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

def _valid(v, schema: dict) -> bool:
  """Enough JSON Schema for these tests: type object / string / integer / boolean / array, enum, properties, required,
  additionalProperties false, items, integer minimum / maximum."""
  t = schema.get("type")
  if "enum" in schema and v not in schema["enum"]: return False
  if t == "object":
    if not isinstance(v, dict) or not set(schema.get("required", [])) <= set(v): return False
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False and set(v) - set(props): return False
    return all(_valid(v[k], props[k]) for k in v if k in props)
  if t == "array": return isinstance(v, list) and len(v) <= schema.get("maxItems", len(v)) and all(_valid(x, schema.get("items", {})) for x in v)
  if t == "string": return isinstance(v, str) and len(v) <= schema.get("maxLength", len(v))
  if t == "boolean": return isinstance(v, bool)
  if t == "integer":
    return isinstance(v, int) and not isinstance(v, bool) and schema.get("minimum", v) <= v <= schema.get("maximum", v)
  return True

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
CLOCK = {"type": "function", "function": {"name": "get_time", "parameters": {"type": "object", "properties": {}}}}

# ---------------------------------------------------------------- live Hermes streams

def test_hermes_split():
  sp = TL.ToolSpec([WEATHER], "auto", None, "hermes")
  assert TL.hermes_split(sp, "It is") == ("It is", "", False)
  assert TL.hermes_split(sp, "It is ") == ("It is", " ", False)                     # trailing whitespace held
  assert TL.hermes_split(sp, "Let me check.\n<tool") == ("Let me check.", "\n<tool", False)   # a partial tag held
  assert TL.hermes_split(sp, "a <b") == ("a <b", "", False)                          # not a tag prefix
  assert TL.hermes_split(sp, "Sure.\n\n<tool_call>\n{") == ("Sure.", "\n\n<tool_call>\n{", True)
  assert TL.hermes_split(sp, "<tool_call>") == ("", "<tool_call>", True)
  assert TL.hermes_split(sp, "x <tool", final=True) == ("x <tool", "", False)
  named = TL.ToolSpec([WEATHER], "named", "get_weather", "hermes")
  assert TL.hermes_split(named, '{"city"') == ("", '{"city"', True)                  # starts inside the call
  assert TL.streams_live(sp) and not TL.streams_live(TL.ToolSpec([WEATHER], "auto", None, "mistral"))

def _assemble(ev: list[dict]) -> dict:
  """A streamed chat choice put back together as a client does: content, tool_calls by index, finish reason."""
  content, calls, fin = "", {}, None
  for e in ev:
    if not e["choices"]: continue
    c = e["choices"][0]
    d = c["delta"]
    content += d.get("content") or ""
    for tc in d.get("tool_calls") or []:
      x = calls.setdefault(tc["index"], {"id": None, "name": "", "arguments": ""})
      x["id"] = tc.get("id") or x["id"]
      f = tc.get("function") or {}
      x["name"] += f.get("name") or ""
      x["arguments"] += f.get("arguments") or ""
    fin = c["finish_reason"] or fin
  return {"content": content or None, "calls": [calls[k] for k in sorted(calls)], "finish": fin}

def _message(r: dict) -> dict:
  ch = r["choices"][0]
  m = ch["message"]
  return {"content": m.get("content"), "calls": [{"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                                                 for c in m.get("tool_calls") or []], "finish": ch["finish_reason"]}

HERMES_CASES = {
  "content": "It is sunny in Paris today, around 18 degrees.",
  "call": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
  "text_then_calls": 'Let me look that up.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
                     '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>',
  "malformed": 'Checking.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": }\n</tool_call>',
  "tag_in_text": "Use the <tool> tag, or <tool_ for short.",
}

@pytest.mark.parametrize("case", list(HERMES_CASES))
def test_hermes_stream_equals_reply(case):
  """The live stream reassembled is the non-streamed reply (ids aside); content before a call arrives in several deltas,
  before the call is generated."""
  import httpx
  from tests.test_serve_openai_ext import FakeEngine, _sse, _serve, _tok
  from fusezap.serve.api import create_app
  tok = _tok(QWEN)
  script = HERMES_CASES[case]
  class Slow(FakeEngine):   # one token every 3 ms: a live stream shows up as many deltas
    def step(self): time.sleep(0.003); return super().step()
  server, th, url = _serve(create_app(Slow(tok, script), tok, "m"))
  try:
    body = {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER, CLOCK]}
    want = _message(httpx.post(url + "/v1/chat/completions", json=body, timeout=30).json())
    ev = _sse(httpx.post(url + "/v1/chat/completions", json={**body, "stream": True}, timeout=30))
    got = _assemble(ev)
    for c in (*got["calls"], *want["calls"]): assert c.pop("id").startswith("call_")
    assert got == want, (got, want)
    parts = [e["choices"][0]["delta"].get("content") for e in ev[1:] if e["choices"][0]["delta"].get("content")]
    if case in ("content", "text_then_calls"): assert len(parts) >= 3, parts   # live, not one held chunk
    if case == "text_then_calls":
      assert [c["name"] for c in got["calls"]] == ["get_weather", "get_time"] and got["content"] == "Let me look that up."
    if case == "malformed": assert not got["calls"] and got["content"] == script and got["finish"] == "stop"
  finally:
    server.should_exit = True; th.join(10)

def test_hermes_stream_length_cut_and_named():
  import httpx
  from tests.test_serve_openai_ext import FakeEngine, _sse, _serve, _tok
  from fusezap.serve.api import create_app
  tok = _tok(QWEN)
  call = 'Ok.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
  server, th, url = _serve(create_app(FakeEngine(tok, call), tok, "m"))
  try:
    body = {"messages": [{"role": "user", "content": "Weather in Paris?"}], "tools": [WEATHER], "stream": True, "max_tokens": 8}
    got = _assemble(_sse(httpx.post(url + "/v1/chat/completions", json=body, timeout=30)))
    assert got["finish"] == "length" and not got["calls"] and got["content"].startswith("Ok.\n<tool_call>")
  finally:
    server.should_exit = True; th.join(10)
  server, th, url = _serve(create_app(FakeEngine(tok, '{"city": "Oslo"}}\n</tool_call>'), tok, "m"))
  try:
    body = {"messages": [{"role": "user", "content": "Oslo?"}], "tools": [WEATHER, CLOCK], "stream": True, "n": 2,
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}
    ev = _sse(httpx.post(url + "/v1/chat/completions", json=body, timeout=30))
    for i in (0, 1):
      got = _assemble([e for e in ev if e["choices"] and e["choices"][0]["index"] == i])
      assert got["finish"] == "stop" and [c["name"] for c in got["calls"]] == ["get_weather"] and json.loads(got["calls"][0]["arguments"]) == {"city": "Oslo"}
  finally:
    server.should_exit = True; th.join(10)

# ---------------------------------------------------------------- the official openai client (fake engine)

def test_openai_client_tools_and_response_format():
  openai = pytest.importorskip("openai")
  from tests.test_serve_openai_ext import FakeEngine, _serve, _tok
  from fusezap.serve.api import create_app
  tok = _tok(QWEN)
  two = ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
         '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>')
  server, th, url = _serve(create_app(FakeEngine(tok, two), tok, "m"))
  try:
    cl = openai.OpenAI(base_url=url + "/v1", api_key="x")
    msgs = [{"role": "user", "content": "Weather and time in Paris?"}]
    r = cl.chat.completions.create(model="m", messages=msgs, tools=[WEATHER, CLOCK])
    ch = r.choices[0]
    assert ch.finish_reason == "tool_calls" and ch.message.content is None
    assert [(c.type, c.function.name, json.loads(c.function.arguments)) for c in ch.message.tool_calls] == [("function", "get_weather", {"city": "Paris"}), ("function", "get_time", {})]
    # streamed: accumulate the deltas the way the client exposes them
    calls: dict[int, dict] = {}
    fin = None
    for c in cl.chat.completions.create(model="m", messages=msgs, tools=[WEATHER, CLOCK], stream=True):
      d = c.choices[0].delta
      for tc in d.tool_calls or []:
        x = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
        x["id"] = tc.id or x["id"]
        x["name"] += (tc.function.name or "") if tc.function else ""
        x["arguments"] += (tc.function.arguments or "") if tc.function else ""
      fin = c.choices[0].finish_reason or fin
    assert fin == "tool_calls" and [(x["name"], json.loads(x["arguments"])) for x in calls.values()] == [("get_weather", {"city": "Paris"}), ("get_time", {})]
    assert all(x["id"].startswith("call_") for x in calls.values())
    # the conversation goes on with the calls and their results
    follow = msgs + [ch.message.model_dump(exclude_none=True)] + [{"role": "tool", "tool_call_id": c.id, "content": "18C" if i == 0 else "12:00"}
                                                                   for i, c in enumerate(ch.message.tool_calls)]
    assert cl.chat.completions.create(model="m", messages=follow, tools=[WEATHER, CLOCK]).choices[0].message is not None
    with pytest.raises(openai.BadRequestError):
      cl.chat.completions.create(model="m", messages=msgs, tools=[WEATHER], tool_choice={"type": "function", "function": {"name": "nope"}})
  finally:
    server.should_exit = True; th.join(10)
  pytest.importorskip("llguidance")
  server, th, url = _serve(create_app(FakeEngine(tok, '{"city": "Paris", "temp_c": 18}'), tok, "m"))
  try:
    cl = openai.OpenAI(base_url=url + "/v1", api_key="x")
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "temp_c": {"type": "integer"}}, "required": ["city", "temp_c"], "additionalProperties": False}
    for rf in ({"type": "json_object"}, {"type": "json_schema", "json_schema": {"name": "w", "schema": schema, "strict": True}}):
      r = cl.chat.completions.create(model="m", messages=[{"role": "user", "content": "Paris weather as JSON"}], response_format=rf)
      assert json.loads(r.choices[0].message.content) == {"city": "Paris", "temp_c": 18}
      s = "".join(c.choices[0].delta.content or "" for c in cl.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}], response_format=rf, stream=True))
      assert json.loads(s) == {"city": "Paris", "temp_c": 18}
    with pytest.raises(openai.BadRequestError):
      cl.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}], response_format={"type": "json_schema", "json_schema": {"name": "w", "schema": {"type": "array", "uniqueItems": True}}})
  finally:
    server.should_exit = True; th.join(10)

# ---------------------------------------------------------------- NEURON: host-side masking on the fixed-shape decode path

SCHEMA = {"type": "object", "properties": {"type": {"type": "string", "enum": ["a", "b"]}, "ok": {"type": "boolean"},
                                           "n": {"type": "integer", "minimum": 0, "maximum": 999}}, "required": ["type", "ok", "n"], "additionalProperties": False}

def _constrained_run(model_dir: str, tok: TinyTok, device: str) -> tuple[list[list[int]], object]:
  """Two unconstrained greedy rows and two schema-constrained rows (greedy, and sampled with a seed) in one batch."""
  from fusezap.dtype import dtypes
  from fusezap.serve import constrain as C
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.models import load_model
  m = load_model(model_dir, dtype=dtypes.float32, device=device, max_len=128, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
  cs = C.json_schema(SCHEMA, tok)
  free = SamplingParams(max_tokens=12, ignore_eos=True)
  ps = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7], [40, 41, 42, 43], [5] * 20]
  params = [free, free, SamplingParams(max_tokens=60, constraint=cs), SamplingParams(max_tokens=60, temperature=1.0, seed=3, constraint=cs)]
  return eng.generate(ps, params), eng

@pytest.fixture(scope="module")
def tiny_json(tmp_path_factory):
  pytest.importorskip("llguidance")
  from tests.test_tp import tiny_model_dir
  tok = TinyTok(tiny_tokenizer(str(tmp_path_factory.mktemp("tok"))))
  V = tok.tok.get_vocab_size()
  d = tiny_model_dir(str(tmp_path_factory.mktemp("m")), seed=1, cfg={"vocab_size": V + 6, "eos_token_id": tok.eos})
  cpu, _ = _constrained_run(d, tok, "CPU")
  for o in cpu[2:]:
    assert _valid(json.loads(tok.decode(o)), SCHEMA), tok.decode(o)
  return d, tok, cpu

def test_constrained_on_nkishim_matches_cpu(tiny_json, monkeypatch):
  """NKISHIM (the NEURON kernels simulated on the host), N8 defaults (layers and head fused, device argmax): constrained
  rows masked on the host give CPU's tokens; the unconstrained rows of the same steps keep theirs."""
  from fusezap.codegen import nki_kernels as nk
  d, tok, cpu = tiny_json
  got, eng = _constrained_run(d, tok, "NKISHIM")
  assert type(eng.cache).__name__ == "NeuronKVCache" and eng.device_argmax and nk.head_argmax()
  assert got == cpu

def test_constrained_on_the_persistent_neuron_path(tiny_json, nrt, neuron_dev):
  """The same on NEURON's persistent NEFF path (fake nrtpy running the kernels through NKISHIM): CPU's tokens."""
  d, tok, cpu = tiny_json
  got, eng = _constrained_run(d, tok, "NEURON")
  assert got == cpu and eng.device_argmax

# ---------------------------------------------------------------- Qwen2.5-0.5B with the openai client (slow)

@pytest.fixture(scope="module")
def real():
  openai = pytest.importorskip("openai")
  pytest.importorskip("llguidance")
  from fusezap.serve.weights import resolve_model
  from fusezap.serve.tokenizer import Tokenizer
  try: path = resolve_model(QWEN, allow_download=False)
  except Exception as e: pytest.skip(f"{QWEN} not in the HF cache: {e}")
  from tests.test_serve_openai_ext import _serve
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine
  from fusezap.serve.api import create_app, warmup
  from tests.test_scheduler import _device
  dev = _device()   # METAL on a Mac (Qwen2.5-0.5B float32 on the CPU C backend is minutes per request), else CPU; FZ_DEVICE overrides
  m = load_model(QWEN, dtype=dtypes.float32, device=dev, max_len=1024, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=256, block_size=16, max_num_seqs=4, max_prefill_tokens=256, decode_beam=0)
  eng.tokenizer_path = os.path.join(path, "tokenizer.json")
  warmup(eng)
  server, th, url = _serve(create_app(eng, Tokenizer(path), QWEN))
  yield openai.OpenAI(base_url=url + "/v1", api_key="x", timeout=600), eng, path, dev
  server.should_exit = True; th.join(30)

PERSON = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 40}, "age": {"type": "integer", "minimum": 0, "maximum": 150},   # bounded:
                                           "langs": {"type": "array", "items": {"type": "string", "maxLength": 20}, "maxItems": 4},   # a 0.5B model loops in a free string
                                           "role": {"type": "string", "enum": ["dev", "ops", "pm"]}},
          "required": ["name", "age", "langs", "role"], "additionalProperties": False}
STRICT_WEATHER = {"type": "function", "function": {"name": "get_weather", "strict": True, "description": "Current weather for a city",
                                                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                                                                  "required": ["city", "unit"], "additionalProperties": False}}}

@pytest.mark.slow
def test_real_openai_client_structured(real):
  cl, eng, _, dev = real
  ask = [{"role": "user", "content": "Describe Ana, a 31 year old operations engineer who writes Go and Python, as JSON."}]
  for stream in (False, True):
    rf = {"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON, "strict": True}}
    if stream: text = "".join(c.choices[0].delta.content or "" for c in cl.chat.completions.create(model=QWEN, messages=ask, response_format=rf, max_tokens=200, temperature=0, stream=True))
    else: text = cl.chat.completions.create(model=QWEN, messages=ask, response_format=rf, max_tokens=200, temperature=0).choices[0].message.content
    assert _valid(json.loads(text), PERSON), text
  for seed in range(3):   # sampled: still valid
    r = cl.chat.completions.create(model=QWEN, messages=ask, response_format={"type": "json_schema", "json_schema": {"name": "person", "schema": PERSON}}, max_tokens=200, temperature=1.0, seed=seed)
    assert _valid(json.loads(r.choices[0].message.content), PERSON), r.choices[0].message.content
  q = [{"role": "user", "content": "What's the weather in Lyon, in celsius?"}]
  for choice in ({"type": "function", "function": {"name": "get_weather"}}, "required"):
    r = cl.chat.completions.create(model=QWEN, messages=q, tools=[STRICT_WEATHER, CLOCK], tool_choice=choice, max_tokens=60, temperature=0)
    tc = r.choices[0].message.tool_calls
    assert tc and tc[0].function.name in ("get_weather", "get_time"), r
    if tc[0].function.name == "get_weather": assert _valid(json.loads(tc[0].function.arguments), STRICT_WEATHER["function"]["parameters"])
  # auto: streamed = non-streamed (greedy)
  one = cl.chat.completions.create(model=QWEN, messages=q, tools=[WEATHER, CLOCK], max_tokens=60, temperature=0)
  calls, content = {}, ""
  for c in cl.chat.completions.create(model=QWEN, messages=q, tools=[WEATHER, CLOCK], max_tokens=60, temperature=0, stream=True):
    d = c.choices[0].delta
    content += d.content or ""
    for t in d.tool_calls or []:
      x = calls.setdefault(t.index, ["", ""])
      x[0] += (t.function.name or "") if t.function else ""
      x[1] += (t.function.arguments or "") if t.function else ""
  want = [[t.function.name, t.function.arguments] for t in one.choices[0].message.tool_calls or []]
  assert [calls[k] for k in sorted(calls)] == want and (content or None) == one.choices[0].message.content

@pytest.mark.slow
def test_real_mask_overhead(real):
  """The host-side mask's cost: llguidance's per-token mask on the 151,665-token Qwen2.5 vocabulary (plus the row's
  -inf fill) against the time of a decode step, and TPOT with and without response_format json_object."""
  import httpx
  from fusezap.serve import constrain as C
  from fusezap.serve.tokenizer import Tokenizer
  cl, eng, path, dev = real
  tok = Tokenizer(path)
  V = eng.model.cfg.vocab_size
  c = C.json_schema(PERSON, tok)
  text = json.dumps({"name": "Ana", "age": 31, "langs": ["Go", "Python"], "role": "ops"})
  ids = tok.encode(text, add_special_tokens=False)
  m = c.matcher(V, sorted(eng.stop_ids))
  row = np.random.default_rng(0).standard_normal(V).astype(np.float32)
  t_mask = []
  for t in ids:
    t0 = time.perf_counter(); m.mask(row); t_mask.append(time.perf_counter() - t0)
    m.advance(t)
  url = cl.base_url.__str__().rstrip("/").removesuffix("/v1")
  body = {"model": QWEN, "messages": [{"role": "user", "content": "List twenty fruits as a JSON object."}], "max_tokens": 48, "temperature": 0, "ignore_eos": True}
  def tpot(b):
    t0 = time.perf_counter(); r = httpx.post(url + "/v1/chat/completions", json=b, timeout=600).json()
    return (time.perf_counter() - t0) / r["usage"]["completion_tokens"]
  tpot(body)
  free = min(tpot(body) for _ in range(2))
  cons = min(tpot({**body, "response_format": {"type": "json_object"}}) for _ in range(2))
  print(f"\n[{dev}] mask p50 {np.median(t_mask) * 1e3:.3f} ms, max {max(t_mask) * 1e3:.3f} ms over {len(ids)} tokens (V={V}); "
        f"TPOT unconstrained {free * 1e3:.2f} ms, json_object {cons * 1e3:.2f} ms (+{(cons - free) * 1e3:.2f} ms)")
  assert np.median(t_mask) < 0.02 and cons < free * 1.5 + 0.005

@pytest.mark.slow
def test_cpu_mask_overhead_tiny_model(tmp_path):
  """The host mask's cost on the CPU backend, where it shares the cores with the forward: a tiny random model with the
  Qwen2.5 vocabulary (151,936 logits per row, so the mask and the logits read are full size while the forward is
  small: the worst case for the relative overhead), one request, TPOT with and without a JSON-schema constraint."""
  pytest.importorskip("llguidance")
  from tests.test_serve_openai_ext import _tok
  from tests.test_tp import tiny_model_dir
  from fusezap.dtype import dtypes
  from fusezap.serve import constrain as C
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.models import load_model
  tok = _tok(QWEN)
  V = 151936
  d = tiny_model_dir(str(tmp_path / "m"), seed=2, cfg={"vocab_size": V, "eos_token_id": tok.stop_token_ids[0]})
  m = load_model(d, dtype=dtypes.float32, device="CPU", max_len=256, prefill_chunk=32)
  eng = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=1, max_prefill_tokens=32, max_model_len=256, decode_beam=0)
  cs = C.json_schema({"type": "object", "properties": {"notes": {"type": "string"}}, "required": ["notes"]}, tok)   # a long free string
  def tpot(params, n=3):
    best = 1e9
    for _ in range(n):
      t0 = time.perf_counter(); out = eng.generate([[1, 2, 3, 4]], params)[0]; best = min(best, (time.perf_counter() - t0) / len(out))
    return best, len(out)
  tpot(SamplingParams(max_tokens=64, ignore_eos=True), 1); tpot(SamplingParams(max_tokens=64, constraint=cs), 1)   # compile / warm
  free, nf = tpot(SamplingParams(max_tokens=64, ignore_eos=True))
  cons, nc = tpot(SamplingParams(max_tokens=64, constraint=cs))
  print(f"\n[CPU tiny, V={V}] TPOT unconstrained {free * 1e3:.2f} ms ({nf} tokens), json_schema {cons * 1e3:.2f} ms ({nc} tokens), +{(cons - free) * 1e3:.2f} ms/token")
  assert nc >= 8 and cons < free + 0.005
