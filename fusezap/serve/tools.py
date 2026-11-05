"""Tool calling for `fusezap serve` (OpenAI `tools` / `tool_choice`); strict tools are grammar-constrained.

  spec = tool_spec(body, tok)          # validated tools, the mode ("none" | "auto" | "required" | "named") and the parser
  msgs = request_messages(msgs, spec.fmt)
  spec.forced_prefix                   # text appended after the generation prompt for "required" / a named tool
  call_constraint(spec)                # what the decoder must write after forced_prefix ("required" / named): the hook
                                       # for grammar-constrained decoding (CallConstraint); None for "none" / "auto"
  tool_calls, content = parse(spec, text)
  stream_kind(spec, text_so_far)       # llama3_json streams: "tool" | "content" | None (undecided, keep buffering)

Formats (picked from the tokenizer's vocabulary and chat template, not the model name):
  hermes       Qwen2.5 (and other Hermes-style templates): `<tool_call>\\n{"name": ..., "arguments": {...}}\\n</tool_call>`,
               one block per call, optional text before the first block.
  mistral      Mistral v0.3: `[TOOL_CALLS] [{"name": ..., "arguments": {...}}, ...]` ([TOOL_CALLS] is a special token, so
               the generated ids are decoded with special tokens kept for parsing).
  llama3_json  Llama 3.1 / 3.2 / 3.3 (vLLM's `--tool-call-parser llama3_json`): the vocabulary has `<|python_tag|>` and the
               chat template renders `tools` (JSON function definitions in the first user message, "Respond in the format
               {"name": function name, "parameters": ...}"), assistant tool_calls as that JSON, and `tool` / `ipython`
               messages as ipython turns. The model answers with `{"name": ..., "parameters": {...}}` (or "arguments"),
               optionally after `<|python_tag|>`, several calls separated by `;` / newlines (or as a JSON list). The
               output is a call only when it starts with the JSON (as vLLM); anything else is content. The template takes
               one call per assistant message, so an assistant message with several calls in a request is split into
               one message per call, each followed by its tool result (matched by tool_call_id).
tool_choice "none" renders no tools into the prompt and parses nothing. "auto" renders the tools and parses the output.
"required" and a named function also force the start of the call by appending it to the prompt (hermes:
`<tool_call>\\n{"name": "` or `<tool_call>\\n{"name": "<name>", "arguments": `; llama3_json: `{"name": "` or
`{"name": "<name>", "parameters": `), so the model writes the rest. When the tools involved set "strict": true, the
rest is grammar-constrained (ToolSpec.constraint -> call_constraint -> fusezap/serve/constrain.py: the call's envelope
and arguments matching the parameter schema); otherwise nothing constrains the arguments, so they can still be invalid
JSON, which is then returned as content. tool_choice "auto" is never constrained. A named tool_choice returns the first
call only, and only when it names that function.
`arguments` in the response is always a JSON string; assistant messages in a request may carry it as a string or object.
"""
from __future__ import annotations
import json, re, uuid
from dataclasses import dataclass, field, replace
from typing import Any

class ToolError(ValueError):
  def __init__(self, message: str, param: str = "tools"):
    super().__init__(message); self.param = param

@dataclass
class ToolSpec:
  tools: list[dict] = field(default_factory=list)   # rendered into the template (empty: none)
  mode: str = "none"                                 # none | auto | required | named
  name: str | None = None                            # the named function
  fmt: str = "hermes"
  @property
  def active(self) -> bool: return self.mode != "none" and bool(self.tools)
  @property
  def forced_prefix(self) -> str:
    if self.mode not in ("required", "named"): return ""
    head = {"mistral": "[TOOL_CALLS] [{\"name\": \"", "llama3_json": "{\"name\": \""}.get(self.fmt, "<tool_call>\n{\"name\": \"")
    return head + (f"{self.name}\", \"{self.args_key}\": " if self.mode == "named" else "")
  def constraint(self, tok: Any, prefix_ids: list[int], parallel: bool = True) -> Any:
    """The grammar constraint (constrain.Constraint) of a named / required tool_choice whose tools set "strict": true,
    built from call_constraint(self) (I61's hook) by constrain.call_constraint; None when nothing is constrained."""
    cs = call_constraint(self)
    if cs is None: return None
    if not parallel and cs.parallel: cs = replace(cs, parallel=False)
    from fusezap.serve import constrain as C
    try: return C.call_constraint(cs, tok, prefix_ids)
    except C.ConstraintError as e: raise ToolError(f"strict tools: {e}") from None
  @property
  def args_key(self) -> str: return "parameters" if self.fmt == "llama3_json" else "arguments"   # as the chat template shows the model

LLAMA_TAG = "<|python_tag|>"

def llama_tools_template(src: str | None) -> bool:
  """A Llama 3.1+ chat template that renders custom tools: it uses `tools`, the ipython role / python tag and asks for
  "parameters" (Llama 3.0's template has none of these)."""
  return bool(src) and "tools" in src and "parameters" in src and ("ipython" in src or LLAMA_TAG in src)

def tool_format(tok: Any) -> str | None:
  t = getattr(tok, "tok", None)
  if t is None: return "hermes"
  if t.token_to_id("<tool_call>") is not None: return "hermes"
  if t.token_to_id("[TOOL_CALLS]") is not None: return "mistral"
  if t.token_to_id(LLAMA_TAG) is not None and llama_tools_template(getattr(tok, "template_src", None)): return "llama3_json"
  return None

def tool_spec(body: dict, tok: Any) -> ToolSpec:
  tools, choice = body.get("tools"), body.get("tool_choice")
  if tools is None or tools == []:
    if choice not in (None, "none", "auto"): raise ToolError("'tool_choice' needs 'tools'", "tool_choice")
    return ToolSpec()
  if not isinstance(tools, list): raise ToolError("'tools' must be a list of function tools")
  names = []
  for t in tools:
    f = t.get("function") if isinstance(t, dict) else None
    if not isinstance(t, dict) or t.get("type", "function") != "function" or not isinstance(f, dict) or not isinstance(f.get("name"), str) or not f["name"]:
      raise ToolError("each tool must be {\"type\": \"function\", \"function\": {\"name\": ..., \"parameters\": {...}}}")
    if "parameters" in f and not isinstance(f["parameters"], dict): raise ToolError(f"tool {f['name']!r}: 'parameters' must be a JSON Schema object")
    names.append(f["name"])
  if choice is None: choice = "auto"
  if choice == "none": return ToolSpec()
  fmt = tool_format(tok)
  if fmt is None: raise ToolError("this model's tokenizer has no known tool-call format (supported: Hermes <tool_call>, Mistral [TOOL_CALLS], "
                                "Llama 3.1+ JSON with <|python_tag|> and a tools chat template)")
  if choice in ("auto", "required"): return ToolSpec(tools, choice, None, fmt)
  if isinstance(choice, dict) and choice.get("type", "function") == "function" and isinstance(choice.get("function"), dict):
    name = choice["function"].get("name")
    if name not in names: raise ToolError(f"tool_choice names {name!r}, which is not in 'tools'", "tool_choice")
    return ToolSpec(tools, "named", name, fmt)
  raise ToolError("'tool_choice' must be \"none\", \"auto\", \"required\" or {\"type\": \"function\", \"function\": {\"name\": ...}}", "tool_choice")

def request_messages(msgs: list[dict], fmt: str = "hermes") -> list[dict]:
  """Assistant tool_calls in a request carry `arguments` as a JSON string (OpenAI); templates want an object (as vLLM
  does: parse it, keep it when it is not valid JSON). llama3_json: an empty / null tool_calls is dropped (the template
  tests `'tool_calls' in message`) and an assistant message with n > 1 calls becomes n messages of one call, each followed
  by its tool result when the next messages are those results (the template accepts one call per message)."""
  out = []
  for m in msgs:
    calls = m.get("tool_calls")
    if m.get("role") == "assistant" and isinstance(calls, list):
      fixed = []
      for c in calls:
        f = dict((c or {}).get("function") or {})
        if isinstance(f.get("arguments"), str):
          try: f["arguments"] = json.loads(f["arguments"])
          except json.JSONDecodeError: pass
        fixed.append({**c, "function": f})
      m = {**m, "tool_calls": fixed}
    if fmt == "llama3_json" and "tool_calls" in m and not m["tool_calls"]: m = {k: v for k, v in m.items() if k != "tool_calls"}
    out.append(m)
  return _split_calls(out) if fmt == "llama3_json" else out

def _split_calls(msgs: list[dict]) -> list[dict]:
  out, i = [], 0
  while i < len(msgs):
    m, i = msgs[i], i + 1
    calls = m.get("tool_calls") if m.get("role") == "assistant" else None
    if not isinstance(calls, list) or len(calls) < 2: out.append(m); continue
    j = i
    while j < len(msgs) and msgs[j].get("role") in ("tool", "ipython"): j += 1
    results, i = msgs[i:j], j
    by_id = {r.get("tool_call_id"): r for r in results if r.get("tool_call_id")}
    rest = [r for r in results if not r.get("tool_call_id")]
    for k, c in enumerate(calls):
      out.append({**m, "content": m.get("content") if k == 0 else "", "tool_calls": [c]})
      r = by_id.pop(c.get("id"), None) if isinstance(c, dict) else None
      if r is None and rest: r = rest.pop(0)
      if r is not None: out.append(r)
    out += list(by_id.values()) + rest
  return out

def _call(name: Any, args: Any) -> dict | None:
  if not isinstance(name, str) or not name: return None
  if args is None: args = {}
  if isinstance(args, str):   # some models emit the arguments as a JSON string already
    try: json.loads(args)
    except json.JSONDecodeError: return None
    a = args
  else: a = json.dumps(args, ensure_ascii=False)
  return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": name, "arguments": a}}

_HERMES = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)

def parse(spec: ToolSpec, text: str) -> tuple[list[dict], str | None]:
  """(tool_calls, content) from the generated text, with spec.forced_prefix already prepended by the caller. No call
  parsed: ([], text). A block that is not valid JSON makes the whole output content (nothing half-parsed). A named
  tool_choice keeps the first call, and only when it names that function."""
  calls, content = {"mistral": _parse_mistral, "llama3_json": _parse_llama}.get(spec.fmt, _parse_hermes)(text)
  if calls and spec.mode == "named":
    if calls[0]["function"]["name"] != spec.name: return [], text
    calls = calls[:1]
  return calls, content

def _parse_mistral(text: str) -> tuple[list[dict], str | None]:
  i = text.find("[TOOL_CALLS]")
  if i < 0: return [], text
  body = text[i + len("[TOOL_CALLS]"):].strip()
  try: arr, _ = json.JSONDecoder().raw_decode(body)
  except json.JSONDecodeError: return [], text
  arr = arr if isinstance(arr, list) else [arr]
  calls = [_call(c.get("name"), c.get("arguments")) if isinstance(c, dict) else None for c in arr]
  if not calls or any(c is None for c in calls): return [], text
  return calls, (text[:i].strip() or None)

def _parse_hermes(text: str) -> tuple[list[dict], str | None]:
  i = text.find("<tool_call>")
  if i < 0: return [], text
  calls = []
  for m in _HERMES.finditer(text, i):
    blk = m.group(1).strip()
    if not blk: continue
    try: obj, _ = json.JSONDecoder().raw_decode(blk)
    except json.JSONDecodeError: return [], text
    c = _call(obj.get("name"), obj.get("arguments", obj.get("parameters"))) if isinstance(obj, dict) else None
    if c is None: return [], text
    calls.append(c)
  if not calls: return [], text
  return calls, (text[:i].strip() or None)

_LLAMA_SEP = " \t\r\n;,"

def _parse_llama(text: str) -> tuple[list[dict], str | None]:
  """`[<|python_tag|>]{"name": ..., "parameters": {...}}[; {...}]...` (or a JSON list of such objects) as the whole
  output; anything else (text first, trailing text, an object without a name) is content."""
  s = text.strip()
  if s.startswith(LLAMA_TAG): s = s[len(LLAMA_TAG):].lstrip()
  if not s.startswith(("{", "[")): return [], text
  dec, objs, i = json.JSONDecoder(), [], 0
  while i < len(s):
    try: obj, i = dec.raw_decode(s, i)
    except json.JSONDecodeError: return [], text
    objs += obj if isinstance(obj, list) else [obj]
    while i < len(s) and s[i] in _LLAMA_SEP: i += 1
    if s.startswith(LLAMA_TAG, i): i += len(LLAMA_TAG)
  calls = [_call(o.get("name"), o["parameters"] if "parameters" in o else o.get("arguments")) if isinstance(o, dict) else None for o in objs]
  if not calls or any(c is None for c in calls): return [], text
  return calls, None

def stream_kind(spec: ToolSpec, text: str, final: bool = False) -> str | None:
  """For a live llama3_json stream: "tool" once the output (forced prefix + text so far) starts like a call (buffer it
  to the end, then parse), "content" once it cannot be one (stream it as content from here), None while undecided
  (whitespace, a partial <|python_tag|>, a lone "["). `final`: the output is complete, so undecided means content."""
  s = (spec.forced_prefix + text).lstrip()
  if s.startswith(("{", LLAMA_TAG)) or (s.startswith("[") and s[1:].lstrip().startswith("{")): return "tool"
  if final: return "content"
  if not s or LLAMA_TAG.startswith(s) or (s.startswith("[") and not s[1:].strip()): return None
  return "content"

def streams_live(spec: ToolSpec) -> bool:
  """llama3_json output is a call only when it starts as one, so its stream is sent live (content) or buffered only for
  a call; the other formats allow text before the call, so their streams are sent once generation is done."""
  return spec.active and spec.fmt == "llama3_json"

@dataclass(frozen=True)
class CallConstraint:
  """The hook for grammar-constrained decoding of tool calls (tool_choice "required" or named; call_constraint(spec)).
  The prompt already ends with `prefix` (spec.forced_prefix); the generated text must then be

    named:     JSON(args valid against tools[name]) + close
    required:  NAME + '", "' + args_key + '": ' + JSON(args valid against tools[NAME]) + close     (NAME in tools)

  so that prefix + generated is one call in the model's own format (call_schema() is that call object's JSON Schema).
  `parallel` (required only): further calls may follow, each separator + call_open + NAME + ... + close; a decoder may
  also end after one. The model's stop token comes last. `strict`: the functions sent with "strict": true."""
  fmt: str
  mode: str                        # "required" | "named"
  prefix: str                      # already in the prompt
  args_key: str                    # "arguments" (hermes, mistral) | "parameters" (llama3_json)
  close: str                       # after the arguments object: "}" + the format's end ("}\n</tool_call>", "}]", "}")
  tools: dict                      # function name -> its parameters JSON Schema ({} when the tool has none)
  name: str | None = None          # the named function
  strict: frozenset = frozenset()
  call_open: str = ""              # a further call's text before NAME (hermes '<tool_call>\n{"name": "', llama3_json '{"name": "')
  separator: str = ""              # between calls (hermes "\n", llama3_json "; ")
  parallel: bool = False
  def call_schema(self, name: str | None = None) -> dict:
    names = [name or self.name] if (name or self.name) else list(self.tools)
    one = [{"type": "object", "properties": {"name": {"const": n}, self.args_key: self.tools[n] or {"type": "object"}},
            "required": ["name", self.args_key], "additionalProperties": False} for n in names]
    return one[0] if len(one) == 1 else {"anyOf": one}

def call_constraint(spec: ToolSpec) -> CallConstraint | None:
  if not spec.active or spec.mode not in ("required", "named"): return None
  fns = [t["function"] for t in spec.tools]
  close = {"mistral": "}]", "llama3_json": "}"}.get(spec.fmt, "}\n</tool_call>")
  call_open = {"mistral": "", "llama3_json": "{\"name\": \""}.get(spec.fmt, "<tool_call>\n{\"name\": \"")
  return CallConstraint(spec.fmt, spec.mode, spec.forced_prefix, spec.args_key, close, {f["name"]: f.get("parameters") or {} for f in fns}, spec.name,
                        frozenset(f["name"] for f in fns if f.get("strict") is True), call_open,
                        {"mistral": "", "llama3_json": "; "}.get(spec.fmt, "\n"), spec.mode == "required" and spec.fmt != "mistral")

def check_response_format(body: dict, tok: Any = None) -> Any:
  """response_format -> its constrain.Constraint, or None for "text" / none. json_object: any JSON object; json_schema:
  JSON valid against json_schema.schema (enforced whether or not json_schema.strict is set, as vLLM does). Both need the
  optional llguidance package (grammar-constrained decoding, fusezap/serve/constrain.py): without it, or for a schema it
  cannot enforce, a 400 that says so."""
  rf = body.get("response_format")
  if rf is None: return None
  if not isinstance(rf, dict) or not isinstance(rf.get("type"), str):
    raise ToolError("'response_format' must be an object with a 'type'", "response_format")
  t = rf["type"]
  if t == "text": return None
  if t == "json_schema":
    js = rf.get("json_schema")
    if not isinstance(js, dict) or not isinstance(js.get("schema", {}), dict) or not isinstance(js.get("name", "x"), str):
      raise ToolError("'response_format.json_schema' must be {\"name\": ..., \"schema\": {...}}", "response_format")
  if t not in ("json_object", "json_schema"):
    raise ToolError(f"unknown response_format type {t!r} (expected text, json_object or json_schema)", "response_format")
  from fusezap.serve import constrain as C
  try: return C.json_object(tok) if t == "json_object" else C.json_schema(js.get("schema") or {"type": "object"}, tok)
  except C.ConstraintError as e: raise ToolError(f"response_format {t!r}: {e}", "response_format") from None
