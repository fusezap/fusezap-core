"""Thin tokenizer wrapper over the `tokenizers` library, with the HF chat template rendered in Python (SPEC §11.3).

  tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
  ids = tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True)
  text = tok.decode(ids)

Templates: Qwen2.5's ChatML template (incl. tools / tool_calls / tool messages, with the default system prompt read from
the template) is rendered in Python, so jinja is not needed for it. Every other template is rendered by jinja2 in the same environment as HF's apply_chat_template (sandboxed, trim/lstrip blocks,
loopcontrols, `{% generation %}`, tojson, raise_exception, strftime_now): Llama 3.1+ templates render `tools` and tool / ipython messages this way. Without jinja2, other ChatML templates get a
plain ChatML renderer (default system prompt from the template, no tools) and plain Llama 3 templates a header renderer. The template comes from chat_template.jinja
when the model ships one (as HF prefers), else tokenizer_config.json. `IncrementalDecoder` turns a token stream into
text deltas without splitting UTF-8.
"""
from __future__ import annotations
import json, os, re
from typing import Any, Sequence

def _tojson(x: Any) -> str: return json.dumps(x, ensure_ascii=False)   # HF's jinja `tojson` filter

def _get(m: Any, k: str, default=None):
  return m.get(k, default) if isinstance(m, dict) else getattr(m, k, default)

class ChatMLTemplate:
  """Qwen2.5's template (tokenizer_config.json chat_template), rendered in Python. With qwen_tools=False it is plain
  ChatML (the no-jinja fallback for other ChatML templates): system/user/assistant turns only, and tools are refused."""
  def __init__(self, default_system: str | None, qwen_tools: bool = True):
    self.default_system, self.qwen_tools = default_system, qwen_tools
  def render(self, messages: Sequence[dict], add_generation_prompt: bool = False, tools: Sequence[dict] | None = None) -> str:
    out, msgs = [], list(messages)
    first_sys = bool(msgs) and _get(msgs[0], "role") == "system"
    if not self.qwen_tools and (tools or any(_get(m, "role") == "tool" or _get(m, "tool_calls") for m in msgs)):
      raise ValueError("this model's chat template needs jinja2 for tools; install jinja2")
    if tools:
      out.append("<|im_start|>system\n")
      out.append(_get(msgs[0], "content") if first_sys else (self.default_system or ""))
      out.append("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with "
                 "function signatures within <tools></tools> XML tags:\n<tools>")
      for t in tools: out.append("\n" + _tojson(t))
      out.append("\n</tools>\n\nFor each function call, return a json object with function name and arguments within "
                 "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
                 "</tool_call><|im_end|>\n")
    elif first_sys: out.append("<|im_start|>system\n" + _get(msgs[0], "content") + "<|im_end|>\n")
    elif self.default_system is not None: out.append("<|im_start|>system\n" + self.default_system + "<|im_end|>\n")
    for i, m in enumerate(msgs):
      role, content, calls = _get(m, "role"), _get(m, "content"), _get(m, "tool_calls")
      if role == "user" or (role == "system" and i != 0) or (role == "assistant" and not calls):
        out.append(f"<|im_start|>{role}\n{content or ''}<|im_end|>\n")
      elif role == "assistant":
        out.append("<|im_start|>assistant")
        if content: out.append("\n" + content)
        for tc in calls:
          f = _get(tc, "function", tc)
          args = _get(f, "arguments")
          out.append('\n<tool_call>\n{"name": "' + _get(f, "name") + '", "arguments": ' + _tojson(args) + "}\n</tool_call>")
        out.append("<|im_end|>\n")
      elif role == "tool":
        if i == 0 or _get(msgs[i - 1], "role") != "tool": out.append("<|im_start|>user")
        out.append("\n<tool_response>\n" + (content or "") + "\n</tool_response>")
        if i == len(msgs) - 1 or _get(msgs[i + 1], "role") != "tool": out.append("<|im_end|>\n")
    if add_generation_prompt: out.append("<|im_start|>assistant\n")
    return "".join(out)

class Llama3Template:
  """Llama 3 header format (no tools, no date preamble): the no-jinja fallback for Llama 3.0's template."""
  def __init__(self, bos: str): self.bos = bos
  def render(self, messages: Sequence[dict], add_generation_prompt: bool = False, tools: Sequence[dict] | None = None) -> str:
    assert not tools, "tools are not supported by the built-in Llama 3 template"
    out = [self.bos]
    for m in messages: out.append(f"<|start_header_id|>{_get(m, 'role')}<|end_header_id|>\n\n{(_get(m, 'content') or '').strip()}<|eot_id|>")
    if add_generation_prompt: out.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
    return "".join(out)

QWEN25_MARKERS = ("# Tools\\n\\nYou may call one or more functions", "<tool_response>", "message.tool_calls", "<|im_start|>")
_STR = r"""(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)")"""   # a jinja string literal (may hold literal newlines)

def chatml_default_system(src: str) -> str | None:
  """The default system prompt a ChatML template emits as one `{{ '<|im_start|>system\n...<|im_end|>\n' }}` literal."""
  for m in re.finditer(r"\{\{-?\s*" + _STR + r"\s*-?\}\}", src):
    lit = m.group(1) if m.group(1) is not None else m.group(2)
    lit = re.sub(r"\\(.)", lambda e: {"n": "\n", "t": "\t"}.get(e.group(1), e.group(1)), lit)
    if lit.startswith("<|im_start|>system\n") and lit.endswith("<|im_end|>\n"):
      return lit[len("<|im_start|>system\n"):-len("<|im_end|>\n")]
  return None

def hf_jinja_env():
  """The jinja2 environment of HF's apply_chat_template (transformers.utils.chat_template_utils)."""
  import jinja2, jinja2.ext, jinja2.nodes
  from datetime import datetime
  from jinja2.sandbox import ImmutableSandboxedEnvironment
  class GenerationTag(jinja2.ext.Extension):   # `{% generation %}...{% endgeneration %}` renders its body unchanged
    tags = {"generation"}
    def parse(self, parser):
      lineno = next(parser.stream).lineno
      body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
      return jinja2.nodes.CallBlock(self.call_method("_body"), [], [], body).set_lineno(lineno)
    def _body(self, caller): return caller()
  def raise_exception(msg): raise jinja2.exceptions.TemplateError(msg)
  def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
  env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationTag, jinja2.ext.loopcontrols])
  env.filters["tojson"] = tojson
  env.globals["raise_exception"] = raise_exception
  env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
  return env

class JinjaTemplate:
  def __init__(self, src: str, special: dict[str, Any]):
    self.tpl, self.special = hf_jinja_env().from_string(src), special
  def render(self, messages: Sequence[dict], add_generation_prompt: bool = False, tools: Sequence[dict] | None = None) -> str:
    return self.tpl.render(messages=list(messages), tools=list(tools) if tools else None, documents=None,
                           add_generation_prompt=add_generation_prompt, **self.special)

def _has_jinja() -> bool:
  try: import jinja2; return True  # noqa: F401
  except ImportError: return False

def special_tokens(cfg: dict) -> dict[str, Any]:
  """HF's special_tokens_map as template variables: bos_token, eos_token, ... (strings) and additional_special_tokens."""
  def s(v): return v.get("content", "") if isinstance(v, dict) else v
  out: dict[str, Any] = {k: s(v) for k, v in cfg.items() if k.endswith("_token") and isinstance(v, (str, dict)) and s(v)}
  if cfg.get("additional_special_tokens"): out["additional_special_tokens"] = [s(v) for v in cfg["additional_special_tokens"]]
  return out

def template_source(cfg: dict, src: str | None = None) -> str | None:
  """The chat template text: chat_template.jinja's (src) or tokenizer_config's (the "default" one of a named list)."""
  if src is None: src = cfg.get("chat_template")
  if isinstance(src, list): src = next((t["template"] for t in src if t.get("name") == "default"), src[0]["template"])
  return src or None

def make_template(cfg: dict, src: str | None = None) -> Any:
  src = template_source(cfg, src)
  if not src: return ChatMLTemplate(None)
  special = special_tokens(cfg)
  if all(k in src for k in QWEN25_MARKERS): return ChatMLTemplate(chatml_default_system(src))
  if _has_jinja(): return JinjaTemplate(src, special)
  if "<|start_header_id|>" in src and not any(k in src for k in ("date_string", "strftime", "tools")):
    return Llama3Template(special.get("bos_token", ""))
  if "<|im_start|>" in src and "<|im_end|>" in src: return ChatMLTemplate(chatml_default_system(src), qwen_tools=False)
  raise ImportError("this model's chat template needs jinja2 (pip install jinja2)")

class Tokenizer:
  def __init__(self, path: str):
    from tokenizers import Tokenizer as _HFTokenizer
    self.path = path
    self.tok = _HFTokenizer.from_file(os.path.join(path, "tokenizer.json"))
    cfg_fn = os.path.join(path, "tokenizer_config.json")
    self.config: dict = json.load(open(cfg_fn)) if os.path.exists(cfg_fn) else {}
    stm = os.path.join(path, "special_tokens_map.json")   # HF merges it over tokenizer_config's special tokens
    if os.path.exists(stm): self.config.update({k: v for k, v in json.load(open(stm)).items() if v})
    jt = os.path.join(path, "chat_template.jinja")       # HF prefers it over tokenizer_config's chat_template
    self.template_src = template_source(self.config, open(jt).read() if os.path.exists(jt) else None)   # tools.tool_format reads it
    self.template = make_template(self.config, self.template_src)
    def tid(k: str) -> int | None:
      v = self.config.get(k)
      v = v.get("content") if isinstance(v, dict) else v
      return self.tok.token_to_id(v) if v else None
    self.eos_token_id, self.bos_token_id, self.pad_token_id = tid("eos_token"), tid("bos_token"), tid("pad_token")
    gen = os.path.join(path, "generation_config.json")
    eos = json.load(open(gen)).get("eos_token_id") if os.path.exists(gen) else None
    self.stop_token_ids: list[int] = sorted({*([eos] if isinstance(eos, int) else (eos or [])), *([self.eos_token_id] if self.eos_token_id is not None else [])})
    self.special_ids = {i for i, t in self.tok.get_added_tokens_decoder().items() if t.special}

  @staticmethod
  def from_pretrained(name_or_path: str) -> "Tokenizer":
    from fusezap.serve.weights import resolve_model
    return Tokenizer(resolve_model(name_or_path))
  @property
  def vocab_size(self) -> int: return self.tok.get_vocab_size(with_added_tokens=True)
  def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
    return self.tok.encode(text, add_special_tokens=add_special_tokens).ids
  def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
    return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)
  def render_chat(self, messages: Sequence[dict], add_generation_prompt: bool = True, tools: Sequence[dict] | None = None) -> str:
    return self.template.render(messages, add_generation_prompt, tools)
  def apply_chat_template(self, messages: Sequence[dict], add_generation_prompt: bool = True, tools: Sequence[dict] | None = None) -> list[int]:
    # the template already contains the special tokens (like HF, which encodes with add_special_tokens=False)
    return self.encode(self.render_chat(messages, add_generation_prompt, tools), add_special_tokens=False)

class IncrementalDecoder:
  """Token ids in, text deltas out. Holds back text while the tail is an incomplete UTF-8 sequence (U+FFFD)."""
  def __init__(self, tok: Tokenizer, skip_special_tokens: bool = True):
    self.tok, self.skip, self.ids, self.sent = tok, skip_special_tokens, [], 0
  def push(self, tid: int) -> str:
    self.ids.append(int(tid))
    text = self.tok.decode(self.ids, skip_special_tokens=self.skip)
    if text.endswith("�"): return ""
    delta, self.sent = text[self.sent:], len(text)
    return delta
  def push_many(self, tids: list[int]) -> str:
    """push() for several tokens with one decode (the deltas concatenate to the same text)."""
    if not tids: return ""
    self.ids.extend(int(t) for t in tids)
    text = self.tok.decode(self.ids, skip_special_tokens=self.skip)
    if text.endswith("\ufffd"): return ""
    delta, self.sent = text[self.sent:], len(text)
    return delta
  def flush(self) -> str:
    text = self.tok.decode(self.ids, skip_special_tokens=self.skip)
    delta, self.sent = text[self.sent:], len(text)
    return delta
