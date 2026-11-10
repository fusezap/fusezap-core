"""Grammar-constrained decoding for `fusezap serve` (I62): response_format json_schema / json_object and strict tool calls.

  c = json_schema(schema, tok) | json_object(tok) | call_constraint(call_spec, tok, prefix_ids)   # call_spec: CallSpec / tools.CallConstraint
  SamplingParams(..., constraint=c)          # c is plain data (grammar text, token ids, a path): it pickles to the engine process
  # in the engine (LLMEngine.add_request): s.cons = c.matcher(vocab, stop_ids); per token: row = s.cons.mask(logits_row),
  # tok = sampler(row); s.cons.advance(tok); s.cons.done -> the request finishes with "stop"

Library: llguidance (MIT; Rust core, no Python dependencies, abi3 wheels for macOS arm64 / x86_64 and Linux x86_64 /
aarch64, ~3 MB; vLLM's "guidance" structured-output backend). It is an OPTIONAL dependency, in the `serve` extra (`pip
install 'fusezap[serve]'`, or `pip install llguidance`); without it these requests get HTTP 400 and nothing else changes.
Picked over xgrammar (Apache-2.0, vLLM's default, but its wheel hard-requires torch, transformers and triton: gigabytes
in Core's runtime path) and outlines-core (regex FSM: JSON Schema coverage is narrower, the index is built per schema
over the whole vocabulary, seconds for large schemas); llguidance computes a 152k-token mask in ~10-100 us on the CPU
with no per-schema precompute (an Earley parser over a lexer with a token trie).

Grammars: JSON Schema via llguidance's own compiler (Draft 2020-12 subset: types, enum/const, properties / required /
additionalProperties, items / prefixItems, min/max (exclusive), lengths, pattern, anyOf / oneOf / allOf, $ref / $defs;
an unsupported keyword such as uniqueItems is a 400, not silently ignored), compact separators (", " and ": ", no other
whitespace: the output cannot run away in whitespace and reads like the chat templates' own tool-call JSON). Tool calls
are a Lark grammar (call_grammar) built from a format-independent description of the call (CallSpec: prefix, args_key,
close, call_open, separator, the tools' schemas, strict, parallel: the fields of tools.CallConstraint), whose argument
rules are `%json {schema}`; added tokens in that text (`<tool_call>`, `[TOOL_CALLS]`, ...) become token references. So
any format whose call is "prefix NAME ...args_key: ARGS close" (Hermes, Mistral, Llama 3 JSON) needs no grammar code.

Token prefix: the API appends a forced prefix to the prompt for named / required tool_choice (tools.ToolSpec.forced_prefix);
the grammar describes prefix + call and the matcher consumes the prefix's token ids first.
"""
from __future__ import annotations
import functools, json, os
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

INSTALL = "pip install llguidance (or pip install 'fusezap[serve]')"
JSON_OPTS = {"whitespace_flexible": False, "item_separator": ", ", "key_separator": ": "}

class ConstraintError(ValueError):
  """A request whose output cannot be constrained (no llguidance, a bad schema): the API answers 400."""

def available() -> bool:
  try: import llguidance  # noqa: F401
  except ImportError: return False
  return True

def _llg():
  try: import llguidance
  except ImportError: raise ConstraintError(f"grammar-constrained decoding needs the optional llguidance package, which is not installed on this server ({INSTALL})")
  return llguidance

@dataclass(frozen=True)
class Constraint:
  """What a request's output must match. grammar: an llguidance grammar (JSON text, from grammar_from_json_schema or
  grammar_from_lark); prefix: token ids already at the end of the prompt that the grammar starts with; tokenizer: the
  tokenizer.json the grammar's tokens refer to (the engine builds its llguidance tokenizer from it once)."""
  grammar: str
  tokenizer: str
  prefix: tuple[int, ...] = ()
  kind: str = "json"
  def matcher(self, n_vocab: int, eos: Sequence[int]) -> "Matcher": return Matcher(self, n_vocab, eos)

def tokenizer_file(tok: Any) -> str:
  p = getattr(tok, "path", None)
  if p is None: raise ConstraintError("grammar-constrained decoding needs the model's tokenizer.json")
  return os.path.join(p, "tokenizer.json")

def _check(grammar: str) -> str:
  err = _llg().LLMatcher.validate_grammar(grammar)
  if err: raise ConstraintError(f"cannot constrain the output to this schema: {err}")
  return grammar

def json_schema(schema: dict, tok: Any, prefix: Sequence[int] = ()) -> Constraint:
  if not isinstance(schema, dict): raise ConstraintError("the JSON schema must be an object")
  g = _check(_llg().LLMatcher.grammar_from_json_schema(schema, overrides=JSON_OPTS))
  return Constraint(g, tokenizer_file(tok), tuple(prefix), "json_schema")

def json_object(tok: Any) -> Constraint:
  c = json_schema({"type": "object"}, tok)
  return Constraint(c.grammar, c.tokenizer, (), "json_object")

# ---------------------------------------------------------------- tool calls (the hook for tool-call formats)

@dataclass(frozen=True)
class CallSpec:
  """What a named / required tool call must look like, format-independent: the same fields as I61's
  tools.CallConstraint (tools.call_constraint(spec)), which call_constraint() accepts as is (duck-typed). The prompt
  already ends with `prefix`; the generated text must then be
    named:     JSON(args valid against tools[name]) + close
    required:  NAME + '", "' + args_key + '": ' + JSON(args valid against tools[NAME]) + close     (NAME in tools)
  and with `parallel` (required only) further calls: separator + call_open + NAME + ... + close. The model's stop token
  comes last (allowed once a call is complete; the engine also finishes when nothing else can follow)."""
  fmt: str
  mode: str                        # "required" | "named"
  prefix: str
  args_key: str                    # "arguments" (hermes, mistral) | "parameters" (llama3_json)
  close: str                       # "}\n</tool_call>" (hermes), "}]" (mistral), "}" (llama3_json)
  tools: dict                      # function name -> its parameters JSON Schema ({} or None: any object)
  name: str | None = None
  strict: frozenset = frozenset()  # the functions sent with "strict": true
  call_open: str = ""              # a further call's text before NAME
  separator: str = ""              # between calls
  parallel: bool = False

def lit(s: str) -> str:
  """A Lark string literal matching exactly s."""
  return json.dumps(s, ensure_ascii=False)

def args_rule(schema: dict | None) -> str:
  """A Lark rule body matching a JSON value valid against schema (compact separators, as JSON_OPTS)."""
  s = dict(schema) if isinstance(schema, dict) and schema else {"type": "object"}
  s["x-guidance"] = {**JSON_OPTS, **(s.get("x-guidance") or {})}
  return "%json " + json.dumps(s, ensure_ascii=False)

def text_rule(text: str, tok: Any) -> str:
  """A Lark sequence matching text, with each added token of tok's vocabulary that occurs in it (`<tool_call>`,
  `[TOOL_CALLS]`, `<|python_tag|>`, ...) as a token reference `<[id]>`: llguidance never matches an added token by its
  text. '' -> an empty sequence."""
  t = getattr(tok, "tok", None)
  added = {v.content: k for k, v in t.get_added_tokens_decoder().items() if v.content} if t is not None else {}
  out, buf, i = [], "", 0
  while i < len(text):
    hit = max((a for a in added if text.startswith(a, i)), key=len, default=None)
    if hit is None: buf += text[i]; i += 1; continue
    if buf: out.append(lit(buf)); buf = ""
    out.append(f"<[{added[hit]}]>"); i += len(hit)
  if buf: out.append(lit(buf))
  return " ".join(out) or '""'

def call_grammar(cs: Any, tok: Any) -> str | None:
  """The Lark grammar of prefix + the call text (cs: a CallSpec or tools.CallConstraint), or None when nothing is to be
  constrained: tool_choice named whose function is not strict, required with no strict function. A strict function's
  arguments match its parameter schema; a non-strict one's (required, next to strict ones) are any JSON object."""
  if cs is None or cs.mode not in ("named", "required"): return None
  strict = set(cs.strict or ())
  names = [cs.name] if cs.mode == "named" else list(cs.tools)
  if not strict & set(names) or not names: return None
  args = lambda n: args_rule(cs.tools.get(n) if n in strict else None)
  if cs.mode == "named": return f"start: {text_rule(cs.prefix, tok)} a0 {text_rule(cs.close, tok)}\na0: {args(cs.name)}\n"
  head = lambda n: lit(f'{n}", "{cs.args_key}": ')
  rules = ["fn: " + " | ".join(f"{head(n)} a{i}" for i, n in enumerate(names)), *(f"a{i}: {args(n)}" for i, n in enumerate(names))]
  more = f" ({text_rule(cs.separator + cs.call_open, tok)} fn {text_rule(cs.close, tok)})*" if cs.parallel and cs.call_open else ""
  return "\n".join([f"start: {text_rule(cs.prefix, tok)} fn {text_rule(cs.close, tok)}{more}", *rules]) + "\n"

def call_constraint(cs: Any, tok: Any, prefix_ids: Sequence[int]) -> Constraint | None:
  """The Constraint of a named / required tool call (cs: CallSpec or tools.CallConstraint; prefix_ids: cs.prefix's
  tokens as appended to the prompt), or None (unconstrained: see call_grammar)."""
  g = call_grammar(cs, tok)
  if g is None: return None
  g = _check(_llg().LLMatcher.grammar_from_lark(g))
  return Constraint(g, tokenizer_file(tok), tuple(int(t) for t in prefix_ids), "tool_call")

# ---------------------------------------------------------------- engine side

@functools.lru_cache(maxsize=4)
def lltokenizer(path: str, n_vocab: int, eos: tuple[int, ...]) -> Any:
  """llguidance's tokenizer for tokenizer.json at path (~0.5 s for a 152k vocabulary: built once per engine, and at
  warm-up when llguidance is installed). n_vocab: the logits' width (rows past the tokenizer's vocabulary are never
  allowed); eos: the engine's stop ids (allowed once the grammar accepts)."""
  with open(path) as f: return _llg().LLTokenizer(f.read(), n_vocab=n_vocab, eos_token=list(eos))

def prewarm(path: str | None, n_vocab: int, eos: Sequence[int]) -> bool:
  """Build the llguidance tokenizer ahead of the first constrained request (False: llguidance missing or no path)."""
  if not path or not available() or not os.path.exists(path): return False
  lltokenizer(path, int(n_vocab), tuple(sorted(int(e) for e in eos)))
  return True

class Matcher:
  """One request's grammar state. mask(row) -> the logits row with every token the grammar does not allow next at -inf
  (a new array); advance(tok) after the token is chosen; done once the grammar is complete (no token but EOS can follow)."""
  __slots__ = ("m", "bits", "n", "kind", "_ok")
  def __init__(self, c: Constraint, n_vocab: int, eos: Sequence[int]):
    llg = _llg()
    self.m = llg.LLMatcher(lltokenizer(c.tokenizer, int(n_vocab), tuple(sorted(int(e) for e in eos))), c.grammar, log_level=0)
    if self.m.is_error(): raise ConstraintError(f"cannot constrain the output: {self.m.get_error()}")
    if c.prefix and not self.m.consume_tokens(list(c.prefix)): raise ConstraintError(f"the forced prompt prefix does not match the grammar: {self.m.get_error()}")
    self.n, self.kind = int(n_vocab), c.kind
    self.bits = np.full((self.n + 31) // 32, -1, np.int32)
    self._ok: np.ndarray | None = None
  @property
  def done(self) -> bool: return self.m.is_stopped()
  def allowed(self) -> np.ndarray:
    """(n_vocab,) bool: the tokens the grammar allows next (cached until advance)."""
    if self._ok is None:
      self.m.unsafe_compute_mask_ptr(self.bits.ctypes.data, self.bits.nbytes)
      if self.m.is_error(): raise RuntimeError(f"constrained decoding: {self.m.get_error()}")
      self._ok = np.unpackbits(self.bits.view(np.uint8), bitorder="little")[:self.n].view(bool)
    return self._ok
  def mask(self, row: np.ndarray) -> np.ndarray:
    ok = self.allowed()
    row = np.asarray(row, np.float32)
    if row.shape[-1] != self.n: ok = np.pad(ok, (0, max(0, row.shape[-1] - self.n)))[:row.shape[-1]]
    if not ok.any(): raise RuntimeError("constrained decoding: the grammar allows no token")
    return np.where(ok, row, np.float32(-np.inf))
  def advance(self, tok: int) -> None:
    self._ok = None
    if not self.m.consume_token(int(tok)): raise RuntimeError(f"constrained decoding: token {tok} rejected: {self.m.get_error()}")
