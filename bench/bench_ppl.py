"""WikiText-2 perplexity of Qwen2.5-0.5B-Instruct for f16, int8, int4 and fp8 weights on METAL (P4-QUANT gate).

  uv run python bench/bench_ppl.py                      # f16, int8, int4 (asym), int4-sym, fp8; writes docs/quant.md
  uv run python bench/bench_ppl.py --tokens 8192 --no-write

Protocol (fixed, HF's strided perplexity recipe): the raw test split joined with "\\n\\n", tokenized with the model's
tokenizer (no BOS: Qwen has none), the first --tokens (32768) tokens, windows of --ctx (1024) tokens every --stride
(512); --tokens + 1 tokens are read so that there are exactly --tokens next-token predictions. The first window scores
all its ctx predictions; every later window scores only its last `stride`, each of which therefore sees at least
ctx - stride tokens of context. PPL = exp(mean NLL) over the 32768 scored tokens. Every window starts at position 0 (no KV cache carried between windows).

Exits nonzero unless int8 perplexity is within +1% of f16. The int4 and fp8 deltas are reported, not gated.
The dataset is read from the HF cache (Salesforce/wikitext, wikitext-2-raw-v1, parquet) with a small built-in parquet
reader, since neither `datasets` nor `pyarrow` is a dependency.
"""
from __future__ import annotations
import argparse, glob, math, os, struct, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DOC = os.path.join(os.path.dirname(__file__), "..", "docs", "quant.md")

# ---------------------------------------------------------------- minimal parquet reader (one string column)

def _varint(b: bytes, i: int) -> tuple[int, int]:
  r = s = 0
  while True:
    x = b[i]; i += 1
    r |= (x & 0x7F) << s; s += 7
    if x < 0x80: return r, i

def _zz(n: int) -> int: return (n >> 1) ^ -(n & 1)

def _thrift(b: bytes, i: int) -> tuple[dict, int]:
  """Thrift compact-protocol struct -> {field id: value}."""
  out, fid = {}, 0
  while True:
    h = b[i]; i += 1
    if h == 0: return out, i
    d, t = h >> 4, h & 0xF
    if d: fid += d
    else: v, i = _varint(b, i); fid = _zz(v)
    out[fid], i = _tval(b, i, t)

def _tval(b: bytes, i: int, t: int) -> tuple:
  if t in (1, 2): return t == 1, i
  if t == 3: return b[i], i + 1
  if t in (4, 5, 6): v, i = _varint(b, i); return _zz(v), i
  if t == 7: return struct.unpack("<d", b[i:i + 8])[0], i + 8
  if t == 8: n, i = _varint(b, i); return b[i:i + n], i + n
  if t in (9, 10):
    h = b[i]; i += 1
    n, et = h >> 4, h & 0xF
    if n == 15: n, i = _varint(b, i)
    out = []
    for _ in range(n):
      if et in (1, 2): out.append(b[i] == 1); i += 1
      else: v, i = _tval(b, i, et); out.append(v)
    return out, i
  if t == 11:
    n, i = _varint(b, i)
    if n == 0: return {}, i
    kv = b[i]; i += 1
    m = {}
    for _ in range(n):
      k, i = _tval(b, i, kv >> 4); v, i = _tval(b, i, kv & 0xF); m[k] = v
    return m, i
  if t == 12: return _thrift(b, i)
  raise ValueError(f"thrift type {t}")

def snappy_decompress(b: bytes) -> bytes:
  n, i = _varint(b, 0)
  out = bytearray()
  while i < len(b):
    tag = b[i]; i += 1
    k = tag & 3
    if k == 0:
      ln = tag >> 2
      if ln >= 60: nb = ln - 59; ln = int.from_bytes(b[i:i + nb], "little"); i += nb
      ln += 1
      out += b[i:i + ln]; i += ln
      continue
    if k == 1: ln, off = ((tag >> 2) & 7) + 4, ((tag >> 5) << 8) | b[i]; i += 1
    elif k == 2: ln, off = (tag >> 2) + 1, int.from_bytes(b[i:i + 2], "little"); i += 2
    else: ln, off = (tag >> 2) + 1, int.from_bytes(b[i:i + 4], "little"); i += 4
    s = len(out) - off
    for j in range(ln): out.append(out[s + j])   # overlapping copies repeat the pattern
  assert len(out) == n, (len(out), n)
  return bytes(out)

def _rle_hybrid(b: bytes, i: int, end: int, width: int, count: int) -> list[int]:
  out: list[int] = []
  nb = (width + 7) // 8
  while len(out) < count and i < end:
    h, i = _varint(b, i)
    if h & 1:
      g = h >> 1
      bits = int.from_bytes(b[i:i + g * width], "little"); i += g * width
      out += [(bits >> (j * width)) & ((1 << width) - 1) for j in range(g * 8)]
    else:
      v = int.from_bytes(b[i:i + nb], "little"); i += nb
      out += [v] * (h >> 1)
  return out[:count]

def _plain_strings(b: bytes, n: int) -> list[str]:
  out, i = [], 0
  for _ in range(n):
    ln = struct.unpack("<I", b[i:i + 4])[0]; i += 4
    out.append(b[i:i + ln].decode("utf-8")); i += ln
  return out

def read_parquet_strings(fn: str) -> list[str]:
  """Every row of the first (optional or required BYTE_ARRAY) column of a parquet file; nulls become ''."""
  data = open(fn, "rb").read()
  assert data[:4] == b"PAR1" and data[-4:] == b"PAR1", "not a parquet file"
  ml = struct.unpack("<I", data[-8:-4])[0]
  meta, _ = _thrift(data, len(data) - 8 - ml)
  optional = meta[2][1].get(3, 0) == 1           # schema[1].repetition_type: 0 required, 1 optional
  rows: list[str] = []
  for rg in meta[4]:
    cm = rg[1][0][3]
    codec, nvals = cm[4], cm[5]
    pos = cm[11] if cm.get(11) else cm[9]   # dictionary page first, if any
    dictionary: list[str] = []
    got = 0
    while got < nvals:
      ph, pos = _thrift(data, pos)
      raw = data[pos:pos + ph[3]]; pos += ph[3]
      ptype = ph[1]
      if ptype == 3:      # DATA_PAGE_V2: levels are never compressed
        h2 = ph[8]
        dl, rl = h2[5], h2[6]
        levels, body = raw[:rl + dl], raw[rl + dl:]
        if h2.get(7, True) and codec: body = _decompress(body, codec)
        n, enc = h2[1], h2[4]
        defs = _rle_hybrid(levels, rl, rl + dl, 1, n) if optional else [1] * n
      else:
        page = _decompress(raw, codec) if codec else raw
        if ptype == 2:    # DICTIONARY_PAGE
          dictionary = _plain_strings(page, ph[7][1]); continue
        n, enc, i = ph[5][1], ph[5][2], 0
        if optional:
          ln = struct.unpack("<I", page[:4])[0]
          defs = _rle_hybrid(page, 4, 4 + ln, 1, n); i = 4 + ln
        else: defs = [1] * n
        body = page[i:]
      nn = sum(defs)
      if enc in (2, 8): vals = [dictionary[k] for k in _rle_hybrid(body, 1, len(body), body[0], nn)]
      elif enc == 0: vals = _plain_strings(body, nn)
      else: raise NotImplementedError(f"parquet encoding {enc}")
      it = iter(vals)
      rows += [next(it) if d else "" for d in defs]
      got += n
  return rows

def _decompress(b: bytes, codec: int) -> bytes:
  if codec == 1: return snappy_decompress(b)
  if codec == 2:
    import gzip
    return gzip.decompress(b)
  raise NotImplementedError(f"parquet codec {codec}")

def wikitext2_test() -> str:
  pat = os.path.join(os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub")),
                     "datasets--Salesforce--wikitext", "snapshots", "*", "wikitext-2-raw-v1", "test-*.parquet")
  fns = sorted(glob.glob(pat))
  if not fns:
    from huggingface_hub import hf_hub_download
    fns = [hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset",
                           local_files_only=os.environ.get("FZ_HF_DOWNLOAD", "0") != "1")]
  return "\n\n".join(r for fn in fns for r in read_parquet_strings(fn))

# ---------------------------------------------------------------- perplexity

def windows(n: int, ctx: int, stride: int) -> list[tuple[int, int]]:
  """(begin, first scored index within the window) for full windows over n tokens."""
  return [(b, 0 if b == 0 else ctx - stride) for b in range(0, n - ctx + 1, stride)]

class NLLEval:
  """Per-token NLL of a fusezap.serve Llama over fixed-length windows: one fz.jit step (forward over all positions with
  no KV cache, then log-softmax and the target's log-prob on the device), so only (T,) floats come back per window."""
  def __init__(self, model, ctx: int):
    from fusezap.jit import Jit
    from fusezap.serve.models.llama import KVCache
    class NoCache(KVCache):
      def state(self): return []
      def bind(self, state, inputs=None): return self
      def load(self, state): pass
      def update(self, layer, k, v, start, T): return k, v
    self.m, self.ctx, self.cache = model, ctx, NoCache()
    self.jit = Jit(self._step)
  def _step(self, x, tgt):
    from fusezap.tensor import Tensor
    lg = self.m.forward(x, self.cache, 0, self.ctx, last_only=False).float()[0]       # (T, V)
    V = lg.shape[-1]
    pick = (Tensor.arange(V, device=self.m.device).reshape(1, V) == tgt.reshape(-1, 1)).where(lg, 0.0).sum(-1)
    return lg.logsumexp(-1) - pick
  def __call__(self, ids: np.ndarray) -> np.ndarray:
    """ids: (ctx + 1,) tokens -> (ctx,) NLL of ids[1:] given the prefix."""
    from fusezap.tensor import Tensor
    from fusezap.dtype import dtypes
    assert len(ids) == self.ctx + 1
    x = Tensor(self.m.embed(np.asarray(ids[:-1])[None]), device=self.m.device, dtype=self.m.dtype)
    t = Tensor(np.asarray(ids[1:], np.int32), device=self.m.device, dtype=dtypes.int32)
    with self.m._opt_env(prefill=True): out = self.jit(x, t)
    return out.numpy()

def perplexity(model, ids: np.ndarray, ctx: int, stride: int, log=None) -> tuple[float, int, float]:
  """(ppl, scored tokens, seconds) with the module's protocol. Window b covers ids[b:b+ctx]; its predictions are for
  ids[b+1:b+ctx+1], so the input window needs ctx+1 tokens; the last one is only a target."""
  ev = NLLEval(model, ctx)
  nll, cnt, t0 = 0.0, 0, time.perf_counter()
  ws = windows(len(ids) - 1, ctx, stride)
  for k, (b, first) in enumerate(ws):
    v = ev(ids[b:b + ctx + 1]).astype(np.float64)[first if b else 0:]
    nll, cnt = nll + float(v.sum()), cnt + len(v)
    if log and (k % 16 == 0 or k == len(ws) - 1): log(f"  window {k + 1}/{len(ws)}  running ppl {math.exp(nll / cnt):.4f}")
  return math.exp(nll / cnt), cnt, time.perf_counter() - t0

# ---------------------------------------------------------------- main

def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--tokens", type=int, default=32768)
  ap.add_argument("--ctx", type=int, default=1024)
  ap.add_argument("--stride", type=int, default=512)
  ap.add_argument("--modes", default="f16,int8,int4,int4-sym,fp8")
  ap.add_argument("--device", default=os.environ.get("FZ_DEVICE", "METAL"))
  ap.add_argument("--no-write", action="store_true")
  a = ap.parse_args()
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  from fusezap.serve.tokenizer import Tokenizer
  from fusezap.serve.weights import resolve_model
  from fusezap.serve.quant import quantize_model, weight_bytes
  path = resolve_model(MODEL)
  tok = Tokenizer.from_pretrained(path)
  text = wikitext2_test()
  ids = np.asarray(tok.encode(text), np.int64)
  print(f"wikitext-2-raw-v1 test: {len(text)} chars, {len(ids)} tokens; using the first {a.tokens + 1}")
  ids = ids[:a.tokens + 1]
  rows, base = [], None
  for mode in a.modes.split(","):
    t0 = time.perf_counter()
    m = load_model(MODEL, dtype=dtypes.float16, device=a.device, max_len=a.ctx, prefill_chunk=16)
    if mode != "f16": quantize_model(m, mode)
    tq = time.perf_counter() - t0
    print(f"{mode}: loaded + quantized in {tq:.1f}s")
    ppl, n, dt = perplexity(m, ids, a.ctx, a.stride, log=print)
    wb = weight_bytes(m)
    base = ppl if mode == "f16" else base
    rows.append((mode, ppl, n, dt, wb))
    print(f"{mode}: ppl {ppl:.4f} over {n} tokens in {dt:.1f}s ({n / dt:.0f} scored tok/s); linear {wb['linear'] / 2**20:.1f} MiB, device {wb['device_total'] / 2**20:.1f} MiB")
    del m
  table = ["| weights | perplexity | Δ vs f16 | linear weights | device weights (total) | eval time |",
           "|---|---|---|---|---|---|"]
  for mode, ppl, n, dt, wb in rows:
    d = "—" if base is None or mode == "f16" else f"{(ppl / base - 1) * 100:+.2f}%"
    name = {"f16": "float16", "int8": "int8 (per-channel)", "int4": "int4 asym (g128)", "int4-sym": "int4 sym (g128)", "fp8": "fp8 e4m3 (per-tensor)"}.get(mode, mode)
    table.append(f"| {name} | {ppl:.4f} | {d} | {wb['linear'] / 2**20:.1f} MiB | {wb['device_total'] / 2**20:.1f} MiB | {dt:.0f} s |")
  print("\n".join(table))
  ok = True
  i8 = next((r for r in rows if r[0] == "int8"), None)
  if base is not None and i8 is not None:
    ok = i8[1] <= base * 1.01
    print(f"gate: int8 ppl {i8[1]:.4f} vs f16 {base:.4f} ({(i8[1] / base - 1) * 100:+.3f}%, limit +1%): {'PASS' if ok else 'FAIL'}")
  if not a.no_write and os.path.exists(DOC):
    src = open(DOC).read()
    s, e = src.find("<!-- ppl-table -->"), src.find("<!-- /ppl-table -->")
    if s >= 0 and e > s:
      proto = (f"Protocol: {MODEL}, WikiText-2 raw test, first {a.tokens} tokens (+1 target), ctx {a.ctx}, stride {a.stride}, "
               f"{rows[0][2]} scored tokens, {a.device}, activations float16. Generated by `bench/bench_ppl.py` on {time.strftime('%Y-%m-%d')}.")
      open(DOC, "w").write(src[:s] + "<!-- ppl-table -->\n" + proto + "\n\n" + "\n".join(table) + "\n" + src[e:])
      print(f"wrote {os.path.normpath(DOC)}")
  return 0 if ok else 1

if __name__ == "__main__":
  sys.exit(main())
