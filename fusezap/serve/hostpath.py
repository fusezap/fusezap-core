"""I69 follow-up (served first token): switches for the host side of `fusezap serve` between the client and the engine step.
The 14 Nov inf2 run served Qwen2.5-7B TP=2 short c1 at TTFT p50 52.15 ms against 48.84 in process and a 45.24 ms prefill
execute; the I78 request traces put ~1.3 ms between the client's send and the handler (uvicorn's pure-Python h11 parser,
starlette, the JSON body) and ~1.2 ms between the first output reaching the handler and the client (the response start,
the first SSE chunk, the client's parse). None of these changes a token: they move or shorten host work only.

  FZ_SERVE_HTTP        auto (default: uvicorn's own choice, httptools + uvloop when installed, else h11 + asyncio, which is
                       what `uv sync --extra serve` gives today) | fast (httptools + uvloop: an error if either is missing)
                       | h11 (h11 + asyncio whatever is installed: the stack every inf2 TTFT so far was measured on)
  FZ_SERVE_ACCESS_LOG  1 (default: uvicorn's per-request access line) | 0 (none: one log call less as the response starts)
  FZ_SERVE_EARLY_HEADERS  0 (default: a stream's status and headers wait for its first output, so an engine error before
                       it is still a proper HTTP error) | 1 (the stream's response starts at once, before the prefill; an
                       engine error before the first token becomes the stream's `error` event, as one after it does now)
  FZ_SERVE_GC          0 (default) | freeze (after warm-up: gc.collect() then gc.freeze() in the server, the engine process and
                       every tensor-parallel worker, so the start-up objects (weights' Python objects, captured graphs, NEFF
                       tables) leave the collector's generations and a collection during a step only walks what serving made)
"""
from __future__ import annotations
import gc, os
from typing import Any

HTTP_MODES = ("auto", "fast", "h11")

def http_mode() -> str:
  v = (os.environ.get("FZ_SERVE_HTTP", "") or "auto").strip().lower()
  if v not in HTTP_MODES: raise ValueError(f"FZ_SERVE_HTTP={v!r}: one of {', '.join(HTTP_MODES)}")
  return v

def access_log() -> bool: return os.environ.get("FZ_SERVE_ACCESS_LOG", "1").strip() not in ("0", "off", "false")

def early_headers() -> bool: return os.environ.get("FZ_SERVE_EARLY_HEADERS", "0").strip() in ("1", "on", "true")

def gc_mode() -> str:
  v = (os.environ.get("FZ_SERVE_GC", "") or "0").strip().lower()
  if v not in ("0", "freeze"): raise ValueError(f"FZ_SERVE_GC={v!r}: 0 or freeze")
  return v

def uvicorn_opts() -> dict[str, Any]:
  """Keyword arguments for uvicorn.Config / uvicorn.run from the switches above (only those that differ from uvicorn's
  defaults, so FZ_SERVE_HTTP=auto with the access log on passes nothing)."""
  m, out = http_mode(), {}
  if m == "fast":
    import importlib.util as U
    missing = [p for p in ("httptools", "uvloop") if U.find_spec(p) is None]
    if missing: raise RuntimeError(f"FZ_SERVE_HTTP=fast needs {' and '.join(missing)} (pip install httptools uvloop)")
    out.update(http="httptools", loop="uvloop")
  elif m == "h11": out.update(http="h11", loop="asyncio")
  if not access_log(): out["access_log"] = False
  return out

def describe(opts: dict[str, Any] | None = None) -> str:
  """One line for the server log: the HTTP stack uvicorn will run and the other switches."""
  import importlib.util as U
  o = uvicorn_opts() if opts is None else opts
  have = lambda p: U.find_spec(p) is not None
  http = o.get("http") or ("httptools" if have("httptools") else "h11")
  loop = o.get("loop") or ("uvloop" if have("uvloop") else "asyncio")
  return (f"http {http} + {loop} (FZ_SERVE_HTTP={http_mode()}), access log {'on' if o.get('access_log', True) else 'off'}, "
          f"early stream headers {'on' if early_headers() else 'off'}, gc {gc_mode()}")

def gc_settle(g: Any = None) -> bool:
  """FZ_SERVE_GC=freeze: collect, then freeze what is left (this process, and with a tensor-parallel group `g` on rank 0
  every worker too, through op "call"). Returns whether it froze."""
  if gc_mode() != "freeze": return False
  if g is not None and getattr(g, "world", 1) > 1 and getattr(g, "rank", 0) == 0:
    with g.spmd("call", "fusezap.serve.hostpath:_freeze"): _freeze(g)
  else: _freeze(g)
  return True

def _freeze(g: Any = None) -> int:
  """gc.collect() then gc.freeze(); the objects now frozen (op "call" target: fn(group))."""
  gc.collect()
  gc.freeze()
  return gc.get_freeze_count()
