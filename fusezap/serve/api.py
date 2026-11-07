"""OpenAI-compatible HTTP API over LLMEngine (SPEC §11.3), via starlette + uvicorn.

  app = create_app(engine, tokenizer, model_name)         # starlette app; uvicorn.run(app, ...)
  GET  /health   GET /v1/models   POST /v1/chat/completions   POST /v1/completions   (stream and non-stream, SSE ends with `data: [DONE]`)
  GET  /metrics  (Prometheus text format, vLLM's metric names: fusezap/serve/metrics.py; open without the API key, as vLLM)

`AsyncEngine` owns the LLMEngine on one worker thread: HTTP handlers submit requests through a queue, the thread drains it
between steps (so concurrent requests join the running batch: continuous batching) and posts each step's RequestOutput
back to the asyncio queue of its request. `ProcEngine` (the `fusezap serve` default on CUDA and NEURON) runs the same loop
in a spawned engine process instead, so the HTTP work does not share a GIL with the engine's steps. Handlers detokenize
incrementally, apply stop strings (aborting the engine request when one matches) and format chunks. Supported: max_tokens /
max_completion_tokens, temperature (default 1.0, as OpenAI), top_p, top_k (0 or -1: off), seed, stop, n, repetition_penalty,
stream_options.include_usage, logprobs (completions: `logprobs` k and `echo`, also for token-id prompts; chat: `logprobs` /
`top_logprobs`; vLLM's `return_tokens_as_token_ids`; computed by a teacher-forced scoring pass, fusezap/serve/logprobs.py),
chat `tools` / `tool_choice` (none, auto, required, a named function; Hermes, Mistral and Llama 3.1+ JSON formats parsed into
`tool_calls`, fusezap/serve/tools.py), `response_format` json_object / json_schema and strict tools with a named / required
tool_choice (grammar-constrained decoding, fusezap/serve/constrain.py: needs the optional llguidance package, else 400).
A streamed request with logprobs, echo or active Hermes / Mistral tools gets its chunks after generation finishes (the output
must be complete to be scored or parsed); with Llama 3.1+ tools the stream is live until the output starts like a call
(then that choice is held to its end and sent as tool_calls deltas: id, type and name first, then the arguments).
`parallel_tool_calls: false` keeps at most one tool call per choice. Fields that would change the output and are not
implemented (presence / frequency penalties, logit_bias, best_of, suffix and the others in UNSUPPORTED) are a 400
`unsupported_parameter` unless they carry their neutral value (0, null, {}, absent), so a request never gets a silently
different result; informational fields (user, metadata, store, service_tier, ...) are accepted and have no effect.
"""
from __future__ import annotations
import asyncio, contextlib, json, os, queue, sys, threading, time, traceback, uuid
from typing import Any, AsyncIterator, Callable
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from fusezap.serve.engine import LLMEngine, RequestOutput, SamplingParams
from fusezap.serve.scheduler import SeqStatus
from fusezap.serve.tokenizer import IncrementalDecoder, Tokenizer
from fusezap.serve import logprobs as LP, tools as TL, metrics as MX, serveprof as SP

class ReqTrace:
  """FZ_REQ_TRACE=<prefix>: one JSON line per request with wall-clock (time.time) timestamps, for the turnaround between
  closed-loop waves (speed run 4 round 12). The API process writes <prefix>.api.jsonl: recv (handler entered), parsed
  (body parsed and checked), submit (sent to the engine), first (first output event), last (final output event),
  done ([DONE] written to the stream). The engine process writes <prefix>.engine.jsonl: add (the add command received),
  sched (the step that first scheduled it began after this), fin (its last token emitted), out (that step's outputs
  sent). Both carry the API's request number (rid) of the engine pipe; off (no file) when FZ_REQ_TRACE is unset."""
  def __init__(self, suffix: str):
    import os
    p = os.environ.get("FZ_REQ_TRACE")
    self.f = open(f"{p}.{suffix}.jsonl", "a", buffering=1) if p else None
  def write(self, rec: dict) -> None:
    if self.f is not None: self.f.write(json.dumps(rec) + "\n")

class APIError(Exception):
  def __init__(self, message: str, status: int = 400, type: str = "invalid_request_error", param: str | None = None, code: str | None = None):
    super().__init__(message)
    self.message, self.status, self.type, self.param, self.code = message, status, type, param, code
  def response(self) -> JSONResponse:
    return JSONResponse({"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}, self.status)

def _err_kind(e: BaseException) -> Any:
  """How the engine process reports an exception over its pipe: True (a ValueError: the request is bad), "oom" (a
  MemoryError: the device is out of memory for now, e.g. NEURON's HBM budget, N8), else False."""
  return "oom" if isinstance(e, MemoryError) else isinstance(e, ValueError)

def _release_memory(eng: Any) -> None:
  """After a step ran out of device memory (N8): drop the scorer's jitted steps and hand the allocator's cached blocks back,
  so the next requests find the memory a scoring pass held."""
  with contextlib.suppress(Exception):
    from fusezap.serve import logprobs as LP_
    if getattr(eng, "_fz_scorer", None) is not None: LP_.scorer(eng).release()
    else:
      from fusezap.device import Device
      if hasattr(a := getattr(Device[eng.model.device], "allocator", None), "free_cache"): a.free_cache()

def _api_error(e: BaseException, prefix: str = "") -> APIError:
  """The HTTP error for an engine-side exception: 400 for a ValueError, 503 (retryable, engine_out_of_memory) for a
  MemoryError (N8, I71: the device had no memory for this request's kernels; the engine stays up), else 500."""
  if isinstance(e, MemoryError): return APIError(f"{prefix}{e}", 503, "server_error", code="engine_out_of_memory")
  bad = isinstance(e, ValueError)
  return APIError(f"{prefix}{e}", 400 if bad else 500, "invalid_request_error" if bad else "server_error")

# ---------------------------------------------------------------- engine on a worker thread

class _Handle:
  def __init__(self, loop: asyncio.AbstractEventLoop):
    self.loop, self.q, self.rid = loop, asyncio.Queue(), None
  def post(self, item: Any) -> None:
    with contextlib.suppress(RuntimeError): self.loop.call_soon_threadsafe(self.q.put_nowait, item)

def _deliver(items: list[tuple[_Handle, Any]]) -> None:
  for h, o in items: h.q.put_nowait(o)

class AsyncEngine:
  """Runs LLMEngine.step() on a worker thread; all engine calls happen on that thread."""
  def __init__(self, engine: LLMEngine):
    self.eng, self.cmds, self.handles = engine, queue.Queue(), {}
    self.thread: threading.Thread | None = None
    self.stats = MX.EngineStats(engine)   # updated on the engine thread only; readers take stats.snap
    self.prof = SP.from_env()             # FZ_SERVE_PROFILE: one line per step (fusezap/serve/serveprof.py)
    self.gate = engine.admit_gate() if hasattr(engine, "admit_gate") else None   # I76: gather a burst of short prompts
  @property
  def snap(self) -> MX.Snap: return self.stats.snap
  def start(self) -> None:
    self.thread = threading.Thread(target=self._run, name="fusezap-engine", daemon=True)
    self.thread.start()
  def stop(self) -> None:
    if self.thread is not None: self.cmds.put(None); self.thread.join(timeout=30); self.thread = None
  def submit(self, ids: list[int], params: SamplingParams) -> _Handle:
    h = _Handle(asyncio.get_running_loop())
    self.cmds.put(("add", h, ids, params))
    return h
  def abort(self, h: _Handle) -> None: self.cmds.put(("abort", h))
  async def score(self, ids: list[int], start: int, k: int) -> dict:
    """LP.Scorer.score on the engine thread, between steps."""
    h = _Handle(asyncio.get_running_loop())
    self.cmds.put(("score", h, ids, start, k))
    r = await h.q.get()
    if isinstance(r, BaseException): raise r
    return r

  def _cmd(self, cmd: tuple) -> None:
    if cmd[0] == "score":
      _, h, ids, start, k = cmd
      try: h.post(LP.scorer(self.eng).score(ids, start, k))
      except Exception as e: h.post(e)
    elif cmd[0] == "add":
      _, h, ids, params = cmd
      try: h.rid = self.eng.add_request(ids, params)
      except Exception as e: h.post(e); return
      self.handles[h.rid] = h
      self.stats.prompt += len(ids)
      if self.gate is not None: self.gate.added()
      if self.prof is not None: self.prof.adds += 1
    elif (h := cmd[1]).rid in self.handles:
      self.eng.abort(h.rid); self.eng.release(h.rid); del self.handles[h.rid]
  def _step(self) -> None:
    pf, rec = self.prof, None
    if pf is not None: wall = time.time(); pf.begin(self.eng)
    try:
      outs = self.eng.step()
      if pf is not None: rec = pf.end(self.eng, wall)
    except Exception as e:      # fail every in-flight request; the engine stays usable for new ones
      if isinstance(e, MemoryError): _release_memory(self.eng)   # N8: drop the scorer's captures, hand cached blocks back
      for rid, h in list(self.handles.items()):
        h.post(e)
        with contextlib.suppress(Exception): self.eng.abort(rid); self.eng.release(rid)
      self.handles.clear()
      self.stats.update()
      return
    by_loop: dict[asyncio.AbstractEventLoop, list[tuple[_Handle, RequestOutput]]] = {}
    if self.gate is not None: self.gate.finished(sum(o.finished for o in outs))
    for o in outs:
      if (h := self.handles.get(o.request_id)) is None: continue
      by_loop.setdefault(h.loop, []).append((h, o))
      self.stats.gen += len(o.new_token_ids)
      if o.finished: del self.handles[o.request_id]; self.eng.release(o.request_id)
    self.stats.update()
    if rec is not None:   # FZ_SERVE_PROFILE: the hand-off time, for the handlers' lag
      tp = time.perf_counter()
      for items in by_loop.values():
        for _, o in items: o.t_post = tp
    for loop, items in by_loop.items():   # one wake-up of each event loop per step, not one per request
      with contextlib.suppress(RuntimeError): loop.call_soon_threadsafe(_deliver, items)
    if rec is not None: self.prof.done(rec, len(outs))
  def _run(self) -> None:
    while True:
      try:
        if self.prof is not None and not self.eng.has_unfinished(): self.prof.idle()   # waiting for requests: not a gap
        cmd = self.cmds.get(block=not self.eng.has_unfinished())
        while True:
          if cmd is None: return
          self._cmd(cmd)
          cmd = self.cmds.get_nowait()
      except queue.Empty: pass
      self.stats.update()   # adds and aborts show at once, also when no step follows (the last request aborted)
      if self.gate is not None and self.eng.has_unfinished() and (w := self.gate.hold()) > 0:   # I76: wait for the burst's next prompt
        try: cmd = self.cmds.get(timeout=w)
        except queue.Empty: pass
        else:
          if cmd is None: return
          self._cmd(cmd)
          continue             # drain the rest, then ask again
      if self.eng.has_unfinished(): self._step()

# ---------------------------------------------------------------- engine in its own process

class ProcEngine:
  """AsyncEngine's interface with the LLMEngine in a spawned child process (`fusezap serve` on CUDA, SPEED-R3; NEURON, N7):
  the HTTP side (parsing, detokenizing, SSE) and the engine's step loop then hold separate GILs. In one process the event loop's
  per-token work (a few ms per step at 32 streams) took the GIL from the engine thread at every CUDA call it made, which
  stretched each decode step. The child builds the engine from the `serve` arguments (cli.build_engine), warms it up and
  runs the same loop as AsyncEngine; requests, aborts and each step's new tokens cross one pipe. The engine, scheduler
  and kernels are the same objects as in-process, so the tokens are the same.
  If the child dies while serving (OOM kill, segfault, fatal CUDA error), `dead` is set: requests in flight fail, new ones
  get 503, /health returns 503, and `on_dead` runs (`fusezap serve` stops the server and exits 1, so a supervisor restarts it)."""
  def __init__(self, args: Any, on_dead: Callable[[], None] | None = None, target: Callable | None = None):
    import multiprocessing as mp
    self.args, self.ctx = args, mp.get_context("spawn")
    self.target = target or _engine_main   # the child's entry (bench/serveq_sim.py wraps it to fake the forwards' latency)
    self.handles: dict[int, _Handle] = {}
    self.ids, self.lock = 0, threading.Lock()
    self.proc, self.conn, self.reader, self.info = None, None, None, {}
    self.on_dead, self.dead, self.stopping = on_dead, None, False
    self.snap = MX.Snap()   # the engine process's latest EngineStats snapshot (set by the reader thread)
    self.prof_on = SP.path_from_env() is not None   # FZ_SERVE_PROFILE: stamp each output's hand-off
  def start(self) -> None:
    """Spawn the engine process and wait until it has loaded and warmed up (raises if it failed)."""
    if self.proc is not None: return
    self.conn, child = self.ctx.Pipe()
    self.proc = self.ctx.Process(target=self.target, args=(child, self.args), name="fusezap-engine", daemon=True)
    self.proc.start()
    child.close()
    try: msg = self.conn.recv()
    except EOFError: msg = ("dead", f"engine process exited with {self.proc.join(5) or self.proc.exitcode}")
    if msg[0] != "ready": self.stop(); raise RuntimeError(f"engine process failed:\n{msg[1]}")
    self.info = msg[1]
    self.reader = threading.Thread(target=self._read, name="fusezap-engine-reader", daemon=True)
    self.reader.start()
  def stop(self) -> None:
    if self.proc is None: return
    self.stopping = True
    with contextlib.suppress(OSError): self._send(None)
    self.proc.join(timeout=30)
    if self.proc.is_alive(): self.proc.terminate(); self.proc.join(timeout=10)
    if self.reader is not None: self.reader.join(timeout=5)
    self.conn.close()
    self.proc = self.reader = None
  def _send(self, msg: Any) -> None:
    with self.lock: self.conn.send(msg)
  def submit(self, ids: list[int], params: SamplingParams) -> _Handle:
    h = _Handle(asyncio.get_running_loop())
    with self.lock:
      self.ids += 1
      h.rid = self.ids
    self.handles[h.rid] = h
    try:
      if self.dead is None: self._send(("add", h.rid, ids, params)); return h   # dead is set before the reader fails the handles
    except OSError: pass
    self.handles.pop(h.rid, None)
    raise APIError(f"the engine is not available: {self.dead or 'the engine process exited'}", 503, "server_error", code="engine_unavailable")
  def abort(self, h: _Handle) -> None:
    if self.handles.pop(h.rid, None) is not None:
      with contextlib.suppress(OSError): self._send(("abort", h.rid))
  async def score(self, ids: list[int], start: int, k: int) -> dict:
    """LP.Scorer.score in the engine process, between steps (a "score" command; the result comes back as one message)."""
    h = _Handle(asyncio.get_running_loop())
    with self.lock:
      self.ids += 1
      h.rid = self.ids
    self.handles[h.rid] = h
    try:
      if self.dead is not None: raise OSError
      self._send(("score", h.rid, ids, start, k))
    except OSError:
      self.handles.pop(h.rid, None)
      raise APIError(f"the engine is not available: {self.dead or 'the engine process exited'}", 503, "server_error", code="engine_unavailable")
    r = await h.q.get()
    if isinstance(r, BaseException): raise r
    return r

  def _read(self) -> None:
    """Reader thread: the child's messages -> each request's asyncio queue (one loop wake-up per message)."""
    dead = False
    while not dead:
      try: msg = self.conn.recv()
      except (EOFError, OSError): msg, dead = self._died(), True
      kind, by_loop = msg[0], {}
      if kind == "out":
        tp = time.perf_counter() if self.prof_on else None
        for rid, new, fin, reason in msg[1]:
          if (h := self.handles.get(rid)) is None: continue
          o = RequestOutput(rid, new, [], fin, reason)   # _gen counts the tokens itself
          if tp is not None: o.t_post = tp
          by_loop.setdefault(h.loop, []).append((h, o))
          if fin: del self.handles[rid]
        if len(msg) > 2: self.snap = msg[2]
      elif kind == "stats": self.snap = msg[1]
      elif kind == "score":
        if (h := self.handles.pop(msg[1], None)) is not None: by_loop.setdefault(h.loop, []).append((h, msg[2]))
      else:   # "err" (one request: add_request refused it) or "fail" (a step raised: every request in flight)
        rids, value, text = ([msg[1]], msg[2], msg[3]) if kind == "err" else msg[1:]
        for rid in rids:
          if (h := self.handles.pop(rid, None)) is not None: by_loop.setdefault(h.loop, []).append((h, (MemoryError if value == "oom" else ValueError if value else RuntimeError)(text)))
      for loop, items in by_loop.items():
        with contextlib.suppress(RuntimeError): loop.call_soon_threadsafe(_deliver, items)
    if not self.stopping and self.on_dead is not None: self.on_dead()

  def _died(self) -> tuple:
    """The pipe closed: mark the engine dead (before listing the handles, so a racing submit sees it), fail every request."""
    if self.stopping: return ("fail", list(self.handles), False, "the engine process stopped")   # stop(): not a crash
    proc, code = self.proc, None
    if proc is not None:
      with contextlib.suppress(Exception): proc.join(5); code = proc.exitcode
    self.dead = f"the engine process exited (exit code {code})"
    print(f"fusezap: {self.dead}; failing the requests in flight and shutting down", flush=True)
    return ("fail", list(self.handles), False, self.dead)

def _engine_main(conn: Any, args: Any) -> None:
  """The engine process: build and warm up the engine as `fusezap serve` does, report ready, then serve the pipe: drain the
  commands between steps (blocking only while idle) and send each step's outputs as one message."""
  import signal, time, traceback
  signal.signal(signal.SIGINT, signal.SIG_IGN)   # Ctrl-C goes to the server process, which stops this one
  try:
    from fusezap.serve.cli import build_engine
    t0 = time.perf_counter()
    eng = build_engine(args, tokenizer=False)
    print(f"fusezap: loaded {args.model} on {eng.model.device} ({args.dtype}{', ' + args.quant if args.quant else ''}) in {time.perf_counter() - t0:.1f}s", flush=True)
    if not args.no_warmup:
      t0 = time.perf_counter(); warmup(eng)
      print(f"fusezap: warm-up {time.perf_counter() - t0:.1f}s (first run on a cold ~/.cache/fusezap also tunes kernels: minutes)", flush=True)
    print(f"fusezap: mode {eng.mode}", flush=True)
    _print_memory(eng)
    conn.send(("ready", {"max_model_len": eng.max_model_len, "vocab": int(eng.model.embed_host.shape[0]), "mode": eng.mode}))
  except BaseException:
    with contextlib.suppress(OSError): conn.send(("dead", traceback.format_exc()))
    return
  live: set[int] = set()
  st = MX.EngineStats(eng)   # sent with each step's outputs, or alone ("stats") when only it changed
  alog = AllocLog(eng, getattr(args, "log_level", "info") in ("info", "debug", "trace"))
  tr = ReqTrace("engine")
  tt: dict[int, dict] = {}     # rid -> timestamps (FZ_REQ_TRACE)
  pf = SP.from_env()           # FZ_SERVE_PROFILE: one line per step
  gate = eng.admit_gate()      # I76: gather a burst of short prompts into one prefill forward (None: off)
  try:
    while True:
      alog.tick()
      if pf is not None and not eng.has_unfinished(): pf.idle()   # waiting for requests: not a gap
      if not eng.has_unfinished() or conn.poll():
        cmd = conn.recv()
        while True:
          if cmd is None: alog.line("at shutdown"); return
          if cmd[0] == "score":
            _, rid, ids, start, k = cmd
            try: conn.send(("score", rid, LP.scorer(eng).score(ids, start, k)))
            except Exception as e: conn.send(("err", rid, _err_kind(e), str(e)))
          elif cmd[0] == "add":
            _, rid, ids, params = cmd
            try:
              eng.add_request(ids, params, request_id=rid); live.add(rid); st.prompt += len(ids)
              if gate is not None: gate.added()
              if pf is not None: pf.adds += 1
            except Exception as e: conn.send(("err", rid, _err_kind(e), str(e)))
            if tr.f is not None: tt[rid] = {"rid": rid, "add": time.time()}
          elif cmd[1] in live:
            eng.abort(cmd[1]); eng.release(cmd[1]); live.discard(cmd[1])
          if not conn.poll(): break
          cmd = conn.recv()
      if not eng.has_unfinished():
        if st.update(): conn.send(("stats", st.snap))   # e.g. the last running request aborted: no step will report it
        continue
      if gate is not None:   # I76: a burst's next prompts are on their way: wait for them (bounded), then prefill them together
        w = gate.hold()
        while w > 0 and not conn.poll(w): w = gate.hold()
        if w > 0: continue   # a command came: drain it, then ask again
      if tr.f is not None:
        t0 = time.time()
        for rid in live:
          if (d := tt.get(rid)) is not None and "sched" not in d and (s := eng.seqs.get(rid)) is not None and s.first_step < 0: d["pre"] = t0
      rec = None
      if pf is not None: wall = time.time(); pf.begin(eng)
      try:
        outs = eng.step()
        if pf is not None: rec = pf.end(eng, wall)
      except Exception as e:   # fail every in-flight request; the engine stays usable for new ones
        if isinstance(e, MemoryError): _release_memory(eng)
        conn.send(("fail", sorted(live), _err_kind(e), str(e)))
        for rid in live:
          with contextlib.suppress(Exception): eng.abort(rid); eng.release(rid)
        live.clear()
        if st.update(): conn.send(("stats", st.snap))
        continue
      res, fin = [], []
      if gate is not None: gate.finished(sum(o.finished for o in outs))
      for o in outs:
        if o.request_id not in live: continue
        res.append((o.request_id, o.new_token_ids, o.finished, o.finish_reason))
        st.gen += len(o.new_token_ids)
        if o.finished: live.discard(o.request_id); eng.release(o.request_id); fin.append(o.request_id)
      if tr.f is not None:
        for rid, d in tt.items():
          if "sched" not in d and "pre" in d and ((s := eng.seqs.get(rid)) is None or s.first_step >= 0): d["sched"] = d.pop("pre")
      ch = st.update()
      if res: conn.send(("out", res, st.snap))
      elif ch: conn.send(("stats", st.snap))
      if rec is not None: pf.done(rec, len(res))
      if tr.f is not None and fin:
        t1 = time.time()
        for rid in fin:
          if (d := tt.pop(rid, None)) is not None: d["fin"] = d["out"] = t1; tr.write(d)
  except (EOFError, OSError): alog.line("at shutdown"); return   # the server process went away
  finally: close_tp(eng, exit_process=True)

def close_tp(eng: Any, exit_process: bool = False) -> None:
  """Stop the engine's tensor-parallel workers (fusezap.serve.dist: bounded, NCCL aborted); with exit_process (the engine
  process, after its last message) leave with os._exit (1 if an exception is on its way out) so no library teardown can
  keep a stopping server waiting."""
  if (g := getattr(eng, "tp", None)) is None or g.world == 1: return
  from fusezap.serve import dist
  failed = sys.exc_info()[0] is not None   # called from a finally: an exception on its way out
  if failed and exit_process: traceback.print_exc()
  g.close()
  if exit_process:
    dist.dbg(f"engine process: exit {int(failed)}")
    for f in (sys.stdout, sys.stderr):
      with contextlib.suppress(Exception): f.flush()
    os._exit(int(failed))

class AllocLog:
  """Device allocator driver traffic since warm-up (the engine process, CUDA): one line every `every` seconds while the
  engine is busy (when `on`: --log-level info or debug) and one at shutdown. In steady state a serving step should make
  no driver call at all (graph replays reuse their buffers, the free list serves the rest): allocs / frees growing with
  the steps mean cuMemAlloc / cuMemFree, each a device synchronization, on the critical path (round 10's -5% at c32)."""
  def __init__(self, engine: LLMEngine, on: bool = True, every: float = 30.0):
    from fusezap.device import Device
    import time
    al = getattr(Device[engine.model.device], "allocator", None)
    self.al = al if hasattr(al, "counters") and engine.model.device.upper().startswith("CUDA") else None
    self.on, self.every, self.eng = on, every, engine
    self.base = self.al.counters() if self.al else None
    self.t = time.monotonic()
  def line(self, when: str) -> str | None:
    if self.al is None: return None
    c, b, G = self.al.counters(), self.base, 2 ** 30
    msg = (f"fusezap: allocator since warm-up ({when}): {c['allocs'] - b['allocs']} driver allocs "
           f"({(c['alloc_bytes'] - b['alloc_bytes']) / G:.2f} GiB), {c['frees'] - b['frees']} driver frees "
           f"({(c['free_bytes'] - b['free_bytes']) / G:.2f} GiB, {(c['evicted_bytes'] - b['evicted_bytes']) / G:.2f} GiB of it "
           f"evicted over the cap); holds {c['held'] / G:.2f} GiB, {c['cached'] / G:.2f} GiB cached; "
           f"{getattr(self.eng, 'stats', {}).get('steps', 0)} engine steps")
    if self.on or when == "at shutdown": print(msg, flush=True)
    return msg
  def tick(self) -> None:
    import time
    if self.al is None or not self.on or (now := time.monotonic()) - self.t < self.every: return
    self.t = now
    self.line("periodic")

def release_cache(engine: LLMEngine) -> int:
  """Give the device allocator's free list back to the driver (bytes released). Tracing and capturing a forward holds
  all of its intermediates at once; once freed they stay in the allocator's per-size cache, and each forward size traced
  (prefill sizes, decode buckets, throughput mode's mixed sizes) left a different set of sizes behind. On a GPU shared
  with another server (Validate runs vLLM at 45% next to FuseZap) that cache took the memory the other server allocates
  at run time: round 8's throughput-mode run lost its vLLM baseline at the first request."""
  from fusezap.device import Device
  if (tp := getattr(engine.model, "tp", None)) is not None and tp.rank == 0 and tp.depth == 0: tp.post("release")
  dev = Device[engine.model.device]
  al = getattr(dev, "allocator", None)
  if al is None or not hasattr(al, "free_cache"): return 0
  n = al.cached
  if hasattr(dev, "synchronize"): dev.synchronize()
  al.free_cache()
  return n

def warmup(engine: LLMEngine) -> None:
  """Capture both jitted forwards (decode and prefill) before serving; the larger prefill sizes (CUDA) are traced,
  captured and replayed here too, each by prompts that fill it, so their first real step does not compile. The
  allocator's cache is released after each phase and at the end (release_cache), so the memory left in use is the
  weights, the KV pool and the captured graphs' own buffers, whatever the mode."""
  freed = 0
  if getattr(engine, "neuron", False):   # I75: every forward at every pool prefix captured, compiled (in parallel) and loaded
    from fusezap.serve.neuron_prewarm import prewarm
    engine.prewarm_report = prewarm(engine, lambda m: print(m, flush=True))
  engine.generate([[1000 + i for i in range(40)]] * 2, SamplingParams(max_tokens=3, ignore_eos=True))
  freed += release_cache(engine)
  sizes = getattr(engine, "decode_sizes", [])
  for n in sizes if len(sizes) > 1 else []:
    # every decode bucket: enough steps at exactly n rows to trace, capture and replay its graph
    if engine.max_model_len < 48 or n * (48 // engine.cache.bs + 1) > engine.cache.num_free_blocks: break
    engine.generate([[1000 + (i * 7 + j) % 20000 for j in range(40)] for i in range(n)], SamplingParams(max_tokens=8, ignore_eos=True))
  freed += release_cache(engine)
  for C in getattr(engine, "mix_sizes", []):
    # each mixed forward (a prompt chunk of C rows next to decode rows): one decoding request, then four times prompts of
    # C tokens in all (several when C exceeds max_model_len), admitted together
    ps = _fill_prompts(engine, C)
    if ps is None or len(ps) >= engine.cfg.max_num_seqs or engine.max_model_len < 48: continue
    rd = engine.add_request([1000 + j for j in range(40)], SamplingParams(max_tokens=min(40, engine.max_model_len - 41), ignore_eos=True))
    while engine.seqs[rd].in_prompt: engine.step()
    for k in range(4):
      rps = [engine.add_request([2000 + (k * 13 + j) % 20000 for j in p], SamplingParams(max_tokens=1, ignore_eos=True)) for p in ps]
      while any(engine.seqs[r].status != SeqStatus.FINISHED for r in rps): engine.step()
      for r in rps: engine.release(r)
    engine.abort(rd)
    while engine.has_unfinished(): engine.step()
    engine.release(rd)
    freed += release_cache(engine)
  for n in engine.prefill_sizes[1:]:
    if (ps := _fill_prompts(engine, n)) is None: continue
    for _ in range(3): engine.generate(ps, SamplingParams(max_tokens=1, ignore_eos=True))
    freed += release_cache(engine)
  engine.warmup_released = freed
  from fusezap.serve import constrain
  # the llguidance tokenizer (~0.5 s for a 152k vocabulary), so the first constrained request does not stall the steps
  with contextlib.suppress(Exception): constrain.prewarm(getattr(engine, "tokenizer_path", None), engine.model.cfg.vocab_size, engine.stop_ids)

def _print_memory(engine: LLMEngine) -> None:
  """One line on stdout: the device memory free after warm-up and what warm-up released (CUDA)."""
  from fusezap.device import Device
  dev = Device[engine.model.device]
  if getattr(engine, "neuron", False):   # N8 (I71): what holds the NeuronCore's HBM
    from fusezap.serve.neuron_kv import hbm_line
    with contextlib.suppress(Exception): print(hbm_line(engine, "after warm-up"), flush=True)
    return
  if not hasattr(dev, "mem_free") or not engine.model.device.upper().startswith("CUDA"): return
  try:
    dev.ctx()
    free, total = (int(x) for x in dev.cu.cuMemGetInfo()[1:3])
    print(f"fusezap: GPU memory after warm-up: this process holds {dev.allocator.held / 2**30:.2f} GiB (weights, KV pool, "
          f"captured forwards; {dev.allocator.cached / 2**30:.2f} GiB of it cached free blocks, cap "
          f"{(dev.allocator.cache_max or 0) / 2**30:.1f} GiB); {(total - free) / 2**30:.1f} of {total / 2**30:.1f} GiB in use "
          f"on the device (all processes); warm-up released {getattr(engine, 'warmup_released', 0) / 2**30:.2f} GiB", flush=True)
  except Exception: pass

def _fill_prompts(engine: LLMEngine, n: int) -> list[list[int]] | None:
  """Prompts of n tokens in all that one prefill step admits together (None if max_num_seqs, max_model_len or the KV
  pool cannot hold them: then no step reaches that size)."""
  L = engine.max_model_len - 1
  k = -(-n // L)
  lens = [n // k + (i < n % k) for i in range(k)]
  bs = engine.cache.bs
  if k > engine.cfg.max_num_seqs or sum(-(-(m + 1) // bs) for m in lens) > engine.cache.num_free_blocks: return None
  return [[1000 + (i * 7 + j) % 20000 for j in range(m)] for i, m in enumerate(lens)]

# ---------------------------------------------------------------- stop strings

def stop_scan(buf: str, stops: list[str], final: bool) -> tuple[str, bool, str]:
  """(text to emit, stopped, text held back). Holds back a tail that may still grow into a stop string."""
  hits = [i for s in stops if (i := buf.find(s)) >= 0]
  if hits: return buf[:min(hits)], True, ""
  if final or not stops: return buf, False, ""
  hold = max((k for s in stops for k in range(min(len(s) - 1, len(buf)), 0, -1) if buf.endswith(s[:k])), default=0)
  return buf[:len(buf) - hold], False, buf[len(buf) - hold:]

# ---------------------------------------------------------------- request parsing

def _num(body: dict, k: str, default, lo: float, hi: float, cast=float):
  v = body.get(k)
  if v is None: return default
  if isinstance(v, bool) or not isinstance(v, (int, float)): raise APIError(f"{k!r} must be a number", param=k)
  if not lo <= v <= hi: raise APIError(f"{k!r} must be between {lo} and {hi}", param=k)
  return cast(v)

def _text(content: Any) -> str:
  if content is None or isinstance(content, str): return content or ""
  if isinstance(content, list):
    if all(isinstance(p, dict) and p.get("type") == "text" for p in content): return "".join(p.get("text", "") for p in content)
  raise APIError("message content must be a string or a list of text parts", param="messages")

# Request fields that change the output and are not implemented yet: (field, its neutral values or a predicate
# (value, body) -> neutral, what to send instead). A neutral value, as SDKs and frameworks send by default, is accepted;
# any other value is a 400 unsupported_parameter. Fields with no effect on the generated tokens (user, metadata, store,
# service_tier, prediction, prompt_cache_key, safety_identifier, stream_options.include_obfuscation) are accepted.
_ZERO, _OUT = "send 0 or leave it out", "leave it out"
UNSUPPORTED: tuple[tuple[str, Any, str], ...] = (
  ("presence_penalty", (None, 0), _ZERO), ("frequency_penalty", (None, 0), _ZERO),
  ("logit_bias", (None, {}), "send {} or leave it out"),
  ("best_of", lambda v, b: v in (None, 1, b.get("n") or 1), "send 1 or leave it out (n sets the number of choices)"),
  ("suffix", (None, ""), "send an empty string or leave it out"),
  ("functions", (None, []), "use tools instead"), ("function_call", (None, "none", "auto"), "use tool_choice instead"),
  ("modalities", (None, ["text"]), 'send ["text"] or leave it out'), ("audio", (None,), _OUT),
  ("web_search_options", (None,), _OUT),
  ("reasoning_effort", (None, "medium"), 'send "medium" or leave it out'), ("verbosity", (None, "medium"), 'send "medium" or leave it out'),
  # vLLM sampling and template extensions
  ("min_p", (None, 0), _ZERO),
  ("min_tokens", lambda v, b: v in (None, 0) or (b.get("ignore_eos") is True and not b.get("stop")), "send 0 or leave it out (or set ignore_eos)"),
  ("stop_token_ids", (None, []), "use stop strings instead"),
  ("use_beam_search", (None, False), "send false or leave it out"),
  ("include_stop_str_in_output", (None, False), "send false or leave it out"),
  ("skip_special_tokens", (None, True), "send true or leave it out"),
  ("prompt_logprobs", (None,), "use /v1/completions with echo and logprobs instead"),
  ("truncate_prompt_tokens", (None,), _OUT),
  ("guided_json", (None,), "use response_format json_schema instead"), ("guided_regex", (None,), _OUT),
  ("guided_choice", (None,), _OUT), ("guided_grammar", (None,), _OUT),
  ("chat_template", (None,), _OUT), ("chat_template_kwargs", (None, {}), "send {} or leave it out"),
  ("add_generation_prompt", (None, True), "send true or leave it out"), ("continue_final_message", (None, False), "send false or leave it out"),
)

def check_unsupported(body: dict) -> None:
  """A 400 unsupported_parameter for the first UNSUPPORTED field with a value other than its neutral ones."""
  for k, ok, hint in UNSUPPORTED:
    v = body.get(k)
    if not (ok(v, body) if callable(ok) else any(v is o if isinstance(v, bool) else v == o for o in ok)):   # false is not 0
      raise APIError(f"{k} is not supported by FuseZap Serve yet; {hint}", param=k, code="unsupported_parameter")

class Server:
  """engine: an LLMEngine (stepped on a worker thread, AsyncEngine) or a started ProcEngine."""
  def __init__(self, engine: LLMEngine | ProcEngine, tokenizer: Tokenizer, model_name: str, api_key: str | None = None):
    self.trace = ReqTrace("api")
    self.tok, self.name, self.api_key = tokenizer, model_name, api_key
    if isinstance(engine, ProcEngine):
      self.async_engine, self.max_model_len, self.vocab = engine, engine.info["max_model_len"], engine.info["vocab"]
      self.mode = engine.info.get("mode", "exact")
    else:
      self.async_engine, self.max_model_len, self.vocab = AsyncEngine(engine), engine.max_model_len, engine.model.embed_host.shape[0]
      self.mode = getattr(engine, "mode", "exact")
    self.created = int(time.time())
    # FZ_SERVE_PROFILE: one line per request (the handler's lag and time per output)
    self.prof = getattr(self.async_engine, "prof", None) or SP.from_env()

  def _params(self, body: dict, n_prompt: int, default_max: int | None) -> tuple[SamplingParams, list[str]]:
    room = self.max_model_len - n_prompt
    mt = body.get("max_completion_tokens", body.get("max_tokens"))
    if mt is None: mt = default_max if default_max is not None else room
    elif isinstance(mt, bool) or not isinstance(mt, int) or mt < 1: raise APIError("'max_tokens' must be a positive integer", param="max_tokens")
    if mt > room:
      if body.get("max_tokens") is None and body.get("max_completion_tokens") is None: mt = room
      else: raise APIError(f"This model's maximum context length is {self.max_model_len} tokens; the request has {n_prompt} prompt tokens "
                           f"and asks for {mt} completion tokens", param="max_tokens", code="context_length_exceeded")
    if mt < 1: raise APIError(f"prompt of {n_prompt} tokens does not fit the context length {self.max_model_len}", param="messages", code="context_length_exceeded")
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else (stop or [])
    if not isinstance(stops, list) or len(stops) > 4 or not all(isinstance(s, str) for s in stops): raise APIError("'stop' must be a string or a list of up to 4 strings", param="stop")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)): raise APIError("'seed' must be an integer", param="seed")
    temp, top_p, rp = _num(body, "temperature", 1.0, 0, 2), _num(body, "top_p", 1.0, 0, 1), _num(body, "repetition_penalty", 1.0, 0, 10)
    if rp <= 0: raise APIError("'repetition_penalty' must be greater than 0", param="repetition_penalty")
    if top_p <= 0: temp, top_p = 0.0, 1.0      # top_p=0 means "only the most likely token", i.e. greedy
    ignore_eos = body.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool): raise APIError("'ignore_eos' must be a boolean", param="ignore_eos")
    sp = SamplingParams(max_tokens=mt, temperature=temp, top_p=top_p, top_k=0 if body.get("top_k") == -1 else _num(body, "top_k", 0, 0, 1 << 30, int), repetition_penalty=rp, seed=seed,
                        ignore_eos=ignore_eos)
    return sp, [s for s in stops if s]

  def _check_model(self, body: dict) -> None:
    if body.get("model") not in (None, self.name):
      raise APIError(f"The model {body.get('model')!r} does not exist", 404, code="model_not_found", param="model")

  async def _gen(self, ids: list[int], sp: SamplingParams, stops: list[str], trace: dict | None = None) -> AsyncIterator[tuple[str, str | None, int, list[int]]]:
    """Yields (text delta, finish_reason or None, completion tokens so far, the new token ids) for one generation.
    trace (FZ_REQ_TRACE): filled with the engine request number and the submit / first / last output times."""
    if trace is not None: trace["submit"] = time.time()
    h = self.async_engine.submit(ids, sp)
    if trace is not None: trace["rid"] = h.rid
    dec, buf, done, ntok, pend = IncrementalDecoder(self.tok), "", False, 0, []
    held: Any = None   # an item taken while coalescing that must be handled on its own (an error)
    pf = self.prof
    if pf is not None: lags, hms, coal, t_got = [], [], 0, None
    try:
      while True:
        if pf is not None and t_got is not None: hms.append(1e3 * (time.perf_counter() - t_got))
        if held is not None: out, held = held, None
        else: out = await h.q.get()
        if pf is not None:
          t_got = time.perf_counter()
          if (tp := getattr(out, "t_post", None)) is not None: lags.append(1e3 * (t_got - tp))
          coal += h.q.qsize() if not isinstance(out, BaseException) and not out.finished else 0
        # coalesce: outputs that queued up while this handler was busy go out as one chunk (vLLM's output collector does
        # the same). Round 12: with 32 streams the API ran ~2 steps behind the engine and the client further behind the
        # API (the last [DONE] of a wave reached the client ~49 ms after it was written); one chunk per backlog, not per
        # token, cuts that work. Token counts come from usage; the tokens and text are unchanged.
        while not isinstance(out, BaseException) and not out.finished and not h.q.empty():
          nxt = h.q.get_nowait()
          if isinstance(nxt, BaseException): held = nxt; break
          out = RequestOutput(out.request_id, list(out.new_token_ids) + list(nxt.new_token_ids), out.token_ids, nxt.finished, nxt.finish_reason, out.num_prompt_tokens)
        if trace is not None:
          trace.setdefault("first", time.time())
          if not isinstance(out, BaseException) and out.finished: trace["last"] = time.time()
        if isinstance(out, BaseException):
          done = True
          raise _api_error(out)
        text = dec.push_many(out.new_token_ids)
        ntok += len(out.new_token_ids)
        pend += out.new_token_ids
        fin = ("length" if out.finish_reason == "length" else "stop") if out.finished else None
        if fin: text += dec.flush()
        emit, stopped, buf = stop_scan(buf + text, stops, bool(fin))
        if stopped: yield emit, "stop", ntok, pend; return
        if fin: done = True
        if emit or fin: yield emit, fin, ntok, pend; pend = []
        if fin: return
    finally:
      if not done: self.async_engine.abort(h)
      if pf is not None:
        if t_got is not None: hms.append(1e3 * (time.perf_counter() - t_got))
        pf.request(h.rid, ntok, lags, hms, coal)

  async def _merge(self, gens: list[AsyncIterator]) -> AsyncIterator[tuple[int, tuple]]:
    if len(gens) == 1:   # one choice (n = 1, one prompt): no pump task and queue hop per token
      try:
        async for item in gens[0]: yield 0, item
      finally: await gens[0].aclose()
      return
    q: asyncio.Queue = asyncio.Queue()
    async def pump(i: int, g: AsyncIterator):
      try:
        async for item in g: await q.put((i, item))
        await q.put((i, None))
      except BaseException as e:
        await q.put((i, e))
        if isinstance(e, asyncio.CancelledError): raise
    tasks = [asyncio.create_task(pump(i, g)) for i, g in enumerate(gens)]
    try:
      left = len(tasks)
      while left:
        i, item = await q.get()
        if isinstance(item, BaseException): raise item
        if item is None: left -= 1
        else: yield i, item
    finally:
      for t in tasks: t.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)
      for g in gens: await g.aclose()

  def _extras(self, body: dict, chat: bool) -> dict:
    """The request's logprobs / echo / tool / response_format options, validated: {k (top-k, None = no logprobs), echo,
    as_ids (return_tokens_as_token_ids), tools (TL.ToolSpec), constraint (TL.CallConstraint for a "required" / named
    tool_choice, else None: the structure a grammar-constrained decoder must enforce after the forced prefix)}."""
    as_ids = body.get("return_tokens_as_token_ids", False)
    if as_ids is None: as_ids = False
    if not isinstance(as_ids, bool): raise APIError("'return_tokens_as_token_ids' must be a boolean", param="return_tokens_as_token_ids")
    if (ptc := body.get("parallel_tool_calls")) is not None and not isinstance(ptc, bool):
      raise APIError("'parallel_tool_calls' must be a boolean", param="parallel_tool_calls")
    if chat:
      lp, top = body.get("logprobs", False), body.get("top_logprobs")
      if lp is None: lp = False
      if not isinstance(lp, bool): raise APIError("'logprobs' must be a boolean", param="logprobs")
      if top is not None:
        if isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= LP.MAX_TOP: raise APIError(f"'top_logprobs' must be an integer between 0 and {LP.MAX_TOP}", param="top_logprobs")
        if not lp: raise APIError("'top_logprobs' needs 'logprobs': true", param="top_logprobs")
      if body.get("echo"): raise APIError("'echo' is only supported on /v1/completions", param="echo")
      k, echo = ((top or 0) if lp else None), False
    else:
      k, echo = body.get("logprobs"), body.get("echo", False)
      if echo is None: echo = False
      if k is not None and (isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= LP.MAX_TOP): raise APIError(f"'logprobs' must be an integer between 0 and {LP.MAX_TOP}", param="logprobs")
      if not isinstance(echo, bool): raise APIError("'echo' must be a boolean", param="echo")
      if body.get("tools"): raise APIError("'tools' is only supported on /v1/chat/completions", param="tools")
    try:
      rf = TL.check_response_format(body, self.tok)
      tools = TL.tool_spec(body, self.tok) if chat else TL.ToolSpec()
    except TL.ToolError as e: raise APIError(str(e), param=e.param)
    return {"k": k, "echo": echo, "as_ids": as_ids, "tools": tools, "constraint": rf}

  async def _handle(self, req: Request, chat: bool):
    t_recv = time.time()
    try: body = await req.json()
    except Exception: raise APIError("request body is not valid JSON")
    if not isinstance(body, dict): raise APIError("request body must be a JSON object")
    self._check_model(body)
    check_unsupported(body)
    ex = self._extras(body, chat)
    spec: TL.ToolSpec = ex["tools"]
    texts_in: list[str] = []   # completions: the prompt strings as given (echo)
    cons = ex["constraint"]    # grammar-constrained output (constrain.Constraint): response_format, or strict tools below
    if chat:
      msgs = body.get("messages")
      if not isinstance(msgs, list) or not msgs or not all(isinstance(m, dict) and "role" in m for m in msgs): raise APIError("'messages' must be a non-empty list of messages", param="messages")
      msgs = TL.request_messages([{**m, "content": _text(m.get("content"))} for m in msgs], spec.fmt)
      try:
        p0 = self.tok.apply_chat_template(msgs, True, spec.tools if spec.active else None)
        pre = self.tok.encode(spec.forced_prefix, add_special_tokens=False) if spec.forced_prefix else []
        prompts = [p0 + pre]
      except Exception as e: raise APIError(f"cannot apply the chat template: {e}", param="messages")
      if spec.active:
        try: cons = spec.constraint(self.tok, pre, body.get("parallel_tool_calls") is not False) or cons
        except TL.ToolError as e: raise APIError(str(e), param=e.param)
    else:
      p = body.get("prompt")
      if isinstance(p, str): prompts, texts_in = [self.tok.encode(p, add_special_tokens=False)], [p]
      elif isinstance(p, list) and p and all(isinstance(t, int) for t in p): prompts = [p]
      elif isinstance(p, list) and p and all(isinstance(t, str) for t in p): prompts, texts_in = [self.tok.encode(t, add_special_tokens=False) for t in p], list(p)
      elif isinstance(p, list) and p and all(isinstance(t, list) and t and all(isinstance(x, int) for x in t) for t in p): prompts = p
      else: raise APIError("'prompt' must be a string, a list of strings or token ids, or a list of token id lists", param="prompt")
    vocab = self.vocab
    if any(p and (min(p) < 0 or max(p) >= vocab) for p in prompts): raise APIError(f"token ids must be in [0, {vocab})", param="prompt")
    if any(not p for p in prompts): raise APIError("empty prompt", param="messages" if chat else "prompt")
    n = body.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 16: raise APIError("'n' must be an integer between 1 and 16", param="n")
    work = []      # one entry per choice: (prompt ids, sampling params, stops)
    src = []       # per choice: the index of its prompt
    for pi, p in enumerate(prompts):
      sp, stops = self._params(body, len(p), None if chat else 16)
      sp.constraint = cons
      for j in range(n):
        work.append((p, SamplingParams(**{**sp.__dict__, "seed": None if sp.seed is None else sp.seed + j}) if j else sp, stops))
        src.append(pi)
    stream = bool(body.get("stream"))
    # logprobs, echo and Hermes / Mistral tool calls need the whole output (to score or parse it): such a stream is sent once it is done
    buffered = ex["k"] is not None or ex["echo"] or (spec.active and not TL.streams_live(spec))
    one_call = body.get("parallel_tool_calls") is False   # at most one tool call per choice (strict tools: also in the grammar)
    live_tools = spec.active and not buffered   # llama3_json: content streams live, a call is held to the end of its choice
    usage_in_stream = bool((body.get("stream_options") or {}).get("include_usage"))
    rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
    created = int(time.time())
    n_prompt = sum(len(p) for p in prompts)
    traces = [{"recv": t_recv, "parsed": time.time()} for _ in work] if self.trace.f is not None else None
    gens = [self._gen(*w, trace=traces[i] if traces else None) for i, w in enumerate(work)]

    def chunk(index: int, text: str, fin: str | None, first: bool = False) -> dict:
      if chat:
        delta = ({"role": "assistant", "content": text} if first else {"content": text}) if (text or first) else {}
        ch = {"index": index, "delta": delta, "finish_reason": fin}
        return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.name, "choices": [ch]}
      return {"id": rid, "object": "text_completion", "created": created, "model": self.name, "choices": [{"index": index, "text": text, "finish_reason": fin}]}
    def usage(ntoks: list[int]) -> dict:
      c = sum(ntoks)
      return {"prompt_tokens": n_prompt, "completion_tokens": c, "total_tokens": n_prompt + c}

    async def collect() -> tuple[list[str], list[str], list[int], list[list[int]]] | None:
      """Every choice run to its end: (texts, finish reasons, token counts, token ids); None if the client went away
      (the generations are then aborted)."""
      texts, fins, ntoks, ids = [""] * len(work), ["stop"] * len(work), [0] * len(work), [[] for _ in work]
      merged = self._merge(gens)
      async def watch():      # abort the generations once the client goes away (Starlette does not cancel plain handlers)
        while not await req.is_disconnected(): await asyncio.sleep(0.25)
      wt = asyncio.create_task(watch())
      try:
        async for i, (t, f, k, new) in merged:
          texts[i] += t; ntoks[i] = k; ids[i] += new
          if f: fins[i] = f
          if wt.done(): return None
      finally:
        wt.cancel(); await merged.aclose()
      return texts, fins, ntoks, ids

    async def finish(texts: list[str], fins: list[str], ids: list[list[int]]) -> list[dict]:
      """The response's choices: logprobs scored, echo prepended, tool calls parsed."""
      out = []
      for i in range(len(work)):
        p, sc = work[i][0], None
        if ex["k"] is not None:
          try: sc = await self.async_engine.score(list(p) + ids[i], 0 if ex["echo"] else len(p), ex["k"])
          except APIError: raise
          except Exception as e:
            raise _api_error(e, "logprobs: ")
        if chat:
          msg: dict[str, Any] = {"role": "assistant", "content": texts[i]}
          fin = fins[i]
          if spec.active:
            raw = spec.forced_prefix + (self.tok.decode(ids[i], skip_special_tokens=False) if spec.fmt == "mistral" else texts[i])
            calls, content = TL.parse(spec, raw)
            calls = calls[:1] if one_call else calls
            if calls:
              msg = {"role": "assistant", "content": content, "tool_calls": calls}
              if fin == "stop" and spec.mode != "named": fin = "tool_calls"   # a named tool_choice finishes with "stop" (OpenAI, vLLM)
          out.append({"index": i, "message": msg, "logprobs": LP.chat_logprobs(self.tok, ids[i], sc, ex["k"]) if sc else None, "finish_reason": fin})
        else:
          text, pre = texts[i], []
          if ex["echo"]:
            text, pre = (texts_in[src[i]] if texts_in else self.tok.decode(p, skip_special_tokens=False)) + text, list(p)
          lp = LP.completion_logprobs(self.tok, pre + ids[i], sc, len(pre), ex["as_ids"]) if sc is not None else None
          out.append({"index": i, "text": text, "finish_reason": fins[i], "logprobs": lp})
      return out

    if not stream:
      res = await collect()
      if res is None: return JSONResponse({}, status_code=499)
      texts, fins, ntoks, ids = res
      choices = await finish(texts, fins, ids)
      return JSONResponse({"id": rid, "object": "chat.completion" if chat else "text_completion", "created": created, "model": self.name, "choices": choices, "usage": usage(ntoks)})

    enc = lambda o: b"data: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n"
    if buffered:   # generate (and score / parse) first, so errors are still a proper HTTP error; then stream the result
      res = await collect()
      if res is None: return JSONResponse({}, status_code=499)
      texts, fins, ntoks, ids = res
      choices = await finish(texts, fins, ids)
      async def sse_done() -> AsyncIterator[bytes]:
        for ch in choices:
          i = ch["index"]
          if chat:
            yield enc(chunk(i, "", None, first=True))
            m = ch["message"]
            delta: dict[str, Any] = {}
            if m.get("content"): delta["content"] = m["content"]
            if m.get("tool_calls"): delta["tool_calls"] = [{"index": j, **c} for j, c in enumerate(m["tool_calls"])]
            if delta or ch["logprobs"]:
              yield enc({"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.name,
                         "choices": [{"index": i, "delta": delta, "logprobs": ch["logprobs"], "finish_reason": None}]})
            yield enc(chunk(i, "", ch["finish_reason"]))
          else:
            yield enc({"id": rid, "object": "text_completion", "created": created, "model": self.name,
                       "choices": [{"index": i, "text": ch["text"], "logprobs": ch["logprobs"], "finish_reason": ch["finish_reason"]}]})
        if usage_in_stream:
          yield enc({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion", "created": created, "model": self.name, "choices": [], "usage": usage(ntoks)})
        yield b"data: [DONE]\n\n"
      return StreamingResponse(sse_done(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # Peek at the first event so errors (bad request, engine failure) become a proper HTTP error instead of a broken stream.
    merged = self._merge(gens)
    try: first = await merged.__anext__()
    except StopAsyncIteration: first = None
    except BaseException:
      await merged.aclose(); raise
    kind: list[str | None] = [None] * len(work)   # live_tools: per choice None (undecided) | "content" | "tool"
    held = [""] * len(work)
    def tool_end(i: int, text: str, fin: str) -> list[dict]:
      """A held llama3_json choice at its end: tool_calls deltas (per call: id, type, name and empty arguments, then the
      arguments) and finish_reason "tool_calls" (a named tool_choice: "stop"); not a call: the text as content."""
      calls, content = TL.parse(spec, spec.forced_prefix + text)
      calls = calls[:1] if one_call else calls
      def ev(delta): return {"id": rid, "object": "chat.completion.chunk", "created": created, "model": self.name, "choices": [{"index": i, "delta": delta, "finish_reason": None}]}
      if not calls: return ([chunk(i, text, None)] if text else []) + [chunk(i, "", fin)]
      out = [ev({"content": content})] if content else []
      for j, c in enumerate(calls):
        out.append(ev({"tool_calls": [{"index": j, "id": c["id"], "type": "function", "function": {"name": c["function"]["name"], "arguments": ""}}]}))
        out.append(ev({"tool_calls": [{"index": j, "function": {"arguments": c["function"]["arguments"]}}]}))
      return out + [chunk(i, "", "tool_calls" if fin == "stop" and spec.mode != "named" else fin)]
    async def sse() -> AsyncIterator[bytes]:
      ntoks = [0] * len(work)
      try:
        if chat:
          for i in range(len(work)): yield enc(chunk(i, "", None, first=True))
        pending = [first] if first else []
        it = merged.__aiter__()
        while True:
          if pending: item = pending.pop()
          else:
            try: item = await it.__anext__()
            except StopAsyncIteration: break
          i, (t, f, k, _) = item
          ntoks[i] = k
          if live_tools and kind[i] != "content":
            held[i] += t
            if kind[i] is None:
              kind[i] = TL.stream_kind(spec, held[i], final=bool(f))
              if kind[i] == "content": t, held[i] = held[i], ""
            if kind[i] != "content":   # undecided (never at the end) or a call held to the end
              if f:
                for o in tool_end(i, held[i], f): yield enc(o)
              continue
          if t or f: yield enc(chunk(i, t, f))
        if usage_in_stream:
          yield enc({"id": rid, "object": "chat.completion.chunk" if chat else "text_completion", "created": created, "model": self.name, "choices": [], "usage": usage(ntoks)})
        yield b"data: [DONE]\n\n"
        if traces:
          td = time.time()
          for t in traces: self.trace.write({**t, "done": td})
      except APIError as e:
        yield enc({"error": {"message": e.message, "type": e.type, "param": e.param, "code": e.code}})
      finally:
        await merged.aclose()
    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

  # ---- routes
  async def chat(self, req: Request): return await self._handle(req, True)
  async def completions(self, req: Request): return await self._handle(req, False)
  async def models(self, req: Request):
    return JSONResponse({"object": "list", "data": [{"id": self.name, "object": "model", "created": self.created, "owned_by": "fusezap",
                                                       "max_model_len": self.max_model_len, "fusezap_mode": self.mode}]})
  async def model(self, req: Request):
    if req.path_params["model_id"] != self.name: raise APIError(f"The model {req.path_params['model_id']!r} does not exist", 404, code="model_not_found")
    return JSONResponse({"id": self.name, "object": "model", "created": self.created, "owned_by": "fusezap", "max_model_len": self.max_model_len,
                         "fusezap_mode": self.mode})
  async def metrics(self, req: Request):
    return Response(MX.render(self.name, self.async_engine.snap, MX.rss_bytes()), media_type=MX.CONTENT_TYPE)
  async def health(self, req: Request):
    if (dead := getattr(self.async_engine, "dead", None)) is not None: return JSONResponse({"status": "error", "error": dead}, 503)
    return JSONResponse({"status": "ok"})

def create_app(engine: LLMEngine | ProcEngine, tokenizer: Tokenizer, model_name: str, api_key: str | None = None) -> Starlette:
  s = Server(engine, tokenizer, model_name, api_key)
  @contextlib.asynccontextmanager
  async def lifespan(app):
    s.async_engine.start()
    try: yield
    finally: s.async_engine.stop()
  def guard(fn):
    async def h(req: Request):
      if api_key and req.url.path not in ("/health", "/metrics") and req.headers.get("authorization") != f"Bearer {api_key}":
        return APIError("Incorrect API key provided", 401, "authentication_error", code="invalid_api_key").response()
      try: r = await fn(req)
      except APIError as e: r = e.response()
      r.headers["x-fusezap-mode"] = s.mode   # exact | throughput: clients (Validate) record which mode served them
      return r
    return h
  routes = [Route("/health", guard(s.health)), Route("/metrics", guard(s.metrics)), Route("/v1/models", guard(s.models)), Route("/v1/models/{model_id:path}", guard(s.model)),
            Route("/v1/chat/completions", guard(s.chat), methods=["POST"]), Route("/v1/completions", guard(s.completions), methods=["POST"])]
  app = Starlette(routes=routes, lifespan=lifespan)
  app.state.server = s
  return app
