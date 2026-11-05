"""I81 D7: OpenAI-compatible image generation over ImageEngine (fusezap/serve/image_engine.py), starlette + uvicorn.

  app = create_image_app(engine, model_name)       # starlette app; uvicorn.run(app, ...)
  POST /v1/images/generations   GET /v1/models   GET /health   GET /metrics
  fusezap serve-image <pipeline> [--device D] [--port N] ...   (image_cli.py; <pipeline>: a registered name or module:callable)

Request (OpenAI images API, https://platform.openai.com/docs/api-reference/images/create):
  prompt (required), model (must be the served name if given), n 1..10, size "WxH" (or "auto": the model default),
  response_format "b64_json" (default; "url" needs storage: 400), output_format png (default) | jpeg | webp,
  output_compression 0..100 (jpeg / webp quality, default 100), stream (false only).
  Accepted and ignored: quality, style, background ("transparent" is refused), moderation, user, safety_identifier,
  partial_images (0 only).
FuseZap extensions: seed (int; image k of n uses seed + k; default random, echoed back), num_inference_steps (default:
the model's, 4 for FLUX.2 klein), guidance_scale (ignored: the target model is guidance-distilled).
Response: {created, data: [{b64_json, revised_prompt: null, seed}], output_format, size, quality: "auto", background:
"opaque", usage: {input_tokens: 0, output_tokens: 0, total_tokens: 0, ...}} (the openai client's ImagesResponse).

`AsyncImageEngine` owns the ImageEngine on one worker thread: handlers submit through a queue; the thread admits new
requests between steps (so a request joins its size bucket's running batch at the next step) and resolves each
request's future when its n images are done. PNG / JPEG / WebP encoding runs on the event loop's default executor.

/metrics (Prometheus text format 0.0.4; FuseZap names, there is no vLLM equivalent for images):
  fusezap:image_requests_running   gauge    images in the running buckets (denoising or decoding)
  fusezap:image_requests_waiting   gauge    images queued, not admitted yet (queue depth)
  fusezap:image_requests_total     counter  requests accepted
  fusezap:images_generated_total   counter  images finished
  fusezap:image_steps_total        counter  batched denoising steps run (one transformer forward each)
  fusezap:image_row_steps_total    counter  image-steps run (sum of the steps' real batch rows)
  fusezap:image_step_seconds       summary  _sum / _count over the batched steps (device time + host sync)
  fusezap:image_e2e_seconds        summary  _sum / _count, request queued -> image ready (per image)
  fusezap:image_encode_seconds_total, fusezap:image_decode_seconds_total   counters
  fusezap:images_per_second        gauge    images finished in the last 60 s / 60
  process_resident_memory_bytes    gauge
"""
from __future__ import annotations
import asyncio, base64, contextlib, io, queue, threading, time, traceback
from typing import Any, Callable
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from fusezap.serve.image_engine import ImageEngine, ImageParams, ImageResult
from fusezap.serve import metrics as MX

FORMATS = {"png": ("PNG", "image/png"), "jpeg": ("JPEG", "image/jpeg"), "webp": ("WEBP", "image/webp")}

class APIError(Exception):
  def __init__(self, message: str, status: int = 400, type: str = "invalid_request_error", param: str | None = None, code: str | None = None):
    super().__init__(message)
    self.message, self.status, self.type, self.param, self.code = message, status, type, param, code
  def response(self) -> JSONResponse:
    return JSONResponse({"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}, self.status)

def encode_image(img: Any, fmt: str = "png", compression: int = 100) -> bytes:
  """(H, W, 3) uint8 -> PNG (lossless; compression is ignored) / JPEG / WebP bytes (quality = compression)."""
  from PIL import Image
  name, _ = FORMATS[fmt]
  buf = io.BytesIO()
  im = Image.fromarray(img, "RGB")
  if fmt == "png": im.save(buf, name)
  elif fmt == "jpeg": im.save(buf, name, quality=max(1, min(100, int(compression))))
  else: im.save(buf, name, quality=max(0, min(100, int(compression))), lossless=int(compression) >= 100)
  return buf.getvalue()

class AsyncImageEngine:
  def __init__(self, engine: ImageEngine):
    self.engine = engine
    self.cmds: queue.Queue = queue.Queue()
    self.thread: threading.Thread | None = None
    self.dead: str | None = None
    self._pending: dict[int, tuple[asyncio.AbstractEventLoop, asyncio.Future, list[ImageResult], int]] = {}
  def start(self) -> None:
    if self.thread is None:
      self.thread = threading.Thread(target=self._run, name="fusezap-image-engine", daemon=True)
      self.thread.start()
  def stop(self) -> None:
    if self.thread is not None:
      self.cmds.put(("stop",))
      self.thread.join(timeout=30)
      self.thread = None
  def submit(self, p: ImageParams, n: int) -> asyncio.Future:
    """A future resolving to the n ImageResults in seed order; ValueError / OverflowError for a bad or excess request."""
    self.engine.resolve(p)   # validate in the handler (raises ValueError), before queueing
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    self.cmds.put(("add", p, n, loop, fut))
    return fut
  def cancel(self, fut: asyncio.Future) -> None: self.cmds.put(("cancel", fut))
  def _resolve(self, loop, fut, value=None, exc: BaseException | None = None) -> None:
    def f():
      if not fut.done(): fut.set_exception(exc) if exc is not None else fut.set_result(value)
    loop.call_soon_threadsafe(f)
  def _cmd(self, c: tuple) -> bool:
    if c[0] == "stop": return False
    if c[0] == "add":
      _, p, n, loop, fut = c
      try: rid = self.engine.add(p, n)
      except Exception as e: self._resolve(loop, fut, exc=e); return True
      self._pending[rid] = (loop, fut, [], n)
    elif c[0] == "cancel":
      for rid, (_, fut, _, _) in list(self._pending.items()):
        if fut is c[1]: self.engine.abort(rid); del self._pending[rid]
    return True
  def _run(self) -> None:
    eng = self.engine
    try:
      while True:
        try: c = self.cmds.get(timeout=None if not eng.has_work() else 0)   # idle: block until work arrives
        except queue.Empty: c = None
        while c is not None:
          if not self._cmd(c): return
          try: c = self.cmds.get_nowait()
          except queue.Empty: c = None
        if not eng.has_work(): continue
        try: done = eng.step()
        except Exception as e:   # fail every request in flight; the engine stays up for new ones
          traceback.print_exc()
          eng.stats.failed += len(self._pending)
          for rid, (loop, fut, _, _) in list(self._pending.items()):
            eng.abort(rid); self._resolve(loop, fut, exc=RuntimeError(f"image generation failed: {e}"))
          self._pending.clear()
          continue
        for r in done:
          if (p := self._pending.get(r.rid)) is None: continue
          p[2].append(r)
          if len(p[2]) == p[3]:
            del self._pending[r.rid]
            self._resolve(p[0], p[1], sorted(p[2], key=lambda x: x.idx))
    except BaseException as e:
      self.dead = f"{type(e).__name__}: {e}"
      raise

def _int(body: dict, k: str, default: int | None, lo: int, hi: int) -> int | None:
  v = body.get(k, default)
  if v is None: return None
  if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v: raise APIError(f"{k} must be an integer", param=k)
  if not lo <= int(v) <= hi: raise APIError(f"{k} must be in [{lo}, {hi}], got {v}", param=k)
  return int(v)

class ImageServer:
  def __init__(self, engine: ImageEngine, model_name: str, api_key: str | None = None):
    self.engine, self.name, self.api_key = engine, model_name, api_key
    self.async_engine = AsyncImageEngine(engine)
    self.created = int(time.time())

  def parse(self, body: dict) -> tuple[ImageParams, int, str, int]:
    if not isinstance(body, dict): raise APIError("the request body must be a JSON object")
    if body.get("model") not in (None, self.name):
      raise APIError(f"The model {body.get('model')!r} does not exist", 404, code="model_not_found", param="model")
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt: raise APIError("prompt is required (a non-empty string)", param="prompt")
    if len(prompt) > 32000: raise APIError("prompt is longer than 32000 characters", param="prompt")
    n = _int(body, "n", 1, 1, 10)
    rf = body.get("response_format") or "b64_json"
    if rf != "b64_json": raise APIError("response_format 'url' is not supported: use 'b64_json'", param="response_format")
    fmt = body.get("output_format") or "png"
    if fmt not in FORMATS: raise APIError(f"output_format must be png, jpeg or webp, got {fmt!r}", param="output_format")
    comp = _int(body, "output_compression", 100, 0, 100)
    if body.get("stream"): raise APIError("stream is not supported", param="stream")
    if body.get("partial_images"): raise APIError("partial_images is not supported", param="partial_images")
    if body.get("background") == "transparent": raise APIError("background 'transparent' is not supported", param="background")
    size = body.get("size") or "auto"
    if size == "auto": w, h = self.engine.default_size
    else:
      try: w, h = (int(s) for s in str(size).lower().split("x"))
      except ValueError: raise APIError(f"size must be 'WIDTHxHEIGHT' or 'auto', got {size!r}", param="size") from None
    seed = _int(body, "seed", None, 0, 2 ** 63 - 1)
    steps = _int(body, "num_inference_steps", None, 1, 1000)
    p = ImageParams(prompt=prompt, seed=seed, width=w, height=h, steps=steps)
    try: self.engine.resolve(p)
    except ValueError as e: raise APIError(str(e), param="size" if "size" in str(e) else None) from None
    return p, n, fmt, comp

  async def generations(self, req: Request):
    try: body = await req.json()
    except Exception: raise APIError("the request body is not valid JSON") from None
    p, n, fmt, comp = self.parse(body)
    if p.seed is None:   # one random seed per request, echoed back, so the images can be reproduced
      import os
      p.seed = int.from_bytes(os.urandom(4), "little")
    try: fut = self.async_engine.submit(p, n)
    except ValueError as e: raise APIError(str(e)) from None
    async def watch():   # abort once the client goes away
      while not fut.done():
        if await req.is_disconnected(): self.async_engine.cancel(fut); return
        await asyncio.sleep(0.25)
    w = asyncio.ensure_future(watch())
    try: res = await fut
    except OverflowError as e: raise APIError(str(e), 429, "rate_limit_error") from None
    except ValueError as e: raise APIError(str(e)) from None
    except RuntimeError as e: raise APIError(str(e), 500, "server_error") from None
    finally: w.cancel()
    loop = asyncio.get_running_loop()
    blobs = await asyncio.gather(*[loop.run_in_executor(None, encode_image, r.image, fmt, comp) for r in res])
    data = [{"b64_json": base64.b64encode(b).decode(), "revised_prompt": None, "seed": r.seed,
             "fusezap_timings": {k: round(v, 6) for k, v in r.timings.items()}} for r, b in zip(res, blobs)]
    return JSONResponse({"created": int(time.time()), "data": data, "output_format": fmt, "size": f"{res[0].width}x{res[0].height}",
                         "quality": "auto", "background": "opaque",
                         "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
                                   "input_tokens_details": {"image_tokens": 0, "text_tokens": 0}},
                         "fusezap": {"num_inference_steps": res[0].steps, "model": self.name}})

  async def models(self, req: Request):
    return JSONResponse({"object": "list", "data": [{"id": self.name, "object": "model", "created": self.created, "owned_by": "fusezap",
                                                       "fusezap_kind": "image", "default_size": "%dx%d" % self.engine.default_size,
                                                       "default_steps": self.engine.default_steps}]})
  async def health(self, req: Request):
    if self.async_engine.dead: return JSONResponse({"status": "error", "error": self.async_engine.dead}, 503)
    return JSONResponse({"status": "ok"})
  async def metrics(self, req: Request):
    return Response(render_metrics(self.name, self.engine), media_type=MX.CONTENT_TYPE)

def render_metrics(model: str, eng: ImageEngine) -> str:
  s = eng.stats
  lab = f'{{model_name="{MX._esc(model)}"}}'
  rows = [("fusezap:image_requests_running", "gauge", "Images in the running buckets.", s.running),
          ("fusezap:image_requests_waiting", "gauge", "Images queued, not admitted yet.", s.waiting),
          ("fusezap:image_requests_total", "counter", "Image requests accepted.", s.requests),
          ("fusezap:images_generated_total", "counter", "Images finished.", s.images),
          ("fusezap:image_steps_total", "counter", "Batched denoising steps run.", s.steps),
          ("fusezap:image_row_steps_total", "counter", "Image-steps run (real batch rows).", s.row_steps),
          ("fusezap:image_encode_seconds_total", "counter", "Seconds spent encoding prompts and preparing latents.", round(s.encode_s, 6)),
          ("fusezap:image_decode_seconds_total", "counter", "Seconds spent decoding latents.", round(s.decode_s, 6)),
          ("fusezap:images_per_second", "gauge", "Images finished in the last 60 s, per second.", round(s.images_per_s(), 6))]
  out = []
  for name, kind, hlp, v in rows: out += [f"# HELP {name} {hlp}", f"# TYPE {name} {kind}", f"{name}{lab} {MX._num(v)}"]
  for name, hlp, sm, cnt in (("fusezap:image_step_seconds", "Batched denoising step latency.", s.step_s, s.steps),
                             ("fusezap:image_e2e_seconds", "Request queued to image ready, per image.", s.e2e_s, s.images)):
    out += [f"# HELP {name} {hlp}", f"# TYPE {name} summary", f"{name}_sum{lab} {MX._num(round(sm, 6))}", f"{name}_count{lab} {cnt}"]
  if (rss := MX.rss_bytes()) is not None:
    out += ["# HELP process_resident_memory_bytes Resident memory size in bytes.", "# TYPE process_resident_memory_bytes gauge",
            f"process_resident_memory_bytes {rss}"]
  return "\n".join(out) + "\n"

def create_image_app(engine: ImageEngine, model_name: str | None = None, api_key: str | None = None) -> Starlette:
  s = ImageServer(engine, model_name or engine.name, api_key)
  @contextlib.asynccontextmanager
  async def lifespan(app):
    s.async_engine.start()
    try: yield
    finally: s.async_engine.stop()
  def guard(fn: Callable):
    async def h(req: Request):
      if api_key and req.url.path not in ("/health", "/metrics") and req.headers.get("authorization") != f"Bearer {api_key}":
        return APIError("Incorrect API key provided", 401, "authentication_error", code="invalid_api_key").response()
      try: return await fn(req)
      except APIError as e: return e.response()
    return h
  routes = [Route("/health", guard(s.health)), Route("/metrics", guard(s.metrics)), Route("/v1/models", guard(s.models)),
            Route("/v1/images/generations", guard(s.generations), methods=["POST"])]
  app = Starlette(routes=routes, lifespan=lifespan)
  app.state.server = s
  return app

# ---------------------------------------------------------------- pipelines and the CLI

def _klein(device: str | None = None, dtype: str = "bfloat16") -> Any:
  from fusezap.serve.models.klein_image import _factory   # needs the FLUX.2 modules (I81 D4-D6)
  return _factory(device, dtype)

PIPELINES: dict[str, Callable[..., Any]] = {"flux2-klein": _klein}   # name -> factory(device=, dtype=)

def load_pipeline(spec: str, device: str | None = None, dtype: str = "float32") -> Any:
  """A registered pipeline name, or "module:callable" (called with device= and dtype=)."""
  if spec in PIPELINES: return PIPELINES[spec](device=device, dtype=dtype)
  if ":" in spec:
    import importlib
    mod, attr = spec.split(":", 1)
    fn = getattr(importlib.import_module(mod), attr)
    try: return fn(device=device, dtype=dtype)
    except TypeError: return fn(device=device) if device else fn()
  raise ValueError(f"unknown image pipeline {spec!r} (registered: {sorted(PIPELINES)}; or pass module:callable)")
