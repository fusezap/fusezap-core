"""I81 D7: ImageEngine, the non-autoregressive engine for diffusion image models (docs/I82_DIFFUSION_SCOPE.md §3).

  eng = ImageEngine(pipe, max_batch=4)                          # pipe: a DiffusionPipeline (below)
  rid = eng.add(ImageParams(prompt="a red fox", seed=7, width=512, height=512, steps=4))
  while eng.has_work(): done = eng.step()                       # -> [ImageResult(rid, image uint8 (H, W, 3), ...)]
  eng.generate([ImageParams(...), ...])                         # the same, blocking, results in order

No KV cache and no tokens: a request is encoded once (its text conditioning), then runs a fixed number of denoising
steps, then a decode. The engine owns the loop so that requests can share steps:
  - Buckets. Requests are grouped by size (width, height). Each bucket runs one fz.jit-captured step per padded batch
    size (batch_sizes, default 1, 2, 4): the transformer's velocity and the flow-match Euler update
    x <- x + (sigma_next - sigma) * v in one replay. The timestep sigma and the step size are per-row *tensor* inputs,
    so one capture serves every step index of every request (and, on NEURON, one compiled kernel set per bucket).
  - Batching: FIFO per bucket, up to max_batch rows. Rows are independent requests that may be at different step
    indices (step-level continuous batching): a request admitted while others are mid-trajectory joins their next
    step. Padding rows (batch below the bucket size) get zero latents, zero conditioning and dt = 0 and are dropped.
    With batch_sizes = (max_batch,) every step runs one capture, so a request's result cannot depend on which other
    requests shared its steps (on row-independent kernels: the same bits alone or batched).
  - Buckets take turns oldest-first: each engine step serves the bucket whose oldest active request arrived first.
  - Host-side: noise (numpy, seeded: the same noise as a reference run), the sigma schedule (incl. any shift), the text
    conditioning (encoded once per prompt, shared by a request's n images), the final uint8 conversion.

DiffusionPipeline (what the engine needs; the FLUX.2 klein pipeline implements it, I81 D6):
  device: str                                   device the step runs on
  name: str                                     model id
  default_size: tuple[int, int], default_steps: int
  check_size(width, height) -> None             raise ValueError if the pipeline cannot make that size (optional)
  encode(prompts: list[str]) -> np.ndarray      (B, L, C) conditioning, float32 host array, fixed (L, C) per pipeline
  noise(seed, width, height) -> np.ndarray      (*latent_shape) float32 initial latents (host RNG, numpy)
  sigmas(steps, width, height) -> np.ndarray    (steps + 1,) float32, decreasing, last 0
  timesteps(steps, width, height) -> np.ndarray (optional) (steps,) float32: the t fed to velocity at each step (default
                                                the sigmas; FLUX.2 passes (sigma * 1000) / 1000 rounded in float32)
  velocity(x, cond, t, width, height) -> Tensor traceable: x (B, *latent_shape) f32, cond (B, L, C) f32, t (B,) sigma f32
  decode(x, width, height) -> Tensor            traceable: (B, *latent_shape) -> (B, 3, H, W) image in [-1, 1]
A pipeline that only offers `generate(prompt=None, prompt_embeds=None, seed=, size=(w, h), steps=) -> image` ((3, H, W)
or (1, 3, H, W) in [-1, 1], a Tensor or numpy) is served one request at a time through it (MonolithicPipeline), with
no step batching.
"""
from __future__ import annotations
import collections, itertools, os, threading, time
from dataclasses import dataclass, field
from typing import Any, Sequence
import numpy as np
from fusezap.tensor import Tensor
from fusezap.jit import Jit

@dataclass
class ImageParams:
  prompt: str | None = None
  prompt_embeds: np.ndarray | None = None     # (L, C) or (1, L, C): skip the text encoder
  seed: int | None = None                      # None: a random seed (reported in the result)
  width: int | None = None
  height: int | None = None
  steps: int | None = None

@dataclass
class ImageResult:
  rid: int
  image: np.ndarray                            # (H, W, 3) uint8
  seed: int
  width: int
  height: int
  steps: int
  timings: dict = field(default_factory=dict)  # queue_s, encode_s, denoise_s, decode_s, total_s
  idx: int = 0                                 # which of the request's n images (seed = the request's seed + idx)

def to_uint8(img: np.ndarray) -> np.ndarray:
  """(3, H, W) or (H, W, 3) float in [-1, 1] -> (H, W, 3) uint8, as diffusers' VaeImageProcessor.postprocess +
  numpy_to_pil: (x / 2 + 0.5).clamp(0, 1), * 255, round half to even (numpy round), uint8."""
  a = np.asarray(img, np.float32)
  if a.ndim == 4: a = a[0]
  if a.shape[0] == 3 and a.shape[-1] != 3: a = a.transpose(1, 2, 0)
  return (np.clip(a / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)

class MonolithicPipeline:
  """Adapter for a pipeline that only has generate(): ImageEngine runs it as one opaque call per image."""
  monolithic = True
  def __init__(self, pipe: Any):
    self.pipe = pipe
    self.name = getattr(pipe, "name", type(pipe).__name__)
    self.device = getattr(pipe, "device", None)
    self.default_size = tuple(getattr(pipe, "default_size", (1024, 1024)))
    self.default_steps = int(getattr(pipe, "default_steps", 4))
  def check_size(self, w: int, h: int) -> None:
    if (f := getattr(self.pipe, "check_size", None)) is not None: f(w, h)
  def run(self, prompt: str | None, embeds: np.ndarray | None, seed: int, w: int, h: int, steps: int) -> np.ndarray:
    out = self.pipe.generate(prompt=prompt, prompt_embeds=embeds, seed=seed, size=(w, h), steps=steps)
    return to_uint8(out.numpy() if isinstance(out, Tensor) else out)

@dataclass
class _Row:
  """One image: a request with n images is n rows (seeds seed, seed + 1, ...)."""
  rid: int
  idx: int                       # its index among the request's images
  seed: int
  w: int
  h: int
  steps: int
  cond: np.ndarray | None        # (L, C)
  x: np.ndarray | None = None    # current latents
  sig: np.ndarray | None = None  # (steps + 1,)
  tv: np.ndarray | None = None   # (steps,) the model's timestep per step
  i: int = 0                     # steps done
  t_add: float = 0.0
  t_admit: float = 0.0
  enc_s: float = 0.0
  den_s: float = 0.0
  prompt: str | None = None

class EngineStats:
  """Counters and gauges for /metrics (fusezap/serve/image_api.py); written by the engine thread only."""
  def __init__(self):
    self.requests = self.images = self.steps = self.row_steps = self.failed = 0
    self.step_s = self.e2e_s = self.encode_s = self.decode_s = 0.0
    self.running = self.waiting = 0
    self.done_times: collections.deque = collections.deque(maxlen=4096)
  def images_per_s(self, window: float = 60.0) -> float:
    now = time.monotonic()
    n = sum(1 for t in list(self.done_times) if now - t <= window)
    return n / window

class ImageEngine:
  def __init__(self, pipe: Any, max_batch: int = 4, batch_sizes: Sequence[int] | None = None, max_waiting: int = 256):
    self.pipe = pipe if hasattr(pipe, "velocity") or isinstance(pipe, MonolithicPipeline) else MonolithicPipeline(pipe)
    self.mono = isinstance(self.pipe, MonolithicPipeline)
    self.max_batch = 1 if self.mono else int(max_batch)
    bs = sorted({int(b) for b in (batch_sizes or [b for b in (1, 2, 4, 8, 16) if b < self.max_batch] + [self.max_batch])})
    assert bs and bs[-1] >= self.max_batch, f"batch_sizes {bs} must reach max_batch {self.max_batch}"
    self.batch_sizes = bs
    self.max_waiting = max_waiting
    self.device = getattr(self.pipe, "device", None)
    self.name = getattr(self.pipe, "name", "image-model")
    self.default_size = tuple(getattr(self.pipe, "default_size", (1024, 1024)))
    self.default_steps = int(getattr(self.pipe, "default_steps", 4))
    self.waiting: collections.deque[_Row] = collections.deque()
    self.active: dict[tuple[int, int], list[_Row]] = {}
    from fusezap.helpers import getenv
    self.use_jit = bool(getattr(pipe, "jit", True)) and bool(getenv("FZ_IMAGE_JIT", 1))   # see _denoise
    self._step_jits: dict[tuple[int, int, int], Jit] = {}
    self._decode_jits: dict[tuple[int, int], Jit] = {}
    self._rid = itertools.count()
    self._n: dict[int, int] = {}                 # rid -> images still pending
    self.stats = EngineStats()
    self._lock = threading.Lock()                # add() may be called from another thread than step()

  # ---- requests
  def resolve(self, p: ImageParams) -> tuple[int, int, int, int]:
    """(width, height, steps, seed) with defaults filled in; raises ValueError on a bad request."""
    w = int(self.default_size[0] if p.width is None else p.width)
    h = int(self.default_size[1] if p.height is None else p.height)
    steps = int(self.default_steps if p.steps is None else p.steps)
    if w <= 0 or h <= 0: raise ValueError(f"bad size {w}x{h}")
    if not 1 <= steps <= 1000: raise ValueError(f"num_inference_steps must be in [1, 1000], got {steps}")
    if (f := getattr(self.pipe, "check_size", None)) is not None: f(w, h)
    seed = int.from_bytes(os.urandom(4), "little") if p.seed is None else int(p.seed)
    if not 0 <= seed < 2 ** 63: raise ValueError(f"seed must be in [0, 2**63), got {seed}")
    if p.prompt is None and p.prompt_embeds is None: raise ValueError("a prompt (or prompt_embeds) is required")
    return w, h, steps, seed

  def add(self, p: ImageParams, n: int = 1) -> int:
    """Queue n images of one prompt (seeds seed .. seed + n - 1). Returns the request id."""
    w, h, steps, seed = self.resolve(p)
    emb = None if p.prompt_embeds is None else np.asarray(p.prompt_embeds, np.float32).reshape(np.shape(p.prompt_embeds)[-2:])
    with self._lock:
      if len(self.waiting) + n > self.max_waiting: raise OverflowError(f"queue full ({len(self.waiting)} images waiting)")
      rid = next(self._rid)
      now = time.monotonic()
      for k in range(n): self.waiting.append(_Row(rid, k, (seed + k) % 2 ** 63, w, h, steps, emb, t_add=now, prompt=p.prompt))
      self._n[rid] = n
      self.stats.requests += 1
      self.stats.waiting = len(self.waiting)
    return rid

  def has_work(self) -> bool: return bool(self.waiting) or any(self.active.values())

  # ---- the loop
  def _admit(self) -> None:
    """Move waiting rows into their bucket while it has room (FIFO per bucket); encode prompts that need it (once per
    request: its rows share the conditioning)."""
    with self._lock:
      take, keep = [], collections.deque()
      room = {k: self.max_batch - len(v) for k, v in self.active.items()}
      for r in self.waiting:
        k = (r.w, r.h)
        if room.setdefault(k, self.max_batch) > 0: take.append(r); room[k] -= 1
        else: keep.append(r)
      self.waiting = keep
    if self.mono:
      for r in take: self.active.setdefault((r.w, r.h), []).append(r)
      return
    enc: dict[int, np.ndarray] = {}
    for r in take:
      t0 = time.perf_counter()
      if r.cond is None:
        if r.rid not in enc: enc[r.rid] = np.asarray(self.pipe.encode([r.prompt]), np.float32)[0]
        r.cond = enc[r.rid]
      r.x = np.asarray(self.pipe.noise(r.seed, r.w, r.h), np.float32)
      r.sig = np.asarray(self.pipe.sigmas(r.steps, r.w, r.h), np.float32)
      assert r.sig.shape == (r.steps + 1,), f"sigmas {r.sig.shape} for {r.steps} steps"
      r.tv = _timesteps(self.pipe, r.sig, r.steps, r.w, r.h)
      r.enc_s = time.perf_counter() - t0
      self.stats.encode_s += r.enc_s
      r.t_admit = time.monotonic()
      self.active.setdefault((r.w, r.h), []).append(r)

  def _bucket_batch(self, n: int) -> int: return next(b for b in self.batch_sizes if b >= n)

  def _step_fn(self, w: int, h: int):
    pipe = self.pipe
    def f(x: Tensor, cond: Tensor, t: Tensor, dt: Tensor) -> Tensor:
      v = pipe.velocity(x, cond, t, w, h)
      return (x + dt.reshape((x.shape[0],) + (1,) * (x.ndim - 1)) * v.float()).contiguous()
    return f

  def _denoise(self, key: tuple[int, int], rows: list[_Row]) -> None:
    w, h = key
    B = self._bucket_batch(len(rows))
    x = np.zeros((B,) + rows[0].x.shape, np.float32)
    cond = np.zeros((B,) + rows[0].cond.shape, np.float32)
    t, dt = np.ones(B, np.float32), np.zeros(B, np.float32)
    for j, r in enumerate(rows):
      x[j], cond[j], t[j], dt[j] = r.x, r.cond, r.tv[r.i], r.sig[r.i + 1] - r.sig[r.i]
    jk = (B, w, h)
    # plan: the replayed intermediates share memory by live range (FZ_IMAGE_PLAN=0: each its own buffer). A pipeline with
    # jit = False (or FZ_IMAGE_JIT=0) runs the step eagerly: a capture holds every intermediate of the forward at once
    # (each recorded kernel keeps its buffers), +26 GB for the klein 4B step at 512^2 and more than a 48 GB Mac at 1024^2
    if jk not in self._step_jits: self._step_jits[jk] = Jit(self._step_fn(w, h), plan=_plan()) if self.use_jit else self._step_fn(w, h)
    t0 = time.perf_counter()
    dev = self.device
    out = self._step_jits[jk](Tensor(x, device=dev), Tensor(cond, device=dev), Tensor(t, device=dev), Tensor(dt, device=dev)).numpy()
    el = time.perf_counter() - t0
    self.stats.steps += 1; self.stats.row_steps += len(rows); self.stats.step_s += el
    self.last_step = {"batch": len(rows), "bucket_batch": B, "size": key, "s": el}
    for j, r in enumerate(rows):
      r.x, r.i = out[j], r.i + 1
      r.den_s += el

  def _decode(self, r: _Row) -> np.ndarray:
    k = (r.w, r.h)
    if k not in self._decode_jits:
      pipe = self.pipe
      fn = lambda x, w=r.w, h=r.h: pipe.decode(x, w, h).float().contiguous()
      self._decode_jits[k] = Jit(fn, plan=_plan()) if self.use_jit else fn
    return to_uint8(self._decode_jits[k](Tensor(r.x[None], device=self.device)).numpy()[0])

  def _finish(self, r: _Row, img: np.ndarray, dec_s: float) -> ImageResult:
    now = time.monotonic()
    tm = {"queue_s": r.t_admit - r.t_add, "encode_s": r.enc_s, "denoise_s": r.den_s, "decode_s": dec_s, "total_s": now - r.t_add}
    self.stats.images += 1; self.stats.e2e_s += tm["total_s"]; self.stats.decode_s += dec_s
    self.stats.done_times.append(now)
    with self._lock:
      self._n[r.rid] -= 1
      if not self._n[r.rid]: del self._n[r.rid]
    return ImageResult(r.rid, img, r.seed, r.w, r.h, r.steps, tm, r.idx)

  def step(self) -> list[ImageResult]:
    """Admit what fits, then advance one bucket (the one whose oldest active request is oldest) by one denoising
    step; rows that finish are decoded. Returns the images finished in this step."""
    self._admit()
    live = [(min(r.t_add for r in rows), k) for k, rows in self.active.items() if rows]
    out: list[ImageResult] = []
    if live:
      key = min(live)[1]
      rows = self.active[key]
      if self.mono:
        r = rows.pop(0)
        r.t_admit = time.monotonic()
        t0 = time.perf_counter()
        img = self.pipe.run(r.prompt, r.cond, r.seed, r.w, r.h, r.steps)
        r.den_s = time.perf_counter() - t0
        self.stats.steps += 1; self.stats.row_steps += r.steps; self.stats.step_s += r.den_s
        out.append(self._finish(r, img, 0.0))
      else:
        self._denoise(key, rows)
        for r in [r for r in rows if r.i >= r.steps]:
          rows.remove(r)
          t0 = time.perf_counter()
          img = self._decode(r)
          out.append(self._finish(r, img, time.perf_counter() - t0))
    with self._lock:
      self.stats.waiting = len(self.waiting)
      self.stats.running = sum(len(v) for v in self.active.values())
    return out

  def abort(self, rid: int) -> None:
    """Drop a request's rows that are queued or in flight."""
    with self._lock:
      self.waiting = collections.deque(r for r in self.waiting if r.rid != rid)
      for k in self.active: self.active[k] = [r for r in self.active[k] if r.rid != rid]
      self._n.pop(rid, None)

  def generate(self, params: Sequence[ImageParams], n: int = 1) -> list[list[ImageResult]]:
    """Blocking: run the requests to completion; per request its n images in seed order."""
    rids = [self.add(p, n) for p in params]
    got: dict[int, list[ImageResult]] = {r: [] for r in rids}
    while self.has_work():
      for res in self.step(): got[res.rid].append(res)
    return [sorted(got[r], key=lambda x: x.idx) for r in rids]

  def warmup(self, sizes: Sequence[tuple[int, int]] | None = None, steps: int = 1) -> None:
    """Capture the step and decode jits of each size bucket at every batch size (fz.jit captures on its 2nd call)."""
    if self.mono: return
    for (w, h) in sizes or [self.default_size]:
      for B in self.batch_sizes:
        for _ in range(2): self.generate([ImageParams(prompt="warmup", seed=k, width=w, height=h, steps=steps) for k in range(B)])
    self.stats = EngineStats()

def _timesteps(pipe: Any, sig: np.ndarray, steps: int, w: int, h: int) -> np.ndarray:
  if (f := getattr(pipe, "timesteps", None)) is None: return sig[:-1]
  tv = np.asarray(f(steps, w, h), np.float32)
  assert tv.shape == (steps,), f"timesteps {tv.shape} for {steps} steps"
  return tv

def _plan() -> bool:
  from fusezap.helpers import getenv
  return bool(getenv("FZ_IMAGE_PLAN", 1))

def reference_image(pipe: Any, p: ImageParams) -> np.ndarray:
  """The plain loop (no jit, batch 1, no engine): encode, noise, Euler steps on host-driven sigmas, decode -> uint8.
  ImageEngine's images must match it (tests)."""
  w, h, steps, seed = int(p.width or pipe.default_size[0]), int(p.height or pipe.default_size[1]), int(p.steps or pipe.default_steps), int(p.seed or 0)
  dev = pipe.device
  cond = np.asarray(pipe.encode([p.prompt]), np.float32) if p.prompt_embeds is None else np.asarray(p.prompt_embeds, np.float32).reshape((1,) + np.shape(p.prompt_embeds)[-2:])
  x = Tensor(np.asarray(pipe.noise(seed, w, h), np.float32)[None], device=dev)
  sig = np.asarray(pipe.sigmas(steps, w, h), np.float32)
  tv = _timesteps(pipe, sig, steps, w, h)
  c = Tensor(cond, device=dev)
  for i in range(steps):
    v = pipe.velocity(x, c, Tensor(tv[i:i + 1], device=dev), w, h)
    x = (x + Tensor(sig[i + 1:i + 2] - sig[i:i + 1], device=dev).reshape((1,) + (1,) * (x.ndim - 1)) * v.float()).realize()
  return to_uint8(pipe.decode(x, w, h).float().numpy()[0])
