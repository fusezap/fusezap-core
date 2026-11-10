"""I81 D8 / I95 end to end (docs/I82_DIFFUSION_SCOPE.md §4 G6 bf16 and §5 D8): FLUX.2 [klein] 4B in bfloat16 through
FuseZap (text encoder D3 + transformer D4 + VAE D5 + ImageEngine D7) against diffusers bfloat16 on the same device,
both compared with the float32 CPU reference. Device generic: METAL (torch "mps"), CUDA (torch "cuda"), CPU.

The reference is scripts/diff_ref.py's (diffusers + transformers float32 on the CPU, its PROMPTS and numpy noise):
  uv run python scripts/diff_ref.py denoise --sizes 1024 --prompts 0-3 --seeds 0
  uv run python scripts/diff_ref.py decode  --sizes 1024 --prompts 0-3 --seeds 0
  uv run python scripts/diff_klein_e2e.py --device CUDA [--out results/diff/cuda/h100] [--ref-root DIR] [--prompts 0-7]
Only <ref-root>/<ref>/decode/<size>_p<i>_s<seed>.png and (for fz-bf16-ref) <ref-root>/<ref>/embeds/p<i>.npz are read.

Phases, each in its own subprocess so one pipeline is in memory at a time; a failed phase is recorded
(<out>/<tag>.error.json: stage + traceback) and the others still run:
  df-bf16     diffusers bfloat16 on the device from the prompt (its own encoder), the same noise; timed (s/image, warm)
  fz-bf16     FuseZap bfloat16 from the prompt (Qwen3TextEncoder), the same noise, through ImageEngine; timed, with the
              engine's encode / denoise / decode split. It first runs a smoke image (256^2, 1 step): if that fails (a
              missing kernel, a compile error), the remaining FuseZap phases are skipped.
  fz-bf16-ref FuseZap bfloat16 from the reference embeds (isolates the transformer + VAE from the encoder)
The FuseZap VAE is float32 except on CUDA GPUs under 40 GB, where KleinImagePipeline picks bfloat16 (FZ_KLEIN_VAE_DTYPE
overrides); report.json records it as fz_vae_dtype.
--fz-encode staged (default on CUDA GPUs under 40 GB, e.g. an RTX 4090): the float32 encoder is loaded alone, encodes
every prompt, is released, then the transformer + VAE are loaded without it (KleinImagePipeline(load_encoder=False));
s/image then adds the prompt's (warm) encode time. resident: the encoder stays loaded (the serving layout).
Peak memory: METAL phys_footprint; CUDA torch.cuda.max_memory_allocated / reserved (diffusers), the FuseZap allocator's
peak held bytes (fz), and the device's nvidia-smi memory.used peak sampled every 0.25 s (both); CPU ru_maxrss.
--deadline (unix time): phases are killed at it and the per-prompt loops stop early, so a job finishes in its budget.

Gate G6 (bf16, relative): PSNR(fz-bf16, ref-f32) >= PSNR(df-bf16, ref-f32) - 1 dB, per prompt and seed. SSIM is
reported. Writes <out>/report.json, the PNGs, and exits 1 if a gate fails or a phase did not produce its images.
--model DIR (a diffusers-layout dir) replaces the 4B checkpoint; the noise shape follows its configs (tests: a tiny model).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, threading, time, traceback
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_ref import PROMPTS, OUT, parse_list

REPO = "black-forest-labs/FLUX.2-klein-4B"
TORCH_DEVICE = {"METAL": "mps", "CUDA": "cuda", "CPU": "cpu"}
FZ_PHASES = ("fz-bf16", "fz-bf16-ref")
SMOKE = int(os.environ.get("FZ_E2E_SMOKE_SIZE", 256))   # the fail-fast image side (min with --size)

# ---------------------------------------------------------------- model layout, noise

def _root(a) -> str:
  from fusezap.serve.models.text_encoder import resolve_subfolders
  return resolve_subfolders(a.model, ("text_encoder", "tokenizer", "transformer", "vae", "scheduler"), allow_download=True)

def latent_layout(root: str) -> tuple[int, int]:
  """(latent channels, pixels per latent token side) from the transformer and VAE configs: (128, 16) for klein."""
  tc = json.load(open(os.path.join(root, "transformer", "config.json")))
  vc = json.load(open(os.path.join(root, "vae", "config.json")))
  return int(tc["in_channels"]), 2 ** (len(vc["block_out_channels"]) - 1) * 2

def noise(seed: int, root: str, size: int) -> np.ndarray:   # == diff_ref.noise for klein (numpy PCG64, (1, C, H/16, W/16))
  from fusezap.serve.models.flux2_klein import noise as fz_noise
  c, f = latent_layout(root)
  return fz_noise(seed, c, size // f, size // f)

# ---------------------------------------------------------------- memory

def peak_footprint_gb() -> float:
  """This process's peak phys_footprint (task_info TASK_VM_INFO ledger_phys_footprint_peak): Metal buffers included."""
  import ctypes, ctypes.util
  class TVI(ctypes.Structure):
    _fields_ = [("v", ctypes.c_uint64), ("rc", ctypes.c_int32), ("ps", ctypes.c_int32)] + [(f"u{i}", ctypes.c_uint64) for i in range(16)] + \
               [("phys_footprint", ctypes.c_uint64), ("min_address", ctypes.c_uint64), ("max_address", ctypes.c_uint64),
                ("peak", ctypes.c_uint64), ("pad", ctypes.c_uint64 * 40)]
  libc = ctypes.CDLL(ctypes.util.find_library("c"))
  info, cnt = TVI(), ctypes.c_uint32(ctypes.sizeof(TVI) // 4)
  if libc.task_info(libc.mach_task_self(), 22, ctypes.byref(info), ctypes.byref(cnt)) != 0: return float("nan")
  return info.peak / 1e9

def maxrss_gb() -> float:
  import resource
  r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9   # bytes on macOS, KiB on Linux

class GpuMemSampler:
  """Peak nvidia-smi memory.used (GB) of GPU 0 (CUDA_VISIBLE_DEVICES order) while running; None without nvidia-smi."""
  def __init__(self, period: float = 0.25):
    self.period, self.peak, self._stop = period, None, threading.Event()
    self._th = threading.Thread(target=self._run, daemon=True)
  def _read(self) -> float | None:
    try:
      out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
      return float(out[0]) / 1024 if out else None   # MiB -> GiB
    except Exception: return None
  def _run(self) -> None:
    while not self._stop.is_set():
      v = self._read()
      if v is not None: self.peak = v if self.peak is None else max(self.peak, v)
      self._stop.wait(self.period)
  def __enter__(self): self._th.start(); return self
  def __exit__(self, *exc): self._stop.set(); self._th.join(timeout=15)

def track_fz_peak(device: str) -> dict:
  """Wraps the FuseZap device allocator's alloc to record its peak `held` bytes (in use + cached)."""
  from fusezap.device import Device
  st: dict = {"peak": 0}
  try: al = Device[device].allocator
  except Exception: return st
  if not hasattr(al, "held"): return st
  st["peak"], orig = al.held, al.alloc
  def alloc(nbytes: int):
    h = orig(nbytes); st["peak"] = max(st["peak"], al.held); return h
  al.alloc = alloc
  return st

def memory_report(device: str, sampler: GpuMemSampler | None, fz_peak: dict | None = None) -> dict:
  m: dict = {}
  if device == "METAL": m["peak_footprint_gb"] = peak_footprint_gb()
  if device == "CPU": m["peak_footprint_gb"] = maxrss_gb()
  if device == "CUDA":
    try:
      import torch
      if torch.cuda.is_initialized():
        m["torch_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        m["torch_max_reserved_gb"] = torch.cuda.max_memory_reserved() / 2 ** 30
    except Exception: pass
    if fz_peak is not None: m["fz_alloc_peak_gb"] = fz_peak.get("peak", 0) / 2 ** 30
    if sampler is not None: m["gpu_mem_used_peak_gb"] = sampler.peak
    m["peak_footprint_gb"] = m.get("torch_max_reserved_gb", m.get("fz_alloc_peak_gb"))   # the summary's one number
  m["host_maxrss_gb"] = maxrss_gb()
  return m

def cuda_total_gb() -> float | None:
  try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    return float(out[0]) / 1024 if out else None
  except Exception: return None

# ---------------------------------------------------------------- references, images

def ref_png(a, pi: int, seed: int) -> str: return os.path.join(a.ref_root, a.ref, "decode", f"{a.size}_p{pi}_s{seed}.png")
def ref_embeds(a, pi: int) -> np.ndarray: return np.load(os.path.join(a.ref_root, a.ref, "embeds", f"p{pi}.npz"))["embeds"]

def _png(path: str, img: np.ndarray) -> None:
  from PIL import Image
  Image.fromarray(img, "RGB").save(path)

def _np_uint8(x: np.ndarray) -> np.ndarray:   # diffusers output_type="np" ([0, 1] floats) -> numpy_to_pil's uint8
  return (np.asarray(x[0], np.float32) * 255).round().astype(np.uint8)

def _out_of_time(a, need_s: float = 0.0) -> bool: return bool(a.deadline) and time.time() + need_s > a.deadline

# ---------------------------------------------------------------- phases

def _sync(device: str) -> None:
  import torch
  if device == "CUDA": torch.cuda.synchronize()
  elif device == "METAL": torch.mps.synchronize()

def _diffusers(root: str, dtype, tdev: str):
  """The stock Flux2KleinPipeline, built per component (works for the 4B dir and for a tiny test layout)."""
  import transformers
  from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
  j = lambda s: os.path.join(root, s)
  mi = json.load(open(j("model_index.json")))
  pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(j("scheduler")),
                            vae=AutoencoderKLFlux2.from_pretrained(j("vae"), torch_dtype=dtype),
                            text_encoder=transformers.Qwen3ForCausalLM.from_pretrained(j("text_encoder"), torch_dtype=dtype),
                            tokenizer=transformers.AutoTokenizer.from_pretrained(j("tokenizer")),
                            transformer=Flux2Transformer2DModel.from_pretrained(j("transformer"), torch_dtype=dtype),
                            is_distilled=bool(mi.get("is_distilled", True)))
  offload = False
  try: pipe = pipe.to(tdev)
  except RuntimeError as e:   # CUDA OOM on a small card: diffusers' own offload (recorded; slower)
    if tdev != "cuda" or "out of memory" not in str(e).lower(): raise
    import torch; torch.cuda.empty_cache()
    pipe.enable_model_cpu_offload(); offload = True
  return pipe, offload

def phase_df_bf16(a) -> None:
  import torch
  root, tdev, tag = _root(a), TORCH_DEVICE[a.device], "df-bf16"
  with GpuMemSampler() if a.device == "CUDA" else _Null() as smp:
    t0 = time.time()
    pipe, offload = _diffusers(root, torch.bfloat16, tdev)
    load_s, times = time.time() - t0, []
    for pi in a.prompt_list:
      for seed in a.seeds:
        if times and _out_of_time(a, 2.5 * times[-1]): print(f"{tag}: out of time, stopping before p{pi} s{seed}", flush=True); break
        for rep in range(2):   # the second run is the warm timing
          _sync(a.device); t0 = time.time()
          with torch.no_grad():
            img = pipe(prompt=PROMPTS[pi], latents=torch.tensor(noise(seed, root, a.size), dtype=torch.bfloat16), height=a.size,
                       width=a.size, num_inference_steps=4, output_type="np").images
          _sync(a.device); dt = time.time() - t0
        times.append(dt)
        print(f"{tag} p{pi} s{seed}: {dt:.2f} s warm", flush=True)
        _png(os.path.join(a.out, f"{tag}_p{pi}_s{seed}.png"), _np_uint8(img))
    mem = memory_report(a.device, smp if a.device == "CUDA" else None)
  json.dump({"s_per_image": times, "load_s": load_s, "offload": offload, "device": a.device, **mem},
            open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)

class _Null:
  peak = None
  def __enter__(self): return self
  def __exit__(self, *exc): pass

def _staged_embeds(a, root: str) -> tuple[dict[int, np.ndarray], dict[int, float], float]:
  """The float32 Qwen3 encoder alone on the device: every prompt encoded (twice; the second is the timed one), then the
  encoder is released and the device cache trimmed before the transformer is loaded."""
  import gc
  from fusezap.dtype import dtypes
  from fusezap.serve.models.text_encoder import Qwen3TextEncoder
  from fusezap.serve.models.flux2_vae import trim_cache
  t0 = time.time()
  enc = Qwen3TextEncoder.from_pretrained(root, getattr(dtypes, os.environ.get("FZ_KLEIN_TE_DTYPE", "float32")), a.device)
  load_s = time.time() - t0
  emb, secs = {}, {}
  for pi in a.prompt_list:
    for rep in range(2):
      t0 = time.time(); e = np.asarray(enc.encode([PROMPTS[pi]]), np.float32)[0]; secs[pi] = time.time() - t0
    emb[pi] = e
    print(f"staged encode p{pi}: {secs[pi]:.2f} s warm", flush=True)
  enc.release(); del enc; gc.collect(); trim_cache(a.device)
  return emb, secs, load_s

def phase_fz(a, use_ref_embeds: bool) -> None:
  from fusezap.serve.models.klein_image import KleinImagePipeline
  from fusezap.serve.image_engine import ImageEngine, ImageParams
  tag = "fz-bf16-ref" if use_ref_embeds else "fz-bf16"
  root, stage, info = _root(a), "load", {"device": a.device, "fz_encode": a.fz_encode,
                                         "env": {k: v for k, v in os.environ.items() if k.startswith("FZ_KLEIN") or k == "FZ_FLASH"}}
  staged = a.fz_encode == "staged"
  fz_peak = track_fz_peak(a.device)
  with GpuMemSampler() if a.device == "CUDA" else _Null() as smp:
    try:
      emb, enc_s = {}, {}
      if staged and not use_ref_embeds:
        stage = "staged-encode"
        emb, enc_s, info["encoder_load_s"] = _staged_embeds(a, root)
      stage = "load"
      t0 = time.time()
      pipe = KleinImagePipeline(root, device=a.device, dtype="bfloat16", load_encoder=not staged)
      info["load_s"] = time.time() - t0
      info["vae_dtype"] = pipe.vae_dtype
      eng = ImageEngine(pipe, max_batch=1, batch_sizes=(1,))
      def params(pi: int, seed: int, size: int, steps: int):
        if use_ref_embeds: return ImageParams(prompt_embeds=ref_embeds(a, pi), seed=seed, width=size, height=size, steps=steps)
        if staged: return ImageParams(prompt_embeds=emb[pi], seed=seed, width=size, height=size, steps=steps)
        return ImageParams(prompt=PROMPTS[pi], seed=seed, width=size, height=size, steps=steps)
      if not use_ref_embeds and a.smoke:   # fail fast: one small image through every kernel kind before the 1024^2 runs
        stage = "smoke"
        t0 = time.time()
        (r,), = eng.generate([params(a.prompt_list[0], a.seeds[0], min(SMOKE, a.size), 1)])
        info["smoke_s"] = time.time() - t0
        print(f"{tag} smoke {min(SMOKE, a.size)}^2 1 step: {info['smoke_s']:.1f} s (first call, compiles included)", flush=True)
      stage, times = "run", []
      for pi in a.prompt_list:
        for seed in a.seeds:
          if times and _out_of_time(a, 3 * times[-1]["first_s"]): print(f"{tag}: out of time, stopping before p{pi} s{seed}", flush=True); break
          p, cold = params(pi, seed, a.size, 4), None
          for rep in range(3 if eng.use_jit else 2):   # jit: capture (2 calls) then the timed replay; eager: a warm-up, then timed
            t0 = time.time()
            (r,), = eng.generate([p])
            dt = time.time() - t0
            cold = dt if cold is None else cold
          tm = dict(r.timings)
          if staged and not use_ref_embeds: tm["encode_s"] = enc_s[pi]; dt += enc_s[pi]
          times.append({"prompt": pi, "seed": seed, "s": dt, "first_s": cold, **tm})
          print(f"{tag} p{pi} s{seed}: {dt:.2f} s warm ({', '.join(f'{k} {v:.2f}' for k, v in tm.items())}), first {cold:.1f} s", flush=True)
          _png(os.path.join(a.out, f"{tag}_p{pi}_s{seed}.png"), r.image)
    except BaseException as e:
      err = {"stage": stage, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(), **info,
             **memory_report(a.device, smp if a.device == "CUDA" else None, fz_peak)}
      json.dump(err, open(os.path.join(a.out, f"{tag}.error.json"), "w"), indent=1)
      print(f"{tag} FAILED at {stage}: {err['error']}", flush=True)
      raise
    mem = memory_report(a.device, smp if a.device == "CUDA" else None, fz_peak)
  json.dump({"s_per_image": times, **info, **mem}, open(os.path.join(a.out, f"{tag}.json"), "w"), indent=1)

PHASES = {"df-bf16": phase_df_bf16, "fz-bf16": lambda a: phase_fz(a, False), "fz-bf16-ref": lambda a: phase_fz(a, True)}

# ---------------------------------------------------------------- metrics

def psnr(a: np.ndarray, b: np.ndarray) -> float:
  mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
  return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

def ssim(a: np.ndarray, b: np.ndarray) -> float:
  """Mean SSIM on the luma (BT.601) with an 11x11 Gaussian window (sigma 1.5), K1 0.01, K2 0.03, as Wang et al. 2004."""
  def luma(x): return x.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
  x, y = luma(a), luma(b)
  g = np.exp(-0.5 * (np.arange(11) - 5) ** 2 / 1.5 ** 2); g /= g.sum()
  def blur(z):
    z = np.apply_along_axis(lambda r: np.convolve(r, g, mode="valid"), 1, z)
    return np.apply_along_axis(lambda c: np.convolve(c, g, mode="valid"), 0, z)
  mx, my = blur(x), blur(y)
  sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
  c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
  return float(np.mean(((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))))

# ---------------------------------------------------------------- driver

def build_report(a, phases: list[str], phase_status: dict[str, str]) -> tuple[dict, bool]:
  from PIL import Image
  load = lambda fn: np.asarray(Image.open(fn).convert("RGB"))
  rows, ok = [], True
  for pi in a.prompt_list:
    for s in a.seeds:
      ref = load(ref_png(a, pi, s))
      r = {"prompt": pi, "seed": s}
      for tag in phases:
        fn = os.path.join(a.out, f"{tag}_p{pi}_s{s}.png")
        if os.path.exists(fn):
          im = load(fn)
          r[f"{tag}.psnr"], r[f"{tag}.ssim"] = psnr(im, ref), ssim(im, ref)
      if "df-bf16" in phases and "fz-bf16" in phases:
        if "df-bf16.psnr" in r and "fz-bf16.psnr" in r:
          r["G6.threshold_db"] = r["df-bf16.psnr"] - 1.0
          r["G6.pass"] = bool(r["fz-bf16.psnr"] >= r["G6.threshold_db"])
          r["fz_closer_than_df"] = bool(r["fz-bf16.psnr"] >= r["df-bf16.psnr"])   # the strict reading, reported only
        else: r["G6.pass"] = None   # an image is missing (failed or out of time): not a pass
        ok &= r["G6.pass"] is True
      rows.append(r)
  vae = next((json.load(open(fn)).get("vae_dtype") for tag in FZ_PHASES if os.path.exists(fn := os.path.join(a.out, f"{tag}.json"))), None)
  rep = {"repo": a.model, "ref": a.ref, "size": a.size, "steps": 4, "device": a.device, "fz_encode": a.fz_encode,
         "fz_vae_dtype": vae or a.vae_dtype,
         "phase_status": phase_status, "rows": rows}
  errors = {}
  for tag in phases:
    fn = os.path.join(a.out, f"{tag}.json")
    if os.path.exists(fn): rep[f"{tag}.timing"] = json.load(open(fn))
    efn = os.path.join(a.out, f"{tag}.error.json")
    if os.path.exists(efn):
      e = json.load(open(efn)); errors[tag] = {"stage": e.get("stage"), "error": e.get("error")}
  if errors: rep["errors"] = errors
  ok &= all(v == "ok" for v in phase_status.values())
  def mean(xs): return float(np.mean(xs)) if xs else None
  summ = {}
  for tag in phases:
    t = rep.get(f"{tag}.timing", {})
    sp = t.get("s_per_image", [])
    secs = [x["s"] if isinstance(x, dict) else x for x in sp]
    mem = {k: t.get(k) for k in ("peak_footprint_gb", "torch_max_allocated_gb", "torch_max_reserved_gb", "fz_alloc_peak_gb",
                                 "gpu_mem_used_peak_gb") if t.get(k) is not None}
    summ[tag] = {"s_per_image_mean": mean(secs), "n": len(secs), **mem,
                 **({k: mean([x[k] for x in sp if k in x]) for k in ("encode_s", "denoise_s", "decode_s")} if sp and isinstance(sp[0], dict) else {}),
                 "psnr_mean": mean([r[f"{tag}.psnr"] for r in rows if f"{tag}.psnr" in r])}
  rep["summary"] = summ
  rep["pass"] = bool(ok)
  return rep, bool(ok)

def main(argv: list[str] | None = None, default_device: str = "CUDA", default_out: str | None = None) -> None:
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--out", default=default_out)
  ap.add_argument("--ref", default="cpu_f32", help="the scripts/diff_ref.py run the images are compared with")
  ap.add_argument("--ref-root", default=OUT, help="the dir holding <ref>/decode and <ref>/embeds (default results/diff)")
  ap.add_argument("--model", default=REPO, help="HF repo or a local diffusers-layout dir")
  ap.add_argument("--size", type=int, default=1024)
  ap.add_argument("--seeds", type=int, nargs="+", default=[0])
  ap.add_argument("--prompts", default="0-3", help="diff_ref PROMPTS indices, e.g. 0-3 or 0,2")
  ap.add_argument("--device", default=default_device, choices=sorted(TORCH_DEVICE))
  ap.add_argument("--phases", default="df-bf16,fz-bf16,fz-bf16-ref")
  ap.add_argument("--fz-encode", default="auto", choices=["auto", "resident", "staged"],
                  help="auto: staged on a CUDA GPU under 40 GB, else resident")
  ap.add_argument("--no-smoke", dest="smoke", action="store_false", help="skip fz-bf16's small fail-fast image")
  ap.add_argument("--deadline", type=float, default=0.0, help="unix time by which every phase must end (0: none)")
  ap.add_argument("--reuse", action="store_true", help="skip phases whose outputs exist")
  ap.add_argument("--phase", default=None, help=argparse.SUPPRESS)
  a = ap.parse_args(argv)
  a.out = a.out or ("results/diff/e2e" if a.device == "METAL" else f"results/diff/{a.device.lower()}")
  a.prompt_list = parse_list(a.prompts)
  if a.fz_encode == "auto":
    tot = cuda_total_gb() if a.device == "CUDA" else None
    a.fz_encode = "staged" if tot is not None and tot < 40 else "resident"
  if not a.phase:   # the VAE dtype FuseZap will pick (KleinImagePipeline: bf16 on a CUDA GPU under 40 GB), logged up front
    tot = cuda_total_gb() if a.device == "CUDA" else None
    env = os.environ.get("FZ_KLEIN_VAE_DTYPE")
    a.vae_dtype = env or ("bfloat16" if tot is not None and tot < 40 else "float32")
    print(f"diff_klein_e2e: {a.device}{f' {tot:.0f} GB' if tot else ''}: fz encode {a.fz_encode}, fz VAE {a.vae_dtype}"
          f" ({'FZ_KLEIN_VAE_DTYPE' if env else 'auto'})", flush=True)
  os.makedirs(a.out, exist_ok=True)
  if a.phase:
    try: PHASES[a.phase](a)
    except BaseException as e:   # phase_fz writes its own (with the stage); anything else gets one here
      efn = os.path.join(a.out, f"{a.phase}.error.json")
      if not os.path.exists(efn):
        json.dump({"stage": "phase", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, open(efn, "w"), indent=1)
      raise
    return
  missing = [ref_png(a, pi, s) for pi in a.prompt_list for s in a.seeds if not os.path.exists(ref_png(a, pi, s))]
  if "fz-bf16-ref" in a.phases:
    missing += [p for pi in a.prompt_list if not os.path.exists(p := os.path.join(a.ref_root, a.ref, "embeds", f"p{pi}.npz"))]
  if missing: sys.exit(f"missing references (run scripts/diff_ref.py encode/denoise/decode --sizes {a.size} first): {missing}")
  phases = a.phases.split(",")
  done = lambda tag: os.path.exists(os.path.join(a.out, f"{tag}_p{a.prompt_list[-1]}_s{a.seeds[-1]}.png")) and os.path.exists(os.path.join(a.out, f"{tag}.json"))
  status: dict[str, str] = {}
  for ph in phases:
    if a.reuse and done(ph): status[ph] = "ok"; continue
    if ph in FZ_PHASES and status.get("fz-bf16", "").startswith("failed at smoke"):
      status[ph] = "skipped (fz-bf16 smoke failed)"; continue
    if _out_of_time(a, 60): status[ph] = "skipped (deadline)"; continue
    for fn in (f"{ph}.json", f"{ph}.error.json"):
      if os.path.exists(p := os.path.join(a.out, fn)): os.remove(p)
    print(f"[{time.strftime('%H:%M:%S')}] {ph} ({a.device}, fz encode {a.fz_encode}) ...", flush=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--phase", ph, "--out", a.out, "--ref", a.ref, "--ref-root", a.ref_root,
           "--model", a.model, "--size", str(a.size), "--prompts", a.prompts, "--device", a.device, "--fz-encode", a.fz_encode,
           "--deadline", str(a.deadline), "--seeds", *map(str, a.seeds)] + ([] if a.smoke else ["--no-smoke"])
    try:
      rc = subprocess.run(cmd, timeout=(a.deadline - time.time()) if a.deadline else None).returncode
      if rc == 0: status[ph] = "ok"
      else:
        efn = os.path.join(a.out, f"{ph}.error.json")
        stage = json.load(open(efn)).get("stage") if os.path.exists(efn) else "?"
        status[ph] = f"failed at {stage} (exit {rc})"
    except subprocess.TimeoutExpired: status[ph] = "killed (deadline)"
    print(f"[{time.strftime('%H:%M:%S')}] {ph}: {status[ph]}", flush=True)
  rep, ok = build_report(a, phases, status)
  json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1)
  print(json.dumps({"rows": rep["rows"], "summary": rep["summary"], "phase_status": status, "errors": rep.get("errors"), "pass": ok}, indent=1))
  sys.exit(0 if ok else 1)

if __name__ == "__main__": main()
