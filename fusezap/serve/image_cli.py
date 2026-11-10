"""`fusezap serve-image <pipeline> ...` (I81 D7): argparse and the server entry for fusezap/serve/image_api.py. Kept apart
from image_api so `fusezap --help` does not import starlette."""
from __future__ import annotations
import time
from typing import Any

def add_parser(sub: Any) -> None:
  s = sub.add_parser("serve-image", help="serve a diffusion image model over an OpenAI-compatible /v1/images/generations")
  s.add_argument("pipeline", help="a registered image pipeline name or module:callable")
  s.add_argument("--device", default=None)
  s.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
  s.add_argument("--host", default="127.0.0.1")
  s.add_argument("--port", type=int, default=8000)
  s.add_argument("--served-model-name", default=None)
  s.add_argument("--api-key", default=None)
  s.add_argument("--max-batch", type=int, default=4, help="images per denoising step and size bucket")
  s.add_argument("--batch-sizes", default=None, help="comma-separated padded batch sizes to capture (default 1,2,4 up to --max-batch)")
  s.add_argument("--warmup-sizes", default="", help="comma-separated WxH buckets to capture before serving (default: the model's default size)")
  s.add_argument("--no-warmup", action="store_true")
  s.add_argument("--log-level", default="info")

def serve_image(a: Any) -> int:
  import uvicorn
  from fusezap.device import Device
  from fusezap.serve.image_api import load_pipeline, create_image_app
  from fusezap.serve.image_engine import ImageEngine
  t0 = time.perf_counter()
  pipe = load_pipeline(a.pipeline, a.device or Device.DEFAULT, a.dtype)
  bs = [int(x) for x in a.batch_sizes.split(",")] if a.batch_sizes else None
  eng = ImageEngine(pipe, max_batch=a.max_batch, batch_sizes=bs)
  print(f"fusezap: loaded image pipeline {eng.name} on {eng.device} in {time.perf_counter() - t0:.1f}s", flush=True)
  if not a.no_warmup:
    t0 = time.perf_counter()
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.warmup_sizes.split(",") if s] or None
    eng.warmup(sizes)
    print(f"fusezap: warm-up {time.perf_counter() - t0:.1f}s", flush=True)
  uvicorn.run(create_image_app(eng, a.served_model_name, a.api_key), host=a.host, port=a.port, log_level=a.log_level)
  return 0
