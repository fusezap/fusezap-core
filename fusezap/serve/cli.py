"""`fusezap serve <hf-model-id> [--device D] [--dtype auto|float32|float16|bfloat16] [--quant int8|int4|fp8] [--port N]`
(also `python -m fusezap serve ...`). Loads the model, builds an LLMEngine, warms up (captures the jitted forwards) and serves
the OpenAI-compatible API (fusezap/serve/api.py) with uvicorn. On CUDA and NEURON (--engine-process auto) the engine is
built, warmed up and stepped in a child process, and this process runs only the HTTP server."""
from __future__ import annotations
import argparse, os, sys, time

def build_parser() -> argparse.ArgumentParser:
  ap = argparse.ArgumentParser(prog="fusezap", description="FuseZap: compile-first AI inference engine")
  sub = ap.add_subparsers(dest="cmd", required=True)
  s = sub.add_parser("serve", help="serve a model over an OpenAI-compatible HTTP API")
  s.add_argument("model", help="Hugging Face model id or local path")
  s.add_argument("--device", default=None, help="METAL, CUDA, CPU, ... (default: the best available)")
  s.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                 help="auto (default): bfloat16 on CUDA, float32 elsewhere")
  s.add_argument("--quant", default=None, choices=["int8", "int4", "fp8"], help="weight-only quantization")
  s.add_argument("--host", default="127.0.0.1")
  s.add_argument("--port", type=int, default=8000)
  s.add_argument("--served-model-name", default=None, help="model id reported by the API (default: the model argument)")
  s.add_argument("--api-key", default=None, help="require this bearer token")
  s.add_argument("--max-model-len", type=int, default=2048)
  s.add_argument("--max-num-seqs", type=int, default=32, help="max concurrent sequences (decode batch rows)")
  s.add_argument("--max-prefill-tokens", type=int, default=None, help="prompt tokens per step (default on CUDA: 2048, or less if the free GPU memory cannot hold that forward; 256 elsewhere)")
  s.add_argument("--num-blocks", type=int, default=1024, help="KV cache blocks (shared by all requests)")
  s.add_argument("--block-size", type=int, default=16)
  s.add_argument("--beam", type=int, default=None, help="BEAM width for the decode kernels (default 4)")
  s.add_argument("--no-warmup", action="store_true")
  s.add_argument("--log-level", default="info")
  s.add_argument("--mode", default=None, choices=["exact", "throughput"],
                 help="exact (default; FZ_MODE): a request's tokens are bit-identical alone or batched. throughput (opt-in, CUDA "
                      "sm_90): decode rows ride inside prefill forwards like vLLM's mixed batches; faster when prompts and decodes "
                      "overlap, near-tie tokens may depend on the batch. A request served alone gets the exact tokens either way")
  s.add_argument("--tensor-parallel-size", "--tp", "-tp", type=int, default=1, dest="tensor_parallel_size",
                 help="shard the model over N devices (one process per device; rank r on CUDA:r, or on --device's ordinal + r); "
                      "N must divide the attention heads, the KV heads and the intermediate size. 1 (default): one device")
  s.add_argument("--engine-process", default="auto", choices=["auto", "on", "off"],
                 help="run the engine in its own process, apart from the HTTP server (auto: on for CUDA and NEURON, i.e. --device / "
                      "FZ_DEVICE names one of them, or neither is set and the CUDA driver is installed; off elsewhere, as before)")
  from fusezap.serve.image_cli import add_parser as _image_parser   # I81 D7: `fusezap serve-image`
  _image_parser(sub)
  return ap

def resolve_dtype(name: str, device: str | None) -> str:
  if name != "auto": return name
  from fusezap.device import Device
  return "bfloat16" if Device.canonicalize(device).split(":")[0].upper() == "CUDA" else "float32"

def load_timeout(model: str, world: int) -> float:
  """Seconds rank 0 waits for the workers' weight load: dist.TIMEOUT (FZ_TP_TIMEOUT, 600 s), or a rank's share of the
  checkpoint at 100 MB/s when that is longer (70B on 2 ranks: 660 s; M2 loaded it in 57 s from the page cache)."""
  from fusezap.serve import dist
  from fusezap.serve.weights import resolve_model, checkpoint_bytes
  try: nb = checkpoint_bytes(resolve_model(model))
  except Exception: nb = 0
  return max(dist.TIMEOUT, nb / max(1, world) / 100e6)

def build_engine(a: argparse.Namespace, tokenizer: bool = True):
  from fusezap.dtype import dtypes
  a.dtype = resolve_dtype(a.dtype, a.device)
  from fusezap.serve.models import load_model, LlamaConfig
  from fusezap.serve.engine import LLMEngine
  from fusezap.serve.tokenizer import Tokenizer
  W = getattr(a, "tensor_parallel_size", 1) or 1
  g = None
  if W > 1:   # tensor parallelism: this process is rank 0; the workers load their shards while it loads its own
    from fusezap.serve import dist
    from fusezap.serve.weights import resolve_model, load_config
    if a.quant: raise ValueError("--quant is not supported with --tensor-parallel-size > 1 yet")
    dist.check_tp(LlamaConfig.from_hf(load_config(resolve_model(a.model))), W)
    g = dist.launch(W, a.device)
    g.post("load", {"model": a.model, "dtype": a.dtype, "max_len": a.max_model_len, "prefill_chunk": 128})
  m = load_model(a.model, dtype=getattr(dtypes, a.dtype), device=g.device if g else a.device, max_len=a.max_model_len, prefill_chunk=128,
                 **({"tp": g} if g else {}))
  if a.quant:
    from fusezap.serve.quant import quantize_model
    quantize_model(m, a.quant)
  if g is not None: g.barrier("load", load_timeout(a.model, W))
  eng = LLMEngine(m, num_blocks=a.num_blocks, block_size=a.block_size, max_num_seqs=a.max_num_seqs, max_prefill_tokens=a.max_prefill_tokens,
                  max_model_len=a.max_model_len, decode_beam=a.beam, mode=getattr(a, "mode", None))
  from fusezap.serve.weights import resolve_model
  eng.tokenizer_path = os.path.join(resolve_model(a.model), "tokenizer.json")   # constrained decoding's tokenizer (warmup prebuilds it)
  if g is not None:
    g.post("engine", eng.tp_spec(), eng.tp_signature())
    g.barrier("engine")
    print(f"fusezap: tensor parallel {W} ({g.backend}): ranks on {', '.join(dist.rank_device(g.device, r) for r in range(W))}; "
          f"{g.ar_info}; control {'shared-memory ring' if g.ring is not None else 'socket'}", flush=True)
  return (eng, Tokenizer.from_pretrained(a.model)) if tokenizer else eng

def engine_process(a: argparse.Namespace) -> bool:
  """--engine-process auto: a separate engine process on CUDA and NEURON (N7 serve: the event loop's per-token work no
  longer shares a GIL with the engine's step between two device round trips). Decided without opening a device, so the
  server process never creates a CUDA context (the engine process owns the GPU)."""
  if a.engine_process != "auto": return a.engine_process == "on"
  name = (a.device or os.environ.get("FZ_DEVICE") or "").split(":")[0].upper()
  if name: return name in ("CUDA", "NEURON")
  import ctypes.util
  return sys.platform != "darwin" and ctypes.util.find_library("cuda") is not None   # the default device would be CUDA

def serve(a: argparse.Namespace) -> int:
  import uvicorn
  from fusezap.serve.api import create_app, warmup, ProcEngine
  if engine_process(a):   # the engine process loads, warms up and prints as below
    from fusezap.serve.tokenizer import Tokenizer
    box: list = []   # the engine process died: stop serving and exit 1, so a supervisor restarts the server
    eng = ProcEngine(a, on_dead=lambda: [setattr(s, "should_exit", True) for s in box])
    eng.start()
    server = uvicorn.Server(uvicorn.Config(create_app(eng, Tokenizer.from_pretrained(a.model), a.served_model_name or a.model, a.api_key),
                                           host=a.host, port=a.port, log_level=a.log_level))
    box.append(server)
    if eng.dead: server.should_exit = True   # died before box held the server
    server.run()
    return 1 if eng.dead else 0
  t0 = time.perf_counter()
  eng, tok = build_engine(a)
  print(f"fusezap: loaded {a.model} on {eng.model.device} ({a.dtype}{', ' + a.quant if a.quant else ''}) in {time.perf_counter() - t0:.1f}s", flush=True)
  if not a.no_warmup:
    t0 = time.perf_counter(); warmup(eng)
    print(f"fusezap: warm-up {time.perf_counter() - t0:.1f}s (first run on a cold ~/.cache/fusezap also tunes kernels: minutes)", flush=True)
  from fusezap.serve.api import _print_memory
  _print_memory(eng)
  try: uvicorn.run(create_app(eng, tok, a.served_model_name or a.model, a.api_key), host=a.host, port=a.port, log_level=a.log_level)
  finally:
    from fusezap.serve.api import close_tp
    close_tp(eng)
  return 0

def main(argv: list[str] | None = None) -> int:
  a = build_parser().parse_args(argv)
  if a.cmd == "serve-image":
    from fusezap.serve.image_cli import serve_image
    return serve_image(a)
  return serve(a) if a.cmd == "serve" else 2

if __name__ == "__main__": sys.exit(main())
