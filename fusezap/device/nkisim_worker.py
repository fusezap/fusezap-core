"""NKISIM worker: runs inside .venv-nki (nki 0.6.0, Linux x86_64), never in fusezap's own env (docs/NKI_BACKEND.md §4
exception 5: the nki wheel is linux_x86_64-only and conflicts with nrtpy, so the compiler and simulator are a subprocess).

Started by fusezap.device.neuron as `<.venv-nki python> nkisim_worker.py`. It imports only numpy, ml_dtypes and nki.
Protocol: pickle 5 frames with out-of-band array buffers on the original stdout/stdin (`_send`/`_recv`; fusezap.device.neuron has the same pair) (the kernel's own prints go to stderr). Requests:
  {"op": "version"}                                       -> {"ok", "nki", "neuronxcc"}
  {"op": "sim", "src", "name", "arrays", "dtypes", "outs"} -> {"ok", "outs": {i: array}, "seconds"}   (nki.simulate)
  {"op": "neff", "src", "name", "arrays", "dtypes", "target", "dir"} -> {"ok", "neff": bytes, "seconds", "io", "how"}
      (neuronx-cc, compile only: no device is touched. N4 run 1 fix: every compile gets a fresh unique artifacts dir and
      NKI_ARTIFACTS_DIR is never left set. The run-1 worker set it once for a NEFF, so every later nki.jit compile reused that
      dir and failed with neuronx-cc NCC_IDRV077 "Compilation artifacts already exist". `dir`, if given, gets a copy of kernel.neff.)
  {"op": "hw", "src", "name", "arrays", "dtypes", "outs"} -> {"ok", "outs": {i: array}, "seconds", "out_mode"}
      (N4 fallback, FZ_NEURON_RT=jit: the nki.jit kernel runs on the NeuronCore. Run 1: correct, outputs returned
      (out_mode "returned"), but ~1.31 s per call at any size: nki 0.6's standalone path compiles with enable_cache=False and
      loads the NEFF again on every call (nki/framework/compiled.py). The persistent path is nrt_worker.py.)
`arrays` are flat host arrays; `dtypes[i]` is the nl dtype name of buffer i ("bfloat16" buffers travel as uint16 bits).
Errors come back as {"ok": False, "err": "<type>: <message>", "tb": "<traceback>"}; the worker keeps serving.
"""
import contextlib, hashlib, importlib.util, os, pickle, shutil, struct, sys, tempfile, time, traceback

def _send(f, obj) -> None:
  """Pickle 5 with out-of-band buffers: header (meta length, buffer count, buffer lengths), meta, then each array's memory
  written to the pipe as is, so a big array is never copied into an in-band pickle (a 4 GB pod holds 0.5 GB weights)."""
  bufs: list = []
  meta = pickle.dumps(obj, protocol=5, buffer_callback=bufs.append)
  raws = [b.raw() for b in bufs]
  f.write(struct.pack("<QI", len(meta), len(raws)) + b"".join(struct.pack("<Q", r.nbytes) for r in raws) + meta)
  for r in raws: f.write(r)
  f.flush()

def _recv(f):
  h = f.read(12)
  if len(h) < 12: return None
  n, k = struct.unpack("<QI", h)
  lens = struct.unpack(f"<{k}Q", f.read(8 * k)) if k else ()
  meta, bufs = f.read(n), []
  for m in lens:
    b = bytearray(m)
    mv, got = memoryview(b), 0
    while got < m:
      if not (r := f.readinto(mv[got:])): raise EOFError
      got += r
    bufs.append(b)
  return pickle.loads(meta, buffers=bufs)

_MODS: dict = {}
_DIR = tempfile.mkdtemp(prefix="fz_nkisim_")

def _kernel(src: str, name: str):
  """Import the kernel source from a real file (nki.jit reads the function's source), cached by source hash."""
  key = hashlib.sha256(src.encode()).hexdigest()[:20]
  if (m := _MODS.get(key)) is None:
    path = os.path.join(_DIR, f"k_{key}.py")
    with open(path, "w") as f: f.write(src)
    spec = importlib.util.spec_from_file_location(f"fz_nki_{key}", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _MODS[key] = m
  return getattr(m, name)

def _inputs(arrays, dtypes):
  """Host arrays for nki; an (elements, numpy dtype) pair is a zero array allocated here (untouched pages cost no memory:
  a NEFF compile needs only the shapes, so a full-size weight need not travel through the pipe)."""
  import ml_dtypes, numpy as np
  xs = [np.zeros(a[0], a[1]) if isinstance(a, tuple) else np.ascontiguousarray(a) for a in arrays]
  return [a.view(ml_dtypes.bfloat16) if dt == "bfloat16" else a for a, dt in zip(xs, dtypes)]

def _host(a, dt):
  import numpy as np
  return np.asarray(a).view(np.uint16) if dt == "bfloat16" else np.asarray(a)

def _fresh_dir() -> str:
  """A new, empty, unique artifacts directory for one compile (neuronx-cc refuses a dir with old artifacts: NCC_IDRV077)."""
  base = os.path.join(_DIR, "neff")
  os.makedirs(base, exist_ok=True)
  return tempfile.mkdtemp(prefix="c_", dir=base)

@contextlib.contextmanager
def _artifacts_env(d: str):
  """NKI_ARTIFACTS_DIR=d for one call, then the previous value (run 1 left it set, so later compiles reused one dir)."""
  old = os.environ.get("NKI_ARTIFACTS_DIR")
  os.environ["NKI_ARTIFACTS_DIR"] = d
  try: yield
  finally:
    if old is None: os.environ.pop("NKI_ARTIFACTS_DIR", None)
    else: os.environ["NKI_ARTIFACTS_DIR"] = old

def _compile_neff(k, xs, work: str):
  """Compile kernel k for arrays xs into the empty dir `work` without running it: (neff path or None, io, how, error).
  "compile_only": nki 0.6.0's StandaloneKernel with an executor that only records the CompiledKernel (internals read from the
  0.6.0 wheel: framework/compiled.py `_executor`, compiler/driver.py run_from_bir), so no Neuron runtime starts in this
  process and the NeuronCore stays free for the persistent runtime worker. io holds the NEFF's input names, output names and
  aliases. "jit_fallback": if those internals are missing, nki.jit as before (compiles, then runs or fails without libnrt)."""
  io: dict = {}
  try:
    from nki.compiler.frontend import resolve_frontend_cls
    from nki.framework.compiled import StandaloneKernel
  except Exception:
    StandaloneKernel = None
  err, how = None, "compile_only"
  if StandaloneKernel is not None and hasattr(k, "_to_subclass"):
    def record(compiled, inputs, outputs, **kw):
      io.update(neff=str(compiled.neff_path), inputs=[s.name for s in compiled.input_specs],
                outputs=[s.name for s in compiled.output_specs],
                aliases={str(o): str(i) for o, i in (compiled.input_output_aliases or {}).items()})
    sk = k._to_subclass(StandaloneKernel, _frontend_cls=resolve_frontend_cls(), _executor=record, artifacts_dir=work)
    with _artifacts_env(work):
      try: sk(*xs)
      except Exception as e: err = f"{type(e).__name__}: {str(e)[:2000]}"
  else:
    how = "jit_fallback"
    with _artifacts_env(work):
      try: k(*xs)
      except Exception as e: err = f"{type(e).__name__}: {str(e)[:2000]}"
  for p in (os.path.join(work, "kernel.neff"), io.get("neff", "")):
    if p and os.path.isabs(p) and os.path.exists(p): return p, io, how, err
  return None, io, how, err

def handle(req: dict) -> dict:
  import nki, numpy as np
  if req["op"] == "version":
    try:
      import neuronxcc; cc = getattr(neuronxcc, "__version__", "installed")
    except ImportError: cc = None
    from importlib.metadata import version
    return {"ok": True, "nki": version("nki"), "neuronxcc": cc}
  k = _kernel(req["src"], req["name"])
  xs = _inputs(req["arrays"], req["dtypes"])
  if req["op"] == "sim":
    st = time.perf_counter()
    nki.simulate(k)(*xs)                 # verified on 0.6.0: outputs are written into the argument arrays in place
    et = time.perf_counter() - st
    return {"ok": True, "outs": {i: _host(xs[i], req["dtypes"][i]) for i in req["outs"]}, "seconds": et}
  if req["op"] == "neff":
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = req.get("target", "inf2")
    work = _fresh_dir()
    try:
      st = time.perf_counter()
      neff_path, io, how, err = _compile_neff(k, xs, work)
      et = time.perf_counter() - st
      if neff_path is None: return {"ok": False, "err": f"no kernel.neff ({err})", "tb": "", "seconds": et}
      with open(neff_path, "rb") as f: neff = f.read()
      if req.get("dir"):   # a copy for the caller (the N4 probe's NEFF cases); never compiled into
        os.makedirs(req["dir"], exist_ok=True)
        shutil.copyfile(neff_path, os.path.join(req["dir"], "kernel.neff"))
      return {"ok": True, "neff": neff, "seconds": et, "io": io, "how": how, "run_err": err}
    finally: shutil.rmtree(work, ignore_errors=True)
  if req["op"] == "hw":
    work = _fresh_dir()
    try:
      with _artifacts_env(work):   # a fresh dir per call, never one an earlier compile left behind (NCC_IDRV077)
        st = time.perf_counter()
        ret = k(*xs)
        et = time.perf_counter() - st
    finally: shutil.rmtree(work, ignore_errors=True)
    outs, mode = {i: _host(xs[i], req["dtypes"][i]) for i in req["outs"]}, "in_place"
    if ret is not None:
      rs = list(ret) if isinstance(ret, (tuple, list)) else [ret]
      if len(rs) == len(req["outs"]):
        mode = "returned"
        for i, r in zip(req["outs"], rs): outs[i] = _host(np.asarray(r).reshape(-1), req["dtypes"][i])
    return {"ok": True, "outs": outs, "seconds": et, "out_mode": mode}
  return {"ok": False, "err": f"unknown op {req['op']!r}", "tb": ""}

def main() -> None:
  # nki runs neuronx-cc from PATH; the venv may be used without being activated
  os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
  os.environ.pop("NKI_ARTIFACTS_DIR", None)   # every compile names its own fresh dir
  fin, fout = os.fdopen(os.dup(0), "rb"), os.fdopen(os.dup(1), "wb")
  os.dup2(2, 1); sys.stdout = sys.stderr   # anything the kernel or nki prints must not corrupt the protocol
  while (req := _recv(fin)) is not None:
    try: rep = handle(req)
    except BaseException as e:
      if isinstance(e, KeyboardInterrupt): raise
      rep = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:4000]}", "tb": traceback.format_exc()[-6000:]}
    _send(fout, rep)

if __name__ == "__main__": main()
