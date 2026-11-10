"""Probe the installed nki 0.6.0 (run with .venv-nki/bin/python on a Linux x86_64 pod; docs/NKI_BACKEND.md N2).

Answers the design doc's [unverified] questions from the real package: API surface, simulator entry point and its
I/O semantics, integer divide/mod support, bf16 arguments, and whether `nki.jit` compiles to a NEFF without a device.
Prints a report; nothing here is imported by fusezap.
"""
import importlib, inspect, os, subprocess, sys, time, traceback
import numpy as np

def sec(t): print(f"\n===== {t}", flush=True)
def tryrun(label, fn):
  try:
    r = fn(); print(f"[ok] {label}: {r}", flush=True); return r
  except Exception as e:
    print(f"[err] {label}: {type(e).__name__}: {str(e)[:600]}", flush=True)

import nki, nki.language as nl, nki.isa as nisa
sec("package")
print("nki", getattr(nki, "__version__", "?"), nki.__file__)
print(subprocess.run([sys.executable, "-m", "pip", "show", "nki"], capture_output=True, text=True).stdout)
root = os.path.dirname(nki.__file__)
for d, _, fs in os.walk(root):
  depth = d[len(root):].count(os.sep)
  if depth <= 1: print(" ", d[len(root):] or "/", sorted(f for f in fs if not f.endswith(".pyc"))[:40])
print("nki attrs:", [n for n in dir(nki) if not n.startswith("_")])
print("nl attrs:", [n for n in dir(nl) if not n.startswith("_")])
print("nisa attrs:", [n for n in dir(nisa) if not n.startswith("_")])
for n in ("simulate", "simulate_kernel", "jit", "baremetal", "benchmark", "profile"):
  if hasattr(nki, n):
    f = getattr(nki, n)
    print(f"--- nki.{n}{tryrun('sig', lambda: str(inspect.signature(f))) or ''}\n{(inspect.getdoc(f) or '')[:1500]}")
for n in ("tensor_tensor", "tensor_scalar", "tensor_copy", "iota", "activation"):
  if hasattr(nisa, n): print(f"--- nisa.{n}{tryrun('sig', lambda: str(inspect.signature(getattr(nisa, n))))}\n{(inspect.getdoc(getattr(nisa, n)) or '')[:2500]}")
for n in ("divide", "floor_divide", "mod", "fmod", "remainder", "power", "bitwise_and", "right_shift", "left_shift", "floor", "abs"):
  print(f"nl.{n}:", hasattr(nl, n))
sec("grep source for env vars / compiler")
out = subprocess.run(["grep", "-rIl", "-e", "NEURON_PLATFORM_TARGET_OVERRIDE", "-e", "neuronx-cc", "-e", "neuronxcc", root], capture_output=True, text=True).stdout
print(out[:3000])
out = subprocess.run(["grep", "-rIn", "-e", "NEURON_PLATFORM_TARGET_OVERRIDE", "-e", "NKI_ARTIFACTS_DIR", "-e", "neuronx-cc", root], capture_output=True, text=True).stdout
print(out[:4000])
print("so files:", subprocess.run(["bash", "-c", f"find {root} -name '*.so*' | head -40"], capture_output=True, text=True).stdout)
print("neuronx-cc on PATH:", subprocess.run(["bash", "-c", "command -v neuronx-cc || echo none"], capture_output=True, text=True).stdout)

sec("simulate: in-place output, returned value")
@nki.jit
def k_add(a, b, c):
  t = nl.ndarray((4, 6), dtype=nl.float32, buffer=nl.sbuf)
  u = nl.ndarray((4, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=a)
  nisa.dma_copy(dst=u, src=b)
  nisa.tensor_tensor(dst=t, data1=t, data2=u, op=nl.add)
  nisa.dma_copy(dst=c, src=t)
  return c
a, b = np.arange(24, dtype=np.float32).reshape(4, 6), np.ones((4, 6), np.float32)
c = np.zeros((4, 6), np.float32)
sim = getattr(nki, "simulate", None) or getattr(nki, "simulate_kernel")
r = tryrun("simulate(k)(a,b,c)", lambda: sim(k_add)(a, b, c))
print("c in place:", c.ravel()[:6], "returned type", type(r), None if r is None else np.asarray(r).ravel()[:6])
if hasattr(nki, "simulate_kernel"): tryrun("simulate_kernel(k, a, b, c)", lambda: nki.simulate_kernel(k_add, a, b, c))

sec("integer divide / mod on int32 tiles")
def int_kernel(op, engine=None):
  @nki.jit
  def k(a, b, c):
    t = nl.ndarray((4, 6), dtype=nl.int32, buffer=nl.sbuf)
    u = nl.ndarray((4, 6), dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    nisa.dma_copy(dst=u, src=b)
    kw = {} if engine is None else {"engine": engine}
    nisa.tensor_tensor(dst=t, data1=t, data2=u, op=op, **kw)
    nisa.dma_copy(dst=c, src=t)
    return c
  return k
ia = np.array([[7, -7, 7, -7, 2**24 + 1, -(2**24 + 3)]] * 4, np.int32)
ib = np.array([[2, 2, -2, -2, 3, 5]] * 4, np.int32)
for opn in ("divide", "floor_divide", "mod", "fmod", "remainder"):
  if not hasattr(nl, opn): continue
  for eng in (None, "gpsimd", "vector"):
    e = None if eng is None else getattr(nisa, f"{eng}_engine", None) or getattr(getattr(nisa, "engine", object), eng, None)
    if eng is not None and e is None: print(f"no engine {eng}"); continue
    ic = np.zeros_like(ia)
    tryrun(f"int32 {opn} engine={eng}", lambda: (sim(int_kernel(getattr(nl, opn), e))(ia, ib, ic), ic[0].tolist())[1])
print("python //:", (ia[0] // ib[0]).tolist(), "np.fmod:", np.fmod(ia[0], ib[0]).tolist(), "%:", (ia[0] % ib[0]).tolist())

sec("tensor_scalar int ops")
@nki.jit
def k_ts(a, c):
  t = nl.ndarray((4, 6), dtype=nl.int32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=a)
  nisa.tensor_scalar(dst=t, data=t, op0=nl.add, operand0=1)
  nisa.dma_copy(dst=c, src=t)
  return c
ic = np.zeros_like(ia); tryrun("tensor_scalar int add 1", lambda: (sim(k_ts)(ia, ic), ic[0].tolist())[1])

sec("bf16 arguments")
tryrun("import ml_dtypes", lambda: importlib.import_module("ml_dtypes").__version__)
try:
  import ml_dtypes
  @nki.jit
  def k_bf(a, c):
    t = nl.ndarray((4, 6), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=t, src=a)
    nisa.tensor_scalar(dst=t, data=t, op0=nl.multiply, operand0=3.0)
    nisa.dma_copy(dst=c, src=t)
    return c
  ab = (np.arange(24, dtype=np.float32) / 7).reshape(4, 6).astype(ml_dtypes.bfloat16)
  cb = np.zeros_like(ab)
  tryrun("bf16 sim", lambda: (sim(k_bf)(ab, cb), cb[0].astype(np.float32).tolist())[1])
  ub = ab.view(np.uint16); cu = np.zeros_like(ub)
  tryrun("uint16-as-bf16 arg (expect dtype mismatch?)", lambda: (sim(k_bf)(ub, cu), cu[0].tolist())[1])
except Exception: traceback.print_exc()

sec("nki.jit compile to NEFF without a device")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "inf2")
art = "/tmp/nki_art"; os.makedirs(art, exist_ok=True); os.environ["NKI_ARTIFACTS_DIR"] = art
c = np.zeros((4, 6), np.float32)
st = time.time(); r = tryrun("k_add(a, b, c) direct call", lambda: k_add(a, b, c)); print(f"took {time.time() - st:.2f}s")
print(subprocess.run(["bash", "-c", f"ls -laR {art} | head -40"], capture_output=True, text=True).stdout)
for n in ("compile", "trace", "jit"):
  print(n, "->", [x for x in dir(getattr(nki, n, None)) if not x.startswith("__")][:30] if hasattr(nki, n) else None)
print("PROBE DONE")
