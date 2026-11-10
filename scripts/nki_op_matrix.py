"""Which nki 0.6 ops the simulator AND neuronx-cc 2.27 accept, per engine and dtype (NKI-N2; docs/NKI_BACKEND.md §2).

Run with .venv-nki/bin/python on a Linux x86_64 host after `scripts/nki_env.sh --cc`:  .venv-nki/bin/python
scripts/nki_op_matrix.py [inf2|trn1]. Each case is a (4, 8) kernel written to a real file (the NKI parser reads the
source; it rejects `**kwargs`), simulated against exact numpy (int results wrap to int32) and compiled to a NEFF.
Prints a markdown table.
"""
import importlib.util, os, sys, tempfile, time
import numpy as np
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = sys.argv[1] if len(sys.argv) > 1 else "inf2"
import nki

TT = """import nki
import nki.language as nl
import nki.isa as nisa
@nki.jit
def k(a, b, c):
  t = nl.ndarray((4, 8), dtype=nl.{dt}, buffer=nl.sbuf)
  u = nl.ndarray((4, 8), dtype=nl.{dt}, buffer=nl.sbuf)
  o = nl.ndarray((4, 8), dtype=nl.{odt}, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=a)
  nisa.dma_copy(dst=u, src=b)
  {instr}
  nisa.dma_copy(dst=c, src=o)
  return c
"""

def load(src: str):
  d = tempfile.mkdtemp(prefix="opm_src_")
  p = os.path.join(d, "kern.py")
  with open(p, "w") as f: f.write(src)
  spec = importlib.util.spec_from_file_location(f"opm_{abs(hash(src))}", p)
  m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
  return m.k

r = np.random.default_rng(0)
A = {"int32": np.concatenate([np.array([2**24 + 1, -(2**24 + 1), 7, -7, 2**30, -2**30, 5, 0]), r.integers(-1000, 1000, 24)]).astype(np.int32).reshape(4, 8),
     "float32": r.standard_normal((4, 8)).astype(np.float32) * 10}
B = {"int32": np.concatenate([np.array([2**24, -(2**24), 2, 2, 3, 7, -2, 3]), r.integers(1, 50, 24)]).astype(np.int32).reshape(4, 8),
     "float32": r.standard_normal((4, 8)).astype(np.float32) + 3}
NP = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "maximum": np.maximum, "minimum": np.minimum,
      "fmod": np.fmod, "mod": np.mod, "equal": np.equal, "not_equal": np.not_equal, "less": np.less, "greater_equal": np.greater_equal,
      "bitwise_and": np.bitwise_and, "right_shift": np.right_shift, "left_shift": np.left_shift}
CMP = ("equal", "not_equal", "less", "greater_equal")

def exact(opn, a, b):
  with np.errstate(all="ignore"):
    if opn == "divide": return np.trunc(a / b) if a.dtype.kind == "i" else a / b
    if a.dtype.kind == "i":
      bb = b.astype(np.int64) & 31 if "shift" in opn else b.astype(np.int64)
      return NP[opn](a.astype(np.int64), bb).astype(np.int32)   # integer results wrap
    return NP[opn](a, b)

def run(src, a, b, cdt, ref):
  c = np.zeros((4, 8), cdt)
  try:
    nki.simulate(load(src))(a, b, c)
    if cdt != np.float32: sim = "exact" if np.array_equal(c.astype(np.int64), np.asarray(ref).astype(np.int64)) else f"WRONG ({int((c.astype(np.int64) != np.asarray(ref).astype(np.int64)).sum())}/32)"
    else: sim = "ok" if np.allclose(c, ref, rtol=1e-5, atol=1e-5, equal_nan=True) else "WRONG"
  except Exception as e: sim = f"sim error: {type(e).__name__}: {str(e).strip().splitlines()[-1][:80]}"
  d = tempfile.mkdtemp(prefix="opm_"); os.environ["NKI_ARTIFACTS_DIR"] = d
  st = time.time()
  try: load(src)(a, b, np.zeros((4, 8), cdt)); cc = "ran?!"
  except Exception as e:
    msg = str(e)
    if os.path.exists(os.path.join(d, "kernel.neff")): cc = "NEFF ok"
    else: cc = "REJECTED: " + (msg.split("diagnostics:")[-1].strip().splitlines()[0] if "diagnostics:" in msg else msg.strip().splitlines()[0])[:100]
  return sim, cc, time.time() - st

print(f"target {os.environ['NEURON_PLATFORM_TARGET_OVERRIDE']}, nki {nki.__version__ if hasattr(nki, '__version__') else '0.6'}")
print("| instruction | op | dtype | engine | simulate (vs exact) | neuronx-cc | s |\n|---|---|---|---|---|---|---|")
for opn in ["add", "subtract", "multiply", "maximum", "minimum", "divide", "fmod", "mod", "equal", "not_equal", "less",
            "greater_equal", "bitwise_and", "right_shift", "left_shift"]:
  for dtn in ("int32", "float32"):
    if dtn == "float32" and ("shift" in opn or "bitwise" in opn): continue
    for eng in ("unknown", "vector", "gpsimd"):
      a, b = A[dtn], B[dtn]
      cmp = opn in CMP
      kw = "" if eng == "unknown" else f", engine=nisa.{eng}_engine"
      src = TT.format(dt=dtn, odt="uint8" if cmp else dtn, instr=f"nisa.tensor_tensor(dst=o, data1=t, data2=u, op=nl.{opn}{kw})")
      sim, cc, s = run(src, a, b, np.uint8 if cmp else a.dtype, exact(opn, a, b))
      print(f"| tensor_tensor | {opn} | {dtn} | {eng} | {sim} | {cc} | {s:.1f} |", flush=True)
for opn, operand in [("add", 1.0), ("multiply", 3.0), ("divide", 3.0), ("fmod", 1.0), ("mod", 1.0), ("maximum", 0.0), ("equal", 0.0),
                     ("not_equal", 0.0), ("subtract", 2.0)]:
  for dtn in ("float32", "int32"):
    a = A[dtn]
    cmp = opn in CMP
    src = TT.format(dt=dtn, odt="uint8" if cmp else dtn, instr=f"nisa.tensor_scalar(dst=o, data=t, op0=nl.{opn}, operand0={operand})")
    with np.errstate(all="ignore"):
      f = {"divide": np.divide, **NP}[opn](a.astype(np.float32), np.float32(operand))
    ref = f if cmp or dtn == "float32" else np.clip(np.rint(f), -2**31, 2**31 - 1)
    sim, cc, s = run(src, a, B[dtn], np.uint8 if cmp else a.dtype, ref)
    print(f"| tensor_scalar | {opn} {operand} | {dtn} | unknown | {sim} | {cc} | {s:.1f} |", flush=True)
for name, instr, cdt in [("tensor_copy f32->i32", "nisa.tensor_copy(dst=o, src=t)", "int32"),
                         ("activation sin", "nisa.activation(dst=o, op=nl.sin, data=t)", "float32"),
                         ("activation exp scale", "nisa.activation(dst=o, op=nl.exp, data=t, scale=0.5)", "float32"),
                         ("reciprocal", "nisa.reciprocal(dst=o, data=t)", "float32")]:
  src = TT.format(dt="float32", odt=cdt, instr=instr)
  a = np.clip(A["float32"] / 4, -3, 3)
  ref = {"tensor_copy f32->i32": np.rint(a), "activation sin": np.sin(a), "activation exp scale": np.exp(a * 0.5), "reciprocal": 1 / a}[name]
  sim, cc, s = run(src, a, B["float32"], np.int32 if cdt == "int32" else np.float32, ref)
  print(f"| {name} | - | float32 | - | {sim} | {cc} | {s:.1f} |", flush=True)
