"""NKI-N3 probe: the TensorE / PSUM / mask instructions the matmul and attention kernels use, against the installed
nki 0.6.0 simulator and (with neuronx-cc) the inf2 compiler. Run with .venv-nki/bin/python on a Linux x86_64 pod.

Each case is a tiny @nki.jit kernel: simulated and checked against numpy, then compiled for inf2 (NEFF, no device).
Prints one line per case; the record of what was verified is docs/status/NKI-N3.md. Nothing here is imported by fusezap.
"""
import inspect, os, sys, tempfile, time, traceback
import numpy as np
import nki, nki.language as nl, nki.isa as nisa

os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
CC = os.system("command -v neuronx-cc >/dev/null") == 0
R = np.random.default_rng(0)
f32 = np.float32

def neff(k, args, target="inf2"):
  d = tempfile.mkdtemp(prefix="n3_")
  os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"], os.environ["NKI_ARTIFACTS_DIR"] = target, d
  st = time.perf_counter()
  try: k(*[a.copy() for a in args])
  except Exception as e: err = f"{type(e).__name__}: {str(e)[:300]}"
  else: err = None
  p = os.path.join(d, "kernel.neff")
  return (os.path.getsize(p) if os.path.exists(p) else None), time.perf_counter() - st, err

def case(name, k, args, out_idx, ref, tol=(1e-5, 1e-6)):
  try:
    xs = [a.copy() for a in args]
    nki.simulate(k)(*xs)
    got = xs[out_idx]
    ok = np.allclose(got, ref, rtol=tol[0], atol=tol[1], equal_nan=True)
    err = float(np.nanmax(np.abs(got.astype(np.float64) - ref))) if got.size else 0.0
    sim = f"sim {'OK' if ok else 'MISMATCH'} maxabs={err:.3g}"
  except Exception as e:
    sim = f"sim ERROR {type(e).__name__}: {str(e)[:400]}"
  if CC:
    n, s, e = neff(k, args)
    cc = f"neff {n} B {s:.2f}s" if n else f"neff FAIL ({e})"
  else: cc = "neff skipped (no neuronx-cc)"
  print(f"[{name}] {sim} | {cc}", flush=True)

print("nki", nki.__version__ if hasattr(nki, "__version__") else "?", "neuronx-cc on PATH:", CC)
for n in ("nc_matmul", "nc_transpose", "affine_select", "iota", "activation", "tensor_scalar", "tensor_copy_predicated"):
  print(f"nisa.{n}{inspect.signature(getattr(nisa, n))}")
print("nl.psum:", nl.psum, "nl.equal/greater_equal/less_equal:", hasattr(nl, "greater_equal"), hasattr(nl, "less_equal"))

# 1. matmul: K=256 in two PSUM-accumulated steps, Aᵀ loaded by a strided DMA, B K-major (contiguous rows)
@nki.jit
def k_mm(a, b, c):   # a (M=64, K=256) row-major, b (K=256, N=300) row-major, c (64, 300)
  ps = nl.ndarray((64, 300), dtype=nl.float32, buffer=nl.psum)
  for i in range(2):
    at = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
    bt = nl.ndarray((128, 300), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a.ap(pattern=[[1, 128], [256, 64]], offset=i * 128))
    nisa.dma_copy(dst=bt, src=b.ap(pattern=[[300, 128], [1, 300]], offset=i * 128 * 300))
    nisa.nc_matmul(dst=ps, stationary=at, moving=bt, accumulate=(i > 0))
  o = nl.ndarray((64, 300), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=o, src=ps)
  nisa.dma_copy(dst=c.ap(pattern=[[300, 64], [1, 300]], offset=0), src=o)
  return c
a, b = R.standard_normal((64, 256)).astype(f32), R.standard_normal((256, 300)).astype(f32)
case("nc_matmul f32 K=2x128 psum accumulate", k_mm, [a, b, np.zeros((64, 300), f32)], 2, a @ b, (1e-5, 1e-4))

# 1b. accumulate=None (compiler infers) and B N-major (weight as torch Linear stores it: (N, K)); output scaled on the way out of PSUM
@nki.jit
def k_mm2(a, w, c):   # a (8, 256), w (40, 256), c (8, 40) = 0.5 * a @ w.T
  ps = nl.ndarray((8, 40), dtype=nl.float32, buffer=nl.psum)
  for i in range(2):
    at = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.sbuf)
    wt = nl.ndarray((128, 40), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=at, src=a.ap(pattern=[[1, 128], [256, 8]], offset=i * 128))
    nisa.dma_copy(dst=wt, src=w.ap(pattern=[[1, 128], [256, 40]], offset=i * 128))
    nisa.nc_matmul(dst=ps, stationary=at, moving=wt)
  o = nl.ndarray((8, 40), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=o, data=ps, op0=nl.multiply, operand0=0.5)
  nisa.dma_copy(dst=c.ap(pattern=[[40, 8], [1, 40]], offset=0), src=o)
  return c
a, w = R.standard_normal((8, 256)).astype(f32), R.standard_normal((40, 256)).astype(f32)
case("nc_matmul accumulate=None, N-major weight, tensor_scalar from PSUM", k_mm2, [a, w, np.zeros((8, 40), f32)], 2, 0.5 * (a @ w.T), (1e-5, 1e-4))

# 2. nc_transpose (TensorE, SBUF -> PSUM) of a (100, 72) f32 tile
@nki.jit
def k_tr(x, y):   # x (100, 72) -> y (72, 100)
  t = nl.ndarray((100, 72), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=x.ap(pattern=[[72, 100], [1, 72]], offset=0))
  p = nl.ndarray((72, 100), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_transpose(dst=p, data=t)
  o = nl.ndarray((72, 100), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=o, src=p)
  nisa.dma_copy(dst=y.ap(pattern=[[100, 72], [1, 100]], offset=0), src=o)
  return y
x = R.standard_normal((100, 72)).astype(f32)
case("nc_transpose f32 100x72 -> PSUM", k_tr, [x, np.zeros((72, 100), f32)], 1, x.T, (0, 0))

# 2b. nc_transpose of a column slice of a wider tile (the P chunk of flash attention)
@nki.jit
def k_tr2(x, y):   # x (64, 300): y (128, 64) = x[:, 128:256].T
  t = nl.ndarray((64, 300), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=x.ap(pattern=[[300, 64], [1, 300]], offset=0))
  p = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_transpose(dst=p, data=t[:, 128:256])
  o = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=o, src=p)
  nisa.dma_copy(dst=y.ap(pattern=[[64, 128], [1, 64]], offset=0), src=o)
  return y
x = R.standard_normal((64, 300)).astype(f32)
case("nc_transpose of a free-axis slice", k_tr2, [x, np.zeros((128, 64), f32)], 1, x[:, 128:256].T, (0, 0))

# 3. causal mask by affine_select: keep s[l, j] where (l0 + l + coff) - (s0 + j) >= 0, else -inf
@nki.jit
def k_causal(x, y):   # x (96, 200) -> y; l0 = 32, s0 = 64, coff = 5
  t = nl.ndarray((96, 200), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=x.ap(pattern=[[200, 96], [1, 200]], offset=0))
  o = nl.ndarray((96, 200), dtype=nl.float32, buffer=nl.sbuf)
  nisa.affine_select(dst=o, pattern=[[-1, 200]], channel_multiplier=1, on_true_tile=t, on_false_value=-np.inf,
                     cmp_op=nl.greater_equal, offset=32 + 5 - 64)
  nisa.dma_copy(dst=y.ap(pattern=[[200, 96], [1, 200]], offset=0), src=o)
  return y
x = R.standard_normal((96, 200)).astype(f32)
l, j = np.arange(96)[:, None], np.arange(200)[None, :]
case("affine_select causal (-inf)", k_causal, [x, np.zeros((96, 200), f32)], 1, np.where(32 + l + 5 - (64 + j) >= 0, x, -np.inf), (0, 0))

# 4. position-tensor mask: key index (iota along F, offset s0) <= pos[row] (an int32 per partition), as a uint8 predicate
@nki.jit
def k_pos(x, pos, y):   # x (7, 300) f32, pos (7,) int32 -> y = where(s0 + j <= pos[l], x, -inf), s0 = 100
  t = nl.ndarray((7, 300), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=x.ap(pattern=[[300, 7], [1, 300]], offset=0))
  pi = nl.ndarray((7, 1), dtype=nl.int32, buffer=nl.sbuf)
  nisa.dma_copy(dst=pi, src=pos.ap(pattern=[[1, 7], [1, 1]], offset=0))
  pf = nl.ndarray((7, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=pf, src=pi)
  ki = nl.ndarray((7, 300), dtype=nl.int32, buffer=nl.sbuf)
  nisa.iota(dst=ki, pattern=[[1, 300]], offset=100, channel_multiplier=0)
  kf = nl.ndarray((7, 300), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=kf, src=ki)
  m = nl.ndarray((7, 300), dtype=nl.uint8, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=m, data=kf, op0=nl.less_equal, operand0=pf)
  o = nl.ndarray((7, 300), dtype=nl.float32, buffer=nl.sbuf)
  nisa.memset(dst=o, value=-np.inf)
  nisa.tensor_copy_predicated(dst=o, src=t, predicate=m)
  nisa.dma_copy(dst=y.ap(pattern=[[300, 7], [1, 300]], offset=0), src=o)
  return y
x, pos = R.standard_normal((7, 300)).astype(f32), np.array([100, 150, 399, 250, 99, 300, 120], np.int32)
case("iota + position tensor mask", k_pos, [x, pos, np.zeros((7, 300), f32)], 2,
     np.where(100 + np.arange(300)[None, :] <= pos[:, None], x, -np.inf), (0, 0))

# 4b. iota straight into float32
@nki.jit
def k_iotaf(y):
  t = nl.ndarray((16, 40), dtype=nl.float32, buffer=nl.sbuf)
  nisa.iota(dst=t, pattern=[[1, 40]], offset=3, channel_multiplier=40)
  nisa.dma_copy(dst=y.ap(pattern=[[40, 16], [1, 40]], offset=0), src=t)
  return y
case("iota into float32", k_iotaf, [np.zeros((16, 40), f32)], 0, (3 + np.arange(640).reshape(16, 40)).astype(f32), (0, 0))

# 5. online softmax pieces: exp(x - m) with a per-partition bias tile, the row sum, and alpha rescale with a (P, 1) operand
@nki.jit
def k_soft(x, y):   # x (50, 128) -> y = exp(x - rowmax) * 2 / rowsum
  t = nl.ndarray((50, 128), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t, src=x.ap(pattern=[[128, 50], [1, 128]], offset=0))
  mx = nl.ndarray((50, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_reduce(dst=mx, op=nl.maximum, data=t, axis=1, negate=True)
  e = nl.ndarray((50, 128), dtype=nl.float32, buffer=nl.sbuf)
  nisa.activation(dst=e, op=nl.exp, data=t, bias=mx, scale=1.0)
  s = nl.ndarray((50, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_reduce(dst=s, op=nl.add, data=e, axis=1)
  r = nl.ndarray((50, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.reciprocal(dst=r, data=s)
  nisa.tensor_scalar(dst=e, data=e, op0=nl.multiply, operand0=r, op1=nl.multiply, operand1=2.0)
  nisa.dma_copy(dst=y.ap(pattern=[[128, 50], [1, 128]], offset=0), src=e)
  return y
x = R.standard_normal((50, 128)).astype(f32)
ex = np.exp(x - x.max(1, keepdims=True))
case("exp with (P,1) bias, reduce, per-partition scale", k_soft, [x, np.zeros((50, 128), f32)], 1, ex * 2 / ex.sum(1, keepdims=True), (1e-5, 1e-6))

# 6. tensor_tensor with a PSUM operand (acc += O_psum)
@nki.jit
def k_psum_add(a, b, c, y):   # y = c + a.T @ b  (a (128, 16), b (128, 64))
  at = nl.ndarray((128, 16), dtype=nl.float32, buffer=nl.sbuf)
  bt = nl.ndarray((128, 64), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=at, src=a.ap(pattern=[[16, 128], [1, 16]], offset=0))
  nisa.dma_copy(dst=bt, src=b.ap(pattern=[[64, 128], [1, 64]], offset=0))
  ps = nl.ndarray((16, 64), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=ps, stationary=at, moving=bt)
  ct = nl.ndarray((16, 64), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=ct, src=c.ap(pattern=[[64, 16], [1, 64]], offset=0))
  nisa.tensor_tensor(dst=ct, data1=ct, data2=ps, op=nl.add)
  nisa.dma_copy(dst=y.ap(pattern=[[64, 16], [1, 64]], offset=0), src=ct)
  return y
a, b, c = (R.standard_normal(s).astype(f32) for s in ((128, 16), (128, 64), (16, 64)))
case("tensor_tensor SBUF + PSUM", k_psum_add, [a, b, c, np.zeros((16, 64), f32)], 3, c + a.T @ b, (1e-5, 1e-5))

# 7. many unrolled tiles: 4 x 16 PSUM tiles of 128 x 512 (the allocator must reuse SBUF/PSUM across unrolled iterations)
@nki.jit
def k_many(a, b, c):   # a (128, 512) as Aᵀ, b (512, 8192), c (128, 8192)
  for n in range(16):
    ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
    for k in range(4):
      at = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
      bt = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
      nisa.dma_copy(dst=at, src=a.ap(pattern=[[1, 128], [512, 128]], offset=k * 128))
      nisa.dma_copy(dst=bt, src=b.ap(pattern=[[8192, 128], [1, 512]], offset=k * 128 * 8192 + n * 512))
      nisa.nc_matmul(dst=ps, stationary=at, moving=bt, accumulate=(k > 0))
    o = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=o, src=ps)
    nisa.dma_copy(dst=c.ap(pattern=[[8192, 128], [1, 512]], offset=n * 512), src=o)
  return c
a, b = R.standard_normal((128, 512)).astype(f32), R.standard_normal((512, 8192)).astype(f32)
case("64 unrolled matmuls (allocation reuse)", k_many, [a, b, np.zeros((128, 8192), f32)], 2, a @ b, (1e-4, 1e-3))
print("done", flush=True)
