"""NKI renderer: kernel AST -> NKI tile-program source (Phase C, N1; docs/NKI_BACKEND.md §2 and §4). Owned by C-N1.

UOps are bypassed (exception 1): NKI programs work on 128-partition tiles, so the renderer reads the kernel AST and the
ShapeTrackers directly and writes nki 0.6 Python: `nl.ndarray` SBUF tiles, dst-first `nisa.*` instructions and HBM
access patterns (`data1.ap(pattern=[[stride, count], ...], offset=...)`). Symbolic dims are bound at render time
(exception 2: `var_vals` are part of the source, so they are part of any compile key).

Layout. After size-1 dims are dropped and mergeable neighbours (same kind, affine in every access, unmasked) are
merged, the kernel has a few dims. The free axis F is the last reduce dim (reduce kernels) or the last dim. The
partition axis P is the largest remaining non-reduce dim; an elementwise kernel whose P is short splits F into
(P <= 128) x F. All other dims are loops: outer loops for non-reduce dims, accumulation loops for reduce dims.
Tiles are at most `pmax` x `fmax`. nki 0.6 unrolls every `range` at trace time, so tile starts are Python ints:
ragged tails are smaller tiles (`pn = min(128, N - p0)`), and PAD masks become static sub-box DMAs into a zeroed
tile (nki 0.6 has no `mask=` arguments).

Ops. ADD/MUL/MAX/CMPLT/CMPEQ -> `nisa.tensor_tensor`, or `nisa.tensor_scalar` against a float constant.
EXP2 / LOG2 / SQRT -> exact_exp / exact_log2 / refined_sqrt on VectorE (N5: ScalarE's exp, log and sqrt are inexact on inf2;
FZ_NKI_FAST_EXP=1 keeps ScalarE exp with scale=ln2, log times log2(e), and sqrt). SIN -> Cody-Waite range reduction into [-pi, pi]
(k = rint(x/2pi) by a float -> int32 copy, r = x - k*C1 - k*C2 - k*C3), then ScalarE sin (its fit has no range reduction).
RECIP -> `nisa.reciprocal`. NEG -> times -1 (floats), 0 - x (ints), x == 0 (bool). WHERE -> copy + predicated copy.
CAST -> `tensor_copy` (through float32); float -> int rounds with a float -> int32 copy and steps back toward zero where
that rounded away from it (C truncation; the engines round to nearest even); -> bool is x != 0. BITCAST -> a same-size `.view(dtype)` of the tile.
SUM/MAX -> `nisa.tensor_reduce` along F; several chunks accumulate into a (P, 1) float32 tile. Integer SUM/MAX
never go through tensor_reduce (it computes in float32): an int32 tree of in-place GpSimd `tensor_tensor` ops per chunk,
accumulated in int32, so they are exact (SUM wraps) like the other tiers.

int32 add/subtract/multiply name GpSimd (native integer ALU); int32 CMPLT/MAX split each value into hi/lo 16-bit halves
compared in float32, CMPEQ tests x - y == 0, and integer MAX reduces like integer SUM (an int32 tree), so all are exact.
Everything emitted compiles with neuronx-cc 2.27 for inf2 (NKI-N2: no divide/fmod/mod, no GpSimd compares or max).

Not lowered (NKIUnsupported, never silently wrong): IDIV/MOD (neuronx-cc rejects divide/fmod/mod on inf2), int64, multi-view ShapeTrackers (they need div/mod
index math: a gather kernel, N2+), partition-axis reductions of a kernel with no non-reduce dim run on one
partition (correct, slow), several distinct REDUCE nodes, a reduce over an empty axis with an epilogue LOAD.
Semantics gaps vs the other tiers: int8/int16/uint8 arithmetic and narrowing int casts saturate instead of wrapping,
and int compares/ints in float paths are exact only below 2**24 (the engines compute in float32).
"""
from __future__ import annotations
import math, os
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from fusezap.dtype import DType, dtypes
from fusezap.helpers import prod
from fusezap.ir import LazyOp
from fusezap.ops import BufferOps, UnaryOps, BinaryOps, TernaryOps, ReduceOps
from fusezap.shape.symbolic import Node, sym_infer

class NKIUnsupported(NotImplementedError):
  """The kernel needs something the NKI renderer does not lower yet."""

_NL = {dtypes.float32: "float32", dtypes.float16: "float16", dtypes.bfloat16: "bfloat16", dtypes.int32: "int32",
       dtypes.int16: "int16", dtypes.int8: "int8", dtypes.uint8: "uint8", dtypes.bool: "uint8"}   # bool is uint8 0/1

def nl_dtype(dt: DType) -> str:
  if dt not in _NL: raise NKIUnsupported(f"dtype {dt} has no NKI equivalent")
  return _NL[dt]

# every NKI spelling the renderer writes lives in these tables (docs/NKI_BACKEND.md risk 2: API churn)
TT_OP = {BinaryOps.ADD: "add", BinaryOps.MUL: "multiply", BinaryOps.MAX: "maximum", BinaryOps.CMPLT: "less", BinaryOps.CMPEQ: "equal"}
COMMUTATIVE = {"add", "multiply", "maximum", "equal"}
RED_OP = {ReduceOps.SUM: "add", ReduceOps.MAX: "maximum"}
# int32 add/subtract/multiply name GpSimd (native 32-bit integer ALU, exact and wrapping). Verified with nki 0.6.0 +
# neuronx-cc 2.27 for inf2 (scripts/nki_op_matrix.py): GpSimd compiles only add/subtract/multiply/bitwise ops; maximum,
# minimum and compares on GpSimd are rejected, and on VectorE they compute in float32, so int32 MAX/CMPLT/CMPEQ use the
# exact forms in `int_lt`/`int_eq`. divide/fmod/mod are rejected on every engine, so IDIV/MOD are not lowered.
GPSIMD = ", engine=nisa.gpsimd_engine"
INT_EXACT_OPS = ("add", "subtract", "multiply")
LN2, LOG2E, PI = math.log(2), 1 / math.log(2), math.pi
INV2PI = float(np.float32(1 / (2 * math.pi)))
def _cw(x: float, bits: int) -> float: return math.ldexp(round(math.ldexp(x, bits - math.frexp(x)[1])), math.frexp(x)[1] - bits)
# SIN's Cody-Waite 2pi = C1 + C2 + C3: C1, C2 have 8 significant bits, so k*C1 and k*C2 are exact in float32 for |k| < 2**16
TWOPI_CW = (6.28125, _cw(2 * math.pi - 6.28125, 8))
TWOPI_CW += (float(np.float32(2 * math.pi - TWOPI_CW[0] - TWOPI_CW[1])),)

# ---------------------------------------------------------------- exact transcendentals (N5)
# N4 run 2 on inf2: ScalarE's exp is ~1e-5 relative (exp2 and tanh failed the op suite), its sqrt 3.5e-6, and its log is
# wrong away from 1 (log2 of inputs in [1e-30, 1e30] was off by up to 2e19: the activation table has a limited domain and
# no range reduction). VectorE add/multiply/compare/copy and `reciprocal` matched the simulator bit for bit, so EXP2, LOG2
# and the exp inside the attention kernel are computed there: range reduction by exact float/int steps and a short
# polynomial, ~1 ulp, the same on the simulator and on hardware. FZ_NKI_FAST_EXP=1 keeps the single ScalarE instructions.
EXACT = os.environ.get("FZ_NKI_FAST_EXP", "0") in ("", "0")
EXP2_POLY = tuple(float(np.float32(LN2 ** k / math.factorial(k))) for k in range(10))   # 2^f = e^(f ln2), |f| <= 1
EXP_POLY = tuple(float(np.float32(1 / math.factorial(k))) for k in range(10))           # e^r, |r| <= ln2: 7e-9 truncation
LN2_HI = _cw(LN2, 12)                        # n * LN2_HI is exact for |n| < 2**12
LN2_LO = float(np.float32(LN2 - LN2_HI))
LOG2_POLY = tuple(float(np.float32(2 / (LN2 * (2 * k + 1)))) for k in range(5))   # log2(m) = s * P(s^2), s = (m-1)/(m+1)
FLT_MAX_ = float(np.finfo(np.float32).max)
FLT_MIN_ = float(np.finfo(np.float32).tiny)

def exact_exp(emit, new, dst: str, x: str, base2: bool = True) -> None:
  """dst = 2**x (base2) or e**x, float32 tiles, on VectorE. x is clamped (2**254 overflows to inf, 2**-252 underflows to 0),
  n = rint (a float -> int32 copy; the engines round to nearest even, a truncating one only widens |f| to < 1, which the
  degree-9 polynomial still covers), f = x - n exactly (base 2) or r = x - n*ln2 by Cody-Waite (base e), p = Horner on
  f, then p * 2**n1 * 2**n2 with n1 = rint(n/2), each power built from its exponent bits (so denormal results round once).
  NaN is not handled here (the caller copies x back where x != x). `new(dtype)` declares a tile like x; `emit` writes a line."""
  xc, n, f, p, s = (new(dtypes.float32) for _ in range(5))
  ni = new(dtypes.int32)
  lo, hi = (-252.0, 254.0) if base2 else (-175.0, 177.0)
  emit(f"nisa.tensor_scalar(dst={xc}, data={x}, op0=nl.maximum, operand0={lo!r})")
  emit(f"nisa.tensor_scalar(dst={xc}, data={xc}, op0=nl.minimum, operand0={hi!r})")
  if base2:
    emit(f"nisa.tensor_copy(dst={ni}, src={xc})")
    emit(f"nisa.tensor_copy(dst={n}, src={ni})")
    emit(f"nisa.tensor_tensor(dst={f}, data1={xc}, data2={n}, op=nl.subtract)")
  else:
    emit(f"nisa.tensor_scalar(dst={f}, data={xc}, op0=nl.multiply, operand0={LOG2E!r})")
    emit(f"nisa.tensor_copy(dst={ni}, src={f})")
    emit(f"nisa.tensor_copy(dst={n}, src={ni})")
    for c in (LN2_HI, LN2_LO):   # r = x - n*LN2_HI - n*LN2_LO: the first step is exact
      emit(f"nisa.tensor_scalar(dst={p}, data={n}, op0=nl.multiply, operand0={c!r})")
      emit(f"nisa.tensor_tensor(dst={f}, data1={xc if c == LN2_HI else f}, data2={p}, op=nl.subtract)")
  c = EXP2_POLY if base2 else EXP_POLY
  emit(f"nisa.tensor_scalar(dst={p}, data={f}, op0=nl.multiply, operand0={c[9]!r})")
  emit(f"nisa.tensor_scalar(dst={p}, data={p}, op0=nl.add, operand0={c[8]!r})")
  for k in range(7, -1, -1):
    emit(f"nisa.tensor_tensor(dst={p}, data1={p}, data2={f}, op=nl.multiply)")
    emit(f"nisa.tensor_scalar(dst={p}, data={p}, op0=nl.add, operand0={c[k]!r})")
  emit(f"nisa.tensor_scalar(dst={s}, data={n}, op0=nl.multiply, operand0=0.5)")
  emit(f"nisa.tensor_copy(dst={ni}, src={s})")
  emit(f"nisa.tensor_copy(dst={s}, src={ni})")                                     # n1 = rint(n / 2)
  emit(f"nisa.tensor_tensor(dst={n}, data1={n}, data2={s}, op=nl.subtract)")       # n2 = n - n1; both in [-126, 127]
  for t, out in ((s, p), (n, dst)):   # 2**k = the float32 with exponent bits k + 127: (k + 127) * 2**23 is an exact int
    emit(f"nisa.tensor_scalar(dst={t}, data={t}, op0=nl.add, operand0=127.0)")
    emit(f"nisa.tensor_scalar(dst={t}, data={t}, op0=nl.multiply, operand0=8388608.0)")
    emit(f"nisa.tensor_copy(dst={ni}, src={t})")
    emit(f"{(v := new(None))} = {ni}.view(nl.float32)")
    emit(f"nisa.tensor_tensor(dst={out}, data1={p}, data2={v}, op=nl.multiply)")

def exact_log2(emit, new, dst: str, x: str) -> None:
  """dst = log2(x) for a float32 tile x, on VectorE: subnormals are scaled by 2**23 first; e = the exponent field, m = the
  significand in [1, 2), moved to [sqrt(2)/2, sqrt(2)) (e + 1); log2(m) = s * P(s*s) with s = (m - 1) / (m + 1) (m - 1 is
  exact; `reciprocal` matched the simulator on inf2). 0 -> -inf, +inf -> inf, x < 0 or NaN -> NaN."""
  sub, sc, xn, e, h, num, den, s, z, p = (new(dtypes.float32) for _ in range(10))
  cst, eb, mb = (new(dtypes.int32) for _ in range(3))
  u = new(dtypes.uint8)
  emit(f"nisa.tensor_scalar(dst={sub}, data={x}, op0=nl.less, operand0={FLT_MIN_!r})")
  emit(f"nisa.tensor_scalar(dst={sc}, data={sub}, op0=nl.multiply, operand0=8388607.0)")
  emit(f"nisa.tensor_scalar(dst={sc}, data={sc}, op0=nl.add, operand0=1.0)")
  emit(f"nisa.tensor_tensor(dst={xn}, data1={x}, data2={sc}, op=nl.multiply)")
  emit(f"{(bits := new(None))} = {xn}.view(nl.int32)")
  emit(f"nisa.memset(dst={cst}, value=2139095040)")                                # 0x7F800000: the exponent field
  emit(f"nisa.tensor_tensor(dst={eb}, data1={bits}, data2={cst}, op=nl.bitwise_and)")
  emit(f"nisa.tensor_copy(dst={e}, src={eb})")                                     # exact: a multiple of 2**23 below 2**31
  emit(f"nisa.tensor_scalar(dst={e}, data={e}, op0=nl.multiply, operand0={2.0 ** -23!r})")
  emit(f"nisa.tensor_scalar(dst={e}, data={e}, op0=nl.add, operand0=-127.0)")
  emit(f"nisa.tensor_scalar(dst={h}, data={sub}, op0=nl.multiply, operand0=-23.0)")
  emit(f"nisa.tensor_tensor(dst={e}, data1={e}, data2={h}, op=nl.add)")
  emit(f"nisa.memset(dst={cst}, value=8388607)")                                   # 0x007FFFFF: the significand
  emit(f"nisa.tensor_tensor(dst={mb}, data1={bits}, data2={cst}, op=nl.bitwise_and)")
  emit(f"nisa.memset(dst={cst}, value=1065353216)")                                # 0x3F800000: exponent of 1.0
  emit(f"nisa.tensor_tensor(dst={mb}, data1={mb}, data2={cst}, op=nl.bitwise_or)")
  emit(f"{(m := new(None))} = {mb}.view(nl.float32)")
  emit(f"nisa.tensor_scalar(dst={h}, data={m}, op0=nl.greater, operand0={math.sqrt(2)!r})")
  emit(f"nisa.tensor_tensor(dst={e}, data1={e}, data2={h}, op=nl.add)")
  emit(f"nisa.tensor_scalar(dst={h}, data={h}, op0=nl.multiply, operand0=-0.5)")
  emit(f"nisa.tensor_scalar(dst={h}, data={h}, op0=nl.add, operand0=1.0)")
  emit(f"nisa.tensor_tensor(dst={xn}, data1={m}, data2={h}, op=nl.multiply)")      # m in [sqrt(2)/2, sqrt(2))
  emit(f"nisa.tensor_scalar(dst={num}, data={xn}, op0=nl.add, operand0=-1.0)")
  emit(f"nisa.tensor_scalar(dst={den}, data={xn}, op0=nl.add, operand0=1.0)")
  emit(f"nisa.reciprocal(dst={sc}, data={den})")
  emit(f"nisa.tensor_tensor(dst={s}, data1={num}, data2={sc}, op=nl.multiply)")
  emit(f"nisa.tensor_tensor(dst={z}, data1={s}, data2={s}, op=nl.multiply)")
  emit(f"nisa.tensor_scalar(dst={p}, data={z}, op0=nl.multiply, operand0={LOG2_POLY[4]!r})")
  emit(f"nisa.tensor_scalar(dst={p}, data={p}, op0=nl.add, operand0={LOG2_POLY[3]!r})")
  for k in (2, 1, 0):
    emit(f"nisa.tensor_tensor(dst={p}, data1={p}, data2={z}, op=nl.multiply)")
    emit(f"nisa.tensor_scalar(dst={p}, data={p}, op0=nl.add, operand0={LOG2_POLY[k]!r})")
  emit(f"nisa.tensor_tensor(dst={p}, data1={p}, data2={s}, op=nl.multiply)")
  emit(f"nisa.tensor_tensor(dst={dst}, data1={p}, data2={e}, op=nl.add)")
  for val, op, c in (("float('-inf')", "equal", 0.0), ("float('inf')", "greater", FLT_MAX_)):
    emit(f"nisa.memset(dst={h}, value={val})")
    emit(f"nisa.tensor_scalar(dst={u}, data={x}, op0=nl.{op}, operand0={c!r})")
    emit(f"nisa.tensor_copy_predicated(dst={dst}, src={h}, predicate={u})")
  emit(f"nisa.memset(dst={h}, value=float('nan'))")                                # x < 0 or NaN: not (x >= 0)
  emit(f"nisa.tensor_scalar(dst={num}, data={x}, op0=nl.greater_equal, operand0=0.0)")
  emit(f"nisa.tensor_scalar(dst={u}, data={num}, op0=nl.equal, operand0=0.0)")
  emit(f"nisa.tensor_copy_predicated(dst={dst}, src={h}, predicate={u})")

def refined_sqrt(emit, new, dst: str, x: str) -> None:
  """dst = sqrt(x) for float32 tiles, range-reduced (N6): ScalarE sqrt is 3.5e-6 relative near 1 but, like its log, wrong far
  from 1 on inf2 (N5 run 1, sqrt_wide over [2**-126, 2**127]: 69 % bitwise equal, normwise error 4.7, which one Newton step
  from a bad start cannot fix). x = m * 2**(2k) with m in [1, 4) from the exponent and significand bits (subnormals scaled
  by 2**24 first, k -= 12), y = ScalarE sqrt(m), one Newton step y' = (y + m/y) / 2 (m/y by the exact `reciprocal`), then
  y' * 2**k, a power of two built from exponent bits (exact). 0 and -0 -> x, +inf -> inf, x < 0 or NaN -> NaN."""
  sub, sc, xn, e, pf, m, k, y, r = (new(dtypes.float32) for _ in range(9))
  cst, eb, ib = (new(dtypes.int32) for _ in range(3))
  u = new(dtypes.uint8)
  emit(f"nisa.tensor_scalar(dst={sub}, data={x}, op0=nl.less, operand0={FLT_MIN_!r})")
  emit(f"nisa.tensor_scalar(dst={sc}, data={sub}, op0=nl.multiply, operand0=16777215.0)")
  emit(f"nisa.tensor_scalar(dst={sc}, data={sc}, op0=nl.add, operand0=1.0)")
  emit(f"nisa.tensor_tensor(dst={xn}, data1={x}, data2={sc}, op=nl.multiply)")      # subnormals * 2**24 (exact)
  emit(f"{(bits := new(None))} = {xn}.view(nl.int32)")
  emit(f"nisa.memset(dst={cst}, value=2139095040)")                                # 0x7F800000: the exponent field
  emit(f"nisa.tensor_tensor(dst={eb}, data1={bits}, data2={cst}, op=nl.bitwise_and)")
  emit(f"nisa.tensor_copy(dst={e}, src={eb})")                                     # exact: a multiple of 2**23 below 2**31
  emit(f"nisa.tensor_scalar(dst={e}, data={e}, op0=nl.multiply, operand0={2.0 ** -23!r})")
  emit(f"nisa.tensor_scalar(dst={e}, data={e}, op0=nl.add, operand0=-127.0)")      # E, the unbiased exponent
  emit(f"nisa.memset(dst={cst}, value=8388608)")                                   # 0x00800000: the exponent's low bit
  emit(f"nisa.tensor_tensor(dst={eb}, data1={bits}, data2={cst}, op=nl.bitwise_and)")
  emit(f"nisa.tensor_copy(dst={pf}, src={eb})")
  emit(f"nisa.tensor_scalar(dst={pf}, data={pf}, op0=nl.multiply, operand0={2.0 ** -23!r})")   # 1 iff E is even
  emit(f"nisa.memset(dst={cst}, value=8388607)")                                   # 0x007FFFFF: the significand
  emit(f"nisa.tensor_tensor(dst={ib}, data1={bits}, data2={cst}, op=nl.bitwise_and)")
  emit(f"nisa.memset(dst={cst}, value=1065353216)")                                # 0x3F800000: exponent of 1.0
  emit(f"nisa.tensor_tensor(dst={ib}, data1={ib}, data2={cst}, op=nl.bitwise_or)")
  emit(f"{(m1 := new(None))} = {ib}.view(nl.float32)")                             # the significand in [1, 2)
  emit(f"nisa.tensor_scalar(dst={r}, data={pf}, op0=nl.multiply, operand0=-1.0)")
  emit(f"nisa.tensor_scalar(dst={r}, data={r}, op0=nl.add, operand0=2.0)")
  emit(f"nisa.tensor_tensor(dst={m}, data1={m1}, data2={r}, op=nl.multiply)")      # m in [1, 4): times 2 when E is odd
  emit(f"nisa.tensor_tensor(dst={k}, data1={e}, data2={pf}, op=nl.add)")
  emit(f"nisa.tensor_scalar(dst={k}, data={k}, op0=nl.add, operand0=-1.0)")
  emit(f"nisa.tensor_scalar(dst={k}, data={k}, op0=nl.multiply, operand0=0.5)")    # k = (E - 1 + pf) / 2, exact
  emit(f"nisa.tensor_scalar(dst={sc}, data={sub}, op0=nl.multiply, operand0=-12.0)")
  emit(f"nisa.tensor_tensor(dst={k}, data1={k}, data2={sc}, op=nl.add)")
  emit(f"nisa.activation(dst={y}, op=nl.sqrt, data={m})")
  emit(f"nisa.reciprocal(dst={r}, data={y})")
  emit(f"nisa.tensor_tensor(dst={r}, data1={m}, data2={r}, op=nl.multiply)")
  emit(f"nisa.tensor_tensor(dst={r}, data1={r}, data2={y}, op=nl.add)")
  emit(f"nisa.tensor_scalar(dst={r}, data={r}, op0=nl.multiply, operand0=0.5)")
  emit(f"nisa.tensor_scalar(dst={k}, data={k}, op0=nl.add, operand0=127.0)")       # 2**k = the float32 with exponent bits k + 127
  emit(f"nisa.tensor_scalar(dst={k}, data={k}, op0=nl.multiply, operand0=8388608.0)")
  emit(f"nisa.tensor_copy(dst={eb}, src={k})")
  emit(f"{(p2 := new(None))} = {eb}.view(nl.float32)")
  emit(f"nisa.tensor_tensor(dst={dst}, data1={r}, data2={p2}, op=nl.multiply)")
  for c, op in ((0.0, "equal"), (FLT_MAX_, "greater")):                            # +-0 and +inf: x itself
    emit(f"nisa.tensor_scalar(dst={u}, data={x}, op0=nl.{op}, operand0={c!r})")
    emit(f"nisa.tensor_copy_predicated(dst={dst}, src={x}, predicate={u})")
  emit(f"nisa.memset(dst={r}, value=float('nan'))")                                # x < 0 or NaN: not (x >= 0)
  emit(f"nisa.tensor_scalar(dst={y}, data={x}, op0=nl.greater_equal, operand0=0.0)")
  emit(f"nisa.tensor_scalar(dst={u}, data={y}, op0=nl.equal, operand0=0.0)")
  emit(f"nisa.tensor_copy_predicated(dst={dst}, src={r}, predicate={u})")

@dataclass(frozen=True)
class NKIProgram:
  name: str
  src: str
  n_bufs: int
  outs: tuple[int, ...]
  buf_dtypes: dict[int, DType]

@dataclass
class _Acc:
  """How one LOAD/CONST/STORE addresses its buffer in kernel dims: offset + sum(i_d * strides[d]), valid in the box."""
  strides: list[int]
  offset: int
  box: list[tuple[int, int]]
  dead: bool = False      # nothing valid: a load reads 0, a store writes nothing

@dataclass
class _Level:
  """Where values are computed: the tile shape (rows, cols) and the index (int or loop variable) of each kernel dim."""
  name: str
  rows: Any
  cols: Any
  at: dict[int, Any] = field(default_factory=dict)    # dim -> tile start (P, F) or loop index; missing dims are pinned at 0

def _int(x: Any, what: str) -> int:
  if isinstance(x, (int, np.integer)): return int(x)
  raise NKIUnsupported(f"{what} {x} is symbolic after binding var_vals")

def _lin(terms: list[tuple[int, Any]], const: int = 0) -> Any:
  """sum(k * x) + const as an int (all x ints) or as Python source."""
  parts: list[str] = []
  for k, x in terms:
    if k == 0 or (isinstance(x, int) and x == 0): continue
    if isinstance(x, int): const += k * x
    else: parts.append(x if k == 1 else f"-{x}" if k == -1 else f"{k}*{x}")
  if not parts: return const
  s = " + ".join(parts).replace("+ -", "- ")
  return s if const == 0 else f"{s} + {const}" if const > 0 else f"{s} - {-const}"

def _add(a: Any, b: Any) -> Any: return _lin([(1, a), (1, b)])
def _sub(a: Any, b: Any) -> Any: return _lin([(1, a), (-1, b)])

def _num(v: Any, dt: DType) -> str:
  """A constant rounded to `dt` (what the other tiers compute with), as Python source."""
  if dt == dtypes.bool: return "1" if v else "0"
  if not dt.is_float: return str(int(v))
  f = float(np.float32(v)) if dt == dtypes.float32 else float(np.float16(v)) if dt == dtypes.float16 else _bf16(float(v))
  return repr(f) if math.isfinite(f) else f"float('{f}')"

def _bf16(f: float) -> float:
  from fusezap.interpret import _bf16_to_f32, _f32_to_bf16
  return float(_bf16_to_f32(_f32_to_bf16(np.array([f], np.float32)))[0])

def _identity(op: ReduceOps, dt: DType) -> Any:
  if op is ReduceOps.SUM or dt == dtypes.bool: return 0
  return -math.inf if dt.is_float else int(np.iinfo(np.dtype(dt.fmt)).min)

def _view_acc(st: Any, full: tuple[int, ...], const: bool) -> _Acc:
  """_Acc over the kernel dims `full` of a bound ShapeTracker whose shape is `full` (or `full` with reduce dims at 1)."""
  n = len(full)
  if const and all(v.mask is None for v in st.views): return _Acc([0] * n, 0, [(0, s) for s in full])
  if len(st.views) != 1: raise NKIUnsupported(f"multi-view ShapeTracker ({len(st.views)} views) needs a gather kernel")
  v = st.views[0]
  shape = [_int(s, "shape") for s in v.shape]
  strides = [_int(s, "stride") for s in v.strides]
  box = [(0, s) for s in shape] if v.mask is None else [(_int(b, "mask"), _int(e, "mask")) for b, e in v.mask]
  a = _Acc([0] * n, 0 if const else _int(v.offset, "offset"), [(0, s) for s in full])
  for d in range(n):
    lo, hi = box[d]
    if hi <= lo: a.dead = True
    if shape[d] == full[d]: a.strides[d], a.box[d] = (0 if const else strides[d]), (lo, hi)
    elif not lo <= 0 < hi: a.dead = True       # a store dim kept at 1 by a reduce: only index 0 exists
  return a

class _Kernel:
  def __init__(self, ast: tuple[LazyOp, ...], var_vals: dict, pmax: int, fmax: int):
    self.ast, self.vv, self.pmax, self.fmax = ast, var_vals, pmax, fmax
    self.lines: list[str] = []
    self.depth, self.ntile, self.nspan, self.scope = 1, 0, 0, 0
    self.span_names: dict[tuple, str] = {}   # (scope, start, ext, lo, hi) -> k of an emitted `ak, bk = ...` line
    self.memo: dict[tuple[int, str], tuple[str, Any, DType]] = {}
    self.tile_dtype: dict[str, DType] = {}

  # ---------------------------------------------------------------- emission helpers
  def emit(self, s: str) -> None: self.lines.append("  " * self.depth + s)
  def open(self, header: str) -> None:
    self.emit(header)
    self.depth, self.scope = self.depth + 1, self.scope + 1
  def _name(self) -> str:
    name = f"t{self.ntile}"; self.ntile += 1
    return name

  def tile(self, lvl: _Level, dt: DType, rows: Any = None, cols: Any = None) -> str:
    name = self._name()
    self.tile_dtype[name] = dt
    self.emit(f"{name} = nl.ndarray(({lvl.rows if rows is None else rows}, {lvl.cols if cols is None else cols}), "
              f"dtype=nl.{nl_dtype(dt)}, buffer=nl.sbuf)")
    return name

  # ---------------------------------------------------------------- layout
  def layout(self) -> None:
    nodes = [n for root in self.ast for n in root.walk()]
    if any(not isinstance(r, LazyOp) or r.op is not BufferOps.STORE for r in self.ast): raise NKIUnsupported("AST roots must be STOREs")
    self.leaves = [n for n in nodes if n.op in (BufferOps.LOAD, BufferOps.CONST)]
    reds = list({id(n): n for n in nodes if isinstance(n.op, ReduceOps)}.values())
    if len(reds) > 1: raise NKIUnsupported(f"{len(reds)} distinct REDUCE nodes in one kernel")
    self.red = reds[0] if reds else None
    for n in nodes:
      if n.op in (BinaryOps.IDIV, BinaryOps.MOD):
        raise NKIUnsupported(f"{n.op}: neuronx-cc 2.27 rejects divide/fmod/mod on inf2 for every engine (scripts/nki_op_matrix.py)")
    bind = lambda st: st.subs(self.vv) if st.vars() else st
    R = tuple(_int(sym_infer(s, self.vv), "store dim") for s in bind(self.ast[0].arg.st).shape)
    F = tuple(_int(s, "dim") for s in bind(self.leaves[0].arg.st).shape) if self.leaves else R
    if any(tuple(_int(s, "store dim") for s in bind(r.arg.st).shape) != R for r in self.ast): raise NKIUnsupported("STOREs of different shapes")
    self.F, self.R = F, R
    raxes = [d for d in range(len(F)) if R[d] == 1 and F[d] != 1] if self.red is not None else []
    self.raxes, self.red_noop = raxes, self.red is not None and not raxes   # a reduce over size-1 axes is the identity
    self.accs: dict[int, _Acc] = {id(n): _view_acc(bind(n.arg.st), F, n.op is BufferOps.CONST) for n in self.leaves}
    self.accs.update({id(r): _view_acc(bind(r.arg.st), F, False) for r in self.ast})
    self.empty = prod(R) == 0
    self.red_empty = self.red is not None and any(F[d] == 0 for d in raxes)
    sizes, kinds = list(F), ["r" if d in raxes else "n" for d in range(len(F))]
    accs = list(self.accs.values())
    if self.empty: self.sizes, self.kinds = sizes, kinds; return
    if self.red_empty:   # nothing to reduce: the result is the identity, the epilogue reads the reduce dims at index 0
      if any(n.op is BufferOps.LOAD for n in self.leaves if not self._under_reduce(n)): raise NKIUnsupported("reduce over an empty axis with an epilogue LOAD")
      for d in reversed(raxes):
        for a in accs: del a.strides[d], a.box[d]
        del sizes[d], kinds[d]
    for d in reversed(range(len(sizes))):    # size-1 dims: index 0 only
      if sizes[d] != 1: continue
      for a in accs:
        if not a.box[d][0] <= 0 < a.box[d][1]: a.dead = True
        del a.strides[d], a.box[d]
      del sizes[d], kinds[d]
    d = len(sizes) - 2
    while d >= 0:                            # merge neighbours that are affine and unmasked in every access
      if kinds[d] == kinds[d + 1] and all(a.dead or (a.box[d] == (0, sizes[d]) and a.box[d + 1] == (0, sizes[d + 1]) and
                                                     a.strides[d] == a.strides[d + 1] * sizes[d + 1]) for a in accs):
        for a in accs: a.strides[d:d + 2], a.box[d:d + 2] = [a.strides[d + 1]], [(0, sizes[d] * sizes[d + 1])]
        sizes[d:d + 2], kinds[d:d + 2] = [sizes[d] * sizes[d + 1]], [kinds[d]]
      d -= 1
    rdims, ndims = [d for d, k in enumerate(kinds) if k == "r"], [d for d, k in enumerate(kinds) if k == "n"]
    fdim = rdims[-1] if rdims else ndims[-1] if ndims else None
    cands = [d for d in ndims if d != fdim]
    pdim = max(cands, key=lambda d: (sizes[d], d)) if cands else None
    if not rdims and fdim is not None and (pdim is None or sizes[pdim] < self.pmax) and \
       all(a.dead or a.box[fdim] == (0, sizes[fdim]) for a in accs):
      n = sizes[fdim]
      p = max(q for q in range(1, min(self.pmax, n) + 1) if n % q == 0)
      if p > (sizes[pdim] if pdim is not None else 1) and n // p >= 2:   # split F into (P, F/P)
        for a in accs: a.strides[fdim:fdim + 1], a.box[fdim:fdim + 1] = [a.strides[fdim] * (n // p), a.strides[fdim]], [(0, p), (0, n // p)]
        sizes[fdim:fdim + 1], kinds[fdim:fdim + 1] = [p, n // p], ["n", "n"]
        pdim, fdim = fdim, fdim + 1
    self.sizes, self.kinds, self.pdim, self.fdim = sizes, kinds, pdim, fdim
    self.outer = [d for d, k in enumerate(kinds) if k == "n" and d not in (pdim, fdim)]
    self.rloops = [d for d, k in enumerate(kinds) if k == "r" and d != fdim]

  def _under_reduce(self, leaf: LazyOp) -> bool:
    return self.red is not None and any(n is leaf for n in self.red.walk())

  # ---------------------------------------------------------------- memory access
  def _spans(self, a: _Acc, lvl: _Level) -> tuple[list[str], list[tuple[Any, Any, Any]], bool] | None:
    """For the P and F axes: (start, count, slice of the tile) of the valid part, plus guard conditions.
    None if nothing is valid in this tile; the bool says the tile is only partly covered."""
    conds: list[str] = []
    for d in self.outer + self.rloops:
      lo, hi = a.box[d]
      if (lo, hi) == (0, self.sizes[d]): continue
      i = lvl.at.get(d, 0)
      if isinstance(i, int):
        if not lo <= i < hi: return None
      else: conds.append(f"{lo} <= {i} < {hi}")
    spans, partial = [], bool(conds)
    for d, ext in ((self.pdim, lvl.rows), (self.fdim, lvl.cols)):
      start = lvl.at.get(d, 0) if d is not None else 0
      lo, hi = a.box[d] if d is not None else (0, 1)
      if d is None or (lo, hi) == (0, self.sizes[d]): spans.append((start, ext, None)); continue
      if isinstance(start, int) and isinstance(ext, int):
        s, e = max(start, lo), min(start + ext, hi)
        if s >= e: return None
        if (s, e) != (start, start + ext): partial = True
        spans.append((s, e - s, None if (s, e) == (start, start + ext) else f"{s - start}:{e - start}"))
        continue
      if (k := self.span_names.get((self.scope, start, ext, lo, hi))) is None:
        k = self.span_names[(self.scope, start, ext, lo, hi)] = self.nspan; self.nspan += 1
        self.emit(f"a{k}, b{k} = max({start}, {lo}), min({_add(start, ext)}, {hi})")
      conds.append(f"a{k} < b{k}"); partial = True
      spans.append((f"a{k}", f"b{k} - a{k}", f"{_sub(f'a{k}', start)}:{_sub(f'b{k}', start)}"))
    return conds, spans, partial

  def _ap(self, buf: int, a: _Acc, lvl: _Level, spans: list[tuple[Any, Any, Any]]) -> str:
    sp = a.strides[self.pdim] if self.pdim is not None else 0
    sf = a.strides[self.fdim] if self.fdim is not None else 0
    terms = [(a.strides[d], lvl.at.get(d, 0)) for d in self.outer + self.rloops]
    terms += [(sp, spans[0][0]), (sf, spans[1][0])]
    return f"data{buf}.ap(pattern=[[{sp}, {spans[0][1]}], [{sf}, {spans[1][1]}]], offset={_lin(terms, a.offset)})"

  def _guarded(self, conds: list[str], line: str) -> None:
    if conds: self.emit(f"if {' and '.join(conds)}: {line}")
    else: self.emit(line)

  def _sub_tile(self, t: str, spans: list[tuple[Any, Any, Any]]) -> str:
    if spans[0][2] is None and spans[1][2] is None: return t
    return f"{t}[{spans[0][2] or ':'}, {spans[1][2] or ':'}]"

  def load(self, n: LazyOp, lvl: _Level) -> tuple[str, Any, DType]:
    a, dt = self.accs[id(n)], n.dtype
    zero = _num(0, dt)
    if n.op is BufferOps.CONST:
      v = n.arg.val
      if isinstance(v, Node): v = sym_infer(v, self.vv)
      if not a.dead and all(b == (0, s) for b, s in zip(a.box, self.sizes)): return ("s", _num(v, dt), dt)
    r = None if a.dead else self._spans(a, lvl)
    t = self.tile(lvl, dt)
    if r is None:
      self.emit(f"nisa.memset(dst={t}, value={zero})")
      return ("t", t, dt)
    conds, spans, partial = r
    if partial: self.emit(f"nisa.memset(dst={t}, value={zero})")
    if n.op is BufferOps.CONST and spans[0][2] is not None:
      # compute engines must start on a partition quadrant (0/32/64/96; neuronx-cc rejects a memset at partition 1),
      # DMAs need not: fill a whole tile, then DMA the valid sub-box across
      self.emit(f"nisa.memset(dst={(tv := self.tile(lvl, dt))}, value={_num(v, dt)})")
      self._guarded(conds, f"nisa.dma_copy(dst={self._sub_tile(t, spans)}, src={self._sub_tile(tv, spans)})")
    elif n.op is BufferOps.CONST: self._guarded(conds, f"nisa.memset(dst={self._sub_tile(t, spans)}, value={_num(v, dt)})")
    else: self._dma(conds, n.arg.idx, a, lvl, spans, t, load=True)
    return ("t", t, dt)

  def store(self, root: LazyOp, lvl: _Level) -> None:
    a = self.accs[id(root)]
    val = self.as_tile(self.value(root.src[0], lvl), lvl)
    if a.dead or (r := self._spans(a, lvl)) is None: return
    conds, spans, _ = r
    self._dma(conds, root.arg.idx, a, lvl, spans, val, load=False)

  def _dma(self, conds: list[str], buf: int, a: _Acc, lvl: _Level, spans: list[tuple[Any, Any, Any]], t: str, load: bool) -> None:
    """HBM <-> tile DMA. neuronx-cc rejects a negative partition step in an HBM access pattern (a flip of the partition
    axis), so that becomes one DMA per partition row (negative free-axis steps compile)."""
    sp = a.strides[self.pdim] if self.pdim is not None else 0
    if sp >= 0:
      ap, sub = self._ap(buf, a, lvl, spans), self._sub_tile(t, spans)
      self._guarded(conds, f"nisa.dma_copy(dst={sub}, src={ap})" if load else f"nisa.dma_copy(dst={ap}, src={sub})")
      return
    depth = self.depth
    if conds: self.open(f"if {' and '.join(conds)}:")
    q = f"q{self.nspan}"; self.nspan += 1
    self.open(f"for {q} in range({spans[0][1]}):")
    sf = a.strides[self.fdim] if self.fdim is not None else 0
    terms = [(a.strides[d], lvl.at.get(d, 0)) for d in self.outer + self.rloops] + [(sp, spans[0][0]), (sp, q), (sf, spans[1][0])]
    ap = f"data{buf}.ap(pattern=[[1, 1], [{sf}, {spans[1][1]}]], offset={_lin(terms, a.offset)})"
    p0 = spans[0][2].split(":")[0] if spans[0][2] is not None else "0"
    row = f"{t}[{_add(p0 if not p0.lstrip('-').isdigit() else int(p0), q)}:{_add(_add(p0 if not p0.lstrip('-').isdigit() else int(p0), q), 1)}, {spans[1][2] or ':'}]"
    self.emit(f"nisa.dma_copy(dst={row}, src={ap})" if load else f"nisa.dma_copy(dst={ap}, src={row})")
    self.depth = depth

  # ---------------------------------------------------------------- values
  def as_tile(self, v: tuple[str, Any, DType], lvl: _Level) -> str:
    if v[0] == "t": return v[1]
    t = self.tile(lvl, v[2])
    self.emit(f"nisa.memset(dst={t}, value={v[1]})")
    return t

  def value(self, n: LazyOp, lvl: _Level) -> tuple[str, Any, DType]:
    key = (id(n), lvl.name)
    if key not in self.memo: self.memo[key] = self._value(n, lvl)
    return self.memo[key]

  def _value(self, n: LazyOp, lvl: _Level) -> tuple[str, Any, DType]:
    if n.op in (BufferOps.LOAD, BufferOps.CONST): return self.load(n, lvl)
    if isinstance(n.op, ReduceOps):
      if self.red_noop: return self.value(n.src[0], lvl)
      if self.red_empty: return ("s", _num(_identity(n.op, n.dtype), n.dtype), n.dtype)
      return self.red_val
    dt = n.dtype
    if n.op is UnaryOps.EXP2 and EXACT and (fused := self._exp_scale(n.src[0], lvl)) is not None and fused[1] == 1.0:
      t = self.tile(lvl, dtypes.float32)   # Tensor.exp = EXP2(x * log2(e)): e**x straight from x (Cody-Waite), not via the rounded product
      x = fused[0]
      if dt != dtypes.float32 or self.tile_dtype.get(x, dtypes.float32) != dtypes.float32:
        self.emit(f"nisa.tensor_copy(dst={(xf := self.tile(lvl, dtypes.float32))}, src={x})"); x = xf
      exact_exp(self.emit, lambda d: self.tile(lvl, d) if d is not None else self._name(), t, x, base2=False)
      self.emit(f"nisa.tensor_tensor(dst={(u := self.tile(lvl, dtypes.uint8))}, data1={x}, data2={x}, op=nl.not_equal)")
      self.emit(f"nisa.tensor_copy_predicated(dst={t}, src={x}, predicate={u})")
      if dt != dtypes.float32: self.emit(f"nisa.tensor_copy(dst={(t2 := self.tile(lvl, dt))}, src={t})"); t = t2
      return ("t", t, dt)
    if n.op is UnaryOps.EXP2 and not EXACT and (fused := self._exp_scale(n.src[0], lvl)) is not None:   # exp(x) = EXP2(x*log2e): one ScalarE op
      x, scale = fused
      self.emit(f"nisa.activation(dst={(t := self.tile(lvl, dt))}, op=nl.exp, data={x}, scale={scale!r})")
      return ("t", t, dt)
    srcs = [self.value(s, lvl) for s in n.src]
    sdt = n.src[0].dtype
    if n.op is TernaryOps.WHERE:
      c, x, y = srcs
      if c[0] == "s": return x if c[1] != "0" else y            # constant condition
      t = self.tile(lvl, dt)
      self.emit(f"nisa.memset(dst={t}, value={y[1]})" if y[0] == "s" else f"nisa.tensor_copy(dst={t}, src={y[1]})")
      self.emit(f"nisa.tensor_copy_predicated(dst={t}, src={self.as_tile(x, lvl)}, predicate={c[1]})")   # src must be a tile
      return ("t", t, dt)
    if n.op in TT_OP:
      op = "maximum" if (n.op is BinaryOps.ADD and sdt == dtypes.bool) else TT_OP[n.op]   # bool ADD is OR
      (x, y), t = srcs, None
      if sdt == dtypes.int32 and op not in INT_EXACT_OPS: return self.int_cmp_max(op, self.as_tile(x, lvl), self.as_tile(y, lvl), dt, lvl)
      if sdt.is_float and (x[0] == "s") != (y[0] == "s"):
        data, c, rev = (y, x, True) if x[0] == "s" else (x, y, False)
        t = self.tile(lvl, dt)
        self.emit(f"nisa.tensor_scalar(dst={t}, data={data[1]}, op0=nl.{op}, operand0={c[1]}"
                  f"{', reverse0=True' if rev and op not in COMMUTATIVE else ''})")
        return ("t", t, dt)
      xa, ya = self.as_tile(x, lvl), self.as_tile(y, lvl)
      t = self.tile(lvl, dt)
      self.emit(f"nisa.tensor_tensor(dst={t}, data1={xa}, data2={ya}, op=nl.{op}{GPSIMD if sdt == dtypes.int32 else ''})")
      return ("t", t, dt)
    x = self.as_tile(srcs[0], lvl)
    if n.op is UnaryOps.BITCAST and dt != dtypes.bool:
      if dt.itemsize != sdt.itemsize: raise NKIUnsupported(f"BITCAST {sdt} -> {dt} changes the element size")
      t = f"t{self.ntile}"; self.ntile += 1
      self.emit(f"{t} = {x}.view(nl.{nl_dtype(dt)})")
      return ("t", t, dt)
    if n.op is UnaryOps.NEG:
      if sdt == dtypes.bool: self.emit(f"nisa.tensor_scalar(dst={(t := self.tile(lvl, dt))}, data={x}, op0=nl.equal, operand0=0.0)")
      elif sdt.is_float: self.emit(f"nisa.tensor_scalar(dst={(t := self.tile(lvl, dt))}, data={x}, op0=nl.multiply, operand0=-1.0)")
      else:
        z = self.tile(lvl, sdt)
        self.emit(f"nisa.memset(dst={z}, value=0)")
        self.emit(f"nisa.tensor_tensor(dst={(t := self.tile(lvl, dt))}, data1={z}, data2={x}, op=nl.subtract{GPSIMD if sdt == dtypes.int32 else ''})")
    elif n.op in (UnaryOps.EXP2, UnaryOps.LOG2, UnaryOps.SQRT) and EXACT: t = self.exact_unary(n.op, x, dt, lvl)
    elif n.op is UnaryOps.EXP2: self.emit(f"nisa.activation(dst={(t := self.tile(lvl, dt))}, op=nl.exp, data={x}, scale={LN2!r})")
    elif n.op is UnaryOps.LOG2:
      ln = self.tile(lvl, dtypes.float32) if dt != dtypes.float32 else (t := self.tile(lvl, dt))
      self.emit(f"nisa.activation(dst={ln}, op=nl.log, data={x})")
      if dt != dtypes.float32: t = self.tile(lvl, dt)
      self.emit(f"nisa.tensor_scalar(dst={t}, data={ln}, op0=nl.multiply, operand0={LOG2E!r})")
    elif n.op is UnaryOps.SIN:   # ScalarE sin has no range reduction: r = x - k*2pi in [-pi, pi], k = rint(x/2pi)
      q, k, m, r = (self.tile(lvl, dtypes.float32) for _ in range(4))   # no nl.mod on inf2: k = rint(x/2pi) by a float->int copy
      self.emit(f"nisa.tensor_scalar(dst={q}, data={x}, op0=nl.multiply, operand0={INV2PI!r})")
      self.emit(f"nisa.tensor_copy(dst={(ki := self.tile(lvl, dtypes.int32))}, src={q})")   # engines round to nearest even
      self.emit(f"nisa.tensor_copy(dst={k}, src={ki})")
      src = x
      for c in TWOPI_CW:   # Cody-Waite: only the k*C3 step rounds, so |r| is within ~1 ulp of pi for |x| < ~4e5
        self.emit(f"nisa.tensor_scalar(dst={m}, data={k}, op0=nl.multiply, operand0={c!r})")
        self.emit(f"nisa.tensor_tensor(dst={r}, data1={src}, data2={m}, op=nl.subtract)")
        src = r
      self.emit(f"nisa.activation(dst={(t := self.tile(lvl, dt))}, op=nl.sin, data={r})")
    elif n.op is UnaryOps.SQRT: self.emit(f"nisa.activation(dst={(t := self.tile(lvl, dt))}, op=nl.sqrt, data={x})")
    elif n.op is UnaryOps.RECIP: self.emit(f"nisa.reciprocal(dst={(t := self.tile(lvl, dt))}, data={x})")
    elif n.op in (UnaryOps.CAST, UnaryOps.BITCAST):
      if dt == dtypes.bool: self.emit(f"nisa.tensor_scalar(dst={(t := self.tile(lvl, dt))}, data={x}, op0=nl.not_equal, operand0=0.0)")
      elif sdt.is_float and not dt.is_float: t = self.trunc_cast(x, dt, lvl)   # C casts truncate; the engines round to nearest even
      else: self.emit(f"nisa.tensor_copy(dst={(t := self.tile(lvl, dt))}, src={x})")
    else: raise NKIUnsupported(f"no NKI lowering for {n.op}")
    return ("t", t, dt)

  def exact_unary(self, op: Any, x: str, dt: DType, lvl: _Level) -> str:
    """EXP2 / LOG2 / SQRT of a float tile by exact_exp / exact_log2 / refined_sqrt (in float32, then cast to dt)."""
    def new(d: DType | None) -> str: return self.tile(lvl, d) if d is not None else self._name()
    xf = x
    if dt != dtypes.float32: self.emit(f"nisa.tensor_copy(dst={(xf := self.tile(lvl, dtypes.float32))}, src={x})")
    r = self.tile(lvl, dtypes.float32)
    if op is UnaryOps.EXP2:
      exact_exp(self.emit, new, r, xf)
      self.emit(f"nisa.tensor_tensor(dst={(u := self.tile(lvl, dtypes.uint8))}, data1={xf}, data2={xf}, op=nl.not_equal)")
      self.emit(f"nisa.tensor_copy_predicated(dst={r}, src={xf}, predicate={u})")   # NaN stays NaN
    elif op is UnaryOps.LOG2: exact_log2(self.emit, new, r, xf)
    else: refined_sqrt(self.emit, new, r, xf)
    if dt == dtypes.float32: return r
    self.emit(f"nisa.tensor_copy(dst={(t := self.tile(lvl, dt))}, src={r})")
    return t

  def trunc_cast(self, x: str, dt: DType, lvl: _Level) -> str:
    """float -> int with C truncation, without fmod/mod (inf2 has neither): k = rint(x) by a float->int32 copy, then
    k - 1 where x > 0 and k > x, k + 1 where x < 0 and k < x. Once |x| >= 2**23 it is an integer and k == x, so the
    float32 adjustments are exact. Saturation and NaN -> int max are the engine's, as before."""
    k, g, l, a = (self.tile(lvl, dtypes.float32) for _ in range(4))
    self.emit(f"nisa.tensor_copy(dst={(ki := self.tile(lvl, dtypes.int32))}, src={x})")
    self.emit(f"nisa.tensor_copy(dst={k}, src={ki})")
    self.emit(f"nisa.tensor_tensor(dst={g}, data1={k}, data2={x}, op=nl.greater)")      # rounded up past x
    self.emit(f"nisa.tensor_scalar(dst={a}, data={x}, op0=nl.greater, operand0=0.0)")
    self.emit(f"nisa.tensor_tensor(dst={g}, data1={g}, data2={a}, op=nl.multiply)")
    self.emit(f"nisa.tensor_tensor(dst={l}, data1={k}, data2={x}, op=nl.less)")         # rounded down past x
    self.emit(f"nisa.tensor_scalar(dst={a}, data={x}, op0=nl.less, operand0=0.0)")
    self.emit(f"nisa.tensor_tensor(dst={l}, data1={l}, data2={a}, op=nl.multiply)")
    self.emit(f"nisa.tensor_tensor(dst={k}, data1={k}, data2={g}, op=nl.subtract)")
    self.emit(f"nisa.tensor_tensor(dst={k}, data1={k}, data2={l}, op=nl.add)")
    if dt == dtypes.int32:
      self.emit(f"nisa.tensor_copy(dst={ki}, src={k})")
      return ki
    self.emit(f"nisa.tensor_copy(dst={(t := self.tile(lvl, dt))}, src={k})")
    return t

  def int_split(self, x: str, rows: Any, cols: Any) -> tuple[str, str]:
    """int32 x -> (hi, lo): lo = x & 0xFFFF, hi = x - lo (a multiple of 65536). Both are exact in float32."""
    lvl = _Level("split", rows, cols)
    self.emit(f"nisa.memset(dst={(m := self.tile(lvl, dtypes.int32))}, value=65535)")
    self.emit(f"nisa.tensor_tensor(dst={(lo := self.tile(lvl, dtypes.int32))}, data1={x}, data2={m}, op=nl.bitwise_and)")
    self.emit(f"nisa.tensor_tensor(dst={(hi := self.tile(lvl, dtypes.int32))}, data1={x}, data2={lo}, op=nl.subtract{GPSIMD})")
    return hi, lo

  def int_lt(self, x: str, y: str, rows: Any, cols: Any, dt: DType) -> str:
    """Exact int32 x < y (VectorE compares in float32): (hi_x < hi_y) + (hi_x == hi_y) * (lo_x < lo_y), into a `dt` tile."""
    lvl = _Level("lt", rows, cols)
    (hx, lx), (hy, ly) = self.int_split(x, rows, cols), self.int_split(y, rows, cols)
    a, e, b = (self.tile(lvl, dtypes.float32) for _ in range(3))
    self.emit(f"nisa.tensor_tensor(dst={a}, data1={hx}, data2={hy}, op=nl.less)")
    self.emit(f"nisa.tensor_tensor(dst={e}, data1={hx}, data2={hy}, op=nl.equal)")
    self.emit(f"nisa.tensor_tensor(dst={b}, data1={lx}, data2={ly}, op=nl.less)")
    self.emit(f"nisa.tensor_tensor(dst={e}, data1={e}, data2={b}, op=nl.multiply)")
    self.emit(f"nisa.tensor_tensor(dst={(t := self.tile(lvl, dt))}, data1={a}, data2={e}, op=nl.add)")
    return t

  def int_max_into(self, dst: str, x: str, y: str, rows: Any, cols: Any) -> None:
    """dst = max(x, y) for int32 tiles, exactly: copy x, then a predicated copy of y where x < y (dst may alias x)."""
    p = self.int_lt(x, y, rows, cols, dtypes.uint8)
    if dst != x: self.emit(f"nisa.tensor_copy(dst={dst}, src={x})")
    self.emit(f"nisa.tensor_copy_predicated(dst={dst}, src={y}, predicate={p})")

  def int_cmp_max(self, op: str, x: str, y: str, dt: DType, lvl: _Level) -> tuple[str, Any, DType]:
    """int32 CMPLT / CMPEQ / MAX, exact for every int32 (the float32 engines are exact only below 2**24)."""
    if op == "less": return ("t", self.int_lt(x, y, lvl.rows, lvl.cols, dt), dt)
    if op == "equal":   # x - y wraps on GpSimd and is 0 exactly when x == y; a nonzero int32 is nonzero in float32
      self.emit(f"nisa.tensor_tensor(dst={(d := self.tile(lvl, dtypes.int32))}, data1={x}, data2={y}, op=nl.subtract{GPSIMD})")
      self.emit(f"nisa.tensor_scalar(dst={(t := self.tile(lvl, dt))}, data={d}, op0=nl.equal, operand0=0.0)")
      return ("t", t, dt)
    assert op == "maximum", op
    self.int_max_into((t := self.tile(lvl, dt)), x, y, lvl.rows, lvl.cols)
    return ("t", t, dt)

  def _exp_scale(self, m: LazyOp, lvl: _Level) -> tuple[str, float] | None:
    """EXP2(MUL(x, c)) with a finite float constant c: (tile of x, c*ln2) for a single activation(exp, scale=...)."""
    if m.op is not BinaryOps.MUL or not m.dtype.is_float: return None
    for i in (0, 1):
      c = self.value(m.src[i], lvl) if m.src[i].op is BufferOps.CONST else None
      if c is None or c[0] != "s" or c[1].startswith("float("): continue   # inf/nan constants are spelled float('...')
      return self.as_tile(self.value(m.src[1 - i], lvl), lvl), float(np.float32(float(c[1]) * LN2))
    return None

  # ---------------------------------------------------------------- loops
  def loop(self, var: str, size: int, step: int) -> tuple[Any, Any]:
    """A tile loop over `size` in steps of `step`: (start, extent) as ints when one tile covers it."""
    if size <= step: return 0, size
    ext = "pn" if var == "p0" else "fn"
    self.open(f"for {var} in range(0, {size}, {step}):")
    self.emit(f"{ext} = min({step}, {size} - {var})")
    return var, ext

  def render(self, name: str) -> str:
    self.layout()
    nb = 1 + max([n.arg.idx for n in self.leaves if n.op is BufferOps.LOAD] + [r.arg.idx for r in self.ast])
    self.n_bufs, self.outs = nb, tuple(r.arg.idx for r in self.ast)
    head = [f"# fusezap NKI kernel (nki 0.6 API), rendered by fusezap.codegen.nki from F={self.F} R={self.R}"]
    if not self.empty:
      dims = lambda ds: ", ".join(f"d{d}[{self.sizes[d]}]" for d in ds) or "-"
      head.append(f"# kernel dims {self.sizes} ({''.join(self.kinds) or '-'}): P {dims([self.pdim] if self.pdim is not None else [])} "
                  f"F {dims([self.fdim] if self.fdim is not None else [])} loops {dims(self.outer + self.rloops)}")
      self.body()
    outs = ", ".join(f"data{i}" for i in self.outs)
    sig = ", ".join(f"data{i}" for i in range(nb))
    return "\n".join(head + ["import nki", "import nki.language as nl", "import nki.isa as nisa", "", "@nki.jit",
                             f"def {name}({sig}):"] + (self.lines or ["  pass"]) + [f"  return {outs}", ""])

  def body(self) -> None:
    at: dict[int, Any] = {}
    for k, d in enumerate(self.outer):
      self.open(f"for i{k} in range({self.sizes[d]}):")
      at[d] = f"i{k}"
    p0, pn = self.loop("p0", self.sizes[self.pdim], self.pmax) if self.pdim is not None else (0, 1)
    if self.pdim is not None: at[self.pdim] = p0
    if self.red is None or self.red_noop or self.red_empty:
      f0, fn = self.loop("f0", self.sizes[self.fdim], self.fmax) if self.fdim is not None else (0, 1)
      if self.fdim is not None: at[self.fdim] = f0
      lvl = _Level("main", pn, fn, at)
      for root in self.ast: self.store(root, lvl)
      return
    self.reduce(dict(at), pn)
    epi = _Level("epi", pn, 1, at)
    for root in self.ast: self.store(root, epi)

  def reduce(self, at: dict[int, Any], pn: Any) -> None:
    red, depth = self.red, self.depth
    dt, op = red.dtype, RED_OP[red.op]
    acc_dt = dtypes.int32 if not dt.is_float else dtypes.float32   # ints never reduce through tensor_reduce (float32)
    epi = _Level("epi", pn, 1, at)
    F = self.sizes[self.fdim]
    if not self.rloops and F <= self.fmax:   # one chunk: reduce straight into the result tile
      pro = _Level("pro", pn, F, {**at, self.fdim: 0})
      x = self.as_tile(self.value(red.src[0], pro), pro)
      if acc_dt == dtypes.int32:
        part = self.int_tree(x, pro, op)
        if dt == acc_dt: self.red_val = ("t", part, dt); return
        self.emit(f"nisa.tensor_copy(dst={(t := self.tile(epi, dt))}, src={part})")
      else: self.emit(f"nisa.tensor_reduce(dst={(t := self.tile(epi, dt))}, op=nl.{op}, data={x}, axis=1, keepdims=True)")
      self.red_val = ("t", t, dt)
      return
    acc = self.tile(epi, acc_dt)
    ident = ("0" if red.op is ReduceOps.SUM else str(-2**31)) if acc_dt == dtypes.int32 else "0.0" if red.op is ReduceOps.SUM else "float('-inf')"
    self.emit(f"nisa.memset(dst={acc}, value={ident})")
    pat = dict(at)
    for k, d in enumerate(self.rloops):
      self.open(f"for r{k} in range({self.sizes[d]}):")
      pat[d] = f"r{k}"
    f0, fn = self.loop("f0", F, self.fmax)
    pat[self.fdim] = f0
    pro = _Level("pro", pn, fn, pat)
    x = self.as_tile(self.value(red.src[0], pro), pro)
    if acc_dt == dtypes.int32: part = self.int_tree(x, pro, op)
    else: self.emit(f"nisa.tensor_reduce(dst={(part := self.tile(epi, acc_dt))}, op=nl.{op}, data={x}, axis=1, keepdims=True)")
    if acc_dt == dtypes.int32 and op == "maximum": self.int_max_into(acc, acc, part, pn, 1)
    else: self.emit(f"nisa.tensor_tensor(dst={acc}, data1={acc}, data2={part}, op=nl.{op}{GPSIMD if acc_dt == dtypes.int32 else ''})")
    self.depth, self.scope = depth, self.scope + 1
    if dt == acc_dt: self.red_val = ("t", acc, dt); return
    t = self.tile(epi, dt)
    self.emit(f"nisa.tensor_copy(dst={t}, src={acc})")
    self.red_val = ("t", t, dt)

  def int_tree(self, x: str, pro: _Level, op: str) -> str:
    """Exact int32 row sums/maxes of tile `x` (pro.rows, pro.cols): halve the width with in-place GpSimd int32 ops
    (tensor_reduce computes in float32, so it is wrong past 2**24). nki 0.6 traces Python, so the while loop unrolls."""
    w = self.tile(pro, dtypes.int32)
    self.emit(f"nisa.tensor_copy(dst={w}, src={x})")
    self.emit(f"{w}_n = {pro.cols}")
    self.open(f"while {w}_n > 1:")
    self.emit(f"{w}_h = {w}_n // 2")
    lo, hi = f"{w}[:, 0:{w}_h]", f"{w}[:, {w}_n - {w}_h:{w}_n]"
    if op == "maximum": self.int_max_into(lo, lo, hi, pro.rows, f"{w}_h")
    else: self.emit(f"nisa.tensor_tensor(dst={lo}, data1={lo}, data2={hi}, op=nl.{op}{GPSIMD})")
    self.emit(f"{w}_n -= {w}_h")
    self.depth -= 1
    self.emit(f"{(t := f't{self.ntile}')} = {w}[:, 0:1]"); self.ntile += 1
    return t

def kernel_name(ast: tuple[LazyOp, ...], var_vals: dict | None = None) -> str:
  """Same scheme as fusezap.codegen.lower: E_<out dims> or r_<out dims>_<reduce dims>."""
  vv = var_vals or {}
  nodes = [n for root in ast for n in root.walk()]
  leaves = [n for n in nodes if n.op in (BufferOps.LOAD, BufferOps.CONST)]
  dim = lambda s: int(sym_infer(s, vv)) if isinstance(s, Node) else int(s)
  R = tuple(dim(s) for s in ast[0].arg.st.shape)
  F = tuple(dim(s) for s in leaves[0].arg.st.shape) if leaves else R
  red = any(isinstance(n.op, ReduceOps) for n in nodes)
  raxes = [d for d in range(len(F)) if R[d] == 1 and F[d] != 1] if red else []
  return "_".join(["r" if red else "E"] + [str(R[d]) for d in range(len(R)) if R[d] != 1] + [str(F[d]) for d in raxes])

class NKIRenderer:
  """Renders kind="kernel" ScheduleItem ASTs to NKI source. `pmax`/`fmax` are the tile limits (the NEURON BEAM knobs, N6)."""
  device = "NEURON"
  def __init__(self, pmax: int = 128, fmax: int = 512):
    assert 1 <= pmax <= 128 and fmax >= 1
    self.pmax, self.fmax = pmax, fmax
  def render(self, ast: tuple[LazyOp, ...], var_vals: dict | None = None, name: str | None = None) -> NKIProgram:
    ast = tuple(ast)
    k = _Kernel(ast, var_vals or {}, self.pmax, self.fmax)
    name = name or kernel_name(ast, var_vals)
    src = k.render(name)
    bdt: dict[int, DType] = {}
    for n in [n for root in ast for n in root.walk()]:
      if n.op in (BufferOps.LOAD, BufferOps.STORE): bdt.setdefault(n.arg.idx, n.arg.dtype)
    for d in bdt.values(): nl_dtype(d)
    return NKIProgram(name, src, k.n_bufs, k.outs, bdt)
