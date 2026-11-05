"""NKI custom kernels for the NEURON device (NKI-N3; docs/NKI_BACKEND.md §2 "Matmul"/"Attention", §4 exception 3, §5).

The scheduler hook (fusezap.schedule, SPEC §11.1) turns a matched matmul or attention subgraph into one kind="custom"
item; NEURONDevice renders it here with every parameter baked in (exception 2) and runs it on the tier's runtime (NKISHIM
shim, NKISIM simulator; NEURON is N4). The kernels follow the nki-samples tutorial structure (MIT-0, github.com/aws-neuron/
nki-samples: tiled matmul with PSUM accumulation, fused attention with an online softmax), written for the nki 0.6 API.

render_matmul(params)  C[b] = A[b] @ B[b] in float32 on TensorE. For each b and row block m0 (<= 128 rows) the Aᵀ chunks
    [K <= 128, M] (stationary) are DMAed in once with K on the partitions and kept in SBUF across the N tiles (re-read per
    N tile when they would take > 64 KiB per partition). Each N tile (<= 512 columns, one PSUM bank) accumulates ceil(K/128)
    `nc_matmul` steps in PSUM (accumulate=(i > 0)), is evicted to SBUF and stored. The moving tile B[K <= 128, N <= 512] is
    one contiguous run per partition when B is K-major (row k is contiguous: what `kmajor(w)` gives a weight once at load
    time); a torch-layout weight (N, K) is read with a strided DMA instead. A negative partition step (a flipped operand)
    becomes one DMA per row, since neuronx-cc rejects it in an HBM access pattern (NKI-N2).
render_attention(spec)  O = softmax(scale * Q Kᵀ masked) V in float32, flash-style. Per (batch, block of <= 128 query rows)
    qᵀ is loaded once (D on the partitions); per block of <= 512 keys S = qᵀ·kᵀ accumulates in PSUM over D chunks, is scaled
    on the way to SBUF and masked; VectorE takes the row max and computes exp(s - m) exactly (nki.exact_exp; FZ_NKI_FAST_EXP=1:
    ScalarE exp with the new max as a (P, 1) bias),
    the running sum and the output are rescaled by exp(m_old - m_new), and P·V accumulates in PSUM over 128-key chunks
    (P is transposed on TensorE). The running max starts at -FLT_MAX, not -inf, so a fully masked block gives zeros, not NaN;
    a row with no visible key at all gives NaN, like the composed softmax.
    Masks: "tensor" (a materialized bool mask, the scheduler's WHERE(mask, scores, -inf)), "causal" (key s visible to row r
    iff s <= r + coff: `affine_select` on the partition/free index, and key blocks past the tile's last visible key are
    skipped) and "pos" (key s visible to row r iff s <= pos[r]: `iota` key indices compared with an int32 position tensor,
    the decode mask of a fixed-size KV cache).
Supported here: float32 operands (the scheduler patterns are float32-only). bf16 operands straight to TensorE are N5.
"""
from __future__ import annotations
import functools, math, os, struct
from dataclasses import dataclass
from fusezap.dtype import dtypes
from fusezap.codegen import nki
from fusezap.codegen.nki import NKIProgram, NKIUnsupported

TK, TM, TN = 128, 128, 512        # TensorE: contraction and stationary free <= 128, moving free <= 512 (one PSUM bank of float32)
TS = 512                          # attention key block (S tile in PSUM)
A_KEEP = 64 * 1024                # bytes per partition for keeping a row block's Aᵀ chunks in SBUF across N tiles
FLT_MAX = float(struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0])
HEADER = "import nki\nimport nki.language as nl\nimport nki.isa as nisa\n\n@nki.jit\n"

def _bits_f32(x: int) -> float: return struct.unpack("<f", struct.pack("<i", int(x)))[0]

def _fl(x: float) -> str: return repr(float(x)) if math.isfinite(x) else f"float('{x}')"

def _dma_in(tile: str, cols: str, src: str, ps: int, pn: str, fs: int, fn: str, off: str, ind: str) -> list[str]:
  """tile[0:pn, cols] <- src.ap([[ps, pn], [fs, fn]], off). neuronx-cc rejects a negative partition step in an HBM access
  pattern (NKI-N2), so that becomes one DMA per partition row (DMAs may start on any partition)."""
  if ps >= 0: return [f"{ind}nisa.dma_copy(dst={tile}[0:{pn}, {cols}], src={src}.ap(pattern=[[{ps}, {pn}], [{fs}, {fn}]], offset={off}))"]
  return [f"{ind}for r in range({pn}):",
          f"{ind}  nisa.dma_copy(dst={tile}[r:r + 1, {cols}], src={src}.ap(pattern=[[1, 1], [{fs}, {fn}]], offset={off} + r * {ps}))"]

# N6: an operand that needs its contiguous rows on the free axis (Aᵀ of a matmul, kᵀ of attention: the contraction dim on the
# partitions) was a strided DMA of single 4-byte elements; for row-contiguous sources it is now loaded as rows (one
# contiguous run per partition) and transposed on TensorE (exact: an identity matmul of finite values, as attention's P).
# FZ_NEURON_KT=0: attention's kᵀ by the strided DMA again; FZ_NEURON_AT_ROWS=m: a matmul's Aᵀ by rows from m rows up
# (default 16; 0 never). Neither changes a bit on NKISHIM / NKISIM (the transpose is exact there).
KT_ROWS = os.environ.get("FZ_NEURON_KT", "1") not in ("", "0")
AT_ROWS = int(os.environ.get("FZ_NEURON_AT_ROWS", "16") or 0)

def _tload(tile: str, cols: str, src: str, rs: int, rn: str, cn: str, off: str, ind: str, tag: str) -> list[str]:
  """tile[0:cn, cols] <- the transpose of rn rows (stride rs, cn contiguous elements each, rn and cn <= 128) of src at off:
  one DMA of the rows, nc_transpose into PSUM, a copy to SBUF."""
  return [f"{ind}{tag}r = nl.ndarray(({rn}, {cn}), dtype=nl.float32, buffer=nl.sbuf)",
          f"{ind}nisa.dma_copy(dst={tag}r, src={src}.ap(pattern=[[{rs}, {rn}], [1, {cn}]], offset={off}))",
          f"{ind}{tag}p = nl.ndarray(({cn}, {rn}), dtype=nl.float32, buffer=nl.psum)",
          f"{ind}nisa.nc_transpose(dst={tag}p, data={tag}r)",
          f"{ind}nisa.tensor_copy(dst={tile}[0:{cn}, {cols}], src={tag}p)"]

def _program(name: str, lines: list[str], n: int, dts: dict) -> NKIProgram:
  return NKIProgram(name, HEADER + "\n".join(lines) + "\n", n, (0,), dts)

# ---------------------------------------------------------------- matmul

MATMUL_PARAMS = ("Bt", "M", "N", "K", "sab", "sam", "sak", "sbb", "sbk", "sbn", "scb", "scm", "scn", "oa", "ob")

def render_matmul(params: tuple[int, ...]) -> NKIProgram:
  """fusezap.schedule's matmul params (C, A, B over float32 buffers) -> an NKI kernel `fz_matmul_...(data0=C, data1=A, data2=B)`."""
  Bt, M, N, K, sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, oa, ob = (int(x) for x in params)
  if min(Bt, M, N, K) <= 0: raise NKIUnsupported(f"fz_matmul: empty matmul {Bt}x{M}x{N}x{K}")
  nk = -(-K // TK)
  keep = nk * min(M, TM) * 4 <= A_KEEP
  name = f"fz_matmul_{Bt}_{M}_{N}_{K}"
  a_off = f"{oa} + b * {sab} + m0 * {sam} + i * {TK * sak}"
  rows = AT_ROWS and M >= AT_ROWS and sak == 1 and sam > 0   # Aᵀ chunks from contiguous rows of A, transposed on TensorE
  ain = (lambda t, cols, ind: _tload(t, cols, "data1", sam, "mt", "kt", a_off, ind, "a")) if rows else \
        (lambda t, cols, ind: _dma_in(t, cols, "data1", sak, "kt", sam, "mt", a_off, ind))
  L = [f"def {name}(data0, data1, data2):",
       f"  for b in range({Bt}):",
       f"    for m0 in range(0, {M}, {TM}):",
       f"      mt = min({TM}, {M} - m0)"]
  if keep:   # every Aᵀ chunk of this row block, side by side: chunk i is at[0:kt, i*mt:(i+1)*mt]
    L += [f"      at = nl.ndarray(({min(K, TK)}, {nk} * mt), dtype=nl.float32, buffer=nl.sbuf)",
          f"      for i in range({nk}):",
          f"        kt = min({TK}, {K} - i * {TK})"]
    L += ain("at", "i * mt:i * mt + mt", "        ")
  L += [f"      for n0 in range(0, {N}, {TN}):",
        f"        nt = min({TN}, {N} - n0)",
        f"        ps = nl.ndarray((mt, nt), dtype=nl.float32, buffer=nl.psum)",
        f"        for i in range({nk}):",
        f"          kt = min({TK}, {K} - i * {TK})"]
  if keep: stat = "at[0:kt, i * mt:i * mt + mt]"
  else:
    L += [f"          at = nl.ndarray((kt, mt), dtype=nl.float32, buffer=nl.sbuf)"]
    L += ain("at", "0:mt", "          ")
    stat = "at"
  L += [f"          bt = nl.ndarray((kt, nt), dtype=nl.float32, buffer=nl.sbuf)"]
  L += _dma_in("bt", ":", "data2", sbk, "kt", sbn, "nt", f"{ob} + b * {sbb} + i * {TK * sbk} + n0 * {sbn}", "          ")
  L += [f"          nisa.nc_matmul(dst=ps, stationary={stat}, moving=bt, accumulate=(i > 0))",
        f"        ot = nl.ndarray((mt, nt), dtype=nl.float32, buffer=nl.sbuf)",
        f"        nisa.tensor_copy(dst=ot, src=ps)",
        f"        nisa.dma_copy(dst=data0.ap(pattern=[[{scm}, mt], [{scn}, nt]], offset=b * {scb} + m0 * {scm} + n0 * {scn}), src=ot)",
        f"  return data0"]
  return _program(name, L, 3, {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.float32})

# ---------------------------------------------------------------- attention

@dataclass(frozen=True)
class AttnSpec:
  """One attention launch. Batch index b = b1 * nb[1] + b2; query row r = r1 * nr[1] + r2 (L = nr[0] * nr[1] rows).
  Strides are in elements over each buffer: q/o = (b1, b2, row, d, offset) with one stride over all rows; k/v = (b1, b2,
  key, d, offset). mask "tensor": m = (b1, b2, r1, r2, key, offset) over a bool (uint8) buffer; "pos": m = (r1, r2) strides
  into an int32 position buffer (key s visible to row r iff s <= pos); "causal": key s visible iff s <= r2 + coff;
  "keyrange" (N5, the paged serving cache): m = (r1, r2, lo, hi, kid) over one int32 buffer: key s visible to row r iff
  lo[r] <= kid[s] <= hi[r] (kid: a per-key id, e.g. sequence row * stride + position; -1 never matches), and a row
  with no visible key gives zeros (the paged kernels' padding rows), not NaN."""
  nb: tuple[int, int]
  nr: tuple[int, int]
  S: int
  D: int
  Dv: int
  q: tuple[int, ...]
  k: tuple[int, ...]
  v: tuple[int, ...]
  o: tuple[int, ...]
  scale: float
  mask: str = "none"
  m: tuple[int, ...] = ()
  coff: int = 0
  rs: tuple[int, int] = ()   # (q, o) strides of the outer row level r1 when rows are not one stride (N5 serving: rows =
                             # (token, q head of the group), the q/o row stride is then the r2 stride); () = one stride
  kv_one: bool = False       # K and V live in one buffer (the serving pool), passed once: data2 holds both, the mask is data3
  @property
  def L(self) -> int: return self.nr[0] * self.nr[1]
  @property
  def name(self) -> str:
    return f"fz_attention_{self.mask}_{self.nb[0] * self.nb[1]}_{self.L}_{self.S}_{self.D}_{self.Dv}"
  def render(self) -> NKIProgram: return render_attention(self)
  def ref(self, arrays: list) -> None:
    """numpy reference of this launch on flat arrays (O, Q, K, V[, M]), float64 math, for tests."""
    import numpy as np
    O, Q, K, V = (np.asarray(a) for a in arrays[:4])
    M = arrays[4] if len(arrays) > 4 else None
    (NB1, NB2), (NR1, NR2) = self.nb, self.nr
    for b1 in range(NB1):
      for b2 in range(NB2):
        for r in range(self.L):
          r1, r2 = divmod(r, NR2)
          qa = self.q[4] + b1 * self.q[0] + b2 * self.q[1] + (r1 * self.rs[0] + r2 * self.q[2] if self.rs else r * self.q[2])
          q = Q[qa + np.arange(self.D) * self.q[3]].astype(np.float64)
          ks = np.stack([K[self.k[4] + b1 * self.k[0] + b2 * self.k[1] + s * self.k[2] + np.arange(self.D) * self.k[3]] for s in range(self.S)])
          vs = np.stack([V[self.v[4] + b1 * self.v[0] + b2 * self.v[1] + s * self.v[2] + np.arange(self.Dv) * self.v[3]] for s in range(self.S)])
          sc = ks.astype(np.float64) @ q * self.scale
          if self.mask == "keyrange":
            lo, hi = M[self.m[2] + r1 * self.m[0] + r2 * self.m[1]], M[self.m[3] + r1 * self.m[0] + r2 * self.m[1]]
            kid = M[self.m[4]:self.m[4] + self.S]
            sc = np.where((kid >= lo) & (kid <= hi), sc, -np.inf)
          oa = self.o[4] + b1 * self.o[0] + b2 * self.o[1] + (r1 * self.rs[1] + r2 * self.o[2] if self.rs else r * self.o[2])
          if np.isneginf(sc).all(): out = np.zeros(self.Dv)
          else: e = np.exp(sc - sc.max()); out = (e / e.sum()) @ vs.astype(np.float64)
          O[oa + np.arange(self.Dv) * self.o[3]] = out

ATTN_MASK_PARAMS = ("mb1", "mb2", "smb1", "smb2", "mr1", "mr2", "smr1", "smr2", "sms", "om")

def spec_from_params(params: tuple[int, ...], masked: bool) -> AttnSpec:
  """fusezap.schedule's attention params (+ the masked pattern's ATTN_MASK_PARAMS) -> AttnSpec."""
  Bt, L, S, D, Dv, sqb, sql, sqd, skb, sks, skd, svb, svs, svd, sob, sol, sod, oq, ok, ov, sc, _ = (int(x) for x in params[:22])
  nb, nr, m = (Bt, 1), (1, L), ()
  if masked:
    mb1, mb2, smb1, smb2, mr1, mr2, smr1, smr2, sms, om = (int(x) for x in params[22:32])
    nb, nr, m = (mb1, mb2), (mr1, mr2), (smb1, smb2, smr1, smr2, sms, om)
  b = lambda s: (s * nb[1], s)
  return AttnSpec(nb, nr, S, D, Dv, (*b(sqb), sql, sqd, oq), (*b(skb), sks, skd, ok), (*b(svb), svs, svd, ov), (*b(sob), sol, sod, 0),
                  _bits_f32(sc), "tensor" if masked else "none", m)

def render_attention(sp: AttnSpec) -> NKIProgram:
  """An AttnSpec -> `fz_attention_...(data0=O, data1=Q, data2=K, data3=V[, data4=mask or positions])`."""
  NB1, NB2 = sp.nb
  NR1, NR2 = sp.nr
  S, D, Dv, L = sp.S, sp.D, sp.Dv, sp.L
  if min(NB1, NB2, L, Dv) <= 0 or S <= 0 or D <= 0: raise NKIUnsupported(f"{sp.name}: empty attention")
  sqb1, sqb2, sql, sqd, oq = sp.q
  skb1, skb2, sks, skd, ok = sp.k
  svb1, svb2, svs, svd, ov = sp.v
  sob1, sob2, sol, sod, oo = sp.o
  if min(sol, sod) < 0: raise NKIUnsupported(f"{sp.name}: negative output strides")
  nd = -(-D // TK)
  causal = sp.mask == "causal"
  two = bool(sp.rs)
  if two and (causal or min(sqd, sql, sol, sod) < 0): raise NKIUnsupported(f"{sp.name}: two-level rows need positive strides, no causal")
  n_in = 5 if sp.mask in ("tensor", "pos", "keyrange") else 4
  args = ", ".join(f"data{i}" for i in range(n_in))
  L_ = [f"def {sp.name}({args}):",
        f"  for b1 in range({NB1}):",
        f"    for b2 in range({NB2}):"]
  i = "      "
  if causal:   # row tiles stay inside one r1, so the partition index is affine in r2 (affine_select)
    L_ += [f"{i}for r1 in range({NR1}):", f"{i}  for t0 in range(0, {NR2}, {TM}):",
           f"{i}    lt = min({TM}, {NR2} - t0)", f"{i}    l0 = r1 * {NR2} + t0"]
    i += "    "
  else: L_ += [f"{i}for l0 in range(0, {L}, {TM}):", f"{i}  lt = min({TM}, {L} - l0)"]; i += "  "
  qo = f"{oq} + b1 * {sqb1} + b2 * {sqb2} + l0 * {sql}"
  L_ += [f"{i}qt = nl.ndarray(({min(D, TK)}, {nd} * lt), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i}for i in range({nd}):",
         f"{i}  dt = min({TK}, {D} - i * {TK})"]
  if two:   # one DMA per r1 run of the tile's rows (r = r1 * NR2 + r2 sits at r1 * rs[0] + r2 * sql)
    L_ += [f"{i}  for r1 in range(l0 // {NR2}, (l0 + lt - 1) // {NR2} + 1):",
           f"{i}    ra = max(l0, r1 * {NR2})",
           f"{i}    re = min(l0 + lt, r1 * {NR2} + {NR2})",
           f"{i}    nisa.dma_copy(dst=qt[0:dt, i * lt + ra - l0:i * lt + re - l0], src=data1.ap(pattern=[[{sqd}, dt], [{sql}, re - ra]], "
           f"offset={oq} + b1 * {sqb1} + b2 * {sqb2} + r1 * {sp.rs[0]} + (ra - r1 * {NR2}) * {sql} + i * {TK * sqd}))"]
  else: L_ += _dma_in("qt", "i * lt:i * lt + lt", "data1", sqd, "dt", sql, "lt", f"{qo} + i * {TK * sqd}", i + "  ")
  L_ += [f"{i}mr = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i}nisa.memset(dst=mr, value={_fl(-FLT_MAX)})",
         f"{i}lr = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i}nisa.memset(dst=lr, value=0.0)",
         f"{i}acc = nl.ndarray((lt, {Dv}), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i}nisa.memset(dst=acc, value=0.0)"]
  if sp.mask == "pos":   # this tile's row positions, as float32 (exact below 2**24)
    L_ += [f"{i}pi = nl.ndarray((lt, 1), dtype=nl.int32, buffer=nl.sbuf)"]
    L_ += _rows_dma("pi", "data4", sp, "[1, 1]", "0", i)
    L_ += [f"{i}pf = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)", f"{i}nisa.tensor_copy(dst=pf, src=pi)"]
  if sp.mask == "keyrange":   # this tile's [lo, hi] key-id bounds, as float32 (exact below 2**24)
    for nm, off in (("lo", sp.m[2]), ("hi", sp.m[3])):
      L_ += [f"{i}{nm}i = nl.ndarray((lt, 1), dtype=nl.int32, buffer=nl.sbuf)"]
      L_ += _rows_dma(f"{nm}i", "data4", sp, "[1, 1]", str(off), i)
      L_ += [f"{i}{nm}f = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)", f"{i}nisa.tensor_copy(dst={nm}f, src={nm}i)"]
  s_end = f"max(0, min({S}, t0 + lt + {sp.coff}))" if causal else f"{S}"   # causal: keys past the tile's last visible key are skipped
  L_ += [f"{i}for s0 in range(0, {s_end}, {TS}):",
         f"{i}  st = min({TS}, {S} - s0)",
         f"{i}  ss = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.psum)",
         f"{i}  for i in range({nd}):",
         f"{i}    dt = min({TK}, {D} - i * {TK})",
         f"{i}    kt = nl.ndarray((dt, st), dtype=nl.float32, buffer=nl.sbuf)"]
  if KT_ROWS and skd == 1 and sks > 0:   # N6: 128 key rows at a time (contiguous), transposed on TensorE
    L_ += [f"{i}    for j in range(0, st, {TK}):", f"{i}      ct = min({TK}, st - j)"]
    L_ += _tload("kt", "j:j + ct", "data2", sks, "ct", "dt", f"{ok} + b1 * {skb1} + b2 * {skb2} + (s0 + j) * {sks} + i * {TK}", i + "      ", "k")
  else: L_ += _dma_in("kt", ":", "data2", skd, "dt", sks, "st", f"{ok} + b1 * {skb1} + b2 * {skb2} + s0 * {sks} + i * {TK * skd}", i + "    ")
  L_ += [f"{i}    nisa.nc_matmul(dst=ss, stationary=qt[0:dt, i * lt:i * lt + lt], moving=kt, accumulate=(i > 0))",
         f"{i}  sc = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i}  nisa.tensor_scalar(dst=sc, data=ss, op0=nl.multiply, operand0={_fl(sp.scale)})"]
  i2 = i + "  "
  if sp.mask != "none": L_ += [f"{i2}sm = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.sbuf)"]
  if causal:   # row r2 = t0 + p sees key s0 + j iff (t0 + p + coff) - (s0 + j) >= 0
    L_ += [f"{i2}nisa.affine_select(dst=sm, pattern=[[-1, st]], channel_multiplier=1, on_true_tile=sc, on_false_value=float('-inf'), "
           f"cmp_op=nl.greater_equal, offset=t0 + {sp.coff} - s0)"]
  elif sp.mask in ("tensor", "pos", "keyrange"):
    L_ += [f"{i2}mk = nl.ndarray((lt, st), dtype=nl.uint8, buffer=nl.sbuf)"]
    if sp.mask == "keyrange":   # the block's key ids, the same row on every partition (a partition-step-0 DMA)
      L_ += [f"{i2}kk = nl.ndarray((lt, st), dtype=nl.int32, buffer=nl.sbuf)",
             f"{i2}nisa.dma_copy(dst=kk, src=data4.ap(pattern=[[0, lt], [1, st]], offset={sp.m[4]} + s0))",
             f"{i2}ki = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.sbuf)",
             f"{i2}nisa.tensor_copy(dst=ki, src=kk)",
             f"{i2}k1 = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.sbuf)",
             f"{i2}nisa.tensor_scalar(dst=k1, data=ki, op0=nl.greater_equal, operand0=lof)",
             f"{i2}k2 = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.sbuf)",
             f"{i2}nisa.tensor_scalar(dst=k2, data=ki, op0=nl.less_equal, operand0=hif)",
             f"{i2}nisa.tensor_tensor(dst=mk, data1=k1, data2=k2, op=nl.multiply)"]
    elif sp.mask == "tensor":
      smb1, smb2, _, _, sms, om = sp.m
      L_ += _rows_dma("mk", "data4", sp, f"[{sms}, st]", f"{om} + b1 * {smb1} + b2 * {smb2} + s0 * {sms}", i2)
    else:
      L_ += [f"{i2}ki = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.sbuf)",
             f"{i2}nisa.iota(dst=ki, pattern=[[1, st]], offset=s0, channel_multiplier=0)",
             f"{i2}nisa.tensor_scalar(dst=mk, data=ki, op0=nl.less_equal, operand0=pf)"]
    L_ += [f"{i2}nisa.memset(dst=sm, value=float('-inf'))",
           f"{i2}nisa.tensor_copy_predicated(dst=sm, src=sc, predicate=mk)"]
  s = "sm" if sp.mask != "none" else "sc"
  L_ += [f"{i2}bm = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i2}nisa.tensor_reduce(dst=bm, op=nl.maximum, data={s}, axis=1)",
         f"{i2}mn = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i2}nisa.tensor_tensor(dst=mn, data1=mr, data2=bm, op=nl.maximum)",
         f"{i2}ng = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i2}nisa.tensor_scalar(dst=ng, data=mn, op0=nl.multiply, operand0=-1.0)",
         f"{i2}al = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i2}p = nl.ndarray((lt, st), dtype=nl.float32, buffer=nl.sbuf)"]
  L_ += _exp(["al", "p"], ["mr", s], ["(lt, 1)", "(lt, st)"], i2)
  L_ += [
         f"{i2}rs = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i2}nisa.tensor_reduce(dst=rs, op=nl.add, data=p, axis=1)",
         f"{i2}nisa.tensor_tensor(dst=lr, data1=lr, data2=al, op=nl.multiply)",
         f"{i2}nisa.tensor_tensor(dst=lr, data1=lr, data2=rs, op=nl.add)",
         f"{i2}nisa.tensor_scalar(dst=acc, data=acc, op0=nl.multiply, operand0=al)",
         f"{i2}nisa.tensor_copy(dst=mr, src=mn)",
         f"{i2}for c0 in range(0, {Dv}, {TN}):",
         f"{i2}  cw = min({TN}, {Dv} - c0)",
         f"{i2}  op = nl.ndarray((lt, cw), dtype=nl.float32, buffer=nl.psum)",
         f"{i2}  for j in range(0, st, {TK}):",
         f"{i2}    ct = min({TK}, st - j)",
         f"{i2}    pp = nl.ndarray((ct, lt), dtype=nl.float32, buffer=nl.psum)",
         f"{i2}    nisa.nc_transpose(dst=pp, data=p[:, j:j + ct])",
         f"{i2}    pt = nl.ndarray((ct, lt), dtype=nl.float32, buffer=nl.sbuf)",
         f"{i2}    nisa.tensor_copy(dst=pt, src=pp)",
         f"{i2}    vt = nl.ndarray((ct, cw), dtype=nl.float32, buffer=nl.sbuf)"]
  L_ += _dma_in("vt", ":", "data3", svs, "ct", svd, "cw", f"{ov} + b1 * {svb1} + b2 * {svb2} + (s0 + j) * {svs} + c0 * {svd}", i2 + "    ")
  L_ += [f"{i2}    nisa.nc_matmul(dst=op, stationary=pt, moving=vt, accumulate=(j > 0))",
         f"{i2}  nisa.tensor_tensor(dst=acc[:, c0:c0 + cw], data1=acc[:, c0:c0 + cw], data2=op, op=nl.add)",
         f"{i}rl = nl.ndarray((lt, 1), dtype=nl.float32, buffer=nl.sbuf)"]
  if sp.mask == "keyrange":   # no visible key: lr = 0 and acc = 0; 0 * (1 / FLT_MIN) = 0 instead of 0 * inf = NaN (lr >= 1 otherwise)
    L_ += [f"{i}nisa.tensor_scalar(dst=lr, data=lr, op0=nl.maximum, operand0={nki.FLT_MIN_!r})"]
  L_ += [f"{i}nisa.reciprocal(dst=rl, data=lr)",
         f"{i}nisa.tensor_scalar(dst=acc, data=acc, op0=nl.multiply, operand0=rl)"]
  if two:
    L_ += [f"{i}for r1 in range(l0 // {NR2}, (l0 + lt - 1) // {NR2} + 1):",
           f"{i}  ra = max(l0, r1 * {NR2})",
           f"{i}  re = min(l0 + lt, r1 * {NR2} + {NR2})",
           f"{i}  nisa.dma_copy(dst=data0.ap(pattern=[[{sol}, re - ra], [{sod}, {Dv}]], offset={oo} + b1 * {sob1} + b2 * {sob2} + "
           f"r1 * {sp.rs[1]} + (ra - r1 * {NR2}) * {sol}), src=acc[ra - l0:re - l0, :])"]
  else:
    L_ += [f"{i}nisa.dma_copy(dst=data0.ap(pattern=[[{sol}, lt], [{sod}, {Dv}]], offset={oo} + b1 * {sob1} + b2 * {sob2} + l0 * {sol}), src=acc)"]
  L_ += ["  return data0"]
  dts = {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.float32, 3: dtypes.float32}
  if n_in == 5: dts[4] = dtypes.bool if sp.mask == "tensor" else dtypes.int32
  if sp.kv_one:   # one argument for K and V (a buffer bound to two NEFF inputs is avoided): V reads data2, the mask moves to data3
    L_ = [ln.replace("data3", "data2").replace("data4", "data3") for ln in L_]
    L_[0] = f"def {sp.name}({', '.join(f'data{i}' for i in range(n_in - 1))}):"
    dts = {i: dts[i + (i >= 3)] for i in range(n_in - 1)}
    n_in -= 1
  return _program(sp.name, L_, n_in, dts)

def _exp(dsts: list[str], xs: list[str], shapes: list[str], ind: str) -> list[str]:
  """dst = exp(x + ng) for each (dst, x, tile shape): one ScalarE activation with ng as its (P, 1) bias, or (default, N5) the
  exact VectorE exp of fusezap.codegen.nki.exact_exp (ScalarE exp is ~1e-5 relative on inf2, N4 run 2)."""
  if not nki.EXACT: return [f"{ind}nisa.activation(dst={d}, op=nl.exp, data={x}, bias=ng, scale=1.0)" for d, x in zip(dsts, xs)]
  out: list[str] = []
  for j, (d, x, shape) in enumerate(zip(dsts, xs, shapes)):
    k = iter(range(10 ** 6))
    def new(dt, j=j, shape=shape, k=k) -> str:
      name = f"ex{j}_{next(k)}"
      if dt is not None: out.append(f"{ind}{name} = nl.ndarray({shape}, dtype=nl.{nki.nl_dtype(dt)}, buffer=nl.sbuf)")
      return name
    xb = new(dtypes.float32)
    out.append(f"{ind}nisa.tensor_scalar(dst={xb}, data={x}, op0=nl.add, operand0=ng)")
    nki.exact_exp(lambda ln: out.append(ind + ln), new, d, xb, base2=False)
  return out

def _rows_dma(tile: str, src: str, sp: AttnSpec, fpat: str, base: str, ind: str) -> list[str]:
  """tile[0:lt] <- the rows l0 .. l0+lt-1 of a per-row operand (mask rows or positions) addressed r1 * s1 + r2 * s2: one DMA
  per run of rows inside one r1 (one DMA in all when the two row levels form one affine stride)."""
  (NR1, NR2), (s1, s2) = sp.nr, (sp.m[2], sp.m[3]) if sp.mask == "tensor" else sp.m[:2]
  if min(s1, s2) < 0: raise NKIUnsupported(f"{sp.name}: negative mask/position row strides")
  if NR1 == 1 or s1 == s2 * NR2:
    return [f"{ind}nisa.dma_copy(dst={tile}, src={src}.ap(pattern=[[{s2}, lt], {fpat}], offset={base} + l0 * {s2}))"]
  if NR2 == 1:
    return [f"{ind}nisa.dma_copy(dst={tile}, src={src}.ap(pattern=[[{s1}, lt], {fpat}], offset={base} + l0 * {s1}))"]
  return [f"{ind}for r1 in range(l0 // {NR2}, (l0 + lt - 1) // {NR2} + 1):",
          f"{ind}  ra = max(l0, r1 * {NR2})",
          f"{ind}  re = min(l0 + lt, r1 * {NR2} + {NR2})",
          f"{ind}  nisa.dma_copy(dst={tile}[ra - l0:re - l0, :], src={src}.ap(pattern=[[{s2}, re - ra], {fpat}], "
          f"offset={base} + r1 * {s1} + (ra - r1 * {NR2}) * {s2}))"]

# ---------------------------------------------------------------- RMSNorm with the residual add (N5 serving)

@dataclass(frozen=True)
class NormSpec:
  """h = w * (xo * rsqrt(mean(xo * xo) + eps)) over rows of D, xo = x + d (add) or x. The composed graph is two kernels (the
  row sum, then the elementwise apply, which re-read every earlier residual term while the stream stayed lazy); this is one
  launch that also stores the residual stream xo. The same instructions as the renderer's composed kernels in the same
  order (chunks of `fmax` columns summed into a (P, 1) accumulator from 0.0, mean = sum * float32(1/D), the exact sqrt
  and reciprocal), so the same bits."""
  N: int
  D: int
  eps: float
  add: bool
  fmax: int = 512
  @property
  def name(self) -> str: return f"fz_rmsnorm{'_add' if self.add else ''}_{self.N}_{self.D}"
  def render(self) -> NKIProgram: return render_rmsnorm(self)

def render_rmsnorm(sp: NormSpec) -> NKIProgram:
  """`fz_rmsnorm_add(data0=h, data1=xo, data2=x, data3=d, data4=w)` or `fz_rmsnorm(data0=h, data1=x, data2=w)`, float32."""
  N, D = sp.N, sp.D
  if min(N, D) <= 0: raise NKIUnsupported(f"{sp.name}: empty")
  x, w = ("data2", "data4") if sp.add else ("data1", "data2")
  rows = lambda t: f"{t}.ap(pattern=[[{D}, pn], [1, {D}]], offset=p0 * {D})"
  L = [f"def {sp.name}({', '.join(f'data{i}' for i in range(5 if sp.add else 3))}):",
       f"  wt = nl.ndarray(({min(N, TM)}, {D}), dtype=nl.float32, buffer=nl.sbuf)",
       f"  nisa.dma_copy(dst=wt, src={w}.ap(pattern=[[0, {min(N, TM)}], [1, {D}]], offset=0))",
       f"  for p0 in range(0, {N}, {TM}):",
       f"    pn = min({TM}, {N} - p0)",
       f"    xt = nl.ndarray((pn, {D}), dtype=nl.float32, buffer=nl.sbuf)",
       f"    nisa.dma_copy(dst=xt, src={rows(x)})"]
  if sp.add:
    L += [f"    dt = nl.ndarray((pn, {D}), dtype=nl.float32, buffer=nl.sbuf)",
          f"    nisa.dma_copy(dst=dt, src={rows('data3')})",
          f"    nisa.tensor_tensor(dst=xt, data1=xt, data2=dt, op=nl.add)",
          f"    nisa.dma_copy(dst={rows('data1')}, src=xt)"]
  L += _norm_rows("    ", "xt", "wt[0:pn, :]", "pn", D, sp.eps, sp.fmax)
  L += [f"    nisa.dma_copy(dst={rows('data0')}, src=yt)",
        f"  return {'data0, data1' if sp.add else 'data0'}"]
  n = 5 if sp.add else 3
  return NKIProgram(sp.name, HEADER + "\n".join(L) + "\n", n, (0, 1) if sp.add else (0,), {i: dtypes.float32 for i in range(n)})

def _norm_rows(ind: str, xt: str, wt: str, pn: str, D: int, eps: float, fmax: int, tag: str = "", out: str | None = None) -> list[str]:
  """yt<tag> = wt * (xt * rsqrt(mean(xt * xt) + eps)) over the rows of an SBUF tile xt (pn, D): NormSpec's instructions in
  NormSpec's order (shared by render_rmsnorm and render_layer, so both give the same bits). out: write into this existing
  tile instead of declaring yt<tag> (N8 tiled layer)."""
  t = lambda s: s + tag
  L = [f"{ind}{t('acc')} = nl.ndarray(({pn}, 1), dtype=nl.float32, buffer=nl.sbuf)",
       f"{ind}nisa.memset(dst={t('acc')}, value=0.0)"]
  for c0 in range(0, D, fmax):
    cw = min(fmax, D - c0)
    L += [f"{ind}{t('sq')} = nl.ndarray(({pn}, {cw}), dtype=nl.float32, buffer=nl.sbuf)",
          f"{ind}nisa.tensor_tensor(dst={t('sq')}, data1={xt}[:, {c0}:{c0 + cw}], data2={xt}[:, {c0}:{c0 + cw}], op=nl.multiply)",
          f"{ind}{t('part')} = nl.ndarray(({pn}, 1), dtype=nl.float32, buffer=nl.sbuf)",
          f"{ind}nisa.tensor_reduce(dst={t('part')}, op=nl.add, data={t('sq')}, axis=1, keepdims=True)",
          f"{ind}nisa.tensor_tensor(dst={t('acc')}, data1={t('acc')}, data2={t('part')}, op=nl.add)"]
  L += [f"{ind}nisa.tensor_scalar(dst={t('acc')}, data={t('acc')}, op0=nl.multiply, operand0={float(nki.np.float32(1.0 / D))!r})",
        f"{ind}nisa.tensor_scalar(dst={t('acc')}, data={t('acc')}, op0=nl.add, operand0={float(nki.np.float32(eps))!r})"]
  k = iter(range(10 ** 6))
  def new(dt) -> str:
    name = f"rs{next(k)}{tag}"
    if dt is not None: L.append(f"{ind}{name} = nl.ndarray(({pn}, 1), dtype=nl.{nki.nl_dtype(dt)}, buffer=nl.sbuf)")
    return name
  rt = new(dtypes.float32)
  if nki.EXACT: nki.refined_sqrt(lambda ln: L.append(ind + ln), new, rt, t("acc"))
  else: L.append(f"{ind}nisa.activation(dst={rt}, op=nl.sqrt, data={t('acc')})")
  y = out or t("yt")
  L += [f"{ind}nisa.reciprocal(dst={rt}, data={rt})"] + ([] if out else [f"{ind}{y} = nl.ndarray(({pn}, {D}), dtype=nl.float32, buffer=nl.sbuf)"])
  L += [f"{ind}nisa.tensor_scalar(dst={y}, data={xt}, op0=nl.multiply, operand0={rt})",
        f"{ind}nisa.tensor_tensor(dst={y}, data1={wt}, data2={y}, op=nl.multiply)"]
  return L

# ---------------------------------------------------------------- paged KV write without indirect DMA (N5 serving)

@dataclass(frozen=True)
class KVWriteSpec:
  """pool[p, :] = X[t, xoff:xoff + C2] for the one t with WT[t, p] == 1, else unchanged: pool (P, C2), WT (N, P) the step's
  one-hot slot matrix (float32 0/1), X (N, xs) rows holding [k | v] at xoff and a 1.0 at xoff + C2. Per 128 pool rows:
  hit = WT[:, rows]ᵀ · X[:, xoff + C2] and the rows' new values = WT[:, rows]ᵀ · X[:, xoff:xoff + C2] on TensorE (exact:
  one 1.0 x value product plus zeros per element), then a predicated copy into the loaded pool rows and a store back.
  One launch per layer (fusezap/serve/neuron_kv.py); the composed form was a one-hot matmul plus a WHERE kernel."""
  P: int
  N: int
  C2: int
  xs: int
  xoff: int
  @property
  def name(self) -> str: return f"fz_kv_write_{self.P}_{self.N}_{self.C2}"
  def render(self) -> NKIProgram: return render_kv_write(self)

def render_kv_write(sp: KVWriteSpec) -> NKIProgram:
  """`fz_kv_write(data0=pool (in/out), data1=WT, data2=X)`, float32."""
  P, N, C2, xs, xo = sp.P, sp.N, sp.C2, sp.xs, sp.xoff
  if min(P, N, C2) <= 0: raise NKIUnsupported(f"{sp.name}: empty")
  nk = -(-N // TK)
  L = [f"def {sp.name}(data0, data1, data2):",
       f"  xt = nl.ndarray(({min(N, TK)}, {nk * (C2 + 1)}), dtype=nl.float32, buffer=nl.sbuf)"]
  for i in range(nk):   # X's [k | v | 1] columns, K-chunk i at columns i*(C2+1): moving tiles, loaded once
    kt = min(TK, N - i * TK)
    L += [f"  nisa.dma_copy(dst=xt[0:{kt}, {i * (C2 + 1)}:{(i + 1) * (C2 + 1)}], src=data2.ap(pattern=[[{xs}, {kt}], [1, {C2 + 1}]], offset={i * TK * xs + xo}))"]
  L += [f"  for p0 in range(0, {P}, {TM}):",
        f"    pt = min({TM}, {P} - p0)",
        f"    wt = nl.ndarray(({min(N, TK)}, {nk} * pt), dtype=nl.float32, buffer=nl.sbuf)"]
  for i in range(nk):
    kt = min(TK, N - i * TK)
    L += [f"    nisa.dma_copy(dst=wt[0:{kt}, {i} * pt:{i + 1} * pt], src=data1.ap(pattern=[[{P}, {kt}], [1, pt]], offset={i * TK * P} + p0))"]
  def mm(dst: str, c0: int, cw: int) -> list[str]:
    out = []
    for i in range(nk):
      kt = min(TK, N - i * TK)
      out.append(f"    nisa.nc_matmul(dst={dst}, stationary=wt[0:{kt}, {i} * pt:{i + 1} * pt], moving=xt[0:{kt}, {i * (C2 + 1) + c0}:{i * (C2 + 1) + c0 + cw}], accumulate={i > 0})")
    return out
  L += [f"    hp = nl.ndarray((pt, 1), dtype=nl.float32, buffer=nl.psum)"] + mm("hp", C2, 1)
  L += [f"    hit = nl.ndarray((pt, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.tensor_copy(dst=hit, src=hp)",
        f"    pool = nl.ndarray((pt, {C2}), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.dma_copy(dst=pool, src=data0.ap(pattern=[[{C2}, pt], [1, {C2}]], offset=p0 * {C2}))"]
  for c0 in range(0, C2, TN):
    cw = min(TN, C2 - c0)
    L += [f"    dp = nl.ndarray((pt, {cw}), dtype=nl.float32, buffer=nl.psum)"] + mm("dp", c0, cw)
    L += [f"    dv = nl.ndarray((pt, {cw}), dtype=nl.float32, buffer=nl.sbuf)",
          f"    nisa.tensor_copy(dst=dv, src=dp)",
          f"    hb = nl.ndarray((pt, {cw}), dtype=nl.float32, buffer=nl.sbuf)",
          f"    nisa.memset(dst=hb, value=0.0)",
          f"    nisa.tensor_scalar(dst=hb, data=hb, op0=nl.add, operand0=hit)",
          f"    pr = nl.ndarray((pt, {cw}), dtype=nl.uint8, buffer=nl.sbuf)",
          f"    nisa.tensor_scalar(dst=pr, data=hb, op0=nl.greater, operand0=0.5)",
          f"    nisa.tensor_copy_predicated(dst=pool[:, {c0}:{c0 + cw}], src=dv, predicate=pr)"]
  L += [f"    nisa.dma_copy(dst=data0.ap(pattern=[[{C2}, pt], [1, {C2}]], offset=p0 * {C2}), src=pool)",
        "  return data0"]
  return NKIProgram(sp.name, HEADER + "\n".join(L) + "\n", 3, (0,), {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.float32})

# ---------------------------------------------------------------- N7: the lm_head and the greedy argmax (serving head)
# N7 trn1 (FUSE=4, step 1): the head NEFF (final norm, lm_head 151936 x 896 float32, argmax) took 3.45 ms device per launch,
# 2.16 of it without the argmax. lm_head through render_matmul: 297 N tiles x 7 K chunks = 2079 weight DMAs of 256 KiB
# (2 KiB runs, row stride V) feeding 2079 nc_matmul, each N tile evicted and stored alone. The argmax (neuron_kv.argmax_rows
# composed: max, then where(x == max, V - index, 0) max) ran as renderer kernels with the R rows on R partitions: every
# VectorE instruction walked 151936 / 512 chunks of one row per lane (~1.3 ms). FZ_NEURON_HEAD=v2: HeadSpec
# streams a weight relaid once at load (head_tiles: per block of W columns one contiguous run per partition, all K chunks),
# and ArgmaxSpec puts the vocabulary on the 128 partitions (1187 columns each for Qwen2.5). v1: render_matmul and the
# composed argmax (N7 step 1). trn1 head_bench (device ms per launch, 1 / 4 rows): v1 3.45 / 3.54, v2 (W 1024, 4 DMAs per
# block) 2.59 / 2.63, logits heads v1 2.16 / 2.22 and v2 2.58 / 2.59: ArgmaxSpec ~0.01-0.04 ms. HeadSpec at 1 row: 1024:1
# 2.07, 512:4 2.65, 2048:4 2.86, 1024:8 3.85 (split blocks hurt), so HEAD_DMAS defaults to 1. v3 (the default after
# step 2b): render_matmul on kmajor(lm_head) (no extra 545 MB) + ArgmaxSpec. N8: v2:1024:1 is the default (trn1 step 4
# head_bench, bitwise: v2:1024:1 2.067 ms at 1 row, v3 2.171, v1 2.154; 2.130 at 4 rows). bf16 weights (I68): HeadSpec is
# float32-only, so a bf16 lm_head runs as MatmulBF16 (+ ArgmaxSpec) whatever FZ_NEURON_HEAD says.
HEAD_W = int(os.environ.get("FZ_NEURON_HEAD_W", "") or 1024)      # v2 head block: columns per weight DMA (a multiple of TN)
HEAD_DMAS = int(os.environ.get("FZ_NEURON_HEAD_DMAS", "") or 1)   # DMAs per block (partition groups); trn1: 1 is best (1024:1
                                                                  # 2.07 ms, :4 2.59, :8 3.85 at 1 row)
HEAD_SBUF = 128 * 1024   # bytes per partition for one block's weight tile (W * K / 128 * 4): wider blocks are NKIUnsupported

def head_mode() -> str:
  """FZ_NEURON_HEAD: v2 (default since N8; HeadSpec at FZ_NEURON_HEAD_W=1024, FZ_NEURON_HEAD_DMAS=1 + ArgmaxSpec), v3
  (render_matmul on kmajor(lm_head) + ArgmaxSpec), v1 (step 1), v4 (N7 step 4: HeadSpec weight-stationary + ArgmaxSpec;
  off by default: on trn1 step 4 it was NOT bitwise, and slower: 2.23-2.28 ms vs v2:1024:1's 2.067)."""
  m = (os.environ.get("FZ_NEURON_HEAD", "") or "v2").lower()
  return {"1": "v1", "0": "v1", "2": "v2", "3": "v3", "4": "v4"}.get(m, m if m in ("v1", "v2", "v3", "v4") else "v2")
def head_v2() -> bool: return head_mode() in ("v2", "v4")            # HeadSpec on head_tiles' copy of lm_head
def head_argmax() -> bool: return head_mode() in ("v2", "v3", "v4")  # ArgmaxSpec

def head_tiles(w, W: int | None = None):
  """A torch-layout lm_head (V, K) as numpy -> the flat float32 layout HeadSpec streams. Block b holds columns
  [b*W, b*W + wb) at offset b*W*K: first a (128, nkf*wb) region (partition p: chunk i's row k = i*128 + p, wb values each,
  i = 0..nkf-1, so one contiguous run per partition), then the K tail (kt, wb) when K % 128 != 0."""
  import numpy as np
  W = W or HEAD_W
  V, K = w.shape
  nkf, kt = divmod(K, TK)
  out = np.empty(V * K, np.float32)
  for b0 in range(0, V, W):
    wb = min(W, V - b0)
    blk = np.asarray(w[b0:b0 + wb], np.float32).T              # (K, wb): row k = column k of the block's weights
    o = b0 * K
    if nkf: out[o:o + nkf * TK * wb] = blk[:nkf * TK].reshape(nkf, TK, wb).transpose(1, 0, 2).reshape(-1)
    if kt: out[o + nkf * TK * wb:o + K * wb] = blk[nkf * TK:].reshape(-1)
  return out

@dataclass(frozen=True)
class HeadSpec:
  """logits (M, V) = h (M, K) @ lm_headᵀ in float32 from head_tiles' layout: the same nc_matmul as render_matmul (Aᵀ chunks
  stationary, 512-column moving tiles, PSUM accumulation over the K chunks in order), so the same bits; only the weight
  loads (one block of W columns and every K chunk per HEAD_DMAS DMAs, contiguous) and the eviction (one store per block)
  differ."""
  M: int
  V: int
  K: int
  W: int = 1024
  dmas: int = 1
  ws: bool = False   # N7 step 4 (FZ_NEURON_HEAD=v4): weight-stationary, logitsᵀ (vocab on the partitions) back to rows by transposes;
                     # off by default: NOT bitwise on trn1 (step 4)
  @property
  def name(self) -> str: return f"fz_head{'w' if self.ws else ''}_{self.M}_{self.V}_{self.K}"
  def render(self) -> NKIProgram: return render_head(self)

def render_head(sp: HeadSpec) -> NKIProgram:
  """`fz_head(data0=logits (M, V), data1=h (M, K), data2=head_tiles(lm_head, W))`, float32."""
  M, V, K, W = sp.M, sp.V, sp.K, sp.W
  if min(M, V, K) <= 0 or W % TN: raise NKIUnsupported(f"{sp.name}: empty, or W={W} not a multiple of {TN}")
  nkf, kt = divmod(K, TK)
  nk = nkf + (kt > 0)
  if nk * min(M, TM) * 4 > A_KEEP: raise NKIUnsupported(f"{sp.name}: K={K} too wide to keep Aᵀ in SBUF")
  if nk * W * 4 > HEAD_SBUF: raise NKIUnsupported(f"{sp.name}: W={W} x {nk} K chunks = {nk * W * 4} B per partition > {HEAD_SBUF}")
  a_off = f"m0 * {K} + i * {TK}"
  rows = AT_ROWS and M >= AT_ROWS                 # Aᵀ as render_matmul loads it (the same values either way)
  L = [f"def {sp.name}(data0, data1, data2):",
       f"  for m0 in range(0, {M}, {TM}):",
       f"    mt = min({TM}, {M} - m0)",
       f"    at = nl.ndarray(({min(K, TK)}, {nk} * mt), dtype=nl.float32, buffer=nl.sbuf)",
       f"    for i in range({nk}):",
       f"      kt = min({TK}, {K} - i * {TK})"]
  L += _tload("at", "i * mt:i * mt + mt", "data1", K, "mt", "kt", a_off, "      ", "a") if rows else \
       _dma_in("at", "i * mt:i * mt + mt", "data1", 1, "kt", K, "mt", a_off, "      ")
  L += [f"    for b0 in range(0, {V}, {W}):",
        f"      wb = min({W}, {V} - b0)"]
  if nkf:   # the block's full K chunks: partition p holds rows p, 128 + p, ... (nkf * wb contiguous values), in `dmas` groups
    g = -(-TK // max(1, min(sp.dmas, TK)))
    L += [f"      wt = nl.ndarray(({TK}, {nkf} * wb), dtype=nl.float32, buffer=nl.sbuf)"]
    L += [f"      nisa.dma_copy(dst=wt[{p0}:{min(TK, p0 + g)}, :], src=data2.ap(pattern=[[{nkf} * wb, {min(TK, p0 + g) - p0}], [1, {nkf} * wb]], "
          f"offset=b0 * {K} + {p0} * {nkf} * wb))" for p0 in range(0, TK, g)]
  if kt:
    L += [f"      tt = nl.ndarray(({kt}, wb), dtype=nl.float32, buffer=nl.sbuf)",
          f"      nisa.dma_copy(dst=tt, src=data2.ap(pattern=[[wb, {kt}], [1, wb]], offset=b0 * {K} + {nkf * TK} * wb))"]
  L += [f"      ot = nl.ndarray((mt, wb), dtype=nl.float32, buffer=nl.sbuf)"]
  if sp.ws:   # per 128-vocab chunk: wt's columns as the stationary, hᵀ chunk i moving, PSUM accumulation over i in order
    gq = max(1, TN // min(M, TM))   # vocab chunks per PSUM bank (mt columns each)
    L += [f"      lt = nl.ndarray(({TK}, {-(-W // TK)} * mt), dtype=nl.float32, buffer=nl.sbuf)",
          f"      for g0 in range(0, wb, {gq * TK}):",
          f"        gw = min({gq * TK}, wb - g0)",
          f"        pz = nl.ndarray(({TK}, -(-gw // {TK}) * mt), dtype=nl.float32, buffer=nl.psum)",
          f"        for c0 in range(g0, g0 + gw, {TK}):",
          f"          fw = min({TK}, wb - c0)"]
    for i in range(nk):
      st_ = f"wt[0:{TK}, {i} * wb + c0:{i} * wb + c0 + fw]" if i < nkf else "tt[:, c0:c0 + fw]"
      L += [f"          nisa.nc_matmul(dst=pz[0:fw, (c0 - g0) // {TK} * mt:(c0 - g0) // {TK} * mt + mt], stationary={st_}, "
            f"moving=at[0:{TK if i < nkf else kt}, {i} * mt:{i} * mt + mt], accumulate={i > 0})"]
    L += [f"        nisa.tensor_copy(dst=lt[:, g0 // {TK} * mt:g0 // {TK} * mt + -(-gw // {TK}) * mt], src=pz)",
          f"      for s0 in range(0, wb, {TN}):",
          f"        st = min({TN}, wb - s0)",
          f"        tq = nl.ndarray((mt, st), dtype=nl.float32, buffer=nl.psum)",
          f"        for c0 in range(s0, s0 + st, {TK}):",
          f"          fw = min({TK}, wb - c0)",
          f"          nisa.nc_transpose(dst=tq[:, c0 - s0:c0 - s0 + fw], data=lt[0:fw, c0 // {TK} * mt:c0 // {TK} * mt + mt])",
          f"        nisa.tensor_copy(dst=ot[:, s0:s0 + st], src=tq)"]
  else:
    L += [f"      for s0 in range(0, wb, {TN}):",
          f"        st = min({TN}, wb - s0)",
          f"        ps = nl.ndarray((mt, st), dtype=nl.float32, buffer=nl.psum)"]
    for i in range(nk):
      mv = f"wt[0:{TK}, {i} * wb + s0:{i} * wb + s0 + st]" if i < nkf else "tt[:, s0:s0 + st]"
      L += [f"        nisa.nc_matmul(dst=ps, stationary=at[0:{TK if i < nkf else kt}, {i} * mt:{i} * mt + mt], moving={mv}, accumulate={i > 0})"]
    L += [f"        nisa.tensor_copy(dst=ot[:, s0:s0 + st], src=ps)"]
  L += [f"      nisa.dma_copy(dst=data0.ap(pattern=[[{V}, mt], [1, wb]], offset=m0 * {V} + b0), src=ot)",
        f"  return data0"]
  return _program(sp.name, L, 3, {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.float32})

@dataclass(frozen=True)
class ArgmaxSpec:
  """idx (R,) int32 = np.argmax of each row of x (R, V) float32 (finite rows; the lowest index of the maximum, -0.0 == 0.0).
  Row r is loaded as (pp, c) tiles, c = ceil(V / 128) consecutive entries per partition (the tail partition padded with
  -FLT_MAX), so each VectorE instruction walks c values per lane instead of V: per partition the max m and, where x == m,
  the largest V - index (its first maximum); both (pp, 1) columns are transposed on TensorE (exact for finite values) and
  reduced the same way on one partition, so the lowest partition, then the lowest index in it, wins every tie."""
  R: int
  V: int
  @property
  def name(self) -> str: return f"fz_argmax_{self.R}_{self.V}"
  def render(self) -> NKIProgram: return render_argmax(self)

def render_argmax(sp: ArgmaxSpec) -> NKIProgram:
  """`fz_argmax(data0=idx (R,) int32, data1=x (R, V) float32)`."""
  R, V = sp.R, sp.V
  if min(R, V) <= 0 or V >= 1 << 24: raise NKIUnsupported(f"{sp.name}: empty, or V >= 2**24 (indices exact in float32)")
  c = -(-V // TM)
  pp, (full, rem) = -(-V // c), divmod(V, c)
  if c * 4 * 4 > 160 * 1024: raise NKIUnsupported(f"{sp.name}: {c} values per partition do not fit SBUF")
  L = [f"def {sp.name}(data0, data1):",
       f"  ri = nl.ndarray(({pp}, {c}), dtype=nl.float32, buffer=nl.sbuf)",
       f"  nisa.iota(dst=ri, pattern=[[1, {c}]], offset=0, channel_multiplier={c})",
       f"  rv = nl.ndarray(({pp}, {c}), dtype=nl.float32, buffer=nl.sbuf)",
       f"  nisa.tensor_scalar(dst=rv, data=ri, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1={float(V)!r})",
       f"  res = nl.ndarray((1, {R}), dtype=nl.float32, buffer=nl.sbuf)",
       f"  for r in range({R}):",
       f"    xt = nl.ndarray(({pp}, {c}), dtype=nl.float32, buffer=nl.sbuf)"]
  if rem: L += [f"    nisa.memset(dst=xt, value={_fl(-FLT_MAX)})"]
  if full: L += [f"    nisa.dma_copy(dst=xt[0:{full}, :], src=data1.ap(pattern=[[{c}, {full}], [1, {c}]], offset=r * {V}))"]
  if rem: L += [f"    nisa.dma_copy(dst=xt[{full}:{full + 1}, 0:{rem}], src=data1.ap(pattern=[[{c}, 1], [1, {rem}]], offset=r * {V} + {full * c}))"]
  L += [f"    pm = nl.ndarray(({pp}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.tensor_reduce(dst=pm, op=nl.maximum, data=xt, axis=1, keepdims=True)",
        f"    eq = nl.ndarray(({pp}, {c}), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.tensor_scalar(dst=eq, data=xt, op0=nl.equal, operand0=pm)",
        f"    nisa.tensor_tensor(dst=eq, data1=eq, data2=rv, op=nl.multiply)",
        f"    pr = nl.ndarray(({pp}, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.tensor_reduce(dst=pr, op=nl.maximum, data=eq, axis=1, keepdims=True)"]
  for t in ("pm", "pr"):
    L += [f"    {t}p = nl.ndarray((1, {pp}), dtype=nl.float32, buffer=nl.psum)",
          f"    nisa.nc_transpose(dst={t}p, data={t})",
          f"    {t}t = nl.ndarray((1, {pp}), dtype=nl.float32, buffer=nl.sbuf)",
          f"    nisa.tensor_copy(dst={t}t, src={t}p)"]
  L += [f"    gm = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.tensor_reduce(dst=gm, op=nl.maximum, data=pmt, axis=1, keepdims=True)",
        f"    e2 = nl.ndarray((1, {pp}), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.tensor_scalar(dst=e2, data=pmt, op0=nl.equal, operand0=gm)",
        f"    nisa.tensor_tensor(dst=e2, data1=e2, data2=prt, op=nl.multiply)",
        f"    nisa.tensor_reduce(dst=res[0:1, r:r + 1], op=nl.maximum, data=e2, axis=1, keepdims=True)",
        f"  nisa.tensor_scalar(dst=res, data=res, op0=nl.multiply, operand0=-1.0, op1=nl.add, operand1={float(V)!r})",
        f"  ix = nl.ndarray((1, {R}), dtype=nl.int32, buffer=nl.sbuf)",
        f"  nisa.tensor_copy(dst=ix, src=res)",
        f"  nisa.dma_copy(dst=data0.ap(pattern=[[{R}, 1], [1, {R}]], offset=0), src=ix)",
        f"  return data0"]
  return _program(sp.name, L, 2, {0: dtypes.int32, 1: dtypes.float32})

# ---------------------------------------------------------------- N7 step 3: a decoder layer as one kernel (serving decode)
# N7 serve on trn1 (FUSE=4, pool prefix 512, 1 row): 4 layers per NEFF took 1.56 ms device, ~0.39 ms per layer, against a
# weight-read floor of ~0.15 ms (60 MB at ~410 GB/s). A layer was ten member kernels whose intermediates went through HBM
# (scripts/nki_layer_cost.py): norm, qkv matmul, bias + RoPE (a renderer kernel), KV write, attention, o_proj, norm, gate_up,
# silu*mul (a renderer kernel), down. Attention per kv head and 512-key block ran its exact exp (44 VectorE instructions)
# on (7 rows, 512 keys) tiles: 7 of 128 lanes busy, ~18 us per head and block, ~37 us per layer at 512 slots (the measured
# 1024-vs-512 delta is 0.074 ms per layer per 512 slots). FZ_NEURON_LAYER=v2: LayerSpec, the same arithmetic in one kernel:
#   - intermediates stay in SBUF: hᵀ and the MLP activation by TensorE transposes (exact), the qkv rows, X = [q | k | v | 1],
#     the updated pool rows (attention reads K and V from them, no reload), gate/up, silu*mul (on (128, ·) tiles: features on
#     the partitions, which is also the down projection's stationary Aᵀ); X and the attention output still pass through
#     HBM once each (the qᵀ and o_proj Aᵀ loads, as before);
#   - attention: both kv heads' rows in one tile (R = Hq * N rows: the softmax instructions run once for all heads), the
#     key mask once per block for all heads, and the exact exp on the transposed block (keys on the partitions: (128, keys
#     / 128 * R) tiles instead of (R, 512)), which is also the P·V stationary; the row sums from the transposed-back block.
# Every value is computed by the same instructions in the same order as the members (same nc_matmul sequence, PSUM
# accumulation over the same K chunks, the renderer kernels' elementwise chains, NormSpec's norm), so the same bits;
# the only new instructions are exact moves (transposes of finite values, DMAs) and adds of +0.0 where the members' cat
# added zero padding (-0.0 -> +0.0 as there). Two heads' scores accumulate in one PSUM tile (x + 0.0). v1: the members.

def layer_mode() -> str:
  """FZ_NEURON_LAYER: v2 (default; LayerSpec, one kernel per decoder layer when the step fits) or v1 (the ten members)."""
  m = os.environ.get("FZ_NEURON_LAYER", "v2").lower()
  return {"1": "v1", "0": "v1", "2": "v2"}.get(m, "v1" if m == "v1" else "v2")

NEG_LOG2E = float(nki.np.float32(-1 / math.log(2)))   # silu's sigmoid: exp2(x * -log2(e)) as the renderer emits it

# N7 step 4 (trn1 step 3: v2s 0.318 ms per layer at 1 row / 512 slots; the step 3 cost model's 0.218 was v1-calibrated, its
# uncalibrated schedule said 0.306): two more LayerSpec options, each behind its own switch.
#   early (FZ_NEURON_LAYER_EARLY=1): the same instructions on the same values, reordered and with exact moves instead of HBM
#     round trips: every input-only load (weights for norms and biases, x, d, cos / sin, the slot one-hot, the pool rows, the
#     masks' key ids and every key block's mask) is issued at the top, off the serial norm -> qkv -> RoPE -> KV write ->
#     attention -> o_proj chain; qᵀ comes from X by TensorE transposes (attention rows g-major inside a kv head: row g*N + t,
#     so a q head's tokens are one transpose; rows are independent, so the order changes no bit) instead of X -> HBM -> qᵀ;
#     o_proj's Aᵀ comes from the attention output by one transpose per kv-head group and 32-partition-aligned copies instead of
#     o -> HBM -> Aᵀ (hd in {32, 64, 128}); the first key block skips the online-softmax rescale (lr and acc start at +0.0, so
#     lr * al + rs == rs + 0.0 and acc * al + o == o + 0.0 for any finite al: the exp of the old max is not needed). The X and o
#     scratch buffers are still stored (tests and the fallback read them), off the critical path. Bitwise identical by
#     construction (transposes of finite values are exact: step 3 measured hᵀ and the MLP activation that way on trn1).
#   mm (FZ_NEURON_LAYER_MM=ws): gate_up and down weight-stationary: each 128-column slice of a K-major weight tile is the
#     stationary and hᵀ / the activation (N columns) the moving operand, so gᵀ, uᵀ come out of PSUM in the layout silu * mul
#     and down want (no 2 * I / 128 transposes) and down's dnᵀ goes back to rows by D / 128 transposes. The same products
#     and the same PSUM accumulation over the same K chunks in order; bitwise only if TensorE's float32 matmul is symmetric in
#     its operands (unknown: NKISHIM's np.matmul is for >= 2 rows; scripts/nki_n7_probe.py times both forms), so it is a
#     separate switch, never a default, and the step 4 job checks its bits and tokens.
# trn1 step 4 (docs/status/nki-n7-step4-trn1): early is bitwise and fastest (v2se 0.314 ms per layer at 1 row / 512 slots,
# v2s 0.317, v1 0.396), so since N8 the default layer is v2se (FZ_NEURON_LAYER_HEADS=stack, FZ_NEURON_LAYER_EARLY=1). The
# weight-stationary forms (mm=ws: v2sw, v2swe, v2swep; and the head's v4) were NOT bitwise on the device, and v2swep (pf=48)
# was wrong (device kernel max |diff| 142; full model 0/24 prompts token-exact, 29/1441 tokens): a bug, not float drift.
# They stay available behind their switches, default off.

@dataclass(frozen=True)
class LayerSpec:
  """One decoder layer of the serving forward (Llama.forward_tokens on NeuronKVCache): x (N, D) residual, d (N, D) the last
  layer's down projection (add) -> xo = x + d + o_proj(attention(...)), dn = down(silu(gate) * up) of norm2(xo); the pool
  (P slots, rows [k | v]) gets the step's k / v at its slots (the one-hot WT (N, P)); meta: lo (N), hi (N), kid (P) at
  base (NeuronKVCache's keyrange). Weights K-major (neuron.kmajor). Rows R = Hq * N of attention must fit 128 partitions."""
  N: int
  D: int
  Hq: int
  Hkv: int
  hd: int
  I: int
  P: int
  eps: float
  scale: float
  base: int
  add: bool = True
  qkv_bias: bool = True
  o_bias: bool = False
  heads: str = "each"   # attention: "each" kv head on its own rows (the mask shared), "stack": all heads' rows in one tile
  rope: str = "slice"   # RoPE: "slice" per-head column slices, "ap": 3-D access patterns over every head at once
  mm: str = "mv"        # gate_up / down: "mv" (render_matmul's form: hᵀ stationary, weight tiles moving) or "ws" (weight-stationary:
                        # NOT bitwise on trn1 step 4, off by default)
  early: bool = False   # input-only loads first, qᵀ / oᵀ on chip, the first key block without the rescale (N7 step 4)
  pf: int = 0           # mm=ws and early: this many gate weight tiles (128 x 512) loaded at the top, during attention (N7 step 4;
                        # v2swep (pf=48) gave wrong tokens on trn1, 0/24 prompts: a bug; off by default)
  fmax: int = 512
  tile: bool = False    # N8 (I71): the rows tiled for wide layers (Qwen2.5-7B), see render_layer's "tile" notes; FZ_NEURON_LAYER_TILE
  wcols: int = 512      # N8 tile: weight columns per DMA (a multiple of 512; FZ_NEURON_LAYER_WCOLS): longer runs per partition, same bits
  part: str = ""        # N8 TP (I72): "" the whole layer; "attn" steps 1-6 up to o_proj's (partial) product; "mlp" from the bias /
                        # residual add after o_proj's all-reduce through down's (partial) product. A tensor-parallel shard runs the
                        # two halves around its two all-reduces; the instructions are the whole layer's, cut at o_proj's output.
                        # "tp" (N8 I72 follow-up, FZ_NEURON_TP_LAYER=fused): the whole layer with both all-reduces inside it as
                        # nki.collectives.all_reduce (o_proj's product before the bias, down's product before its store), over
                        # tpw ranks: one kernel (one NEFF, fusable across layers) per shard and layer, no host in the path
  tpw: int = 0          # part "tp": the tensor-parallel world (the collectives' ReplicaGroup [0 .. tpw-1])
  @property
  def name(self) -> str:
    v = ("" if self.heads == "each" else "s") + ("" if self.rope == "slice" else "a") + ("w" if self.mm == "ws" else "") + ("e" if self.early else "") + \
        (f"p{self.pf}" if self.pf and self.early and self.mm == "ws" else "") + \
        (("t" + (str(self.wcols // TN) if self.wcols != TN else "")) if self.tile else "")
    pt = f"_tp{self.tpw}" if self.part == "tp" else ("_" + self.part if self.part else "")
    return f"fz_layer{v}{'_add' if self.add else ''}{pt}_{self.N}_{self.P}_{self.D}_{self.I}"
  def args(self) -> tuple[str, ...]:
    """Argument names in order: the written buffers first (xo, dn, the pool, X and o scratch), then the inputs.
    part attn: xo = x + d (the residual), po = o_proj's product (no bias); part mlp: x = that residual, o = the reduced o_proj
    product, xo = x + o (+ bo), dn = down's product."""
    if self.part == "attn":
      return tuple(["xo", "po", "pool", "xs", "os", "x"] + (["d"] if self.add else []) + ["ln1", "wqkv"] + (["bqkv"] if self.qkv_bias else [])
                   + ["cos", "sin", "wt", "meta", "wo"])
    if self.part == "mlp": return tuple(["xo", "dn", "x", "o"] + (["bo"] if self.o_bias else []) + ["ln2", "wgu", "wd"])
    return tuple(["xo", "dn", "pool", "xs", "os", "x"] + (["d"] if self.add else []) + ["ln1", "wqkv"] + (["bqkv"] if self.qkv_bias else [])
                 + ["cos", "sin", "wt", "meta", "wo"] + (["bo"] if self.o_bias else []) + ["ln2", "wgu", "wd"])
  def outs(self) -> tuple[int, ...]: return (0, 1) if self.part == "mlp" else (0, 1, 2, 3, 4)
  def render(self) -> NKIProgram: return render_layer(self)

LAYER_ENV = ("FZ_NEURON_LAYER", "FZ_NEURON_LAYER_HEADS", "FZ_NEURON_LAYER_ROPE", "FZ_NEURON_LAYER_MM", "FZ_NEURON_LAYER_EARLY", "FZ_NEURON_LAYER_PF",
             "FZ_NEURON_LAYER_TILE", "FZ_NEURON_LAYER_WCOLS")

def _lv(heads: str, rope: str = "slice", mm: str = "mv", early: bool = False, pf: int = 0, tile: str = "auto", wcols: int = 0) -> dict:
  return {"FZ_NEURON_LAYER": "v2", "FZ_NEURON_LAYER_HEADS": heads, "FZ_NEURON_LAYER_ROPE": rope, "FZ_NEURON_LAYER_MM": mm,
          "FZ_NEURON_LAYER_EARLY": "1" if early else "0", "FZ_NEURON_LAYER_PF": str(pf), "FZ_NEURON_LAYER_TILE": tile,
          "FZ_NEURON_LAYER_WCOLS": str(wcols) if wcols else ""}

# the layer paths a hardware job compares (scripts/nki_n7_layer_bench.py, nki_layer_cost.py): name -> env switches. Letters
# after v2: s = kv heads stacked, a = RoPE by 3-D access patterns, w = weight-stationary MLP, e = early, p = the first 48 gate
# weight tiles prefetched during attention (12.6 MB; with w the MLP is DMA-bound by the model) (N7 step 4)
LAYER_VARIANTS = {"v1": {"FZ_NEURON_LAYER": "v1"}, "v2": _lv("each"), "v2a": _lv("each", "ap"), "v2s": _lv("stack"), "v2sa": _lv("stack", "ap"),
                  "v2e": _lv("each", early=True), "v2se": _lv("stack", early=True), "v2sw": _lv("stack", mm="ws"),
                  "v2swe": _lv("stack", mm="ws", early=True), "v2swep": _lv("stack", mm="ws", early=True, pf=48),
                  "v2set": _lv("stack", early=True, tile="1", wcols=512), "v2et": _lv("each", early=True, tile="1", wcols=512),
                  "v2set2": _lv("stack", early=True, tile="1", wcols=1024), "v2set4": _lv("stack", early=True, tile="1", wcols=2048)}
# t: N8 tiled rows (7B), t2 / t4: 1024 / 2048 weight columns per DMA
LAYER_BITWISE = {v for v in LAYER_VARIANTS if "w" not in v[2:]}   # by construction; "w" needs TensorE's operand symmetry

LAYER_WCOLS = 2048   # N8: the tiled layer's default weight columns per DMA (FZ_NEURON_LAYER_WCOLS; see _wide_block)

def layer_variant() -> dict:
  """LayerSpec's options from the env (defaults since N8: trn1 step 4's v2se): heads (FZ_NEURON_LAYER_HEADS stack | each),
  rope (FZ_NEURON_LAYER_ROPE slice | ap), mm (FZ_NEURON_LAYER_MM mv | ws: NOT bitwise on trn1), early (FZ_NEURON_LAYER_EARLY
  1 | 0), pf (FZ_NEURON_LAYER_PF: gate tiles prefetched, with ws and early; 0: v2swep (48) gave wrong tokens on trn1), tile
  (FZ_NEURON_LAYER_TILE auto | 1 | 0, N8: auto tiles a layer whose untiled rows do not fit SBUF (Qwen2.5-7B), 1 every layer,
  0 never: such a layer runs as the members, I69's path)."""
  g = lambda k, d: os.environ.get(k, "") or d
  return {"heads": g("FZ_NEURON_LAYER_HEADS", "stack"), "rope": g("FZ_NEURON_LAYER_ROPE", "slice"), "mm": g("FZ_NEURON_LAYER_MM", "mv"),
          "early": g("FZ_NEURON_LAYER_EARLY", "1") not in ("", "0"), "pf": int(g("FZ_NEURON_LAYER_PF", "0")),
          "tile": {"1": "1", "on": "1", "0": "0", "off": "0"}.get(g("FZ_NEURON_LAYER_TILE", "auto").lower(), "auto"),
          "wcols": int(g("FZ_NEURON_LAYER_WCOLS", str(LAYER_WCOLS)))}

def layer_ok(N: int, Hq: int, Hkv: int, hd: int, heads: str = "each") -> bool:
  """LayerSpec covers the step: exact exp, a kv head's attention rows (Hq/Hkv * N) on the partitions."""
  if not (nki.EXACT and 0 < N <= TM and 0 < Hkv and Hq % Hkv == 0 and hd % 2 == 0 and hd <= TK): return False
  lt = Hq // Hkv * N
  return lt <= TM and (heads != "stack" or Hkv * 32 * -(-lt // 32) <= TM)

def _mm_rows(at: str, M: int, K: int, Nn: int, w: str, dst: str, tag: str) -> list[str]:
  """dst[:, 0:Nn] (SBUF, M rows) = A @ w: render_matmul's nc_matmul sequence (Aᵀ chunk i stationary at[0:kt, i*M:(i+1)*M],
  512-column moving tiles of the K-major weight w, PSUM accumulation over the K chunks in order), evicted by tensor_copy."""
  return [f"  for n0{tag} in range(0, {Nn}, {TN}):",
          f"    nt{tag} = min({TN}, {Nn} - n0{tag})",
          f"    ps{tag} = nl.ndarray(({M}, nt{tag}), dtype=nl.float32, buffer=nl.psum)",
          f"    for i{tag} in range({-(-K // TK)}):",
          f"      kt{tag} = min({TK}, {K} - i{tag} * {TK})",
          f"      bt{tag} = nl.ndarray((kt{tag}, nt{tag}), dtype=nl.float32, buffer=nl.sbuf)",
          f"      nisa.dma_copy(dst=bt{tag}[0:kt{tag}, :], src={w}.ap(pattern=[[{Nn}, kt{tag}], [1, nt{tag}]], offset=i{tag} * {TK * Nn} + n0{tag}))",
          f"      nisa.nc_matmul(dst=ps{tag}, stationary={at}[0:kt{tag}, i{tag} * {M}:i{tag} * {M} + {M}], moving=bt{tag}, accumulate=(i{tag} > 0))",
          f"    nisa.tensor_copy(dst={dst}[:, n0{tag}:n0{tag} + nt{tag}], src=ps{tag})"]

def _mm_ws(mv: str, M: int, K: int, w: str, ws: int, c_off: int, F: int, dst: str, tag: str, pre: list | None = None, npf: int = 0) -> list[str]:
  """dst (min(F, 128), ceil(F/128) * M) = the transposed product: chunk c (output features c*128 ..) at dst[0:fw, c*M:(c+1)*M]
  = (A @ w[:, c_off + c*128 ..])ᵀ, weight-stationary: each 128-column slice of a K-major weight tile (row stride ws, up to
  4 chunks = 512 columns per DMA, as _mm_rows loads them) is the stationary, Aᵀ chunk i (mv[0:kt, i*M:(i+1)*M]) the moving,
  PSUM accumulation over the K chunks in order (the same as _mm_rows per output element); one PSUM bank per TN // M chunks.
  pre / npf: the first npf weight tiles are loaded into pre (the kernel's top) instead, as tiles pf<tag><k>."""
  nc, nk, gq = -(-F // TK), -(-K // TK), max(1, TN // M)
  k = 0
  L = [f"  {dst} = nl.ndarray(({min(F, TK)}, {nc * M}), dtype=nl.float32, buffer=nl.sbuf)"]
  for c0 in range(0, nc, gq):
    cn = min(gq, nc - c0)
    L.append(f"  pw{tag} = nl.ndarray(({min(F, TK)}, {cn * M}), dtype=nl.float32, buffer=nl.psum)")
    for w0 in range(c0, c0 + cn, TN // TK):
      wn = min(TN // TK, c0 + cn - w0)
      cols = min(wn * TK, F - w0 * TK)
      for i in range(nk):
        kt = min(TK, K - i * TK)
        bw = f"pf{tag}{k}" if pre is not None and k < npf else f"bw{tag}"
        (pre if bw != f"bw{tag}" else L).extend([f"  {bw} = nl.ndarray(({kt}, {cols}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  nisa.dma_copy(dst={bw}, src={w}.ap(pattern=[[{ws}, {kt}], [1, {cols}]], offset={i * TK * ws + c_off + w0 * TK}))"])
        k += 1
        for j in range(wn):
          c = w0 + j
          fw = min(TK, F - c * TK)
          L.append(f"  nisa.nc_matmul(dst=pw{tag}[0:{fw}, {(c - c0) * M}:{(c - c0 + 1) * M}], stationary={bw}[0:{kt}, {j * TK}:{j * TK + fw}], "
                   f"moving={mv}[0:{kt}, {i * M}:{(i + 1) * M}], accumulate={i > 0})")
    L.append(f"  nisa.tensor_copy(dst={dst}[:, {c0 * M}:{(c0 + cn) * M}], src=pw{tag})")
  return L

def _ws_tiles(K: int, F: int) -> int: return -(-K // TK) * -(-F // TN)   # _mm_ws's weight tiles (one per K chunk and 512 columns)

def _cols_t(src: str, M: int, F: int, dst: str, tag: str) -> list[str]:
  """dst (M, F) rows = the transpose of _mm_ws's chunk layout src (min(F, 128), ceil(F/128) * M), by TensorE transposes
  into one PSUM bank per 512 columns (exact for finite values)."""
  L = []
  for s0 in range(0, F, TN):
    st = min(TN, F - s0)
    L.append(f"  tq{tag} = nl.ndarray(({M}, {st}), dtype=nl.float32, buffer=nl.psum)")
    for c in range(s0 // TK, -(-(s0 + st) // TK)):
      fw = min(TK, F - c * TK)
      L.append(f"  nisa.nc_transpose(dst=tq{tag}[:, {c * TK - s0}:{c * TK - s0 + fw}], data={src}[0:{fw}, {c * M}:{(c + 1) * M}])")
    L.append(f"  nisa.tensor_copy(dst={dst}[:, {s0}:{s0 + st}], src=tq{tag})")
  return L

def _rows_t(src: str, c0: int, M: int, K: int, dst: str, tag: str) -> list[str]:
  """dst (min(K, 128), ceil(K/128) * M) = the Aᵀ chunks of columns c0 .. c0+K of the SBUF row tile src (M rows): chunk i at
  dst[0:kt, i*M:(i+1)*M], by TensorE transposes (exact for finite values)."""
  return [f"  {dst} = nl.ndarray(({min(K, TK)}, {-(-K // TK) * M}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  for c{tag} in range({-(-K // TK)}):",
          f"    kc{tag} = min({TK}, {K} - c{tag} * {TK})",
          f"    tp{tag} = nl.ndarray((kc{tag}, {M}), dtype=nl.float32, buffer=nl.psum)",
          f"    nisa.nc_transpose(dst=tp{tag}, data={src}[:, {c0} + c{tag} * {TK}:{c0} + c{tag} * {TK} + kc{tag}])",
          f"    nisa.tensor_copy(dst={dst}[0:kc{tag}, c{tag} * {M}:c{tag} * {M} + {M}], src=tp{tag})"]

def _tiles(L: list[str], shape: str, prefix: str):
  """A `new(dtype)` for nki.exact_exp & co: fresh SBUF tiles of `shape` named <prefix><k>."""
  k = iter(range(10 ** 6))
  def new(dt) -> str:
    name = f"{prefix}{next(k)}"
    if dt is not None: L.append(f"  {name} = nl.ndarray({shape}, dtype=nl.{nki.nl_dtype(dt)}, buffer=nl.sbuf)")
    return name
  return new

def _silu_mul(T: str, g: str, u: str, dst: str, ind: str, decl: bool) -> list[str]:
  """dst = silu(g) * u on (128, ·) tiles of shape T, the renderer's elementwise chain (exp2(g * -log2 e) by nki.exact_exp, NaN
  copied through, 1 / (1 + e), g * that, times u): render_layer's, shared by the tiled MLP blocks (elementwise, so per block
  the same bits). decl: declare dst (else it is a slice of an existing tile)."""
  L = [f"{ind}sa = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)",
       f"{ind}nisa.tensor_scalar(dst=sa, data={g}, op0=nl.multiply, operand0={NEG_LOG2E!r})",
       f"{ind}se = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)"]
  k = iter(range(10 ** 6))
  def new(dt) -> str:
    name = f"sx{next(k)}"
    if dt is not None: L.append(f"{ind}{name} = nl.ndarray({T}, dtype=nl.{nki.nl_dtype(dt)}, buffer=nl.sbuf)")
    return name
  nki.exact_exp(lambda ln: L.append(ind + ln), new, "se", "sa", base2=True)
  L += [f"{ind}su = nl.ndarray({T}, dtype=nl.uint8, buffer=nl.sbuf)", f"{ind}nisa.tensor_tensor(dst=su, data1=sa, data2=sa, op=nl.not_equal)",
        f"{ind}nisa.tensor_copy_predicated(dst=se, src=sa, predicate=su)",
        f"{ind}s1 = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)", f"{ind}nisa.tensor_scalar(dst=s1, data=se, op0=nl.add, operand0=1.0)",
        f"{ind}s2 = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)", f"{ind}nisa.reciprocal(dst=s2, data=s1)",
        f"{ind}s3 = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)", f"{ind}nisa.tensor_tensor(dst=s3, data1={g}, data2=s2, op=nl.multiply)"]
  L += ([f"{ind}{dst} = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)"] if decl else []) + \
       [f"{ind}nisa.tensor_tensor(dst={dst}, data1=s3, data2={u}, op=nl.multiply)"]
  return L

def _wide_block(at: str, M: int, K: int, w: str, ws: int, off: int, c0: int, wb: int, tag: str) -> tuple[list[str], list[tuple[str, int, int]]]:
  """N8 tiled layer: columns c0 .. c0+wb of A @ w[:, off:] (K-major weight, row stride ws) as render_matmul computes them, in a
  one-iteration `for n0<tag>` loop (nki_bf16's edits look for it): per K chunk one weight DMA of (kt, wb) (wb up to
  FZ_NEURON_LAYER_WCOLS: runs of wb elements per partition instead of 512) and one nc_matmul per 512-column slice of it into
  that slice's PSUM tile ps<tag>_<j>, accumulating over the K chunks in order. Per output column the same stationary Aᵀ chunks,
  the same moving columns and the same accumulation as _mm_rows: the same bits. Returns (lines, [(psum tile, column offset in
  the block, width)]); the caller evicts each PSUM tile."""
  nk, ind = -(-K // TK), "    "
  subs = [(j, min(TN, wb - j)) for j in range(0, wb, TN)]
  L = [f"  for n0{tag} in range({c0}, {c0 + 1}):   # columns {c0} .. {c0 + wb - 1}"]
  L += [f"{ind}ps{tag}_{j // TN} = nl.ndarray(({M}, {nt}), dtype=nl.float32, buffer=nl.psum)" for j, nt in subs]
  L += [f"{ind}for i{tag} in range({nk}):",
        f"{ind}  kt{tag} = min({TK}, {K} - i{tag} * {TK})",
        f"{ind}  bt{tag} = nl.ndarray((kt{tag}, {wb}), dtype=nl.float32, buffer=nl.sbuf)",
        f"{ind}  nisa.dma_copy(dst=bt{tag}[0:kt{tag}, :], src={w}.ap(pattern=[[{ws}, kt{tag}], [1, {wb}]], offset=i{tag} * {TK * ws} + {off} + n0{tag}))"]
  L += [f"{ind}  nisa.nc_matmul(dst=ps{tag}_{j // TN}, stationary={at}[0:kt{tag}, i{tag} * {M}:i{tag} * {M} + {M}], "
        f"moving=bt{tag}[0:kt{tag}, {j}:{j + nt}], accumulate=(i{tag} > 0))" for j, nt in subs]
  return L, [(f"ps{tag}_{j // TN}", j, nt) for j, nt in subs]

def _mm_wide(at: str, M: int, K: int, Nn: int, w: str, dst: str, tag: str, WC: int) -> list[str]:
  """dst[:, 0:Nn] = A @ w (_mm_rows's result and bits) in blocks of WC columns (_wide_block)."""
  L = []
  for c0 in range(0, Nn, WC):
    blk, ps = _wide_block(at, M, K, w, Nn, 0, c0, min(WC, Nn - c0), tag)
    L += blk + [f"    nisa.tensor_copy(dst={dst}[:, {c0 + j}:{c0 + j + nt}], src={p})" for p, j, nt in ps]
  return L

def _qkv_tiles(at: str, M: int, K: int, Nq: int, w: str, bias: bool, hd: int, QK: int, WC: int = TN) -> list[str]:
  """N8 tiled qkv: y = A @ w in blocks of WC columns (_wide_block: render_matmul's matmuls per 512-column slice), each slice
  evicted to a (M, nt) tile yq that gets the bias (X[:, cols], loaded there before) and is rotated head by head into X:
  q / k heads X = yq*cos + flip(yq)*sin (Llama._rope's products and sum; the caller adds the +0.0 over X[:, 0:QK]), v heads
  X = yq + 0.0. Per element the untiled layer's instructions on the same values. Heads must not straddle a slice (hd | 512)."""
  if TN % hd: raise NKIUnsupported(f"tiled qkv: head_dim {hd} does not divide {TN}")
  h2 = hd // 2
  L = []
  for c0 in range(0, Nq, WC):
    blk, ps = _wide_block(at, M, K, w, Nq, 0, c0, min(WC, Nq - c0), "q")
    L += blk
    for p, j, nt in ps:
      s0 = c0 + j
      L += [f"    yq = nl.ndarray(({M}, {nt}), dtype=nl.float32, buffer=nl.sbuf)", f"    nisa.tensor_copy(dst=yq, src={p})"]
      if bias: L += [f"    nisa.tensor_tensor(dst=yq, data1=yq, data2=X[:, {s0}:{s0 + nt}], op=nl.add)"]
      for c in range(s0, s0 + nt, hd):
        k = c - s0
        if c < QK:
          L += [f"    xa = nl.ndarray(({M}, {hd}), dtype=nl.float32, buffer=nl.sbuf)", f"    fl = nl.ndarray(({M}, {hd}), dtype=nl.float32, buffer=nl.sbuf)",
                f"    nisa.tensor_tensor(dst=xa, data1=yq[:, {k}:{k + hd}], data2=cs, op=nl.multiply)",
                f"    nisa.tensor_tensor(dst=fl[:, 0:{h2}], data1=yq[:, {k + h2}:{k + hd}], data2=sn[:, 0:{h2}], op=nl.multiply)",
                f"    nisa.tensor_tensor(dst=fl[:, {h2}:{hd}], data1=yq[:, {k}:{k + h2}], data2=sn[:, {h2}:{hd}], op=nl.multiply)",
                f"    nisa.tensor_tensor(dst=X[:, {c}:{c + hd}], data1=xa, data2=fl, op=nl.add)"]
        else: L += [f"    nisa.tensor_scalar(dst=X[:, {c}:{c + hd}], data=yq[:, {k}:{k + hd}], op0=nl.add, operand0=0.0)"]
  return L

def _mlp_blocks(at: str, M: int, K: int, I: int, w: str, WC: int = TN) -> list[str]:
  """N8 tiled MLP front: aT (min(I, 128), ceil(I/128) * M) = silu(A @ w[:, :I]) * (A @ w[:, I:2I]), transposed (features on the
  partitions), one block of WC columns of gate and of up at a time (_wide_block: render_matmul's matmuls per 512-column slice,
  the same PSUM accumulation over the K chunks; per output column the members' bits, whatever the block width), each block
  to (128, ·) tiles by transposes (_rows_t's) and silu(gate) * up into its columns of aT (render_layer's chain, elementwise).
  So aT has the untiled layer's bits, without its (M, 2I) row tile."""
  L = [f"  aT = nl.ndarray(({min(I, TK)}, {-(-I // TK) * M}), dtype=nl.float32, buffer=nl.sbuf)"]
  for c0 in range(0, I, WC):
    wb = min(WC, I - c0)
    cb = -(-wb // TK)
    for part, off in (("g", 0), ("v", I)):   # gate block, then up block: the (M, 2I) product's columns off + c0 ..
      blk, ps = _wide_block(at, M, K, w, 2 * I, off, c0, wb, "u")
      L += blk + [f"    {part}T = nl.ndarray(({min(I, TK)}, {cb * M}), dtype=nl.float32, buffer=nl.sbuf)"]
      for p, j, nt in ps:   # each 512-column slice: PSUM -> a (M, nt) row tile -> its 128-column chunks transposed into the block's (128, ·) tile
        L += [f"    {part}b = nl.ndarray(({M}, {nt}), dtype=nl.float32, buffer=nl.sbuf)", f"    nisa.tensor_copy(dst={part}b, src={p})"]
        for c in range(-(-nt // TK)):
          kc, cc = min(TK, nt - c * TK), j // TK + c
          L += [f"    tp{part} = nl.ndarray(({kc}, {M}), dtype=nl.float32, buffer=nl.psum)",
                f"    nisa.nc_transpose(dst=tp{part}, data={part}b[:, {c * TK}:{c * TK + kc}])",
                f"    nisa.tensor_copy(dst={part}T[0:{kc}, {cc * M}:{(cc + 1) * M}], src=tp{part})"]
    a0 = c0 // TK * M
    L += _silu_mul(f"({min(I, TK)}, {cb * M})", "gT", "vT", f"aT[:, {a0}:{a0 + cb * M}]", "    ", False)
  return L

def _cc_rows(t: str, N: int, D: int, W: int, tag: str, dst: str | None = None) -> list[str]:
  """N8 TP fused layer: the (N, D) float32 SBUF tile t summed over ranks [0, W) by one nki.collectives.all_reduce, as
  AllReduceSpec's phbm form does it (the collective reads and writes private HBM: inf2 refuses IO tensors and SBUF, probe run
  1): t -> private scratch by DMA, all_reduce scratch -> scratch2, then back into t, or (dst) straight to that HBM view."""
  rg = f"nki.collectives.ReplicaGroup([{list(range(W))}])"
  return [f"  cs{tag} = nl.ndarray(({N}, {D}), dtype=nl.float32, buffer=nl.private_hbm)",
          f"  cr{tag} = nl.ndarray(({N}, {D}), dtype=nl.float32, buffer=nl.private_hbm)",
          f"  nisa.dma_copy(dst=cs{tag}, src={t})",
          f"  nki.collectives.all_reduce(srcs=[cs{tag}], dsts=[cr{tag}], replica_group={rg}, op=nl.add)",
          f"  nisa.dma_copy(dst={dst or t}, src=cr{tag})"]

def render_layer(sp: LayerSpec) -> NKIProgram:
  """LayerSpec -> `fz_layer...(data0=xo, data1=dn, data2=pool, data3=X scratch, data4=o scratch, inputs...)`, float32 (meta int32)."""
  N, D, Hq, Hkv, hd, I, P = sp.N, sp.D, sp.Hq, sp.Hkv, sp.hd, sp.I, sp.P
  if min(N, D, Hq, Hkv, hd, I, P) <= 0 or not layer_ok(N, Hq, Hkv, hd, sp.heads) or sp.heads not in ("each", "stack") or \
     sp.rope not in ("slice", "ap") or sp.mm not in ("mv", "ws"): raise NKIUnsupported(f"{sp.name}: outside LayerSpec")
  if sp.part not in ("", "attn", "mlp", "tp") or (sp.part == "tp") != (sp.tpw >= 1): raise NKIUnsupported(f"{sp.name}: part {sp.part!r}, tpw {sp.tpw}")
  tl, WC = sp.tile, sp.wcols
  if tl and (sp.mm != "mv" or sp.rope != "slice" or WC <= 0 or WC % TN or WC > 4 * TN):
    raise NKIUnsupported(f"{sp.name}: the tiled layer takes mm=mv, rope=slice and 512 / 1024 / 1536 / 2048 weight columns")
  G, C = Hq // Hkv, Hkv * hd
  C2, Hqk, h2 = 2 * C, Hq + Hkv, hd // 2
  Nq, QK, Ho = (Hq + 2 * Hkv) * hd, (Hq + Hkv) * hd, Hq * hd
  Wx, nP = Nq + hd, -(-P // TM)
  early = sp.early
  gmaj = early and N > 1                       # attention rows g-major inside a kv head (row g*N + t; N = 1: the same rows)
  oc = early and hd % 32 == 0 and TK % hd == 0   # o_proj's Aᵀ on chip: q heads at 32-partition offsets of the 128-row chunks
  A = {nm: f"data{i}" for i, nm in enumerate(sp.args())}
  rows = lambda a, w: f"{A[a]}.ap(pattern=[[{w}, {N}], [1, {w}]], offset=0)"
  bcast = lambda a, w: f"{A[a]}.ap(pattern=[[0, {N}], [1, {w}]], offset=0)"   # a vector on every row (partition step 0)
  sb = lambda t, shape, dt="float32": f"  {t} = nl.ndarray({shape}, dtype=nl.{dt}, buffer=nl.sbuf)"
  L: list[str] = []
  pre: list[str] = ["  # 0. input-only loads and initializations first (early)"] if early else []
  H = pre if early else L   # where the input-only loads go
  d1, h1, h2t, yo, x2, w2, dn = ("ra", "rh", "rh", "ra", "x1", "w1", "ra") if tl else ("d1", "yt_1", "yt_2", "yo", "x2", "w2", "dn")
  if sp.part != "mlp":
    L += ["  # 1. x1 = x + d, h = rmsnorm(x1) (NormSpec's instructions)"]
    H += [sb("w1", (N, D)), f"  nisa.dma_copy(dst=w1, src={bcast('ln1', D)})", sb("x1", (N, D)), f"  nisa.dma_copy(dst=x1, src={rows('x', D)})"]
    # N8 tile: the row tiles reused (ra: d, then o_proj's output, then down's; rh: both norm outputs; w1: ln1, then ln2; x1: x1,
    # then x2 in place; X holds the qkv bias until the bias add reads it), RoPE per head, gate_up per 512-column block
    if tl: H += [sb("ra", (N, D)), sb("rh", (N, D)), sb("X", (N, Wx))]
    if sp.add:
      H += ([] if tl else [sb("d1", (N, D))]) + [f"  nisa.dma_copy(dst={d1}, src={rows('d', D)})"]
      L += [f"  nisa.tensor_tensor(dst=x1, data1=x1, data2={d1}, op=nl.add)"]
    L += _norm_rows("  ", "x1", "w1", str(N), D, sp.eps, sp.fmax, "_1", h1 if tl else None)
    if tl and sp.part != "attn": L += [f"  nisa.dma_copy(dst=w1, src={bcast('ln2', D)})"]   # ln2 into ln1's tile once norm 1 has read it
    if tl:
      L += ["  # 2. y = h @ Wqkv per 512-column tile (render_matmul's tile, matmuls and PSUM accumulation), + b (the bias rows sit",
            "  # in X's columns until then), and 3. RoPE of the tile's heads into X ((y*cos + flip(y)*sin), + 0 below), v + 0: no (N, Nq) row tile"]
      L += _rows_t(h1, 0, N, D, "at1", "h")
      H += [sb("cs", (N, hd)), f"  nisa.dma_copy(dst=cs, src={rows('cos', hd)})", sb("sn", (N, hd)), f"  nisa.dma_copy(dst=sn, src={rows('sin', hd)})"]
      if sp.qkv_bias: H += [f"  nisa.dma_copy(dst=X[:, 0:{Nq}], src={bcast('bqkv', Nq)})"]
      L += _qkv_tiles("at1", N, D, Nq, A["wqkv"], sp.qkv_bias, hd, QK, WC)
      yb = None
    else:
      L += ["  # 2. y = h @ Wqkv (+ b): hᵀ by transposes, render_matmul's matmuls"] + _rows_t(h1, 0, N, D, "at1", "h") + [sb("y", (N, Nq))]
      L += _mm_rows("at1", N, D, Nq, A["wqkv"], "y", "q")
      yb = "y"
    if sp.qkv_bias and not tl:
      H += [sb("bq", (N, Nq)), f"  nisa.dma_copy(dst=bq, src={bcast('bqkv', Nq)})"]
      L += [sb("yb", (N, Nq)), "  nisa.tensor_tensor(dst=yb, data1=y, data2=bq, op=nl.add)"]
      yb = "yb"
    if not tl: L += ["  # 3. X = [rot(q) | rot(k) | v | 1]: (y*cos + flip(y)*sin) + 0, v + 0 (the members' cat added zero padding)"]
    if tl: pass
    elif sp.rope == "ap":   # cos / sin repeated over the heads by one DMA each, flip(y) * sin as two 3-D access patterns
      half = lambda t, w, o: f"{t}.ap(pattern=[[{w}, {N}], [{hd}, {Hqk}], [1, {h2}]], offset={o})"   # one half of every head's columns
      H += [sb("cs", (N, QK)), f"  nisa.dma_copy(dst=cs, src={A['cos']}.ap(pattern=[[{hd}, {N}], [0, {Hqk}], [1, {hd}]], offset=0))",
            sb("sn", (N, QK)), f"  nisa.dma_copy(dst=sn, src={A['sin']}.ap(pattern=[[{hd}, {N}], [0, {Hqk}], [1, {hd}]], offset=0))"]
      L += [sb("xa", (N, QK)), f"  nisa.tensor_tensor(dst=xa, data1={yb}[:, 0:{QK}], data2=cs, op=nl.multiply)",
            sb("fl", (N, QK)),
            f"  nisa.tensor_tensor(dst={half('fl', QK, 0)}, data1={half(yb, Nq, h2)}, data2={half('sn', QK, 0)}, op=nl.multiply)",
            f"  nisa.tensor_tensor(dst={half('fl', QK, h2)}, data1={half(yb, Nq, 0)}, data2={half('sn', QK, h2)}, op=nl.multiply)"]
    else:                 # per head: column slices of the (N, hd) cos / sin rows
      H += [sb("cs", (N, hd)), f"  nisa.dma_copy(dst=cs, src={rows('cos', hd)})", sb("sn", (N, hd)), f"  nisa.dma_copy(dst=sn, src={rows('sin', hd)})"]
      L += [sb("xa", (N, QK)), sb("fl", (N, QK))]
      for h in range(Hqk):
        c = h * hd
        L += [f"  nisa.tensor_tensor(dst=xa[:, {c}:{c + hd}], data1={yb}[:, {c}:{c + hd}], data2=cs, op=nl.multiply)",
              f"  nisa.tensor_tensor(dst=fl[:, {c}:{c + h2}], data1={yb}[:, {c + h2}:{c + hd}], data2=sn[:, 0:{h2}], op=nl.multiply)",
              f"  nisa.tensor_tensor(dst=fl[:, {c + h2}:{c + hd}], data1={yb}[:, {c}:{c + h2}], data2=sn[:, {h2}:{hd}], op=nl.multiply)"]
    L += ([] if tl else [sb("X", (N, Wx)), f"  nisa.tensor_tensor(dst=X[:, 0:{QK}], data1=xa, data2=fl, op=nl.add)"]) + [
          f"  nisa.tensor_scalar(dst=X[:, 0:{QK}], data=X[:, 0:{QK}], op0=nl.add, operand0=0.0)"] + \
         ([] if tl else [f"  nisa.tensor_scalar(dst=X[:, {QK}:{Nq}], data={yb}[:, {QK}:{Nq}], op0=nl.add, operand0=0.0)"]) + [
          f"  nisa.memset(dst=X[:, {Nq}:{Wx}], value=1.0)",
          f"  nisa.dma_copy(dst={rows('xs', Wx)}, src=X)",
          "  # 4. the KV write (KVWriteSpec's instructions); the pool rows stay in SBUF for attention"]
    H += [sb("pl", (min(P, TM), nP * C2))]
    for j in range(nP):
      p0, pt = j * TM, min(TM, P - j * TM)
      H += [sb(f"wt{j}", (N, pt)), f"  nisa.dma_copy(dst=wt{j}, src={A['wt']}.ap(pattern=[[{P}, {N}], [1, {pt}]], offset={p0}))"]
      L += [f"  hp{j} = nl.ndarray(({pt}, 1), dtype=nl.float32, buffer=nl.psum)",
            f"  nisa.nc_matmul(dst=hp{j}, stationary=wt{j}, moving=X[:, {Nq}:{Nq + 1}], accumulate=False)",
            sb(f"hit{j}", (pt, 1)), f"  nisa.tensor_copy(dst=hit{j}, src=hp{j})"]
      H += [f"  nisa.dma_copy(dst=pl[0:{pt}, {j * C2}:{(j + 1) * C2}], src={A['pool']}.ap(pattern=[[{C2}, {pt}], [1, {C2}]], offset={p0 * C2}))"]
      for c0 in range(0, C2, TN):
        cw = min(TN, C2 - c0)
        L += [f"  dp = nl.ndarray(({pt}, {cw}), dtype=nl.float32, buffer=nl.psum)",
              f"  nisa.nc_matmul(dst=dp, stationary=wt{j}, moving=X[:, {Ho + c0}:{Ho + c0 + cw}], accumulate=False)",
              sb("dv", (pt, cw)), "  nisa.tensor_copy(dst=dv, src=dp)",
              sb("hb", (pt, cw)), "  nisa.memset(dst=hb, value=0.0)",
              f"  nisa.tensor_scalar(dst=hb, data=hb, op0=nl.add, operand0=hit{j})",
              sb("pr", (pt, cw), "uint8"), "  nisa.tensor_scalar(dst=pr, data=hb, op0=nl.greater, operand0=0.5)",
              f"  nisa.tensor_copy_predicated(dst=pl[0:{pt}, {j * C2 + c0}:{j * C2 + c0 + cw}], src=dv, predicate=pr)"]
      L += [f"  nisa.dma_copy(dst={A['pool']}.ap(pattern=[[{C2}, {pt}], [1, {C2}]], offset={p0 * C2}), src=pl[0:{pt}, {j * C2}:{(j + 1) * C2}])"]
    lt = G * N   # a kv head's rows: (token, q head of its group); g-major (q head, token) when gmaj
    if sp.heads == "stack":   # every kv head in one tile, head h's rows from partition h * S32 (compute engines start on quadrants)
      S32 = 32 * -(-lt // 32)
      groups = [(list(range(Hkv)), Hkv * S32, {h: h * S32 for h in range(Hkv)})]
    else: groups = [([h], lt, {h: 0}) for h in range(Hkv)]
    L += [f"  # 5. attention ({sp.heads}{', g-major rows' if gmaj else ''}): kv head groups {[(g[0], g[1]) for g in groups]}; keys and values from the pool rows in SBUF"]
    for h in range(Hkv):
      if early:   # qᵀ of kv head h from X by transposes: q head g's tokens are columns g*N .. g*N+N-1
        L += [f"  pq{h} = nl.ndarray(({hd}, {lt}), dtype=nl.float32, buffer=nl.psum)"]
        L += [f"  nisa.nc_transpose(dst=pq{h}[:, {g * N}:{(g + 1) * N}], data=X[:, {(h * G + g) * hd}:{(h * G + g + 1) * hd}])" for g in range(G)]
        L += [sb(f"qt{h}", (hd, lt)), f"  nisa.tensor_copy(dst=qt{h}, src=pq{h})"]
      else:       # qᵀ of kv head h, as render_attention loads it (one DMA per token)
        L += [sb(f"qt{h}", (hd, lt))]
        L += [f"  nisa.dma_copy(dst=qt{h}[:, {t * G}:{(t + 1) * G}], src={A['xs']}.ap(pattern=[[1, {hd}], [{hd}, {G}]], offset={t * Wx + h * G * hd}))"
              for t in range(N)]
    Rm = max(g[1] for g in groups)   # mask rows: one row set for every head (the same tokens), padding rows see no key
    offs = sorted({o for g in groups for o in g[2].values()})
    for nm, off, pad in (("lo", sp.base, 1), ("hi", sp.base + N, 0)):
      H += [sb(f"{nm}i", (Rm, 1), "int32")] + ([f"  nisa.memset(dst={nm}i, value={pad})"] if Rm > lt else [])
      if gmaj: H += [f"  nisa.dma_copy(dst={nm}i[{o + g * N}:{o + (g + 1) * N}, :], src={A['meta']}.ap(pattern=[[1, {N}], [1, 1]], offset={off}))"
                     for o in offs for g in range(G)]
      else: H += [f"  nisa.dma_copy(dst={nm}i[{o + t * G}:{o + (t + 1) * G}, :], src={A['meta']}.ap(pattern=[[0, {G}], [1, 1]], offset={off + t}))"
                  for o in offs for t in range(N)]
      H += [sb(f"{nm}f", (Rm, 1)), f"  nisa.tensor_copy(dst={nm}f, src={nm}i)"]
    for gi, (hs, R, off) in enumerate(groups):
      if not early: H += [sb(f"mr{gi}", (R, 1)), f"  nisa.memset(dst=mr{gi}, value={_fl(-FLT_MAX)})", sb(f"lr{gi}", (R, 1)), f"  nisa.memset(dst=lr{gi}, value=0.0)"]
      else: H += [sb(f"mr{gi}", (R, 1)), sb(f"lr{gi}", (R, 1))]   # written by the first key block
      H += [sb(f"acc{gi}", (R, hd)), f"  nisa.memset(dst=acc{gi}, value=0.0)"]
    rows_h = lambda t, o: f"{t}[{o}:{o + lt}, :]"
    def mask(s0: int, st: int, mk: str, k: str) -> list[str]:   # the key block's visibility mask, every row: lo <= kid <= hi
      return [sb(mk, (Rm, st), "uint8"), sb(f"kk{k}", (Rm, st), "int32"),
              f"  nisa.dma_copy(dst=kk{k}, src={A['meta']}.ap(pattern=[[0, {Rm}], [1, {st}]], offset={sp.base + 2 * N + s0}))",
              sb(f"ki{k}", (Rm, st)), f"  nisa.tensor_copy(dst=ki{k}, src=kk{k})",
              sb(f"k1{k}", (Rm, st)), f"  nisa.tensor_scalar(dst=k1{k}, data=ki{k}, op0=nl.greater_equal, operand0=lof)",
              sb(f"k2{k}", (Rm, st)), f"  nisa.tensor_scalar(dst=k2{k}, data=ki{k}, op0=nl.less_equal, operand0=hif)",
              f"  nisa.tensor_tensor(dst={mk}, data1=k1{k}, data2=k2{k}, op=nl.multiply)"]
    if early:
      for b, s0 in enumerate(range(0, P, TS)): pre += mask(s0, min(TS, P - s0), f"mk{b}", f"_{b}")
    for b, s0 in enumerate(range(0, P, TS)):
      st = min(TS, P - s0)
      js = [(jj, j, min(TK, st - j), (s0 + j) // TM) for jj, j in enumerate(range(0, st, TK))]   # (chunk, key offset, keys, pool tile)
      nc, pp = len(js), min(st, TK)
      first = early and s0 == 0   # the first key block: no rescale of lr and acc (+0.0 at the start)
      more = s0 + TS < P
      mk = f"mk{b}" if early else "mk"
      L += [f"  # keys {s0} .. {s0 + st - 1}"] + ([] if early else mask(s0, st, "mk", ""))
      for gi, (hs, R, off) in enumerate(groups):
        L += [sb("sc", (R, st))]
        for h in hs:   # S of kv head h: render_attention's matmul (the same shapes), scaled into its rows
          L += [sb(f"kt{h}", (hd, st))]
          for jj, j, ct, g in js:
            L += [f"  kp = nl.ndarray(({hd}, {ct}), dtype=nl.float32, buffer=nl.psum)",
                  f"  nisa.nc_transpose(dst=kp, data=pl[0:{ct}, {g * C2 + h * hd}:{g * C2 + (h + 1) * hd}])",
                  f"  nisa.tensor_copy(dst=kt{h}[:, {j}:{j + ct}], src=kp)"]
          L += [f"  ss = nl.ndarray(({lt}, {st}), dtype=nl.float32, buffer=nl.psum)",
                f"  nisa.nc_matmul(dst=ss, stationary=qt{h}, moving=kt{h}, accumulate=False)",
                f"  nisa.tensor_scalar(dst={rows_h('sc', off[h]) if R > lt else 'sc'}, data=ss, op0=nl.multiply, operand0={_fl(sp.scale)})"]
        mr, lr, acc = f"mr{gi}", f"lr{gi}", f"acc{gi}"
        L += [sb("sm", (R, st)), "  nisa.memset(dst=sm, value=float('-inf'))",
              f"  nisa.tensor_copy_predicated(dst=sm, src=sc, predicate={mk if R == Rm else f'{mk}[0:{R}, :]'})",
              sb("bm", (R, 1)), "  nisa.tensor_reduce(dst=bm, op=nl.maximum, data=sm, axis=1)", sb("mn", (R, 1))]
        if first: L += [f"  nisa.tensor_scalar(dst=mn, data=bm, op0=nl.maximum, operand0={_fl(-FLT_MAX)})"]   # max(mr = -FLT_MAX, bm)
        else: L += [f"  nisa.tensor_tensor(dst=mn, data1={mr}, data2=bm, op=nl.maximum)"]
        L += [sb("ng", (R, 1)), "  nisa.tensor_scalar(dst=ng, data=mn, op0=nl.multiply, operand0=-1.0)"]
        if not first: L += [sb("al", (R, 1))] + _exp(["al"], [mr], [f"({R}, 1)"], "  ")
        # exp(sm + ng) as _exp computes it, transposed after its first (clamping) steps: keys on the partitions
        L += [sb("x0", (R, st)), "  nisa.tensor_scalar(dst=x0, data=sm, op0=nl.add, operand0=ng)",
              "  nisa.tensor_scalar(dst=x0, data=x0, op0=nl.maximum, operand0=-175.0)",
              "  nisa.tensor_scalar(dst=x0, data=x0, op0=nl.minimum, operand0=177.0)", sb("xT", (pp, nc * R))]
        for jj, j, ct, g in js:   # 128-key chunk jj of the block -> columns jj*R.. of xT (keys on the partitions)
          L += [f"  xp = nl.ndarray(({ct}, {R}), dtype=nl.float32, buffer=nl.psum)", f"  nisa.nc_transpose(dst=xp, data=x0[:, {j}:{j + ct}])",
                f"  nisa.tensor_copy(dst=xT[0:{ct}, {jj * R}:{(jj + 1) * R}], src=xp)"]
        L += [sb("pT", (pp, nc * R))]
        nki.exact_exp(lambda ln: L.append("  " + ln), _tiles(L, f"({pp}, {nc * R})", "ep"), "pT", "xT", base2=False)   # its clamps: no-ops on x0
        L += [sb("p", (R, st))]
        for jj, j, ct, g in js:   # back to (rows, keys) for the row sums: the same values in render_attention's layout
          L += [f"  pb = nl.ndarray(({R}, {ct}), dtype=nl.float32, buffer=nl.psum)", f"  nisa.nc_transpose(dst=pb, data=pT[0:{ct}, {jj * R}:{(jj + 1) * R}])",
                f"  nisa.tensor_copy(dst=p[:, {j}:{j + ct}], src=pb)"]
        L += [sb("rs", (R, 1)), "  nisa.tensor_reduce(dst=rs, op=nl.add, data=p, axis=1)"]
        if first: L += [f"  nisa.tensor_scalar(dst={lr}, data=rs, op0=nl.add, operand0=0.0)"]   # lr * al + rs with lr = +0.0
        else: L += [f"  nisa.tensor_tensor(dst={lr}, data1={lr}, data2=al, op=nl.multiply)",
                    f"  nisa.tensor_tensor(dst={lr}, data1={lr}, data2=rs, op=nl.add)",
                    f"  nisa.tensor_scalar(dst={acc}, data={acc}, op0=nl.multiply, operand0=al)"]
        if not first or more: L += [f"  nisa.tensor_copy(dst={mr}, src=mn)"]
        for h in hs:   # P·V of kv head h: render_attention's matmuls (pᵀ chunk of its rows stationary, its V columns moving)
          L += [f"  op = nl.ndarray(({lt}, {hd}), dtype=nl.float32, buffer=nl.psum)"]
          L += [f"  nisa.nc_matmul(dst=op, stationary=pT[0:{ct}, {jj * R + off[h]}:{jj * R + off[h] + lt}], "
                f"moving=pl[0:{ct}, {g * C2 + C + h * hd}:{g * C2 + C + (h + 1) * hd}], accumulate={jj > 0})" for jj, j, ct, g in js]
          dst = rows_h(acc, off[h]) if R > lt else acc
          if first: L += [f"  nisa.tensor_scalar(dst={dst}, data=op, op0=nl.add, operand0=0.0)"]   # acc * al + o with acc = +0.0
          else: L += [f"  nisa.tensor_tensor(dst={dst}, data1={dst}, data2=op, op=nl.add)"]
    for gi, (hs, R, off) in enumerate(groups):
      L += [sb("rl", (R, 1)), f"  nisa.tensor_scalar(dst=lr{gi}, data=lr{gi}, op0=nl.maximum, operand0={nki.FLT_MIN_!r})",
            f"  nisa.reciprocal(dst=rl, data=lr{gi})", f"  nisa.tensor_scalar(dst=acc{gi}, data=acc{gi}, op0=nl.multiply, operand0=rl)"]
      if gmaj: L += [f"  nisa.dma_copy(dst={A['os']}.ap(pattern=[[{Ho}, {N}], [1, {hd}]], offset={(h * G + g) * hd}), "
                     f"src=acc{gi}[{off[h] + g * N}:{off[h] + (g + 1) * N}, :])" for h in hs for g in range(G)]
      else: L += [f"  nisa.dma_copy(dst={A['os']}.ap(pattern=[[{hd}, {G}], [1, {hd}]], offset={t * Ho + h * G * hd}), "
                  f"src=acc{gi}[{off[h] + t * G}:{off[h] + (t + 1) * G}, :])" for h in hs for t in range(N)]
    L += ["  # 6. o_proj (Aᵀ from the stored rows, as render_matmul loads it), + bias, x2 = x1 + o, h2 = rmsnorm(x2)",
          sb("at2", (min(Ho, TK), -(-Ho // TK) * N))]
    if oc:   # Aᵀ by one transpose of each group's rows (hd, R), then each q head's columns to its 32-aligned rows of its chunk
      for gi, (hs, R, off) in enumerate(groups):
        L += [f"  pa{gi} = nl.ndarray(({hd}, {R}), dtype=nl.float32, buffer=nl.psum)", f"  nisa.nc_transpose(dst=pa{gi}, data=acc{gi})"]
        for h in hs:
          for g in range(G):
            i, r0 = divmod((h * G + g) * hd, TK)
            L += [f"  nisa.tensor_copy(dst=at2[{r0}:{r0 + hd}, {i * N}:{(i + 1) * N}], src=pa{gi}[:, {off[h] + g * N}:{off[h] + (g + 1) * N}])"]
    else:
      L += [f"  for i in range({-(-Ho // TK)}):", f"    kt = min({TK}, {Ho} - i * {TK})"]
      if AT_ROWS and N >= AT_ROWS: L += _tload("at2", f"i * {N}:i * {N} + {N}", A["os"], Ho, str(N), "kt", f"i * {TK}", "    ", "a")
      else: L += _dma_in("at2", f"i * {N}:i * {N} + {N}", A["os"], 1, "kt", Ho, str(N), f"i * {TK}", "    ")
    L += _mm_wide("at2", N, Ho, D, A["wo"], yo, "o", WC) if tl else [sb("yo", (N, D))] + _mm_rows("at2", N, Ho, D, A["wo"], yo, "o")
  else:   # N8 TP: the mlp half starts from the residual x1 (x) and the reduced o_proj product (o), as step 6 has them
    H += [sb("x1", (N, D)), f"  nisa.dma_copy(dst=x1, src={rows('x', D)})"]
    if tl: H += [sb("ra", (N, D)), sb("rh", (N, D)), sb("w1", (N, D)), f"  nisa.dma_copy(dst=w1, src={bcast('ln2', D)})"]
    else: H += [sb("yo", (N, D))]
    L += ["  # 6. (mlp half) the all-reduced o_proj product", f"  nisa.dma_copy(dst={yo}, src={rows('o', D)})"]
  if sp.part == "attn":   # N8 TP: o_proj's product (before its all-reduce and bias) and the residual x1 out
    L += [f"  nisa.dma_copy(dst={rows('po', D)}, src={yo})", f"  nisa.dma_copy(dst={rows('xo', D)}, src=x1)", "  return data0, data1, data2, data3, data4"]
    L = [f"def {sp.name}({', '.join(A.values())}):"] + pre + L
    return NKIProgram(sp.name, HEADER + "\n".join(L) + "\n", len(A), sp.outs(), {i: dtypes.int32 if nm == "meta" else dtypes.float32 for nm, i in
                                                                              ((nm, int(x[4:])) for nm, x in A.items())})
  if sp.part == "tp":   # N8 TP fused: o_proj's product all-reduced on the device, in place in its SBUF tile (the halves' cut point)
    L += ["  # 6b. all-reduce of o_proj's product over the tensor-parallel ranks (nki.collectives, private HBM scratch)"] + _cc_rows(yo, N, D, sp.tpw, "o")
  if sp.o_bias:
    H += [sb("bo", (N, D)), f"  nisa.dma_copy(dst=bo, src={bcast('bo', D)})"]
    L += [f"  nisa.tensor_tensor(dst={yo}, data1={yo}, data2=bo, op=nl.add)"]
  if not tl: H += [sb("w2", (N, D)), f"  nisa.dma_copy(dst=w2, src={bcast('ln2', D)})"]
  L += ([] if tl else [sb("x2", (N, D))]) + [f"  nisa.tensor_tensor(dst={x2}, data1=x1, data2={yo}, op=nl.add)", f"  nisa.dma_copy(dst={rows('xo', D)}, src={x2})"]
  L += _norm_rows("  ", x2, w2, str(N), D, sp.eps, sp.fmax, "_2", h2t if tl else None)
  nkI = -(-I // TK)
  if sp.mm == "ws":
    L += ["  # 7. gate_up weight-stationary (gᵀ, uᵀ out of PSUM: features on the partitions), silu(gate) * up on (128, ·) tiles"]
    pf = sp.pf if early else 0
    L += _rows_t("yt_2", 0, N, D, "at3", "g") + _mm_ws("at3", N, D, A["wgu"], 2 * I, 0, I, "gT", "a", pre if pf else None, pf)
    L += _mm_ws("at3", N, D, A["wgu"], 2 * I, I, I, "uT", "b", pre if pf > _ws_tiles(D, I) else None, pf - _ws_tiles(D, I))
  elif tl:
    L += ["  # 7. gate_up per 512-column block of gate and of up (render_matmul's matmuls per tile), each block to (128, ·) tiles by",
          "  # transposes and silu(gate) * up into its columns of aT (the down projection's Aᵀ): no (N, 2I) row tile"]
    L += _rows_t(h2t, 0, N, D, "at3", "g") + _mlp_blocks("at3", N, D, I, A["wgu"], WC)
  else:
    L += ["  # 7. gate_up, then silu(gate) * up on (128, ·) tiles (features on the partitions = the down projection's Aᵀ)"]
    L += _rows_t("yt_2", 0, N, D, "at3", "g") + [sb("gu", (N, 2 * I))] + _mm_rows("at3", N, D, 2 * I, A["wgu"], "gu", "u")
    L += _rows_t("gu", 0, N, I, "gT", "a") + _rows_t("gu", I, N, I, "uT", "b")
  T = f"({min(I, TK)}, {nkI * N})"
  if not tl: L += _silu_mul(T, "gT", "uT", "aT", "  ", True)
  if sp.mm == "ws":
    L += ["  # 8. down weight-stationary (dnᵀ), back to rows by transposes"]
    L += _mm_ws("aT", N, I, A["wd"], D, 0, D, "dT", "d") + [sb("dn", (N, D))] + _cols_t("dT", N, D, "dn", "d")
  else:
    L += ["  # 8. down"]
    L += _mm_wide("aT", N, I, D, A["wd"], dn, "d", WC) if tl else [sb("dn", (N, D))] + _mm_rows("aT", N, I, D, A["wd"], dn, "d")
  if sp.part == "tp":   # N8 TP fused: down's product all-reduced, the sum stored as dn (the next layer's residual term)
    L += ["  # 8b. all-reduce of down's product, stored to dn"] + _cc_rows(dn, N, D, sp.tpw, "d", rows("dn", D))
  else: L += [f"  nisa.dma_copy(dst={rows('dn', D)}, src={dn})"]
  L += ["  return data0, data1" + ("" if sp.part == "mlp" else ", data2, data3, data4")]
  L = [f"def {sp.name}({', '.join(A.values())}):"] + pre + L
  n = len(A)
  return NKIProgram(sp.name, (CC_HEADER if sp.part == "tp" else HEADER) + "\n".join(L) + "\n", n, sp.outs(), {i: dtypes.int32 if nm == "meta" else dtypes.float32 for nm, i in
                                                                              ((nm, int(a[4:])) for nm, a in A.items())})

# ---------------------------------------------------------------- N7 step 4: microbenchmarks (scripts/nki_n7_probe.py)
# Two questions the step 3 numbers leave open. (1) What one float32 nc_matmul costs on trn1 in each operand form: the head
# (2079 matmuls of 512 moving columns) took ~2.06 ms whatever its weight DMAs looked like (render_matmul's 2079 DMAs of 2 KiB
# runs 2.17, HeadSpec's 149 DMAs of 28 KiB runs 2.06, 298 of 14 KiB 2.08), so it looks TensorE-bound at ~1 us per matmul,
# ~0.5 G moving columns/s; a GEMV's weight as the stationary (128 x 128 per instruction, the hᵀ columns moving) may load
# faster than it streams. (2) Why one block's weight split over several DMAs was slower (1024:4 2.59 ms, 1024:8 3.85 vs
# 1024:1 2.06): DMAs into one tile serialized with a fixed cost each, or partition groups getting a share of the bandwidth.
# Each probe is one kernel on resident or streamed tiles; device time / n = the cost of one instruction or block.

PROBE_KINDS = ("mv", "ws", "tr", "dma")
PROBE_DMA_MODES = ("blk", "grp4", "col4", "run2k")

@dataclass(frozen=True)
class ProbeSpec:
  """One microbenchmark kernel `fz_probe_...(data0=out, data1=in)`, float32 HBM buffers (in: >= probe_in_size floats).
  kind "mv": n nc_matmul, stationary (128, m) and 8 moving (128, 512) tiles resident in SBUF, one PSUM accumulation chain;
  "ws": n nc_matmul, 32 stationary (128, 128) slices of 8 resident tiles, moving (128, m); "tr": n nc_transpose of
  (m, 128) tiles into PSUM; dt bfloat16: the matmul operands cast once (the bf16 rate, for reference). "dma": n blocks of
  (128, f) floats from in (block b at b * 128 * f), each read by one VectorE copy of a column (so no load is dead): mode blk
  (one DMA per block into one tile, HeadSpec's :1), grp4 (four DMAs of 32 partitions into one tile, HeadSpec's :4), col4
  (four DMAs of f/4 columns into four tiles), run2k (f/512 DMAs of (128, 512) tiles, 2 KiB runs at row stride f:
  render_matmul's K-major weight loads)."""
  kind: str
  n: int
  m: int = 1
  f: int = 7168
  mode: str = "blk"
  dt: str = "float32"
  @property
  def name(self) -> str:
    if self.kind == "dma": return f"fz_probe_dma_{self.mode}_{self.n}_{self.f}"
    return f"fz_probe_{self.kind}{'16' if self.dt == 'bfloat16' else ''}_{self.n}_{self.m}"
  def in_size(self) -> int:
    return self.n * TK * self.f if self.kind == "dma" else 8 * TK * TN + TK * TK
  def out_size(self) -> int: return max(TK * self.n if self.kind == "dma" else TK * TN, TK * TN)
  def render(self) -> NKIProgram: return render_probe(self)

def render_probe(sp: ProbeSpec) -> NKIProgram:
  n, m, f = sp.n, sp.m, sp.f
  if sp.kind not in PROBE_KINDS or sp.dt not in ("float32", "bfloat16") or n <= 0 or not 0 < m <= TK: raise NKIUnsupported(f"{sp.name}: no such probe")
  L = [f"def {sp.name}(data0, data1):"]
  if sp.kind == "dma":
    if sp.mode not in PROBE_DMA_MODES or f % (TN if sp.mode == "run2k" else 4) or f * 4 > 128 * 1024: raise NKIUnsupported(f"{sp.name}: f={f}")
    L += [f"  acc = nl.ndarray(({TK}, {n}), dtype=nl.float32, buffer=nl.sbuf)"]
    for b in range(n):
      o = b * TK * f
      if sp.mode in ("blk", "grp4"):
        L += [f"  w{b} = nl.ndarray(({TK}, {f}), dtype=nl.float32, buffer=nl.sbuf)"]
        g = TK if sp.mode == "blk" else TK // 4
        L += [f"  nisa.dma_copy(dst=w{b}[{p0}:{p0 + g}, :], src=data1.ap(pattern=[[{f}, {g}], [1, {f}]], offset={o + p0 * f}))" for p0 in range(0, TK, g)]
        L += [f"  nisa.tensor_copy(dst=acc[:, {b}:{b + 1}], src=w{b}[:, 0:1])"]
      else:
        q = f // 4 if sp.mode == "col4" else TN
        for j in range(f // q):
          L += [f"  w{b}_{j} = nl.ndarray(({TK}, {q}), dtype=nl.float32, buffer=nl.sbuf)",
                f"  nisa.dma_copy(dst=w{b}_{j}, src=data1.ap(pattern=[[{f}, {TK}], [1, {q}]], offset={o + j * q}))"]
        L += [f"  nisa.tensor_copy(dst=acc[:, {b}:{b + 1}], src=w{b}_{j}[:, 0:1])" for j in range(f // q)]
    L += [f"  nisa.dma_copy(dst=data0.ap(pattern=[[{n}, {TK}], [1, {n}]], offset=0), src=acc)", "  return data0"]
    return _program(sp.name, L, 2, {0: dtypes.float32, 1: dtypes.float32})
  dt = sp.dt
  L += [f"  t{j} = nl.ndarray(({TK}, {TN}), dtype=nl.float32, buffer=nl.sbuf)" for j in range(8)]
  L += [f"  nisa.dma_copy(dst=t{j}, src=data1.ap(pattern=[[{TN}, {TK}], [1, {TN}]], offset={j * TK * TN}))" for j in range(8)]
  if sp.kind == "tr":
    L += [f"  src = nl.ndarray(({m}, {TK}), dtype=nl.float32, buffer=nl.sbuf)", f"  nisa.tensor_copy(dst=src, src=t0[0:{m}, 0:{TK}])",
          f"  ps = nl.ndarray(({TK}, {m}), dtype=nl.float32, buffer=nl.psum)"]
    L += [f"  nisa.nc_transpose(dst=ps, data={'src' if r % 2 == 0 else f't{(r // 2) % 8}[0:{m}, 0:{TK}]'})" for r in range(n)]
    L += [f"  ot = nl.ndarray(({TK}, {m}), dtype=nl.float32, buffer=nl.sbuf)", "  nisa.tensor_copy(dst=ot, src=ps)",
          f"  nisa.dma_copy(dst=data0.ap(pattern=[[{m}, {TK}], [1, {m}]], offset=0), src=ot)", "  return data0"]
    return _program(sp.name, L, 2, {0: dtypes.float32, 1: dtypes.float32})
  if dt != "float32":
    L += [f"  c{j} = nl.ndarray(({TK}, {TN}), dtype=nl.{dt}, buffer=nl.sbuf)" for j in range(8)]
    L += [f"  nisa.tensor_copy(dst=c{j}, src=t{j})" for j in range(8)]
    tl = "c"
  else: tl = "t"
  L += [f"  v = nl.ndarray(({TK}, {m}), dtype=nl.{dt}, buffer=nl.sbuf)", f"  nisa.tensor_copy(dst=v, src={tl}0[:, 0:{m}])"]
  if sp.kind == "mv":
    L += [f"  ps = nl.ndarray(({m}, {TN}), dtype=nl.float32, buffer=nl.psum)"]
    L += [f"  nisa.nc_matmul(dst=ps, stationary=v, moving={tl}{r % 8}, accumulate={r > 0})" for r in range(n)]
    L += [f"  ot = nl.ndarray(({m}, {TN}), dtype=nl.float32, buffer=nl.sbuf)", "  nisa.tensor_copy(dst=ot, src=ps)",
          f"  nisa.dma_copy(dst=data0.ap(pattern=[[{TN}, {m}], [1, {TN}]], offset=0), src=ot)", "  return data0"]
  else:
    L += [f"  ps = nl.ndarray(({TK}, {m}), dtype=nl.float32, buffer=nl.psum)"]
    L += [f"  nisa.nc_matmul(dst=ps, stationary={tl}{(r // 4) % 8}[:, {(r % 4) * TK}:{(r % 4 + 1) * TK}], moving=v, accumulate={r > 0})" for r in range(n)]
    L += [f"  ot = nl.ndarray(({TK}, {m}), dtype=nl.float32, buffer=nl.sbuf)", "  nisa.tensor_copy(dst=ot, src=ps)",
          f"  nisa.dma_copy(dst=data0.ap(pattern=[[{m}, {TK}], [1, {m}]], offset=0), src=ot)", "  return data0"]
  return _program(sp.name, L, 2, {0: dtypes.float32, 1: dtypes.float32})

# ---------------------------------------------------------------- N8 (I72): the tensor-parallel all-reduce as an NKI collective

AR_KINDS = ("phbm", "hbm", "sbuf")
CC_HEADER = "import nki\nimport nki.language as nl\nimport nki.isa as nisa\nimport nki.collectives\n\n@nki.jit\n"

def ar_kind() -> str:
  """FZ_NEURON_TP_AR_KIND: how AllReduceSpec feeds nki.collectives.all_reduce (phbm, default; hbm; sbuf; see AllReduceSpec)."""
  k = os.environ.get("FZ_NEURON_TP_AR_KIND", "").strip() or "phbm"
  if k not in AR_KINDS: raise ValueError(f"FZ_NEURON_TP_AR_KIND={k!r}: one of {', '.join(AR_KINDS)}")
  return k

def _pf(n: int) -> tuple[int, int]:
  """(P, F): n floats as P <= 128 partitions of F contiguous floats (the largest P that divides n)."""
  p = next(p for p in range(min(TK, n), 0, -1) if n % p == 0)
  return p, n // p

@dataclass(frozen=True)
class AllReduceSpec:
  """`fz_all_reduce_<kind>_w<world>_<n>(data0=out, data1=in)`: out = the sum over the ranks [0, world) of in (n float32), one
  nki.collectives.all_reduce (nki >= 0.5: a CollectiveCompute instruction neuronx-cc puts in the NEFF; the runtime runs it
  between the NeuronCores of the ranks, which load the NEFF with nrt_load_collectives: nrtpy's load_from_neff(cc_enabled=True,
  rank_id, world_size)). The tensor-parallel decode's all-reduce stays on the device, inside the step's batch of launches,
  instead of a host round trip per all-reduce (docs/NKI_BACKEND.md §11). kind (FZ_NEURON_TP_AR_KIND, which form the
  hardware accepts is unverified until the probe, scripts/nki_tp_probe.py):
    phbm  in -> a private-HBM scratch (P, F) by DMA, all_reduce scratch -> scratch2 (both private_hbm, which nki 0.7's
          docs require of LNC-1 sources), DMA back to out (default);
    hbm   all_reduce straight from the argument in to the argument out;
    sbuf  in -> SBUF tiles (P, <= 4096 floats per partition), all_reduce SBUF -> SBUF per tile, store to out.
  fusable (FZ_NEURON_TP_AR_FUSE=1): fuse_schedule may put it inside a fused layer kernel (one NEFF with the collective);
  default off: the all-reduce is its own launch, so the fused member kernels around it need no collectives.
  fuse (I75, FZ_NEURON_TP_PREFILL=fused): this all-reduce is fusable whatever FZ_NEURON_TP_AR_FUSE says: a prefill forward
  (rows beyond LayerSpec, so its layers run as the members) records its all-reduces that way, and fuse_schedule puts each
  group of FZ_NEURON_TP_PREFILL_FUSE layers, collectives included, in one NEFF. The kernel source does not change (same name,
  same instructions, same NEFF)."""
  n: int
  world: int
  kind: str = "phbm"
  fuse: bool = False
  @property
  def name(self) -> str: return f"fz_all_reduce_{self.kind}_w{self.world}_{self.n}"
  @property
  def fusable(self) -> bool: return self.fuse or os.environ.get("FZ_NEURON_TP_AR_FUSE", "0") not in ("", "0")
  def render(self) -> NKIProgram: return render_all_reduce(self)

def render_all_reduce(sp: AllReduceSpec) -> NKIProgram:
  n, W = sp.n, sp.world
  if n <= 0 or W < 1 or sp.kind not in AR_KINDS: raise NKIUnsupported(f"{sp.name}: no such all-reduce")
  P, F = _pf(n)
  rg = f"nki.collectives.ReplicaGroup([{list(range(W))}])"
  L = [f"def {sp.name}(data0, data1):"]
  if sp.kind == "hbm":
    L += [f"  nki.collectives.all_reduce(srcs=[data1], dsts=[data0], replica_group={rg}, op=nl.add)"]
  elif sp.kind == "phbm":
    view = lambda t: f"{t}.ap(pattern=[[{F}, {P}], [1, {F}]], offset=0)"
    L += [f"  s = nl.ndarray(({P}, {F}), dtype=nl.float32, buffer=nl.private_hbm)",
          f"  r = nl.ndarray(({P}, {F}), dtype=nl.float32, buffer=nl.private_hbm)",
          f"  nisa.dma_copy(dst=s, src={view('data1')})",
          f"  nki.collectives.all_reduce(srcs=[s], dsts=[r], replica_group={rg}, op=nl.add)",
          f"  nisa.dma_copy(dst={view('data0')}, src=r)"]
  else:
    for c0 in range(0, F, 4096):
      cw = min(4096, F - c0)
      v = lambda t: f"{t}.ap(pattern=[[{F}, {P}], [1, {cw}]], offset={c0})"
      L += [f"  s{c0} = nl.ndarray(({P}, {cw}), dtype=nl.float32, buffer=nl.sbuf)",
            f"  r{c0} = nl.ndarray(({P}, {cw}), dtype=nl.float32, buffer=nl.sbuf)",
            f"  nisa.dma_copy(dst=s{c0}, src={v('data1')})",
            f"  nki.collectives.all_reduce(srcs=[s{c0}], dsts=[r{c0}], replica_group={rg}, op=nl.add)",
            f"  nisa.dma_copy(dst={v('data0')}, src=r{c0})"]
  L += ["  return data0"]
  return NKIProgram(sp.name, CC_HEADER + "\n".join(L) + "\n", 2, (0,), {0: dtypes.float32, 1: dtypes.float32})

@dataclass(frozen=True)
class ARChainSpec:
  """N8 I72 follow-up (scripts/nki_tp_probe.py, phase "chain"): k all-reduces in one kernel, to split a collective's cost
  into a per-launch part and a per-collective part (run 1: one AllReduceSpec launch 0.36-0.39 ms at 14 KiB - 4 MiB, a plain
  small kernel 0.11 ms). `fz_ar_chain_<form>_w<world>_<n>_k<k>(data0=out, data1=in)`, n float32, phbm-style (private HBM):
    dep    the layer's pattern: k times (HBM -> private scratch by DMA, all_reduce, DMA back to out), each step reading the
           previous one's out (data1 for the first): a serial chain, out = 2^(k-1) * sum(in) exactly (one rank's values
           doubled by the two ranks' identical sums from step 2 on: world 2 only)
    indep  one DMA in, k all_reduces of the same scratch into k results (no dependencies between them), the last DMA'd out:
           out = sum(in); whether the runtime overlaps independent collectives"""
  n: int
  world: int
  k: int
  form: str = "dep"
  @property
  def name(self) -> str: return f"fz_ar_chain_{self.form}_w{self.world}_{self.n}_k{self.k}"
  def render(self) -> NKIProgram:
    n, W, k = self.n, self.world, self.k
    if n <= 0 or W < 1 or k < 1 or self.form not in ("dep", "indep"): raise NKIUnsupported(f"{self.name}: no such chain")
    P, F = _pf(n)
    rg = f"nki.collectives.ReplicaGroup([{list(range(W))}])"
    view = lambda t: f"{t}.ap(pattern=[[{F}, {P}], [1, {F}]], offset=0)"
    dec = lambda t: f"  {t} = nl.ndarray(({P}, {F}), dtype=nl.float32, buffer=nl.private_hbm)"
    L = [f"def {self.name}(data0, data1):"]
    if self.form == "dep":
      for j in range(k):
        L += [dec(f"s{j}"), dec(f"r{j}"), f"  nisa.dma_copy(dst=s{j}, src={view('data1' if j == 0 else 'data0')})",
              f"  nki.collectives.all_reduce(srcs=[s{j}], dsts=[r{j}], replica_group={rg}, op=nl.add)",
              f"  nisa.dma_copy(dst={view('data0')}, src=r{j})"]
    else:
      L += [dec("s"), f"  nisa.dma_copy(dst=s, src={view('data1')})"]
      for j in range(k): L += [dec(f"r{j}"), f"  nki.collectives.all_reduce(srcs=[s], dsts=[r{j}], replica_group={rg}, op=nl.add)"]
      L += [f"  nisa.dma_copy(dst={view('data0')}, src=r{k - 1})"]
    L += ["  return data0"]
    return NKIProgram(self.name, CC_HEADER + "\n".join(L) + "\n", 2, (0,), {0: dtypes.float32, 1: dtypes.float32})

# ---------------------------------------------------------------- N6: several kernels as one NEFF (fusion of scheduler items)

@dataclass(frozen=True)
class FenceSpec:
  """A no-op item that Llama.forward_tokens records between layers on NEURON (neuron_kv.fence): the fusion boundary that
  NEURONDevice.fuse_schedule splits a captured forward at, so every middle layer becomes the same fused kernel (one NEFF,
  compiled once, launched per layer on that layer's buffers). It never reaches a runtime."""
  layer: int = 0
  hard: bool = False   # always a boundary (before the head), whatever FZ_NEURON_FUSE groups
  every: int = 0       # I75: layers per fused kernel for this forward (a TP prefill's FZ_NEURON_TP_PREFILL_FUSE); 0: FZ_NEURON_FUSE
  @property
  def name(self) -> str: return "fz_fence"

@dataclass(frozen=True)
class FusedSpec:
  """N6 (docs/NKI_BACKEND.md §4 exception 9): consecutive kernels of a captured forward as one NKI kernel, so one NEFF launch
  replaces len(parts) (N5 run 1: ~0.66 ms per launch in the decode loop, 245 per token). parts: (member kernel name, its
  rendered source, member argument i -> fused argument). The body is each member's body in order, its data<i> renamed to the
  fused arguments and every other name suffixed by the member's index, so it runs exactly the members' instructions in
  their order (the same bits as the separate launches); intermediates stay in their HBM buffers (fused arguments). n: the
  fused argument count, outs: the arguments some member writes (returned, so nki makes them must-alias outputs), dts:
  (argument, dtype) pairs."""
  parts: tuple[tuple[str, str, tuple[int, ...]], ...]
  n: int
  outs: tuple[int, ...]
  dts: tuple
  @functools.cached_property
  def name(self) -> str:   # N7: once per spec (a replay asked for it per launch: repr + sha256 of every member's source, ~0.5 ms)
    import hashlib
    h = hashlib.sha256(repr((self.parts, self.n, self.outs)).encode()).hexdigest()[:10]
    return f"fz_fused{len(self.parts)}_{h}"
  def __hash__(self) -> int:   # N7: cached too (the spec is a program-cache key; hashing re-walked every member's source)
    if (h := self.__dict__.get("_h")) is None: object.__setattr__(self, "_h", h := hash((self.parts, self.n, self.outs, self.dts)))
    return h
  @property
  def members(self) -> list[str]: return [p[0] for p in self.parts]
  def render(self) -> NKIProgram: return render_fused(self)

_KEEP_NAMES = {"nl", "nisa", "nki"}

def render_fused(sp: FusedSpec) -> NKIProgram:
  """FusedSpec -> `fz_fused<k>_<hash>(data0, ...)`: the members' bodies in order (see FusedSpec), nothing else."""
  import ast, builtins
  body: list[str] = []
  for k, (nm, src, amap) in enumerate(sp.parts):
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef))
    args = [a.arg for a in fn.args.args]
    if len(args) != len(amap): raise NKIUnsupported(f"fused part {nm}: {len(args)} arguments, {len(amap)} mapped")
    ren = {a: f"data{amap[i]}" for i, a in enumerate(args)}
    stmts = list(fn.body)
    if stmts and isinstance(stmts[-1], ast.Return): stmts = stmts[:-1]
    for s in stmts:
      for n in ast.walk(s):
        if isinstance(n, ast.Return): raise NKIUnsupported(f"fused part {nm}: a return before the end")
        if isinstance(n, ast.Name):
          if n.id in ren: n.id = ren[n.id]
          elif n.id not in _KEEP_NAMES and not hasattr(builtins, n.id): n.id = f"{n.id}_{k}"
    body.append(f"  # part {k}: {nm}")
    body += ["  " + ln for s in stmts for ln in ast.unparse(s).split("\n")]
  L = [f"def {sp.name}({', '.join(f'data{i}' for i in range(sp.n))}):"] + body + [f"  return {', '.join(f'data{i}' for i in sp.outs)}"]
  cc = any("nki.collectives" in src for _, src, _ in sp.parts)   # N8: a fused all-reduce (FZ_NEURON_TP_AR_FUSE) needs the import
  return NKIProgram(sp.name, (CC_HEADER if cc else HEADER) + "\n".join(L) + "\n", sp.n, tuple(sp.outs), dict(sp.dts))
