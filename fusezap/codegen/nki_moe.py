"""I105 A2: sparse mixture-of-experts decode on NEURON, the chosen experts' weight tiles read by indirect DMA.

The dense masked form (fusezap/serve/models/moe.py) reads all E experts' weights every decode token. Here a decode step's
expert MLP is two kernels that read only the experts each row picked:

  MoeGateUpSparse  G (N, K, 2 Iw) f32 = for row n and its j-th chosen expert e (in increasing expert order):
                   [h_n @ Wgu[:, gate cols of e], h_n @ Wgu[:, up cols of e]]
  MoeDownSparse    Y (N, D) f32 = for row n: sum over its chosen experts, in increasing expert order, of
                   (w_ne * a_nj) @ Wd[rows of e]       (a = silu(gate) * up, computed between the two kernels)

over the dense form's own weights (no copy): Wgu the stacked K-major (D, 2 E Iw) gate | up weight (expert e's gate
columns [e Iw, (e+1) Iw), its up columns E Iw + the same), Wd the stacked K-major (E Iw, D) down weight. The routing comes
in as `key` (N, E) float32: key[n, e] = the number of chosen experts below e if e is chosen, else E (moe.route_keys), so
the j-th chosen expert of row n is the e with key[n, e] == j; the kernels find it with an (1, E) compare, a multiply by
iota and a sum (exact small integers), and broadcast the expert numbers to the 128 partitions with a 0/1 TensorE matmul.

**Bits.** Every matmul instruction is the dense kernel's: stationary = all N rows' Aᵀ chunk (K <= 128 on the partitions),
moving = a (<= 128, <= 512) weight tile, PSUM accumulation over K in the same order; only the moving tile's address comes
from the routing. Row n's result is taken from the N-row PSUM tile (copied to SBUF, then one DMA from partition n). For the
down projection the dense kernel's accumulation chain over all E Iw / 128 chunks adds exact zeros for the experts a row did
not pick (their activations are multiplied by a zero weight), so when Iw is a multiple of 128 the sparse chain over the
chosen experts' chunks in increasing expert order adds the same nonzero terms in the same order: the same float32 sums
(tests/test_olmoe_sparse.py: bitwise on NKISHIM with FZ_NKISHIM_MM=f64, whose matmul is row- and shape-invariant like the
PE array). The activation's weighting (a * w in float32) and its bf16 rounding are the dense path's.

**The new exception (NKI_BACKEND.md exception 11): runtime addresses from a device tensor.** The weight tile DMAs take
their row offsets from int32 SBUF values computed in the kernel (nki 0.6 `ap(..., vector_offset=, indirect_dim=0)` or
`ap(..., scalar_offset=)`). The loop bounds stay static: every row reads exactly K experts' tiles, so a step's cost does not
depend on the routing, and padding rows read real (in-range) experts with weight 0. FZ_MOE_IDMA picks how the address is
expressed (the API's semantics are **unverified on hardware**; scripts/olmoe_sparse_selftest.py tries each on the chip and
keeps the first that compiles and gives the reference's numbers):
  vec     pattern [[1, P], [1, F]], vector_offset[p] = the absolute element offset of partition p's row (an index replacing
          p, times the partition step 1, so "replace" and "scaled by the step" mean the same) (default)
  vecadd  pattern [[row stride, P], [1, F]], vector_offset[p] = the expert's offset, added to the affine address
  scalar  pattern [[row stride, P], [1, F]], scalar_offset = the expert's offset, one (1, 1) int32 value
Offsets are computed in float32 (iota, a multiply, an add) and converted to int32; the renderer checks that every value
they can take is an exactly representable float32 integer < 2**31 (else NKIUnsupported: the dense path stays).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
from fusezap.dtype import dtypes
from fusezap.codegen.nki import NKIProgram, NKIUnsupported

TK, TN = 128, 512
HEADER = "import nki\nimport nki.language as nl\nimport nki.isa as nisa\n\n@nki.jit\n"
VARIANTS = ("vec", "vecadd", "scalar")

def idma_variant() -> str:
  # default scalar: the form chip-validated in A2 runs 2 and 3 (vec failed to compile in run 2, NCC_INLA001)
  v = os.environ.get("FZ_MOE_IDMA", "scalar").strip() or "scalar"
  if v not in VARIANTS: raise NKIUnsupported(f"FZ_MOE_IDMA={v!r}: one of {VARIANTS}")
  return v

def _check_exact(base: np.ndarray, step: int, E: int, what: str) -> None:
  """Every base + e * step (e < E) and its parts are float32-exact integers below 2**31 (the kernel adds them in float32)."""
  b = np.asarray(base, np.int64).reshape(-1)
  eo = np.arange(E, dtype=np.int64) * step
  tot = (b[:, None] + eo[None, :]).reshape(-1)
  for v in (b, eo, tot):
    if v.size and (v.max() >= 2 ** 31 or (v.astype(np.float32).astype(np.int64) != v).any()):
      raise NKIUnsupported(f"{what}: offsets up to {int(v.max())} are not exact in float32")
  if ((b.astype(np.float32)[:, None] + eo.astype(np.float32)[None, :]).reshape(-1).astype(np.int64) != tot).any():
    raise NKIUnsupported(f"{what}: a float32 offset sum is not exact")

def _picks(N: int, E: int, K: int, key: str, wts: str | None) -> list[str]:
  """Kernel prologue: er (1, N K) f32 with er[0, n K + j] = the j-th chosen expert of row n (clamped to [0, E-1]); with wts
  also wr (1, K N) with wr[0, j N + n] = that expert's routing weight. All on partition 0 (compute starts on a quadrant)."""
  L = [f"  et = nl.ndarray((1, {E}), dtype=nl.float32, buffer=nl.sbuf)",
       f"  nisa.iota(dst=et, pattern=[[1, {E}]], offset=0, channel_multiplier=0)",
       f"  er = nl.ndarray((1, {N * K}), dtype=nl.float32, buffer=nl.sbuf)"]
  if wts: L += [f"  wr = nl.ndarray((1, {K * N}), dtype=nl.float32, buffer=nl.sbuf)"]
  L += [f"  for n in range({N}):",
        f"    kr = nl.ndarray((1, {E}), dtype=nl.float32, buffer=nl.sbuf)",
        f"    nisa.dma_copy(dst=kr, src={key}.ap(pattern=[[{E}, 1], [1, {E}]], offset=n * {E}))"]
  if wts: L += [f"    wv = nl.ndarray((1, {E}), dtype=nl.float32, buffer=nl.sbuf)",
                f"    nisa.dma_copy(dst=wv, src={wts}.ap(pattern=[[{E}, 1], [1, {E}]], offset=n * {E}))"]
  L += [f"    for j in range({K}):",
        f"      mk = nl.ndarray((1, {E}), dtype=nl.float32, buffer=nl.sbuf)",
        f"      nisa.tensor_scalar(dst=mk, data=kr, op0=nl.equal, operand0=float(j))",
        f"      me = nl.ndarray((1, {E}), dtype=nl.float32, buffer=nl.sbuf)",
        f"      nisa.tensor_tensor(dst=me, data1=mk, data2=et, op=nl.multiply)",
        f"      nisa.tensor_reduce(dst=er[0:1, n * {K} + j:n * {K} + j + 1], op=nl.add, data=me, axis=1, keepdims=True)"]
  if wts: L += [f"      mw = nl.ndarray((1, {E}), dtype=nl.float32, buffer=nl.sbuf)",
                f"      nisa.tensor_tensor(dst=mw, data1=mk, data2=wv, op=nl.multiply)",
                f"      nisa.tensor_reduce(dst=wr[0:1, j * {N} + n:j * {N} + n + 1], op=nl.add, data=mw, axis=1, keepdims=True)"]
  L += [f"  erc = nl.ndarray((1, {N * K}), dtype=nl.float32, buffer=nl.sbuf)",
        f"  nisa.tensor_scalar(dst=erc, data=er, op0=nl.minimum, operand0={float(E - 1)}, op1=nl.maximum, operand1=0.0)",
        f"  ones = nl.ndarray((1, {TK}), dtype=nl.float32, buffer=nl.sbuf)",
        f"  nisa.memset(dst=ones, value=1.0)"]
  return L

def _offsets(N: int, K: int, step: int, var: str, base_pat: tuple[int, int] | None) -> list[str]:
  """eo (128, N K) f32: every partition holds er * step (the chosen expert's element offset in the weight); vecadd /
  scalar: its int32 copy eoi; vec: base (128, nb) f32 = the partition rows' static offsets (base_pat = (columns, row
  stride))."""
  L = [f"  ebp = nl.ndarray(({TK}, {N * K}), dtype=nl.float32, buffer=nl.psum)",
       f"  nisa.nc_matmul(dst=ebp, stationary=ones, moving=erc, accumulate=False)",
       f"  eo = nl.ndarray(({TK}, {N * K}), dtype=nl.float32, buffer=nl.sbuf)",
       f"  nisa.tensor_scalar(dst=eo, data=ebp, op0=nl.multiply, operand0={float(step)})"]
  if var == "vec":   # base[p, i] = (i 128 + p) * row stride: iota of the row (its steps must fit int16: chip run 2's
    # neuronx-cc rejected an iota step of 8388608, NCC_INLA001 checkIota), then one float32 multiply (exact: _check_exact)
    nb, ps = base_pat
    L += [f"  bi = nl.ndarray(({TK}, {nb}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  nisa.iota(dst=bi, pattern=[[{TK}, {nb}]], offset=0, channel_multiplier=1)",
          f"  base = nl.ndarray(({TK}, {nb}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  nisa.tensor_scalar(dst=base, data=bi, op0=nl.multiply, operand0={float(ps)})"]
  else:
    L += [f"  eoi = nl.ndarray(({TK}, {N * K}), dtype=nl.int32, buffer=nl.sbuf)",
          f"  nisa.tensor_copy(dst=eoi, src=eo)"]
  return L

def _vtable(nb: int, ind: str) -> list[str]:
  """vec: vo (128, nb) int32 = base + this pick's expert offset (eo's column nj)."""
  return [f"{ind}vf = nl.ndarray(({TK}, {nb}), dtype=nl.float32, buffer=nl.sbuf)",
          f"{ind}nisa.tensor_scalar(dst=vf, data=base, op0=nl.add, operand0=eo[:, nj:nj + 1])",
          f"{ind}vo = nl.ndarray(({TK}, {nb}), dtype=nl.int32, buffer=nl.sbuf)",
          f"{ind}nisa.tensor_copy(dst=vo, src=vf)"]

def _wload(var: str, src: str, rs: int, kt: str, nt: str, static: str, vcol: str, ind: str, wdt: str) -> list[str]:
  """bt (kt, nt) <- the weight tile whose partition p row starts at the runtime offset (see the module docstring)."""
  if var == "vec": ap = f"{src}.ap(pattern=[[1, {kt}], [1, {nt}]], offset={static}, vector_offset=vo[0:{kt}, {vcol}:{vcol} + 1], indirect_dim=0)"
  elif var == "vecadd": ap = f"{src}.ap(pattern=[[{rs}, {kt}], [1, {nt}]], offset={static}, vector_offset=eoi[0:{kt}, nj:nj + 1], indirect_dim=0)"
  else: ap = f"{src}.ap(pattern=[[{rs}, {kt}], [1, {nt}]], offset={static}, scalar_offset=eoi[0:1, nj:nj + 1])"
  return [f"{ind}bt = nl.ndarray(({kt}, {nt}), dtype={wdt}, buffer=nl.sbuf)", f"{ind}nisa.dma_copy(dst=bt, src={ap})"]

@dataclass(frozen=True)
class MoeGateUpSparse:
  """G (N, K, 2 Iw) f32 = the chosen experts' gate | up products of h (N, D) f32 (data0=G, data1=H, data2=KEY (N, E) f32,
  data3=Wgu (D, 2 E Iw) K-major, bf16 or f32). Picks in increasing expert order."""
  N: int
  D: int
  E: int
  Iw: int
  K: int
  bf16: bool = True
  var: str = "vec"
  @property
  def name(self) -> str: return f"fz_moe_gu_{self.var}_{self.N}_{self.D}_{self.E}_{self.Iw}_{self.K}" + ("_bf16" if self.bf16 else "")
  def render(self) -> NKIProgram:
    N, D, E, Iw, K, var = self.N, self.D, self.E, self.Iw, self.K, self.var
    if var not in VARIANTS or N * K > TN: raise NKIUnsupported(f"{self.name}: variant / {N} x {K} picks")
    RS, nk = 2 * E * Iw, -(-D // TK)
    wdt = "nl.bfloat16" if self.bf16 else "nl.float32"
    if var == "vec": _check_exact((np.arange(nk)[:, None] * TK + np.arange(TK)[None, :]) * RS, Iw, E, self.name)
    else: _check_exact(np.zeros(1), Iw, E, self.name)
    L = [f"def {self.name}(data0, data1, data2, data3):"] + _picks(N, E, K, "data2", None) + \
        _offsets(N, K, Iw, var, (nk, RS))
    # Aᵀ chunks of every row (the dense kernel's stationary operands): at[0:kt, i N:(i+1) N] = h[:, i 128:(i+1) 128]ᵀ
    L += [f"  at = nl.ndarray(({min(D, TK)}, {nk * N}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  for i in range({nk}):",
          f"    kt = min({TK}, {D} - i * {TK})",
          f"    nisa.dma_copy(dst=at[0:kt, i * {N}:i * {N} + {N}], src=data1.ap(pattern=[[1, kt], [{D}, {N}]], offset=i * {TK}))"]
    sa = "at"
    if self.bf16:
      L += [f"  at_bf = nl.ndarray(({min(D, TK)}, {nk * N}), dtype=nl.bfloat16, buffer=nl.sbuf)", f"  nisa.tensor_copy(dst=at_bf, src=at)"]
      sa = "at_bf"
    L += [f"  for n in range({N}):",
          f"    for j in range({K}):",
          f"      nj = n * {K} + j"]
    if var == "vec": L += _vtable(nk, "      ")
    L += [f"      for s in range(2):",
          f"        for c0 in range(0, {Iw}, {TN}):",
          f"          nt = min({TN}, {Iw} - c0)",
          f"          ps = nl.ndarray(({N}, nt), dtype=nl.float32, buffer=nl.psum)",
          f"          for i in range({nk}):",
          f"            kt = min({TK}, {D} - i * {TK})"]
    static = f"s * {E * Iw} + c0" if var == "vec" else f"i * {TK * RS} + s * {E * Iw} + c0"
    L += _wload(var, "data3", RS, "kt", "nt", static, "i", "            ", wdt)
    L += [f"            nisa.nc_matmul(dst=ps, stationary={sa}[0:kt, i * {N}:i * {N} + {N}], moving=bt, accumulate=(i > 0))",
          f"          ot = nl.ndarray(({N}, nt), dtype=nl.float32, buffer=nl.sbuf)",
          f"          nisa.tensor_copy(dst=ot, src=ps)",
          f"          nisa.dma_copy(dst=data0.ap(pattern=[[{2 * Iw}, 1], [1, nt]], offset=nj * {2 * Iw} + s * {Iw} + c0), src=ot[n:n + 1, 0:nt])",
          f"  return data0"]
    return NKIProgram(self.name, HEADER + "\n".join(L) + "\n", 4, (0,),
                      {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.float32, 3: dtypes.bfloat16 if self.bf16 else dtypes.float32})

@dataclass(frozen=True)
class MoeDownSparse:
  """Y (N, D) f32 = per row the sum over its chosen experts (increasing order) of (w * a) @ Wd[expert rows] (data0=Y,
  data1=A (N, K, Iw) f32 = silu(gate) * up of the picks in G's order, data2=KEY (N, E), data3=W (N, E) the routing
  weights, data4=Wd (E Iw, D) K-major, bf16 or f32)."""
  N: int
  D: int
  E: int
  Iw: int
  K: int
  bf16: bool = True
  var: str = "vec"
  @property
  def name(self) -> str: return f"fz_moe_down_{self.var}_{self.N}_{self.D}_{self.E}_{self.Iw}_{self.K}" + ("_bf16" if self.bf16 else "")
  def render(self) -> NKIProgram:
    N, D, E, Iw, K, var = self.N, self.D, self.E, self.Iw, self.K, self.var
    if var not in VARIANTS or N * K > TN: raise NKIUnsupported(f"{self.name}: variant / {N} x {K} picks")
    nc = -(-Iw // TK)
    wdt = "nl.bfloat16" if self.bf16 else "nl.float32"
    if var == "vec": _check_exact((np.arange(nc)[:, None] * TK + np.arange(TK)[None, :]) * D, Iw * D, E, self.name)
    else: _check_exact(np.zeros(1), Iw * D, E, self.name)
    L = [f"def {self.name}(data0, data1, data2, data3, data4):"] + _picks(N, E, K, "data2", "data3") + \
        _offsets(N, K, Iw * D, var, (nc, D))
    # the routing weights on every partition: wb[:, j N + n] = w of row n's j-th pick
    L += [f"  wbp = nl.ndarray(({TK}, {K * N}), dtype=nl.float32, buffer=nl.psum)",
          f"  nisa.nc_matmul(dst=wbp, stationary=ones, moving=wr, accumulate=False)",
          f"  wb = nl.ndarray(({TK}, {K * N}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  nisa.tensor_copy(dst=wb, src=wbp)"]
    # stationary operands, shared by every row: aw[0:kt, (j nc + c) N + m] = w[m, j] * a[m, j, c 128 + p]
    W = K * nc * N
    L += [f"  at = nl.ndarray(({min(Iw, TK)}, {W}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  aw = nl.ndarray(({min(Iw, TK)}, {W}), dtype=nl.float32, buffer=nl.sbuf)",
          f"  for j in range({K}):",
          f"    for c in range({nc}):",
          f"      kt = min({TK}, {Iw} - c * {TK})",
          f"      q = (j * {nc} + c) * {N}",
          f"      nisa.dma_copy(dst=at[0:kt, q:q + {N}], src=data1.ap(pattern=[[1, kt], [{K * Iw}, {N}]], offset=j * {Iw} + c * {TK}))",
          f"      nisa.tensor_tensor(dst=aw[0:kt, q:q + {N}], data1=at[0:kt, q:q + {N}], data2=wb[0:kt, j * {N}:j * {N} + {N}], op=nl.multiply)"]
    sa = "aw"
    if self.bf16:
      L += [f"  aw_bf = nl.ndarray(({min(Iw, TK)}, {W}), dtype=nl.bfloat16, buffer=nl.sbuf)", f"  nisa.tensor_copy(dst=aw_bf, src=aw)"]
      sa = "aw_bf"
    L += [f"  for n in range({N}):"]
    if var == "vec":   # this row's offset tables, one per pick: vt[:, j nc + c]
      L += [f"    vt = nl.ndarray(({TK}, {K * nc}), dtype=nl.int32, buffer=nl.sbuf)",
            f"    for j in range({K}):",
            f"      nj = n * {K} + j",
            f"      vf = nl.ndarray(({TK}, {nc}), dtype=nl.float32, buffer=nl.sbuf)",
            f"      nisa.tensor_scalar(dst=vf, data=base, op0=nl.add, operand0=eo[:, nj:nj + 1])",
            f"      nisa.tensor_copy(dst=vt[:, j * {nc}:j * {nc} + {nc}], src=vf)"]
    L += [f"    for d0 in range(0, {D}, {TN}):",
          f"      nt = min({TN}, {D} - d0)",
          f"      ps = nl.ndarray(({N}, nt), dtype=nl.float32, buffer=nl.psum)",
          f"      for j in range({K}):",
          f"        nj = n * {K} + j",
          f"        for c in range({nc}):",
          f"          kt = min({TK}, {Iw} - c * {TK})",
          f"          q = (j * {nc} + c) * {N}"]
    if var == "vec":
      L += [f"          bt = nl.ndarray((kt, nt), dtype={wdt}, buffer=nl.sbuf)",
            f"          nisa.dma_copy(dst=bt, src=data4.ap(pattern=[[1, kt], [1, nt]], offset=d0, vector_offset=vt[0:kt, j * {nc} + c:j * {nc} + c + 1], indirect_dim=0))"]
    else: L += _wload(var, "data4", D, "kt", "nt", f"c * {TK * D} + d0", "", "          ", wdt)
    L += [f"          nisa.nc_matmul(dst=ps, stationary={sa}[0:kt, q:q + {N}], moving=bt, accumulate=(j * {nc} + c > 0))",
          f"      ot = nl.ndarray(({N}, nt), dtype=nl.float32, buffer=nl.sbuf)",
          f"      nisa.tensor_copy(dst=ot, src=ps)",
          f"      nisa.dma_copy(dst=data0.ap(pattern=[[{D}, 1], [1, nt]], offset=n * {D} + d0), src=ot[n:n + 1, 0:nt])",
          f"  return data0"]
    dts = {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.float32, 3: dtypes.float32, 4: dtypes.bfloat16 if self.bf16 else dtypes.float32}
    return NKIProgram(self.name, HEADER + "\n".join(L) + "\n", 5, (0,), dts)

# ---------------------------------------------------------------- numpy references (tests, the chip self-test)

def picks_ref(key: np.ndarray, K: int) -> np.ndarray:
  """(N, K) int: row n's chosen experts in increasing order (key[n, e] == j)."""
  key = np.asarray(key)
  out = np.zeros((key.shape[0], K), np.int64)
  for j in range(K): out[:, j] = ((key == j) * np.arange(key.shape[1])[None, :]).sum(-1)
  return np.clip(out, 0, key.shape[1] - 1)

def gate_up_ref(h: np.ndarray, key: np.ndarray, wgu: np.ndarray, E: int, Iw: int, K: int, bf16: bool) -> np.ndarray:
  """MoeGateUpSparse in float64 (bf16: the activation rounded to bf16 first; wgu given as float values)."""
  from fusezap.codegen.nki_bf16 import bf16_round
  h = (bf16_round(h) if bf16 else np.asarray(h, np.float32)).astype(np.float64)
  w = np.asarray(wgu)
  ex = picks_ref(key, K)
  G = np.zeros((h.shape[0], K, 2 * Iw))
  for n in range(h.shape[0]):
    for j in range(K):
      e = ex[n, j]
      G[n, j] = h[n] @ np.concatenate([w[:, e * Iw:(e + 1) * Iw], w[:, E * Iw + e * Iw:E * Iw + (e + 1) * Iw]], 1).astype(np.float64)
  return G

def down_ref(a: np.ndarray, key: np.ndarray, wts: np.ndarray, wd: np.ndarray, Iw: int, K: int, bf16: bool) -> np.ndarray:
  """MoeDownSparse in float64 (the weighted activation a * w in float32, rounded to bf16 with bf16)."""
  from fusezap.codegen.nki_bf16 import bf16_round
  ex = picks_ref(key, K)
  wd = np.asarray(wd)
  Y = np.zeros((a.shape[0], wd.shape[1]))
  for n in range(a.shape[0]):
    for j in range(K):
      e = ex[n, j]
      aw = (np.asarray(a[n, j], np.float32) * np.float32(wts[n, e])).astype(np.float32)
      Y[n] += (bf16_round(aw) if bf16 else aw).astype(np.float64) @ wd[e * Iw:(e + 1) * Iw].astype(np.float64)
  return Y
