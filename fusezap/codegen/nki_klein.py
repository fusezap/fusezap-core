"""I93: NKI kernels for the FLUX.2 [klein] transformer on NEURON (fusezap/serve/models/flux2.py), each behind its switch.

RopeSpec (FZ_KLEIN_NEURON_ROPE=1): the q / k path from the per-head RMSNorm output to the attention kernel's float32
(B, H, S, D) input, one launch per q / k instead of four (five with the double block's cat). The generic lowering of
diffusers' interleaved RoPE (flux2.Flux2Transformer._rope) upcasts x, builds rot = stack([-x_odd, x_even], -1) and then
x * cos + rot * sin. The stack's kernel has 2 elements on the free axis and S * H * D / 2 rows, so every DMA moves 128
single floats (element-granular descriptors), and the negate reads with a 2-element stride: at 1024^2 those two kernels
issue ~1.1 M DMAs per step, the largest non-attention cost of a step (scripts/i93_klein_cost.py).

RopeSpec keeps the pairs on the free axis of a (tokens, heads x D) tile and reads the even / odd lanes through stride-2
access patterns on SBUF, so every HBM transfer is whole rows. The arithmetic is the generic kernels', in the same order
and dtype, so the output is bitwise the composed path's:
    xf = f32(x)                         (x bf16, or float32 when the model runs in float32)
    a  = xf * cos                       (every lane)
    r_even = xf_odd * sin_even,  r_odd = xf_even * sin_odd
    o_even = a_even - r_even            (= a + (-x_odd) * sin_even: IEEE (-u) * v = -(u * v) and a + (-b) = a - b)
    o_odd  = a_odd + r_odd
    out = f32(bf16(o))                  (Flux2Transformer._rope's cast back to x's dtype, then sdpa's .float(); skipped in float32)
written in the (B, H, S, D) layout the flash attention kernel reads (the composed path's permute + .float() kernel).
The double block's q = cat(text q, image q) is folded in: the rows come from up to two inputs (text rows first).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from fusezap.dtype import dtypes
from fusezap.codegen.nki import NKIProgram, NKIUnsupported

def klein_rope() -> bool:
  """FZ_KLEIN_NEURON_ROPE=1: Flux2Transformer on a NEURON tier runs q / k RoPE as one RopeSpec launch each (default 0)."""
  return os.environ.get("FZ_KLEIN_NEURON_ROPE", "0") not in ("", "0")

HEAD_CHUNK = 4   # at most this many heads per column chunk of a 128-row tile (4 x 128 float32 = 2 KiB per partition per tile)

@dataclass(frozen=True)
class RopeSpec:
  """out (B, H, S, D) float32 = RoPE(cat(x1, x2) over the rows), x_i (B, S_i, H * D) in `xdt` ("bf16" or "f32"), cos / sin
  (S, D) float32 (pairs (2i, 2i+1) rotate together). Arguments: data0 = out, data1 = x1, [data2 = x2], then cos, sin."""
  B: int
  parts: tuple[int, ...]   # S_i per input, in row order
  H: int
  D: int
  xdt: str = "bf16"
  @property
  def S(self) -> int: return sum(self.parts)
  @property
  def name(self) -> str: return f"fz_klein_rope_{self.B}_{'_'.join(map(str, self.parts))}_{self.H}_{self.D}_{self.xdt}"
  def render(self) -> NKIProgram:
    B, H, D, S, parts = self.B, self.H, self.D, self.S, self.parts
    if D % 2 or not 1 <= len(parts) <= 2 or self.xdt not in ("bf16", "f32") or min(parts) < 1:
      raise NKIUnsupported(f"{self.name}: RopeSpec takes an even head dim, one or two inputs, bf16 / f32")
    HD, xnl = H * D, "bfloat16" if self.xdt == "bf16" else "float32"
    n = len(parts)
    ci, si = n + 1, n + 2
    args = ", ".join(f"data{i}" for i in range(n + 3))
    L = ["import nki", "import nki.language as nl", "import nki.isa as nisa", "", "@nki.jit", f"def {self.name}({args}):"]
    sb = lambda nm, shape, dt="float32": f"      {nm} = nl.ndarray({shape}, dtype=nl.{dt}, buffer=nl.sbuf)"
    ev = lambda t, w, o: f"{t}.ap(pattern=[[{w}, pn], [2, {w // 2}]], offset={o})"   # the even (o=0) / odd (o=1) lanes of a (pn, w) tile
    L += [f"  for b in range({B}):", f"    for p0 in range(0, {S}, 128):", f"      pn = min(128, {S} - p0)"]
    hc = max(d for d in range(1, HEAD_CHUNK + 1) if H % d == 0)   # heads per chunk: every chunk the same width
    w = hc * D
    # cos / sin repeated over a chunk's heads by the DMA (a 0-stride level), loaded once per row tile: every chunk reads them
    L += [sb("cs", f"(pn, {w})"), f"      nisa.dma_copy(dst=cs, src=data{ci}.ap(pattern=[[{D}, pn], [0, {hc}], [1, {D}]], offset=p0 * {D}))",
          sb("sn", f"(pn, {w})"), f"      nisa.dma_copy(dst=sn, src=data{si}.ap(pattern=[[{D}, pn], [0, {hc}], [1, {D}]], offset=p0 * {D}))"]
    L += [sb("xb", f"(pn, {HD})", xnl)]
    r0 = 0
    for i, Si in enumerate(parts):   # rows [r0, r0 + Si) of the step come from data<i+1>
      L += [f"      a{i}, e{i} = max(p0, {r0}), min(p0 + pn, {r0 + Si})",
            f"      if a{i} < e{i}: nisa.dma_copy(dst=xb[a{i} - p0:e{i} - p0, :], src=data{i + 1}.ap(pattern=[[{HD}, e{i} - a{i}], [1, {HD}]], "
            f"offset=b * {Si * HD} + (a{i} - {r0}) * {HD}))"]
      r0 += Si
    for h0 in range(0, H, hc):
      c0 = h0 * D
      L += [sb("xf", f"(pn, {w})"), f"      nisa.tensor_copy(dst=xf, src=xb[:, {c0}:{c0 + w}])"]
      L += [sb("xa", f"(pn, {w})"), "      nisa.tensor_tensor(dst=xa, data1=xf, data2=cs, op=nl.multiply)",
            sb("xr", f"(pn, {w})"),
            f"      nisa.tensor_tensor(dst={ev('xr', w, 0)}, data1={ev('xf', w, 1)}, data2={ev('sn', w, 0)}, op=nl.multiply)",
            f"      nisa.tensor_tensor(dst={ev('xr', w, 1)}, data1={ev('xf', w, 0)}, data2={ev('sn', w, 1)}, op=nl.multiply)",
            sb("xo", f"(pn, {w})"),
            f"      nisa.tensor_tensor(dst={ev('xo', w, 0)}, data1={ev('xa', w, 0)}, data2={ev('xr', w, 0)}, op=nl.subtract)",
            f"      nisa.tensor_tensor(dst={ev('xo', w, 1)}, data1={ev('xa', w, 1)}, data2={ev('xr', w, 1)}, op=nl.add)"]
      out = "xo"
      if self.xdt == "bf16":   # _rope's cast back to bf16, then sdpa's .float()
        L += [sb("xh", f"(pn, {w})", "bfloat16"), "      nisa.tensor_copy(dst=xh, src=xo)", sb("xq", f"(pn, {w})"), "      nisa.tensor_copy(dst=xq, src=xh)"]
        out = "xq"
      L += [f"      nisa.dma_copy(dst=data0.ap(pattern=[[{D}, pn], [{S * D}, {hc}], [1, {D}]], offset=b * {H * S * D} + {h0 * S * D} + p0 * {D}), src={out})"]
    L += ["  return data0", ""]
    dts = {0: dtypes.float32, ci: dtypes.float32, si: dtypes.float32}
    for i in range(n): dts[i + 1] = dtypes.bfloat16 if self.xdt == "bf16" else dtypes.float32
    return NKIProgram(self.name, "\n".join(L), n + 3, (0,), dts)

# ---------------------------------------------------------------- MatmulBF16G: weight tiles reused across row blocks

def klein_mm_group() -> int:
  """FZ_KLEIN_NEURON_MMG=<G>: klein's bf16 linears with M >= 256 rows run as MatmulBF16G with up to G row blocks per weight
  tile (default 0: nki_bf16.MatmulBF16)."""
  try: return max(0, int(os.environ.get("FZ_KLEIN_NEURON_MMG", "0") or 0))
  except ValueError: return 0

MMG_SBUF = 64 * 1024   # bytes per partition for the group's bf16 Aᵀ chunks (render_matmul's A_KEEP budget)

@dataclass(frozen=True)
class MatmulBF16G:
  """C (M, N) float32 = A (M, K) float32 @ W (K, N) bf16 K-major: MatmulBF16's instructions per output tile (Aᵀ chunks by
  TensorE transposes, rounded to bf16 once per row block; 512-column moving weight tiles; PSUM accumulation over K in
  order), with the loops reordered so one weight tile feeds G row blocks (G PSUM banks accumulate side by side) instead of
  being loaded again for each 128-row block. MatmulBF16 reads the whole weight ceil(M / 128) times: at klein's 1024^2
  (M = 4608) the single block's 3072 x 27648 projection is 6.1 GB of weight DMA per call; G = 4 makes it 1.5 GB.
  Every output element is the same nc_matmul sequence on the same operands, so C is bitwise MatmulBF16's."""
  M: int
  N: int
  K: int
  G: int = 4
  @property
  def g(self) -> int:
    from fusezap.codegen.nki_kernels import TK, TM
    nk = -(-self.K // TK)
    return max(1, min(self.G, 6, MMG_SBUF // (nk * TM * 2), -(-self.M // TM)))   # <= 6 PSUM banks of 8 (transposes use another)
  @property
  def name(self) -> str: return f"fz_matmul_1_{self.M}_{self.N}_{self.K}_bf16_g{self.g}"
  def render(self) -> NKIProgram:
    from fusezap.codegen.nki_kernels import TK, TM, TN, HEADER
    M, N, K, G = self.M, self.N, self.K, self.g
    nk = -(-K // TK)
    L = [f"def {self.name}(data0, data1, data2):"]
    def group(base: str, cnt: int, ind: str) -> None:
      """cnt row blocks starting at row `base` (a source expression)."""
      for j in range(cnt):   # every block's Aᵀ chunks, as MatmulBF16 builds them (float32 by transposes, then one bf16 copy)
        L.extend([f"{ind}m{j} = {base} + {j * TM}", f"{ind}mt{j} = min({TM}, {M} - m{j})",
                  f"{ind}at{j} = nl.ndarray(({min(K, TK)}, {nk} * mt{j}), dtype=nl.float32, buffer=nl.sbuf)",
                  f"{ind}for i in range({nk}):", f"{ind}  kt = min({TK}, {K} - i * {TK})",
                  f"{ind}  ar = nl.ndarray((mt{j}, kt), dtype=nl.float32, buffer=nl.sbuf)",
                  f"{ind}  nisa.dma_copy(dst=ar, src=data1.ap(pattern=[[{K}, mt{j}], [1, kt]], offset=m{j} * {K} + i * {TK}))",
                  f"{ind}  ap = nl.ndarray((kt, mt{j}), dtype=nl.float32, buffer=nl.psum)",
                  f"{ind}  nisa.nc_transpose(dst=ap, data=ar)",
                  f"{ind}  nisa.tensor_copy(dst=at{j}[0:kt, i * mt{j}:i * mt{j} + mt{j}], src=ap)",
                  f"{ind}at{j}_bf = nl.ndarray(({min(K, TK)}, {nk} * mt{j}), dtype=nl.bfloat16, buffer=nl.sbuf)",
                  f"{ind}nisa.tensor_copy(dst=at{j}_bf, src=at{j})"])
      L.extend([f"{ind}for n0 in range(0, {N}, {TN}):", f"{ind}  nt = min({TN}, {N} - n0)"])
      L.extend(f"{ind}  ps{j} = nl.ndarray((mt{j}, nt), dtype=nl.float32, buffer=nl.psum)" for j in range(cnt))
      L.extend([f"{ind}  for i in range({nk}):", f"{ind}    kt = min({TK}, {K} - i * {TK})",
                f"{ind}    bt = nl.ndarray((kt, nt), dtype=nl.bfloat16, buffer=nl.sbuf)",
                f"{ind}    nisa.dma_copy(dst=bt[0:kt, :], src=data2.ap(pattern=[[{N}, kt], [1, nt]], offset=i * {TK * N} + n0))"])
      L.extend(f"{ind}    nisa.nc_matmul(dst=ps{j}, stationary=at{j}_bf[0:kt, i * mt{j}:i * mt{j} + mt{j}], moving=bt, accumulate=(i > 0))" for j in range(cnt))
      for j in range(cnt):
        L.extend([f"{ind}  ot{j} = nl.ndarray((mt{j}, nt), dtype=nl.float32, buffer=nl.sbuf)", f"{ind}  nisa.tensor_copy(dst=ot{j}, src=ps{j})",
                  f"{ind}  nisa.dma_copy(dst=data0.ap(pattern=[[{N}, mt{j}], [1, nt]], offset=m{j} * {N} + n0), src=ot{j})"])
    nb = -(-M // TM)
    full, tail = divmod(nb, G)
    if full:
      L.append(f"  for g0 in range(0, {full * G * TM}, {G * TM}):")
      group("g0", G, "    ")
    if tail: group(str(full * G * TM), tail, "  ")
    L.append("  return data0")
    return NKIProgram(self.name, HEADER + "\n".join(L) + "\n", 3, (0,), {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.bfloat16})
