"""I91 (I81 D9): an implicit-GEMM conv2d NKI kernel for the scheduler's "conv" match (fusezap/schedule.py _match_conv), so a
composed Tensor.conv2d on a NEURON tier runs as one kernel that reads X directly instead of writing the (C*T, P) im2col
buffer (the FLUX.2 VAE's largest is 9.0 GiB at 1024^2, docs/I96_DIFFUSION_NEURON_SCOPE.md §2).

  out[o*so + p*sp] = sum_k A[oa + o*sam + k*sak] * col[k, p], col[k, p] = X[c, ho*sh + i*dh - ph, wo*sw + j*dw - pw]
  (0 outside X), k = c*T + i*kw + j, p = ho*Wo + wo, X a contiguous (C, H, W) float32 buffer; params CONV_PARAMS.

The kernel (`fz_conv_...(data0=out, data1=A, data2=X)`, float32, TensorE):
  - per block of <= 128 output channels (m0): the stationary Wᵀ tiles, one per tap t and block of <= 128 input channels
    (c0): Wᵀ[c, o] = A[o, (c0 + c)*T + t], (cc, mt) with the channels on the partitions (an element-strided DMA, as
    render_matmul's Aᵀ without rows); kept in SBUF across the pixel tiles while they take <= 64 KiB per partition, else
    reloaded per pixel tile;
  - per pixel tile of <= 512 outputs (R whole output rows when Wo <= 512, else 512-column segments of one row), one PSUM
    bank (mt, R * Wn) accumulates, tap by tap and channel block by channel block, nc_matmul(Wᵀ tile, moving X window): the
    moving tile (cc, R * Wn) is X's shifted window for that tap, DMAed as one (cc, rows, cols) access pattern on both
    sides (HBM: channel stride H*W, row stride sh*W, column stride sw; SBUF: the tile's free axis viewed as (R, Wn)) over
    the tap's valid rows and columns (computed at trace time); the rest of the tile is zero (memset first, only when the
    window is partial: the padding). A tap with no valid position in the tile is skipped.
  - evicted to SBUF and stored with the output strides.
The summation order is tap-major (t, then c), the composed matmul's is channel-major (k = c*T + t in 128-element chunks), so
float32 results agree to rounding, not bit for bit (as METAL's render_conv). FZ_NEURON_CONV_ROWDMA=1: the X windows as one
DMA per valid output row (2-D access patterns only), in case the 3-D on-chip pattern does not compile.

Switch: FZ_NEURON_CONV=1 offers "conv" to NEURON tiers (default off: not run on a NeuronCore yet; NKISHIM tests in
tests/test_nki_conv.py; NKISIM + inf2 compile in scripts/nki_i91_pod.sh). Python loops in the source are unrolled at trace
time (the kernel has no runtime integers), so a 1024^2 VAE conv is a large kernel: the pod job measures its compile.
"""
from __future__ import annotations
import os
from fusezap.dtype import dtypes
from fusezap.codegen.nki import NKIProgram, NKIUnsupported

TK, TM, TN = 128, 128, 512
W_KEEP = 64 * 1024   # bytes per partition for keeping a row block's Wᵀ tiles in SBUF across the pixel tiles

def conv_enabled() -> bool: return os.environ.get("FZ_NEURON_CONV", "0") not in ("", "0")
def conv_rowdma() -> bool: return os.environ.get("FZ_NEURON_CONV_ROWDMA", "0") not in ("", "0")

def render_conv(params: tuple[int, ...], rowdma: bool | None = None) -> NKIProgram:
  """schedule.CONV_PARAMS -> `fz_conv_<O>_<C>x<H>x<W>_<kh>x<kw>...(data0=out, data1=A, data2=X)` (see the module doc)."""
  from fusezap.codegen.nki_kernels import HEADER, _dma_in
  M, N, K, sam, sak, oa, so, sp, C, H, W, Wo, kh, kw, sh, sw, dh, dw, ph, pw = (int(x) for x in params)
  T = kh * kw
  if min(M, N, K, C, H, W, Wo, kh, kw, sh, sw, dh, dw) <= 0 or ph < 0 or pw < 0: raise NKIUnsupported(f"fz_conv: bad params {params}")
  if K != C * T or N % Wo: raise NKIUnsupported(f"fz_conv: K {K} != C*T {C * T} or N {N} not whole rows of {Wo}")
  rowdma = conv_rowdma() if rowdma is None else rowdma
  Ho = N // Wo
  R, Wt = (max(1, TN // Wo), Wo) if Wo <= TN else (1, TN)
  ncc = -(-C // TK)
  keep = T * ncc * min(M, TM) * 4 <= W_KEEP
  name = f"fz_conv_{M}_{C}x{H}x{W}_{kh}x{kw}_s{sh}{sw}_d{dh}{dw}_p{ph}{pw}_{Ho}x{Wo}" + ("_r" if rowdma else "")
  w_off = f"{oa} + m0 * {sam} + (c0 * {T} + t) * {sak}"
  def wload(tile: str, cols: str, ind: str) -> list[str]:   # Wᵀ[c, o] of tap t, channels c0 .. c0 + cc
    return _dma_in(tile, cols, "data1", T * sak, "cc", sam, "mt", w_off, ind)
  L = [f"def {name}(data0, data1, data2):",
       f"  for m0 in range(0, {M}, {TM}):",
       f"    mt = min({TM}, {M} - m0)"]
  if keep:   # every tap's and channel block's Wᵀ, side by side: (t, block) at wt[0:cc, (t * {ncc} + b) * mt ...]
    L += [f"    wt = nl.ndarray(({min(C, TK)}, {T * ncc} * mt), dtype=nl.float32, buffer=nl.sbuf)",
          f"    for t in range({T}):",
          f"      for b in range({ncc}):",
          f"        c0 = b * {TK}",
          f"        cc = min({TK}, {C} - c0)"]
    L += wload("wt", "(t * {0} + b) * mt:(t * {0} + b) * mt + mt".format(ncc), "        ")
  L += [f"    for ho0 in range(0, {Ho}, {R}):",
        f"      nr = min({R}, {Ho} - ho0)",
        f"      for wo0 in range(0, {Wo}, {Wt}):",
        f"        nw = min({Wt}, {Wo} - wo0)",
        f"        ps = nl.ndarray((mt, nr * nw), dtype=nl.float32, buffer=nl.psum)",
        f"        first = True",
        f"        for t in range({T}):",
        f"          i, j = t // {kw}, t % {kw}",
        # the tap's valid output rows [ha, hb) and columns [wa, wb) inside this tile (input row ho*sh + i*dh - ph in [0, H))
        f"          ha, hb = max(ho0, -((i * {dh} - {ph}) // {sh})), min(ho0 + nr, ({H} - 1 + {ph} - i * {dh}) // {sh} + 1)",
        f"          wa, wb = max(wo0, -((j * {dw} - {pw}) // {sw})), min(wo0 + nw, ({W} - 1 + {pw} - j * {dw}) // {sw} + 1)",
        f"          if ha < hb and wa < wb:",   # else the tap has no valid position in this tile: skipped
        f"           for b in range({ncc}):",
        f"            c0 = b * {TK}",
        f"            cc = min({TK}, {C} - c0)"]
  if keep: stat = "wt[0:cc, (t * {0} + b) * mt:(t * {0} + b) * mt + mt]".format(ncc)
  else:
    L += [f"            wt = nl.ndarray((cc, mt), dtype=nl.float32, buffer=nl.sbuf)"]
    L += wload("wt", "0:mt", "            ")
    stat = "wt"
  L += [f"            xt = nl.ndarray((cc, nr * nw), dtype=nl.float32, buffer=nl.sbuf)",
        f"            if hb - ha < nr or wb - wa < nw: nisa.memset(dst=xt, value=0.0)",
        f"            xo = c0 * {H * W} + (ha * {sh} + i * {dh} - {ph}) * {W} + (wa * {sw} + j * {dw} - {pw})"]
  if rowdma:
    L += [f"            for r in range(ha, hb):",
          f"              nisa.dma_copy(dst=xt[0:cc, (r - ho0) * nw + wa - wo0:(r - ho0) * nw + wb - wo0], "
          f"src=data2.ap(pattern=[[{H * W}, cc], [{sw}, wb - wa]], offset=xo + (r - ha) * {sh * W}))"]
  else:
    L += [f"            nisa.dma_copy(dst=xt.ap(pattern=[[nr * nw, cc], [nw, hb - ha], [1, wb - wa]], offset=(ha - ho0) * nw + wa - wo0), "
          f"src=data2.ap(pattern=[[{H * W}, cc], [{sh * W}, hb - ha], [{sw}, wb - wa]], offset=xo))"]
  L += [f"            nisa.nc_matmul(dst=ps, stationary={stat}, moving=xt, accumulate=(not first))",
        f"            first = False",
        f"        ot = nl.ndarray((mt, nr * nw), dtype=nl.float32, buffer=nl.sbuf)",
        f"        if first: nisa.memset(dst=ot, value=0.0)",
        f"        else: nisa.tensor_copy(dst=ot, src=ps)",
        f"        nisa.dma_copy(dst=data0.ap(pattern=[[{so}, mt], [{sp}, nr * nw]], offset=m0 * {so} + (ho0 * {Wo} + wo0) * {sp}), src=ot)",
        f"  return data0"]
  return NKIProgram(name, HEADER + "\n".join(L) + "\n", 3, (0,), {0: dtypes.float32, 1: dtypes.float32, 2: dtypes.float32})
