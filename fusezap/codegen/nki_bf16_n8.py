"""N8 (I70): the bf16 decode path's weight streaming, closer to AWS NxD Inference on trn1 (Qwen2.5-0.5B, 1 NeuronCore).

What NxDI 0.10 does for Qwen2 decode in bf16 at TP=1 (its source; docs/NKI_BACKEND.md "N8 bf16" has the citations): one
token-generation graph (all layers, final norm, lm_head, on-device argmax, KV cache update by aliasing) compiled by
neuronx-cc from XLA (no NKI kernels by default for Qwen2: fused QKV, the attention / MLP / QKV kernels and flash decoding
are all off), with every parameter, activation, the residual stream and the KV cache in bf16 (RMSNorm and softmax compute
in float32 inside and round back), and the KV scan bounded by the sequence bucket (128 / 256 / 512 for max_length 512).
trn1 (I68): NxDI 5.11 ms per token, greedy 10/24 prompts vs the float32 reference (below the repo's bf16 bar 0.70);
FuseZap bf16 9.49 ms at FUSE=4, 22/24.

Where FuseZap's bf16 time goes (trn1, I68): a decoder layer 0.239 ms for ~30 MB of bf16 weights (~125 GB/s), against
0.314 ms for 60 MB in float32 (~190 GB/s, TensorE-bound). bf16 made TensorE ~4x cheaper but left the weight DMAs at the
float32 path's shape: one DMA per 128 K rows x 512 columns, i.e. 1 KiB contiguous runs per partition (float32: 2 KiB),
~31 k runs per layer. The lm_head in bf16 is MatmulBF16 (render_matmul): 2079 DMAs of 1 KiB runs for 272 MB. The probes of
N7 step 4 put one (128, 7168 B) block per DMA at ~276 GB/s. So the switches here change only how the bf16 weights are
streamed, never the arithmetic (each is bitwise equal to the bf16 path it replaces, by construction and on NKISHIM):

  FZ_NEURON_BF16_WN=<cols>   LayerBF16's weight matmuls load moving tiles of <cols> columns (1024 / 2048 / 4096; 512 = off;
                             default 4096 on a whole untiled layer that fits, see WN_DEFAULT): a (128, cols) bf16 tile per K chunk is one DMA of cols * 2 B runs, and its 512-column
                             slices feed the same nc_matmul as before, each into its own PSUM tile; every PSUM accumulation
                             chain (one 512-column output block over the K chunks in order) is the same instructions on the
                             same values, so the same bits; only the DMA count (/ cols / 512) and the run length change.
  FZ_NEURON_BF16_PF=<blocks> the first <blocks> wide gate_up blocks (every K chunk of <cols> columns) load at the start of
                             attention, when the DMA queues are otherwise idle (the norm -> qkv -> RoPE -> KV write ->
                             attention -> o_proj chain is VectorE / TensorE bound); the matmuls read those tiles. SBUF
                             permitting (nki_bf16.layer_sbuf_bytes + the prefetched tiles <= FZ_NEURON_LAYER_SBUF): the
                             count is clamped, and the kernel name says how many (_p<k>).
  FZ_NEURON_BF16_PFQO=1      every qkv weight tile loads at the top of the kernel (with the input-only loads) and every
                             o_proj tile at the start of attention (_q in the name).
  FZ_NEURON_BF16_HEAD=1      (default without TP; 0 = MatmulBF16) the lm_head as HeadBF16: nki_kernels.HeadSpec's kernel with bf16 weight tiles (head_tiles'
                             layout, bf16, relaid once at load; the K-major bf16 lm_head is not kept) and the Aᵀ chunks rounded
                             to bf16 as MatmulBF16 rounds them: the same nc_matmul sequence as MatmulBF16 (512-column moving
                             tiles, PSUM accumulation over the K chunks in order), so the same logits; one DMA per block of
                             FZ_NEURON_BF16_HEAD_W columns (default 2048: 28 KiB runs per partition, 75 DMAs for Qwen2.5's
                             151936 x 896, against MatmulBF16's 2079 DMAs of 1 KiB runs).

Not done here, and why (see the report): the bf16 residual stream / activations NxDI keeps (most of its accuracy loss; in a
LayerSpec the activations never leave SBUF, so storing them in bf16 saves nothing), a bf16 KV pool (attention reads ~0.5 MB
per layer at 512 slots; its cost is VectorE instructions, not bytes: bf16a was 0.243 vs 0.239 ms), weight-stationary bf16
(TensorE is no longer the bound in bf16: ~50 us of ~240 per layer). NxDI's 256-slot bucket for this workload is
FZ_NEURON_CTX_MIN=256 (an existing switch; not bitwise vs 512 slots: the last key block is narrower; token-identical
24/24 on trn1 in the N8 run, kept opt-in for now).
"""
from __future__ import annotations
import os, re
from dataclasses import dataclass
from fusezap.dtype import dtypes
from fusezap.codegen.nki import NKIProgram, NKIUnsupported

TK, TN = 128, 512

def _env_int(k: str, d: int) -> int:
  v = os.environ.get(k, "").strip()
  return int(v) if v else d

# N8 defaults from trn1 (docs/status/nki-n8-bf16-trn1, Qwen2.5-0.5B bf16): wide weight tiles of 4096 columns (layer 0.178 ms
# at 1 row, 0.185 at 4) and HeadBF16 at W 2048 (1.04 ms vs MatmulBF16's 1.52) were bitwise vs the bf16 members on device and
# token-identical (24/24 prompts) to the bf16 baseline; prefetch (_PF / _PFQO) stays off. layer_bf16w applies the wide default
# only to a whole untiled layer whose SBUF estimate fits (4096, else 2048, else 1024, else off); the tiled layer (I71, its own
# FZ_NEURON_LAYER_WCOLS) and the TP halves (I72) are unchanged. HeadBF16 is the default without TP (under TP: FZ_NEURON_BF16_HEAD=1).
# FZ_NEURON_BF16_WN=512 / FZ_NEURON_BF16_HEAD=0 give I68's bf16 kernels.
WN_DEFAULT = 8 * TN

def bf16_wn() -> int:
  """FZ_NEURON_BF16_WN: columns per weight DMA in LayerBF16 (a multiple of 512, <= 4096; 512 = the unchanged kernel; unset:
  WN_DEFAULT, subject to layer_bf16w's fit check)."""
  wn = _env_int("FZ_NEURON_BF16_WN", WN_DEFAULT)
  if wn % TN or not TN <= wn <= 8 * TN: raise ValueError(f"FZ_NEURON_BF16_WN={wn}: a multiple of {TN} in [{TN}, {8 * TN}]")
  return wn
def bf16_pf() -> int: return max(0, _env_int("FZ_NEURON_BF16_PF", 0))
def bf16_pfqo() -> bool: return os.environ.get("FZ_NEURON_BF16_PFQO", "0") not in ("", "0")
def bf16_head(tp: bool = False) -> bool:
  """FZ_NEURON_BF16_HEAD: HeadBF16 for the bf16 lm_head; unset: on without tensor parallelism, off under TP."""
  v = os.environ.get("FZ_NEURON_BF16_HEAD", "").strip()
  return (not tp) if v == "" else v != "0"
def bf16_head_w() -> int:
  w = _env_int("FZ_NEURON_BF16_HEAD_W", 2048)
  if w % TN or w <= 0: raise ValueError(f"FZ_NEURON_BF16_HEAD_W={w}: a positive multiple of {TN}")
  return w
N8_ENV = ("FZ_NEURON_BF16_WN", "FZ_NEURON_BF16_PF", "FZ_NEURON_BF16_PFQO", "FZ_NEURON_BF16_HEAD", "FZ_NEURON_BF16_HEAD_W")

# ---------------------------------------------------------------- wide weight tiles (+ prefetch) on a rendered layer

# one _mm_rows block as it reads after nki_bf16.bf16_weight_matmuls (the moving tile bt<tag> loaded from a weight argument,
# the stationary <at>_bf): tag t, Nn output columns, M rows, nk K chunks of K, weight w (row stride Nn), eviction into dst
_BLOCK = re.compile(
  rf"^(?P<ind> *)for n0(?P<t>\w+) in range\(0, (?P<Nn>\d+), {TN}\):\n"
  rf" *nt(?P=t) = min\({TN}, (?P=Nn) - n0(?P=t)\)\n"
  rf" *ps(?P=t) = nl\.ndarray\(\((?P<M>\d+), nt(?P=t)\), dtype=nl\.float32, buffer=nl\.psum\)\n"
  rf" *for i(?P=t) in range\((?P<nk>\d+)\):\n"
  rf" *kt(?P=t) = min\({TK}, (?P<K>\d+) - i(?P=t) \* {TK}\)\n"
  rf" *bt(?P=t) = nl\.ndarray\(\(kt(?P=t), nt(?P=t)\), dtype=nl\.(?P<dt>\w+), buffer=nl\.sbuf\)\n"
  rf" *nisa\.dma_copy\(dst=bt(?P=t)\[0:kt(?P=t), :\], src=(?P<w>data\d+)\.ap\(pattern=\[\[(?P=Nn), kt(?P=t)\], \[1, nt(?P=t)\]\], "
  rf"offset=i(?P=t) \* (?P<st>\d+) \+ n0(?P=t)\)\)\n"
  rf" *nisa\.nc_matmul\(dst=ps(?P=t), stationary=(?P<at>\w+)\[0:kt(?P=t), i(?P=t) \* (?P=M):i(?P=t) \* (?P=M) \+ (?P=M)\], "
  rf"moving=bt(?P=t), accumulate=\(i(?P=t) > 0\)\)\n"
  rf"(?P<ind2> *)nisa\.tensor_copy\(dst=(?P<dst>\w+)\[:, n0(?P=t):n0(?P=t) \+ nt(?P=t)\], src=ps(?P=t)\)$", re.M)

def _blocks(src: str) -> dict:
  out = {}
  for m in _BLOCK.finditer(src):
    g = m.groupdict()
    Nn, K = int(g["Nn"]), int(g["K"])
    if int(g["st"]) != TK * Nn or int(g["nk"]) != -(-K // TK): raise NKIUnsupported(f"bf16 wide: block {g['t']} is not K-major ({g})")
    out[g["t"]] = m
  return out

def _wide(m: "re.Match", wn: int, pre: set, pref: dict) -> str:
  """The replacement of one matched block: wide moving tiles of wn columns (one DMA per K chunk and block), each 512-column
  slice into its own PSUM tile (the same accumulation chain per output block as before); blocks in `pre` read the
  prefetched tiles pf<t><b>_<i> (pref: block -> the lines that load them, placed by the caller)."""
  g = m.groupdict()
  ind, t, Nn, M, nk, K, dt, w, at, dst = g["ind"], g["t"], int(g["Nn"]), int(g["M"]), int(g["nk"]), int(g["K"]), g["dt"], g["w"], g["at"], g["dst"]
  L: list[str] = []
  for b, w0 in enumerate(range(0, Nn, wn)):
    nw = min(wn, Nn - w0)
    subs = [(j, s0, min(TN, nw - s0)) for j, s0 in enumerate(range(0, nw, TN))]
    L += [f"{ind}# wide block {b}: columns {w0} .. {w0 + nw - 1} ({len(subs)} PSUM tiles){' prefetched' if b in pre else ''}"]
    L += [f"{ind}ps{t}{b}_{j} = nl.ndarray(({M}, {sw}), dtype=nl.float32, buffer=nl.psum)" for j, s0, sw in subs]
    if b in pre:   # unrolled over the K chunks: tile pf<t><b>_<i> was loaded earlier (pref)
      lines = []
      for i in range(nk):
        kt = min(TK, K - i * TK)
        lines += [f"{ind}pf{t}{b}_{i} = nl.ndarray(({kt}, {nw}), dtype=nl.{dt}, buffer=nl.sbuf)",
                  f"{ind}nisa.dma_copy(dst=pf{t}{b}_{i}[0:{kt}, :], src={w}.ap(pattern=[[{Nn}, {kt}], [1, {nw}]], offset={i * TK * Nn + w0}))"]
        L += [f"{ind}nisa.nc_matmul(dst=ps{t}{b}_{j}, stationary={at}[0:{kt}, {i * M}:{i * M + M}], moving=pf{t}{b}_{i}[0:{kt}, {s0}:{s0 + sw}], "
              f"accumulate={i > 0})" for j, s0, sw in subs]
      pref[b] = lines
    else:
      L += [f"{ind}for i{t} in range({nk}):",
            f"{ind}  kt{t} = min({TK}, {K} - i{t} * {TK})",
            f"{ind}  bt{t} = nl.ndarray((kt{t}, {nw}), dtype=nl.{dt}, buffer=nl.sbuf)",
            f"{ind}  nisa.dma_copy(dst=bt{t}[0:kt{t}, :], src={w}.ap(pattern=[[{Nn}, kt{t}], [1, {nw}]], offset=i{t} * {TK * Nn} + {w0}))"]
      L += [f"{ind}  nisa.nc_matmul(dst=ps{t}{b}_{j}, stationary={at}[0:kt{t}, i{t} * {M}:i{t} * {M} + {M}], moving=bt{t}[0:kt{t}, {s0}:{s0 + sw}], "
            f"accumulate=(i{t} > 0))" for j, s0, sw in subs]
    L += [f"{ind}nisa.tensor_copy(dst={dst}[:, {w0 + s0}:{w0 + s0 + sw}], src=ps{t}{b}_{j})" for j, s0, sw in subs]
  return "\n".join(L)

def wide_weights(prog: NKIProgram, wn: int, pf: int = 0, pfqo: bool = False, suffix: str | None = None) -> NKIProgram:
  """A rendered LayerSpec / LayerBF16 program with its weight matmuls (render_layer's _mm_rows blocks: tags q = qkv, o =
  o_proj, u = gate_up, d = down) on wn-column moving tiles; pf: the first pf gate_up blocks loaded at the start of attention
  (section "# 5."); pfqo: every qkv tile at the top (after the "# 0." loads, or the def line) and every o_proj tile at the
  start of attention. The same nc_matmul per PSUM chain in the same order: bitwise equal to `prog`."""
  if wn % TN or wn < TN: raise NKIUnsupported(f"bf16 wide: {wn} columns is not a multiple of {TN}")
  src = prog.src
  bl = _blocks(src)
  if not bl: raise NKIUnsupported(f"bf16 wide {prog.name}: no K-major weight matmul blocks")
  need = {"u"} if pf else set()
  if pfqo: need |= {"q", "o"}
  if need - set(bl): raise NKIUnsupported(f"bf16 wide {prog.name}: blocks {sorted(need - set(bl))} not found (have {sorted(bl)})")
  top: list[str] = []; att: list[str] = []
  # replace from the last block back so earlier offsets stay valid
  for t, m in sorted(bl.items(), key=lambda x: -x[1].start()):
    Nn = int(m.group("Nn"))
    nb = -(-Nn // wn)
    pre = set(range(nb)) if pfqo and t in ("q", "o") else set(range(min(pf, nb))) if t == "u" else set()
    pref: dict = {}
    src = src[:m.start()] + _wide(m, wn, pre, pref) + src[m.end():]
    lines = [ln for b in sorted(pref) for ln in pref[b]]
    if t == "q": top = lines
    elif t == "o": att = lines + att
    elif t == "u": att = att + lines
  lines = src.split("\n")
  if att:
    k = next((i for i, ln in enumerate(lines) if ln.strip().startswith("# 5. attention")), None)
    if k is None: raise NKIUnsupported(f"bf16 wide {prog.name}: no attention section for the prefetch")
    ind = re.match(r"^(\s*)", lines[k]).group(1)
    lines[k + 1:k + 1] = [f"{ind}# N8 prefetch: o_proj / gate_up weight tiles while attention runs"] + [ind + ln.lstrip() for ln in att]
  if top:
    k = next((i for i, ln in enumerate(lines) if ln.strip().startswith("# 0. input-only loads")), None)
    if k is None: k = next(i for i, ln in enumerate(lines) if ln.startswith("def "))
    lines[k + 1:k + 1] = ["  # N8 prefetch: qkv weight tiles"] + ["  " + ln.lstrip() for ln in top]
  sfx = suffix if suffix is not None else f"_w{wn}" + (f"p{pf}" if pf else "") + ("q" if pfqo else "")
  out = "\n".join(lines)
  name = prog.name + sfx
  out, n = re.subn(rf"\bdef {re.escape(prog.name)}\(", f"def {name}(", out)
  if n != 1: raise NKIUnsupported(f"bf16 wide {prog.name}: kernel definition not found")
  return NKIProgram(name, out, prog.n_bufs, prog.outs, dict(prog.buf_dtypes))

def prefetch_bytes(sp, wn: int, pf: int, pfqo: bool) -> int:
  """SBUF bytes per partition the prefetched bf16 tiles add to a LayerSpec sp (all live from their load to their matmul):
  pf gate_up blocks (D / 128 K chunks of wn columns each), with pfqo every qkv and o_proj tile."""
  D, I, Nq, Ho = sp.D, sp.I, (sp.Hq + 2 * sp.Hkv) * sp.hd, sp.Hq * sp.hd
  def tiles(K: int, Nn: int, nb: int) -> int:   # the first nb blocks of a (K, Nn) weight: ceil(K / 128) tiles of <= wn columns each
    return sum(-(-K // TK) * 2 * min(wn, Nn - b * wn) for b in range(min(nb, -(-Nn // wn))))
  return tiles(D, 2 * I, pf) + ((tiles(D, Nq, 1 << 30) + tiles(Ho, D, 1 << 30)) if pfqo else 0)

@dataclass(frozen=True)
class LayerBF16W:
  """nki_bf16.LayerBF16 with its weights streamed wide (FZ_NEURON_BF16_WN) and optionally prefetched (FZ_NEURON_BF16_PF /
  _PFQO): the same arguments, the same bits."""
  base: object   # nki_bf16.LayerBF16
  wn: int
  pf: int = 0
  pfqo: bool = False
  @property
  def name(self) -> str: return self.base.name + f"_w{self.wn}" + (f"p{self.pf}" if self.pf else "") + ("q" if self.pfqo else "")
  def args(self) -> tuple[str, ...]: return self.base.args()
  @property
  def heads(self) -> str: return self.base.heads
  def render(self) -> NKIProgram:
    return wide_weights(self.base.render(), self.wn, self.pf, self.pfqo, self.name[len(self.base.name):])

def wide_sbuf_bytes(wn: int) -> int:
  """SBUF bytes per partition the wide moving tiles add over the 512-column ones (two (128, wn) bf16 tiles in flight)."""
  return 2 * 2 * (wn - TN)

def layer_bf16w(sp, wn: int | None = None, pf: int | None = None, pfqo: bool | None = None):
  """NeuronKVCache.layer's hook (I68's LayerBF16 sp in, the kernel spec out): LayerBF16W when FZ_NEURON_BF16_WN / _PF / _PFQO
  (or the WN default) ask for it, else sp; the prefetch is clamped to the SBUF budget (nki_bf16.LAYER_SBUF).
  Only a whole untiled layer: the tiled layer (LayerSpec.tile, I71: its own wcols, a different block shape wide_weights does
  not match) and a TP half (LayerSpec.part, I72: not validated with wide tiles) come back unchanged. The default width (WN
  unset) steps down 4096 -> 2048 -> 1024 -> off until the layer's SBUF estimate plus the wider tiles fits; PSUM: a wn-column
  block holds wn / 512 <= 8 (M, 512) float32 tiles, i.e. at most the 8 banks of 2 KiB (validated at 4096 on trn1)."""
  from fusezap.codegen.nki_bf16 import LAYER_SBUF, layer_sbuf_bytes
  ls = sp.base
  if getattr(ls, "tile", False) or getattr(ls, "part", ""): return sp
  default = wn is None and os.environ.get("FZ_NEURON_BF16_WN", "").strip() == ""
  wn = bf16_wn() if wn is None else wn
  pf = bf16_pf() if pf is None else pf
  pfqo = bf16_pfqo() if pfqo is None else pfqo
  if default:
    while wn > TN and layer_sbuf_bytes(ls) + wide_sbuf_bytes(wn) > LAYER_SBUF: wn //= 2
  if wn == TN and not pf and not pfqo: return sp
  room = LAYER_SBUF - layer_sbuf_bytes(ls)
  if pfqo and prefetch_bytes(ls, wn, 0, True) > room: pfqo = False
  while pf and prefetch_bytes(ls, wn, pf, pfqo) > room: pf -= 1
  return LayerBF16W(sp, wn, pf, pfqo)

# ---------------------------------------------------------------- the lm_head: HeadSpec with bf16 weight tiles

def head_tiles_bf16(w_bits, W: int):
  """A torch-layout lm_head (V, K) as bf16 bit patterns (uint16) -> nki_kernels.head_tiles' flat layout, bf16 (uint16)."""
  import numpy as np
  w = np.asarray(w_bits, np.uint16)
  V, K = w.shape
  nkf, kt = divmod(K, TK)
  out = np.empty(V * K, np.uint16)
  for b0 in range(0, V, W):
    wb = min(W, V - b0)
    blk = w[b0:b0 + wb].T
    o = b0 * K
    if nkf: out[o:o + nkf * TK * wb] = blk[:nkf * TK].reshape(nkf, TK, wb).transpose(1, 0, 2).reshape(-1)
    if kt: out[o + nkf * TK * wb:o + K * wb] = blk[nkf * TK:].reshape(-1)
  return out

def head_bf16_program(prog: NKIProgram, name: str) -> NKIProgram:
  """render_head's float32 program (not ws) -> the weight tiles (wt, tt) bf16 and the matmuls' Aᵀ stationary a bf16 copy of
  `at` made once per row block (as MatmulBF16 casts it): the mechanical edits of nki_bf16, on HeadSpec's source."""
  lines = prog.src.split("\n")
  n_w = 0
  for i, ln in enumerate(lines):
    if re.match(r"^\s*(wt|tt) = nl\.ndarray\(.*dtype=nl\.float32, buffer=nl\.sbuf\)\s*$", ln):
      lines[i] = ln.replace("dtype=nl.float32", "dtype=nl.bfloat16"); n_w += 1
  if not n_w: raise NKIUnsupported(f"bf16 head {prog.name}: no weight tiles wt / tt")
  if any("pz = nl.ndarray" in ln for ln in lines): raise NKIUnsupported(f"bf16 head {prog.name}: the weight-stationary head has no bf16 form")
  k = next((i for i, ln in enumerate(lines) if re.match(r"^\s*for b0 in range\(", ln)), None)
  a = next((i for i, ln in enumerate(lines) if re.match(r"^\s*at = nl\.ndarray\((\(.*?\)), dtype=nl\.float32, buffer=nl\.sbuf\)\s*$", ln)), None)
  if k is None or a is None or a > k: raise NKIUnsupported(f"bf16 head {prog.name}: no Aᵀ tile before the block loop")
  shape = re.match(r"^\s*at = nl\.ndarray\((\(.*?\)), dtype", lines[a]).group(1)
  ind = re.match(r"^(\s*)", lines[k]).group(1)
  n_mm = 0
  for i, ln in enumerate(lines):
    if "nisa.nc_matmul(" in ln and "stationary=at[" in ln: lines[i] = ln.replace("stationary=at[", "stationary=at_bf["); n_mm += 1
  if not n_mm: raise NKIUnsupported(f"bf16 head {prog.name}: no matmul reads at")
  lines[k:k] = [f"{ind}at_bf = nl.ndarray({shape}, dtype=nl.bfloat16, buffer=nl.sbuf)", f"{ind}nisa.tensor_copy(dst=at_bf, src=at)"]
  src, n = re.subn(rf"\bdef {re.escape(prog.name)}\(", f"def {name}(", "\n".join(lines))
  if n != 1: raise NKIUnsupported(f"bf16 head {prog.name}: kernel definition not found")
  dts = dict(prog.buf_dtypes); dts[2] = dtypes.bfloat16
  return NKIProgram(name, src, prog.n_bufs, prog.outs, dts)

@dataclass(frozen=True)
class HeadBF16:
  """logits (M, V) float32 = h (M, K) float32 @ lm_headᵀ with the weight bf16 in head_tiles_bf16's layout (W columns per
  block): HeadSpec's kernel with nki_bf16's edits, the same bits as MatmulBF16 (`fz_headb<W>(data0=logits, data1=h, data2=tiles)`)."""
  M: int
  V: int
  K: int
  W: int = 2048
  @property
  def name(self) -> str: return f"fz_headb{self.W}_{self.M}_{self.V}_{self.K}"
  def render(self) -> NKIProgram:
    from fusezap.codegen.nki_kernels import HeadSpec
    return head_bf16_program(HeadSpec(self.M, self.V, self.K, self.W, 1, False).render(), self.name)
