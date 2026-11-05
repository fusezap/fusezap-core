"""I68: bf16 on NEURON (FZ_NEURON_DTYPE=bf16, or `--dtype bfloat16` with `--device NEURON`).

What is bf16 here: the linear weights (qkv, o_proj, gate_up, down, lm_head) live in HBM as bf16, K-major (half the bytes
of the float32 path, which is what a decode step mostly reads), and every weight matmul runs on TensorE with bf16
operands: the weight's moving tiles straight from HBM, the activation's stationary Aᵀ chunks rounded to bf16 (round to
nearest even, one tensor_copy per chunk block) from the float32 tile the float32 kernel builds. TensorE multiplies bf16
x bf16 exactly and accumulates in float32 PSUM (the only accumulator NeuronCore-v2 has), so a matmul is
sum(bf16(a) * bf16(w)) in float32, the same contract as the CUDA bf16 GEMM (cuBLAS: bf16 operands, f32 accumulation).
Everything else stays float32, as in the float32 path: the residual stream, RMSNorm, RoPE, the softmax (and its exact exp),
silu, the logits and the argmax. That is more float32 than the CUDA bf16 path (whose residual stream, norm outputs and
activations are bf16 between kernels): its rounding points are only the operands of the weight matmuls.
Attention: by default q·kᵀ and p·v stay float32 on a float32 KV pool (the KV write and attention kernels unchanged); at
Qwen2.5-0.5B / 7B decode they read ~0.5-2 MB per layer at a 512-slot prefix against 30-470 MB of weights, so bf16 there
buys little. `FZ_NEURON_BF16_ATTN=1` (variant "bf16a") rounds q, k (after RoPE), p and v to bf16 for those two matmuls
too (the KV pool itself stays float32, so the KV write stays an exact one-hot scatter).

How: the bf16 kernels are the float32 kernels' sources with three mechanical edits (`bf16_weight_matmuls`), so they track
every change to render_layer / render_matmul instead of forking them:
  1. a weight's moving tile `bt<tag> = nl.ndarray(..., dtype=nl.float32 ...)` (the tile a `dma_copy(dst=bt<tag>..., src=<a
     weight argument>...)` fills) becomes bfloat16;
  2. before the loop `for n0<tag> in range(` that consumes it, the stationary tile `<at>` gets a bfloat16 copy `<at>_bf`
     (declared with <at>'s shape) and that loop's matmuls read `<at>_bf` (a stationary tile reloaded inside the loop gets
     its copy right before its matmul);
  3. the weight arguments' dtypes become bfloat16 (NEFF I/O and the NKISHIM / NKISIM arrays).
Each edit asserts that it found its pattern, so a renderer change that breaks one fails loudly (NKIUnsupported), never
silently in float32. Specs: MatmulBF16 (C (M, N) f32 = A (M, K) f32 @ W (K, N) bf16, K-major; Llama.linear on NEURON
with a bf16 weight: prefill, select rows, lm_head) and LayerBF16 (a LayerSpec with bf16 weights; NeuronKVCache.layer
picks it when the layer's weights are bf16). The fused kernels (FZ_NEURON_FUSE) take their members' sources and dtypes,
so they need nothing here.
"""
from __future__ import annotations
import os, re
from dataclasses import dataclass
from fusezap.dtype import dtypes
from fusezap.codegen.nki import NKIProgram, NKIUnsupported

def neuron_bf16_env() -> bool:
  """FZ_NEURON_DTYPE=bf16 (bfloat16): Llama on NEURON loads its linear weights as bf16 whatever dtype the caller asked for."""
  return os.environ.get("FZ_NEURON_DTYPE", "").strip().lower() in ("bf16", "bfloat16")

def bf16_attn() -> bool:
  """FZ_NEURON_BF16_ATTN=1: LayerBF16 also rounds attention's matmul operands (q, k, p, v) to bf16 (default 0: float32)."""
  return os.environ.get("FZ_NEURON_BF16_ATTN", "0") not in ("", "0")

_MM = re.compile(r"nisa\.nc_matmul\(dst=ps(\w*?)(?:_\d+)?, stationary=(\w+)(\[[^\]]*\])?, moving=bt(\w*?)(?:\[[^\]]*\])?[,)]")   # N8: ps<tag>_<j> and slices of bt<tag> (wide weight tiles)

def _decl_shape(lines: list[str], name: str, before: int) -> tuple[int, str, str]:
  """(line index, indent, shape text) of the last float32 SBUF declaration of `name` before line `before`."""
  pat = re.compile(rf"^(\s*){re.escape(name)} = nl\.ndarray\((\(.*?\)), dtype=nl\.float32, buffer=nl\.sbuf\)\s*$")
  for i in range(before - 1, -1, -1):
    if (m := pat.match(lines[i])): return i, m.group(1), m.group(2)
  raise NKIUnsupported(f"bf16: no float32 declaration of {name!r} before line {before}")

def bf16_weight_matmuls(prog: NKIProgram, weights: tuple[int, ...], suffix: str = "_bf16") -> NKIProgram:
  """The float32 program with its weight matmuls in bf16 (see the module docstring): weights = the argument indices that
  hold bf16 K-major weights. Returns a new NKIProgram named <name><suffix>."""
  lines = prog.src.split("\n")
  wnames = {f"data{i}" for i in weights}
  # 1. moving tiles filled from a weight argument
  tags: dict[str, list[int]] = {}   # matmul tag -> lines of its weight DMAs (N8's tiled MLP fills btu twice: gate, then up)
  for i, ln in enumerate(lines):
    if (m := re.search(r"nisa\.dma_copy\(dst=bt(\w*)\[[^\]]*\], src=(data\d+)\.ap\(", ln)) and m.group(2) in wnames: tags.setdefault(m.group(1), []).append(i)
  if not tags: raise NKIUnsupported(f"bf16 {prog.name}: no weight moving tile loaded from {sorted(wnames)}")
  for i, ln in enumerate(lines):   # a weight loaded into any other tile (a weight-stationary form) would stay float32: refuse
    if (m := re.search(r"nisa\.dma_copy\(dst=(\w+)[\[,].*src=(data\d+)\.ap\(", ln)) and m.group(2) in wnames and not m.group(1).startswith("bt"):
      raise NKIUnsupported(f"bf16 {prog.name}: weight {m.group(2)} loaded into {m.group(1)} (line {i}), not a moving tile bt<tag>")
  for tag, js in tags.items():
    for j in js:
      k, ind, shape = _decl_shape(lines, f"bt{tag}", j)
      lines[k] = f"{ind}bt{tag} = nl.ndarray({shape}, dtype=nl.bfloat16, buffer=nl.sbuf)"
  # 2. the stationary tiles of those matmuls: one bf16 copy before the loop that consumes them
  inserts: list[tuple[int, list[str]]] = []
  done: set[tuple[str, str]] = set()
  for i, ln in enumerate(lines):
    m = _MM.search(ln)
    if not m or m.group(4) not in tags: continue
    tag, at = m.group(4), m.group(2)
    if m.group(1) != tag: raise NKIUnsupported(f"bf16 {prog.name}: matmul into ps{m.group(1)} reads bt{tag}")
    loop = next((k for k in range(i - 1, -1, -1) if re.match(rf"^\s*for n0{re.escape(tag)} in range\(", lines[k])), None)
    if loop is None: raise NKIUnsupported(f"bf16 {prog.name}: no `for n0{tag}` loop around line {i}")
    d, ind, shape = _decl_shape(lines, at, i)
    if d > loop:   # the stationary tile is (re)declared inside the loop: cast it right before this matmul
      at_ind = re.match(r"^(\s*)", ln).group(1)
      inserts.append((i, [f"{at_ind}{at}_bf = nl.ndarray({shape}, dtype=nl.bfloat16, buffer=nl.sbuf)", f"{at_ind}nisa.tensor_copy(dst={at}_bf, src={at})"]))
    elif (tag, at) not in done:
      loop_ind = re.match(r"^(\s*)", lines[loop]).group(1)
      inserts.append((loop, [f"{loop_ind}{at}_bf = nl.ndarray({shape}, dtype=nl.bfloat16, buffer=nl.sbuf)", f"{loop_ind}nisa.tensor_copy(dst={at}_bf, src={at})"]))
    done.add((tag, at))
    lines[i] = ln.replace(f"stationary={at}", f"stationary={at}_bf", 1)
  if not done: raise NKIUnsupported(f"bf16 {prog.name}: no matmul reads the weight tiles")
  for at_line, new in sorted(inserts, key=lambda x: -x[0]): lines[at_line:at_line] = new
  # 3. the name and the weight arguments' dtypes
  src = "\n".join(lines)
  name = prog.name + suffix
  src, n = re.subn(rf"\bdef {re.escape(prog.name)}\(", f"def {name}(", src)
  if n != 1: raise NKIUnsupported(f"bf16 {prog.name}: kernel definition not found")
  dts = dict(prog.buf_dtypes)
  for i in weights: dts[i] = dtypes.bfloat16
  return NKIProgram(name, src, prog.n_bufs, prog.outs, dts)

def bf16_attention(prog: NKIProgram) -> NKIProgram:
  """LayerBF16 with FZ_NEURON_BF16_ATTN: attention's q·kᵀ and p·v with bf16 operands (q, k after RoPE, p, v rounded to
  nearest even; the scores and the p·v sums still accumulate in float32 PSUM, the softmax stays float32). The pool rows
  pl stay float32 (the KV write and kᵀ's transposes unchanged); v reads a bf16 copy of them, made once after the KV write."""
  lines = prog.src.split("\n")
  out: list[str] = []
  att = next((i for i, ln in enumerate(lines) if ln.strip().startswith("# 5. attention")), None)
  if att is None: raise NKIUnsupported(f"bf16a {prog.name}: no attention section")
  _, ind, shape = _decl_shape(lines, "pl", att)
  qk = re.compile(r"^(\s*)nisa\.nc_matmul\(dst=ss, stationary=(qt\d+), moving=(kt\d+), accumulate=False\)\s*$")
  pv = re.compile(r"^(\s*)nisa\.nc_matmul\(dst=op, stationary=pT\[(.*?)\], moving=pl\[(.*?)\], accumulate=(\w+)\)\s*$")
  nqk = npv = 0
  pt_cast = False
  for i, ln in enumerate(lines):
    if i == att: out += [f"{ind}pl_bf = nl.ndarray({shape}, dtype=nl.bfloat16, buffer=nl.sbuf)", f"{ind}nisa.tensor_copy(dst=pl_bf, src=pl)"]
    if ln.strip().startswith("# keys ") or re.match(r"^\s*pT = nl\.ndarray\(", ln): pt_cast = False   # pT is recomputed
    if (m := qk.match(ln)):
      s = m.group(1)
      q, k = m.group(2), m.group(3)
      _, _, qs = _decl_shape(lines, q, i)
      _, _, ks = _decl_shape(lines, k, i)
      out += [f"{s}{q}_bf = nl.ndarray({qs}, dtype=nl.bfloat16, buffer=nl.sbuf)", f"{s}nisa.tensor_copy(dst={q}_bf, src={q})",
              f"{s}{k}_bf = nl.ndarray({ks}, dtype=nl.bfloat16, buffer=nl.sbuf)", f"{s}nisa.tensor_copy(dst={k}_bf, src={k})",
              f"{s}nisa.nc_matmul(dst=ss, stationary={q}_bf, moving={k}_bf, accumulate=False)"]
      nqk += 1
      continue
    if (m := pv.match(ln)):
      s = m.group(1)
      if not pt_cast:
        _, _, ps = _decl_shape(lines, "pT", i)
        out += [f"{s}pT_bf = nl.ndarray({ps}, dtype=nl.bfloat16, buffer=nl.sbuf)", f"{s}nisa.tensor_copy(dst=pT_bf, src=pT)"]
        pt_cast = True
      out.append(f"{s}nisa.nc_matmul(dst=op, stationary=pT_bf[{m.group(2)}], moving=pl_bf[{m.group(3)}], accumulate={m.group(4)})")
      npv += 1
      continue
    out.append(ln)
  if not nqk or not npv: raise NKIUnsupported(f"bf16a {prog.name}: attention matmuls not found ({nqk} q·kᵀ, {npv} p·v)")
  name = prog.name.replace("_bf16", "_bf16a")
  src = "\n".join(out).replace(f"def {prog.name}(", f"def {name}(")
  return NKIProgram(name, src, prog.n_bufs, prog.outs, dict(prog.buf_dtypes))

# I69: LayerSpec keeps a step's rows whole in SBUF (x, d, h, the qkv rows, X, gate_up, ...: ~22 rows of D-sized float32
# values per partition, plus the pool rows). Qwen2.5-0.5B needs ~100 KiB per partition; Qwen2.5-7B (D 3584, I 18944: the
# gate_up row alone is 148 KiB) ~360 KiB, more than a partition's 192 KiB. A layer above the budget runs as the members
# (v1), which tile every operand, instead of failing in neuronx-cc.
# N8 (I71): the tiled layer (LayerSpec.tile, FZ_NEURON_LAYER_TILE=auto) keeps four (N, D) rows and no (N, 2I) or (N, Nq) one:
# Qwen2.5-7B at 4 rows needs ~166 KiB by this estimate at a 1024-slot prefix (~134 KiB at 512).
LAYER_SBUF = int(os.environ.get("FZ_NEURON_LAYER_SBUF", "") or 176 * 1024)   # of 192 KiB

def layer_sbuf_bytes(sp) -> int:
  """A conservative estimate of LayerSpec's SBUF bytes per partition (the row tiles it declares, all live at once)."""
  D, I, Hq, Hkv, hd, P = sp.D, sp.I, sp.Hq, sp.Hkv, sp.hd, sp.P
  Nq, QK = (Hq + 2 * Hkv) * hd, (Hq + Hkv) * hd
  rows = 7 * D + 3 * Nq + 2 * QK + (Nq + hd) + 2 * I + 2 * hd
  early = P if getattr(sp, "early", False) else 0   # N7 step 4 early: every key block's uint8 mask (mk<b>) issued at the top, all live
  # (its int32 / float32 temporaries die once each mask is made: one block's worth, inside the slack). pf (weight-stationary
  # gate tiles prefetched; off, not bitwise on trn1) is not counted: those tiles stand in for the MLP's own rotating tiles.
  if getattr(sp, "tile", False):   # N8 tiled layer: four (N, D) row tiles (x1, ra, rh, w1 reused), X, one 512-column qkv tile, RoPE's
    # (N, hd) tiles, one gate / up block and its (128, 4N) transposes and silu temporaries, aT (the down projection's Aᵀ)
    N, I128, wc = sp.N, -(-I // 128), getattr(sp, "wcols", 512) // 128
    rows = 4 * D + (Nq + hd) + 512 + 4 * hd + 2 * 512 + 16 * wc * N + I128 * N + (wc * 128 - 512)   # + two wider bf16 weight tiles in flight
  return 4 * rows + 4 * -(-P // 128) * 4 * Hkv * hd + early + 4 * 4096   # + the pool rows + slack for the small tiles

def layer_fits(sp) -> bool: return layer_sbuf_bytes(sp) <= LAYER_SBUF

# ---------------------------------------------------------------- specs (NeuronCall payloads)

@dataclass(frozen=True)
class MatmulBF16:
  """C (M, N) float32 = A (M, K) float32 @ W (K, N) bf16 (K-major, row-major A and C): render_matmul's kernel with its
  weight matmuls in bf16. Llama.linear on NEURON for a bf16 weight (`fz_matmul_1_M_N_K_bf16(data0=C, data1=A, data2=W)`)."""
  M: int
  N: int
  K: int
  @property
  def name(self) -> str: return f"fz_matmul_1_{self.M}_{self.N}_{self.K}_bf16"
  def render(self) -> NKIProgram:
    from fusezap.codegen.nki_kernels import render_matmul
    M, N, K = self.M, self.N, self.K
    return bf16_weight_matmuls(render_matmul((1, M, N, K, 0, K, 1, 0, N, 1, 0, N, 1, 0, 0)), (2,))

@dataclass(frozen=True)
class LayerBF16:
  """A LayerSpec decoder layer whose four weights (wqkv, wo, wgu, wd) are bf16 K-major; attn: FZ_NEURON_BF16_ATTN."""
  base: object   # nki_kernels.LayerSpec
  attn: bool = False
  @property
  def name(self) -> str: return self.base.name + ("_bf16a" if self.attn else "_bf16")
  def args(self) -> tuple[str, ...]: return self.base.args()
  @property
  def heads(self) -> str: return self.base.heads
  def render(self) -> NKIProgram:
    if getattr(self.base, "mm", "mv") != "mv":   # weight-stationary (FZ_NEURON_LAYER_MM=ws; off, not bitwise on trn1): no bf16 form
      raise NKIUnsupported(f"{self.name}: bf16 needs FZ_NEURON_LAYER_MM=mv (the weight tiles moving)")
    names = self.base.args()
    # N8 TP: a half (LayerSpec.part) has two of the four weights; the mlp half has no attention
    prog = bf16_weight_matmuls(self.base.render(), tuple(names.index(w) for w in ("wqkv", "wo", "wgu", "wd") if w in names))
    return bf16_attention(prog) if self.attn and getattr(self.base, "part", "") != "mlp" else prog

# ---------------------------------------------------------------- numpy references (tests)

def bf16_round(x):
  """float32 -> the nearest bf16 value (ties to even), as float32: what a tensor_copy into a bfloat16 tile stores."""
  import numpy as np
  from fusezap.serve.weights import bf16_to_f32, f32_to_bf16
  return bf16_to_f32(f32_to_bf16(np.asarray(x, np.float32)))

def matmul_ref(a, w_bf16_bits):
  """MatmulBF16's contract in float64: bf16(A) @ W (W given as bf16 bit patterns (K, N)) -> float64 (M, N)."""
  import numpy as np
  from fusezap.serve.weights import bf16_to_f32
  return bf16_round(a).astype(np.float64) @ bf16_to_f32(w_bf16_bits).astype(np.float64)
