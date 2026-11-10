"""I77: prefill kernels for NEURON (FZ_NEURON_PREFILL_KERNEL=v2): a prompt chunk's decoder layer as five kernels instead of
the ~12 member kernels, for chunks of 128 rows and more (the decode layer kernel, nki_kernels.LayerSpec, fits ~18 tokens at
Qwen2.5-7B's 7 query heads per kv head; a prefill of 128 rows ran as the members: render_matmul, the renderer's RoPE and
silu*mul kernels, NormSpec, KVWriteSpec, render_attention).

Where the members' time went (inf2, 7B TP=2 shard, 128 rows, I75 run: one fused NEFF of the 28 layers' members 81.6 ms
device; a separate-launch run put the MLP half of a layer at 1.84 ms and the attention half at 1.60 ms at the 1024-slot
pool, ~1.0 at a 128-slot prefix): every bf16 weight matmul streamed its weight in (128, 512) tiles (1 KiB runs per
partition, ~125 GB/s against ~207 GB/s for the decode NEFF's 4 KiB runs) and re-streamed it for every 128-row block of a
bigger chunk; gate_up wrote its (N, 2I) float32 product to HBM for a renderer silu*mul kernel that wrote (N, I) back for
down to re-read; attention ran (token, q head) rows: ~76 small DMAs per 128-row tile (q one token per DMA with D on the
partitions, lo / hi per token, the output per token), ~1,070 per layer, each tile recomputing the same key mask.

The kernels here compute the members' values with the members' instructions in the members' order per value (bitwise
equal on NKISHIM with FZ_NKISHIM_MM=f64, tests/test_nki_prefill.py), only moved: exact moves (TensorE transposes of finite
values, DMAs) replace HBM round trips, a weight tile feeds every row block of the chunk, rows are regrouped (attention).
  PFLinear  C (M, N) = A (M, K) @ W[:, c_off:c_off + N] (bf16 K-major, row stride ws): MatmulBF16's arithmetic (bf16(Aᵀ)
            chunks stationary, PSUM accumulation over the K chunks in order per 512-column output tile) with wide weight
            tiles (one DMA per K chunk of up to FZ_NEURON_PFK_WN columns, runs of 2 * wn bytes), each feeding every row
            block (weights read once per chunk whatever its rows); o_proj, and the lm_head (its vocabulary window per rank
            under TP: FZ_NEURON_PFK_HEAD=split).
  PFQKV     xo = x + d, h = rmsnorm(xo) (NormSpec's instructions), X = [rot(q) | rot(k) | v | 1] of h @ Wqkv + b (PFLinear's
            matmuls; bias and RoPE in the eviction of each 512-column tile: the renderer RoPE kernel's products and sum, the
            cat's + 0.0): the members NormSpec, MatmulBF16, the RoPE kernel, without the h, y round trips.
  PFAttn    keyrange attention (render_attention's per-row instructions) with rows = the chunk's tokens of ONE q head: q rows
            by one DMA and a transpose, the pool prefix's rows in SBUF once (kᵀ per kv head by transposes, v straight from
            them), each key block's mask once per row block (shared by every head), lo / hi once, the output one DMA per head.
  PFMLP     xo = x + o (+ bo), h = rmsnorm(xo), a = silu(h @ Wg) * (h @ Wu) per column block in SBUF (the renderer chain:
            render_layer's _silu_mul), aᵀ by transposes straight to the bf16 stationary tiles, dn = a @ Wd: the members
            NormSpec, MatmulBF16 gate_up, the silu*mul kernel and MatmulBF16 down, with no (N, 2I) / (N, I) HBM rows.
The KV write stays KVWriteSpec.

FZ_NEURON_PFK_EXP: exact (default since I77 run 1: the members' bits; ~49.5 ms per 128-row 7B TP=2 chunk against act's 42.6)
or act, which puts attention's softmax exp and silu's exp on ScalarE (one activation instead of the ~41
VectorE instructions of nki.exact_exp per element; ScalarE exp is ~1e-5 relative on inf2, N4 run 2): NOT the members'
bits, within the bf16 bar (the weights' bf16 rounding is ~1e-3). exact: nki.exact_exp on VectorE, bitwise equal to the
members. act's case: by scripts/nki_prefill_cost.py's model (7B shard, 128 rows, 128-slot prefix) the exact
exp keeps VectorE on the critical path, 1.54 ms per layer against 1.13 with act (the members 2.45); at 512 rows exact is
VectorE-bound (5.3 ms per layer, 1.2x cheaper per token than 128 rows) while act is TensorE-bound (2.8 ms: 2.2x cheaper
per token than 128 rows).
FZ_NEURON_PFK_EVICT: vector (default: inf2's NeuronCore-v2 rejects the ScalarE tensor_copy, I77 run 1) or act, which evicts PFMLP's gate / up PSUM tiles with ScalarE copies (float32 -> float32: exact
whatever the engine) so the next block's matmuls never wait behind VectorE's silu.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from fusezap.dtype import dtypes
from fusezap.codegen import nki
from fusezap.codegen.nki import NKIProgram, NKIUnsupported
from fusezap.codegen.nki_kernels import HEADER, TK, TM, TN, TS, FLT_MAX, _fl, _norm_rows, _silu_mul

# ---------------------------------------------------------------- switches

def _env_int(k: str, d: int) -> int:
  v = os.environ.get(k, "").strip()
  return int(v) if v else d

def pfk_exp() -> str:
  """FZ_NEURON_PFK_EXP: exact (default; nki.exact_exp on VectorE, the members' bits) or act (ScalarE exp, not bitwise)."""
  v = (os.environ.get("FZ_NEURON_PFK_EXP", "") or "exact").strip().lower()
  if v not in ("exact", "act"): raise ValueError(f"FZ_NEURON_PFK_EXP={v!r}: exact or act")
  return v

def pfk_evict() -> str:
  """FZ_NEURON_PFK_EVICT: vector (default) or act (PFMLP's gate / up PSUM evictions on ScalarE)."""
  v = (os.environ.get("FZ_NEURON_PFK_EVICT", "") or "vector").strip().lower()
  if v not in ("act", "vector"): raise ValueError(f"FZ_NEURON_PFK_EVICT={v!r}: act or vector")
  return v

def pfk_banks() -> int:
  """FZ_NEURON_PFK_BANKS: PSUM banks (2 KiB per partition each, 8 in all) a wide matmul block may hold at once (default 8:
  N8's 4096-column decode blocks held 8 on trn1); gate / up blocks, whose eviction also transposes, take at most 4."""
  b = _env_int("FZ_NEURON_PFK_BANKS", 8)
  if not 1 <= b <= 8: raise ValueError(f"FZ_NEURON_PFK_BANKS={b}: 1 .. 8")
  return b

def pfk_wn() -> int:
  """FZ_NEURON_PFK_WN: the most weight columns one DMA loads (a multiple of 512, default 4096: 8 KiB runs of bf16)."""
  w = _env_int("FZ_NEURON_PFK_WN", 8 * TN)
  if w % TN or not TN <= w <= 8 * TN: raise ValueError(f"FZ_NEURON_PFK_WN={w}: a multiple of {TN} in [{TN}, {8 * TN}]")
  return w

def block_cols(rows: int, banks: int, wn: int) -> int:
  """Columns per weight DMA for `rows` rows: every row block holds one PSUM bank per 512 columns of the block, so
  ceil(rows / 128) * cols / 512 <= banks (at least 512 columns), capped at wn."""
  mb = -(-rows // TM)
  return max(TN, min(wn, (banks // mb) * TN))

def _rblocks(N: int) -> list[tuple[int, int]]: return [(t0, min(TM, N - t0)) for t0 in range(0, N, TM)]

# ---------------------------------------------------------------- shared emitters

def _rows(a: str, t0: int, Mr: int, W: int, c0: int = 0, cw: int | None = None) -> str:
  """Rows t0 .. t0+Mr-1 (columns c0 .. c0+cw) of an (·, W) row-major HBM argument as a DMA operand."""
  return f"{a}.ap(pattern=[[{W}, {Mr}], [1, {W if cw is None else cw}]], offset={t0 * W + c0})"

def _bcast(a: str, M: int, W: int, c0: int = 0) -> str:
  """A vector of W elements (from c0) on M rows (partition step 0)."""
  return f"{a}.ap(pattern=[[0, {M}], [1, {W}]], offset={c0})"

def _sb(t: str, shape, dt: str = "float32", ind: str = "  ") -> str: return f"{ind}{t} = nl.ndarray({shape}, dtype=nl.{dt}, buffer=nl.sbuf)"

def _at_bf(L: list[str], src: str, Mr: int, K: int, dst: str, tag: str) -> None:
  """dst (min(K,128), ceil(K/128) * Mr) bfloat16 = bf16 of the Aᵀ chunks of the SBUF row tile src (Mr, K) float32: chunk i at
  dst[0:kt, i*Mr:(i+1)*Mr]. MatmulBF16's values: TensorE transposes (exact) into a float32 tile, then one rounding copy
  of the whole tile (render_matmul's `at` and nki_bf16's `at_bf`)."""
  nk = -(-K // TK)
  L += [_sb(f"{dst}f", (min(K, TK), nk * Mr)),
        f"  for c{tag} in range({nk}):",
        f"    kc{tag} = min({TK}, {K} - c{tag} * {TK})",
        f"    tp{tag} = nl.ndarray((kc{tag}, {Mr}), dtype=nl.float32, buffer=nl.psum)",
        f"    nisa.nc_transpose(dst=tp{tag}, data={src}[:, c{tag} * {TK}:c{tag} * {TK} + kc{tag}])",
        f"    nisa.tensor_copy(dst={dst}f[0:kc{tag}, c{tag} * {Mr}:c{tag} * {Mr} + {Mr}], src=tp{tag})",
        _sb(dst, (min(K, TK), nk * Mr), "bfloat16"), f"  nisa.tensor_copy(dst={dst}, src={dst}f)"]

def _mm(L: list[str], ats: list[tuple[str, int]], K: int, w: str, ws: int, c_off: int, Nn: int, wc: int, tag: str, evict) -> None:
  """Columns [0, Nn) of A @ w[:, c_off:] for every row block (ats: (bf16 Aᵀ tile, rows) per block, chunk i at
  [0:kt, i*rows:(i+1)*rows]): per block of wc columns, one weight DMA per K chunk ((kt, nw) bf16, runs of nw elements), one
  nc_matmul per row block and 512-column slice into its PSUM tile ps<tag>_<r>_<j>, accumulating over the K chunks in order
  (render_matmul's chain per output tile, so MatmulBF16's bits); then evict(r, psum tile, column, width) per tile."""
  nk = -(-K // TK)
  for c0 in range(0, Nn, wc):
    nw = min(wc, Nn - c0)
    subs = [(j, s0, min(TN, nw - s0)) for j, s0 in enumerate(range(0, nw, TN))]
    L.append(f"  # {tag}: columns {c0} .. {c0 + nw - 1} ({len(ats)} row blocks x {len(subs)} PSUM tiles), one weight DMA per K chunk")
    for r, (at, Mr) in enumerate(ats):
      L += [f"  ps{tag}_{r}_{j} = nl.ndarray(({Mr}, {sw}), dtype=nl.float32, buffer=nl.psum)" for j, s0, sw in subs]
    L += [f"  for i{tag} in range({nk}):",
          f"    kt{tag} = min({TK}, {K} - i{tag} * {TK})",
          f"    bt{tag} = nl.ndarray((kt{tag}, {nw}), dtype=nl.bfloat16, buffer=nl.sbuf)",
          f"    nisa.dma_copy(dst=bt{tag}[0:kt{tag}, :], src={w}.ap(pattern=[[{ws}, kt{tag}], [1, {nw}]], offset=i{tag} * {TK * ws} + {c_off + c0}))"]
    for r, (at, Mr) in enumerate(ats):
      L += [f"    nisa.nc_matmul(dst=ps{tag}_{r}_{j}, stationary={at}[0:kt{tag}, i{tag} * {Mr}:i{tag} * {Mr} + {Mr}], "
            f"moving=bt{tag}[0:kt{tag}, {s0}:{s0 + sw}], accumulate=(i{tag} > 0))" for j, s0, sw in subs]
    for r, (at, Mr) in enumerate(ats):
      for j, s0, sw in subs: L += evict(r, f"ps{tag}_{r}_{j}", c0 + s0, sw)

def _exp2(dsts: list[str], xs: list[str], shapes: list[str], ind: str, mode: str) -> list[str]:
  """dst = exp(x + ng) per (dst, x, shape): nki_kernels._exp (exact: VectorE, the members' bits) or (act) one ScalarE
  activation each with ng as its bias."""
  if mode == "act": return [f"{ind}nisa.activation(dst={d}, op=nl.exp, data={x}, bias=ng, scale=1.0)" for d, x in zip(dsts, xs)]
  from fusezap.codegen.nki_kernels import _exp
  if not nki.EXACT: raise NKIUnsupported("prefill kernels: FZ_NKI_FAST_EXP is the global switch; use FZ_NEURON_PFK_EXP=act")
  return _exp(dsts, xs, shapes, ind)

def _silu_act(T: str, g: str, u: str, dst: str, ind: str) -> list[str]:
  """dst = silu(g) * u with g * 1 / (1 + exp(-g)): the exp on ScalarE (FZ_NEURON_PFK_EXP=act; not the renderer's bits)."""
  return [f"{ind}se = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)", f"{ind}nisa.activation(dst=se, op=nl.exp, data={g}, scale=-1.0)",
          f"{ind}s1 = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)", f"{ind}nisa.tensor_scalar(dst=s1, data=se, op0=nl.add, operand0=1.0)",
          f"{ind}s2 = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)", f"{ind}nisa.reciprocal(dst=s2, data=s1)",
          f"{ind}s3 = nl.ndarray({T}, dtype=nl.float32, buffer=nl.sbuf)", f"{ind}nisa.tensor_tensor(dst=s3, data1={g}, data2=s2, op=nl.multiply)",
          f"{ind}nisa.tensor_tensor(dst={dst}, data1=s3, data2={u}, op=nl.multiply)"]

def _program(name: str, args: tuple[str, ...], L: list[str], outs: tuple[int, ...], bf16: tuple[str, ...] = (), ints: tuple[str, ...] = ()) -> NKIProgram:
  A = {nm: f"data{i}" for i, nm in enumerate(args)}
  src = "\n".join(L)
  for nm, d in sorted(A.items(), key=lambda x: -len(x[0])): src = src.replace(f"@{nm}@", d)
  body = [f"def {name}({', '.join(A.values())}):", src, f"  return {', '.join(f'data{i}' for i in outs)}"]
  dts = {i: dtypes.bfloat16 if nm in bf16 else dtypes.int32 if nm in ints else dtypes.float32 for i, nm in enumerate(args)}
  return NKIProgram(name, HEADER + "\n".join(body) + "\n", len(args), outs, dts)

# ---------------------------------------------------------------- PFLinear: o_proj, lm_head

@dataclass(frozen=True)
class PFLinear:
  """C (M, N) float32 = A (M, K) float32 @ W[:, c_off:c_off + N] (bf16, K-major, row stride ws; ws = N, c_off = 0 for a
  whole weight), MatmulBF16's bits with wide weight tiles feeding every row block (module docstring).
  `fz_pflin_M_N_K[_o<c_off>](data0=C, data1=A, data2=W)`. pad (the split lm_head): C is (M, ws) and the product lands in
  its columns [c_off, c_off + N), every other column 0.0: the zero-padded half a sum all-reduce gathers (no pad kernel)."""
  M: int
  N: int
  K: int
  ws: int = 0
  c_off: int = 0
  banks: int = 8
  wn: int = 8 * TN
  pad: bool = False
  @property
  def name(self) -> str:
    return f"fz_pflin_{self.M}_{self.N}_{self.K}" + (f"_w{self.ws}o{self.c_off}" if (self.ws or self.N) != self.N or self.c_off else "") + \
           f"_b{block_cols(self.M, self.banks, self.wn)}" + ("_pad" if self.pad else "")
  def args(self) -> tuple[str, ...]: return ("c", "a", "w")
  def render(self) -> NKIProgram:
    M, N, K, ws = self.M, self.N, self.K, self.ws or self.N
    if min(M, N, K) <= 0 or self.c_off < 0 or self.c_off + N > ws: raise NKIUnsupported(f"{self.name}: empty or outside the weight")
    wc = block_cols(M, self.banks, self.wn)
    cs, co = (ws, self.c_off) if self.pad else (N, 0)   # C's row stride, the product's first column in C
    L: list[str] = []
    if self.pad and ws > N:   # the columns outside the window: zeros, from one zero tile per row block
      for t0, Mr in _rblocks(M):
        zw = min(4096, max(self.c_off, ws - self.c_off - N))
        L += [_sb("zt", (Mr, zw)), "  nisa.memset(dst=zt, value=0.0)"]
        for z0, z1 in ((0, self.c_off), (self.c_off + N, ws)):
          for c in range(z0, z1, zw):
            L += [f"  nisa.dma_copy(dst={_rows('@c@', t0, Mr, cs, c, min(zw, z1 - c))}, src=zt[:, 0:{min(zw, z1 - c)}])"]
    ats = []
    for r, (t0, Mr) in enumerate(_rblocks(M)):
      L += [_sb(f"ar{r}", (Mr, K)), f"  nisa.dma_copy(dst=ar{r}, src={_rows('@a@', t0, Mr, K)})"]
      _at_bf(L, f"ar{r}", Mr, K, f"at{r}", f"t{r}")
      ats.append((f"at{r}", Mr))
    rb = _rblocks(M)
    def ev(r: int, ps: str, c: int, w: int) -> list[str]:
      t0, Mr = rb[r]
      return [_sb("ot", (Mr, w)), f"  nisa.tensor_copy(dst=ot, src={ps})", f"  nisa.dma_copy(dst={_rows('@c@', t0, Mr, cs, co + c, w)}, src=ot)"]
    _mm(L, ats, K, "@w@", ws, self.c_off, N, wc, "m", ev)
    return _program(self.name, self.args(), L, (0,), bf16=("w",))
  def sizes(self) -> list[int]: return [self.M * ((self.ws or self.N) if self.pad else self.N), self.M * self.K, self.K * (self.ws or self.N)]

# ---------------------------------------------------------------- PFQKV: add + norm + qkv + bias + RoPE

@dataclass(frozen=True)
class PFQKV:
  """xo = x + d (add), h = rmsnorm(xo) * ln1 (NormSpec), X = [rot(q) | rot(k) | v | 1] (N, (Hq + 2 Hkv + 1) hd) of
  y = h @ Wqkv (+ bqkv): the members NormSpec(add), MatmulBF16 and the RoPE kernel of NeuronKVCache.attend_rope, in one
  kernel. cos / sin: (N, hd) RoPE rows ([cos, cos], [-sin, sin]). Without add, xo is not written (x is the residual)."""
  N: int
  D: int
  Hq: int
  Hkv: int
  hd: int
  eps: float
  add: bool = True
  bias: bool = True
  banks: int = 8
  wn: int = 8 * TN
  fmax: int = 512
  @property
  def name(self) -> str: return f"fz_pfqkv{'_add' if self.add else ''}_{self.N}_{self.D}_{self.Hq}_{self.Hkv}_{self.hd}_b{block_cols(self.N, self.banks, self.wn)}"
  def args(self) -> tuple[str, ...]:
    return tuple((["xo"] if self.add else []) + ["xs", "x"] + (["d"] if self.add else []) + ["ln1", "wqkv"] + (["bqkv"] if self.bias else []) + ["cos", "sin"])
  def outs(self) -> tuple[int, ...]: return (0, 1) if self.add else (0,)
  def render(self) -> NKIProgram:
    N, D, Hq, Hkv, hd = self.N, self.D, self.Hq, self.Hkv, self.hd
    if min(N, D, Hq, Hkv, hd) <= 0 or Hq % Hkv or hd % 2 or TN % hd: raise NKIUnsupported(f"{self.name}: heads must tile 512 columns")
    Nq, QK, h2 = (Hq + 2 * Hkv) * hd, (Hq + Hkv) * hd, hd // 2
    Wx = Nq + hd
    wc = block_cols(N, self.banks, self.wn)
    rb = _rblocks(N)
    M0 = rb[0][1]
    L = ["  # 1. x1 = x + d, h = rmsnorm(x1) (NormSpec's instructions); the residual out; hᵀ (bf16) for the matmul",
         _sb("w1", (M0, D)), f"  nisa.dma_copy(dst=w1, src={_bcast('@ln1@', M0, D)})"]
    if self.bias: L += [_sb("bq", (M0, Nq)), f"  nisa.dma_copy(dst=bq, src={_bcast('@bqkv@', M0, Nq)})"]
    ats = []
    for r, (t0, Mr) in enumerate(rb):
      L += [_sb(f"x1_{r}", (Mr, D)), f"  nisa.dma_copy(dst=x1_{r}, src={_rows('@x@', t0, Mr, D)})"]
      if self.add:
        L += [_sb(f"d1_{r}", (Mr, D)), f"  nisa.dma_copy(dst=d1_{r}, src={_rows('@d@', t0, Mr, D)})",
              f"  nisa.tensor_tensor(dst=x1_{r}, data1=x1_{r}, data2=d1_{r}, op=nl.add)",
              f"  nisa.dma_copy(dst={_rows('@xo@', t0, Mr, D)}, src=x1_{r})"]
      L += _norm_rows("  ", f"x1_{r}", f"w1[0:{Mr}, :]", str(Mr), D, self.eps, self.fmax, f"_{r}")
      _at_bf(L, f"yt_{r}", Mr, D, f"at{r}", f"h{r}")
      ats.append((f"at{r}", Mr))
      L += [_sb(f"cs{r}", (Mr, hd)), f"  nisa.dma_copy(dst=cs{r}, src={_rows('@cos@', t0, Mr, hd)})",
            _sb(f"sn{r}", (Mr, hd)), f"  nisa.dma_copy(dst=sn{r}, src={_rows('@sin@', t0, Mr, hd)})",
            _sb(f"X{r}", (Mr, Wx))]
    L += ["  # 2. y = h @ Wqkv (+ b) per 512-column tile, 3. RoPE of the tile's q / k heads into X (y*cos + flip(y)*sin), v + 0.0"]
    def ev(r: int, ps: str, s0: int, nt: int) -> list[str]:
      Mr = rb[r][1]
      E = [_sb("yq", (Mr, nt)), f"  nisa.tensor_copy(dst=yq, src={ps})"]
      if self.bias: E += [f"  nisa.tensor_tensor(dst=yq, data1=yq, data2=bq[0:{Mr}, {s0}:{s0 + nt}], op=nl.add)"]
      for c in range(s0, s0 + nt, hd):
        k = c - s0
        if c < QK:
          E += [_sb("xa", (Mr, hd)), _sb("fl", (Mr, hd)),
                f"  nisa.tensor_tensor(dst=xa, data1=yq[:, {k}:{k + hd}], data2=cs{r}, op=nl.multiply)",
                f"  nisa.tensor_tensor(dst=fl[:, 0:{h2}], data1=yq[:, {k + h2}:{k + hd}], data2=sn{r}[:, 0:{h2}], op=nl.multiply)",
                f"  nisa.tensor_tensor(dst=fl[:, {h2}:{hd}], data1=yq[:, {k}:{k + h2}], data2=sn{r}[:, {h2}:{hd}], op=nl.multiply)",
                f"  nisa.tensor_tensor(dst=X{r}[:, {c}:{c + hd}], data1=xa, data2=fl, op=nl.add)"]
        else: E += [f"  nisa.tensor_scalar(dst=X{r}[:, {c}:{c + hd}], data=yq[:, {k}:{k + hd}], op0=nl.add, operand0=0.0)"]
      return E
    _mm(L, ats, D, "@wqkv@", Nq, 0, Nq, wc, "q", ev)
    for r, (t0, Mr) in enumerate(rb):   # the cat's + 0.0 over the rotated columns, the ones column, X out
      L += [f"  nisa.tensor_scalar(dst=X{r}[:, 0:{QK}], data=X{r}[:, 0:{QK}], op0=nl.add, operand0=0.0)",
            f"  nisa.memset(dst=X{r}[:, {Nq}:{Wx}], value=1.0)",
            f"  nisa.dma_copy(dst={_rows('@xs@', t0, Mr, Wx)}, src=X{r})"]
    return _program(self.name, self.args(), L, self.outs(), bf16=("wqkv",))
  def sizes(self) -> list[int]:
    N, D, hd = self.N, self.D, self.hd
    Nq = (self.Hq + 2 * self.Hkv) * hd
    sz = {"xo": N * D, "xs": N * (Nq + hd), "x": N * D, "d": N * D, "ln1": D, "wqkv": D * Nq, "bqkv": Nq, "cos": N * hd, "sin": N * hd}
    return [sz[a] for a in self.args()]

# ---------------------------------------------------------------- PFAttn: g-major prefill attention

@dataclass(frozen=True)
class PFAttn:
  """o (N, Hq*hd) = keyrange attention of X's q columns (N, Wx) over the pool prefix (P slots, rows [k | v] of 2C = 2 Hkv hd)
  with meta's lo (N), hi (N), kid (P) at base: NeuronKVCache.attend_rope's AttnSpec (rows = (token, q head), two row levels)
  regrouped as rows = the tokens of one q head (module docstring). `fz_pfattn_...(data0=o, data1=X, data2=pool, data3=meta)`."""
  N: int
  Hq: int
  Hkv: int
  hd: int
  P: int
  scale: float
  base: int
  exp: str = "exact"
  @property
  def name(self) -> str: return f"fz_pfattn_{self.N}_{self.Hq}_{self.Hkv}_{self.hd}_{self.P}" + ("_act" if self.exp == "act" else "")
  def args(self) -> tuple[str, ...]: return ("os", "xs", "pool", "meta")
  def render(self) -> NKIProgram:
    N, Hq, Hkv, hd, P = self.N, self.Hq, self.Hkv, self.hd, self.P
    if min(N, Hq, Hkv, hd, P) <= 0 or Hq % Hkv or hd > TK: raise NKIUnsupported(f"{self.name}: head_dim <= 128, Hq a multiple of Hkv")
    if self.exp not in ("exact", "act"): raise NKIUnsupported(f"{self.name}: exp {self.exp!r}")
    G, C = Hq // Hkv, Hkv * hd
    C2, Ho, Wx = 2 * C, Hq * hd, (Hq + 2 * Hkv + 1) * hd
    nP = -(-P // TM)
    base = self.base
    L = ["  # the pool prefix's rows in SBUF (pool tile j: slots j*128 ..), kᵀ of each kv head by transposes",
         _sb("pl", (min(P, TM), nP * C2))]
    for j in range(nP):
      p0, pt = j * TM, min(TM, P - j * TM)
      L += [f"  nisa.dma_copy(dst=pl[0:{pt}, {j * C2}:{(j + 1) * C2}], src=@pool@.ap(pattern=[[{C2}, {pt}], [1, {C2}]], offset={p0 * C2}))"]
    for h in range(Hkv):
      L += [_sb(f"kt{h}", (hd, P))]
      for j in range(nP):
        p0, pt = j * TM, min(TM, P - j * TM)
        L += [f"  kp = nl.ndarray(({hd}, {pt}), dtype=nl.float32, buffer=nl.psum)",
              f"  nisa.nc_transpose(dst=kp, data=pl[0:{pt}, {j * C2 + h * hd}:{j * C2 + (h + 1) * hd}])",
              f"  nisa.tensor_copy(dst=kt{h}[:, {p0}:{p0 + pt}], src=kp)"]
    blocks = [(b, s0, min(TS, P - s0)) for b, s0 in enumerate(range(0, P, TS))]
    for r, (t0, lt) in enumerate(_rblocks(N)):
      L += [f"  # rows {t0} .. {t0 + lt - 1}: lo / hi and each key block's mask once, for every head"]
      for nm, off in (("lo", base), ("hi", base + N)):
        L += [_sb(f"{nm}i", (lt, 1), "int32"), f"  nisa.dma_copy(dst={nm}i, src=@meta@.ap(pattern=[[1, {lt}], [1, 1]], offset={off + t0}))",
              _sb(f"{nm}f", (lt, 1)), f"  nisa.tensor_copy(dst={nm}f, src={nm}i)"]
      for b, s0, st in blocks:
        L += [_sb(f"mk{b}", (lt, st), "uint8"), _sb("kk", (lt, st), "int32"),
              f"  nisa.dma_copy(dst=kk, src=@meta@.ap(pattern=[[0, {lt}], [1, {st}]], offset={base + 2 * N + s0}))",
              _sb("ki", (lt, st)), "  nisa.tensor_copy(dst=ki, src=kk)",
              _sb("k1", (lt, st)), "  nisa.tensor_scalar(dst=k1, data=ki, op0=nl.greater_equal, operand0=lof)",
              _sb("k2", (lt, st)), "  nisa.tensor_scalar(dst=k2, data=ki, op0=nl.less_equal, operand0=hif)",
              f"  nisa.tensor_tensor(dst=mk{b}, data1=k1, data2=k2, op=nl.multiply)"]
      for hq in range(Hq):
        h = hq // G
        L += [f"  # q head {hq} (kv head {h})",
              _sb("qr", (lt, hd)), f"  nisa.dma_copy(dst=qr, src={_rows('@xs@', t0, lt, Wx, hq * hd, hd)})",
              f"  qp = nl.ndarray(({hd}, {lt}), dtype=nl.float32, buffer=nl.psum)", "  nisa.nc_transpose(dst=qp, data=qr)",
              _sb("qt", (hd, lt)), "  nisa.tensor_copy(dst=qt, src=qp)",
              _sb("mr", (lt, 1)), f"  nisa.memset(dst=mr, value={_fl(-FLT_MAX)})", _sb("lr", (lt, 1)), "  nisa.memset(dst=lr, value=0.0)",
              _sb("acc", (lt, hd)), "  nisa.memset(dst=acc, value=0.0)"]
        for b, s0, st in blocks:   # render_attention's key block, instruction for instruction
          L += [f"  ss = nl.ndarray(({lt}, {st}), dtype=nl.float32, buffer=nl.psum)",
                f"  nisa.nc_matmul(dst=ss, stationary=qt[0:{hd}, 0:{lt}], moving=kt{h}[:, {s0}:{s0 + st}], accumulate=False)",
                _sb("sc", (lt, st)), f"  nisa.tensor_scalar(dst=sc, data=ss, op0=nl.multiply, operand0={_fl(self.scale)})",
                _sb("sm", (lt, st)), "  nisa.memset(dst=sm, value=float('-inf'))",
                f"  nisa.tensor_copy_predicated(dst=sm, src=sc, predicate=mk{b})",
                _sb("bm", (lt, 1)), "  nisa.tensor_reduce(dst=bm, op=nl.maximum, data=sm, axis=1)",
                _sb("mn", (lt, 1)), "  nisa.tensor_tensor(dst=mn, data1=mr, data2=bm, op=nl.maximum)",
                _sb("ng", (lt, 1)), "  nisa.tensor_scalar(dst=ng, data=mn, op0=nl.multiply, operand0=-1.0)",
                _sb("al", (lt, 1)), _sb("p", (lt, st))]
          L += _exp2(["al", "p"], ["mr", "sm"], [f"({lt}, 1)", f"({lt}, {st})"], "  ", self.exp)
          L += [_sb("rs", (lt, 1)), "  nisa.tensor_reduce(dst=rs, op=nl.add, data=p, axis=1)",
                "  nisa.tensor_tensor(dst=lr, data1=lr, data2=al, op=nl.multiply)",
                "  nisa.tensor_tensor(dst=lr, data1=lr, data2=rs, op=nl.add)",
                "  nisa.tensor_scalar(dst=acc, data=acc, op0=nl.multiply, operand0=al)",
                "  nisa.tensor_copy(dst=mr, src=mn)"]
          for c0 in range(0, hd, TN):
            cw = min(TN, hd - c0)
            L += [f"  op = nl.ndarray(({lt}, {cw}), dtype=nl.float32, buffer=nl.psum)"]
            for jj, j in enumerate(range(0, st, TK)):
              ct, g = min(TK, st - j), (s0 + j) // TM
              L += [f"  pp = nl.ndarray(({ct}, {lt}), dtype=nl.float32, buffer=nl.psum)", f"  nisa.nc_transpose(dst=pp, data=p[:, {j}:{j + ct}])",
                    _sb("pt", (ct, lt)), "  nisa.tensor_copy(dst=pt, src=pp)",
                    f"  nisa.nc_matmul(dst=op, stationary=pt, moving=pl[0:{ct}, {g * C2 + C + h * hd + c0}:{g * C2 + C + h * hd + c0 + cw}], "
                    f"accumulate={jj > 0})"]
            L += [f"  nisa.tensor_tensor(dst=acc[:, {c0}:{c0 + cw}], data1=acc[:, {c0}:{c0 + cw}], data2=op, op=nl.add)"]
        L += [_sb("rl", (lt, 1)), f"  nisa.tensor_scalar(dst=lr, data=lr, op0=nl.maximum, operand0={nki.FLT_MIN_!r})",
              "  nisa.reciprocal(dst=rl, data=lr)", "  nisa.tensor_scalar(dst=acc, data=acc, op0=nl.multiply, operand0=rl)",
              f"  nisa.dma_copy(dst={_rows('@os@', t0, lt, Ho, hq * hd, hd)}, src=acc)"]
    return _program(self.name, self.args(), L, (0,), ints=("meta",))
  def sizes(self, meta: int | None = None) -> list[int]:
    Wx, C2 = (self.Hq + 2 * self.Hkv + 1) * self.hd, 2 * self.Hkv * self.hd
    return [self.N * self.Hq * self.hd, self.N * Wx, self.P * C2, meta or self.base + 2 * self.N + self.P]

# ---------------------------------------------------------------- PFMLP: add + norm + gate_up + silu*mul + down

@dataclass(frozen=True)
class PFMLP:
  """xo = x + o (+ bo), h = rmsnorm(xo) * ln2, dn = (silu(h @ Wg) * (h @ Wu)) @ Wd, Wgu (D, 2I) = [Wg | Wu] and Wd (I, D)
  bf16 K-major: the members NormSpec(add), MatmulBF16 gate_up, the renderer's silu*mul and MatmulBF16 down, in one kernel
  (module docstring). `fz_pfmlp_...(data0=xo, data1=dn, data2=x, data3=o[, bo], ln2, wgu, wd)`."""
  N: int
  D: int
  I: int
  eps: float
  o_bias: bool = False
  exp: str = "exact"
  banks: int = 8
  wn: int = 8 * TN
  fmax: int = 512
  evict: str = "act"
  @property
  def gu_cols(self) -> int: return block_cols(self.N, min(self.banks, 4), self.wn)
  @property
  def down_cols(self) -> int:
    """down's columns per weight DMA. I13: one row block whose down block would take every PSUM bank (D >= 4096 at 8 banks:
    Llama-3.1-8B) leaves a bank for the last silu block's transposes, still live when down starts (onchip_peak: 16.5 KiB)."""
    full = block_cols(self.N, self.banks, self.wn)
    return block_cols(self.N, self.banks - 1, self.wn) if self.N <= TM and min(full, self.D) >= self.banks * TN else full
  @property
  def name(self) -> str:
    dc, full = self.down_cols, block_cols(self.N, self.banks, self.wn)
    return f"fz_pfmlp_{self.N}_{self.D}_{self.I}_b{self.gu_cols}_{full}" + (f"_d{dc}" if dc != full else "") + ("_act" if self.exp == "act" else "") + \
           ("_ev" if self.evict != "act" else "")
  def args(self) -> tuple[str, ...]: return tuple(["xo", "dn", "x", "o"] + (["bo"] if self.o_bias else []) + ["ln2", "wgu", "wd"])
  def render(self) -> NKIProgram:
    N, D, I = self.N, self.D, self.I
    if min(N, D, I) <= 0 or self.exp not in ("exact", "act") or self.evict not in ("act", "vector"):
      raise NKIUnsupported(f"{self.name}: empty, exp {self.exp!r} or evict {self.evict!r}")
    rb = _rblocks(N)
    M0 = rb[0][1]
    nkI = -(-I // TK)
    L = ["  # 1. x2 = x + o (+ bo) (the residual out), h = rmsnorm(x2) (NormSpec's instructions), hᵀ (bf16)",
         _sb("w2", (M0, D)), f"  nisa.dma_copy(dst=w2, src={_bcast('@ln2@', M0, D)})"]
    if self.o_bias: L += [_sb("bo", (M0, D)), f"  nisa.dma_copy(dst=bo, src={_bcast('@bo@', M0, D)})"]
    ats, aTs = [], []
    for r, (t0, Mr) in enumerate(rb):
      L += [_sb(f"x2_{r}", (Mr, D)), f"  nisa.dma_copy(dst=x2_{r}, src={_rows('@x@', t0, Mr, D)})",
            _sb(f"o2_{r}", (Mr, D)), f"  nisa.dma_copy(dst=o2_{r}, src={_rows('@o@', t0, Mr, D)})"]
      if self.o_bias: L += [f"  nisa.tensor_tensor(dst=o2_{r}, data1=o2_{r}, data2=bo[0:{Mr}, :], op=nl.add)"]
      L += [f"  nisa.tensor_tensor(dst=x2_{r}, data1=x2_{r}, data2=o2_{r}, op=nl.add)",
            f"  nisa.dma_copy(dst={_rows('@xo@', t0, Mr, D)}, src=x2_{r})"]
      L += _norm_rows("  ", f"x2_{r}", f"w2[0:{Mr}, :]", str(Mr), D, self.eps, self.fmax, f"_{r}")
      _at_bf(L, f"yt_{r}", Mr, D, f"at{r}", f"h{r}")
      ats.append((f"at{r}", Mr))
      L += [_sb(f"aT{r}", (min(I, TK), nkI * Mr), "bfloat16")]
      aTs.append((f"aT{r}", Mr))
    L += ["  # 2. per column block: gate, then up (wide weight tiles, every row block), silu(gate) * up on the rows (the renderer's",
          "  # chain), aᵀ by transposes straight into the bf16 stationary tiles of down"]
    wc = self.gu_cols
    # FZ_NEURON_PFK_EVICT=act: gate / up leave PSUM by ScalarE copies (float32 -> float32, exact on any engine), so
    # the next block's matmuls never queue behind VectorE's silu chain; vector (default): VectorE (the compiler's usual pick)
    evict = ", engine=nisa.scalar_engine" if self.evict == "act" else ""
    def silu(c0: int, nw: int, q: int) -> None:   # block c0's silu(gate) * up and its aᵀ chunks (tiles of parity q)
      for r, (t0, Mr) in enumerate(rb):
        for s0 in range(0, nw, TN):
          sw = min(TN, nw - s0)
          T, g, u = f"({Mr}, {sw})", f"g{r}_{q}[:, {s0}:{s0 + sw}]", f"u{r}_{q}[:, {s0}:{s0 + sw}]"
          L.extend([_sb("a", T)] + (_silu_act(T, g, u, "a", "  ") if self.exp == "act" else _silu_mul(T, g, u, "a", "  ", False)))
          for c in range(0, sw, TK):
            kc, ci = min(TK, sw - c), (c0 + s0 + c) // TK
            L.extend([f"  ta = nl.ndarray(({kc}, {Mr}), dtype=nl.float32, buffer=nl.psum)", f"  nisa.nc_transpose(dst=ta, data=a[:, {c}:{c + kc}])",
                      f"  nisa.tensor_copy(dst=aT{r}[0:{kc}, {ci * Mr}:{(ci + 1) * Mr}], src=ta)"])
    # software pipelined by one block in program order: block b's matmuls are issued before block b-1's silu and transposes,
    # so TensorE (in order) multiplies the next block while VectorE runs the silu chain (gate / up tiles alternate by parity)
    pend = None
    for b, c0 in enumerate(range(0, I, wc)):
      nw, q = min(wc, I - c0), b % 2
      for part, off in (("g", 0), ("u", I)):
        for r, (t0, Mr) in enumerate(rb): L += [_sb(f"{part}{r}_{q}", (Mr, nw))]
        def ev(r: int, ps: str, c: int, w: int, part=part, q=q) -> list[str]:   # c: the column inside this block
          return [f"  nisa.tensor_copy(dst={part}{r}_{q}[:, {c}:{c + w}], src={ps}{evict})"]
        _mm(L, ats, D, "@wgu@", 2 * I, off + c0, nw, nw, part, ev)
      if pend is not None: silu(*pend)
      pend = (c0, nw, q)
    silu(*pend)
    L += ["  # 3. down"]
    def evd(r: int, ps: str, c: int, w: int) -> list[str]:
      t0, Mr = rb[r]
      return [_sb("dt", (Mr, w)), f"  nisa.tensor_copy(dst=dt, src={ps})", f"  nisa.dma_copy(dst={_rows('@dn@', t0, Mr, D, c, w)}, src=dt)"]
    _mm(L, aTs, I, "@wd@", D, 0, D, self.down_cols, "dn", evd)
    return _program(self.name, self.args(), L, (0, 1), bf16=("wgu", "wd"))
  def sizes(self) -> list[int]:
    N, D, I = self.N, self.D, self.I
    sz = {"xo": N * D, "dn": N * D, "x": N * D, "o": N * D, "bo": D, "ln2": D, "wgu": D * 2 * I, "wd": I * D}
    return [sz[a] for a in self.args()]

# ---------------------------------------------------------------- on-chip memory (tests, scripts/nki_prefill_cost.py)

def onchip_peak(prog: NKIProgram, sizes: list[int]) -> dict:
  """The largest sum of live SBUF and of live PSUM tiles (bytes per partition) along the kernel's NKISHIM trace (record only,
  nothing computed): a tile is live from its allocation to its last use. A lower bound of what neuronx-cc allocates (it may
  keep more buffers in flight); SBUF has 192 KiB per partition, PSUM 16 KiB (8 banks of 2 KiB)."""
  import numpy as np
  from fusezap.device import nkishim as S
  from fusezap.dtype import to_np
  live: dict = {}
  k = [0]
  orig = S.nl.ndarray   # whatever is installed (scripts/nki_layer_cost.py records allocations through its own)
  def nd(shape, dtype, buffer=S.SBUF, name="", address=None):
    t = orig(shape, dtype, buffer, name, address)
    if buffer in (S.SBUF, S.PSUM): live[id(t.storage)] = [k[0], k[0], int(np.prod(t.shape[1:])) * dtype.itemsize, buffer]
    return t
  def rec(nm):
    def g(*a, **kw):
      k[0] += 1
      for v in list(a) + list(kw.values()):
        if isinstance(v, S.NkiTensor) and id(v.storage) in live: live[id(v.storage)][1] = k[0]
    return g
  names = ("dma_copy", "memset", "tensor_copy", "tensor_tensor", "tensor_scalar", "activation", "reciprocal", "tensor_reduce",
           "tensor_copy_predicated", "nc_matmul", "nc_transpose", "iota", "affine_select")
  saved = {nm: getattr(S.nisa, nm) for nm in names}
  for nm in names: setattr(S.nisa, nm, rec(nm))
  S.nl.ndarray = nd
  try:
    arrays = [np.zeros(max(1, s), to_np(prog.buf_dtypes.get(i)) if i in prog.buf_dtypes else np.uint8) for i, s in enumerate(sizes)]
    S.execute(prog, arrays)
  finally:
    for nm, f in saved.items(): setattr(S.nisa, nm, f)
    S.nl.ndarray = orig
  out = {}
  for buf, key in ((S.SBUF, "sbuf"), (S.PSUM, "psum")):
    d = np.zeros(k[0] + 2)
    for a, b, nb, bb in live.values():
      if bb == buf: d[a] += nb; d[b + 1] -= nb
    out[key] = int(np.cumsum(d).max()) if len(d) else 0
  out["instructions"] = k[0]
  return out

# ---------------------------------------------------------------- numpy references (tests, the device self-test)

def _bf(x):
  from fusezap.codegen.nki_bf16 import bf16_round
  return bf16_round(x)

def ref_rmsnorm(x, w, eps: float):
  import numpy as np
  x = np.asarray(x, np.float64)
  return w * (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))

def ref_linear(a, w_bits):
  """bf16(a) @ W in float64 (W: bf16 bit patterns (K, N))."""
  from fusezap.codegen.nki_bf16 import matmul_ref
  return matmul_ref(a, w_bits)

def ref_qkv(sp: PFQKV, x, d, ln1, w_bits, b, cos, sin):
  """(xo, X) of PFQKV in float64 (bf16 matmul operands)."""
  import numpy as np
  xo = np.asarray(x, np.float64) + (np.asarray(d, np.float64) if d is not None else 0.0)
  h = ref_rmsnorm(xo.astype(np.float32), np.asarray(ln1, np.float64), sp.eps)
  y = ref_linear(h.astype(np.float32), w_bits) + (np.asarray(b, np.float64) if b is not None else 0.0)
  N, hd, h2 = sp.N, sp.hd, sp.hd // 2
  Hqk, Nq = sp.Hq + sp.Hkv, (sp.Hq + 2 * sp.Hkv) * hd
  X = np.ones((N, Nq + hd))
  yr = y[:, :Hqk * hd].reshape(N, Hqk, hd)
  fl = np.concatenate([yr[..., h2:], yr[..., :h2]], -1)
  X[:, :Hqk * hd] = (yr * np.asarray(cos, np.float64)[:, None, :] + fl * np.asarray(sin, np.float64)[:, None, :]).reshape(N, -1)
  X[:, Hqk * hd:Nq] = y[:, Hqk * hd:]
  return xo, X

def ref_attn(sp: PFAttn, X, pool, meta):
  """PFAttn in float64: keyrange attention of X's q columns over the pool prefix."""
  import numpy as np
  N, Hq, Hkv, hd, P, G = sp.N, sp.Hq, sp.Hkv, sp.hd, sp.P, sp.Hq // sp.Hkv
  C = Hkv * hd
  lo, hi = meta[sp.base:sp.base + N], meta[sp.base + N:sp.base + 2 * N]
  kid = meta[sp.base + 2 * N:sp.base + 2 * N + P]
  pl = np.asarray(pool, np.float64).reshape(-1, 2 * C)[:P]
  out = np.zeros((N, Hq * hd))
  for t in range(N):
    vis = (kid >= lo[t]) & (kid <= hi[t])
    if not vis.any(): continue
    for hq in range(Hq):
      h = hq // G
      q = np.asarray(X[t, hq * hd:(hq + 1) * hd], np.float64)
      s = pl[:, h * hd:(h + 1) * hd] @ q * sp.scale
      s = np.where(vis, s, -np.inf)
      e = np.exp(s - s.max())
      out[t, hq * hd:(hq + 1) * hd] = (e / e.sum()) @ pl[:, C + h * hd:C + (h + 1) * hd]
  return out

def ref_mlp(sp: PFMLP, x, o, bo, ln2, wgu_bits, wd_bits):
  """(xo, dn) of PFMLP in float64 (bf16 matmul operands, a rounded to bf16 for down as MatmulBF16 rounds it)."""
  import numpy as np
  xo = np.asarray(x, np.float64) + np.asarray(o, np.float64) + (np.asarray(bo, np.float64) if bo is not None else 0.0)
  h = ref_rmsnorm(xo.astype(np.float32), np.asarray(ln2, np.float64), sp.eps)
  gu = ref_linear(h.astype(np.float32), wgu_bits)
  g, u = gu[:, :sp.I], gu[:, sp.I:]
  a = g / (1 + np.exp(-g)) * u
  return xo, ref_linear(a.astype(np.float32), wd_bits)
