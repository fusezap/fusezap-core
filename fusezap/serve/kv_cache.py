"""Paged KV cache (SPEC §11.3): fixed-size blocks, a block allocator, per-sequence block tables.

  cache = PagedKVCache(n_layers, n_kv, head_dim, num_blocks=512, block_size=16, dtype=..., device="METAL", max_seq_len=2048)
  cache.ensure(seq_id, n_tokens)      # grow seq_id's block table to hold n_tokens positions (MemoryError if the pool is full)
  cache.slot(seq_id, pos)             # flat slot (block * block_size + offset) that holds position pos
  cache.free(seq_id)                  # return its blocks to the allocator

Storage: per layer one (2, num_blocks, n_kv, block_size, head_dim) tensor (k then v). A sequence's position p lives in
block table[p // block_size] at offset p % block_size, so a sequence grows one block at a time and blocks of finished or
preempted sequences are reused immediately.

Two ways a model step uses it (both through the KVCache interface of fusezap/serve/models/llama.py):
  - The ragged (continuous batching) path, `Llama.forward_tokens`: a step is N tokens from any number of sequences, each
    with its own position. `attend(layer, qkv)` scatters the step's k/v rows into their slots (in place) and runs paged
    attention: every query token attends to keys 0..pos of its own sequence, read through the block table. The per-step
    int32 metadata (token -> block-table row, position, slot; logits rows; block tables) is one input tensor, so the
    jitted step's input shapes never change and nothing in it is symbolic.
  - The P4-MODEL interface, `update(layer, k, v, start, T)`, for `Llama.prefill/decode` with B rows that share positions:
    `set_rows(seq_ids)` maps batch row b to a sequence; update scatters the rows by block table and gathers each row's
    blocks back into a position-ordered (B, Hkv, max_blocks * block_size, hd) view.

Kernels: on METAL these are MSL kernels launched as kind="custom" ScheduleItems whose payload is a `ServeCall`. Inside a
`fz.jit` capture they are appended to the recording, so a replay launches them in order with the other kernels. On CUDA
the same kernels run as CUDA C++ (NVRTC), and paged attention is P3-ATTN's split-K kernel on this layout
(fusezap/device/cuda_kernels/attention.py). Other devices run a numpy reference of the same kernels (slow; for tests). The scatter writes the pool buffers in place (there
is no in-place store op in the IR; see docs/status/P4-SERVE.md).
"""
from __future__ import annotations
import heapq, math, struct
from dataclasses import dataclass
from typing import Any, Callable, Sequence
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import DType, dtypes, to_np
from fusezap.device import Device, Buffer
from fusezap.ir import ScheduleItem
from fusezap.lazy import LazyBuffer
from fusezap.shape.symbolic import Node, Var, sym_infer
from fusezap.serve.models.llama import KVCache

# ---------------------------------------------------------------- block allocator

class BlockAllocator:
  """Free list of block ids 0..num_blocks-1 (LIFO, so recently freed, cache-warm blocks are reused first).
  lowest_first: the lowest free id first (a heap), so the blocks in use stay packed at the bottom of the pool whatever the
  order of frees: NEURON's steps run at the smallest pool prefix that holds every block of their sequences
  (NeuronKVCache.meta_ctx), and after interleaved frees LIFO handed a lone request blocks above 512 slots."""
  def __init__(self, num_blocks: int, lowest_first: bool = False):
    assert num_blocks > 0
    self.num_blocks, self.lowest_first = num_blocks, lowest_first
    self._free = list(range(num_blocks)) if lowest_first else list(range(num_blocks - 1, -1, -1))   # a sorted list is a heap
    self._used: set[int] = set()
  @property
  def num_free(self) -> int: return len(self._free)
  def allocate(self, n: int = 1) -> list[int]:
    if n > len(self._free): raise MemoryError(f"need {n} KV blocks, {len(self._free)} free")
    out = [heapq.heappop(self._free) for _ in range(n)] if self.lowest_first else [self._free.pop() for _ in range(n)]
    self._used.update(out)
    return out
  def free(self, blocks: Sequence[int]) -> None:
    for b in reversed(list(blocks)):
      assert b in self._used, f"block {b} is not allocated"
      self._used.discard(b)
      if self.lowest_first: heapq.heappush(self._free, b)
      else: self._free.append(b)

# ---------------------------------------------------------------- kernels

_MSL_T = {dtypes.float32: "float", dtypes.float16: "half", dtypes.bfloat16: "bfloat"}
_EW = 256   # threads per threadgroup for the elementwise kernels

def _f32_bits(x: float) -> int: return struct.unpack("<i", struct.pack("<f", x))[0]
def _bits_f32(i: int) -> float: return struct.unpack("<f", struct.pack("<i", i))[0]

def _args(bufs: list[str], params: Sequence[str]) -> str:
  a = bufs + [f"constant int& {p} [[buffer({len(bufs) + i})]]" for i, p in enumerate(params)]
  return ", ".join(a + ["uint3 gid [[threadgroup_position_in_grid]]", "uint tid [[thread_index_in_threadgroup]]",
                        "uint sg [[simdgroup_index_in_threadgroup]]", "uint lane [[thread_index_in_simdgroup]]"])

_HDR = "#include <metal_stdlib>\nusing namespace metal;\n"

WRITE_PARAMS = ("n", "H", "hd", "bs", "NB", "xs", "xk", "xv", "moff")
def msl_kv_write(name: str, T: str) -> str:
  """P[0|1, blk, h, off, d] = X[t, xk|xv + h*hd + d] for slot = M[moff + t] >= 0 (blk = slot / bs, off = slot % bs)."""
  return _HDR + f"""kernel void {name}({_args([f"device {T}* P [[buffer(0)]]", f"device const {T}* X [[buffer(1)]]", "device const int* M [[buffer(2)]]"], WRITE_PARAMS)}) {{
  const int i = gid.x * {_EW} + tid;
  if (i >= n * H * hd) return;
  const int d = i % hd, h = (i / hd) % H, t = i / (hd * H), slot = M[moff + t];
  if (slot < 0) return;
  const int dst = ((slot / bs * H + h) * bs + slot % bs) * hd + d, src = t * xs + h * hd + d;
  P[dst] = X[src + xk];
  P[NB * H * bs * hd + dst] = X[src + xv];
}}"""

WRITE_ROWS_PARAMS = ("B", "H", "T", "hd", "bs", "NB", "MB", "start")
def msl_kv_write_rows(name: str, T: str) -> str:
  """KV (2, B, H, T, hd) contiguous -> the pool at positions start..start+T-1 of row b's sequence (block table BT[b])."""
  return _HDR + f"""kernel void {name}({_args([f"device {T}* P [[buffer(0)]]", f"device const {T}* KV [[buffer(1)]]", "device const int* BT [[buffer(2)]]"], WRITE_ROWS_PARAMS)}) {{
  const int i = gid.x * {_EW} + tid, per = B * H * T * hd;
  if (i >= per) return;
  const int d = i % hd, t = (i / hd) % T, h = (i / (hd * T)) % H, b = i / (hd * T * H), pos = start + t, j = pos / bs;
  if (j >= MB) return;
  const int blk = BT[b * MB + j];
  if (blk < 0) return;
  const int dst = ((blk * H + h) * bs + pos % bs) * hd + d;
  P[dst] = KV[i];
  P[NB * H * bs * hd + dst] = KV[per + i];
}}"""

GATHER_PARAMS = ("B", "H", "MB", "bs", "hd", "NB")
def msl_kv_gather(name: str, T: str) -> str:
  """O (2, B, H, MB*bs, hd): row j of batch row b = position j of its sequence (zeros where no block is allocated)."""
  return _HDR + f"""kernel void {name}({_args([f"device {T}* O [[buffer(0)]]", f"device const {T}* P [[buffer(1)]]", "device const int* BT [[buffer(2)]]"], GATHER_PARAMS)}) {{
  const int S = MB * bs, i = gid.x * {_EW} + tid;
  if (i >= 2 * B * H * S * hd) return;
  const int d = i % hd, j = (i / hd) % S, h = (i / (hd * S)) % H, b = (i / (hd * S * H)) % B, kv = i / (hd * S * H * B);
  const int blk = BT[b * MB + j / bs];
  O[i] = blk < 0 ? ({T})0 : P[(((kv * NB + blk) * H + h) * bs + j % bs) * hd + d];
}}"""

ROWS_PARAMS = ("R", "D", "moff")
def msl_rows(name: str, T: str) -> str:
  """O[r, :] = X[M[moff + r], :] (zeros for index < 0): an exact row gather."""
  return _HDR + f"""kernel void {name}({_args([f"device {T}* O [[buffer(0)]]", f"device const {T}* X [[buffer(1)]]", "device const int* M [[buffer(2)]]"], ROWS_PARAMS)}) {{
  const int i = gid.x * {_EW} + tid;
  if (i >= R * D) return;
  const int r = i / D, src = M[moff + r];
  O[i] = src < 0 ? ({T})0 : X[src * D + i % D];
}}"""

ATTN_PARAMS = ("N", "Hkv", "bs", "NB", "MB", "xs", "mrow", "mpos", "mbt", "scale_bits")
def msl_paged_attn(name: str, T: str, hd: int, G: int) -> str:
  """Paged attention for N query tokens: token t (block-table row M[mrow+t], position M[mpos+t]; row < 0 = padding, output
  0) attends to keys 0..pos of its sequence. One threadgroup per (token, kv head) with G simdgroups, one per query head
  of the group. Per chunk of 32 keys each lane scores one key (full head-dim dot product in float), the chunk updates a
  running max and sum (online softmax), and each lane accumulates hd/32 output columns in float.
  X holds the step's rotated q (row t, heads 0..Hq-1, row stride xs); P is the layer's pool (2, NB, Hkv, bs, hd)."""
  DC = -(-hd // 32)
  return _HDR + f"""kernel void {name}({_args([f"device {T}* O [[buffer(0)]]", f"device const {T}* X [[buffer(1)]]", f"device const {T}* P [[buffer(2)]]", "device const int* M [[buffer(3)]]"], ATTN_PARAMS)}) {{
  const int t = gid.x / Hkv, kh = gid.x % Hkv, h = kh * {G} + (int)sg;
  device {T}* o = O + (t * Hkv * {G} + h) * {hd};
  const int row = M[mrow + t];
  if (row < 0) {{ for (int d = lane; d < {hd}; d += 32) o[d] = ({T})0; return; }}
  const int pos = M[mpos + t];
  device const int* bt = M + mbt + row * MB;
  const float scale = as_type<float>(scale_bits), log2e = 1.4426950408889634f;
  float q[{hd}];
  device const {T}* qp = X + t * xs + h * {hd};
  for (int d = 0; d < {hd}; d++) q[d] = (float)qp[d];
  float acc[{DC}];
  for (int c = 0; c < {DC}; c++) acc[c] = 0.0f;
  float m = -INFINITY, den = 0.0f;
  const int vo = NB * Hkv * bs * {hd};
  for (int j0 = 0; j0 <= pos; j0 += 32) {{
    const int j = j0 + (int)lane;
    float s = -INFINITY;
    int base = 0;
    if (j <= pos) {{
      base = ((bt[j / bs] * Hkv + kh) * bs + j % bs) * {hd};
      float dot = 0.0f;
      for (int d = 0; d < {hd}; d++) dot += q[d] * (float)P[base + d];
      s = dot * scale;
    }}
    const float mn = max(m, simd_max(s));
    const float p = j <= pos ? exp2((s - mn) * log2e) : 0.0f, alpha = exp2((m - mn) * log2e);
    den = den * alpha + simd_sum(p);
    for (int c = 0; c < {DC}; c++) acc[c] *= alpha;
    const int nk = min(32, pos - j0 + 1);
    for (int k = 0; k < nk; k++) {{
      const float pk = simd_shuffle(p, (ushort)k);
      const int bk = simd_shuffle(base, (ushort)k) + vo;
      for (int c = 0; c < {DC}; c++) {{ const int d = (int)lane + 32 * c; if (d < {hd}) acc[c] += pk * (float)P[bk + d]; }}
    }}
    m = mn;
  }}
  for (int c = 0; c < {DC}; c++) {{ const int d = (int)lane + 32 * c; if (d < {hd}) o[d] = ({T})(acc[c] / den); }}
}}"""

# CUDA C++ versions (NVRTC; the same params and index math as the MSL kernels). They only move elements, so 16-bit types
# travel as their bit patterns (exact for f16 and bf16).

_CU_T = {dtypes.float32: "float", dtypes.float16: "unsigned short", dtypes.bfloat16: "unsigned short"}

def _cu(name: str, bufs: list[str], params: Sequence[str], body: str) -> str:
  a = ", ".join(bufs + [f"const int {p}" for p in params])
  return f"""extern "C" __global__ void __launch_bounds__({_EW}) {name}({a}) {{
  const int i = blockIdx.x * {_EW} + threadIdx.x;
{body}
}}"""

def cu_kv_write(name: str, T: str) -> str:
  return _cu(name, [f"{T}* __restrict__ P", f"const {T}* __restrict__ X", "const int* __restrict__ M"], WRITE_PARAMS, f"""  if (i >= n * H * hd) return;
  const int d = i % hd, h = (i / hd) % H, t = i / (hd * H), slot = M[moff + t];
  if (slot < 0) return;
  const int dst = ((slot / bs * H + h) * bs + slot % bs) * hd + d, src = t * xs + h * hd + d;
  P[dst] = X[src + xk];
  P[NB * H * bs * hd + dst] = X[src + xv];""")

def cu_kv_write_rows(name: str, T: str) -> str:
  return _cu(name, [f"{T}* __restrict__ P", f"const {T}* __restrict__ KV", "const int* __restrict__ BT"], WRITE_ROWS_PARAMS, f"""  const int per = B * H * T * hd;
  if (i >= per) return;
  const int d = i % hd, t = (i / hd) % T, h = (i / (hd * T)) % H, b = i / (hd * T * H), pos = start + t, j = pos / bs;
  if (j >= MB) return;
  const int blk = BT[b * MB + j];
  if (blk < 0) return;
  const int dst = ((blk * H + h) * bs + pos % bs) * hd + d;
  P[dst] = KV[i];
  P[NB * H * bs * hd + dst] = KV[per + i];""")

def cu_kv_gather(name: str, T: str) -> str:
  return _cu(name, [f"{T}* __restrict__ O", f"const {T}* __restrict__ P", "const int* __restrict__ BT"], GATHER_PARAMS, f"""  const int S = MB * bs;
  if (i >= 2 * B * H * S * hd) return;
  const int d = i % hd, j = (i / hd) % S, h = (i / (hd * S)) % H, b = (i / (hd * S * H)) % B, kv = i / (hd * S * H * B);
  const int blk = BT[b * MB + j / bs];
  O[i] = blk < 0 ? ({T})0 : P[(((kv * NB + blk) * H + h) * bs + j % bs) * hd + d];""")

def cu_rows(name: str, T: str) -> str:
  return _cu(name, [f"{T}* __restrict__ O", f"const {T}* __restrict__ X", "const int* __restrict__ M"], ROWS_PARAMS, f"""  if (i >= R * D) return;
  const int r = i / D, src = M[moff + r];
  O[i] = src < 0 ? ({T})0 : X[src * D + i % D];""")

# numpy references (CPU/NPY devices and tests); arrays are flat, outputs first, `p` the int params

def _np_kv_write(a: list[np.ndarray], p: dict) -> None:
  P, X, M = a
  n, H, hd, bs, NB = p["n"], p["H"], p["hd"], p["bs"], p["NB"]
  P4 = P.reshape(2, NB, H, bs, hd)
  for t in range(n):
    if (slot := int(M[p["moff"] + t])) < 0: continue
    r = X[t * p["xs"]:]
    P4[0, slot // bs, :, slot % bs] = r[p["xk"]:p["xk"] + H * hd].reshape(H, hd)
    P4[1, slot // bs, :, slot % bs] = r[p["xv"]:p["xv"] + H * hd].reshape(H, hd)

def _np_kv_write_rows(a: list[np.ndarray], p: dict) -> None:
  P, KV, BT = a
  B, H, T, hd, bs, NB, MB, st = (p[k] for k in WRITE_ROWS_PARAMS)
  P4, KV5, BT2 = P.reshape(2, NB, H, bs, hd), KV.reshape(2, B, H, T, hd), BT.reshape(B, MB)
  for b in range(B):
    for t in range(T):
      pos = st + t
      if pos // bs >= MB or (blk := int(BT2[b, pos // bs])) < 0: continue
      P4[:, blk, :, pos % bs] = KV5[:, b, :, t]

def _np_kv_gather(a: list[np.ndarray], p: dict) -> None:
  O, P, BT = a
  B, H, MB, bs, hd, NB = (p[k] for k in GATHER_PARAMS)
  O6, P5, BT2 = O.reshape(2, B, H, MB, bs, hd), P.reshape(2, NB, H, bs, hd), BT.reshape(B, MB)
  for b in range(B):
    for j in range(MB):
      O6[:, b, :, j] = 0 if (blk := int(BT2[b, j])) < 0 else P5[:, blk]

def _np_rows(a: list[np.ndarray], p: dict) -> None:
  O, X, M = a
  R, D = p["R"], p["D"]
  O2, X2 = O.reshape(R, D), X.reshape(-1, D)
  for r in range(R): O2[r] = 0 if (i := int(M[p["moff"] + r])) < 0 else X2[i]

def _np_paged_attn(a: list[np.ndarray], p: dict, hd: int, G: int) -> None:
  O, X, P, M = a
  N, Hkv, bs, NB, MB = p["N"], p["Hkv"], p["bs"], p["NB"], p["MB"]
  Hq, sc = Hkv * G, _bits_f32(p["scale_bits"])
  P5, O3 = P.reshape(2, NB, Hkv, bs, hd).astype(np.float32), O.reshape(N, Hq, hd)
  for t in range(N):
    if (row := int(M[p["mrow"] + t])) < 0:
      O3[t] = 0
      continue
    pos, bt = int(M[p["mpos"] + t]), M[p["mbt"] + row * MB:]
    js = np.arange(pos + 1)
    blk = bt[js // bs].astype(np.int64)
    K, V = P5[0, blk, :, js % bs], P5[1, blk, :, js % bs]          # (S, Hkv, hd)
    q = X[t * p["xs"]:t * p["xs"] + Hq * hd].reshape(Hkv, G, hd).astype(np.float32)
    s = np.einsum("kgd,skd->kgs", q, K) * np.float32(sc)
    e = np.exp(s - s.max(-1, keepdims=True))
    O3[t] = (np.einsum("kgs,skd->kgd", e / e.sum(-1, keepdims=True), V)).reshape(Hq, hd)

@dataclass(frozen=True, eq=False)
class ServeCall:
  """Payload (`ast[0]`) of a kind="custom" ScheduleItem for one of this module's kernels. params may hold Nodes (e.g. a
  symbolic start position); they are evaluated with the item's var_vals at launch, so a jit replay passes new values."""
  name: str
  render: Callable[[str], str]           # MSL type name -> source defining `name`
  names: tuple[str, ...]                 # param names (np references read them by name)
  params: tuple[Any, ...]
  global_size: tuple[int, int, int]
  local_size: tuple[int, int, int]
  ref: Callable[[list[np.ndarray], dict], None]
  cuda: Callable[[str], str] | None = None   # CUDA C type name -> CUDA C++ source defining `name` (None: numpy reference)
  @property
  def kernel(self) -> "ServeCall": return self   # CustomCall-like: si.ast[0].kernel.name
  def launch(self, dev: Any, bufs: Sequence[Buffer], var_vals: dict) -> float | None: return _launch(dev, self, bufs, var_vals)
  def pretty(self, indent: int = 0) -> str: return f"{'  ' * indent}SERVE {self.name} {self.params}"

def _launch(dev: Any, call: ServeCall, bufs: Sequence[Buffer], var_vals: dict) -> float | None:
  vals = tuple(int(sym_infer(x, var_vals)) if isinstance(x, Node) else int(x) for x in call.params)
  if dev.name.split(":")[0].upper() == "METAL":
    T = _MSL_T[bufs[0].dtype]
    progs = dev.__dict__.setdefault("_fz_serve_progs", {})
    if (prg := progs.get((call.name, T))) is None:
      progs[(call.name, T)] = prg = dev.runtime(call.name, dev.compiler.compile_cached(call.render(T)))
    return prg(*[b.ensure_allocated().handle for b in bufs], global_size=call.global_size, local_size=call.local_size, vals=vals)
  if call.cuda is not None and dev.name.split(":")[0].upper() == "CUDA":
    T = _CU_T[bufs[0].dtype]
    progs = dev.__dict__.setdefault("_fz_serve_progs", {})
    if (prg := progs.get((call.name, T))) is None:
      progs[(call.name, T)] = prg = dev.runtime(call.name, dev.compiler.compile_cached(call.cuda(T)))
    if bufs[0].size == 0: return None
    return prg(*[b.ensure_allocated().handle for b in bufs], global_size=call.global_size, local_size=call.local_size, vals=vals)
  assert bufs[0].dtype != dtypes.bfloat16, "the numpy reference kernels do not take bfloat16"
  arrs = [b.numpy() for b in bufs]
  call.ref(arrs, dict(zip(call.names, vals)))
  bufs[0].copyin(memoryview(np.ascontiguousarray(arrs[0])).cast("B"))
  return None

def run_call(call: ServeCall, bufs: Sequence[Buffer], n_out: int = 1) -> None:
  """Execute `call` now and, if a fz.jit capture is recording, append it to the recording (so replays launch it)."""
  from fusezap import jit as J
  vv = {v: val for v, val in J.current_var_vals().items() if any(isinstance(x, Node) and v in x.vars() for x in call.params)} \
       if any(isinstance(x, Node) for x in call.params) else {}
  J.run_item(ScheduleItem((call,), tuple(bufs), vv, "custom", n_out))   # type: ignore[arg-type]

def _ew(n: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]: return (max(1, -(-n // _EW)), 1, 1), (_EW, 1, 1)
def _buf(t: Tensor) -> Buffer: return t._realized_contiguous().realized
def _wrap(buf: Buffer, shape: tuple[int, ...]) -> Tensor: return Tensor(LazyBuffer.from_buffer(buf, shape))

def kv_write(pool: Buffer, x: Buffer, meta: Buffer, n: int, H: int, hd: int, bs: int, NB: int, xs: int, xk: int, xv: int, moff: int) -> None:
  pr = (n, H, hd, bs, NB, xs, xk, xv, moff)
  run_call(ServeCall("fz_kv_write", lambda T: msl_kv_write("fz_kv_write", T), WRITE_PARAMS, pr, *_ew(n * H * hd), _np_kv_write,
                     lambda T: cu_kv_write("fz_kv_write", T)), (pool, x, meta))

def paged_attention(x: Buffer, pool: Buffer, meta: Buffer, N: int, Hkv: int, G: int, hd: int, bs: int, NB: int, MB: int,
                    xs: int, mrow: int, mpos: int, mbt: int, scale: float, prefill: bool = False, np_n: int = 0, mix: int = 0,
                    mix_exact: bool = True) -> Buffer:
  if x.device.split(":")[0].upper() == "CUDA":   # P3-ATTN's paged kernels read this layout and metadata directly
    from fusezap.device.cuda_kernels import attention as cuda_attn
    return cuda_attn.paged_attention(x, pool, meta, N, Hkv, G, hd, bs, NB, MB, xs, mrow, mpos, mbt, scale, prefill, np_n, mix, mix_exact)
  # mix (LLMEngine mixed forwards; off CUDA only under FZ_MIX_ANY for tests): this kernel treats prompt and decode rows alike
  out = Buffer(x.device, N * Hkv * G * hd, x.dtype)
  nm = f"fz_paged_attn_{hd}_{G}"
  pr = (N, Hkv, bs, NB, MB, xs, mrow, mpos, mbt, _f32_bits(scale))
  run_call(ServeCall(nm, lambda T: msl_paged_attn(nm, T, hd, G), ATTN_PARAMS, pr, (N * Hkv, 1, 1), (32 * G, 1, 1),
                     lambda a, p: _np_paged_attn(a, p, hd, G)), (out, x, pool, meta))
  return out

def gather_rows(x: Tensor, meta: Buffer, R: int, moff: int) -> Tensor:
  """x (N, D) -> (R, D) with row r = x[meta[moff + r]] (zeros for -1). Exact (a copy)."""
  D = x.shape[-1]
  xb = _buf(x)
  out = Buffer(xb.device, R * D, xb.dtype)
  run_call(ServeCall("fz_rows", lambda T: msl_rows("fz_rows", T), ROWS_PARAMS, (R, D, moff), *_ew(R * D), _np_rows,
                     lambda T: cu_rows("fz_rows", T)), (out, xb, meta))
  return _wrap(out, (R, D))

def lib_linear(x: Tensor, w: Tensor, b: Tensor | None = None) -> Tensor:
  """x (M, K) @ w (N, K)^T (+ b) with METAL's simdgroup GEMM (P2-MTL), recorded like the other custom items so a jit
  replays it. Used for the wide decode projections, where the composed kernels are slow at a few dozen rows."""
  from fusezap.schedule import CustomCall
  (M, K), N = x.shape, w.shape[0]
  xb, wb = _buf(x), _buf(w)
  dev = Device[xb.device]
  out = Buffer(xb.device, M * N, xb.dtype)
  si = ScheduleItem((CustomCall(dev.custom_kernels["matmul"], (1, M, N, K, 0, K, 1, 0, 1, K, 0, N, 1, 0, 0)),), (out, xb, wb), {}, "custom", 1)  # type: ignore[arg-type]
  from fusezap import jit as J
  J.run_item(si)
  y = _wrap(out, (M, N))
  return y + b if b is not None else y

# ---------------------------------------------------------------- the cache

@dataclass(frozen=True)
class KVDest:
  """A layer's paged pool (2, NB, Hkv, bs, hd) and the step metadata whose int32 at moff + row is the row's slot (< 0:
  padding): what a kernel needs to store the step's k / v rows the way kv_write does (PagedKVCache.kv_dest)."""
  pool: Buffer
  meta: Buffer
  moff: int
  Hkv: int
  hd: int
  bs: int
  NB: int

@dataclass
class StepLayout:
  """Offsets of one ragged step's int32 metadata (one flat input tensor): per token its block-table row (-1 = padding),
  position and slot; R logits rows (token index or -1); R block tables of max_blocks entries (-1 = none)."""
  N: int
  R: int
  MB: int
  prefill: bool = False   # the rows are prompt chunks (CUDA: tensor-core paged prefill attention)
  np_n: int = 0           # CUDA split-K attention: size its partitions for this many rows (0: N), so forwards of several
                          # row counts sum each token's keys in one order
  mix: int = 0            # CUDA mixed forward: rows < mix are prompt chunks, rows >= mix decode rows (0: not mixed)
  mix_exact: bool = True  # the decode rows must carry the decode forward's bits (exact mode); throughput mode: they need not
  @property
  def row(self) -> int: return 0
  @property
  def pos(self) -> int: return self.N
  @property
  def slot(self) -> int: return 2 * self.N
  @property
  def sel(self) -> int: return 3 * self.N
  @property
  def bt(self) -> int: return 3 * self.N + self.R
  @property
  def size(self) -> int: return 3 * self.N + self.R + self.R * self.MB

class PagedKVCache(KVCache):
  def __init__(self, n_layers: int, n_kv: int, head_dim: int, num_blocks: int, block_size: int, dtype: DType, device: str | None = None,
               max_seq_len: int = 2048, _shared: "PagedKVCache | None" = None, _state: list[Tensor] | None = None,
               _inputs: list[Tensor] | None = None):
    self.n_layers, self.n_kv, self.hd, self.num_blocks, self.bs, self.dtype = n_layers, n_kv, head_dim, num_blocks, block_size, dtype
    self.device = Device.canonicalize(device)
    self.max_seq_len, self.max_blocks = max_seq_len, -(-max_seq_len // block_size)
    assert 2 * num_blocks * n_kv * block_size * head_dim < 2 ** 31, "pool too large for int32 indexing"
    if _shared is None:
      self.allocator, self.tables = BlockAllocator(num_blocks), {}
      self.pools = [Tensor.zeros(2, num_blocks, n_kv, block_size, head_dim, dtype=dtype, device=self.device).contiguous().realize()
                    for _ in range(n_layers)]
      self._inputs: list[Tensor] = []
      self._rows: tuple[int, ...] = ()
    else:   # a bound view (inside a traced step): same bookkeeping, the step's state/input tensors
      self.allocator, self.tables, self._rows = _shared.allocator, _shared.tables, _shared._rows
      self.pools, self._inputs = list(_state or _shared.pools), list(_inputs if _inputs is not None else _shared._inputs)
    self.layout: StepLayout | None = _shared.layout if _shared is not None else None

  @staticmethod
  def for_model(model: Any, num_blocks: int, block_size: int = 16, max_seq_len: int | None = None) -> "PagedKVCache":
    c = model.cfg
    return PagedKVCache(c.n_layers, c.n_kv_heads, c.head_dim, num_blocks, block_size, model.dtype, model.device,
                        max_seq_len or model.max_len)

  # ---- bookkeeping (host)
  @property
  def num_free_blocks(self) -> int: return self.allocator.num_free
  def num_blocks_for(self, n_tokens: int) -> int: return -(-n_tokens // self.bs)
  def blocks_needed(self, seq_id: int, n_tokens: int) -> int:
    return max(0, self.num_blocks_for(n_tokens) - len(self.tables.get(seq_id, ())))
  def can_ensure(self, seq_id: int, n_tokens: int) -> bool: return self.blocks_needed(seq_id, n_tokens) <= self.allocator.num_free
  def ensure(self, seq_id: int, n_tokens: int) -> list[int]:
    """Grow seq_id's block table so it holds positions 0..n_tokens-1."""
    assert n_tokens <= self.max_seq_len, f"{n_tokens} tokens exceed max_seq_len {self.max_seq_len}"
    t = self.tables.setdefault(seq_id, [])
    if (k := self.blocks_needed(seq_id, n_tokens)): t.extend(self.allocator.allocate(k))
    return t
  def free(self, seq_id: int) -> None:
    if (t := self.tables.pop(seq_id, None)): self.allocator.free(t)
  def reset(self) -> None:
    for s in list(self.tables): self.free(s)
  def slot(self, seq_id: int, pos: int) -> int:
    t = self.tables[seq_id]
    assert pos // self.bs < len(t), f"position {pos} of sequence {seq_id} has no block"
    return t[pos // self.bs] * self.bs + pos % self.bs
  def block_table(self, seq_id: int) -> np.ndarray:
    out = np.full(self.max_blocks, -1, np.int32)
    t = self.tables.get(seq_id, [])
    out[:len(t)] = t
    return out

  # ---- KVCache interface
  def state(self) -> list[Tensor]: return list(self.pools)
  def inputs(self) -> list[Tensor]: return list(self._inputs)
  def bind(self, state: list[Tensor], inputs: list[Tensor] | None = None) -> "PagedKVCache":
    return PagedKVCache(self.n_layers, self.n_kv, self.hd, self.num_blocks, self.bs, self.dtype, self.device, self.max_seq_len,
                        _shared=self, _state=state, _inputs=inputs)
  def load(self, state: list[Tensor]) -> None: self.pools = list(state)

  # ---- ragged steps (continuous batching)
  def step_meta(self, N: int, R: int, tokens: Sequence[tuple[int, int]], sel: Sequence[int]) -> np.ndarray:
    """int32 metadata for a ragged step of N token rows: tokens[i] = (seq_id, position) of row i (rows past len(tokens),
    and entries whose seq_id is None, are padding); sel = the token rows whose logits the step returns (at most R). Every
    listed position must have a block."""
    L = StepLayout(N, R, self.max_blocks)
    m = np.full(L.size, -1, np.int32)
    rows: dict[int, int] = {}
    assert len(tokens) <= N and len(sel) <= R
    for i, (s, p) in enumerate(tokens):
      if s is None: continue
      if s not in rows:
        assert len(rows) < R, f"more than {R} sequences in one step"
        rows[s] = len(rows)
        m[L.bt + rows[s] * L.MB:L.bt + (rows[s] + 1) * L.MB] = self.block_table(s)
      m[L.row + i], m[L.pos + i], m[L.slot + i] = rows[s], p, self.slot(s, p)
    m[L.sel:L.sel + len(sel)] = sel
    return m
  def step_meta_rows(self, N: int, R: int, sid: np.ndarray, pos: np.ndarray, sel: Sequence[int]) -> np.ndarray:
    """step_meta from arrays (the same int32 metadata, bit for bit; tests/test_kv_cache.py): sid[i] the sequence of row
    i (-1: padding), pos[i] its position, for the first len(sid) rows. A 4096-row prefill step took 6.6-6.8 ms of host
    time through the per-row loop (speed run 4 c32-analysis); this is a few numpy ops per sequence in the step."""
    L = StepLayout(N, R, self.max_blocks)
    m = np.full(L.size, -1, np.int32)
    n = len(sid)
    assert n <= N and len(sel) <= R and len(pos) == n
    sid, pos = np.asarray(sid, np.int64), np.asarray(pos, np.int64)
    real = np.flatnonzero(sid >= 0)
    if len(real):
      u, first = np.unique(sid[real], return_index=True)
      order = u[np.argsort(first, kind="stable")]       # rows of the block tables: in order of first appearance
      assert len(order) <= R, f"more than {R} sequences in one step"
      row = np.empty(n, np.int64)
      slot = np.empty(n, np.int64)
      for r, sq in enumerate(order.tolist()):
        t = np.asarray(self.tables[sq], np.int64)
        idx = real[sid[real] == sq]
        p = pos[idx]
        assert int(p.max()) // self.bs < len(t), f"position {int(p.max())} of sequence {sq} has no block"
        row[idx], slot[idx] = r, t[p // self.bs] * self.bs + p % self.bs
        m[L.bt + r * L.MB:L.bt + (r + 1) * L.MB] = self.block_table(sq)
      m[L.row + real], m[L.pos + real], m[L.slot + real] = row[real], pos[real], slot[real]
    m[L.sel:L.sel + len(sel)] = sel
    return m
  def set_step(self, layout: StepLayout, meta: Tensor) -> None: self.layout, self._inputs = layout, [meta]
  def kv_dest(self, layer: int) -> "KVDest | None":
    """Where this ragged step's k / v rows of `layer` go, for a kernel that stores them itself (cuda.rope_qkv kvw=, I60);
    then attend(..., written=True) skips its kv_write. CUDA only (None elsewhere)."""
    if self.layout is None or not self._inputs or self.device.split(":")[0].upper() != "CUDA": return None
    return KVDest(_buf(self.pools[layer]), _buf(self._inputs[0]), self.layout.slot, self.n_kv, self.hd, self.bs, self.num_blocks)
  def attend(self, layer: int, qkv: Tensor, n_heads: int, scale: float, written: bool = False) -> Tensor:
    """qkv: (N, Hq + 2*Hkv, hd) rotated q, rotated k, v of the step's tokens. Writes k and v into the tokens' slots (unless
    `written`: the qkv kernel stored them already, see kv_dest), then returns paged attention (N, Hq*hd): token i
    attends to positions 0..pos_i of its own sequence."""
    L, H, hd = self.layout, self.n_kv, self.hd
    assert L is not None and self._inputs, "set_step() first"
    N, rs = qkv.shape[0], (n_heads + 2 * H) * hd
    xb, meta, pool = _buf(qkv), _buf(self._inputs[0]), _buf(self.pools[layer])
    if not written: kv_write(pool, xb, meta, N, H, hd, self.bs, self.num_blocks, rs, n_heads * hd, (n_heads + H) * hd, L.slot)
    o = paged_attention(xb, pool, meta, N, H, n_heads // H, hd, self.bs, self.num_blocks, L.MB, rs, L.row, L.pos, L.bt, scale, L.prefill, L.np_n, L.mix, L.mix_exact)
    return _wrap(o, (N, n_heads * hd))
  def select(self, x: Tensor) -> Tensor:
    """The step's logits rows of x (N, D) -> (R, D)."""
    assert self.layout is not None
    return gather_rows(x, _buf(self._inputs[0]), self.layout.R, self.layout.sel)

  # ---- the P4-MODEL interface (Llama.prefill / decode, rows sharing positions)
  def set_rows(self, seq_ids: Sequence[int]) -> None:
    """Batch row b of the model's prefill/decode is sequence seq_ids[b]; its block table becomes the step input."""
    self._rows = tuple(seq_ids)
    bt = np.stack([self.block_table(s) for s in seq_ids]).astype(np.int32)
    self._inputs = [Tensor(bt, device=self.device, dtype=dtypes.int32).realize()]
  def update(self, layer: int, k: Tensor, v: Tensor, start: Any, T: Any) -> tuple[Tensor, Tensor]:
    B, H, hd, MB = k.shape[0], self.n_kv, self.hd, self.max_blocks
    assert isinstance(T, int), "update() needs a concrete T"
    assert self._inputs and self._inputs[0].shape == (B, MB), "set_rows() first"
    kv = _buf(k.unsqueeze(0).cat(v.unsqueeze(0), axis=0))                     # (2, B, H, T, hd)
    bt, pool = _buf(self._inputs[0]), _buf(self.pools[layer])
    pr = (B, H, T, hd, self.bs, self.num_blocks, MB, start)
    run_call(ServeCall("fz_kv_write_rows", lambda t: msl_kv_write_rows("fz_kv_write_rows", t), WRITE_ROWS_PARAMS, pr,
                       *_ew(B * H * T * hd), _np_kv_write_rows, lambda t: cu_kv_write_rows("fz_kv_write_rows", t)), (pool, kv, bt))
    S = MB * self.bs
    out = Buffer(pool.device, 2 * B * H * S * hd, pool.dtype)
    run_call(ServeCall("fz_kv_gather", lambda t: msl_kv_gather("fz_kv_gather", t), GATHER_PARAMS, (B, H, MB, self.bs, hd, self.num_blocks),
                       *_ew(2 * B * H * S * hd), _np_kv_gather, lambda t: cu_kv_gather("fz_kv_gather", t)), (out, pool, bt))
    g = _wrap(out, (2, B, H, S, hd))
    return g.shrink(((0, 1), None, None, None, None)).reshape(B, H, S, hd), g.shrink(((1, 2), None, None, None, None)).reshape(B, H, S, hd)
