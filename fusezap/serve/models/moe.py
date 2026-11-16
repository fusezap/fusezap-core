"""Mixture-of-experts MLP for the Llama-style decoder (tracker I105; first model: allenai/OLMoE-1B-7B-0125-Instruct).

OLMoE's layer is Llama's with two changes: q_norm / k_norm, an RMSNorm over the whole q (k) projection (all heads at
once, before RoPE; Qwen3's is per head), and the MLP replaced by a router over E experts, each a SwiGLU of width Ie:

    p = softmax(h @ Wr^T)                    (N, E), float32
    sel = the top_k largest p per row
    y = sum over e in sel of p_e * down_e(silu(gate_e h) * up_e h)        (norm_topk_prob: p_e / sum over sel of p)

**Dense masked form (every device, every kernel static).** The experts' gate / up rows are stacked into one (2 E Ie, D)
weight and their down columns into one (D, E Ie) weight at load time, so the expert MLP is exactly the dense SwiGLU of
width I = E Ie that Llama already runs (`cfg.intermediate_size` = E Ie), with one extra elementwise product between the
two matmuls: the activation a (N, E, Ie) times the routing weight w (N, E, 1), zero for experts outside the row's top-k.
Then down(a * w) = sum over e of w_e down_e(a_e): a zero-weight expert adds exact zeros. Top-k is computed without a sort
or a gather: rank_e = #{j : p_j > p_e} + #{j < e : p_j == p_e} (an (N, E, E) compare and a sum; ties go to the lower
index), sel = rank < top_k. So nothing here has a runtime-dependent loop bound, gather or scatter: on NEURON the router
is one float32 TensorE matmul (D x E) plus renderer elementwise / reduce kernels, and the two expert matmuls are the
same bf16 K-major matmuls (nki_bf16.MatmulBF16) a dense MLP uses (exceptions 2 and 3 unchanged).

Cost (why this form for the bring-up): every token reads every expert's weights, E / top_k = 8x the expert bytes a
sparse kernel would read. Decode is bandwidth-bound, so a decode step costs what a dense model of the same total size
(6.9B) costs; prefill does E / top_k = 8x the expert FLOPs. The alternatives and their costs are in docs/status/I105.md.

**Expert parallelism (Llama(tp=group) with an MoE config; I105 run 1 on one NeuronCore ran out of HBM).** Rank r holds
experts [r E/W, (r+1) E/W) and everything else whole: attention, norms, router and lm_head are replicated, so every rank
computes the same activations bit for bit (the q/k norm over all heads stays on one core), routes over all E experts,
masks with its own E/W columns, and runs down_proj over its K = E/W x Ie; the rank partials are all-reduced (dist.all_reduce:
the group's collective or shm / host sum in rank order, float32). Reduction order vs one core: one core's down_proj
accumulates the K chunks 0 .. E Ie / 128 - 1 in order in PSUM; EP=2 accumulates each rank's half in order, then adds the two
partials (rank 0 + rank 1). So the result is the same sum grouped differently: equal within float32 rounding, not bitwise.
"""
from __future__ import annotations
import os
from typing import Any
import numpy as np
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes

def expert_names(prefix: str, n_experts: int, proj: str, first: int = 0) -> list[str]:
  """HF checkpoint names of one projection of experts first .. first + n_experts - 1 of a layer, in expert order (prefix:
  'model.layers.{i}.mlp.'; first > 0: an expert-parallel rank's share)."""
  return [f"{prefix}experts.{e}.{proj}_proj.weight" for e in range(first, first + n_experts)]

def tie_order(n_experts: int, device: str) -> Tensor:
  """(E, E) float32 constant: [e, j] = 1 iff j < e (an equal probability at a lower index ranks first)."""
  return Tensor(np.tri(n_experts, k=-1, dtype=np.float32), device=device).realize()

def pair_matrices(n_experts: int, device: str) -> tuple[Tensor, Tensor]:
  """NEURON (route): (E, E E) 0/1 matrices: p @ first = [p_e repeated over j] (column e E + j), p @ second = [p_j]."""
  E = n_experts
  rr = np.kron(np.eye(E, dtype=np.float32), np.ones((1, E), np.float32))      # [e, e E + j] = 1
  rt = np.tile(np.eye(E, dtype=np.float32), (1, E))                           # [j, e E + j] = 1
  return Tensor(rr, device=device).realize(), Tensor(rt, device=device).realize()

def _rank(p: Tensor, tri: Tensor, pairs: tuple[Tensor, Tensor] | None = None) -> Tensor:
  """rank_e = #{j : p_j > p_e} + #{j < e : p_j == p_e} (exact small integers in float32; route's docstring)."""
  E = p.shape[-1]
  if pairs is not None:
    lead = p.shape[:-1]
    pe, pj = p @ pairs[0], p @ pairs[1]                              # (..., E E): exact (one product with 1 per output)
    beats = (pj > pe).cast(dtypes.float32) + (pj == pe).cast(dtypes.float32) * tri.reshape(E * E)
    rank = beats.reshape(*lead, E, E).sum(-1)
  else:
    pe, pj = p.unsqueeze(-1), p.unsqueeze(-2)                      # (..., E, 1) the expert's own, (..., 1, E) the others
    rank = ((pj > pe).cast(dtypes.float32) + (pj == pe).cast(dtypes.float32) * tri).sum(-1)   # (..., E), exact small ints
  return rank

def route(p: Tensor, tri: Tensor, top_k: int, norm_topk: bool, pairs: tuple[Tensor, Tensor] | None = None) -> Tensor:
  """p (..., E) router probabilities -> (..., E) routing weights: p_e for the top_k largest p of a row, 0 elsewhere
  (renormalized over the chosen ones with norm_topk). Static shapes only (module docstring). pairs (NEURON, I105 run 1):
  the (..., E, E) operands are built by two exact TensorE matmuls (pair_matrices) instead of stride-0 broadcasts, whose
  DMA access patterns cost one descriptor per element (N E E of them) in the chip's DMA rings."""
  return _weights(p, _rank(p, tri, pairs), top_k, norm_topk)

def _weights(p: Tensor, rank: Tensor, top_k: int, norm_topk: bool) -> Tensor:
  w = (rank < top_k).where(p, 0.0)
  return w / w.sum(-1, keepdim=True) if norm_topk else w

# ---------------------------------------------------------------- I105 A2: sparse decode (only the chosen experts' weights)

def sparse_enabled() -> bool:
  """FZ_MOE_SPARSE=1: on NEURON, decode steps (not prefill forwards) of an MoE model run the expert MLP on the chosen
  experts only (fusezap/codegen/nki_moe.py: their weight tiles read by indirect DMA; docs/status/I105.md part A2). Default
  on since A2 chip run 3 (served, Invarix PROMOTE, c1 TPOT 83.7 -> 31.2 ms); FZ_MOE_SPARSE=0 runs every expert (dense).
  With tensor parallelism it implies the expert-sliced layout (FZ_MOE_TP=slice)."""
  return os.environ.get("FZ_MOE_SPARSE", "1").strip().lower() in ("1", "on", "true", "yes")

def sparse_rows() -> int:
  """The largest decode forward (rows) the sparse kernels serve (FZ_MOE_SPARSE_ROWS, default 8); bigger ones stay dense."""
  return int(os.environ.get("FZ_MOE_SPARSE_ROWS", "") or 8)

def tp_layout() -> str:
  """How an MoE model is split over a tensor-parallel group: "experts" (I105 follow-up: rank r holds experts [r E/W,
  (r+1) E/W) whole) or "slice" (A2: every rank holds every expert's columns [r Ie/W, (r+1) Ie/W) of the expert width, so
  each rank reads 1/W of every chosen expert: the layout sparse decode needs). FZ_MOE_TP; default slice with FZ_MOE_SPARSE."""
  v = os.environ.get("FZ_MOE_TP", "").strip().lower() or ("slice" if sparse_enabled() else "experts")
  assert v in ("experts", "slice"), f"FZ_MOE_TP={v!r}: experts or slice"
  return v

def upper_order(n_experts: int, device: str) -> Tensor:
  """(E, E) float32 constant: [j, e] = 1 iff j < e, so sel @ upper = the number of chosen experts below each expert."""
  return Tensor(np.triu(np.ones((n_experts, n_experts), np.float32), k=1), device=device).realize()

def route_keys(p: Tensor, tri: Tensor, upper: Tensor, top_k: int, norm_topk: bool, pairs: tuple[Tensor, Tensor] | None = None) -> tuple[Tensor, Tensor]:
  """(w, key): w = route()'s routing weights (the same bits), key (..., E) float32 = for a chosen expert the number of
  chosen experts below it (0 .. top_k - 1: its place in increasing expert order), E for the others. sel @ upper is an
  exact 0/1 matmul (a TensorE kernel on NEURON)."""
  E = p.shape[-1]
  rank = _rank(p, tri, pairs)
  chosen = rank < top_k
  pos = chosen.cast(dtypes.float32) @ upper
  return _weights(p, rank, top_k, norm_topk), chosen.where(pos, float(E))

def sparse_gate_up(h: Tensor, key: Tensor, wgu: Tensor, n_experts: int, top_k: int, var: str | None = None) -> Tensor:
  """One MoeGateUpSparse launch: (N, K, 2 Iw) float32, the chosen experts' gate | up products of h (N, D) float32."""
  from fusezap.codegen.nki_moe import MoeGateUpSparse, idma_variant
  from fusezap.device.neuron import NeuronCall
  from fusezap.serve.neuron_kv import _buf, _wrap
  from fusezap.device import Buffer
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  N, D = int(h.shape[0]), int(h.shape[-1])
  Iw = int(wgu.shape[1]) // (2 * n_experts)
  hb = _buf(h if h.dtype == dtypes.float32 else h.float())
  g = Buffer(hb.device, N * top_k * 2 * Iw, dtypes.float32)
  sp = MoeGateUpSparse(N, D, n_experts, Iw, top_k, wgu.dtype == dtypes.bfloat16, var or idma_variant())
  J.run_item(ScheduleItem((NeuronCall(sp),), (g, hb, _buf(key), _buf(wgu)), {}, "custom", 1))
  return _wrap(g, (N, top_k, 2 * Iw))

def sparse_down(a: Tensor, key: Tensor, w: Tensor, wd: Tensor, n_experts: int, top_k: int, var: str | None = None) -> Tensor:
  """One MoeDownSparse launch: (N, D) float32 = per row the chosen experts' down products of w * a (a (N, K, Iw))."""
  from fusezap.codegen.nki_moe import MoeDownSparse, idma_variant
  from fusezap.device.neuron import NeuronCall
  from fusezap.serve.neuron_kv import _buf, _wrap
  from fusezap.device import Buffer
  from fusezap.ir import ScheduleItem
  from fusezap import jit as J
  N, Iw, D = int(a.shape[0]), int(a.shape[-1]), int(wd.shape[1])
  ab = _buf(a.contiguous())
  y = Buffer(ab.device, N * D, dtypes.float32)
  sp = MoeDownSparse(N, D, n_experts, Iw, top_k, wd.dtype == dtypes.bfloat16, var or idma_variant())
  J.run_item(ScheduleItem((NeuronCall(sp),), (y, ab, _buf(key), _buf(w), _buf(wd)), {}, "custom", 1))
  return _wrap(y, (N, D))

def moe_sparse(model: Any, h: Tensor, l: Any) -> Tensor:
  """The expert MLP's output (N, D) float32 of decode rows h (N, D) from the chosen experts only (NEURON): router and
  route_keys as the dense form, MoeGateUpSparse -> silu(gate) * up (the dense form's elementwise ops) -> MoeDownSparse.
  Under the sliced tensor-parallel layout this is the rank's partial sum (all-reduced by the caller). Equal to the dense
  form's down(moe_act(...)) (nki_moe.py: bitwise when the expert width is a multiple of 128)."""
  c = model.cfg
  E, K = c.n_experts, c.top_k
  Iw = c.intermediate_size // E   # this rank's columns of each expert under the sliced layout
  w, key = route_keys(router_probs(model, h, l), model._moe_tri, model._moe_upper, K, c.norm_topk, getattr(model, "_moe_pairs", None))
  w, key = w.float().contiguous(), key.float().contiguous()
  G = sparse_gate_up(h, key, l.wgu, E, K)
  a = model._act(G.shrink((None, None, (0, Iw)))) * G.shrink((None, None, (Iw, 2 * Iw)))
  return sparse_down(a, key, w, l.wd, E, K)

def router_probs(model: Any, h: Tensor, l: Any) -> Tensor:
  """softmax of the router logits in float32 (HF: F.softmax(router_logits, dim=1, dtype=torch.float))."""
  return model.linear(h, l.wr).float().softmax(-1)

def mix(a: Tensor, w: Tensor, n_experts: int, rep: Tensor | None = None) -> Tensor:
  """a (..., E Ie) = silu(gate) * up of every expert, w (..., E) routing weights -> a with expert e's columns times w_e.
  rep (NEURON, I105 run 1): the (E, E Ie) 0/1 block matrix that repeats each weight Ie times, so w is widened by a TensorE
  matmul (exact: one product w_e * 1 per output, the rest adds zeros) and the product is a same-shape elementwise kernel.
  The broadcast form's stride-0 operand made a DMA descriptor per element: 128 rows x 65536 x 64 B = 537 MB of DMA rings
  in one prefill NEFF on the chip (NRT_RESOURCE, 'dma rings io')."""
  sh = a.shape
  Ie = sh[-1] // n_experts
  if rep is not None: return a * (w.cast(dtypes.float32) @ rep).cast(a.dtype)
  return (a.reshape(*sh[:-1], n_experts, Ie) * w.cast(a.dtype).unsqueeze(-1)).reshape(*sh)

def repeat_matrix(n_experts: int, width: int, device: str) -> Tensor:
  """(E, E width) float32: row e is 1 on columns [e width, (e + 1) width)."""
  return Tensor(np.kron(np.eye(n_experts, dtype=np.float32), np.ones((1, width), np.float32)), device=device).realize()

def moe_act(model: Any, h: Tensor, a: Tensor, l: Any) -> Tensor:
  """The dense SwiGLU activation a (gate / up of every expert) masked by h's routing weights: down(result) is the MoE."""
  c = model.cfg
  w = route(router_probs(model, h, l), model._moe_tri, c.top_k, c.norm_topk, getattr(model, "_moe_pairs", None))   # (..., E): every expert
  if getattr(model, "ep", False) and not getattr(model, "moe_slice", False):   # expert parallelism: this rank's columns [ep_lo, ep_lo + E/W); down's sum is all-reduced
    ne = c.n_experts // model.tp.world
    w = w.shrink(tuple([None] * (len(w.shape) - 1)) + ((model.ep_lo, model.ep_lo + ne),))
    return mix(a, w, ne, getattr(model, "_moe_rep", None))
  return mix(a, w, c.n_experts, getattr(model, "_moe_rep", None))
