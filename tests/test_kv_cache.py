"""Paged KV cache (P4-SERVE): block allocator, block tables, slot mapping, and the paged kernels.

Kernel tests run on FZ_DEVICE (default METAL) and compare paged attention with contiguous attention (fusezap's composed
masked sdpa on the same k/v laid out contiguously, and torch); the numpy reference kernels (other devices) are checked
against METAL. The model test runs Qwen2.5-0.5B with the paged cache plugged into Llama.prefill/decode (the P4-MODEL
KVCache interface) and requires the same tokens as the contiguous cache.
"""
from __future__ import annotations
import math, os
import numpy as np
import pytest
from fusezap.tensor import Tensor
from fusezap.dtype import dtypes
from fusezap.device import Device
from fusezap.jit import Jit
from fusezap.serve.kv_cache import BlockAllocator, PagedKVCache, StepLayout

def _device() -> str: return os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
DEV = _device()

# ---------------------------------------------------------------- allocator and block tables

def test_block_allocator():
  a = BlockAllocator(4)
  b = a.allocate(3)
  assert len(set(b)) == 3 and a.num_free == 1
  with pytest.raises(MemoryError): a.allocate(2)
  a.free(b[:2])
  assert a.num_free == 3
  assert set(a.allocate(2)) == set(b[:2])        # LIFO: freed blocks come back first
  with pytest.raises(AssertionError): a.free([b[2], b[2]])

def test_block_allocator_lowest_first():
  """lowest_first (NEURON's pool prefixes): the lowest free ids first, whatever the order of frees; LIFO hands out the
  most recently freed (here the high ones)."""
  for low in (True, False):
    a = BlockAllocator(8, lowest_first=low)
    x, y = a.allocate(3), a.allocate(3)
    assert sorted(x + y) == list(range(6))
    a.free(x); a.free(y)               # all free again; y (3, 4, 5) freed last
    got = a.allocate(2)
    assert got == ([0, 1] if low else [3, 4])
    a.free(got)
    assert a.num_free == 8
    with pytest.raises(AssertionError): a.free([7])

def test_block_tables_and_slots():
  c = PagedKVCache(1, 2, 8, num_blocks=8, block_size=4, dtype=dtypes.float32, device=DEV, max_seq_len=20)
  assert c.max_blocks == 5
  t = c.ensure(7, 5)
  assert len(t) == 2 and c.num_free_blocks == 6
  assert c.ensure(7, 8) == t and c.blocks_needed(7, 9) == 1
  c.ensure(7, 9)
  assert [c.slot(7, p) for p in (0, 3, 4, 8)] == [t[0] * 4, t[0] * 4 + 3, t[1] * 4, t[2] * 4]
  c.ensure(9, 1)
  assert not set(c.tables[7]) & set(c.tables[9])
  assert list(c.block_table(7)) == t + [-1, -1] and c.can_ensure(9, 4 * 5) is True
  with pytest.raises(AssertionError): c.slot(9, 4)
  with pytest.raises(AssertionError): c.ensure(9, 21)    # past max_seq_len
  c.free(7)
  assert c.num_free_blocks == 7 and 7 not in c.tables
  c.reset()
  assert c.num_free_blocks == 8 and not c.tables

def test_step_meta_layout():
  c = PagedKVCache(1, 1, 4, num_blocks=6, block_size=2, dtype=dtypes.float32, device=DEV, max_seq_len=8)
  c.ensure(3, 3); c.ensure(5, 1)
  m = c.step_meta(6, 3, [(3, 1), (3, 2), (5, 0)], [1, 2])
  L = StepLayout(6, 3, 4)
  assert m.size == L.size
  assert list(m[L.row:L.row + 6]) == [0, 0, 1, -1, -1, -1]
  assert list(m[L.pos:L.pos + 3]) == [1, 2, 0]
  assert list(m[L.slot:L.slot + 3]) == [c.slot(3, 1), c.slot(3, 2), c.slot(5, 0)] and list(m[L.slot + 3:L.slot + 6]) == [-1] * 3
  assert list(m[L.sel:L.sel + 3]) == [1, 2, -1]
  assert list(m[L.bt:L.bt + 4]) == list(c.block_table(3)) and list(m[L.bt + 4:L.bt + 8]) == list(c.block_table(5))
  assert all(m[L.bt + 8:L.bt + 12] == -1)

# ---------------------------------------------------------------- paged attention vs contiguous attention

def _ragged(rng, lens: dict[int, tuple[int, int]], Hq: int, Hkv: int, hd: int, bs: int, NB: int, dt, dev=DEV, pad: int = 3):
  """A cache whose sequences already hold `start` positions (written by an earlier step) and a step that adds `n` more
  for each: lens[seq] = (start, n). Blocks are handed out interleaved so no sequence's blocks are contiguous."""
  c = PagedKVCache(1, Hkv, hd, NB, bs, dt, dev, max_seq_len=NB * bs)
  total = {s: a + n for s, (a, n) in lens.items()}
  for k in range(max(-(-t // bs) for t in total.values())):  # round robin
    for s, t in total.items():
      if k * bs < t: c.ensure(s, (k + 1) * bs if (k + 1) * bs <= t else t)
  hist = {s: rng.standard_normal((a, 2, Hkv, hd)).astype(np.float32) for s, (a, _) in lens.items()}   # earlier k/v
  pool = np.zeros((2, NB, Hkv, bs, hd), np.float32)
  for s, kv in hist.items():
    for p in range(len(kv)): pool[:, c.slot(s, p) // bs, :, c.slot(s, p) % bs] = kv[p].reshape(2, Hkv, hd)
  pool = pool.astype(np.float16) if dt == dtypes.float16 else pool
  c.pools = [Tensor(pool, device=dev, dtype=dt).realize()]
  toks = [(s, p) for s, (a, n) in lens.items() for p in range(a, a + n)]
  N = len(toks) + pad
  qkv = rng.standard_normal((N, Hq + 2 * Hkv, hd)).astype(np.float32)
  if dt == dtypes.float16: qkv = qkv.astype(np.float16)
  return c, toks, N, qkv, hist

def _contiguous_ref(toks, qkv, hist, Hq, Hkv, hd, dt):
  """Composed fusezap attention per sequence on contiguous K/V (earlier positions, then this step's), causal by position."""
  G, out = Hq // Hkv, {}
  for s in {s for s, _ in toks}:
    rows = [i for i, (s2, _) in enumerate(toks) if s2 == s]
    pos = np.array([toks[i][1] for i in rows])
    kv_new = qkv[rows, Hq:].astype(np.float32).reshape(len(rows), 2, Hkv, hd)
    kv = np.concatenate([hist[s], kv_new], 0)                       # (S, 2, Hkv, hd), row j = position j
    S = kv.shape[0]
    q = Tensor(qkv[rows, :Hq].transpose(1, 0, 2)[None].copy(), device=DEV, dtype=dt)                                  # (1, Hq, T, hd)
    K = Tensor(np.repeat(kv[:, 0], G, axis=1).transpose(1, 0, 2)[None].astype(qkv.dtype).copy(), device=DEV, dtype=dt)  # (1, Hq, S, hd)
    V = Tensor(np.repeat(kv[:, 1], G, axis=1).transpose(1, 0, 2)[None].astype(qkv.dtype).copy(), device=DEV, dtype=dt)
    mask = Tensor(np.arange(S)[None, :] <= pos[:, None], device=DEV)
    o = q.scaled_dot_product_attention(K, V, attn_mask=mask).numpy()[0].transpose(1, 0, 2)   # (T, Hq, hd)
    for r, i in enumerate(rows): out[i] = o[r]
  return out

@pytest.mark.parametrize("dt", [dtypes.float32, dtypes.float16])
@pytest.mark.parametrize("Hq,Hkv,hd,bs", [(14, 2, 64, 16), (4, 4, 32, 4), (6, 2, 40, 8), (8, 1, 128, 16)])
def test_paged_attention_matches_contiguous(dt, Hq, Hkv, hd, bs):
  rng = np.random.default_rng(Hq * hd + bs)
  # a decode token (long history), a fresh prompt, a continued chunk, a one-token prompt, and a history crossing blocks
  lens = {10: (37, 1), 11: (0, 23), 12: (5, 9), 13: (0, 1), 14: (2 * bs, 1)}
  c, toks, N, qkv, hist = _ragged(rng, lens, Hq, Hkv, hd, bs, 64, dt)
  c.set_step(StepLayout(N, 5, c.max_blocks), Tensor(c.step_meta(N, 5, toks, []), device=DEV, dtype=dtypes.int32))
  o = c.attend(0, Tensor(qkv, device=DEV, dtype=dt), Hq, 1.0 / math.sqrt(hd)).numpy().astype(np.float32).reshape(N, Hq, hd)
  ref = _contiguous_ref(toks, qkv, hist, Hq, Hkv, hd, dt)
  tol = dict(rtol=1e-4, atol=1e-5) if dt == dtypes.float32 else dict(rtol=1e-2, atol=1e-2)
  for i in range(len(toks)): np.testing.assert_allclose(o[i], ref[i].astype(np.float32), **tol)
  assert np.all(o[len(toks):] == 0)          # padding rows: zeros
  # the step's k/v landed in their slots, nothing else was written
  pool = c.pools[0].numpy().astype(np.float32)
  for i, (s, p) in enumerate(toks):
    sl = c.slot(s, p)
    np.testing.assert_array_equal(pool[:, sl // bs, :, sl % bs], qkv[i, Hq:].astype(np.float32).reshape(2, Hkv, hd))
  free = sorted(set(range(64)) - {b for t in c.tables.values() for b in t})
  assert np.all(pool[:, free] == 0)

def test_paged_attention_vs_torch():
  torch = pytest.importorskip("torch")
  rng = np.random.default_rng(1)
  Hq, Hkv, hd, bs = 14, 2, 64, 16
  c, toks, N, qkv, hist = _ragged(rng, {0: (100, 1), 1: (0, 40)}, Hq, Hkv, hd, bs, 32, dtypes.float32, pad=0)
  c.set_step(StepLayout(N, 2, c.max_blocks), Tensor(c.step_meta(N, 2, toks, []), device=DEV, dtype=dtypes.int32))
  o = c.attend(0, Tensor(qkv, device=DEV), Hq, 1.0 / math.sqrt(hd)).numpy().reshape(N, Hq, hd)
  for s, rows in ((0, [0]), (1, list(range(1, 41)))):
    kv = np.concatenate([hist[s], qkv[rows, Hq:].reshape(len(rows), 2, Hkv, hd)], 0)
    q = torch.tensor(qkv[rows, :Hq]).transpose(0, 1)[None]
    K = torch.tensor(kv[:, 0]).transpose(0, 1)[None].repeat_interleave(Hq // Hkv, 1)
    V = torch.tensor(kv[:, 1]).transpose(0, 1)[None].repeat_interleave(Hq // Hkv, 1)
    pos = torch.tensor([toks[i][1] for i in rows])
    ref = torch.nn.functional.scaled_dot_product_attention(q, K, V, attn_mask=torch.arange(len(kv))[None] <= pos[:, None])
    np.testing.assert_allclose(o[rows], ref[0].transpose(0, 1).numpy(), rtol=1e-4, atol=1e-5)

def test_paged_kernels_reference_matches_metal():
  if DEV != "METAL" or not Device.available("CPU"): pytest.skip("needs METAL and CPU")
  rng = np.random.default_rng(2)
  Hq, Hkv, hd, bs = 4, 2, 16, 4
  lens = {0: (6, 1), 1: (0, 7), 2: (3, 2)}
  outs = []
  for dev in ("METAL", "CPU"):
    c, toks, N, qkv, _ = _ragged(np.random.default_rng(2), lens, Hq, Hkv, hd, bs, 16, dtypes.float32, dev=dev)
    sel = [0, 7, 9]
    c.set_step(StepLayout(N, 4, c.max_blocks), Tensor(c.step_meta(N, 4, toks, sel), device=dev, dtype=dtypes.int32))
    o = c.attend(0, Tensor(qkv, device=dev), Hq, 1.0 / math.sqrt(hd))
    outs.append((o.numpy(), c.pools[0].numpy(), c.select(o).numpy()))
  np.testing.assert_allclose(outs[0][0], outs[1][0], rtol=1e-4, atol=1e-5)
  np.testing.assert_array_equal(outs[0][1], outs[1][1])
  np.testing.assert_allclose(outs[0][2], outs[1][2], rtol=1e-4, atol=1e-5)
  assert np.array_equal(outs[0][2][:3], outs[0][0][[0, 7, 9]]) and np.all(outs[0][2][3] == 0)

def test_paged_attention_jit_replay():
  """attend() inside a fz.jit step: the custom kernels are recorded with the composed ones and replay with new metadata
  and new pool contents (the third call does not trace)."""
  Hq, Hkv, hd, bs, NB = 4, 2, 32, 4, 32
  c = PagedKVCache(1, Hkv, hd, NB, bs, dtypes.float32, DEV, max_seq_len=64)
  N, R = 8, 4
  L = StepLayout(N, R, c.max_blocks)
  calls = []
  def step(x: Tensor, meta: Tensor, pools: list[Tensor]) -> Tensor:
    calls.append(1)
    b = c.bind(pools, [meta]); b.layout = L
    o = b.attend(0, (x * 2.0).reshape(N, Hq + 2 * Hkv, hd), Hq, 0.25)
    return b.select(o + 1.0)
  jf = Jit(step)
  rng = np.random.default_rng(3)
  hist: dict[int, list[np.ndarray]] = {0: [], 1: []}
  for it in range(5):
    s_lens = {0: 1 if it else 5, 1: 2}
    toks = []
    for s, n in s_lens.items():
      a = len(hist[s])
      c.ensure(s, a + n)
      toks += [(s, p) for p in range(a, a + n)]
    x = rng.standard_normal((N, (Hq + 2 * Hkv) * hd)).astype(np.float32)
    sel = [len(toks) - 1, 0]
    out = jf(Tensor(x, device=DEV), Tensor(c.step_meta(N, R, toks, sel), device=DEV, dtype=dtypes.int32), c.state()).numpy()
    q = (2 * x).reshape(N, Hq + 2 * Hkv, hd)
    for i, (s, p) in enumerate(toks): hist[s].append(q[i, Hq:])
    for r, i in enumerate(sel):
      s, p = toks[i]
      kv = np.stack(hist[s][:p + 1])                                  # (S, 2*Hkv, hd)
      for h in range(Hq):
        sc = kv[:, h // 2] @ q[i, h] * 0.25
        e = np.exp(sc - sc.max())
        np.testing.assert_allclose(out[r, h * hd:(h + 1) * hd], (e / e.sum()) @ kv[:, Hkv + h // 2] + 1.0, rtol=1e-4, atol=1e-5)
  assert len(calls) == 2

# ---------------------------------------------------------------- the paged cache inside Llama.prefill / decode

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
def _model_path() -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(MODEL, allow_download=False)
  except Exception: return None
needs_model = pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")

@needs_model
def test_llama_with_paged_cache_matches_contiguous():
  """The P4-MODEL KVCache interface: Llama.prefill/decode with a PagedKVCache (scatter by block table, gather back to a
  position-ordered view) produce the contiguous cache's logits and greedy tokens, with interleaved block tables."""
  from fusezap.serve.models import load_model
  from fusezap.serve.sampling import Sampler
  m = load_model(MODEL, dtype=dtypes.float32, device=DEV, max_len=128, prefill_chunk=32)
  ids = np.random.default_rng(4).integers(1000, 30000, 45).tolist()
  ref = list(m.generate(ids, 10, Sampler(), stop=[]))
  m.reset()
  ref_lg = m.prefill([ids])
  pc = PagedKVCache.for_model(m, num_blocks=24, block_size=8, max_seq_len=m.max_len)
  pc.ensure(99, 3 * 8)              # another sequence owns blocks in between, so block 0..n of ours are not contiguous
  pc.ensure(1, 8); pc.ensure(99, 5 * 8); pc.ensure(1, m.max_len)
  pc.set_rows([1])
  m.cache, m.pos = pc, 0
  m._decode_jit.reset(); m._prefill_jits.clear()   # the captured steps were traced against the contiguous cache
  lg = m.prefill([ids])
  np.testing.assert_allclose(lg, ref_lg, rtol=1e-4, atol=1e-3)
  toks, hist = [], list(ids)
  for _ in range(10):
    t = int(Sampler()(lg[0], hist))
    toks.append(t); hist.append(t)
    lg = m.decode([[t]])
  assert toks == ref
