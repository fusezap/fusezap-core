"""I77: the prefill kernels (fusezap/codegen/nki_prefill.py, FZ_NEURON_PREFILL_KERNEL=v2) on the Mac (NKISHIM):

  1. kernels alone: PFLinear bit for bit vs MatmulBF16 (row tails, K tails, a few rows, a vocabulary window); PFQKV, PFAttn,
     PFMLP against their float64 references (bf16 operands) at Qwen2.5-7B's head layout (14 / 2 heads per rank, head_dim
     128); ScalarE exp (FZ_NEURON_PFK_EXP=act) within tolerance; the weight block widths (PSUM banks)
  2. the serving forward: a prefill's layers on the kernels give the members' logits bit for bit (FZ_NKISHIM_MM=f64: row
     moves do not change a float32 sum), one device (two chunks, the second over the first one's keys) and TP=2 with the
     collectives inside the one prefill NEFF and the lm_head split over the ranks; LLMEngine's greedy tokens with prefill
     sizes 32 and 64 (FZ_NEURON_PREFILL_SIZES); the self-test gate (a failing kernel turns v2 off: the members run)
  3. the switches: FZ_NEURON_PREFILL_SIZES, the prewarm plan skipping unreachable prefixes of a bigger prefill size
"""
from __future__ import annotations
import os
import numpy as np
import pytest
from fusezap.device import nkishim as S
from fusezap.codegen import nki_prefill as K
from fusezap.codegen.nki_bf16 import MatmulBF16
from fusezap.serve import neuron_prefill as PF
from fusezap.serve.weights import f32_to_bf16
from fusezap.codegen.nki_kernels import HEADER
from fusezap.dtype import dtypes

TIER = "NKISHIM"
Q7B = {"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 28, "num_key_value_heads": 4, "head_dim": 128}   # 14 / 2 per rank
# Llama-3.1-8B's head layout (I13): 32 / 8 heads (16 / 4 per rank), no biases, eps 1e-5, llama3 RoPE scaling
L8B = {"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "model_type": "llama",
       "rms_norm_eps": 1e-5, "rope_theta": 5e5, "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                                                                "high_freq_factor": 4.0, "original_max_position_embeddings": 64}}
LAYOUTS = {"qwen7b": (Q7B, True), "llama8b": (L8B, False)}   # (config, q/k/v and o_proj biases)

def _r(rng, shape, sc=1.0): return (rng.standard_normal(shape) * sc).astype(np.float32)

@pytest.fixture(autouse=True)
def _f64(monkeypatch):
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  for k in ("FZ_NEURON_PREFILL_KERNEL", "FZ_NEURON_PFK_EXP", "FZ_NEURON_PFK_HEAD", "FZ_NEURON_PFK_BANKS", "FZ_NEURON_PFK_WN",
            "FZ_NEURON_PREFILL_SIZES", "FZ_NEURON_PFK_EVICT", "FZ_NEURON_PFK_SELFTEST", "FZ_NEURON_PFK_ATTN_HB", "FZ_NEURON_PFK_SILU",
            "FZ_NEURON_PFK_EXPF"): monkeypatch.delenv(k, raising=False)
  PF._STATE.update(ok=None, why="", override=None, knobs=None)
  yield
  PF._STATE.update(ok=None, why="", override=None, knobs=None)

# ---------------------------------------------------------------- 1. kernels

@pytest.mark.parametrize("M,N,Kd,banks,wn", [(128, 1792, 384, 8, 4096), (200, 1000, 300, 4, 2048), (4, 2600, 256, 8, 4096), (384, 700, 128, 8, 4096)])
def test_pflinear_bitwise_vs_matmul_bf16(M, N, Kd, banks, wn):
  rng = np.random.default_rng(M + N)
  a, w = _r(rng, (M, Kd)), f32_to_bf16(_r(rng, (Kd, N), 0.1))
  c1, c2 = np.zeros(M * N, np.float32), np.zeros(M * N, np.float32)
  S.execute(K.PFLinear(M, N, Kd, banks=banks, wn=wn).render(), [c1, a.reshape(-1), w.reshape(-1)])
  S.execute(MatmulBF16(M, N, Kd).render(), [c2, a.reshape(-1), w.reshape(-1)])
  np.testing.assert_array_equal(c1.view(np.int32), c2.view(np.int32))

def test_pflinear_window_and_name():
  """A vocabulary window (the split lm_head): columns [c_off, c_off + N) of a (K, ws) weight."""
  rng = np.random.default_rng(1)
  M, Nf, Kd = 4, 2048, 256
  a, w = _r(rng, (M, Kd)), f32_to_bf16(_r(rng, (Kd, Nf), 0.1))
  sp = K.PFLinear(M, 1024, Kd, Nf, 1024)
  c = np.zeros(M * 1024, np.float32)
  S.execute(sp.render(), [c, a.reshape(-1), w.reshape(-1)])
  full = np.zeros(M * Nf, np.float32)
  S.execute(MatmulBF16(M, Nf, Kd).render(), [full, a.reshape(-1), w.reshape(-1)])
  np.testing.assert_array_equal(c.reshape(M, -1), full.reshape(M, Nf)[:, 1024:])
  assert "_w2048o1024" in sp.name and sp.sizes() == [M * 1024, M * Kd, Kd * Nf]
  with pytest.raises(Exception): K.PFLinear(M, 1024, Kd, Nf, 1536).render()
  for off in (0, 1024):   # pad: the product in its window of a (M, ws) output, zeros elsewhere (the split head's gather input)
    sp = K.PFLinear(M, 1024, Kd, Nf, off, pad=True)
    c = np.full(M * Nf, np.nan, np.float32)
    S.execute(sp.render(), [c, a.reshape(-1), w.reshape(-1)])
    want = np.zeros((M, Nf), np.float32); want[:, off:off + 1024] = full.reshape(M, Nf)[:, off:off + 1024]
    np.testing.assert_array_equal(c.reshape(M, Nf), want)
    assert sp.sizes()[0] == M * Nf and sp.name.endswith("_pad")

def test_block_cols():
  assert [K.block_cols(n, 8, 4096) for n in (1, 128, 129, 256, 384, 512)] == [4096, 4096, 2048, 2048, 1024, 1024]
  assert [K.block_cols(n, 4, 4096) for n in (128, 256, 512)] == [2048, 1024, 512]
  assert K.block_cols(128, 8, 1024) == 1024

def _meta(N: int, P: int, base: int, start: int = 0) -> np.ndarray:
  """One sequence: token t at position start + t, the pool's slots 0 .. P-1 its positions (kid = slot)."""
  m = np.zeros(base + 2 * N + P, np.int32)
  m[base:base + N], m[base + N:base + 2 * N], m[base + 2 * N:] = 0, start + np.arange(N), np.arange(P)
  return m

@pytest.mark.parametrize("layout", ["qwen7b", "llama8b"])
@pytest.mark.parametrize("N", [128, 160])
def test_kernels_vs_float64_references(N, layout):
  """PFQKV, PFAttn, PFMLP at the 7B head layout (q/k/v bias, o_proj bias) and at Llama-3.1-8B's (16 / 4 heads per rank, no
  biases) against nki_prefill's float64 references (bf16 matmul operands)."""
  rng = np.random.default_rng(N)
  qwen = layout == "qwen7b"
  D, Hq, Hkv, hd, I, P = 256, (14 if qwen else 16), (2 if qwen else 4), 128, 384, 256
  Nq = (Hq + 2 * Hkv) * hd
  sp = PF.specs(N, D, Hq, Hkv, hd, I, P, 1e-6, 1 / np.sqrt(hd), 8, True, qwen, qwen, {"banks": 8, "wn": 4096, "exp": "exact"})
  q = sp["qkv"]
  x, d, ln1 = _r(rng, (N, D)), _r(rng, (N, D)), 1 + _r(rng, D, 0.1)
  w, bq = f32_to_bf16(_r(rng, (D, Nq), 0.05)), (_r(rng, Nq, 0.1) if qwen else None)
  ang = rng.uniform(0, 3, (N, hd // 2)).astype(np.float32)
  cs, sn = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([-np.sin(ang), np.sin(ang)], -1)
  arr = {"xo": np.zeros(N * D, np.float32), "xs": np.zeros(N * (Nq + hd), np.float32), "x": x, "d": d, "ln1": ln1, "wqkv": w, "bqkv": bq, "cos": cs, "sin": sn}
  arrays = [np.ascontiguousarray(arr[k]).reshape(-1) for k in q.args()]
  assert ("bqkv" in q.args()) == qwen
  S.execute(q.render(), arrays)
  xo_w, X_w = K.ref_qkv(q, x, d, ln1, w, bq, cs, sn)
  np.testing.assert_allclose(arrays[0].reshape(N, D), xo_w, rtol=1e-6, atol=1e-6)
  assert np.abs(arrays[1].reshape(N, -1) - X_w).max() <= 1e-4 * np.abs(X_w).max()
  at = sp["attn"]
  X = _r(rng, (N, Nq + hd)); pool = _r(rng, (P, 2 * Hkv * hd))
  meta = _meta(N, P, 8, start=P - N)
  o = np.zeros(N * Hq * hd, np.float32)
  S.execute(at.render(), [o, X.reshape(-1), pool.reshape(-1), meta])
  want = K.ref_attn(at, X, pool.reshape(-1), meta)
  assert np.abs(o.reshape(N, -1) - want).max() <= 1e-5 * np.abs(want).max()
  m = sp["mlp"]
  xm, om, bo, ln2 = _r(rng, (N, D)), _r(rng, (N, D)), (_r(rng, D, 0.3) if qwen else None), 1 + _r(rng, D, 0.1)
  wgu, wd = f32_to_bf16(_r(rng, (D, 2 * I), 0.05)), f32_to_bf16(_r(rng, (I, D), 0.05))
  arr = {"xo": np.zeros(N * D, np.float32), "dn": np.zeros(N * D, np.float32), "x": xm, "o": om, "bo": bo, "ln2": ln2, "wgu": wgu, "wd": wd}
  arrays = [np.ascontiguousarray(arr[k]).reshape(-1) for k in m.args()]
  S.execute(m.render(), arrays)
  xo_w, dn_w = K.ref_mlp(m, xm, om, bo, ln2, wgu, wd)
  np.testing.assert_allclose(arrays[0].reshape(N, D), xo_w, rtol=1e-6, atol=1e-6)
  # bf16(a): a rounding flip of a near a bf16 tie moves dn by one bf16 ulp of a times w; the reference rounds a from float64
  assert np.abs(arrays[1].reshape(N, D) - dn_w).max() <= 2e-2 * np.abs(dn_w).max()

def test_exp_act_within_tolerance():
  """FZ_NEURON_PFK_EXP=act: one ScalarE exp per softmax block / silu (the shim's exp is numpy's: within float32 of exact)."""
  rng = np.random.default_rng(5)
  N, Hq, Hkv, hd, P = 128, 14, 2, 128, 128
  ex, ac = K.PFAttn(N, Hq, Hkv, hd, P, 0.088, 4), K.PFAttn(N, Hq, Hkv, hd, P, 0.088, 4, "act")
  assert ac.name.endswith("_act") and "nl.exp" in ac.render().src and "nl.exp" not in ex.render().src
  X = _r(rng, (N, (Hq + 2 * Hkv + 1) * hd)); pool = _r(rng, (P, 4 * hd))
  meta = _meta(N, P, 4)
  outs = []
  for sp in (ex, ac):
    o = np.zeros(N * Hq * hd, np.float32)
    S.execute(sp.render(), [o, X.reshape(-1), pool.reshape(-1), meta]); outs.append(o)
  assert np.abs(outs[0] - outs[1]).max() <= 1e-5 * np.abs(outs[0]).max()
  D, I = 256, 384
  m1, m2 = K.PFMLP(N, D, I, 1e-6), K.PFMLP(N, D, I, 1e-6, exp="act")
  xm, om, ln2 = _r(rng, (N, D)), _r(rng, (N, D)), 1 + _r(rng, D, 0.1)
  wgu, wd = f32_to_bf16(_r(rng, (D, 2 * I), 0.05)), f32_to_bf16(_r(rng, (I, D), 0.05))
  dns = []
  for m in (m1, m2):
    arrays = [np.zeros(N * D, np.float32), np.zeros(N * D, np.float32), xm.reshape(-1), om.reshape(-1), ln2, wgu.reshape(-1), wd.reshape(-1)]
    S.execute(m.render(), arrays); dns.append(arrays[1])
  assert np.abs(dns[0] - dns[1]).max() <= 2e-2 * np.abs(dns[0]).max()

@pytest.mark.parametrize("N,P", [(128, 128), (128, 1024), (512, 512), (512, 1024)])
def test_onchip_budget_at_7b_shard_shapes(N, P):
  """Every kernel at Qwen2.5-7B's TP=2 shard shapes (D 3584, I 9472, 14 / 2 heads, head_dim 128) fits the on-chip memory by
  its live tiles (nki_prefill.onchip_peak): SBUF <= 176 KiB (nki_bf16.LAYER_SBUF of 192), PSUM <= 16 KiB (8 banks), in both
  exp modes, with main's kernels and with I69's wide exact-exp forms; the head's vocabulary window too."""
  seen: set = set()
  for exp in ("exact", "act"):
    for i69 in ({}, {"attn_hb": 2048, "silu": 2048, "expf": True}):   # main's kernels, I69's
      sp = PF.specs(N, 3584, 14, 2, 128, 9472, P, 1e-6, 0.088, 64, True, True, False, {"banks": 8, "wn": 4096, "exp": exp, "evict": "act", **i69})
      for nm, s in sp.items():
        if nm == "kvw" or (i69 and s.name in seen): continue   # I69's knobs leave this kernel as main's (act, 512 rows): checked
        seen.add(s.name)
        pk = K.onchip_peak(s.render(), s.sizes())
        assert pk["sbuf"] <= 176 * 1024 and pk["psum"] <= 16 * 1024, (nm, exp, i69, pk)
  h = K.PFLinear(4, 76032, 3584, 152064, 76032)
  pk = K.onchip_peak(h.render(), h.sizes())
  assert pk["sbuf"] <= 176 * 1024 and pk["psum"] <= 16 * 1024, pk

@pytest.mark.parametrize("N,P", [(128, 128), (128, 1024), (512, 512), (512, 1024)])
def test_onchip_budget_at_llama8b_shard_shapes(N, P):
  """I13: the same at Llama-3.1-8B's TP=2 shard shapes (D 4096, I 7168, 16 / 4 heads, head_dim 128, no biases, eps 1e-5)
  and its split lm_head (128,256 / 2 = 64,128 vocabulary columns per rank)."""
  seen: set = set()
  for exp in ("exact", "act"):
    for i69 in ({}, {"attn_hb": 2048, "silu": 2048, "expf": True}):   # main's kernels, I69's
      sp = PF.specs(N, 4096, 16, 4, 128, 7168, P, 1e-5, 0.088, 64, True, False, False, {"banks": 8, "wn": 4096, "exp": exp, "evict": "act", **i69})
      for nm, s in sp.items():
        if nm == "kvw" or (i69 and s.name in seen): continue   # I69's knobs leave this kernel as main's (act, 512 rows): checked
        seen.add(s.name)
        pk = K.onchip_peak(s.render(), s.sizes())
        assert pk["sbuf"] <= 176 * 1024 and pk["psum"] <= 16 * 1024, (nm, exp, i69, pk)
  h = K.PFLinear(4, 64128, 4096, 128256, 64128)
  pk = K.onchip_peak(h.render(), h.sizes())
  assert pk["sbuf"] <= 176 * 1024 and pk["psum"] <= 16 * 1024, pk

def test_attention_dmas_per_layer():
  """The point of PFAttn: ~4 DMAs per q head and row block (q rows, output) plus per row block lo / hi and one key-id row per
  key block, instead of render_attention's one per token for q, lo, hi and the output (~1,070 per layer at 7B, 128 rows)."""
  from fusezap.codegen.nki_kernels import AttnSpec
  N, Hq, Hkv, hd, P = 128, 14, 2, 128, 512
  G, Wx, C = Hq // Hkv, (Hq + 2 * Hkv + 1) * hd, Hkv * hd
  new = K.PFAttn(N, Hq, Hkv, hd, P, 0.088, 0).render().src.count("nisa.dma_copy(")
  old = AttnSpec((Hkv, 1), (N, G), P, hd, hd, (G * hd, 0, hd, 1, 0), (hd, 0, 2 * C, 1, 0), (hd, 0, 2 * C, 1, C), (G * hd, 0, hd, 1, 0),
                 0.088, "keyrange", (1, 0, 0, N, 2 * N), rs=(Wx, Hq * hd), kv_one=True).render().src
  assert new <= 2 * Hq + 2 + P // 512 + P // 128, new
  assert "for r1 in range(" in old   # the old kernel: one DMA per token run (traced per token)

# ---------------------------------------------------------------- 2. the serving forward

def _model(tmp_path, seed=3, layers=2, cfg=None, chunk=64, max_len=256, bias=True):
  from test_tp import tiny_model_dir
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  p = tiny_model_dir(str(tmp_path / f"m{seed}"), seed=seed, cfg={**(cfg or Q7B), "num_hidden_layers": layers}, bias=bias)
  return load_model(p, dtype=dtypes.bfloat16, device=TIER, max_len=max_len, prefill_chunk=chunk), p

def _prefill_names(m) -> set:
  out = set()
  for (N, R), jf in m._neuron_steps.jits.items():
    if N == R or jf.captured is None: continue
    for it in jf.captured.items:
      sp = getattr(it.si.ast[0], "spec", None) if it.si.ast else None
      if sp is not None: out |= {p[0] for p in sp.parts} if hasattr(sp, "parts") else {sp.name}
  return out

def _two_chunks(m, rng_seed=7):
  m.reset()
  ids = list(np.random.default_rng(rng_seed).integers(0, 199, 50))
  a = m.prefill([ids])
  b = m.prefill([list(np.random.default_rng(rng_seed + 1).integers(0, 199, 30))], start=50)
  return [a, b] + [m.decode([[int(t)]]) for t in (5, 17, 33)]

@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_prefill_v2_bitwise_vs_members_one_device(tmp_path, monkeypatch, layout):
  """One device, the 7B head layout (and Llama-3.1-8B's: no biases), bf16: v2's prefill logits (a 64-row chunk, then a second
  chunk over the first one's keys) are the members' bit for bit, and the decode after them too; v2's forward holds the PF
  kernels and no MatmulBF16 / attention."""
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "exact")
  runs = {}
  cfg, bias = LAYOUTS[layout]
  for mode in ("v1", "v2"):
    monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", mode)
    m, _ = _model(tmp_path, cfg=cfg, bias=bias)
    assert (m.layers[0].bqkv is not None) == bias
    first, out = _two_chunks(m), _two_chunks(m)   # the second round replays the captured forwards
    for a, b in zip(first, out): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
    runs[mode] = (out, _prefill_names(m))
  for a, b in zip(runs["v1"][0], runs["v2"][0]): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
  names = runs["v2"][1]
  assert any(n.startswith("fz_pfqkv_add") for n in names) and any(n.startswith("fz_pfattn") for n in names) and \
         any(n.startswith("fz_pfmlp") for n in names) and any(n.startswith("fz_pflin") for n in names), names
  assert not any(n.startswith(("fz_attention", "fz_matmul_1_64")) for n in names), names
  assert any(n.startswith("fz_attention") for n in runs["v1"][1])

def test_prefill_v2_act_close_to_members(tmp_path, monkeypatch):
  """v2 with FZ_NEURON_PFK_EXP=act (ScalarE exp in attention and silu) and FZ_NEURON_PFK_EVICT=act is not bitwise but
  within float32 noise of the members on the shim (its exp is numpy's), with the same argmax; ScalarE copies evict the MLP."""
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "act"); monkeypatch.setenv("FZ_NEURON_PFK_EVICT", "act")
  runs = {}
  for mode in ("v1", "v2"):
    monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", mode)
    m, _ = _model(tmp_path, seed=6)
    _two_chunks(m)
    runs[mode] = (_two_chunks(m), _prefill_names(m))
  for a, b in zip(runs["v1"][0], runs["v2"][0]):
    assert np.abs(a - b).max() <= 1e-2 * np.abs(a).max() and (a.argmax(-1) == b.argmax(-1)).all()
  names = runs["v2"][1]
  assert any(n.startswith("fz_pfattn") and n.endswith("_act") for n in names), names
  assert any(n.startswith("fz_pfmlp") and n.endswith("_act") for n in names), names
  assert "engine=nisa.scalar_engine" in K.PFMLP(64, 256, 384, 1e-6).render().src
  assert "engine=nisa.scalar_engine" not in K.PFMLP(64, 256, 384, 1e-6, evict="vector").render().src

def test_gate_failing_selftest_runs_the_members(tmp_path, monkeypatch):
  """v2 asked for, the self-test failing (set_ok False, as gate() does on a compile or value failure): the members run."""
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v2")
  PF.set_ok(None, False, "injected")
  assert PF.kernel_mode() == "v2" and not PF.enabled()
  m, _ = _model(tmp_path, seed=4, layers=1)
  _two_chunks(m); _two_chunks(m)
  names = _prefill_names(m)
  assert not any(n.startswith("fz_pf") for n in names) and any(n.startswith("fz_attention") for n in names), names

def test_selftest_off_neuron_and_gate(monkeypatch):
  """selftest only runs kernels on NEURON (NKISHIM: ok, nothing run); gate() sets the verdict once."""
  r = PF.selftest(type("G", (), {"device": "NKISHIM", "rank": 0})(), object())
  assert r["ok"] and "not a NEURON" in r["why"]
  class Eng:
    prefill_sizes, tp = [128], None
    cache = type("C", (), {"ctx_buckets": lambda self, pf=False: [128, 256]})()
    model = type("M", (), {"device": "NKISHIM"})()
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v1")
  assert PF.gate(Eng()) == "" and PF._STATE["ok"] is None   # v1: no test
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v2")
  line = PF.gate(Eng())
  assert "v2) on" in line and PF._STATE["ok"] is True and PF.enabled()
  PF._STATE["ok"] = None
  monkeypatch.setenv("FZ_NEURON_PFK_SELFTEST", "0")
  assert "skipped" in PF.gate(Eng()) and PF.enabled()

@pytest.mark.parametrize("exp", ["act", "exact"])
def test_selftest_runs_and_catches_a_wrong_kernel(tmp_path, monkeypatch, exp):
  """The self-test's own path (forced on NKISHIM): every kernel at the model's shapes within its reference tolerance; a kernel
  that computes the wrong thing (a RoPE sign flipped) fails it, naming the kernel."""
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", exp)
  m, _ = _model(tmp_path, seed=9, layers=1)
  G = type("G", (), {"device": "NKISHIM", "rank": 0})()
  r = PF.selftest(G, m, (64, 32), (128,), force=True)   # every prefill size
  assert r["ok"], r
  assert len(r["kernels"]) == 8 and sum(k.startswith("fz_pfmlp") for k in r["kernels"]) == 2, r["kernels"]
  orig = K.PFQKV.render
  def bad(self):
    p = orig(self)
    return type(p)(p.name, p.src.replace("op=nl.add)\n", "op=nl.subtract)\n", 1), p.n_bufs, p.outs, p.buf_dtypes)
  monkeypatch.setattr(K.PFQKV, "render", bad)
  r = PF.selftest(G, m, (40 if exp == "act" else 24,), (128,), force=True)   # shapes no other test runs: the device caches programs by spec
  assert not r["ok"] and "fz_pfqkv" in r["why"], r

def test_selftest_falls_back_to_vector_evictions(tmp_path, monkeypatch):
  """A PFMLP that does not compile with ScalarE PSUM evictions (injected): the self-test settles on fallback level 1
  (FZ_NEURON_PFK_EVICT=vector), gate() keeps v2 on with that knob, and the forward's PFMLP is the vector one."""
  from fusezap.codegen.nki import NKIUnsupported
  monkeypatch.setenv("FZ_NEURON_PFK_EVICT", "act")   # level 0: ScalarE evictions (the default is vector)
  m, _ = _model(tmp_path, seed=10, layers=1)
  orig = K.PFMLP.render
  def picky(self):
    if self.evict == "act": raise NKIUnsupported("injected: no ScalarE PSUM copy")
    return orig(self)
  monkeypatch.setattr(K.PFMLP, "render", picky)
  r = PF.selftest(type("G", (), {"device": "NKISHIM", "rank": 0})(), m, (48,), (128,), force=True)
  assert r["ok"] and r["level"] == 1 and r["knobs"] == {"evict": "vector"} and "injected" in r["tried"][0]["why"], r
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v2")
  monkeypatch.setattr(PF, "selftest", lambda g, model, rows, ctxs, log=None: {**r, "level": 1})
  class Eng:
    prefill_sizes, tp, model = [48], None, m
    cache = type("C", (), {"ctx_buckets": lambda self, pf=False: [128]})()
  line = PF.gate(Eng())
  assert PF.enabled() and PF._knobs()["evict"] == "vector" and "fallback level 1" in line, line
  assert PF.specs(48, 256, 28, 4, 128, 384, 128, 1e-6, 0.1, 0, True, True, False)["mlp"].evict == "vector"

def test_engine_gate_drops_big_sizes_when_v2_is_off(tmp_path, monkeypatch):
  """LLMEngine on a NEURON tier runs the gate before fixing its prefill sizes: v2 refused by the self-test (injected) keeps
  only the smallest FZ_NEURON_PREFILL_SIZES size (a 512-row forward of the members would re-stream the weights per 128
  rows), max_prefill_tokens follows; v2 passing keeps both; v1 never runs the self-test."""
  from fusezap.serve.engine import LLMEngine
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "32")
  m, _ = _model(tmp_path, seed=12, layers=1)
  calls = []
  monkeypatch.setattr(PF, "selftest", lambda g, model, rows, ctxs, log=None: calls.append(rows) or {"ok": False, "why": "injected", "level": None})
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v2")
  e = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
  assert calls == [(32, 64)] and e.prefill_sizes == [32] and e.cfg.max_prefill_tokens == 32 and not PF.enabled()
  PF._STATE.update(ok=None, why="", knobs=None)
  monkeypatch.setattr(PF, "selftest", lambda g, model, rows, ctxs, log=None: calls.append(rows) or {"ok": True, "why": "", "level": 0})
  e = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
  assert e.prefill_sizes == [32, 64] and e.cfg.max_prefill_tokens == 64 and PF.enabled()
  PF._STATE.update(ok=None, why="", knobs=None)
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v1")
  n = len(calls)
  e = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
  assert len(calls) == n and e.prefill_sizes == [32]

def test_switches(monkeypatch):
  assert PF.kernel_mode() == "v2" and PF.enabled()   # default since I77 run 1
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v1"); assert not PF.enabled()
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v3")
  with pytest.raises(ValueError): PF.kernel_mode()
  monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", "v2")
  assert PF.enabled()
  PF.set_mode(None, "v1"); assert not PF.enabled()
  PF.set_mode(None, None); assert PF.enabled()
  assert PF.head_mode(True) == "split" and PF.head_mode(False) == "off"
  monkeypatch.setenv("FZ_NEURON_PFK_HEAD", "wide"); assert PF.head_mode(True) == "wide"
  assert K.pfk_exp() == "exact" and K.pfk_evict() == "vector"
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "fast")
  with pytest.raises(ValueError): K.pfk_exp()
  monkeypatch.setenv("FZ_NEURON_PFK_WN", "1000")
  with pytest.raises(ValueError): K.pfk_wn()

def test_prefill_sizes_env_and_prewarm_plan(monkeypatch):
  """FZ_NEURON_PREFILL_SIZES on a NEURON tier: the prefill forwards' sizes below max_prefill_tokens; the prewarm plan leaves a
  bigger size's prefixes that only a smaller size's steps would reach."""
  from fusezap.serve.engine import prefill_sizes
  from fusezap.serve import neuron_prewarm as PW
  monkeypatch.delenv("FZ_NEURON_PREFILL_SIZES", raising=False)
  assert prefill_sizes("NKISHIM", 128) == [128]
  assert PF.prefill_sizes(128, "NEURON") == PF.prefill_sizes(128, "NEURON:1") == [128, 512] and PF.prefill_sizes(128, "NKISHIM") is None   # I78: the chip's default
  assert prefill_sizes("NEURON", 512) == [128, 512]
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "128"); assert PF.prefill_sizes(128, "NEURON") == [128]
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "128,512")
  assert prefill_sizes("NKISHIM", 512) == [128, 512] and prefill_sizes("NKISHIM", 128) == [128] and prefill_sizes("CPU", 256) == [256]
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "100")
  with pytest.raises(ValueError): prefill_sizes("NKISHIM", 512)
  class F:
    def __init__(self, n): self.N = n
  eng = type("E", (), {"_decodes": {4: F(4)}, "_prefills": {128: F(128), 512: F(512)},
                       "cache": type("C", (), {"ctx_buckets": lambda self, pf=False: [128, 256, 512, 1024] if pf else [1024]})()})()
  got = [(f.N, c) for f, c in PW.plan(eng)]
  assert got == [(4, 1024), (128, 128), (128, 256), (128, 512), (128, 1024), (512, 256), (512, 512), (512, 1024)]

def test_engine_v2_prefill_sizes_tokens_equal_members(tmp_path, monkeypatch):
  """LLMEngine with prefill forwards of 32 and 64 rows (FZ_NEURON_PREFILL_SIZES=32 below max_prefill_tokens 64; v1 keeps 32
  only: bigger sizes need the prefill kernels): v2's greedy tokens equal the members' for prompts that fill one or both
  sizes, several per step."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "32")
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "exact")
  toks = {}
  for mode in ("v1", "v2"):
    monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", mode)
    m, _ = _model(tmp_path, seed=5)
    e = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=64, decode_beam=0)
    assert e.prefill_sizes == ([32, 64] if mode == "v2" else [32])   # bigger sizes only on the prefill kernels
    ps = [[int(t) for t in np.random.default_rng(20 + i).integers(0, 199, n)] for i, n in enumerate((7, 30, 50, 12))]
    toks[mode] = e.generate(ps, SamplingParams(max_tokens=6, ignore_eos=True))
  assert toks["v1"] == toks["v2"] and len({t for o in toks["v1"] for t in o}) > 3

@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_tp2_prefill_v2_bitwise_vs_members(tmp_path, monkeypatch, layout):
  """TP=2 (rank 1 a worker process), the fused prefill (collectives inside one NEFF per rank and chunk): v2's layers and its
  split lm_head (each rank half the vocabulary, one zero-padded all-reduce) give the members' prefill and decode logits bit
  for bit; the prefill NEFF holds the PF kernels and 2 collectives per layer + 1 for the head."""
  from test_tp import tiny_model_dir
  from test_neuron_tp import _pair, _setenv, _tp_env_clear, _collectives
  from fusezap.serve import dist, neuron_tp as NT
  from fusezap.codegen import nki_kernels as nk
  _tp_env_clear(monkeypatch)
  for k in ("FZ_NEURON_TP_PREFILL", "FZ_NEURON_TP_PREFILL_FUSE"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "exact")   # the worker inherits it
  here = os.path.dirname(os.path.abspath(__file__))
  g = dist.launch(2, TIER, timeout=120, env={"PYTHONPATH": os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)})
  L = 2
  try:
    assert NT._comm(g).prefill_cc, g.ar_info
    path = tiny_model_dir(str(tmp_path / "p"), seed=11, cfg={**LAYOUTS[layout][0], "num_hidden_layers": L}, bias=LAYOUTS[layout][1])
    runs = {}
    for mode in ("v1", "v2", "v2big"):   # v2big: a "big" chunk (FZ_NEURON_PFK_BIG_ROWS=16) grouped 1 layer per NEFF (FZ_NEURON_PFK_FUSE_BIG)
      env = {"FZ_NEURON_PREFILL_KERNEL": mode[:2], "FZ_NEURON_LAYER": "v1", "FZ_NEURON_PFK_BIG_ROWS": "16" if mode == "v2big" else None,
             "FZ_NEURON_PFK_FUSE_BIG": "1" if mode == "v2big" else None}
      g.post("call", "test_neuron_tp:_setenv", env); g.barrier("env")
      _setenv(g, env)
      _, m = _pair(g, path, "bfloat16")
      _two_chunks(m)
      out = _two_chunks(m)
      specs = [it.si.ast[0].spec for (N, R), jf in m._neuron_steps.jits.items() if N != R and jf.captured is not None
               for it in jf.captured.items if it.si.ast and getattr(it.si.ast[0], "spec", None) is not None]
      runs[mode] = (out, specs)
    for v in ("v2", "v2big"):
      for a, b in zip(runs["v1"][0], runs[v][0]): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
    big = [s for s in runs["v2big"][1] if isinstance(s, nk.FusedSpec)]
    assert len(big) == L and sorted(sum(_collectives(p[0]) for p in s.parts) for s in big) == [2, 3], [s.name for s in big]   # a group per layer
    fused = [s for s in runs["v2"][1] if isinstance(s, nk.FusedSpec)]
    assert fused and all(len([s for s in runs["v2"][1] if isinstance(s, nk.FusedSpec)]) for _ in [0])
    for s in fused:
      names = [p[0] for p in s.parts]
      assert sum(_collectives(n) for n in names) == 2 * L + 1, names   # + the head's gather
      assert any(n.startswith("fz_pfmlp") for n in names) and any(n.startswith("fz_pflin") and "_w" in n for n in names), names
  finally:
    _setenv(g, {"FZ_NEURON_PREFILL_KERNEL": None, "FZ_NEURON_LAYER": None, "FZ_NEURON_PFK_BIG_ROWS": None, "FZ_NEURON_PFK_FUSE_BIG": None})
    g.close()

def test_big_fuse(monkeypatch):
  from fusezap.device.neuron import FUSE_ALL
  assert PF.big_fuse(128, FUSE_ALL) == FUSE_ALL and PF.big_fuse(512, FUSE_ALL) == 7 and PF.big_fuse(512, 4) == 4 and PF.big_fuse(512, 0) == 0
  monkeypatch.setenv("FZ_NEURON_PFK_FUSE_BIG", "all"); assert PF.big_fuse(512, FUSE_ALL) == FUSE_ALL
  monkeypatch.setenv("FZ_NEURON_PFK_BIG_ROWS", "512"); monkeypatch.setenv("FZ_NEURON_PFK_FUSE_BIG", "2"); assert PF.big_fuse(512, FUSE_ALL) == FUSE_ALL

# ---------------------------------------------------------------- 4. the chip job's scripts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def test_prefill_bench_in_process(tmp_path, monkeypatch):
  """scripts/nki_n6_serve.py prefill_bench: every prefill forward of the engine at each prewarmed prefix (a 64-row forward
  above a 32-row one skips the prefix only the smaller one reaches), timed, its first logits row hashed; v1 (the smallest
  size only) and v2 exact give the same hashes."""
  import sys
  sys.path.insert(0, os.path.join(ROOT, "scripts"))
  import nki_n6_serve as S6
  from fusezap.serve.engine import LLMEngine
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "32")
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "32,64")
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "exact")
  out = {}
  for mode in ("v1", "v2"):
    monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", mode)
    m, _ = _model(tmp_path, seed=8, layers=1)
    e = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
    out[mode] = S6.prefill_bench(e, 1)
  assert sorted(out["v2"]) == ["32@128", "32@32", "32@64", "64@128", "64@64"], sorted(out["v2"])
  assert sorted(out["v1"]) == ["32@128", "32@32", "32@64"], sorted(out["v1"])   # v1: the smallest size only
  for k, x in out["v2"].items(): assert x["wall_ms_min"] > 0 and x["tokens"] == min(x["rows"], x["ctx"])
  for k, x in out["v1"].items(): assert out["v2"][k]["logits_sha"] == x["logits_sha"] and out["v2"][k]["top"] == x["top"], k

def test_pfk_job_mode_and_summary(tmp_path):
  """scripts/nki_n8_prefill_inf2.sh --pfk: bash -n, its own results dir and budget, the new path first (the cold compile the
  served phases reuse), old with the code defaults, the single-core bench + neuron-profile, served with and without the I76
  gate (pfk_http, not fz7b_http), the exact path last; scripts/nki_n8_pfk_summary.py's bar on synthetic result files."""
  import subprocess, sys
  sh = os.path.join(ROOT, "scripts", "nki_n8_prefill_inf2.sh")
  assert subprocess.run(["bash", "-n", sh]).returncode == 0
  src = open(sh).read()
  blk = src[src.index('if [ "$PFK" = 1 ]; then'):src.index("\nfi\n", src.index('if [ "$PFK" = 1 ]; then'))]
  assert "results/nki_n8_pfk" in src and "FZ_PF_BUDGET_MIN:-66" in src and "--max-hours 1.25 --max-usd 2.9" in src
  order = ["tp_phase fz_pfk_new", "tp_phase fz_pfk_old", "phase pfk_bench", "tp_phase fzhttp_pfk 780", "tp_phase fzhttp_pfk_gate0", "tp_phase fz_pfk_exact"]
  assert [blk.index(x) for x in order] == sorted(blk.index(x) for x in order)
  assert "FZ_NEURON_PREFILL_KERNEL=v2 FZ_NEURON_PREFILL_SIZES=128,512" in blk and "--max-prefill-tokens 512" in blk and "FZ_ADMIT_WAIT_MS=0" in blk
  assert "fz7b_http" not in blk and "neuron-profile" in src and "--prefill-bench 4" in blk and "--ttft-lens 40,128,512" in blk
  assert (1.25 + 0.17) * 1.9904 <= 2.9   # the launch guard
  def bench(rows_ctx: dict, sha: str = "a") -> dict:
    return {f"{r}@{c}": {"rows": r, "ctx": c, "tokens": min(r, c), "wall_ms_min": ms, "top": 5, "logits_sha": sha} for (r, c), ms in rows_ctx.items()}
  old = {"prefill_bench": bench({(128, 128): 86.0, (128, 512): 95.0}), "c1": {"tpot_ms_median": 39.1}, "c4": {"tpot_ms_median": 38.7},
         "ttft_by_len": {"128": {"c1_p50": 88, "c1_p99": 89, "c4_p50": 176, "c4_p99": 350, "c1_ids": [1, 2]}}}
  new = {"prefill_bench": bench({(128, 128): 50.0, (128, 512): 55.0, (512, 512): 105.0}, "b"), "c1": {"tpot_ms_median": 39.2}, "c4": {"tpot_ms_median": 38.8},
         "ttft_by_len": {"128": {"c1_p50": 52, "c1_p99": 53, "c4_p50": 60, "c4_p99": 110, "c1_ids": [1, 2]}},
         "greedy": {"exact": 23, "prompts": 24, "match_tokens": 1460, "ref_tokens": 1486}}
  exact = {"prefill_bench": bench({(128, 128): 60.0, (128, 512): 66.0})}
  for k, v in (("old", old), ("new", new), ("exact", exact)):
    with open(tmp_path / f"serve_pfk_{k}.json", "w") as f: json_dump(v, f)
  run = lambda: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_n8_pfk_summary.py"), str(tmp_path)], capture_output=True, text=True)
  r = run()
  assert r.returncode == 0 and "perf bar: PROMOTE" in r.stdout, r.stdout + r.stderr
  new["prefill_bench"]["128@128"]["wall_ms_min"] = 70.0
  with open(tmp_path / "serve_pfk_new.json", "w") as f: json_dump(new, f)
  assert "DO NOT PROMOTE" in run().stdout
  new["prefill_bench"]["128@128"]["wall_ms_min"] = 50.0
  exact["prefill_bench"]["128@512"]["logits_sha"] = "z"   # the exact path's bits differ from old's somewhere: not promoted
  for k, v in (("new", new), ("exact", exact)):
    with open(tmp_path / f"serve_pfk_{k}.json", "w") as f: json_dump(v, f)
  assert "DO NOT PROMOTE" in run().stdout

def test_pfk2_job_mode_and_summary(tmp_path):
  """scripts/nki_n8_prefill_inf2.sh --pfk2: bash -n, its own results dir, budget and launch guard (< 1.4 h wall with the
  self-stop grace), the cold in-process phase first (128,512 with the chunk check: its NEFFs are the served phases'),
  the fixed single-core bench, served main's defaults then 128,512 (Invarix profiles incl. p200, p512 at c1), the members'
  chunk check last; scripts/nki_n8_pfk2_summary.py's recommendation on synthetic result files."""
  import csv, subprocess, sys
  sh = os.path.join(ROOT, "scripts", "nki_n8_prefill_inf2.sh")
  assert subprocess.run(["bash", "-n", sh]).returncode == 0
  src = open(sh).read()
  start = src.index('if [ "$PFK2" = 1 ]; then')
  blk = src[start:src.index("\nfi\n", start)]
  assert "results/nki_n8_pfk2" in src and "FZ_PF_BUDGET_MIN:-60" in src and "--max-hours 1.15 --max-usd 2.65" in src and "--pfk2\"" in src
  assert (1.15 + 0.17) * 1.9904 <= 2.65 and 1.15 + 10 / 60 < 1.4   # the launch guard; the self-stop (max hours + 10 min grace)
  order = ["tp_phase fz_pfk2_b", "phase pfk_bench", "tp_phase fzhttp_pfk2_a", "tp_phase fzhttp_pfk2_b", "tp_phase fz_pfk2_v1", "nki_n8_pfk2_summary.py"]
  assert [blk.index(x) for x in order] == sorted(blk.index(x) for x in order)
  assert "FZ_NEURON_PREFILL_SIZES=128,512" in blk and "--max-prefill-tokens 512" in blk and "--chunk-check 200,400,512" in blk
  assert "pfk2_http pfk2_a 128\n" in blk and "pfk2_http pfk2_b 512 $SIZES_ENV" in blk and "FZ_NEURON_PREFILL_KERNEL=v1" in blk
  assert '"p200", "input_tokens": 200, "output_tokens": 56' in blk and '"p512", "input_tokens": 512' in blk and "--concurrency 1 " in blk
  assert "neuron-explorer" in src and "v2exact:512:512" in blk and "--prefill-bench 3" in blk and "--ttft-lens 40,128,200,512" in blk
  def write(lab: str, rows: list, sub: str = "perf") -> None:
    d = tmp_path / f"fz_tp2_bf16_7b_pfk2_{lab}_{sub}"
    d.mkdir(exist_ok=True)
    with open(d / "performance.csv", "w", newline="") as f:
      w = csv.DictWriter(f, ["workload", "concurrency", "errors", "output_tok_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms"])
      w.writeheader()
      for prof, c, p50, p99, tpot, tok in rows:
        w.writerow({"workload": prof, "concurrency": c, "errors": 0, "output_tok_s": tok, "ttft_p50_ms": p50, "ttft_p99_ms": p99, "tpot_p50_ms": tpot})
  A = [("short", 1, 52, 54, 39.0, 25.6), ("short", 4, 125, 205, 39.2, 100.0), ("short40", 1, 51, 53, 39.0, 25.7), ("short40", 4, 100, 110, 39.1, 100.5),
       ("p200", 1, 104, 106, 39.0, 24.0), ("p200", 4, 210, 300, 39.3, 95.0)]
  Bp = [("short", 1, 52, 55, 39.1, 25.6), ("short", 4, 170, 178, 39.3, 99.8), ("short40", 1, 51, 53, 39.0, 25.7), ("short40", 4, 100, 110, 39.1, 100.5),
        ("p200", 1, 105, 107, 39.1, 24.0), ("p200", 4, 200, 260, 39.3, 95.5)]
  write("a", A); write("b", Bp); write("a", [("p512", 1, 230, 232, 39.0, 22.0)], "long_perf"); write("b", [("p512", 1, 176, 178, 39.0, 23.0)], "long_perf")
  cc = lambda sha: {L: {"auto": {"sizes": [128, 128] if L == "200" else [512], "ttft_ms": [101, 100], "logits_sha": sha, "stable": True},
                        "small": {"sizes": [128, 128], "ttft_ms": [101, 100], "logits_sha": sha, "stable": True},
                        "large": {"sizes": [512], "ttft_ms": [170, 170], "logits_sha": sha, "stable": True}} for L in ("200", "512")}
  b = {"prefill_bench": {"128@256": {"rows": 128, "ctx": 256, "tokens": 128, "wall_ms_min": 53.7}, "512@512": {"rows": 512, "ctx": 512, "tokens": 512, "wall_ms_min": 168.2}},
       "prefill_cost": {"128": 50.0, "512": 172.0}, "chunk_check": cc("s1"), "greedy": {"exact": 23, "prompts": 24}}
  v1 = {"chunk_check": {L: {"auto": {"sizes": [128] * (2 if L == "200" else 4), "ttft_ms": [100], "logits_sha": "s1"}} for L in ("200", "512")}}
  for k, v in (("serve_pfk2_b.json", b), ("serve_pfk2_v1.json", v1), ("bf16_compare_b.json", {"verdict": {"top-1": True, "greedy": True}})):
    with open(tmp_path / k, "w") as f: json_dump(v, f)
  run = lambda: subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "nki_n8_pfk2_summary.py"), str(tmp_path), "--json", str(tmp_path / "s.json")],
                               capture_output=True, text=True)
  r = run()
  assert r.returncode == 0 and "default: RECOMMEND" in r.stdout and "FZ_NEURON_PREFILL_COST_MS=128:54,512:168" in r.stdout, r.stdout + r.stderr
  assert "bits equal across b/auto, b/small, b/large, v1: True" in r.stdout
  v1["chunk_check"]["512"]["auto"]["logits_sha"] = "s2"   # the 512-row forward's bits differ from the members'
  with open(tmp_path / "serve_pfk2_v1.json", "w") as f: json_dump(v1, f)
  assert "DO NOT RECOMMEND" in run().stdout
  v1["chunk_check"]["512"]["auto"]["logits_sha"] = "s1"
  with open(tmp_path / "serve_pfk2_v1.json", "w") as f: json_dump(v1, f)
  write("b", [x if x[0] != "p200" or x[1] != 1 else ("p200", 1, 167, 169, 39.1, 23.0) for x in Bp])   # a 200-token prompt on one 512-row forward
  out = run().stdout
  assert "DO NOT RECOMMEND" in out and "p200 c1 TTFT p50 <= a + 5: False" in out

def test_pfk_bench_applies_selftest_fallback(tmp_path, monkeypatch):
  """scripts/nki_pfk_bench.py runs the serving path's self-test before a v2 config's capture and applies its fallback knobs
  (I77 run 1 on inf2: ScalarE evictions do not compile, the bench captured them anyway and failed): an injected level-1
  fallback (VectorE evictions) over FZ_NEURON_PFK_EVICT=act ends up in the captured PFMLP; a self-test failing at every
  level makes the config an error, not a member-kernel run labelled v2."""
  import json, sys
  from fusezap.device import Device
  sys.path.insert(0, os.path.join(ROOT, "scripts"))
  import nki_pfk_bench as B
  monkeypatch.setattr(Device, "DEFAULT", "NKISHIM")
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_DTYPE", "FZ_NEURON_BF16_HEAD", "FZ_NEURON_PLAN", "FZ_NEURON_PFK_EXP",
            "FZ_NEURON_PFK_ATTN_HB", "FZ_NEURON_PFK_SILU", "FZ_NEURON_PFK_EXPF"):
    monkeypatch.setenv(k, "")   # the bench sets these: restored after the test
  monkeypatch.setenv("FZ_NEURON_PFK_EVICT", "act")
  seen = []
  def fake(g, model, rows, ctxs, log=None, force=False):
    seen.append((rows, ctxs))
    return {"ok": True, "level": 1, "knobs": {"evict": "vector"}, "why": "fallback level 1: injected", "seconds": 0.0}
  monkeypatch.setattr(PF, "selftest", fake)
  out = tmp_path / "b"
  monkeypatch.setattr(sys, "argv", ["nki_pfk_bench.py", "--hidden", "256", "--inter", "384", "--configs", "v1:32:64,v2exact:32:64", "--out", str(out)])
  B.main()
  recs = {r["config"]: r for r in json.load(open(out / "pfk_bench.json"))["records"]}
  r = recs["v2exact:32:64"]
  assert "error" not in r, r.get("error")
  assert seen == [((32,), (64,))] and r["selftest"]["level"] == 1 and r["knobs"]["evict"] == "vector" and r["knobs"]["exp"] == "exact"
  assert r["logits_bitwise_vs_v1"] is True and recs["v1:32:64"]["knobs"] is None
  assert PF._STATE["knobs"] is None and PF._STATE["ok"] is None   # reset for the next config / test
  monkeypatch.setattr(PF, "selftest", lambda g, model, rows, ctxs, log=None, force=False: {"ok": False, "why": "injected compile error", "level": None})
  monkeypatch.setattr(sys, "argv", ["nki_pfk_bench.py", "--hidden", "256", "--inter", "384", "--configs", "v2exact:32:64", "--out", str(out)])
  B.main()
  r = json.load(open(out / "pfk_bench.json"))["records"][0]
  assert "self-test failed" in r["error"] and "injected compile error" in r["error"]

# ---------------------------------------------------------------- 5. I69: the exact exp in fewer, wider instructions

I69_ON = {"attn_hb": 2048, "silu": 2048, "expf": True}
I69_OFF = {"attn_hb": 0, "silu": 512, "expf": False}

@pytest.mark.parametrize("base2", [True, False])
def test_exact_exp_fused_bitwise(base2):
  """nki.exact_exp's fused form (the clamp in one two-op tensor_scalar, each power's exponent bits in one, straight to int32)
  gives the unfused form's bits over the whole float32 range: overflow, underflow into denormals and zero, the clamp
  bounds, +-inf, NaN, tiny and huge arguments."""
  from fusezap.codegen import nki
  from fusezap.dtype import dtypes
  rng = np.random.default_rng(1 if base2 else 2)
  x = np.concatenate([rng.uniform(-300, 300, 3000), rng.uniform(-30, 30, 3000), rng.standard_normal(2000) * 1e-3,
                      [0.0, -0.0, 1e-38, -1e-38, 1e-45, 252.0, -252.0, 254.0, 253.9, -149.5, -126.0, 127.99, 88.7, -87.3, -103.97,
                       175.0, -175.0, 177.0, 1e30, -1e30, np.inf, -np.inf, np.nan]]).astype(np.float32)
  x = np.resize(x, (128, -(-x.size // 128))).astype(np.float32)
  outs = []
  for fused in (False, True):
    L = ["def k(data0, data1):", f"  x = nl.ndarray({x.shape}, dtype=nl.float32, buffer=nl.sbuf)",
         f"  nisa.dma_copy(dst=x, src=data1.ap(pattern=[[{x.shape[1]}, {x.shape[0]}], [1, {x.shape[1]}]], offset=0))",
         f"  d = nl.ndarray({x.shape}, dtype=nl.float32, buffer=nl.sbuf)"]
    k = iter(range(100))
    def new(dt):
      nm = f"t{next(k)}"
      if dt is not None: L.append(f"  {nm} = nl.ndarray({x.shape}, dtype=nl.{nki.nl_dtype(dt)}, buffer=nl.sbuf)")
      return nm
    nki.exact_exp(lambda ln: L.append("  " + ln), new, "d", "x", base2=base2, fused=fused)
    L += [f"  nisa.dma_copy(dst=data0.ap(pattern=[[{x.shape[1]}, {x.shape[0]}], [1, {x.shape[1]}]], offset=0), src=d)", "  return data0"]
    src = "\n".join(L)
    assert ("op1=nl.minimum" in src) == fused and src.count("nisa.") < (60 if fused else 80)
    prog = nki.NKIProgram("k", HEADER + src + "\n", 2, (0,), {0: dtypes.float32, 1: dtypes.float32})
    o = np.zeros(x.size, np.float32)
    S.execute(prog, [o, x.reshape(-1).copy()])
    outs.append((o, src.count("nisa.")))
  (a, na), (b, nb) = outs
  np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
  assert na - nb == 5, (na, nb)   # the clamp 2 -> 1, each power 3 -> 1
  fin = np.isfinite(x.reshape(-1)) & (np.abs(x.reshape(-1)) < 80)
  with np.errstate(over="ignore"): want = np.exp2(x.reshape(-1).astype(np.float64)) if base2 else np.exp(x.reshape(-1).astype(np.float64))
  assert np.max(np.abs(a[fin] - want[fin]) / want[fin]) < 4e-7   # still ~1 ulp

@pytest.mark.parametrize("N,P,layout", [(128, 128, "qwen7b"), (128, 640, "qwen7b"), (160, 256, "qwen7b"), (64, 1024, "llama8b"), (16, 48, "qwen7b")])
def test_i69_attention_bitwise_vs_per_head(N, P, layout):
  """PFAttn with its exps batched over q heads (FZ_NEURON_PFK_ATTN_HB) and / or the fused exact exp (FZ_NEURON_PFK_EXPF):
  main's per-head kernel bit for bit, over one or several key blocks (640, 1024 slots), a row tail (160 rows), group tails
  (14 heads in groups of 4 / 8), a narrow prefix, a fully masked row and rows whose keys start mid-block."""
  rng = np.random.default_rng(N + P)
  Hq, Hkv, hd = (14, 2, 128) if layout == "qwen7b" else (16, 4, 128)
  X, pool = _r(rng, (N, (Hq + 2 * Hkv + 1) * hd)), _r(rng, (P, 2 * Hkv * hd))
  meta = _meta(N, P, 8, start=P - N)
  meta[8 + N + 3] = -1                       # row 3: hi < lo, no key visible
  meta[8 + 5] = P // 3; meta[8 + 7] = P - N  # rows 5, 7: a later first key
  outs = {}
  for hb, xf in ((0, False), (2048, False), (2048, True), (0, True), (256, True), (512, False)):
    sp = K.PFAttn(N, Hq, Hkv, hd, P, 0.088, 8, "exact", hb, xf)
    o = np.zeros(N * Hq * hd, np.float32)
    S.execute(sp.render(), [o, X.reshape(-1), pool.reshape(-1), meta])
    outs[(hb, xf)] = (o, sp)
  ref, base = outs[(0, False)]
  assert base.name == f"fz_pfattn_{N}_{Hq}_{Hkv}_{hd}_{P}" and base.heads_per_group == 0
  for (hb, xf), (o, sp) in outs.items():
    np.testing.assert_array_equal(o.view(np.int32), ref.view(np.int32), err_msg=sp.name)
    assert (f"_hb{sp.heads_per_group}" in sp.name) == bool(hb) and sp.name.endswith("_xf") == xf, sp.name
  assert outs[(2048, True)][1].heads_per_group == min(Hq, 2048 // min(512, P))
  assert K.PFAttn(N, Hq, Hkv, hd, P, 0.088, 8, "act", 2048, True).name == f"fz_pfattn_{N}_{Hq}_{Hkv}_{hd}_{P}_act"   # act: unchanged

@pytest.mark.parametrize("N,I", [(128, 1300), (160, 2600), (64, 384), (512, 1100)])
def test_i69_mlp_bitwise_vs_slices(N, I):
  """PFMLP with one silu chain per gate / up block (FZ_NEURON_PFK_SILU) and / or the fused exact exp: main's PFMLP (a chain
  per 512-column slice) bit for bit, incl. column tails, row tails and a 512-row chunk (its blocks are 512 columns: no change)."""
  rng = np.random.default_rng(N + I)
  D = 256
  x, o, ln2 = _r(rng, (N, D)), _r(rng, (N, D)), 1 + _r(rng, D, 0.1)
  wgu, wd = f32_to_bf16(_r(rng, (D, 2 * I), 0.3)), f32_to_bf16(_r(rng, (I, D), 0.05))
  outs = {}
  for silu, xf in ((512, False), (2048, False), (1024, True), (2048, True), (512, True)):
    sp = K.PFMLP(N, D, I, 1e-6, False, "exact", 8, 4096, evict="vector", silu=silu, expf=xf)
    arrs = [np.zeros(N * D, np.float32), np.zeros(N * D, np.float32), x.reshape(-1), o.reshape(-1), ln2, wgu.reshape(-1), wd.reshape(-1)]
    S.execute(sp.render(), arrs)
    outs[(silu, xf)] = (arrs[0], arrs[1], sp)
  rx, rd, base = outs[(512, False)]
  for k, (ox, od, sp) in outs.items():
    np.testing.assert_array_equal(ox.view(np.int32), rx.view(np.int32), err_msg=sp.name)
    np.testing.assert_array_equal(od.view(np.int32), rd.view(np.int32), err_msg=sp.name)
  sp = outs[(2048, True)][2]
  assert sp.silu_cols == min(2048, sp.gu_cols) and sp.name.endswith(("_s2048_xf", "_s1024_xf", "_xf"))
  assert ("_s" in sp.name) == (sp.silu_cols != 512) and "_s" not in base.name

def test_i69_switches_and_fallbacks(monkeypatch):
  """FZ_NEURON_PFK_ATTN_HB / _SILU / _EXPF: defaults (hb and xf on, wide silu off: slower on inf2), main's values, bad values;
  the knobs reach the specs;
  the self-test's fallback levels 2 and 3 turn them off (main's kernels) should they not compile on a chip; i69_env."""
  assert (K.pfk_attn_hb(), K.pfk_silu(), K.pfk_expf()) == (2048, 512, True)
  kn = PF._knobs()
  assert {k: kn[k] for k in I69_ON} == {**I69_ON, "silu": 512}
  sp = PF.specs(128, 3584, 14, 2, 128, 9472, 128, 1e-6, 0.088, 64, True, True, False)
  assert sp["attn"].name.endswith("_hb14_xf") and sp["mlp"].name == "fz_pfmlp_128_3584_9472_b2048_4096_ev_xf", (sp["attn"].name, sp["mlp"].name)
  for k, v in K.i69_env([]).items(): monkeypatch.setenv(k, v)
  sp = PF.specs(128, 3584, 14, 2, 128, 9472, 128, 1e-6, 0.088, 64, True, True, False)
  assert sp["attn"].name == "fz_pfattn_128_14_2_128_128" and sp["mlp"].name == "fz_pfmlp_128_3584_9472_b2048_4096_ev"
  assert K.i69_env(["i69"]) == {"FZ_NEURON_PFK_ATTN_HB": "2048", "FZ_NEURON_PFK_SILU": "2048", "FZ_NEURON_PFK_EXPF": "1"}
  assert K.i69_env(["hb"])["FZ_NEURON_PFK_SILU"] == "512"
  with pytest.raises(ValueError): K.i69_env(["fast"])
  for k, v in (("FZ_NEURON_PFK_ATTN_HB", "-1"), ("FZ_NEURON_PFK_SILU", "700"), ("FZ_NEURON_PFK_EXPF", "2")):
    monkeypatch.setenv(k, v)
    with pytest.raises(ValueError): PF._knobs()
    monkeypatch.delenv(k)
  assert PF.FALLBACKS[1] == {"evict": "vector"} and all({k: f[k] for k in I69_OFF} == I69_OFF for f in PF.FALLBACKS[2:])
  PF.set_ok(None, True, "", PF.FALLBACKS[2])
  assert {k: PF._knobs()[k] for k in I69_OFF} == I69_OFF
  sp = PF.specs(128, 3584, 14, 2, 128, 9472, 128, 1e-6, 0.088, 64, True, True, False)
  assert "_hb" not in sp["attn"].name and "_s2048" not in sp["mlp"].name and "_xf" not in sp["mlp"].name

def test_i69_selftest_falls_back_to_main_kernels(tmp_path, monkeypatch):
  """A chip whose compiler refuses the I69 forms (injected: a PFAttn with batched heads fails to render): the self-test runs
  levels 0 and 1 (both I69), then settles on level 2, main's kernels with VectorE evictions."""
  from fusezap.codegen.nki import NKIUnsupported
  m, _ = _model(tmp_path, seed=13, layers=1)
  orig = K.PFAttn.render
  def picky(self):
    if self.heads_per_group: raise NKIUnsupported("injected: batched heads")
    return orig(self)
  monkeypatch.setattr(K.PFAttn, "render", picky)
  r = PF.selftest(type("G", (), {"device": "NKISHIM", "rank": 0})(), m, (56,), (112,), force=True)   # shapes no other test runs
  assert r["ok"] and r["level"] == 2 and r["knobs"]["attn_hb"] == 0 and "injected" in r["tried"][0]["why"], r

def test_i69_forward_bitwise_wide_mlp(tmp_path, monkeypatch):
  """The serving forward with an intermediate size above one 512-column slice (1300: the silu chain spans the gate / up block)
  and 14 / 2 heads: v2 with every I69 form (wide silu too) == v2 with main's kernels == the members, bit for bit, prefill
  (two chunks) and the decode after it."""
  monkeypatch.setenv("FZ_NEURON_PFK_EXP", "exact")
  cfg = {**Q7B, "intermediate_size": 1300}
  runs = {}
  for lab, mode, env in (("v1", "v1", {}), ("main", "v2", K.i69_env([])), ("i69", "v2", K.i69_env(["i69"]))):
    monkeypatch.setenv("FZ_NEURON_PREFILL_KERNEL", mode)
    for k, v in env.items(): monkeypatch.setenv(k, v)
    m, _ = _model(tmp_path, seed=14, cfg=cfg)
    _two_chunks(m)
    runs[lab] = (_two_chunks(m), _prefill_names(m))
  for lab in ("main", "i69"):
    for a, b in zip(runs["v1"][0], runs[lab][0]): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32), err_msg=lab)
  assert any("_hb" in n for n in runs["i69"][1]) and any(n.startswith("fz_pfmlp") and "_s" in n and n.endswith("_xf") for n in runs["i69"][1]), runs["i69"][1]
  assert not any("_hb" in n or n.endswith("_xf") for n in runs["main"][1]), runs["main"][1]

def test_i69_embed_bf16_upload_bitwise(tmp_path, monkeypatch):
  """FZ_NEURON_EMBED_BF16 (default on): with a bf16 embedding table the engine's prefill forwards upload the token rows as
  bf16 (half the bytes) and widen them inside the forward: every prefill forward's logits bits (scripts/nki_n6_serve.py
  prefill_bench, each prefix) and the greedy tokens equal the float32 upload's; decode forwards and a float32 table keep
  the float32 upload."""
  import sys
  sys.path.insert(0, os.path.join(ROOT, "scripts"))
  import nki_n6_serve as S6
  from fusezap.serve import neuron_kv as NK
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.weights import STArray
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "32")
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "32,64")
  seen = []
  orig = NK.upload
  monkeypatch.setattr(NK, "upload", lambda a, device, dtype: seen.append((a.shape, dtype)) or orig(a, device, dtype))
  out, toks = {}, {}
  for flag in ("0", "1"):
    monkeypatch.setenv("FZ_NEURON_EMBED_BF16", flag)
    m, _ = _model(tmp_path, seed=15, layers=1)
    assert not NK.embed_bf16(m)   # the tiny checkpoint's table is float32: never rounded to bf16
    tb = f32_to_bf16(np.asarray(m.embed_host, np.float32)).view(STArray)   # the same table as bf16 (a real checkpoint's)
    tb.st_dtype = "BF16"
    m.embed_host = tb
    assert NK.embed_bf16(m) == (flag == "1")
    e = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
    assert e.embed_bf16 == (flag == "1")
    seen.clear()
    out[flag] = S6.prefill_bench(e, 1)
    assert any(dt == dtypes.bfloat16 for _, dt in seen) == (flag == "1"), seen
    rids = [e.add_request(list(np.random.default_rng(i).integers(0, 199, n)), SamplingParams(max_tokens=6)) for i, n in ((1, 20), (2, 45))]
    while e.has_unfinished(): e.step()
    toks[flag] = [list(e.seqs[r].output) for r in rids]
  assert out["0"].keys() == out["1"].keys() and out["0"]
  for k in out["0"]: assert out["0"][k]["logits_sha"] == out["1"][k]["logits_sha"], k
  assert toks["0"] == toks["1"], toks

def test_i69_job_mode_and_summary(tmp_path):
  """scripts/nki_n8_prefill_inf2.sh --i69: bash -n, its own results dir, budget and launch guard (<= $2.50, the self-stop
  grace included), the phases in order (single-core bench, main, cand, the kernels-only fallback, the choice, served chosen
  with accuracy, served main), every phase on 128-row prefill forwards; scripts/nki_n8_i69_summary.py's choice (bitwise and
  faster, else main), the kernels-only env from the bench and the bar on synthetic result files."""
  import csv, subprocess, sys
  sh = os.path.join(ROOT, "scripts", "nki_n8_prefill_inf2.sh")
  assert subprocess.run(["bash", "-n", sh]).returncode == 0
  src = open(sh).read()
  start = src.index('if [ "$I69" = 1 ]; then')
  blk = src[start:src.index("\nfi\n", start)]
  assert "results/nki_n8_i69" in src and "FZ_PF_BUDGET_MIN:-52" in src and "--max-hours 1.05 --max-usd 2.5 --hf-token -- \"bash scripts/nki_n8_prefill_inf2.sh --i69\"" in src
  assert (1.05 + 0.17) * 1.996 <= 2.5   # the launch guard: max hours + the 10 min self-stop grace at inf2.8xlarge's $1.996/h
  order = ["phase pfk_bench", "i69_variant main", "i69_variant cand", "i69_variant kern", "tp_phase fzhttp_i69_chosen", "tp_phase fzhttp_i69_main", "nki_n8_i69_summary.py \"$OUT\" --json"]
  assert [blk.index(x) for x in order] == sorted(blk.index(x) for x in order)
  assert "FZ_NEURON_PFK_ATTN_HB=0 FZ_NEURON_PFK_SILU=512 FZ_NEURON_PFK_EXPF=0 FZ_NEURON_EMBED_BF16=0" in blk
  assert "FZ_NEURON_PFK_ATTN_HB=2048 FZ_NEURON_PFK_SILU=2048 FZ_NEURON_PFK_EXPF=1 FZ_NEURON_EMBED_BF16=1" in blk
  assert blk.count("FZ_NEURON_PREFILL_SIZES=128") >= 2 and "FZ_NEURON_PREFILL_SIZES=128 \"$@\"" in src and "--prefill-bench 5" in blk
  assert "fz7b_http tp2_bf16_7b_i69_chosen auto" in blk and "fz7b_http tp2_bf16_7b_i69_main none" in blk
  # the env words the job uses are the ones the code reads
  for k, v in K.i69_env(["i69"]).items(): assert f"{k}={v}" in blk
  for k, v in K.i69_env([]).items(): assert f"{k}={v}" in blk
  def bench(sha: str, w128: float, extra: dict | None = None) -> dict:
    pb = {f"128@{c}": {"rows": 128, "ctx": c, "tokens": 128, "wall_ms_min": w128 + (c - 128) / 50, "top": 5, "logits_sha": f"{sha}{c}"} for c in (128, 256, 512, 1024)}
    return {"prefill_bench": pb, "ttft_by_len": {"128": {"c1_p50": w128 + 1, "c4_p99": 4 * w128, "c1_ids": [1, 2], "c4_ids": [3, 4, 5, 6]}},
            "c1": {"tpot_ms_median": 38.6}, "greedy": {"rows": [{"id": "p0", "ids": [7, 8, 9]}]}, **(extra or {})}
  with open(tmp_path / "i69_variants.tsv", "w") as f:
    f.write("main\tFZ_NEURON_PFK_ATTN_HB=0 FZ_NEURON_EMBED_BF16=0\ncand\tFZ_NEURON_PFK_ATTN_HB=2048 FZ_NEURON_EMBED_BF16=1\n")
  for n, v in (("main", bench("a", 51.3)), ("cand", bench("a", 45.2))):
    with open(tmp_path / f"serve_i69_{n}.json", "w") as f: json_dump(v, f)
  os.makedirs(tmp_path / "pfk_bench")
  recs = [{"config": "v1:128:128", "path": "v1", "ctx": 128, "layer_device_ms": 2.43, "logits_bitwise_vs_v1": True},
          {"config": "v2exact:128:128", "path": "v2exact", "ctx": 128, "layer_device_ms": 1.41, "logits_bitwise_vs_v1": True},
          {"config": "v2exact+hb:128:128", "path": "v2exact+hb", "ctx": 128, "layer_device_ms": 1.30, "logits_bitwise_vs_v1": True},
          {"config": "v2exact+silu:128:128", "path": "v2exact+silu", "ctx": 128, "layer_device_ms": 1.44, "logits_bitwise_vs_v1": True},
          {"config": "v2exact+xf:128:128", "path": "v2exact+xf", "ctx": 128, "layer_device_ms": 1.36, "logits_bitwise_vs_v1": False}]
  with open(tmp_path / "pfk_bench" / "pfk_bench.json", "w") as f: json_dump({"records": recs}, f)
  def write(lab: str, rows: list, dec: str | None = None) -> None:
    d = tmp_path / f"fz_tp2_bf16_7b_i69_{lab}_perf"
    d.mkdir(exist_ok=True)
    with open(d / "performance.csv", "w", newline="") as f:
      w = csv.DictWriter(f, ["workload", "concurrency", "errors", "output_tok_s", "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms"])
      w.writeheader()
      for c, p50, p99, tpot, tok in rows: w.writerow({"workload": "short", "concurrency": c, "errors": 0, "output_tok_s": tok, "ttft_p50_ms": p50, "ttft_p99_ms": p99, "tpot_p50_ms": tpot})
    if dec:
      (tmp_path / f"fz_tp2_bf16_7b_i69_{lab}_acc").mkdir(exist_ok=True)
      with open(tmp_path / f"fz_tp2_bf16_7b_i69_{lab}_acc" / "report.json", "w") as f: json_dump({"decision": {"label": dec}}, f)
  write("chosen", [(1, 49.0, 50.1, 38.5, 25.9), (4, 130.0, 205.0, 39.2, 100.2)], "PROMOTE")
  write("main", [(1, 55.2, 56.4, 38.5, 25.8), (4, 145.0, 236.0, 39.3, 99.7)])
  script = os.path.join(ROOT, "scripts", "nki_n8_i69_summary.py")
  run = lambda *x: subprocess.run([sys.executable, script, str(tmp_path), *x], capture_output=True, text=True)
  r = run("--choose")
  assert r.returncode == 0 and r.stdout.split("\t")[0] == "cand" and "EMBED_BF16=1" in r.stdout, r.stdout + r.stderr
  assert run("--kern-env").stdout.strip() == "FZ_NEURON_PFK_ATTN_HB=2048 FZ_NEURON_PFK_SILU=512 FZ_NEURON_PFK_EXPF=0 FZ_NEURON_EMBED_BF16=0"
  r = run("--json", str(tmp_path / "s.json"))
  assert r.returncode == 0 and "RECOMMEND cand" in r.stdout and "DO NOT" not in r.stdout, r.stdout + r.stderr
  assert "saves 3.08 ms per 28 layers vs v2exact" in r.stdout and "costs 0.84 ms" in r.stdout   # +hb (1.41 - 1.30) x 28; +silu
  cand = bench("a", 45.2); cand["prefill_bench"]["128@512"]["logits_sha"] = "z"   # one forward's bits differ: main is chosen
  with open(tmp_path / "serve_i69_cand.json", "w") as f: json_dump(cand, f)
  assert run("--choose").stdout.startswith("main\t") and "forward 128@512" in run("--choose").stderr
  assert "DO NOT RECOMMEND main" in run().stdout
  cand = bench("a", 45.2); cand["greedy"]["rows"][0]["ids"] = [7, 8, 1]
  with open(tmp_path / "serve_i69_cand.json", "w") as f: json_dump(cand, f)
  assert run("--choose").stdout.startswith("main\t")
  with open(tmp_path / "serve_i69_cand.json", "w") as f: json_dump(bench("a", 52.0), f)   # bitwise but slower: main
  assert run("--choose").stdout.startswith("main\t")

def test_i69_served_job_mode_and_licence():
  """scripts/nki_n8_prefill_inf2.sh --i69-served: bash -n, its own results dir, 40 min budget and launch guard (<= $2.00 with
  the self-stop grace), no single-core bench, in process main then the new defaults (the code's, spelled out), the choice,
  served chosen with accuracy, served main perf; scripts/nki_n6b_lib.sh exports the chip-run licence when it exists and no
  licence is set (run 1's served phases failed without one), and the licence file is gitignored, never tracked."""
  import subprocess
  sh = os.path.join(ROOT, "scripts", "nki_n8_prefill_inf2.sh")
  assert subprocess.run(["bash", "-n", sh]).returncode == 0
  src = open(sh).read()
  start = src.index('if [ "$I69S" = 1 ]; then')
  blk = src[start:src.index("\nfi\n", start)]
  assert "results/nki_n8_i69s" in src and "FZ_PF_BUDGET_MIN:-40" in src
  assert "--max-hours 0.8 --max-usd 2.0 --hf-token -- \"bash scripts/nki_n8_prefill_inf2.sh --i69-served\"" in src
  assert (0.8 + 0.17) * 1.996 <= 2.0
  order = ["i69_variant main", "i69_variant new", "nki_n8_i69_summary.py \"$OUT\" --choose", "tp_phase fzhttp_i69_chosen", "tp_phase fzhttp_i69_main"]
  assert [blk.index(x) for x in order] == sorted(blk.index(x) for x in order) and "pfk_bench" not in blk
  new = "FZ_NEURON_PFK_ATTN_HB=2048 FZ_NEURON_PFK_SILU=512 FZ_NEURON_PFK_EXPF=1 FZ_NEURON_EMBED_BF16=1"
  assert new in blk and "fz7b_http tp2_bf16_7b_i69_chosen auto" in blk
  assert new.split() == [f"{k}={v}" for k, v in (("FZ_NEURON_PFK_ATTN_HB", K.pfk_attn_hb()), ("FZ_NEURON_PFK_SILU", K.pfk_silu()),
                                                 ("FZ_NEURON_PFK_EXPF", int(K.pfk_expf())), ("FZ_NEURON_EMBED_BF16", 1))]   # the code defaults
  lib = os.path.join(ROOT, "scripts", "nki_n6b_lib.sh")
  probe = 'cd "$1"; . scripts/nki_n6b_lib.sh; echo "LIC=${FUSEZAP_LICENSE_FILE:-}"'
  def run(d, env):
    r = subprocess.run(["bash", "-c", probe.replace("scripts/nki_n6b_lib.sh", lib), "x", str(d)], capture_output=True, text=True,
                       env={k: v for k, v in {**os.environ, **env}.items() if v is not None})
    return r.stdout.strip().splitlines()[-1]
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    env = {"FUSEZAP_LICENSE_FILE": None}
    assert run(d, env) == "LIC="
    os.makedirs(os.path.join(d, ".scratch", "licence"))
    open(os.path.join(d, ".scratch", "licence", "chip-runs.lic"), "w").write("x")
    got = run(d, env)
    assert got.startswith("LIC=") and os.path.samefile(got[4:], os.path.join(d, ".scratch", "licence", "chip-runs.lic")), got
    assert run(d, {"FUSEZAP_LICENSE_FILE": "/given.lic"}) == "LIC=/given.lic"
  r = subprocess.run(["git", "check-ignore", "-q", ".scratch/licence/chip-runs.lic"], cwd=ROOT)
  assert r.returncode == 0
  assert subprocess.run(["git", "ls-files", ".scratch"], cwd=ROOT, capture_output=True, text=True).stdout == ""

def json_dump(v, f):
  import json
  json.dump(v, f)

# ---------------------------------------------------------------- I69 follow-up: the prefill's token rows gathered on the device

@pytest.mark.parametrize("N,V,D", [(128, 1000, 384), (130, 333, 200), (5, 70, 3584)])
def test_embed_gather_kernel_exact(N, V, D):
  """EmbedGather: row n of the output is table row ids[n], widened bf16 -> float32 bit for bit (two row blocks at 130
  tokens; repeated ids; the first and last rows; padding id 0)."""
  rng = np.random.default_rng(N + V)
  tab = f32_to_bf16(_r(rng, (V, D)))
  ids = rng.integers(0, V, N); ids[0], ids[-1] = V - 1, 0; ids[N // 2] = ids[1]
  x = np.full(N * D, np.nan, np.float32)
  S.execute(K.EmbedGather(N, V, D).render(), [x, K.embed_offsets(ids, D).reshape(-1), tab.reshape(-1)])
  want = (tab[ids].astype(np.uint32) << 16).view(np.float32)
  np.testing.assert_array_equal(x.reshape(N, D).view(np.int32), want.view(np.int32))
  assert K.EmbedGather(N, V, D).sizes() == [N * D, N, V * D]
  with pytest.raises(Exception): K.EmbedGather(4, 2 ** 20, 4096).render()   # offsets past int32

def test_embed_dev_engine_bitwise(tmp_path, monkeypatch):
  """FZ_NEURON_EMBED_DEV=1 (default 0): with a bf16 table the prefill forwards take N int32 offsets and gather their rows
  on the device (EmbedGather inside the captured forward), so no (N, D) token rows are uploaded: every prefill forward's
  logits bits at each prefix (nki_n6_serve.py prefill_bench) and the greedy tokens equal the bf16 upload's (I69's
  default); the gather's check failing (injected) turns it off with a message and the upload stays."""
  import sys
  sys.path.insert(0, os.path.join(ROOT, "scripts"))
  import nki_n6_serve as S6
  from fusezap.serve import neuron_kv as NK
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.weights import STArray
  monkeypatch.setenv("FZ_NEURON_PREFILL_SIZES", "32")
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "32,64")
  seen = []
  orig = NK.upload
  monkeypatch.setattr(NK, "upload", lambda a, device, dtype: seen.append((a.shape, dtype)) or orig(a, device, dtype))
  out, toks, names = {}, {}, {}
  for flag in ("0", "1"):
    monkeypatch.setenv("FZ_NEURON_EMBED_DEV", flag)
    m, _ = _model(tmp_path, seed=16, layers=1)
    tb = f32_to_bf16(np.asarray(m.embed_host, np.float32)).view(STArray)   # a real checkpoint's bf16 table
    tb.st_dtype = "BF16"
    m.embed_host = tb
    assert NK.embed_bf16(m) and NK.embed_dev(m) == (flag == "1")
    e = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
    assert e.embed_dev == (flag == "1")
    seen.clear()
    out[flag] = S6.prefill_bench(e, 1)
    big = [s for s, dt in seen if dt == dtypes.bfloat16 and len(s) == 2 and s[0] > 1]   # (N, D) token rows uploaded
    assert bool(big) == (flag == "0") and any(dt == dtypes.int32 and s[0] == 1 for s, dt in seen) == (flag == "1"), seen
    rids = [e.add_request(list(np.random.default_rng(i).integers(0, 199, n)), SamplingParams(max_tokens=6)) for i, n in ((1, 20), (2, 45))]
    while e.has_unfinished(): e.step()
    toks[flag] = [list(e.seqs[r].output) for r in rids]
    names[flag] = set()
    for f in e._prefills.values():
      for j in [f.jit, *f.ctx_jits.values()]:
        for it in (j.captured.items if j.captured is not None else []):
          sp = getattr(it.si.ast[0], "spec", None) if it.si.ast else None
          if sp is not None: names[flag] |= {p[0] for p in sp.parts} if hasattr(sp, "parts") else {sp.name}
  assert out["0"].keys() == out["1"].keys() and out["0"]
  for k in out["0"]: assert out["0"][k]["logits_sha"] == out["1"][k]["logits_sha"], k
  assert toks["0"] == toks["1"], toks
  assert any(n.startswith("fz_embgather_32_") for n in names["1"]) and not any(n.startswith("fz_embgather") for n in names["0"]), names
  # the check failing (a gather that returns the wrong rows): off, uploads as before
  monkeypatch.setenv("FZ_NEURON_EMBED_DEV", "1")
  monkeypatch.setattr(K, "embed_offsets", lambda ids, D: (np.asarray(ids, np.int64).reshape(1, -1) * D + 1).astype(np.int32))
  m, _ = _model(tmp_path, seed=16, layers=1)
  tb = f32_to_bf16(np.asarray(m.embed_host, np.float32)).view(STArray); tb.st_dtype = "BF16"; m.embed_host = tb
  e = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=2, max_prefill_tokens=64, decode_beam=0)
  assert not e.embed_dev and e.embed_bf16
