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
            "FZ_NEURON_PREFILL_SIZES", "FZ_NEURON_PFK_EVICT", "FZ_NEURON_PFK_SELFTEST"): monkeypatch.delenv(k, raising=False)
  PF._STATE.update(ok=None, why="", override=None)
  yield
  PF._STATE.update(ok=None, why="", override=None)

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
  exp modes; the head's vocabulary window too."""
  for exp in ("exact", "act"):
    sp = PF.specs(N, 3584, 14, 2, 128, 9472, P, 1e-6, 0.088, 64, True, True, False, {"banks": 8, "wn": 4096, "exp": exp, "evict": "act"})
    for nm, s in sp.items():
      if nm == "kvw": continue
      pk = K.onchip_peak(s.render(), s.sizes())
      assert pk["sbuf"] <= 176 * 1024 and pk["psum"] <= 16 * 1024, (nm, exp, pk)
  h = K.PFLinear(4, 76032, 3584, 152064, 76032)
  pk = K.onchip_peak(h.render(), h.sizes())
  assert pk["sbuf"] <= 176 * 1024 and pk["psum"] <= 16 * 1024, pk

@pytest.mark.parametrize("N,P", [(128, 128), (128, 1024), (512, 512), (512, 1024)])
def test_onchip_budget_at_llama8b_shard_shapes(N, P):
  """I13: the same at Llama-3.1-8B's TP=2 shard shapes (D 4096, I 7168, 16 / 4 heads, head_dim 128, no biases, eps 1e-5)
  and its split lm_head (128,256 / 2 = 64,128 vocabulary columns per rank)."""
  for exp in ("exact", "act"):
    sp = PF.specs(N, 4096, 16, 4, 128, 7168, P, 1e-5, 0.088, 64, True, False, False, {"banks": 8, "wn": 4096, "exp": exp, "evict": "act"})
    for nm, s in sp.items():
      if nm == "kvw": continue
      pk = K.onchip_peak(s.render(), s.sizes())
      assert pk["sbuf"] <= 176 * 1024 and pk["psum"] <= 16 * 1024, (nm, exp, pk)
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
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_DTYPE", "FZ_NEURON_BF16_HEAD", "FZ_NEURON_PLAN", "FZ_NEURON_PFK_EXP"):
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

def json_dump(v, f):
  import json
  json.dump(v, f)
