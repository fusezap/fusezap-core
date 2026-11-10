"""N8 (I70): the bf16 decode path's weight streaming (fusezap/codegen/nki_bf16_n8.py), checked on the Mac (NKISHIM;
FZ_TEST_TIERS=NEURON on hardware prints the share of equal bits and asserts closeness):
  * LayerBF16W (FZ_NEURON_BF16_WN wide moving tiles, FZ_NEURON_BF16_PF / _PFQO prefetched weight tiles) bit for bit against
    the bf16 members and against LayerBF16 (the same nc_matmul per PSUM chain; on NKISHIM with the order-independent matmul,
    FZ_NKISHIM_MM=f64, because BLAS rounds a 512-column slice of a wider tile like any other operand only up to its kernel's
    blocking);
  * HeadBF16 (FZ_NEURON_BF16_HEAD) bit for bit against MatmulBF16 (the lm_head it replaces), every row-block layout;
  * the model: FZ_NEURON_BF16_HEAD loads only the bf16 head tiles, and the switches give the default bf16 path's logits bit for
    bit (prefill and decode, fused or not) and the engine's tokens;
  * the transforms refuse what they cannot place (NKIUnsupported), and the SBUF clamp of the prefetch.
"""
from __future__ import annotations
import numpy as np
import pytest
from fusezap.codegen import nki_kernels as nk, nki_bf16 as nb, nki_bf16_n8 as n8
from fusezap.serve.weights import f32_to_bf16
from test_neuron_serve import LAYER_CASES, TIER, _layer_pair, _serve, tiny   # noqa: F401  (tiny: fixture)
from test_neuron_bf16 import _load, _run

R = np.random.default_rng(70)
N8_VARIANTS = {"w1024": {"FZ_NEURON_BF16_WN": "1024"}, "w2048": {"FZ_NEURON_BF16_WN": "2048"}, "w4096": {"FZ_NEURON_BF16_WN": "4096"},
               "w2048p2q": {"FZ_NEURON_BF16_WN": "2048", "FZ_NEURON_BF16_PF": "2", "FZ_NEURON_BF16_PFQO": "1"},
               "w512p4": {"FZ_NEURON_BF16_PF": "4"}}

OFF = {"FZ_NEURON_BF16_WN": "512", "FZ_NEURON_BF16_HEAD": "0"}   # I68's bf16 kernels (the merged N8 defaults are WN 4096 + HeadBF16)

def _setenv(monkeypatch, env: dict | None, off: bool = True) -> None:
  """env over the switches off (env None and off False: every N8 switch unset, i.e. the defaults)."""
  for k in n8.N8_ENV: monkeypatch.delenv(k, raising=False)
  for k, v in {**(OFF if off else {}), **(env or {})}.items(): monkeypatch.setenv(k, v)

@pytest.mark.parametrize("n8v", list(N8_VARIANTS))
@pytest.mark.parametrize("case", list(LAYER_CASES))
def test_wide_layer_matches_the_bf16_members(case, n8v, monkeypatch):
  """LayerBF16W (v2se, the default layer) vs the bf16 members (v1): residual, down projection, pool, X, attention output."""
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  _setenv(monkeypatch, N8_VARIANTS[n8v])
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  seen: list = []
  real = n8.wide_weights
  monkeypatch.setattr(n8, "wide_weights", lambda *a, **k: (seen.append(a[1:]), real(*a, **k))[1])
  for k, (a, b) in _layer_pair(*LAYER_CASES[case], "v2se", monkeypatch).items():
    if TIER == "NEURON":
      print(f"layer bf16 {case} v2se+{n8v} {k} on NEURON: {np.mean(a.view(np.int32) == b.view(np.int32)):.6f} bitwise equal, max |diff| {np.abs(a - b).max():.3g}")
      np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {n8v} {k}: max |diff| {np.abs(a - b).max()}"
  assert seen, "the wide kernel was not built"

@pytest.mark.parametrize("variant", ["v2", "v2s", "v2se"])
@pytest.mark.parametrize("case", ["qwen_1row_512", "qwen_4rows_768_of_1024", "odd_shapes"])
def test_wide_layer_equals_layer_bf16_in_the_default_shim(case, variant, monkeypatch):
  """Without the order-independent matmul too: the wide kernel's every matmul has the shapes and operand values of
  LayerBF16's (a 512-column slice of the wide tile), so NKISHIM's float32 matmul gives the same bits."""
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  monkeypatch.delenv("FZ_NKISHIM_MM", raising=False)
  outs = {}
  for lab, env in (("base", {}), ("wide", N8_VARIANTS["w2048p2q"])):
    _setenv(monkeypatch, env)
    outs[lab] = _layer_pair(*LAYER_CASES[case], variant, monkeypatch)
  for k in outs["base"]:
    a, b = outs["base"][k][1], outs["wide"][k][1]
    if TIER == "NEURON": np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {variant} {k}"

def test_wide_layer_source():
  """The transform: one DMA per K chunk and wn-column block (gate_up 7 x 5 at 2048 instead of 7 x 19), every 512-column
  slice its own PSUM tile; prefetched tiles load in the attention section (gate_up, o_proj) and at the top (qkv)."""
  sp = nk.LayerSpec(1, 896, 14, 2, 64, 4864, 512, 1e-6, 0.125, 64, True, True, False, "stack", "slice", "mv", True, 0)
  base = nb.LayerBF16(sp)
  src0 = base.render().src
  w = n8.LayerBF16W(base, 2048)
  p = w.render()
  assert p.name == w.name == base.name + "_w2048" and p.buf_dtypes == base.render().buf_dtypes
  dmas = lambda s, d: sum(1 for ln in s.split("\n") if "nisa.dma_copy(dst=" in ln and f"src={d}.ap(" in ln)
  wgu = "data" + str(sp.args().index("wgu"))
  assert "for n0u in range(0, 9728, 512)" in src0 and "for n0u in" not in p.src
  assert p.src.count("ps" + "u0_") >= 4 and p.src.count("moving=btu[0:ktu, 1536:2048]") == 4   # blocks 0..3 full, block 4 is 1536 wide
  assert dmas(p.src, wgu) == 5 and dmas(src0, wgu) == 1   # one DMA line per wide block (inside its K loop), vs one in the 512 loop
  q = n8.LayerBF16W(base, 2048, 2, True).render()
  lines = q.src.split("\n")
  att = next(i for i, ln in enumerate(lines) if ln.strip().startswith("# 5. attention"))
  top = next(i for i, ln in enumerate(lines) if ln.strip().startswith("# 0. input-only"))
  first = lambda pat: next(i for i, ln in enumerate(lines) if pat in ln)
  assert top < first("pfq0_0 = nl.ndarray") < first("# 1. x1") and att < first("pfo0_0 = nl.ndarray") < first("pfu0_0 = nl.ndarray") < first("pfu1_6 = ")
  assert "pfu2_0" not in q.src and q.name.endswith("_w2048p2q")
  with pytest.raises(nk.NKIUnsupported): n8.wide_weights(nb.MatmulBF16(1, 512, 128).render(), 1000)
  with pytest.raises(nk.NKIUnsupported): n8.wide_weights(nk.render_kv_write(nk.KVWriteSpec(256, 2, 64, 97, 0)), 2048)

def test_prefetch_is_clamped_to_sbuf(monkeypatch):
  sp = nk.LayerSpec(1, 896, 14, 2, 64, 4864, 512, 1e-6, 0.125, 64, True, True, False, "stack", "slice", "mv", True, 0)
  base = nb.LayerBF16(sp)
  assert n8.layer_bf16w(base, 512, 0, False) is base   # off: unchanged
  room = nb.LAYER_SBUF - nb.layer_sbuf_bytes(sp)
  got = n8.layer_bf16w(base, 2048, 50, True)
  assert n8.prefetch_bytes(sp, 2048, got.pf, got.pfqo) <= room < n8.prefetch_bytes(sp, 2048, got.pf + 1, got.pfqo)
  assert n8.prefetch_bytes(sp, 2048, 1, False) == 7 * 2048 * 2   # one gate_up block: 7 K chunks of 2048 bf16 columns
  monkeypatch.setattr(nb, "LAYER_SBUF", nb.layer_sbuf_bytes(sp))
  assert n8.layer_bf16w(base, 2048, 3, True) == n8.LayerBF16W(base, 2048, 0, False)
  monkeypatch.setenv("FZ_NEURON_BF16_WN", "700")
  with pytest.raises(ValueError): n8.bf16_wn()

@pytest.mark.parametrize("M,V,K,W", [(1, 5000, 896, 2048), (4, 4096, 300, 1024), (130, 1536, 200, 512), (3, 700, 64, 4096),
                                     (4, 128256, 256, 2048)],
                         ids=["decode_row", "rows4_ktail", "two_row_blocks", "one_block", "llama31_vocab"])
def test_head_bf16_matches_matmul_bf16(M, V, K, W):
  """HeadBF16 on head_tiles_bf16's layout = MatmulBF16 on the K-major weight, bit for bit (the same Aᵀ rounding, the same
  512-column nc_matmul per PSUM chain); only the weight DMAs (one per W-column block) and the stores differ."""
  h = R.standard_normal(M * K).astype(np.float32)
  wt = f32_to_bf16(R.standard_normal((V, K)).astype(np.float32))   # torch layout (V, K), bf16 bits
  want = _run(nb.MatmulBF16(M, V, K).render(), [np.full(M * V, np.nan, np.float32), h, np.ascontiguousarray(wt.T).reshape(-1)])[0]
  sp = n8.HeadBF16(M, V, K, W)
  prog = sp.render()
  assert prog.name == sp.name and prog.buf_dtypes[2].itemsize == 2 and "at_bf" in prog.src
  got = _run(prog, [np.full(M * V, np.nan, np.float32), h, n8.head_tiles_bf16(wt, W)])[0]
  if TIER == "NEURON":
    print(f"head bf16 {M}x{V}x{K} W {W} on NEURON: {np.mean(got.view(np.int32) == want.view(np.int32)):.6f} bitwise equal")
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
  else: assert np.array_equal(got.view(np.int32), want.view(np.int32))

def test_head_tiles_bf16_layout():
  wt = np.arange(5 * 300, dtype=np.uint16).reshape(5, 300)
  f32 = nk.head_tiles(wt.astype(np.float32), 512)
  assert np.array_equal(n8.head_tiles_bf16(wt, 512), f32.astype(np.uint16))

@pytest.mark.parametrize("env,fuse,batch,hf", [({"FZ_NEURON_BF16_HEAD": "1"}, "1", 1, "0"), (N8_VARIANTS["w2048p2q"], "1", 2, "0"),
                                              ({"FZ_NEURON_BF16_HEAD": "1", **N8_VARIANTS["w1024"]}, "24", 2, "1"),
                                              ({"FZ_NEURON_BF16_HEAD": "1", "FZ_NEURON_BF16_HEAD_W": "512", **N8_VARIANTS["w2048p2q"]}, "", 1, "")],
                         ids=["head", "wide-prefetch", "both-fuse24-headfuse", "defaults-head512"])
def test_n8_model_logits_equal_the_bf16_path(tiny, monkeypatch, env, fuse, batch, hf):
  """Through the model (prefill and decode logits): the N8 switches give the default bf16 path's bits, and the decode
  forward runs the N8 kernels (a fz_layer*_w* layer, the fz_headb head)."""
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_FUSE_MAX", "FZ_NEURON_HEAD"): monkeypatch.delenv(k, raising=False)
  if fuse: monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
  if hf: monkeypatch.setenv("FZ_NEURON_HEAD_FUSE", hf)
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  out, names = {}, {}
  for lab, e in (("base", {}), ("n8", env)):
    _setenv(monkeypatch, e)
    m = _load(tiny, TIER, batch)
    if lab == "n8" and "FZ_NEURON_BF16_HEAD" in e:
      assert m.lm_head is m.lm_head_t and m.lm_head_k is None and m.lm_head.dtype.itemsize == 2
    out[lab] = [m.prefill([list(range(5 + b, 21 + b)) for b in range(batch)], chunk=16)] + [m.decode([[3 + i + b] for b in range(batch)]) for i in range(4)]
    nm = []
    for it in m._neuron_steps.jits[(batch, batch)].captured.items:
      if not it.const and (sp := getattr(it.si.ast[0], "spec", None)) is not None: nm += [p[0] for p in sp.parts] if hasattr(sp, "parts") else [sp.name]
    names[lab] = nm
  for a, b in zip(out["base"], out["n8"]): assert np.array_equal(a.view(np.int32), b.view(np.int32))
  assert any(n.startswith("fz_headb") for n in names["n8"]) == ("FZ_NEURON_BF16_HEAD" in env)
  assert any(n.startswith("fz_layer") and "_bf16_w" in n for n in names["n8"]) == ("FZ_NEURON_BF16_WN" in env or "FZ_NEURON_BF16_PF" in env)

def test_n8_engine_tokens_match_cpu(tiny, monkeypatch):
  """LLMEngine on NEURON, bf16 with every N8 switch on (four requests, preempting): CPU float32's greedy tokens."""
  cpu, _ = _serve(tiny, "CPU", 6)
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  _setenv(monkeypatch, {"FZ_NEURON_BF16_HEAD": "1", **N8_VARIANTS["w2048p2q"]})
  got, eng = _serve(tiny, TIER, 6)
  assert eng.model.nbf16 and eng.model.lm_head_t is not None and got == cpu

# ---------------------------------------------------------------- the merged defaults (trn1 N8: WN 4096, HeadBF16 at W 2048)

_SP05 = nk.LayerSpec(1, 896, 14, 2, 64, 4864, 512, 1e-6, 0.125, 64, True, True, False, "stack", "slice", "mv", True, 0)

def test_default_wide_width_and_its_fit_fallback(monkeypatch):
  """Switches unset: a whole untiled layer (Qwen2.5-0.5B's) gets 4096-column tiles, no prefetch; where the SBUF estimate plus
  the wider tiles does not fit, 2048, then 1024, then I68's layer unchanged. An explicit width is kept as given."""
  _setenv(monkeypatch, None, off=False)
  assert n8.bf16_wn() == n8.WN_DEFAULT == 4096
  base = nb.LayerBF16(_SP05)
  assert n8.layer_bf16w(base) == n8.LayerBF16W(base, 4096, 0, False)
  used = nb.layer_sbuf_bytes(_SP05)
  for budget, want in ((used + n8.wide_sbuf_bytes(4096), 4096), (used + n8.wide_sbuf_bytes(4096) - 1, 2048),
                       (used + n8.wide_sbuf_bytes(1024), 1024), (used + n8.wide_sbuf_bytes(1024) - 1, None)):
    monkeypatch.setattr(nb, "LAYER_SBUF", budget)
    got = n8.layer_bf16w(base)
    assert (got is base) if want is None else got == n8.LayerBF16W(base, want, 0, False), (budget, got)
  monkeypatch.setenv("FZ_NEURON_BF16_WN", "4096")   # explicit: not stepped down (the caller asked for it)
  assert n8.layer_bf16w(base) == n8.LayerBF16W(base, 4096, 0, False)
  monkeypatch.setenv("FZ_NEURON_BF16_WN", "512")
  assert n8.layer_bf16w(base) is base

@pytest.mark.parametrize("env", [None, {"FZ_NEURON_BF16_WN": "2048", "FZ_NEURON_BF16_PF": "2", "FZ_NEURON_BF16_PFQO": "1"}], ids=["defaults", "explicit"])
def test_tiled_layers_and_tp_halves_stay_unchanged(env, monkeypatch):
  """layer_bf16w leaves the tiled layer (I71: its own wcols; wide_weights does not match its blocks) and the TP halves (I72:
  not validated with wide tiles) as they are, defaults or explicit; forcing LayerBF16W on a tiled layer fails loudly."""
  import dataclasses
  _setenv(monkeypatch, env, off=False)
  sp7 = nk.LayerSpec(1, 3584, 28, 4, 128, 18944, 512, 1e-6, 0.088, 64, True, True, False, "stack", "slice", "mv", True, 0)
  for sp in (dataclasses.replace(_SP05, tile=True, wcols=2048), dataclasses.replace(_SP05, part="attn"), dataclasses.replace(_SP05, part="mlp"),
             dataclasses.replace(sp7, tile=True, wcols=2048)):
    b = nb.LayerBF16(sp)
    assert n8.layer_bf16w(b) is b
  with pytest.raises(nk.NKIUnsupported): n8.LayerBF16W(nb.LayerBF16(dataclasses.replace(_SP05, tile=True)), 4096).render()

def test_default_head_is_bf16_without_tp(monkeypatch):
  _setenv(monkeypatch, None, off=False)
  assert n8.bf16_head() and not n8.bf16_head(tp=True) and n8.bf16_head_w() == 2048
  monkeypatch.setenv("FZ_NEURON_BF16_HEAD", "1"); assert n8.bf16_head(tp=True)
  monkeypatch.setenv("FZ_NEURON_BF16_HEAD", "0"); assert not n8.bf16_head()

@pytest.mark.parametrize("variant", ["v2se", "v2set4"])
@pytest.mark.parametrize("case", ["qwen_1row_512", "qwen_4rows_768_of_1024"])
def test_default_layer_matches_the_bf16_members(case, variant, monkeypatch):
  """Switches unset, through NeuronKVCache.layer: the whole layer runs wide (4096), the tiled one (I71) as LayerBF16 with its
  own wcols; both bit for bit the bf16 members."""
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  _setenv(monkeypatch, None, off=False)
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  seen: list = []   # the specs NeuronKVCache.layer ran (rendered programs are cached by name across tests: spy the hook)
  real = n8.layer_bf16w
  monkeypatch.setattr(n8, "layer_bf16w", lambda *a, **k: (r := real(*a, **k), seen.append(r))[0])
  for k, (a, b) in _layer_pair(*LAYER_CASES[case], variant, monkeypatch).items():
    if TIER == "NEURON": np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {variant} {k}: max |diff| {np.abs(a - b).max()}"
  assert seen
  if variant == "v2set4": assert all(isinstance(s, nb.LayerBF16) and s.base.tile for s in seen)
  else: assert all(isinstance(s, n8.LayerBF16W) and s.wn == 4096 for s in seen)

def test_default_model_logits_equal_the_i68_path(tiny, monkeypatch):
  """Switches unset (the merged defaults): the model runs the wide layer and HeadBF16 and gives I68's bf16 logits bit for bit."""
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_FUSE_MAX", "FZ_NEURON_HEAD"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  out, names = {}, {}
  for lab, off in (("i68", True), ("defaults", False)):
    _setenv(monkeypatch, None, off)
    m = _load(tiny, TIER, 2)
    assert (m.lm_head is m.lm_head_t and m.lm_head_k is None) == (not off)
    out[lab] = [m.prefill([list(range(5 + b, 21 + b)) for b in range(2)], chunk=16)] + [m.decode([[3 + i + b] for b in range(2)]) for i in range(4)]
    nm = []
    for it in m._neuron_steps.jits[(2, 2)].captured.items:
      if not it.const and (sp := getattr(it.si.ast[0], "spec", None)) is not None: nm += [p[0] for p in sp.parts] if hasattr(sp, "parts") else [sp.name]
    names[lab] = nm
  for a, b in zip(out["i68"], out["defaults"]): assert np.array_equal(a.view(np.int32), b.view(np.int32))
  assert any(n.startswith("fz_headb2048") for n in names["defaults"]) and not any(n.startswith("fz_headb") for n in names["i68"])
  assert any("_bf16_w4096" in n for n in names["defaults"]) and not any("_bf16_w" in n for n in names["i68"])
