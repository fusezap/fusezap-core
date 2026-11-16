"""W15: the float32 decode layer's weights in wide tiles (fusezap/codegen/nki_f32w.py, FZ_NEURON_F32_WN), checked on the Mac
(NKISHIM; FZ_TEST_TIERS=NEURON on hardware prints the share of equal bits and asserts closeness):
  * LayerF32W bit for bit against the float32 members (v1) and against LayerSpec itself (the same nc_matmul per PSUM chain;
    against the members on NKISHIM with the order-independent matmul FZ_NKISHIM_MM=f64, as every early-layer test);
  * through the model (prefill and decode logits, fused 24 + head fused, and FUSE=1) and the engine's tokens;
  * the hook: off by default, tiled layers and TP halves unchanged, the SBUF step-down, bad widths refused.
"""
from __future__ import annotations
import dataclasses
import numpy as np
import pytest
from fusezap.codegen import nki_kernels as nk, nki_bf16 as nb, nki_f32w as fw
from test_neuron_serve import LAYER_CASES, TIER, _layer_pair, _serve, tiny   # noqa: F401  (tiny: fixture)
from test_neuron_bf16 import _load
from fusezap.dtype import dtypes

_SP05 = nk.LayerSpec(1, 896, 14, 2, 64, 4864, 512, 1e-6, 0.125, 64, True, True, False, "stack", "slice", "mv", True, 0)

def _wn(monkeypatch, wn: str | None) -> None:
  monkeypatch.delenv("FZ_NEURON_DTYPE", raising=False)
  if wn is None: monkeypatch.delenv("FZ_NEURON_F32_WN", raising=False)
  else: monkeypatch.setenv("FZ_NEURON_F32_WN", wn)

@pytest.mark.parametrize("case,wn", [("qwen_1row_512", "2048"), ("qwen_1row_512", "4096"), ("qwen_4rows_768_of_1024", "4096"),
                                     ("odd_shapes", "1024"), ("first_layer_two_blocks", "1024"), ("one_kv_head_5rows", "2048")])
def test_wide_f32_layer_matches_the_members(case, wn, monkeypatch):
  """LayerF32W (v2se, the default layer) vs the float32 members (v1): residual, down projection, pool, X, attention output."""
  _wn(monkeypatch, wn)
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  seen: list = []
  real = fw.LayerF32W.render
  monkeypatch.setattr(fw.LayerF32W, "render", lambda self: (seen.append(self.wn), real(self))[1])
  for k, (a, b) in _layer_pair(*LAYER_CASES[case], "v2se", monkeypatch).items():
    if TIER == "NEURON":
      print(f"layer f32 {case} v2se+w{wn} {k} on NEURON: {np.mean(a.view(np.int32) == b.view(np.int32)):.6f} bitwise equal, max |diff| {np.abs(a - b).max():.3g}")
      np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} w{wn} {k}: max |diff| {np.abs(a - b).max()}"
  assert seen, "the wide float32 kernel was not built"

@pytest.mark.parametrize("case", ["qwen_1row_512", "odd_shapes"])
def test_wide_f32_layer_equals_layerspec_in_the_default_shim(case, monkeypatch):
  """Without the order-independent matmul too: every matmul of the wide kernel has the shapes and operand values of
  LayerSpec's (a 512-column slice of the wide tile), so NKISHIM's float32 matmul gives the same bits."""
  monkeypatch.delenv("FZ_NKISHIM_MM", raising=False)
  outs = {}
  for lab, wn in (("base", None), ("wide", "2048")):
    _wn(monkeypatch, wn)
    outs[lab] = _layer_pair(*LAYER_CASES[case], "v2se", monkeypatch)
  for k in outs["base"]:
    a, b = outs["base"][k][1], outs["wide"][k][1]
    if TIER == "NEURON": np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {k}"

def test_wide_f32_source():
  """One DMA per K chunk and wn-column block (gate_up 7 x 5 at 2048 instead of 7 x 19), float32 tiles, every 512-column slice
  its own PSUM tile; the name carries the width; the argument dtypes are LayerSpec's."""
  base = _SP05.render()
  p = fw.LayerF32W(_SP05, 2048).render()
  assert p.name == fw.LayerF32W(_SP05, 2048).name == _SP05.name + "_w2048" and p.buf_dtypes == base.buf_dtypes
  wgu = "data" + str(_SP05.args().index("wgu"))
  dmas = lambda s: sum(1 for ln in s.split("\n") if "nisa.dma_copy(dst=" in ln and f"src={wgu}.ap(" in ln)
  assert "for n0u in range(0, 9728, 512)" in base.src and "for n0u in" not in p.src
  assert dmas(p.src) == 5 and dmas(base.src) == 1
  assert "btu = nl.ndarray((ktu, 2048), dtype=nl.float32" in p.src and "bfloat16" not in p.src
  assert p.src.count("moving=btu[0:ktu, 1536:2048]") == 4   # blocks 0..3 full, block 4 is 1536 wide

def test_layer_f32w_hook(monkeypatch):
  """Off unless FZ_NEURON_F32_WN asks; tiled layers (I71) and TP halves (I72) unchanged; the width steps down until the SBUF
  estimate plus two wide float32 tiles fits; widths outside [512, 4096] or not multiples of 512 are refused."""
  _wn(monkeypatch, None)
  assert fw.f32_wn() == 512 and fw.layer_f32w(_SP05) is _SP05
  _wn(monkeypatch, "4096")
  assert fw.layer_f32w(_SP05) == fw.LayerF32W(_SP05, 4096)
  for sp in (dataclasses.replace(_SP05, tile=True, wcols=2048), dataclasses.replace(_SP05, part="attn"), dataclasses.replace(_SP05, part="mlp")):
    assert fw.layer_f32w(sp) is sp
  used = nb.layer_sbuf_bytes(_SP05)
  for budget, want in ((used + fw.wide_sbuf_bytes_f32(4096), 4096), (used + fw.wide_sbuf_bytes_f32(4096) - 1, 2048),
                       (used + fw.wide_sbuf_bytes_f32(1024), 1024), (used + fw.wide_sbuf_bytes_f32(1024) - 1, None)):
    monkeypatch.setattr(nb, "LAYER_SBUF", budget)
    got = fw.layer_f32w(_SP05)
    assert (got is _SP05) if want is None else got == fw.LayerF32W(_SP05, want), (budget, got)
  for bad in ("700", "256", "4608"):
    monkeypatch.setenv("FZ_NEURON_F32_WN", bad)
    with pytest.raises(ValueError): fw.f32_wn()

@pytest.mark.parametrize("fuse,batch,hf", [("24", 2, "1"), ("1", 1, "0")], ids=["fuse24-headfuse", "fuse1"])
def test_wide_f32_model_logits_equal_the_default(tiny, monkeypatch, fuse, batch, hf):
  """Through the model, float32 (prefill and decode logits): FZ_NEURON_F32_WN gives the default float32 path's bits, and
  the decode forward runs a fz_layer*_w* kernel."""
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_FUSE_MAX", "FZ_NEURON_HEAD"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
  monkeypatch.setenv("FZ_NEURON_HEAD_FUSE", hf)
  if TIER == "NKISHIM": monkeypatch.delenv("FZ_NKISHIM_MM", raising=False)
  out, names = {}, {}
  for lab, wn in (("base", None), ("wide", "1024")):
    _wn(monkeypatch, wn)
    m = _load(tiny, TIER, batch, dtype=dtypes.float32)
    assert not m.nbf16
    out[lab] = [m.prefill([list(range(5 + b, 21 + b)) for b in range(batch)], chunk=16)] + [m.decode([[3 + i + b] for b in range(batch)]) for i in range(4)]
    nm = []
    for it in m._neuron_steps.jits[(batch, batch)].captured.items:
      if not it.const and (sp := getattr(it.si.ast[0], "spec", None)) is not None: nm += [p[0] for p in sp.parts] if hasattr(sp, "parts") else [sp.name]
    names[lab] = nm
  for a, b in zip(out["base"], out["wide"]): assert np.array_equal(a.view(np.int32), b.view(np.int32))
  assert not any("_w1024" in n for n in names["base"]) and any(n.startswith("fz_layer") and n.endswith("_w1024") for n in names["wide"]), names["wide"]

def test_wide_f32_engine_tokens_match_cpu(tiny, monkeypatch):
  """LLMEngine on NEURON, float32 with FZ_NEURON_F32_WN=2048 (four requests, preempting): CPU float32's greedy tokens."""
  cpu, _ = _serve(tiny, "CPU", 6)
  _wn(monkeypatch, "2048")
  got, eng = _serve(tiny, TIER, 6)
  assert not eng.model.nbf16 and got == cpu
