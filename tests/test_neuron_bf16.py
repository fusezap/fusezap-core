"""I68: bf16 on NEURON (fusezap/codegen/nki_bf16.py), checked on the Mac (NKISHIM; FZ_TEST_TIERS=NEURON / NKISIM select those):
  * MatmulBF16 against numpy's bf16-emulated reference (bf16(A) @ W in float64: exact products, so only float32
    accumulation separates them), every layout render_matmul takes (A kept / reloaded per tile, row blocks, K tails);
  * LayerBF16 (one decoder layer, bf16 weight matmuls) bit for bit against the members on the bf16 path (MatmulBF16 for
    every linear, the float32 kernels for the rest), for the LayerSpec cases and variants;
  * FZ_NEURON_BF16_ATTN (bf16 q, k, p, v) against the float32 layer within bf16 tolerance, and its attention against a
    numpy reference with bf16-rounded operands;
  * the model: Llama on NEURON with --dtype bfloat16 / FZ_NEURON_DTYPE=bf16 loads bf16 K-major weights, v1 and v2 give the
    same logits bit for bit, fused or not, and the engine's tokens equal CPU's float32 tokens on the tiny checkpoint
    (and CPU's logits within bf16 tolerance).
"""
from __future__ import annotations
import os
import numpy as np
import pytest
from fusezap.device import Buffer, Device
from fusezap.dtype import dtypes
from fusezap.codegen import nki_kernels as nk, nki_bf16 as nb
from fusezap.serve.weights import bf16_to_f32, f32_to_bf16
from test_neuron_serve import LAYER_CASES, TIER, _layer_pair, _serve, tiny   # noqa: F401  (tiny: fixture)

R = np.random.default_rng(11)

@pytest.fixture(autouse=True)
def _i68_kernels(monkeypatch):
  """These tests pin I68's bf16 kernels (MatmulBF16 lm_head, 512-column weight tiles); the merged N8 defaults (wide tiles,
  HeadBF16) are checked in test_neuron_bf16_n8.py."""
  monkeypatch.setenv("FZ_NEURON_BF16_WN", "512")
  monkeypatch.setenv("FZ_NEURON_BF16_HEAD", "0")

def _run(prog, arrays: list[np.ndarray]) -> list[np.ndarray]:
  """A rendered program on TIER over device copies of flat arrays (float32, int32, or uint16 = bf16 bits)."""
  dev = Device[TIER]
  dt = {np.dtype(np.float32): dtypes.float32, np.dtype(np.int32): dtypes.int32, np.dtype(np.uint16): dtypes.bfloat16}
  bufs = [Buffer(TIER, a.size, dt[a.dtype]).copyin(memoryview(np.ascontiguousarray(a)).cast("B")) for a in arrays]
  dev.run_program(dev.runtime(prog, dev.compiler.compile(prog.src)), tuple(bufs))
  return [b.numpy() for b in bufs]

@pytest.mark.parametrize("M,N,K", [(1, 1000, 300), (4, 4864, 896), (130, 520, 300), (128, 520, 16640), (3, 77, 64)],
                         ids=["decode_row", "qwen_gate_up_4rows", "two_row_blocks", "A_reloaded_per_tile", "small"])
def test_matmul_bf16_matches_the_emulated_reference(M, N, K):
  """C = bf16(A) @ W with W bf16 in HBM: TensorE's bf16 x bf16 products are exact in float32 and accumulate in float32
  PSUM, so the result is the float64 product of the rounded operands up to float32 accumulation (|err| <= ~K * 2^-24 of
  the sum of |products|)."""
  a = R.standard_normal(M * K).astype(np.float32)
  w = f32_to_bf16(R.standard_normal((K, N)).astype(np.float32))
  sp = nb.MatmulBF16(M, N, K)
  prog = sp.render()
  assert prog.buf_dtypes[2] == dtypes.bfloat16 and prog.name == sp.name and "dtype=nl.bfloat16" in prog.src
  got = _run(prog, [np.full(M * N, np.nan, np.float32), a, w.reshape(-1)])[0].reshape(M, N)
  ref = nb.matmul_ref(a.reshape(M, K), w)
  bound = np.abs(nb.bf16_round(a.reshape(M, K))).astype(np.float64) @ np.abs(bf16_to_f32(w)).astype(np.float64)
  assert np.all(np.abs(got - ref) <= 4 * K * 2.0 ** -24 * bound + 1e-30), float(np.max(np.abs(got - ref) / (bound + 1e-30)))
  f32 = a.reshape(M, K).astype(np.float64) @ bf16_to_f32(w).astype(np.float64)   # the activation rounding is visible
  assert np.abs(got - f32).max() > 0 and np.abs(got - f32).max() <= 2 ** -8 * bound.max()

def test_bf16_edits_fail_loudly():
  """A kernel without weight tiles (or a renamed pattern) is NKIUnsupported, never silently float32."""
  prog = nk.render_kv_write(nk.KVWriteSpec(256, 2, 64, 97, 0))
  with pytest.raises(nk.NKIUnsupported): nb.bf16_weight_matmuls(prog, (1,))
  with pytest.raises(nk.NKIUnsupported): nb.bf16_attention(nb.MatmulBF16(1, 512, 128).render())

def _bf16_layer_pair(case: str, variant: str, monkeypatch) -> dict:
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  return _layer_pair(*LAYER_CASES[case], variant, monkeypatch)

@pytest.mark.parametrize("variant", ["v2", "v2s", "v2a", "v2e", "v2se"])
@pytest.mark.parametrize("case", list(LAYER_CASES))
def test_layer_bf16_matches_the_members(case, variant, monkeypatch):
  """LayerBF16 = LayerSpec's instructions with the weight matmuls' operands in bf16; the members on the bf16 path round
  the same Aᵀ values the same way (MatmulBF16), so every output is the same bits (residual, down projection, pool, X,
  attention output). Early (e: rows moved) on NKISHIM with the order-independent matmul (FZ_NKISHIM_MM=f64, as
  test_layer_kernel_matches_the_members)."""
  if variant[2:].strip("sa") and TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  for k, (a, b) in _bf16_layer_pair(case, variant, monkeypatch).items():
    if TIER == "NEURON":
      print(f"layer bf16 {case} {variant} {k} on NEURON: {np.mean(a.view(np.int32) == b.view(np.int32)):.6f} bitwise equal, max |diff| {np.abs(a - b).max():.3g}")
      np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {variant} {k}: max |diff| {np.abs(a - b).max()}"

@pytest.mark.parametrize("case", ["qwen_4rows_768_of_1024", "odd_shapes", "first_layer_two_blocks"])
def test_layer_bf16a_early_matches_bf16a(case, monkeypatch):
  """N8: the default layer (v2se: early, heads stacked) with FZ_NEURON_BF16_ATTN=1 gives v2s's bf16a bits (early only
  reorders, moves values exactly and skips a rescale of zeros; the bf16 casts of q, k, p, v see the same values). On
  NKISHIM with FZ_NKISHIM_MM=f64 (early moves attention rows: test_layer_kernel_matches_the_members)."""
  monkeypatch.setenv("FZ_NEURON_BF16_ATTN", "1")
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  a, b = _bf16_layer_pair(case, "v2s", monkeypatch), _bf16_layer_pair(case, "v2se", monkeypatch)
  for k in a:
    x, y = a[k][1], b[k][1]
    if TIER == "NEURON": np.testing.assert_allclose(x, y, rtol=1e-3, atol=1e-4)
    else: assert np.array_equal(x.view(np.int32), y.view(np.int32)), f"{case} {k}: max |diff| {np.abs(x - y).max()}"

def test_layer_bf16_renders_every_variant():
  """Every LAYER_VARIANTS LayerSpec (Qwen2.5-0.5B's shapes) with mm=mv has a bf16 form (the weight DMAs into bf16 tiles, a
  bf16 copy of each weight matmul's other operand, bf16 weight arguments); the weight-stationary forms (mm=ws: off, not
  bitwise on trn1) have none and say so (NeuronKVCache.layer runs the members for them), as does a weight loaded into a
  tile the edits do not know (never silently float32)."""
  for v, env in nk.LAYER_VARIANTS.items():
    if v == "v1": continue
    for attn in (False, True):
      sp = nb.LayerBF16(nk.LayerSpec(1, 896, 14, 2, 64, 4864, 512, 1e-6, 0.125, 64, True, True, False, env["FZ_NEURON_LAYER_HEADS"],
                                     env["FZ_NEURON_LAYER_ROPE"], env["FZ_NEURON_LAYER_MM"], env["FZ_NEURON_LAYER_EARLY"] == "1",
                                     int(env["FZ_NEURON_LAYER_PF"])), attn)
      if env["FZ_NEURON_LAYER_MM"] == "ws":
        with pytest.raises(nk.NKIUnsupported): sp.render()
        with pytest.raises(nk.NKIUnsupported):   # the DMA guard alone (past LayerBF16's mm check)
          nb.bf16_weight_matmuls(sp.base.render(), tuple(sp.args().index(w) for w in ("wqkv", "wo", "wgu", "wd")))
        continue
      prog = sp.render()
      assert prog.name.endswith("_bf16a" if attn else "_bf16") and "_bf = nl.ndarray(" in prog.src
      assert all(prog.buf_dtypes[sp.args().index(w)] == dtypes.bfloat16 for w in ("wqkv", "wo", "wgu", "wd"))

def test_layer_sbuf_counts_early_masks():
  """layer_fits: the early layer keeps every key block's mask live (P bytes per partition more than the late one)."""
  late = nk.LayerSpec(1, 896, 14, 2, 64, 4864, 4096, 1e-6, 0.125, 0, heads="stack")
  early = nk.LayerSpec(1, 896, 14, 2, 64, 4864, 4096, 1e-6, 0.125, 0, heads="stack", early=True)
  assert nb.layer_sbuf_bytes(early) == nb.layer_sbuf_bytes(late) + 4096 and nb.layer_fits(early)

@pytest.mark.parametrize("case", ["qwen_1row_512", "odd_shapes", "one_kv_head_5rows"])
def test_layer_bf16_is_close_to_float32(case, monkeypatch):
  """The bf16 layer against the float32 layer (same inputs): within bf16 rounding of the matmul operands (2^-8 relative
  per operand), far from equal; FZ_NEURON_BF16_ATTN=1 moves the attention output by at most the same order."""
  f32 = _layer_pair(*LAYER_CASES[case], "v2", monkeypatch)
  b16 = _bf16_layer_pair(case, "v2", monkeypatch)
  monkeypatch.setenv("FZ_NEURON_BF16_ATTN", "1")
  b16a = _bf16_layer_pair(case, "v2", monkeypatch)
  for k in ("residual", "down", "attention"):
    a, b, c = f32[k][1], b16[k][1], b16a[k][1]
    scale = np.abs(a).max()
    assert 0 < np.abs(a - b).max() <= 0.05 * scale, (k, np.abs(a - b).max(), scale)
    assert np.abs(b - c).max() <= 0.05 * scale, (k, np.abs(b - c).max(), scale)
  assert not np.array_equal(b16["attention"][1], b16a["attention"][1])
  assert np.array_equal(b16["X"][1], b16a["X"][1])   # q, k, v before attention: unchanged by the attention switch

def test_bf16_attention_matches_the_emulated_reference(monkeypatch):
  """FZ_NEURON_BF16_ATTN: the layer's attention output against AttnSpec.ref (float64 softmax) on bf16-rounded q, k, v (p
  is rounded on the chip too: a relative error of ~2^-9 per weight, so a loose tolerance)."""
  D, I, Hq, Hkv, hd, rows, P, bs, ctx, add, ob = LAYER_CASES["odd_shapes"]
  monkeypatch.setenv("FZ_NEURON_BF16_ATTN", "1")
  got = _bf16_layer_pair("odd_shapes", "v2", monkeypatch)
  X, O, pool = got["X"][1], got["attention"][1], got["pool"][1]
  C, Wx, G = Hkv * hd, (Hq + 2 * Hkv + 1) * hd, Hq // Hkv
  from fusezap.serve.neuron_kv import NeuronKVCache
  rng = np.random.default_rng(D * 7 + I + rows * 31 + P)   # _layer_pair's positions (same seed, same draws)
  r = lambda *s, sc=0.1: (rng.standard_normal(s) * sc).astype(np.float32)
  for s in [(64, D), (D,), (64, D), (D,), (D,), (Hq * hd, D), (Hkv * hd, D), (Hkv * hd, D), (D, Hq * hd), (I, D), (I, D), (D, I),
            (Hq * hd,), (Hkv * hd,), (Hkv * hd,), (D,)]: r(*s)
  pos = rng.integers(0, P // rows - bs, rows)
  cache = NeuronKVCache(1, Hkv, hd, P // bs, bs, dtypes.float32, "CPU", max_seq_len=P)
  for b in range(rows): cache.ensure(b, int(pos[b]) + 1)
  meta = cache.step_meta_rows(rows, rows, np.arange(rows), pos, list(range(rows)))
  base = int(len(meta) - 2 * rows - P)
  sp = nk.AttnSpec((Hkv, 1), (rows, G), P, hd, hd, (G * hd, 0, hd, 1, 0), (hd, 0, 2 * C, 1, 0), (hd, 0, 2 * C, 1, C), (G * hd, 0, hd, 1, 0),
                   1 / np.sqrt(hd), "keyrange", (1, 0, base, base + rows, base + 2 * rows), rs=(Wx, Hq * hd), kv_one=True)
  ref = np.zeros(rows * Hq * hd)
  q, pl = nb.bf16_round(X), nb.bf16_round(pool)
  sp.ref([ref, q, pl, pl, meta])
  np.testing.assert_allclose(O, ref, rtol=0, atol=2e-2 * np.abs(ref).max())

# ---------------------------------------------------------------- the model

def _load(path: str, device: str, batch: int = 1, dtype=dtypes.bfloat16):
  from fusezap.serve.models import load_model
  return load_model(path, dtype=dtype, device=device, max_len=128, prefill_chunk=16, max_batch=batch)

def test_model_loads_bf16_weights(tiny):
  """--dtype bfloat16 on NEURON: bf16 K-major linear weights and lm_head (no torch-layout copy), float32 activations."""
  m = _load(tiny, TIER)
  l, c = m.layers[0], m.cfg
  assert m.nbf16 and m.dtype == dtypes.float32 and m.wdtype == dtypes.bfloat16
  assert l.wqkv.dtype == l.wo.dtype == l.wgu.dtype == l.wd.dtype == m.lm_head.dtype == dtypes.bfloat16
  assert tuple(l.wqkv.shape) == (c.hidden_size, (c.n_heads + 2 * c.n_kv_heads) * c.head_dim) and tuple(m.lm_head.shape) == (c.hidden_size, c.vocab_size)
  assert l.ln1.dtype == dtypes.float32 and (l.bqkv is None or l.bqkv.dtype == dtypes.float32)
  cpu = _load(tiny, "CPU")
  assert not cpu.nbf16 and cpu.dtype == dtypes.bfloat16   # other devices: unchanged

@pytest.mark.parametrize("variant,fuse,batch,hf", [("v2", "1", 1, "0"), ("v2s", "0", 2, "0"), ("v2", "2", 3, "0"), ("v2se", "24", 2, "1"),
                                                   ("v2se", "", 1, "")], ids=["v2-fuse1", "v2s-unfused", "v2-fuse2", "v2se-fuse24-headfuse", "defaults"])
def test_bf16_model_layer_paths_are_bit_identical(tiny, monkeypatch, variant, fuse, batch, hf):
  """bf16 through the model: prefill and decode logits of v1 (members, MatmulBF16) and v2 (LayerBF16) bit for bit, fused
  or not, every decode launch a bf16 kernel; and within bf16 tolerance of CPU float32 (the same top-1 here). N8: the
  defaults (FZ_NEURON_FUSE=24, FZ_NEURON_HEAD_FUSE=1, head v2: a bf16 lm_head stays MatmulBF16, HeadSpec is float32-only)."""
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_FUSE_MAX", "FZ_NEURON_HEAD"): monkeypatch.delenv(k, raising=False)
  if fuse: monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
  if hf: monkeypatch.setenv("FZ_NEURON_HEAD_FUSE", hf)
  out = {}
  for v in ("v1", variant):
    for k, val in nk.LAYER_VARIANTS[v].items(): monkeypatch.setenv(k, val)
    m = _load(tiny, TIER, batch)
    out[v] = [m.prefill([list(range(5 + b, 21 + b)) for b in range(batch)], chunk=16)] + [m.decode([[3 + i + b] for b in range(batch)]) for i in range(4)]
    names = []
    for it in m._neuron_steps.jits[(batch, batch)].captured.items:
      if not it.const and (sp := getattr(it.si.ast[0], "spec", None)) is not None: names += [p[0] for p in sp.parts] if hasattr(sp, "parts") else [sp.name]
    assert sum(n.startswith("fz_layer") and n.endswith("_bf16") for n in names) == (2 if v != "v1" else 0)
    assert sum(n.startswith("fz_matmul") and n.endswith("_bf16") for n in names) == (1 if v != "v1" else 2 * 4 + 1)   # lm_head (+ 4 linears per layer)
  for a, b in zip(out["v1"], out[variant]): assert np.array_equal(a.view(np.int32), b.view(np.int32))
  ref = _load(tiny, "CPU", batch, dtypes.float32)
  want = [ref.prefill([list(range(5 + b, 21 + b)) for b in range(batch)], chunk=16)] + [ref.decode([[3 + i + b] for b in range(batch)]) for i in range(4)]
  for a, b in zip(out[variant], want):
    assert np.abs(a - b).max() <= 0.05 * np.abs(b).max() and np.abs(a - b).max() > 0
    assert (a.argmax(-1) == b.argmax(-1)).all()

def test_bf16_engine_tokens_match_cpu(tiny, monkeypatch):
  """LLMEngine on NEURON with FZ_NEURON_DTYPE=bf16 (four requests, preempting): CPU float32's greedy tokens (the tiny
  checkpoint has no near ties); FZ_NEURON_BF16_ATTN=1 keeps most of them (here a late near tie flips)."""
  cpu, _ = _serve(tiny, "CPU", 6)
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  got, eng = _serve(tiny, TIER, 6)
  assert eng.model.nbf16 and eng.scheduler.num_preemptions > 0 and got == cpu
  monkeypatch.setenv("FZ_NEURON_BF16_ATTN", "1")
  got2, _ = _serve(tiny, TIER, 6)
  prefix = [next((i for i, (x, y) in enumerate(zip(got2[r], cpu[r])) if x != y), len(cpu[r])) for r in cpu]
  assert sum(prefix) >= 0.75 * sum(len(t) for t in cpu.values()), (prefix, got2, cpu)   # bf16 attention: near ties may flip

# ---------------------------------------------------------------- I69: Qwen2.5-7B's head layout, and layers too wide for SBUF

def test_qwen7b_shaped_bf16_tokens_match_cpu(tmp_path, monkeypatch):
  """Qwen2.5-7B's attention layout (28 query / 4 kv heads, head_dim 128: 7 query heads per kv head, 4 kv heads in the
  layer kernel) at small widths, bf16 on NEURON: CPU float32's greedy tokens, with LayerBF16 and with the members."""
  from test_tp import tiny_model_dir
  p = tiny_model_dir(str(tmp_path / "q7"), seed=3, cfg={"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 28,
                                                         "num_key_value_heads": 4, "head_dim": 128})
  cpu, _ = _serve(p, "CPU")
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  for layer in ("v2", "v1"):
    monkeypatch.setenv("FZ_NEURON_LAYER", layer)
    got, eng = _serve(p, TIER)
    prefix = [next((i for i, (x, y) in enumerate(zip(got[r], cpu[r])) if x != y), len(cpu[r])) for r in cpu]
    assert sum(prefix) >= 0.75 * sum(len(t) for t in cpu.values()), (layer, prefix)

def test_layer_too_wide_for_sbuf_runs_the_members(tiny, monkeypatch):
  """A layer whose rows exceed the SBUF budget (Qwen2.5-7B: ~400 KiB per partition) runs as the members, bf16 or not."""
  from fusezap.codegen.nki_bf16 import layer_fits
  q7 = nk.LayerSpec(1, 3584, 28, 4, 128, 18944, 512, 1e-6, 0.088, 0)
  q05 = nk.LayerSpec(4, 896, 14, 2, 64, 4864, 4096, 1e-6, 0.125, 0)
  assert not layer_fits(q7) and layer_fits(q05)
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  monkeypatch.setattr("fusezap.codegen.nki_bf16.LAYER_SBUF", 1024)
  m = _load(tiny, TIER)
  m.decode([[3]]); m.decode([[4]])
  names = [it.si.ast[0].spec.name for it in m._neuron_steps.jits[(1, 1)].captured.items if not it.const and hasattr(it.si.ast[0], "spec")]
  assert names and not any("fz_layer" in n for n in names)

# ---------------------------------------------------------------- N8 (I71): the tiled layer (Qwen2.5-7B's layer on one NeuronCore)

TILED = ["v2et", "v2set", "v2set2", "v2set4"]

@pytest.mark.parametrize("dt", ["f32", "bf16"])
@pytest.mark.parametrize("variant", TILED)
@pytest.mark.parametrize("case", list(LAYER_CASES))
def test_tiled_layer_matches_the_members(case, variant, dt, monkeypatch):
  """N8: LayerSpec.tile (four reused (N, D) row tiles, qkv evicted per 512-column slice with its bias and RoPE straight into X,
  gate / up in column blocks transposed and silu'd into aT, weight DMAs of 512 / 1024 / 2048 columns) gives the members'
  bits, float32 and bf16: residual, down projection, every pool row, X and the attention output. Early rows: checked with
  FZ_NKISHIM_MM=f64 on NKISHIM (test_layer_kernel_matches_the_members)."""
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  if dt == "bf16": monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  for k, (a, b) in _layer_pair(*LAYER_CASES[case], variant, monkeypatch).items():
    if TIER == "NEURON":
      print(f"tiled layer {dt} {case} {variant} {k} on NEURON: {np.mean(a.view(np.int32) == b.view(np.int32)):.6f} bitwise equal, max |diff| {np.abs(a - b).max():.3g}")
      np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-4)
    else: assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {variant} {dt} {k}: max |diff| {np.abs(a - b).max()}"

def test_tiled_layer_fits_sbuf_at_qwen7b():
  """The SBUF estimate: Qwen2.5-7B's untiled layer needs ~400 KiB per partition (the members run it); tiled it fits the
  176 KiB budget at 4 rows up to a 1024-slot prefix with 2048 weight columns, and at 2048 slots it does not (the members)."""
  q = lambda **k: nk.LayerSpec(4, 3584, 28, 4, 128, 18944, k.pop("P", 1024), 1e-6, 0.088, 0, heads="stack", early=True, **k)
  assert not nb.layer_fits(q()) and nb.layer_sbuf_bytes(q()) > 380 * 1024
  for wc in (512, 1024, 2048): assert nb.layer_fits(q(tile=True, wcols=wc)), wc
  assert not nb.layer_fits(q(P=2048, tile=True))
  assert q(tile=True, wcols=2048).name == "fz_layerset4_add_4_1024_3584_18944" and q(tile=True).name == "fz_layerset_add_4_1024_3584_18944"
  with pytest.raises(nk.NKIUnsupported): q(tile=True, wcols=768).render()
  with pytest.raises(nk.NKIUnsupported): nk.LayerSpec(1, 256, 4, 2, 64, 512, 64, 1e-6, 0.1, 0, rope="ap", tile=True).render()

def test_tiled_layer_bf16_edits():
  """LayerBF16 of the tiled layer: every weight DMA (qkv per slice, gate and up per block, o_proj and down per block) lands in a
  bf16 tile, every weight matmul's stationary is a bf16 copy made once, and the weight arguments are bf16."""
  sp = nb.LayerBF16(nk.LayerSpec(4, 3584, 28, 4, 128, 18944, 1024, 1e-6, 0.088, 0, heads="stack", early=True, tile=True, wcols=2048))
  prog = sp.render()
  src = prog.src
  assert "dtype=nl.float32" not in "".join(l for l in src.split("\n") if " bt" in l and "nl.ndarray" in l)
  assert sum(1 for l in src.split("\n") if l.strip().startswith(("btq =", "btu =", "bto =", "btd =")) and "bfloat16" in l) == 3 + 2 * 10 + 2 + 2
  assert [l.strip().split(" =")[0] for l in src.split("\n") if "_bf = nl.ndarray" in l] == ["at1_bf", "at2_bf", "at3_bf", "aT_bf"]
  assert all(prog.buf_dtypes[sp.args().index(w)] == dtypes.bfloat16 for w in ("wqkv", "wo", "wgu", "wd"))

@pytest.mark.parametrize("wcols", [512, 2048])
def test_tiled_layer_at_qwen7b_shapes(wcols, monkeypatch):
  """Qwen2.5-7B's full layer (D 3584, I 18944, 28 / 4 heads, head_dim 128), bf16, 4 rows, 512 slots: the tiled kernel gives
  the untiled LayerBF16's bits (NKISHIM has no SBUF limit, so the untiled one runs here; it is the members' bits by
  test_layer_bf16_matches_the_members) for every output."""
  if TIER != "NKISHIM": pytest.skip("the untiled 7B layer does not fit SBUF on a device")
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  N, D, Hq, Hkv, hd, I, P = 4, 3584, 28, 4, 128, 18944, 512
  Nq, C2, Wx = (Hq + 2 * Hkv) * hd, 2 * Hkv * hd, (Hq + 2 * Hkv + 1) * hd
  rng = np.random.default_rng(7)
  r = lambda n, sc: (rng.standard_normal(n) * sc).astype(np.float32)
  w = lambda n, sc: f32_to_bf16(r(n, sc))
  slots = np.array([300, 301, 302, 303])
  wt = np.zeros((N, P), np.float32); wt[np.arange(N), slots] = 1.0
  kid = np.full(P, -1, np.int32); kid[:304] = np.arange(304)
  meta = np.concatenate([np.array([0, 0, 0, 0], np.int32), np.array([300, 301, 302, 303], np.int32), kid])
  ang = rng.uniform(0, 3, (N, hd // 2)).astype(np.float32)
  cos, sin = np.concatenate([np.cos(ang)] * 2, -1).reshape(-1), np.concatenate([-np.sin(ang), np.sin(ang)], -1).reshape(-1)
  ins = {"x": r(N * D, 1.0), "d": r(N * D, 1.0), "ln1": 1 + r(D, 0.1), "wqkv": w(D * Nq, 0.02), "bqkv": r(Nq, 0.3), "cos": cos, "sin": sin,
         "wt": wt.reshape(-1), "meta": meta, "wo": w(Hq * hd * D, 0.02), "ln2": 1 + r(D, 0.1), "wgu": w(D * 2 * I, 0.02), "wd": w(I * D, 0.01)}
  pool = r(P * C2, 1.0)
  outs = {}
  for tile in (False, True):
    sp = nb.LayerBF16(nk.LayerSpec(N, D, Hq, Hkv, hd, I, P, 1e-6, 1 / np.sqrt(hd), 0, heads="stack", early=True, tile=tile, wcols=wcols))
    o = {"xo": np.full(N * D, np.nan, np.float32), "dn": np.full(N * D, np.nan, np.float32), "pool": pool.copy(),
         "xs": np.full(N * Wx, np.nan, np.float32), "os": np.full(N * Hq * hd, np.nan, np.float32)}
    res = _run(sp.render(), [o[a] if a in o else ins[a] for a in sp.args()])
    outs[tile] = dict(zip(sp.args()[:5], res[:5]))
  for k in outs[False]:
    a, b = outs[False][k], outs[True][k]
    assert np.isfinite(a).all() and np.array_equal(a.view(np.int32), b.view(np.int32)), f"{k}: max |diff| {np.abs(a - b).max()}"

@pytest.mark.parametrize("fuse", ["1", "24"])
def test_tiled_layer_through_the_model(tmp_path, monkeypatch, fuse):
  """Qwen2.5-7B's head layout at small widths, bf16: prefill and decode logits with every decode layer tiled
  (FZ_NEURON_LAYER_TILE=1, 2048 weight columns) are the members' (FZ_NEURON_LAYER=v1) bit for bit, fused per layer or all in one."""
  from test_tp import tiny_model_dir
  p = tiny_model_dir(str(tmp_path / "q7t"), seed=5, cfg={"hidden_size": 256, "intermediate_size": 640, "num_attention_heads": 28,
                                                          "num_key_value_heads": 4, "head_dim": 128})
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  monkeypatch.setenv("FZ_NEURON_FUSE", fuse)
  if TIER == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  out = {}
  for v in ("v1", "v2set4"):
    for k, val in nk.LAYER_VARIANTS[v].items(): monkeypatch.setenv(k, val)
    m = _load(p, TIER, 2)
    out[v] = [m.prefill([list(range(5 + b, 21 + b)) for b in range(2)], chunk=16)] + [m.decode([[3 + i + b] for b in range(2)]) for i in range(3)]
    names = []
    for it in m._neuron_steps.jits[(2, 2)].captured.items:
      if not it.const and (sp := getattr(it.si.ast[0], "spec", None)) is not None: names += [q[0] for q in sp.parts] if hasattr(sp, "parts") else [sp.name]
    assert sum(n.startswith("fz_layerset4") for n in names) == (m.cfg.n_layers if v != "v1" else 0), names
  for a, b in zip(out["v1"], out["v2set4"]): assert np.array_equal(a.view(np.int32), b.view(np.int32))
