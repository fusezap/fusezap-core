"""I105 B2: renderer kernels over the DMA descriptor budget run as several launches (fusezap/codegen/nki.py dma_budget,
NKIRenderer.render_split; fusezap/device/neuron.py SplitProgram). The inf2 chip run of 2026-11-14 could not load the float32
Gemma 3 server: kmajor's transpose of the 262144 x 1152 lm_head (302 M element-granular descriptors in one NEFF) hit
`TDRV: Descriptor limit reached` (16,777,200 per ring). These tests: the estimate and the split at that shape (render
only, no data), and on NKISHIM that split launches give the whole kernel's bits.
"""
from __future__ import annotations
import numpy as np
import pytest
from fusezap import Tensor, dtypes
from fusezap.codegen.nki import DMA_RING_MAX, DMA_RINGS, NKIRenderer, dma_budget, dma_part_budget
from fusezap.device import Device
from fusezap.schedule import create_schedule

@pytest.fixture
def shim(monkeypatch):
  from fusezap import helpers
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.delenv("FZ_NEURON_DMA_SPLIT", raising=False)
  monkeypatch.delenv("FZ_NEURON_DMA_DESC", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "NKISHIM", raising=False)
  yield Device["NKISHIM"]
  helpers.getenv.cache_clear()

def _kernel(t: Tensor):
  (si,) = [si for si in create_schedule([t.lazydata]) if si.kind == "kernel"]
  return si

def test_gemma_head_transpose_estimate_and_split(shim):
  """kmajor(lm_head) at Gemma 3's float32 head: one strided descriptor per element (302,579,712 with the stores' rows),
  18.9 M per ring > the 16,777,200 limit; split into 3 launches of 384 partition rows, each 6.3 M per ring."""
  w = Tensor.empty(262144, 1152, dtype=dtypes.float32, device="NKISHIM")
  si = _kernel(w.transpose(-1, -2).contiguous())
  whole, parts = NKIRenderer().render_split(si.ast, si.var_vals)
  assert whole.dma_desc == 262144 * 1152 + 1152 * 512   # loads: one per element; stores: one per partition row per tile
  assert whole.dma_desc / DMA_RINGS > DMA_RING_MAX       # the chip run's failure
  assert dma_budget() == DMA_RINGS * (DMA_RING_MAX * 3 // 4)                  # split above 3/4 of the ring limit
  assert len(parts) == 3 and all(p.dma_desc / DMA_RINGS <= DMA_RING_MAX // 2 for p in parts)   # parts under half of it
  assert all(p.dma_desc <= dma_part_budget(dma_budget()) for p in parts)
  assert sum(p.dma_desc for p in parts) == whole.dma_desc
  assert [next(l.strip() for l in p.src.splitlines() if "for p0" in l) for p in parts] == \
         [f"for p0 in range({a}, {b}, 128):" for a, b in ((0, 384), (384, 768), (768, 1152))]
  assert whole.src == NKIRenderer().render(si.ast, si.var_vals).src   # the unsplit source is the renderer's, unchanged

def test_kernels_that_loaded_on_chips_are_not_split(shim):
  """Under the trigger nothing changes: no parts, the same source as before (so the same NEFFs). Gemma's float32 layer
  transposes (gate_up 13824 x 1152: ~1.0 M per ring), Qwen2.5-0.5B's float32 kmajor(lm_head) (151936 x 896: 8.5 M per ring,
  loaded on trn1) and the scorer's 256-row log_softmax over 262144 (8.4 M per ring, loaded on inf2 in chip run 1)."""
  for shape in ((13824, 1152), (1152, 13824), (151936, 896)):
    si = _kernel(Tensor.empty(*shape, dtype=dtypes.float32, device="NKISHIM").transpose(-1, -2).contiguous())
    whole, parts = NKIRenderer().render_split(si.ast, si.var_vals)
    assert parts == [] and 0 < whole.dma_desc <= dma_budget()
  x = Tensor.empty(256, 262144, dtype=dtypes.float32, device="NKISHIM")
  m, l = (Tensor.empty(256, 1, dtype=dtypes.float32, device="NKISHIM") for _ in range(2))
  si = _kernel(x - m - l)   # log_softmax's epilogue: two (rows, 1) operands broadcast along the vocabulary (stride 0)
  whole, parts = NKIRenderer().render_split(si.ast, si.var_vals)
  assert parts == [] and whole.dma_desc / DMA_RINGS > 8_000_000 and whole.dma_desc <= dma_budget()

@pytest.mark.parametrize("desc", ["2000", "10000"])
def test_split_transpose_is_bitwise(shim, monkeypatch, desc):
  monkeypatch.setenv("FZ_NEURON_DMA_DESC", desc)   # per ring; x 16 rings = the launch budget
  a = np.random.default_rng(0).standard_normal((1000 + int(desc) % 7, 300)).astype(np.float32)
  a[3, 5], a[7, 1] = -0.0, np.nan
  shim._cache.clear()
  out = Tensor(a).transpose(-1, -2).contiguous().realize()
  from fusezap.device.neuron import SplitProgram
  sp = [p for p in shim._cache.values() if isinstance(p, SplitProgram)]
  assert len(sp) == 1 and len(sp[0].parts) >= 2
  assert out.numpy().tobytes() == np.ascontiguousarray(a.T).tobytes()

def test_split_outer_loop_and_reduce(shim, monkeypatch):
  """A 3-d permute (an outer loop: split on i0) and a row reduce over a strided view (split on the partition tiles only,
  never on the reduce axis) give the unsplit kernel's bits."""
  rng = np.random.default_rng(1)
  a = rng.standard_normal((6, 200, 130)).astype(np.float32)
  b = rng.standard_normal((700, 260)).astype(np.float32)
  monkeypatch.setenv("FZ_NEURON_DMA_SPLIT", "0")
  shim._cache.clear()
  ref_p = Tensor(a).permute(0, 2, 1).contiguous().numpy()
  ref_r = Tensor(b).transpose(-1, -2).sum(axis=1).numpy()
  monkeypatch.setenv("FZ_NEURON_DMA_SPLIT", "1")
  monkeypatch.setenv("FZ_NEURON_DMA_DESC", "1000")
  shim._cache.clear()
  got_p = Tensor(a).permute(0, 2, 1).contiguous().numpy()
  got_r = Tensor(b).transpose(-1, -2).sum(axis=1).numpy()
  from fusezap.device.neuron import SplitProgram
  sps = [p for p in shim._cache.values() if isinstance(p, SplitProgram)]
  assert len(sps) == 2
  assert ref_p.tobytes() == got_p.tobytes() == np.ascontiguousarray(a.transpose(0, 2, 1)).tobytes()
  assert ref_r.tobytes() == got_r.tobytes()
  import re
  for p in sps:   # parts cut an outer loop or the partition tiles; a reduce kernel's free (reduce) loop always starts at 0
    for part in p.parts: assert all(m == "0" for m in re.findall(r"for f0 in range\((\d+),", part.prog.src))

def test_kmajor_float32_on_the_host_or_split(shim, monkeypatch):
  """neuron.kmajor (the load-time relayout that failed on the chip): float32 is relaid on the host (no device kernel at
  all); FZ_NEURON_KMAJOR_HOST=0 keeps the device transpose, through split launches when over the budget. Same bits."""
  from fusezap.device.neuron import SplitProgram, kmajor
  w = np.random.default_rng(2).standard_normal((2048, 384)).astype(np.float32)
  w[0, 0], w[5, 9] = -0.0, np.inf
  want = np.ascontiguousarray(w.T).tobytes()
  t = Tensor(w).realize()
  shim._cache.clear()
  assert kmajor(t).numpy().tobytes() == want and not shim._cache   # host: nothing rendered
  monkeypatch.setenv("FZ_NEURON_KMAJOR_HOST", "0")
  monkeypatch.setenv("FZ_NEURON_DMA_DESC", "4096")
  assert kmajor(t).numpy().tobytes() == want
  assert [len(p.parts) for p in shim._cache.values() if isinstance(p, SplitProgram)] == [4]   # the 4 free-axis tiles (most trips)

def test_float32_o_proj_reaches_tensore(shim, tmp_path):
  """Llama.forward on NEURON float32 (the logprobs scorer's trunk): o_proj's input is materialized, so `o @ wo` is the
  fz_matmul custom item, not a composed MUL+SUM kernel r_<rows>_<D>_<Hq*hd> (Gemma 3's 256-row scorer bucket: 302 M
  descriptors in one NEFF). Tiny random Gemma 3 (D = Hq * hd = 128); the scorer's rows are finite log-probabilities."""
  import re
  pytest.importorskip("torch")
  pytest.importorskip("transformers")
  import sys, os
  sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
  from test_swa import _model_dir
  from fusezap.codegen import nki
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine
  from fusezap.serve.logprobs import scorer
  m = load_model(_model_dir(tmp_path / "g", "gemma3", seed=6), dtype=dtypes.float32, device="NKISHIM", max_len=64, prefill_chunk=16)
  names: list[str] = []
  orig = nki.NKIRenderer._render
  def rec(self, ast, vv, name, split):
    p, k = orig(self, ast, vv, name, split)
    names.append(p.name)
    return p, k
  nki.NKIRenderer._render = rec
  try:
    eng = LLMEngine(m, num_blocks=8, block_size=16, max_num_seqs=1, max_prefill_tokens=16, max_model_len=64, decode_beam=0)
    rows = scorer(eng).logprobs(list(range(3, 40)))
  finally: nki.NKIRenderer._render = orig
  assert rows.shape == (37, 512) and np.isfinite(rows).all()
  assert names and not [n for n in names if re.fullmatch(r"r_\d+_128_128", n)], sorted(set(names))
