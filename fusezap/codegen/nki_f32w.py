"""W15 (trn1, float32 served c4 vs AWS vLLM + vllm-neuron): the float32 decode layer's weights streamed in wide tiles.

The public Status row "Trainium1, output tokens/s with 4 users (Qwen2.5-0.5B, fp32)" cites N7 serve (27 Oct, 198.6 tok/s
against AWS vLLM 0.16 + vllm-neuron 0.5.3 TP=1 float32 on-device greedy, 238.4). N7 step 4 (28 Oct, the trn1
step 4 run) already served 303.8 tok/s with v2se + one launch per token, and that became the
float32 default (N8 base); the matrix row was never re-pointed. What is left in a float32 decode layer (trn1 step 4: v2se
0.314 ms per layer at 1 row / 512 slots, 0.355 at 4 rows / 1024, ~60 MB of weights, ~190 GB/s): the weight DMAs keep
render_matmul's shape, one (128, 512) float32 tile per K chunk = 2 KiB runs per partition. N8 measured, in bf16 on the same
chip, that wider runs of the same bytes (FZ_NEURON_BF16_WN 4096: 8 KiB runs) took the bf16 layer from 0.234 to 0.178 ms, bitwise.

  FZ_NEURON_F32_WN=<cols>    (default 512 = off) LayerSpec's four weight matmuls (qkv, o_proj, gate_up, down) load one
                             (128, <cols>) float32 tile per K chunk (1024 / 2048 / 4096 columns: 4-16 KiB runs) and feed its
                             512-column slices to the same nc_matmul, each into its own PSUM tile: the transform N8 wrote for
                             LayerBF16 (nki_bf16_n8.wide_weights; its block pattern matches LayerSpec's float32 blocks too), so
                             every PSUM chain (one 512-column output block over the K chunks in order) is the same instructions on
                             the same values: the same bits by construction (tests/test_neuron_f32w.py on NKISHIM).

Not a default: on trn1 the float32 lm_head got slower with wider blocks (N7 step 2b: HeadSpec W 1024 2.06 ms, 2048 2.71,
4096 3.85), so wider float32 runs are not known to help; the job (scripts/nki_w15_trn1_c4.sh) measures 1024 / 2048 / 4096
in the layer bench, checks the bits on the device, and serves the fastest bitwise width next to the default.
Only a whole untiled layer: the tiled layer (I71: its own FZ_NEURON_LAYER_WCOLS) and the TP halves (I72) come back unchanged.
SBUF: two (128, cols) float32 tiles in flight add 2 * 4 * (cols - 512) bytes per partition to nki_bf16.layer_sbuf_bytes; the
width steps down (4096 -> 2048 -> 1024 -> off) until that fits nki_bf16.LAYER_SBUF. PSUM: cols / 512 <= 8 (M, 512) float32
tiles per block, the 8 banks (as N8's 4096, validated on trn1).
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from fusezap.codegen.nki import NKIProgram

TN = 512
F32_ENV = ("FZ_NEURON_F32_WN",)

def f32_wn() -> int:
  """FZ_NEURON_F32_WN: columns per float32 weight DMA in LayerSpec (a multiple of 512 in [512, 4096]; unset or 512: off)."""
  v = os.environ.get("FZ_NEURON_F32_WN", "").strip()
  wn = int(v) if v else TN
  if wn % TN or not TN <= wn <= 8 * TN: raise ValueError(f"FZ_NEURON_F32_WN={wn}: a multiple of {TN} in [{TN}, {8 * TN}]")
  return wn

def wide_sbuf_bytes_f32(wn: int) -> int:
  """SBUF bytes per partition the wide float32 moving tiles add over the 512-column ones (two (128, wn) tiles in flight)."""
  return 2 * 4 * (wn - TN)

@dataclass(frozen=True)
class LayerF32W:
  """nki_kernels.LayerSpec (float32 weights) with its weight matmuls on wn-column moving tiles: the same arguments, the same bits."""
  base: object   # nki_kernels.LayerSpec
  wn: int
  @property
  def name(self) -> str: return self.base.name + f"_w{self.wn}"
  def args(self) -> tuple[str, ...]: return self.base.args()
  @property
  def heads(self) -> str: return self.base.heads
  def render(self) -> NKIProgram:
    from fusezap.codegen.nki_bf16_n8 import wide_weights
    return wide_weights(self.base.render(), self.wn, 0, False, f"_w{self.wn}")

def layer_f32w(sp, wn: int | None = None):
  """NeuronKVCache.layer's float32 hook (LayerSpec in, the kernel spec out): LayerF32W when FZ_NEURON_F32_WN asks for it and
  the layer is whole and untiled, at the widest width <= the one asked whose SBUF estimate fits; else sp unchanged."""
  from fusezap.codegen.nki_bf16 import LAYER_SBUF, layer_sbuf_bytes
  if getattr(sp, "tile", False) or getattr(sp, "part", ""): return sp
  wn = f32_wn() if wn is None else wn
  while wn > TN and layer_sbuf_bytes(sp) + wide_sbuf_bytes_f32(wn) > LAYER_SBUF: wn = wn // 2 // TN * TN
  return sp if wn <= TN else LayerF32W(sp, wn)
