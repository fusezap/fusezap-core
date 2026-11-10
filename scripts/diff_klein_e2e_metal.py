"""I81 D8 end to end on METAL: scripts/diff_klein_e2e.py with --device METAL and --out results/diff/e2e by default (the
device-generic driver: phases df-bf16 / fz-bf16 / fz-bf16-ref, Gate G6, report.json; see its docstring).

  uv run python scripts/diff_ref.py denoise --sizes 1024 --prompts 0-3 --seeds 0
  uv run python scripts/diff_ref.py decode  --sizes 1024 --prompts 0-3 --seeds 0
  uv run python scripts/diff_klein_e2e_metal.py [--out results/diff/e2e] [--size 1024] [--prompts 0-3] [--seeds 0]
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_klein_e2e import main, psnr, ssim, peak_footprint_gb, REPO   # noqa: F401  (the old module's names)

if __name__ == "__main__": main(default_device="METAL", default_out="results/diff/e2e")
