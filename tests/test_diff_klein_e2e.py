"""scripts/diff_klein_e2e.py (I81 D8 / I95) on the tiny diffusers-layout klein dir of test_klein_image, on the CPU (and METAL): every
phase (diffusers bf16 baseline, FuseZap from the prompt resident and staged, FuseZap from reference embeds) runs in its
subprocess and report.json carries G6 rows, timings and memory; a failing FuseZap smoke skips the other FuseZap phases
and still keeps the diffusers baseline. The 4B run is the CUDA job (scripts/diff_klein_cuda_job.sh) / the Metal one."""
from __future__ import annotations
import json, os, sys
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("diffusers")
pytest.importorskip("tokenizers")
from test_klein_image import klein_dir  # noqa: F401  (fixture)

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
sys.path.insert(0, SCRIPTS)
import diff_klein_e2e as e2e  # noqa: E402
SIZE = 32   # tiny VAE scale 2 x patch 2 -> an 8x8 token grid

@pytest.fixture(scope="module")
def refs(klein_dir, tmp_path_factory):  # noqa: F811
  """A float32 CPU reference in diff_ref's layout (decode PNG + embeds) for prompt 0, seed 0, from the stock pipeline."""
  from PIL import Image
  root = str(tmp_path_factory.mktemp("refs"))
  os.makedirs(os.path.join(root, "f32", "decode")); os.makedirs(os.path.join(root, "f32", "embeds"))
  pipe, _ = e2e._diffusers(klein_dir, torch.float32, "cpu")
  with torch.no_grad():
    emb = pipe.encode_prompt(prompt=e2e.PROMPTS[0], device="cpu")[0]
    img = pipe(prompt=e2e.PROMPTS[0], latents=torch.tensor(e2e.noise(0, klein_dir, SIZE)), height=SIZE, width=SIZE,
               num_inference_steps=4, output_type="np").images
  np.savez(os.path.join(root, "f32", "embeds", "p0.npz"), embeds=emb[0].float().numpy())
  Image.fromarray(e2e._np_uint8(img), "RGB").save(os.path.join(root, "f32", "decode", f"{SIZE}_p0_s0.png"))
  return root

def _run(klein_dir, refs, out, *extra) -> tuple[int, dict]:  # noqa: F811
  with pytest.raises(SystemExit) as ex:
    e2e.main(["--device", "CPU", *extra, "--model", klein_dir, "--size", str(SIZE), "--prompts", "0", "--ref", "f32", "--ref-root", refs,
              "--out", str(out)])
  return ex.value.code, json.load(open(os.path.join(out, "report.json")))

def test_noise_matches_diff_ref():
  import diff_ref, tempfile
  # klein's layout without the 4B weights: transformer in_channels 128, a 4-level VAE
  d = os.path.join(tempfile.mkdtemp(), "k")
  os.makedirs(os.path.join(d, "transformer")); os.makedirs(os.path.join(d, "vae"))
  json.dump({"in_channels": 128}, open(os.path.join(d, "transformer", "config.json"), "w"))
  json.dump({"block_out_channels": [128, 256, 512, 512]}, open(os.path.join(d, "vae", "config.json"), "w"))
  assert e2e.latent_layout(d) == (128, 16)
  np.testing.assert_array_equal(e2e.noise(3, d, 256), diff_ref.noise(3, 256))

def _cases() -> list[tuple[str, str]]:
  from fusezap.device import Device
  return [("CPU", "resident"), ("CPU", "staged")] + ([("METAL", "resident")] if Device.available("METAL") else [])

@pytest.mark.parametrize("fzdev,encode", _cases())
def test_e2e_all_phases(klein_dir, refs, tmp_path, fzdev, encode):  # noqa: F811
  code, rep = _run(klein_dir, refs, tmp_path / "out", "--fz-encode", encode, "--device", fzdev)
  assert rep["phase_status"] == {"df-bf16": "ok", "fz-bf16": "ok", "fz-bf16-ref": "ok"}, rep.get("errors")
  (row,) = rep["rows"]
  for tag in ("df-bf16", "fz-bf16", "fz-bf16-ref"):
    assert np.isfinite(row[f"{tag}.psnr"]) and row[f"{tag}.psnr"] > 10, (tag, row)
    s = rep["summary"][tag]
    assert s["n"] == 1 and s["s_per_image_mean"] > 0 and s["peak_footprint_gb"] > 0
  assert isinstance(row["G6.pass"], bool) and code == (0 if row["G6.pass"] else 1)
  assert rep["fz_encode"] == encode and rep["fz-bf16.timing"]["fz_encode"] == encode
  assert rep["summary"]["fz-bf16"]["encode_s"] > 0 and "smoke_s" in rep["fz-bf16.timing"]

def test_e2e_smoke_failure_keeps_baseline(klein_dir, refs, tmp_path, monkeypatch):  # noqa: F811
  # FZ_KLEIN_TE_DTYPE=nope: the staged FuseZap encoder cannot be built -> fz-bf16 fails before its smoke image
  monkeypatch.setenv("FZ_KLEIN_TE_DTYPE", "nope")
  code, rep = _run(klein_dir, refs, tmp_path / "out", "--phases", "df-bf16,fz-bf16", "--fz-encode", "staged")
  assert code == 1 and rep["pass"] is False
  assert rep["phase_status"]["df-bf16"] == "ok" and rep["phase_status"]["fz-bf16"].startswith("failed at staged-encode")
  assert "df-bf16.psnr" in rep["rows"][0] and rep["rows"][0]["G6.pass"] is None
  assert "AttributeError" in rep["errors"]["fz-bf16"]["error"] or "nope" in rep["errors"]["fz-bf16"]["error"]

def test_e2e_smoke_failure_skips_fz(klein_dir, refs, tmp_path, monkeypatch):  # noqa: F811
  monkeypatch.setenv("FZ_E2E_SMOKE_SIZE", "30")   # 30 is not a multiple of the tiny model's 4: check_size raises at smoke
  code, rep = _run(klein_dir, refs, tmp_path / "out", "--phases", "df-bf16,fz-bf16,fz-bf16-ref")
  assert code == 1
  assert rep["phase_status"]["fz-bf16"].startswith("failed at smoke")
  assert rep["phase_status"]["fz-bf16-ref"] == "skipped (fz-bf16 smoke failed)"
  assert rep["phase_status"]["df-bf16"] == "ok" and "df-bf16.psnr" in rep["rows"][0]
