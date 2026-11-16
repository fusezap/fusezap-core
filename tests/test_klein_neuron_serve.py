"""I93: FLUX.2 [klein] served on a NEURON tier (NKISHIM here: the numpy model of nki 0.6), the placement of
`fusezap serve-image flux2-klein --device NEURON --dtype bfloat16`: the transformer on the device in bf16, the text encoder
(transformers' Qwen3) and the VAE (diffusers' AutoencoderKLFlux2, TorchVAEDecoder) on the host CPU in float32.

On the tiny diffusers-layout klein of tests/test_klein_image.py:
- TorchVAEDecoder == FuseZap's float32 VAE on CPU (the host decode is the same function);
- KleinImagePipeline on NKISHIM picks hf / torch, one image per step, K-major bf16 weights; ImageEngine's image is close to
  the same placement with the transformer on CPU in bf16, and bitwise unchanged by the I93 switches (ROPE, MMG);
- the OpenAI images endpoint serves it (starlette TestClient) with per-image timings.
"""
from __future__ import annotations
import base64, io, os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("tokenizers")
from test_klein_image import klein_dir, PROMPT, H, W   # noqa: F401  (the module-scoped tiny model fixture)

DEV = "NKISHIM"

def test_torch_vae_matches_fusezap_cpu(klein_dir):
  from fusezap.dtype import dtypes
  from fusezap.tensor import Tensor
  from fusezap.serve.models.flux2_vae import Flux2VAEDecoder
  from fusezap.serve.models.klein_image import TorchVAEDecoder
  tv = TorchVAEDecoder(klein_dir)
  fz = Flux2VAEDecoder.from_pretrained(klein_dir, dtypes.float32, "CPU")
  h = w = 4
  x = np.random.default_rng(0).standard_normal((1, h * w, 16)).astype(np.float32)   # the transformer's 16 packed channels
  a = tv.decode_packed(Tensor(x, device="CPU"), h, w).numpy()
  b = fz.decode_packed(Tensor(x, device="CPU"), h, w).float().numpy()
  assert a.shape == b.shape and a.shape[:2] == (1, 3)
  assert np.abs(a - b).max() <= 1e-4 * max(1.0, float(np.abs(b).max())), np.abs(a - b).max()

@pytest.fixture(scope="module")
def neuron_pipe(klein_dir):
  from fusezap.serve.models.klein_image import KleinImagePipeline
  return KleinImagePipeline(klein_dir, device=DEV, dtype="bfloat16")

def _image(pipe, seed=5, steps=4):
  from fusezap.serve.image_engine import ImageEngine, ImageParams
  eng = ImageEngine(pipe, max_batch=getattr(pipe, "max_batch_cap", 1), batch_sizes=(1,))
  (r,), = eng.generate([ImageParams(prompt=PROMPT, seed=seed, width=W, height=H, steps=steps)])
  return r

def test_neuron_placement(neuron_pipe):
  p = neuron_pipe
  from fusezap.serve.models.klein_image import TorchVAEDecoder
  from fusezap.serve.models.flux2_klein import HFQwen3PromptEncoder
  assert p.encoder_kind == "hf" and isinstance(p.encoder, HFQwen3PromptEncoder)
  assert p.vae_kind == "torch" and isinstance(p.vae, TorchVAEDecoder) and p.pipe.vae_scale == 2
  assert p.max_batch_cap == 1 and p.tr.kmajor and p.tr.device == DEV

def test_neuron_image_close_to_cpu_bf16(klein_dir, neuron_pipe):
  """The NKISHIM image vs the same placement with the transformer on CPU in bf16 (MatmulBF16 rounds only the matmul
  operands, CPU bf16 also the activations between ops): the bf16 tiny-model tolerance, not bitwise."""
  from fusezap.serve.models.klein_image import KleinImagePipeline
  r = _image(neuron_pipe)
  assert r.image.shape == (H, W, 3) and r.image.std() > 5
  assert set(r.timings) >= {"encode_s", "denoise_s", "decode_s", "total_s"}
  cpu = KleinImagePipeline(klein_dir, device="CPU", dtype="bfloat16", encoder="hf", vae="torch")
  ref = _image(cpu)
  d = np.abs(r.image.astype(int) - ref.image.astype(int))
  assert d.mean() <= 2.0 and (d <= 8).mean() >= 0.98, (d.mean(), d.max())

def test_neuron_switches_bitwise(neuron_pipe, monkeypatch):
  """FZ_KLEIN_NEURON_ROPE=1 and FZ_KLEIN_NEURON_MMG=4 (both bitwise by construction) leave the served image unchanged."""
  base = _image(neuron_pipe, seed=7, steps=2).image
  monkeypatch.setenv("FZ_KLEIN_NEURON_ROPE", "1"); monkeypatch.setenv("FZ_KLEIN_NEURON_MMG", "4")
  assert np.array_equal(_image(neuron_pipe, seed=7, steps=2).image, base)

def test_images_endpoint_on_neuron(neuron_pipe):
  pytest.importorskip("starlette"); pytest.importorskip("httpx")
  from PIL import Image
  from starlette.testclient import TestClient
  from fusezap.serve.image_api import create_image_app
  from fusezap.serve.image_engine import ImageEngine
  eng = ImageEngine(neuron_pipe, max_batch=neuron_pipe.max_batch_cap, batch_sizes=(1,))
  with TestClient(create_image_app(eng, "flux2-klein")) as c:
    r = c.post("/v1/images/generations", json={"prompt": PROMPT, "size": f"{W}x{H}", "seed": 5, "num_inference_steps": 4})
    assert r.status_code == 200, r.text
    j = r.json()
    d = j["data"][0]
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(d["b64_json"]))).convert("RGB"))
    assert img.shape == (H, W, 3) and d["seed"] == 5 and j["fusezap"]["num_inference_steps"] == 4
    assert {"encode_s", "denoise_s", "decode_s", "total_s"} <= set(d["fusezap_timings"])
    assert np.array_equal(img, _image(neuron_pipe).image)   # the served PNG is the engine's image
    m = c.get("/metrics").text
    assert "fusezap:images_generated_total" in m

def test_serve_image_caps_batch(monkeypatch):
  """serve-image clamps --max-batch / --batch-sizes to the pipeline's max_batch_cap (klein on NEURON: 1)."""
  pytest.importorskip("uvicorn")
  import types
  from fusezap.serve import image_cli, image_api, image_engine
  seen = {}
  class P:
    device, name, default_size, default_steps, max_batch_cap = "NEURON", "x", (64, 64), 4, 1
    def velocity(self, *a): pass
  class E:
    def __init__(self, pipe, max_batch, batch_sizes): seen.update(mb=max_batch, bs=batch_sizes); self.name, self.device = "x", "NEURON"
  monkeypatch.setattr(image_api, "load_pipeline", lambda *a, **k: P())
  monkeypatch.setattr(image_engine, "ImageEngine", E)
  import uvicorn
  monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
  monkeypatch.setattr(image_api, "create_image_app", lambda *a, **k: None)
  a = types.SimpleNamespace(pipeline="flux2-klein", device="NEURON", dtype="bfloat16", batch_sizes="1,2,4", max_batch=4, no_warmup=True,
                            served_model_name=None, api_key=None, host="127.0.0.1", port=0, log_level="info", warmup_sizes="")
  assert image_cli.serve_image(a) == 0
  assert seen == {"mb": 1, "bs": [1]}

def _batch_images(pipe, B: int, seed: int, n: int, steps: int = 2) -> list:
  from fusezap.serve.image_engine import ImageEngine, ImageParams
  eng = ImageEngine(pipe, max_batch=B, batch_sizes=(B,))
  (rs,) = eng.generate([ImageParams(prompt=PROMPT, seed=seed, width=W, height=H, steps=steps)], n=n)
  assert eng.last_step["batch"] == min(n, B) and eng.last_step["bucket_batch"] == B
  return [r.image for r in rs]

@pytest.mark.parametrize("switches", [{}, {"FZ_KLEIN_NEURON_ROPE": "1", "FZ_KLEIN_NEURON_MMG": "4"}], ids=["main", "rope-mmg"])
def test_neuron_batch4_equals_four_single_images(neuron_pipe, monkeypatch, switches):
  """I93 batch 4 (images API n=4, serve-image --max-batch 4 with FZ_KLEIN_NEURON_BATCH=4): one request of 4 images, run as
  one batch-4 step, gives the 4 images of seeds s .. s+3 run one per step, bit for bit (every row of a step is computed
  independently: the matmul rows, the per-image attention, the per-row norms; on NKISHIM with the order-independent matmul,
  FZ_NKISHIM_MM=f64, because BLAS rounds a 4-image row block like any other operand only up to its kernel's blocking).
  A 3-image request in the batch-4 bucket (one padding row) gives the same first 3."""
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  for k, v in switches.items(): monkeypatch.setenv(k, v)
  singles = [_batch_images(neuron_pipe, 1, 21 + k, 1)[0] for k in range(4)]
  four = _batch_images(neuron_pipe, 4, 21, 4)
  for k in range(4): assert np.array_equal(four[k], singles[k]), f"image {k}: max |diff| {np.abs(four[k].astype(int) - singles[k].astype(int)).max()}"
  three = _batch_images(neuron_pipe, 4, 21, 3)
  for k in range(3): assert np.array_equal(three[k], singles[k])

def test_neuron_batch_cap_from_env(klein_dir, monkeypatch):
  """FZ_KLEIN_NEURON_BATCH=4 lifts klein's NEURON cap (serve-image then keeps --max-batch 4 --batch-sizes 1,4)."""
  from fusezap.serve.models.klein_image import KleinImagePipeline
  monkeypatch.setenv("FZ_KLEIN_NEURON_BATCH", "4")
  assert KleinImagePipeline(klein_dir, device=DEV, dtype="bfloat16").max_batch_cap == 4

def test_speed_script_on_nkishim(klein_dir, tmp_path):
  """scripts/i93_klein_speed.py end to end on NKISHIM with the tiny model and made-up refs: every variant runs, the
  bitwise variants match the base, the JSON has s/step, launches and G3."""
  import json, subprocess, sys
  rng = np.random.default_rng(0)
  refs = tmp_path / "refs"
  (refs / "denoise").mkdir(parents=True); (refs / "embeds").mkdir()
  size, L = 64, (64 // 16) ** 2
  np.savez(refs / "denoise" / f"{size}_p0_s0.npz", x=rng.standard_normal((5, L, 16)).astype(np.float32),
           v=rng.standard_normal((4, L, 16)).astype(np.float32), sigmas=np.array([1, .75, .5, .25, 0], np.float32),
           timesteps=np.array([1000, 750, 500, 250], np.float32))
  np.savez(refs / "embeds" / "p0.npz", embeds=rng.standard_normal((8, 96)).astype(np.float32))
  out = tmp_path / "speed.json"
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  r = subprocess.run([sys.executable, os.path.join(root, "scripts", "i93_klein_speed.py"), "--model", klein_dir, "--refs", str(refs),
                      "--sizes", str(size), "--device", DEV, "--steps", "2", "--json", str(out),
                      "--variants", "main=;rope=FZ_KLEIN_NEURON_ROPE=1;ropemmg=FZ_KLEIN_NEURON_ROPE=1,FZ_KLEIN_NEURON_MMG=4",
                      "--bitwise", "rope,ropemmg", "--profile", "main"],
                     capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=root), timeout=600)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
  j = json.loads(out.read_text())
  assert [x["variant"] for x in j["runs"]] == ["main", "rope", "ropemmg"] and j["bitwise_ok"]
  assert all(x["launches"][0] > 0 and len(x["G3"]) == 2 and x["s_per_step"] > 0 for x in j["runs"])
  assert j["runs"][1]["launches"][0] < j["runs"][0]["launches"][0]   # RopeSpec: fewer launches

def test_serve_bench_script_on_nkishim(klein_dir, tmp_path):
  """scripts/i93_klein_serve_bench.py: serve-image flux2-klein on NKISHIM (the tiny model) in a subprocess, two requests,
  per-image wall latency and the server's timings in the JSON, the PNG saved."""
  pytest.importorskip("uvicorn")
  import json, socket, subprocess, sys
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  out = tmp_path / "serve"
  env = dict(os.environ, PYTHONPATH=root, FZ_KLEIN_PATH=klein_dir, FZ_KLEIN_NEURON_ROPE="1")
  r = subprocess.run([sys.executable, os.path.join(root, "scripts", "i93_klein_serve_bench.py"), "--device", DEV, "--sizes", str(W),
                      "--requests", "2", "--steps", "2", "--port", str(port), "--out", str(out), "--startup-s", "300"],
                     capture_output=True, text=True, env=env, timeout=600)
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:] + (out / "serve.log").read_text()[-2000:]
  j = json.loads((out / "serve_bench.json").read_text())
  s = j["sizes"][str(W)]
  assert len(s["requests"]) == 2 and s["summary"]["median_wall_s"] > 0 and "median_decode_s" in s["summary"]
  assert (out / f"served_{W}.png").exists() and j["env"]["FZ_KLEIN_NEURON_ROPE"] == "1"
  assert "max-batch 4 -> 1" not in (out / "serve.log").read_text()   # --max-batch 1 asked explicitly

def test_serve_bench_batch_and_soak_on_nkishim(klein_dir, tmp_path):
  """scripts/i93_klein_serve_bench.py --batch 4 --n-list 1,4 --soak-minutes: the server keeps --max-batch 4 (FZ_KLEIN_NEURON_BATCH),
  an n=4 request returns 4 images, and the soak repeats every (size, n) pair with byte-identical images, no failures."""
  pytest.importorskip("uvicorn")
  import json, socket, subprocess, sys
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  out = tmp_path / "serve_b4"
  env = dict(os.environ, PYTHONPATH=root, FZ_KLEIN_PATH=klein_dir, FZ_KLEIN_NEURON_ROPE="1")
  r = subprocess.run([sys.executable, os.path.join(root, "scripts", "i93_klein_serve_bench.py"), "--device", DEV, "--sizes", str(W),
                      "--requests", "1", "--steps", "2", "--port", str(port), "--out", str(out), "--startup-s", "300",
                      "--batch", "4", "--n-list", "1,4", "--soak-minutes", "0.15"],
                     capture_output=True, text=True, env=env, timeout=900)
  log = (out / "serve.log").read_text()
  assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:] + log[-2000:]
  j = json.loads((out / "serve_bench.json").read_text())
  assert j["batch"] == 4 and "--max-batch" in j["cmd"] and j["cmd"][j["cmd"].index("--batch-sizes") + 1] == "1,4" and "max-batch 4 -> 1" not in log
  assert j["sizes"][f"{W}_n4"]["summary"]["median_s_per_image"] > 0 and len(j["sizes"][f"{W}_n4"]["requests"][0]["timings"]) == 4
  assert all((out / f"served_{W}_n4_{k}.png").exists() for k in range(4))
  sm = j["soak"]["summary"]
  assert sm["requests"] >= 2 and sm["failures"] == 0 and sm["mismatched"] == 0 and sm["images"] >= 1, sm
