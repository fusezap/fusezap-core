"""I81 D7: ImageEngine (fusezap/serve/image_engine.py) and /v1/images/generations (fusezap/serve/image_api.py) with the
tiny random flow pipeline (tests/tiny_diffusion.py), driven in-process and through the official openai client against a
live uvicorn server."""
from __future__ import annotations
import asyncio, base64, io, os, socket, subprocess, sys, threading, time
import numpy as np
import pytest
from fusezap.device import Device
from fusezap.serve.image_engine import ImageEngine, ImageParams, reference_image, to_uint8

sys.path.insert(0, os.path.dirname(__file__))
from tiny_diffusion import TinyFlow, TinyFlowMonolithic  # noqa: E402

def _devices() -> list[str]:
  if os.environ.get("FZ_DEVICE"): return [os.environ["FZ_DEVICE"]]
  return ["CPU"] + (["METAL"] if Device.available("METAL") else [])

def _png(b64: str) -> np.ndarray:
  from PIL import Image
  return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB"))

# ---------------------------------------------------------------- engine

def test_to_uint8_matches_diffusers_postprocess():
  x = np.array([[-1.5, -1.0, -0.002, 0.0, 0.5, 1.0, 2.0]], np.float32).reshape(1, 1, 7).repeat(3, 0)
  got = to_uint8(x)
  assert got.shape == (1, 7, 3)
  assert got[0, :, 0].tolist() == [0, 0, 127, 128, 191, 255, 255]   # (x/2+0.5).clamp(0,1)*255, round half to even

@pytest.mark.parametrize("dev", _devices())
def test_engine_matches_reference_and_batches(dev):
  """Images equal the plain unbatched loop's, for requests that share steps at different step indices and sizes."""
  p = TinyFlow(dev)
  eng = ImageEngine(p, max_batch=4)
  reqs = [ImageParams(prompt=f"prompt {i}", seed=10 + i, width=64, height=64, steps=4) for i in range(3)]
  late = [ImageParams(prompt="late", seed=99, width=64, height=64, steps=3), ImageParams(prompt="wide", seed=5, width=64, height=32, steps=2)]
  rids = [eng.add(r) for r in reqs]
  got: dict[int, list] = {}
  eng.step(); eng.step()                         # the first three are at step 2 of 4 ...
  rids += [eng.add(r) for r in late]             # ... when two more arrive
  batches = []
  while eng.has_work():
    for res in eng.step(): got.setdefault(res.rid, []).append(res)
    batches.append(eng.last_step["batch"])
  assert max(batches) == 4, batches              # "late" joined the running 64x64 bucket mid-trajectory
  for rid, req in zip(rids, reqs + late):
    (res,) = got[rid]
    assert res.image.shape == (req.height, req.width, 3) and res.image.dtype == np.uint8 and res.seed == req.seed
    ref = reference_image(p, req)
    assert np.abs(ref.astype(int) - res.image.astype(int)).max() <= (0 if dev == "CPU" else 1)
  assert eng.stats.images == 5 and eng.stats.requests == 5 and not eng.has_work()

@pytest.mark.parametrize("dev", _devices())
def test_one_capture_per_bucket_timestep_is_a_tensor(dev):
  """One step jit per (batch, size): every step index and every request replays it (the timestep is an input)."""
  p = TinyFlow(dev)
  eng = ImageEngine(p, max_batch=2, batch_sizes=(2,))
  first = eng.generate([ImageParams(prompt="a", seed=1, steps=4), ImageParams(prompt="b", seed=2, steps=4)])
  assert list(eng._step_jits) == [(2, 64, 64)] and list(eng._decode_jits) == [(64, 64)]
  jit = eng._step_jits[(2, 64, 64)]
  eng.generate([ImageParams(prompt="c", seed=3, steps=6)])
  assert list(eng._step_jits) == [(2, 64, 64)] and eng._step_jits[(2, 64, 64)] is jit
  assert jit.captured is not None   # captured once (on its 2nd call), replayed for the other 8 steps
  # with one padded batch size, a request's image cannot depend on its batch companions
  again = eng.generate([ImageParams(prompt="a", seed=1, steps=4)])[0][0]
  assert np.array_equal(again.image, first[0][0].image)

def test_fifo_per_bucket_and_n_seeds():
  p = TinyFlow("CPU")
  eng = ImageEngine(p, max_batch=2)
  a = eng.add(ImageParams(prompt="a", seed=7, steps=2), n=3)              # 3 images: seeds 7, 8, 9
  b = eng.add(ImageParams(prompt="b", seed=1, steps=2, width=32, height=32))
  eng.step()
  assert [len(v) for v in eng.active.values()] == [2, 1]                  # 64x64 full at 2; the 32x32 bucket runs alone
  assert [r.seed for r in eng.waiting] == [9]
  done: list = []
  while eng.has_work(): done += eng.step()
  assert [(r.rid, r.seed) for r in done if r.rid == a] == [(a, 7), (a, 8), (a, 9)]
  imgs = {r.seed: r.image for r in done if r.rid == a}
  assert np.array_equal(imgs[8], reference_image(p, ImageParams(prompt="a", seed=8, steps=2)))
  assert not np.array_equal(imgs[7], imgs[8])

def test_bad_requests_and_abort():
  eng = ImageEngine(TinyFlow("CPU"), max_batch=2, max_waiting=3)
  with pytest.raises(ValueError, match="not supported"): eng.add(ImageParams(prompt="x", width=48, height=48))
  with pytest.raises(ValueError): eng.add(ImageParams(prompt=None))
  with pytest.raises(ValueError): eng.add(ImageParams(prompt="x", steps=0))
  rid = eng.add(ImageParams(prompt="x", seed=0), n=3)
  with pytest.raises(OverflowError): eng.add(ImageParams(prompt="y"))
  eng.step()
  eng.abort(rid)
  assert not eng.has_work()

def test_prompt_embeds_skip_the_encoder():
  p = TinyFlow("CPU")
  emb = p.encode(["hello"])[0]
  eng = ImageEngine(p)
  (r,), = eng.generate([ImageParams(prompt_embeds=emb, seed=3, steps=2)])
  assert np.array_equal(r.image, reference_image(p, ImageParams(prompt="hello", seed=3, steps=2)))

def test_timesteps_hook():
  """A pipeline's timesteps() (the t its model takes per step) is what velocity sees; dt still comes from the sigmas."""
  class Shifted(TinyFlow):
    def timesteps(self, steps, w, h): return (self.sigmas(steps, w, h)[:-1] * 0.5).astype(np.float32)
  p, q = Shifted("CPU"), TinyFlow("CPU")
  req = ImageParams(prompt="t", seed=2, steps=3)
  (r,), = ImageEngine(p).generate([req])
  assert np.array_equal(r.image, reference_image(p, req)) and not np.array_equal(r.image, reference_image(q, req))

def test_monolithic_pipeline_fallback():
  p = TinyFlowMonolithic("CPU")
  eng = ImageEngine(p, max_batch=4)
  assert eng.mono and eng.max_batch == 1
  out = eng.generate([ImageParams(prompt="a", seed=1, steps=2), ImageParams(prompt="b", seed=2, steps=2)])
  assert p.calls == 2
  assert np.array_equal(out[1][0].image, reference_image(p.inner, ImageParams(prompt="b", seed=2, steps=2)))

# ---------------------------------------------------------------- HTTP + openai client

@pytest.fixture(scope="module")
def srv():
  import uvicorn
  from fusezap.serve.image_api import create_image_app
  dev = _devices()[-1]
  pipe = TinyFlow(dev)
  eng = ImageEngine(pipe, max_batch=4)
  eng.warmup()
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  server = uvicorn.Server(uvicorn.Config(create_image_app(eng, "tiny-flow"), host="127.0.0.1", port=port, log_level="warning"))
  th = threading.Thread(target=server.run, daemon=True); th.start()
  t0 = time.time()
  while not server.started:
    assert time.time() - t0 < 60 and th.is_alive(), "server did not start"
    time.sleep(0.05)
  yield {"url": f"http://127.0.0.1:{port}", "pipe": pipe, "eng": eng}
  server.should_exit = True; th.join(timeout=30)

def test_openai_client_images_generate(srv):
  from openai import OpenAI
  c = OpenAI(base_url=srv["url"] + "/v1", api_key="x", timeout=120)
  r = c.images.generate(model="tiny-flow", prompt="a red fox", n=2, size="64x64", response_format="b64_json",
                        extra_body={"seed": 42, "num_inference_steps": 3})
  assert len(r.data) == 2 and r.created > 0 and r.output_format == "png" and r.size == "64x64"
  for k, d in enumerate(r.data):
    img = _png(d.b64_json)
    ref = reference_image(srv["pipe"], ImageParams(prompt="a red fox", seed=42 + k, width=64, height=64, steps=3))
    assert img.shape == (64, 64, 3) and np.abs(img.astype(int) - ref.astype(int)).max() <= 1
  assert not np.array_equal(_png(r.data[0].b64_json), _png(r.data[1].b64_json))
  # defaults: size auto (64x64), the model's steps; a random seed is echoed back and reproduces the image
  r = c.images.generate(prompt="a red fox")
  seed = r.model_extra.get("data", [{}])[0].get("seed") if r.model_extra else None
  seed = seed if seed is not None else r.data[0].model_extra["seed"]
  ref = reference_image(srv["pipe"], ImageParams(prompt="a red fox", seed=seed))
  assert np.abs(_png(r.data[0].b64_json).astype(int) - ref.astype(int)).max() <= 1
  # jpeg / webp
  for fmt, magic in (("jpeg", b"\xff\xd8"), ("webp", b"RIFF")):
    r = c.images.generate(model="tiny-flow", prompt="x", size="64x32", output_format=fmt, output_compression=90, extra_body={"seed": 1})
    raw = base64.b64decode(r.data[0].b64_json)
    assert raw[:len(magic)] == magic and r.output_format == fmt and _png(r.data[0].b64_json).shape == (32, 64, 3)

def test_openai_client_errors(srv):
  import openai
  c = openai.OpenAI(base_url=srv["url"] + "/v1", api_key="x", timeout=60)
  with pytest.raises(openai.BadRequestError, match="not supported"): c.images.generate(prompt="x", size="1024x1024")
  with pytest.raises(openai.BadRequestError, match="url"): c.images.generate(prompt="x", response_format="url")
  with pytest.raises(openai.BadRequestError): c.images.generate(prompt="x", n=11)
  with pytest.raises(openai.BadRequestError): c.images.generate(prompt="x", output_format="gif")  # type: ignore[arg-type]
  with pytest.raises(openai.BadRequestError): c.images.generate(prompt="x", extra_body={"num_inference_steps": 0})
  with pytest.raises(openai.NotFoundError): c.images.generate(prompt="x", model="dall-e-3")
  with pytest.raises(openai.BadRequestError): c.images.generate(prompt="")

def test_concurrent_requests_share_steps_and_metrics(srv):
  """Concurrent clients' images are batched into shared steps (row-steps > steps) and equal the reference."""
  import httpx
  from openai import AsyncOpenAI
  eng = srv["eng"]
  s0, rs0 = eng.stats.steps, eng.stats.row_steps
  async def go():
    async with AsyncOpenAI(base_url=srv["url"] + "/v1", api_key="x", timeout=120) as c:
      return await asyncio.gather(*[c.images.generate(prompt=f"p{i}", size="64x64", extra_body={"seed": i, "num_inference_steps": 4}) for i in range(6)])
  rs = asyncio.run(go())
  for i, r in enumerate(rs):
    ref = reference_image(srv["pipe"], ImageParams(prompt=f"p{i}", seed=i, steps=4))
    assert np.abs(_png(r.data[0].b64_json).astype(int) - ref.astype(int)).max() <= 1
  steps, rows = eng.stats.steps - s0, eng.stats.row_steps - rs0
  assert rows == 24 and steps < rows, (steps, rows)
  m = httpx.get(srv["url"] + "/metrics").text
  for name in ("fusezap:images_generated_total", "fusezap:image_requests_waiting", "fusezap:image_requests_running",
               "fusezap:image_step_seconds_sum", "fusezap:image_step_seconds_count", "fusezap:images_per_second", "fusezap:image_e2e_seconds_sum"):
    assert f'{name}{{model_name="tiny-flow"}}' in m, name
  got = {l.split("{")[0]: float(l.split()[-1]) for l in m.splitlines() if l.startswith("fusezap:")}
  assert got["fusezap:images_generated_total"] == eng.stats.images and got["fusezap:image_requests_waiting"] == 0
  assert httpx.get(srv["url"] + "/health").json() == {"status": "ok"}
  ms = httpx.get(srv["url"] + "/v1/models").json()
  assert ms["data"][0]["id"] == "tiny-flow" and ms["data"][0]["default_size"] == "64x64"

def test_api_key_and_cli_serve_image(tmp_path):
  """`fusezap serve-image tiny_diffusion:TinyFlow --api-key k` as a subprocess: 401 without the key, an image with it."""
  import httpx
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]
  env = {**os.environ, "PYTHONPATH": os.path.dirname(__file__) + os.pathsep + os.environ.get("PYTHONPATH", "")}
  cmd = [sys.executable, "-m", "fusezap", "serve-image", "tiny_diffusion:TinyFlow", "--device", "CPU", "--port", str(port),
         "--api-key", "k", "--max-batch", "2", "--log-level", "warning"]
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
  url = f"http://127.0.0.1:{port}"
  try:
    t0 = time.time()
    while True:
      assert proc.poll() is None, proc.stdout.read()
      assert time.time() - t0 < 300, "server did not start"
      try:
        if httpx.get(url + "/health", timeout=1).status_code == 200: break
      except httpx.HTTPError: pass
      time.sleep(0.3)
    assert httpx.post(url + "/v1/images/generations", json={"prompt": "x"}).status_code == 401
    r = httpx.post(url + "/v1/images/generations", json={"prompt": "x", "seed": 4, "num_inference_steps": 2},
                   headers={"authorization": "Bearer k"}, timeout=120)
    assert r.status_code == 200, r.text
    ref = reference_image(TinyFlow("CPU"), ImageParams(prompt="x", seed=4, steps=2))
    assert np.array_equal(_png(r.json()["data"][0]["b64_json"]), ref)
  finally:
    proc.terminate(); proc.wait(timeout=30)
