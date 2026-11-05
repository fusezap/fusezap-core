"""I81 D6 x D7 integration: FLUX.2 [klein] through FuseZap end to end on a tiny diffusers-layout model directory (tiny
random Qwen3 text encoder with a byte-level tokenizer and a Qwen3-style chat template, the I82 §1 tiny transformer and
VAE), against the stock diffusers Flux2KleinPipeline run from the prompt with the same numpy noise.

- KleinImagePipeline (fusezap/serve/models/klein_image.py) served by ImageEngine: the real Flux2Transformer /
  Flux2VAEDecoder / Qwen3TextEncoder modules, on CPU and METAL; the engine's image == the plain loop's and ~= diffusers'
- `fusezap generate --prompt` with the fusezap encoder (the "auto" choice) and with the transformers fallback
"""
from __future__ import annotations
import json, os
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
diffusers = pytest.importorskip("diffusers")
pytest.importorskip("tokenizers")
from fusezap.device import Device
from test_flux2 import tiny_transformer
from test_flux2_vae import tiny_vae
from test_flux2_pipeline import KLEIN_SCHED

H = W = 32   # tiny VAE scale 2 -> an 8x8 token grid
PROMPT = "a red fox in fresh snow"
TEMPLATE = ("{%- for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{%- endfor %}"
            "{%- if add_generation_prompt %}<|im_start|>assistant\n{%- if enable_thinking is defined and enable_thinking is false %}"
            "<think>\n\n</think>\n\n{%- endif %}{%- endif %}")
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]

def _devices() -> list[str]:
  if os.environ.get("FZ_DEVICE"): return [os.environ["FZ_DEVICE"]]
  return ["CPU"] + (["METAL"] if Device.available("METAL") else [])

def _tokenizer(d: str) -> None:
  """A byte-level BPE with no merges (256 byte tokens) + Qwen's special tokens, the pad token <|endoftext|>."""
  from tokenizers import Tokenizer, models, pre_tokenizers, decoders
  alphabet = pre_tokenizers.ByteLevel.alphabet()
  vocab = {c: i for i, c in enumerate(sorted(alphabet))}
  tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
  tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
  tok.decoder = decoders.ByteLevel()
  tok.add_special_tokens(SPECIAL)
  os.makedirs(d, exist_ok=True)
  tok.save(os.path.join(d, "tokenizer.json"))
  json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "chat_template": TEMPLATE, "pad_token": "<|endoftext|>",
             "eos_token": "<|im_end|>", "model_max_length": 131072}, open(os.path.join(d, "tokenizer_config.json"), "w"))

@pytest.fixture(scope="module")
def klein_dir(tmp_path_factory):
  """A diffusers FLUX.2 [klein] layout: text_encoder/ (Qwen3, 28 layers so hidden_states 9/18/27 exist; 3 x 32 = the
  transformer's joint_attention_dim 96), tokenizer/, transformer/, vae/, scheduler/, model_index.json."""
  d = str(tmp_path_factory.mktemp("klein"))
  _tokenizer(os.path.join(d, "tokenizer"))
  torch.manual_seed(1)
  te = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
    hidden_size=32, intermediate_size=64, num_hidden_layers=28, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
    vocab_size=260, rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=True, max_position_embeddings=1024,
    initializer_range=0.3, attention_bias=False, pad_token_id=256, eos_token_id=258)).eval()
  te.save_pretrained(os.path.join(d, "text_encoder"), safe_serialization=True)
  tiny_transformer().save_pretrained(os.path.join(d, "transformer"))
  tiny_vae(2).save_pretrained(os.path.join(d, "vae"))
  diffusers.FlowMatchEulerDiscreteScheduler(**KLEIN_SCHED).save_pretrained(os.path.join(d, "scheduler"))
  json.dump({"_class_name": "Flux2KleinPipeline", "is_distilled": True}, open(os.path.join(d, "model_index.json"), "w"))
  return d

def _diffusers_image(d: str, prompt: str, seed: int, steps: int) -> np.ndarray:
  """The stock pipeline from the prompt (its own HF tokenizer + Qwen3 encoder), float32 on the CPU, numpy noise."""
  from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel, AutoencoderKLFlux2, FlowMatchEulerDiscreteScheduler
  from fusezap.serve.models.flux2_klein import noise
  pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(d, "scheduler")),
                            vae=AutoencoderKLFlux2.from_pretrained(os.path.join(d, "vae")).eval(),
                            text_encoder=transformers.Qwen3ForCausalLM.from_pretrained(os.path.join(d, "text_encoder")).eval(),
                            tokenizer=transformers.AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer")),
                            transformer=Flux2Transformer2DModel.from_pretrained(os.path.join(d, "transformer")).eval(), is_distilled=True)
  with torch.no_grad():
    img = pipe(prompt=prompt, latents=torch.tensor(noise(seed, 16, H // 4, W // 4)), height=H, width=W,
               num_inference_steps=steps, output_type="np").images[0]
  return np.round(img * 255).astype(np.uint8)

def _within1(a: np.ndarray, b: np.ndarray) -> float: return float((np.abs(a.astype(int) - b.astype(int)) <= 1).mean())

@pytest.fixture(scope="module")
def ref_image(klein_dir):
  img = _diffusers_image(klein_dir, PROMPT, 5, 4)
  assert img.std() > 10 and len(np.unique(img)) > 100, "degenerate reference image"
  return img

@pytest.mark.parametrize("dev", _devices())
def test_klein_image_engine_serves_flux2_modules(klein_dir, ref_image, dev):
  """ImageEngine over KleinImagePipeline (float32): the adapter wires the real FLUX.2 modules; the served image equals
  the engine-free loop (reference_image) and matches diffusers from the prompt (G6 tiny: >= 99.9% within +-1)."""
  from fusezap.serve.models.klein_image import KleinImagePipeline
  from fusezap.serve.models.flux2 import Flux2Transformer
  from fusezap.serve.models.flux2_vae import Flux2VAEDecoder
  from fusezap.serve.models.text_encoder import Qwen3TextEncoder
  from fusezap.serve.image_engine import ImageEngine, ImageParams, reference_image
  pipe = KleinImagePipeline(klein_dir, device=dev, dtype="float32")
  assert isinstance(pipe.tr, Flux2Transformer) and isinstance(pipe.vae, Flux2VAEDecoder) and isinstance(pipe.encoder, Qwen3TextEncoder)
  assert pipe.device == dev and pipe.pipe.vae_scale == 2
  emb = pipe.encode([PROMPT])
  assert emb.shape == (1, 512, 96) and emb.dtype == np.float32
  with pytest.raises(ValueError): pipe.check_size(30, 32)
  eng = ImageEngine(pipe, max_batch=2, batch_sizes=(1, 2))
  req = ImageParams(prompt=PROMPT, seed=5, width=W, height=H, steps=4)
  (r,), = eng.generate([req])
  assert r.image.shape == (H, W, 3) and r.image.dtype == np.uint8 and r.seed == 5
  assert np.abs(r.image.astype(int) - reference_image(pipe, req).astype(int)).max() <= (0 if dev == "CPU" else 1)
  assert _within1(r.image, ref_image) >= 0.999, np.abs(r.image.astype(int) - ref_image.astype(int)).max()
  # a batch of two (another prompt and size-compatible seed) and the prompt_embeds path give the same first image
  (a,), (b,) = eng.generate([ImageParams(prompt_embeds=emb[0], seed=5, width=W, height=H, steps=4),
                             ImageParams(prompt="a bowl of ramen", seed=6, width=W, height=H, steps=4)])
  assert np.abs(a.image.astype(int) - r.image.astype(int)).max() <= (0 if dev == "CPU" else 1)
  assert not np.array_equal(a.image, b.image)

@pytest.mark.parametrize("dev", _devices())
def test_generate_cli_fusezap_encoder(klein_dir, ref_image, dev, tmp_path, capsys):
  """`fusezap generate --prompt` on the tiny model: "auto" picks the fusezap Qwen3 encoder on --device, and the PNG
  matches diffusers from the same prompt; --encoder hf (transformers, the fallback) gives the same image."""
  from PIL import Image
  from fusezap.__main__ import main
  from fusezap.serve.generate import pick_encoder, ENCODERS
  assert pick_encoder("auto") == "fusezap" and {"fusezap", "hf"} <= set(ENCODERS)
  out = str(tmp_path / "fz.png")
  assert main(["generate", "--model", klein_dir, "--prompt", PROMPT, "--seed", "5", "--size", str(W), "--steps", "4",
               "--device", dev, "--out", out]) == 0
  assert "(fusezap encoder)" in capsys.readouterr().out
  png = np.asarray(Image.open(out).convert("RGB"))
  assert png.shape == (H, W, 3)
  assert _within1(png, ref_image) >= 0.999, np.abs(png.astype(int) - ref_image.astype(int)).max()
  if dev != "CPU": return
  out2 = str(tmp_path / "hf.png")
  assert main(["generate", "--model", klein_dir, "--prompt", PROMPT, "--seed", "5", "--size", str(W), "--steps", "4",
               "--device", dev, "--encoder", "hf", "--encoder-inproc", "--out", out2]) == 0
  assert "(hf encoder)" in capsys.readouterr().out
  assert _within1(np.asarray(Image.open(out2).convert("RGB")), png) >= 0.999

def test_generate_auto_falls_back_to_transformers(klein_dir, tmp_path, monkeypatch, capsys):
  """auto: when the fusezap encoder cannot load (here: an ImportError), generate falls back to transformers."""
  from fusezap.__main__ import main
  from fusezap.serve import generate
  def broken(*a, **k): raise ImportError("no fusezap encoder")
  monkeypatch.setitem(generate.ENCODERS, "fusezap", broken)
  monkeypatch.setattr(Device, "DEFAULT", "CPU" if Device.available("CPU") else "NPY", raising=False)
  assert main(["generate", "--model", klein_dir, "--prompt", PROMPT, "--seed", "5", "--size", str(W), "--steps", "1",
               "--encoder-inproc", "--out", str(tmp_path / "x.png")]) == 0
  o = capsys.readouterr().out
  assert "falling back to hf" in o and "(hf encoder)" in o
  with pytest.raises(ImportError):   # an explicit --encoder does not fall back
    main(["generate", "--model", klein_dir, "--prompt", PROMPT, "--encoder", "fusezap", "--size", str(W), "--out", str(tmp_path / "y.png")])

def test_default_vae_dtype(monkeypatch):
  """I95: bf16 VAE on a CUDA GPU under 40 GB (the RTX 4090 ran out of memory with the f32 one), f32 elsewhere; the
  argument and FZ_KLEIN_VAE_DTYPE override."""
  from fusezap.serve.models import klein_image as K
  monkeypatch.delenv("FZ_KLEIN_VAE_DTYPE", raising=False)
  sizes = {"CUDA": 23.6, "CUDA:1": 79.1}
  monkeypatch.setattr(K, "device_total_gb", lambda d: sizes.get(d))
  assert K.default_vae_dtype("CUDA")[0] == "bfloat16" and K.default_vae_dtype("CUDA:1")[0] == "float32"
  assert K.default_vae_dtype("METAL")[0] == "float32" and K.default_vae_dtype("CPU")[0] == "float32"
  assert K.default_vae_dtype("CUDA", "float32") == ("float32", "requested")
  monkeypatch.setenv("FZ_KLEIN_VAE_DTYPE", "float32")
  assert K.default_vae_dtype("CUDA") == ("float32", "FZ_KLEIN_VAE_DTYPE")
  monkeypatch.undo()
  assert K.device_total_gb("METAL") is None
