"""I81 D3: Qwen3 in fusezap/serve/models/llama.py (per-head q/k RMSNorm) and Llama.encode (HF hidden_states without
lm_head, padded batches), vs HF transformers float32 on torch CPU.

- tiny random Qwen3 (the I82 §1 config): greedy tokens through generate() and LLMEngine (forward_tokens), and
  encode()'s hidden states 1/2/3 with a right-padded mask, on CPU and METAL
- Qwen/Qwen3-0.6B (skips unless in the HF cache): greedy 64 tokens token-exact in float32
The klein Qwen3-4B encoder's G2 gate is scripts/diff_text_encoder_gate.py (16 GB in f32, run by hand).
"""
from __future__ import annotations
import os
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve.sampling import Sampler

TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, vocab_size=256, rms_norm_eps=1e-6, rope_theta=1e6, tie_word_embeddings=True, max_position_embeddings=512,
            initializer_range=0.3, attention_bias=False, bos_token_id=1, eos_token_id=2)

def _devices() -> list[str]:
  from fusezap.device import Device
  if os.environ.get("FZ_DEVICE"): return [os.environ["FZ_DEVICE"]]
  return ["CPU"] + (["METAL"] if Device.available("METAL") else [])

@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  torch.manual_seed(0)
  hfm = tr.Qwen3ForCausalLM(tr.Qwen3Config(**TINY)).eval()
  with torch.no_grad():   # non-trivial norm weights, so a missing or misplaced q/k norm shows
    for n, p in hfm.named_parameters():
      if n.endswith("norm.weight"): p.copy_(1 + 0.5 * torch.randn_like(p))
  d = tmp_path_factory.mktemp("qwen3tiny")
  hfm.save_pretrained(d, safe_serialization=True)
  return hfm, str(d)

def _hf_greedy(hfm, ids: list[int], n: int) -> list[int]:
  import torch
  out, past, inp = [], None, torch.tensor([ids])
  with torch.no_grad():
    for _ in range(n):
      o = hfm(inp, past_key_values=past, use_cache=True)
      past = o.past_key_values
      t = int(o.logits[0, -1].argmax())
      out.append(t)
      inp = torch.tensor([[t]])
  return out

def _rel(a: np.ndarray, b: np.ndarray) -> float: return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))

def test_qwen3_config():
  from fusezap.serve.models.llama import LlamaConfig
  c = LlamaConfig.from_hf({**TINY, "model_type": "qwen3", "hidden_act": "silu"})
  assert c.qk_norm and c.model_type == "qwen3" and c.head_dim == 8 and c.tie_embeddings
  assert not LlamaConfig.from_hf({**TINY, "model_type": "qwen2", "hidden_act": "silu"}).qk_norm

@pytest.mark.parametrize("dev", _devices())
def test_tiny_qwen3_greedy_matches_hf(tiny, dev):
  from fusezap.serve.models import load_model
  hfm, path = tiny
  m = load_model(path, dtype=dtypes.float32, device=dev, max_len=128, prefill_chunk=64)
  assert m.cfg.qk_norm and all(l.qn is not None and l.kn is not None for l in m.layers)
  ids = [int(i) for i in np.random.default_rng(0).integers(3, TINY["vocab_size"], 20)]
  ref = _hf_greedy(hfm, ids, 32)
  assert len(set(ref)) > 3, "degenerate reference"
  mine = list(m.generate(ids, 32, Sampler(), stop=[]))
  assert mine == ref, f"first mismatch at {next((i for i, (a, b) in enumerate(zip(mine, ref)) if a != b), len(mine))}"

@pytest.mark.parametrize("dev", _devices())
def test_tiny_qwen3_engine_matches_hf(tiny, dev):
  """LLMEngine's ragged forward_tokens path (paged cache) with the q/k norms, two prompts batched."""
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  hfm, path = tiny
  m = load_model(path, dtype=dtypes.float32, device=dev, max_len=128, prefill_chunk=64)
  eng = LLMEngine(m, num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
  prompts = [[int(i) for i in np.random.default_rng(s).integers(3, TINY["vocab_size"], n)] for s, n in ((1, 9), (2, 17))]
  got = eng.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True))
  assert got == [_hf_greedy(hfm, p, 16) for p in prompts]

@pytest.mark.parametrize("dev", _devices())
def test_tiny_qwen3_encode_hidden_states(tiny, dev):
  """encode() == HF hidden_states[k] (k = 1, 2, 3 and 0) for a right-padded batch with a 0/1 attention mask, every row
  (pad rows included: the klein pipeline keeps them), max rel err <= 1e-4 (G1)."""
  import torch
  from fusezap.serve.models.llama import Llama, LlamaConfig
  from fusezap.serve.weights import load_safetensors, load_config
  hfm, path = tiny
  rng = np.random.default_rng(3)
  T, lens = 24, (24, 11, 5)
  ids = rng.integers(3, TINY["vocab_size"], (len(lens), T))
  mask = np.zeros_like(ids)
  for b, n in enumerate(lens): mask[b, :n] = 1; ids[b, n:] = 0
  with torch.no_grad(): ref = hfm(torch.tensor(ids), attention_mask=torch.tensor(mask), output_hidden_states=True).hidden_states
  m = Llama(LlamaConfig.from_hf(load_config(path)), load_safetensors(path), dtypes.float32, dev, max_len=64, lm_head=False)
  assert m.lm_head is None
  for layers in ((1, 2, 3), (0, 2)):
    hs = m.encode(ids, mask, layers)
    assert hs.shape == (len(lens), len(layers), T, TINY["hidden_size"]) and hs.dtype == np.float32
    for j, k in enumerate(layers):
      assert _rel(hs[:, j], ref[k].numpy()) <= 1e-4, (layers, k, _rel(hs[:, j], ref[k].numpy()))
  # load_layers: only the layers the encoder reads are loaded; the result is the same
  m2 = Llama(LlamaConfig.from_hf(load_config(path)), load_safetensors(path), dtypes.float32, dev, max_len=64, lm_head=False, load_layers=2)
  assert len(m2.layers) == 2 and m2.cfg.n_layers == 2
  np.testing.assert_array_equal(m2.encode(ids, mask, (1, 2)), m.encode(ids, mask, (1, 2)))

def test_tiny_qwen3_encode_bf16_metal(tiny):
  """bf16 weights and activations on METAL: close to HF float32 within the bf16 tolerance (rtol/atol 1e-2 scaled)."""
  from fusezap.device import Device
  if not Device.available("METAL"): pytest.skip("no METAL")
  import torch
  from fusezap.serve.models.llama import Llama, LlamaConfig
  from fusezap.serve.weights import load_safetensors, load_config
  hfm, path = tiny
  ids = np.random.default_rng(4).integers(3, TINY["vocab_size"], (1, 16))
  with torch.no_grad(): ref = hfm(torch.tensor(ids), output_hidden_states=True).hidden_states
  m = Llama(LlamaConfig.from_hf(load_config(path)), load_safetensors(path), dtypes.bfloat16, "METAL", max_len=64, lm_head=False)
  hs = m.encode(ids, None, (1, 3))
  for j, k in enumerate((1, 3)):
    r = ref[k].numpy()
    assert np.linalg.norm(hs[:, j] - r) / np.linalg.norm(r) < 2e-2

# ---------------------------------------------------------------- Qwen/Qwen3-0.6B

Q06 = "Qwen/Qwen3-0.6B"

def _q06() -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(Q06, allow_download=False)
  except Exception: return None

@pytest.mark.skipif(_q06() is None, reason=f"{Q06} not in the HF cache")
def test_qwen3_06b_greedy_64_tokens_match_hf():
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  from fusezap.serve.models import load_model
  path = _q06()
  ids = tr.AutoTokenizer.from_pretrained(path).apply_chat_template([{"role": "user", "content": "Write a short poem about the sea."}],
                                                                    add_generation_prompt=True, enable_thinking=False, tokenize=True)
  ids = [int(i) for i in (ids["input_ids"] if hasattr(ids, "keys") else ids)]
  hfm = tr.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
  ref = _hf_greedy(hfm, ids, 64)
  del hfm
  dev = os.environ.get("FZ_DEVICE") or _devices()[-1]
  m = load_model(Q06, dtype=dtypes.float32, device=dev, max_len=256, prefill_chunk=128)
  mine = list(m.generate(ids, 64, Sampler(), stop=[]))
  assert mine == ref, f"first mismatch at {next((i for i, (a, b) in enumerate(zip(mine, ref)) if a != b), len(mine))}"

# ---------------------------------------------------------------- the klein encoder's prompt formatting

def _klein_tok() -> str | None:
  try:
    from fusezap.serve.models.text_encoder import resolve_subfolders
    return os.path.join(resolve_subfolders("black-forest-labs/FLUX.2-klein-4B", ("tokenizer",), allow_download=False), "tokenizer")
  except Exception: return None

@pytest.mark.skipif(_klein_tok() is None, reason="FLUX.2-klein-4B tokenizer not in the HF cache")
def test_klein_tokenize_matches_diffusers_formatting():
  """Qwen3TextEncoder.tokenize == the diffusers pipeline's: chat template (enable_thinking=False), right padding with
  the pad token to 512, truncation (a prompt over 512 tokens), 0/1 mask; the encoder object is the generate seam."""
  tr = pytest.importorskip("transformers")
  from fusezap.serve.models.text_encoder import Qwen3TextEncoder
  from fusezap.serve.tokenizer import Tokenizer
  path = _klein_tok()
  enc = Qwen3TextEncoder(None, Tokenizer(path))
  hf = tr.AutoTokenizer.from_pretrained(path)
  prompts = ["a red fox", "日本の古い寺院、桜", "word " * 700, ""]
  ids, mask = enc.tokenize(prompts)
  for b, p in enumerate(prompts):
    text = hf.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert enc.prompt_text(p) == text
    r = hf(text, padding="max_length", truncation=True, max_length=512)
    assert ids[b].tolist() == r["input_ids"] and mask[b].tolist() == r["attention_mask"], b
  assert mask[2].all() and callable(enc) and enc.__call__ == enc.encode
