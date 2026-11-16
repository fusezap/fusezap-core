"""Sliding-window attention and its layer variants (tracker I105 part B; fusezap/serve/models/swa.py), on the Mac:

  * config parsing: Gemma 3 text in the published (transformers 4.x) form (sliding_window_pattern 6, rope_local_base_freq,
    query_pre_attn_scalar) and transformers 5's rope_parameters per layer type; OLMo 3 (3 sliding : 1 full, YaRN on the
    full layers); Mistral's all-layer window;
  * YaRN tables == HF's Olmo3RotaryEmbedding (full layers, attention factor included); the sliding layers' plain tables;
  * random-weight Gemma3ForCausalLM (5:1 local / global, window 8, sandwich norms, (1 + w) norms, per-head q/k norm,
    sqrt(D) embedding scale, GELU, tied head, local theta 1e4) and Olmo3ForCausalLM (3:1, window 8, post-norms, q/k norm
    over all heads, YaRN) and MistralForCausalLM (every layer, window 8) saved by HF transformers: HF's float32 greedy tokens
    through load_model on CPU and METAL over 52 positions (the window is passed); without the window the logits move far;
  * LLMEngine (paged cache, windowed paged attention) == Llama.generate's tokens on CPU and METAL; on NKISHIM (the keyrange
    attention with the windowed layers' own lo per row) float32 tokens == CPU's, bf16 near them.
"""
from __future__ import annotations
import os
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve.sampling import Sampler

GEMMA3_1B_PUBLISHED = {   # google/gemma-3-1b-it config.json keys as published (gated; the test only parses them)
  "architectures": ["Gemma3ForCausalLM"], "attention_bias": False, "attn_logit_softcapping": None, "bos_token_id": 2, "eos_token_id": [1, 106],
  "final_logit_softcapping": None, "head_dim": 256, "hidden_activation": "gelu_pytorch_tanh", "hidden_size": 1152, "intermediate_size": 6912,
  "max_position_embeddings": 32768, "model_type": "gemma3_text", "num_attention_heads": 4, "num_hidden_layers": 26, "num_key_value_heads": 1,
  "query_pre_attn_scalar": 256, "rms_norm_eps": 1e-06, "rope_local_base_freq": 10000, "rope_scaling": None, "rope_theta": 1000000,
  "sliding_window": 512, "sliding_window_pattern": 6, "vocab_size": 262144}
OLMO3_7B = {"architectures": ["Olmo3ForCausalLM"], "attention_bias": False, "eos_token_id": 100257, "hidden_act": "silu", "hidden_size": 4096,
  "intermediate_size": 11008, "max_position_embeddings": 65536, "model_type": "olmo3", "num_attention_heads": 32, "num_hidden_layers": 32,
  "num_key_value_heads": 32, "rms_norm_eps": 1e-06, "rope_scaling": {"attention_factor": 1.2079441541679836, "beta_fast": 32.0, "beta_slow": 1.0,
  "factor": 8.0, "original_max_position_embeddings": 8192, "rope_type": "yarn"}, "rope_theta": 500000, "sliding_window": 4096,
  "tie_word_embeddings": False, "vocab_size": 100278,
  "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"] * 8}

def test_config_gemma3_olmo3_mistral():
  from fusezap.serve.models.llama import LlamaConfig
  g = LlamaConfig.from_hf(GEMMA3_1B_PUBLISHED)
  assert g.layer_windows == tuple(0 if (i + 1) % 6 == 0 else 512 for i in range(26)) and g.sliding_window is None
  assert (g.rope_theta, g.rope_local_theta, g.rope_scaling, g.attn_scale) == (1e6, 1e4, None, 256 ** -0.5)
  assert g.norm_plus_one and g.post_norm and g.pre_norm and g.embed_scale and g.act == "gelu_pytorch_tanh" and g.tie_embeddings
  assert g.qk_norm and not g.qk_norm_full and (g.n_heads, g.n_kv_heads, g.head_dim) == (4, 1, 256) and not g.plain
  o = LlamaConfig.from_hf(OLMO3_7B)
  assert o.layer_windows == (4096, 4096, 4096, 0) * 8 and o.rope_scaling["rope_type"] == "yarn" and o.rope_local_theta == 5e5
  assert o.post_norm and not o.pre_norm and o.qk_norm_full and not o.norm_plus_one and o.act == "silu"
  m = LlamaConfig.from_hf({**OLMO3_7B, "model_type": "mistral", "layer_types": None, "rope_scaling": None, "num_key_value_heads": 8})
  assert m.layer_windows == (4096,) * 32 and not m.post_norm and m.rope_local_theta is None
  with pytest.raises(NotImplementedError): LlamaConfig.from_hf({**GEMMA3_1B_PUBLISHED, "model_type": "gemma2"})
  with pytest.raises(NotImplementedError): LlamaConfig.from_hf({**GEMMA3_1B_PUBLISHED, "final_logit_softcapping": 30.0})

def test_yarn_tables_match_hf():
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  from transformers.models.olmo3.modeling_olmo3 import Olmo3RotaryEmbedding
  from fusezap.serve.models.llama import LlamaConfig, rope_tables, rope_local_tables
  hc = tr.Olmo3Config(**{k: v for k, v in OLMO3_7B.items() if k != "architectures"})
  c = LlamaConfig.from_hf(OLMO3_7B)
  rot = Olmo3RotaryEmbedding(hc)
  pos = np.array([0, 1, 100, 1023, 4095, 8191, 20000])
  for kind, (cos, sin) in (("full_attention", rope_tables(c, 20001)), ("sliding_attention", rope_local_tables(c, 20001))):
    hcs, hsn = rot(torch.zeros(1, len(pos), 128), torch.tensor(pos)[None], kind)
    atol = (3e-5 + 1.5e-7 * pos)[:, None]
    assert (np.abs(cos[pos] - hcs[0, :, :64].numpy()) <= atol).all() and (np.abs(sin[pos] - hsn[0, :, :64].numpy()) <= atol).all(), kind

def _model_dir(path, kind: str, seed: int = 0) -> str:
  import torch, transformers as tr
  torch.manual_seed(seed)
  if kind == "gemma3":   # 5 local : 1 global; small embeddings (tied, times sqrt(D)) so the layers decide the next token
    m = tr.Gemma3ForCausalLM(tr.Gemma3TextConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=6, num_attention_heads=4,
          num_key_value_heads=1, head_dim=32, vocab_size=512, sliding_window=8, query_pre_attn_scalar=24, initializer_range=0.3,
          max_position_embeddings=256))
  elif kind == "olmo3":
    m = tr.Olmo3ForCausalLM(tr.Olmo3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=4, num_attention_heads=4, vocab_size=512,
          sliding_window=8, initializer_range=0.08, max_position_embeddings=512, eos_token_id=3,
          rope_scaling={"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 64, "attention_factor": 1.2, "beta_fast": 32.0, "beta_slow": 1.0}))
  else:
    m = tr.MistralForCausalLM(tr.MistralConfig(hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
          num_key_value_heads=2, vocab_size=512, sliding_window=8, initializer_range=0.08, max_position_embeddings=256))
  m = m.eval()
  with torch.no_grad():
    for n, p in m.named_parameters():
      if "norm" in n: p.add_(torch.randn_like(p) * 0.2)
      if "embed" in n and kind == "gemma3": p.mul_(0.1)
  m.save_pretrained(path, safe_serialization=True)
  return str(path)

def _hf_greedy(hfm, ids: list[int], n: int) -> list[int]:
  import torch
  out, past, inp = [], None, torch.tensor([ids])
  with torch.no_grad():
    for _ in range(n):
      o = hfm(inp, past_key_values=past, use_cache=True)
      past, t = o.past_key_values, int(o.logits[0, -1].argmax())
      out.append(t)
      inp = torch.tensor([[t]])
  return out

IDS = [int(i) for i in np.random.default_rng(0).integers(3, 512, 20)]

@pytest.mark.parametrize("dev", ["CPU", "METAL"])
@pytest.mark.parametrize("kind", ["gemma3", "olmo3", "mistral"])
def test_tiny_swa_greedy_matches_hf(tmp_path, kind, dev):
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  from fusezap.device import Device
  if dev == "METAL" and not Device.available("METAL"): pytest.skip("no METAL")
  from fusezap.serve.models import load_model
  p = _model_dir(tmp_path / kind, kind, seed=1)
  hfm = tr.AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32).eval()
  m = load_model(p, dtype=dtypes.float32, device=dev, max_len=128, prefill_chunk=64)
  assert m.cfg.layer_windows and 8 in m.cfg.layer_windows
  ref = _hf_greedy(hfm, IDS, 32)
  assert len(set(ref)) > 3, "degenerate reference"
  assert list(m.generate(IDS, 32, Sampler(), stop=[])) == ref
  with torch.no_grad(): want = hfm(torch.tensor([IDS + ref])).logits[0, -1].numpy()
  m.reset()
  got = m.prefill([IDS + ref])[0]
  np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
  if dev == "CPU":   # the window matters at 52 positions: the same weights with full attention everywhere move the logits far
    import dataclasses
    from fusezap.serve.models.llama import Llama
    from fusezap.serve.weights import load_safetensors
    m2 = Llama(dataclasses.replace(m.cfg, layer_windows=tuple(0 for _ in m.cfg.layer_windows)), load_safetensors(p), dtypes.float32, "CPU",
               max_len=128, prefill_chunk=64)
    assert np.abs(m2.prefill([IDS + ref])[0] - want).max() > 100 * np.abs(got - want).max() + 1e-3

PROMPTS = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7], [5] * 20, list(range(30, 60))]

def _engine_tokens(path: str, device: str, dtype=dtypes.float32) -> dict:
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  m = load_model(path, dtype=dtype, device=device, max_len=128, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
  return dict(enumerate(eng.generate(PROMPTS, SamplingParams(max_tokens=12, ignore_eos=True))))

def _generate_tokens(path: str, device: str) -> dict:
  from fusezap.serve.models import load_model
  m = load_model(path, dtype=dtypes.float32, device=device, max_len=128, prefill_chunk=64)
  return {i: list(m.generate(p, 12, Sampler(), stop=[])) for i, p in enumerate(PROMPTS)}

@pytest.mark.parametrize("kind", ["gemma3", "olmo3"])
def test_engine_windowed_paged_attention(tmp_path, kind):
  """LLMEngine (ragged steps, paged cache: the windowed paged-attention kernel) gives Llama.generate's tokens on CPU and METAL."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from fusezap.device import Device
  p = _model_dir(tmp_path / kind, kind, seed=2)
  want = _generate_tokens(p, "CPU")
  assert _engine_tokens(p, "CPU") == want
  if Device.available("METAL"): assert _engine_tokens(p, "METAL") == want

def _neuron_tier() -> str:
  return next((t for t in os.getenv("FZ_TEST_TIERS", "").split(",") if t.strip() in ("NEURON", "NKISIM")), "NKISHIM").strip()

@pytest.mark.parametrize("kind", ["gemma3", "olmo3"])
@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_neuron_engine_windowed(tmp_path, kind, dt, monkeypatch):
  """NEURON tier: the windowed layers' keyrange attention (their own lo per row in the step metadata), per-layer RoPE rows,
  post-norms on the members. float32 tokens == CPU float32's; bf16 near them."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  p = _model_dir(tmp_path / kind, kind, seed=3)
  cpu = _engine_tokens(p, "CPU")
  if dt == "float32":
    assert _engine_tokens(p, _neuron_tier()) == cpu
    return
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  got = _engine_tokens(p, _neuron_tier(), dtypes.bfloat16)
  prefix = [next((i for i, (x, y) in enumerate(zip(got[r], cpu[r])) if x != y), len(cpu[r])) for r in cpu]
  assert sum(prefix) >= 0.6 * sum(len(t) for t in cpu.values()), (prefix, got, cpu)
