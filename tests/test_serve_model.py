"""fusezap.serve model package (P4-MODEL): Llama-arch decoder vs HF transformers, weights, tokenizer, sampling.

The model tests run Qwen/Qwen2.5-0.5B-Instruct from the local HF cache on METAL (FZ_DEVICE overrides) against the
transformers float32 torch CPU reference; they skip if the weights or transformers are not available.
"""
from __future__ import annotations
import json, os
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve.sampling import Sampler, apply_repetition_penalty, top_k_filter, top_p_filter, softmax
from fusezap.serve.weights import read_safetensors, load_safetensors, to_dtype, bf16_to_f32, f32_to_bf16

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = [{"role": "user", "content": "Write a short poem about the sea."}]

def _model_path() -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(MODEL, allow_download=False)
  except Exception: return None

def _device() -> str:
  from fusezap.device import Device
  want = os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
  return want

needs_model = pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")

# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def hf():
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  m = tr.AutoModelForCausalLM.from_pretrained(_model_path(), dtype=torch.float32).eval()
  t = tr.AutoTokenizer.from_pretrained(_model_path())
  return m, t

@pytest.fixture(scope="module")
def tok():
  from fusezap.serve.tokenizer import Tokenizer
  return Tokenizer.from_pretrained(_model_path())

@pytest.fixture(scope="module")
def model():
  from fusezap.serve.models import load_model
  return load_model(MODEL, dtype=dtypes.float32, device=_device(), max_len=256, prefill_chunk=128)

@pytest.fixture(scope="module")
def prompt_ids(tok):
  return tok.apply_chat_template(PROMPT, add_generation_prompt=True)

def _hf_greedy(hfm, ids: list[int], n: int, repetition_penalty: float = 1.0) -> list[int]:
  """Plain greedy loop with HF's KV cache (no generation_config defaults involved)."""
  import torch
  out, past, inp, hist = [], None, torch.tensor([ids]), list(ids)
  with torch.no_grad():
    for _ in range(n):
      o = hfm(inp, past_key_values=past, use_cache=True)
      past, lg = o.past_key_values, o.logits[0, -1].numpy()
      t = int(np.argmax(apply_repetition_penalty(lg, hist, repetition_penalty)))
      out.append(t); hist.append(t)
      inp = torch.tensor([[t]])
  return out

def _hf_forward(hfm, ids: list[int]):
  import torch
  with torch.no_grad(): return hfm(torch.tensor([ids]), output_hidden_states=True)

# ---------------------------------------------------------------- model vs HF

@needs_model
def test_greedy_64_tokens_match_hf(hf, model, prompt_ids):
  ref = _hf_greedy(hf[0], prompt_ids, 64)
  mine = list(model.generate(prompt_ids, 64, Sampler(), stop=[]))
  assert len(mine) == 64
  assert mine == ref, f"first mismatch at {next(i for i, (a, b) in enumerate(zip(mine, ref)) if a != b)}"

@needs_model
def test_per_layer_logits_close_to_hf(hf, model, prompt_ids):
  """Full-sequence forward: every layer's hidden state, its logit-lens logits (final norm + lm_head) and the final
  logits vs HF float32: logits atol 1e-3; hidden states (up to ~1.7e3 in Qwen) rtol 1e-4, atol 1e-3 + 2e-5 * max|h|."""
  from fusezap import Tensor
  hfm = hf[0]
  o = _hf_forward(hfm, prompt_ids)
  model.reset()
  hs: list = []
  x = Tensor(model.embed(np.array([prompt_ids])), device=model.device, dtype=model.dtype)
  lg = model.forward(x, model.cache.bind(model.cache.state()), 0, len(prompt_ids), last_only=False, hiddens=hs)
  assert len(hs) == len(o.hidden_states) == model.cfg.n_layers + 1
  import torch
  for i, (a, b) in enumerate(zip(hs, o.hidden_states)):
    a, b = a.numpy(), b[0].numpy()
    np.testing.assert_allclose(a[0], b, rtol=1e-4, atol=1e-3 + 2e-5 * float(np.abs(b).max()), err_msg=f"hidden state {i}")
    # logit lens: HF's last hidden state is already normed; earlier ones go through the final norm
    with torch.no_grad():
      hb = torch.tensor(b) if i == len(hs) - 1 else hfm.model.norm(torch.tensor(b))
      ref_lg = hfm.lm_head(hb).numpy()
    t = Tensor(a, device=model.device)
    ours = (t if i == len(hs) - 1 else model._rmsnorm(t, model.norm)).linear(model.lm_head).numpy()[0]
    np.testing.assert_allclose(ours, ref_lg, rtol=0, atol=1e-3, err_msg=f"logit-lens logits of layer {i}")
  np.testing.assert_allclose(lg.numpy()[0], o.logits[0].numpy(), rtol=0, atol=1e-3)

@needs_model
def test_decode_logits_teacher_forced(hf, model, prompt_ids):
  """Jitted prefill (padded bucket) + jitted decode steps vs HF's full forward over the same tokens."""
  cont = [785, 9396, 11, 264, 12767, 323, 6822, 1717, 7947, 345, 32, 1992]
  R = _hf_forward(hf[0], prompt_ids + cont).logits[0].numpy()
  n = len(prompt_ids)
  model.reset()
  lg = model.prefill([prompt_ids])
  np.testing.assert_allclose(lg[0], R[n - 1], atol=1e-3, rtol=0)
  for i, t in enumerate(cont[:-1]):
    lg = model.decode([[t]])
    np.testing.assert_allclose(lg[0], R[n + i], atol=1e-3, rtol=0, err_msg=f"decode step {i}")

@needs_model
def test_chunked_and_continued_prefill(hf, model, prompt_ids):
  """Prefill in chunks of 16 (several buckets, symbolic start), and a second turn prefilled at start > 0, give the
  same logits as HF on the whole sequence."""
  turn2 = [151645, 198, 151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198]
  R = _hf_forward(hf[0], prompt_ids + turn2).logits[0].numpy()
  model.reset()
  lg = model.prefill([prompt_ids], chunk=16)
  np.testing.assert_allclose(lg[0], R[len(prompt_ids) - 1], atol=1e-3, rtol=0)
  lg = model.prefill([turn2])
  assert model.pos == len(prompt_ids) + len(turn2)
  np.testing.assert_allclose(lg[0], R[-1], atol=1e-3, rtol=0)

@needs_model
def test_decode_replays_without_tracing(model, prompt_ids):
  """After capture, decode steps neither run the traced step function nor lower any kernel."""
  import fusezap.codegen.lower as L
  model.reset()
  model.prefill([prompt_ids])
  for t in (1, 2, 3): model.decode([[t]])   # capture happened on an earlier test or here
  jf = model._decode_jit
  assert jf.captured is not None
  calls, fn, lower = [0, 0], jf.fn, L.lower
  def traced(*a, **k):
    calls[0] += 1
    return fn(*a, **k)
  def lowered(*a, **k):
    calls[1] += 1
    return lower(*a, **k)
  jf.fn, L.lower = traced, lowered
  try:
    for t in (4, 5, 6): model.decode([[t]])
  finally: jf.fn, L.lower = fn, lower
  assert calls == [0, 0], f"decode replay traced {calls[0]} and lowered {calls[1]} times"

@needs_model
def test_repetition_penalty_greedy_matches_hf(hf, model, prompt_ids):
  """Qwen's generation_config uses repetition_penalty=1.1: greedy with it matches HF's processor on the same history."""
  ref = _hf_greedy(hf[0], prompt_ids, 24, repetition_penalty=1.1)
  import torch
  from transformers import RepetitionPenaltyLogitsProcessor
  lg = np.random.default_rng(0).standard_normal((1, 50)).astype(np.float32)
  hist = [3, 7, 7, 20, 49]
  want = RepetitionPenaltyLogitsProcessor(1.1)(torch.tensor([hist]), torch.tensor(lg)).numpy()
  np.testing.assert_allclose(apply_repetition_penalty(lg, hist, 1.1), want, rtol=1e-6)
  mine = list(model.generate(prompt_ids, 24, Sampler(repetition_penalty=1.1), stop=[]))
  assert mine == ref

@needs_model
def test_float16_top1_agreement(hf, prompt_ids):
  """f16 weights/activations: the top-1 token matches the f32 HF reference at >= 95% of teacher-forced positions."""
  from fusezap import Tensor
  from fusezap.serve.models import load_model
  seq = prompt_ids + _hf_greedy(hf[0], prompt_ids, 32)
  R = _hf_forward(hf[0], seq).logits[0].numpy()
  m16 = load_model(MODEL, dtype=dtypes.float16, device=_device(), max_len=128, prefill_chunk=64)
  x = Tensor(m16.embed(np.array([seq])), device=m16.device, dtype=dtypes.float16)
  lg = m16.forward(x, m16.cache.bind(m16.cache.state()), 0, len(seq), last_only=False).float().numpy()[0]
  agree = float((lg.argmax(-1) == R.argmax(-1))[len(prompt_ids) - 1:].mean())
  assert agree >= 0.95, agree
  del m16

def test_config_and_rope_vs_hf():
  pytest.importorskip("transformers")
  if _model_path() is None: pytest.skip("no model")
  import torch
  from fusezap.serve.models.llama import LlamaConfig, rope_inv_freq, rope_tables
  from transformers import AutoConfig
  from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
  raw = json.load(open(os.path.join(_model_path(), "config.json")))
  c = LlamaConfig.from_hf(raw)
  assert (c.n_layers, c.n_heads, c.n_kv_heads, c.head_dim, c.tie_embeddings, c.rope_theta) == (24, 14, 2, 64, True, 1e6)
  rot = Qwen2RotaryEmbedding(AutoConfig.from_pretrained(_model_path()))
  np.testing.assert_allclose(rope_inv_freq(c), rot.inv_freq.numpy(), rtol=1e-6)
  cos, sin = rope_tables(c, 300)
  x = torch.zeros(1, 300, 64)
  hc, hs = rot(x, torch.arange(300)[None])
  np.testing.assert_allclose(cos, hc[0, :, :32].numpy(), atol=2e-6)
  np.testing.assert_allclose(sin, hs[0, :, :32].numpy(), atol=2e-6)
  # llama3 rope scaling vs HF's implementation
  from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
  from transformers import LlamaConfig as HFLlamaConfig
  rs = {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}
  hc_ = HFLlamaConfig(hidden_size=4096, num_attention_heads=32, rope_theta=500000.0, rope_scaling=rs, max_position_embeddings=131072)
  assert LlamaConfig.from_hf({**hc_.to_dict(), "rope_parameters": None, "rope_theta": 500000.0, "rope_scaling": rs}).rope_scaling == rs
  inv, _ = ROPE_INIT_FUNCTIONS["llama3"](hc_, "cpu")
  c3 = LlamaConfig.from_hf(hc_.to_dict())   # transformers 5 keeps rope_theta / scaling in rope_parameters
  np.testing.assert_allclose(rope_inv_freq(c3), inv.numpy(), rtol=1e-5)

# mistralai/Mistral-7B-Instruct-v0.3 config.json (MODEL2), as published; the tiny test below shrinks only the sizes
MISTRAL_V03 = {"architectures": ["MistralForCausalLM"], "model_type": "mistral", "hidden_size": 4096, "intermediate_size": 14336,
  "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "vocab_size": 32768,
  "rms_norm_eps": 1e-05, "rope_theta": 1000000.0, "sliding_window": None, "tie_word_embeddings": False, "hidden_act": "silu",
  "max_position_embeddings": 32768, "bos_token_id": 1, "eos_token_id": 2, "attention_dropout": 0.0, "initializer_range": 0.02}

def test_tiny_mistral_greedy_matches_hf(tmp_path):
  """MODEL2's Mac evidence: a random-weight 2-layer MistralForCausalLM with the v0.3 config (no q/k/v biases, 32/8 GQA,
  eps 1e-5, theta 1e6, untied lm_head, no sliding window), only the sizes shrunk, loads through load_model and gives
  HF transformers' float32 greedy tokens for 32 steps."""
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  from fusezap.serve.models import load_model
  cfg = {**MISTRAL_V03, "hidden_size": 256, "intermediate_size": 896, "num_hidden_layers": 2, "head_dim": 8, "vocab_size": 512,
         "initializer_range": 0.08}
  torch.manual_seed(0)
  hfm = tr.MistralForCausalLM(tr.MistralConfig(**cfg)).eval()
  assert not any(n.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias")) for n, _ in hfm.named_parameters())
  hfm.save_pretrained(tmp_path, safe_serialization=True)
  m = load_model(str(tmp_path), dtype=dtypes.float32, device=_device(), max_len=128, prefill_chunk=64)
  c = m.cfg
  assert (c.model_type, c.n_heads, c.n_kv_heads, c.rms_eps, c.rope_theta, c.tie_embeddings, c.sliding_window) == \
    ("mistral", 32, 8, 1e-5, 1e6, False, None)
  ids = [1] + list(np.random.default_rng(0).integers(3, cfg["vocab_size"], 20))
  ref = _hf_greedy(hfm, [int(i) for i in ids], 32)
  mine = list(m.generate([int(i) for i in ids], 32, Sampler(), stop=[]))
  assert mine == ref, f"first mismatch at {next((i for i, (a, b) in enumerate(zip(mine, ref)) if a != b), len(mine))}"
  assert len(set(ref)) > 3, "degenerate reference: raise initializer_range"

# ---------------------------------------------------------------- weights

def test_safetensors_reader(tmp_path):
  st = pytest.importorskip("safetensors.numpy")
  rng = np.random.default_rng(0)
  a, b = rng.standard_normal((3, 5)).astype(np.float32), rng.integers(-5, 5, (4,)).astype(np.int64)
  h = rng.standard_normal((2, 7)).astype(np.float16)
  st.save_file({"a": a, "b": b}, str(tmp_path / "model-00001-of-00002.safetensors"))
  st.save_file({"h": h}, str(tmp_path / "model-00002-of-00002.safetensors"))
  wm = {"a": "model-00001-of-00002.safetensors", "b": "model-00001-of-00002.safetensors", "h": "model-00002-of-00002.safetensors"}
  (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
  w = load_safetensors(str(tmp_path))
  assert sorted(w) == ["a", "b", "h"]
  np.testing.assert_array_equal(w["a"], a); np.testing.assert_array_equal(w["b"], b); np.testing.assert_array_equal(w["h"], h)
  np.testing.assert_array_equal(to_dtype(w["h"], dtypes.float32), h.astype(np.float32))
  one = read_safetensors(str(tmp_path / "model-00001-of-00002.safetensors"))
  assert one["a"].dtype == np.float32 and not one["a"].flags.writeable

def test_bf16_checkpoint_conversion(tmp_path):
  torch = pytest.importorskip("torch")
  stt = pytest.importorskip("safetensors.torch")
  x = torch.randn(64, 33) * 10
  x[0, 0] = float("inf")
  stt.save_file({"w": x.to(torch.bfloat16)}, str(tmp_path / "model.safetensors"))
  w = load_safetensors(str(tmp_path))["w"]
  assert w.dtype == np.uint16 and getattr(w, "st_dtype") == "BF16"
  ref = x.to(torch.bfloat16).float().numpy()
  np.testing.assert_array_equal(to_dtype(w, dtypes.float32), ref)
  np.testing.assert_array_equal(to_dtype(w[3:5], dtypes.float32), ref[3:5])                   # views keep the bf16 tag
  np.testing.assert_array_equal(to_dtype(w[[1, 7]], dtypes.float16), ref[[1, 7]].astype(np.float16))
  np.testing.assert_array_equal(to_dtype(w, dtypes.bfloat16), w)
  y = torch.randn(1000) * 3
  np.testing.assert_array_equal(f32_to_bf16(y.numpy()), y.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
  np.testing.assert_array_equal(bf16_to_f32(f32_to_bf16(ref)), ref)

@needs_model
def test_model_weights_match_torch_load():
  from safetensors.torch import load_file
  w = load_safetensors(_model_path())
  t = load_file(os.path.join(_model_path(), "model.safetensors"))
  assert set(w) == set(t)
  for k in ("model.layers.3.self_attn.q_proj.bias", "model.norm.weight", "model.layers.0.mlp.down_proj.weight"):
    np.testing.assert_array_equal(to_dtype(w[k], dtypes.float32), t[k].float().numpy())

# ---------------------------------------------------------------- tokenizer

CONVS = [
  [{"role": "user", "content": "hi"}],
  [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"},
   {"role": "user", "content": "And 3+3? Ünïcödé ✓ 你好"}],
  [{"role": "user", "content": "weather in Paris?"},
   {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
   {"role": "tool", "content": "18C sunny"}, {"role": "tool", "content": "wind 5 km/h"}],
  [{"role": "user", "content": "a"}, {"role": "system", "content": "late system"}, {"role": "assistant", "content": "b"}],
]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the weather",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

@needs_model
@pytest.mark.parametrize("i", range(len(CONVS)))
@pytest.mark.parametrize("gen", [True, False])
def test_chat_template_matches_hf(hf, tok, i, gen):
  t = hf[1]
  for tools in (None, TOOLS):
    want = t.apply_chat_template(CONVS[i], tools=tools, add_generation_prompt=gen, tokenize=False)
    assert tok.render_chat(CONVS[i], gen, tools) == want
    assert tok.apply_chat_template(CONVS[i], gen, tools) == t(want, add_special_tokens=False)["input_ids"]

@needs_model
def test_tokenizer_roundtrip_and_stream(hf, tok):
  t = hf[1]
  s = "Hello, world! Ünïcödé ✓ 你好 🙂 <|im_end|> tabs\tand\nnewlines  "
  assert tok.encode(s) == t(s)["input_ids"]
  ids = tok.encode(s)
  assert tok.decode(ids) == t.decode(ids) == s
  assert tok.decode(ids, skip_special_tokens=True) == t.decode(ids, skip_special_tokens=True)
  assert tok.stop_token_ids == [151643, 151645] and tok.eos_token_id == 151645
  from fusezap.serve.tokenizer import IncrementalDecoder
  d = IncrementalDecoder(tok, skip_special_tokens=False)
  parts = [d.push(i) for i in ids] + [d.flush()]
  assert "".join(parts) == s
  assert all("�" not in p for p in parts)

# chat templates other than Qwen2.5 go through HF's jinja environment (review regressions)
SMOL_TPL = ("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\n"
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + "
            "message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}{% endif %}")

def test_chatml_template_with_literal_newline_default_system():
  from fusezap.serve.tokenizer import make_template, chatml_default_system, ChatMLTemplate
  want = ("<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n"
          "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
  t = make_template({"chat_template": SMOL_TPL})
  assert not isinstance(t, ChatMLTemplate)              # not the Qwen renderer: no Qwen tool preamble
  assert t.render([{"role": "user", "content": "hi"}], True) == want
  assert t.render([{"role": "user", "content": "hi"}], True, TOOLS) == want   # this template has no tool support
  assert chatml_default_system(SMOL_TPL) == "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
  fb = ChatMLTemplate(chatml_default_system(SMOL_TPL), qwen_tools=False)        # the no-jinja fallback
  assert fb.render([{"role": "user", "content": "hi"}], True) == want
  with pytest.raises(ValueError): fb.render([{"role": "user", "content": "hi"}], True, TOOLS)

def test_jinja_env_matches_hf_globals():
  import datetime
  from fusezap.serve.tokenizer import make_template
  tpl = ("{%- if strftime_now is defined %}{%- set d = strftime_now('%d %b %Y') %}{%- else %}{%- set d = '26 Jul 2024' %}"
         "{%- endif %}{{ bos_token }}Today Date: {{ d }}\n{% for m in messages %}{% if m.role == 'skip' %}{% continue %}{% endif %}"
         "{% generation %}{{ m.content | tojson }}{% endgeneration %}{% if loop.index == 3 %}{% break %}{% endif %}{% endfor %}")
  t = make_template({"chat_template": tpl, "bos_token": {"content": "<s>"}})
  out = t.render([{"role": "skip", "content": "x"}, {"role": "user", "content": "é"}, {"role": "user", "content": "z"},
                  {"role": "user", "content": "never"}], True)
  assert out == f"<s>Today Date: {datetime.datetime.now().strftime('%d %b %Y')}\n\"é\"\"z\""

def _cached_snapshot(repo: str) -> str | None:
  import glob
  from huggingface_hub.constants import HF_HUB_CACHE
  hits = [d for d in glob.glob(os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"), "snapshots", "*"))
          if os.path.exists(os.path.join(d, "tokenizer.json")) and os.path.exists(os.path.join(d, "tokenizer_config.json"))]
  return hits[0] if hits else None

@pytest.mark.parametrize("repo", ["HuggingFaceTB/SmolLM2-135M-Instruct", "unsloth/Llama-3.2-1B-Instruct",
                                  "unsloth/Meta-Llama-3.1-8B-Instruct", "unsloth/llama-3-8b-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"])
def test_other_chat_templates_match_hf(repo):
  tr = pytest.importorskip("transformers")
  path = _cached_snapshot(repo)
  if path is None: pytest.skip(f"{repo} tokenizer not in the HF cache")
  from fusezap.serve.tokenizer import Tokenizer
  ours, ref = Tokenizer(path), tr.AutoTokenizer.from_pretrained(path)
  for conv in ([{"role": "user", "content": "hi"}], CONVS[0]):
    for tools in (None, TOOLS):
      want = ref.apply_chat_template(conv, tools=tools, add_generation_prompt=True, tokenize=False)
      assert ours.render_chat(conv, True, tools) == want, (repo, tools)

def test_resolve_model_skips_consolidated_weights(tmp_path, monkeypatch):
  """A Mistral-style repo ships HF shards and consolidated.safetensors: only the indexed shards are fetched."""
  import huggingface_hub
  from huggingface_hub.utils import filter_repo_objects
  from fusezap.serve import weights
  shards = [f"model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)]
  repo = {"config.json": "{}", "tokenizer.json": "{}", "tokenizer.model": "", "params.json": "{}",
          "model.safetensors.index.json": json.dumps({"weight_map": {f"w{i}": s for i, s in enumerate(shards)}}),
          "consolidated.safetensors": "", "original/consolidated.00.safetensors": "", "README.md": "", **{s: "" for s in shards}}
  fetched: set[str] = set()
  def fake_snapshot_download(repo_id, local_files_only=False, allow_patterns=None, ignore_patterns=None, **kw):
    if local_files_only: raise FileNotFoundError("not cached")
    for fn in filter_repo_objects(list(repo), allow_patterns=allow_patterns, ignore_patterns=ignore_patterns):
      (tmp_path / fn).parent.mkdir(parents=True, exist_ok=True); (tmp_path / fn).write_text(repo[fn]); fetched.add(fn)
    return str(tmp_path)
  monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
  with pytest.raises(FileNotFoundError): weights.resolve_model("org/mistral-like", allow_download=False)
  assert weights.resolve_model("org/mistral-like", allow_download=True) == str(tmp_path)
  assert fetched == set(repo) - {"consolidated.safetensors", "original/consolidated.00.safetensors", "README.md"}
  # no index: every top-level *.safetensors except consolidated*
  del repo["model.safetensors.index.json"]; fetched.clear()
  for f in tmp_path.iterdir():
    if f.is_file(): f.unlink()
  weights.resolve_model("org/mistral-like", allow_download=True)
  assert {f for f in fetched if f.endswith(".safetensors")} == set(shards)

# ---------------------------------------------------------------- sampling

def test_greedy_and_ties():
  lg = np.array([0.1, 3.0, -1.0, 3.0], np.float32)
  assert Sampler()(lg) == 1 and Sampler(temperature=1.0, top_k=1)(lg) == 1
  p = Sampler().probs(lg)
  assert p[1] == 1.0 and p.sum() == 1.0

def test_temperature_distribution():
  lg = np.array([1.0, 2.0, 0.5, -1.0], np.float32)
  for T in (0.5, 1.0, 2.0):
    np.testing.assert_allclose(Sampler(temperature=T).probs(lg), softmax(lg / T), rtol=1e-6)
  s, n = Sampler(temperature=1.0, seed=123), 20000
  counts = np.bincount([s(lg) for _ in range(n)], minlength=4) / n
  np.testing.assert_allclose(counts, softmax(lg), atol=0.015)

def test_seeded_reproducible():
  lg = np.random.default_rng(1).standard_normal(1000).astype(np.float32)
  a = [Sampler(temperature=0.9, top_p=0.9, seed=7)(lg) for _ in range(1)]
  s1, s2, s3 = Sampler(temperature=0.9, top_p=0.9, seed=7), Sampler(temperature=0.9, top_p=0.9, seed=7), Sampler(temperature=0.9, top_p=0.9, seed=8)
  x, y, z = [s1(lg) for _ in range(50)], [s2(lg) for _ in range(50)], [s3(lg) for _ in range(50)]
  assert x == y and x != z and x[0] == a[0]
  s1.reseed(7)
  assert [s1(lg) for _ in range(50)] == x

def test_top_k_top_p_vs_hf():
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  rng = np.random.default_rng(0)
  for _ in range(20):
    lg = rng.standard_normal(500).astype(np.float32) * 3
    for k in (1, 5, 50):
      want = tr.TopKLogitsWarper(k)(None, torch.tensor(lg[None])).numpy()[0]
      np.testing.assert_array_equal(np.isinf(top_k_filter(lg, k)), np.isinf(want))
    for p in (0.1, 0.5, 0.9, 0.99):
      want = tr.TopPLogitsWarper(p)(None, torch.tensor(lg[None])).numpy()[0]
      np.testing.assert_array_equal(np.isinf(top_p_filter(lg, p)), np.isinf(want))
  # combined pipeline matches HF's warper chain on the kept set and the renormalized distribution
  lg = rng.standard_normal(300).astype(np.float32) * 2
  hist = [1, 5, 5, 200]
  s = Sampler(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.1)
  x = torch.tensor(lg[None])
  x = tr.RepetitionPenaltyLogitsProcessor(1.1)(torch.tensor([hist]), x)
  x = tr.TemperatureLogitsWarper(0.7)(None, x)
  x = tr.TopKLogitsWarper(20)(None, x)
  x = tr.TopPLogitsWarper(0.8)(None, x)
  np.testing.assert_allclose(s.probs(lg, hist), torch.softmax(x.double(), -1).numpy()[0], atol=1e-6)

def test_top_k_top_p_samples_stay_in_set():
  lg = np.linspace(-2, 2, 64).astype(np.float32)
  s = Sampler(temperature=1.0, top_k=5, seed=0)
  assert {s(lg) for _ in range(500)} <= set(range(59, 64))
  s = Sampler(temperature=1.0, top_p=0.3, seed=0)
  keep = np.where(~np.isinf(top_p_filter(lg, 0.3)))[0]
  assert {s(lg) for _ in range(500)} <= set(keep.tolist()) and len(keep) >= 1

def test_repetition_penalty_signs():
  lg = np.array([2.0, -2.0, 1.0, 0.0], np.float32)
  out = apply_repetition_penalty(lg, [0, 1, 1], 2.0)
  np.testing.assert_array_equal(out, [1.0, -4.0, 1.0, 0.0])
  np.testing.assert_array_equal(lg, [2.0, -2.0, 1.0, 0.0])  # input untouched
  assert Sampler(repetition_penalty=2.0)(np.array([3.0, 2.0], np.float32), [0]) == 1
