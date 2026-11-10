"""Llama-3.1-8B-Instruct (tracker I13: the second model on Inferentia2), checked on the Mac without the gated weights:

  * a random-weight Llama-3.1 checkpoint (the published config.json and generation_config.json, only the sizes shrunk:
    no q/k/v biases, GQA 4 query heads per kv head, eps 1e-5, theta 5e5, llama3 RoPE scaling, untied lm_head, three
    EOS ids) loads through load_model and gives HF transformers' float32 greedy tokens on CPU, with the real 128,256-token
    vocabulary and with RoPE scaling that reaches every frequency band at short positions;
  * rope_inv_freq / rope_tables with the 8B's llama3 scaling against HF's LlamaRotaryEmbedding (head_dim 128, positions
    up to 32k, both config formats: the published `rope_scaling` and transformers 5's `rope_parameters`);
  * EOS: config.json + generation_config.json give the engine 128001 / 128008 / 128009 as stop ids, and the engine stops
    on each of them; the chat template (the cached Llama 3.1 tokenizer, when there is one) renders HF's ids;
  * the NEURON path (NKISHIM) at the 8B's head layout (32 / 8 heads, head_dim 128; 16 / 4 per TP=2 rank) without q/k/v
    biases: float32 and bf16 engine tokens equal CPU's, TP=2 equals TP=1, the decoder-layer kernels (LayerSpec, tiled and
    TP halves) give the members' bits, the prefill kernels v2 the members' logits, and the 8B's shapes fit SBUF / PSUM.
"""
from __future__ import annotations
import json, math, os
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve.sampling import Sampler

# meta-llama/Llama-3.1-8B-Instruct config.json and generation_config.json as published (revision 0e9e39f2); the tests
# shrink only the sizes
LLAMA31_8B = {"architectures": ["LlamaForCausalLM"], "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 128000,
  "eos_token_id": [128001, 128008, 128009], "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02,
  "intermediate_size": 14336, "max_position_embeddings": 131072, "mlp_bias": False, "model_type": "llama", "num_attention_heads": 32,
  "num_hidden_layers": 32, "num_key_value_heads": 8, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
  "rope_scaling": {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"},
  "rope_theta": 500000.0, "tie_word_embeddings": False, "torch_dtype": "bfloat16", "use_cache": True, "vocab_size": 128256}
GEN31 = {"bos_token_id": 128000, "do_sample": True, "eos_token_id": [128001, 128008, 128009], "temperature": 0.6, "top_p": 0.9}
EOS31 = [128001, 128008, 128009]
# the 8B's head layout at small widths (the NKISHIM tests): 32 / 8 heads, head_dim 128 (16 / 4 per TP=2 rank)
L8B = {"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128}

def _llama3_dir(path, seed: int = 0, vocab: int = 128256, **over) -> str:
  """A random-weight LlamaForCausalLM saved by HF, then config.json / generation_config.json rewritten in the published
  format (legacy `rope_scaling`, the eos list), sizes from `over`."""
  import torch, transformers as tr
  cfg = {**LLAMA31_8B, "hidden_size": 256, "intermediate_size": 512, "num_hidden_layers": 2, "head_dim": 16, "vocab_size": vocab,
         "initializer_range": 0.08, **over}
  torch.manual_seed(seed)
  hfm = tr.LlamaForCausalLM(tr.LlamaConfig(**cfg)).eval()
  assert not any(n.endswith(("q_proj.bias", "k_proj.bias", "v_proj.bias", "o_proj.bias")) for n, _ in hfm.named_parameters())
  hfm.save_pretrained(path, safe_serialization=True)
  with open(os.path.join(path, "config.json"), "w") as f: json.dump(cfg, f, indent=2)
  with open(os.path.join(path, "generation_config.json"), "w") as f: json.dump(GEN31, f, indent=2)
  return str(path)

def _hf(path):
  import torch, transformers as tr
  return tr.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()

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

# ---------------------------------------------------------------- CPU: greedy tokens vs HF transformers

@pytest.mark.parametrize("case", ["published", "short_original_len"])
def test_tiny_llama31_greedy_matches_hf(tmp_path, case):
  """The random Llama-3.1 checkpoint gives HF's float32 greedy tokens for 32 steps on CPU. published: the 8B's rope
  scaling and the full 128,256-token vocabulary (lm_head 128256 x 256). short_original_len: original_max_position_embeddings
  16, so the scaled and the smoothed bands are both reached within the test's 52 positions (and the logits differ from
  the same model without scaling: the scaling is not a no-op here)."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from fusezap.serve.models import load_model
  from fusezap.serve.models.llama import rope_inv_freq
  over = {} if case == "published" else {"rope_scaling": {**LLAMA31_8B["rope_scaling"], "original_max_position_embeddings": 16}}
  vocab = 128256 if case == "published" else 2048
  p = _llama3_dir(tmp_path / case, seed=1, vocab=vocab, **over)
  hfm = _hf(p)
  m = load_model(p, dtype=dtypes.float32, device="CPU", max_len=128, prefill_chunk=64)
  c = m.cfg
  assert (c.model_type, c.n_heads, c.n_kv_heads, c.head_dim, c.rms_eps, c.rope_theta, c.tie_embeddings, c.vocab_size) == \
    ("llama", 32, 8, 16, 1e-5, 5e5, False, vocab)
  assert c.rope_scaling["rope_type"] == "llama3" and c.eos_token_ids == EOS31 and m.layers[0].bqkv is None and m.layers[0].bo is None
  np.testing.assert_allclose(rope_inv_freq(c), hfm.model.rotary_emb.inv_freq.numpy(), rtol=1e-6)
  ids = [128000 if vocab > 128000 else 1] + [int(i) for i in np.random.default_rng(0).integers(3, min(vocab, 128000), 20)]
  ref = _hf_greedy(hfm, ids, 32)
  mine = list(m.generate(ids, 32, Sampler(), stop=[]))
  assert mine == ref, f"first mismatch at {next((i for i, (a, b) in enumerate(zip(mine, ref)) if a != b), len(mine))}"
  assert len(set(ref)) > 3, "degenerate reference: raise initializer_range"
  if case == "short_original_len":   # the scaling matters at these positions: without it the last logits move well past float noise
    import torch
    with torch.no_grad(): want = hfm(torch.tensor([ids + ref])).logits[0, -1].numpy()
    from fusezap.serve.models.llama import Llama
    from fusezap.serve.weights import load_safetensors
    m.reset()
    got = m.prefill([ids + ref])[0]
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
    c.rope_scaling = None
    m2 = Llama(c, load_safetensors(p), dtypes.float32, "CPU", max_len=128, prefill_chunk=64)
    assert np.abs(m2.prefill([ids + ref])[0] - want).max() > 100 * np.abs(got - want).max() + 1e-3

# ---------------------------------------------------------------- RoPE: llama3 scaling vs HF

@pytest.mark.parametrize("fmt", ["rope_scaling", "rope_parameters"])
def test_llama3_rope_tables_match_hf(fmt):
  """The 8B's llama3 frequency scaling (factor 8, low 1, high 4, original length 8192, theta 5e5, head_dim 128):
  rope_inv_freq equals HF's inverse frequencies, and the cos / sin tables HF's LlamaRotaryEmbedding at positions up to
  32k (past the original 8192), from the published config format and from transformers 5's rope_parameters."""
  torch = pytest.importorskip("torch")
  tr = pytest.importorskip("transformers")
  from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
  from fusezap.serve.models.llama import LlamaConfig, rope_inv_freq, rope_tables
  hc = tr.LlamaConfig(**{k: v for k, v in LLAMA31_8B.items() if k not in ("architectures", "torch_dtype")})
  raw = dict(LLAMA31_8B) if fmt == "rope_scaling" else hc.to_dict()
  if fmt == "rope_parameters": assert "rope_parameters" in raw and raw.get("rope_scaling") is None or raw["rope_scaling"] == raw.get("rope_parameters")
  c = LlamaConfig.from_hf(raw)
  assert c.rope_theta == 5e5 and c.rope_scaling["rope_type"] == "llama3" and c.rope_scaling["original_max_position_embeddings"] == 8192
  rot = LlamaRotaryEmbedding(hc)
  assert getattr(rot, "attention_scaling", 1.0) == 1.0   # llama3 scaling changes only the frequencies
  inv = rope_inv_freq(c)
  np.testing.assert_allclose(inv, rot.inv_freq.numpy(), rtol=1e-6)
  plain = rope_inv_freq(LlamaConfig.from_hf({**LLAMA31_8B, "rope_scaling": None}))
  assert (inv < plain).sum() >= 16 and (inv == plain).sum() >= 16   # both the scaled and the untouched bands exist
  pos = np.array([0, 1, 7, 100, 2047, 4095, 8191, 8192, 9000, 16383, 32767])
  cos, sin = rope_tables(c, 32768)
  hc_, hs_ = rot(torch.zeros(1, len(pos), 128), torch.tensor(pos)[None])
  # numpy's float32 pow and torch's differ by one ulp in 2 of the 64 base frequencies (the unscaled band; Qwen2.5-7B's theta
  # 1e6 has one such frequency too): a one-ulp angle difference, which grows with the position (~1e-3 in sin at 32767).
  # torch's vectorised sin / cos also differ by CPU (x86 CI runners vary), so the bar has 3x headroom over that: a wrong
  # scaling band or theta is off by O(1) at these positions, far above it.
  atol = (2e-5 + 1.2e-7 * pos)[:, None]
  dc, ds = np.abs(cos[pos] - hc_[0, :, :64].numpy()), np.abs(sin[pos] - hs_[0, :, :64].numpy())
  assert (dc <= atol).all() and (ds <= atol).all(), f"max excess over the bar: cos {(dc - atol).max():.3g}, sin {(ds - atol).max():.3g}"

# ---------------------------------------------------------------- EOS ids and the chat template

def test_eos_ids_and_engine_stops_on_each(tmp_path):
  """config.json's and generation_config.json's eos lists become the engine's stop ids (128001 <|end_of_text|>, 128008
  <|eom_id|>, 128009 <|eot_id|>). The engine stops on each: the lm_head rows of the third greedy token and of the stop
  id are swapped, so the run's third token becomes that id (the first two are unchanged) and it finishes with 'stop'."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.weights import load_safetensors
  from test_tp import write_safetensors
  p = _llama3_dir(tmp_path / "e", seed=2)
  ws = {k: np.array(v, dtype=np.float32) for k, v in load_safetensors(p).items()}
  prompt = [128000, 128006, 882, 128007, 271, 9906, 128009, 128006, 78191, 128007, 271]
  def run(path, ignore_eos):
    m = load_model(path, dtype=dtypes.float32, device="CPU", max_len=128, prefill_chunk=64)
    eng = LLMEngine(m, num_blocks=16, block_size=16, max_num_seqs=2, max_prefill_tokens=64, max_model_len=128, decode_beam=0)
    assert eng.stop_ids == set(EOS31)
    rid = eng.add_request(prompt, SamplingParams(max_tokens=8, ignore_eos=ignore_eos))
    last = None
    while eng.has_unfinished():
      for o in eng.step(): last = o
    return last.token_ids, last.finish_reason
  base, why = run(p, True)
  assert why == "length" and len(base) == 8 and not set(base[:3]) & set(EOS31)
  for eos in EOS31:
    w = dict(ws)
    h = w["lm_head.weight"].copy()
    h[[base[2], eos]] = h[[eos, base[2]]]
    w["lm_head.weight"] = h
    d = tmp_path / f"stop{eos}"
    os.makedirs(d)
    for f in ("config.json", "generation_config.json"):
      with open(os.path.join(p, f)) as a, open(d / f, "w") as b: b.write(a.read())
    write_safetensors(str(d / "model.safetensors"), w)
    toks, why = run(str(d), False)
    assert why == "stop" and toks[:2] == base[:2] and toks[-1] == eos and len(toks) == 3, (eos, toks, why)

def _cached_llama31_tokenizer() -> str | None:
  from huggingface_hub import try_to_load_from_cache
  for repo in ("meta-llama/Llama-3.1-8B-Instruct", "unsloth/Meta-Llama-3.1-8B-Instruct"):
    f = [try_to_load_from_cache(repo, n) for n in ("tokenizer.json", "tokenizer_config.json", "generation_config.json")]
    if all(isinstance(x, str) for x in f): return os.path.dirname(f[0])
  return None

def test_llama31_chat_template_and_stop_ids():
  """The Llama 3.1 tokenizer from the HF cache (meta-llama's, else unsloth's copy of the same files; skips when neither is
  cached: the job checks the real one on the instance): FuseZap's chat rendering gives HF apply_chat_template's ids for
  a user turn, a system + user turn, and a multi-turn conversation; the stop ids are 128001 / 128008 / 128009."""
  tr = pytest.importorskip("transformers")
  path = _cached_llama31_tokenizer()
  if path is None: pytest.skip("no Llama 3.1 tokenizer in the HF cache")
  from fusezap.serve.tokenizer import Tokenizer
  tok, hft = Tokenizer(path), tr.AutoTokenizer.from_pretrained(path)
  assert tok.stop_token_ids == EOS31 and tok.bos_token_id == 128000
  convs = [[{"role": "user", "content": "Write a short poem about the sea."}],
           [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "What is 2+2?"}],
           [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"}, {"role": "user", "content": "Name a color."}]]
  for msgs in convs:
    want = hft.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    want = list(want["input_ids"] if hasattr(want, "keys") else want)
    got = tok.apply_chat_template(msgs, add_generation_prompt=True)
    assert got == want, (msgs, tok.decode(got)[:400])
    assert got[0] == 128000 and got[-4:] == [128006, 78191, 128007, 271]   # <|start_header_id|>assistant<|end_header_id|>\n\n

# ---------------------------------------------------------------- NEURON (NKISHIM): the 8B's head layout, no q/k/v biases

def _neuron_tier() -> str:
  return next((t for t in os.getenv("FZ_TEST_TIERS", "").split(",") if t.strip() in ("NEURON", "NKISIM")), "NKISHIM").strip()

def _l8b_dir(path, seed: int, layers: int = 2, **over) -> str:
  """tiny_model_dir (tests/test_tp.py) with the 8B's head layout, no biases at all, and the 8B's rope / eps / eos."""
  from test_tp import tiny_model_dir
  cfg = {**L8B, "num_hidden_layers": layers, "model_type": "llama", "rms_norm_eps": 1e-5, "rope_theta": 5e5,
         "rope_scaling": dict(LLAMA31_8B["rope_scaling"], original_max_position_embeddings=64), "eos_token_id": 63,
         "bos_token_id": 1, "max_position_embeddings": 131072, **over}
  return tiny_model_dir(str(path), seed=seed, cfg=cfg, bias=False)

PROMPTS = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7], [5] * 20, list(range(30, 60))]

def _engine_tokens(path: str, device: str, dtype=dtypes.float32, **kw) -> dict:
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  m = load_model(path, dtype=dtype, device=device, max_len=128, prefill_chunk=kw.pop("prefill_chunk", 128))
  eng = LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0, **kw)
  return dict(enumerate(eng.generate(PROMPTS, SamplingParams(max_tokens=10, ignore_eos=True))))

@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_neuron_engine_tokens_at_8b_head_layout(tmp_path, dt, monkeypatch):
  """LLMEngine on the NEURON tier (decode layers as LayerSpec, the device head, prefill-first) with the 8B's head
  layout and no q/k/v biases. float32: CPU float32's greedy tokens. bf16 (I68: bf16 K-major weights): the layer kernel's
  tokens equal the bf16 members' (FZ_NEURON_LAYER=v1), and stay near CPU float32's (this random checkpoint has near
  ties: seeds 3 / 4 keep 65-75% of the tokens before the first flip, with the members alike; Qwen's layout 90-100%)."""
  tier = _neuron_tier()
  p = _l8b_dir(tmp_path / "m", seed=3)
  cpu = _engine_tokens(p, "CPU")
  if dt == "float32":
    assert _engine_tokens(p, tier) == cpu
    return
  monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
  got = _engine_tokens(p, tier, dtypes.bfloat16)
  monkeypatch.setenv("FZ_NEURON_LAYER", "v1")
  assert _engine_tokens(p, tier, dtypes.bfloat16) == got
  prefix = [next((i for i, (x, y) in enumerate(zip(got[r], cpu[r])) if x != y), len(cpu[r])) for r in cpu]
  assert sum(prefix) >= 0.6 * sum(len(t) for t in cpu.values()), (prefix, got, cpu)

LAYER_8B = {   # D, I, Hq, Hkv, hd, rows, pool slots, block size, prefix, residual term, o_proj bias, q/k/v bias
  "llama8b_heads_1row": (256, 384, 32, 8, 128, 1, 512, 16, None, True, False, False),
  "llama8b_heads_4rows": (256, 384, 32, 8, 128, 4, 1024, 16, 768, True, False, False),
  "llama8b_shard_heads_4rows": (256, 384, 16, 4, 128, 4, 1024, 16, None, True, False, False),
}
# 8 kv heads never stack (8 x 32 rows > 128 partitions: neuron_kv falls back to heads "each", which _layer_pair's
# variant check would flag), so the full-model cases take the "each" variants; the TP=2 shard (4 kv heads) the stacked ones
# (the full layout's 32 query heads also make the untiled layer too wide even at D 256: it runs tiled, v2et)
LAYER_8B_RUNS = [(c, "v2et") for c in ("llama8b_heads_1row", "llama8b_heads_4rows")] + \
                [("llama8b_shard_heads_4rows", v) for v in ("v2", "v2s", "v2se", "v2set", "v2set2", "v2set4")]

@pytest.mark.parametrize("case,variant", LAYER_8B_RUNS)
def test_layer_kernel_at_8b_head_layout_matches_the_members(case, variant, monkeypatch):
  """LayerSpec (untiled and tiled, heads each / stacked) without a q/k/v bias at the 8B's head layout: the members' bits
  (tests/test_neuron_serve.py's check; Qwen's cases all carry the bias)."""
  from test_neuron_serve import _layer_pair
  from fusezap.codegen import nki_kernels as nk
  if variant[2:].strip("sa") and _neuron_tier() == "NKISHIM": monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  D, I, Hq, Hkv, hd, rows = LAYER_8B[case][:6]
  assert nk.layer_ok(rows, Hq, Hkv, hd, "stack") == (Hkv == 4)
  for k, (a, b) in _layer_pair(*LAYER_8B[case][:11], variant, monkeypatch, qbias=LAYER_8B[case][11]).items():
    assert np.array_equal(a.view(np.int32), b.view(np.int32)), f"{case} {variant} {k}: max |diff| {np.abs(a - b).max()}"

def test_8b_layer_sbuf_and_the_width_step_down():
  """The SBUF estimate (nki_bf16.layer_sbuf_bytes, budget 176 KiB) at Llama-3.1-8B's real layer, 4 rows. On one NeuronCore
  (D 4096, I 14336, 32 / 8 heads) neither the untiled nor the tiled layer fits (the members run it: one reason the 8B
  runs at TP=2). The TP=2 shard (I 7168, 16 / 4 heads) fits tiled at 512 slots at every width and at 1024 slots up to 1024
  weight columns: 2048 needs 180,608 bytes, 384 over (Qwen2.5-7B's shard: 133 KiB), so NeuronKVCache.layer steps the width
  down to 1024 (same bits) instead of running the members."""
  from fusezap.codegen import nki_kernels as nk
  from fusezap.codegen import nki_bf16 as nb
  sp = lambda Hq, Hkv, I, P, **k: nk.LayerSpec(4, 4096, Hq, Hkv, 128, I, P, 1e-5, 1 / math.sqrt(128), 64, True, False, False,
                                               "each" if Hkv == 8 else "stack", "slice", "mv", True, 0, **k)
  for P in (512, 1024):
    assert not nb.layer_fits(sp(32, 8, 14336, P)) and not nb.layer_fits(sp(32, 8, 14336, P, tile=True, wcols=512))
    assert not nb.layer_fits(sp(16, 4, 7168, P))
    for wc in (512, 1024, 2048):
      assert nb.layer_fits(sp(16, 4, 7168, P, tile=True, wcols=wc)) == (P == 512 or wc <= 1024), (P, wc)
  assert nb.layer_sbuf_bytes(sp(16, 4, 7168, 1024, tile=True, wcols=2048)) == 180608 == nb.LAYER_SBUF + 384

def test_tiled_layer_width_steps_down_in_the_engine(tmp_path, monkeypatch):
  """NeuronKVCache.layer with FZ_NEURON_LAYER_WCOLS=2048 and a budget the 2048-column tiled layer exceeds but the 1024-column
  one meets: the step runs the tiled layer at 1024 columns (not the members), with the same tokens as 2048 columns."""
  from fusezap.codegen import nki_bf16 as nb
  from fusezap import jit as J
  p = _l8b_dir(tmp_path / "w", seed=4, num_attention_heads=16, num_key_value_heads=4)
  monkeypatch.setenv("FZ_NEURON_LAYER_TILE", "1"); monkeypatch.setenv("FZ_NEURON_LAYER_WCOLS", "2048")
  ran: list = []
  real = J.run_item
  def spy(si, *a, **k):
    sp = getattr(si.ast[0], "spec", None) if si.ast else None
    while sp is not None and not hasattr(sp, "wcols") and hasattr(sp, "base"): sp = sp.base
    if sp is not None and getattr(sp, "tile", False): ran.append(sp)
    return real(si, *a, **k)
  monkeypatch.setattr(J, "run_item", spy)
  monkeypatch.setenv("FZ_NEURON_CTX", "0")   # every step at the whole pool: one spec per row count
  want = _engine_tokens(p, _neuron_tier())
  assert ran and {s.wcols for s in ran} == {2048}
  import dataclasses
  big = max(ran, key=nb.layer_sbuf_bytes)
  assert nb.layer_sbuf_bytes(dataclasses.replace(big, wcols=1024)) < nb.layer_sbuf_bytes(big)
  monkeypatch.setattr(nb, "LAYER_SBUF", nb.layer_sbuf_bytes(dataclasses.replace(big, wcols=1024)))   # 1024 columns fit, 2048 do not
  ran.clear()
  assert _engine_tokens(p, _neuron_tier()) == want
  assert ran and {s.wcols for s in ran if s.N == big.N} == {1024}, {(s.N, s.wcols) for s in ran}

# ---------------------------------------------------------------- the inf2 job (scripts/nki_llama8b_inf2.sh)

def test_llama8b_job_script():
  """The I13 job: valid bash, launched with --hf-token (the gated repo), pinned to the revision the committed reference was
  built from, and on exactly the Qwen2.5-7B runs' NEURON switches (nki_n8_prefill_inf2.sh's FZ7B_ENV / LAYER_ENV)."""
  import re, subprocess
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  src = open(os.path.join(root, "scripts", "nki_llama8b_inf2.sh")).read()
  q = open(os.path.join(root, "scripts", "nki_n8_prefill_inf2.sh")).read()
  assert subprocess.run(["bash", "-n", os.path.join(root, "scripts", "nki_llama8b_inf2.sh")]).returncode == 0
  assert re.search(r'aws_remote\.sh job --type inf2\.8xlarge .*--hf-token -- "bash scripts/nki_llama8b_inf2\.sh"', src)
  get = lambda s, k: re.search(rf'^export {k}="([^"]*)"', s, re.M).group(1)
  assert get(src, "LAYER_ENV") == get(q, "LAYER_ENV")
  assert get(src, "FZ8B_ENV") == get(q, "FZ7B_ENV")
  rev = re.search(r"REV=([0-9a-f]{40})", src).group(1)
  ref = os.path.join(root, "docs", "invarix", "examples", "refs", "llama-3.1-8b-instruct-f32.json")
  if os.path.exists(ref):
    r = json.load(open(ref))
    assert (r["model"], r["revision"], r["dtype"], len(r["prompts"])) == ("meta-llama/Llama-3.1-8B-Instruct", rev, "float32", 24)
    assert r["eos_token_ids"] == EOS31
  assert "hf_" not in src.replace("hf_hub_download", "")   # no token in the script
