"""OLMoE (tracker I105: a mixture-of-experts model on the Llama-style decoder; fusezap/serve/models/moe.py), on the Mac:

  * the published allenai/OLMoE-1B-7B-0125-Instruct config parses to the dense-masked layout (64 experts x 1024 =
    intermediate 65536, top 8, no renormalization, q/k norm over all heads);
  * moe.route's sort-free top-k (rank by pairwise compares) picks torch.topk's experts, with ties to the lower index;
  * a random-weight OlmoeForCausalLM saved by HF transformers (per-expert checkpoint names, the published structure with
    the sizes shrunk) gives HF's float32 greedy tokens through load_model, on CPU and METAL, with and without
    norm_topk_prob, and the routing matters (the same weights with every expert on give other logits);
  * the NEURON path (NKISHIM) at OLMoE's head layout (16 / 16 heads, head_dim 128): LLMEngine float32 tokens == CPU's, bf16
    near them; the members run (no LayerSpec / prefill kernels: their shapes have no router), and the prefill-kernel self-test
    is skipped for an MoE model;
  * the inf2 job script (scripts/nki_olmoe_inf2.sh): valid bash, the launch command and its cost guard.
"""
from __future__ import annotations
import json, os, re, subprocess
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve.sampling import Sampler

# allenai/OLMoE-1B-7B-0125-Instruct config.json as published (revision b89a7c4b); the tests shrink only the sizes
OLMOE = {"architectures": ["OlmoeForCausalLM"], "attention_bias": False, "attention_dropout": 0.0, "clip_qkv": None,
  "eos_token_id": 50279, "hidden_act": "silu", "hidden_size": 2048, "initializer_range": 0.02, "intermediate_size": 1024,
  "max_position_embeddings": 4096, "model_type": "olmoe", "norm_topk_prob": False, "num_attention_heads": 16, "num_experts": 64,
  "num_experts_per_tok": 8, "num_hidden_layers": 16, "num_key_value_heads": 16, "output_router_logits": False, "pad_token_id": 1,
  "rms_norm_eps": 1e-05, "rope_scaling": None, "rope_theta": 10000.0, "router_aux_loss_coef": 0.01, "tie_word_embeddings": False,
  "torch_dtype": "bfloat16", "use_cache": False, "vocab_size": 50304}
TINY = {"hidden_size": 128, "intermediate_size": 32, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 4,
        "head_dim": 32, "num_experts": 16, "num_experts_per_tok": 4, "vocab_size": 512, "initializer_range": 0.08}

def _olmoe_dir(path, seed: int = 0, **over) -> str:
  """A random-weight OlmoeForCausalLM saved by HF (per-expert tensor names, as the published checkpoint), the q / k norm
  and layer norm weights moved off 1 so a wrong normalization axis shows."""
  import torch, transformers as tr
  cfg = {**OLMOE, **TINY, **over}
  torch.manual_seed(seed)
  hfm = tr.OlmoeForCausalLM(tr.OlmoeConfig(**{k: v for k, v in cfg.items() if k not in ("architectures", "torch_dtype")})).eval()
  with torch.no_grad():
    for n, p in hfm.named_parameters():
      if "norm" in n: p.add_(torch.randn_like(p) * 0.2)
  hfm.save_pretrained(path, safe_serialization=True)
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

def test_olmoe_config_from_hf():
  from fusezap.serve.models.llama import LlamaConfig
  c = LlamaConfig.from_hf(OLMOE)
  assert (c.model_type, c.n_experts, c.top_k, c.norm_topk, c.intermediate_size) == ("olmoe", 64, 8, False, 64 * 1024)
  assert c.qk_norm and c.qk_norm_full and c.sliding_window is None and not c.tie_embeddings
  assert (c.n_heads, c.n_kv_heads, c.head_dim, c.rms_eps, c.rope_theta, c.eos_token_ids) == (16, 16, 128, 1e-5, 1e4, [50279])
  q = LlamaConfig.from_hf({**OLMOE, "model_type": "qwen3", "num_experts": 0})   # other models: unchanged
  assert q.n_experts == 0 and q.qk_norm and not q.qk_norm_full and q.intermediate_size == 1024
  with pytest.raises(AssertionError): LlamaConfig.from_hf({**OLMOE, "clip_qkv": 8.0})

def test_route_matches_torch_topk():
  """route(): torch.topk's experts and probabilities on random rows (64 experts, top 8, as OLMoE), and on rows with ties
  the lower indices first, exactly top_k chosen; norm_topk renormalizes over the chosen ones."""
  torch = pytest.importorskip("torch")
  from fusezap.tensor import Tensor
  from fusezap.serve.models.moe import route, tie_order
  rng = np.random.default_rng(0)
  lg = rng.standard_normal((3, 5, 64)).astype(np.float32) * 2
  p = torch.softmax(torch.tensor(lg), -1)
  tv, ti = torch.topk(p, 8, -1)
  want = torch.zeros_like(p).scatter(-1, ti, tv).numpy()
  tri = tie_order(64, "CPU")
  got = route(Tensor(p.numpy(), device="CPU"), tri, 8, False).numpy()
  assert np.array_equal(got, want)
  gn = route(Tensor(p.numpy(), device="CPU"), tri, 8, True).numpy()
  np.testing.assert_allclose(gn, want / want.sum(-1, keepdims=True), rtol=1e-6)
  ties = np.array([[0.1, 0.3, 0.3, 0.3, 0.0, 0.0], [0.2] * 5 + [0.0]], np.float32)
  got = route(Tensor(ties, device="CPU"), tie_order(6, "CPU"), 2, False).numpy()
  assert np.array_equal(got != 0, np.array([[0, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], bool))

def test_neuron_route_and_mix_matmul_forms_are_exact():
  """NEURON's forms (I105 run 1: the broadcast forms' stride-0 DMA patterns took one DMA descriptor per element, 537 MB of
  DMA rings in a 128-row prefill NEFF at OLMoE's 64 x 1024 experts): the pair operands of the top-k rank and the widened
  routing weights come from exact 0/1 TensorE matmuls, so route() and mix() give the broadcast forms' bits on NKISHIM,
  including tied probabilities."""
  from fusezap.tensor import Tensor
  from fusezap.serve.models.moe import route, mix, tie_order, pair_matrices, repeat_matrix
  rng = np.random.default_rng(1)
  p = rng.random((5, 64)).astype(np.float32); p[2, 10:14] = p[2, 3]   # ties
  p /= p.sum(-1, keepdims=True)
  a = rng.standard_normal((5, 64 * 16)).astype(np.float32)
  T = lambda x: Tensor(x, device="NKISHIM")
  tri = tie_order(64, "NKISHIM")
  for nt in (False, True):
    w0 = route(T(p), tri, 8, nt).numpy()
    w1 = route(T(p), tri, 8, nt, pair_matrices(64, "NKISHIM")).numpy()
    assert np.array_equal(w0.view(np.int32), w1.view(np.int32)) and ((w0 != 0).sum(-1) == 8).all()
    m0, m1 = mix(T(a), T(w0), 64).numpy(), mix(T(a), T(w0), 64, repeat_matrix(64, 16, "NKISHIM")).numpy()
    assert np.array_equal(m0.view(np.int32), m1.view(np.int32))

@pytest.mark.parametrize("dev", ["CPU", "METAL"])
@pytest.mark.parametrize("norm_topk", [False, True])
def test_tiny_olmoe_greedy_matches_hf(tmp_path, dev, norm_topk):
  """The random OLMoE checkpoint gives HF's float32 greedy tokens for 32 steps, and the prompt's last logits within float
  noise of HF's."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from fusezap.device import Device
  if dev == "METAL" and not Device.available("METAL"): pytest.skip("no METAL")
  from fusezap.serve.models import load_model
  p = _olmoe_dir(tmp_path / "m", seed=1, norm_topk_prob=norm_topk)
  hfm = _hf(p)
  m = load_model(p, dtype=dtypes.float32, device=dev, max_len=128, prefill_chunk=64)
  assert m.cfg.n_experts == 16 and m.cfg.intermediate_size == 16 * 32 and m.layers[0].wr is not None and m.layers[0].qn.shape == (128,)
  ids = [int(i) for i in np.random.default_rng(0).integers(3, 512, 20)]
  ref = _hf_greedy(hfm, ids, 32)
  assert len(set(ref)) > 3, "degenerate reference"
  assert list(m.generate(ids, 32, Sampler(), stop=[])) == ref
  import torch
  with torch.no_grad(): want = hfm(torch.tensor([ids])).logits[0, -1].numpy()
  m.reset()
  got = m.prefill([ids])[0]
  np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)
  # the routing matters: every expert on (weight = the softmax probability) moves the logits far past that tolerance
  m.cfg.top_k = m.cfg.n_experts
  m.reset()
  m._prefill_jits.clear()
  assert np.abs(m.prefill([ids])[0] - want).max() > 100 * np.abs(got - want).max() + 1e-3

# ---------------------------------------------------------------- NEURON (NKISHIM)

def _neuron_tier() -> str:
  return next((t for t in os.getenv("FZ_TEST_TIERS", "").split(",") if t.strip() in ("NEURON", "NKISIM")), "NKISHIM").strip()

PROMPTS = [[1, 2, 3, 4, 5, 6, 7], [9, 8, 7], [5] * 20, list(range(30, 60))]
# OLMoE's head layout (16 / 16 heads of 128; HF sizes q_norm by hidden_size, so head_dim = hidden / heads: D 2048)
HEADS = {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 16, "head_dim": 128}

def _engine_tokens(path: str, device: str, dtype=dtypes.float32) -> dict:
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  m = load_model(path, dtype=dtype, device=device, max_len=128, prefill_chunk=128)
  eng = LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=32, max_model_len=128, decode_beam=0)
  return dict(enumerate(eng.generate(PROMPTS, SamplingParams(max_tokens=10, ignore_eos=True))))

@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_neuron_engine_tokens_olmoe(tmp_path, dt, monkeypatch):
  """LLMEngine on the NEURON tier with OLMoE's head layout and 64 experts, top 8: float32 tokens == CPU float32's; bf16
  (I68: bf16 K-major expert weights, float32 router) keeps most of them (near ties of a random checkpoint flip some).
  The decoder layers run the members (cache.layer / the prefill kernels are never called), and the engine's prefill-kernel
  self-test does not run for an MoE model."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from fusezap.serve import neuron_kv, neuron_prefill
  p = _olmoe_dir(tmp_path / "m", seed=3, num_experts=64, num_experts_per_tok=8, intermediate_size=16, **HEADS)
  cpu = _engine_tokens(p, "CPU")
  called = []
  monkeypatch.setattr(neuron_kv.NeuronKVCache, "layer", lambda *a, **k: called.append("layer"))
  monkeypatch.setattr(neuron_prefill, "layer", lambda *a, **k: called.append("pfk"))
  monkeypatch.setattr(neuron_prefill, "selftest", lambda *a, **k: called.append("selftest"))
  if dt == "float32":
    assert _engine_tokens(p, _neuron_tier()) == cpu
  else:
    monkeypatch.setenv("FZ_NEURON_DTYPE", "bf16")
    got = _engine_tokens(p, _neuron_tier(), dtypes.bfloat16)
    prefix = [next((i for i, (x, y) in enumerate(zip(got[r], cpu[r])) if x != y), len(cpu[r])) for r in cpu]
    assert sum(prefix) >= 0.6 * sum(len(t) for t in cpu.values()), (prefix, got, cpu)
  assert not called, called

def test_olmo3_style_tp_is_refused(tmp_path):
  """A dense model with the q / k norm over all heads (OLMo 3's) under tensor parallelism: a clear error (the norm spans
  every rank's heads). An MoE model with a group runs expert parallelism instead (test_ep2_*)."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  import dataclasses
  from fusezap.serve.models.llama import Llama, LlamaConfig
  from fusezap.serve.weights import load_safetensors, load_config
  p = _olmoe_dir(tmp_path / "m", seed=2)
  cfg = dataclasses.replace(LlamaConfig.from_hf(load_config(p)), n_experts=0)
  tp = type("TP", (), {"world": 2, "rank": 0})()
  with pytest.raises(NotImplementedError, match="tensor parallelism"):
    Llama(cfg, load_safetensors(p), dtypes.float32, "CPU", max_len=64, tp=tp)

# ---------------------------------------------------------------- the inf2 job (scripts/nki_olmoe_inf2.sh)

def test_olmoe_job_script():
  """Valid bash; the launch command with its cost guard (inf2.8xlarge at ~$2.0/h incl. disk and IPv4, <= $3 a run); the
  pinned revision; the MoE model on one NeuronCore with the bf16 path."""
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  fn = os.path.join(root, "scripts", "nki_olmoe_inf2.sh")
  src = open(fn).read()
  assert subprocess.run(["bash", "-n", fn]).returncode == 0
  m = re.search(r'scripts/aws_remote\.sh job --type inf2\.8xlarge --disk (\d+) --max-hours ([\d.]+) --max-usd ([\d.]+) -- "bash scripts/nki_olmoe_inf2\.sh"', src)
  assert m, "launch command"
  disk, hours, usd = int(m.group(1)), float(m.group(2)), float(m.group(3))
  assert usd <= 3.0 and (hours + 10 / 60) * 1.996 <= usd + 1e-9, (hours, usd)
  assert re.search(r"REV=b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e", src) and "MODEL=allenai/OLMoE-1B-7B-0125-Instruct" in src
  assert "FZ_NEURON_DTYPE=bf16" in src and "--tp 2" in src   # expert parallel over the two NeuronCores (run 1: one core, HBM)
  env = re.search(r'^export FZMOE_ENV="([^"]*)"', src, re.M).group(1).split()
  for k in ("FZ_NEURON_FUSE=1", "FZ_NEURON_TP_PREFILL=members", "FZ_NEURON_TP_HEAD=vocab", "FZ_NEURON_PREFILL_SIZES=128"): assert k in env
  assert 'FUSEZAP_LICENSE_FILE="$PWD/.scratch/licence/chip-runs.lic"' in src and ".lic" not in src.replace("chip-runs.lic", "")
  assert "hf_" not in src.replace("hf_hub_download", "")

# ---------------------------------------------------------------- expert parallelism (EP=2) on NKISHIM, two processes

@pytest.fixture(scope="module")
def group():
  from fusezap.serve import dist
  old = {k: os.environ.pop(k, None) for k in ("FZ_NEURON_TP_AR", "FZ_NEURON_TP_HEAD", "FZ_NEURON_DTYPE")}
  g = dist.launch(2, "NKISHIM", timeout=120)
  yield g
  g.close()
  for k, v in old.items():
    if v is not None: os.environ[k] = v

def _ep_pair(g, path: str, dtype: str = "float32"):
  from fusezap.serve.models import load_model
  ref = load_model(path, dtype=getattr(dtypes, dtype), device="NKISHIM", max_len=128, prefill_chunk=32)
  g.post("load", {"model": path, "dtype": dtype, "max_len": 128, "prefill_chunk": 32})
  m = load_model(path, dtype=getattr(dtypes, dtype), device=g.device, max_len=128, prefill_chunk=32, tp=g)
  g.barrier("load")
  return ref, m

def _logits(m, ids):
  m.reset()
  out = [m.prefill([ids])]
  for t in np.random.default_rng(4).integers(3, 512, 6): out.append(m.decode([[int(t)]]))
  return out

def test_ep2_matches_one_core(group, tmp_path):
  """Expert parallelism over two ranks (NKISHIM, rank 1 a worker process): each rank holds 8 of 16 experts and the whole
  attention (16 / 16 heads: the q/k norm over all heads stays on one core), router and lm_head; only the expert MLP's down
  product is all-reduced. Llama.prefill / decode logits == one core's within float32 rounding (down_proj's K sum is grouped
  per rank, then rank 0 + rank 1), the same argmax, and the same greedy tokens."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  p = _olmoe_dir(tmp_path / "ep", seed=4)
  ref, m = _ep_pair(group, p)
  assert m.ep and m.ep_lo == 0 and (m.cfg.n_heads, m.cfg.n_kv_heads, m.cfg.intermediate_size) == (4, 4, 8 * 32)
  assert tuple(m.layers[0].wgu.shape) == (128, 2 * 8 * 32) and tuple(m.layers[0].wd.shape) == (8 * 32, 128)   # K-major, 8 experts
  assert tuple(m.layers[0].wqkv.shape) == (128, 3 * 128) and tuple(m.layers[0].wr.shape) == (128, 16)        # whole attention, router
  ids = [int(i) for i in np.random.default_rng(3).integers(3, 512, 40)]
  for a, b in zip(_logits(ref, ids), _logits(m, ids)):
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4)
    assert (a.argmax(-1) == b.argmax(-1)).all()
  caps = [jf.captured for jf in m._neuron_steps.jits.values() if jf.captured is not None]
  assert caps
  for cap in caps:   # one all-reduce per layer (the expert MLP's), none after o_proj
    names = []
    for it in cap.items:
      sp = getattr(it.si.ast[0], "spec", None) if it.si.ast else None
      if sp is not None: names += [q[0] for q in sp.parts] if hasattr(sp, "parts") else [sp.name]
    assert sum(n.startswith("fz_all_reduce") for n in names) == m.cfg.n_layers, names
  ra = list(ref.generate(ids[:20], 16, stop=[]))
  assert list(m.generate(ids[:20], 16, stop=[])) == ra and len(set(ra)) >= 3

@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
def test_ep2_engine_tokens_equal_one_core(group, tmp_path, dt, monkeypatch):
  """LLMEngine over the EP=2 model (both ranks' NeuronKVCache hold all 16 kv heads) at OLMoE's head layout, 64 experts top 8:
  one core's greedy tokens, float32 and bf16."""
  pytest.importorskip("torch"); pytest.importorskip("transformers")
  from fusezap.serve.engine import LLMEngine, SamplingParams
  p = _olmoe_dir(tmp_path / f"e{dt}", seed=5, num_experts=64, num_experts_per_tok=8, intermediate_size=16, **HEADS)
  ref, m = _ep_pair(group, p, dt)
  kw = dict(num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=32, decode_beam=0)
  e1, e2 = LLMEngine(ref, **kw), LLMEngine(m, **kw)
  assert e2.cache.n_kv == 16 and tuple(m.layers[0].wd.shape) == (32 * 16, 2048)
  group.post("engine", e2.tp_spec(), e2.tp_signature())
  group.barrier("engine")
  sp = SamplingParams(max_tokens=10, ignore_eos=True)
  want = e1.generate(PROMPTS, sp)
  got = e2.generate(PROMPTS, sp)
  if dt == "float32": assert got == want
  else:   # bf16: the same bf16 operands; only the down sum's grouping differs, so near ties may flip late
    prefix = [next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a)) for a, b in zip(want, got)]
    assert sum(prefix) >= 0.8 * sum(len(t) for t in want), (prefix, got, want)
