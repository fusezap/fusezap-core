"""Host memory of a weight load stays bounded (I57 M6): the 26 Oct 2x H200 pod stalled loading Llama-3.3-70B TP=2 at
~60 GB per rank because every uploaded weight's LazyBuffer kept its host array (a slice copy, a fused q/k/v or
gate/up concatenation, or an mmap view) for the model's lifetime, so a rank's host memory grew with its share of the
checkpoint. tracemalloc sees numpy's and ctypes' allocations (the CPU device's buffers too), so host memory held after
the load is the traced total minus the device bytes; the peak during the load must stay a few layers above that.
"""
from __future__ import annotations
import gc, json, os, tracemalloc, types
import numpy as np
import pytest
from fusezap.dtype import dtypes
from tests.test_tp import write_safetensors

CFG = {"model_type": "llama", "hidden_size": 1024, "intermediate_size": 4096, "num_hidden_layers": 8, "num_attention_heads": 8,
       "num_key_value_heads": 2, "head_dim": 128, "vocab_size": 4096, "rms_norm_eps": 1e-5, "rope_theta": 10000.0,
       "tie_word_embeddings": False, "hidden_act": "silu", "max_position_embeddings": 512, "eos_token_id": 1}

@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> str:
  """A ~200 MB float16 Llama checkpoint (8 layers of 25 MB)."""
  d = str(tmp_path_factory.mktemp("big"))
  c, blk = CFG, (np.random.default_rng(0).standard_normal(1 << 16) * 0.02).astype(np.float16)
  D, I, H, KV, hd, V = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["vocab_size"]
  r = lambda *s: np.resize(blk, s)
  ts = {"model.embed_tokens.weight": r(V, D), "model.norm.weight": np.ones(D, np.float16), "lm_head.weight": r(V, D)}
  for i in range(c["num_hidden_layers"]):
    p = f"model.layers.{i}."
    ts.update({p + "input_layernorm.weight": np.ones(D, np.float16), p + "post_attention_layernorm.weight": np.ones(D, np.float16),
               p + "self_attn.q_proj.weight": r(H * hd, D), p + "self_attn.k_proj.weight": r(KV * hd, D), p + "self_attn.v_proj.weight": r(KV * hd, D),
               p + "self_attn.o_proj.weight": r(D, H * hd), p + "mlp.gate_proj.weight": r(I, D), p + "mlp.up_proj.weight": r(I, D),
               p + "mlp.down_proj.weight": r(D, I)})
  write_safetensors(os.path.join(d, "model.safetensors"), ts)
  with open(os.path.join(d, "config.json"), "w") as f: json.dump(c, f)
  return d

def _load(build) -> tuple[int, int, int]:
  """(host bytes held after the load, peak host bytes during it, device bytes: the CPU allocator's growth)."""
  from fusezap.device import Device
  al = Device["CPU"].allocator
  gc.collect(); al.free_cache(); h0 = al.held   # no cached blocks from an earlier model: every buffer is a new allocation
  tracemalloc.start()
  try:
    m = build(); gc.collect()
    cur, peak = tracemalloc.get_traced_memory()
  finally: tracemalloc.stop()
  dev = al.held - h0
  del m
  return cur - dev, peak - dev, dev

@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1)])
def test_checkpoint_load_host_memory_bounded(ckpt, world, rank):
  from fusezap.serve.models.llama import Llama, LlamaConfig
  from fusezap.serve.weights import load_config, load_safetensors, checkpoint_bytes
  tp = types.SimpleNamespace(rank=rank, world=world) if world > 1 else None
  held, peak, dev = _load(lambda: Llama(LlamaConfig.from_hf(load_config(ckpt)), load_safetensors(ckpt), dtypes.float16, "CPU",
                                        max_len=64, prefill_chunk=16, tp=tp))
  ck = checkpoint_bytes(ckpt)
  assert dev > 0.45 * ck / world   # the rank's share really went to the device
  assert held < 0.03 * ck, f"{held / 1e6:.0f} MB of host arrays held after the load (checkpoint {ck / 1e6:.0f} MB)"
  assert peak < 0.2 * ck, f"host peak {peak / 1e6:.0f} MB during the load (checkpoint {ck / 1e6:.0f} MB)"

def test_random_weights_host_memory_bounded():
  """The random-weight model (bench/tp_step.py --random): the host keeps one array per distinct shape, not per layer."""
  from fusezap.serve.models import random_weights as R
  sh = dict(R.PRESETS["tiny"], hidden_size=1024, intermediate_size=4096, n_heads=8, n_kv_heads=2, vocab_size=4096, n_layers=8)
  for tp in (None, types.SimpleNamespace(rank=1, world=2)):
    held, peak, dev = _load(lambda: R.build(sh, "CPU", dtypes.float16, 64, 16, tp=tp))
    per_layer = dev / 8
    assert held < 3 * per_layer + 2 * 4096 * 1024 * 2, f"{held / 1e6:.0f} MB held, device {dev / 1e6:.0f} MB"
    assert peak < held + 3 * per_layer
