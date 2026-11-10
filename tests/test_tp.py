"""Tensor parallelism (I57 M1, fusezap/serve/dist.py): Megatron sharding of the Llama decoder over two processes.

  1. the sharded linear algebra (dist.tp_slice) equals the unsharded layer (numpy, tiny random weights, GQA + biases)
  2. the host all-reduce over two real processes (f32 / f16 / bf16: the rank-order float32 sum, identical on both ranks)
  3. a tiny random Qwen2-style model (GQA 4/2 heads, q/k/v biases) TP=2 (rank 0 here, rank 1 a worker process, host
     backend) gives TP=1's logits within float32 tolerance: Llama.prefill / decode, and LLMEngine's batched tokens
  4. (slow) Qwen2.5-0.5B-Instruct float32: TP=2 greedy 32 tokens == TP=1, engine and model paths
  5. (slow) `fusezap serve --tp 2`: the OpenAI API answers, streams, a cancelled stream stops, a killed worker stops the
     server instead of hanging it
The device is METAL when available, else CPU (FZ_DEVICE overrides; CUDA runs the NCCL backend: scripts/tp_m1_cuda.sh).
"""
from __future__ import annotations
import json, os, socket, struct, subprocess, sys, time
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.serve import dist
from fusezap.serve.models.llama import LlamaConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def _device() -> str:
  from fusezap.device import Device
  return os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")

def write_safetensors(fn: str, ts: dict[str, np.ndarray]) -> None:
  """A minimal safetensors writer (float32 / float16 only)."""
  hdr, off, blobs = {}, 0, []
  for k, a in ts.items():
    a = np.ascontiguousarray(a)
    b = a.tobytes()
    hdr[k] = {"dtype": {np.float32: "F32", np.float16: "F16"}[a.dtype.type], "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
    off += len(b); blobs.append(b)
  h = json.dumps(hdr).encode()
  h += b" " * (-len(h) % 8)
  with open(fn, "wb") as f: f.write(struct.pack("<Q", len(h)) + h + b"".join(blobs))

TINY = {"model_type": "qwen2", "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 200, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "tie_word_embeddings": False, "hidden_act": "silu", "max_position_embeddings": 512, "eos_token_id": 199}

def tiny_model_dir(path: str, seed: int = 0, cfg: dict | None = None, bias: bool = True) -> str:
  """A random Qwen2-architecture checkpoint (config.json + model.safetensors) with q/k/v biases and an o_proj bias."""
  c = {**TINY, **(cfg or {})}
  rng = np.random.default_rng(seed)
  D, I, H, KV, hd = c["hidden_size"], c["intermediate_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
  r = lambda *s, sc=0.15: (rng.standard_normal(s) * sc).astype(np.float32)
  ts = {"model.embed_tokens.weight": r(c["vocab_size"], D, sc=0.5), "model.norm.weight": 1 + r(D, sc=0.1), "lm_head.weight": r(c["vocab_size"], D)}
  for i in range(c["num_hidden_layers"]):
    p = f"model.layers.{i}."
    ts.update({p + "input_layernorm.weight": 1 + r(D, sc=0.1), p + "post_attention_layernorm.weight": 1 + r(D, sc=0.1),
               p + "self_attn.q_proj.weight": r(H * hd, D), p + "self_attn.k_proj.weight": r(KV * hd, D),
               p + "self_attn.v_proj.weight": r(KV * hd, D), p + "self_attn.o_proj.weight": r(D, H * hd),
               p + "mlp.gate_proj.weight": r(I, D), p + "mlp.up_proj.weight": r(I, D), p + "mlp.down_proj.weight": r(D, I)})
    if bias:
      ts.update({p + "self_attn.q_proj.bias": r(H * hd, sc=0.3), p + "self_attn.k_proj.bias": r(KV * hd, sc=0.3),
                 p + "self_attn.v_proj.bias": r(KV * hd, sc=0.3), p + "self_attn.o_proj.bias": r(D, sc=0.3)})
  os.makedirs(path, exist_ok=True)
  write_safetensors(os.path.join(path, "model.safetensors"), ts)
  with open(os.path.join(path, "config.json"), "w") as f: json.dump(c, f)
  return path

# ---------------------------------------------------------------- 1. sharded math (numpy)

def _rms(x, w, eps=1e-6): return w * x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)

def _layer(ws: dict, p: str, x: np.ndarray, H: int, KV: int, hd: int) -> np.ndarray:
  """One decoder layer's attention + MLP *deltas* without RoPE (enough to check the head grouping): x (T, D) -> the
  o_proj output and the down_proj output, before any bias of those two (row-parallel partial sums add up to these)."""
  g = lambda n: ws.get(p + n)
  q = x @ g("self_attn.q_proj.weight").T + g("self_attn.q_proj.bias")
  k = x @ g("self_attn.k_proj.weight").T + g("self_attn.k_proj.bias")
  v = x @ g("self_attn.v_proj.weight").T + g("self_attn.v_proj.bias")
  T, G = x.shape[0], H // KV
  q, k, v = q.reshape(T, H, hd), k.reshape(T, KV, hd), v.reshape(T, KV, hd)
  o = np.zeros((T, H, hd), np.float64)
  for h in range(H):
    s = q[:, h] @ k[:, h // G].T / np.sqrt(hd)
    s = np.where(np.tril(np.ones((T, T), bool)), s, -np.inf)
    a = np.exp(s - s.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
    o[:, h] = a @ v[:, h // G]
  ao = o.reshape(T, H * hd) @ g("self_attn.o_proj.weight").T
  gt, up = x @ g("mlp.gate_proj.weight").T, x @ g("mlp.up_proj.weight").T
  md = (gt / (1 + np.exp(-gt)) * up) @ g("mlp.down_proj.weight").T
  return ao, md

@pytest.mark.parametrize("world,H,KV", [(2, 4, 2), (2, 8, 2), (4, 8, 4), (2, 14, 2)])
def test_sharded_layer_math_equals_unsharded(tmp_path, world, H, KV):
  """Each rank's q/k/v (+ biases) rows and gate/up rows, o/down columns (tp_slice): the per-rank attention over its own
  head group and its partial o_proj / down_proj products sum to the unsharded layer's (GQA groups stay whole)."""
  from fusezap.serve.weights import load_safetensors
  d = tiny_model_dir(str(tmp_path / "m"), seed=world + H, cfg={"num_attention_heads": H, "num_key_value_heads": KV, "head_dim": 8,
                                                               "intermediate_size": 64})
  full = {k: np.asarray(v, np.float64) for k, v in load_safetensors(d).items()}
  cfg = LlamaConfig.from_hf(json.load(open(os.path.join(d, "config.json"))))
  x = np.random.default_rng(1).standard_normal((5, cfg.hidden_size))
  p = "model.layers.0."
  ref_a, ref_m = _layer(full, p, x, H, KV, 8)
  lc = dist.local_cfg(cfg, world)
  assert (lc.n_heads, lc.n_kv_heads, lc.intermediate_size) == (H // world, KV // world, 64 // world)
  sa, sm = 0, 0
  for r in range(world):
    ws = {k: dist.tp_slice(k, v, cfg, r, world) for k, v in full.items()}
    assert ws[p + "self_attn.o_proj.bias"].shape == (cfg.hidden_size,)   # row-parallel bias: whole, added once
    a, m = _layer(ws, p, x, H // world, KV // world, 8)
    sa, sm = sa + a, sm + m
  np.testing.assert_allclose(sa, ref_a, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(sm, ref_m, rtol=1e-10, atol=1e-10)

def test_check_tp_errors():
  c = LlamaConfig.from_hf({**TINY, "num_attention_heads": 6, "num_key_value_heads": 3, "intermediate_size": 96})
  with pytest.raises(ValueError, match="num_key_value_heads 3"): dist.check_tp(c, 2)
  with pytest.raises(ValueError, match="num_attention_heads 6"): dist.check_tp(c, 4)
  dist.check_tp(c, 3)
  assert dist.rank_device("CUDA:2", 1) == "CUDA:3" and dist.rank_device("CUDA", 1) == "CUDA:1" and dist.rank_device("CUDA", 0) == "CUDA"
  assert dist.rank_device("METAL", 1) == "METAL"

# ---------------------------------------------------------------- 2.-3. two processes (host backend)

@pytest.fixture(scope="module")
def group():
  dead: list = []
  g = dist.launch(2, _device(), on_dead=lambda *a: dead.append(a))   # NCCL on CUDA (2 GPUs), the host backend elsewhere
  yield g
  dist.dbg(f"test_tp: module fixture teardown: {g!r} close")   # FZ_TP_DEBUG=1: each shutdown step, timestamped (stderr)
  t0 = time.monotonic()
  g.close()
  dt = time.monotonic() - t0
  dist.dbg(f"test_tp: module fixture teardown done in {dt:.2f} s; worker exit codes {[p.poll() for p in g.procs.values()]}")
  print(f"TP_TEARDOWN {json.dumps({'seconds': round(dt, 3), 'backend': g.backend, 'codes': [p.poll() for p in g.procs.values()]})}")
  assert not dead, f"a worker died: {dead}"
  assert dt < 3 * dist.CLOSE_TIMEOUT + 10, f"TPGroup.close took {dt:.1f} s"

def _stall(g, seconds: float, marker: str) -> None:
  """Run on a worker (op "call"): block like a worker stuck in a collective or a long kernel (marker: written first)."""
  open(marker, "w").close()
  time.sleep(seconds)

def test_load_prefetches_ahead_releases_and_logs_progress(tmp_path, monkeypatch, capfd):
  """The weight load reads ahead (posix_fadvise WILLNEED on the file range each rank's slice reads, FZ_PREFETCH_LAYERS
  layers ahead, then lm_head), releases every range once uploaded (DONTNEED, FZ_RELEASE_LAG layers behind), logs its
  progress and host memory per rank (FZ_TP_DEBUG=1), and rank 0 waits for the workers' load for max(FZ_TP_TIMEOUT, a
  rank's checkpoint share at 100 MB/s)."""
  import fusezap.serve.weights as WT
  from fusezap.serve import cli
  from fusezap.serve.models import load_model
  d = tiny_model_dir(str(tmp_path / "tiny"), bias=False)
  st = WT.load_safetensors(d)
  cfg = LlamaConfig.from_hf(TINY)
  a = st["model.layers.1.self_attn.o_proj.weight"]
  fn, off, nb = a.st_span
  assert fn.endswith("model.safetensors") and nb == a.nbytes and dist.tp_slice("model.layers.1.self_attn.o_proj.weight", a, cfg, 1, 2).st_span == a.st_span
  with open(fn, "rb") as f: f.seek(off); assert f.read(nb) == a.tobytes()
  q = st["model.layers.0.self_attn.q_proj.weight"]   # a column-parallel slice reads its own rows only
  assert WT.span(dist.tp_slice("model.layers.0.self_attn.q_proj.weight", q, cfg, 1, 2))[2:] == (q.st_span[1] + q.nbytes // 2, q.nbytes // 2)
  calls: list = []
  monkeypatch.setattr(WT.os, "posix_fadvise", lambda fd, o, n, adv: calls.append((o, n, adv)), raising=False)
  monkeypatch.setattr(WT.os, "POSIX_FADV_WILLNEED", 3, raising=False)
  monkeypatch.setattr(WT.os, "POSIX_FADV_DONTNEED", 4, raising=False)
  monkeypatch.setenv("FZ_TP_DEBUG", "1")
  class G: rank, world = 0, 2
  m = load_model(d, dtype=dtypes.float32, device=_device(), max_len=64, prefill_chunk=16, tp=G())
  spans = {WT.span(dist.tp_slice(k, v, cfg, 0, 2))[2:] for k, v in st.items() if k.startswith("model.layers.")}
  head = st["lm_head.weight"].st_span[1:]
  will, dont = [c[:2] for c in calls if c[2] == 3], [c[:2] for c in calls if c[2] == 4]
  assert set(will[:-1]) == spans and will[-1] == head and len(will) == len(spans) + 1 and not WT._FDS
  assert set(dont[:-1]) == spans and dont[-1] == head and len(dont) == len(spans) + 1
  err = capfd.readouterr().err
  assert "load: 2 layers and lm_head" in err and "load: start" in err and "host: " in err and m.cfg.n_heads == 2
  monkeypatch.setenv("FZ_PREFETCH", "0"); calls.clear()
  load_model(d, dtype=dtypes.float32, device=_device(), max_len=64, prefill_chunk=16)
  assert calls and all(c[2] == 4 for c in calls)
  monkeypatch.setenv("FZ_RELEASE", "0"); calls.clear()
  load_model(d, dtype=dtypes.float32, device=_device(), max_len=64, prefill_chunk=16)
  assert not calls
  assert cli.load_timeout(d, 2) == dist.TIMEOUT and WT.checkpoint_bytes(d) == os.path.getsize(os.path.join(d, "model.safetensors"))
  monkeypatch.setattr(WT, "checkpoint_bytes", lambda p: 132 * 10 ** 9)
  assert cli.load_timeout(d, 2) == pytest.approx(660.0)

def test_close_is_bounded_and_logged(tmp_path, monkeypatch, capfd):
  """A fresh group (host backend): one all-reduce, then close(): rank 0 sends quit, the worker closes its collective and
  exits 0 within a second or two; FZ_TP_DEBUG=1 logs every step of both ranks with a timestamp (stderr)."""
  monkeypatch.setenv("FZ_TP_DEBUG", "1")
  g = dist.launch(2, _device(), backend="host", on_dead=lambda *a: None)
  with g.spmd("call", "fusezap.serve.dist:selftest_allreduce", "float32", 64):
    assert dist.selftest_allreduce(g, "float32", 64)["ok"]
  t0 = time.monotonic()
  g.close()
  dt = time.monotonic() - t0
  assert [p.poll() for p in g.procs.values()] == [0], "the worker did not exit cleanly on quit"
  assert dt < 5, f"close took {dt:.1f} s"
  g.close()   # idempotent (atexit calls it again)
  err = capfd.readouterr().err
  for want in ("fusezap tp rank 0", "post quit: done", "worker rank 1 exit code 0", "close done", "fusezap tp rank 1", "exit 0"):
    assert want in err, f"{want!r} not in the debug log:\n{err}"
  import re
  assert re.search(r"^\[\d\d:\d\d:\d\d\.\d{3}\] fusezap tp rank 1 pid \d+: quit$", err, re.M), err

def test_close_kills_a_stuck_worker(tmp_path):
  """A worker that never reads "quit" (stuck in a long call): close(timeout) waits `timeout`, kills it, and returns in
  about that time instead of hanging."""
  g = dist.launch(2, _device(), backend="host", on_dead=lambda *a: None)
  marker = tmp_path / "stalled"
  g.post("call", "tests.test_tp:_stall", 120, str(marker))
  t0 = time.monotonic()
  while not marker.exists():
    assert time.monotonic() - t0 < 60 and all(p.poll() is None for p in g.procs.values()), "the worker did not start the call"
    time.sleep(0.05)
  t0 = time.monotonic()
  g.close(timeout=1.0)
  dt = time.monotonic() - t0
  (p,) = g.procs.values()
  assert p.poll() is not None and p.poll() != 0, "the stuck worker was not killed"
  assert dt < 8, f"close took {dt:.1f} s"

@pytest.mark.parametrize("dt", ["float32", "float16", "bfloat16"])
def test_host_all_reduce_two_processes(group, dt):
  """dist.selftest_allreduce on both ranks: the device all-reduce item (AllReduceCall through fz.jit.run_item) of each
  rank's tensor; rank 0 checks the sum against the rank-order float32 sum and both ranks' bits (allgathered)."""
  with group.spmd("call", "fusezap.serve.dist:selftest_allreduce", dt, 1000):
    res = dist.selftest_allreduce(group, dt, 1000)
  assert res["ok"], res

def _tiny_pair(group, tmp_path, max_len=128):
  from fusezap.serve.models import load_model
  d = tiny_model_dir(str(tmp_path / "tiny"))
  ref = load_model(d, dtype=dtypes.float32, device=_device(), max_len=max_len, prefill_chunk=32)
  group.post("load", {"model": d, "dtype": "float32", "max_len": max_len, "prefill_chunk": 32})
  m = load_model(d, dtype=dtypes.float32, device=group.device, max_len=max_len, prefill_chunk=32, tp=group)
  group.barrier("load")
  return ref, m

def test_tiny_model_tp2_logits_equal_tp1(group, tmp_path):
  """Rank 0's local model holds half the heads (2 q / 1 kv) and half the MLP; with rank 1's half and the all-reduce,
  prefill (two chunks: a bucket and its continuation) and 8 teacher-forced decode steps give TP=1's logits."""
  ref, m = _tiny_pair(group, tmp_path)
  assert (m.cfg.n_heads, m.cfg.n_kv_heads, m.cfg.intermediate_size, m.full_cfg.n_heads) == (2, 1, 48, 4)
  assert m.layers[0].wqkv.shape == (4 * 16, 64) and m.layers[0].wo.shape == (64, 32) and m.layers[0].wd.shape == (64, 48)
  ids = list(np.random.default_rng(3).integers(0, 199, 45))
  a, b = ref.prefill([ids]), m.prefill([ids])
  np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4)
  assert np.abs(a).max() > 0.5
  for t in np.random.default_rng(4).integers(0, 199, 8):
    a, b = ref.decode([[int(t)]]), m.decode([[int(t)]])
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4)
  ref.reset(); m.reset()
  ra = list(ref.generate(ids[:20], 16, stop=[]))
  assert list(m.generate(ids[:20], 16, stop=[])) == ra and len(set(ra)) > 3

def test_tiny_engine_tp2_equals_tp1(group, tmp_path):
  """LLMEngine (paged cache, continuous batching) over the TP=2 model: rank 1 runs each forward from rank 0's host
  arrays; five prompts batched with a few prefill chunks give TP=1's greedy tokens, and the worker's engine agrees."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  ref, m = _tiny_pair(group, tmp_path / "e")
  kw = dict(num_blocks=64, block_size=16, max_num_seqs=8, max_prefill_tokens=64, decode_beam=0)
  e1 = LLMEngine(ref, **kw)
  e2 = LLMEngine(m, **kw)
  group.post("engine", e2.tp_spec(), e2.tp_signature())
  group.barrier("engine")
  rng = np.random.default_rng(5)
  ps = [[int(t) for t in rng.integers(0, 199, n)] for n in (5, 33, 70, 12, 90)]
  sp = SamplingParams(max_tokens=12, ignore_eos=True)
  want = e1.generate(ps, sp)
  assert e2.generate(ps, sp) == want and len({t for o in want for t in o}) > 5
  assert e2._nfwd == e1._nfwd > 10

def test_tiny_engine_fused_add_norm_path(tmp_path, monkeypatch):
  """The fused all-reduce + residual add + RMSNorm path of Llama.forward_tokens (I57 M3; on CUDA one custom kernel),
  emulated on this device with the same composed ops (FZ_TP_AR_FUSE_EMULATE=1, on both ranks): the decode rows'
  o_proj / down_proj products are reduced inside the next norm (the last layer's down_proj plainly), prefill rows (over
  FZ_TP_AR_FUSE_ROWS) unfused; TP=1's greedy tokens, and per-rank FZ_TP_PROFILE lines for every forward."""
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve import tpprof
  from fusezap.serve.models import load_model
  monkeypatch.setenv("FZ_TP_AR_FUSE_EMULATE", "1")
  monkeypatch.setenv("FZ_TP_AR_FUSE_ROWS", "8")
  monkeypatch.setenv("FZ_TP_PROFILE", str(tmp_path / "prof"))
  g = dist.launch(2, _device(), backend="host", on_dead=lambda *a: None)
  try:
    d = tiny_model_dir(str(tmp_path / "tiny"), bias=False)
    ref = load_model(d, dtype=dtypes.float32, device=_device(), max_len=128, prefill_chunk=32)
    g.post("load", {"model": d, "dtype": "float32", "max_len": 128, "prefill_chunk": 32})
    m = load_model(d, dtype=dtypes.float32, device=g.device, max_len=128, prefill_chunk=32, tp=g)
    g.barrier("load")
    kw = dict(num_blocks=64, block_size=16, max_num_seqs=8, max_prefill_tokens=64, decode_beam=0)
    e1, e2 = LLMEngine(ref, **kw), LLMEngine(m, **kw)
    g.post("engine", e2.tp_spec(), e2.tp_signature())
    g.barrier("engine")
    calls: list = []
    real = dist.all_reduce_add_norm
    monkeypatch.setattr(dist, "all_reduce_add_norm", lambda *a: calls.append(a[0].shape) or real(*a))
    rng = np.random.default_rng(6)
    ps = [[int(t) for t in rng.integers(0, 199, n)] for n in (5, 33, 12)]
    sp = SamplingParams(max_tokens=10, ignore_eos=True)
    assert e2.generate(ps, sp) == e1.generate(ps, sp)
    # decode rows only, 2 per layer but the last: the decode buckets (max_num_seqs 8 on METAL / CPU; on CUDA 1, 2, 4, 8, so
    # the 3 prompts decode at 4 rows: the first M3 run failed here on `== 8`, not on the custom all-reduce, which this
    # emulated host-backend path never uses)
    assert calls and all(s[0] in e2.decode_sizes and s[0] <= 8 for s in calls), (calls, e2.decode_sizes)
    g.barrier("done")
    e2._tpprof.flush()
  finally: g.close()
  rows = tpprof.summarize(str(tmp_path / "prof"))
  dec = [r for r in rows if r["kind"] == "d"]
  assert dec and dec[0]["n"] > 5 and dec[0]["chan_us"][1] is not None and dec[0]["chan_us"][1] >= 0, rows
  assert {r["kind"] for r in rows} >= {"d", "p"}

# ---------------------------------------------------------------- 4. Qwen2.5-0.5B (slow)

def _model_path() -> str | None:
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(MODEL, allow_download=False)
  except Exception: return None

@pytest.mark.slow
@pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")
def test_qwen05b_tp2_greedy_equals_tp1(group):
  """Qwen2.5-0.5B-Instruct float32 (14 q / 2 kv heads: 7 / 1 per rank, the q/k/v biases sliced): the TP=2 engine's greedy
  32 tokens for three chat prompts == TP=1's, and the model path's prefill logits agree."""
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.tokenizer import Tokenizer
  tok = Tokenizer.from_pretrained(_model_path())
  prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True)
             for q in ("Write a short poem about the sea.", "What is 17 * 23? Show your steps.", "List three uses of copper.")]
  ref = load_model(MODEL, dtype=dtypes.float32, device=_device(), max_len=256, prefill_chunk=128)
  group.post("load", {"model": MODEL, "dtype": "float32", "max_len": 256, "prefill_chunk": 128})
  m = load_model(MODEL, dtype=dtypes.float32, device=group.device, max_len=256, prefill_chunk=128, tp=group)
  group.barrier("load", 900)
  a, b = ref.prefill([prompts[0]]), m.prefill([prompts[0]])
  np.testing.assert_allclose(b, a, rtol=1e-3, atol=1e-3)
  assert int(np.argmax(a)) == int(np.argmax(b))
  kw = dict(num_blocks=128, block_size=16, max_num_seqs=4, max_prefill_tokens=128, decode_beam=0)
  e1, e2 = LLMEngine(ref, **kw), LLMEngine(m, **kw)
  group.post("engine", e2.tp_spec(), e2.tp_signature())
  group.barrier("engine", 900)
  sp = SamplingParams(max_tokens=32, ignore_eos=True)
  want = e1.generate(prompts, sp)
  got = e2.generate(prompts, sp)
  print("TP=1", want, "\nTP=2", got)
  assert got == want

# ---------------------------------------------------------------- 5. fusezap serve --tp 2 (slow)

def _descendants(pid: int) -> list[tuple[int, str]]:
  rows = [l.split(None, 2) for l in subprocess.run(["ps", "-A", "-o", "pid=,ppid=,command="], capture_output=True, text=True).stdout.splitlines()]
  kids: dict[int, list[tuple[int, str]]] = {}
  for r in rows:
    if len(r) == 3: kids.setdefault(int(r[1]), []).append((int(r[0]), r[2]))
  out, todo = [], [pid]
  while todo:
    for c in kids.get(todo.pop(), []): out.append(c); todo.append(c[0])
  return out

def _free_port() -> int:
  with socket.socket() as s: s.bind(("127.0.0.1", 0)); return s.getsockname()[1]

@pytest.mark.slow
@pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")
def test_serve_tp2_openai_stream_cancel_and_dead_worker():
  import httpx
  from openai import OpenAI
  port = _free_port()
  cmd = [sys.executable, "-m", "fusezap", "serve", MODEL, "--device", _device(), "--tp", "2", "--port", str(port), "--max-model-len", "512",
         "--beam", "0", "--log-level", "warning"]
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
  try:
    t0 = time.time()
    while True:
      if proc.poll() is not None: raise RuntimeError(f"server exited with {proc.returncode}:\n{proc.stdout.read()}")
      try:
        if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200: break
      except httpx.HTTPError: pass
      assert time.time() - t0 < 900, "server did not come up"
      time.sleep(0.5)
    c = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")
    msgs = [{"role": "user", "content": "Name three primary colors."}]
    chunks = list(c.chat.completions.create(model=MODEL, messages=msgs, max_tokens=32, temperature=0, stream=True))
    text = "".join(ch.choices[0].delta.content or "" for ch in chunks if ch.choices)
    assert "red" in text.lower() and "blue" in text.lower(), text
    full = c.chat.completions.create(model=MODEL, messages=msgs, max_tokens=32, temperature=0)
    assert full.choices[0].message.content == text
    # logprobs (the scorer runs mirrored on rank 1)
    lp = c.completions.create(model=MODEL, prompt="The capital of France is", max_tokens=1, temperature=0, logprobs=2, echo=True)
    assert lp.choices[0].logprobs.top_logprobs and len(lp.choices[0].logprobs.token_logprobs) > 3
    # cancel: read a few chunks of a long stream, then close it; the server stops the request
    s = c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Count from 1 to 400."}], max_tokens=400,
                                  temperature=0, stream=True)
    for i, _ in enumerate(s):
      if i >= 3: break
    s.close()
    t0 = time.time()
    while time.time() - t0 < 30:
      m = httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=5).text
      if "vllm:num_requests_running 0" in m.replace("{", " ").replace("}", "") or any(
          l.startswith("vllm:num_requests_running") and l.rstrip().endswith(" 0") for l in m.splitlines()): break
      time.sleep(0.5)
    else: raise AssertionError(f"the cancelled request is still running:\n{m}")
    again = c.chat.completions.create(model=MODEL, messages=msgs, max_tokens=32, temperature=0)
    assert again.choices[0].message.content == text
    # a dead worker: the server exits (non-zero) instead of hanging
    kids = [p for p, cmd in _descendants(proc.pid) if "fusezap.serve.dist" in cmd]   # CUDA: a child of the engine process
    assert len(kids) == 1, f"expected one worker process under {proc.pid}, found {kids}"
    os.kill(kids[0], 9)
    try: rc = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
      proc.kill()
      raise AssertionError(f"the server did not stop after its worker {kids} died:\n{proc.stdout.read()}")
    assert rc != 0
    assert "tensor-parallel worker rank 1 exited" in proc.stdout.read()
  finally:
    if proc.poll() is None:
      proc.terminate()
      try: proc.wait(timeout=30)
      except subprocess.TimeoutExpired: proc.kill()

@pytest.mark.slow
@pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")
@pytest.mark.parametrize("engine_process,sig", [("on", "SIGTERM"), ("off", "SIGINT")])
def test_serve_tp2_stops_quickly(engine_process, sig):
  """`fusezap serve --tp 2` stops within seconds on SIGTERM (to the server process) and on Ctrl-C (SIGINT to its process
  group), with the engine in its own process (as on CUDA) or in-process: the workers are gone too, none orphaned."""
  import contextlib, httpx, signal
  port = _free_port()
  cmd = [sys.executable, "-m", "fusezap", "serve", MODEL, "--device", _device(), "--tp", "2", "--port", str(port), "--max-model-len", "256",
         "--beam", "0", "--no-warmup", "--log-level", "warning", "--engine-process", engine_process]
  proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True,
                          env={**os.environ, "FZ_TP_DEBUG": "1"})
  try:
    t0 = time.time()
    while True:
      if proc.poll() is not None: raise RuntimeError(f"server exited with {proc.returncode}:\n{proc.stdout.read()}")
      try:
        if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200: break
      except httpx.HTTPError: pass
      assert time.time() - t0 < 900, "server did not come up"
      time.sleep(0.5)
    workers = [p for p, c in _descendants(proc.pid) if "fusezap.serve.dist" in c]
    assert len(workers) == 1, workers
    t0 = time.monotonic()
    if sig == "SIGTERM": proc.send_signal(signal.SIGTERM)
    else: os.killpg(proc.pid, signal.SIGINT)
    try: proc.wait(timeout=30)
    except subprocess.TimeoutExpired: raise AssertionError(f"the server did not stop within 30 s of {sig}") from None
    dt = time.monotonic() - t0
    out = proc.stdout.read()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
      try: os.kill(workers[0], 0)
      except ProcessLookupError: break
      time.sleep(0.1)
    else: raise AssertionError(f"worker {workers[0]} still running after the server stopped:\n{out}")
    print(f"TP_SERVE_STOP {json.dumps({'engine_process': engine_process, 'signal': sig, 'seconds': round(dt, 2)})}")
    assert dt < 10, f"{sig}: the server took {dt:.1f} s to stop:\n{out}"
    assert "worker rank 1 exit code 0" in out and "close done" in out, out
  finally:
    if proc.poll() is None:
      with contextlib.suppress(ProcessLookupError): os.killpg(proc.pid, 9)

# ---------------------------------------------------------------- CUDA only (scripts/tp_m1_cuda.sh: 2 GPUs, NCCL)

def _cuda2() -> bool:
  if not _device().upper().startswith("CUDA"): return False
  from fusezap.device import Device
  return Device.available("CUDA:1")
needs_cuda2 = pytest.mark.skipif(not _cuda2(), reason="needs FZ_DEVICE=CUDA and two GPUs")

@needs_cuda2
def test_nccl_all_reduce_bandwidth(group):
  """NCCL all-reduce inside a replayed fz.jit step (a CUDA graph with the NCCL kernels captured): prints the time per
  all-reduce for a decode-sized (32 x 8192 bf16) and a prefill-sized (2048 x 8192) message."""
  assert group.backend == "nccl"
  print("NCCL", dist.nccl_version(), "NCCL_P2P_DISABLE=" + os.environ.get("NCCL_P2P_DISABLE", ""))
  for n in (32 * 8192, 2048 * 8192):
    with group.spmd("call", "fusezap.serve.dist:bench_allreduce", "bfloat16", n, 20):
      r = dist.bench_allreduce(group, "bfloat16", n, 20)
    print("TP_BENCH", json.dumps(r))
    assert r["us"] > 0

@needs_cuda2
@pytest.mark.slow
@pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")
def test_qwen05b_bf16_tp2_lookahead_and_vs_tp1(group):
  """CUDA bf16 (the serving path: tensor-core GEMMs, fused kernels, CUDA graphs, device inputs): the TP=2 engine with
  lookahead (the worker feeds its own device argmax into the next decode) gives the same tokens as without lookahead,
  so the worker's logits are rank 0's bit for bit. TP=2 vs TP=1 in bf16 differ by the partial sums' rounding: the
  prefill top-1 must agree; the greedy prefix lengths are printed (TP_MATCH)."""
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve.tokenizer import Tokenizer
  tok = Tokenizer.from_pretrained(_model_path())
  prompts = [tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True)
             for q in ("Write a short poem about the sea.", "What is 17 * 23? Show your steps.", "List three uses of copper.",
                       "Explain what a hash table is in two sentences.")]
  ref = load_model(MODEL, dtype=dtypes.bfloat16, device=_device(), max_len=512, prefill_chunk=128)
  group.post("load", {"model": MODEL, "dtype": "bfloat16", "max_len": 512, "prefill_chunk": 128})
  m = load_model(MODEL, dtype=dtypes.bfloat16, device=group.device, max_len=512, prefill_chunk=128, tp=group)
  group.barrier("load", 900)
  kw = dict(num_blocks=256, block_size=16, max_num_seqs=8, max_prefill_tokens=256)
  e1, e2 = LLMEngine(ref, **kw), LLMEngine(m, **kw)
  group.post("engine", e2.tp_spec(), e2.tp_signature())
  group.barrier("engine", 900)
  assert e2.lookahead and e2.dsteps is not None
  sp = SamplingParams(max_tokens=48, ignore_eos=True)
  a = e2.generate(prompts, sp)
  assert e2.stats["lookahead_steps"] > 10
  e2.lookahead = False
  b = e2.generate(prompts, sp)
  assert a == b, "lookahead changed a TP=2 token: the worker's argmax differs from rank 0's"
  want = e1.generate(prompts, sp)
  match = [next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), len(p)) for p, q in zip(a, want)]
  print("TP_MATCH", json.dumps({"greedy_prefix_match": match, "of": 48}))
  for p in prompts:
    ref.reset(); m.reset()
    assert int(np.argmax(ref.prefill([p]))) == int(np.argmax(m.prefill([p])))
