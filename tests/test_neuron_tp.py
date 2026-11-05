"""I72: tensor parallelism on NEURON (fusezap/serve/neuron_tp.py), checked on the Mac (NKISHIM, two real processes):

  1. AllReduceSpec (nki.collectives.all_reduce) in each form renders and runs through the shim's collective; a fused
     kernel with it keeps the collectives import
  2. the runtime workers' shared-memory all-reduce (nrt_worker.ShmAR) over two processes: the rank-order float32 sum, the
     same bits on both ranks, many calls of mixed sizes in a row (the two-slot handshake); the worker's "ar" request and a
     collective NEFF's load (nrtpy cc_enabled / rank_id / world_size) against the fake nrtpy; NeuronAllocator queues "ar"
  3. TP=2 on NKISHIM (rank 0 here, rank 1 a worker process): the launch picks the NKI collective all-reduce (the shim's
     collective is the group's host sum); a tiny Qwen2-style model's logits match TP=1 within float32 tolerance and its
     greedy tokens are identical, through Llama.prefill / decode and through LLMEngine, with the all-reduce recorded as an
     AllReduceSpec launch in the captured step; bf16 at Qwen2.5-7B's head layout (28 / 4 heads, head_dim 128) gives TP=1
     bf16's tokens; FZ_NEURON_TP_AR=host and FZ_NEURON_TP_HEAD=vocab (vocab-parallel lm_head, logits all-gathered) give
     the collective path's logits bit for bit.
"""
from __future__ import annotations
import dataclasses, multiprocessing as mp, os, sys, time
import numpy as np
import pytest
from fusezap.dtype import dtypes
from fusezap.codegen import nki_kernels as nk
from fusezap.device import nkishim, nrt_worker
from fusezap.serve import dist, neuron_tp as NT

TIER = "NKISHIM"

# ---------------------------------------------------------------- 1. the collective kernel on the shim

@pytest.mark.parametrize("kind", nk.AR_KINDS)
@pytest.mark.parametrize("n", [3584, 4 * 3584, 100, 128 * 4100])
def test_all_reduce_spec_runs_on_the_shim(kind, n, monkeypatch):
  monkeypatch.setattr(nkishim, "COLLECTIVE", [lambda a, op, g: a * np.float32(3.0)])   # three identical ranks
  sp = nk.AllReduceSpec(n, 3, kind)
  prog = sp.render()
  assert "nki.collectives.all_reduce" in prog.src and prog.src.startswith(nk.CC_HEADER) and prog.n_bufs == 2 and prog.outs == (0,)
  assert "ReplicaGroup([[0, 1, 2]])" in prog.src and not sp.fusable
  x = np.random.default_rng(n).standard_normal(n).astype(np.float32)
  out = np.zeros(n, np.float32)
  nkishim.execute(prog, [out, x])
  np.testing.assert_array_equal(out, x * np.float32(3.0))
  monkeypatch.setattr(nkishim, "COLLECTIVE", [])
  with pytest.raises(RuntimeError, match="tensor-parallel group"): nkishim.execute(prog, [out, x])

def test_all_reduce_kind_env_and_fusable(monkeypatch):
  monkeypatch.setenv("FZ_NEURON_TP_AR_KIND", "sbuf")
  assert nk.ar_kind() == "sbuf"
  monkeypatch.setenv("FZ_NEURON_TP_AR_KIND", "nope")
  with pytest.raises(ValueError): nk.ar_kind()
  monkeypatch.setenv("FZ_NEURON_TP_AR_FUSE", "1")
  assert nk.AllReduceSpec(8, 2).fusable

def test_fused_kernel_with_a_collective_keeps_the_import():
  ar, norm = nk.AllReduceSpec(3584, 2).render(), nk.NormSpec(1, 3584, 1e-6, False).render()
  fused = nk.FusedSpec(((norm.name, norm.src, (2, 0, 3)), (ar.name, ar.src, (1, 2))), 4, (0, 1, 2), ())
  src = fused.render().src
  assert src.startswith(nk.CC_HEADER) and "nki.collectives.all_reduce" in src
  plain = nk.FusedSpec(((norm.name, norm.src, (2, 0, 3)),), 4, (0, 2), ()).render().src
  assert plain.startswith(nk.HEADER) and "collectives" not in plain   # every other fused kernel: unchanged source, same NEFF key

# ---------------------------------------------------------------- 2. the runtime workers' shm all-reduce

def _shm_rank(path: str, rank: int, world: int, sizes: list[int], q) -> None:
  ar = nrt_worker.ShmAR(path, rank, world, 1 << 16, timeout=30)
  q.put((rank, [ar.reduce((np.random.default_rng(1000 * i + rank).standard_normal(n) * (rank + 1)).astype(np.float32)) for i, n in enumerate(sizes)]))

def test_shm_all_reduce_two_processes():
  path = NT.shm_create(2, 1 << 16)
  sizes = [3584, 1, 7168, 3584, 16384, 5, 3584] * 6
  ctx = mp.get_context("spawn")
  q = ctx.Queue()
  ps = [ctx.Process(target=_shm_rank, args=(path, r, 2, sizes, q)) for r in range(2)]
  for p in ps: p.start()
  got = dict(q.get(timeout=60) for _ in ps)
  for p in ps: p.join(30)
  for i, n in enumerate(sizes):
    want = (np.random.default_rng(1000 * i).standard_normal(n)).astype(np.float32) + (np.random.default_rng(1000 * i + 1).standard_normal(n) * 2).astype(np.float32)
    np.testing.assert_array_equal(got[0][i], want)
    np.testing.assert_array_equal(got[1][i], got[0][i])

def test_shm_all_reduce_rejects_a_large_message_and_times_out():
  path = NT.shm_create(2, 64)
  ar = nrt_worker.ShmAR(path, 0, 2, 64, timeout=0.2)
  with pytest.raises(ValueError, match="slot"): ar.reduce(np.zeros(17, np.float32))
  with pytest.raises(TimeoutError, match="rank 1"): ar.reduce(np.zeros(4, np.float32))

@pytest.fixture
def nrt(monkeypatch):
  from test_neuron_nrt import fake_nrtpy, NrtpyModel
  m = fake_nrtpy()
  monkeypatch.setitem(sys.modules, "nrtpy", m)
  monkeypatch.setattr(nrt_worker, "S", nrt_worker.State())
  monkeypatch.setattr(nrt_worker, "_SHM", [])
  return m, NrtpyModel

def test_worker_ar_request_and_collective_load(nrt, monkeypatch, tmp_path):
  m, NrtpyModel = nrt
  monkeypatch.setenv("FZ_NEURON_TP_SHM", NT.shm_create(1, 1 << 16))
  monkeypatch.setenv("FZ_NEURON_CC_WORLD", "1")
  monkeypatch.setenv("FZ_NEURON_TP_SLOT", str(1 << 16))
  x = np.arange(10, dtype=np.float32)
  rep = nrt_worker.handle({"op": "batch", "new": {1: 64, 2: 48}, "free": [], "reqs": [
    {"op": "write", "id": 1, "data": np.concatenate([x, np.full(6, 7, np.float32)]).view(np.uint8)},
    {"op": "ar", "in": 1, "out": 2, "n": 40},   # world 1: out = in's first 40 bytes; the rest of out is kept
    {"op": "read", "id": 2}]})
  assert rep["ok"] and rep["seconds"] == [], rep
  np.testing.assert_array_equal(np.frombuffer(rep["data"], np.float32)[:10], x)
  seen = {}
  def load(cls, path, name=None, core_id=0, **kw):
    seen.update(kw)
    return cls({"tensors": ({"data0.must_alias_input": (4, "float32"), "data1": (4, "float32")}, {"data0": (4, "float32")},
                            {"data0": "data0.must_alias_input"})})
  monkeypatch.setattr(NrtpyModel, "load_from_neff", classmethod(load))
  assert nrt_worker.handle({"op": "load", "key": "k", "path": str(tmp_path / "x.neff"), "cc": {"rank": 1, "world": 2}})["ok"]
  assert seen == {"cc_enabled": True, "rank_id": 1, "world_size": 2}
  seen.clear()
  assert nrt_worker.handle({"op": "load", "key": "k2", "path": str(tmp_path / "y.neff")})["ok"] and seen == {}

def test_allocator_queues_ar_with_the_launches():
  from fusezap.device.neuron import NeuronAllocator
  sent = []
  class W:
    def call(self, req): sent.append(req); return {"ok": True, "seconds": [], "data": np.zeros(4, np.uint8)}
  a = NeuronAllocator(W())
  a.call({"op": "exec", "key": "k", "in": {}, "out": {}, "alias": {}, "iters": 1})
  a.call({"op": "ar", "in": 1, "out": 2, "n": 16})
  assert not sent and [r["op"] for r in a.pending] == ["exec", "ar"]
  a.call({"op": "read", "id": 2, "n": 4})
  assert len(sent) == 1 and [r["op"] for r in sent[0]["reqs"]] == ["exec", "ar", "read"]   # one round trip

def test_rank_envs_pick_cores_and_one_bootstrap(monkeypatch):
  monkeypatch.delenv("NEURON_RT_ROOT_COMM_ID", raising=False)
  monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "2-3")
  e = NT.rank_envs("NEURON", 2)
  assert [x["NEURON_RT_VISIBLE_CORES"] for x in e] == ["2", "3"] and [x["FZ_NEURON_CC_RANK"] for x in e] == ["0", "1"]
  assert e[0]["NEURON_RT_ROOT_COMM_ID"] == e[1]["NEURON_RT_ROOT_COMM_ID"] and e[0]["FZ_NEURON_TP_SHM"] == e[1]["FZ_NEURON_TP_SHM"]
  monkeypatch.delenv("NEURON_RT_VISIBLE_CORES")
  assert [x["NEURON_RT_VISIBLE_CORES"] for x in NT.rank_envs("NEURON", 2)] == ["0", "1"]
  assert "NEURON_RT_VISIBLE_CORES" not in NT.rank_envs("NKISHIM", 2)[0]

# ---------------------------------------------------------------- 3. TP=2 on NKISHIM

def _launch(monkeypatch, **env):
  for k, v in env.items(): monkeypatch.setenv(k, v)   # rank 0 reads them here; the worker inherits them
  return dist.launch(2, TIER, timeout=120)

@pytest.fixture(scope="module")
def group():
  old = {k: os.environ.pop(k, None) for k in ("FZ_NEURON_TP_AR", "FZ_NEURON_TP_HEAD", "FZ_NEURON_DTYPE")}
  g = dist.launch(2, TIER, timeout=120)
  yield g
  g.close()
  for k, v in old.items():
    if v is not None: os.environ[k] = v

def _pair(g, path: str, dtype: str = "float32", max_len: int = 128):
  from fusezap.serve.models import load_model
  ref = load_model(path, dtype=getattr(dtypes, dtype), device=TIER, max_len=max_len, prefill_chunk=32)
  g.post("load", {"model": path, "dtype": dtype, "max_len": max_len, "prefill_chunk": 32})
  m = load_model(path, dtype=getattr(dtypes, dtype), device=g.device, max_len=max_len, prefill_chunk=32, tp=g)
  g.barrier("load")
  return ref, m

def _collectives(name: str) -> int:   # NKI collectives a kernel holds: AllReduceSpec one, a fused TP layer (LayerSpec part "tp") two
  return 1 if name.startswith("fz_all_reduce") else (2 if "_tp2_" in name else 0)

def _ar_launches(m) -> int:
  """The NKI collectives recorded in the captured steps (separate AllReduceSpec launches, or inside the fused TP layers)."""
  n = 0
  for jf in m._neuron_steps.jits.values():
    for it in (jf.captured.items if jf.captured is not None else []):
      sp = getattr(it.si.ast[0], "spec", None) if it.si.ast else None
      if sp is None: continue
      n += sum(_collectives(p[0]) for p in sp.parts) if hasattr(sp, "parts") else _collectives(sp.name)
  return n

def _logits(m, ids):
  m.reset()
  out = [m.prefill([ids])]
  for t in np.random.default_rng(4).integers(0, 199, 6): out.append(m.decode([[int(t)]]))
  return out

def test_tp2_picks_the_collective_and_matches_tp1(group, tmp_path):
  """The launch's self-test picks the NKI collective (here: the shim's collective through the host sum); Llama.prefill /
  decode on the TP=2 model give TP=1's logits (float32 tolerance: the row-parallel sums group differently) and greedy tokens,
  each layer's two all-reduces recorded in the captured steps (since the I72 follow-up inside the fused TP layer kernels)."""
  from test_tp import tiny_model_dir
  assert group.ar_info.startswith("NKI collective all-reduce") and NT._comm(group).mode == "cc", group.ar_info
  ref, m = _pair(group, tiny_model_dir(str(tmp_path / "t")))
  assert (m.cfg.n_heads, m.cfg.n_kv_heads, m.cfg.intermediate_size) == (2, 1, 48) and m.kmajor
  assert tuple(m.layers[0].wqkv.shape) == (64, 4 * 16) and tuple(m.layers[0].wd.shape) == (48, 64)   # K-major shards
  ids = list(np.random.default_rng(3).integers(0, 199, 45))
  for a, b in zip(_logits(ref, ids), _logits(m, ids)):
    np.testing.assert_allclose(b, a, rtol=1e-4, atol=1e-4)
    assert (a.argmax(-1) == b.argmax(-1)).all()
  assert _ar_launches(m) >= 2 * m.cfg.n_layers
  ra = list(ref.generate(ids[:20], 16, stop=[]))
  assert list(m.generate(ids[:20], 16, stop=[])) == ra and len(set(ra)) > 3

def test_tp2_engine_tokens_equal_tp1(group, tmp_path):
  """LLMEngine (NeuronKVCache with the rank's kv heads, continuous batching, prefill chunks) over the TP=2 model: TP=1's
  greedy tokens for five prompts; the worker mirrors every forward."""
  from test_tp import tiny_model_dir
  from fusezap.serve.engine import LLMEngine, SamplingParams
  ref, m = _pair(group, tiny_model_dir(str(tmp_path / "e"), seed=1))
  kw = dict(num_blocks=64, block_size=16, max_num_seqs=8, max_prefill_tokens=64, decode_beam=0)
  e1, e2 = LLMEngine(ref, **kw), LLMEngine(m, **kw)
  assert e2.cache.n_kv == 1 and e1.cache.n_kv == 2
  group.post("engine", e2.tp_spec(), e2.tp_signature())
  group.barrier("engine")
  rng = np.random.default_rng(5)
  ps = [[int(t) for t in rng.integers(0, 199, n)] for n in (5, 33, 70, 12, 90)]
  sp = SamplingParams(max_tokens=12, ignore_eos=True)
  want = e1.generate(ps, sp)
  assert e2.generate(ps, sp) == want and len({t for o in want for t in o}) > 5

# (query heads, kv heads, q/k/v and o_proj biases, extra config): Qwen2.5-7B's head layout, and Llama-3.1-8B's (I13: 32 / 8
# heads, no biases, llama3 RoPE scaling)
BF16_LAYOUTS = {"qwen7b": (28, 4, True, {}),
                "llama8b": (32, 8, False, {"model_type": "llama", "rms_norm_eps": 1e-5, "rope_theta": 5e5,
                                           "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                                                            "original_max_position_embeddings": 64}})}

@pytest.mark.parametrize("layout", list(BF16_LAYOUTS))
def test_tp2_bf16_qwen7b_layout_tokens_equal_tp1(group, tmp_path, layout):
  """bf16 per shard (K-major bf16 slices, MatmulBF16) at Qwen2.5-7B's head layout (28 / 4 heads, head_dim 128: 14 / 2 per
  rank) and at Llama-3.1-8B's (32 / 8, no biases: 16 / 4 per rank): TP=1 bf16's greedy tokens through LLMEngine, logits
  within bf16 tolerance."""
  from test_tp import tiny_model_dir
  from fusezap.serve.engine import LLMEngine, SamplingParams
  Hq, Hkv, bias, extra = BF16_LAYOUTS[layout]
  p = tiny_model_dir(str(tmp_path / "q7"), seed=3, cfg={"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": Hq,
                                                         "num_key_value_heads": Hkv, "head_dim": 128, **extra}, bias=bias)
  ref, m = _pair(group, p, "bfloat16")
  assert m.nbf16 and m.layers[0].wqkv.dtype == dtypes.bfloat16 and (m.cfg.n_heads, m.cfg.n_kv_heads) == (Hq // 2, Hkv // 2)
  assert (m.layers[0].bqkv is not None) == bias
  ids = list(np.random.default_rng(7).integers(0, 199, 20))
  for a, b in zip(_logits(ref, ids), _logits(m, ids)):
    assert np.abs(a - b).max() <= 2e-2 * np.abs(a).max()
  kw = dict(num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=64, decode_beam=0)
  e1, e2 = LLMEngine(ref, **kw), LLMEngine(m, **kw)
  group.post("engine", e2.tp_spec(), e2.tp_signature())
  group.barrier("engine")
  ps = [[int(t) for t in np.random.default_rng(9 + i).integers(0, 199, n)] for i, n in enumerate((7, 30, 18))]
  sp = SamplingParams(max_tokens=10, ignore_eos=True)
  assert e2.generate(ps, sp) == e1.generate(ps, sp)

@pytest.mark.parametrize("env", [{"FZ_NEURON_TP_AR": "host"}, {"FZ_NEURON_TP_HEAD": "vocab"}], ids=["host", "vocab-head"])
def test_tp2_host_mode_and_vocab_head_match_the_collective(group, tmp_path, monkeypatch, env):
  """FZ_NEURON_TP_AR=host (dist.HostComm): the default group's logits (the collective) bit for bit (the same sums).
  FZ_NEURON_TP_HEAD=vocab (each rank V/2 rows of lm_head, the logits all-gathered as a zero-padded sum, exact): the same
  logits up to the shim's float32 BLAS, whose bits depend on the matrix width (V/2 vs V columns), and the same top-1."""
  from test_tp import tiny_model_dir
  path = tiny_model_dir(str(tmp_path / "v"), seed=2)
  ids = list(np.random.default_rng(11).integers(0, 199, 30))
  _, m = _pair(group, path)
  want = _logits(m, ids)
  g2 = _launch(monkeypatch, **env)
  try:
    _, m2 = _pair(g2, path)
    got = _logits(m2, ids)
    if "FZ_NEURON_TP_AR" in env:
      assert g2.ar_info.startswith("host all-reduce") and not m2.tp_vocab, g2.ar_info
      for a, b in zip(want, got): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
    else:
      assert g2.ar_info.startswith("NKI collective") and m2.tp_vocab and tuple(m2.lm_head.shape) == (100, 64)   # float32: torch layout
      for a, b in zip(want, got):
        np.testing.assert_allclose(b, a, rtol=1e-6, atol=1e-6)
        assert (a.argmax(-1) == b.argmax(-1)).all()
  finally: g2.close()

# ---------------------------------------------------------------- 4. N8 run 1's hang: a worker rank's launches stayed queued

FAKE_NRT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_nrtpy")

def _queued_rank(rank: int, shm: str, flush: bool, q) -> None:
  """One rank as in a decode step on the persistent path: the step's launches (here one shm all-reduce) queued in
  NeuronAllocator.pending; rank 0 then reads its result (which sends its batch), a worker rank never reads."""
  os.environ.update({"FZ_NRT_PYTHON": sys.executable, "PYTHONPATH": FAKE_NRT, "FZ_NEURON_TP_SHM": shm, "FZ_NEURON_CC_RANK": str(rank),
                     "FZ_NEURON_CC_WORLD": "2", "FZ_NEURON_TP_SLOT": str(1 << 16), "FZ_TP_TIMEOUT": "3"})
  from fusezap.device.neuron import NeuronAllocator
  al = NeuronAllocator()
  a, b = al._alloc(16), al._alloc(16)
  al.copyin(a, memoryview(np.full(4, rank + 1, np.float32)).cast("B"))
  al.call({"op": "ar", "in": a.id, "out": b.id, "n": 16})
  try:
    if rank == 0:
      out = np.zeros(4, np.float32)
      al.copyout(memoryview(out).cast("B"), b)
      q.put((rank, out.tolist()))
    else:
      if flush: al.flush()   # what NeuronComm.flush (dist's end_op) now does at the end of every mirrored call
      time.sleep(5)
      q.put((rank, None))
  except Exception as e: q.put((rank, f"{type(e).__name__}: {str(e)[:300]}"))

@pytest.mark.parametrize("flush", [False, True], ids=["worker-never-flushes", "end-of-call-flush"])
def test_worker_rank_queued_collective_hang_and_fix(flush):
  """N8 inf2 run 1 hung in the first step: a worker rank never reads results, so the launches after its last NEFF load
  (collectives among them) stayed in NeuronAllocator.pending, while rank 0's read sent its own and its runtime waited in
  the collective (CCOM "Timeout waiting for RX"). Reproduced with two real runtime workers (fake nrtpy, the shm
  all-reduce, a 3 s timeout): without the flush rank 0 times out; with it (dist's end_op) rank 0 gets the sum."""
  path = NT.shm_create(2, 1 << 16)
  ctx = mp.get_context("spawn")
  q = ctx.Queue()
  ps = [ctx.Process(target=_queued_rank, args=(r, path, flush, q)) for r in range(2)]
  for p in ps: p.start()
  got = dict(q.get(timeout=60) for _ in ps)
  for p in ps: p.join(30)
  if flush: assert got[0] == [3.0] * 4, got
  else: assert isinstance(got[0], str) and "did not post" in got[0], got

def _skew(g) -> None: NT._comm(g).n += 1   # a worker that ran one collective rank 0 did not

def test_desync_fails_fast_naming_the_call(tmp_path, monkeypatch):
  """The desync guard: rank 0's collective count travels with every message; a worker whose count differs refuses the
  next call (TPDesync) instead of pairing its collectives with the wrong launches or waiting in one forever."""
  dead: list = []
  for k in ("FZ_NEURON_TP_AR", "FZ_NEURON_TP_HEAD"): monkeypatch.delenv(k, raising=False)
  here = os.path.dirname(os.path.abspath(__file__))   # the worker imports test_neuron_tp:_skew
  g = dist.launch(2, TIER, timeout=120, on_dead=lambda *a: dead.append(a),
                  env={"PYTHONPATH": os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)})
  try:
    from test_tp import tiny_model_dir
    _, m = _pair(g, tiny_model_dir(str(tmp_path / "d")))
    m.decode([[5]], 0)                              # in step: both ranks counted the same collectives
    assert NT._comm(g).count() > 0
    g.post("call", "test_neuron_tp:_skew")
    g.barrier("skew")
    g.post("decode", np.array([[6]]), 1)
    with pytest.raises(RuntimeError, match="desync before 'decode'.*rank 0 has launched"): g.barrier("after", 30)
  finally: g.close()

# ---------------------------------------------------------------- 5. the per-shard layer kernel in two halves (merge with I71)

def _setenv(g, env: dict) -> None:
  for k, v in env.items():
    if v is None: os.environ.pop(k, None)
    else: os.environ[k] = v

def test_layer_halves_render_and_args():
  """LayerSpec.part: the attn half ends at o_proj's product (po) and the residual (xo), the mlp half starts from them; the
  whole layer's source is unchanged by the split; bf16 halves edit only their own two weights."""
  from fusezap.codegen import nki_bf16 as nb
  sp = nk.LayerSpec(4, 256, 14, 2, 128, 192, 512, 1e-6, 0.088, 100, True, True, True, "each", tile=True, wcols=2048)
  a, m = (dataclasses.replace(sp, part=p) for p in ("attn", "mlp"))
  assert a.args()[:2] == ("xo", "po") and "wgu" not in a.args() and "bo" not in a.args() and a.outs() == (0, 1, 2, 3, 4)
  assert m.args() == ("xo", "dn", "x", "o", "bo", "ln2", "wgu", "wd") and m.outs() == (0, 1)
  for s in (a, m, dataclasses.replace(a, tile=False), dataclasses.replace(m, tile=False)):
    p = s.render()
    assert p.name == s.name and ("_attn_" in s.name or "_mlp_" in s.name) and p.outs == s.outs()
    nb.LayerBF16(s).render()
  assert "_attn" not in sp.name and sp.render().outs == (0, 1, 2, 3, 4)

def test_tp_layer_chooser(tmp_path):
  """scripts/nki_n8_tp_choose.py: the halves only when faster with the members' c1 tokens; else, or with nothing, the members."""
  import json, subprocess
  script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "nki_n8_tp_choose.py")
  run = lambda: subprocess.run([sys.executable, script, str(tmp_path)], capture_output=True, text=True).stdout.strip()
  assert run() == "FZ_NEURON_TP_LAYER=members"
  w = lambda n, t, ids: (tmp_path / f"serve_tp2_ab_{n}.json").write_text(json.dumps({"c1": {"tpot_ms_median": t, "ids": ids}}))
  w("halves", 40.0, [1, 2]); w("members", 60.0, [1, 2])
  assert run() == "FZ_NEURON_TP_LAYER=halves"
  w("halves", 40.0, [1, 3])
  assert run() == "FZ_NEURON_TP_LAYER=members"

def _logits_tp(g, path, dtype, env):
  g.post("call", "test_neuron_tp:_setenv", env); g.barrier("env")
  _setenv(g, env)
  _, m = _pair(g, path, dtype)
  out = _logits(m, list(np.random.default_rng(13).integers(0, 199, 21)))
  names = set()
  for jf in m._neuron_steps.jits.values():
    for it in (jf.captured.items if jf.captured is not None else []):
      sp = getattr(it.si.ast[0], "spec", None) if it.si.ast else None
      names |= {p[0] for p in sp.parts} if hasattr(sp, "parts") else ({sp.name} if sp is not None else set())
  return out, names

_L8B = {"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128,
        **BF16_LAYOUTS["llama8b"][3]}
@pytest.mark.parametrize("dtype,cfg,tile", [("float32", None, "0"), ("float32", None, "1"),
                                            ("bfloat16", {"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 28,
                                                          "num_key_value_heads": 4, "head_dim": 128}, "1"),
                                            ("bfloat16", _L8B, "1")],
                         ids=["f32", "f32-tiled", "bf16-7b-layout-tiled", "bf16-llama8b-layout-tiled"])
def test_tp2_layer_halves_bitwise_vs_shard_members(tmp_path, monkeypatch, dtype, cfg, tile):
  """TP=2 decode with each shard's layer as LayerSpec's two halves (FZ_NEURON_TP_LAYER=halves, the default) gives the
  per-shard members' logits bit for bit (FZ_NKISHIM_MM=f64: TensorE's row-invariant products), tiled or not, float32 and
  bf16 at Qwen2.5-7B's head layout (14 / 2 heads per rank); and TP=1's greedy tokens."""
  from test_tp import tiny_model_dir
  for k in ("FZ_NEURON_TP_AR", "FZ_NEURON_TP_HEAD", "FZ_NEURON_DTYPE"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  here = os.path.dirname(os.path.abspath(__file__))
  g = dist.launch(2, TIER, timeout=120, env={"PYTHONPATH": os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)})
  try:
    path = tiny_model_dir(str(tmp_path / "h"), seed=5, cfg=cfg, bias=cfg is not _L8B)   # Llama-3.1-8B's layout: no biases
    mem, n1 = _logits_tp(g, path, dtype, {"FZ_NEURON_TP_LAYER": "members", "FZ_NEURON_LAYER_TILE": tile})
    hal, n2 = _logits_tp(g, path, dtype, {"FZ_NEURON_TP_LAYER": "halves", "FZ_NEURON_LAYER_TILE": tile})
    assert not any("_attn_" in n for n in n1) and any("_attn_" in n for n in n2) and any("_mlp_" in n for n in n2), (n1, n2)
    assert all(("t4" in n) == (tile == "1") for n in n2 if "_attn_" in n or "_mlp_" in n) if tile == "1" else True
    for a, b in zip(mem, hal): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
    from fusezap.serve.models import load_model
    ref = load_model(path, dtype=getattr(dtypes, dtype), device=TIER, max_len=128, prefill_chunk=32)
    _, m = _pair(g, path, dtype)
    ids = list(np.random.default_rng(17).integers(0, 199, 12))
    assert list(m.generate(ids, 10, stop=[])) == list(ref.generate(ids, 10, stop=[]))
  finally:
    _setenv(g, {"FZ_NEURON_TP_LAYER": None, "FZ_NEURON_LAYER_TILE": None})
    g.close()

# ---------------------------------------------------------------- 5. N8 I72 follow-up: the all-reduces inside the layer kernel

def test_layer_tp_render():
  """LayerSpec(part="tp"): the whole layer's source plus two collectives (o_proj's product in its SBUF tile, down's product
  on its way to dn), the collectives import, the whole layer's arguments and outputs; bf16 edits the same four weights."""
  from fusezap.codegen import nki_bf16 as nb
  sp = nk.LayerSpec(4, 256, 14, 2, 128, 192, 512, 1e-6, 0.088, 100, True, True, True, "stack", early=True, tile=True, wcols=2048)
  t = dataclasses.replace(sp, part="tp", tpw=2)
  p, w = t.render(), sp.render()
  assert "_tp2_" in t.name and t.args() == sp.args() and p.outs == w.outs and p.src.startswith(nk.CC_HEADER)
  assert p.src.count("nki.collectives.all_reduce(") == 2 and "ReplicaGroup([[0, 1]])" in p.src
  wl = set(w.src.split("\n"))
  extra = [l for l in p.src.split("\n") if l not in wl]
  assert len(extra) == 2 + 2 * 6 and sum("dma_copy" in l for l in extra) == 4, extra   # import, def, per collective a comment + 5
  bf = lambda s: nb.LayerBF16(s).render().src.count("dtype=nl.bfloat16")
  assert bf(t) == bf(sp) > 0
  for bad in (dataclasses.replace(sp, part="tp"), dataclasses.replace(sp, tpw=2), dataclasses.replace(sp, part="x")):
    with pytest.raises(nk.NKIUnsupported): bad.render()

@pytest.mark.parametrize("form,k", [("dep", 1), ("dep", 7), ("indep", 5)])
def test_ar_chain_spec_on_the_shim(form, k, monkeypatch):
  """ARChainSpec (the probe's k collectives in one kernel): dep doubles the sum per step after the first (2^(k-1) * sum,
  exact), indep gives the sum; k collectives reach the peer hook."""
  calls = []
  def peer(a, op, g):
    calls.append(a.size)
    return a * np.float32(2.0)   # two identical ranks
  monkeypatch.setattr(nkishim, "COLLECTIVE", [peer])
  sp = nk.ARChainSpec(3584, 2, k, form)
  p = sp.render()
  assert p.src.count("nki.collectives.all_reduce(") == k and p.name == sp.name and f"_k{k}" in sp.name
  x = np.random.default_rng(k).standard_normal(3584).astype(np.float32)
  out = np.zeros_like(x)
  nkishim.execute(p, [out, x])
  want = x * np.float32(2.0) * (np.float32(2.0 ** (k - 1)) if form == "dep" else np.float32(1.0))
  np.testing.assert_array_equal(out, want)
  assert len(calls) == k
  with pytest.raises(nk.NKIUnsupported): nk.ARChainSpec(8, 2, 0).render()

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")

def test_probe_chain_and_layer_dry(tmp_path):
  """scripts/nki_tp_probe.py --only chain,layer --dry (NKISHIM in process, the two ranks as threads): the chain is exact at
  every k and fits a per-launch / per-collective line; the tiny shard layer's tp kernel gives the halves + shm path's bits
  (xo, dn, pool, scratch) on both ranks, and FUSE tp layers in one fused NEFF give the single launches' bits."""
  import json, subprocess
  out = tmp_path / "p.json"
  p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "nki_tp_probe.py"), "--dry", "--only", "chain,layer", "--tiny", "--rows", "1,4",
                      "--fuse-layers", "3", "--chain-k", "1,3", "--iters", "2", "--f32", "--out", str(out)],
                     capture_output=True, text=True, timeout=600, env={**os.environ, "FZ_NKISHIM_MM": "f64"})
  assert p.returncode == 0, p.stderr[-3000:]
  R = json.loads(out.read_text())
  runs = R["chain"]["runs"]
  assert len(runs) == 6 and all(r.get("exact") for r in runs), runs
  assert {"per_launch_ms", "per_collective_ms"} <= set(R["chain"]["fit_dep_3584"])
  for rec in R["layer"]["runs"]:
    assert rec.get("ok") and all(rec["bitwise_tp_vs_halves"].values()) and rec["fused_bitwise_vs_single"], rec
    assert "_tp2_" in rec["kernel"]["tp"] and rec["fused3_ms_per_layer"]
  assert "PREDICTION rows 4" in p.stdout

def test_ar_chooser(tmp_path):
  """scripts/nki_n8_ar_choose.py: the fastest variant whose c1 tokens equal the baseline's (the first row); none: the baseline."""
  import json, subprocess
  run = lambda: subprocess.run([sys.executable, os.path.join(SCRIPTS, "nki_n8_ar_choose.py"), str(tmp_path)], capture_output=True, text=True).stdout.strip()
  assert run() == "FZ_NEURON_TP_LAYER=halves"
  (tmp_path / "ar_variants.tsv").write_text("ar_h\tFZ_NEURON_TP_LAYER=halves FZ_NEURON_TP_AR=shm\nar_f1\tFZ_NEURON_TP_LAYER=fused FZ_NEURON_FUSE=1\n"
                                            "ar_f28\tFZ_NEURON_TP_LAYER=fused FZ_NEURON_FUSE=28\n")
  w = lambda n, t, ids, c4=60.0: (tmp_path / f"serve_{n}.json").write_text(json.dumps({"c1": {"tpot_ms_median": t, "ids": ids}, "c4": {"tpot_ms_median": c4}}))
  w("ar_h", 58.0, [1, 2])
  assert run() == "FZ_NEURON_TP_LAYER=halves FZ_NEURON_TP_AR=shm"
  w("ar_f1", 45.0, [1, 2]); w("ar_f28", 40.0, [1, 3])   # f28 faster but different tokens: not taken
  assert run() == "FZ_NEURON_TP_LAYER=fused FZ_NEURON_FUSE=1"
  w("ar_f28", 44.5, [1, 2], 50.0)   # within 2% of f1 at c1, better c4
  assert run() == "FZ_NEURON_TP_LAYER=fused FZ_NEURON_FUSE=28"

def test_ar_job_script_parses():
  import subprocess
  assert subprocess.run(["bash", "-n", os.path.join(SCRIPTS, "nki_n8_ar_inf2.sh")]).returncode == 0

def test_prefill_job_script_parses_and_keeps_the_defaults():
  """scripts/nki_n8_prefill_inf2.sh (I75): bash -n; it leaves the TP decode defaults alone (no FZ_NEURON_FUSE / _HEAD_FUSE /
  _TP_AR pins: they are unset), verifies the default mix first, reuses the I73 AWS numbers (a valid NxDI result file)."""
  import json, subprocess
  p = os.path.join(SCRIPTS, "nki_n8_prefill_inf2.sh")
  assert subprocess.run(["bash", "-n", p]).returncode == 0
  src = open(p).read()
  assert "unset FZ_NEURON_FUSE FZ_NEURON_FUSE_MAX FZ_NEURON_HEAD_FUSE FZ_NEURON_TP_AR" in src and "export FZ_NEURON_FUSE" not in src
  assert src.index("pf_variant mix_shm_cc") < src.index("pf_variant mix_cc_cc") < src.index("pf_variant pf_fused ")
  assert "aws7b" not in src and "setup_nxdi" not in src.split("set -u", 1)[1]
  ref = json.load(open(os.path.join(SCRIPTS, "nki_n8_prefill_aws_ref.json")))
  assert ref["label"] == "tp2_bf16" and ref["c1"]["ttft_ms_p50"] == 50.616 and ref["c1"]["tpot_ms_p50"] == 40.601

def test_prefill_chooser(tmp_path):
  """scripts/nki_n8_prefill_choose.py: the lowest c4 burst TTFT p99 among the variants with the baseline's tokens (c1 ids,
  greedy ids, TTFT first tokens); none recorded: the members."""
  import json, subprocess
  run = lambda: subprocess.run([sys.executable, os.path.join(SCRIPTS, "nki_n8_prefill_choose.py"), str(tmp_path)], capture_output=True, text=True).stdout.strip()
  assert run() == "FZ_NEURON_TP_PREFILL=members"
  (tmp_path / "pf_variants.tsv").write_text("mix\tFZ_NEURON_TP_PREFILL=members\nf\tFZ_NEURON_TP_PREFILL=fused\n"
                                            "fc\tFZ_NEURON_TP_PREFILL=fused FZ_NEURON_PREFILL_CTX=1\nfs\tFZ_NEURON_TP_PREFILL=fused FZ_NEURON_PREFILL_CTX=128,256,512\n")
  def w(n, c4, c1, ids=(1, 2), g=(5, 6)):
    (tmp_path / f"serve_{n}.json").write_text(json.dumps({"c1": {"ids": list(ids), "tpot_ms_median": 38.5},
      "greedy": {"rows": [{"id": "p00", "ids": list(g)}]}, "ttft": {"c4_p99": c4, "c4_p50": c4 / 2, "c1_p50": c1, "c1_ids": [7], "c4_ids": [8]}}))
  w("mix", 760.0, 190.0)
  assert run() == "FZ_NEURON_TP_PREFILL=members"
  w("f", 480.0, 120.0); w("fc", 400.0, 100.0)
  assert run() == "FZ_NEURON_TP_PREFILL=fused FZ_NEURON_PREFILL_CTX=1"
  w("fs", 200.0, 50.0, g=(5, 7))   # faster, but its greedy tokens differ: not taken
  assert run() == "FZ_NEURON_TP_PREFILL=fused FZ_NEURON_PREFILL_CTX=1"
  w("fs", 395.0, 60.0)             # within 2% at c4, better c1
  assert run() == "FZ_NEURON_TP_PREFILL=fused FZ_NEURON_PREFILL_CTX=128,256,512"

def test_tp_layer_env(monkeypatch):
  monkeypatch.delenv("FZ_NEURON_TP_LAYER", raising=False)
  assert NT.tp_layer() == "fused"   # the default since the I72 follow-up run
  monkeypatch.setenv("FZ_NEURON_TP_LAYER", "fused")
  assert NT.tp_layer() == "fused" and NT.fused_layer(None) == 0
  monkeypatch.setenv("FZ_NEURON_TP_LAYER", "halves")
  assert NT.tp_layer() == "halves"
  monkeypatch.setenv("FZ_NEURON_TP_LAYER", "nope")
  with pytest.raises(ValueError): NT.tp_layer()

def _spec_items(jf) -> list:
  return [getattr(it.si.ast[0], "spec", None) for it in jf.captured.items if it.si.ast and getattr(it.si.ast[0], "spec", None) is not None]

@pytest.mark.parametrize("dtype,cfg,tile,fuse", [("float32", None, "0", "1"), ("float32", None, "1", "24"),
                                                 ("bfloat16", {"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 28,
                                                               "num_key_value_heads": 4, "head_dim": 128}, "1", "24")],
                         ids=["f32-fuse1", "f32-tiled-fuse24", "bf16-7b-layout-tiled-fuse24"])
def test_tp2_fused_layer_bitwise_vs_halves(tmp_path, monkeypatch, dtype, cfg, tile, fuse):
  """FZ_NEURON_TP_LAYER=fused (one LayerSpec(part="tp") kernel per shard and layer with both collectives inside it;
  FZ_NEURON_FUSE then puts several layers and the head in one kernel) gives the halves' logits bit for bit (the halves'
  all-reduce here: the shim's collective = the group's float32 host sum, which is also the shm path's sum for two ranks),
  tiled or not, float32 and bf16 at Qwen2.5-7B's head layout; with FUSE=24 a decode step is one launch per rank holding
  every layer's two collectives (NxD Inference's shape); TP=1's greedy tokens."""
  from test_tp import tiny_model_dir
  for k in ("FZ_NEURON_TP_AR", "FZ_NEURON_TP_HEAD", "FZ_NEURON_DTYPE"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  here = os.path.dirname(os.path.abspath(__file__))
  g = dist.launch(2, TIER, timeout=120, env={"PYTHONPATH": os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)})
  try:
    path = tiny_model_dir(str(tmp_path / "f"), seed=6, cfg=cfg)
    env = {"FZ_NEURON_LAYER_TILE": tile, "FZ_NEURON_FUSE": fuse}
    hal, n1 = _logits_tp(g, path, dtype, {**env, "FZ_NEURON_TP_LAYER": "halves"})
    c0 = NT._comm(g).n
    fus, n2 = _logits_tp(g, path, dtype, {**env, "FZ_NEURON_TP_LAYER": "fused"})
    assert NT.fused_layer(g) == 2 and NT._comm(g).n > c0
    assert any("_attn_" in n for n in n1) and not any("_tp2_" in n for n in n1), n1
    assert any("_tp2_" in n for n in n2) and not any("_attn_" in n or "_mlp_" in n for n in n2), n2
    for a, b in zip(hal, fus): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
    _, m = _pair(g, path, dtype)
    _logits(m, [1, 2, 3])
    dec = [jf for (N, R), jf in m._neuron_steps.jits.items() if N == 1 and jf.captured is not None]
    assert dec
    for jf in dec:
      sps = _spec_items(jf)
      assert not any(isinstance(s, nk.AllReduceSpec) for s in sps), [s.name for s in sps]   # no all-reduce launch left
      if fuse == "24": assert len(sps) == 1 and isinstance(sps[0], nk.FusedSpec), [s.name for s in sps]   # one launch per step
    from fusezap.serve.models import load_model
    ref = load_model(path, dtype=getattr(dtypes, dtype), device=TIER, max_len=128, prefill_chunk=32)
    ids = list(np.random.default_rng(19).integers(0, 199, 12))
    assert list(m.generate(ids, 10, stop=[])) == list(ref.generate(ids, 10, stop=[]))
  finally:
    _setenv(g, {"FZ_NEURON_TP_LAYER": None, "FZ_NEURON_LAYER_TILE": None, "FZ_NEURON_FUSE": None})
    g.close()

# ---------------------------------------------------------------- 6. the TP defaults since the I72 follow-up inf2 run

def test_fuse_defaults_follow_the_tp_world(monkeypatch):
  """neuron.fuse_layers / fuse_max: TP=1 keeps 24 layers and 44 items per fused kernel; under TP (neuron_tp.install) every
  layer and no cap; an explicit FZ_NEURON_FUSE / _FUSE_MAX wins either way ("" = 0 as before)."""
  from fusezap.device import neuron as nd
  for k in ("FZ_NEURON_FUSE", "FZ_NEURON_FUSE_MAX"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setattr(nd, "TP_WORLD", [1])
  assert (nd.fuse_layers(), nd.fuse_max(), nd.head_fuse()) == (24, 44, True)
  nd.TP_WORLD[0] = 2
  assert (nd.fuse_layers(), nd.fuse_max()) == (nd.FUSE_ALL, 0)
  monkeypatch.setenv("FZ_NEURON_FUSE", "4"); monkeypatch.setenv("FZ_NEURON_FUSE_MAX", "")
  assert (nd.fuse_layers(), nd.fuse_max()) == (4, 0)

def _tp_env_clear(monkeypatch) -> None:
  for k in ("FZ_NEURON_TP_AR", "FZ_NEURON_TP_HEAD", "FZ_NEURON_DTYPE", "FZ_NEURON_TP_LAYER", "FZ_NEURON_FUSE", "FZ_NEURON_FUSE_MAX",
            "FZ_NEURON_HEAD_FUSE", "FZ_NEURON_LAYER_TILE"): monkeypatch.delenv(k, raising=False)

def _decode_specs(m) -> list:
  return [[s.name for s in _spec_items(jf)] for (N, R), jf in m._neuron_steps.jits.items() if N == 1 and jf.captured is not None]

def test_tp2_defaults_fused_one_launch_per_step_and_fallback(tmp_path, monkeypatch):
  """No switches set: a TP=2 group runs the decode layers fused with their collectives inside (the launch line says so),
  FZ_NEURON_FUSE defaults to every layer with the head fused, so a decode step is ONE launch per rank (a FusedSpec holding
  every layer's two collectives), with TP=1's greedy tokens; the group's close restores the TP=1 defaults. If the collective
  fails its launch self-test the group falls back on its own: the layers run as the halves around the separate all-reduces
  (here the host mode: the shim has no shm), with the same logits bit for bit."""
  from test_tp import tiny_model_dir
  from fusezap.device import neuron as nd
  from fusezap.serve.models import load_model
  _tp_env_clear(monkeypatch)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  path = tiny_model_dir(str(tmp_path / "d"), seed=8)
  ids = list(np.random.default_rng(23).integers(0, 199, 21))
  g = dist.launch(2, TIER, timeout=120)
  try:
    assert "one kernel each with its two NKI collectives inside" in g.ar_info and NT._comm(g).layer_cc, g.ar_info
    assert NT.fused_layer(g) == 2 and nd.fuse_layers() == nd.FUSE_ALL and nd.fuse_max() == 0 and nd.head_fuse()
    _, m = _pair(g, path)
    want = _logits(m, ids)
    dec = _decode_specs(m)
    assert dec and all(len(d) == 1 and d[0].startswith("fz_fused") for d in dec), dec
    sp = next(_spec_items(jf)[0] for (N, R), jf in m._neuron_steps.jits.items() if N == 1 and jf.captured is not None)
    assert sum(_collectives(p[0]) for p in sp.parts) == 2 * m.cfg.n_layers and any(p[0].startswith("fz_argmax") or "head" in p[0] for p in sp.parts), sp.members
    ref = load_model(path, dtype=dtypes.float32, device=TIER, max_len=128, prefill_chunk=32)
    assert list(m.generate(ids[:12], 10, stop=[])) == list(ref.generate(ids[:12], 10, stop=[]))
  finally: g.close()
  assert nd.TP_WORLD[0] == 1 and nd.fuse_layers() == 24
  real = NT.selftest
  def failing(g, mode, n, *a, **k):   # rank 0 still takes part in the peer's all-reduces, then reports a wrong sum for cc
    r = real(g, mode, n, *a, **k)
    return {"ok": False, "error": "injected", "mode": mode} if mode == "cc" else r
  monkeypatch.setattr(NT, "selftest", failing)
  g = dist.launch(2, TIER, timeout=120)
  try:
    c = NT._comm(g)
    assert c.mode == "host" and not c.layer_cc and NT.fused_layer(g) == 0 and "halves" in g.ar_info and "injected" in g.ar_info, g.ar_info
    _, m = _pair(g, path)
    got = _logits(m, ids)
    names = {n for d in _decode_specs(m) for n in d}
    assert not any("_tp2_" in n for n in names), names
    for a, b in zip(want, got): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
  finally: g.close()

def test_tp2_explicit_overrides_keep_the_old_paths(tmp_path, monkeypatch):
  """FZ_NEURON_TP_LAYER=halves and FZ_NEURON_FUSE=1 (the I72 job's pins): the halves around AllReduceSpec launches, one
  layer per kernel, the defaults' logits bit for bit."""
  from test_tp import tiny_model_dir
  _tp_env_clear(monkeypatch)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  path = tiny_model_dir(str(tmp_path / "o"), seed=9)
  ids = list(np.random.default_rng(29).integers(0, 199, 17))
  g = dist.launch(2, TIER, timeout=120)
  try:
    _, m = _pair(g, path)
    want = _logits(m, ids)
  finally: g.close()
  monkeypatch.setenv("FZ_NEURON_TP_LAYER", "halves"); monkeypatch.setenv("FZ_NEURON_FUSE", "1")
  g = dist.launch(2, TIER, timeout=120)
  try:
    from fusezap.device import neuron as nd
    assert nd.fuse_layers() == 1 and NT.fused_layer(g) == 0 and "LayerSpec halves" in g.ar_info, g.ar_info
    _, m = _pair(g, path)
    got = _logits(m, ids)
    names = {n for d in _decode_specs(m) for n in d}
    assert any(n.startswith("fz_all_reduce") for n in names) and not any("_tp2_" in n for n in names), names
    for a, b in zip(want, got): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
  finally: g.close()

# ---------------------------------------------------------------- 7. I75: the prefill's all-reduces inside its fused kernels

Q7B = {"hidden_size": 256, "intermediate_size": 384, "num_attention_heads": 28, "num_key_value_heads": 4, "head_dim": 128}   # 14 / 2 per rank

def test_plan_mix_states_the_default_mix():
  """neuron_tp.plan_mix (setup's choice from the modes that passed): the separate all-reduces take the cheapest measured
  mode; the decode layers and the prefill hold their collectives inside their kernels whenever cc passed. inf2's measured
  costs give the default mix: shm separate, cc in the decode layers and in the prefill's fused kernels."""
  inf2 = {"shm": 0.052, "cc": 0.384}
  assert NT.plan_mix(inf2, "fused", "fused") == {"separate": "shm", "layer_cc": True, "prefill_cc": True, "cc_ok": True, "cost": inf2}
  assert NT.plan_mix({"cc": 0.4}, "fused", "fused")["separate"] == "cc"
  p = NT.plan_mix({"shm": 0.05}, "fused", "fused")   # cc failed its load or self-test: no collectives inside kernels
  assert (p["separate"], p["layer_cc"], p["prefill_cc"], p["cc_ok"]) == ("shm", False, False, False)
  p = NT.plan_mix({}, "fused", "fused")
  assert (p["separate"], p["layer_cc"], p["prefill_cc"]) == ("host", False, False)
  p = NT.plan_mix(inf2, "halves", "members")
  assert (p["separate"], p["layer_cc"], p["prefill_cc"]) == ("shm", False, False)
  assert NT.plan_mix(inf2, "fused", "members")["layer_cc"] and not NT.plan_mix(inf2, "fused", "members")["prefill_cc"]

def test_tp_prefill_env(monkeypatch):
  from fusezap.device import neuron as nd
  for k in ("FZ_NEURON_TP_PREFILL", "FZ_NEURON_TP_PREFILL_FUSE"): monkeypatch.delenv(k, raising=False)
  assert NT.tp_prefill() == "fused" and NT.prefill_fuse() == nd.FUSE_ALL and NT.PREFILL_FUSE == "all" and NT.prefill_fuse_name() == "all" and NT.fused_prefill(None) == 0   # I75 inf2: all
  monkeypatch.setenv("FZ_NEURON_TP_PREFILL", "members")
  assert NT.tp_prefill() == "members"
  monkeypatch.setenv("FZ_NEURON_TP_PREFILL", "nope")
  with pytest.raises(ValueError): NT.tp_prefill()
  for v, want in (("all", nd.FUSE_ALL), ("0", nd.FUSE_ALL), ("7", 7), ("1", 1)):
    monkeypatch.setenv("FZ_NEURON_TP_PREFILL_FUSE", v)
    assert NT.prefill_fuse() == want
  assert NT.prefill_fuse_name() == "1"
  monkeypatch.setenv("FZ_NEURON_TP_PREFILL_FUSE", "x")
  with pytest.raises(ValueError): NT.prefill_fuse()

def test_comm_call_fuse_and_mix(monkeypatch):
  """NeuronComm.call(fuse=True): the phbm collective marked fusable when the prefill holds its collectives (prefill_cc),
  whatever the separate mode; without prefill_cc the separate mode's item (shm: None, the AllReduceCall); mix() says which."""
  monkeypatch.delenv("FZ_NEURON_TP_AR_FUSE", raising=False)
  class B:
    rank, world = 0, 2
  c = NT.NeuronComm(B(), "NEURON", "shm")
  assert c.call(64, dtypes.float32) is None and c.call(64, dtypes.float32, fuse=True) is None
  c.prefill_cc = True
  sp = c.call(64, dtypes.float32, fuse=True).spec
  assert isinstance(sp, nk.AllReduceSpec) and sp.fuse and sp.fusable and sp.kind == "phbm" and sp.name == nk.AllReduceSpec(64, 2).name
  assert c.call(64, dtypes.float32) is None and c.call(64, dtypes.bfloat16, fuse=True) is None
  c.mode = "cc"
  assert not c.call(64, dtypes.float32).spec.fusable
  c.mode, c.layer_cc = "shm", True
  m = c.mix()
  assert m["separate"].startswith("shm") and m["decode_layers"].startswith("cc") and m["prefill"].startswith("cc in the fused prefill")
  c.layer_cc = c.prefill_cc = False
  assert c.mix() == {"separate": f"shm, host above {NT.SHM_SLOT} bytes", "decode_layers": "halves around shm all-reduces", "prefill": "members + shm all-reduces"}

def test_separate_mode_by_message_size(monkeypatch):
  """NeuronComm.mode_for (I75): in shm mode a float32 message from FZ_NEURON_TP_AR_BIG bytes (default 256 KiB: a 128-row
  prefill's 1.8 MB, a scoring bucket's 7-15 MB) takes the cc launch when cc passed its self-test; above the shm slot always
  (cc, else the host); small messages and the other modes keep the setup's mode; call() records what mode_for says."""
  monkeypatch.delenv("FZ_NEURON_TP_AR_BIG", raising=False)
  class B:
    rank, world = 0, 2
  c = NT.NeuronComm(B(), "NEURON", "shm")
  row, big, huge = 3584, 128 * 3584, NT.SHM_SLOT // 4 + 1
  assert [c.mode_for(n, dtypes.float32) for n in (row, big, huge)] == ["shm", "shm", "host"]   # no cc: shm, the host above the slot
  c.cc_ok = True
  assert NT.ar_big() == 256 * 1024 and [c.mode_for(n, dtypes.float32) for n in (row, big, huge)] == ["shm", "cc", "cc"]
  assert c.call(row, dtypes.float32) is None and isinstance(c.call(big, dtypes.float32).spec, nk.AllReduceSpec)
  assert c.mix()["separate"] == "shm, cc from 262144 bytes"
  monkeypatch.setenv("FZ_NEURON_TP_AR_BIG", "0")
  assert [c.mode_for(n, dtypes.float32) for n in (row, big, huge)] == ["shm", "shm", "cc"]
  c.mode = "host"
  assert c.mode_for(big, dtypes.float32) == "host" and c.call(big, dtypes.float32) is None
  c.mode = "cc"
  assert c.mode_for(row, dtypes.float32) == "cc"
  g = type("G", (), {"comm": type("T", (), {"base": c})()})()   # dist.TPComm(NeuronComm)
  NT.set_mode(g, "shm", True, False)   # cc_ok follows the in-kernel collectives unless given
  assert c.cc_ok and c.layer_cc and not c.prefill_cc
  NT.set_mode(g, "shm", False, False, False)
  assert not c.cc_ok

def _prefill_specs(m) -> list:
  return [_spec_items(jf) for (N, R), jf in m._neuron_steps.jits.items() if N != R and jf.captured is not None]

def _coll(sp) -> int:
  return sum(_collectives(p[0]) for p in sp.parts) if hasattr(sp, "parts") else _collectives(sp.name)

@pytest.mark.parametrize("dtype,cfg", [("float32", None), ("bfloat16", Q7B)], ids=["f32", "bf16-7b-layout"])
def test_tp2_fused_prefill_bitwise_vs_members(tmp_path, monkeypatch, dtype, cfg):
  """FZ_NEURON_TP_PREFILL=fused (the default): a prefill's member layers (rows beyond LayerSpec, e.g. 7B's 14 / 2 heads per
  rank at 32 rows) record their all-reduces as collectives inside the fused kernels, FZ_NEURON_TP_PREFILL_FUSE layers per
  kernel (no all-reduce launch left; `all`: one launch per rank and chunk with the head in it), with the logits of the
  members around separate all-reduce launches bit for bit (the same instructions), and TP=1's greedy tokens."""
  from test_tp import tiny_model_dir
  _tp_env_clear(monkeypatch)
  for k in ("FZ_NEURON_TP_PREFILL", "FZ_NEURON_TP_PREFILL_FUSE"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  here = os.path.dirname(os.path.abspath(__file__))
  g = dist.launch(2, TIER, timeout=120, env={"PYTHONPATH": os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)})
  L = 3
  try:
    assert NT._comm(g).prefill_cc and "prefill: the collectives inside its fused kernels, all layers" in g.ar_info, g.ar_info
    path = tiny_model_dir(str(tmp_path / "p"), seed=11, cfg={**(cfg or {}), "num_hidden_layers": L})
    runs = {}
    for name, env in (("members", {"FZ_NEURON_TP_PREFILL": "members", "FZ_NEURON_LAYER": "v1"}), ("fused1", {"FZ_NEURON_TP_PREFILL_FUSE": "1", "FZ_NEURON_LAYER": "v1"}),
                      ("fusedall", {"FZ_NEURON_TP_PREFILL_FUSE": "all", "FZ_NEURON_LAYER": "v1"}), ("default", {"FZ_NEURON_LAYER": "v1"})):
      env = {"FZ_NEURON_TP_PREFILL": None, "FZ_NEURON_TP_PREFILL_FUSE": None, "FZ_NEURON_PREFILL_KERNEL": "v1", **env}   # I75's members path (v2: test_nki_prefill)
      g.post("call", "test_neuron_tp:_setenv", env); g.barrier("env")
      _setenv(g, env)
      _, m = _pair(g, path, dtype)
      ids = list(np.random.default_rng(31).integers(0, 199, 21))
      first, out = _logits(m, ids), _logits(m, ids)   # the second prefill replays the captured forward
      for a, b in zip(first, out): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
      runs[name] = (out, _prefill_specs(m), m)
    (mem, ps_m, _), (f1, ps_1, _), (fa, ps_a, _), (fd, ps_d, m) = (runs[k] for k in ("members", "fused1", "fusedall", "default"))
    sep = [s for ss in ps_m for s in ss if isinstance(s, nk.AllReduceSpec)]
    assert len(sep) == 2 * L and not any(s.fusable for s in sep), [s.name for ss in ps_m for s in ss]   # cc separate (the shim has no shm)
    for ps, groups in ((ps_1, L), (ps_a, 1), (ps_d, 1)):   # the default (all since I75): one group
      assert ps and all(not any(isinstance(s, nk.AllReduceSpec) for s in ss) for ss in ps), ps
      for ss in ps:
        fused = [s for s in ss if isinstance(s, nk.FusedSpec)]
        assert len(fused) == groups and sum(_coll(s) for s in fused) == 2 * L, [s.name for s in ss]
    assert all(len(ss) == 1 for ss in ps_a), ps_a   # all: the chunk's layers and the head in one launch per rank
    for run in (f1, fa, fd):
      for a, b in zip(mem, run): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
    from fusezap.serve.models import load_model
    ref = load_model(path, dtype=getattr(dtypes, dtype), device=TIER, max_len=128, prefill_chunk=32)
    ids = list(np.random.default_rng(37).integers(0, 199, 25))
    assert list(m.generate(ids, 8, stop=[])) == list(ref.generate(ids, 8, stop=[]))
  finally:
    _setenv(g, {"FZ_NEURON_TP_PREFILL": None, "FZ_NEURON_TP_PREFILL_FUSE": None, "FZ_NEURON_LAYER": None, "FZ_NEURON_PREFILL_KERNEL": None})
    g.close()

def test_tp2_fused_prefill_falls_back_without_the_collective(tmp_path, monkeypatch):
  """The collective failing its launch self-test: the prefill runs as the members around the separate mode (host here),
  the launch line and mix() say so, with the fused prefill's logits bit for bit."""
  from test_tp import tiny_model_dir
  _tp_env_clear(monkeypatch)
  for k in ("FZ_NEURON_TP_PREFILL", "FZ_NEURON_TP_PREFILL_FUSE"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  path = tiny_model_dir(str(tmp_path / "b"), seed=12, cfg={**Q7B, "num_hidden_layers": 2})
  ids = list(np.random.default_rng(41).integers(0, 199, 19))
  g = dist.launch(2, TIER, timeout=120)
  try:
    _, m = _pair(g, path, "bfloat16")
    assert not nk.layer_ok(32, m.cfg.n_heads, m.cfg.n_kv_heads, m.cfg.head_dim, "stack")   # the 32-row prefill runs as the members
    _logits(m, ids)
    want, ps = _logits(m, ids), _prefill_specs(m)
    assert NT._comm(g).prefill_cc and ps and not any(isinstance(s, nk.AllReduceSpec) for ss in ps for s in ss) and any(_coll(s) for ss in ps for s in ss)
  finally: g.close()
  real = NT.selftest
  def failing(g, mode, n, *a, **k):
    r = real(g, mode, n, *a, **k)
    return {"ok": False, "error": "injected", "mode": mode} if mode == "cc" else r
  monkeypatch.setattr(NT, "selftest", failing)
  g = dist.launch(2, TIER, timeout=120)
  try:
    c = NT._comm(g)
    assert not c.prefill_cc and NT.fused_prefill(g) == 0 and "prefill: the members around the host all-reduces" in g.ar_info, g.ar_info
    assert "the fused prefill needs the NKI collective" in g.ar_info and NT.mix(g)["prefill"] == "members + host all-reduces"
    _, m = _pair(g, path, "bfloat16")
    _logits(m, ids)
    got, ps = _logits(m, ids), _prefill_specs(m)
    assert ps and not any(_coll(s) for ss in ps for s in ss)   # no collective kernel at all: the host all-reduces
    for a, b in zip(want, got): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
  finally: g.close()

# ---------------------------------------------------------------- 8. I75 follow-up: defaults, the 4-layer fallback, prewarm under TP

_REAL_SPEC_PROGRAM: list = []

def _inject_fused_failure(g, limit) -> None:
  """op "call" (and rank 0): a FusedSpec holding more than `limit` collectives fails to compile (NKISimError) on this rank;
  None restores the device."""
  from fusezap.device import neuron as nd
  if not _REAL_SPEC_PROGRAM: _REAL_SPEC_PROGRAM.append(nd.NEURONDevice.spec_program)
  real = _REAL_SPEC_PROGRAM[0]
  if limit is None:
    nd.NEURONDevice.spec_program = real
    return
  def spec_program(self, sp):
    if isinstance(sp, nk.FusedSpec) and _coll(sp) > limit: raise nd.NKISimError(f"NCC_FAKE: {_coll(sp)} collectives")
    return real(self, sp)
  nd.NEURONDevice.spec_program = spec_program

def _rank_state(g) -> dict:
  """op "call": this rank's unfusable kernels and its engine's captured (forward kind, rows, pool prefix) triples."""
  from fusezap.device import Device
  e = g.worker.eng if g.worker is not None else None
  cap = []
  if e is not None:
    for f in [*e._prefills.values(), *e._decodes.values()]:
      for c, j in [(e.cache.P, f.jit), *f.ctx_jits.items()]:
        if j.captured is not None: cap.append((f.key[0], f.N, c))
  return {"unfusable": len(Device[g.device].__dict__.get("_unfusable", ())), "captured": sorted(cap)}

def test_prefill_ctx_default_under_tp(monkeypatch):
  """I75 defaults: under TP (neuron.TP_WORLD > 1) an unset FZ_NEURON_PREFILL_CTX gives the prefill 128, 256, 512 and the
  pool (prefixes at or past the pool drop); an explicit value wins ("" follows FZ_NEURON_CTX); TP=1 follows FZ_NEURON_CTX."""
  from fusezap.device import neuron as nd
  from fusezap.serve.neuron_kv import NeuronKVCache
  for k in ("FZ_NEURON_PREFILL_CTX", "FZ_NEURON_CTX", "FZ_NEURON_CTX_MIN"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setattr(nd, "TP_WORLD", [1])
  big, small = NeuronKVCache(1, 2, 16, 256, 16, dtypes.float32, TIER), NeuronKVCache(1, 2, 16, 16, 16, dtypes.float32, TIER)
  assert big.ctx_buckets(prefill=True) == big.ctx_buckets() == [512, 1024, 2048, 4096]
  nd.TP_WORLD[0] = 2
  assert big.ctx_buckets(prefill=True) == [128, 256, 512, 4096] and big.ctx_buckets() == [512, 1024, 2048, 4096]
  assert small.ctx_buckets(prefill=True) == [128, 256]   # a 256-slot pool: clamped
  monkeypatch.setenv("FZ_NEURON_CTX", "0")
  assert big.ctx_buckets(prefill=True) == [128, 256, 512, 4096] and big.ctx_buckets() == [4096]
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "")
  assert big.ctx_buckets(prefill=True) == [4096]
  monkeypatch.setenv("FZ_NEURON_PREFILL_CTX", "1024")
  assert big.ctx_buckets(prefill=True) == [1024, 4096]

def _pypath() -> dict:
  here = os.path.dirname(os.path.abspath(__file__))
  return {"PYTHONPATH": os.pathsep.join([here, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}

def test_tp2_fused_all_prefill_falls_back_to_groups_of_4(tmp_path, monkeypatch, capsys):
  """FZ_NEURON_TP_PREFILL_FUSE=all (the default): a 6-layer prefill is one fused kernel per rank whose FusedCall carries
  its split into groups of 4 layers (FusedCall.subs: layers 0-3, then 4-5 and the head). When that kernel fails to compile
  (injected on both ranks: more than 8 collectives) every rank runs the two smaller fused kernels instead (logged), not
  the members, with the same logits bit for bit."""
  from test_tp import tiny_model_dir
  from fusezap.device import Device, neuron as nd
  _tp_env_clear(monkeypatch)
  for k in ("FZ_NEURON_TP_PREFILL", "FZ_NEURON_TP_PREFILL_FUSE"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64"); monkeypatch.setenv("FZ_NEURON_LAYER", "v1")
  path = tiny_model_dir(str(tmp_path / "fb"), seed=13, cfg={"num_hidden_layers": 6})
  ids = list(np.random.default_rng(43).integers(0, 199, 23))
  g = dist.launch(2, TIER, timeout=120, env=_pypath())
  try:
    _, m = _pair(g, path)
    _logits(m, ids); _logits(m, ids)
    want = _logits(m, ids)
    calls = [it.si.ast[0] for (N, R), jf in m._neuron_steps.jits.items() if N != R and jf.captured is not None
             for it in jf.captured.items if it.si.ast and isinstance(it.si.ast[0], nd.FusedCall)]
    assert len(calls) == 1 and _coll(calls[0].spec) == 12 and len(calls[0].subs) == 2, calls
    assert [_coll(s.ast[0].spec) for s, _ in calls[0].subs] == [8, 4] and calls[0].arg_specs
    g.post("call", "test_neuron_tp:_inject_fused_failure", 8); g.barrier("inject")
    _inject_fused_failure(g, 8)
    _, m2 = _pair(g, path)
    _logits(m2, ids); _logits(m2, ids)   # eager, capture: the third call replays the fused kernel
    got = _logits(m2, ids)
    assert Device[g.device]._unfusable and "launching as 2 kernels of up to 4 layers" in capsys.readouterr().err
    g.post("call", "test_neuron_tp:_rank_state")
    assert g.barrier("state")[0]["unfusable"] >= 1
    for a, b in zip(want, got): np.testing.assert_array_equal(a.view(np.int32), b.view(np.int32))
  finally:
    _inject_fused_failure(g, None)
    try:
      g.post("call", "test_neuron_tp:_inject_fused_failure", None); g.barrier("restore")
    except Exception: pass
    Device[g.device].__dict__.pop("_unfusable", None)
    g.close()

def test_tp2_engine_prewarm_mirrors_and_keeps_the_tokens(tmp_path, monkeypatch):
  """I75 prewarm under TP=2 (rank 0 drives; the worker mirrors the synthetic steps, runs precompile through op "call" and
  answers the barriers): every prefill prefix (the TP default 128, 256, and the 512-slot pool) and every decode forward is
  captured on both ranks before the first request, no desync, and the engine's greedy tokens equal TP=1's."""
  from test_tp import tiny_model_dir
  from fusezap.serve.engine import LLMEngine, SamplingParams
  from fusezap.serve import neuron_prewarm as PW
  _tp_env_clear(monkeypatch)
  for k in ("FZ_NEURON_TP_PREFILL", "FZ_NEURON_TP_PREFILL_FUSE", "FZ_NEURON_PREFILL_CTX", "FZ_NEURON_CTX"): monkeypatch.delenv(k, raising=False)
  monkeypatch.setenv("FZ_NKISHIM_MM", "f64")
  g = dist.launch(2, TIER, timeout=120, env=_pypath())
  try:
    ref, m = _pair(g, tiny_model_dir(str(tmp_path / "pw"), seed=14))
    kw = dict(num_blocks=32, block_size=16, max_num_seqs=4, max_prefill_tokens=64, decode_beam=0)
    e1, e2 = LLMEngine(ref, **kw), LLMEngine(m, **kw)
    g.post("engine", e2.tp_spec(), e2.tp_signature()); g.barrier("engine")
    assert e2.cache.ctx_buckets(True) == [128, 256, 512]
    lines: list = []
    rep = PW.prewarm(e2, lines.append)
    want = sorted([("p", 64, c) for c in (128, 256, 512)] + [("d", n, 512) for n in e2.decode_sizes])
    got = sorted((x["forward"][0], int(x["forward"].split("_")[1]), x["ctx"]) for x in rep["forwards"])
    assert got == want and not rep["skipped"], (got, rep["skipped"])
    g.post("call", "test_neuron_tp:_rank_state")
    assert g.barrier("state")[0]["captured"] == want
    assert any(l.startswith("fusezap: prewarm ") and "parallel compile" in l for l in lines), lines
    rng = np.random.default_rng(6)
    ps = [[int(t) for t in rng.integers(0, 199, n)] for n in (5, 40, 70, 12)]
    sp = SamplingParams(max_tokens=10, ignore_eos=True)
    assert e2.generate(ps, sp) == e1.generate(ps, sp)
  finally: g.close()

def test_prefill_job_confirm_mode_runs_the_defaults():
  """scripts/nki_n8_prefill_inf2.sh --confirm (I75 follow-up): its own results dir and 45 min budget, an empty NEFF cache
  for the cold start, the in-process and served phases with NO prefill switch (the code defaults), the in-process run with
  --no-prewarm (c1 = the first request after start); every fz7b_http call passes <label> <accuracy method> [env words];
  the served phase runs the prewarm probe before Validate."""
  import re
  src = open(os.path.join(SCRIPTS, "nki_n8_prefill_inf2.sh")).read()
  conf = src[src.index('if [ "$CONFIRM" = 1 ]; then'):src.index("\nfi\n", src.index('if [ "$CONFIRM" = 1 ]; then'))]
  assert "results/nki_n8_prefill_confirm" in src and "FZ_PF_BUDGET_MIN:-45" in src and 'rm -rf "$FZ_NEFF_CACHE"' in conf
  assert "--no-prewarm" in conf and "tp_phase fz_confirm" in conf and "fz7b_http tp2_bf16_7b_confirm auto" in conf
  assert not re.search(r"FZ_NEURON_(TP_PREFILL|PREFILL_CTX|TP_PREFILL_FUSE)", conf), conf
  assert "unset FZ_NEURON_PREWARM FZ_NEURON_PREWARM_JOBS" in src
  calls = re.findall(r"tp_phase \S+ \d+ fz7b_http (\S+) (\S+)", src)
  assert len(calls) == 3, calls
  for call in calls:
    assert call[1] in ("auto", "none", "text"), call
  body = src.split("fz7b_http() {", 1)[1].split("\n}\n", 1)[0]
  assert body.index("nki_ttft_probe.py") < body.index("validate_pair")

def test_ttft_probe_flags_a_first_request_that_compiles(tmp_path):
  """scripts/nki_ttft_probe.py against a stub server: passes when each length's first request is about as fast as the
  second; fails (exit 1, the length named) when one first request is slow (a compile or load after start)."""
  import http.server, json, subprocess, threading
  seen: dict = {}
  slow = {"n": 0}
  class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
      body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
      n = len(body["prompt"])
      seen[n] = seen.get(n, 0) + 1
      if n == slow["n"] and seen[n] == 1: time.sleep(0.4)
      self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
      self.wfile.write(b'data: {"choices": [{"index": 0, "text": "a"}]}\n\ndata: [DONE]\n\n')
  srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  url = f"http://127.0.0.1:{srv.server_address[1]}"
  run = lambda: subprocess.run([sys.executable, os.path.join(SCRIPTS, "nki_ttft_probe.py"), "--base-url", url, "--model", "m",
                                "--lengths", "10,20", "--slack", "100", "--json", str(tmp_path / "p.json")], capture_output=True, text=True)
  try:
    r = run()
    assert r.returncode == 0 and "PASS" in r.stdout and json.load(open(tmp_path / "p.json"))["ok"], r.stdout + r.stderr
    seen.clear(); slow["n"] = 20
    r = run()
    assert r.returncode == 1 and "[20]" in r.stdout and "NOT PREWARMED" in r.stdout, r.stdout + r.stderr
  finally: srv.shutdown()
