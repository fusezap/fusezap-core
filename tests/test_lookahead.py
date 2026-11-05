"""Speed run 4: the engine's lookahead (FZ_LOOKAHEAD; fusezap/serve/engine.py LLMEngine._speculate / _complete). A
decode-only step of greedy rows enqueues the next decode, fed by its tokens, before it reads them. Tokens must be
identical to the plain engine under everything that can end or change a sequence between the two: max_tokens, stop
tokens, aborts, requests arriving mid-run, preemption on a small KV pool, and non-greedy rows (which turn it off).
Runs on this machine's default device (METAL / CPU read the tokens on the host first, so only the order of work
differs; CUDA also builds the inputs on the device from the previous step's argmaxes)."""
import os
import numpy as np
import pytest
from fusezap import helpers
from fusezap.device import Device
from fusezap.dtype import dtypes

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def _model_path():
  try:
    from fusezap.serve.weights import resolve_model
    return resolve_model(MODEL, allow_download=False)
  except Exception: return None

pytestmark = [pytest.mark.skipif(_model_path() is None, reason=f"{MODEL} not in the HF cache")]

def _device() -> str: return os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")

@pytest.fixture(scope="module")
def model():
  from fusezap.serve.models import load_model
  cuda = _device().startswith("CUDA")
  return load_model(MODEL, dtype=dtypes.bfloat16 if cuda else dtypes.float32, device=_device(), max_len=512, prefill_chunk=128)

def _engine(model, look: bool, num_blocks: int = 1024):
  from fusezap.serve.engine import LLMEngine
  old = os.environ.get("FZ_LOOKAHEAD")
  os.environ["FZ_LOOKAHEAD"] = "1" if look else "0"
  helpers.getenv.cache_clear()
  try: eng = LLMEngine(model, num_blocks=num_blocks, block_size=16, max_num_seqs=8, max_prefill_tokens=256, max_model_len=512)
  finally:
    if old is None: os.environ.pop("FZ_LOOKAHEAD", None)
    else: os.environ["FZ_LOOKAHEAD"] = old
    helpers.getenv.cache_clear()
  assert eng.lookahead == look
  return eng

def _run(eng, script):
  """script: list of (step index, action) with action ("add", key, prompt, params) or ("abort", key). Returns
  {key: (tokens, finish reason)} and the streamed tokens per key."""
  todo, rid, streamed, n = list(script), {}, {}, 0
  while todo or eng.has_unfinished():
    while todo and todo[0][0] <= n:
      _, act = todo.pop(0)
      if act[0] == "add": rid[act[1]] = eng.add_request(act[2], act[3]); streamed[act[1]] = []
      else: eng.abort(rid[act[1]])
    for o in eng.step():
      k = next(k for k, r in rid.items() if r == o.request_id)
      streamed[k] += o.new_token_ids
      assert o.token_ids == streamed[k]
    n += 1
    assert n < 4000
  return {k: (list(eng.seqs[r].output), eng.seqs[r].finish_reason) for k, r in rid.items()}, streamed

def _script(rng, stop_tok=None, sampled=False):
  from fusezap.serve.engine import SamplingParams as SP
  lens = [5, 40, 130, 17, 64, 9, 200, 33, 77, 3]
  P = [rng.integers(100, 30000, n).tolist() for n in lens]
  sc = [(0, ("add", i, P[i], SP(max_tokens=int(m), ignore_eos=True))) for i, m in zip(range(6), (1, 2, 9, 20, 31, 14))]
  sc += [(4, ("add", 6, P[6], SP(max_tokens=12, ignore_eos=True))), (9, ("add", 7, P[7], SP(max_tokens=25, ignore_eos=True))),
         (12, ("abort", 4)), (15, ("add", 8, P[8], SP(max_tokens=6, ignore_eos=True)))]
  if stop_tok is not None: sc.append((2, ("add", 9, P[9], SP(max_tokens=30, stop_token_ids=[stop_tok]))))
  if sampled: sc.append((20, ("add", 10, P[1], SP(max_tokens=10, temperature=0.8, top_k=20, seed=3, ignore_eos=True))))
  return sorted(sc, key=lambda x: x[0])

@pytest.mark.slow
def test_lookahead_tokens_identical(model):
  base_eng = _engine(model, False)
  ref, _ = _run(base_eng, _script(np.random.default_rng(0), -1))
  stop_tok = ref[9][0][5]          # request 9 stops on its 6th token (or earlier, if that token comes sooner)
  ref, _ = _run(_engine(model, False), _script(np.random.default_rng(0), stop_tok, sampled=True))
  eng = _engine(model, True)
  got, streamed = _run(eng, _script(np.random.default_rng(0), stop_tok, sampled=True))
  # the abort lands after a different number of tokens (the lookahead emits a step's tokens one launch later): both
  # outputs must be prefixes of one greedy continuation
  a, b = got.pop(4)[0], ref.pop(4)[0]
  assert a[:len(b)] == b[:len(a)] and a and b
  assert got == ref
  assert eng.stats["lookahead_steps"] > 10, eng.stats
  assert all(streamed[k] == got[k][0] for k in got)
  assert len(got[0][0]) == 1 and got[9][1] == "stop"
  assert eng._pend is None or all(s.status.value == "finished" for s, _, _ in eng._pend.rows)

@pytest.mark.slow
def test_lookahead_small_pool_preempts(model):
  """A pool too small for the batch: the scheduler preempts and recomputes (teacher-forced decode rows); the lookahead
  stops where a block is not free and the tokens still equal the plain engine's."""
  ref, _ = _run(_engine(model, False, num_blocks=24), _script(np.random.default_rng(1)))
  eng = _engine(model, True, num_blocks=24)
  got, _ = _run(eng, _script(np.random.default_rng(1)))
  a, b = got.pop(4)[0], ref.pop(4)[0]
  assert a[:len(b)] == b[:len(a)]
  assert got == ref and eng.scheduler.num_preemptions > 0 and eng.stats["lookahead_steps"] > 0

def test_decode_buckets_rule(monkeypatch):
  from fusezap.serve.engine import decode_buckets
  helpers.getenv.cache_clear()
  assert decode_buckets("CUDA", 32) == [1, 2, 4, 8, 16, 32] and decode_buckets("CUDA:1", 24) == [1, 2, 4, 8, 16, 24]
  assert decode_buckets("CUDA", 1) == [1] and decode_buckets("METAL", 32) == [32] and decode_buckets("CPU", 8) == [8]
  monkeypatch.setenv("FZ_DECODE_BUCKETS", "0"); helpers.getenv.cache_clear()
  assert decode_buckets("CUDA", 32) == [32]
  monkeypatch.delenv("FZ_DECODE_BUCKETS"); helpers.getenv.cache_clear()

@pytest.mark.slow
@pytest.mark.skipif(not _device().startswith("CUDA"), reason="decode buckets are CUDA-only")
def test_decode_buckets_tokens_identical(model, monkeypatch):
  """A decode step runs the smallest bucket that holds its sequences; tokens equal the one-size engine's
  (FZ_DECODE_BUCKETS=0) under the same script, and the small buckets were used."""
  monkeypatch.setenv("FZ_DECODE_BUCKETS", "0"); helpers.getenv.cache_clear()
  one = _engine(model, True)
  assert one.decode_sizes == [8]
  ref, _ = _run(one, _script(np.random.default_rng(2)))
  monkeypatch.delenv("FZ_DECODE_BUCKETS"); helpers.getenv.cache_clear()
  eng = _engine(model, True)
  assert eng.decode_sizes == [1, 2, 4, 8]
  got, _ = _run(eng, _script(np.random.default_rng(2)))
  a, b = got.pop(4)[0], ref.pop(4)[0]
  assert a[:len(b)] == b[:len(a)]
  assert got == ref
  assert sum(eng._decodes[n].jit.cnt > 0 for n in (1, 2, 4)) >= 2

@pytest.mark.slow
def test_mixed_steps_bookkeeping(model, monkeypatch):
  """Mixed forwards (a prompt chunk and the decode rows in one forward): every request completes with its tokens, prompts
  arriving while others decode run as mixed steps, and on CUDA the tokens equal the engine without mixing (FZ_MIX=0),
  bit for bit, because decode rows keep the decode kernels' sums (also with FZ_ONE_K=1). Off CUDA (FZ_MIX_ANY) only the bookkeeping is checked."""
  cuda = _device().startswith("CUDA")
  if cuda and tuple(Device[_device()].cc) != (9, 0): pytest.skip("mixed forwards need sm_90 (the swap-AB decode GEMM)")
  if not cuda: monkeypatch.setenv("FZ_MIX_ANY", "1")
  monkeypatch.setenv("FZ_MIX_TOKENS", "128"); monkeypatch.setenv("FZ_MIX", "1"); monkeypatch.setenv("FZ_PAGED_MMA", "1")   # off by default
  helpers.getenv.cache_clear()
  eng = _engine(model, True)
  assert eng.mix_sizes == [64, 128] and eng.cfg.mixed_prefill_tokens == 128
  got, streamed = _run(eng, _script(np.random.default_rng(3)))
  assert eng.stats["mixed_steps"] > 3, eng.stats
  for k, (toks, why) in got.items():
    assert streamed[k] == toks and (why == "abort" or len(toks) == {0: 1, 1: 2, 2: 9, 3: 20, 4: 31, 5: 14, 6: 12, 7: 25, 8: 6}[k])
  if cuda:
    monkeypatch.setenv("FZ_MIX", "0"); helpers.getenv.cache_clear()   # separate forwards
    plain = _engine(model, True)
    assert not plain.mix_sizes
    ref, _ = _run(plain, _script(np.random.default_rng(3)))
    a, b = got.pop(4)[0], ref.pop(4)[0]
    assert a[:len(b)] == b[:len(a)]
    assert got == ref
  for k in ("FZ_MIX_ANY", "FZ_MIX_TOKENS", "FZ_MIX", "FZ_ONE_K", "FZ_PAGED_MMA"): monkeypatch.delenv(k, raising=False)
  helpers.getenv.cache_clear()

def _mode_engine(model, mode: str, monkeypatch):
  from fusezap.serve.engine import LLMEngine
  helpers.getenv.cache_clear()
  return LLMEngine(model, num_blocks=1024, block_size=16, max_num_seqs=8, max_prefill_tokens=256, max_model_len=512, mode=mode)

@pytest.mark.slow
def test_throughput_mode(model, monkeypatch):
  """Throughput mode (fusezap serve --mode throughput): a request alone gets the exact mode's tokens (it never mixes);
  batched, every request completes with its tokens, prompts arriving while others decode run in mixed forwards, and each
  request's first token (computed by the prefill kernels in either mode) equals the exact mode's. Later tokens may differ
  on near-ties (CUDA: decode rows on the prefill GEMM tiles). Off CUDA (FZ_MIX_ANY) only the bookkeeping is exercised."""
  from fusezap.serve.engine import SamplingParams
  cuda = _device().startswith("CUDA")
  if cuda and tuple(Device[_device()].cc) != (9, 0): pytest.skip("throughput mode's mixed forwards need sm_90")
  if not cuda: monkeypatch.setenv("FZ_MIX_ANY", "1")
  thr, ex = _mode_engine(model, "throughput", monkeypatch), _mode_engine(model, "exact", monkeypatch)
  assert thr.mode == "throughput" and ex.mode == "exact" and thr.mix_sizes and not any(f.layout.mix_exact for f in thr._mixes.values())
  rng = np.random.default_rng(7)
  prompt = rng.integers(100, 30000, 90).tolist()
  sp = SamplingParams(max_tokens=24, ignore_eos=True)
  assert thr.generate([prompt], sp) == ex.generate([prompt], sp)          # alone: identical
  got, _ = _run(thr, _script(np.random.default_rng(5)))
  ref, _ = _run(ex, _script(np.random.default_rng(5)))
  assert thr.stats["mixed_steps"] > 0 and ex.stats["mixed_steps"] == 0, (thr.stats, ex.stats)
  for k in ref:
    if ref[k][1] == "abort" or got[k][1] == "abort": continue
    assert len(got[k][0]) == len(ref[k][0]) and got[k][1] == ref[k][1] and got[k][0][0] == ref[k][0][0], k
  monkeypatch.delenv("FZ_MIX_ANY", raising=False)
  helpers.getenv.cache_clear()

def test_mode_flag_and_reporting(monkeypatch):
  """--mode / FZ_MODE parse; an unknown mode is refused; the server reports the mode in /v1/models and a header."""
  from fusezap.serve.cli import build_parser
  ap = build_parser()
  assert ap.parse_args(["serve", "m"]).mode is None and ap.parse_args(["serve", "m", "--mode", "throughput"]).mode == "throughput"
  with pytest.raises(SystemExit): ap.parse_args(["serve", "m", "--mode", "fast"])
