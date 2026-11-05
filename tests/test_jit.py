"""Kernel replay (fusezap/jit.py, SPEC §11.1): replayed outputs == eager outputs, symbolic vars, buffer handling."""
from __future__ import annotations
import os, sys
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fusezap import Tensor, Device
from fusezap.shape.symbolic import Var
from fusezap.jit import jit, bind, bound
import fusezap.schedule

DEVS = [d for d in ("NPY", "CPU", "METAL") if Device.available(d)]

@pytest.fixture(params=DEVS)
def dev(request, monkeypatch):
  from fusezap import helpers
  monkeypatch.delenv("FZ_INTERP", raising=False)
  helpers.getenv.cache_clear()
  yield request.param
  helpers.getenv.cache_clear()

def _t(a, dev): return Tensor(np.asarray(a), device=dev).realize()

class _Counter:
  def __init__(self, fn): self.fn, self.n = fn, 0
  def __call__(self, *a, **kw):
    self.n += 1
    return self.fn(*a, **kw)

@pytest.fixture
def nsched(monkeypatch):
  """Counts create_schedule calls: a replay must not schedule."""
  cnt, orig = [0], fusezap.schedule.create_schedule
  def counted(*a, **kw):
    cnt[0] += 1
    return orig(*a, **kw)
  monkeypatch.setattr(fusezap.schedule, "create_schedule", counted)
  return cnt

def test_elementwise_replay(dev, nsched):
  rng = np.random.default_rng(0)
  w = _t(rng.standard_normal((4, 8), dtype=np.float32), dev)
  f = _Counter(lambda x, y: ((x * y + 1.0) @ w.T).relu())
  jf = jit(f)
  for i in range(6):
    x, y = rng.standard_normal((3, 8), dtype=np.float32), rng.standard_normal((3, 8), dtype=np.float32)
    tx, ty, n0 = _t(x, dev), _t(y, dev), nsched[0]
    out = jf(tx, ty)
    assert i < 2 or nsched[0] == n0
    ref = torch.relu((torch.tensor(x) * torch.tensor(y) + 1.0) @ torch.tensor(w.numpy()).T).numpy()
    np.testing.assert_allclose(out.numpy(), ref, rtol=1e-4, atol=1e-5)
  assert f.n == 2, "fn runs only on the eager call and the capture call"

def test_outputs_are_fresh_and_multiple(dev):
  jf = jit(lambda x: (x + 1, {"neg": -x, "same": x * 1}))
  outs = []
  for i in range(5):
    a = np.full((5,), float(i), np.float32)
    outs.append((a, jf(_t(a, dev))))
  for a, (p, d) in outs:  # earlier outputs stay valid after later calls
    np.testing.assert_array_equal(p.numpy(), a + 1)
    np.testing.assert_array_equal(d["neg"].numpy(), -a)
    np.testing.assert_array_equal(d["same"].numpy(), a)

def test_reshaped_output_needs_no_copy(dev):
  """A returned reshape of a base the call computes is that base's buffer: no copy kernel is captured (N5), and every
  replay still returns a fresh buffer of the reshaped shape."""
  jf = jit(lambda x: (x * 2 + 1).reshape(2, 6))
  outs = []
  for i in range(4):
    a = np.arange(12, dtype=np.float32).reshape(3, 4) + i
    outs.append((a, jf(_t(a, dev))))
  assert len(jf.captured.items) == 1
  for a, o in outs:
    assert o.shape == (2, 6)
    np.testing.assert_array_equal(o.numpy(), (a * 2 + 1).reshape(2, 6))

def test_non_contiguous_and_kwargs(dev):
  jf = jit(lambda x, *, y: x.sum(axis=1) * y)
  for i in range(4):
    a = np.arange(12, dtype=np.float32).reshape(3, 4) + i
    b = np.float32(i + 1) * np.ones(4, np.float32)
    x = _t(a, dev).T                           # a view: the jit makes it a contiguous input
    out = jf(x, y=_t(b, dev).reshape(4))
    np.testing.assert_allclose(out.numpy(), a.T.sum(1) * b, rtol=1e-5)

def test_symbolic_input_length(dev, nsched):
  n = Var("n", 1, 16)
  f = _Counter(lambda x, n: (x.shrink(((0, n),)) * 2.0).sum().reshape(1) + x[:1])
  jf = jit(f)
  a = np.random.default_rng(1).standard_normal(16, dtype=np.float32)
  for i, k in enumerate([3, 7, 1, 16, 9, 2]):
    x, n0 = _t(a, dev), nsched[0]
    out = jf(x, bind(n, k))
    assert i < 2 or nsched[0] == n0
    np.testing.assert_allclose(out.numpy(), [2 * a[:k].sum() + a[0]], rtol=1e-4, atol=1e-5)
  assert f.n == 2

def test_symbolic_output_shape(dev):
  n = Var("n", 1, 10)
  jf = jit(lambda x, n: x.shrink(((0, n), None)) + 1.0)
  a = np.arange(30, dtype=np.float32).reshape(10, 3)
  for k in [4, 4, 2, 10, 7]:
    out = jf(_t(a, dev), n, var_vals={n: k})
    assert out.shape == (k, 3)
    np.testing.assert_array_equal(out.numpy(), a[:k] + 1)

def test_bound_eager(dev):
  n = Var("n", 1, 8)
  a = np.arange(8, dtype=np.float32)
  with bound({n: 5}):
    s = _t(a, dev).shrink(((0, n),)).sum().numpy()
  assert s == a[:5].sum()

def test_mismatch_errors(dev):
  n = Var("n", 1, 8)
  jf = jit(lambda x, n: x.shrink(((0, n),)).sum())
  for k in (1, 2, 3): jf(_t(np.ones(8, np.float32), dev), bind(n, k))
  with pytest.raises(AssertionError): jf(_t(np.ones(7, np.float32), dev), bind(n, 2))
  with pytest.raises(AssertionError): jf(_t(np.ones(8, np.int32), dev), bind(n, 2))
  with pytest.raises(AssertionError): jf(_t(np.ones(8, np.float32), dev))
  with pytest.raises(AssertionError): bind(n, 9)

def test_state_through_args(dev):
  """A running accumulator passed in and returned (like a KV cache) ping-pongs between fresh buffers."""
  jf = jit(lambda acc, x: acc + x)
  acc, tot = _t(np.zeros(4, np.float32), dev), np.zeros(4, np.float32)
  for i in range(6):
    x = np.full(4, i + 0.5, np.float32)
    acc, tot = jf(acc, _t(x, dev)), tot + x
  np.testing.assert_array_equal(acc.numpy(), tot)

def test_gpt_decode_symbolic(dev, nsched):
  """GPT decode with a KV cache: jit replay == eager (same symbolic step) == full forward on the prefix."""
  from bench.bench_jit import decode_step, make_model
  T = 10
  m, caches0 = make_model(dev, T=T)
  pos = Var("pos", 0, T - 1)
  toks = list(np.random.default_rng(0).integers(0, 20, T).astype(np.int32))
  f = _Counter(lambda tok, caches, p: decode_step(m, tok, caches, p))
  jf = jit(f)
  ce, cj = caches0, caches0
  for i, t in enumerate(toks):
    tok = _t(np.array([[t]], np.int32), dev)
    with bound({pos: i}):
      le, ce = decode_step(m, tok, ce, pos)
      for c in ce: c.realize()   # realize under this step's binding: the lazy graph refers to the Var, not its value
      le = le.numpy()
    n0 = nsched[0]
    lj, cj = jf(tok, cj, bind(pos, i))
    assert i < 2 or nsched[0] == n0
    np.testing.assert_allclose(lj.numpy(), le, rtol=1e-5, atol=1e-6, err_msg=f"step {i}")
    if i in (0, 4, T - 1):
      full = m(_t(np.array([toks[:i + 1]], np.int32), dev)).numpy()[:, -1]
      np.testing.assert_allclose(lj.numpy(), full, rtol=1e-4, atol=1e-5, err_msg=f"step {i} vs full forward")
  assert f.n == 2

def test_replay_without_cached_programs(dev, monkeypatch):
  """If the lowering cache has no entry for a kernel, replay falls back to Device.run with the new buffers."""
  import fusezap.jit as J
  monkeypatch.setattr(J, "_programs", lambda: {})
  n = Var("n", 1, 6)
  jf = jit(lambda x, n: (x.shrink(((0, n),)).exp().sum().reshape(1), Tensor.arange(6, device=x.device) * x))
  a = np.linspace(-1, 1, 6).astype(np.float32)
  for k in (2, 5, 3, 6):
    s, p = jf(_t(a, dev), bind(n, k))
    np.testing.assert_allclose(s.numpy(), [np.exp(a[:k]).sum()], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(p.numpy(), np.arange(6) * a, rtol=1e-5, atol=1e-6)

def _other(dev): return "CPU" if dev == "NPY" else "NPY"

def test_symbolic_cross_device_copy(dev, nsched):
  """A COPY of a symbolic view (the scheduler makes a contiguous source) and a COPY into a persistent intermediate
  move the current var value's bytes, not the capture's or vmax's."""
  n, o = Var("n", 0, 8), _other(dev)
  a = np.arange(8, dtype=np.float32)
  j1 = jit(lambda x, n: x.shrink(((0, n),)).to(o))
  j2 = jit(lambda x, n: ((x.shrink(((0, n),)) + 1).to(o) * 2).to(dev))
  for i, k in enumerate([2, 2, 3, 8, 5, 1, 0, 7]):
    x1, x2 = _t(a, dev), _t(a, dev)
    n0 = nsched[0]
    r1, r2 = j1(x1, bind(n, k)), j2(x2, bind(n, k))
    assert i < 2 or nsched[0] == n0
    assert r1.device == o and r2.device == dev and r1.shape == r2.shape == (k,)
    np.testing.assert_array_equal(r1.numpy(), a[:k])
    np.testing.assert_array_equal(r2.numpy(), (a[:k] + 1) * 2)

def test_nested_jit(dev):
  """A jitted fn called inside another jit's capture runs inline, so the outer replay recomputes it."""
  inner = jit(lambda x: x * 2)
  for _ in range(3): inner(_t(np.ones(3, np.float32), dev))
  outer = jit(lambda x: inner(x) + 1)
  assert [float(outer(_t(np.full(3, float(i), np.float32), dev)).numpy()[0]) for i in range(5)] == [1, 3, 5, 7, 9]
  # vars: the outer one passes through to the inner jit; a constant bind inside is baked at its captured value
  n, m = Var("n", 1, 8), Var("m", 1, 8)
  sub = jit(lambda x, n, m: x.shrink(((0, n),)).sum().reshape(1) + x.shrink(((0, m),)).sum().reshape(1))
  top = jit(lambda x, n: sub(x, n, bind(m, 2)) * 10)
  a = np.arange(8, dtype=np.float32)
  for k in (3, 4, 1, 8, 6):
    np.testing.assert_allclose(top(_t(a, dev), bind(n, k)).numpy(), [10 * (a[:k].sum() + a[:2].sum())])
  # an inner output with a symbolic shape stays symbolic inside the outer capture
  sym = jit(lambda x, n: x.shrink(((0, n),)) + 1.0)
  for k in (2, 3, 4): sym(_t(a, dev), bind(n, k))
  tot = jit(lambda x, n: sym(x, n).sum().reshape(1))
  for k in (3, 4, 1, 8, 6): np.testing.assert_allclose(tot(_t(a, dev), bind(n, k)).numpy(), [(a[:k] + 1).sum()])

def test_capture_at_empty_size(dev):
  """A capture where a symbolic buffer is empty (its kernels are skipped) is not kept: the next call captures again."""
  n = Var("n", 0, 8)
  f = _Counter(lambda x, n: (x.shrink(((0, n),)) + 1.0, (x.shrink(((0, n),)) * 3.0).sum().reshape(1)))
  jf = jit(f)
  a = np.arange(8, dtype=np.float32)
  for k in (0, 0, 3, 0, 5, 8, 0, 2):
    p, s = jf(_t(a, dev), bind(n, k))
    assert p.shape == (k,)
    np.testing.assert_array_equal(p.numpy(), a[:k] + 1)
    np.testing.assert_allclose(s.numpy(), [3 * a[:k].sum()])
  assert f.n == 3

def test_plan_shares_intermediates(dev):
  """Jit(plan=True) (CUDA prefill, run-2 review fix): intermediates whose live ranges do not overlap share one allocation,
  and replays return exactly the unplanned jit's results (also a constant-only item, which runs once, and the outputs)."""
  from fusezap.jit import Jit
  rng = np.random.default_rng(3)
  w = _t(rng.standard_normal((16, 16), dtype=np.float32), dev)
  def f(x):
    h = x
    for i in range(6): h = ((h @ (w * float(i + 1)).contiguous()).relu() + x).contiguous()
    return h.sum(axis=1), h * 2
  plain, planned = Jit(f), Jit(f, plan=True)
  for i in range(5):
    x = rng.standard_normal((4, 16), dtype=np.float32)
    for a, b in zip(plain(_t(x, dev)), planned(_t(x, dev))): np.testing.assert_array_equal(a.numpy(), b.numpy())
  c = planned.captured
  mid = [b for b in c.persist if b is not None]
  assert c.arenas and len(c.arenas) < len(mid) and sum(a.nbytes for a in c.arenas) < sum(b.nbytes for b in mid)
  assert not hasattr(plain.captured, "arenas")
