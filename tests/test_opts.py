"""P2-OPT: kernel Opts (upcast, unroll, local, group-for-reduce) and FZ_BEAM search.

Every optimized kernel is checked against the unoptimized lowering of the same kernel (same inputs) and against numpy.
CPU covers upcast/unroll; METAL and CUDA (when available; CUDA only when FZ_TEST_TIERS names it) also cover local tiling and
group-for-reduce (Metal simd_sum / CUDA warp shuffles).
"""
from __future__ import annotations
import itertools, json, math, os, random
import numpy as np
import pytest
from fusezap import Tensor
from fusezap.device import Buffer, Device
from fusezap.codegen.lower import (Opts, OptError, lower, check_opts, opt_actions, apply_action, heuristic_opts, beam_search,
                                   opts_to_json, opts_from_json, _beam_cache_path, _cache)
from fusezap.dtype import dtypes, to_np
from fusezap.realize import run_schedule
from fusezap.schedule import create_schedule

_TIERS = [t.strip() for t in os.getenv("FZ_TEST_TIERS", "").split(",") if t.strip()]
DEVS = [d for d in ("CPU", "METAL") if Device.available(d)] + \
       (["CUDA"] if "CUDA" in _TIERS and Device.available("CUDA") else [])
pytestmark = pytest.mark.skipif(not DEVS, reason="no compiled device")
rng = np.random.default_rng(0)

def kernel(t: Tensor, dev: str):
  """Realize everything but the last kernel; return that kernel's ScheduleItem (inputs on `dev`)."""
  from fusezap import helpers
  old = os.environ.get("FZ_CUSTOM")
  os.environ["FZ_CUSTOM"] = "0"; helpers.getenv.cache_clear()  # composed kernels only (METAL matches matmul to a custom item)
  try: s = create_schedule([t.lazydata])
  finally:
    if old is None: os.environ.pop("FZ_CUSTOM", None)
    else: os.environ["FZ_CUSTOM"] = old
    helpers.getenv.cache_clear()
  assert s[-1].kind == "kernel"
  run_schedule(s[:-1])
  for b in s[-1].bufs: b.ensure_allocated()
  return s[-1]

def run_opts(si, dev: str, opts: Opts | None) -> np.ndarray:
  d = Device[dev]
  spec = lower(si.ast, d.renderer, opts)
  prg = d.runtime(spec.name, d.compiler.compile_cached(spec.src))
  out = Buffer(dev, si.bufs[0].size, si.bufs[0].dtype)
  bufs = [out.ensure_allocated()] + list(si.bufs[1:])
  gs = tuple(int(s if isinstance(s, int) else s.substitute(si.var_vals)) for s in spec.global_size)
  prg(*[b.handle for b in bufs[:spec.n_bufs]], global_size=gs, local_size=spec.local_size, vals=tuple(si.var_vals[v] for v in spec.vars), wait=True)
  return out.numpy()

def check(t, ref: np.ndarray, dev: str, opts_list, tol=(1e-4, 1e-5)):
  si = kernel(t, dev) if isinstance(t, Tensor) else t   # a Tensor, or its already-scheduled kernel
  base = run_opts(si, dev, Opts())
  np.testing.assert_allclose(base.reshape(ref.shape), ref, rtol=tol[0], atol=tol[1] * max(1.0, float(np.abs(ref).max())))
  ran = 0
  for o in opts_list:
    try: check_opts(o, si.ast, Device[dev].renderer)
    except OptError: continue
    got = run_opts(si, dev, o)
    if base.dtype.kind == "f": np.testing.assert_allclose(got, base, rtol=tol[0], atol=tol[1] * max(1.0, float(np.abs(base).max())), err_msg=str(o), equal_nan=True)
    else: np.testing.assert_array_equal(got, base, err_msg=str(o))
    ran += 1
  return ran

def f32(*s): return rng.standard_normal(s).astype(np.float32)

# ---------------------------------------------------------------- per-opt correctness on matmul / reduce

MM_OPTS = [Opts(upcast=((0, 4),)), Opts(upcast=((2, 4),)), Opts(upcast=((0, 2), (2, 8))), Opts(upcast=((2, 3),)),
           Opts(unroll=((1, 4),)), Opts(unroll=((1, 8),), upcast=((2, 2),)),
           Opts(local=((2, 8),)), Opts(local=((2, 16), (0, 4))), Opts(local=((0, 4), (2, 4))),
           Opts(upcast=((0, 4), (2, 4)), unroll=((1, 4),), local=((2, 8), (0, 8))),
           Opts(group_reduce=32), Opts(group_reduce=32, upcast=((2, 4),), local=((0, 2),)), Opts(group_reduce=32, unroll=((1, 2),))]

@pytest.mark.parametrize("dev", DEVS)
def test_matmul_opts(dev):
  a, b = f32(32, 64), f32(64, 96)
  n = check(Tensor(a, device=dev) @ Tensor(b, device=dev), a @ b, dev, MM_OPTS)
  assert n >= (13 if dev in ("METAL", "CUDA") else 6)

RED_OPTS = [Opts(unroll=((1, 2),)), Opts(unroll=((1, 8),)), Opts(upcast=((0, 4),)), Opts(upcast=((0, 3),), unroll=((1, 4),)),
            Opts(local=((0, 8),)), Opts(group_reduce=32), Opts(group_reduce=32, unroll=((1, 4),)),
            Opts(group_reduce=32, local=((0, 4),), upcast=((0, 2),))]

@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("op", ["sum", "max", "mean"])
def test_reduce_last_axis(dev, op):
  a = f32(24, 256)
  t = getattr(Tensor(a, device=dev), op)(axis=-1)
  assert check(t, getattr(a, op)(axis=-1), dev, RED_OPTS) >= 3

@pytest.mark.parametrize("dev", DEVS)
def test_reduce_first_axis(dev):
  a = f32(128, 24)
  opts = [Opts(unroll=((0, 4),)), Opts(upcast=((1, 4),)), Opts(local=((1, 8),)), Opts(group_reduce=32), Opts(group_reduce=32, unroll=((0, 4),))]
  check(Tensor(a, device=dev).sum(axis=0), a.sum(axis=0), dev, opts)

@pytest.mark.parametrize("dev", DEVS)
def test_full_reduce_and_multi_axis(dev):
  a = f32(8, 64, 32)
  t = Tensor(a, device=dev).sum(axis=(0, 2))  # two reduce axes; group goes on the innermost
  opts = [Opts(unroll=((0, 2), (2, 4))), Opts(upcast=((1, 4),)), Opts(group_reduce=32), Opts(group_reduce=32, unroll=((0, 8),), local=((1, 4),))]
  check(t, a.sum(axis=(0, 2)), dev, opts)
  b = f32(4096)
  check(Tensor(b, device=dev).sum(), b.sum(), dev, [Opts(unroll=((0, 8),)), Opts(group_reduce=32), Opts(group_reduce=32, unroll=((0, 4),))])

@pytest.mark.parametrize("dev", DEVS)
def test_max_nan_propagates(dev):
  a = f32(8, 128); a[3, 77] = np.nan; a[5, 0] = np.nan
  t = Tensor(a, device=dev).max(axis=-1)
  check(t, a.max(axis=-1), dev, [Opts(unroll=((1, 4),)), Opts(group_reduce=32), Opts(group_reduce=32, unroll=((1, 2),))])

@pytest.mark.parametrize("dev", DEVS)
def test_epilogue_softmax_like(dev):
  a = f32(16, 64)
  t = (Tensor(a, device=dev) - Tensor(a, device=dev).max(axis=-1, keepdim=True)).exp()
  e = np.exp(a - a.max(axis=-1, keepdims=True))
  check(t, e, dev, [Opts(upcast=((1, 4),)), Opts(upcast=((0, 2), (1, 2))), Opts(local=((1, 16),))], tol=(1e-5, 1e-6))
  s = Tensor(a, device=dev).exp().sum(axis=-1) * 0.5 + 1.0  # reduce + epilogue in one kernel
  check(s, np.exp(a).sum(-1) * 0.5 + 1.0, dev, RED_OPTS)

@pytest.mark.parametrize("dev", DEVS)
def test_broadcast_reduce_not_cse(dev):
  # the reduced value does not depend on the reduce index: unrolled acc updates must not be CSE'd together
  a = f32(8, 1)
  t = Tensor(a, device=dev).expand(8, 64).sum(axis=-1)
  check(t, np.broadcast_to(a, (8, 64)).sum(-1), dev, [Opts(unroll=((1, 4),)), Opts(unroll=((1, 8),), upcast=((0, 2),)), Opts(group_reduce=32, unroll=((1, 2),))])
  c = Tensor.full((4, 32), 2.0, device=dev).sum(axis=-1)
  check(c, np.full((4,), 64.0, np.float32), dev, [Opts(unroll=((1, 4),)), Opts(group_reduce=32)])

@pytest.mark.parametrize("dev", DEVS)
def test_masked_views(dev):
  a = f32(12, 30)
  t = Tensor(a, device=dev).pad(((2, 2), (1, 1))).sum(axis=-1)  # (16, 32): padded loads under the opts
  ref = np.pad(a, ((2, 2), (1, 1))).sum(-1)
  check(t, ref, dev, [Opts(upcast=((0, 4),)), Opts(unroll=((1, 8),)), Opts(group_reduce=32), Opts(local=((0, 4),), unroll=((1, 2),))])
  e = Tensor(a, device=dev).pad(((2, 2), (1, 1))).flip(0) * 2.0  # elementwise, masked store side
  check(e, np.pad(a, ((2, 2), (1, 1)))[::-1] * 2.0, dev, [Opts(upcast=((1, 4),)), Opts(upcast=((0, 2), (1, 8))), Opts(local=((1, 8), (0, 2)))], tol=(1e-5, 1e-6))

@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("dt", ["float16", "int32"])
def test_dtypes(dev, dt):
  a = (rng.standard_normal((16, 128)) * (1 if dt == "float16" else 50)).astype(dt)
  t = Tensor(a, device=dev).sum(axis=-1)
  ref = a.astype(np.float32).sum(-1).astype(dt) if dt == "float16" else a.sum(-1, dtype=np.int32)
  check(t, ref, dev, RED_OPTS + [Opts(unroll=((1, 4),), upcast=((0, 4),))], tol=(1e-2, 1e-2))

@pytest.mark.parametrize("dev", DEVS)
def test_random_opts_fuzz(dev):
  """Random multi-step opt sequences from opt_actions (what BEAM explores) all match the unoptimized kernel."""
  r = random.Random(1)
  a, b = f32(16, 32), f32(32, 24)
  for t, ref in ((Tensor(a, device=dev) @ Tensor(b, device=dev), a @ b), (Tensor(a, device=dev).sum(axis=-1), a.sum(-1))):
    si = kernel(t, dev)
    acts, opts = opt_actions(si.ast, Device[dev].renderer), []
    for _ in range(25):
      o = Opts()
      for _ in range(r.randint(1, 4)): o = apply_action(o, r.choice(acts))
      opts.append(o)
    assert check(si, ref, dev, opts) >= 5

# ---------------------------------------------------------------- validity

def test_check_opts_rejects():
  a, b = Tensor(f32(32, 64), device=DEVS[0]), Tensor(f32(64, 48), device=DEVS[0])
  si = kernel(a @ b, DEVS[0])
  from fusezap.codegen.cstyle import CRenderer
  cr = CRenderer()
  for o in (Opts(upcast=((1, 4),)),          # reduce axis cannot be upcast
            Opts(unroll=((0, 4),)),          # output axis cannot be unrolled
            Opts(upcast=((0, 5),)),          # not a divisor
            Opts(upcast=((0, 4), (0, 2))),   # repeated axis
            Opts(upcast=((0, 1),)),          # amount < 2
            Opts(local=((0, 4),)),           # CPU has no local dims
            Opts(group_reduce=32)):          # group_reduce is METAL / CUDA only
    with pytest.raises(OptError): check_opts(o, si.ast, cr)
    with pytest.raises(OptError): lower(si.ast, cr, o)
  check_opts(Opts(upcast=((0, 4),), unroll=((1, 8),)), si.ast, cr)
  for gd in [d for d in ("METAL", "CUDA") if d in DEVS]:
    mr = Device[gd].renderer
    for o in (Opts(group_reduce=16), Opts(local=((0, 32), (2, 48))), Opts(local=((0, 2),), upcast=((0, 32),))):
      with pytest.raises(OptError): check_opts(o, si.ast, mr)

def test_symbolic_axes_not_split():
  from fusezap.shape.symbolic import Var
  from fusezap.ops import ReduceOps
  dev, n = DEVS[0], Var("n", 1, 64)
  a = f32(64, 32)
  e = Tensor(a, device=dev).lazydata.shrink(((0, n), (0, 32))).r(ReduceOps.SUM, (1,)).contiguous()
  sched = create_schedule([e], var_vals={n: 7})
  run_schedule(sched[:-1])
  si, r = sched[-1], Device[dev].renderer
  with pytest.raises(OptError): check_opts(Opts(upcast=((0, 2),)), si.ast, r)   # symbolic output axis
  assert not [x for x in opt_actions(si.ast, r) if x[0] in ("upcast", "local")]
  for b in si.bufs: b.ensure_allocated()
  got = run_opts(si, dev, Opts(unroll=((1, 4),)))   # the static reduce axis may still be unrolled
  np.testing.assert_allclose(got[:7], a[:7].sum(-1), rtol=1e-4, atol=1e-4)

def test_opts_json_roundtrip():
  o = Opts(upcast=((0, 4), (2, 2)), unroll=((1, 8),), local=((2, 16),), group_reduce=32)
  assert opts_from_json(opts_to_json(o)) == o and opts_from_json(opts_to_json(Opts())) == Opts()

@pytest.mark.parametrize("dev", DEVS)
def test_heuristic_opts_valid_and_correct(dev):
  a, b = f32(64, 128), f32(128, 64)
  for t, ref in ((Tensor(a, device=dev) @ Tensor(b, device=dev), a @ b), (Tensor(a, device=dev).sum(axis=-1), a.sum(-1)),
                 (Tensor(f32(4096), device=dev).sum(), None)):
    si = kernel(t, dev)
    o = heuristic_opts(si.ast, Device[dev].renderer)
    check_opts(o, si.ast, Device[dev].renderer)
    if ref is not None: check(si, ref, dev, [o])
  if dev in ("METAL", "CUDA"): assert heuristic_opts(kernel(Tensor(a, device=dev) @ Tensor(b, device=dev), dev).ast, Device[dev].renderer) != Opts()

# ---------------------------------------------------------------- FZ_BEAM

@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
  import fusezap.helpers as h
  monkeypatch.setattr(h, "CACHE_DIR", tmp_path)
  monkeypatch.delenv("FZ_NOCACHE", raising=False)
  return tmp_path

@pytest.mark.parametrize("dev", DEVS)
def test_beam_search_and_cache(dev, tmp_cache, monkeypatch):
  a, b = f32(64, 64), f32(64, 64)
  si = kernel(Tensor(a, device=dev) @ Tensor(b, device=dev), dev)
  before = si.bufs[0].numpy().copy()
  d = Device[dev]
  o = beam_search(d, si.ast, list(si.bufs), si.var_vals, 2, max_rounds=2)
  check_opts(o, si.ast, d.renderer)
  np.testing.assert_array_equal(si.bufs[0].numpy(), before)   # timing used scratch outputs
  path = _beam_cache_path(si.ast, d, 2)
  assert path.exists() and opts_from_json(path.read_text()) == o
  assert path.is_relative_to(tmp_cache / "beam")
  import fusezap.codegen.lower as L
  monkeypatch.setattr(L, "time_opts", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache miss")))
  assert beam_search(d, si.ast, list(si.bufs), si.var_vals, 2) == o   # served from the cache, no timing
  check(Tensor(a, device=dev) @ Tensor(b, device=dev), a @ b, dev, [o])

@pytest.mark.parametrize("dev", DEVS)
def test_beam_cache_ignores_bad_entry(dev, tmp_cache):
  a = f32(16, 64)
  si = kernel(Tensor(a, device=dev).sum(axis=-1), dev)
  d = Device[dev]
  path = _beam_cache_path(si.ast, d, 1)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(opts_to_json(Opts(upcast=((1, 4),))))   # invalid for this kernel -> searched again
  o = beam_search(d, si.ast, list(si.bufs), si.var_vals, 1, max_rounds=1)
  check_opts(o, si.ast, d.renderer)
  assert opts_from_json(path.read_text()) == o

@pytest.mark.parametrize("dev", DEVS)
def test_fz_beam_end_to_end(dev, tmp_cache, monkeypatch):
  monkeypatch.setenv("FZ_BEAM", "1")
  _cache.clear()
  a, b = f32(32, 48), f32(48, 32)
  out = (Tensor(a, device=dev) @ Tensor(b, device=dev)).relu().sum(axis=0).numpy()
  np.testing.assert_allclose(out, np.maximum(a @ b, 0).sum(0), rtol=1e-4, atol=1e-4)
  assert any((tmp_cache / "beam").iterdir())
  monkeypatch.setenv("FZ_BEAM", "0"); monkeypatch.setenv("FZ_OPT", "1")
  out = (Tensor(a, device=dev) @ Tensor(b, device=dev)).numpy()
  np.testing.assert_allclose(out, a @ b, rtol=1e-4, atol=1e-4)
  _cache.clear()

def test_cuda_group_reduce_renders_warp_shuffles():
  """B2-INT: group_reduce is allowed on CUDA (warp shuffles), and the heuristic picks it for a decode RMSNorm-like row
  reduce (1 x 3584): on one thread per row that kernel was 69% of a 7B decode step on H100 (docs/status/P4-BENCH.md).
  Renders only: no GPU needed."""
  from fusezap.codegen.cuda import CUDARenderer
  cr = CUDARenderer("sm_90")
  si = kernel((Tensor(f32(1, 3584), device=DEVS[0]) ** 2).mean(axis=-1), DEVS[0])
  o = heuristic_opts(si.ast, cr)
  assert o.group_reduce == 32, o
  src = lower(si.ast, cr, o).src
  assert "__shfl_xor_sync" in src and "simd_sum(" in src
  assert any(a[0] == "group_reduce" for a in opt_actions(si.ast, cr))
  check_opts(Opts(group_reduce=32, unroll=((1, 4),)), si.ast, cr)
  with pytest.raises(OptError): check_opts(Opts(group_reduce=16), si.ast, cr)
