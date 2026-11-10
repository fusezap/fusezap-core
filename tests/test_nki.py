"""C-N1 gate: the NKI renderer (fusezap/codegen/nki.py) + NKISHIM (fusezap/device/nkishim.py) on the Mac.

1. Op suite subset: elementwise and reduce cases of tests/test_ops.py run with every kernel rendered to NKI source and
   executed by the shim (strict: a kernel the renderer cannot lower fails the case). Results are checked against torch
   by the op suite's own assertions. The subset is every device test minus EXCLUDED below (docs/status/C-N1.md).
   Each case runs twice: with the default tile limits and with tiny ones (pmax=2, fmax=3), so the small op-suite shapes
   still exercise tile loops, ragged tails, chunked accumulation and PAD sub-box DMAs at dynamic tile starts.
2. Golden sources: a few rendered kernels are snapshotted verbatim, and rendering is deterministic.
3. Shim semantics: the hardware behaviours the renderer relies on or works around.
With FZ_NKI_GATE=NKISIM (Linux with .venv-nki, NKI-N2) the gate's kernels run in the real nki 0.6 simulator instead of
the shim; add FZ_NKISIM_SHIM_CHECK=<file> to log every kernel where the shim disagrees with the simulator.
"""
from __future__ import annotations
import importlib, itertools, os, re
import numpy as np
import pytest
from fusezap import Tensor, dtypes
from fusezap.codegen.nki import NKIRenderer, NKIUnsupported
from fusezap.device import nkishim
from fusezap.dtype import to_np
from fusezap.device.nkishim import nl, nisa

test_ops = importlib.import_module("test_ops")

# ---------------------------------------------------------------- op suite through renderer + shim

# Every tests/test_ops.py function that takes a device runs, except these (reason). Elementwise and reduce kernels are
# the subject; composed ops (matmul = MUL+SUM, attention, cross-entropy) and movement copies are elementwise/reduce
# kernels too, so they stay in. The gate list is printed in docs/status/C-N1.md.
NO_DIV = "IDIV/MOD are not lowered: neuronx-cc 2.27 rejects divide/fmod/mod on inf2 (NKI-N2, scripts/nki_op_matrix.py)"
EXCLUDED = {
  "test_int_floordiv_mod": NO_DIV,
  "test_pow": f"pow lowers to BinaryOps.MOD; {NO_DIV}",
  "test_operators": f"includes y ** x, which lowers to BinaryOps.MOD; {NO_DIV}",
  "test_movement_backward": "multi-view ShapeTrackers (reshape of a padded/permuted view) need the gather kernel (N2+)",
  "test_repeat": "multi-view ShapeTrackers (repeat = expand + reshape) need the gather kernel (N2+)",
}
EXCLUDED_CASES = [  # (function, kwargs predicate, reason)
  ("test_binary_backward", lambda kw: kw["name"] == "pow", f"pow lowers to BinaryOps.MOD; {NO_DIV}"),
]
ELEMENTWISE_REDUCE = ["test_cast", "test_cast_helpers", "test_bitcast", "test_unary", "test_unary_noncontig", "test_unary_int",
  "test_logical_not", "test_binary_broadcast", "test_binary_int", "test_div_int_promotes", "test_bool_arith", "test_binary_scalar",
  "test_compare", "test_clip", "test_where_masked_fill", "test_binary_noncontig", "test_nan_inf", "test_size_one_broadcast_chain",
  "test_zero_size", "test_reduce", "test_reduce_scalar_input", "test_reduce_int_bool", "test_sum_dtype", "test_reduce_noncontig",
  "test_reduce_large_f16_accumulates_f32", "test_var_std", "test_var_f16", "test_argmax_argmin", "test_argmax_1d", "test_logsumexp",
  "test_logsumexp_stable", "test_softmax", "test_softmax_default_and_stable", "test_cumsum", "test_layernorm", "test_bf16_binary",
  "test_bf16_unary", "test_bf16_reduce", "test_bf16_softmax", "test_bf16_roundtrip_cast"]   # the core of the gate (docs)

def _gate_functions() -> list[str]:
  import inspect
  return [n for n in dir(test_ops) if n.startswith("test_") and n not in EXCLUDED
          and "device" in inspect.signature(getattr(test_ops, n)).parameters]

def _ident(v) -> str:
  if callable(v): return getattr(v, "__name__", "fn")
  return str(v).replace(" ", "")

def _grid(fn) -> list[tuple[str, dict]]:
  axes = []
  for m in reversed([m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]):
    names = [n.strip() for n in m.args[0].split(",")] if isinstance(m.args[0], str) else list(m.args[0])
    rows = []
    for v in m.args[1]:
      v = tuple(v) if len(names) > 1 else (v,)
      rows.append(("-".join(_ident(x) for n, x in zip(names, v) if n != "tf"), dict(zip(names, v))))
    axes.append(rows)
  out = []
  for combo in itertools.product(*axes):
    out.append(("-".join(i for i, _ in combo), {k: v for _, kw in combo for k, v in kw.items()}))
  return out or [("", {})]

CASES, IDS = [], []
for _fname in _gate_functions():
  for _cid, _kw in _grid(getattr(test_ops, _fname)):
    reason = next((r for f, pred, r in EXCLUDED_CASES if f == _fname and pred(_kw)), None)
    CASES.append(pytest.param(_fname, _kw, marks=pytest.mark.skip(reason=reason)) if reason else (_fname, _kw))
    IDS.append(f"{_fname}[{_cid}]" if _cid else _fname)

TILES = {"default": NKIRenderer(), "tiny": NKIRenderer(pmax=2, fmax=3)}   # tiny: ragged tails, loops and chunked reduces everywhere

def _route_to_nki(monkeypatch, renderer: NKIRenderer) -> dict:
  """Make NPY the default device and run its kind="kernel" items through NKIRenderer + NKISHIM; returns a counter."""
  from fusezap import helpers
  from fusezap.device import Device
  helpers.getenv.cache_clear()
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.setattr(Device, "DEFAULT", "NPY", raising=False)
  dev = Device["NPY"]
  stats = {"kernels": 0}
  def run(si):
    assert si.kind == "kernel", f"NKI gate: unexpected {si.kind} item"
    stats["kernels"] += 1
    if os.getenv("FZ_NKI_GATE") == "NKISIM": return _run_sim(si, renderer)   # NKI-N2: the real nki 0.6 simulator (Linux, .venv-nki)
    return nkishim.run_item(si, renderer)
  monkeypatch.setattr(dev, "run", run)
  monkeypatch.setattr(type(dev), "custom_kernels", property(lambda self: {}))   # composed matmul, no reference kernels
  return stats

def _run_sim(si, renderer: NKIRenderer) -> float:
  from fusezap.device.neuron import NKISimProgram
  from fusezap.dtype import to_np
  prog = renderer.render(si.ast, si.var_vals)
  arrays = [b.ensure_allocated().handle[:b.nbytes].view(to_np(b.dtype)) if b.size else np.zeros(1, to_np(b.dtype)) for b in si.bufs]
  arrays += [np.zeros(1, np.uint8)] * (prog.n_bufs - len(arrays))
  return NKISimProgram(prog, prog.src.encode())(*arrays[:prog.n_bufs], wait=True) or 0.0

@pytest.fixture(params=list(TILES))
def nki_tier(request, monkeypatch):
  yield _route_to_nki(monkeypatch, TILES[request.param])
  from fusezap import helpers
  helpers.getenv.cache_clear()

@pytest.fixture
def nki(monkeypatch):
  """Default tile limits; tensors must be created inside the test (after the default device is switched)."""
  yield _route_to_nki(monkeypatch, TILES["default"])

@pytest.mark.parametrize("fname,kw", CASES, ids=IDS)
def test_op_suite_on_nki(nki_tier, assert_close, fname, kw):
  import inspect
  fn = getattr(test_ops, fname)
  params = inspect.signature(fn).parameters
  args = {k: v for k, v in kw.items() if k in params}
  if "device" in params: args["device"] = "NKISHIM"
  if "assert_close" in params: args["assert_close"] = assert_close
  if "rng" in params: args["rng"] = np.random.default_rng(1337)
  fn(**args)
  if nki_tier["kernels"] == 0: pytest.skip("schedules no kernel (empty output or pure view)")

# ---------------------------------------------------------------- golden sources

def _z(*shape) -> Tensor: return Tensor(np.zeros(shape, np.float32))

def _kernels(t: Tensor, var_vals: dict | None = None) -> list:
  from fusezap.schedule import create_schedule
  return [si for si in create_schedule([t.lazydata], var_vals) if si.kind == "kernel"]

def _srcs(t: Tensor, renderer: NKIRenderer | None = None) -> list[str]:
  r = renderer or NKIRenderer()
  return [r.render(si.ast, si.var_vals).src for si in _kernels(t)]

SNAPSHOT_GRAPHS = {
  "broadcast_scale_add": lambda: _z(300, 1000) + _z(1000) * 2,     # P/F tiles with ragged tails, broadcast DMA, tensor_scalar
  "row_max_chunked": lambda: _z(300, 1000).max(axis=1),             # F in two chunks accumulated into a (P, 1) tile
  "mean_epilogue": lambda: _z(4, 6).mean(axis=1),                    # one-chunk reduce + an epilogue op
  "pad_where_sin_cast": lambda: (_z(3, 4).pad(((1, 1), (2, 0)), value=2.5) > 0).where(_z(5, 6).sin(), -1.0).cast(dtypes.int32),
  "split_flat_recip": lambda: _z(4096).reciprocal(),                 # 1-D split onto 128 partitions; one VectorE instruction
}

SNAPSHOTS = {
  "broadcast_scale_add": """\
# fusezap NKI kernel (nki 0.6 API), rendered by fusezap.codegen.nki from F=(300, 1000) R=(300, 1000)
# kernel dims [300, 1000] (nn): P d0[300] F d1[1000] loops -
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def E_300_1000(data0, data1, data2):
  for p0 in range(0, 300, 128):
    pn = min(128, 300 - p0)
    for f0 in range(0, 1000, 512):
      fn = min(512, 1000 - f0)
      t0 = nl.ndarray((pn, fn), dtype=nl.float32, buffer=nl.sbuf)
      nisa.dma_copy(dst=t0, src=data1.ap(pattern=[[1000, pn], [1, fn]], offset=1000*p0 + f0))
      t1 = nl.ndarray((pn, fn), dtype=nl.float32, buffer=nl.sbuf)
      nisa.dma_copy(dst=t1, src=data2.ap(pattern=[[0, pn], [1, fn]], offset=f0))
      t2 = nl.ndarray((pn, fn), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_scalar(dst=t2, data=t1, op0=nl.multiply, operand0=2.0)
      t3 = nl.ndarray((pn, fn), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_tensor(dst=t3, data1=t0, data2=t2, op=nl.add)
      nisa.dma_copy(dst=data0.ap(pattern=[[1000, pn], [1, fn]], offset=1000*p0 + f0), src=t3)
  return data0
""",
  "row_max_chunked": """\
# fusezap NKI kernel (nki 0.6 API), rendered by fusezap.codegen.nki from F=(300, 1000) R=(300, 1)
# kernel dims [300, 1000] (nr): P d0[300] F d1[1000] loops -
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def r_300_1000(data0, data1):
  for p0 in range(0, 300, 128):
    pn = min(128, 300 - p0)
    t0 = nl.ndarray((pn, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=t0, value=float('-inf'))
    for f0 in range(0, 1000, 512):
      fn = min(512, 1000 - f0)
      t1 = nl.ndarray((pn, fn), dtype=nl.float32, buffer=nl.sbuf)
      nisa.dma_copy(dst=t1, src=data1.ap(pattern=[[1000, pn], [1, fn]], offset=1000*p0 + f0))
      t2 = nl.ndarray((pn, 1), dtype=nl.float32, buffer=nl.sbuf)
      nisa.tensor_reduce(dst=t2, op=nl.maximum, data=t1, axis=1, keepdims=True)
      nisa.tensor_tensor(dst=t0, data1=t0, data2=t2, op=nl.maximum)
    nisa.dma_copy(dst=data0.ap(pattern=[[1, pn], [0, 1]], offset=p0), src=t0)
  return data0
""",
  "mean_epilogue": """\
# fusezap NKI kernel (nki 0.6 API), rendered by fusezap.codegen.nki from F=(4, 6) R=(4, 1)
# kernel dims [4, 6] (nr): P d0[4] F d1[6] loops -
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def r_4_6(data0, data1):
  t0 = nl.ndarray((4, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t0, src=data1.ap(pattern=[[6, 4], [1, 6]], offset=0))
  t1 = nl.ndarray((4, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_reduce(dst=t1, op=nl.add, data=t0, axis=1, keepdims=True)
  t2 = nl.ndarray((4, 1), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=t2, data=t1, op0=nl.multiply, operand0=0.1666666716337204)
  nisa.dma_copy(dst=data0.ap(pattern=[[1, 4], [0, 1]], offset=0), src=t2)
  return data0
""",
  "pad_where_sin_cast": """\
# fusezap NKI kernel (nki 0.6 API), rendered by fusezap.codegen.nki from F=(5, 6) R=(5, 6)
# kernel dims [5, 6] (nn): P d0[5] F d1[6] loops -
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def E_5_6(data0, data1, data2):
  t0 = nl.ndarray((5, 6), dtype=nl.uint8, buffer=nl.sbuf)
  nisa.memset(dst=t0, value=0)
  t1 = nl.ndarray((5, 6), dtype=nl.uint8, buffer=nl.sbuf)
  nisa.memset(dst=t1, value=1)
  nisa.dma_copy(dst=t0[1:4, 2:6], src=t1[1:4, 2:6])
  t2 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.memset(dst=t2, value=0.0)
  nisa.dma_copy(dst=t2[1:4, 2:6], src=data1.ap(pattern=[[4, 3], [1, 4]], offset=0))
  t3 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.memset(dst=t3, value=2.5)
  nisa.tensor_copy_predicated(dst=t3, src=t2, predicate=t0)
  t4 = nl.ndarray((5, 6), dtype=nl.uint8, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=t4, data=t3, op0=nl.less, operand0=0.0, reverse0=True)
  t5 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t5, src=data2.ap(pattern=[[6, 5], [1, 6]], offset=0))
  t6 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  t7 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  t8 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  t9 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.tensor_scalar(dst=t6, data=t5, op0=nl.multiply, operand0=0.15915493667125702)
  t10 = nl.ndarray((5, 6), dtype=nl.int32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=t10, src=t6)
  nisa.tensor_copy(dst=t7, src=t10)
  nisa.tensor_scalar(dst=t8, data=t7, op0=nl.multiply, operand0=6.28125)
  nisa.tensor_tensor(dst=t9, data1=t5, data2=t8, op=nl.subtract)
  nisa.tensor_scalar(dst=t8, data=t7, op0=nl.multiply, operand0=0.0019378662109375)
  nisa.tensor_tensor(dst=t9, data1=t9, data2=t8, op=nl.subtract)
  nisa.tensor_scalar(dst=t8, data=t7, op0=nl.multiply, operand0=-2.559031372584286e-06)
  nisa.tensor_tensor(dst=t9, data1=t9, data2=t8, op=nl.subtract)
  t11 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.activation(dst=t11, op=nl.sin, data=t9)
  t12 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  nisa.memset(dst=t12, value=-1.0)
  nisa.tensor_copy_predicated(dst=t12, src=t11, predicate=t4)
  t13 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  t14 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  t15 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  t16 = nl.ndarray((5, 6), dtype=nl.float32, buffer=nl.sbuf)
  t17 = nl.ndarray((5, 6), dtype=nl.int32, buffer=nl.sbuf)
  nisa.tensor_copy(dst=t17, src=t12)
  nisa.tensor_copy(dst=t13, src=t17)
  nisa.tensor_tensor(dst=t14, data1=t13, data2=t12, op=nl.greater)
  nisa.tensor_scalar(dst=t16, data=t12, op0=nl.greater, operand0=0.0)
  nisa.tensor_tensor(dst=t14, data1=t14, data2=t16, op=nl.multiply)
  nisa.tensor_tensor(dst=t15, data1=t13, data2=t12, op=nl.less)
  nisa.tensor_scalar(dst=t16, data=t12, op0=nl.less, operand0=0.0)
  nisa.tensor_tensor(dst=t15, data1=t15, data2=t16, op=nl.multiply)
  nisa.tensor_tensor(dst=t13, data1=t13, data2=t14, op=nl.subtract)
  nisa.tensor_tensor(dst=t13, data1=t13, data2=t15, op=nl.add)
  nisa.tensor_copy(dst=t17, src=t13)
  nisa.dma_copy(dst=data0.ap(pattern=[[6, 5], [1, 6]], offset=0), src=t17)
  return data0
""",
  "split_flat_recip": """\
# fusezap NKI kernel (nki 0.6 API), rendered by fusezap.codegen.nki from F=(4096,) R=(4096,)
# kernel dims [128, 32] (nn): P d0[128] F d1[32] loops -
import nki
import nki.language as nl
import nki.isa as nisa

@nki.jit
def E_4096(data0, data1):
  t0 = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.sbuf)
  nisa.dma_copy(dst=t0, src=data1.ap(pattern=[[32, 128], [1, 32]], offset=0))
  t1 = nl.ndarray((128, 32), dtype=nl.float32, buffer=nl.sbuf)
  nisa.reciprocal(dst=t1, data=t0)
  nisa.dma_copy(dst=data0.ap(pattern=[[32, 128], [1, 32]], offset=0), src=t1)
  return data0
""",}

@pytest.mark.parametrize("name", list(SNAPSHOT_GRAPHS))
def test_golden_source(name):
  (src,) = _srcs(SNAPSHOT_GRAPHS[name]())
  assert src == SNAPSHOTS[name], f"rendered source changed:\n{src}"
  compile(src, name, "exec")

@pytest.mark.parametrize("name", list(SNAPSHOT_GRAPHS))
def test_render_is_deterministic(name):
  t = SNAPSHOT_GRAPHS[name]()
  (si,) = _kernels(t)
  r = NKIRenderer()
  assert r.render(si.ast, si.var_vals).src == r.render(si.ast, si.var_vals).src == _srcs(SNAPSHOT_GRAPHS[name]())[0]

# ---------------------------------------------------------------- end to end at real tile sizes

R0 = np.random.default_rng(7)

@pytest.mark.parametrize("shape", [(300, 1000), (129, 2049), (4096,), (1009,), (2, 3, 130, 7)])
def test_large_elementwise(nki, shape):
  a, b = R0.standard_normal(shape, dtype=np.float32), R0.standard_normal(shape[-1:], dtype=np.float32)
  out = (Tensor(a) * Tensor(b) + 1.5).relu().numpy()
  assert nki["kernels"] == 1
  np.testing.assert_allclose(out, np.maximum(a * b + np.float32(1.5), 0), rtol=1e-6, atol=1e-6)

@pytest.mark.parametrize("shape,axis", [((300, 1000), 1), ((300, 1000), 0), ((130, 1500), 1), ((5000,), 0), ((3, 257, 600), (0, 2))])
def test_large_reduce(nki, shape, axis):
  a = R0.standard_normal(shape, dtype=np.float32)
  s = (Tensor(a).sum(axis=axis) * 0.5).numpy()
  np.testing.assert_allclose(s, a.sum(axis=axis) * 0.5, rtol=1e-4, atol=1e-4)
  m = Tensor(a).max(axis=axis).numpy()
  np.testing.assert_array_equal(m, a.max(axis=axis))

def test_large_softmax_layernorm(nki):
  a = R0.standard_normal((200, 1300), dtype=np.float32)
  e = np.exp(a - a.max(1, keepdims=True)); e /= e.sum(1, keepdims=True)
  np.testing.assert_allclose(Tensor(a).softmax().numpy(), e, rtol=1e-5, atol=1e-6)
  ln = (a - a.mean(1, keepdims=True)) / np.sqrt(a.var(1, keepdims=True) + 1e-5)
  np.testing.assert_allclose(Tensor(a).layernorm().numpy(), ln, rtol=1e-4, atol=1e-4)

@pytest.mark.parametrize("lim", [20, 1000, 4096, 65536])
def test_sin_range_reduction(nki, lim):   # SPEC elementwise tolerance at RoPE-sized arguments (Cody-Waite reduction)
  a = np.concatenate([np.linspace(-lim, lim, 20001, dtype=np.float32), R0.uniform(-lim, lim, 20000).astype(np.float32)])
  np.testing.assert_allclose(Tensor(a).sin().numpy(), np.sin(a), rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("shape,tiles", [((4, 100), "default"), ((4, 2000), "default"), ((5, 23), "tiny"), ((3, 2, 300), "default")])
def test_int_sum_is_exact_past_2_24(monkeypatch, shape, tiles):   # small inputs, partial sums > 2**24: tensor_reduce is float32
  _route_to_nki(monkeypatch, TILES[tiles])
  a = np.full(shape, 1_000_001, np.int32)
  a[..., 0] += np.arange(int(np.prod(shape[:-1]))).reshape(shape[:-1]).astype(np.int32)
  np.testing.assert_array_equal(Tensor(a).sum(axis=-1).numpy(), a.sum(axis=-1, dtype=np.int32))
  b = np.full((7, 50), 2**30, np.int32)   # wraps in int32 like the other tiers
  np.testing.assert_array_equal(Tensor(b).sum(axis=1).numpy(), b.sum(axis=1, dtype=np.int32))
  assert "tensor_reduce" not in "".join(_srcs(Tensor(np.zeros(shape, np.int32)).sum(axis=-1), TILES[tiles]))

@pytest.mark.parametrize("dt", [dtypes.float32, dtypes.int32])
def test_where_constant_true_branch_is_a_tile(nki, dt):   # real nki 0.6 tensor_copy_predicated reads src.shape
  a = R0.standard_normal((6, 9)).astype(np.float32)
  graph = lambda: (Tensor(a) > 0).where(7, Tensor(a).cast(dt))
  for src in _srcs(graph()): assert not re.search(r"tensor_copy_predicated\(dst=\w+, src=[-\d(f]", src), src
  np.testing.assert_array_equal(graph().numpy(), np.where(a > 0, 7, a.astype(to_np(dt))).astype(to_np(dt)))

def test_shim_predicated_copy_rejects_scalar_src():
  d, p = nl.ndarray((1, 2), nl.float32), _sb([[1, 0]], nl.uint8)
  with pytest.raises(AssertionError, match="src must be a tile"): nisa.tensor_copy_predicated(dst=d, src=1.0, predicate=p)

def test_cast_truncates(nki):
  a = np.array([-2.7, -2.5, -0.5, 0.5, 1.5, 2.5, 2.7, 1e6 + 0.5], np.float32)
  np.testing.assert_array_equal(Tensor(a).cast(dtypes.int32).numpy(), np.trunc(a).astype(np.int32))

def test_symbolic_dims_are_bound(nki):
  from fusezap.jit import bound
  from fusezap.shape.symbolic import Var
  n = Var("n", 1, 16)
  a = np.arange(16, dtype=np.float32)
  (si,) = _kernels((Tensor(a).shrink(((0, n),)) * 2.0).sum(), {n: 3})
  for k in (3, 16):
    src = NKIRenderer().render(si.ast, {n: k}).src   # var_vals are compiled in (NKI_BACKEND.md exception 2)
    assert f"F=({k},) R=(1,)" in src
    with bound({n: k}): out = (Tensor(a).shrink(((0, n),)) * 2.0).sum().numpy()
    assert out == pytest.approx(2 * a[:k].sum())
  assert nki["kernels"] == 2

@pytest.mark.parametrize("tiles", list(TILES))
def test_two_stores_and_masked_store(tiles):
  """Hand-built AST: two STORE roots, one through a padded view (only the valid box is written), vs the NPY device."""
  from fusezap.device import Buffer
  from fusezap.device.npy import NPYDevice
  from fusezap.ir import ConstBuffer, LazyOp, MemBuffer, ScheduleItem
  from fusezap.ops import BinaryOps, BufferOps, UnaryOps
  from fusezap.shape.view import ShapeTracker
  st, f32 = ShapeTracker.from_shape((5, 6)), dtypes.float32
  ld = LazyOp(BufferOps.LOAD, (), MemBuffer(2, f32, st))
  three = LazyOp(BufferOps.CONST, (), ConstBuffer(3.0, f32, st))
  s0 = LazyOp(BufferOps.STORE, (LazyOp(BinaryOps.ADD, (ld, LazyOp(UnaryOps.SQRT, (ld,)))),), MemBuffer(0, f32, st))
  s1 = LazyOp(BufferOps.STORE, (LazyOp(BinaryOps.MUL, (ld, three)),), MemBuffer(1, f32, ShapeTracker.from_shape((3, 4)).pad(((1, 1), (1, 1)))))
  x = np.random.default_rng(3).random(30, dtype=np.float32)
  outs = []
  for runner in ("npy", "nki"):
    bufs = [Buffer("NPY", 30, f32), Buffer("NPY", 12, f32), Buffer("NPY", 30, f32)]
    bufs[1].copyin(memoryview(np.full(12, -7.0, np.float32)).cast("B")); bufs[2].copyin(memoryview(x).cast("B"))
    si = ScheduleItem((s0, s1), tuple(bufs), {}, "kernel", 2)
    if runner == "npy": NPYDevice("NPY").run(si)
    else: nkishim.run_item(si, TILES[tiles])
    outs.append([b.numpy() for b in bufs[:2]])
  np.testing.assert_allclose(outs[1][0], x + np.sqrt(x), rtol=1e-6)
  np.testing.assert_array_equal(outs[1][1], (x.reshape(5, 6)[1:4, 1:5] * 3).reshape(-1))
  for a, b in zip(*outs): np.testing.assert_allclose(a, b, rtol=1e-6)

# ---------------------------------------------------------------- what is not lowered (reported, never silently wrong)

def test_unsupported_is_reported():
  with pytest.raises(NKIUnsupported, match="IDIV.*rejects"): [NKIRenderer().render(si.ast) for si in _kernels(Tensor(np.arange(6, dtype=np.int32)) // 4)]
  multi = _z(2, 3, 4).permute(1, 0, 2).reshape(3, 8)[:, :6].reshape(3, 6) + 1
  with pytest.raises(NKIUnsupported, match="multi-view"): [NKIRenderer().render(si.ast) for si in _kernels(multi)]
  with pytest.raises(NKIUnsupported, match="int64"): [NKIRenderer().render(si.ast) for si in _kernels(Tensor(np.arange(4, dtype=np.int64)) + 1)]

# ---------------------------------------------------------------- shim semantics

def _sb(vals, dt=None) -> "nkishim.NkiTensor":
  vals = np.asarray(vals)
  t = nl.ndarray(vals.shape, dt or nl.float32)
  t.write_raw(vals.astype(t.storage.dtype) if (dt or nl.float32).name != "bfloat16" else nkishim._bf16_encode(vals))
  return t

def test_shim_float_to_int_rounds_to_nearest_even_and_saturates():
  x, d = _sb([[0.5, 1.5, 2.5, -2.5, 3e9, -3e9, np.nan, np.inf]]), nl.ndarray((1, 8), nl.int32)
  nisa.tensor_copy(dst=d, src=x)
  np.testing.assert_array_equal(d.values(), [[0, 2, 2, -2, 2**31 - 1, -2**31, 2**31 - 1, 2**31 - 1]])

def test_shim_bf16_rounds_on_write():
  x, d = _sb([[1.0 + 2**-8, 1.0 + 3 * 2**-8, 3.0]]), nl.ndarray((1, 3), nl.bfloat16)
  nisa.tensor_copy(dst=d, src=x)
  np.testing.assert_array_equal(d.values(), [[1.0, 1.0 + 2**-6, 3.0]])

def test_shim_int32_tensor_tensor_is_native_and_wraps():
  a, b, d = _sb([[2**31 - 1, 16777217]], nl.int32), _sb([[1, 0]], nl.int32), nl.ndarray((1, 2), nl.int32)
  nisa.tensor_tensor(dst=d, data1=a, data2=b, op=nl.add)
  np.testing.assert_array_equal(d.values(), [[-2**31, 16777217]])
  f = nl.ndarray((1, 2), nl.float32)
  nisa.tensor_scalar(dst=d, data=b, op0=nl.add, operand0=16777217.0)   # float32 math: 2**24 + 1 is not representable
  nisa.tensor_copy(dst=f, src=d)
  assert f.values()[0, 0] == 16777216.0

def test_shim_int32_engine_dispatch_matches_nki_0_6():
  """Verified against the real 0.6.0 simulator and neuronx-cc 2.27 (NKI-N2): with no engine named, only int32
  add/subtract/multiply run on GpSimd (exact); maximum runs on VectorE in float32. What inf2 cannot compile is refused."""
  a, b, d = _sb([[2**24 + 1, 7]], nl.int32), _sb([[2**24, 2]], nl.int32), nl.ndarray((1, 2), nl.int32)
  nisa.tensor_tensor(dst=d, data1=a, data2=b, op=nl.maximum)
  assert d.values()[0, 0] == 2**24                                       # float32: 2**24 + 1 rounds down
  nisa.tensor_tensor(dst=d, data1=a, data2=b, op=nl.subtract, engine=nisa.gpsimd_engine)
  np.testing.assert_array_equal(d.values(), [[1, 5]])
  with pytest.raises(AssertionError, match="rejects"): nisa.tensor_tensor(dst=d, data1=a, data2=b, op=nl.maximum, engine=nisa.gpsimd_engine)
  for op in (nl.divide, nl.fmod, nl.mod):
    with pytest.raises(AssertionError, match="rejects"): nisa.tensor_tensor(dst=d, data1=a, data2=b, op=op)
    with pytest.raises(AssertionError, match="rejects"): nisa.tensor_scalar(dst=d, data=a, op0=op, operand0=1.0)
  nisa.tensor_tensor(dst=d, data1=_sb([[-1, 70000]], nl.int32), data2=_sb([[65535, 65535]], nl.int32), op=nl.bitwise_and)
  np.testing.assert_array_equal(d.values(), [[65535, 70000 & 65535]])

@pytest.mark.parametrize("tiles", list(TILES))
def test_int32_max_compare_are_exact_past_2_24(monkeypatch, tiles):
  """int32 MAX/CMPLT/CMPEQ and the integer MAX reduce: exact for every int32 (float32 engines are exact below 2**24)."""
  _route_to_nki(monkeypatch, TILES[tiles])
  r = np.random.default_rng(2)
  a = np.concatenate([[2**24 + 1, 2**24, 3, -(2**24 + 3), -(2**24 + 1), -(2**31), 2**31 - 1, 2**31 - 2, -1, 0, 65535, 65536],
                      r.integers(-2**31, 2**31, 36)]).astype(np.int32).reshape(4, 12)
  b = np.concatenate([[2**24, 2**24 + 1, 3, -(2**24 + 1), -(2**24 + 3), 2**31 - 1, -(2**31), 2**31 - 1, 0, -1, 65536, 65535],
                      r.integers(-2**31, 2**31, 36)]).astype(np.int32).reshape(4, 12)
  np.testing.assert_array_equal(Tensor(a).max(axis=1).numpy(), a.max(axis=1))
  np.testing.assert_array_equal(Tensor(a).max(axis=0).numpy(), a.max(axis=0))
  np.testing.assert_array_equal(Tensor(a).maximum(Tensor(b)).numpy(), np.maximum(a, b))
  np.testing.assert_array_equal((Tensor(a) < Tensor(b)).numpy(), a < b)
  np.testing.assert_array_equal((Tensor(a) == Tensor(b)).numpy(), a == b)
  np.testing.assert_array_equal((Tensor(a) == Tensor(a)).numpy(), np.ones_like(a, bool))

@pytest.mark.parametrize("dt", [np.float32, np.float16])
def test_trunc_cast_without_fmod(nki, dt):
  x = np.array([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.999, -2.999, 7.0, -7.0, 0.0, 1e6 + 0.5, -1e6 - 0.5, 2**23 + 1, 3e9, -3e9],
               np.float32)
  if dt == np.float16: x = np.clip(x, -6e4, 6e4).astype(dt)
  want = np.clip(np.trunc(x.astype(np.float64)), -2**31, 2**31 - 1).astype(np.int32)
  np.testing.assert_array_equal(Tensor(x).cast(dtypes.int32).numpy(), want)

def test_shim_refuses_what_inf2_cannot_compile():
  """neuronx-cc 2.27 (inf2) rejects these although the simulator runs them (NKI-N2 NEFF run of the op suite)."""
  t = nl.ndarray((5, 6), nl.float32)
  nisa.memset(dst=t[0:5, 2:6], value=1.0)                               # free-axis slices are fine
  with pytest.raises(AssertionError, match="partition 1"): nisa.memset(dst=t[1:4, :], value=1.0)
  nisa.dma_copy(dst=t[1:4, :], src=nkishim.tensor(np.zeros(18, np.float32), nl.float32).ap(pattern=[[6, 3], [1, 6]], offset=0))
  h = nkishim.tensor(np.zeros(30, np.float32), nl.float32)
  with pytest.raises(AssertionError, match="negative partition step"): nisa.dma_copy(dst=t, src=h.ap(pattern=[[-6, 5], [1, 6]], offset=24))
  nisa.dma_copy(dst=t, src=h.ap(pattern=[[6, 5], [-1, 6]], offset=5))   # a negative free-axis step compiles

def test_shim_hw_sin_has_no_range_reduction():
  x, d = _sb([[1.0, -1.0, 9.0]]), nl.ndarray((1, 3), nl.float32)
  nisa.activation(dst=d, op=nl.sin, data=x)
  np.testing.assert_allclose(d.values()[0, :2], np.sin([1.0, -1.0]), rtol=1e-6)
  assert abs(d.values()[0, 2] - 27.65) < 0.1   # the cubic fit outside [-pi, pi], not sin(9) = 0.41

def test_shim_checks():
  with pytest.raises(AssertionError, match="partition dim"): nl.ndarray((129, 4), nl.float32)
  with pytest.raises(AssertionError, match="per partition"): nl.ndarray((128, 64 * 1024), nl.float32)
  h = nkishim.tensor(np.zeros(8, np.float32), nl.float32)
  t = nl.ndarray((2, 4), nl.float32)
  with pytest.raises(IndexError, match="out-of-bounds"): nisa.dma_copy(dst=t, src=h.ap(pattern=[[4, 2], [1, 4]], offset=1))
  with pytest.raises(AssertionError, match="cannot access"): nisa.tensor_tensor(dst=t, data1=h.ap(pattern=[[4, 2], [1, 4]]), data2=t, op=nl.add)
  with pytest.raises(AssertionError, match="trailing free dims"): nisa.tensor_reduce(dst=nl.ndarray((1, 4), nl.float32), op=nl.add, data=t, axis=0)
  with pytest.raises(AssertionError, match="not an activation"): nisa.activation(dst=t, op=nl.add, data=t)
  assert np.isnan(t.values()).all()   # fresh tiles hold garbage, never zeros
  nisa.dma_copy(dst=t, src=h.ap(pattern=[[4, 2], [1, 4]]))
  assert (t.values() == 0).all()

def test_shim_slices_and_views():
  h = nkishim.tensor(np.arange(24, dtype=np.float32), nl.float32)
  v = h.ap(pattern=[[6, 4], [1, 6]], offset=0)
  np.testing.assert_array_equal(v[1:3, 2:5].values(), np.arange(24).reshape(4, 6)[1:3, 2:5])
  np.testing.assert_array_equal(v[2].values(), np.arange(24).reshape(4, 6)[2])        # HBM: an int index drops the dim
  t = nl.ndarray((4, 6), nl.float32)
  nisa.dma_copy(dst=t, src=v)
  assert t[2].shape == (1, 6)                                                       # on-chip: the partition dim stays
  assert t.view(nl.int32).values()[0, 1] == np.float32(1).view(np.int32)

def test_generated_source_imports_real_nki_names():
  """The source runs against the shim through an import hook; it must name only nki, nki.language and nki.isa."""
  src = SNAPSHOTS["broadcast_scale_add"]
  assert "import nki\nimport nki.language as nl\nimport nki.isa as nisa\n" in src
  fn = nkishim.load(src, "E_300_1000")
  assert getattr(fn, "__nki_jit__", False)

# ---------------------------------------------------------------- N5: exact transcendentals (N4 run 2 on inf2)

def _ulps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
  return np.abs(np.where(ia < 0, -2**31 - ia, ia) - np.where(ib < 0, -2**31 - ib, ib))

SPECIAL = np.array([0, -0.0, 1, -1, 0.5, 2, np.inf, -np.inf, np.nan, 1e-40, 1.4e-45, 3.4e38, 128, -149, 127.99, -126.5], np.float32)

@pytest.mark.parametrize("name,ref,x", [
  ("exp2", np.exp2, np.random.default_rng(1).uniform(-160, 130, 40000)),
  ("exp", np.exp, np.random.default_rng(2).uniform(-110, 89, 40000)),
  ("log2", np.log2, np.exp2(np.random.default_rng(3).uniform(-149, 128, 40000))),   # every binade, subnormals too
  ("log2", np.log2, np.random.default_rng(4).uniform(1e-30, 1e30, 40000)),           # run 2's probe input: ScalarE log broke here
  ("sqrt", np.sqrt, np.exp2(np.random.default_rng(5).uniform(-149, 128, 40000)))], ids=["exp2", "exp", "log2_binades", "log2_run2", "sqrt"])
def test_exact_transcendentals(name, ref, x):
  """EXP2 / exp / LOG2 / SQRT render to VectorE range reduction + polynomial (no ScalarE exp/log activation): within 1 ulp (log2:
  2, s = (m-1)/(m+1) rounds twice) of the correctly rounded value over the whole float32 range and right on 0, -0, +-inf, NaN, negatives and subnormals."""
  x = np.concatenate([x.astype(np.float32), SPECIAL])
  srcs = _srcs(getattr(Tensor(x), name)())
  assert "op=nl.log" not in "".join(srcs) and ("op=nl.exp" not in "".join(srcs))
  with np.errstate(all="ignore"): r = ref(x.astype(np.float64)).astype(np.float32)
  y = getattr(Tensor(x, device="NKISHIM"), name)().numpy()
  np.testing.assert_array_equal(np.isnan(y), np.isnan(r))
  fin = np.isfinite(r) & ~np.isnan(y)
  np.testing.assert_array_equal(y[~fin & ~np.isnan(r)], r[~fin & ~np.isnan(r)])   # +-inf where the reference overflows
  assert _ulps(y[fin], r[fin]).max() <= (2 if name == "log2" else 1), (name, x[fin][np.argmax(_ulps(y[fin], r[fin]))])

def _hw_like_sqrt(x):
  """A ScalarE sqrt like inf2's (N5 run 1, sqrt_wide): ~3e-6 relative on [1, 4), far off outside (a limited-domain table)."""
  x = np.asarray(x, np.float32)
  with np.errstate(all="ignore"):
    near = np.sqrt(x.astype(np.float64)) * (1 + 3e-6 * np.cos(x.astype(np.float64) * 7))
    far = np.sqrt(np.clip(x, 0.25, 16).astype(np.float64)) * 1.7
    return np.where((x >= 1) & (x < 4), near, np.where(np.isnan(x), np.nan, far)).astype(np.float32)

def test_sqrt_is_range_reduced(monkeypatch):
  """N6: SQRT feeds ScalarE only significands in [1, 4) (x = m * 2**(2k)), so a sqrt activation that is only right near 1, as
  on inf2, still gives <= 1 ulp over the whole float32 range (subnormals, both parities of the exponent) and the special values."""
  from fusezap.device import nkishim
  monkeypatch.setattr(nkishim.ACTIVATION["sqrt"], "fn", _hw_like_sqrt)
  x = np.concatenate([np.exp2(np.random.default_rng(6).uniform(-149, 128, 40000)).astype(np.float32), SPECIAL, np.float32([4, 0.25, 1e-45, 2.0 ** -126])])
  with np.errstate(all="ignore"): r = np.sqrt(x.astype(np.float64)).astype(np.float32)
  y = Tensor(x, device="NKISHIM").sqrt().numpy()
  np.testing.assert_array_equal(np.isnan(y), np.isnan(r))
  fin = np.isfinite(r)
  assert _ulps(y[fin], r[fin]).max() <= 1
  assert np.array_equal(y[~fin & ~np.isnan(r)], r[~fin & ~np.isnan(r)]) and np.signbit(y[x == 0]).tolist() == np.signbit(x[x == 0]).tolist()
  from fusezap.serve.neuron_kv import rmsnorm   # the RMSNorm kernel's rsqrt too: rows of 1e-30 .. 1e30 scale
  rows = (np.random.default_rng(8).standard_normal((6, 64)) * np.float64([1e-30, 1e-10, 1, 1e10, 1e18, 1e-18])[:, None]).astype(np.float32)
  h = rmsnorm(Tensor(rows, device="NKISHIM"), Tensor(np.ones(64, np.float32), device="NKISHIM"), 1e-38)[1].numpy()
  x64 = rows.astype(np.float64)
  np.testing.assert_allclose(h, x64 / np.sqrt((x64 * x64).mean(-1, keepdims=True) + 1e-38), rtol=2e-6)

def test_exact_exp_in_attention_and_fast_switch(monkeypatch):
  """The attention kernel's exp(s - m) is the exact VectorE one too; FZ_NKI_FAST_EXP=1 (nki.EXACT False) restores ScalarE."""
  from fusezap.codegen import nki, nki_kernels as nk
  sp = nk.AttnSpec((1, 1), (1, 4), 8, 16, 16, (0, 0, 16, 1, 0), (0, 0, 16, 1, 0), (0, 0, 16, 1, 0), (0, 0, 16, 1, 0), 0.25, "causal")
  assert "op=nl.exp" not in nk.render_attention(sp).src
  monkeypatch.setattr(nki, "EXACT", False)
  assert nk.render_attention(sp).src.count("op=nl.exp") == 2
  assert "op=nl.log" in "".join(_srcs(Tensor(np.ones(4, np.float32)).log2()))
