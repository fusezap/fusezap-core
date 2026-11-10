"""Shared fixtures (SPEC §10). Initial version by the integrator; WP7 extends it (keep these names stable).

  device        parametrized over available tiers: "interp", "NPY", "CPU", "METAL" by default.
                The fixture sets FZ_INTERP / the default device for the test and yields the device name.
                Restrict with FZ_TEST_TIERS=interp,CPU. "CUDA" (P3-CUDA) and "NKISHIM" (NEURON N1: generated NKI
                kernels run by the numpy shim, docs/NKI_BACKEND.md §6) and "NKISIM" (N2: the real nki 0.6 simulator in .venv-nki,
                Linux x86_64 only) and "NEURON" (N4: a real NeuronCore on inf2/trn1) run only when FZ_TEST_TIERS names them.
  assert_close  assert_close(ne_tensor_or_array, torch_tensor_or_array, dtype=None, reduce=False)
"""
from __future__ import annotations
import os
import numpy as np
import pytest

os.environ.setdefault("FUSEZAP_DEV", "1")   # tests run as a source checkout (dev mode)

TIERS = ("interp", "NPY", "CPU", "METAL")   # the default tiers
OPT_IN_TIERS = ("CUDA", "NKISHIM", "NKISIM", "NEURON")   # selected only when FZ_TEST_TIERS lists them

TOLS = {  # (rtol, atol): elementwise, reductions/matmul
  "float32": ((1e-5, 1e-6), (1e-4, 1e-5)),
  "float16": ((1e-2, 1e-2), (1e-2, 1e-2)),
  "bfloat16": ((1e-2, 1e-2), (1e-2, 1e-2)),
}

NKI_TIERS = ("NKISHIM", "NKISIM", "NEURON")   # NEURON tiers: a kernel the NKI renderer refuses (NKIUnsupported) reports as xfailed, with the reason

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
  rep = (yield).get_result()
  if rep.when != "call" or call.excinfo is None: return
  if getattr(getattr(item, "callspec", None), "params", {}).get("device") not in NKI_TIERS: return
  from fusezap.codegen.nki import NKIUnsupported
  if call.excinfo.errisinstance(NKIUnsupported): rep.outcome, rep.wasxfail = "skipped", f"NKIUnsupported: {call.excinfo.value}"

def _tier_available(tier: str) -> bool:
  if tier == "interp": return True
  from fusezap.device import Device
  return Device.available(tier)

def _selected_tiers() -> list[str]:
  want = os.getenv("FZ_TEST_TIERS")
  tiers = [t.strip() for t in want.split(",")] if want else list(TIERS)
  return [t for t in tiers if t in TIERS + OPT_IN_TIERS]

def pytest_generate_tests(metafunc):
  if "device" in metafunc.fixturenames:
    metafunc.parametrize("device", _selected_tiers(), indirect=True)

@pytest.fixture
def device(request, monkeypatch):
  tier = request.param
  if not _tier_available(tier): pytest.skip(f"{tier} not available")
  from fusezap import helpers
  from fusezap.device import Device
  helpers.getenv.cache_clear()
  if tier == "interp":
    monkeypatch.setenv("FZ_INTERP", "1")
    dev = "CPU" if _tier_available("CPU") else "NPY"
  else:
    monkeypatch.delenv("FZ_INTERP", raising=False)
    dev = tier
  monkeypatch.setattr(Device, "DEFAULT", dev, raising=False)
  yield tier
  helpers.getenv.cache_clear()

def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
  """uint16 bfloat16 bit pattern -> float32."""
  return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)

def f32_to_bf16(a: np.ndarray) -> np.ndarray:
  """float32 -> uint16 bfloat16 bits, round-to-nearest-even (NaN not handled)."""
  u = np.ascontiguousarray(a, np.float32).view(np.uint32)
  return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def _to_np(x) -> np.ndarray:
  if hasattr(x, "detach") and hasattr(x, "cpu"): return x.detach().cpu().float().numpy() if x.is_floating_point() else x.detach().cpu().numpy()
  if hasattr(x, "numpy"): return np.asarray(x.numpy())
  return np.asarray(x)

@pytest.fixture
def assert_close():
  def _assert(actual, expected, dtype: str | None = None, reduce: bool = False):
    a, e = _to_np(actual), _to_np(expected)
    if dtype == "bfloat16" and a.dtype == np.uint16: a = bf16_to_f32(a)  # host bf16 is uint16 bits (SPEC §4)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape}"
    name = dtype or (str(e.dtype) if e.dtype.kind == "f" else None)
    if name is None or a.dtype.kind in "biu":
      np.testing.assert_array_equal(a, e)
      return
    rtol, atol = TOLS.get(name, TOLS["float32"])[1 if reduce else 0]
    np.testing.assert_allclose(a.astype(np.float32), e.astype(np.float32), rtol=rtol, atol=atol)
  return _assert

# ---- WP7 additions: helpers for the op conformance suite (tests/test_ops.py)
def pytest_configure(config):
  config.addinivalue_line("markers", "slow: slow tests (real-model serving, perplexity, CLI end-to-end); skipped unless FZ_SLOW=1 or -m slow")

def _skip_public_export_gaps(items):
  """In the public source tree, tests/public_export_missing.txt lists the bench/
  and scripts/ code the export leaves out (job scripts that run FuseZap Invarix, and what uses them). A test whose own
  source names one of them is skipped there. In this repo the file does not exist and nothing changes."""
  path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public_export_missing.txt")
  if not os.path.exists(path): return
  import inspect, re
  with open(path) as f: names = [os.path.basename(ln.strip()) for ln in f if ln.strip()]
  if not names: return
  stems = {n[:-3] for n in names if n.endswith(".py")}
  rx = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b"
                  + (r"|(?:\bimport\s+(?:[\w.]+(?:\s+as\s+\w+)?\s*,\s*)*|\bfrom\s+)(" + "|".join(map(re.escape, sorted(stems))) + r")\b" if stems else ""))
  def sources(fn, mod, seen):   # the test, the module-level helpers it calls and the module's fixtures it uses
    fn = getattr(fn, "__wrapped__", fn)
    if id(fn) in seen or not callable(fn) or getattr(fn, "__module__", None) != getattr(mod, "__name__", None): return
    seen.add(id(fn))
    try: yield inspect.getsource(fn)
    except (OSError, TypeError): return
    for name in getattr(getattr(fn, "__code__", None), "co_names", ()): yield from sources(getattr(mod, name, None), mod, seen)
  for it in items:
    fn, mod = getattr(it, "function", None), getattr(it, "module", None)
    if fn is None or mod is None: continue
    seen: set = set()
    src = "\n".join([*sources(fn, mod, seen), *(s for f in getattr(it, "fixturenames", ()) for s in sources(getattr(mod, f, None), mod, seen))])
    if m := rx.search(src):
      it.add_marker(pytest.mark.skip(reason=f"needs {m.group(1) or m.group(2)}, which is not in the public export"))

def pytest_collection_modifyitems(config, items):
  _skip_public_export_gaps(items)
  if os.getenv("FZ_SLOW", "") not in ("", "0") or "slow" in (config.getoption("-m") or ""): return
  skip = pytest.mark.skip(reason="slow: set FZ_SLOW=1 or pass -m slow")
  for it in items:
    if "slow" in it.keywords: it.add_marker(skip)

@pytest.fixture
def rng():
  return np.random.default_rng(1337)
