"""Integrator smoke tests: the shared, already-implemented pieces work and every stub imports."""
import importlib
import pytest
from fusezap.dtype import dtypes, promote
from fusezap.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps, LoadOps

MODULES = ["fusezap", "fusezap.helpers", "fusezap.dtype", "fusezap.ops", "fusezap.ir", "fusezap.lazy", "fusezap.tensor", "fusezap.autograd",
           "fusezap.realize", "fusezap.interpret", "fusezap.schedule", "fusezap.shape", "fusezap.codegen.uops", "fusezap.codegen.lower",
           "fusezap.codegen.cstyle", "fusezap.codegen.metal", "fusezap.device", "fusezap.device.cpu", "fusezap.device.metal",
           "fusezap.device.npy", "fusezap.nn", "fusezap.optim"]

@pytest.mark.parametrize("mod", MODULES)
def test_imports(mod): importlib.import_module(mod)

def test_primitive_op_count():
  assert sum(len(e) for e in (UnaryOps, BinaryOps, TernaryOps, ReduceOps, MovementOps, LoadOps)) == 28

@pytest.mark.parametrize("a,b,out", [
  (dtypes.int32, dtypes.float16, dtypes.float16), (dtypes.float16, dtypes.bfloat16, dtypes.float32),
  (dtypes.bool, dtypes.int8, dtypes.int8), (dtypes.int32, dtypes.int64, dtypes.int64),
  (dtypes.bfloat16, dtypes.float32, dtypes.float32)])
def test_promote(a, b, out): assert promote(a, b) == promote(b, a) == out
