"""Integrator: top-level exports (fz.jit, fz.bind, fz.bound, fz.Var) and that `import fusezap.jit` stays the module."""
import numpy as np
import fusezap as fz
from fusezap import Tensor

def test_exports():
  import fusezap.jit as J
  assert J._programs and J.Jit is fz.Jit and fz.bind is J.bind and fz.bound is J.bound
  assert isinstance(fz.jit(lambda x: x), fz.Jit)

def test_fz_jit_decorator_symbolic():
  n = fz.Var("n", 1, 8)
  @fz.jit
  def f(x: Tensor, m) -> Tensor: return (x * 2 + 1).realize()
  for v in (3, 5, 2, 7):
    a = np.arange(8, dtype=np.float32)
    out = f(Tensor(a), fz.bind(n, v))
    np.testing.assert_allclose(out.numpy(), a * 2 + 1)
