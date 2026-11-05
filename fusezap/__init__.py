from fusezap._version import __version__  # noqa: F401
from fusezap.dtype import dtypes, DType  # noqa: F401
from fusezap.device import Device  # noqa: F401
from fusezap.tensor import Tensor, manual_seed, no_grad  # noqa: F401
from fusezap.shape.symbolic import Var  # noqa: F401
from fusezap.jit import Jit, bind, bound  # noqa: F401
import sys as _sys, types as _types

class _JitModule(_types.ModuleType):
  """`fz.jit` is the fusezap.jit module made callable, so `@fz.jit` works and `import fusezap.jit` still gets the module."""
  def __call__(self, fn): return self.jit(fn)
_sys.modules["fusezap.jit"].__class__ = _JitModule
