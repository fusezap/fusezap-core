"""Reverse-mode autograd over differentiable primitive Functions (SPEC §9). Implemented by WP2.

Contract:
  class Function:
    def __init__(self, device, *tensors)          records parents and needs_input_grad
    def forward(self, *lbs: LazyBuffer, **kw) -> LazyBuffer
    def backward(self, grad: LazyBuffer) -> LazyBuffer | None | tuple[LazyBuffer | None, ...]
    @classmethod apply(cls, *tensors, **kw) -> Tensor

Required Functions (each maps to one primitive, plus the backward rules):
  Cast, Neg, Reciprocal, Sin, Relu, Log, Exp, Sqrt, Sigmoid(optional fused), Less/Eq (no grad),
  Add, Sub, Mul, Div, Where, Max (binary), Sum, MaxReduce, Expand, Reshape, Permute, Pad, Shrink, Flip,
  Contiguous, Copy (device transfer).
Composite Tensor methods (softmax, matmul, ...) need no Function: they differentiate through these.

Extra Functions beyond the required list: Exp2, Log2 (exact base-2 forms), IDiv, Mod (ints, no grad).
`differentiable = False` Functions produce outputs with requires_grad=False.
Backward builds lazy graph (LazyBuffer ops), so forward and backward are scheduled together.
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any
from fusezap.dtype import dtypes
from fusezap.helpers import argsort, LOG2E, LN2
from fusezap.ops import UnaryOps, BinaryOps, TernaryOps, ReduceOps
from fusezap.lazy import LazyBuffer

if TYPE_CHECKING:
  from fusezap.tensor import Tensor

class Function:
  differentiable: bool = True
  def __init__(self, device: str, *tensors: "Tensor"):
    self.device = device
    self.needs_input_grad = [t.requires_grad for t in tensors]
    self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False
    if not self.differentiable: self.requires_grad = False
    if self.requires_grad: self.parents = tensors
  def forward(self, *lbs: LazyBuffer, **kw) -> LazyBuffer: raise NotImplementedError(f"forward not implemented for {type(self)}")
  def backward(self, grad: LazyBuffer) -> Any: raise RuntimeError(f"backward not implemented for {type(self)}")
  @classmethod
  def apply(cls, *x: "Tensor", **kw) -> "Tensor":
    from fusezap.tensor import Tensor
    ctx = cls(x[0].device, *x)
    out = ctx.forward(*[t.lazydata for t in x], **kw)
    rg = False if Tensor.no_grad else ctx.requires_grad
    ret = Tensor(out, device=ctx.device, requires_grad=rg)
    if rg and ctx.requires_grad: ret._ctx = ctx
    return ret

def _c(lb: LazyBuffer, v) -> LazyBuffer: return lb.const(v)

# ---------------------------------------------------------------- unary

class Contiguous(Function):
  def forward(self, x): return x.contiguous()
  def backward(self, grad): return grad

class Cast(Function):
  def forward(self, x, dtype, bitcast=False):
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.cast(dtype, bitcast)
  def backward(self, grad):
    if self.bitcast or not self.input_dtype.is_float: return None
    return grad.cast(self.input_dtype)

class Neg(Function):
  def forward(self, x): return x.e(UnaryOps.NEG)
  def backward(self, grad): return grad.e(UnaryOps.NEG)

class Reciprocal(Function):
  def forward(self, x):
    self.ret = x.e(UnaryOps.RECIP)
    return self.ret
  def backward(self, grad): return grad.e(UnaryOps.NEG).e(BinaryOps.MUL, self.ret).e(BinaryOps.MUL, self.ret)

class Sin(Function):
  def forward(self, x):
    self.x = x
    return x.e(UnaryOps.SIN)
  def backward(self, grad):  # cos(x) = sin(pi/2 - x)
    return _c(self.x, math.pi / 2).e(BinaryOps.ADD, self.x.e(UnaryOps.NEG)).e(UnaryOps.SIN).e(BinaryOps.MUL, grad)

class Relu(Function):
  def forward(self, x):
    self.ret = x.e(BinaryOps.MAX, _c(x, 0))
    return self.ret
  def backward(self, grad):
    return _c(self.ret, 0).e(BinaryOps.CMPLT, self.ret).cast(grad.dtype).e(BinaryOps.MUL, grad)

class Log(Function):
  def forward(self, x):
    self.x = x
    return x.e(UnaryOps.LOG2).e(BinaryOps.MUL, _c(x, LN2))
  def backward(self, grad): return grad.e(BinaryOps.MUL, self.x.e(UnaryOps.RECIP))

class Log2(Function):
  def forward(self, x):
    self.x = x
    return x.e(UnaryOps.LOG2)
  def backward(self, grad): return grad.e(BinaryOps.MUL, self.x.e(BinaryOps.MUL, _c(self.x, LN2)).e(UnaryOps.RECIP))

class Exp(Function):
  def forward(self, x):
    self.ret = x.e(BinaryOps.MUL, _c(x, LOG2E)).e(UnaryOps.EXP2)
    return self.ret
  def backward(self, grad): return self.ret.e(BinaryOps.MUL, grad)

class Exp2(Function):
  def forward(self, x):
    self.ret = x.e(UnaryOps.EXP2)
    return self.ret
  def backward(self, grad): return self.ret.e(BinaryOps.MUL, _c(self.ret, LN2)).e(BinaryOps.MUL, grad)

class Sqrt(Function):
  def forward(self, x):
    self.ret = x.e(UnaryOps.SQRT)
    return self.ret
  def backward(self, grad): return grad.e(BinaryOps.MUL, self.ret.e(BinaryOps.MUL, _c(self.ret, 2)).e(UnaryOps.RECIP))

class Sigmoid(Function):
  def forward(self, x):
    self.ret = _c(x, 1).e(BinaryOps.ADD, x.e(BinaryOps.MUL, _c(x, -LOG2E)).e(UnaryOps.EXP2)).e(UnaryOps.RECIP)
    return self.ret
  def backward(self, grad):
    return self.ret.e(BinaryOps.MUL, _c(self.ret, 1).e(BinaryOps.ADD, self.ret.e(UnaryOps.NEG))).e(BinaryOps.MUL, grad)

# ---------------------------------------------------------------- binary / ternary

class Less(Function):
  differentiable = False
  def forward(self, x, y): return x.e(BinaryOps.CMPLT, y)

class Eq(Function):
  differentiable = False
  def forward(self, x, y): return x.e(BinaryOps.CMPEQ, y)

class IDiv(Function):
  differentiable = False
  def forward(self, x, y): return x.e(BinaryOps.IDIV, y)

class Mod(Function):
  differentiable = False
  def forward(self, x, y): return x.e(BinaryOps.MOD, y)

class Add(Function):
  def forward(self, x, y): return x.e(BinaryOps.ADD, y)
  def backward(self, grad):
    return grad if self.needs_input_grad[0] else None, grad if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x, y): return x.e(BinaryOps.ADD, y.e(UnaryOps.NEG))
  def backward(self, grad):
    return grad if self.needs_input_grad[0] else None, grad.e(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x.e(BinaryOps.MUL, y)
  def backward(self, grad):
    return (self.y.e(BinaryOps.MUL, grad) if self.needs_input_grad[0] else None,
            self.x.e(BinaryOps.MUL, grad) if self.needs_input_grad[1] else None)

class Div(Function):
  def forward(self, x, y):
    self.x, self.y = x, y
    return x.e(BinaryOps.MUL, y.e(UnaryOps.RECIP))
  def backward(self, grad):
    ry = self.y.e(UnaryOps.RECIP)
    return (grad.e(BinaryOps.MUL, ry) if self.needs_input_grad[0] else None,
            grad.e(UnaryOps.NEG).e(BinaryOps.MUL, self.x).e(BinaryOps.MUL, ry).e(BinaryOps.MUL, ry) if self.needs_input_grad[1] else None)

class Where(Function):
  def forward(self, c, x, y):
    self.c = c
    return c.e(TernaryOps.WHERE, x, y)
  def backward(self, grad):
    z = _c(grad, 0)
    return (None, self.c.e(TernaryOps.WHERE, grad, z) if self.needs_input_grad[1] else None,
            self.c.e(TernaryOps.WHERE, z, grad) if self.needs_input_grad[2] else None)

class Max(Function):
  """Binary max; like torch.maximum, a tie sends half the gradient to each side."""
  def forward(self, x, y):
    self.x, self.y = x, y
    return x.e(BinaryOps.MAX, y)
  def backward(self, grad):
    gt = self.y.e(BinaryOps.CMPLT, self.x).cast(grad.dtype)
    eq = self.x.e(BinaryOps.CMPEQ, self.y).cast(grad.dtype).e(BinaryOps.MUL, _c(grad, 0.5))
    lt = self.x.e(BinaryOps.CMPLT, self.y).cast(grad.dtype)
    return (gt.e(BinaryOps.ADD, eq).e(BinaryOps.MUL, grad) if self.needs_input_grad[0] else None,
            lt.e(BinaryOps.ADD, eq).e(BinaryOps.MUL, grad) if self.needs_input_grad[1] else None)

class Pow(Function):
  """b ** e with torch's backward rules. forward(b, e, r) passes through r = b ** e (computed by the frontend).
  grad_b = 0 where e == 0, else e * b**(e-1). grad_e = 0 where b == 0 and e >= 0, else r * ln(b) (NaN for b < 0)."""
  def forward(self, b, e, r):
    self.b, self.e, self.r = b, e, r
    return r
  def backward(self, grad):
    from fusezap.tensor import Tensor, _pow_value, no_grad
    with no_grad():
      b, e, r, g = Tensor(self.b), Tensor(self.e), Tensor(self.r), Tensor(grad)
      gb = (e == 0).where(0.0, g * e * _pow_value(b, e - 1.0)) if self.needs_input_grad[0] else None
      ge = (b == 0).mul(e >= 0).where(0.0, g * r * b.log()) if self.needs_input_grad[1] else None
    return (gb.lazydata if gb is not None else None, ge.lazydata if ge is not None else None, None)

# ---------------------------------------------------------------- reduce

class Sum(Function):
  def forward(self, x, axis):
    self.input_shape = x.shape
    return x.r(ReduceOps.SUM, axis)
  def backward(self, grad): return grad.expand(self.input_shape)

class MaxReduce(Function):
  """Gradient is split evenly among the maxima (like torch.amax)."""
  def forward(self, x, axis):
    self.x, self.axis = x, axis
    self.ret = x.r(ReduceOps.MAX, axis)
    return self.ret
  def backward(self, grad):
    is_max = self.x.e(BinaryOps.CMPEQ, self.ret.expand(self.x.shape)).cast(grad.dtype)
    div = is_max.r(ReduceOps.SUM, self.axis).expand(self.x.shape)
    return is_max.e(BinaryOps.MUL, div.e(UnaryOps.RECIP)).e(BinaryOps.MUL, grad.expand(self.x.shape))

# ---------------------------------------------------------------- movement

class Expand(Function):
  def forward(self, x, shape):
    self.input_shape = x.shape
    return x.expand(shape)
  def backward(self, grad):
    axes = tuple(i for i, (s, g) in enumerate(zip(self.input_shape, grad.shape)) if s != g)
    return grad.r(ReduceOps.SUM, axes) if axes else grad

class Reshape(Function):
  def forward(self, x, shape):
    self.input_shape = x.shape
    return x.reshape(shape)
  def backward(self, grad): return grad.reshape(self.input_shape)

class Permute(Function):
  def forward(self, x, order):
    self.order = tuple(order)
    return x.permute(self.order)
  def backward(self, grad): return grad.permute(tuple(argsort(list(self.order))))

class Pad(Function):
  def forward(self, x, arg):
    self.narg = tuple((b, s + b) for s, (b, _) in zip(x.shape, arg))
    return x.pad(arg)
  def backward(self, grad): return grad.shrink(self.narg)

class Shrink(Function):
  def forward(self, x, arg):
    self.narg = tuple((b, s - e) for s, (b, e) in zip(x.shape, arg))
    return x.shrink(arg)
  def backward(self, grad): return grad.pad(self.narg)

class Flip(Function):
  def forward(self, x, arg):
    self.arg = tuple(arg)
    return x.flip(self.arg)
  def backward(self, grad): return grad.flip(self.arg)

class Copy(Function):
  def forward(self, x, device):
    self.src_device = x.device
    self.device = device
    return x.copy_to_device(device)
  def backward(self, grad): return grad.copy_to_device(self.src_device)
