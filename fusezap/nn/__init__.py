"""Neural network layers (SPEC §9). Implemented by WP6.

Modules are plain classes with Tensor attributes and `__call__(x)`; no base class is required.
Initializations and defaults match torch.nn so parity tests can copy weights:
  Linear(in_features, out_features, bias=True)            weight: (out, in)
  Embedding(num_embeddings, embedding_dim)                 weight: (num, dim)
  LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)
  RMSNorm(dim, eps=1e-6)
  Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True)   weight: (out, in, kh, kw)
  GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)
  Dropout(p=0.5)
Functions:
  scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
  get_parameters(obj) -> list[Tensor]            recursive walk over attributes, lists, tuples and dicts
  get_state_dict(obj) -> dict[str, Tensor]       dotted names, matching torch naming for the same structure
  load_state_dict(obj, state: dict[str, Tensor | np.ndarray], strict=True)
"""
from __future__ import annotations
import math
import numpy as np
from fusezap.tensor import Tensor

class Linear:
  def __init__(self, in_features: int, out_features: int, bias: bool = True):
    k = 1 / math.sqrt(in_features)
    self.weight = Tensor.uniform(out_features, in_features, low=-k, high=k, requires_grad=True)
    self.bias = Tensor.uniform(out_features, low=-k, high=k, requires_grad=True) if bias else None
  def __call__(self, x: Tensor) -> Tensor: return x.linear(self.weight, self.bias)

class Embedding:
  def __init__(self, num_embeddings: int, embedding_dim: int):
    self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
    self.weight = Tensor.randn(num_embeddings, embedding_dim, requires_grad=True)
  def __call__(self, idx: Tensor) -> Tensor:
    oh = (idx.unsqueeze(-1) == Tensor.arange(self.num_embeddings, device=idx.device)).cast(self.weight.dtype)
    return oh @ self.weight

class LayerNorm:
  def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.eps, self.elementwise_affine = eps, elementwise_affine
    self.weight = Tensor.ones(*self.normalized_shape, requires_grad=True) if elementwise_affine else None
    self.bias = Tensor.zeros(*self.normalized_shape, requires_grad=True) if elementwise_affine else None
  def __call__(self, x: Tensor) -> Tensor:
    axes = tuple(range(-len(self.normalized_shape), 0))
    y = x.layernorm(axes[0], self.eps) if len(axes) == 1 else _ln_multi(x, axes, self.eps)
    return y * self.weight + self.bias if self.elementwise_affine else y

def _ln_multi(x: Tensor, axes, eps: float) -> Tensor:
  d = x - x.mean(axes, keepdim=True)
  return d * ((d * d).mean(axes, keepdim=True) + eps).rsqrt()

class RMSNorm:
  def __init__(self, dim: int, eps: float = 1e-6):
    self.eps, self.weight = eps, Tensor.ones(dim, requires_grad=True)
  def __call__(self, x: Tensor) -> Tensor:
    xf = x._upcast()
    return (xf * (xf * xf).mean(-1, keepdim=True).add(self.eps).rsqrt()).cast(x.dtype) * self.weight.cast(x.dtype)

class Conv2d:
  """torch.nn.Conv2d (groups=1, padding_mode='zeros'): weight (out, in, kh, kw), bias (out,), both U(-1/sqrt(fan_in),
  1/sqrt(fan_in)) like torch's kaiming_uniform(a=sqrt(5)) default."""
  def __init__(self, in_channels: int, out_channels: int, kernel_size, stride=1, padding=0, dilation=1, groups: int = 1,
               bias: bool = True):
    if groups != 1: raise NotImplementedError("Conv2d: groups != 1 is not supported")
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    self.stride, self.padding, self.dilation = stride, padding, dilation
    k = 1 / math.sqrt(in_channels * kh * kw)
    self.weight = Tensor.uniform(out_channels, in_channels, kh, kw, low=-k, high=k, requires_grad=True)
    self.bias = Tensor.uniform(out_channels, low=-k, high=k, requires_grad=True) if bias else None
  def __call__(self, x: Tensor) -> Tensor: return x.conv2d(self.weight, self.bias, self.stride, self.padding, self.dilation)

class GroupNorm:
  """torch.nn.GroupNorm(num_groups, num_channels, eps=1e-5, affine=True): weight ones, bias zeros."""
  def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True):
    assert num_channels % num_groups == 0, f"GroupNorm: {num_channels} channels do not split into {num_groups} groups"
    self.num_groups, self.eps = num_groups, eps
    self.weight = Tensor.ones(num_channels, requires_grad=True) if affine else None
    self.bias = Tensor.zeros(num_channels, requires_grad=True) if affine else None
  def __call__(self, x: Tensor) -> Tensor: return x.group_norm(self.num_groups, self.weight, self.bias, self.eps)

class Dropout:
  def __init__(self, p: float = 0.5): self.p = p
  def __call__(self, x: Tensor) -> Tensor: return x.dropout(self.p)

def _walk(obj, prefix: str, out: dict, stack: set):
  if isinstance(obj, Tensor):  # shared tensors keep every name, like torch state_dict()
    out[prefix] = obj
    return
  if isinstance(obj, (type, str, bytes, int, float, bool)) or obj is None or id(obj) in stack: return  # stack: break back-reference cycles
  stack.add(id(obj))
  if isinstance(obj, (list, tuple)):
    for i, v in enumerate(obj): _walk(v, f"{prefix}.{i}" if prefix else str(i), out, stack)
  elif isinstance(obj, dict):
    for k, v in obj.items(): _walk(v, f"{prefix}.{k}" if prefix else str(k), out, stack)
  elif hasattr(obj, "__dict__"):
    for k, v in vars(obj).items():
      if not k.startswith("_"): _walk(v, f"{prefix}.{k}" if prefix else k, out, stack)
  stack.discard(id(obj))

def get_state_dict(obj) -> dict[str, Tensor]:
  out: dict[str, Tensor] = {}
  _walk(obj, "", out, set())
  return out

def get_parameters(obj) -> list[Tensor]:
  seen: set[int] = set()
  return [t for t in get_state_dict(obj).values() if t.requires_grad and not (id(t) in seen or seen.add(id(t)))]

def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, attn_mask: Tensor | None = None,
                                 dropout_p: float = 0.0, is_causal: bool = False) -> Tensor:
  return query.scaled_dot_product_attention(key, value, attn_mask, dropout_p, is_causal)

def load_state_dict(obj, state: dict, strict: bool = True) -> None:
  mine = get_state_dict(obj)
  if strict:
    missing, extra = sorted(set(mine) - set(state)), sorted(set(state) - set(mine))
    assert not missing and not extra, f"load_state_dict: missing {missing}, unexpected {extra}"
  for k, t in mine.items():
    if k not in state: continue
    v = state[k]
    if not isinstance(v, Tensor): v = Tensor(np.ascontiguousarray(v), device=t.device, dtype=t.dtype)
    assert tuple(v.shape) == tuple(t.shape), f"shape mismatch for {k}: {v.shape} vs {t.shape}"
    t.assign(v.to(t.device).detach()).realize()
