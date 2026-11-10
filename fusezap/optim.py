"""Optimizers (SPEC §9). Implemented by WP6.

  Optimizer(params): zero_grad(), step() (updates through Tensor.assign, then realizes params and state
                     in one schedule)
  SGD(params, lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False)
  AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)   == torch.optim.AdamW
"""
from __future__ import annotations
from fusezap.tensor import Tensor

class Optimizer:
  def __init__(self, params):
    self.params = [p for p in params]
    for p in self.params: p.requires_grad = True
  def zero_grad(self):
    for p in self.params: p.grad = None
  def _realize(self, extra: list[Tensor]):
    from fusezap.realize import realize
    lbs = [t.lazydata for t in self.params + extra if not t.lazydata.is_unrealized_const() and not t.lazydata.is_realized()]
    if lbs: realize(lbs)
  def step(self): raise NotImplementedError

class SGD(Optimizer):
  def __init__(self, params, lr=1e-3, momentum=0.0, weight_decay=0.0, nesterov=False):
    super().__init__(params)
    self.lr, self.momentum, self.weight_decay, self.nesterov = lr, momentum, weight_decay, nesterov
    self.b = [None] * len(self.params)
  def step(self):
    with Tensor.inference():
      extra = []
      for i, p in enumerate(self.params):
        if p.grad is None: continue
        g = p.grad.detach()
        if self.weight_decay: g = g + p.detach() * self.weight_decay
        if self.momentum:
          b = g if self.b[i] is None else self.b[i] * self.momentum + g
          self.b[i] = b.contiguous()
          extra.append(self.b[i])
          g = g + b * self.momentum if self.nesterov else b
        p.assign(p.detach() - g * self.lr)
      self._realize(extra)

class AdamW(Optimizer):
  def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
    super().__init__(params)
    self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, weight_decay
    self.t = [0] * len(self.params)  # per-parameter step counts, like torch
    self.m = [Tensor.zeros(*p.shape, dtype=p.dtype, device=p.device).contiguous().realize() for p in self.params]
    self.v = [Tensor.zeros(*p.shape, dtype=p.dtype, device=p.device).contiguous().realize() for p in self.params]
  def step(self):
    with Tensor.inference():
      for i, p in enumerate(self.params):
        if p.grad is None: continue
        self.t[i] += 1
        bc1, bc2 = 1 - self.b1 ** self.t[i], 1 - self.b2 ** self.t[i]
        g, w = p.grad.detach(), p.detach()
        self.m[i].assign(self.m[i] * self.b1 + g * (1 - self.b1))
        self.v[i].assign(self.v[i] * self.b2 + g * g * (1 - self.b2))
        denom = self.v[i].sqrt() * (1 / bc2 ** 0.5) + self.eps
        p.assign(w * (1 - self.lr * self.wd) - (self.m[i] / denom) * (self.lr / bc1))
      self._realize(self.m + self.v)
