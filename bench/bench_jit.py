"""Per-step overhead of a GPT decode step (KV cache, symbolic sequence length) with and without fusezap.jit.

  uv run python bench/bench_jit.py [--device CPU|METAL] [--steps N]

The decode step uses the char-level GPT from examples/gpt.py: one token in, the KV cache (fixed max length, updated with
a WHERE at the symbolic position) in and out. Attention reads the cache shrunk to `pos + 1` rows, so the sequence
length is a symbolic Var and one capture serves every step.
"Overhead" is the host time of one step call (before synchronizing) minus the time kernels execute inside it (CPU runs
kernels synchronously; METAL only enqueues them), i.e. the Python cost of building the graph, scheduling, cache lookups
and launching. "total" also includes the final device synchronize.
"""
from __future__ import annotations
import argparse, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fusezap import Tensor, Device, manual_seed
from fusezap.dtype import dtypes
from fusezap.shape.symbolic import Var
from fusezap.jit import jit, bind, bound
import fusezap.realize as _realize
from examples.gpt import GPT, CHARS

def decode_step(m: GPT, tok: Tensor, caches: list[Tensor], pos: Var) -> tuple[Tensor, list[Tensor]]:
  """tok: (B, 1) int32. caches: [k0, v0, k1, v1, ...], each (B, h, Tmax, hd). Returns logits (B, vocab) and new caches."""
  B = tok.shape[0]
  Tmax, d = m.pos.weight.shape
  x = m.tok(tok) + m.pos.weight.shrink(((pos, pos + 1), None)).reshape(1, 1, d)
  sel = (Tensor.arange(Tmax, device=tok.device) == Tensor(pos, device=tok.device)).reshape(1, 1, Tmax, 1)
  new = []
  for i, blk in enumerate(m.blocks):
    h, hd = blk.h, d // blk.h
    q, k, v = [t.reshape(B, 1, h, hd).transpose(1, 2) for t in blk.qkv(blk.ln1(x)).chunk(3, axis=-1)]
    kc, vc = [sel.where(t.expand(B, h, Tmax, hd), c) for t, c in ((k, caches[2 * i]), (v, caches[2 * i + 1]))]
    new += [kc, vc]
    ks, vs = kc.shrink((None, None, (0, pos + 1), None)), vc.shrink((None, None, (0, pos + 1), None))
    a = ((q @ ks.transpose(-1, -2)) * (1.0 / math.sqrt(hd))).softmax(-1) @ vs
    x = x + blk.proj(a.transpose(1, 2).reshape(B, 1, d))
    x = x + blk.out(blk.fc(blk.ln2(x)).gelu())
  return m.head(m.ln_f(x)).reshape(B, -1), new

def make_model(device: str, B: int = 1, L: int = 2, d: int = 64, h: int = 4, T: int = 32):
  manual_seed(0)
  m = GPT(len(CHARS), d=d, h=h, L=L, T=T)
  from fusezap.nn import get_parameters
  for p in get_parameters(m):
    p.lazydata = Tensor(p.lazydata).to(device).lazydata if p.device != device else p.lazydata
    p.requires_grad = False
    p.realize()
  caches = [Tensor.zeros(B, h, T, d // h, device=device).contiguous().realize() for _ in range(2 * L)]
  return m, caches

class _Timer:
  """Sum of kernel execution time spent inside the host call. Only synchronous devices (CPU: the kernel runs inside
  Program.__call__) need it; async devices (METAL) only enqueue, so their host time already is pure overhead."""
  def __init__(self, device: str):
    rt = Device[device].runtime
    self.cls, self.total, self.sync = getattr(rt, "func", rt), 0.0, device.split(":")[0] == "CPU"
    self.orig = self.cls.__call__
  def __enter__(self):
    if not self.sync: return self
    t = self
    def call(prg, *a, **kw):
      kw["wait"] = True
      et = t.orig(prg, *a, **kw)
      t.total += et or 0.0
      return et
    self.cls.__call__ = call
    return self
  def __exit__(self, *a): self.cls.__call__ = self.orig

def run(step, m, caches, toks, pos: Var, device: str, use_jit: bool):
  """Decode len(toks) tokens (teacher forced). Returns (logits per step, host seconds per step, device seconds per step)."""
  outs, host, devt, tot = [], [], [], []
  Device[device].synchronize()
  for i, t in enumerate(toks):
    tok = Tensor(np.array([[t]], np.int32), device=device).realize()
    with _Timer(device) as tm:
      st = time.perf_counter()
      if use_jit: lg, caches = step(tok, caches, bind(pos, i))
      else:
        with bound({pos: i}):
          lg, caches = step(tok, caches, pos)
          _realize.realize([t.lazydata for t in [lg] + caches])   # one schedule for the whole step, like the jit
      host.append(time.perf_counter() - st)
    devt.append(tm.total)
    Device[device].synchronize()
    tot.append(time.perf_counter() - st)
    outs.append(lg.numpy())
  return outs, host, devt, tot

def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--device", default=Device.DEFAULT)
  ap.add_argument("--steps", type=int, default=32)
  args = ap.parse_args()
  dev, T = args.device.upper(), max(args.steps, 4)
  m, caches0 = make_model(dev, T=T)
  pos = Var("pos", 0, T - 1)
  toks = list(np.random.default_rng(0).integers(0, len(CHARS), T))
  plain = lambda tok, caches, p: decode_step(m, tok, caches, p)
  jstep = jit(lambda tok, caches, p: decode_step(m, tok, caches, p))
  # warm up the compile caches with one non-jit pass, then time both
  run(plain, m, caches0, toks[:4], pos, dev, False)
  ref, h0, d0, t0 = run(plain, m, caches0, toks, pos, dev, False)
  got, h1, d1, t1 = run(jstep, m, caches0, toks, pos, dev, True)
  for i, (a, b) in enumerate(zip(ref, got)):
    np.testing.assert_allclose(b, a, rtol=1e-5, atol=1e-6, err_msg=f"step {i}: jit output differs from non-jit")
  skip = 3  # jit steps 0, 1 are the eager and capture calls
  ov0 = np.median([h - d for h, d in zip(h0[skip:], d0[skip:])])
  ov1 = np.median([h - d for h, d in zip(h1[skip:], d1[skip:])])
  print(f"device {dev}, GPT L=2 d=64 h=4, {T} decode steps, symbolic pos")
  print(f"  without jit: {np.median(t0[skip:])*1e3:8.3f} ms/step total, {ov0*1e3:8.3f} ms/step overhead")
  print(f"  with jit:    {np.median(t1[skip:])*1e3:8.3f} ms/step total, {ov1*1e3:8.3f} ms/step overhead")
  print(f"  overhead reduction: {ov0/ov1:.1f}x  (target >= 5x); outputs match: yes")
  assert ov0 / ov1 >= 5, f"per-step overhead only {ov0/ov1:.1f}x lower with jit"

if __name__ == "__main__": main()
