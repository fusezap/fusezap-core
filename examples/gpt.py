"""Small char-level GPT on a tiny built-in text (no download). `python examples/gpt.py` trains it."""
from __future__ import annotations
import math, numpy as np
from fusezap import Tensor, manual_seed
from fusezap.nn import Linear, Embedding, LayerNorm, get_parameters
from fusezap.optim import AdamW

TEXT = ("the quick brown fox jumps over the lazy dog. " * 4 + "to be, or not to be, that is the question. " * 4 +
        "all that glitters is not gold. " * 4 + "a stitch in time saves nine. " * 4)
CHARS = sorted(set(TEXT))
STOI = {c: i for i, c in enumerate(CHARS)}
def encode(s: str) -> np.ndarray: return np.array([STOI[c] for c in s], dtype=np.int32)

class Block:
  def __init__(self, d: int, h: int):
    self.h = h
    self.ln1, self.ln2 = LayerNorm(d), LayerNorm(d)
    self.qkv, self.proj = Linear(d, 3 * d), Linear(d, d)
    self.fc, self.out = Linear(d, 4 * d), Linear(4 * d, d)
  def __call__(self, x: Tensor) -> Tensor:
    B, T, C = x.shape
    q, k, v = [t.reshape(B, T, self.h, C // self.h).transpose(1, 2) for t in self.qkv(self.ln1(x)).chunk(3, axis=-1)]
    a = q.scaled_dot_product_attention(k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
    x = x + self.proj(a)
    return x + self.out(self.fc(self.ln2(x)).gelu())

class GPT:
  def __init__(self, vocab: int, d: int = 64, h: int = 4, L: int = 2, T: int = 32):
    self.tok, self.pos = Embedding(vocab, d), Embedding(T, d)
    self.blocks = [Block(d, h) for _ in range(L)]
    self.ln_f, self.head = LayerNorm(d), Linear(d, vocab, bias=False)
  def __call__(self, idx: Tensor) -> Tensor:
    x = self.tok(idx) + self.pos(Tensor.arange(idx.shape[1], device=idx.device))
    for b in self.blocks: x = b(x)
    return self.head(self.ln_f(x))
  def loss(self, idx: Tensor, tgt: Tensor) -> Tensor:
    lg = self(idx)
    return lg.reshape(-1, lg.shape[-1]).cross_entropy(tgt.reshape(-1))

def get_batch(data: np.ndarray, B: int, T: int, rng: np.random.Generator):
  ix = rng.integers(0, len(data) - T - 1, B)
  return np.stack([data[i:i + T] for i in ix]), np.stack([data[i + 1:i + T + 1] for i in ix])

if __name__ == "__main__":
  manual_seed(0)
  data, rng = encode(TEXT), np.random.default_rng(0)
  m = GPT(len(CHARS))
  opt = AdamW(get_parameters(m), lr=3e-3)
  for step in range(100):
    x, y = get_batch(data, 8, 32, rng)
    opt.zero_grad()
    loss = m.loss(Tensor(x), Tensor(y))
    loss.backward()
    opt.step()
    if step % 10 == 0: print(f"step {step} loss {loss.item():.4f}")
