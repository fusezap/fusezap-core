"""Token sampling on the host (SPEC §11.3): greedy, temperature, top-k, top-p, repetition penalty, seeded RNG.

  s = Sampler(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.1, seed=0)
  tok = s(logits, history)          # logits: (vocab,) float array; history: token ids seen so far

Order (as HF's generate): repetition penalty -> temperature -> top-k -> top-p -> softmax -> draw.
temperature == 0 (or top_k == 1) is greedy: argmax, first index on ties (like torch.argmax).
Repetition penalty follows HF (CTRL): a seen token's logit is divided by the penalty if positive, multiplied if negative.
top_p keeps the smallest set of most likely tokens whose probability reaches top_p (always at least one), like HF's
TopPLogitsWarper. A seeded `numpy.random.Generator` per Sampler makes draws reproducible.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np

def apply_repetition_penalty(logits: np.ndarray, history: Sequence[int], penalty: float) -> np.ndarray:
  if penalty == 1.0 or len(history) == 0: return logits
  out = logits.copy()
  idx = np.unique(np.asarray(history, np.int64))
  v = out[..., idx]
  out[..., idx] = np.where(v > 0, v / penalty, v * penalty)
  return out

def top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
  """Keep the k largest (ties at the k-th value are kept, like HF); the rest become -inf."""
  if k <= 0 or k >= logits.shape[-1]: return logits
  kth = np.partition(logits, -k, axis=-1)[..., -k, None]
  return np.where(logits < kth, -np.inf, logits)

def softmax(x: np.ndarray) -> np.ndarray:
  x = np.asarray(x, np.float64)
  e = np.exp(x - np.max(x, axis=-1, keepdims=True))
  return e / e.sum(-1, keepdims=True)

def top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
  """Nucleus filter: drop tokens outside the smallest top set whose cumulative probability reaches p."""
  if p >= 1.0: return logits
  order = np.argsort(-logits, axis=-1, kind="stable")
  probs = np.take_along_axis(softmax(logits), order, -1)
  cum = np.cumsum(probs, -1)
  drop_sorted = (cum - probs) >= p          # tokens whose preceding mass already reaches p; the top one always stays
  drop = np.zeros_like(drop_sorted)
  np.put_along_axis(drop, order, drop_sorted, -1)
  return np.where(drop, -np.inf, logits)

@dataclass
class Sampler:
  temperature: float = 0.0
  top_k: int = 0
  top_p: float = 1.0
  repetition_penalty: float = 1.0
  seed: int | None = None
  rng: np.random.Generator = field(init=False, repr=False)
  def __post_init__(self):
    assert self.temperature >= 0 and 0 < self.top_p <= 1 and self.repetition_penalty > 0, f"bad sampling params {self}"
    self.rng = np.random.default_rng(self.seed)
  @property
  def greedy(self) -> bool: return self.temperature == 0 or self.top_k == 1
  def reseed(self, seed: int | None) -> None: self.seed, self.rng = seed, np.random.default_rng(seed)
  def probs(self, logits: np.ndarray, history: Sequence[int] = ()) -> np.ndarray:
    """The distribution a draw samples from (float64, sums to 1). Greedy: one-hot on the argmax."""
    lg = apply_repetition_penalty(np.asarray(logits, np.float32), history, self.repetition_penalty).astype(np.float64)
    if self.greedy:
      out = np.zeros_like(lg)
      out[np.argmax(lg)] = 1.0
      return out
    lg = top_p_filter(top_k_filter(lg / self.temperature, self.top_k), self.top_p)
    return softmax(lg)
  def __call__(self, logits: np.ndarray, history: Sequence[int] = ()) -> int:
    if self.greedy: return int(np.argmax(apply_repetition_penalty(np.asarray(logits, np.float32), history, self.repetition_penalty)))
    p = self.probs(logits, history)
    return int(np.searchsorted(np.cumsum(p), self.rng.random() * p.sum(), side="right").clip(0, len(p) - 1))

def greedy(logits: np.ndarray) -> int: return int(np.argmax(logits))
