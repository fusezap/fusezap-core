"""Small shared utilities. Owned by the integrator; builders must not edit."""
from __future__ import annotations
import functools, hashlib, math, os, pathlib
from typing import Iterable, TypeVar

T = TypeVar("T")

@functools.lru_cache(maxsize=None)
def getenv(key: str, default=0):
  """Read an env var, coerced to the type of `default`."""
  return type(default)(os.getenv(key, default))

DEBUG = getenv("FZ_DEBUG", 0)          # 1: kernel count + time, 2: schedule, 3: kernel AST, 4: generated source
CACHE_DIR = pathlib.Path(os.getenv("FZ_CACHE_DIR", pathlib.Path.home() / ".cache" / "fusezap"))

def prod(xs: Iterable[T], start=1) -> T:
  return functools.reduce(lambda a, b: a * b, xs, start)

def dedup(xs: Iterable[T]) -> list[T]:
  return list(dict.fromkeys(xs))

def all_same(xs: list) -> bool:
  return all(x == xs[0] for x in xs)

def argsort(xs: list) -> list[int]:
  return sorted(range(len(xs)), key=xs.__getitem__)

def flatten(xss: Iterable[Iterable[T]]) -> list[T]:
  return [x for xs in xss for x in xs]

def round_up(n: int, multiple: int) -> int:
  return (n + multiple - 1) // multiple * multiple

def stable_hash(s: str | bytes) -> str:
  return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()[:32]

def canonicalize_axis(axis: int, ndim: int) -> int:
  if not -ndim <= axis < max(ndim, 1): raise IndexError(f"axis {axis} out of range for ndim {ndim}")
  return axis % ndim if ndim else 0

LOG2E, LN2 = math.log2(math.e), math.log(2)
