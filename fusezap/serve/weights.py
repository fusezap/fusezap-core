"""safetensors loading (SPEC §11.3): mmap'd numpy reader, dtype conversion, sharded checkpoints, HF Hub resolution.

  path = resolve_model("Qwen/Qwen2.5-0.5B-Instruct")      # local dir, or the HF cache (download only if allowed)
  st = load_safetensors(path)                               # {name: np.ndarray}, zero-copy views of an mmap
  w = to_dtype(st["model.embed_tokens.weight"], dtypes.float32)

bfloat16 tensors come back as uint16 bit arrays (fusezap's host bf16 format, SPEC §4); `to_dtype` converts them.
Nothing here imports torch or the `safetensors` package: the format is an 8-byte little-endian header length, a JSON
header {name: {dtype, shape, data_offsets}}, then the raw bytes.
"""
from __future__ import annotations
import json, mmap, os, struct
import numpy as np
from fusezap.dtype import DType, dtypes, to_np

ST_DTYPES: dict[str, np.dtype] = {"F64": np.dtype(np.float64), "F32": np.dtype(np.float32), "F16": np.dtype(np.float16),
  "BF16": np.dtype(np.uint16), "I64": np.dtype(np.int64), "I32": np.dtype(np.int32), "I16": np.dtype(np.int16),
  "I8": np.dtype(np.int8), "U8": np.dtype(np.uint8), "BOOL": np.dtype(np.bool_)}

class STArray(np.ndarray):
  """An ndarray view that remembers the safetensors dtype string (so BF16 uint16 bits are not mistaken for ints) and
  where its tensor lies in the file (st_span: (file, offset, bytes) of the whole tensor; slices keep it), for prefetch."""
  st_dtype: str = ""
  st_span: tuple = ()
  def __array_finalize__(self, obj): self.st_dtype, self.st_span = getattr(obj, "st_dtype", ""), getattr(obj, "st_span", ())

def read_safetensors(fn: str) -> dict[str, np.ndarray]:
  """One .safetensors file -> {name: array} as views of one read-only np.memmap (no copy until used)."""
  with open(fn, "rb") as f: n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n))
  header.pop("__metadata__", None)
  mm = np.memmap(fn, dtype=np.uint8, mode="r")
  out: dict[str, np.ndarray] = {}
  for k, v in header.items():
    s, e = v["data_offsets"]
    dt = ST_DTYPES[v["dtype"]]
    a = mm[8 + n + s:8 + n + e].view(dt).reshape(v["shape"]).view(STArray)
    a.st_dtype, a.st_span = v["dtype"], (fn, 8 + n + s, e - s)
    out[k] = a
  return out

_FDS: dict[str, int] = {}
def _fd(fn: str) -> int:
  if (fd := _FDS.get(fn)) is None:
    if len(_FDS) > 256: close_prefetch()
    _FDS[fn] = fd = os.open(fn, os.O_RDONLY)
  return fd

def span(a: np.ndarray) -> tuple | None:
  """(file, memmap, offset, bytes) of the file range an mmap'd array (or a slice of one: a TP rank's rows or columns)
  actually reads, None for arrays not backed by load_safetensors' memmaps."""
  if not getattr(a, "st_span", ()) or a.size == 0: return None
  mm, b = None, a
  while isinstance(b, np.ndarray): mm, b = (b if isinstance(b, np.memmap) else mm), b.base   # the root memmap: offset 0
  if mm is None or getattr(mm, "_mmap", None) is None or b is not mm._mmap: return None
  off = a.__array_interface__["data"][0] - mm.__array_interface__["data"][0]
  if off < 0 or any(st < 0 for st in a.strides): return None
  return a.st_span[0], mm, off, sum((n - 1) * st for n, st in zip(a.shape, a.strides)) + a.itemsize

def prefetch(arrays: list) -> int:
  """Ask the kernel to read these arrays' file ranges ahead (posix_fadvise WILLNEED: asynchronous, many requests in
  flight) and return the bytes asked for. A cold mmap otherwise faults its pages in one readahead window at a time: on
  the pods' network-backed overlay disk that ran at 10-30 MB/s (the first M3 run's 70B ranks loaded at 0.5-2 GB per
  minute once the concurrent download had evicted the weights from the page cache). Only the range a slice reads is
  asked for (a TP rank's q/k/v/gate/up rows: its half). No-op where posix_fadvise is missing (macOS) or FZ_PREFETCH=0."""
  if not hasattr(os, "posix_fadvise") or os.environ.get("FZ_PREFETCH", "1") == "0": return 0
  n = 0
  for a in arrays:
    if (sp := span(a)) is None: continue
    try: os.posix_fadvise(_fd(sp[0]), sp[2], sp[3], os.POSIX_FADV_WILLNEED); n += sp[3]
    except OSError: pass
  return n

def release(arrays: list) -> int:
  """Give back the host pages of these arrays' file ranges once they are on the device: unmap them from this process
  (madvise DONTNEED on the read-only mapping: the next access refaults them, so a later read stays correct) and drop
  them from the page cache (posix_fadvise DONTNEED, Linux). Without it a 70B rank kept every page it had read mapped
  and cached until the end of the load (the 26 Oct 2x H200 pod's container limit was reached at ~60 GB per rank).
  Returns the bytes released. FZ_RELEASE=0: keep the pages."""
  if os.environ.get("FZ_RELEASE", "1") == "0": return 0
  n, pg = 0, mmap.PAGESIZE
  for a in arrays:
    if (sp := span(a)) is None: continue
    fn, mm, off, nb = sp
    lo = off - off % pg
    try: mm._mmap.madvise(mmap.MADV_DONTNEED, lo, min(off + nb, len(mm._mmap)) - lo)
    except (OSError, ValueError, AttributeError): pass
    if hasattr(os, "posix_fadvise"):
      try: os.posix_fadvise(_fd(fn), off, nb, os.POSIX_FADV_DONTNEED)
      except OSError: pass
    n += nb
  return n

def close_prefetch() -> None:
  for fd in _FDS.values():
    try: os.close(fd)
    except OSError: pass
  _FDS.clear()

def _read(fn: str) -> str:
  try:
    with open(fn) as f: return f.read()
  except OSError: return ""
def host_mem() -> dict[str, float]:
  """Host memory in bytes (Linux; on macOS only the peak RSS): rss (this process), peak_rss, avail (MemAvailable),
  total (MemTotal), limit / used (the container's cgroup: memory.max / memory.current, or cgroup v1's
  memory.limit_in_bytes / memory.usage_in_bytes; inf when unlimited). Missing values are left out."""
  import resource, sys
  out: dict[str, float] = {"peak_rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
  for line in _read("/proc/self/status").splitlines():
    if line.startswith("VmRSS:"): out["rss"] = int(line.split()[1]) * 1024
  for line in _read("/proc/meminfo").splitlines():
    k, _, v = line.partition(":")
    if k in ("MemAvailable", "MemTotal"): out["avail" if k == "MemAvailable" else "total"] = int(v.split()[0]) * 1024
  for lim, cur in (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                   ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes")):
    if (v := _read(lim).strip()):
      out["limit"] = float("inf") if v == "max" or int(v) >= 1 << 60 else float(v)
      if (u := _read(cur).strip()).isdigit(): out["used"] = float(u)
      break
  return out
def host_mem_str(m: dict | None = None) -> str:
  m = host_mem() if m is None else m
  g = lambda k: "unlimited" if m[k] == float("inf") else f"{m[k] / 1e9:.1f} GB"
  return ", ".join(f"{name} {g(k)}" for k, name in (("rss", "rss"), ("peak_rss", "peak rss"), ("avail", "MemAvailable"),
                                                      ("limit", "cgroup limit"), ("used", "cgroup used")) if k in m)

def checkpoint_bytes(path: str) -> int:
  """Bytes of the .safetensors files of a model dir (symlinks followed: the HF cache's blobs), 0 if none."""
  try: return sum(os.path.getsize(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".safetensors"))
  except OSError: return 0

def load_safetensors(path: str) -> dict[str, np.ndarray]:
  """A model dir (single file or `model.safetensors.index.json` shards) or one .safetensors file."""
  if os.path.isfile(path): return read_safetensors(path)
  idx = os.path.join(path, "model.safetensors.index.json")
  if os.path.exists(idx):
    with open(idx) as f: wm: dict[str, str] = json.load(f)["weight_map"]
    out: dict[str, np.ndarray] = {}
    for shard in sorted(set(wm.values())): out.update(read_safetensors(os.path.join(path, shard)))
    missing = set(wm) - set(out)
    assert not missing, f"index lists tensors missing from the shards: {sorted(missing)[:5]}"
    return out
  fns = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
  assert fns, f"no .safetensors files in {path}"
  out = {}
  for fn in fns: out.update(read_safetensors(os.path.join(path, fn)))
  return out

def bf16_to_f32(bits: np.ndarray) -> np.ndarray: return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)
def f32_to_bf16(x: np.ndarray) -> np.ndarray:
  """Round to nearest even (like torch); NaN stays NaN."""
  u = np.ascontiguousarray(x, np.float32).view(np.uint32)
  r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
  return np.where(np.isnan(x), np.uint16(0x7FC0), r)

def is_bf16(a: np.ndarray) -> bool: return getattr(a, "st_dtype", "") == "BF16"

def to_dtype(a: np.ndarray, dt: DType) -> np.ndarray:
  """Host array in fusezap's host format for `dt` (bf16 -> uint16 bits). Accepts safetensors views (BF16 aware)."""
  src_bf16 = is_bf16(a)
  a = np.asarray(a)
  if dt == dtypes.bfloat16: return np.ascontiguousarray(a) if src_bf16 else f32_to_bf16(a.astype(np.float32))
  if src_bf16: a = bf16_to_f32(a)
  return np.ascontiguousarray(a.astype(to_np(dt), copy=False))

META_PATTERNS = ["*.json", "*.txt", "*.model", "*.jinja"]
IGNORE_PATTERNS = ["*/*", "consolidated*"]   # top level only: no original/ checkpoints, no Mistral consolidated.safetensors

def weight_patterns(meta_dir: str) -> list[str]:
  """The .safetensors files to fetch: the shards the index lists, else every top-level *.safetensors (minus IGNORE)."""
  idx = os.path.join(meta_dir, "model.safetensors.index.json")
  if not os.path.exists(idx): return ["*.safetensors"]
  with open(idx) as f: return sorted(set(json.load(f)["weight_map"].values()))

def resolve_model(name_or_path: str, allow_download: bool | None = None) -> str:
  """A local directory, or an HF Hub repo id resolved through the HF cache. Downloads only if allow_download (default:
  FZ_HF_DOWNLOAD=1) and the files are not cached. Metadata comes first; then only the safetensors the index lists, so
  repos that also ship other weight copies (Mistral's consolidated.safetensors) are not fetched twice."""
  if os.path.isdir(name_or_path): return name_or_path
  from huggingface_hub import snapshot_download
  def fetch(local: bool) -> str:
    meta = snapshot_download(name_or_path, local_files_only=local, allow_patterns=META_PATTERNS, ignore_patterns=IGNORE_PATTERNS)
    return snapshot_download(name_or_path, local_files_only=local, allow_patterns=META_PATTERNS + weight_patterns(meta),
                             ignore_patterns=IGNORE_PATTERNS)
  try: return fetch(True)
  except Exception:
    if allow_download is None: allow_download = os.environ.get("FZ_HF_DOWNLOAD", "0") == "1"
    if not allow_download: raise
    return fetch(False)

def load_config(path: str) -> dict:
  with open(os.path.join(path, "config.json")) as f: return json.load(f)
