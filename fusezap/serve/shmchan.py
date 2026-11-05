"""Rank 0 -> worker control messages through a shared-memory ring (I57 M3), instead of a pickled socket send per worker.

  ring = ShmRing.create(n_readers=1)                 # rank 0: a file in /dev/shm (else the temp dir), mmap'd
  ring.send(obj, big=send_over_socket, wake=...)    # pickled into the next slot; published by its sequence number
  r = ShmRing.attach(path, reader=0)                 # a worker (path from FZ_TP_SHM); rank 0 then unlinks the file
  obj = r.recv(big=recv_from_socket, idle=poll_socket)

Layout: a header of 64-byte lines (line 0: the writer's sequence; line 1 + r: reader r's last sequence read, line
1 + MAXR + r: reader r's "asleep" flag), then `slots` slots of [u64 seq | u64 nbytes | u32 crc | pad 8 | payload].
The writer fills slot (seq - 1) % slots once every reader has read the message that held it (it waits otherwise),
writes the payload, then the slot's seq (the publish). A reader spins on the next slot's seq (FZ_TP_SPIN_MS, default
50 ms: a decode step's next message arrives within it, so it is picked up within a microsecond or so), then sets its
asleep flag and sleeps on `idle` (the control socket's poll, 2 ms at a time); the writer sends a wake token over the
socket to a reader whose flag is set. A payload larger than a slot goes over the socket after a marker slot (in
order: the marker is read first). x86-64 keeps stores in order (TSO), so a published seq means the payload is visible;
elsewhere (arm64: the Mac tests) the payload's crc32 is checked too and re-read until it matches.
"""
from __future__ import annotations
import os, pickle, platform, struct, tempfile, time, zlib
from typing import Any, Callable
import mmap

MAXR = 15
LINE = 64
HDR = LINE * (1 + 2 * MAXR)
SLOT_HDR = 32
BIG, BIG_MARK = 1 << 62, 0x5a17b16
CHECK = os.environ.get("FZ_TP_SHM_CRC", "") == "1" or platform.machine().lower() not in ("x86_64", "amd64")

def spin_s() -> float:
  try: return float(os.environ.get("FZ_TP_SPIN_MS", "") or 50.0) / 1000.0
  except ValueError: return 0.05

class ShmRing:
  def __init__(self, path: str, size: int, slots: int, slot_bytes: int, reader: int = -1):
    self.path, self.slots, self.slot_bytes, self.reader = path, slots, slot_bytes, reader
    fd = os.open(path, os.O_RDWR)
    try: self.mm = mmap.mmap(fd, size)
    finally: os.close(fd)
    self.q = memoryview(self.mm).cast("Q")   # 8-byte words
    self.n_readers = 0   # set by create / attach
    self.wseq = 0
    self.rseq = 0
    self.waits = 0   # sends that waited for a full ring
  # ---- word offsets
  def _rd(self, r: int) -> int: return (LINE * (1 + r)) // 8
  def _asleep(self, r: int) -> int: return (LINE * (1 + MAXR + r)) // 8
  def _slot(self, seq: int) -> int: return HDR + ((seq - 1) % self.slots) * self.slot_bytes
  @staticmethod
  def create(n_readers: int, slots: int = 8, slot_bytes: int = 1 << 20) -> "ShmRing":
    """8 MB by default (a container's /dev/shm is often 64 MB, shared with NCCL); the space is reserved up front
    (posix_fallocate: a full tmpfs raises OSError here instead of SIGBUS on a later write)."""
    assert 1 <= n_readers <= MAXR
    d = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    fd, path = tempfile.mkstemp(prefix="fusezap-tp-", suffix=".ring", dir=d)
    size = HDR + slots * slot_bytes
    try:
      os.ftruncate(fd, size)
      if hasattr(os, "posix_fallocate"): os.posix_fallocate(fd, 0, size)
    except OSError:
      os.close(fd); os.unlink(path)
      raise
    else: os.close(fd)
    r = ShmRing(path, size, slots, slot_bytes)
    r.n_readers = n_readers
    r.q[1] = n_readers; r.q[2] = slots; r.q[3] = slot_bytes   # line 0: [wseq, n_readers, slots, slot_bytes]
    return r
  @staticmethod
  def attach(path: str, reader: int) -> "ShmRing":
    with open(path, "rb") as f: hdr = struct.unpack("<4Q", f.read(32))
    _, n, slots, sb = hdr
    r = ShmRing(path, HDR + slots * sb, slots, sb, reader)
    r.n_readers = n
    assert 0 <= reader < n
    r.rseq = r.q[r._rd(reader)]
    return r
  def unlink(self) -> None:
    try: os.unlink(self.path)
    except OSError: pass
  def close(self) -> None:
    try:
      self.q.release()
      self.mm.close()
    except Exception: pass
  # ---- writer (rank 0)
  def send(self, obj: Any, big: Callable[[bytes], None] | None = None, wake: Callable[[int], None] | None = None,
           alive: Callable[[], None] | None = None, timeout: float = 600.0) -> None:
    """Publish obj to every reader. big(payload): sends a payload larger than a slot (after its marker); wake(r): wakes
    reader r (asleep); alive(): raises if a reader is gone (checked while the ring is full)."""
    data = pickle.dumps(obj, protocol=5)
    seq = self.wseq + 1
    q, off = self.q, self._slot(seq)
    if seq > self.slots:   # the slot's previous message (seq - slots) must be read by every reader
      need, t_end, k = seq - self.slots, None, 0
      while min(q[self._rd(r)] for r in range(self.n_readers)) < need:
        if t_end is None: t_end, self.waits = time.monotonic() + timeout, self.waits + 1
        k += 1
        if k > 200: time.sleep(0.0002)
        if k % 1000 == 0 and alive is not None: alive()
        if time.monotonic() > t_end: raise RuntimeError(f"tensor-parallel control ring: a worker read nothing in {timeout:.0f} s")
    n = len(data)
    w0 = off // 8
    if n <= self.slot_bytes - SLOT_HDR:
      self.mm[off + SLOT_HDR:off + SLOT_HDR + n] = data
      q[w0 + 1] = n
      q[w0 + 2] = zlib.crc32(data) if CHECK else 0
      q[w0] = seq   # publish
    else:
      assert big is not None, "control ring: message larger than a slot and no big-message path"
      q[w0 + 1], q[w0 + 2] = BIG, BIG_MARK
      q[w0] = seq
    self.wseq = q[0] = seq
    if n > self.slot_bytes - SLOT_HDR: big(data)   # type: ignore[misc]
    if wake is not None:
      for r in range(self.n_readers):
        if q[self._asleep(r)]: wake(r)
  # ---- reader (a worker)
  def recv(self, big: Callable[[], bytes] | None = None, idle: Callable[[float], None] | None = None) -> Any:
    """The next message (in order). big(): the payload of a marker slot (the socket); idle(s): sleep up to s seconds or
    until woken (after FZ_TP_SPIN_MS of spinning)."""
    seq = self.rseq + 1
    q, off = self.q, self._slot(seq)
    w0 = off // 8
    if q[w0] != seq:
      t_spin, a = time.monotonic() + spin_s(), self._asleep(self.reader)
      k = 0
      while q[w0] != seq:
        k += 1
        if (k & 255) == 0 and time.monotonic() > t_spin:
          q[a] = 1
          if q[w0] == seq: break
          if idle is not None: idle(0.002)
          else: time.sleep(0.0005)
      q[a] = 0
    k = 0
    while True:   # x86-64: one pass. Elsewhere the slot's fields may become visible after its seq: re-read to the checksum
      n, crc = q[w0 + 1], q[w0 + 2]
      if n == BIG:
        if not CHECK or crc == BIG_MARK: break
      elif n <= self.slot_bytes - SLOT_HDR:
        data = bytes(self.mm[off + SLOT_HDR:off + SLOT_HDR + n])
        if not CHECK or zlib.crc32(data) == crc: break
      k += 1
      if k > 1000000: raise RuntimeError("control ring: a slot never matched its checksum")
    if n == BIG:
      assert big is not None, "control ring: a big message and no socket to read it from"
      data = big()
    self.rseq = seq
    q[self._rd(self.reader)] = seq   # frees the slot (the payload is copied out)
    return pickle.loads(data)
