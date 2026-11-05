"""The tensor-parallel control ring (fusezap/serve/shmchan.py, I57 M3): rank 0 -> workers through shared memory.

  1. one process: messages in order across many wraps of the ring, payloads larger than a slot through the side path
  2. two processes (spawned, like a worker): 400 messages of random sizes (some over a slot: the socket path), pauses
     longer than the spin window (the reader sleeps and is woken), the order and every byte checked by the reader
  3. a full ring: the writer waits for the slowest reader, and gives up with an error when the reader is gone
  4. dist.launch over the ring: a barrier round trip and the worker answering in order (the host backend)
"""
from __future__ import annotations
import multiprocessing as mp
import os, time, zlib
import numpy as np
import pytest
from fusezap.serve.shmchan import ShmRing, SLOT_HDR

def test_one_process_wraps_and_big():
  w = ShmRing.create(1, slots=4, slot_bytes=4096)
  try:
    r = ShmRing.attach(w.path, 0)
    side: list[bytes] = []
    for i in range(50):
      msg = ("op", (i, b"x" * (i * 97 % 9000)), float(i))
      w.send(msg, big=side.append)
      assert r.recv(big=lambda: side.pop(0)) == msg
    assert not side and w.wseq == 50 and r.rseq == 50
  finally: w.unlink()

def _reader(path: str, conn, n: int) -> None:
  """A worker-like reader: receive n messages, check each, reply with a digest of all of them."""
  r = ShmRing.attach(path, 0)
  pend: list[bytes] = []
  def idle(s: float) -> None:
    if conn.poll(s) and (m := conn.recv_bytes()) != b"w": pend.append(m)
  def big() -> bytes:
    if pend: return pend.pop(0)
    while (m := conn.recv_bytes()) == b"w": pass
    return m
  h, slept = 0, 0
  for i in range(n):
    t0 = time.monotonic()
    op, (k, data), _ = r.recv(big=big, idle=idle)
    slept += time.monotonic() - t0 > 0.1
    assert op == "fwd" and k == i, (op, k, i)
    h = zlib.crc32(data, h)
  conn.send(("done", h, slept))

def test_two_processes_order_big_and_wake(monkeypatch):
  monkeypatch.setenv("FZ_TP_SPIN_MS", "20")
  w = ShmRing.create(1, slots=8, slot_bytes=1 << 14)
  a, b = mp.Pipe()
  ctx = mp.get_context("spawn")
  p = ctx.Process(target=_reader, args=(w.path, b, 400), daemon=True)
  p.start()
  try:
    rng = np.random.default_rng(0)
    h, woken = 0, 0
    for i in range(400):
      n = int(rng.choice([0, 10, 1000, 16000, 40000]))   # 16000 / 40000 bytes: over the slot, sent over the pipe
      data = rng.integers(0, 255, n, dtype=np.uint8).tobytes()
      h = zlib.crc32(data, h)
      def wake(r: int) -> None:
        nonlocal woken
        woken += 1
        a.send_bytes(b"w")
      w.send(("fwd", (i, data), time.monotonic()), big=a.send_bytes, wake=wake, alive=lambda: None, timeout=60)
      if i % 100 == 50: time.sleep(0.15)   # longer than the spin window: the reader goes to sleep
    assert a.poll(60)
    tag, hr, slept = a.recv()
    assert tag == "done" and hr == h and slept >= 1 and woken >= 1, (slept, woken)
  finally:
    p.join(10)
    w.unlink()
  assert p.exitcode == 0

def _slow(path: str, conn) -> None:
  r = ShmRing.attach(path, 0)
  conn.send("up")
  time.sleep(0.5)
  for _ in range(3): r.recv()
  conn.send("read3")
  time.sleep(30)

def test_full_ring_waits_then_fails_on_a_dead_reader():
  w = ShmRing.create(1, slots=2, slot_bytes=1024)
  a, b = mp.Pipe()
  p = mp.get_context("spawn").Process(target=_slow, args=(w.path, b), daemon=True)
  p.start()
  try:
    assert a.poll(30) and a.recv() == "up"
    t0 = time.monotonic()
    for i in range(4): w.send(("x", (i,), 0.0))   # slots 2: the 3rd send waits for the reader (asleep for 0.5 s)
    assert time.monotonic() - t0 > 0.3 and w.waits >= 1
    assert a.poll(30) and a.recv() == "read3"
    w.send(("x", (4,), 0.0))   # reader read 3 of 4: one slot free
    p.kill(); p.join(10)
    def alive() -> None:
      if p.exitcode is not None: raise RuntimeError("reader gone")
    with pytest.raises(RuntimeError, match="reader gone"): w.send(("x", (5,), 0.0), alive=alive, timeout=30)
  finally:
    if p.is_alive(): p.kill()
    w.unlink()

def test_slot_header_size():
  assert SLOT_HDR == 32

def test_launch_uses_the_ring_and_unlinks_it():
  from fusezap.serve import dist
  g = dist.launch(2, "CPU", backend="host", on_dead=lambda *a: None)
  try:
    assert g.ring is not None and not os.path.exists(g.ring.path)   # unlinked once the worker had it mapped
    t0 = time.perf_counter()
    for i in range(50): g.barrier(f"b{i}")
    per = (time.perf_counter() - t0) / 50
    big = np.arange(400000, dtype=np.int64)   # 3.2 MB: over a slot, through the socket, in order with the ring
    g.post("call", "tests.test_shmchan:_echo_sum", big)
    assert g.barrier("echo") == [int(big.sum())]
    g.post("call", "tests.test_shmchan:_echo_sum", big[:10])
    assert g.barrier("echo2") == [45]
    print(f"barrier round trip {per * 1e6:.0f} us")
    assert per < 0.05
  finally: g.close()

def _echo_sum(g, a) -> int: return int(np.asarray(a).sum())
