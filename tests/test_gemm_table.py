"""The per-shape decode GEMM table (fusezap/device/cuda_kernels/gemm_table.py, I57 M3b): the choice from timed
candidates, the precedence disk > shipped > rule, wg_swap_default's use of it (SK / NC fixed per shape: a row's sum does
not depend on the batch), first-writer-wins storage, and the autotune driver. No GPU: stand-in devices and times."""
from __future__ import annotations
import json, math
import pytest
import fusezap.device.cuda_kernels as ck
from fusezap.device.cuda_kernels import gemm_table as GT, wgmma

class D:
  """A stand-in sm_90 device (the attributes the config rules read)."""
  def __init__(self, name: str | None = "NVIDIA H200", sms: int = 132):
    self.__dict__.update(_fz_sms=sms, _fz_smem=232448, cc=(9, 0))
    if name: self.gpu_name = name

@pytest.fixture(autouse=True)
def table_file(tmp_path, monkeypatch):
  monkeypatch.setenv("FZ_GEMM_TABLE_FILE", str(tmp_path / "t.json"))
  monkeypatch.delenv("FZ_GEMM_TABLE", raising=False)
  GT.reset()
  yield tmp_path / "t.json"
  GT.reset()

def test_pick_shares_sk_and_nc_over_bx():
  """SK by the sum over BX of each BX's best time at that SK (not each BX's own best SK), then NC, then S per BX."""
  t = {(8, 1, 4, 4): 24.6, (8, 1, 4, 3): 24.7, (8, 1, 8, 1): 33.5, (32, 1, 6, 3): 33.7, (32, 1, 6, 4): 42.1, (32, 1, 8, 1): 145.0,
       (8, 1, 6, 3): 24.7, (8, 2, 4, 3): 30.0, (32, 2, 6, 3): 30.0, (8, 1, 4, 6): 20.0}   # SK 6 not timed at BX 32: not eligible
  nc, sk, S, us = GT.pick(t, [8, 32])
  assert (sk, nc) == (3, 1) and S == {8: 4, 32: 6} and us == {8: 24.7, 32: 33.7}   # NC 1: 58.4 vs NC 2: 60.0
  assert GT.pick({(8, 1, 4, 1): 10.0, (8, 1, 8, 1): 10.0, (8, 1, 4, 2): 10.0}, [8])[:3] == (1, 1, {8: 4})   # ties: smaller SK, S
  assert GT.pick({(8, 1, 4, 1): math.inf}, [8]) is None and GT.pick({(8, 1, 4, 1): 1.0}, [8, 16]) is None

def test_lookup_precedence_and_off_switch(monkeypatch, table_file):
  d, k = D(), "NVIDIA H200|132|bf16|5120|8192|1"
  assert GT.lookup(d, "bf16", 5120, 8192) == GT.SHIPPED[k] == (1, 3, {8: 4, 32: 6})
  assert GT.lookup(D("NVIDIA H100 80GB HBM3"), "bf16", 5120, 8192) is None and GT.lookup(D(None), "bf16", 5120, 8192) is None
  assert GT.lookup(D(sms=114), "bf16", 5120, 8192) is None   # another SM count (H100 PCIe-like): no shipped entry
  GT.put(k, {"NC": 2, "SK": 4, "S": {"8": 6}})
  assert GT.lookup(d, "bf16", 5120, 8192) == (2, 4, {8: 6}) and GT.tuned(k)
  GT.reset()
  assert GT.lookup(d, "bf16", 5120, 8192) == (2, 4, {8: 6})   # from the file
  monkeypatch.setenv("FZ_GEMM_TABLE", "0")
  assert GT.lookup(d, "bf16", 5120, 8192) is None

def test_put_first_writer_wins(table_file):
  a, b = {"NC": 1, "SK": 2, "S": {"8": 4}}, {"NC": 1, "SK": 3, "S": {"8": 8}}
  assert GT.put("x", a) == a
  GT.reset()                                    # another process: its memo did not see the first write
  assert GT.put("x", b) == a and GT.put("y", b) == b
  d = json.loads(table_file.read_text())
  assert d["version"] == GT.VERSION and d["entries"] == {"x": a, "y": b}

def test_wg_swap_default_uses_the_table_per_shape():
  """The shipped H200 70B-rank entries: SK and NC the same at every M (rows stay batch-independent), S per BX from the
  table where it has the BX, else the rule's depth; table=False and other GPUs keep the rule; FZ_ONE_K bypasses it."""
  d, qkv, dn = D(), (5120, 8192), (8192, 14336)
  cs = {M: ck.wg_swap_default(d, "bf16", M, *qkv) for M in (1, 2, 4, 8, 16, 32, 64)}
  assert {(c.NC, c.SK) for c in cs.values()} == {(1, 3)} and [c.BX for c in cs.values()] == [8, 8, 8, 8, 16, 32, 64]
  rule = {M: ck.wg_swap_default(d, "bf16", M, *qkv, table=False) for M in cs}
  assert cs[1].S == 4 and cs[32].S == 6 and cs[16].S == rule[16].S == 8 and {c.SK for c in rule.values()} == {1}
  assert all(wgmma.swap_smem_bytes(c) <= 232448 for c in cs.values())
  assert {ck.wg_swap_default(d, "bf16", M, *dn).SK for M in (1, 8, 32)} == {2} and ck.wg_swap_default(D(None), "bf16", 1, *dn).SK == 1
  assert ck.wg_swap_default(D("NVIDIA H100 80GB HBM3"), "bf16", 32, *qkv) == ck.wg_swap_default(d, "bf16", 32, *qkv, table=False)
  assert ck.wg_swap_candidates(d, "bf16", 32, *qkv)[:2] == [cs[32], rule[32]]
  try:
    ck.set_one_k_order(True)
    assert ck.wg_swap_default(d, "bf16", 32, *qkv).SK == 1
  finally: ck.set_one_k_order(False)
  GT.put("NVIDIA H200|132|bf16|5120|8192|1", {"NC": 2, "SK": 2, "S": {"8": 12}})   # a tuned entry overrides the shipped one
  c8, c32 = ck.wg_swap_default(d, "bf16", 1, *qkv), ck.wg_swap_default(d, "bf16", 32, *qkv)
  assert (c8.NC, c8.SK, c8.S) == (2, 2, 12) and (c32.NC, c32.SK) == (2, 2) and wgmma.swap_smem_bytes(c32) <= 232448

def test_autotune_driver(monkeypatch, table_file):
  """Cached shapes are not timed again, tiny shapes keep the rule, a timeout writes nothing; one log line per shape."""
  d, logs, calls = D(), [], []
  def fake(dev, kind, out, N, K, ws, bxs, deadline, rule):
    calls.append((N, K, out, tuple(bxs), rule(8).SK))
    if N == 999 * 8: return None   # the budget ran out
    return {"NC": 1, "SK": 2, "S": {str(b): 6 for b in bxs}, "us": {str(b): 10.0 for b in bxs}, "rule_us": {str(b): 20.0 for b in bxs}}, {}
  monkeypatch.setattr(GT, "tune_shape", fake)
  GT.put("NVIDIA H200|132|bf16|8192|4096|1", {"NC": 1, "SK": 1, "S": {"8": 8}})
  shapes = [("qkv", 5120, 8192, "bf16", [1, 2]), ("o", 8192, 4096, "bf16", [3]), ("tiny", 896, 896, "bf16", [4]), ("big", 999 * 8, 8192, "f32", [5])]
  rep = GT.autotune(d, "bf16", shapes, [1, 2, 4, 8, 16, 32], log=logs.append)
  assert [r["status"] for r in rep] == ["tuned", "cached", "skipped", "timeout"]
  assert calls == [(5120, 8192, "bf16", (8, 16, 32), 1), (7992, 8192, "f32", (8, 16, 32), 1)]   # the rule (table=False): SK 1
  assert GT.lookup(d, "bf16", 5120, 8192) == (1, 2, {8: 6, 16: 6, 32: 6}) and GT.lookup(d, "bf16", 7992, 8192) is None
  assert len(logs) == 5 and "tuned: NC 1 SK 2" in logs[0] and "60.0 us" in logs[0] and "cached" in logs[1] and "gemm autotune" in logs[-1]
  assert GT.autotune(D(None), "bf16", shapes, [1]) == [] and GT.autotune(d, "bf16", shapes, [128]) == []

def test_tune_shape_search(monkeypatch):
  """tune_shape's two stages over a modelled time (no GPU): stage A finds the SK, stage B the NC and depths at it; the
  entry records the rule's times; each stage is compiled before it is timed; a passed deadline gives None."""
  import time
  d, seen, pre = D(), [], []
  d.name = "CUDA"
  class B:
    handle = 0
    def copyin(self, mv): pass
  model = lambda c: (20.0 + abs(c.SK - 3) * 5 + abs(c.S - (4 if c.BX == 8 else 6)) + (3 if c.NC == 2 else 0)) * (1 + c.BX / 64)
  def fake_gemm(dev, C, A, w, M, N, K, kind, out, impl, cfg, bind):
    assert impl == "wgswap" and bind and M == cfg.BX
    seen.append(cfg); return cfg
  monkeypatch.setattr(ck, "gemm", fake_gemm)
  monkeypatch.setattr(ck, "_new", lambda dev, n, dt: B())
  monkeypatch.setattr(GT, "_time_us", lambda dev, fs, reps: model(fs[0]))
  monkeypatch.setattr(GT, "_precompile", lambda dev, kind, out, cfgs: pre.append(len(cfgs)))
  rule = lambda b: ck.wg_swap_default(d, "bf16", b, 5120, 8192, table=False)
  entry, times = GT.tune_shape(d, "bf16", "bf16", 5120, 8192, [1, 2, 3], [8, 16, 32], time.monotonic() + 60, rule)
  assert (entry["NC"], entry["SK"], entry["S"]) == (1, 3, {"8": 4, "16": 6, "32": 6}) and entry["src"] == "autotune"
  assert entry["rule_us"]["8"] == round(model(rule(8)), 2) and len(pre) == 2 and pre[0] > 0
  assert {c.SK for c in seen} >= {1, 2, 3, 4, 6} and {c.NC for c in seen if c.SK == 3} == {1, 2}
  assert GT.tune_shape(d, "bf16", "bf16", 5120, 8192, [1], [8], time.monotonic() - 1, rule) is None
