"""Speed run 4 profiling tools (docs/status/SPEED.md "Run 4"): bench/kernel_prof.py on synthetic nsys / kineto files,
bench/step_prof_summary.py, and FZ_STEP_PROF (fusezap/serve/stepprof.py) on an engine step."""
import gzip, json, os, sqlite3, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench"))
import kernel_prof, step_prof_summary  # noqa: E402

L = 2   # layers of the synthetic model
LAYER = [("fz_wgswap_bf16_qkv", (36, 1, 1), 10.0), ("E_32_36_128", (72, 1, 1), 2.0), ("fz_paged_split_bf16_d128", (28, 4, 1), 5.0),
         ("fz_wgswap_bf16_o", (28, 1, 2), 8.0), ("fz_rmsnorm_bf16_3584", (32, 1, 1), 1.0), ("fz_wgswap_bf16_gu", (296, 1, 1), 40.0),
         ("E_32_18944", (148, 1, 1), 3.0), ("fz_wgswap_bf16_down", (28, 1, 2), 20.0), ("fz_rmsnorm_bf16_3584", (32, 1, 1), 1.0)]
TAIL = [("fz_wgswap_bf16_lm", (1188, 1, 1), 300.0), ("fz_argmax_rows_152064", (32, 1, 1), 4.0)]

def _timeline(forwards: int, gap_us: float = 50.0) -> list[tuple]:
  """(start ns, end ns, name, grid) of `forwards` back-to-back forwards with gap_us of GPU idle between them."""
  t, out = 0, []
  for _ in range(forwards):
    for _ in range(L):
      for n, g, us in LAYER: out.append((t, t + int(us * 1e3), n, g)); t += int(us * 1e3) + 500
    for n, g, us in TAIL: out.append((t, t + int(us * 1e3), n, g)); t += int(us * 1e3) + 500
    t += int(gap_us * 1e3)
  return out

def _sqlite(path: str, tl: list[tuple]) -> None:
  db = sqlite3.connect(path)
  db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
  db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, demangledName INT, shortName INT, gridX INT, gridY INT, gridZ INT, "
             "blockX INT, blockY INT, blockZ INT, graphNodeId INT)")
  db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INT, end INT, nameId INT)")
  db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, copyKind INT, bytes INT)")
  ids: dict[str, int] = {}
  for s, e, n, g in tl:
    if n not in ids: ids[n] = len(ids); db.execute("INSERT INTO StringIds VALUES (?, ?)", (ids[n], n))
    db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?,?,?,?,?,?)", (s, e, ids[n], ids[n], *g, 128, 1, 1, 7))
  ids["cuGraphLaunch"] = len(ids); db.execute("INSERT INTO StringIds VALUES (?, ?)", (ids["cuGraphLaunch"], "cuGraphLaunch"))
  db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (0, 3000, ?)", (ids["cuGraphLaunch"],))
  db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (0, 2000, 1, 4096)")
  db.commit(); db.close()

def test_kernel_classes():
  k = kernel_prof.klass
  assert k("fz_paged_split_bf16_d128") == k("void cutlass::device_kernel<flash::FlashAttnFwdSm90<...>>") == "attention"
  assert k("fz_wgswap_bf16_nt") == k("nvjet_tst_128x64_64x8") == k("sm90_xmma_gemm_bf16bf16_bf16f32") == "gemm"
  assert k("fz_rmsnorm_bf16_3584") == k("void vllm::fused_add_rms_norm_kernel<...>") == k("triton_red_fused_add_mean_pow_rsqrt_0") == "norm"
  assert k("void vllm::act_and_mul_kernel<silu>") == k("fz_silu_mul_bf16_18944") == "act"
  assert k("void vllm::reshape_and_cache_flash_kernel") == "kv_write" and k("fz_rope_qkv_b_bf16") == "rope"
  assert k("fz_argmax_rows_152064") == "sampling" and k("E_32_18944") == "elementwise" and k("r_3584") == "reduce"

@pytest.mark.parametrize("fmt", ["sqlite", "kineto"])
def test_kernel_prof_forwards_and_classes(tmp_path, fmt, capsys):
  tl = _timeline(10)
  if fmt == "sqlite":
    path = str(tmp_path / "w.sqlite"); _sqlite(path, tl)
  else:
    path = str(tmp_path / "w.json.gz")
    with gzip.open(path, "wt") as f:
      json.dump({"traceEvents": [{"ph": "X", "cat": "kernel", "name": n, "ts": s / 1e3, "dur": (e - s) / 1e3, "args": {"grid": list(g), "block": [128, 1, 1]}}
                                 for s, e, n, g in tl]}, f)
  r = kernel_prof.analyze(kernel_prof.load(path), L, 10)
  assert r["forwards"] == 10 and r["norms_per_forward"] == 2 * L
  gemm = L * (10 + 8 + 40 + 20) + 300
  assert r["class_ms"]["gemm"] == pytest.approx(gemm / 1e3, rel=1e-3)
  assert r["class_ms"]["attention"] == pytest.approx(L * 5 / 1e3, rel=1e-3)
  assert r["kernels_per_forward"] == L * len(LAYER) + len(TAIL)
  assert len(r["layer"]) == len(LAYER) and r["layer"][0]["name"].startswith("fz_paged_split")
  assert r["tail"][0]["name"].startswith("fz_paged_split") and r["tail"][-1]["name"].startswith("E_32_36_128")
  assert any(x["name"].startswith("fz_argmax") for x in r["tail"]) and max(x["gap_before_us"] for x in r["tail"]) >= 50
  # the 50 us gap between forwards is GPU idle; the per-forward step time is the window over the forwards
  assert r["gpu_idle_ms"] > 0.045 and r["step_ms"] == pytest.approx(r["gpu_busy_ms"] + r["gpu_idle_ms"], abs=2e-4)
  if fmt == "sqlite": assert r["graph_kernels_pct"] == 100.0 and r["api"][0]["name"] == "cuGraphLaunch" and "HtoD" in r["memcpy"]
  # two inputs print the side-by-side table first
  sys.argv = ["kernel_prof", f"{path}:A", f"{path}:B"]
  assert kernel_prof.main() == 0
  out = capsys.readouterr().out
  assert out.startswith("ms per forward") and "== A" in out and "== B" in out

def test_step_prof_summary(tmp_path):
  steps = [{"t": 100.0 + i * 0.01, "rows_d": 32, "rows_p": 0, "size_p": 0, "sched": 0.1, "d_prep": 0.3, "d_call": 0.2, "post": 8.0,
            "total": 8.6, "fwd": [["s", 8.0, 1.5]]} for i in range(50)]
  steps.append({"t": 100.6, "rows_d": 30, "rows_p": 2048, "size_p": 2048, "total": 60.0, "fwd": [["d", 8.0, None], ["p", 50.0, 0.1]]})
  s = step_prof_summary.summarize(steps, 1.0)
  assert s["kinds"]["D"]["n"] == 50 and s["kinds"]["D"]["gpu_d"] == 8.0 and s["kinds"]["D"]["gap"] == 1.5 and s["kinds"]["DP"]["size_p"] == [2048]
  assert s["gpu_in_forwards_pct"] == pytest.approx(100 * (50 * 8 + 58) / 1000) and s["gpu_gaps_pct"] == pytest.approx(100 * (75 + 0.1) / 1000)

def test_step_prof_engine(tmp_path, monkeypatch):
  """FZ_STEP_PROF on a real engine (the 0.5B model if cached, on this machine's default device): one line per step with
  the host phases; off by default."""
  from fusezap.serve.weights import resolve_model
  try: resolve_model("Qwen/Qwen2.5-0.5B-Instruct", allow_download=False)
  except Exception: pytest.skip("Qwen2.5-0.5B-Instruct not in the HF cache")
  from fusezap.device import Device
  from fusezap.dtype import dtypes
  from fusezap.serve.models import load_model
  from fusezap.serve.engine import LLMEngine, SamplingParams
  dev = os.environ.get("FZ_DEVICE") or ("METAL" if Device.available("METAL") else "CPU")
  m = load_model("Qwen/Qwen2.5-0.5B-Instruct", dtype=dtypes.float32, device=dev, max_len=256, prefill_chunk=64)
  monkeypatch.delenv("FZ_STEP_PROF", raising=False)
  assert LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=64)._prof is None
  path = tmp_path / "steps.jsonl"
  monkeypatch.setenv("FZ_STEP_PROF", str(path))
  eng = LLMEngine(m, num_blocks=64, block_size=16, max_num_seqs=4, max_prefill_tokens=64)
  rng = np.random.default_rng(0)
  out = eng.generate([rng.integers(100, 30000, n).tolist() for n in (40, 90)], SamplingParams(max_tokens=4, ignore_eos=True))
  eng._prof.f.flush()
  rs = [json.loads(l) for l in open(path)]
  assert len(rs) == eng.stats["steps"] and all(len(o) == 4 for o in out)
  kinds = {step_prof_summary.kind(r) for r in rs}
  assert "P" in kinds and ("D" in kinds or "DP" in kinds)
  for r in rs:
    assert r["total"] >= r["sched"] >= 0 and r["post"] >= 0
    if r["rows_d"]: assert "d_prep" in r and "d_call" in r
    if r["size_p"]: assert r["size_p"] == 64 and 0 < r["rows_p"] <= 64 and "p_prep" in r
    if Device[m.device].__class__.__name__ == "CUDADevice": assert "fwd" in r

def test_kernel_prof_clips_bad_end_stamps():
  """Profile 1's c32 nsys export: kernel end stamps ~25 ms late. They are clipped to the next kernel's start, so the class
  times are the kernels' own again."""
  tl = _timeline(20)
  bad = [(s, e + 25_000_000 if i % 3 else e, n, g) for i, (s, e, n, g) in enumerate(tl)]
  mk = lambda t: {"kernels": [(s, e, n, g, (128, 1, 1), True) for s, e, n, g in t], "memcpy": [], "api": [], "source": "t"}
  good, r = kernel_prof.analyze(mk(tl), L, 5), kernel_prof.analyze(mk(bad), L, 5)
  assert r["clipped_kernels"] > 0 and r["forwards"] == 20 and good["clipped_kernels"] == 0
  assert r["class_ms"]["gemm"] == pytest.approx(good["class_ms"]["gemm"], rel=0.05)
  assert r["step_ms"] == pytest.approx(good["step_ms"], rel=0.05)

# ---------------------------------------------------------------- bench/decode_kernels.py (I60)

import decode_kernels  # noqa: E402

class _Ast:
  def __init__(self, name, M=None, N=None, K=None):
    self.name = name
    if M is not None: self.M, self.N, self.K = M, N, K
class _It:
  def __init__(self, name=None, ast=None):
    from types import SimpleNamespace
    self.prg = SimpleNamespace(name=name) if name else None
    self.si = SimpleNamespace(ast=(ast,)) if ast is not None else None

def test_decode_kernels_classes_and_ideal():
  """The per-item classes of a 70B TP=2 rank's decode layer (GEMMs by projection, composed kernels by the GEMM before
  them) and the ideal bytes: one step's weights at N = 1 are the rank's ~70.6 GB (14.7 ms at 4.8 TB/s)."""
  from fusezap.serve.models.llama import LlamaConfig
  sh = decode_kernels.PRESETS["llama70b-tp2"]
  cfg = LlamaConfig(sh["hidden_size"], sh["intermediate_size"], sh["n_layers"], sh["n_heads"], sh["n_kv_heads"], 128, sh["vocab_size"])
  g = lambda N, K: _It(ast=_Ast("fz_serve_gemm_bf16_bf16", 1, N, K))
  items = [_It("fz_rmsnorm_res_bf16_8192"), g(5120, 8192), _It("E_40_128"), _It(ast=_Ast("fz_kv_write")), _It("fz_paged_tcd_bf16_d128_g8"),
           _It("fz_paged_tcd_bf16_d128_g8_combine"), g(8192, 4096), _It(ast=_Ast("fz_all_reduce_copy")), g(28672, 8192), _It("E_14336"),
           g(8192, 14336), _It("fz_rope_qkv_kvw_bf16_40_36_128"), g(128256, 8192), _It("E_128256"), _It("fz_argmax_rows_128256")]
  got = [c for c, _ in decode_kernels.classify(items, cfg)]
  assert got == ["norm:add+rms", "gemm:qkv", "rope(composed)", "kv_write", "attn:partial", "attn:combine", "gemm:o", "allreduce",
                 "gemm:gate_up", "act(composed)", "gemm:down", "rope+kv_write", "gemm:lm_head", "logits(composed)", "argmax"]
  ib = decode_kernels.ideal_bytes(cfg, 1, 512, False)
  w = sum(ib[k] for k in ("gemm:qkv", "gemm:o", "gemm:gate_up", "gemm:down", "gemm:lm_head"))
  assert 70.0e9 < w < 71.2e9 and w / 4.8e12 * 1e3 == pytest.approx(14.7, abs=0.15)
  assert ib["attn"] == pytest.approx(80 * (512 * 4 * 128 * 2 * 2 + 2 * 32 * 128 * 2))
  assert decode_kernels._group("attn:combine") == "attn" and decode_kernels._group("act(composed)") == "act"

def _metal() -> bool:
  from fusezap.device import Device
  return Device.available("METAL")

@pytest.mark.skipif(not _metal(), reason="needs METAL (the dry run takes ~1 s there, minutes on CPU)")
def test_decode_kernels_dry_run(tmp_path):
  """The whole script on a tiny random model: two variants (one with the all-reduce copies), the class table, JSON."""
  import subprocess
  out = tmp_path / "k.json"
  script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "decode_kernels.py")
  p = subprocess.run([sys.executable, script, "--dry", "--variants", "a:;b:FZ_KPROF_AR=copy", "--json", str(out)], capture_output=True,
                     text=True, timeout=600, env={**os.environ, "FZ_DEVICE": "METAL"})
  assert p.returncode == 0, p.stderr[-3000:]
  r = json.load(open(out))
  assert [x["variant"] for x in r["runs"]] == ["a", "a", "b", "b"]
  cl = lambda x: {c["class"]: c["count"] for c in x["breakdown"]["classes"]}
  assert "allreduce" not in cl(r["runs"][0]) and cl(r["runs"][2])["allreduce"] == 2 * 2   # 2 layers x (o, down)
  assert "== A/B" in p.stdout and all(x["tpot_ms"] > 0 and x["breakdown"]["sum_ms"] > 0 for x in r["runs"])

# ---------------------------------------------------------------- all-reduce stamps, random TP weights, tp_step (I60 M5)

def _stamped(F: int, J: int, off: float, skew: list[float], intr: float, seg: float) -> list[dict]:
  """Two ranks' stamp_read results for F consecutive forwards with J all-reduces: rank 1's clock is `off` us ahead,
  rank 1 arrives skew[j % len(skew)] us after rank 0 at all-reduce j (negative: before), the latest arriver's all-reduce
  takes intr us, each segment of compute seg us on the rank without the skew."""
  out = [{"rank": r, "n": 1, "ars": J, "fwd": []} for r in (0, 1)]
  t = 1000.0
  for s in range(F):
    t0 = t
    pre, post = [[], []], [[], []]
    for j in range(J):
      sk = skew[j % len(skew)]
      a0, a1 = t + seg + max(0.0, -sk), t + seg + max(0.0, sk)   # arrivals on the common clock
      end = max(a0, a1) + intr
      for r, a in ((0, a0), (1, a1)): pre[r].append(a); post[r].append(end)
      t = end
    for r in (0, 1):
      sh = off if r else 0.0
      out[r]["fwd"].append({"seq": s, "t0": int((t0 + sh) * 1e3), "pre": [int((x + sh) * 1e3) for x in pre[r]], "post": [int((x + sh) * 1e3) for x in post[r]]})
    t += 30.0   # tail (lm_head, ...) to the next forward's begin
  return out

def test_stamp_stats_splits_intrinsic_and_skew():
  """stamp_stats on a synthetic two-rank timeline: the clock offset is recovered from the all-reduce ends, rank 1 late
  by 40 us at the o_proj all-reduces and rank 0 late by 10 us at the down_proj ones: per all-reduce 9 us intrinsic,
  the waits, the late rank per parity, the period and the compute per rank."""
  from fusezap.serve import tpprof
  rs = _stamped(F=6, J=8, off=123456.0, skew=[40.0, -10.0], intr=9.0, seg=50.0)
  s = tpprof.stamp_stats(rs)
  assert s["forwards"] == 6 and s["ars"] == 8 and s["clock_offset_us"] == [0.0, pytest.approx(123456.0, abs=0.01)]
  assert s["intrinsic_us"] == pytest.approx(9.0, abs=0.01) and s["intrinsic_ms"] == pytest.approx(8 * 9e-3, abs=1e-3)
  assert s["wait_ms"] == pytest.approx(4 * 40e-3 + 4 * 10e-3, abs=1e-3)
  bp = s["by_parity"]
  assert bp["attn_o"]["wait_us"] == pytest.approx(40.0, abs=0.01) and bp["attn_o"]["late"] == [0, 6 * 4]
  assert bp["mlp_down"]["wait_us"] == pytest.approx(10.0, abs=0.01) and bp["mlp_down"]["late"] == [6 * 4, 0]
  per = 8 * (50 + 9) + 4 * 40 + 4 * 10 + 30
  assert s["period_ms"] == [pytest.approx(per / 1e3, abs=1e-3)] * 2 and s["tail_ms"] == [pytest.approx(0.03, abs=1e-3)] * 2
  assert s["compute_ms"][0] == pytest.approx((8 * 50 + 4 * 10) / 1e3, abs=1e-3) and s["compute_ms"][1] == pytest.approx((8 * 50 + 4 * 40) / 1e3, abs=1e-3)
  assert s["ar_ms"][0] == pytest.approx((8 * 9 + 4 * 40) / 1e3, abs=1e-3)
  assert s["begin_skew_us"] == [0.0, pytest.approx(0.0, abs=0.01)]
  assert "error" in tpprof.stamp_stats([{"rank": 0, "error": "x"}, rs[1]])
  rs[1]["fwd"] = rs[1]["fwd"][3:]   # forwards held by both ranks only
  assert tpprof.stamp_stats(rs)["forwards"] == 3

def test_stamp_source_renders():
  """The stamp kernel's CUDA source after %-formatting: the globaltimer asm and the ring arithmetic."""
  from fusezap.serve import tpprof
  src = tpprof._STAMP_SRC
  assert '"mov.u64 %0, %%globaltimer;"' in src and "c % R" in src and f"#define R {tpprof.STAMP_R}" in src and "%%%%" not in src

@pytest.mark.skipif(not (os.environ.get("FZ_DEVICE") or "").upper().startswith("CUDA"), reason="needs FZ_DEVICE=CUDA")
def test_stamps_cuda_eager_and_graph():
  """The stamp kernel on CUDA, eager and inside a replayed fz.jit CUDA graph: a begin stamp takes the next ring slot
  (time, its id as tag, the sequence), the pre / post stamps around a kernel land in that slot, GPU clock in order."""
  from fusezap.serve import tpprof
  from fusezap.tensor import Tensor
  from fusezap.jit import Jit
  def step(x: Tensor) -> Tensor:
    tpprof.stamp(x._realized_contiguous().realized, "begin")
    tpprof.stamp(x._realized_contiguous().realized, "pre")
    y = (x * 2.0 + 1.0).realize()
    tpprof.stamp(y._realized_contiguous().realized, "post")
    return y
  j = Jit(step, plan=True)
  x = Tensor(np.arange(4096, dtype=np.float32), device="CUDA").realize()
  for _ in range(8): y = j(x)
  assert np.array_equal(y.numpy(), np.arange(4096, dtype=np.float32) * 2 + 1)
  c = j.captured
  sts = [it.si.ast[0] for it in c.items if it.si is not None and it.si.ast and isinstance(it.si.ast[0], tpprof.StampCall)]
  assert [s.kind for s in sts] == ["begin", "pre", "post"] and c.graph is not None
  buf, meta = sts[0].ring.read()
  rows = [s for s in range(tpprof.STAMP_R) if meta[s, 1] == sts[0].sid and meta[s, 0] > 0]
  assert len(rows) >= 3, meta[:8]
  for s in rows:
    t0, a, b = int(meta[s, 0]), int(buf[s, sts[1].sid]), int(buf[s, sts[2].sid])
    assert 0 < t0 <= a <= b < t0 + 10**8, (t0, a, b)

def test_random_weights_tp_slices():
  """random_weights: one host array per shape (layers share it), bf16 tagged; a model at a preset's shapes."""
  from fusezap.serve.models import random_weights as RW
  from fusezap.dtype import dtypes
  sh = RW.PRESETS["tiny"]
  w = RW.weights(sh, dtypes.bfloat16, 2)
  assert w["model.layers.0.mlp.gate_proj.weight"] is w["model.layers.1.mlp.gate_proj.weight"] and w["lm_head.weight"].st_dtype == "BF16"
  assert w["model.layers.1.self_attn.q_proj.weight"].shape == (4 * 128, 256) and w["model.norm.weight"][0] == 0x3F80
  assert RW.PRESETS["llama70b"]["intermediate_size"] == 2 * RW.PRESETS["llama70b-tp2"]["intermediate_size"]
  m = RW.build(sh, "CPU", dtypes.float32, 64, 16, layers=1)
  assert len(m.layers) == 1 and m.layers[0].wgu.shape == (2 * 512, 256)

@pytest.mark.skipif(not _metal(), reason="needs METAL (two processes, host backend)")
def test_tp_step_random_stamps_dry_run(tmp_path):
  """bench/tp_step.py over two processes (host backend) with random weights at the tiny preset (each rank its slices):
  a plain mode, a stamped env variant (mode=), the stamp split per bucket, the interleaved all-reduce bench, the JSON."""
  import subprocess
  out = tmp_path / "s.json"
  script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bench", "tp_step.py")
  p = subprocess.run([sys.executable, script, "--random", "tiny", "--layers", "1", "--device", "METAL", "--dtype", "float32", "--modes", "nccl",
                      "--env-ab", "st:mode=nccl,FZ_TP_AR_STAMP=1", "--buckets", "1,2", "--steps", "4", "--ctx", "16", "--max-model-len", "128",
                      "--num-blocks", "64", "--breakdown", "", "--ar-rows", "1", "--ar-chain", "both", "--prefix", "x_", "--json", str(out)],
                     capture_output=True, text=True, timeout=600, env={**os.environ, "FZ_DEVICE": "METAL"})
  assert p.returncode == 0, p.stderr[-3000:]
  r = json.load(open(out))
  assert [(s["variant"], s["b"]) for s in r["steps"]] == [("x_nccl", 1), ("x_nccl", 2), ("x_st", 1), ("x_st", 2)]
  st = [s["stamps"] for s in r["steps"] if "stamps" in s]
  assert len(st) == 2 and all(x["ars"] == 2 and x["forwards"] >= 2 and x["intrinsic_ms"] > 0 for x in st), st
  assert {a["chain"] for a in r["ar"]} == {True, False} and r["info"]["random"] == "tiny"
  assert "all-reduce stamps" in p.stdout and "x_st" in p.stdout
