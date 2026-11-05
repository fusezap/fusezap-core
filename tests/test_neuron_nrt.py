"""N4b: the persistent NEURON path and the N4 run-1 fixes, checked on the Mac with fakes (no nki, no nrtpy, no device).

* the .venv-nki worker compiles each NEFF in a fresh unique directory and never leaves NKI_ARTIFACTS_DIR set (run 1: every
  numerics case failed with neuronx-cc NCC_IDRV077 "Compilation artifacts already exist in the output directory");
* the NEFF disk cache: one compile per (source, argument sizes, versions), a key directory written once;
* the NEFF I/O binding from inputs_info / outputs_info / alias_info (run 1: NRT_INVALID because data0.must_alias_input was
  not passed), and the runtime worker (nrt_worker.py) against a fake nrtpy that rejects what libnrt rejected;
* the whole NEURON device on the persistent path (NeuronAllocator + NeuronNeffProgram) with a fake nrtpy whose `execute`
  runs the kernel through NKISHIM on the device buffers: results, one compile and one load per kernel, no host traffic
  for intermediates;
* the default-device fallback (run 1: the NEURON test fixtures raised "generator raised StopIteration").
"""
from __future__ import annotations
import os, pickle, re, sys, types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from fusezap import Tensor, helpers
from fusezap.device import Device, _DeviceRegistry, neuron, nkisim_worker, nrt_worker

# ---------------------------------------------------------------- fake nrtpy (the nrtpy 2.35.30 API, read from the wheel)

class NrtError(RuntimeError): pass

class _Ref:
  def __init__(self, mem: memoryview, name=None, base=None): self.mem, self.name, self.base = mem, name, base
  @property
  def root(self): return self.base.root if self.base is not None else self

class _Session:
  def __init__(self): self.execs, self.slices, self.last = 0, 0, None
  def slice_from_tensor(self, ref, offset=0, size=0, name=None):
    self.slices += 1
    return _Ref(ref.mem[offset:offset + (size or len(ref.mem) - offset)], name, ref)
  def execute(self, model_ref, inputs, outputs, save_trace=False, ntff_name=None):
    m = model_ref.model
    if set(inputs) != set(m.input_tensors_info) or set(outputs) != set(m.output_tensors_info):   # what libnrt rejected in run 1
      raise NrtError("NRT Error NRT_INVALID(2): Failed to execute model")
    for o, i in m.alias_info.items():
      if outputs[o].root is not inputs[i].root: raise NrtError("NRT Error NRT_INVALID(2): must-alias output is not its input")
    for n, r in [*inputs.items(), *outputs.items()]:
      want = (m.input_tensors_info.get(n) or m.output_tensors_info[n]).size
      if len(r.mem) != want: raise NrtError(f"NRT Error NRT_INVALID(2): {n} has {len(r.mem)} bytes, expected {want}")
    self.execs += 1
    self.last = (dict(inputs), dict(outputs))
    if m.prog is None: return
    from fusezap.device import nkishim
    arrays = []
    for k in range(m.prog.n_bufs):
      r = inputs.get(f"data{k}") or inputs.get(f"data{k}.must_alias_input") or outputs[f"data{k}"]
      arrays.append(np.frombuffer(r.mem, dtype=m.npd[k]))
    nkishim.execute(m.prog, arrays)

class NrtpyTensor:
  def __init__(self, tensor_ref, shape, dtype, name=None):
    self.tensor_ref, self.shape, self.dtype, self.name = tensor_ref, tuple(shape), np.dtype(dtype), name
  @classmethod
  def from_numpy(cls, a, name=None, core_id=0):
    a = np.ascontiguousarray(a)
    return cls(_Ref(memoryview(bytearray(a.tobytes())), name), a.shape, a.dtype, name)
  def write_from_numpy(self, a):
    a = np.ascontiguousarray(a)
    if a.nbytes != len(self.tensor_ref.mem): raise ValueError("Size mismatch")
    self.tensor_ref.mem[:] = a.tobytes()
  def numpy(self): return np.frombuffer(bytes(self.tensor_ref.mem), self.dtype).reshape(self.shape).copy()

def _info(n, d): return SimpleNamespace(shape=(n,), dtype=np.dtype(d).name, size=n * np.dtype(d).itemsize)

class NrtpyModel:
  """Loads either a fake NEFF written by FakeNki (a kernel to run through NKISHIM) or a dict of declared tensors."""
  loads = 0
  def __init__(self, d: dict):
    NrtpyModel.loads += 1
    self.model_ref = SimpleNamespace(model=self, core_id=0)
    if "tensors" in d:
      self.prog, (ins, outs, self.alias_info) = None, d["tensors"]
      self.input_tensors_info = {n: _info(*s) for n, s in ins.items()}
      self.output_tensors_info = {n: _info(*s) for n, s in outs.items()}
      return
    from fusezap.codegen.nki import _NL
    inv = {v: k for k, v in reversed(list(_NL.items()))}
    outs = tuple(int(x) for x in re.findall(r"data(\d+)", re.search(r"^\s*return (.+)$", d["src"], re.M).group(1)))
    specs = d["specs"]
    self.prog = SimpleNamespace(name=d["name"], src=d["src"], n_bufs=len(specs), outs=outs,
                                buf_dtypes={i: inv[t] for i, t in enumerate(d["dtypes"]) if t in inv and i < len(specs)})
    self.npd = [np.dtype(t) for _, t in specs]
    self.input_tensors_info = {(f"data{k}.must_alias_input" if k in outs else f"data{k}"): _info(n, t) for k, (n, t) in enumerate(specs)}
    self.output_tensors_info = {f"data{k}": _info(*specs[k]) for k in outs}
    self.alias_info = {f"data{k}": f"data{k}.must_alias_input" for k in outs}
  @classmethod
  def load_from_neff(cls, path, name=None, core_id=0):
    with open(path, "rb") as f: return cls(pickle.load(f))
  def benchmark(self, inputs, outputs, warmup_iter=10, benchmark_iter=100, mode="device"):   # nrtpy_model.NrtpyModel.benchmark
    sess = sys.modules["nrtpy"].get_nrtpy_singleton()
    for _ in range(benchmark_iter): sess.execute(self.model_ref, {k: t.tensor_ref for k, t in inputs.items()}, {k: t.tensor_ref for k, t in outputs.items()})
    return SimpleNamespace(mean_ms=0.01, min_ms=0.01, max_ms=0.01, std_dev_ms=0.0, iterations=benchmark_iter, durations_ms=[0.01] * benchmark_iter)
  def __call__(self, inputs, outputs=None):   # nrtpy_model.NrtpyModel.__call__: must-alias outputs become slices of their inputs
    sess, auto = sys.modules["nrtpy"].get_nrtpy_singleton(), outputs is None
    outputs = dict(outputs or {})
    for o, v in self.output_tensors_info.items():
      if o in self.alias_info and self.alias_info[o] in inputs:
        src = inputs[self.alias_info[o]]
        outputs[o] = NrtpyTensor(sess.slice_from_tensor(src.tensor_ref, name=o), src.shape, src.dtype, o)
      elif o not in outputs: outputs[o] = NrtpyTensor.from_numpy(np.zeros(v.size, np.uint8), o)
    sess.execute(self.model_ref, {k: t.tensor_ref for k, t in inputs.items()}, {k: t.tensor_ref for k, t in outputs.items()})
    return outputs if auto else None

def fake_nrtpy() -> types.ModuleType:
  m = types.ModuleType("nrtpy")
  sess = _Session()
  m.NrtpyTensor, m.NrtpyModel, m.NrtError, m.get_nrtpy_singleton, m.session = NrtpyTensor, NrtpyModel, NrtError, (lambda: sess), sess
  return m

@pytest.fixture
def nrt(monkeypatch):
  m = fake_nrtpy()
  monkeypatch.setitem(sys.modules, "nrtpy", m)
  monkeypatch.setattr(nrt_worker, "S", nrt_worker.State())
  NrtpyModel.loads = 0
  return m

def wcall(req: dict) -> dict:
  """nrt_worker.handle behind a pickle round trip, with main()'s error reply."""
  req = pickle.loads(pickle.dumps(req, protocol=5))
  try: rep = nrt_worker.handle(req)
  except Exception as e: rep = {"ok": False, "err": f"{type(e).__name__}: {e}", "tb": ""}
  return pickle.loads(pickle.dumps(rep, protocol=5))

# ---------------------------------------------------------------- binding

RUN1_IN = {"data2": ([524288], "float32"), "data1": ([131072], "float32"), "data0.must_alias_input": ([65536], "float32")}

def test_neff_binding_run1_names():
  """The run-1 matmul NEFF: inputs data0.must_alias_input / data1 / data2, output data0 aliasing argument 0."""
  ins, outs, al = neuron.neff_binding(RUN1_IN, {"data0": None}, {"data0": "data0.must_alias_input"}, 3)
  assert ins == {"data0.must_alias_input": 0, "data1": 1, "data2": 2} and outs == {} and al == {"data0": "data0.must_alias_input"}
  assert neuron.neff_binding(RUN1_IN, ["data0"], {}, 3) == (ins, outs, al)   # the suffix alone implies the pair
  assert neuron.neff_binding(["data1"], ["data0"], {}, 2) == ({"data1": 1}, {"data0": 0}, {})   # an output-only argument
  with pytest.raises(neuron.NeuronBindError, match="'x'"): neuron.neff_binding(["x"], [], {}, 2)
  with pytest.raises(neuron.NeuronBindError, match="data0..data1"): neuron.neff_binding(["data2"], [], {}, 2)
  with pytest.raises(neuron.NeuronBindError, match="does not declare"): neuron.neff_binding(["data1"], ["data0"], {"data0": "data0.must_alias_input"}, 2)

# ---------------------------------------------------------------- runtime worker against the fake nrtpy

def _load_run1(tmp_path: Path) -> None:
  p = tmp_path / "run1.neff"
  p.write_bytes(pickle.dumps({"tensors": ({"data0.must_alias_input": (4, "float32"), "data1": (8, "float32")},
                                          {"data0": (4, "float32")}, {"data0": "data0.must_alias_input"})}))
  rep = wcall({"op": "load", "key": "k", "path": str(p), "new": {1: 16, 2: 32}})
  assert rep["ok"] and rep["alias"] == {"data0": "data0.must_alias_input"} and rep["inputs"]["data1"]["size"] == 32

def test_nrt_worker_binds_every_input_and_aliases_outputs(nrt, tmp_path):
  _load_run1(tmp_path)
  bad = wcall({"op": "exec", "key": "k", "in": {"data1": 2}, "out": {}, "alias": {}})   # run 1's mapping
  assert not bad["ok"] and "data0.must_alias_input" in bad["err"]
  ins, outs, al = neuron.neff_binding(wcall({"op": "load", "key": "k", "path": "unused"})["inputs"], ["data0"], {"data0": "data0.must_alias_input"}, 2)
  req = {"op": "exec", "key": "k", "in": {n: i + 1 for n, i in ins.items()}, "out": {}, "alias": al, "iters": 3}
  rep = wcall(req)
  assert rep["ok"] and len(rep["seconds"]) == 3 and nrt.session.execs == 3
  i_, o_ = nrt.session.last
  assert o_["data0"].root is nrt_worker.S.bufs[1].tensor_ref and i_["data1"] is nrt_worker.S.bufs[2].tensor_ref
  assert nrt.session.slices == 1 and wcall(req)["ok"] and nrt.session.slices == 1   # bindings are built once
  assert wcall({"op": "write", "id": 2, "data": np.arange(8, dtype=np.float32).view(np.uint8)})["ok"]
  assert np.array_equal(wcall({"op": "read", "id": 2, "n": 32})["data"].view(np.float32), np.arange(8, dtype=np.float32))
  assert wcall({"op": "stats", "free": [2]})["buffers"] == 1 and not nrt_worker.S.refs   # a freed buffer drops its bindings
  assert wcall({"op": "load", "key": "k", "path": "unused"})["ok"] and NrtpyModel.loads == 1   # loaded once per key

def test_nrt_worker_slices_a_larger_buffer(nrt, tmp_path):
  _load_run1(tmp_path)
  rep = wcall({"op": "exec", "key": "k", "in": {"data0.must_alias_input": 1, "data1": 3}, "out": {}, "alias": {"data0": "data0.must_alias_input"}, "new": {3: 64}})
  assert rep["ok"] and len(nrt.session.last[0]["data1"].mem) == 32
  rep = wcall({"op": "exec", "key": "k", "in": {"data0.must_alias_input": 1, "data1": 4}, "out": {}, "alias": {"data0": "data0.must_alias_input"}, "new": {4: 8}})
  assert not rep["ok"] and "needs 32" in rep["err"]

def test_nrt_worker_batch_carries_writes_and_a_read(nrt, tmp_path):
  """N7: one batch = the step's queued writes, its launches and its token read, in order; the reply carries the read's data."""
  _load_run1(tmp_path)
  ins, _, al = neuron.neff_binding(wcall({"op": "load", "key": "k", "path": "unused"})["inputs"], ["data0"], {"data0": "data0.must_alias_input"}, 2)
  x = np.arange(8, dtype=np.float32)
  reqs = [{"op": "write", "id": 2, "data": x.view(np.uint8)}, {"op": "exec", "key": "k", "in": {n: i + 1 for n, i in ins.items()}, "out": {}, "alias": al},
          {"op": "read", "id": 2, "n": 32}]
  rep = wcall({"op": "batch", "reqs": reqs})
  assert rep["ok"] and rep["n"] == 3 and len(rep["seconds"]) == 1 and nrt.session.execs == 1
  assert np.array_equal(np.frombuffer(rep["data"], np.float32), x)
  assert "data" not in wcall({"op": "batch", "reqs": reqs[1:2]})

class _Rec:
  def __init__(self): self.reqs = []
  def call(self, req: dict) -> dict:
    self.reqs.append(pickle.loads(pickle.dumps(req, protocol=5)))
    return {"ok": True, "seconds": [], "data": np.zeros(16, np.uint8)} if req["op"] in ("read", "batch") else {"ok": True}

def test_allocator_queues_small_writes_and_reads_once(monkeypatch):
  """N7 (FZ_NEURON_HOSTFAST): a small write is queued (a copy: the caller may reuse its array) and goes with the next read
  as one batch; a write over FZ_NEURON_QWRITE_MAX is sent at once; with the switch off every write and read is its own trip."""
  monkeypatch.setenv("FZ_NEURON_QWRITE_MAX", "64")
  for fast in ("1", "0"):
    monkeypatch.setenv("FZ_NEURON_HOSTFAST", fast)
    w = _Rec()
    al = neuron.NeuronAllocator(worker=w)
    b, big = al.alloc(16), al.alloc(128)
    src = np.arange(16, dtype=np.uint8)
    al.copyin(b, memoryview(src)); src[:] = 99
    al.call({"op": "exec", "key": "k", "in": {}, "out": {}, "alias": {}})
    out = memoryview(bytearray(16))
    al.copyout(out, b)
    al.copyin(big, memoryview(np.ones(128, np.uint8)))
    ops = [r["op"] for r in w.reqs]
    if fast == "1":
      assert ops == ["batch", "write"] and al.trips == 2
      assert [r["op"] for r in w.reqs[0]["reqs"]] == ["write", "exec", "read"] and w.reqs[0]["reqs"][0]["new"] == {b.id: 16, big.id: 128}
      assert np.array_equal(w.reqs[0]["reqs"][0]["data"], np.arange(16, dtype=np.uint8))
    else: assert ops == ["write", "batch", "read", "write"] and al.trips == 4

# ---------------------------------------------------------------- compile: unique directories, the NEFF cache

class _FakeCompile:
  """nki 0.6 StandaloneKernel stand-in: compiles into artifacts_dir (else $NKI_ARTIFACTS_DIR) and fails like neuronx-cc on a
  directory that already holds artifacts (NCC_IDRV077); then calls the executor, as run_from_bir does."""
  def __init__(self, executor=None, artifacts_dir=""): self.executor, self.dir, self.dirs = executor, artifacts_dir, []
  def __call__(self, *xs):
    d = self.dir or os.environ["NKI_ARTIFACTS_DIR"]
    if os.path.exists(os.path.join(d, "kernel.neff")): raise RuntimeError(f"[NCC_IDRV077] Compilation artifacts already exist in the output directory ({d})")
    Path(d, "kernel.neff").write_bytes(b"NEFF")
    self.dirs.append(d)
    if self.executor: self.executor(SimpleNamespace(neff_path="kernel.neff", input_specs=[SimpleNamespace(name="data0.must_alias_input")],
                                                    output_specs=[SimpleNamespace(name="data0")], input_output_aliases={0: "data0"}), {}, {})

class _FakeKernel:
  def __init__(self): self.made = []
  def _to_subclass(self, cls, _frontend_cls=None, _executor=None, artifacts_dir=""):
    self.made.append(_FakeCompile(_executor, artifacts_dir)); return self.made[-1]

def test_worker_compiles_each_neff_in_a_fresh_dir(monkeypatch, tmp_path):
  for name in ("nki", "nki.compiler", "nki.compiler.frontend", "nki.framework", "nki.framework.compiled"):
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
  sys.modules["nki.compiler.frontend"].resolve_frontend_cls = lambda: object   # type: ignore[attr-defined]
  sys.modules["nki.framework.compiled"].StandaloneKernel = object              # type: ignore[attr-defined]
  stale = tmp_path / "stale"; stale.mkdir(); (stale / "kernel.neff").write_bytes(b"old")
  monkeypatch.setenv("NKI_ARTIFACTS_DIR", str(stale))   # what the run-1 worker left behind after its first NEFF
  k, dirs = _FakeKernel(), []
  for _ in range(3):
    w = nkisim_worker._fresh_dir()
    path, io, how, err = nkisim_worker._compile_neff(k, [], w)
    assert err is None and how == "compile_only" and path == os.path.join(w, "kernel.neff") and io["aliases"] == {"0": "data0"}
    dirs.append(w)
  assert len(set(dirs)) == 3 and os.environ["NKI_ARTIFACTS_DIR"] == str(stale)   # never written over, never left changed
  jit = _FakeCompile()   # no _to_subclass: the nki.jit fallback reads NKI_ARTIFACTS_DIR, set only for the call
  path, io, how, err = nkisim_worker._compile_neff(jit, [], w2 := nkisim_worker._fresh_dir())
  assert how == "jit_fallback" and err is None and jit.dirs == [w2] and os.environ["NKI_ARTIFACTS_DIR"] == str(stale)
  monkeypatch.delenv("NKI_ARTIFACTS_DIR")
  nkisim_worker._compile_neff(_FakeCompile(), [], nkisim_worker._fresh_dir())
  assert "NKI_ARTIFACTS_DIR" not in os.environ

def test_neff_cache_compiles_once(monkeypatch, tmp_path):
  monkeypatch.setenv("FZ_NEFF_CACHE", str(tmp_path / "neff"))
  calls = []
  def fake(prog, arrays, target="inf2", out_dir=None):
    calls.append(arrays); return {"neff": b"N" * 10, "seconds": 0.5, "io": {"inputs": ["data0"]}, "how": "compile_only"}
  monkeypatch.setattr(neuron, "_compile_neff_rep", fake)
  prog = SimpleNamespace(name="E_4", src="def E_4(data0): ...")
  v = {"nki": "0.6.0", "neuronxcc": "2.27"}
  p1, m1 = neuron.cached_neff(prog, [(4, "<f4")], "inf2", v)
  p2, m2 = neuron.cached_neff(prog, [(4, "<f4")], "inf2", v)
  assert p1 == p2 and p1.read_bytes() == b"N" * 10 and len(calls) == 1 and m2["key"] == m1["key"] == p1.parent.name
  assert neuron.cached_neff(prog, [(8, "<f4")], "inf2", v)[0] != p1 and len(calls) == 2            # other sizes: another NEFF
  assert neuron.cached_neff(prog, [(4, "<f4")], "inf2", {**v, "nki": "0.7"})[0] != p1 and len(calls) == 3
  assert sorted(x.name for x in (tmp_path / "neff").iterdir() if x.name.startswith(".")) == []   # no temp dirs left
  def racing(prog, arrays, target="inf2", out_dir=None):   # another process writes the same key while we compile
    d = Path(os.environ["FZ_NEFF_CACHE"]) / neuron.neff_key(prog, arrays, target, v); d.mkdir(); (d / "kernel.neff").write_bytes(b"theirs")
    return fake(prog, arrays, target)
  monkeypatch.setattr(neuron, "_compile_neff_rep", racing)
  p, _ = neuron.cached_neff(prog, [(16, "<f4")], "inf2", v)
  assert p.read_bytes() == b"theirs" and not [x for x in p.parent.parent.iterdir() if x.name.startswith(".")]

def test_neuron_target_follows_the_instance_type(monkeypatch, tmp_path):
  """N6b: a trn1 host compiles for trn1 (DMI product name = the EC2 instance type), anything else for inf2; FZ_NEURON_TARGET
  wins; the target is part of the NEFF key."""
  monkeypatch.delenv("FZ_NEURON_TARGET", raising=False)
  f = tmp_path / "product_name"
  for name, want in (("trn1.2xlarge\n", "trn1"), ("trn1n.32xlarge", "trn1"), ("inf2.xlarge", "inf2"), ("Standard PC", "inf2")):
    f.write_text(name)
    assert neuron.neuron_target(str(f)) == want
  assert neuron.neuron_target(str(tmp_path / "missing")) == "inf2"
  monkeypatch.setenv("FZ_NEURON_TARGET", "inf2")
  f.write_text("trn1.2xlarge")
  assert neuron.neuron_target(str(f)) == "inf2"
  prog = SimpleNamespace(name="E_4", src="def E_4(data0): ...")
  assert neuron.neff_key(prog, [(4, "<f4")], "inf2", {}) != neuron.neff_key(prog, [(4, "<f4")], "trn1", {})

# ---------------------------------------------------------------- the NEURON device on the persistent path, end to end

class FakeNki:
  """The .venv-nki worker: 'compiles' a kernel to a fake NEFF that names its source, argument sizes and dtypes."""
  def __init__(self): self.neffs = 0
  def call(self, req: dict) -> dict:
    if req["op"] == "version": return {"ok": True, "nki": "0.6.0+fake", "neuronxcc": "2.27+fake"}
    assert req["op"] == "neff", req["op"]   # the persistent path never asks this worker to run anything
    self.neffs += 1
    specs = [(int(a[0]), np.dtype(a[1]).str) if isinstance(a, tuple) else (a.size, a.dtype.str) for a in req["arrays"]]
    return {"ok": True, "neff": pickle.dumps({"src": req["src"], "name": req["name"], "specs": specs, "dtypes": req["dtypes"]}),
            "seconds": 0.01, "io": {}, "how": "fake"}

class FakeNrt:
  def call(self, req: dict) -> dict: return wcall(req)

@pytest.fixture
def neuron_dev(nrt, monkeypatch, tmp_path):
  monkeypatch.setenv("FZ_NEFF_CACHE", str(tmp_path / "neff"))
  monkeypatch.delenv("FZ_NEURON_RT", raising=False)
  monkeypatch.delenv("FZ_INTERP", raising=False)
  monkeypatch.setattr(neuron, "neuron_device_present", lambda: True)
  monkeypatch.setattr(neuron, "nki_python", lambda: Path(sys.executable))
  monkeypatch.setattr(neuron, "nrt_python", lambda: Path(sys.executable))
  nki = FakeNki()
  monkeypatch.setattr(neuron.NKIWorker, "get", classmethod(lambda cls: nki))
  monkeypatch.setattr(neuron.NrtWorker, "get", classmethod(lambda cls: FakeNrt()))
  neuron._VERSIONS.clear()
  helpers.getenv.cache_clear()
  _DeviceRegistry.__getitem__.cache_clear()
  from fusezap.serve import neuron_kv   # device constants cached per device name would point at an earlier fake worker's ids
  monkeypatch.setattr(neuron_kv, "_REV", {}); monkeypatch.setattr(neuron_kv, "_FENCE_BUF", {})
  monkeypatch.setattr(Device, "DEFAULT", "NEURON", raising=False)
  dev = Device["NEURON"]
  yield dev, nki
  _DeviceRegistry.__getitem__.cache_clear()
  neuron._VERSIONS.clear()
  helpers.getenv.cache_clear()

def test_neuron_persistent_path_end_to_end(neuron_dev, nrt):
  dev, nki = neuron_dev
  assert isinstance(dev.allocator, neuron.NeuronAllocator) and dev.runtime is neuron.NeuronNeffProgram
  R = np.random.default_rng(3)
  a, w = R.uniform(-1, 1, (64, 96)).astype(np.float32), R.uniform(-1, 1, (96, 40)).astype(np.float32)
  def graph():
    x = Tensor(a)
    return (((x * 2 + 1).exp() @ Tensor(w)).relu().sum(axis=1)).numpy()
  ref = np.maximum(np.exp(a.astype(np.float64) * 2 + 1) @ w, 0).sum(1)
  calls0 = dict(dev.allocator.calls)
  np.testing.assert_allclose(graph(), ref, rtol=1e-4)
  kernels, execs = NrtpyModel.loads, nrt.session.execs
  assert kernels >= 2 and nki.neffs == kernels and execs >= kernels
  c = {k: v - calls0[k] for k, v in dev.allocator.calls.items()}
  assert c["write"] == 2 and c["read"] == 1 and c["load"] == kernels   # host traffic only at the graph boundary
  np.testing.assert_allclose(graph(), ref, rtol=1e-4)   # warm: no compile, no load, only launches
  assert nki.neffs == kernels and NrtpyModel.loads == kernels and nrt.session.execs == 2 * execs
  dev._cache.clear()   # a new process: the NEFFs come from the disk cache (one load each, no compile)
  hits = neuron.NEFF_STATS["hits"]
  nrt_worker.S.models.clear()
  np.testing.assert_allclose(graph(), ref, rtol=1e-4)
  assert nki.neffs == kernels and neuron.NEFF_STATS["hits"] - hits == kernels and NrtpyModel.loads == 2 * kernels

def test_neuron_persistent_path_custom_kernels_and_zero_size(neuron_dev):
  dev, _ = neuron_dev
  R = np.random.default_rng(5)
  q, k, v = (R.standard_normal(s).astype(np.float32) for s in ((1, 4, 9, 16), (1, 2, 13, 16), (1, 2, 13, 16)))
  out = neuron.attention(Tensor(q), Tensor(k), Tensor(v), causal=True, coff=4).numpy()
  kk, vv = np.repeat(k, 2, 1).astype(np.float64), np.repeat(v, 2, 1).astype(np.float64)
  s = q @ kk.swapaxes(-1, -2) / 4.0
  s = np.where(np.arange(13)[None, :] <= np.arange(9)[:, None] + 4, s, -np.inf)
  e = np.exp(s - s.max(-1, keepdims=True))
  np.testing.assert_allclose(out, (e / e.sum(-1, keepdims=True)) @ vv, rtol=1e-4, atol=1e-5)
  x, w = R.standard_normal((5, 33)).astype(np.float32), R.standard_normal((33, 7)).astype(np.float32)
  np.testing.assert_allclose((Tensor(x) @ Tensor(w)).numpy(), x @ w, rtol=1e-4, atol=1e-5)
  assert any(m[0].prog.name.startswith("fz_matmul") for m in nrt_worker.S.models.values())
  assert (Tensor(np.zeros((0, 3), np.float32)) + 1).numpy().shape == (0, 3)

def test_neuron_rt_jit_fallback_selectable(neuron_dev, monkeypatch):
  monkeypatch.setenv("FZ_NEURON_RT", "jit")
  d = neuron.NEURONDevice("NEURON")
  assert d.runtime is neuron.NeuronHwProgram and not isinstance(d.allocator, neuron.NeuronAllocator)
  monkeypatch.setenv("FZ_NEURON_RT", "bogus")
  with pytest.raises(RuntimeError, match="FZ_NEURON_RT"): neuron.NEURONDevice("NEURON")

def test_n4_nrtpy_script_passes_the_must_alias_input(nrt, tmp_path):
  """scripts/nki_n4_nrtpy.py on run 1's matmul NEFF names: the first run must bind data0.must_alias_input (run 1: NRT_INVALID)."""
  import importlib.util
  path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "nki_n4_nrtpy.py")
  spec = importlib.util.spec_from_file_location("nki_n4_nrtpy", path)
  script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)   # type: ignore[union-attr]
  d = tmp_path / "matmul"; d.mkdir()
  (d / "kernel.neff").write_bytes(pickle.dumps({"tensors": ({"data0.must_alias_input": (4, "float32"), "data1": (8, "float32"), "data2": (6, "float32")},
                                                            {"data0": (4, "float32")}, {"data0": "data0.must_alias_input"})}))
  np.savez(d / "io.npz", in0=np.zeros(4, np.float32), in1=np.ones(8, np.float32), in2=np.ones(6, np.float32), out0=np.zeros(4, np.float32))
  script.run_case(str(d), 3)
  r = script.RES["cases"]["matmul"]
  assert r["arg_mapping"] == {"data0.must_alias_input": "data0", "data1": "data1", "data2": "data2"}
  assert "error" not in r["first_run"] and r["first_run"]["data0"]["bitwise_equal_to_sim"] == 1.0, r["first_run"]
  assert "error" not in r["preallocated_outputs"] and "error" not in r["call_warm_host_loop"]

def test_n4_probe_nrt_section_dry_run(neuron_dev, capsys):
  """scripts/nki_n4_probe.py's first rerun section (loaded-NEFF launch latency) runs end to end on the fake runtime."""
  import importlib.util
  path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "nki_n4_probe.py")
  spec = importlib.util.spec_from_file_location("nki_n4_probe", path)
  probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)   # type: ignore[union-attr]
  probe.sec_nrt(5)
  rows = probe.OUT["sections"]["nrt"]
  adds = [r for r in rows if "first_launch_s" in r]
  assert len(adds) == 2 and all(r["correct"] and r["chain_200_launches"]["correct"] and r["compiled"] == 1 for r in adds), rows
  assert adds[0]["neff_io"]["alias"] == {"data0": "data0.must_alias_input"}
  assert len([r for r in rows if r.get("what", "").startswith("copy")]) == 3
  probe.summary()
  assert "dependent chain" in capsys.readouterr().out

def test_n3_gate_runs_on_the_persistent_path(neuron_dev):
  """scripts/nki_n3_gate.py on the persistent NEURON path (N4 run 2: its matmul cases called the program on numpy arrays and
  raised "NeuronNeffProgram runs on device buffers"): a 0.5B qkv matmul and a decode attention, within the gate tolerances."""
  import importlib.util
  path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "nki_n3_gate.py")
  spec = importlib.util.spec_from_file_location("nki_n3_gate", path)
  gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)   # type: ignore[union-attr]
  dev, _ = neuron_dev
  gate.capture(dev)
  gate.mm_case(dev, "0.5B", "qkv", 1, 896, 1152, "kmajor", False)
  gate.mm_case(dev, "0.5B", "down", 3, 640, 96, "torch", False)
  gate.attn_case(dev, "0.5B", "pos", 1, 96, 90, False)
  assert len(gate.ROWS) == 3 and all(r["ok"] for r in gate.ROWS), gate.ROWS

# ---------------------------------------------------------------- run 1: the fixture's StopIteration

def test_default_device_falls_back_to_npy(monkeypatch):
  """On the inf2 DLAMI no METAL/CUDA/CPU (no clang) is available: DEFAULT must be NPY, not a StopIteration that turns every
  generator fixture that reads it (monkeypatch.setattr(Device, "DEFAULT", ...)) into "generator raised StopIteration"."""
  monkeypatch.delenv("FZ_DEVICE", raising=False)
  helpers.getenv.cache_clear()
  reg = _DeviceRegistry()
  monkeypatch.setattr(reg, "available", lambda name: False)
  assert reg.DEFAULT == "NPY"
  def fixture_like():
    monkeypatch.setattr(reg, "DEFAULT", "NEURON", raising=False)
    yield reg.DEFAULT
  reg2 = _DeviceRegistry(); monkeypatch.setattr(reg2, "available", lambda name: False); reg = reg2
  assert next(fixture_like()) == "NEURON"
  helpers.getenv.cache_clear()

# ---------------------------------------------------------------- N8 (I71): HBM budget, refused allocations, accounting

def test_allocator_budget_frees_the_cache_then_refuses(monkeypatch):
  """FZ_NEURON_HBM_GB: an allocation past the budget first hands the free list back (LRUAllocator retries once), then raises
  NeuronOOM, a MemoryError, before anything reaches the runtime worker; FZ_NEURON_ALLOC_CACHE_MB caps the free list."""
  monkeypatch.setenv("FZ_NEURON_HBM_GB", str(1000 / 2**30))   # 1000 bytes
  monkeypatch.setenv("FZ_NEURON_ALLOC_CACHE_MB", str(300 / 2**20))   # 300 bytes
  al = neuron.NeuronAllocator(worker=_Rec())
  a, b = al.alloc(400), al.alloc(400)
  al.free(a, 400)                     # cached (400 > the 300-byte cap: the least recently used size goes back... itself, alone over the cap)
  assert al.cached <= 300
  c = al.alloc(200)
  al.free(c, 200)
  assert al.cached == 200 and al.held == 600
  d = al.alloc(500)                   # 600 + 500 > 1000 with 200 cached: the cache goes back, then it fits
  assert al.held == 900 and al.cached == 0 and al.peak == 900
  with pytest.raises(neuron.NeuronOOM) as e: al.alloc(300)
  assert isinstance(e.value, MemoryError) and "HBM budget" in str(e.value) and al.ooms >= 1
  al.free(b, 400); al.free_cache()
  assert al.alloc(300) and al.held == 800
  monkeypatch.setenv("FZ_NEURON_HBM_GB", "0")
  assert neuron.NeuronAllocator(worker=_Rec()).budget == 0   # 0: no budget

def test_nrt_worker_refuses_a_batch_before_running_any_of_it(nrt, tmp_path, monkeypatch):
  """I69 run 1 failed at launch 3 of a batch of 4 (NRT_RESOURCE), after launches 0..2 had run. Now every new buffer of a
  batch is allocated before its first launch; if the runtime refuses one, the ones made are freed, nothing runs, and the
  reply says oom (the host raises NeuronOOM)."""
  _load_run1(tmp_path)
  ins, _, al_ = neuron.neff_binding(wcall({"op": "load", "key": "k", "path": "unused"})["inputs"], ["data0"], {"data0": "data0.must_alias_input"}, 2)
  real = NrtpyTensor.from_numpy.__func__
  def limited(cls, a, name=None, core_id=0):
    if sum(len(t.tensor_ref.mem) for t in nrt_worker.S.bufs.values()) + a.nbytes > 200: raise NrtError("NRT Error NRT_RESOURCE(4): Failed to allocate tensor")
    return real(cls, a, name, core_id)
  monkeypatch.setattr(NrtpyTensor, "from_numpy", classmethod(limited))
  ex = {"op": "exec", "key": "k", "in": {n: i + 1 for n, i in ins.items()}, "out": {}, "alias": al_}
  before = set(nrt_worker.S.bufs)
  reqs = [dict(ex), {**ex, "new": {10: 64}}, {**ex, "new": {11: 64}}, {**ex, "new": {12: 256}}]
  req = {"op": "batch", "reqs": reqs}
  try: nrt_worker.handle(pickle.loads(pickle.dumps(req, protocol=5))); rep = {"ok": True}
  except nrt_worker.OOM as e: rep = {"ok": False, "oom": True, "err": str(e)}
  assert rep == {"ok": False, "oom": True, "err": rep["err"]} and "NRT_RESOURCE" in rep["err"]
  assert nrt.session.execs == 0 and set(nrt_worker.S.bufs) == before   # nothing ran, nothing kept
  w = _Rec()
  w.call = lambda r: {"ok": False, "oom": True, "err": "OOM: NRT_RESOURCE"}
  host = neuron.NeuronAllocator(worker=w)
  host.alloc(64)
  with pytest.raises(neuron.NeuronOOM, match="nothing of the request ran"): host.call({"op": "read", "id": 1, "n": 4})

def test_api_maps_out_of_memory_to_503():
  """A MemoryError from the engine (NEURON's budget, or the runtime's refusal) is HTTP 503 engine_out_of_memory, retryable,
  not 500; a ValueError stays 400; the engine process reports it as "oom" over its pipe."""
  from fusezap.serve import api
  e = api._api_error(neuron.NeuronOOM("no room"), "logprobs: ")
  assert e.status == 503 and e.code == "engine_out_of_memory" and e.message == "logprobs: no room"
  assert api._api_error(ValueError("bad")).status == 400 and api._api_error(RuntimeError("x")).status == 500
  assert api._err_kind(MemoryError()) == "oom" and api._err_kind(ValueError()) is True and api._err_kind(RuntimeError()) is False
