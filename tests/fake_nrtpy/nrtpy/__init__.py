"""A minimal nrtpy for tests/test_neuron_tp.py's two-process runtime-worker runs (FZ_NRT_PYTHON = this interpreter, this
directory on PYTHONPATH): device tensors are host byte arrays; enough for nrt_worker's alloc / write / read / "ar"."""
import numpy as np
class NrtpyTensor:
  def __init__(self, ref, shape, dtype, name=None): self.tensor_ref, self.shape, self.dtype, self.name = ref, tuple(shape), np.dtype(dtype), name
  @classmethod
  def from_numpy(cls, a, name=None, core_id=0):
    a = np.ascontiguousarray(a); return cls(bytearray(a.tobytes()), a.shape, a.dtype, name)
  def write_from_numpy(self, a):
    a = np.ascontiguousarray(a)
    assert a.nbytes == len(self.tensor_ref); self.tensor_ref[:] = a.tobytes()
  def numpy(self): return np.frombuffer(bytes(self.tensor_ref), self.dtype).reshape(self.shape).copy()
def get_nrtpy_singleton(): return None
