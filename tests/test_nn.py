import numpy as np, pytest, torch
import fusezap.nn as nn
from fusezap import Tensor, manual_seed
from fusezap.optim import SGD, AdamW

def _t(a): return Tensor(np.ascontiguousarray(a, np.float32))

def test_linear_embedding_norms(device, assert_close):
  manual_seed(0)
  lin, emb, ln, rms = nn.Linear(5, 3), nn.Embedding(7, 4), nn.LayerNorm(4), nn.RMSNorm(4)
  x = np.random.default_rng(0).standard_normal((2, 5)).astype(np.float32)
  tl = torch.nn.Linear(5, 3); tl.weight.data = torch.tensor(lin.weight.numpy()); tl.bias.data = torch.tensor(lin.bias.numpy())
  assert_close(lin(_t(x)).numpy(), tl(torch.tensor(x)).detach(), reduce=True)
  assert float(np.abs(lin.weight.numpy()).max()) <= 1 / np.sqrt(5) + 1e-6
  idx = np.array([[1, 6], [0, 3]], np.int32)
  assert_close(emb(Tensor(idx)).numpy(), torch.tensor(emb.weight.numpy())[torch.tensor(idx).long()])
  y = np.random.default_rng(1).standard_normal((3, 4)).astype(np.float32)
  assert_close(ln(_t(y)).numpy(), torch.nn.functional.layer_norm(torch.tensor(y), (4,)), reduce=True)
  ty = torch.tensor(y)
  assert_close(rms(_t(y)).numpy(), ty * torch.rsqrt(ty.pow(2).mean(-1, keepdim=True) + 1e-6), reduce=True)

def test_dropout(device):
  d = nn.Dropout(0.5)
  x = Tensor.ones(4, 4)
  assert (d(x).numpy() == 1).all()
  Tensor.training = True
  try: assert set(np.unique(d(x).numpy())) <= {0.0, 2.0}
  finally: Tensor.training = False

class M:
  def __init__(self):
    self.a, self.layers = nn.Linear(3, 4), [nn.Linear(4, 2), nn.LayerNorm(2)]
    self.d = {"e": nn.Embedding(5, 2)}

def test_state_dict(device):
  m = M()
  keys = list(nn.get_state_dict(m))
  assert keys == ["a.weight", "a.bias", "layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias", "d.e.weight"]
  assert len(nn.get_parameters(m)) == 7
  sd = {k: np.full(v.shape, i, np.float32) for i, (k, v) in enumerate(nn.get_state_dict(m).items())}
  nn.load_state_dict(m, sd)
  assert (m.layers[0].bias.numpy() == 3).all()
  with pytest.raises(AssertionError): nn.load_state_dict(m, {"a.weight": sd["a.weight"]})
  nn.load_state_dict(m, {"a.weight": np.zeros((4, 3), np.float32)}, strict=False)
  assert (m.a.weight.numpy() == 0).all()

@pytest.mark.parametrize("name,kw", [("sgd", {}), ("sgd", dict(momentum=0.9, weight_decay=0.1)), ("sgd", dict(momentum=0.9, nesterov=True)),
                                     ("adamw", {}), ("adamw", dict(weight_decay=0.1, betas=(0.8, 0.9)))])
def test_optim_vs_torch(device, name, kw):
  w0 = np.random.default_rng(0).standard_normal((3, 4)).astype(np.float32)
  x = np.random.default_rng(1).standard_normal((5, 4)).astype(np.float32)
  w = Tensor(w0.copy(), requires_grad=True)
  tw = torch.tensor(w0.copy(), requires_grad=True)
  opt = (SGD if name == "sgd" else AdamW)([w], lr=1e-2, **kw)
  topt = (torch.optim.SGD if name == "sgd" else torch.optim.AdamW)([tw], lr=1e-2, **kw)
  for _ in range(5):
    opt.zero_grad(); ((Tensor(x) @ w.T) ** 2).mean().backward(); opt.step()
    topt.zero_grad(); ((torch.tensor(x) @ tw.T) ** 2).mean().backward(); topt.step()
  np.testing.assert_allclose(w.numpy(), tw.detach().numpy(), rtol=1e-5, atol=1e-6)

def test_adamw_per_param_step_count(device):
  w0, u0 = np.ones((2, 2), np.float32), np.ones((2, 2), np.float32)
  w, u, tw, tu = Tensor(w0.copy()), Tensor(u0.copy()), torch.tensor(w0.copy(), requires_grad=True), torch.tensor(u0.copy(), requires_grad=True)
  opt, topt = AdamW([w, u], lr=0.1), torch.optim.AdamW([tw, tu], lr=0.1)
  for i in range(4):
    opt.zero_grad(); topt.zero_grad()
    l = (w * w).sum() + ((u * u).sum() if i >= 2 else 0); l.backward()
    tl = (tw * tw).sum() + ((tu * tu).sum() if i >= 2 else 0); tl.backward()
    opt.step(); topt.step()
  np.testing.assert_allclose(u.numpy(), tu.detach().numpy(), rtol=1e-5)
  np.testing.assert_allclose(w.numpy(), tw.detach().numpy(), rtol=1e-5)

def test_rmsnorm_fp16_large(device):
  x = (300 * np.random.default_rng(0).standard_normal((4, 64))).astype(np.float16)
  y = nn.RMSNorm(64)(Tensor(x))
  assert y.dtype == Tensor(x).dtype
  ref = torch.nn.functional.rms_norm(torch.tensor(x), (64,), eps=1e-6).numpy()
  np.testing.assert_allclose(y.numpy().astype(np.float32), ref.astype(np.float32), rtol=2e-2, atol=2e-2)

class _Tied:
  def __init__(self):
    self.tok = nn.Embedding(5, 3); self.head = nn.Linear(3, 5, bias=False); self.head.weight = self.tok.weight

def test_tied_state_dict_names(device):
  m = _Tied()
  sd = nn.get_state_dict(m)
  assert list(sd) == ["tok.weight", "head.weight"]
  assert len(nn.get_parameters(m)) == 1
  nn.load_state_dict(m, {k: np.zeros((5, 3), np.float32) for k in sd})
  assert (m.tok.weight.numpy() == 0).all()

def test_cycle_walk(device):
  class Blk:
    def __init__(self, p): self.parent, self.w = p, Tensor.ones(2, requires_grad=True)
  class Top:
    def __init__(self): self.b = Blk(self); self.v = Tensor.ones(3, requires_grad=True)
  m = Top()
  assert list(nn.get_state_dict(m)) == ["b.w", "v"]
  assert len(nn.get_parameters(m)) == 2
  nn.load_state_dict(m, {"b.w": np.zeros(2, np.float32), "v": np.zeros(3, np.float32)})

def test_sdpa_function(device):
  r = np.random.default_rng(0)
  q, k, v = (r.standard_normal((1, 2, 4, 8)).astype(np.float32) for _ in range(3))
  out = nn.scaled_dot_product_attention(_t(q), _t(k), _t(v), is_causal=True).numpy()
  ref = torch.nn.functional.scaled_dot_product_attention(*(torch.tensor(a) for a in (q, k, v)), is_causal=True).numpy()
  np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-5)

def test_optimizer_single_realize_call(device, monkeypatch):
  import fusezap.realize as R
  calls = []
  orig = R.realize
  monkeypatch.setattr(R, "realize", lambda lbs: (calls.append(len(lbs)), orig(lbs))[1])
  ps = [Tensor.ones(3, requires_grad=True) for _ in range(3)]
  opt = AdamW(ps, lr=0.1)
  sum((p * p).sum() for p in ps).backward()
  calls.clear(); opt.step()
  assert len(calls) == 1 and calls[0] >= 3
