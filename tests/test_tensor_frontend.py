"""WP2: Tensor frontend edge cases beyond tests/test_ops.py, against torch. Uses the `device` fixture (all tiers)."""
from __future__ import annotations
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from fusezap import Tensor, dtypes, no_grad

R = np.random.default_rng(7)

def rnd(*shape): return R.uniform(-2, 2, size=shape).astype(np.float32)

def close(a, b, tol=1e-5):
  np.testing.assert_allclose(a.numpy() if isinstance(a, Tensor) else a, b.detach().numpy(), rtol=tol, atol=tol)

GET = [lambda t, i: t[i], lambda t, i: t[:, i], lambda t, i: t[None, i], lambda t, i: t[1, :, i],
       lambda t, i: t[..., i], lambda t, i: t[i, None, 1:], lambda t, i: t[1:3, i, 0], lambda t, i: t[1:, None, i]]

@pytest.mark.parametrize("k", range(len(GET)))
def test_getitem_tensor_mixed(device, k):
  a = rnd(4, 5, 3); idx = np.array([[0, 2], [-1, 1]], np.int32)
  t = torch.tensor(a); ti = torch.tensor(idx, dtype=torch.long)
  x = Tensor(a); xi = Tensor(idx)
  try: e = GET[k](t, ti)
  except IndexError: pytest.skip("torch rejects")
  close(GET[k](x, xi), e)
  np.testing.assert_array_equal(x[::-1, xi].numpy(), a[::-1][:, idx])

def test_getitem_errors(device):
  x = Tensor(rnd(3, 4))
  with pytest.raises(IndexError): x[3]
  with pytest.raises(IndexError): x[0, 0, 0]

def test_getitem_steps_exhaustive(device):
  a = rnd(7)
  for start in (None, 0, 2, -3):
    for stop in (None, 5, -1, 0):
      for step in (1, 2, 3, -1, -2, -4):
        s = slice(start, stop, step)
        np.testing.assert_array_equal(Tensor(a)[s].numpy(), a[s])

def test_getitem_slice_backward(device):
  a = rnd(6, 5); g = rnd(2, 3)
  x = Tensor(a, requires_grad=True)
  x[4:0:-2, ::2].backward(Tensor(g))
  e = np.zeros((6, 5), np.float32); e[4:0:-2, ::2] = g  # torch has no negative steps: numpy reference
  np.testing.assert_array_equal(x.grad.numpy(), e)

def test_weak_scalars_and_promotion(device):
  b = Tensor(np.array([True, False]))
  assert (b + 1).dtype == dtypes.int32 and (b * 1.5).dtype == dtypes.float32
  i8 = Tensor(np.array([1, 2], np.int8)); assert (i8 + 300).dtype == dtypes.int8
  assert (Tensor(np.array([1, 2], np.int8)) + Tensor(np.array([1], np.int16))).dtype == dtypes.int16
  assert (Tensor(np.ones(2, np.float16)) + Tensor(np.ones(2, np.float32)).cast(dtypes.bfloat16)).dtype == dtypes.float32
  assert Tensor([1, 2]).dtype == dtypes.int32 and Tensor([1.0, 2]).dtype == dtypes.float32 and Tensor([True]).dtype == dtypes.bool
  assert Tensor(np.float64(1.5)).dtype == dtypes.float32 and Tensor(True).dtype == dtypes.bool

def test_axis_dim_keywords(device):
  a = rnd(3, 4); x, t = Tensor(a), torch.tensor(a)
  close(x.sum(dim=1), t.sum(dim=1), 1e-4); close(x.softmax(dim=0), t.softmax(dim=0), 1e-4)
  close(x.max(dim=1, keepdim=True), t.amax(dim=1, keepdim=True)); close(x.unsqueeze(dim=0), t.unsqueeze(dim=0))

def test_pow_negative_base(device):
  a = np.array([-2.0, -1.5, 0.0, 2.0], np.float32)
  for e in (2, 3, 2.0, 0.5, -1, 5, 0):
    np.testing.assert_allclose(Tensor(a).pow(e).numpy(), torch.tensor(a).pow(e).numpy(), rtol=1e-5, equal_nan=True)
  ex = np.array([2.0, 3.0, 0.0, 0.5], np.float32)
  np.testing.assert_allclose(Tensor(a).pow(Tensor(ex)).numpy(), torch.tensor(a).pow(torch.tensor(ex)).numpy(), rtol=1e-5, equal_nan=True)

def test_no_grad_and_inference(device):
  w = Tensor(rnd(2), requires_grad=True)
  with Tensor.inference(): assert not (w * 2).requires_grad
  with no_grad():
    with no_grad(): pass
    assert Tensor.no_grad
  assert not Tensor.no_grad

def test_grad_accumulates_and_leaves_only(device):
  a = rnd(3); x = Tensor(a, requires_grad=True)
  y = x * 3
  (y * y).sum().backward()
  assert y.grad is None and x.grad is not None
  np.testing.assert_allclose(x.grad.numpy(), 18 * a, rtol=1e-5)
  assert not x.grad.requires_grad

def test_maximum_tie_gradient(device):
  a = np.array([1.0, 2.0, 3.0], np.float32); b = np.array([1.0, 1.0, 4.0], np.float32)
  x, y = Tensor(a, requires_grad=True), Tensor(b, requires_grad=True)
  tx, ty = torch.tensor(a, requires_grad=True), torch.tensor(b, requires_grad=True)
  x.maximum(y).sum().backward(); torch.maximum(tx, ty).sum().backward()
  close(x.grad, tx.grad); close(y.grad, ty.grad)

def test_assign_keeps_object(device):
  w = Tensor(rnd(2, 2)); w2 = w
  w.assign(w * 0 + 5)
  assert w2 is w and np.all(w.numpy() == 5)
  w.assign(Tensor(np.ones((2, 2), np.int32)))
  assert w.dtype == dtypes.float32 and np.all(w.numpy() == 1)

def test_sdpa_bool_mask_broadcast(device):
  q, k, v = rnd(2, 3, 4), rnd(2, 5, 4), rnd(2, 5, 4); m = R.random((3, 5)) > 0.3; m[:, 0] = True
  e = F.scaled_dot_product_attention(torch.tensor(q), torch.tensor(k), torch.tensor(v), attn_mask=torch.tensor(m))
  close(Tensor(q).scaled_dot_product_attention(Tensor(k), Tensor(v), attn_mask=Tensor(m)), e, 1e-4)

def test_gather_negative_and_backward(device):
  a = rnd(3, 4); idx = np.array([[0, -1], [2, 1], [-4, 3]], np.int32)
  x = Tensor(a, requires_grad=True); t = torch.tensor(a, requires_grad=True)
  ti = torch.tensor(np.where(idx < 0, idx + 4, idx), dtype=torch.long)
  out = x.gather(1, Tensor(idx)); te = t.gather(1, ti); close(out, te)
  g = rnd(3, 2); out.backward(Tensor(g)); te.backward(torch.tensor(g)); close(x.grad, t.grad)

def test_bf16_frontend(device):
  a = rnd(4, 6)
  x = Tensor(a, dtype=dtypes.bfloat16); t = torch.tensor(a).to(torch.bfloat16)
  assert x.dtype == dtypes.bfloat16 and x.numpy().dtype == np.uint16
  np.testing.assert_array_equal(x.float().numpy(), t.float().numpy())
  assert isinstance(x[0, 0].item(), float) and x[0, 0].item() == t[0, 0].item()
  np.testing.assert_allclose(np.array(x.tolist()), t.float().numpy())
  np.testing.assert_allclose((x @ x.T).float().numpy(), (t.float() @ t.float().T).to(torch.bfloat16).float().numpy(), rtol=1e-2, atol=1e-2)

def test_composite_big_chain_backward(device):
  # a GPT-ish block: embedding by indexing, attention, layernorm, cross-entropy
  V, T, D = 11, 6, 8
  emb, wq = rnd(V, D), rnd(D, D) * 0.3
  tok = R.integers(0, V, size=(2, T)).astype(np.int32); tgt = R.integers(0, V, size=(2, T)).astype(np.int32)
  E, Wq = Tensor(emb, requires_grad=True), Tensor(wq, requires_grad=True)
  tE, tWq = torch.tensor(emb, requires_grad=True), torch.tensor(wq, requires_grad=True)
  def fz():
    h = E[Tensor(tok)]; q = h @ Wq
    a = q.scaled_dot_product_attention(h, h, is_causal=True).layernorm() + h
    return (a @ E.T).reshape(-1, V).cross_entropy(Tensor(tgt).reshape(-1))
  def th():
    h = tE[torch.tensor(tok, dtype=torch.long)]; q = h @ tWq
    a = F.layer_norm(F.scaled_dot_product_attention(q, h, h, is_causal=True), (D,)) + h
    return F.cross_entropy((a @ tE.T).reshape(-1, V), torch.tensor(tgt, dtype=torch.long).reshape(-1))
  l, tl = fz(), th(); close(l, tl, 1e-4)
  l.backward(); tl.backward(); close(E.grad, tE.grad, 1e-4); close(Wq.grad, tWq.grad, 1e-4)
