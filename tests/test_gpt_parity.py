import importlib.util, pathlib, numpy as np, torch
from fusezap import Tensor, manual_seed
from fusezap.nn import get_parameters, get_state_dict
from fusezap.optim import AdamW

_spec = importlib.util.spec_from_file_location("gpt_example", pathlib.Path(__file__).parent.parent / "examples" / "gpt.py")
gpt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(gpt)

class TBlock(torch.nn.Module):
  def __init__(s, d, h):
    super().__init__()
    s.h, s.ln1, s.ln2 = h, torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)
    s.qkv, s.proj, s.fc, s.out = torch.nn.Linear(d, 3 * d), torch.nn.Linear(d, d), torch.nn.Linear(d, 4 * d), torch.nn.Linear(4 * d, d)
  def forward(s, x):
    B, T, C = x.shape
    q, k, v = [t.reshape(B, T, s.h, C // s.h).transpose(1, 2) for t in s.qkv(s.ln1(x)).chunk(3, -1)]
    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
    x = x + s.proj(a)
    return x + s.out(torch.nn.functional.gelu(s.fc(s.ln2(x)), approximate="tanh"))

class TGPT(torch.nn.Module):
  def __init__(s, vocab, d, h, L, T):
    super().__init__()
    s.tok, s.pos = torch.nn.Embedding(vocab, d), torch.nn.Embedding(T, d)
    s.blocks = torch.nn.ModuleList([TBlock(d, h) for _ in range(L)])
    s.ln_f, s.head = torch.nn.LayerNorm(d), torch.nn.Linear(d, vocab, bias=False)
  def forward(s, idx):
    x = s.tok(idx) + s.pos(torch.arange(idx.shape[1]))
    for b in s.blocks: x = b(x)
    return s.head(s.ln_f(x))

def test_gpt_parity(device):
  L, d, h, T, B, lr = 2, 64, 4, 32, 4, 1e-3
  manual_seed(0)
  data = gpt.encode(gpt.TEXT); V = len(gpt.CHARS)
  m = gpt.GPT(V, d, h, L, T)
  tm = TGPT(V, d, h, L, T)
  tsd = tm.state_dict()
  sd = get_state_dict(m)
  assert set(sd) == set(tsd), set(sd) ^ set(tsd)
  for k, v in sd.items(): tsd[k].copy_(torch.tensor(v.numpy()))
  opt, topt = AdamW(get_parameters(m), lr=lr), torch.optim.AdamW(tm.parameters(), lr=lr)
  rng = np.random.default_rng(0)
  for step in range(20):
    x, y = gpt.get_batch(data, B, T, rng)
    opt.zero_grad(); loss = m.loss(Tensor(x), Tensor(y)); loss.backward(); opt.step()
    topt.zero_grad(); tl = torch.nn.functional.cross_entropy(tm(torch.tensor(x).long()).reshape(-1, V), torch.tensor(y).long().reshape(-1))
    tl.backward(); topt.step()
    assert abs(loss.item() - tl.item()) < 1e-4, (step, loss.item(), tl.item())
