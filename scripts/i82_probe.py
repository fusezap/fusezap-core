"""I82 probe: VAE-style layers composed from the 28 primitives, against torch, on one device."""
import sys, time, numpy as np, torch, torch.nn.functional as TF
from fusezap.tensor import Tensor
dev = sys.argv[1] if len(sys.argv) > 1 else "CPU"

def conv2d(x, w, b=None, stride=1, padding=1):
  # x (N,C,H,W), w (O,C,kh,kw): im2col by kh*kw shifted views (PAD + SHRINK), then one matmul
  N, C, H, W = x.shape; O, _, kh, kw = w.shape
  xp = x.pad(((0,0),(0,0),(padding,padding),(padding,padding)))
  Ho, Wo = (H + 2*padding - kh)//stride + 1, (W + 2*padding - kw)//stride + 1
  cols = []
  for i in range(kh):
    for j in range(kw):
      v = xp.shrink(((0,N),(0,C),(i, i + stride*(Ho-1) + 1),(j, j + stride*(Wo-1) + 1)))
      if stride > 1: v = v[:, :, ::stride, ::stride]
      cols.append(v)
  col = Tensor.stack(cols, axis=2)                      # (N, C, kh*kw, Ho, Wo)
  col = col.reshape(N, C*kh*kw, Ho*Wo).permute(0, 2, 1) # (N, HoWo, C*k*k)
  out = col @ w.reshape(O, C*kh*kw).T                  # (N, HoWo, O)
  out = out.permute(0, 2, 1).reshape(N, O, Ho, Wo)
  return out + b.reshape(1, O, 1, 1) if b is not None else out

def group_norm(x, G, g, b, eps=1e-6):
  N, C, H, W = x.shape
  xg = x.reshape(N, G, C//G * H * W)
  m = xg.mean(-1, keepdim=True); d = xg - m
  y = (d * ((d*d).mean(-1, keepdim=True) + eps).rsqrt()).reshape(N, C, H, W)
  return y * g.reshape(1, C, 1, 1) + b.reshape(1, C, 1, 1)

def up2(x):  # nearest 2x
  N, C, H, W = x.shape
  return x.reshape(N, C, H, 1, W, 1).expand(N, C, H, 2, W, 2).reshape(N, C, 2*H, 2*W)

rng = np.random.default_rng(0)
def run(name, f_fz, f_t, *arrs):
  ts = [Tensor(a, device=dev) for a in arrs]
  t0 = time.perf_counter(); y = f_fz(*ts).numpy(); t1 = time.perf_counter()
  ref = f_t(*[torch.from_numpy(a) for a in arrs]).numpy()
  err = np.abs(y - ref).max() / (np.abs(ref).max() + 1e-30)
  print(f"{dev} {name:28s} shape={tuple(y.shape)} max_rel_err={err:.2e} time={t1-t0:.2f}s")

N, C, H, W, O = 1, 32, 32, 32, 64
x = rng.standard_normal((N, C, H, W), dtype=np.float32)
w = rng.standard_normal((O, C, 3, 3), dtype=np.float32) * 0.05
b = rng.standard_normal((O,), dtype=np.float32)
run("conv3x3 s1 p1", lambda x,w,b: conv2d(x,w,b), lambda x,w,b: TF.conv2d(x,w,b,padding=1), x, w, b)
run("conv3x3 s2 (downsample)", lambda x,w,b: conv2d(x,w,b,stride=2,padding=1), lambda x,w,b: TF.conv2d(x,w,b,stride=2,padding=1), x, w, b)
w1 = rng.standard_normal((O, C, 1, 1), dtype=np.float32) * 0.1
run("conv1x1", lambda x,w,b: conv2d(x,w,b,padding=0), lambda x,w,b: TF.conv2d(x,w,b), x, w1, b)
g = rng.standard_normal((C,), dtype=np.float32); bb = rng.standard_normal((C,), dtype=np.float32)
run("group_norm G=8", lambda x,g,b: group_norm(x,8,g,b), lambda x,g,b: TF.group_norm(x,8,g,b,eps=1e-6), x, g, bb)
run("upsample nearest 2x", up2, lambda x: TF.interpolate(x, scale_factor=2, mode="nearest"), x)
run("GN+SiLU+conv (resnet half)", lambda x,g,b,w,c: conv2d(group_norm(x,8,g,b).silu(), w, c),
    lambda x,g,b,w,c: TF.conv2d(TF.silu(TF.group_norm(x,8,g,b,eps=1e-6)), w, c, padding=1), x, g, bb, w, b)
