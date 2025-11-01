# === Colab T4: Robust & Fast PINN via Bi-factorization (Best Option) ===
# Problem: u'(x) = u(x) + ∫_0^x t u(t) dt,  u(0)=1
# Transform: u = e^{x/2} v,  v'' - (x+1/4) v = 0
# Factor out Bi: v = Bi(x+1/4) * y  =>  y'' + 2 (Bi'/Bi) y' = 0
# Enforce y(0), y'(0) exactly via trial; train on residual of y; reconstruct u.

import numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from scipy.special import airy

# -------------------- device & dtype --------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)     # fp32 is fast on T4
SEED = 777
torch.manual_seed(SEED); np.random.seed(SEED)

# -------------------- domain & precompute Bi, Bi' on a grid --------------------
X_MIN, X_MAX = 0.0, 5.0
GRID_N = 20001                           # dense but fast (one-time)
xg = np.linspace(X_MIN, X_MAX, GRID_N, dtype=np.float64)
Ai, Aip, Bi, Bip = airy(xg + 0.25)
Bi = Bi.astype(np.float64); Bip = Bip.astype(np.float64)
ratio = (Bip / Bi).astype(np.float64)    # (Bi'/Bi)(x+1/4)

# push precomputed tables to GPU (no grad needed)
xg_t     = torch.tensor(xg,    device=DEVICE)
Bi_t     = torch.tensor(Bi,    device=DEVICE)
ratio_t  = torch.tensor(ratio, device=DEVICE)

dx = float(xg[1]-xg[0])

def interp_on_grid(x_t: torch.Tensor, table_t: torch.Tensor) -> torch.Tensor:
    """Fast linear interpolation of a 1D table sampled on xg_t."""
    z = torch.clamp(x_t.squeeze(-1), X_MIN, X_MAX)
    t = (z - X_MIN) / dx
    i0 = torch.clamp(t.floor().long(), 0, GRID_N-2)
    frac = (t - i0.to(t.dtype)).unsqueeze(-1)
    v0 = table_t[i0].unsqueeze(-1)
    v1 = table_t[i0+1].unsqueeze(-1)
    return v0 + frac*(v1 - v0)

# -------------------- exact solution utilities --------------------
def airy_constants():
    z = np.array([0.25], dtype=np.float64)
    Ai, Aip, Bi0, Bip0 = airy(z)
    c1 = np.pi * (Bip0 - Bi0/2)
    c2 = np.pi * (Ai/2 - Aip)
    return float(c1[0]), float(c2[0])

C1, C2 = airy_constants()

def exact_u_numpy(x):
    x = np.asarray(x, dtype=np.float64)        # robust to lists/arrays
    z = x + 0.25
    Ai, Aip, Bi_val, Bip_val = airy(z)
    return np.exp(x/2.0) * (C1*Ai + C2*Bi_val)

def exact_u_torch(x):
    x_np = x.detach().cpu().numpy().ravel()
    u = exact_u_numpy(x_np)
    return torch.tensor(u, dtype=torch.get_default_dtype(), device=x.device).view(-1,1)

# -------------------- ICs for y at x=0 --------------------
Bi0   = float(Bi[0])
Bip0  = float(Bip[0])
y0    = 1.0 / Bi0
y0p   = (0.5 - Bip0*y0) / Bi0

# -------------------- model (small & fast) --------------------
class MLP(nn.Module):
    def __init__(self, width=64, depth=4, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(1, width), act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):  # x in [0,L], shape (N,1)
        s = x / X_MAX      # normalize input ~ [0,1]
        return self.net(s)

N = MLP().to(DEVICE)

def y_theta(x):
    # enforce y(0)=y0, y'(0)=y0p  via trial function
    return y0 + y0p*x + (x**2) * N(x)

def u_pred_from_y(x):
    Bi_here = interp_on_grid(x, Bi_t)            # Bi(x+1/4)
    y       = y_theta(x)
    v       = Bi_here * y
    return torch.exp(0.5*x) * v

# Residual: y'' + 2 (Bi'/Bi) y' = 0, with (Bi'/Bi) from precomputed table
def residual_y(x):
    x.requires_grad_(True)
    y  = y_theta(x)
    dy = torch.autograd.grad(y,  x, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
    d2y= torch.autograd.grad(dy, x, grad_outputs=torch.ones_like(dy), create_graph=True)[0]
    r  = interp_on_grid(x, ratio_t)              # (Bi'/Bi)(x+1/4)
    return d2y + 2.0*r*dy

# -------------------- training settings --------------------
stages = [
    dict(L=3.0, n_col=2048, adam_steps=2500, lr=2e-3,  lbfgs_steps=80),
    dict(L=4.0, n_col=2048, adam_steps=2500, lr=1.5e-3,lbfgs_steps=100),
    dict(L=5.0, n_col=3072, adam_steps=4000, lr=1.0e-3,lbfgs_steps=140),
]
CLIP = 3.0

def sample_x(L, n):
    # strong right-end coverage (where growth is hardest)
    n_main = int(0.7*n); n_edge = n - n_main
    z = torch.rand((n_main,1), device=DEVICE)
    z = 0.25*z + 0.75*(z**3)               # bias right
    x_main = L*z
    x_edge = 0.85*L + 0.15*L*torch.rand((n_edge,1), device=DEVICE)
    return torch.cat([x_main, x_edge], dim=0).requires_grad_(True)

def stage_loss(L, n_col):
    x = sample_x(L, n_col)
    R = residual_y(x)
    w = 1.0 + 3.0*(x/L)**3                 # emphasize right end
    return (w*(R**2)).mean()

loss_hist = []

# -------------------- train (curriculum) --------------------
for si, st in enumerate(stages, 1):
    L, n_col, steps, lr, lbfgs_steps = st["L"], st["n_col"], st["adam_steps"], st["lr"], st["lbfgs_steps"]
    print(f"\n== Stage {si}: [0,{L}] ==")

    # Adam
    opt = optim.Adam(N.parameters(), lr=lr, betas=(0.9,0.999))
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-5)
    for k in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = stage_loss(L, n_col)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(N.parameters(), CLIP)
        opt.step(); sch.step()
        if (k+1) % 800 == 0:
            print(f"[Adam s{si}] {k+1}/{steps}  loss={loss.item():.3e}")
        loss_hist.append(float(loss.detach().item()))

    # L-BFGS polish on fixed grid
    x_fix = torch.linspace(0.0, L, 2500, device=DEVICE).view(-1,1).requires_grad_(True)
    lbfgs = optim.LBFGS(N.parameters(), max_iter=lbfgs_steps,
                        tolerance_grad=1e-9, tolerance_change=1e-9,
                        line_search_fn='strong_wolfe')
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        y  = y_theta(x_fix)
        dy = torch.autograd.grad(y,  x_fix, grad_outputs=torch.ones_like(y),  create_graph=True)[0]
        d2y= torch.autograd.grad(dy, x_fix, grad_outputs=torch.ones_like(dy), create_graph=True)[0]
        r  = interp_on_grid(x_fix, ratio_t)
        R  = d2y + 2.0*r*dy
        w  = 1.0 + 3.0*(x_fix/L)**3
        Ltot = (w*(R**2)).mean()
        Ltot.backward()
        return Ltot
    final_val = lbfgs.step(closure)
    if isinstance(final_val, torch.Tensor): final_val = final_val.detach().item()
    print(f"[LBFGS s{si}] final loss ~ {final_val:.4e}")
    loss_hist.append(final_val)

    # per-stage error
    with torch.no_grad():
        xs = torch.linspace(0.0, L, 1501, device=DEVICE).view(-1,1)
        h  = (L - 0.0)/(xs.numel()-1)
        wts= torch.full_like(xs, h); wts[0]=h/2; wts[-1]=h/2
        up = u_pred_from_y(xs)
        ue = exact_u_torch(xs)
        relL2 = torch.sqrt(torch.sum(wts*(up-ue)**2)/torch.sum(wts*(ue**2))).item()
        print(f"[Stage {si}] rel L2 on [0,{L}]: {relL2:.4e}")

# -------------------- final evaluation on [0,5] --------------------
with torch.no_grad():
    x_eval = torch.linspace(0.0, X_MAX, 2001, device=DEVICE).view(-1,1)
    h   = (X_MAX - 0.0)/(x_eval.numel()-1)
    wts = torch.full_like(x_eval, h); wts[0]=h/2; wts[-1]=h/2
    u_pred = u_pred_from_y(x_eval)
    u_star = exact_u_torch(x_eval)
    relL2 = torch.sqrt(torch.sum(wts*(u_pred-u_star)**2)/torch.sum(wts*(u_star**2))).item()
    print(f"\nRelative L2 error on [0,5]: {relL2:.6e}")

    # curves for plot (vectorized exact call)
    x_plot = torch.linspace(0.0, X_MAX, 1500, device=DEVICE).view(-1,1)
    u_plot = u_pred_from_y(x_plot).cpu().numpy().ravel()
    x_plot_np = x_plot.cpu().numpy().ravel()
    u_star_plot = exact_u_numpy(x_plot_np)

# -------------------- plots --------------------
plt.figure()
plt.plot(x_plot_np, u_star_plot, label="Exact")
plt.plot(x_plot_np, u_plot, '--', label="PINN (Bi-factorization)")
plt.xlabel("x"); plt.ylabel("u(x)"); plt.title("Volterra IDE: Exact vs PINN")
plt.legend(); plt.tight_layout(); plt.show()

plt.figure()
plt.plot(np.arange(len(loss_hist)), loss_hist)
plt.yscale('log'); plt.xlabel("Iteration"); plt.ylabel("Loss")
plt.title("Training Loss (all stages)"); plt.tight_layout(); plt.show()
