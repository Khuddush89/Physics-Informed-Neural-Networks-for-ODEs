#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ===============================================================
# PINN for a Fredholm Integro-Differential Equation (1D)
#   u'(x) = 1 + ∫_0^1 x t u(t) dt,   u(0) = 0,   x∈[0,1]
# Exact solution: u(x) = x + (4/21) x^2
#
# Colab/T4-friendly; uses float32 on GPU if available.
# Shows: PINN vs exact, training loss (log), pointwise residual, error curve.
# ===============================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# --------------------- device / dtype / seeds ---------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32 if DEVICE.type == "cuda" else torch.float64
torch.set_default_dtype(DTYPE)
torch.manual_seed(42); np.random.seed(42)
print(f"Device: {DEVICE.type} | dtype: {DTYPE} | CUDA: {torch.cuda.is_available()}")

# --------------------- problem setup ---------------------
a, b   = 0.0, 1.0
Mq     = 512            # Gauss–Legendre quadrature nodes for ∫_0^1
Nx     = 256            # Chebyshev–Lobatto collocation points
epochs = 10_000
lr0, lr_min = 2e-3, 2e-4
ic_weight   = 10.0      # penalty for u(0)=0
do_lbfgs    = True

# exact solution
def u_exact_np(x):
    return x + (4.0/21.0)*(x**2)

def u_exact_torch(x):
    return x + (4.0/21.0)*(x**2)

# --------------------- quadrature / collocation ---------------------
def gauss_legendre_01(M):
    # Legendre–Gauss on [-1,1], then map to [0,1]
    xs, ws = np.polynomial.legendre.leggauss(M)
    t = (xs + 1.0)/2.0
    w = 0.5*ws
    return t.astype(np.float64), w.astype(np.float64)

def cheb_lobatto(N, a, b):
    # Ascending formula (no flip → no negative stride)
    j  = np.arange(N, dtype=np.float64)
    xL = -np.cos(np.pi * j / (N-1))
    return ((xL + 1.0) * 0.5 * (b - a) + a).astype(np.float64)

# nodes for ∫_0^1 (global)
tq_np, wq_np = gauss_legendre_01(Mq)
tq = torch.tensor(tq_np, dtype=DTYPE, device=DEVICE).view(-1,1)  # (Mq,1)
wq = torch.tensor(wq_np, dtype=DTYPE, device=DEVICE).view(-1,1)  # (Mq,1)

# Chebyshev–Lobatto collocation in [0,1]
x_col_np = cheb_lobatto(Nx, a, b)  # ascending, contiguous
x_col = torch.tensor(x_col_np.copy(), dtype=DTYPE, device=DEVICE).view(-1,1)

# --------------------- model ---------------------
class MLP(nn.Module):
    def __init__(self, in_dim=1, width=64, depth=3, act=nn.Tanh):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), act()]
        self.net = nn.Sequential(*layers)
        self.out = nn.Linear(width, 1)
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # simple residual skip for stability
        h1 = self.net[0](x)
        h1 = self.net[1](h1)
        h  = h1
        for k in range(2, len(self.net), 2):
            h = self.net[k](h)
            h = torch.tanh(h) + h  # tiny skip
        return self.out(h)

model = MLP(in_dim=1, width=64, depth=3, act=nn.Tanh).to(DEVICE)

# --------------------- helpers ---------------------
def compute_C(model):
    """
    C = ∫_0^1 t u(t) dt ≈ Σ wq * t * u(t)
    (keeps graph intact for training)
    """
    u_t = model(tq)            # (Mq,1)
    return torch.sum(wq * tq * u_t)  # scalar tensor

def rel_l2_vs_exact(model, n_dense=2001):
    x = torch.linspace(a, b, n_dense, device=DEVICE, dtype=DTYPE).view(-1,1)
    with torch.no_grad():
        uN = model(x)
        uE = u_exact_torch(x)
        w  = torch.ones_like(x)
        w[0] *= 0.5; w[-1] *= 0.5
        w *= (b - a)/(n_dense - 1)
        num = torch.sqrt(torch.sum(w * (uN - uE)**2))
        den = torch.sqrt(torch.sum(w * (uE)**2) + 1e-30)
        return (num/den).item()

# --------------------- optimizer / scheduler ---------------------
opt   = optim.Adam(model.parameters(), lr=lr0)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr_min)

# --------------------- training ---------------------
loss_hist, rel_hist, lr_hist = [], [], []

model.train()
for ep in range(1, epochs+1):
    opt.zero_grad(set_to_none=True)

    # collocation x as a fresh leaf with grad
    x_eval = x_col.detach().clone().requires_grad_(True)

    # C depends on model parameters → keep graph
    C = compute_C(model)  # scalar tensor

    # residual R(x) = u'(x) - 1 - x*C
    u_x  = model(x_eval)
    du   = torch.autograd.grad(u_x, x_eval,
                               grad_outputs=torch.ones_like(u_x),
                               create_graph=True, retain_graph=True)[0]
    R    = du - 1.0 - x_eval * C
    loss_int = torch.mean(R**2)

    # IC penalty: u(0)=0 (use the first collocation point which is 0)
    u0 = model(x_eval[:1])
    loss_ic = (u0**2).mean()

    loss = loss_int + ic_weight*loss_ic
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step(); sched.step()

    # logging
    loss_hist.append(float(loss.detach().cpu().item()))
    lr_hist.append(sched.get_last_lr()[0] if hasattr(sched, "get_last_lr") else opt.param_groups[0]['lr'])
    if ep % 100 == 0 or ep == 1:
        rr = rel_l2_vs_exact(model, n_dense=2001)
        rel_hist.append(rr)
        print(f"[Adam {ep:6d}] loss={loss_hist[-1]:.3e} | RelErr={rr:.3e}")

# --------------------- L-BFGS polish (optional) ---------------------
if do_lbfgs:
    model.train()
    lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=200, history_size=100, line_search_fn="strong_wolfe",
        tolerance_grad=1e-12, tolerance_change=1e-12
    )

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        x_eval = x_col.detach().clone().requires_grad_(True)
        C = compute_C(model)
        u_x = model(x_eval)
        du  = torch.autograd.grad(u_x, x_eval,
                                  grad_outputs=torch.ones_like(u_x),
                                  create_graph=True, retain_graph=True)[0]
        R = du - 1.0 - x_eval * C
        loss_int = torch.mean(R**2)
        u0 = model(x_eval[:1])
        loss_ic = (u0**2).mean()
        L = loss_int + ic_weight*loss_ic
        L.backward()
        return L

    print("[L-BFGS] starting...")
    final_loss = lbfgs.step(closure)
    print(f"[L-BFGS] final loss = {final_loss.item():.3e}")

# --------------------- evaluation ---------------------
model.eval()
rel_final = rel_l2_vs_exact(model, n_dense=4097)
print(f"Final Relative L2 Error (PINN vs exact): {rel_final:.3e}")

# checkpoints
for x0 in [0.00, 0.25, 0.50, 0.75, 1.00]:
    xv = torch.tensor([[x0]], dtype=DTYPE, device=DEVICE)
    with torch.no_grad():
        up = model(xv).item()
    ue = u_exact_np(np.array([x0]))[0]
    print(f"x={x0:4.2f}  u_PINN={up: .6f}  u_exact={ue: .6f}  |err|={abs(up-ue):.2e}")

# --------------------- plots ---------------------
# 1) PINN vs exact
xd = torch.linspace(a, b, 1001, device=DEVICE, dtype=DTYPE).view(-1,1)
with torch.no_grad():
    uN = model(xd).squeeze(1).cpu().numpy()
uE = u_exact_np(xd.squeeze(1).cpu().numpy())

plt.figure(figsize=(7.2,4.2))
plt.plot(xd.squeeze(1).cpu().numpy(), uE, 'k--', lw=2, label='Exact')
plt.plot(xd.squeeze(1).cpu().numpy(), uN,  label='PINN')
plt.xlabel('$x$'); plt.ylabel('$u(x)$')
plt.title('Fredholm IDE: PINN vs exact')
plt.legend(); plt.tight_layout()
plt.savefig('fredholm_PINN_vs_exact.png', dpi=150); plt.show()

# 2) training loss (log) + moving average
loss_np = np.array(loss_hist, dtype=float)
def moving_avg(y, w=101):
    w = min(w, len(y) if len(y)%2==1 else len(y)-1)  # force odd and ≤ len
    if w < 3: return y
    k = (w-1)//2
    padL, padR = y[0], y[-1]
    ypad = np.concatenate([np.full(k, padL), y, np.full(k, padR)])
    ker  = np.ones(w)/w
    return np.convolve(ypad, ker, mode='valid')

plt.figure(figsize=(7.2,4.2))
plt.semilogy(loss_np, alpha=0.35, label='loss (raw)')
plt.semilogy(moving_avg(loss_np, w=101), lw=2, label='loss (moving avg)')
plt.xlabel('epoch'); plt.ylabel('loss (log)')
plt.title('Training loss')
plt.legend(); plt.tight_layout()
plt.savefig('fredholm_training_loss.png', dpi=150); plt.show()

# 3) pointwise residual on collocation points (grad-safe)
with torch.no_grad():
    C_eval = torch.sum(wq * tq * model(tq))  # no need for grads here

x_eval = x_col.detach().clone().requires_grad_(True)
u_x    = model(x_eval)
du_dx  = torch.autograd.grad(u_x, x_eval,
                             grad_outputs=torch.ones_like(u_x),
                             create_graph=False, retain_graph=False)[0]
R_plot = (du_dx - 1.0 - x_eval * C_eval).squeeze(1).detach().cpu().numpy()
x_plot = x_eval.squeeze(1).detach().cpu().numpy()

plt.figure(figsize=(7.2,3.8))
plt.plot(x_plot, np.abs(R_plot), '.', ms=3)
plt.yscale('log')
plt.xlabel('$x$'); plt.ylabel(r'$|\mathcal{R}(x)|$')
plt.title('Pointwise residual (log scale)')
plt.tight_layout()
plt.savefig('fredholm_residual_points.png', dpi=150); plt.show()

# 4) absolute error curve
err_curve = np.abs(uN - uE)
plt.figure(figsize=(7.2,3.8))
plt.semilogy(xd.squeeze(1).cpu().numpy(), err_curve)
plt.xlabel('$x$'); plt.ylabel(r'$|u_{\mathrm{PINN}}-u_{\mathrm{exact}}|$')
plt.title('Absolute error')
plt.tight_layout()
plt.savefig('fredholm_abs_error.png', dpi=150); plt.show()
