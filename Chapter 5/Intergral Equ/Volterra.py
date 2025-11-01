#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
# A-PINN for Volterra example:
#   u(x) = 1 + λ ∫_0^x t u(t) dt,  x ∈ [0,1]
#
# Minimal auxiliary residuals:
#   R1 = u - 1 - λ A  = 0
#   R2 = A' - x u     = 0
#
# Hard constraints:
#   u(x) = 1 + x * uhat(x)   => u(0)=1
#   A(x) = x * Ahat(x)       => A(0)=0
# ===============================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ----------------------- settings -----------------------
torch.set_default_dtype(torch.float64)
torch.set_printoptions(sci_mode=True, precision=6)

SEED       = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Problem / training hyperparameters
LAMBDA     = 2.0          # λ
EPOCHS     = 30000        # Adam epochs
N_COLLOC   = 512          # collocation points per epoch
MU         = 1.0          # weight for R2
INIT_LR    = 2e-3         # Adam LR (cosine annealed)
ETA_MIN    = 5e-5         # min LR for cosine schedule
WIDTH      = 64           # hidden width
DEPTH      = 4            # number of hidden layers
CLIP_NORM  = 1.0          # gradient clipping
PLOT_N     = 600          # grid points for plotting

# Optional second stage: L-BFGS (set False to avoid it)
USE_LBFGS  = True

# ----------------------- exact solution ------------------------
def u_exact(x, lam=LAMBDA):
    # x: (N,1) torch tensor
    return torch.exp(0.5 * lam * x * x)

# ----------------------- model --------------------------------
class APINN(nn.Module):
    """
    Two-head network with hard constraints:
        u(x) = 1 + x*uhat(x),   A(x) = x*Ahat(x)
    """
    def __init__(self, depth=DEPTH, width=WIDTH):
        super().__init__()
        layers = [nn.Linear(1, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        self.trunk = nn.Sequential(*layers)
        self.uhead = nn.Linear(width, 1)
        self.ahead = nn.Linear(width, 1)

        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h    = self.trunk(x)
        uhat = self.uhead(h)     # unconstrained outputs
        Ahat = self.ahead(h)
        u    = 1.0 + x * uhat    # enforce u(0)=1
        A    = x * Ahat          # enforce A(0)=0
        return u, A

model = APINN().to(DEVICE)

# ----------------------- sampling (curriculum) -----------------
def sample_collocation(n, epoch, total_epochs):
    """
    Early epochs: bias samples near 0 (causal build-up).
    Later: blend to uniform.
    """
    frac = min(1.0, epoch / (0.3 * total_epochs))  # blend by 30% of training
    # Beta(1,2) peaks near 0
    x_bias = torch.distributions.Beta(1.0, 2.0).sample((n, 1))
    x_uni  = torch.rand(n, 1)
    x = (1 - frac) * x_bias + frac * x_uni
    return x

# ----------------------- residuals -----------------------------
def residuals(model, x):
    x = x.clone().requires_grad_(True)
    u, A = model(x)
    A_x = torch.autograd.grad(
        A, x, grad_outputs=torch.ones_like(A),
        create_graph=True, retain_graph=True
    )[0]
    R1 = u - 1.0 - LAMBDA * A
    R2 = A_x - x * u
    return R1, R2, u, A, x

# ----------------------- optimizers ----------------------------
adam  = optim.Adam(model.parameters(), lr=INIT_LR)
sched = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=EPOCHS, eta_min=ETA_MIN)

# ----------------------- logs ---------------------------------
hist_total, hist_r1, hist_r2, hist_rel = [], [], [], []

# ----------------------- training: Adam ------------------------
model.train()
for epoch in range(1, EPOCHS + 1):
    adam.zero_grad(set_to_none=True)

    x = sample_collocation(N_COLLOC, epoch, EPOCHS).to(DEVICE)
    R1, R2, u, A, x = residuals(model, x)

    loss_r1 = torch.mean(R1**2)
    loss_r2 = torch.mean(R2**2)
    loss = loss_r1 + MU * loss_r2

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
    adam.step()
    sched.step()

    # metrics (on a fixed grid)
    with torch.no_grad():
        xx = torch.linspace(0, 1, PLOT_N, device=DEVICE).view(-1, 1)
        up, _ = model(xx)
        ue = u_exact(xx)
        rel = torch.norm(up - ue) / torch.norm(ue)
        hist_total.append(loss.item())
        hist_r1.append(loss_r1.item())
        hist_r2.append(loss_r2.item())
        hist_rel.append(rel.item())

    if epoch % 2000 == 0 or epoch == 1:
        print(f"[Adam {epoch:5d}] loss={loss.item():.3e} | "
              f"R1={loss_r1.item():.3e} | R2={loss_r2.item():.3e} | RelErr={rel.item():.3e}")

# ----------------------- L-BFGS polish (optional) --------------
if USE_LBFGS:
    lbfgs = optim.LBFGS(
        model.parameters(),
        max_iter=400,
        history_size=100,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        line_search_fn="strong_wolfe",
    )

    # NOTE: We compute gradients inside the closure and return a DETACHED loss;
    # this keeps LBFGS happy but avoids the "requires_grad to scalar" warning.
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        x = torch.rand(N_COLLOC, 1, device=DEVICE)
        R1, R2, _, _, _ = residuals(model, x)
        loss = torch.mean(R1**2) + MU * torch.mean(R2**2)
        loss.backward()
        return loss.detach()

    print("[L-BFGS] starting...")
    final_loss = lbfgs.step(closure)  # this is already a tensor (detached)
    print(f"[L-BFGS] final loss = {final_loss.item():.3e}")

# ----------------------- evaluation & plots --------------------
model.eval()
with torch.no_grad():
    x_plot = torch.linspace(0, 1, PLOT_N, device=DEVICE).view(-1, 1)
    u_pred, A_pred = model(x_plot)
    u_ex = u_exact(x_plot)
    rel_final = torch.norm(u_pred - u_ex) / torch.norm(u_ex)
    print(f"Final Relative L2 Error: {rel_final.item():.3e}")

# Plot 1: u(x) prediction vs exact
plt.figure(figsize=(7, 4.5))
plt.plot(x_plot.cpu().numpy(), u_ex.cpu().numpy(), label="Exact $u^{\\ast}(x)$")
plt.plot(x_plot.cpu().numpy(), u_pred.cpu().numpy(), linestyle="--", label="PINN $u_\\theta(x)$")
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.title("Volterra example: prediction vs exact")
plt.legend()
plt.tight_layout()
plt.savefig("volterra_u_vs_exact.png", dpi=150)
plt.show()

# Plot 2: training losses (Adam stage)
steps = np.arange(1, EPOCHS + 1)
plt.figure(figsize=(7, 4.5))
plt.semilogy(steps, hist_total, label="Total")
plt.semilogy(steps, hist_r1, label="$\\|\\mathcal{R}_1\\|^2$")
plt.semilogy(steps, hist_r2, label="$\\|\\mathcal{R}_2\\|^2$")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training losses (log scale) — Adam stage")
plt.legend()
plt.tight_layout()
plt.savefig("volterra_losses.png", dpi=150)
plt.show()

# Plot 3: relative L2 error over epochs (Adam stage)
plt.figure(figsize=(7, 4.5))
plt.semilogy(steps, hist_rel)
plt.xlabel("Epoch")
plt.ylabel("Relative $L^2$ error")
plt.title("Convergence of relative error — Adam stage")
plt.tight_layout()
plt.savefig("volterra_relerr.png", dpi=150)
plt.show()

# ----------------------- quick sanity prints -------------------
with torch.no_grad():
    for x0 in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xx = torch.tensor([[x0]], device=DEVICE)
        up, _ = model(xx)
        print(f"x={x0:4.2f}  PINN u={up.item():.6f}  Exact u={u_exact(xx).item():.6f}")
