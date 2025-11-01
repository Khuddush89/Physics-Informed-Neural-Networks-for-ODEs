# ===============================================================
# A-PINN for convolution-type IDE:
#   u'(x) = u(x) + ∫_0^x e^{-(x-t)} u(t) dt,   u(0)=1
# Internal variable: z = ∫_0^x e^{-(x-t)} u(t) dt
# System:
#   u' = u + z
#   z' = -z + u
# Exact:
#   u(x) = cosh(√2 x) + (√2/2) sinh(√2 x)
#   z(x) = (√2/2) sinh(√2 x)
#
# Saves (for LaTeX):
#   conv_u_vs_exact.pdf
#   conv_z_vs_exact.pdf
#   conv_training_loss.pdf
#   conv_training_relerr.pdf
# ===============================================================
import math, numpy as np
import torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt

# ------------------------ device / dtype / seeds ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float32
torch.set_default_dtype(DTYPE)
torch.manual_seed(1234); np.random.seed(1234)
print(f"Device: {DEVICE} | dtype: {DTYPE} | CUDA: {torch.cuda.is_available()}")

# ------------------------ problem setup ------------------------
L = 1.0
Nx = 512                      # collocation points (Chebyshev–Lobatto)
N_eval = 2000                 # dense grid for validation plots/errors

# Chebyshev–Lobatto points on [0,L]
def cheb_lobatto(n, a=0.0, b=1.0):
    k = np.arange(n)
    x = np.cos(np.pi * k / (n-1))
    x = x[::-1].copy()         # increasing order
    return (a+b)/2 + (b-a)*x/2

x_col_np = cheb_lobatto(Nx, 0.0, L)
x_col = torch.tensor(x_col_np, dtype=DTYPE, device=DEVICE).view(-1,1)

# Dense grid for plotting/validation
xx = torch.linspace(0.0, L, N_eval, device=DEVICE, dtype=DTYPE).view(-1,1)
x_eval_np = xx.squeeze(1).detach().cpu().numpy()

# ------------------------ exact solution ------------------------
sqrt2 = math.sqrt(2.0)
def u_exact_np(x):
    return np.cosh(sqrt2*x) + (sqrt2/2.0)*np.sinh(sqrt2*x)
def z_exact_np(x):
    return (sqrt2/2.0)*np.sinh(sqrt2*x)

u_ex_np = u_exact_np(x_eval_np)
z_ex_np = z_exact_np(x_eval_np)

# ------------------------ model ------------------------
class MLP2(nn.Module):
    """Two-output MLP: [u, z] with tanh activations."""
    def __init__(self, width=64, depth=3):
        super().__init__()
        layers = []
        in_dim = 1
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.Tanh())
        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        self.backbone = nn.Sequential(*layers)
        self.head_u = nn.Linear(width, 1)
        self.head_z = nn.Linear(width, 1)
        # Xavier init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.backbone(x)
        u = self.head_u(h)
        z = self.head_z(h)
        return u, z

model = MLP2(width=64, depth=3).to(DEVICE)

# ------------------------ optimizer / schedule ------------------------
adam = optim.Adam(model.parameters(), lr=2e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=10000, eta_min=2e-4)

# ------------------------ training params ------------------------
epochs = 10000
print_every = 100
clip_grad = 1.0

w_u = 1.0
w_z = 1.0
w_ic = 100.0

# ------------------------ helpers ------------------------
def rel_L2(u_num, u_ref, x_np):
    """Composite-trapezoid relative L2 on grid x_np."""
    w = np.ones_like(x_np)
    w[0] *= 0.5; w[-1] *= 0.5
    w *= (x_np[-1]-x_np[0])/(len(x_np)-1)
    num = np.sqrt(np.sum(w * (u_num - u_ref)**2))
    den = np.sqrt(np.sum(w * (u_ref)**2))
    return float(num/den)

# ------------------------ logs ------------------------
loss_hist = []
relerr_hist = []
lr_hist = []

# ------------------------ training loop ------------------------
for ep in range(1, epochs+1):
    model.train()
    adam.zero_grad(set_to_none=True)

    # Make x require grad for AD each iteration
    x = x_col.clone().detach().requires_grad_(True)

    u, z = model(x)
    ones = torch.ones_like(u)

    du_dx = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    dz_dx = torch.autograd.grad(z, x, grad_outputs=ones, create_graph=True)[0]

    # Residuals for system:
    #   u' = u + z
    #   z' = -z + u
    R_u = du_dx - (u + z)
    R_z = dz_dx - (-z + u)

    # ICs (soft)
    x0 = torch.zeros_like(x[:1])
    u0, z0 = model(x0)
    loss = w_u * torch.mean(R_u**2) \
         + w_z * torch.mean(R_z**2) \
         + w_ic * ((u0 - 1.0)**2 + (z0 - 0.0)**2)

    loss.backward()
    if clip_grad is not None:
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    adam.step()
    scheduler.step()

    # logs
    loss_hist.append(float(loss.detach().cpu().item()))
    lr_hist.append(adam.param_groups[0]['lr'])

    # validation rel L2 error vs exact u(x)
    if ep % print_every == 0 or ep == 1:
        model.eval()
        with torch.no_grad():
            u_eval, _ = model(xx)
            u_eval_np = u_eval.squeeze(1).detach().cpu().numpy()
        rel = rel_L2(u_eval_np, u_ex_np, x_eval_np)
        relerr_hist.append((ep, rel))
        print(f"[Adam {ep:6d}] loss={loss_hist[-1]:.3e} | RelErr={rel:.3e}")

# ------------------------ optional L-BFGS polish ------------------------
def lbfgs_closure():
    model.zero_grad(set_to_none=True)
    x = x_col.clone().detach().requires_grad_(True)
    u, z = model(x)
    ones = torch.ones_like(u)
    du_dx = torch.autograd.grad(u, x, grad_outputs=ones, create_graph=True)[0]
    dz_dx = torch.autograd.grad(z, x, grad_outputs=ones, create_graph=True)[0]
    R_u = du_dx - (u + z)
    R_z = dz_dx - (-z + u)
    x0 = torch.zeros_like(x[:1])
    u0, z0 = model(x0)
    L = w_u * torch.mean(R_u**2) \
      + w_z * torch.mean(R_z**2) \
      + w_ic * ((u0 - 1.0)**2 + (z0 - 0.0)**2)
    L.backward()
    return L

lbfgs = optim.LBFGS(model.parameters(), max_iter=200, history_size=100, line_search_fn="strong_wolfe")
print("[L-BFGS] starting...")
final_loss = lbfgs.step(lbfgs_closure)
print(f"[L-BFGS] final loss = {final_loss.item():.3e}")

# Final validation
model.eval()
with torch.no_grad():
    u_eval, z_eval = model(xx)
u_eval_np = u_eval.squeeze(1).detach().cpu().numpy()
z_eval_np = z_eval.squeeze(1).detach().cpu().numpy()
final_rel = rel_L2(u_eval_np, u_ex_np, x_eval_np)
print(f"Final Relative L2 Error (PINN vs exact): {final_rel:.3e}")

# Print checkpoints
for x0 in [0.0, 0.25, 0.50, 0.75, 1.0]:
    xv = torch.tensor([[x0]], device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        up = model(xv)[0].item()
    ue = u_exact_np(np.array([x0]))[0]
    print(f"x={x0:4.2f}  u_PINN={up: .6f}  u_exact={ue: .6f}  |err|={abs(up-ue):.2e}")

# ------------------------ plots ------------------------
# 1) u vs exact
plt.figure(figsize=(7.2,4.6))
plt.plot(x_eval_np, u_ex_np, 'k--', lw=2, label='exact $u$')
plt.plot(x_eval_np, u_eval_np, label='$u_\\theta$')
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.title("Convolution IDE: $u_\\theta$ vs exact")
plt.legend()
plt.tight_layout()
plt.savefig("conv_u_vs_exact.pdf")
plt.show()

# 2) z vs exact  (FIXED QUOTES HERE)
plt.figure(figsize=(7.2,4.6))
plt.plot(x_eval_np, z_ex_np, 'k--', lw=2, label='exact $z$')
plt.plot(x_eval_np, z_eval_np, label='$z_\\theta$')
plt.xlabel("$x$")
plt.ylabel("$z(x)$")
plt.title("Convolution IDE: $z_\\theta$ vs exact")
plt.legend()
plt.tight_layout()
plt.savefig("conv_z_vs_exact.pdf")
plt.show()

# 3) training loss (semilogy)
plt.figure(figsize=(7.2,4.6))
plt.semilogy(np.arange(1, len(loss_hist)+1), loss_hist)
plt.xlabel("Iteration")
plt.ylabel("Training loss")
plt.title("Training loss (semilog)")
plt.tight_layout()
plt.savefig("conv_training_loss.pdf")
plt.show()

# 4) validation relative L2 error vs iteration
if len(relerr_hist) == 0:
    relerr_hist = [(epochs, final_rel)]
iters, rels = zip(*relerr_hist)
plt.figure(figsize=(7.2,4.6))
plt.semilogy(list(iters), list(rels), marker='o', ms=3, lw=1)
plt.xlabel("Iteration")
plt.ylabel("Relative $L^2$ error (vs exact)")
plt.title("Validation relative $L^2$ error")
plt.tight_layout()
plt.savefig("conv_training_relerr.pdf")
plt.show()
