# %% [markdown]
# PINN for Fredholm integral equation (Colab-ready, single cell)
# u(x) = 1 + (1/2) ∫_0^1 (x t) u(t) dt, x∈[0,1]
# Exact: u(x) = 1 + 0.3 x
# Modes: 'aux' (recommended) or 'standard'
# Features: Chebyshev collocation, 2-pt Gauss exact quadrature, AMP (new API),
# 100k Adam steps, L-BFGS polish (500 iters), early stopping (patience 10000),
# diagnostics, and training curves (loss, slope error, C-error).
# Improvements: No detach in aux loss, normalized constraint, s_theta init~0.5,
# rel L2 error, LR scheduler, adaptive activation option (SiLU), log-scale plot.
import math, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
from contextlib import contextmanager
# --------------------
# User settings
# --------------------
MODE = 'aux' # 'aux' or 'standard'
EPOCHS = 100000 # Max Adam epochs
LR = 1e-2 # Increased initial learning rate
N_COL = 256 # collocation points
M_QUAD = 2 # Gauss–Legendre nodes (2 is exact for this example)
HIDDEN = 64
LAYERS = 4
LAMBDA_S = 50.0 # Reduced constraint weight
USE_AMP = True # mixed precision on CUDA
SEED = 1234
LOG_EVERY = 200 # log & record metrics every N steps
LBFGS_ITERS = 500 # L-BFGS polish iterations
PATIENCE = 10000 # Increased patience
ACTIVATION = nn.Tanh # or nn.SiLU
# --------------------
# Setup
# --------------------
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | MODE: {MODE} | AMP: {USE_AMP}")
def chebyshev_points(n, device, dtype=torch.float32):
    k = torch.arange(1, n+1, device=device, dtype=dtype)
    x = torch.cos((2*k - 1) / (2*n) * math.pi)
    x = 0.5 * (x + 1.0)
    return torch.sort(x).values.unsqueeze(1)
def gauss_legendre(n, a=0.0, b=1.0, dtype=torch.float32, device='cpu'):
    xs, ws = np.polynomial.legendre.leggauss(n)
    xm, xr = 0.5*(b+a), 0.5*(b-a)
    xs, ws = xm + xr*xs, xr*ws
    return torch.tensor(xs, dtype=dtype, device=device).unsqueeze(1), torch.tensor(ws, dtype=dtype, device=device)
def normalize(x):
    return 2.0 * x - 1.0
def exact_u(x_np):
    return 1.0 + 0.3 * x_np
class ResPINN(nn.Module):
    def __init__(self, hidden_size=64, num_layers=4, activation=nn.Tanh):
        super().__init__()
        self.fc_in = nn.Linear(1, hidden_size)
        self.blocks = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)])
        self.fc_out = nn.Linear(hidden_size, 1)
        self.act = activation()
    def forward(self, x):
        y = self.act(self.fc_in(x))
        for layer in self.blocks:
            y_res = y
            y = self.act(layer(y)) + y_res
        return self.fc_out(y)
class AuxResPINN(nn.Module):
    def __init__(self, hidden_size=64, num_layers=4, activation=nn.Tanh):
        super().__init__()
        self.pinn = ResPINN(hidden_size, num_layers, activation)
        self.s_theta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
    def forward(self, x):
        return self.pinn(x)
@contextmanager
def autocast_if(enabled, device_type='cpu'):
    if enabled and device_type == 'cuda':
        with torch.amp.autocast("cuda", dtype=torch.float16):
            yield
    else:
        yield
# --------------------
# Data
# --------------------
x_col = chebyshev_points(N_COL, device)
t_quad, w_quad = gauss_legendre(M_QUAD, 0.0, 1.0, device=device)
# --------------------
# Losses
# --------------------
def loss_standard(model, x_col, t_quad, w_quad, use_amp=False, device_type='cpu'):
    x_col_norm, t_quad_norm = normalize(x_col), normalize(t_quad)
    with autocast_if(use_amp, device_type):
        u_x = model(x_col_norm)
        u_t = model(t_quad_norm)
        C = torch.sum(w_quad * (t_quad.squeeze(1) * u_t.squeeze(1)))
        residual = u_x - 1.0 - 0.5 * x_col * C
        loss = torch.mean(residual**2)
    return loss
def loss_aux(model, x_col, t_quad, w_quad, lambda_s=LAMBDA_S, use_amp=False, device_type='cpu'):
    x_col_norm, t_quad_norm = normalize(x_col), normalize(t_quad)
    with autocast_if(use_amp, device_type):
        u_x = model(x_col_norm)
        u_t = model(t_quad_norm)
        C_quad = torch.sum(w_quad * (t_quad.squeeze(1) * u_t.squeeze(1)))
        r_point = u_x - 1.0 - 0.5 * x_col * model.s_theta
        r_cons = model.s_theta - C_quad
        loss = torch.mean(r_point**2) + lambda_s * torch.mean(r_cons**2)
    return loss
# --------------------
# Model & optimizer
# --------------------
if MODE == 'aux':
    model = AuxResPINN(hidden_size=HIDDEN, num_layers=LAYERS, activation=ACTIVATION).to(device)
    loss_fn = lambda m, xc, tq, wq, use_amp=False, device_type='cpu': loss_aux(m, xc, tq, wq, lambda_s=LAMBDA_S, use_amp=use_amp, device_type=device_type)
    title = "Fredholm PINN — Auxiliary Scalar"
else:
    model = ResPINN(hidden_size=HIDDEN, num_layers=LAYERS, activation=ACTIVATION).to(device)
    loss_fn = loss_standard
    title = "Fredholm PINN — Standard"
opt = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2000, factor=0.5, min_lr=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and device.type=='cuda'))
# --------------------
# Metric buffers
# --------------------
iters_hist, loss_hist = [], []
slope_err_hist, C_err_hist = [], []
# --------------------
# Evaluation helpers
# --------------------
xs_eval = torch.linspace(0, 1, 201, device=device).unsqueeze(1)
xs_eval_n = normalize(xs_eval)
u_true_eval = exact_u(xs_eval.squeeze(1).cpu().numpy())
def eval_metrics():
    with torch.no_grad():
        u_pred = model(xs_eval_n).squeeze(1).cpu().numpy()
        A = np.vstack([np.ones_like(u_pred), xs_eval.squeeze(1).cpu().numpy()]).T
        slope = float(np.linalg.lstsq(A, u_pred, rcond=None)[0][1])
        slope_err = abs(slope - 0.3)
        u_t = model(normalize(t_quad)).squeeze(1)
        C_hat = float(torch.sum(w_quad * (t_quad.squeeze(1) * u_t)).cpu().item())
        C_err = abs(C_hat - 0.6)
    return slope_err, C_err
# --------------------
# Train
# --------------------
best_loss = float('inf')
patience_counter = PATIENCE
for ep in range(EPOCHS):
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model, x_col, t_quad, w_quad, use_amp=USE_AMP, device_type=device.type)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    scheduler.step(loss.detach())
    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = PATIENCE
    else:
        patience_counter -= 1
    if patience_counter <= 0:
        print(f"Early stopping at epoch {ep}")
        break
    if (ep % LOG_EVERY) == 0:
        current_lr = opt.param_groups[0]['lr']
        slope_err, C_err = eval_metrics()
        iters_hist.append(ep)
        loss_hist.append(loss.item())
        slope_err_hist.append(slope_err)
        C_err_hist.append(C_err)
        print(f"[Adam] epoch {ep:5d} | loss = {loss.item():.3e} | lr = {current_lr:.2e} | slope_err={slope_err:.2e} | C_err={C_err:.2e}")
print(f"[Adam] final loss = {loss.item():.3e}")
# L-BFGS polish
print("\nStarting L-BFGS polish...")
optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=LBFGS_ITERS, history_size=50)
def closure():
    optimizer_lbfgs.zero_grad()
    loss = loss_fn(model, x_col, t_quad, w_quad, use_amp=USE_AMP, device_type=device.type)
    loss.backward()
    return loss
optimizer_lbfgs.step(closure)
final_loss = loss_fn(model, x_col, t_quad, w_quad, use_amp=USE_AMP, device_type=device.type)
print(f"[L-BFGS] final loss = {final_loss.item():.3e}")
# --------------------
# Final evaluation & plot results
# --------------------
model.eval()
with torch.no_grad():
    u_pred = model(xs_eval_n).squeeze(1).cpu().numpy()
    l2 = float(np.sqrt(np.mean((u_pred - u_true_eval)**2)))
    rel_l2 = l2 / float(np.sqrt(np.mean(u_true_eval**2)))
    A = np.vstack([np.ones_like(u_pred), xs_eval.squeeze(1).cpu().numpy()]).T
    slope = float(np.linalg.lstsq(A, u_pred, rcond=None)[0][1])
    u_t = model(normalize(t_quad)).squeeze(1)
    C_hat = float(torch.sum(w_quad * (t_quad.squeeze(1) * u_t)).cpu().item())
print("\n=== Final Evaluation ===")
print(f"L2 error on 201 points: {l2:.6e} | Rel L2: {rel_l2:.6e}")
print(f"Learned slope: {slope:.6f} | exact: 0.300000")
print(f"C_hat = ∫_0^1 t u(t) dt: {C_hat:.6f} | exact: 0.600000")
if MODE == 'aux' and hasattr(model, 's_theta'):
    print(f"S_theta (parameter): {float(model.s_theta.detach().cpu().item()):.6f}")
# --- Plot prediction vs exact
plt.figure(figsize=(6.5, 4.6))
plt.plot(xs_eval.squeeze(1).cpu().numpy(), u_true_eval, label="Exact: 1 + 0.3 x")
plt.plot(xs_eval.squeeze(1).cpu().numpy(), u_pred, '--', label=f"PINN ({MODE})")
plt.xlabel("x"); plt.ylabel("u(x)")
plt.title(title)
plt.legend(); plt.tight_layout(); plt.show()
# --- Plot training loss
plt.figure(figsize=(6.5, 4.6))
plt.plot(iters_hist, loss_hist)
plt.yscale('log')  # Log scale to reveal small changes
plt.xlabel("epoch"); plt.ylabel("MSE residual loss")
plt.title("Training Loss"); plt.tight_layout(); plt.show()
# --- Plot slope error and C error
fig, ax = plt.subplots(2, 1, figsize=(6.5, 7.2), sharex=True)
ax[0].plot(iters_hist, slope_err_hist)
ax[0].set_ylabel("|slope - 0.3|")
ax[0].set_title("Slope Error")
ax[1].plot(iters_hist, C_err_hist)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("|C - 0.6|")
ax[1].set_title(r"$C$ Error  ($C = \int_0^1 t\,u(t)\,dt$)")
plt.tight_layout(); plt.show()