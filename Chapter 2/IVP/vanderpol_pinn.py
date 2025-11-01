# revised_vanderpol_pinn_with_lossplot.py
# Revised PINN training for van der Pol with stronger physics enforcement and Adam->L-BFGS hybrid
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.integrate import solve_ivp

# --------- Config ----------
OUTDIR = "Chapter2.Plots"
os.makedirs(OUTDIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    torch.cuda.empty_cache()
    a = torch.randn(1, device=device); b = torch.randn(1, device=device); _ = a @ b; del a, b

# Problem parameters
mu = 0.5
y0 = 1.0
y1 = 0.0
t0 = 0.0
T = 10.0

# Training hyperparams (adjustable)
M_colloc = 500             # number of collocation points
hidden = [128, 128, 128]   # network architecture
activation = 'tanh'
lambda_phy = 1.0           # physics weight (try 1.0 then 5.0)
adam_epochs = 5000         # Adam warm-up
lbfgs_max_iter = 500       # L-BFGS fine-tune
lr_adam = 5e-4
grad_clip = 1.0
seed = 42

# Reproducibility
torch.manual_seed(seed); np.random.seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(seed)

# --------- Helpers ----------
def normalize_t(t_np, t_min, t_max):
    """Map t in [t_min,t_max] to [-1,1]"""
    return 2.0 * (t_np - t_min) / (t_max - t_min) - 1.0

def denormalize_t(t_norm, t_min, t_max):
    return (t_norm + 1.0) * (t_max - t_min) / 2.0 + t_min

def kth_derivative(y, t, k):
    """Compute k-th derivative of y wrt t using autograd (repeated differentiations)."""
    assert k >= 1
    current = y
    for _ in range(k):
        grad = torch.autograd.grad(
            outputs=current,
            inputs=t,
            grad_outputs=torch.ones_like(current, device=t.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        current = grad
    return current

# --------- Model ----------
class PINN(nn.Module):
    def __init__(self, hidden_layers, activation='tanh'):
        super().__init__()
        layers = []
        in_dim = 1
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            if activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'swish':
                # use swish via lambda in forward if desired
                layers.append(nn.Tanh())  # placeholder
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, t):
        # t shape: (batch,1)
        out = self.net(t)
        if self.activation == 'swish':
            return out * torch.sigmoid(out)
        return out

# --------- van der Pol RHS closure ----------
def make_vdp_rhs(mu):
    def f_rhs(t, y, y1):
        # y, y1 shape (batch,1)
        return -mu * (y**2 - 1.0) * y1 - y
    return f_rhs

# --------- Reference solver ----------
def compute_reference(mu, y0, y1, t0, T, t_eval):
    def fun(t, y):
        return [y[1], -mu * (y[0]**2 - 1.0) * y[1] - y[0]]
    sol = solve_ivp(fun, (t0, T), [y0, y1], t_eval=t_eval, rtol=1e-6, atol=1e-8, method='RK45')
    return sol.t, sol.y[0]

# --------- Trainer functions ----------
class VanDerPolPINNTrainer:
    def __init__(self, model, f_rhs, t0, y0_list, domain, M_colloc, device):
        self.model = model.to(device)
        self.f_rhs = f_rhs
        self.t0 = t0
        self.y0_list = y0_list
        self.t_min, self.t_max = domain
        self.M = M_colloc
        self.device = device

    def sample_collocation(self):
        # random uniform collocation in original t, then normalize
        t_np = np.random.rand(self.M, 1) * (self.t_max - self.t_min) + self.t_min
        t_norm = normalize_t(t_np, self.t_min, self.t_max)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=self.device, requires_grad=True)
        return t_tensor

    def compute_losses(self, t_colloc, lambda_phy):
        # t_colloc is normalized in [-1,1] with requires_grad=True
        # Map model input expects normalized t
        y_colloc = self.model(t_colloc)
        # compute derivatives with respect to normalized t, but chain rule needed because derivatives wrt real t:
        # if tau = normalized t, real t = denormalize(tau) = a * tau + b, dt/dtau = a
        a = (self.t_max - self.t_min) / 2.0  # dt/dtau
        # compute dy/dtau then dy/dt = (1/a) * dy/dtau
        dy_dtau = kth_derivative(y_colloc, t_colloc, 1)
        dy_dt = dy_dtau / a
        # second derivative: d^2y/dt^2 = (1/a^2) * d^2y/dtau^2
        d2y_dtau2 = kth_derivative(dy_dtau, t_colloc, 1)
        d2y_dt2 = d2y_dtau2 / (a**2)

        # f_rhs expects y and y' with respect to real t
        f_val = self.f_rhs(t_colloc, y_colloc, dy_dt)  # t_colloc here is normalized but f doesn't use t anyway
        residual = d2y_dt2 - f_val
        physics_loss = torch.mean(residual**2)

        # initial condition losses (at t0)
        t0_norm = normalize_t(np.array([[self.t0]]), self.t_min, self.t_max)
        t0_tensor = torch.tensor(t0_norm, dtype=torch.float32, device=self.device, requires_grad=True)
        y0_pred = self.model(t0_tensor)
        # derivatives at t0 normalized -> convert to real derivatives
        dy_dtau_0 = kth_derivative(y0_pred, t0_tensor, 1)
        dy_dt_0 = dy_dtau_0 / a
        data_loss = torch.mean((y0_pred - torch.tensor([[self.y0_list[0]]], device=self.device))**2)
        data_loss = data_loss + torch.mean((dy_dt_0 - torch.tensor([[self.y0_list[1]]], device=self.device))**2)

        return data_loss, physics_loss, residual.detach().cpu().numpy().flatten()

    def train_adam(self, epochs, lr, lambda_phy, grad_clip=None, print_every=500):
        opt = Adam(self.model.parameters(), lr=lr)
        history = {"epoch": [], "total_loss": [], "data_loss": [], "physics_loss": []}
        best_loss = float('inf'); best_state = None; patience = 0

        for ep in range(epochs+1):
            self.model.train()
            t_colloc = self.sample_collocation()
            data_loss, physics_loss, _ = self.compute_losses(t_colloc, lambda_phy)
            total_loss = data_loss + lambda_phy * physics_loss

            opt.zero_grad()
            total_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            opt.step()

            history["epoch"].append(ep)
            history["total_loss"].append(total_loss.item())
            history["data_loss"].append(data_loss.item())
            history["physics_loss"].append(physics_loss.item())

            if total_loss.item() < best_loss - 1e-12:
                best_loss = total_loss.item()
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1

            if ep % print_every == 0:
                print(f"Adam Epoch {ep}: Total {total_loss.item():.6e}, Data {data_loss.item():.6e}, Phys {physics_loss.item():.6e}")

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def train_lbfgs(self, max_iter, lambda_phy, print_every=50):
        # full-batch LBFGS using fixed collocation grid for stable closure
        # prepare fixed collocation grid normalized
        t_np = np.linspace(self.t_min, self.t_max, self.M).reshape(-1,1)
        t_norm = normalize_t(t_np, self.t_min, self.t_max)
        t_fixed = torch.tensor(t_norm, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.LBFGS(self.model.parameters(), lr=1.0, max_iter=max_iter, history_size=50, line_search_fn="strong_wolfe")

        iteration = {"count": 0}
        def closure():
            optimizer.zero_grad()
            data_loss, physics_loss, _ = self.compute_losses(t_fixed, lambda_phy)
            loss = data_loss + lambda_phy * physics_loss
            loss.backward()
            iteration["count"] += 1
            if iteration["count"] % print_every == 0:
                print(f"LBFGS iter {iteration['count']}: Total {loss.item():.6e}, Data {data_loss.item():.6e}, Phys {physics_loss.item():.6e}")
            return loss

        optimizer.step(closure)

    def evaluate(self, t_eval_np):
        # returns y_pin, residuals, and t_real
        self.model.eval()
        # normalize t_eval
        t_norm = normalize_t(t_eval_np.reshape(-1,1), self.t_min, self.t_max)
        t_tensor = torch.tensor(t_norm, dtype=torch.float32, device=self.device, requires_grad=True)
        with torch.no_grad():
            y_pred = self.model(t_tensor).detach().cpu().numpy().flatten()

        # compute residuals (grad-enabled)
        t_req = t_tensor.clone().detach().requires_grad_(True)
        y_req = self.model(t_req)
        a = (self.t_max - self.t_min) / 2.0
        dy_dtau = kth_derivative(y_req, t_req, 1)
        dy_dt = dy_dtau / a
        d2y_dtau2 = kth_derivative(dy_dtau, t_req, 1)
        d2y_dt2 = d2y_dtau2 / (a**2)
        f_val = self.f_rhs(t_req, y_req, dy_dt)
        residual = (d2y_dt2 - f_val).detach().cpu().numpy().flatten()

        t_real = denormalize_t(t_norm.flatten(), self.t_min, self.t_max)
        return t_real, y_pred, residual

# --------- run training ----------
if __name__ == "__main__":
    # Build model & trainer
    model = PINN(hidden, activation=activation)
    vdp_rhs = make_vdp_rhs(mu)
    trainer = VanDerPolPINNTrainer(model, vdp_rhs, t0=t0, y0_list=[y0, y1], domain=(t0, T), M_colloc=M_colloc, device=device)

    # Adam warm-up
    print("Starting Adam warm-up...")
    history_adam = trainer.train_adam(epochs=adam_epochs, lr=lr_adam, lambda_phy=lambda_phy, grad_clip=grad_clip, print_every=500)

    # L-BFGS fine-tune
    print("Starting L-BFGS fine-tuning...")
    trainer.train_lbfgs(max_iter=lbfgs_max_iter, lambda_phy=lambda_phy, print_every=50)

    # Save combined loss history (Adam only here; LBFGS not logged per iteration)
    loss_df = pd.DataFrame({
        "epoch": history_adam["epoch"],
        "total_loss": history_adam["total_loss"],
        "data_loss": history_adam["data_loss"],
        "physics_loss": history_adam["physics_loss"]
    })
    loss_df.to_csv("loss_history.csv", index=False)
    print("Saved loss_history.csv")

    # Evaluate and compare with RK45
    t_eval = np.linspace(t0, T, 1000)
    t_real, y_pin, residual = trainer.evaluate(t_eval)
    t_ref, y_ref = compute_reference(mu, y0, y1, t0, T, t_eval)

    # Save predictions and residuals
    pred_df = pd.DataFrame({"t": t_real, "y_pin": y_pin, "y_ref": y_ref})
    pred_df.to_csv("predictions.csv", index=False)
    res_df = pd.DataFrame({"t": t_real, "residual": residual})
    res_df.to_csv("residuals.csv", index=False)
    print("Saved predictions.csv and residuals.csv")

    # Diagnostics & plots
    abs_err = np.abs(y_pin - y_ref)
    mse = np.mean((y_pin - y_ref)**2)
    rmse = np.sqrt(mse)
    rel_l2 = np.linalg.norm(y_pin - y_ref) / (np.linalg.norm(y_ref) + 1e-12)
    print(f"MSE: {mse:.6e}, RMSE: {rmse:.6e}, RelL2%: {rel_l2*100:.4f}%")

    plt.figure(figsize=(10,5))
    plt.plot(t_ref, y_ref, label='RK45 reference', linewidth=1.6)
    plt.plot(t_real, y_pin, '--', label='PINN prediction', linewidth=1.2)
    plt.xlabel('t'); plt.ylabel('y(t)'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "Ch2.PINN22_revised.png"), dpi=300); plt.show()
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(t_real, abs_err); plt.xlabel('t'); plt.ylabel('|y_pin - y_ref|'); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "Ch2_abs_error_revised.png"), dpi=300); plt.show()
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(t_real, residual); plt.xlabel('t'); plt.ylabel('residual'); plt.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, "Ch2_residual_revised.png"), dpi=300); plt.show()
    plt.close()

    # ======= NEW: LOSS PLOT (total, data, physics) =========
    plt.figure(figsize=(8,5))
    plt.semilogy(loss_df["epoch"], loss_df["total_loss"], label="Total loss")
    plt.semilogy(loss_df["epoch"], loss_df["data_loss"], label="Data loss")
    plt.semilogy(loss_df["epoch"], loss_df["physics_loss"], label="Physics loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss (log scale)")
    plt.title("Training loss history (Total / Data / Physics)")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Ch2.Loss_revised.png"), dpi=300)
    plt.show()
    plt.close()

    print("Plots and CSVs saved. Inspect loss_history.csv, predictions.csv, residuals.csv, and images in", OUTDIR)