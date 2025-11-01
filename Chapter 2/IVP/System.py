# lotka_volterra_pinn_fixed.py
# Fixed PINN training for Lotka-Volterra using log-transform
# Produces CSVs only: predictions_volterra_log.csv, residuals_volterra_log.csv, loss_history_volterra_log.csv
# Shows direct plots without saving

import os
import math
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
from torch.optim import Adam, LBFGS
import matplotlib.pyplot as plt

# ----------------- Config -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Lotka-Volterra parameters
a, b, c, d = 1.0, 0.1, 1.5, 0.075
x0_val, y0_val = 10.0, 5.0
t0, T = 0.0, 15.0

# Training hyperparameters
M_colloc = 2000           # collocation points
t_test_points = 2000      # evaluation points
adam_epochs = 30000       # Adam training
lbfgs_max_iter = 1000     # L-BFGS fine-tuning
lr_adam = 1e-3
lambda_data = 10.0        # data loss weight
lambda_phy = 1.0          # physics loss weight

# Network architecture
n_hidden_layers = 4
hidden_size = 64
activation = nn.Tanh()

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

# ----------------- Reference Solution -----------------
def lv_system(t, z):
    x, y = z
    return [a * x - b * x * y, -c * y + d * x * y]

sol = solve_ivp(lv_system, (t0, T), [x0_val, y0_val], 
                rtol=1e-8, atol=1e-10, method='RK45', dense_output=True)
t_ref_dense = np.linspace(t0, T, t_test_points)
x_ref_dense, y_ref_dense = sol.sol(t_ref_dense)

# ----------------- PINN Model -----------------
class PINN_Log(nn.Module):
    def __init__(self, n_hidden, hidden_size):
        super().__init__()
        layers = [nn.Linear(1, hidden_size), activation]
        
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), activation])
        
        layers.append(nn.Linear(hidden_size, 2))  # outputs: z=log(x), w=log(y)
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, t):
        return self.net(t)

# ----------------- Physics Residuals -----------------
def physics_residuals_log(model, t_tensor):
    t_tensor.requires_grad_(True)
    out = model(t_tensor)
    z = out[:, 0:1]  # log(x)
    w = out[:, 1:2]  # log(y)
    
    # First derivatives
    dz_dt = torch.autograd.grad(z, t_tensor, torch.ones_like(z), 
                               create_graph=True, retain_graph=True)[0]
    dw_dt = torch.autograd.grad(w, t_tensor, torch.ones_like(w), 
                               create_graph=True, retain_graph=True)[0]
    
    # Residuals in log-space
    # z' = a - b*exp(w), w' = -c + d*exp(z)
    r1 = dz_dt - (a - b * torch.exp(w))
    r2 = dw_dt - (-c + d * torch.exp(z))
    
    return r1, r2, z, w

# ----------------- Training Utilities -----------------
def sample_collocation(M):
    t_np = np.random.uniform(t0, T, (M, 1)).astype(np.float32)
    return torch.tensor(t_np, dtype=torch.float32, device=device)

def evaluate_model(model, t_eval_np):
    t_tensor = torch.tensor(t_eval_np.reshape(-1, 1).astype(np.float32), 
                           device=device, requires_grad=True)
    
    with torch.no_grad():
        out = model(t_tensor).cpu().numpy()
    
    z_pred = out[:, 0]
    w_pred = out[:, 1]
    x_pred = np.exp(z_pred)
    y_pred = np.exp(w_pred)
    
    # Compute residuals
    r1, r2, _, _ = physics_residuals_log(model, t_tensor)
    r1_np = r1.detach().cpu().numpy().flatten()
    r2_np = r2.detach().cpu().numpy().flatten()
    
    return x_pred, y_pred, r1_np, r2_np

# ----------------- Initialize Model -----------------
model = PINN_Log(n_hidden_layers, hidden_size).to(device)
optimizer = Adam(model.parameters(), lr=lr_adam)
# Fixed: Use StepLR instead of ReduceLROnPlateau to avoid the tensor conversion warning
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.8)

# Training history
history = {
    "epoch": [],
    "total_loss": [], 
    "data_loss": [],
    "physics_loss": []
}

# ----------------- Initial Condition Tensors -----------------
t0_tensor = torch.tensor([[t0]], dtype=torch.float32, device=device, requires_grad=True)
z0_true = torch.tensor([[math.log(x0_val)]], dtype=torch.float32, device=device)
w0_true = torch.tensor([[math.log(y0_val)]], dtype=torch.float32, device=device)

print("Starting Adam training...")

# ----------------- Adam Training -----------------
for epoch in range(1, adam_epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    # Sample collocation points
    t_colloc = sample_collocation(M_colloc).requires_grad_(True)
    
    # Data loss (initial conditions)
    out0 = model(t0_tensor)
    z0_pred = out0[:, 0:1]
    w0_pred = out0[:, 1:2]
    
    data_loss = (torch.nn.functional.mse_loss(z0_pred, z0_true) + 
                 torch.nn.functional.mse_loss(w0_pred, w0_true))
    
    # Physics loss
    r1, r2, _, _ = physics_residuals_log(model, t_colloc)
    physics_loss = torch.mean(r1**2 + r2**2)
    
    # Total loss
    total_loss = lambda_data * data_loss + lambda_phy * physics_loss
    
    total_loss.backward()
    optimizer.step()
    scheduler.step()
    
    # Record history
    if epoch % 500 == 0 or epoch == 1 or epoch == adam_epochs:
        history["epoch"].append(epoch)
        history["total_loss"].append(total_loss.item())
        history["data_loss"].append(data_loss.item())
        history["physics_loss"].append(physics_loss.item())
        
        print(f"Epoch {epoch}: Total Loss = {total_loss.item():.6e}, "
              f"Data Loss = {data_loss.item():.6e}, "
              f"Physics Loss = {physics_loss.item():.6e}")

print("Adam training completed.")

# ----------------- L-BFGS Fine-tuning -----------------
print("Starting L-BFGS fine-tuning...")

# Fixed collocation points for L-BFGS
t_fixed_np = np.linspace(t0, T, M_colloc).reshape(-1, 1).astype(np.float32)
t_fixed = torch.tensor(t_fixed_np, dtype=torch.float32, device=device, requires_grad=True)

optimizer_lbfgs = LBFGS(model.parameters(), max_iter=lbfgs_max_iter, 
                       history_size=100, line_search_fn='strong_wolfe')

def closure():
    optimizer_lbfgs.zero_grad()
    
    # Data loss
    out0 = model(t0_tensor)
    z0_pred = out0[:, 0:1]
    w0_pred = out0[:, 1:2]
    data_loss = (torch.nn.functional.mse_loss(z0_pred, z0_true) + 
                 torch.nn.functional.mse_loss(w0_pred, w0_true))
    
    # Physics loss
    r1, r2, _, _ = physics_residuals_log(model, t_fixed)
    physics_loss = torch.mean(r1**2 + r2**2)
    
    total_loss = lambda_data * data_loss + lambda_phy * physics_loss
    total_loss.backward()
    
    return total_loss

# Run L-BFGS
optimizer_lbfgs.step(closure)
print("L-BFGS fine-tuning completed.")

# ----------------- Final Evaluation -----------------
print("Evaluating final model...")

# Add final evaluation to history
model.eval()
with torch.no_grad():
    out0 = model(t0_tensor)
    z0_pred = out0[:, 0:1]
    w0_pred = out0[:, 1:2]
    final_data_loss = (torch.nn.functional.mse_loss(z0_pred, z0_true) + 
                      torch.nn.functional.mse_loss(w0_pred, w0_true)).item()

t_colloc_eval = sample_collocation(1000).requires_grad_(True)
r1, r2, _, _ = physics_residuals_log(model, t_colloc_eval)
final_physics_loss = torch.mean(r1**2 + r2**2).item()
final_total_loss = lambda_data * final_data_loss + lambda_phy * final_physics_loss

history["epoch"].append(adam_epochs + lbfgs_max_iter)
history["total_loss"].append(final_total_loss)
history["data_loss"].append(final_data_loss)
history["physics_loss"].append(final_physics_loss)

# Get predictions
t_eval = t_ref_dense
x_pred, y_pred, r1_vals, r2_vals = evaluate_model(model, t_eval)

# Calculate metrics
mse_x = np.mean((x_pred - x_ref_dense)**2)
mse_y = np.mean((y_pred - y_ref_dense)**2)
mse_total = mse_x + mse_y
rmse_total = np.sqrt(mse_total)

norm_ref = np.linalg.norm(np.concatenate([x_ref_dense, y_ref_dense]))
norm_error = np.linalg.norm(np.concatenate([x_pred - x_ref_dense, y_pred - y_ref_dense]))
rel_l2 = norm_error / norm_ref if norm_ref > 0 else norm_error

print(f"\nFinal Metrics:")
print(f"MSE (x): {mse_x:.6e}")
print(f"MSE (y): {mse_y:.6e}")
print(f"Total MSE: {mse_total:.6e}")
print(f"Total RMSE: {rmse_total:.6e}")
print(f"Relative L2 Error: {rel_l2*100:.6f}%")

# ----------------- Direct Plots (without saving) -----------------
print("\nDisplaying plots...")

# 1) PINN vs RK45 comparison
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t_eval, x_ref_dense, 'b-', label='x (RK45)', linewidth=2)
plt.plot(t_eval, x_pred, 'r--', label='x (PINN)', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Prey Population x')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Prey Population: PINN vs RK45')

plt.subplot(2, 2, 2)
plt.plot(t_eval, y_ref_dense, 'g-', label='y (RK45)', linewidth=2)
plt.plot(t_eval, y_pred, 'm--', label='y (PINN)', linewidth=2)
plt.xlabel('Time t')
plt.ylabel('Predator Population y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Predator Population: PINN vs RK45')

plt.subplot(2, 2, 3)
plt.plot(t_eval, np.abs(x_pred - x_ref_dense), 'b-', label='x error', linewidth=1.5)
plt.plot(t_eval, np.abs(y_pred - y_ref_dense), 'r-', label='y error', linewidth=1.5)
plt.xlabel('Time t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.title('Absolute Errors (Log Scale)')

plt.subplot(2, 2, 4)
plt.plot(x_ref_dense, y_ref_dense, 'k-', label='RK45', linewidth=2)
plt.plot(x_pred, y_pred, 'c--', label='PINN', linewidth=2)
plt.xlabel('Prey x')
plt.ylabel('Predator y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Phase Space')

plt.tight_layout()
plt.show()

# 2) Loss history
plt.figure(figsize=(10, 6))
plt.semilogy(history["epoch"], history["total_loss"], 'b-', label='Total Loss', linewidth=2)
plt.semilogy(history["epoch"], history["data_loss"], 'g--', label='Data Loss', linewidth=1.5)
plt.semilogy(history["epoch"], history["physics_loss"], 'r--', label='Physics Loss', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Training Loss History')
plt.show()

# 3) Residuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(t_eval, r1_vals, 'b-', label='Residual r1', linewidth=1.5)
plt.plot(t_eval, r2_vals, 'r-', label='Residual r2', linewidth=1.5)
plt.xlabel('Time t')
plt.ylabel('Residual Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Physics Residuals')

plt.subplot(1, 2, 2)
residual_norm = np.sqrt(r1_vals**2 + r2_vals**2)
plt.plot(t_eval, residual_norm, 'k-', linewidth=1.5)
plt.xlabel('Time t')
plt.ylabel('Residual Norm')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.title('Physics Residual Norm (Log Scale)')

plt.tight_layout()
plt.show()

# ----------------- Save CSV Files -----------------
print("\nSaving CSV files...")

# Predictions CSV
pred_df = pd.DataFrame({
    "t": t_eval,
    "x_ref": x_ref_dense,
    "y_ref": y_ref_dense,
    "x_pinn": x_pred,
    "y_pinn": y_pred,
    "abs_err_x": np.abs(x_pred - x_ref_dense),
    "abs_err_y": np.abs(y_pred - y_ref_dense),
    "rel_err_x": np.abs(x_pred - x_ref_dense) / (np.abs(x_ref_dense) + 1e-12),
    "rel_err_y": np.abs(y_pred - y_ref_dense) / (np.abs(y_ref_dense) + 1e-12)
})
pred_df.to_csv("predictions_volterra_log.csv", index=False)
print("Saved: predictions_volterra_log.csv")

# Residuals CSV
res_df = pd.DataFrame({
    "t": t_eval,
    "r1": r1_vals,
    "r2": r2_vals,
    "r_norm": np.sqrt(r1_vals**2 + r2_vals**2)
})
res_df.to_csv("residuals_volterra_log.csv", index=False)
print("Saved: residuals_volterra_log.csv")

# Loss history CSV
loss_df = pd.DataFrame(history)
loss_df.to_csv("loss_history_volterra_log.csv", index=False)
print("Saved: loss_history_volterra_log.csv")

print("\nAll CSV files saved successfully!")
print(f"Final loss: {final_total_loss:.6e}")