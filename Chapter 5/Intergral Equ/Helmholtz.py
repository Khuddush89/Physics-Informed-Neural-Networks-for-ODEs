#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ===============================================================
# PINN for 1D Helmholtz IE on [0,1]
#   u(x) = 1 + scale * (i/(2k)) ∫_0^1 e^{ik|x-t|} u(t) dt
# Quadratic Filon (3 nodes/panel) with PARTIAL weights (s=0,1/2,1)
# SIREN PINN; Nyström reference uses the SAME G = i/(2k) e^{ik|x-t|}.
# Saves: loss_curve.png, relerr_curve.png,
#        helmholtz_compare_re_im.png, helmholtz_compare_mag.png
# ===============================================================
import math, numpy as np, torch, torch.nn as nn, torch.optim as optim
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1234); np.random.seed(1234)

# ---------------- problem / training knobs ----------------
k      = 10.0       # wavenumber
scale  = 6.0        # kernel scale; larger => stronger coupling
epochs = 20000
lr_init, lr_min = 5e-4, 5e-5
clip_norm = 3.0
do_lbfgs  = True
DATA_ASSIST = False      # set True to add a tiny supervised term from Nyström

def f_of_x(x):  # RHS
    return torch.ones_like(x, dtype=torch.cdouble)

# ---------------- panels & Filon (full + partial) ----------
PANELS_PER_WAVELENGTH = 4.0
wavelength = 2.0 * math.pi / k
P = max(200, int(math.ceil((1.0 / wavelength) * PANELS_PER_WAVELENGTH)))
h = 1.0 / P
theta = k*h

# Lagrange on r∈[0,1]: L0=2(r^2-3r/2+1/2), L1=-4(r^2-r), L2=2(r^2-r/2)
def _Jm_partial(theta, s):
    th = theta
    if abs(th) < 1e-8:
        J0 = s + 1j*th*s*s/2 - (th**2)*s**3/6 - 1j*(th**3)*s**4/24
        J1 = s**2/2 + 1j*th*s**3/3 - (th**2)*s**4/8 - 1j*(th**3)*s**5/30
        J2 = s**3/3 + 1j*th*s**4/4 - (th**2)*s**5/10 - 1j*(th**3)*s**6/36
    else:
        eiths = np.exp(1j*th*s)
        J0 = (eiths - 1) / (1j*th)
        J1 = (eiths*(1j*th*s - 1) + 1) / ((1j*th)**2)
        J2 = (eiths*((1j*th*s)**2 - 2j*th*s + 2) - 2) / ((1j*th)**3)
    return J0, J1, J2

def _alphas_partial(theta, s):
    J0,J1,J2 = _Jm_partial(theta, s)
    a0 = 2.0*(J2 - 1.5*J1 + 0.5*J0)
    a1 = -4.0*(J2 - J1)
    a2 = 2.0*(J2 - 0.5*J1)
    return np.array([a0,a1,a2], dtype=np.complex128)

def _alphas_full(theta): return _alphas_partial(theta, 1.0)
# left (0..s) with e^{-ik r h}, right (s..1) with e^{+ik r h}
alpha_minus_full = _alphas_full(-theta)
alpha_plus_full  = _alphas_full(+theta)
alpha_minus_s0   = _alphas_partial(-theta, 0.0)
alpha_minus_s05  = _alphas_partial(-theta, 0.5)
alpha_minus_s1   = _alphas_partial(-theta, 1.0)
alpha_plus_s0R   = alpha_plus_full - _alphas_partial(+theta, 0.0)
alpha_plus_s05R  = alpha_plus_full - _alphas_partial(+theta, 0.5)
alpha_plus_s1R   = alpha_plus_full - _alphas_partial(+theta, 1.0)

def col(v): return torch.tensor(v, dtype=torch.cdouble, device=DEVICE).view(3,1)
a_minus_full_t = col(alpha_minus_full); a_plus_full_t = col(alpha_plus_full)
a_minus_s0_t=a_minus_s0_t  = col(alpha_minus_s0)
a_minus_s05_t               = col(alpha_minus_s05)
a_minus_s1_t                = col(alpha_minus_s1)
a_plus_s0R_t                = col(alpha_plus_s0R)
a_plus_s05R_t               = col(alpha_plus_s05R)
a_plus_s1R_t                = col(alpha_plus_s1R)

# --------------------- SIREN PINN ---------------------
class Sine(nn.Module):
    def __init__(self, w0=30.0): super().__init__(); self.w0 = w0
    def forward(self, x): return torch.sin(self.w0 * x)

class SIRENBlock(nn.Module):
    def __init__(self, in_dim, out_dim, w0=30.0, is_first=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, dtype=torch.float64)
        self.act = Sine(w0 if is_first else 1.0)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_dim, 1/in_dim)
            else:
                c = math.sqrt(6/in_dim)
                self.linear.weight.uniform_(-c, c)
            self.linear.bias.zero_()
    def forward(self, x): return self.act(self.linear(x))

class HelmholtzPINN(nn.Module):
    def __init__(self, width=128, depth=3, w0=10.0):
        super().__init__()
        self.inp = SIRENBlock(1, width, w0=w0, is_first=True)
        self.hids = nn.ModuleList([SIRENBlock(width, width) for _ in range(depth-1)])
        self.out_re = nn.Linear(width, 1, dtype=torch.float64)
        self.out_im = nn.Linear(width, 1, dtype=torch.float64)
        nn.init.xavier_uniform_(self.out_re.weight); nn.init.zeros_(self.out_re.bias)
        nn.init.xavier_uniform_(self.out_im.weight); nn.init.zeros_(self.out_im.bias)
    def forward(self, x):
        h = self.inp(x)
        for blk in self.hids:
            h = h + blk(h)      # residual skip
        re = self.out_re(h); im = self.out_im(h)
        return torch.complex(re, im)

model = HelmholtzPINN(width=128, depth=3, w0=10.0).to(DEVICE)

# ---------------- geometry & phases -------------------
a_starts = (torch.arange(P, dtype=torch.float64, device=DEVICE).view(P,1) * h)  # (P,1)
t_nodes  = torch.stack([a_starts, a_starts + 0.5*h, a_starts + h], dim=1)       # (P,3,1)
phase_minus_panel = torch.exp(-1j * k * a_starts)  # e^{-ik a_p}
phase_plus_panel  = torch.exp(+1j * k * a_starts)  # e^{+ik a_p}

# ---------------- residual (3 per panel) ---------------
def residuals(model, data_xy=None, lam_data=0.0):
    u_nodes = model(t_nodes.reshape(-1,1)).reshape(P,3,1)  # (P,3,1)
    uP3     = u_nodes.squeeze(-1)                          # (P,3)

    Cminus_full = (phase_minus_panel * (uP3 @ a_minus_full_t)) * h
    Cplus_full  = (phase_plus_panel  * (uP3 @ a_plus_full_t))  * h

    zeros = torch.zeros(1,1, dtype=torch.cdouble, device=DEVICE)
    cum_minus      = torch.cumsum(Cminus_full, dim=0)
    sum_left_before = torch.cat([zeros, cum_minus[:-1,:]], dim=0)
    sum_right_from  = torch.flip(torch.cumsum(torch.flip(Cplus_full,[0]), dim=0), [0])

    def res_at_s(s, a_minus_s_t, a_plus_sR_t, node_idx):
        left_part  = (phase_minus_panel * (uP3 @ a_minus_s_t)) * h
        right_part = (phase_plus_panel  * (uP3 @ a_plus_sR_t)) * h
        Left  = sum_left_before + left_part
        Right = (sum_right_from - Cplus_full) + right_part
        x_s = a_starts + s*h
        ker = scale * (1.0j/(2.0*k)) * (torch.exp(1.0j*k*x_s)*Left + torch.exp(-1.0j*k*x_s)*Right)
        u_x = uP3[:, node_idx:node_idx+1]
        return u_x - f_of_x(x_s) - ker

    R0 = res_at_s(0.0, a_minus_s0_t,  a_plus_s0R_t,  0)
    Rm = res_at_s(0.5, a_minus_s05_t, a_plus_s05R_t, 1)
    R1 = res_at_s(1.0, a_minus_s1_t,  a_plus_s1R_t,  2)
    Rall = torch.cat([R0, Rm, R1], dim=0)
    loss_phys = torch.mean((Rall.conj()*Rall).real)

    loss_data = torch.tensor(0.0, dtype=torch.float64, device=DEVICE)
    if data_xy is not None and lam_data > 0.0:
        x_b, u_b = data_xy
        pred = model(x_b)
        loss_data = torch.mean(torch.abs(pred - u_b)**2)

    return loss_phys + lam_data*loss_data, loss_phys, loss_data

# --------------- Nyström reference (CONSISTENT) --------------
def nystrom_reference(k, scale, Npw=64):
    wavelength = 2.0 * math.pi / k
    N = int(max(800, math.ceil((1.0 / wavelength) * Npw)))
    x = np.linspace(0.0, 1.0, N)
    w = np.ones_like(x); w[0]=w[-1]=0.5; w *= (1.0/(N-1))
    X, T = np.meshgrid(x, x, indexing='ij')
    # *** IMPORTANT: SAME GREEN AS PINN ***
    G = (1j/(2*k)) * np.exp(1j * k * np.abs(X - T))
    A = np.eye(N, dtype=np.complex128) - scale * (G * w[None,:])
    b = np.ones(N, dtype=np.complex128)
    u = np.linalg.solve(A, b)
    return x, w, u

x_ref, w_ref, u_ref = nystrom_reference(k, scale)

def rel_L2_vs_ref(model):
    xs = torch.linspace(0,1,1500,device=DEVICE).view(-1,1)
    with torch.no_grad():
        up = model(xs).cpu().numpy().reshape(-1)
    xg = xs.cpu().numpy().ravel()
    ur = np.interp(xg, x_ref, u_ref.real) + 1j*np.interp(xg, x_ref, u_ref.imag)
    w = np.concatenate(([0.5], np.ones(len(xg)-2), [0.5])) * (xg[-1]-xg[0])/(len(xg)-1)
    num = np.sqrt(np.sum(w * np.abs(up-ur)**2))
    den = np.sqrt(np.sum(w * np.abs(ur)**2))
    return float(num/den)

# tiny supervised assist (optional)
if DATA_ASSIST:
    xb = torch.linspace(0,1,1024,device=DEVICE).view(-1,1)
    ub = np.interp(xb.cpu().numpy().ravel(), x_ref, u_ref.real) \
       + 1j*np.interp(xb.cpu().numpy().ravel(), x_ref, u_ref.imag)
    ub = torch.tensor(ub, dtype=torch.cdouble, device=DEVICE).view(-1,1)
    data_xy = (xb, ub)
    lam_data = 1e-3
else:
    data_xy = None; lam_data = 0.0

# ---------------------- training ----------------------
adam  = optim.Adam(model.parameters(), lr=lr_init)
sched = optim.lr_scheduler.CosineAnnealingLR(adam, T_max=epochs, eta_min=lr_min)
hist_loss, hist_relE = [], []

for ep in range(1, epochs+1):
    adam.zero_grad(set_to_none=True)
    loss, loss_phys, loss_data = residuals(model, data_xy, lam_data)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    adam.step(); sched.step()
    hist_loss.append(loss.item())
    if ep % 500 == 0 or ep == 1:
        relE = rel_L2_vs_ref(model)
        hist_relE.append((ep, relE))
        print(f"[Adam {ep:5d}] loss={loss.item():.3e} | RelErr={relE:.3e}")

if do_lbfgs:
    lbfgs = optim.LBFGS(model.parameters(), max_iter=400, history_size=100,
                        line_search_fn="strong_wolfe",
                        tolerance_grad=1e-12, tolerance_change=1e-12)
    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss, _, _ = residuals(model, data_xy, lam_data)
        loss.backward()
        return loss
    print("[L-BFGS] starting...")
    final_loss = lbfgs.step(closure)
    print(f"[L-BFGS] final loss = {final_loss.item():.3e}")
    hist_loss.append(final_loss.item())
    relE = rel_L2_vs_ref(model)
    hist_relE.append((epochs, relE))
    print(f"Final Relative L2 Error (PINN vs Nyström): {relE:.3e}")

# ---------------- plots & checkpoints -----------------
xx = torch.linspace(0.0, 1.0, 1200, device=DEVICE).view(-1,1)
with torch.no_grad():
    u_pinn = model(xx).cpu().numpy().reshape(-1)
x_np = xx.cpu().numpy().ravel()
u_ref_interp = np.interp(x_np, x_ref, u_ref.real) + 1j*np.interp(x_np, x_ref, u_ref.imag)

# training loss
plt.figure(figsize=(7.2,4.2))
plt.semilogy(hist_loss, lw=1.8)
plt.xlabel("epoch"); plt.ylabel("loss (MSE residual)")
plt.title("Training loss"); plt.tight_layout()
plt.savefig("loss_curve.png", dpi=150); plt.show()

# relative L2 error (vs Nyström)
if hist_relE:
    e_ep, e_val = zip(*hist_relE)
    plt.figure(figsize=(7.2,4.2))
    plt.semilogy(e_ep, e_val, 'o-', lw=1.8)
    plt.xlabel("epoch"); plt.ylabel("relative $L^2$ error vs Nyström")
    plt.title("PINN → reference error"); plt.tight_layout()
    plt.savefig("relerr_curve.png", dpi=150); plt.show()

# Re/Im comparison
plt.figure(figsize=(7.2,4.6))
plt.plot(x_ref, u_ref.real, 'k--', lw=2, label='Re $u_{\\mathrm{ref}}$ (Nyström)')
plt.plot(x_ref, u_ref.imag, 'k:',  lw=2, label='Im $u_{\\mathrm{ref}}$ (Nyström)')
plt.plot(x_np,  u_pinn.real,             label='Re $u_\\theta$')
plt.plot(x_np,  u_pinn.imag,             label='Im $u_\\theta$')
plt.xlabel("$x$"); plt.title(f"Helmholtz IE on [0,1] (k={k}, scale={scale}): PINN vs Nyström")
plt.legend(ncol=2); plt.tight_layout()
plt.savefig("helmholtz_compare_re_im.png", dpi=150); plt.show()

# magnitude comparison
plt.figure(figsize=(7.2,4.6))
plt.plot(x_ref, np.abs(u_ref), 'k--', lw=2, label='$|u_{\\mathrm{ref}}|$ (Nyström)')
plt.plot(x_np,  np.abs(u_pinn),          label='$|u_\\theta|$')
plt.xlabel("$x$"); plt.title("Magnitude comparison"); plt.legend()
plt.tight_layout(); plt.savefig("helmholtz_compare_mag.png", dpi=150); plt.show()

# checkpoints
for x0 in [0.0, 0.25, 0.50, 0.75, 1.0]:
    xv = torch.tensor([[x0]], device=DEVICE)
    with torch.no_grad():
        up = model(xv).item()
    ur = np.interp([x0], x_ref, u_ref.real)[0] + 1j*np.interp([x0], x_ref, u_ref.imag)[0]
    print(f"x={x0:4.2f}  Re(u_PINN)={up.real: .6f}  Im(u_PINN)={up.imag: .6f}  "
          f"Re(u_ref)={ur.real: .6f}  Im(u_ref)={ur.imag: .6f}  |err|={abs(up-ur):.2e}")
