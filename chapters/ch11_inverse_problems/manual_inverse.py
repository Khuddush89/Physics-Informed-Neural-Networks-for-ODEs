"""A parameter identification problem computed by hand (Chapter 11).

Book: Chapter 11 (Inverse problems), Section "An Inverse Problem Computed by Hand".

Model     y' = -p y,  y(0) = 1,  so y(t; p) = exp(-p t); unknown p (true value 1).
Data      d = (0.62, 0.36) at t = (0.5, 1.0)  (noisy observations of exp(-t)).
Classical output least squares: minimize J(p) = sum_i (exp(-p t_i) - d_i)^2 by Gauss-Newton,
          p_{k+1} = p_k - (S^T r)/(S^T S),  r_i = exp(-p_k t_i) - d_i,  S_i = -t_i exp(-p_k t_i)
          (sensitivities), starting from p_0 = 0.5.  (In results.json the sensitivities are
          stored under "J" and J(p) under "Phi".)
PINN      y_theta(t) = 1 + t N(t) (hard initial condition), N as in Chapters 5-8
          (w = (0.5, 0.5), b = (-0.5, 0.5), v = (0.5, 0.6), c = 0.1, xi = 2t - 1), trainable p
          with p_0 = 0.5, collocation points tau = (0.25, 0.5, 0.75), lambda_d = lambda_r = 1:
          L = (1/2) sum_i (y_theta(t_i) - d_i)^2 + (1/3) sum_j (y_theta'(tau_j) + p y_theta(tau_j))^2.
The script evaluates the loss and its gradient at the initialization by hand and with PyTorch and
performs one Adam step (learning rate 1e-3) for p.

Run:  python chapters/ch11_inverse_problems/manual_inverse.py   (well under 1 s; --quick is identical)
"""
from __future__ import annotations

import numpy as np
import torch

from pinnode import Example

W = np.array([0.5, 0.5])
B = np.array([-0.5, 0.5])
V = np.array([0.5, 0.6])
C = 0.1
T_DATA = np.array([0.5, 1.0])
D = np.array([0.62, 0.36])
TAU = np.array([0.25, 0.5, 0.75])
P0 = 0.5


def gauss_newton(p0=P0, iters=6):
    rows = []
    p = p0
    for k in range(iters):
        y = np.exp(-p * T_DATA)
        r = y - D
        J = -T_DATA * y
        phi = float(r @ r)
        step = -float(J @ r) / float(J @ J)
        rows.append({"k": k, "p": p, "r": r, "J": J, "Phi": phi, "JTr": float(J @ r),
                     "JTJ": float(J @ J), "step": step})
        p = p + step
    return rows, p


def features(t):
    t = np.atleast_1d(t)[:, None]
    h = np.tanh(W * (2.0 * t - 1.0) + B)
    return h, 1.0 - h ** 2


def y_theta(t):
    t = np.atleast_1d(t)
    h, s = features(t)
    N = h @ V + C
    dN = (2.0 * W * s) @ V
    return 1.0 + t * N, N + t * dN


def pinn_hand(p=P0):
    y_d, _ = y_theta(T_DATA)
    mis = y_d - D
    y_c, dy_c = y_theta(TAU)
    R = dy_c + p * y_c
    L_d = float(np.mean(mis ** 2))
    L_r = float(np.mean(R ** 2))
    dL_dp = 2.0 / len(TAU) * float(R @ y_c)
    return {"y_data": y_d, "misfit": mis, "y_col": y_c, "dy_col": dy_c, "R": R,
            "L_d": L_d, "L_r": L_r, "L": L_d + L_r, "dL_dp": dL_dp}


def torch_check(p0=P0):
    torch.set_default_dtype(torch.float64)
    w = torch.tensor(W, requires_grad=True)
    b = torch.tensor(B, requires_grad=True)
    v = torch.tensor(V, requires_grad=True)
    c = torch.tensor(C, requires_grad=True)
    p = torch.tensor(p0, requires_grad=True)
    params = [w, b, v, c, p]

    def y(t):
        return 1 + t * (torch.tanh(w * (2 * t[:, None] - 1) + b) @ v + c)

    def loss():
        tau = torch.tensor(TAU, requires_grad=True)
        yc = y(tau)
        dy, = torch.autograd.grad(yc.sum(), tau, create_graph=True)
        R = dy + p * yc
        Ld = ((y(torch.tensor(T_DATA)) - torch.tensor(D)) ** 2).mean()
        return Ld + (R ** 2).mean()

    L = loss()
    g = torch.autograd.grad(L, params)
    opt = torch.optim.Adam(params, lr=1e-3)
    for prm, gg in zip(params, g):
        prm.grad = gg.clone()
    opt.step()
    L_after = loss().item()
    return L.item(), [a.detach().numpy() for a in g], p.item(), L_after


def tolist(d):
    if isinstance(d, dict):
        return {str(k): tolist(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [tolist(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def main():
    ex = Example("ch11_inverse_problems", "manual_inverse",
                 description="Hand calculation: Gauss-Newton versus the inverse-PINN loss and gradient",
                 default_seed=0)
    np.set_printoptions(precision=6)
    rows, p_hat = gauss_newton()
    for r in rows:
        print(f"GN k={r['k']}: p={r['p']:.10f} r={r['r']} J={r['J']} Phi={r['Phi']:.6e} "
              f"J^Tr={r['JTr']:.6e} J^TJ={r['JTJ']:.6f} step={r['step']:.6e}")
    print("least-squares estimate p_hat =", p_hat, " residuals", np.exp(-p_hat * T_DATA) - D)
    ph = pinn_hand()
    Lt, gt, p_new, L_after = torch_check()
    print("PINN y(t_data)", ph["y_data"], " misfit", ph["misfit"], " L_d", ph["L_d"])
    print("PINN y(tau), y'(tau)", ph["y_col"], ph["dy_col"], " R", ph["R"], " L_r", ph["L_r"])
    print("L [hand | torch]", ph["L"], Lt, " dL/dp [hand | torch]", ph["dL_dp"], gt[4])
    print("torch gradient w, b, v, c:", gt[:4])
    print("after one Adam step: p =", p_new, " L =", L_after)
    diffs = {"L": abs(ph["L"] - Lt), "dLdp": abs(ph["dL_dp"] - float(gt[4]))}
    print("checks:", {k: f"{v:.1e}" for k, v in diffs.items()})
    assert max(diffs.values()) < 1e-12
    ex.save_results({"problem": "y' = -p y, y(0) = 1; data d = (0.62, 0.36) at t = (0.5, 1)",
                     "gauss_newton": tolist(rows), "p_hat": p_hat,
                     "pinn": tolist(ph), "torch": {"L": Lt, "grad": tolist(gt), "p_after_adam": p_new,
                                                    "L_after_adam": L_after},
                     "checks": diffs})


if __name__ == "__main__":
    main()
