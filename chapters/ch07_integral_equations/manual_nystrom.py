"""Nystrom method and PINN quadrature residual computed by hand (Chapter 7).

Book: Chapter 7 (Integral equations), Section "An Integral Equation Computed by Hand".

Problem   u(x) = 1 + (1/2) int_0^1 x t u(t) dt on [0, 1];  exact solution u(x) = 1 + 3x/10.
Nystrom   with the one-point Gauss rule (midpoint) and the two-point Gauss rule on [0, 1]:
          the nodal values solve a 1 x 1 and a 2 x 2 linear system, and the Nystrom
          interpolant gives u on [0, 1].
PINN      u_theta(x) = N(x) = v_1 tanh(w_1 xi + b_1) + v_2 tanh(w_2 xi + b_2) + c, xi = 2x - 1,
          w = (0.5, 0.5), b = (-0.5, 0.5), v = (0.5, 0.6), c = 0.1  (as in Chapters 5 and 6);
          residual R(x_i) = u_theta(x_i) - 1 - (x_i/2) S,  S = sum_j w_j t_j u_theta(t_j)
          (two-point Gauss rule) at x_i = 0, 1/2, 1;  loss L = (1/3) sum_i R(x_i)^2.
Because the kernel is separable, S is computed once and shared by all collocation points.

The script checks the residual, the loss and the gradient against PyTorch, and solves for the
output weights (v_1, v_2, c) that make the three residuals vanish (the residual is affine in them)
with the one- and the two-point rule, showing that the network inherits the quadrature error.

Run:  python chapters/ch07_integral_equations/manual_nystrom.py   (well under 1 s; --quick is identical)
"""
from __future__ import annotations

import numpy as np
import torch

from pinnode import Example

W = np.array([0.5, 0.5])
B = np.array([-0.5, 0.5])
V = np.array([0.5, 0.6])
C = 0.1
X_COL = np.array([0.0, 0.5, 1.0])
LAM = 0.5
RULES = {1: (np.array([0.5]), np.array([1.0])),
         2: (np.array([0.5 - np.sqrt(3.0) / 6.0, 0.5 + np.sqrt(3.0) / 6.0]), np.array([0.5, 0.5]))}


def exact(x):
    return 1.0 + 0.3 * x


def nystrom(q):
    t, w = RULES[q]
    A = np.eye(q) - LAM * np.outer(t, w * t)          # u_i - lam t_i sum_j w_j t_j u_j = 1
    u = np.linalg.solve(A, np.ones(q))
    S = float(np.sum(w * t * u))
    slope = LAM * S                                    # interpolant u(x) = 1 + lam x S
    return {"nodes": t, "weights": w, "matrix": A, "u_nodes": u, "S": S, "slope": slope,
            "max_error": abs(slope - 0.3)}


def features(x):
    x = np.atleast_1d(x)[:, None]
    return np.tanh(W * (2.0 * x - 1.0) + B)


def pinn_residual(q, v, c):
    t, w = RULES[q]
    u_nodes = features(t) @ v + c
    S = float(np.sum(w * t * u_nodes))
    u_col = features(X_COL) @ v + c
    R = u_col - 1.0 - LAM * X_COL * S
    return u_nodes, S, u_col, R


def solve_output(q):
    """Output weights with R(x_i) = 0, i = 1..3 (R is affine in v_1, v_2, c)."""
    t, w = RULES[q]
    Ht = features(t)
    Hx = features(X_COL)
    M = np.column_stack([Hx - LAM * X_COL[:, None] * (w * t) @ Ht,
                         1.0 - LAM * X_COL * np.sum(w * t)])
    coef = np.linalg.solve(M, np.ones(3))
    xx = np.linspace(0.0, 1.0, 1001)
    uu = features(xx) @ coef[:2] + coef[2]
    return {"coef": coef, "max_error": float(np.max(np.abs(uu - exact(xx)))),
            "max_dev_from_nystrom1": float(np.max(np.abs(uu - (1.0 + 2.0 * xx / 7.0)))),
            "u_at_1": float(uu[-1]), "cond": float(np.linalg.cond(M))}


def torch_check():
    torch.set_default_dtype(torch.float64)
    w = torch.tensor(W, requires_grad=True)
    b = torch.tensor(B, requires_grad=True)
    v = torch.tensor(V, requires_grad=True)
    c = torch.tensor(C, requires_grad=True)
    t, wq = (torch.tensor(a) for a in RULES[2])
    x = torch.tensor(X_COL)
    net = lambda s: torch.tanh(w * (2 * s[:, None] - 1) + b) @ v + c
    S = (wq * t * net(t)).sum()
    R = net(x) - 1 - LAM * x * S
    L = (R ** 2).mean()
    g = torch.autograd.grad(L, [w, b, v, c])
    return R.detach().numpy(), L.item(), [a.detach().numpy() for a in g]


def tolist(d):
    if isinstance(d, dict):
        return {str(k): tolist(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [tolist(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def main():
    ex = Example("ch07_integral_equations", "manual_nystrom",
                 description="Hand calculation: Nystrom with one and two Gauss nodes, PINN quadrature residual",
                 default_seed=0)
    np.set_printoptions(precision=6)
    ny = {q: nystrom(q) for q in (1, 2)}
    for q, d in ny.items():
        print(f"Nystrom Q={q}: nodes {d['nodes']} weights {d['weights']}\n matrix\n{d['matrix']}\n"
              f" u(nodes) {d['u_nodes']}  S {d['S']:.6f}  slope {d['slope']:.6f}  max error {d['max_error']:.6e}")
    u_nodes, S, u_col, R = pinn_residual(2, V, C)
    L = float(np.mean(R ** 2))
    dRdc = 1.0 - LAM * X_COL * np.sum(RULES[2][1] * RULES[2][0])
    dLdc = 2.0 / 3.0 * float(R @ dRdc)
    Rt, Lt, gt = torch_check()
    print("PINN: features at Gauss nodes\n", features(RULES[2][0]))
    print("u_theta(Gauss nodes)", u_nodes, " S =", S)
    print("features at x_i\n", features(X_COL))
    print("u_theta(x_i)", u_col)
    print("R(x_i) [hand | torch]", R, Rt)
    print("L [hand | torch]", L, Lt)
    print("dR/dc", dRdc, " dL/dc [hand | torch]", dLdc, gt[3])
    print("torch gradient w, b, v, c:", gt)
    sol = {q: solve_output(q) for q in (1, 2)}
    for q, d in sol.items():
        print(f"output weights with zero residual, Q={q}: {d['coef']}  cond {d['cond']:.3g}  "
              f"max |u - exact| {d['max_error']:.3e}  max |u - (1+2x/7)| {d['max_dev_from_nystrom1']:.3e}  u(1) {d['u_at_1']:.6f}")
    diffs = {"R": float(np.max(np.abs(R - Rt))), "L": abs(L - Lt), "dLdc": abs(dLdc - float(gt[3]))}
    print("checks:", {k: f"{v:.1e}" for k, v in diffs.items()})
    assert max(diffs.values()) < 1e-12
    ex.save_results({"problem": "u(x) = 1 + 0.5 int_0^1 x t u(t) dt; exact 1 + 0.3x",
                     "nystrom": tolist(ny),
                     "pinn": {"parameters": {"w": W, "b": B, "v": V, "c": C}, "x": X_COL,
                              "u_nodes": u_nodes, "S": S, "u_col": u_col, "R": R, "L": L,
                              "dRdc": dRdc, "dLdc": dLdc, "torch_grad": tolist(gt)},
                     "output_solve": tolist(sol), "checks": diffs})


if __name__ == "__main__":
    main()
