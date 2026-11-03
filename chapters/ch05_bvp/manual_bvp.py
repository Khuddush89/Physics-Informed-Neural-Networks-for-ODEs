"""A boundary value problem computed by hand (all numbers of the section in Chapter 5).

Book: Chapter 5 (Boundary value problems), Section "A Boundary Value Problem Computed by Hand".

Problem   y'' = f(t) = -pi^2 sin(pi t) on (0, 1),  y(0) = y(1) = 0;  exact solution sin(pi t).
Trial     y_theta(t) = t (1 - t) N(t)   (both Dirichlet conditions hold for every network)
Network   N(t) = v_1 tanh(w_1 xi + b_1) + v_2 tanh(w_2 xi + b_2) + c,  xi = 2t - 1
Values    w = (0.5, 0.5), b = (-0.5, 0.5), v = (0.5, 0.6), c = 0.1
Points    the three interior Chebyshev-Gauss-Lobatto points t_i = (1 - cos(i pi/4))/2
Loss      L = (1/3) sum_i R(t_i)^2,  R = y_theta'' - f

The script computes
  1. N, N', N'' and y_theta'' at the collocation points by the product and chain rules,
     the residuals and the loss, and checks them against PyTorch (automatic differentiation);
  2. the gradient of L with respect to the output weights (v_1, v_2, c), by hand and by autograd;
  3. the output weights that make all three residuals vanish (a 3 x 3 linear system, because the
     residual is affine in v_1, v_2, c when w, b are fixed) and the resulting error;
  4. the second-order finite-difference solution with the same number of unknowns (h = 1/4) and
     with h = 1/8, for comparison.
All values quoted in the book are taken from results.json.

Run:  python chapters/ch05_bvp/manual_bvp.py      (well under 1 s; --quick is identical)
"""
from __future__ import annotations

import numpy as np
import torch

from pinnode import Example

W = np.array([0.5, 0.5])
B = np.array([-0.5, 0.5])
V0 = np.array([0.5, 0.6])
C0 = 0.1
T_COL = np.array([(1.0 - np.cos(i * np.pi / 4.0)) / 2.0 for i in (1, 2, 3)])
NAMES_OUT = ["v1", "v2", "c"]


def f(t):
    return -np.pi ** 2 * np.sin(np.pi * t)


def features(t):
    """z, h = tanh z, s = 1 - h^2 for xi = 2t - 1; shape (len(t), 2)."""
    t = np.atleast_1d(t)[:, None]
    z = W * (2.0 * t - 1.0) + B
    h = np.tanh(z)
    return z, h, 1.0 - h ** 2


def phi(t):
    """Coefficient of v_k in y_theta''(t):  -2 h_k + 4 (1-2t) w_k s_k - 8 t(1-t) w_k^2 h_k s_k."""
    t1 = np.atleast_1d(t)
    _, h, s = features(t1)
    tc = t1[:, None]
    return -2.0 * h + 4.0 * (1.0 - 2.0 * tc) * W * s - 8.0 * tc * (1.0 - tc) * W ** 2 * h * s


def network(t, v, c):
    """N, N', N'' and y_theta, y_theta'' (hand formulas)."""
    t1 = np.atleast_1d(t)
    _, h, s = features(t1)
    N = h @ v + c
    dN = (2.0 * W * s) @ v                       # dz/dt = 2 w
    d2N = ((2.0 * W) ** 2 * (-2.0 * h * s)) @ v  # d s/dz = -2 h s
    q, dq, d2q = t1 * (1.0 - t1), 1.0 - 2.0 * t1, -2.0
    y = q * N
    d2y = d2q * N + 2.0 * dq * dN + q * d2N
    return N, dN, d2N, y, d2y


def hand_calculation():
    out = {}
    z, h, s = features(T_COL)
    N, dN, d2N, y, d2y = network(T_COL, V0, C0)
    R = d2y - f(T_COL)
    L = float(np.mean(R ** 2))
    out.update(t=T_COL, xi=2 * T_COL - 1, z=z, h=h, s=s, N=N, dN=dN, d2N=d2N, y=y, d2y=d2y,
               f=f(T_COL), R=R, L=L, exact=np.sin(np.pi * T_COL))
    # gradient with respect to the output layer: R = v.phi - 2c - f
    A = np.column_stack([phi(T_COL), -2.0 * np.ones(3)])
    grad_out = 2.0 / 3.0 * A.T @ R
    out.update(A=A, grad_out=grad_out, check_affine=float(np.max(np.abs(A @ np.r_[V0, C0] - f(T_COL) - R))))
    # output weights with zero residual at the three points
    x = np.linalg.solve(A, f(T_COL))
    tt = np.linspace(0.0, 1.0, 1001)
    _, _, _, y_fine, d2y_fine = network(tt, x[:2], x[2])
    err = np.abs(y_fine - np.sin(np.pi * tt))
    res_fine = d2y_fine - f(tt)
    out.update(x_solve=x, cond_A=float(np.linalg.cond(A)), max_error_solve=float(err.max()),
               argmax_error_solve=float(tt[err.argmax()]),
               y_half_solve=float(network(0.5, x[:2], x[2])[3][0]),
               residual_fine_rms=float(np.sqrt(np.mean(res_fine ** 2))),
               residual_fine_max=float(np.abs(res_fine).max()))
    # finite differences with the same number of unknowns (h = 1/4) and h = 1/8
    fd = {}
    for n in (4, 8):
        hgrid = 1.0 / n
        tg = np.linspace(0.0, 1.0, n + 1)[1:-1]
        K = (np.diag(-2.0 * np.ones(n - 1)) + np.diag(np.ones(n - 2), 1) + np.diag(np.ones(n - 2), -1)) / hgrid ** 2
        yfd = np.linalg.solve(K, f(tg))
        fd[n] = {"t": tg, "y": yfd, "error": yfd - np.sin(np.pi * tg),
                 "max_error": float(np.max(np.abs(yfd - np.sin(np.pi * tg)))),
                 "factor": float(np.pi ** 2 * hgrid ** 2 / (4.0 * np.sin(np.pi * hgrid / 2.0) ** 2))}
    out["fd"] = fd
    return out


def torch_calculation():
    torch.set_default_dtype(torch.float64)
    w = torch.tensor(W, requires_grad=True)
    b = torch.tensor(B, requires_grad=True)
    v = torch.tensor(V0, requires_grad=True)
    c = torch.tensor(C0, requires_grad=True)
    t = torch.tensor(T_COL, requires_grad=True)
    N = torch.tanh(w * (2 * t[:, None] - 1) + b) @ v + c
    y = t * (1 - t) * N
    dy, = torch.autograd.grad(y.sum(), t, create_graph=True)
    d2y, = torch.autograd.grad(dy.sum(), t, create_graph=True)
    R = d2y + np.pi ** 2 * torch.sin(np.pi * t)
    L = (R ** 2).mean()
    gw, gb, gv, gc = torch.autograd.grad(L, [w, b, v, c])
    return {"d2y": d2y.detach().numpy(), "R": R.detach().numpy(), "L": L.item(),
            "grad_w": gw.numpy(), "grad_b": gb.numpy(), "grad_v": gv.numpy(), "grad_c": gc.item()}


def tolist(d):
    if isinstance(d, dict):
        return {str(k): tolist(v) for k, v in d.items()}
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def main():
    ex = Example("ch05_bvp", "manual_bvp",
                 description="Hand calculation for a Dirichlet BVP: residuals, collocation system, finite differences",
                 default_seed=0)
    hand = hand_calculation()
    ad = torch_calculation()
    np.set_printoptions(precision=6, suppress=False)
    print("collocation points t_i      ", hand["t"])
    print("xi = 2t - 1                 ", hand["xi"])
    print("z (rows = points)\n", hand["z"])
    print("h = tanh z\n", hand["h"])
    print("s = 1 - h^2\n", hand["s"])
    print("N, N', N''                  ", hand["N"], hand["dN"], hand["d2N"])
    print("y_theta(t_i)                ", hand["y"])
    print("y_theta''(t_i) [hand]       ", hand["d2y"])
    print("y_theta''(t_i) [autograd]   ", ad["d2y"])
    print("f(t_i)                      ", hand["f"])
    print("R(t_i)                      ", hand["R"])
    print("L = mean R^2  [hand | torch] ", hand["L"], ad["L"])
    print("matrix A = [phi_1 phi_2 -2]\n", hand["A"])
    print("grad wrt (v1, v2, c) [hand]  ", hand["grad_out"])
    print("grad wrt (v1, v2, c) [torch] ", ad["grad_v"], ad["grad_c"])
    print("grad wrt w, b [torch]        ", ad["grad_w"], ad["grad_b"])
    print("solve A x = f: x            ", hand["x_solve"], " cond(A) =", hand["cond_A"])
    print("max error on 1001 points    ", hand["max_error_solve"], "at t =", hand["argmax_error_solve"])
    print("y(0.5) with solved weights  ", hand["y_half_solve"])
    print("fine-grid residual rms, max ", hand["residual_fine_rms"], hand["residual_fine_max"])
    for n, d in hand["fd"].items():
        print(f"FD h=1/{n}: y =", d["y"], " max error", d["max_error"], " factor", d["factor"])

    diffs = {"d2y": float(np.max(np.abs(hand["d2y"] - ad["d2y"]))),
             "loss": abs(hand["L"] - ad["L"]),
             "grad_out": float(np.max(np.abs(hand["grad_out"] - np.r_[ad["grad_v"], ad["grad_c"]]))),
             "affine": hand["check_affine"]}
    print("max |hand - torch|:", {k: f"{v:.1e}" for k, v in diffs.items()})
    assert max(diffs.values()) < 1e-12, "hand calculation and autograd disagree"
    ex.save_results({"problem": "y'' = -pi^2 sin(pi t), y(0)=y(1)=0; trial t(1-t)N(t), xi = 2t-1",
                     "parameters": {"w": W, "b": B, "v": V0, "c": C0},
                     "hand": tolist(hand), "torch": tolist(ad),
                     "max_abs_difference_hand_vs_torch": diffs})


if __name__ == "__main__":
    main()
