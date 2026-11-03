"""A Volterra integro-differential equation computed by hand (Chapter 8).

Book: Chapter 8 (Integro-differential equations), Section "An Integro-Differential Equation
Computed by Hand".

Problem   u'(x) = u(x) + int_0^x exp(-(x - t)) u(t) dt,  u(0) = 1  (the convolution example);
          exact solution u = cosh(sqrt2 x) + sinh(sqrt2 x)/sqrt2.
Classical (a) the implicit trapezoidal direct quadrature method with h = 1/2 (two steps);
          (b) the internal variable z(x) = int_0^x exp(-(x-t)) u(t) dt, z' = u - z, z(0) = 0,
              and the trapezoidal rule for the system (u, z)' = A (u, z), h = 1/2.
PINN      u_theta(x) = 1 + x N(x), N as in Chapters 5-7 (w = (0.5, 0.5), b = (-0.5, 0.5),
          v = (0.5, 0.6), c = 0.1, xi = 2x - 1); residual at x_i = 1/2 and x_i = 1 with the
          two-point Gauss rule on [0, x_i]:  R(x_i) = u_theta'(x_i) - u_theta(x_i)
          - sum_j w_j exp(-(x_i - t_j)) u_theta(t_j).
Each collocation point needs its own quadrature nodes, i.e. extra network evaluations.

Run:  python chapters/ch08_integro_differential/manual_ide.py   (well under 1 s; --quick is identical)
"""
from __future__ import annotations

import numpy as np
import torch

from pinnode import Example

W = np.array([0.5, 0.5])
B = np.array([-0.5, 0.5])
V = np.array([0.5, 0.6])
C = 0.1
H = 0.5
R2 = np.sqrt(2.0)


def exact(x):
    return np.cosh(R2 * x) + np.sinh(R2 * x) / R2


def exact_z(x):
    # z = u' - u
    du = R2 * np.sinh(R2 * x) + np.cosh(R2 * x)
    return du - exact(x)


def trapezoidal_direct(n_steps=2, H=H):
    """u_{n+1} = u_n + h/2 (G_n + G_{n+1}),  G_n = u_n + h sum''_j K(x_n, x_j) u_j."""
    x = H * np.arange(n_steps + 1)
    u = np.zeros(n_steps + 1)
    u[0] = 1.0
    G = np.zeros(n_steps + 1)
    G[0] = u[0]
    steps = []
    for n in range(n_steps):
        m = n + 1
        K = np.exp(-(x[m] - x[: m + 1]))
        w = np.full(m + 1, H)
        w[0] = w[-1] = H / 2.0
        known = float(np.sum(w[:-1] * K[:-1] * u[:m]))      # memory term without the new value
        coef = 1.0 + w[-1] * K[-1]                           # G_m = coef * u_m + known
        # u_m = u_n + h/2 (G_n + coef u_m + known)
        u[m] = (u[n] + H / 2.0 * (G[n] + known)) / (1.0 - H / 2.0 * coef)
        G[m] = coef * u[m] + known
        steps.append({"x": x[m], "memory_known": known, "G_coef": coef, "u": u[m], "G": G[m]})
    return x, u, G, steps


def trapezoidal_internal(n_steps=2, H=H):
    A = np.array([[1.0, 1.0], [1.0, -1.0]])
    M1 = np.eye(2) - H / 2.0 * A
    M2 = np.eye(2) + H / 2.0 * A
    U = [np.array([1.0, 0.0])]
    for _ in range(n_steps):
        U.append(np.linalg.solve(M1, M2 @ U[-1]))
    return M1, M2, np.array(U)


def convergence(Ns=(2, 4, 8, 16, 32)):
    """Errors at x = 1 of both trapezoidal methods for h = 1/N and the observed orders."""
    rows = []
    for N in Ns:
        e_dir = abs(trapezoidal_direct(N, 1.0 / N)[1][-1] - exact(1.0))
        e_int = abs(trapezoidal_internal(N, 1.0 / N)[2][-1, 0] - exact(1.0))
        rows.append({"N": N, "error_direct": e_dir, "error_internal": e_int})
    for a, b in zip(rows[:-1], rows[1:]):
        b["order_direct"] = float(np.log2(a["error_direct"] / b["error_direct"]))
        b["order_internal"] = float(np.log2(a["error_internal"] / b["error_internal"]))
    return rows


def features(x):
    x = np.atleast_1d(x)[:, None]
    z = W * (2.0 * x - 1.0) + B
    h = np.tanh(z)
    return h, 1.0 - h ** 2


def u_theta(x):
    x = np.atleast_1d(x)
    h, s = features(x)
    N = h @ V + C
    dN = (2.0 * W * s) @ V
    return 1.0 + x * N, N + x * dN, N


def pinn_residual(xi):
    t = xi / 2.0 + np.array([-1.0, 1.0]) * xi / (2.0 * np.sqrt(3.0))
    wq = np.array([xi / 2.0, xi / 2.0])
    u_nodes = u_theta(t)[0]
    K = np.exp(-(xi - t))
    I = float(np.sum(wq * K * u_nodes))
    u_x, du_x, N_x = (a[0] for a in u_theta(xi))
    R = du_x - u_x - I
    return {"x": xi, "nodes": t, "weights": wq, "kernel": K, "u_nodes": u_nodes, "integral": I,
            "u": u_x, "du": du_x, "N": N_x, "R": R}


def torch_check():
    torch.set_default_dtype(torch.float64)
    w = torch.tensor(W, requires_grad=True)
    b = torch.tensor(B, requires_grad=True)
    v = torch.tensor(V, requires_grad=True)
    c = torch.tensor(C, requires_grad=True)

    def u(x):
        return 1 + x * (torch.tanh(w * (2 * x[:, None] - 1) + b) @ v + c)

    Rs = []
    for xi in (0.5, 1.0):
        x = torch.tensor([xi], requires_grad=True)
        ux = u(x)
        du, = torch.autograd.grad(ux.sum(), x, create_graph=True)
        t = torch.tensor(xi / 2.0 + np.array([-1.0, 1.0]) * xi / (2.0 * np.sqrt(3.0)))
        I = (xi / 2.0 * torch.exp(-(xi - t)) * u(t)).sum()
        Rs.append(du[0] - ux[0] - I)
    R = torch.stack(Rs)
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
    if isinstance(d, (np.floating,)):
        return float(d)
    return d


def main():
    ex = Example("ch08_integro_differential", "manual_ide",
                 description="Hand calculation: trapezoidal methods and PINN residual for a convolution IDE",
                 default_seed=0)
    np.set_printoptions(precision=6)
    x, u_dir, G, steps = trapezoidal_direct()
    M1, M2, U = trapezoidal_internal()
    ex_u = exact(x)
    print("exact u(0), u(1/2), u(1):", ex_u, " exact z:", exact_z(x))
    print("direct trapezoidal: u =", u_dir, " G =", G, " errors", u_dir - ex_u)
    for st in steps:
        print("   step", st)
    print("internal variable trapezoidal: I - hA/2 =\n", M1, "\n I + hA/2 =\n", M2, "\n U =\n", U,
          "\n errors u:", U[:, 0] - ex_u, " errors z:", U[:, 1] - exact_z(x))
    conv = convergence()
    for r in conv:
        print("convergence:", r)
    pr = [pinn_residual(xi) for xi in (0.5, 1.0)]
    for d in pr:
        print("PINN at x =", d["x"], {k: d[k] for k in ("nodes", "weights", "kernel", "u_nodes", "integral", "u", "du", "N", "R")})
    R = np.array([d["R"] for d in pr])
    L = float(np.mean(R ** 2))
    Rt, Lt, gt = torch_check()
    print("R [hand | torch]", R, Rt, " L [hand | torch]", L, Lt)
    print("torch gradient w, b, v, c:", gt)
    diffs = {"R": float(np.max(np.abs(R - Rt))), "L": abs(L - Lt)}
    print("checks:", {k: f"{v:.1e}" for k, v in diffs.items()})
    assert max(diffs.values()) < 1e-12
    ex.save_results({"problem": "u' = u + int_0^x exp(-(x-t)) u dt, u(0) = 1",
                     "x": x, "exact_u": ex_u, "exact_z": exact_z(x),
                     "direct": {"u": u_dir, "G": G, "steps": tolist(steps), "error": u_dir - ex_u},
                     "internal": {"I_minus": M1, "I_plus": M2, "U": U, "error_u": U[:, 0] - ex_u,
                                  "error_z": U[:, 1] - exact_z(x)},
                     "convergence": conv,
                     "pinn": tolist(pr), "L": L, "torch": {"R": Rt, "L": Lt, "grad": tolist(gt)},
                     "checks": diffs})


if __name__ == "__main__":
    main()
