"""A terminal value problem computed by hand (all numbers of the section in Chapter 6).

Book: Chapter 6 (Terminal value problems), Section "A Terminal Value Problem Computed by Hand".

Problem   y'' + y = 0 on [0, 1],  y(1) = cos 1,  y'(1) = -sin 1;  exact solution cos t.
Trial     y_theta(t) = c_0 + c_1 (t - 1) + (1 - t)^2 N(t),  c_0 = cos 1, c_1 = -sin 1
Network   N(t) = v_1 tanh(w_1 xi + b_1) + v_2 tanh(w_2 xi + b_2) + c,  xi = 2t - 1
Values    w = (0.5, 0.5), b = (-0.5, 0.5), v = (0.5, 0.6), c = 0.1  (as in Chapter 5)
Points    the Legendre-Gauss-Lobatto points 0, 1/2, 1 (a symmetric set)
Loss      L = (1/3) sum_i R(t_i)^2,  R = y_theta'' + y_theta

The script computes the trial function and its derivatives by hand, the residuals and the loss,
checks them with PyTorch, and verifies Proposition "A TVP-PINN is a reflected IVP-PINN": with
s = 1 - t, the reversed IVP Y'' + Y = 0, Y(0) = c_0, Y'(0) = -c_1, the trial function
Y(s) = c_0 - c_1 s + s^2 N_tilde(s) and the network N_tilde with negated first-layer weights has
the same residuals at the mirrored points and hence the same loss, while the loss gradient with
respect to w changes sign. It also evaluates the backward map z(0) = exp(-A) z(1) of the
first-order system z = (y, y'), z' = A z.

Run:  python chapters/ch06_tvp/manual_tvp.py      (well under 1 s; --quick is identical)
"""
from __future__ import annotations

import numpy as np
import torch

from pinnode import Example

W = np.array([0.5, 0.5])
B = np.array([-0.5, 0.5])
V = np.array([0.5, 0.6])
C = 0.1
C0, C1 = float(np.cos(1.0)), float(-np.sin(1.0))
T_COL = np.array([0.0, 0.5, 1.0])


def net_derivs(t, w):
    """N, N', N'' at t for first-layer weights w (xi = 2t - 1, so dz/dt = 2w)."""
    t = np.atleast_1d(t)[:, None]
    z = w * (2.0 * t - 1.0) + B
    h = np.tanh(z)
    s = 1.0 - h ** 2
    N = h @ V + C
    dN = (2.0 * w * s) @ V
    d2N = ((2.0 * w) ** 2 * (-2.0 * h * s)) @ V
    return z, h, s, N, dN, d2N


def tvp_trial(t):
    z, h, s, N, dN, d2N = net_derivs(t, W)
    u = 1.0 - t
    y = C0 + C1 * (t - 1.0) + u ** 2 * N
    dy = C1 - 2.0 * u * N + u ** 2 * dN
    d2y = 2.0 * N - 4.0 * u * dN + u ** 2 * d2N
    return dict(z=z, h=h, s=s, N=N, dN=dN, d2N=d2N, y=y, dy=dy, d2y=d2y, R=d2y + y)


def ivp_trial(s_pts):
    """Reversed IVP with N_tilde = network with first-layer weights -w."""
    z, h, s, N, dN, d2N = net_derivs(s_pts, -W)
    Y = C0 - C1 * s_pts + s_pts ** 2 * N
    dY = -C1 + 2.0 * s_pts * N + s_pts ** 2 * dN
    d2Y = 2.0 * N + 4.0 * s_pts * dN + s_pts ** 2 * d2N
    return dict(N=N, Y=Y, dY=dY, d2Y=d2Y, R=d2Y + Y)


def torch_losses():
    torch.set_default_dtype(torch.float64)

    def loss(points, w, sign_c1, kind):
        b = torch.tensor(B, requires_grad=True)
        v = torch.tensor(V, requires_grad=True)
        c = torch.tensor(C, requires_grad=True)
        t = torch.tensor(points, requires_grad=True)
        N = torch.tanh(w * (2 * t[:, None] - 1) + b) @ v + c
        if kind == "tvp":
            y = C0 + C1 * (t - 1) + (1 - t) ** 2 * N
        else:
            y = C0 - C1 * t + t ** 2 * N
        dy, = torch.autograd.grad(y.sum(), t, create_graph=True)
        d2y, = torch.autograd.grad(dy.sum(), t, create_graph=True)
        R = d2y + y
        L = (R ** 2).mean()
        g = torch.autograd.grad(L, [w, b, v, c])
        return y.detach().numpy(), dy.detach().numpy(), R.detach().numpy(), L.item(), [x.detach().numpy() for x in g]

    w = torch.tensor(W, requires_grad=True)
    tvp = loss(T_COL, w, 1, "tvp")
    w2 = torch.tensor(-W, requires_grad=True)
    ivp = loss(1.0 - T_COL, w2, -1, "ivp")
    return tvp, ivp


def tolist(d):
    if isinstance(d, dict):
        return {str(k): tolist(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [tolist(v) for v in d]
    if isinstance(d, np.ndarray):
        return d.tolist()
    return d


def main():
    ex = Example("ch06_tvp", "manual_tvp",
                 description="Hand calculation for a second-order TVP and its time reversal",
                 default_seed=0)
    np.set_printoptions(precision=6)
    tv = tvp_trial(T_COL)
    L = float(np.mean(tv["R"] ** 2))
    iv = ivp_trial(1.0 - T_COL)
    L_ivp = float(np.mean(iv["R"] ** 2))
    (y_t, dy_t, R_t, L_t, g_t), (Y_s, dY_s, R_s, L_s, g_s) = torch_losses()
    A = np.array([[0.0, 1.0], [-1.0, 0.0]])
    expmA = np.array([[np.cos(1.0), -np.sin(1.0)], [np.sin(1.0), np.cos(1.0)]])   # exp(-A), T = 1
    z0 = expmA @ np.array([C0, C1])

    print("c0 = cos 1, c1 = -sin 1      ", C0, C1)
    print("t                            ", T_COL)
    print("z (rows = points)\n", tv["z"])
    print("h\n", tv["h"])
    print("N, N', N''                   ", tv["N"], tv["dN"], tv["d2N"])
    print("y, y', y''                   ", tv["y"], tv["dy"], tv["d2y"])
    print("exact cos t                  ", np.cos(T_COL))
    print("R = y'' + y [hand | torch]   ", tv["R"], R_t)
    print("L [hand | torch]             ", L, L_t)
    print("reversed IVP at s = 1 - t:   N_tilde", iv["N"], " Y", iv["Y"], " Y''", iv["d2Y"])
    print("R_ivp(s) [hand | torch]      ", iv["R"], R_s)
    print("L_ivp [hand | torch]         ", L_ivp, L_s)
    print("grad L wrt w, b, v, c (TVP)  ", g_t)
    print("grad L wrt w~, b, v, c (IVP) ", g_s)
    print("exp(-A) =\n", expmA, "\n z(0) = exp(-A) z(1) =", z0, " ||exp(-A)||_2 =", np.linalg.norm(expmA, 2))

    diffs = {"R_tvp": float(np.max(np.abs(tv["R"] - R_t))), "L_tvp": abs(L - L_t),
             "R_reflection": float(np.max(np.abs(tv["R"] - iv["R"]))), "L_reflection": abs(L - L_ivp),
             "grad_w_sign": float(np.max(np.abs(g_t[0] + g_s[0]))),
             "grad_other": float(max(np.max(np.abs(g_t[k] - g_s[k])) for k in (1, 2, 3))),
             "terminal_values": float(max(abs(tv["y"][-1] - C0), abs(tv["dy"][-1] - C1)))}
    print("checks:", {k: f"{v:.1e}" for k, v in diffs.items()})
    assert max(diffs.values()) < 1e-12
    ex.save_results({"problem": "y'' + y = 0, y(1) = cos 1, y'(1) = -sin 1; trial c0 + c1(t-1) + (1-t)^2 N",
                     "parameters": {"w": W, "b": B, "v": V, "c": C, "c0": C0, "c1": C1},
                     "tvp": tolist(tv), "L": L, "ivp": tolist(iv), "L_ivp": L_ivp,
                     "torch": {"R_tvp": R_t, "L_tvp": L_t, "grad_tvp": tolist(g_t),
                               "R_ivp": R_s, "L_ivp": L_s, "grad_ivp": tolist(g_s)},
                     "backward_map": {"exp_minus_A": expmA, "z0": z0,
                                      "norm": float(np.linalg.norm(expmA, 2))},
                     "checks": diffs})


if __name__ == "__main__":
    main()
