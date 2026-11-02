"""Shared definitions for the Chapter 3 experiments: the Airy oscillator benchmark.

Equation      y''(t) + t y(t) = 0,   t in [0, T],  T = 10.
Exact         y(t) = Ai(-t)   (Ai = Airy function of the first kind; y' = -Ai'(-t)).

The local angular frequency is sqrt(t) (WKB), so the solution is a chirp with six zeros in
(0, 10) (three in (0, 6)) whose frequency grows and whose amplitude decays slowly. Three
problems are built from it:

* Dirichlet BVP   y(0) = Ai(0),  y(T) = Ai(-T)                         (architecture study)
* IVP             y(0) = Ai(0),  y'(0) = -Ai'(0)  on [0, T_IVP]        (soft vs hard)
* Robin BVP       y(0) = Ai(0),  y'(T) + y(T) = -Ai'(-T) + Ai(-T)      (soft vs hard)

All three have the unique solution Ai(-t) (see ``well_posedness``).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import airy

from pinnode import derivative

T_END = 10.0
T_IVP = 6.0                # shorter interval for the IVP (see soft_vs_hard.py)
N_COLLOCATION = 200        # equispaced collocation points on [0, T]
N_TEST = 1001              # equispaced test points on [0, T]

AI0, AIP0, BI0, BIP0 = (float(v) for v in airy(0.0))
AIT, AIPT, BIT, BIPT = (float(v) for v in airy(-T_END))

Y_A = AI0                  # y(0)
V_A = -AIP0                # y'(0)
Y_B = AIT                  # y(T)
V_B = -AIPT                # y'(T)
ROBIN_BETA1, ROBIN_BETA2 = 1.0, 1.0
ROBIN_GAMMA = ROBIN_BETA1 * Y_B + ROBIN_BETA2 * V_B


def exact(t):
    """y(t) = Ai(-t)."""
    return airy(-np.asarray(t, dtype=float))[0]


def exact_derivative(t):
    """y'(t) = -Ai'(-t)."""
    return -airy(-np.asarray(t, dtype=float))[1]


def residual(model, t):
    """Residual y'' + t y of a model at collocation points t (requires_grad=True)."""
    y = model(t)
    return derivative(y, t, 2) + t * y


def well_posedness(n_cheb: int = 120) -> dict:
    """Indicators that the BVPs are uniquely solvable and how well conditioned they are.

    phi(t) = Bi(0) Ai(-t) - Ai(0) Bi(-t) spans the solutions with phi(0) = 0; the Dirichlet
    (Robin) problem is uniquely solvable iff phi(T) != 0 (beta1 phi(T) + beta2 phi'(T) != 0).
    The smallest |eigenvalue| of v -> v'' + t v with homogeneous Dirichlet conditions,
    computed by Chebyshev collocation, bounds the amplification of residual errors.
    """
    phi_T = BI0 * AIT - AI0 * BIT
    dphi_T = -(BI0 * AIPT - AI0 * BIPT)
    k = np.arange(n_cheb + 1)
    x = np.cos(np.pi * k / n_cheb)
    c = np.where((k == 0) | (k == n_cheb), 2.0, 1.0) * (-1.0) ** k
    X = np.tile(x, (n_cheb + 1, 1)).T
    D = np.outer(c, 1.0 / c) / (X - X.T + np.eye(n_cheb + 1))
    D -= np.diag(D.sum(axis=1))
    D *= 2.0 / T_END                      # d/dt on [0, T]
    t = (x + 1.0) * T_END / 2.0
    A = (D @ D + np.diag(t))[1:-1, 1:-1]
    eig = np.linalg.eigvals(A)
    # Robin condition beta1 v(T) + beta2 v'(T) = 0 at t = T (row 0) and v(0) = 0 (row n):
    # generalized eigenproblem with the boundary rows removed from the mass matrix
    from scipy.linalg import eig as geig
    Ar = D @ D + np.diag(t)
    Br = np.eye(n_cheb + 1)
    Ar[0, :] = ROBIN_BETA2 * D[0, :] + ROBIN_BETA1 * np.eye(n_cheb + 1)[0]
    Br[0, :] = 0.0
    Ar[-1, :] = np.eye(n_cheb + 1)[-1]
    Br[-1, :] = 0.0
    wr = geig(Ar, Br, right=False)
    wr = wr[np.isfinite(wr)]
    return {"phi_T": phi_T,
            "robin_phi": ROBIN_BETA1 * phi_T + ROBIN_BETA2 * dphi_T,
            "min_abs_eig_dirichlet": float(np.min(np.abs(eig))),
            "min_abs_eig_robin": float(np.min(np.abs(wr)))}


# ----------------------------------------------------------------------------- output transforms
def dirichlet_normalized(ya: float = Y_A, yb: float = Y_B, a: float = 0.0, b: float = T_END):
    """y = ya (b-t)/(b-a) + yb (t-a)/(b-a) + (1 - xi^2) N,  xi = (2t-a-b)/(b-a)."""
    def transform(t, n):
        xi = (2.0 * t - a - b) / (b - a)
        return ya * (b - t) / (b - a) + yb * (t - a) / (b - a) + (1.0 - xi ** 2) * n
    return transform


def ivp_normalized(y0: float = Y_A, v0: float = V_A, t0: float = 0.0, tau: float = T_END):
    """y = y0 + v0 (t-t0) + ((t-t0)/tau)^2 N  (tau = 1 reproduces pinnode.ivp_transform)."""
    def transform(t, n):
        return y0 + v0 * (t - t0) + ((t - t0) / tau) ** 2 * n
    return transform


class DirichletRobinModel(torch.nn.Module):
    """Constrained expression for y(a) = ya and beta1 y(b) + beta2 y'(b) = gamma.

    y(t) = N(t) + eta1(t) (ya - N(a)) + eta2(t) (gamma - beta1 N(b) - beta2 N'(b)),
    eta1 = 1 - beta1 (t-a)/(beta1 L + beta2),  eta2 = (t-a)/(beta1 L + beta2),  L = b - a.
    """

    def __init__(self, net, a=0.0, b=T_END, ya=Y_A, beta1=ROBIN_BETA1, beta2=ROBIN_BETA2,
                 gamma=ROBIN_GAMMA):
        super().__init__()
        self.net = net
        self.a, self.b, self.ya = a, b, ya
        self.beta1, self.beta2, self.gamma = beta1, beta2, gamma
        self.den = beta1 * (b - a) + beta2
        if abs(self.den) < 1e-12:
            raise ValueError("beta1 (b - a) + beta2 must not vanish")

    def forward(self, t):
        n = self.net(t)
        with torch.enable_grad():
            ta = torch.full((1, 1), self.a, dtype=t.dtype)
            tb = torch.full((1, 1), self.b, dtype=t.dtype, requires_grad=True)
            na = self.net(ta)
            nb = self.net(tb)
            dnb = derivative(nb, tb, 1)
        eta1 = 1.0 - self.beta1 * (t - self.a) / self.den
        eta2 = (t - self.a) / self.den
        return n + eta1 * (self.ya - na) + eta2 * (self.gamma - self.beta1 * nb - self.beta2 * dnb)


def test_grid(T: float = T_END):
    t = np.linspace(0.0, T, N_TEST)
    return t, exact(t)


def collocation(T: float = T_END):
    return np.linspace(0.0, T, N_COLLOCATION)


def load_history(path, switch_iter=None):
    """Rebuild a pinnode History (iterations, loss, relative error) from a saved history CSV.

    Used by the ``--plot-only`` option of the chapter scripts to redraw figures without
    retraining.
    """
    import pandas as pd

    from pinnode import History

    df = pd.read_csv(path)
    h = History()
    h.iters = df["iter"].astype(int).tolist()
    h.phase = df["phase"].tolist()
    h.loss = df["loss"].astype(float).tolist()
    h.time = df["time_s"].astype(float).tolist()
    if "rel_l2" in df:
        m = df.dropna(subset=["rel_l2"])
        h.eval_iters = m["iter"].astype(int).tolist()
        h.metrics = {"rel_l2": m["rel_l2"].astype(float).tolist()}
    h.switch_iter = switch_iter
    return h
