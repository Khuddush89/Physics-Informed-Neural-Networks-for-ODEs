"""Shared helpers for the Chapter 5 examples (boundary value problems).

Not an example itself (the leading underscore keeps it out of ``scripts/run_all.py`` and
the smoke tests). Contents:

* closed-form solutions of the Duffing-type BVP  y'' + y + eps y^3 = 0,  y(0) = 1,
  y'(pi) = 0  in terms of the Jacobi elliptic function cn;
* the exact solution of the boundary-layer problem  eps y'' + y' = 1,  y(0) = 0,
  y(1) + y'(1) = 1;
* layer-adapted (Shishkin) collocation points;
* a timing helper that reports both wall-clock and CPU time (medians of repeats).
"""
from __future__ import annotations

import time
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipj

# ----------------------------------------------------------------------------- Duffing BVP
EPS_DUFFING = 0.1      # coefficient of the cubic term
T_END = np.pi          # right endpoint of [0, T_END]
Y0 = 1.0               # y(0) = Y0
DY_END = 0.0           # y'(T_END) = DY_END


def duffing_params(A, eps: float = EPS_DUFFING):
    """Frequency w and elliptic parameter m = k^2 of y = A cn(w (t - T), m).

    y = A cn(w s, m) solves y'' + y + eps y^3 = 0 iff  w^2 = 1 + eps A^2  and
    m = eps A^2 / (2 w^2).
    """
    A = np.asarray(A, dtype=float)
    w = np.sqrt(1.0 + eps * A ** 2)
    m = eps * A ** 2 / (2.0 * (1.0 + eps * A ** 2))
    return w, m


def duffing_bc_function(A, eps: float = EPS_DUFFING, T: float = T_END, y0: float = Y0):
    """F(A) = A cn(w(A) T, m(A)) - y0; its roots A = y(T) are the solutions of the BVP."""
    w, m = duffing_params(A, eps)
    return np.asarray(A) * ellipj(w * T, m)[1] - y0


def duffing_roots(A_max: float = 12.0, n_grid: int = 240001, eps: float = EPS_DUFFING,
                  T: float = T_END):
    """All roots of F in [-A_max, A_max] (sign changes on a fine grid, then Brent), by |A|."""
    As = np.linspace(-A_max, A_max, n_grid)
    F = duffing_bc_function(As, eps, T)
    idx = np.where(np.sign(F[:-1]) * np.sign(F[1:]) < 0)[0]
    f = lambda a: float(duffing_bc_function(a, eps, T))
    roots = [brentq(f, As[i], As[i + 1], xtol=1e-15, rtol=1e-15, maxiter=200) for i in idx]
    return sorted(roots, key=abs)


def duffing_exact(t, A, eps: float = EPS_DUFFING, T: float = T_END):
    """y(t) = A cn(w (t - T), m)  (satisfies y'(T) = 0 for every A)."""
    w, m = duffing_params(A, eps)
    return A * ellipj(w * (np.asarray(t, float) - T), m)[1]


def duffing_exact_dt(t, A, eps: float = EPS_DUFFING, T: float = T_END):
    """y'(t) = -A w sn dn."""
    w, m = duffing_params(A, eps)
    sn, cn, dn, _ = ellipj(w * (np.asarray(t, float) - T), m)
    return -A * w * sn * dn


def duffing_A1(eps: float = EPS_DUFFING, T: float = T_END) -> float:
    """The root of smallest modulus (the solution close to cos t)."""
    f = lambda a: float(duffing_bc_function(a, eps, T))
    return brentq(f, -1.2, -0.9, xtol=1e-15, rtol=1e-15, maxiter=200)


def duffing_rhs_system(t, z, eps: float = EPS_DUFFING):
    """First-order form z = (y, y') for scipy (vectorised in the solve_bvp sense)."""
    return np.vstack((z[1], -z[0] - eps * z[0] ** 3))


def duffing_bc_residual(za, zb):
    return np.array([za[0] - Y0, zb[1] - DY_END])


# ----------------------------------------------------------------------------- boundary layer
def layer_constant(eps: float) -> float:
    return 1.0 / (1.0 - (1.0 - 1.0 / eps) * np.exp(-1.0 / eps))


def layer_exact(t, eps: float):
    """Exact solution of eps y'' + y' = 1, y(0) = 0, y(1) + y'(1) = 1."""
    t = np.asarray(t, float)
    return t - layer_constant(eps) * (1.0 - np.exp(-t / eps))


def shishkin_points(M: int, eps: float, sigma: float = 2.0, T: float = 1.0) -> np.ndarray:
    """Piecewise-uniform (Shishkin) points: M//2 intervals in [0, tau], M - M//2 - 1 in [tau, T],
    tau = min(T/2, sigma * eps * ln M); M points in total, endpoints included."""
    tau = min(0.5 * T, sigma * eps * np.log(M))
    n1 = M // 2
    left = np.linspace(0.0, tau, n1 + 1)
    right = np.linspace(tau, T, M - n1)[1:]
    return np.concatenate([left, right])


# ----------------------------------------------------------------------------- re-plotting
def load_history(path, switch_iter=None):
    """Rebuild a pinnode History from a CSV written by ``Example.save_history``."""
    import pandas as pd

    from pinnode import History

    df = pd.read_csv(path)
    h = History()
    h.iters = df["iter"].astype(int).tolist()
    h.phase = df["phase"].tolist()
    h.loss = df["loss"].tolist()
    known = {"iter", "phase", "loss", "lr", "time_s", "rel_l2"}
    for c in df.columns:
        if c not in known:
            h.parts[c] = df[c].tolist()
    if "rel_l2" in df.columns:
        m = df.dropna(subset=["rel_l2"])
        h.eval_iters = m["iter"].astype(int).tolist()
        h.metrics["rel_l2"] = m["rel_l2"].tolist()
    h.switch_iter = switch_iter
    return h


# ----------------------------------------------------------------------------- timing
def time_call(fn: Callable, min_repeats: int = 5, min_total: float = 0.3,
              max_repeats: int = 200) -> Tuple[object, float, float, int]:
    """Run ``fn()`` repeatedly; return (last result, median wall s, median CPU s, repeats)."""
    walls, cpus, out = [], [], None
    while len(walls) < max_repeats and (len(walls) < min_repeats or sum(walls) < min_total):
        w0, c0 = time.perf_counter(), time.process_time()
        out = fn()
        cpus.append(time.process_time() - c0)
        walls.append(time.perf_counter() - w0)
    return out, float(np.median(walls)), float(np.median(cpus)), len(walls)
