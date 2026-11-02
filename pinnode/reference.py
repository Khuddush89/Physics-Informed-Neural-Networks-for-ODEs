"""High-accuracy classical reference solutions (used for every error figure and table)."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp


def ivp_reference(f: Callable, t_span: Sequence[float], y0: Sequence[float], t_eval: np.ndarray,
                  method: str = "DOP853", rtol: float = 1e-12, atol: float = 1e-12, **kw) -> np.ndarray:
    """Solve y' = f(t, y) with tight tolerances; returns array of shape (len(t_eval), dim).

    Use ``method="Radau"`` (implicit, L-stable) for stiff problems.
    """
    sol = solve_ivp(f, t_span, np.atleast_1d(y0).astype(float), method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol, **kw)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def bvp_reference(fun: Callable, bc: Callable, x_eval: np.ndarray, y_guess: np.ndarray,
                  x_mesh: np.ndarray | None = None, tol: float = 1e-10, max_nodes: int = 200000) -> np.ndarray:
    """Collocation solution (scipy.solve_bvp) evaluated at ``x_eval``; shape (len(x_eval), dim)."""
    x_mesh = np.linspace(x_eval[0], x_eval[-1], 200) if x_mesh is None else x_mesh
    sol = solve_bvp(fun, bc, x_mesh, y_guess, tol=tol, max_nodes=max_nodes)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol(x_eval).T


def nystrom_fredholm(kernel: Callable, g: Callable, lam: float, a: float, b: float, n: int = 64):
    """Nystrom solution of u(x) = g(x) + lam int_a^b K(x, s) u(s) ds with n-point Gauss rule.

    Returns a callable evaluating the Nystrom interpolant at arbitrary points.
    """
    from numpy.polynomial.legendre import leggauss

    z, w = leggauss(n)
    s = 0.5 * (a + b) + 0.5 * (b - a) * z
    w = 0.5 * (b - a) * w
    A = np.eye(n) - lam * kernel(s[:, None], s[None, :]) * w[None, :]
    us = np.linalg.solve(A, g(s))

    def u(x):
        x = np.asarray(x, float)
        return g(x) + lam * (kernel(x[:, None], s[None, :]) * w[None, :]) @ us

    return u


def volterra_trapezoid(kernel: Callable, g: Callable, lam: float, a: float, b: float, n: int = 4001):
    """Second-order trapezoidal solution of u(x) = g(x) + lam int_a^x K(x, s) u(s) ds (linear)."""
    x = np.linspace(a, b, n)
    h = x[1] - x[0]
    u = np.empty(n)
    u[0] = g(x[0])
    for i in range(1, n):
        Ki = kernel(x[i], x[: i + 1])
        acc = 0.5 * Ki[0] * u[0] + np.dot(Ki[1:i], u[1:i])
        u[i] = (g(x[i]) + lam * h * acc) / (1.0 - lam * h * 0.5 * Ki[i])
    return x, u
