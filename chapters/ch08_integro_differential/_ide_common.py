"""Shared helpers for the Chapter 8 examples: exact solutions, classical solvers, timing.

The PINN code of every example lives in its own script; this module only collects the
pieces that several scripts need (closed-form solutions, the classical reference solvers
used in the comparison tables, CPU timing).  It is not an example itself (the leading
underscore keeps it out of ``scripts/run_all.py`` and the smoke tests).
"""
from __future__ import annotations

import math
import pickle
import time
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import airy


# ----------------------------------------------------------------------------- timing
class CpuWallTimer:
    """Context manager recording process CPU time and wall-clock time (seconds).

    On a shared machine wall-clock time depends on the load; the CPU time of the
    (single-threaded) process is the better proxy for the cost of a computation.
    """

    def __enter__(self):
        self.c0, self.w0 = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.cpu = time.process_time() - self.c0
        self.wall = time.perf_counter() - self.w0
        return False


def time_call(fn: Callable[[], object], min_time: float = 0.2, max_repeat: int = 2000
              ) -> Tuple[object, float, float]:
    """Call ``fn`` repeatedly (at least ``min_time`` CPU seconds) and return
    ``(result, cpu_seconds_per_call, wall_seconds_per_call)``."""
    result = fn()                                   # warm-up (imports, caches)
    n, c0, w0 = 0, time.process_time(), time.perf_counter()
    while True:
        result = fn()
        n += 1
        if time.process_time() - c0 >= min_time or n >= max_repeat:
            break
    return result, (time.process_time() - c0) / n, (time.perf_counter() - w0) / n


# ----------------------------------------------------------------------------- resumable runs
RESUME_ARG = (("--resume",), {"action": "store_true",
                              "help": "reuse finished variants cached in the output folder"})


def cached(ex, name: str, fn: Callable[[], Dict]) -> Dict:
    """Run ``fn()`` (one trained variant) or, with ``--resume``, load its cached result.

    Every variant reseeds the generators before building its network, so a cached result
    is identical to a fresh one; the cache only protects long runs on shared machines.
    Entries named ``model`` are dropped before pickling.
    """
    path = ex.path(f"cache_{name}.pkl")
    if getattr(ex.args, "resume", False) and path.exists():
        with open(path, "rb") as fh:
            ex.log(f"{name}: loaded cached result {path.name}")
            return pickle.load(fh)
    res = fn()
    res = {k: v for k, v in res.items() if k != "model"}
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return res


# ----------------------------------------------------------------------------- Volterra example
#   u'(x) = u(x) + int_0^x t u(t) dt,  u(0) = 1.
# Differentiating gives u'' - u' - x u = 0 with u(0) = 1, u'(0) = 1; u = e^{x/2} w turns
# this into the Airy equation w'' = (x + 1/4) w, hence
#   u(x) = e^{x/2} [c1 Ai(x + 1/4) + c2 Bi(x + 1/4)],
#   c1 = pi (Bi'(1/4) - Bi(1/4)/2),  c2 = pi (Ai(1/4)/2 - Ai'(1/4))   (Wronskian 1/pi).
_AI0, _AIP0, _BI0, _BIP0 = (float(v) for v in airy(0.25))
VOLTERRA_C1 = math.pi * (_BIP0 - 0.5 * _BI0)
VOLTERRA_C2 = math.pi * (0.5 * _AI0 - _AIP0)


def volterra_exact(x) -> np.ndarray:
    """Exact solution u(x) of the Volterra example."""
    x = np.asarray(x, dtype=float)
    ai, _, bi, _ = airy(x + 0.25)
    return np.exp(0.5 * x) * (VOLTERRA_C1 * ai + VOLTERRA_C2 * bi)


def volterra_exact_aux(x) -> np.ndarray:
    """Exact auxiliary variable A(x) = int_0^x t u(t) dt = u'(x) - u(x)."""
    x = np.asarray(x, dtype=float)
    ai, aip, bi, bip = airy(x + 0.25)
    v = VOLTERRA_C1 * ai + VOLTERRA_C2 * bi
    vp = VOLTERRA_C1 * aip + VOLTERRA_C2 * bip
    return np.exp(0.5 * x) * (vp - 0.5 * v)


def volterra_rhs(x, y):
    """ODE system (u, A)' = (u + A, x u) equivalent to the Volterra example."""
    return [y[0] + y[1], x * y[0]]


# ----------------------------------------------------------------------------- convolution example
#   u'(x) = u(x) + int_0^x e^{-(x-t)} u(t) dt,  u(0) = 1;  z = int_0^x e^{-(x-t)} u dt.
SQRT2 = math.sqrt(2.0)


def convolution_exact(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.cosh(SQRT2 * x) + 0.5 * SQRT2 * np.sinh(SQRT2 * x)


def convolution_exact_aux(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * SQRT2 * np.sinh(SQRT2 * x)


# ----------------------------------------------------------------------------- Fredholm example
#   u'(x) = 1 + int_0^1 x t u(t) dt,  u(0) = 0;  u = x + C x^2 / 2 with C = 8/21.
FREDHOLM_C = 8.0 / 21.0


def fredholm_exact(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x + 0.5 * FREDHOLM_C * x ** 2


# ----------------------------------------------------------------------------- classical solvers
def trapezoidal_vide(a: Callable, kernel: Callable, f: Callable, u0: float, L: float, N: int):
    """Implicit trapezoidal direct-quadrature method for the linear Volterra IDE

        u'(x) = a(x) u(x) + f(x) + int_0^x K(x, t) u(t) dt,   u(0) = u0,   x in [0, L].

    The IDE is integrated over [x_n, x_{n+1}] with the trapezoidal rule and the memory
    term at every grid point with the composite trapezoidal rule (second order for smooth
    data; Linz 1985, Brunner 2004).  The unknown u_{n+1} enters linearly, so each step is
    explicit after one division.  Cost O(N^2) kernel evaluations (no structure of K used).

    Returns the grid x_n = n L / N and the approximations u_n (arrays of length N + 1).
    """
    x = np.linspace(0.0, L, N + 1)
    h = L / N
    u = np.empty(N + 1)
    u[0] = u0
    G = a(x[0]) * u0 + f(x[0])                 # right-hand side at x_0 (memory term = 0)
    for n in range(N):
        xn1 = x[n + 1]
        K = kernel(xn1, x[: n + 2])             # K(x_{n+1}, x_j), j = 0..n+1
        P = f(xn1) + h * (0.5 * K[0] * u[0] + np.dot(K[1:n + 1], u[1:n + 1]))
        c = a(xn1) + 0.5 * h * K[n + 1]
        u[n + 1] = (u[n] + 0.5 * h * (G + P)) / (1.0 - 0.5 * h * c)
        G = P + c * u[n + 1]
    return x, u


def dop853_system(rhs: Callable, y0, L: float, x_eval: np.ndarray, rtol: float = 1e-12,
                  atol: float = 1e-12) -> np.ndarray:
    """Solve an ODE system with DOP853 (tight tolerances); returns shape (len(x_eval), dim)."""
    sol = solve_ivp(rhs, (0.0, L), np.atleast_1d(y0).astype(float), method="DOP853",
                    t_eval=x_eval, rtol=rtol, atol=atol)
    if not sol.success:  # pragma: no cover
        raise RuntimeError(sol.message)
    return sol.y.T


def errors(pred, ref) -> Dict[str, float]:
    """Relative L2, maximum absolute and maximum relative error on a grid."""
    p, r = np.asarray(pred, float).ravel(), np.asarray(ref, float).ravel()
    e = np.abs(p - r)
    return {"rel_l2": float(np.linalg.norm(p - r) / np.linalg.norm(r)),
            "linf": float(e.max()),
            "max_rel": float(np.max(e / np.maximum(np.abs(r), 1e-300)))}
