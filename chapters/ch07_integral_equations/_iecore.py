"""Shared helpers for the Chapter 7 examples (imported by the scripts; not an example itself).

Contents
--------
* Filon-type panel quadrature for oscillatory integrals with the phase e^{+-ikt}:
  moments J_m(z) = int_0^1 xi^m e^{i z xi} dxi (m = 0, 1, 2), the quadratic Filon weights
  alpha_j(theta) and the partial-panel weights beta_j(theta, s), and the matrix W(x) of the
  product-integration rule  sum_j W[i, j] u(t_j) ~ int_0^1 e^{ik|x_i - t|} u(t) dt.
* The closed-form solution of the Helmholtz-type model equation
      u(x) = 1 + lam * (i / (2k)) int_0^1 e^{ik|x-t|} u(t) dt ,   x in [0, 1],
  obtained from the equivalent two-point boundary-value problem (book, Chapter 7).
* A product-integration (Filon) Nystrom solver for that equation.
* ``best_time``: wall-clock time of a classical solver as the minimum over repetitions
  (robust against load from other processes).

Numerical stability
-------------------
The closed-form moments, e.g. J_2(z) = (e^{iz}((iz)^2 - 2iz + 2) - 2) / (iz)^3, lose about
log10(1/|z|^3) digits to cancellation (relative error ~1e-7 at |z| = 1e-3 and no correct
digit at |z| = 1e-6).  ``moments`` therefore uses the power series
J_m(z) = sum_n (iz)^n / (n! (m + n + 1)) for |z| < 1, which converges to machine precision
with 30 terms, and the closed form only for |z| >= 1, where it is accurate.
"""
from __future__ import annotations

import time
from typing import Callable, Tuple

import numpy as np

# --------------------------------------------------------------------------- Filon moments
_NSER = 30
_FACT = np.cumprod(np.r_[1.0, np.arange(1, _NSER, dtype=float)])  # 0!, 1!, ..., 29!


def moments_closed(z) -> np.ndarray:
    """Closed-form J_0, J_1, J_2 at real z != 0 (unstable for small |z|; for verification)."""
    z = np.asarray(z, float)
    e, iz = np.exp(1j * z), 1j * z
    return np.stack([(e - 1.0) / iz,
                     (e * (iz - 1.0) + 1.0) / iz ** 2,
                     (e * (iz ** 2 - 2.0 * iz + 2.0) - 2.0) / iz ** 3])


def moments_series(z, nterms: int = _NSER) -> np.ndarray:
    """Power series of J_0, J_1, J_2 (accurate for |z| <~ 1)."""
    z = np.asarray(z, float)
    n = np.arange(nterms)
    pw = (1j * z[..., None]) ** n / _FACT[:nterms]
    return np.stack([(pw / (m + n + 1.0)).sum(-1) for m in range(3)])


def moments(z) -> np.ndarray:
    """J_m(z) = int_0^1 xi^m e^{i z xi} dxi for m = 0, 1, 2; shape (3, *z.shape).

    Series for |z| < 1, closed form otherwise: accurate to rounding for every real z.
    """
    z0 = np.asarray(z, float)
    z = np.atleast_1d(z0).ravel()
    out = np.empty((3, z.size), complex)
    small = np.abs(z) < 1.0
    if small.any():
        out[:, small] = moments_series(z[small])
    if (~small).any():
        out[:, ~small] = moments_closed(z[~small])
    return out.reshape((3,) + z0.shape)


def weights_from_moments(J: np.ndarray) -> np.ndarray:
    """Quadratic Lagrange weights from moments: L0 = 1-3xi+2xi^2, L1 = 4xi-4xi^2, L2 = -xi+2xi^2."""
    J0, J1, J2 = J
    return np.stack([J0 - 3.0 * J1 + 2.0 * J2, 4.0 * J1 - 4.0 * J2, -J1 + 2.0 * J2])


def filon_weights(theta, s=1.0) -> np.ndarray:
    """beta_j(theta, s) = int_0^s e^{i theta xi} L_j(xi) dxi, j = 0, 1, 2 (alpha_j = beta_j(., 1)).

    Uses int_0^s xi^m e^{i theta xi} dxi = s^{m+1} J_m(theta s).  Returns shape (3, *bcast).
    """
    theta, s = np.broadcast_arrays(np.asarray(theta, float), np.asarray(s, float))
    J = moments(theta * s)
    J = np.stack([s * J[0], s ** 2 * J[1], s ** 3 * J[2]])
    return weights_from_moments(J)


# --------------------------------------------------------------------------- panel matrix
def panel_nodes(P: int) -> np.ndarray:
    """Nodes t_j = j h / 2 (j = 0..2P) of P uniform panels on [0, 1] (endpoints and midpoints)."""
    return np.linspace(0.0, 1.0, 2 * P + 1)


def filon_matrix(x, P: int, k: float, phase: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Matrix W (len(x), 2P+1) with  sum_j W[i, j] u(t_j) ~ int_0^1 e^{ik|x_i - t|} u(t) dt.

    The integral is split at t = x_i.  On every panel (or part of a panel) u is replaced by
    its quadratic interpolant at the three panel nodes.

    phase=True  (Filon):   the factors e^{-+ikt} are integrated exactly against the
                           interpolant (weights alpha_j(-+kh), beta_j(-+kh, s)).
    phase=False (Simpson): the whole integrand e^{ik|x_i - t|} u(t) is interpolated on each
                           side of x_i (same nodes, weights alpha_j(0), beta_j(0, s)).
    """
    x = np.asarray(x, float).ravel()
    h = 1.0 / P
    a = h * np.arange(P)
    th = k * h if phase else 0.0
    am, ap = filon_weights(-th), filon_weights(th)               # (3,)
    pstar = np.minimum(np.floor(x / h + 1e-12).astype(int), P - 1)
    s = np.clip((x - a[pstar]) / h, 0.0, 1.0)
    bm, bp = filon_weights(-th, s).T, filon_weights(th, s).T     # (N, 3)
    q = np.arange(P)[None, :]
    left = (q < pstar[:, None])[..., None]
    self_ = (q == pstar[:, None])[..., None]
    right = (q > pstar[:, None])[..., None]
    if phase:
        EL = np.exp(1j * k * (x[:, None] - a[None, :]))[..., None]   # e^{ik(x - a_q)}
        ER = np.conj(EL)                                             # e^{-ik(x - a_q)}
    else:
        tj = a[:, None] + h * np.array([0.0, 0.5, 1.0])[None, :]
        EL = np.exp(1j * k * (x[:, None, None] - tj[None]))          # e^{ik(x - t_j)}
        ER = np.conj(EL)
    C = (left * EL * am + right * ER * ap
         + self_ * (EL * bm[:, None, :] + ER * (ap - bp)[:, None, :])) * h
    W = np.zeros((x.size, 2 * P + 1), complex)
    W[:, 0:2 * P:2] += C[:, :, 0]
    W[:, 1:2 * P:2] += C[:, :, 1]
    W[:, 2:2 * P + 1:2] += C[:, :, 2]
    return W, panel_nodes(P)


# --------------------------------------------------------------------------- model problem
def helmholtz_exact(x, k: float, lam: float) -> np.ndarray:
    """Closed-form solution of u = 1 + lam (i/2k) int_0^1 e^{ik|x-t|} u dt  (lam > -k^2).

    u(x) = k^2/mu^2 + (i lam k / mu^2) cos(mu (x - 1/2)) / (mu sin(mu/2) + i k cos(mu/2)),
    mu = sqrt(k^2 + lam).
    """
    x = np.asarray(x, float)
    mu = np.sqrt(k ** 2 + lam)
    C = 1j * lam * k / mu ** 2 / (mu * np.sin(mu / 2) + 1j * k * np.cos(mu / 2))
    return k ** 2 / mu ** 2 + C * np.cos(mu * (x - 0.5))


def filon_nystrom(P: int, k: float, lam: float, phase: bool = True):
    """Product-integration Nystrom solution at the 2P+1 panel nodes (f = 1).

    Returns (nodes, u_nodes, interpolant) where interpolant(x) = 1 + lam (i/2k) W(x) u_nodes
    is the natural Nystrom interpolant.
    """
    W, t = filon_matrix(panel_nodes(P), P, k, phase)
    A = np.eye(t.size) - lam * (1j / (2.0 * k)) * W
    un = np.linalg.solve(A, np.ones(t.size, complex))

    def interp(x):
        Wx, _ = filon_matrix(x, P, k, phase)
        return 1.0 + lam * (1j / (2.0 * k)) * (Wx @ un)

    return t, un, interp


# --------------------------------------------------------------------------- timing
def best_time(fn: Callable[[], object], repeat: int = 7):
    """Run ``fn`` ``repeat`` times; return (last result, minimum wall-clock time in s)."""
    best, out = np.inf, None
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


# --------------------------------------------------------------------------- figure data
NYSTROM = r"Nystr$\ddot{\mathrm{o}}$m"      # renders the umlaut with the book's mathtext fonts
REPLOT_ARG = (["--replot"], {"action": "store_true",
                             "help": "redraw the figure from figdata.npz of a previous full run"})


def history_arrays(hist, prefix: str) -> dict:
    """Flatten a pinnode History into arrays for np.savez (used by --replot)."""
    d = {f"{prefix}__iters": np.asarray(hist.iters), f"{prefix}__loss": np.asarray(hist.loss),
         f"{prefix}__eval_iters": np.asarray(hist.eval_iters),
         f"{prefix}__switch": np.asarray(-1 if hist.switch_iter is None else hist.switch_iter)}
    for k, v in hist.parts.items():
        d[f"{prefix}__part__{k}"] = np.asarray(v)
    for k, v in hist.metrics.items():
        d[f"{prefix}__metric__{k}"] = np.asarray(v)
    return d


def history_from_arrays(data, prefix: str):
    """Inverse of :func:`history_arrays` (returns a pinnode History)."""
    from pinnode.training import History

    h = History()
    h.iters = list(data[f"{prefix}__iters"])
    h.loss = list(data[f"{prefix}__loss"])
    h.eval_iters = list(data[f"{prefix}__eval_iters"])
    sw = int(data[f"{prefix}__switch"])
    h.switch_iter = None if sw < 0 else sw
    for key in data.files:
        if key.startswith(f"{prefix}__part__"):
            h.parts[key.split("__part__")[1]] = list(data[key])
        elif key.startswith(f"{prefix}__metric__"):
            h.metrics[key.split("__metric__")[1]] = list(data[key])
    return h
