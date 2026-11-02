"""Collocation points and quadrature rules on an interval [a, b].

All functions return NumPy arrays (use :func:`to_tensor` to convert) so that they can be
used both by PINN losses and by classical reference solvers.
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.polynomial.legendre import leggauss, legder, legroots


def to_tensor(x, requires_grad: bool = False) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x), dtype=torch.get_default_dtype())
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    return t.requires_grad_(requires_grad)


def uniform(n: int, a: float, b: float, endpoint: bool = True) -> np.ndarray:
    return np.linspace(a, b, n, endpoint=endpoint)


def random_uniform(n: int, a: float, b: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    return np.sort(rng.uniform(a, b, n))


def chebyshev_lobatto(n: int, a: float, b: float) -> np.ndarray:
    """n Chebyshev-Gauss-Lobatto points (extrema of T_{n-1}), increasing, endpoints included."""
    k = np.arange(n)
    x = -np.cos(np.pi * k / (n - 1))
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def chebyshev_gauss(n: int, a: float, b: float) -> np.ndarray:
    """n Chebyshev-Gauss points (zeros of T_n), increasing, endpoints excluded."""
    k = np.arange(1, n + 1)
    x = -np.cos((2 * k - 1) * np.pi / (2 * n))
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def legendre_gauss_lobatto(n: int, a: float, b: float) -> np.ndarray:
    """n Legendre-Gauss-Lobatto nodes: endpoints and the zeros of P'_{n-1}."""
    if n < 2:
        raise ValueError("n >= 2 required")
    c = np.zeros(n)
    c[-1] = 1.0
    inner = np.sort(np.real(legroots(legder(c)))) if n > 2 else np.array([])
    x = np.concatenate([[-1.0], inner, [1.0]])
    return 0.5 * (a + b) + 0.5 * (b - a) * x


def gauss_legendre(n: int, a: float, b: float):
    """Nodes and weights of the n-point Gauss-Legendre rule on [a, b] (exact to degree 2n-1)."""
    x, w = leggauss(n)
    return 0.5 * (a + b) + 0.5 * (b - a) * x, 0.5 * (b - a) * w


def composite_gauss_legendre(panels: int, n: int, a: float, b: float):
    """Composite Gauss-Legendre rule with ``panels`` equal sub-intervals and n nodes each."""
    edges = np.linspace(a, b, panels + 1)
    xs, ws = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x, w = gauss_legendre(n, lo, hi)
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Weights of the (non-uniform) composite trapezoidal rule on the nodes x."""
    x = np.asarray(x)
    w = np.zeros_like(x)
    h = np.diff(x)
    w[:-1] += 0.5 * h
    w[1:] += 0.5 * h
    return w


def volterra_gauss(x: np.ndarray, n: int, a: float = 0.0):
    """For every x_i, the n-point Gauss-Legendre nodes/weights on [a, x_i].

    Returns arrays ``S`` (nodes) and ``W`` (weights) of shape (len(x), n), so that
    ``sum_j W[i, j] f(x_i, S[i, j]) ~ int_a^{x_i} f(x_i, s) ds``.
    """
    z, w = leggauss(n)
    x = np.asarray(x).reshape(-1, 1)
    S = a + 0.5 * (x - a) * (z.reshape(1, -1) + 1.0)
    W = 0.5 * (x - a) * w.reshape(1, -1)
    return S, W
