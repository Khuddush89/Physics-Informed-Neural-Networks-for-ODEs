"""Derivatives of network outputs with respect to inputs (reverse-mode AD).

``derivative(u, t, k)`` returns d^k u / dt^k for every output column of ``u``. The graph is
kept (``create_graph=True``) so that the result can itself be differentiated with respect
to the network parameters during training.
"""
from __future__ import annotations

from typing import List

import torch


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g, = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True,
                             allow_unused=True)
    return torch.zeros_like(x) if g is None else g


def derivative(u: torch.Tensor, t: torch.Tensor, order: int = 1) -> torch.Tensor:
    """k-th derivative of each column of ``u`` (shape (N, m)) w.r.t. the scalar input ``t``.

    ``t`` must have shape (N, 1) and ``requires_grad=True``. Returns a tensor of shape (N, m).
    """
    if order == 0:
        return u
    cols = []
    for j in range(u.shape[1]):
        d = u[:, j:j + 1]
        for _ in range(order):
            d = _grad(d, t)[:, :1]
        cols.append(d)
    return torch.cat(cols, dim=1)


def derivatives(u: torch.Tensor, t: torch.Tensor, max_order: int) -> List[torch.Tensor]:
    """[u, u', u'', ..., u^(max_order)] computed incrementally (each column separately)."""
    out = [u]
    for _ in range(max_order):
        prev = out[-1]
        out.append(torch.cat([_grad(prev[:, j:j + 1], t)[:, :1] for j in range(prev.shape[1])], dim=1))
    return out


def gradient(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Full gradient of a scalar-output network u (N, 1) w.r.t. multivariate input x (N, d)."""
    return _grad(u, x)
