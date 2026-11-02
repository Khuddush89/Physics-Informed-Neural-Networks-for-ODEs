"""Error measures used consistently in every table of the book."""
from __future__ import annotations

import numpy as np


def _np(a):
    try:
        import torch

        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(a)


def rel_l2(pred, ref) -> float:
    """Discrete relative L2 error ||pred - ref||_2 / ||ref||_2 on a uniform test grid."""
    p, r = _np(pred).ravel(), _np(ref).ravel()
    return float(np.linalg.norm(p - r) / max(np.linalg.norm(r), 1e-300))


def linf(pred, ref) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(_np(pred).ravel() - _np(ref).ravel())))


def rmse(pred, ref) -> float:
    p, r = _np(pred).ravel(), _np(ref).ravel()
    return float(np.sqrt(np.mean((p - r) ** 2)))


def summary(pred, ref) -> dict:
    return {"rel_l2": rel_l2(pred, ref), "linf": linf(pred, ref), "rmse": rmse(pred, ref)}
