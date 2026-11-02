"""Shared helpers for the Chapter 11 examples (inverse problems, data assimilation, UQ).

This module is imported by the example scripts of this folder; it is not an example itself
(files starting with an underscore are skipped by ``scripts/run_all.py`` and the smoke tests).

Contents
--------
* timing:            :class:`CPUTimer` (process CPU time and wall-clock time of a block);
* caching:           :func:`cached` (``--resume`` reuses the fits of an interrupted run);
* noise:             :func:`add_noise` (additive Gaussian noise with sd = eta * max|y|);
* logistic model:    exact solution, sensitivities, synthetic data sets, the inverse PINN
                     (:func:`fit_logistic_pinn`) and the classical output-least-squares fit
                     with ``solve_ivp`` inside ``least_squares`` (:func:`fit_logistic_nls`);
* LaTeX output:      :func:`write_tabular` and number formatting for the auto-generated tables
                     that the book includes with ``\\input``.

Timing: the companion machine was shared while the book was produced, so wall-clock times
are inflated. Every fit therefore records the CPU time of the process (one thread), which is
what the generated tables report; on an idle machine CPU and wall-clock time coincide.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from pinnode import ConstrainedModel, MLP, derivative, to_tensor, train, uniform


# ============================================================================ timing
class CPUTimer:
    """Context manager recording process CPU time (``cpu``) and wall-clock time (``wall``)."""

    def __enter__(self):
        self.c0, self.w0 = time.process_time(), time.perf_counter()
        self.cpu = self.wall = 0.0
        return self

    def __exit__(self, *exc):
        self.cpu = time.process_time() - self.c0
        self.wall = time.perf_counter() - self.w0
        return False


# ============================================================================ caching
RESUME_ARG = (("--resume",), {"action": "store_true",
                              "help": "reuse fits cached by an interrupted run (same configuration)"})


def cached(ex, key: str, fn, **config):
    """Run ``fn()`` (returning picklable data) and cache the result in ``outdir/_cache``.

    With ``--resume`` a cached result with the same key and configuration is loaded instead of
    recomputed. Every fit seeds its own random number generators, so a resumed run reproduces
    the numbers of an uninterrupted run exactly (including the recorded CPU times).
    """
    folder = Path(ex.outdir) / "_cache"
    folder.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha1(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:10]
    f = folder / f"{key}_{tag}.pkl"
    if getattr(ex.args, "resume", False) and f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    res = fn()
    with open(f, "wb") as fh:
        pickle.dump(res, fh)
    return res


# ============================================================================ noise
def add_noise(y: np.ndarray, eta: float, rng: np.random.Generator):
    """Return ``(y + sigma * xi, sigma)`` with ``xi ~ N(0, I)`` and ``sigma = eta * max|y|``."""
    y = np.asarray(y, dtype=float)
    sigma = float(eta * np.max(np.abs(y)))
    return y + sigma * rng.standard_normal(y.shape), sigma


def data_rng(example_id: int, eta: float, realization: int) -> np.random.Generator:
    """Independent, reproducible noise stream for (example, noise level, realization)."""
    return np.random.default_rng([example_id, int(round(1000 * eta)), realization])


# ============================================================================ logistic model
# y' = r y (1 - y/K),  y(0) = y0  on [0, T];  true parameters used throughout Chapter 11.
LOGISTIC = {"r": 0.8, "K": 10.0, "y0": 0.5, "T": 10.0, "N": 15}
LOGISTIC_P_INIT = (0.3, 5.0)          # common initial guess (r0, K0) for every method
LOGISTIC_ID = 11                      # example id for the noise streams


def logistic_exact(t, r: float, K: float, y0: float):
    """Closed-form solution y(t) = K / (1 + (K/y0 - 1) exp(-r t))."""
    t = np.asarray(t, dtype=float)
    return K / (1.0 + (K / y0 - 1.0) * np.exp(-r * t))


def logistic_sensitivities(t, r: float, K: float, y0: float) -> np.ndarray:
    """Columns dy/dr and dy/dK of the closed-form solution (shape (len(t), 2))."""
    t = np.asarray(t, dtype=float)
    e = np.exp(-r * t)
    D = 1.0 + (K / y0 - 1.0) * e
    dr = K * (K / y0 - 1.0) * t * e / D ** 2
    dK = 1.0 / D - K * e / (y0 * D ** 2)
    return np.stack([dr, dK], axis=1)


def logistic_obs_times(T: float = LOGISTIC["T"], N: int = LOGISTIC["N"]) -> np.ndarray:
    """N equally spaced observation times in (0, T] (y(0) = y0 is known)."""
    return np.linspace(0.0, T, N + 1)[1:]


def logistic_data(eta: float, realization: int, t_obs: Optional[np.ndarray] = None):
    """Synthetic data set number ``realization`` at noise level ``eta``.

    Returns ``(t_obs, d_obs, sigma)``. The same (eta, realization) pair always gives the same
    data, so different scripts (Sections 11.2, 11.6 and 11.7) can share data sets.
    """
    p = LOGISTIC
    t_obs = logistic_obs_times() if t_obs is None else np.asarray(t_obs, dtype=float)
    y = logistic_exact(t_obs, p["r"], p["K"], p["y0"])
    d, sigma = add_noise(y, eta, data_rng(LOGISTIC_ID, eta, realization))
    return t_obs, d, sigma


def fit_logistic_pinn(t_obs, d_obs, *, y0: float = LOGISTIC["y0"], T: float = LOGISTIC["T"],
                      p_init: Sequence[float] = LOGISTIC_P_INIT, lam_r: float = 1.0,
                      width: int = 20, depth: int = 2, n_col: int = 100, adam_iters: int = 2000,
                      lbfgs_iters: int = 500, lr: float = 5e-3, seed: int = 0, scale: float = 10.0,
                      p_true: Optional[Sequence[float]] = None, eval_every: int = 25,
                      t_extra: Optional[np.ndarray] = None, verbose: bool = False) -> Dict:
    """Inverse PINN for the logistic model with unknown (r, K) and known y(0) = y0.

    Trial solution (hard initial condition, output scaled by ``scale`` ~ max|y|)::

        y_theta(t) = y0 + (t / T) * scale * N_theta(t)

    Positive parameters are log-parametrized, r = exp(rho), K = exp(kappa). The loss is the
    scaled data misfit plus lam_r times the scaled residual (both are O(1) quantities)::

        L = (1/N) sum_i ((y_theta(t_i) - d_i)/scale)^2
            + lam_r (1/M) sum_j ((T/scale) (y_theta'(t_j) - r y_theta (1 - y_theta/K)))^2
    """
    torch.manual_seed(seed)
    net = MLP(1, 1, width=width, depth=depth, activation="tanh", domain=(0.0, T))
    model = ConstrainedModel(net, lambda t, n: y0 + (t / T) * scale * n)
    log_p = torch.nn.Parameter(torch.log(torch.as_tensor(np.asarray(p_init, dtype=float))))
    t_col = to_tensor(uniform(n_col, 0.0, T), requires_grad=True)
    t_d, d = to_tensor(t_obs), to_tensor(d_obs)

    def loss_fn():
        r, K = torch.exp(log_p)
        y = model(t_col)
        res = (T / scale) * (derivative(y, t_col) - r * y * (1.0 - y / K))
        L_r = (res ** 2).mean()
        L_d = (((model(t_d) - d) / scale) ** 2).mean()
        return L_d + lam_r * L_r, {"data": L_d, "residual": L_r}

    def eval_fn():
        r, K = torch.exp(log_p).detach().numpy()
        out = {"r": r, "K": K}
        if p_true is not None:
            out["err_r"] = abs(r / p_true[0] - 1.0)
            out["err_K"] = abs(K / p_true[1] - 1.0)
        return out

    with CPUTimer() as tm:
        hist = train(list(model.parameters()) + [log_p], loss_fn, adam_iters=adam_iters, lr=lr,
                     lbfgs_iters=lbfgs_iters, eval_fn=eval_fn, eval_every=eval_every,
                     verbose=verbose, print_every=1000 if verbose else 0)
    t_grid = np.linspace(0.0, T, 1001)
    with torch.no_grad():
        misfit = float(torch.sqrt(((model(t_d) - d) ** 2).mean()))
        y_grid = model(to_tensor(t_grid)).numpy().ravel()
        y_extra = None if t_extra is None else model(to_tensor(t_extra)).numpy().ravel()
    return {"params": np.exp(log_p.detach().numpy()), "hist": hist, "t_grid": t_grid, "y_grid": y_grid,
            "y_extra": y_extra, "cpu_s": tm.cpu, "wall_s": hist.train_time, "rms_misfit": misfit,
            "final_data_loss": hist.parts["data"][-1], "final_residual_loss": hist.parts["residual"][-1]}


def logistic_solve(t_eval, r: float, K: float, y0: float, rtol: float = 1e-10, atol: float = 1e-12):
    """Numerical forward solve with DOP853 (used by the classical fit as a black box)."""
    sol = solve_ivp(lambda t, y: r * y * (1.0 - y / K), (0.0, float(np.max(t_eval))), [y0],
                    t_eval=t_eval, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[0]


def fit_logistic_nls(t_obs, d_obs, *, y0: float = LOGISTIC["y0"],
                     p_init: Sequence[float] = LOGISTIC_P_INIT) -> Dict:
    """Classical output least squares: ``least_squares`` over (log r, log K), with the model
    values computed by ``solve_ivp`` (DOP853, rtol=1e-10, atol=1e-12) and a finite-difference
    Jacobian (relative step 1e-6). Returns estimates, the Jacobian with respect to (r, K),
    CPU time and the number of forward solves.
    """
    t_obs, d_obs = np.asarray(t_obs, float), np.asarray(d_obs, float)
    count = {"n": 0}

    def resid(q):
        count["n"] += 1
        r, K = np.exp(q)
        return logistic_solve(t_obs, r, K, y0) - d_obs

    with CPUTimer() as tm:
        res = least_squares(resid, np.log(np.asarray(p_init, float)), method="trf", diff_step=1e-6,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
    p = np.exp(res.x)
    jac_p = res.jac / p[None, :]               # chain rule: d/dp = (1/p) d/dlog p
    return {"params": p, "jac": jac_p, "cpu_s": tm.cpu, "wall_s": tm.wall,
            "n_forward_solves": count["n"], "rss": float(2.0 * res.cost),
            "rms_misfit": float(np.sqrt(2.0 * res.cost / len(d_obs))), "success": bool(res.success)}


def gauss_newton_covariance(jac: np.ndarray, sigma: float) -> np.ndarray:
    """Linearized covariance sigma^2 (J^T J)^{-1} of an output-least-squares estimate."""
    return sigma ** 2 * np.linalg.inv(jac.T @ jac)


# ============================================================================ light-weight CSV output
def _fmt(v):
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.10e}"
    if isinstance(v, (np.integer,)):
        return str(int(v))
    return str(v)


def save_rows_csv(path, rows) -> Path:
    """Write a list of dicts (or a dict of equal-length sequences) as CSV without pandas.

    The examples avoid importing pandas and matplotlib during training to keep the memory
    footprint small on shared machines.
    """
    import csv
    if isinstance(rows, dict):
        keys = list(rows)
        n = len(next(iter(rows.values())))
        rows = [{k: rows[k][i] for k in keys} for i in range(n)]
    keys = list(rows[0].keys()) if rows else []
    path = Path(path)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([_fmt(r[k]) for k in keys])
    return path


def save_history_csv(path, hist) -> Path:
    """Training history (loss components and evaluated metrics) as CSV without pandas."""
    import csv
    metrics = {it: {k: hist.metrics[k][i] for k in hist.metrics} for i, it in enumerate(hist.eval_iters)}
    pkeys, mkeys = list(hist.parts), list(hist.metrics)
    path = Path(path)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iter", "phase", "loss", "lr", "time_s", *pkeys, *mkeys])
        for j, it in enumerate(hist.iters):
            m = metrics.get(it, {})
            w.writerow([it, hist.phase[j], _fmt(hist.loss[j]), _fmt(hist.lr[j]), _fmt(hist.time[j]),
                        *[_fmt(hist.parts[k][j]) for k in pkeys],
                        *[(_fmt(m[k]) if k in m else "") for k in mkeys]])
    return path


# ============================================================================ LaTeX tables
def fmt_num(x: float, digits: int = 2) -> str:
    """siunitx-friendly scientific notation, e.g. 1.23e-04 -> \\num{1.23e-4}."""
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return r"\num{0}"
    m, e = f"{x:.{digits}e}".split("e")
    return rf"\num{{{m}e{int(e)}}}"


def fmt_fixed(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:.{digits}f}"


def fmt_pm(mean: float, sd: float, digits: int = 3) -> str:
    """mean \\pm sd in fixed-point notation (math mode)."""
    return rf"${mean:.{digits}f} \pm {sd:.{digits}f}$"


def fmt_time(seconds: float) -> str:
    """CPU seconds with two significant digits."""
    if seconds is None or not np.isfinite(seconds):
        return "--"
    if seconds >= 10:
        return f"{seconds:.0f}"
    if seconds >= 1:
        return f"{seconds:.1f}"
    return f"{seconds:.2g}"


def write_tabular(path: Path, colspec: str, header: Sequence[str] | str, rows: Iterable,
                  comment: str = "") -> Path:
    """Write a booktabs ``tabular`` (no float, no caption) to ``path``.

    ``rows`` holds lists of cell strings, or the string ``"midrule"`` for a rule. The book
    wraps the file in a table float with its own caption and label.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"% generated by {comment or 'chapters/ch11_inverse_problems'} -- do not edit by hand",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    head = header if isinstance(header, str) else " & ".join(header) + r" \\"
    lines += [head, r"\midrule"]
    for row in rows:
        if isinstance(row, str):
            lines.append(r"\midrule" if row == "midrule" else row)
        else:
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def median(xs: Iterable[float]) -> float:
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return float(np.median(xs)) if xs else math.nan


__all__ = [
    "CPUTimer", "RESUME_ARG", "cached", "add_noise", "data_rng", "LOGISTIC", "LOGISTIC_P_INIT", "LOGISTIC_ID",
    "logistic_exact", "logistic_sensitivities", "logistic_obs_times", "logistic_data",
    "fit_logistic_pinn", "logistic_solve", "fit_logistic_nls", "gauss_newton_covariance",
    "save_rows_csv", "save_history_csv", "fmt_num", "fmt_fixed", "fmt_pm", "fmt_time", "write_tabular", "median",
]
