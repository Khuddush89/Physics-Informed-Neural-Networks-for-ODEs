"""Caputo fractional relaxation  D^alpha y = -y,  y(0) = 1,  t in [0, 2],  0 < alpha < 1  (fPINN).

Book: Chapter 12 (Further equation classes), Section "Fractional differential equations".

Caputo derivative: D^alpha y(t) = 1/Gamma(1-alpha) int_0^t (t-s)^(-alpha) y'(s) ds.
Exact solution: y(t) = E_alpha(-t^alpha) (Mittag-Leffler function), evaluated here by its power
series in multiple precision (mpmath, precision adapted to t) and verified against
    alpha = 1/2:  E_{1/2}(-sqrt(t)) = exp(t) erfc(sqrt(t)) = erfcx(sqrt(t)),
    alpha = 0.8:  the spectral representation
        E_alpha(-t^alpha) = (sin(alpha pi)/pi) int_0^inf r^(alpha-1) e^(-r t) / (r^(2alpha) + 2 r^alpha cos(alpha pi) + 1) dr.

fPINN residual: the Caputo derivative of the network is replaced by the L1 formula on a grid
0 = t_0 < t_1 < ... < t_N = T,
    D^alpha y(t_n) ~ sum_{j=1}^n (y_j - y_{j-1}) / (t_j - t_{j-1})
                     * [(t_n - t_{j-1})^(1-alpha) - (t_n - t_j)^(1-alpha)] / Gamma(2 - alpha),
so R_n = (L1 y_theta)(t_n) + y_theta(t_n), n = 1..N.  The trial solution y = 1 + t^alpha N(t)
satisfies y(0) = 1 and carries the t^alpha singularity of the exact solution.  Uniform grids and
graded grids t_n = T (n/N)^r with r = (2 - alpha)/alpha are compared, together with the classical
L1 finite-difference solution of the same discrete equations.

Run:  python chapters/ch12_further_classes/fractional_relaxation.py          (about 3-5 min on 1 core)
      python chapters/ch12_further_classes/fractional_relaxation.py --quick  (smoke test)
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import torch
from scipy.integrate import quad
from matplotlib.ticker import NullFormatter
from scipy.special import erfcx, gamma

from pinnode import ConstrainedModel, Example, MLP, linf, rel_l2, to_tensor, train
from pinnode import plotting as P

T_END = 2.0
ALPHAS = [0.5, 0.8]
GRIDS = [16, 32, 64, 128, 256]


# ----------------------------------------------------------------------------- Mittag-Leffler
def ml_relaxation(alpha: float, t) -> np.ndarray:
    """E_alpha(-t^alpha), t >= 0, by the power series sum_k (-t^alpha)^k / Gamma(alpha k + 1).

    The terms grow to about exp(t) before decaying, so about t/ln(10) significant digits
    cancel; the series is therefore summed in multiple precision with 30 + t_max/ln(10) + 10
    digits (reliable for the moderate t used here) by Horner's rule with precomputed
    coefficients 1/Gamma(alpha k + 1), truncated where the terms fall below 10^-(digits).
    """
    t = np.atleast_1d(np.asarray(t, float))
    t_max = float(t.max()) if t.size else 0.0
    dps = 30 + int(t_max / math.log(10.0)) + 10
    with mp.workdps(dps):
        a = mp.mpf(alpha)
        xmax = mp.mpf(t_max) ** a
        coef, k = [], 0
        while True:                            # coefficients until terms at t_max are negligible
            c = 1 / mp.gamma(a * k + 1)
            coef.append(c)
            if k > 2 * t_max / alpha + 10 and c * xmax ** k < mp.mpf(10) ** (-dps):
                break
            k += 1
        out = np.empty_like(t)
        for i, ti in enumerate(t):
            x = -(mp.mpf(ti) ** a)
            s = coef[-1]
            for c in reversed(coef[:-1]):
                s = s * x + c
            out[i] = float(s)
    return out


def ml_relaxation_integral(alpha: float, t: float) -> float:
    """Spectral (completely monotone) representation, used only to verify the series."""
    c, s = math.cos(alpha * math.pi), math.sin(alpha * math.pi)

    def k(r):
        return r ** (alpha - 1) * math.exp(-r * t) / (r ** (2 * alpha) + 2 * r ** alpha * c + 1)

    v1, _ = quad(k, 0.0, 1.0, limit=200, epsabs=1e-15, epsrel=1e-13)
    v2, _ = quad(k, 1.0, np.inf, limit=200, epsabs=1e-15, epsrel=1e-13)
    return s / math.pi * (v1 + v2)


# ----------------------------------------------------------------------------- L1 scheme
def make_grid(n: int, alpha: float, graded: bool) -> np.ndarray:
    r = (2.0 - alpha) / alpha if graded else 1.0
    return T_END * (np.arange(n + 1) / n) ** r


def l1_matrix(t: np.ndarray, alpha: float) -> np.ndarray:
    """A (N x (N+1)) with (A y)_n ~ D^alpha y(t_n), n = 1..N, on the (non-uniform) grid t."""
    n_pts = len(t) - 1
    A = np.zeros((n_pts, n_pts + 1))
    g = gamma(2.0 - alpha)
    for n in range(1, n_pts + 1):
        j = np.arange(1, n + 1)
        w = ((t[n] - t[j - 1]) ** (1 - alpha) - (t[n] - t[j]) ** (1 - alpha)) / (g * (t[j] - t[j - 1]))
        A[n - 1, j] += w
        A[n - 1, j - 1] -= w
    return A


def l1_solve(t: np.ndarray, alpha: float) -> np.ndarray:
    """Classical L1 finite-difference solution of D^alpha y = -y, y(0) = 1 (lower triangular)."""
    A = l1_matrix(t, alpha)
    M = A[:, 1:] + np.eye(len(t) - 1)
    y = np.empty(len(t))
    y[0] = 1.0
    y[1:] = np.linalg.solve(M, -A[:, 0] * y[0])
    return y


# ----------------------------------------------------------------------------- fPINN
def train_fpinn(ex: Example, alpha: float, t_grid: np.ndarray, t_test, y_test, verbose=False):
    torch.manual_seed(ex.seed)
    net = MLP(1, 1, width=20, depth=2, activation="tanh", domain=(0.0, T_END))
    model = ConstrainedModel(net, lambda t, n: 1.0 + t.clamp_min(0.0) ** alpha * n)
    A = torch.as_tensor(l1_matrix(t_grid, alpha))
    tg = to_tensor(t_grid)
    tt = to_tensor(t_test)

    def loss_fn():
        y = model(tg)                          # values at t_0..t_N; no autodiff in t needed
        r = A @ y + y[1:]
        Lr = (r ** 2).mean()
        return Lr, {"residual": Lr}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(tt), y_test)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(2500, 40), lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(500, 10), eval_fn=eval_fn, eval_every=100,
                 print_every=2500 if verbose else 0, verbose=verbose)
    with torch.no_grad():
        y_nodes = model(tg).numpy().ravel()
        y_pred = model(tt).numpy().ravel()
    return model, hist, y_nodes, y_pred


def main():
    ex = Example("ch12_further_classes", "fractional_relaxation",
                 description="Caputo relaxation D^alpha y = -y: fPINN with the L1 scheme",
                 default_seed=1234)
    t_test = np.linspace(0.0, T_END, 1001)
    grids = ex.pick(GRIDS, [16, 32])

    # ------------------------------------------------------------ reference and its verification
    ref = {a: ml_relaxation(a, t_test) for a in ALPHAS}
    t_chk = np.linspace(0.05, T_END, 40)
    verification = {
        "alpha0.5_vs_erfcx_max_abs": float(np.max(np.abs(ml_relaxation(0.5, t_chk) - erfcx(np.sqrt(t_chk))))),
        "alpha0.8_vs_integral_max_abs": float(np.max(np.abs(
            ml_relaxation(0.8, t_chk) - np.array([ml_relaxation_integral(0.8, s) for s in t_chk])))),
        "alpha1.0_vs_exp_max_abs": float(np.max(np.abs(ml_relaxation(1.0, t_chk) - np.exp(-t_chk)))),
    }
    ex.log(f"Mittag-Leffler verification: {verification}")

    rows, keep = [], {}
    for alpha in ALPHAS:
        for graded in (False, True):
            for n in grids:
                tg = make_grid(n, alpha, graded)
                y_ex_nodes = ml_relaxation(alpha, tg)
                y_fd = l1_solve(tg, alpha)
                verbose = (n == grids[-1])
                model, hist, y_nodes, y_pred = train_fpinn(ex, alpha, tg, t_test, ref[alpha], verbose)
                mesh = "graded" if graded else "uniform"
                row = {"alpha": alpha, "mesh": mesh, "N": n,
                       "pinn_nodal_max_err": float(np.max(np.abs(y_nodes - y_ex_nodes))),
                       "fd_nodal_max_err": float(np.max(np.abs(y_fd - y_ex_nodes))),
                       "pinn_minus_fd_max": float(np.max(np.abs(y_nodes - y_fd))),
                       "pinn_rel_l2": rel_l2(y_pred, ref[alpha]), "pinn_linf": linf(y_pred, ref[alpha]),
                       "final_loss": hist.final_loss, "train_time_s": hist.train_time}
                rows.append(row)
                ex.log(f"alpha={alpha} {mesh:7s} N={n:4d}  PINN nodal={row['pinn_nodal_max_err']:.2e} "
                       f"L1-FD nodal={row['fd_nodal_max_err']:.2e} |PINN-FD|={row['pinn_minus_fd_max']:.1e} "
                       f"relL2={row['pinn_rel_l2']:.2e} t={hist.train_time:.1f}s")
                keep[(alpha, mesh, n)] = {"y_pred": y_pred, "hist": hist, "t_grid": tg}
                if n == grids[-1]:
                    ex.save_history(hist, f"alpha{alpha}_{mesh}_N{n}")
    ex.save_table(rows, "convergence")

    # observed convergence rates (least squares slope of log error vs log N)
    rates = {}
    for alpha in ALPHAS:
        for mesh in ("uniform", "graded"):
            sel = [r for r in rows if r["alpha"] == alpha and r["mesh"] == mesh]
            if len(sel) >= 2:
                lx = np.log([r["N"] for r in sel])
                for key in ("pinn_nodal_max_err", "fd_nodal_max_err"):
                    ly = np.log([r[key] for r in sel])
                    rates[f"alpha{alpha}_{mesh}_{key}"] = float(-np.polyfit(lx, ly, 1)[0])
    ex.save_results({
        "problem": {"equation": "Caputo D^alpha y = -y, y(0) = 1", "T": T_END, "alphas": ALPHAS,
                    "exact": "E_alpha(-t^alpha)", "trial_solution": "y = 1 + t^alpha N(t)"},
        "hyperparameters": {"width": 20, "depth": 2, "activation": "tanh",
                            "adam_iters": ex.pick(2500, 40), "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters": ex.pick(500, 10), "grids": grids,
                            "graded_exponent": "r = (2 - alpha)/alpha"},
        "mittag_leffler_verification": verification,
        "theory_rates": {"uniform": "alpha", "graded": "2 - alpha"},
        "observed_rates": rates,
        "convergence": rows,
        "train_time_s": {f"alpha{r['alpha']}_{r['mesh']}_N{r['N']}": r["train_time_s"] for r in rows},
        "test_grid": "1001 uniform points on [0, 2]",
    })
    if ex.args.no_figures:
        return

    # ------------------------------------------------------------ figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    n_show = 64 if 64 in grids else grids[-1]
    ax = axes[0, 0]
    ax.plot(t_test, np.exp(-t_test), color=P.COLORS["light"], lw=1.0, ls="-", label=r"$\alpha=1$: $e^{-t}$")
    for k, alpha in enumerate(ALPHAS):
        P.plot_reference(ax, t_test, ref[alpha], label="exact $E_\\alpha(-t^\\alpha)$" if k == 0 else None)
        P.plot_prediction(ax, t_test, keep[(alpha, "graded", n_show)]["y_pred"],
                          label=f"fPINN, $\\alpha={alpha}$", k=k, markevery=(60 * k, 120))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for k, (alpha, mesh) in enumerate([(0.5, "uniform"), (0.5, "graded"), (0.8, "uniform"), (0.8, "graded")]):
        P.plot_error(ax, t_test[1:], (keep[(alpha, mesh, n_show)]["y_pred"] - ref[alpha])[1:],
                     label=f"$\\alpha={alpha}$, {mesh}", k=k)     # y(0) = 1 exactly: t > 0 shown
    ax.set_title(f"fPINN error, $N={n_show}$")
    ax.legend(loc="upper right", fontsize=6.5)

    for ax, alpha in zip(axes[1], ALPHAS):
        for k, mesh in enumerate(("uniform", "graded")):
            sel = [r for r in rows if r["alpha"] == alpha and r["mesh"] == mesh]
            Ns = [r["N"] for r in sel]
            ax.loglog(Ns, [r["fd_nodal_max_err"] for r in sel], color=P.SERIES[k],
                      ls=P.LINESTYLES[k], lw=1.1, label=f"L1 finite differences, {mesh}")
            ax.loglog(Ns, [r["pinn_nodal_max_err"] for r in sel], ls="none", marker=P.MARKERS[k],
                      ms=4.0, mfc="white", mec=P.SERIES[k], mew=0.9, label=f"fPINN, {mesh}")
        Ns = np.array(grids, float)
        for k, (rate, txt) in enumerate(((alpha, r"$N^{-\alpha}$"), (2 - alpha, r"$N^{-(2-\alpha)}$"))):
            e0 = [r for r in rows if r["alpha"] == alpha and r["mesh"] == ("uniform", "graded")[k]][0]
            guide = e0["fd_nodal_max_err"] * 0.3 * (Ns / Ns[0]) ** (-rate)
            ax.loglog(Ns, guide, color=P.COLORS["muted"], lw=0.6, ls="-")
            ax.text(Ns[-1] * 1.08, guide[-1], txt, fontsize=7, va="center", color=P.COLORS["muted"])
        ax.set_xlabel("grid points $N$")
        ax.set_ylabel("max nodal error")
        ax.set_title(f"$\\alpha={alpha}$")
        ax.set_xlim(Ns[0] * 0.8, Ns[-1] * 2.2)
        ax.set_xticks(grids)
        ax.set_xticklabels([str(n) for n in grids])
        ax.xaxis.set_minor_formatter(NullFormatter())
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=6.5)
    fig.get_layout_engine().set(h_pad=0.06)
    P.label_panels(axes)
    ex.savefig(fig, "fractional_summary")


if __name__ == "__main__":
    main()
