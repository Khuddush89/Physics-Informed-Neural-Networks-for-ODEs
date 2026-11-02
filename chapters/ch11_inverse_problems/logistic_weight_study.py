"""Choosing the residual weight lam_r in an inverse PINN (logistic model, 5 % noise).

Problem and data
    y' = r y (1 - y/K), y(0) = 0.5 on [0, 10], true (r, K) = (0.8, 10); the data set is
    realization 0 at noise level eta = 0.05 of Section 11.2 (N = 15 points, additive Gaussian
    noise, sigma = 0.05 max|y(t_i)|). An independent validation set (N = 15 points at the
    midpoints between the training times, same noise level) plays the role of held-out data.
Study
    The inverse PINN of Section 11.2 is trained for lam_r in {1e-4, ..., 1e3}. For each weight
    we record the parameter errors, the distance to the classical least-squares estimate, the
    RMS misfit to the training data (to be compared with sigma: discrepancy principle), the RMS
    misfit to the validation data and the RMS of the scaled residual.

Book: Chapter 11, Sections "Formulation" (choice of lam_r) and "Practical guidance"
      (Figure inv:fig:weights).
Run:  PYTHONPATH=. python chapters/ch11_inverse_problems/logistic_weight_study.py
      (about 1 min CPU on one core; --quick for a smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnode import Example  # noqa: E402

from _common import (save_rows_csv, save_history_csv, LOGISTIC, RESUME_ARG, add_noise, cached, data_rng, fit_logistic_nls,  # noqa: E402
                     fit_logistic_pinn, logistic_data, logistic_exact)

LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3]
ETA = 0.05


def main():
    ex = Example("ch11_inverse_problems", "logistic_weight_study",
                 description="Logistic inverse PINN: effect of the residual weight lam_r",
                 default_seed=1234, extra_args=[RESUME_ARG])
    p_true = np.array([LOGISTIC["r"], LOGISTIC["K"]])
    t_obs, d_obs, sigma = logistic_data(ETA, 0)
    # independent validation data at the midpoints of the training times
    dt = t_obs[1] - t_obs[0]
    t_val = t_obs - 0.5 * dt
    y_val = logistic_exact(t_val, LOGISTIC["r"], LOGISTIC["K"], LOGISTIC["y0"])
    d_val, _ = add_noise(y_val, ETA, data_rng(111, ETA, 0))
    nls = fit_logistic_nls(t_obs, d_obs)
    ex.log(f"least squares: r={nls['params'][0]:.5f} K={nls['params'][1]:.4f} rms={nls['rms_misfit']:.4f} "
           f"sigma={sigma:.4f}")
    y_nls_val = logistic_exact(t_val, *nls["params"], LOGISTIC["y0"])
    nls_val = float(np.sqrt(np.mean((y_nls_val - d_val) ** 2)))

    lambdas = LAMBDAS if not ex.quick else [1e-2, 1.0, 1e2]
    rows = []
    for lam in lambdas:
        kw = dict(lam_r=lam, seed=ex.seed, adam_iters=ex.pick(2000, 60), lbfgs_iters=ex.pick(500, 10),
                  lr=5e-3, p_true=tuple(p_true))
        fit = cached(ex, f"pinn_lam{lam:g}", lambda: fit_logistic_pinn(t_obs, d_obs, t_extra=t_val, **kw), **kw)
        val = float(np.sqrt(np.mean((fit["y_extra"] - d_val) ** 2)))
        p = fit["params"]
        rows.append({"lam_r": lam, "r": p[0], "K": p[1], "err_r": abs(p[0] / p_true[0] - 1),
                     "err_K": abs(p[1] / p_true[1] - 1),
                     "gap_r": abs(p[0] - nls["params"][0]) / p_true[0],
                     "gap_K": abs(p[1] - nls["params"][1]) / p_true[1],
                     "rms_train_misfit": fit["rms_misfit"], "rms_val_misfit": val,
                     "rms_scaled_residual": float(np.sqrt(fit["final_residual_loss"])),
                     "cpu_s": fit["cpu_s"], "wall_s": fit["wall_s"]})
        ex.log(f"lam_r={lam:8.0e}: r={p[0]:.5f} K={p[1]:.4f} misfit={fit['rms_misfit']:.4f} "
               f"val={val:.4f} res={np.sqrt(fit['final_residual_loss']):.2e} ({fit['cpu_s']:.1f}s CPU)")
    save_rows_csv(ex.path("weights.csv"), rows)
    ex.save_results({
        "problem": {**LOGISTIC, "eta": ETA, "sigma": sigma, "data": "logistic_data(0.05, 0)",
                    "validation": "15 points at midpoints, independent noise"},
        "nls": {"r": nls["params"][0], "K": nls["params"][1], "rms_misfit": nls["rms_misfit"],
                "rms_val_misfit": nls_val, "err_r": abs(nls["params"][0] / p_true[0] - 1),
                "err_K": abs(nls["params"][1] / p_true[1] - 1)},
        "hyperparameters": {"width": 20, "depth": 2, "collocation": 100, "adam_iters": ex.pick(2000, 60),
                            "lr": "5e-3 -> 5e-5 (cosine)", "lbfgs_iters": ex.pick(500, 10)},
        "rows": rows})

    if ex.args.no_figures:
        return
    from pinnode import plotting as P   # imported late: keeps the training footprint small

    lam = np.array([r["lam_r"] for r in rows])
    fig, axes = P.figure(1, 2, aspect=0.8)
    ax = axes[0]
    ax.loglog(lam, [max(r["gap_r"], 1e-12) for r in rows], color=P.SERIES[0], ls=P.LINESTYLES[0],
              marker="o", mfc="white", ms=3.8, lw=1.2, label=r"$|\hat r-\hat r_{\mathrm{LS}}|/r$")
    ax.loglog(lam, [max(r["gap_K"], 1e-12) for r in rows], color=P.SERIES[1], ls=P.LINESTYLES[1],
              marker="s", mfc="white", ms=3.8, lw=1.2, label=r"$|\hat K-\hat K_{\mathrm{LS}}|/K$")
    ax.loglog(lam, [r["rms_scaled_residual"] for r in rows], color=P.SERIES[2], ls=P.LINESTYLES[2],
              marker="^", mfc="white", ms=3.8, lw=1.2, label="RMS scaled residual")
    ax.set_xlabel(r"residual weight $\lambda_r$")
    ax.set_ylabel("relative difference / residual")
    ax.set_title("approach to output least squares")
    lo = min(min(max(r["gap_r"], 1e-12), max(r["gap_K"], 1e-12), r["rms_scaled_residual"]) for r in rows)
    ax.set_ylim(0.3 * lo, 3e3)                 # head room for the legend above the data
    ax.legend(loc="upper right", fontsize=7)
    ax = axes[1]
    ax.loglog(lam, [r["rms_train_misfit"] for r in rows], color=P.SERIES[0], ls="--", marker="o", mfc="white",
              ms=3.8, lw=1.2, label="training data")
    ax.loglog(lam, [r["rms_val_misfit"] for r in rows], color=P.SERIES[1], ls="-.", marker="s", mfc="white",
              ms=3.8, lw=1.2, label="validation data")
    ax.axhline(sigma, color=P.COLORS["ref"], lw=0.9, ls="-", label=r"noise level $\sigma$")
    ax.axhline(nls["rms_misfit"], color=P.SERIES[0], lw=0.8, ls=":", label="least squares, training data")
    ax.set_ylim(0.05, 1.0)
    ax.set_xlabel(r"residual weight $\lambda_r$")
    ax.set_ylabel("RMS misfit")
    ax.set_title("misfit to the data")
    ax.legend(loc="lower right", fontsize=7)
    P.label_panels(axes)
    ex.savefig(fig, "logistic_weight_study")


if __name__ == "__main__":
    main()
