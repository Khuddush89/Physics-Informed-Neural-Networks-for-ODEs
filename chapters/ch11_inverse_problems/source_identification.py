"""Identification of an unknown time-dependent source (input rate) from noisy concentrations.

Problem (one-compartment model with known elimination rate k and unknown input rate f)
    y'(t) = -k y(t) + f(t),  y(0) = 0,  t in [0, 12],  k = 0.5,
    f(t) = 2 exp(-(t - 2)^2 / 0.5) + exp(-(t - 6)^2 / 2)   (two absorption pulses);
    reference y by solve_ivp (DOP853, rtol = atol = 1e-12).
Data
    N = 40 equally spaced times in (0, 12], additive Gaussian noise with sigma = 0.02 max|y(t_i)|;
    an independent validation set of 40 points at the midpoints (same noise level) plays the role
    of held-out data.
Method
    Two networks (MLP 1-24-24-24-1, tanh): y_theta(t) = (1 - exp(-t)) 1.5 N_theta(t) satisfies y(0) = 0,
    f_phi(t) = 2 M_phi(t). Loss = data misfit + residual + lam_s * mean((f_phi'/2)^2) (an H^1
    smoothness penalty, i.e. Tikhonov regularization of the source) at M = 200 collocation points;
    Adam (3000 iterations, lr 5e-3 -> 5e-5) followed by L-BFGS (at most 700 iterations).
    lam_s in {0, 1e-3, 1e-2, 1e-1, 1, 10} for the noisy data, and lam_s = 0 for noise-free data
    (resolution limit set by the sampling alone).

Book: Chapter 11, Section "Identification of a time-dependent source" (Figure inv:fig:source,
      Table inv:tab:source).
Run:  PYTHONPATH=. python chapters/ch11_inverse_problems/source_identification.py
      (about 3 min CPU on one core; --quick for a smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnode import Example, MLP, derivative, rel_l2, to_tensor, train, uniform  # noqa: E402

from _common import (save_rows_csv, save_history_csv, CPUTimer, RESUME_ARG, add_noise, cached, data_rng, fmt_num, fmt_time,  # noqa: E402
                     write_tabular)

KEL, T, N, ETA = 0.5, 12.0, 40, 0.02
SY, SF = 1.5, 2.0
LAMBDAS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def f_true(t):
    t = np.asarray(t, dtype=float)
    return 2.0 * np.exp(-(t - 2.0) ** 2 / 0.5) + np.exp(-(t - 6.0) ** 2 / 2.0)


def reference():
    sol = solve_ivp(lambda t, y: -KEL * y + f_true(t), (0.0, T), [0.0], dense_output=True,
                    method="DOP853", rtol=1e-12, atol=1e-12)
    return lambda t: sol.sol(np.asarray(t, float))[0]


def fit_source(t_obs, d_obs, t_val, lam_s, seed, adam_iters, lbfgs_iters, width=24, depth=3, n_col=200):
    torch.manual_seed(seed)
    net_y = MLP(1, 1, width=width, depth=depth, activation="tanh", domain=(0.0, T))
    net_f = MLP(1, 1, width=width, depth=depth, activation="tanh", domain=(0.0, T))

    def y_th(t):
        return (1.0 - torch.exp(-t)) * SY * net_y(t)

    def f_th(t):
        return SF * net_f(t)

    t_col = to_tensor(uniform(n_col, 0.0, T), requires_grad=True)
    t_d, d = to_tensor(t_obs), to_tensor(d_obs)

    def loss_fn():
        y, f = y_th(t_col), f_th(t_col)
        res = (derivative(y, t_col) + KEL * y - f) / SF
        L_r = (res ** 2).mean()
        L_d = (((y_th(t_d) - d) / SY) ** 2).mean()
        L_s = ((derivative(f, t_col) / SF) ** 2).mean()
        loss = L_d + L_r + lam_s * L_s if lam_s > 0 else L_d + L_r
        return loss, {"data": L_d, "residual": L_r, "smoothness": L_s}

    with CPUTimer() as tm:
        hist = train(list(net_y.parameters()) + list(net_f.parameters()), loss_fn, adam_iters=adam_iters,
                     lr=5e-3, lbfgs_iters=lbfgs_iters, verbose=False)
    t_grid = np.linspace(0.0, T, 1001)
    with torch.no_grad():
        tg = to_tensor(t_grid)
        out = {"t_grid": t_grid, "y": y_th(tg).numpy().ravel(), "f": f_th(tg).numpy().ravel(),
               "y_obs": y_th(t_d).numpy().ravel(), "y_val": y_th(to_tensor(t_val)).numpy().ravel()}
    out.update({"hist": hist, "cpu_s": tm.cpu, "wall_s": hist.train_time,
                "rms_residual": float(np.sqrt(hist.parts["residual"][-1])) * SF})
    return out


def main():
    ex = Example("ch11_inverse_problems", "source_identification",
                 description="Unknown source f(t) in y' = -k y + f(t) from noisy data (two networks)",
                 default_seed=1234, extra_args=[RESUME_ARG])
    ref = reference()
    t_obs = np.linspace(0.0, T, N + 1)[1:]
    t_val = t_obs - 0.5 * (t_obs[1] - t_obs[0])
    y_obs, y_val_true = ref(t_obs), ref(t_val)
    d_obs, sigma = add_noise(y_obs, ETA, data_rng(19, ETA, 0))
    d_val = y_val_true + sigma * data_rng(119, ETA, 0).standard_normal(N)
    t = np.linspace(0.0, T, 1001)
    y_ref, f_ref = ref(t), f_true(t)
    adam, lbfgs = ex.pick(3000, 60), ex.pick(700, 10)
    lambdas = LAMBDAS if not ex.quick else [0.0, 1e-1]

    runs = []
    cases = [("noise-free", 0.0, y_obs)] + [("noisy", lam, d_obs) for lam in lambdas]
    for kind, lam, data in cases:
        kw = dict(lam_s=lam, seed=ex.seed, adam_iters=adam, lbfgs_iters=lbfgs)
        fit = cached(ex, f"{kind}_lam{lam:g}", lambda: fit_source(t_obs, data, t_val, **kw), kind=kind, **kw)
        e = fit["f"] - f_ref
        tail = t >= 10.0
        peak_window = (t >= 1.0) & (t <= 3.0)
        row = {"data": kind, "lam_s": lam, "rel_l2_f": rel_l2(fit["f"], f_ref), "linf_f": float(np.max(np.abs(e))),
               "linf_f_interior": float(np.max(np.abs(e[t <= 10.0]))), "linf_f_tail": float(np.max(np.abs(e[tail]))),
               "peak1_height": float(np.max(fit["f"][peak_window])), "rel_l2_y": rel_l2(fit["y"], y_ref),
               "rms_train_misfit": float(np.sqrt(np.mean((fit["y_obs"] - data) ** 2))),
               "rms_val_misfit": float(np.sqrt(np.mean((fit["y_val"] - (d_val if kind == "noisy" else y_val_true)) ** 2))),
               "rms_residual": fit["rms_residual"], "cpu_s": fit["cpu_s"], "wall_s": fit["wall_s"]}
        runs.append({**row, "fit": fit})
        ex.log(f"{kind:10s} lam_s={lam:7.0e}: relL2 f={row['rel_l2_f']:.3e} max|f err|={row['linf_f']:.3f} "
               f"peak={row['peak1_height']:.3f} train={row['rms_train_misfit']:.4f} val={row['rms_val_misfit']:.4f} "
               f"(sigma={sigma:.4f}) ({fit['cpu_s']:.1f}s CPU)")
    rows = [{k: v for k, v in r.items() if k != "fit"} for r in runs]
    save_rows_csv(ex.path("sweep.csv"), rows)
    noisy = [r for r in runs if r["data"] == "noisy"]
    best_val = min(noisy, key=lambda r: r["rms_val_misfit"])
    best_err = min(noisy, key=lambda r: r["rel_l2_f"])
    # discrepancy principle: largest lam_s with training misfit not above sigma
    admissible = [r for r in noisy if r["rms_train_misfit"] <= sigma]
    discrepancy = max(admissible, key=lambda r: r["lam_s"]) if admissible else None
    ex.log(f"validation choice lam_s={best_val['lam_s']:g}, error-optimal lam_s={best_err['lam_s']:g}, "
           f"discrepancy choice lam_s={discrepancy['lam_s'] if discrepancy else None}")

    body = []
    for r in runs:
        lab = "noise-free" if r["data"] == "noise-free" else "noisy"
        lam = "0" if r["lam_s"] == 0 else fmt_num(r["lam_s"], 0)
        body.append([lab, lam, fmt_num(r["rel_l2_f"]), f"{r['linf_f']:.3f}", f"{r['peak1_height']:.2f}",
                     f"{r['rms_train_misfit']:.4f}", f"{r['rms_val_misfit']:.4f}", fmt_time(r["cpu_s"])])
        if r["data"] == "noise-free":
            body.append("midrule")
    write_tabular(ex.path("tab_source.tex"), "lccccccr",
                  ["data", r"$\lambda_s$", r"rel.\ $L^2$ error", r"max.\ error", "peak",
                   "RMS misfit", "RMS valid.", "CPU (s)"], body,
                  comment="chapters/ch11_inverse_problems/source_identification.py")
    ex.save_results({
        "problem": {"k": KEL, "T": T, "N": N, "eta": ETA, "sigma": sigma,
                    "f": "2 exp(-(t-2)^2/0.5) + exp(-(t-6)^2/2)", "y0": 0.0,
                    "validation": "40 points at midpoints, independent noise"},
        "hyperparameters": {"width": 24, "depth": 3, "networks": 2, "collocation": 200, "adam_iters": adam,
                            "lr": "5e-3 -> 5e-5 (cosine)", "lbfgs_iters": lbfgs,
                            "penalty": "lam_s * mean((f'/2)^2)", "lam_r": 1.0},
        "rows": rows, "validation_choice": best_val["lam_s"], "error_optimal": best_err["lam_s"],
        "discrepancy_choice": discrepancy["lam_s"] if discrepancy else None})

    if ex.args.no_figures:
        return
    from pinnode import plotting as P   # imported late: keeps the training footprint small

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, t, y_ref, label="exact")
    P.plot_prediction(ax, t, best_val["fit"]["y"], label=rf"PINN, $\lambda_s={best_val['lam_s']:g}$", k=0)
    ax.plot(t_obs, d_obs, ls="none", marker="o", ms=3.4, mfc="white", mec=P.COLORS["ref"], mew=0.8,
            label="data", zorder=4)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(r"concentration ($2\,\%$ noise)")
    ax.set_ylim(-0.1, 1.45 * np.max(y_ref))
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax = axes[0, 1]
    P.plot_reference(ax, t, f_ref, label="exact $f$")
    show = [r for r in noisy if r["lam_s"] == 0.0] + [best_val] + ([discrepancy] if discrepancy else []) + \
        [r for r in noisy if r["lam_s"] == 0.1]
    seen, k = set(), 0
    for r in show:
        if r["lam_s"] in seen:
            continue
        seen.add(r["lam_s"])
        P.plot_prediction(ax, t, r["fit"]["f"], label=rf"$\lambda_s={r['lam_s']:g}$", k=k, lw=1.1)
        k += 1
    ax.set_ylim(-0.1, 1.5 * np.max(f_ref))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$f(t)$")
    ax.set_title("reconstructed source")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    lam_plot = np.array([max(r["lam_s"], 1e-4) for r in noisy])
    ax = axes[1, 0]
    ax.loglog(lam_plot, [r["rel_l2_f"] for r in noisy], color=P.SERIES[0], ls="--", marker="o", mfc="white",
              ms=3.8, lw=1.2, label=r"noisy data")
    nf = runs[0]
    ax.axhline(nf["rel_l2_f"], color=P.COLORS["ref"], lw=0.8, ls=":", label=r"noise-free data, $\lambda_s=0$")
    ax.set_xlabel(r"smoothness weight $\lambda_s$ ($\lambda_s=0$ plotted at $10^{-4}$)")
    ax.set_ylabel(r"relative $L^2$ error of $f$")
    ax.set_title("source error")
    ax.legend(loc="upper left", fontsize=7)
    ax = axes[1, 1]
    ax.loglog(lam_plot, [r["rms_train_misfit"] for r in noisy], color=P.SERIES[0], ls="--", marker="o",
              mfc="white", ms=3.8, lw=1.2, label="training data")
    ax.loglog(lam_plot, [r["rms_val_misfit"] for r in noisy], color=P.SERIES[1], ls="-.", marker="s",
              mfc="white", ms=3.8, lw=1.2, label="validation data")
    ax.axhline(sigma, color=P.COLORS["ref"], lw=0.9, label=r"noise level $\sigma$")
    ax.set_xlabel(r"smoothness weight $\lambda_s$")
    ax.set_ylabel("RMS misfit")
    ax.set_title("misfit to the data")
    ax.legend(loc="upper left", fontsize=7)
    P.label_panels(axes)
    ex.savefig(fig, "source_identification")


if __name__ == "__main__":
    main()
