"""Parameter identification in the logistic ODE: inverse PINN versus classical least squares.

Problem
    y' = r y (1 - y/K),  y(0) = y0 = 0.5  on [0, T], T = 10, true (r, K) = (0.8, 10);
    exact solution y(t) = K / (1 + (K/y0 - 1) exp(-r t)).
Data
    N = 15 equally spaced observation times in (0, 10]; additive Gaussian noise with standard
    deviation sigma = eta * max_i |y(t_i)|, eta in {0, 0.01, 0.05, 0.10}; five independent noise
    realizations per level (for eta = 0 the five runs differ only in the network initialization).
Methods (same initial guess (r0, K0) = (0.3, 5) for both)
    * inverse PINN: y_theta = y0 + (t/T) 10 N_theta(t), N_theta an MLP 1-20-20-1 (tanh),
      unknowns (theta, log r, log K), loss = data misfit + lam_r * residual (lam_r = 1),
      M = 100 uniform collocation points, Adam (2000 iterations, lr 5e-3 -> 5e-5, cosine)
      followed by L-BFGS (at most 500 iterations);
    * classical output least squares: scipy.optimize.least_squares over (log r, log K) with
      the model values computed by solve_ivp (DOP853, rtol = 1e-10, atol = 1e-12).
Practical identifiability
    linearized (Gauss-Newton) standard deviations of (r, K) for observations on (0, 10] and on
    the early window (0, 3], plus one fit of each method on early-window data (eta = 0.05).

Book: Chapter 11, Section "Parameter identification in a scalar ODE" (Figure inv:fig:logistic,
      Table inv:tab:logistic).
Run:  PYTHONPATH=. python chapters/ch11_inverse_problems/logistic_parameter_id.py
      (about 2-3 min CPU on one core; --quick for a smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnode import Example, rel_l2, linf  # noqa: E402

from _common import (save_rows_csv, save_history_csv, LOGISTIC, LOGISTIC_P_INIT, RESUME_ARG, cached, fit_logistic_nls,  # noqa: E402
                     fit_logistic_pinn, fmt_num, fmt_pm, fmt_time, gauss_newton_covariance, logistic_data,
                     logistic_exact, logistic_obs_times, logistic_sensitivities, median,
                     write_tabular)

ETAS = [0.0, 0.01, 0.05, 0.10]


def state_errors(y_grid=None, r=None, K=None):
    """Errors of a state on the uniform 1001-point grid of [0, T]: the PINN prediction
    ``y_grid`` or, if it is None, the classical curve y(t; r, K)."""
    t = np.linspace(0.0, LOGISTIC["T"], 1001)
    y_true = logistic_exact(t, LOGISTIC["r"], LOGISTIC["K"], LOGISTIC["y0"])
    y = y_grid if y_grid is not None else logistic_exact(t, r, K, LOGISTIC["y0"])
    return {"rel_l2": rel_l2(y, y_true), "linf": linf(y, y_true)}


def linearized_sd(t_obs, eta):
    """Relative Gauss-Newton standard deviations of (r, K) at the true parameters."""
    p = LOGISTIC
    S = logistic_sensitivities(t_obs, p["r"], p["K"], p["y0"])
    y = logistic_exact(t_obs, p["r"], p["K"], p["y0"])
    sigma = eta * np.max(np.abs(y))
    cov = gauss_newton_covariance(S, sigma)
    sd = np.sqrt(np.diag(cov))
    S_rel = S * np.array([p["r"], p["K"]])[None, :]          # relative sensitivities
    sv = np.linalg.svd(S_rel, compute_uv=False)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    return {"sd_r_rel": sd[0] / p["r"], "sd_K_rel": sd[1] / p["K"], "corr_rK": corr,
            "sv_rel_sensitivity": sv.tolist(), "cond_rel_sensitivity": sv[0] / sv[-1], "sigma": sigma}


def main():
    ex = Example("ch11_inverse_problems", "logistic_parameter_id",
                 description="Logistic ODE: identify (r, K) from noisy data, PINN vs least squares",
                 default_seed=1234, extra_args=[RESUME_ARG])
    p_true = (LOGISTIC["r"], LOGISTIC["K"])
    n_real = ex.pick(5, 2)
    pinn_kw = dict(adam_iters=ex.pick(2000, 60), lbfgs_iters=ex.pick(500, 10), lr=5e-3, lam_r=1.0,
                   width=20, depth=2, n_col=100, p_true=p_true, eval_every=ex.pick(10, 10))
    rows, keep = [], {}
    for eta in ETAS:
        for k in range(n_real):
            t_obs, d_obs, sigma = logistic_data(eta, k)
            pinn = cached(ex, f"pinn_eta{eta}_k{k}",
                          lambda: fit_logistic_pinn(t_obs, d_obs, seed=ex.seed + k, **pinn_kw),
                          seed=ex.seed + k, **pinn_kw)
            nls = cached(ex, f"nls_eta{eta}_k{k}", lambda: fit_logistic_nls(t_obs, d_obs))
            se_p = state_errors(pinn["y_grid"])
            se_n = state_errors(None, *nls["params"])
            for name, fit, se in (("PINN", pinn, se_p), ("NLS", nls, se_n)):
                r, K = fit["params"]
                rows.append({"eta": eta, "realization": k, "method": name, "r": r, "K": K,
                             "err_r": abs(r / p_true[0] - 1), "err_K": abs(K / p_true[1] - 1),
                             "rms_misfit": fit["rms_misfit"], "sigma": sigma,
                             "state_rel_l2": se["rel_l2"], "state_linf": se["linf"],
                             "cpu_s": fit["cpu_s"], "wall_s": fit["wall_s"],
                             "iterations": (pinn["hist"].iters[-1] if name == "PINN" else nls["n_forward_solves"])})
            ex.log(f"eta={eta:.2f} k={k}: PINN r={pinn['params'][0]:.5f} K={pinn['params'][1]:.4f} "
                   f"({pinn['cpu_s']:.1f}s CPU) | NLS r={nls['params'][0]:.5f} K={nls['params'][1]:.4f} "
                   f"({nls['cpu_s']:.2f}s CPU, {nls['n_forward_solves']} solves)")
            if k == 0:
                keep[eta] = {"t_obs": t_obs, "d_obs": d_obs, "pinn": pinn, "nls": nls}
                save_history_csv(ex.path("history_" + f"pinn_eta{int(round(100 * eta)):02d}" + ".csv"), pinn["hist"])
    save_rows_csv(ex.path("runs.csv"), rows)

    # ---------------------------------------------------------------- summary per noise level
    summary = []
    for eta in ETAS:
        for name in ("PINN", "NLS"):
            sel = [r for r in rows if r["eta"] == eta and r["method"] == name]
            rr = np.array([s["r"] for s in sel])
            KK = np.array([s["K"] for s in sel])
            summary.append({
                "eta": eta, "method": name, "r_mean": rr.mean(), "r_sd": rr.std(ddof=1) if len(rr) > 1 else 0.0,
                "K_mean": KK.mean(), "K_sd": KK.std(ddof=1) if len(KK) > 1 else 0.0,
                "mean_err_r": float(np.mean([s["err_r"] for s in sel])),
                "mean_err_K": float(np.mean([s["err_K"] for s in sel])),
                "max_err_r": float(np.max([s["err_r"] for s in sel])),
                "max_err_K": float(np.max([s["err_K"] for s in sel])),
                "mean_state_rel_l2": float(np.mean([s["state_rel_l2"] for s in sel])),
                "median_cpu_s": median(s["cpu_s"] for s in sel),
                "median_wall_s": median(s["wall_s"] for s in sel)})
    save_rows_csv(ex.path("summary.csv"), summary)
    cpu_pinn = median(r["cpu_s"] for r in rows if r["method"] == "PINN")
    cpu_nls = median(r["cpu_s"] for r in rows if r["method"] == "NLS")
    # PINN versus NLS estimates on identical data (how close are the two estimators?)
    gaps = []
    for eta in ETAS[1:]:
        for k in range(n_real):
            a = [r for r in rows if r["eta"] == eta and r["realization"] == k]
            pn = next(r for r in a if r["method"] == "PINN")
            nl = next(r for r in a if r["method"] == "NLS")
            gaps.append({"eta": eta, "k": k, "gap_r": abs(pn["r"] - nl["r"]) / p_true[0],
                         "gap_K": abs(pn["K"] - nl["K"]) / p_true[1]})
    save_rows_csv(ex.path("pinn_vs_nls_gap.csv"), gaps)

    # ---------------------------------------------------------------- practical identifiability
    t_full = logistic_obs_times(LOGISTIC["T"], LOGISTIC["N"])
    t_early = logistic_obs_times(3.0, LOGISTIC["N"])
    ident = {"full_window": linearized_sd(t_full, 0.05), "early_window": linearized_sd(t_early, 0.05)}
    t_e, d_e, _ = logistic_data(0.05, 0, t_obs=t_early)
    pinn_e = cached(ex, "pinn_early", lambda: fit_logistic_pinn(t_e, d_e, seed=ex.seed, **pinn_kw),
                    seed=ex.seed, **pinn_kw)
    nls_e = cached(ex, "nls_early", lambda: fit_logistic_nls(t_e, d_e))
    ident["early_fit"] = {"pinn_r": pinn_e["params"][0], "pinn_K": pinn_e["params"][1],
                          "nls_r": nls_e["params"][0], "nls_K": nls_e["params"][1],
                          "pinn_err_K": abs(pinn_e["params"][1] / p_true[1] - 1),
                          "nls_err_K": abs(nls_e["params"][1] / p_true[1] - 1),
                          "pinn_err_r": abs(pinn_e["params"][0] / p_true[0] - 1),
                          "nls_err_r": abs(nls_e["params"][0] / p_true[0] - 1)}
    ex.log(f"identifiability: full window sd(r)/r={ident['full_window']['sd_r_rel']:.3f} "
           f"sd(K)/K={ident['full_window']['sd_K_rel']:.3f}; early window sd(r)/r="
           f"{ident['early_window']['sd_r_rel']:.3f} sd(K)/K={ident['early_window']['sd_K_rel']:.3f}")
    ex.log(f"early-window fits: PINN K={pinn_e['params'][1]:.3f}, NLS K={nls_e['params'][1]:.3f}")

    # ---------------------------------------------------------------- LaTeX table
    body = []
    for eta in ETAS:
        for j, name in enumerate(("PINN", "NLS")):
            s = next(x for x in summary if x["eta"] == eta and x["method"] == name)
            lab = (rf"\multirow{{2}}{{*}}{{{int(round(100 * eta))}\,\%}}" if j == 0 else "")
            meth = "PINN" if name == "PINN" else "least squares"
            body.append([lab, meth, fmt_pm(s["r_mean"], s["r_sd"], 4), fmt_pm(s["K_mean"], s["K_sd"], 3),
                         fmt_num(s["mean_err_r"]), fmt_num(s["mean_err_K"]), fmt_time(s["median_cpu_s"])])
        if eta != ETAS[-1]:
            body.append("midrule")
    write_tabular(ex.path("tab_logistic.tex"), "llccccr",
                  [r"$\eta$", "method", r"$\hat r$ (mean $\pm$ sd)", r"$\hat K$ (mean $\pm$ sd)",
                   r"mean $|\hat r - r|/r$", r"mean $|\hat K - K|/K$", "CPU (s)"], body,
                  comment="chapters/ch11_inverse_problems/logistic_parameter_id.py")

    ex.save_results({
        "problem": {**LOGISTIC, "p_init": LOGISTIC_P_INIT, "noise": "additive Gaussian, sd = eta*max|y(t_i)|",
                    "etas": ETAS, "realizations": n_real},
        "hyperparameters": {**{k: v for k, v in pinn_kw.items() if k != "p_true"},
                            "lr_schedule": "cosine 5e-3 -> 5e-5", "activation": "tanh",
                            "output_scale": 10.0, "nls": "least_squares(trf) + solve_ivp(DOP853, 1e-10/1e-12), FD jac step 1e-6"},
        "summary": summary, "median_cpu_s": {"PINN": cpu_pinn, "NLS": cpu_nls, "ratio": cpu_pinn / cpu_nls},
        "max_gap_pinn_nls": {"r": max(g["gap_r"] for g in gaps), "K": max(g["gap_K"] for g in gaps)},
        "identifiability": ident,
        "timing_note": "CPU time of the process (1 thread); wall-clock times were inflated by a shared machine"})

    if ex.args.no_figures:
        return
    from pinnode import plotting as P   # imported late: keeps the training footprint small

    # ---------------------------------------------------------------- figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    t = np.linspace(0.0, LOGISTIC["T"], 1001)
    kp = keep[0.10]
    P.plot_reference(ax, t, logistic_exact(t, *p_true, LOGISTIC["y0"]), label="exact")
    P.plot_prediction(ax, t, kp["pinn"]["y_grid"], label="PINN", k=0)
    P.plot_prediction(ax, t, logistic_exact(t, *kp["nls"]["params"], LOGISTIC["y0"]), label="least squares", k=1)
    ax.plot(kp["t_obs"], kp["d_obs"], ls="none", marker="o", ms=3.8, mfc="white", mec=P.COLORS["ref"],
            mew=0.8, label="data", zorder=4)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(r"data with $\eta = 10\,\%$")
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    styles = {(0.0, "r"): (P.SERIES[0], "-"), (0.0, "K"): (P.SERIES[1], "-"),
              (0.05, "r"): (P.SERIES[0], "--"), (0.05, "K"): (P.SERIES[1], "--")}
    for eta in (0.0, 0.05):
        h = keep[eta]["pinn"]["hist"]
        for par in ("r", "K"):
            c, ls = styles[(eta, par)]
            ax.semilogy(h.eval_iters, np.maximum(h.metrics[f"err_{par}"], 1e-12), color=c, ls=ls, lw=1.1,
                        label=rf"${par}$, $\eta={int(100 * eta)}\,\%$")
    for par, c in (("r", P.SERIES[0]), ("K", P.SERIES[1])):
        e_nls = abs(keep[0.05]["nls"]["params"][0 if par == "r" else 1] / p_true[0 if par == "r" else 1] - 1)
        ax.axhline(e_nls, color=c, lw=0.7, ls=":")
    sw = keep[0.0]["pinn"]["hist"].switch_iter
    if sw:
        ax.axvline(sw, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative parameter error")
    ax.set_title("convergence of the estimates")
    lo = min(min(keep[e]["pinn"]["hist"].metrics[f"err_{q}"]) for e in (0.0, 0.05) for q in ("r", "K"))
    ax.set_ylim(max(0.3 * lo, 1e-12), 80.0)        # head room for the legend
    ax.legend(loc="upper right", ncol=2)

    for ax, par in ((axes[1, 0], "r"), (axes[1, 1], "K")):
        for j, (name, mk) in enumerate((("PINN", "o"), ("NLS", "s"))):
            xs, ys = [], []
            for i, eta in enumerate(ETAS):
                sel = [r for r in rows if r["eta"] == eta and r["method"] == name]
                xs += [i + (j - 0.5) * 0.22] * len(sel)
                ys += [max(s[f"err_{par}"], 1e-12) for s in sel]
            ax.semilogy(xs, ys, ls="none", marker=mk, ms=3.6, mfc="white", mec=P.SERIES[j], mew=0.9,
                        label="PINN" if name == "PINN" else "least squares")
        ax.set_xticks(range(len(ETAS)))
        ax.set_xticklabels([f"{int(round(100 * e))}%" for e in ETAS])
        ax.set_xlabel(r"noise level $\eta$")
        ax.set_ylabel(rf"$|\hat {par} - {par}|/{par}$")
        ax.set_title(f"error in ${par}$ ({n_real} data sets per level)")
        ax.legend(loc="lower right")
    P.label_panels(axes)
    ex.savefig(fig, "logistic_parameter_id")


if __name__ == "__main__":
    main()
