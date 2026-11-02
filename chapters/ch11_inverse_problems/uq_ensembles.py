"""Uncertainty quantification for the logistic inverse problem: ensembles versus a reference posterior.

Problem and data
    y' = r y (1 - y/K), y(0) = 0.5 on [0, 10], true (r, K) = (0.8, 10); data = realization 0 at
    noise level eta = 0.05 of Section 11.2 (N = 15 points, additive Gaussian noise with known
    standard deviation sigma = 0.05 max|y(t_i)|).
Methods
    * reference posterior: random-walk Metropolis for (r, K) with a flat prior on the box
      (0, 5) x (0, 100), Gaussian likelihood with known sigma and the closed-form forward model;
      60 000 steps, 10 000 discarded, Gaussian proposal = (2.38^2/2) x linearized covariance;
    * Gauss-Newton (linearized) covariance sigma^2 (J^T J)^{-1} at the least-squares estimate;
    * deep ensemble, initialization only: 10 inverse PINNs (Section 11.2) trained on the same data
      from different random initializations of the network and of (r0, K0);
    * ensemble with perturbed data (randomized maximum likelihood):
      member m is trained on d + sigma xi^(m), xi^(m) ~ N(0, I), with its own initialization.
    For each method: estimate, standard deviation and 95 % interval of r and K, and a pointwise
    95 % band for the state y(t) on 1001 points.

Book: Chapter 11, Section "Uncertainty quantification" (Figure inv:fig:uq, Table inv:tab:uq).
Run:  PYTHONPATH=. python chapters/ch11_inverse_problems/uq_ensembles.py
      (about 2-3 min CPU on one core; --quick for a smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnode import Example  # noqa: E402

from _common import (save_rows_csv, CPUTimer, LOGISTIC, RESUME_ARG, cached, fit_logistic_nls, fit_logistic_pinn,  # noqa: E402
                     fmt_time,
                     gauss_newton_covariance, logistic_data, logistic_exact, logistic_sensitivities,
                     write_tabular)

ETA = 0.05
Z95 = 1.959963984540054


def iact(x: np.ndarray) -> float:
    """Integrated autocorrelation time with Sokal's automatic window (c = 5)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 1.0
    for m in range(1, n):
        tau = 1.0 + 2.0 * acf[1:m + 1].sum()
        if m >= 5.0 * tau:
            break
    return float(tau)


def metropolis(t_obs, d_obs, sigma, p_start, cov, n_steps, rng):
    """Random-walk Metropolis for (r, K) with flat prior on (0, 5) x (0, 100)."""
    y0 = LOGISTIC["y0"]

    def log_post(p):
        r, K = p
        if not (0.0 < r < 5.0 and 0.0 < K < 100.0):
            return -np.inf
        res = logistic_exact(t_obs, r, K, y0) - d_obs
        return -0.5 * float(res @ res) / sigma ** 2

    L = np.linalg.cholesky((2.38 ** 2 / 2.0) * cov)
    chain = np.empty((n_steps, 2))
    p, lp = np.asarray(p_start, float), log_post(p_start)
    acc = 0
    for i in range(n_steps):
        q = p + L @ rng.standard_normal(2)
        lq = log_post(q)
        if np.log(rng.uniform()) < lq - lp:
            p, lp = q, lq
            acc += 1
        chain[i] = p
    return chain, acc / n_steps


def run_ensemble(ex, t_obs, d_obs, sigma, n_members, perturb: bool, base_seed: int, pinn_kw):
    rng = np.random.default_rng(base_seed)
    members = []
    # draw all random inputs first, so that cached members (--resume) do not shift the streams
    inits = [(rng.uniform(0.2, 0.6), rng.uniform(3.0, 15.0)) for _ in range(n_members)]
    noise = [rng.standard_normal(d_obs.shape) for _ in range(n_members)]
    for m in range(n_members):
        p_init = inits[m]
        d = d_obs + sigma * noise[m] if perturb else d_obs
        kw = dict(p_init=p_init, seed=base_seed + m, **pinn_kw)
        fit = cached(ex, f"member_{'perturbed' if perturb else 'init'}_{m}",
                     lambda: fit_logistic_pinn(t_obs, d, **kw), perturb=perturb, **kw)
        members.append({"r": fit["params"][0], "K": fit["params"][1], "y": fit["y_grid"], "cpu_s": fit["cpu_s"],
                        "wall_s": fit["wall_s"], "p_init": p_init})
        ex.log(f"  {'perturbed' if perturb else 'init-only'} member {m}: r={fit['params'][0]:.5f} "
               f"K={fit['params'][1]:.4f} ({fit['cpu_s']:.1f}s CPU)")
    return members


def summarize_samples(r, K):
    out = {}
    for name, v in (("r", r), ("K", K)):
        out[name] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                     "q025": float(np.quantile(v, 0.025)), "q975": float(np.quantile(v, 0.975))}
    return out


def main():
    ex = Example("ch11_inverse_problems", "uq_ensembles",
                 description="Logistic inverse problem: ensembles vs Gauss-Newton vs Metropolis",
                 default_seed=1234, extra_args=[RESUME_ARG])
    p_true = np.array([LOGISTIC["r"], LOGISTIC["K"]])
    y0, T = LOGISTIC["y0"], LOGISTIC["T"]
    t_obs, d_obs, sigma = logistic_data(ETA, 0)
    t = np.linspace(0.0, T, 1001)
    y_true = logistic_exact(t, *p_true, y0)

    # ---------------------------------------------------------------- classical: NLS + Gauss-Newton
    nls = fit_logistic_nls(t_obs, d_obs)
    p_hat = nls["params"]
    cov = gauss_newton_covariance(nls["jac"], sigma)
    sd_gn = np.sqrt(np.diag(cov))
    G = logistic_sensitivities(t, *p_hat, y0)
    band_gn = Z95 * np.sqrt(np.einsum("ij,jk,ik->i", G, cov, G))
    gn = {"r": {"mean": p_hat[0], "sd": sd_gn[0], "q025": p_hat[0] - Z95 * sd_gn[0], "q975": p_hat[0] + Z95 * sd_gn[0]},
          "K": {"mean": p_hat[1], "sd": sd_gn[1], "q025": p_hat[1] - Z95 * sd_gn[1], "q975": p_hat[1] + Z95 * sd_gn[1]}}
    ex.log(f"least squares: r={p_hat[0]:.5f}+-{sd_gn[0]:.5f}, K={p_hat[1]:.4f}+-{sd_gn[1]:.4f}")

    # ---------------------------------------------------------------- reference posterior (Metropolis)
    n_steps, burn = ex.pick(60000, 4000), ex.pick(10000, 1000)
    with CPUTimer() as tm_mcmc:
        chain, acc = metropolis(t_obs, d_obs, sigma, p_hat, cov, n_steps, np.random.default_rng(ex.seed))
    post = chain[burn:]
    tau = [iact(post[:, 0]), iact(post[:, 1])]
    ess = [len(post) / tau[0], len(post) / tau[1]]
    mc = summarize_samples(post[:, 0], post[:, 1])
    thin = post[:: max(1, len(post) // 2000)]
    Y = logistic_exact(t[None, :], thin[:, :1], thin[:, 1:2], y0)
    mc_lo, mc_hi = np.quantile(Y, 0.025, axis=0), np.quantile(Y, 0.975, axis=0)
    mc_mean = Y.mean(axis=0)
    ex.log(f"Metropolis: acceptance={acc:.3f}, IACT={tau[0]:.1f}/{tau[1]:.1f}, "
           f"r={mc['r']['mean']:.5f}+-{mc['r']['sd']:.5f}, K={mc['K']['mean']:.4f}+-{mc['K']['sd']:.4f}")

    # ---------------------------------------------------------------- ensembles
    n_members = ex.pick(10, 3)
    pinn_kw = dict(adam_iters=ex.pick(2000, 60), lbfgs_iters=ex.pick(500, 10), lr=5e-3, lam_r=1.0,
                   width=20, depth=2, n_col=100, eval_every=50)
    ens = {}
    for key, perturb, base in (("init", False, 2000), ("perturbed", True, 3000)):
        ex.log(f"ensemble '{key}'")
        members = run_ensemble(ex, t_obs, d_obs, sigma, n_members, perturb, base, pinn_kw)
        r = np.array([m["r"] for m in members])
        K = np.array([m["K"] for m in members])
        Ym = np.array([m["y"] for m in members])
        stats = summarize_samples(r, K)
        for name in ("r", "K"):        # Gaussian 95 % interval from the ensemble spread
            s = stats[name]
            s["q025"], s["q975"] = s["mean"] - Z95 * s["sd"], s["mean"] + Z95 * s["sd"]
        ens[key] = {"members": members, "r": r, "K": K, "stats": stats, "y_mean": Ym.mean(axis=0),
                    "y_halfwidth": Z95 * Ym.std(axis=0, ddof=1),
                    "cpu_s": float(sum(m["cpu_s"] for m in members)),
                    "wall_s": float(sum(m["wall_s"] for m in members))}
        save_rows_csv(ex.path(f"ensemble_{key}.csv"), {"r": r, "K": K, "cpu_s": [m["cpu_s"] for m in members]})

    # ---------------------------------------------------------------- tables and results
    methods = [("Metropolis (reference)", mc, tm_mcmc.cpu),
               ("Gauss--Newton (linearized)", gn, nls["cpu_s"]),
               ("ensemble, initialization only", ens["init"]["stats"], ens["init"]["cpu_s"]),
               ("ensemble, perturbed data", ens["perturbed"]["stats"], ens["perturbed"]["cpu_s"])]
    body = []
    for name, st, cpu in methods:
        cells = [name]
        for par, dig in (("r", 4), ("K", 3)):
            s = st[par]
            cells += [f"{s['mean']:.{dig}f}", f"{s['sd']:.{dig}f}", f"$[{s['q025']:.{dig}f},\\,{s['q975']:.{dig}f}]$"]
        cells.append(fmt_time(cpu))
        body.append(cells)
    header = (r"& \multicolumn{3}{c}{$r$ (true value $0.8$)} & \multicolumn{3}{c}{$K$ (true value $10$)} & \\"
              "\n" r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}" "\n"
              r"method & estimate & sd & 95\,\% interval & estimate & sd & 95\,\% interval & CPU (s) \\")
    write_tabular(ex.path("tab_uq.tex"), "lccccccr", header, body,
                  comment="chapters/ch11_inverse_problems/uq_ensembles.py")
    widths = {"metropolis": float(np.mean(mc_hi - mc_lo) / 2), "gauss_newton": float(np.mean(band_gn)),
              "ensemble_init": float(np.mean(ens["init"]["y_halfwidth"])),
              "ensemble_perturbed": float(np.mean(ens["perturbed"]["y_halfwidth"]))}
    inside = {k: float(np.mean((y_true >= lo) & (y_true <= hi))) for k, (lo, hi) in {
        "metropolis": (mc_lo, mc_hi),
        "ensemble_perturbed": (ens["perturbed"]["y_mean"] - ens["perturbed"]["y_halfwidth"],
                               ens["perturbed"]["y_mean"] + ens["perturbed"]["y_halfwidth"]),
        "ensemble_init": (ens["init"]["y_mean"] - ens["init"]["y_halfwidth"],
                          ens["init"]["y_mean"] + ens["init"]["y_halfwidth"])}.items()}
    ex.save_results({
        "problem": {**LOGISTIC, "eta": ETA, "sigma": sigma, "data": "logistic_data(0.05, 0)"},
        "metropolis": {"steps": n_steps, "burn_in": burn, "acceptance": acc, "iact": tau, "ess": ess,
                       "summary": mc, "cpu_s": tm_mcmc.cpu, "wall_s": tm_mcmc.wall},
        "gauss_newton": {"summary": gn, "corr_rK": float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])),
                         "cpu_s": nls["cpu_s"]},
        "ensemble_init": {"summary": ens["init"]["stats"], "cpu_s": ens["init"]["cpu_s"],
                          "members": n_members},
        "ensemble_perturbed": {"summary": ens["perturbed"]["stats"], "cpu_s": ens["perturbed"]["cpu_s"],
                               "members": n_members},
        "sd_ratio_to_metropolis": {k: {p: ens[k]["stats"][p]["sd"] / mc[p]["sd"] for p in ("r", "K")}
                                   for k in ("init", "perturbed")},
        "gn_sd_ratio_to_metropolis": {p: gn[p]["sd"] / mc[p]["sd"] for p in ("r", "K")},
        "mean_band_halfwidth": widths, "fraction_of_exact_curve_inside_band": inside,
        "hyperparameters": {**pinn_kw, "p_init": "r0 ~ U(0.2, 0.6), K0 ~ U(3, 15) per member"}})

    if ex.args.no_figures:
        return
    from pinnode import plotting as P   # imported late: keeps the training footprint small

    # ---------------------------------------------------------------- figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    e = ens["perturbed"]
    lo, hi = e["y_mean"] - e["y_halfwidth"], e["y_mean"] + e["y_halfwidth"]
    ax.fill_between(t, lo, hi, color=P.SERIES[0], alpha=0.18, lw=0, zorder=1)
    ax.plot(t, lo, color=P.SERIES[0], lw=0.7, zorder=2)
    ax.plot(t, hi, color=P.SERIES[0], lw=0.7, zorder=2, label="ensemble, 95%")
    ax.plot(t, mc_lo, color=P.COLORS["ref"], lw=0.8, ls="--", zorder=3)
    ax.plot(t, mc_hi, color=P.COLORS["ref"], lw=0.8, ls="--", zorder=3, label="Metropolis, 95%")
    P.plot_reference(ax, t, y_true, label="exact", lw=0.9, ls=":")
    ax.plot(t_obs, d_obs, ls="none", marker="o", ms=3.8, mfc="white", mec=P.COLORS["ref"], mew=0.8,
            label="data", zorder=4)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("state and 95% bands")
    ax.set_ylim(-0.3, 1.25 * np.max(d_obs))
    ax.legend(loc="upper left", fontsize=6.5)

    ax = axes[0, 1]
    ax.semilogy(t, np.maximum((mc_hi - mc_lo) / 2, 1e-12), color=P.COLORS["ref"], lw=1.1, label="Metropolis")
    ax.semilogy(t, np.maximum(band_gn, 1e-12), color=P.COLORS["muted"], lw=1.1, ls=(0, (1, 1)), label="Gauss-Newton")
    ax.semilogy(t, np.maximum(ens["perturbed"]["y_halfwidth"], 1e-12), color=P.SERIES[0], lw=1.2, ls="--",
                label="ensemble, perturbed data")
    ax.semilogy(t, np.maximum(ens["init"]["y_halfwidth"], 1e-12), color=P.SERIES[1], lw=1.2, ls="-.",
                label="ensemble, init. only")
    late = t >= 0.5
    lows = [np.min(v[late]) for v in ((mc_hi - mc_lo) / 2, band_gn, ens["perturbed"]["y_halfwidth"],
                                      ens["init"]["y_halfwidth"])]
    highs = [np.max(v) for v in ((mc_hi - mc_lo) / 2, band_gn, ens["perturbed"]["y_halfwidth"],
                                 ens["init"]["y_halfwidth"])]
    ax.set_ylim(0.3 * max(min(lows), 1e-12), 3.0 * max(highs))
    ax.set_xlabel("$t$")
    ax.set_ylabel("half-width of 95% band")
    ax.set_title("width of the state bands")
    ax.legend(loc="center right", fontsize=7)

    ax = axes[1, 0]
    sub = post[:: max(1, len(post) // 3000)]
    ax.plot(sub[:, 0], sub[:, 1], ls="none", marker=".", ms=1.5, color=P.COLORS["light"], zorder=1,
            label="Metropolis samples")
    w, V = np.linalg.eigh(cov)
    ang = np.linspace(0, 2 * np.pi, 200)
    rad = np.sqrt(5.991)                      # 95 % quantile of chi^2 with 2 degrees of freedom
    ell = p_hat[:, None] + V @ (rad * np.sqrt(w)[:, None] * np.vstack([np.cos(ang), np.sin(ang)]))
    ax.plot(ell[0], ell[1], color=P.COLORS["ref"], lw=1.0, label="Gauss-Newton 95% ellipse")
    ax.plot(ens["perturbed"]["r"], ens["perturbed"]["K"], ls="none", marker="o", ms=3.8, mfc="white",
            mec=P.SERIES[0], mew=0.9, label="ensemble, perturbed data", zorder=3)
    ax.plot(ens["init"]["r"], ens["init"]["K"], ls="none", marker="s", ms=3.8, mfc="white",
            mec=P.SERIES[1], mew=0.9, label="ensemble, init. only", zorder=4)
    ax.plot([p_true[0]], [p_true[1]], ls="none", marker="+", ms=8, mew=1.3, color=P.COLORS["ref"],
            label="true value", zorder=5)
    ax.set_xlabel("$r$")
    ax.set_ylabel("$K$")
    ax.set_title("parameter uncertainty")
    k_lo, k_hi = np.min(sub[:, 1]), np.max(sub[:, 1])
    ax.set_ylim(k_lo - 0.05 * (k_hi - k_lo), k_hi + 0.95 * (k_hi - k_lo))   # head room for the legend
    ax.legend(loc="upper right", ncol=1, fontsize=6)

    ax = axes[1, 1]
    ax.hist(post[:, 0], bins=60, density=True, color=P.COLORS["light"], histtype="stepfilled",
            label="Metropolis marginal")
    rr = np.linspace(post[:, 0].min(), post[:, 0].max(), 300)
    ax.plot(rr, np.exp(-0.5 * ((rr - p_hat[0]) / sd_gn[0]) ** 2) / (sd_gn[0] * np.sqrt(2 * np.pi)),
            color=P.COLORS["ref"], lw=1.0, ls="--", label="Gauss-Newton")
    ymax = ax.get_ylim()[1]
    for j, (key, lab) in enumerate((("perturbed", "ensemble, perturbed data"), ("init", "ensemble, init. only"))):
        ax.plot(ens[key]["r"], np.full(n_members, (0.06 + 0.07 * j) * ymax), ls="none",
                marker="o" if j == 0 else "s", ms=3.6, mfc="white", mec=P.SERIES[j], mew=0.9, label=lab)
    ax.axvline(p_true[0], color=P.COLORS["ref"], lw=0.7, ls=":")
    ax.set_xlabel("$r$")
    ax.set_ylabel("density")
    ax.set_title("marginal of $r$")
    ax.set_ylim(0.0, 1.7 * ymax)
    ax.legend(loc="upper right", ncol=1, fontsize=6)
    P.label_panels(axes)
    ex.savefig(fig, "uq_ensembles")


if __name__ == "__main__":
    main()
