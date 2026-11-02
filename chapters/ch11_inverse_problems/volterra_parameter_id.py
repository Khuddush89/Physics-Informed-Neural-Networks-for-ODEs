"""Identifying the parameter of a Volterra integral equation from noisy samples of its solution.

Problem
    u(x) = 1 + lam * int_0^x s u(s) ds,  x in [0, 1],  true lam = 2,  exact u(x) = exp(lam x^2 / 2).
Data
    N = 20 equally spaced points x_i = i/20 in (0, 1], additive Gaussian noise with standard
    deviation sigma = eta * max|u(x_i)|, eta in {0, 0.01, 0.05, 0.10}; five realizations per level.
Methods (same initial value lam0 = 0.5 where an initial value is needed)
    * inverse PINN: u_theta = MLP 1-20-20-1 (tanh) without any constraint at x = 0 (the equation
      itself gives u(0) = 1); residual R(x_j) = u_theta(x_j) - 1 - lam sum_k W_jk S_jk u_theta(S_jk)
      with 16-point Gauss-Legendre rules on [0, x_j] at M = 50 uniform points; lam is a plain
      trainable scalar (no sign constraint); Adam (2000 iterations, lr 5e-3 -> 5e-5) + L-BFGS;
    * integral form, data only: lam_hat = sum (d_i - 1) I_i / sum I_i^2 with I_i the cumulative
      trapezoidal approximation of int_0^{x_i} s u(s) ds built from the data (and u(0) = 1);
    * differential form, data only: u' = lam x u, with central differences of the data,
      lam_hat = sum D_i x_i d_i / sum (x_i d_i)^2 (interior points);
    * classical least squares: scipy.optimize.least_squares over lam with the forward model computed
      by the trapezoidal Volterra solver (pinnode.reference.volterra_trapezoid, 4001 nodes).

Book: Chapter 11, Section "An inverse problem for an integral equation" (Figure inv:fig:volterra,
      Table inv:tab:volterra).
Run:  PYTHONPATH=. python chapters/ch11_inverse_problems/volterra_parameter_id.py
      (about 2 min CPU on one core; --quick for a smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnode import Example, MLP, rel_l2, linf, to_tensor, train, uniform, volterra_gauss  # noqa: E402
from pinnode.reference import volterra_trapezoid  # noqa: E402

from _common import (save_rows_csv, save_history_csv, CPUTimer, RESUME_ARG, add_noise, cached, data_rng, fmt_num, fmt_time,  # noqa: E402
                     median, write_tabular)

LAM, N = 2.0, 20
ETAS = [0.0, 0.01, 0.05, 0.10]
LAM0 = 0.5
LAM_R_STRONG = 100.0      # second PINN variant: residual weight 100 (model trusted, Section 11.1)


def exact(x, lam=LAM):
    return np.exp(lam * np.asarray(x) ** 2 / 2.0)


def make_data(eta, k):
    x = np.arange(1, N + 1) / N
    d, sigma = add_noise(exact(x), eta, data_rng(17, eta, k))
    return x, d, sigma


def fit_pinn(x_obs, d_obs, seed, adam_iters, lbfgs_iters, n_col=50, n_quad=16, lam_r=1.0):
    torch.manual_seed(seed)
    model = MLP(1, 1, width=20, depth=2, activation="tanh", domain=(0.0, 1.0))
    lam = torch.nn.Parameter(torch.tensor(LAM0))
    x_col = uniform(n_col, 0.0, 1.0)
    S, W = volterra_gauss(x_col, n_quad, a=0.0)
    x_c, S_t, W_t = to_tensor(x_col), to_tensor(S.reshape(-1, 1)), torch.as_tensor(W)
    x_d, d = to_tensor(x_obs), to_tensor(d_obs)

    def loss_fn():
        uS = model(S_t).reshape(n_col, n_quad)
        integral = (W_t * S_t.reshape(n_col, n_quad) * uS).sum(dim=1, keepdim=True)
        res = model(x_c) - 1.0 - lam * integral
        L_r = (res ** 2).mean()
        L_d = ((model(x_d) - d) ** 2).mean()
        return L_d + lam_r * L_r, {"data": L_d, "residual": L_r}

    def eval_fn():
        return {"lam": float(lam.detach()), "err_lam": abs(float(lam.detach()) / LAM - 1.0)}

    with CPUTimer() as tm:
        hist = train(list(model.parameters()) + [lam], loss_fn, adam_iters=adam_iters, lr=5e-3,
                     lbfgs_iters=lbfgs_iters, eval_fn=eval_fn, eval_every=10, verbose=False)
    x_grid = np.linspace(0.0, 1.0, 1001)
    with torch.no_grad():
        u_grid = model(to_tensor(x_grid)).numpy().ravel()
    return {"lam": float(lam.detach()), "x_grid": x_grid, "u_grid": u_grid, "hist": hist,
            "cpu_s": tm.cpu, "wall_s": hist.train_time}


def integral_estimator(x_obs, d_obs):
    """Integral form u - 1 = lam int_0^x s u ds with trapezoidal integrals of the data."""
    with CPUTimer() as tm:
        x = np.r_[0.0, x_obs]
        g = x * np.r_[1.0, d_obs]                         # s u(s) with u(0) = 1 known
        I = np.r_[0.0, np.cumsum(0.5 * np.diff(x) * (g[1:] + g[:-1]))][1:]
        lam = float(np.sum((d_obs - 1.0) * I) / np.sum(I ** 2))
    return lam, tm.cpu


def differential_estimator(x_obs, d_obs):
    """Differential form u' = lam x u with central differences of the data (interior points)."""
    with CPUTimer() as tm:
        h = x_obs[1] - x_obs[0]
        D = (d_obs[2:] - d_obs[:-2]) / (2 * h)
        xd = x_obs[1:-1] * d_obs[1:-1]
        lam = float(np.sum(D * xd) / np.sum(xd ** 2))
    return lam, tm.cpu


def nls_estimator(x_obs, d_obs, n_nodes=4001):
    def forward(lam):
        xs, us = volterra_trapezoid(lambda x, s: s * np.ones_like(np.asarray(s, float)),
                                    lambda x: np.ones_like(np.asarray(x, float)), lam, 0.0, 1.0, n=n_nodes)
        return np.interp(x_obs, xs, us)

    with CPUTimer() as tm:
        res = least_squares(lambda q: forward(q[0]) - d_obs, [LAM0], method="trf", diff_step=1e-6,
                            xtol=1e-12, ftol=1e-12, gtol=1e-12)
    return float(res.x[0]), tm.cpu, int(res.nfev)


def main():
    ex = Example("ch11_inverse_problems", "volterra_parameter_id",
                 description="Volterra equation u = 1 + lam int_0^x s u ds: identify lam from noisy data",
                 default_seed=1234, extra_args=[RESUME_ARG])
    n_real = ex.pick(5, 2)
    adam, lbfgs = ex.pick(2000, 60), ex.pick(500, 10)
    n_nodes = ex.pick(4001, 401)
    rows, keep = [], {}
    for eta in ETAS:
        for k in range(n_real):
            x_obs, d_obs, sigma = make_data(eta, k)
            pinn = cached(ex, f"pinn_eta{eta}_k{k}", lambda: fit_pinn(x_obs, d_obs, ex.seed + k, adam, lbfgs),
                          seed=ex.seed + k, adam=adam, lbfgs=lbfgs)
            pinn100 = cached(ex, f"pinn_lr100_eta{eta}_k{k}",
                             lambda: fit_pinn(x_obs, d_obs, ex.seed + k, adam, lbfgs, lam_r=LAM_R_STRONG),
                             seed=ex.seed + k, adam=adam, lbfgs=lbfgs, lam_r=LAM_R_STRONG)
            lam_int, cpu_int = integral_estimator(x_obs, d_obs)
            lam_dif, cpu_dif = differential_estimator(x_obs, d_obs)
            lam_nls, cpu_nls, nfev = nls_estimator(x_obs, d_obs, n_nodes)
            for name, val, cpu in (("PINN", pinn["lam"], pinn["cpu_s"]),
                                   ("PINN100", pinn100["lam"], pinn100["cpu_s"]), ("integral", lam_int, cpu_int),
                                   ("differential", lam_dif, cpu_dif), ("NLS", lam_nls, cpu_nls)):
                rows.append({"eta": eta, "realization": k, "method": name, "lam": val,
                             "err": abs(val / LAM - 1.0), "cpu_s": cpu})
            ex.log(f"eta={eta:.2f} k={k}: PINN {pinn['lam']:.5f} (lam_r=100: {pinn100['lam']:.5f}) | integral {lam_int:.5f} | "
                   f"differential {lam_dif:.5f} | least squares {lam_nls:.5f} ({pinn['cpu_s']:.1f}s CPU)")
            if k == 0:
                keep[eta] = {"x": x_obs, "d": d_obs, "pinn": pinn, "pinn100": pinn100}
                save_history_csv(ex.path("history_" + f"pinn_eta{int(round(100 * eta)):02d}" + ".csv"), pinn["hist"])
    save_rows_csv(ex.path("runs.csv"), rows)
    methods = [("PINN", r"PINN, $\lambda_r=1$"), ("PINN100", r"PINN, $\lambda_r=100$"),
               ("integral", "integral form, trapezoid on data"),
               ("differential", "differential form, finite differences"),
               ("NLS", "least squares, Volterra solver")]
    summary = {}
    body = []
    for key, label in methods:
        cells = [label]
        for eta in ETAS:
            errs = [r["err"] for r in rows if r["method"] == key and r["eta"] == eta]
            lams = [r["lam"] for r in rows if r["method"] == key and r["eta"] == eta]
            summary[f"{key}_eta{eta}"] = {"mean_err": float(np.mean(errs)), "max_err": float(np.max(errs)),
                                          "lam_mean": float(np.mean(lams)),
                                          "lam_sd": float(np.std(lams, ddof=1)) if len(lams) > 1 else 0.0}
            cells.append(fmt_num(float(np.mean(errs))))
        cpu = median(r["cpu_s"] for r in rows if r["method"] == key)
        summary[f"{key}_median_cpu_s"] = cpu
        cells.append(fmt_time(cpu) if cpu >= 1e-3 else r"$<10^{-3}$")
        body.append(cells)
    write_tabular(ex.path("tab_volterra.tex"), "lccccr",
                  ["method"] + [rf"$\eta = {int(round(100 * e))}\,\%$" for e in ETAS] + ["CPU (s)"], body,
                  comment="chapters/ch11_inverse_problems/volterra_parameter_id.py")
    # accuracy of the PINN state for noise-free data
    xg = np.linspace(0.0, 1.0, 1001)
    u0 = keep[0.0]["pinn"]["u_grid"]
    ex.save_results({
        "problem": {"lam": LAM, "N": N, "etas": ETAS, "realizations": n_real, "lam0": LAM0,
                    "noise": "additive Gaussian, sd = eta*max|u(x_i)|"},
        "hyperparameters": {"width": 20, "depth": 2, "collocation": 50, "quadrature_nodes": 16,
                            "adam_iters": adam, "lr": "5e-3 -> 5e-5 (cosine)", "lbfgs_iters": lbfgs,
                            "lam_r": [1.0, LAM_R_STRONG], "volterra_trapezoid_nodes": n_nodes},
        "summary": summary,
        "pinn_state_eta0": {"rel_l2": rel_l2(u0, exact(xg)), "linf": linf(u0, exact(xg))},
        "pinn_state_eta5": {"lam_r_1": {"rel_l2": rel_l2(keep[0.05]["pinn"]["u_grid"], exact(xg)),
                                        "linf": linf(keep[0.05]["pinn"]["u_grid"], exact(xg)),
                                        "u0": float(keep[0.05]["pinn"]["u_grid"][0])},
                            "lam_r_100": {"rel_l2": rel_l2(keep[0.05]["pinn100"]["u_grid"], exact(xg)),
                                          "linf": linf(keep[0.05]["pinn100"]["u_grid"], exact(xg)),
                                          "u0": float(keep[0.05]["pinn100"]["u_grid"][0])}}})

    if ex.args.no_figures:
        return
    from pinnode import plotting as P   # imported late: keeps the training footprint small

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    kp = keep[0.05]
    P.plot_reference(ax, xg, exact(xg), label="exact")
    P.plot_prediction(ax, xg, kp["pinn"]["u_grid"], label=r"PINN, $\lambda_r=1$", k=0)
    P.plot_prediction(ax, xg, kp["pinn100"]["u_grid"], label=r"PINN, $\lambda_r=100$", k=1, lw=1.3)
    ax.plot(kp["x"], kp["d"], ls="none", marker="o", ms=3.6, mfc="white", mec=P.COLORS["ref"], mew=0.8,
            label="data", zorder=4)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")
    ax.set_title(r"data with $\eta = 5\,\%$")
    ax.legend(loc="upper left")
    ax = axes[0, 1]
    for j, eta in enumerate(ETAS):
        h = keep[eta]["pinn"]["hist"]
        ax.semilogy(h.eval_iters, np.maximum(h.metrics["err_lam"], 1e-12), color=P.SERIES[j],
                    ls="-" if j == 0 else P.LINESTYLES[j - 1], lw=1.1, label=rf"$\eta = {int(round(100 * eta))}\,\%$")
    sw = keep[0.0]["pinn"]["hist"].switch_iter
    if sw:
        ax.axvline(sw, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$|\lambda_k - \lambda|/\lambda$")
    ax.set_title(r"convergence of $\lambda$ (PINN)")
    ax.legend(loc="lower left", ncol=2)
    ax = axes[1, 0]
    for j, (key, label) in enumerate(methods):
        xs, ys = [], []
        for i, eta in enumerate(ETAS):
            sel = [r for r in rows if r["method"] == key and r["eta"] == eta]
            xs += [i + (j - 2.0) * 0.13] * len(sel)
            ys += [max(s["err"], 1e-12) for s in sel]
        ax.semilogy(xs, ys, ls="none", marker=P.MARKERS[j], ms=3.4, mfc="white", mec=P.SERIES[j], mew=0.9,
                    label={"PINN": r"PINN, $\lambda_r=1$", "PINN100": r"PINN, $\lambda_r=100$",
                           "integral": "integral form", "differential": "differential form",
                           "NLS": "least squares"}[key])
    ax.set_xticks(range(len(ETAS)))
    ax.set_xticklabels([f"{int(round(100 * e))}%" for e in ETAS])
    ax.set_xlabel(r"noise level $\eta$")
    ax.set_ylabel(r"$|\hat\lambda - \lambda|/\lambda$")
    ax.set_title(f"all estimates ({n_real} data sets per level)")
    ax.legend(loc="lower right", fontsize=7)
    ax = axes[1, 1]
    for j, (eta, key, lab) in enumerate(((0.0, "pinn", r"$\eta=0$, $\lambda_r=1$"),
                                          (0.05, "pinn", r"$\eta=5\,\%$, $\lambda_r=1$"),
                                          (0.05, "pinn100", r"$\eta=5\,\%$, $\lambda_r=100$"))):
        P.plot_error(ax, xg, keep[eta][key]["u_grid"] - exact(xg), label=lab, k=j,
                     xlabel="$x$", ylabel=r"$|u_{\theta}(x) - u(x)|$")
    ax.set_title("pointwise error of the PINN state")
    P.label_panels(axes)
    ex.savefig(fig, "volterra_parameter_id")


if __name__ == "__main__":
    main()
