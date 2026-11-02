"""Partially observed Lotka-Volterra system: recover all parameters and the unobserved predator.

Problem
    x' = a x - b x y,  y' = -c y + d x y  on [0, 12],  (a, b, c, d) = (1, 0.1, 1.5, 0.075),
    x(0) = y(0) = 5 (reference solution: solve_ivp, DOP853, rtol = atol = 1e-12).
Data
    Only the prey is observed: N = 30 equally spaced times in [0, 12] with multiplicative
    (log-normal) noise, d_i = x(t_i) exp(eta xi_i), eta = 0.02, xi_i ~ N(0, 1); hence the data
    misfit is a least-squares misfit of log x.
Cases
    A  the initial predator density y(0) = 5 is known (all parameters structurally identifiable);
    B  y(0) is unknown: only the product b y(t) is identifiable from prey data, because
       (x, y, b) -> (x, k y, b/k) maps solutions to solutions.
Methods
    * inverse PINN with log-states u = log(x/20), v = log(y/10) (an MLP 1-32-32-32-2, tanh),
      residuals u' - (a - b y) and v' - (-c + d x) at M = 300 points, log-parametrized
      (a, b, c, d), lam_r = 1, Adam (5000 iterations, lr 5e-3 -> 5e-5) + L-BFGS (<= 1000);
      y(0) = 5 imposed exactly in case A (v = log(0.5) + (t/T) N_2(t));
    * classical output least squares on log x with solve_ivp (DOP853, rtol = atol = 1e-10),
      unknowns (log a, ..., log d, log x0), started from the same initial guesses and from
      20 random starts (log-uniform in [p/3, 3p]); the fit started at the true values gives the
      maximum-likelihood reference and its Gauss-Newton standard deviations.
    Initial guesses p0 = p* (0.5, 2, 0.5, 2) and p0' = p* (2, 0.5, 2, 0.5).

Book: Chapter 11, Section "A partially observed system" (Figure inv:fig:lv, Table inv:tab:lv).
Run:  PYTHONPATH=. python chapters/ch11_inverse_problems/lotka_volterra_partial.py
      (about 4 min CPU on one core; --quick for a smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pinnode import Example, MLP, derivative, rel_l2, to_tensor, train, uniform  # noqa: E402

from _common import save_rows_csv, save_history_csv, CPUTimer, RESUME_ARG, cached, data_rng, fmt_time, median, write_tabular  # noqa: E402

P_TRUE = np.array([1.0, 0.1, 1.5, 0.075])
X0, Y0, T, N, ETA = 5.0, 5.0, 12.0, 30, 0.02
SX, SY = 20.0, 10.0
P0 = P_TRUE * np.array([0.5, 2.0, 0.5, 2.0])
P0B = P_TRUE * np.array([2.0, 0.5, 2.0, 0.5])
NAMES = ("a", "b", "c", "d")


def rhs(t, z, p):
    a, b, c, d = p
    x, y = z
    return [a * x - b * x * y, -c * y + d * x * y]


def _blowup(t, z, p):
    """Terminal event: stop the integration when a population exceeds 1e6 (trial parameters far
    from the truth can make the solution explode, and the solver then takes very many steps)."""
    return 1e6 - max(z[0], z[1])


_blowup.terminal = True


class TooExpensive(RuntimeError):
    """Raised when a forward solve needs more right-hand-side evaluations than allowed."""


def lv_solve(p, x0, y0, t_eval, rtol=1e-12, atol=1e-12, guard=False, max_rhs=None):
    count = [0]

    def f(t, z, p):
        count[0] += 1
        if max_rhs is not None and count[0] > max_rhs:
            raise TooExpensive
        return rhs(t, z, p)

    sol = solve_ivp(f, (0.0, float(np.max(t_eval))), [x0, y0], args=(p,), t_eval=t_eval,
                    method="DOP853", rtol=rtol, atol=atol, events=_blowup if guard else None)
    return sol


def make_data():
    t_obs = np.linspace(0.0, T, N)
    x = lv_solve(P_TRUE, X0, Y0, t_obs).y[0]
    rng = data_rng(13, ETA, 0)
    return t_obs, x * np.exp(ETA * rng.standard_normal(N))


# ----------------------------------------------------------------------------- PINN
def fit_lv_pinn(t_obs, d_obs, *, known_y0: bool, p_init, seed: int, adam_iters: int, lbfgs_iters: int,
                lam_r: float = 1.0, width: int = 32, depth: int = 3, n_col: int = 300, verbose=False):
    torch.manual_seed(seed)
    net = MLP(1, 2, width=width, depth=depth, activation="tanh", domain=(0.0, T))
    log_p = torch.nn.Parameter(torch.log(torch.as_tensor(np.asarray(p_init, float))))
    v0 = float(np.log(Y0 / SY))

    def states(t):
        n = net(t)
        u = n[:, :1]
        v = v0 + (t / T) * n[:, 1:2] if known_y0 else n[:, 1:2]
        return u, v

    t_col = to_tensor(uniform(n_col, 0.0, T), requires_grad=True)
    t_d = to_tensor(t_obs)
    ld = to_tensor(np.log(d_obs) - np.log(SX))

    def loss_fn():
        a, b, c, d = torch.exp(log_p)
        u, v = states(t_col)
        x, y = SX * torch.exp(u), SY * torch.exp(v)
        r_u = derivative(u, t_col) - (a - b * y)
        r_v = derivative(v, t_col) - (-c + d * x)
        L_r = (r_u ** 2).mean() + (r_v ** 2).mean()
        L_d = ((states(t_d)[0] - ld) ** 2).mean()
        return L_d + lam_r * L_r, {"data": L_d, "residual": L_r}

    def eval_fn():
        p = torch.exp(log_p).detach().numpy()
        return {f"err_{k}": abs(p[i] / P_TRUE[i] - 1.0) for i, k in enumerate(NAMES)}

    with CPUTimer() as tm:
        hist = train(list(net.parameters()) + [log_p], loss_fn, adam_iters=adam_iters, lr=5e-3,
                     lbfgs_iters=lbfgs_iters, eval_fn=eval_fn, eval_every=50, verbose=verbose,
                     print_every=1000 if verbose else 0)

    t_grid = np.linspace(0.0, T, 1001)
    with torch.no_grad():
        u, v = states(to_tensor(t_grid))
    return {"params": np.exp(log_p.detach().numpy()), "hist": hist, "t_grid": t_grid,
            "x": SX * np.exp(u.numpy().ravel()), "y": SY * np.exp(v.numpy().ravel()),
            "cpu_s": tm.cpu, "wall_s": hist.train_time,
            "rms_log_misfit": float(np.sqrt(hist.parts["data"][-1]))}


# ----------------------------------------------------------------------------- least squares
def fit_lv_nls(t_obs, d_obs, p_init, x0_init, max_nfev=200):
    """Output least squares on log x; unknowns (log a, log b, log c, log d, log x0); y(0) = 5."""
    ld = np.log(d_obs)

    def resid(q):
        try:
            sol = lv_solve(np.exp(q[:4]), np.exp(q[4]), Y0, t_obs, rtol=1e-10, atol=1e-10, guard=True,
                           max_rhs=100_000)
        except TooExpensive:                            # extremely fast dynamics for trial parameters
            return np.full(len(t_obs), 10.0)
        if not sol.success or sol.y.shape[1] != len(t_obs) or np.any(sol.y[0] <= 0):
            return np.full(len(t_obs), 10.0)            # penalty for exploding or failed solves
        return np.log(sol.y[0]) - ld

    q0 = np.r_[np.log(p_init), np.log(x0_init)]
    with CPUTimer() as tm:
        res = least_squares(resid, q0, method="trf", diff_step=1e-6, max_nfev=max_nfev)
    p = np.exp(res.x[:4])
    J = res.jac.copy()
    J[:, :4] /= p[None, :]
    J[:, 4] /= np.exp(res.x[4])
    return {"params": p, "x0": float(np.exp(res.x[4])), "rms_log_misfit": float(np.sqrt(2 * res.cost / len(t_obs))),
            "jac": J, "cpu_s": tm.cpu, "wall_s": tm.wall, "nfev": int(res.nfev)}


def sensitivity_svd_case_b(t_obs):
    """Singular values of d log x(t_i) / d(log a, log b, log c, log d, log x0, log y0) at p*."""
    q = np.log(np.r_[P_TRUE, X0, Y0])

    def logx(q):
        return np.log(lv_solve(np.exp(q[:4]), np.exp(q[4]), np.exp(q[5]), t_obs).y[0])

    J = np.zeros((len(t_obs), 6))
    for k in range(6):
        h = 1e-5
        qp, qm = q.copy(), q.copy()
        qp[k] += h
        qm[k] -= h
        J[:, k] = (logx(qp) - logx(qm)) / (2 * h)
    return np.linalg.svd(J, compute_uv=False)


def gn_rel_sd(x0: float, y0: float, t_obs) -> np.ndarray:
    """Linearized relative standard deviations of (a, b, c, d) for log-normal prey data with
    noise level ETA on the orbit through (x0, y0); unknowns (a, b, c, d, x0), y(0) known."""
    q = np.log(np.r_[P_TRUE, x0])

    def logx(q):
        return np.log(lv_solve(np.exp(q[:4]), np.exp(q[4]), y0, t_obs).y[0])

    J = np.zeros((len(t_obs), 5))
    for k in range(5):
        qp, qm = q.copy(), q.copy()
        qp[k] += 1e-5
        qm[k] -= 1e-5
        J[:, k] = (logx(qp) - logx(qm)) / 2e-5          # derivative w.r.t. log p = relative sensitivity
    return ETA * np.sqrt(np.diag(np.linalg.inv(J.T @ J)))[:4]


def main():
    ex = Example("ch11_inverse_problems", "lotka_volterra_partial",
                 description="Lotka-Volterra observed through the prey only: parameters and predator",
                 default_seed=1234, extra_args=[RESUME_ARG])
    t_obs, d_obs = make_data()
    t = np.linspace(0.0, T, 1001)
    ref = lv_solve(P_TRUE, X0, Y0, t).y
    adam, lbfgs = ex.pick(5000, 60), ex.pick(1000, 10)

    # ---------------------------------------------------------------- PINN runs
    runs = {}
    for key, known, p_init in (("A_p0", True, P0), ("A_p0b", True, P0B), ("B_p0", False, P0)):
        ex.log(f"PINN case {key}")
        kw = dict(known_y0=known, p_init=p_init, seed=ex.seed, adam_iters=adam, lbfgs_iters=lbfgs)
        fit = cached(ex, f"pinn_{key}", lambda: fit_lv_pinn(t_obs, d_obs, **kw), **kw)
        xp, yp = fit["x"], fit["y"]
        fit.update({"x": xp, "y": yp, "rel_l2_x": rel_l2(xp, ref[0]), "rel_l2_y": rel_l2(yp, ref[1]),
                    "b_y0": float(fit["params"][1] * yp[0])})
        runs[key] = fit
        save_history_csv(ex.path("history_" + f"pinn_{key}" + ".csv"), fit["hist"])
        ex.log(f"  p={np.round(fit['params'], 5)}, relL2 x={fit['rel_l2_x']:.2e}, y={fit['rel_l2_y']:.2e}, "
               f"b*y(0)={fit['b_y0']:.4f} ({fit['cpu_s']:.1f}s CPU)")
    # case B: rescaling test  (b_hat/b) * y_hat  should reproduce y
    kB = runs["B_p0"]["params"][1] / P_TRUE[1]
    rescaled_err = rel_l2(kB * runs["B_p0"]["y"], ref[1])

    # ---------------------------------------------------------------- least squares
    nls = {}
    for key, p_init in (("p0", P0), ("p0b", P0B), ("truth", P_TRUE)):
        fit = cached(ex, f"nls_{key}", lambda: fit_lv_nls(t_obs, d_obs, p_init, d_obs[0], max_nfev=ex.pick(200, 5)),
                     p_init=p_init, max_nfev=ex.pick(200, 5), guard="blowup1e6-maxrhs1e5")
        sol = lv_solve(fit["params"], fit["x0"], Y0, t)
        ok = sol.success and sol.y.shape[1] == len(t)
        fit["rel_l2_x"] = rel_l2(sol.y[0], ref[0]) if ok else np.nan
        fit["rel_l2_y"] = rel_l2(sol.y[1], ref[1]) if ok else np.nan
        fit["x"] = sol.y[0] if ok else np.full_like(t, np.nan)
        nls[key] = fit
        ex.log(f"least squares from {key}: p={np.round(fit['params'], 5)} rms={fit['rms_log_misfit']:.4f} "
               f"({fit['cpu_s']:.1f}s CPU, nfev={fit['nfev']})")
    mle = nls["truth"]
    # hybrid strategy: least squares started from the PINN estimate (case A, start p0)
    pin = runs["A_p0"]
    fit = cached(ex, "nls_from_pinn", lambda: fit_lv_nls(t_obs, d_obs, pin["params"], pin["x"][0],
                                                          max_nfev=ex.pick(200, 5)),
                 p_init=pin["params"], x0=pin["x"][0], max_nfev=ex.pick(200, 5), guard="blowup1e6-maxrhs1e5")
    sol = lv_solve(fit["params"], fit["x0"], Y0, t)
    fit["rel_l2_x"], fit["rel_l2_y"] = rel_l2(sol.y[0], ref[0]), rel_l2(sol.y[1], ref[1])
    fit["reached_mle"] = bool(np.max(np.abs(fit["params"] / mle["params"] - 1.0)) < 1e-3)
    nls["from_pinn"] = fit
    ex.log(f"least squares from the PINN estimate: p={np.round(fit['params'], 5)} nfev={fit['nfev']} "
           f"reached MLE: {fit['reached_mle']}")
    cov = ETA ** 2 * np.linalg.inv(mle["jac"].T @ mle["jac"])
    sd_gn = np.sqrt(np.diag(cov))[:4]
    # multi-start statistics for least squares
    rng = np.random.default_rng(ex.seed)
    n_starts = ex.pick(20, 2)
    ms = []
    starts = [P_TRUE * np.exp(rng.uniform(-np.log(3.0), np.log(3.0), 4)) for _ in range(n_starts)]
    for j, p_init in enumerate(starts):
        fit = cached(ex, f"nls_multistart_{j}",
                     lambda: fit_lv_nls(t_obs, d_obs, p_init, d_obs[0], max_nfev=ex.pick(200, 5)),
                     p_init=p_init, max_nfev=ex.pick(200, 5), guard="blowup1e6-maxrhs1e5")
        hit = bool(np.max(np.abs(fit["params"] / mle["params"] - 1.0)) < 1e-3)
        ms.append({"start": j, **{f"p0_{k}": p_init[i] for i, k in enumerate(NAMES)},
                   **{f"p_{k}": fit["params"][i] for i, k in enumerate(NAMES)},
                   "rms_log_misfit": fit["rms_log_misfit"], "reached_mle": hit, "cpu_s": fit["cpu_s"],
                   "nfev": fit["nfev"]})
    save_rows_csv(ex.path("least_squares_multistart.csv"), ms)
    n_hit = int(sum(m["reached_mle"] for m in ms))
    ex.log(f"least squares multistart: {n_hit}/{n_starts} starts reached the maximum-likelihood estimate")
    svals = sensitivity_svd_case_b(t_obs)
    design = {"orbit_5_5": gn_rel_sd(5.0, 5.0, t_obs), "orbit_10_5": gn_rel_sd(10.0, 5.0, t_obs)}
    ex.log(f"linearized relative sd, orbit (5,5): {np.round(design['orbit_5_5'], 4)}; "
           f"orbit (10,5): {np.round(design['orbit_10_5'], 4)}")

    # ---------------------------------------------------------------- table
    def row(label, p, by0, ry, cpu):
        return [label] + [f"{v:.4f}" for v in p] + [f"{by0:.4f}" if by0 is not None else "--",
                                                    ((f"{100 * ry:.1f}" if ry <= 1.0 else "$>100$")
                                                     if ry is not None and np.isfinite(ry) else "--"),
                                                    fmt_time(cpu) if cpu is not None else "--"]
    body = [row("true values", P_TRUE, P_TRUE[1] * Y0, None, None), "midrule",
            row(r"PINN, start $p_0$", runs["A_p0"]["params"], runs["A_p0"]["b_y0"], runs["A_p0"]["rel_l2_y"], runs["A_p0"]["cpu_s"]),
            row(r"PINN, start $p_0'$", runs["A_p0b"]["params"], runs["A_p0b"]["b_y0"], runs["A_p0b"]["rel_l2_y"], runs["A_p0b"]["cpu_s"]),
            row(r"least squares, start $p_0$", nls["p0"]["params"], nls["p0"]["params"][1] * Y0, nls["p0"]["rel_l2_y"], nls["p0"]["cpu_s"]),
            row(r"least squares, start $p_0'$", nls["p0b"]["params"], nls["p0b"]["params"][1] * Y0, nls["p0b"]["rel_l2_y"], nls["p0b"]["cpu_s"]),
            row(r"least squares, start $p^\ast$ (MLE)", mle["params"], mle["params"][1] * Y0, mle["rel_l2_y"], mle["cpu_s"]),
            row(r"least squares, start = PINN estimate", nls["from_pinn"]["params"], nls["from_pinn"]["params"][1] * Y0,
                nls["from_pinn"]["rel_l2_y"], nls["from_pinn"]["cpu_s"]),
            [r"Gauss--Newton sd of the MLE"] + [f"{v:.4f}" for v in sd_gn] + ["--", "--", "--"],
            "midrule",
            row(r"PINN, $y(0)$ unknown, start $p_0$", runs["B_p0"]["params"], runs["B_p0"]["b_y0"], runs["B_p0"]["rel_l2_y"], runs["B_p0"]["cpu_s"])]
    write_tabular(ex.path("tab_lv.tex"), "lccccccr",
                  ["method", "$a$", "$b$", "$c$", "$d$", r"$b\,y(0)$", r"$y$: rel.\ $L^2$ (\%)", "CPU (s)"],
                  body, comment="chapters/ch11_inverse_problems/lotka_volterra_partial.py")

    ex.save_results({
        "problem": {"p_true": P_TRUE, "x0": X0, "y0": Y0, "T": T, "N": N, "eta": ETA,
                    "noise": "multiplicative log-normal: d = x exp(eta xi)", "p0": P0, "p0b": P0B},
        "hyperparameters": {"width": 32, "depth": 3, "collocation": 300, "adam_iters": adam,
                            "lr": "5e-3 -> 5e-5 (cosine)", "lbfgs_iters": lbfgs, "lam_r": 1.0,
                            "states": "u = log(x/20), v = log(y/10)", "nls": "least_squares(trf) on log x, solve_ivp DOP853 1e-10, max_nfev=200, solves stopped (penalty residual) when a population exceeds 1e6 or after 1e5 right-hand-side evaluations"},
        "pinn": {k: {"params": v["params"], "rel_err": np.abs(v["params"] / P_TRUE - 1), "rel_l2_x": v["rel_l2_x"],
                     "rel_l2_y": v["rel_l2_y"], "b_y0": v["b_y0"], "rms_log_misfit": v["rms_log_misfit"],
                     "cpu_s": v["cpu_s"], "wall_s": v["wall_s"]} for k, v in runs.items()},
        "case_B_rescaling": {"k = b_hat/b": kB, "rel_l2_of_rescaled_predator": rescaled_err,
                             "y0_hat": float(runs["B_p0"]["y"][0])},
        "least_squares": {k: {"params": v["params"], "x0": v["x0"], "rel_err": np.abs(v["params"] / P_TRUE - 1),
                              "rms_log_misfit": v["rms_log_misfit"], "rel_l2_x": v["rel_l2_x"],
                              "rel_l2_y": v["rel_l2_y"], "cpu_s": v["cpu_s"], "nfev": v["nfev"],
                              "reached_mle": v.get("reached_mle")} for k, v in nls.items()},
        "gauss_newton_sd": dict(zip(NAMES, sd_gn)), "gauss_newton_rel_sd": dict(zip(NAMES, sd_gn / P_TRUE)),
        "multistart": {"starts": n_starts, "reached_mle": n_hit,
                       "median_cpu_s": median(m["cpu_s"] for m in ms)},
        "design_rel_sd_at_truth": {k: dict(zip(NAMES, v)) for k, v in design.items()},
        "case_B_sensitivity_singular_values": svals, "case_B_condition_number": float(svals[0] / svals[-1]),
        "pinn_vs_mle_max_rel_gap": float(np.max(np.abs(runs["A_p0"]["params"] / mle["params"] - 1)))})

    if ex.args.no_figures:
        return
    from pinnode import plotting as P   # imported late: keeps the training footprint small

    # ---------------------------------------------------------------- figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, t, ref[0], label="exact")
    P.plot_prediction(ax, t, runs["A_p0"]["x"], label="PINN", k=0)
    P.plot_prediction(ax, t, nls["p0"]["x"], label=r"least squares from $p_0$", k=1, lw=1.2)
    ax.plot(t_obs, d_obs, ls="none", marker="o", ms=3.6, mfc="white", mec=P.COLORS["ref"], mew=0.8,
            label="prey data", zorder=4)
    ax.set_xlabel("$t$")
    ax.set_ylabel("prey $x(t)$")
    ax.set_title("observed prey")
    ax.set_ylim(0.0, 1.45 * np.max(ref[0]))
    ax.legend(loc="upper center", ncol=2, fontsize=7)
    ax = axes[0, 1]
    P.plot_reference(ax, t, ref[1], label="exact")
    P.plot_prediction(ax, t, runs["A_p0"]["y"], label=r"PINN, $y(0)$ known", k=0)
    ax.set_xlabel("$t$")
    ax.set_ylabel("predator $y(t)$")
    ax.set_title("unobserved predator")
    ax.set_ylim(0.0, 1.35 * np.max(ref[1]))
    ax.legend(loc="upper center", ncol=2, fontsize=7)
    ax = axes[1, 0]
    h = runs["A_p0"]["hist"]
    for i, k in enumerate(NAMES):
        ax.semilogy(h.eval_iters, np.maximum(h.metrics[f"err_{k}"], 1e-8), color=P.SERIES[i],
                    ls="-" if i == 0 else P.LINESTYLES[i - 1], lw=1.1, label=f"${k}$")
    if h.switch_iter:
        ax.axvline(h.switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative parameter error")
    ax.set_title(r"convergence of $(a,b,c,d)$")
    lo = min(min(h.metrics[f"err_{k}"]) for k in NAMES)
    ax.set_ylim(max(0.5 * lo, 1e-8), 20.0)
    ax.legend(loc="upper right", ncol=4, fontsize=7)
    ax = axes[1, 1]
    P.plot_reference(ax, t, ref[1], label="exact $y$")
    P.plot_prediction(ax, t, runs["B_p0"]["y"], label=r"PINN $\hat y$, $y(0)$ unknown", k=1)
    P.plot_prediction(ax, t, kB * runs["B_p0"]["y"], label=r"rescaled $(\hat b/b)\,\hat y$", k=2, lw=1.3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("predator")
    ax.set_title("predator known only up to scale")
    ax.set_ylim(0.0, 1.62 * np.max(ref[1]))
    ax.legend(loc="upper left", ncol=1, fontsize=7)
    P.label_panels(axes)
    ex.savefig(fig, "lotka_volterra_partial")


if __name__ == "__main__":
    main()
