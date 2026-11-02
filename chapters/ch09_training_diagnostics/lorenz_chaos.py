"""Chaos: the Lorenz system on short windows, Lyapunov exponent and predictability.

Book: Chapter 9, Section "Chaotic dynamics".

    x' = sigma (y - x),  y' = x (rho - z) - y,  z' = x y - beta z,
    sigma = 10, rho = 28, beta = 8/3,  (x, y, z)(0) = (-8, 7, 27).

1. Largest Lyapunov exponent by Benettin's method: the variational equation is integrated
   along a trajectory on the attractor (transient 50 time units), the tangent vector is
   renormalized every time unit, 1000 renormalizations (DOP853, rtol 1e-10, atol 1e-12).
2. Twin experiment: two reference trajectories whose initial values differ by 1e-8 in x.
3. Logarithmic norm mu_2(J) = lambda_max((J + J^T)/2) along the reference on [0, 5]
   (the rate in the one-sided Gronwall bound).
4. PINNs (hard initial condition u = u0 + 20 t N(t), residual divided by 20, tanh MLP 64 x 3,
   M = 100 T + 100 uniform points, Adam 4000 + L-BFGS 500) on [0, T] for T = 0.5, 1, 3, and a
   time-marching PINN on [0, 5] with twenty windows of length 0.25 (tanh MLP 48 x 3 per window,
   100 points, Adam 2000 + L-BFGS 500 per window, state carried between windows).

Reference: DOP853 with rtol = atol = 1e-12. Errors: Euclidean norm of the state error on the
uniform 1001-point grid; the divergence time is the first t with |u_theta - u| > 1.

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/lorenz_chaos.py          (~6 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/lorenz_chaos.py --quick  (smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CPUTimer, RunCache, latex_table, num, replot_arg  # noqa: E402
from pinnode import ConstrainedModel, Example, MLP, derivative, rel_l2, to_tensor, train  # noqa: E402
from pinnode import plotting as P  # noqa: E402

SIG, RHO, BETA = 10.0, 28.0, 8.0 / 3.0
U0 = np.array([-8.0, 7.0, 27.0])
SCALE = 20.0
T_LIST = [0.5, 1.0, 3.0]
DIV_TOL = 1.0


def f(t, u):
    x, y, z = u
    return np.array([SIG * (y - x), x * (RHO - z) - y, x * y - BETA * z])


def jac(u):
    x, y, z = u
    return np.array([[-SIG, SIG, 0.0], [RHO - z, -1.0, -x], [y, x, -BETA]])


def reference(t_eval, u0=U0):
    sol = solve_ivp(f, (0.0, float(t_eval[-1])), u0, method="DOP853", t_eval=t_eval, rtol=1e-12, atol=1e-12)
    return sol.y.T


def lyapunov_exponent(n_renorm: int, dt: float = 1.0, transient: float = 50.0) -> float:
    u = solve_ivp(f, (0.0, transient), U0, method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
    d = np.array([1.0, 0.0, 0.0])
    acc = 0.0

    def rhs(t, w):
        return np.concatenate([f(t, w[:3]), jac(w[:3]) @ w[3:]])

    for _ in range(n_renorm):
        w = solve_ivp(rhs, (0.0, dt), np.concatenate([u, d]), method="DOP853", rtol=1e-10, atol=1e-12).y[:, -1]
        u, d = w[:3], w[3:]
        nd = np.linalg.norm(d)
        acc += np.log(nd)
        d = d / nd
    return acc / (n_renorm * dt)


def lorenz_residual(model, t_col):
    u = model(t_col)
    du = derivative(u, t_col)
    x, y, z = u[:, 0:1], u[:, 1:2], u[:, 2:3]
    rhs = torch.cat([SIG * (y - x), x * (RHO - z) - y, x * y - BETA * z], dim=1)
    return (du - rhs) / SCALE


def train_window(ex, a, b, u_start, width, M, n_adam, n_lbfgs):
    net = MLP(1, 3, width, 3, "tanh", domain=(a, b))
    u0 = torch.as_tensor(u_start, dtype=torch.get_default_dtype()).reshape(1, 3)
    model = ConstrainedModel(net, lambda t, n: u0 + SCALE * (t - a) * n)
    t_col = to_tensor(np.linspace(a, b, M), requires_grad=True)

    def loss_fn():
        r = lorenz_residual(model, t_col)
        L = (r ** 2).mean()
        return L, {"residual": L}

    h = train(net.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5, lbfgs_iters=n_lbfgs,
              print_every=0, verbose=False)
    return model, h


def predict(pieces, t):
    out = np.empty((len(t), 3))
    for a, b, model in pieces:
        m = (t >= a) & (t <= b)
        if np.any(m):
            with torch.no_grad():
                out[m] = model(to_tensor(t[m])).numpy()
    return out


def main():
    ex = Example("ch09_training_diagnostics", "lorenz_chaos",
                 description="Lorenz system: Lyapunov exponent, twin experiment, global and time-marching PINNs",
                 default_seed=1234,
                 extra_args=[replot_arg()])
    n_adam, n_lbfgs = ex.pick(4000, 40), ex.pick(500, 10)
    n_adam_w, n_lbfgs_w = ex.pick(2000, 20), ex.pick(500, 5)
    t_list = T_LIST if not ex.quick else [1.0]
    T_march = 5.0 if not ex.quick else 1.0
    n_windows = 20 if not ex.quick else 2
    width_w = 48

    cache = RunCache(ex)
    lam1 = cache.get("lyapunov", lambda: lyapunov_exponent(ex.pick(1000, 20)))
    ex.log(f"largest Lyapunov exponent = {lam1:.4f}")
    t_twin = np.linspace(0.0, 30.0, 3001)
    sep = np.linalg.norm(reference(t_twin) - reference(t_twin, U0 + np.array([1e-8, 0.0, 0.0])), axis=1)
    t5 = np.linspace(0.0, 5.0, 1001)
    u5 = reference(t5)
    mu2 = np.array([np.max(np.linalg.eigvalsh(0.5 * (jac(u) + jac(u).T))) for u in u5])

    rows, store = [], {}
    for T in t_list:
        M = int(100 * T + 100)
        t = np.linspace(0.0, T, 1001)

        def job(T=T, M=M, t=t):
            torch.manual_seed(ex.seed)
            with CPUTimer() as tm:
                model, h = train_window(ex, 0.0, T, U0, 64, M if not ex.quick else 50, n_adam, n_lbfgs)
            return {"u_pred": predict([(0.0, T, model)], t), "h": h, "cpu": tm.cpu}
        res = cache.get(f"global_T{T:g}", job)
        u_pred, h = res["u_pred"], res["h"]
        u_ref = reference(t)
        err = np.linalg.norm(u_pred - u_ref, axis=1)
        t_div = float(t[np.argmax(err > DIV_TOL)]) if np.any(err > DIV_TOL) else None
        store[("global", T)] = {"t": t, "u": u_pred, "err": err}
        ex.save_history(h, f"global_T{T:g}")
        rows.append({"method": "global", "T": T, "rel_l2": rel_l2(u_pred, u_ref), "max_error": float(err.max()),
                     "divergence_time": t_div, "final_loss": h.final_loss, "collocation": M,
                     "train_time_s": h.train_time, "cpu_time_s": res["cpu"]})
        ex.log(f"global T={T:g}: relL2={rows[-1]['rel_l2']:.2e} max err={err.max():.2e} t_div={t_div} "
               f"loss={h.final_loss:.2e}")

    # time-marching on [0, T_march]
    t = np.linspace(0.0, T_march, 1001)

    def march():
        torch.manual_seed(ex.seed)
        edges = np.linspace(0.0, T_march, n_windows + 1)
        pieces, u_start, hists = [], U0.copy(), []
        with CPUTimer() as tm:
            for k in range(n_windows):
                a, b = float(edges[k]), float(edges[k + 1])
                model, h = train_window(ex, a, b, u_start, width_w, ex.pick(100, 30), n_adam_w, n_lbfgs_w)
                with torch.no_grad():
                    u_start = model(to_tensor([b])).numpy().ravel()
                pieces.append((a, b, model))
                hists.append(h)
        return {"u_pred": predict(pieces, t), "hists": hists, "cpu": tm.cpu}
    res = cache.get("marching", march)
    u_pred, hists = res["u_pred"], res["hists"]
    u_ref = reference(t)
    err = np.linalg.norm(u_pred - u_ref, axis=1)
    t_div = float(t[np.argmax(err > DIV_TOL)]) if np.any(err > DIV_TOL) else None
    store[("marching", T_march)] = {"t": t, "u": u_pred, "err": err}
    for k, h in enumerate(hists):
        ex.save_history(h, f"marching_w{k}")
    rows.append({"method": "marching", "T": T_march, "rel_l2": rel_l2(u_pred, u_ref), "max_error": float(err.max()),
                 "divergence_time": t_div, "final_loss": float(np.max([h.final_loss for h in hists])),
                 "windows": n_windows, "train_time_s": float(sum(h.train_time for h in hists)),
                 "cpu_time_s": res["cpu"]})
    ex.log(f"marching T={T_march:g}: relL2={rows[-1]['rel_l2']:.2e} max err={err.max():.2e} t_div={t_div}")
    # growth rate of the marching error (least-squares slope of log error on the second half)
    m = (t > 0.5) & (err > 0)
    slope = float(np.polyfit(t[m], np.log(err[m]), 1)[0]) if np.sum(m) > 10 else None

    ex.save_table(rows, "summary")
    ex.save_table({"t": t_twin, "separation": sep}, "twin_separation")
    ex.save_table({"t": t5, "mu2": mu2}, "log_norm")
    ex.save_results({
        "problem": {"sigma": SIG, "rho": RHO, "beta": BETA, "u0": U0.tolist(), "reference": "DOP853 1e-12"},
        "lyapunov_exponent": lam1, "lyapunov_time": 1.0 / lam1,
        "horizon_estimate_tol_1e-12": float(np.log(1e12) / lam1), "lyapunov_method": "Benettin, renormalization every 1.0, 1000 steps, transient 50",
        "log_norm_max_on_0_5": float(mu2.max()), "log_norm_mean_on_0_5": float(mu2.mean()),
        "twin_initial_separation": 1e-8,
        "twin_time_separation_reaches_1": float(t_twin[np.argmax(sep > 1.0)]) if np.any(sep > 1.0) else None,
        "marching_error_growth_rate": slope,
        "hyperparameters": {"global": {"width": 64, "depth": 3, "collocation": "100 T + 100", "adam_iters": n_adam,
                                       "lbfgs_iters": n_lbfgs},
                            "marching": {"windows": n_windows, "width": width_w, "depth": 3, "collocation": 100,
                                         "adam_iters": n_adam_w, "lbfgs_iters": n_lbfgs_w},
                            "trial": "u0 + 20 (t - a) N(t)", "residual_scale": SCALE,
                            "divergence_threshold": DIV_TOL},
        "summary": rows})
    cache.clear()
    tab = [["global" if r["method"] == "global" else f"time-marching ({r['windows']} windows)",
            f"\\num{{{r['T']:g}}}", num(r["rel_l2"]), num(r["max_error"]),
            ("--" if r["divergence_time"] is None else f"\\num{{{r['divergence_time']:.2f}}}")] for r in rows]
    latex_table(ex.path("tab_lorenz.tex"), "lrrrr",
                ["PINN", "$T$", r"rel.\ $L^2$", "max error", r"$t$ with error $>1$"], tab,
                comment="Chapter 9, lorenz_chaos.py")

    if ex.args.no_figures:
        return
    fig, axes = P.figure(2, 2, aspect=0.74)
    Tm = t_list[-1]
    ax = axes[0, 0]
    P.plot_reference(ax, t5[t5 <= T_march], u5[t5 <= T_march, 0], label="reference")
    d = store[("global", Tm)]
    P.plot_prediction(ax, d["t"], d["u"][:, 0], label=f"global PINN, $T={Tm:g}$", k=0, lw=1.2)
    d = store[("marching", T_march)]
    P.plot_prediction(ax, d["t"], d["u"][:, 0], label="time-marching PINN", k=2, lw=1.2)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x(t)$")
    ax.set_title("first component")
    ax.legend(loc="lower left", fontsize=6.5)

    ax = axes[0, 1]
    for k, T in enumerate(t_list):
        d = store[("global", T)]
        P.plot_error(ax, d["t"], d["err"], label=f"global, $T={T:g}$", k=k, ylabel=r"$|u_\theta(t)-u(t)|$")
    d = store[("marching", T_march)]
    P.plot_error(ax, d["t"], d["err"], label="time-marching", k=3, ylabel=r"$|u_\theta(t)-u(t)|$")
    tg = np.linspace(0.5, T_march, 50)
    e_anchor = d["err"][np.argmin(np.abs(d["t"] - 0.5))]
    ax.semilogy(tg, e_anchor * np.exp(lam1 * (tg - 0.5)), color=P.COLORS["ref"], lw=0.8, ls=":",
                label=r"$\propto e^{\lambda_1 t}$")
    lo = min(float(np.min(store[k_]["err"][store[k_]["t"] > 0.02])) for k_ in store)
    ax.set_ylim(bottom=max(lo / 3.0, 1e-12))
    ax.set_title("state error")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.66), fontsize=6)

    ax = axes[1, 0]
    ax.semilogy(t_twin, np.maximum(sep, 1e-16), color=P.SERIES[0], lw=1.0, label=r"$|u(t)-\tilde u(t)|$")
    ax.semilogy(t_twin[t_twin < 22], 1e-8 * np.exp(lam1 * t_twin[t_twin < 22]), color=P.COLORS["ref"], lw=0.8,
                ls=":", label=rf"$10^{{-8}}e^{{\lambda_1 t}}$, $\lambda_1={lam1:.2f}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("separation")
    ax.set_title(r"twin trajectories, $|\delta u(0)| = 10^{-8}$")
    ax.legend(loc="lower right", fontsize=6.5)

    ax = axes[1, 1]
    P.plot_reference(ax, u5[:, 0], u5[:, 2], label="reference")
    d = store[("global", Tm)]
    P.plot_prediction(ax, d["u"][:, 0], d["u"][:, 2], label=f"global PINN, $T={Tm:g}$", k=0, lw=1.1)
    d = store[("marching", T_march)]
    P.plot_prediction(ax, d["u"][:, 0], d["u"][:, 2], label="time-marching PINN", k=2, lw=1.1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_title(r"$(x,z)$ projection, $t\in[0,5]$")
    ax.legend(loc="upper left", fontsize=6.5)
    P.label_panels(axes)
    ex.savefig(fig, "lorenz_summary")


if __name__ == "__main__":
    main()
