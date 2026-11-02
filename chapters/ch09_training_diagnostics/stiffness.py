"""Stiffness: y' = -lambda (y - cos t), y(0) = 0 on [0, 2 pi] for lambda = 1, 10, 100, 1000.

Book: Chapter 9, Section "Stiffness".

Exact solution  y(t) = A cos t + B sin t - A exp(-lambda t),  A = lambda^2/(1+lambda^2),
B = lambda/(1+lambda^2): an initial layer of width 1/lambda followed by the slow manifold.

PINN variants (tanh MLP, width 32, depth 3, M = 200 collocation points):

* vanilla   - soft initial condition (weight 1), raw residual, uniform collocation in t;
* hard      - hard initial condition y = t N(t) and residual scaled by 1/lambda;
* stretched - as "hard", but the network sees the stretched time
              tau = log(1 + lambda t) / log(1 + lambda T), and the collocation points are
              uniform in tau (hence geometrically graded in t);
* marching  - two windows [0, t1], [t1, T] with t1 = min(10/lambda, T/4), one network each,
              hard initial condition carried from the first window to the second.

Classical references: Radau IIA (implicit, rtol = atol = 1e-10) for accuracy, and Radau IIA
versus the explicit RK45 method at the same tolerance 1e-6 for the step counts, which show the
classical signature of stiffness.
Errors: relative L2 on the uniform 1001-point grid, and maximum error on that grid
augmented by 200 geometrically spaced points in [1e-3/lambda, 10/lambda] (resolves the layer).

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/stiffness.py          (~3 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/stiffness.py --quick  (smoke test)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CPUTimer, RunCache, latex_table, num, replot_arg  # noqa: E402
from pinnode import (ConstrainedModel, Example, MLP, derivative, linf, rel_l2,  # noqa: E402
                     to_tensor, train)
from pinnode import plotting as P  # noqa: E402

T = 2.0 * math.pi
LAMBDAS = [1.0, 10.0, 100.0, 1000.0]
VARIANTS = ["vanilla", "hard", "stretched", "marching"]
LABELS = {"vanilla": "vanilla (soft IC)", "hard": "hard IC + scaled residual",
          "stretched": "stretched time", "marching": "time-marching"}
WIDTH, DEPTH = 32, 3


def exact(t, lam):
    A, B = lam ** 2 / (1 + lam ** 2), lam / (1 + lam ** 2)
    return A * np.cos(t) + B * np.sin(t) - A * np.exp(-lam * t)


def test_grids(lam):
    tu = np.linspace(0.0, T, 1001)
    tl = np.geomspace(1e-3 / lam, min(10.0 / lam, T), 200)
    return tu, np.unique(np.concatenate([tu, tl]))


def train_window(ex, model, params, t_col, lam, scale, n_adam, n_lbfgs, soft_ic=False):
    t0 = to_tensor([0.0])

    def loss_fn():
        y = model(t_col)
        r = (derivative(y, t_col) + lam * (y - torch.cos(t_col))) / scale
        L_r = (r ** 2).mean()
        if soft_ic:
            L_ic = (model(t0) ** 2).mean()
            return L_r + L_ic, {"residual": L_r, "ic": L_ic}
        return L_r, {"residual": L_r}

    return train(params, loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5, lbfgs_iters=n_lbfgs,
                 print_every=0, verbose=False)


def run_pinn(ex, variant, lam, M, n_adam, n_lbfgs):
    torch.manual_seed(ex.seed)
    hists = []
    if variant == "vanilla":
        net = MLP(1, 1, WIDTH, DEPTH, "tanh", domain=(0.0, T))
        t_col = to_tensor(np.linspace(0.0, T, M), requires_grad=True)
        hists.append(train_window(ex, net, net.parameters(), t_col, lam, 1.0, n_adam, n_lbfgs, soft_ic=True))
        predict = lambda t: net(to_tensor(t)).detach().numpy().ravel()  # noqa: E731
        t_used = [t_col.detach().numpy().ravel()]
    elif variant in ("hard", "stretched"):
        if variant == "hard":
            net = MLP(1, 1, WIDTH, DEPTH, "tanh", domain=(0.0, T))
            model = ConstrainedModel(net, lambda t, n: t * n)
            t_np = np.linspace(0.0, T, M)
        else:
            c = math.log1p(lam * T)
            net = MLP(1, 1, WIDTH, DEPTH, "tanh", domain=(0.0, 1.0))

            def model(t):  # y = tau N(tau), tau = log(1 + lam t)/log(1 + lam T)
                tau = torch.log1p(lam * t) / c
                return tau * net(tau)
            t_np = np.expm1(np.linspace(0.0, 1.0, M) * c) / lam
        t_col = to_tensor(t_np, requires_grad=True)
        hists.append(train_window(ex, model, net.parameters(), t_col, lam, lam, n_adam, n_lbfgs))
        predict = lambda t: model(to_tensor(t)).detach().numpy().ravel()  # noqa: E731
        t_used = [t_np]
    elif variant == "marching":
        t1 = min(10.0 / lam, T / 4)
        windows, y_start, parts, t_used = [(0.0, t1), (t1, T)], 0.0, [], []
        for a, b in windows:
            net = MLP(1, 1, WIDTH, DEPTH, "tanh", domain=(a, b))
            model = ConstrainedModel(net, (lambda a_, ya_: (lambda t, n: ya_ + (t - a_) * n))(a, y_start))
            t_np = np.linspace(a, b, M // 2)
            t_col = to_tensor(t_np, requires_grad=True)
            hists.append(train_window(ex, model, net.parameters(), t_col, lam, lam, n_adam // 2, n_lbfgs // 2))
            y_start = float(model(to_tensor([b])).detach())
            parts.append((a, b, model))
            t_used.append(t_np)

        def predict(t):
            out = np.empty_like(t, dtype=float)
            for a, b, mdl in parts:
                m = (t >= a) & (t <= b)
                if np.any(m):
                    out[m] = mdl(to_tensor(t[m])).detach().numpy().ravel()
            return out
    else:
        raise ValueError(variant)
    return predict, hists, t_used


def classical(lam):
    f = lambda t, y: -lam * (y - np.cos(t))  # noqa: E731
    jac = lambda t, y: np.array([[-lam]])  # noqa: E731
    tu, tall = test_grids(lam)
    out = {}
    for name, method, tol, kw in [("radau", "Radau", 1e-10, {"jac": jac}), ("radau6", "Radau", 1e-6, {"jac": jac}),
                                  ("rk45", "RK45", 1e-6, {})]:
        with CPUTimer() as tm:
            sol = solve_ivp(f, (0.0, T), [0.0], method=method, rtol=tol, atol=tol, dense_output=True, **kw)
        y_all = sol.sol(tall)[0]
        out[name] = {"steps": int(len(sol.t) - 1), "nfev": int(sol.nfev),
                     "linf": linf(y_all, exact(tall, lam)),
                     "rel_l2": rel_l2(sol.sol(tu)[0], exact(tu, lam)), "cpu_time_s": tm.cpu}
    return out


def main():
    ex = Example("ch09_training_diagnostics", "stiffness",
                 description="Stiff linear IVP y'=-lambda(y-cos t): PINN variants vs Radau",
                 default_seed=1234,
                 extra_args=[replot_arg()])
    M = ex.pick(200, 40)
    n_adam, n_lbfgs = ex.pick(4000, 40), ex.pick(400, 10)
    rows, preds = [], {}
    cache = RunCache(ex)
    for lam in LAMBDAS:
        tu, tall = test_grids(lam)
        yu, yall = exact(tu, lam), exact(tall, lam)
        cl = classical(lam)
        for v in VARIANTS:
            def job(v=v, lam=lam, tu=tu, tall=tall):
                with CPUTimer() as tm:
                    predict, hists, t_used = run_pinn(ex, v, lam, M, n_adam, n_lbfgs)
                return {"pu": predict(tu), "pall": predict(tall), "hists": hists, "t_used": t_used,
                        "cpu": tm.cpu}
            res = cache.get(f"lam{int(lam)}_{v}", job)
            pu, pall, hists, t_used = res["pu"], res["pall"], res["hists"], res["t_used"]
            e_rel, e_inf = rel_l2(pu, yu), linf(pall, yall)
            preds[(lam, v)] = {"t": tall, "y": pall, "hists": hists, "t_col": t_used}
            for k, h in enumerate(hists):
                ex.save_history(h, f"lam{int(lam)}_{v}" + (f"_w{k}" if len(hists) > 1 else ""))
            rows.append({"lambda": lam, "variant": v, "rel_l2": e_rel, "linf_layer_grid": e_inf,
                         "abs_error_t0": float(abs(pall[0] - yall[0])),
                         "final_loss": hists[-1].final_loss,
                         "train_time_s": float(sum(h.train_time for h in hists)), "cpu_time_s": res["cpu"]})
            ex.log(f"lambda={lam:6g} {v:9s}: relL2={e_rel:.2e} linf(layer grid)={e_inf:.2e} cpu={res['cpu']:.0f}s")
        for name in ("radau", "radau6", "rk45"):
            c = cl[name]
            rows.append({"lambda": lam, "variant": name, "rel_l2": c["rel_l2"], "linf_layer_grid": c["linf"],
                         "steps": c["steps"], "nfev": c["nfev"], "cpu_time_s": c["cpu_time_s"]})
            ex.log(f"lambda={lam:6g} {name:9s}: linf={c['linf']:.2e} steps={c['steps']} nfev={c['nfev']}")
        ex.save_table({"t": tall, "exact": yall, **{v: preds[(lam, v)]["y"] for v in VARIANTS}},
                      f"predictions_lam{int(lam)}")
    ex.save_table(rows, "summary")
    ex.save_results({
        "problem": {"ode": "y' = -lambda (y - cos t), y(0) = 0", "T": T, "lambdas": LAMBDAS,
                    "exact": "A cos t + B sin t - A exp(-lambda t), A=l^2/(1+l^2), B=l/(1+l^2)"},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "collocation": M, "adam_iters": n_adam,
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": n_lbfgs, "w_ic_vanilla": 1.0,
                            "marching_t1": "min(10/lambda, T/4)",
                            "test_grids": "uniform 1001 points; plus 200 geometric points in [1e-3/l, 10/l]"},
        "classical": {"radau_tol_reference": 1e-10, "radau_tol_steps": 1e-6, "rk45_tol": 1e-6},
        "summary": rows})
    cache.clear()

    # ---------------------------------------------------------------- LaTeX table (accuracy + steps)
    def get(lam, v, key):
        for r in rows:
            if r["lambda"] == lam and r["variant"] == v:
                return r.get(key)
        return None
    tab = []
    for lam in LAMBDAS:
        tab.append([f"\\num{{{int(lam)}}}"] + [num(get(lam, v, "linf_layer_grid")) for v in VARIANTS]
                   + [num(get(lam, "radau", "linf_layer_grid")),
                      f"{get(lam, 'radau6', 'steps')}", f"{get(lam, 'rk45', 'steps')}"])
    latex_table(ex.path("tab_stiffness.tex"), "rrrrrrrr",
                [r"$\lambda$", "vanilla", "hard", "stretched", "marching", "Radau",
                 r"\multicolumn{1}{c}{\shortstack{Radau\\steps}}",
                 r"\multicolumn{1}{c}{\shortstack{RK45\\steps}}"], tab,
                comment="Chapter 9, stiffness.py: maximum error on the layer-resolving grid")

    if ex.args.no_figures:
        return
    # ---------------------------------------------------------------- figure
    fig, axes = P.figure(2, 2, aspect=0.74)
    lam = 1000.0
    ax = axes[0, 0]
    d = preds[(lam, "vanilla")]
    mask = d["t"] > 0
    P.plot_reference(ax, d["t"][mask], exact(d["t"][mask], lam), label="exact")
    for k, v in enumerate(VARIANTS):
        dd = preds[(lam, v)]
        P.plot_prediction(ax, dd["t"][mask], dd["y"][mask], label=LABELS[v], k=k, lw=1.3)
    ax.set_xscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(r"$\lambda = 1000$ (log time axis)")
    ax.legend(loc="lower center", fontsize=6.5)

    ax = axes[0, 1]
    for k, v in enumerate(VARIANTS):
        dd = preds[(lam, v)]
        e = np.abs(dd["y"][mask] - exact(dd["t"][mask], lam))
        P.plot_error(ax, dd["t"][mask], e, label=LABELS[v], k=k)
    ax.set_xscale("log")
    ax.set_title(r"pointwise error, $\lambda = 1000$")
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    for k, v in enumerate(VARIANTS):
        ax.loglog(LAMBDAS, [get(l_, v, "linf_layer_grid") for l_ in LAMBDAS], color=P.SERIES[k],
                  ls=P.LINESTYLES[k], marker=P.MARKERS[k], markerfacecolor="white", label=LABELS[v])
    ax.loglog(LAMBDAS, [get(l_, "radau", "linf_layer_grid") for l_ in LAMBDAS], color=P.COLORS["ref"],
              ls="-", lw=1.0, marker="x", label="Radau")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("maximum error")
    ax.set_title("accuracy against stiffness")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.38), fontsize=6.5)

    ax = axes[1, 1]
    ax.loglog(LAMBDAS, [get(l_, "rk45", "steps") for l_ in LAMBDAS], color=P.SERIES[0], ls="--",
              marker="o", markerfacecolor="white", label=r"RK45 (explicit), tol $10^{-6}$")
    ax.loglog(LAMBDAS, [get(l_, "radau6", "steps") for l_ in LAMBDAS], color=P.SERIES[1], ls="-.",
              marker="s", markerfacecolor="white", label=r"Radau IIA (implicit), tol $10^{-6}$")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("accepted steps")
    ax.set_title("classical solvers")
    ax.legend(loc="upper left")
    P.label_panels(axes)
    ex.savefig(fig, "stiffness_summary")


if __name__ == "__main__":
    main()
