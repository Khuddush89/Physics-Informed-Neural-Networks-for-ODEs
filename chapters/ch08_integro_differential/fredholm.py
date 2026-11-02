"""Fredholm IDE  u'(x) = 1 + int_0^1 x t u(t) dt,  u(0) = 0,  on [0, 1]:
global-quadrature PINN versus moment-variable (A-PINN) formulation.

Book: Chapter 8 (Integro-differential equations), Section "Fredholm integro-differential
equations", worked example: Figure ide:fig:fredholm, Table ide:tab:fredholm.

Exact solution: with the moment C = int_0^1 t u(t) dt the equation reads u' = 1 + C x, so
u = x + C x^2 / 2, and the self-consistency condition C = 1/3 + C/8 gives C = 8/21, i.e.
    u(x) = x + (4/21) x^2.

Formulations (hard initial condition u_theta(x) = x N(x)):
  * global quadrature: C_theta = sum_j w_j t_j u_theta(t_j) with the 32-point Gauss-Legendre
    rule on [0, 1] and R(x_i) = u_theta'(x_i) - 1 - x_i C_theta;
  * moment variable: C is an additional trainable scalar (initialized at 0),
    R(x_i) = u_theta'(x_i) - 1 - x_i C, and the loss contains the consistency penalty
    (C - sum_j w_j t_j u_theta(t_j))^2.
Classical reference: integrating the IDE from 0 to x gives the Fredholm integral equation of
the second kind  u(x) = x + int_0^1 (x^2 t / 2) u(t) dt,  solved by the Nystrom method with
8 Gauss-Legendre nodes (pinnode.reference.nystrom_fredholm).

Errors are measured on a uniform grid of 1001 points in [0, 1].

Run:  PYTHONPATH=. python chapters/ch08_integro_differential/fredholm.py          (~1 min, 1 core)
      PYTHONPATH=. python chapters/ch08_integro_differential/fredholm.py --quick  (smoke test)
"""
from __future__ import annotations

import time
import numpy as np
import torch

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, gauss_legendre,
                     to_tensor, train, uniform)
from pinnode import plotting as P
from pinnode.reference import nystrom_fredholm

from _ide_common import FREDHOLM_C, RESUME_ARG, CpuWallTimer, cached, errors, fredholm_exact, time_call

WIDTH, DEPTH = 32, 3
NQ = 32                    # Gauss-Legendre nodes on [0, 1] for the moment
N_TEST = 1001
LABELS = {"direct": "global quadrature", "moment": "moment variable"}


def build():
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, 1.0))
    return ConstrainedModel(net, lambda x, n: x * n)                       # u(0) = 0 exactly


def run(ex: Example, name: str, x_test: np.ndarray, u_ref: np.ndarray):
    torch.manual_seed(ex.seed)
    model = build()
    M = ex.pick(100, 30)
    xc = to_tensor(uniform(M, 0.0, 1.0), requires_grad=True)
    tq, wq = gauss_legendre(NQ, 0.0, 1.0)
    tq_t, wtq = to_tensor(tq), to_tensor(wq * tq)          # nodes, weights * kernel factor t

    def moment_quadrature():                               # sum_j w_j t_j u_theta(t_j)
        return (wtq * model(tq_t)).sum()

    params = list(model.parameters())
    if name == "moment":
        C = torch.nn.Parameter(torch.zeros(()))
        params.append(C)

    def loss_fn():
        u = model(xc)
        Cq = moment_quadrature()
        Cr = Cq if name == "direct" else C
        r = derivative(u, xc) - 1.0 - xc * Cr
        L_r = (r ** 2).mean()
        if name == "direct":
            return L_r, {"residual": L_r}
        L_c = (C - Cq) ** 2
        return L_r + L_c, {"residual": L_r, "consistency": L_c}

    xt = to_tensor(x_test)

    def current_C():
        with torch.no_grad():
            return float(moment_quadrature()) if name == "direct" else float(C)

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": errors(model(xt).numpy(), u_ref)["rel_l2"],
                    "C_error": abs(current_C() - FREDHOLM_C)}

    n_adam, n_lbfgs = ex.pick(5000, 60), ex.pick(1000, 10)
    with CpuWallTimer() as tm:
        hist = train(params, loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                     lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50, print_every=2500)
    with torch.no_grad():
        u_pred = model(xt).numpy().ravel()
        C_quad = float(moment_quadrature())
    res = {"hist": hist, "u": u_pred, "errors": errors(u_pred, u_ref), "cpu_time_s": tm.cpu,
           "train_time_s": hist.train_time, "iterations": int(hist.iters[-1]) + 1,
           "final_loss": hist.final_loss, "parameters": count_parameters(model) + (name == "moment"),
           "C": current_C(), "C_error": abs(current_C() - FREDHOLM_C),
           "C_quadrature": C_quad, "C_quadrature_error": abs(C_quad - FREDHOLM_C)}
    res["cpu_per_iter_ms"] = 1e3 * tm.cpu / res["iterations"]
    return res


def classical(x_test: np.ndarray, u_ref: np.ndarray):
    """Nystrom method for u(x) = x + int_0^1 (x^2 t / 2) u(t) dt."""
    rows = []
    for n in (2, 8):
        def solve():
            u = nystrom_fredholm(lambda x, s: 0.5 * x ** 2 * s, lambda x: x, 1.0, 0.0, 1.0, n=n)
            return u(x_test)
        u, cpu, wall = time_call(solve, min_time=0.2)
        tq, wq = gauss_legendre(n, 0.0, 1.0)
        unodes = nystrom_fredholm(lambda x, s: 0.5 * x ** 2 * s, lambda x: x, 1.0, 0.0, 1.0, n=n)(tq)
        C_n = float(np.sum(wq * tq * unodes))
        rows.append({"method": f"Nystrom, n={n}", "n": n, **errors(u, u_ref),
                     "C_error": abs(C_n - FREDHOLM_C), "cpu_time_s": cpu, "wall_time_s": wall})
    return rows


def main():
    ex = Example("ch08_integro_differential", "fredholm",
                 description="Fredholm IDE: global quadrature vs moment variable (hard IC)",
                 default_seed=1234, extra_args=[RESUME_ARG])
    x_test = np.linspace(0.0, 1.0, N_TEST)
    u_ref = fredholm_exact(x_test)
    runs = {}
    for name in ("direct", "moment"):
        ex.log(f"training {name}")
        runs[name] = cached(ex, name, lambda: run(ex, name, x_test, u_ref))
        ex.save_history(runs[name]["hist"], name)
        e = runs[name]["errors"]
        ex.log(f"{name}: relL2={e['rel_l2']:.2e} linf={e['linf']:.2e} "
               f"|C-8/21|={runs[name]['C_error']:.2e} cpu={runs[name]['cpu_time_s']:.1f}s")
    cls = classical(x_test, u_ref)
    for row in cls:
        ex.log(f"{row['method']}: relL2={row['rel_l2']:.2e} cpu={row['cpu_time_s']:.2e}s")
    ex.save_table({"x": x_test, "u_exact": u_ref, "u_direct": runs["direct"]["u"],
                   "u_moment": runs["moment"]["u"]}, "predictions")
    keys = ("errors", "cpu_time_s", "train_time_s", "iterations", "cpu_per_iter_ms", "final_loss",
            "parameters", "C", "C_error", "C_quadrature", "C_quadrature_error")
    best_classical = min(r["cpu_time_s"] for r in cls)
    ex.save_results({
        "problem": {"equation": "u' = 1 + int_0^1 x t u(t) dt, u(0) = 0", "interval": [0.0, 1.0],
                    "exact": "u = x + 4 x^2 / 21", "C_exact": FREDHOLM_C, "test_grid": N_TEST},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "collocation": "uniform, M = %d" % ex.pick(100, 30),
                            "quadrature_nodes": NQ, "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters_max": ex.pick(1000, 10),
                            "initial_condition": "hard", "consistency_weight": 1.0},
        "pinn": {n: {k: runs[n][k] for k in keys} for n in runs},
        "classical": cls,
        "speed_ratio_pinn_over_fastest_classical": {
            n: runs[n]["cpu_time_s"] / best_classical for n in runs},
        "train_time_s": sum(runs[n]["train_time_s"] for n in runs),
        "cpu_time_training_s": sum(runs[n]["cpu_time_s"] for n in runs),
        "cpu_time_this_process_s": time.process_time(),   # small if variants were cached
    })
    if ex.args.no_figures:
        return
    plot(ex, x_test, u_ref, runs)


def plot(ex, x, u_ref, runs):
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, x, u_ref, label="exact")
    for k, n in enumerate(("direct", "moment")):
        P.plot_prediction(ax, x, runs[n]["u"], label=LABELS[n], k=k, markevery=(40 + 60 * k, 120))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")
    ax.legend(loc="upper left")
    ax.set_title("solution")

    ax = axes[0, 1]
    for k, n in enumerate(("direct", "moment")):
        P.plot_error(ax, x[1:], runs[n]["u"][1:] - u_ref[1:], label=LABELS[n], k=k, xlabel="$x$")
    ax.set_title("pointwise error")

    ax = axes[1, 0]
    hd, hm = runs["direct"]["hist"], runs["moment"]["hist"]
    w = max(5, len(hd.iters) // 60)
    series = [(hd.iters, hd.loss, "global quadrature", P.COLORS["ref"], "-"),
              (hm.iters, hm.parts["residual"], "moment: residual", P.SERIES[1], P.LINESTYLES[0]),
              (hm.iters, hm.parts["consistency"], "moment: consistency", P.SERIES[2], P.LINESTYLES[1])]
    for it, y, lab, c, ls in series:
        y = np.maximum(np.asarray(y), 1e-30)
        ax.semilogy(it, y, color=c, lw=0.5, alpha=0.25)
        ax.semilogy(it, P.rolling_median(y, w), color=c, lw=1.2, ls=ls, label=lab)
    if hd.switch_iter:
        ax.axvline(hd.switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.set_title("loss components")

    ax = axes[1, 1]
    for k, n in enumerate(("direct", "moment")):
        h = runs[n]["hist"]
        ax.semilogy(h.eval_iters, np.maximum(h.metrics["C_error"], 1e-17), color=P.SERIES[k],
                    ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.2, label=LABELS[n])
    ax.axvline(hd.switch_iter or 0, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("$|C_\\theta - 8/21|$")
    ax.legend(loc="upper right")
    ax.set_title("moment error")
    P.label_panels(axes)
    ex.savefig(fig, "fredholm")


if __name__ == "__main__":
    main()
