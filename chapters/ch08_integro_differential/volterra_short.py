"""Volterra IDE  u'(x) = u(x) + int_0^x t u(t) dt,  u(0) = 1,  on [0, 2]:
direct-quadrature PINN versus auxiliary-variable PINN (A-PINN), both with hard initial conditions.

Book: Chapter 8 (Integro-differential equations), Section "Volterra integro-differential
equations", worked example on [0, 2]: Figure ide:fig:volterra, Table ide:tab:volterra.

Formulations
  * direct quadrature: u_theta(x) = 1 + x N(x) and
        R(x_i) = u_theta'(x_i) - u_theta(x_i) - sum_j W_ij s_ij u_theta(s_ij),
    where (s_ij, W_ij) is the 20-point Gauss-Legendre rule on [0, x_i];
  * A-PINN: two outputs (u_theta, A_theta) = (1 + x N_1(x), x N_2(x)) for u and the
    memory variable A(x) = int_0^x t u(t) dt, with the local residuals
        R_1 = u' - u - A,     R_2 = A' - x u.
Exact solution (Airy functions):
    u(x) = e^{x/2} [c1 Ai(x + 1/4) + c2 Bi(x + 1/4)],
    c1 = pi (Bi'(1/4) - Bi(1/4)/2) = 0.3313995...,  c2 = pi (Ai(1/4)/2 - Ai'(1/4)) = 1.2398110...
Classical references: implicit trapezoidal direct-quadrature method with N steps and DOP853
(rtol = atol = 1e-12) for the equivalent ODE system (u, A)' = (u + A, x u).

A posteriori check: for each trained network the IDE residual r = u' - u - int_0^x t u dt is
evaluated on the test grid with an independent 40-point Gauss rule, and the residual-to-error
bound  max|u_theta - u| <= exp(L + L^3/6) ||r||_{L^1(0, L)}  of the chapter is evaluated.

Errors are measured on a uniform grid of 1001 points in [0, 2].

Run:  PYTHONPATH=. python chapters/ch08_integro_differential/volterra_short.py          (~1 min, 1 core)
      PYTHONPATH=. python chapters/ch08_integro_differential/volterra_short.py --quick  (smoke test)
"""
from __future__ import annotations

import time
import math

import numpy as np
import torch

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, to_tensor,
                     train, uniform, volterra_gauss)
from pinnode import plotting as P

from _ide_common import (RESUME_ARG, VOLTERRA_C1, VOLTERRA_C2, CpuWallTimer, cached, dop853_system,
                         errors, time_call, trapezoidal_vide, volterra_exact, volterra_exact_aux,
                         volterra_rhs)

L = 2.0                    # interval [0, L]
WIDTH, DEPTH = 32, 3       # tanh MLP
NQ = 20                    # Gauss nodes on [0, x_i] in the training residual
NQ_CHECK = 40              # independent rule for the a posteriori residual
N_TEST = 1001


def kernel(x, t):
    """K(x, t) = t."""
    return t


# ----------------------------------------------------------------------------- models
def build_direct():
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, L))
    return ConstrainedModel(net, lambda x, n: 1.0 + x * n)                  # u(0) = 1 exactly


def build_apinn():
    net = MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, L))
    return ConstrainedModel(net, lambda x, n: torch.cat([1.0 + x * n[:, :1],  # u(0) = 1
                                                         x * n[:, 1:2]], dim=1))  # A(0) = 0


class MemoryQuadrature:
    """sum_j W_ij K(x_i, s_ij) u(s_ij) ~ int_0^{x_i} K(x_i, t) u(t) dt for all x_i at once."""

    def __init__(self, x: np.ndarray, n: int):
        S, W = volterra_gauss(x, n)                      # shapes (M, n)
        self.shape = S.shape
        self.nodes = to_tensor(S.reshape(-1))            # (M n, 1)
        self.weights = torch.as_tensor(W * kernel(np.asarray(x).reshape(-1, 1), S))

    def __call__(self, u_fn) -> torch.Tensor:
        return (self.weights * u_fn(self.nodes).reshape(self.shape)).sum(dim=1, keepdim=True)


# ----------------------------------------------------------------------------- losses
def direct_loss(model, xc, memory):
    def loss_fn():
        u = model(xc)
        r = derivative(u, xc) - u - memory(model)
        L_r = (r ** 2).mean()
        return L_r, {"residual": L_r}
    return loss_fn


def apinn_loss(model, xc):
    def loss_fn():
        out = model(xc)
        d = derivative(out, xc)
        u, A = out[:, :1], out[:, 1:2]
        r1 = d[:, :1] - u - A
        r2 = d[:, 1:2] - xc * u
        L1, L2 = (r1 ** 2).mean(), (r2 ** 2).mean()
        return L1 + L2, {"R1": L1, "R2": L2}
    return loss_fn


def ide_residual(u_fn, x_np: np.ndarray) -> np.ndarray:
    """r(x) = u'(x) - u(x) - int_0^x t u(t) dt (40-point Gauss rule for the memory term)."""
    x = to_tensor(x_np, requires_grad=True)
    u = u_fn(x)
    mem = MemoryQuadrature(x_np, NQ_CHECK)(u_fn)
    return (derivative(u, x) - u - mem).detach().numpy().ravel()


# ----------------------------------------------------------------------------- one run
def run(ex: Example, name: str, x_test: np.ndarray, u_ref: np.ndarray):
    torch.manual_seed(ex.seed)
    M = ex.pick(100, 30)
    x_col = uniform(M, 0.0, L)
    xc = to_tensor(x_col, requires_grad=True)
    if name == "direct":
        model = build_direct()
        loss_fn = direct_loss(model, xc, MemoryQuadrature(x_col, NQ))
        u_fn = model
    else:
        model = build_apinn()
        loss_fn = apinn_loss(model, xc)
        u_fn = lambda x: model(x)[:, :1]                                    # noqa: E731
    xt = to_tensor(x_test)

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": errors(u_fn(xt).numpy(), u_ref)["rel_l2"]}

    n_adam, n_lbfgs = ex.pick(5000, 60), ex.pick(1000, 10)
    with CpuWallTimer() as tm:
        hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                     lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50, print_every=2500)
    with torch.no_grad():
        out = model(xt).numpy()
    u_pred = out[:, 0]
    r = ide_residual(u_fn, x_test)
    res = {"model": model, "hist": hist, "u": u_pred, "r": r,
           "A": out[:, 1] if out.shape[1] > 1 else None,
           "errors": errors(u_pred, u_ref), "cpu_time_s": tm.cpu, "train_time_s": hist.train_time,
           "iterations": int(hist.iters[-1]) + 1, "final_loss": hist.final_loss,
           "parameters": count_parameters(model), "collocation": M}
    res["cpu_per_iter_ms"] = 1e3 * tm.cpu / res["iterations"]
    # residual-to-error bound  max|e| <= exp(L + L^3/6) ||r||_{L^1}
    r_l1 = float(np.trapezoid(np.abs(r), x_test))
    res["bound"] = {"r_L1": r_l1, "amplification": math.exp(L + L ** 3 / 6.0),
                    "bound": math.exp(L + L ** 3 / 6.0) * r_l1,
                    "max_abs_error": res["errors"]["linf"]}
    return res


def classical(x_test: np.ndarray, u_ref: np.ndarray, quick: bool):
    rows = []
    for N in ((100, 1000) if not quick else (100,)):
        (x, u), cpu, wall = time_call(lambda: trapezoidal_vide(lambda s: 1.0, kernel, lambda s: 0.0,
                                                               1.0, L, N), min_time=0.2)
        rows.append({"method": f"trapezoidal, N={N}", "N": N, **errors(u, volterra_exact(x)),
                     "cpu_time_s": cpu, "wall_time_s": wall})
    Y, cpu, wall = time_call(lambda: dop853_system(volterra_rhs, [1.0, 0.0], L, x_test), min_time=0.2)
    rows.append({"method": "DOP853 (u, A) system", "N": None, **errors(Y[:, 0], u_ref),
                 "cpu_time_s": cpu, "wall_time_s": wall})
    return rows


def main():
    ex = Example("ch08_integro_differential", "volterra_short",
                 description="Volterra IDE on [0,2]: direct quadrature vs A-PINN (hard ICs)",
                 default_seed=1234, extra_args=[RESUME_ARG])
    x_test = np.linspace(0.0, L, N_TEST)
    u_ref, A_ref = volterra_exact(x_test), volterra_exact_aux(x_test)
    runs = {}
    for name in ("direct", "apinn"):
        ex.log(f"training {name}")
        runs[name] = cached(ex, name, lambda: run(ex, name, x_test, u_ref))
        ex.save_history(runs[name]["hist"], name)
        e = runs[name]["errors"]
        ex.log(f"{name}: relL2={e['rel_l2']:.2e} max_rel={e['max_rel']:.2e} "
               f"cpu={runs[name]['cpu_time_s']:.1f}s bound={runs[name]['bound']['bound']:.2e}")
    cls = classical(x_test, u_ref, ex.quick)
    for row in cls:
        ex.log(f"{row['method']}: relL2={row['rel_l2']:.2e} cpu={row['cpu_time_s']:.2e}s")

    ex.save_table({"x": x_test, "u_exact": u_ref, "A_exact": A_ref,
                   "u_direct": runs["direct"]["u"], "u_apinn": runs["apinn"]["u"],
                   "A_apinn": runs["apinn"]["A"], "r_direct": runs["direct"]["r"],
                   "r_apinn": runs["apinn"]["r"]}, "predictions")
    keys = ("errors", "cpu_time_s", "train_time_s", "iterations", "cpu_per_iter_ms", "final_loss",
            "parameters", "collocation", "bound")
    pinn = {n: {k: runs[n][k] for k in keys} for n in runs}
    pinn["apinn"]["errors_A"] = errors(runs["apinn"]["A"], A_ref)
    best_classical = min(r["cpu_time_s"] for r in cls)
    ex.save_results({
        "problem": {"equation": "u' = u + int_0^x t u(t) dt, u(0) = 1", "interval": [0.0, L],
                    "c1": VOLTERRA_C1, "c2": VOLTERRA_C2,
                    "u_at_L": float(u_ref[-1]), "test_grid": N_TEST},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "collocation": "uniform, M = %d" % ex.pick(100, 30),
                            "quadrature_nodes": NQ, "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters_max": ex.pick(1000, 10),
                            "initial_conditions": "hard"},
        "pinn": pinn,
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
    labels = {"direct": "direct quadrature", "apinn": "A-PINN"}
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, x, u_ref, label="exact")
    for k, n in enumerate(("direct", "apinn")):
        P.plot_prediction(ax, x, runs[n]["u"], label=labels[n], k=k, markevery=(40 + 60 * k, 120))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")
    ax.legend(loc="upper left")
    ax.set_title("solution")

    ax = axes[0, 1]
    for k, n in enumerate(("direct", "apinn")):
        P.plot_error(ax, x[1:], (runs[n]["u"][1:] - u_ref[1:]) / u_ref[1:], label=labels[n], k=k,
                     xlabel="$x$", ylabel="relative error")
    ax.set_title("pointwise relative error")

    ax = axes[1, 0]
    hd, ha = runs["direct"]["hist"], runs["apinn"]["hist"]
    w = max(5, len(hd.iters) // 60)
    series = [(hd.iters, hd.loss, "direct quadrature", P.COLORS["ref"], "-"),
              (ha.iters, ha.parts["R1"], "A-PINN: $R_1$ term", P.SERIES[1], P.LINESTYLES[0]),
              (ha.iters, ha.parts["R2"], "A-PINN: $R_2$ term", P.SERIES[2], P.LINESTYLES[1])]
    for it, y, lab, c, ls in series:
        y = np.asarray(y)
        ax.semilogy(it, y, color=c, lw=0.5, alpha=0.25)
        ax.semilogy(it, P.rolling_median(y, w), color=c, lw=1.2, ls=ls, label=lab)
    if hd.switch_iter:
        ax.axvline(hd.switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    ax.set_title("loss components")

    ax = axes[1, 1]
    for k, n in enumerate(("direct", "apinn")):
        P.plot_error(ax, x, runs[n]["r"], label=labels[n], k=k, xlabel="$x$",
                     ylabel="$|r(x)|$")
    ax.set_title("a posteriori IDE residual")
    P.label_panels(axes)
    ex.savefig(fig, "volterra_short")


if __name__ == "__main__":
    main()
