"""Volterra IDE  u'(x) = u(x) + int_0^x t u(t) dt,  u(0) = 1,  on the long interval [0, 5]:
rapid growth (u(5) = 1.73e4) and how the scaling of the unknowns decides success or failure.

Book: Chapter 8 (Integro-differential equations), Section "Volterra integro-differential
equations", subsection on rapid growth: Figure ide:fig:growth, Table ide:tab:volterra.

All PINNs impose the initial conditions exactly and use the same tanh MLP (width 32, depth 3),
200 uniform collocation points, 5000 Adam + <= 1000 L-BFGS iterations.

  unscaled   A-PINN for (u, A), A = int_0^x t u dt:  (u, A) = (1 + x N_1, x N_2),
             R_1 = u' - u - A,  R_2 = A' - x u.
  exp        the same system for u = e^{x/2} v, A = e^{x/2} b (the substitution that removes
             the first-order term of the reduced ODE u'' - u' - x u = 0):
             R_1 = v' - v/2 - b,  R_2 = b' + b/2 - x v      (= unscaled residuals / e^{x/2}).
  log        logarithmic variables w = log u, a = A/u (u > 0 follows from the equation):
             (w, a) = (x N_1, x N_2),  R_1 = w' - 1 - a,  R_2 = a' + a w' - x
             (= unscaled residuals / u, i.e. *relative* residuals).
  log-quad   direct quadrature in the variable w (single output, w = x N):
             R = w'(x) - 1 - int_0^x t exp(w(t) - w(x)) dt   (20-point Gauss rule on [0, x]).
  bifactor   (remark in the book; presupposes the Airy structure of the exact solution)
             u = e^{x/2} Bi(x + 1/4) y with y'' + 2 (Bi'/Bi)(x + 1/4) y' = 0,
             y = y(0) + y'(0) x + x^2 N(x),  y(0) = 1/Bi(1/4),  y'(0) = (1/2 - Bi'(1/4) y(0))/Bi(1/4).

Exact solution: u(x) = e^{x/2}[c1 Ai(x + 1/4) + c2 Bi(x + 1/4)] (see volterra_short.py).
Classical references: implicit trapezoidal direct-quadrature method (N = 1000, 10000) and DOP853
(rtol = atol = 1e-12) for the (u, A) system.  Errors on a uniform grid of 1001 points in [0, 5];
because u spans four orders of magnitude, the maximum *relative* error is the meaningful measure.

Run:  PYTHONPATH=. python chapters/ch08_integro_differential/volterra_growth.py          (~3 min, 1 core)
      PYTHONPATH=. python chapters/ch08_integro_differential/volterra_growth.py --quick  (smoke test)
"""
from __future__ import annotations

import time
import numpy as np
import torch
from scipy.special import airy

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, to_tensor,
                     train, uniform, volterra_gauss)
from pinnode import plotting as P

from _ide_common import (RESUME_ARG, VOLTERRA_C1, VOLTERRA_C2, CpuWallTimer, cached, dop853_system,
                         errors, time_call, trapezoidal_vide, volterra_exact, volterra_exact_aux,
                         volterra_rhs)

L = 5.0
WIDTH, DEPTH = 32, 3
NQ = 20
N_TEST = 1001
VARIANTS = ("unscaled", "exp", "log", "logquad", "bifactor")
LABELS = {"unscaled": "A-PINN, unscaled", "exp": "A-PINN, $u=e^{x/2}v$",
          "log": "A-PINN, log variables", "logquad": "quadrature, log variable",
          "bifactor": "Bi-factorized (remark)"}

_, _, BI0, BIP0 = (float(v) for v in airy(0.25))
Y0 = 1.0 / BI0
DY0 = (0.5 - BIP0 * Y0) / BI0


def decaying_solution(x):
    """Decaying solution e^{x/2} Ai(x + 1/4) / Ai(1/4) of the homogeneous system (u(0) = 1,
    but A(0) = u'(0) - 1 != 0); the failed formulations converge to functions of this type."""
    ai, _, _, _ = airy(np.asarray(x) + 0.25)
    return np.exp(0.5 * np.asarray(x)) * ai / float(airy(0.25)[0])


def mlp(out_dim: int):
    return MLP(1, out_dim, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, L))


def setup(name: str, x_col: np.ndarray):
    """Return (model, loss_fn, u_fn) for one formulation; u_fn maps x -> u_theta(x)."""
    xc = to_tensor(x_col, requires_grad=True)
    if name in ("unscaled", "exp"):
        model = ConstrainedModel(mlp(2), lambda x, n: torch.cat([1.0 + x * n[:, :1], x * n[:, 1:2]], 1))

        def loss_fn():
            out = model(xc)
            d = derivative(out, xc)
            p, q = out[:, :1], out[:, 1:2]
            if name == "unscaled":                   # p = u, q = A
                r1, r2 = d[:, :1] - p - q, d[:, 1:2] - xc * p
            else:                                    # p = v, q = b
                r1, r2 = d[:, :1] - 0.5 * p - q, d[:, 1:2] + 0.5 * q - xc * p
            L1, L2 = (r1 ** 2).mean(), (r2 ** 2).mean()
            return L1 + L2, {"R1": L1, "R2": L2}
        scale = (lambda x: 1.0) if name == "unscaled" else (lambda x: torch.exp(0.5 * x))
        u_fn = lambda x: scale(x) * model(x)[:, :1]                         # noqa: E731
        aux_fn = lambda x: model(x)[:, 1:2] / model(x)[:, :1]               # noqa: E731  A/u
    elif name == "log":
        model = ConstrainedModel(mlp(2), lambda x, n: x * n)              # w(0) = 0, a(0) = 0

        def loss_fn():
            out = model(xc)
            d = derivative(out, xc)
            a, dw, da = out[:, 1:2], d[:, :1], d[:, 1:2]
            r1, r2 = dw - 1.0 - a, da + a * dw - xc
            L1, L2 = (r1 ** 2).mean(), (r2 ** 2).mean()
            return L1 + L2, {"R1": L1, "R2": L2}
        u_fn = lambda x: torch.exp(model(x)[:, :1])                         # noqa: E731
        aux_fn = lambda x: model(x)[:, 1:2]                                 # noqa: E731
    elif name == "logquad":
        model = ConstrainedModel(mlp(1), lambda x, n: x * n)              # w(0) = 0
        S, W = volterra_gauss(x_col, NQ)
        nodes, WS = to_tensor(S.reshape(-1)), torch.as_tensor(W * S)

        def loss_fn():
            w = model(xc)
            ws = model(nodes).reshape(S.shape)
            memory = (WS * torch.exp(ws - w)).sum(dim=1, keepdim=True)    # = A(x)/u(x)
            r = derivative(w, xc) - 1.0 - memory
            L_r = (r ** 2).mean()
            return L_r, {"residual": L_r}
        u_fn = lambda x: torch.exp(model(x))                                # noqa: E731
        aux_fn = None
    elif name == "bifactor":
        model = ConstrainedModel(mlp(1), lambda x, n: Y0 + DY0 * x + x ** 2 * n)
        _, _, bi, bip = airy(x_col + 0.25)
        rho = to_tensor(bip / bi)

        def loss_fn():
            y = model(xc)
            dy = derivative(y, xc)
            d2y = derivative(dy, xc)
            r = d2y + 2.0 * rho * dy
            L_r = (r ** 2).mean()
            return L_r, {"residual": L_r}

        def u_fn(x):
            xn = x.detach().numpy().ravel()
            bi_x = to_tensor(airy(xn + 0.25)[2])
            return torch.exp(0.5 * x) * bi_x * model(x)
        aux_fn = None
    else:  # pragma: no cover
        raise ValueError(name)
    return model, loss_fn, u_fn, aux_fn


def run(ex: Example, name: str, x_test: np.ndarray, u_ref: np.ndarray):
    torch.manual_seed(ex.seed)
    M = ex.pick(200, 40)
    model, loss_fn, u_fn, aux_fn = setup(name, uniform(M, 0.0, L))
    xt = to_tensor(x_test)

    def eval_fn():
        with torch.no_grad():
            e = errors(u_fn(xt).numpy(), u_ref)
        return {"rel_l2": e["rel_l2"], "max_rel": e["max_rel"]}

    n_adam, n_lbfgs = ex.pick(5000, 60), ex.pick(1000, 10)
    with CpuWallTimer() as tm:
        hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                     lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50, print_every=2500)
    with torch.no_grad():
        u_pred = u_fn(xt).numpy().ravel()
        aux = aux_fn(xt).numpy().ravel() if aux_fn is not None else None
    res = {"hist": hist, "u": u_pred, "aux": aux, "errors": errors(u_pred, u_ref),
           "cpu_time_s": tm.cpu, "train_time_s": hist.train_time,
           "iterations": int(hist.iters[-1]) + 1, "final_loss": hist.final_loss,
           "parameters": count_parameters(model), "collocation": M}
    res["cpu_per_iter_ms"] = 1e3 * tm.cpu / res["iterations"]
    return res


def classical(x_test: np.ndarray, u_ref: np.ndarray, quick: bool):
    rows = []
    for N in ((1000, 10000) if not quick else (1000,)):
        (x, u), cpu, wall = time_call(lambda: trapezoidal_vide(lambda s: 1.0, lambda s, t: t,
                                                               lambda s: 0.0, 1.0, L, N), min_time=0.2)
        rows.append({"method": f"trapezoidal, N={N}", "N": N, **errors(u, volterra_exact(x)),
                     "cpu_time_s": cpu, "wall_time_s": wall})
    Y, cpu, wall = time_call(lambda: dop853_system(volterra_rhs, [1.0, 0.0], L, x_test), min_time=0.2)
    rows.append({"method": "DOP853 (u, A) system", "N": None, **errors(Y[:, 0], u_ref),
                 "cpu_time_s": cpu, "wall_time_s": wall})
    return rows


def main():
    ex = Example("ch08_integro_differential", "volterra_growth",
                 description="Volterra IDE on [0,5]: rapid growth, scaling of the unknowns",
                 default_seed=1234, extra_args=[RESUME_ARG])
    x_test = np.linspace(0.0, L, N_TEST)
    u_ref = volterra_exact(x_test)
    a_ref = volterra_exact_aux(x_test) / u_ref
    runs = {}
    for name in VARIANTS:
        ex.log(f"training {name}")
        runs[name] = cached(ex, name, lambda: run(ex, name, x_test, u_ref))
        ex.save_history(runs[name]["hist"], name)
        e = runs[name]["errors"]
        ex.log(f"{name}: relL2={e['rel_l2']:.2e} max_rel={e['max_rel']:.2e} "
               f"loss={runs[name]['final_loss']:.2e} cpu={runs[name]['cpu_time_s']:.1f}s")
    cls = classical(x_test, u_ref, ex.quick)
    for row in cls:
        ex.log(f"{row['method']}: relL2={row['rel_l2']:.2e} max_rel={row['max_rel']:.2e} "
               f"cpu={row['cpu_time_s']:.2e}s")
    table = {"x": x_test, "u_exact": u_ref, "a_exact": a_ref}
    for n in VARIANTS:
        table[f"u_{n}"] = runs[n]["u"]
        if runs[n]["aux"] is not None:
            table[f"a_{n}"] = runs[n]["aux"]
    ex.save_table(table, "predictions")
    keys = ("errors", "cpu_time_s", "train_time_s", "iterations", "cpu_per_iter_ms", "final_loss",
            "parameters", "collocation")
    best_classical = min(r["cpu_time_s"] for r in cls)
    ex.save_results({
        "problem": {"equation": "u' = u + int_0^x t u(t) dt, u(0) = 1", "interval": [0.0, L],
                    "u_at_L": float(u_ref[-1]), "test_grid": N_TEST,
                    "v_at_L": float(u_ref[-1] * np.exp(-0.5 * L)),      # u = e^{x/2} v
                    "w_at_L": float(np.log(u_ref[-1])), "a_at_L": float(a_ref[-1]),
                    "a_max": float(a_ref.max()),
                    "gronwall_amplification": float(np.exp(L + L ** 3 / 6.0)),
                    "airy_at_quarter": dict(zip(("Ai", "Ai_prime", "Bi", "Bi_prime"),
                                                (float(v) for v in airy(0.25)))),
                    "c1": VOLTERRA_C1, "c2": VOLTERRA_C2,
                    "bifactor_initial_values": {"y0": Y0, "dy0": DY0}},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "collocation": "uniform, M = %d" % ex.pick(200, 40),
                            "quadrature_nodes": NQ, "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters_max": ex.pick(1000, 10),
                            "initial_conditions": "hard"},
        "pinn": {n: {k: runs[n][k] for k in keys} for n in runs},
        "max_deviation_from_decaying_solution": {
            n: float(np.max(np.abs(runs[n]["u"] - decaying_solution(x_test)))) for n in runs},
        "classical": cls,
        "speed_ratio_pinn_over_fastest_classical": {
            n: runs[n]["cpu_time_s"] / best_classical for n in runs},
        "train_time_s": sum(runs[n]["train_time_s"] for n in runs),
        "cpu_time_training_s": sum(runs[n]["cpu_time_s"] for n in runs),
        "cpu_time_this_process_s": time.process_time(),   # small if variants were cached
    })
    if ex.args.no_figures:
        return
    plot(ex, x_test, u_ref, a_ref, runs, cls)


def plot(ex, x, u_ref, a_ref, runs, cls):
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, x, u_ref, label="exact")
    ax.plot(x, decaying_solution(x), color=P.COLORS["muted"], lw=0.9, ls=(0, (1, 1.5)),
            label="decaying solution", zorder=2)
    for k, n in enumerate(("unscaled", "exp", "log")):
        P.plot_prediction(ax, x, np.abs(runs[n]["u"]), label=LABELS[n], k=k,
                          markevery=(30 + 40 * k, 120))
    ax.set_yscale("log")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|u(x)|$")
    ax.legend(loc="upper left")
    ax.set_title("solution (log scale)")

    ax = axes[0, 1]
    for k, n in enumerate(("log", "logquad", "bifactor")):
        P.plot_error(ax, x[1:], (runs[n]["u"][1:] - u_ref[1:]) / u_ref[1:], label=LABELS[n],
                     k=k + 2, xlabel="$x$", ylabel="relative error")
    ax.set_title("pointwise relative error")

    ax = axes[1, 0]
    for k, n in enumerate(VARIANTS):
        h = runs[n]["hist"]
        ax.semilogy(h.eval_iters, h.metrics["max_rel"], color=P.SERIES[k],
                    ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.1, label=LABELS[n])
    sw = runs["log"]["hist"].switch_iter
    if sw:
        ax.axvline(sw, color=P.COLORS["muted"], lw=0.6)
    ax.set_ylim(1e-6, 8.0)                 # leaves room for the legend below the curves
    ax.set_xlabel("iteration")
    ax.set_ylabel("max. relative error")
    ax.legend(loc="lower left", ncol=1)
    ax.set_title("training history")

    ax = axes[1, 1]                       # the unknowns of the log formulation are O(1)
    w_ref = np.log(u_ref)
    P.plot_reference(ax, x, w_ref, label="exact")
    P.plot_reference(ax, x, a_ref, label=None)
    P.plot_prediction(ax, x, np.log(runs["log"]["u"]), label=LABELS["log"], k=2,
                      markevery=(30, 120))
    P.plot_prediction(ax, x, runs["log"]["aux"], label=None, k=2, markevery=(90, 120))
    ax.text(3.1, np.interp(3.1, x, w_ref) + 0.9, "$w=\\log u$", ha="right", va="bottom")
    ax.text(3.6, np.interp(3.6, x, a_ref) + 0.6, "$a=A/u$", ha="left", va="bottom")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$w(x)$, $a(x)$")
    ax.legend(loc="upper left")
    ax.set_title("logarithmic variables")
    P.label_panels(axes)
    ex.savefig(fig, "volterra_growth")


if __name__ == "__main__":
    main()
