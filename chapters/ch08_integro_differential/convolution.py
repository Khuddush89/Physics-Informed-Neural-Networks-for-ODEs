"""Convolution IDE  u'(x) = u(x) + int_0^x e^{-(x-t)} u(t) dt,  u(0) = 1,  on [0, 1]:
direct quadrature versus the internal (auxiliary) variable z(x) = int_0^x e^{-(x-t)} u(t) dt.

Book: Chapter 8 (Integro-differential equations), Section "Convolution kernels and internal
variables", worked example: Figure ide:fig:convolution, Table ide:tab:convolution.

Since z' = u - z, the IDE is equivalent to the linear system (u, z)' = (u + z, u - z),
u(0) = 1, z(0) = 0, whose matrix has eigenvalues +-sqrt(2).  Exact solution:
    u(x) = cosh(sqrt2 x) + (sqrt2/2) sinh(sqrt2 x),   z(x) = (sqrt2/2) sinh(sqrt2 x).

Formulations (hard initial conditions):
  * direct quadrature (any kernel): u_theta = 1 + x N(x),
        R(x_i) = u' - u - sum_j W_ij e^{-(x_i - s_ij)} u_theta(s_ij)
    with the 20-point Gauss-Legendre rule on [0, x_i];
  * internal variable (exponential kernels): (u_theta, z_theta) = (1 + x N_1, x N_2),
        R_u = u' - u - z,    R_z = z' + z - u.
Classical references: DOP853 (rtol = atol = 1e-12) for the (u, z) system and the implicit
trapezoidal direct-quadrature method with N steps applied to the IDE itself (O(N^2) work,
no use of the exponential structure).

Errors are measured on a uniform grid of 1001 points in [0, 1].

Run:  PYTHONPATH=. python chapters/ch08_integro_differential/convolution.py          (~1 min, 1 core)
      PYTHONPATH=. python chapters/ch08_integro_differential/convolution.py --quick  (smoke test)
"""
from __future__ import annotations

import time
import numpy as np
import torch

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, to_tensor,
                     train, uniform, volterra_gauss)
from pinnode import plotting as P

from _ide_common import (RESUME_ARG, CpuWallTimer, cached, convolution_exact, convolution_exact_aux,
                         dop853_system, errors, time_call, trapezoidal_vide)

L = 1.0
WIDTH, DEPTH = 32, 3
NQ = 20
N_TEST = 1001
LABELS = {"direct": "direct quadrature", "internal": "internal variable"}


def kernel(x, t):
    return np.exp(-(x - t))


def build_direct():
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, L))
    return ConstrainedModel(net, lambda x, n: 1.0 + x * n)


def build_internal():
    net = MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, L))
    return ConstrainedModel(net, lambda x, n: torch.cat([1.0 + x * n[:, :1], x * n[:, 1:2]], dim=1))


def run(ex: Example, name: str, x_test: np.ndarray, u_ref: np.ndarray):
    torch.manual_seed(ex.seed)
    M = ex.pick(100, 30)
    x_col = uniform(M, 0.0, L)
    xc = to_tensor(x_col, requires_grad=True)
    if name == "direct":
        model = build_direct()
        S, W = volterra_gauss(x_col, NQ)
        nodes = to_tensor(S.reshape(-1))
        weights = torch.as_tensor(W * kernel(x_col.reshape(-1, 1), S))

        def loss_fn():
            u = model(xc)
            memory = (weights * model(nodes).reshape(S.shape)).sum(dim=1, keepdim=True)
            r = derivative(u, xc) - u - memory
            L_r = (r ** 2).mean()
            return L_r, {"residual": L_r}
        u_fn = model
    else:
        model = build_internal()

        def loss_fn():
            out = model(xc)
            d = derivative(out, xc)
            u, z = out[:, :1], out[:, 1:2]
            r_u = d[:, :1] - u - z
            r_z = d[:, 1:2] + z - u
            Lu, Lz = (r_u ** 2).mean(), (r_z ** 2).mean()
            return Lu + Lz, {"R_u": Lu, "R_z": Lz}
        u_fn = lambda x: model(x)[:, :1]                                    # noqa: E731
    xt = to_tensor(x_test)

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": errors(u_fn(xt).numpy(), u_ref)["rel_l2"]}

    n_adam, n_lbfgs = ex.pick(5000, 60), ex.pick(1000, 10)
    with CpuWallTimer() as tm:
        hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                     lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50, print_every=2500)
    # cost of one loss + gradient evaluation (the unit of work of every optimizer step)
    def one_step():
        model.zero_grad(set_to_none=True)
        loss, _ = loss_fn()
        loss.backward()
    _, step_cpu, _ = time_call(one_step, min_time=ex.pick(1.0, 0.05))
    with torch.no_grad():
        out = model(xt).numpy()
    res = {"hist": hist, "u": out[:, 0], "z": out[:, 1] if out.shape[1] > 1 else None,
           "errors": errors(out[:, 0], u_ref), "cpu_time_s": tm.cpu,
           "train_time_s": hist.train_time, "iterations": int(hist.iters[-1]) + 1,
           "final_loss": hist.final_loss, "parameters": count_parameters(model),
           "cpu_per_loss_gradient_ms": 1e3 * step_cpu}
    res["cpu_per_iter_ms"] = 1e3 * tm.cpu / res["iterations"]
    return res


def classical(x_test: np.ndarray, u_ref: np.ndarray, quick: bool):
    rows = []
    for N in ((100, 1000) if not quick else (100,)):
        (x, u), cpu, wall = time_call(lambda: trapezoidal_vide(lambda s: 1.0, kernel, lambda s: 0.0,
                                                               1.0, L, N), min_time=0.2)
        rows.append({"method": f"trapezoidal, N={N}", "N": N, **errors(u, convolution_exact(x)),
                     "cpu_time_s": cpu, "wall_time_s": wall})
    rhs = lambda s, y: [y[0] + y[1], y[0] - y[1]]                          # noqa: E731
    Y, cpu, wall = time_call(lambda: dop853_system(rhs, [1.0, 0.0], L, x_test), min_time=0.2)
    rows.append({"method": "DOP853 (u, z) system", "N": None, **errors(Y[:, 0], u_ref),
                 "cpu_time_s": cpu, "wall_time_s": wall})
    return rows


def main():
    ex = Example("ch08_integro_differential", "convolution",
                 description="Convolution IDE: direct quadrature vs internal variable (hard ICs)",
                 default_seed=1234, extra_args=[RESUME_ARG])
    x_test = np.linspace(0.0, L, N_TEST)
    u_ref, z_ref = convolution_exact(x_test), convolution_exact_aux(x_test)
    runs = {}
    for name in ("direct", "internal"):
        ex.log(f"training {name}")
        runs[name] = cached(ex, name, lambda: run(ex, name, x_test, u_ref))
        ex.save_history(runs[name]["hist"], name)
        e = runs[name]["errors"]
        ex.log(f"{name}: relL2={e['rel_l2']:.2e} linf={e['linf']:.2e} "
               f"cpu={runs[name]['cpu_time_s']:.1f}s step={runs[name]['cpu_per_loss_gradient_ms']:.2f}ms")
    cls = classical(x_test, u_ref, ex.quick)
    for row in cls:
        ex.log(f"{row['method']}: relL2={row['rel_l2']:.2e} cpu={row['cpu_time_s']:.2e}s")
    ex.save_table({"x": x_test, "u_exact": u_ref, "z_exact": z_ref, "u_direct": runs["direct"]["u"],
                   "u_internal": runs["internal"]["u"], "z_internal": runs["internal"]["z"]},
                  "predictions")
    keys = ("errors", "cpu_time_s", "train_time_s", "iterations", "cpu_per_iter_ms",
            "cpu_per_loss_gradient_ms", "final_loss", "parameters")
    pinn = {n: {k: runs[n][k] for k in keys} for n in runs}
    pinn["internal"]["errors_z"] = errors(runs["internal"]["z"], z_ref)
    best_classical = min(r["cpu_time_s"] for r in cls)
    ex.save_results({
        "problem": {"equation": "u' = u + int_0^x exp(-(x-t)) u(t) dt, u(0) = 1",
                    "interval": [0.0, L], "test_grid": N_TEST,
                    "exact": "u = cosh(sqrt2 x) + sqrt2/2 sinh(sqrt2 x)"},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "collocation": "uniform, M = %d" % ex.pick(100, 30),
                            "quadrature_nodes": NQ, "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters_max": ex.pick(1000, 10),
                            "initial_conditions": "hard"},
        "pinn": pinn,
        "classical": cls,
        "cost_ratio_direct_over_internal_per_step":
            runs["direct"]["cpu_per_loss_gradient_ms"] / runs["internal"]["cpu_per_loss_gradient_ms"],
        "speed_ratio_pinn_over_fastest_classical": {
            n: runs[n]["cpu_time_s"] / best_classical for n in runs},
        "train_time_s": sum(runs[n]["train_time_s"] for n in runs),
        "cpu_time_training_s": sum(runs[n]["cpu_time_s"] for n in runs),
        "cpu_time_this_process_s": time.process_time(),   # small if variants were cached
    })
    if ex.args.no_figures:
        return
    plot(ex, x_test, u_ref, z_ref, runs)


def plot(ex, x, u_ref, z_ref, runs):
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, x, u_ref, label="exact")
    P.plot_reference(ax, x, z_ref, label=None)
    P.plot_prediction(ax, x, runs["direct"]["u"], label="$u_\\theta$, direct", k=0,
                      markevery=(40, 120))
    P.plot_prediction(ax, x, runs["internal"]["u"], label="$u_\\theta$, internal", k=1,
                      markevery=(100, 120))
    P.plot_prediction(ax, x, runs["internal"]["z"], label="$z_\\theta$, internal", k=2,
                      markevery=(70, 120))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$, $z(x)$")
    ax.legend(loc="upper left")
    ax.set_title("solution and internal variable")

    ax = axes[0, 1]
    s = slice(1, None)                    # x = 0 is exact by construction (hard conditions)
    P.plot_error(ax, x[s], runs["direct"]["u"][s] - u_ref[s], label="$u$, direct", k=0, xlabel="$x$")
    P.plot_error(ax, x[s], runs["internal"]["u"][s] - u_ref[s], label="$u$, internal", k=1,
                 xlabel="$x$")
    P.plot_error(ax, x[s], runs["internal"]["z"][s] - z_ref[s], label="$z$, internal", k=2,
                 xlabel="$x$")
    ax.set_title("pointwise error")

    ax = axes[1, 0]
    hd, hi = runs["direct"]["hist"], runs["internal"]["hist"]
    w = max(5, len(hd.iters) // 60)
    series = [(hd.iters, hd.loss, "direct", P.COLORS["ref"], "-"),
              (hi.iters, hi.parts["R_u"], "internal: $R_u$ term", P.SERIES[1], P.LINESTYLES[0]),
              (hi.iters, hi.parts["R_z"], "internal: $R_z$ term", P.SERIES[2], P.LINESTYLES[1])]
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
    for k, n in enumerate(("direct", "internal")):
        P.plot_metric(ax, runs[n]["hist"], "rel_l2", label=LABELS[n], k=k)
    ax.set_title("relative $L^2$ error of $u_\\theta$")
    P.label_panels(axes)
    ex.savefig(fig, "convolution")


if __name__ == "__main__":
    main()
