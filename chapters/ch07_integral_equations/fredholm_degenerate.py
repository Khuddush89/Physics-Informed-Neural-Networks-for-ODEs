"""Fredholm equation with a degenerate kernel:  u(x) = 1 + (1/2) int_0^1 x t u(t) dt,  x in [0, 1].

Book: Chapter 7 (Integral equations), Section "A Fredholm equation with a degenerate kernel"
(figure ie:fig:fredholm, table ie:tab:fredholm).

Exact solution: u(x) = 1 + 3x/10, and C = int_0^1 t u(t) dt = 3/5.

Three PINNs share the network (tanh MLP with 2 hidden layers of 20 neurons, inputs scaled to
[-1, 1]) and the M = 64 Chebyshev-Gauss collocation points:

* quad   - the integral in the residual is replaced by the 2-point Gauss-Legendre rule, which
           is exact for the degree-2 integrand t u(t) of the solution;
* aux    - auxiliary scalar: a trainable number S replaces the integral in the pointwise
           residual and is tied to it by the constraint residual S - sum_j w_j t_j u(t_j);
* quad1  - as 'quad' but with the 1-point (midpoint) rule.  The PINN then converges to the
           solution u_1(x) = 1 + 2x/7 of the *discretized* equation, not to u.

The variants 'quad' and 'aux' are trained with three seeds (seed, seed+1, seed+2) to judge
run-to-run variation; the figure shows the first seed.

Classical comparison: Nystrom method with the n-point Gauss-Legendre rule, n = 1, 2, 4
(pinnode.reference.nystrom_fredholm), timed as the best of 7 repetitions.

A posteriori check (Neumann series bound, kappa = sup_x int_0^1 |x t| dt = 1/2):
    ||u_theta - u||_inf <= ||r||_inf / (1 - |lambda| kappa),
where r(x) = u_theta(x) - 1 - (1/2) x int_0^1 t u_theta(t) dt is evaluated on the test grid
with a 20-point Gauss rule.

Errors are measured on the uniform grid of 1001 points in [0, 1].

Run:  PYTHONPATH=. python chapters/ch07_integral_equations/fredholm_degenerate.py   (a few minutes, 1 core)
      PYTHONPATH=. python chapters/ch07_integral_equations/fredholm_degenerate.py --quick
      PYTHONPATH=. python chapters/ch07_integral_equations/fredholm_degenerate.py --replot  (figure only)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _iecore import REPLOT_ARG, best_time, history_arrays, history_from_arrays  # noqa: E402
from pinnode import Example, MLP, linf, rel_l2, to_tensor, train  # noqa: E402
from pinnode import plotting as P  # noqa: E402
from pinnode.quadrature import chebyshev_gauss, gauss_legendre  # noqa: E402
from pinnode.reference import nystrom_fredholm  # noqa: E402

LAM = 0.5                 # lambda in u = f + lambda int K u
KAPPA = 0.5               # sup_x int_0^1 |K(x, t)| dt for K = x t
C_EXACT = 0.6             # int_0^1 t u(t) dt
SLOPE_EXACT = 0.3
LAM_S = 1.0               # weight of the constraint residual (aux variant)
WIDTH, DEPTH, M_COL = 20, 2, 64
N_ADAM, N_LBFGS = 2000, 300
VARIANTS = [("quad", 2), ("aux", 2), ("quad1", 1)]
N_SEEDS = {"quad": 3, "aux": 3, "quad1": 1}
LABELS = {"quad": "PINN, quadrature ($Q=2$)", "aux": "PINN, auxiliary scalar ($Q=2$)",
          "quad1": "PINN, quadrature ($Q=1$)"}


def exact(x):
    return 1.0 + SLOPE_EXACT * np.asarray(x)


def ls_slope(x, u):
    """Least-squares slope of u(x) on the grid x (the 'learned slope')."""
    A = np.vstack([np.ones_like(x), x]).T
    return float(np.linalg.lstsq(A, u, rcond=None)[0][1])


def run(ex: Example, variant: str, n_quad: int, x_test: np.ndarray, seed: int):
    torch.manual_seed(seed)
    model = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, 1.0))
    x_col = to_tensor(chebyshev_gauss(ex.pick(M_COL, 16), 0.0, 1.0))
    tq, wq = gauss_legendre(n_quad, 0.0, 1.0)
    tq_t, wq_t = to_tensor(tq), to_tensor(wq)
    params = list(model.parameters())
    S = None
    if variant == "aux":
        S = torch.nn.Parameter(torch.tensor(0.5))   # deliberately away from 3/5
        params.append(S)
    xt = to_tensor(x_test)
    u_ref = exact(x_test)

    def c_quad():                                   # C_theta = sum_j w_j t_j u_theta(t_j)
        return (wq_t * tq_t * model(tq_t)).sum()

    def loss_fn():
        C = c_quad()
        u = model(x_col)
        if variant == "aux":
            r = u - 1.0 - LAM * x_col * S
            L_r, L_S = (r ** 2).mean(), (S - C) ** 2
            return L_r + LAM_S * L_S, {"residual": L_r, "constraint": L_S}
        r = u - 1.0 - LAM * x_col * C
        L_r = (r ** 2).mean()
        return L_r, {"residual": L_r}

    def eval_fn():
        with torch.no_grad():
            u = model(xt).numpy().ravel()
            out = {"rel_l2": rel_l2(u, u_ref), "C_err": abs(float(c_quad()) - C_EXACT),
                   "slope_err": abs(ls_slope(x_test, u) - SLOPE_EXACT)}
            if S is not None:
                out["S_err"] = abs(float(S) - C_EXACT)
            return out

    n_adam = ex.pick(N_ADAM, 60)
    hist = train(params, loss_fn, adam_iters=n_adam, lr=1e-2, lr_final=1e-4,
                 lbfgs_iters=ex.pick(N_LBFGS, 10), eval_fn=eval_fn, eval_every=25,
                 print_every=1000)
    with torch.no_grad():
        u = model(xt).numpy().ravel()
        C_final = float(c_quad())
        # continuous residual r(x) = u(x) - 1 - lam x int_0^1 t u(t) dt (20-point Gauss rule)
        tg, wg = gauss_legendre(20, 0.0, 1.0)
        C_acc = float(np.sum(wg * tg * model(to_tensor(tg)).numpy().ravel()))
    r = u - 1.0 - LAM * x_test * C_acc
    # distance to the solution of the *discretized* equation (Nystrom with the same rule)
    u_disc = nystrom_fredholm(lambda x, s: x * s, lambda x: np.ones_like(np.asarray(x, float)), LAM,
                              0.0, 1.0, n_quad)(x_test)
    res = {"variant": variant, "seed": seed, "n_quad": n_quad, "rel_l2": rel_l2(u, u_ref), "linf": linf(u, u_ref),
           "rel_l2_to_nystrom_same_rule": rel_l2(u, u_disc),
           "slope": ls_slope(x_test, u), "C_theta": C_final,
           "S_theta": float(S.detach()) if S is not None else None,
           "final_loss": hist.final_loss, "train_time_s": hist.train_time,
           "adam_iters": n_adam, "lbfgs_iters_done": int(hist.iters[-1] - n_adam),
           "residual_inf": float(np.max(np.abs(r))),
           "error_bound": float(np.max(np.abs(r)) / (1.0 - abs(LAM) * KAPPA))}
    return res, hist, u


def nystrom_study(ex: Example, x_test: np.ndarray):
    rows = []
    kernel = lambda x, s: x * s                                   # noqa: E731
    g = lambda x: np.ones_like(np.asarray(x, float))              # noqa: E731
    for n in (1, 2, 4):
        def solve():
            return nystrom_fredholm(kernel, g, LAM, 0.0, 1.0, n)(x_test)
        u, t = best_time(solve, ex.pick(7, 1))
        rows.append({"method": "Nystrom (Gauss-Legendre)", "n": n, "rel_l2": rel_l2(u, exact(x_test)),
                     "linf": linf(u, exact(x_test)), "slope": ls_slope(x_test, u), "time_s": t})
    return rows


def make_figure(ex: Example, x_test, preds, hists) -> None:
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, x_test, exact(x_test), label="exact")
    for k, (v, _) in enumerate(VARIANTS):
        P.plot_prediction(ax, x_test, preds[v], label=LABELS[v], k=k, markevery=(60 * k + 30, 200))
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$")
    ax.legend(loc="upper left")
    ax.set_title("solution")
    ax = axes[0, 1]
    for k, (v, _) in enumerate(VARIANTS):
        P.plot_error(ax, x_test, preds[v] - exact(x_test), label=LABELS[v].replace("PINN, ", ""), k=k,
                     xlabel="$x$")
    ax.set_title("pointwise error")
    ax.set_ylim(1e-8, 1e-1)
    ax.legend(loc="lower right")
    ax = axes[1, 0]
    P.plot_history(ax, hists["aux"], parts=["residual", "constraint"],
                   labels={"loss": "total", "residual": "pointwise residual",
                           "constraint": "constraint ($S_{\\theta}$)"})
    ax.set_ylim(1e-20, 1e2)
    ax.legend(loc="lower center")
    ax.set_title("loss components (auxiliary scalar)")
    ax = axes[1, 1]
    h = hists["aux"]
    ax.semilogy(h.eval_iters, h.metrics["S_err"], color=P.SERIES[1], lw=1.2, ls="-",
                label="$|S_{\\theta}-3/5|$ (aux.)")
    ax.semilogy(h.eval_iters, h.metrics["C_err"], color=P.SERIES[1], lw=1.2, ls=P.LINESTYLES[0],
                label="$|C_{\\theta}-3/5|$ (aux.)")
    hq = hists["quad"]
    ax.semilogy(hq.eval_iters, hq.metrics["C_err"], color=P.SERIES[0], lw=1.2, ls=P.LINESTYLES[1],
                label="$|C_{\\theta}-3/5|$ (quad.)")
    ax.axvline(h.switch_iter, color=P.COLORS["muted"], lw=0.6, zorder=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel("absolute error")
    ax.set_title("auxiliary scalar and quadrature value")
    ax.legend(loc="upper right")
    P.label_panels(axes)
    ex.savefig(fig, "fredholm_summary")


def main():
    ex = Example("ch07_integral_equations", "fredholm_degenerate",
                 description="Fredholm IE with degenerate kernel x t: quadrature vs auxiliary scalar",
                 default_seed=1234, extra_args=[REPLOT_ARG])
    x_test = np.linspace(0.0, 1.0, 1001)
    if ex.args.replot:
        d = np.load(ex.path("figdata.npz"))
        make_figure(ex, d["x"], {v: d[f"u_{v}"] for v, _ in VARIANTS},
                    {v: history_from_arrays(d, v) for v in ("quad", "aux")})
        return
    results, hists, preds, seed_runs = {}, {}, {}, {}
    for variant, nq in VARIANTS:
        seed_runs[variant] = []
        for i in range(ex.pick(N_SEEDS[variant], 1)):
            seed = ex.seed + i
            ex.log(f"training variant '{variant}' (Q = {nq}), seed {seed}")
            res, hist, u = run(ex, variant, nq, x_test, seed)
            seed_runs[variant].append(res)
            ex.log(f"{variant}/{seed}: relL2={res['rel_l2']:.3e} linf={res['linf']:.3e} slope={res['slope']:.8f} "
                   f"C={res['C_theta']:.8f} loss={res['final_loss']:.2e} time={res['train_time_s']:.1f}s "
                   f"| r_inf={res['residual_inf']:.2e} bound={res['error_bound']:.2e}")
            if i == 0:
                results[variant], hists[variant], preds[variant] = res, hist, u
                ex.save_history(hist, variant)
    spread = {v: {"seeds": [r["seed"] for r in runs], "rel_l2": [r["rel_l2"] for r in runs],
                  "linf": [r["linf"] for r in runs], "train_time_s": [r["train_time_s"] for r in runs]}
              for v, runs in seed_runs.items()}
    nys = nystrom_study(ex, x_test)
    for row in nys:
        ex.log(f"Nystrom n={row['n']}: relL2={row['rel_l2']:.2e} slope={row['slope']:.12f} time={row['time_s']:.2e}s")
    ex.save_table({"x": x_test, "u_exact": exact(x_test), **{f"u_{v}": preds[v] for v in preds}},
                  "predictions")
    ex.save_table(nys, "nystrom")
    ex.save_results({
        "problem": {"equation": "u(x) = 1 + lam int_0^1 x t u(t) dt", "lam": LAM, "domain": [0, 1],
                    "exact": "1 + 0.3 x", "C_exact": C_EXACT, "discrete_solution_Q1": "1 + 2x/7"},
        "hyperparameters": {"network": f"MLP tanh {DEPTH}x{WIDTH}", "collocation": "Chebyshev-Gauss",
                            "M": ex.pick(M_COL, 16), "adam_iters": ex.pick(N_ADAM, 60),
                            "lr": "1e-2 -> 1e-4 (cosine)", "lbfgs_iters_max": ex.pick(N_LBFGS, 10),
                            "lambda_S": LAM_S, "S_init": 0.5, "test_grid": 1001},
        "pinn": results, "seed_study": spread, "nystrom": nys,
        "train_time_s": float(sum(r["train_time_s"] for runs in seed_runs.values() for r in runs))})

    np.savez(ex.path("figdata.npz"), x=x_test, **{f"u_{v}": preds[v] for v in preds},
             **history_arrays(hists["quad"], "quad"), **history_arrays(hists["aux"], "aux"))
    if not ex.args.no_figures:
        make_figure(ex, x_test, preds, hists)


if __name__ == "__main__":
    main()
