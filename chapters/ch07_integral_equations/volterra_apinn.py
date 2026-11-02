"""Volterra equation  u(x) = 1 + lam int_0^x t u(t) dt,  lam = 2,  x in [0, 1].

Book: Chapter 7 (Integral equations), Section "Volterra equations", worked example
(figure ie:fig:volterra, table ie:tab:volterra).

Exact solution: u(x) = exp(lam x^2 / 2) = e^{x^2}; running integral
A(x) = int_0^x t u(t) dt = (e^{x^2} - 1) / lam.

Two PINNs with the same tanh MLP (3 hidden layers of 32 neurons) and M = 101 uniform
collocation points:

* apinn - auxiliary-output PINN (A-PINN): two outputs (N_1, N_2) with the hard constraints
          u = 1 + x N_1 and A = x N_2 (so u(0) = 1, A(0) = 0 exactly) and the residuals
          R1 = u - 1 - lam A,   R2 = A' - x u    (A' by automatic differentiation);
          no quadrature is needed.
* quad  - direct quadrature: u = 1 + x N and, for every collocation point x_i, the 8-point
          Gauss-Legendre rule mapped to [0, x_i] (pinnode.volterra_gauss).

Classical comparison: trapezoidal rule (pinnode.reference.volterra_trapezoid) with
n = 11, 101, 1001, 10001 nodes (cubic-spline interpolation to the test grid), timed as the
best of 7 repetitions.

A posteriori check (Gronwall): |u_theta(x) - u(x)| <= ||r||_inf exp(lam x^2 / 2), where
r(x) = u_theta(x) - 1 - lam int_0^x t u_theta(t) dt is evaluated on the test grid with a
32-point Gauss rule on every [0, x]; for the A-PINN r = R1 + lam int_0^x R2(t) dt.

Errors are measured on the uniform grid of 1001 points in [0, 1].

Run:  PYTHONPATH=. python chapters/ch07_integral_equations/volterra_apinn.py   (a few minutes, 1 core)
      PYTHONPATH=. python chapters/ch07_integral_equations/volterra_apinn.py --quick
      PYTHONPATH=. python chapters/ch07_integral_equations/volterra_apinn.py --replot  (figure only)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _iecore import REPLOT_ARG, best_time, history_arrays, history_from_arrays  # noqa: E402
from pinnode import (ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor,  # noqa: E402
                     train, volterra_gauss)
from pinnode import plotting as P  # noqa: E402
from pinnode.reference import volterra_trapezoid  # noqa: E402

LAM = 2.0
WIDTH, DEPTH, M_COL, N_GAUSS = 32, 3, 101, 8
LABELS = {"apinn": "A-PINN", "quad": "PINN, direct quadrature"}


def u_exact(x):
    return np.exp(0.5 * LAM * np.asarray(x) ** 2)


def A_exact(x):
    return (u_exact(x) - 1.0) / LAM


def build(variant: str):
    if variant == "apinn":
        net = MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, 1.0))
        return ConstrainedModel(net, lambda x, n: torch.cat([1.0 + x * n[:, :1], x * n[:, 1:2]], dim=1))
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, 1.0))
    return ConstrainedModel(net, lambda x, n: 1.0 + x * n)


def ie_residual(model, x_test: np.ndarray, n: int = 32) -> np.ndarray:
    """r(x) = u(x) - 1 - lam int_0^x t u(t) dt with an n-point Gauss rule on every [0, x]."""
    S, W = volterra_gauss(x_test, n)
    with torch.no_grad():
        uS = model(to_tensor(S.reshape(-1)))[:, :1].numpy().reshape(S.shape)
        ux = model(to_tensor(x_test))[:, :1].numpy().ravel()
    return ux - 1.0 - LAM * np.sum(W * S * uS, axis=1)


def run(ex: Example, variant: str, x_test: np.ndarray):
    torch.manual_seed(ex.seed)
    model = build(variant)
    m = ex.pick(M_COL, 21)
    x_np = np.linspace(0.0, 1.0, m)
    x_col = to_tensor(x_np, requires_grad=True)
    if variant == "quad":
        S, W = volterra_gauss(x_np, N_GAUSS)
        S_t, W_t = to_tensor(S.reshape(-1)), torch.as_tensor(W)
        S_np = S
    xt = to_tensor(x_test)
    u_ref = u_exact(x_test)

    def loss_fn():
        if variant == "apinn":
            out = model(x_col)
            u, A = out[:, :1], out[:, 1:2]
            R1 = u - 1.0 - LAM * A
            R2 = derivative(A, x_col) - x_col * u
            L1, L2 = (R1 ** 2).mean(), (R2 ** 2).mean()
            return L1 + L2, {"R1": L1, "R2": L2}
        u = model(x_col)
        uS = model(S_t).reshape(S_np.shape)
        integral = (W_t * torch.as_tensor(S_np) * uS).sum(dim=1, keepdim=True)
        R = u - 1.0 - LAM * integral
        L = (R ** 2).mean()
        return L, {"residual": L}

    def eval_fn():
        with torch.no_grad():
            out = model(xt)
            met = {"rel_l2": rel_l2(out[:, 0], u_ref)}
            if variant == "apinn":
                met["rel_l2_A"] = rel_l2(out[:, 1], A_exact(x_test))
            return met

    n_adam = ex.pick(5000, 60)
    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=5e-3, lr_final=5e-5,
                 lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=25, print_every=1000)
    xg = to_tensor(x_test, requires_grad=True)
    out = model(xg)
    res = {"variant": variant, "final_loss": hist.final_loss, "train_time_s": hist.train_time,
           "adam_iters": n_adam, "lbfgs_iters_done": int(hist.iters[-1] - n_adam)}
    # reduction of each loss component by L-BFGS (last Adam record / final record)
    i_adam = max(i for i, ph in enumerate(hist.phase) if ph == "adam")
    res["lbfgs_reduction"] = {k: float(v[i_adam] / v[-1]) for k, v in hist.parts.items() if v[-1] > 0}
    res["lbfgs_reduction"]["total"] = float(hist.loss[i_adam] / hist.loss[-1])
    u = out[:, 0].detach().numpy()
    res.update({"rel_l2": rel_l2(u, u_ref), "linf": linf(u, u_ref)})
    preds = {"u": u}
    if variant == "apinn":
        A = out[:, 1:2]
        R1 = (out[:, :1] - 1.0 - LAM * A).detach().numpy().ravel()
        R2 = (derivative(A, xg) - xg * out[:, :1]).detach().numpy().ravel()
        preds["A"] = A.detach().numpy().ravel()
        res.update({"rel_l2_A": rel_l2(preds["A"], A_exact(x_test)), "linf_A": linf(preds["A"], A_exact(x_test)),
                    "R1_inf": float(np.max(np.abs(R1))), "R2_inf": float(np.max(np.abs(R2))),
                    "bound_from_R1_R2": float(np.exp(0.5 * LAM) * (np.max(np.abs(R1)) + LAM * np.max(np.abs(R2))))})
    r = ie_residual(model, x_test)
    res.update({"residual_inf": float(np.max(np.abs(r))),
                "error_bound": float(np.exp(0.5 * LAM) * np.max(np.abs(r)))})
    return res, hist, preds


def trapezoid_study(ex: Example, x_test: np.ndarray):
    rows = []
    kernel = lambda x, s: s                                          # noqa: E731
    g = lambda x: np.ones_like(np.asarray(x, float))                 # noqa: E731
    for n in ex.pick((11, 101, 1001, 10001), (11, 101)):
        (x, u), t = best_time(lambda: volterra_trapezoid(kernel, g, LAM, 0.0, 1.0, n), ex.pick(7, 1))
        u_test = CubicSpline(x, u)(x_test)
        rows.append({"method": "trapezoidal rule", "n": n, "nodal_max_err": float(np.max(np.abs(u - u_exact(x)))),
                     "rel_l2": rel_l2(u_test, u_exact(x_test)), "linf": linf(u_test, u_exact(x_test)),
                     "time_s": t})
    return rows


def make_figure(ex: Example, x_test, preds, hists) -> None:
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, x_test, u_exact(x_test), label="exact $u$")
    P.plot_prediction(ax, x_test, preds["apinn"]["u"], label="A-PINN $u_{\\theta}$", k=0, markevery=(50, 200))
    P.plot_prediction(ax, x_test, preds["quad"]["u"], label="direct quadrature", k=1, markevery=(150, 200))
    P.plot_reference(ax, x_test, A_exact(x_test), label=None, lw=0.8, ls=":")
    P.plot_prediction(ax, x_test, preds["apinn"]["A"], label="A-PINN $A_{\\theta}$", k=2, markevery=(100, 200))
    ax.text(0.62, 0.6, "$A(x)$", fontsize=8, color=P.COLORS["muted"])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(x)$, $A(x)$")
    ax.set_title("solution and auxiliary output")
    ax.legend(loc="upper left")
    ax = axes[0, 1]
    P.plot_error(ax, x_test, preds["apinn"]["u"] - u_exact(x_test), label="A-PINN, $u$", k=0, xlabel="$x$")
    P.plot_error(ax, x_test, preds["quad"]["u"] - u_exact(x_test), label="direct quadrature, $u$", k=1,
                 xlabel="$x$")
    P.plot_error(ax, x_test, preds["apinn"]["A"] - A_exact(x_test), label="A-PINN, $A$", k=2, xlabel="$x$")
    ax.set_title("pointwise error")
    ax.set_ylim(1e-9, 1e-2)
    ax.legend(loc="lower right")
    ax = axes[1, 0]
    P.plot_history(ax, hists["apinn"], parts=["R1", "R2"],
                   labels={"loss": "total", "R1": "$\\|R_1\\|^2$ (equation)", "R2": "$\\|R_2\\|^2$ (auxiliary)"})
    ax.set_ylim(1e-11, 3.0)
    ax.legend(loc="lower left")
    ax.set_title("loss components (A-PINN)")
    ax = axes[1, 1]
    P.plot_metric(ax, hists["apinn"], "rel_l2", label="A-PINN", k=0)
    P.plot_metric(ax, hists["quad"], "rel_l2", label="direct quadrature", k=1, show_switch=False)
    ax.set_title("relative $L^2$ error of $u_{\\theta}$")
    P.label_panels(axes)
    ex.savefig(fig, "volterra_summary")


def main():
    ex = Example("ch07_integral_equations", "volterra_apinn",
                 description="Volterra IE u = 1 + 2 int_0^x t u dt: A-PINN vs direct quadrature",
                 default_seed=1234, extra_args=[REPLOT_ARG])
    x_test = np.linspace(0.0, 1.0, 1001)
    if ex.args.replot:
        d = np.load(ex.path("figdata.npz"))
        preds = {"apinn": {"u": d["u_apinn"], "A": d["A_apinn"]}, "quad": {"u": d["u_quad"]}}
        make_figure(ex, d["x"], preds, {v: history_from_arrays(d, v) for v in ("apinn", "quad")})
        return
    results, hists, preds = {}, {}, {}
    for variant in ("apinn", "quad"):
        ex.log(f"training variant '{variant}'")
        res, hist, pr = run(ex, variant, x_test)
        results[variant], hists[variant], preds[variant] = res, hist, pr
        ex.save_history(hist, variant)
        ex.log(f"{variant}: relL2={res['rel_l2']:.3e} linf={res['linf']:.3e} loss={res['final_loss']:.2e} "
               f"time={res['train_time_s']:.1f}s | r_inf={res['residual_inf']:.2e} bound={res['error_bound']:.2e}")
    trap = trapezoid_study(ex, x_test)
    for row in trap:
        ex.log(f"trapezoid n={row['n']}: nodal={row['nodal_max_err']:.2e} relL2={row['rel_l2']:.2e} "
               f"time={row['time_s']:.2e}s")
    ex.save_table({"x": x_test, "u_exact": u_exact(x_test), "u_apinn": preds["apinn"]["u"],
                   "A_apinn": preds["apinn"]["A"], "u_quad": preds["quad"]["u"]}, "predictions")
    ex.save_table(trap, "trapezoid")
    ex.save_results({
        "problem": {"equation": "u(x) = 1 + lam int_0^x t u(t) dt", "lam": LAM, "domain": [0, 1],
                    "exact": "exp(lam x^2/2)"},
        "hyperparameters": {"network": f"MLP tanh {DEPTH}x{WIDTH}", "collocation": "uniform",
                            "M": ex.pick(M_COL, 21), "gauss_nodes_per_point": N_GAUSS,
                            "adam_iters": ex.pick(5000, 60), "lr": "5e-3 -> 5e-5 (cosine)",
                            "lbfgs_iters_max": ex.pick(1000, 10), "loss_weights": "R1: 1, R2: 1",
                            "test_grid": 1001},
        "pinn": results, "trapezoid": trap,
        "train_time_s": float(sum(r["train_time_s"] for r in results.values()))})

    np.savez(ex.path("figdata.npz"), x=x_test, u_apinn=preds["apinn"]["u"], A_apinn=preds["apinn"]["A"],
             u_quad=preds["quad"]["u"], **history_arrays(hists["apinn"], "apinn"),
             **history_arrays(hists["quad"], "quad"))
    if not ex.args.no_figures:
        make_figure(ex, x_test, preds, hists)


if __name__ == "__main__":
    main()
