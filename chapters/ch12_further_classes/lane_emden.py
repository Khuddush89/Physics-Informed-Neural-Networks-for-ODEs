"""Singular Lane-Emden problem  y'' + (2/x) y' + y^m = 0,  y(0) = 1, y'(0) = 0,  x in [0, 7].

Book: Chapter 12 (Further equation classes), Section "Singular problems of Lane-Emden type".

The coefficient 2/x is singular at x = 0.  The PINN
* multiplies the equation by x:   R(x) = x y'' + 2 y' + x y^m   (bounded at x = 0), and
* uses the trial solution  y(x) = 1 + (x/L)^2 N(x),  which satisfies y(0) = 1 and y'(0) = 0 exactly
  (and makes (2/x) y' = (4 N + 2 x N')/L^2 regular).  The factor 1/L^2 keeps the initial output of
  order one; with 1 + x^2 N(x) the initial loss is many orders of magnitude larger (the script
  records both initial losses).

Exact solutions: m = 0: 1 - x^2/6;  m = 1: sin(x)/x;  m = 5: (1 + x^2/3)^(-1/2).
For m = 3 (no closed form) the reference is the series
    y = 1 - x^2/6 + m x^4/120 - m(8m - 5) x^6/15120 + ...
on [0, 0.01] followed by DOP853 (rtol = atol = 1e-13).  For m = 0, 1, 3 the first zero xi_1
and omega = -xi_1^2 y'(xi_1) are reported (Chandrasekhar's table: xi_1 = 6.89685,
omega = 2.01824 for m = 3).

Run:  python chapters/ch12_further_classes/lane_emden.py           (about 2-3 min on 1 CPU core)
      python chapters/ch12_further_classes/lane_emden.py --quick   (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from pinnode import ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor, train, uniform
from pinnode import plotting as P

L = 7.0
CASES = [0, 1, 3, 5]
X0_SERIES = 1e-2


def series(x, m):
    return 1 - x ** 2 / 6 + m * x ** 4 / 120 - m * (8 * m - 5) * x ** 6 / 15120


def series_d(x, m):
    return -x / 3 + m * x ** 3 / 30 - m * (8 * m - 5) * x ** 5 / 2520


def numerical_reference(m, x):
    """Series on [0, 0.01], then DOP853 (rtol = atol = 1e-13); integer m. Returns (y, y', xi1, omega)."""
    x = np.asarray(x, float)

    def f(s, u):
        return [u[1], -2.0 / s * u[1] - u[0] ** m]

    def zero(s, u):
        return u[0]

    zero.terminal, zero.direction = False, -1
    xs = x[x > X0_SERIES]
    sol = solve_ivp(f, (X0_SERIES, L), [series(X0_SERIES, m), series_d(X0_SERIES, m)], method="DOP853",
                    rtol=1e-13, atol=1e-13, t_eval=xs, events=zero, dense_output=True)
    y, dy = series(x, m), series_d(x, m)
    y[x > X0_SERIES], dy[x > X0_SERIES] = sol.y[0], sol.y[1]
    if len(sol.t_events[0]):
        xi1 = float(sol.t_events[0][0])
        return y, dy, xi1, float(-xi1 ** 2 * sol.sol(xi1)[1])
    return y, dy, None, None


def reference(m, x):
    """Exact solution for m in {0, 1, 5}; series + DOP853 otherwise. Returns (y, y', info)."""
    x = np.asarray(x, float)
    if m == 0:
        return 1 - x ** 2 / 6, -x / 3, {"xi1": np.sqrt(6.0), "omega": 2 * np.sqrt(6.0), "kind": "exact"}
    if m == 1:
        xs = np.where(x > 0, x, 1.0)
        y = np.where(x > 0, np.sin(x) / xs, 1.0)
        dy = np.where(x > 0, (x * np.cos(x) - np.sin(x)) / xs ** 2, 0.0)
        return y, dy, {"xi1": np.pi, "omega": np.pi, "kind": "exact"}
    if m == 5:
        return (1 + x ** 2 / 3) ** -0.5, -(x / 3) * (1 + x ** 2 / 3) ** -1.5, {"xi1": None, "omega": None,
                                                                             "kind": "exact"}
    y, dy, xi1, omega = numerical_reference(m, x)
    return y, dy, {"xi1": xi1, "omega": omega, "kind": "series + DOP853 (rtol=atol=1e-13)"}


def residual_loss(model, x_col, m):
    y = model(x_col)
    dy = derivative(y, x_col)
    d2y = derivative(dy, x_col)
    r = x_col * d2y + 2.0 * dy + x_col * y ** m          # equation multiplied by x
    return (r ** 2).mean()


def run_case(ex: Example, m: int, x_test):
    torch.manual_seed(ex.seed)
    net = MLP(1, 1, width=32, depth=3, activation="tanh", domain=(0.0, L))
    model = ConstrainedModel(net, lambda x, n: 1.0 + (x / L) ** 2 * n)   # y(0) = 1, y'(0) = 0
    x_col = to_tensor(uniform(ex.pick(200, 30), 0.0, L), requires_grad=True)
    y_ref, dy_ref, info = reference(m, x_test)
    x_test_t = to_tensor(x_test)
    # diagnostic: initial loss of the scaled and of the unscaled trial function (same network)
    unscaled = ConstrainedModel(net, lambda x, n: 1.0 + x ** 2 * n)
    init_loss = {"scaled": float(residual_loss(model, x_col, m)),
                 "unscaled": float(residual_loss(unscaled, x_col, m))}

    def loss_fn():
        Lr = residual_loss(model, x_col, m)
        return Lr, {"residual": Lr}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(x_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(6000, 60), lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=100, print_every=3000)
    xt = to_tensor(x_test, requires_grad=True)
    y = model(xt)
    dy = derivative(y, xt)
    y_pred, dy_pred = y.detach().numpy().ravel(), dy.detach().numpy().ravel()
    out = {"m": m, "hist": hist, "y_ref": y_ref, "y_pred": y_pred, "rel_l2": rel_l2(y_pred, y_ref),
           "initial_loss": init_loss,
           "linf": linf(y_pred, y_ref), "deriv_linf": linf(dy_pred, dy_ref), "ref_kind": info["kind"],
           "xi1_ref": info["xi1"], "omega_ref": info["omega"], "xi1_pinn": None, "omega_pinn": None}
    if info["xi1"] is not None and (y_pred < 0).any():   # first zero and omega = -xi^2 y'(xi)
        def g(s):
            with torch.no_grad():
                return float(model(to_tensor([s])).item())
        i = int(np.argmax(y_pred < 0))
        xi = brentq(g, x_test[i - 1], x_test[i], xtol=1e-14, rtol=1e-14)
        s_t = to_tensor([xi], requires_grad=True)
        d = derivative(model(s_t), s_t).item()
        out["xi1_pinn"], out["omega_pinn"] = xi, -xi ** 2 * d
    return out


def main():
    ex = Example("ch12_further_classes", "lane_emden",
                 description="Lane-Emden y'' + (2/x) y' + y^m = 0: trial solution 1 + (x/L)^2 N(x)",
                 default_seed=1234)
    x_test = np.linspace(0.0, L, 1001)
    runs, rows = {}, []
    for m in CASES:
        ex.log(f"m = {m}")
        r = run_case(ex, m, x_test)
        runs[m] = r
        ex.save_history(r["hist"], f"m{m}")
        rows.append({"m": m, "reference": r["ref_kind"], "rel_l2": r["rel_l2"], "linf": r["linf"],
                     "initial_loss_scaled": r["initial_loss"]["scaled"],
                     "initial_loss_unscaled": r["initial_loss"]["unscaled"],
                     "deriv_linf": r["deriv_linf"], "xi1_ref": r["xi1_ref"], "xi1_pinn": r["xi1_pinn"],
                     "omega_ref": r["omega_ref"], "omega_pinn": r["omega_pinn"],
                     "final_loss": r["hist"].final_loss, "train_time_s": r["hist"].train_time})
        ex.log(f"m={m} relL2={r['rel_l2']:.2e} linf={r['linf']:.2e} xi1={r['xi1_pinn']} "
               f"(ref {r['xi1_ref']}) omega={r['omega_pinn']} (ref {r['omega_ref']}) "
               f"time={r['hist'].train_time:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_table({"x": x_test, **{f"ref_m{m}": runs[m]["y_ref"] for m in CASES},
                   **{f"pinn_m{m}": runs[m]["y_pred"] for m in CASES}}, "predictions")
    # validate the series + DOP853 reference (used for m = 3) against the closed forms
    check = {}
    for m in (0, 1, 5):
        y_num, _, xi_num, _ = numerical_reference(m, x_test)
        y_ex, _, info = reference(m, x_test)
        check[f"m{m}"] = {"max_abs_diff": float(np.max(np.abs(y_num - y_ex))),
                          "xi1_diff": (None if info["xi1"] is None else abs(xi_num - info["xi1"]))}
    ex.log(f"reference check: {check}")
    ex.save_results({
        "reference_check_series_dop853_vs_exact": check,
        "problem": {"equation": "y'' + (2/x) y' + y^m = 0, y(0)=1, y'(0)=0", "domain": [0.0, L],
                    "cases": CASES, "residual": "x y'' + 2 y' + x y^m",
                    "trial_solution": "y = 1 + (x/L)^2 N(x), L = 7"},
        "hyperparameters": {"width": 32, "depth": 3, "activation": "tanh",
                            "collocation": ex.pick(200, 30), "adam_iters": ex.pick(6000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10)},
        "chandrasekhar_m3": {"xi1": 6.89685, "omega": 2.01824},
        "summary": rows,
        "rel_l2": {f"m{m}": runs[m]["rel_l2"] for m in CASES},
        "train_time_s": {f"m{m}": runs[m]["hist"].train_time for m in CASES},
        "test_grid": "1001 uniform points on [0, 7]",
    })
    if ex.args.no_figures:
        return

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    for k, m in enumerate(CASES):
        r = runs[m]
        P.plot_reference(ax, x_test, r["y_ref"], label="reference" if k == 0 else None)
        P.plot_prediction(ax, x_test, r["y_pred"], label=f"$m={m}$", k=k, markevery=(25 * k, 100))
    ax.axhline(0.0, color=P.COLORS["light"], lw=0.6, zorder=0)
    ax.set_ylim(-1.0, 1.05)                    # the m = 0 parabola leaves the range at x ~ 3.5
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y(x)$")
    ax.legend(loc="lower right", ncol=2, fontsize=6.5)

    ax = axes[0, 1]
    for k, m in enumerate(CASES):
        P.plot_error(ax, x_test[1:], (runs[m]["y_pred"] - runs[m]["y_ref"])[1:], label=f"$m={m}$", k=k,
                     xlabel="$x$")                      # y(0) = 1 exactly: x > 0 shown
    ax.legend(loc="lower right", ncol=2)

    ax = axes[1, 0]
    for k, m in enumerate(CASES):
        h = runs[m]["hist"]
        ax.semilogy(h.iters, P.rolling_median(h.loss, 9), color=P.SERIES[k],
                    ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.1, label=f"$m={m}$")
    ax.axvline(runs[CASES[0]]["hist"].switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[1, 1]
    for k, m in enumerate(CASES):
        P.plot_metric(ax, runs[m]["hist"], "rel_l2", label=f"$m={m}$", k=k, show_switch=(k == 0))
    ax.legend(loc="upper right", ncol=2)
    P.label_panels(axes)
    ex.savefig(fig, "lane_emden_summary")


if __name__ == "__main__":
    main()
