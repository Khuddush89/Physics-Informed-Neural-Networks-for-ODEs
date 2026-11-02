"""Lotka-Volterra predator-prey system solved by a PINN in logarithmic variables.

Book: Chapter 4 (Initial value problems), Section "Systems of first-order equations: the
Lotka-Volterra model" (figures ivp:fig:lv and ivp:fig:lv-long, table ivp:tab:lv).

    y1' = a y1 - b y1 y2,   y2' = -c y2 + d y1 y2,   (a, b, c, d) = (1, 0.1, 1.5, 0.075),
    y1(0) = 10, y2(0) = 5,  t in [0, T].

For positive solutions the substitution u_i = ln y_i gives the equivalent system

    u1' = a - b exp(u2),    u2' = -c + d exp(u1),

whose residual is the *relative* residual of the original equations, (y_i' - f_i(y)) / y_i.
The network represents u through the trial solution  u_theta(t) = ln y(0) + t N_theta(t)
(exact initial conditions) and is trained on the residual alone: no loss weights and no
solution data are used. y_theta = exp(u_theta) is positive by construction.

The orbit through (10, 5) has period 5.48. The main example uses T = 6 (about 1.1 periods).
With the same network and budget on the horizon of 2.7 periods (T = 15) the PINN follows the
first half period and then spirals into the equilibrium (20, 10); that run is reproduced with

    python chapters/ch04_ivp/lotka_volterra.py --T 15 --outdir outputs/ch04_ivp/lotka_volterra_T15

Reference: DOP853 with rtol = atol = 1e-12. Diagnostics: relative L2 / max errors of each
component on 1001 uniform points and the drift of the first integral
H(y1, y2) = d y1 - c ln y1 + b y2 - a ln y2, which is constant along exact solutions.

Run:  python chapters/ch04_ivp/lotka_volterra.py            (about 2 min of CPU time on one core)
      python chapters/ch04_ivp/lotka_volterra.py --quick    (smoke test)
      add --replot to redraw the figures from the saved predictions and history (no training)
"""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, linf, rel_l2,
                     to_tensor, train, uniform)
from pinnode.reference import ivp_reference

A, B, C, D = 1.0, 0.1, 1.5, 0.075
Y10, Y20 = 10.0, 5.0
T0, T = 0.0, 6.0
WIDTH, DEPTH, N_COLLOC, N_TEST = 64, 3, 400, 1001


def rhs(t, y):
    return [A * y[0] - B * y[0] * y[1], -C * y[1] + D * y[0] * y[1]]


def first_integral(y1, y2):
    return D * y1 - C * np.log(y1) + B * y2 - A * np.log(y2)


def orbit_period() -> float:
    """Period of the orbit through (10, 5) from successive upward crossings of y1 = 10."""
    def crossing(t, y):
        return y[0] - Y10
    crossing.direction = 1
    orb = solve_ivp(rhs, (T0, 60.0), [Y10, Y20], method="DOP853", rtol=1e-12, atol=1e-12, events=crossing)
    te = orb.t_events[0]
    te = te[np.concatenate([[True], np.diff(te) > 1e-6])]
    return float(np.mean(np.diff(te)))


def make_figures(ex: Example, t, ref, y, hist) -> None:
    from pinnode import plotting as P

    H_pinn, H0 = first_integral(y[:, 0], y[:, 1]), first_integral(Y10, Y20)
    fig, axes = P.figure(2, 2, aspect=0.70)
    ax = axes[0, 0]
    P.plot_reference(ax, t, ref[:, 0], label="reference")
    P.plot_reference(ax, t, ref[:, 1], label=None)
    P.plot_prediction(ax, t, y[:, 0], label="prey $y_1$ (PINN)", k=0, markevery=(0, 60))
    P.plot_prediction(ax, t, y[:, 1], label="predator $y_2$ (PINN)", k=1, markevery=(30, 60))
    ax.set_xlabel("$t$")
    ax.set_ylabel("population")
    ax.set_title("time series")
    ax.set_ylim(0.0, 1.5 * float(np.max(ref)))           # head room for the legend
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    P.plot_reference(ax, ref[:, 0], ref[:, 1], label="reference")
    P.plot_prediction(ax, y[:, 0], y[:, 1], label="PINN", k=0, markevery=(0, 80))
    ax.plot([C / D], [A / B], "+", color=P.COLORS["ref"], ms=6, mew=0.9, label="equilibrium")
    ax.plot([Y10], [Y20], "o", color=P.COLORS["ref"], ms=3.5, zorder=4)
    ax.annotate("$t=0$", (Y10, Y20), xytext=(5, -10), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("prey $y_1$")
    ax.set_ylabel("predator $y_2$")
    ax.set_title("phase portrait")
    ax.set_ylim(top=1.3 * float(np.max(ref[:, 1])))
    ax.legend(loc="upper right")

    ax = axes[1, 0]                       # t = 0 omitted: the trial solution is exact there
    P.plot_error(ax, t[1:], (y[:, 0] - ref[:, 0])[1:], label="$|y_{1,\\theta}-y_1|$", k=0, lw=1.0,
                 ls=P.LINESTYLES[0])
    P.plot_error(ax, t[1:], (y[:, 1] - ref[:, 1])[1:], label="$|y_{2,\\theta}-y_2|$", k=1, lw=1.0,
                 ls=P.LINESTYLES[1])
    P.plot_error(ax, t[1:], (H_pinn - H0)[1:], label="$|H(y_\\theta)-H(y(0))|$", k=2, lw=1.0,
                 ls=P.LINESTYLES[2])
    ax.set_ylim(1e-9, max(1.0, 10.0 * float(np.max(np.abs(y - ref)))))
    ax.set_ylabel("error")
    ax.set_title("pointwise error")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    P.plot_history(ax, hist, parts=["prey", "predator"],
                   labels={"loss": "total loss", "prey": "residual $u_1$", "predator": "residual $u_2$"},
                   show_switch=False)
    ax.semilogy(hist.eval_iters, hist.metrics["rel_l2"], color=P.SERIES[3], ls=P.LINESTYLES[3], lw=1.1,
                label="rel. $L^2$ error")
    if hist.switch_iter:
        ax.axvline(hist.switch_iter, color=P.COLORS["muted"], lw=0.6, zorder=1)
        ax.text(hist.switch_iter, 0.97, "L-BFGS $\\rightarrow$ ", transform=ax.get_xaxis_transform(),
                ha="right", va="top", fontsize=7.5, color=P.COLORS["muted"])
    ax.set_ylabel("loss / error")
    ax.set_title("training history")
    ax.legend(loc="lower left")
    P.label_panels(axes)
    ex.savefig(fig, "lotka_volterra_summary")

    # half-width phase portrait (used for the long-horizon run)
    fig, ax = P.figure(1, 1, width="half", aspect=0.95)
    P.plot_reference(ax, ref[:, 0], ref[:, 1], label="reference")
    P.plot_prediction(ax, y[:, 0], y[:, 1], label="PINN", k=0, markevery=(0, 80))
    ax.plot([C / D], [A / B], "+", color=P.COLORS["ref"], ms=6, mew=0.9, label="equilibrium")
    ax.plot([Y10], [Y20], "o", color=P.COLORS["ref"], ms=3.5, zorder=4)
    ax.set_xlabel("prey $y_1$")
    ax.set_ylabel("predator $y_2$")
    ax.set_ylim(top=1.3 * float(np.max(ref[:, 1])))     # head room for the legend
    ax.legend(loc="upper right", ncol=2)
    ex.savefig(fig, "lotka_volterra_phase")


def load_saved(ex: Example):
    """Predictions and history written by a previous full run (for --replot)."""
    import pandas as pd

    pr = pd.read_csv(ex.path("predictions.csv"))
    df = pd.read_csv(ex.path("history.csv"))
    ev = df.dropna(subset=["rel_l2"])
    res = json.loads(ex.path("results.json").read_text())
    n_adam = res["hyperparameters"]["adam_iters"]
    hist = SimpleNamespace(iters=df["iter"].tolist(), loss=df["loss"].tolist(),
                           parts={"prey": df["prey"].tolist(), "predator": df["predator"].tolist()},
                           eval_iters=ev["iter"].tolist(), metrics={"rel_l2": ev["rel_l2"].tolist()},
                           switch_iter=n_adam if (df["phase"] == "lbfgs").any() else None)
    t = pr["t"].to_numpy()
    ref = pr[["y1_ref", "y2_ref"]].to_numpy()
    y = pr[["y1_pinn", "y2_pinn"]].to_numpy()
    return t, ref, y, hist


def main():
    ex = Example("ch04_ivp", "lotka_volterra",
                 description="Lotka-Volterra system, PINN in log variables with hard ICs",
                 default_seed=1234,
                 extra_args=[(("--T",), {"type": float, "default": T, "help": "final time"}),
                             (("--replot",), {"action": "store_true",
                                              "help": "redraw figures from saved outputs (no training)"})])
    if ex.args.replot:
        make_figures(ex, *load_saved(ex))
        return
    t_end = ex.args.T
    t_test = np.linspace(T0, t_end, N_TEST)
    ref = ivp_reference(rhs, (T0, t_end), [Y10, Y20], t_test)     # DOP853, rtol = atol = 1e-12
    u0 = torch.tensor([[math.log(Y10), math.log(Y20)]])
    period = orbit_period()
    ex.log(f"period of the orbit = {period:.6f}, horizon covers {t_end / period:.2f} periods")

    torch.manual_seed(ex.seed)
    net = MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh", domain=(T0, t_end))
    model = ConstrainedModel(net, lambda t, n: u0 + (t - T0) * n)
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 40), T0, t_end), requires_grad=True)
    t_test_t = to_tensor(t_test)

    def residuals(u, t):
        du = derivative(u, t, 1)
        r1 = du[:, :1] - (A - B * torch.exp(u[:, 1:2]))
        r2 = du[:, 1:2] - (-C + D * torch.exp(u[:, :1]))
        return r1, r2

    def loss_fn():
        r1, r2 = residuals(model(t_col), t_col)
        L1, L2 = (r1 ** 2).mean(), (r2 ** 2).mean()
        return L1 + L2, {"prey": L1, "predator": L2}

    def eval_fn():
        with torch.no_grad():
            y = torch.exp(model(t_test_t)).numpy()
        return {"rel_l2": rel_l2(y, ref)}

    n_adam, n_lbfgs = ex.pick(12000, 100), ex.pick(1000, 10)
    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=100, print_every=2000)
    ex.save_history(hist)

    t_req = to_tensor(t_test, requires_grad=True)
    u = model(t_req)
    r1, r2 = residuals(u, t_req)
    y = torch.exp(u).detach().numpy()
    res = np.hstack([r1.detach().numpy(), r2.detach().numpy()])
    H_pinn = first_integral(y[:, 0], y[:, 1])
    H_ref = first_integral(ref[:, 0], ref[:, 1])
    H0 = first_integral(Y10, Y20)
    metrics = {
        "rel_l2": rel_l2(y, ref), "linf": linf(y, ref),
        "rel_l2_prey": rel_l2(y[:, 0], ref[:, 0]), "rel_l2_predator": rel_l2(y[:, 1], ref[:, 1]),
        "linf_prey": linf(y[:, 0], ref[:, 0]), "linf_predator": linf(y[:, 1], ref[:, 1]),
        "max_abs_residual_log": float(np.max(np.abs(res))),
        "H_drift_pinn": float(np.max(np.abs(H_pinn - H0))),
        "H_drift_reference": float(np.max(np.abs(H_ref - H0))),
        "final_loss": hist.final_loss, "iterations": hist.iters[-1],
        "train_time_s": hist.train_time, "n_params": count_parameters(model),
    }
    for k, v in metrics.items():
        ex.log(f"{k} = {v:.3e}" if isinstance(v, float) else f"{k} = {v}")
    ex.save_table({"t": t_test, "y1_ref": ref[:, 0], "y2_ref": ref[:, 1], "y1_pinn": y[:, 0],
                   "y2_pinn": y[:, 1], "r_log1": res[:, 0], "r_log2": res[:, 1], "H_pinn": H_pinn},
                  "predictions")
    ex.save_results({
        "problem": {"a": A, "b": B, "c": C, "d": D, "y0": [Y10, Y20], "interval": [T0, t_end],
                    "equilibrium": [C / D, A / B], "H0": H0,
                    "reference": "DOP853, rtol=atol=1e-12", "period": period,
                    "periods_covered": t_end / period,
                    "small_oscillation_period": 2 * math.pi / math.sqrt(A * C),
                    "y1_range": [float(ref[:, 0].min()), float(ref[:, 0].max())],
                    "y2_range": [float(ref[:, 1].min()), float(ref[:, 1].max())],
                    "u_range": [float(np.log(ref).min()), float(np.log(ref).max())]},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "outputs": "u = ln y (2)", "trial": "u(0) + t N(t)",
                            "collocation": ex.pick(N_COLLOC, 40), "collocation_type": "uniform, fixed",
                            "adam_iters": n_adam, "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters_max": n_lbfgs, "loss_weights": "none", "test_grid": N_TEST},
        **metrics,
    })
    if not ex.args.no_figures:
        make_figures(ex, t_test, ref, y, hist)


if __name__ == "__main__":
    main()
