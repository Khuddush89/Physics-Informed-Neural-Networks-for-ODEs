"""Soft versus hard enforcement of conditions: an IVP and a Robin BVP for the Airy oscillator.

Book: Chapter 3, Section "Numerical comparison" (arch:sec:softhard-num), Figure arch:fig:softhard,
Table arch:tab:softhard.

Equation  y'' + t y = 0 on [0, 10], exact solution y(t) = Ai(-t).

IVP   y(0) = Ai(0), y'(0) = -Ai'(0) on [0, T_IVP], T_IVP = 6 (three zeros of the solution):
      soft        loss = mean(R^2) + lambda_ic [(y(0) - y0)^2 + (y'(0) - v0)^2],  lambda_ic = 1
      hard tau=1  y = y0 + v0 t + t^2 N             (pinnode.ivp_transform)
      hard tau=T  y = y0 + v0 t + (t/T_IVP)^2 N     (normalized multiplier, Proposition arch:prop:ivp)
IVP on the full interval [0, 10] (six zeros), hard tau=T only: shows that the
      IVP is much harder than the BVP on the same interval under the same budget.
Robin BVP   y(0) = Ai(0), y'(10) + y(10) = -Ai'(-10) + Ai(-10):
      soft        loss = mean(R^2) + lambda_bc [(y(0) - ya)^2 + (y'(T) + y(T) - gamma)^2], lambda_bc = 1
      hard        constrained expression (arch:eq:dirichlet-robin)

Network: Fourier-feature network (16 random frequencies B_k ~ N(0, 1), two hidden layers of 32
tanh units), the most accurate network of architecture_comparison.py (``--network chnn`` uses the
Chebyshev network of that script instead); 200 equispaced collocation points; Adam
(1e-3 -> 1e-5, cosine) followed by at most 1000 L-BFGS iterations; errors and residual
diagnostics on 1001 equispaced test points.

Run:  PYTHONPATH=. python chapters/ch03_architectures/soft_vs_hard.py          (~4 min on 1 core)
      PYTHONPATH=. python chapters/ch03_architectures/soft_vs_hard.py --quick
"""
from __future__ import annotations

import numpy as np
import torch

from _ch03_common import (N_COLLOCATION, N_TEST, ROBIN_BETA1, ROBIN_BETA2, ROBIN_GAMMA, T_END, T_IVP,
                         V_A, Y_A, DirichletRobinModel, collocation, ivp_normalized, load_history,
                         residual, test_grid)
from pinnode import (ChebyshevNet, ConstrainedModel, Example, FourierFeatureNet, count_parameters,
                     derivative, ivp_transform, linf, rel_l2, to_tensor, train)
from pinnode import plotting as P

DEGREE, WIDTH, DEPTH, N_FF, SIGMA_FF = 10, 32, 2, 16, 1.0
NETWORK = {"name": "ff"}                       # set from the command line in main()
LAMBDA = 1.0
CASES = [("ivp", "soft"), ("ivp", "hard_tau1"), ("ivp", "hard_tauT"), ("ivp10", "hard_tauT"),
         ("robin", "soft"), ("robin", "hard")]
LABELS = {("ivp", "soft"): "soft ($\\lambda_{ic}=1$)", ("ivp", "hard_tau1"): "hard, $\\tau=1$",
          ("ivp", "hard_tauT"): "hard, $\\tau=T$", ("ivp10", "hard_tauT"): "hard, $\\tau=T$ ($T=10$)",
          ("robin", "soft"): "soft ($\\lambda_{bc}=1$)",
          ("robin", "hard"): "hard (constrained expr.)"}


def interval_end(problem: str) -> float:
    return T_IVP if problem == "ivp" else T_END          # "ivp10" and "robin" use [0, 10]


def make_net(T: float):
    if NETWORK["name"] == "chnn":
        return ChebyshevNet(degree=DEGREE, hidden=(WIDTH,) * DEPTH, activation="tanh", domain=(0.0, T))
    return FourierFeatureNet(1, 1, n_features=N_FF, sigma=SIGMA_FF, width=WIDTH, depth=DEPTH,
                             activation="tanh", domain=(0.0, T))


def build(problem: str, variant: str) -> torch.nn.Module:
    net = make_net(interval_end(problem))
    if problem.startswith("ivp") and variant == "hard_tau1":
        return ConstrainedModel(net, ivp_transform(0.0, [Y_A], [V_A]))
    if problem.startswith("ivp") and variant == "hard_tauT":
        return ConstrainedModel(net, ivp_normalized(Y_A, V_A, 0.0, interval_end(problem)))
    if problem == "robin" and variant == "hard":
        return DirichletRobinModel(net)
    return net                                     # soft constraints: y = N


def value_and_derivative(model, t: float):
    tt = torch.full((1, 1), t, requires_grad=True)
    y = model(tt)
    return y, derivative(y, tt, 1)


def condition_terms(model, problem: str):
    """Signed violations of the two conditions (tensors, differentiable)."""
    if problem.startswith("ivp"):
        y0, dy0 = value_and_derivative(model, 0.0)
        return [y0 - Y_A, dy0 - V_A]
    ya, _ = value_and_derivative(model, 0.0)
    yb, dyb = value_and_derivative(model, T_END)
    return [ya - Y_A, ROBIN_BETA1 * yb + ROBIN_BETA2 * dyb - ROBIN_GAMMA]


def run_one(ex: Example, problem: str, variant: str, n_adam: int, n_lbfgs: int) -> dict:
    torch.manual_seed(ex.seed)
    model = build(problem, variant)
    soft = variant == "soft"
    T = interval_end(problem)
    t_col = to_tensor(collocation(T), requires_grad=True)
    t_test, y_ref = test_grid(T)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        r = residual(model, t_col)
        L_r = (r ** 2).mean()
        if not soft:
            return L_r, {"residual": L_r}
        c1, c2 = condition_terms(model, problem)
        L_c = (c1 ** 2 + c2 ** 2).sum()
        return L_r + LAMBDA * L_c, {"residual": L_r, "conditions": L_c}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=100, print_every=2000)
    with torch.no_grad():
        y_pred = model(t_test_t).numpy().ravel()
    viol = [abs(float(c.detach())) for c in condition_terms(model, problem)]
    # residual diagnostics: at the collocation points (what the loss sees) and on the test grid
    res_col = residual(model, t_col).detach().numpy().ravel()
    t_req = to_tensor(t_test, requires_grad=True)
    res_test = residual(model, t_req).detach().numpy().ravel()
    lbfgs_done = (hist.iters[-1] - n_adam) if n_lbfgs > 0 else 0
    return {"hist": hist, "t": t_test, "y_ref": y_ref, "y_pred": y_pred,
            "row": {"problem": problem, "variant": variant, "parameters": count_parameters(model),
                    "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
                    "condition_violation_1": viol[0], "condition_violation_2": viol[1],
                    "max_condition_violation": max(viol), "final_loss": hist.final_loss,
                    "rms_residual_collocation": float(np.sqrt(np.mean(res_col ** 2))),
                    "rms_residual_test": float(np.sqrt(np.mean(res_test ** 2))),
                    "max_abs_residual_test": float(np.max(np.abs(res_test))),
                    "adam_iters": n_adam, "lbfgs_iters_done": int(lbfgs_done),
                    "train_time_s": hist.train_time}}


def main():
    ex = Example("ch03_architectures", "soft_vs_hard",
                 description="Airy oscillator: soft vs hard conditions for an IVP and a Robin BVP",
                 default_seed=1234,
                 extra_args=[(("--adam",), {"type": int, "default": None,
                                            "help": "override the number of Adam iterations"}),
                             (("--lbfgs",), {"type": int, "default": None,
                                             "help": "override the maximum number of L-BFGS iterations"}),
                             (("--only",), {"type": str, "default": "",
                                            "help": "comma-separated subset, e.g. ivp:soft,robin:hard"}),
                             (("--plot-only",), {"action": "store_true",
                                                 "help": "redraw the figure from saved outputs"}),
                             (("--network",), {"type": str, "default": "ff", "choices": ["ff", "chnn"],
                                               "help": "Fourier-feature (default) or Chebyshev network"})])
    NETWORK["name"] = ex.args.network
    n_adam = ex.args.adam if ex.args.adam is not None else ex.pick(full=4000, quick=60)
    n_lbfgs = ex.args.lbfgs if ex.args.lbfgs is not None else ex.pick(full=1000, quick=10)
    cases = [tuple(c.split(":")) for c in ex.args.only.split(",") if c] or CASES
    if ex.args.plot_only:
        import json

        import pandas as pd
        res = json.loads(ex.path("results.json").read_text())
        n_adam = res["hyperparameters"]["adam_iters"]
        cases = [(r["problem"], r["variant"]) for r in res["summary"]]
        pred = pd.read_csv(ex.path("predictions.csv"))
        out = {(p, v): {"t": pred[f"t_{p}"].to_numpy(), "y_ref": pred[f"exact_{p}"].to_numpy(),
                        "y_pred": pred[f"pred_{p}_{v}"].to_numpy(),
                        "hist": load_history(ex.path(f"history_{p}_{v}.csv"), switch_iter=n_adam)}
               for p, v in cases}
        make_figure(ex, out, cases, n_adam)
        return
    out, rows = {}, []
    for problem, variant in cases:
        ex.log(f"training {problem} / {variant}")
        r = run_one(ex, problem, variant, n_adam, n_lbfgs)
        out[(problem, variant)] = r
        rows.append(r["row"])
        ex.save_history(r["hist"], f"{problem}_{variant}")
        ex.log(f"{problem:5s} {variant:9s} relL2={r['row']['rel_l2']:.3e} linf={r['row']['linf']:.3e} "
               f"viol={r['row']['max_condition_violation']:.2e} loss={r['row']['final_loss']:.3e} "
               f"rmsR_col={r['row']['rms_residual_collocation']:.2e} "
               f"rmsR_test={r['row']['rms_residual_test']:.2e} time={r['row']['train_time_s']:.1f}s")
    ex.save_table(rows, "summary")
    table = {}
    for p, v in cases:                       # the IVP on [0, 6] and the [0, 10] problems use different grids
        table.setdefault(f"t_{p}", out[(p, v)]["t"])
        table.setdefault(f"exact_{p}", out[(p, v)]["y_ref"])
        table[f"pred_{p}_{v}"] = out[(p, v)]["y_pred"]
    ex.save_table(table, "predictions")
    ex.save_results({
        "problem": {"equation": "y'' + t y = 0", "interval": [0.0, T_END], "exact": "Ai(-t)",
                    "ivp": {"y(0)": Y_A, "y'(0)": V_A, "interval": [0.0, T_IVP]},
                    "robin_bvp": {"y(0)": Y_A, "beta1": ROBIN_BETA1, "beta2": ROBIN_BETA2,
                                  "gamma": ROBIN_GAMMA, "interval": [0.0, T_END]}},
        "hyperparameters": {"network": ("FourierFeatureNet (q=16, sigma=1)" if NETWORK["name"] == "ff"
                                        else f"ChebyshevNet (degree {DEGREE})"),
                            "width": WIDTH, "hidden_layers": DEPTH, "activation": "tanh",
                            "collocation": N_COLLOCATION, "test_points": N_TEST,
                            "adam_iters": n_adam, "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters_max": n_lbfgs, "lambda_ic = lambda_bc": LAMBDA},
        "summary": rows,
    })

    if ex.args.no_figures:
        return
    make_figure(ex, out, cases, n_adam)


def make_figure(ex: Example, out: dict, cases: list, n_adam: int) -> None:
    """2x2 figure: pointwise error and relative error history for the IVP (on [0, T_IVP]) and
    the Robin BVP; one style per case in both panels of a column."""
    def style(k, every):
        return {"color": P.SERIES[k % 6], "ls": P.LINESTYLES[k % 6], "marker": P.MARKERS[k % 6],
                "markevery": every, "markersize": 3.0, "markerfacecolor": "white",
                "markeredgewidth": 0.8, "lw": 1.2}

    fig, axes = P.figure(2, 2, aspect=0.72)
    for col, (problem, title) in enumerate((("ivp", "IVP on $[0,6]$"), ("robin", "Robin BVP"))):
        sel = [c for c in cases if c[0] == problem]
        ax_err, ax_hist = axes[0, col], axes[1, col]
        for k, case in enumerate(sel):
            r = out[case]
            err = np.maximum(np.abs(r["y_pred"] - r["y_ref"]), 1e-13)
            ax_err.semilogy(r["t"], err, label=LABELS[case], **style(k, (25 * k, 125)))
            h = r["hist"]
            ax_hist.semilogy(h.eval_iters, h.metrics["rel_l2"], label=LABELS[case], **style(k, (2 * k, 8)))
        ax_err.set_ylim(1e-8, 5.0)
        ax_err.set_xlabel("$t$")
        ax_err.set_ylabel("absolute error")
        ax_err.set_title(f"{title}: pointwise error")
        ax_hist.axvline(n_adam, color=P.COLORS["muted"], lw=0.6, zorder=1)
        ax_hist.set_xlabel("iteration")
        ax_hist.set_ylabel("relative $L^2$ error")
        ax_hist.set_title(f"{title}: relative $L^2$ error")
        ax_err.legend(loc="upper right", fontsize=6.5, ncol=1)
        ax_hist.legend(loc="upper right" if problem == "ivp" else "lower left", fontsize=6.5)
    P.label_panels(axes)
    ex.savefig(fig, "soft_vs_hard")


if __name__ == "__main__":
    main()
