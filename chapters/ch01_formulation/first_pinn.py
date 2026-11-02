"""A first PINN:  y' = -y + sin t,  y(0) = 1  on [0, T],  T = 10.

Book: Chapter 1 (Formulation and loss design), Section "The training loop", the worked
example "A first PINN" (code listing, results table and composite figure).

Two loss designs are compared on the same network and collocation set:

* soft - the initial condition is a penalty term:  L = L_r + lambda_ic (y_theta(0) - 1)^2;
* hard - the trial function y_theta(t) = 1 + t N_theta(t) satisfies y(0) = 1 exactly,
         so that L = L_r (exact enforcement of conditions is the subject of Chapter 3).

For an honest comparison the same problem is also solved with two classical adaptive
Runge-Kutta codes (RK45 and DOP853 from SciPy); for these we record the number of
right-hand-side evaluations, which, unlike wall-clock time, does not depend on the load
of the machine.

Exact solution: y(t) = 1.5 e^{-t} + (sin t - cos t) / 2.
Errors are measured on a uniform test grid of 1001 points.

Run:  PYTHONPATH=. python chapters/ch01_formulation/first_pinn.py          (about 1 min, 1 core)
      PYTHONPATH=. python chapters/ch01_formulation/first_pinn.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, ivp_transform,
                     linf, rel_l2, to_tensor, train, uniform)
from pinnode import plotting as P

T, Y0 = 10.0, 1.0
WIDTH, DEPTH, N_COLLOC = 32, 3, 200
LAM_IC = 1.0


def exact(t):
    return 1.5 * np.exp(-t) + 0.5 * (np.sin(t) - np.cos(t))


def rhs(t, y):
    return -y + np.sin(t)


def run_pinn(ex: Example, hard: bool):
    torch.manual_seed(ex.seed)
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, T))
    model = ConstrainedModel(net, ivp_transform(0.0, [Y0])) if hard else net
    t = to_tensor(uniform(ex.pick(N_COLLOC, 30), 0.0, T), requires_grad=True)
    t0 = to_tensor([0.0])
    t_test = np.linspace(0.0, T, 1001)
    y_ref = exact(t_test)
    t_test_t = to_tensor(t_test)

    # ------------------------------------------------------------- the loss (book listing)
    def loss_fn():
        y = model(t)
        r = derivative(y, t) + y - torch.sin(t)          # ODE residual at the collocation points
        L_r = (r ** 2).mean()
        if hard:
            return L_r, {"residual": L_r}
        L_ic = ((model(t0) - Y0) ** 2).sum()             # initial condition y(0) = 1
        return L_r + LAM_IC * L_ic, {"residual": L_r, "ic": L_ic}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(5000, 60), lr=1e-3,
                 lr_final=1e-5, lbfgs_iters=ex.pick(500, 10), eval_fn=eval_fn, eval_every=50,
                 print_every=2500)
    t_req = to_tensor(t_test, requires_grad=True)
    y = model(t_req)
    res = (derivative(y, t_req) + y - torch.sin(t_req)).detach().numpy().ravel()
    y = y.detach().numpy().ravel()
    with torch.no_grad():
        e0 = float(model(t0).item() - Y0)
    return {"hist": hist, "t": t_test, "y_ref": y_ref, "y_pred": y, "residual": res,
            "rel_l2": rel_l2(y, y_ref), "linf": linf(y, y_ref), "ic_defect": e0,
            "n_params": count_parameters(model)}


def run_classical(method: str, rtol: float, atol: float, t_test: np.ndarray):
    sol = solve_ivp(rhs, (0.0, T), [Y0], method=method, rtol=rtol, atol=atol, dense_output=True)
    y = sol.sol(t_test)[0]
    y_ref = exact(t_test)
    return {"method": method, "rtol": rtol, "atol": atol, "nfev": int(sol.nfev),
            "n_steps": int(len(sol.t) - 1), "rel_l2": rel_l2(y, y_ref), "linf": linf(y, y_ref)}


class _SavedHistory:
    """Minimal stand-in for pinnode.History, rebuilt from a saved history CSV."""

    def __init__(self, path):
        import pandas as pd

        df = pd.read_csv(path)
        self.iters = df["iter"].to_numpy()
        self.loss = df["loss"].to_numpy()
        self.parts = {c: df[c].to_numpy() for c in ("residual", "ic") if c in df}
        ev = df.dropna(subset=["rel_l2"]) if "rel_l2" in df else df.iloc[:0]
        self.eval_iters = ev["iter"].to_numpy()
        self.metrics = {"rel_l2": ev["rel_l2"].to_numpy()} if len(ev) else {}
        has_lbfgs = bool((df["phase"] == "lbfgs").any())
        self.switch_iter = int(df.loc[df["phase"] == "adam", "iter"].max()) + 1 if has_lbfgs else None


def make_figure(ex: Example, t_test, y_ref, pred, hist):
    fig, axes = P.figure(2, 2, aspect=0.70)
    ax = axes[0, 0]
    P.plot_reference(ax, t_test, y_ref, label="exact")
    P.plot_prediction(ax, t_test, pred["soft"], label="PINN, soft IC", k=0, markevery=(0, 100))
    P.plot_prediction(ax, t_test, pred["hard"], label="PINN, hard IC", k=1, markevery=(50, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.legend(loc="upper right")
    ax = axes[0, 1]
    P.plot_error(ax, t_test, pred["soft"] - y_ref, label="soft IC", k=0)
    P.plot_error(ax, t_test, pred["hard"] - y_ref, label="hard IC", k=1)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_ylabel(r"$|y_{\theta}(t)-y(t)|$")
    ax.legend(loc="upper left", ncol=2)
    ax = axes[1, 0]
    P.plot_history(ax, hist["soft"], parts=["residual", "ic"],
                   labels={"loss": "total (soft)", "residual": r"$\mathcal{L}_r$",
                           "ic": r"$\mathcal{L}_{ic}$"})
    ax.legend(loc="lower left")
    ax = axes[1, 1]
    P.plot_metric(ax, hist["soft"], "rel_l2", label="soft IC", k=0)
    P.plot_metric(ax, hist["hard"], "rel_l2", label="hard IC", k=1)
    P.label_panels(axes)
    ex.savefig(fig, "first_pinn_summary")


def main():
    ex = Example("ch01_formulation", "first_pinn",
                 description="First PINN for y' = -y + sin t: soft vs hard initial condition",
                 default_seed=1234,
                 extra_args=[(("--plot-only",), {"action": "store_true",
                                                 "help": "redraw the figure from saved outputs"})])
    if ex.args.plot_only:
        import pandas as pd

        df = pd.read_csv(ex.path("predictions.csv"))
        hist = {tag: _SavedHistory(ex.path(f"history_{tag}.csv")) for tag in ("soft", "hard")}
        make_figure(ex, df["t"].to_numpy(), df["y_exact"].to_numpy(),
                    {"soft": df["y_soft"].to_numpy(), "hard": df["y_hard"].to_numpy()}, hist)
        return
    out = {}
    rows = []
    for hard in (False, True):
        tag = "hard" if hard else "soft"
        ex.log(f"training PINN ({tag} initial condition)")
        r = run_pinn(ex, hard)
        out[tag] = r
        ex.save_history(r["hist"], tag)
        n_it = int(r["hist"].iters[-1]) if r["hist"].iters else 0
        rows.append({"formulation": tag, "rel_l2": r["rel_l2"], "linf": r["linf"],
                     "final_loss": r["hist"].final_loss, "ic_defect": r["ic_defect"],
                     "max_abs_residual_test": float(np.max(np.abs(r["residual"]))),
                     "rms_residual_test": float(np.sqrt(np.mean(r["residual"] ** 2))),
                     "iterations": n_it, "n_params": r["n_params"],
                     "train_time_s": r["hist"].train_time})
    t_test = out["soft"]["t"]
    classical = [run_classical("RK45", 1e-6, 1e-9, t_test),
                 run_classical("DOP853", 1e-12, 1e-12, t_test)]
    ex.save_table(rows, "summary")
    ex.save_table(classical, "classical")
    ex.save_table({"t": t_test, "y_exact": out["soft"]["y_ref"], "y_soft": out["soft"]["y_pred"],
                   "y_hard": out["hard"]["y_pred"], "r_soft": out["soft"]["residual"],
                   "r_hard": out["hard"]["residual"]}, "predictions")
    ex.save_results({
        "problem": {"ode": "y' = -y + sin t", "y0": Y0, "T": T,
                    "exact": "1.5 exp(-t) + (sin t - cos t)/2", "test_grid": 1001},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "collocation": ex.pick(N_COLLOC, 30), "collocation_type": "uniform",
                            "adam_iters": ex.pick(5000, 60), "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters_max": ex.pick(500, 10), "lambda_ic": LAM_IC},
        "summary": rows, "classical": classical,
        "rel_l2": out["hard"]["rel_l2"], "linf": out["hard"]["linf"],
        "train_time_s": out["hard"]["hist"].train_time})
    for row in rows:
        ex.log(f"{row['formulation']}: relL2={row['rel_l2']:.2e} linf={row['linf']:.2e} "
               f"loss={row['final_loss']:.2e} iters={row['iterations']} time={row['train_time_s']:.1f}s")
    for c in classical:
        ex.log(f"{c['method']} (rtol={c['rtol']:g}): relL2={c['rel_l2']:.2e} linf={c['linf']:.2e} "
               f"nfev={c['nfev']} steps={c['n_steps']}")

    if ex.args.no_figures:
        return
    make_figure(ex, t_test, out["soft"]["y_ref"],
                {"soft": out["soft"]["y_pred"], "hard": out["hard"]["y_pred"]},
                {"soft": out["soft"]["hist"], "hard": out["hard"]["hist"]})


if __name__ == "__main__":
    main()
