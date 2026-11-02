"""Van der Pol oscillator  y'' - mu (1 - y^2) y' + y = 0,  y(0) = 1, y'(0) = 0,  t in [0, 10].

Book: Chapter 4 (Initial value problems), Section "Higher-order equations: the van der Pol
oscillator" (figure ivp:fig:vdp, table ivp:tab:vdp).

Three treatments of the two initial conditions are trained with the same network, seed,
collocation points and optimizer budget:

* hard_bounded - y_theta(t) = 1 + phi(t) N_theta(t),  phi(t) = t^2 / (1 + t^2)  (bounded);
* hard_power   - y_theta(t) = 1 + t^2 N_theta(t)  (the multiplier (t - t0)^n of the book's
                 trial solution; it grows to 100 at t = 10);
* soft         - y_theta = N_theta and the penalty  lambda_ic [(y_theta(0) - 1)^2 + y_theta'(0)^2].

In both hard variants phi(0) = phi'(0) = 0, so y_theta(0) = 1 and y_theta'(0) = 0 exactly.

Reference solution: DOP853 with rtol = atol = 1e-12 (pinnode.reference.ivp_reference).
Errors of y and y' are measured on a uniform grid of 1001 points.

Run:  python chapters/ch04_ivp/vanderpol.py           (3 trainings; about 3-4 min of CPU time on one core)
      python chapters/ch04_ivp/vanderpol.py --quick   (smoke test)
      --only <variant> trains one variant in its own process and checkpoints it; --resume reuses
      checkpoints (a final call with --resume assembles table and figure)
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, linf, rel_l2,
                     to_tensor, train, uniform)
from pinnode.reference import ivp_reference

MU, Y0, V0, T0, T = 0.5, 1.0, 0.0, 0.0, 10.0
WIDTH, DEPTH, N_COLLOC, N_TEST = 48, 3, 300, 1001
LAMBDA_IC = 1.0
VARIANTS = ["hard_bounded", "hard_power", "soft"]
LABELS = {"hard_bounded": "hard, $t^2/(1+t^2)$", "hard_power": "hard, $t^2$", "soft": "soft"}


def rhs(t, z):
    """First-order form used by the reference solver: z = (y, y')."""
    return [z[1], MU * (1.0 - z[0] ** 2) * z[1] - z[0]]


def build(variant: str):
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(T0, T))
    if variant == "hard_bounded":
        return ConstrainedModel(net, lambda t, n: Y0 + V0 * (t - T0) + (t - T0) ** 2 / (1.0 + (t - T0) ** 2) * n)
    if variant == "hard_power":
        return ConstrainedModel(net, lambda t, n: Y0 + V0 * (t - T0) + (t - T0) ** 2 * n)
    return net


def budget(ex: Example):
    return ex.pick(6000, 100), ex.pick(1000, 10)


def run_one(ex: Example, variant: str, t_test: np.ndarray, ref: np.ndarray):
    torch.manual_seed(ex.seed)
    model = build(variant)
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 40), T0, T), requires_grad=True)
    t0 = to_tensor([T0], requires_grad=True)
    t_test_t = to_tensor(t_test)
    hard = variant.startswith("hard")

    def loss_fn():
        y = model(t_col)
        dy = derivative(y, t_col, 1)
        d2y = derivative(dy, t_col, 1)
        r = d2y - MU * (1.0 - y ** 2) * dy + y
        L_r = (r ** 2).mean()
        if hard:
            return L_r, {"residual": L_r}
        y_0 = model(t0)
        L_ic = ((y_0 - Y0) ** 2).sum() + ((derivative(y_0, t0, 1) - V0) ** 2).sum()
        return L_r + LAMBDA_IC * L_ic, {"residual": L_r, "ic": L_ic}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), ref[:, 0])}

    n_adam, n_lbfgs = budget(ex)
    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=100, print_every=2000)
    t_req = to_tensor(t_test, requires_grad=True)
    y = model(t_req)
    dy = derivative(y, t_req, 1)
    d2y = derivative(dy, t_req, 1)
    res = (d2y - MU * (1.0 - y ** 2) * dy + y).detach().numpy().ravel()
    y, dy = y.detach().numpy().ravel(), dy.detach().numpy().ravel()
    return {"hist": hist, "y": y, "dy": dy, "residual": res,
            "rel_l2": rel_l2(y, ref[:, 0]), "linf": linf(y, ref[:, 0]),
            "rel_l2_dy": rel_l2(dy, ref[:, 1]), "linf_dy": linf(dy, ref[:, 1]),
            "max_abs_residual": float(np.max(np.abs(res))),
            "ic_error": [abs(y[0] - Y0), abs(dy[0] - V0)],
            "initial_loss": float(hist.loss[0]),
            "n_params": count_parameters(model)}


SCALARS = ("rel_l2", "linf", "rel_l2_dy", "linf_dy", "max_abs_residual", "ic_error", "initial_loss",
           "n_params")


def save_run(ex: Example, tag: str, r: dict, n_adam: int) -> None:
    ex.save_history(r["hist"], tag)
    np.savez(ex.path(f"run_{tag}.npz"), y=r["y"], dy=r["dy"], residual=r["residual"])
    meta = {k: r[k] for k in SCALARS}
    meta.update(train_time=r["hist"].train_time, final_loss=r["hist"].final_loss, n_adam=n_adam,
                last_iter=r["hist"].iters[-1])
    ex.path(f"run_{tag}.json").write_text(json.dumps(meta, indent=1, default=float))


def load_run(ex: Example, tag: str, n_adam: int):
    files = [ex.path(f"run_{tag}.json"), ex.path(f"run_{tag}.npz"), ex.path(f"history_{tag}.csv")]
    if not all(f.exists() for f in files):
        return None
    meta = json.loads(files[0].read_text())
    if meta.get("n_adam") != n_adam:
        return None
    import pandas as pd

    arr = np.load(files[1])
    df = pd.read_csv(files[2])
    ev = df.dropna(subset=["rel_l2"])
    hist = SimpleNamespace(iters=df["iter"].tolist(), loss=df["loss"].tolist(),
                           eval_iters=ev["iter"].tolist(), metrics={"rel_l2": ev["rel_l2"].tolist()},
                           switch_iter=n_adam if (df["phase"] == "lbfgs").any() else None,
                           train_time=meta["train_time"], final_loss=meta["final_loss"])
    return {"hist": hist, **{k: arr[k] for k in arr.files}, **{k: meta[k] for k in SCALARS}}


def make_figure(ex: Example, t, ref, out) -> None:
    from pinnode import plotting as P

    main = out["hard_bounded"]
    fig, axes = P.figure(2, 2, aspect=0.70)
    ax = axes[0, 0]
    P.plot_reference(ax, t, ref[:, 0], label="reference (DOP853)")
    for k, tag in enumerate(VARIANTS[:2]):
        P.plot_prediction(ax, t, out[tag]["y"], label=f"PINN, {LABELS[tag]}", k=k, markevery=(25 * k, 50))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("solution")
    ax.set_ylim(-2.3, 4.6)                                       # head room for the legend
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    P.plot_reference(ax, ref[:, 0], ref[:, 1], label="reference")
    P.plot_prediction(ax, main["y"], main["dy"], label=f"PINN, {LABELS['hard_bounded']}", k=0,
                      markevery=(0, 50))
    ax.plot([Y0], [V0], "o", color=P.COLORS["ref"], ms=3.5, zorder=4)
    ax.annotate("$t=0$", (Y0, V0), xytext=(6, -10), textcoords="offset points", fontsize=7.5)
    ax.set_xlabel("$y$")
    ax.set_ylabel("$y'$")
    ax.set_title("phase plane")
    ax.set_ylim(-2.4, 4.0)
    ax.legend(loc="upper center")

    ax = axes[1, 0]
    for k, tag in enumerate(VARIANTS):        # t = 0 omitted: the hard variants are exact there
        P.plot_error(ax, t[1:], (out[tag]["y"] - ref[:, 0])[1:], label=f"$|y_\\theta-y|$, {LABELS[tag]}",
                     k=k, lw=1.0, ls=P.LINESTYLES[k])
    P.plot_error(ax, t[1:], (main["dy"] - ref[:, 1])[1:], label=f"$|y_\\theta'-y'|$, {LABELS['hard_bounded']}",
                 k=3, lw=1.0, ls=P.LINESTYLES[3])
    ax.set_ylim(1e-17, 1e2)
    ax.set_title("pointwise error")
    ax.legend(loc="lower center", ncol=1)

    ax = axes[1, 1]
    for k, tag in enumerate(VARIANTS):
        h = out[tag]["hist"]
        ax.semilogy(h.eval_iters, h.metrics["rel_l2"], color=P.SERIES[k], ls=P.LINESTYLES[k], lw=1.1,
                    label=LABELS[tag])
    sw = main["hist"].switch_iter
    if sw:
        ax.axvline(sw, color=P.COLORS["muted"], lw=0.6, zorder=1)
        ax.text(sw, 0.97, "L-BFGS $\\rightarrow$ ", transform=ax.get_xaxis_transform(), ha="right",
                va="top", fontsize=7.5, color=P.COLORS["muted"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("relative $L^2$ error")
    ax.set_title("error during training")
    ax.legend(loc="lower left", ncol=1)
    P.label_panels(axes)
    ex.savefig(fig, "vanderpol_summary")


def main():
    ex = Example("ch04_ivp", "vanderpol",
                 description="van der Pol oscillator (mu=0.5): bounded/power hard ICs vs soft ICs",
                 default_seed=1234,
                 extra_args=[(("--resume",), {"action": "store_true",
                                              "help": "reuse checkpointed trainings"}),
                             (("--only",), {"choices": VARIANTS, "default": None,
                                            "help": "train one variant, checkpoint it and exit"})])
    t_test = np.linspace(T0, T, N_TEST)
    ref = ivp_reference(rhs, (T0, T), [Y0, V0], t_test)          # DOP853, rtol = atol = 1e-12
    n_adam, n_lbfgs = budget(ex)

    if ex.args.only:                                               # one training in its own process
        tag = ex.args.only
        if ex.args.resume and load_run(ex, tag, n_adam) is not None:
            ex.log(f"checkpoint {tag} exists")
            return
        ex.log(f"training {tag}")
        r = run_one(ex, tag, t_test, ref)
        save_run(ex, tag, r, n_adam)
        ex.log(f"{tag}: relL2={r['rel_l2']:.2e} relL2(y')={r['rel_l2_dy']:.2e} "
               f"loss={r['hist'].final_loss:.2e} initial loss={r['initial_loss']:.2e}")
        return

    def maximum(t, z):                                             # y' = 0 from above: maxima of y
        return z[1]
    maximum.direction = -1
    lc = solve_ivp(rhs, (T0, 100.0), [Y0, V0], method="DOP853", rtol=1e-12, atol=1e-12, events=maximum)
    period = float(np.mean(np.diff(lc.t_events[0][-5:])))          # after the transient
    ex.log(f"limit-cycle period = {period:.6f}")

    out, rows = {}, []
    for tag in VARIANTS:
        r = load_run(ex, tag, n_adam) if ex.args.resume else None
        if r is None:
            ex.log(f"training {tag}")
            r = run_one(ex, tag, t_test, ref)
            save_run(ex, tag, r, n_adam)
        else:
            ex.log(f"reusing checkpoint {tag}")
        out[tag] = r
        rows.append({"formulation": tag, "rel_l2": r["rel_l2"], "linf": r["linf"],
                     "rel_l2_dy": r["rel_l2_dy"], "linf_dy": r["linf_dy"],
                     "max_abs_residual": r["max_abs_residual"], "initial_loss": r["initial_loss"],
                     "final_loss": r["hist"].final_loss,
                     "ic_error_y": r["ic_error"][0], "ic_error_dy": r["ic_error"][1],
                     "iterations": r["hist"].iters[-1], "train_time_s": r["hist"].train_time})
        ex.log(f"{tag}: relL2={r['rel_l2']:.2e} linf={r['linf']:.2e} relL2(y')={r['rel_l2_dy']:.2e} "
               f"loss={r['hist'].final_loss:.2e} iters={r['hist'].iters[-1]} time={r['hist'].train_time:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_table({"t": t_test, "y_ref": ref[:, 0], "dy_ref": ref[:, 1],
                   **{f"y_{tag}": out[tag]["y"] for tag in VARIANTS},
                   **{f"dy_{tag}": out[tag]["dy"] for tag in VARIANTS},
                   "residual_hard_bounded": out["hard_bounded"]["residual"]}, "predictions")
    main_row = rows[0]
    ex.save_results({
        "problem": {"mu": MU, "y0": Y0, "dy0": V0, "interval": [T0, T],
                    "reference": "DOP853, rtol=atol=1e-12", "limit_cycle_period": period,
                    "max_abs_y_on_interval": float(np.max(np.abs(ref[:, 0])))},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "n_params": out["hard_bounded"]["n_params"],
                            "collocation": ex.pick(N_COLLOC, 40), "collocation_type": "uniform, fixed",
                            "adam_iters": n_adam, "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters_max": n_lbfgs, "lambda_ic_soft": LAMBDA_IC,
                            "trial_hard_bounded": "1 + t^2/(1+t^2) N(t)", "trial_hard_power": "1 + t^2 N(t)",
                            "test_grid": N_TEST},
        "main_variant": "hard_bounded",
        "rel_l2": main_row["rel_l2"], "linf": main_row["linf"], "train_time_s": main_row["train_time_s"],
        "iterations": main_row["iterations"],
        "summary": rows,
    })
    if not ex.args.no_figures:
        make_figure(ex, t_test, ref, out)


if __name__ == "__main__":
    main()
