"""Logistic ODE  y' = r y (1 - y),  y(0) = y0  on [0, T]: activation-function study.

Book: Chapter 4 (Initial value problems), Section "First-order equations: the logistic ODE"
(figure ivp:fig:logistic, table ivp:tab:logistic).

Two formulations are compared for four activations (tanh, sigmoid, ReLU, swish):

* soft - the initial condition enters the loss as a penalty  w_ic (y_theta(0) - y0)^2;
* hard - the trial solution  y_theta(t) = y0 + t N_theta(t)  satisfies y(0) = y0 exactly.

Diagnostics written to results.json explain the plateaus seen during training:

* the minimum of the soft loss over affine functions y = A + B t (and of the hard loss over
  y = y0 + t (B + C t)), computed on the same collocation points; a network that stays in the
  nearly linear regime of its activations cannot go below this value. Because the
  equilibrium y = 0 solves the ODE exactly, the best affine function lies close to zero;
* the relative error at the end of the Adam phase (snapshot) and after L-BFGS;
* for ReLU networks, the number of hidden units that are active somewhere on [0, T].

Exact solution: y(t) = y0 e^{rt} / (1 + y0 (e^{rt} - 1)). Errors are measured on a uniform
grid of 1001 points.

Run:  python chapters/ch04_ivp/logistic_activation_study.py           (8 trainings; about 3-4 min
                                                                       of CPU time on one core)
      python chapters/ch04_ivp/logistic_activation_study.py --quick   (smoke test)
      add --resume to reuse trainings already completed (each run is checkpointed)
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, linf, rel_l2,
                     to_tensor, train, uniform)

# pandas, scipy.optimize and the plotting module are imported where they are used, so that a
# single training (option --only) runs in a small process.

R, Y0, T = 2.0, 0.1, 2.0
W_IC = 1.0
WIDTH, DEPTH = 32, 3
ACTIVATIONS = ["tanh", "sigmoid", "relu", "swish"]
LABELS = {"tanh": "tanh", "sigmoid": "sigmoid", "relu": "ReLU", "swish": "swish"}
N_TEST = 1001


def exact(t):
    return Y0 * np.exp(R * t) / (1.0 + Y0 * (np.exp(R * t) - 1.0))


def build(activation: str, hard: bool):
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation=activation, domain=(0.0, T))
    if hard:
        return ConstrainedModel(net, lambda t, n: Y0 + t * n), net
    return net, net


def affine_plateaus(t_col: np.ndarray) -> dict:
    """Minimum of the discrete losses over low-dimensional trial families (see docstring)."""
    from scipy.optimize import minimize

    def soft_affine(p):
        A, B = p
        y = A + B * t_col
        r = B - R * y * (1.0 - y)
        return np.mean(r ** 2) + W_IC * (A - Y0) ** 2

    def hard_affine_n(p):
        B, C = p
        y = Y0 + t_col * (B + C * t_col)
        r = (B + 2.0 * C * t_col) - R * y * (1.0 - y)
        return np.mean(r ** 2)

    opts = dict(xatol=1e-12, fatol=1e-18, maxiter=20000)
    s = minimize(soft_affine, [0.05, 0.0], method="Nelder-Mead", options=opts)
    h = minimize(hard_affine_n, [0.1, 0.0], method="Nelder-Mead", options=opts)
    A, B = s.x
    y = A + B * t_col
    res = np.mean((B - R * y * (1.0 - y)) ** 2)
    return {"soft_affine_y": {"A": A, "B": B, "loss": s.fun, "residual_part": res,
                              "ic_part": W_IC * (A - Y0) ** 2},
            "hard_affine_N": {"B": h.x[0], "C": h.x[1], "loss": h.fun},
            "trivial_solution_loss_soft": W_IC * Y0 ** 2}


def relu_active_units(net: MLP, t: torch.Tensor) -> list:
    """Number of hidden units per layer that are positive at some test point."""
    x = net.scale(t)
    counts = []
    with torch.no_grad():
        for layer in net.body:
            x = layer(x)
            if isinstance(layer, torch.nn.ReLU):
                counts.append(int((x > 0).any(dim=0).sum()))
    return counts


def run_one(ex: Example, activation: str, hard: bool, n_adam: int, n_lbfgs: int):
    torch.manual_seed(ex.seed)
    model, net = build(activation, hard)
    t_col_np = uniform(ex.pick(100, 20), 0.0, T)
    t_col = to_tensor(t_col_np, requires_grad=True)
    t0 = to_tensor([0.0])
    t_test = np.linspace(0.0, T, N_TEST)
    y_ref = exact(t_test)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y = model(t_col)
        r = derivative(y, t_col) - R * y * (1.0 - y)
        L_r = (r ** 2).mean()
        if hard:
            return L_r, {"residual": L_r}
        L_ic = ((model(t0) - Y0) ** 2).mean()
        return L_r + W_IC * L_ic, {"residual": L_r, "ic": L_ic}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    def snapshot_fn():
        with torch.no_grad():
            return model(t_test_t).numpy().ravel()

    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50, print_every=4000,
                 snapshot_fn=snapshot_fn, snapshot_every=max(n_adam - 1, 1))
    t_req = to_tensor(t_test, requires_grad=True)
    y_pred = model(t_req)
    res = (derivative(y_pred, t_req) - R * y_pred * (1.0 - y_pred)).detach().numpy().ravel()
    y_pred = y_pred.detach().numpy().ravel()
    y_adam = hist.snapshots[hist.snapshot_iters.index(max(n_adam - 1, 0))]   # end of Adam phase
    coef = np.polyfit(t_test, y_adam, 1)
    out = {"hist": hist, "t": t_test, "y_ref": y_ref, "y_pred": y_pred, "residual": res,
           "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
           "rel_l2_after_adam": rel_l2(y_adam, y_ref),
           "loss_after_adam": float(hist.loss[hist.iters.index(n_adam - 1)]),
           "affine_dev_after_adam": float(np.max(np.abs(np.polyval(coef, t_test) - y_adam))),
           "affine_fit_after_adam": coef.tolist(),
           "y_theta_0": float(y_pred[0]), "n_params": count_parameters(model)}
    if activation == "relu":
        out["relu_active_units"] = relu_active_units(net, t_test_t)
    return out


SCALARS = ("rel_l2", "linf", "rel_l2_after_adam", "loss_after_adam", "affine_dev_after_adam",
           "affine_fit_after_adam", "y_theta_0", "n_params", "relu_active_units")


def save_run(ex: Example, tag: str, act: str, r: dict, n_adam: int) -> None:
    """Checkpoint one training (history CSV, arrays, scalars) so that --resume can skip it."""
    ex.save_history(r["hist"], f"{tag}_{act}")
    np.savez(ex.path(f"run_{tag}_{act}.npz"), t=r["t"], y_ref=r["y_ref"], y_pred=r["y_pred"],
             residual=r["residual"])
    meta = {k: r[k] for k in SCALARS if k in r}
    meta.update(train_time=r["hist"].train_time, final_loss=r["hist"].final_loss, n_adam=n_adam,
                last_iter=r["hist"].iters[-1])
    ex.path(f"run_{tag}_{act}.json").write_text(json.dumps(meta, indent=1))


def load_run(ex: Example, tag: str, act: str, n_adam: int):
    """Return a checkpointed run, or None if it does not exist (or has another budget)."""
    files = [ex.path(f"run_{tag}_{act}.{e}") for e in ("json", "npz")] + [ex.path(f"history_{tag}_{act}.csv")]
    if not all(f.exists() for f in files):
        return None
    meta = json.loads(files[0].read_text())
    if meta.get("n_adam") != n_adam:
        return None
    import pandas as pd

    arr = np.load(files[1])
    df = pd.read_csv(files[2])
    hist = SimpleNamespace(iters=df["iter"].tolist(), loss=df["loss"].tolist(),
                           switch_iter=n_adam if (df["phase"] == "lbfgs").any() else None,
                           train_time=meta["train_time"], final_loss=meta["final_loss"])
    return {"hist": hist, **{k: arr[k] for k in arr.files}, **{k: meta[k] for k in SCALARS if k in meta}}


def make_figure(ex: Example, out: dict, plateaus: dict) -> None:
    from pinnode import plotting as P

    style = {act: dict(color=P.SERIES[k], ls=P.LINESTYLES[k]) for k, act in enumerate(ACTIVATIONS)}
    fig, axes = P.figure(2, 2, aspect=0.70)
    ax = axes[0, 0]
    r0 = out[("soft", "tanh")]
    P.plot_reference(ax, r0["t"], r0["y_ref"], label="exact")
    for k, act in enumerate(ACTIVATIONS):
        r = out[("soft", act)]
        P.plot_prediction(ax, r["t"], r["y_pred"], label=LABELS[act], k=k, markevery=(25 * k, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("soft initial condition: solutions")
    ax.legend(loc="upper left", ncol=1)

    ax = axes[0, 1]
    for act in ACTIVATIONS:
        r = out[("hard", act)]
        P.plot_error(ax, r["t"][1:], (r["y_pred"] - r["y_ref"])[1:], label=LABELS[act], lw=1.0,
                     **style[act])
    ax.set_ylim(1e-9, 1e0)
    ax.set_title("hard initial condition: error")
    ax.legend(loc="upper center", ncol=2)

    for ax, tag in zip(axes[1], ("soft", "hard")):
        for act in ACTIVATIONS:
            h = out[(tag, act)]["hist"]
            ax.semilogy(h.iters, P.rolling_median(np.asarray(h.loss), 9), lw=1.1, label=LABELS[act],
                        **style[act])
        sw = out[(tag, "tanh")]["hist"].switch_iter
        if sw:
            ax.axvline(sw, color=P.COLORS["muted"], lw=0.6, zorder=1)
            ax.text(sw, 0.97, "L-BFGS $\\rightarrow$ ", transform=ax.get_xaxis_transform(),
                    ha="right", va="top", fontsize=7.5, color=P.COLORS["muted"])
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_title(f"{tag} initial condition: training loss")
    ax = axes[1, 0]
    h_aff = ax.axhline(plateaus["soft_affine_y"]["loss"], color=P.COLORS["ref"], lw=0.8,
                       ls=(0, (1, 1.5)), zorder=1, label="best affine $y$")
    ax.legend(handles=[h_aff], loc="lower right")      # activations: same styles as in (a), (b), (d)
    axes[1, 1].legend(loc="lower center", ncol=2)
    for ax in axes[1]:
        ax.set_ylim(1e-12, 1e2)
    P.label_panels(axes)
    ex.savefig(fig, "logistic_activation_study")


def main():
    ex = Example("ch04_ivp", "logistic_activation_study",
                 description="Logistic ODE: activation study, soft vs hard initial condition",
                 default_seed=1234,
                 extra_args=[(("--resume",), {"action": "store_true",
                                              "help": "reuse checkpointed trainings"}),
                             (("--only",), {"default": None,
                                            "help": "train one configuration, e.g. hard_sigmoid, save its "
                                                    "checkpoint and exit (combine runs with --resume)"})])
    n_adam, n_lbfgs = ex.pick(8000, 100), ex.pick(500, 10)
    if ex.args.only:                                   # one training in its own process
        tag, act = ex.args.only.split("_", 1)
        if not (ex.args.resume and load_run(ex, tag, act, n_adam) is not None):
            ex.log(f"training {tag} / {act}")
            r = run_one(ex, act, tag == "hard", n_adam, n_lbfgs)
            save_run(ex, tag, act, r, n_adam)
            ex.log(f"{tag} {act}: relL2={r['rel_l2']:.2e} (after Adam {r['rel_l2_after_adam']:.2e})")
        else:
            ex.log(f"checkpoint {tag} / {act} exists")
        return
    t_col_np = uniform(ex.pick(100, 20), 0.0, T)
    plateaus = affine_plateaus(t_col_np)
    ex.log(f"best affine soft loss = {plateaus['soft_affine_y']['loss']:.4e}, "
           f"best (N affine) hard loss = {plateaus['hard_affine_N']['loss']:.4e}")
    out, rows = {}, []
    preds = {"t": np.linspace(0.0, T, N_TEST), "exact": exact(np.linspace(0.0, T, N_TEST))}
    for hard in (False, True):
        tag = "hard" if hard else "soft"
        for act in ACTIVATIONS:
            r = load_run(ex, tag, act, n_adam) if ex.args.resume else None
            if r is None:
                ex.log(f"training {tag} / {act}")
                r = run_one(ex, act, hard, n_adam, n_lbfgs)
                save_run(ex, tag, act, r, n_adam)
            else:
                ex.log(f"reusing checkpoint {tag} / {act}")
            out[(tag, act)] = r
            preds[f"{tag}_{act}"] = r["y_pred"]
            row = {"formulation": tag, "activation": act, "rel_l2": r["rel_l2"], "linf": r["linf"],
                   "rel_l2_after_adam": r["rel_l2_after_adam"],
                   "loss_after_adam": r["loss_after_adam"], "final_loss": r["hist"].final_loss,
                   "abs_ic_error": abs(r["y_theta_0"] - Y0),
                   "max_abs_residual": float(np.max(np.abs(r["residual"]))),
                   "affine_dev_after_adam": r["affine_dev_after_adam"],
                   "iterations": r["hist"].iters[-1], "train_time_s": r["hist"].train_time}
            if "relu_active_units" in r:
                row["relu_active_units"] = "/".join(map(str, r["relu_active_units"]))
            if tag == "soft":                  # iterations spent near the affine plateau
                L_aff = plateaus["soft_affine_y"]["loss"]
                it, ls = np.asarray(r["hist"].iters), np.asarray(r["hist"].loss)
                inside = it[(ls > 0.94 * L_aff) & (ls < 1.08 * L_aff)]
                row["plateau_first_iter"] = int(inside.min()) if inside.size else -1
                row["plateau_last_iter"] = int(inside.max()) if inside.size else -1
            rows.append(row)
    ex.save_table(rows, "summary")
    ex.save_table(preds, "predictions")
    ex.save_results({"problem": {"r": R, "y0": Y0, "T": T, "exact": "y0 e^{rt}/(1+y0(e^{rt}-1))"},
                     "hyperparameters": {"width": WIDTH, "depth": DEPTH,
                                         "n_params": out[("soft", "tanh")]["n_params"],
                                         "collocation": ex.pick(100, 20),
                                         "adam_iters": n_adam, "lr": "1e-3 -> 1e-5 (cosine)",
                                         "lbfgs_iters_max": n_lbfgs, "w_ic": W_IC,
                                         "test_grid": N_TEST, "init": "Xavier normal, zero bias",
                                         "input_scaling": "[0,T] -> [-1,1]"},
                     "plateaus": plateaus,
                     "affine_fit_after_adam": {f"{k[0]}_{k[1]}": v["affine_fit_after_adam"]
                                               for k, v in out.items()},
                     "summary": rows})
    for row in rows:
        ex.log(f"{row['formulation']:4s} {row['activation']:8s} relL2={row['rel_l2']:.2e} "
               f"(after Adam {row['rel_l2_after_adam']:.2e}, loss {row['loss_after_adam']:.2e}) "
               f"linf={row['linf']:.2e} loss={row['final_loss']:.2e} time={row['train_time_s']:.1f}s")
    if not ex.args.no_figures:
        make_figure(ex, out, plateaus)


if __name__ == "__main__":
    main()
