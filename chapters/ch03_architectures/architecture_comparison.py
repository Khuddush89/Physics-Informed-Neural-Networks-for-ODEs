"""Architecture comparison on the Airy oscillator BVP: MLP, ChNN, LeNN, Fourier features, SIREN.

Book: Chapter 3, Section "Numerical comparison" (arch:sec:numerics), Figure arch:fig:architectures,
Table arch:tab:architectures.

Problem   y'' + t y = 0 on [0, 10],  y(0) = Ai(0),  y(10) = Ai(-10).
Exact     y(t) = Ai(-t)  (a chirp with six zeros in (0, 10); local frequency sqrt(t)).

Every network has the same body (two hidden layers of 32 tanh units, or 32 sine units for the
SIREN) and differs only in the encoding of the scaled input xi = t/5 - 1:

* MLP    identity encoding                                    (1153 parameters)
* ChNN   Chebyshev features T_0..T_10                         (1473)
* LeNN   Legendre features P_0..P_10                          (1473)
* FF     16 random Fourier features, B_k ~ N(0, 1), frozen    (2145)
* SIREN  sine activations, first-layer factor w0 = 6 and w0 = 30 (the value proposed for
         images by Sitzmann et al.); w0 = 6 is comparable to the frequency scale 2 pi sigma
         of the Fourier features                              (1153)

All share the hard Dirichlet trial function  y = g(t) + (1 - xi^2) N(t)  (g linear), the same
200 equispaced collocation points, the same optimizer (Adam, learning rate 1e-3 -> 1e-5 with
cosine decay, then at most 1000 L-BFGS iterations) and the same seed for the initialization.
Errors are measured on 1001 equispaced test points against the exact solution.

Run:  PYTHONPATH=. python chapters/ch03_architectures/architecture_comparison.py         (~4 min on 1 core)
      PYTHONPATH=. python chapters/ch03_architectures/architecture_comparison.py --quick
"""
from __future__ import annotations

import numpy as np
import torch

from _ch03_common import (N_COLLOCATION, N_TEST, T_END, Y_A, Y_B, collocation, dirichlet_normalized,
                         load_history, residual, test_grid, well_posedness)
from pinnode import (ChebyshevNet, ConstrainedModel, Example, FourierFeatureNet, LegendreNet, MLP,
                     SirenNet, count_parameters, linf, rel_l2, to_tensor, train)
from pinnode import plotting as P

WIDTH, DEPTH, DEGREE, N_FF, SIGMA_FF = 32, 2, 10, 16, 1.0
SIREN_W0 = {"SIREN6": 6.0, "SIREN30": 30.0}
ARCHS = ["MLP", "ChNN", "LeNN", "FF", "SIREN6", "SIREN30"]
LABELS = {"MLP": "MLP", "ChNN": "ChNN ($m=10$)", "LeNN": "LeNN ($m=10$)",
          "FF": "Fourier features", "SIREN6": "SIREN ($\\omega_0=6$)", "SIREN30": "SIREN ($\\omega_0=30$)"}


def build(name: str) -> torch.nn.Module:
    dom = (0.0, T_END)
    if name == "MLP":
        return MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=dom)
    if name == "ChNN":
        return ChebyshevNet(degree=DEGREE, hidden=(WIDTH,) * DEPTH, activation="tanh", domain=dom)
    if name == "LeNN":
        return LegendreNet(degree=DEGREE, hidden=(WIDTH,) * DEPTH, activation="tanh", domain=dom)
    if name == "FF":
        return FourierFeatureNet(1, 1, n_features=N_FF, sigma=SIGMA_FF, width=WIDTH, depth=DEPTH,
                                 activation="tanh", domain=dom)
    if name in SIREN_W0:
        return SirenNet(1, 1, width=WIDTH, depth=DEPTH, w0_first=SIREN_W0[name], w0=1.0, domain=dom)
    raise ValueError(name)


def l2_norm(v: np.ndarray, t: np.ndarray) -> float:
    """Continuous L2 norm on [0, T] by the trapezoidal rule on the test grid."""
    return float(np.sqrt(np.trapezoid(np.asarray(v) ** 2, t)))


def run_one(ex: Example, name: str, n_adam: int, n_lbfgs: int, lam_min: float) -> dict:
    torch.manual_seed(ex.seed)
    model = ConstrainedModel(build(name), dirichlet_normalized())
    t_col = to_tensor(collocation(), requires_grad=True)
    t_test, y_ref = test_grid()
    t_test_t = to_tensor(t_test)

    def loss_fn():
        r = residual(model, t_col)
        L_r = (r ** 2).mean()
        return L_r, {"residual": L_r}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=100, print_every=2000)
    t_req = to_tensor(t_test, requires_grad=True)
    res = residual(model, t_req).detach().numpy().ravel()
    with torch.no_grad():
        y_pred = model(t_test_t).numpy().ravel()
    lbfgs_done = (hist.iters[-1] - n_adam) if n_lbfgs > 0 else 0
    # a posteriori bound: e = y_pred - y vanishes at t = 0, T and e'' + t e = R, so
    # ||e||_L2 <= ||R||_L2 / min|eig| (self-adjoint operator with Dirichlet conditions)
    res_l2, err_l2 = l2_norm(res, t_test), l2_norm(y_pred - y_ref, t_test)
    return {"hist": hist, "t": t_test, "y_ref": y_ref, "y_pred": y_pred, "residual_test": res,
            "row": {"architecture": name, "parameters": count_parameters(model),
                    "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
                    "final_loss": hist.final_loss,
                    "max_abs_residual_test": float(np.max(np.abs(res))),
                    "residual_l2_test": res_l2, "error_l2_test": err_l2,
                    "error_bound_l2": res_l2 / lam_min,
                    "adam_iters": n_adam, "lbfgs_iters_done": int(lbfgs_done),
                    "train_time_s": hist.train_time,
                    "time_per_adam_iter_ms": adam_ms_per_iter(hist, n_adam)}}


def adam_ms_per_iter(hist, n_adam: int) -> float:
    """Wall-clock milliseconds per Adam iteration (time stamp of the last Adam record)."""
    idx = [i for i, p in enumerate(hist.phase) if p == "adam"]
    return 1e3 * hist.time[idx[-1]] / max(n_adam, 1) if idx else float("nan")


def main():
    ex = Example("ch03_architectures", "architecture_comparison",
                 description="Airy BVP: MLP vs ChNN vs LeNN vs Fourier features vs SIREN (hard Dirichlet)",
                 default_seed=1234,
                 extra_args=[(("--adam",), {"type": int, "default": None,
                                            "help": "override the number of Adam iterations"}),
                             (("--lbfgs",), {"type": int, "default": None,
                                             "help": "override the maximum number of L-BFGS iterations"}),
                             (("--archs",), {"type": str, "default": ",".join(ARCHS),
                                             "help": "comma-separated subset of " + ",".join(ARCHS)}),
                             (("--plot-only",), {"action": "store_true",
                                                 "help": "redraw the figure from saved outputs"})])
    n_adam = ex.args.adam if ex.args.adam is not None else ex.pick(full=4000, quick=60)
    n_lbfgs = ex.args.lbfgs if ex.args.lbfgs is not None else ex.pick(full=1000, quick=10)
    archs = [a for a in ex.args.archs.split(",") if a]
    if ex.args.plot_only:
        import json

        import pandas as pd
        res = json.loads(ex.path("results.json").read_text())
        n_adam = res["hyperparameters"]["adam_iters"]
        archs = [r["architecture"] for r in res["summary"]]
        pred = pd.read_csv(ex.path("predictions.csv"))
        out = {n: {"t": pred["t"].to_numpy(), "y_ref": pred["exact"].to_numpy(),
                   "y_pred": pred[f"pred_{n}"].to_numpy(),
                   "hist": load_history(ex.path(f"history_{n}.csv"), switch_iter=n_adam)}
               for n in archs}
        make_figure(ex, out, archs, n_adam)
        return
    wp = well_posedness()
    ex.log(f"well-posedness: phi(T)={wp['phi_T']:.4f}, min|eig|={wp['min_abs_eig_dirichlet']:.4f}")

    out, rows = {}, []
    for name in archs:
        ex.log(f"training {name}")
        r = run_one(ex, name, n_adam, n_lbfgs, wp["min_abs_eig_dirichlet"])
        out[name] = r
        rows.append(r["row"])
        ex.save_history(r["hist"], name)
        ex.log(f"{name:6s} P={r['row']['parameters']:5d} relL2={r['row']['rel_l2']:.3e} "
               f"linf={r['row']['linf']:.3e} loss={r['row']['final_loss']:.3e} "
               f"errL2={r['row']['error_l2_test']:.2e} bound={r['row']['error_bound_l2']:.2e} "
               f"time={r['row']['train_time_s']:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_table({"t": out[archs[0]]["t"], "exact": out[archs[0]]["y_ref"],
                   **{f"pred_{n}": out[n]["y_pred"] for n in archs}}, "predictions")
    ex.save_results({
        "problem": {"equation": "y'' + t y = 0", "interval": [0.0, T_END], "y(0)": Y_A,
                    "y(T)": Y_B, "exact": "Ai(-t)", "well_posedness": wp},
        "hyperparameters": {"width": WIDTH, "hidden_layers": DEPTH, "activation": "tanh (SIREN: sin)",
                            "chebyshev_legendre_degree": DEGREE, "fourier_features": N_FF,
                            "fourier_sigma": SIGMA_FF, "siren_w0_first": SIREN_W0,
                            "collocation": N_COLLOCATION, "collocation_spacing": "uniform",
                            "test_points": N_TEST, "adam_iters": n_adam,
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters_max": n_lbfgs,
                            "constraint": "hard Dirichlet, y = g + (1 - xi^2) N",
                            "init": "Glorot normal (SIREN: Sitzmann et al.)"},
        "summary": rows,
    })

    if ex.args.no_figures:
        return
    make_figure(ex, out, archs, n_adam)


def make_figure(ex: Example, out: dict, archs: list, n_adam: int) -> None:
    """2x2 figure; one style (colour, line style, marker) per network in every panel."""
    def style(k, every):
        return {"color": P.SERIES[k % 6], "ls": P.LINESTYLES[k % 6], "marker": P.MARKERS[k % 6],
                "markevery": every, "markersize": 3.0, "markerfacecolor": "white",
                "markeredgewidth": 0.8, "lw": 1.2}

    fig, axes = P.figure(2, 2, aspect=0.70)
    ax = axes[0, 0]
    t = out[archs[0]]["t"]
    P.plot_reference(ax, t, out[archs[0]]["y_ref"], label="exact $\\mathrm{Ai}(-t)$")
    for k, name in enumerate(archs):
        ax.plot(t, out[name]["y_pred"], label=LABELS[name], zorder=3, **style(k, (20 * k, 125)))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("solution")
    ax = axes[0, 1]
    for k, name in enumerate(archs):
        err = np.maximum(np.abs(out[name]["y_pred"] - out[name]["y_ref"]), 1e-13)
        ax.semilogy(t, err, **style(k, (20 * k, 125)))
    ax.set_ylim(1e-8, 2.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel("absolute error")
    ax.set_title("pointwise error")
    ax = axes[1, 0]
    for k, name in enumerate(archs):
        h = out[name]["hist"]
        ax.semilogy(h.eval_iters, h.metrics["rel_l2"], **style(k, (2 * k, 8)))
    ax.set_ylabel("relative $L^2$ error")
    ax.set_title("relative $L^2$ error")
    ax = axes[1, 1]
    for k, name in enumerate(archs):
        h = out[name]["hist"]
        ax.semilogy(h.iters, P.rolling_median(h.loss, 9), **style(k, (15 * k, 90)))
    ax.set_ylabel("loss")
    ax.set_title("training loss")
    for ax in axes[1]:
        ax.axvline(n_adam, color=P.COLORS["muted"], lw=0.6, zorder=1)
        ax.set_xlabel("iteration")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=7)
    P.label_panels(axes)
    ex.savefig(fig, "architecture_comparison")


if __name__ == "__main__":
    main()
