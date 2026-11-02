"""Diagnostics: validation residual versus training loss, and residual-based adaptive refinement.

Book: Chapter 9, Section "Diagnostics and a troubleshooting guide".

Part A (a small training loss does not certify a small error).
    y' = 2 pi cos(2 pi t) + 0.8 pi cos(8 pi t), y(0) = 0 on [0, 1]; exact y = sin(2 pi t) + 0.1 sin(8 pi t).
    Only M = 24 uniform collocation points. A tanh MLP (32 x 3) is compared with a Fourier-feature
    network whose frequencies are far too high (sigma = 20). The training loss (mean squared
    residual at the 24 points) is compared with the validation residual on 1001 fresh points.

Part B (residual-based adaptive refinement, RAR, Lu et al. 2021).
    The stiff problem y' = -lambda (y - cos t), y(0) = 0 on [0, 2 pi] with lambda = 100 (hard
    initial condition, residual scaled by 1/lambda, tanh MLP 32 x 3). RAR starts from 100 uniform
    points and adds, in 4 rounds, the 25 points of a 4000-point candidate grid with the largest
    |residual| (200 points in total); it is compared with 100 and with 200 uniform points.
    Equal optimizer budget: 5000 Adam steps + 300 L-BFGS.

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/diagnostics.py          (~1 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/diagnostics.py --quick  (smoke test)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CPUTimer, RunCache, latex_table, num, replot_arg  # noqa: E402
from pinnode import (ConstrainedModel, Example, FourierFeatureNet, MLP, derivative, linf,  # noqa: E402
                     rel_l2, to_tensor, train)
from pinnode import plotting as P  # noqa: E402

# ----------------------------------------------------------------------------- Part A
KA = 4


def exact_a(t):
    return np.sin(2 * np.pi * t) + 0.1 * np.sin(2 * np.pi * KA * t)


def rhs_a(t):
    return 2 * np.pi * torch.cos(2 * np.pi * t) + 0.1 * 2 * np.pi * KA * torch.cos(2 * np.pi * KA * t)


def part_a(ex, n_adam, n_lbfgs):
    M = 24
    t_col = to_tensor(np.linspace(0.0, 1.0, M), requires_grad=True)
    t_val_np = np.linspace(0.0, 1.0, 1001)
    t_val = to_tensor(t_val_np, requires_grad=True)
    y_ref = exact_a(t_val_np)
    out = {}
    for key in ("mlp", "ff20"):
        torch.manual_seed(ex.seed)
        if key == "mlp":
            net = MLP(1, 1, 32, 3, "tanh", domain=(0.0, 1.0))
        else:
            net = FourierFeatureNet(1, 1, n_features=32, sigma=20.0, width=32, depth=3, domain=(0.0, 1.0))
        model = ConstrainedModel(net, lambda t, n: t * n)
        curve = {"iter": [], "train_loss": [], "val_residual": [], "rel_l2": []}

        def loss_fn():
            r = derivative(model(t_col), t_col) - rhs_a(t_col.detach())
            L = (r ** 2).mean()
            return L, {"residual": L}

        def eval_fn():
            y = model(t_val)
            rv = derivative(y, t_val) - rhs_a(t_val.detach())
            L, _ = loss_fn()
            e = rel_l2(y.detach().numpy().ravel(), y_ref)
            curve["train_loss"].append(float(L.detach()))
            curve["val_residual"].append(float((rv ** 2).mean().detach()))
            curve["rel_l2"].append(e)
            return {"rel_l2": e, "val_residual": curve["val_residual"][-1]}

        with CPUTimer() as tm:
            h = train(net.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                      lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50, print_every=0, verbose=False)
        curve["iter"] = list(h.eval_iters)
        n = min(len(curve["iter"]), len(curve["rel_l2"]))
        curve = {k: v[:n] for k, v in curve.items()}
        y_pred = model(to_tensor(t_val_np)).detach().numpy().ravel()
        out[key] = {"curve": curve, "y_pred": y_pred, "rel_l2": rel_l2(y_pred, y_ref),
                    "linf": linf(y_pred, y_ref), "train_loss": curve["train_loss"][-1],
                    "val_residual": curve["val_residual"][-1], "cpu_time_s": tm.cpu,
                    "train_time_s": h.train_time, "t_col": t_col.detach().numpy().ravel()}
        ex.log(f"Part A {key}: train loss={out[key]['train_loss']:.2e} val residual={out[key]['val_residual']:.2e} "
               f"relL2={out[key]['rel_l2']:.2e}")
    return out, t_val_np, y_ref


# ----------------------------------------------------------------------------- Part B
LAM, TB = 100.0, 2.0 * math.pi


def exact_b(t):
    A, B = LAM ** 2 / (1 + LAM ** 2), LAM / (1 + LAM ** 2)
    return A * np.cos(t) + B * np.sin(t) - A * np.exp(-LAM * t)


def part_b(ex, n_adam, n_lbfgs, rounds, add_per_round):
    t_cand = np.linspace(0.0, TB, 4000)
    t_test = np.unique(np.concatenate([np.linspace(0.0, TB, 1001), np.geomspace(1e-3 / LAM, 10.0 / LAM, 200)]))
    t_fine = np.unique(np.concatenate([np.linspace(0.0, TB, 2001), np.geomspace(1e-4, TB, 400)]))
    y_ref = exact_b(t_test)
    out = {}
    for key in ("uniform100", "rar200", "uniform200"):
        torch.manual_seed(ex.seed)
        net = MLP(1, 1, 32, 3, "tanh", domain=(0.0, TB))
        model = ConstrainedModel(net, lambda t, n: t * n)
        pts = np.linspace(0.0, TB, 200 if key == "uniform200" else 100)
        n_rounds = rounds if key == "rar200" else 0
        hists, history_pts = [], [pts.copy()]

        def scaled_residual(t):
            y = model(t)
            return (derivative(y, t) + LAM * (y - torch.cos(t))) / LAM

        with CPUTimer() as tm:
            for rd in range(n_rounds + 1):
                t_col = to_tensor(np.sort(pts), requires_grad=True)

                def loss_fn(t_col=t_col):
                    r = scaled_residual(t_col)
                    L = (r ** 2).mean()
                    return L, {"residual": L}

                last = rd == n_rounds
                n_ad = n_adam // (n_rounds + 1)
                hists.append(train(net.parameters(), loss_fn, adam_iters=n_ad, lr=1e-3, lr_final=1e-5,
                                   lbfgs_iters=n_lbfgs if last else 0, print_every=0, verbose=False))
                if not last:
                    tc = to_tensor(t_cand, requires_grad=True)
                    rc = np.abs(scaled_residual(tc).detach().numpy().ravel())
                    rc[np.isin(t_cand, pts)] = -1.0
                    new = t_cand[np.argsort(rc)[::-1][:add_per_round]]
                    pts = np.concatenate([pts, new])
                    history_pts.append(pts.copy())
        tf = to_tensor(t_fine, requires_grad=True)
        res_fine = scaled_residual(tf).detach().numpy().ravel()
        y_pred = model(to_tensor(t_test)).detach().numpy().ravel()
        tu = np.linspace(0.0, TB, 1001)
        out[key] = {"points": np.sort(pts), "history_points": history_pts, "res_fine": res_fine,
                    "y_pred": y_pred, "linf": linf(y_pred, y_ref),
                    "rel_l2": rel_l2(model(to_tensor(tu)).detach().numpy().ravel(), exact_b(tu)),
                    "max_abs_residual_fine": float(np.max(np.abs(res_fine))), "cpu_time_s": tm.cpu,
                    "train_time_s": float(sum(h.train_time for h in hists)), "hists": hists}
        ex.log(f"Part B {key}: points={len(pts)} linf={out[key]['linf']:.2e} relL2={out[key]['rel_l2']:.2e} "
               f"max|r| fine={out[key]['max_abs_residual_fine']:.2e}")
    return out, t_test, y_ref, t_fine


def main():
    ex = Example("ch09_training_diagnostics", "diagnostics",
                 description="Validation residual vs training loss; residual-based adaptive refinement",
                 default_seed=1234,
                 extra_args=[replot_arg()])
    n_adam, n_lbfgs = ex.pick(4000, 40), ex.pick(300, 10)
    cache = RunCache(ex)
    A, t_val, y_ref_a = cache.get("partA", lambda: part_a(ex, n_adam, n_lbfgs))
    B, t_test_b, y_ref_b, t_fine = cache.get("partB", lambda: part_b(ex, ex.pick(5000, 50), n_lbfgs, rounds=4,
                                                                      add_per_round=25))
    for k, v in A.items():
        ex.save_table(v["curve"], f"partA_curve_{k}")
    for k, v in B.items():
        ex.save_table({"t": v["points"]}, f"partB_points_{k}")
        for j, h in enumerate(v["hists"]):
            ex.save_history(h, f"partB_{k}_round{j}")
    rows_a = [{"network": k, "collocation": 24, "train_loss": v["train_loss"], "val_residual": v["val_residual"],
               "rel_l2": v["rel_l2"], "linf": v["linf"], "cpu_time_s": v["cpu_time_s"],
               "train_time_s": v["train_time_s"]} for k, v in A.items()]
    rows_b = [{"sampling": k, "points": int(len(v["points"])), "linf_layer_grid": v["linf"], "rel_l2": v["rel_l2"],
               "max_abs_scaled_residual_fine": v["max_abs_residual_fine"],
               "points_in_layer_t_lt_0.05": int(np.sum(v["points"] < 0.05)),
               "cpu_time_s": v["cpu_time_s"], "train_time_s": v["train_time_s"]} for k, v in B.items()]
    ex.save_table(rows_a, "summary_partA")
    ex.save_table(rows_b, "summary_partB")
    ex.save_results({
        "partA": {"problem": "y' = 2 pi cos(2 pi t) + 0.8 pi cos(8 pi t), y(0)=0, [0,1]",
                  "collocation": 24, "validation_points": 1001, "fourier_sigma": 20.0,
                  "adam_iters": n_adam, "lbfgs_iters": n_lbfgs, "summary": rows_a},
        "partB": {"problem": "y' = -100 (y - cos t), y(0)=0, [0, 2 pi]", "candidates": 4000, "rounds": 4,
                  "added_per_round": 25, "initial_points_rar": 100, "adam_iters_total": ex.pick(5000, 50), "lbfgs_iters": n_lbfgs,
                  "residual": "scaled by 1/lambda", "summary": rows_b}})
    cache.clear()
    lab_a = {"mlp": r"tanh MLP", "ff20": r"FF, $\sigma=20$"}
    lab_b = {"uniform100": "uniform, 100 points", "rar200": "RAR, 100 + 4$\\times$25 points",
             "uniform200": "uniform, 200 points"}
    tab = [[lab_a[r["network"]], "24", num(r["train_loss"]), num(r["val_residual"]), num(r["rel_l2"])] for r in rows_a]
    latex_table(ex.path("tab_diagnostics_a.tex"), "lrrrr",
                ["network", "points", "training loss", "validation residual", r"rel.\ $L^2$"], tab,
                comment="Chapter 9, diagnostics.py, part A")
    tab = [[lab_b[r["sampling"]], f"{r['points']}", f"{r['points_in_layer_t_lt_0.05']}",
            num(r["max_abs_scaled_residual_fine"]), num(r["linf_layer_grid"])] for r in rows_b]
    latex_table(ex.path("tab_diagnostics_b.tex"), "lrrrr",
                ["sampling", "points", r"points in $[0,0.05)$", r"max $|r/\lambda|$ (fine grid)",
                 "max error"], tab, comment="Chapter 9, diagnostics.py, part B")

    if ex.args.no_figures:
        return
    fig, axes = P.figure(2, 2, aspect=0.74)
    ax = axes[0, 0]
    for k, key in enumerate(("mlp", "ff20")):
        c = A[key]["curve"]
        col = P.SERIES[0 if key == "mlp" else 1]
        ax.semilogy(c["iter"], c["train_loss"], color=col, ls="-", lw=1.1, label=f"{lab_a[key]}: training loss")
        ax.semilogy(c["iter"], c["val_residual"], color=col, ls="--", lw=1.1,
                    label=f"{lab_a[key]}: validation residual")
    ax.axvline(n_adam, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("mean squared residual")
    ax.set_title("24 collocation points")
    ax.legend(loc="lower left", fontsize=6)

    ax = axes[0, 1]
    for k, key in enumerate(("mlp", "ff20")):
        P.plot_error(ax, t_val, A[key]["y_pred"] - y_ref_a, label=lab_a[key], k=k)
    P.rug(ax, A["mlp"]["t_col"])
    lo = min(float(np.min(np.abs(A[k]["y_pred"] - y_ref_a)[t_val > 0.01])) for k in A)
    ax.set_ylim(bottom=max(lo / 3.0, 1e-12))
    ax.set_title("pointwise error")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    names = ("uniform100", "rar200", "uniform200")
    for k, key in enumerate(names):
        rr = np.maximum(np.abs(B[key]["res_fine"]), 1e-12)
        m = t_fine > 0
        ax.loglog(t_fine[m], rr[m], color=P.SERIES[k], ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.0,
                  label=lab_b[key])
    pts = B["rar200"]["points"]
    ax.plot(pts[pts > 0], np.full(np.sum(pts > 0), 0.02), "|", color=P.SERIES[1], ms=5, mew=0.7,
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|r(t)|/\lambda$ on a fine grid")
    ax.set_title(r"residual map, $\lambda = 100$")

    ax = axes[1, 1]
    for k, key in enumerate(names):
        e = np.abs(B[key]["y_pred"] - y_ref_b)
        m = t_test_b > 0
        P.plot_error(ax, t_test_b[m], e[m], label=lab_b[key], k=k)
    ax.set_xscale("log")
    lo = min(float(np.min(np.abs(B[k]["y_pred"] - y_ref_b)[t_test_b > 1e-4])) for k in names)
    ax.set_ylim(bottom=max(lo / 3.0, 1e-12))
    ax.set_title(r"pointwise error, $\lambda = 100$")
    if ax.get_legend() is not None:  # plot_error adds a per-axes legend; use one shared legend
        ax.get_legend().remove()
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    P.label_panels(axes)
    ex.savefig(fig, "diagnostics_summary")


if __name__ == "__main__":
    main()
