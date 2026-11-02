"""Imbalance between residual and boundary terms:  y'' = -omega^2 sin(omega t),  y(0)=0, y(1)=1.

Book: Chapter 9, Section "Imbalance between loss terms".

Exact solution y(t) = sin(omega t) + t with omega = 6 pi. The residual term carries the factor
omega^2 = 355 (and its gradient roughly omega^4), whereas the boundary term is O(1); the
residual alone determines y only up to a + b t, so an under-weighted boundary term shows up
as an error that is nearly linear in t.

Variants (tanh MLP 32 x 3, M = 128 uniform collocation points, Adam 5000 then L-BFGS 500):

* fixed   - L = L_r + L_b (lambda_b = 1);
* scaled  - residual divided by omega^2 (nondimensionalized equation), lambda_b = 1;
* lra     - learning-rate annealing (Wang, Teng & Perdikaris 2021): every 10 Adam steps
            lambda_hat = max|grad L_r| / mean|grad L_b|, lambda_b <- 0.1 lambda_b + 0.9 lambda_hat;
* sa      - self-adaptive weights (McClenny & Braga-Neto 2023): one weight per residual and
            boundary point, mask m(l) = l^2, gradient ascent (Adam, lr 5e-3) on the weights;
* hard    - boundary conditions built into y = t + t (1 - t) N(t);
* hard_scaled - the same trial function with the output scaled by omega,
            y = t + omega t (1 - t) N(t), so that N is of order one.

The weights are frozen during L-BFGS. Per-term gradient norms ||grad L_r|| and
||grad lambda_b L_b|| are recorded every 50 iterations.

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/loss_imbalance.py          (~4 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/loss_imbalance.py --quick  (smoke test)
      ... --omega-over-pi 4 --outdir outputs/ch09_training_diagnostics/loss_imbalance_4pi   (other omega)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CPUTimer, RunCache, grad_norm, grad_vector, latex_table, num, replot_arg  # noqa: E402
from pinnode import (ConstrainedModel, Example, MLP, derivative, dirichlet_transform,  # noqa: E402
                     linf, rel_l2, to_tensor, train)
from pinnode import plotting as P  # noqa: E402

OMEGA = 6.0 * math.pi
VARIANTS = ["fixed", "scaled", "lra", "sa", "hard", "hard_scaled"]
LABELS = {"fixed": r"fixed, $\lambda_b=1$", "scaled": r"residual $/\omega^2$",
          "lra": "learning-rate annealing", "sa": "self-adaptive", "hard": r"hard BC, $t(1-t)N$",
          "hard_scaled": r"hard BC, $\omega t(1-t)N$"}
LRA_ALPHA, LRA_EVERY = 0.9, 10
SA_LR = 5e-3


def exact(t):
    return np.sin(OMEGA * t) + t


def run(ex, variant, M, n_adam, n_lbfgs, grad_every):
    torch.manual_seed(ex.seed)
    net = MLP(1, 1, 32, 3, "tanh", domain=(0.0, 1.0))
    if variant == "hard":
        model = ConstrainedModel(net, dirichlet_transform(0.0, 0.0, 1.0, 1.0))
    elif variant == "hard_scaled":
        model = ConstrainedModel(net, lambda t, n: t + OMEGA * t * (1.0 - t) * n)
    else:
        model = net
    params = list(net.parameters())
    t_col = to_tensor(np.linspace(0.0, 1.0, M), requires_grad=True)
    t_b, y_b = to_tensor([0.0, 1.0]), to_tensor([0.0, 1.0])
    f_col = -OMEGA ** 2 * torch.sin(OMEGA * t_col.detach())
    scale = OMEGA ** 2 if variant == "scaled" else 1.0
    t_test = np.linspace(0.0, 1.0, 1001)
    y_ref = exact(t_test)
    t_test_t = to_tensor(t_test)
    state = {"lam_b": 1.0}
    lam_r = torch.ones(M, 1, requires_grad=True)     # self-adaptive weights (variant "sa")
    lam_bp = torch.ones(2, 1, requires_grad=True)
    diag = {"iter": [], "grad_r": [], "grad_b": [], "lam_b": []}

    def terms():
        y = model(t_col)
        r = (derivative(y, t_col, 2) - f_col) / scale
        return r, model(t_b) - y_b

    def loss_fn():
        r, b = terms()
        L_r, L_b = (r ** 2).mean(), (b ** 2).mean()
        if variant in ("hard", "hard_scaled"):
            return L_r, {"residual": L_r}
        if variant == "sa":
            L = ((lam_r ** 2) * r ** 2).mean() + ((lam_bp ** 2) * b ** 2).mean()
            return L, {"residual": L_r, "boundary": L_b}
        return L_r + state["lam_b"] * L_b, {"residual": L_r, "boundary": L_b}

    sa_opt = torch.optim.Adam([lam_r, lam_bp], lr=SA_LR, maximize=True) if variant == "sa" else None

    def weight_update(k):
        if variant == "sa" and k > 0:          # ascent step with the gradients of the last loss
            sa_opt.step()
            sa_opt.zero_grad(set_to_none=True)
        need_lra = variant == "lra" and k % LRA_EVERY == 0
        need_diag = k % grad_every == 0
        if not (need_lra or need_diag):
            return
        r, b = terms()
        L_r, L_b = (r ** 2).mean(), (b ** 2).mean()
        if need_lra:
            g_r = grad_vector(L_r, params).abs()
            g_b = grad_vector(L_b, params).abs()
            lam_hat = float(g_r.max() / g_b.mean())
            state["lam_b"] = (1.0 - LRA_ALPHA) * state["lam_b"] + LRA_ALPHA * lam_hat
        if need_diag:
            diag["iter"].append(k)
            diag["grad_r"].append(grad_norm(L_r, params))
            if variant == "sa":
                eff_b = float((lam_bp.detach() ** 2).mean())
            else:
                eff_b = state["lam_b"]
            hard = variant in ("hard", "hard_scaled")
            diag["grad_b"].append(grad_norm(eff_b * L_b, params) if not hard else float("nan"))
            diag["lam_b"].append(eff_b if not hard else float("nan"))

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(params, loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5, lbfgs_iters=0,
                 eval_fn=eval_fn, eval_every=50, weight_update=weight_update, print_every=0,
                 verbose=False)
    with torch.no_grad():
        y_adam = model(t_test_t).numpy().ravel()
    if variant == "sa":
        lam_r.requires_grad_(False)
        lam_bp.requires_grad_(False)
    hist2 = train(params, loss_fn, adam_iters=0, lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50,
                  print_every=0, verbose=False)
    with torch.no_grad():
        y_final = model(t_test_t).numpy().ravel()
        bnd = (model(t_b) - y_b).numpy().ravel()
    return {"hist": hist, "hist_lbfgs": hist2, "diag": diag, "y_adam": y_adam, "y_final": y_final,
            "rel_l2_adam": rel_l2(y_adam, y_ref), "rel_l2": rel_l2(y_final, y_ref),
            "linf": linf(y_final, y_ref), "bc_error_adam": [float(abs(y_adam[0])), float(abs(y_adam[-1] - 1.0))],
            "bc_error": [float(abs(bnd[0])), float(abs(bnd[1]))], "lam_b_final": diag["lam_b"][-1] if diag["lam_b"] else 1.0,
            "sa_weights": {"boundary_mean": float((lam_bp.detach() ** 2).mean()),
                           "residual_mean": float((lam_r.detach() ** 2).mean())},
            "train_time_s": hist.train_time + hist2.train_time}


def main():
    global OMEGA
    ex = Example("ch09_training_diagnostics", "loss_imbalance",
                 description="BVP y''=f with large omega: fixed vs rescaled vs annealed vs self-adaptive vs hard BC",
                 default_seed=1234,
                 extra_args=[replot_arg(), (["--omega-over-pi"], {"type": float, "default": 6.0,
                                                    "help": "omega / pi (default 6)"})])
    OMEGA = ex.args.omega_over_pi * math.pi
    M = ex.pick(128, 32)
    n_adam, n_lbfgs = ex.pick(5000, 40), ex.pick(500, 10)
    grad_every = ex.pick(50, 10)
    t_test = np.linspace(0.0, 1.0, 1001)
    y_ref = exact(t_test)
    runs, rows = {}, []
    cache = RunCache(ex)
    for v in VARIANTS:
        def job(v=v):
            with CPUTimer() as tm:
                res = run(ex, v, M, n_adam, n_lbfgs, grad_every)
            res["cpu_time_s"] = tm.cpu
            return res
        res = cache.get(v, job)
        runs[v] = res
        ex.save_history(res["hist"], f"{v}_adam")
        ex.save_history(res["hist_lbfgs"], f"{v}_lbfgs")
        ex.save_table(res["diag"], f"gradnorms_{v}")
        d = res["diag"]
        ratio0 = (d["grad_r"][0] / d["grad_b"][0] if v not in ("hard", "hard_scaled") and d["grad_b"][0] > 0
                  else float("nan"))
        rows.append({"variant": v, "rel_l2_after_adam": res["rel_l2_adam"], "rel_l2": res["rel_l2"],
                     "linf": res["linf"], "bc_error_adam_t0": res["bc_error_adam"][0],
                     "bc_error_adam_t1": res["bc_error_adam"][1], "bc_error_t0": res["bc_error"][0],
                     "bc_error_t1": res["bc_error"][1], "lam_b_final": res["lam_b_final"],
                     "grad_ratio_init": ratio0, "sa_weights": res["sa_weights"],
                     "train_time_s": res["train_time_s"], "cpu_time_s": res["cpu_time_s"]})
        ex.log(f"{v:6s}: relL2 Adam={res['rel_l2_adam']:.2e} final={res['rel_l2']:.2e} "
               f"BC err (Adam)={res['bc_error_adam']} lam_b={res['lam_b_final']:.2e} ratio0={ratio0:.1e} "
               f"cpu={res['cpu_time_s']:.0f}s")
    ex.save_table(rows, "summary")
    ex.save_table({"t": t_test, "exact": y_ref, **{f"{v}_adam": runs[v]["y_adam"] for v in VARIANTS},
                   **{f"{v}_final": runs[v]["y_final"] for v in VARIANTS}}, "predictions")
    ex.save_results({
        "problem": {"ode": "y'' = -omega^2 sin(omega t), y(0)=0, y(1)=1", "omega": OMEGA,
                    "exact": "sin(omega t) + t"},
        "hyperparameters": {"width": 32, "depth": 3, "collocation": M, "adam_iters": n_adam,
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": n_lbfgs,
                            "lra": {"alpha": LRA_ALPHA, "every": LRA_EVERY},
                            "sa": {"mask": "lambda^2", "lr": SA_LR, "init": 1.0},
                            "weights_frozen_during_lbfgs": True},
        "summary": rows})
    cache.clear()

    tab = [[LABELS[r["variant"]], num(r["rel_l2_after_adam"]), num(r["rel_l2"]),
            ("exact" if r["variant"].startswith("hard") else num(max(r["bc_error_adam_t0"], r["bc_error_adam_t1"]))),
            (num(r["lam_b_final"]) if r["variant"] in ("lra", "sa")
             else ("--" if r["variant"].startswith("hard") else "1"))]
           for r in rows]
    latex_table(ex.path("tab_loss_imbalance.tex"), "lrrrr",
                ["loss", r"rel.\ $L^2$ (Adam)", r"rel.\ $L^2$ (+L-BFGS)", r"BC error (Adam)",
                 r"final $\lambda_b$"], tab, comment="Chapter 9, loss_imbalance.py")

    if ex.args.no_figures:
        return
    fig, axes = P.figure(2, 2, aspect=0.74)
    ax = axes[0, 0]
    for k, v in enumerate(["fixed", "lra"]):
        d = runs[v]["diag"]
        c = P.SERIES[0 if v == "fixed" else 2]
        ax.semilogy(d["iter"], d["grad_r"], color=c, ls="-", lw=1.1, label=rf"$\|\nabla L_r\|$, {LABELS[v]}")
        ax.semilogy(d["iter"], d["grad_b"], color=c, ls="--", lw=1.1,
                    label=rf"$\|\nabla \lambda_b L_b\|$, {LABELS[v]}")
    ax.set_xlabel("iteration (Adam)")
    ax.set_ylabel("gradient norm")
    ax.set_title("per-term gradient norms")
    ax.set_ylim(bottom=1e-4)
    ax.legend(loc="lower right", fontsize=6)

    ax = axes[0, 1]
    for k, v in enumerate(["lra", "sa"]):
        d = runs[v]["diag"]
        ax.semilogy(d["iter"], d["lam_b"], color=P.SERIES[2 if v == "lra" else 3], ls=P.LINESTYLES[k],
                    lw=1.2, label=LABELS[v] + (r" ($\lambda_b$)" if v == "lra" else r" (mean $m(\lambda_{b,j})$)"))
    ax.set_xlabel("iteration (Adam)")
    ax.set_ylabel("boundary weight")
    ax.set_title("adaptive boundary weights")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.3), fontsize=6.5)

    ax = axes[1, 0]
    for k, v in enumerate(VARIANTS):
        h, h2 = runs[v]["hist"], runs[v]["hist_lbfgs"]
        it = list(h.eval_iters) + [n_adam + i for i in h2.eval_iters]
        val = list(h.metrics["rel_l2"]) + list(h2.metrics["rel_l2"])
        ax.semilogy(it, val, color=P.SERIES[k], ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.2,
                    label=LABELS[v])
    ax.axvline(n_adam, color=P.COLORS["muted"], lw=0.6)
    ax.text(n_adam, 0.97, " L-BFGS", transform=ax.get_xaxis_transform(), ha="left", va="top",
            fontsize=7, color=P.COLORS["muted"])
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("training error")

    ax = axes[1, 1]
    for k, v in enumerate(VARIANTS):
        P.plot_error(ax, t_test, runs[v]["y_adam"] - y_ref, k=k)
    ax.set_ylim(bottom=1e-6)
    ax.set_title("pointwise error after Adam")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=6.5)
    P.label_panels(axes)
    ex.savefig(fig, "loss_imbalance_summary")


if __name__ == "__main__":
    main()
