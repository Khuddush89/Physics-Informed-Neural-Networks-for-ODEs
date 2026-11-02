"""Attraction of an equilibrium: the logistic ODE  y' = r y (1 - y),  y(0) = y0  on [0, T].

Book: Chapter 9, Section "Trivial solutions and equilibria".

With r = 1 and y0 = 0.1, the residual of any function close to the unstable equilibrium
y = 0 is small, O(y). The family y_a(t) = a exp(-r t) (decay towards the equilibrium,
i.e. the backward-in-time linearized dynamics) has the soft-constraint loss

    lambda_ic (a - y0)^2 + 2 r a^2 / T     (for r T >> 1, neglecting O(a^3)),

minimized by a* = lambda_ic T y0 / (lambda_ic T + 2 r) with value
L*(T) = 2 r lambda_ic y0^2 / (lambda_ic T + 2 r); with a hard initial condition (a = y0)
the value is 2 r y0^2 / T. Both decrease with T: long intervals make the spurious solution
more attractive. Exact solution: y(t) = y0 e^{rt} / (1 + y0 (e^{rt} - 1)).

Experiments (tanh MLP 32 x 3, M = 128 collocation points, Adam 3000 + L-BFGS 300, 3 seeds):

* Part A: soft initial condition (weight 1) for T in {2, 5, 10, 20};
* Part B: T = 10 with soft weight 100, hard initial condition, a curriculum in T (4 stages),
  causal weighting (Wang et al.), and a relative residual in logarithmic variables
  (y = y0 exp(t N(t)), residual (log y)' - r (1 - y)).

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/trivial_solutions.py          (~4 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/trivial_solutions.py --quick  (smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CPUTimer, RunCache, causal_weights, latex_table, num, replot_arg  # noqa: E402
from pinnode import (ConstrainedModel, Example, MLP, derivative, linf, rel_l2,  # noqa: E402
                     to_tensor, train)
from pinnode import plotting as P  # noqa: E402

R, Y0 = 1.0, 0.1
T_SWEEP = [2.0, 5.0, 10.0, 20.0]
T_MAIN = 10.0
VARIANTS_B = ["soft100", "hard", "curriculum", "causal", "relative"]
LABELS = {"soft": r"soft IC, $\lambda_{ic}=1$", "soft100": r"soft IC, $\lambda_{ic}=100$",
          "hard": "hard IC", "curriculum": "curriculum in $T$", "causal": "causal weights",
          "relative": "relative residual"}
FAIL_TOL = 0.1   # a run "fails" if its relative L2 error exceeds this value


def exact(t, T=None):
    return Y0 * np.exp(R * t) / (1.0 + Y0 * (np.exp(R * t) - 1.0))


def spurious_loss(T, lam_ic=1.0, hard=False):
    """Loss of the best decaying exponential a exp(-r t) (leading order, r T >> 1)."""
    if hard:
        return 2.0 * R * Y0 ** 2 / T
    return 2.0 * R * lam_ic * Y0 ** 2 / (lam_ic * T + 2.0 * R)


def run(ex, variant: str, T: float, seed: int, M: int, n_adam: int, n_lbfgs: int):
    torch.manual_seed(seed)
    net = MLP(1, 1, 32, 3, "tanh", domain=(0.0, T))
    if variant in ("hard", "curriculum", "causal"):
        model = ConstrainedModel(net, lambda t, n: Y0 + t * n)
    elif variant == "relative":
        model = ConstrainedModel(net, lambda t, n: Y0 * torch.exp(t * n))
    else:
        model = net
    lam_ic = 100.0 if variant == "soft100" else 1.0
    soft = variant in ("soft", "soft100")
    t0 = to_tensor([0.0])
    t_test = np.linspace(0.0, T, 1001)
    y_ref = exact(t_test)
    t_test_t = to_tensor(t_test)

    def make_loss(t_col, causal_state=None):
        def residual():
            y = model(t_col)
            if variant == "relative":       # (log y)' - r (1 - y) = N + t N' - r (1 - y)
                n = net(t_col)
                return derivative(t_col * n, t_col) - R * (1.0 - y)
            return derivative(y, t_col) - R * y * (1.0 - y)

        def loss_fn():
            r = residual()
            L_r = (r ** 2).mean()
            parts = {"residual": L_r}
            if causal_state is not None:
                Lc = (r ** 2).reshape(causal_state["chunks"], -1).mean(dim=1)
                w = causal_weights(Lc, causal_state["eps"])
                if float(w.min()) > 0.99 and causal_state["k"] < len(causal_state["eps_list"]) - 1:
                    causal_state["k"] += 1
                    causal_state["eps"] = causal_state["eps_list"][causal_state["k"]]
                parts["w_min"] = float(w.min())
                return (w * Lc).mean(), parts
            if soft:
                L_ic = ((model(t0) - Y0) ** 2).mean()
                parts["ic"] = L_ic
                return L_r + lam_ic * L_ic, parts
            return L_r, parts
        return loss_fn

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hists = []
    t_full = to_tensor(np.linspace(0.0, T, M), requires_grad=True)
    if variant == "curriculum":
        stages = [T * s / 4.0 for s in (1, 2, 3, 4)]
        for k, Ts in enumerate(stages):
            t_col = to_tensor(np.linspace(0.0, Ts, M), requires_grad=True)
            hists.append(train(model.parameters(), make_loss(t_col), adam_iters=n_adam // 4, lr=1e-3,
                               lr_final=1e-5, lbfgs_iters=0, eval_fn=eval_fn, eval_every=50,
                               verbose=False))
        hists.append(train(model.parameters(), make_loss(t_full), adam_iters=0, lbfgs_iters=n_lbfgs,
                           eval_fn=eval_fn, eval_every=50, verbose=False))
    elif variant == "causal":
        state = {"chunks": 16, "eps_list": [1e-2, 1e-1, 1.0, 10.0, 100.0], "k": 0, "eps": 1e-2}
        hists.append(train(model.parameters(), make_loss(t_full, state), adam_iters=n_adam, lr=1e-3,
                           lr_final=1e-5, lbfgs_iters=0, eval_fn=eval_fn, eval_every=50,
                           verbose=False))
        hists.append(train(model.parameters(), make_loss(t_full), adam_iters=0, lbfgs_iters=n_lbfgs,
                           eval_fn=eval_fn, eval_every=50, verbose=False))
    else:
        hists.append(train(model.parameters(), make_loss(t_full), adam_iters=n_adam, lr=1e-3,
                           lr_final=1e-5, lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=50,
                           verbose=False))
    with torch.no_grad():
        y_pred = model(t_test_t).numpy().ravel()
        y_at0 = float(model(t0))
    # final loss in the variant's own formulation and the plain mean squared residual
    t_chk = to_tensor(np.linspace(0.0, T, M), requires_grad=True)
    y_chk = model(t_chk)
    plain_res = float(((derivative(y_chk, t_chk) - R * y_chk * (1.0 - y_chk)) ** 2).mean().detach())
    return {"t": t_test, "y_pred": y_pred, "y_ref": y_ref, "hists": hists,
            "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
            "final_loss": hists[-1].final_loss, "plain_residual_loss": plain_res, "y_at_0": y_at0,
            "train_time_s": float(sum(h.train_time for h in hists))}


def main():
    ex = Example("ch09_training_diagnostics", "trivial_solutions",
                 description="Logistic ODE: attraction of the equilibrium y=0 and remedies",
                 default_seed=1234,
                 extra_args=[replot_arg()])
    M = ex.pick(128, 32)
    n_adam, n_lbfgs = ex.pick(3000, 40), ex.pick(300, 10)
    seeds = [ex.seed + k for k in range(ex.pick(3, 1))]
    t_sweep = T_SWEEP if not ex.quick else [2.0, 10.0]
    rows, runs = [], {}
    jobs = [("soft", T) for T in t_sweep] + [(v, T_MAIN) for v in VARIANTS_B]
    cache = RunCache(ex)
    for variant, T in jobs:
        for s in seeds:
            def job(variant=variant, T=T, s=s):
                with CPUTimer() as tm:
                    res = run(ex, variant, T, s, M, n_adam, n_lbfgs)
                res["cpu_time_s"] = tm.cpu
                return res
            res = cache.get(f"{variant}_T{int(T)}_s{s}", job)
            runs[(variant, T, s)] = res
            for k, h in enumerate(res["hists"]):
                ex.save_history(h, f"{variant}_T{int(T)}_s{s}" + (f"_{k}" if len(res["hists"]) > 1 else ""))
            rows.append({"variant": variant, "T": T, "seed": s, "rel_l2": res["rel_l2"], "linf": res["linf"],
                         "failed": bool(res["rel_l2"] > FAIL_TOL), "final_loss": res["final_loss"],
                         "plain_residual_loss": res["plain_residual_loss"], "y_at_0": res["y_at_0"],
                         "predicted_spurious_loss": spurious_loss(T, 100.0 if variant == "soft100" else 1.0,
                                                                  hard=variant not in ("soft", "soft100")),
                         "train_time_s": res["train_time_s"], "cpu_time_s": res["cpu_time_s"]})
            ex.log(f"{variant:10s} T={T:4g} seed={s}: relL2={res['rel_l2']:.2e} loss={res['final_loss']:.2e} "
                   f"y(0)={res['y_at_0']:.4f} pred.spurious={rows[-1]['predicted_spurious_loss']:.2e} "
                   f"cpu={res['cpu_time_s']:.0f}s")
    ex.save_table(rows, "summary")
    # aggregated table
    agg = []
    for variant, T in jobs:
        sel = [r for r in rows if r["variant"] == variant and r["T"] == T]
        agg.append({"variant": variant, "T": T, "failures": int(sum(r["failed"] for r in sel)),
                    "runs": len(sel), "median_rel_l2": float(np.median([r["rel_l2"] for r in sel])),
                    "min_rel_l2": float(np.min([r["rel_l2"] for r in sel])),
                    "max_rel_l2": float(np.max([r["rel_l2"] for r in sel]))})
    ex.save_table(agg, "aggregate")
    ex.save_results({
        "problem": {"ode": "y' = r y (1-y), y(0) = y0", "r": R, "y0": Y0, "T_sweep": t_sweep, "T_main": T_MAIN,
                    "spurious_loss_soft": "2 r lam y0^2/(lam T + 2 r)", "spurious_loss_hard": "2 r y0^2/T"},
        "hyperparameters": {"width": 32, "depth": 3, "collocation": M, "adam_iters": n_adam,
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": n_lbfgs, "seeds": seeds,
                            "curriculum_stages": "T/4, T/2, 3T/4, T (Adam split equally), then L-BFGS on [0,T]",
                            "causal": "16 chunks, eps in [1e-2,1e-1,1,10,100], switch when min w > 0.99; "
                                      "then L-BFGS on the unweighted loss",
                            "failure_threshold_rel_l2": FAIL_TOL},
        "summary": rows, "aggregate": agg})
    cache.clear()

    tab = []
    for a in agg:
        lab = LABELS[a["variant"]]
        lstar = "--" if a["variant"] == "relative" else num(
            spurious_loss(a["T"], 100.0 if a["variant"] == "soft100" else 1.0,
                          hard=a["variant"] not in ("soft", "soft100")))
        tab.append([lab, f"\\num{{{int(a['T'])}}}", f"{a['failures']}/{a['runs']}", num(a["median_rel_l2"]),
                    lstar])
    latex_table(ex.path("tab_trivial_solutions.tex"), "lrrrr",
                ["formulation", "$T$", "failed runs", r"median rel.\ $L^2$", r"$L^\ast$ (spurious)"], tab,
                midrules=[len(t_sweep)], comment="Chapter 9, trivial_solutions.py")

    if ex.args.no_figures:
        return
    # ---------------------------------------------------------------- figure
    fig, axes = P.figure(2, 2, aspect=0.74)
    s0 = seeds[0]
    ax = axes[0, 0]
    r0 = runs[("soft", T_MAIN, s0)] if ("soft", T_MAIN, s0) in runs else runs[("hard", T_MAIN, s0)]
    P.plot_reference(ax, r0["t"], r0["y_ref"], label="exact")
    a_star = T_MAIN * Y0 / (T_MAIN + 2 * R)
    ax.plot(r0["t"], a_star * np.exp(-R * r0["t"]), color=P.COLORS["muted"], lw=0.8, ls=":",
            label=r"$a^\ast e^{-rt}$")
    for k, v in enumerate(["soft", "hard", "curriculum", "relative"]):
        key = (v, T_MAIN, s0)
        if key in runs:
            P.plot_prediction(ax, runs[key]["t"], runs[key]["y_pred"], label=LABELS[v], k=k, lw=1.3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(f"$T = {int(T_MAIN)}$, seed {s0}")
    ax.legend(loc="center right", fontsize=6.5)

    ax = axes[0, 1]
    for k, s in enumerate(seeds):
        key = ("soft", T_MAIN, s)
        if key in runs:
            h = runs[key]["hists"][-1]
            ax.semilogy(h.iters, P.rolling_median(h.loss, 9), color=P.SERIES[k],
                        ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.1, label=f"seed {s}")
    ax.axhline(spurious_loss(T_MAIN), color=P.COLORS["ref"], lw=0.8, ls=":")
    ax.text(0.55, spurious_loss(T_MAIN) * 1.5, r"$L^\ast(10)$", transform=ax.get_yaxis_transform(),
            fontsize=7, va="bottom")
    ax.axvline(n_adam, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.set_title(f"soft IC, $T = {int(T_MAIN)}$")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    Ts = np.geomspace(1.5, 30, 100)
    ax.loglog(Ts, [spurious_loss(T) for T in Ts], color=P.COLORS["ref"], lw=1.0,
              label=r"$L^\ast(T)$ (soft, $\lambda_{ic}=1$)")
    for T in t_sweep:
        for s in seeds:
            r_ = [r for r in rows if r["variant"] == "soft" and r["T"] == T and r["seed"] == s][0]
            failed = r_["failed"]
            ax.loglog([T], [r_["final_loss"]], ls="none", marker="o" if failed else "s",
                      color=P.SERIES[1] if failed else P.SERIES[0],
                      markerfacecolor=P.SERIES[1] if failed else "white", markersize=4)
    ax.plot([], [], ls="none", marker="o", color=P.SERIES[1], label="final loss, failed run")
    ax.plot([], [], ls="none", marker="s", color=P.SERIES[0], markerfacecolor="white",
            label="final loss, successful run")
    ax.set_xlabel("$T$")
    ax.set_ylabel("final loss")
    ax.set_xticks(t_sweep)
    ax.set_xticklabels([f"{T:g}" for T in t_sweep])
    ax.minorticks_off()
    ax.set_title(r"spurious minimum $L^\ast(T)$")
    ax.legend(loc="lower right", fontsize=6.5)

    ax = axes[1, 1]
    names = ["soft"] + VARIANTS_B
    for k, v in enumerate(names):
        vals = [r["rel_l2"] for r in rows if r["variant"] == v and r["T"] == T_MAIN]
        xs = k + np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [k]
        ax.semilogy(xs, vals, ls="none", marker=P.MARKERS[k % len(P.MARKERS)], color=P.SERIES[k % len(P.SERIES)],
                    markerfacecolor="white", markersize=4)
    ax.axhline(FAIL_TOL, color=P.COLORS["muted"], lw=0.6, ls=":")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["soft\n1", "soft\n100", "hard", "curric.", "causal", "relative"], fontsize=7)
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title(f"formulations at $T = {int(T_MAIN)}$ ({len(seeds)} seeds)")
    P.label_panels(axes)
    ex.savefig(fig, "trivial_solutions_summary")


if __name__ == "__main__":
    main()
