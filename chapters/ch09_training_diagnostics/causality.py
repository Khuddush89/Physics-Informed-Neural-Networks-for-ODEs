"""Long horizons and causality: the harmonic oscillator  y'' + y = 0,  y(0) = 1,  y'(0) = 0.

Book: Chapter 9, Section "Long horizons and causality".

Exact solution y = cos t on [0, T] for T in {2 pi, 8 pi, 16 pi} (1, 4 and 8 periods). All
methods use hard initial conditions and the same optimizer budget (Adam + L-BFGS):

* vanilla  - one tanh MLP (64 x 3) on [0, T], M = 256 uniform collocation points;
* causal   - the same network and points, residuals grouped into 32 consecutive chunks with
             weights w_1 = 1, w_i = exp(-eps sum_{k<i} L_k) (no gradient through w), loss
             (1/32) sum_i w_i L_i; eps runs through 1e-2, 1e-1, 1, 10, 100 and is increased
             whenever min_i w_i > 0.99 (Wang, Sankaran & Perdikaris 2024); the final L-BFGS
             phase minimizes the unweighted loss;
* causal_fixed - as "causal" with the fixed causality parameter eps = 1 from the start;
* marching - one window per period, each with its own tanh MLP (32 x 3), the initial values
             (y, y') carried over from the end of the previous window; the budget is split
             equally between the windows.

For u = (y, y') the residual r = y'' + y enters as u' = A u + (0, r) with skew-symmetric A,
so the one-sided Lipschitz constant is 0 and |u_theta(t) - u(t)| <= int_0^t |r(s)| ds; this
a posteriori bound is evaluated for every run.

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/causality.py          (~9 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/causality.py --quick  (smoke test)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CPUTimer, RunCache, causal_weights, latex_table, num, replot_arg  # noqa: E402
from pinnode import ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor, train  # noqa: E402
from pinnode import plotting as P  # noqa: E402

PERIODS = [1, 4, 8]
METHODS = ["vanilla", "causal", "causal_fixed", "marching"]
LABELS = {"vanilla": "vanilla", "causal": r"causal, annealed $\varepsilon$",
          "causal_fixed": r"causal, $\varepsilon=1$", "marching": "time-marching"}
EPS_FIXED = 1.0
EPS_LIST = [1e-2, 1e-1, 1.0, 10.0, 100.0]
DELTA = 0.99
N_CHUNKS = 32


def ic_transform(t0, y0, v0):
    return lambda t, n: y0 + v0 * (t - t0) + (t - t0) ** 2 * n


def evaluate(pieces, t):
    """Values, derivatives and residuals of a piecewise network solution on the grid t."""
    y, dy, r = np.empty_like(t), np.empty_like(t), np.empty_like(t)
    for a, b, model in pieces:
        m = (t >= a) & (t <= b)
        if not np.any(m):
            continue
        tt = to_tensor(t[m], requires_grad=True)
        yy = model(tt)
        d1 = derivative(yy, tt)
        d2 = derivative(d1, tt)
        y[m] = yy.detach().numpy().ravel()
        dy[m] = d1.detach().numpy().ravel()
        r[m] = (d2 + yy).detach().numpy().ravel()
    return y, dy, r


def run(ex, method, T, M, n_adam, n_lbfgs, snapshot_iters):
    torch.manual_seed(ex.seed)
    info = {"weights": [], "weight_iters": [], "eps_switch": []}
    hists = []
    if method in ("vanilla", "causal", "causal_fixed"):
        net = MLP(1, 1, 64, 3, "tanh", domain=(0.0, T))
        model = ConstrainedModel(net, ic_transform(0.0, 1.0, 0.0))
        t_col = to_tensor(np.linspace(0.0, T, M), requires_grad=True)
        state = {"k": 0, "eps": EPS_LIST[0] if method == "causal" else EPS_FIXED, "it": 0}

        def residual():
            y = model(t_col)
            return derivative(y, t_col, 2) + y

        def loss_plain():
            r = residual()
            L = (r ** 2).mean()
            return L, {"residual": L}

        def loss_causal():
            r = residual()
            Lc = (r ** 2).reshape(N_CHUNKS, -1).mean(dim=1)
            w = causal_weights(Lc, state["eps"])
            if state["it"] in snapshot_iters:
                info["weights"].append(w.numpy().copy())
                info["weight_iters"].append(state["it"])
            if method == "causal" and float(w.min()) > DELTA and state["k"] < len(EPS_LIST) - 1:
                state["k"] += 1
                state["eps"] = EPS_LIST[state["k"]]
                info["eps_switch"].append(state["it"])
            state["it"] += 1
            return (w * Lc).mean(), {"residual": (r ** 2).mean(), "w_min": float(w.min()), "eps": state["eps"]}

        hists.append(train(net.parameters(), loss_plain if method == "vanilla" else loss_causal,
                           adam_iters=n_adam, lr=1e-3, lr_final=1e-5, lbfgs_iters=0, print_every=0,
                           verbose=False))
        if method in ("causal", "causal_fixed"):
            Lc = (residual() ** 2).reshape(N_CHUNKS, -1).mean(dim=1)
            info["weights"].append(causal_weights(Lc, state["eps"]).numpy().copy())
            info["weight_iters"].append(n_adam)
            info["eps_final"] = state["eps"]
        hists.append(train(net.parameters(), loss_plain, adam_iters=0, lbfgs_iters=n_lbfgs,
                           print_every=0, verbose=False))
        pieces = [(0.0, T, model)]
    else:
        K = int(round(T / (2.0 * math.pi)))
        edges = np.linspace(0.0, T, K + 1)
        y0, v0, pieces = 1.0, 0.0, []
        for k in range(K):
            a, b = float(edges[k]), float(edges[k + 1])
            net = MLP(1, 1, 32, 3, "tanh", domain=(a, b))
            model = ConstrainedModel(net, ic_transform(a, y0, v0))
            t_col = to_tensor(np.linspace(a, b, max(64, M // K)), requires_grad=True)

            def loss_fn(model=model, t_col=t_col):
                y = model(t_col)
                r = derivative(y, t_col, 2) + y
                L = (r ** 2).mean()
                return L, {"residual": L}

            hists.append(train(net.parameters(), loss_fn, adam_iters=n_adam // K, lr=1e-3, lr_final=1e-5,
                               lbfgs_iters=max(1, n_lbfgs // K), print_every=0, verbose=False))
            tb = to_tensor([b], requires_grad=True)
            yb = model(tb)
            vb = derivative(yb, tb)
            y0, v0 = float(yb.detach()), float(vb.detach())
            pieces.append((a, b, model))
        info["windows"] = K
    return pieces, hists, info


def main():
    ex = Example("ch09_training_diagnostics", "causality",
                 description="Harmonic oscillator on long intervals: vanilla vs causal weights vs time-marching",
                 default_seed=1234,
                 extra_args=[replot_arg()])
    M = ex.pick(256, 64)
    n_adam, n_lbfgs = ex.pick(6000, 64), ex.pick(500, 8)
    periods = PERIODS if not ex.quick else [1, 2]
    snapshot_iters = set([0, n_adam // 16, n_adam // 4, n_adam // 2])
    rows, store = [], {}
    cache = RunCache(ex)
    for P_ in periods:
        T = 2.0 * math.pi * P_
        t_test = np.linspace(0.0, T, 1001)
        y_ref, dy_ref = np.cos(t_test), -np.sin(t_test)
        t_fine = np.linspace(0.0, T, 4001)
        for meth in METHODS:
            if meth == "marching" and P_ == 1:
                continue          # one window = vanilla
            def job(meth=meth, T=T, t_test=t_test, t_fine=t_fine):
                with CPUTimer() as tm:
                    pieces, hists, info = run(ex, meth, T, M, n_adam, n_lbfgs, snapshot_iters)
                y, dy, _ = evaluate(pieces, t_test)
                _, _, r_f = evaluate(pieces, t_fine)
                return {"y": y, "dy": dy, "r_f": r_f, "hists": hists, "info": info, "cpu": tm.cpu}
            res = cache.get(f"P{P_}_{meth}", job)
            y, dy, r_f, hists, info = res["y"], res["dy"], res["r_f"], res["hists"], res["info"]
            bound_f = np.concatenate([[0.0], np.cumsum(0.5 * (np.abs(r_f[1:]) + np.abs(r_f[:-1])) * np.diff(t_fine))])
            bound = np.interp(t_test, t_fine, bound_f)
            err_u = np.hypot(y - y_ref, dy - dy_ref)
            store[(P_, meth)] = {"t": t_test, "y": y, "err_u": err_u, "bound": bound, "info": info}
            for j, h in enumerate(hists):
                ex.save_history(h, f"P{P_}_{meth}" + (f"_{j}" if len(hists) > 1 else ""))
            rows.append({"periods": P_, "T": T, "method": meth, "rel_l2": rel_l2(y, y_ref), "linf": linf(y, y_ref),
                         "max_state_error": float(err_u.max()), "a_posteriori_bound_at_T": float(bound[-1]),
                         "bound_holds": bool(np.all(err_u <= bound + 1e-10)),
                         "eps_final": info.get("eps_final"), "eps_switch_iters": info.get("eps_switch"),
                         "windows": info.get("windows"),
                         "train_time_s": float(sum(h.train_time for h in hists)), "cpu_time_s": res["cpu"]})
            ex.log(f"T={P_}x2pi {meth:9s}: relL2={rows[-1]['rel_l2']:.2e} max|u-u_ref|={err_u.max():.2e} "
                   f"bound(T)={bound[-1]:.2e} eps_final={info.get('eps_final')} cpu={res['cpu']:.0f}s")
        ex.save_table({"t": t_test, "exact": y_ref, **{m: store[(P_, m)]["y"] for m in METHODS if (P_, m) in store},
                       **{f"{m}_bound": store[(P_, m)]["bound"] for m in METHODS if (P_, m) in store}},
                      f"predictions_P{P_}")
    ex.save_table(rows, "summary")
    ex.save_results({
        "problem": {"ode": "y'' + y = 0, y(0)=1, y'(0)=0", "exact": "cos t", "periods": periods},
        "hyperparameters": {"vanilla_causal_net": "tanh MLP 64 x 3", "marching_net": "tanh MLP 32 x 3 per period",
                            "collocation": M, "adam_iters": n_adam, "lbfgs_iters": n_lbfgs,
                            "lr": "1e-3 -> 1e-5 (cosine)", "causal": {"chunks": N_CHUNKS, "eps": EPS_LIST,
                                                                      "delta": DELTA}},
        "summary": rows})
    cache.clear()

    def get(P_, m, key):
        for r in rows:
            if r["periods"] == P_ and r["method"] == m:
                return r[key]
        return None
    tab = [[f"{P_}", f"\\num{{{2 * P_}}}$\\pi$"] + [num(get(P_, m, "rel_l2")) if get(P_, m, "rel_l2") is not None
                                                   else "--" for m in METHODS] for P_ in periods]
    latex_table(ex.path("tab_causality.tex"), "rrrrrr",
                ["periods", "$T$", "vanilla", r"causal (annealed $\varepsilon$)", r"causal ($\varepsilon=1$)",
                 "time-marching"], tab,
                comment="Chapter 9, causality.py: relative L2 errors")

    if ex.args.no_figures:
        return
    Pm = periods[-1]
    fig, axes = P.figure(2, 2, aspect=0.74)
    ax = axes[0, 0]
    d0 = store[(Pm, "vanilla")]
    P.plot_reference(ax, d0["t"], np.cos(d0["t"]), label="exact")
    for k, m in enumerate(METHODS):
        if (Pm, m) in store:
            P.plot_prediction(ax, store[(Pm, m)]["t"], store[(Pm, m)]["y"], label=LABELS[m], k=k, lw=1.2)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_ylim(-1.6, 1.9)
    ax.set_title(f"$T = {2 * Pm}\\pi$")
    ax.legend(loc="upper right", ncol=2, fontsize=6.5)

    ax = axes[0, 1]
    for k, m in enumerate(METHODS):
        if (Pm, m) in store:
            d = store[(Pm, m)]
            P.plot_error(ax, d["t"], d["err_u"], label=LABELS[m], k=k, ylabel=r"$|u_\theta(t)-u(t)|$")
            ax.semilogy(d["t"], np.maximum(d["bound"], 1e-13), color=P.SERIES[k], lw=0.7, ls=":")
    lo = min(float(np.min(store[(Pm, m)]["err_u"][store[(Pm, m)]["t"] > 0.1])) for m in METHODS if (Pm, m) in store)
    ax.set_ylim(bottom=max(lo / 3.0, 1e-12))
    ax.set_title("state error; dotted: a posteriori bound")
    ax.legend(loc="lower right", fontsize=6.5)

    ax = axes[1, 0]
    info = store[(Pm, "causal_fixed")]["info"]
    T = 2.0 * math.pi * Pm
    mids = (np.arange(N_CHUNKS) + 0.5) * T / N_CHUNKS
    for k, (it, w) in enumerate(zip(info["weight_iters"], info["weights"])):
        ax.plot(mids, w, color=P.SERIES[k % len(P.SERIES)], ls=P.LINESTYLES[k % len(P.LINESTYLES)],
                marker=P.MARKERS[k % len(P.MARKERS)], markevery=4, markerfacecolor="white", markersize=3,
                lw=1.0, label=f"iteration {it}")
    ax.set_xlabel("$t$ (chunk midpoint)")
    ax.set_ylabel("causal weight $w_i$")
    ax.set_title(f"causal weights ($\\varepsilon=1$), $T = {2 * Pm}\\pi$")
    ax.legend(loc="upper right", fontsize=6)

    ax = axes[1, 1]
    for k, m in enumerate(METHODS):
        xs = [2 * p for p in periods if get(p, m, "rel_l2") is not None]
        ys = [get(p, m, "rel_l2") for p in periods if get(p, m, "rel_l2") is not None]
        ax.semilogy(xs, ys, color=P.SERIES[k], ls="-" if k == 0 else P.LINESTYLES[k - 1], marker=P.MARKERS[k],
                    markerfacecolor="white", label=LABELS[m])
    ax.set_xlabel(r"$T/\pi$")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("error against interval length")
    ax.legend(loc="upper left")
    P.label_panels(axes)
    ex.savefig(fig, "causality_summary")


if __name__ == "__main__":
    main()
