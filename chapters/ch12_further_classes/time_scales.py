"""Dynamic equation y^Delta = p y, y(0) = 1, on three time scales with one PINN code.

Book: Chapter 12 (Further equation classes), Section "Dynamic equations on time scales".

A time scale is a closed subset T of R.  The forward jump is sigma(t) = inf{s in T : s > t},
the graininess mu(t) = sigma(t) - t, and the delta derivative is

    y^Delta(t) = y'(t)                              if mu(t) = 0 (right-dense t),
    y^Delta(t) = (y(sigma(t)) - y(t)) / mu(t)        if mu(t) > 0 (right-scattered t).

The PINN residual therefore uses automatic differentiation at right-dense collocation points
and a forward difference of the network at right-scattered points.  The same code is run on

* the continuum           T = [0, 7]                         (y' = p y),
* the integers            T = {0, 1, ..., 7}                  (y(t+1) = (1 + p) y(t)),
* the hybrid time scale   T = P_{1,1} = U_{k=0}^{3} [2k, 2k+1]   (intervals joined by gaps of length 1).

On P_{1,1} two network designs are compared:
* single - one network output on [0, 7] (it must bridge the gaps);
* heads  - one output per connected component [2k, 2k+1] of T (same hidden layers).
A tanh network is real-analytic, so a single output that equals e^{p t} on [0, 1] cannot satisfy
the jump condition y(2) = (1 + p) y(1) exactly; one head per component removes this obstruction.

Exact solution: the time-scale exponential
    e_p(t, 0) = exp(p * |T cap [0, t]|) * prod_{s in T right-scattered, s < t} (1 + mu(s) p),
i.e. e^{pt}, (1 + p)^t and ((1 + p) e^p)^k e^{p (t - 2k)} on [2k, 2k+1], respectively.

Run:  python chapters/ch12_further_classes/time_scales.py           (about 2-4 min on 1 CPU core)
      python chapters/ch12_further_classes/time_scales.py --quick   (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch

from pinnode import Example, MLP, derivative, linf, rel_l2, to_tensor, train
from pinnode import plotting as P

P_COEF = 0.5          # coefficient p (regressive: 1 + mu p != 0 everywhere)
T_END = 7.0


class TimeScale:
    """A time scale given as a finite union of disjoint closed intervals [a_i, b_i], a_i <= b_i.

    Degenerate intervals (a_i = b_i) are isolated points.  Only what the PINN needs is
    implemented: the forward jump and graininess at the right end points, collocation
    points, the component containing a point and the exact exponential e_p(t, 0) (constant p).
    """

    def __init__(self, name, components):
        self.name = name
        self.comp = [(float(a), float(b)) for a, b in sorted(components)]

    @property
    def scattered(self):
        """Right-scattered points b_i (all right end points except max T) with sigma and mu."""
        return [(b, a2, a2 - b) for (a, b), (a2, _) in zip(self.comp[:-1], self.comp[1:])]

    def component(self, t):
        """Index of the component [a_i, b_i] that contains each t (t must lie in T)."""
        t = np.atleast_1d(np.asarray(t, float))
        starts = np.array([a for a, _ in self.comp])
        return np.searchsorted(starts, t + 1e-12, side="right") - 1

    def dense_points(self, n_per_unit: int):
        """Uniform right-dense collocation points in the nondegenerate components.

        The right end point of a component is right-scattered (except max T) and is
        therefore excluded; max T is right-dense by convention (sigma(max T) = max T).
        """
        pts = []
        last = len(self.comp) - 1
        for i, (a, b) in enumerate(self.comp):
            if b > a:
                n = max(2, int(round(n_per_unit * (b - a))))
                x = np.linspace(a, b, n + 1)
                pts.append(x if i == last else x[:-1])
        return np.concatenate(pts) if pts else np.zeros(0)

    def grid(self, n_total: int):
        """Test points on T: uniform on the intervals, plus every isolated point."""
        length = sum(b - a for a, b in self.comp)
        pts = []
        for a, b in self.comp:
            if b > a:
                pts.append(np.linspace(a, b, max(2, int(round(n_total * (b - a) / length)))))
            else:
                pts.append(np.array([a]))
        return np.concatenate(pts)

    def exp_p(self, t, p):
        """Exact time-scale exponential e_p(t, 0) for constant p (t must lie in T)."""
        t = np.atleast_1d(np.asarray(t, float))
        out = np.empty_like(t)
        for j, tj in enumerate(t):
            dense = sum(max(0.0, min(b, tj) - a) for a, b in self.comp if a <= tj)
            jumps = np.prod([1.0 + mu * p for s, _, mu in self.scattered if s < tj])
            out[j] = np.exp(p * dense) * jumps
        return out


def make_time_scales():
    return {
        "continuous": TimeScale(r"$\mathbb{T}=[0,7]$", [(0.0, T_END)]),
        "discrete": TimeScale(r"$\mathbb{T}=\{0,1,\dots,7\}$", [(k, k) for k in range(8)]),
        "hybrid": TimeScale(r"$\mathbb{T}=\mathbb{P}_{1,1}$", [(2 * k, 2 * k + 1) for k in range(4)]),
    }


class TimeScaleModel(torch.nn.Module):
    """y(t) = 1 + t N_c(t): one shared MLP, output c = component of t (or a single output)."""

    def __init__(self, ts: TimeScale, heads: bool):
        super().__init__()
        self.ts, self.heads = ts, heads
        self.net = MLP(1, len(ts.comp) if heads else 1, width=32, depth=3, activation="tanh",
                       domain=(0.0, T_END))

    def forward(self, t, idx=None):
        n = self.net(t)
        if self.heads:
            n = n.gather(1, idx)
        return 1.0 + t * n                       # y(0) = 1 exactly

    def index(self, t_np):
        return torch.as_tensor(self.ts.component(t_np), dtype=torch.long).reshape(-1, 1)


def solve(ex: Example, ts: TimeScale, tag: str, heads: bool = False):
    torch.manual_seed(ex.seed)
    model = TimeScaleModel(ts, heads)
    p = P_COEF
    t_dense_np = ts.dense_points(ex.pick(50, 10))
    has_dense = t_dense_np.size > 0
    t_dense = to_tensor(t_dense_np, requires_grad=True) if has_dense else None
    i_dense = model.index(t_dense_np) if has_dense else None
    sc = ts.scattered
    s_np, sig_np = np.array([s for s, _, _ in sc]), np.array([sg for _, sg, _ in sc])
    t_sc, t_sig = to_tensor(s_np), to_tensor(sig_np)
    i_sc, i_sig = model.index(s_np), model.index(sig_np)
    mu = to_tensor([m for _, _, m in sc])

    def loss_fn():
        parts, loss = {}, 0.0
        if has_dense:                         # mu = 0: delta derivative = ordinary derivative
            y = model(t_dense, i_dense)
            r = derivative(y, t_dense) - p * y
            parts["dense"] = (r ** 2).mean()
            loss = loss + parts["dense"]
        if len(sc):                           # mu > 0: forward difference of the network
            y_t, y_sig = model(t_sc, i_sc), model(t_sig, i_sig)
            r = (y_sig - y_t) / mu - p * y_t
            parts["scattered"] = (r ** 2).mean()
            loss = loss + parts["scattered"]
        return loss, parts

    t_test = ts.grid(1001)
    y_ref = ts.exp_p(t_test, p)
    t_test_t, i_test = to_tensor(t_test), model.index(t_test)

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t, i_test), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(5000, 60), lr=1e-3,
                 lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=100,
                 print_every=2500)
    with torch.no_grad():
        y_pred = model(t_test_t, i_test).numpy().ravel()
    rel_err = np.abs(y_pred - y_ref) / np.abs(y_ref)
    ex.save_history(hist, tag)
    return {"ts": ts, "hist": hist, "t": t_test, "y_ref": y_ref, "y_pred": y_pred,
            "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
            "max_rel_err": float(rel_err.max()), "n_dense": int(t_dense_np.size),
            "n_scattered": len(sc), "heads": heads}


def main():
    ex = Example("ch12_further_classes", "time_scales",
                 description="y^Delta = p y on R, Z and P_{1,1}: one PINN for continuous, discrete "
                             "and hybrid time", default_seed=1234)
    scales = make_time_scales()
    plan = [("continuous", "continuous", False), ("discrete", "discrete", False),
            ("hybrid_single", "hybrid", False), ("hybrid_heads", "hybrid", True)]
    runs = {}
    for tag, key, heads in plan:
        ex.log(f"training: {tag}")
        runs[tag] = solve(ex, scales[key], tag, heads)
        r = runs[tag]
        ex.log(f"{tag:14s} relL2={r['rel_l2']:.2e} linf={r['linf']:.2e} "
               f"max rel={r['max_rel_err']:.2e} time={r['hist'].train_time:.1f}s")
    p = P_COEF
    closed = {"continuous": np.exp(7 * p), "discrete": (1 + p) ** 7,
              "hybrid": ((1 + p) * np.exp(p)) ** 3 * np.exp(p)}
    check = {k: float(abs(scales[k].exp_p([7.0], p)[0] - v) / v) for k, v in closed.items()}
    rows = [{"run": tag, "time_scale": key, "heads": heads, "n_dense": runs[tag]["n_dense"],
             "n_scattered": runs[tag]["n_scattered"], "rel_l2": runs[tag]["rel_l2"],
             "linf": runs[tag]["linf"], "max_rel_err": runs[tag]["max_rel_err"],
             "final_loss": runs[tag]["hist"].final_loss, "train_time_s": runs[tag]["hist"].train_time,
             "y_exact_T": closed[key]} for tag, key, heads in plan]
    ex.save_table(rows, "summary")
    for tag, _, _ in plan:
        ex.save_table({"t": runs[tag]["t"], "y_exact": runs[tag]["y_ref"], "y_pinn": runs[tag]["y_pred"]},
                      f"predictions_{tag}")
    ex.save_results({
        "problem": {"equation": "y^Delta = p y, y(0) = 1", "p": p, "T_end": T_END,
                    "time_scales": {"continuous": "[0,7]", "discrete": "{0,...,7}",
                                    "hybrid": "union_{k=0}^{3} [2k, 2k+1]"}},
        "hyperparameters": {"width": 32, "depth": 3, "activation": "tanh",
                            "dense_points_per_unit_length": ex.pick(50, 10),
                            "adam_iters": ex.pick(5000, 60), "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters": ex.pick(1000, 10),
                            "loss": "mean dense residual^2 + mean scattered residual^2",
                            "heads": "hybrid_heads: one output per connected component of T"},
        "exp_p_check_rel_diff": check,
        "summary": rows,
        "rel_l2": {tag: runs[tag]["rel_l2"] for tag, _, _ in plan},
        "train_time_s": {tag: runs[tag]["hist"].train_time for tag, _, _ in plan},
        "test_grid": "1001 points uniform on the intervals of T (plus every isolated point)",
    })
    if ex.args.no_figures:
        return

    # ------------------------------------------------------------------ figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    names = {"continuous": r"$\mathbb{R}$", "discrete": r"$\mathbb{Z}$",
             "hybrid_single": r"$\mathbb{P}_{1,1}$, single output",
             "hybrid_heads": r"$\mathbb{P}_{1,1}$, one head per interval"}
    ax = axes[0, 0]
    ts = scales["hybrid"]
    r = runs["hybrid_heads"]
    for j, (a, b) in enumerate(ts.comp):
        m = (r["t"] >= a) & (r["t"] <= b)
        P.plot_reference(ax, r["t"][m], r["y_ref"][m], label="exact $e_p(t,0)$" if j == 0 else None)
        P.plot_prediction(ax, r["t"][m], r["y_pred"][m], k=2,
                          label="PINN, one head per interval" if j == 0 else None)
    for s, sg, _ in ts.scattered:
        ys, yg = ts.exp_p([s, sg], P_COEF)
        ax.annotate("", xy=(sg, yg), xytext=(s, ys),
                    arrowprops=dict(arrowstyle="->", color=P.COLORS["muted"], lw=0.7, ls=":"))
    for (a, b) in ts.comp:
        ax.plot([a, b], [0.0, 0.0], color=P.COLORS["ref"], lw=3.0, solid_capstyle="butt",
                transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(r"hybrid time scale $\mathbb{P}_{1,1}$, $p=1/2$")
    ax.legend(loc="upper left")

    ax = axes[0, 1]
    lab = {"continuous": r"$\mathbb{R}$: $e^{pt}$", "discrete": r"$\mathbb{Z}$: $(1+p)^t$",
           "hybrid_heads": r"$\mathbb{P}_{1,1}$"}
    for k, tag in enumerate(("continuous", "discrete", "hybrid_heads")):
        rr = runs[tag]
        if tag == "discrete":
            ax.plot(rr["t"], rr["y_ref"], ls="none", marker="o", ms=4.5, mfc="none",
                    mec=P.COLORS["ref"], mew=0.8, zorder=2)
            ax.plot(rr["t"], rr["y_pred"], ls="none", marker=P.MARKERS[1], ms=2.6,
                    color=P.SERIES[1], zorder=3, label=lab[tag])
        else:
            for j, (a, b) in enumerate(rr["ts"].comp):
                m = (rr["t"] >= a) & (rr["t"] <= b)
                P.plot_reference(ax, rr["t"][m], rr["y_ref"][m], label=None)
                P.plot_prediction(ax, rr["t"][m], rr["y_pred"][m], k=k if k < 2 else 2,
                                  label=lab[tag] if j == 0 else None)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("one code, three time scales")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    for k, tag in enumerate(("continuous", "discrete", "hybrid_heads", "hybrid_single")):
        rr = runs[tag]
        pos = rr["t"] > 0                    # y(0) = 1 holds exactly (hard constraint)
        e = np.maximum(np.abs(rr["y_pred"] - rr["y_ref"]) / np.abs(rr["y_ref"]), 1e-13)
        if tag == "discrete":
            ax.semilogy(rr["t"][pos], e[pos], ls="none", marker=P.MARKERS[k], ms=3.2, color=P.SERIES[k],
                        label=names[tag])
        else:
            for j, (a, b) in enumerate(rr["ts"].comp):
                m = (rr["t"] >= a) & (rr["t"] <= b) & pos
                ax.semilogy(rr["t"][m], e[m], color=P.SERIES[k], lw=1.0,
                            ls="-" if k == 0 else P.LINESTYLES[k - 1],
                            label=names[tag] if j == 0 else None)
    ax.set_xlabel("$t$")
    ax.set_ylabel("relative error")
    ax.set_ylim(1e-11, 1e-1)
    ax.legend(loc="lower right", ncol=1, fontsize=6.5)

    ax = axes[1, 1]
    for k, tag in enumerate(("hybrid_heads", "hybrid_single")):
        h = runs[tag]["hist"]
        for part, ls in (("dense", "-"), ("scattered", ":")):
            ax.semilogy(h.iters, P.rolling_median(h.parts[part], 9), color=P.SERIES[k + 2], lw=1.1,
                        ls=ls, label=f"{'heads' if k == 0 else 'single'}: "
                                     f"{'right-dense' if part == 'dense' else 'right-scattered'}")
    ax.axvline(runs["hybrid_heads"]["hist"].switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"loss on $\mathbb{P}_{1,1}$")
    ax.legend(loc="lower left", fontsize=6.5)
    P.label_panels(axes)
    ex.savefig(fig, "time_scales_summary")


if __name__ == "__main__":
    main()
