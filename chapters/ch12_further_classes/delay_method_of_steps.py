"""Delay differential equation y'(t) = -y(t - 1) on [0, 3] with history y = 1 on [-1, 0].

Book: Chapter 12 (Further equation classes), Section "Delay differential equations".

Exact solution (method of steps): on [n-1, n], n = 1, 2, 3,
    y(t) = sum_{k=0}^{n} (-1)^k (t - k + 1)^k / k!,
i.e. 1 - t on [0, 1], 3/2 - 2t + t^2/2 on [1, 2], and 3/2 - 2t + t^2/2 - (t-2)^3/6 on [2, 3].
The derivative jumps at t = 0 (from 0 to -1); y'' jumps at t = 1 (from 0 to 1) and y''' at
t = 2 (from 0 to -1): the breaking points of the delay equation.

Two PINN formulations are compared:

* global - one network on [0, 3], y = 1 + t N(t); the delayed term is the history for
           t - 1 <= 0 and the network itself for t - 1 > 0;
* steps  - the method of steps: one network per interval [k, k+1], y_k(t) = y_{k-1}(k) + (t-k) N_k(t),
           trained sequentially and initialized with the hidden-layer weights of the previous
           interval (continuation); the delayed term is the (frozen) network of the previous
           interval, so every breaking point is an interval end point.

Run:  python chapters/ch12_further_classes/delay_method_of_steps.py          (about 1-2 min on 1 core)
      python chapters/ch12_further_classes/delay_method_of_steps.py --quick  (smoke test)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from pinnode import ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor, train, uniform
from pinnode import plotting as P

TAU, T_END = 1.0, 3.0


def history(t):
    return np.ones_like(np.asarray(t, float))


def exact(t, order: int = 0):
    """Exact solution (order 0) or its derivatives (order 1, 2) by the method of steps."""
    t = np.atleast_1d(np.asarray(t, float))
    out = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 0.0:
            out[i] = 1.0 if order == 0 else 0.0
            continue
        n = int(math.ceil(ti - 1e-15))
        s = 0.0
        for k in range(order, n + 1):
            s += (-1) ** k * (ti - k + 1) ** (k - order) / math.factorial(k - order)
        out[i] = s
    return out


def train_global(ex: Example, t_test):
    torch.manual_seed(ex.seed)
    net = MLP(1, 1, width=32, depth=3, activation="tanh", domain=(0.0, T_END))
    model = ConstrainedModel(net, lambda t, n: 1.0 + t * n)       # continuity with the history
    t_col = to_tensor(uniform(ex.pick(300, 30), 0.0, T_END), requires_grad=True)
    lag = t_col.detach() - TAU                 # collocation points are sorted, so the
    n_hist = int((lag <= 0.0).sum())           # first n_hist delayed arguments lie in [-1, 0]
    lag_net = lag[n_hist:]                     # delayed arguments inside (0, T - 1]
    hist_vals = to_tensor(history(lag[:n_hist].numpy().ravel()))
    y_ref = exact(t_test)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y = model(t_col)
        delayed = torch.cat([hist_vals, model(lag_net)])   # history, then the network's own past
        r = derivative(y, t_col) + delayed
        L = (r ** 2).mean()
        return L, {"residual": L}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(8000, 60), lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=100, print_every=4000)
    return model, hist


def train_steps(ex: Example, t_test):
    """Method of steps: one small network per delay interval, trained one after the other."""
    models, hists = [], []
    left_value = 1.0                           # y(0) from the history
    for k in range(int(T_END / TAU)):
        a, b = k * TAU, (k + 1) * TAU
        torch.manual_seed(ex.seed + k)
        net = MLP(1, 1, width=32, depth=3, activation="tanh", domain=(a, b))
        if models:                             # continuation: start from the previous interval's
            net.body.load_state_dict(models[-1].net.body.state_dict())   # weights (not its scaling)
        model = ConstrainedModel(net, (lambda c, a0: (lambda t, n: c + (t - a0) * n))(left_value, a))
        t_col = to_tensor(uniform(ex.pick(100, 20), a, b), requires_grad=True)
        with torch.no_grad():                  # delayed term: known function on [a - 1, b - 1]
            lag = t_col.detach() - TAU
            delayed = (torch.ones_like(lag) if k == 0 else models[-1](lag))
        m_test = (t_test >= a) & (t_test <= b)
        t_k = to_tensor(t_test[m_test])
        y_k = exact(t_test[m_test])

        def loss_fn(model=model, t_col=t_col, delayed=delayed):
            y = model(t_col)
            r = derivative(y, t_col) + delayed
            L = (r ** 2).mean()
            return L, {"residual": L}

        def eval_fn(model=model, t_k=t_k, y_k=y_k):
            with torch.no_grad():
                return {"rel_l2": rel_l2(model(t_k), y_k)}

        hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(3000, 40), lr=1e-3,
                     lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), lbfgs_rtol=1e-8, eval_fn=eval_fn,
                     eval_every=100, print_every=3000)
        with torch.no_grad():
            left_value = float(model(to_tensor([b])).item())
        models.append(model)
        hists.append(hist)
    return models, hists


def eval_steps(models, t):
    """Evaluate the piecewise (method-of-steps) solution and its first two derivatives."""
    t = np.asarray(t, float)
    out = [np.zeros_like(t) for _ in range(3)]
    for k, model in enumerate(models):
        a, b = k * TAU, (k + 1) * TAU
        m = (t >= a) & ((t < b) if k < len(models) - 1 else (t <= b))
        if not m.any():
            continue
        tt = to_tensor(t[m], requires_grad=True)
        y = model(tt)
        d1 = derivative(y, tt)
        d2 = derivative(d1, tt)
        for j, v in enumerate((y, d1, d2)):
            out[j][m] = v.detach().numpy().ravel()
    return out


def eval_global(model, t):
    tt = to_tensor(t, requires_grad=True)
    y = model(tt)
    d1 = derivative(y, tt)
    d2 = derivative(d1, tt)
    return [v.detach().numpy().ravel() for v in (y, d1, d2)]


def main():
    ex = Example("ch12_further_classes", "delay_method_of_steps",
                 description="DDE y'(t) = -y(t-1): global PINN versus method of steps",
                 default_seed=1234)
    t_test = np.linspace(0.0, T_END, 1001)
    y_ref = exact(t_test)

    ex.log("global network on [0, 3]")
    g_model, g_hist = train_global(ex, t_test)
    ex.save_history(g_hist, "global")
    ex.log("method of steps: three networks")
    s_models, s_hists = train_steps(ex, t_test)
    for k, h in enumerate(s_hists):
        ex.save_history(h, f"steps_{k}")

    yg, dg, d2g = eval_global(g_model, t_test)
    ys, ds, d2s = eval_steps(s_models, t_test)
    # residual of the global network on the test grid (diagnostic)
    lag = t_test - TAU
    delayed_g = np.where(lag <= 0, 1.0, np.interp(lag, t_test, yg))
    res_g = dg + delayed_g
    rows = []
    for name, y, hist_time, loss in (
            ("global", yg, g_hist.train_time, g_hist.final_loss),
            ("steps", ys, sum(h.train_time for h in s_hists), max(h.final_loss for h in s_hists))):
        err = np.abs(y - y_ref)
        rows.append({"formulation": name, "rel_l2": rel_l2(y, y_ref), "linf": linf(y, y_ref),
                     "max_err_near_t1": float(err[(t_test > 0.9) & (t_test < 1.1)].max()),
                     "final_loss": loss, "train_time_s": hist_time})
        ex.log(f"{name:7s} relL2={rows[-1]['rel_l2']:.2e} linf={rows[-1]['linf']:.2e} "
               f"time={hist_time:.1f}s")
    # maximum error on each delay interval (where the method of steps loses accuracy)
    per_interval = {name: [float(np.max(np.abs(y - y_ref)[(t_test >= k) & (t_test <= k + 1)]))
                           for k in range(int(T_END / TAU))] for name, y in (("global", yg), ("steps", ys))}
    final_losses_steps = [float(h.final_loss) for h in s_hists]
    # second-derivative jump at t = 1 as seen by each model
    i_lo, i_hi = np.searchsorted(t_test, 0.97), np.searchsorted(t_test, 1.03)
    jump = {"exact": 1.0, "global": float(d2g[i_hi] - d2g[i_lo]), "steps": float(d2s[i_hi] - d2s[i_lo])}
    ex.save_table(rows, "summary")
    ex.save_table({"t": t_test, "y_exact": y_ref, "y_global": yg, "y_steps": ys,
                   "d2_exact": exact(t_test, 2), "d2_global": d2g, "d2_steps": d2s}, "predictions")
    ex.save_results({
        "problem": {"equation": "y'(t) = -y(t-1)", "history": "y = 1 on [-1, 0]", "T": T_END,
                    "breaking_points": [0, 1, 2]},
        "hyperparameters": {
            "global": {"width": 32, "depth": 3, "collocation": ex.pick(300, 30),
                       "adam_iters": ex.pick(8000, 60), "lbfgs_iters": ex.pick(1000, 10)},
            "steps": {"networks": 3, "width": 32, "depth": 3, "collocation_per_interval": ex.pick(100, 20),
                      "adam_iters": ex.pick(3000, 40), "lbfgs_iters": ex.pick(1000, 10),
                      "lbfgs_rtol": 1e-8, "warm_start": "hidden layers of the previous interval"},
            "lr": "1e-3 -> 1e-5 (cosine)"},
        "summary": rows,
        "d2_jump_t1_window_0.97_1.03": jump,
        "max_error_per_interval": per_interval,
        "final_loss_per_interval_steps": final_losses_steps,
        "max_abs_residual_global": float(np.max(np.abs(res_g))),
        "rel_l2": {r["formulation"]: r["rel_l2"] for r in rows},
        "train_time_s": {r["formulation"]: r["train_time_s"] for r in rows},
        "test_grid": "1001 uniform points on [0, 3]",
    })
    if ex.args.no_figures:
        return

    # ------------------------------------------------------------------ figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    th = np.linspace(-1.0, 0.0, 50)
    ax.plot(th, history(th), color=P.COLORS["muted"], lw=1.0, ls="-", label=r"history $\varphi\equiv1$")
    P.plot_reference(ax, t_test, y_ref, label="exact")
    P.plot_prediction(ax, t_test, yg, label="global PINN", k=0, markevery=(0, 100))
    P.plot_prediction(ax, t_test, ys, label="method of steps", k=1, markevery=(50, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    P.plot_error(ax, t_test[1:], (yg - y_ref)[1:], label="global PINN", k=0)     # y(0) = 1 exactly
    P.plot_error(ax, t_test[1:], (ys - y_ref)[1:], label="method of steps", k=1)
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    P.plot_reference(ax, t_test, exact(t_test, 2), label="exact $y''$")
    P.plot_prediction(ax, t_test, d2g, label="global PINN", k=0)
    P.plot_prediction(ax, t_test, d2s, label="method of steps", k=1)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y''(t)$")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    P.plot_metric(ax, g_hist, "rel_l2", label="global PINN", k=0, show_switch=False)
    offset = 0
    for k, h in enumerate(s_hists):
        it = np.asarray(h.eval_iters) + offset
        ax.semilogy(it, h.metrics["rel_l2"], color=P.SERIES[1], lw=1.2, ls=P.LINESTYLES[0],
                    label="method of steps" if k == 0 else None)
        offset = it[-1]
        ax.axvline(offset, color=P.COLORS["light"], lw=0.5)
    ax.set_xlabel("iteration")
    ax.legend(loc="upper right")
    for a in (axes[0, 0], axes[0, 1], axes[1, 0]):
        for tb in (1.0, 2.0):
            a.axvline(tb, color=P.COLORS["light"], lw=0.6, ls="-", zorder=0)
    P.label_panels(axes)
    ex.savefig(fig, "delay_summary")


if __name__ == "__main__":
    main()
