"""Damped pendulum as a terminal value problem: soft versus hard terminal conditions.

Problem:  y'' + beta y' + sin(y) = 0  on [0, T],  y(T) = y_T,  y'(T) = 0,
          beta = 0.1, T = 5, y_T = 0.4 (no small-angle approximation).

Reference: DOP853 (rtol = atol = 1e-12) integrating *backward* from t = T to t = 0
(scipy's solve_ivp accepts a decreasing time span, so no reformulation or shooting is
needed); cross-checked against forward integration of the time-reversed system in
s = T - t,  Y'' = beta Y' - sin(Y),  Y(0) = y_T,  Y'(0) = -y'(T).

PINN: Legendre feature network (LeNN, Chapter 3) of degree 7 followed by two tanh layers of
width 32; 128 Legendre-Gauss-Lobatto collocation points on [0, T]; Adam (cosine schedule)
followed by L-BFGS.  Two ways of imposing the terminal data are compared:

* soft  - penalty  lambda_tc [(y(T) - y_T)^2 + y'(T)^2]  with lambda_tc = 10;
* hard  - trial function  y_theta(t) = y_T + ((T - t)/T)^2 N_theta(t)  (exact terminal data).

Errors are measured on a uniform grid of 1001 points.

Book: Chapter 6, Section "Worked examples", Figure tvp:fig:pendulum, Table tvp:tab:examples.
Run:  PYTHONPATH=. python chapters/ch06_tvp/pendulum_tvp.py          (about 2 min, 1 core)
      PYTHONPATH=. python chapters/ch06_tvp/pendulum_tvp.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import (ConstrainedModel, Example, LegendreNet, Timer, derivatives,
                     legendre_gauss_lobatto, linf, rel_l2, to_tensor, train)

BETA, T, Y_T, V_T = 0.1, 5.0, 0.4, 0.0
DEGREE, HIDDEN, N_COLLOC, LAMBDA_TC = 7, (32, 32), 128, 10.0


def rhs(t, z):
    return [z[1], -BETA * z[1] - np.sin(z[0])]


def reference(t_eval):
    """Backward DOP853 from t = T; returns y and y' on t_eval (increasing)."""
    sol = solve_ivp(rhs, (T, 0.0), [Y_T, V_T], method="DOP853", rtol=1e-12, atol=1e-12,
                    t_eval=t_eval[::-1])
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y[0][::-1], sol.y[1][::-1]


def reversed_reference(t_eval):
    """Same solution from the time-reversed IVP in s = T - t (independent check)."""
    sol = solve_ivp(lambda s, z: [z[1], BETA * z[1] - np.sin(z[0])], (0.0, T), [Y_T, -V_T],
                    method="DOP853", rtol=1e-12, atol=1e-12, t_eval=(T - t_eval)[::-1])
    return sol.y[0][::-1]


def build(hard: bool):
    net = LegendreNet(DEGREE, HIDDEN, 1, activation="tanh", domain=(0.0, T))
    if hard:
        return ConstrainedModel(net, lambda t, n: Y_T + V_T * (t - T) + ((T - t) / T) ** 2 * n)
    return net


def run(ex: Example, hard: bool, t_test, y_ref):
    torch.manual_seed(ex.seed)
    model = build(hard)
    t_col = to_tensor(np.real(legendre_gauss_lobatto(ex.pick(N_COLLOC, 16), 0.0, T)),
                      requires_grad=True)
    t_end = to_tensor([T], requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y, dy, ddy = derivatives(model(t_col), t_col, 2)
        r = ddy + BETA * dy + torch.sin(y)
        L_r = (r ** 2).mean()
        if hard:
            return L_r, {"residual": L_r}
        yb, dyb = derivatives(model(t_end), t_end, 1)
        L_tc = ((yb - Y_T) ** 2 + (dyb - V_T) ** 2).sum()
        return L_r + LAMBDA_TC * L_tc, {"residual": L_r, "terminal": L_tc}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(6000, 60), lr=1e-3,
                 lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn,
                 eval_every=50, print_every=2000)
    t_req = to_tensor(t_test, requires_grad=True)
    y, dy, ddy = derivatives(model(t_req), t_req, 2)
    res = (ddy + BETA * dy + torch.sin(y)).detach().numpy().ravel()
    y_pred = y.detach().numpy().ravel()
    yb, dyb = derivatives(model(t_end), t_end, 1)
    return {"hist": hist, "y_pred": y_pred, "rel_l2": rel_l2(y_pred, y_ref),
            "linf": linf(y_pred, y_ref), "err_yT": abs(yb.item() - Y_T),
            "err_dyT": abs(dyb.item() - V_T), "max_abs_residual": float(np.max(np.abs(res))),
            "final_loss": hist.final_loss, "train_time_s": hist.train_time}


# ----------------------------------------------------------------------------- caching
# Finished trainings are cached in the output folder; with --resume a rerun (for example after
# the process was interrupted) reuses them instead of training again. Times are those of the
# original run.
RESUME_ARG = (["--resume"], {"action": "store_true",
                             "help": "reuse trainings cached in the output folder"})


def load_cache(ex, tag):
    import pickle
    path = ex.path(f"cache_{tag}.pkl")
    if getattr(ex.args, "resume", False) and path.exists():
        with open(path, "rb") as fh:
            ex.log(f"resumed '{tag}' from {path.name}")
            return pickle.load(fh)
    return None


def save_cache(ex, tag, obj):
    import pickle
    with open(ex.path(f"cache_{tag}.pkl"), "wb") as fh:
        pickle.dump(obj, fh)


def main():
    ex = Example("ch06_tvp", "pendulum_tvp",
                 description="Damped pendulum TVP: LeNN with soft and hard terminal conditions",
                 default_seed=1234, extra_args=[RESUME_ARG])
    t_test = np.linspace(0.0, T, 1001)
    y_ref, dy_ref = reference(t_test)
    times = []
    for _ in range(5):                      # wall time of the classical backward solve (best of 5)
        with Timer() as tm:
            reference(t_test)
        times.append(tm.elapsed)
    ref_time = min(times)
    check = float(np.max(np.abs(reversed_reference(t_test) - y_ref)))
    ex.log(f"reference: y(0)={y_ref[0]:.12f}, y'(0)={dy_ref[0]:.12f}; "
           f"time-reversed IVP agrees to {check:.1e}")

    out, rows = {}, []
    for hard in (False, True):
        tag = "hard" if hard else "soft"
        ex.log(f"training {tag} terminal conditions")
        r = load_cache(ex, tag) or run(ex, hard, t_test, y_ref)
        save_cache(ex, tag, r)
        out[tag] = r
        ex.save_history(r["hist"], tag)
        rows.append({"variant": tag, "rel_l2": r["rel_l2"], "linf": r["linf"],
                     "err_yT": r["err_yT"], "err_dyT": r["err_dyT"],
                     "max_abs_residual": r["max_abs_residual"], "final_loss": r["final_loss"],
                     "train_time_s": r["train_time_s"]})
        ex.log(f"{tag}: relL2={r['rel_l2']:.2e} linf={r['linf']:.2e} |y(T)-yT|={r['err_yT']:.1e} "
               f"|y'(T)|={r['err_dyT']:.1e} max|res|={r['max_abs_residual']:.1e} "
               f"time={r['train_time_s']:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_table({"t": t_test, "y_ref": y_ref, "y_soft": out["soft"]["y_pred"],
                   "y_hard": out["hard"]["y_pred"]}, "predictions")
    ex.save_results({
        "problem": {"beta": BETA, "T": T, "y_T": Y_T, "dy_T": V_T,
                    "reference": "DOP853 backward, rtol=atol=1e-12",
                    "reference_y0": float(y_ref[0]), "reference_dy0": float(dy_ref[0]),
                    "reversed_ivp_check": check, "reference_time_s": ref_time},
        "hyperparameters": {"network": f"LeNN degree {DEGREE} + tanh {HIDDEN}",
                            "collocation": f"{N_COLLOC} LGL nodes", "lambda_tc": LAMBDA_TC,
                            "adam_iters": ex.pick(6000, 60), "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters": ex.pick(1000, 10)},
        "summary": rows,
        "rel_l2": out["hard"]["rel_l2"], "linf": out["hard"]["linf"],
        "train_time_s": sum(r["train_time_s"] for r in rows),
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P          # imported only when figures are drawn

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, t_test, y_ref, label="reference")
    P.plot_prediction(ax, t_test, out["soft"]["y_pred"], label="soft terminal", k=0,
                      markevery=(30, 100))
    P.plot_prediction(ax, t_test, out["hard"]["y_pred"], label="hard terminal", k=1,
                      markevery=(80, 100))
    ax.plot([T], [Y_T], ls="none", marker="*", color=P.COLORS["ref"], markersize=6,
            label="terminal data")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$y(t)$")
    ax.set_ylim(-0.55, 0.9)
    ax.legend(loc="upper left", ncol=2)

    ax = axes[0, 1]
    P.plot_error(ax, t_test, out["soft"]["y_pred"] - y_ref, label="soft terminal", k=0)
    P.plot_error(ax, t_test, out["hard"]["y_pred"] - y_ref, label="hard terminal", k=1)

    ax = axes[1, 0]
    hs, hh = out["soft"]["hist"], out["hard"]["hist"]
    ax.semilogy(hs.iters, P.rolling_median(np.asarray(hs.parts["residual"]), 9),
                color=P.SERIES[0], ls="--", lw=1.1, label="soft: residual")
    ax.semilogy(hs.iters, P.rolling_median(np.asarray(hs.parts["terminal"]), 9),
                color=P.SERIES[0], ls=":", lw=1.1, label="soft: terminal")
    ax.semilogy(hh.iters, P.rolling_median(np.asarray(hh.parts["residual"]), 9),
                color=P.SERIES[1], ls="-.", lw=1.1, label="hard: residual")
    if hs.switch_iter:
        ax.axvline(hs.switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss component")
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    P.plot_metric(ax, hs, "rel_l2", label="soft terminal", k=0)
    P.plot_metric(ax, hh, "rel_l2", label="hard terminal", k=1)
    P.label_panels(axes)
    ex.savefig(fig, "pendulum_tvp")


if __name__ == "__main__":
    main()
