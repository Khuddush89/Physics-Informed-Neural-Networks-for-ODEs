"""Third-order (jerk-type) terminal value problem: network architecture and terminal constraints.

Problem:  y''' + d y'' + c (y')^2 + k y sin(y) = 0  on [0, 1],
          y(1) = 0.1,  y'(1) = 0.2,  y''(1) = 0.3,   k = 0.5, c = 0.05, d = 0.1.

Reference: the equivalent first-order system integrated backward from t = 1 with DOP853
(rtol = atol = 1e-12); cross-checked by shooting (Newton-type root finding on the unknown
initial state and forward integration).  The backward sensitivity matrix
d(y, y', y'')(0) / d(y, y', y'')(1) is computed from the variational equation; its norm
(about 2) shows that this TVP is well conditioned.

Three PINN variants with the same optimizer and 64 Legendre-Gauss-Lobatto points:

* leg-soft : Legendre feature network (degree 7, two tanh layers of width 20) with the terminal
             data as penalties, weight lambda_tc = 200;
* mlp-soft : tanh MLP (three layers of width 32) with the same penalties;
* mlp-hard : the same MLP inside the terminal Taylor trial function
             y_theta(t) = 0.1 + 0.2 (t - 1) + 0.15 (t - 1)^2 + (1 - t)^3 N_theta(t),
             which satisfies the three terminal conditions exactly.

Errors are measured on a uniform grid of 1001 points.

Book: Chapter 6, Section "Worked examples", Figure tvp:fig:jerk, Table tvp:tab:examples.
Run:  PYTHONPATH=. python chapters/ch06_tvp/jerk_tvp.py          (about 3 min, 1 core)
      PYTHONPATH=. python chapters/ch06_tvp/jerk_tvp.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.polynomial.legendre import legder, legval
from scipy.integrate import solve_ivp
from scipy.optimize import root

from pinnode import (ConstrainedModel, Example, LegendreNet, MLP, Timer, derivatives,
                     legendre_gauss_lobatto, linf, rel_l2, to_tensor, train)

K, C, D = 0.5, 0.05, 0.1
T = 1.0
TC = [0.1, 0.2, 0.3]                     # y(T), y'(T), y''(T)
LAMBDA_TC, N_COLLOC = 200.0, 64
VARIANTS = ["leg-soft", "mlp-soft", "mlp-hard"]
LABELS = {"leg-soft": "LeNN, soft", "mlp-soft": "MLP, soft", "mlp-hard": "MLP, hard"}


def rhs(t, Y):
    return [Y[1], Y[2], -D * Y[2] - C * Y[1] ** 2 - K * Y[0] * np.sin(Y[0])]


def reference(t_eval):
    sol = solve_ivp(rhs, (T, 0.0), TC, method="DOP853", rtol=1e-12, atol=1e-12,
                    t_eval=t_eval[::-1])
    Y = sol.y[:, ::-1]

    def shoot(y0):
        return solve_ivp(rhs, (0.0, T), y0, method="DOP853", rtol=1e-12,
                         atol=1e-12).y[:, -1] - np.array(TC)

    guess = np.array([TC[0] - TC[1] * T + 0.5 * TC[2] * T ** 2, TC[1] - TC[2] * T, TC[2]])
    y0_shoot = root(shoot, guess, method="hybr", tol=1e-14).x
    Ys = solve_ivp(rhs, (0.0, T), y0_shoot, method="DOP853", rtol=1e-12, atol=1e-12,
                   t_eval=t_eval).y

    def var(t, Z):
        y, yp = Z[0], Z[1]
        J = np.array([[0, 1, 0], [0, 0, 1],
                      [-K * (np.sin(y) + y * np.cos(y)), -2 * C * yp, -D]])
        return np.concatenate([rhs(t, Z[:3]), (J @ Z[3:].reshape(3, 3)).ravel()])

    sv = solve_ivp(var, (T, 0.0), np.concatenate([TC, np.eye(3).ravel()]), method="DOP853",
                   rtol=1e-12, atol=1e-12)
    sens = float(np.linalg.norm(sv.y[3:, -1].reshape(3, 3), 2))
    return Y, float(np.max(np.abs(Ys[0] - Y[0]))), sens


def build(variant: str):
    if variant == "leg-soft":
        return LegendreNet(7, (20, 20), 1, activation="tanh", domain=(0.0, T))
    net = MLP(1, 1, width=32, depth=3, activation="tanh", domain=(0.0, T))
    if variant == "mlp-hard":
        return ConstrainedModel(net, lambda t, n: TC[0] + TC[1] * (t - T) + 0.5 * TC[2] * (t - T) ** 2
                                + ((T - t) / T) ** 3 * n)
    return net


def run(ex: Example, variant: str, t_test, y_ref):
    torch.manual_seed(ex.seed)
    model = build(variant)
    hard = variant.endswith("hard")
    t_col = to_tensor(np.real(legendre_gauss_lobatto(ex.pick(N_COLLOC, 16), 0.0, T)),
                      requires_grad=True)
    t_end = to_tensor([T], requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y, d1, d2, d3 = derivatives(model(t_col), t_col, 3)
        r = d3 + D * d2 + C * d1 ** 2 + K * y * torch.sin(y)
        L_r = (r ** 2).mean()
        if hard:
            return L_r, {"residual": L_r}
        yb = derivatives(model(t_end), t_end, 2)
        L_tc = sum(((yb[j] - TC[j]) ** 2).sum() for j in range(3))
        return L_r + LAMBDA_TC * L_tc, {"residual": L_r, "terminal": L_tc}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(4000, 40), lr=1e-3,
                 lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=50,
                 print_every=2000)
    t_req = to_tensor(t_test, requires_grad=True)
    y, d1, d2, d3 = derivatives(model(t_req), t_req, 3)
    res = (d3 + D * d2 + C * d1 ** 2 + K * y * torch.sin(y)).detach().numpy().ravel()
    y_pred = y.detach().numpy().ravel()
    yb = [v.item() for v in derivatives(model(t_end), t_end, 2)]
    return {"hist": hist, "y_pred": y_pred, "rel_l2": rel_l2(y_pred, y_ref),
            "linf": linf(y_pred, y_ref), "err_tc": [abs(yb[j] - TC[j]) for j in range(3)],
            "max_abs_residual": float(np.max(np.abs(res))), "final_loss": hist.final_loss,
            "final_residual_loss": float(hist.parts["residual"][-1]),
            "train_time_s": hist.train_time}


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
    ex = Example("ch06_tvp", "jerk_tvp",
                 description="Third-order TVP: LeNN/MLP with soft or hard terminal conditions",
                 default_seed=1234, extra_args=[RESUME_ARG])
    t_test = np.linspace(0.0, T, 1001)
    Y, shoot_diff, sens = reference(t_test)
    y_ref = Y[0]
    times = []
    for _ in range(5):                      # wall time of the classical backward solve (best of 5)
        with Timer() as tm:
            solve_ivp(rhs, (T, 0.0), TC, method="DOP853", rtol=1e-12, atol=1e-12,
                      t_eval=t_test[::-1])
        times.append(tm.elapsed)
    ref_time = min(times)
    ex.log(f"reference state at t=0: {Y[:, 0]};  shooting agrees to {shoot_diff:.1e};  "
           f"backward sensitivity norm {sens:.3f}")
    # third derivative of the highest Legendre feature P_7 at the end point, in xi and in t
    p7_d3 = float(legval(1.0, legder(np.eye(8)[7], 3)))
    p7_d3_t = p7_d3 * (2.0 / T) ** 3                         # chain rule, xi = 2t/T - 1

    out, rows = {}, []
    for v in VARIANTS:
        ex.log(f"training {v}")
        r = load_cache(ex, v) or run(ex, v, t_test, y_ref)
        save_cache(ex, v, r)
        out[v] = r
        ex.save_history(r["hist"], v)
        rows.append({"variant": v, "rel_l2": r["rel_l2"], "linf": r["linf"],
                     "err_y_T": r["err_tc"][0], "err_dy_T": r["err_tc"][1],
                     "err_ddy_T": r["err_tc"][2], "max_abs_residual": r["max_abs_residual"],
                     "final_residual_loss": r["final_residual_loss"],
                     "initial_residual_loss": float(r["hist"].parts["residual"][0]),
                     "final_loss": r["final_loss"], "train_time_s": r["train_time_s"]})
        ex.log(f"{v}: relL2={r['rel_l2']:.2e} linf={r['linf']:.2e} tc_err={r['err_tc']} "
               f"max|res|={r['max_abs_residual']:.1e} time={r['train_time_s']:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_table({"t": t_test, "y_ref": y_ref, **{f"y_{v}": out[v]["y_pred"] for v in VARIANTS}},
                  "predictions")
    ex.save_results({
        "problem": {"k": K, "c": C, "d": D, "T": T, "terminal_data": TC,
                    "reference": "DOP853 backward, rtol=atol=1e-12",
                    "reference_state_t0": Y[:, 0], "shooting_check": shoot_diff,
                    "backward_sensitivity_norm": sens, "reference_time_s": ref_time,
                    "legendre_P7_d3_at_1_in_xi": p7_d3, "legendre_P7_d3_at_T_in_t": p7_d3_t},
        "hyperparameters": {"leg-soft": "LeNN degree 7 + tanh (20, 20), lambda_tc = 200",
                            "mlp-soft": "tanh MLP 3x32, lambda_tc = 200",
                            "mlp-hard": "tanh MLP 3x32, terminal Taylor trial function",
                            "collocation": f"{N_COLLOC} LGL", "adam_iters": ex.pick(4000, 40),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10)},
        "summary": rows,
        "rel_l2": out["mlp-hard"]["rel_l2"], "linf": out["mlp-hard"]["linf"],
        "train_time_s": sum(r["train_time_s"] for r in rows),
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P          # imported only when figures are drawn

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, t_test, y_ref, label="reference")
    for k, v in enumerate(VARIANTS):
        P.plot_prediction(ax, t_test, out[v]["y_pred"], label=LABELS[v], k=k,
                          markevery=(20 + 30 * k, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.legend(loc="upper center")

    ax = axes[0, 1]
    for k, v in enumerate(VARIANTS):
        P.plot_error(ax, t_test, out[v]["y_pred"] - y_ref, label=LABELS[v], k=k)
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    for k, v in enumerate(VARIANTS):
        h = out[v]["hist"]
        ax.semilogy(h.iters, P.rolling_median(np.asarray(h.parts["residual"]), 9),
                    color=P.SERIES[k], ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.1,
                    label=LABELS[v])
    ax.axvline(out["mlp-hard"]["hist"].switch_iter or 0, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual loss")
    ax.set_ylim(top=1e15)
    ax.legend(loc="upper right")

    ax = axes[1, 1]
    for k, v in enumerate(VARIANTS):
        P.plot_metric(ax, out[v]["hist"], "rel_l2", label=LABELS[v], k=k, show_switch=(k == 0))
    ax.set_ylim(top=1e3)
    ax.legend(loc="upper right")
    P.label_panels(axes)
    ex.savefig(fig, "jerk_tvp")


if __name__ == "__main__":
    main()
