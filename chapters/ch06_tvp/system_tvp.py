"""System of two third-order terminal value problems (exercise of Chapter 6, with solution).

    y1''' = -k1 y1 - c1 y1^2 y2' - d1 y1'' + e1 y2 y1',
    y2''' = -k2 y2 + c2 (y1')^2 - d2 y2' + e2 y1^3,            t in [0, 1],
    y1(1) = 0, y1'(1) = 0.5, y1''(1) = 0,   y2(1) = 1, y2'(1) = 0, y2''(1) = -0.2,
    k1 = 2, k2 = 1.5, c1 = 0.4, c2 = 0.5, d1 = 0.3, d2 = 0.25, e1 = 0.1, e2 = 0.2.

Reference: the equivalent first-order system of dimension six integrated backward from t = 1
with DOP853 (rtol = atol = 1e-12), cross-checked with Radau; the norm of the backward
sensitivity matrix d(state at 0)/d(state at 1) is computed from the variational equation.

PINN: tanh MLP with two outputs and the terminal Taylor trial functions
    y_j(t) = c_j0 + c_j1 (t - 1) + c_j2 (t - 1)^2 / 2 + (1 - t)^3 N_j(t),   j = 1, 2,
which satisfy all six terminal conditions exactly; 100 Legendre-Gauss-Lobatto points;
Adam followed by L-BFGS.  Errors on a uniform grid of 1001 points.

Book: Chapter 6, Exercise tvp:ex:system and its solution in Appendix C.
Run:  PYTHONPATH=. python chapters/ch06_tvp/system_tvp.py          (about 1.5 min, 1 core)
      PYTHONPATH=. python chapters/ch06_tvp/system_tvp.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import (ConstrainedModel, Example, MLP, derivatives, legendre_gauss_lobatto, linf,
                     rel_l2, to_tensor, train)

K1, K2, C1, C2, D1, D2, E1, E2 = 2.0, 1.5, 0.4, 0.5, 0.3, 0.25, 0.1, 0.2
T = 1.0
TC = np.array([[0.0, 0.5, 0.0],        # y1(T), y1'(T), y1''(T)
               [1.0, 0.0, -0.2]])      # y2(T), y2'(T), y2''(T)
WIDTH, DEPTH, N_COLLOC = 32, 3, 100


def rhs(t, Y):
    y1, y1p, y1pp, y2, y2p, y2pp = Y
    return [y1p, y1pp, -K1 * y1 - C1 * y1 ** 2 * y2p - D1 * y1pp + E1 * y2 * y1p,
            y2p, y2pp, -K2 * y2 + C2 * y1p ** 2 - D2 * y2p + E2 * y1 ** 3]


def jacobian(Y):
    y1, y1p, y1pp, y2, y2p, y2pp = Y
    J = np.zeros((6, 6))
    J[0, 1] = J[1, 2] = J[3, 4] = J[4, 5] = 1.0
    J[2] = [-K1 - 2 * C1 * y1 * y2p, E1 * y2, -D1, E1 * y1p, -C1 * y1 ** 2, 0.0]
    J[5] = [3 * E2 * y1 ** 2, 2 * C2 * y1p, 0.0, -K2, -D2, 0.0]
    return J


def reference(t_eval):
    YT = [TC[0, 0], TC[0, 1], TC[0, 2], TC[1, 0], TC[1, 1], TC[1, 2]]
    out = {}
    for method in ("DOP853", "Radau"):
        sol = solve_ivp(rhs, (T, 0.0), YT, method=method, rtol=1e-12, atol=1e-12,
                        t_eval=t_eval[::-1])
        out[method] = sol.y[:, ::-1]

    def var(t, Z):
        return np.concatenate([rhs(t, Z[:6]), (jacobian(Z[:6]) @ Z[6:].reshape(6, 6)).ravel()])

    sv = solve_ivp(var, (T, 0.0), np.concatenate([YT, np.eye(6).ravel()]), method="DOP853",
                   rtol=1e-12, atol=1e-12)
    sens = float(np.linalg.norm(sv.y[6:, -1].reshape(6, 6), 2))
    return out["DOP853"], float(np.max(np.abs(out["Radau"] - out["DOP853"]))), sens


def best_time(fn, repeats=5):
    """Best-of-n wall time of a classical reference computation (seconds)."""
    from pinnode import Timer
    best = float("inf")
    for _ in range(repeats):
        with Timer() as tm:
            fn()
        best = min(best, tm.elapsed)
    return best


# ----------------------------------------------------------------------------- caching
# The trained weights and the history are cached in the output folder; with --resume a rerun
# (for example after the process was interrupted) reuses them. Times are those of the original run.
RESUME_ARG = (["--resume"], {"action": "store_true",
                             "help": "reuse the training cached in the output folder"})


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
    ex = Example("ch06_tvp", "system_tvp",
                 description="Two coupled third-order TVPs: backward reference and hard-constrained PINN",
                 default_seed=1234, extra_args=[RESUME_ARG])
    t_test = np.linspace(0.0, T, 1001)
    Y, check, sens = reference(t_test)
    y1_ref, y2_ref = Y[0], Y[3]
    YT = [TC[0, 0], TC[0, 1], TC[0, 2], TC[1, 0], TC[1, 1], TC[1, 2]]
    ref_time = best_time(lambda: solve_ivp(rhs, (T, 0.0), YT, method="DOP853", rtol=1e-12,
                                           atol=1e-12, t_eval=t_test[::-1]))
    ex.log(f"state at t=0: {Y[:, 0]};  Radau vs DOP853: {check:.1e};  backward sensitivity "
           f"norm {sens:.3f}")

    torch.manual_seed(ex.seed)
    c = torch.as_tensor(TC)

    def transform(t, n):
        s = t - T
        base = c[:, 0] + c[:, 1] * s + 0.5 * c[:, 2] * s ** 2          # (M, 2) by broadcasting
        return base + ((T - t) / T) ** 3 * n

    model = ConstrainedModel(MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh",
                                 domain=(0.0, T)), transform)
    t_col = to_tensor(np.real(legendre_gauss_lobatto(ex.pick(N_COLLOC, 16), 0.0, T)),
                      requires_grad=True)
    t_test_t = to_tensor(t_test)
    ref = np.stack([y1_ref, y2_ref], axis=1)

    def loss_fn():
        y, d1, d2, d3 = derivatives(model(t_col), t_col, 3)
        y1, y2 = y[:, :1], y[:, 1:2]
        r1 = d3[:, :1] + K1 * y1 + C1 * y1 ** 2 * d1[:, 1:2] + D1 * d2[:, :1] - E1 * y2 * d1[:, :1]
        r2 = d3[:, 1:2] + K2 * y2 - C2 * d1[:, :1] ** 2 + D2 * d1[:, 1:2] - E2 * y1 ** 3
        L1, L2 = (r1 ** 2).mean(), (r2 ** 2).mean()
        return L1 + L2, {"residual_1": L1, "residual_2": L2}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), ref)}

    cache = load_cache(ex, "pinn")
    if cache is None:
        hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(5000, 60), lr=1e-3,
                     lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=50,
                     print_every=2000)
        save_cache(ex, "pinn", {"state": model.state_dict(), "hist": hist})
    else:
        model.load_state_dict(cache["state"])
        hist = cache["hist"]
    ex.save_history(hist, "pinn")
    with torch.no_grad():
        pr = model(t_test_t).numpy()
    summary = {
        "state_at_0_ref": Y[:, 0], "radau_check": check, "backward_sensitivity_norm": sens,
        "rel_l2_y1": rel_l2(pr[:, 0], y1_ref), "linf_y1": linf(pr[:, 0], y1_ref),
        "rel_l2_y2": rel_l2(pr[:, 1], y2_ref), "linf_y2": linf(pr[:, 1], y2_ref),
        "y1_0_pinn": float(pr[0, 0]), "y2_0_pinn": float(pr[0, 1]),
        "final_loss": hist.final_loss, "reference_time_s": ref_time,
    }
    for k_, v in summary.items():
        ex.log(f"{k_} = {v}")
    ex.save_table({"t": t_test, "y1_ref": y1_ref, "y2_ref": y2_ref, "y1_pinn": pr[:, 0],
                   "y2_pinn": pr[:, 1]}, "predictions")
    ex.save_results({
        "problem": {"k1": K1, "k2": K2, "c1": C1, "c2": C2, "d1": D1, "d2": D2, "e1": E1, "e2": E2,
                    "T": T, "terminal_data": TC},
        "hyperparameters": {"network": f"tanh MLP {DEPTH}x{WIDTH}, 2 outputs",
                            "collocation": f"{N_COLLOC} LGL", "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10)},
        **summary, "rel_l2": rel_l2(pr, ref), "linf": linf(pr, ref),
        "train_time_s": hist.train_time,
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P          # imported only when figures are drawn

    fig, axes = P.figure(1, 2, aspect=0.78)
    ax = axes[0]
    P.plot_reference(ax, t_test, y1_ref, label="reference")
    P.plot_reference(ax, t_test, y2_ref, label=None)
    P.plot_prediction(ax, t_test, pr[:, 0], label=r"PINN $y_1$", k=0, markevery=(30, 100))
    P.plot_prediction(ax, t_test, pr[:, 1], label=r"PINN $y_2$", k=1, markevery=(70, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("solution")
    ax.legend(loc="best")
    ax = axes[1]
    P.plot_error(ax, t_test, pr[:, 0] - y1_ref, label=r"$y_1$", k=0)
    P.plot_error(ax, t_test, pr[:, 1] - y2_ref, label=r"$y_2$", k=1)
    P.label_panels(axes)
    ex.savefig(fig, "system_tvp")


if __name__ == "__main__":
    main()
