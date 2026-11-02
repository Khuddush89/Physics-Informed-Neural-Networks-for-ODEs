"""Riccati terminal value problem of the linear-quadratic regulator, solved by a PINN.

LQR for the double integrator  x1' = x2,  x2' = u:
    A = [[0, 1], [0, 0]],  B = [[0], [1]],  Q = I,  R = 1,  S = I,  horizon T = 5.
Minimise  J = 1/2 x(T)^T S x(T) + 1/2 int_0^T (x^T Q x + u^T R u) dt.
The optimal feedback is u = -R^{-1} B^T P(t) x, where P solves the Riccati TVP

    P' = -A^T P - P A + P B R^{-1} B^T P - Q,    P(T) = S,

and the optimal cost from x(0) = x0 is J* = 1/2 x0^T P(0) x0 (x0 = (1, 0)).

Exact solution: with the Hamiltonian matrix H = [[A, -B R^{-1} B^T], [-Q, -A^T]],
[X(t); L(t)] = expm(H (t - T)) [I; S] and P(t) = L(t) X(t)^{-1}; cross-checked by backward
DOP853 integration (rtol = atol = 1e-12).  As T - t grows, P(t) approaches the stabilising
solution of the algebraic Riccati equation, P_inf = [[sqrt 3, 1], [1, sqrt 3]].

PINN: tanh MLP with three outputs (p11, p12, p22); the symmetric trial matrix
P_theta(t) = S + ((T - t)/T) N_theta(t) satisfies P(T) = S exactly; 100 uniform collocation
points; Adam followed by L-BFGS.  Errors on a uniform grid of 1001 points.

Book: Chapter 6, Section "Optimal control: Pontryagin's principle and the LQR",
Figure tvp:fig:riccati, Table tvp:tab:lqr.
Run:  PYTHONPATH=. python chapters/ch06_tvp/lqr_riccati.py          (about 1 min, 1 core)
      PYTHONPATH=. python chapters/ch06_tvp/lqr_riccati.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm, solve_continuous_are

from pinnode import ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor, train, uniform

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
Q = np.eye(2)
R = np.array([[1.0]])
S = np.eye(2)
T = 5.0
X0 = np.array([1.0, 0.0])
WIDTH, DEPTH, N_COLLOC = 32, 3, 100
IDX = [(0, 0), (0, 1), (1, 1)]          # stored entries of the symmetric matrix
NAMES = ["p11", "p12", "p22"]


def hamiltonian():
    BRB = B @ np.linalg.solve(R, B.T)
    return np.block([[A, -BRB], [-Q, -A.T]])


def exact_P(t):
    """P(t) = L X^{-1} from the Hamiltonian flow; returns array (len(t), 2, 2)."""
    H = hamiltonian()
    out = np.empty((len(t), 2, 2))
    for i, ti in enumerate(t):
        Z = expm(H * (ti - T)) @ np.vstack([np.eye(2), S])
        out[i] = Z[2:] @ np.linalg.inv(Z[:2])
    return 0.5 * (out + out.transpose(0, 2, 1))


def riccati_rhs(t, p):
    Pm = np.array([[p[0], p[1]], [p[1], p[2]]])
    dP = -A.T @ Pm - Pm @ A + Pm @ B @ np.linalg.solve(R, B.T) @ Pm - Q
    return [dP[0, 0], dP[0, 1], dP[1, 1]]


def backward_P(t):
    sol = solve_ivp(riccati_rhs, (T, 0.0), [S[0, 0], S[0, 1], S[1, 1]], method="DOP853",
                    rtol=1e-12, atol=1e-12, t_eval=t[::-1])
    return sol.y[:, ::-1].T


def best_time(fn, repeats=5):
    """Best-of-n wall time of a classical reference computation (seconds)."""
    from pinnode import Timer
    best = float("inf")
    for _ in range(repeats):
        with Timer() as tm:
            fn()
        best = min(best, tm.elapsed)
    return best


def sym(entries):
    """(M, 3) entries -> (M, 2, 2) symmetric matrices."""
    p11, p12, p22 = entries[:, 0], entries[:, 1], entries[:, 2]
    return torch.stack([torch.stack([p11, p12], -1), torch.stack([p12, p22], -1)], -2)


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
    ex = Example("ch06_tvp", "lqr_riccati",
                 description="LQR Riccati terminal value problem (double integrator) by a PINN",
                 default_seed=1234, extra_args=[RESUME_ARG])
    t_test = np.linspace(0.0, T, 1001)
    P_ex = exact_P(t_test)
    P_ex_e = np.stack([P_ex[:, i, j] for i, j in IDX], axis=1)
    P_bw = backward_P(t_test)
    ref_time = best_time(lambda: backward_P(t_test))
    P_inf = solve_continuous_are(A, B, Q, R)
    check = float(np.max(np.abs(P_bw - P_ex_e)))
    ex.log(f"exact P(0) = {P_ex[0].tolist()};  backward DOP853 agrees to {check:.1e};  "
           f"P_inf = {P_inf.tolist()}")

    torch.manual_seed(ex.seed)
    S_t = to_tensor(np.array([[S[0, 0], S[0, 1], S[1, 1]]]))
    model = ConstrainedModel(MLP(1, 3, width=WIDTH, depth=DEPTH, activation="tanh",
                                 domain=(0.0, T)),
                             lambda t, n: S_t + ((T - t) / T) * n)
    A_t, BRB_t, Q_t = (torch.as_tensor(M) for M in (A, B @ np.linalg.solve(R, B.T), Q))
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 20), 0.0, T), requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        e = model(t_col)
        Pm, dPm = sym(e), sym(derivative(e, t_col, 1))
        Res = dPm + A_t.T @ Pm + Pm @ A_t - Pm @ BRB_t @ Pm + Q_t
        r = torch.stack([Res[:, i, j] for i, j in IDX], dim=1)
        L_r = (r ** 2).sum(dim=1).mean()
        return L_r, {"residual": L_r}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), P_ex_e)}

    cache = load_cache(ex, "pinn")
    if cache is None:
        hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(4000, 60), lr=1e-3,
                     lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=50,
                     print_every=2000)
        save_cache(ex, "pinn", {"state": model.state_dict(), "hist": hist})
    else:
        model.load_state_dict(cache["state"])
        hist = cache["hist"]
    ex.save_history(hist, "pinn")
    with torch.no_grad():
        P_pr = model(t_test_t).numpy()
    J_exact = 0.5 * X0 @ P_ex[0] @ X0
    P0_pinn = np.array([[P_pr[0, 0], P_pr[0, 1]], [P_pr[0, 1], P_pr[0, 2]]])
    J_pinn = 0.5 * X0 @ P0_pinn @ X0
    K0_exact = np.linalg.solve(R, B.T @ P_ex[0]).ravel()
    K0_pinn = np.linalg.solve(R, B.T @ P0_pinn).ravel()
    per_entry = {n: {"rel_l2": rel_l2(P_pr[:, j], P_ex_e[:, j]), "linf": linf(P_pr[:, j], P_ex_e[:, j])}
                 for j, n in enumerate(NAMES)}
    summary = {
        "rel_l2": rel_l2(P_pr, P_ex_e), "linf": linf(P_pr, P_ex_e), "per_entry": per_entry,
        "J_exact": float(J_exact), "J_pinn": float(J_pinn),
        "rel_err_J": float(abs(J_pinn - J_exact) / J_exact),
        "K0_exact": K0_exact, "K0_pinn": K0_pinn,
        "P0_exact": P_ex[0], "P0_pinn": P0_pinn, "P_inf": P_inf,
        "dist_P0_to_Pinf": float(np.max(np.abs(P_ex[0] - P_inf))),
        "backward_dop853_check": check, "final_loss": hist.final_loss,
        "reference_time_s": ref_time,
    }
    for k_, v in summary.items():
        ex.log(f"{k_} = {v}")
    ex.save_table({"t": t_test, **{f"{n}_exact": P_ex_e[:, j] for j, n in enumerate(NAMES)},
                   **{f"{n}_pinn": P_pr[:, j] for j, n in enumerate(NAMES)}}, "predictions")
    ex.save_results({
        "problem": {"A": A, "B": B, "Q": Q, "R": R, "S": S, "T": T, "x0": X0},
        "hyperparameters": {"network": f"tanh MLP {DEPTH}x{WIDTH}, 3 outputs",
                            "collocation": f"{N_COLLOC} uniform", "adam_iters": ex.pick(4000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10)},
        **summary, "train_time_s": hist.train_time,
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P          # imported only when figures are drawn

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    labels = [r"$p_{11}$", r"$p_{12}$", r"$p_{22}$"]
    for j in range(3):
        P.plot_reference(ax, t_test, P_ex_e[:, j], label="exact" if j == 0 else None)
        P.plot_prediction(ax, t_test, P_pr[:, j], label=f"PINN {labels[j]}", k=j,
                          markevery=(25 * j + 20, 100))
    ax.axhline(np.sqrt(3.0), color=P.COLORS["muted"], lw=0.6, ls=":")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$P(t)$ entries")
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    for j in range(3):
        P.plot_error(ax, t_test, P_pr[:, j] - P_ex_e[:, j], label=labels[j], k=j)

    ax = axes[1, 0]
    P.plot_history(ax, hist, parts=None, total=True, ylabel="residual loss")

    ax = axes[1, 1]
    P.plot_metric(ax, hist, "rel_l2", ylabel=r"relative $L^2$ error of $P$")
    P.label_panels(axes)
    ex.savefig(fig, "lqr_riccati")


if __name__ == "__main__":
    main()
