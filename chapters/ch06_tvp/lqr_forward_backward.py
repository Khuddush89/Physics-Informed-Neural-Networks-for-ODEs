"""Forward-backward Hamiltonian system of the LQR, solved by ONE PINN with mixed conditions.

Same LQR problem as lqr_riccati.py (double integrator, Q = I, R = 1, S = I, T = 5,
x0 = (1, 0)).  Pontryagin's principle gives the two-point boundary value problem

    x'      =  A x - B R^{-1} B^T lambda,        x(0) = x0          (forward in time)
    lambda' = -Q x - A^T lambda,                 lambda(T) = S x(T)  (backward in time)

with optimal control u = -R^{-1} B^T lambda.  One tanh MLP with four outputs represents
(x, lambda) through
    x_theta(t)      = x0 + (t/T) N_x(t),
    lambda_theta(t) = S x_theta(T) + ((T - t)/T) N_lambda(t),
so the initial condition and the coupled terminal condition are exact.

References: the analytic solution z(t) = expm(H t) [x0; P(0) x0] (H the Hamiltonian matrix,
P(0) from the Riccati solution), cross-checked with scipy's solve_bvp (tol 1e-10).  The cost
of the PINN control, J = 1/2 x(T)^T S x(T) + 1/2 int (x^T Q x + u^T R u) dt (64-point
Gauss-Legendre), is compared with the optimal value J* = 1/2 x0^T P(0) x0, both along the PINN
trajectory and after applying u_theta(t) in open loop to the true dynamics (DOP853); in the latter
case J >= J* and the excess is of second order in the control error.  Errors on a uniform grid of
1001 points.

Book: Chapter 6, Section "Optimal control: Pontryagin's principle and the LQR",
Figure tvp:fig:fbsystem, Table tvp:tab:lqr.
Run:  PYTHONPATH=. python chapters/ch06_tvp/lqr_forward_backward.py          (about 1 min, 1 core)
      PYTHONPATH=. python chapters/ch06_tvp/lqr_forward_backward.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_bvp, solve_ivp
from scipy.linalg import expm

from pinnode import Example, MLP, derivative, gauss_legendre, linf, rel_l2, to_tensor, train, uniform

A = np.array([[0.0, 1.0], [0.0, 0.0]])
B = np.array([[0.0], [1.0]])
Q = np.eye(2)
R = np.array([[1.0]])
S = np.eye(2)
T = 5.0
X0 = np.array([1.0, 0.0])
WIDTH, DEPTH, N_COLLOC = 32, 3, 100
BRB = B @ np.linalg.solve(R, B.T)
H = np.block([[A, -BRB], [-Q, -A.T]])


def riccati_P0():
    Z = expm(-H * T) @ np.vstack([np.eye(2), S])
    return Z[2:] @ np.linalg.inv(Z[:2])


def exact_solution(t):
    z0 = np.concatenate([X0, riccati_P0() @ X0])
    return np.stack([expm(H * ti) @ z0 for ti in t])          # (len(t), 4): x1, x2, l1, l2


def bvp_solution(t):
    def fun(_, z):
        return H @ z

    def bc(za, zb):
        return np.concatenate([za[:2] - X0, zb[2:] - S @ zb[:2]])

    mesh = np.linspace(0.0, T, 101)
    sol = solve_bvp(fun, bc, mesh, np.zeros((4, mesh.size)), tol=1e-10, max_nodes=100000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol(t).T


def best_time(fn, repeats=5):
    """Best-of-n wall time of a classical reference computation (seconds)."""
    from pinnode import Timer
    best = float("inf")
    for _ in range(repeats):
        with Timer() as tm:
            fn()
        best = min(best, tm.elapsed)
    return best


class FBModel(nn.Module):
    """t -> (x_theta, lambda_theta) with x(0) = x0 and lambda(T) = S x(T) built in."""

    def __init__(self):
        super().__init__()
        self.net = MLP(1, 4, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, T))
        self.register_buffer("x0", torch.as_tensor(X0).reshape(1, 2))
        self.register_buffer("S", torch.as_tensor(S))
        self.register_buffer("t_end", torch.tensor([[T]]))

    def forward(self, t):
        n = self.net(t)
        x = self.x0 + (t / T) * n[:, :2]
        x_T = self.x0 + self.net(self.t_end)[:, :2]
        lam = x_T @ self.S.T + ((T - t) / T) * n[:, 2:]
        return torch.cat([x, lam], dim=1)


def control_cost(x, u, w, x_T):
    run = np.sum(w * (np.einsum("ni,ij,nj->n", x, Q, x) + np.einsum("ni,ij,nj->n", u, R, u)))
    return 0.5 * x_T @ S @ x_T + 0.5 * run


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
    ex = Example("ch06_tvp", "lqr_forward_backward",
                 description="LQR forward-backward Hamiltonian system solved by one PINN",
                 default_seed=1234, extra_args=[RESUME_ARG])
    t_test = np.linspace(0.0, T, 1001)
    z_ex = exact_solution(t_test)
    z_bvp = bvp_solution(t_test)
    ref_time = best_time(lambda: bvp_solution(t_test), repeats=3)
    check = float(np.max(np.abs(z_bvp - z_ex)))
    P0 = riccati_P0()
    J_star = float(0.5 * X0 @ P0 @ X0)
    ex.log(f"lambda(0) = P(0) x0 = {P0 @ X0};  solve_bvp agrees with expm solution to {check:.1e};"
           f"  J* = {J_star:.12f}")

    torch.manual_seed(ex.seed)
    model = FBModel()
    A_t, BRB_t, Q_t = (torch.as_tensor(M) for M in (A, BRB, Q))
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 20), 0.0, T), requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        z = model(t_col)
        x, lam = z[:, :2], z[:, 2:]
        dz = derivative(z, t_col, 1)
        r_x = dz[:, :2] - (x @ A_t.T - lam @ BRB_t.T)
        r_l = dz[:, 2:] - (-x @ Q_t.T - lam @ A_t)
        L_x, L_l = (r_x ** 2).sum(1).mean(), (r_l ** 2).sum(1).mean()
        return L_x + L_l, {"state": L_x, "costate": L_l}

    def eval_fn():
        with torch.no_grad():
            z = model(t_test_t).numpy()
        return {"rel_l2_x": rel_l2(z[:, :2], z_ex[:, :2]), "rel_l2_lambda": rel_l2(z[:, 2:], z_ex[:, 2:])}

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
        z_pr = model(t_test_t).numpy()
        s, w = gauss_legendre(64, 0.0, T)
        zs = model(to_tensor(s)).numpy()
        zT = model(to_tensor([T])).numpy()[0]
    u_pr = -(np.linalg.solve(R, B.T) @ z_pr[:, 2:].T).T
    u_ex = -(np.linalg.solve(R, B.T) @ z_ex[:, 2:].T).T
    us = -(np.linalg.solve(R, B.T) @ zs[:, 2:].T).T
    J_pinn = float(control_cost(zs[:, :2], us, w, zT[:2]))
    # open loop: apply u_theta(t) to the true dynamics x' = A x + B u_theta(t), x(0) = x0;
    # by optimality J(u_theta) >= J* and the excess is second order in u_theta - u*
    Rinv_BT = np.linalg.solve(R, B.T)

    def u_theta(t):
        with torch.no_grad():
            lam = model(to_tensor([float(t)])).numpy()[0, 2:]
        return -Rinv_BT @ lam

    sim = solve_ivp(lambda t, x: A @ x + B @ u_theta(t), (0.0, T), X0, method="DOP853",
                    rtol=1e-11, atol=1e-12, dense_output=True)
    xs_sim = sim.sol(s).T
    J_open_loop = float(control_cost(xs_sim, us, w, sim.y[:, -1]))
    summary = {
        "rel_l2_x": rel_l2(z_pr[:, :2], z_ex[:, :2]), "linf_x": linf(z_pr[:, :2], z_ex[:, :2]),
        "rel_l2_lambda": rel_l2(z_pr[:, 2:], z_ex[:, 2:]), "linf_lambda": linf(z_pr[:, 2:], z_ex[:, 2:]),
        "rel_l2_u": rel_l2(u_pr, u_ex), "linf_u": linf(u_pr, u_ex),
        "lambda0_exact": z_ex[0, 2:], "lambda0_pinn": z_pr[0, 2:],
        "terminal_condition_residual": float(np.max(np.abs(z_pr[-1, 2:] - S @ z_pr[-1, :2]))),
        "J_star": J_star, "J_pinn_control": J_pinn, "rel_err_J": abs(J_pinn - J_star) / J_star,
        "J_open_loop": J_open_loop, "rel_excess_J_open_loop": (J_open_loop - J_star) / J_star,
        "solve_bvp_check": check, "final_loss": hist.final_loss,
        "reference_time_s": ref_time,
    }
    for k_, v in summary.items():
        ex.log(f"{k_} = {v}")
    names = ["x1", "x2", "lambda1", "lambda2"]
    ex.save_table({"t": t_test, **{f"{n}_exact": z_ex[:, j] for j, n in enumerate(names)},
                   **{f"{n}_pinn": z_pr[:, j] for j, n in enumerate(names)}}, "predictions")
    ex.save_results({
        "problem": {"A": A, "B": B, "Q": Q, "R": R, "S": S, "T": T, "x0": X0},
        "hyperparameters": {"network": f"tanh MLP {DEPTH}x{WIDTH}, 4 outputs",
                            "collocation": f"{N_COLLOC} uniform", "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10)},
        **summary, "rel_l2": summary["rel_l2_x"], "linf": summary["linf_x"],
        "train_time_s": hist.train_time,
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P          # imported only when figures are drawn

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, t_test, z_ex[:, 0], label="exact")
    P.plot_reference(ax, t_test, z_ex[:, 1], label=None)
    P.plot_prediction(ax, t_test, z_pr[:, 0], label=r"PINN $x_1$", k=0, markevery=(20, 100))
    P.plot_prediction(ax, t_test, z_pr[:, 1], label=r"PINN $x_2$", k=1, markevery=(60, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("state")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    P.plot_reference(ax, t_test, z_ex[:, 2], label="exact")
    P.plot_reference(ax, t_test, z_ex[:, 3], label=None)
    P.plot_prediction(ax, t_test, z_pr[:, 2], label=r"PINN $\lambda_1$", k=2, markevery=(20, 100))
    P.plot_prediction(ax, t_test, z_pr[:, 3], label=r"PINN $\lambda_2 = -u$", k=3,
                      markevery=(60, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("costate")
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    for j, (lab, k) in enumerate(zip([r"$x_1$", r"$x_2$", r"$\lambda_1$", r"$\lambda_2$"], range(4))):
        P.plot_error(ax, t_test, z_pr[:, j] - z_ex[:, j], label=lab, k=k)
    ax.legend(loc="lower right", ncol=2)

    ax = axes[1, 1]
    P.plot_history(ax, hist, parts=["state", "costate"],
                   labels={"state": "state residual", "costate": "costate residual"})
    ax.legend(loc="lower left")
    P.label_panels(axes)
    ex.savefig(fig, "lqr_forward_backward")


if __name__ == "__main__":
    main()
