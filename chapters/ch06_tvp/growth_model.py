"""Optimal growth with a terminal capital condition (finite-horizon Ramsey-Cass-Koopmans model).

A planner chooses consumption c(t) > 0 to maximise  int_0^T e^{-rho t} c^{1-sigma}/(1-sigma) dt
subject to  k' = A k^alpha - delta k - c,  k(0) = k0  and the terminal capital requirement
k(T) = kT.  Pontryagin's principle (current-value Hamiltonian, costate mu = c^{-sigma}) gives
the Keynes-Ramsey rule  c'/c = (alpha A k^{alpha-1} - delta - rho)/sigma.  With z = ln c:

    k' = A k^alpha - delta k - e^z,                      k(0) = k0,  k(T) = kT,
    z' = (alpha A k^{alpha-1} - delta - rho) / sigma.

Parameters: A = 1, alpha = 0.3, delta = 0.05, rho = 0.03, sigma = 2, T = 50, k0 = kT = 1
(the planner must hand over the capital stock it inherited).  Steady state:
k* = (alpha A/(delta + rho))^{1/(1-alpha)} = 6.6076, c* = A k*^alpha - delta k* = 1.4316.

Reference: scipy's solve_bvp (tol 1e-10).  The ill-conditioning of shooting on c(0) is
quantified by the sensitivity dk(T)/dc(0), obtained from the variational equation along the
reference solution.

PINN: tanh MLP with two outputs and the trial functions
    k_theta(t) = k0 + (kT - k0) t/T + 4 k* (t/T)(1 - t/T) N_k(t),   z_theta(t) = ln c* + N_z(t),
so both capital conditions hold exactly; residuals are multiplied by T (and the capital
residual divided by k*) so that both are O(1).  Errors on a uniform grid of 1001 points.

Book: Chapter 6, Section "Terminal conditions in economics", Figure tvp:fig:growth.
Run:  PYTHONPATH=. python chapters/ch06_tvp/growth_model.py          (about 1 min, 1 core)
      PYTHONPATH=. python chapters/ch06_tvp/growth_model.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_bvp, solve_ivp

from pinnode import ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor, train, uniform

A_TFP, ALPHA, DELTA, RHO, SIGMA = 1.0, 0.3, 0.05, 0.03, 2.0
T, K0, KT = 50.0, 1.0, 1.0
K_STAR = (ALPHA * A_TFP / (DELTA + RHO)) ** (1.0 / (1.0 - ALPHA))
C_STAR = A_TFP * K_STAR ** ALPHA - DELTA * K_STAR
WIDTH, DEPTH, N_COLLOC = 32, 3, 150


def rhs(t, Y):
    k, z = Y
    k = np.maximum(k, 1e-10)
    return np.vstack([A_TFP * k ** ALPHA - DELTA * k - np.exp(z),
                      (ALPHA * A_TFP * k ** (ALPHA - 1.0) - DELTA - RHO) / SIGMA])


def reference(t_eval):
    mesh = np.linspace(0.0, T, 200)
    guess = np.vstack([K0 + (KT - K0) * mesh / T, np.full_like(mesh, np.log(C_STAR))])
    sol = solve_bvp(rhs, lambda ya, yb: np.array([ya[0] - K0, yb[0] - KT]), mesh, guess,
                    tol=1e-10, max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.sol(t_eval), sol


def best_time(fn, repeats=5):
    """Best-of-n wall time of a classical reference computation (seconds)."""
    from pinnode import Timer
    best = float("inf")
    for _ in range(repeats):
        with Timer() as tm:
            fn()
        best = min(best, tm.elapsed)
    return best


def shooting_sensitivity(z0):
    """dk(T)/dc(0) along the optimal path (variational equation, DOP853)."""
    def var(t, Z):
        k, z, a, b = Z
        f = rhs(t, np.array([k, z]))[:, 0]
        J = np.array([[ALPHA * A_TFP * k ** (ALPHA - 1.0) - DELTA, -np.exp(z)],
                      [ALPHA * (ALPHA - 1.0) * A_TFP * k ** (ALPHA - 2.0) / SIGMA, 0.0]])
        return np.concatenate([f, J @ np.array([a, b])])

    sol = solve_ivp(var, (0.0, T), [K0, z0, 0.0, 1.0], method="DOP853", rtol=1e-12, atol=1e-12)
    return float(sol.y[2, -1] / np.exp(z0))        # dk(T)/dz(0) * dz(0)/dc(0)


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
    ex = Example("ch06_tvp", "growth_model",
                 description="Finite-horizon optimal growth with terminal capital (PINN vs solve_bvp)",
                 default_seed=1234, extra_args=[RESUME_ARG])
    t_test = np.linspace(0.0, T, 1001)
    (k_ref, z_ref), sol = reference(t_test)
    ref_time = best_time(lambda: reference(t_test), repeats=3)
    c_ref = np.exp(z_ref)
    sens = shooting_sensitivity(z_ref[0])
    ex.log(f"k* = {K_STAR:.6f}, c* = {C_STAR:.6f};  reference c(0) = {c_ref[0]:.10f}, "
           f"max k = {k_ref.max():.6f};  dk(T)/dc(0) = {sens:.3e}")

    torch.manual_seed(ex.seed)
    net = MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, T))

    def transform(t, n):
        tau = t / T
        k = K0 + (KT - K0) * tau + 4.0 * K_STAR * tau * (1.0 - tau) * n[:, :1]
        z = np.log(C_STAR) + n[:, 1:2]
        return torch.cat([k, z], dim=1)

    model = ConstrainedModel(net, transform)
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 20), 0.0, T), requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        out = model(t_col)
        k, z = out[:, :1], out[:, 1:2]
        dk, dz = derivative(out, t_col, 1).split(1, dim=1)
        kp = torch.clamp(k, min=1e-3)
        r_k = (dk - (A_TFP * kp ** ALPHA - DELTA * kp - torch.exp(z))) * T / K_STAR
        r_z = (dz - (ALPHA * A_TFP * kp ** (ALPHA - 1.0) - DELTA - RHO) / SIGMA) * T
        L_k, L_z = (r_k ** 2).mean(), (r_z ** 2).mean()
        return L_k + L_z, {"capital": L_k, "euler": L_z}

    def eval_fn():
        with torch.no_grad():
            out = model(t_test_t).numpy()
        return {"rel_l2_k": rel_l2(out[:, 0], k_ref), "rel_l2_c": rel_l2(np.exp(out[:, 1]), c_ref)}

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
        out = model(t_test_t).numpy()
    k_pr, c_pr = out[:, 0], np.exp(out[:, 1])
    summary = {
        "rel_l2_k": rel_l2(k_pr, k_ref), "linf_k": linf(k_pr, k_ref),
        "rel_l2_c": rel_l2(c_pr, c_ref), "linf_c": linf(c_pr, c_ref),
        "c0_ref": float(c_ref[0]), "c0_pinn": float(c_pr[0]),
        "rel_err_c0": float(abs(c_pr[0] - c_ref[0]) / c_ref[0]),
        "cT_ref": float(c_ref[-1]), "max_k_ref": float(k_ref.max()),
        "t_max_k_ref": float(t_test[np.argmax(k_ref)]),
        "k_star": K_STAR, "c_star": C_STAR, "dkT_dc0": sens,
        "solve_bvp_nodes": int(sol.x.size), "final_loss": hist.final_loss,
        "reference_time_s": ref_time,
    }
    for k_, v in summary.items():
        ex.log(f"{k_} = {v}")
    ex.save_table({"t": t_test, "k_ref": k_ref, "c_ref": c_ref, "k_pinn": k_pr, "c_pinn": c_pr},
                  "predictions")
    ex.save_results({
        "problem": {"A": A_TFP, "alpha": ALPHA, "delta": DELTA, "rho": RHO, "sigma": SIGMA,
                    "T": T, "k0": K0, "kT": KT},
        "hyperparameters": {"network": f"tanh MLP {DEPTH}x{WIDTH}, 2 outputs",
                            "collocation": f"{N_COLLOC} uniform", "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10)},
        **summary, "rel_l2": summary["rel_l2_k"], "linf": summary["linf_k"],
        "train_time_s": hist.train_time,
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P          # imported only when figures are drawn

    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    ax.axhline(K_STAR, color=P.COLORS["muted"], lw=0.7, ls=":", label=r"steady state $k^*$")
    P.plot_reference(ax, t_test, k_ref, label="reference")
    P.plot_prediction(ax, t_test, k_pr, label="PINN", k=0, markevery=(40, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("capital $k(t)$")
    ax.legend(loc="lower center")

    ax = axes[0, 1]
    ax.axhline(C_STAR, color=P.COLORS["muted"], lw=0.7, ls=":", label=r"steady state $c^*$")
    P.plot_reference(ax, t_test, c_ref, label="reference")
    P.plot_prediction(ax, t_test, c_pr, label="PINN", k=1, markevery=(70, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("consumption $c(t)$")
    ax.legend(loc="upper left")

    ax = axes[1, 0]
    P.plot_error(ax, t_test, k_pr - k_ref, label="capital", k=0)
    P.plot_error(ax, t_test, c_pr - c_ref, label="consumption", k=1)

    ax = axes[1, 1]
    P.plot_history(ax, hist, parts=["capital", "euler"],
                   labels={"capital": "capital residual", "euler": "Euler residual"})
    ax.legend(loc="lower left")
    P.label_panels(axes)
    ex.savefig(fig, "growth_model")


if __name__ == "__main__":
    main()
