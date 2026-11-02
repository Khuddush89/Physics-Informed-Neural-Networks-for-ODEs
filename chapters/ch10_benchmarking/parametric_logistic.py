"""Parametric logistic family: one PINN y_theta(t, r) versus repeated classical solves.

Book: Chapter 10 (Benchmarking against classical solvers), Section "Where PINNs can pay off",
Figure "parametric family: accuracy and cumulative cost" and the break-even table.

Family: y' = r y (1 - y), y(0) = 0.1, t in [0, 2], r in [1, 3];
exact y(t; r) = y0 e^{rt} / (1 + y0 (e^{rt} - 1)).

PINN: y_theta(t, r) = y0 + t N_theta(t, r) (exact initial condition for every r), tanh MLP
with input scaling of (t, r) to [-1, 1]^2, collocation on the tensor grid of 41 (t) x 21 (r)
uniform points, Adam (cosine 1e-3 -> 1e-5, 10000 steps) + L-BFGS (<= 1000 iterations), seed
1234; two networks: Q1 (32 x 3) and Q2 (64 x 3). Test set: 101 equispaced members r_k in [1, 3]
on the 1001-point grid in t; the relative L2 error is computed member by member.

Two kinds of queries are costed (CPU time, median of 5 timed calls after one warm-up):
  trajectory - y(t; r) on the 1001-point grid (PINN: one forward pass on 1001 points;
               classical: solve_ivp with t_eval);
  point      - y(T; r) only (PINN: one forward pass at one point; classical: solve_ivp without
               dense output).
Classical baselines:
  repeated solves - RK45 and DOP853 per member, at the loosest tolerance of the half-decade grid
                    whose worst member error does not exceed the worst error of the PINN;
  Chebyshev       - surrogate in r: DOP853 solves (rtol = atol = 1e-10) at n_r Chebyshev-Lobatto
                    points of [1, 3] and barycentric interpolation in r, with the smallest n_r
                    whose worst member error does not exceed the worst error of the PINN.
Break-even member count of a PINN against repeated solves: n* = C_train / (c_solve - c_eval)
(infinite when evaluating the network is not cheaper than a solve).

Run:  PYTHONPATH=. python chapters/ch10_benchmarking/parametric_logistic.py          (about 3 min CPU)
      PYTHONPATH=. python chapters/ch10_benchmarking/parametric_logistic.py --quick  (smoke test)
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import math

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import ConstrainedModel, Example, MLP, count_parameters, gradient, to_tensor

Y0, T, R_MIN, R_MAX = 0.1, 2.0, 1.0, 3.0
N_T, N_R_COL, N_T_COL, N_MEMBERS = 1001, 21, 41, 101
CONFIGS = [dict(tag="Q1", width=32, depth=3, adam=10000, lbfgs=1000),
           dict(tag="Q2", width=64, depth=3, adam=10000, lbfgs=1000)]
CHEB_N = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]


def exact(t, r):
    e = np.exp(np.multiply.outer(r, t))
    return Y0 * e / (1.0 + Y0 * (e - 1.0))


def member_errors(Y: np.ndarray, Yref: np.ndarray) -> np.ndarray:
    """Relative L2 error of every member (rows) on the test grid."""
    return np.linalg.norm(Y - Yref, axis=1) / np.linalg.norm(Yref, axis=1)


def solve_member(r: float, t, method: str, rtol: float):
    """Trajectory on the grid t (t_eval) or, for t=None, the final value y(T) only."""
    sol = solve_ivp(lambda s, y: r * y * (1.0 - y), (0.0, T), [Y0], method=method, t_eval=t,
                    rtol=rtol, atol=rtol)
    return sol.y[0] if t is not None else sol.y[0, -1]


def solve_all(rs, t, method, rtol):
    return np.array([solve_member(r, t, method, rtol) for r in rs])


def cheb_lobatto(n: int):
    j = np.arange(n)
    x = 0.5 * (R_MIN + R_MAX) - 0.5 * (R_MAX - R_MIN) * np.cos(np.pi * j / (n - 1))
    w = (-1.0) ** j
    w[0] *= 0.5
    w[-1] *= 0.5
    return x, w


def barycentric(nodes, weights, values, r_eval):
    """Barycentric interpolation in r of the rows of ``values`` (n_nodes x n_t)."""
    d = r_eval[:, None] - nodes[None, :]
    hit = np.isclose(d, 0.0, atol=1e-14)
    d[hit] = 1.0
    c = weights[None, :] / d
    out = (c @ values) / c.sum(axis=1, keepdims=True)
    rows, cols = np.nonzero(hit)
    out[rows] = values[cols]
    return out


def build_family_model(cfg: dict) -> torch.nn.Module:
    net = MLP(2, 1, width=cfg["width"], depth=cfg["depth"], activation="tanh",
              domain=[(0.0, T), (R_MIN, R_MAX)])
    return ConstrainedModel(net, lambda x, n: Y0 + x[:, :1] * n)


def train_family(ex: Example, cfg: dict, Yref: np.ndarray, r_test: np.ndarray, t: np.ndarray):
    """Train one network for the whole family; returns run_pinn records plus the weights."""
    torch.manual_seed(ex.seed)
    model = build_family_model(cfg)
    tt, rr = np.meshgrid(np.linspace(0.0, T, N_T_COL), np.linspace(R_MIN, R_MAX, N_R_COL))
    X = to_tensor(np.column_stack([tt.ravel(), rr.ravel()]), requires_grad=True)

    def loss_fn():
        y = model(X)
        dy = gradient(y, X)[:, :1]
        res = dy - X[:, 1:2] * y * (1.0 - y)
        L = (res ** 2).mean()
        return L, {"residual": L}

    T_grid, R_grid = np.meshgrid(t, r_test)
    X_test = to_tensor(np.column_stack([T_grid.ravel(), R_grid.ravel()]))

    def predict_all():
        with torch.no_grad():
            return model(X_test).numpy().reshape(len(r_test), N_T)

    out = B.run_pinn(model, loss_fn, X.shape[0],
                     lambda: float(np.max(member_errors(predict_all(), Yref))), predict_all,
                     adam_iters=ex.pick(cfg["adam"], 60), lbfgs_iters=ex.pick(cfg["lbfgs"], 5),
                     eval_every=ex.pick(250, 20), print_every=ex.pick(5000, 50))
    out["n_col"] = X.shape[0]
    out["state_dict"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return out


def main():
    ex = Example("ch10_benchmarking", "parametric_logistic",
                 description="Parametric logistic family: one PINN vs repeated classical solves",
                 default_seed=1234, extra_args=B.RESUME_ARGS)
    reps = ex.pick(B.REPEATS, 1)
    t = np.linspace(0.0, T, N_T)
    r_test = np.linspace(R_MIN, R_MAX, N_MEMBERS if not ex.quick else 11)
    Yref = exact(t, r_test)
    rtols = B.RTOLS if not ex.quick else B.RTOLS[::6]

    # ------------------------------------------------------------- classical: errors per rtol
    sweep = []
    for method in ("RK45", "DOP853"):
        for rtol in rtols:
            e = member_errors(solve_all(r_test, t, method, rtol), Yref)
            sweep.append({"method": method, "rtol": rtol, "worst_rel_l2": float(e.max()),
                          "median_rel_l2": float(np.median(e)), "errors": e})
    ex.save_table([{k: v for k, v in s.items() if k != "errors"} for s in sweep], "classical_sweep")
    cost_cache = {}

    def classical_cost(method, rtol):
        """CPU time per member for trajectory and point queries (median of ``reps``)."""
        if (method, rtol) not in cost_cache:
            _, c_traj, w_traj = B.timed(lambda: solve_all(r_test, t, method, rtol), repeats=reps)
            _, c_pt, w_pt = B.timed(lambda: solve_all(r_test, None, method, rtol), repeats=reps)
            n = len(r_test)
            cost_cache[(method, rtol)] = {"traj_cpu": c_traj / n, "traj_wall": w_traj / n,
                                          "point_cpu": c_pt / n, "point_wall": w_pt / n}
        return cost_cache[(method, rtol)]

    # ------------------------------------------------------------- classical Chebyshev surrogate
    cheb_rows = []
    for n in (CHEB_N if not ex.quick else CHEB_N[:3]):
        nodes, w = cheb_lobatto(n)
        vals, c_build, w_build = B.timed(lambda: solve_all(nodes, t, "DOP853", 1e-10), repeats=reps)
        Yc = barycentric(nodes, w, vals, r_test)
        e = member_errors(Yc, Yref)
        _, c_ev, _ = B.timed(lambda: barycentric(nodes, w, vals, np.array([2.0])), repeats=reps)
        _, c_evp, _ = B.timed(lambda: barycentric(nodes, w, vals[:, -1:], np.array([2.0])), repeats=reps)
        cheb_rows.append({"n_r": n, "worst_rel_l2": float(e.max()), "median_rel_l2": float(np.median(e)),
                          "build_cpu_s": c_build, "build_wall_s": w_build,
                          "eval_traj_cpu_s": c_ev, "eval_point_cpu_s": c_evp, "errors": e})
        ex.log(f"Chebyshev n_r={n}: worst={e.max():.2e} build cpu={c_build*1e3:.2f} ms")
    ex.save_table([{k: v for k, v in c.items() if k != "errors"} for c in cheb_rows], "chebyshev")

    # ------------------------------------------------------------- PINNs for the whole family
    results, errors_table = [], {"r": r_test}
    for cfg in CONFIGS:
        ex.log(f"PINN {cfg['tag']}: {cfg}")
        out = B.cached(ex, cfg["tag"], lambda: train_family(ex, cfg, Yref, r_test, t))
        model = build_family_model(cfg)
        model.load_state_dict(out["state_dict"])
        ex.save_history(out["hist"], cfg["tag"])
        err = member_errors(out["pred"], Yref)
        worst = float(err.max())
        errors_table[f"{cfg['tag']}_rel_l2"] = err
        x_traj = to_tensor(np.column_stack([t, np.full_like(t, 2.0)]))
        x_pt = to_tensor(np.array([[T, 2.0]]))

        def ev_traj():
            with torch.no_grad():
                return model(x_traj)

        def ev_pt():
            with torch.no_grad():
                return model(x_pt)

        _, c_ev_traj, _ = B.timed(ev_traj, repeats=reps)
        _, c_ev_pt, _ = B.timed(ev_pt, repeats=reps)
        rec = {"tag": cfg["tag"], "width": cfg["width"], "depth": cfg["depth"],
               "params": count_parameters(model), "collocation": out["n_col"],
               "adam_iters": ex.pick(cfg["adam"], 60), "lbfgs_max_iters": ex.pick(cfg["lbfgs"], 5),
               "worst_rel_l2": worst, "median_rel_l2": float(np.median(err)),
               "train_cpu_s": out["train_cpu_s"], "train_time_s": out["train_time_s"],
               "train_wall_s": out["train_wall_s"], "loss_calls": out["loss_calls"],
               "final_loss": out["final_loss"], "eval_traj_cpu_s": c_ev_traj,
               "eval_point_cpu_s": c_ev_pt,
               "eval_traj_batched_cpu_s": out["eval_cpu_s"] / len(r_test)}
        # matched classical baselines and break-even counts
        for method in ("RK45", "DOP853"):
            cand = [s for s in sweep if s["method"] == method and s["worst_rel_l2"] <= worst]
            s = cand[0] if cand else [s for s in sweep if s["method"] == method][-1]
            cc = classical_cost(method, s["rtol"])
            rec[f"{method}_rtol"] = s["rtol"]
            rec[f"{method}_worst_rel_l2"] = s["worst_rel_l2"]
            for q in ("traj", "point"):
                c_solve = cc[f"{q}_cpu"]
                c_ev = c_ev_traj if q == "traj" else c_ev_pt
                rec[f"{method}_{q}_cpu_s"] = c_solve
                rec[f"{method}_{q}_break_even"] = (out["train_cpu_s"] / (c_solve - c_ev)
                                                    if c_solve > c_ev else math.inf)
            errors_table[f"{cfg['tag']}_{method}_rel_l2"] = s["errors"]
        cheb = next((c for c in cheb_rows if c["worst_rel_l2"] <= worst), cheb_rows[-1])
        rec.update({"cheb_n_r": cheb["n_r"], "cheb_worst_rel_l2": cheb["worst_rel_l2"],
                    "cheb_build_cpu_s": cheb["build_cpu_s"], "cheb_eval_traj_cpu_s": cheb["eval_traj_cpu_s"],
                    "cheb_eval_point_cpu_s": cheb["eval_point_cpu_s"]})
        errors_table[f"{cfg['tag']}_cheb_rel_l2"] = cheb["errors"]
        results.append(rec)
        ex.save_table(B.trajectory_rows(cfg["tag"], out), f"trajectory_{cfg['tag']}")
        ex.log(f"  {cfg['tag']}: worst={worst:.2e} median={np.median(err):.2e} "
               f"train cpu={out['train_cpu_s']:.1f}s eval traj={c_ev_traj*1e3:.3f} ms "
               f"point={c_ev_pt*1e6:.1f} us | RK45 rtol={rec['RK45_rtol']:.0e} "
               f"traj={rec['RK45_traj_cpu_s']*1e3:.3f} ms point={rec['RK45_point_cpu_s']*1e3:.3f} ms "
               f"| n*(traj)={rec['RK45_traj_break_even']:.3g} n*(point)={rec['RK45_point_break_even']:.3g} "
               f"| Chebyshev n_r={cheb['n_r']}")
    ex.save_table(results, "summary")
    ex.save_table(errors_table, "member_errors")
    ex.save_results({
        "problem": {"family": "y' = r y (1 - y), y(0) = 0.1, t in [0, 2]", "r_range": [R_MIN, R_MAX]},
        "test_set": f"{len(r_test)} equispaced r in [1, 3] x {N_T} uniform t in [0, 2]",
        "collocation": f"{N_T_COL} x {N_R_COL} tensor grid",
        "configs": CONFIGS, "summary": results,
        "chebyshev": [{k: v for k, v in c.items() if k != "errors"} for c in cheb_rows],
        "hardware": B.hardware_info(),
    })


if __name__ == "__main__":
    main()
