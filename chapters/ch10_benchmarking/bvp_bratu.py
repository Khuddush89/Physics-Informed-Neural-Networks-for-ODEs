"""Bratu problem  y'' + lam e^y = 0,  y(0) = y(1) = 0  (lam = 1): BVP work-precision benchmark.

Book: Chapter 10 (Benchmarking against classical solvers), Section "Boundary value problems",
Figure "Bratu work-precision diagram" and the BVP table.

Exact solution (lower branch): y(t) = -2 ln( cosh((t - 1/2) theta/2) / cosh(theta/4) ), where
theta = 1.51716459905075... is the smaller root of theta = sqrt(2 lam) cosh(theta/4).
(For lam = 1 a second, upper-branch solution exists; every method below, started from zero,
converges to the lower branch, which is verified by the error.)

Classical methods (median of 5 timed calls after one warm-up):
  solve_bvp  - SciPy collocation (Kierzenka-Shampine), analytic Jacobians, initial mesh of 5
               nodes, zero initial guess, tol = 1e-2, 10^-2.5, ..., 1e-10;
  shooting   - y'(0) = s found by brentq on [0, 2] (bracketing the lower branch only), each
               IVP solved with DOP853 at rtol = atol = tau, brentq xtol = tau/10,
               tau = 1e-3, ..., 1e-12; final trajectory evaluated with dense output;
  FD-Newton  - second-order central differences on N = 2^k - 1 interior nodes (k = 3..14),
               Newton's method with a tridiagonal Jacobian (solve_banded) from y = 0,
               cubic-spline interpolation to the test grid.
PINN: y_theta(t) = t (1 - t) N_theta(t) (exact boundary conditions), M = 100 uniform
collocation points, Adam (cosine 1e-3 -> 1e-5) + L-BFGS; tanh MLP (B1, B2) and Chebyshev
network ChNN (C1, C2), seed 1234.

Errors: relative L2 and maximum error on the uniform 1001-point test grid on [0, 1].

Run:  PYTHONPATH=. python chapters/ch10_benchmarking/bvp_bratu.py          (about 2 min CPU, 1 core)
      PYTHONPATH=. python chapters/ch10_benchmarking/bvp_bratu.py --quick  (smoke test)
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import math

import numpy as np
import torch
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import solve_banded
from scipy.optimize import brentq

from pinnode import (ChebyshevNet, ConstrainedModel, Example, MLP, count_parameters, derivatives,
                     dirichlet_transform, linf, rel_l2, to_tensor, uniform)

LAM = 1.0
N_TEST, M_COL = 1001, 100
THETA = brentq(lambda th: th - math.sqrt(2.0 * LAM) * math.cosh(th / 4.0), 0.1, 4.0, xtol=1e-15)
CONFIGS = [
    dict(tag="B1", arch="MLP", width=16, depth=2, adam=2000, lbfgs=200),
    dict(tag="B2", arch="MLP", width=32, depth=3, adam=6000, lbfgs=600),
    dict(tag="C1", arch="ChNN", degree=8, hidden=(16,), adam=2000, lbfgs=200),
    dict(tag="C2", arch="ChNN", degree=12, hidden=(32,), adam=6000, lbfgs=600),
]
BVP_TOLS = [10.0 ** (-2.0 - 0.5 * k) for k in range(17)]
FD_K = list(range(3, 15))


def exact(t):
    return -2.0 * np.log(np.cosh((t - 0.5) * THETA / 2.0) / math.cosh(THETA / 4.0))


# ----------------------------------------------------------------------------- classical solvers
class Counted:
    """Right-hand side that counts pointwise evaluations (vectorized calls count every point)."""

    def __init__(self):
        self.n = 0

    def __call__(self, t, Y):
        self.n += np.size(t)
        return np.vstack([Y[1], -LAM * np.exp(Y[0])])


def run_solve_bvp(tol: float, t_test: np.ndarray):
    f = Counted()

    def fun_jac(t, Y):
        J = np.zeros((2, 2, np.size(t)))
        J[0, 1] = 1.0
        J[1, 0] = -LAM * np.exp(Y[0])
        return J

    def bc(ya, yb):
        return np.array([ya[0], yb[0]])

    def bc_jac(ya, yb):
        return np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])

    x0 = np.linspace(0.0, 1.0, 5)
    sol = solve_bvp(f, bc, x0, np.zeros((2, x0.size)), fun_jac=fun_jac, bc_jac=bc_jac, tol=tol,
                    max_nodes=100000)
    y = sol.sol(t_test)[0] if sol.success else np.full_like(t_test, np.nan)
    return y, {"evals": f.n, "nodes": int(sol.x.size), "success": bool(sol.success)}


def run_shooting(tau: float, t_test: np.ndarray):
    count = {"nfev": 0, "ivps": 0}

    def rhs(t, z):
        return [z[1], -LAM * math.exp(z[0])]

    def miss(s):
        sol = solve_ivp(rhs, (0.0, 1.0), [0.0, s], method="DOP853", rtol=tau, atol=tau)
        count["nfev"] += sol.nfev
        count["ivps"] += 1
        return sol.y[0, -1]

    s = brentq(miss, 0.0, 2.0, xtol=0.1 * tau)
    sol = solve_ivp(rhs, (0.0, 1.0), [0.0, s], method="DOP853", rtol=tau, atol=tau, t_eval=t_test)
    count["nfev"] += sol.nfev
    count["ivps"] += 1
    return sol.y[0], {"evals": count["nfev"], "ivps": count["ivps"], "slope": s}


def run_fd(k: int, t_test: np.ndarray):
    n = 2 ** k - 1                       # interior nodes
    t = np.linspace(0.0, 1.0, n + 2)
    h = t[1] - t[0]
    y = np.zeros(n)
    ab = np.zeros((3, n))
    ab[0, 1:] = 1.0 / h ** 2
    ab[2, :-1] = 1.0 / h ** 2
    iters = 0
    for iters in range(1, 31):
        yp = np.concatenate([[0.0], y, [0.0]])
        F = (yp[:-2] - 2.0 * yp[1:-1] + yp[2:]) / h ** 2 + LAM * np.exp(y)
        ab[1] = -2.0 / h ** 2 + LAM * np.exp(y)
        dy = solve_banded((1, 1), ab, -F)
        y += dy
        if np.max(np.abs(dy)) < 1e-13:
            break
    yfull = np.concatenate([[0.0], y, [0.0]])
    return CubicSpline(t, yfull)(t_test), {"evals": iters * n, "nodes": n, "newton": iters}


# ----------------------------------------------------------------------------- PINN
def build_model(cfg: dict) -> torch.nn.Module:
    if cfg["arch"] == "MLP":
        net = MLP(1, 1, width=cfg["width"], depth=cfg["depth"], activation="tanh", domain=(0.0, 1.0))
    else:
        net = ChebyshevNet(degree=cfg["degree"], hidden=cfg["hidden"], activation="tanh",
                           domain=(0.0, 1.0))
    return ConstrainedModel(net, dirichlet_transform(0.0, 0.0, 1.0, 0.0))


def train_config(ex: Example, cfg: dict, t_test: np.ndarray, y_ref: np.ndarray) -> dict:
    torch.manual_seed(ex.seed)
    model = build_model(cfg)
    t_col = to_tensor(uniform(M_COL, 0.0, 1.0), requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y, _, d2y = derivatives(model(t_col), t_col, 2)
        r = d2y + LAM * torch.exp(y)
        L_r = (r ** 2).mean()
        return L_r, {"residual": L_r}

    def predict():
        with torch.no_grad():
            return model(t_test_t).numpy().ravel()

    out = B.run_pinn(model, loss_fn, M_COL, lambda: rel_l2(predict(), y_ref), predict,
                     adam_iters=ex.pick(cfg["adam"], 60), lbfgs_iters=ex.pick(cfg["lbfgs"], 5),
                     eval_every=ex.pick(100, 20), print_every=ex.pick(5000, 50))
    out["params"] = count_parameters(model)
    return out


def classical_row(method, setting, y, info, cpu, wall, y_ref):
    ok = np.all(np.isfinite(y))
    return {"method": method, "setting": setting, "rel_l2": rel_l2(y, y_ref) if ok else math.nan,
            "linf": linf(y, y_ref) if ok else math.nan, "cpu_s": cpu, "wall_s": wall, **info}


def main():
    ex = Example("ch10_benchmarking", "bvp_bratu",
                 description="Bratu BVP: collocation, shooting, finite differences vs PINN",
                 default_seed=1234, extra_args=B.RESUME_ARGS)
    t_test = np.linspace(0.0, 1.0, N_TEST)
    y_ref = exact(t_test)
    reps = ex.pick(B.REPEATS, 1)
    preds = {"t": t_test, "exact": y_ref}

    def run_classical():
        rows, preds = [], {}
        for tol in (BVP_TOLS if not ex.quick else BVP_TOLS[::8]):
            (y, info), cpu, wall = B.timed(lambda: run_solve_bvp(tol, t_test), repeats=reps)
            rows.append(classical_row("solve_bvp", tol, y, info, cpu, wall, y_ref))
            ex.log(f"solve_bvp tol={tol:.1e} relL2={rows[-1]['rel_l2']:.2e} nodes={info['nodes']} "
                   f"cpu={cpu*1e3:.2f} ms")
            if abs(tol - 1e-6) < 1e-12:
                preds["solve_bvp_tol1e-6"] = y
        for tau in (B.RTOLS if not ex.quick else B.RTOLS[::6]):
            (y, info), cpu, wall = B.timed(lambda: run_shooting(tau, t_test), repeats=reps)
            rows.append(classical_row("shooting", tau, y, info, cpu, wall, y_ref))
            ex.log(f"shooting tau={tau:.1e} relL2={rows[-1]['rel_l2']:.2e} ivps={info['ivps']} "
                   f"cpu={cpu*1e3:.2f} ms")
            if abs(tau - 1e-6) < 1e-12:
                preds["shooting_tau1e-6"] = y
        for k in (FD_K if not ex.quick else FD_K[::4]):
            (y, info), cpu, wall = B.timed(lambda: run_fd(k, t_test), repeats=reps)
            rows.append(classical_row("FD-Newton", 2 ** k - 1, y, info, cpu, wall, y_ref))
            ex.log(f"FD N={2**k-1} relL2={rows[-1]['rel_l2']:.2e} newton={info['newton']} "
                   f"cpu={cpu*1e3:.2f} ms")
            if k == 7:
                preds["fd_N127"] = y
        return rows, preds

    rows, cpreds = B.cached(ex, "classical", run_classical)
    preds.update(cpreds)
    ex.save_table(rows, "classical")

    pinn_rows, traj_rows = [], []
    for cfg in CONFIGS:
        ex.log(f"PINN {cfg['tag']}: {cfg}")
        out = B.cached(ex, cfg["tag"], lambda: train_config(ex, cfg, t_test, y_ref))
        setting = (f"{cfg['width']}x{cfg['depth']}" if cfg["arch"] == "MLP"
                   else f"m={cfg['degree']}, {'x'.join(map(str, cfg['hidden']))}")
        row = B.pinn_record(cfg["tag"], out, out["pred"], y_ref, arch=cfg["arch"],
                            params=out["params"], collocation=M_COL,
                            adam_iters=ex.pick(cfg["adam"], 60),
                            lbfgs_max_iters=ex.pick(cfg["lbfgs"], 5), setting=setting)
        pinn_rows.append(row)
        ex.save_history(out["hist"], cfg["tag"])
        preds[cfg["tag"]] = out["pred"]
        traj_rows += B.trajectory_rows(cfg["tag"], out)
        ex.log(f"  {cfg['tag']}: relL2={row['rel_l2']:.2e} linf={row['linf']:.2e} "
               f"cpu={row['cpu_s']:.1f}s wall={row['wall_s']:.1f}s")
    ex.save_table(pinn_rows, "pinn")
    ex.save_table(traj_rows, "pinn_trajectories")
    ex.save_table(preds, "predictions")

    groups = {m: [r for r in rows if r["method"] == m] for m in ("solve_bvp", "shooting", "FD-Newton")}
    groups["PINN-MLP"] = [r for r in pinn_rows if r["arch"] == "MLP"]
    groups["PINN-ChNN"] = [r for r in pinn_rows if r["arch"] == "ChNN"]
    matched = B.matched_table(groups)
    ex.save_table(matched, "matched")
    for m in matched:
        ex.log(f"matched {m['method']:9s} target={m['target']:.0e}: "
               + (f"cpu={m['cpu_s']:.3e}s" if m["reached"] else "not reached"))
    ex.save_results({
        "problem": {"equation": "y'' + lam e^y = 0, y(0) = y(1) = 0", "lam": LAM, "theta": THETA,
                    "exact": "-2 ln(cosh((t-1/2) theta/2) / cosh(theta/4))"},
        "test_grid": f"{N_TEST} uniform points on [0, 1]",
        "classical_protocol": {"solve_bvp_tols": BVP_TOLS, "shooting_taus": B.RTOLS,
                               "fd_interior_nodes": [2 ** k - 1 for k in FD_K],
                               "timing": f"median of {B.REPEATS} after 1 warm-up"},
        "pinn_protocol": {"trial_solution": "t (1 - t) N(t)", "collocation": f"{M_COL} uniform",
                          "optimizer": "Adam (1e-3 -> 1e-5 cosine) + L-BFGS", "configs": CONFIGS},
        "best_error": {m: B.best_error(g) for m, g in groups.items()},
        "pinn": pinn_rows, "matched": matched, "hardware": B.hardware_info(),
    })


if __name__ == "__main__":
    main()
