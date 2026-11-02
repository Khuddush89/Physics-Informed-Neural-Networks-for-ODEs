"""Integral and integro-differential equations: classical quadrature methods versus PINNs.

Book: Chapter 10 (Benchmarking against classical solvers), Section "Integral and
integro-differential equations", Figure "IE/IDE work-precision diagrams" and the IE table.

Problems on [0, 1] (exact solutions in brackets):
  F1   u(x) = 1 + (1/2) int_0^1 x t u(t) dt                          [u = 1 + 3x/10]
  F2   u(x) = f(x) + (1/2) int_0^1 e^{xt} u(t) dt,
       f(x) = e^x - (e^{x+1} - 1) / (2 (x + 1))                       [u = e^x]
  V    u(x) = 1 + 2 int_0^x t u(t) dt                                 [u = e^{x^2}]
  IDE  u'(x) = u(x) + int_0^x e^{-(x-t)} u(t) dt,  u(0) = 1           [u = cosh(sqrt2 x) + sinh(sqrt2 x)/sqrt2]

Classical methods (median of 5 timed calls after one warm-up; every timed call returns the
solution on the 1001-point test grid):
  Nystrom   - n-point Gauss-Legendre rule, linear solve, Nystrom interpolation (F1, F2);
  trapezoid - trapezoidal product integration on n uniform nodes (V; for the IDE combined with
              the trapezoidal rule for u'), O(n^2) work, cubic-spline interpolation;
  RK45/DOP853 on the equivalent ODE system u' = u + z, z' = u - z, z(0) = 0 (IDE only),
              rtol = 1e-3, ..., 1e-12, atol = rtol.
PINN: tanh MLP, M = 64 uniform collocation points, integrals by a 16-point Gauss-Legendre rule
(on [0, 1] for F1/F2 and on [0, x_i] for V/IDE), u_theta = 1 + x N_theta(x) for the IDE;
Adam (cosine 1e-3 -> 1e-5) + L-BFGS; configurations I1 (16x2) and I2 (32x3), seed 1234.

Run:  PYTHONPATH=. python chapters/ch10_benchmarking/integral_equations.py          (about 1.5 min CPU)
      PYTHONPATH=. python chapters/ch10_benchmarking/integral_equations.py --quick  (smoke test)
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import math

import numpy as np
import torch
from scipy.interpolate import CubicSpline

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivative, gauss_legendre,
                     linf, rel_l2, to_tensor, uniform, volterra_gauss)
from pinnode.reference import nystrom_fredholm, volterra_trapezoid

N_TEST, M_COL, N_QUAD = 1001, 64, 16
SQ2 = math.sqrt(2.0)
CONFIGS = [
    dict(tag="I1", width=16, depth=2, adam=2000, lbfgs=200),
    dict(tag="I2", width=32, depth=3, adam=5000, lbfgs=500),
]
NYSTROM_N = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
TRAP_N = [10 * 2 ** k + 1 for k in range(10)]          # 11, 21, ..., 5121 nodes


def f_F2(x):
    return np.exp(x) - (np.exp(x + 1.0) - 1.0) / (2.0 * (x + 1.0))


PROBLEMS = {
    "F1": dict(kind="fredholm", lam=0.5, kernel=lambda x, t: x * t, f=lambda x: np.ones_like(x),
               exact=lambda x: 1.0 + 0.3 * x, label="Fredholm, $K = xt$"),
    "F2": dict(kind="fredholm", lam=0.5, kernel=lambda x, t: np.exp(x * t), f=f_F2,
               exact=lambda x: np.exp(x), label="Fredholm, $K = e^{xt}$"),
    "V": dict(kind="volterra", lam=2.0, kernel=lambda x, t: t + 0.0 * x, f=lambda x: np.ones_like(x),
              exact=lambda x: np.exp(x ** 2), label="Volterra, $K = t$"),
    "IDE": dict(kind="ide", exact=lambda x: np.cosh(SQ2 * x) + np.sinh(SQ2 * x) / SQ2,
                label="IDE, $K = e^{-(x-t)}$"),
}


# ----------------------------------------------------------------------------- classical methods
def nystrom_solve(p: dict, n: int, x_test: np.ndarray) -> np.ndarray:
    u = nystrom_fredholm(p["kernel"], p["f"], p["lam"], 0.0, 1.0, n=n)
    return u(x_test)


def volterra_solve(p: dict, n: int, x_test: np.ndarray) -> np.ndarray:
    x, u = volterra_trapezoid(p["kernel"], p["f"], p["lam"], 0.0, 1.0, n=n)
    return CubicSpline(x, u)(x_test)


def ide_trapezoid(n: int, x_test: np.ndarray) -> np.ndarray:
    """Trapezoidal rule for u' combined with trapezoidal product integration of the memory
    term (general-kernel algorithm, O(n^2) work, second order)."""
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    u, F = np.empty(n), np.empty(n)
    u[0], F[0] = 1.0, 1.0                   # F_i = u_i + I_i with I_0 = 0
    a = 1.0 - 0.5 * h * (1.0 + 0.5 * h)
    for i in range(1, n):
        Ki = np.exp(-(x[i] - x[:i]))       # kernel K(x_i, x_j), j < i
        G = h * (0.5 * Ki[0] * u[0] + np.dot(Ki[1:], u[1:i]))
        u[i] = (u[i - 1] + 0.5 * h * (F[i - 1] + G)) / a
        F[i] = (1.0 + 0.5 * h) * u[i] + G
    return CubicSpline(x, u)(x_test)


def ide_rhs(x, U):
    return [U[0] + U[1], U[0] - U[1]]


def ide_jac(x, U):
    return np.array([[1.0, 1.0], [1.0, -1.0]])


# ----------------------------------------------------------------------------- PINNs
def make_pinn(name: str, cfg: dict):
    """Model, loss closure and network evaluations per loss call for problem ``name``."""
    p = PROBLEMS[name]
    torch.manual_seed(CONFIG_SEED[0])
    net = MLP(1, 1, width=cfg["width"], depth=cfg["depth"], activation="tanh", domain=(0.0, 1.0))
    xc = uniform(M_COL, 0.0, 1.0)
    if p["kind"] == "fredholm":
        model = net
        tq, wq = gauss_legendre(N_QUAD, 0.0, 1.0)
        Kw = torch.tensor(p["kernel"](xc[:, None], tq[None, :]) * wq[None, :])
        x_t, t_t, f_t = to_tensor(xc), to_tensor(tq), to_tensor(p["f"](xc))

        def loss_fn():
            r = model(x_t) - f_t - p["lam"] * Kw @ model(t_t)
            L = (r ** 2).mean()
            return L, {"residual": L}
        per_call = M_COL + N_QUAD
    else:
        S, W = volterra_gauss(xc, N_QUAD, a=0.0)
        s_t = to_tensor(S.reshape(-1))
        if p["kind"] == "volterra":
            model = net
            KW = torch.tensor(W * S * p["lam"])                     # lam K(x_i, s_ij) w_ij
            x_t = to_tensor(xc)

            def loss_fn():
                us = model(s_t).reshape(M_COL, N_QUAD)
                r = model(x_t) - 1.0 - (KW * us).sum(dim=1, keepdim=True)
                L = (r ** 2).mean()
                return L, {"residual": L}
        else:
            model = ConstrainedModel(net, lambda x, n: 1.0 + x * n)
            KW = torch.tensor(W * np.exp(-(xc[:, None] - S)))
            x_t = to_tensor(xc, requires_grad=True)

            def loss_fn():
                u = model(x_t)
                us = model(s_t).reshape(M_COL, N_QUAD)
                r = derivative(u, x_t) - u - (KW * us).sum(dim=1, keepdim=True)
                L = (r ** 2).mean()
                return L, {"residual": L}
        per_call = M_COL * (1 + N_QUAD)
    return model, loss_fn, per_call


CONFIG_SEED = [1234]


def classical_row(problem, method, setting, y, cpu, wall, y_ref):
    return {"problem": problem, "method": method, "setting": setting,
            "rel_l2": rel_l2(y, y_ref), "linf": linf(y, y_ref), "cpu_s": cpu, "wall_s": wall}


def main():
    ex = Example("ch10_benchmarking", "integral_equations",
                 description="Fredholm, Volterra and IDE: quadrature methods vs PINNs",
                 default_seed=1234, extra_args=B.RESUME_ARGS)
    CONFIG_SEED[0] = ex.seed
    reps = ex.pick(B.REPEATS, 1)
    x_test = np.linspace(0.0, 1.0, N_TEST)
    rows, pinn_rows, traj_rows = [], [], []
    preds = {"x": x_test}

    for name, p in PROBLEMS.items():
        y_ref = p["exact"](x_test)
        preds[f"{name}_exact"] = y_ref
        # ------------------------------------------------------------- classical
        def run_classical(p=p, name=name, y_ref=y_ref):
            rows = []
            if p["kind"] == "fredholm":
                for n in (NYSTROM_N if not ex.quick else NYSTROM_N[:3]):
                    y, cpu, wall = B.timed(lambda: nystrom_solve(p, n, x_test), repeats=reps)
                    rows.append(classical_row(name, "Nystrom", n, y, cpu, wall, y_ref))
            elif p["kind"] == "volterra":
                for n in (TRAP_N if not ex.quick else TRAP_N[:3]):
                    y, cpu, wall = B.timed(lambda: volterra_solve(p, n, x_test), repeats=reps)
                    rows.append(classical_row(name, "trapezoid", n, y, cpu, wall, y_ref))
            else:
                for n in (TRAP_N if not ex.quick else TRAP_N[:3]):
                    y, cpu, wall = B.timed(lambda: ide_trapezoid(n, x_test), repeats=reps)
                    rows.append(classical_row(name, "trapezoid", n, y, cpu, wall, y_ref))
                sweep = B.ivp_sweep(ide_rhs, (0.0, 1.0), [1.0, 0.0], x_test, y_ref.reshape(-1, 1),
                                    methods=["RK45", "DOP853"],
                                    rtols=B.RTOLS if not ex.quick else B.RTOLS[::6],
                                    jac=ide_jac, components=[0], repeats=reps)
                for r in sweep:
                    rows.append({"problem": name, "method": r["method"], "setting": r["rtol"],
                                 "rel_l2": r["rel_l2"], "linf": r["linf"], "cpu_s": r["cpu_s"],
                                 "wall_s": r["wall_s"], "nfev": r["nfev"]})
            return rows

        rows += B.cached(ex, f"classical-{name}", run_classical)
        for r in rows:
            if r["problem"] == name:
                ex.log(f"{name:3s} {r['method']:9s} {r['setting']!s:>10s} relL2={r['rel_l2']:.2e} "
                       f"cpu={r['cpu_s']*1e3:8.3f} ms")
        # ------------------------------------------------------------- PINNs
        x_test_t = to_tensor(x_test)
        for cfg in CONFIGS:
            model, loss_fn, per_call = make_pinn(name, cfg)

            def predict(model=model):
                with torch.no_grad():
                    return model(x_test_t).numpy().ravel()

            tag = f"{name}-{cfg['tag']}"
            out = B.cached(ex, tag, lambda: B.run_pinn(
                model, loss_fn, per_call, lambda: rel_l2(predict(), y_ref), predict,
                adam_iters=ex.pick(cfg["adam"], 60), lbfgs_iters=ex.pick(cfg["lbfgs"], 5),
                eval_every=ex.pick(100, 20), print_every=ex.pick(5000, 50)))
            row = B.pinn_record(cfg["tag"], out, out["pred"], y_ref, problem=name,
                                width=cfg["width"], depth=cfg["depth"],
                                params=count_parameters(model), collocation=M_COL,
                                quadrature=N_QUAD, adam_iters=ex.pick(cfg["adam"], 60),
                                lbfgs_max_iters=ex.pick(cfg["lbfgs"], 5),
                                setting=f"{cfg['width']}x{cfg['depth']}")
            pinn_rows.append(row)
            ex.save_history(out["hist"], tag)
            preds[f"{name}_{cfg['tag']}"] = out["pred"]
            traj_rows += [dict(r, problem=name) for r in B.trajectory_rows(cfg["tag"], out)]
            ex.log(f"{name:3s} PINN {cfg['tag']}: relL2={row['rel_l2']:.2e} linf={row['linf']:.2e} "
                   f"cpu={row['cpu_s']:.1f}s wall={row['wall_s']:.1f}s")

    ex.save_table(rows, "classical")
    ex.save_table(pinn_rows, "pinn")
    ex.save_table(traj_rows, "pinn_trajectories")
    ex.save_table(preds, "predictions")

    matched = []
    for name in PROBLEMS:
        methods = sorted({r["method"] for r in rows if r["problem"] == name},
                         key=["Nystrom", "trapezoid", "RK45", "DOP853"].index)
        groups = {m: [r for r in rows if r["problem"] == name and r["method"] == m] for m in methods}
        groups["PINN"] = [r for r in pinn_rows if r["problem"] == name]
        for m in B.matched_table(groups):
            m["problem"] = name
            m["best_rel_l2"] = B.best_error(groups[m["method"]])
            matched.append(m)
    ex.save_table(matched, "matched")
    ex.save_results({
        "problems": {k: {kk: vv for kk, vv in v.items() if isinstance(vv, (str, float, int))}
                     for k, v in PROBLEMS.items()},
        "test_grid": f"{N_TEST} uniform points on [0, 1]",
        "classical_protocol": {"nystrom_nodes": NYSTROM_N, "trapezoid_nodes": TRAP_N,
                               "ide_ode_rtols": B.RTOLS,
                               "timing": f"median of {B.REPEATS} after 1 warm-up"},
        "pinn_protocol": {"collocation": f"{M_COL} uniform", "quadrature": f"{N_QUAD}-point Gauss",
                          "optimizer": "Adam (1e-3 -> 1e-5 cosine) + L-BFGS", "configs": CONFIGS},
        "pinn": pinn_rows, "matched": matched, "hardware": B.hardware_info(),
    })


if __name__ == "__main__":
    main()
