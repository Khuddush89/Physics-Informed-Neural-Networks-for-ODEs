"""Logistic ODE  y' = r y (1 - y),  y(0) = y0  on [0, T]: work-precision benchmark.

Book: Chapter 10 (Benchmarking against classical solvers), Section "Initial value problems",
Figure "IVP work-precision diagrams" (panel a) and the IVP matched-accuracy table.

Problem: r = 2, y0 = 0.1, T = 2. Exact solution y(t) = y0 e^{rt} / (1 + y0 (e^{rt} - 1)).

Classical: scipy.integrate.solve_ivp with RK45, DOP853, Radau, BDF and LSODA for
rtol = 1e-3, 10^-3.5, ..., 1e-12 and atol = rtol (analytic Jacobian for the implicit methods);
each run is timed (CPU and wall clock) as the median of 5 calls after one warm-up call and
returns the solution on the test grid through the solver's dense output (t_eval).

PINN: hard initial condition y_theta(t) = y0 + t N_theta(t), tanh MLP with input scaling,
M = 100 uniform collocation points, Adam (cosine schedule 1e-3 -> 1e-5) followed by L-BFGS;
four configurations P1-P4 (network size x iterations), one run each with seed 1234.

Errors: relative L2 and maximum error on the uniform 1001-point test grid on [0, T].

Outputs (outputs/ch10_benchmarking/ivp_logistic/): classical.csv, pinn.csv,
pinn_trajectories.csv, predictions.csv, matched.csv, history_P*.csv, results.json.
Book tables and figures are produced from these files by write_book_assets.py.

Run:  PYTHONPATH=. python chapters/ch10_benchmarking/ivp_logistic.py          (about 1.5 min CPU, 1 core)
      PYTHONPATH=. python chapters/ch10_benchmarking/ivp_logistic.py --quick  (smoke test)
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import numpy as np
import torch

from pinnode import ConstrainedModel, Example, MLP, count_parameters, derivative, rel_l2, to_tensor, uniform

R, Y0, T = 2.0, 0.1, 2.0
N_TEST, M_COL = 1001, 100
CONFIGS = [
    dict(tag="P1", width=16, depth=2, adam=2000, lbfgs=200),
    dict(tag="P2", width=32, depth=2, adam=4000, lbfgs=400),
    dict(tag="P3", width=32, depth=3, adam=8000, lbfgs=800),
    dict(tag="P4", width=64, depth=3, adam=10000, lbfgs=1000),
]


def exact(t):
    return Y0 * np.exp(R * t) / (1.0 + Y0 * (np.exp(R * t) - 1.0))


def rhs(t, y):
    return R * y * (1.0 - y)


def jac(t, y):
    return np.array([[R * (1.0 - 2.0 * y[0])]])


def build_model(width: int, depth: int) -> torch.nn.Module:
    net = MLP(1, 1, width=width, depth=depth, activation="tanh", domain=(0.0, T))
    return ConstrainedModel(net, lambda t, n: Y0 + t * n)


def train_config(ex: Example, cfg: dict, t_test: np.ndarray, y_ref: np.ndarray) -> dict:
    """Train one PINN configuration; returns the bookkeeping dict of _bench.run_pinn."""
    torch.manual_seed(ex.seed)
    model = build_model(cfg["width"], cfg["depth"])
    t_col = to_tensor(uniform(M_COL, 0.0, T), requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y = model(t_col)
        r = derivative(y, t_col) - R * y * (1.0 - y)
        L_r = (r ** 2).mean()
        return L_r, {"residual": L_r}

    def predict():
        with torch.no_grad():
            return model(t_test_t).numpy().ravel()

    out = B.run_pinn(model, loss_fn, M_COL, lambda: rel_l2(predict(), y_ref), predict,
                     adam_iters=ex.pick(cfg["adam"], 60), lbfgs_iters=ex.pick(cfg["lbfgs"], 5),
                     eval_every=ex.pick(100, 20), print_every=ex.pick(5000, 50))
    out["model"], out["params"] = model, count_parameters(model)
    return out


def main():
    ex = Example("ch10_benchmarking", "ivp_logistic",
                 description="Logistic ODE: classical solvers vs PINN (work-precision)",
                 default_seed=1234, extra_args=B.RESUME_ARGS)
    t_test = np.linspace(0.0, T, N_TEST)
    y_ref = exact(t_test)

    # ------------------------------------------------------------- classical sweep
    rtols = B.RTOLS if not ex.quick else B.RTOLS[::6]
    classical = B.cached(ex, "classical", lambda: B.ivp_sweep(
        rhs, (0.0, T), [Y0], t_test, y_ref.reshape(-1, 1), rtols=rtols, jac=jac,
        repeats=ex.pick(B.REPEATS, 1), log=ex.log))
    ex.save_table(classical, "classical")

    # ------------------------------------------------------------- PINN configurations
    pinn_rows, traj_rows, preds = [], [], {"t": t_test, "exact": y_ref}
    for cfg in CONFIGS:
        ex.log(f"PINN {cfg['tag']}: width={cfg['width']} depth={cfg['depth']} "
               f"adam={cfg['adam']} lbfgs={cfg['lbfgs']}")
        out = B.cached(ex, cfg["tag"], lambda: B.strip_model(train_config(ex, cfg, t_test, y_ref)))
        row = B.pinn_record(cfg["tag"], out, out["pred"], y_ref, width=cfg["width"],
                            depth=cfg["depth"], params=out["params"], collocation=M_COL,
                            adam_iters=ex.pick(cfg["adam"], 60),
                            lbfgs_max_iters=ex.pick(cfg["lbfgs"], 5),
                            setting=f"{cfg['width']}x{cfg['depth']}")
        pinn_rows.append(row)
        ex.save_history(out["hist"], cfg["tag"])
        preds[cfg["tag"]] = out["pred"]
        traj_rows += B.trajectory_rows(cfg["tag"], out)
        ex.log(f"  {cfg['tag']}: relL2={row['rel_l2']:.2e} linf={row['linf']:.2e} "
               f"cpu={row['cpu_s']:.1f}s wall={row['wall_s']:.1f}s evals={row['evals']:.2e}")
    ex.save_table(pinn_rows, "pinn")
    ex.save_table(traj_rows, "pinn_trajectories")
    ex.save_table(preds, "predictions")

    groups = {m: [r for r in classical if r["method"] == m] for m in B.IVP_METHODS}
    groups["PINN"] = pinn_rows
    matched = B.matched_table(groups)
    ex.save_table(matched, "matched")
    for m in matched:
        ex.log(f"matched {m['method']:7s} target={m['target']:.0e}: "
               + (f"cpu={m['cpu_s']:.3e}s evals={m['evals']}" if m["reached"] else "not reached"))

    ex.save_results({
        "problem": {"equation": "y' = r y (1 - y)", "r": R, "y0": Y0, "T": T,
                    "exact": "y0 e^{rt} / (1 + y0 (e^{rt} - 1))"},
        "test_grid": f"{N_TEST} uniform points on [0, {T}]",
        "classical_protocol": {"methods": B.IVP_METHODS, "rtols": rtols, "atol": "= rtol",
                               "timing": f"median of {B.REPEATS} after 1 warm-up",
                               "jacobian": "analytic (Radau, BDF, LSODA)"},
        "pinn_protocol": {"trial_solution": "y0 + t N(t)", "activation": "tanh",
                          "collocation": f"{M_COL} uniform",
                          "optimizer": "Adam (1e-3 -> 1e-5 cosine) + L-BFGS", "configs": CONFIGS},
        "best_classical_error": B.best_error(classical),
        "pinn": pinn_rows, "matched": matched, "hardware": B.hardware_info(),
    })


if __name__ == "__main__":
    main()
