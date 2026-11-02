"""Stiff linear system  y' = A y  with eigenvalues -1 and -lambda: work-precision benchmark.

Book: Chapter 10 (Benchmarking against classical solvers), Section "Initial value problems",
Figure "IVP work-precision diagrams" (panel c) and the IVP matched-accuracy table.

Problem: A = [[-(lam+1)/2, (lam-1)/2], [(lam-1)/2, -(lam+1)/2]] with lam = 1000, y(0) = (2, 0),
t in [0, 1]. The eigenvectors are (1, 1) (eigenvalue -1) and (1, -1) (eigenvalue -lam), so

    y(t) = e^{-t} (1, 1) + e^{-lam t} (1, -1):

a slow mode plus an initial layer of width 1/lam (stiffness ratio 1000).

Classical: solve_ivp (RK45, DOP853, Radau, BDF, LSODA), rtol = 1e-3, ..., 1e-12, atol = rtol,
analytic Jacobian (= A) for the implicit methods; median of 5 timed calls after one warm-up.

PINN: hard initial condition y_theta = y(0) + t N_theta(t) (two outputs), tanh MLP with input
scaling, Adam (cosine 1e-3 -> 1e-5) + L-BFGS, M = 200 collocation points:
  S1  uniform points on [0, 1];
  S2  layer-adapted points (100 geometric in [1e-5, 1e-2], 100 uniform in [1e-2, 1]);
  S3  as S2 with a larger network and twice the iterations.

Errors: relative L2 (both components) and maximum error on the uniform 1001-point grid.

Run:  PYTHONPATH=. python chapters/ch10_benchmarking/ivp_stiff.py          (about 3 min CPU, 1 core)
      PYTHONPATH=. python chapters/ch10_benchmarking/ivp_stiff.py --quick  (smoke test)
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import numpy as np
import torch

from pinnode import ConstrainedModel, Example, MLP, count_parameters, derivative, rel_l2, to_tensor

LAM, T = 1000.0, 1.0
Y0 = np.array([2.0, 0.0])
A = np.array([[-(LAM + 1) / 2, (LAM - 1) / 2], [(LAM - 1) / 2, -(LAM + 1) / 2]])
N_TEST, M_COL = 1001, 200
CONFIGS = [
    dict(tag="S1", width=32, depth=3, adam=5000, lbfgs=500, points="uniform"),
    dict(tag="S2", width=32, depth=3, adam=5000, lbfgs=500, points="layer"),
    dict(tag="S3", width=64, depth=3, adam=10000, lbfgs=1000, points="layer"),
]


def exact(t):
    t = np.asarray(t).reshape(-1, 1)
    return np.exp(-t) * np.array([[1.0, 1.0]]) + np.exp(-LAM * t) * np.array([[1.0, -1.0]])


def rhs(t, y):
    return A @ y


def jac(t, y):
    return A


def collocation(kind: str, m: int) -> np.ndarray:
    if kind == "uniform":
        return np.linspace(0.0, T, m)
    half = m // 2
    return np.concatenate([np.geomspace(1e-5, 1e-2, half, endpoint=False),
                           np.linspace(1e-2, T, m - half)])


def build_model(width: int, depth: int) -> torch.nn.Module:
    net = MLP(1, 2, width=width, depth=depth, activation="tanh", domain=(0.0, T))
    y0 = torch.tensor(Y0).reshape(1, 2)
    return ConstrainedModel(net, lambda t, n: y0 + t * n)


def train_config(ex: Example, cfg: dict, t_test: np.ndarray, y_ref: np.ndarray) -> dict:
    torch.manual_seed(ex.seed)
    model = build_model(cfg["width"], cfg["depth"])
    t_col = to_tensor(collocation(cfg["points"], M_COL), requires_grad=True)
    A_t = torch.tensor(A)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y = model(t_col)
        r = derivative(y, t_col) - y @ A_t.T
        L_r = (r ** 2).sum(dim=1).mean()
        return L_r, {"residual": L_r}

    def predict():
        with torch.no_grad():
            return model(t_test_t).numpy()

    out = B.run_pinn(model, loss_fn, M_COL, lambda: rel_l2(predict(), y_ref), predict,
                     adam_iters=ex.pick(cfg["adam"], 60), lbfgs_iters=ex.pick(cfg["lbfgs"], 5),
                     eval_every=ex.pick(100, 20), print_every=ex.pick(5000, 50))
    out["params"] = count_parameters(model)
    return out


def main():
    ex = Example("ch10_benchmarking", "ivp_stiff",
                 description="Stiff linear system (ratio 1000): classical solvers vs PINN",
                 default_seed=1234, extra_args=B.RESUME_ARGS)
    t_test = np.linspace(0.0, T, N_TEST)
    y_ref = exact(t_test)

    rtols = B.RTOLS if not ex.quick else B.RTOLS[::6]
    classical = B.cached(ex, "classical", lambda: B.ivp_sweep(
        rhs, (0.0, T), Y0, t_test, y_ref, rtols=rtols, jac=jac, repeats=ex.pick(B.REPEATS, 1),
        log=ex.log))
    ex.save_table(classical, "classical")

    pinn_rows, traj_rows = [], []
    preds = {"t": t_test, "exact_y1": y_ref[:, 0], "exact_y2": y_ref[:, 1]}
    for cfg in CONFIGS:
        ex.log(f"PINN {cfg['tag']}: width={cfg['width']} depth={cfg['depth']} "
               f"adam={cfg['adam']} lbfgs={cfg['lbfgs']} points={cfg['points']}")
        out = B.cached(ex, cfg["tag"], lambda: train_config(ex, cfg, t_test, y_ref))
        row = B.pinn_record(cfg["tag"], out, out["pred"], y_ref, width=cfg["width"],
                            depth=cfg["depth"], params=out["params"], collocation=M_COL,
                            points=cfg["points"], adam_iters=ex.pick(cfg["adam"], 60),
                            lbfgs_max_iters=ex.pick(cfg["lbfgs"], 5),
                            setting=f"{cfg['width']}x{cfg['depth']}, {cfg['points']}")
        # error away from the layer (t >= 0.01) separates the two error sources
        far = t_test >= 0.01
        row["rel_l2_outside_layer"] = rel_l2(out["pred"][far], y_ref[far])
        pinn_rows.append(row)
        ex.save_history(out["hist"], cfg["tag"])
        preds[f"{cfg['tag']}_y1"], preds[f"{cfg['tag']}_y2"] = out["pred"][:, 0], out["pred"][:, 1]
        traj_rows += B.trajectory_rows(cfg["tag"], out)
        ex.log(f"  {cfg['tag']}: relL2={row['rel_l2']:.2e} (t>=0.01: {row['rel_l2_outside_layer']:.2e}) "
               f"linf={row['linf']:.2e} cpu={row['cpu_s']:.1f}s wall={row['wall_s']:.1f}s")
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
        "problem": {"equation": "y' = A y", "lambda": LAM, "A": A.tolist(), "y0": Y0.tolist(),
                    "T": T, "exact": "e^{-t}(1,1) + e^{-lambda t}(1,-1)"},
        "test_grid": f"{N_TEST} uniform points on [0, {T}]; error over both components",
        "classical_protocol": {"methods": B.IVP_METHODS, "rtols": rtols, "atol": "= rtol",
                               "timing": f"median of {B.REPEATS} after 1 warm-up",
                               "jacobian": "analytic (Radau, BDF, LSODA)"},
        "pinn_protocol": {"trial_solution": "y0 + t N(t)", "activation": "tanh",
                          "collocation": "200: uniform (S1) or layer-adapted (S2, S3)",
                          "optimizer": "Adam (1e-3 -> 1e-5 cosine) + L-BFGS", "configs": CONFIGS},
        "best_classical_error": B.best_error(classical),
        "pinn": pinn_rows, "matched": matched, "hardware": B.hardware_info(),
    })


if __name__ == "__main__":
    main()
