"""Van der Pol oscillator  y'' - mu (1 - y^2) y' + y = 0  (mu = 0.5): work-precision benchmark.

Book: Chapter 10 (Benchmarking against classical solvers), Section "Initial value problems",
Figure "IVP work-precision diagrams" (panel b) and the IVP matched-accuracy table.

Problem: mu = 0.5, y(0) = 1, y'(0) = 0, t in [0, 10] (non-stiff, about 1.6 periods).
Reference: DOP853 with rtol = 2.5e-14, atol = 1e-15 on the test grid; its distance to a Radau
solution with rtol = atol = 1e-13 is stored in results.json as a check of the reference.

Classical: solve_ivp (RK45, DOP853, Radau, BDF, LSODA) applied to the first-order system for
rtol = 1e-3, ..., 1e-12 (half decades), atol = rtol, analytic Jacobian for implicit methods;
median of 5 timed calls after one warm-up. The error is measured on y only.

PINN: second-order residual with the hard initial conditions y_theta = 1 + phi(t) N_theta(t),
where phi(0) = phi'(0) = 0; tanh MLP with input scaling, M = 200 uniform collocation points,
Adam (cosine 1e-3 -> 1e-5) followed by L-BFGS, seed 1234. Configurations:
  V1  phi = t^2/(1+t^2) (bounded multiplier), 32x3, 5000 Adam + <= 500 L-BFGS;
  V2  phi = t^2/(1+t^2), 64x3, 10000 Adam + <= 1000 L-BFGS;
  V3  phi = t^2 (unbounded multiplier), 32x3, 5000 Adam + <= 500 L-BFGS. With phi = t^2 the
      residual is dominated by the factor t^2 <= 100 and training converges to a wrong solution;
      V3 documents this sensitivity to the choice of the trial function.

Errors: relative L2 and maximum error of y on the uniform 1001-point test grid on [0, 10].

Run:  PYTHONPATH=. python chapters/ch10_benchmarking/ivp_vanderpol.py          (about 4 min CPU, 1 core)
      PYTHONPATH=. python chapters/ch10_benchmarking/ivp_vanderpol.py --quick  (smoke test)
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import ConstrainedModel, Example, MLP, count_parameters, derivatives, rel_l2, to_tensor, uniform

MU, Y0, V0, T = 0.5, 1.0, 0.0, 10.0
N_TEST, M_COL = 1001, 200
CONFIGS = [
    dict(tag="V1", width=32, depth=3, adam=5000, lbfgs=500, multiplier="bounded"),
    dict(tag="V2", width=64, depth=3, adam=10000, lbfgs=1000, multiplier="bounded"),
    dict(tag="V3", width=32, depth=3, adam=5000, lbfgs=500, multiplier="power"),
]


def rhs(t, z):
    return [z[1], MU * (1.0 - z[0] ** 2) * z[1] - z[0]]


def jac(t, z):
    return np.array([[0.0, 1.0], [-2.0 * MU * z[0] * z[1] - 1.0, MU * (1.0 - z[0] ** 2)]])


def reference(t_test: np.ndarray):
    sol = solve_ivp(rhs, (0.0, T), [Y0, V0], method="DOP853", t_eval=t_test, rtol=2.5e-14,
                    atol=1e-15)
    chk = solve_ivp(rhs, (0.0, T), [Y0, V0], method="Radau", t_eval=t_test, rtol=1e-13,
                    atol=1e-13, jac=jac)
    if not (sol.success and chk.success):
        raise RuntimeError("reference solution failed")
    return sol.y[0], rel_l2(chk.y[0], sol.y[0])


def multiplier(t, kind: str):
    """phi with phi(0) = phi'(0) = 0: t^2 ("power") or t^2 / (1 + t^2) ("bounded")."""
    return t ** 2 if kind == "power" else t ** 2 / (1.0 + t ** 2)


def build_model(width: int, depth: int, kind: str) -> torch.nn.Module:
    net = MLP(1, 1, width=width, depth=depth, activation="tanh", domain=(0.0, T))
    return ConstrainedModel(net, lambda t, n: Y0 + V0 * t + multiplier(t, kind) * n)


def train_config(ex: Example, cfg: dict, t_test: np.ndarray, y_ref: np.ndarray) -> dict:
    torch.manual_seed(ex.seed)
    model = build_model(cfg["width"], cfg["depth"], cfg["multiplier"])
    t_col = to_tensor(uniform(M_COL, 0.0, T), requires_grad=True)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y, dy, d2y = derivatives(model(t_col), t_col, 2)
        r = d2y - MU * (1.0 - y ** 2) * dy + y
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


def main():
    ex = Example("ch10_benchmarking", "ivp_vanderpol",
                 description="Van der Pol (mu=0.5): classical solvers vs PINN (work-precision)",
                 default_seed=1234, extra_args=B.RESUME_ARGS)
    t_test = np.linspace(0.0, T, N_TEST)
    y_ref, ref_check = reference(t_test)
    ex.log(f"reference check (DOP853 vs Radau, rel L2): {ref_check:.2e}")

    rtols = B.RTOLS if not ex.quick else B.RTOLS[::6]
    classical = B.cached(ex, "classical", lambda: B.ivp_sweep(
        rhs, (0.0, T), [Y0, V0], t_test, y_ref.reshape(-1, 1), rtols=rtols, jac=jac,
        components=[0], repeats=ex.pick(B.REPEATS, 1), log=ex.log))
    ex.save_table(classical, "classical")

    pinn_rows, traj_rows, preds = [], [], {"t": t_test, "reference": y_ref}
    for cfg in CONFIGS:
        ex.log(f"PINN {cfg['tag']}: width={cfg['width']} depth={cfg['depth']} "
               f"adam={cfg['adam']} lbfgs={cfg['lbfgs']} multiplier={cfg['multiplier']}")
        out = B.cached(ex, cfg["tag"], lambda: train_config(ex, cfg, t_test, y_ref))
        row = B.pinn_record(cfg["tag"], out, out["pred"], y_ref, width=cfg["width"],
                            depth=cfg["depth"], params=out["params"], collocation=M_COL,
                            multiplier=cfg["multiplier"],
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
        "problem": {"equation": "y'' - mu (1 - y^2) y' + y = 0", "mu": MU, "y0": Y0, "v0": V0,
                    "T": T},
        "reference": "DOP853 rtol=2.5e-14 atol=1e-15", "reference_check_rel_l2": ref_check,
        "test_grid": f"{N_TEST} uniform points on [0, {T}]; error measured on y",
        "classical_protocol": {"methods": B.IVP_METHODS, "rtols": rtols, "atol": "= rtol",
                               "timing": f"median of {B.REPEATS} after 1 warm-up",
                               "jacobian": "analytic (Radau, BDF, LSODA)"},
        "pinn_protocol": {"trial_solution": "1 + phi(t) N(t), phi = t^2/(1+t^2) or t^2", "activation": "tanh",
                          "collocation": f"{M_COL} uniform",
                          "optimizer": "Adam (1e-3 -> 1e-5 cosine) + L-BFGS", "configs": CONFIGS},
        "best_classical_error": B.best_error(classical),
        "pinn": pinn_rows, "matched": matched, "hardware": B.hardware_info(),
    })


if __name__ == "__main__":
    main()
