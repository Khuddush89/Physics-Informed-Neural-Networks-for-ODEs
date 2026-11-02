"""A-posteriori residual bounds as computable error certificates (logistic ODE).

Book: Chapter 10 (Benchmarking against classical solvers), Section "Where PINNs can pay off",
Figure "residual certificates" and the certificate table.

For y' = f(y) = r y (1 - y), y(0) = y0 in (0, 1), and any C^1 approximation v with residual
R = v' - f(v) and initial defect e0 = v(0) - y0, the error e = v - y satisfies

    f(v) - f(y) = r (1 - v - y) e   and   e' = r (1 - 2v) e + r e^2 + R   (exactly).

Three computable bounds on |e(t)| are evaluated (the first two are those of the a posteriori
theorem for IVPs in Chapter 2):
  crude   - Lipschitz constant L = r (|f'(y)| <= r for 0 <= y <= 1):
            e^{rt}|e0| + int_0^t e^{r(t-s)} |R(s)| ds;
  sharp   - one-sided function mu(t) = r (1 - y0 - v(t)), valid because y(t) >= y0:
            e^{M(t)}|e0| + int_0^t e^{M(t)-M(s)} |R(s)| ds,  M = int mu;
  refined - comparison solution of E' = r (1 - 2v) E + r E^2 + |R|, E(0) = |e0|.
All integrals are evaluated on a fine grid of 20001 points on [0, 2] (cumulative Simpson
rule; the quadratic term of the refined bound by one Picard correction) for
  (i)  the PINN P3 of ivp_logistic.py (32x3 tanh MLP, hard initial condition, Adam 8000 steps
       + L-BFGS <= 800 iterations, seed 1234), retrained here with identical settings;
  (ii) the C^1 cubic Hermite interpolant of the accepted RK45 steps (rtol = atol = 1e-5).

Run:  PYTHONPATH=. python chapters/ch10_benchmarking/residual_certificate.py          (about 0.5 min CPU)
      PYTHONPATH=. python chapters/ch10_benchmarking/residual_certificate.py --quick  (smoke test)
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import numpy as np
import torch
from scipy.integrate import cumulative_simpson, solve_ivp
from scipy.interpolate import CubicHermiteSpline

import ivp_logistic as L
from pinnode import Example, derivative, rel_l2, to_tensor

N_FINE = 20001
RK_RTOL = 1e-5


def _propagate(t: np.ndarray, mu: np.ndarray, src: np.ndarray, e0: float) -> np.ndarray:
    """e^{M(t)}|e0| + int_0^t e^{M(t)-M(s)} src(s) ds with M = int_0^t mu (Simpson rule)."""
    M = cumulative_simpson(mu, x=t, initial=0.0)
    return np.exp(M) * (abs(e0) + cumulative_simpson(np.exp(-M) * src, x=t, initial=0.0))


def bounds(t: np.ndarray, v: np.ndarray, res: np.ndarray, e0: float = 0.0):
    """Crude (Lipschitz), sharp (one-sided, Chapter 2) and refined (comparison) bounds."""
    r, absR = L.R, np.abs(res)
    crude = _propagate(t, np.full_like(t, r), absR, e0)
    sharp = _propagate(t, r * (1.0 - L.Y0 - v), absR, e0)
    mu_lin = r * (1.0 - 2.0 * v)
    lin = _propagate(t, mu_lin, absR, e0)
    refined = _propagate(t, mu_lin, absR + r * lin ** 2, e0)     # one Picard correction
    return crude, sharp, refined, float(np.max(refined - lin))


def main():
    ex = Example("ch10_benchmarking", "residual_certificate",
                 description="A-posteriori residual bounds for a PINN and an RK45 interpolant",
                 default_seed=1234, extra_args=B.RESUME_ARGS)
    t_test = np.linspace(0.0, L.T, L.N_TEST)
    y_test = L.exact(t_test)
    t = np.linspace(0.0, L.T, N_FINE)
    y = L.exact(t)

    # ------------------------------------------------------------- (i) PINN P3
    cfg = next(c for c in L.CONFIGS if c["tag"] == "P3")
    out = L.train_config(ex, cfg, t_test, y_test)
    model = out["model"]
    tt = to_tensor(t, requires_grad=True)
    v_t = model(tt)
    dv = derivative(v_t, tt)
    v = v_t.detach().numpy().ravel()
    res_pinn = (dv.detach().numpy().ravel() - L.R * v * (1.0 - v))
    e0_pinn = float(model(to_tensor([0.0])).item() - L.Y0)

    # ------------------------------------------------------------- (ii) RK45 + cubic Hermite
    sol = solve_ivp(L.rhs, (0.0, L.T), [L.Y0], method="RK45", rtol=RK_RTOL, atol=RK_RTOL)
    H = CubicHermiteSpline(sol.t, sol.y[0], L.rhs(sol.t, sol.y[0]))
    h = H(t)
    res_rk = H(t, 1) - L.R * h * (1.0 - h)

    rows, curves = [], {"t": t}
    for name, approx, res, e0, extra in (
            ("PINN P3", v, res_pinn, e0_pinn, {"rel_l2_test": rel_l2(out["pred"], y_test)}),
            ("RK45 + Hermite", h, res_rk, 0.0, {"rk_steps": int(sol.t.size - 1),
                                                "rk_rtol": RK_RTOL,
                                                "rel_l2_test": rel_l2(H(t_test), y_test)})):
        crude, sharp, refined, quad_term = bounds(t, approx, res, e0)
        err = np.abs(approx - y)
        # Where the residual keeps one sign (e.g. inside the first RK45 step) a bound is attained
        # exactly, and the comparison is decided by the quadrature error of the fine grid
        # (relative size below 1e-4 with Simpson's rule); the check allows for it.
        tol = err * (1 - 1e-4) - 1e-16
        row = {"approximation": name, "min_value": float(approx.min()),
               "max_abs_residual": float(np.max(np.abs(res))), "e0": e0,
               "max_error": float(err.max()), "error_T": float(err[-1]),
               "max_crude_bound": float(crude.max()), "max_sharp_bound": float(sharp.max()),
               "max_refined_bound": float(refined.max()),
               "effectivity_crude": float(crude.max() / err.max()),
               "effectivity_sharp": float(sharp.max() / err.max()),
               "effectivity_refined": float(refined.max() / err.max()),
               "bounds_hold_everywhere": bool(np.all(crude >= tol) and np.all(sharp >= tol)
                                              and np.all(refined >= tol)),
               "min_refined_over_error": float(np.min(refined[err > 0] / err[err > 0])),
               "quadratic_term_contribution": quad_term, **extra}
        rows.append(row)
        key = "pinn" if name.startswith("PINN") else "rk45"
        curves.update({f"{key}_error": err, f"{key}_residual": res, f"{key}_crude": crude,
                       f"{key}_sharp": sharp, f"{key}_refined": refined})
        ex.log(f"{name}: max|e|={row['max_error']:.2e}  refined={row['max_refined_bound']:.2e} "
               f"sharp={row['max_sharp_bound']:.2e} crude={row['max_crude_bound']:.2e} "
               f"max|R|={row['max_abs_residual']:.2e} holds={row['bounds_hold_everywhere']}")
    curves["rk45_nodes"] = np.interp(t, sol.t, np.arange(sol.t.size))   # step index (for plots)
    ex.save_table(rows, "certificates")
    ex.save_table(curves, "curves")
    ex.save_table({"t_nodes": sol.t}, "rk45_nodes")
    ex.save_results({
        "problem": {"equation": "y' = r y (1 - y)", "r": L.R, "y0": L.Y0, "T": L.T},
        "fine_grid": f"{N_FINE} uniform points on [0, {L.T}]",
        "pinn_config": cfg, "rk45_rtol": RK_RTOL, "certificates": rows,
        "hardware": B.hardware_info(),
    })


if __name__ == "__main__":
    main()
