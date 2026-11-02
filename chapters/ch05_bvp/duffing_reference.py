"""Closed-form solutions of the Duffing-type BVP and their verification (no neural networks).

    y'' + y + eps y^3 = 0  on [0, pi],   y(0) = 1,   y'(pi) = 0,   eps = 0.1.

Book: Chapter 5, Section "Worked example: a Duffing-type BVP with mixed boundary
conditions", subsection "Closed-form solutions" (table of solutions, verification numbers).

Since the equation conserves  E = y'^2/2 + y^2/2 + eps y^4/4,  every solution with y'(pi) = 0
is  y(t) = A cn(w (t - pi), m),  w^2 = 1 + eps A^2,  m = k^2 = eps A^2 / (2 w^2),  where
A = y(pi). The remaining condition y(0) = 1 is the scalar equation  F(A) = A cn(w pi, m) - 1 = 0,
which has infinitely many roots. This script

1. finds all roots with |A| <= 12 (sign changes on a fine grid + Brent's method) and records,
   for each solution, y'(0), max |y| and the number of zeros in (0, pi);
2. checks that the closed form satisfies the ODE (y'' from the derivative formulas of
   sn, cn, dn);
3. compares the smallest-amplitude solution with scipy's solve_bvp (tol 1e-10, and the
   settings 100 nodes, tol 1e-6) and with shooting (DOP853 at rtol = atol = 1e-13 plus
   Brent's method on y'(0));
4. follows the two smallest-amplitude solutions (A_1, A_2 < 0) as eps grows: they merge at a
   fold eps_c (where max F = 0 on the interval between them), beyond which the solution close
   to cos t no longer exists; the smallest roots for a few values of eps are also recorded.

Run:  PYTHONPATH=. python chapters/ch05_bvp/duffing_reference.py          (a few seconds)
      PYTHONPATH=. python chapters/ch05_bvp/duffing_reference.py --quick
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ellipj

from pinnode import Example
from pinnode.reference import bvp_reference

from _bvp_common import (DY_END, EPS_DUFFING, T_END, Y0, duffing_A1, duffing_bc_residual,
                         duffing_exact, duffing_exact_dt, duffing_params, duffing_rhs_system,
                         duffing_roots)


def second_derivative(t, A):
    """y'' = -A w^2 cn (dn^2 - m sn^2), from (sn)' = cn dn, (cn)' = -sn dn, (dn)' = -m sn cn."""
    w, m = duffing_params(A)
    sn, cn, dn, _ = ellipj(w * (t - T_END), m)
    return -A * w ** 2 * cn * (dn ** 2 - m * sn ** 2)


def fold_point(eps_lo: float = 0.2, eps_hi: float = 0.5):
    """eps at which the roots A_1, A_2 < 0 merge (F has a double root), and the root there."""
    from _bvp_common import duffing_bc_function

    def peak(eps):
        As = np.linspace(-3.0, -0.95, 20001)
        f = duffing_bc_function(As, eps)
        i = int(np.argmax(f))
        res = minimize_scalar(lambda a: -float(duffing_bc_function(a, eps)),
                              bracket=(As[i - 1], As[i], As[i + 1]), tol=1e-12)
        return -res.fun, float(res.x)

    eps_c = brentq(lambda e: peak(e)[0], eps_lo, eps_hi, xtol=1e-12)
    return eps_c, peak(eps_c)[1]


def main():
    ex = Example("ch05_bvp", "duffing_reference",
                 description="Duffing-type BVP: closed-form solutions, multiplicity, verification",
                 default_seed=1234)
    t = np.linspace(0.0, T_END, 1001)
    t_fine = np.linspace(0.0, T_END, ex.pick(100001, 10001))
    roots = duffing_roots(A_max=12.0, n_grid=ex.pick(240001, 24001))
    rows = []
    for k, A in enumerate(roots, start=1):
        w, m = duffing_params(A)
        y = duffing_exact(t_fine, A)
        ode_res = second_derivative(t_fine, A) + y + EPS_DUFFING * y ** 3
        rows.append({"k": k, "A": A, "dy0": float(duffing_exact_dt(0.0, A)), "w": float(w), "m": float(m),
                     "max_abs_y": float(np.max(np.abs(y))),
                     "zeros": int(np.sum(np.sign(y[:-1]) * np.sign(y[1:]) < 0)),
                     "bc_y0_error": float(abs(duffing_exact(0.0, A) - Y0)),
                     "bc_dy_end_error": float(abs(duffing_exact_dt(T_END, A) - DY_END)),
                     "max_ode_residual": float(np.max(np.abs(ode_res)))})
        ex.log(f"solution {k}: A=y(pi)={A:+.10f}  y'(0)={rows[-1]['dy0']:+.8f}  max|y|={rows[-1]['max_abs_y']:.4f}"
               f"  zeros={rows[-1]['zeros']}  max ODE residual={rows[-1]['max_ode_residual']:.1e}")
    ex.save_table(rows, "solutions")

    A1 = duffing_A1()
    y1 = duffing_exact(t, A1)
    # (a) collocation with tight tolerance, zero initial guess
    y_bvp = bvp_reference(duffing_rhs_system, duffing_bc_residual, t, np.zeros((2, 200)), tol=1e-10)[:, 0]
    # (b) the settings of the original code (100 nodes, zero guess, tol 1e-6)
    sol_old = solve_bvp(duffing_rhs_system, duffing_bc_residual, np.linspace(0, T_END, 100),
                        np.zeros((2, 100)), tol=1e-6, max_nodes=1000)
    y_old = sol_old.sol(t)[0]
    # (c) shooting on s = y'(0)
    rhs = lambda tt, z: [z[1], -z[0] - EPS_DUFFING * z[0] ** 3]

    def G(s):
        return solve_ivp(rhs, (0.0, T_END), [Y0, s], method="DOP853", rtol=1e-13, atol=1e-13).y[1, -1] - DY_END

    s1 = brentq(G, -1.0, 1.0, xtol=1e-15)
    y_shoot = solve_ivp(rhs, (0.0, T_END), [Y0, s1], method="DOP853", rtol=1e-13, atol=1e-13,
                        t_eval=t).y[0]
    verification = {
        "A1": A1, "dy0_closed_form": float(duffing_exact_dt(0.0, A1)), "dy0_shooting": s1,
        "max_diff_solve_bvp_tol1e-10": float(np.max(np.abs(y_bvp - y1))),
        "max_diff_solve_bvp_original_settings": float(np.max(np.abs(y_old - y1))),
        "original_settings_success": bool(sol_old.success),
        "max_diff_shooting_dop853_1e-13": float(np.max(np.abs(y_shoot - y1))),
        "max_ode_residual_closed_form": rows[0]["max_ode_residual"],
        "max_y_and_location": [float(np.max(y1)), float(t[np.argmax(y1)])],
    }
    for kk, v in verification.items():
        ex.log(f"{kk}: {v}")
    eps_c, A_c = fold_point()
    family = {f"{e:g}": [float(a) for a in duffing_roots(A_max=4.0, n_grid=ex.pick(80001, 8001), eps=e)[:3]]
              for e in (0.05, 0.1, 0.25, 0.3, 0.375, 0.4, 0.5)}
    ex.log(f"fold of the small-amplitude branch: eps_c = {eps_c:.6f}, A = {A_c:.6f}")
    for e, rts in family.items():
        ex.log(f"eps = {e:5s}: smallest roots {', '.join(f'{a:+.5f}' for a in rts)}")
    ex.save_results({"problem": {"eps": EPS_DUFFING, "interval": [0.0, T_END], "bc": "y(0)=1, y'(pi)=0",
                                 "closed_form": "y = A cn(w (t - pi), m), w^2 = 1 + eps A^2, m = eps A^2/(2 w^2)"},
                     "solutions": rows, "verification": verification,
                     "fold": {"eps_c": eps_c, "A_c": A_c,
                              "smallest_roots_by_eps": family}})


if __name__ == "__main__":
    main()
