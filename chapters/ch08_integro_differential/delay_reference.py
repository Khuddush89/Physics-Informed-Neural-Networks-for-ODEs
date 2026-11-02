"""Reference solution of the delay integro-differential equation of the Chapter 8 exercise

    u'(x) = alpha u(x) + beta u(x - tau) + lam x int_0^x t u(t - tau) dt,   0 <= x <= 1,
    u(x)  = phi(x) = 1 + x                                       for -tau <= x <= 0,

with alpha = 1, beta = 1/2, lam = 1, tau = 0.2.  (No PINN is trained here: this script
provides the classical reference against which the exercise and the notebook
notebooks/ch08_ide_assignment.ipynb check a PINN.)

Book: Chapter 8, Exercise "Delay integro-differential equation" (check values u(x_k)).

With the history moment C(x) = int_0^x t u(t - tau) dt the equation becomes the delay system
    u' = alpha u + beta u(x - tau) + lam x C,     C' = x u(x - tau),     u(0) = 1, C(0) = 0.
Method of steps: on [k tau, (k+1) tau] the delayed values are known (phi for k = 0, the dense
output of the previous step otherwise), so each step is an ODE system solved with DOP853
(rtol = atol = 1e-12).  An independent check uses the trapezoidal rule with h = tau / 4000,
for which the delayed arguments are grid points (second order; the breaking points
x = k tau are grid points as well).

Breaking points: u' jumps at x = 0 (u'(0+) = alpha phi(0) + beta phi(-tau) = 1.4, phi' = 1), so
u'' jumps at x = tau by beta (u'(0+) - phi'(0-)) = 0.2, u''' at x = 2 tau, and so on.

Run:  PYTHONPATH=. python chapters/ch08_integro_differential/delay_reference.py   (a few seconds)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from pinnode import Example

ALPHA, BETA, LAM, TAU = 1.0, 0.5, 1.0, 0.2
X_END = 1.0
CHECK_POINTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def phi(x):
    return 1.0 + np.asarray(x, dtype=float)


def method_of_steps(rtol: float = 1e-12, atol: float = 1e-12):
    """Return a callable u(x) on [-tau, 1] and the list of DOP853 segment solutions."""
    segments = []

    def u_delayed(x):                       # u(x - tau) using history or previous segments
        s = x - TAU
        if s <= 0.0:
            return float(phi(s))
        k = min(int(s // TAU), len(segments) - 1)
        return float(segments[k].sol(s)[0])

    y = np.array([phi(0.0), 0.0])
    n_steps = int(round(X_END / TAU))
    for k in range(n_steps):
        a, b = k * TAU, (k + 1) * TAU

        def rhs(x, yy):
            ud = u_delayed(x)
            return [ALPHA * yy[0] + BETA * ud + LAM * x * yy[1], x * ud]
        sol = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=rtol, atol=atol, dense_output=True)
        if not sol.success:  # pragma: no cover
            raise RuntimeError(sol.message)
        segments.append(sol)
        y = sol.y[:, -1]

    def u(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x)
        for i, xi in enumerate(x):
            if xi <= 0.0:
                out[i] = phi(xi)
            else:
                k = min(int(xi // TAU), n_steps - 1)
                out[i] = segments[k].sol(xi)[0]
        return out

    def C(x):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        out = np.empty_like(x)
        for i, xi in enumerate(x):
            k = min(int(max(xi, 0.0) // TAU), n_steps - 1)
            out[i] = segments[k].sol(max(xi, 0.0))[1]
        return out

    return u, C


def trapezoid_check(n_per_tau: int = 4000):
    """Trapezoidal rule on a grid that contains all delayed arguments (h = tau / n_per_tau)."""
    h = TAU / n_per_tau
    N = int(round(X_END / h))
    m = n_per_tau
    x = np.linspace(-TAU, X_END, N + m + 1)          # index j <-> x = -tau + j h
    u = np.empty_like(x)
    u[: m + 1] = phi(x[: m + 1])
    C = 0.0
    F = ALPHA * u[m] + BETA * u[0] + LAM * x[m] * C
    for j in range(m, N + m):
        xj, xj1 = x[j], x[j + 1]
        C_new = C + 0.5 * h * (xj * u[j - m] + xj1 * u[j + 1 - m])
        rest = BETA * u[j + 1 - m] + LAM * xj1 * C_new
        u[j + 1] = (u[j] + 0.5 * h * (F + rest)) / (1.0 - 0.5 * h * ALPHA)
        F = ALPHA * u[j + 1] + rest
        C = C_new
    return x[m:], u[m:]


def main():
    ex = Example("ch08_integro_differential", "delay_reference",
                 description="Delay IDE of the exercise: method-of-steps reference", default_seed=0)
    u, C = method_of_steps()
    x_test = np.linspace(0.0, X_END, 1001)
    u_ref, C_ref = u(x_test), C(x_test)
    xt, ut = trapezoid_check(ex.pick(4000, 200))
    diff = float(np.max(np.abs(np.interp(x_test, xt, ut) - u_ref)))
    eps = 1e-6
    du0 = (u(eps)[0] - u(0.0)[0]) / eps                 # one-sided difference quotients
    jump_u1_at_0 = ALPHA * phi(0.0) + BETA * phi(-TAU) - 1.0
    checks = {f"{xk:.2f}": float(u(xk)[0]) for xk in CHECK_POINTS}
    ex.save_table({"x": x_test, "u_ref": u_ref, "C_ref": C_ref}, "reference")
    ex.save_results({
        "problem": {"alpha": ALPHA, "beta": BETA, "lambda": LAM, "tau": TAU,
                    "history": "phi(x) = 1 + x", "interval": [0.0, X_END]},
        "u_checkpoints": checks,
        "C_at_1": float(C(1.0)[0]),
        "max_abs_difference_to_trapezoid": diff,
        "trapezoid_step": TAU / ex.pick(4000, 200),
        "breaking_points": {"jump_du_at_0": float(jump_u1_at_0),
                            "du_right_at_0_numerical": float(du0),
                            "jump_d2u_at_tau": float(BETA * jump_u1_at_0)},
    })
    for k, v in checks.items():
        ex.log(f"u({k}) = {v:.10f}")
    ex.log(f"C(1) = {float(C(1.0)[0]):.10f};  max |u_MoS - u_trap| = {diff:.2e}")


if __name__ == "__main__":
    main()
