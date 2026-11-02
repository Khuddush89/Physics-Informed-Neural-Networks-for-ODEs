"""Reference solution for the coupled third-order BVP of the Chapter 5 exercise.

    y1''' = -k1 y1 - c1 y1^2 y2' - d1 y1'' + e1 y2 y1',
    y2''' = -k2 y2 + c2 (y1')^2 - d2 y2' + e2 y1^3,          t in [0, 1],
    y1(0) = 0,  y1'(0) = 0.5,  y1''(1) = 0,   y2(0) = 1,  y2'(1) = 0,  y2''(1) = -0.2,

with k1 = 2, k2 = 1.5, c1 = 0.4, c2 = 0.5, d1 = 0.3, d2 = 0.25, e1 = 0.1, e2 = 0.2.

Book: Chapter 5, Exercises (third-order system) and its solution in Appendix C.

The system is written in first-order form z = (y1, y1', y1'', y2, y2', y2'') and solved with
scipy's solve_bvp (tol 1e-10) from three different initial guesses (zero, linear, random).
The zero and the linear guess lead to the same solution; the random guess (seed 1234) leads to
a second solution, so this nonlinear system, like the Duffing problem of the chapter, has more
than one solution. Values of each distinct solution at selected points are recorded (for
"check your answer" cells), and the solution reached from the zero guess is saved on 1001
uniform points.

Run:  PYTHONPATH=. python chapters/ch05_bvp/third_order_system_reference.py   (a few seconds)
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

from pinnode import Example

K1, K2, C1, C2, D1, D2, E1, E2 = 2.0, 1.5, 0.4, 0.5, 0.3, 0.25, 0.1, 0.2


def rhs(t, z):
    y1, y1p, y1pp, y2, y2p, y2pp = z
    y1ppp = -K1 * y1 - C1 * y1 ** 2 * y2p - D1 * y1pp + E1 * y2 * y1p
    y2ppp = -K2 * y2 + C2 * y1p ** 2 - D2 * y2p + E2 * y1 ** 3
    return np.vstack([y1p, y1pp, y1ppp, y2p, y2pp, y2ppp])


def bc(za, zb):
    return np.array([za[0], za[1] - 0.5, zb[2], za[3] - 1.0, zb[4], zb[5] + 0.2])


def main():
    ex = Example("ch05_bvp", "third_order_system_reference",
                 description="Coupled third-order BVP (exercise): solve_bvp reference", default_seed=1234)
    x0 = np.linspace(0.0, 1.0, 50)
    rng = np.random.default_rng(ex.seed)
    guesses = {
        "zero": np.zeros((6, x0.size)),
        "linear": np.vstack([0.5 * x0, 0.5 * np.ones_like(x0), 0 * x0, np.ones_like(x0), 0 * x0, 0 * x0]),
        "random": 3.0 * rng.standard_normal((6, x0.size)),
    }
    t = np.linspace(0.0, 1.0, 1001)
    distinct = []                       # list of dicts: values, guesses, z on the grid
    for name, g in guesses.items():
        sol = solve_bvp(rhs, bc, x0, g, tol=ex.pick(1e-10, 1e-6), max_nodes=100000)
        if not sol.success:
            raise RuntimeError(f"{name}: {sol.message}")
        z = sol.sol(t)
        ex.log(f"guess={name:7s} nodes={sol.x.size:5d} max rms residual={np.max(sol.rms_residuals):.1e}")
        for d in distinct:
            if np.max(np.abs(d["z"][[0, 3]] - z[[0, 3]])) < 1e-6:
                d["guesses"].append(name)
                break
        else:
            distinct.append({"guesses": [name], "z": z, "nodes": int(sol.x.size),
                             "values": {"y1(1)": z[0, -1], "y2(1)": z[3, -1], "y1'(1)": z[1, -1],
                                        "y1''(0)": z[2, 0], "y2'(0)": z[4, 0], "y2''(0)": z[5, 0],
                                        "max|y1|": float(np.max(np.abs(z[0]))),
                                        "max|y2|": float(np.max(np.abs(z[3]))),
                                        "max|y1''|": float(np.max(np.abs(z[2]))),
                                        "max|y2''|": float(np.max(np.abs(z[5])))}})
    for k, d in enumerate(distinct, start=1):
        ex.log(f"solution {k} (from guesses {', '.join(d['guesses'])}, {d['nodes']} nodes): "
               + ", ".join(f"{key}={val:+.6f}" for key, val in d["values"].items()))
    z = distinct[0]["z"]
    spread = (float(np.max(np.abs(distinct[1]["z"][[0, 3]] - z[[0, 3]]))) if len(distinct) > 1 else 0.0)
    ex.log(f"{len(distinct)} distinct solution(s); max |y difference| between solutions 1 and 2: {spread:.3e}")
    ex.save_table({"t": t, "y1": z[0], "dy1": z[1], "d2y1": z[2], "y2": z[3], "dy2": z[4], "d2y2": z[5]},
                  "reference_solution")
    ex.save_results({"parameters": {"k1": K1, "k2": K2, "c1": C1, "c2": C2, "d1": D1, "d2": D2,
                                    "e1": E1, "e2": E2},
                     "solver": "scipy.integrate.solve_bvp, tol 1e-10, 50 initial nodes",
                     "nodes": distinct[0]["nodes"], "values": distinct[0]["values"],
                     "distinct_solutions": [{"guesses": d["guesses"], "nodes": d["nodes"], "values": d["values"]}
                                            for d in distinct],
                     "max_y_difference_between_solutions": spread})


if __name__ == "__main__":
    main()
