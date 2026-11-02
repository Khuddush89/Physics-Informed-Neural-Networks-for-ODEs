"""Well-posedness indicators of the Airy oscillator problems used in Chapter 3.

Book: Chapter 3, Section "The Airy oscillator" (arch:sec:airy) and the discussion of the
soft/hard experiment (arch:sec:softhard-num).

For y'' + t y = 0 on [0, 10] the script records
* phi(10) and phi'(10) + phi(10) for phi(t) = Bi(0) Ai(-t) - Ai(0) Bi(-t), whose nonvanishing
  shows that the Dirichlet and Dirichlet-Robin problems are uniquely solvable;
* the smallest eigenvalue modulus of v -> v'' + t v with v(0) = v(10) = 0 and with
  v(0) = 0, v'(10) + v(10) = 0 (Chebyshev collocation, 120 and 160 points); both operators are
  self-adjoint, so ||e||_L2 <= ||R||_L2 / min|eig| for a trial function satisfying the
  conditions exactly.

Run:  PYTHONPATH=. python chapters/ch03_architectures/well_posedness.py      (about one second)
"""
from __future__ import annotations

from _ch03_common import well_posedness
from pinnode import Example


def main():
    ex = Example("ch03_architectures", "well_posedness",
                 description="Airy oscillator: unique solvability and conditioning", default_seed=1234)
    res = {f"n_cheb={n}": well_posedness(n) for n in (120, 160)}
    for k, v in res.items():
        ex.log(f"{k}: " + ", ".join(f"{a}={b:.4f}" for a, b in v.items()))
    ex.save_results({"indicators": res})


if __name__ == "__main__":
    main()
