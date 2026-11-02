# Chapter 2 - Approximation, Optimization and Error Analysis

Scripts for the activation-function figure and the numerical illustration of the
residual-based error bounds of Chapter 2. Run every script from the repository root, e.g.

```bash
PYTHONPATH=. python chapters/ch02_theory/aposteriori_bounds.py           # full run
PYTHONPATH=. python chapters/ch02_theory/aposteriori_bounds.py --quick   # smoke test
```

Outputs are written to `outputs/ch02_theory/<example>/`; every number quoted in the chapter
is read from the `results.json` and CSV files written by these scripts (seed 1234, double
precision, one CPU thread).

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `activations.py` | Sect. 2.2, Fig. 2.2, Table 2.1 | tanh, sigmoid, sine, softplus, SiLU, GELU and ReLU with first and second derivatives (AD, checked against closed forms); key values quoted in the text (extrema of SiLU and GELU, softplus-ReLU gap, accuracy of the tanh approximation of GELU). | a few seconds |
| `aposteriori_bounds.py` | Sect. 2.5.5, Fig. 2.3, Table 2.2 | trains PINNs (MLP 3x32 tanh, 200 points, 5000 Adam + L-BFGS) for A: y' = -y + sin t on [0,10], B: y' = y(1-y) on [0,6], B': the same on [0,10] (failure case), C: -y'' + y^3 = g on (0,1); evaluates the a posteriori bounds (one-sided / Lipschitz Gronwall bounds, Green's-function bound) after 500 and 5000 Adam iterations and after L-BFGS, and compares them with the true errors. `--plot-only` redraws the figure from saved outputs. | about 14 min* |

\* Measured on a shared, heavily loaded machine; Appendix B of the book lists runtimes
measured on an idle machine.

The assignment notebook for this chapter is `notebooks/ch02_error_bounds_assignment.ipynb`.
