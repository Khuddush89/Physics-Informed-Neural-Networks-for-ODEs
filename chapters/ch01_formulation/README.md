# Chapter 1 - Physics-Informed Neural Networks: Formulation and Loss Design

Scripts for the worked example and the automatic-differentiation figure of Chapter 1.
Run every script from the repository root, for example

```bash
PYTHONPATH=. python chapters/ch01_formulation/first_pinn.py           # full run
PYTHONPATH=. python chapters/ch01_formulation/first_pinn.py --quick   # smoke test (seconds)
```

Outputs (CSV, JSON, PDF and PNG) are written to `outputs/ch01_formulation/<example>/`.
All numbers quoted in the chapter are read from the `results.json` files written by these
scripts (seed 1234, double precision, one CPU thread).

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `first_pinn.py` | Sect. 1.5.6 (A first example), Fig. 1.3, Table 1.3, code listing of Sect. 1.5.5 | PINN for y' = -y + sin t, y(0) = 1 on [0, 10] with soft and hard initial condition (MLP 3x32 tanh, 200 collocation points, 5000 Adam + L-BFGS); comparison with RK45 and DOP853 (errors and numbers of right-hand-side evaluations). `--plot-only` redraws the figure from saved outputs. | 4-7 min* |
| `ad_exactness_cost.py` | Sect. 1.5.4 (Automatic differentiation), Fig. 1.2 | (a) AD versus central finite differences for the first two derivatives of a one-hidden-layer tanh network with known closed-form derivatives; (b) floating-point operations (matrix products) of residuals containing derivatives of order 0-4 and of the corresponding loss-and-gradient evaluation. | under 1 min* |

\* Measured on a shared, heavily loaded machine; Appendix B of the book lists runtimes
measured on an idle machine.

The assignment notebook for this chapter is `notebooks/ch01_first_pinn_assignment.ipynb`.
