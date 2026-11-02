# Chapter 5: Boundary value problems

Scripts for Chapter 5 of the book. Run them from the repository root, for example

```bash
PYTHONPATH=. python chapters/ch05_bvp/duffing_pinn.py            # full run
PYTHONPATH=. python chapters/ch05_bvp/duffing_pinn.py --quick    # smoke test (seconds)
```

Every script writes `outputs/ch05_bvp/<example>/results.json` (seed, settings, errors, timings,
library versions) plus CSV files and, where applicable, the figure as PDF and PNG.
The tables of the chapter are generated from these files by `make_tables.py`.

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `duffing_reference.py` | Sec. "Closed-form solutions", Table `bvp:tab:roots` | closed-form solutions `y = A cn(w(t - pi), m)` of `y'' + y + 0.1 y^3 = 0`, `y(0) = 1`, `y'(pi) = 0`; all roots with `|A| <= 12`; checks against `solve_bvp` and shooting; fold of the small-amplitude branch at `eps_c` | a few seconds |
| `duffing_pinn.py` | Sec. "Worked example: a Duffing-type BVP", Fig. `bvp:fig:duffing`, Table `bvp:tab:duffing` | ChNN and MLP PINNs, soft (lambda_bc = 1, 100) and hard (constrained expression) boundary conditions; a run started from a fitted guess converges to the second solution | about 2.2 min CPU |
| `duffing_classical.py` | Sec. "Comparison with classical methods", Table `bvp:tab:classical` | collocation (`solve_bvp`), shooting (DOP853 + Brent) and finite differences + Newton: errors and CPU times | about 1 s CPU (all repeats) |
| `boundary_layer.py` | Sec. "Worked example: a boundary layer", Fig. `bvp:fig:layer`, Table `bvp:tab:layer` | `eps y'' + y' = 1`, `y(0) = 0`, `y(1) + y'(1) = 1`, eps = 0.01: uniform vs Chebyshev-Gauss-Lobatto vs Shishkin collocation points (all three end on a wrong function close to y = t) and CGL points with the layer-enriched network y = N_1 + exp(-t/eps) N_2; mesh chosen by `solve_bvp` for comparison | about 2.3 min CPU |
| `resonance_loss_floor.py` | Sec. "Existence, uniqueness and what the loss sees"; Exercise (loss floor) | `y'' + y = 0`, `y(0) = 0`, `y(pi) = 1` has no solution: PINN losses stall at the predicted floors `2/pi^2` (hard) and `2 lambda/(lambda pi^2 + 4)` (soft) | about 56 s CPU |
| `third_order_system_reference.py` | Exercise (coupled third-order system) and its solution in Appendix C | `solve_bvp` references for the coupled third-order system from three initial guesses (two distinct solutions) | a few seconds |
| `make_tables.py` | all tables of the chapter | writes `tab_*.tex` from the `results.json` files (CPU time by default, `--time wall` for wall-clock time) | under a second |
| `_bvp_common.py` | (helper module) | closed forms, Shishkin points, timing helper, history loader | - |

Regenerating the book tables (after running the scripts above):

```bash
PYTHONPATH=. python chapters/ch05_bvp/make_tables.py --tex-dir ../book/figures/ch05
```

`duffing_pinn.py` and `boundary_layer.py` accept `--plot-only` to redraw the figure from saved outputs,
and `--only <names>` to run a subset of the configurations; `boundary_layer.py` also accepts `--eps` and `--M`.

Notebook assignment: `notebooks/ch05_bvp_assignment.ipynb`.
