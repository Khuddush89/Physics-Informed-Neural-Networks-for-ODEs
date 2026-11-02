# Chapter 9 - Training difficulties, diagnostics and remedies

Each script isolates one failure mode of PINN training on a small ODE, demonstrates a remedy and
writes its numbers to `outputs/ch09_training_diagnostics/<example>/results.json` (fixed seed 1234,
double precision, one CPU thread). Every script supports `--quick` (smoke test, seconds) and is
run from the repository root:

```bash
PYTHONPATH=. python chapters/ch09_training_diagnostics/spectral_bias.py           # full run
PYTHONPATH=. python chapters/ch09_training_diagnostics/spectral_bias.py --quick   # smoke test
```

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `spectral_bias.py` | Sec. 9.2, Fig. `spectral_bias_summary`, Tab. `tab_spectral_bias` | y' = f with modes k = 1 and k = 16; tanh MLP vs Fourier features (sigma = 1, 8, 20) vs SIREN; frequency-resolved error; residual NTK spectrum at initialization | about 3 min CPU |
| `loss_imbalance.py` | Sec. 9.3, Fig. `loss_imbalance_summary`, Tab. `tab_loss_imbalance` | y'' = -omega^2 sin(omega t), omega = 6 pi; fixed weights vs rescaled residual vs learning-rate annealing vs self-adaptive weights vs hard BC; per-term gradient norms | about 3 min CPU |
| `trivial_solutions.py` | Sec. 9.4, Fig. `trivial_solutions_summary`, Tab. `tab_trivial_solutions` | logistic ODE: attraction of the equilibrium y = 0 for T = 2..20 (3 seeds), predicted spurious loss; hard IC, curriculum, causal weights, relative residual | about 4 min CPU |
| `stiffness.py` | Sec. 9.5, Fig. `stiffness_summary`, Tab. `tab_stiffness` | y' = -lambda (y - cos t), lambda = 1..1000; vanilla vs hard IC + scaling vs stretched time vs time-marching; Radau and RK45 step counts | about 3 min CPU |
| `causality.py` | Sec. 9.6, Fig. `causality_summary`, Tab. `tab_causality` | harmonic oscillator on 1, 4, 8 periods; vanilla vs causal weights vs time-marching; a posteriori bound int_0^t abs(r) | about 9 min CPU |
| `lorenz_chaos.py` | Sec. 9.7, Fig. `lorenz_summary`, Tab. `tab_lorenz` | Lorenz system: Lyapunov exponent (Benettin), twin trajectories, global PINNs for T = 0.5, 1, 3 and a time-marching PINN (20 windows on [0, 5]) | about 6 min CPU |
| `optimizers.py` | Sec. 9.8, Fig. `optimizers_summary`, Tab. `tab_optimizers` | nonlinear pendulum: Adam, Adam -> L-BFGS, short Adam -> L-BFGS, L-BFGS only, in float64 and float32 | about 4 min CPU |
| `diagnostics.py` | Sec. 9.9, Fig. `diagnostics_summary`, Tabs. `tab_diagnostics_a/b` | training loss vs validation residual (24 points, sigma = 20); residual-based adaptive refinement on the stiff layer | about 1 min CPU |
| `_common.py` | - | helpers: gradient norms, causal weights, CPU timer, Fourier amplitudes, LaTeX tables | - |
| `_make_tables.py` | all tables | copies the `tab_*.tex` fragments (and with `--figures` the figure PDFs) into the book; rebuilds the run-time table of `optimizers.py` from `results.json` | seconds |

Runtimes are process CPU times measured on a shared machine and are indicative only; the exact
values are stored as `cpu_time_s` and `train_time_s` in each `results.json`.

Each training run is cached in `outputs/ch09_training_diagnostics/<example>/_cache/`, so an
interrupted full run resumes where it stopped; after a successful run the cache is moved to
`_last_run/`. The option `--replot` redraws the figures and tables from `_last_run/` without
training.

To refresh the book tables after re-running the scripts (for instance on an idle machine):

```bash
PYTHONPATH=. python chapters/ch09_training_diagnostics/_make_tables.py --dest ../book/figures/ch09 --figures
```

The assignment notebook is `notebooks/ch09_training_diagnostics_assignment.ipynb`.
