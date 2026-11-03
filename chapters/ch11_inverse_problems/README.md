# Chapter 11 — Inverse problems, data assimilation and uncertainty quantification

All scripts run from the repository root, use double precision on one CPU thread, fix their
seeds, and write `results.json`, CSV files, figures and the LaTeX tables included by the book
(`tab_*.tex`) to `outputs/ch11_inverse_problems/<script>/`. `--quick` runs a smoke test (results
not meaningful); `--resume` reuses fits cached by an interrupted run (identical results).

| script | book section / figure / table | what it does | CPU time (full run, 1 core) |
|---|---|---|---|
| `manual_inverse.py` | Sec. "An inverse problem computed by hand" | y' = -p y with data (0.62, 0.36) at t = (0.5, 1): Gauss-Newton iterations for output least squares; inverse PINN loss, gradient with respect to p and one Adam step at the initialization | < 1 s |
| `logistic_parameter_id.py` | Sec. 11.2, Fig. `inv:fig:logistic`, Tab. `inv:tab:logistic` | logistic ODE, (r, K) from 15 noisy points at 4 noise levels x 5 data sets: inverse PINN vs `least_squares` + `solve_ivp`; practical identifiability (early-window data) | about 2.7 min |
| `logistic_weight_study.py` | Sec. 11.1/11.7, Fig. `inv:fig:weights` | effect of the residual weight lam_r (1e-4 ... 1e3): parameter errors, discrepancy principle, validation misfit | about 1.0 min |
| `lotka_volterra_partial.py` | Sec. 11.3, Fig. `inv:fig:lv`, Tab. `inv:tab:lv` | Lotka-Volterra observed through the prey only (log-normal noise): parameters and unobserved predator; least squares from the same and from 20 random initial guesses; structural non-identifiability when y(0) is unknown | about 4.2 min |
| `source_identification.py` | Sec. 11.4, Fig. `inv:fig:source`, Tab. `inv:tab:source` | unknown input rate f(t) in y' = -k y + f(t) with a second network; H^1 regularization sweep, validation and discrepancy principle | about 2.2 min |
| `volterra_parameter_id.py` | Sec. 11.5, Fig. `inv:fig:volterra`, Tab. `inv:tab:volterra` | lambda in u(x) = 1 + lambda int_0^x s u(s) ds from noisy samples: PINN (lam_r = 1 and 100), integral-form and differential-form estimators, least squares with a Volterra solver | about 4.0 min |
| `uq_ensembles.py` | Sec. 11.6, Fig. `inv:fig:uq`, Tab. `inv:tab:uq` | uncertainty of (r, K): random-walk Metropolis reference, Gauss-Newton covariance, deep ensemble (initialization only) and ensemble with perturbed data | about 1.9 min |
| `_common.py` | — | shared helpers (logistic model, inverse PINN, classical fit, caching, LaTeX tables) | — |
| `_reproduce.py` | — | runs all scripts in sequence (restarting killed runs with `--resume`) and copies figures and tables to `book/figures/ch11/` | about 16 min |

Reproduce everything and refresh the book assets:

```bash
PYTHONPATH=. python chapters/ch11_inverse_problems/_reproduce.py --book ../book
```

Timing: the tables report the CPU time of the process (one thread), which on an idle machine
equals the wall-clock time; the wall-clock times are stored in `results.json` as well.
