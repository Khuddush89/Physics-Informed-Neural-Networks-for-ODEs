# Chapter 6: Terminal value problems, adjoint equations and forward-backward systems

Scripts for Chapter 6 of the book. Run them from the repository root, for example

```bash
PYTHONPATH=. python chapters/ch06_tvp/pendulum_tvp.py            # full run
PYTHONPATH=. python chapters/ch06_tvp/pendulum_tvp.py --quick    # smoke test (seconds)
```

Every script uses double precision on one CPU thread with a fixed seed (1234), writes
`outputs/ch06_tvp/<example>/results.json` (all numbers quoted in the book), CSV histories and
predictions, and the composite figure as PDF/PNG. Finished trainings are cached in the output
folder; `--resume` reuses them after an interruption (recorded training times are those of the
original run).

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `tvp_conditioning.py` | Sec. 6.1, Fig. `tvp:fig:conditioning` | backward vs forward amplification for `y'' + (1+kappa) y' + kappa y = 0` (matrix exponential, perturbed DOP853 integration), TVP-PINN vs IVP-PINN with identical settings, backward sensitivity of the pendulum TVP | 3 min of training |
| `pendulum_tvp.py` | Sec. 6.3.1, Fig. `tvp:fig:pendulum`, Tables `tvp:tab:examples`, `tvp:tab:runtime` | damped pendulum TVP, Legendre network with soft and hard terminal conditions, backward DOP853 reference | 9 min of training |
| `jerk_tvp.py` | Sec. 6.3.2, Fig. `tvp:fig:jerk`, Tables `tvp:tab:examples`, `tvp:tab:runtime` | third-order TVP: LeNN (soft) vs MLP (soft) vs MLP with terminal trial function; backward DOP853 and shooting references | 12 min of training |
| `adjoint_sensitivity.py` | Sec. 6.4, Fig. `tvp:fig:adjoint`, Table `tvp:tab:adjoint` | one PINN for the coupled state/adjoint system of a logistic misfit functional; gradient vs finite differences and the classical adjoint method | 3 min of training |
| `lqr_riccati.py` | Sec. 6.5(a), Fig. `tvp:fig:riccati`, Table `tvp:tab:lqr` | Riccati terminal value problem of the LQR (double integrator) with a trial matrix; exact solution from the Hamiltonian matrix exponential | 4 min of training |
| `lqr_forward_backward.py` | Sec. 6.5(b), Fig. `tvp:fig:fbsystem`, Table `tvp:tab:lqr` | Hamiltonian state/costate system with mixed conditions solved by one PINN; `solve_bvp` and analytic references; cost of the PINN control | 4 min of training |
| `growth_model.py` | Sec. 6.6, Fig. `tvp:fig:growth` | finite-horizon Ramsey-Cass-Koopmans model with terminal capital condition; PINN vs `solve_bvp`; shooting sensitivity | 2 min of training |
| `system_tvp.py` | Exercise `tvp:ex:system` (solution in Appendix C) | two coupled third-order TVPs: backward reference, sensitivity norm, PINN with terminal trial functions | 17 min of training |
| `_make_tables.py` | Tables `tvp:tab:examples`, `tvp:tab:adjoint`, `tvp:tab:lqr`, `tvp:tab:runtime` | writes the LaTeX tables from the `results.json` files (run after the scripts) | seconds |

The runtimes above were measured on a shared two-core machine under heavy load and are
upper bounds; on an idle machine they are considerably shorter. The run-time table of the book is
regenerated with

```bash
PYTHONPATH=. python chapters/ch06_tvp/_make_tables.py --dest ../book/figures/ch06
```

The assignment notebook is `notebooks/ch06_tvp_assignment.ipynb`.
