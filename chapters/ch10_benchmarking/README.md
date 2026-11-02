# Chapter 10 - Benchmarking against classical solvers

Accuracy-versus-cost comparisons of PINNs with classical solvers at matched accuracy
(work-precision diagrams). Every timing number of the chapter is produced by these scripts
and written into the generated LaTeX tables; nothing is typed in by hand.

## Reproduce the chapter

```bash
PYTHONPATH=. python chapters/ch10_benchmarking/write_book_assets.py --run   # all examples + assets
```

`--run` executes the seven examples below one after the other and then writes all figures
(`outputs/ch10_benchmarking/book_assets/figures/`) and LaTeX tables
(`outputs/ch10_benchmarking/book_assets/tables/`, including `numbers.tex` with the accuracy
numbers quoted in the text). If the book sources are found next to the repository
(`../book/figures/ch10`) or given with `--book-dir`, the PDFs and `.tex` files are copied there.
Without `--run` the assets are rebuilt from existing outputs.

* Timing protocol: one thread for PyTorch and BLAS, double precision, one warm-up call and the
  median of 5 timed calls for every classical run, one seeded training run (seed 1234) per PINN
  configuration. The reported cost is the **CPU time** of the process (`time.process_time`),
  which equals the wall-clock time on an idle machine and is not inflated when other jobs
  share the processor; wall-clock times (`time.perf_counter`, `hist.train_time`) are recorded
  in every `results.json`. `--cost wall` builds the tables from wall-clock times instead.
* `--resume` reuses PINN configurations and classical sweeps cached by an interrupted run
  (`outputs/ch10_benchmarking/<example>/cache/`); a plain run recomputes everything.
* `--quick` is a smoke test (tiny runs; results are not meaningful and nothing is copied).

## Scripts

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `ivp_logistic.py` | Sec. "Initial value problems"; Fig. `bench:fig:ivp-wp`(a); Tabs. `bench:tab:ivp-matched`, `bench:tab:ivp-pinn` | logistic ODE: `solve_ivp` sweep (RK45, DOP853, Radau, BDF, LSODA; 19 tolerances) vs four PINN configurations | 1.4 min |
| `ivp_vanderpol.py` | same, panel (b) | van der Pol (mu = 0.5, [0, 10]): same sweep vs three PINNs (bounded and unbounded hard-IC multiplier) | 3.7 min |
| `ivp_stiff.py` | same, panel (c) | stiff linear system (eigenvalues -1, -1000): same sweep vs PINNs with uniform and layer-adapted collocation | 2.7 min |
| `bvp_bratu.py` | Sec. "Boundary value problems"; Fig. `bench:fig:bvp`; Tab. `bench:tab:bvp` | Bratu BVP: `solve_bvp`, shooting (DOP853 + Brent), finite differences + Newton vs PINN (MLP, ChNN) | 1.9 min |
| `integral_equations.py` | Sec. "Integral and integro-differential equations"; Fig. `bench:fig:ie`; Tab. `bench:tab:ie` | two Fredholm, one Volterra and one integro-differential equation: Nystrom, trapezoidal product integration, ODE reformulation vs PINN | 1.4 min |
| `parametric_logistic.py` | Sec. "Where PINNs can pay off" (parametric family); Fig. `bench:fig:param`; Tab. `bench:tab:param` | one PINN y(t, r) for r in [1, 3] vs repeated RK45/DOP853 solves and a Chebyshev surrogate in r; break-even counts | 2.9 min |
| `residual_certificate.py` | Sec. "Where PINNs can pay off" (certificates); Fig. `bench:fig:certificate`; Tab. `bench:tab:certificate` | crude, sharp and refined a posteriori bounds for a PINN and for the cubic Hermite interpolant of RK45 | 0.5 min |
| `write_book_assets.py` | all figures and tables of Chapter 10 | driver: `--run` runs all of the above, then writes figures, tables and `numbers.tex` | about 15 min (with `--run`) |
| `_bench.py` | - | shared helpers: timing protocol, `solve_ivp` sweeps, PINN bookkeeping (residual-evaluation counter, error-vs-time trajectory), matched-accuracy extraction, plot style | - |

Runtimes are CPU times of a full run on one core of the machine described in the
environment table of the chapter (Intel Xeon, 2.8 GHz).

## Notebook

`notebooks/ch10_benchmarking_assignment.ipynb`: timing protocol, a classical sweep, a small PINN
with its full cost, a work-precision diagram, break-even counts and the stability-limited cost
of an explicit method on a stiff problem (runs on a CPU in a few minutes).
