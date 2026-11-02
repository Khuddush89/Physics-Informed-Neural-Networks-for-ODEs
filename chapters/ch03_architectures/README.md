# Chapter 3 - Architectures and exact enforcement of conditions

Scripts for Chapter 3 of the book. Every number, table and figure of the chapter is produced
here with a fixed seed (1234), double precision and one CPU thread. Run from the repository root:

```bash
PYTHONPATH=. python chapters/ch03_architectures/feature_conditioning.py
PYTHONPATH=. python chapters/ch03_architectures/architecture_comparison.py
PYTHONPATH=. python chapters/ch03_architectures/soft_vs_hard.py
PYTHONPATH=. python chapters/ch03_architectures/soft_vs_hard.py --network chnn \
        --outdir outputs/ch03_architectures/soft_vs_hard_chnn --no-figures
PYTHONPATH=. python chapters/ch03_architectures/well_posedness.py
PYTHONPATH=. python chapters/ch03_architectures/_make_book_assets.py --book ../book   # tables + figures
```

Add `--quick` to any experiment script for a smoke test (seconds; results not meaningful).
`architecture_comparison.py` accepts `--archs MLP,ChNN,...`, `soft_vs_hard.py` accepts
`--only ivp:soft,robin:hard`, and both accept `--adam N --lbfgs N` to change the training budget
(useful for the exercises). Outputs go to `outputs/ch03_architectures/<script>/`.

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `feature_conditioning.py` | Sec. "Functional-link polynomial networks"; Fig. `arch:fig:features`, Table `arch:tab:conditioning` | checks the Chebyshev/Legendre feature maps of `pinnode` (AD of the recurrences) against NumPy and the endpoint formulas; condition numbers of feature matrices for monomial, scaled-monomial, Chebyshev and Legendre bases | a few seconds |
| `architecture_comparison.py` | Sec. "Numerical comparison"; Fig. `arch:fig:architectures`, Table `arch:tab:architectures` | Airy oscillator BVP `y'' + t y = 0`, `y(0)=Ai(0)`, `y(10)=Ai(-10)`: MLP, ChNN, LeNN, Fourier features and SIREN (two frequency scales) with the same body, trial function, collocation set and optimizer | 39 min wall-clock on the shared 2-core machine used for the book (load average about 16); roughly 5 min on an idle core |
| `soft_vs_hard.py` | Sec. "Numerical comparison"; Fig. `arch:fig:softhard`, Table `arch:tab:softhard` | same equation as IVP and as Dirichlet-Robin BVP; soft penalties vs hard constraints (IVP: multiplier `t^2` vs `(t/T)^2`; BVP: constrained expression) with the Fourier-feature network (`--network chnn` for the Chebyshev network); residual diagnostics on the test grid; the ChNN variant (`--network chnn`, 18 min wall-clock) supplies the last two columns of the table | 15 min wall-clock on the shared machine (load average about 8-10); roughly 3-4 min on an idle core |
| `well_posedness.py` | Sec. "The Airy oscillator" and the soft/hard discussion | unique solvability (`phi(10)`, Robin analogue) and smallest eigenvalue moduli of the Dirichlet and Robin operators (Chebyshev collocation) | about one second |
| `_make_book_assets.py` | all tables of the chapter | regenerates `tab_*.tex` from `results.json` (including timings) and copies the figure PDFs into `book/figures/ch03/` | seconds |
| `_ch03_common.py` | - | shared problem definitions (exact solution `Ai(-t)`, transforms, Dirichlet-Robin constrained expression, well-posedness check) | - |

Wall-clock times depend strongly on the machine load; the timing columns of the tables are
regenerated from `train_time_s` in `results.json` by `_make_book_assets.py`, so re-running the two
training scripts on an idle machine followed by `_make_book_assets.py` refreshes them. With the
same software versions, double precision and one thread, the accuracy numbers are reproduced
exactly. `--plot-only` redraws a figure from the saved CSV files without retraining.

Notebook assignment: `notebooks/ch03_architectures_assignment.ipynb`.
