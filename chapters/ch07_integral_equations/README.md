# Chapter 7: Integral equations

Scripts for the worked examples of Chapter 7 (Fredholm, Volterra and Helmholtz-type
integral equations). Run them from the repository root, one at a time:

```bash
PYTHONPATH=. python chapters/ch07_integral_equations/fredholm_degenerate.py
PYTHONPATH=. python chapters/ch07_integral_equations/volterra_apinn.py
PYTHONPATH=. python chapters/ch07_integral_equations/helmholtz_filon.py
PYTHONPATH=. python chapters/ch07_integral_equations/filon_weights.py
PYTHONPATH=. python chapters/ch07_integral_equations/_tables.py      # LaTeX tables of the chapter
```

Every script accepts `--quick` (smoke test, seconds), `--seed`, `--no-figures`, and the three
PINN scripts accept `--replot` (redraw the figure from `figdata.npz` of a previous full run).
Outputs go to `outputs/ch07_integral_equations/<example>/` (`results.json`, CSV histories and
predictions, `figures/*.pdf|png`). All numbers quoted in the book come from these files
(default seed 1234, double precision, one CPU thread).

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `fredholm_degenerate.py` | Sec. "A Fredholm equation with a degenerate kernel"; Fig. `ie:fig:fredholm`, Table `ie:tab:fredholm` | u = 1 + (1/2) int_0^1 x t u dt (exact 1 + 3x/10): PINN with Gauss quadrature (Q = 2 and Q = 1) and with an auxiliary scalar (3 seeds each for Q = 2); Nystrom comparison; a posteriori bound | 2.9 min |
| `volterra_apinn.py` | Sec. "Volterra equations", worked example; Fig. `ie:fig:volterra`, Table `ie:tab:volterra` | u = 1 + 2 int_0^x t u dt (exact e^{x^2}): A-PINN with hard constraints vs direct quadrature (8 mapped Gauss nodes); trapezoidal rule comparison; Gronwall bound | 5.1 min |
| `helmholtz_filon.py` | Sec. "Oscillatory kernels"; Fig. `ie:fig:helmholtz`, Table `ie:tab:helmholtz` | u = 1 + 6 (i/20) int_0^1 e^{10 i abs(x-t)} u dt with closed-form solution: SIREN PINN with Filon panel quadrature (20 panels); Nystrom with global Gauss-Legendre, Simpson and Filon panels; a posteriori bound | 2.8 min |
| `filon_weights.py` | Sec. "Filon-type panel quadrature"; Fig. `ie:fig:filon`, Table `ie:tab:filon` | accuracy of the closed-form, Taylor-switched and series-switched Filon weights against composite Gauss quadrature; 17-node Gauss, Simpson and Filon rules on a model oscillatory integral | 3 s |
| `_tables.py` | Tables `ie:tab:fredholm`, `ie:tab:volterra`, `ie:tab:helmholtz`, `ie:tab:filon`, `ie:tab:aposteriori` | writes the booktabs table bodies from the `results.json` files (default target `../book/figures/ch07`) | seconds |
| `_iecore.py` | (module) | Filon moments and weights (stable series for small arguments), Filon panel matrix with partial panels, closed-form solution of the Helmholtz-type model problem, Filon-Nystrom solver, robust timer, helpers for `--replot` | - |

Runtimes are the `wall_time_total_s` of the full runs used for the book, measured on a heavily
loaded shared 2-core machine (one thread per script); on an idle core they are several times
shorter. The tables in the book report the times stored in the results files, and only their
orders of magnitude matter.

The assignment notebook for this chapter is `notebooks/ch07_integral_equations_assignment.ipynb`.
