# Chapter 8 — Integro-differential equations

Scripts for Chapter 8 of the book. Run every command from the repository root, e.g.

```bash
PYTHONPATH=. python chapters/ch08_integro_differential/volterra_short.py          # full run
PYTHONPATH=. python chapters/ch08_integro_differential/volterra_short.py --quick  # smoke test
```

Outputs (CSV, JSON, PDF/PNG figures) go to `outputs/ch08_integro_differential/<example>/`.
All PINNs use double precision, one CPU thread, the seed 1234, a tanh MLP with three hidden
layers of width 32 and initial conditions imposed exactly by trial functions. Errors are
measured on a uniform grid of 1001 points.

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `volterra_short.py` | Sec. "Quadrature residual versus auxiliary variable on [0,2]", Fig. `ide:fig:volterra`, Table `ide:tab:volterra` (top) | Volterra IDE u' = u + ∫₀ˣ t u dt on [0,2]: quadrature PINN (20 Gauss nodes per point) vs A-PINN (u, A); a posteriori residual and residual-to-error bound; trapezoidal method and DOP853 references | about 66 s CPU |
| `volterra_growth.py` | Sec. "Long intervals and rapidly growing solutions", Remark `ide:rem:bifactor`, Fig. `ide:fig:growth`, Table `ide:tab:volterra` (bottom) | Same IDE on [0,5] (u grows to 1.7e4): unscaled A-PINN, A-PINN after u = e^{x/2} v, A-PINN in logarithmic variables (w, a) = (log u, A/u), quadrature PINN in w, Bi-factorized PINN; classical references | about 3 min CPU |
| `fredholm.py` | Sec. "Fredholm integro-differential equations", Fig. `ide:fig:fredholm`, Table `ide:tab:fredholm` | u' = 1 + ∫₀¹ x t u dt: global-quadrature PINN vs trainable moment variable with consistency penalty; Nyström method for the integrated equation | about 40 s CPU |
| `convolution.py` | Sec. "Convolution kernels and internal variables", Fig. `ide:fig:convolution`, Table `ide:tab:convolution` | u' = u + ∫₀ˣ e^{-(x-t)} u dt on [0,1]: direct quadrature vs internal variable z; cost per loss/gradient evaluation; DOP853 and trapezoidal references | about 66 s CPU |
| `delay_reference.py` | Exercise `ide:ex:delay`, notebook `notebooks/ch08_ide_assignment.ipynb` | Method-of-steps reference (DOP853 on each delay interval, checked against a trapezoidal scheme) for the delay IDE of the exercise | a few seconds |
| `write_tables.py` | Tables `ide:tab:volterra`, `ide:tab:fredholm`, `ide:tab:convolution` | Writes the LaTeX tables from the `results.json` files; `--book ../book/figures/ch08` also copies tables and figure PDFs into the book | < 5 s |
| `_ide_common.py` | — | Shared helpers: exact solutions, trapezoidal direct-quadrature method for linear Volterra IDEs, DOP853 wrapper, CPU timing, resumable runs | — |

Runtimes are the CPU times of the training runs (one thread), measured on a shared machine;
the tables report CPU time for the same reason (use `write_tables.py --time wall` for
wall-clock times, which are also stored as `train_time_s` in every `results.json`).

To regenerate all numbers, tables and figures of the chapter:

```bash
for s in volterra_short volterra_growth fredholm convolution delay_reference; do
  PYTHONPATH=. python chapters/ch08_integro_differential/$s.py
done
PYTHONPATH=. python chapters/ch08_integro_differential/write_tables.py --book ../book/figures/ch08
```

`--resume` reuses variants already trained in an interrupted run (cached as
`cache_<variant>.pkl` in the output folder; delete these files after changing a script).
