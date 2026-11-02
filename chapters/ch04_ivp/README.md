# Chapter 4 - Initial value problems

Scripts for Chapter 4 of the book. Run them from the repository root, e.g.

```bash
PYTHONPATH=. python chapters/ch04_ivp/vanderpol.py            # full run
PYTHONPATH=. python chapters/ch04_ivp/vanderpol.py --quick    # smoke test (seconds)
```

Every script uses double precision, one CPU thread and a fixed seed, and writes its numbers to
`outputs/ch04_ivp/<example>/results.json` (plus CSV histories/predictions and the figures in
`figures/`). The scripts with several trainings checkpoint each training; after an interruption,
rerun with `--resume` to skip the trainings already completed. With `--only <name>` a single
training runs in its own (smaller) process, e.g.
`--only hard_sigmoid` for the logistic study or `--only hard` for the van der Pol example; a final
call with `--resume` then assembles the tables and figures from the checkpoints.

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `manual_single_point.py` | Sec. "A PINN computed by hand", Table `ivp:tab:manual` | forward pass, derivative, residual, loss, gradient and one Adam step for a 2-neuron network on `y' = y`, by closed-form formulas and by PyTorch; asserts agreement | < 1 s |
| `logistic_activation_study.py` | Sec. "First-order equations: the logistic equation", Fig. `ivp:fig:logistic`, Table `ivp:tab:logistic` | soft vs hard initial condition x four activations (tanh, sigmoid, ReLU, swish); minimum of the loss over affine functions (plateau); errors after Adam and after L-BFGS | about 3.5 min CPU (8 trainings) |
| `vanderpol.py` | Sec. "Higher-order equations: the van der Pol oscillator", Fig. `ivp:fig:vdp`, Table `ivp:tab:vdp` | van der Pol, mu = 0.5, on [0, 10]; trial solutions `1 + phi(t) N(t)` with `phi = t^2/(1+t^2)` and `phi = t^2`, and soft initial conditions; errors of y and y'; limit-cycle period | about 3 min CPU (3 trainings) |
| `lotka_volterra.py` | Sec. "Systems of first-order equations: the Lotka-Volterra model", Figs. `ivp:fig:lv`, `ivp:fig:lv-long`, Table `ivp:tab:lv` | PINN in logarithmic variables with hard initial conditions and an unweighted residual loss on [0, 6]; component errors and drift of the first integral. `--T 15 --outdir outputs/ch04_ivp/lotka_volterra_T15` reproduces the long-horizon failure; `--replot` redraws figures without training | about 1.5 min CPU per run |
| `third_order_system.py` | Exercise `ivp:ex:third-order`, solution in Appendix C (Table `ivp:tab:third-order`) | Routh-Hurwitz roots of the linear parts, blow-up time of the reference, PINN with the trial solution `y0 + v0 t + phi(t) N(t)`, `phi = t^3/(1+t^3)`; derivatives up to order 3 propagated forward through the network (checked against autograd) | about 1.5 min CPU |
| `write_tables.py` | Tables `ivp:tab:manual`, `ivp:tab:logistic`, `ivp:tab:vdp`, `ivp:tab:lv`, `ivp:tab:cost` (Sec. "How accurate and how fast?"), `ivp:tab:third-order` | reads the results files of the scripts above, times RK45 at matched accuracy and DOP853 at 1e-12, writes the LaTeX tables; run it last | about 10-30 s |

Regenerate all numbers, figures and tables of the chapter (sequentially, on an idle machine for
meaningful timings):

```bash
for s in manual_single_point logistic_activation_study vanderpol lotka_volterra third_order_system; do
    PYTHONPATH=. python chapters/ch04_ivp/$s.py
done
PYTHONPATH=. python chapters/ch04_ivp/lotka_volterra.py --T 15 --outdir outputs/ch04_ivp/lotka_volterra_T15
PYTHONPATH=. python chapters/ch04_ivp/write_tables.py --table-dir ../book/figures/ch04
cp outputs/ch04_ivp/logistic_activation_study/figures/logistic_activation_study.pdf \
   outputs/ch04_ivp/vanderpol/figures/vanderpol_summary.pdf \
   outputs/ch04_ivp/lotka_volterra/figures/lotka_volterra_summary.pdf ../book/figures/ch04/
cp outputs/ch04_ivp/lotka_volterra_T15/figures/lotka_volterra_phase.pdf \
   ../book/figures/ch04/lotka_volterra_long_phase.pdf
```

The runtimes above are CPU times measured on a shared machine; training times reported in
`results.json` are wall-clock times and depend on the load of the machine.

Reference solutions: closed form for `y' = y` and for the logistic equation; otherwise DOP853 with
`rtol = atol = 1e-12` (`pinnode.reference.ivp_reference`). Errors are always measured on 1001
uniformly spaced test points.
