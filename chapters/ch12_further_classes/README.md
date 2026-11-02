# Chapter 12 - Further equation classes

Scripts for the worked examples of Chapter 12 (stiff, singular, delay, fractional and
differential-algebraic equations, Sturm-Liouville eigenvalue problems, dynamic equations on
time scales). Every script supports `--quick` (smoke test, seconds) and writes
`outputs/ch12_further_classes/<example>/results.json`, CSV histories/tables and the figure
(PDF + PNG). Run from the repository root:

```bash
PYTHONPATH=. python chapters/ch12_further_classes/time_scales.py            # full run
PYTHONPATH=. python chapters/ch12_further_classes/time_scales.py --quick    # smoke test
python chapters/ch12_further_classes/_make_tables.py                        # LaTeX tables -> ../book/figures/ch12/
```

| script | book section / figure / table | what it does | runtime (full run, 1 core) |
|---|---|---|---|
| `stiff_robertson.py` | Sec. 12.1, Fig. `ext:fig:robertson`, Tab. `ext:tab:robertson` | Robertson kinetics on [0, 40]: PINN in logarithmic time with scaled variables and normalized residuals vs. PINN in physical time vs. Radau/BDF (four tolerances); Jacobian eigenvalues along the solution | 2 min |
| `lane_emden.py` | Sec. 12.2, Fig. `ext:fig:lane-emden`, Tab. `ext:tab:lane-emden` | Lane-Emden equation for m = 0, 1, 3, 5 on [0, 7] with the residual x y'' + 2y' + x y^m and the trial solution 1 + (x/L)^2 N(x); first zero and mass parameter; initial loss of the unscaled trial 1 + x^2 N(x) for comparison; series + DOP853 reference for m = 3 | 3 min |
| `delay_method_of_steps.py` | Sec. 12.3, Fig. `ext:fig:dde` | y'(t) = -y(t-1), history 1: one global network vs. the method of steps (one network per delay interval, warm-started from the previous one); second derivative at the breaking point t = 1; error per interval | 63 s |
| `fractional_relaxation.py` | Sec. 12.4, Fig. `ext:fig:fractional` | Caputo relaxation D^alpha y = -y (alpha = 0.5, 0.8): fPINN with the L1 scheme on uniform and graded grids (N = 16 ... 256) vs. the L1 finite-difference solution; Mittag-Leffler reference (mpmath series, verified) | 5 min |
| `dae_nonlinear_resistor.py` | Sec. 12.5, Fig. `ext:fig:dae` | index-1 DAE y' = -z, 0 = z + z^3 - y (Lambert-W solution): algebraic variable by a differentiable Newton layer (hard) vs. penalty with w_a = 1, 100 | 86 s |
| `sturm_liouville.py` | Sec. 12.6, Fig. `ext:fig:eigen`, Tab. `ext:tab:eigen`, Alg. `ext:alg:eigen` | first three eigenpairs of -y'' = lambda y and of a variable-coefficient problem: Rayleigh-quotient selection, then trainable eigenvalue with normalization and orthogonality (deflation) penalties and scale-invariant residual; Chebyshev collocation reference | 8 min |
| `time_scales.py` | Sec. 12.7, Fig. `ext:fig:timescales`, Alg. `ext:alg:timescale` | y^Delta = p y on R, Z and the hybrid time scale P_{1,1}: automatic differentiation at right-dense points, forward differences at right-scattered points; single output vs. one head per interval | 4 min |
| `_make_tables.py` | Tables `ext:tab:robertson`, `ext:tab:lane-emden`, `ext:tab:eigen`, `ext:tab:summary` | reads the `results.json` files and writes `book/figures/ch12/tab_*.tex` (not run by the smoke tests) | seconds |

Runtimes are wall-clock totals (`wall_time_total_s` in `results.json`) measured on a shared
two-core machine under load; on an idle core they are shorter. The tables
of the book contain times, so re-run the scripts one at a time and then `_make_tables.py` to
refresh them on your machine.

The Mittag-Leffler reference of `fractional_relaxation.py` uses `mpmath`, which is installed
with PyTorch (through SymPy).

Assignment notebook: `notebooks/ch12_time_scales_assignment.ipynb` (PINNs on time scales).
