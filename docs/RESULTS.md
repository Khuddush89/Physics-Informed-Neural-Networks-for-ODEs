# Examples and runtimes

Wall-clock time of each script (full run, one CPU thread, double precision).

| chapter | script | example | runtime (s) | seed |
|---|---|---|---|---|
| ch01_formulation | `ad_exactness_cost` | AD exactness versus finite differences; cost of input derivatives | 6.3 | 1234 |
| ch01_formulation | `first_pinn` | First PINN for y' = -y + sin t: soft vs hard initial condition | 43 | 1234 |
| ch02_theory | `activations` | Activation functions and their first two derivatives | 0.7 | 1234 |
| ch02_theory | `aposteriori_bounds` | A-posteriori residual-based error bounds versus true PINN errors | 83 | 1234 |
| ch03_architectures | `architecture_comparison` | Airy BVP: MLP vs ChNN vs LeNN vs Fourier features vs SIREN (hard Dirichlet) | 222 | 1234 |
| ch03_architectures | `feature_conditioning` | Chebyshev/Legendre feature maps: derivative check and conditioning | 0.4 | 1234 |
| ch03_architectures | `soft_vs_hard` | Airy oscillator: soft vs hard conditions for an IVP and a Robin BVP | 235 | 1234 |
| ch03_architectures | `soft_vs_hard_chnn` | Airy oscillator: soft vs hard conditions for an IVP and a Robin BVP | 298 | 1234 |
| ch03_architectures | `well_posedness` | Airy oscillator: unique solvability and conditioning | < 0.1 | 1234 |
| ch04_ivp | `logistic_activation_study` | Logistic ODE: activation study, soft vs hard initial condition | 146 | 1234 |
| ch04_ivp | `lotka_volterra` | Lotka-Volterra system, PINN in log variables with hard ICs | 73 | 1234 |
| ch04_ivp | `lotka_volterra_T15` | Lotka-Volterra system, PINN in log variables with hard ICs | 83 | 1234 |
| ch04_ivp | `manual_polynomial_networks` | Hand calculation for Chebyshev, Legendre and MLP networks (y' = y) | < 0.1 | 0 |
| ch04_ivp | `manual_single_point` | Hand calculation of a PINN loss, gradient and Adam step (y'=y) | 0.4 | 0 |
| ch04_ivp | `third_order_system` | Coupled third-order system: stability analysis and PINN | 64 | 1234 |
| ch04_ivp | `vanderpol` | van der Pol oscillator (mu=0.5): bounded/power hard ICs vs soft ICs | 135 | 1234 |
| ch05_bvp | `boundary_layer` | Boundary-layer BVP eps y'' + y' = 1: point sets and a layer-enriched network | 100 | 1234 |
| ch05_bvp | `duffing_classical` | Duffing-type BVP: collocation, shooting and finite differences (accuracy, time) | 3.7 | 1234 |
| ch05_bvp | `duffing_pinn` | Duffing-type BVP with mixed conditions: ChNN/MLP PINNs, soft/hard BCs | 109 | 1234 |
| ch05_bvp | `duffing_reference` | Duffing-type BVP: closed-form solutions, multiplicity, verification | 0.8 | 1234 |
| ch05_bvp | `manual_bvp` | Hand calculation for a Dirichlet BVP: residuals, collocation system, finite differences | < 0.1 | 0 |
| ch05_bvp | `resonance_loss_floor` | BVP without solution (y''+y=0, y(0)=0, y(pi)=1): loss floor of PINNs | 24 | 1234 |
| ch05_bvp | `third_order_system_reference` | Coupled third-order BVP (exercise): solve_bvp reference | 1.3 | 1234 |
| ch06_tvp | `adjoint_sensitivity` | Adjoint sensitivity of a logistic misfit with one state/adjoint PINN | 22 | 1234 |
| ch06_tvp | `growth_model` | Finite-horizon optimal growth with terminal capital (PINN vs solve_bvp) | 25 | 1234 |
| ch06_tvp | `jerk_tvp` | Third-order TVP: LeNN/MLP with soft or hard terminal conditions | 163 | 1234 |
| ch06_tvp | `lqr_forward_backward` | LQR forward-backward Hamiltonian system solved by one PINN | 39 | 1234 |
| ch06_tvp | `lqr_riccati` | LQR Riccati terminal value problem (double integrator) by a PINN | 25 | 1234 |
| ch06_tvp | `manual_tvp` | Hand calculation for a second-order TVP and its time reversal | < 0.1 | 0 |
| ch06_tvp | `pendulum_tvp` | Damped pendulum TVP: LeNN with soft and hard terminal conditions | 109 | 1234 |
| ch06_tvp | `system_tvp` | Two coupled third-order TVPs: backward reference and hard-constrained PINN | 222 | 1234 |
| ch06_tvp | `tvp_conditioning` | Backward amplification for TVPs: classical solver and PINN | 36 | 1234 |
| ch07_integral_equations | `filon_weights` | Filon weights: stable evaluation and accuracy on a model integral | 0.5 | 1234 |
| ch07_integral_equations | `fredholm_degenerate` | Fredholm IE with degenerate kernel x t: quadrature vs auxiliary scalar | 28 | 1234 |
| ch07_integral_equations | `helmholtz_filon` | 1D Helmholtz-type IE (k=10, lam=6): SIREN PINN with Filon panels | 30 | 1234 |
| ch07_integral_equations | `manual_nystrom` | Hand calculation: Nystrom with one and two Gauss nodes, PINN quadrature residual | < 0.1 | 0 |
| ch07_integral_equations | `volterra_apinn` | Volterra IE u = 1 + 2 int_0^x t u dt: A-PINN vs direct quadrature | 38 | 1234 |
| ch08_integro_differential | `convolution` | Convolution IDE: direct quadrature vs internal variable (hard ICs) | 62 | 1234 |
| ch08_integro_differential | `delay_reference` | Delay IDE of the exercise: method-of-steps reference | 0.3 | 0 |
| ch08_integro_differential | `fredholm` | Fredholm IDE: global quadrature vs moment variable (hard IC) | 33 | 1234 |
| ch08_integro_differential | `manual_ide` | Hand calculation: trapezoidal methods and PINN residual for a convolution IDE | < 0.1 | 0 |
| ch08_integro_differential | `volterra_growth` | Volterra IDE on [0,5]: rapid growth, scaling of the unknowns | 148 | 1234 |
| ch08_integro_differential | `volterra_short` | Volterra IDE on [0,2]: direct quadrature vs A-PINN (hard ICs) | 68 | 1234 |
| ch09_training_diagnostics | `causality` | Harmonic oscillator on long intervals: vanilla vs causal weights vs time-marching | 467 | 1234 |
| ch09_training_diagnostics | `diagnostics` | Validation residual vs training loss; residual-based adaptive refinement | 51 | 1234 |
| ch09_training_diagnostics | `lorenz_chaos` | Lorenz system: Lyapunov exponent, twin experiment, global and time-marching PINNs | 350 | 1234 |
| ch09_training_diagnostics | `loss_imbalance` | BVP y''=f with large omega: fixed vs rescaled vs annealed vs self-adaptive vs hard BC | 130 | 1234 |
| ch09_training_diagnostics | `optimizers` | Adam vs Adam->L-BFGS vs L-BFGS, float64 vs float32 (nonlinear pendulum) | 114 | 1234 |
| ch09_training_diagnostics | `spectral_bias` | Spectral bias: tanh MLP vs Fourier features vs SIREN on y'=f | 142 | 1234 |
| ch09_training_diagnostics | `stiffness` | Stiff linear IVP y'=-lambda(y-cos t): PINN variants vs Radau | 178 | 1234 |
| ch09_training_diagnostics | `trivial_solutions` | Logistic ODE: attraction of the equilibrium y=0 and remedies | 231 | 1234 |
| ch10_benchmarking | `bvp_bratu` | Bratu BVP: collocation, shooting, finite differences vs PINN | 125 | 1234 |
| ch10_benchmarking | `integral_equations` | Fredholm, Volterra and IDE: quadrature methods vs PINNs | 70 | 1234 |
| ch10_benchmarking | `ivp_logistic` | Logistic ODE: classical solvers vs PINN (work-precision) | 69 | 1234 |
| ch10_benchmarking | `ivp_stiff` | Stiff linear system (ratio 1000): classical solvers vs PINN | 111 | 1234 |
| ch10_benchmarking | `ivp_vanderpol` | Van der Pol (mu=0.5): classical solvers vs PINN (work-precision) | 181 | 1234 |
| ch10_benchmarking | `parametric_logistic` | Parametric logistic family: one PINN vs repeated classical solves | 136 | 1234 |
| ch10_benchmarking | `residual_certificate` | A-posteriori residual bounds for a PINN and an RK45 interpolant | 19 | 1234 |
| ch11_inverse_problems | `logistic_parameter_id` | Logistic ODE: identify (r, K) from noisy data, PINN vs least squares | 106 | 1234 |
| ch11_inverse_problems | `logistic_weight_study` | Logistic inverse PINN: effect of the residual weight lam_r | 45 | 1234 |
| ch11_inverse_problems | `lotka_volterra_partial` | Lotka-Volterra observed through the prey only: parameters and predator | 223 | 1234 |
| ch11_inverse_problems | `manual_inverse` | Hand calculation: Gauss-Newton versus the inverse-PINN loss and gradient | < 0.1 | 0 |
| ch11_inverse_problems | `source_identification` | Unknown source f(t) in y' = -k y + f(t) from noisy data (two networks) | 109 | 1234 |
| ch11_inverse_problems | `uq_ensembles` | Logistic inverse problem: ensembles vs Gauss-Newton vs Metropolis | 101 | 1234 |
| ch11_inverse_problems | `volterra_parameter_id` | Volterra equation u = 1 + lam int_0^x s u ds: identify lam from noisy data | 230 | 1234 |
| ch12_further_classes | `dae_nonlinear_resistor` | Index-1 DAE y' = -z, 0 = z + z^3 - y: hard (Newton layer) vs penalty | 59 | 1234 |
| ch12_further_classes | `delay_method_of_steps` | DDE y'(t) = -y(t-1): global PINN versus method of steps | 62 | 1234 |
| ch12_further_classes | `fractional_relaxation` | Caputo relaxation D^alpha y = -y: fPINN with the L1 scheme | 75 | 1234 |
| ch12_further_classes | `lane_emden` | Lane-Emden y'' + (2/x) y' + y^m = 0: trial solution 1 + (x/L)^2 N(x) | 118 | 1234 |
| ch12_further_classes | `stiff_robertson` | Robertson kinetics on [0, 40]: log-time PINN vs vanilla PINN vs Radau/BDF | 74 | 1234 |
| ch12_further_classes | `sturm_liouville` | Sturm-Liouville eigenpairs: Rayleigh-quotient selection + residual with trainable eigenvalue and deflation | 166 | 1234 |
| ch12_further_classes | `time_scales` | y^Delta = p y on R, Z and P_{1,1}: one PINN for continuous, discrete and hybrid time | 71 | 1234 |
