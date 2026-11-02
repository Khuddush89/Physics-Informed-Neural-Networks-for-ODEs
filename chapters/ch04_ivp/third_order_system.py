"""Coupled third-order system (Chapter 4, exercise ivp:ex:third-order; solution in Appendix C).

    y1''' = -k1 y1 - c1 y1^2 y2' - d1 y1'',
    y2''' = -k2 y2 + c2 (y1')^2 - d2 y2',          t in [0, 4],
    y1(0) = 0.5, y1'(0) = 0, y1''(0) = 0,   y2(0) = 0, y2'(0) = 1, y2''(0) = 0,
    (k1, k2, c1, c2, d1, d2) = (1, 0.8, 0.3, 0.4, 0.2, 0.15).

The script
* computes the roots of the characteristic polynomials of the linear parts
  (lambda^3 + d1 lambda^2 + k1 and lambda^3 + d2 lambda + k2): both have a complex pair with
  positive real part, so the linearized dynamics are oscillatory and growing;
* locates the time at which the reference solution exceeds 1e6 in magnitude (blow-up);
* trains a PINN with the trial solution  y = y(0) + y'(0) t + y''(0) t^2/2 + phi(t) N_theta(t),
  phi(t) = t^3 / (1 + t^3)  (bounded; phi and its first two derivatives vanish at t = 0, so all
  six initial conditions hold exactly; the unbounded factor t^3 reaches 64 at t = 4 and makes
  the initial loss of order 1e5) and compares it with DOP853 (rtol = atol = 1e-12) on 1001
  uniform points.

Derivatives with respect to t. Nested reverse-mode differentiation needs one backward pass per
output and per order (six passes here) and is slow for third derivatives. Because the input t is
a scalar, the script instead propagates the values and the first three t-derivatives of every
layer forward through the tanh network (Faa di Bruno's formula, "Taylor-mode" differentiation)
and then applies the Leibniz rule to the trial solution. A check at start-up compares the
result with pinnode.derivatives (nested automatic differentiation) and stops if they differ.

Run:  python chapters/ch04_ivp/third_order_system.py           (about 1-2 min of CPU time on one core)
      python chapters/ch04_ivp/third_order_system.py --quick   (smoke test)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import (ConstrainedModel, Example, MLP, count_parameters, derivatives, linf, rel_l2,
                     to_tensor, train, uniform)
from pinnode.reference import ivp_reference

K1, K2, C1, C2, D1, D2 = 1.0, 0.8, 0.3, 0.4, 0.2, 0.15
Y0 = np.array([0.5, 0.0])      # y1(0), y2(0)
V0 = np.array([0.0, 1.0])      # y1'(0), y2'(0)
A0 = np.array([0.0, 0.0])      # y1''(0), y2''(0)
T0, T = 0.0, 4.0
WIDTH, DEPTH, N_COLLOC, N_TEST = 32, 3, 200, 1001


def rhs(t, z):
    y1, y1p, y1pp, y2, y2p, y2pp = z
    return [y1p, y1pp, -K1 * y1 - C1 * y1 ** 2 * y2p - D1 * y1pp,
            y2p, y2pp, -K2 * y2 + C2 * y1p ** 2 - D2 * y2p]


def mlp_taylor3(net: MLP, t: torch.Tensor):
    """Values and t-derivatives up to order 3 of a tanh MLP with scalar input (forward mode).

    A linear layer maps the derivatives linearly. For h = tanh(z), with g1 = 1 - h^2,
    g2 = -2 h g1 and g3 = g1 (6 h^2 - 2), Faa di Bruno's formula gives
    h^(1) = g1 z^(1),  h^(2) = g2 (z^(1))^2 + g1 z^(2),
    h^(3) = g3 (z^(1))^3 + 3 g2 z^(1) z^(2) + g1 z^(3).
    """
    lo, hi = net.scale.lo, net.scale.hi
    x = net.scale(t)
    d1 = torch.ones_like(x) * (2.0 / (hi - lo))
    d2 = torch.zeros_like(x)
    d3 = torch.zeros_like(x)
    for layer in net.body:
        if isinstance(layer, torch.nn.Linear):
            x = layer(x)
            d1, d2, d3 = d1 @ layer.weight.T, d2 @ layer.weight.T, d3 @ layer.weight.T
        elif isinstance(layer, torch.nn.Tanh):
            h = torch.tanh(x)
            g1 = 1.0 - h ** 2
            g2 = -2.0 * h * g1
            g3 = g1 * (6.0 * h ** 2 - 2.0)
            x, d1, d2, d3 = (h, g1 * d1, g2 * d1 ** 2 + g1 * d2,
                             g3 * d1 ** 3 + 3.0 * g2 * d1 * d2 + g1 * d3)
        else:
            raise TypeError("only Linear and Tanh layers are supported")
    return x, d1, d2, d3


def phi_derivatives(t):
    """phi = t^3/(1+t^3) and its first three derivatives (phi, phi', phi'' vanish at t = 0)."""
    q = 1.0 + t ** 3
    phi = t ** 3 / q
    d1 = 3 * t ** 2 / q ** 2
    d2 = 6 * t / q ** 2 - 18 * t ** 4 / q ** 3
    d3 = 6 / q ** 2 - 108 * t ** 3 / q ** 3 + 162 * t ** 6 / q ** 4
    return phi, d1, d2, d3


def trial_taylor3(net, t, y0_t, v0_t, a0_t):
    """y = y0 + v0 t + a0 t^2/2 + phi(t) N(t) and its first three t-derivatives (Leibniz rule)."""
    n, n1, n2, n3 = mlp_taylor3(net, t)
    p0, p1, p2, p3 = phi_derivatives(t)
    y = y0_t + v0_t * t + 0.5 * a0_t * t ** 2 + p0 * n
    dy = v0_t + a0_t * t + p1 * n + p0 * n1
    d2y = a0_t + p2 * n + 2 * p1 * n1 + p0 * n2
    d3y = p3 * n + 3 * p2 * n1 + 3 * p1 * n2 + p0 * n3
    return y, dy, d2y, d3y


def main():
    ex = Example("ch04_ivp", "third_order_system",
                 description="Coupled third-order system: stability analysis and PINN", default_seed=1234)
    # ------------------------------------------------------------- linear stability and blow-up
    roots1 = np.roots([1.0, D1, 0.0, K1])
    roots2 = np.roots([1.0, 0.0, D2, K2])
    z0 = [Y0[0], V0[0], A0[0], Y0[1], V0[1], A0[1]]

    def big(t, z):
        return np.max(np.abs(z)) - 1e6
    big.terminal = True
    sol = solve_ivp(rhs, (T0, 20.0), z0, method="DOP853", rtol=1e-12, atol=1e-12, events=big)
    t_blow = float(sol.t_events[0][0]) if len(sol.t_events[0]) else float("nan")
    sol_fail = solve_ivp(rhs, (T0, 20.0), z0, method="DOP853", rtol=1e-12, atol=1e-12)
    t_fail = float(sol_fail.t[-1])            # last time reached before the integrator gives up
    ex.log(f"roots y1: {roots1}, roots y2: {roots2}; |z| > 1e6 at t = {t_blow:.4f}")

    # ------------------------------------------------------------- reference on [0, T]
    t_test = np.linspace(T0, T, N_TEST)
    ref = ivp_reference(rhs, (T0, T), z0, t_test)[:, [0, 3]]

    # ------------------------------------------------------------- PINN
    y0_t, v0_t, a0_t = (torch.tensor(v).reshape(1, 2) for v in (Y0, V0, A0))
    torch.manual_seed(ex.seed)
    net = MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh", domain=(T0, T))
    model = ConstrainedModel(net, lambda t, n: y0_t + v0_t * t + 0.5 * a0_t * t ** 2 + phi_derivatives(t)[0] * n)
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 30), T0, T), requires_grad=True)
    t_test_t = to_tensor(t_test)

    # check the forward-mode derivatives against nested reverse-mode automatic differentiation
    t_chk = to_tensor(np.linspace(T0, T, 7), requires_grad=True)
    fwd = trial_taylor3(net, t_chk, y0_t, v0_t, a0_t)
    rev = derivatives(model(t_chk), t_chk, 3)
    chk = max(float((a - b).abs().max()) for a, b in zip(fwd, rev))
    ex.log(f"max |forward-mode - reverse-mode| derivatives = {chk:.1e}")
    assert chk < 1e-10, "forward derivative propagation disagrees with autograd"

    # cost of one loss + gradient evaluation with both ways of computing the t-derivatives
    def timed(method, reps):
        best = float("inf")
        for _ in range(reps):
            t_start = time.perf_counter()
            if method == "forward":
                y, dy, d2y, d3y = trial_taylor3(net, t_col, y0_t, v0_t, a0_t)
            else:
                y, dy, d2y, d3y = derivatives(model(t_col), t_col, 3)
            (d3y.square().mean() + y.square().mean()).backward()
            best = min(best, time.perf_counter() - t_start)
        net.zero_grad(set_to_none=True)
        return best
    reps = ex.pick(10, 2)
    cost_rev, cost_fwd = timed("reverse", reps), timed("forward", reps)
    ex.log(f"loss+gradient evaluation: nested reverse mode {cost_rev * 1e3:.1f} ms, "
           f"forward propagation {cost_fwd * 1e3:.1f} ms (ratio {cost_rev / cost_fwd:.1f})")

    def residuals(t):
        y, dy, d2y, d3y = trial_taylor3(net, t, y0_t, v0_t, a0_t)
        r1 = d3y[:, :1] + K1 * y[:, :1] + C1 * y[:, :1] ** 2 * dy[:, 1:2] + D1 * d2y[:, :1]
        r2 = d3y[:, 1:2] + K2 * y[:, 1:2] - C2 * dy[:, :1] ** 2 + D2 * dy[:, 1:2]
        return r1, r2

    def loss_fn():
        r1, r2 = residuals(t_col)
        L1, L2 = (r1 ** 2).mean(), (r2 ** 2).mean()
        return L1 + L2, {"r1": L1, "r2": L2}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), ref)}

    n_adam, n_lbfgs = ex.pick(8000, 100), ex.pick(1000, 10)
    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=100, print_every=2000)
    ex.save_history(hist)
    with torch.no_grad():
        y = model(t_test_t).numpy()
    t_req = to_tensor(t_test, requires_grad=True)
    r1, r2 = residuals(t_req)
    res = np.hstack([r1.detach().numpy(), r2.detach().numpy()])
    metrics = {"rel_l2": rel_l2(y, ref), "linf": linf(y, ref),
               "rel_l2_y1": rel_l2(y[:, 0], ref[:, 0]), "rel_l2_y2": rel_l2(y[:, 1], ref[:, 1]),
               "linf_y1": linf(y[:, 0], ref[:, 0]), "linf_y2": linf(y[:, 1], ref[:, 1]),
               "max_abs_residual": float(np.max(np.abs(res))), "final_loss": hist.final_loss,
               "iterations": hist.iters[-1], "train_time_s": hist.train_time,
               "n_params": count_parameters(model), "derivative_check_max_diff": chk,
               "derivative_cost_reverse_s": cost_rev, "derivative_cost_forward_s": cost_fwd,
               "derivative_cost_ratio": cost_rev / cost_fwd}
    for k, v in metrics.items():
        ex.log(f"{k} = {v:.3e}" if isinstance(v, float) else f"{k} = {v}")
    ex.save_table({"t": t_test, "y1_ref": ref[:, 0], "y2_ref": ref[:, 1], "y1_pinn": y[:, 0],
                   "y2_pinn": y[:, 1]}, "predictions")
    ex.save_results({
        "problem": {"k1": K1, "k2": K2, "c1": C1, "c2": C2, "d1": D1, "d2": D2,
                    "y0": Y0, "dy0": V0, "d2y0": A0, "interval": [T0, T],
                    "reference": "DOP853, rtol=atol=1e-12"},
        "stability": {"roots_y1": [[r.real, r.imag] for r in roots1],
                      "roots_y2": [[r.real, r.imag] for r in roots2],
                      "max_real_part": float(max(roots1.real.max(), roots2.real.max())),
                      "t_exceeds_1e6": t_blow, "t_integration_stops": t_fail,
                      "integration_message": sol_fail.message},
        "reference_values": {"y1(T)": float(ref[-1, 0]), "y2(T)": float(ref[-1, 1]),
                             "max_abs_y1": float(np.max(np.abs(ref[:, 0]))),
                             "max_abs_y2": float(np.max(np.abs(ref[:, 1])))},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "trial": "y0 + v0 t + a0 t^2/2 + t^3/(1+t^3) N(t)",
                            "collocation": ex.pick(N_COLLOC, 30), "adam_iters": n_adam,
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters_max": n_lbfgs,
                            "test_grid": N_TEST},
        **metrics,
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P

    fig, axes = P.figure(1, 2, aspect=0.78)
    ax = axes[0]
    P.plot_reference(ax, t_test, ref[:, 0], label="reference")
    P.plot_reference(ax, t_test, ref[:, 1], label=None)
    P.plot_prediction(ax, t_test, y[:, 0], label="$y_1$ (PINN)", k=0, markevery=(0, 50))
    P.plot_prediction(ax, t_test, y[:, 1], label="$y_2$ (PINN)", k=1, markevery=(25, 50))
    ax.set_xlabel("$t$")
    ax.set_ylabel("solution")
    ax.legend(loc="lower left")
    ax = axes[1]
    P.plot_error(ax, t_test, y[:, 0] - ref[:, 0], label="$|y_{1,\\theta}-y_1|$", k=0, ls=P.LINESTYLES[0])
    P.plot_error(ax, t_test, y[:, 1] - ref[:, 1], label="$|y_{2,\\theta}-y_2|$", k=1, ls=P.LINESTYLES[1])
    ax.set_ylim(1e-10, 1e-1)
    P.label_panels(axes)
    ex.savefig(fig, "third_order_system")


if __name__ == "__main__":
    main()
