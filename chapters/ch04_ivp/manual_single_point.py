"""A PINN computed by hand: one collocation point, one Adam step (all numbers of the section).

Book: Chapter 4 (Initial value problems), Section "A PINN computed by hand".

Problem   y' = y,  y(0) = 1  (exact solution e^t); a single collocation point t_1 = 0.5.
Network   y_theta(t) = v_1 tanh(w_1 t + b_1) + v_2 tanh(w_2 t + b_2) + c
          (one hidden layer with two tanh neurons, linear output; 7 parameters)
Values    w = (0.1, 0.2), b = (0.1, 0.2), v = (0.5, 0.6), c = 0.1
Loss      L = (y_theta'(t_1) - y_theta(t_1))^2 + (y_theta(0) - 1)^2   (lambda_r = lambda_ic = 1)

The script evaluates every quantity of the hand calculation twice -- with the closed-form
formulas of the book (NumPy) and with PyTorch (automatic differentiation for y', backward
pass for the gradient, torch.optim.Adam for the update) -- prints both and checks that they
agree to rounding error. All values quoted in the book are taken from results.json.

Run:  python chapters/ch04_ivp/manual_single_point.py      (well under 1 s; --quick is identical)
"""
from __future__ import annotations

import numpy as np
import torch

from pinnode import Example

# parameters of the hand calculation
W = np.array([0.1, 0.2])
B = np.array([0.1, 0.2])
V = np.array([0.5, 0.6])
C = 0.1
T1, T0, Y0 = 0.5, 0.0, 1.0
ADAM = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
NAMES = ["w1", "w2", "b1", "b2", "v1", "v2", "c"]


def hand_calculation():
    """Closed-form forward pass, derivative, residual, loss, gradient and Adam step."""
    out = {}
    # forward pass at the collocation point
    z = W * T1 + B
    h = np.tanh(z)
    y = float(V @ h + C)
    s = 1.0 - h ** 2                       # tanh'(z) = 1 - tanh(z)^2
    dh_dt = W * s                          # d h_k / dt
    yp = float(V @ dh_dt)                  # y_theta'(t_1)
    r = yp - y                             # residual of y' - y = 0
    # forward pass at the initial time
    z0 = W * T0 + B
    h0 = np.tanh(z0)
    y_at0 = float(V @ h0 + C)
    s0 = 1.0 - h0 ** 2
    L_r = r ** 2
    L_ic = (y_at0 - Y0) ** 2
    L = L_r + L_ic
    out.update(z=z, h=h, one_minus_h2=s, dh_dt=dh_dt, y=y, yp=yp, residual=r,
               z0=z0, h0=h0, y0_pred=y_at0, L_r=L_r, L_ic=L_ic, L=L)

    # derivatives of y and y' with respect to the 7 parameters (order: w1 w2 b1 b2 v1 v2 c)
    dy = np.concatenate([V * s * T1, V * s, h, [1.0]])
    dyp = np.concatenate([V * s * (1.0 - 2.0 * W * h * T1),     # d y'/d w_k
                          -2.0 * V * W * h * s,                 # d y'/d b_k
                          W * s,                                # d y'/d v_k
                          [0.0]])                               # d y'/d c
    dr = dyp - dy
    dy0 = np.concatenate([V * s0 * T0, V * s0, h0, [1.0]])
    g_r = 2.0 * r * dr
    g_ic = 2.0 * (y_at0 - Y0) * dy0
    g = g_r + g_ic
    out.update(dr=dr, dy0=dy0, grad_residual=g_r, grad_ic=g_ic, grad=g)

    # one Adam step from m_0 = v_0 = 0
    m1 = (1.0 - ADAM["beta1"]) * g
    v1 = (1.0 - ADAM["beta2"]) * g ** 2
    m_hat = m1 / (1.0 - ADAM["beta1"])
    v_hat = v1 / (1.0 - ADAM["beta2"])
    step = -ADAM["lr"] * m_hat / (np.sqrt(v_hat) + ADAM["eps"])
    theta = np.concatenate([W, B, V, [C]])
    out.update(m1=m1, v1=v1, m_hat=m_hat, v_hat=v_hat, step=step, theta=theta,
               theta_new=theta + step)

    # loss after the step (shows that one Adam step of size ~lr changes little)
    w_n, b_n, v_n, c_n = (theta + step)[0:2], (theta + step)[2:4], (theta + step)[4:6], (theta + step)[6]
    h_n = np.tanh(w_n * T1 + b_n)
    r_n = float(v_n @ (w_n * (1 - h_n ** 2))) - float(v_n @ h_n + c_n)
    y0_n = float(v_n @ np.tanh(b_n) + c_n)
    out["L_after_step"] = r_n ** 2 + (y0_n - Y0) ** 2
    return out


def torch_calculation():
    """The same quantities with PyTorch autograd and torch.optim.Adam."""
    torch.set_default_dtype(torch.float64)
    net = torch.nn.Sequential(torch.nn.Linear(1, 2), torch.nn.Tanh(), torch.nn.Linear(2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor(W).reshape(2, 1))
        net[0].bias.copy_(torch.tensor(B))
        net[2].weight.copy_(torch.tensor(V).reshape(1, 2))
        net[2].bias.copy_(torch.tensor([C]))
    t1 = torch.tensor([[T1]], requires_grad=True)
    t0 = torch.tensor([[T0]])
    y = net(t1)
    yp, = torch.autograd.grad(y, t1, torch.ones_like(y), create_graph=True)
    r = yp - y
    y_at0 = net(t0)
    L_r = (r ** 2).sum()
    L_ic = ((y_at0 - Y0) ** 2).sum()
    L = L_r + L_ic
    vals = {"y": y.item(), "yp": yp.item(), "residual": r.item(), "y0_pred": y_at0.item(),
            "L_r": L_r.item(), "L_ic": L_ic.item(), "L": L.item()}
    opt = torch.optim.Adam(net.parameters(), lr=ADAM["lr"], betas=(ADAM["beta1"], ADAM["beta2"]),
                           eps=ADAM["eps"])
    opt.zero_grad()
    L.backward()
    p = [net[0].weight.grad.ravel(), net[0].bias.grad, net[2].weight.grad.ravel(), net[2].bias.grad]
    grad = torch.cat(p).detach().numpy()
    opt.step()
    theta_new = torch.cat([net[0].weight.ravel(), net[0].bias, net[2].weight.ravel(),
                           net[2].bias]).detach().numpy()
    return {**vals, "grad": grad, "theta_new": theta_new}


def main():
    ex = Example("ch04_ivp", "manual_single_point",
                 description="Hand calculation of a PINN loss, gradient and Adam step (y'=y)",
                 default_seed=0)
    hand = hand_calculation()
    ad = torch_calculation()

    def p(label, value):
        print(f"  {label:<34s} {value}")

    print("\nForward pass at t_1 = 0.5")
    p("z = w t_1 + b", hand["z"])
    p("h = tanh(z)", hand["h"])
    p("y_theta(t_1)  [hand | torch]", f"{hand['y']:.10f} | {ad['y']:.10f}")
    print("Derivative by the chain rule / automatic differentiation")
    p("1 - h^2", hand["one_minus_h2"])
    p("dh/dt = w (1 - h^2)", hand["dh_dt"])
    p("y_theta'(t_1) [hand | torch]", f"{hand['yp']:.10f} | {ad['yp']:.10f}")
    print("Residual and loss")
    p("r(t_1) = y' - y", f"{hand['residual']:.10f} | {ad['residual']:.10f}")
    p("h(0) = tanh(b)", hand["h0"])
    p("y_theta(0)", f"{hand['y0_pred']:.10f} | {ad['y0_pred']:.10f}")
    p("L_r = r^2", f"{hand['L_r']:.10f} | {ad['L_r']:.10f}")
    p("L_ic = (y(0) - 1)^2", f"{hand['L_ic']:.10f} | {ad['L_ic']:.10f}")
    p("L = L_r + L_ic", f"{hand['L']:.10f} | {ad['L']:.10f}")
    print("Gradient (hand formula | backward pass) and one Adam step")
    print(f"  {'param':6s} {'theta':>8s} {'dr/dtheta':>13s} {'dL_r':>13s} {'dL_ic':>13s} "
          f"{'dL (hand)':>13s} {'dL (torch)':>13s} {'theta_new':>12s} {'torch':>12s}")
    for k, name in enumerate(NAMES):
        print(f"  {name:6s} {hand['theta'][k]:8.3f} {hand['dr'][k]:13.6e} {hand['grad_residual'][k]:13.6e} "
              f"{hand['grad_ic'][k]:13.6e} {hand['grad'][k]:13.6e} {ad['grad'][k]:13.6e} "
              f"{hand['theta_new'][k]:12.8f} {ad['theta_new'][k]:12.8f}")
    p("Adam: m_hat = g, v_hat = g^2  =>  step", "-lr * sign(g) * |g|/(|g|+eps)")
    p("loss after the step", f"{hand['L_after_step']:.10f}")

    diffs = {
        "forward": abs(hand["y"] - ad["y"]),
        "derivative": abs(hand["yp"] - ad["yp"]),
        "loss": abs(hand["L"] - ad["L"]),
        "gradient": float(np.max(np.abs(hand["grad"] - ad["grad"]))),
        "adam_update": float(np.max(np.abs(hand["theta_new"] - ad["theta_new"]))),
    }
    print("\nmax |hand - torch|:", {k: f"{v:.1e}" for k, v in diffs.items()})
    assert max(diffs.values()) < 1e-12, "hand calculation and autograd disagree"

    ex.save_table({"parameter": NAMES, "theta": hand["theta"], "dr_dtheta": hand["dr"],
                   "dLr_dtheta": hand["grad_residual"], "dLic_dtheta": hand["grad_ic"],
                   "dL_dtheta": hand["grad"], "dL_dtheta_torch": ad["grad"],
                   "theta_after_adam": hand["theta_new"], "theta_after_adam_torch": ad["theta_new"]},
                  "gradient_table")
    ex.save_results({
        "problem": {"ode": "y' = y", "y0": Y0, "t1": T1, "exact_y_t1": float(np.exp(T1))},
        "parameters": {"w": W, "b": B, "v": V, "c": C},
        "adam": ADAM,
        "hand": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in hand.items()},
        "torch": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in ad.items()},
        "max_abs_difference_hand_vs_torch": diffs,
    })


if __name__ == "__main__":
    main()
