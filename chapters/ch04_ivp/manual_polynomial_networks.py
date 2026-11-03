"""The hand calculation of Chapter 4 for polynomial feature networks (and the MLP).

Book: Chapter 4 (Initial value problems), Section "A PINN computed by hand" and Subsection
"The same calculation for Chebyshev and Legendre networks".

Problem   y' = y, y(0) = 1, one collocation point t_1 = 0.5, soft initial condition,
          L = (y'(t_1) - y(t_1))^2 + (y(0) - 1)^2.
Networks  'mlp'   : y = v1 tanh(w1 t + b1) + v2 tanh(w2 t + b2) + c          (7 parameters)
          'cheb1' : features (T_0, T_1) = (1, t)  (Chebyshev, first kind)    (9 parameters)
          'leg'   : features (P_0, P_1) = (1, t)  (Legendre; identical to 'cheb1' at degree one)
          'cheb2' : features (U_0, U_1) = (1, 2t) (Chebyshev, second kind)
          hidden layer z_k = w_k1 * phi_0 + w_k2 * phi_1 + b_k, no input rescaling (xi = t).
For every network the script evaluates the forward pass, the derivative in t, the residual, the
loss terms, all gradients (automatic differentiation) and one Adam step, and checks the manual
Adam update against torch.optim.Adam. All values quoted in the book are taken from results.json.

Run:  python chapters/ch04_ivp/manual_polynomial_networks.py   (well under 1 s; --quick is identical)
"""
from __future__ import annotations

import torch

from pinnode import Example

T1 = 0.5
ADAM = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}


def run(kind):
    if kind == "mlp":
        names = ["w1", "w2", "b1", "b2", "v1", "v2", "c"]
        init = [0.1, 0.2, 0.1, 0.2, 0.5, 0.6, 0.1]
    else:
        names = ["w11", "w12", "w21", "w22", "b1", "b2", "v1", "v2", "c"]
        init = [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.5, 0.6, 0.1]
    P = [torch.tensor(x, requires_grad=True) for x in init]

    def net(t):
        if kind == "mlp":
            w1, w2, b1, b2, v1, v2, c = P
            z1, z2 = w1 * t + b1, w2 * t + b2
        else:
            w11, w12, w21, w22, b1, b2, v1, v2, c = P
            phi1 = 2 * t if kind == "cheb2" else t
            z1, z2 = w11 + w12 * phi1 + b1, w21 + w22 * phi1 + b2
        return v1 * torch.tanh(z1) + v2 * torch.tanh(z2) + c, (z1, z2)

    def loss():
        t = torch.tensor(T1, requires_grad=True)
        y, (z1, z2) = net(t)
        dy = torch.autograd.grad(y, t, create_graph=True)[0]
        r = dy - y
        y0, _ = net(torch.tensor(0.0))
        Lr, Li = r ** 2, (y0 - 1) ** 2
        info = dict(z=(z1.item(), z2.item()), h=(torch.tanh(z1).item(), torch.tanh(z2).item()),
                    y=y.item(), dy=dy.item(), r=r.item(), y0=y0.item(), Lr=Lr.item(), Li=Li.item())
        return Lr + Li, Lr, Li, info

    L, Lr, Li, info = loss()
    gr = torch.autograd.grad(Lr, P, retain_graph=True)
    gi = torch.autograd.grad(Li, P, retain_graph=True)
    g = torch.autograd.grad(L, P)
    opt = torch.optim.Adam(P, lr=ADAM["lr"], betas=(ADAM["beta1"], ADAM["beta2"]), eps=ADAM["eps"])
    for p, gg in zip(P, g):
        p.grad = gg.clone()
    opt.step()
    manual = []
    for x, gg in zip(init, g):
        gg = gg.item()
        m = (1 - ADAM["beta1"]) * gg
        v = (1 - ADAM["beta2"]) * gg * gg
        mh, vh = m / (1 - ADAM["beta1"]), v / (1 - ADAM["beta2"])
        manual.append(dict(g=gg, m=m, v=v, mh=mh, vh=vh, new=x - ADAM["lr"] * mh / (vh ** 0.5 + ADAM["eps"])))
    L_after = loss()[0].item()
    max_diff = max(abs(p.item() - mm["new"]) for p, mm in zip(P, manual))
    assert max_diff < 1e-15, "manual Adam step and torch.optim.Adam disagree"
    return dict(kind=kind, names=names, init=init, info=info, L=L.item(),
                grad_residual=[x.item() for x in gr], grad_ic=[x.item() for x in gi],
                adam=manual, L_after=L_after, max_abs_diff_adam=max_diff)


def main():
    ex = Example("ch04_ivp", "manual_polynomial_networks",
                 description="Hand calculation for Chebyshev, Legendre and MLP networks (y' = y)",
                 default_seed=0)
    out = [run(k) for k in ("mlp", "cheb1", "leg", "cheb2")]
    for o in out:
        print("==", o["kind"], o["info"], "L=%.6g" % o["L"], "L_after=%.6g" % o["L_after"])
        for n, a, b, ad in zip(o["names"], o["grad_residual"], o["grad_ic"], o["adam"]):
            print(f"  {n:4s} dLr={a:.6g} dLic={b:.6g} g={ad['g']:.6g} m={ad['m']:.6g} "
                  f"v={ad['v']:.6g} theta_1={ad['new']:.6f}")
    ex.save_results({"problem": "y' = y, y(0) = 1, t_1 = 0.5; soft initial condition",
                     "adam": ADAM, "networks": out})


if __name__ == "__main__":
    main()
