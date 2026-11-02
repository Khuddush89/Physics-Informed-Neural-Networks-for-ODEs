"""A BVP without solution: what the PINN loss does.

    y'' + y = 0  on [0, pi],   y(0) = 0,   y(pi) = 1.

Book: Chapter 5, Section "Two-point boundary value problems", subsection "Existence,
uniqueness and what the loss sees"; Exercise (loss floor) and its solution in Appendix C.

The homogeneous problem has the nontrivial solution sin t, and integrating R = y'' + y
against sin t gives  int_0^pi R(t) sin t dt = y(0) + y(pi)  for every C^2 function. Hence

* hard boundary conditions: every admissible y has  (1/pi) int R^2 dt >= 2/pi^2 = 0.2026,
  with equality exactly for  y = -t cos(t)/pi + c sin t,  c in R (a whole family);
* soft boundary conditions with weight lam: the infimum of
  (1/pi) int R^2 dt + lam (y(0)^2 + (y(pi) - 1)^2)  is  2 lam / (lam pi^2 + 4), attained with
  y(0) = -2/(lam pi^2 + 4)  and  y(pi) = 1 - 2/(lam pi^2 + 4).

The script trains a small MLP PINN with hard conditions (two seeds) and with soft
conditions (lam = 1), records the training loss (mean over 100 uniform collocation points),
the mean-square residual on a fine grid (trapezoidal rule on 1001 points, an approximation of
(1/pi) int R^2 dt) and compares with the predicted infima. For the hard runs it also
projects the network onto the minimiser family to obtain c.

Run:  PYTHONPATH=. python chapters/ch05_bvp/resonance_loss_floor.py          (about 1 min CPU)
      PYTHONPATH=. python chapters/ch05_bvp/resonance_loss_floor.py --quick
"""
from __future__ import annotations

import time

import numpy as np
import torch

from pinnode import (ConstrainedModel, Example, MLP, derivative, dirichlet_transform, to_tensor,
                     train, trapezoid_weights, uniform)

T = np.pi
LAM_SOFT = 1.0


def floor_hard() -> float:
    return 2.0 / np.pi ** 2


def floor_soft(lam: float) -> float:
    return 2.0 * lam / (lam * np.pi ** 2 + 4.0)


def run(ex: Example, hard: bool, seed: int, lam: float = LAM_SOFT):
    torch.manual_seed(seed)
    net = MLP(1, 1, width=20, depth=2, activation="tanh", domain=(0.0, T))
    model = ConstrainedModel(net, dirichlet_transform(0.0, 0.0, T, 1.0)) if hard else net
    t_col = to_tensor(uniform(ex.pick(100, 20), 0.0, T), requires_grad=True)
    ta, tb = to_tensor([0.0]), to_tensor([T])

    def loss_fn():
        y = model(t_col)
        L_r = ((derivative(y, t_col, 2) + y) ** 2).mean()
        if hard:
            return L_r, {"residual": L_r}
        L_bc = (model(ta) ** 2).sum() + ((model(tb) - 1.0) ** 2).sum()
        return L_r + lam * L_bc, {"residual": L_r, "bc": L_bc}

    c0 = time.process_time()
    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(3000, 30), lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(500, 5), print_every=1000)
    cpu = time.process_time() - c0
    t = np.linspace(0.0, T, 1001)
    tr = to_tensor(t, requires_grad=True)
    y = model(tr)
    R = (derivative(y, tr, 2) + y).detach().numpy().ravel()
    y = y.detach().numpy().ravel()
    w = trapezoid_weights(t)
    out = {"formulation": "hard" if hard else "soft", "seed": seed, "lam_bc": None if hard else lam,
           "final_loss": hist.final_loss, "final_L_r": hist.parts["residual"][-1],
           "mean_square_residual_fine_grid": float(np.sum(w * R ** 2) / T),
           "y0": float(y[0]), "yT": float(y[-1]),
           "predicted_floor": floor_hard() if hard else floor_soft(lam),
           "train_time_s": hist.train_time, "cpu_time_s": cpu}
    if hard:
        # least-squares projection onto the minimiser family  -t cos t / pi + c sin t
        base = -t * np.cos(t) / np.pi
        c = float(np.sum(w * (y - base) * np.sin(t)) / np.sum(w * np.sin(t) ** 2))
        out["c"] = c
        out["max_dev_from_family"] = float(np.max(np.abs(y - base - c * np.sin(t))))
    else:
        out["predicted_y0"] = -2.0 / (lam * np.pi ** 2 + 4.0)
        out["predicted_yT"] = 1.0 - 2.0 / (lam * np.pi ** 2 + 4.0)
    ex.log(", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in out.items()))
    ex.save_history(hist, f"{out['formulation']}_seed{seed}")
    return out


def main():
    ex = Example("ch05_bvp", "resonance_loss_floor",
                 description="BVP without solution (y''+y=0, y(0)=0, y(pi)=1): loss floor of PINNs",
                 default_seed=1234)
    runs = [run(ex, hard=True, seed=ex.seed), run(ex, hard=True, seed=ex.seed + 1),
            run(ex, hard=False, seed=ex.seed)]
    ex.save_table(runs, "summary")
    ex.save_results({"problem": {"equation": "y'' + y = 0", "interval": [0.0, T], "bc": "y(0)=0, y(pi)=1",
                                 "floor_hard": floor_hard(), "floor_soft_lam1": floor_soft(LAM_SOFT)},
                     "hyperparameters": {"net": "MLP 2 x 20 tanh", "collocation": "100 uniform points",
                                         "adam_iters": ex.pick(3000, 30), "lr": "1e-3 -> 1e-5 (cosine)",
                                         "lbfgs_iters_max": ex.pick(500, 5)},
                     "runs": runs})


if __name__ == "__main__":
    main()
