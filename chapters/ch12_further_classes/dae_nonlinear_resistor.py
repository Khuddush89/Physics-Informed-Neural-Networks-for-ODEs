"""Index-1 semi-explicit DAE: a capacitor discharging through a nonlinear resistor.

Book: Chapter 12 (Further equation classes), Section "Differential-algebraic equations".

    y'(t) = -z(t),                 (capacitor: charge = voltage y, current z)
    0     = z + z^3 - y,           (resistor with characteristic v = i + i^3)
    y(0)  = 2,  t in [0, 3].

Since dg/dz = 1 + 3 z^2 >= 1, the DAE has index one; the consistent initial value is z(0) = 1.
Differentiating the constraint gives (1 + 3 z^2) z' = -z, hence ln z + 3 z^2 / 2 = 3/2 - t and

    z(t) = sqrt( W(3 exp(3 - 2t)) / 3 ),     y(t) = z + z^3,

with W the principal branch of the Lambert W function.

Three PINN formulations:
* hard      - the network represents y only (y = 2 + t N(t)); z = Z(y) solves the constraint inside
              the forward pass by a few Newton steps (differentiated by autograd), so g = 0 to
              rounding and the loss is the differential residual alone;
* penalty   - the network outputs (y, z); the loss is mean (y' + z)^2 + w_a mean g^2, w_a = 1 and 100.

Run:  python chapters/ch12_further_classes/dae_nonlinear_resistor.py          (about 1-2 min on 1 core)
      python chapters/ch12_further_classes/dae_nonlinear_resistor.py --quick  (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.special import lambertw

from pinnode import ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor, train, uniform
from pinnode import plotting as P

Y0, T_END = 2.0, 3.0
PENALTY_WEIGHTS = [1.0, 100.0]


def exact(t):
    z = np.sqrt(lambertw(3.0 * np.exp(3.0 - 2.0 * np.asarray(t, float))).real / 3.0)
    return z + z ** 3, z


def newton_z(y: torch.Tensor, iters: int = 8) -> torch.Tensor:
    """Solve z + z^3 = y for z by Newton's method (differentiable; one real root for every y)."""
    z = y / (1.0 + y ** 2) ** (1.0 / 3.0)          # smooth initial guess ~ y (small y), y^(1/3) (large y)
    for _ in range(iters):
        z = z - (z + z ** 3 - y) / (1.0 + 3.0 * z ** 2)
    return z


def run(ex: Example, mode: str, w_a: float, t_test):
    torch.manual_seed(ex.seed)
    t_col = to_tensor(uniform(ex.pick(200, 30), 0.0, T_END), requires_grad=True)
    t_test_t = to_tensor(t_test)
    y_ref, z_ref = exact(t_test)
    if mode == "hard":
        net = MLP(1, 1, width=32, depth=3, activation="tanh", domain=(0.0, T_END))
        ymodel = ConstrainedModel(net, lambda t, n: Y0 + t * n)

        def fields(t):
            y = ymodel(t)
            return y, newton_z(y)
    else:
        net = MLP(1, 2, width=32, depth=3, activation="tanh", domain=(0.0, T_END))

        def fields(t):
            n = net(t)
            return Y0 + t * n[:, :1], n[:, 1:2]

    def loss_fn():
        y, z = fields(t_col)
        r = derivative(y, t_col) + z
        L_d = (r ** 2).mean()
        if mode == "hard":
            return L_d, {"differential": L_d}
        g = z + z ** 3 - y
        L_a = (g ** 2).mean()
        return L_d + w_a * L_a, {"differential": L_d, "algebraic": L_a}

    def eval_fn():
        with torch.no_grad():
            y, z = fields(t_test_t)
            return {"rel_l2_y": rel_l2(y, y_ref), "rel_l2_z": rel_l2(z, z_ref)}

    hist = train(net.parameters(), loss_fn, adam_iters=ex.pick(5000, 60), lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=100, print_every=2500)
    with torch.no_grad():
        y, z = fields(t_test_t)
        y, z = y.numpy().ravel(), z.numpy().ravel()
        z0 = float(fields(to_tensor([0.0]))[1].item())
    g = z + z ** 3 - y
    return {"hist": hist, "y": y, "z": z, "g": g, "rel_l2_y": rel_l2(y, y_ref), "rel_l2_z": rel_l2(z, z_ref),
            "linf_y": linf(y, y_ref), "linf_z": linf(z, z_ref), "max_abs_g": float(np.max(np.abs(g))),
            "z0": z0}


def main():
    ex = Example("ch12_further_classes", "dae_nonlinear_resistor",
                 description="Index-1 DAE y' = -z, 0 = z + z^3 - y: hard (Newton layer) vs penalty",
                 default_seed=1234)
    t_test = np.linspace(0.0, T_END, 1001)
    y_ref, z_ref = exact(t_test)
    # check of the closed form: residuals of both equations (finite differences for y')
    dy_fd = np.gradient(y_ref, t_test, edge_order=2)
    exact_check = {"max_abs_constraint": float(np.max(np.abs(z_ref + z_ref ** 3 - y_ref))),
                   "max_abs_ode_residual_fd": float(np.max(np.abs(dy_fd + z_ref)[5:-5])),
                   "z0": float(exact(0.0)[1])}
    ex.log(f"exact-solution check: {exact_check}")
    variants = [("hard", "hard", 0.0)] + [(f"penalty_w{int(w)}", "penalty", w) for w in PENALTY_WEIGHTS]
    runs, rows = {}, []
    for tag, mode, w in variants:
        ex.log(f"training {tag}")
        r = run(ex, mode, w, t_test)
        runs[tag] = r
        ex.save_history(r["hist"], tag)
        rows.append({"formulation": tag, "w_a": w, "rel_l2_y": r["rel_l2_y"], "rel_l2_z": r["rel_l2_z"],
                     "linf_y": r["linf_y"], "linf_z": r["linf_z"], "max_abs_g": r["max_abs_g"],
                     "z0": r["z0"], "final_loss": r["hist"].final_loss, "train_time_s": r["hist"].train_time})
        ex.log(f"{tag:12s} relL2 y={r['rel_l2_y']:.2e} z={r['rel_l2_z']:.2e} max|g|={r['max_abs_g']:.1e} "
               f"z(0)={r['z0']:.6f} time={r['hist'].train_time:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_results({
        "problem": {"equations": "y' = -z, 0 = z + z^3 - y", "y0": Y0, "T": T_END, "index": 1,
                    "exact": "z = sqrt(W(3 exp(3-2t))/3), y = z + z^3"},
        "exact_solution_check": exact_check,
        "hyperparameters": {"width": 32, "depth": 3, "activation": "tanh",
                            "collocation": ex.pick(200, 30), "adam_iters": ex.pick(5000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10),
                            "newton_iterations": 8, "penalty_weights": PENALTY_WEIGHTS},
        "summary": rows,
        "rel_l2": {r["formulation"]: {"y": r["rel_l2_y"], "z": r["rel_l2_z"]} for r in rows},
        "train_time_s": {r["formulation"]: r["train_time_s"] for r in rows},
        "test_grid": "1001 uniform points on [0, 3]",
    })
    if ex.args.no_figures:
        return

    labels = {"hard": "hard (Newton layer)", "penalty_w1": "penalty, $w_a=1$",
              "penalty_w100": "penalty, $w_a=100$"}
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    P.plot_reference(ax, t_test, y_ref, label="exact")
    P.plot_reference(ax, t_test, z_ref, label=None)
    for k, tag in enumerate(labels):
        P.plot_prediction(ax, t_test, runs[tag]["y"], label=labels[tag], k=k, markevery=(40 * k, 150))
        P.plot_prediction(ax, t_test, runs[tag]["z"], label=None, k=k, markevery=(40 * k + 75, 150))
    ax.text(0.45, 1.80, "$y$", fontsize=9)
    ax.text(0.45, 0.66, "$z$", fontsize=9)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$, $z(t)$")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for k, tag in enumerate(labels):
        P.plot_error(ax, t_test[1:], (runs[tag]["z"] - z_ref)[1:], label=labels[tag], k=k,
                     ylabel="error in $z$")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    for k, tag in enumerate(labels):
        g = np.maximum(np.abs(runs[tag]["g"]), 1e-17)
        ax.semilogy(t_test, g, color=P.SERIES[k], lw=1.0, ls="-" if k == 0 else P.LINESTYLES[k - 1],
                    label=labels[tag])
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"constraint $|z+z^3-y|$")
    ax.legend(loc="center right")

    ax = axes[1, 1]
    for k, tag in enumerate(labels):
        h = runs[tag]["hist"]
        ax.semilogy(h.iters, P.rolling_median(h.loss, 9), color=P.SERIES[k],
                    ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.1, label=labels[tag])
    ax.axvline(runs["hard"]["hist"].switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")
    P.label_panels(axes)
    ex.savefig(fig, "dae_summary")


if __name__ == "__main__":
    main()
