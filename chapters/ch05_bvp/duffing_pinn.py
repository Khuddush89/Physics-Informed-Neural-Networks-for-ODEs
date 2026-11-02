"""Duffing-type BVP  y'' + y + eps y^3 = 0  on [0, pi],  y(0) = 1,  y'(pi) = 0,  eps = 0.1.

Book: Chapter 5 (Boundary value problems), Section "Worked example: a Duffing-type BVP
with mixed boundary conditions" (composite figure and the table of PINN variants).

Exact solutions: every solution has the form y(t) = A cn(w (t - pi), m) with
w^2 = 1 + eps A^2 and m = eps A^2 / (2 w^2) (Jacobi elliptic function cn); the boundary
condition y(0) = 1 selects the roots of F(A) = A cn(w pi, m) - 1. The BVP has infinitely
many solutions; the one of smallest amplitude (A_1 = -1.00704...) is close to cos t.
See duffing_reference.py for the verification against solve_bvp and shooting.

PINN variants (same collocation points, optimiser and budget):

* ChNN, soft boundary conditions, lambda_bc = 1           (main run, figure panels a-c)
* ChNN, soft boundary conditions, lambda_bc = 100         (weight of the original example)
* ChNN, hard boundary conditions through the constrained expression
      y(t) = 1 + N(t) - N(0) - t N'(pi)                   (exact y(0) = 1, y'(pi) = 0)
* MLP (2 x 20, tanh), soft, lambda_bc = 1                 (architecture control)
* ChNN, soft, lambda_bc = 1, initialised by fitting the crude guess -3.6 cos(1.5 (t - pi))
  before physics training: converges to a different solution of the same BVP (panel d).

ChNN = Chebyshev features T_0..T_4 of xi = 2t/pi - 1, followed by 2 tanh layers of 10
neurons (the architecture of the original example). Collocation: 64 Chebyshev-Gauss-Lobatto
points. Optimiser: Adam (3000 iterations, cosine decay 1e-3 -> 1e-5) then L-BFGS (<= 1000
iterations, strong Wolfe line search). Errors on 1001 uniform points of [0, pi].

Run:  PYTHONPATH=. python chapters/ch05_bvp/duffing_pinn.py          (about 2-3 min CPU, 1 core)
      PYTHONPATH=. python chapters/ch05_bvp/duffing_pinn.py --quick  (smoke test)
"""
from __future__ import annotations

import time

import numpy as np
import torch

from pinnode import (ChebyshevNet, Example, MLP, chebyshev_lobatto, count_parameters,
                     derivative, linf, rel_l2, to_tensor, train)

from _bvp_common import (DY_END, EPS_DUFFING, T_END, Y0, duffing_A1, duffing_exact,
                         duffing_roots)

DEGREE, HIDDEN = 4, (10, 10)          # ChNN of the original example
MLP_WIDTH, MLP_DEPTH = 20, 2


class MixedConstraint(torch.nn.Module):
    """Constrained expression for y(a) = alpha, y'(b) = beta (theory of functional connections):

        y(t) = alpha + beta (t - a) + N(t) - N(a) - (t - a) N'(b).

    Every smooth function satisfying both conditions is of this form (take N = y), and every
    network N gives an admissible function, so the boundary conditions hold exactly.
    """

    def __init__(self, net: torch.nn.Module, a: float, alpha: float, b: float, beta: float):
        super().__init__()
        self.net, self.a, self.alpha, self.b, self.beta = net, a, alpha, b, beta

    def forward(self, t):
        ta = torch.full((1, 1), self.a)
        tb = torch.full((1, 1), self.b, requires_grad=True)
        with torch.enable_grad():
            nb = self.net(tb)
            dnb, = torch.autograd.grad(nb.sum(), tb, create_graph=True)
        return self.alpha + self.beta * (t - self.a) + self.net(t) - self.net(ta) - (t - self.a) * dnb


def guess_branch2(t):
    """Crude guess with the shape of the second solution (one interior maximum, y(pi) ~ -3.6)."""
    return -3.6 * np.cos(1.5 * (np.asarray(t) - T_END))


def make_model(kind: str, hard: bool):
    if kind == "chnn":
        net = ChebyshevNet(degree=DEGREE, hidden=HIDDEN, activation="tanh", domain=(0.0, T_END))
    elif kind == "mlp":
        net = MLP(1, 1, width=MLP_WIDTH, depth=MLP_DEPTH, activation="tanh", domain=(0.0, T_END))
    else:
        raise ValueError(kind)
    return MixedConstraint(net, 0.0, Y0, T_END, DY_END) if hard else net


def residual(model, t):
    y = model(t)
    return y, derivative(y, t, 2) + y + EPS_DUFFING * y ** 3


def run(ex: Example, cfg: dict, t_col_np: np.ndarray, t_test: np.ndarray, roots):
    torch.manual_seed(ex.seed)
    model = make_model(cfg["net"], cfg["hard"])
    lam = cfg.get("lam_bc", 1.0)
    t_col = to_tensor(t_col_np, requires_grad=True)
    ta = to_tensor([0.0])
    tb = to_tensor([T_END], requires_grad=True)
    tt = to_tensor(t_test)

    def loss_fn():
        _, r = residual(model, t_col)
        L_r = (r ** 2).mean()
        if cfg["hard"]:
            return L_r, {"residual": L_r}
        g0 = model(ta) - Y0
        g1 = derivative(model(tb), tb, 1) - DY_END
        L_bc = (g0 ** 2).sum() + (g1 ** 2).sum()
        return L_r + lam * L_bc, {"residual": L_r, "bc": L_bc}

    A_target = roots[0]
    y_target = duffing_exact(t_test, A_target)

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(tt), y_target)}

    c0 = time.process_time()
    pre_time = 0.0
    if cfg.get("pretrain"):
        # least-squares fit of the network to a crude guess (no physics), as one would give an
        # initial guess to a Newton-type BVP solver
        t_fit = to_tensor(np.linspace(0.0, T_END, 200))
        g_fit = to_tensor(guess_branch2(np.linspace(0.0, T_END, 200)))

        def fit_loss():
            L = ((model(t_fit) - g_fit) ** 2).mean()
            return L, {"fit": L}

        h_pre = train(model.parameters(), fit_loss, adam_iters=ex.pick(1000, 20), lr=1e-2,
                      lbfgs_iters=0, verbose=False)
        pre_time = h_pre.train_time
    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(3000, 30), lr=1e-3,
                 lr_final=1e-5, lbfgs_iters=ex.pick(1000, 5), eval_fn=eval_fn, eval_every=50,
                 print_every=1000)
    cpu_time = time.process_time() - c0

    # identify the solution branch that was found (nearest closed-form root in y(pi))
    with torch.no_grad():
        y_end = float(model(to_tensor([T_END])).item())
    A_found = min(roots, key=lambda a: abs(a - y_end))
    y_ref = duffing_exact(t_test, A_found)
    t_req = to_tensor(t_test, requires_grad=True)
    y, r = residual(model, t_req)
    y_pred = y.detach().numpy().ravel()
    res = r.detach().numpy().ravel()
    y0_err = abs(float(model(ta).item()) - Y0)
    dyb_err = abs(float(derivative(model(tb), tb, 1).item()) - DY_END)
    out = {
        "name": cfg["name"], "label": cfg["label"], "net": cfg["net"], "hard": cfg["hard"],
        "lam_bc": None if cfg["hard"] else lam, "pretrain": bool(cfg.get("pretrain")),
        "n_params": count_parameters(model),
        "A_found": A_found, "branch_index": int(roots.index(A_found)) + 1,
        "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
        "bc_error_y0": y0_err, "bc_error_dy_end": dyb_err,
        "max_abs_residual_test": float(np.max(np.abs(res))),
        "rms_residual_test": float(np.sqrt(np.mean(res ** 2))),
        "final_loss": hist.final_loss,
        "final_L_r": hist.parts["residual"][-1],
        "final_L_bc": hist.parts["bc"][-1] if "bc" in hist.parts else 0.0,
        "adam_iters": ex.pick(3000, 30),
        "lbfgs_iters_done": int(hist.iters[-1] - ex.pick(3000, 30)),
        "train_time_s": hist.train_time + pre_time, "cpu_time_s": cpu_time,
    }
    ex.save_history(hist, cfg["name"])
    save_run(ex, out, t_test, y_pred, y_ref, res)
    ex.log(f"{cfg['name']:14s} branch A={A_found:+.5f}  relL2={out['rel_l2']:.2e}  "
           f"linf={out['linf']:.2e}  |y(0)-1|={y0_err:.1e}  |y'(pi)|={dyb_err:.1e}  "
           f"max|R|={out['max_abs_residual_test']:.1e}  cpu={cpu_time:.1f}s")
    return out, hist, y_pred, y_ref, res


def save_run(ex: Example, out: dict, t_test, y_pred, y_ref, res) -> None:
    """Per-configuration artifacts, so that an interrupted run can be resumed (--resume)."""
    import json

    ex.path(f"run_{out['name']}.json").write_text(json.dumps(out, indent=2))
    ex.save_table({"t": t_test, "y_pred": y_pred, "y_ref": y_ref, "residual": res}, f"pred_{out['name']}")


def load_run(ex: Example, name: str):
    """Return (out, hist, y_pred, y_ref, res) of a finished configuration, or None."""
    import json

    import pandas as pd

    from _bvp_common import load_history

    f = ex.path(f"run_{name}.json")
    if not (f.exists() and ex.path(f"pred_{name}.csv").exists() and ex.path(f"history_{name}.csv").exists()):
        return None
    out = json.loads(f.read_text())
    df = pd.read_csv(ex.path(f"pred_{name}.csv"))
    hist = load_history(ex.path(f"history_{name}.csv"), out["adam_iters"])
    return out, hist, df["y_pred"].to_numpy(), df["y_ref"].to_numpy(), df["residual"].to_numpy()


CONFIGS = [
    {"name": "chnn_soft", "label": "ChNN, soft, $\\lambda_{bc}=1$", "net": "chnn", "hard": False, "lam_bc": 1.0},
    {"name": "chnn_soft_w100", "label": "ChNN, soft, $\\lambda_{bc}=100$", "net": "chnn", "hard": False, "lam_bc": 100.0},
    {"name": "chnn_hard", "label": "ChNN, hard (constrained expression)", "net": "chnn", "hard": True},
    {"name": "mlp_soft", "label": "MLP $2\\times 20$, soft, $\\lambda_{bc}=1$", "net": "mlp", "hard": False, "lam_bc": 1.0},
    {"name": "chnn_branch2", "label": "ChNN, soft, $\\lambda_{bc}=1$, fitted to guess first", "net": "chnn",
     "hard": False, "lam_bc": 1.0, "pretrain": True},
]


ERR_LABELS = {"chnn_soft": r"ChNN, $\lambda_{bc}=1$", "chnn_soft_w100": r"ChNN, $\lambda_{bc}=100$",
              "chnn_hard": "ChNN, hard", "mlp_soft": r"MLP, $\lambda_{bc}=1$"}


def make_figure(ex: Example, t_test, t_col_np, runs, preds, hists):
    """2 x 2 summary: (a) solution and CGL nodes, (b) pointwise errors, (c) loss components,
    (d) a second solution reached from a different initialisation."""
    from pinnode import plotting as P   # imported lazily: keeps training processes small

    byname = {r["name"]: r for r in runs}
    fig, axes = P.figure(2, 2, aspect=0.74)
    ax = axes[0, 0]
    y_main, y_ref_main, _ = preds["chnn_soft"]
    P.plot_reference(ax, t_test, y_ref_main, label="exact")
    P.plot_prediction(ax, t_test, y_main, label="PINN (ChNN)", k=0, markevery=(40, 100))
    P.rug(ax, t_col_np, label="CGL nodes")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("solution", fontsize=8)
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for k, name in enumerate(["chnn_soft", "chnn_soft_w100", "chnn_hard", "mlp_soft"]):
        y_pred, y_ref, _ = preds[name]
        P.plot_error(ax, t_test, y_pred - y_ref, label=ERR_LABELS[name], k=k)
    ax.set_ylabel(r"$|y_\theta(t)-y(t)|$")
    ax.set_title("pointwise error", fontsize=8)
    ax.set_ylim(1e-10, 1e-2)       # the hard variant is exact at t = 0 (error floored otherwise)
    ax.legend(loc="upper center", ncol=2)

    ax = axes[1, 0]
    P.plot_history(ax, hists["chnn_soft"], parts=["residual", "bc"],
                   labels={"loss": "total", "residual": r"$\mathcal{L}_r$", "bc": r"$\mathcal{L}_{bc}$"})
    ax.legend(loc="lower left")
    ax.set_title(r"loss components (ChNN, $\lambda_{bc}=1$)", fontsize=8)

    ax = axes[1, 1]
    y_b2, y_ref_b2, _ = preds["chnn_branch2"]
    A_b2 = byname["chnn_branch2"]["A_found"]
    ax.plot(t_test, guess_branch2(t_test), color=P.COLORS["muted"], lw=1.0, ls=":", label="initial guess")
    P.plot_reference(ax, t_test, y_ref_b2, label=f"exact, $A={A_b2:.3f}$")
    P.plot_prediction(ax, t_test, y_b2, label="PINN (ChNN)", k=0, markevery=(40, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("another solution of the same BVP", fontsize=8)
    ax.legend(loc="lower left")
    P.label_panels(axes)
    ex.savefig(fig, "duffing_summary")


def replot(ex: Example, t_col_np):
    """Regenerate the figure from results.json, predictions.csv and the history files."""
    import json

    import pandas as pd

    from _bvp_common import load_history

    runs = json.loads(ex.path("results.json").read_text())["runs"]
    df = pd.read_csv(ex.path("predictions.csv"))
    t_test = df["t"].to_numpy()
    preds, hists = {}, {}
    for r in runs:
        name = r["name"]
        preds[name] = (df[f"y_{name}"].to_numpy(), duffing_exact(t_test, r["A_found"]),
                       df[f"residual_{name}"].to_numpy())
        hists[name] = load_history(ex.path(f"history_{name}.csv"), r["adam_iters"])
    make_figure(ex, t_test, t_col_np, runs, preds, hists)


def main():
    ex = Example("ch05_bvp", "duffing_pinn",
                 description="Duffing-type BVP with mixed conditions: ChNN/MLP PINNs, soft/hard BCs",
                 default_seed=1234,
                 extra_args=[(["--only"], {"default": None,
                                           "help": "comma-separated subset of configurations (no figure)"}),
                             (["--plot-only"], {"action": "store_true",
                                                "help": "redraw the figure from saved outputs"}),
                             (["--resume"], {"action": "store_true",
                                             "help": "reuse configurations finished in an earlier run"})])
    M = ex.pick(64, 16)
    t_col_np = chebyshev_lobatto(M, 0.0, T_END)
    if ex.args.plot_only:
        replot(ex, t_col_np)
        return
    t_test = np.linspace(0.0, T_END, 1001)
    roots = duffing_roots(A_max=12.0, n_grid=ex.pick(240001, 24001))
    A1 = duffing_A1()
    assert abs(roots[0] - A1) < 1e-12
    ex.log(f"closed-form roots A = y(pi): {', '.join(f'{a:+.6f}' for a in roots)}")

    configs = CONFIGS
    if ex.args.only:
        keep = ex.args.only.split(",")
        configs = [c for c in CONFIGS if c["name"] in keep]
    runs, preds, hists = [], {}, {}
    for cfg in configs:
        done = load_run(ex, cfg["name"]) if ex.args.resume else None
        if done is not None:
            ex.log(f"reusing finished configuration {cfg['name']}")
            out, hist, y_pred, y_ref, res = done
        else:
            ex.log(f"training {cfg['name']}")
            out, hist, y_pred, y_ref, res = run(ex, cfg, t_col_np, t_test, roots)
        runs.append(out)
        preds[cfg["name"]] = (y_pred, y_ref, res)
        hists[cfg["name"]] = hist

    pred_table = {"t": t_test, "y_exact_A1": duffing_exact(t_test, A1)}
    for name, (y_pred, y_ref, res) in preds.items():
        pred_table[f"y_{name}"] = y_pred
        pred_table[f"residual_{name}"] = res
    ex.save_table(pred_table, "predictions")
    ex.save_results({
        "problem": {"equation": "y'' + y + eps y^3 = 0", "eps": EPS_DUFFING, "interval": [0.0, T_END],
                    "bc": "y(0) = 1, y'(pi) = 0", "roots_A": roots, "A1": A1},
        "hyperparameters": {"chnn": {"degree": DEGREE, "hidden": list(HIDDEN), "activation": "tanh"},
                            "mlp": {"width": MLP_WIDTH, "depth": MLP_DEPTH, "activation": "tanh"},
                            "collocation": f"{M} Chebyshev-Gauss-Lobatto points",
                            "adam_iters": ex.pick(3000, 30), "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters_max": ex.pick(1000, 5), "init": "Xavier normal, zero bias",
                            "pretrain_branch2": {"iters": ex.pick(1000, 20), "lr": "1e-2 -> 1e-4",
                                                 "guess": "-3.6 cos(1.5 (t - pi))"},
                            "test_grid": "1001 uniform points on [0, pi]"},
        "runs": runs,
    })
    if ex.args.no_figures or ex.args.only:
        return
    make_figure(ex, t_test, t_col_np, runs, preds, hists)


if __name__ == "__main__":
    main()
