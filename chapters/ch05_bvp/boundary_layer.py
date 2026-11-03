"""Boundary-layer BVP: collocation points, a spurious layer, and a layer-enriched network.

    eps y'' + y' = 1  on [0, 1],   y(0) = 0,   y(1) + y'(1) = 1   (Robin condition),  eps = 0.01.

Book: Chapter 5, Section "Worked example: a boundary layer" (figure and table).

Exact solution:  y(t) = t - C (1 - exp(-t/eps)),  C = 1 / (1 - (1 - 1/eps) exp(-1/eps)).
The reduced equation y' = 1 cannot satisfy both conditions; the solution follows the outer
solution t - C away from t = 0 and has a layer of width O(eps) at t = 0, where y' = O(1/eps)
and y'' = O(1/eps^2).

The same PINN (MLP 3 x 32, tanh, soft boundary conditions with lambda_bc = 1, Adam 4000
iterations with cosine decay 1e-3 -> 1e-5, then L-BFGS <= 1000 iterations) is trained with
M = 64 collocation points placed in three ways:

* uniform points;
* Chebyshev-Gauss-Lobatto (CGL) points (clustered at both ends);
* a layer-adapted Shishkin mesh: M/2 intervals on [0, tau], tau = 2 eps ln M, and M/2 - 1 on [tau, 1].

Trained from scratch, all three end on a wrong function: the network follows the smooth function
y = t (which satisfies the equation and y(0) = 0) and meets the Robin condition through a
spurious layer hidden between the last collocation points near t = 1 (CGL points leave no room
for it, and training stalls instead). A fourth run uses the CGL points and a layer-enriched
network  y(t) = N_1(t) + exp(-t/eps) N_2(t)  (one MLP with two outputs, same budget), which
represents the exact solution with the smooth functions N_1 = t - C and N_2 = C.

For comparison, the number of mesh nodes chosen by the adaptive collocation solver
solve_bvp (tol 1e-8) and how many of them fall into the layer are also recorded.
Errors and residuals are evaluated on 1001 uniform points of [0, 1] (spacing 0.1 eps).

Run:  PYTHONPATH=. python chapters/ch05_bvp/boundary_layer.py          (about 3 min CPU, 1 core)
      PYTHONPATH=. python chapters/ch05_bvp/boundary_layer.py --quick
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.integrate import solve_bvp

from pinnode import (Example, MLP, chebyshev_lobatto, derivative, linf, rel_l2, to_tensor, train,
                     uniform)

from _bvp_common import layer_exact, shishkin_points

DISTRIBUTIONS = ["uniform", "cgl", "shishkin", "enriched"]
POINT_SETS = ["uniform", "cgl", "shishkin"]           # the enriched run uses the CGL points
LABELS = {"uniform": "uniform", "cgl": "CGL", "shishkin": "Shishkin",
          "enriched": "CGL + layer function"}
WIDTH, DEPTH, LAM_BC = 32, 3, 1.0


class LayerEnriched(torch.nn.Module):
    """y(t) = N_1(t) + exp(-t/eps) N_2(t) for a network N with two outputs."""

    def __init__(self, net: torch.nn.Module, eps: float):
        super().__init__()
        self.net, self.eps = net, eps

    def forward(self, t):
        n = self.net(t)
        return n[:, :1] + torch.exp(-t / self.eps) * n[:, 1:2]


def collocation_points(kind: str, M: int, eps: float) -> np.ndarray:
    if kind == "uniform":
        return uniform(M, 0.0, 1.0)
    if kind in ("cgl", "enriched"):
        return chebyshev_lobatto(M, 0.0, 1.0)
    if kind == "shishkin":
        return shishkin_points(M, eps)
    raise ValueError(kind)


def make_model(kind: str, eps: float):
    if kind == "enriched":
        return LayerEnriched(MLP(1, 2, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, 1.0)), eps)
    return MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, 1.0))


def residual(model, t, eps):
    y = model(t)
    d1 = derivative(y, t, 1)
    d2 = derivative(d1, t, 1)
    return y, eps * d2 + d1 - 1.0


def run(ex: Example, kind: str, M: int, eps: float, t_test: np.ndarray, y_ref: np.ndarray):
    torch.manual_seed(ex.seed)
    model = make_model(kind, eps)
    pts = collocation_points(kind, M, eps)
    t_col = to_tensor(pts, requires_grad=True)
    t0 = to_tensor([0.0])
    t1 = to_tensor([1.0], requires_grad=True)
    tt = to_tensor(t_test)

    def loss_fn():
        _, r = residual(model, t_col, eps)
        L_r = (r ** 2).mean()
        y1 = model(t1)
        g1 = y1 + derivative(y1, t1, 1) - 1.0
        L_bc = (model(t0) ** 2).sum() + (g1 ** 2).sum()
        return L_r + LAM_BC * L_bc, {"residual": L_r, "bc": L_bc}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(tt), y_ref)}

    n_adam = ex.pick(4000, 30)
    c0 = time.process_time()
    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(1000, 5), eval_fn=eval_fn, eval_every=50, print_every=2000)
    cpu = time.process_time() - c0
    t_req = to_tensor(t_test, requires_grad=True)
    y, r = residual(model, t_req, eps)
    y_pred = y.detach().numpy().ravel()
    res = r.detach().numpy().ravel()
    out = {"distribution": kind, "M": M, "eps": eps, "nodes_in_layer": int(np.sum(pts <= 5 * eps)),
           "min_spacing": float(np.min(np.diff(pts))),
           "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
           "final_loss": hist.final_loss, "final_L_r": hist.parts["residual"][-1],
           "final_L_bc": hist.parts["bc"][-1],
           "rms_residual_test": float(np.sqrt(np.mean(res ** 2))),
           "max_abs_residual_test": float(np.max(np.abs(res))),
           "adam_iters": n_adam, "lbfgs_iters_done": int(hist.iters[-1] - n_adam),
           "train_time_s": hist.train_time, "cpu_time_s": cpu}
    ex.save_history(hist, kind)
    save_run(ex, out, t_test, y_pred, res)
    ex.log(f"{kind:9s} nodes in [0,5eps]={out['nodes_in_layer']:3d}  relL2={out['rel_l2']:.2e}  "
           f"linf={out['linf']:.2e}  loss={out['final_loss']:.2e}  rms R(test)={out['rms_residual_test']:.2e}"
           f"  cpu={cpu:.1f}s")
    return out, hist, pts, y_pred, res


def save_run(ex: Example, out: dict, t_test, y_pred, res) -> None:
    """Per-distribution artifacts, so that an interrupted run can be resumed (--resume)."""
    import json

    ex.path(f"run_{out['distribution']}.json").write_text(json.dumps(out, indent=2))
    ex.save_table({"t": t_test, "y_pred": y_pred, "residual": res}, f"pred_{out['distribution']}")


def load_run(ex: Example, kind: str, M: int, eps: float):
    """Return (out, hist, pts, y_pred, res) of a finished run with the same M and eps, or None."""
    import json

    import pandas as pd

    from _bvp_common import load_history

    f = ex.path(f"run_{kind}.json")
    if not (f.exists() and ex.path(f"pred_{kind}.csv").exists() and ex.path(f"history_{kind}.csv").exists()):
        return None
    out = json.loads(f.read_text())
    if out.get("M") != M or abs(out.get("eps", eps) - eps) > 0:
        return None
    df = pd.read_csv(ex.path(f"pred_{kind}.csv"))
    hist = load_history(ex.path(f"history_{kind}.csv"), out["adam_iters"])
    return out, hist, collocation_points(kind, M, eps), df["y_pred"].to_numpy(), df["residual"].to_numpy()


def adaptive_collocation(eps: float):
    """Mesh selected by solve_bvp (residual control) for the same problem."""
    fun = lambda t, z: np.vstack((z[1], (1.0 - z[1]) / eps))
    bc = lambda za, zb: np.array([za[0], zb[0] + zb[1] - 1.0])
    x0 = np.linspace(0.0, 1.0, 11)
    sol = solve_bvp(fun, bc, x0, np.zeros((2, x0.size)), tol=1e-8, max_nodes=100000)
    t = np.linspace(0.0, 1.0, 1001)
    y = sol.sol(t)[0]
    return {"success": bool(sol.success), "nodes": int(sol.x.size),
            "nodes_in_layer": int(np.sum(sol.x <= 5 * eps)),
            "rel_l2": rel_l2(y, layer_exact(t, eps)), "linf": linf(y, layer_exact(t, eps))}


def main():
    ex = Example("ch05_bvp", "boundary_layer",
                 description="Boundary-layer BVP eps y'' + y' = 1: point sets and a layer-enriched network",
                 default_seed=1234,
                 extra_args=[(["--eps"], {"type": float, "default": 0.01}),
                             (["--M"], {"type": int, "default": 64}),
                             (["--only"], {"default": None, "help": "comma-separated subset (no figure)"}),
                             (["--plot-only"], {"action": "store_true",
                                                "help": "redraw the figure from saved outputs"}),
                             (["--resume"], {"action": "store_true",
                                             "help": "reuse distributions finished in an earlier run"})])
    if ex.args.plot_only:
        replot(ex)
        return
    eps, M = ex.args.eps, ex.pick(ex.args.M, 16)
    t_test = np.linspace(0.0, 1.0, 1001)
    y_ref = layer_exact(t_test, eps)
    kinds = ex.args.only.split(",") if ex.args.only else DISTRIBUTIONS
    rows, store = [], {}
    for kind in kinds:
        done = load_run(ex, kind, M, eps) if ex.args.resume else None
        if done is not None:
            ex.log(f"reusing finished run {kind}")
            out, hist, pts, y_pred, res = done
        else:
            ex.log(f"training with {kind} collocation points (M = {M}, eps = {eps})")
            out, hist, pts, y_pred, res = run(ex, kind, M, eps, t_test, y_ref)
        rows.append(out)
        store[kind] = (hist, pts, y_pred, res)
    adapt = adaptive_collocation(eps)
    ex.log(f"solve_bvp (tol 1e-8): {adapt}")
    table = {"t": t_test, "y_exact": y_ref}
    for kind, (_, _, y_pred, res) in store.items():
        table[f"y_{kind}"] = y_pred
        table[f"residual_{kind}"] = res
    ex.save_table(table, "predictions")
    ex.save_results({"problem": {"equation": "eps y'' + y' = 1", "eps": eps, "interval": [0.0, 1.0],
                                 "bc": "y(0) = 0, y(1) + y'(1) = 1",
                                 "exact": "y = t - C (1 - exp(-t/eps)), C = 1/(1 - (1 - 1/eps) exp(-1/eps))"},
                     "hyperparameters": {"net": f"MLP {DEPTH} x {WIDTH} tanh", "M": M, "lambda_bc": LAM_BC,
                                         "adam_iters": ex.pick(4000, 30), "lr": "1e-3 -> 1e-5 (cosine)",
                                         "lbfgs_iters_max": ex.pick(1000, 5),
                                         "shishkin": "tau = min(1/2, 2 eps ln M), M/2 intervals on [0, tau], M/2 - 1 on [tau, 1]",
                                         "test_grid": "1001 uniform points on [0, 1]"},
                     "runs": rows, "solve_bvp_tol1e-8": adapt})
    if ex.args.no_figures or ex.args.only:
        return
    make_figure(ex, t_test, y_ref, eps, store)


def make_figure(ex: Example, t_test, y_ref, eps, store):
    """(a) layer region with collocation points, (b) pointwise error, (c) residual, (d) error history."""
    from pinnode import plotting as P   # imported lazily: keeps training processes small

    fig, axes = P.figure(2, 2, aspect=0.74)
    ax = axes[0, 0]
    zoom = t_test <= 8 * eps
    P.plot_reference(ax, t_test[zoom], y_ref[zoom], label="exact")
    for k, kind in enumerate(DISTRIBUTIONS):
        if kind not in store:
            continue
        _, pts, y_pred, _ = store[kind]
        P.plot_prediction(ax, t_test[zoom], y_pred[zoom], label=LABELS[kind], k=k)
        if kind in POINT_SETS:                       # one row of ticks per point set
            sel = pts[pts <= 8 * eps]
            ax.plot(sel, np.full(sel.shape, 0.05 + 0.07 * k), ls="none", marker="|", ms=4.5, mew=0.8,
                    color=P.SERIES[k], transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xlim(0.0, 8 * eps)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title("layer region", fontsize=8)
    ax.legend(loc="center right")

    ax = axes[0, 1]
    for k, kind in enumerate(DISTRIBUTIONS):
        if kind not in store:
            continue
        _, _, y_pred, _ = store[kind]
        P.plot_error(ax, t_test, y_pred - y_ref, label=LABELS[kind], k=k)
    ax.set_xscale("symlog", linthresh=eps, linscale=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylabel(r"$|y_\theta(t)-y(t)|$")
    ax.set_title("pointwise error", fontsize=8)
    ax.set_ylim(1e-7, 1e3)
    ax.legend(loc="upper center", ncol=2, handlelength=1.5, columnspacing=0.8)

    ax = axes[1, 0]
    for k, kind in enumerate(DISTRIBUTIONS):
        if kind not in store:
            continue
        _, _, _, res = store[kind]
        P.plot_error(ax, t_test, res, label=LABELS[kind], k=k, ylabel=r"$|\mathcal{R}(t)|$")
    ax.set_xscale("symlog", linthresh=eps, linscale=0.6)
    ax.set_xlim(0.0, 1.0)
    ax.set_title("residual on the 1001-point test grid", fontsize=8)
    ax.set_ylim(1e-7, 1e5)
    ax.legend(loc="upper center", ncol=2, handlelength=1.5, columnspacing=0.8)

    ax = axes[1, 1]
    for k, kind in enumerate(DISTRIBUTIONS):
        if kind not in store:
            continue
        P.plot_metric(ax, store[kind][0], "rel_l2", label=LABELS[kind], k=k, show_switch=(k == 0))
    ax.set_title(r"relative $L^2$ error during training", fontsize=8)
    ax.set_ylim(1e-4, 1e3)
    ax.legend(loc="upper center", ncol=2, handlelength=1.5, columnspacing=0.8)
    P.label_panels(axes)
    ex.savefig(fig, "boundary_layer_summary")


def replot(ex: Example):
    """Redraw the figure from results.json, predictions.csv and the history files."""
    import json

    import pandas as pd

    from _bvp_common import load_history

    res = json.loads(ex.path("results.json").read_text())
    eps = res["problem"]["eps"]
    df = pd.read_csv(ex.path("predictions.csv"))
    t_test, y_ref = df["t"].to_numpy(), df["y_exact"].to_numpy()
    store = {}
    for r in res["runs"]:
        kind = r["distribution"]
        hist = load_history(ex.path(f"history_{kind}.csv"), r["adam_iters"])
        store[kind] = (hist, collocation_points(kind, r["M"], eps), df[f"y_{kind}"].to_numpy(),
                       df[f"residual_{kind}"].to_numpy())
    make_figure(ex, t_test, y_ref, eps, store)


if __name__ == "__main__":
    main()
