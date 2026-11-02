"""Optimizers and floating-point precision on the nonlinear pendulum.

Book: Chapter 9, Section "Optimizers and floating-point precision".

Problem:  theta'' + sin(theta) = 0,  theta(0) = 1,  theta'(0) = 0,  t in [0, 10]
(about 1.5 periods of a large-amplitude pendulum). Reference: DOP853, rtol = atol = 1e-12.
Trial function theta(t) = 1 + t^2 N(t) (hard initial conditions); tanh MLP 32 x 3; M = 200.

Optimizer schedules, each in float64 and float32:

* adam        - 10000 Adam steps, learning rate 1e-3 -> 1e-5 (cosine);
* adam+lbfgs  - the same Adam run continued by at most 1000 L-BFGS iterations;
* short+lbfgs - 1000 Adam steps (same schedule compressed) then at most 1000 L-BFGS iterations;
* lbfgs       - at most 1000 L-BFGS iterations from the initialization.

Both precisions start from the same initial weights (drawn in float64, then cast).

L-BFGS uses a strong Wolfe line search, history 50, and stops early when the loss decreases
by less than 1e-4 (relative) over 50 iterations (pinnode defaults). The number of loss and
gradient evaluations and the process CPU time are recorded; the table written to
``tab_optimizers.tex`` is regenerated from ``results.json`` by ``_make_tables.py``.

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/optimizers.py          (~4 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/optimizers.py --quick  (smoke test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CPUTimer, RunCache, replot_arg  # noqa: E402
from pinnode import ConstrainedModel, Example, MLP, derivative, linf, rel_l2, to_tensor, train  # noqa: E402
from pinnode import plotting as P  # noqa: E402
from pinnode.reference import ivp_reference  # noqa: E402

T, THETA0 = 10.0, 1.0
SCHEDULES = ["adam", "adam+lbfgs", "short+lbfgs", "lbfgs"]
LABELS = {"adam": "Adam", "adam+lbfgs": r"Adam $\to$ L-BFGS", "short+lbfgs": r"short Adam $\to$ L-BFGS",
          "lbfgs": "L-BFGS only"}


def reference(t):
    return ivp_reference(lambda s, u: [u[1], -np.sin(u[0])], (0.0, T), [THETA0, 0.0], t)[:, 0]


class Problem:
    """Network, loss and evaluation counters for one precision."""

    def __init__(self, seed, dtype, M, t_test, y_ref):
        # initialize in double precision and cast, so that both precisions start from the same
        # weights (drawing them directly in float32 would give a different initialization)
        torch.manual_seed(seed)
        self.dtype = dtype
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        net = MLP(1, 1, 32, 3, "tanh", domain=(0.0, T))
        torch.set_default_dtype(prev)
        self.net = net.to(dtype)
        self.model = ConstrainedModel(self.net, lambda t, n: THETA0 + t ** 2 * n)
        self.t_col = to_tensor(np.linspace(0.0, T, M), requires_grad=True)
        self.t_test_t, self.y_ref = to_tensor(t_test), y_ref
        self.n_evals = 0
        self.curve = {"evals": [], "rel_l2": [], "loss": []}

    def loss_fn(self):
        self.n_evals += 1
        y = self.model(self.t_col)
        r = derivative(y, self.t_col, 2) + torch.sin(y)
        L = (r ** 2).mean()
        return L, {"residual": L}

    def eval_fn(self):
        with torch.no_grad():
            yp = self.model(self.t_test_t).double().numpy().ravel()
        e = rel_l2(yp, self.y_ref)
        L, _ = self.loss_fn()
        self.n_evals -= 1          # do not count monitoring evaluations
        self.curve["evals"].append(self.n_evals)
        self.curve["rel_l2"].append(e)
        self.curve["loss"].append(float(L.detach()))
        return {"rel_l2": e}

    def predict(self):
        with torch.no_grad():
            return self.model(self.t_test_t).double().numpy().ravel()


def run_precision(ex, dtype, n_adam, n_short, n_lbfgs, M, t_test, y_ref):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    out = {}
    try:
        common = dict(lr=1e-3, lr_final=1e-5, eval_every=50, print_every=0, verbose=False)
        # (1) Adam, then (2) the same model continued by L-BFGS
        pb = Problem(ex.seed, dtype, M, t_test, y_ref)
        with CPUTimer() as tm:
            h = train(pb.net.parameters(), pb.loss_fn, adam_iters=n_adam, lbfgs_iters=0,
                      eval_fn=pb.eval_fn, **common)
        y = pb.predict()
        out["adam"] = {"pred": y, "curve": {k: list(v) for k, v in pb.curve.items()}, "cpu": tm.cpu,
                       "wall": tm.wall, "iters": n_adam, "evals": pb.n_evals, "loss": h.final_loss}
        with CPUTimer() as tm2:
            h2 = train(pb.net.parameters(), pb.loss_fn, adam_iters=0, lbfgs_iters=n_lbfgs,
                       eval_fn=pb.eval_fn, **common)
        out["adam+lbfgs"] = {"pred": pb.predict(), "curve": {k: list(v) for k, v in pb.curve.items()},
                             "cpu": tm.cpu + tm2.cpu, "wall": tm.wall + tm2.wall,
                             "iters": n_adam + (h2.iters[-1] if h2.iters else 0), "evals": pb.n_evals,
                             "loss": h2.final_loss}
        # (3) short Adam + L-BFGS
        pb = Problem(ex.seed, dtype, M, t_test, y_ref)
        with CPUTimer() as tm:
            train(pb.net.parameters(), pb.loss_fn, adam_iters=n_short, lbfgs_iters=0, eval_fn=pb.eval_fn,
                  **common)
            h3 = train(pb.net.parameters(), pb.loss_fn, adam_iters=0, lbfgs_iters=n_lbfgs,
                       eval_fn=pb.eval_fn, **common)
        out["short+lbfgs"] = {"pred": pb.predict(), "curve": pb.curve, "cpu": tm.cpu, "wall": tm.wall,
                              "iters": n_short + (h3.iters[-1] if h3.iters else 0), "evals": pb.n_evals,
                              "loss": h3.final_loss}
        # (4) L-BFGS only
        pb = Problem(ex.seed, dtype, M, t_test, y_ref)
        pb.eval_fn()
        with CPUTimer() as tm:
            h4 = train(pb.net.parameters(), pb.loss_fn, adam_iters=0, lbfgs_iters=n_lbfgs,
                       eval_fn=pb.eval_fn, **common)
        out["lbfgs"] = {"pred": pb.predict(), "curve": pb.curve, "cpu": tm.cpu, "wall": tm.wall,
                        "iters": (h4.iters[-1] if h4.iters else 0), "evals": pb.n_evals, "loss": h4.final_loss}
    finally:
        torch.set_default_dtype(prev)
    return out


def main():
    ex = Example("ch09_training_diagnostics", "optimizers",
                 description="Adam vs Adam->L-BFGS vs L-BFGS, float64 vs float32 (nonlinear pendulum)",
                 default_seed=1234,
                 extra_args=[replot_arg()])
    M = ex.pick(200, 40)
    n_adam, n_short, n_lbfgs = ex.pick(10000, 60), ex.pick(1000, 20), ex.pick(1000, 15)
    t_test = np.linspace(0.0, T, 1001)
    y_ref = reference(t_test)
    results, rows = {}, []
    cache = RunCache(ex)
    for name, dtype in (("float64", torch.float64), ("float32", torch.float32)):
        ex.log(f"precision {name}")
        res = cache.get(name, lambda dtype=dtype: run_precision(ex, dtype, n_adam, n_short, n_lbfgs, M,
                                                                t_test, y_ref))
        results[name] = res
        for s in SCHEDULES:
            r = res[s]
            rows.append({"precision": name, "schedule": s, "iterations": int(r["iters"]),
                         "evaluations": int(r["evals"]), "final_loss": float(r["loss"]),
                         "rel_l2": rel_l2(r["pred"], y_ref), "linf": linf(r["pred"], y_ref),
                         "cpu_time_s": r["cpu"], "train_time_s": r["wall"]})
            ex.log(f"  {s:12s}: relL2={rows[-1]['rel_l2']:.2e} loss={r['loss']:.2e} iters={r['iters']} "
                   f"evals={r['evals']} cpu={r['cpu']:.1f}s")
            ex.save_table(r["curve"], f"curve_{name}_{s.replace('+', '_')}")
    base = [r for r in rows if r["precision"] == "float64" and r["schedule"] == "adam"][0]["cpu_time_s"]
    for r in rows:
        r["cpu_time_relative"] = r["cpu_time_s"] / base
    ex.save_table(rows, "summary")
    ex.save_results({
        "problem": {"ode": "theta'' + sin(theta) = 0, theta(0)=1, theta'(0)=0", "T": T,
                    "reference": "DOP853 rtol=atol=1e-12"},
        "hyperparameters": {"width": 32, "depth": 3, "collocation": M, "adam_iters": n_adam,
                            "short_adam_iters": n_short, "lbfgs_max_iters": n_lbfgs,
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs": "strong Wolfe, history 50, "
                            "stop if relative decrease < 1e-4 over 50 iterations"},
        "summary": rows})
    cache.clear()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from _make_tables import table_optimizers  # noqa: E402
    table_optimizers(ex.outdir, ex.outdir)

    if ex.args.no_figures:
        return
    fig, axes = P.figure(2, 2, aspect=0.74, sharex="col")
    for j, name in enumerate(("float64", "float32")):
        for k, s in enumerate(SCHEDULES):
            c = results[name][s]["curve"]
            ev = np.maximum(np.asarray(c["evals"], float), 1.0)
            axes[0, j].loglog(ev, c["rel_l2"], color=P.SERIES[k], ls="-" if k == 0 else P.LINESTYLES[k - 1],
                              lw=1.2, label=LABELS[s])
            axes[1, j].loglog(ev, c["loss"], color=P.SERIES[k], ls="-" if k == 0 else P.LINESTYLES[k - 1],
                              lw=1.2, label=LABELS[s])
        axes[0, j].set_title(name)
        axes[0, j].set_ylabel(r"relative $L^2$ error")
        axes[1, j].set_ylabel("loss")
        axes[1, j].set_xlabel("loss and gradient evaluations")
    ymin = min(min(results[n][s]["curve"]["rel_l2"]) for n in results for s in SCHEDULES)
    ymax = max(max(results[n][s]["curve"]["rel_l2"]) for n in results for s in SCHEDULES)
    for j in range(2):
        axes[0, j].set_ylim(0.5 * ymin, 2 * ymax)
    lmin = min(min(results[n][s]["curve"]["loss"]) for n in results for s in SCHEDULES)
    lmax = max(max(results[n][s]["curve"]["loss"]) for n in results for s in SCHEDULES)
    for j in range(2):
        axes[1, j].set_ylim(0.5 * lmin, 2 * lmax)
    axes[0, 0].legend(loc="lower left", fontsize=6.5)
    P.label_panels(axes)
    ex.savefig(fig, "optimizers_summary")


if __name__ == "__main__":
    main()
