"""Sturm-Liouville eigenvalue problems: the first three eigenpairs by PINNs with deflation.

Book: Chapter 12 (Further equation classes), Section "Sturm-Liouville eigenvalue problems".

    -(p(x) y')' + q(x) y = lambda w(x) y,   x in (0, pi),   y(0) = y(pi) = 0.

Problem A: p = w = 1, q = 0  ->  lambda_k = k^2, y_k = sqrt(2/pi) sin(kx).
Problem B: p = 1 + x/pi, q = sin x, w = 2 - x/pi  ->  reference from a Chebyshev collocation
           discretisation (generalised eigenproblem, N = 64, checked against N = 96).

PINN for the k-th eigenpair (k = 1, 2, 3), trial function y = x (pi - x) N(x):
 1. selection: minimise the Rayleigh quotient int (p y'^2 + q y^2) / int w y^2 plus a penalty on
    the w-overlaps with the eigenfunctions already found (min-max principle: this picks the
    k-th eigenpair);
 2. refinement: lambda becomes a trainable parameter (initialised with the Rayleigh quotient) and
    the loss is  mean R^2 / m + (m - 1)^2 + sum_j (int w y y_j)^2 / m  with m = int w y^2 (residual,
    normalisation and orthogonality/deflation penalties; dividing by m makes the residual and the
    overlaps scale invariant, so that the trivial solution y = 0 cannot attract the iteration),
    minimised by Adam and L-BFGS.
Integrals use 128-point Gauss-Legendre quadrature, whose nodes are also the collocation points.

Run:  python chapters/ch12_further_classes/sturm_liouville.py          (about 4-8 min on 1 CPU core)
      python chapters/ch12_further_classes/sturm_liouville.py --quick  (smoke test)
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.interpolate import BarycentricInterpolator
from scipy.linalg import eig

from pinnode import ConstrainedModel, Example, MLP, derivative, gauss_legendre, rel_l2, to_tensor, train
from pinnode import plotting as P

PI = math.pi
N_EIG = 3
ORTH_WEIGHT_RQ = 50.0     # must exceed lambda_k - lambda_1 so that the penalised minimiser is orthogonal

PROBLEMS = {
    "A": {"p": lambda x: np.ones_like(x), "dp": lambda x: np.zeros_like(x),
          "q": lambda x: np.zeros_like(x), "w": lambda x: np.ones_like(x),
          "label": r"$-y''=\lambda y$"},
    "B": {"p": lambda x: 1.0 + x / PI, "dp": lambda x: np.full_like(x, 1.0 / PI),
          "q": lambda x: np.sin(x), "w": lambda x: 2.0 - x / PI,
          "label": r"$-((1+x/\pi)y')'+(\sin x)\,y=\lambda(2-x/\pi)y$"},
}


# ----------------------------------------------------------------------------- reference
def cheb(n):
    """Chebyshev differentiation matrix and points on [-1, 1] (Trefethen, Spectral Methods in MATLAB)."""
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.hstack([2.0, np.ones(n - 1), 2.0]) * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    return D - np.diag(D.sum(axis=1)), x


def chebyshev_reference(prob, n=64, k=N_EIG):
    """First k eigenvalues and eigenfunction interpolants (normalised: int w y^2 = 1, y'(0) > 0)."""
    D, xi = cheb(n)
    x = 0.5 * PI * (xi + 1.0)
    D1 = D * (2.0 / PI)
    D2 = D1 @ D1
    Lmat = -np.diag(prob["p"](x)) @ D2 - np.diag(prob["dp"](x)) @ D1 + np.diag(prob["q"](x))
    B = np.diag(prob["w"](x))
    lam, V = eig(Lmat[1:-1, 1:-1], B[1:-1, 1:-1])
    lam = lam.real
    order = np.argsort(lam)
    xg, wg = gauss_legendre(200, 0.0, PI)
    funcs = []
    for j in order[:k]:
        v = np.hstack([0.0, V[:, j].real, 0.0])
        f = BarycentricInterpolator(x, v)
        nrm = math.sqrt(np.sum(wg * prob["w"](xg) * f(xg) ** 2))
        sgn = np.sign(f(np.array([1e-3]))[0])
        funcs.append(lambda s, f=f, c=sgn / nrm: c * f(s))
    return lam[order[:k]], funcs


# ----------------------------------------------------------------------------- PINN
def train_eigenpair(ex: Example, prob, found, k_index, x_test, ref_fun, lam_ref):
    """k-th eigenpair; ``found`` holds the eigenfunctions already computed (values at the nodes)."""
    torch.manual_seed(ex.seed + k_index)
    net = MLP(1, 1, width=32, depth=3, activation="tanh", domain=(0.0, PI))
    model = ConstrainedModel(net, lambda x, n: x * (PI - x) * n)
    xq_np, wq_np = gauss_legendre(ex.pick(128, 32), 0.0, PI)
    xq = to_tensor(xq_np, requires_grad=True)
    wq = to_tensor(wq_np)
    p, dp, q, w = (to_tensor(prob[c](xq_np)) for c in ("p", "dp", "q", "w"))
    prev = [to_tensor(v) for v in found]                 # previous eigenfunctions at the nodes
    x_test_t = to_tensor(x_test)
    y_ref = ref_fun(x_test)

    def overlaps(y):
        return [torch.sum(wq * w * y * yj) for yj in prev]

    # ---------------- phase 1: Rayleigh quotient with deflation (selects the k-th eigenpair)
    def loss_rq():
        y = model(xq)
        dy = derivative(y, xq)
        mass = torch.sum(wq * w * y ** 2)
        rq = torch.sum(wq * (p * dy ** 2 + q * y ** 2)) / mass
        orth = sum((o ** 2 for o in overlaps(y)), torch.zeros(())) / mass
        return rq + ORTH_WEIGHT_RQ * orth, {"rayleigh": rq, "orth": orth}

    h1 = train(net.parameters(), loss_rq, adam_iters=ex.pick(1500, 30), lr=1e-3, lr_final=1e-4,
               lbfgs_iters=0, print_every=0, verbose=False)
    lam0 = float(h1.parts["rayleigh"][-1])

    # ---------------- phase 2: residual with trainable lambda + normalisation + orthogonality
    lam = torch.nn.Parameter(torch.tensor(lam0))

    def loss_res():
        y = model(xq)
        dy = derivative(y, xq)
        d2y = derivative(dy, xq)
        r = -(p * d2y + dp * dy) + q * y - lam * w * y
        mass = torch.sum(wq * w * y ** 2)
        # residual and overlaps relative to the size of y: scale invariant, so that y = 0 (a
        # stationary point of the normalisation penalty) cannot attract the iteration
        L_r = (r ** 2).mean() / mass
        L_n = (mass - 1.0) ** 2
        L_o = sum((o ** 2 for o in overlaps(y)), torch.zeros(())) / mass
        return L_r + L_n + L_o, {"residual": L_r, "normalisation": L_n, "orthogonality": L_o,
                                 "lambda": lam.detach().clone()}

    def eval_fn():
        return {"lambda_rel_err": abs(lam.item() - lam_ref) / lam_ref}

    h2 = train(list(net.parameters()) + [lam], loss_res, adam_iters=ex.pick(4000, 40), lr=1e-3,
               lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=50,
               print_every=2000)
    with torch.no_grad():                      # normalise (int w y^2 = 1) and fix the sign
        yq = model(to_tensor(xq_np)).numpy().ravel()
        nrm = math.sqrt(float(np.sum(wq_np * prob["w"](xq_np) * yq ** 2)))
        y_pred = model(x_test_t).numpy().ravel() / nrm
    sgn = 1.0 if np.dot(y_pred, y_ref) >= 0 else -1.0
    y_pred = sgn * y_pred
    return {"lambda": float(lam.item()), "lambda_rayleigh_phase1": lam0, "y_pred": y_pred,
            "y_ref": y_ref, "rel_l2": rel_l2(y_pred, y_ref), "hist1": h1, "hist2": h2,
            "train_time_s": h1.train_time + h2.train_time, "node_values": sgn * yq / nrm}


def main():
    ex = Example("ch12_further_classes", "sturm_liouville",
                 description="Sturm-Liouville eigenpairs: Rayleigh-quotient selection + residual with "
                             "trainable eigenvalue and deflation", default_seed=1234)
    x_test = np.linspace(0.0, PI, 1001)
    rows, runs, ref_info = [], {}, {}
    for name, prob in PROBLEMS.items():
        lam64, funcs64 = chebyshev_reference(prob, 64)
        lam96, _ = chebyshev_reference(prob, 96)
        ref_info[name] = {"lambda_N64": lam64.tolist(), "lambda_N96": lam96.tolist(),
                          "max_diff_N64_N96": float(np.max(np.abs(lam64 - lam96)))}
        if name == "A":
            exact_l = np.array([1.0, 4.0, 9.0])
            ref_info[name]["max_diff_cheb_vs_exact"] = float(np.max(np.abs(lam64 - exact_l)))
            lam_ref = exact_l
            funcs = [lambda s, k=k: math.sqrt(2.0 / PI) * np.sin(k * s) for k in (1, 2, 3)]
        else:
            lam_ref, funcs = lam64, funcs64
        ex.log(f"problem {name}: reference eigenvalues {lam_ref} (N64-N96 diff "
               f"{ref_info[name]['max_diff_N64_N96']:.1e})")
        found = []
        for k in range(N_EIG):
            ex.log(f"problem {name}: eigenpair {k + 1}")
            r = train_eigenpair(ex, prob, found, k, x_test, funcs[k], float(lam_ref[k]))
            found.append(r["node_values"])             # deflation uses the PINN eigenfunction
            runs[(name, k)] = r
            ex.save_history(r["hist2"], f"{name}_k{k + 1}")
            rows.append({"problem": name, "k": k + 1, "lambda_ref": float(lam_ref[k]),
                         "lambda_pinn": r["lambda"], "lambda_rayleigh_phase1": r["lambda_rayleigh_phase1"],
                         "lambda_rel_err": abs(r["lambda"] - lam_ref[k]) / lam_ref[k],
                         "eigfun_rel_l2": r["rel_l2"], "train_time_s": r["train_time_s"]})
            ex.log(f"  lambda={r['lambda']:.10f} (ref {lam_ref[k]:.10f}) rel err "
                   f"{rows[-1]['lambda_rel_err']:.1e}  eigfun relL2={r['rel_l2']:.1e} "
                   f"time={r['train_time_s']:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_results({
        "problem": {"A": "p=w=1, q=0 on (0,pi), Dirichlet", "B": "p=1+x/pi, q=sin x, w=2-x/pi on (0,pi)",
                    "eigenpairs": N_EIG},
        "reference": ref_info,
        "hyperparameters": {"width": 32, "depth": 3, "activation": "tanh",
                            "quadrature_nodes": ex.pick(128, 32), "phase1_adam": ex.pick(1500, 30),
                            "phase1_orth_weight": ORTH_WEIGHT_RQ, "phase2_adam": ex.pick(4000, 40),
                            "phase2_lbfgs": ex.pick(1000, 10), "lr": "1e-3 -> 1e-5 (cosine)",
                            "loss": "mean(R^2)/m + (m-1)^2 + sum_j overlap_j^2/m, m = int w y^2"},
        "summary": rows,
        "train_time_s": {f"{r['problem']}_k{r['k']}": r["train_time_s"] for r in rows},
        "test_grid": "1001 uniform points on [0, pi]",
    })
    if ex.args.no_figures:
        return

    fig, axes = P.figure(2, 2, aspect=0.72)
    for ax, name in zip(axes[0], PROBLEMS):
        for k in range(N_EIG):
            r = runs[(name, k)]
            P.plot_reference(ax, x_test, r["y_ref"], label="reference" if k == 0 else None)
            P.plot_prediction(ax, x_test, r["y_pred"], k=k, markevery=(30 * k, 100),
                              label=f"$k={k + 1}$, $\\lambda={r['lambda']:.4f}$")
        ymax = max(np.max(np.abs(runs[(name, k)]["y_ref"])) for k in range(N_EIG))
        ax.set_ylim(-1.75 * ymax, 1.15 * ymax)           # room for the legend below the curves
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y_k(x)$")
        ax.set_title(f"problem {name}")
        ax.legend(loc="lower center", ncol=2, fontsize=6.5)
    ax = axes[1, 0]
    for name, ls_off in (("A", 0), ("B", 3)):
        for k in range(N_EIG):
            r = runs[(name, k)]
            e = np.maximum(np.abs(r["y_pred"] - r["y_ref"]), 1e-13)[1:-1]   # exact at x = 0, pi
            ax.semilogy(x_test[1:-1], e, color=P.SERIES[k], lw=1.0,
                        ls="-" if name == "A" else P.LINESTYLES[k], label=f"{name}, $k={k + 1}$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("eigenfunction error")
    ax = axes[1, 1]
    for name in PROBLEMS:
        for k in range(N_EIG):
            h = runs[(name, k)]["hist2"]
            ax.semilogy(h.eval_iters, np.maximum(h.metrics["lambda_rel_err"], 1e-16), color=P.SERIES[k],
                        lw=1.0, ls="-" if name == "A" else P.LINESTYLES[k], label=f"{name}, $k={k + 1}$")
    ax.axvline(runs[("A", 0)]["hist2"].switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration (refinement phase)")
    ax.set_ylabel(r"$|\lambda_\theta-\lambda_k|/\lambda_k$")
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=6.5)
    fig.get_layout_engine().set(h_pad=0.06)
    P.label_panels(axes)
    ex.savefig(fig, "sturm_liouville_summary")


if __name__ == "__main__":
    main()
