"""Helmholtz-type integral equation with an oscillatory kernel (Filon panel quadrature, SIREN):

    u(x) = 1 + lam (i / (2k)) int_0^1 e^{ik|x - t|} u(t) dt,   x in [0, 1],   k = 10, lam = 6.

Book: Chapter 7 (Integral equations), Section "Oscillatory kernels: a Helmholtz-type equation"
(figure ie:fig:helmholtz, table ie:tab:helmholtz).

Exact solution (from the equivalent two-point boundary-value problem, see the book):
    u(x) = k^2/mu^2 + (i lam k / mu^2) cos(mu (x - 1/2)) / (mu sin(mu/2) + i k cos(mu/2)),
    mu = sqrt(k^2 + lam).

PINN: SIREN (3 hidden layers of 48 neurons, first-layer frequency w0 = 8, inputs scaled to
[-1, 1]) with two real outputs (Re u, Im u).  The integral is evaluated with the quadratic
Filon panel rule on P = 20 uniform panels (41 nodes), split at t = x, so that the
residual at an arbitrary collocation point x_i is
    R_i = u(x_i) - 1 - lam (i / (2k)) sum_j W_ij u(t_j),
with the precomputed complex matrix W (module _iecore).  Collocation: M = 201 uniform points
(different from the quadrature nodes).  Loss: mean of |R_i|^2.

Classical comparison (accuracy and time, best of 7 repetitions, including evaluation of the
Nystrom interpolant on the test grid):
  * Nystrom method with the global n-point Gauss-Legendre rule (pinnode.reference), which
    converges only like n^{-2} because the kernel has a derivative jump at t = x;
  * product-integration Nystrom method with the same Filon panel rule as the PINN, and with
    the 'Simpson' variant that interpolates the whole integrand (phase=False).

A posteriori check: |lam| sup_x int_0^1 |G(x, t)| dt = |lam| / (2k) = 0.3 < 1, hence
||u_theta - u||_inf <= ||r||_inf / (1 - 0.3), with the continuous residual r evaluated on the
test grid by a fine Filon rule (P = 400).

Errors are measured on the uniform grid of 1001 points in [0, 1] (complex modulus).

Run:  PYTHONPATH=. python chapters/ch07_integral_equations/helmholtz_filon.py   (a few minutes, 1 core)
      PYTHONPATH=. python chapters/ch07_integral_equations/helmholtz_filon.py --quick
      PYTHONPATH=. python chapters/ch07_integral_equations/helmholtz_filon.py --replot  (figure only)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _iecore import (NYSTROM, REPLOT_ARG, best_time, filon_matrix, filon_nystrom,  # noqa: E402
                     helmholtz_exact, history_arrays, history_from_arrays)
from pinnode import Example, SirenNet, linf, rel_l2, to_tensor, train  # noqa: E402
from pinnode import plotting as P  # noqa: E402
from pinnode.reference import nystrom_fredholm  # noqa: E402

K_WAVE, LAM = 10.0, 6.0
P_PANELS, M_COL = 20, 201
WIDTH, DEPTH, W0 = 48, 3, 8.0
P_FINE = 400


def complex_from(out: torch.Tensor):
    return out[:, 0], out[:, 1]


def run_pinn(ex: Example, x_test: np.ndarray):
    torch.manual_seed(ex.seed)
    model = SirenNet(1, 2, width=WIDTH, depth=DEPTH, w0_first=W0, w0=1.0, domain=(0.0, 1.0))
    P_ = ex.pick(P_PANELS, 10)
    x_col = np.linspace(0.0, 1.0, ex.pick(M_COL, 41))
    W, nodes = filon_matrix(x_col, P_, K_WAVE)
    G = LAM * (1j / (2.0 * K_WAVE)) * W                          # (M, 2P+1) complex
    Gr, Gi = torch.as_tensor(G.real), torch.as_tensor(G.imag)
    xc, tn = to_tensor(x_col), to_tensor(nodes)
    xt = to_tensor(x_test)
    u_ref = helmholtz_exact(x_test, K_WAVE, LAM)

    def loss_fn():
        ur, ui = complex_from(model(xc))
        vr, vi = complex_from(model(tn))
        Rr = ur - 1.0 - (Gr @ vr - Gi @ vi)
        Ri = ui - (Gr @ vi + Gi @ vr)
        L = (Rr ** 2 + Ri ** 2).mean()
        return L, {"residual": L}

    def predict(x):
        with torch.no_grad():
            out = model(x).numpy()
        return out[:, 0] + 1j * out[:, 1]

    def eval_fn():
        return {"rel_l2": rel_l2(predict(xt), u_ref)}

    n_adam = ex.pick(4000, 60)
    hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=25, print_every=1000)
    u = predict(xt)
    # continuous residual on the test grid with a fine Filon rule
    Pf = ex.pick(P_FINE, 40)
    Wf, nf = filon_matrix(x_test, Pf, K_WAVE)
    r = u - 1.0 - LAM * (1j / (2.0 * K_WAVE)) * (Wf @ predict(to_tensor(nf)))
    neumann = 1.0 / (1.0 - abs(LAM) / (2.0 * K_WAVE))
    # discrete estimate of ||(I - lam T)^{-1}||_inf from the fine Filon-Nystrom matrix
    Wn, _ = filon_matrix(nf, Pf, K_WAVE)
    inv_norm = float(np.max(np.abs(np.linalg.inv(np.eye(nf.size) - LAM * (1j / (2 * K_WAVE)) * Wn)).sum(axis=1)))
    e = u - u_ref
    res = {"rel_l2": rel_l2(u, u_ref), "linf": linf(u, u_ref),
           "linf_re": float(np.max(np.abs(e.real))), "linf_im": float(np.max(np.abs(e.imag))),
           "linf_abs": float(np.max(np.abs(np.abs(u) - np.abs(u_ref)))),
           "final_loss": hist.final_loss, "train_time_s": hist.train_time, "adam_iters": n_adam,
           "lbfgs_iters_done": int(hist.iters[-1] - n_adam),
           "panels": P_, "quadrature_nodes": int(nodes.size), "collocation": int(x_col.size),
           "residual_inf": float(np.max(np.abs(r))), "neumann_constant": neumann,
           "resolvent_norm_estimate": inv_norm,
           "error_bound": float(neumann * np.max(np.abs(r)))}
    # discretization floor of the PINN's quadrature: Filon-Nystrom with the same panels
    _, _, interp = filon_nystrom(P_, K_WAVE, LAM)
    res["filon_nystrom_same_panels_rel_l2"] = rel_l2(interp(x_test), u_ref)
    return res, hist, u


def classical_study(ex: Example, x_test: np.ndarray):
    rows = []
    u_ref = helmholtz_exact(x_test, K_WAVE, LAM)
    kern = lambda x, s: (1j / (2.0 * K_WAVE)) * np.exp(1j * K_WAVE * np.abs(x - s))   # noqa: E731
    g = lambda x: np.ones_like(np.asarray(x, float))                                    # noqa: E731
    rep = ex.pick(7, 1)
    for n in ex.pick((20, 40, 80, 160, 320, 640), (20, 40)):
        u, t = best_time(lambda: nystrom_fredholm(kern, g, LAM, 0.0, 1.0, n)(x_test), rep)
        rows.append({"method": "Nystrom, Gauss-Legendre", "param": n, "nodes": n,
                     "rel_l2": rel_l2(u, u_ref), "linf": linf(u, u_ref), "time_s": t})
    for phase, name in ((True, "Nystrom, Filon panels"), (False, "Nystrom, Simpson panels")):
        for Pn in ex.pick((5, 10, 20, 40, 80), (5, 10)):
            def solve():
                return filon_nystrom(Pn, K_WAVE, LAM, phase=phase)[2](x_test)
            u, t = best_time(solve, rep)
            rows.append({"method": name, "param": Pn, "nodes": 2 * Pn + 1,
                         "rel_l2": rel_l2(u, u_ref), "linf": linf(u, u_ref), "time_s": t})
    return rows


def make_figure(ex: Example, x_test, u, u_ref, hist, floor: float, panels: int) -> None:
    fig, axes = P.figure(3, 2, aspect=0.56)
    me = (100, 200)
    for ax, comp, name, loc in ((axes[0, 0], np.real, "$\\mathrm{Re}\\,u$", "lower center"),
                                (axes[0, 1], np.imag, "$\\mathrm{Im}\\,u$", "upper center"),
                                (axes[1, 0], np.abs, "$|u|$", "lower center")):
        P.plot_reference(ax, x_test, comp(u_ref), label="exact")
        P.plot_prediction(ax, x_test, comp(u), label="PINN", k=0, markevery=me)
        lo, hi = float(np.min(comp(u_ref))), float(np.max(comp(u_ref)))
        if loc == "lower center":
            ax.set_ylim(lo - 0.55 * (hi - lo), hi + 0.08 * (hi - lo))
        else:
            ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.55 * (hi - lo))
        ax.set_xlabel("$x$")
        ax.set_ylabel(name)
        ax.legend(loc=loc, ncol=2)
    axes[0, 0].set_title("real part")
    axes[0, 1].set_title("imaginary part")
    axes[1, 0].set_title("modulus")
    ax = axes[1, 1]
    e = u - u_ref
    P.plot_error(ax, x_test, e.real, label="$|\\mathrm{Re}(u_{\\theta}-u)|$", k=0, xlabel="$x$")
    P.plot_error(ax, x_test, e.imag, label="$|\\mathrm{Im}(u_{\\theta}-u)|$", k=1, xlabel="$x$")
    P.plot_error(ax, x_test, np.abs(u) - np.abs(u_ref), label="$||u_{\\theta}|-|u||$", k=2, xlabel="$x$")
    ax.set_ylim(1e-11, 1e-2)
    ax.set_title("pointwise errors")
    ax.legend(loc="lower center", ncol=2)
    ax = axes[2, 0]
    P.plot_history(ax, hist, labels={"loss": "mean $|R|^2$"})
    ax.set_title("training loss")
    ax = axes[2, 1]
    P.plot_metric(ax, hist, "rel_l2", label="PINN", k=0)
    ax.axhline(floor, color=P.COLORS["ref"], lw=0.8, ls=":", label=f"Filon-{NYSTROM}, $P={panels}$")
    ax.set_ylim(1e-7, 10.0)
    ax.set_title("relative $L^2$ error")
    ax.legend(loc="upper right")
    P.label_panels(axes)
    ex.savefig(fig, "helmholtz_summary")


def main():
    ex = Example("ch07_integral_equations", "helmholtz_filon",
                 description="1D Helmholtz-type IE (k=10, lam=6): SIREN PINN with Filon panels",
                 default_seed=1234, extra_args=[REPLOT_ARG])
    x_test = np.linspace(0.0, 1.0, 1001)
    u_ref = helmholtz_exact(x_test, K_WAVE, LAM)
    if ex.args.replot:
        d = np.load(ex.path("figdata.npz"))
        make_figure(ex, d["x"], d["u_re"] + 1j * d["u_im"], u_ref, history_from_arrays(d, "pinn"),
                    float(d["floor"]), int(d["panels"]))
        return
    res, hist, u = run_pinn(ex, x_test)
    ex.save_history(hist, "pinn")
    ex.log(f"PINN: relL2={res['rel_l2']:.3e} linf={res['linf']:.3e} loss={res['final_loss']:.2e} "
           f"time={res['train_time_s']:.1f}s | r_inf={res['residual_inf']:.2e} bound={res['error_bound']:.2e} "
           f"| resolvent~{res['resolvent_norm_estimate']:.3f} | Filon-Nystrom floor={res['filon_nystrom_same_panels_rel_l2']:.2e}")
    rows = classical_study(ex, x_test)
    for r in rows:
        ex.log(f"{r['method']:26s} param={r['param']:4d}: relL2={r['rel_l2']:.2e} time={r['time_s']:.2e}s")
    ex.save_table({"x": x_test, "re_exact": u_ref.real, "im_exact": u_ref.imag,
                   "re_pinn": u.real, "im_pinn": u.imag}, "predictions")
    ex.save_table(rows, "classical")
    ex.save_results({
        "problem": {"equation": "u = 1 + lam (i/2k) int_0^1 exp(ik|x-t|) u(t) dt", "k": K_WAVE,
                    "lam": LAM, "domain": [0, 1], "exact": "closed form (equivalent BVP)"},
        "hyperparameters": {"network": f"SIREN {DEPTH}x{WIDTH}, w0_first={W0}, 2 outputs",
                            "panels": ex.pick(P_PANELS, 10), "collocation": ex.pick(M_COL, 41),
                            "adam_iters": ex.pick(4000, 60), "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters_max": ex.pick(1000, 10), "test_grid": 1001},
        "pinn": res, "classical": rows, "train_time_s": res["train_time_s"]})
    np.savez(ex.path("figdata.npz"), x=x_test, u_re=u.real, u_im=u.imag,
             floor=res["filon_nystrom_same_panels_rel_l2"], panels=res["panels"], **history_arrays(hist, "pinn"))
    if not ex.args.no_figures:
        make_figure(ex, x_test, u, u_ref, hist, res["filon_nystrom_same_panels_rel_l2"], res["panels"])


if __name__ == "__main__":
    main()
