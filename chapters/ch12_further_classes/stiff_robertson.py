"""Robertson's stiff chemical kinetics on [0, 40]: log-time PINN versus Radau and BDF.

Book: Chapter 12 (Further equation classes), Section "Stiff systems".

    y1' = -k1 y1 + k3 y2 y3,
    y2' =  k1 y1 - k3 y2 y3 - k2 y2^2,     k1 = 0.04, k2 = 3e7, k3 = 1e4,
    y3' =  k2 y2^2,                        y(0) = (1, 0, 0).

y2 stays below 3.7e-5 and reaches its quasi-steady state within t ~ 1e-3, while y1 and y3 evolve
on t ~ 10.  The script records the Jacobian eigenvalues along the solution: besides the zero
eigenvalue of the conservation law, a fast one of order -1e3 and a slow one of order -1e-1 to -1e-2.

Reference: Radau (rtol = 1e-12, atol = (1e-14, 1e-20, 1e-14), exact Jacobian), checked against BDF.
The test grid has 1001 points uniform in the log-time variable s defined below.

Two PINNs with the same network (width 32, depth 3, two outputs) and budget:

* vanilla  - physical time t, 256 uniform collocation points in [0, 40], residuals in t;
* log-time - the variable s = log(1 + t/tau)/S in [0, 1] (tau = 1e-4, S = log(1 + T/tau)), so that
             dt/ds = S (tau + t); y2 is scaled by Y2 = 4e-5 and the residuals are normalised:
                 R1 = (y1_s - S(tau+t) f1) / (1 + S(tau+t) k1),
                 R2 = (y2_s - S(tau+t) f2) / (Y2 (1 + S(tau+t) k1/Y2)).
Both use y1 = 1 + s N1, y2 = Y2 s N2 (y2 = Y2 t N2 for vanilla) and y3 = 1 - y1 - y2, which enforces
the conservation law y1 + y2 + y3 = 1 exactly (the index-1 DAE form of the problem).

Run:  python chapters/ch12_further_classes/stiff_robertson.py          (about 1-4 min on 1 CPU core)
      python chapters/ch12_further_classes/stiff_robertson.py --quick  (smoke test)
"""
from __future__ import annotations

import time

import numpy as np
import torch
from scipy.integrate import solve_ivp

from pinnode import Example, MLP, derivative, rel_l2, to_tensor, train
from pinnode import plotting as P

K1, K2, K3 = 0.04, 3.0e7, 1.0e4
T_END, TAU, Y2S = 40.0, 1.0e-4, 4.0e-5
S_LOG = float(np.log1p(T_END / TAU))
TOLS = [1e-4, 1e-6, 1e-8, 1e-10]


def rhs(t, y):
    y1, y2, y3 = y
    return [-K1 * y1 + K3 * y2 * y3, K1 * y1 - K3 * y2 * y3 - K2 * y2 ** 2, K2 * y2 ** 2]


def jac(t, y):
    y1, y2, y3 = y
    return [[-K1, K3 * y3, K3 * y2], [K1, -K3 * y3 - 2 * K2 * y2, -K3 * y2], [0.0, 2 * K2 * y2, 0.0]]


def t_of_s(s):
    return np.minimum(TAU * np.expm1(S_LOG * np.asarray(s, float)), T_END)


def classical(method, rtol, t_eval, repeats=3):
    """Solve with scipy (exact Jacobian); return solution at t_eval, best-of-repeats time, nfev."""
    atol = rtol * np.array([1e-4, 1e-9, 1e-4])
    best, sol = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        sol = solve_ivp(rhs, (0.0, T_END), [1.0, 0.0, 0.0], method=method, rtol=rtol, atol=atol,
                        jac=jac, t_eval=t_eval)
        best = min(best, time.perf_counter() - t0)
    return sol.y.T, best, int(sol.nfev), int(sol.njev)


def build(log_time: bool):
    net = MLP(1, 2, width=32, depth=3, activation="tanh", domain=(0.0, 1.0) if log_time else (0.0, T_END))

    def fields(x):
        n = net(x)
        y1 = 1.0 + x * n[:, :1]
        y2 = Y2S * x * n[:, 1:2]
        return y1, y2, 1.0 - y1 - y2               # conservation law built in
    return net, fields


def train_pinn(ex: Example, log_time: bool, s_test, y_ref):
    torch.manual_seed(ex.seed)
    net, fields = build(log_time)
    m = ex.pick(256, 32)
    x_col = to_tensor(np.linspace(0.0, 1.0 if log_time else T_END, m), requires_grad=True)
    x_test = to_tensor(s_test if log_time else t_of_s(s_test))

    def loss_fn():
        x = x_col
        y1, y2, y3 = fields(x)
        f1 = -K1 * y1 + K3 * y2 * y3
        f2 = K1 * y1 - K3 * y2 * y3 - K2 * y2 ** 2
        if log_time:
            t = TAU * torch.expm1(S_LOG * x)
            dtds = S_LOG * (TAU + t)
            r1 = (derivative(y1, x) - dtds * f1) / (1.0 + dtds * K1)
            r2 = (derivative(y2, x) - dtds * f2) / (Y2S * (1.0 + dtds * K1 / Y2S))
        else:
            r1 = derivative(y1, x) - f1
            r2 = (derivative(y2, x) - f2) / Y2S
        L1, L2 = (r1 ** 2).mean(), (r2 ** 2).mean()
        return L1 + L2, {"r1": L1, "r2": L2}

    def eval_fn():
        with torch.no_grad():
            y1, y2, y3 = fields(x_test)
        return {"e1": rel_l2(y1, y_ref[:, 0]), "e2": rel_l2(y2, y_ref[:, 1]), "e3": rel_l2(y3, y_ref[:, 2])}

    hist = train(net.parameters(), loss_fn, adam_iters=ex.pick(8000, 60), lr=1e-3, lr_final=1e-5,
                 lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn, eval_every=100, print_every=2000)
    with torch.no_grad():
        y = torch.cat(fields(x_test), dim=1).numpy()
    return y, hist


def errors(y, y_ref):
    return [rel_l2(y[:, j], y_ref[:, j]) for j in range(3)]


def main():
    ex = Example("ch12_further_classes", "stiff_robertson",
                 description="Robertson kinetics on [0, 40]: log-time PINN vs vanilla PINN vs Radau/BDF",
                 default_seed=1234)
    s_test = np.linspace(0.0, 1.0, 1001)
    t_test = t_of_s(s_test)
    sol = solve_ivp(rhs, (0.0, T_END), [1.0, 0.0, 0.0], method="Radau", rtol=1e-12,
                    atol=[1e-14, 1e-20, 1e-14], jac=jac, t_eval=t_test)
    y_ref = sol.y.T
    chk = solve_ivp(rhs, (0.0, T_END), [1.0, 0.0, 0.0], method="BDF", rtol=1e-12,
                    atol=[1e-14, 1e-20, 1e-14], jac=jac, t_eval=t_test).y.T
    ref_check = {"radau_vs_bdf_rel_l2": errors(chk, y_ref), "y_T_reference": y_ref[-1].tolist()}
    ex.log(f"reference check: {ref_check}")
    # Jacobian eigenvalues along the reference solution (stiffness diagnostics)
    dense = solve_ivp(rhs, (0.0, T_END), [1.0, 0.0, 0.0], method="Radau", rtol=1e-12,
                      atol=[1e-14, 1e-20, 1e-14], jac=jac, dense_output=True)
    eig = {}
    for tt in (1e-2, 1e-1, 1.0, 10.0, 40.0):
        ev = np.sort(np.linalg.eigvals(np.array(jac(tt, dense.sol(tt)))).real)
        eig[f"{tt:g}"] = ev.tolist()
    ex.log(f"Jacobian eigenvalues: {eig}")

    rows = []
    for method in ("Radau", "BDF"):
        for rtol in TOLS:
            y, secs, nfev, njev = classical(method, rtol, t_test, repeats=ex.pick(3, 1))
            e = errors(y, y_ref)
            rows.append({"method": method, "setting": f"rtol={rtol:.0e}", "rtol": rtol,
                         "rel_l2_y1": e[0], "rel_l2_y2": e[1], "rel_l2_y3": e[2], "time_s": secs,
                         "nfev": nfev, "njev": njev})
            ex.log(f"{method:5s} rtol={rtol:.0e}: errors {e[0]:.1e} {e[1]:.1e} {e[2]:.1e}  "
                   f"time={secs:.3f}s nfev={nfev}")
    pinn = {}
    for tag, log_time in (("log-time", True), ("vanilla", False)):
        ex.log(f"training PINN ({tag})")
        y, hist = train_pinn(ex, log_time, s_test, y_ref)
        pinn[tag] = {"y": y, "hist": hist}
        ex.save_history(hist, tag.replace("-", "_"))
        e = errors(y, y_ref)
        rows.append({"method": "PINN", "setting": tag, "rtol": None, "rel_l2_y1": e[0],
                     "rel_l2_y2": e[1], "rel_l2_y3": e[2], "time_s": hist.train_time,
                     "nfev": None, "njev": None, "final_loss": hist.final_loss})
        ex.log(f"PINN {tag}: errors {e[0]:.2e} {e[1]:.2e} {e[2]:.2e} loss={hist.final_loss:.2e} "
               f"time={hist.train_time:.1f}s")
    ex.save_table(rows, "summary")
    ex.save_table({"s": s_test, "t": t_test, **{f"ref_y{j + 1}": y_ref[:, j] for j in range(3)},
                   **{f"logtime_y{j + 1}": pinn["log-time"]["y"][:, j] for j in range(3)},
                   **{f"vanilla_y{j + 1}": pinn["vanilla"]["y"][:, j] for j in range(3)}}, "predictions")
    ex.save_results({
        "problem": {"k": [K1, K2, K3], "T": T_END, "y0": [1, 0, 0]},
        "reference": "Radau rtol=1e-12 atol=(1e-14,1e-20,1e-14), exact Jacobian",
        "reference_check": ref_check,
        "jacobian_eigenvalues": eig,
        "hyperparameters": {"width": 32, "depth": 3, "activation": "tanh", "outputs": 2,
                            "collocation": ex.pick(256, 32), "adam_iters": ex.pick(8000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10),
                            "tau": TAU, "S": S_LOG, "Y2_scale": Y2S,
                            "classical_atol": "rtol * (1e-4, 1e-9, 1e-4)"},
        "summary": rows,
        "y_T_pinn_logtime": pinn["log-time"]["y"][-1].tolist(),
        "train_time_s": {k: v["hist"].train_time for k, v in pinn.items()},
        "test_grid": "1001 points uniform in s = log(1 + t/tau)/S on [0, 1]",
    })
    if ex.args.no_figures:
        return

    # ------------------------------------------------------------------ figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    tp = np.maximum(t_test, 1e-7)
    sl = slice(1, None)                                  # skip t = 0 on the log axis
    ax = axes[0, 0]
    for j, name in ((0, "$y_1$"), (2, "$y_3$")):
        P.plot_reference(ax, tp[sl], y_ref[sl, j], label="Radau" if j == 0 else None)
        P.plot_prediction(ax, tp[sl], pinn["log-time"]["y"][sl, j], k=0, label="PINN, log-time" if j == 0 else None)
        P.plot_prediction(ax, tp[sl], pinn["vanilla"]["y"][sl, j], k=1, label="PINN, vanilla" if j == 0 else None)
    ax.text(2e-6, 0.88, "$y_1$", fontsize=8)
    ax.text(2e-6, 0.08, "$y_3$", fontsize=8)
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.1)                   # the vanilla PINN leaves [0, 1] at late times
    ax.set_xlabel("$t$")
    ax.set_ylabel("concentration")
    ax.legend(loc="center left")

    ax = axes[0, 1]
    P.plot_reference(ax, tp[sl], 1e5 * y_ref[sl, 1], label="Radau")
    P.plot_prediction(ax, tp[sl], 1e5 * pinn["log-time"]["y"][sl, 1], k=0, label="PINN, log-time")
    P.plot_prediction(ax, tp[sl], 1e5 * pinn["vanilla"]["y"][sl, 1], k=1, label="PINN, vanilla")
    ax.set_xscale("log")
    ax.set_ylim(-1.0, 4.5)
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$10^5\,y_2$")
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    for j in range(3):
        e = np.abs(pinn["log-time"]["y"][sl, j] - y_ref[sl, j]) / np.max(np.abs(y_ref[:, j]))
        ax.loglog(tp[sl], np.maximum(e, 1e-12), color=P.SERIES[j], lw=1.0,
                  ls="-" if j == 0 else P.LINESTYLES[j - 1], label=f"$y_{j + 1}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|y_i-y_i^{\mathrm{ref}}|/\max|y_i^{\mathrm{ref}}|$")
    ax.set_title("log-time PINN")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    for k, tag in enumerate(("log-time", "vanilla")):
        h = pinn[tag]["hist"]
        ax.semilogy(h.iters, P.rolling_median(h.loss, 9), color=P.SERIES[k],
                    ls="-" if k == 0 else P.LINESTYLES[0], lw=1.1, label=f"PINN, {tag}")
    ax.axvline(pinn["log-time"]["hist"].switch_iter, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(loc="center right")
    P.label_panels(axes)
    ex.savefig(fig, "robertson_summary")


if __name__ == "__main__":
    main()
