"""Conditioning of terminal value problems: backward amplification and its effect on a PINN.

Problem (two decay rates 1 and kappa):

    y'' + (1 + kappa) y' + kappa y = 0   on [0, T],

with exact solution y(t) = (kappa e^{-t} - e^{-kappa t}) / (kappa - 1) for y(0) = 1, y'(0) = 0.
The terminal value problem (TVP) prescribes y(T), y'(T) taken from this exact solution; the
initial value problem (IVP) prescribes y(0) = 1, y'(0) = 0. Both have the same solution.

Part 1 (classical).  For the first-order system z' = A z, A = [[0, 1], [-kappa, -(1 + kappa)]],
the terminal-to-initial map is z(0) = e^{-AT} z(T).  We tabulate the amplification factors
||e^{-AT}||_2 (backward) and ||e^{AT}||_2 (forward) for a range of kappa, confirm them by
integrating perturbed terminal data backward with DOP853 (rtol = atol = 1e-12), and record the
error of plain backward integration from exact terminal data.

Part 3 (examples).  The backward sensitivity matrix of the pendulum TVP of Section 6.3 is
obtained from its variational equation (norm and determinant; det = exp(beta T) by Liouville).

Part 2 (PINN).  The same tanh MLP, collocation set and optimizer are applied to the TVP and to
the IVP (conditions imposed exactly through trial functions) for kappa = 10, T = 1.  The
residual losses are comparable; the errors are not: the TVP error grows like e^{kappa (T - t)}
towards t = 0.  Errors are measured on a uniform grid of 1001 points.

Book: Chapter 6 (Terminal value problems), Section "Terminal value problems and time
reversal", Figure tvp:fig:conditioning.

Run:  PYTHONPATH=. python chapters/ch06_tvp/tvp_conditioning.py           (about 1 min, 1 core)
      PYTHONPATH=. python chapters/ch06_tvp/tvp_conditioning.py --quick   (smoke test)
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from pinnode import ConstrainedModel, Example, MLP, derivatives, linf, rel_l2, to_tensor, train, uniform

T = 1.0
KAPPA = 10.0                                   # PINN experiment
KAPPAS = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0])   # amplification study
DELTA = 1e-6                                   # size of the terminal perturbation (linear problem)
WIDTH, DEPTH, N_COLLOC = 32, 3, 100


def system_matrix(kappa: float) -> np.ndarray:
    return np.array([[0.0, 1.0], [-kappa, -(1.0 + kappa)]])


def exact(t, kappa):
    return (kappa * np.exp(-t) - np.exp(-kappa * t)) / (kappa - 1.0)


def dexact(t, kappa):
    return kappa * (np.exp(-kappa * t) - np.exp(-t)) / (kappa - 1.0)


# ----------------------------------------------------------------------------- part 1
def amplification_study():
    rows = []
    for kappa in KAPPAS:
        A = system_matrix(kappa)
        Mb, Mf = expm(-A * T), expm(A * T)
        U, s, Vt = np.linalg.svd(Mb)
        v = Vt[0]                                   # most amplified terminal direction
        zT = np.array([exact(T, kappa), dexact(T, kappa)])
        z0 = np.array([1.0, 0.0])
        rhs = lambda t, z: A @ z
        back = lambda zt: solve_ivp(rhs, (T, 0.0), zt, method="DOP853", rtol=1e-12,
                                    atol=1e-12).y[:, -1]
        zb = back(zT)
        zb_pert = back(zT + DELTA * v)
        zf = solve_ivp(rhs, (0.0, T), z0, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
        rows.append({
            "kappa": float(kappa),
            "amp_backward": float(s[0]),
            "amp_forward": float(np.linalg.norm(Mf, 2)),
            "amp_backward_measured": float(np.linalg.norm(zb_pert - zb) / DELTA),
            "backward_error_y0": float(abs(zb[0] - 1.0)),
            "forward_error_yT": float(abs(zf[0] - zT[0])),
            "rel_condition_y0": float(s[0] * np.linalg.norm(zT) / np.linalg.norm(z0)),
        })
    return rows


def pendulum_sensitivity(beta=0.1, T_p=5.0, y_T=0.4):
    """Backward sensitivity matrix d(y, y')(0) / d(y, y')(T) of the pendulum TVP of Section 6.3.

    Integrates the variational equation Phi' = J(t) Phi, Phi(T) = I backward with DOP853; by
    Liouville's formula det Phi(0) = exp(beta T).
    """
    def rhs(t, Z):
        y, v = Z[0], Z[1]
        J = np.array([[0.0, 1.0], [-np.cos(y), -beta]])
        return np.concatenate([[v, -beta * v - np.sin(y)], (J @ Z[2:].reshape(2, 2)).ravel()])

    sol = solve_ivp(rhs, (T_p, 0.0), np.concatenate([[y_T, 0.0], np.eye(2).ravel()]),
                    method="DOP853", rtol=1e-12, atol=1e-12)
    Phi = sol.y[2:, -1].reshape(2, 2)
    return {"norm2": float(np.linalg.norm(Phi, 2)), "det": float(np.linalg.det(Phi)),
            "exp_beta_T": float(np.exp(beta * T_p)), "matrix": Phi}


# ----------------------------------------------------------------------------- part 2
def run_pinn(ex: Example, mode: str):
    """mode = 'ivp' (conditions at t = 0) or 'tvp' (conditions at t = T)."""
    torch.manual_seed(ex.seed)
    kappa = KAPPA
    yT, vT = exact(T, kappa), dexact(T, kappa)
    net = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, T))
    if mode == "ivp":
        model = ConstrainedModel(net, lambda t, n: 1.0 + 0.0 * t + (t / T) ** 2 * n)
    else:
        model = ConstrainedModel(net, lambda t, n: yT + vT * (t - T) + ((T - t) / T) ** 2 * n)
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 20), 0.0, T), requires_grad=True)
    t_test = np.linspace(0.0, T, 1001)
    y_ref = exact(t_test, kappa)
    t_test_t = to_tensor(t_test)

    def loss_fn():
        y, dy, ddy = derivatives(model(t_col), t_col, 2)
        r = ddy + (1.0 + kappa) * dy + kappa * y
        L_r = (r ** 2).mean()
        return L_r, {"residual": L_r}

    def eval_fn():
        with torch.no_grad():
            return {"rel_l2": rel_l2(model(t_test_t), y_ref)}

    hist = train(model.parameters(), loss_fn, adam_iters=ex.pick(4000, 60), lr=1e-3,
                 lr_final=1e-5, lbfgs_iters=ex.pick(1000, 10), eval_fn=eval_fn,
                 eval_every=50, print_every=2000)
    with torch.no_grad():
        y_pred = model(t_test_t).numpy().ravel()
    err = np.abs(y_pred - y_ref)
    return {"hist": hist, "t": t_test, "y_ref": y_ref, "y_pred": y_pred, "err": err,
            "rel_l2": rel_l2(y_pred, y_ref), "linf": linf(y_pred, y_ref),
            "err_t0": float(err[0]), "err_tmid": float(err[len(err) // 2]),
            "err_tT": float(err[-1]), "final_loss": hist.final_loss,
            "train_time_s": hist.train_time}


# ----------------------------------------------------------------------------- caching
# Finished trainings are cached in the output folder; with --resume a rerun (for example after
# the process was interrupted) reuses them instead of training again. Times are those of the
# original run.
RESUME_ARG = (["--resume"], {"action": "store_true",
                             "help": "reuse trainings cached in the output folder"})


def load_cache(ex, tag):
    import pickle
    path = ex.path(f"cache_{tag}.pkl")
    if getattr(ex.args, "resume", False) and path.exists():
        with open(path, "rb") as fh:
            ex.log(f"resumed '{tag}' from {path.name}")
            return pickle.load(fh)
    return None


def save_cache(ex, tag, obj):
    import pickle
    with open(ex.path(f"cache_{tag}.pkl"), "wb") as fh:
        pickle.dump(obj, fh)


def main():
    ex = Example("ch06_tvp", "tvp_conditioning",
                 description="Backward amplification for TVPs: classical solver and PINN",
                 default_seed=1234, extra_args=[RESUME_ARG])
    rows = amplification_study()
    ex.save_table(rows, "amplification")
    for r in rows:
        ex.log(f"kappa={r['kappa']:5.1f}  ||e^(-AT)||={r['amp_backward']:.3e} "
               f"(measured {r['amp_backward_measured']:.3e})  ||e^(AT)||={r['amp_forward']:.3e}  "
               f"backward err y(0)={r['backward_error_y0']:.1e}  forward err y(T)={r['forward_error_yT']:.1e}")

    pend = pendulum_sensitivity()
    ex.log(f"pendulum TVP: ||dY(0)/dY(T)||_2 = {pend['norm2']:.4f}, det = {pend['det']:.6f} "
           f"(exp(beta T) = {pend['exp_beta_T']:.6f})")

    res = {}
    for mode in ("ivp", "tvp"):
        ex.log(f"training PINN for the {mode.upper()} (kappa={KAPPA}, T={T})")
        res[mode] = load_cache(ex, mode) or run_pinn(ex, mode)
        save_cache(ex, mode, res[mode])
        ex.save_history(res[mode]["hist"], mode)
        r = res[mode]
        ex.log(f"{mode}: loss={r['final_loss']:.2e} relL2={r['rel_l2']:.2e} linf={r['linf']:.2e} "
               f"err(0)={r['err_t0']:.2e} err(T/2)={r['err_tmid']:.2e} time={r['train_time_s']:.1f}s")
    ex.save_table({"t": res["ivp"]["t"], "y_exact": res["ivp"]["y_ref"],
                   "y_pinn_ivp": res["ivp"]["y_pred"], "y_pinn_tvp": res["tvp"]["y_pred"]},
                  "predictions")
    pinn_summary = {m: {k: res[m][k] for k in ("rel_l2", "linf", "err_t0", "err_tmid", "err_tT",
                                                "final_loss", "train_time_s")} for m in res}
    growth_mid_to_0 = res["tvp"]["err_t0"] / max(res["tvp"]["err_tmid"], 1e-300)
    ex.save_results({
        "problem": {"ode": "y'' + (1+kappa) y' + kappa y = 0", "T": T, "kappa_pinn": KAPPA,
                    "kappas_study": KAPPAS, "delta": DELTA},
        "hyperparameters": {"network": f"tanh MLP {DEPTH}x{WIDTH}", "collocation": N_COLLOC,
                            "collocation_type": "uniform", "adam_iters": ex.pick(4000, 60),
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": ex.pick(1000, 10),
                            "constraints": "hard (trial functions)"},
        "amplification": rows,
        "pendulum_backward_sensitivity": pend,
        "pinn": pinn_summary,
        "tvp_error_growth_mid_to_0": growth_mid_to_0,
        "predicted_growth_exp_kappa_T_half": float(np.exp(KAPPA * T / 2)),
        "rel_l2": res["tvp"]["rel_l2"], "linf": res["tvp"]["linf"],
        "train_time_s": res["tvp"]["train_time_s"] + res["ivp"]["train_time_s"],
    })
    if ex.args.no_figures:
        return
    from pinnode import plotting as P          # imported only when figures are drawn

    # ---------------------------------------------------------------- composite figure
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax = axes[0, 0]
    t = res["ivp"]["t"]
    P.plot_reference(ax, t, res["ivp"]["y_ref"], label="exact")
    P.plot_prediction(ax, t, res["ivp"]["y_pred"], label="PINN, IVP", k=0, markevery=(40, 100))
    P.plot_prediction(ax, t, res["tvp"]["y_pred"], label="PINN, TVP", k=1, markevery=(90, 100))
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    P.plot_error(ax, t, res["ivp"]["err"], label="PINN, IVP", k=0)
    P.plot_error(ax, t, res["tvp"]["err"], label="PINN, TVP", k=1)
    guide = res["tvp"]["err_tmid"] * np.exp(KAPPA * (T / 2 - t))
    mask = (t < 0.85 * T)
    ax.semilogy(t[mask], guide[mask], color=P.COLORS["muted"], lw=0.9, ls=":",
                label=r"$\propto e^{\kappa(T-t)}$")
    ax.set_ylim(1e-10, 1.0)
    ax.legend(loc="upper right")

    ax = axes[1, 0]
    kap = np.array([r["kappa"] for r in rows])
    ax.semilogy(kap, [r["amp_backward"] for r in rows], color=P.SERIES[1], ls="-", lw=1.1,
                label=r"backward $\|e^{-AT}\|_2$")
    ax.semilogy(kap, [r["amp_backward_measured"] for r in rows], ls="none", marker="o",
                markerfacecolor="white", markeredgecolor=P.SERIES[1], markersize=3.5,
                label="measured (DOP853)")
    ax.semilogy(kap, [r["amp_forward"] for r in rows], color=P.SERIES[0], ls="--", lw=1.1,
                marker="s", markersize=3, markerfacecolor="white", label=r"forward $\|e^{AT}\|_2$")
    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel("amplification factor")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    for k, mode in enumerate(("ivp", "tvp")):
        h = res[mode]["hist"]
        ax.semilogy(h.iters, P.rolling_median(np.asarray(h.loss), 9), color=P.SERIES[k],
                    ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.1,
                    label=f"PINN, {mode.upper()}")
    ax.axvline(res["tvp"]["hist"].switch_iter or 0, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual loss")
    ax.legend(loc="upper right")
    P.label_panels(axes)
    ex.savefig(fig, "tvp_conditioning")


if __name__ == "__main__":
    main()
