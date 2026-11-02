"""Spectral bias on a two-scale first-order problem: tanh MLP vs Fourier features vs SIREN.

Problem (book Section "Spectral bias and multi-scale solutions", Chapter 9):

    y'(t) = 2 pi cos(2 pi t) + 0.1 (2 pi K) cos(2 pi K t),  t in [0, 1],  y(0) = 0,  K = 16,

with exact solution y(t) = sin(2 pi t) + 0.1 sin(2 pi K t) (one slow and one fast mode).
Every network uses the hard initial condition y_theta(t) = t N_theta(t), so the loss is the
mean squared residual on M = 200 uniform collocation points. Compared networks (width 64,
depth 3): tanh MLP, Fourier-feature networks with Gaussian frequencies of standard
deviation sigma in {1, 8, 20} (the rule sigma = K T / 2 gives 8), and SIREN.

Outputs: relative L2 error during training, the amplitudes of the error in the slow (k=1)
and fast (k=K) modes, and the eigen-decomposition of the empirical neural tangent kernel of
the residual at initialization (eigenvalue against the dominant frequency of its eigenvector).

Run:  PYTHONPATH=. python chapters/ch09_training_diagnostics/spectral_bias.py          (~3 min, 1 core)
      PYTHONPATH=. python chapters/ch09_training_diagnostics/spectral_bias.py --quick  (smoke test)
      ... --K 8 --outdir outputs/ch09_training_diagnostics/spectral_bias_K8   (other fast frequency;
      the middle Fourier-feature network then uses sigma = K/2)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (CPUTimer, RunCache, dominant_frequency, latex_table, num,  # noqa: E402
                     projection_amplitude, replot_arg)
from pinnode import (ConstrainedModel, Example, FourierFeatureNet, MLP, SirenNet,  # noqa: E402
                     count_parameters, derivative, linf, rel_l2, to_tensor, train, uniform)
from pinnode import plotting as P  # noqa: E402

K = 16          # fast frequency (cycles per unit time)
A_FAST = 0.1    # amplitude of the fast mode
T = 1.0
WIDTH, DEPTH = 64, 3
MID = "ff8"     # Fourier-feature network with sigma = K T / 2
NETS = [  # (key, legend label)
    ("mlp", "tanh MLP"),
    ("ff1", r"FF, $\sigma=1$"),
    (MID, r"FF, $\sigma=8$"),
    ("ff20", r"FF, $\sigma=20$"),
    ("siren", "SIREN"),
]


def configure(k_fast: int) -> None:
    """Set the fast frequency K and the middle Fourier-feature network (sigma = K/2)."""
    global K, MID, NETS
    K = int(k_fast)
    MID = f"ff{K * T / 2:g}"
    NETS = [NETS[0], NETS[1], (MID, rf"FF, $\sigma={K * T / 2:g}$"), NETS[3], NETS[4]]


def exact(t):
    return np.sin(2 * np.pi * t) + A_FAST * np.sin(2 * np.pi * K * t)


def rhs(t):
    return 2 * np.pi * torch.cos(2 * np.pi * t) + A_FAST * 2 * np.pi * K * torch.cos(2 * np.pi * K * t)


def build(key: str):
    if key == "mlp":
        net = MLP(1, 1, WIDTH, DEPTH, "tanh", domain=(0.0, T))
    elif key.startswith("ff"):
        sigma = float(key[2:])
        net = FourierFeatureNet(1, 1, n_features=32, sigma=sigma, width=WIDTH, depth=DEPTH,
                                activation="tanh", domain=(0.0, T))
    elif key == "siren":
        net = SirenNet(1, 1, WIDTH, DEPTH, w0_first=30.0, domain=(0.0, T))
    else:
        raise ValueError(key)
    return ConstrainedModel(net, lambda t, n: t * n)


def residual_ntk(model, t_col):
    """Empirical NTK K = J J^T of the residual r_i = y_theta'(t_i) - f(t_i) w.r.t. theta."""
    params = [p for p in model.parameters() if p.requires_grad]
    t = t_col.detach().clone().requires_grad_(True)
    dy = derivative(model(t), t).reshape(-1)
    rows = []
    for i in range(dy.shape[0]):
        g = torch.autograd.grad(dy[i], params, retain_graph=True, allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(p) if gi is None else gi).reshape(-1)
                               for gi, p in zip(g, params)]))
    J = torch.stack(rows)
    Kmat = (J @ J.T).detach().numpy()
    evals, evecs = np.linalg.eigh(Kmat)
    return evals[::-1].copy(), evecs[:, ::-1].copy()


def main():
    ex = Example("ch09_training_diagnostics", "spectral_bias",
                 description="Spectral bias: tanh MLP vs Fourier features vs SIREN on y'=f",
                 default_seed=1234,
                 extra_args=[replot_arg(), (["--K"], {"type": int, "default": 16, "help": "fast frequency (default 16)"})])
    configure(ex.args.K)
    M = ex.pick(200, 64)
    n_adam, n_lbfgs = ex.pick(6000, 60), ex.pick(500, 10)
    eval_every = ex.pick(50, 10)
    t_col_np = uniform(M, 0.0, T)
    t_col = to_tensor(t_col_np, requires_grad=True)
    t_test = np.linspace(0.0, T, 1001)
    y_ref = exact(t_test)
    t_test_t = to_tensor(t_test)
    f_col = rhs(t_col.detach())

    runs, rows = {}, []
    cache = RunCache(ex)
    for key, label in NETS:
        cached = cache.get(key, lambda key=key, label=label: train_one(ex, key, label, t_col, t_col_np, f_col,
                                                                    t_test, y_ref, t_test_t, n_adam, n_lbfgs,
                                                                    eval_every))
        runs[key], row = cached["run"], cached["row"]
        rows.append(row)
        ex.save_history(runs[key]["hist"], key)
        ex.log(f"{label}: relL2={row['rel_l2']:.2e} (after Adam {row['rel_l2_after_adam']:.2e}) "
               f"linf={row['linf']:.2e} A1={row['amp_k1_final']:.1e} AK={row['amp_kK_final']:.1e} "
               f"cpu={row['cpu_time_s']:.0f}s")
    _finish(ex, runs, rows, t_test, y_ref, M, n_adam, n_lbfgs)
    cache.clear()


def train_one(ex, key, label, t_col, t_col_np, f_col, t_test, y_ref, t_test_t, n_adam, n_lbfgs, eval_every):
    """Train one network; returns picklable results (no model objects)."""
    torch.manual_seed(ex.seed)
    model = build(key)
    ex.log(f"{label}: {count_parameters(model)} parameters")
    # ---- NTK of the residual at initialization
    evals, evecs = residual_ntk(model, t_col)
    freqs = np.array([dominant_frequency(evecs[:, j], t_col_np) for j in range(evecs.shape[1])])

    def loss_fn():
        y = model(t_col)
        r = derivative(y, t_col) - f_col
        L = (r ** 2).mean()
        return L, {"residual": L}

    def eval_fn():
        with torch.no_grad():
            yp = model(t_test_t).numpy().ravel()
        e = yp - y_ref
        a1, aK = projection_amplitude(e, t_test, 1.0), projection_amplitude(e, t_test, K)
        return {"rel_l2": rel_l2(yp, y_ref), "amp_k1": a1, "amp_kK": aK}

    with CPUTimer() as tm:
        hist = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
                     lbfgs_iters=n_lbfgs, eval_fn=eval_fn, eval_every=eval_every, print_every=2000)
    with torch.no_grad():
        y_pred = model(t_test_t).numpy().ravel()
    e_rel, e_inf = rel_l2(y_pred, y_ref), linf(y_pred, y_ref)
    i_switch = int(np.searchsorted(hist.eval_iters, n_adam))
    rel_adam = hist.metrics["rel_l2"][min(i_switch, len(hist.eval_iters) - 1)]
    run = {"label": label, "hist": hist, "y_pred": y_pred, "evals": evals, "freqs": freqs}
    row = {"network": key, "label": label, "parameters": count_parameters(model),
           "rel_l2": e_rel, "linf": e_inf, "rel_l2_after_adam": rel_adam,
           "amp_k1_final": hist.metrics["amp_k1"][-1], "amp_kK_final": hist.metrics["amp_kK"][-1],
           "final_loss": hist.final_loss, "train_time_s": hist.train_time, "cpu_time_s": tm.cpu,
           "ntk_lambda_max": float(evals[0]),
           "ntk_eigs_above_1e-3_lmax": int(np.sum(evals > 1e-3 * evals[0]))}
    return {"run": run, "row": row}


def _finish(ex, runs, rows, t_test, y_ref, M, n_adam, n_lbfgs):
    ex.save_table(rows, "summary")
    ex.save_table({"t": t_test, "exact": y_ref, **{k: runs[k]["y_pred"] for k, _ in NETS}}, "predictions")
    ex.save_table({**{f"{k}_eig": runs[k]["evals"] for k, _ in NETS},
                   **{f"{k}_freq": runs[k]["freqs"] for k, _ in NETS}}, "ntk_spectrum")
    ex.save_results({
        "problem": {"ode": "y' = 2 pi cos(2 pi t) + 0.1*2 pi K cos(2 pi K t), y(0)=0", "K": K,
                    "amplitude_fast": A_FAST, "T": T, "exact": "sin(2 pi t) + 0.1 sin(2 pi K t)"},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "collocation": M,
                            "collocation_type": "uniform", "adam_iters": n_adam,
                            "lr": "1e-3 -> 1e-5 (cosine)", "lbfgs_iters": n_lbfgs,
                            "fourier_features": 32, "siren_w0_first": 30.0,
                            "constraint": "y = t N(t) (hard IC)", "test_grid": "1001 uniform points"},
        "summary": rows})

    # ---------------------------------------------------------------- LaTeX table
    tab_rows = [[r["label"], f"{r['parameters']}", num(r["rel_l2_after_adam"]), num(r["rel_l2"]),
                 num(r["linf"]), num(r["amp_kK_final"])] for r in rows]
    latex_table(ex.path("tab_spectral_bias.tex"), "lrrrrr",
                ["network", "parameters", r"rel.\ $L^2$ (Adam)", r"rel.\ $L^2$ (final)",
                 r"max error", rf"$A_{{{K}}}$ (final)"], tab_rows,
                comment="Chapter 9, spectral_bias.py")

    if ex.args.no_figures:
        return
    # ---------------------------------------------------------------- figure
    fig, axes = P.figure(2, 2, aspect=0.74)
    ax = axes[0, 0]
    zoom = t_test >= 0.75
    P.plot_reference(ax, t_test[zoom], y_ref[zoom], label="exact")
    for k, key in enumerate(["mlp", MID, "siren"]):
        P.plot_prediction(ax, t_test[zoom], runs[key]["y_pred"][zoom], label=runs[key]["label"],
                          k=[0, 2, 4][k], lw=1.3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$y(t)$")
    ax.set_title(r"solutions on $[0.75, 1]$")
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    for k, (key, label) in enumerate(NETS):
        if key == "ff20":      # diverges (relative error > 1); reported in the table
            continue
        h = runs[key]["hist"]
        ax.semilogy(h.eval_iters, h.metrics["rel_l2"], color=P.SERIES[k],
                    ls="-" if k == 0 else P.LINESTYLES[k - 1], lw=1.2, label=label)
    ax.axvline(n_adam, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"relative $L^2$ error")
    ax.set_title("training error")
    ax.set_ylim(top=8.0)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5)

    ax = axes[1, 0]
    for k, key in enumerate(["mlp", MID]):
        h = runs[key]["hist"]
        c = P.SERIES[0 if key == "mlp" else 2]
        short = "MLP" if key == "mlp" else runs[key]["label"].replace("FF, ", "FF ")
        ax.semilogy(h.eval_iters, h.metrics["amp_k1"], color=c, ls="-", lw=1.2,
                    label=f"{short}, $k=1$")
        ax.semilogy(h.eval_iters, h.metrics["amp_kK"], color=c, ls="--", lw=1.2,
                    label=f"{short}, $k={K}$")
    ax.axvline(n_adam, color=P.COLORS["muted"], lw=0.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel("error amplitude $A_k$")
    ax.set_title("frequency-resolved error")
    ax.set_ylim(top=8.0)
    ax.legend(loc="upper right", ncol=2, fontsize=6.5)

    ax = axes[1, 1]
    for k, key in enumerate(["mlp", MID, "siren"]):
        rr = runs[key]
        lam = rr["evals"] / rr["evals"][0]
        keep = lam > 1e-16
        ax.semilogy(rr["freqs"][keep], lam[keep], ls="none", marker=P.MARKERS[k],
                    color=P.SERIES[[0, 2, 4][k]], markersize=2.6, markerfacecolor="white",
                    markeredgewidth=0.7, label=rr["label"])
    ax.axvline(K, color=P.COLORS["muted"], lw=0.6, ls=":")
    ax.set_xlim(-2, 60)
    ax.set_xlabel("dominant frequency of eigenvector")
    ax.set_ylabel(r"$\lambda_j/\lambda_{\max}$")
    ax.set_title("residual NTK at initialization")
    ax.legend(loc="lower center", fontsize=6.5)
    P.label_panels(axes)
    ex.savefig(fig, "spectral_bias_summary")


if __name__ == "__main__":
    main()
