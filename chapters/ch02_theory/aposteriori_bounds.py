"""Computable a-posteriori error bounds for PINNs: true error versus residual-based bound.

Book: Chapter 2 (Approximation, optimization and error analysis), Section "Error analysis",
the numerical illustration (composite figure and table of effectivity indices).

For a trained network y_theta with residual r(t) and condition defects, the bounds of
Chapter 2 are evaluated along t without using the exact solution, and compared with the
true error |y_theta(t) - y(t)| at three stages of training (after 500 and 5000 Adam
iterations and after the final L-BFGS phase).

Problem A (linear IVP)   y' = -y + sin t, y(0) = 1, t in [0, 10],
    exact y = 1.5 e^{-t} + (sin t - cos t)/2.  Lipschitz constant L = 1, one-sided
    constant mu = -1.  Bounds:
      sharp  B(t) = e^{-t}|e_0| + int_0^t e^{-(t-s)} |r(s)| ds          (mu = -1)
      crude  B(t) = e^{ t}|e_0| + int_0^t e^{ (t-s)} |r(s)| ds          (L = 1)
Problem B (logistic IVP) y' = y (1 - y), y(0) = 0.1, t in [0, 6],
    exact y = 0.1 e^t / (1 + 0.1 (e^t - 1)).  Since y(t) >= y(0) = 0.1, the error equation
    e' = (1 - y_theta - y) e + r gives the computable one-sided bound
      mu(t) = 1 - 0.1 - y_theta(t),
    the crude bound uses the Lipschitz constant L = max |1 - 2s| over the range of y, y_theta.
Problem B' is the same equation on [0, 10] (table only): with the settings of this script the
    training is attracted to a spurious solution near the unstable equilibrium y = 0 (see
    Chapter 9), and the bound, which remains valid, reports a large error.
Problem C (nonlinear BVP) -y'' + y^3 = g on (0, 1), y(0) = 0, y(1) = 1/2,
    exact y = sin(pi t) + t/2.  The nonlinearity is nondecreasing, so (maximum principle)
      sharp  |e(t)| <= int_0^1 G_0(t,s) |r(s)| ds + (1-t)|d_0| + t|d_1|,
      crude  |e(t)| <= ||r||_{L^1}/4 + max(|d_0|, |d_1|),
    with G_0(t,s) = min(t,s)(1 - max(t,s)) the Green's function of -d^2/dt^2.

All integrals are evaluated with the trapezoidal rule on 4001 uniform points; errors are
reported on the same grid (the figure shows every fourth point, i.e. 1001 points).

Run:  PYTHONPATH=. python chapters/ch02_theory/aposteriori_bounds.py          (about 3 min, 1 core)
      PYTHONPATH=. python chapters/ch02_theory/aposteriori_bounds.py --quick  (smoke test)
"""
from __future__ import annotations

import math

import numpy as np
import torch

from pinnode import Example, MLP, derivative, linf, rel_l2, to_tensor, train, uniform
from pinnode import plotting as P

WIDTH, DEPTH, N_COLLOC, N_FINE = 32, 3, 200, 4001
STAGE_LABELS = ["Adam 500", "Adam 5000", "Adam + L-BFGS"]


# ----------------------------------------------------------------------------- helpers
def cumtrapz(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(t))])


def ivp_bound(t, r_abs, e0_abs, mu):
    """e^{M(t)} (|e0| + int_0^t e^{-M(s)} |r(s)| ds) with M(t) = int_0^t mu."""
    mu = np.broadcast_to(np.asarray(mu, float), t.shape)
    M = cumtrapz(mu, t)
    return np.exp(M) * (e0_abs + cumtrapz(np.exp(-M) * r_abs, t))


def green_bound(t, r_abs, d0_abs, d1_abs):
    """int_0^1 G_0(t,s)|r(s)| ds + (1-t)|d0| + t|d1| on [0, 1], G_0 = min(t,s)(1-max(t,s))."""
    A = cumtrapz(t * r_abs, t)                     # int_0^t s |r(s)| ds
    Bc = cumtrapz((1.0 - t) * r_abs, t)            # int_0^t (1-s)|r(s)| ds
    return (1.0 - t) * A + t * (Bc[-1] - Bc) + (1.0 - t) * d0_abs + t * d1_abs


# ----------------------------------------------------------------------------- problems
def exact_A(t):
    return 1.5 * np.exp(-t) + 0.5 * (np.sin(t) - np.cos(t))


def exact_B(t, y0=0.1):
    return y0 * np.exp(t) / (1.0 + y0 * (np.exp(t) - 1.0))


def exact_C(t):
    return np.sin(np.pi * t) + 0.5 * t


PROBLEMS = {
    "linear_ivp": {"title": r"A: $y'=-y+\sin t$", "a": 0.0, "b": 10.0, "kind": "ivp", "y0": 1.0,
                   "exact": exact_A},
    "logistic_ivp": {"title": r"B: $y'=y(1-y)$", "a": 0.0, "b": 6.0, "kind": "ivp", "y0": 0.1,
                     "exact": exact_B},
    "logistic_long": {"title": r"B': $y'=y(1-y)$ on $[0,10]$", "a": 0.0, "b": 10.0, "kind": "ivp",
                      "y0": 0.1, "exact": exact_B, "in_figure": False},
    "monotone_bvp": {"title": r"C: $-y''+y^3=g$", "a": 0.0, "b": 1.0, "kind": "bvp",
                     "ya": 0.0, "yb": 0.5, "exact": exact_C},
}


def residual(name: str, model, t):
    y = model(t)
    if name == "linear_ivp":
        return y, derivative(y, t) + y - torch.sin(t)
    if name in ("logistic_ivp", "logistic_long"):
        return y, derivative(y, t) - y * (1.0 - y)
    if name == "monotone_bvp":
        g = math.pi ** 2 * torch.sin(math.pi * t) + (torch.sin(math.pi * t) + 0.5 * t) ** 3
        return y, -derivative(y, t, 2) + y ** 3 - g
    raise ValueError(name)


def defects(name: str, model):
    p = PROBLEMS[name]
    if p["kind"] == "ivp":
        return {"e0": model(to_tensor([p["a"]])).reshape(()) - p["y0"]}
    return {"d0": model(to_tensor([p["a"]])).reshape(()) - p["ya"],
            "d1": model(to_tensor([p["b"]])).reshape(()) - p["yb"]}


def evaluate(name: str, model, t_fine: np.ndarray, loss_parts: dict) -> dict:
    p = PROBLEMS[name]
    ys, rs = [], []
    for chunk in np.array_split(t_fine, max(1, len(t_fine) // 500)):   # chunks keep memory low
        tt = to_tensor(chunk, requires_grad=True)
        y_c, r_c = residual(name, model, tt)
        ys.append(y_c.detach().numpy().ravel())
        rs.append(r_c.detach().numpy().ravel())
        del tt, y_c, r_c
    y = np.concatenate(ys)
    r_abs = np.abs(np.concatenate(rs))
    d = {k: float(v.detach()) for k, v in defects(name, model).items()}
    y_ex = p["exact"](t_fine)
    err = np.abs(y - y_ex)
    if name == "linear_ivp":
        sharp = ivp_bound(t_fine, r_abs, abs(d["e0"]), -1.0)
        crude = ivp_bound(t_fine, r_abs, abs(d["e0"]), 1.0)
        consts = {"mu": -1.0, "L": 1.0}
    elif name in ("logistic_ivp", "logistic_long"):
        mu_t = 1.0 - p["y0"] - y
        lo, hi = min(0.0, y.min()), max(1.0, y.max())
        L = float(max(abs(1 - 2 * lo), abs(1 - 2 * hi)))
        sharp = ivp_bound(t_fine, r_abs, abs(d["e0"]), mu_t)
        crude = ivp_bound(t_fine, r_abs, abs(d["e0"]), L)
        consts = {"mu(t)": "0.9 - y_theta(t)", "L": L, "mu_min": float(mu_t.min()),
                  "mu_max": float(mu_t.max())}
    else:
        sharp = green_bound(t_fine, r_abs, abs(d["d0"]), abs(d["d1"]))
        r_l1 = float(cumtrapz(r_abs, t_fine)[-1])
        crude = np.full_like(t_fine, 0.25 * r_l1 + max(abs(d["d0"]), abs(d["d1"])))
        consts = {"G_max": 0.25, "r_L1": r_l1}
    msr = float(cumtrapz(r_abs ** 2, t_fine)[-1] / (t_fine[-1] - t_fine[0]))
    return {"t": t_fine, "y": y, "y_exact": y_ex, "err": err, "r_abs": r_abs, "sharp": sharp,
            "crude": crude, "defects": d, "consts": consts,
            "loss": float(sum(loss_parts.values())), "loss_parts": loss_parts,
            "msr_fine": msr, "max_err": float(err.max()), "rel_l2": rel_l2(y, y_ex),
            "max_sharp": float(sharp.max()), "max_crude": float(crude.max()),
            "effectivity": float(sharp.max() / err.max()),
            "effectivity_crude": float(crude.max() / err.max()),
            # the bound holds exactly; on the grid it can only fail by the quadrature error of
            # the trapezoidal rule, so validity is checked with a relative tolerance of 1e-3
            "valid": bool(np.all(sharp >= err * (1 - 1e-3))),
            "min_ratio": float(np.min(sharp / np.maximum(err, 1e-300)))}


def run_problem(ex: Example, name: str):
    p = PROBLEMS[name]
    torch.manual_seed(ex.seed)
    model = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(p["a"], p["b"]))
    t_col = to_tensor(uniform(ex.pick(N_COLLOC, 40), p["a"], p["b"]), requires_grad=True)
    t_fine = np.linspace(p["a"], p["b"], ex.pick(N_FINE, 401))

    def loss_fn():
        _, r = residual(name, model, t_col)
        L_r = (r ** 2).mean()
        d = defects(name, model)
        L_c = sum(v ** 2 for v in d.values())
        key = "ic" if p["kind"] == "ivp" else "bc"
        return L_r + L_c, {"residual": L_r, key: L_c}

    def current_parts():
        _, parts = loss_fn()
        return {k: float(v.detach()) for k, v in parts.items()}

    n_adam = ex.pick(5000, 60)
    first_stop = ex.pick(500, 20)
    stages = []

    def hook(k):                                     # called before every Adam step
        if k == first_stop:
            stages.append(evaluate(name, model, t_fine, current_parts()))

    h1 = train(model.parameters(), loss_fn, adam_iters=n_adam, lr=1e-3, lr_final=1e-5,
               lbfgs_iters=0, weight_update=hook, print_every=2500)
    stages.append(evaluate(name, model, t_fine, current_parts()))
    h2 = train(model.parameters(), loss_fn, adam_iters=0, lbfgs_iters=ex.pick(500, 10),
               print_every=2500)
    stages.append(evaluate(name, model, t_fine, current_parts()))
    lbfgs_done = int(h2.iters[-1]) if h2.iters else 0
    return {"stages": stages, "hist_adam": h1, "hist_lbfgs": h2, "lbfgs_iters_done": lbfgs_done,
            "train_time_s": h1.train_time + h2.train_time}


def make_figure(ex: Example, stages: dict) -> None:
    fig_problems = [n for n in PROBLEMS if PROBLEMS[n].get("in_figure", True)]
    fig, axes = P.figure(3, 2, aspect=0.60)
    for row, name in enumerate(fig_problems):
        st = stages[name]
        ax = axes[row, 0]
        for k, s in enumerate(st):
            t = s["t"][::4]
            ax.semilogy(t, np.maximum(s["err"][::4], 1e-12), color=P.SERIES[k], lw=1.1, ls="-",
                        label=STAGE_LABELS[k])
            ax.semilogy(t, s["sharp"][::4], color=P.SERIES[k], lw=1.1, ls="--")
        ax.set_ylabel("error / bound")
        ax.set_title(PROBLEMS[name]["title"], fontsize=8)
        ax = axes[row, 1]
        s = st[-1]
        t = s["t"][::4]
        ax.semilogy(t, s["r_abs"][::4], color=P.COLORS["light"], lw=0.8, ls="-", label="$|r(t)|$")
        ax.semilogy(t, np.maximum(s["err"][::4], 1e-12), color=P.COLORS["ref"], lw=1.1, ls="-",
                    label="true error")
        ax.semilogy(t, s["sharp"][::4], color=P.SERIES[0], lw=1.3, ls=P.LINESTYLES[0],
                    label="sharp bound")
        ax.semilogy(t, s["crude"][::4], color=P.SERIES[1], lw=1.3, ls=P.LINESTYLES[1],
                    label="crude bound")
        ax.set_title(PROBLEMS[name]["title"] + ", final stage", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("$t$")
    from matplotlib.lines import Line2D
    stage_handles = [Line2D([], [], color=P.SERIES[k], lw=1.1, label=STAGE_LABELS[k]) for k in range(3)]
    style_handles = [Line2D([], [], color="k", lw=1.0, ls="-", label="true error"),
                     Line2D([], [], color="k", lw=1.0, ls="--", label="bound")]
    for ax in axes[0, :]:                    # head room for the legends above the curves
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 10 ** 3.2)
    axes[0, 0].legend(handles=stage_handles + style_handles, loc="upper center", ncol=3,
                      fontsize=6.5, handlelength=1.8, columnspacing=0.8)
    axes[0, 1].legend(loc="upper left", ncol=2, fontsize=6.5, handlelength=1.8, columnspacing=0.8)
    P.label_panels(axes)
    ex.savefig(fig, "aposteriori_bounds")


def load_stages(ex: Example) -> dict:
    out = {}
    for name in PROBLEMS:
        f = ex.path(f"stages_{name}.npz")
        if not f.exists():
            continue
        z = np.load(f)
        out[name] = [{k: z[f"{k}_{i}"] for k in ("t", "err", "sharp", "crude", "r_abs")}
                     for i in range(len(STAGE_LABELS))]
    return out


def main():
    ex = Example("ch02_theory", "aposteriori_bounds",
                 description="A-posteriori residual-based error bounds versus true PINN errors",
                 default_seed=1234,
                 extra_args=[(("--plot-only",), {"action": "store_true",
                                                 "help": "redraw the figure from saved outputs"})])
    if ex.args.plot_only:
        make_figure(ex, load_stages(ex))
        return
    results, rows = {}, []
    for name in PROBLEMS:
        ex.log(f"training {name}")
        out = run_problem(ex, name)
        results[name] = out
        ex.save_history(out["hist_adam"], f"{name}_adam")
        ex.save_history(out["hist_lbfgs"], f"{name}_lbfgs")
        for label, s in zip(STAGE_LABELS, out["stages"]):
            rows.append({"problem": name, "stage": label, "loss": s["loss"],
                         **{f"loss_{k}": v for k, v in s["loss_parts"].items()},
                         "msr_fine": s["msr_fine"], "max_err": s["max_err"], "rel_l2": s["rel_l2"],
                         "max_bound": s["max_sharp"], "effectivity": s["effectivity"],
                         "max_crude_bound": s["max_crude"],
                         "effectivity_crude": s["effectivity_crude"], "valid": s["valid"],
                         "min_ratio": s["min_ratio"],
                         **{f"defect_{k}": v for k, v in s["defects"].items()}})
            ex.log(f"  {label:14s} loss={s['loss']:.2e} msr={s['msr_fine']:.2e} "
                   f"max|e|={s['max_err']:.2e} maxB={s['max_sharp']:.2e} "
                   f"eff={s['effectivity']:.2f} crude={s['max_crude']:.2e} valid={s['valid']}")
        final = out["stages"][-1]
        ex.save_table({"t": final["t"][::4], "y_theta": final["y"][::4],
                       "y_exact": final["y_exact"][::4], "abs_error": final["err"][::4],
                       "abs_residual": final["r_abs"][::4], "bound_sharp": final["sharp"][::4],
                       "bound_crude": final["crude"][::4]}, f"{name}_final_stage")
    ex.save_table(rows, "summary")
    ex.save_results({
        "problems": {"linear_ivp": "y' = -y + sin t, y(0)=1, [0,10]",
                     "logistic_ivp": "y' = y(1-y), y(0)=0.1, [0,6]",
                     "logistic_long": "y' = y(1-y), y(0)=0.1, [0,10] (failure case)",
                     "monotone_bvp": "-y'' + y^3 = g, y(0)=0, y(1)=1/2, exact sin(pi t)+t/2"},
        "hyperparameters": {"width": WIDTH, "depth": DEPTH, "activation": "tanh",
                            "collocation": ex.pick(N_COLLOC, 40), "fine_grid": ex.pick(N_FINE, 401),
                            "adam_iters": ex.pick(5000, 60), "lr": "1e-3 -> 1e-5 (cosine)",
                            "lbfgs_iters_max": ex.pick(500, 10), "condition_weight": 1.0,
                            "stages": STAGE_LABELS},
        "constants_final_stage": {n: results[n]["stages"][-1]["consts"] for n in results},
        "lbfgs_iters_done": {n: results[n]["lbfgs_iters_done"] for n in results},
        "train_time_s": {n: results[n]["train_time_s"] for n in results},
        "summary": rows})

    for name in PROBLEMS:                 # all stage arrays, so the figure can be redrawn
        st = results[name]["stages"]
        np.savez_compressed(ex.path(f"stages_{name}.npz"),
                            **{f"{k}_{i}": st[i][k] for i in range(len(st))
                               for k in ("t", "err", "sharp", "crude", "r_abs")})
    if ex.args.no_figures:
        return
    make_figure(ex, {n: results[n]["stages"] for n in PROBLEMS})


if __name__ == "__main__":
    main()
