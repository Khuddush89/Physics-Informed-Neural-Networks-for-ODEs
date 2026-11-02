"""Chapter 10 driver: run the benchmarks and build every figure and LaTeX table of the chapter.

    PYTHONPATH=. python chapters/ch10_benchmarking/write_book_assets.py --run   # all examples + assets
    PYTHONPATH=. python chapters/ch10_benchmarking/write_book_assets.py         # assets only

``--run`` executes the seven example scripts one after the other (about 15 min of CPU time on
one core in total) and then builds the assets; without it the assets are rebuilt from the files
already in ``outputs/ch10_benchmarking``. All timing numbers of the chapter appear only in the
generated tables and figures, so re-running this driver on an idle machine refreshes them all.

Assets (outputs/ch10_benchmarking/book_assets/):
  figures/  ivp_work_precision, bvp_bratu, ie_work_precision, parametric_family, residual_certificate
  tables/   tab_environment, tab_ivp_matched, tab_ivp_pinn, tab_bvp, tab_ie, tab_parametric,
            tab_certificate (.tex, booktabs + siunitx), numbers.tex (accuracy numbers quoted in the
            text, as \\benchnum{key})
When the book sources are found next to the repository (../book/figures/ch10) or given with
--book-dir, the figure PDFs and the .tex files are copied there.

Options: --cost cpu|wall  timing measure (default cpu: CPU time of the single-threaded process,
                          equal to the wall-clock time on an idle machine; wall: perf_counter /
                          hist.train_time); --resume  pass --resume to the examples (reuse PINN runs
                          cached by an interrupted run); --quick  smoke test (tiny runs, no copy).
"""
from __future__ import annotations

import _bench as B  # noqa: I001  (must come first: single-threaded BLAS)
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pinnode import Example
from pinnode import plotting as P
from pinnode.utils import OUTPUT_ROOT, REPO_ROOT

CHAPTER = "ch10_benchmarking"
SCRIPTS = ["ivp_logistic", "ivp_vanderpol", "ivp_stiff", "bvp_bratu", "integral_equations",
           "parametric_logistic", "residual_certificate"]
IVP_PANELS = [("ivp_logistic", "logistic"), ("ivp_vanderpol", r"van der Pol, $\mu = 0.5$"),
              ("ivp_stiff", r"stiff linear, $\lambda = 10^3$")]
IVP_NAMES = {"ivp_logistic": "Logistic", "ivp_vanderpol": "Van der Pol", "ivp_stiff": "Stiff linear"}
TRAJ_TAG = {"ivp_logistic": "P4", "ivp_vanderpol": "V2", "ivp_stiff": "S3"}   # training curve shown
# label offsets (points) of the PINN tags in the IVP work-precision diagrams
TAG_OFFSETS = {"P1": (5, -9), "P2": (6, -3), "P3": (-15, 2), "P4": (5, 3),
               "V1": (5, -8), "V2": (5, 2), "V3": (-15, 4),
               "S1": (4, -9), "S2": (-15, -9), "S3": (-15, 2)}


# ----------------------------------------------------------------------------- formatting
def num(x, digits: int = 2) -> str:
    """siunitx number with ``digits`` significant digits (scientific notation, no '+')."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---" if x is None or not math.isinf(x) else r"$\infty$"
    if x == 0:
        return r"\num{0}"
    mant, ex = f"{float(x):.{digits - 1}e}".split("e")
    return rf"\num{{{mant}e{int(ex)}}}"


def secs(x) -> str:
    """Times: fixed notation with two significant digits from 1 s up, scientific below."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    if x >= 10:
        return rf"\num{{{x:.0f}}}" if x >= 99.5 else rf"\num{{{float(f'{x:.2g}'):.0f}}}"
    if x >= 1:
        return rf"\num{{{x:.1f}}}"
    return num(x)


def count(x) -> str:
    """Operation counts: plain integers below 1e5, two significant digits above."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "---"
    return rf"\num{{{int(round(x))}}}" if x < 1e5 else num(x)


def intnum(x) -> str:
    return "---" if x is None or (isinstance(x, float) and not math.isfinite(x)) else rf"\num{{{int(round(x))}}}"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Data:
    """Access to the output folder of one example."""

    def __init__(self, root: Path, name: str):
        self.dir = root / name
        self.name = name

    def exists(self) -> bool:
        return (self.dir / "results.json").exists()

    def csv(self, stem: str) -> pd.DataFrame:
        return pd.read_csv(self.dir / f"{stem}.csv")

    def json(self) -> dict:
        return json.loads((self.dir / "results.json").read_text())


def rows(df: pd.DataFrame) -> list:
    return df.to_dict("records")


def cost_key(args) -> str:
    return "cpu_s" if args.cost == "cpu" else "wall_s"


def lbfgs_done(hist: pd.DataFrame, adam_iters: int) -> int:
    return int(max(hist["iter"].max() - adam_iters, 0)) if (hist["phase"] == "lbfgs").any() else 0


def log_axes(ax, xticks=(1e-12, 1e-8, 1e-4, 1.0)):
    ax.xaxis.set_major_locator(mticker.FixedLocator(list(xticks)))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


# ----------------------------------------------------------------------------- figures
def fig_ivp(ex, root, cost):
    fig, axes = P.figure(1, 3, aspect=1.12, sharey=True)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.89))     # room for the shared legend
    for ax, (name, title) in zip(axes, IVP_PANELS):
        d = Data(root, name)
        cl, pinn, traj = d.csv("classical"), d.csv("pinn"), d.csv("pinn_trajectories")
        for m in B.IVP_METHODS:
            B.plot_classical(ax, rows(cl[cl.method == m]), m, cost=cost)
        tr = traj[traj.tag == TRAJ_TAG[name]]
        B.plot_trajectory(ax, tr[cost].values, tr["rel_l2"].values, label="PINN training")
        B.plot_pinn(ax, rows(pinn), cost=cost, offsets=[TAG_OFFSETS.get(t, (4, 3)) for t in pinn["tag"]])
        B.mark_targets(ax)
        ax.set_title(title)
        ax.set_xlabel(r"relative $L^2$ error")
        ax.set_xlim(1e-14, max(3.0, 4.0 * float(pinn["rel_l2"].max())))
        log_axes(ax, xticks=(1e-12, 1e-8, 1e-4, 1.0))
    axes[0].set_ylabel("CPU time [s]" if cost == "cpu_s" else "wall-clock time [s]")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=7, handlelength=2.2,
               columnspacing=1.0)
    P.label_panels(axes)
    ex.savefig(fig, "ivp_work_precision")


def fig_bvp(ex, root, cost):
    d = Data(root, "bvp_bratu")
    cl, pinn, pred = d.csv("classical"), d.csv("pinn"), d.csv("predictions")
    traj = d.csv("pinn_trajectories")
    fig, axes = P.figure(1, 2, aspect=0.82)
    ax = axes[0]
    for m, lab in (("solve_bvp", "collocation"), ("shooting", "shooting"), ("FD-Newton", "FD-Newton")):
        B.plot_classical(ax, rows(cl[cl.method == m]), m, label=lab, cost=cost)
    tr = traj[traj.tag == "B2"]
    B.plot_trajectory(ax, tr[cost].values, tr["rel_l2"].values, label="PINN training (B2)")
    B.plot_pinn(ax, rows(pinn[pinn.arch == "MLP"]), label="PINN (MLP)", cost=cost)
    B.plot_pinn(ax, rows(pinn[pinn.arch == "ChNN"]), label="PINN (ChNN)", cost=cost,
                style=B.PINN_STYLE_ALT)
    B.mark_targets(ax)
    ax.set_xlim(1e-14, 3.0)
    log_axes(ax)
    B.finish_wp_axes(ax, ylabel="CPU time [s]" if cost == "cpu_s" else "wall-clock time [s]")
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_title("work-precision")
    ax = axes[1]
    t, ex_y = pred["t"].values, pred["exact"].values
    series = [("solve_bvp_tol1e-6", r"collocation, tol $10^{-6}$"), ("shooting_tau1e-6", r"shooting, $\tau = 10^{-6}$"),
              ("fd_N127", r"FD-Newton, $N = 127$"), ("B1", "PINN B1 (MLP)"), ("C1", "PINN C1 (ChNN)")]
    for k, (col, lab) in enumerate(series):
        if col in pred:
            P.plot_error(ax, t, pred[col].values - ex_y, label=lab, k=k, floor=1e-16)
    ax.set_ylim(1e-16, 1e3)
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=7))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(loc="upper center", fontsize=6.5, ncol=2)
    ax.set_title("pointwise error")
    P.label_panels(axes)
    ex.savefig(fig, "bvp_bratu")


IE_TAG_OFFSETS = {("F1", "I1"): (-13, -7), ("F1", "I2"): (5, 2), ("F2", "I1"): (-13, -7),
                  ("F2", "I2"): (5, 2), ("V", "I1"): (5, -7), ("V", "I2"): (-13, 2),
                  ("IDE", "I1"): (5, -7), ("IDE", "I2"): (-13, 2)}
IE_PANELS = [("F1", r"Fredholm, $K = xt$"), ("F2", r"Fredholm, $K = e^{xt}$"),
             ("V", r"Volterra, $K = t$"), ("IDE", r"IDE, $K = e^{-(x-t)}$")]


def fig_ie(ex, root, cost):
    d = Data(root, "integral_equations")
    cl, pinn = d.csv("classical"), d.csv("pinn")
    fig, axes = P.figure(2, 2, aspect=0.66, sharey=True, sharex=True)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.945))
    labels = {"Nystrom": r"Nystr$\ddot{\mathrm{o}}$m", "trapezoid": "trapezoidal", "RK45": "RK45 (ODE system)",
              "DOP853": "DOP853 (ODE system)"}
    for ax, (prob, title) in zip(axes.ravel(), IE_PANELS):
        sub = cl[cl.problem == prob].copy()
        sub["rel_l2"] = sub["rel_l2"].clip(lower=1e-16)
        for m in ("Nystrom", "trapezoid", "RK45", "DOP853"):
            if (sub.method == m).any():
                B.plot_classical(ax, rows(sub[sub.method == m]), m, label=labels[m], cost=cost)
        sub_p = pinn[pinn.problem == prob]
        B.plot_pinn(ax, rows(sub_p), cost=cost,
                    offsets=[IE_TAG_OFFSETS.get((prob, t), (4, 2)) for t in sub_p["tag"]])
        B.mark_targets(ax)
        ax.set_title(title)
        ax.set_xlim(5e-17, 3.0)
        log_axes(ax, xticks=(1e-16, 1e-12, 1e-8, 1e-4, 1.0))
    for ax in axes[1]:
        ax.set_xlabel(r"relative $L^2$ error")
    for ax in axes[:, 0]:
        ax.set_ylabel("CPU time [s]" if cost == "cpu_s" else "wall-clock time [s]")
    handles = {}
    for ax in axes.ravel():
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    order = [labels[m] for m in ("Nystrom", "trapezoid", "RK45", "DOP853")] + ["PINN"]
    order = [o for o in order if o in handles]
    fig.legend([handles[o] for o in order], order, loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=5)
    P.label_panels(axes)
    ex.savefig(fig, "ie_work_precision")


def fig_parametric(ex, root):
    d = Data(root, "parametric_logistic")
    summ, err = d.csv("summary"), d.csv("member_errors")
    q2 = summ[summ.tag == "Q2"].iloc[0]
    fig, axes = P.figure(1, 2, aspect=0.82)
    ax = axes[0]
    r = err["r"].values
    ax.semilogy(r, err["Q1_rel_l2"], color=P.COLORS["red"], ls="-", lw=1.1, label="PINN Q1 (32x3)")
    ax.semilogy(r, err["Q2_rel_l2"], color=P.COLORS["red"], ls="--", lw=1.3, label="PINN Q2 (64x3)")
    ax.semilogy(r, err["Q2_RK45_rel_l2"], color=B.STYLE["RK45"]["color"], ls="-.", lw=1.1,
                label=rf"RK45, rtol $= 10^{{{int(round(np.log10(q2['RK45_rtol'])))}}}$")
    ax.semilogy(r, err["Q2_cheb_rel_l2"], color=B.STYLE["Chebyshev"]["color"], ls=":", lw=1.4,
                label=rf"Chebyshev in $r$, $n_r = {int(q2['cheb_n_r'])}$")
    ax.set_xlabel("$r$")
    ax.set_ylabel(r"relative $L^2$ error of member")
    ax.legend(loc="best", fontsize=6.5)
    ax.set_title("accuracy across the family")
    ax = axes[1]
    n = np.logspace(0, 8, 200)
    lines = [
        ("RK45 (trajectory)", n * q2["RK45_traj_cpu_s"], B.STYLE["RK45"]["color"], "-"),
        ("RK45 (point)", n * q2["RK45_point_cpu_s"], B.STYLE["RK45"]["color"], "--"),
        ("PINN Q2 (trajectory)", q2["train_cpu_s"] + n * q2["eval_traj_cpu_s"], P.COLORS["red"], "-"),
        ("PINN Q2 (point)", q2["train_cpu_s"] + n * q2["eval_point_cpu_s"], P.COLORS["red"], "--"),
        ("Chebyshev (trajectory)", q2["cheb_build_cpu_s"] + n * q2["cheb_eval_traj_cpu_s"],
         B.STYLE["Chebyshev"]["color"], "-"),
        ("Chebyshev (point)", q2["cheb_build_cpu_s"] + n * q2["cheb_eval_point_cpu_s"],
         B.STYLE["Chebyshev"]["color"], "--"),
    ]
    for lab, y, c, ls in lines:
        ax.loglog(n, y, color=c, ls=ls, lw=1.1, label=lab)
    for q, col in (("traj", "RK45_traj_break_even"), ("point", "RK45_point_break_even")):
        ns = q2[col]
        if np.isfinite(ns) and ns < 1e8:
            ax.plot([ns], [ns * q2[f"RK45_{q}_cpu_s"]], "o", ms=4, mfc="white", mec="#1a1a1a", mew=0.8,
                    zorder=5)
    ax.set_xlabel("number of members evaluated")
    ax.set_ylabel("cumulative CPU time [s]")
    ax.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=8))
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10, numticks=5))
    ax.legend(loc="upper left", fontsize=6.2)
    ax.set_title("cumulative cost")
    P.label_panels(axes)
    ex.savefig(fig, "parametric_family")


def fig_certificate(ex, root):
    d = Data(root, "residual_certificate")
    c = d.csv("curves")
    fig, axes = P.figure(1, 2, aspect=0.8, sharey=True)
    for ax, key, title in ((axes[0], "pinn", "PINN P3"), (axes[1], "rk45", "RK45 + cubic Hermite")):
        t = c["t"].values
        ax.semilogy(t, np.maximum(c[f"{key}_error"].values, 1e-16), color=P.COLORS["ref"], lw=1.0,
                    label=r"true error $|e(t)|$")
        ax.semilogy(t, c[f"{key}_refined"].values, color=P.SERIES[0], ls="--", lw=1.3,
                    label="refined bound")
        ax.semilogy(t, c[f"{key}_sharp"].values, color=P.SERIES[2], ls=":", lw=1.5,
                    label=r"sharp bound ($\mu(t)$)")
        ax.semilogy(t, c[f"{key}_crude"].values, color=P.SERIES[1], ls="-.", lw=1.3,
                    label=r"crude bound ($L = r$)")
        ax.set_xlabel("$t$")
        ax.set_title(title)
        ax.set_ylim(1e-10, 1e-2)
    axes[0].set_ylabel("absolute error / bound")
    axes[0].legend(loc="lower right", fontsize=6.5)
    P.label_panels(axes)
    ex.savefig(fig, "residual_certificate")


# ----------------------------------------------------------------------------- tables
def tab_environment(root) -> str:
    info = None
    for s in SCRIPTS:
        d = Data(root, s)
        if d.exists():
            info = d.json().get("hardware", d.json().get("environment"))
            break
    if info is None:
        return "% no results found\n"
    virt = ""
    try:
        with open("/proc/cpuinfo") as fh:
            virt = " (virtual machine)" if " hypervisor" in fh.read() else ""
    except OSError:
        pass
    torch_v = str(info.get("torch", "")).split("+")[0]
    rows_ = [
        ("Processor", f"{info.get('cpu_model', 'unknown')}{virt}, {info.get('logical_cpus', '?')} logical CPUs"),
        ("Threads used", "1 (PyTorch and BLAS/OpenMP restricted to one thread)"),
        ("Operating system", info.get("platform", "").split("-with-")[0].replace("_", r"\_")),
        ("Software", f"Python {info.get('python')}, NumPy {info.get('numpy')}, "
                     f"SciPy {info.get('scipy')}, PyTorch {torch_v}"),
        ("Arithmetic", "IEEE double precision (float64) for all methods"),
        ("Cost measure", r"CPU time of the process (\code{time.process_time}); wall-clock time recorded"),
        ("Classical runs", f"median of {B.REPEATS} timed calls after one warm-up call"),
        ("PINN runs", r"one training run per configuration, seed 1234; Adam then L-BFGS"),
    ]
    body = "\n".join(rf"{a} & {b} \\" for a, b in rows_)
    return ("\\begin{tabularx}{\\textwidth}{@{}lX@{}}\n\\toprule\n" + body +
            "\n\\bottomrule\n\\end{tabularx}\n")


def tab_ivp_matched(root, cost) -> str:
    out = ["\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
           r" & & \multicolumn{2}{c}{error $\le 10^{-4}$} & \multicolumn{2}{c}{error $\le 10^{-6}$} & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"Problem & Method & time [s] & evaluations & time [s] & evaluations & best error \\",
           "\\midrule"]
    for j, (name, _) in enumerate(IVP_PANELS):
        d = Data(root, name)
        cl, pinn = rows(d.csv("classical")), rows(d.csv("pinn"))
        groups = {m: [r for r in cl if r["method"] == m] for m in B.IVP_METHODS}
        groups["PINN"] = pinn
        for i, (m, g) in enumerate(groups.items()):
            cells = []
            for tgt in B.TARGETS:
                best = B.time_to_reach(g, tgt, cost=cost)
                cells += [secs(best[cost]) if best else "---", count(best["evals"]) if best else "---"]
            first = rf"\multirow{{6}}{{*}}{{{IVP_NAMES[name]}}}" if i == 0 else ""
            out.append(rf"{first} & {m} & " + " & ".join(cells) + rf" & {num(B.best_error(g))} \\")
        if j < len(IVP_PANELS) - 1:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_ivp_pinn(root, cost) -> str:
    out = ["\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{@{}llrrrrrrrr@{}}", "\\toprule",
           r"Problem & Tag & network & param. & Adam & L-BFGS & rel.\ $L^2$ error & max error & $N_{\mathrm{res}}$ & time [s] \\",
           "\\midrule"]
    for j, (name, _) in enumerate(IVP_PANELS):
        d = Data(root, name)
        for i, r in enumerate(rows(d.csv("pinn"))):
            h = d.csv(f"history_{r['tag']}")
            net = f"{r['width']}x{r['depth']}"
            if name == "ivp_stiff":
                net += ", " + ("unif." if r.get("points") == "uniform" else "layer")
            if name == "ivp_vanderpol":
                net += ", " + (r"$t^2$" if r.get("multiplier") == "power" else r"$\varphi$")
            first = rf"\multirow{{{len(d.csv('pinn'))}}}{{*}}{{{IVP_NAMES[name]}}}" if i == 0 else ""
            out.append(rf"{first} & {r['tag']} & {net} & {intnum(r['params'])} & {intnum(r['adam_iters'])} & "
                       rf"{intnum(lbfgs_done(h, r['adam_iters']))} & {num(r['rel_l2'])} & {num(r['linf'])} & "
                       rf"{count(r['evals'])} & {secs(r[cost])} \\")
        if j < len(IVP_PANELS) - 1:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_bvp(root, cost) -> str:
    d = Data(root, "bvp_bratu")
    cl, pinn = rows(d.csv("classical")), rows(d.csv("pinn"))
    out = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
           r" & \multicolumn{2}{c}{error $\le 10^{-4}$} & \multicolumn{2}{c}{error $\le 10^{-6}$} & \\",
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
           r"Method & setting & time [s] & setting & time [s] & best error \\", "\\midrule"]
    def setting(method, r):
        if method == "solve_bvp":
            return rf"tol {num(float(r['setting']), 1)}"
        if method == "shooting":
            return rf"$\tau$ {num(float(r['setting']), 1)}"
        if method == "FD-Newton":
            return rf"$N$ = {intnum(float(r['setting']))}"
        return r["tag"]

    groups = {m: [r for r in cl if r["method"] == m] for m in ("solve_bvp", "shooting", "FD-Newton")}
    groups["PINN (MLP)"] = [r for r in pinn if r["arch"] == "MLP"]
    groups["PINN (ChNN)"] = [r for r in pinn if r["arch"] == "ChNN"]
    for m, g in groups.items():
        cells = []
        for tgt in B.TARGETS:
            best = B.time_to_reach(g, tgt, cost=cost)
            cells += ([setting(m, best), secs(best[cost])] if best else ["---", "---"])
        label = {"solve_bvp": "collocation"}.get(m, m)
        out.append(rf"{label} & " + " & ".join(cells) + rf" & {num(B.best_error(g))} \\")
    out.append("\\midrule")
    out.append(r"\multicolumn{6}{@{}l}{PINN configurations (rel.\ $L^2$ error, time [s]):} \\")
    for r in pinn:
        arch = (f"MLP {r['setting']}" if r["arch"] == "MLP" else f"ChNN, {r['setting']}")
        h = d.csv(f"history_{r['tag']}")
        out.append(rf"\quad {r['tag']}: {arch} & \multicolumn{{2}}{{l}}{{Adam {intnum(r['adam_iters'])}, "
                   rf"L-BFGS {intnum(lbfgs_done(h, r['adam_iters']))}}} & {num(r['rel_l2'])} & {secs(r[cost])} & \\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_ie(root, cost) -> str:
    d = Data(root, "integral_equations")
    cl, pinn = rows(d.csv("classical")), rows(d.csv("pinn"))
    names = {"Nystrom": r"Nystr\"om", "trapezoid": "trapezoidal", "RK45": "RK45 (ODE)", "DOP853": "DOP853 (ODE)"}
    titles = {"F1": r"Fredholm, $K=xt$", "F2": r"Fredholm, $K=e^{xt}$", "V": r"Volterra, $K=t$",
              "IDE": r"IDE, $K=e^{-(x-t)}$"}
    out = ["\\setlength{\\tabcolsep}{5pt}", "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
           r" & & \multicolumn{2}{c}{error $\le 10^{-4}$} & \multicolumn{2}{c}{error $\le 10^{-6}$} & \\",
           r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
           r"Problem & Method & setting & time [s] & setting & time [s] & best error \\", "\\midrule"]
    for j, prob in enumerate(("F1", "F2", "V", "IDE")):
        methods = [m for m in ("Nystrom", "trapezoid", "RK45", "DOP853") if any(r["problem"] == prob and r["method"] == m for r in cl)]
        groups = {m: [r for r in cl if r["problem"] == prob and r["method"] == m] for m in methods}
        groups["PINN"] = [r for r in pinn if r["problem"] == prob]
        for i, (m, g) in enumerate(groups.items()):
            cells = []
            for tgt in B.TARGETS:
                best = B.time_to_reach(g, tgt, cost=cost)
                if best is None:
                    cells += ["---", "---"]
                    continue
                if m == "PINN":
                    s = best["tag"]
                elif m in ("RK45", "DOP853"):
                    s = rf"rtol {num(float(best['setting']), 1)}"
                else:
                    s = rf"$n$ = {intnum(float(best['setting']))}"
                cells += [s, secs(best[cost])]
            first = rf"\multirow{{{len(groups)}}}{{*}}{{{titles[prob]}}}" if i == 0 else ""
            out.append(rf"{first} & {names.get(m, m)} & " + " & ".join(cells) + rf" & {num(B.best_error(g))} \\")
        if j < 3:
            out.append("\\midrule")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_parametric(root) -> str:
    d = Data(root, "parametric_logistic")
    s = {r["tag"]: r for r in rows(d.csv("summary"))}
    q = [s["Q1"], s["Q2"]]

    def be(x):
        return r"never" if not math.isfinite(x) else num(x)

    lines = [
        ("network (parameters)", [f"{r['width']}x{r['depth']} ({intnum(r['params'])})" for r in q]),
        (r"worst / median member error", [f"{num(r['worst_rel_l2'])} / {num(r['median_rel_l2'])}" for r in q]),
        ("training time [s]", [secs(r["train_cpu_s"]) for r in q]),
        ("evaluation per member [s]: trajectory / point",
         [f"{num(r['eval_traj_cpu_s'])} / {num(r['eval_point_cpu_s'])}" for r in q]),
        ("matched RK45 tolerance (worst error)", [f"{num(r['RK45_rtol'], 1)} ({num(r['RK45_worst_rel_l2'])})" for r in q]),
        ("RK45 per member [s]: trajectory / point", [f"{num(r['RK45_traj_cpu_s'])} / {num(r['RK45_point_cpu_s'])}" for r in q]),
        ("break-even vs RK45: trajectory / point", [f"{be(r['RK45_traj_break_even'])} / {be(r['RK45_point_break_even'])}" for r in q]),
        ("matched DOP853 tolerance (worst error)", [f"{num(r['DOP853_rtol'], 1)} ({num(r['DOP853_worst_rel_l2'])})" for r in q]),
        ("break-even vs DOP853: trajectory / point", [f"{be(r['DOP853_traj_break_even'])} / {be(r['DOP853_point_break_even'])}" for r in q]),
        (r"Chebyshev surrogate: $n_r$ (worst error)", [f"{int(r['cheb_n_r'])} ({num(r['cheb_worst_rel_l2'])})" for r in q]),
        ("Chebyshev build time [s]", [num(r["cheb_build_cpu_s"]) for r in q]),
    ]
    out = ["\\begin{tabular}{@{}lll@{}}", "\\toprule", r" & Q1 & Q2 \\", "\\midrule"]
    out += [rf"{a} & {b[0]} & {b[1]} \\" for a, b in lines]
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


def tab_certificate(root) -> str:
    d = Data(root, "residual_certificate")
    out = ["\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
           r"Approximation & $\max|\Res|$ & $\max|e|$ & refined bound & sharp bound & crude bound \\",
           "\\midrule"]
    for r in rows(d.csv("certificates")):
        name = r["approximation"].replace("Hermite", "cubic Hermite")
        cells = [f"{num(r[f'max_{b}_bound'])} ({r[f'effectivity_{b}']:.1f})" for b in ("refined", "sharp", "crude")]
        out.append(rf"{name} & {num(r['max_abs_residual'])} & {num(r['max_error'])} & " + " & ".join(cells) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- quoted numbers
def numbers_tex(root) -> str:
    """Accuracy numbers (not timings) quoted in the chapter text, as \\benchnum{key}."""
    vals = {}

    def put(key, x, digits=2):
        vals[key] = num(x, digits) if isinstance(x, (int, float, np.floating)) else str(x)

    for name, short in (("ivp_logistic", "log"), ("ivp_vanderpol", "vdp"), ("ivp_stiff", "stiff")):
        d = Data(root, name)
        if not d.exists():
            continue
        pinn = rows(d.csv("pinn"))
        cl = rows(d.csv("classical"))
        for r in pinn:
            put(f"{short}-{r['tag']}-err", r["rel_l2"])
            put(f"{short}-{r['tag']}-linf", r["linf"])
            put(f"{short}-{r['tag']}-evals", count(r["evals"]))
            if "rel_l2_outside_layer" in r:
                put(f"{short}-{r['tag']}-err-out", r["rel_l2_outside_layer"])
        put(f"{short}-pinn-best", B.best_error(pinn))
        put(f"{short}-classical-best", B.best_error(cl))
        for m in B.IVP_METHODS:
            g = [r for r in cl if r["method"] == m]
            for tgt, lab in zip(B.TARGETS, ("4", "6")):
                best = B.time_to_reach(g, tgt)
                if best:
                    put(f"{short}-{m}-evals{lab}", intnum(best["evals"]))
            for k in (3, 6, 9, 12):          # evaluations and error at rtol = 10^-k
                hit = [r for r in g if abs(np.log10(r["rtol"]) + k) < 1e-6]
                if hit:
                    put(f"{short}-{m}-nfev-{k}", intnum(hit[0]["nfev"]))
                    put(f"{short}-{m}-err-{k}", hit[0]["rel_l2"])
        js = d.json()
        if "reference_check_rel_l2" in js:
            put(f"{short}-refcheck", js["reference_check_rel_l2"])
    d = Data(root, "bvp_bratu")
    if d.exists():
        for r in rows(d.csv("pinn")):
            put(f"bratu-{r['tag']}-err", r["rel_l2"])
        for m, v in d.json()["best_error"].items():
            put(f"bratu-best-{m}", v)
        put("bratu-theta", f"{d.json()['problem']['theta']:.10f}")
    d = Data(root, "integral_equations")
    if d.exists():
        for r in rows(d.csv("pinn")):
            put(f"ie-{r['problem']}-{r['tag']}-err", r["rel_l2"])
        for r in rows(d.csv("matched")):
            put(f"ie-{r['problem']}-{r['method']}-best", r["best_rel_l2"])
        cl = rows(d.csv("classical"))
        for prob in ("F1", "F2"):
            for n in (1, 2, 4, 8):
                g = [r for r in cl if r["problem"] == prob and r["method"] == "Nystrom" and int(float(r["setting"])) == n]
                if g:
                    put(f"ie-{prob}-nystrom{n}", g[0]["rel_l2"])
    d = Data(root, "parametric_logistic")
    if d.exists():
        for r in rows(d.csv("summary")):
            t = r["tag"]
            put(f"par-{t}-worst", r["worst_rel_l2"])
            put(f"par-{t}-median", r["median_rel_l2"])
            put(f"par-{t}-rk45rtol", r["RK45_rtol"], 1)
            put(f"par-{t}-chebn", str(int(r["cheb_n_r"])))
            put(f"par-{t}-chebworst", r["cheb_worst_rel_l2"])
        cheb = rows(d.csv("chebyshev"))
        if cheb:
            put("par-cheb-best", min(c["worst_rel_l2"] for c in cheb))
            put("par-cheb-best-n", str(int(min(cheb, key=lambda c: c["worst_rel_l2"])["n_r"])))
    d = Data(root, "residual_certificate")
    if d.exists():
        for r in rows(d.csv("certificates")):
            k = "pinn" if r["approximation"].startswith("PINN") else "rk"
            put(f"cert-{k}-maxerr", r["max_error"])
            put(f"cert-{k}-maxres", r["max_abs_residual"])
            for b in ("refined", "sharp", "crude"):
                put(f"cert-{k}-{b}", r[f"max_{b}_bound"])
                put(f"cert-{k}-eff{b}", f"{r[f'effectivity_{b}']:.1f}")
            put(f"cert-{k}-holds", "yes" if r["bounds_hold_everywhere"] else "no")
            if "rk_steps" in r and r["rk_steps"] == r["rk_steps"]:
                put("cert-rk-steps", str(int(r["rk_steps"])))
    lines = ["% Generated by repo/chapters/ch10_benchmarking/write_book_assets.py -- do not edit.",
             "% Accuracy numbers quoted in Chapter 10; use as \\benchnum{key}.",
             r"\providecommand{\benchnum}[1]{\ifcsname benchnum@#1\endcsname\csname benchnum@#1\endcsname\else\textbf{??}\fi}"]
    lines += [rf"\expandafter\def\csname benchnum@{k}\endcsname{{{v}}}" for k, v in sorted(vals.items())]
    return "\n".join(lines) + "\n"


def dominated_fraction(root, cost, missing) -> dict:
    """Fraction of the PINN training-curve points (cost above the cheapest classical run) for
    which some classical run is both cheaper and at least as accurate."""
    out = {}
    for name, _ in IVP_PANELS:
        if name in missing:
            continue
        d = Data(root, name)
        cl = d.csv("classical")
        traj = d.csv("pinn_trajectories")
        c_min = cl[cost].min()
        best = traj.groupby("tag")["rel_l2"].cummin()          # running minimum per configuration
        pts = traj.assign(best=best)
        pts = pts[pts[cost] >= c_min]
        dom = [bool(((cl["rel_l2"] <= e) & (cl[cost] <= c)).any()) for e, c in zip(pts["best"], pts[cost])]
        out[name] = {"points": len(dom), "dominated": int(sum(dom)),
                     "fraction": float(np.mean(dom)) if dom else None}
    return out


def cost_ratios(root, cost, missing) -> list:
    """Matched cost of the PINN divided by that of the fastest classical method (for the prose,
    which states only orders of magnitude)."""
    out = []
    groups_by_problem = {}
    for name, _ in IVP_PANELS:
        if name not in missing:
            d = Data(root, name)
            cl = rows(d.csv("classical"))
            groups_by_problem[IVP_NAMES[name]] = ({m: [r for r in cl if r["method"] == m] for m in B.IVP_METHODS},
                                                  rows(d.csv("pinn")))
    if "bvp_bratu" not in missing:
        d = Data(root, "bvp_bratu")
        cl = rows(d.csv("classical"))
        groups_by_problem["Bratu"] = ({m: [r for r in cl if r["method"] == m] for m in ("solve_bvp", "shooting", "FD-Newton")},
                                      rows(d.csv("pinn")))
    if "integral_equations" not in missing:
        d = Data(root, "integral_equations")
        cl, pinn = rows(d.csv("classical")), rows(d.csv("pinn"))
        for prob in ("F1", "F2", "V", "IDE"):
            ms = {r["method"] for r in cl if r["problem"] == prob}
            groups_by_problem[prob] = ({m: [r for r in cl if r["problem"] == prob and r["method"] == m] for m in ms},
                                       [r for r in pinn if r["problem"] == prob])
    for prob, (classical, pinn) in groups_by_problem.items():
        for tgt in B.TARGETS:
            p = B.time_to_reach(pinn, tgt, cost=cost)
            best = [(m, B.time_to_reach(g, tgt, cost=cost)) for m, g in classical.items()]
            best = [(m, b) for m, b in best if b is not None]
            m_fast, b_fast = min(best, key=lambda mb: mb[1][cost]) if best else (None, None)
            out.append({"problem": prob, "target": tgt, "fastest": m_fast,
                        "ratio": (p[cost] / b_fast[cost]) if (p and b_fast) else None})
    return out


# ----------------------------------------------------------------------------- main
def run_examples(args) -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg")
    here = Path(__file__).resolve().parent
    for s in SCRIPTS:
        cmd = [sys.executable, "-u", str(here / f"{s}.py")]
        cmd += ["--quick"] if args.quick else []
        cmd += ["--resume"] if args.resume else []
        cmd += ["--no-figures"]
        print(f"==> {' '.join(cmd[2:])}", flush=True)
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
        if proc.returncode != 0:
            raise SystemExit(f"{s} failed with exit code {proc.returncode}")


def main():
    ex = Example(CHAPTER, "book_assets", description="Chapter 10 driver: figures and LaTeX tables",
                 extra_args=[(("--run",), {"action": "store_true", "help": "run all examples first"}),
                             (("--cost",), {"choices": ["cpu", "wall"], "default": "cpu"}),
                             (("--resume",), {"action": "store_true"}),
                             (("--book-dir",), {"default": None})])
    args = ex.args
    if args.run:
        run_examples(args)
    root = OUTPUT_ROOT / ("_quick" if args.quick else "") / CHAPTER
    cost = cost_key(args)
    missing = [s for s in SCRIPTS if not Data(root, s).exists()]
    tables_dir = ex.outdir / "tables"
    built = []
    if missing:
        ex.log(f"missing results for {missing}; run with --run (assets for these are skipped)")
    jobs = [("ivp_work_precision", ["ivp_logistic", "ivp_vanderpol", "ivp_stiff"], lambda: fig_ivp(ex, root, cost)),
            ("bvp_bratu", ["bvp_bratu"], lambda: fig_bvp(ex, root, cost)),
            ("ie_work_precision", ["integral_equations"], lambda: fig_ie(ex, root, cost)),
            ("parametric_family", ["parametric_logistic"], lambda: fig_parametric(ex, root)),
            ("residual_certificate", ["residual_certificate"], lambda: fig_certificate(ex, root))]
    for stem, needs, fn in jobs:
        if not any(n in missing for n in needs):
            fn()
            built.append(f"figures/{stem}.pdf")
    tables = [("tab_environment", [], lambda: tab_environment(root)),
              ("tab_ivp_matched", ["ivp_logistic", "ivp_vanderpol", "ivp_stiff"], lambda: tab_ivp_matched(root, cost)),
              ("tab_ivp_pinn", ["ivp_logistic", "ivp_vanderpol", "ivp_stiff"], lambda: tab_ivp_pinn(root, cost)),
              ("tab_bvp", ["bvp_bratu"], lambda: tab_bvp(root, cost)),
              ("tab_ie", ["integral_equations"], lambda: tab_ie(root, cost)),
              ("tab_parametric", ["parametric_logistic"], lambda: tab_parametric(root)),
              ("tab_certificate", ["residual_certificate"], lambda: tab_certificate(root)),
              ("numbers", [], lambda: numbers_tex(root))]
    for stem, needs, fn in tables:
        if not any(n in missing for n in needs):
            write(tables_dir / f"{stem}.tex", fn())
            built.append(f"tables/{stem}.tex")
    # copy to the book (never in quick mode)
    book = Path(args.book_dir) if args.book_dir else REPO_ROOT.parent / "book" / "figures" / "ch10"
    copied = []
    if not args.quick and not args.no_figures and book.parent.exists() and not missing:
        book.mkdir(parents=True, exist_ok=True)
        for b in built:
            src = ex.outdir / b
            if src.exists():
                shutil.copy2(src, book / src.name)
                copied.append(src.name)
        ex.log(f"copied {len(copied)} files to {book}")
    ratios = cost_ratios(root, cost, missing)
    dominated = dominated_fraction(root, cost, missing)
    ex.log(f"training-curve points dominated by a classical run: {dominated}")
    for r in ratios:
        ex.log(f"cost ratio PINN/fastest classical  {r['problem']:12s} target {r['target']:.0e}: "
               + (f"{r['ratio']:.2e} (vs {r['fastest']})" if r["ratio"] else "PINN did not reach the target"))
    ex.save_results({"cost_measure": args.cost, "cost_ratios": ratios, "dominated_training_points": dominated,
                     "built": built, "missing_inputs": missing,
                     "copied_to": str(book) if copied else None, "copied": copied,
                     "hardware": B.hardware_info()})


if __name__ == "__main__":
    main()
