#!/usr/bin/env python3
"""Collect every ``outputs/<chapter>/<example>/results.json`` into summary tables.

    python scripts/collect_results.py                     # -> outputs/results_summary.csv, docs/RESULTS.md
    python scripts/collect_results.py --latex path.tex     # also a LaTeX table (Appendix B of the book)

Fields are read defensively: ``rel_l2``/``linf`` (top level or in nested dicts),
``train_time_s`` (or the sum of nested ``train_time_s`` values), ``wall_time_total_s``,
``hyperparameters`` and ``seed``.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _walk(obj, key, acc):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, (int, float)) and not isinstance(v, bool):
                acc.append(float(v))
            else:
                _walk(v, key, acc)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, key, acc)


def best_error(res):
    for key in ("rel_l2", "relative_l2", "rel_l2_error", "rel_err"):
        if key in res and isinstance(res[key], (int, float)):
            return float(res[key])
    acc = []
    for key in ("rel_l2", "relative_l2", "rel_l2_error"):
        _walk(res, key, acc)
    acc = [a for a in acc if math.isfinite(a)]
    return min(acc) if acc else float("nan")


def train_time(res):
    if isinstance(res.get("train_time_s"), (int, float)):
        return float(res["train_time_s"])
    acc = []
    _walk(res, "train_time_s", acc)
    return float(sum(acc)) if acc else float("nan")


def hp_summary(res):
    hp = res.get("hyperparameters") or {}
    if not isinstance(hp, dict):
        return str(hp)[:60]
    keys = [("width", "w"), ("depth", "d"), ("hidden", "h"), ("degree", "deg"), ("collocation", "M"),
            ("n_colloc", "M"), ("adam_iters", "Adam"), ("lbfgs_iters", "L-BFGS")]
    parts = []
    for k, short in keys:
        if k in hp:
            parts.append(f"{short}={hp[k]}")
    return ", ".join(parts)


def collect(base: Path):
    rows = []
    for p in sorted(base.glob("ch*/*/results.json")):
        try:
            res = json.loads(p.read_text())
        except Exception:
            continue
        rows.append({
            "chapter": p.parent.parent.name,
            "example": p.parent.name,
            "description": res.get("description", ""),
            "seed": res.get("seed", ""),
            "rel_l2": best_error(res),
            "train_time_s": train_time(res),
            "wall_time_s": res.get("wall_time_total_s", float("nan")),
            "hyperparameters": hp_summary(res),
        })
    return rows


def fmt_e(x):
    return "--" if not (isinstance(x, float) and math.isfinite(x)) else f"{x:.1e}"


def fmt_t(x, latex=False):
    if not (isinstance(x, (int, float)) and math.isfinite(x)):
        return "--"
    if x < 0.05:
        return "$<0.1$" if latex else "< 0.1"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


# Typeset descriptions for the LaTeX table of the book (the scripts keep plain-text ones).
TEX_DESCRIPTIONS = {
    "first_pinn": r"First PINN for $y'=-y+\sin t$: soft vs.\ hard initial condition",
    "aposteriori_bounds": r"A posteriori residual-based error bounds versus true PINN errors",
    "architecture_comparison": r"Airy BVP: MLP, ChNN, LeNN, Fourier features and SIREN (hard Dirichlet conditions)",
    "soft_vs_hard": r"Airy oscillator: soft vs.\ hard conditions for an IVP and a Robin BVP (Fourier features)",
    "soft_vs_hard_chnn": r"The same with a ChNN (option \code{--network chnn})",
    "logistic_activation_study": r"Logistic ODE: activation study, soft vs.\ hard initial condition",
    "lotka_volterra": r"Lotka--Volterra system on $[0,6]$, PINN in log variables with hard initial conditions",
    "lotka_volterra_T15": r"The same on $[0,15]$ (option \code{--T 15})",
    "manual_polynomial_networks": r"Hand calculation for Chebyshev, Legendre and MLP networks ($y'=y$)",
    "manual_single_point": r"Hand calculation of a PINN loss, gradient and Adam step ($y'=y$)",
    "vanderpol": r"van der Pol oscillator ($\mu=0.5$): hard (bounded or power multiplier) vs.\ soft initial conditions",
    "boundary_layer": r"Boundary-layer BVP $\varepsilon y''+y'=1$: point sets and a layer-enriched network",
    "duffing_pinn": r"Duffing-type BVP with mixed conditions: ChNN and MLP PINNs, soft and hard conditions",
    "resonance_loss_floor": r"BVP without solution ($y''+y=0$, $y(0)=0$, $y(\pi)=1$): loss floor of PINNs",
    "jerk_tvp": r"Third-order TVP: LeNN and MLP with soft or hard terminal conditions",
    "lqr_forward_backward": r"LQR forward--backward Hamiltonian system solved by one PINN",
    "fredholm_degenerate": r"Fredholm IE with degenerate kernel $xt$: quadrature vs.\ auxiliary scalar",
    "helmholtz_filon": r"Helmholtz-type IE ($k=10$, $\lambda=6$): SIREN PINN with Filon panels",
    "manual_nystrom": r"Hand calculation: Nystr\"om method with one and two Gauss nodes, PINN quadrature residual",
    "volterra_apinn": r"Volterra IE $u=1+2\int_0^x tu\,\mathrm{d}t$: A-PINN vs.\ direct quadrature",
    "convolution": r"Convolution IDE: direct quadrature vs.\ internal variable (hard initial conditions)",
    "fredholm": r"Fredholm IDE: global quadrature vs.\ moment variable (hard initial condition)",
    "volterra_short": r"Volterra IDE on $[0,2]$: direct quadrature vs.\ A-PINN (hard initial conditions)",
    "causality": r"Harmonic oscillator on long intervals: vanilla PINN, causal weights, time-marching",
    "diagnostics": r"Validation residual vs.\ training loss; residual-based adaptive refinement",
    "lorenz_chaos": r"Lorenz system: Lyapunov exponent, twin experiment, global and time-marching PINNs",
    "loss_imbalance": r"BVP $y''=f$ with large $\omega$: fixed, rescaled, annealed and self-adaptive weights, hard conditions",
    "optimizers": r"Adam, Adam$\to$L-BFGS and L-BFGS in double and single precision (nonlinear pendulum)",
    "spectral_bias": r"Spectral bias: $\tanh$ MLP, Fourier features and SIREN for $y'=f$",
    "stiffness": r"Stiff linear IVP $y'=-\lambda(y-\cos t)$: PINN variants vs.\ Radau",
    "trivial_solutions": r"Logistic ODE: attraction of the equilibrium $y=0$ and remedies",
    "bvp_bratu": r"Bratu BVP: collocation, shooting and finite differences vs.\ PINN",
    "integral_equations": r"Fredholm, Volterra and IDE: quadrature methods vs.\ PINNs",
    "ivp_logistic": r"Logistic ODE: classical solvers vs.\ PINN (work--precision)",
    "ivp_stiff": r"Stiff linear system (stiffness ratio $1000$): classical solvers vs.\ PINN",
    "ivp_vanderpol": r"van der Pol oscillator ($\mu=0.5$): classical solvers vs.\ PINN (work--precision)",
    "parametric_logistic": r"Parametric logistic family: one PINN vs.\ repeated classical solves",
    "residual_certificate": r"A posteriori residual bounds for a PINN and an RK45 interpolant",
    "logistic_parameter_id": r"Logistic ODE: identify $(r,K)$ from noisy data, PINN vs.\ least squares",
    "logistic_weight_study": r"Logistic inverse PINN: effect of the residual weight $\lambda_r$",
    "lotka_volterra_partial": r"Lotka--Volterra system observed through the prey only: parameters and predator",
    "manual_inverse": r"Hand calculation: Gauss--Newton iteration vs.\ inverse-PINN loss and gradient",
    "source_identification": r"Unknown source $f(t)$ in $y'=-ky+f(t)$ from noisy data (two networks)",
    "uq_ensembles": r"Logistic inverse problem: ensembles, Gauss--Newton and Metropolis sampling",
    "volterra_parameter_id": r"Volterra equation $u=1+\lambda\int_0^x su\,\mathrm{d}s$: identify $\lambda$ from noisy data",
    "dae_nonlinear_resistor": r"Index-1 DAE $y'=-z$, $0=z+z^3-y$: hard (Newton layer) vs.\ penalty",
    "delay_method_of_steps": r"DDE $y'(t)=-y(t-1)$: global PINN vs.\ method of steps",
    "fractional_relaxation": r"Caputo relaxation $D^\alpha y=-y$: fPINN with the L1 scheme",
    "lane_emden": r"Lane--Emden equation $y''+(2/x)y'+y^m=0$: trial solution $1+(x/L)^2N(x)$",
    "stiff_robertson": r"Robertson kinetics on $[0,40]$: log-time PINN, vanilla PINN, Radau and BDF",
    "sturm_liouville": r"Sturm--Liouville eigenpairs: Rayleigh-quotient selection, trainable eigenvalue, deflation",
    "time_scales": r"$y^\Delta=py$ on $\mathbb{R}$, $\mathbb{Z}$ and $\mathbb{P}_{1,1}$: one PINN for continuous, discrete and hybrid time",
}


def tex_escape(s: str) -> str:
    table = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&", "%": r"\%", "#": r"\#",
             "$": r"\$", "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}
    return "".join(table.get(ch, ch) for ch in str(s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", default=None)
    ap.add_argument("--base", default=str(ROOT / "outputs"))
    args = ap.parse_args()
    rows = collect(Path(args.base))
    import csv

    out_csv = ROOT / "outputs" / "results_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["chapter"])
        w.writeheader()
        w.writerows(rows)
    WRITERS = ("write_tables", "tables", "book_assets")
    ex_rows = [r for r in rows if r["example"] not in WRITERS]
    md = ["# Examples and runtimes", "",
          "Wall-clock time of each script (full run, one CPU thread, double precision).", "",
          "| chapter | script | example | runtime (s) | seed |", "|---|---|---|---|---|"]
    for r in ex_rows:
        md.append(f"| {r['chapter']} | `{r['example']}` | {r['description']} | {fmt_t(r['wall_time_s'])} | {r['seed']} |")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(md) + "\n")
    if args.latex:
        total = sum(r["wall_time_s"] for r in ex_rows if isinstance(r["wall_time_s"], (int, float)))
        lines = [r"\begin{small}", r"\setlength{\tabcolsep}{4pt}",
                 r"\begin{longtable}{@{}r>{\raggedright\arraybackslash}p{0.33\textwidth}>{\raggedright\arraybackslash}p{0.43\textwidth}r@{}}",
                 r"\caption{Examples of the book and the wall-clock time of each script (full run, one CPU thread of the processor in \cref{app:tab:versions}, double precision). Total: " + f"{total / 60:.0f}" + r" minutes.}\label{app:tab:runs}\\",
                 r"\toprule Ch. & Script & Example & Time (s)\\ \midrule \endfirsthead",
                 r"\toprule Ch. & Script & Example & Time (s)\\ \midrule \endhead",
                 r"\bottomrule \endfoot"]
        for r in ex_rows:
            ch = int(r["chapter"].split("_")[0].replace("ch", ""))
            desc = TEX_DESCRIPTIONS.get(r["example"], tex_escape(r["description"]))
            lines.append(f"{ch} & \\code{{{r['example']}}} & {desc} & {fmt_t(r['wall_time_s'], latex=True)}\\\\")
        lines += [r"\end{longtable}", r"\end{small}"]
        Path(args.latex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.latex).write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} result files collected -> {out_csv.relative_to(ROOT)}, docs/RESULTS.md"
          + (f", {args.latex}" if args.latex else ""))


if __name__ == "__main__":
    main()
