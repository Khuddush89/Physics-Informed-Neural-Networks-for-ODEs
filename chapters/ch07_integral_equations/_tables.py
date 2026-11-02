"""Write the LaTeX tables of Chapter 7 from the results files of the example scripts.

Usage (from the repository root, after the full runs of the four scripts):

    PYTHONPATH=. python chapters/ch07_integral_equations/_tables.py [--outdir ../book/figures/ch07]

Reads   outputs/ch07_integral_equations/<example>/results.json
Writes  tab_fredholm.tex, tab_volterra.tex, tab_helmholtz.tex, tab_filon.tex, tab_aposteriori.tex
        (booktabs tabular bodies; the book wraps them in table floats with captions).

The default output folder is ../book/figures/ch07 when it exists (book and repository side by
side), otherwise outputs/ch07_integral_equations/tables.  Times are copied from the results
files as measured; only their orders of magnitude are meaningful on a shared machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ch07_integral_equations"


def load(name: str) -> dict:
    with open(OUT / name / "results.json") as fh:
        return json.load(fh)


def num(x: float, digits: int = 2) -> str:
    """siunitx number in scientific notation, e.g. \\num{1.34e-4}."""
    if x == 0:
        return "\\num{0}"
    m, e = f"{x:.{digits - 1}e}".split("e")
    return f"\\num{{{m}e{int(e)}}}"


def sec(t: float) -> str:
    """Times: two significant digits in scientific notation."""
    return num(t, 2)


def fixed(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}f}"


def write(path: Path, lines) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path}")


# ----------------------------------------------------------------------------- tables
def table_fredholm(outdir: Path) -> None:
    r = load("fredholm_degenerate")
    lab = {"quad": "PINN, quadrature", "aux": "PINN, auxiliary scalar", "quad1": "PINN, quadrature"}
    L = ["\\begin{tabular}{@{}llcccc@{}}", "\\toprule",
         "Method & Rule & rel.\\ $L^2$ error & max.\\ error & slope & time (s) \\\\", "\\midrule"]
    for v in ("quad", "aux", "quad1"):
        p = r["pinn"][v]
        L.append(f"{lab[v]} & $Q={p['n_quad']}$ & {num(p['rel_l2'])} & {num(p['linf'])} & "
                 f"{fixed(p['slope'])} & {sec(p['train_time_s'])} \\\\")
    L.append("\\midrule")
    for row in r["nystrom"]:
        L.append(f"Nystr\\\"om & $n={row['n']}$ & {num(row['rel_l2'])} & {num(row['linf'])} & "
                 f"{fixed(row['slope'])} & {sec(row['time_s'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write(outdir / "tab_fredholm.tex", L)


def table_volterra(outdir: Path) -> None:
    r = load("volterra_apinn")
    L = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
         "Method & Discretization & rel.\\ $L^2$ error & max.\\ error & time (s) \\\\", "\\midrule"]
    p = r["pinn"]["apinn"]
    L.append(f"A-PINN & $M={r['hyperparameters']['M']}$, no quadrature & {num(p['rel_l2'])} & "
             f"{num(p['linf'])} & {sec(p['train_time_s'])} \\\\")
    p = r["pinn"]["quad"]
    L.append(f"PINN, direct quadrature & $M={r['hyperparameters']['M']}$, "
             f"{r['hyperparameters']['gauss_nodes_per_point']} Gauss nodes per point & {num(p['rel_l2'])} & "
             f"{num(p['linf'])} & {sec(p['train_time_s'])} \\\\")
    L.append("\\midrule")
    for row in r["trapezoid"]:
        L.append(f"trapezoidal rule & $n={row['n']}$ & {num(row['rel_l2'])} & {num(row['linf'])} & "
                 f"{sec(row['time_s'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write(outdir / "tab_volterra.tex", L)


def table_helmholtz(outdir: Path) -> None:
    r = load("helmholtz_filon")
    p = r["pinn"]
    L = ["\\begin{tabular}{@{}lrrccc@{}}", "\\toprule",
         "Method & Parameter & Nodes & rel.\\ $L^2$ error & max.\\ error & time (s) \\\\", "\\midrule",
         f"PINN (SIREN), Filon panels & $P={p['panels']}$ & {p['quadrature_nodes']} & {num(p['rel_l2'])} & "
         f"{num(p['linf'])} & {sec(p['train_time_s'])} \\\\", "\\midrule"]
    groups = [("Nystrom, Gauss-Legendre", "Nystr\\\"om, Gauss--Legendre", "n", (40, 160, 640)),
              ("Nystrom, Simpson panels", "Nystr\\\"om, Simpson panels", "P", (5, 10, 20)),
              ("Nystrom, Filon panels", "Nystr\\\"om, Filon panels", "P", (5, 10, 20, 40))]
    for key, name, sym, keep in groups:
        rows = [row for row in r["classical"] if row["method"] == key and row["param"] in keep]
        for i, row in enumerate(rows):
            L.append(f"{name if i == 0 else ''} & ${sym}={row['param']}$ & {row['nodes']} & {num(row['rel_l2'])} & "
                     f"{num(row['linf'])} & {sec(row['time_s'])} \\\\")
        L.append("\\midrule" if key != groups[-1][0] else "\\bottomrule")
    L.append("\\end{tabular}")
    write(outdir / "tab_helmholtz.tex", L)


def table_filon(outdir: Path) -> None:
    r = load("filon_weights")
    L = ["\\begin{tabular}{@{}lccc@{}}", "\\toprule",
         "Range of $|\\theta|$ & closed form & closed form + 3-term Taylor & series for $|\\theta|<1$ \\\\",
         "\\midrule"]
    bins = r["weights_error_bins"]
    for i, b in enumerate(bins):
        lo, hi = int(round(np.log10(b["theta_min"]))), int(round(np.log10(b["theta_max"])))
        close = "]" if i == len(bins) - 1 else ")"
        rng = f"$[10^{{{lo}}}, 10^{{{hi}}}{close}$"
        L.append(f"{rng} & {num(b['closed'], 1)} & {num(b['taylor3'], 1)} & {num(b['stable'], 1)} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write(outdir / "tab_filon.tex", L)


def table_aposteriori(outdir: Path) -> None:
    f, v, h = load("fredholm_degenerate"), load("volterra_apinn"), load("helmholtz_filon")
    rows = [("Fredholm (\\cref{ie:sec:fredholm}), quadrature, $Q=2$", "$4/3$", f["pinn"]["quad"]),
            ("Volterra (\\cref{ie:sec:volterra-example}), A-PINN", "$e$", v["pinn"]["apinn"]),
            ("Volterra (\\cref{ie:sec:volterra-example}), direct quadrature", "$e$", v["pinn"]["quad"]),
            ("Helmholtz-type (\\cref{ie:sec:helmholtz-pinn})", "$1/0.7$", h["pinn"])]
    L = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
         "Example & $\\norm{r}_\\infty$ & constant & bound & actual $\\norm{u_{\\btheta}-u}_\\infty$ \\\\",
         "\\midrule"]
    for name, const, p in rows:
        L.append(f"{name} & {num(p['residual_inf'])} & {const} & {num(p['error_bound'])} & {num(p['linf'])} \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    write(outdir / "tab_aposteriori.tex", L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = ROOT.parent / "book" / "figures" / "ch07"
    ap.add_argument("--outdir", default=str(default if default.exists() else OUT / "tables"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for fn in (table_fredholm, table_volterra, table_helmholtz, table_filon, table_aposteriori):
        try:
            fn(outdir)
        except FileNotFoundError as exc:
            print(f"skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    main()
