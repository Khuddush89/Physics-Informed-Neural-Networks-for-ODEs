"""Write the Chapter 8 comparison tables (PINN versus classical solvers) as LaTeX from the
results.json files of the other scripts in this folder, and optionally copy tables and
figures into the book.

Book: Chapter 8, Tables ide:tab:volterra, ide:tab:fredholm, ide:tab:convolution (the book
\\input's the generated files figures/ch08/tab_*.tex) and Figures ide:fig:volterra,
ide:fig:growth, ide:fig:fredholm, ide:fig:convolution.

The timing column is the process CPU time (training, or one classical solve), which is far
less sensitive to the load of a shared machine than wall-clock time; ``--time wall`` uses the
wall-clock times stored in the same files instead.

Run (after the four example scripts):
    PYTHONPATH=. python chapters/ch08_integro_differential/write_tables.py
    PYTHONPATH=. python chapters/ch08_integro_differential/write_tables.py --book ../book/figures/ch08
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

from pinnode import Example
from pinnode.utils import OUTPUT_ROOT

CHAPTER = "ch08_integro_differential"
FIGURES = {"volterra_short": "volterra_short", "volterra_growth": "volterra_growth",
           "fredholm": "fredholm", "convolution": "convolution"}


def sci(v, sig: int = 2) -> str:
    """\\num{m.me-k} with ``sig`` significant digits; '--' for missing values."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "--"
    v = float(v)
    if v == 0.0:
        return "\\num{0}"
    e = math.floor(math.log10(abs(v)))
    m = round(v / 10 ** e, sig - 1)
    if abs(m) >= 10.0:
        m, e = m / 10.0, e + 1
    return f"\\num{{{m:.{sig - 1}f}e{e}}}" if e != 0 else f"\\num{{{m:.{sig - 1}f}}}"


def tsec(v) -> str:
    """Times: plain decimals above 1 s, scientific notation below."""
    if v is None:
        return "--"
    v = float(v)
    if v >= 100:
        return f"\\num{{{v:.0f}}}"
    if v >= 1:
        return f"\\num{{{v:.1f}}}"
    return sci(v)


def load(root: Path, name: str):
    p = root / CHAPTER / name / "results.json"
    if not p.exists():
        print(f"  [missing] {p}")
        return None
    with open(p) as fh:
        return json.load(fh)


def pinn_time(entry, mode):
    return entry.get("cpu_time_s") if mode == "cpu" else entry.get("train_time_s")


def cls_time(row, mode):
    return row.get("cpu_time_s") if mode == "cpu" else row.get("wall_time_s")


def classical_rows(res, mode, cols):
    out = []
    for row in (res or {}).get("classical", []):
        m = row["method"]
        if m.startswith("trapezoidal"):
            label = f"trapezoidal method, $N={row['N']}$"
        elif m.startswith("DOP853"):
            label = ("DOP853, $(u,A)$ system" if "(u, A)" in m else
                     "DOP853, $(u,z)$ system" if "(u, z)" in m else "DOP853")
        elif m.startswith("Nystrom"):
            label = f"Nystr\\\"om, $n={row['n']}$ nodes"
        else:
            label = m
        out.append((label, [cols[c](row) for c in cols], tsec(cls_time(row, mode))))
    return out


def volterra_table(short, growth, mode):
    lines = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
             "Interval & Method & rel.\\ $L^2$ error & max.\\ rel.\\ error & time (s) \\\\",
             "\\midrule"]
    cols = {"rel": lambda r: sci(r.get("rel_l2")), "max": lambda r: sci(r.get("max_rel"))}

    def block(interval, res, pinn_labels):
        rows = []
        for key, lab in pinn_labels:
            e = (res or {}).get("pinn", {}).get(key)
            if e is None:
                rows.append((lab, ["--", "--"], "--"))
            else:
                rows.append((lab, [sci(e["errors"]["rel_l2"]), sci(e["errors"]["max_rel"])],
                             tsec(pinn_time(e, mode))))
        rows += classical_rows(res, mode, cols)
        out = []
        for i, (lab, vals, t) in enumerate(rows):
            first = f"\\multirow{{{len(rows)}}}{{*}}{{{interval}}}" if i == 0 else ""
            out.append(f"{first} & {lab} & {vals[0]} & {vals[1]} & {t} \\\\")
        return out

    lines += block("$[0,2]$", short, [("direct", "PINN, direct quadrature"),
                                      ("apinn", "A-PINN, $(u,A)$")])
    lines.append("\\midrule")
    lines += block("$[0,5]$", growth, [("unscaled", "A-PINN, $(u,A)$ unscaled"),
                                       ("exp", "A-PINN, $u=e^{x/2}v$"),
                                       ("log", "A-PINN, $(w,a)$ log variables"),
                                       ("logquad", "PINN, quadrature in $w=\\log u$"),
                                       ("bifactor", "Bi-factorized PINN (\\cref{ide:rem:bifactor})")])
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def fredholm_table(res, mode):
    lines = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             "Method & rel.\\ $L^2$ error & max.\\ error & $|C-8/21|$ & time (s) \\\\", "\\midrule"]
    for key, lab in (("direct", "PINN, global quadrature"), ("moment", "PINN, moment variable $C$")):
        e = (res or {}).get("pinn", {}).get(key)
        if e is None:
            lines.append(f"{lab} & -- & -- & -- & -- \\\\")
            continue
        lines.append(f"{lab} & {sci(e['errors']['rel_l2'])} & {sci(e['errors']['linf'])} & "
                     f"{sci(e['C_error'])} & {tsec(pinn_time(e, mode))} \\\\")
    cols = {"rel": lambda r: sci(r.get("rel_l2")), "max": lambda r: sci(r.get("linf")),
            "C": lambda r: sci(r.get("C_error"))}
    for lab, vals, t in classical_rows(res, mode, cols):
        lines.append(f"{lab} & {vals[0]} & {vals[1]} & {vals[2]} & {t} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def convolution_table(res, mode):
    lines = ["\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             "Method & rel.\\ $L^2$ error & max.\\ error & time (s) & ms per gradient \\\\",
             "\\midrule"]
    for key, lab in (("direct", "PINN, direct quadrature"), ("internal", "A-PINN, internal variable $z$")):
        e = (res or {}).get("pinn", {}).get(key)
        if e is None:
            lines.append(f"{lab} & -- & -- & -- & -- \\\\")
            continue
        lines.append(f"{lab} & {sci(e['errors']['rel_l2'])} & {sci(e['errors']['linf'])} & "
                     f"{tsec(pinn_time(e, mode))} & {sci(e['cpu_per_loss_gradient_ms'])} \\\\")
    cols = {"rel": lambda r: sci(r.get("rel_l2")), "max": lambda r: sci(r.get("linf"))}
    for lab, vals, t in classical_rows(res, mode, cols):
        lines.append(f"{lab} & {vals[0]} & {vals[1]} & {t} & -- \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ex = Example(CHAPTER, "write_tables", description="LaTeX tables for Chapter 8 from results.json",
                 extra_args=[(("--book",), {"default": None,
                                            "help": "copy tables and figure PDFs to this folder"}),
                             (("--time",), {"default": "cpu", "choices": ["cpu", "wall"]})])
    root = OUTPUT_ROOT / "_quick" if ex.quick else OUTPUT_ROOT
    res = {n: load(root, n) for n in FIGURES}
    mode = ex.args.time
    tables = {"tab_volterra.tex": volterra_table(res["volterra_short"], res["volterra_growth"], mode),
              "tab_fredholm.tex": fredholm_table(res["fredholm"], mode),
              "tab_convolution.tex": convolution_table(res["convolution"], mode)}
    header = ("% Generated by repo/chapters/ch08_integro_differential/write_tables.py "
              f"(time column: {'process CPU time' if mode == 'cpu' else 'wall-clock time'}).\n"
              "% Do not edit by hand; re-run the script after re-running the examples.\n")
    written = []
    for fname, body in tables.items():
        p = ex.path(fname)
        p.write_text(header + body)
        written.append(str(p))
    copied = []
    if ex.args.book:
        dest = Path(ex.args.book)
        dest.mkdir(parents=True, exist_ok=True)
        for fname in tables:
            shutil.copy2(ex.path(fname), dest / fname)
            copied.append(str(dest / fname))
        for name, stem in FIGURES.items():
            pdf = root / CHAPTER / name / "figures" / f"{stem}.pdf"
            if pdf.exists():
                shutil.copy2(pdf, dest / f"{stem}.pdf")
                copied.append(str(dest / f"{stem}.pdf"))
            else:
                print(f"  [missing figure] {pdf}")
    ex.save_results({"tables": written, "copied": copied, "time_column": mode,
                     "sources_found": {n: r is not None for n, r in res.items()}})
    for w in written + copied:
        ex.log(f"wrote {w}")


if __name__ == "__main__":
    main()
