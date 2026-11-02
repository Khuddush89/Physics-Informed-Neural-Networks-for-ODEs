"""Regenerate the Chapter 3 tables (LaTeX) from results.json and copy the figures into the book.

Reads   outputs/ch03_architectures/{feature_conditioning,architecture_comparison,soft_vs_hard,
        soft_vs_hard_chnn}/  (the last one from ``soft_vs_hard.py --network chnn --outdir ...``)
Writes  <book>/figures/ch03/tab_conditioning.tex, tab_architectures.tex, tab_softhard.tex
        <book>/figures/ch03/features_conditioning.pdf, architecture_comparison.pdf, soft_vs_hard.pdf

Typical use (from the repository root, after running the three experiment scripts):

    PYTHONPATH=. python chapters/ch03_architectures/_make_book_assets.py --book ../book

The timing columns are taken from ``train_time_s`` in results.json, so re-running the
experiment scripts on an idle machine and then this script refreshes every number in the
tables without editing the chapter.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "ch03_architectures"
FIGURES = {"feature_conditioning": "features_conditioning.pdf",
           "architecture_comparison": "architecture_comparison.pdf",
           "soft_vs_hard": "soft_vs_hard.pdf"}
KAPPA_CAP = 1e15


def sci(x: float, digits: int = 2) -> str:
    """siunitx number with `digits` significant digits, e.g. \\num{3.7e-4}."""
    if x == 0 or not math.isfinite(x):
        return r"\num{0}" if x == 0 else "--"
    s = f"{x:.{digits - 1}e}"
    mant, exp = s.split("e")
    return rf"\num{{{mant}e{int(exp)}}}"


def load(name: str, out: Path = OUT) -> dict:
    p = out / name / "results.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run chapters/ch03_architectures/{name}.py first")
    with open(p) as fh:
        res = json.load(fh)
    if res.get("quick"):
        print(f"WARNING: {p} comes from a --quick run; numbers are not meaningful")
    return res


def table_conditioning(res: dict) -> str:
    rows = res["condition_numbers_table"]
    head = (r"\begin{tabular}{rcccc}" "\n" r"\toprule" "\n"
            r"degree $m$ & $t^k$, $t\in[0,10]$ & $\xi^k$ & $T_k(\xi)$ & $P_k(\xi)$ \\" "\n" r"\midrule")
    body = []
    for r in rows:
        cells = []
        for b in ("monomial_t", "monomial_xi", "chebyshev", "legendre"):
            v = r[b]
            cells.append(r"$>10^{15}$" if v > KAPPA_CAP else sci(v, 2))
        body.append(f"{r['degree']} & " + " & ".join(cells) + r" \\")
    return "\n".join([head, *body, r"\bottomrule", r"\end{tabular}", ""])


ARCH_LABEL = {"MLP": "MLP", "ChNN": r"ChNN ($m=10$)", "LeNN": r"LeNN ($m=10$)",
              "FF": r"Fourier features", "SIREN6": r"SIREN ($\omega_0=6$)",
              "SIREN30": r"SIREN ($\omega_0=30$)"}


def table_architectures(res: dict) -> str:
    rows = res["summary"]
    t_ref = next(r["train_time_s"] for r in rows if r["architecture"] == "MLP")
    head = (r"\begin{tabular}{lrccccc}" "\n" r"\toprule" "\n"
            r"network & $P$ & rel.\ $L^2$ error & $L^\infty$ error & final loss & time [s] & rel.\ time \\"
            "\n" r"\midrule")
    body = [f"{ARCH_LABEL.get(r['architecture'], r['architecture'])} & {r['parameters']} & "
            f"{sci(r['rel_l2'])} & {sci(r['linf'])} & {sci(r['final_loss'])} & "
            f"{r['train_time_s']:.0f} & {r['train_time_s'] / t_ref:.2f}" + r" \\" for r in rows]
    return "\n".join([head, *body, r"\bottomrule", r"\end{tabular}", ""])


SH_PROBLEM = {"ivp": r"IVP, $[0,6]$", "ivp10": r"IVP, $[0,10]$", "robin": r"Robin BVP, $[0,10]$"}
SH_VARIANT = {"soft": r"soft ($\lambda=1$)", "hard_tau1": r"hard, $\tau=1$", "hard_tauT": r"hard, $\tau=T$",
              "hard": r"hard, \eqref{arch:eq:dirichlet-robin}"}


def table_softhard(res: dict, res_chnn: dict | None = None) -> str:
    rows = res["summary"]
    chnn = {(r["problem"], r["variant"]): r for r in (res_chnn or {}).get("summary", [])}
    extra_head = r" & \multicolumn{2}{c}{ChNN}" if chnn else ""
    cols = "ll" + ("cccccc" if chnn else "ccccc")
    head = [r"\begin{tabular}{" + cols + "}", r"\toprule",
            r" & & \multicolumn{4}{c}{Fourier-feature network} & " + extra_head.lstrip(" &") + r" \\"
            if chnn else r" & & \multicolumn{5}{c}{Fourier-feature network} \\",
            (r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}" if chnn else r"\cmidrule(lr){3-7}"),
            r"problem & conditions & rel.\ $L^2$ & max.\ viol. & loss & time [s]"
            + (r" & rel.\ $L^2$ & max.\ viol. \\" if chnn else r" & $L^\infty$ \\"),
            r"\midrule"]
    body, last = [], None
    for r in rows:
        prob = SH_PROBLEM[r["problem"]] if r["problem"] != last else ""
        if last is not None and r["problem"] != last:
            body.append(r"\midrule")
        last = r["problem"]
        cells = [prob, SH_VARIANT[r["variant"]], sci(r["rel_l2"]), sci(r["max_condition_violation"]),
                 sci(r["final_loss"]), f"{r['train_time_s']:.0f}"]
        if chnn:
            c = chnn.get((r["problem"], r["variant"]))
            cells += [sci(c["rel_l2"]), sci(c["max_condition_violation"])] if c else ["--", "--"]
        else:
            cells += [sci(r["linf"])]
        body.append(" & ".join(cells) + r" \\")
    return "\n".join([*head, *body, r"\bottomrule", r"\end{tabular}", ""])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    book_default = REPO.parent / "book"
    ap.add_argument("--book", default=str(book_default) if book_default.is_dir() else None,
                    help="path of the book directory (default: ../book if it exists, else outputs/)")
    ap.add_argument("--outputs", default=str(OUT), help="folder with the script outputs")
    args = ap.parse_args()
    out = Path(args.outputs)
    dest = (Path(args.book) / "figures" / "ch03") if args.book else (out / "book_assets")
    dest.mkdir(parents=True, exist_ok=True)
    makers = {"feature_conditioning": ("tab_conditioning.tex", table_conditioning),
              "architecture_comparison": ("tab_architectures.tex", table_architectures),
              "soft_vs_hard": ("tab_softhard.tex", table_softhard)}
    for name, (fname, fn) in makers.items():
        try:
            res = load(name, out)
        except FileNotFoundError as err:
            print(f"skipped {fname}: {err}")
            continue
        if name == "soft_vs_hard":
            try:
                res_chnn = load("soft_vs_hard_chnn", out)
            except FileNotFoundError:
                res_chnn = None
            (dest / fname).write_text(fn(res, res_chnn))
        else:
            (dest / fname).write_text(fn(res))
        print(f"wrote {dest / fname}")
        fig = out / name / "figures" / FIGURES[name]
        if fig.exists():
            shutil.copy2(fig, dest / FIGURES[name])
            print(f"copied {fig.name}")
        else:
            print(f"figure {fig} missing")


if __name__ == "__main__":
    main()
