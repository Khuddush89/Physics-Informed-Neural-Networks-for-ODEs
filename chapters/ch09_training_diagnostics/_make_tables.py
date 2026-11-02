"""Collect (and where needed regenerate) the LaTeX tables and figures of Chapter 9.

Every Chapter 9 script writes its table fragment ``tab_*.tex`` next to its ``results.json``.
Tables that contain run times are rebuilt here from ``results.json`` so that they can be
refreshed after re-running a script on an idle machine, without editing the book by hand.

    PYTHONPATH=. python chapters/ch09_training_diagnostics/_make_tables.py --dest ../book/figures/ch09
    PYTHONPATH=. python chapters/ch09_training_diagnostics/_make_tables.py --dest ../book/figures/ch09 --figures

The first command copies/regenerates all ``tab_*.tex``; ``--figures`` also copies the figure
PDFs used by the book. The leading underscore keeps this helper out of the smoke tests.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import fixed, latex_table, num  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ch09_training_diagnostics"
EXPERIMENTS = ["spectral_bias", "loss_imbalance", "trivial_solutions", "stiffness", "causality",
               "lorenz_chaos", "optimizers", "diagnostics"]
SCHED_LABEL = {"adam": "Adam", "adam+lbfgs": r"Adam$\to$L-BFGS",
               "short+lbfgs": r"short Adam$\to$L-BFGS", "lbfgs": "L-BFGS only"}


def table_optimizers(src: Path, dest: Path) -> Path:
    """tab_optimizers.tex from optimizers/results.json (CPU times as measured, plus ratios)."""
    res = json.loads((Path(src) / "results.json").read_text())
    rows, mids = [], []
    for k, r in enumerate(res["summary"]):
        if k > 0 and r["precision"] != res["summary"][k - 1]["precision"]:
            mids.append(len(rows))
        rows.append([r["precision"], SCHED_LABEL[r["schedule"]], f"{r['iterations']}", f"{r['evaluations']}",
                     num(r["final_loss"]), num(r["rel_l2"]), fixed(r["cpu_time_s"], 1),
                     fixed(r["cpu_time_relative"], 2)])
    return latex_table(Path(dest) / "tab_optimizers.tex", "llrrrrrr",
                       ["precision", "schedule", "iter.", "eval.", "final loss",
                        r"rel.\ $L^2$", "CPU (s)", "CPU ratio"], rows, midrules=mids,
                       comment="Chapter 9, optimizers.py (regenerate with _make_tables.py)")


FIGURES = {"spectral_bias": ["spectral_bias_summary"], "loss_imbalance": ["loss_imbalance_summary"],
           "trivial_solutions": ["trivial_solutions_summary"], "stiffness": ["stiffness_summary"],
           "causality": ["causality_summary"], "lorenz_chaos": ["lorenz_summary"],
           "optimizers": ["optimizers_summary"], "diagnostics": ["diagnostics_summary"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(OUT), help="outputs/ch09_training_diagnostics")
    ap.add_argument("--dest", default=str(OUT / "tables"))
    ap.add_argument("--figures", action="store_true", help="also copy the figure PDFs")
    args = ap.parse_args()
    src, dest = Path(args.src), Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in EXPERIMENTS:
        d = src / name
        if not (d / "results.json").exists():
            print(f"  [skip] {name}: no results.json")
            continue
        if name == "optimizers":
            print(f"  regenerated {table_optimizers(d, dest).name}")
        else:
            for tab in sorted(d.glob("tab_*.tex")):
                shutil.copy2(tab, dest / tab.name)
                print(f"  copied {tab.name}")
        if args.figures:
            for stem in FIGURES.get(name, []):
                pdf = d / "figures" / f"{stem}.pdf"
                if pdf.exists():
                    shutil.copy2(pdf, dest / pdf.name)
                    print(f"  copied {pdf.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
