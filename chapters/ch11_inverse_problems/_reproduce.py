#!/usr/bin/env python3
"""Reproduce all numbers, tables and figures of Chapter 11 and copy them into the book.

Run from the repository root (single thread, about 15-20 min of CPU time in total):

    PYTHONPATH=. python chapters/ch11_inverse_problems/_reproduce.py --book ../book
    PYTHONPATH=. python chapters/ch11_inverse_problems/_reproduce.py --quick        # smoke test

The example scripts are run one after another. A script that is killed by the operating
system (for instance by the out-of-memory killer on a shared machine) is restarted with
``--resume``, which reuses the fits it had already completed; every fit seeds its own random
number generators, so the results are identical to those of an uninterrupted run. Finally the
figure PDFs and the generated LaTeX tables (``tab_*.tex``, which contain all CPU times quoted
in the chapter) are copied to ``<book>/figures/ch11/``.

Options: ``--only name1,name2`` runs a subset (script stems); ``--resume`` reuses cached fits
on the first attempt as well; ``--copy-only`` skips the runs and only copies the assets.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHAPTER = HERE.name
SCRIPTS = ["logistic_parameter_id", "logistic_weight_study", "lotka_volterra_partial",
           "source_identification", "volterra_parameter_id", "uq_ensembles"]


def run(stem: str, quick: bool, resume_first: bool, attempts: int = 40) -> bool:
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for k in range(attempts):
        cmd = [sys.executable, str(HERE / f"{stem}.py")] + (["--quick"] if quick else [])
        if k > 0 or resume_first:
            cmd.append("--resume")
        t0 = time.perf_counter()
        print(f"==> {stem} (attempt {k + 1}{', resume' if '--resume' in cmd else ''})", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, env=env)
        print(f"    exit code {proc.returncode} after {time.perf_counter() - t0:.0f} s wall", flush=True)
        if proc.returncode == 0:
            return True
        if proc.returncode not in (-9, 137, -15, 143):     # not killed: a genuine error
            return False
    return False


def copy_assets(book: Path, quick: bool) -> None:
    out = ROOT / "outputs" / ("_quick" if quick else "") / CHAPTER
    dest = book / "figures" / "ch11"
    dest.mkdir(parents=True, exist_ok=True)
    for stem in SCRIPTS:
        folder = out / stem
        pdf = folder / "figures" / f"{stem}.pdf"
        if pdf.exists():
            shutil.copy2(pdf, dest / pdf.name)
            print(f"copied {pdf.relative_to(ROOT)}")
        for tab in sorted(folder.glob("tab_*.tex")):
            shutil.copy2(tab, dest / tab.name)
            print(f"copied {tab.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--book", default=None, help="path of the book folder (default: no copy)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--copy-only", action="store_true")
    args = ap.parse_args()
    stems = SCRIPTS if not args.only else [s for s in SCRIPTS if s in args.only.split(",")]
    failed = []
    if not args.copy_only:
        for stem in stems:
            if not run(stem, args.quick, args.resume):
                failed.append(stem)
    if args.book:
        copy_assets(Path(args.book).resolve(), args.quick)
    if failed:
        print("FAILED: " + ", ".join(failed))
        return 1
    print("all scripts finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
