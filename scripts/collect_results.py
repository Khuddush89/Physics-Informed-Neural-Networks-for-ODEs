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


def fmt_t(x):
    if not (isinstance(x, (int, float)) and math.isfinite(x)):
        return "--"
    return f"{x:.0f}" if x >= 10 else f"{x:.1f}"


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
            lines.append(f"{ch} & \\code{{{r['example']}}} & {tex_escape(r['description'])} & {fmt_t(r['wall_time_s'])}\\\\")
        lines += [r"\end{longtable}", r"\end{small}"]
        Path(args.latex).parent.mkdir(parents=True, exist_ok=True)
        Path(args.latex).write_text("\n".join(lines) + "\n")
    print(f"{len(rows)} result files collected -> {out_csv.relative_to(ROOT)}, docs/RESULTS.md"
          + (f", {args.latex}" if args.latex else ""))


if __name__ == "__main__":
    main()
