#!/usr/bin/env python3
"""Run every example of the book (or a subset) and collect a summary table.

Examples
--------
    python scripts/run_all.py                 # full runs, all chapters (about 2 hours on 1 core)
    python scripts/run_all.py --jobs 2        # two chapters at a time (2 cores)
    python scripts/run_all.py --quick         # smoke test of every script (a few minutes)
    python scripts/run_all.py --chapter ch07  # one chapter
    python scripts/run_all.py --list          # show the scripts that would run

Each script writes ``outputs/<chapter>/<example>/results.json``; this runner gathers the
``train_time_s``/``wall_time_total_s`` fields into ``outputs/summary.csv`` (Appendix B of
the book reports these runtimes).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def discover(chapter: str | None):
    """Every non-underscore script, plus extra invocations listed in chapters/*/extra_runs.txt
    (one line per run: ``script.py --option value ...``)."""
    jobs = []
    for folder in sorted((ROOT / "chapters").glob("ch*")):
        if not folder.is_dir() or (chapter and not folder.name.startswith(chapter)):
            continue
        # table/asset writers read the results of the other scripts, so they run last
        writer = ("make_tables", "write_tables", "write_book_assets")
        files = sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))
        files = [p for p in files if not p.stem.startswith(writer)] + \
                [p for p in files if p.stem.startswith(writer)]
        extra_jobs = []
        extra = folder / "extra_runs.txt"
        if extra.exists():
            for line in extra.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    extra_jobs.append((folder / parts[0], parts[1:]))
        normal = [(p, []) for p in files if not p.stem.startswith(writer)]
        writers = [(p, []) for p in files if p.stem.startswith(writer)]
        jobs.extend(normal + extra_jobs + writers)
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--chapter", default=None, help="prefix such as ch04 or ch04_ivp")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--jobs", type=int, default=1,
                    help="chapters run in parallel (each script stays single-threaded)")
    args = ap.parse_args()

    scripts = discover(args.chapter)
    if args.list:
        for s, extra in scripts:
            print(" ".join([str(s.relative_to(ROOT)), *extra]))
        return 0
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               MPLBACKEND="Agg")
    rows, failures = [], []

    def run_job(job):
        s, extra = job
        if args.quick and "--outdir" in extra:          # keep quick outputs out of outputs/<chapter>
            i = extra.index("--outdir")
            extra = extra[:i] + extra[i + 2:]
        cmd = [sys.executable, str(s)] + list(extra) + (["--quick"] if args.quick else []) + \
              (["--no-figures"] if args.no_figures else [])
        name = " ".join([str(s.relative_to(ROOT)), *extra])
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
        dt = time.perf_counter() - t0
        status = "ok" if proc.returncode == 0 else "FAILED"
        print(f"==> {name}: {status} in {dt:.1f} s", flush=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-4000:], flush=True)
        return {"script": name, "status": status, "wall_time_s": round(dt, 1)}

    # group by chapter: scripts of one chapter run sequentially (writers last)
    groups = {}
    for job in scripts:
        groups.setdefault(job[0].parent.name, []).append(job)

    def run_group(jobs):
        return [run_job(j) for j in jobs]

    if args.jobs > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for res in pool.map(run_group, groups.values()):
                rows.extend(res)
    else:
        for jobs in groups.values():
            rows.extend(run_group(jobs))
    failures = [r["script"] for r in rows if r["status"] != "ok"]

    summary = ROOT / "outputs" / ("summary_quick.csv" if args.quick else "summary.csv")
    summary.parent.mkdir(parents=True, exist_ok=True)
    with open(summary, "w") as fh:
        fh.write("script,status,wall_time_s\n")
        for r in rows:
            fh.write(f"{r['script']},{r['status']},{r['wall_time_s']}\n")
    print(f"\nsummary written to {summary.relative_to(ROOT)}")
    if failures:
        print("FAILED scripts:\n  " + "\n  ".join(failures))
        return 1
    print(f"all {len(rows)} scripts succeeded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
