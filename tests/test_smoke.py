"""Smoke tests: every chapter script must run end-to-end in --quick mode.

    pytest -q                      # all scripts (a few minutes on one CPU core)
    pytest -q -k ch07              # one chapter

Quick mode shrinks iteration counts so that the tests check the *code path* (shapes,
autograd, quadrature, plotting, file output), not the accuracy reported in the book.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p for p in (ROOT / "chapters").glob("ch*/*.py") if not p.name.startswith("_"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[f"{p.parent.name}/{p.stem}" for p in SCRIPTS])
def test_script_runs_quick(script, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), MPLBACKEND="Agg")
    proc = subprocess.run([sys.executable, str(script), "--quick", "--outdir", str(tmp_path)],
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stdout[-3000:] + "\n" + proc.stderr[-3000:]
    assert (tmp_path / "results.json").exists(), "every example must write results.json"


def test_library_imports():
    import pinnode

    assert pinnode.__version__
