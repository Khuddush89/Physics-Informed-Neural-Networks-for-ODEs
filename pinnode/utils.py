"""Reproducibility, configuration and output helpers shared by every example.

Every script in ``chapters/`` follows the same pattern::

    from pinnode.utils import Example

    ex = Example("ch04_ivp", "logistic", description="Logistic ODE, activation study")
    n_adam = ex.pick(full=8000, quick=150)      # --quick shrinks the run for smoke tests
    ...
    ex.save_history(history, "tanh")
    ex.save_results({"rel_l2": err, "train_time_s": hist.train_time})
    ex.savefig(fig, "logistic_solutions")

Outputs are written to ``outputs/<chapter>/<example>/`` (CSV, JSON, PDF and PNG), so a
run never overwrites the source tree and every number quoted in the book can be traced
to a file produced by a fixed seed.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:  # torch is required for every PINN example but not for the plotting helpers
    import torch
except ImportError:  # pragma: no cover
    torch = None

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs"


# ----------------------------------------------------------------------------- seeds
def set_seed(seed: int = 1234, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch and (optionally) request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # pragma: no cover - very old torch
                pass


def configure_torch(dtype: str = "float64", threads: int = 1) -> None:
    """Use double precision and a fixed thread count (both matter for reproducibility).

    Double precision is the default throughout the book: PINN losses routinely reach
    1e-8 to 1e-12, below the resolution of single precision, and L-BFGS line searches
    are unreliable in float32.
    """
    if torch is None:  # pragma: no cover
        return
    torch.set_default_dtype(torch.float64 if dtype == "float64" else torch.float32)
    if threads is not None and threads > 0:
        torch.set_num_threads(threads)


def environment_info() -> Dict[str, str]:
    """Versions recorded alongside every result file."""
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import scipy

        info["scipy"] = scipy.__version__
    except ImportError:  # pragma: no cover
        pass
    if torch is not None:
        info["torch"] = torch.__version__
        info["torch_threads"] = str(torch.get_num_threads())
        info["torch_default_dtype"] = str(torch.get_default_dtype())
    return info


# ----------------------------------------------------------------------------- timing
class Timer:
    """Context manager measuring wall-clock time in seconds."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0
        return False


# ----------------------------------------------------------------------------- example
def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


@dataclass
class Example:
    """Command-line interface, seeding and output management for one book example.

    Parameters
    ----------
    chapter : str
        Chapter folder name, e.g. ``"ch04_ivp"``.
    name : str
        Short example name, used for the output folder.
    description : str
        One-line description printed at start-up and stored in ``results.json``.
    default_seed : int
        Seed used when ``--seed`` is not given. The seeds quoted in the book are the
        defaults hard-coded in each script.
    """

    chapter: str
    name: str
    description: str = ""
    default_seed: int = 1234
    dtype: str = "float64"
    argv: Optional[list] = None
    extra_args: Optional[list] = None
    args: argparse.Namespace = field(init=False)

    def __post_init__(self):
        parser = argparse.ArgumentParser(description=self.description or self.name)
        parser.add_argument("--quick", action="store_true",
                            help="tiny run for smoke tests (results are NOT meaningful)")
        parser.add_argument("--seed", type=int, default=self.default_seed)
        parser.add_argument("--threads", type=int, default=1)
        parser.add_argument("--outdir", type=str, default=None,
                            help="override output folder (default outputs/<chapter>/<name>)")
        parser.add_argument("--no-figures", action="store_true", help="skip figure generation")
        for spec in self.extra_args or []:
            flags, kwargs = spec
            parser.add_argument(*flags, **kwargs)
        self.args = parser.parse_args(self.argv if self.argv is not None else sys.argv[1:])
        configure_torch(self.dtype, self.args.threads)
        set_seed(self.args.seed)
        out = Path(self.args.outdir) if self.args.outdir else OUTPUT_ROOT / self.chapter / self.name
        if self.args.quick and not self.args.outdir:
            out = OUTPUT_ROOT / "_quick" / self.chapter / self.name
        self.outdir = out
        self.figdir = out / "figures"
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.figdir.mkdir(parents=True, exist_ok=True)
        self._results: Dict[str, Any] = {}
        self._t_start = time.perf_counter()
        mode = "QUICK (smoke test)" if self.args.quick else "full"
        print(f"[{self.chapter}/{self.name}] {self.description} | seed={self.args.seed} | mode={mode}")

    # --------------------------------------------------------------- run-size helpers
    @property
    def quick(self) -> bool:
        return bool(self.args.quick)

    @property
    def seed(self) -> int:
        return int(self.args.seed)

    def pick(self, full, quick):
        """Return ``quick`` in smoke-test mode and ``full`` otherwise."""
        return quick if self.quick else full

    # --------------------------------------------------------------- saving
    def path(self, filename: str) -> Path:
        return self.outdir / filename

    def save_history(self, history, tag: str = "") -> Path:
        name = f"history_{tag}.csv" if tag else "history.csv"
        p = self.path(name)
        history.save_csv(p)
        return p

    def save_table(self, table, name: str) -> Path:
        """Save a pandas DataFrame or a dict of equal-length arrays as CSV."""
        import pandas as pd

        df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
        p = self.path(f"{name}.csv")
        df.to_csv(p, index=False, float_format="%.10e")
        return p

    def save_results(self, results: Dict[str, Any], name: str = "results") -> Path:
        """Merge ``results`` into ``results.json`` (with seed, timing and versions)."""
        self._results.update(results)
        payload = {
            "chapter": self.chapter,
            "example": self.name,
            "description": self.description,
            "seed": self.seed,
            "quick": self.quick,
            "wall_time_total_s": round(time.perf_counter() - self._t_start, 3),
            "environment": environment_info(),
            **self._results,
        }
        p = self.path(f"{name}.json")
        with open(p, "w") as fh:
            json.dump(payload, fh, indent=2, default=_json_default)
        return p

    def savefig(self, fig, stem: str, formats=("pdf", "png"), dpi: int = 300) -> None:
        if self.args.no_figures:
            return
        from .plotting import savefig as _savefig

        _savefig(fig, self.figdir / stem, formats=formats, dpi=dpi)

    def log(self, msg: str) -> None:
        print(f"[{self.chapter}/{self.name}] {msg}")
