"""Small helpers shared by the Chapter 9 scripts (training difficulties and diagnostics).

The leading underscore keeps this module out of ``scripts/run_all.py`` and the smoke tests.
Every function is short enough to be read next to the book:

* :func:`grad_norm`            - Euclidean norm of the gradient of one loss term (diagnostic).
* :func:`causal_weights`       - temporal weights w_i = exp(-eps sum_{k<i} L_k) (causal training).
* :class:`CPUTimer`            - process CPU time (robust to other jobs on a shared machine).
* :func:`projection_amplitude` - amplitude of the Fourier mode exp(2 pi i k t) in a sampled signal.
* :func:`latex_table`          - write a booktabs ``tabular`` fragment that the book \\input's.
* :class:`RunCache`            - pickle each independent run so that a killed script resumes.
"""
from __future__ import annotations

import math
import pickle
import shutil
import time
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch


def grad_norm(loss: torch.Tensor, params: Sequence[torch.Tensor]) -> float:
    """||d loss / d theta||_2 without touching the .grad fields (graph is retained)."""
    grads = torch.autograd.grad(loss, list(params), retain_graph=True, allow_unused=True)
    sq = sum(float((g ** 2).sum()) for g in grads if g is not None)
    return math.sqrt(sq)


def grad_vector(loss: torch.Tensor, params: Sequence[torch.Tensor]) -> torch.Tensor:
    """Flattened gradient of ``loss`` (graph retained); unused parameters give zeros."""
    grads = torch.autograd.grad(loss, list(params), retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(grads, params)])


def causal_weights(chunk_losses: torch.Tensor, eps: float) -> torch.Tensor:
    """w_1 = 1, w_i = exp(-eps * sum_{k<i} L_k); computed without gradient (stop-gradient)."""
    with torch.no_grad():
        cum = torch.cumsum(chunk_losses.detach(), dim=0)
        prev = torch.cat([torch.zeros(1, dtype=cum.dtype), cum[:-1]])
        return torch.exp(-eps * prev)


class CPUTimer:
    """Context manager returning process CPU time and wall time in seconds."""

    def __enter__(self):
        self.c0, self.w0 = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.cpu = time.process_time() - self.c0
        self.wall = time.perf_counter() - self.w0
        return False


def projection_amplitude(e: np.ndarray, t: np.ndarray, k: float) -> float:
    """Amplitude of the mode of frequency k (cycles per unit time) in e(t) on [t_0, t_end].

    Uses the trapezoidal rule for a_k = (2/L) int e cos(2 pi k t), b_k = (2/L) int e sin(...)
    and returns sqrt(a_k^2 + b_k^2); for e = A sin(2 pi k t + phi) on whole periods this is A.
    """
    L = t[-1] - t[0]
    c, s = np.cos(2 * np.pi * k * t), np.sin(2 * np.pi * k * t)
    a = 2.0 / L * np.trapezoid(e * c, t)
    b = 2.0 / L * np.trapezoid(e * s, t)
    return float(np.hypot(a, b))


def dominant_frequency(v: np.ndarray, t: np.ndarray) -> float:
    """Frequency (cycles per unit time) of the largest DFT coefficient of a uniformly sampled v."""
    v = np.asarray(v, float) - np.mean(v)
    spec = np.abs(np.fft.rfft(v * np.hanning(len(v))))
    freqs = np.fft.rfftfreq(len(v), d=(t[1] - t[0]))
    return float(freqs[int(np.argmax(spec))])


# ----------------------------------------------------------------------------- LaTeX output
def num(x, digits: int = 2) -> str:
    """siunitx number, e.g. 1.234e-05 -> \\num{1.23e-5}; NaN -> '--'."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    if isinstance(x, (int, np.integer)):
        return f"\\num{{{int(x)}}}"
    m, e = f"{float(x):.{digits}e}".split("e")
    if int(e) == 0:
        return f"\\num{{{m}}}"
    return f"\\num{{{m}e{int(e)}}}"


def fixed(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "--"
    return f"\\num{{{float(x):.{digits}f}}}"


def latex_table(path: Path, colspec: str, header: Iterable[str], rows: List[Iterable[str]],
                midrules: Sequence[int] = (), comment: str = "") -> Path:
    """Write ``\\begin{tabular}{colspec} ... \\end{tabular}`` with booktabs rules.

    ``midrules`` lists row indices before which an extra ``\\midrule`` is inserted. The
    caption and label stay in the chapter file, so the fragment can be regenerated freely.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if comment:
        lines.append(f"% {comment}")
    lines.append("% generated automatically - do not edit by hand")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for i, row in enumerate(rows):
        if i in midrules and i > 0:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    path.write_text("\n".join(lines) + "\n")
    return path


class RunCache:
    """Store the result of each independent run in ``<outdir>/_cache/<key>.pkl``.

    Every run seeds the random number generators itself, so a script that was interrupted
    (for example by the out-of-memory killer on a shared machine) resumes with exactly the
    results it would have produced in one go. After a successful run :meth:`clear` moves the
    pickles to ``<outdir>/_last_run``; ``--replot`` (see :func:`replot_arg`) redraws figures
    and tables from there without training. A normal run never reads ``_last_run``.
    """

    def __init__(self, ex):
        self.dir = Path(ex.outdir) / "_cache"
        self.last = Path(ex.outdir) / "_last_run"
        self.replot = bool(getattr(ex.args, "replot", False))

    def get(self, key: str, fn):
        if self.replot:
            with open(self.last / f"{key}.pkl", "rb") as fh:
                return pickle.load(fh)
        path = self.dir / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                return pickle.load(fh)
        result = fn()
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(result, fh)
        tmp.replace(path)
        return result

    def clear(self) -> None:
        if self.replot:
            return
        shutil.rmtree(self.last, ignore_errors=True)
        if self.dir.exists():
            self.dir.rename(self.last)


def replot_arg():
    """Extra command-line option shared by the Chapter 9 scripts."""
    return (["--replot"], {"action": "store_true",
                           "help": "redraw figures and tables from the last completed run (no training)"})
