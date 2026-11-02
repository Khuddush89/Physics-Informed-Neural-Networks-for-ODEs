"""One visual system for every figure in the book and the repository.

Design rules (applied by :func:`use_book_style` and the helpers below)

* Figures are drawn at their printed size (5.5 in wide, 9 pt type) so that fonts match the
  body text when included at \\textwidth; they are saved as vector PDF (book) and 300 dpi PNG (repository).
* Computer Modern text and mathematics (bundled with Matplotlib), matching the LaTeX body.
* A color-blind-safe categorical palette, always paired with a second encoding (line
  style and/or marker) so that every figure remains readable in grayscale print.
* Reference solutions are thin solid near-black lines; PINN predictions are colored
  dashed lines drawn on top; errors and losses use logarithmic axes.
* Thin solid hairline grid, no top/right spines, a legend whenever there are >= 2 series.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib

logging.getLogger("fontTools").setLevel(logging.ERROR)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TEXT_WIDTH_IN = 5.5   # printed text width the figures are designed for (inches)
HALF_WIDTH_IN = 2.7

COLORS = {
    "ref": "#1a1a1a",
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#6250d6",
    "red": "#e34948",
    "yellow": "#eda100",
    "magenta": "#e87ba4",
    "green": "#008300",
    "grid": "#e6e6e6",
    "muted": "#6e6e6e",
    "light": "#bdbdbd",
}
# Fixed categorical order (validated for color-vision deficiency, adjacent pairs).
SERIES = [COLORS[k] for k in ("blue", "orange", "aqua", "violet", "red", "yellow")]
LINESTYLES = ["--", "-.", ":", (0, (5, 1.5, 1, 1.5)), (0, (1, 1)), (0, (4, 1, 1, 1, 1, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def use_book_style(font_size: float = 9.0) -> None:
    """Apply the book's Matplotlib style (idempotent)."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmr10", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "axes.unicode_minus": False,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#d9d9d9",
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.handlelength": 2.4,
        "legend.labelspacing": 0.3,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#4d4d4d",
        "axes.labelcolor": "#1a1a1a",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.5,
        "grid.linestyle": "-",
        "xtick.color": "#4d4d4d",
        "ytick.color": "#4d4d4d",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
    })


use_book_style()


# ----------------------------------------------------------------------------- figures
def figure(nrows: int = 1, ncols: int = 1, width: str | float = "full",
           height: Optional[float] = None, aspect: float = 0.62, **kwargs):
    """Create a figure at printed size. ``width`` is "full", "half" or inches."""
    w = {"full": TEXT_WIDTH_IN, "half": HALF_WIDTH_IN}.get(width, width)
    if height is None:
        height = w / ncols * aspect * nrows + 0.15 * (nrows - 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, height), constrained_layout=True, **kwargs)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.04, hspace=0.04)
    return fig, axes


def label_panels(axes, labels: Optional[Sequence[str]] = None, x: float = -0.02,
                 y: float = 1.04) -> None:
    """Add (a), (b), ... in the upper-left corner outside each axes."""
    axes = np.atleast_1d(axes).ravel()
    labels = labels or [f"({chr(97 + i)})" for i in range(len(axes))]
    for ax, lab in zip(axes, labels):
        ax.text(x, y, lab, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=plt.rcParams["font.size"], fontweight="normal")


def savefig(fig, stem, formats: Iterable[str] = ("pdf", "png"), dpi: int = 300) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)


# ----------------------------------------------------------------------------- series helpers
def plot_reference(ax, x, y, label: str = "reference", **kw):
    kw = {"color": COLORS["ref"], "lw": 1.0, "ls": "-", "zorder": 2, **kw}
    return ax.plot(x, y, label=label, **kw)


def plot_prediction(ax, x, y, label: str = "PINN", k: int = 0, markevery=None, **kw):
    """PINN prediction: series color k, dashed style k, optional sparse markers."""
    kw = {"color": SERIES[k % len(SERIES)], "lw": 1.6, "ls": LINESTYLES[k % len(LINESTYLES)],
          "zorder": 3, **kw}
    if markevery is not None:
        kw.setdefault("marker", MARKERS[k % len(MARKERS)])
        kw.setdefault("markevery", markevery)
        kw.setdefault("markersize", 3.2)
        kw.setdefault("markerfacecolor", "white")
        kw.setdefault("markeredgewidth", 0.8)
    return ax.plot(x, y, label=label, **kw)


def plot_solution(ax, x, ref=None, pred=None, ref_label="reference", pred_label="PINN",
                  k: int = 0, xlabel: str = "$t$", ylabel: str = "$y(t)$", markevery=None):
    if ref is not None:
        plot_reference(ax, x, ref, label=ref_label)
    if pred is not None:
        plot_prediction(ax, x, pred, label=pred_label, k=k, markevery=markevery)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")


def plot_error(ax, x, err, label: Optional[str] = None, k: int = 0, floor: float = 1e-13,
               xlabel: str = "$t$", ylabel: str = "absolute error", **kw):
    """Pointwise absolute error on a logarithmic axis (zeros are floored)."""
    e = np.maximum(np.abs(np.asarray(err)), floor)
    kw = {"color": SERIES[k % len(SERIES)], "lw": 1.1, "ls": "-" if k == 0 else LINESTYLES[k - 1], **kw}
    ax.semilogy(x, e, label=label, **kw)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if label:
        ax.legend(loc="best")


def rolling_median(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, float)
    if window <= 1 or len(y) < window:
        return y
    pad = window // 2
    ypad = np.pad(y, (pad, window - 1 - pad), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view

    return np.median(sliding_window_view(ypad, window), axis=1)


def plot_history(ax, hist, parts: Optional[Sequence[str]] = None, labels: Optional[dict] = None,
                 total: bool = True, smooth: bool = True, show_switch: bool = True,
                 window: Optional[int] = None, ylabel: str = "loss"):
    """Training history on a log axis: faint raw curves, solid rolling medians.

    ``parts`` selects loss components stored in ``hist.parts``; ``labels`` maps component
    names to legend labels. The Adam -> L-BFGS switch is marked by a vertical hairline.
    """
    labels = labels or {}
    it = np.asarray(hist.iters)
    series = []
    if total:
        series.append(("loss", np.asarray(hist.loss), COLORS["ref"], "-"))
    for j, p in enumerate(parts or []):
        series.append((p, np.asarray(hist.parts[p]), SERIES[j % len(SERIES)], LINESTYLES[j % len(LINESTYLES)]))
    w = window or max(5, len(it) // 60)
    for name, y, c, ls in series:
        lab = labels.get(name, "total" if name == "loss" else name)
        if smooth and len(y) > 3 * w:
            ax.semilogy(it, y, color=c, lw=0.5, alpha=0.25, ls="-")
            ax.semilogy(it, rolling_median(y, w), color=c, lw=1.2, ls=ls, label=lab)
        else:
            ax.semilogy(it, y, color=c, lw=1.1, ls=ls, label=lab)
    if show_switch and getattr(hist, "switch_iter", None):
        ax.axvline(hist.switch_iter, color=COLORS["muted"], lw=0.6, ls="-", zorder=1)
        ax.text(hist.switch_iter, 0.97, " L-BFGS", transform=ax.get_xaxis_transform(),
                ha="left", va="top", fontsize=plt.rcParams["font.size"] - 1.5, color=COLORS["muted"])
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if len(series) > 1:
        ax.legend(loc="upper right")


def plot_metric(ax, hist, key: str, label: Optional[str] = None, k: int = 0,
                ylabel: str = "relative $L^2$ error", show_switch: bool = True):
    ax.semilogy(hist.eval_iters, hist.metrics[key], color=SERIES[k % len(SERIES)], lw=1.2,
                ls="-" if k == 0 else LINESTYLES[k - 1], label=label)
    if show_switch and getattr(hist, "switch_iter", None):
        ax.axvline(hist.switch_iter, color=COLORS["muted"], lw=0.6, zorder=1)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    if label:
        ax.legend(loc="best")


def rug(ax, x, y0: Optional[float] = None, color: str = COLORS["muted"], label: Optional[str] = None):
    """Mark collocation points as short ticks along the bottom of the axes."""
    trans = ax.get_xaxis_transform()
    ax.plot(x, np.zeros_like(np.asarray(x)) + 0.015, "|", color=color, ms=4, mew=0.6,
            transform=trans, label=label, clip_on=False)
