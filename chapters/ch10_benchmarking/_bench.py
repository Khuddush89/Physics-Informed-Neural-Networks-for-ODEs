"""Shared helpers for the Chapter 10 benchmarks (imported by the scripts, not an example).

Importing this module first in every Chapter 10 script

* restricts the BLAS/OpenMP libraries used by NumPy/SciPy to ONE thread (this must happen
  before NumPy is imported, hence the environment variables at the top of the file);
  PyTorch is restricted to one thread by ``pinnode.utils.Example``;
* provides the timing protocol of Section 10.1: one warm-up call, then the median CPU time
  (``time.process_time``) and wall-clock time (``time.perf_counter``) of ``repeats`` calls.
  CPU time of the single-threaded process is the cost measure used in the book: it equals
  the wall-clock time on an idle machine and, unlike wall-clock time, is not inflated when
  other jobs share the processor;
* runs classical work-precision sweeps with ``scipy.integrate.solve_ivp``;
* wraps PINN training (pinnode.train) with a residual-evaluation counter and an
  error-versus-time trajectory recorder whose own cost is excluded from the training time;
* extracts matched-accuracy costs and draws work-precision diagrams in the book style.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import math  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from typing import Callable, Dict, Iterable, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from pinnode import linf, rel_l2, train  # noqa: E402
from pinnode import plotting as P  # noqa: E402
from pinnode.utils import environment_info  # noqa: E402

# Relative tolerances of the classical sweeps: 1e-3, 10^-3.5, ..., 1e-12 (half decades).
RTOLS = [10.0 ** (-3.0 - 0.5 * k) for k in range(19)]
IVP_METHODS = ["RK45", "DOP853", "Radau", "BDF", "LSODA"]
TARGETS = (1e-4, 1e-6)
COST = "cpu_s"          # cost measure used in tables and figures

# One fixed visual identity per method across all figures of the chapter.
STYLE: Dict[str, dict] = {
    "RK45": dict(color=P.SERIES[0], marker="o", ls="-"),
    "DOP853": dict(color=P.SERIES[1], marker="s", ls="-"),
    "Radau": dict(color=P.SERIES[2], marker="^", ls="-"),
    "BDF": dict(color=P.SERIES[3], marker="D", ls="-"),
    "LSODA": dict(color=P.SERIES[5], marker="v", ls="-"),
    "solve_bvp": dict(color=P.SERIES[0], marker="o", ls="-"),
    "shooting": dict(color=P.SERIES[1], marker="s", ls="-"),
    "FD-Newton": dict(color=P.SERIES[2], marker="^", ls="-"),
    "Nystrom": dict(color=P.SERIES[3], marker="D", ls="-"),
    "trapezoid": dict(color=P.SERIES[2], marker="^", ls="-"),
    "Chebyshev": dict(color=P.SERIES[3], marker="D", ls="-"),
}
PINN_STYLE = dict(color=P.COLORS["red"], marker="*", ms=8.0, mec="#1a1a1a", mew=0.5, ls="none",
                  zorder=6)
PINN_STYLE_ALT = dict(color=P.SERIES[3], marker="P", ms=7.0, mec="#1a1a1a", mew=0.5, ls="none",
                      zorder=6)


# ----------------------------------------------------------------------------- resume cache
# ``--resume`` lets an interrupted run (e.g. killed on a memory-starved shared machine) reuse
# the PINN configurations that had already finished; a plain run always recomputes everything.
RESUME_ARGS = [(("--resume",), {"action": "store_true",
                                "help": "reuse PINN results cached by an interrupted run"})]


def cached(ex, key: str, fn: Callable[[], dict]) -> dict:
    """Return ``fn()``; the (picklable) result is stored in ``<outdir>/cache/<key>.pkl`` and
    reloaded instead of recomputed when the script runs with ``--resume``."""
    import pickle

    path = ex.outdir / "cache" / f"{key}.pkl"
    if getattr(ex.args, "resume", False) and path.exists():
        ex.log(f"resume: reusing cached result '{key}'")
        with open(path, "rb") as fh:
            return pickle.load(fh)
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(res, fh)
    return res


def strip_model(out: dict) -> dict:
    """Drop non-picklable entries (the model) from a run_pinn result."""
    return {k: v for k, v in out.items() if k != "model"}


# ----------------------------------------------------------------------------- environment
def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover - non-Linux
        pass
    return platform.processor() or "unknown"


def hardware_info() -> dict:
    info = environment_info()
    try:
        load = os.getloadavg()
    except OSError:  # pragma: no cover
        load = (math.nan,) * 3
    info.update({"cpu_model": cpu_model(), "logical_cpus": os.cpu_count(),
                 "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                 "load_average_at_end": [round(x, 2) for x in load],
                 "process_cpu_time_s": round(time.process_time(), 2),
                 "timers": "time.process_time (CPU, reported), time.perf_counter (wall, recorded)"})
    return info


# ----------------------------------------------------------------------------- timing
REPEATS = 5             # timed repetitions of every classical run (median reported)


def timed(fn: Callable[[], object], repeats: int = REPEATS, warmup: int = 1):
    """Call ``fn`` ``warmup`` times (discarded), then ``repeats`` times.

    Returns ``(result_of_last_call, median_cpu_s, median_wall_s)``.
    """
    for _ in range(warmup):
        fn()
    walls, cpus, out = [], [], None
    for _ in range(max(1, repeats)):
        c0, t0 = time.process_time(), time.perf_counter()
        out = fn()
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    return out, float(np.median(cpus)), float(np.median(walls))


# ----------------------------------------------------------------------------- classical IVPs
def ivp_sweep(f: Callable, t_span: Sequence[float], y0: Sequence[float], t_test: np.ndarray,
              y_ref: np.ndarray, *, methods: Iterable[str] = IVP_METHODS,
              rtols: Iterable[float] = RTOLS, atol_factor: float = 1.0,
              jac: Optional[Callable] = None, components: Optional[Sequence[int]] = None,
              repeats: int = REPEATS, log: Optional[Callable[[str], None]] = None) -> List[dict]:
    """Work-precision sweep of ``solve_ivp`` methods over relative tolerances.

    Every run uses ``atol = atol_factor * rtol`` and returns the solution on the test grid via
    ``t_eval`` (the solver's own dense output), so the timed call delivers exactly what the
    error is measured on. Implicit methods receive the analytic Jacobian ``jac`` when given.
    ``components`` selects the solution components compared with ``y_ref``.
    """
    y0 = np.atleast_1d(np.asarray(y0, float))
    comps = list(range(len(y0))) if components is None else list(components)
    rows = []
    for m in methods:
        for rtol in rtols:
            kw = dict(method=m, t_eval=t_test, rtol=rtol, atol=atol_factor * rtol)
            if jac is not None and m in ("Radau", "BDF", "LSODA"):
                kw["jac"] = jac
            sol, cpu, wall = timed(lambda: solve_ivp(f, t_span, y0, **kw), repeats=repeats)
            ok = bool(sol.success) and sol.y.shape[1] == len(t_test)
            Y = sol.y.T[:, comps] if ok else np.full((len(t_test), len(comps)), np.nan)
            row = dict(method=m, rtol=rtol, atol=atol_factor * rtol, success=ok,
                       rel_l2=rel_l2(Y, y_ref) if ok else math.nan,
                       linf=linf(Y, y_ref) if ok else math.nan,
                       cpu_s=cpu, wall_s=wall, nfev=int(sol.nfev), njev=int(sol.njev),
                       nlu=int(sol.nlu), evals=int(sol.nfev))
            rows.append(row)
            if log is not None:
                log(f"{m:7s} rtol={rtol:.1e}  relL2={row['rel_l2']:.2e}  cpu={cpu*1e3:8.2f} ms "
                    f"wall={wall*1e3:8.2f} ms nfev={row['nfev']}")
    return rows


def time_to_reach(rows: List[dict], target: float, key: str = "rel_l2",
                  cost: str = COST) -> Optional[dict]:
    """Cheapest measured run with ``rows[i][key] <= target`` (None if never reached)."""
    ok = [r for r in rows if r.get(key) is not None and np.isfinite(r[key]) and r[key] <= target]
    return min(ok, key=lambda r: r[cost]) if ok else None


def matched_table(groups: Dict[str, List[dict]], targets: Sequence[float] = TARGETS) -> List[dict]:
    """For every method (key of ``groups``) and target: cost, evaluations and error of the
    cheapest run that reaches the target."""
    out = []
    for name, rows in groups.items():
        for tgt in targets:
            best = time_to_reach(rows, tgt)
            out.append({"method": name, "target": tgt, "reached": best is not None,
                        "cpu_s": best["cpu_s"] if best else None,
                        "wall_s": best.get("wall_s") if best else None,
                        "evals": best.get("evals") if best else None,
                        "rel_l2": best["rel_l2"] if best else None,
                        "setting": best.get("setting", best.get("rtol")) if best else None})
    return out


def best_error(rows: List[dict], key: str = "rel_l2") -> float:
    vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(min(vals)) if vals else math.nan


# ----------------------------------------------------------------------------- PINN bookkeeping
class CountingLoss:
    """Wrap a pinnode loss closure and count its calls (Adam steps, L-BFGS evaluations)."""

    def __init__(self, loss_fn: Callable, points_per_call: int):
        self.loss_fn = loss_fn
        self.points = int(points_per_call)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.loss_fn()

    @property
    def residual_evals(self) -> int:
        return self.calls * self.points


class Trajectory:
    """``eval_fn`` for :func:`pinnode.train` that records (CPU time, wall time, error).

    The time stamps exclude the cost of the evaluations themselves, so the recorded curve is
    the error a user would have obtained by stopping the training at that moment.
    """

    def __init__(self, error_fn: Callable[[], float]):
        self.error_fn = error_fn
        self.start()

    def start(self) -> None:
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        self.wall_overhead = self.cpu_overhead = 0.0
        self.wall: List[float] = []
        self.cpu: List[float] = []
        self.errors: List[float] = []

    def __call__(self) -> Dict[str, float]:
        s, c = time.perf_counter(), time.process_time()
        with torch.no_grad():
            e = float(self.error_fn())
        self.wall.append(s - self.t0 - self.wall_overhead)
        self.cpu.append(c - self.c0 - self.cpu_overhead)
        self.errors.append(e)
        self.wall_overhead += time.perf_counter() - s
        self.cpu_overhead += time.process_time() - c
        return {"rel_l2": e}


def run_pinn(model: torch.nn.Module, loss_fn: Callable, points_per_call: int,
             error_fn: Callable[[], float], predict_fn: Callable[[], np.ndarray], *,
             adam_iters: int, lbfgs_iters: int, lr: float = 1e-3, lr_final: float = 1e-5,
             eval_every: int = 100, print_every: int = 2000, params=None) -> dict:
    """Train a PINN with the book's two-stage optimizer and return cost/accuracy records.

    ``error_fn`` returns the test error (no gradients needed) and is used for the training
    trajectory; ``predict_fn`` returns the prediction on the test grid and is timed separately
    (median of 3 calls) because evaluating the surrogate is part of delivering the solution.
    """
    loss0, _ = loss_fn()                       # warm-up (graph construction, allocator)
    loss0.backward()
    model.zero_grad(set_to_none=True)
    counter = CountingLoss(loss_fn, points_per_call)
    traj = Trajectory(error_fn)
    traj.start()
    c0 = time.process_time()
    hist = train(model.parameters() if params is None else params, counter,
                 adam_iters=adam_iters, lr=lr, lr_final=lr_final, lbfgs_iters=lbfgs_iters,
                 eval_fn=traj, eval_every=eval_every, print_every=print_every)
    cpu_train = time.process_time() - c0 - traj.cpu_overhead
    wall_train = hist.train_time - traj.wall_overhead
    pred, cpu_eval, wall_eval = timed(predict_fn, repeats=3, warmup=1)
    return {"hist": hist, "pred": np.asarray(pred), "train_time_s": hist.train_time,
            "train_cpu_s": cpu_train, "train_wall_s": wall_train,
            "eval_cpu_s": cpu_eval, "eval_wall_s": wall_eval,
            "cpu_s": cpu_train + cpu_eval, "wall_s": wall_train + wall_eval,
            "loss_calls": counter.calls, "evals": counter.residual_evals,
            "traj_cpu": np.asarray(traj.cpu), "traj_wall": np.asarray(traj.wall),
            "traj_err": np.asarray(traj.errors), "final_loss": hist.final_loss}


def pinn_record(tag: str, out: dict, pred: np.ndarray, ref: np.ndarray, **extra) -> dict:
    """Flat result row of one PINN configuration (for CSV/JSON and the matched tables)."""
    row = {"tag": tag, **extra, "rel_l2": rel_l2(pred, ref), "linf": linf(pred, ref),
           "final_loss": out["final_loss"], "train_cpu_s": out["train_cpu_s"],
           "train_time_s": out["train_time_s"], "eval_cpu_s": out["eval_cpu_s"],
           "eval_wall_s": out["eval_wall_s"], "cpu_s": out["cpu_s"], "wall_s": out["wall_s"],
           "loss_calls": out["loss_calls"], "evals": out["evals"]}
    return row


def trajectory_rows(tag: str, out: dict) -> List[dict]:
    return [{"tag": tag, "cpu_s": c, "wall_s": w, "rel_l2": e}
            for c, w, e in zip(out["traj_cpu"], out["traj_wall"], out["traj_err"])]


# ----------------------------------------------------------------------------- plotting
def plot_classical(ax, rows: List[dict], name: str, label: Optional[str] = None,
                   key: str = "rel_l2", cost: str = COST, style: Optional[dict] = None):
    """One classical method as a line with markers in the (error, cost) plane."""
    pts = sorted([(r[key], r[cost]) for r in rows
                  if r.get(key) is not None and np.isfinite(r[key]) and r[key] > 0],
                 key=lambda p: -p[0])
    if not pts:
        return
    x, y = zip(*pts)
    st = dict(STYLE.get(name, {}), **(style or {}))
    st.setdefault("lw", 0.9)
    st.setdefault("ms", 3.4)
    st.setdefault("mfc", "white")
    st.setdefault("mew", 0.8)
    ax.loglog(x, y, label=label or name, **st)


def plot_pinn(ax, rows: List[dict], label: str = "PINN", key: str = "rel_l2", cost: str = COST,
              annotate: bool = True, style: Optional[dict] = None, offsets=None):
    """PINN configurations as large highlighted markers, optionally tagged P1, P2, ..."""
    st = dict(PINN_STYLE, **(style or {}))
    x = [r[key] for r in rows]
    y = [r[cost] for r in rows]
    ax.loglog(x, y, label=label, **st)
    if annotate:
        for i, r in enumerate(rows):
            dx, dy = (offsets[i] if offsets is not None else (4, 3))
            ax.annotate(r.get("tag", str(i + 1)), (r[key], r[cost]), textcoords="offset points",
                        xytext=(dx, dy), fontsize=6.5, color="#1a1a1a")


def plot_trajectory(ax, cost: np.ndarray, errors: np.ndarray, label: Optional[str] = None,
                    color: Optional[str] = None):
    """Smallest PINN error reached up to a given training cost (running minimum, thin dotted
    line): the most favorable result that stopping the training earlier could give."""
    cost, errors = np.asarray(cost), np.asarray(errors)
    m = (cost > 0) & np.isfinite(errors)
    best = np.minimum.accumulate(errors[m])
    ax.loglog(best, cost[m], ls=":", lw=0.9, color=color or P.COLORS["red"], label=label,
              zorder=4)


def mark_targets(ax, targets: Sequence[float] = TARGETS):
    for t in targets:
        ax.axvline(t, color=P.COLORS["muted"], lw=0.5, ls=(0, (2, 2)), zorder=1)


def finish_wp_axes(ax, xlabel: str = "relative $L^2$ error", ylabel: str = "CPU time [s]"):
    """Error on the horizontal axis, cost on the vertical axis (both logarithmic): methods
    closer to the lower-left corner are more efficient."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
