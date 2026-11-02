"""Two-stage optimisation (Adam, then L-BFGS) with complete, plottable histories.

``loss_fn`` is a closure without arguments that returns ``(loss, parts)``, where ``loss`` is
a scalar tensor and ``parts`` a dict of floats or scalar tensors (e.g. residual and
initial-condition terms) recorded at every logged iteration::

    def loss_fn():
        t = colloc.requires_grad_(True)
        y = model(t)
        r = derivative(y, t) - f(t, y)
        L_r, L_ic = (r**2).mean(), ((model(t0) - y0)**2).mean()
        return L_r + L_ic, {"residual": L_r, "ic": L_ic}

    hist = train(model.parameters(), loss_fn, adam_iters=5000, lr=1e-3, lbfgs_iters=500,
                 eval_fn=lambda: {"rel_l2": rel_l2(predict(t_test), y_ref)})

The returned :class:`History` stores per-iteration loss components, learning rate and
elapsed time, plus metrics evaluated every ``eval_every`` iterations; ``history.switch_iter``
marks the Adam -> L-BFGS switch for plotting.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import torch


def _to_float(v) -> float:
    if isinstance(v, torch.Tensor):
        return float(v.detach().cpu().item())
    return float(v)


@dataclass
class History:
    iters: List[int] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)
    loss: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    parts: Dict[str, List[float]] = field(default_factory=dict)
    eval_iters: List[int] = field(default_factory=list)
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    snapshots: List[np.ndarray] = field(default_factory=list)
    snapshot_iters: List[int] = field(default_factory=list)
    switch_iter: Optional[int] = None
    train_time: float = 0.0

    # ---------------------------------------------------------------- recording
    def record(self, it: int, phase: str, loss: float, parts: Dict, lr: float, t: float) -> None:
        self.iters.append(it)
        self.phase.append(phase)
        self.loss.append(loss)
        self.lr.append(lr)
        self.time.append(t)
        for k, v in parts.items():
            self.parts.setdefault(k, [math.nan] * (len(self.iters) - 1)).append(_to_float(v))
        for k in self.parts:
            if len(self.parts[k]) < len(self.iters):
                self.parts[k].append(math.nan)

    def record_metrics(self, it: int, metrics: Dict) -> None:
        self.eval_iters.append(it)
        for k, v in metrics.items():
            self.metrics.setdefault(k, []).append(_to_float(v))

    # ---------------------------------------------------------------- accessors
    @property
    def final_loss(self) -> float:
        return self.loss[-1] if self.loss else math.nan

    def final_metric(self, key: str) -> float:
        return self.metrics[key][-1] if self.metrics.get(key) else math.nan

    def to_dataframe(self):
        import pandas as pd

        df = pd.DataFrame({"iter": self.iters, "phase": self.phase, "loss": self.loss,
                           "lr": self.lr, "time_s": self.time, **self.parts})
        if self.eval_iters:
            m = pd.DataFrame({"iter": self.eval_iters, **self.metrics})
            df = df.merge(m, on="iter", how="left")
        return df

    def save_csv(self, path) -> None:
        self.to_dataframe().to_csv(path, index=False, float_format="%.10e")


def train(params: Iterable[torch.Tensor], loss_fn: Callable, *, adam_iters: int = 5000,
          lr: float = 1e-3, lr_final: Optional[float] = None, schedule: str = "cosine",
          lbfgs_iters: int = 0, lbfgs_lr: float = 1.0, lbfgs_history: int = 50,
          lbfgs_chunk: int = 5, lbfgs_rtol: float = 1e-4, lbfgs_window: int = 10,
          grad_clip: Optional[float] = None, log_every: int = 10,
          eval_fn: Optional[Callable[[], Dict]] = None, eval_every: int = 100,
          snapshot_fn: Optional[Callable[[], np.ndarray]] = None, snapshot_every: int = 0,
          print_every: int = 1000, verbose: bool = True,
          weight_update: Optional[Callable[[int], None]] = None) -> History:
    """Minimise ``loss_fn`` with Adam (optionally followed by L-BFGS).

    Parameters
    ----------
    params : iterable of tensors
        Trainable parameters (``model.parameters()`` or a list including extra unknowns such
        as physical parameters in inverse problems).
    loss_fn : callable
        Closure returning ``(loss, parts)``.
    adam_iters, lr, lr_final, schedule : Adam settings. ``schedule`` is ``"cosine"``
        (default), ``"exp"`` or ``"constant"``; ``lr_final`` defaults to ``lr / 100``.
    lbfgs_iters : int
        Maximum number of L-BFGS iterations after Adam (strong-Wolfe line search). L-BFGS
        stops early when the loss decreases by less than ``lbfgs_rtol`` (relative) over
        ``lbfgs_window`` consecutive chunks of ``lbfgs_chunk`` iterations.
    grad_clip : float, optional
        Clip the global gradient norm during Adam.
    weight_update : callable, optional
        Called as ``weight_update(it)`` before every Adam step; used by adaptive
        loss-weighting schemes that modify weights stored outside this function.
    """
    params = [p for p in params if p.requires_grad]
    hist = History()
    lr_final = lr / 100.0 if lr_final is None else lr_final
    t0 = time.perf_counter()

    def lr_at(k: int) -> float:
        if schedule == "constant" or adam_iters <= 1:
            return lr
        frac = k / max(adam_iters - 1, 1)
        if schedule == "cosine":
            return lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * frac))
        if schedule == "exp":
            return lr * (lr_final / lr) ** frac
        raise ValueError(schedule)

    def maybe_eval(it: int) -> None:
        if eval_fn is not None and it >= 0 and (it % eval_every == 0):
            with torch.enable_grad():
                hist.record_metrics(it, eval_fn())

    def maybe_snapshot(it: int) -> None:
        if snapshot_fn is not None and snapshot_every and it % snapshot_every == 0:
            hist.snapshots.append(np.asarray(snapshot_fn()))
            hist.snapshot_iters.append(it)

    # ------------------------------------------------------------------ Adam
    if adam_iters > 0:
        opt = torch.optim.Adam(params, lr=lr)
        for k in range(adam_iters):
            for g in opt.param_groups:
                g["lr"] = lr_at(k)
            if weight_update is not None:
                weight_update(k)
            opt.zero_grad(set_to_none=True)
            loss, parts = loss_fn()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            if k % log_every == 0 or k == adam_iters - 1:
                hist.record(k, "adam", _to_float(loss), parts, opt.param_groups[0]["lr"],
                            time.perf_counter() - t0)
            maybe_eval(k)
            maybe_snapshot(k)
            if verbose and print_every and (k % print_every == 0 or k == adam_iters - 1):
                comp = " ".join(f"{n}={_to_float(v):.2e}" for n, v in parts.items())
                print(f"  adam {k:6d}  loss={_to_float(loss):.3e}  {comp}")
            if not torch.isfinite(loss):
                print("  non-finite loss encountered; stopping Adam")
                break
            opt.step()
    hist.switch_iter = adam_iters if lbfgs_iters > 0 else None

    # ------------------------------------------------------------------ L-BFGS
    if lbfgs_iters > 0:
        # NOTE: torch's default max_eval = 1.25*max_iter would leave the strong-Wolfe line
        # search no evaluations when max_iter is small, so max_eval is set explicitly.
        chunk = max(1, min(lbfgs_chunk, lbfgs_iters))
        opt = torch.optim.LBFGS(params, lr=lbfgs_lr, max_iter=chunk, max_eval=25 * chunk,
                                history_size=lbfgs_history, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-14, tolerance_change=1e-16)
        last = {}

        def closure():
            opt.zero_grad(set_to_none=True)
            loss, parts = loss_fn()
            loss.backward()
            last["loss"], last["parts"] = loss, parts
            return loss

        state = opt.state[params[0]]
        recent: List[float] = []
        done_iters, n_chunk = 0, 0
        while done_iters < lbfgs_iters:
            opt.step(closure)
            n_chunk += 1
            new_total = int(state.get("n_iter", done_iters + chunk))
            progressed = new_total > done_iters
            done_iters = new_total if progressed else done_iters + chunk
            it = adam_iters + done_iters
            with torch.enable_grad():          # loss at the accepted point
                cur_t, cur_parts = loss_fn()
            cur = _to_float(cur_t)
            hist.record(it, "lbfgs", cur, cur_parts, lbfgs_lr, time.perf_counter() - t0)
            if eval_fn is not None and n_chunk % max(1, eval_every // chunk) == 0:
                with torch.enable_grad():
                    hist.record_metrics(it, eval_fn())
            if snapshot_fn is not None and snapshot_every and n_chunk % max(1, snapshot_every // chunk) == 0:
                hist.snapshots.append(np.asarray(snapshot_fn()))
                hist.snapshot_iters.append(it)
            if verbose and print_every and (n_chunk % max(1, print_every // (10 * chunk)) == 0):
                print(f"  lbfgs {done_iters:5d}  loss={cur:.3e}")
            if not math.isfinite(cur):
                print("  non-finite loss in L-BFGS; stopping")
                break
            recent.append(cur)
            if len(recent) > lbfgs_window:
                old = recent.pop(0)
                if (old - cur) <= lbfgs_rtol * abs(cur):
                    if verbose:
                        print(f"  L-BFGS stopped at iteration {done_iters}: relative decrease "
                              f"< {lbfgs_rtol:g} over {lbfgs_window * chunk} iterations (loss={cur:.3e})")
                    break
    # final evaluation
    if eval_fn is not None:
        last_it = (hist.iters[-1] if hist.iters else 0)
        if not hist.eval_iters or hist.eval_iters[-1] != last_it:
            hist.record_metrics(last_it, eval_fn())
    hist.train_time = time.perf_counter() - t0
    return hist
