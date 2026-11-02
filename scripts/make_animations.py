#!/usr/bin/env python3
"""Train small PINNs while recording snapshots, then render the README motion gallery:

* animated "doodle" SVGs (hand-drawn wobble + SMIL morphing of the PINN curve through the
  real training snapshots), and
* GIF animations in a sketch (xkcd) style and in the book style.

    python scripts/make_animations.py               # ~5-10 minutes on one CPU core
    python scripts/make_animations.py --from-cache  # re-render artwork from saved snapshots

Everything shown is a genuine training run of the corresponding book example (smaller
networks / fewer iterations than in the book so that the script stays fast).
"""
from __future__ import annotations

import io
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinnode import (ConstrainedModel, MLP, configure_torch, derivative, set_seed,  # noqa: E402
                     to_tensor, train, volterra_gauss)
from pinnode.reference import ivp_reference  # noqa: E402

OUT = ROOT / "assets" / "motion"
BG0, BG1, BG2 = "#0b1026", "#16123f", "#2a0f45"
CYAN, VIOLET, PINK, AMBER, MINT, INK, MUTED = ("#22d3ee", "#a78bfa", "#f472b6", "#fbbf24", "#34d399",
                                               "#e5e7f0", "#9aa3c7")


def log_snap_iters(n_total: int, k: int = 22):
    its = np.unique(np.round(np.geomspace(1, n_total, k)).astype(int))
    return set(int(i) for i in its) | {0}


def select(hist, k: int = 24):
    """Indices of k snapshots, geometrically spaced in recording order, first and last included."""
    n = len(hist.snapshot_iters)
    idx = np.unique(np.round(np.geomspace(1, n, k)).astype(int) - 1)
    return sorted(set(int(i) for i in idx) | {0, n - 1})


# ----------------------------------------------------------------------------- runs
def run_logistic():
    R, Y0, T = 2.0, 0.1, 2.0
    set_seed(7)
    model = ConstrainedModel(MLP(1, 1, 24, 2, "tanh", domain=(0, T)), lambda t, n: Y0 + t * n)
    tc = to_tensor(np.linspace(0, T, 80), requires_grad=True)
    tg = np.linspace(0, T, 140)
    tgt = to_tensor(tg)
    snaps, its = [], []
    keep = log_snap_iters(3000)

    def loss_fn():
        y = model(tc)
        r = derivative(y, tc) - R * y * (1 - y)
        return (r ** 2).mean(), {}

    def snap():
        with torch.no_grad():
            return model(tgt).numpy().ravel()

    hist = train(model.parameters(), loss_fn, adam_iters=3000, lr=3e-3, lbfgs_iters=0, log_every=5,
                 snapshot_fn=snap, snapshot_every=1, verbose=False)
    sel = select(hist, 22)
    exact = Y0 * np.exp(R * tg) / (1 + Y0 * (np.exp(R * tg) - 1))
    return dict(name="logistic", t=tg, exact=exact, snaps=[hist.snapshots[i] for i in sel],
                iters=[hist.snapshot_iters[i] for i in sel], loss_it=np.array(hist.iters), loss=np.array(hist.loss),
                title="Logistic ODE   y' = r y (1 - y)", ylab="y(t)")


def run_oscillator():
    """Damped Duffing-type oscillator y'' + 0.2 y' + y + y^3 = 0, y(0)=1, y'(0)=0 on [0, 8]."""
    T = 8.0
    set_seed(11)
    model = ConstrainedModel(MLP(1, 1, 40, 3, "tanh", domain=(0, T)), lambda t, n: 1.0 + (t / T) ** 2 * n)
    tc = to_tensor(np.linspace(0, T, 240), requires_grad=True)
    tg = np.linspace(0, T, 160)
    tgt = to_tensor(tg)
    keep = log_snap_iters(9000, 26)

    def loss_fn():
        y = model(tc)
        y1 = derivative(y, tc)
        y2 = derivative(y1, tc)
        r = y2 + 0.2 * y1 + y + y ** 3
        return (r ** 2).mean(), {}

    def snap():
        with torch.no_grad():
            return model(tgt).numpy().ravel()

    hist = train(model.parameters(), loss_fn, adam_iters=6000, lr=2e-3, lr_final=2e-5, lbfgs_iters=400,
                 log_every=10, snapshot_fn=snap, snapshot_every=1, verbose=False)
    sel = select(hist, 28)
    ref = ivp_reference(lambda t, z: [z[1], -0.2 * z[1] - z[0] - z[0] ** 3], (0, T), [1.0, 0.0], tg)[:, 0]
    return dict(name="oscillator", t=tg, exact=ref, snaps=[hist.snapshots[i] for i in sel],
                iters=[hist.snapshot_iters[i] for i in sel], loss_it=np.array(hist.iters), loss=np.array(hist.loss),
                title="Nonlinear oscillator   y'' + 0.2y' + y + y³ = 0", ylab="y(t)")


def run_lotka():
    a, b, c, d = 1.0, 0.1, 1.5, 0.075
    T = 8.0
    set_seed(5)
    # log-variables u = ln x, v = ln y keep the populations positive
    x0, y0 = 10.0, 5.0
    net = MLP(1, 2, 48, 3, "tanh", domain=(0, T))
    init = torch.tensor([[math.log(x0), math.log(y0)]])
    model = ConstrainedModel(net, lambda t, n: init + t * n)
    tc = to_tensor(np.linspace(0, T, 300), requires_grad=True)
    tg = np.linspace(0, T, 220)
    tgt = to_tensor(tg)
    keep = log_snap_iters(9000, 26)

    def loss_fn():
        z = model(tc)
        dz = derivative(z, tc)
        X, Y = torch.exp(z[:, :1]), torch.exp(z[:, 1:])
        r1 = dz[:, :1] - (a - b * Y)
        r2 = dz[:, 1:] - (-c + d * X)
        return (r1 ** 2).mean() + (r2 ** 2).mean(), {}

    def snap():
        with torch.no_grad():
            return torch.exp(model(tgt)).numpy()

    hist = train(model.parameters(), loss_fn, adam_iters=6000, lr=2e-3, lr_final=2e-5, lbfgs_iters=400,
                 log_every=10, snapshot_fn=snap, snapshot_every=1, verbose=False)
    sel = select(hist, 28)
    ref = ivp_reference(lambda t, z: [a * z[0] - b * z[0] * z[1], -c * z[1] + d * z[0] * z[1]], (0, T),
                        [x0, y0], tg)
    return dict(name="lotka", t=tg, exact=ref, snaps=[hist.snapshots[i] for i in sel],
                iters=[hist.snapshot_iters[i] for i in sel], loss_it=np.array(hist.iters), loss=np.array(hist.loss),
                title="Lotka-Volterra phase portrait", ylab="predator")


def run_volterra():
    """u(x) = 1 + 2 int_0^x t u(t) dt, exact exp(x^2), solved with a quadrature residual."""
    lam = 2.0
    set_seed(3)
    model = ConstrainedModel(MLP(1, 1, 32, 3, "tanh", domain=(0, 1)), lambda x, n: 1.0 + x * n)
    xc = np.linspace(0, 1, 64)
    S, W = volterra_gauss(xc, 16)
    St, Wt = to_tensor(S.reshape(-1, 1)), torch.as_tensor(W)
    xct = to_tensor(xc)
    xg = np.linspace(0, 1, 140)
    xgt = to_tensor(xg)
    keep = log_snap_iters(3000)

    def loss_fn():
        u = model(xct)
        us = model(St).reshape(S.shape)
        integral = (Wt * torch.as_tensor(S) * us).sum(dim=1, keepdim=True)
        r = u - 1.0 - lam * integral
        return (r ** 2).mean(), {}

    def snap():
        with torch.no_grad():
            return model(xgt).numpy().ravel()

    hist = train(model.parameters(), loss_fn, adam_iters=3000, lr=3e-3, lbfgs_iters=0, log_every=5,
                 snapshot_fn=snap, snapshot_every=1, verbose=False)
    sel = select(hist, 22)
    return dict(name="volterra", t=xg, exact=np.exp(lam * xg ** 2 / 2), snaps=[hist.snapshots[i] for i in sel],
                iters=[hist.snapshot_iters[i] for i in sel], loss_it=np.array(hist.iters), loss=np.array(hist.loss),
                title="Volterra integral equation   u = 1 + 2∫ t u", ylab="u(x)")


# ----------------------------------------------------------------------------- doodle SVG
def _poly(xs, ys):
    return "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in zip(xs, ys))


def doodle_svg(run: dict, phase: bool = False, W=600, H=380, color=CYAN, dur=9.0) -> str:
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_readme_assets import DOODLE_FILTER, text_path

    L, Rm, Tp, B = 58, 206, 64, 58
    pw, ph = W - L - Rm, H - Tp - B
    if phase:
        ex = run["exact"]
        xs_all = np.concatenate([ex[:, 0]] + [s[:, 0] for s in run["snaps"][-6:]])
        ys_all = np.concatenate([ex[:, 1]] + [s[:, 1] for s in run["snaps"][-6:]])
    else:
        xs_all = run["t"]
        ys_all = np.concatenate([run["exact"]] + list(run["snaps"][-8:]))
    x0, x1 = float(np.min(xs_all)), float(np.max(xs_all))
    y0, y1 = float(np.min(ys_all)), float(np.max(ys_all))
    pad = 0.08 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad
    padx = 0.03 * (x1 - x0)
    x0, x1 = x0 - padx, x1 + padx

    def X(v):
        return L + (np.asarray(v) - x0) / (x1 - x0) * pw

    def Y(v):
        return Tp + ph - (np.clip(np.asarray(v), y0 - 3 * pad, y1 + 3 * pad) - y0) / (y1 - y0) * ph

    if phase:
        ex_d = _poly(X(run["exact"][:, 0]), Y(run["exact"][:, 1]))
        frames = [_poly(X(s[:, 0]), Y(s[:, 1])) for s in run["snaps"]]
    else:
        ex_d = _poly(X(run["t"]), Y(run["exact"]))
        frames = [_poly(X(run["t"]), Y(s)) for s in run["snaps"]]
    n = len(frames)
    morph = 0.8  # fraction of the cycle spent morphing
    kts = [morph * (i / (n - 1)) ** 1.0 for i in range(n)] + [1.0]
    vals = ";".join(frames + [frames[-1]])
    kt = ";".join(f"{k:.4f}" for k in kts)
    # loss inset (log10 loss, drawn progressively)
    li, lv = run["loss_it"], np.log10(np.maximum(run["loss"], 1e-16))
    ix0, iy0, iw, ih = W - 182, Tp + 62, 158, 92
    lx = ix0 + (li - li.min()) / max(li.max() - li.min(), 1) * iw
    ly = iy0 + ih - (lv - lv.min()) / max(lv.max() - lv.min(), 1e-9) * ih
    step = max(1, len(lx) // 160)
    loss_d = _poly(lx[::step], ly[::step])
    loss_len = float(np.sum(np.hypot(np.diff(lx[::step]), np.diff(ly[::step])))) + 10
    title_d, _ = text_path(run["title"], 17, L, 36, weight="bold")
    lab_d, _ = text_path("training loss (log scale)", 11, ix0 + iw / 2, iy0 - 14, weight="normal", anchor="middle")
    leg1, _ = text_path("reference", 12, L + 34, H - 22, weight="normal")
    leg2, _ = text_path("PINN during training", 12, L + 160, H - 22, weight="normal")
    it_max = run["iters"][-1]
    prog_d, _ = text_path(f"iterations 0 → {it_max}", 11, ix0 + iw / 2, iy0 + ih + 26, weight="normal", anchor="middle")
    axes = (f'M{L} {Tp - 6} L{L} {Tp + ph} L{L + pw + 8} {Tp + ph}')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{run['title']}: PINN training animation">
  <defs>
    <linearGradient id="bgd" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{BG0}"/><stop offset="1" stop-color="{BG2}"/></linearGradient>
    {DOODLE_FILTER}
  </defs>
  <style>.loss {{ stroke-dasharray: {loss_len:.0f}; stroke-dashoffset: {loss_len:.0f}; animation: dl {dur}s linear infinite; }}
    @keyframes dl {{ 0% {{ stroke-dashoffset: {loss_len:.0f}; }} {morph * 100:.0f}%,100% {{ stroke-dashoffset: 0; }} }}</style>
  <rect width="{W}" height="{H}" rx="18" fill="url(#bgd)"/>
  <path d="{title_d}" fill="{INK}"/>
  <path d="{axes}" fill="none" stroke="{MUTED}" stroke-width="2" stroke-linecap="round" filter="url(#doodle)"/>
  <path d="{ex_d}" fill="none" stroke="{INK}" stroke-opacity="0.85" stroke-width="2.2" stroke-dasharray="7 6" filter="url(#doodle)"/>
  <path fill="none" stroke="{color}" stroke-width="3.2" stroke-linecap="round" stroke-linejoin="round" filter="url(#doodle)">
    <animate attributeName="d" values="{vals}" keyTimes="{kt}" dur="{dur}s" repeatCount="indefinite"/>
  </path>
  <rect x="{ix0 - 8}" y="{iy0 - 8}" width="{iw + 16}" height="{ih + 16}" rx="10" fill="{BG0}" fill-opacity="0.75" stroke="{MUTED}" stroke-opacity="0.35"/>
  <path class="loss" d="{loss_d}" fill="none" stroke="{AMBER}" stroke-width="2" filter="url(#doodle)"/>
  <path d="{lab_d}" fill="{MUTED}"/>
  <path d="M{L} {H - 26} l24 0" stroke="{INK}" stroke-width="2.2" stroke-dasharray="7 6"/>
  <path d="{leg1}" fill="{MUTED}"/>
  <path d="M{L + 128} {H - 26} l24 0" stroke="{color}" stroke-width="3.2"/>
  <path d="{leg2}" fill="{MUTED}"/>
  <path d="{prog_d}" fill="{MUTED}"/>
  <rect x="{L}" y="{H - 12}" height="3" rx="1.5" fill="{color}" width="0">
    <animate attributeName="width" values="0;{W - 24 - L};{W - 24 - L}" keyTimes="0;{morph};1" dur="{dur}s" repeatCount="indefinite"/>
  </rect>
</svg>
"""


# ----------------------------------------------------------------------------- GIFs
def gif(run: dict, path: Path, sketch: bool, phase: bool = False, color="#2a78d6"):
    import logging

    import imageio.v2 as imageio
    import matplotlib

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pinnode import plotting as P

    frames = []
    n = len(run["snaps"])
    seq = list(range(n)) + [n - 1] * 8
    for k in seq:
        ctx = plt.xkcd(scale=0.8, length=120, randomness=2) if sketch else plt.rc_context()
        with ctx:
            if not sketch:
                P.use_book_style(10)
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=90,
                                     gridspec_kw={"width_ratios": [1.6, 1]})
            ax, ax2 = axes
            s = run["snaps"][k]
            if phase:
                ax.plot(run["exact"][:, 0], run["exact"][:, 1], color="0.15", lw=1.4, ls="--", label="reference")
                ax.plot(s[:, 0], s[:, 1], color=color, lw=2.4, label="PINN")
                ax.set_xlabel("prey")
                ax.set_ylabel("predator")
            else:
                ax.plot(run["t"], run["exact"], color="0.15", lw=1.4, ls="--", label="reference")
                ax.plot(run["t"], s, color=color, lw=2.4, label="PINN")
                ax.set_xlabel("x" if run["name"] == "volterra" else "t")
                ax.set_ylabel(run["ylab"])
            ex = run["exact"]
            lo, hi = float(np.min(ex)), float(np.max(ex))
            pad = 0.15 * (hi - lo)
            if not phase:
                ax.set_ylim(lo - pad, hi + pad)
            ax.legend(loc="best", fontsize=8)
            it = run["iters"][k]
            m = run["loss_it"] <= it
            ax2.semilogy(run["loss_it"][m], run["loss"][m], color="#eb6834", lw=1.6)
            ax2.set_xlim(0, run["loss_it"].max())
            ax2.set_ylim(max(run["loss"].min() * 0.5, 1e-14), run["loss"].max() * 2)
            ax2.set_xlabel("iteration")
            ax2.set_title(f"loss, it = {it}", fontsize=10)
            fig.suptitle(run["title"].replace("\u222b", "$\\int$").replace("\u00b3", "$^3$"), fontsize=11)
            fig.tight_layout()
            buf = io.BytesIO()
            with plt.rc_context({"savefig.bbox": "standard"}):        # fixed frame size
                fig.savefig(buf, format="png", dpi=90)
            plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
    imageio.mimsave(path, frames, duration=0.16, loop=0)


def main():
    configure_torch("float64", threads=1)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    cache = OUT / "_cache"
    cache.mkdir(exist_ok=True)
    meta = {"logistic": ("Logistic ODE   y' = r y (1 - y)", "y(t)"),
            "volterra": ("Volterra integral equation   u = 1 + 2\u222b t u", "u(x)"),
            "oscillator": ("Nonlinear oscillator   y'' + 0.2y' + y + y\u00b3 = 0", "y(t)"),
            "lotka": ("Lotka-Volterra phase portrait", "predator")}
    for fn in (run_logistic, run_volterra, run_oscillator, run_lotka):
        name = fn.__name__.replace("run_", "")
        f = cache / f"{name}.npz"
        only = next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--only=")), None)
        use_cache = ("--from-cache" in sys.argv) or (only is not None and name not in only)
        if use_cache and f.exists():
            z = np.load(f)
            r = dict(name=name, t=z["t"], exact=z["exact"], snaps=list(z["snaps"]), iters=list(z["iters"]),
                     loss_it=z["loss_it"], loss=z["loss"], title=meta[name][0], ylab=meta[name][1])
        else:
            print("training", fn.__name__, flush=True)
            r = fn()
            np.savez_compressed(f, t=r["t"], exact=r["exact"], snaps=np.array(r["snaps"]),
                                iters=np.array(r["iters"]), loss_it=r["loss_it"], loss=r["loss"])
        runs[r["name"]] = r
    (OUT / "doodle_logistic.svg").write_text(doodle_svg(runs["logistic"], color=CYAN, dur=7))
    (OUT / "doodle_volterra.svg").write_text(doodle_svg(runs["volterra"], color=PINK, dur=7))
    (OUT / "doodle_oscillator.svg").write_text(doodle_svg(runs["oscillator"], color=MINT, dur=10))
    (OUT / "doodle_lotka_phase.svg").write_text(doodle_svg(runs["lotka"], phase=True, color=AMBER, dur=10))
    gif(runs["oscillator"], OUT / "oscillator_sketch.gif", sketch=True, color="#1baf7a")
    gif(runs["lotka"], OUT / "lotka_phase_sketch.gif", sketch=True, phase=True, color="#6250d6")
    gif(runs["logistic"], OUT / "logistic_training.gif", sketch=False, color="#2a78d6")
    gif(runs["volterra"], OUT / "volterra_training.gif", sketch=False, color="#e34948")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name:32s} {p.stat().st_size / 1024:8.1f} kB")


if __name__ == "__main__":
    main()
