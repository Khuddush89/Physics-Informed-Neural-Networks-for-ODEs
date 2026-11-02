"""Activation functions used in the book and their first and second derivatives.

Book: Chapter 2 (Approximation, optimization and error analysis), Section "Activation
functions" (the activation figure and the table of properties).

Activations: tanh, logistic sigmoid, sine (SIREN, omega_0 = 1), softplus, swish/SiLU,
GELU (exact form x Phi(x)) and ReLU.  Derivatives are computed by automatic
differentiation and checked against closed-form expressions; the script also records the
key numbers quoted in the text (values at 0, extrema of SiLU and GELU, maxima of |s'| and
|s''|) and the maximum deviation of the common tanh approximation of GELU.

Run:  PYTHONPATH=. python chapters/ch02_theory/activations.py          (a few seconds)
      PYTHONPATH=. python chapters/ch02_theory/activations.py --quick
"""
from __future__ import annotations

import math

import numpy as np
import torch

from pinnode import Example, to_tensor
from pinnode import plotting as P

SQ2 = math.sqrt(2.0)


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / SQ2))


def gelu_tanh(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


ACTS = {
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "sine": torch.sin,
    "softplus": torch.nn.functional.softplus,
    "silu": lambda x: x * torch.sigmoid(x),
    "gelu": gelu,
    "relu": torch.relu,
}
LABELS = {"tanh": "tanh", "sigmoid": "sigmoid", "sine": r"$\sin$", "softplus": "softplus",
          "silu": "SiLU (swish)", "gelu": "GELU", "relu": "ReLU"}
GROUPS = [["tanh", "sigmoid", "sine"], ["softplus", "silu", "gelu", "relu"]]


def closed_form(name, x):
    """(s, s', s'') in closed form (NumPy) for verification of the AD derivatives."""
    from scipy.special import erf

    if name == "tanh":
        th = np.tanh(x)
        return th, 1 - th ** 2, -2 * th * (1 - th ** 2)
    if name == "sigmoid":
        s = 1 / (1 + np.exp(-x))
        return s, s * (1 - s), s * (1 - s) * (1 - 2 * s)
    if name == "sine":
        return np.sin(x), np.cos(x), -np.sin(x)
    if name == "softplus":
        s = 1 / (1 + np.exp(-x))
        return np.log1p(np.exp(x)), s, s * (1 - s)
    if name == "silu":
        s = 1 / (1 + np.exp(-x))
        return x * s, s * (1 + x * (1 - s)), s * (1 - s) * (2 + x * (1 - 2 * s))
    if name == "gelu":
        Phi = 0.5 * (1 + erf(x / np.sqrt(2)))
        phi = np.exp(-x ** 2 / 2) / np.sqrt(2 * np.pi)
        return x * Phi, Phi + x * phi, (2 - x ** 2) * phi
    if name == "relu":
        return np.maximum(x, 0), (x > 0).astype(float), np.zeros_like(x)
    raise ValueError(name)


def derivatives_ad(fn, x: np.ndarray):
    xt = to_tensor(x, requires_grad=True)
    y = fn(xt)
    d1, = torch.autograd.grad(y.sum(), xt, create_graph=True)
    d2, = torch.autograd.grad(d1.sum(), xt, create_graph=False, allow_unused=True)
    d2 = torch.zeros_like(xt) if d2 is None else d2
    return (y.detach().numpy().ravel(), d1.detach().numpy().ravel(), d2.detach().numpy().ravel())


def main():
    ex = Example("ch02_theory", "activations",
                 description="Activation functions and their first two derivatives",
                 default_seed=1234)
    x = np.linspace(-4.0, 4.0, ex.pick(1601, 161))
    data = {"x": x}
    checks = {}
    for name, fn in ACTS.items():
        s0, s1, s2 = derivatives_ad(fn, x)
        c0, c1, c2 = closed_form(name, x)
        mask = np.abs(x) > 1e-12 if name == "relu" else np.ones_like(x, bool)
        checks[name] = float(max(np.max(np.abs(s0 - c0)), np.max(np.abs(s1 - c1)[mask]),
                                 np.max(np.abs(s2 - c2)[mask])))
        data[f"{name}"], data[f"{name}_d1"], data[f"{name}_d2"] = s0, s1, s2

    # key numbers on a fine grid over a wide interval
    xf = np.linspace(-12.0, 12.0, 240001)
    key = {}
    for name in ACTS:
        c0, c1, c2 = closed_form(name, xf)
        i0 = np.argmin(np.abs(xf))
        key[name] = {"s(0)": float(c0[i0]), "s'(0)": float(c1[i0]),
                     "s''(0)": float(c2[i0]) if name != "relu" else None,
                     "max|s'|": float(np.max(np.abs(c1))),
                     "max|s''|": float(np.max(np.abs(c2))) if name != "relu" else None,
                     "min s": float(np.min(c0)), "argmin s": float(xf[np.argmin(c0)]),
                     "min s'": float(np.min(c1)), "argmin s'": float(xf[np.argmin(c1)]),
                     "max s'": float(np.max(c1)), "argmax s'": float(xf[np.argmax(c1)])}
    xg = to_tensor(xf)
    with torch.no_grad():
        dev = float(torch.max(torch.abs(gelu(xg) - gelu_tanh(xg))).item())
    softplus_gap = {f"|softplus-relu| at |x|={a}": float(np.log1p(np.exp(-a))) for a in (2.0, 3.0)}
    for name, k in key.items():
        ex.log(f"{name:8s} s'(0)={k[chr(115) + chr(39) + '(0)']:.4f} max|s'|={k['max|s' + chr(39) + '|']:.4f} "
               f"min s={k['min s']:.4f} at {k['argmin s']:.4f}; AD check {checks[name]:.1e}")
    ex.log(f"max |GELU - tanh approximation| = {dev:.2e}")
    ex.save_table(data, "activation_values")
    ex.save_results({"ad_vs_closed_form_max_deviation": checks, "key_numbers": key,
                     "gelu_tanh_approx_max_deviation": dev, **softplus_gap})

    if ex.args.no_figures:
        return
    fig, axes = P.figure(3, 2, aspect=0.52, sharex=True)
    ylabels = [r"$\sigma(x)$", r"$\sigma'(x)$", r"$\sigma''(x)$"]
    for col, group in enumerate(GROUPS):
        for row, suffix in enumerate(["", "_d1", "_d2"]):
            ax = axes[row, col]
            for k, name in enumerate(group):
                y = data[f"{name}{suffix}"]
                ls = "-" if k == 0 else P.LINESTYLES[k - 1]
                if name == "relu" and suffix == "_d1":       # do not draw the jump as a line
                    ax.plot(x[x < 0], y[x < 0], color=P.SERIES[k], ls=ls, lw=1.2, label=LABELS[name])
                    ax.plot(x[x > 0], y[x > 0], color=P.SERIES[k], ls=ls, lw=1.2)
                    ax.plot([0], [0], marker="o", ms=3, mfc="white", mec=P.SERIES[k], ls="none")
                    ax.plot([0], [1], marker="o", ms=3, mfc="white", mec=P.SERIES[k], ls="none")
                else:
                    ax.plot(x, y, color=P.SERIES[k], ls=ls, lw=1.2, label=LABELS[name])
            ax.set_ylabel(ylabels[row])
            if row == 2:
                ax.set_xlabel("$x$")
        axes[0, col].legend(loc="upper left", ncol=1)
    axes[0, 0].set_title("bounded activations")
    axes[0, 1].set_title("ReLU-type activations")
    P.label_panels(axes)
    ex.savefig(fig, "activations")


if __name__ == "__main__":
    main()
