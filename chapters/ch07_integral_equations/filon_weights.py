"""Filon-type panel quadrature: verification of the explicit weights and a model integral.

Book: Chapter 7 (Integral equations), Section "Filon-type panel quadrature"
(figure ie:fig:filon, table ie:tab:filon).

(1) Weights.  alpha_j(theta) = int_0^1 e^{i theta xi} L_j(xi) dxi, j = 0, 1, 2, with the quadratic
    Lagrange basis L_j on the nodes 0, 1/2, 1, are evaluated in three ways and compared with a
    composite 20-point Gauss-Legendre rule (accurate to rounding) for |theta| in [1e-8, 1e3]:
      closed   - closed-form moments J_m(theta) for every theta;
      taylor3  - closed form, replaced by the three-term Taylor series for |theta| < 1e-3;
      stable   - 30-term power series for |theta| < 1, closed form otherwise (module _iecore).
    The partial-panel weights beta_j(theta, s) = int_0^s e^{i theta xi} L_j dxi are checked at
    random (theta, s) in the same way.
(2) Panel rules on a model integral  I(k) = int_0^1 e^{ikt} / (1 + t^2) dt,  k in [1, 1000],
    each with 17 nodes: global Gauss-Legendre, composite Simpson (8 panels) and composite
    Filon (8 panels).  Reference: scipy.integrate.quad with the oscillatory weights cos/sin (QAWO).

Run:  PYTHONPATH=. python chapters/ch07_integral_equations/filon_weights.py   (a few seconds)
      PYTHONPATH=. python chapters/ch07_integral_equations/filon_weights.py --quick
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import IntegrationWarning, quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _iecore import (filon_weights, moments_closed, moments_series,  # noqa: E402
                     weights_from_moments)
from pinnode import Example  # noqa: E402
from pinnode import plotting as P  # noqa: E402

LAGRANGE = [np.polynomial.Polynomial([1.0, -3.0, 2.0]), np.polynomial.Polynomial([0.0, 4.0, -4.0]),
            np.polynomial.Polynomial([0.0, -1.0, 2.0])]
Z20, W20 = leggauss(20)


def weights_reference(theta: float, s: float = 1.0) -> np.ndarray:
    """Composite 20-point Gauss-Legendre evaluation of beta_j(theta, s) (about 2 rad per panel)."""
    n_pan = max(1, int(np.ceil(abs(theta) * s / 2.0)))
    edges = np.linspace(0.0, s, n_pan + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    xi = (0.5 * (lo + hi) + 0.5 * (hi - lo) * Z20).ravel()
    w = (0.5 * (hi - lo) * W20).ravel()
    e = np.exp(1j * theta * xi)
    return np.array([np.sum(w * e * L(xi)) for L in LAGRANGE])


def weights_closed(theta):
    return weights_from_moments(moments_closed(theta))


def weights_taylor3(theta):
    if abs(theta) < 1e-3:
        J = np.array([1 + 1j * theta / 2 - theta ** 2 / 6, 0.5 + 1j * theta / 3 - theta ** 2 / 8,
                      1 / 3 + 1j * theta / 4 - theta ** 2 / 10])
        return weights_from_moments(J)
    return weights_closed(theta)


def rel_err(a, ref):
    return float(np.max(np.abs(a - ref)) / np.max(np.abs(ref)))


# ----------------------------------------------------------------------------- model integral
def phi(t):
    return 1.0 / (1.0 + t ** 2)


def model_reference(k: float) -> complex:
    # QUADPACK (QAWO) may flag roundoff at the requested tolerance 1e-14; the reference is
    # nevertheless far more accurate than the Filon/Simpson errors measured against it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        return _model_reference(k)


def _model_reference(k: float) -> complex:
    re = quad(phi, 0.0, 1.0, weight="cos", wvar=k, epsabs=1e-15, epsrel=1e-14, limit=400)[0]
    im = quad(phi, 0.0, 1.0, weight="sin", wvar=k, epsabs=1e-15, epsrel=1e-14, limit=400)[0]
    return re + 1j * im


def rule_gauss(k: float, n: int) -> complex:
    x, w = leggauss(n)
    t, w = 0.5 * (x + 1.0), 0.5 * w
    return np.sum(w * np.exp(1j * k * t) * phi(t))


def rule_simpson(k: float, panels: int) -> complex:
    t = np.linspace(0.0, 1.0, 2 * panels + 1)
    h = 1.0 / panels
    w = np.zeros_like(t)
    w[0:-1:2] += h / 6
    w[1::2] += 4 * h / 6
    w[2::2] += h / 6
    return np.sum(w * np.exp(1j * k * t) * phi(t))


def rule_filon(k: float, panels: int) -> complex:
    h = 1.0 / panels
    alpha = filon_weights(k * h)
    total = 0j
    for p in range(panels):
        a = p * h
        f = phi(a + h * np.array([0.0, 0.5, 1.0]))
        total += np.exp(1j * k * a) * h * np.sum(alpha * f)
    return total


def main():
    ex = Example("ch07_integral_equations", "filon_weights",
                 description="Filon weights: stable evaluation and accuracy on a model integral",
                 default_seed=1234)
    rng = np.random.default_rng(ex.seed)
    thetas = np.logspace(-8, 3, ex.pick(221, 23))
    rows = []
    for th in thetas:
        for sign in (+1.0, -1.0):
            ref = weights_reference(sign * th)
            rows.append({"theta": sign * th,
                         "closed": rel_err(weights_closed(sign * th), ref),
                         "taylor3": rel_err(weights_taylor3(sign * th), ref),
                         "stable": rel_err(filon_weights(sign * th), ref)})
    bins = [(1e-8, 1e-4), (1e-4, 1e-2), (1e-2, 1.0), (1.0, 1e3 + 1)]
    table = []
    for lo, hi in bins:
        sel = [r for r in rows if lo <= abs(r["theta"]) < hi]
        table.append({"theta_min": lo, "theta_max": min(hi, 1e3),
                      **{m: max(r[m] for r in sel) for m in ("closed", "taylor3", "stable")}})
    # partial weights at random (theta, s)
    part = []
    for _ in range(ex.pick(400, 20)):
        th = 10.0 ** rng.uniform(-8, 3) * rng.choice([-1.0, 1.0])
        s = rng.uniform(0.0, 1.0)
        part.append(rel_err(filon_weights(th, s), weights_reference(th, s)))
    # check: series and closed form agree where both are accurate (|theta| ~ 1)
    agree = float(np.max(np.abs(moments_series(np.array([0.999])) - moments_closed(np.array([0.999])))))
    # model integral
    ks = np.logspace(0, 3, ex.pick(61, 7))
    mrows = []
    for k in ks:
        ref = model_reference(k)
        mrows.append({"k": k, "gauss17": abs(rule_gauss(k, 17) - ref) / abs(ref),
                      "simpson17": abs(rule_simpson(k, 8) - ref) / abs(ref),
                      "filon17": abs(rule_filon(k, 8) - ref) / abs(ref)})
    ex.save_table(rows, "weights_errors")
    ex.save_table(table, "weights_error_bins")
    ex.save_table(mrows, "model_integral")
    ex.save_results({"weights_error_bins": table, "partial_weights_max_rel_err": float(np.max(part)),
                     "series_vs_closed_at_0.999": agree,
                     "simpson_limit_alpha_theta0": [complex(a) for a in filon_weights(0.0)],
                     "model_integral": {"phi": "1/(1+t^2)", "nodes": 17,
                                        "max_rel_err_k_ge_100": {m: max(r[m] for r in mrows if r["k"] >= 100)
                                                                 for m in ("gauss17", "simpson17", "filon17")}},
                     "train_time_s": 0.0})
    for r in table:
        ex.log(f"|theta| in [{r['theta_min']:.0e},{r['theta_max']:.0e}): closed={r['closed']:.1e} "
               f"taylor3={r['taylor3']:.1e} stable={r['stable']:.1e}")
    ex.log(f"partial weights: max rel err = {np.max(part):.1e}; series vs closed at 0.999: {agree:.1e}")

    if ex.args.no_figures:
        return
    fig, axes = P.figure(1, 2, aspect=0.8)
    ax = axes[0]
    th = np.array([r["theta"] for r in rows if r["theta"] > 0])
    for k, (m, lab) in enumerate((("closed", "closed form"), ("taylor3", "+ Taylor (3 terms)"),
                                  ("stable", "series, $|\\theta|<1$"))):
        e = np.maximum([r[m] for r in rows if r["theta"] > 0], 1e-17)
        ax.loglog(th, e, color=P.SERIES[k], ls=["-", P.LINESTYLES[0], P.LINESTYLES[1]][k], lw=1.2, label=lab)
    ax.set_xlabel("$\\theta=kh$")
    ax.set_ylabel("relative error of $\\alpha_j(\\theta)$")
    ax.set_ylim(1e-17, 1e3)
    ax.set_title("evaluation of the Filon weights")
    ax.legend(loc="upper right")
    ax = axes[1]
    kk = np.array([r["k"] for r in mrows])
    for k, (m, lab) in enumerate((("gauss17", "Gauss"), ("simpson17", "Simpson"), ("filon17", "Filon"))):
        ax.loglog(kk, np.maximum([r[m] for r in mrows], 1e-17), color=P.SERIES[k],
                  ls=["-", P.LINESTYLES[0], P.LINESTYLES[1]][k], marker=P.MARKERS[k], markevery=6,
                  markersize=3, markerfacecolor="white", lw=1.2, label=lab)
    ax.set_xlabel("$k$")
    ax.set_ylabel("relative error")
    ax.set_title("model integral, 17 nodes")
    ax.set_ylim(1e-18, 1e4)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.17))
    P.label_panels(axes)
    ex.savefig(fig, "filon_weights")


if __name__ == "__main__":
    main()
