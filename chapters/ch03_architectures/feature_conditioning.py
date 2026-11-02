"""Polynomial feature layers: derivative check and conditioning of the feature matrix.

Book: Chapter 3 (Architectures and exact enforcement of conditions), Section
"Functional-link polynomial networks", Figure arch:fig:features and Table arch:tab:conditioning.

No network is trained here. The script

1. checks the Chebyshev and Legendre feature maps of ``pinnode.networks`` (three-term
   recurrences differentiated by automatic differentiation) against independent NumPy
   evaluations of T_k^(r) and P_k^(r) (numpy.polynomial), r = 0, 1, 2, and against the
   endpoint values T_k'(1) = k^2, T_k''(1) = k^2 (k^2 - 1)/3, P_k'(1) = k(k+1)/2,
   P_k''(1) = (k-1)k(k+1)(k+2)/8;
2. computes the 2-norm condition number of the feature matrix Phi_{ik} = phi_k(t_i) on
   M = 200 uniformly spaced points of [0, 10] for four bases of degree m:
   raw monomials t^k, monomials xi^k of the scaled variable xi = 2t/10 - 1 in [-1, 1],
   Chebyshev polynomials T_k(xi) and Legendre polynomials P_k(xi);
3. plots T_0..T_5, P_0..P_5, the condition numbers versus the degree and the endpoint values
   T_k'(1), P_k'(1), T_k''(1), P_k''(1), which grow like k^(2r) (Markov's inequality).

Run:  PYTHONPATH=. python chapters/ch03_architectures/feature_conditioning.py          (a few seconds)
      PYTHONPATH=. python chapters/ch03_architectures/feature_conditioning.py --quick
"""
from __future__ import annotations

import numpy as np
import torch
from numpy.polynomial import chebyshev as C
from numpy.polynomial import legendre as L

from pinnode import Example, derivative, to_tensor
from pinnode import plotting as P
from pinnode.networks import chebyshev_features, legendre_features

A, B = 0.0, 10.0          # physical interval used in the experiments of this chapter
M_POINTS = 200            # number of (uniform) collocation points
DEGREES_TABLE = [4, 8, 12, 16, 20]
KAPPA_CAP = 1e15          # beyond ~1/eps the computed condition number is not meaningful


def feature_matrix(basis: str, t: np.ndarray, m: int) -> np.ndarray:
    xi = 2.0 * (t - A) / (B - A) - 1.0
    k = np.arange(m + 1)
    if basis == "monomial_t":
        return t[:, None] ** k[None, :]
    if basis == "monomial_xi":
        return xi[:, None] ** k[None, :]
    if basis == "chebyshev":
        return C.chebvander(xi, m)
    if basis == "legendre":
        return L.legvander(xi, m)
    raise ValueError(basis)


def _unit(k: int) -> np.ndarray:
    """Coefficient vector of the single basis polynomial of degree k."""
    c = np.zeros(k + 1)
    c[k] = 1.0
    return c


def derivative_check(m: int = 20, n_pts: int = 401) -> dict:
    """Max deviation between AD derivatives of the pinnode feature maps and NumPy."""
    xi = np.linspace(-1.0, 1.0, n_pts)
    x = to_tensor(xi, requires_grad=True)
    out = {}
    families = (("chebyshev", chebyshev_features, C.chebval, C.chebder),
                ("legendre", legendre_features, L.legval, L.legder))
    for name, fmap, val, der in families:
        F = fmap(x, m)                                       # (n_pts, m+1)
        worst = 0.0
        for r in range(3):
            ad = np.column_stack([derivative(F[:, k:k + 1], x, r).detach().numpy().ravel()
                                  for k in range(m + 1)])
            ref = np.column_stack([val(xi, der(_unit(k), r) if r else _unit(k)) for k in range(m + 1)])
            scale = np.maximum(1.0, np.max(np.abs(ref), axis=0))
            worst = max(worst, float(np.max(np.abs(ad - ref) / scale)))
        out[name] = worst
    # endpoint formulas (exact integers)
    k = np.arange(m + 1)
    T1 = np.array([C.chebval(1.0, C.chebder(_unit(j), 1)) for j in k])
    T2 = np.array([C.chebval(1.0, C.chebder(_unit(j), 2)) for j in k])
    P1 = np.array([L.legval(1.0, L.legder(_unit(j), 1)) for j in k])
    P2 = np.array([L.legval(1.0, L.legder(_unit(j), 2)) for j in k])
    endpoint = {
        "T'_k(1)=k^2": float(np.max(np.abs(T1 - k ** 2))),
        "T''_k(1)=k^2(k^2-1)/3": float(np.max(np.abs(T2 - k ** 2 * (k ** 2 - 1) / 3))),
        "P'_k(1)=k(k+1)/2": float(np.max(np.abs(P1 - k * (k + 1) / 2))),
        "P''_k(1)=(k-1)k(k+1)(k+2)/8": float(np.max(np.abs(P2 - (k - 1) * k * (k + 1) * (k + 2) / 8))),
    }
    return {"max_rel_deviation_AD_vs_numpy": out, "endpoint_formula_max_abs_error": endpoint,
            "degree_checked": m, "points": n_pts}


def main():
    ex = Example("ch03_architectures", "feature_conditioning",
                 description="Chebyshev/Legendre feature maps: derivative check and conditioning",
                 default_seed=1234)
    m_max = ex.pick(20, 8)
    t = np.linspace(A, B, M_POINTS)

    check = derivative_check(m=m_max)
    ex.log(f"AD vs NumPy (max rel. deviation, r=0,1,2): {check['max_rel_deviation_AD_vs_numpy']}")
    ex.log(f"endpoint formulas (max abs error): {check['endpoint_formula_max_abs_error']}")

    bases = ["monomial_t", "monomial_xi", "chebyshev", "legendre"]
    degrees = list(range(1, m_max + 1))
    kappa = {b: [float(np.linalg.cond(feature_matrix(b, t, m))) for m in degrees] for b in bases}
    rows = []
    for m in DEGREES_TABLE:
        if m > m_max:
            continue
        row = {"degree": m}
        for b in bases:
            row[b] = kappa[b][m - 1]
        rows.append(row)
        ex.log("m=%2d  " % m + "  ".join(f"{b}={row[b]:.3e}" for b in bases))
    ex.save_table({"degree": degrees, **kappa}, "condition_numbers")
    ex.save_results({
        "problem": {"interval": [A, B], "points": M_POINTS, "spacing": "uniform",
                    "kappa_cap_for_display": KAPPA_CAP},
        "derivative_check": check,
        "condition_numbers_table": rows,
    })

    if ex.args.no_figures:
        return
    fig, axes = P.figure(2, 2, aspect=0.72)
    ax_t, ax_p, ax_k, ax_d = axes.ravel()
    xi = np.linspace(-1.0, 1.0, 801)
    for k in range(6):
        style = {"color": P.SERIES[k], "ls": "-" if k == 0 else P.LINESTYLES[k - 1], "lw": 1.1}
        ax_t.plot(xi, C.chebval(xi, _unit(k)), label=f"$T_{k}$", **style)
        ax_p.plot(xi, L.legval(xi, _unit(k)), label=f"$P_{k}$", **style)
    for ax, title in ((ax_t, "Chebyshev $T_k(\\xi)$"), (ax_p, "Legendre $P_k(\\xi)$")):
        ax.set_xlabel("$\\xi$")
        ax.set_title(title)
        ax.set_ylim(-1.12, 1.78)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.legend(loc="upper center", ncol=6, fontsize=6.5, handlelength=1.6, columnspacing=0.7,
                  handletextpad=0.4)
    labels = {"monomial_t": "$t^k$, $t\\in[0,10]$", "monomial_xi": "$\\xi^k$",
              "chebyshev": "$T_k(\\xi)$", "legendre": "$P_k(\\xi)$"}
    for j, b in enumerate(bases):
        kap = np.minimum(np.array(kappa[b]), 1e17)
        ax_k.semilogy(degrees, kap, color=P.SERIES[j], ls="-" if j == 0 else P.LINESTYLES[j - 1],
                      marker=P.MARKERS[j], markevery=4, markersize=3.0, markerfacecolor="white",
                      lw=1.1, label=labels[b])
    eps_inv = 1.0 / np.finfo(float).eps
    ax_k.axhline(eps_inv, color=P.COLORS["muted"], lw=0.6)
    ax_k.text(degrees[-1], eps_inv / 40.0, "$1/\\epsilon_{\\mathrm{mach}}$", ha="right", va="top",
              fontsize=6.5, color=P.COLORS["muted"])
    ax_k.set_xlabel("degree $m$")
    ax_k.set_ylabel("$\\kappa_2(\\Phi)$")
    ax_k.set_title("condition of the feature matrix")
    ax_k.set_ylim(1.0, 1e25)
    ax_k.set_yticks([1e0, 1e5, 1e10, 1e15, 1e20])
    ax_k.legend(loc="upper left", ncol=2, fontsize=6.5)
    # endpoint values of first and second derivatives (growth ~ k^(2r))
    k1 = np.arange(1, m_max + 1)
    k2 = np.arange(2, m_max + 1)
    series = [(k1, k1 ** 2, "$T_k'(1)=k^2$"),
              (k1, k1 * (k1 + 1) / 2, "$P_k'(1)$"),
              (k2, k2 ** 2 * (k2 ** 2 - 1) / 3, "$T_k''(1)$"),
              (k2, (k2 - 1) * k2 * (k2 + 1) * (k2 + 2) / 8, "$P_k''(1)$")]
    for j, (kk, vals, lab) in enumerate(series):
        ax_d.semilogy(kk, vals, color=P.SERIES[j], ls="-" if j == 0 else P.LINESTYLES[j - 1],
                      marker=P.MARKERS[j], markevery=4, markersize=3.0, markerfacecolor="white",
                      lw=1.1, label=lab)
    ax_d.set_xlabel("degree $k$")
    ax_d.set_ylabel("value at $\\xi=1$")
    ax_d.set_title("endpoint derivatives")
    ax_d.legend(loc="lower right", ncol=2, fontsize=6.5, columnspacing=0.8)
    P.label_panels(axes)
    ex.savefig(fig, "features_conditioning")


if __name__ == "__main__":
    main()
