#!/usr/bin/env python3
"""Generate the animated SVG artwork used in README.md (banner, typing line, section
headings, dividers).

All text is converted to vector outlines (DejaVu fonts shipped with Matplotlib), so the
SVGs render identically on GitHub (light or dark theme) without web fonts. Animations use
CSS keyframes and SMIL only, which GitHub renders inside <img> tags.

    python scripts/make_readme_assets.py        # writes assets/*.svg
"""
from __future__ import annotations

import math
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.textpath import TextPath

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
HEAD = ASSETS / "headings"

# Palette: night background + neon accents
BG0, BG1, BG2 = "#0b1026", "#16123f", "#2a0f45"
CYAN, SKY, VIOLET, PINK, AMBER, MINT = "#22d3ee", "#60a5fa", "#a78bfa", "#f472b6", "#fbbf24", "#34d399"
INK = "#e5e7f0"
MUTED = "#9aa3c7"


# ----------------------------------------------------------------------------- text -> path
def text_path(text: str, size: float, x: float = 0.0, y: float = 0.0, family: str = "DejaVu Sans",
              weight: str = "bold", anchor: str = "start", style: str = "normal"):
    """SVG path data for ``text`` with baseline at (x, y) in SVG coordinates."""
    prop = FontProperties(family=family, weight=weight, style=style)
    tp = TextPath((0, 0), text, size=size, prop=prop)
    ext = tp.get_extents()
    width = ext.x1 - ext.x0 if len(tp.vertices) else 0.0
    dx = {"start": 0.0, "middle": -width / 2 - ext.x0, "end": -width - ext.x0}[anchor]
    parts = []
    for verts, code in tp.iter_segments(curves=True, simplify=False):
        pts = [(x + dx + verts[i], y - verts[i + 1]) for i in range(0, len(verts), 2)]
        if code == MplPath.MOVETO:
            parts.append("M%.2f %.2f" % pts[0])
        elif code == MplPath.LINETO:
            parts.append("L%.2f %.2f" % pts[0])
        elif code == MplPath.CURVE3:
            parts.append("Q%.2f %.2f %.2f %.2f" % (*pts[0], *pts[1]))
        elif code == MplPath.CURVE4:
            parts.append("C%.2f %.2f %.2f %.2f %.2f %.2f" % (*pts[0], *pts[1], *pts[2]))
        elif code == MplPath.CLOSEPOLY:
            parts.append("Z")
    return "".join(parts), width


def wave_path(x0, x1, y0, amp, k, phase=0.0, n=160):
    pts = []
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        y = y0 + amp * math.sin(k * (x - x0) + phase) * (0.65 + 0.35 * math.sin(0.5 * k * (x - x0)))
        pts.append(("M" if i == 0 else "L") + "%.1f %.1f" % (x, y))
    return "".join(pts)


DOODLE_FILTER = """
  <filter id="doodle" x="-5%" y="-20%" width="110%" height="140%">
    <feTurbulence type="fractalNoise" baseFrequency="0.035" numOctaves="2" seed="1" result="n">
      <animate attributeName="seed" values="1;2;3;4" dur="0.48s" calcMode="discrete" repeatCount="indefinite"/>
    </feTurbulence>
    <feDisplacementMap in="SourceGraphic" in2="n" scale="2.6" xChannelSelector="R" yChannelSelector="G"/>
  </filter>
  <filter id="glow" x="-20%" y="-50%" width="140%" height="200%">
    <feGaussianBlur stdDeviation="3.2" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>"""


def shimmer_gradient(gid: str, colors=(CYAN, VIOLET, PINK, AMBER, CYAN), dur="6s"):
    stops = "".join(f'<stop offset="{i / (len(colors) - 1):.2f}" stop-color="{c}"/>' for i, c in enumerate(colors))
    return f"""
  <linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="0" gradientUnits="objectBoundingBox" spreadMethod="reflect">
    {stops}
    <animateTransform attributeName="gradientTransform" type="translate" values="-1 0;1 0;-1 0" dur="{dur}" repeatCount="indefinite"/>
  </linearGradient>"""


# ----------------------------------------------------------------------------- banner
def banner() -> str:
    W, H = 1200, 330
    title, tw = text_path("Physics-Informed Neural Networks", 41, 600, 124, anchor="middle")
    sub, _ = text_path("for Ordinary, Integral and Integro-Differential Equations", 22, 600, 166,
                       weight="normal", anchor="middle")
    cap, _ = text_path("THEORY  ·  ALGORITHMS  ·  PYTHON IMPLEMENTATIONS  ·  COMPANION CODE", 13.5,
                       610, 212, anchor="middle")
    chips = [("y′ = f(t, y)", 150, 268, CYAN), ("u(x) = g(x) + λ∫ K u", 410, 290, VIOLET),
             ("L(θ) = ‖R‖² + λ‖B‖²", 735, 270, PINK),
             ("y(T) = yₜ", 1010, 292, AMBER)]
    chip_svg = []
    for i, (txt, cx, cy, col) in enumerate(chips):
        d, w = text_path(txt, 17, cx, cy, family="DejaVu Sans", weight="normal", anchor="middle")
        chip_svg.append(f"""
    <g class="chip" style="animation-delay:{-i * 1.7:.1f}s">
      <rect x="{cx - w / 2 - 14:.1f}" y="{cy - 22:.1f}" width="{w + 28:.1f}" height="32" rx="16"
            fill="{BG1}" fill-opacity="0.92" stroke="{col}" stroke-opacity="0.75" stroke-width="1.3"/>
      <path d="{d}" fill="{col}"/>
    </g>""")
    waves = []
    for j, (col, amp, k, y0, dur) in enumerate([(CYAN, 16, 0.018, 250, "7s"), (VIOLET, 11, 0.026, 240, "9s"),
                                                 (PINK, 8, 0.034, 258, "11s")]):
        waves.append(f'<path d="{wave_path(0, W, y0, amp, k, j)}" fill="none" stroke="{col}" stroke-width="2" '
                     f'stroke-opacity="0.55" stroke-linecap="round" stroke-dasharray="14 10" filter="url(#doodle)">'
                     f'<animate attributeName="stroke-dashoffset" values="0;-240" dur="{dur}" repeatCount="indefinite"/></path>')
    # neural-network glyph (left) and integral glyph (right)
    nodes, edges = [], []
    layers = [(46, [96, 132, 168]), (86, [78, 114, 150, 186]), (126, [96, 132, 168]), (162, [132])]
    for (xa, ya), (xb, yb) in zip(layers[:-1], layers[1:]):
        for y1 in ya:
            for y2 in yb:
                edges.append(f'<line x1="{xa}" y1="{y1}" x2="{xb}" y2="{y2}" stroke="{SKY}" stroke-opacity="0.28" stroke-width="1"/>')
    k = 0
    for xl, ys in layers:
        for y in ys:
            col = [CYAN, VIOLET, PINK, AMBER][k % 4]
            nodes.append(f'<circle cx="{xl}" cy="{y}" r="6" fill="{col}" class="node" style="animation-delay:{0.18 * k:.2f}s"/>')
            k += 1
    integral, _ = text_path("∫", 120, 1085, 200, family="DejaVu Serif", weight="normal", anchor="middle")
    dydt, _ = text_path("d/dt", 24, 1112, 70, family="DejaVu Serif", weight="normal", style="italic", anchor="middle")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="Physics-Informed Neural Networks for Ordinary, Integral and Integro-Differential Equations">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{BG0}"/><stop offset="0.55" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/>
    </linearGradient>
    <radialGradient id="halo" cx="0.5" cy="0.35" r="0.6">
      <stop offset="0" stop-color="{VIOLET}" stop-opacity="0.30"/><stop offset="1" stop-color="{VIOLET}" stop-opacity="0"/>
    </radialGradient>
    {shimmer_gradient("title")}
    {DOODLE_FILTER}
    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
      <path d="M40 0H0V40" fill="none" stroke="#ffffff" stroke-opacity="0.05" stroke-width="1"/>
    </pattern>
  </defs>
  <style>
    .node {{ animation: pulse 2.4s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}
    @keyframes pulse {{ 0%,100% {{ opacity: .45; transform: scale(.8); }} 50% {{ opacity: 1; transform: scale(1.18); }} }}
    .chip {{ animation: drift 7s ease-in-out infinite; }}
    @keyframes drift {{ 0%,100% {{ transform: translateY(0px); }} 50% {{ transform: translateY(-7px); }} }}
    .title {{ animation: rise 1.4s ease-out; }}
    @keyframes rise {{ from {{ opacity: .35; transform: translateY(12px); }} to {{ opacity: 1; transform: translateY(0); }} }}
    .sub {{ animation: rise 1.7s ease-out; }}
    .cap {{ animation: rise 2.0s ease-out; }}
    .glyph {{ animation: float 6s ease-in-out infinite; }}
    @keyframes float {{ 0%,100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-9px); }} }}
  </style>
  <rect width="{W}" height="{H}" rx="22" fill="url(#bg)"/>
  <rect width="{W}" height="{H}" rx="22" fill="url(#grid)"/>
  <rect width="{W}" height="{H}" rx="22" fill="url(#halo)"/>
  {''.join(waves)}
  <g opacity="0.95">{''.join(edges)}{''.join(nodes)}</g>
  <g class="glyph" filter="url(#glow)"><path d="{integral}" fill="{AMBER}" fill-opacity="0.85"/>
     <path d="{dydt}" fill="{MINT}" fill-opacity="0.9"/></g>
  <g class="title" filter="url(#glow)"><path d="{title}" fill="url(#title)"/></g>
  <g class="sub"><path d="{sub}" fill="{INK}"/></g>
  <g class="cap"><path d="{cap}" fill="{MUTED}"/></g>
  {''.join(chip_svg)}
</svg>
"""


# ----------------------------------------------------------------------------- typing line
def typing(lines, W=980, H=64, period=3.2) -> str:
    n = len(lines)
    total = n * period
    prompt, pw = text_path("$", 20, 30, 40, family="DejaVu Sans Mono", weight="bold")
    groups = []
    for i, (txt, col) in enumerate(lines):
        d, w = text_path(txt, 19, 58, 40, family="DejaVu Sans Mono", weight="normal")
        a, b = i / n, (i + 1) / n
        k1, k2, k3 = a + 0.42 / n, b - 0.12 / n, b - 0.02 / n
        kt = f"0;{a:.4f};{k1:.4f};{k2:.4f};{k3:.4f};1"
        wv = f"0;0;{w + 6:.1f};{w + 6:.1f};0;0"
        xv = f"58;58;{58 + w + 6:.1f};{58 + w + 6:.1f};58;58"
        if i == 0:
            kt = f"0;{k1:.4f};{k2:.4f};{k3:.4f};1"
            wv = f"0;{w + 6:.1f};{w + 6:.1f};0;0"
            xv = f"58;{58 + w + 6:.1f};{58 + w + 6:.1f};58;58"
        groups.append(f"""
  <clipPath id="c{i}"><rect x="58" y="10" height="44" width="0">
    <animate attributeName="width" values="{wv}" keyTimes="{kt}" dur="{total}s" repeatCount="indefinite"/></rect></clipPath>
  <path d="{d}" fill="{col}" clip-path="url(#c{i})"/>
  <rect y="20" width="10" height="26" rx="2" fill="{col}" opacity="0.9">
    <animate attributeName="x" values="{xv}" keyTimes="{kt}" dur="{total}s" repeatCount="indefinite"/>
    <animate attributeName="opacity" values="{'0;1;1;0;0' if i == 0 else '0;0;1;1;0;0'}" keyTimes="{kt}" dur="{total}s" repeatCount="indefinite" calcMode="discrete"/>
  </rect>""")
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
     aria-label="{' | '.join(t for t, _ in lines)}">
  <defs><linearGradient id="tb" x1="0" x2="1"><stop offset="0" stop-color="{BG0}"/><stop offset="1" stop-color="{BG1}"/></linearGradient></defs>
  <rect width="{W}" height="{H}" rx="14" fill="url(#tb)" stroke="{VIOLET}" stroke-opacity="0.45"/>
  <circle cx="{W - 60}" cy="32" r="6" fill="#ff5f57"/><circle cx="{W - 40}" cy="32" r="6" fill="#febc2e"/><circle cx="{W - 20}" cy="32" r="6" fill="#28c840"/>
  <path d="{prompt}" fill="{MINT}"/>
  {''.join(groups)}
</svg>
"""


# ----------------------------------------------------------------------------- section heading
def heading(label: str, icon: str, colors, W=820, H=74) -> str:
    d, w = text_path(label, 34, 70, 46, anchor="start")
    ic, _ = text_path(icon, 34, 22, 48, family="DejaVu Sans", weight="bold", anchor="middle")
    gid = "g" + "".join(ch for ch in label if ch.isalnum())[:12]
    x_end = 70 + w
    ul = wave_path(70, x_end, 62, 3.0, 0.09, 0.0, n=90)
    length = 1.25 * (x_end - 70) + 40
    sparks = []
    for j, (sx, sy, r) in enumerate([(x_end + 22, 18, 5), (x_end + 40, 34, 3.2), (x_end + 14, 40, 2.4)]):
        sparks.append(f'<path d="M{sx} {sy - r * 2}L{sx + r * .5} {sy - r * .5}L{sx + r * 2} {sy}L{sx + r * .5} {sy + r * .5}'
                      f'L{sx} {sy + r * 2}L{sx - r * .5} {sy + r * .5}L{sx - r * 2} {sy}L{sx - r * .5} {sy - r * .5}Z" '
                      f'fill="{colors[(j + 1) % len(colors)]}" class="spark" style="animation-delay:{0.5 * j:.1f}s"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="{label}">
  <defs>{shimmer_gradient(gid, colors=(*colors, colors[0]), dur="5s")}{DOODLE_FILTER}</defs>
  <style>
    .t {{ animation: pop .9s cubic-bezier(.2,.9,.3,1.3); }}
    @keyframes pop {{ from {{ opacity: .4; transform: translateX(-14px); }} to {{ opacity: 1; transform: none; }} }}
    .ul {{ stroke-dasharray: {length:.0f}; stroke-dashoffset: {length:.0f}; animation: draw 2.6s .3s ease-in-out infinite alternate; }}
    @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
    .spark {{ animation: tw 1.8s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}
    @keyframes tw {{ 0%,100% {{ opacity: .2; transform: scale(.6) rotate(0deg); }} 50% {{ opacity: 1; transform: scale(1.15) rotate(45deg); }} }}
    .ic {{ animation: bob 3s ease-in-out infinite; transform-box: fill-box; transform-origin: center; }}
    @keyframes bob {{ 0%,100% {{ transform: rotate(-8deg); }} 50% {{ transform: rotate(8deg) translateY(-3px); }} }}
  </style>
  <g class="ic"><path d="{ic}" fill="url(#{gid})"/></g>
  <g class="t"><path d="{d}" fill="url(#{gid})"/></g>
  <path class="ul" d="{ul}" fill="none" stroke="url(#{gid})" stroke-width="3.2" stroke-linecap="round" filter="url(#doodle)"/>
  {''.join(sparks)}
</svg>
"""


def divider(W=1200, H=28) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="presentation">
  <defs>{shimmer_gradient("dv", dur="8s")}{DOODLE_FILTER}</defs>
  <path d="{wave_path(10, W - 10, 14, 6, 0.03)}" fill="none" stroke="url(#dv)" stroke-width="2.4" stroke-linecap="round"
        stroke-dasharray="18 9" filter="url(#doodle)"><animate attributeName="stroke-dashoffset" values="0;-270" dur="6s" repeatCount="indefinite"/></path>
</svg>
"""


def footer(W=1200, H=120) -> str:
    msg, _ = text_path("Residuals small, errors smaller, figures honest.", 20, W / 2, 70, weight="normal", anchor="middle")
    waves = "".join(
        f'<path d="{wave_path(0, W, 60 + 12 * j, 10 - 2 * j, 0.012 + 0.006 * j, j)}" fill="none" stroke="{c}" '
        f'stroke-opacity="0.5" stroke-width="2"><animateTransform attributeName="transform" type="translate" '
        f'values="0 0;{-60 + 30 * j} 0;0 0" dur="{8 + 2 * j}s" repeatCount="indefinite"/></path>'
        for j, c in enumerate([CYAN, VIOLET, PINK]))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="footer">
  <defs><linearGradient id="fb" x1="0" x2="1"><stop offset="0" stop-color="{BG0}"/><stop offset="1" stop-color="{BG2}"/></linearGradient>
  {shimmer_gradient("ft")}</defs>
  <rect width="{W}" height="{H}" rx="20" fill="url(#fb)"/>
  {waves}
  <path d="{msg}" fill="url(#ft)"/>
</svg>
"""


HEADINGS = {
    "about": ("About the book", "✦", (CYAN, SKY, VIOLET)),
    "motion": ("Motion gallery", "▶", (PINK, VIOLET, CYAN)),
    "chapters": ("Chapters and code", "≡", (AMBER, PINK, VIOLET)),
    "gallery": ("Figure gallery", "\u25a3", (MINT, VIOLET, PINK)),
    "quickstart": ("Quick start", "➜", (MINT, CYAN, SKY)),
    "library": ("The pinnode library", "λ", (VIOLET, PINK, AMBER)),
    "reproducibility": ("Reproducibility", "↻", (CYAN, MINT, AMBER)),
    "layout": ("Repository layout", "▤", (SKY, VIOLET, PINK)),
    "teaching": ("Exercises and notebooks", "✎", (AMBER, MINT, CYAN)),
    "citation": ("How to cite", "“", (PINK, AMBER, MINT)),
    "license": ("License and contributing", "♥", (VIOLET, CYAN, PINK)),
}

TYPING_LINES = [
    ("initial value problems   >  logistic, van der Pol, Lotka-Volterra", CYAN),
    ("boundary value problems  >  Chebyshev and Legendre PINNs, hard constraints", MINT),
    ("terminal problems        >  adjoint equations, Riccati, forward-backward LQR", AMBER),
    ("integral equations       >  Fredholm, Volterra, oscillatory Helmholtz kernels", VIOLET),
    ("integro-differential     >  A-PINN auxiliary variables, convolution kernels", PINK),
    ("practice                 >  failure modes, benchmarks, inverse problems, UQ", SKY),
]


def main():
    ASSETS.mkdir(exist_ok=True)
    HEAD.mkdir(exist_ok=True)
    (ASSETS / "banner.svg").write_text(banner())
    (ASSETS / "typing.svg").write_text(typing(TYPING_LINES))
    (ASSETS / "divider.svg").write_text(divider())
    (ASSETS / "footer.svg").write_text(footer())
    for key, (label, icon, cols) in HEADINGS.items():
        (HEAD / f"{key}.svg").write_text(heading(label, icon, cols))
    print("wrote", sorted(p.name for p in ASSETS.glob("*.svg")), "and", len(HEADINGS), "headings")


if __name__ == "__main__":
    main()
