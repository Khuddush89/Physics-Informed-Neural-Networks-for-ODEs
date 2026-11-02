"""Generate the LaTeX tables of Chapter 5 from the results.json files of the other scripts.

Book: Chapter 5, all tables (the chapter \\input-s figures/ch05/tab_*.tex).

Reads  outputs/ch05_bvp/{duffing_reference, duffing_pinn, duffing_classical,
boundary_layer}/results.json  (run those scripts first) and writes

    tab_duffing_roots.tex   solutions of the Duffing-type BVP (closed form)
    tab_duffing_pinn.tex    PINN variants for the Duffing-type BVP
    tab_classical.tex       PINNs versus collocation, shooting and finite differences
    tab_layer.tex           boundary-layer problem: three collocation-point distributions

Each file contains only the tabular environment; caption and label live in the chapter.
Times are CPU seconds of one thread by default (``--time wall`` for wall-clock seconds); on an
otherwise idle machine the two agree.

Run (from the repository root, after the other scripts):
    PYTHONPATH=. python chapters/ch05_bvp/make_tables.py --tex-dir ../book/figures/ch05
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from pinnode import Example
from pinnode.utils import OUTPUT_ROOT


def num(x, digits: int = 2) -> str:
    """Scientific notation for siunitx, e.g. \\num{1.2e-05}."""
    if x is None:
        return "--"
    if x == 0:
        return r"\num{0}"
    return f"\\num{{{x:.{digits - 1}e}}}"


def tnum(x: float) -> str:
    """Times with two significant digits in plain decimal notation."""
    if x >= 10:
        return f"\\num{{{x:.0f}}}"
    if x >= 1:
        return f"\\num{{{x:.1f}}}"
    if x >= 1e-3:
        return f"\\num{{{float(f'{x:.2g}'):g}}}"
    return num(x)


class Timer:
    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, r: dict) -> float:
        if self.mode == "cpu":
            return r["cpu_time_s"]
        return r.get("wall_time_s", r.get("train_time_s"))


def load(source: Path, name: str):
    p = source / name / "results.json"
    if not p.exists():
        print(f"  [skip] {p} not found")
        return None
    return json.loads(p.read_text())


def tab_roots(res) -> str:
    lines = [r"\begin{tabular}{@{}rcccc@{}}", r"\toprule",
             r"$j$ & $A_j=y_j(\pi)$ & $y_j'(0)$ & $\max_t\abs{y_j(t)}$ & zeros in $(0,\pi)$ \\",
             r"\midrule"]
    for r in res["solutions"]:
        lines.append(f"{r['k']} & \\num{{{r['A']:.6f}}} & \\num{{{r['dy0']:.5f}}} & "
                     f"\\num{{{r['max_abs_y']:.4f}}} & {r['zeros']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


PINN_LABELS = {"chnn_soft": r"ChNN, soft, $\lambda_{bc}=1$",
               "chnn_soft_w100": r"ChNN, soft, $\lambda_{bc}=100$",
               "chnn_hard": r"ChNN, hard",
               "mlp_soft": r"MLP, soft, $\lambda_{bc}=1$",
               "chnn_branch2": r"ChNN, soft, $\lambda_{bc}=1$, from guess"}


def tab_pinn(res, timer: Timer) -> str:
    lines = [r"\begin{tabular}{@{}lccccccr@{}}", r"\toprule",
             r"variant & solution & $\relLtwo$ & $\max\abs{e}$ & $\abs{y_\theta(0)-1}$ & "
             r"$\abs{y_\theta'(\pi)}$ & $\max\abs{\mathcal{R}}$ & time (s) \\", r"\midrule"]
    for r in res["runs"]:
        hard = r["hard"]
        lines.append(" & ".join([
            PINN_LABELS.get(r["name"], r["name"]), f"$j={r['branch_index']}$", num(r["rel_l2"]),
            num(r["linf"]), "--" if hard else num(r["bc_error_y0"]),
            "--" if hard else num(r["bc_error_dy_end"]), num(r["max_abs_residual_test"]),
            tnum(timer(r))]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def _setting(r) -> str:
    p = r["param"]
    if r["method"].startswith("collocation"):
        return f"$\\mathrm{{tol}}=10^{{{int(round(math.log10(p)))}}}$"
    if r["method"].startswith("shooting"):
        return f"$\\mathrm{{rtol}}=10^{{{int(round(math.log10(p)))}}}$"
    return f"$N={int(p)}$"


def tab_classical(cls, pinn, timer: Timer) -> str:
    names = {"collocation (solve_bvp)": r"collocation (\code{solve_bvp})",
             "shooting (DOP853 + Brent)": r"shooting (DOP853 + Brent)",
             "finite differences + Newton": r"finite differences + Newton"}
    lines = [r"\begin{tabular}{@{}llccr@{}}", r"\toprule",
             r"method & setting & $\relLtwo$ & $\max\abs{e}$ & time (s) \\", r"\midrule"]
    groups = {}
    for r in cls["rows"]:
        groups.setdefault(r["method"], []).append(r)
    blocks = []
    for method, rows in groups.items():
        blocks.append([" & ".join([names.get(method, method) if i == 0 else "", _setting(r),
                                   num(r["rel_l2"]), num(r["linf"]), tnum(timer(r))]) + r" \\"
                       for i, r in enumerate(rows)])
    if pinn is not None:
        block = []
        for name, lab in (("chnn_soft", r"soft, $\lambda_{bc}=1$"), ("chnn_hard", "hard")):
            rr = [x for x in pinn["runs"] if x["name"] == name]
            if rr:
                r = rr[0]
                block.append(" & ".join(["PINN (ChNN)" if not block else "", lab, num(r["rel_l2"]),
                                         num(r["linf"]), tnum(timer(r))]) + r" \\")
        if block:
            blocks.append(block)
    for i, b in enumerate(blocks):
        lines += b
        if i < len(blocks) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def tab_layer(res, timer: Timer) -> str:
    lab = {"uniform": "uniform", "cgl": "CGL", "shishkin": "Shishkin",
           "enriched": "CGL + layer function"}
    lines = [r"\begin{tabular}{@{}lcccccr@{}}", r"\toprule",
             r"points & in $[0,5\varepsilon]$ & $\relLtwo$ & $\max\abs{e}$ & final loss & "
             r"RMS of $\mathcal{R}$ & time (s) \\", r"\midrule"]
    for r in res["runs"]:
        lines.append(" & ".join([lab.get(r["distribution"], r["distribution"]), str(r["nodes_in_layer"]),
                                 num(r["rel_l2"]), num(r["linf"]), num(r["final_loss"]),
                                 num(r["rms_residual_test"]), tnum(timer(r))]) + r" \\")
    a = res.get("solve_bvp_tol1e-8")
    if a:
        lines.append(r"\midrule")
        lines.append(" & ".join([r"\code{solve_bvp} mesh ({%d} nodes)" % a["nodes"], str(a["nodes_in_layer"]),
                                 num(a["rel_l2"]), num(a["linf"]), "--", "--", "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ex = Example("ch05_bvp", "tables", description="LaTeX tables of Chapter 5 from results.json files",
                 extra_args=[(["--tex-dir"], {"default": None, "help": "where to write tab_*.tex"}),
                             (["--source"], {"default": None,
                                             "help": "folder with the example outputs (default outputs/ch05_bvp)"}),
                             (["--time"], {"choices": ["cpu", "wall"], "default": "cpu"})])
    source = Path(ex.args.source) if ex.args.source else (
        OUTPUT_ROOT / ("_quick" if ex.quick else "") / "ch05_bvp")
    tex_dir = Path(ex.args.tex_dir) if ex.args.tex_dir else ex.outdir / "tables"
    tex_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer(ex.args.time)
    ref, pinn, cls, layer = (load(source, n) for n in
                             ("duffing_reference", "duffing_pinn", "duffing_classical", "boundary_layer"))
    jobs = [("tab_duffing_roots.tex", ref, lambda: tab_roots(ref)),
            ("tab_duffing_pinn.tex", pinn, lambda: tab_pinn(pinn, timer)),
            ("tab_classical.tex", cls, lambda: tab_classical(cls, pinn, timer)),
            ("tab_layer.tex", layer, lambda: tab_layer(layer, timer))]
    written = []
    for fname, data, make in jobs:
        if data is None:
            continue
        (tex_dir / fname).write_text(make())
        written.append(fname)
        ex.log(f"wrote {tex_dir / fname}")
    ex.save_results({"source": str(source), "tex_dir": str(tex_dir), "time": ex.args.time, "written": written})


if __name__ == "__main__":
    main()
