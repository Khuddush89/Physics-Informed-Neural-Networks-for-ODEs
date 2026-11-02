"""Write the LaTeX tables of Chapter 6 from the results.json files of the chapter scripts.

Every number in these tables is read from ``outputs/ch06_tvp/<example>/results.json``; nothing
is typed by hand.  Run the chapter scripts first, then

    PYTHONPATH=. python chapters/ch06_tvp/_make_tables.py                       # -> outputs/ch06_tvp/tables
    PYTHONPATH=. python chapters/ch06_tvp/_make_tables.py --dest ../book/figures/ch06

The files contain only the ``tabular`` environments; captions and labels live in the chapter.
Run times depend on the machine and its load; the ratios in tab_runtime.tex are the robust
quantity (both times are measured within the same script run).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ch06_tvp"


def load(name):
    p = OUT / name / "results.json"     # OUT may be replaced by --outputs
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run chapters/ch06_tvp/{name}.py first")
    return json.loads(p.read_text())


def sci(x, digits=2):
    """\\num{a.be-x} with the given number of significant digits."""
    x = float(x)
    if x == 0.0 or not math.isfinite(x):
        return r"\num{0}" if x == 0.0 else "--"
    mant, exp = f"{x:.{digits - 1}e}".split("e")
    return rf"\num{{{mant}e{int(exp)}}}"


def fix(x, decimals=1):
    return rf"\num{{{float(x):.{decimals}f}}}"


def write(dest: Path, name: str, lines):
    dest.mkdir(parents=True, exist_ok=True)
    (dest / name).write_text("\n".join(lines) + "\n")
    print(f"wrote {dest / name}")


def table_examples(dest):
    pend, jerk = load("pendulum_tvp"), load("jerk_tvp")
    rows = []
    nets = {"soft": r"LeNN, $m=7$, $2\times 32$", "hard": r"LeNN, $m=7$, $2\times 32$"}
    for r in pend["summary"]:
        term = "exact" if r["variant"] == "hard" else sci(max(r["err_yT"], r["err_dyT"]))
        cond = r"hard" if r["variant"] == "hard" else rf"soft, $\lambda_{{tc}}={pend['hyperparameters']['lambda_tc']:g}$"
        rows.append(("Pendulum", nets[r["variant"]], cond, sci(r["rel_l2"]), sci(r["linf"]), term,
                     sci(r["max_abs_residual"])))
    jnets = {"leg-soft": r"LeNN, $m=7$, $2\times 20$", "mlp-soft": r"MLP, $3\times 32$",
             "mlp-hard": r"MLP, $3\times 32$"}
    for r in jerk["summary"]:
        hard = r["variant"].endswith("hard")
        term = "exact" if hard else sci(max(r["err_y_T"], r["err_dy_T"], r["err_ddy_T"]))
        cond = "hard" if hard else r"soft, $\lambda_{tc}=200$"
        rows.append(("Third order", jnets[r["variant"]], cond, sci(r["rel_l2"]), sci(r["linf"]),
                     term, sci(r["max_abs_residual"])))
    lines = [r"\begin{tabular}{@{}lllcccc@{}}", r"\toprule",
             r"Problem & Network & Terminal data & rel.\ $L^2$ & max error & terminal error & $\max\abs{\Res}$ \\",
             r"\midrule"]
    for i, row in enumerate(rows):
        if i == len(pend["summary"]):
            lines.append(r"\midrule")
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(dest, "tab_examples.tex", lines)


def table_adjoint(dest):
    a = load("adjoint_sensitivity")
    comps = [r"$\partial J/\partial r$", r"$\partial J/\partial K$"]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & finite differences & classical adjoint & PINN adjoint & rel.\ error (PINN) \\",
             r"\midrule"]
    for i in range(2):
        fd, ca, pa = a["grad_fd"][i], a["grad_classical_adjoint"][i], a["grad_pinn"][i]
        lines.append(rf"{comps[i]} & \num{{{fd:.10f}}} & \num{{{ca:.10f}}} & \num{{{pa:.10f}}} & "
                     rf"{sci(a['rel_err_grad_pinn_components'][i])} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(dest, "tab_adjoint.tex", lines)


def table_lqr(dest):
    ric, fb = load("lqr_riccati"), load("lqr_forward_backward")
    lines = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"PINN & Quantity & rel.\ $L^2$ error & max error \\", r"\midrule",
             rf"Riccati TVP \eqref{{tvp:eq:riccati}} & $P_{{\btheta}}(t)$ & {sci(ric['rel_l2'])} & {sci(ric['linf'])} \\",
             rf" & optimal cost $\tfrac12 x_0^\top P_{{\btheta}}(0)x_0$ & \multicolumn{{2}}{{c}}{{rel.\ error {sci(ric['rel_err_J'])}}} \\",
             r"\midrule",
             rf"Hamiltonian system \eqref{{tvp:eq:hamiltonian}} & state $x_{{\btheta}}(t)$ & {sci(fb['rel_l2_x'])} & {sci(fb['linf_x'])} \\",
             rf" & costate $\bm{{\lambda}}_{{\btheta}}(t)$ & {sci(fb['rel_l2_lambda'])} & {sci(fb['linf_lambda'])} \\",
             rf" & control $u_{{\btheta}}=-R^{{-1}}B^\top\bm{{\lambda}}_{{\btheta}}$ & {sci(fb['rel_l2_u'])} & {sci(fb['linf_u'])} \\",
             rf" & cost along $(x_{{\btheta}},u_{{\btheta}})$ & \multicolumn{{2}}{{c}}{{rel.\ error {sci(fb['rel_err_J'])}}} \\",
             rf" & cost of $u_{{\btheta}}$ applied to the system & \multicolumn{{2}}{{c}}{{rel.\ excess over $J^*$: {sci(fb['rel_excess_J_open_loop'])}}} \\",
             r"\bottomrule", r"\end{tabular}"]
    write(dest, "tab_lqr.tex", lines)


def table_runtime(dest):
    pend, jerk, adj = load("pendulum_tvp"), load("jerk_tvp"), load("adjoint_sensitivity")
    ric, fb, gro, sys_ = (load("lqr_riccati"), load("lqr_forward_backward"), load("growth_model"),
                          load("system_tvp"))
    rows = []
    for r in pend["summary"]:
        rows.append(("Pendulum", f"LeNN, {r['variant']}", r["train_time_s"], "DOP853, backward",
                     pend["problem"]["reference_time_s"]))
    names = {"leg-soft": "LeNN, soft", "mlp-soft": "MLP, soft", "mlp-hard": "MLP, hard"}
    for r in jerk["summary"]:
        rows.append(("Third order", names[r["variant"]], r["train_time_s"], "DOP853, backward",
                     jerk["problem"]["reference_time_s"]))
    rows.append(("Adjoint gradient", r"state/adjoint PINN", adj["train_time_s"],
                 "classical adjoint", adj["reference_time_s"]))
    rows.append(("LQR Riccati", "MLP, hard", ric["train_time_s"], "DOP853, backward",
                 ric["reference_time_s"]))
    rows.append(("LQR Hamiltonian", "MLP, hard", fb["train_time_s"], r"\code{solve_bvp}",
                 fb["reference_time_s"]))
    rows.append(("Optimal growth", "MLP, hard", gro["train_time_s"], r"\code{solve_bvp}",
                 gro["reference_time_s"]))
    rows.append(("System (Ex.)", "MLP, hard", sys_["train_time_s"], "DOP853, backward",
                 sys_["reference_time_s"]))
    lines = [r"\begin{tabular}{@{}llrlrr@{}}", r"\toprule",
             r"Problem & PINN & training [s] & reference method & reference [s] & ratio \\",
             r"\midrule"]
    for prob, var, tp, meth, tr in rows:
        lines.append(rf"{prob} & {var} & {fix(tp, 1)} & {meth} & {sci(tr, 2)} & {sci(tp / tr, 1)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write(dest, "tab_runtime.tex", lines)
    ratios = [tp / tr for _, _, tp, _, tr in rows]
    print(f"runtime ratios: min {min(ratios):.2e}, max {max(ratios):.2e}")


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(OUT / "tables"))
    ap.add_argument("--outputs", default=str(OUT),
                    help="folder with the <example>/results.json files (default outputs/ch06_tvp)")
    args = ap.parse_args()
    OUT = Path(args.outputs)
    dest = Path(args.dest)
    for fn in (table_examples, table_adjoint, table_lqr, table_runtime):
        try:
            fn(dest)
        except FileNotFoundError as err:
            print(f"skipped {fn.__name__}: {err}")


if __name__ == "__main__":
    main()
