"""Automatic differentiation: exactness and cost of input derivatives of a network.

Book: Chapter 1 (Formulation and loss design), Section "Automatic differentiation"
(figure on AD versus finite differences and the paragraph on the cost of higher derivatives).

Part (a), exactness.  For a one-hidden-layer tanh network
    N(t) = sum_i v_i tanh(w_i t + b_i) + c
the first two derivatives are known in closed form,
    N'(t)  = sum_i v_i w_i   (1 - tanh^2(z_i)),
    N''(t) = sum_i v_i w_i^2 (-2 tanh(z_i)) (1 - tanh^2(z_i)),   z_i = w_i t + b_i.
We compare reverse-mode AD (pinnode.derivative) and central finite differences with step h
against these formulas: AD agrees to rounding error, finite differences show the usual
trade-off between truncation error O(h^2) and rounding error O(eps/h^k).

Part (b), cost.  For the MLP used throughout Part I (width 32, depth 3, tanh, 200 points)
we count the floating-point operations of all matrix products (a dispatch-mode counter;
deterministic and independent of the machine load) of
  (i)  the residual alone, d^k N_theta/dt^k at all points (nested reverse mode), and
  (ii) one loss-and-gradient evaluation, L = mean_i (d^k N_theta/dt^k(t_i))^2 and grad_theta L,
for derivative orders k = 0, ..., 4, normalised by (ii) for k = 0.  Process CPU times are
stored in results.json for reference only (they depend on the machine and its load).

Run:  PYTHONPATH=. python chapters/ch01_formulation/ad_exactness_cost.py          (< 1 min)
      PYTHONPATH=. python chapters/ch01_formulation/ad_exactness_cost.py --quick
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from pinnode import Example, MLP, derivative, to_tensor
from pinnode import plotting as P

WIDTH_SHALLOW = 16
WIDTH, DEPTH, N_POINTS = 32, 3, 200
MAX_ORDER = 4


# ----------------------------------------------------------------------------- part (a)
def shallow_network(seed: int):
    g = torch.Generator().manual_seed(seed)
    w = 2.0 * torch.randn(WIDTH_SHALLOW, generator=g)
    b = torch.randn(WIDTH_SHALLOW, generator=g)
    v = torch.randn(WIDTH_SHALLOW, generator=g) / np.sqrt(WIDTH_SHALLOW)
    c = torch.randn(1, generator=g)
    net = MLP(1, 1, width=WIDTH_SHALLOW, depth=1, activation="tanh", domain=None)
    with torch.no_grad():
        lin1, lin2 = net.body[0], net.body[2]
        lin1.weight.copy_(w.reshape(-1, 1))
        lin1.bias.copy_(b)
        lin2.weight.copy_(v.reshape(1, -1))
        lin2.bias.copy_(c)
    return net, w.numpy(), b.numpy(), v.numpy(), float(c)


def closed_form(t, w, b, v, c):
    z = np.outer(t, w) + b
    th = np.tanh(z)
    n0 = th @ v + c
    n1 = (1.0 - th ** 2) @ (v * w)
    n2 = (-2.0 * th * (1.0 - th ** 2)) @ (v * w ** 2)
    return n0, n1, n2


def exactness_study(ex: Example):
    net, w, b, v, c = shallow_network(ex.seed)
    t = np.linspace(-1.0, 1.0, 101)
    _, d1, d2 = closed_form(t, w, b, v, c)
    tt = to_tensor(t, requires_grad=True)
    y = net(tt)
    ad1 = derivative(y, tt, 1).detach().numpy().ravel()
    ad2 = derivative(y, tt, 2).detach().numpy().ravel()
    err_ad1 = float(np.max(np.abs(ad1 - d1)))
    err_ad2 = float(np.max(np.abs(ad2 - d2)))

    def f(x):
        with torch.no_grad():
            return net(to_tensor(x)).numpy().ravel()

    hs = np.logspace(-1, -9, ex.pick(33, 9))
    fd1, fd2 = [], []
    f0 = f(t)
    for h in hs:
        fp, fm = f(t + h), f(t - h)
        fd1.append(float(np.max(np.abs((fp - fm) / (2 * h) - d1))))
        fd2.append(float(np.max(np.abs((fp - 2 * f0 + fm) / h ** 2 - d2))))
    fd1, fd2 = np.array(fd1), np.array(fd2)
    return {"h": hs, "fd1": fd1, "fd2": fd2, "ad1": err_ad1, "ad2": err_ad2,
            "scale1": float(np.max(np.abs(d1))), "scale2": float(np.max(np.abs(d2))),
            "best_fd1": float(fd1.min()), "best_h1": float(hs[fd1.argmin()]),
            "best_fd2": float(fd2.min()), "best_h2": float(hs[fd2.argmin()])}


# ----------------------------------------------------------------------------- part (b)
def residual_only(model, t, k):
    y = model(t)
    return derivative(y, t, k) if k > 0 else y


def loss_and_grad(model, t, k):
    model.zero_grad(set_to_none=True)
    loss = (residual_only(model, t, k) ** 2).mean()
    loss.backward()
    return loss


class OperationCounter(TorchDispatchMode):
    """Counts floating-point operations of matrix products (2 m k n for an (m,k)x(k,n)
    product) and, separately, the number of elements produced by all other operations."""

    def __init__(self):
        super().__init__()
        self.matmul_flops = 0
        self.other_elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        aten = torch.ops.aten
        if func is aten.mm.default:
            a, b = args[0], args[1]
            self.matmul_flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
        elif func is aten.addmm.default:
            a, b = args[1], args[2]
            self.matmul_flops += 2 * a.shape[0] * a.shape[1] * b.shape[1]
        elif isinstance(out, torch.Tensor):
            self.other_elements += out.numel()
        return out


def count_flops(fn, *args) -> int:
    with OperationCounter() as oc:
        fn(*args)
    return int(oc.matmul_flops)


def cpu_time(model, t, k, reps: int, trials: int) -> float:
    loss_and_grad(model, t, k)               # warm-up
    best = []
    for _ in range(trials):
        c0 = time.process_time()
        for _ in range(reps):
            loss_and_grad(model, t, k)
        best.append((time.process_time() - c0) / reps)
    return float(np.median(best))


def cost_study(ex: Example):
    torch.manual_seed(ex.seed)
    model = MLP(1, 1, width=WIDTH, depth=DEPTH, activation="tanh", domain=(0.0, 1.0))
    t = to_tensor(np.linspace(0.0, 1.0, N_POINTS), requires_grad=True)
    rows = []
    for k in range(MAX_ORDER + 1):
        fl_res = count_flops(residual_only, model, t, k)
        fl_all = count_flops(loss_and_grad, model, t, k)
        ct = cpu_time(model, t, k, reps=ex.pick(40, 2), trials=ex.pick(5, 1))
        rows.append({"order": k, "flops_residual": fl_res, "flops_loss_and_gradient": fl_all,
                     "cpu_time_s": ct})
    base = rows[0]["flops_loss_and_gradient"]
    for r in rows:
        r["ratio_residual"] = r["flops_residual"] / base
        r["ratio_loss_and_gradient"] = r["flops_loss_and_gradient"] / base
        r["cpu_time_ratio"] = r["cpu_time_s"] / rows[0]["cpu_time_s"]
    return rows


def main():
    ex = Example("ch01_formulation", "ad_exactness_cost",
                 description="AD exactness versus finite differences; cost of input derivatives",
                 default_seed=1234)
    a = exactness_study(ex)
    ex.log(f"AD errors: first {a['ad1']:.2e}, second {a['ad2']:.2e}; best FD: "
           f"first {a['best_fd1']:.2e} (h={a['best_h1']:.1e}), second {a['best_fd2']:.2e} "
           f"(h={a['best_h2']:.1e})")
    rows = cost_study(ex)
    for r in rows:
        ex.log(f"order {r['order']}: flops residual={r['flops_residual']:.3e} "
               f"loss+grad={r['flops_loss_and_gradient']:.3e} (x{r['ratio_loss_and_gradient']:.2f}), "
               f"cpu x{r['cpu_time_ratio']:.2f}")
    ex.save_table({"h": a["h"], "fd_first": a["fd1"], "fd_second": a["fd2"]}, "fd_errors")
    ex.save_table(rows, "ad_cost")
    ex.save_results({
        "exactness": {"network": f"1 hidden layer, width {WIDTH_SHALLOW}, tanh",
                      "points": "101 uniform points in [-1, 1]",
                      "ad_max_error_first": a["ad1"], "ad_max_error_second": a["ad2"],
                      "max_abs_first_derivative": a["scale1"], "max_abs_second_derivative": a["scale2"],
                      "fd_best_error_first": a["best_fd1"], "fd_best_h_first": a["best_h1"],
                      "fd_best_error_second": a["best_fd2"], "fd_best_h_second": a["best_h2"]},
        "cost": {"network": f"MLP width {WIDTH}, depth {DEPTH}, tanh", "points": N_POINTS,
                 "counter": "torch.utils.flop_counter (matrix products)", "rows": rows}})

    if ex.args.no_figures:
        return
    fig, axes = P.figure(1, 2, aspect=0.80)
    ax = axes[0]
    ax.loglog(a["h"], a["fd1"], color=P.SERIES[0], ls=P.LINESTYLES[0], marker=P.MARKERS[0],
              markevery=4, markerfacecolor="white", label="FD, first derivative")
    ax.loglog(a["h"], a["fd2"], color=P.SERIES[1], ls=P.LINESTYLES[1], marker=P.MARKERS[1],
              markevery=4, markerfacecolor="white", label="FD, second derivative")
    floor = 1e-17
    ax.axhline(max(a["ad1"], floor), color=P.SERIES[0], lw=0.9, ls="-", label="AD, first derivative")
    ax.axhline(max(a["ad2"], floor), color=P.SERIES[1], lw=0.9, ls=(0, (1, 1)),
               label="AD, second derivative")
    ax.set_xlabel("finite-difference step $h$")
    ax.set_ylabel("maximum absolute error")
    ax.invert_xaxis()
    lo, _ = ax.get_ylim()
    ax.set_ylim(lo, 1e9)                     # head room for the legend above the curves
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=6.5,
              handlelength=2.4)
    ax = axes[1]
    orders = [r["order"] for r in rows]
    ax.semilogy(orders, [r["ratio_loss_and_gradient"] for r in rows], color=P.SERIES[0],
                ls=P.LINESTYLES[0], marker=P.MARKERS[0], markerfacecolor="white",
                label="residual, loss and gradient")
    ax.semilogy(orders, [r["ratio_residual"] for r in rows], color=P.SERIES[2],
                ls=P.LINESTYLES[2], marker=P.MARKERS[2], markerfacecolor="white",
                label="residual only")
    ax.set_xticks(orders)
    ax.set_xlabel("derivative order $k$ in the residual")
    ax.set_ylabel("operations relative to $k=0$")
    ax.legend(loc="upper left")
    P.label_panels(axes)
    ex.savefig(fig, "ad_exactness_cost")


if __name__ == "__main__":
    main()
