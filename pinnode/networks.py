"""Network architectures used in the book.

All networks map inputs of shape ``(N, d_in)`` to outputs of shape ``(N, d_out)`` and are
smooth (C-infinity) whenever the activation is smooth, so automatic differentiation can
supply as many derivatives as the equation needs.

* :class:`MLP`                - fully connected network with input scaling to [-1, 1].
* :class:`ChebyshevNet`       - Chebyshev feature layer T_0..T_m followed by an MLP (ChNN).
* :class:`LegendreNet`        - Legendre feature layer P_0..P_m followed by an MLP (LeNN).
* :class:`FourierFeatureNet`  - random/deterministic Fourier features followed by an MLP.
* :class:`SirenNet`           - sine-activated network with the SIREN initialisation.
* :class:`ConstrainedModel`   - wraps any network with an output transform (hard constraints).

The feature networks reduce to the classical single-layer functional-link networks of
Mall and Chakraverty when ``hidden=()``.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

Domain = Union[Tuple[float, float], Sequence[Tuple[float, float]], None]


# ----------------------------------------------------------------------------- activations
class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class AdaptiveTanh(nn.Module):
    """tanh(n a x) with a trainable slope a (Jagtap et al. 2020); n is a fixed scale."""

    def __init__(self, n: float = 10.0):
        super().__init__()
        self.n = n
        self.a = nn.Parameter(torch.tensor(1.0 / n))

    def forward(self, x):
        return torch.tanh(self.n * self.a * x)


def make_activation(name: str) -> nn.Module:
    """Return an activation module from its name."""
    name = name.lower()
    table = {
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "relu": nn.ReLU,
        "softplus": nn.Softplus,
        "swish": nn.SiLU,
        "silu": nn.SiLU,
        "gelu": nn.GELU,
        "sin": Sine,
        "adaptive_tanh": AdaptiveTanh,
    }
    if name not in table:
        raise ValueError(f"unknown activation '{name}'; choose from {sorted(table)}")
    return table[name]()


# ----------------------------------------------------------------------------- scaling
class InputScaling(nn.Module):
    """Affine map of each input coordinate from [a_i, b_i] to [-1, 1]."""

    def __init__(self, domain: Domain, in_dim: int):
        super().__init__()
        if domain is None:
            lo = torch.full((in_dim,), -1.0)
            hi = torch.full((in_dim,), 1.0)
        else:
            if isinstance(domain[0], (int, float)):
                domain = [domain] * in_dim
            lo = torch.tensor([float(d[0]) for d in domain])
            hi = torch.tensor([float(d[1]) for d in domain])
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(self, x):
        return 2.0 * (x - self.lo) / (self.hi - self.lo) - 1.0


def _init_linear(layer: nn.Linear, scheme: str = "xavier") -> None:
    if scheme == "xavier":
        nn.init.xavier_normal_(layer.weight)
    elif scheme == "xavier_uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif scheme == "kaiming":
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    else:
        raise ValueError(scheme)
    nn.init.zeros_(layer.bias)


def _mlp_body(in_dim: int, out_dim: int, hidden: Sequence[int], activation: str,
              init: str = "xavier") -> nn.Sequential:
    layers = []
    dims = [in_dim, *hidden]
    for d0, d1 in zip(dims[:-1], dims[1:]):
        lin = nn.Linear(d0, d1)
        _init_linear(lin, init)
        layers += [lin, make_activation(activation)]
    last = nn.Linear(dims[-1], out_dim)
    _init_linear(last, "xavier")
    layers.append(last)
    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------- MLP
class MLP(nn.Module):
    """Fully connected network ``x -> W_{L+1} s(... s(W_1 x~ + b_1) ...) + b_{L+1}``.

    Parameters
    ----------
    in_dim, out_dim : int
        Input and output dimensions.
    width, depth : int
        Neurons per hidden layer and number of hidden layers.
    activation : str
        Name understood by :func:`make_activation`.
    domain : (a, b) or list of (a_i, b_i), optional
        Input box; inputs are mapped affinely to [-1, 1] before the first layer. This
        single line is the most effective "trick" for training PINNs on long intervals.
    """

    def __init__(self, in_dim: int = 1, out_dim: int = 1, width: int = 32, depth: int = 3,
                 activation: str = "tanh", domain: Domain = None, init: str = "xavier"):
        super().__init__()
        self.scale = InputScaling(domain, in_dim)
        self.body = _mlp_body(in_dim, out_dim, [width] * depth, activation, init)

    def forward(self, x):
        return self.body(self.scale(x))


# ----------------------------------------------------------------------------- polynomial features
def chebyshev_features(xi: torch.Tensor, degree: int) -> torch.Tensor:
    """T_0(xi), ..., T_degree(xi) by the three-term recurrence; xi has shape (N, 1)."""
    T = [torch.ones_like(xi), xi]
    for k in range(2, degree + 1):
        T.append(2.0 * xi * T[-1] - T[-2])
    return torch.cat(T[: degree + 1], dim=1)


def legendre_features(xi: torch.Tensor, degree: int) -> torch.Tensor:
    """P_0(xi), ..., P_degree(xi) by Bonnet's recurrence; xi has shape (N, 1)."""
    P = [torch.ones_like(xi), xi]
    for k in range(1, degree):
        P.append(((2 * k + 1) * xi * P[-1] - k * P[-2]) / (k + 1))
    return torch.cat(P[: degree + 1], dim=1)


class _PolynomialFeatureNet(nn.Module):
    feature_fn: Callable = None

    def __init__(self, degree: int = 8, hidden: Sequence[int] = (32, 32), out_dim: int = 1,
                 activation: str = "tanh", domain: Domain = (-1.0, 1.0),
                 output_activation: Optional[str] = None):
        super().__init__()
        self.degree = degree
        self.scale = InputScaling(domain, 1)
        self.body = _mlp_body(degree + 1, out_dim, list(hidden), activation)
        self.out_act = make_activation(output_activation) if output_activation else nn.Identity()

    def features(self, x):
        return type(self).feature_fn(self.scale(x), self.degree)

    def forward(self, x):
        return self.out_act(self.body(self.features(x)))


class ChebyshevNet(_PolynomialFeatureNet):
    """Chebyshev neural network (ChNN): t -> [T_0(xi), ..., T_m(xi)] -> MLP, xi in [-1, 1]."""

    feature_fn = staticmethod(chebyshev_features)


class LegendreNet(_PolynomialFeatureNet):
    """Legendre neural network (LeNN): t -> [P_0(xi), ..., P_m(xi)] -> MLP, xi in [-1, 1]."""

    feature_fn = staticmethod(legendre_features)


# ----------------------------------------------------------------------------- Fourier features
class FourierFeatureNet(nn.Module):
    """gamma(x) = [sin(2 pi B x~), cos(2 pi B x~)] followed by an MLP (Tancik et al. 2020).

    ``frequencies`` may be given explicitly (deterministic features) or drawn as
    ``B ~ N(0, sigma^2)`` with ``n_features`` rows.
    """

    def __init__(self, in_dim: int = 1, out_dim: int = 1, n_features: int = 16,
                 sigma: float = 1.0, frequencies: Optional[Iterable[float]] = None,
                 width: int = 32, depth: int = 3, activation: str = "tanh",
                 domain: Domain = None, trainable: bool = False):
        super().__init__()
        self.scale = InputScaling(domain, in_dim)
        if frequencies is not None:
            B = torch.tensor(list(frequencies), dtype=torch.get_default_dtype()).reshape(-1, 1)
            B = B.repeat(1, in_dim)
        else:
            B = sigma * torch.randn(n_features, in_dim)
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
        self.body = _mlp_body(2 * B.shape[0], out_dim, [width] * depth, activation)

    def forward(self, x):
        z = 2.0 * math.pi * self.scale(x) @ self.B.T
        return self.body(torch.cat([torch.sin(z), torch.cos(z)], dim=1))


# ----------------------------------------------------------------------------- SIREN
class SirenNet(nn.Module):
    """Sine-activated network with the initialisation of Sitzmann et al. (2020)."""

    def __init__(self, in_dim: int = 1, out_dim: int = 1, width: int = 32, depth: int = 3,
                 w0_first: float = 30.0, w0: float = 1.0, domain: Domain = None):
        super().__init__()
        self.scale = InputScaling(domain, in_dim)
        self.w0_first, self.w0 = w0_first, w0
        dims = [in_dim] + [width] * depth
        self.linears = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(width, out_dim)
        with torch.no_grad():
            for i, lin in enumerate(self.linears):
                bound = 1.0 / lin.in_features if i == 0 else math.sqrt(6.0 / lin.in_features) / w0
                lin.weight.uniform_(-bound, bound)
                lin.bias.uniform_(-bound, bound)
            nn.init.xavier_normal_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        h = self.scale(x)
        for i, lin in enumerate(self.linears):
            h = torch.sin((self.w0_first if i == 0 else self.w0) * lin(h))
        return self.out(h)


# ----------------------------------------------------------------------------- hard constraints
class ConstrainedModel(nn.Module):
    """u(x) = transform(x, N(x)): enforces initial/boundary conditions exactly.

    Examples
    --------
    IVP ``y(t0) = y0``:              ``transform = lambda t, n: y0 + (t - t0) * n``
    IVP ``y(t0)=y0, y'(t0)=v0``:     ``transform = lambda t, n: y0 + v0*(t-t0) + (t-t0)**2 * n``
    Dirichlet BVP on [a, b]:         ``transform = lambda t, n: g(t) + (t-a)*(b-t)*n``
    """

    def __init__(self, net: nn.Module, transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self.net = net
        self.transform = transform

    def forward(self, x):
        return self.transform(x, self.net(x))


def ivp_transform(t0: float, y0: Sequence[float], dy0: Optional[Sequence[float]] = None):
    """Trial-solution transform for first- or second-order IVPs (vector-valued allowed)."""
    y0_t = torch.as_tensor(y0, dtype=torch.get_default_dtype()).reshape(1, -1)
    if dy0 is None:
        return lambda t, n: y0_t + (t - t0) * n
    dy0_t = torch.as_tensor(dy0, dtype=torch.get_default_dtype()).reshape(1, -1)
    return lambda t, n: y0_t + dy0_t * (t - t0) + (t - t0) ** 2 * n


def dirichlet_transform(a: float, ya: float, b: float, yb: float):
    """u = ya (b-t)/(b-a) + yb (t-a)/(b-a) + (t-a)(b-t) N(t) for y(a)=ya, y(b)=yb."""
    return lambda t, n: ya * (b - t) / (b - a) + yb * (t - a) / (b - a) + (t - a) * (b - t) * n


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
