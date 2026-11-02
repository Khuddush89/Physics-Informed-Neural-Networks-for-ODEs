"""pinnode: companion library for the book

    Physics-Informed Neural Networks for Ordinary, Integral and Integro-Differential
    Equations: Theory, Algorithms and Python Implementations.

The package collects the pieces that every example shares (architectures, automatic
differentiation helpers, quadrature, two-stage training, reference solvers, metrics and the
book's figure style) so that each chapter script contains only the mathematics of its
example.
"""
from .autodiff import derivative, derivatives, gradient
from .metrics import linf, rel_l2, rmse, summary
from .networks import (ChebyshevNet, ConstrainedModel, FourierFeatureNet, LegendreNet, MLP,
                       SirenNet, count_parameters, dirichlet_transform, ivp_transform,
                       make_activation)
from .quadrature import (chebyshev_gauss, chebyshev_lobatto, composite_gauss_legendre,
                         gauss_legendre, legendre_gauss_lobatto, to_tensor, trapezoid_weights,
                         uniform, volterra_gauss)
from .training import History, train
from .utils import Example, Timer, configure_torch, set_seed

__version__ = "2.0.0"

__all__ = [
    "derivative", "derivatives", "gradient", "linf", "rel_l2", "rmse", "summary",
    "ChebyshevNet", "ConstrainedModel", "FourierFeatureNet", "LegendreNet", "MLP", "SirenNet",
    "count_parameters", "dirichlet_transform", "ivp_transform", "make_activation",
    "chebyshev_gauss", "chebyshev_lobatto", "composite_gauss_legendre", "gauss_legendre",
    "legendre_gauss_lobatto", "to_tensor", "trapezoid_weights", "uniform", "volterra_gauss",
    "History", "train", "Example", "Timer", "configure_torch", "set_seed",
]
