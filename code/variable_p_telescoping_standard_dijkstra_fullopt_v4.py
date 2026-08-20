#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop-in full-optimizer compiled wrapper for the verified Dijkstra core.

The statistical objective, residual blocks, state potentials, warm-start
repair, and ordinary Dijkstra search are imported unchanged from
``variable_p_telescoping_standard_dijkstra_revised.py``.

Compared with the v3 accelerated wrapper, this module additionally moves the
complete L-BFGS-B reverse-communication loop and multistart selection loop into
compiled Cython code.  The compiled backend calls SciPy's low-level compiled
``setulb`` routine directly, avoiding ``scipy.optimize.minimize`` and its
Python callback machinery.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence

import importlib.machinery
import numpy as np

import variable_p_telescoping_standard_dijkstra_revised as _core
try:
    import _full_optimizer_backend_v4 as _fast
except ImportError as exc:
    raise ImportError(
        "Compiled full-optimizer backend not found. Build it with:\n"
        "  python build_full_optimizer_backend_v4.py"
    ) from exc

_backend_file = str(Path(_fast.__file__).resolve())
if not any(_backend_file.endswith(s) for s in importlib.machinery.EXTENSION_SUFFIXES):
    raise ImportError(
        "The full optimizer backend was loaded from a non-compiled file: "
        f"{_backend_file}"
    )

# Immutable references used by validation utilities.
_REF_T = _core.t_nll_and_gradient
_REF_H = _core.t_observed_hessian
_REF_ICA = _core._ica_nll_and_gradient
_REF_FIT_T = _core.fit_t_location_scale
_REF_FIT_ICA = _core.fit_affine_ica_profile


def _as_c1(x):
    return np.asarray(x, dtype=np.float64, order="C").reshape(-1)


def _as_c2(x):
    return np.asarray(x, dtype=np.float64, order="C")


def t_nll_and_gradient(parameters, x, df):
    return _fast.t_nll_gradient(_as_c1(parameters), _as_c1(x), float(df))


def t_observed_hessian(mu, eta, x, df):
    return _fast.t_observed_hessian(float(mu), float(eta), _as_c1(x), float(df))


def _ica_nll_and_gradient(parameters, block, df):
    return _fast.ica_nll_gradient(_as_c1(parameters), _as_c2(block), float(df))


def fit_t_location_scale(x, config):
    """Verified estimator with the entire L-BFGS-B loop compiled."""
    x = _as_c1(x)
    mu0 = float(np.median(x))
    mad = float(np.median(np.abs(x - mu0)))
    sd = float(np.std(x, ddof=0))
    sigma0 = max(1.4826 * mad, sd, config.scale_floor)
    start = np.array([mu0, math.log(sigma0)], dtype=np.float64)

    parameters, nll, _grad_norm, success, _nit, _nfev = _fast.optimize_t(
        x,
        float(config.df),
        start,
        math.log(config.scale_floor),
        int(config.optimizer_maxiter),
        1.0e-12,
        1.0e-8,
        40,
        10,
    )
    # Match the verified Python reference exactly: if L-BFGS-B does not report
    # success, use the robust starting point rather than a partial iterate.
    if not bool(success) or not np.all(np.isfinite(parameters)):
        parameters = start
        nll, _ = t_nll_and_gradient(parameters, x, config.df)
    return float(parameters[0]), float(parameters[1]), -float(nll)


def fit_affine_ica_profile(
    block,
    config,
    seed: int,
    extra_starts: Sequence[np.ndarray] = (),
):
    """Same multistart affine-ICA fit with optimizer and selection compiled."""
    block = _as_c2(block)
    if block.ndim == 1:
        block = block[:, None]
    _n, q = block.shape

    if q == 0:
        return _core.ICAFit(
            loglik=0.0,
            mu=np.empty(0, dtype=float),
            unmixing=np.empty((0, 0), dtype=float),
            converged=True,
            best_start=0,
            gradient_norm=0.0,
        )
    if q == 1:
        mu, eta, loglik = fit_t_location_scale(block[:, 0], config)
        return _core.ICAFit(
            loglik=float(loglik),
            mu=np.array([mu], dtype=float),
            unmixing=np.array([[math.exp(-eta)]], dtype=float),
            converged=True,
            best_start=0,
            gradient_norm=0.0,
        )

    starts: List[np.ndarray] = _core._ica_starting_points(block, config, seed)
    expected_size = q + q * q
    for extra in extra_starts:
        extra_array = _as_c1(extra)
        if extra_array.size != expected_size:
            raise ValueError(
                f"Affine-ICA warm start has size {extra_array.size}; "
                f"expected {expected_size}."
            )
        if np.all(np.isfinite(extra_array)):
            starts.append(extra_array.copy())

    result = _fast.optimize_ica_multistart(
        block,
        float(config.df),
        np.ascontiguousarray(np.vstack(starts), dtype=np.float64),
        int(config.ica_maxiter),
        float(config.ica_ftol),
        float(config.ica_gtol),
        50,
        10,
    )
    parameters = np.asarray(result["x"], dtype=float)
    if not np.all(np.isfinite(parameters)) or not np.isfinite(result["nll"]):
        raise RuntimeError("No finite affine-ICA profile likelihood was found.")

    return _core.ICAFit(
        loglik=-float(result["nll"]),
        mu=np.asarray(parameters[:q], dtype=float),
        unmixing=np.asarray(parameters[q:].reshape(q, q), dtype=float),
        converged=bool(result["success"]),
        best_start=int(result["best_start"]),
        gradient_norm=float(result["gradient_norm"]),
    )


# Patch only numerical fitting functions.  The verified search implementation
# is not copied or changed.
_core.t_nll_and_gradient = t_nll_and_gradient
_core.t_observed_hessian = t_observed_hessian
_core.fit_t_location_scale = fit_t_location_scale
_core._ica_nll_and_gradient = _ica_nll_and_gradient
_core.fit_affine_ica_profile = fit_affine_ica_profile

# Public drop-in API.
Config = _core.Config
SolverResult = _core.SolverResult
ICAFit = _core.ICAFit
ResidualCache = _core.ResidualCache
ExactDPReference = _core.ExactDPReference
CarriedLikelihoodDijkstra = _core.CarriedLikelihoodDijkstra
embedded_child_univariate_start = _core.embedded_child_univariate_start
pair_error = _core.pair_error
make_replication_seed = _core.make_replication_seed
simulate_chain = _core.simulate_chain
permute_columns = _core.permute_columns
run = _core.run
parse_args = _core.parse_args


def backend_info():
    return {
        **_fast.backend_info(),
        "backend_file": _backend_file,
        "core_file": str(Path(_core.__file__).resolve()),
    }


if __name__ == "__main__":
    run(parse_args())
