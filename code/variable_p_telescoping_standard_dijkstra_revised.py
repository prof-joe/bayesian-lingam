#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variable-p exact subset-DP reference versus ordinary telescoping-potential Dijkstra.

The exact DP is used ONLY as an external reference.  It is not used by the
Dijkstra implementation and it does not supply distances, potentials, bounds,
or candidate sets to Dijkstra.

Reference objective
-------------------
For a selected-prefix subset S and a candidate j not in S, let r_j(S) be the
one-dimensional innovation obtained by regressing X_j on X_S.  The exact DP
uses

    c_DP(S,j) = - log g_j(S) / n,

where g_j(S) is the Laplace-approximated Student-t_3 location-scale marginal
likelihood.  The DP enumerates all p*2^(p-1) edges.

Dijkstra reweighting
--------------------
For the residual block at state S, let Lambda(S) be the maximized affine ICA
log likelihood

    Lambda(S) = max_{mu,W} [ n log|det W|
                             + sum_{i,k} log t_3((W(r_i-mu))_k) ].

Lambda(empty terminal)=0, and for a one-dimensional block Lambda is the same
maximized t_3 location-scale log likelihood used below.

The Dijkstra edge is

    w(S,j)
      = [Lambda(S) - ell_j(S) - Lambda(S union {j})] / n
        + [ell_j(S) - log g_j(S)] / n
      = c_DP(S,j) + [Lambda(S)-Lambda(S union {j})] / n.

The first bracket is a maximized likelihood-ratio term.  The second bracket is
the one-dimensional marginal-likelihood correction.  Both must be nonnegative.
If numerical affine-ICA optimization leaves the parent below the feasible
child-plus-univariate submodel, that submodel is embedded as a warm start and
Dijkstra is restarted with the repaired state potential.

The denominator Lambda(S union {j}) computed on an incoming edge is cached as
the state value of the child and is used unchanged as the numerator when that
child is expanded.  No high-dimensional mutual information is re-estimated at
the child.

Along a complete path the state terms telescope:

    sum w(S_{k-1},pi_k)
      = sum c_DP(S_{k-1},pi_k) + [Lambda(root)-Lambda(goal)]/n.

The endpoint term is common to every complete order.  Hence, if all w are
nonnegative, ordinary Dijkstra returns exactly the same optimum as the exact
DP.

Default experiment
------------------
    p=10, n=1000, beta=0.6, gamma=0, repetitions=10.\n\nFor p>12, exact DP verification is skipped automatically and only Dijkstra is run.

Run
---
    python variable_p_telescoping_standard_dijkstra_revised.py --overwrite

After agreement has been established, Dijkstra alone can be run with

    python variable_p_telescoping_standard_dijkstra_revised.py --skip-dp
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln

Array = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Configuration and result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    p: int = 10
    n: int = 1000
    reps: int = 10
    beta: float = 0.6
    df: float = 3.0
    gamma: float = 0.0
    confounded_pairs: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (3, 4),
        (6, 7),
    )
    seed: int = 20260714

    regression_ridge: float = 1.0e-10
    scale_floor: float = 1.0e-8
    hessian_eigen_floor: float = 1.0e-8
    optimizer_maxiter: int = 250

    # Multivariate affine-ICA profile likelihood used only as the Dijkstra
    # state potential.  Two deterministic starts were sufficient in the
    # p=10, gamma=0 verification, but the option is exposed for diagnostics.
    ica_starts: int = 2
    ica_maxiter: int = 500
    ica_ftol: float = 1.0e-10
    ica_gtol: float = 1.0e-6

    # If a likelihood-ratio edge is negative because the parent ICA fit is
    # stuck below a feasible child-plus-univariate submodel, embed that
    # submodel as an exact affine-ICA starting point, improve the parent fit,
    # and restart Dijkstra with the repaired, path-independent state potential.
    ica_warm_start_repair: bool = True
    ica_repair_max_restarts: int = 100
    ica_embedding_tolerance: float = 1.0e-7

    numerical_tolerance: float = 1.0e-8
    score_tolerance: float = 1.0e-8

    output_prefix: str = "variable_p_telescoping_standard_dijkstra_revised"
    skip_dp: bool = False
    max_dp_p: int = 12
    overwrite: bool = False
    fail_on_mismatch: bool = True


@dataclass
class SolverResult:
    order: Tuple[int, ...]
    score: float
    elapsed: float
    expanded_nodes: int
    evaluated_edges: int
    discovered_nodes: int
    max_open_size: int


@dataclass
class ICAFit:
    loglik: float
    mu: Array
    unmixing: Array
    converged: bool
    best_start: int
    gradient_norm: float


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_replication_seed(master_seed: int, replication: int) -> int:
    sequence = np.random.SeedSequence([master_seed, replication])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def simulate_chain(config: Config, rng: np.random.Generator) -> Array:
    errors = rng.standard_t(config.df, size=(config.n, config.p))
    hidden = {
        pair: rng.standard_t(config.df, size=config.n)
        for pair in config.confounded_pairs
    }

    x = np.zeros((config.n, config.p), dtype=float)
    for j in range(config.p):
        value = errors[:, j].copy()
        if j > 0:
            value += config.beta * x[:, j - 1]
        if config.gamma != 0.0:
            for pair, h in hidden.items():
                if j in pair:
                    value += config.gamma * h
        x[:, j] = value

    x -= x.mean(axis=0, keepdims=True)
    return x


def permute_columns(
    x_original: Array,
    rng: np.random.Generator,
) -> Tuple[Array, Tuple[int, ...], Tuple[int, ...]]:
    p = x_original.shape[1]
    permutation = tuple(int(v) for v in rng.permutation(p))
    x = x_original[:, np.asarray(permutation, dtype=int)]

    inverse = np.empty(p, dtype=int)
    inverse[np.asarray(permutation, dtype=int)] = np.arange(p)
    true_order = tuple(int(inverse[j]) for j in range(p))
    return x, permutation, true_order


# ---------------------------------------------------------------------------
# Path-independent residual blocks
# ---------------------------------------------------------------------------

class ResidualCache:
    """Canonical OLS residual block indexed only by the selected subset."""

    def __init__(self, x: Array, ridge: float):
        self.x = np.asarray(x, dtype=float)
        self.n, self.p = self.x.shape
        self.ridge = float(ridge)
        self.cache: Dict[int, Tuple[Tuple[int, ...], Array]] = {}

    def block(self, mask: int) -> Tuple[Tuple[int, ...], Array]:
        if mask in self.cache:
            return self.cache[mask]

        selected = tuple(j for j in range(self.p) if mask & (1 << j))
        remaining = tuple(j for j in range(self.p) if not (mask & (1 << j)))

        if not remaining:
            value = (remaining, np.empty((self.n, 0), dtype=float))
            self.cache[mask] = value
            return value

        y = self.x[:, remaining]
        if not selected:
            residual = y - y.mean(axis=0, keepdims=True)
        else:
            design = np.column_stack([np.ones(self.n), self.x[:, selected]])
            gram = design.T @ design
            gram.flat[:: gram.shape[0] + 1] += self.ridge
            coefficient = np.linalg.solve(gram, design.T @ y)
            residual = y - design @ coefficient

        value = (remaining, np.asarray(residual, dtype=float))
        self.cache[mask] = value
        return value


# ---------------------------------------------------------------------------
# One-dimensional Student-t likelihood and marginal likelihood
# ---------------------------------------------------------------------------

def t_nll_and_gradient(
    parameters: Array,
    x: Array,
    df: float,
) -> Tuple[float, Array]:
    mu = float(parameters[0])
    eta = float(parameters[1])
    sigma = math.exp(eta)

    y = (x - mu) / sigma
    denominator = df + y * y
    c = df + 1.0

    log_constant = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    loglik = np.sum(
        log_constant
        - eta
        - 0.5 * c * np.log1p((y * y) / df)
    )

    gradient = np.array(
        [
            np.sum(-c * y / (sigma * denominator)),
            np.sum(1.0 - c * y * y / denominator),
        ],
        dtype=float,
    )
    return -float(loglik), gradient


def t_observed_hessian(
    mu: float,
    eta: float,
    x: Array,
    df: float,
) -> Array:
    sigma = math.exp(eta)
    y = (x - mu) / sigma
    denominator = df + y * y
    c = df + 1.0

    h_mu_mu = np.sum(
        c * (df - y * y)
        / (sigma * sigma * denominator * denominator)
    )
    h_mu_eta = np.sum(
        2.0 * c * df * y
        / (sigma * denominator * denominator)
    )
    h_eta_eta = np.sum(
        2.0 * c * df * y * y
        / (denominator * denominator)
    )

    return np.array(
        [[h_mu_mu, h_mu_eta], [h_mu_eta, h_eta_eta]],
        dtype=float,
    )


def fit_t_location_scale(
    x: Array,
    config: Config,
) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    mu0 = float(np.median(x))
    mad = float(np.median(np.abs(x - mu0)))
    sd = float(np.std(x, ddof=0))
    sigma0 = max(1.4826 * mad, sd, config.scale_floor)

    start = np.array([mu0, math.log(sigma0)], dtype=float)
    result = minimize(
        fun=lambda par: t_nll_and_gradient(par, x, config.df)[0],
        x0=start,
        jac=lambda par: t_nll_and_gradient(par, x, config.df)[1],
        method="L-BFGS-B",
        bounds=[(None, None), (math.log(config.scale_floor), None)],
        options={
            "maxiter": int(config.optimizer_maxiter),
            "ftol": 1.0e-12,
            "gtol": 1.0e-8,
            "maxls": 40,
        },
    )

    parameters = (
        np.asarray(result.x, dtype=float)
        if result.success and np.all(np.isfinite(result.x))
        else start
    )
    nll, _ = t_nll_and_gradient(parameters, x, config.df)
    return float(parameters[0]), float(parameters[1]), -float(nll)


def log_marginal_t_location_scale(
    x: Array,
    config: Config,
) -> Tuple[float, float]:
    """Return (log marginal likelihood, maximized log likelihood)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    mu, eta, maximum_loglik = fit_t_location_scale(x, config)

    hessian = t_observed_hessian(mu, eta, x, config.df)
    hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = np.maximum(
        np.linalg.eigvalsh(hessian),
        config.hessian_eigen_floor,
    )
    logdet = float(np.sum(np.log(eigenvalues)))

    dimension = 2
    log_g = (
        maximum_loglik
        + 0.5 * dimension * math.log(2.0 * math.pi)
        - 0.5 * logdet
    )
    return float(log_g), float(maximum_loglik)


# ---------------------------------------------------------------------------
# Exact subset DP: reference only
# ---------------------------------------------------------------------------

class ExactDPReference:
    """Unchanged full subset-DAG optimization of -log g_j/n."""

    def __init__(self, x: Array, config: Config):
        self.x = np.asarray(x, dtype=float)
        self.config = config
        self.n, self.p = self.x.shape
        self.full_mask = (1 << self.p) - 1
        self.residuals = ResidualCache(x, config.regression_ridge)
        self.edge_cache: Dict[Tuple[int, int], float] = {}
        self.evaluated_edges = 0

    def edge_cost(self, mask: int, candidate: int) -> float:
        key = (mask, candidate)
        if key in self.edge_cache:
            return self.edge_cache[key]

        remaining, block = self.residuals.block(mask)
        local = remaining.index(candidate)
        log_g, _ = log_marginal_t_location_scale(
            block[:, local],
            self.config,
        )
        value = -log_g / self.n
        self.edge_cache[key] = float(value)
        self.evaluated_edges += 1
        return float(value)

    def solve(self) -> SolverResult:
        started = time.perf_counter()
        number_of_states = 1 << self.p
        distance = np.full(number_of_states, np.inf, dtype=float)
        predecessor_mask = np.full(number_of_states, -1, dtype=np.int64)
        predecessor_variable = np.full(number_of_states, -1, dtype=np.int64)
        distance[0] = 0.0

        for depth in range(self.p):
            for mask in range(number_of_states):
                if mask.bit_count() != depth or not np.isfinite(distance[mask]):
                    continue
                base = float(distance[mask])
                for candidate in range(self.p):
                    if mask & (1 << candidate):
                        continue
                    next_mask = mask | (1 << candidate)
                    proposed = base + self.edge_cost(mask, candidate)
                    old = float(distance[next_mask])
                    if proposed < old - 1.0e-14:
                        distance[next_mask] = proposed
                        predecessor_mask[next_mask] = mask
                        predecessor_variable[next_mask] = candidate
                    elif abs(proposed - old) <= 1.0e-14:
                        old_candidate = int(predecessor_variable[next_mask])
                        if old_candidate < 0 or candidate < old_candidate:
                            predecessor_mask[next_mask] = mask
                            predecessor_variable[next_mask] = candidate

        reversed_order: List[int] = []
        mask = self.full_mask
        while mask:
            parent = int(predecessor_mask[mask])
            candidate = int(predecessor_variable[mask])
            if parent < 0 or candidate < 0:
                raise RuntimeError("Exact DP failed to reconstruct its path.")
            reversed_order.append(candidate)
            mask = parent

        return SolverResult(
            order=tuple(reversed(reversed_order)),
            score=float(distance[self.full_mask]),
            elapsed=float(time.perf_counter() - started),
            expanded_nodes=number_of_states,
            evaluated_edges=int(self.evaluated_edges),
            discovered_nodes=number_of_states,
            max_open_size=0,
        )


# ---------------------------------------------------------------------------
# Affine ICA profile likelihood used as a state potential
# ---------------------------------------------------------------------------

def _ica_nll_and_gradient(
    parameters: Array,
    block: Array,
    df: float,
) -> Tuple[float, Array]:
    n, q = block.shape
    mu = np.asarray(parameters[:q], dtype=float)
    unmixing = np.asarray(parameters[q:], dtype=float).reshape(q, q)

    sign, logabsdet = np.linalg.slogdet(unmixing)
    if sign == 0 or not np.isfinite(logabsdet):
        return 1.0e100, np.zeros_like(parameters)

    centered = block - mu[None, :]
    sources = centered @ unmixing.T

    log_constant = (
        gammaln((df + 1.0) / 2.0)
        - gammaln(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    log_density = (
        log_constant
        - 0.5 * (df + 1.0) * np.log1p((sources * sources) / df)
    )
    loglik = n * logabsdet + float(np.sum(log_density))

    score = -(df + 1.0) * sources / (df + sources * sources)
    try:
        inverse_transpose = np.linalg.inv(unmixing).T
    except np.linalg.LinAlgError:
        return 1.0e100, np.zeros_like(parameters)

    gradient_unmixing = n * inverse_transpose + score.T @ centered
    gradient_mu = -unmixing.T @ np.sum(score, axis=0)
    gradient = np.concatenate(
        [gradient_mu, gradient_unmixing.reshape(-1)]
    )
    return -float(loglik), -np.asarray(gradient, dtype=float)


def _ica_starting_points(
    block: Array,
    config: Config,
    seed: int,
) -> List[Array]:
    n, q = block.shape

    marginal_mu = np.empty(q, dtype=float)
    marginal_inverse_scale = np.empty(q, dtype=float)
    for j in range(q):
        mu, eta, _ = fit_t_location_scale(block[:, j], config)
        marginal_mu[j] = mu
        marginal_inverse_scale[j] = math.exp(-eta)

    starts: List[Tuple[Array, Array]] = [
        (marginal_mu, np.diag(marginal_inverse_scale))
    ]

    covariance = np.cov(block, rowvar=False, bias=True)
    covariance = np.atleast_2d(covariance).astype(float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 1.0e-8)
    whitening = np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    starts.append((np.median(block, axis=0), whitening))

    rng = np.random.default_rng(seed)
    while len(starts) < max(1, config.ica_starts):
        orthogonal, _ = np.linalg.qr(rng.standard_normal((q, q)))
        starts.append((marginal_mu, orthogonal @ np.diag(marginal_inverse_scale)))

    return [
        np.concatenate([mu, unmixing.reshape(-1)])
        for mu, unmixing in starts[: max(1, config.ica_starts)]
    ]


def fit_affine_ica_profile(
    block: Array,
    config: Config,
    seed: int,
    extra_starts: Sequence[Array] = (),
) -> ICAFit:
    block = np.asarray(block, dtype=float)
    if block.ndim == 1:
        block = block[:, None]

    n, q = block.shape
    if q == 0:
        return ICAFit(
            loglik=0.0,
            mu=np.empty(0, dtype=float),
            unmixing=np.empty((0, 0), dtype=float),
            converged=True,
            best_start=0,
            gradient_norm=0.0,
        )

    if q == 1:
        mu, eta, loglik = fit_t_location_scale(block[:, 0], config)
        return ICAFit(
            loglik=float(loglik),
            mu=np.array([mu], dtype=float),
            unmixing=np.array([[math.exp(-eta)]], dtype=float),
            converged=True,
            best_start=0,
            gradient_norm=0.0,
        )

    best_loglik = -math.inf
    best_parameters: Array | None = None
    best_converged = False
    best_start = -1
    best_gradient_norm = math.inf

    starts = _ica_starting_points(block, config, seed)
    expected_size = q + q * q
    for extra in extra_starts:
        extra = np.asarray(extra, dtype=float).reshape(-1)
        if extra.size != expected_size:
            raise ValueError(
                f"Affine-ICA warm start has size {extra.size}; "
                f"expected {expected_size}."
            )
        if np.all(np.isfinite(extra)):
            starts.append(extra.copy())

    for start_index, start in enumerate(starts):
        start_nll, start_gradient = _ica_nll_and_gradient(
            start,
            block,
            config.df,
        )
        start_loglik = -float(start_nll)

        result = minimize(
            fun=lambda par: _ica_nll_and_gradient(
                par, block, config.df
            )[0],
            x0=start,
            jac=lambda par: _ica_nll_and_gradient(
                par, block, config.df
            )[1],
            method="L-BFGS-B",
            options={
                "maxiter": int(config.ica_maxiter),
                "ftol": float(config.ica_ftol),
                "gtol": float(config.ica_gtol),
                "maxls": 50,
            },
        )

        candidate_parameters = (
            np.asarray(result.x, dtype=float)
            if np.all(np.isfinite(result.x))
            else start
        )
        candidate_nll, candidate_gradient = _ica_nll_and_gradient(
            candidate_parameters,
            block,
            config.df,
        )
        candidate_loglik = -float(candidate_nll)
        candidate_gradient_norm = float(
            np.linalg.norm(candidate_gradient, ord=2)
        )

        # A numerical optimizer must never make the retained solution worse
        # than its feasible starting point.
        if candidate_loglik < start_loglik:
            candidate_parameters = start
            candidate_loglik = start_loglik
            candidate_gradient_norm = float(
                np.linalg.norm(start_gradient, ord=2)
            )
            converged = False
        else:
            converged = bool(result.success)

        if candidate_loglik > best_loglik:
            best_loglik = candidate_loglik
            best_parameters = candidate_parameters.copy()
            best_converged = converged
            best_start = start_index
            best_gradient_norm = candidate_gradient_norm

    if best_parameters is None or not np.isfinite(best_loglik):
        raise RuntimeError("No finite affine-ICA profile likelihood was found.")

    mu = best_parameters[:q]
    unmixing = best_parameters[q:].reshape(q, q)
    return ICAFit(
        loglik=float(best_loglik),
        mu=np.asarray(mu, dtype=float),
        unmixing=np.asarray(unmixing, dtype=float),
        converged=bool(best_converged),
        best_start=int(best_start),
        gradient_norm=float(best_gradient_norm),
    )


class LikelihoodPotentialUpdated(RuntimeError):
    """Signal that a state potential was improved and Dijkstra must restart."""


def embedded_child_univariate_start(
    parent_block: Array,
    child_block: Array,
    candidate_local: int,
    candidate_mu: float,
    candidate_eta: float,
    child_fit: ICAFit,
    df: float,
) -> Tuple[Array, Dict[str, float]]:
    """
    Embed the fitted product model

        candidate t-model x child affine-ICA model

    into the parent affine-ICA model.  The transformation from the parent
    block to (candidate, residualized rest) is affine and has determinant
    one.  Hence the embedded parent log likelihood must equal

        ell_candidate + Lambda(child)

    up to numerical roundoff.
    """
    parent_block = np.asarray(parent_block, dtype=float)
    child_block = np.asarray(child_block, dtype=float)
    if parent_block.ndim != 2:
        raise ValueError("parent_block must be a matrix.")

    n, q = parent_block.shape
    if not (0 <= candidate_local < q):
        raise ValueError("candidate_local is out of range.")
    if child_block.shape != (n, q - 1):
        raise ValueError(
            f"child block shape {child_block.shape} is inconsistent with "
            f"parent shape {parent_block.shape}."
        )
    if child_fit.unmixing.shape != (q - 1, q - 1):
        raise ValueError("The child ICA fit has the wrong dimension.")

    rest_indices = [k for k in range(q) if k != candidate_local]
    candidate = parent_block[:, candidate_local]
    rest = parent_block[:, rest_indices]

    # Recover the exact affine residualization map
    # child = rest - intercept - candidate * slope.
    design = np.column_stack([np.ones(n), candidate])
    coefficient, *_ = np.linalg.lstsq(
        design, rest - child_block, rcond=None
    )
    intercept = np.asarray(coefficient[0], dtype=float)
    slope = np.asarray(coefficient[1], dtype=float)

    transform = np.zeros((q, q), dtype=float)
    transform[candidate_local, 0] = 1.0
    for output_index, original_index in enumerate(rest_indices, start=1):
        transform[original_index, output_index] = 1.0
        transform[candidate_local, output_index] = -slope[output_index - 1]

    offset = np.concatenate(
        [np.array([0.0], dtype=float), -intercept]
    )
    transformed = parent_block @ transform + offset[None, :]
    target_block = np.column_stack([candidate, child_block])
    transform_error = float(np.max(np.abs(transformed - target_block)))

    sign, logabsdet = np.linalg.slogdet(transform)
    if sign == 0 or not np.isfinite(logabsdet):
        raise RuntimeError("The residualization transform is singular.")

    mu_transformed = np.concatenate(
        [np.array([candidate_mu], dtype=float), child_fit.mu]
    )
    # mu_parent @ transform + offset = mu_transformed.
    mu_parent = np.linalg.solve(
        transform.T, mu_transformed - offset
    )

    unmixing_transformed = np.zeros((q, q), dtype=float)
    unmixing_transformed[0, 0] = math.exp(-candidate_eta)
    if q > 1:
        unmixing_transformed[1:, 1:] = child_fit.unmixing

    # (parent-mu_parent) W_parent^T
    #   = (transformed-mu_transformed) W_transformed^T.
    unmixing_parent = unmixing_transformed @ transform.T
    start = np.concatenate([mu_parent, unmixing_parent.reshape(-1)])

    start_nll, _ = _ica_nll_and_gradient(start, parent_block, df)
    start_loglik = -float(start_nll)
    target_loglik = float(
        child_fit.loglik
        + _univariate_loglik_from_parameters(
            candidate, candidate_mu, candidate_eta, df
        )
        + n * logabsdet
    )

    return start, {
        "transform_error": transform_error,
        "transform_logabsdet": float(logabsdet),
        "embedded_start_loglik": start_loglik,
        "factorized_target_loglik": target_loglik,
        "embedding_loglik_gap": float(start_loglik - target_loglik),
    }


def _univariate_loglik_from_parameters(
    x: Array,
    mu: float,
    eta: float,
    df: float,
) -> float:
    parameters = np.array([mu, eta], dtype=float)
    nll, _ = t_nll_and_gradient(parameters, np.asarray(x, dtype=float), df)
    return -float(nll)


# ---------------------------------------------------------------------------
# Ordinary Dijkstra.  No DP information is used here.
# ---------------------------------------------------------------------------

class CarriedLikelihoodDijkstra:
    def __init__(self, x: Array, config: Config):
        self.x = np.asarray(x, dtype=float)
        self.config = config
        self.n, self.p = self.x.shape
        self.full_mask = (1 << self.p) - 1
        self.residuals = ResidualCache(x, config.regression_ridge)

        self.state_fit_cache: Dict[int, ICAFit] = {}
        self.state_fit_count_by_dimension: Dict[int, int] = {}
        self.state_fit_time_by_dimension: Dict[int, float] = {}
        # (mu, eta, maximum log likelihood, log marginal likelihood)
        self.innovation_cache: Dict[
            Tuple[int, int], Tuple[float, float, float, float]
        ] = {}

        self.evaluated_edges = 0
        self.repair_restarts = 0
        self.repaired_states = 0
        self.repair_loglik_gain = 0.0
        self.maximum_embedding_transform_error = 0.0
        self.maximum_embedding_loglik_error = 0.0
        self.most_negative_mi_before_repair = 0.0

        self.minimum_mi_term = math.inf
        self.minimum_bayes_correction = math.inf
        self.minimum_edge = math.inf
        self.maximum_decomposition_error = 0.0

    def _record_state_fit(self, q: int, elapsed: float) -> None:
        self.state_fit_count_by_dimension[q] = (
            self.state_fit_count_by_dimension.get(q, 0) + 1
        )
        self.state_fit_time_by_dimension[q] = (
            self.state_fit_time_by_dimension.get(q, 0.0) + elapsed
        )

    def state_fit(self, mask: int) -> ICAFit:
        if mask not in self.state_fit_cache:
            _, block = self.residuals.block(mask)
            q = int(block.shape[1])
            started = time.perf_counter()
            fitted = fit_affine_ica_profile(
                block,
                self.config,
                seed=self.config.seed + 1_000_003 * mask,
            )
            elapsed = time.perf_counter() - started
            self.state_fit_cache[mask] = fitted
            self._record_state_fit(q, elapsed)
        return self.state_fit_cache[mask]

    def innovation_fit(
        self,
        mask: int,
        candidate: int,
    ) -> Tuple[float, float, float, float]:
        key = (mask, candidate)
        if key not in self.innovation_cache:
            remaining, block = self.residuals.block(mask)
            local = remaining.index(candidate)
            innovation = block[:, local]
            mu, eta, maximum_loglik = fit_t_location_scale(
                innovation, self.config
            )

            hessian = t_observed_hessian(
                mu, eta, innovation, self.config.df
            )
            hessian = 0.5 * (hessian + hessian.T)
            eigenvalues = np.maximum(
                np.linalg.eigvalsh(hessian),
                self.config.hessian_eigen_floor,
            )
            logdet = float(np.sum(np.log(eigenvalues)))
            log_g = (
                maximum_loglik
                + math.log(2.0 * math.pi)
                - 0.5 * logdet
            )
            self.innovation_cache[key] = (
                float(mu),
                float(eta),
                float(maximum_loglik),
                float(log_g),
            )
        return self.innovation_cache[key]

    def innovation_terms(
        self,
        mask: int,
        candidate: int,
    ) -> Tuple[float, float]:
        _, _, maximum_loglik, log_g = self.innovation_fit(mask, candidate)
        return maximum_loglik, log_g

    def _repair_parent_fit(
        self,
        mask: int,
        candidate: int,
    ) -> Dict[str, float]:
        if not self.config.ica_warm_start_repair:
            raise RuntimeError(
                "A negative likelihood-ratio term was found and "
                "ica_warm_start_repair is disabled."
            )

        next_mask = mask | (1 << candidate)
        remaining, parent_block = self.residuals.block(mask)
        local = remaining.index(candidate)
        _, child_block = self.residuals.block(next_mask)

        child_fit = self.state_fit(next_mask)
        candidate_mu, candidate_eta, ell_j, _ = self.innovation_fit(
            mask, candidate
        )
        old_parent = self.state_fit(mask)

        embedded_start, embedding = embedded_child_univariate_start(
            parent_block=parent_block,
            child_block=child_block,
            candidate_local=local,
            candidate_mu=candidate_mu,
            candidate_eta=candidate_eta,
            child_fit=child_fit,
            df=self.config.df,
        )

        self.maximum_embedding_transform_error = max(
            self.maximum_embedding_transform_error,
            abs(float(embedding["transform_error"])),
        )
        self.maximum_embedding_loglik_error = max(
            self.maximum_embedding_loglik_error,
            abs(float(embedding["embedding_loglik_gap"])),
        )

        tolerance = float(self.config.ica_embedding_tolerance)
        if embedding["transform_error"] > tolerance:
            raise RuntimeError(
                "The child-to-parent affine embedding is not numerically "
                "consistent.\n"
                f"  mask={mask}, candidate={candidate}\n"
                f"  transform_error={embedding['transform_error']:.12e}"
            )
        if abs(embedding["embedding_loglik_gap"]) > max(
            tolerance, tolerance * self.n
        ):
            raise RuntimeError(
                "The embedded parent likelihood does not equal the "
                "child-plus-univariate likelihood.\n"
                f"  mask={mask}, candidate={candidate}\n"
                f"  embedded={embedding['embedded_start_loglik']:.12f}\n"
                f"  target={embedding['factorized_target_loglik']:.12f}\n"
                f"  gap={embedding['embedding_loglik_gap']:+.12e}"
            )

        q = int(parent_block.shape[1])
        started = time.perf_counter()
        repaired = fit_affine_ica_profile(
            parent_block,
            self.config,
            seed=self.config.seed + 1_000_003 * mask + 97_531,
            extra_starts=(embedded_start,),
        )
        elapsed = time.perf_counter() - started
        self._record_state_fit(q, elapsed)

        # Retain monotonicity even if a numerical optimizer behaves oddly.
        if repaired.loglik < old_parent.loglik:
            repaired = old_parent

        required = ell_j + child_fit.loglik
        if repaired.loglik < required - self.config.numerical_tolerance * self.n:
            raise RuntimeError(
                "Warm-start repair failed to dominate the feasible nested "
                "submodel.\n"
                f"  mask={mask}, candidate={candidate}\n"
                f"  repaired_parent={repaired.loglik:.12f}\n"
                f"  ell_j+child={required:.12f}\n"
                f"  deficit={repaired.loglik-required:+.12e}"
            )

        gain = max(0.0, float(repaired.loglik - old_parent.loglik))
        self.state_fit_cache[mask] = repaired
        if gain > 0.0:
            self.repaired_states += 1
            self.repair_loglik_gain += gain

        return {
            **embedding,
            "old_parent_loglik": float(old_parent.loglik),
            "repaired_parent_loglik": float(repaired.loglik),
            "repair_gain": float(gain),
            "required_nested_loglik": float(required),
        }

    def edge_components(
        self,
        mask: int,
        candidate: int,
    ) -> Tuple[float, float, float]:
        next_mask = mask | (1 << candidate)

        parent_loglik = self.state_fit(mask).loglik
        child_loglik = self.state_fit(next_mask).loglik
        ell_j, log_g_j = self.innovation_terms(mask, candidate)

        mi_term = (parent_loglik - ell_j - child_loglik) / self.n
        bayes_correction = (ell_j - log_g_j) / self.n

        tolerance = self.config.numerical_tolerance
        if mi_term < -tolerance:
            self.most_negative_mi_before_repair = min(
                self.most_negative_mi_before_repair, float(mi_term)
            )
            repair = self._repair_parent_fit(mask, candidate)
            # All edges incident to this state potential must be recomputed
            # consistently.  Discard the current Dijkstra labels and restart
            # with the improved, path-independent potential.
            raise LikelihoodPotentialUpdated(
                "Affine-ICA parent potential improved by an embedded "
                "child-plus-univariate warm start.\n"
                f"  mask={mask}, candidate={candidate}\n"
                f"  old={repair['old_parent_loglik']:.12f}\n"
                f"  new={repair['repaired_parent_loglik']:.12f}\n"
                f"  gain={repair['repair_gain']:.12e}"
            )

        edge = mi_term + bayes_correction
        direct_form = (
            parent_loglik - log_g_j - child_loglik
        ) / self.n
        decomposition_error = edge - direct_form

        self.maximum_decomposition_error = max(
            self.maximum_decomposition_error,
            abs(decomposition_error),
        )
        self.minimum_mi_term = min(self.minimum_mi_term, mi_term)
        self.minimum_bayes_correction = min(
            self.minimum_bayes_correction,
            bayes_correction,
        )
        self.minimum_edge = min(self.minimum_edge, edge)
        self.evaluated_edges += 1

        if bayes_correction < -tolerance:
            raise RuntimeError(
                "The one-dimensional marginal-likelihood correction became "
                "negative.\n"
                f"  mask={mask}, candidate={candidate}\n"
                f"  ell_j={ell_j:.12f}, log_g_j={log_g_j:.12f}\n"
                f"  correction={bayes_correction:+.12e}"
            )
        if edge < -tolerance:
            raise RuntimeError(
                f"A genuine negative Dijkstra edge was obtained: {edge}."
            )

        return (
            max(0.0, float(edge)),
            max(0.0, float(mi_term)),
            max(0.0, float(bayes_correction)),
        )

    def original_path_score(self, order: Sequence[int]) -> float:
        mask = 0
        value = 0.0
        for candidate in order:
            _, log_g_j = self.innovation_terms(mask, int(candidate))
            value += -log_g_j / self.n
            mask |= 1 << int(candidate)
        return float(value)

    def _solve_once(self) -> Tuple[
        Tuple[int, ...], float, int, int, int
    ]:
        """Ordinary Dijkstra with no mathematical tie-breaking rule.

        The heap stores only (distance, insertion_counter, state).  When two
        labels have the same distance within floating-point tolerance, the
        label already stored for that state is retained.  The insertion
        counter exists only so Python never has to compare state payloads.
        """
        distance: Dict[int, float] = {0: 0.0}
        predecessor: Dict[int, Tuple[int, int]] = {}
        settled: set[int] = set()

        insertion = 0
        heap: List[Tuple[float, int, int]] = [(0.0, insertion, 0)]

        expanded = 0
        max_open_size = 1

        while heap:
            current_distance, _, mask = heapq.heappop(heap)

            if mask in settled:
                continue
            if abs(
                current_distance - distance.get(mask, math.inf)
            ) > 1.0e-12:
                continue

            settled.add(mask)
            expanded += 1

            if mask == self.full_mask:
                break

            remaining, _ = self.residuals.block(mask)
            for candidate in remaining:
                edge, _, _ = self.edge_components(mask, candidate)
                next_mask = mask | (1 << candidate)
                proposed_distance = current_distance + edge
                old_distance = distance.get(next_mask, math.inf)

                # Strict improvement only.  Equal-distance alternatives are
                # both optimal and no special ordering rule is required.
                if proposed_distance < old_distance - 1.0e-14:
                    distance[next_mask] = float(proposed_distance)
                    predecessor[next_mask] = (mask, int(candidate))
                    insertion += 1
                    heapq.heappush(
                        heap,
                        (float(proposed_distance), insertion, next_mask),
                    )

            max_open_size = max(max_open_size, len(heap))

        if self.full_mask not in settled:
            raise RuntimeError("Dijkstra did not settle the complete state.")

        reversed_order: List[int] = []
        mask = self.full_mask
        while mask:
            if mask not in predecessor:
                raise RuntimeError("Failed to reconstruct the Dijkstra path.")
            parent, candidate = predecessor[mask]
            reversed_order.append(int(candidate))
            mask = int(parent)

        return (
            tuple(reversed(reversed_order)),
            float(distance[self.full_mask]),
            int(expanded),
            int(len(distance)),
            int(max_open_size),
        )

    def solve(self) -> Tuple[SolverResult, Dict[str, float]]:
        started = time.perf_counter()
        total_expanded = 0
        maximum_open = 1

        while True:
            # These minima describe the final successful Dijkstra pass.
            self.minimum_mi_term = math.inf
            self.minimum_bayes_correction = math.inf
            self.minimum_edge = math.inf
            self.maximum_decomposition_error = 0.0
            try:
                (
                    order,
                    transformed_score,
                    expanded,
                    discovered,
                    max_open_size,
                ) = self._solve_once()
                total_expanded += expanded
                maximum_open = max(maximum_open, max_open_size)
                break
            except LikelihoodPotentialUpdated:
                self.repair_restarts += 1
                if self.repair_restarts > self.config.ica_repair_max_restarts:
                    raise RuntimeError(
                        "Exceeded the maximum number of affine-ICA warm-start "
                        f"repair restarts ({self.config.ica_repair_max_restarts})."
                    )
                # The failed pass is real computational work and is included
                # in elapsed time and edge evaluations.  Its expanded count is
                # not available at the exception point, so the final expanded
                # count remains the successful-pass count; repair_restarts is
                # reported separately.
                continue

        original_score = self.original_path_score(order)
        root_potential = self.state_fit(0).loglik
        terminal_potential = self.state_fit(self.full_mask).loglik
        expected_transformed = (
            original_score
            + (root_potential - terminal_potential) / self.n
        )
        telescoping_error = transformed_score - expected_transformed

        if abs(telescoping_error) > self.config.score_tolerance:
            raise RuntimeError(
                "The final repaired Dijkstra path failed the telescoping "
                "identity.\n"
                f"  error={telescoping_error:+.12e}"
            )

        result = SolverResult(
            order=tuple(int(v) for v in order),
            score=float(original_score),
            elapsed=float(time.perf_counter() - started),
            expanded_nodes=int(total_expanded),
            evaluated_edges=int(self.evaluated_edges),
            discovered_nodes=int(discovered),
            max_open_size=int(maximum_open),
        )
        diagnostics = {
            "transformed_score": float(transformed_score),
            "root_potential": float(root_potential),
            "terminal_potential": float(terminal_potential),
            "telescoping_error": float(telescoping_error),
            "minimum_mi_term": float(self.minimum_mi_term),
            "minimum_bayes_correction": float(
                self.minimum_bayes_correction
            ),
            "minimum_edge": float(self.minimum_edge),
            "maximum_decomposition_error": float(
                self.maximum_decomposition_error
            ),
            "warm_start_repair_restarts": int(self.repair_restarts),
            "warm_start_repaired_states": int(self.repaired_states),
            "warm_start_total_loglik_gain": float(
                self.repair_loglik_gain
            ),
            "most_negative_mi_before_repair": float(
                self.most_negative_mi_before_repair
            ),
            "maximum_embedding_transform_error": float(
                self.maximum_embedding_transform_error
            ),
            "maximum_embedding_loglik_error": float(
                self.maximum_embedding_loglik_error
            ),
            "fitted_state_potentials": float(len(self.state_fit_cache)),
            "state_fit_seconds": float(
                sum(self.state_fit_time_by_dimension.values())
            ),
            "state_fit_count_by_dimension": dict(
                sorted(
                    self.state_fit_count_by_dimension.items(),
                    reverse=True,
                )
            ),
            "state_fit_time_by_dimension": dict(
                sorted(
                    self.state_fit_time_by_dimension.items(),
                    reverse=True,
                )
            ),
        }
        return result, diagnostics


# ---------------------------------------------------------------------------
# Comparison experiment and output
# ---------------------------------------------------------------------------

def pair_error(
    estimated_order: Sequence[int],
    true_order: Sequence[int],
) -> int:
    position = {
        int(variable): index
        for index, variable in enumerate(estimated_order)
    }
    errors = 0
    for a in range(len(true_order)):
        for b in range(a + 1, len(true_order)):
            if position[int(true_order[a])] > position[int(true_order[b])]:
                errors += 1
    return int(errors)


def write_csv(path: Path, rows: List[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(config: Config) -> None:
    if config.p < 2:
        raise ValueError("p must be at least 2.")
    if config.n <= config.p:
        raise ValueError("n must be larger than p.")

    effective_skip_dp = bool(config.skip_dp or config.p > config.max_dp_p)

    prefix = Path(config.output_prefix)
    raw_path = Path(str(prefix) + "_raw.csv")
    config_path = Path(str(prefix) + "_config.json")

    if config.overwrite:
        for path in (raw_path, config_path):
            if path.exists():
                path.unlink()

    config_path.write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows: List[Dict[str, object]] = []
    total_pairs = config.p * (config.p - 1) // 2

    title = f"p={config.p}: exact DP reference vs carried-likelihood ordinary Dijkstra"
    print(title)
    print("=" * len(title))
    print(config)
    print(
        "DP is an independent reference only.  Dijkstra uses no DP "
        "distance, bound, potential, or candidate information."
    )
    print(
        f"Full subset graph: {1 << config.p} nodes, "
        f"{config.p * (1 << (config.p - 1))} edges."
    )
    if config.p > config.max_dp_p and not config.skip_dp:
        print(
            f"Exact DP is skipped automatically because p={config.p} > "
            f"max_dp_p={config.max_dp_p}."
        )
    print()

    for rep in range(1, config.reps + 1):
        replication_seed = make_replication_seed(config.seed, rep)
        rng = np.random.default_rng(replication_seed)
        x_original = simulate_chain(config, rng)
        x, permutation, true_order = permute_columns(x_original, rng)

        print(f"rep={rep:2d}, seed={replication_seed}")
        print(f"  permutation : {permutation}")
        print(f"  true order  : {true_order}")

        dijkstra_solver = CarriedLikelihoodDijkstra(x, config)
        try:
            dijkstra, diagnostic = dijkstra_solver.solve()
        except Exception:
            traceback.print_exc()
            raise

        print(
            f"  Dijkstra    : order={dijkstra.order}, "
            f"score={dijkstra.score:.12f}, "
            f"time={dijkstra.elapsed:.3f}s"
        )
        print(
            f"                expanded={dijkstra.expanded_nodes}/"
            f"{1 << config.p}, "
            f"edges={dijkstra.evaluated_edges}/"
            f"{config.p * (1 << (config.p - 1))}, "
            f"discovered={dijkstra.discovered_nodes}, "
            f"max_OPEN={dijkstra.max_open_size}"
        )
        print(
            f"                min_MI={diagnostic['minimum_mi_term']:+.3e}, "
            f"min_Bayes1D={diagnostic['minimum_bayes_correction']:+.3e}, "
            f"min_edge={diagnostic['minimum_edge']:+.3e}, "
            f"tel_err={diagnostic['telescoping_error']:+.3e}"
        )
        print(
            f"                state fits={int(diagnostic['fitted_state_potentials'])}, "
            f"state-fit time={diagnostic['state_fit_seconds']:.3f}s"
        )
        print(
            f"                fits by q={diagnostic['state_fit_count_by_dimension']}"
        )

        if effective_skip_dp:
            dp = None
            order_match = ""
            score_match = ""
            score_difference = ""
        else:
            dp_solver = ExactDPReference(x, config)
            dp = dp_solver.solve()
            order_match = int(dp.order == dijkstra.order)
            score_difference = float(dijkstra.score - dp.score)
            score_match = int(
                abs(score_difference) <= config.score_tolerance
            )

            print(
                f"  exact DP    : order={dp.order}, "
                f"score={dp.score:.12f}, time={dp.elapsed:.3f}s"
            )
            print(
                f"  agreement   : order={order_match}, "
                f"score={score_match}, "
                f"difference={score_difference:+.3e}"
            )

            if config.fail_on_mismatch and score_match != 1:
                raise RuntimeError(
                    "Dijkstra and exact DP have different minimum objective "
                    "values.  The experiment stops before any larger "
                    "simulation is attempted."
                )
            if order_match != 1 and score_match == 1:
                print(
                    "  note        : the orders differ only because multiple "
                    "minimum-score paths are available; both are optimal."
                )

        dijkstra_pair_error = pair_error(dijkstra.order, true_order)
        print(
            f"  accuracy    : exact={int(dijkstra.order == true_order)}, "
            f"pair_error={dijkstra_pair_error}/{total_pairs}"
        )
        print()

        rows.append(
            {
                "rep": rep,
                "seed": replication_seed,
                "permutation": json.dumps(permutation),
                "true_order": json.dumps(true_order),
                "dijkstra_order": json.dumps(dijkstra.order),
                "dijkstra_original_score": dijkstra.score,
                "dijkstra_transformed_score": diagnostic[
                    "transformed_score"
                ],
                "dijkstra_time": dijkstra.elapsed,
                "expanded_nodes": dijkstra.expanded_nodes,
                "evaluated_edges": dijkstra.evaluated_edges,
                "discovered_nodes": dijkstra.discovered_nodes,
                "max_open_size": dijkstra.max_open_size,
                "minimum_mi_term": diagnostic["minimum_mi_term"],
                "minimum_bayes_correction": diagnostic[
                    "minimum_bayes_correction"
                ],
                "minimum_edge": diagnostic["minimum_edge"],
                "telescoping_error": diagnostic["telescoping_error"],
                "fitted_state_potentials": int(diagnostic["fitted_state_potentials"]),
                "state_fit_seconds": diagnostic["state_fit_seconds"],
                "state_fit_count_by_dimension": json.dumps(diagnostic["state_fit_count_by_dimension"]),
                "state_fit_time_by_dimension": json.dumps(diagnostic["state_fit_time_by_dimension"]),
                "dijkstra_exact": int(dijkstra.order == true_order),
                "dijkstra_pair_error": dijkstra_pair_error,
                "dp_order": "" if dp is None else json.dumps(dp.order),
                "dp_score": "" if dp is None else dp.score,
                "dp_time": "" if dp is None else dp.elapsed,
                "order_match": order_match,
                "score_match": score_match,
                "score_difference": score_difference,
            }
        )
        write_csv(raw_path, rows)

    if not effective_skip_dp:
        all_order_match = all(int(row["order_match"]) == 1 for row in rows)
        all_score_match = all(int(row["score_match"]) == 1 for row in rows)
        print("FINAL VALIDATION")
        print(f"  all order matches = {int(all_order_match)}")
        print(f"  all score matches = {int(all_score_match)}")
    else:
        print("FINAL SUMMARY (DP skipped)")

    print(
        f"  mean expanded = "
        f"{np.mean([row['expanded_nodes'] for row in rows]):.2f}"
    )
    print(
        f"  mean edges    = "
        f"{np.mean([row['evaluated_edges'] for row in rows]):.2f}"
    )
    print(
        f"  mean Dijkstra time = "
        f"{np.mean([row['dijkstra_time'] for row in rows]):.3f}s"
    )
    if not effective_skip_dp:
        print(
            f"  mean DP time       = "
            f"{np.mean([row['dp_time'] for row in rows]):.3f}s"
        )
    print(f"  raw CSV: {raw_path}")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ordinary carried-likelihood Dijkstra against an "
            "independent exact subset-DP reference."
        )
    )
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--ica-starts", type=int, default=2)
    parser.add_argument("--ica-maxiter", type=int, default=500)
    parser.add_argument("--numerical-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--score-tolerance", type=float, default=1.0e-8)
    parser.add_argument(
        "--output-prefix",
        default=None,
    )
    parser.add_argument(
        "--max-dp-p",
        type=int,
        default=12,
        help="Run exact subset-DP verification only when p <= this value.",
    )
    parser.add_argument("--skip-dp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-fail-on-mismatch",
        action="store_true",
        help="Report a mismatch instead of stopping immediately.",
    )
    args = parser.parse_args()

    if args.p < 2:
        parser.error("--p must be at least 2.")
    if args.n <= args.p:
        parser.error("--n must be larger than --p.")
    if args.max_dp_p < 2:
        parser.error("--max-dp-p must be at least 2.")
    if args.reps <= 0:
        parser.error("--reps must be positive.")
    if args.ica_starts <= 0:
        parser.error("--ica-starts must be positive.")

    output_prefix = (
        str(args.output_prefix)
        if args.output_prefix is not None
        else f"p{int(args.p)}_dp_reference_vs_carried_ica_dijkstra"
    )

    default_pairs = ((0, 1), (3, 4), (6, 7))
    valid_pairs = tuple(
        (a, b) for (a, b) in default_pairs
        if a < int(args.p) and b < int(args.p)
    )

    return Config(
        p=int(args.p),
        n=int(args.n),
        reps=int(args.reps),
        beta=float(args.beta),
        gamma=float(args.gamma),
        confounded_pairs=valid_pairs,
        seed=int(args.seed),
        ica_starts=int(args.ica_starts),
        ica_maxiter=int(args.ica_maxiter),
        numerical_tolerance=float(args.numerical_tolerance),
        score_tolerance=float(args.score_tolerance),
        output_prefix=output_prefix,
        skip_dp=bool(args.skip_dp),
        max_dp_p=int(args.max_dp_p),
        overwrite=bool(args.overwrite),
        fail_on_mismatch=not bool(args.no_fail_on_mismatch),
    )


if __name__ == "__main__":
    run(parse_args())
