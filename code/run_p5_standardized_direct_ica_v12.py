#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V12 p=5 comparison: standardized-shape proposal and LiNGAM baselines.

Proposal
--------
Uses ``variable_p_telescoping_standard_dijkstra_fullopt_v4.py``.  That module
imports the verified ordinary-Dijkstra implementation unchanged and replaces
the likelihood kernels, complete L-BFGS-B iteration loop, and multistart
selection by a compiled Cython/C + BLAS/LAPACK backend.

Parallel execution
------------------
The v5 runner can split work by replication, by simulation condition, or by
individual method.  In ``--parallel-unit auto`` mode, ``--jobs 1`` preserves
the original replication-level execution while ``--jobs > 1`` splits each
replication into the 3 sample sizes x 5 conditions.  Therefore even
``--reps 1`` can use several CPU cores.  Every worker regenerates the same
paired errors, hidden variables, permutation, and nested samples from the
replication seed, so statistical pairing is preserved exactly.

For publishable single-process method timings, use ``--jobs 1``.  With
``--jobs > 1``, per-method elapsed columns include CPU contention; use the
reported total wall-clock time to assess experiment completion speed.

Scaled sample-size design
-------------------------
  p=5 : n=50,100,200
  p=10: n=100,200,400
  p=15: n=150,300,600

Conditions for every n
----------------------
  gamma=0, no confounding
  gamma=0.4, adjacent
  gamma=0.4, nonadjacent
  gamma=0.8, adjacent
  gamma=0.8, nonadjacent

Specified pair patterns
-----------------------
p=10 adjacent:
  (0,1),(2,3),(4,5),(6,7),(8,9)
p=15 adjacent:
  above plus (10,11),(12,13)
p=10 nonadjacent:
  (0,3),(2,5),(4,7),(6,9)
p=15 nonadjacent:
  above plus (8,11),(10,13)

For p=5 (optional diagnostic), the corresponding in-range subsets are used:
adjacent=(0,1),(2,3), nonadjacent=(0,3).
"""
from __future__ import annotations

# Set these before importing NumPy/SciPy in this process and its workers.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import heapq
import json
import math
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


N_BY_P = {
    5: (50, 100, 200),
    10: (100, 200, 400),
    15: (150, 300, 600),
}
CONDITIONS = (
    ("none", 0.0),
    ("adjacent", 0.4),
    ("nonadjacent", 0.4),
    ("adjacent", 0.8),
    ("nonadjacent", 0.8),
)


@dataclass(frozen=True)
class ExperimentConfig:
    p_values: Tuple[int, ...] = (5,)
    reps: int = 100
    beta_values: Tuple[float, ...] = (0.4,)
    df: float = 5.0
    seed: int = 20260718
    jobs: int = 1
    parallel_unit: str = "auto"
    methods: Tuple[str, ...] = (
        "proposed_standardized_shape",
        "direct_lingam",
        "ica_lingam",
    )

    normalize_pair_loading: bool = True
    regression_ridge: float = 1.0e-10
    scale_floor: float = 1.0e-8
    hessian_eigen_floor: float = 1.0e-8
    optimizer_maxiter: int = 250
    ica_starts: int = 2
    ica_maxiter: int = 500
    ica_ftol: float = 1.0e-10
    ica_gtol: float = 1.0e-6
    ica_warm_start_repair: bool = True
    ica_repair_max_restarts: int = 100
    ica_embedding_tolerance: float = 1.0e-7
    numerical_tolerance: float = 1.0e-8
    score_tolerance: float = 1.0e-8

    k_ksg: int = 5
    ksg_noise_level: float = 1.0e-10
    ksg_normalize: bool = False
    ksg_backend: str = "ckdtree"
    ksg_leafsize: int = 0
    ksg_tree_workers: int = 1
    direct_measure: str = "pwling"
    lingam_ica_max_iter: int = 1000

    output_prefix: str = "scaled_grid_hybrid_v10_parallel"
    overwrite: bool = False
    resume: bool = True


@dataclass
class MethodResult:
    order: Tuple[int, ...]
    score: float | None
    elapsed: float
    expanded: int | None = None
    evaluated: int | None = None
    discovered: int | None = None
    max_open: int | None = None
    diagnostics: Mapping[str, object] | None = None
    status: str = "ok"
    error: str = ""


def adjacent_pairs(p: int) -> Tuple[Tuple[int, int], ...]:
    values = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    if p >= 15:
        values.extend([(10, 11), (12, 13)])
    return tuple((a, b) for a, b in values if a < p and b < p)


def nonadjacent_pairs(p: int) -> Tuple[Tuple[int, int], ...]:
    values = [(0, 3), (2, 5), (4, 7), (6, 9)]
    if p >= 15:
        values.extend([(8, 11), (10, 13)])
    return tuple((a, b) for a, b in values if a < p and b < p)


def pairs_for(p: int, pattern: str) -> Tuple[Tuple[int, int], ...]:
    if pattern == "none":
        return tuple()
    if pattern == "adjacent":
        return adjacent_pairs(p)
    if pattern == "nonadjacent":
        return nonadjacent_pairs(p)
    raise ValueError(f"Unknown pattern: {pattern}")


def union_pairs(p: int) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted(set(adjacent_pairs(p)) | set(nonadjacent_pairs(p))))


def replication_seed(master: int, p: int, beta_index: int, rep: int) -> int:
    seq = np.random.SeedSequence([master, p, beta_index, rep])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def generate_base(
    max_n: int,
    p: int,
    df: float,
    pair_union: Sequence[Tuple[int, int]],
    seed: int,
):
    rng = np.random.default_rng(seed)
    errors = rng.standard_t(df=df, size=(max_n, p))
    hidden = {
        tuple(sorted(pair)): rng.standard_t(df=df, size=max_n)
        for pair in pair_union
    }
    permutation = tuple(int(v) for v in rng.permutation(p))
    return errors, hidden, permutation


def simulate(
    n: int,
    p: int,
    beta: float,
    gamma: float,
    pairs: Sequence[Tuple[int, int]],
    errors: np.ndarray,
    hidden: Mapping[Tuple[int, int], np.ndarray],
    normalize_pair_loading: bool,
) -> np.ndarray:
    eps = np.asarray(errors[:n, :p], dtype=float)
    degree = np.zeros(p, dtype=int)
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1

    hidden_sum = np.zeros((n, p), dtype=float)
    if gamma != 0.0:
        for a, b in pairs:
            h = np.asarray(hidden[tuple(sorted((a, b)))][:n], dtype=float)
            load_a = gamma / math.sqrt(degree[a]) if normalize_pair_loading else gamma
            load_b = gamma / math.sqrt(degree[b]) if normalize_pair_loading else gamma
            hidden_sum[:, a] += load_a * h
            hidden_sum[:, b] += load_b * h

    x = np.zeros((n, p), dtype=float)
    for j in range(p):
        value = eps[:, j] + hidden_sum[:, j]
        if j > 0:
            value = value + beta * x[:, j - 1]
        x[:, j] = value
    x -= x.mean(axis=0, keepdims=True)
    return x


def apply_permutation(x_original: np.ndarray, permutation: Sequence[int]):
    p = x_original.shape[1]
    perm = np.asarray(permutation, dtype=int)
    x = np.asarray(x_original[:, perm], dtype=float)
    inverse = np.empty(p, dtype=int)
    inverse[perm] = np.arange(p)
    true_order = tuple(int(inverse[j]) for j in range(p))
    return x, true_order


def pair_error(order: Sequence[int], truth: Sequence[int]) -> int:
    position = {int(v): i for i, v in enumerate(order)}
    errors = 0
    for a in range(len(truth)):
        for b in range(a + 1, len(truth)):
            if position[int(truth[a])] > position[int(truth[b])]:
                errors += 1
    return int(errors)


def residualize_rest_on_candidate(block: np.ndarray, local: int, ridge: float):
    candidate = np.asarray(block[:, local], dtype=float)
    keep = [j for j in range(block.shape[1]) if j != local]
    if not keep:
        return candidate, np.empty((len(candidate), 0), dtype=float)
    y = np.asarray(block[:, keep], dtype=float)
    design = np.column_stack([np.ones(len(candidate)), candidate])
    gram = design.T @ design
    gram.flat[:: gram.shape[0] + 1] += ridge
    coefficient = np.linalg.solve(gram, design.T @ y)
    return candidate, y - design @ coefficient


def _standardize_columns(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    sd = np.maximum(z.std(axis=0, ddof=0, keepdims=True), 1.0e-12)
    return (z - z.mean(axis=0, keepdims=True)) / sd


def _prepare_ksg_data(
    x: np.ndarray,
    y: np.ndarray,
    noise_level: float,
    normalize: bool,
    seed: int,
):
    x = np.ascontiguousarray(np.asarray(x, dtype=float).reshape(-1, 1))
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    y = np.ascontiguousarray(y)
    if normalize:
        x = np.ascontiguousarray(_standardize_columns(x))
        y = np.ascontiguousarray(_standardize_columns(y))
    if noise_level > 0.0:
        rng = np.random.default_rng(seed)
        x = np.ascontiguousarray(
            x + noise_level * rng.standard_normal(x.shape)
        )
        y = np.ascontiguousarray(
            y + noise_level * rng.standard_normal(y.shape)
        )
    return x, y


def _query_ball_count(tree, values, radii, workers: int) -> np.ndarray:
    kwargs = {
        "p": np.inf,
        "return_length": True,
    }
    if workers is not None:
        kwargs["workers"] = int(workers)
    try:
        counts = tree.query_ball_point(values, radii, **kwargs)
        return np.asarray(counts, dtype=np.int64)
    except TypeError:
        # Compatibility with older SciPy versions without workers and/or
        # return_length.
        kwargs.pop("workers", None)
        try:
            counts = tree.query_ball_point(values, radii, **kwargs)
            return np.asarray(counts, dtype=np.int64)
        except TypeError:
            return np.fromiter(
                (
                    len(tree.query_ball_point(v, r, p=np.inf))
                    for v, r in zip(values, radii)
                ),
                dtype=np.int64,
                count=len(values),
            )


def ksg_mutual_information_type1_ckdtree(
    x: np.ndarray,
    y: np.ndarray,
    k: int,
    noise_level: float,
    normalize: bool,
    seed: int,
    leafsize: int = 0,
    tree_workers: int = 1,
) -> float:
    """Vectorized KSG-I mutual information using SciPy ``cKDTree``.

    The formula and maximum norm match infomeasure's KSG-I estimator.  A single
    deterministic jitter is applied before tree construction.  ``leafsize=0``
    selects a dimension-adaptive value tuned for the small-n/high-dimensional
    edge problems in this experiment.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x, y = _prepare_ksg_data(x, y, noise_level, normalize, seed)
    n = x.shape[0]
    if y.shape[1] == 0:
        return 0.0
    if not (1 <= int(k) < n):
        raise ValueError(f"KSG requires 1 <= k < n; got k={k}, n={n}.")
    if tree_workers == 0 or tree_workers < -1:
        raise ValueError("ksg_tree_workers must be -1 or a positive integer.")

    joint = np.ascontiguousarray(np.column_stack((x, y)))
    if leafsize > 0:
        joint_leaf = max(1, min(n, int(leafsize)))
        marginal_leaf = joint_leaf
    else:
        # Rebuilding three trees for every Dijkstra edge is expensive.  Larger
        # leaves reduce construction and traversal overhead for n <= 600 and
        # joint dimensions up to 15 without changing the exact neighbor result.
        joint_leaf = max(1, min(n, max(24, min(128, 8 * joint.shape[1]))))
        marginal_leaf = max(1, min(n, max(16, min(64, n // 16))))

    joint_tree = cKDTree(
        joint,
        leafsize=joint_leaf,
        compact_nodes=False,
        balanced_tree=True,
    )
    try:
        distances, _ = joint_tree.query(
            joint,
            k=int(k) + 1,
            p=np.inf,
            workers=int(tree_workers),
        )
    except TypeError:
        distances, _ = joint_tree.query(
            joint,
            k=int(k) + 1,
            p=np.inf,
        )
    epsilon = np.asarray(distances[:, int(k)], dtype=float)
    strict_radius = np.nextafter(epsilon, -np.inf)

    x_tree = cKDTree(
        x,
        leafsize=marginal_leaf,
        compact_nodes=False,
        balanced_tree=True,
    )
    y_tree = cKDTree(
        y,
        leafsize=joint_leaf,
        compact_nodes=False,
        balanced_tree=True,
    )
    self_count = (epsilon > 0.0).astype(np.int64)
    nx = _query_ball_count(x_tree, x, strict_radius, tree_workers) - self_count
    ny = _query_ball_count(y_tree, y, strict_radius, tree_workers) - self_count

    estimate = (
        digamma(int(k))
        + digamma(n)
        - np.mean(digamma(nx + 1) + digamma(ny + 1))
    )
    return float(estimate)


class KSGDijkstra:
    def __init__(self, x, proposal, cfg: ExperimentConfig, seed: int):
        self.x = np.asarray(x, dtype=float)
        self.cfg = cfg
        self.seed = int(seed)
        self.n, self.p = self.x.shape
        self.full = (1 << self.p) - 1
        self.residuals = proposal.ResidualCache(self.x, cfg.regression_ridge)
        self.cache: Dict[Tuple[int, int], float] = {}
        self.evaluated = 0
        self.expanded = 0
        self.max_open = 1
        self.im = None
        if cfg.ksg_backend == "infomeasure":
            import infomeasure as im
            self.im = im

    def edge_seed(self, mask, candidate):
        return int(
            (self.seed * 1_000_003 + mask * 9_176 + candidate * 37)
            % (2**32 - 1)
        )

    def estimate(self, x_data, y_data, seed: int) -> float:
        if self.cfg.ksg_backend == "ckdtree":
            return ksg_mutual_information_type1_ckdtree(
                x=x_data,
                y=y_data,
                k=self.cfg.k_ksg,
                noise_level=self.cfg.ksg_noise_level,
                normalize=self.cfg.ksg_normalize,
                seed=seed,
                leafsize=self.cfg.ksg_leafsize,
                tree_workers=self.cfg.ksg_tree_workers,
            )

        x_ready, y_ready = _prepare_ksg_data(
            x_data,
            y_data,
            self.cfg.ksg_noise_level,
            self.cfg.ksg_normalize,
            seed,
        )
        # Noise and normalization were already applied deterministically above;
        # disable infomeasure's internal jitter to avoid adding it twice.
        try:
            value = self.im.mutual_information(
                x_ready,
                y_ready,
                approach="ksg",
                k=int(self.cfg.k_ksg),
                ksg_id=1,
                noise_level=0.0,
                normalize=False,
            )
        except (ValueError, KeyError, TypeError):
            value = self.im.mutual_information(
                x_ready,
                y_ready,
                approach="metric",
                k=int(self.cfg.k_ksg),
                noise_level=0.0,
                normalize=False,
            )
        return float(value)

    def edge(self, mask: int, candidate: int) -> float:
        key = (mask, candidate)
        if key in self.cache:
            return self.cache[key]
        remaining, block = self.residuals.block(mask)
        local = remaining.index(candidate)
        xj, rest = residualize_rest_on_candidate(
            block, local, self.cfg.regression_ridge
        )
        if rest.shape[1] == 0:
            value = 0.0
        else:
            raw = self.estimate(
                xj,
                rest,
                self.edge_seed(mask, candidate),
            )
            value = max(0.0, float(raw))
        self.cache[key] = float(value)
        self.evaluated += 1
        return float(value)

    def solve(self) -> MethodResult:
        started = time.perf_counter()
        distance = {0: 0.0}
        predecessor = {}
        settled = set()
        insertion = 0
        heap = [(0.0, insertion, 0)]
        while heap:
            d, _, mask = heapq.heappop(heap)
            if mask in settled:
                continue
            if abs(d - distance.get(mask, math.inf)) > 1.0e-12:
                continue
            settled.add(mask)
            self.expanded += 1
            if mask == self.full:
                break
            for candidate in range(self.p):
                if mask & (1 << candidate):
                    continue
                nxt = mask | (1 << candidate)
                nd = d + self.edge(mask, candidate)
                if nd < distance.get(nxt, math.inf) - 1.0e-14:
                    distance[nxt] = float(nd)
                    predecessor[nxt] = (mask, candidate)
                    insertion += 1
                    heapq.heappush(heap, (float(nd), insertion, nxt))
            self.max_open = max(self.max_open, len(heap))

        if self.full not in settled:
            raise RuntimeError("KSG shortest path did not reach the full state.")
        reverse = []
        mask = self.full
        while mask:
            mask, candidate = predecessor[mask]
            reverse.append(candidate)

        if self.cfg.ksg_backend == "ckdtree":
            import scipy
            estimator = "scipy.cKDTree KSG-I"
            version = str(getattr(scipy, "__version__", "unknown"))
        else:
            estimator = "infomeasure KSG-I"
            version = str(getattr(self.im, "__version__", "unknown"))
        return MethodResult(
            order=tuple(reversed(reverse)),
            score=float(distance[self.full]),
            elapsed=float(time.perf_counter() - started),
            expanded=self.expanded,
            evaluated=self.evaluated,
            discovered=len(distance),
            max_open=self.max_open,
            diagnostics={
                "estimator": estimator,
                "version": version,
                "k": int(self.cfg.k_ksg),
                "backend": self.cfg.ksg_backend,
                "leafsize": int(self.cfg.ksg_leafsize),
                "tree_workers": int(self.cfg.ksg_tree_workers),
                "jitter": "single deterministic",
                "negative_correction": "max(0, estimate)",
            },
        )



class StandardizedShapeDijkstra:
    """Exact shortest path for the standardized Student-t shape criterion.

    For every state and candidate variable, the OLS innovation is centered and
    divided by its RMS.  The edge cost is its empirical cross entropy under a
    Student-t distribution with ``df`` degrees of freedom and unit variance.
    All edge costs are strictly positive, so ordinary Dijkstra is exact for the
    criterion.  This implementation intentionally mirrors the V12 df=3 run.
    """

    def __init__(self, x, proposal, cfg: ExperimentConfig):
        self.x = np.asarray(x, dtype=float)
        self.cfg = cfg
        self.n, self.p = self.x.shape
        self.full = (1 << self.p) - 1
        self.residuals = proposal.ResidualCache(self.x, cfg.regression_ridge)
        self.cache: Dict[Tuple[int, int], float] = {}
        self.evaluated = 0
        self.expanded = 0
        self.max_open = 1
        self.discovered = 1

        nu = float(cfg.df)
        if nu <= 2.0:
            raise ValueError(
                "The standardized-shape criterion requires df > 2 so the "
                "unit-variance Student-t scale is defined."
            )
        from scipy.special import gammaln
        self._nu = nu
        # Student-t with scale sqrt((nu-2)/nu), hence variance one.
        self._log_constant = float(
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * math.log(math.pi * (nu - 2.0))
        )

    def edge(self, mask: int, candidate: int) -> float:
        key = (mask, candidate)
        if key in self.cache:
            return self.cache[key]
        remaining, block = self.residuals.block(mask)
        local = remaining.index(candidate)
        innovation = np.asarray(block[:, local], dtype=float)
        centered = innovation - float(np.mean(innovation))
        rms = math.sqrt(float(np.mean(centered * centered)))
        rms = max(rms, float(self.cfg.scale_floor))
        z = centered / rms
        logpdf = self._log_constant - 0.5 * (self._nu + 1.0) * np.log1p(
            (z * z) / (self._nu - 2.0)
        )
        value = -float(np.mean(logpdf))
        self.cache[key] = value
        self.evaluated += 1
        return value

    def solve(self) -> MethodResult:
        started = time.perf_counter()
        distance: Dict[int, float] = {0: 0.0}
        predecessor: Dict[int, Tuple[int, int]] = {}
        settled: set[int] = set()
        insertion = 0
        heap: List[Tuple[float, int, int]] = [(0.0, insertion, 0)]

        while heap:
            current, _, mask = heapq.heappop(heap)
            if mask in settled:
                continue
            if abs(current - distance.get(mask, math.inf)) > 1.0e-12:
                continue
            settled.add(mask)
            self.expanded += 1
            if mask == self.full:
                break
            remaining, _ = self.residuals.block(mask)
            for candidate in remaining:
                nxt = mask | (1 << candidate)
                proposed = current + self.edge(mask, int(candidate))
                old = distance.get(nxt, math.inf)
                if proposed < old - 1.0e-14:
                    distance[nxt] = float(proposed)
                    predecessor[nxt] = (mask, int(candidate))
                    insertion += 1
                    heapq.heappush(heap, (float(proposed), insertion, nxt))
            self.max_open = max(self.max_open, len(heap))

        if self.full not in settled:
            raise RuntimeError("Standardized-shape Dijkstra did not reach the full state.")

        reverse: List[int] = []
        mask = self.full
        while mask:
            parent, candidate = predecessor[mask]
            reverse.append(int(candidate))
            mask = int(parent)
        order = tuple(reversed(reverse))
        self.discovered = len(distance)
        return MethodResult(
            order=order,
            score=float(distance[self.full]),
            elapsed=float(time.perf_counter() - started),
            expanded=int(self.expanded),
            evaluated=int(self.evaluated),
            discovered=int(self.discovered),
            max_open=int(self.max_open),
            diagnostics={
                "solver_mode": "ordinary Dijkstra on standardized-shape score",
                "criterion": (
                    "unit-variance Student-t cross entropy of centered "
                    "RMS-standardized OLS residuals"
                ),
                "df": float(self.cfg.df),
                "edge_shift": 0.0,
                "exact_for_defined_score": 1,
                "separate_exact_dp_audit": 0,
            },
        )

def run_method(method, x, p, n, beta, gamma, pairs, seed, cfg, proposal):
    try:
        if method == "proposed_standardized_shape":
            return StandardizedShapeDijkstra(x, proposal, cfg).solve()

        if method in {"proposed_v11_marginal", "proposed_hybrid"}:
            pcfg = proposal.Config(
                p=p, n=n, reps=1, beta=beta, df=cfg.df, gamma=gamma,
                confounded_pairs=tuple(pairs), seed=seed,
                regression_ridge=cfg.regression_ridge,
                scale_floor=cfg.scale_floor,
                hessian_eigen_floor=cfg.hessian_eigen_floor,
                optimizer_maxiter=cfg.optimizer_maxiter,
                ica_starts=cfg.ica_starts,
                ica_maxiter=cfg.ica_maxiter,
                ica_ftol=cfg.ica_ftol,
                ica_gtol=cfg.ica_gtol,
                ica_warm_start_repair=False,
                ica_repair_max_restarts=cfg.ica_repair_max_restarts,
                ica_embedding_tolerance=cfg.ica_embedding_tolerance,
                numerical_tolerance=cfg.numerical_tolerance,
                score_tolerance=cfg.score_tolerance,
                skip_dp=True,
            )
            import variable_p_hybrid_no_repair_dp_fallback_v10 as hybrid
            result, diagnostics = hybrid.HybridDijkstraDPFallback(x, pcfg).solve()
            diagnostics = {
                **proposal.backend_info(),
                **diagnostics,
                "comparison_label": "previous V11 marginal-likelihood proposal",
                "separate_exact_dp_audit": 0,
            }
            return MethodResult(
                order=tuple(result.order), score=float(result.score),
                elapsed=float(result.elapsed), expanded=int(result.expanded_nodes),
                evaluated=int(result.evaluated_edges),
                discovered=int(result.discovered_nodes),
                max_open=int(result.max_open_size), diagnostics=diagnostics,
            )

        if method == "proposed_exact_dp":
            pcfg = proposal.Config(
                p=p, n=n, reps=1, beta=beta, df=cfg.df, gamma=gamma,
                confounded_pairs=tuple(pairs), seed=seed,
                regression_ridge=cfg.regression_ridge,
                scale_floor=cfg.scale_floor,
                hessian_eigen_floor=cfg.hessian_eigen_floor,
                optimizer_maxiter=cfg.optimizer_maxiter,
                ica_starts=cfg.ica_starts,
                ica_maxiter=cfg.ica_maxiter,
                ica_ftol=cfg.ica_ftol,
                ica_gtol=cfg.ica_gtol,
                ica_warm_start_repair=False,
                ica_repair_max_restarts=cfg.ica_repair_max_restarts,
                ica_embedding_tolerance=cfg.ica_embedding_tolerance,
                numerical_tolerance=cfg.numerical_tolerance,
                score_tolerance=cfg.score_tolerance,
                skip_dp=True,
            )
            import variable_p_direct_marginal_dag_v6 as exact_dag
            result, diagnostics = exact_dag.DirectMarginalDAGShortestPath(x, pcfg).solve()
            diagnostics = {
                **proposal.backend_info(),
                **diagnostics,
                "solver_mode": "exact DP on original criterion",
                "dp_fallback": "",
                "negative_edge_observed": "",
                "exact_dp_reused_from_hybrid": 0,
            }
            return MethodResult(
                order=tuple(result.order), score=float(result.score),
                elapsed=float(result.elapsed), expanded=int(result.expanded_nodes),
                evaluated=int(result.evaluated_edges),
                discovered=int(result.discovered_nodes),
                max_open=int(result.max_open_size), diagnostics=diagnostics,
            )

        if method == "ksg_shortest_path":
            return KSGDijkstra(x, proposal, cfg, seed).solve()

        import lingam
        if method == "direct_lingam":
            started = time.perf_counter()
            model = lingam.DirectLiNGAM(
                random_state=int(seed), measure=cfg.direct_measure
            )
            model.fit(x)
            return MethodResult(
                order=tuple(int(v) for v in model.causal_order_),
                score=None,
                elapsed=float(time.perf_counter() - started),
                diagnostics={"package": "lingam", "version": getattr(lingam, "__version__", "unknown")},
            )
        if method == "ica_lingam":
            started = time.perf_counter()
            model = lingam.ICALiNGAM(
                random_state=int(seed), max_iter=cfg.lingam_ica_max_iter
            )
            model.fit(x)
            return MethodResult(
                order=tuple(int(v) for v in model.causal_order_),
                score=None,
                elapsed=float(time.perf_counter() - started),
                diagnostics={"package": "lingam", "version": getattr(lingam, "__version__", "unknown")},
            )
        raise ValueError(f"Unknown method: {method}")
    except Exception as exc:
        return MethodResult(
            order=tuple(), score=None, elapsed=0.0, status="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}",
        )


def _result_row(
    result: MethodResult,
    cfg: ExperimentConfig,
    p: int,
    n: int,
    beta: float,
    gamma: float,
    pattern: str,
    pairs,
    rep: int,
    seed: int,
    permutation,
    true_order,
    method: str,
    parallel_unit: str,
):
    total_pairs = p * (p - 1) // 2
    if result.status == "ok":
        exact = int(result.order == true_order)
        pe = pair_error(result.order, true_order)
        pa = 1.0 - pe / total_pairs
    else:
        exact = ""
        pe = ""
        pa = ""

    diagnostics = dict(result.diagnostics or {})
    diagnostics["outer_parallel_unit"] = parallel_unit
    return {
        "p": p,
        "n": n,
        "beta": beta,
        "gamma": gamma,
        "pattern": pattern,
        "pairs": json.dumps(pairs),
        "rep": rep,
        "replication_seed": seed,
        "permutation": json.dumps(permutation),
        "true_order": json.dumps(true_order),
        "method": method,
        "estimated_order": (
            json.dumps(result.order) if result.status == "ok" else ""
        ),
        "exact": exact,
        "pair_error": pe,
        "pair_accuracy": pa,
        "score": "" if result.score is None else result.score,
        "elapsed_seconds": result.elapsed,
        "expanded_nodes": "" if result.expanded is None else result.expanded,
        "evaluated_edges": "" if result.evaluated is None else result.evaluated,
        "discovered_nodes": "" if result.discovered is None else result.discovered,
        "max_open_size": "" if result.max_open is None else result.max_open,
        "solver_mode": diagnostics.get("solver_mode", ""),
        "dp_fallback": diagnostics.get("dp_fallback", ""),
        "negative_edge_observed": diagnostics.get("negative_edge_observed", ""),
        "negative_kind": diagnostics.get("negative_kind", ""),
        "negative_depth": diagnostics.get("negative_depth", ""),
        "negative_value": diagnostics.get("negative_value", ""),
        "dijkstra_attempt_expanded": diagnostics.get("dijkstra_attempt_expanded", ""),
        "dijkstra_attempt_evaluated_edges": diagnostics.get("dijkstra_attempt_evaluated_edges", ""),
        "exact_dp_reused_from_hybrid": diagnostics.get("exact_dp_reused_from_hybrid", ""),
        "hybrid_exact_order_match": diagnostics.get("hybrid_exact_order_match", ""),
        "hybrid_exact_score_match": diagnostics.get("hybrid_exact_score_match", ""),
        "hybrid_exact_score_difference": diagnostics.get("hybrid_exact_score_difference", ""),
        "hidden_negative_effect_detected": diagnostics.get("hidden_negative_effect_detected", ""),
        "parallel_jobs": cfg.jobs,
        "diagnostics": json.dumps(diagnostics, default=str),
        "status": result.status,
        "error": result.error,
    }


def run_work_unit(task):
    """Run one work unit while preventing nested BLAS oversubscription."""
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return _run_work_unit_unlimited(task)
    with threadpool_limits(limits=1):
        return _run_work_unit_unlimited(task)


def _run_work_unit_unlimited(task):
    """Run one replication-, condition-, or method-level work unit.

    ``specs`` contains ``(n, condition_index, methods)`` tuples.  The complete
    maximum-n base sample is regenerated from the replication seed so every
    work unit sees exactly the same paired and nested data as the serial code.
    """
    cfg_dict, p, beta, beta_index, rep, parallel_unit, specs = task
    cfg = ExperimentConfig(**cfg_dict)
    # Import inside the worker after thread-count environment variables are set.
    import variable_p_telescoping_standard_dijkstra_fullopt_v4 as proposal

    n_values = N_BY_P[p]
    seed = replication_seed(cfg.seed, p, beta_index, rep)
    errors, hidden, permutation = generate_base(
        max(n_values), p, cfg.df, union_pairs(p), seed
    )
    rows = []
    for n, condition_index, methods in specs:
        pattern, gamma = CONDITIONS[int(condition_index)]
        pairs = pairs_for(p, pattern)
        x0 = simulate(
            int(n),
            p,
            beta,
            gamma,
            pairs,
            errors,
            hidden,
            cfg.normalize_pair_loading,
        )
        x, true_order = apply_permutation(x0, permutation)
        results = {}
        ordered_methods = list(methods)
        if "proposed_hybrid" in ordered_methods:
            results["proposed_hybrid"] = run_method(
                "proposed_hybrid", x, p, int(n), beta, gamma, pairs,
                seed + 10_000, cfg, proposal,
            )

        if "proposed_exact_dp" in ordered_methods:
            hybrid_result = results.get("proposed_hybrid")
            hybrid_diag = dict((hybrid_result.diagnostics or {}) if hybrid_result else {})
            if (hybrid_result is not None and hybrid_result.status == "ok"
                    and int(hybrid_diag.get("dp_fallback", 0) or 0) == 1):
                dp_diag = dict(hybrid_diag.get("dp_diagnostics", {}) or {})
                exact_diag = {
                    **proposal.backend_info(),
                    **dp_diag,
                    "solver_mode": "exact DP on original criterion",
                    "dp_fallback": "",
                    "negative_edge_observed": "",
                    "exact_dp_reused_from_hybrid": 1,
                }
                results["proposed_exact_dp"] = MethodResult(
                    order=tuple(hybrid_result.order),
                    score=float(hybrid_result.score),
                    elapsed=float(hybrid_diag.get("dp_elapsed_seconds", 0.0)),
                    expanded=int(dp_diag.get("number_of_states", 1 << p)),
                    evaluated=int(dp_diag.get("theoretical_edge_count", p * (1 << (p - 1)))),
                    discovered=int(dp_diag.get("number_of_states", 1 << p)),
                    max_open=0,
                    diagnostics=exact_diag,
                )
            else:
                results["proposed_exact_dp"] = run_method(
                    "proposed_exact_dp", x, p, int(n), beta, gamma, pairs,
                    seed + 10_000, cfg, proposal,
                )

        if "proposed_hybrid" in results and "proposed_exact_dp" in results:
            h = results["proposed_hybrid"]
            e = results["proposed_exact_dp"]
            if h.status == "ok" and e.status == "ok":
                order_match = int(tuple(h.order) == tuple(e.order))
                score_diff = float(h.score) - float(e.score)
                score_match = int(abs(score_diff) <= cfg.score_tolerance)
                hdiag = dict(h.diagnostics or {})
                ediag = dict(e.diagnostics or {})
                fallback = int(hdiag.get("dp_fallback", 0) or 0)
                hidden_effect = int(fallback == 0 and (not order_match or not score_match))
                audit = {
                    "hybrid_exact_order_match": order_match,
                    "hybrid_exact_score_match": score_match,
                    "hybrid_exact_score_difference": score_diff,
                    "hidden_negative_effect_detected": hidden_effect,
                }
                h.diagnostics = {**hdiag, **audit}
                e.diagnostics = {**ediag, **audit}

        for method in ordered_methods:
            if method not in results:
                results[method] = run_method(
                    method, x, p, int(n), beta, gamma, pairs,
                    seed + 10_000, cfg, proposal,
                )
            result = results[method]
            rows.append(_result_row(
                result=result, cfg=cfg, p=p, n=int(n), beta=beta, gamma=gamma,
                pattern=pattern, pairs=pairs, rep=rep, seed=seed,
                permutation=permutation, true_order=true_order, method=method,
                parallel_unit=parallel_unit,
            ))
    return p, beta, rep, rows


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]):
    if not rows:
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_key(row):
    return (
        int(row["p"]), float(row["beta"]), int(row["rep"]),
        int(row["n"]), float(row["gamma"]), str(row["pattern"]), str(row["method"]),
    )


def summarize(rows):
    groups = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        key = (
            int(row["p"]), int(row["n"]), float(row["beta"]),
            float(row["gamma"]), str(row["pattern"]), str(row["method"]),
        )
        groups.setdefault(key, []).append(row)
    out = []
    for key, group in sorted(groups.items()):
        p,n,beta,gamma,pattern,method = key
        expanded = [float(r["expanded_nodes"]) for r in group if r["expanded_nodes"] not in ("",None)]
        evaluated = [float(r["evaluated_edges"]) for r in group if r["evaluated_edges"] not in ("",None)]
        fallback_values = [
            float(r["dp_fallback"])
            for r in group
            if r.get("dp_fallback") not in ("", None)
        ]
        negative_values = [
            float(r["negative_edge_observed"])
            for r in group
            if r.get("negative_edge_observed") not in ("", None)
        ]
        out.append({
            "p":p,"n":n,"beta":beta,"gamma":gamma,"pattern":pattern,"method":method,
            "successful_reps":len(group),
            "exact_rate":float(np.mean([float(r["exact"]) for r in group])),
            "mean_pair_error":float(np.mean([float(r["pair_error"]) for r in group])),
            "mean_pair_accuracy":float(np.mean([float(r["pair_accuracy"]) for r in group])),
            "mean_elapsed_seconds":float(np.mean([float(r["elapsed_seconds"]) for r in group])),
            "mean_expanded_nodes":float(np.mean(expanded)) if expanded else "",
            "mean_evaluated_edges":float(np.mean(evaluated)) if evaluated else "",
            "dp_fallback_rate":float(np.mean(fallback_values)) if fallback_values else "",
            "negative_edge_observed_rate":float(np.mean(negative_values)) if negative_values else "",
            "hybrid_exact_order_match_rate": float(np.mean([float(r["hybrid_exact_order_match"]) for r in group if r.get("hybrid_exact_order_match") not in ("", None)])) if any(r.get("hybrid_exact_order_match") not in ("", None) for r in group) else "",
            "hybrid_exact_score_match_rate": float(np.mean([float(r["hybrid_exact_score_match"]) for r in group if r.get("hybrid_exact_score_match") not in ("", None)])) if any(r.get("hybrid_exact_score_match") not in ("", None) for r in group) else "",
            "hidden_negative_effect_rate": float(np.mean([float(r["hidden_negative_effect_detected"]) for r in group if r.get("hidden_negative_effect_detected") not in ("", None)])) if any(r.get("hidden_negative_effect_detected") not in ("", None) for r in group) else "",
            "parallel_jobs":int(group[0]["parallel_jobs"]),
        })
    return out



def overall_summary(rows):
    groups: Dict[Tuple[int, str], List[Mapping[str, object]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        groups.setdefault((int(row["p"]), str(row["method"])), []).append(row)
    preferred = {
        "proposed_standardized_shape": 0,
        "proposed_v11_marginal": 1,
        "direct_lingam": 2,
        "ica_lingam": 3,
    }
    out = []
    for (p, method), group in sorted(
        groups.items(), key=lambda item: (item[0][0], preferred.get(item[0][1], 99))
    ):
        fallback = [
            float(r["dp_fallback"])
            for r in group
            if r.get("dp_fallback") not in ("", None)
        ]
        out.append({
            "p": p,
            "method": method,
            "successful_rows": len(group),
            "number_of_reps": len({int(r["rep"]) for r in group}),
            "exact_rate": float(np.mean([float(r["exact"]) for r in group])),
            "mean_pair_error": float(np.mean([float(r["pair_error"]) for r in group])),
            "mean_pair_accuracy": float(np.mean([float(r["pair_accuracy"]) for r in group])),
            "mean_elapsed_seconds_per_condition": float(
                np.mean([float(r["elapsed_seconds"]) for r in group])
            ),
            "dp_fallback_rate": float(np.mean(fallback)) if fallback else "",
        })
    return out


def paired_comparisons(rows):
    ok = [r for r in rows if r.get("status") == "ok"]
    index = {
        (
            int(r["p"]), int(r["n"]), float(r["beta"]), float(r["gamma"]),
            str(r["pattern"]), int(r["rep"]), str(r["method"]),
        ): r
        for r in ok
    }
    shape = "proposed_standardized_shape"
    competitors = ("proposed_v11_marginal", "direct_lingam", "ica_lingam")
    conditions = sorted({
        (int(r["p"]), int(r["n"]), float(r["beta"]), float(r["gamma"]), str(r["pattern"]))
        for r in ok if r["method"] == shape
    })
    out = []

    def make_row(scope, n, pattern, gamma, pairs, competitor):
        diffs = []
        shape_exact = []
        comp_exact = []
        sb = tie = cb = 0
        serrors = []
        cerrors = []
        for sr, cr in pairs:
            se = float(sr["pair_error"])
            ce = float(cr["pair_error"])
            serrors.append(se); cerrors.append(ce)
            diffs.append(ce - se)
            shape_exact.append(float(sr["exact"]))
            comp_exact.append(float(cr["exact"]))
            if se < ce:
                sb += 1
            elif se == ce:
                tie += 1
            else:
                cb += 1
        return {
            "scope": scope,
            "n": "" if n is None else n,
            "pattern": "" if pattern is None else pattern,
            "gamma": "" if gamma is None else gamma,
            "reference_method": shape,
            "competitor_method": competitor,
            "paired_rows": len(pairs),
            "shape_mean_pair_error": float(np.mean(serrors)),
            "competitor_mean_pair_error": float(np.mean(cerrors)),
            "mean_pair_error_improvement_competitor_minus_shape": float(np.mean(diffs)),
            "shape_exact_rate": float(np.mean(shape_exact)),
            "competitor_exact_rate": float(np.mean(comp_exact)),
            "shape_better_count": sb,
            "tie_count": tie,
            "competitor_better_count": cb,
        }

    for competitor in competitors:
        pairs = []
        for key, sr in index.items():
            if key[-1] != shape:
                continue
            cr = index.get(key[:-1] + (competitor,))
            if cr is not None:
                pairs.append((sr, cr))
        if pairs:
            out.append(make_row("overall", None, None, None, pairs, competitor))

    for p, n, beta, gamma, pattern in conditions:
        for competitor in competitors:
            pairs = []
            reps = sorted({
                int(r["rep"]) for r in ok
                if int(r["p"]) == p and int(r["n"]) == n
                and float(r["beta"]) == beta and float(r["gamma"]) == gamma
                and str(r["pattern"]) == pattern and r["method"] == shape
            })
            for rep in reps:
                base = (p, n, beta, gamma, pattern, rep)
                sr = index.get(base + (shape,))
                cr = index.get(base + (competitor,))
                if sr is not None and cr is not None:
                    pairs.append((sr, cr))
            if pairs:
                out.append(make_row("condition", n, pattern, gamma, pairs, competitor))
    return out

def _effective_parallel_unit(cfg: ExperimentConfig) -> str:
    if cfg.parallel_unit == "auto":
        return "replication" if cfg.jobs == 1 else "condition"
    return cfg.parallel_unit


def _build_work_units(cfg: ExperimentConfig, rowmap, parallel_unit: str):
    tasks = []
    missing_rows = 0
    cfg_dict = asdict(cfg)
    for p in cfg.p_values:
        for beta_index, beta in enumerate(cfg.beta_values):
            for rep in range(1, cfg.reps + 1):
                condition_specs = []
                for n in N_BY_P[p]:
                    for condition_index, (pattern, gamma) in enumerate(CONDITIONS):
                        missing_methods = []
                        for method in cfg.methods:
                            key = (
                                int(p),
                                float(beta),
                                int(rep),
                                int(n),
                                float(gamma),
                                str(pattern),
                                str(method),
                            )
                            row = rowmap.get(key)
                            if row is None or row.get("status") != "ok":
                                missing_methods.append(method)
                        if not missing_methods:
                            continue
                        missing_rows += len(missing_methods)
                        condition_specs.append(
                            (int(n), int(condition_index), tuple(missing_methods))
                        )

                if not condition_specs:
                    continue
                if parallel_unit == "replication":
                    tasks.append(
                        (
                            cfg_dict,
                            p,
                            beta,
                            beta_index,
                            rep,
                            parallel_unit,
                            tuple(condition_specs),
                        )
                    )
                elif parallel_unit == "condition":
                    tasks.extend(
                        (
                            cfg_dict,
                            p,
                            beta,
                            beta_index,
                            rep,
                            parallel_unit,
                            (spec,),
                        )
                        for spec in condition_specs
                    )
                elif parallel_unit == "method":
                    for n, condition_index, methods in condition_specs:
                        tasks.extend(
                            (
                                cfg_dict,
                                p,
                                beta,
                                beta_index,
                                rep,
                                parallel_unit,
                                ((n, condition_index, (method,)),),
                            )
                            for method in methods
                        )
                else:
                    raise ValueError(f"Unknown parallel unit: {parallel_unit}")
    return tasks, missing_rows


def run(cfg: ExperimentConfig):
    for p in cfg.p_values:
        if p not in N_BY_P:
            raise ValueError(f"Unsupported p={p}; choose from {tuple(N_BY_P)}.")
    if cfg.reps < 1:
        raise ValueError("reps must be positive.")
    if cfg.jobs < 1:
        raise ValueError("jobs must be positive.")
    if cfg.parallel_unit not in {"auto", "replication", "condition", "method"}:
        raise ValueError("parallel_unit must be auto, replication, condition, or method.")
    if cfg.ksg_backend not in {"ckdtree", "infomeasure"}:
        raise ValueError("ksg_backend must be ckdtree or infomeasure.")
    if cfg.ksg_leafsize < 0:
        raise ValueError("ksg_leafsize must be zero (adaptive) or positive.")
    if cfg.ksg_tree_workers == 0 or cfg.ksg_tree_workers < -1:
        raise ValueError("ksg_tree_workers must be -1 or a positive integer.")

    parallel_unit = _effective_parallel_unit(cfg)
    prefix = Path(cfg.output_prefix)
    raw_path = Path(str(prefix) + "_raw.csv")
    summary_path = Path(str(prefix) + "_summary.csv")
    overall_path = Path(str(prefix) + "_overall_summary.csv")
    paired_path = Path(str(prefix) + "_paired_comparisons.csv")
    config_path = Path(str(prefix) + "_config.json")
    if cfg.overwrite:
        for path in (raw_path, summary_path, overall_path, paired_path, config_path):
            if path.exists():
                path.unlink()

    existing = read_csv(raw_path) if cfg.resume else []
    rowmap = {row_key(r): r for r in existing}
    tasks, missing_rows = _build_work_units(cfg, rowmap, parallel_unit)

    config_payload = asdict(cfg)
    config_payload["effective_parallel_unit"] = parallel_unit
    config_path.write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8"
    )

    worker_count = min(cfg.jobs, len(tasks)) if tasks else 0
    print("V12 variable-p comparison: standardized shape + DirectLiNGAM + ICA-LiNGAM", flush=True)
    print(
        f"p_values={cfg.p_values}, n_by_p={N_BY_P}, reps={cfg.reps}, "
        f"jobs={cfg.jobs}, active_workers={worker_count}",
        flush=True,
    )
    print(f"parallel_unit={parallel_unit}", flush=True)
    print(f"conditions={CONDITIONS}", flush=True)
    print(f"methods={cfg.methods}", flush=True)
    print("Compiled proposal backend: Cython + BLAS/LAPACK", flush=True)
    if "ksg_shortest_path" in cfg.methods:
        print(
            f"KSG backend={cfg.ksg_backend}, leafsize="
            f"{'adaptive' if cfg.ksg_leafsize == 0 else cfg.ksg_leafsize}, "
            f"tree_workers={cfg.ksg_tree_workers}",
            flush=True,
        )
    if cfg.jobs > 1:
        print(
            "WARNING: per-method elapsed times under parallel execution include "
            "CPU contention; use total wall-clock time for completion speed.",
            flush=True,
        )
    if cfg.jobs > 1 and cfg.ksg_tree_workers != 1:
        print(
            "WARNING: outer process parallelism plus KSG tree workers may "
            "oversubscribe CPUs. Usually keep --ksg-tree-workers 1.",
            flush=True,
        )
    print(
        f"pending work units={len(tasks)}, pending method-condition rows={missing_rows}\n",
        flush=True,
    )

    started = time.perf_counter()
    completed_rows = 0

    def accept_result(p, beta, rep, rows):
        nonlocal completed_rows
        for row in rows:
            rowmap[row_key(row)] = row
        completed_rows += len(rows)
        ordered = sorted(rowmap.values(), key=row_key)
        write_csv(raw_path, ordered)
        write_csv(summary_path, summarize(ordered))
        write_csv(overall_path, overall_summary(ordered))
        write_csv(paired_path, paired_comparisons(ordered))
        ok = sum(r["status"] == "ok" for r in rows)
        print(
            f"completed p={p}, beta={beta}, rep={rep}: "
            f"{ok}/{len(rows)} rows; progress={completed_rows}/{missing_rows}",
            flush=True,
        )

    if cfg.jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            p, beta, rep, rows = run_work_unit(task)
            accept_result(p, beta, rep, rows)
    elif tasks:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(run_work_unit, task): task for task in tasks
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    p, beta, rep, rows = future.result()
                except Exception as exc:
                    print(
                        f"FAILED work unit p={task[1]}, beta={task[2]}, "
                        f"rep={task[4]}: {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                    continue
                accept_result(p, beta, rep, rows)

    ordered = sorted(rowmap.values(), key=row_key)
    write_csv(raw_path, ordered)
    write_csv(summary_path, summarize(ordered))
    write_csv(overall_path, overall_summary(ordered))
    write_csv(paired_path, paired_comparisons(ordered))
    print(
        f"\nFinished in {(time.perf_counter() - started) / 3600:.3f} hours",
        flush=True,
    )
    print(f"Raw     : {raw_path}", flush=True)
    print(f"Summary : {summary_path}", flush=True)
    print(f"Overall : {overall_path}", flush=True)
    print(f"Paired  : {paired_path}", flush=True)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "V12 runner comparing standardized-shape proposal with DirectLiNGAM and ICA-LiNGAM."
        )
    )
    parser.add_argument("--p-values", nargs="+", type=int, default=[5])
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--beta-values", nargs="+", type=float, default=[0.4])
    parser.add_argument("--df", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Maximum number of independent worker processes.",
    )
    parser.add_argument(
        "--parallel-unit",
        choices=["auto", "replication", "condition", "method"],
        default="auto",
        help=(
            "auto uses replication when jobs=1 and condition when jobs>1; "
            "condition gives 15 tasks per (p,beta,rep)."
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "direct_lingam",
            "ica_lingam",
            "proposed_standardized_shape",
            "proposed_v11_marginal",
        ],
        default=[
            "proposed_standardized_shape",
            "direct_lingam",
            "ica_lingam",
        ],
    )
    parser.add_argument("--raw-pair-loading", action="store_true")
    parser.add_argument("--ica-starts", type=int, default=2)
    parser.add_argument("--ica-maxiter", type=int, default=500)
    parser.add_argument("--lingam-ica-max-iter", type=int, default=1000)
    parser.add_argument("--k-ksg", type=int, default=5)
    parser.add_argument("--ksg-noise-level", type=float, default=1.0e-10)
    parser.add_argument("--ksg-normalize", action="store_true")
    parser.add_argument(
        "--ksg-backend",
        choices=["ckdtree", "infomeasure"],
        default="ckdtree",
        help="ckdtree is the faster vectorized KSG-I implementation.",
    )
    parser.add_argument(
        "--ksg-leafsize",
        type=int,
        default=0,
        help="0 selects an adaptive cKDTree leaf size.",
    )
    parser.add_argument(
        "--ksg-tree-workers",
        type=int,
        default=1,
        help="Workers inside each tree query; keep 1 with outer --jobs > 1.",
    )
    parser.add_argument(
        "--output-prefix",
        default="p5_df5_standardized_direct_ica_v12",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    return ExperimentConfig(
        p_values=tuple(args.p_values),
        reps=args.reps,
        beta_values=tuple(args.beta_values),
        df=args.df,
        seed=args.seed,
        jobs=args.jobs,
        parallel_unit=args.parallel_unit,
        methods=tuple(args.methods),
        normalize_pair_loading=not args.raw_pair_loading,
        ica_starts=args.ica_starts,
        ica_maxiter=args.ica_maxiter,
        lingam_ica_max_iter=args.lingam_ica_max_iter,
        k_ksg=args.k_ksg,
        ksg_noise_level=args.ksg_noise_level,
        ksg_normalize=args.ksg_normalize,
        ksg_backend=args.ksg_backend,
        ksg_leafsize=args.ksg_leafsize,
        ksg_tree_workers=args.ksg_tree_workers,
        output_prefix=args.output_prefix,
        overwrite=args.overwrite,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    run(parse_args())
