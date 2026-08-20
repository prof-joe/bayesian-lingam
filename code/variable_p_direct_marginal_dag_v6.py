#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast exact shortest path for the proposed marginal-likelihood criterion.

The proposed complete-order score is

    sum_k -log g_{pi_k}(S_{k-1}) / n,

where g_j(S) is the one-dimensional Laplace marginal likelihood of the
innovation obtained after regressing variable j on the selected prefix S.

The subset graph is a directed acyclic graph.  Therefore its exact shortest
path can be computed in topological (subset-size) order even if an individual
continuous-density code length is negative.  No affine-ICA state potential,
high-dimensional mutual-information estimate, nonconvex multivariate
optimization, or Dijkstra restart is required.

Residual matrices are deliberately not cached across all 2^p states.  Each
state is processed once and then discarded, avoiding the large memory and
paging cost of the older exact-DP reference implementation.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Sequence, Tuple

import numpy as np

import variable_p_telescoping_standard_dijkstra_fullopt_v4 as proposal


class DirectMarginalDAGShortestPath:
    """Exact layered subset-DAG solver for the proposed score."""

    def __init__(self, x: np.ndarray, config: proposal.Config):
        self.x = np.asarray(x, dtype=float)
        self.config = config
        self.n, self.p = self.x.shape
        self.full_mask = (1 << self.p) - 1
        self.evaluated_edges = 0
        self.minimum_edge = math.inf
        self.maximum_edge = -math.inf
        self.negative_edges = 0

    def _residual_block(self, mask: int) -> Tuple[Tuple[int, ...], np.ndarray]:
        selected = tuple(j for j in range(self.p) if mask & (1 << j))
        remaining = tuple(j for j in range(self.p) if not (mask & (1 << j)))
        if not remaining:
            return remaining, np.empty((self.n, 0), dtype=float)

        y = self.x[:, remaining]
        if not selected:
            residual = y - y.mean(axis=0, keepdims=True)
        else:
            design = np.column_stack([np.ones(self.n), self.x[:, selected]])
            gram = design.T @ design
            gram.flat[:: gram.shape[0] + 1] += float(
                self.config.regression_ridge
            )
            coefficient = np.linalg.solve(gram, design.T @ y)
            residual = y - design @ coefficient
        return remaining, np.asarray(residual, dtype=float)

    def _edge_cost_from_column(self, innovation: np.ndarray) -> float:
        log_g, _ = proposal._core.log_marginal_t_location_scale(
            innovation,
            self.config,
        )
        value = -float(log_g) / self.n
        self.evaluated_edges += 1
        self.minimum_edge = min(self.minimum_edge, value)
        self.maximum_edge = max(self.maximum_edge, value)
        if value < 0.0:
            self.negative_edges += 1
        return value

    def solve(self):
        started = time.perf_counter()
        number_of_states = 1 << self.p
        distance = np.full(number_of_states, np.inf, dtype=float)
        predecessor_mask = np.full(number_of_states, -1, dtype=np.int64)
        predecessor_variable = np.full(number_of_states, -1, dtype=np.int16)
        distance[0] = 0.0

        for depth in range(self.p):
            for mask in range(number_of_states):
                if mask.bit_count() != depth or not np.isfinite(distance[mask]):
                    continue

                remaining, block = self._residual_block(mask)
                base = float(distance[mask])
                for local, candidate in enumerate(remaining):
                    edge = self._edge_cost_from_column(block[:, local])
                    next_mask = mask | (1 << int(candidate))
                    proposed_distance = base + edge
                    old_distance = float(distance[next_mask])
                    if proposed_distance < old_distance - 1.0e-14:
                        distance[next_mask] = proposed_distance
                        predecessor_mask[next_mask] = mask
                        predecessor_variable[next_mask] = int(candidate)
                    elif abs(proposed_distance - old_distance) <= 1.0e-14:
                        old_candidate = int(predecessor_variable[next_mask])
                        if old_candidate < 0 or int(candidate) < old_candidate:
                            predecessor_mask[next_mask] = mask
                            predecessor_variable[next_mask] = int(candidate)

        reversed_order = []
        mask = self.full_mask
        while mask:
            parent = int(predecessor_mask[mask])
            candidate = int(predecessor_variable[mask])
            if parent < 0 or candidate < 0:
                raise RuntimeError(
                    "The direct marginal subset-DAG solver failed to "
                    "reconstruct a complete order."
                )
            reversed_order.append(candidate)
            mask = parent

        result = proposal.SolverResult(
            order=tuple(reversed(reversed_order)),
            score=float(distance[self.full_mask]),
            elapsed=float(time.perf_counter() - started),
            expanded_nodes=int(number_of_states),
            evaluated_edges=int(self.evaluated_edges),
            discovered_nodes=int(number_of_states),
            max_open_size=0,
        )
        diagnostics: Dict[str, object] = {
            "solver": "exact topological shortest path on subset DAG",
            "objective": "sum of one-dimensional negative log marginal likelihoods",
            "high_dimensional_mutual_information_estimated": False,
            "affine_ica_state_potentials_used": False,
            "nonconvex_multivariate_optimization_used": False,
            "dijkstra_restarts": 0,
            "number_of_states": int(number_of_states),
            "theoretical_edge_count": int(self.p * (1 << (self.p - 1))),
            "minimum_direct_edge": float(self.minimum_edge),
            "maximum_direct_edge": float(self.maximum_edge),
            "negative_direct_edges": int(self.negative_edges),
            "memory_strategy": "residual block computed once per state and discarded",
        }
        return result, diagnostics
