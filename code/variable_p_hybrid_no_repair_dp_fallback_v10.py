#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair-free Dijkstra with exact-DP fallback on the first observed negative edge.

The state potentials are still computed by the same nonconvex affine-ICA
maximum-likelihood optimization used in v5.  No warm-start repair is attempted.
If a negative likelihood-ratio component, one-dimensional Bayes correction,
or total transformed edge is observed while Dijkstra is running, the Dijkstra
attempt is stopped immediately and the exact subset-DAG dynamic program for the
original one-dimensional marginal-likelihood objective is run on the same data.

A finite-sample caveat remains: a negative edge can exist in a state that
Dijkstra never expands.  Therefore a Dijkstra run that observes no negative
edge is not an absolute finite-sample certificate that every graph edge is
nonnegative.  This fact is reported explicitly in the diagnostics.
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

import variable_p_telescoping_standard_dijkstra_fullopt_v4 as proposal
import variable_p_direct_marginal_dag_v6 as exact_dag


@dataclass
class NegativeEdgeInformation:
    kind: str
    mask: int
    candidate: int
    value: float
    mi_term: float
    bayes_correction: float
    total_edge: float


class NegativeEdgeDetected(RuntimeError):
    def __init__(self, info: NegativeEdgeInformation):
        self.info = info
        super().__init__(
            f"Observed negative {info.kind}: mask={info.mask}, "
            f"candidate={info.candidate}, value={info.value:+.12e}"
        )


class RepairFreeCarriedLikelihoodDijkstra(proposal.CarriedLikelihoodDijkstra):
    """One-pass ordinary Dijkstra with no likelihood repair or restart."""

    def __init__(self, x: np.ndarray, config: proposal.Config):
        super().__init__(x, config)
        self.attempt_expanded = 0
        self.attempt_discovered = 1
        self.attempt_max_open = 1
        self.attempt_settled = 0

    def _raise_negative(
        self,
        kind: str,
        mask: int,
        candidate: int,
        value: float,
        mi_term: float,
        bayes_correction: float,
        total_edge: float,
    ) -> None:
        self.most_negative_mi_before_repair = min(
            self.most_negative_mi_before_repair,
            float(mi_term),
        )
        raise NegativeEdgeDetected(
            NegativeEdgeInformation(
                kind=str(kind),
                mask=int(mask),
                candidate=int(candidate),
                value=float(value),
                mi_term=float(mi_term),
                bayes_correction=float(bayes_correction),
                total_edge=float(total_edge),
            )
        )

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
        edge = mi_term + bayes_correction

        direct_form = (parent_loglik - log_g_j - child_loglik) / self.n
        decomposition_error = edge - direct_form
        self.maximum_decomposition_error = max(
            self.maximum_decomposition_error,
            abs(float(decomposition_error)),
        )
        self.minimum_mi_term = min(self.minimum_mi_term, float(mi_term))
        self.minimum_bayes_correction = min(
            self.minimum_bayes_correction,
            float(bayes_correction),
        )
        self.minimum_edge = min(self.minimum_edge, float(edge))
        # Count the edge even when it triggers fallback.
        self.evaluated_edges += 1

        tolerance = float(self.config.numerical_tolerance)
        if mi_term < -tolerance:
            self._raise_negative(
                "likelihood-ratio term",
                mask,
                candidate,
                mi_term,
                mi_term,
                bayes_correction,
                edge,
            )
        if bayes_correction < -tolerance:
            self._raise_negative(
                "one-dimensional Bayes correction",
                mask,
                candidate,
                bayes_correction,
                mi_term,
                bayes_correction,
                edge,
            )
        if edge < -tolerance:
            self._raise_negative(
                "total Dijkstra edge",
                mask,
                candidate,
                edge,
                mi_term,
                bayes_correction,
                edge,
            )

        return (
            max(0.0, float(edge)),
            max(0.0, float(mi_term)),
            max(0.0, float(bayes_correction)),
        )

    def _solve_once(self) -> Tuple[
        Tuple[int, ...], float, int, int, int
    ]:
        distance: Dict[int, float] = {0: 0.0}
        predecessor: Dict[int, Tuple[int, int]] = {}
        settled: set[int] = set()
        insertion = 0
        heap: List[Tuple[float, int, int]] = [(0.0, insertion, 0)]

        expanded = 0
        max_open_size = 1
        self.attempt_expanded = 0
        self.attempt_discovered = 1
        self.attempt_max_open = 1
        self.attempt_settled = 0

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
            self.attempt_expanded = int(expanded)
            self.attempt_settled = int(len(settled))
            self.attempt_discovered = int(len(distance))
            self.attempt_max_open = int(max_open_size)

            if mask == self.full_mask:
                break

            remaining, _ = self.residuals.block(mask)
            for candidate in remaining:
                edge, _, _ = self.edge_components(mask, candidate)
                next_mask = mask | (1 << candidate)
                proposed_distance = current_distance + edge
                old_distance = distance.get(next_mask, math.inf)
                if proposed_distance < old_distance - 1.0e-14:
                    distance[next_mask] = float(proposed_distance)
                    predecessor[next_mask] = (mask, int(candidate))
                    insertion += 1
                    heapq.heappush(
                        heap,
                        (float(proposed_distance), insertion, next_mask),
                    )
            max_open_size = max(max_open_size, len(heap))
            self.attempt_discovered = int(len(distance))
            self.attempt_max_open = int(max_open_size)

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

    def solve_no_repair(self):
        started = time.perf_counter()
        self.minimum_mi_term = math.inf
        self.minimum_bayes_correction = math.inf
        self.minimum_edge = math.inf
        self.maximum_decomposition_error = 0.0

        order, transformed_score, expanded, discovered, max_open_size = (
            self._solve_once()
        )
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
                "The repair-free Dijkstra path failed the telescoping "
                f"identity: error={telescoping_error:+.12e}"
            )

        result = proposal.SolverResult(
            order=tuple(int(v) for v in order),
            score=float(original_score),
            elapsed=float(time.perf_counter() - started),
            expanded_nodes=int(expanded),
            evaluated_edges=int(self.evaluated_edges),
            discovered_nodes=int(discovered),
            max_open_size=int(max_open_size),
        )
        diagnostics = {
            "solver_mode": "repair-free Dijkstra",
            "dp_fallback": 0,
            "negative_edge_observed": 0,
            "finite_sample_global_nonnegativity_certified": False,
            "unvisited_negative_edges_possible": True,
            "transformed_score": float(transformed_score),
            "root_potential": float(root_potential),
            "terminal_potential": float(terminal_potential),
            "telescoping_error": float(telescoping_error),
            "minimum_observed_mi_term": float(self.minimum_mi_term),
            "minimum_observed_bayes_correction": float(
                self.minimum_bayes_correction
            ),
            "minimum_observed_edge": float(self.minimum_edge),
            "maximum_decomposition_error": float(
                self.maximum_decomposition_error
            ),
            "fitted_state_potentials": int(len(self.state_fit_cache)),
            "state_fit_seconds": float(
                sum(self.state_fit_time_by_dimension.values())
            ),
            "state_fit_count_by_dimension": dict(
                sorted(self.state_fit_count_by_dimension.items(), reverse=True)
            ),
        }
        return result, diagnostics


class HybridDijkstraDPFallback:
    """Use repair-free Dijkstra unless an observed negative edge forces DP."""

    def __init__(self, x: np.ndarray, config: proposal.Config):
        self.x = np.asarray(x, dtype=float)
        self.config = config

    def solve(self):
        total_started = time.perf_counter()
        dijkstra = RepairFreeCarriedLikelihoodDijkstra(self.x, self.config)
        try:
            result, diagnostics = dijkstra.solve_no_repair()
            diagnostics = {
                **diagnostics,
                "total_elapsed_seconds": float(
                    time.perf_counter() - total_started
                ),
            }
            # Preserve a single wall-clock number in the standard result.
            result.elapsed = float(time.perf_counter() - total_started)
            return result, diagnostics
        except NegativeEdgeDetected as exc:
            dijkstra_elapsed = float(time.perf_counter() - total_started)
            dp_started = time.perf_counter()
            dp_solver = exact_dag.DirectMarginalDAGShortestPath(
                self.x,
                self.config,
            )
            dp_result, dp_diagnostics = dp_solver.solve()
            dp_elapsed = float(time.perf_counter() - dp_started)
            total_elapsed = float(time.perf_counter() - total_started)

            final_result = proposal.SolverResult(
                order=tuple(int(v) for v in dp_result.order),
                score=float(dp_result.score),
                elapsed=total_elapsed,
                expanded_nodes=int(
                    dijkstra.attempt_expanded + dp_result.expanded_nodes
                ),
                evaluated_edges=int(
                    dijkstra.evaluated_edges + dp_result.evaluated_edges
                ),
                discovered_nodes=int(dp_result.discovered_nodes),
                max_open_size=int(dijkstra.attempt_max_open),
            )
            info = exc.info
            diagnostics = {
                "solver_mode": "exact DP fallback",
                "dp_fallback": 1,
                "negative_edge_observed": 1,
                "negative_kind": info.kind,
                "negative_mask": int(info.mask),
                "negative_depth": int(info.mask.bit_count()),
                "negative_candidate": int(info.candidate),
                "negative_value": float(info.value),
                "negative_mi_term": float(info.mi_term),
                "negative_bayes_correction": float(info.bayes_correction),
                "negative_total_edge": float(info.total_edge),
                "finite_sample_global_nonnegativity_certified": False,
                "unvisited_negative_edges_possible": False,
                "dijkstra_attempt_elapsed_seconds": dijkstra_elapsed,
                "dijkstra_attempt_expanded": int(
                    dijkstra.attempt_expanded
                ),
                "dijkstra_attempt_evaluated_edges": int(
                    dijkstra.evaluated_edges
                ),
                "dijkstra_attempt_discovered": int(
                    dijkstra.attempt_discovered
                ),
                "dijkstra_attempt_max_open": int(
                    dijkstra.attempt_max_open
                ),
                "dijkstra_fitted_state_potentials": int(
                    len(dijkstra.state_fit_cache)
                ),
                "dijkstra_state_fit_seconds": float(
                    sum(dijkstra.state_fit_time_by_dimension.values())
                ),
                "dp_elapsed_seconds": dp_elapsed,
                "total_elapsed_seconds": total_elapsed,
                "dp_diagnostics": dp_diagnostics,
            }
            return final_result, diagnostics
