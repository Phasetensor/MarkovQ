"""Graph search under declared discrete query or analog oracle-time models."""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike

from ..chains import count
from ..passage import passage_distribution
from ..walks import graph_walk as classical_walk
from .measurement import target_projector
from .walks import CoinedWalk, SzegedyWalk, adj, continuous_walk, ket, square, unitary


class SearchResult(NamedTuple):
    candidates: np.ndarray
    success: np.ndarray
    best_index: int
    discrete_queries: np.ndarray
    oracle_time: np.ndarray
    preparations_per_trial: int = 1
    measurements_per_trial: int = 1

    log_success: Optional[np.ndarray] = None

    def cost_status(self, index: int) -> str:
        from ..chains import state

        i = state(index, len(self.candidates))
        p = float(self.success[i])
        if not np.isfinite(p) or p < 0 or p > 1 + 16 * np.finfo(float).eps:
            raise ValueError("invalid probability")
        if self.log_success is not None:
            logged = float(self.log_success[i])
            if (
                np.isnan(logged)
                or logged > 16 * np.finfo(float).eps
                or (p == 0 and np.exp(logged) > 0)
            ):
                raise ValueError("invalid log probability")
        if p == 0:
            return (
                "probability-underflow"
                if self.log_success is not None and np.isfinite(self.log_success[i])
                else "zero-represented"
            )
        return "reciprocal-overflow" if p < 1 / np.finfo(float).max else "finite"

    def expected_cost(self, index: int) -> dict[str, float]:
        status = self.cost_status(index)
        p = min(1.0, float(self.success[index]))
        logp = (
            float(self.log_success[index])
            if status == "probability-underflow" and self.log_success is not None
            else -np.inf
        )

        def cost(per_trial):
            value = float(per_trial)
            if not np.isfinite(value) or value < 0:
                raise ValueError("invalid per-trial resource")
            if value == 0:
                return 0.0
            if p:
                return value / p
            exponent = np.log(value) - logp
            return (
                np.inf
                if exponent > np.log(np.finfo(float).max)
                else float(np.exp(exponent))
            )

        return {
            "independent_trials": cost(1),
            "initial_preparations": cost(self.preparations_per_trial),
            "final_measurements": cost(self.measurements_per_trial),
            "marking_queries": cost(self.discrete_queries[index]),
            "analog_oracle_time": cost(self.oracle_time[index]),
        }

    def resource_status(self,index: int) -> dict[str,str]:
        """Classify each ideal cost and explicitly identify unmeasured physical costs."""
        pstatus=self.cost_status(index)
        result={}
        for name,value in self.expected_cost(index).items():
            if np.isfinite(value):
                result[name]="finite"
            elif pstatus=="zero-represented":
                result[name]="infinite-for-represented-zero"
            else:
                result[name]="finite-overflow"
        result.update(gate_count="unmeasured",wall_time="unmeasured",hardware_cost="unmeasured")
        return result


def _probabilities(amplitudes):
    from scipy.special import logsumexp

    logs = []
    for v in amplitudes:
        magnitudes = abs(v)
        nonzero = magnitudes[magnitudes > 0]
        logs.append(logsumexp(2 * np.log(nonzero)) if nonzero.size else -np.inf)
    return np.exp(logs), np.array(logs)


def discrete_search(
    step: ArrayLike, initial: ArrayLike, marked: Iterable[int], steps: Iterable[int]
) -> SearchResult:
    """One phase-oracle application followed by the supplied unitary per step."""
    u = unitary(step)
    v = ket(initial, len(u))
    p = target_projector(len(u), marked)
    times = np.array([count(t) for t in steps], dtype=int)
    if not times.size:
        raise ValueError("empty candidate set")
    search = u @ (np.eye(len(u)) - 2 * p)
    success, logs = _probabilities(
        p @ np.linalg.matrix_power(search, t) @ v for t in times
    )
    return SearchResult(
        times,
        success,
        int(success.argmax()),
        times.copy(),
        np.zeros(len(times)),
        log_success=logs,
    )


def coined_search(
    walk: CoinedWalk, initial: ArrayLike, marked: Iterable[int], steps: Iterable[int]
) -> SearchResult:
    mask = (
        target_projector(int(walk.position.max()) + 1, marked).diagonal().astype(bool)
    )
    return discrete_search(
        walk.operator, initial, np.flatnonzero(mask[walk.position]), steps
    )


def szegedy_search(
    walk: SzegedyWalk, law: ArrayLike, marked: Iterable[int], steps: Iterable[int]
) -> SearchResult:
    """Phase oracle marks first-register vertices before each two-reflection step."""
    n = walk.A.shape[1]
    mask = target_projector(n, marked).diagonal()
    return discrete_search(
        walk.operator, walk.coherent(law), np.flatnonzero(np.repeat(mask, n)), steps
    )


def amplitude_amplification(
    initial: ArrayLike, marked: Iterable[int], steps: Iterable[int]
) -> SearchResult:
    v = ket(initial, np.asarray(initial).size)
    return discrete_search(2 * np.outer(v, v.conj()) - np.eye(len(v)), v, marked, steps)


def continuous_search(
    adjacency: ArrayLike,
    gamma: float,
    initial: ArrayLike,
    marked: Iterable[int],
    times: ArrayLike,
) -> SearchResult:
    a = square(adjacency)
    if not np.isfinite(gamma) or not np.allclose(a, adj(a), atol=1e-12, rtol=0):
        raise ValueError("requires finite coupling and Hermitian adjacency")
    p = target_projector(len(a), marked)
    t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or t.min() < 0:
        raise ValueError("invalid candidate times")
    h = -gamma * a - p
    prob, logs = _probabilities(
        p @ continuous_walk(h, initial, x, "spectral") for x in t
    )
    return SearchResult(
        t, prob, int(prob.argmax()), np.zeros(len(t), int), t.copy(), log_success=logs
    )


def classical_search(
    adjacency: ArrayLike, initial: ArrayLike, marked: Iterable[int], horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Observed random-walk hitting and independent uniform queries with replacement.

    At step zero each classical protocol has already inspected its initial vertex.
    """
    chain = classical_walk(adjacency)
    marked = tuple(set(marked))
    _, survival = passage_distribution(chain.P, marked, horizon)
    law = chain.evolve(initial, 0)
    return 1 - survival @ law, 1 - (1 - len(marked) / chain.n) ** np.arange(
        1, horizon + 2
    )
