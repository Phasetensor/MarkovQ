"""Graph search under declared discrete query or analog oracle-time models."""

from __future__ import annotations

from typing import Iterable, NamedTuple

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

    def expected_cost(self, index: int) -> dict[str, float]:
        """Repeat a fixed candidate independently until its marked outcome occurs."""
        from ..chains import state

        i = state(index, len(self.candidates))
        p = float(self.success[i])
        trials = 1 / p if p > 0 else np.inf
        return {
            "initial_preparations": trials,
            "final_measurements": trials,
            "marking_queries": trials * self.discrete_queries[i]
            if self.discrete_queries[i]
            else 0.0,
            "analog_oracle_time": trials * self.oracle_time[i]
            if self.oracle_time[i]
            else 0.0,
        }


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
    success = np.array(
        [np.linalg.norm(p @ np.linalg.matrix_power(search, t) @ v) ** 2 for t in times]
    )
    return SearchResult(
        times, success, int(success.argmax()), times.copy(), np.zeros(len(times))
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
    prob = np.array(
        [np.linalg.norm(p @ continuous_walk(h, initial, x, "spectral")) ** 2 for x in t]
    )
    return SearchResult(t, prob, int(prob.argmax()), np.zeros(len(t), int), t.copy())


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
