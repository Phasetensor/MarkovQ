"""Finite chains use row distributions and chronological right multiplication."""
from __future__ import annotations
from typing import Optional, Iterable
from numpy.typing import ArrayLike
from ._types import MatrixInput, Matrix, Seed


from math import gcd
from operator import index
from typing import NamedTuple

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def count(n: int) -> int:
    n = index(n)
    if n < 0:
        raise ValueError("expected a nonnegative integer")
    return n


def state(i: int, size: int) -> int:
    i = count(i)
    if i >= size:
        raise ValueError("state outside model")
    return i


def matrix(a: MatrixInput) -> Matrix:
    if np.iscomplexobj(a):
        raise ValueError("expected real entries")
    a = (
        sparse.csr_matrix(a, dtype=float, copy=True)
        if sparse.issparse(a)
        else np.array(a, dtype=float, copy=True)
    )
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.shape[0]:
        raise ValueError("expected a nonempty square matrix")
    values = a.data if sparse.issparse(a) else a
    if not np.isfinite(values).all():
        raise ValueError("nonfinite matrix")
    return a


def validate_transition_matrix(a: MatrixInput) -> Matrix:
    a = matrix(a)
    if a.min() < 0 or not np.allclose(a.sum(axis=1), 1, atol=1e-12, rtol=0):
        raise ValueError("invalid transition probabilities")
    return a


def distribution(p: ArrayLike, size: int) -> np.ndarray:
    if np.iscomplexobj(p):
        raise ValueError("complex probabilities")
    p = np.asarray(p, dtype=float)
    if (
        p.shape != (size,)
        or p.min() < 0
        or not np.isclose(p.sum(), 1, rtol=0, atol=1e-12)
    ):
        raise ValueError("invalid distribution")
    return p


class MarkovChain:
    def __init__(self, transition: MatrixInput, *, schedule: Optional[Iterable[MatrixInput]]=None) -> None:
        self.P = validate_transition_matrix(transition)
        self.n = self.P.shape[0]
        self.steps = (
            None
            if schedule is None
            else [self.P] + [validate_transition_matrix(p) for p in schedule]
        )
        if self.steps and any(p.shape != self.P.shape for p in self.steps):
            raise ValueError("inconsistent state spaces")

    def _interval(self, steps: int, start: int) -> range:
        steps, start = count(steps), count(start)
        if self.steps is not None and start + steps > len(self.steps):
            raise ValueError("schedule exhausted")
        return range(start, start + steps)

    def n_step_matrix(self, steps: int, start: int=0) -> Matrix:
        interval = self._interval(steps, start)
        if self.steps is None and steps:
            return (
                self.P**steps
                if sparse.issparse(self.P)
                else np.linalg.matrix_power(self.P, steps)
            )
        out = (
            sparse.eye(self.n, format="csr")
            if sparse.issparse(self.P)
            else np.eye(self.n)
        )
        for t in interval:
            out = out @ self.steps[t]
        return out

    def transition_probability(self, i: int, j: int, steps: int=1, start: int=0) -> float:
        return float(
            self.n_step_matrix(steps, start)[state(i, self.n), state(j, self.n)]
        )

    def evolve(self, initial: ArrayLike, steps: int, start: int=0) -> np.ndarray:
        return distribution(initial, self.n) @ self.n_step_matrix(steps, start)

    def sample_paths(self, initial: ArrayLike, steps: int, paths: int=1, seed: Seed=None, start: int=0) -> np.ndarray:
        interval = self._interval(steps, start)
        rng = np.random.default_rng(seed)
        out = np.empty((count(paths), steps + 1), dtype=int)
        out[:, 0] = rng.choice(self.n, len(out), p=distribution(initial, self.n))
        for col, t in enumerate(interval, 1):
            p = self.P if self.steps is None else self.steps[t]
            for i in np.unique(out[:, col - 1]):
                mask = out[:, col - 1] == i
                row = p.getrow(i).toarray()[0] if sparse.issparse(p) else p[i]
                out[mask, col] = rng.choice(self.n, mask.sum(), p=row)
        return out


def classify_states(transition: MatrixInput) -> list[StateClass]:
    p = sparse.csr_matrix(validate_transition_matrix(transition))
    p.eliminate_zeros()
    k, labels = connected_components(p, directed=True, connection="strong")
    result = []
    for label in range(k):
        group = np.flatnonzero(labels == label)
        closed = all(labels[j] == label for j in p[group].indices)
        depth = {int(group[0]): 0}
        queue = [int(group[0])]
        period = 0
        for i in queue:
            for j in p.indices[p.indptr[i] : p.indptr[i + 1]]:
                if labels[j] != label:
                    continue
                if j not in depth:
                    depth[j] = depth[i] + 1
                    queue.append(j)
                period = gcd(period, depth[i] + 1 - depth[j])
        result.append(StateClass(tuple(map(int, group)), closed, period))
    return result


class StateClass(NamedTuple):
    states: tuple
    recurrent: bool
    period: int

    @property
    def absorbing(self) -> bool:
        return self.recurrent and len(self.states) == 1
