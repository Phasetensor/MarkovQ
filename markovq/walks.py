"""Nearest-neighbour walks, lattice benchmarks and empirical path statistics."""
from __future__ import annotations
from typing import Iterable
from numpy.typing import ArrayLike
from ._types import MatrixInput, Seed


import numpy as np
from scipy import sparse
from .chains import MarkovChain, count, matrix


def graph_walk(adjacency: MatrixInput, lazy: float=0.0) -> MarkovChain:
    a = matrix(adjacency)
    degrees = np.asarray(a.sum(axis=1)).ravel()
    if a.min() < 0 or (degrees <= 0).any() or not 0 <= lazy <= 1:
        raise ValueError(
            "expected nonnegative weights, positive degrees and valid laziness"
        )
    if sparse.issparse(a):
        return MarkovChain(
            (1 - lazy) * sparse.diags(1 / degrees) @ a + lazy * sparse.eye(len(degrees))
        )
    return MarkovChain((1 - lazy) * a / degrees[:, None] + lazy * np.eye(len(a)))


def line_walk(size: int, right: float=0.5, boundary: str="reflect", lazy: float=0.0) -> MarkovChain:
    size = count(size)
    if (
        size < 2
        or not 0 <= right <= 1
        or not 0 <= lazy <= 1
        or boundary not in {"reflect", "absorb", "cycle"}
    ):
        raise ValueError("invalid finite walk")
    p = np.zeros((size, size))
    for i in range(size):
        for j, weight in ((i + 1, right), (i - 1, 1 - right)):
            if boundary == "absorb" and i in (0, size - 1):
                j = i
            j = j % size if boundary == "cycle" else min(max(j, 0), size - 1)
            p[i, j] += weight
    return MarkovChain((1 - lazy) * p + lazy * np.eye(size))


def lattice_paths(dim: int, steps: int, paths: int, seed: Seed=None, right: float=0.5) -> np.ndarray:
    dim, steps, paths = count(dim), count(steps), count(paths)
    if not dim or not 0 <= right <= 1:
        raise ValueError("invalid lattice walk")
    rng = np.random.default_rng(seed)
    out = np.zeros((paths, steps + 1, dim), dtype=int)
    axis = rng.integers(dim, size=(paths, steps, 1))
    signs = np.where(rng.random((paths, steps, 1)) < right, 1, -1)
    np.put_along_axis(out[:, 1:], axis, signs, axis=2)
    return out.cumsum(axis=1)


def path_statistics(paths: ArrayLike) -> dict[str, np.ndarray]:
    x = np.asarray(paths)
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError("expected trajectories by time by coordinate")
    delta = x - x[:, 0:1]
    at_origin = (delta == 0).all(axis=2)
    return dict(
        mean=x.mean(axis=0),
        variance=x.var(axis=0),
        msd=(delta**2).sum(axis=2).mean(axis=0),
        return_probability=at_origin.mean(axis=0),
        origin_visits=at_origin.sum(axis=1),
    )


def lattice_recurrence(dimension: int, right: float=0.5) -> str:
    dimension = count(dimension)
    if dimension == 0 or not 0 <= right <= 1 or (dimension > 1 and right != 0.5):
        raise ValueError("supports biased line or symmetric lattice")
    return "null recurrent" if right == 0.5 and dimension <= 2 else "transient"


def line_return(steps: int, right: float=0.5) -> float:
    from scipy.stats import binom

    steps = count(steps)
    if not 0 <= right <= 1:
        raise ValueError("invalid bias")
    return float(binom.pmf(steps // 2, steps, right)) if steps % 2 == 0 else 0.0


def eventual_line_return(right: float=0.5) -> float:
    if not 0 <= right <= 1:
        raise ValueError("invalid bias")
    return 2 * min(right, 1 - right)


def grid_walk(rows: int, columns: int, lazy: float=0.0) -> MarkovChain:
    rows, columns = count(rows), count(columns)
    if min(rows, columns) < 1 or rows * columns == 1:
        raise ValueError("grid requires at least two vertices")

    def edges(n):
        return sparse.diags([np.ones(n - 1), np.ones(n - 1)], [-1, 1], shape=(n, n))

    return graph_walk(sparse.kronsum(edges(columns), edges(rows)), lazy)


def empirical_passage(paths: ArrayLike, targets: Iterable[int], first_return: bool=False) -> tuple[np.ndarray, np.ndarray]:
    """Return observed hitting steps and censoring indicators for state-index paths."""
    x = np.asarray(paths)
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError("expected a path matrix with an initial observation")
    hit = np.isin(x, list(targets))
    if first_return:
        hit[:, 0] = False
    observed = hit.any(axis=1)
    return np.where(observed, hit.argmax(axis=1), x.shape[1] - 1), ~observed


def occupation_times(paths: ArrayLike, states: Iterable[int]) -> np.ndarray:
    x = np.asarray(paths)
    if x.ndim != 2:
        raise ValueError("expected a path matrix")
    return np.column_stack([(x == i).sum(axis=1) for i in states])


def increment_walk(increments: ArrayLike, probabilities: ArrayLike, steps: int, paths: int=1, seed: Seed=None, initial: float=0.0) -> np.ndarray:
    """Sample real iid increments from a finite distribution on the line."""
    from .chains import distribution

    support = np.asarray(increments, dtype=float)
    if (
        support.ndim != 1
        or not support.size
        or not np.isfinite(support).all()
        or not np.isfinite(initial)
    ):
        raise ValueError("expected finite real increments and origin")
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        support,
        (count(paths), count(steps)),
        p=distribution(probabilities, len(support)),
    )
    return initial + np.column_stack((np.zeros(paths), draws.cumsum(axis=1)))
