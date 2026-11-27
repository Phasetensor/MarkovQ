"""Compound Poisson sampling for user-supplied real-valued jump laws."""

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from ._types import Seed
from .chains import count

JumpSampler = Callable[[np.random.Generator, int], ArrayLike]


def compound_poisson(
    times: ArrayLike,
    paths: int,
    intensity: float,
    sample_jumps: JumpSampler,
    seed: Seed = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paths and counts per interval. The supplied sampler draws iid jumps."""
    grid = np.asarray(times, dtype=float)
    if grid.ndim != 1 or not grid.size or grid[0] != 0 or not np.isfinite(grid).all():
        raise ValueError("expected a finite time grid beginning at zero")
    dt = np.diff(grid)
    if (dt < 0).any() or not np.isfinite(intensity) or intensity < 0:
        raise ValueError("expected ordered times and a nonnegative finite intensity")
    rng = np.random.default_rng(seed)
    number = rng.poisson(intensity * dt, size=(count(paths), len(dt)))
    increments = np.zeros(number.shape)
    for j in range(len(dt)):
        counts = number[:, j]
        total = int(counts.sum())
        if total:
            sizes = np.asarray(sample_jumps(rng, total))
            if (
                sizes.shape != (total,)
                or np.iscomplexobj(sizes)
                or not np.isfinite(sizes).all()
            ):
                raise ValueError("sampler must return one finite real value per jump")
            owners = np.repeat(np.arange(paths), counts)
            increments[:, j] = np.bincount(owners, weights=sizes, minlength=paths)
    return np.column_stack((np.zeros(paths), increments.cumsum(axis=1))), number
