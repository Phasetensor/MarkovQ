"""Finite CTMCs with explicit rate units and event-driven paths."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm
from scipy.stats import poisson

from ._types import MatrixInput, Seed
from .chains import matrix, state
from .passage import hitting


def time_value(t: float) -> float:
    t = float(t)
    if not np.isfinite(t) or t < 0:
        raise ValueError("invalid elapsed time")
    return t


class ContinuousChain:
    def __init__(self, generator: MatrixInput, unit: str) -> None:
        q = matrix(generator)
        self.Q = q if isinstance(q, np.ndarray) else q.toarray()
        off = self.Q.copy()
        np.fill_diagonal(off, 0)
        if (
            off.min() < 0
            or not np.allclose(self.Q.sum(axis=1), 0, rtol=0, atol=1e-12)
            or not isinstance(unit, str)
            or not unit.strip()
        ):
            raise ValueError("invalid rates or missing time unit")
        self.unit = unit

    def transition(self, t: float) -> np.ndarray:
        return expm(time_value(t) * self.Q)

    def derivative(self, t: float) -> np.ndarray:
        return self.transition(t) @ self.Q

    def jump_chain(self) -> np.ndarray:
        out = self.Q.copy()
        for i, rate in enumerate(-self.Q.diagonal()):
            if rate > 0:
                out[i] /= rate
                out[i, i] = 0
            else:
                out[i, i] = 1
        return out

    def uniformize(self, t: float, tol: float = 1e-12) -> np.ndarray:
        t = time_value(t)
        if not 0 < tol < 1:
            raise ValueError("invalid tail tolerance")
        rate = max(-self.Q.diagonal())
        identity = np.eye(len(self.Q))
        if rate * t == 0:
            return identity
        step = identity + self.Q / rate
        power = identity
        out = np.zeros_like(step)
        for k in range(int(poisson.isf(tol, rate * t)) + 1):
            out += poisson.pmf(k, rate * t) * power
            power = power @ step
        return out

    def events(
        self, initial: int, horizon: float, seed: Seed = None
    ) -> tuple[np.ndarray, np.ndarray]:
        i = state(initial, len(self.Q))
        horizon = time_value(horizon)
        rng = np.random.default_rng(seed)
        jump = self.jump_chain()
        times, states = [0.0], [i]
        while self.Q[i, i] < 0:
            t = times[-1] + rng.exponential(-1 / self.Q[i, i])
            if t > horizon:
                break
            i = rng.choice(len(jump), p=jump[i])
            times.append(t)
            states.append(i)
        return np.array(times), np.array(states)

    def absorption_time(self, targets: Iterable[int]) -> np.ndarray:
        targets = tuple(targets)
        result = hitting(self.jump_chain(), targets)
        unknown = np.isfinite(result.mean)
        unknown[list(targets)] = False
        mean = np.where(unknown, 0.0, result.mean)
        mean[unknown] = np.linalg.solve(
            -self.Q[np.ix_(unknown, unknown)], np.ones(unknown.sum())
        )
        return mean


def birth_death(birth: ArrayLike, death: ArrayLike, unit: str) -> ContinuousChain:
    birth, death = np.asarray(birth, float), np.asarray(death, float)
    if birth.ndim != 1 or birth.shape != death.shape:
        raise ValueError("supply one birth and one death rate per edge")
    q = np.diag(birth, 1) + np.diag(death, -1)
    return ContinuousChain(q - np.diag(q.sum(axis=1)), unit)
