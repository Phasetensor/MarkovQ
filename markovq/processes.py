"""Finite-activity Lévy models with Gaussian or deterministic jump sizes."""
from __future__ import annotations
from typing import Union
from numpy.typing import ArrayLike
from ._types import Seed


from dataclasses import dataclass
import numpy as np
from .chains import count
from .continuous import time_value


@dataclass(frozen=True)
class LevyProcess:
    drift: float = 0.0
    volatility: float = 0.0
    intensity: float = 0.0
    jump_mean: float = 1.0
    jump_sd: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.drift,
            self.volatility,
            self.intensity,
            self.jump_mean,
            self.jump_sd,
        )
        if (
            not np.isfinite(values).all()
            or min(self.volatility, self.intensity, self.jump_sd) < 0
        ):
            raise ValueError("invalid process parameters")

    def characteristic(self, u: ArrayLike, t: float) -> Union[np.ndarray, complex]:
        u = np.asarray(u, float)
        jump = np.exp(1j * u * self.jump_mean - 0.5 * (u * self.jump_sd) ** 2)
        return np.exp(
            time_value(t)
            * (
                1j * u * self.drift
                - 0.5 * (u * self.volatility) ** 2
                + self.intensity * (jump - 1)
            )
        )

    def moments(self, t: float) -> tuple[float, float]:
        t = time_value(t)
        return (
            t * (self.drift + self.intensity * self.jump_mean),
            t
            * (
                self.volatility**2
                + self.intensity * (self.jump_mean**2 + self.jump_sd**2)
            ),
        )

    def sample(self, times: ArrayLike, paths: int, seed: Seed=None) -> tuple[np.ndarray, np.ndarray]:
        times = np.asarray(times, float)
        if (
            times.ndim != 1
            or not times.size
            or times[0] != 0
            or not np.isfinite(times).all()
            or (np.diff(times) < 0).any()
        ):
            raise ValueError("expected ordered finite times starting at zero")
        dt = np.diff(times)
        rng = np.random.default_rng(seed)
        jumps = rng.poisson(self.intensity * dt, size=(count(paths), len(dt)))
        mean = self.drift * dt + jumps * self.jump_mean
        variance = self.volatility**2 * dt + jumps * self.jump_sd**2
        increments = rng.normal(mean, np.sqrt(variance))
        return np.column_stack((np.zeros(paths), increments.cumsum(axis=1))), jumps
