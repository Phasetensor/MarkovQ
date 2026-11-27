"""Repeated projective detection, quantum renewal and Cesàro mixing."""

from __future__ import annotations

from typing import Iterable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from ..chains import count, state
from .walks import TOL, adj, ket, square, unitary


def target_projector(n: int, targets: Iterable[int]) -> np.ndarray:
    diagonal = np.zeros(count(n))
    diagonal[[state(j, n) for j in targets]] = 1
    return np.diag(diagonal)


def projector(a: ArrayLike, n: int) -> np.ndarray:
    p = square(a)
    if (
        p.shape != (n, n)
        or not np.allclose(p, adj(p), rtol=0, atol=TOL)
        or not np.allclose(p @ p, p, rtol=0, atol=TOL)
    ):
        raise ValueError("invalid orthogonal projector")
    return p


class Detection(NamedTuple):
    first: np.ndarray
    survival: np.ndarray
    surviving_ket: np.ndarray

    def conditional_state(self) -> np.ndarray:
        norm = np.linalg.norm(self.surviving_ket)
        if norm == 0:
            raise ValueError("zero survival")
        return self.surviving_ket / norm


def first_detection(
    step: ArrayLike, initial: ArrayLike, target: ArrayLike, steps: int
) -> Detection:
    u = unitary(step)
    p = projector(target, len(u))
    v = ket(initial, len(u))
    first = np.zeros(count(steps) + 1)
    survival = np.ones_like(first)
    for t in range(1, len(first)):
        propagated = u @ v
        hit = p @ propagated
        v = propagated - hit
        first[t] = np.vdot(hit, hit).real
        survival[t] = np.vdot(v, v).real
    return Detection(first, survival, v)


def phase_spaces(step: ArrayLike, tol: float = TOL) -> list[np.ndarray]:
    """Orthonormal bases grouped by chord distance on the unit circle."""
    u = unitary(step)
    if not np.isfinite(tol) or not 0 <= tol < 1:
        raise ValueError("invalid eigenphase tolerance")
    t, b = linalg.schur(u, output="complex")
    values = t.diagonal()
    pending = np.ones(len(u), bool)
    bases = []
    for i, z in enumerate(values):
        if pending[i]:
            group = pending & (abs(values - z) <= tol)
            bases.append(b[:, group])
            pending[group] = False
    return bases


class EventualDetection(NamedTuple):
    probability: float
    conditional_mean: float
    dark_projector: np.ndarray


def eventual_detection(
    step: ArrayLike, initial: ArrayLike, target: ArrayLike, tol: float = TOL
) -> EventualDetection:
    """Finite-dimensional bright restriction; conditional mean measured in attempts."""
    u = unitary(step)
    v = ket(initial, len(u))
    p = projector(target, len(u))
    blocks = []
    for basis in phase_spaces(u, tol):
        _, s, vh = linalg.svd(p @ basis, full_matrices=False)
        blocks.append(basis @ adj(vh[s > tol]))
    bright = np.column_stack(blocks)
    dark = np.eye(len(u)) - bright @ adj(bright)
    vb = adj(bright) @ v
    prob = float(np.vdot(vb, vb).real)
    if prob <= np.finfo(float).eps ** 2:
        return EventualDetection(prob, np.nan, dark)
    contraction = adj(bright) @ (np.eye(len(u)) - p) @ u @ bright
    duration = linalg.solve_discrete_lyapunov(adj(contraction), np.eye(len(vb)))
    return EventualDetection(prob, float(np.vdot(vb, duration @ vb).real / prob), dark)


class Recurrence(NamedTuple):
    amplitudes: np.ndarray
    occupancy: np.ndarray
    first_amplitudes: np.ndarray
    monitored: Detection


def recurrence(step: ArrayLike, initial: ArrayLike, steps: int) -> Recurrence:
    u = unitary(step)
    v = ket(initial, len(u))
    current = v
    amp = np.ones(count(steps) + 1, complex)
    first = np.zeros_like(amp)
    for t in range(1, len(amp)):
        current = u @ current
        amp[t] = np.vdot(v, current)
        first[t] = amp[t] - first[1:t] @ amp[t - 1 : 0 : -1]
    return Recurrence(
        amp, abs(amp) ** 2, first, first_detection(u, v, np.outer(v, v.conj()), steps)
    )


class Mixing(NamedTuple):
    instantaneous: np.ndarray
    average: np.ndarray
    limiting_average: np.ndarray
    tv_to_limit: float


def quantum_mixing(
    step: ArrayLike, initial: ArrayLike, horizon: int, positions: Optional[ArrayLike] = None
) -> Mixing:
    u = unitary(step)
    v = ket(initial, len(u))
    horizon = count(horizon)
    labels = np.arange(len(u)) if positions is None else np.asarray(positions)
    if (
        not horizon
        or labels.shape != (len(u),)
        or labels.dtype.kind not in "iu"
        or labels.min() < 0
    ):
        raise ValueError(
            "requires positive horizon and nonnegative integer position labels"
        )

    def marginal(a):
        return np.bincount(labels, abs(a) ** 2)

    limit = sum(
        (marginal(b @ (adj(b) @ v)) for b in phase_spaces(u)),
        start=np.zeros(labels.max() + 1),
    )
    history = []
    for _ in range(horizon):
        history.append(marginal(v))
        v = u @ v
    avg = np.mean(history, axis=0)
    return Mixing(np.array(history), avg, limit, float(abs(avg - limit).sum() / 2))


def dephased_mixing(step: ArrayLike,initial: ArrayLike,horizon: int,strength: float,positions: Optional[ArrayLike]=None) -> np.ndarray:
    """After each unitary step, dephase between distinct position eigenspaces."""
    u=unitary(step)
    v=ket(initial,len(u))
    n=count(horizon)
    labels=np.arange(len(u)) if positions is None else np.asarray(positions)
    if not n or not 0<=strength<=1 or labels.shape!=(len(u),) or labels.dtype.kind not in "iu" or labels.min()<0:
        raise ValueError("invalid horizon, dephasing strength or position labels")
    rho=np.outer(v,v.conj())
    attenuation=np.where(labels[:,None]==labels[None,:],1.,1.-strength)
    result=[]
    for _ in range(n):
        result.append(np.bincount(labels,weights=rho.diagonal().real))
        rho=(u@rho@adj(u))*attenuation
    return np.asarray(result)
