"""Finite-dimensional quantum channels and classical readout (true rows)."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from .chains import count, distribution, validate_transition_matrix
from .quantum.open_systems import density_matrix
from .quantum.walks import square, unitary


def kraus_model(kind: str, p: float) -> tuple[np.ndarray, ...]:
    if not 0 <= p <= 1:
        raise ValueError("noise probability must be in [0,1]")
    eye = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    z = np.diag([1, -1])
    y = np.array([[0, -1j], [1j, 0]])
    if kind == "amplitude-damping":
        return np.diag([1, np.sqrt(1 - p)]), np.array([[0, np.sqrt(p)], [0, 0]])
    if kind == "dephasing":
        return (
            np.sqrt(1 - p) * eye,
            np.sqrt(p) * np.diag([1, 0]),
            np.sqrt(p) * np.diag([0, 1]),
        )
    if kind == "depolarizing":
        return (np.sqrt(1 - 3 * p / 4) * eye, *(np.sqrt(p / 4) * a for a in (x, y, z)))
    if kind in ("bit-flip", "phase-flip"):
        return np.sqrt(1 - p) * eye, np.sqrt(p) * (x if kind == "bit-flip" else z)
    raise ValueError("unknown channel")


def validate_kraus(operators: Iterable[ArrayLike]) -> tuple[np.ndarray, ...]:
    ops = tuple(square(a) for a in operators)
    if not ops or any(a.shape != ops[0].shape for a in ops):
        raise ValueError("inconsistent Kraus dimensions")
    gram = sum(np.einsum("ji,jk->ik", a.conj(), a) for a in ops)
    if not np.allclose(gram, np.eye(len(ops[0])), rtol=0, atol=1e-10):
        raise ValueError("incomplete Kraus operators")
    return ops


def _apply(ops, rho):
    return sum(np.einsum("ij,jk,lk->il", a, rho, a.conj()) for a in ops)


def apply_channel(initial: ArrayLike, operators: Iterable[ArrayLike]) -> np.ndarray:
    rho = density_matrix(initial)
    ops = validate_kraus(operators)
    if ops[0].shape != rho.shape:
        raise ValueError("channel dimension mismatch")
    return _apply(ops, rho)


def readout(probabilities: ArrayLike, assignment: ArrayLike) -> np.ndarray:
    a = validate_transition_matrix(assignment)
    return distribution(probabilities, a.shape[0]) @ a


class NoisyWalk(NamedTuple):
    states: np.ndarray
    populations: np.ndarray
    purity: np.ndarray
    coherence: np.ndarray
    trace_error: np.ndarray


def noisy_walk(
    step: ArrayLike, initial: ArrayLike, operators: Iterable[ArrayLike], steps: int
) -> NoisyWalk:
    u = unitary(step)
    rho = density_matrix(initial)
    ops = validate_kraus(operators)
    if rho.shape != u.shape or ops[0].shape != u.shape:
        raise ValueError("walk and channel dimensions differ")
    history = [rho]
    for _ in range(count(steps)):
        rho = _apply(ops, _apply([u], rho))
        history.append(rho)
    states = np.asarray(history)
    diag = np.diagonal(states, axis1=1, axis2=2)
    return NoisyWalk(
        states,
        diag.real,
        np.einsum("tij,tji->t", states, states).real,
        abs(states).sum(axis=(1, 2)) - abs(diag).sum(axis=1),
        abs(1 - diag.sum(axis=1)),
    )
