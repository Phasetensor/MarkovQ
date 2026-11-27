"""Time-independent finite-dimensional GKSL models with column vectorization."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .walks import TOL, adj, square


def density_matrix(matrix: ArrayLike) -> np.ndarray:
    rho = square(matrix)
    if (
        not np.allclose(rho, adj(rho), atol=TOL, rtol=0)
        or abs(np.trace(rho) - 1) > TOL
        or np.linalg.eigvalsh(rho).min() < -TOL
    ):
        raise ValueError("expected a Hermitian positive unit-trace density matrix")
    return rho


def liouvillian(
    hamiltonian: ArrayLike, collapse: Iterable[ArrayLike] = ()
) -> np.ndarray:
    h = square(hamiltonian)
    if not np.allclose(h, adj(h), atol=1e-12, rtol=0):
        raise ValueError("Hamiltonian must be Hermitian")
    eye = np.eye(len(h))
    generator = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for operator in collapse:
        c = square(operator)
        if c.shape != h.shape:
            raise ValueError("collapse operator dimension mismatch")
        rate = adj(c) @ c
        generator += np.kron(c.conj(), c) - 0.5 * (
            np.kron(eye, rate) + np.kron(rate.T, eye)
        )
    return generator


class OpenEvolution(NamedTuple):
    states: np.ndarray
    populations: np.ndarray
    coherence_l1: np.ndarray
    trace_error: np.ndarray
    hermiticity_error: np.ndarray
    minimum_eigenvalue: np.ndarray


def evolve_density(
    hamiltonian: ArrayLike,
    initial: ArrayLike,
    times: ArrayLike,
    collapse: Iterable[ArrayLike] = (),
    method: str = "expm",
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> OpenEvolution:
    rho = density_matrix(initial)
    g = liouvillian(hamiltonian, collapse)
    t = np.asarray(times, float)
    if (
        g.shape[0] != rho.size
        or t.ndim != 1
        or not t.size
        or not np.isfinite(t).all()
        or t[0] < 0
        or (np.diff(t) <= 0).any()
    ):
        raise ValueError("invalid times or state dimension")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0:
        raise ValueError("invalid solver tolerances")
    x = rho.flatten(order="F")
    if method == "expm":
        vectors = np.array([expm(s * g) @ x for s in t])
    elif method == "ode":
        if t[-1] == 0:
            vectors = x[None, :]
        else:
            sol = solve_ivp(
                lambda _, v: g @ v,
                (0, t[-1]),
                x,
                t_eval=t,
                rtol=rtol,
                atol=atol,
                method="DOP853",
            )
            if not sol.success:
                raise RuntimeError(sol.message)
            vectors = sol.y.T
    else:
        raise ValueError("unknown solver")
    states = np.array([v.reshape(rho.shape, order="F") for v in vectors])
    diag = np.diagonal(states, axis1=1, axis2=2)
    dual = states.swapaxes(1, 2).conj()
    return OpenEvolution(
        states,
        diag.real,
        abs(states).sum(axis=(1, 2)) - abs(diag).sum(axis=1),
        abs(diag.sum(axis=1) - 1),
        np.linalg.norm(states - dual, axis=(1, 2)),
        np.linalg.eigvalsh((states + dual) / 2).min(axis=1),
    )


def stochastic_walk(
    adjacency: ArrayLike, rates: ArrayLike, omega: float
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Independent local jumps |j><i| with rates indexed [i,j]."""
    h = square(adjacency)
    r = np.asarray(rates)
    if (
        not np.allclose(h, adj(h), rtol=0, atol=TOL)
        or r.shape != h.shape
        or not np.isrealobj(r)
        or not np.isfinite(r).all()
        or (r < 0).any()
        or not 0 <= omega <= 1
    ):
        raise ValueError("invalid Hamiltonian, jump rates or interpolation")
    channels = []
    for i, j in zip(*np.nonzero(r)):
        c = np.zeros_like(h)
        c[j, i] = np.sqrt(omega * r[i, j])
        channels.append(c)
    return (1 - omega) * h, channels
