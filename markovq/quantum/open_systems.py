"""Time-independent finite-dimensional GKSL models with column vectorization."""

from __future__ import annotations

from typing import Iterable, NamedTuple

import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import LinearOperator, expm_multiply

from .walks import TOL, _exponential_action, adj, square


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
    hamiltonian: ArrayLike,
    collapse: Iterable[ArrayLike] = (),
    *,
    representation: str = "dense",
):
    if representation in ("sparse", "matrix-free"):
        h, parts = _parts(hamiltonian, collapse)
        return _generator(h, parts, representation == "sparse")
    if representation != "dense":
        raise ValueError("unknown Liouvillian representation")
    h = square(hamiltonian)
    if not np.allclose(h, adj(h), atol=1e-12, rtol=0):
        raise ValueError("Hamiltonian must be Hermitian")
    eye = np.eye(len(h))
    generator = -1j * (np.kron(eye, h) - np.kron(h.T, eye))
    for operator in collapse:
        c = square(operator)
        if c.shape != h.shape:
            raise ValueError("collapse operator dimension mismatch")
        rate = np.einsum("ij,ik->jk", c.conj(), c)
        generator += np.kron(c.conj(), c) - 0.5 * (
            np.kron(eye, rate) + np.kron(rate.T, eye)
        )
    return generator


def _parts(h, collapse):
    h = sparse.csr_matrix(h, dtype=complex)
    if (
        not h.shape[0]
        or h.shape[0] != h.shape[1]
        or not np.isfinite(h.data).all()
        or np.max(abs((h - h.getH()).data), initial=0) > 1e-12
    ):
        raise ValueError("invalid Hamiltonian")
    channels = [sparse.csr_matrix(c, dtype=complex) for c in collapse]
    if any(c.shape != h.shape or not np.isfinite(c.data).all() for c in channels):
        raise ValueError("invalid collapse operator")
    return h, [(c, c.getH(), c.getH() @ c) for c in channels]


def _generator(h, parts, explicit):
    n = h.shape[0]
    if explicit:
        eye = sparse.eye(n)

        def kron(a, b):
            return sparse.kron(a, b, format="csr")

        g = -1j * (kron(eye, h) - kron(h.T, eye))
        for c, cd, r in parts:
            g += kron(c.conj(), c) - 0.5 * (kron(eye, r) + kron(r.T, eye))
        return g

    def apply(vector, dual):
        x = vector.reshape((n, n), order="F")
        y = (1j if dual else -1j) * (h @ x - x @ h)
        for c, cd, r in parts:
            y += (cd @ x @ c if dual else c @ x @ cd) - 0.5 * (r @ x + x @ r)
        return y.ravel(order="F")

    return LinearOperator(
        (n * n, n * n),
        matvec=lambda v: apply(v, False),
        rmatvec=lambda v: apply(v, True),
        dtype=complex,
    )


class OpenEvolution(NamedTuple):
    states: np.ndarray
    populations: np.ndarray
    coherence_l1: np.ndarray
    trace_error: np.ndarray
    hermiticity_error: np.ndarray
    minimum_eigenvalue: np.ndarray

    def violations(self, tol: float = TOL) -> np.ndarray:
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("invalid physicality tolerance")
        return (
            ~np.isfinite(self.states).all(axis=(1, 2))
            | (self.trace_error > tol)
            | (self.hermiticity_error > tol)
            | (self.minimum_eigenvalue < -tol)
        )


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
    h, parts = _parts(hamiltonian, collapse)
    t = np.asarray(times, float)
    if (
        h.shape != rho.shape
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
        g = liouvillian(h.toarray(), [c.toarray() for c, _, _ in parts])
        vectors = np.array([_exponential_action(s * g, x) for s in t])
    elif method in ("sparse", "matrix-free"):
        op = _generator(h, parts, method == "sparse")
        trace = sum(
            abs(c.diagonal().sum()) ** 2 - h.shape[0] * r.diagonal().sum()
            for c, _, r in parts
        )
        vectors = np.array(
            [expm_multiply(s * op, x, traceA=s * trace) if s else x for s in t]
        )
    elif method in ("ode", "bdf"):
        op = _generator(h, parts, method == "bdf")
        if t[-1] == 0:
            vectors = x[None, :]
        else:
            sol = solve_ivp(
                lambda _, v: op @ v,
                (0, t[-1]),
                x,
                t_eval=t,
                rtol=rtol,
                atol=atol,
                method="BDF" if method == "bdf" else "DOP853",
                **({"jac": op} if method == "bdf" else {}),
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
