"""Repeated projective detection, quantum renewal and Cesàro mixing."""

from __future__ import annotations

import warnings
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
        or not np.allclose(np.einsum("ij,jk->ik", p, p), p, rtol=0, atol=TOL)
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
        propagated = np.einsum("ij,j->i", u, v)
        hit = np.einsum("ij,j->i", p, propagated)
        v = propagated - hit
        first[t] = np.vdot(hit, hit).real
        survival[t] = np.vdot(v, v).real
    return Detection(first, survival, v)


def _threshold(value: float) -> float:
    if not np.isfinite(value) or not 0 <= value < 1:
        raise ValueError("invalid numerical threshold")
    return value


def _groups(values: np.ndarray, tol: float) -> list[np.ndarray]:
    """Connected angular components, including the circular wraparound edge."""
    order = np.argsort(np.angle(values), kind="stable")
    gaps = abs(values[order] - values[np.roll(order, 1)])
    cuts = np.flatnonzero(gaps > tol)
    if not len(cuts):
        return [order]
    ring = np.roll(order, -cuts[0])
    return list(np.split(ring, np.flatnonzero(abs(np.diff(values[ring])) > tol) + 1))


def phase_spaces(step: ArrayLike, tol: float = TOL) -> list[np.ndarray]:
    u = unitary(step)
    t, b = linalg.schur(u, output="complex")
    return [b[:, g] for g in _groups(t.diagonal(), _threshold(tol))]


class DetectionDiagnostics(NamedTuple):
    status: str
    mean_status: str
    unconditional_mean: float
    phase_tol: float
    rank_tol: float
    dark_invariance_residual: float
    dark_target_residual: float
    probability_range: tuple[float, float]
    spectral_radius: float
    condition_bound: float
    relative_residual: float
    hermiticity_error: float
    minimum_eigenvalue: float
    unconditional_status: str
    dark_amplitude: float
    dark_amplitude_error: float


class EventualDetection(NamedTuple):
    probability: float
    conditional_mean: float
    dark_projector: np.ndarray
    diagnostics: DetectionDiagnostics


def eventual_detection(
    step: ArrayLike,
    initial: ArrayLike,
    target: ArrayLike,
    tol: float = TOL,
    *,
    phase_tol: Optional[float] = None,
    rank_tol: Optional[float] = None,
    solver_tol: float = 1e-7,
) -> EventualDetection:
    u = unitary(step)
    v = ket(initial, len(u))
    p = projector(target, len(u))
    _threshold(tol)
    pt = _threshold(tol if phase_tol is None else phase_tol)
    rt = _threshold(tol if rank_tol is None else rank_tol)
    if not 0 < solver_tol < 1:
        raise ValueError("invalid solver tolerance")
    schur, b = linalg.schur(u, output="complex")
    eps = float(np.finfo(float).eps)
    floor = 8 * len(u) * eps

    def split(phase, rank):
        blocks = []
        for group in _groups(schur.diagonal(), phase):
            basis = b[:, group]
            _, s, vh = linalg.svd(p @ basis, full_matrices=False)
            blocks.append(basis @ adj(vh[s > rank]))
        return np.column_stack(blocks)

    # Keep individually resolved phases and couplings in the central model.
    bright = split(min(pt, floor), min(rt, floor))
    alternatives = [
        split(pt, rt),
        split(max(floor, pt * 4), max(floor, rt * 4)),
        split(floor, floor),
        split(floor / 8, floor / 8),
    ]
    vb = adj(bright) @ v
    amplitude = float(linalg.norm(vb))
    prob = amplitude**2
    estimates = [prob] + [float(np.linalg.norm(adj(a) @ v) ** 2) for a in alternatives]
    dark = np.eye(len(u)) - bright @ adj(bright)
    inv = float(linalg.norm(u @ dark - dark @ u, 2))
    coupling = float(linalg.norm(p @ dark, 2))
    ambiguous = bool(
        any(x.shape != bright.shape for x in alternatives)
        or max(estimates) - min(estimates) > 1e-8
        or max(inv, coupling) > floor * 8
    )
    # A tiny absolute coupling may still exceed its componentwise error.
    # Use a posteriori eigenvector residual / spectral separation for isolated modes.
    z = schur.diagonal()
    for g in _groups(z, min(pt, floor)):
        if len(g) != 1:
            continue
        k = int(g[0])
        mode = b[:, k]
        strength = float(linalg.norm(p @ mode))
        if 0 < strength <= min(rt, floor):
            gap = float(np.min(abs(np.delete(z, k) - z[k]), initial=2.0))
            product_error = floor * linalg.norm(abs(p) @ abs(mode))
            rotation_error = linalg.norm(u @ mode - z[k] * mode) / gap
            if strength > 4 * (product_error + rotation_error):
                ambiguous = True
    status = "tolerance-sensitive" if ambiguous else "resolved-at-precision"
    mean = np.nan
    radius = condition = residual = hermiticity = minimum = np.nan
    mean_status = "undefined" if amplitude == 0 else "unresolved"
    projection_error = floor * linalg.norm(abs(bright).T @ abs(v))
    if not ambiguous and amplitude > projection_error:
        transformed = linalg.blas.zgemm(1.0, u, bright)
        survivor = transformed - linalg.blas.zgemm(1.0, p, transformed)
        c = linalg.blas.zgemm(1.0, bright, survivor, trans_a=2)
        radius = float(np.max(abs(linalg.eigvals(c))))
        if radius < 1 - 4 * eps:
            try:
                eye = np.eye(len(vb))
                with warnings.catch_warnings():
                    warnings.simplefilter("error", linalg.LinAlgWarning)
                    duration = linalg.solve_discrete_lyapunov(
                        adj(c), eye, method="direct" if len(c) <= 8 else "bilinear"
                    )
                norm = float(linalg.norm(duration, 2))
                condition = (1 + float(linalg.norm(c, 2)) ** 2) * norm
                residual = float(
                    linalg.norm(duration - adj(c) @ duration @ c - eye, 2)
                    / (1 + 2 * norm)
                )
                hermiticity = float(
                    linalg.norm(duration - adj(duration), 2) / max(1, norm)
                )
                minimum = float(linalg.eigvalsh((duration + adj(duration)) / 2)[0])
                normalized = vb / amplitude
                estimate = float(np.vdot(normalized, duration @ normalized).real)
                if (
                    np.isfinite(estimate)
                    and estimate >= 1 - solver_tol
                    and minimum >= 1 - solver_tol
                    and max(condition * max(float(eps), float(residual)), hermiticity)
                    < solver_tol
                ):
                    mean = estimate
                    mean_status = "finite"
            except (ValueError, np.linalg.LinAlgError, linalg.LinAlgWarning):
                mean_status = "unresolved"
    if ambiguous:
        prob = mean = np.nan
        mean_status = "unresolved"
    # Coordinate states outside backward reachability form an exact invariant
    # dark block for the represented input. Its mass need never be squared.
    reachable = np.any(p != 0, axis=0)
    while True:
        previous = reachable
        reachable = previous | np.any(u[previous] != 0, axis=0)
        if np.array_equal(previous, reachable):
            break
    dark_amp = float(linalg.norm(dark @ v))
    variation = max(
        float(linalg.norm((dark - (np.eye(len(u)) - a @ adj(a))) @ v))
        for a in alternatives
    )
    error = (
        variation
        + inv
        + coupling
        + floor * (1 + float(linalg.norm(abs(dark) @ abs(v))))
    )
    unconditional = np.nan
    unconditional_status = "unresolved"
    if np.any(v[~reachable] != 0):
        unconditional, unconditional_status = np.inf, "infinite"
    elif not ambiguous:
        if bright.shape[1] == int(reachable.sum()):
            unconditional, unconditional_status = mean, mean_status
        elif dark_amp > error:
            unconditional, unconditional_status = np.inf, "infinite"
    diag = DetectionDiagnostics(
        status,
        mean_status,
        unconditional,
        pt,
        rt,
        inv,
        coupling,
        (min(estimates), max(estimates)),
        radius,
        condition,
        residual,
        hermiticity,
        minimum,
        unconditional_status,
        dark_amp,
        error,
    )
    return EventualDetection(prob, mean, dark, diag)


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
        current = np.einsum("ij,j->i", u, current)
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
    finite_spectral: np.ndarray
    limit_sensitivity_tv: float
    spectral_status: str
    unresolved_splitting_tv: float


def quantum_mixing(
    step: ArrayLike,
    initial: ArrayLike,
    horizon: int,
    positions: Optional[ArrayLike] = None,
    *,
    phase_tol: float = TOL,
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

    schur, b = linalg.schur(u, output="complex")
    z = schur.diagonal()
    floor = 8 * len(u) * np.finfo(float).eps
    components = b * np.einsum("ij,i->j", b.conj(), v)

    def limiting(tolerance):
        return sum(
            (marginal(components[:, g].sum(axis=1)) for g in _groups(z, tolerance)),
            start=np.zeros(labels.max() + 1),
        )

    tol = _threshold(phase_tol)
    limit = limiting(min(tol, floor))
    sensitivity = float(abs(limit - limiting(max(tol, floor))).sum() / 2)
    angle = np.angle(z[:, None] * z.conj()[None, :])
    scale = angle / (2 * np.pi)
    kernel = (
        np.exp(0.5j * (horizon - 1) * angle) * np.sinc(horizon * scale) / np.sinc(scale)
    )
    spectral = np.bincount(
        labels, np.einsum("ij,jk,ik->i", components, kernel, components.conj()).real
    )
    status = "tolerance-sensitive" if sensitivity > floor else "resolved-at-precision"
    split_tv = float(abs(limit - limiting(0.0)).sum() / 2)
    if split_tv > floor and status == "resolved-at-precision":
        status = "unresolved-phase-splitting"
    history = []
    for _ in range(horizon):
        history.append(marginal(v))
        v = np.einsum("ij,j->i", u, v)
    avg = np.mean(history, axis=0)
    return Mixing(
        np.array(history),
        avg,
        limit,
        float(abs(avg - limit).sum() / 2),
        spectral,
        sensitivity,
        status,
        split_tv,
    )


def dephased_mixing(
    step: ArrayLike,
    initial: ArrayLike,
    horizon: int,
    strength: float,
    positions: Optional[ArrayLike] = None,
) -> np.ndarray:
    """After each unitary step, dephase between distinct position eigenspaces."""
    u = unitary(step)
    v = ket(initial, len(u))
    n = count(horizon)
    labels = np.arange(len(u)) if positions is None else np.asarray(positions)
    if (
        not n
        or not 0 <= strength <= 1
        or labels.shape != (len(u),)
        or labels.dtype.kind not in "iu"
        or labels.min() < 0
    ):
        raise ValueError("invalid horizon, dephasing strength or position labels")
    rho = np.outer(v, v.conj())
    attenuation = np.where(labels[:, None] == labels[None, :], 1.0, 1.0 - strength)
    result = []
    for _ in range(n):
        result.append(np.bincount(labels, weights=rho.diagonal().real))
        rho = (u @ rho @ adj(u)) * attenuation
    return np.asarray(result)
