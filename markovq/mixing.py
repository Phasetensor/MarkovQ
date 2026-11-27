"""Closed-class stationary laws and classical convergence diagnostics."""
from __future__ import annotations
from typing import Optional
from numpy.typing import ArrayLike
from ._types import MatrixInput


import numpy as np
from scipy.special import rel_entr
from .chains import classify_states, count, distribution
from .passage import dense


def stationary_distributions(p: MatrixInput, method: str="linear") -> np.ndarray:
    p = dense(p)
    laws = []
    for cls in classify_states(p):
        if not cls.recurrent:
            continue
        q = p[np.ix_(cls.states, cls.states)]
        if method == "linear":
            a = q.T - np.eye(len(q))
            a[-1] = 1
            b = np.zeros(len(q))
            b[-1] = 1
            v = np.linalg.solve(a, b)
        elif method == "eigen":
            w, u = np.linalg.eig(q.T)
            v = u[:, np.argmin(abs(w - 1))].real
            v = v / v.sum()
        else:
            raise ValueError("unknown stationary solver")
        law = np.zeros(len(p))
        law[list(cls.states)] = v
        laws.append(law)
    return np.array(laws)


def detailed_balance(p: MatrixInput, pi: ArrayLike) -> bool:
    p = dense(p)
    flow = distribution(pi, len(p))[:, None] * p
    return np.allclose(flow, flow.T, atol=1e-12, rtol=0)


def distances(p: ArrayLike, q: ArrayLike) -> dict[str, float]:
    p = distribution(p, len(p))
    q = distribution(q, len(p))
    diff = p - q
    return dict(
        tv=np.linalg.norm(diff, 1) / 2,
        l1=np.linalg.norm(diff, 1),
        l2=np.linalg.norm(diff),
        kl=rel_entr(p, q).sum(),
    )


def iterate(p: MatrixInput, initial: ArrayLike, steps: int, cesaro: bool=False) -> np.ndarray:
    p = dense(p)
    current = distribution(initial, len(p))
    avg = current.copy()
    for t in range(1, count(steps) + 1):
        current = current @ p
        avg += (current - avg) / (t + 1)
    return avg if cesaro else current


def mixing_time(p: MatrixInput, epsilon: float=0.25, max_steps: int=1000) -> Optional[int]:
    """Exact worst-start TV on a finite horizon; None means horizon exceeded."""
    p = dense(p)
    classes = classify_states(p)
    if not 0 < epsilon < 1 or len(classes) != 1 or classes[0].period != 1:
        raise ValueError("requires an irreducible aperiodic model and 0 < epsilon < 1")
    pi = stationary_distributions(p)[0]
    rows = np.eye(len(p))
    for t in range(count(max_steps) + 1):
        if abs(rows - pi).sum(axis=1).max() / 2 <= epsilon:
            return t
        rows = rows @ p
    return None


def reversible_bound(p: MatrixInput, steps: int) -> tuple[float, float]:
    p = dense(p)
    pi = stationary_distributions(p)
    if len(classify_states(p)) != 1 or not detailed_balance(p, pi[0]):
        raise ValueError("requires an irreducible reversible chain")
    r = np.sqrt(pi[0])
    values = np.linalg.eigvalsh(r[:, None] * p / r[None, :])
    rate = max(abs(values[:-1]), default=0.0)
    return 1 - rate, min(1.0, rate ** count(steps) / (2 * r.min()))
