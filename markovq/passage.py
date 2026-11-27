"""Finite-chain boundary equations. Hitting includes time zero."""
from typing import NamedTuple
import numpy as np
from scipy import sparse
from .chains import classify_states, count, state, validate_transition_matrix


def dense(p):
    p = validate_transition_matrix(p)
    return p.toarray() if sparse.issparse(p) else p


def ancestors(p, targets):
    found = np.zeros(len(p), bool)
    found[targets] = True
    while True:
        updated = found | (p[:,found] > 0).any(axis=1)
        if np.array_equal(found,updated):
            return found
        found = updated


class Passage(NamedTuple):
    probability: np.ndarray
    mean: np.ndarray
    conditional_mean: np.ndarray


def hitting(p, targets):
    p = dense(p)
    target = sorted({state(i,len(p)) for i in targets})
    if not target:
        raise ValueError("empty target")
    reachable = ancestors(p,target)
    interior = reachable.copy()
    interior[target] = False
    prob = np.zeros(len(p))
    prob[target] = 1
    a = np.eye(interior.sum()) - p[np.ix_(interior,interior)]
    prob[interior] = np.linalg.solve(a,p[interior][:,target].sum(axis=1))
    first_moment = np.linalg.solve(a,prob[interior])
    conditional = np.full(len(p),np.nan)
    conditional[target] = 0
    conditional[interior] = first_moment/prob[interior]
    killed = p.copy()
    killed[target] = 0
    defective = ancestors(killed,np.flatnonzero(~reachable))
    return Passage(prob,np.where(defective,np.inf,conditional),conditional)


def passage_distribution(p,targets,horizon,*,first_return=False):
    p = dense(p)
    targets = sorted({state(i,len(p)) for i in targets})
    if not targets:
        raise ValueError("empty target")
    out = np.zeros((count(horizon)+1,len(p)))
    out[0,targets] = 1
    if horizon:
        out[1] = p @ out[0]
        if not first_return:
            out[1,targets] = 0
    killed = p.copy()
    killed[:,targets] = 0
    for t in range(2,horizon+1):
        out[t] = killed @ out[t-1]
        if not first_return:
            out[t,targets] = 0
    if first_return:
        out[0] = 0
    return out, 1-out.cumsum(axis=0)


def recurrence(p,i):
    p = dense(p)
    i = state(i,len(p))
    result = hitting(p,[i])
    positive = p[i] > 0
    return float(p[i] @ result.probability), float(1+p[i,positive] @ result.mean[positive])


def return_generators(p,i,z):
    p = dense(p)
    i = state(i,len(p))
    if abs(z) >= 1:
        raise ValueError("requires |z| < 1")
    green = np.linalg.solve(np.eye(len(p))-z*p,np.eye(len(p))[:,i])[i]
    return green, 1-1/green


class Absorption(NamedTuple):
    transient: np.ndarray
    classes: tuple
    visits: np.ndarray
    probabilities: np.ndarray
    mean: np.ndarray
    condition_number: float


def absorption(p):
    p = dense(p)
    classes = tuple(c.states for c in classify_states(p) if c.recurrent)
    transient = np.array(sorted(set(range(len(p))).difference(*classes)),dtype=int)
    a = np.eye(len(transient))-p[np.ix_(transient,transient)]
    visits = np.linalg.solve(a,np.eye(len(transient)))
    rates = np.column_stack([p[transient][:,c].sum(axis=1) for c in classes])
    return Absorption(transient,classes,visits, visits @ rates,visits.sum(axis=1),float(np.linalg.cond(a)) if a.size else 1.)
