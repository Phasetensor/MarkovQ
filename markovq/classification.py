"""Finite-chain communication and reachability queries."""

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

from ._types import MatrixInput
from .chains import classify_states, state, validate_transition_matrix


def accessible(transition: MatrixInput, source: int, target: int) -> bool:
    """Whether a positive-probability path exists, allowing a zero-length path."""
    graph = csr_matrix(validate_transition_matrix(transition))
    graph.eliminate_zeros()
    start = state(source, graph.shape[0])
    end = state(target, graph.shape[0])
    reachable = breadth_first_order(
        graph, start, directed=True, return_predecessors=False
    )
    return bool(end in reachable)


def communicates(transition: MatrixInput, source: int, target: int) -> bool:
    return accessible(transition, source, target) and accessible(
        transition, target, source
    )


def communicating_classes(transition: MatrixInput) -> list[tuple[int, ...]]:
    return [c.states for c in classify_states(transition)]


def closed_classes(transition: MatrixInput) -> list[tuple[int, ...]]:
    return [c.states for c in classify_states(transition) if c.recurrent]


def state_period(transition: MatrixInput, source: int) -> int:
    p = validate_transition_matrix(transition)
    i = state(source, p.shape[0])
    return next(c.period for c in classify_states(p) if i in c.states)


def is_irreducible(transition: MatrixInput) -> bool:
    return len(classify_states(transition)) == 1
