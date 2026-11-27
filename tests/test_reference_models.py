"""Independent small-system references for the classical and spectral operators."""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from markovq import (
    ContinuousChain,
    MarkovChain,
    absorption,
    classify_states,
    graph_operators,
    grid_walk,
    hitting,
    line_walk,
    passage_distribution,
    recurrence,
    spectrum,
    stationary_distributions,
)


def test_reference_path_enumeration():
    """Enumerate every path; stop and count a first hit once per trajectory."""
    p = np.array([[0.2, 0.4, 0.4], [0.1, 0.6, 0.3], [0.7, 0, 0.3]])
    horizon = 4
    final = np.zeros((3, 3))
    first = np.zeros((horizon + 1, 3))
    for initial in range(3):
        for suffix in itertools.product(range(3), repeat=horizon):
            path = (initial, *suffix)
            weight = np.prod([p[i, j] for i, j in zip(path, path[1:])])
            final[initial, path[-1]] += weight
            if 1 in path:
                first[path.index(1), initial] += weight
    assert_allclose(MarkovChain(p).n_step_matrix(horizon), final)
    mass, survival = passage_distribution(p, [1], horizon)
    assert_allclose(mass, first)
    assert_allclose(survival[-1], 1 - first.sum(axis=0), atol=1e-14)


def test_weighted_stationarity_and_kac_return_time():
    adjacency = np.array([[0, 2, 1], [2, 0, 3], [1, 3, 0]])
    degree = adjacency.sum(axis=1)
    p = adjacency / degree[:, None]
    pi = degree / degree.sum()
    assert_allclose(stationary_distributions(p)[0], pi)
    for i in range(3):
        assert_allclose(recurrence(p, i), [1, 1 / pi[i]])


def test_closed_class_entry_and_conditional_time():
    p = [[0.5, 0.2, 0.3, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]
    a = absorption(p)
    assert_allclose(a.mean, [2])
    assert_allclose(a.probabilities[:, a.classes.index((1, 3))], [0.4])
    result = hitting(p, [1, 3])
    assert np.isinf(result.mean[0])
    assert_allclose(result.conditional_mean[0], 2)


@pytest.mark.parametrize("seed", range(6))
def test_communication_against_boolean_reachability(seed):
    rng = np.random.default_rng(seed)
    support = rng.random((7, 7)) < 0.2
    support |= np.eye(7, dtype=bool)
    p = support / support.sum(axis=1, keepdims=True)
    reach = support.copy()
    for pivot in range(7):
        reach |= reach[:, [pivot]] & reach[[pivot], :]
    classes = classify_states(p)
    for c in classes:
        i = c.states[0]
        assert set(c.states) == set(np.flatnonzero(reach[i] & reach[:, i]))
        assert c.recurrent == bool(np.all(~reach[i] | reach[:, i]))


def test_grid_and_disconnected_spectrum():
    p = grid_walk(2, 3).P.toarray()
    adjacency = (p > 0).astype(float)
    isolated = np.pad(adjacency, ((0, 1), (0, 1)))
    _, lap, normalized = graph_operators(isolated)
    assert_allclose(normalized[-1], 0)
    # Cartesian product Laplacian spectrum is the pairwise sum of path spectra.
    assert_allclose(spectrum(lap).values, [0, 0, 1, 2, 3, 3, 5], atol=1e-12)


def test_poisson_count_before_saturation():
    from scipy.stats import poisson

    size = 10
    q = np.diag(np.ones(size - 1), 1) - np.diag([1] * (size - 1) + [0])
    model = ContinuousChain(q, "seconds")
    expected = np.r_[poisson.pmf(np.arange(size - 1), 2), poisson.sf(size - 2, 2)]
    assert_allclose(model.uniformize(2)[0], expected, atol=1e-12)
    assert_allclose(model.transition(2)[0], expected, atol=1e-12)


def test_hitting_finite_horizon_can_miss_eventual_hit():
    p = line_walk(8, boundary="reflect").P
    result = hitting(p, [7])
    assert_allclose(result.probability[0], 1)
    mass, survival = passage_distribution(p, [7], 4)
    assert_allclose(mass[:, 0], 0)
    assert_allclose(survival[:, 0], 1)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_dilation(sparse_input):
    from scipy import sparse

    from markovq.spectral import hermitian_dilation

    directed = np.array([[0, 1], [2, 0]])
    h = hermitian_dilation(sparse.csr_matrix(directed) if sparse_input else directed)
    assert_allclose(spectrum(h).values, [-2, -1, 1, 2])
