"""Independent regression cases for near-absorbing kernels and stiff decay."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse

from markovq import MarkovChain, absorption, hitting
from markovq import quantum as q


@pytest.mark.parametrize("probability", [2**-10, 2**-25, 2**-40])
def test_slow_absorption(probability):
    p = [[1 - probability, probability, 0], [0, 1, 0], [0, 0, 1]]
    expected = 1 / probability
    chain = MarkovChain(sparse.csr_matrix(p))
    assert_allclose(absorption(chain.P).mean, [expected])
    result = hitting(chain.P, [1])
    assert_allclose(result.probability, [1, 1, 0])
    assert_allclose(result.mean, [expected, 0, np.inf])
    survival = np.exp(1024 * np.log1p(-probability))
    assert_allclose(chain.transition_probability(0, 0, 1024), survival)


@pytest.mark.parametrize("method", ["ode", "bdf", "sparse", "matrix-free"])
def test_stiff_decay_across_timescales(method):
    times = np.array([0, 1e-9, 1e-4, 0.02, 1.0])
    collapse = np.array([[0, np.sqrt(500)], [0, 0]])
    timeseries = q.evolve_density(
        np.zeros((2, 2)), np.diag([0.0, 1.0]), times, [collapse], method
    )
    assert_allclose(timeseries.populations[:, 1], np.exp(-500 * times), atol=2e-9)
    assert not np.any(timeseries.violations(1e-9))


def test_matrix_form_avoids_kronecker_products(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError(
            "a matrix-form calculation constructed a Kronecker product"
        )

    monkeypatch.setattr(np, "kron", unexpected)
    monkeypatch.setattr(sparse, "kron", unexpected)
    for method in ("ode", "matrix-free"):
        result = q.evolve_density(np.eye(2), np.eye(2) / 2, [0, 0.1], method=method)
        assert_allclose(result.states[-1], np.eye(2) / 2)


def test_dense_budget_precedes_conversion(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("sparse input was expanded before checking the budget")

    matrix = sparse.eye(1024, format="csr")
    monkeypatch.setattr(sparse.csr_matrix, "toarray", unexpected)
    with pytest.raises(MemoryError):
        q.szegedy(matrix)
    operator = q.implicit_szegedy(matrix)
    assert operator.A.nnz == operator.B.nnz == 1024


def test_solver_accuracy_warning_is_explicit(monkeypatch):
    import warnings

    from scipy import linalg

    def warning(*args, **kwargs):
        warnings.warn("ill conditioned", linalg.LinAlgWarning)

    monkeypatch.setattr(linalg, "solve_discrete_lyapunov", warning)
    r = q.eventual_detection(np.eye(2), [1, 0], np.diag([1, 0]))
    assert r.diagnostics.mean_status == "unresolved"
    assert np.isnan(r.conditional_mean)
