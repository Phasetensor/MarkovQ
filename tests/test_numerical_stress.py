from decimal import Decimal, localcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from scipy import sparse

from markovq import quantum as q
from markovq.quantum.measurement import _groups
from markovq.quantum.search import SearchResult

H = q.hadamard()
P = np.diag([1, 0])
V = np.array([1.0, 0.0])


@pytest.mark.parametrize("delta", [0.0, 1e-3, 1e-7, 1e-11, 1e-15])
def test_phase_resolution(delta):
    u = H @ np.diag([1, np.exp(1j * delta)]) @ H
    r = q.quantum_mixing(u, V, 100)
    analytic = np.mean(np.cos(np.arange(100) * delta / 2) ** 2)
    close(r.average[0], analytic, atol=2e-13)
    close(r.average, r.finite_spectral, atol=2e-13)
    close(r.average.sum(), 1, atol=2e-13)
    if delta == 1e-11:
        assert r.limit_sensitivity_tv > 0.49
        assert (
            q.eventual_detection(u, [0, 1], P).diagnostics.status
            == "tolerance-sensitive"
        )
    if delta >= 1e-7:
        close(r.limiting_average, [0.5, 0.5], atol=1e-8)


def test_cluster_permutation_and_cut():
    angles = np.array([np.pi - 4e-5, -np.pi + 4e-5, -np.pi + 0.00013, 0.0])
    z = np.exp(1j * angles)
    for order in [np.arange(4), [3, 0, 2, 1], [2, 1, 3, 0]]:
        groups = _groups(z[order], 0.0001)
        assert {frozenset(np.asarray(order)[g]) for g in groups} == {
            frozenset([0, 1, 2]),
            frozenset([3]),
        }


@pytest.mark.parametrize("seed", range(5))
def test_multitarget_basis_invariance(seed):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]
    u = basis @ np.diag([1, 1, -1, 1j]) @ basis.conj().T
    target = basis[:, :2] @ basis[:, :2].conj().T
    v = basis @ np.full(4, 0.5)
    e = q.eventual_detection(u, v, target)
    close([e.probability, e.conditional_mean], [0.5, 1], atol=1e-12)
    assert np.isinf(e.diagnostics.unconditional_mean)
    assert e.diagnostics.dark_target_residual < 1e-12


@pytest.mark.parametrize("prob", [0.8, 0.1, 1e-3, 1e-6, 1e-12, 1e-18])
def test_geometric_lyapunov(prob):
    u = np.array(
        [[np.sqrt(1 - prob), np.sqrt(prob)], [-np.sqrt(prob), np.sqrt(1 - prob)]]
    )
    e = q.eventual_detection(u, [0, 1], P)
    with localcontext() as ctx:
        ctx.prec = 70
        exact = float(Decimal(1) / Decimal(str(prob)))
    if prob >= 1e-6:
        close(e.conditional_mean, exact, rtol=1e-8)
        assert e.diagnostics.relative_residual < 1e-10
    else:
        assert e.diagnostics.mean_status == "unresolved" and np.isnan(
            e.conditional_mean
        )


def test_weak_coupling():
    theta = 1e-12
    rot = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    e = q.eventual_detection(rot @ np.diag([1, -1]) @ rot.T, [0, 1], P)
    assert np.isnan(e.probability)
    assert e.diagnostics.probability_range[1] > 0.99


@pytest.mark.parametrize("n", [2, 5, 8])
@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_reflections(n, reverse):
    rng = np.random.default_rng(n)
    p = np.roll(np.eye(n), 1, axis=1) * 0.6 + 0.4 * np.eye(n)
    r = p.T if reverse else p
    dense = q.szegedy(p, r)
    impl = q.implicit_szegedy(sparse.csr_matrix(p), sparse.csr_matrix(r))
    v = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
    v /= np.linalg.norm(v)
    close(
        impl.evolve(v, 3),
        np.einsum(
            "ij,jk,kl,l->i",
            dense.operator,
            dense.operator,
            dense.operator,
            v,
            optimize=False,
        ),
        atol=1e-12,
    )
    close(impl.operator.H @ v, np.einsum("ji,j->i", dense.operator, v), atol=1e-12)
    close(impl.coherent(np.ones(n) / n), dense.coherent(np.ones(n) / n))
    with pytest.raises(MemoryError):
        q.szegedy(p, max_bytes=1)


@pytest.mark.parametrize("method", ["sparse", "matrix-free", "ode", "bdf"])
def test_open_stiff(method):
    h = np.array([[1, 1j], [-1j, -1]])
    c = [np.diag([2, -2]), np.array([[0, 8], [0, 0]])]
    ts = [0, 0.001, 0.12, 1.0]
    d = q.evolve_density(h, np.eye(2) / 2, ts, c)
    other = q.evolve_density(sparse.csr_matrix(h), np.eye(2) / 2, ts, c, method)
    close(other.states, d.states, atol=3e-9)
    assert max(other.trace_error) < 1e-9 and min(other.minimum_eigenvalue) > -1e-9


@pytest.mark.parametrize("p", [0.0, 1.0, 0.2, 1e-16, 1e-320])
def test_cost(p):
    r = SearchResult(
        np.array([0]), np.array([p]), 0, np.array([0]), np.array([0.0]), 2, 3
    )
    out = r.expected_cost(0)
    assert out["marking_queries"] == out["analog_oracle_time"] == 0
    if 0 < p and p >= 1e-16:
        close(out["initial_preparations"], 2 / p)
    else:
        assert np.isinf(out["initial_preparations"])
    with pytest.raises(ValueError):
        r.expected_cost(1)


def test_cost_invalid_and_underflow():
    r = q.discrete_search(np.eye(2), [1, 1e-200], [1], [0])
    assert r.cost_status(0) == "probability-underflow"
    assert np.isinf(r.expected_cost(0)["independent_trials"])
    for p in [-1, np.inf, np.nan, 1.1]:
        with pytest.raises(ValueError):
            r._replace(success=np.array([p])).expected_cost(0)
    with pytest.raises(ValueError):
        r._replace(oracle_time=np.array([np.nan])).expected_cost(0)


def test_multidimensional_survival_tail():
    # Rank-one target and an independently iterated three-level survivor.
    u = np.fft.fft(np.eye(3)) / np.sqrt(3)
    v = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
    target = np.diag([1, 0, 0])
    e = q.eventual_detection(u, v, target)
    # Symmetric initial vector stays in the symmetric bright sector.
    total = 0.0
    remaining = v.copy()
    for _ in range(100):
        total += np.vdot(remaining, remaining).real
        remaining = u @ remaining
        remaining[0] = 0
    # Conditional survival in the non-target symmetric ray has ratio 1/3.
    tail = 1.5 * np.vdot(remaining, remaining).real
    close(e.conditional_mean, total + tail, atol=1e-12)


def test_extreme_times_and_reducible_chain():
    from markovq import MarkovChain, classify_states, hitting

    p = np.array([[0.5, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    close(MarkovChain(p).evolve([1, 0, 0, 0], 1000), [0, 1, 0, 0], atol=1e-12)
    assert sorted(c.period for c in classify_states(p)) == [1, 1, 2]
    close(hitting(p, [1]).mean, [2, 0, np.inf, np.inf])
    for t in (1e-15, 1e4):
        close(
            q.continuous_walk(np.diag([1, 2]), V, t, "spectral"), [np.exp(-1j * t), 0]
        )


def test_open_generator_adjoint_and_failures():
    h = np.array([[1, 1j], [-1j, 0]])
    c = [np.array([[0, 1j], [0, 0]])]
    v = np.arange(4) + 1j * np.arange(4)[::-1]
    g = q.liouvillian(h, c)
    for name in ("sparse", "matrix-free"):
        other = q.liouvillian(h, c, representation=name)
        close(other @ v, g @ v)
        close(
            other.H @ v if name == "matrix-free" else other.getH() @ v, g.conj().T @ v
        )
    for hbad, cs in [
        (np.ones((2, 3)), []),
        (h, [np.ones((1, 2))]),
        (np.array([[np.nan]]), []),
    ]:
        with pytest.raises(ValueError):
            q.evolve_density(hbad, np.eye(2) / 2, [0], cs, "matrix-free")
    with pytest.raises(ValueError):
        q.liouvillian(h, representation="bad")
    with pytest.raises(ValueError):
        q.implicit_szegedy(np.eye(2), np.eye(3))


def test_solver_tolerance_and_invalid_inputs():
    p = 1e-6
    u = np.array([[np.sqrt(1 - p), np.sqrt(p)], [-np.sqrt(p), np.sqrt(1 - p)]])
    assert (
        q.eventual_detection(u, [0, 1], P, solver_tol=1e-12).diagnostics.mean_status
        == "unresolved"
    )
    assert (
        q.eventual_detection(u, [0, 1], P, solver_tol=1e-6).diagnostics.mean_status
        == "finite"
    )
    with pytest.raises(ValueError):
        q.eventual_detection(u, V, P, solver_tol=0)
    with pytest.raises(ValueError):
        q.eventual_detection(u, V, P, rank_tol=-1)


def test_subnormal_cost_and_metadata():
    r = SearchResult(
        np.array([0]),
        np.array([0.0]),
        0,
        np.array([0]),
        np.array([1e-310]),
        log_success=np.array([-750.0]),
    )
    close(r.expected_cost(0)["analog_oracle_time"], np.exp(np.log(1e-310) + 750))
    with pytest.raises(ValueError):
        r._replace(log_success=np.array([np.nan])).cost_status(0)


def test_high_precision_cesaro_cosine():
    # Direct Decimal Taylor series for (1+cos(t*delta))/2, independent of the Schur kernel.
    with localcontext() as ctx:
        ctx.prec = 70
        for text in ["0.0001", "0.00000000001", "0.000000000000001"]:
            delta = Decimal(text)
            values = []
            for t in range(37):
                x = t * delta
                term = value = Decimal(1)
                for k in range(1, 12):
                    term *= -x * x / (2 * k * (2 * k - 1))
                    value += term
                values.append((1 + value) / 2)
            u = H @ np.diag([1, np.exp(1j * float(delta))]) @ H
            r = q.quantum_mixing(u, V, 37)
            close(r.finite_spectral[0], float(sum(values) / 37), atol=2e-13)


def test_raw_physicality_flags():
    e = q.evolve_density(np.eye(2), np.eye(2) / 2, [0])
    assert not e.violations()[0]
    assert e._replace(minimum_eigenvalue=np.array([-1e-7])).violations(1e-8)[0]
    with pytest.raises(ValueError):
        e.violations(-1)


def test_subresolution_coupling_with_exact_diagonal_basis():
    v = np.array([1.0, 1e-18])
    p = np.outer(v, v)
    e = q.eventual_detection(np.diag([1, -1]), [0, 1], p)
    assert e.diagnostics.status == "tolerance-sensitive"


@pytest.mark.parametrize("n", [4, 9, 16])
def test_sparse_szegedy_topologies(n):
    s = int(np.sqrt(n))
    chain = sparse.diags([1.0, 1.0], [-1, 1], shape=(s, s))
    grid = sparse.kronsum(chain, chain)
    rng = np.random.default_rng(612 + n)
    for graph in [
        grid,
        sparse.csr_matrix(rng.random((n, n)) * (rng.random((n, n)) < 0.2))
        + sparse.eye(n),
    ]:
        p = sparse.diags(1 / np.asarray(graph.sum(axis=1)).ravel()) @ graph
        w = q.implicit_szegedy(p)
        v = rng.normal(size=n * n) + 1j * rng.normal(size=n * n)
        v /= np.linalg.norm(v)
        reference = v.reshape(n, n)
        root = np.sqrt(p.toarray())
        reflected = 2 * root * np.sum(root * reference, axis=1)[:, None] - reference
        out = 2 * root.T * np.sum(root.T * reflected, axis=0)[None, :] - reflected
        close(w.operator @ v, out.ravel(), atol=1e-13)
        close(np.linalg.norm(w.evolve(v, 5)), 1)


@pytest.mark.parametrize("value",[-1.,np.nan,0.])
def test_invalid_duration_is_unresolved(monkeypatch,value):
    from scipy import linalg
    def invalid(a,rhs,method):
        return np.eye(len(a))*value
    monkeypatch.setattr(linalg,"solve_discrete_lyapunov",invalid)
    r=q.eventual_detection(H,V,P)
    assert np.isnan(r.conditional_mean)
    assert r.diagnostics.mean_status=="unresolved"


def test_unconditional_versus_conditional():
    r=q.eventual_detection(np.eye(2),[0,1],P)
    assert r.diagnostics.mean_status=="undefined"
    assert np.isinf(r.diagnostics.unconditional_mean)
    assert np.isnan(r.conditional_mean)
    r=q.eventual_detection(np.eye(2),[1e-20,1],P)
    close(r.conditional_mean,1)
    assert np.isinf(r.diagnostics.unconditional_mean)


def test_high_dimensional_lyapunov():
    # With full detection the first measurement always succeeds.
    r=q.eventual_detection(np.eye(9),np.eye(9)[0],np.eye(9))
    close([r.conditional_mean,r.probability],[1,1])
    assert r.diagnostics.spectral_radius==0


def test_cost_categories_overflow():
    r=SearchResult(np.array([0]),np.array([1e-305]),0,np.array([10000]),np.array([0.]))
    costs=r.expected_cost(0)
    assert np.isfinite(costs["independent_trials"])
    assert np.isinf(costs["marking_queries"])
    assert r.resource_status(0)["marking_queries"]=="finite-overflow"
    assert r.resource_status(0)["gate_count"]=="unmeasured"
