from decimal import Decimal,localcontext
import numpy as np
import pytest
from scipy import linalg,sparse
from numpy.testing import assert_allclose as close
from markovq import quantum as q
from markovq.quantum.measurement import _groups
from markovq.quantum.search import SearchResult

H=q.hadamard()
P=np.diag([1,0])
V=np.array([1.,0.])

@pytest.mark.parametrize("delta",[0.,1e-3,1e-7,1e-11,1e-15])
def test_phase_resolution(delta):
    u=H@np.diag([1,np.exp(1j*delta)])@H
    r=q.quantum_mixing(u,V,100)
    analytic=np.mean(np.cos(np.arange(100)*delta/2)**2)
    close(r.average[0],analytic,atol=2e-13)
    close(r.average,r.finite_spectral,atol=2e-13)
    close(r.average.sum(),1,atol=2e-13)
    if delta==1e-11:
        assert r.limit_sensitivity_tv>.49
        assert q.eventual_detection(u,[0,1],P).diagnostics.status=="tolerance-sensitive"
    if delta>=1e-7:
        close(r.limiting_average,[.5,.5],atol=1e-8)


def test_cluster_permutation_and_cut():
    angles=np.array([np.pi-4e-5,-np.pi+4e-5,-np.pi+.00013,0.])
    z=np.exp(1j*angles)
    for order in [np.arange(4),[3,0,2,1],[2,1,3,0]]:
        groups=_groups(z[order],.0001)
        assert {frozenset(np.asarray(order)[g]) for g in groups}=={frozenset([0,1,2]),frozenset([3])}


@pytest.mark.parametrize("seed",range(5))
def test_multitarget_basis_invariance(seed):
    rng=np.random.default_rng(seed)
    basis=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    u=basis@np.diag([1,1,-1,1j])@basis.conj().T
    target=basis[:,:2]@basis[:,:2].conj().T
    v=basis@np.full(4,.5)
    e=q.eventual_detection(u,v,target)
    close([e.probability,e.conditional_mean],[.5,1],atol=1e-12)
    assert np.isinf(e.diagnostics.unconditional_mean)
    assert e.diagnostics.dark_target_residual<1e-12


@pytest.mark.parametrize("prob",[.8,.1,1e-3,1e-6,1e-12,1e-18])
def test_geometric_lyapunov(prob):
    u=np.array([[np.sqrt(1-prob),np.sqrt(prob)],[-np.sqrt(prob),np.sqrt(1-prob)]])
    e=q.eventual_detection(u,[0,1],P)
    with localcontext() as ctx:
        ctx.prec=70
        exact=float(Decimal(1)/Decimal(str(prob)))
    if prob>=1e-6:
        close(e.conditional_mean,exact,rtol=1e-8)
        assert e.diagnostics.relative_residual<1e-10
    else:
        assert e.diagnostics.mean_status=="unresolved" and np.isnan(e.conditional_mean)


def test_weak_coupling():
    theta=1e-12
    rot=np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
    e=q.eventual_detection(rot@np.diag([1,-1])@rot.T,[0,1],P)
    assert np.isnan(e.probability)
    assert e.diagnostics.probability_range[1]>.99


@pytest.mark.parametrize("n",[2,5,8])
@pytest.mark.parametrize("reverse",[False,True])
def test_sparse_reflections(n,reverse):
    rng=np.random.default_rng(n)
    p=np.roll(np.eye(n),1,axis=1)*.6+.4*np.eye(n)
    r=p.T if reverse else p
    dense=q.szegedy(p,r)
    impl=q.implicit_szegedy(sparse.csr_matrix(p),sparse.csr_matrix(r))
    v=rng.normal(size=n*n)+1j*rng.normal(size=n*n)
    v/=np.linalg.norm(v)
    close(impl.evolve(v,3),np.linalg.matrix_power(dense.operator,3)@v,atol=1e-12)
    close(impl.operator.H@v,dense.operator.T@v,atol=1e-12)
    close(impl.coherent(np.ones(n)/n),dense.coherent(np.ones(n)/n))
    with pytest.raises(MemoryError):q.szegedy(p,max_bytes=1)


@pytest.mark.parametrize("method",["sparse","matrix-free","ode","bdf"])
def test_open_stiff(method):
    h=np.array([[1,1j],[-1j,-1]])
    c=[np.diag([2,-2]),np.array([[0,8],[0,0]])]
    ts=[0,.001,.12,1.]
    d=q.evolve_density(h,np.eye(2)/2,ts,c)
    other=q.evolve_density(sparse.csr_matrix(h),np.eye(2)/2,ts,c,method)
    close(other.states,d.states,atol=3e-9)
    assert max(other.trace_error)<1e-9 and min(other.minimum_eigenvalue)>-1e-9


@pytest.mark.parametrize("p",[0.,1.,.2,1e-16,1e-320])
def test_cost(p):
    r=SearchResult(np.array([0]),np.array([p]),0,np.array([0]),np.array([0.]),2,3)
    out=r.expected_cost(0)
    assert out["marking_queries"]==out["analog_oracle_time"]==0
    if 0<p and p>=1e-16:
        close(out["initial_preparations"],2/p)
    else:
        assert np.isinf(out["initial_preparations"])
    with pytest.raises(ValueError):r.expected_cost(1)


def test_cost_invalid_and_underflow():
    r=q.discrete_search(np.eye(2),[1,1e-200],[1],[0])
    assert r.cost_status(0)=="probability-underflow"
    assert np.isinf(r.expected_cost(0)["independent_trials"])
    for p in [-1,np.inf,np.nan,1.1]:
        with pytest.raises(ValueError):r._replace(success=np.array([p])).expected_cost(0)
    with pytest.raises(ValueError):r._replace(oracle_time=np.array([np.nan])).expected_cost(0)
