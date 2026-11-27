import numpy as np
import pytest
from numpy.testing import assert_allclose as close
from scipy import linalg,sparse
from markovq import quantum as q

X=np.array([[0,1],[1,0]])
I=np.eye(2)
V=np.array([1.,0.])
P=np.diag(V)
H=q.hadamard()


def test_line_cycle_and_local_ports():
    x,p=q.line_distribution(2)
    close(p,[.25,0,.5,0,.25])
    x,p=q.line_distribution(5)
    close(p,p[::-1])
    walk=q.cycle_walk(13)
    v=np.kron(np.array([1,1j])/np.sqrt(2),np.eye(13)[6])
    pos,coin=walk.marginals(walk.evolve(v,5))
    close(pos[1:12],p)
    close(coin.sum(),1)
    graph=q.graph_walk([[0,1,0],[1,0,1],[0,1,0]])
    close(graph.operator.conj().T@graph.operator,np.eye(4))
    close(graph.operator@np.eye(4)[0],np.eye(4)[1])
    close(q.graph_walk(X,[np.eye(1)]*2).operator,X)


@pytest.mark.parametrize("method",["krylov","dense","spectral"])
@pytest.mark.parametrize("matrix",[np.asarray,sparse.csr_matrix])
def test_continuous(method,matrix):
    h=matrix(X)
    state=q.continuous_walk(h,V,.71,method)
    close(state,[np.cos(.71),-1j*np.sin(.71)],atol=1e-12)
    close(q.continuous_walk(h,state,-.71,method),V,atol=1e-12)


def test_szegedy_phase_pair_and_reverse():
    p=np.array([[.75,.25],[.25,.75]])
    w=q.szegedy(p)
    for b in [w.A,w.B]:
        close(b.T@b,I)
    for r in [w.reflection_a,w.reflection_b]:
        close(r@r,np.eye(4),atol=1e-12)
    close(w.discriminant,p)
    close(w.operator@w.coherent([.5,.5]),w.coherent([.5,.5]))
    phases=np.sort(np.angle(np.linalg.eigvals(w.operator)))
    close(phases,[-2*np.pi/3,0,0,2*np.pi/3],atol=1e-12)
    directed=np.roll(np.eye(3),1,axis=1)
    two=q.szegedy(directed,directed.T)
    close(two.discriminant,directed)


def test_eventual_and_finite():
    target=q.target_projector(2,[0])
    event=q.eventual_detection(H,V,target)
    close([event.probability,event.conditional_mean],[1,2])
    d=q.first_detection(H,V,P,9)
    close(d.first[1:],.5**np.arange(1,10))
    close(d.first.cumsum()+d.survival,1)
    close(abs(d.conditional_state()),[0,1])
    with pytest.raises(ValueError):
        q.first_detection(X,[0,1],P,1).conditional_state()
    u=linalg.block_diag(H,1)
    v=np.array([1,0,1])/np.sqrt(2)
    result=q.eventual_detection(u,v,np.diag([1,0,0]))
    close([result.probability,result.conditional_mean],[.5,2])
    close(result.dark_projector,np.diag([0,0,1]),atol=1e-12)
    assert np.isnan(q.eventual_detection(I,V,np.zeros((2,2))).conditional_mean)


def test_renewal_and_mixing():
    r=q.recurrence(H,V,10)
    close(r.occupancy[2::2],1)
    close(abs(r.first_amplitudes)**2,r.monitored.first,atol=1e-12)
    for u,v,limit in [(X,V,[.5,.5]),(I,V,[1,0])]:
        m=q.quantum_mixing(u,v,8)
        close(m.average,limit)
        close(m.limiting_average,limit,atol=1e-12)
        assert m.tv_to_limit<1e-12
    m=q.quantum_mixing(X,V,3,[0,0])
    close(m.instantaneous,np.ones((3,1)))


def test_search_resources():
    n=4
    v=np.ones(n)/2
    g=np.ones((n,n))-np.eye(n)
    a=q.amplitude_amplification(v,[0],[0,1,2])
    close(a.success,[.25,1,.25])
    close(a.discrete_queries,[0,1,2])
    assert a.best_index==1
    s=q.continuous_search(g,1/n,v,[0],[0,np.pi])
    close(s.success,[.25,1])
    close(s.oracle_time,[0,np.pi])
    r,b=q.classical_search(g,[.25]*4,[0],2)
    close(r,[.25,.5,2/3])
    close(b,1-.75**np.arange(1,4))
    coined=q.cycle_walk(4)
    close(q.coined_search(coined,np.ones(8)/np.sqrt(8),[2],[0]).success,[.25])
    close(q.szegedy_search(q.szegedy(I),[.5,.5],[1],[0]).success,[.5])


@pytest.mark.parametrize("method",["expm","ode"])
def test_open_dynamics(method):
    c=np.array([[0,1],[0,0]])
    times=np.linspace(0,2,11)
    e=q.evolve_density(X*0,np.diag([0,1]),times,[c],method)
    close(e.populations[:,1],np.exp(-times),atol=1e-9)
    assert max(e.trace_error)<1e-12 and min(e.minimum_eigenvalue)>-1e-12
    close(e.coherence_l1,0)
    close(q.evolve_density(X,P,[0],method=method).states,[P])
    h,channels=q.stochastic_walk(X,[[0,2],[1,0]],1)
    w=q.evolve_density(h,P,times,channels,method)
    close(w.populations[:,1],2/3*(1-np.exp(-3*times)),atol=1e-9)
    close(w.hermiticity_error,0)


@pytest.mark.parametrize("action",[
    lambda:q.grover(0),lambda:q.cycle_walk(1),lambda:q.cycle_walk(4,np.ones((2,2))),
    lambda:q.line_distribution(1,coin=np.eye(3)),lambda:q.graph_walk(I),lambda:q.graph_walk(X,[]),
    lambda:q.continuous_walk(X,V,np.inf),lambda:q.continuous_walk(X,V,1,"bad"),
    lambda:q.szegedy(I,np.eye(3)),lambda:q.first_detection(X,V,X,0),
    lambda:q.first_detection(X,[1,1],P,1),lambda:q.first_detection([],V,P,1),
    lambda:q.eventual_detection(I,V,P,-1),lambda:q.quantum_mixing(I,V,0),
    lambda:q.quantum_mixing(I,V,2,[0,-1]),lambda:q.discrete_search(I,V,[0],[]),
    lambda:q.continuous_search([[0,1],[0,0]],1,V,[0],[0]),lambda:q.continuous_search(X,1,V,[0],[-1]),
    lambda:q.density_matrix(np.diag([2,-1])),lambda:q.liouvillian([[0,1],[0,0]]),
    lambda:q.liouvillian(I,[np.eye(3)]),lambda:q.evolve_density(I,P,[-1]),
    lambda:q.evolve_density(I,P,[0],atol=-1),lambda:q.evolve_density(I,P,[0],method="bad"),
    lambda:q.stochastic_walk(X,[[-1,0],[0,0]],1)])
def test_domain_errors(action):
    with pytest.raises(ValueError):
        action()
