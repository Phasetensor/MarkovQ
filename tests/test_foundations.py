import numpy as np
import pytest
from scipy import sparse
from numpy.testing import assert_allclose as close

import markovq as m
from markovq.walks import grid_walk, occupation_times, empirical_passage
from markovq.spectral import spectral_diagnostics

P = np.array([[.75,.25],[.5,.5]])
SWAP = [[0,1],[1,0]]


@pytest.mark.parametrize("constructor",[np.array,sparse.csr_matrix])
def test_transitions(constructor):
    c = m.MarkovChain(constructor(P))
    for i in range(2):
        for j in range(2):
            assert c.transition_probability(i,j,5)==pytest.approx(np.linalg.matrix_power(P,5)[i,j])
    close(c.evolve([1,0],5),c.evolve(c.evolve([1,0],2),3))
    close(c.evolve([1,0],0),[1,0])
    a=c.sample_paths([1,0],2,10000,12)
    close(a,c.sample_paths([1,0],2,10000,12))
    assert abs(a[:,-1].mean()-c.transition_probability(0,1,2))<.025


def test_chronology():
    c=m.MarkovChain(P,schedule=[SWAP])
    close(c.n_step_matrix(2),P @ SWAP)
    close(c.n_step_matrix(1,1),SWAP)
    paths=c.sample_paths([1,0],2,100,3)
    assert np.all(paths[:,2]==1-paths[:,1])
    with pytest.raises(ValueError):
        c.n_step_matrix(3)


@pytest.mark.parametrize("bad",[[],[[1,0]],[[np.nan]],[[.99]],[[-1,2],[0,1]],[[1j]]])
def test_invalid_matrix(bad):
    with pytest.raises(ValueError):
        m.MarkovChain(bad)


def test_classes():
    c=m.classify_states([[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
    assert {(x.states,x.period,x.recurrent,x.absorbing) for x in c}=={
        ((0,),0,False,False),((1,2),2,True,False),((3,),1,True,True)}


def test_walk_statistics():
    x=m.lattice_paths(1,12,10000,27)
    s=m.path_statistics(x)
    assert abs(s["variance"][-1,0]-12)<.6
    assert abs(s["return_probability"][-1]-m.line_return(12))<.02
    assert m.line_return(3)==0
    for dim in [1,2,3]:
        assert m.lattice_recurrence(dim)==("null recurrent" if dim<3 else "transient")
    assert m.eventual_line_return(.75)==.5
    close(grid_walk(2,3).P.sum(axis=1),np.ones((6,1)))
    close(m.line_walk(4,boundary="cycle").P[0],[0,.5,0,.5])
    close(m.line_walk(3).P[0],[.5,.5,0])
    close(m.graph_walk([[0,2],[3,0]],.25).P,[[.25,.75],[.75,.25]])


def test_ruin():
    c=m.line_walk(9,boundary="absorb")
    a=m.absorption(c.P)
    i=a.transient
    close(a.probabilities[:,a.classes.index((8,))],i/8)
    close(a.mean,i*(8-i))
    close(a.visits.sum(axis=1),a.mean)
    times,censored=empirical_passage(c.sample_paths(np.eye(9)[4],160,10000,71),[0,8])
    assert censored.sum()<5
    assert abs(times.mean()-16)<.65
    close(occupation_times([[0,1,1],[1,0,1]],[0,1]),[[1,2],[1,2]])
    assert m.absorption(np.eye(2)).visits.shape==(0,0)


def test_defective_passage():
    p=[[0,.25,.75],[0,1,0],[0,0,1]]
    r=m.hitting(p,[1])
    close(r.probability,[.25,1,0])
    close(r.mean,[np.inf,0,np.inf])
    close(r.conditional_mean[:2],[1,0])
    assert np.isnan(r.conditional_mean[2])
    mass,s=m.passage_distribution(p,[1],4)
    close(mass[1],[.25,0,0])
    close(s[-1],[.75,0,1])
    mass,_=m.passage_distribution(SWAP,[0],4,first_return=True)
    close(mass[:,0],[0,0,1,0,0])
    close(m.recurrence(SWAP,0),[1,2])
    close(m.recurrence(p,0),[0,np.inf])
    close(m.return_generators(SWAP,0,.5),[4/3,.25])


@pytest.mark.parametrize("method",["linear","eigen"])
def test_stationary(method):
    close(m.stationary_distributions(P,method),[[2/3,1/3]])
    close(np.sort(m.stationary_distributions(np.eye(2),method),axis=0),[[0,0],[1,1]])
    assert m.detailed_balance(P,[2/3,1/3])
    close(m.iterate(SWAP,[1,0],3,True),[.5,.5])
    close(m.iterate(SWAP,[1,0],2),[1,0])
    assert m.distances([1,0],[0,1])["kl"]==np.inf
    assert m.mixing_time(P)==1
    assert m.mixing_time(P,max_steps=0) is None
    gap,bound=m.reversible_bound(P,3)
    assert gap==pytest.approx(.75)
    assert bound>=m.distances(m.iterate(P,[0,1],3),[2/3,1/3])["tv"]
    with pytest.raises(ValueError):
        m.mixing_time(SWAP)


def test_ctmc():
    c=m.birth_death([2],[1],"hours")
    for t in [0,.25,2]:
        exact=np.eye(2)*np.exp(-3*t)+(1-np.exp(-3*t))*np.array([[1/3,2/3]]*2)
        close(c.transition(t),exact,atol=1e-12)
        close(c.uniformize(t),exact,atol=2e-12)
        close(c.derivative(t),c.Q @ exact,atol=1e-12)
    times,states=c.events(0,10,32)
    assert np.all(np.diff(times)>0) and times[-1]<=10 and np.all(np.diff(states)!=0)
    close(c.absorption_time([1]),[.5,0])
    absorbed=m.birth_death([2],[0],"hours")
    close(absorbed.events(1,2,1)[0],[0])


@pytest.mark.parametrize("model",[m.LevyProcess(intensity=2),m.LevyProcess(volatility=1),
                                 m.LevyProcess(.2,.3,1.5,-.7,.4),m.LevyProcess()])
def test_process(model):
    x,n=model.sample([0,.25,1],10000,312)
    mean,var=model.moments(1)
    assert abs(x[:,-1].mean()-mean)<.08
    assert abs(x[:,-1].var()-var)<.13
    assert abs(np.exp(.7j*x[:,-1]).mean()-model.characteristic(.7,1))<.025
    assert abs(n.sum(axis=1).mean()-model.intensity)<.06


@pytest.mark.parametrize("constructor",[np.array,sparse.csr_matrix])
def test_spectra(constructor):
    a=constructor(np.ones((5,5))-np.eye(5))
    degree,lap,norm=m.graph_operators(a)
    for op,expected in [(degree,[4]*5),(lap,[0,5,5,5,5]),(norm,[0,1.25,1.25,1.25,1.25])]:
        close(m.spectrum(op).values,expected,atol=1e-12)
    out=m.spectrum(m.hamiltonian(a))
    close(out.values,[-1,-1,-1,-1,4])
    close(out.projectors()[0][1],np.eye(5)-np.ones((5,5))/5,atol=1e-12)
    assert out.residual<1e-12 and out.orthogonality_error<1e-12
    close(m.spectrum(m.hamiltonian(a,"laplacian"),2).values,[0,5],atol=1e-12)
    assert spectral_diagnostics(lap)[1]==np.inf
    with pytest.raises(ValueError):
        m.hamiltonian([[0,1],[0,0]])


@pytest.mark.parametrize("action",[
    lambda:m.MarkovChain(P,schedule=[np.eye(3)]),
    lambda:m.MarkovChain(P).n_step_matrix(-1),
    lambda:m.MarkovChain(P).transition_probability(2,0),
    lambda:m.MarkovChain(P).evolve([1j,0],1),
    lambda:m.MarkovChain(P).evolve([0,0],1),
    lambda:m.hitting(P,[]),lambda:m.passage_distribution(P,[],2),
    lambda:m.return_generators(P,1,1),
    lambda:m.stationary_distributions(P,"bad"),
    lambda:m.reversible_bound(np.eye(2),1),
    lambda:m.graph_walk([[0]]),lambda:m.line_walk(1),
    lambda:m.lattice_paths(0,1,1),lambda:m.path_statistics([]),
    lambda:m.lattice_recurrence(2,.2),lambda:m.line_return(0,2),
    lambda:m.eventual_line_return(-1),lambda:grid_walk(1,1),
    lambda:empirical_passage([],[]),lambda:occupation_times([],[]),
    lambda:m.ContinuousChain([[1]],"seconds"),
    lambda:m.birth_death([1],[],"hours"),
    lambda:m.birth_death([1],[1],"").events(0,1),
    lambda:m.birth_death([1],[1],"s").uniformize(-1),
    lambda:m.birth_death([1],[1],"s").uniformize(1,2),
    lambda:m.LevyProcess(intensity=-1),lambda:m.LevyProcess().sample([0,-1],1),
    lambda:m.graph_operators([[-1]]),lambda:m.hamiltonian([[0]],gamma=np.inf),
    lambda:m.hamiltonian([[1]],"unknown"),lambda:m.spectrum([]),
    lambda:m.spectrum([[np.nan]]),lambda:m.spectrum(P,1),
    lambda:m.spectrum(np.eye(2),1),lambda:m.spectrum(np.eye(2)).projectors(-1),
    lambda:spectral_diagnostics([[0]],-1)])
def test_errors(action):
    with pytest.raises(ValueError):
        action()


def test_complex_hermitian():
    a=sparse.diags([np.full(5,1j),np.full(5,-1j)],[-1,1],shape=(6,6))
    close(m.spectrum(a,2).values,m.spectrum(a).values[:2],atol=1e-12)


def test_nonreversible_and_nearly_defective():
    with pytest.raises(ValueError):
        m.reversible_bound(m.line_walk(4,1,"cycle").P,2)
    r=m.hitting([[0,1-1e-14,1e-14],[0,1,0],[0,0,1]],[1])
    assert np.isinf(r.mean[0])
    assert r.conditional_mean[0]==pytest.approx(1)


def test_multiple_targets_and_initial_censoring():
    mass,s=m.passage_distribution(P,[0,1],0)
    close(mass,[[1,1]])
    close(s,[[0,0]])
    t,c=empirical_passage([[1,0,0],[0,1,0]],[1],True)
    close(t,[2,1])
    close(c,[True,False])
