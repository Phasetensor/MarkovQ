"""Nearest-neighbour walks, lattice benchmarks and empirical path statistics."""
import numpy as np
from scipy import sparse
from .chains import MarkovChain, count, matrix


def graph_walk(adjacency, lazy=0.):
    a = matrix(adjacency)
    degrees = np.asarray(a.sum(axis=1)).ravel()
    if a.min() < 0 or (degrees <= 0).any() or not 0 <= lazy <= 1:
        raise ValueError("expected nonnegative weights, positive degrees and valid laziness")
    if sparse.issparse(a):
        return MarkovChain((1-lazy)*sparse.diags(1/degrees) @ a + lazy*sparse.eye(len(degrees)))
    return MarkovChain((1-lazy)*a/degrees[:,None] + lazy*np.eye(len(a)))


def line_walk(size, right=.5, boundary="reflect", lazy=0.):
    size = count(size)
    if size < 2 or not 0 <= right <= 1 or not 0 <= lazy <= 1 or boundary not in {"reflect","absorb","cycle"}:
        raise ValueError("invalid finite walk")
    p = np.zeros((size,size))
    for i in range(size):
        for j,weight in ((i+1,right),(i-1,1-right)):
            if boundary == "absorb" and i in (0,size-1):
                j = i
            j = j % size if boundary == "cycle" else min(max(j,0),size-1)
            p[i,j] += weight
    return MarkovChain((1-lazy)*p+lazy*np.eye(size))


def lattice_paths(dim,steps,paths,seed=None,right=.5):
    dim,steps,paths = count(dim),count(steps),count(paths)
    if not dim or not 0 <= right <= 1:
        raise ValueError("invalid lattice walk")
    rng = np.random.default_rng(seed)
    out = np.zeros((paths,steps+1,dim),dtype=int)
    axis = rng.integers(dim,size=(paths,steps,1))
    signs = np.where(rng.random((paths,steps,1))<right,1,-1)
    np.put_along_axis(out[:,1:],axis,signs,axis=2)
    return out.cumsum(axis=1)


def path_statistics(paths):
    x = np.asarray(paths)
    if x.ndim != 3 or min(x.shape) == 0:
        raise ValueError("expected trajectories by time by coordinate")
    delta = x-x[:,0:1]
    at_origin = (delta==0).all(axis=2)
    return dict(mean=x.mean(axis=0),variance=x.var(axis=0),msd=(delta**2).sum(axis=2).mean(axis=0),
                return_probability=at_origin.mean(axis=0),origin_visits=at_origin.sum(axis=1))


def lattice_recurrence(dimension,right=.5):
    dimension = count(dimension)
    if dimension == 0 or not 0<=right<=1 or (dimension>1 and right!=.5):
        raise ValueError("supports biased line or symmetric lattice")
    return "null recurrent" if right==.5 and dimension<=2 else "transient"


def line_return(steps,right=.5):
    from scipy.stats import binom
    steps=count(steps)
    if not 0<=right<=1:
        raise ValueError("invalid bias")
    return float(binom.pmf(steps//2,steps,right)) if steps%2==0 else 0.


def eventual_line_return(right=.5):
    if not 0<=right<=1:
        raise ValueError("invalid bias")
    return 2*min(right,1-right)


def grid_walk(rows,columns,lazy=0.):
    rows,columns=count(rows),count(columns)
    if min(rows,columns)<1 or rows*columns==1:
        raise ValueError("grid requires at least two vertices")
    def edges(n):
        return sparse.diags([np.ones(n-1),np.ones(n-1)],[-1,1],shape=(n,n))
    return graph_walk(sparse.kronsum(edges(columns),edges(rows)),lazy)


def empirical_passage(paths,targets,first_return=False):
    """Return observed hitting steps and censoring indicators for state-index paths."""
    x = np.asarray(paths)
    if x.ndim!=2 or not x.shape[1]:
        raise ValueError("expected a path matrix with an initial observation")
    hit = np.isin(x,list(targets))
    if first_return:
        hit[:,0] = False
    observed = hit.any(axis=1)
    return np.where(observed,hit.argmax(axis=1),x.shape[1]-1),~observed


def occupation_times(paths,states):
    x = np.asarray(paths)
    if x.ndim!=2:
        raise ValueError("expected a path matrix")
    return np.column_stack([(x==i).sum(axis=1) for i in states])
