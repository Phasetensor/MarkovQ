"""Independent endpoint trajectories with worker-invariant random streams."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple
import numpy as np
from scipy import sparse
from .chains import count,distribution,validate_transition_matrix
from ._types import MatrixInput

class BatchResult(NamedTuple):
    counts: np.ndarray
    standard_errors: np.ndarray
    batches: int
    storage_bytes: int

def _simulate(p,law,steps,n,child):
    rng=np.random.default_rng(child)
    counts=rng.multinomial(n,law)
    for _ in range(steps):
        following=np.zeros_like(counts)
        for i in np.flatnonzero(counts):
            start,end=p.indptr[i:i+2]
            following[p.indices[start:end]]+=rng.multinomial(counts[i],p.data[start:end])
        counts=following
    return counts

def _chunk(args):
    return _simulate(*args)

def simulate_endpoints(transition: MatrixInput,initial,steps: int,trajectories: int,*,seed: int,batch_size: int=2048,workers: int=1) -> BatchResult:
    p=sparse.csr_matrix(validate_transition_matrix(transition))
    p.sum_duplicates()
    law=distribution(initial,p.shape[0])
    n,b,w=count(trajectories),count(batch_size),count(workers)
    if not min(n,b,w):
        raise ValueError("positive trajectory, batch and worker counts required")
    steps=count(steps)
    batches=(n+b-1)//b
    children=np.random.SeedSequence(count(seed)).spawn(batches)
    jobs=((p,law,steps,min(b,n-i*b),child) for i,child in enumerate(children))
    if w==1:
        total=sum(map(_chunk,jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(w,batches)) as pool:
            total=sum(pool.map(_chunk,jobs))
    prob=total/n
    return BatchResult(total,np.sqrt(prob*(1-prob)/n),batches,p.data.nbytes+p.indices.nbytes+p.indptr.nbytes)
