"""Transductive seed propagation on fixed undirected weighted networks."""
from __future__ import annotations
from typing import Iterable
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse.linalg import expm_multiply,spsolve
from scipy.stats import rankdata
from .chains import count,matrix,state

def _ids(ids,n):
    return np.array(sorted({state(i,n) for i in ids}),dtype=int)

def rank_nodes(adjacency: ArrayLike,seeds: Iterable[int],restart: float=.2,time: float=1.) -> dict[str,np.ndarray]:
    a=sparse.csr_matrix(matrix(adjacency))
    if a.min()<0 or (a-a.T).nnz or not 0<restart<=1 or not np.isfinite(time) or time<0:
        raise ValueError("expected an undirected nonnegative graph and valid parameters")
    n=a.shape[0]
    ids=_ids(seeds,n)
    if not len(ids):
        raise ValueError("empty seed set")
    s=np.zeros(n)
    s[ids]=1/len(ids)
    degree=np.asarray(a.sum(axis=1)).ravel()
    denom=np.where(degree>0,degree,1)
    p=sparse.diags(1/denom)@a+sparse.diags((degree==0).astype(float))
    lap=sparse.diags(degree)-a
    pr=spsolve((sparse.eye(n)-(1-restart)*p.T).tocsc(),restart*s)
    # Incoherent mixture of localized seeds avoids artificial relative phases.
    qw=np.zeros(n)
    for i in ids:
        v=np.zeros(n)
        v[i]=1
        qw+=abs(expm_multiply((-1j*time)*lap,v))**2/len(ids)
    return {"one_step":p.T@s,"restart_pagerank":pr,"heat":expm_multiply(-time*lap,s),
            "quantum":qw,"degree":degree}

def heldout_metrics(scores: ArrayLike,seeds: Iterable[int],positives: Iterable[int],negatives: Iterable[int],k: int) -> dict[str,float]:
    """AUROC, threshold-group average precision, and tie-averaged precision/recall at k."""
    x=np.asarray(scores,float)
    if x.ndim!=1 or not np.isfinite(x).all():
        raise ValueError("invalid ranking scores")
    train,pos,neg=(_ids(v,len(x)) for v in (seeds,positives,negatives))
    if not len(pos) or not len(neg) or len(set(train)&(set(pos)|set(neg))) or len(set(pos)&set(neg)):
        raise ValueError("require disjoint seeds and labeled positive/negative test nodes")
    y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
    z=x[np.r_[pos,neg]]
    if not 1<=count(k)<=len(y):
        raise ValueError("invalid cutoff")
    auc=(rankdata(z)[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
    seen=hits=0
    ap=top=0.
    for value in np.unique(z)[::-1]:
        group=z==value
        size=int(group.sum())
        found=y[group].sum()
        top+=min(max(k-seen,0),size)*found/size
        seen+=size
        hits+=found
        ap+=found*hits/seen/len(pos)
    return {"auroc":float(auc),"average_precision":float(ap),"precision_at_k":float(top/k),"recall_at_k":float(top/len(pos))}
