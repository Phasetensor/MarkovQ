"""Finite closed-system quantum walks. States are column vectors; hbar = 1."""
from __future__ import annotations

from typing import NamedTuple, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.sparse.linalg import expm_multiply
from .._types import MatrixInput
from ..chains import count, distribution
from ..passage import dense
from ..spectral import hermitian

TOL = 1e-10


def square(a: ArrayLike) -> np.ndarray:
    out=np.asarray(a,dtype=complex)
    if out.ndim!=2 or not len(out) or out.shape[0]!=out.shape[1] or not np.isfinite(out).all():
        raise ValueError("expected a finite nonempty square matrix")
    return out


def adj(a: np.ndarray) -> np.ndarray:
    return a.conj().T


def unitary(a: ArrayLike) -> np.ndarray:
    u=square(a)
    if not np.allclose(np.einsum('ij,ik->jk',u.conj(),u),np.eye(len(u)),rtol=0,atol=TOL):
        raise ValueError("operator must be unitary")
    return u


def ket(a: ArrayLike,n: int) -> np.ndarray:
    v=np.asarray(a,dtype=complex)
    if v.shape!=(n,) or not np.isclose(np.linalg.norm(v),1,rtol=0,atol=TOL):
        raise ValueError("expected a normalized ket of matching dimension")
    return v


def hadamard() -> np.ndarray:
    return np.array([[1,1],[1,-1]])/np.sqrt(2)


def grover(d: int) -> np.ndarray:
    d=count(d)
    if not d:
        raise ValueError("empty coin space")
    return 2*np.ones((d,d))/d-np.eye(d)


class CoinedWalk(NamedTuple):
    operator: np.ndarray
    position: np.ndarray
    coin: np.ndarray

    def evolve(self,initial: ArrayLike,steps: int) -> np.ndarray:
        v=ket(initial,len(self.operator))
        for _ in range(count(steps)):
            v=np.einsum('ij,j->i',self.operator,v)
        return v

    def marginals(self,state: ArrayLike) -> tuple[np.ndarray,np.ndarray]:
        weights=abs(ket(state,len(self.operator)))**2
        return np.bincount(self.position,weights),np.bincount(self.coin,weights)


def cycle_walk(size: int,coin: Optional[ArrayLike]=None) -> CoinedWalk:
    """Coin-major ordering (left block, right block), moving shift on a cycle."""
    size=count(size)
    c=unitary(hadamard() if coin is None else coin)
    if size<2 or c.shape!=(2,2):
        raise ValueError("requires a cycle of size >= 2 and a two-level coin")
    rows=np.r_[np.roll(np.arange(size),-1),size+np.roll(np.arange(size),1)]
    return CoinedWalk(np.kron(c,np.eye(size))[rows],np.tile(np.arange(size),2),np.repeat([0,1],size))


def line_distribution(steps: int,initial_coin: ArrayLike=(1/np.sqrt(2),1j/np.sqrt(2)),coin: Optional[ArrayLike]=None) -> tuple[np.ndarray,np.ndarray]:
    """Only reachable positions are allocated; no periodic wraparound."""
    steps=count(steps)
    c=unitary(hadamard() if coin is None else coin)
    if c.shape!=(2,2):
        raise ValueError("line walk requires a two-level coin")
    amps=ket(initial_coin,2)[:,None]
    for _ in range(steps):
        mixed=c@amps
        amps=np.zeros((2,mixed.shape[1]+2),complex)
        amps[0,:-2]=mixed[0]
        amps[1,2:]=mixed[1]
    return np.arange(-steps,steps+1),(abs(amps)**2).sum(axis=0)


def graph_walk(adjacency: ArrayLike,coins: Optional[list]=None) -> CoinedWalk:
    """Flip-flop shift on oriented arcs of a simple undirected graph; local port ordering."""
    a=square(adjacency)
    if not np.all((a==0)|(a==1)) or not np.array_equal(a,a.T) or a.diagonal().any() or not a.any(axis=1).all():
        raise ValueError("requires a loop-free undirected simple graph without isolates")
    arcs=list(zip(*np.nonzero(a)))
    lookup={arc:i for i,arc in enumerate(arcs)}
    degrees=np.count_nonzero(a,axis=1)
    blocks=[grover(d) for d in degrees] if coins is None else [unitary(c) for c in coins]
    if len(blocks)!=len(a) or any(c.shape!=(d,d) for c,d in zip(blocks,degrees)):
        raise ValueError("one local coin per vertex with matching degree")
    reverse=[lookup[j,i] for i,j in arcs]
    return CoinedWalk(linalg.block_diag(*blocks)[reverse],np.array([i for i,j in arcs]),np.concatenate([np.arange(d) for d in degrees]))


def continuous_walk(hamiltonian: MatrixInput,initial: ArrayLike,time: float,method: str="krylov") -> np.ndarray:
    h=hermitian(hamiltonian)
    v=ket(initial,h.shape[0])
    if not np.isfinite(time):
        raise ValueError("nonfinite time")
    if method=="krylov":
        return expm_multiply((-1j*time)*h,v)
    a=h if isinstance(h,np.ndarray) else h.toarray()
    if method=="dense":
        return linalg.expm(-1j*time*a)@v
    if method=="spectral":
        w,b=linalg.eigh(a)
        return b@(np.exp(-1j*time*w)*(adj(b)@v))
    raise ValueError("method must be krylov, dense or spectral")


class SzegedyWalk(NamedTuple):
    A: np.ndarray
    B: np.ndarray
    reflection_a: np.ndarray
    reflection_b: np.ndarray
    operator: np.ndarray
    discriminant: np.ndarray

    def coherent(self,law: ArrayLike) -> np.ndarray:
        return self.A@np.sqrt(distribution(law,self.A.shape[1]))


def szegedy(transition: MatrixInput,reverse: Optional[MatrixInput]=None) -> SzegedyWalk:
    """B uses swap-embedded rows of reverse, defaulting to P (swap construction)."""
    p=dense(transition)
    q=p if reverse is None else dense(reverse)
    if q.shape!=p.shape:
        raise ValueError("transition dimensions differ")
    n=len(p)
    a=np.zeros((n*n,n))
    b=np.zeros_like(a)
    for i in range(n):
        a[i*n:(i+1)*n,i]=np.sqrt(p[i])
        b[i::n,i]=np.sqrt(q[i])
    ra=2*a@a.T-np.eye(n*n)
    rb=2*b@b.T-np.eye(n*n)
    return SzegedyWalk(a,b,ra,rb,rb@ra,a.T@b)
