"""Sparse isometries for Szegedy reflections in the product state space."""
from __future__ import annotations
from typing import NamedTuple, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from .._types import MatrixInput
from ..chains import count,distribution,validate_transition_matrix
from .walks import ket


class ImplicitWalk(NamedTuple):
    A: sparse.csr_matrix
    B: sparse.csr_matrix
    operator: LinearOperator

    def coherent(self,law: ArrayLike) -> np.ndarray:
        return self.A@np.sqrt(distribution(law,self.A.shape[1]))

    def evolve(self,initial: ArrayLike,steps: int) -> np.ndarray:
        v=ket(initial,self.operator.shape[0])
        for _ in range(count(steps)):
            v=self.operator@v
        return v


def implicit_szegedy(transition: MatrixInput,reverse: Optional[MatrixInput]=None) -> ImplicitWalk:
    p=sparse.coo_matrix(validate_transition_matrix(transition))
    q=p if reverse is None else sparse.coo_matrix(validate_transition_matrix(reverse))
    if p.shape!=q.shape:
        raise ValueError("kernel dimensions differ")
    n=p.shape[0]
    def embedding(kernel,swap):
        row=kernel.col*n+kernel.row if swap else kernel.row*n+kernel.col
        return sparse.csc_matrix((np.sqrt(kernel.data),(row,kernel.row)),shape=(n*n,n))
    a,b=embedding(p,False),embedding(q,True)
    def reflect(m,v):
        return 2*(m@(m.T@v))-v
    op=LinearOperator((n*n,n*n),matvec=lambda v:reflect(b,reflect(a,v)),rmatvec=lambda v:reflect(a,reflect(b,v)),dtype=float)
    return ImplicitWalk(a,b,op)


def dense_szegedy_bytes(n: int) -> int:
    n=count(n)
    return 8*(5*n**4+2*n**3+n*n)
