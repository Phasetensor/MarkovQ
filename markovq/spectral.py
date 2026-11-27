"""Undirected Laplacians, Hermitian graph generators and spectral subspaces."""
from typing import NamedTuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from .chains import matrix, count


def graph_operators(adjacency):
    a = matrix(adjacency)
    if a.min()<0:
        raise ValueError("negative graph weight")
    is_sparse = sparse.issparse(a)
    degrees = np.asarray(a.sum(axis=1)).ravel()
    diag = sparse.diags if is_sparse else np.diag
    degree = diag(degrees)
    lap = degree-a
    scale = np.zeros_like(degrees)
    positive = degrees>0
    scale[positive] = degrees[positive]**-.5
    return degree,lap,diag(scale) @ lap @ diag(scale)


def hermitian(a):
    a = sparse.csr_matrix(a,dtype=complex) if sparse.issparse(a) else np.asarray(a,dtype=complex)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not a.shape[0]:
        raise ValueError("expected square Hermitian operator")
    values = a.data if sparse.issparse(a) else a
    if not np.isfinite(values).all():
        raise ValueError("nonfinite operator")
    diff = a-a.conj().T
    if np.max(np.abs(diff.data if sparse.issparse(a) else diff),initial=0)>1e-12:
        raise ValueError("non-Hermitian operator")
    return a


def hamiltonian(adjacency,kind="adjacency",gamma=1.):
    if not np.isreal(gamma) or not np.isfinite(gamma):
        raise ValueError("invalid coupling")
    if kind=="adjacency":
        return gamma*hermitian(adjacency)
    if kind=="laplacian":
        return gamma*hermitian(graph_operators(adjacency)[1])
    raise ValueError("unknown convention")


class Spectrum(NamedTuple):
    values: np.ndarray
    vectors: np.ndarray
    residual: float
    orthogonality_error: float

    def projectors(self,tol=1e-9):
        if not np.isfinite(tol) or tol<0:
            raise ValueError("invalid degeneracy tolerance")
        groups = np.split(np.arange(len(self.values)),np.flatnonzero(np.diff(self.values)>tol)+1)
        return [(self.values[g[0]],self.vectors[:,g] @ self.vectors[:,g].conj().T) for g in groups]


def spectrum(operator,k=None):
    a = hermitian(operator)
    if k is None:
        values,vectors = np.linalg.eigh(a.toarray() if sparse.issparse(a) else a)
    else:
        if not 0<count(k)<a.shape[0]-1:
            raise ValueError("partial spectrum requires 0 < k < n-1")
        # Hermitian real operators use the symmetric Lanczos solver.
        real = np.all((a.data if sparse.issparse(a) else a).imag==0)
        values,vectors = eigsh(sparse.csr_matrix(a.real if real else a),k=k,which="SA")
        order = values.argsort()
        values,vectors = values[order],vectors[:,order]
    return Spectrum(values,vectors,float(np.linalg.norm(a @ vectors-vectors*values)),
                    float(np.linalg.norm(np.einsum('ij,ik->jk',vectors.conj(),vectors)-np.eye(len(values)))))


def spectral_diagnostics(operator, tolerance=1e-10):
    """Full-spectrum adjacent gaps and numerical 2-norm condition number."""
    if not np.isfinite(tolerance) or tolerance<0:
        raise ValueError("invalid singularity tolerance")
    values = spectrum(operator).values
    magnitudes = abs(values)
    return np.diff(values), (np.inf if magnitudes.min()<=tolerance else magnitudes.max()/magnitudes.min())
