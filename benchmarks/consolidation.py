"""Isolated-process RSS and repeated timing of existing numerical algorithms."""
import contextlib
import io
import json
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy import sparse
from markovq import quantum as q


def trial(kind,n,family):
    edge=sparse.diags(np.ones(n-1),1,shape=(n,n),format="csr")
    h=edge+edge.T
    if family=="cycle":
        h=h.tolil()
        h[0,-1]=h[-1,0]=1
        h=h.tocsr()
    if family=="star":
        h=sparse.coo_matrix((np.ones(n-1),(np.zeros(n-1,int),np.arange(1,n))),shape=(n,n)).tocsr()
        h=h+h.T
    if family=="complete":
        h=sparse.csr_matrix(np.ones((n,n))-np.eye(n))
    if family=="grid":
        side=int(np.sqrt(n))
        a=sparse.diags([1,1],[-1,1],shape=(side,side))
        h=sparse.kronsum(a,a)
    if family=="random":
        r=sparse.random(n,n,density=min(.5,4/n),random_state=504,format="csr")
        h=h+r+r.T
    if family=="weighted":
        h=h*1.75
    v=np.zeros(n)
    v[0]=1
    started=time.perf_counter()
    if kind=="continuous":
        out=q.continuous_walk(h,v,.5)
        elapsed=time.perf_counter()-started
        reference=q.continuous_walk(h,v,.5,"spectral") if n<=64 else q.continuous_walk(h,out,-.5)
        err=np.linalg.norm(reference-(out if n<=64 else v))
        norm=abs(np.linalg.norm(out)-1)
    elif kind.startswith("szegedy"):
        p=sparse.diags(1/np.asarray(h.sum(axis=1)).ravel())@h
        w=q.szegedy(p) if kind=="szegedy-dense" else q.implicit_szegedy(p)
        x=w.coherent(v)
        y=np.einsum("ij,j->i",w.operator,x) if kind=="szegedy-dense" else w.operator@x
        elapsed=time.perf_counter()-started
        back=np.einsum("ij,i->j",w.operator,y) if kind=="szegedy-dense" else w.operator.T@y
        err=np.linalg.norm(back-x)
        norm=abs(np.linalg.norm(y)-1)
    elif kind.startswith("lindblad"):
        method=kind.split("/")[1]
        d=q.evolve_density(h,np.diag(v),[.05],[h*.2],method)
        elapsed=time.perf_counter()-started
        energies,basis=scipy.linalg.eigh(h.toarray())
        delta=energies[:,None]-energies[None,:]
        coefficients=basis.T@v
        ref=np.einsum("ai,ij,bj->ab",basis,np.outer(coefficients,coefficients)*np.exp(-.05j*delta-.001*delta**2),basis)
        err=np.linalg.norm(d.states[0]-ref)
        norm=d.trace_error[0]
    elif kind=="coined":
        x,prob=q.line_distribution(n)
        elapsed=time.perf_counter()-started
        err=np.max(abs(prob-prob[::-1]))
        norm=abs(prob.sum()-1)
    else:
        # Fourier diagonal unitary has exact invariant basis states.
        u=np.eye(n,dtype=complex)
        u[:2,:2]=q.hadamard()
        if kind=="mixing":
            m=q.quantum_mixing(u,v,20)
            elapsed=time.perf_counter()-started
            err=np.linalg.norm(m.average-m.finite_spectral)
            norm=abs(m.average.sum()-1)
        else:
            target=q.target_projector(n,[0])
            d=q.first_detection(u,v,target,20)
            elapsed=time.perf_counter()-started
            err=np.max(abs(d.first-np.r_[0.,.5**np.arange(1,21)]))
            norm=np.max(abs(d.survival+d.first.cumsum()-1))
    return elapsed,float(err),float(norm)


def worker(kind,n,family):
    rows=np.array([trial(kind,n,family) for _ in range(3)])
    peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"method":kind,"size":n,"graph":family,"median_s":float(np.median(rows[:,0])),
            "min_s":float(rows[:,0].min()),"max_s":float(rows[:,0].max()),"peak_rss_mib":peak/(1024**2 if sys.platform=="darwin" else 1024),
            "max_reference_error":float(rows[:,1].max()),"max_conservation_error":float(rows[:,2].max())}


def run():
    cases=[("continuous",n,g) for n in [16,32,64,128,256,512,1024] for g in (["path","cycle","star","complete","grid","random","weighted"] if n==16 else ["weighted"])]
    cases += [(k,n,"line" if k=="coined" else ("embedded Hadamard" if k in ("mixing","detection") else "cycle")) for k,sizes in [
        ("szegedy-dense",[8,16,32]),("szegedy-implicit",[8,16,32,64,256,512,1024]),
        ("lindblad/expm",[4,8,16]),("lindblad/sparse",[4,8,16,32]),("lindblad/matrix-free",[4,8,16,32,64]),
        ("coined",[16,128,1024]),("mixing",[8,16,32]),("detection",[8,16,32])] for n in sizes]
    rows=[]
    for kind,n,family in cases:
        result=subprocess.check_output([sys.executable,"-Werror","-m","benchmarks.consolidation",kind,str(n),family],text=True)
        rows.append(json.loads(result))
    config=io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
    return {"os":platform.platform(),"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
            "numpy_build":config.getvalue(),"random_graph_seed":504,"repetitions":3,"rows":rows}


if __name__=="__main__":
    print(json.dumps(worker(sys.argv[1],int(sys.argv[2]),sys.argv[3]) if len(sys.argv)>1 else run(),indent=2))
