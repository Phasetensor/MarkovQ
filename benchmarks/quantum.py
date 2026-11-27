"""Deterministic reference suite for milestones C–E, including 10,000 measured paths."""
import json
import platform
from time import perf_counter

import numpy as np
import scipy
from scipy import linalg, sparse
from scipy.stats import binom

from markovq import quantum as q


def spreading():
    rows = []
    for steps in (8, 16, 32, 64, 128):
        x, prob = q.line_distribution(steps)
        k = np.arange(steps + 1)
        classical = binom.pmf(k, steps, 0.5)
        rows.append({
            "steps": steps,
            "classical_msd": float(classical @ ((2*k-steps)**2)),
            "quantum_msd": float(prob @ (x*x)),
            "quantum_symmetry_error": float(np.max(abs(prob-prob[::-1]))),
            "quantum_origin_probability": float(prob[steps]),
        })
    return rows


def graph_evolution():
    rows = []
    for n in (4, 9, 16):
        path = np.diag(np.ones(n-1),1)
        path += path.T.copy()
        cycle = path.copy()
        cycle[0,-1] = cycle[-1,0] = 1
        star = np.zeros((n,n))
        star[0,1:] = star[1:,0] = 1
        side = int(np.sqrt(n))
        line = np.diag(np.ones(side-1),1)
        line += line.T.copy()
        grid = np.kron(np.eye(side),line)+np.kron(line,np.eye(side))
        upper = np.triu(np.random.default_rng(610+n).integers(0,2,(n,n)),1)
        for name,h in [("path",path),("cycle",cycle),("star",star),("complete",np.ones((n,n))-np.eye(n)),("grid",grid),("random",upper+upper.T)]:
            for time in (0.25,2.0):
                start = perf_counter()
                ref = q.continuous_walk(h,np.eye(n)[0],time,"dense")
                krylov = q.continuous_walk(sparse.csr_matrix(h),np.eye(n)[0],time)
                spectral = q.continuous_walk(h,np.eye(n)[0],time,"spectral")
                rows.append({"graph":name,"n":n,"time":time,"krylov_l2_error":float(np.linalg.norm(ref-krylov)),
                             "spectral_l2_error":float(np.linalg.norm(ref-spectral)),"seconds_all_methods":perf_counter()-start})
    return rows


def monitored():
    # Repeated projective measurements. A no-click outcome prepares the same
    # conditional state for all surviving trajectories at a given step.
    u = linalg.block_diag(q.hadamard(),1)
    initial = np.array([1,0,1])/np.sqrt(2)
    v = initial.copy()
    rng = np.random.default_rng(501)
    n = 10_000
    alive = np.ones(n,dtype=bool)
    observations = np.zeros(13)
    for step in range(1,13):
        raw = u@v
        click = abs(raw[0])**2
        detected = alive & (rng.random(n)<click)
        observations[step] = detected.sum()/n
        alive[detected] = False
        raw[0] = 0
        v = raw/np.linalg.norm(raw)
    target=q.target_projector(3,[0])
    exact=q.first_detection(u,initial,target,12)
    eventual=q.eventual_detection(u,initial,target)
    return {"trajectories":n,"seed":501,"empirical_mass":observations.tolist(),"reference_mass":exact.first.tolist(),
            "empirical_cumulative":float(1-alive.mean()),"cumulative_standard_error":float(np.sqrt(alive.mean()*(1-alive.mean())/n)),
            "reference_cumulative":float(1-exact.survival[-1]),"eventual_probability":eventual.probability,
            "conditional_mean_attempts":eventual.conditional_mean}


def run():
    rows = graph_evolution()
    p = np.array([[.75,.25],[.25,.75]])
    sz = q.szegedy(p)
    mix = q.quantum_mixing(q.hadamard(),[1,0],128)
    renewal = q.recurrence(q.hadamard(),[1,0],32)
    times=np.linspace(0,2,11)
    zero=np.zeros((2,2))
    damp=[np.array([[0,1],[0,0]])]
    exact=q.evolve_density(zero,np.diag([0,1]),times,damp)
    numerical=q.evolve_density(zero,np.diag([0,1]),times,damp,"ode")
    v=np.ones(4)/2
    g=np.ones((4,4))-np.eye(4)
    discrete=q.amplitude_amplification(v,[0],[0,1,2])
    analog=q.continuous_search(g,.25,v,[0],[0,np.pi/2,np.pi])
    observed,uniform=q.classical_search(g,abs(v)**2,[0],2)
    return {"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
            "spreading":spreading(),"graph_evolution":rows,"monitored":monitored(),
            "szegedy_unitarity_error":float(np.linalg.norm(sz.operator.T@sz.operator-np.eye(4))),
            "renewal_error":float(np.max(abs(abs(renewal.first_amplitudes)**2-renewal.monitored.first))),
            "mixing_tv_T128":mix.tv_to_limit,
            "lindblad":{"exponential_vs_ode":float(abs(exact.states-numerical.states).max()),"min_eigenvalue":float(exact.minimum_eigenvalue.min()),
                        "max_trace_error":float(exact.trace_error.max())},
            "search":{"marked":0,"vertices":4,"discrete_queries":discrete.discrete_queries.tolist(),"discrete_success":discrete.success.tolist(),
                      "analog_oracle_times":analog.oracle_time.tolist(),"analog_success":analog.success.tolist(),"classical_vertex_inspections":[1,2,3],
                      "classical_walk_success":observed.tolist(),"uniform_query_success":uniform.tolist()}}


if __name__=="__main__":
    print(json.dumps(run(),indent=2))
