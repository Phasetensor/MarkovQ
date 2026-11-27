"""Offline PhaseTensor pipeline for binary readout on the two position qubits."""
from __future__ import annotations
from dataclasses import asdict
from datetime import timedelta
from importlib.metadata import version
import numpy as np
from .chains import count
from .circuits import cycle_circuit,simulate
from .noise import readout

def phase_experiment(records,at,*,seed: int,budget: int=10000,steps: int=3) -> dict:
    """Use synthetic directional calibration records and independent final samples."""
    from bornlens import Sample
    from bornlens.advanced import compare_many,distribution_compare,sequential_compare
    from calibmap import as_of
    from calibmap.history import stability
    from calibmap.planning import readiness
    from calibmap.readout import Binary,interval
    from shotweave.experiment import run_experiment
    from shotweave.measurements import Readout
    records=tuple(records)
    snap=as_of(records,at)
    if not snap or any(r.kind!="synthetic" for r in snap):
        raise ValueError("expected an explicitly synthetic calibration history")
    lookup={(r.metric,r.qubits):r for r in snap}
    expected={(name,(bit,)) for name in ("false_one","false_zero") for bit in (0,1)}
    if len(lookup)!=len(snap) or not expected<=lookup.keys():
        raise ValueError("missing or ambiguous directional readout calibration")
    bounds={r.identity(r.unit):.15 for r in snap}
    decision,_=readiness(records,at,timedelta(days=1),bounds)
    if decision!="READY":
        raise ValueError("calibration readiness criteria failed")
    history=stability(records,at,timedelta(days=7),timedelta(days=2))
    models=[Binary(lookup["false_one",(i,)].value,lookup["false_zero",(i,)].value) for i in (0,1)]
    calibration_intervals={}
    for key in sorted(expected):
        r=lookup[key]
        if not r.sample_size or not np.isclose(r.value*r.sample_size,round(r.value*r.sample_size)):
            raise ValueError("require empirical binary calibration counts")
        calibration_intervals[str(key)]=interval(round(r.value*r.sample_size),r.sample_size)
    circuit=cycle_circuit(4,steps)
    initial=np.zeros(8,complex)
    initial[0]=1
    sim=simulate(circuit,initial,seed=seed)
    ideal=sim.pop("probabilities").reshape(2,4).sum(axis=0)
    sim.pop("counts")
    assignment=np.kron(models[1].matrix,models[0].matrix).T
    observed=readout(ideal,assignment)
    ones=np.array([observed[[1,3]].sum(),observed[[2,3]].sum()])
    ideal_ones=np.array([ideal[[1,3]].sum(),ideal[[2,3]].sum()])
    pilot,ref,dist_a,dist_b=[np.random.default_rng(s) for s in np.random.SeedSequence(count(seed)).spawn(4)]
    estimate=run_experiment(budget,(.5,.5),lambda shots:tuple(map(int,pilot.binomial(shots,ones))),
                            "noise-aware",pilot_size=32,readout=tuple(Readout(m.e0,m.e1) for m in models))
    pairs={}
    for i,label in enumerate(("IIZ","IZI")):
        n=estimate.final_shots[i]
        k=estimate.successes[i]
        j=int(ref.binomial(n,ideal_ones[i]))
        pairs[label]=(Sample(label,n-j,j),Sample(label,n-k,k))
    simultaneous=compare_many(pairs,tolerance=.1)
    sequential=sequential_compare(*pairs["IIZ"],tolerance=.1)
    def histogram(rng,p):
        return {format(i,"02b"):int(k) for i,k in enumerate(rng.multinomial(budget,p))}
    comparison=distribution_compare(histogram(dist_a,ideal),histogram(dist_b,observed),seed=seed,draws=10000)
    return {"evidence":"synthetic calibration and simulated sampling","readiness":decision,
        "calibration_intervals":calibration_intervals,"stability_epochs":{str(k):v["epochs"] for k,v in history.items()},
        "ideal":ideal.tolist(),"observed":observed.tolist(),"circuit":sim,"pilot_shots":list(estimate.pilot_shots),
        "final_shots":list(estimate.final_shots),"final_successes":list(estimate.successes),"corrected_average_bit_probability":estimate.value,
        "total_simulated_shots":estimate.total_shots+sum(estimate.final_shots)+2*budget,"pilot_and_final_budget":estimate.total_shots,
        "pauli_comparisons":{k:asdict(v) for k,v in simultaneous.items()},"sequential":asdict(sequential),
        "position_distribution_test":comparison,"versions":{name:version(name) for name in ("calibmap","shotweave","bornlens","qiskit")}}
