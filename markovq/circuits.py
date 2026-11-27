"""Qiskit walk circuits. Position occupies the least significant qubits."""
from __future__ import annotations
from time import perf_counter
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike
from .chains import count
from .quantum.walks import ket,unitary

def cycle_circuit(size: int,steps: int=1):
    from qiskit import QuantumCircuit
    size=count(size)
    if size<2 or size&(size-1):
        raise ValueError("cycle size must be a power of two >= 2")
    m=size.bit_length()-1
    circuit=QuantumCircuit(m+1)
    for _ in range(count(steps)):
        circuit.h(m)
        for increment in (True,False):
            if not increment:
                circuit.x(m)
                circuit.x(range(m))
            for target in reversed(range(m)):
                controls=[m,*range(target)]
                if len(controls)==1:
                    circuit.cx(m,target)
                else:
                    circuit.mcx(controls,target)
            if not increment:
                circuit.x(range(m))
                circuit.x(m)
    return circuit

def graph_circuit(operator: ArrayLike):
    """Exact small-graph embedding; unused computational states evolve by identity."""
    from qiskit import QuantumCircuit
    u=unitary(operator)
    m=max(1,(len(u)-1).bit_length())
    if m>8:
        raise ValueError("dense synthesis supports at most eight logical qubits")
    full=np.eye(2**m,dtype=complex)
    full[:len(u),:len(u)]=u
    circuit=QuantumCircuit(m)
    circuit.unitary(full,range(m))
    return circuit

def simulate(circuit,initial: ArrayLike,*,seed: int,shots: int=0,backend: Optional[object]=None) -> dict:
    """Ideal local statevector and independent multinomial shots; optional target compilation."""
    from qiskit import QuantumCircuit,transpile
    from qiskit.quantum_info import Statevector
    started=perf_counter()
    n=circuit.num_qubits
    v=ket(initial,2**n)
    prepared=QuantumCircuit(n)
    prepared.prepare_state(v,range(n))
    prepared.compose(circuit,inplace=True)
    probabilities=Statevector.from_instruction(prepared).probabilities()
    measured=prepared.measure_all(inplace=False)
    kwargs={"basis_gates":["rz","sx","x","cx"]} if backend is None else {"backend":backend}
    compiled=transpile(measured,seed_transpiler=count(seed),optimization_level=1,**kwargs)
    counts=np.random.default_rng(seed).multinomial(count(shots),probabilities)
    return {"probabilities":probabilities,"counts":counts,"logical_qubits":n,"ancillary_qubits":compiled.num_qubits-n,
        "depth":compiled.depth(),"two_qubit_gates":sum(len(inst.qubits)==2 and inst.operation.name!="barrier" for inst in compiled.data),
        "shots":shots,"provider_jobs":0,"runtime_s":perf_counter()-started,"evidence":"ideal circuit simulation"}
