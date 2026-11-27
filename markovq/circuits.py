"""Qiskit walk circuits. Position occupies the least significant qubits."""

from __future__ import annotations

from time import perf_counter
from typing import Any, Optional

import numpy as np
from numpy.typing import ArrayLike

from .chains import count
from .quantum.walks import ket, unitary


def cycle_circuit(size: int, steps: int = 1):
    from qiskit import QuantumCircuit

    size = count(size)
    if size < 2 or size & (size - 1):
        raise ValueError("cycle size must be a power of two >= 2")
    m = size.bit_length() - 1
    circuit = QuantumCircuit(m + 1)
    for _ in range(count(steps)):
        circuit.h(m)
        for increment in (True, False):
            if not increment:
                circuit.x(m)
                circuit.x(range(m))
            for target in reversed(range(m)):
                controls = [m, *range(target)]
                if len(controls) == 1:
                    circuit.cx(m, target)
                else:
                    circuit.mcx(controls, target)
            if not increment:
                circuit.x(range(m))
                circuit.x(m)
    return circuit


def graph_circuit(operator: ArrayLike):
    """Exact small-graph embedding; unused computational states evolve by identity."""
    from qiskit import QuantumCircuit

    u = unitary(operator)
    m = max(1, (len(u) - 1).bit_length())
    if m > 8:
        raise ValueError("dense synthesis supports at most eight logical qubits")
    full = np.eye(2**m, dtype=complex)
    full[: len(u), : len(u)] = u
    circuit = QuantumCircuit(m)
    circuit.unitary(full, range(m))
    return circuit


def circuit_resources(original, compiled, backend=None) -> dict:
    """Use initial placement to distinguish input-data wires from workspace."""
    operations = [
        op for op in compiled.data if not getattr(op.operation, "_directive", False)
    ]
    used = {compiled.find_bit(q).index for op in operations for q in op.qubits}
    data = [i for i, q in enumerate(original.qubits) if q not in original.ancillas]
    mapping = None
    if compiled.layout is not None:
        mapping = compiled.layout.initial_index_layout(filter_ancillas=True)
    elif compiled.qubits == original.qubits:
        mapping = list(range(original.num_qubits))
    placed = (
        None
        if mapping is None or len(mapping) != original.num_qubits
        else {mapping[i] for i in data}
    )
    return {
        "logical_qubits": len(data),
        "represented_logical_qubits": None if placed is None else len(placed),
        "physical_qubits_addressed": len(used),
        "addressed_wire_indices": sorted(used),
        "ancillary_qubits": None if placed is None else len(used - placed),
        "compiled_width": compiled.num_qubits,
        "backend_capacity": getattr(backend, "num_qubits", None),
        "measurement_count": sum(op.operation.name == "measure" for op in operations),
        "depth": compiled.depth(),
        "two_qubit_gates": sum(len(op.qubits) == 2 for op in operations),
    }


def simulate(
    circuit,
    initial: ArrayLike,
    *,
    seed: int,
    shots: int = 0,
    backend: Optional[object] = None,
) -> dict:
    """Ideal local statevector and independent multinomial shots; optional target compilation."""
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector

    started = perf_counter()
    n = circuit.num_qubits
    v = ket(initial, 2**n)
    prepared = QuantumCircuit(list(circuit.qubits))
    prepared.prepare_state(v, range(n))
    prepared.compose(circuit, inplace=True)
    probabilities = Statevector.from_instruction(prepared).probabilities()
    measured = prepared.measure_all(inplace=False)
    kwargs: dict[str, Any] = (
        {"basis_gates": ["rz", "sx", "x", "cx"]}
        if backend is None
        else {"backend": backend}
    )
    compiled = transpile(
        measured, seed_transpiler=count(seed), optimization_level=1, **kwargs
    )
    counts = np.random.default_rng(seed).multinomial(count(shots), probabilities)
    return {
        "probabilities": probabilities,
        "counts": counts,
        **circuit_resources(circuit, compiled, backend),
        "requested_shots": shots,
        "returned_shots": int(counts.sum()),
        "shots": shots,
        "provider_jobs": 0,
        "runtime_s": perf_counter() - started,
        "evidence": "ideal circuit simulation",
    }


def execute_backend(circuit, initial: ArrayLike, backend, *, shots: int, seed: int):
    """Submit a measured circuit to a caller-selected backend supporting Backend.run.

    The seed controls compilation. Backend sampling and queue policies belong to the
    caller's provider. Runtime includes compilation and waiting for the result.
    """
    from qiskit import QuantumCircuit, transpile

    shots = count(shots)
    if not shots:
        raise ValueError("backend execution requires positive shots")
    started = perf_counter()
    n = circuit.num_qubits
    prepared = QuantumCircuit(list(circuit.qubits))
    prepared.prepare_state(ket(initial, 2**n), range(n))
    prepared.compose(circuit, inplace=True)
    prepared.measure_all()
    compiled = transpile(prepared, backend, seed_transpiler=count(seed))
    job = backend.run(compiled, shots=shots)
    counts = job.result().get_counts()
    if sum(counts.values()) != shots:
        raise ValueError("backend returned an inconsistent shot count")
    return {
        "backend": backend.name,
        "job_id": job.job_id(),
        "counts": dict(counts),
        "shots": shots,
        "provider_jobs": 1,
        **circuit_resources(circuit, compiled, backend),
        "requested_shots": shots,
        "returned_shots": sum(counts.values()),
        "evidence": "caller-selected backend execution",
        "runtime_s": perf_counter() - started,
    }
