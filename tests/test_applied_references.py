import numpy as np
import pytest
from numpy.testing import assert_allclose

from markovq import circuits
from markovq.quantum import cycle_walk


@pytest.mark.parametrize("size", [2, 4, 8, 16])
def test_cycle_circuit(size):
    from qiskit.quantum_info import Operator

    c = circuits.cycle_circuit(size, 2)
    assert Operator(c).equiv(
        Operator(np.linalg.matrix_power(cycle_walk(size).operator, 2))
    )
    v = np.zeros(2 * size)
    v[0] = 1
    result = circuits.simulate(c, v, seed=8, shots=300)
    assert_allclose(
        result["probabilities"], abs(cycle_walk(size).evolve(v, 2)) ** 2, atol=1e-12
    )
    assert result["counts"].sum() == 300 and result["depth"] > 0


def test_graph_padding_and_rejections():
    from qiskit.quantum_info import Operator

    u = np.roll(np.eye(3), 1, axis=0)
    c = circuits.graph_circuit(u)
    assert_allclose(Operator(c).data[:3, :3], u)
    with pytest.raises(ValueError):
        circuits.cycle_circuit(3)


def test_reference_channels():
    from qiskit.quantum_info import DensityMatrix, Kraus

    from markovq.noise import apply_channel, kraus_model

    rho = np.array([[0.4, 0.2j], [-0.2j, 0.6]])
    for kind in (
        "depolarizing",
        "dephasing",
        "amplitude-damping",
        "bit-flip",
        "phase-flip",
    ):
        ops = kraus_model(kind, 0.31)
        ref = DensityMatrix(rho).evolve(Kraus(list(ops)))
        assert_allclose(apply_channel(rho, ops), ref.data, atol=1e-12)


def test_local_backend_execution():
    from qiskit.providers.basic_provider import BasicSimulator

    backend = BasicSimulator()
    c = circuits.cycle_circuit(4, 0)
    result = circuits.execute_backend(c, np.eye(8)[2], backend, shots=32, seed=14)
    assert result["counts"] == {"010": 32}
    assert result["provider_jobs"] == 1 and result["job_id"]
    with pytest.raises(ValueError):
        circuits.execute_backend(c, np.eye(8)[2], backend, shots=0, seed=14)


def test_channel_against_qutip_master_equation():
    import qutip as qt

    from markovq.noise import apply_channel, kraus_model

    rho = qt.ket2dm((qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit())
    for name, collapse in [
        ("amplitude-damping", qt.destroy(2)),
        ("dephasing", qt.sigmaz() / np.sqrt(2)),
    ]:
        p = 0.37
        duration = -np.log1p(-p)
        ref = (duration * qt.liouvillian(0 * qt.sigmaz(), [collapse])).expm()
        expected = qt.vector_to_operator(ref * qt.operator_to_vector(rho)).full()
        assert_allclose(apply_channel(rho.full(), kraus_model(name, p)), expected)


@pytest.mark.parametrize("level", [0, 2])
def test_layout_resources(level):
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit import AncillaRegister, QuantumRegister
    from qiskit.transpiler import CouplingMap

    q = QuantumRegister(3)
    a = AncillaRegister(1)
    c = QuantumCircuit(q, a)
    c.cx(q[0], q[1])
    c.x(a[0])
    line = CouplingMap.from_line(5)
    compiled = transpile(
        c,
        coupling_map=line,
        initial_layout=[0, 4, 2, 3],
        basis_gates=["u", "cx"],
        routing_method="basic",
        optimization_level=level,
    )
    r = circuits.circuit_resources(c, compiled)
    active = {compiled.find_bit(bit).index for op in compiled.data for bit in op.qubits}
    assert r["logical_qubits"] == r["represented_logical_qubits"] == 3
    assert r["ancillary_qubits"] == len(active - {0, 4, 2})
    assert r["compiled_width"] == 5
    ident = QuantumCircuit(2)
    r = circuits.circuit_resources(ident, ident)
    assert r["physical_qubits_addressed"] == r["ancillary_qubits"] == 0


def test_unknown_layout_and_execution_accounting():
    from qiskit import QuantumCircuit
    from qiskit.providers.basic_provider import BasicSimulator

    r = circuits.circuit_resources(QuantumCircuit(2), QuantumCircuit(4))
    assert r["represented_logical_qubits"] is None and r["ancillary_qubits"] is None
    step = circuits.graph_circuit(np.eye(3))
    r = circuits.execute_backend(step, [1, 0, 0, 0], BasicSimulator(), shots=8, seed=7)
    assert r["requested_shots"] == r["returned_shots"] == 8
    assert r["measurement_count"] == 2
