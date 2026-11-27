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
