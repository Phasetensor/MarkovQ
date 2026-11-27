import numpy as np
from numpy.testing import assert_allclose

from markovq import quantum as q


def test_qiskit_coined_operator():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    # q2 is coin. For a move modulo four the least significant bit
    # always flips; the high bit flips iff low-bit equals coin.
    circuit=QuantumCircuit(3)
    circuit.h(2)
    circuit.cx(0,1)
    circuit.cx(2,1)
    circuit.x(1)
    circuit.x(0)
    assert_allclose(Operator(circuit).data,q.cycle_walk(4).operator,atol=1e-13)


def test_qutip_driven_dissipator():
    import qutip as qt
    times=np.linspace(0,1.7,12)
    h=np.array([[.2,.3j],[-.3j,-.2]])
    c=np.array([[0,.8],[0,0]])
    rho=np.array([[.5,.5j],[-.5j,.5]])
    ref=qt.mesolve(qt.Qobj(h),qt.Qobj(rho),times,[qt.Qobj(c)],options={"atol":1e-12,"rtol":1e-10})
    states=np.array([x.full() for x in ref.states])
    assert_allclose(q.evolve_density(h,rho,times,[c]).states,states,atol=1e-9)
