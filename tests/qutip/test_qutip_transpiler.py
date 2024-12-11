import pytest
import qamomile.core.operator as qm_o
from qamomile.qutip.transpiler import QuTiPTranspiler
from qutip import Qobj, sigmaz, sigmax, sigmay, tensor


@pytest.fixture
def transpiler():
    return QuTiPTranspiler()


def test_transpile_hamiltonian(transpiler):
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)
    qutip_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    assert isinstance(qutip_hamiltonian, Qobj)
    assert qutip_hamiltonian.shape == (
        2**hamiltonian.num_qubits,
        2**hamiltonian.num_qubits,
    )

    hamiltonian2 = qm_o.Hamiltonian()
    hamiltonian2 += qm_o.X(0) * qm_o.Y(1) + qm_o.Z(0) * qm_o.X(1)
    qutip_hamiltonian2 = transpiler.transpile_hamiltonian(hamiltonian2)
    H = tensor([sigmax(), sigmay()]) + tensor([sigmaz(), sigmax()])
    assert H == qutip_hamiltonian2
