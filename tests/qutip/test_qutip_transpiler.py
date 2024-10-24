import pytest
import qamomile.core.operator as qm_o
from qamomile.qutip.transpiler import QuTiPTranspiler
from qutip import Qobj, sigmaz, sigmax, sigmay

@pytest.fixture
def transpiler():
    return QuTiPTranspiler()

def test_transpile_hamiltonian(transpiler):
    hamiltonian = qm_o.Hamiltonian()
    hamiltonian += qm_o.X(0) * qm_o.Z(1)

    qutip_hamiltonian = transpiler.transpile_hamiltonian(hamiltonian)

    assert isinstance(qutip_hamiltonian, Qobj)
    assert qutip_hamiltonian.shape == (2**hamiltonian.num_qubits, 2**hamiltonian.num_qubits)
    
   
   