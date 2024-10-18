from qamomile.core.ansatz.efficient_su2 import create_efficient_su2_circuit
import pytest

def test_su2():
    ansatz = create_efficient_su2_circuit(4, ["ry", "rz"], entanglement="linear", skip_final_rotation_layer=False, reps=2)
   
    assert ansatz.num_qubits == 4
    assert len(ansatz.get_parameters()) == 8 * 3
    assert len(ansatz.gates) == 8 * 3 + 6

    ansatz = create_efficient_su2_circuit(3, ["rx", "rz"], entanglement="reverse_linear",skip_final_rotation_layer=True,  reps=4)
    
    assert ansatz.num_qubits == 3
    assert len(ansatz.get_parameters()) == 6 * 4
    assert len(ansatz.gates) == 6 * 4 + 2 * 4

    ansatz = create_efficient_su2_circuit(3, ["rx", "rz"], entanglement="circular",skip_final_rotation_layer=True,  reps=4)
    
    assert ansatz.num_qubits == 3
    assert len(ansatz.get_parameters()) == 6 * 4
    assert len(ansatz.gates) == 6 * 4 + 3 * 4

    ansatz = create_efficient_su2_circuit(3, ["rx", "rz"], entanglement="full",skip_final_rotation_layer=True,  reps=4)
    
    assert ansatz.num_qubits == 3
    assert len(ansatz.get_parameters()) == 6 * 4
    assert len(ansatz.gates) == 6 * 4 + 3 * 4

def test_su2_error():
    with pytest.raises(NotImplementedError, match = "Entanglement type sca not implemented"):
        su2 = create_efficient_su2_circuit(4, ["ry", "rz"], entanglement="sca", skip_final_rotation_layer=False, reps=2)
        

    with pytest.raises(NotImplementedError, match = "Gate y not implemented" ):
        su2 = create_efficient_su2_circuit(4, ["y", "rz"], entanglement="linear", skip_final_rotation_layer=False, reps=2)
        