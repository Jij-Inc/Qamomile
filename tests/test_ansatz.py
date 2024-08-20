from qamomile.core.ansatz.efficient_su2 import EfficientSU2
import pytest

def test_su2():
    su2 = EfficientSU2(4, ["ry", "rz"], entanglement="linear", skip_final_rotation_layer=False, reps=2)
    ansatz = su2.get_ansatz()
    assert ansatz.num_qubits == 4
    assert len(ansatz.get_parameters()) == 8 * 3
    assert len(ansatz.gates) == 8 * 3 + 6

    su2 = EfficientSU2(3, ["rx", "rz"], entanglement="linear",skip_final_rotation_layer=True,  reps=4)
    ansatz = su2.get_ansatz()
    assert ansatz.num_qubits == 3
    assert len(ansatz.get_parameters()) == 6 * 4
    assert len(ansatz.gates) == 6 * 4 + 2 * 4

def test_su2_error():
    with pytest.raises(NotImplementedError, match = "Entanglement type not implemented"):
        su2 = EfficientSU2(4, ["ry", "rz"], entanglement="full", skip_final_rotation_layer=False, reps=2)
        ansatz = su2.get_ansatz()

    with pytest.raises(NotImplementedError, match = "Gate not implemented" ):
        su2 = EfficientSU2(4, ["y", "rz"], entanglement="linear", skip_final_rotation_layer=False, reps=2)
        ansatz = su2.get_ansatz()