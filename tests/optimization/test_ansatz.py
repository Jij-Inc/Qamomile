from qamomile.circuit.estimator import count_gates
import sympy as sp
from qamomile.optimization.ansatz.efficient_su2 import (
    create_efficient_su2_circuit,
    num_entangling_gates,
    num_parameters,
)
import pytest


def _assert_gate_counts(
    num_qubits: int,
    rotation_blocks: list[str],
    entanglement: str,
    skip_final_rotation_layer: bool,
    reps: int,
) -> None:
    ansatz = create_efficient_su2_circuit(
        num_qubits,
        rotation_blocks,
        entanglement=entanglement,
        skip_final_rotation_layer=skip_final_rotation_layer,
        reps=reps,
    )
    counts = count_gates(ansatz.block)
    q_dim0 = sp.Symbol("q_dim0")
    single_qubit = counts.single_qubit.subs(q_dim0, num_qubits)
    two_qubit = counts.two_qubit.subs(q_dim0, num_qubits)
    total = counts.total.subs(q_dim0, num_qubits)
    expected_single = num_parameters(
        num_qubits,
        rotation_blocks=rotation_blocks,
        skip_final_rotation_layer=skip_final_rotation_layer,
        reps=reps,
    )
    expected_two = num_entangling_gates(num_qubits, entanglement=entanglement) * reps

    assert single_qubit == expected_single
    assert two_qubit == expected_two
    assert total == expected_single + expected_two


def test_su2():
    _assert_gate_counts(
        4,
        ["ry", "rz"],
        entanglement="linear",
        skip_final_rotation_layer=False,
        reps=2,
    )

    _assert_gate_counts(
        3,
        ["rx", "rz"],
        entanglement="reverse_linear",
        skip_final_rotation_layer=True,
        reps=4,
    )

    _assert_gate_counts(
        3,
        ["rx", "rz"],
        entanglement="circular",
        skip_final_rotation_layer=True,
        reps=4,
    )

    _assert_gate_counts(
        3,
        ["rx", "rz"],
        entanglement="full",
        skip_final_rotation_layer=True,
        reps=4,
    )

def test_su2_error():
    with pytest.raises(NotImplementedError, match = "Entanglement type sca not implemented"):
        su2 = create_efficient_su2_circuit(4, ["ry", "rz"], entanglement="sca", skip_final_rotation_layer=False, reps=2)
        

    with pytest.raises(NotImplementedError, match = "Gate y not implemented" ):
        su2 = create_efficient_su2_circuit(4, ["y", "rz"], entanglement="linear", skip_final_rotation_layer=False, reps=2)
        
