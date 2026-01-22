"""QURI Parts GateEmitter implementation.

This module provides QuriPartsGateEmitter, which implements the GateEmitter
protocol for QURI Parts backends.

QURI Parts uses LinearMappedUnboundParametricQuantumCircuit for parametric
circuits. Angles are specified as dictionaries: {param: coeff, CONST: offset}.
"""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from quri_parts.circuit import (
        LinearMappedUnboundParametricQuantumCircuit,
        Parameter,
    )


class QuriPartsGateEmitter:
    """GateEmitter implementation for QURI Parts.

    Emits individual quantum gates to QURI Parts circuits.

    QURI Parts parametric circuits accept angles in dictionary form:
    {parameter: coefficient, CONST: constant_offset}
    """

    def __init__(self) -> None:
        """Initialize the emitter."""
        self._param_map: dict[str, "Parameter"] = {}
        self._current_circuit: "LinearMappedUnboundParametricQuantumCircuit | None" = (
            None
        )

    def create_circuit(
        self, num_qubits: int, num_clbits: int
    ) -> "LinearMappedUnboundParametricQuantumCircuit":
        """Create a new QURI Parts parametric circuit.

        Note: QURI Parts does not support classical bits in circuits.
        The num_clbits parameter is accepted for interface compatibility
        but is not used.
        """
        import quri_parts.circuit as qp_c

        circuit = qp_c.LinearMappedUnboundParametricQuantumCircuit(num_qubits)
        self._current_circuit = circuit
        self._param_map.clear()
        return circuit

    def create_parameter(self, name: str) -> "Parameter":
        """Create a QURI Parts parameter and register it with the circuit."""
        if self._current_circuit is None:
            raise RuntimeError("create_circuit must be called before create_parameter")

        if name not in self._param_map:
            param = self._current_circuit.add_parameter(name)
            self._param_map[name] = param
        return self._param_map[name]

    def _make_angle_dict(self, angle: float | Any) -> dict[Any, float] | float:
        """Convert angle to QURI Parts format.

        If angle is a float, return it directly.
        If angle is a Parameter, return {parameter: 1.0}.
        """
        if isinstance(angle, (int, float)):
            return float(angle)
        # Assume it's a QURI Parts Parameter
        return {angle: 1.0}

    # Single-qubit gates (no parameters)
    def emit_h(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        circuit.add_H_gate(qubit)

    def emit_x(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        circuit.add_X_gate(qubit)

    def emit_y(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        circuit.add_Y_gate(qubit)

    def emit_z(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        circuit.add_Z_gate(qubit)

    def emit_s(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        circuit.add_S_gate(qubit)

    def emit_t(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", qubit: int
    ) -> None:
        circuit.add_T_gate(qubit)

    # Single-qubit rotation gates
    def emit_rx(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            circuit.add_RX_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRX_gate(qubit, angle_dict)

    def emit_ry(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            circuit.add_RY_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRY_gate(qubit, angle_dict)

    def emit_rz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            circuit.add_RZ_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRZ_gate(qubit, angle_dict)

    def emit_p(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        angle: float | Any,
    ) -> None:
        """Emit Phase gate.

        QURI Parts doesn't have a native Phase gate. We use RZ which differs
        by a global phase (which is physically irrelevant).
        P(θ) = e^(iθ/2) * RZ(θ)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            circuit.add_RZ_gate(qubit, angle_dict)
        else:
            circuit.add_ParametricRZ_gate(qubit, angle_dict)

    # Two-qubit gates
    def emit_cx(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        circuit.add_CNOT_gate(control, target)

    def emit_cz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        circuit.add_CZ_gate(control, target)

    def emit_swap(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit1: int,
        qubit2: int,
    ) -> None:
        circuit.add_SWAP_gate(qubit1, qubit2)

    # Two-qubit rotation gates
    def emit_cp(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-Phase gate using decomposition.

        CP(ctrl, tgt, θ) =
            RZ(tgt, θ/2)
            CNOT(ctrl, tgt)
            RZ(tgt, -θ/2)
            CNOT(ctrl, tgt)
            RZ(ctrl, θ/2)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            half_angle = angle_dict / 2
            circuit.add_RZ_gate(target, half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RZ_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RZ_gate(control, half_angle)
        else:
            # Parametric case
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_ParametricRZ_gate(target, half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRZ_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRZ_gate(control, half_angle_dict)

    def emit_rzz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit1: int,
        qubit2: int,
        angle: float | Any,
    ) -> None:
        """Emit RZZ gate using ParametricPauliRotation."""
        # QURI Parts pauli_ids: 1=X, 2=Y, 3=Z
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            circuit.add_PauliRotation_gate([qubit1, qubit2], [3, 3], angle_dict)
        else:
            circuit.add_ParametricPauliRotation_gate(
                [qubit1, qubit2], [3, 3], angle_dict
            )

    # Three-qubit gates
    def emit_toffoli(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control1: int,
        control2: int,
        target: int,
    ) -> None:
        circuit.add_TOFFOLI_gate(control1, control2, target)

    # Controlled single-qubit gates (decomposition required)
    def emit_ch(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        """Emit controlled-Hadamard gate using decomposition.

        CH(ctrl, tgt) =
            RY(tgt, π/4)
            CNOT(ctrl, tgt)
            RY(tgt, -π/4)
        """
        circuit.add_RY_gate(target, math.pi / 4)
        circuit.add_CNOT_gate(control, target)
        circuit.add_RY_gate(target, -math.pi / 4)

    def emit_cy(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
    ) -> None:
        """Emit controlled-Y gate using decomposition.

        CY(ctrl, tgt) =
            S†(tgt)
            CNOT(ctrl, tgt)
            S(tgt)
        """
        circuit.add_Sdag_gate(target)
        circuit.add_CNOT_gate(control, target)
        circuit.add_S_gate(target)

    def emit_crx(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RX gate using decomposition.

        CRX(ctrl, tgt, θ) =
            RZ(tgt, π/2)
            CNOT(ctrl, tgt)
            RY(tgt, -θ/2)
            CNOT(ctrl, tgt)
            RY(tgt, θ/2)
            RZ(tgt, -π/2)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            half_angle = angle_dict / 2
            circuit.add_RZ_gate(target, math.pi / 2)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RY_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RY_gate(target, half_angle)
            circuit.add_RZ_gate(target, -math.pi / 2)
        else:
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_RZ_gate(target, math.pi / 2)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRY_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRY_gate(target, half_angle_dict)
            circuit.add_RZ_gate(target, -math.pi / 2)

    def emit_cry(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RY gate using decomposition.

        CRY(ctrl, tgt, θ) =
            RY(tgt, θ/2)
            CNOT(ctrl, tgt)
            RY(tgt, -θ/2)
            CNOT(ctrl, tgt)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            half_angle = angle_dict / 2
            circuit.add_RY_gate(target, half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RY_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
        else:
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_ParametricRY_gate(target, half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRY_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)

    def emit_crz(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        control: int,
        target: int,
        angle: float | Any,
    ) -> None:
        """Emit controlled-RZ gate using decomposition.

        CRZ(ctrl, tgt, θ) =
            RZ(tgt, θ/2)
            CNOT(ctrl, tgt)
            RZ(tgt, -θ/2)
            CNOT(ctrl, tgt)
        """
        angle_dict = self._make_angle_dict(angle)
        if isinstance(angle_dict, float):
            half_angle = angle_dict / 2
            circuit.add_RZ_gate(target, half_angle)
            circuit.add_CNOT_gate(control, target)
            circuit.add_RZ_gate(target, -half_angle)
            circuit.add_CNOT_gate(control, target)
        else:
            half_angle_dict = {k: v / 2 for k, v in angle_dict.items()}
            neg_half_angle_dict = {k: -v / 2 for k, v in angle_dict.items()}
            circuit.add_ParametricRZ_gate(target, half_angle_dict)
            circuit.add_CNOT_gate(control, target)
            circuit.add_ParametricRZ_gate(target, neg_half_angle_dict)
            circuit.add_CNOT_gate(control, target)

    # Measurement - QURI Parts doesn't support mid-circuit measurements
    def emit_measure(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubit: int,
        clbit: int,
    ) -> None:
        """No-op: QURI Parts circuits don't support measurement gates.

        Measurement is handled separately by samplers/estimators.
        """
        pass

    # Barrier - QURI Parts doesn't support barriers
    def emit_barrier(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        qubits: list[int],
    ) -> None:
        """No-op: QURI Parts doesn't support barrier instructions."""
        pass

    # Sub-circuit support - limited in QURI Parts
    def circuit_to_gate(
        self, circuit: "LinearMappedUnboundParametricQuantumCircuit", name: str = "U"
    ) -> Any:
        """QURI Parts doesn't support converting circuits to gates.

        Returns None to signal that manual decomposition should be used.
        """
        return None

    def append_gate(
        self,
        circuit: "LinearMappedUnboundParametricQuantumCircuit",
        gate: Any,
        qubits: list[int],
    ) -> None:
        """Append a gate to the circuit.

        Since circuit_to_gate returns None, this is only called with
        QURI Parts native gates which can be extended into the circuit.
        """
        if gate is not None:
            # If we ever have a gate, try to extend
            try:
                circuit.extend(gate)
            except Exception:
                pass

    def gate_power(self, gate: Any, power: int) -> Any:
        """Create gate raised to a power.

        Returns None since QURI Parts doesn't support this natively.
        """
        return None

    def gate_controlled(self, gate: Any, num_controls: int) -> Any:
        """Create controlled version of a gate.

        Returns None since QURI Parts doesn't support this natively.
        """
        return None

    # Control flow support - not supported by QURI Parts
    def supports_for_loop(self) -> bool:
        """QURI Parts does not support native for loops."""
        return False

    def supports_if_else(self) -> bool:
        """QURI Parts does not support native if/else."""
        return False

    def supports_while_loop(self) -> bool:
        """QURI Parts does not support native while loops."""
        return False
