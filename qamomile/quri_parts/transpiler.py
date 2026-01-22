"""
Qamomile to QuriParts Transpiler Module

This module provides functionality to convert Qamomile quantum circuits, operators,
and measurement results to their QuriParts equivalents. It includes a QuriPartsTranspiler
class that implements the QuantumSDKTranspiler interface for QuriParts compatibility.

Key features:
- Convert Qamomile quantum circuits to QuriParts quantum circuits
- Convert Qamomile Hamiltonians to QuriParts Operators
- Convert QuriParts measurement results to Qamomile BitsSampleSet

Usage:
    from qamomile.quriparts.transpiler import QuriPartsTranspiler

    transpiler = QuriPartsTranspiler()
    qp_circuit = transpiler.transpile_circuit(qamomile_circuit)
    qp_operator = transpiler.transpile_hamiltonian(qamomile_hamiltonian)
    qamomile_results = transpiler.convert_result(quriparts_results)

Note: This module requires both Qamomile and QuriParts to be installed.
"""

from __future__ import annotations

import collections
from typing import Any, Sequence, TYPE_CHECKING

import quri_parts.core.operator as qp_o
import quri_parts.circuit as qp_c
import qamomile.core.bitssample as qm_bs
import qamomile.core.circuit as qm_c
import qamomile.core.operator as qm_o
from qamomile.core.transpiler import QuantumSDKTranspiler
from quri_parts.core.circuit import add_parametric_commuting_paulis_exp_gate

# New qamomile.circuit API imports
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.executable import (
    QuantumExecutor,
    ParameterMetadata,
)

from .parameter_converter import convert_parameter
from .exceptions import QamomileQuriPartsTranspileError
from .emitter import QuriPartsGateEmitter

if TYPE_CHECKING:
    from quri_parts.circuit import ImmutableBoundParametricQuantumCircuit


class QuriPartsTranspiler(QuantumSDKTranspiler[tuple[collections.Counter[int], int]]):
    """
    Transpiler class for converting between Qamomile and QuriParts quantum objects.

    This class implements the QuantumSDKTranspiler interface for QuriParts compatibility,
    providing methods to convert circuits, Hamiltonians, and measurement results.
    """

    def transpile_circuit(
        self, qamomile_circuit: qm_c.QuantumCircuit
    ) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
        """
        Convert a Qamomile quantum circuit to a QuriParts quantum circuit.

        Args:
            qamomile_circuit (qm_c.QuantumCircuit): The Qamomile quantum circuit to convert.

        Returns:
            qp_c.LinearMappedUnboundParametricQuantumCircuit: The converted QuriParts quantum circuit.

        Raises:
            QamomileQuriPartsTranspileError: If there's an error during conversion.
        """
        try:
            parameters = qamomile_circuit.get_parameters()
            qp_circuit = qp_c.LinearMappedUnboundParametricQuantumCircuit(
                qamomile_circuit.num_qubits, qamomile_circuit.num_clbits
            )
            self.param_mapping = {
                param: qp_circuit.add_parameter(param.name) for param in parameters
            }
            return self._circuit_convert(
                qamomile_circuit, qp_circuit, self.param_mapping
            )
        except Exception as e:
            raise QamomileQuriPartsTranspileError(f"Error converting circuit: {str(e)}")

    def _circuit_convert(
        self,
        qamomile_circuit: qm_c.QuantumCircuit,
        qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
        param_mapping: dict[qm_c.Parameter, qp_c.Parameter],
    ) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
        """
        Internal method to recursively convert Qamomile circuits to QuriParts circuits.

        Args:
            qamomile_circuit (qm_c.QuantumCircuit): The Qamomile circuit to convert.
            qp_circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit): The QuriParts circuit.
            param_mapping (Dict[qm_c.Parameter, qp_c.Parameter]): Mapping of parameters.

        Returns:
            qp_c.LinearMappedUnboundParametricQuantumCircuit: The converted QuriParts circuit.

        Raises:
            QamomileQuriPartsTranspileError: If an unsupported gate type is encountered.
        """

        for gate in qamomile_circuit.gates:
            if isinstance(gate, qm_c.SingleQubitGate):
                qp_circuit = _single_qubit_gate(gate, qp_circuit)
            elif isinstance(gate, qm_c.ParametricSingleQubitGate):
                qp_circuit = _parametric_single_qubit_gate(
                    gate, qp_circuit, parameters=param_mapping
                )
            elif isinstance(gate, qm_c.TwoQubitGate):
                qp_circuit = _two_qubit_gate(gate, qp_circuit)
            elif isinstance(gate, qm_c.ParametricTwoQubitGate):
                qp_circuit = _parametric_two_qubit_gate(
                    gate, qp_circuit, parameters=param_mapping
                )
            elif isinstance(gate, qm_c.ThreeQubitGate):
                qp_circuit = _three_qubit_gate(gate, qp_circuit)
            elif isinstance(gate, qm_c.ParametricExpGate):
                qp_operator = self.transpile_hamiltonian(gate.hamiltonian)
                qp_circuit = _parametric_exp_gate(
                    gate, qp_circuit, parameters=param_mapping, qp_operator=qp_operator
                )
            elif isinstance(gate, qm_c.Operator):
                qp_circuit = self._circuit_convert(
                    gate.circuit, qp_circuit, param_mapping
                )
            elif isinstance(gate, qm_c.MeasurementGate):
                # QURI-Parts circuits don't have measurement gates
                pass
            else:
                raise QamomileQuriPartsTranspileError(
                    f"Unsupported gate type: {type(gate)}"
                )
        return qp_circuit

    def convert_result(
        self, result: tuple[collections.Counter[int], int]
    ) -> qm_bs.BitsSampleSet:
        """
        Convert QuriParts measurement results to Qamomile BitsSampleSet.

        Args:
            result (tuple[collections.Counter[int], int]): QuriParts measurement results.

        Returns:
            qm_bs.BitsSampleSet: Converted Qamomile BitsSampleSet.
        """

        counter, num_bits = result
        int_counts = dict(counter)
        return qm_bs.BitsSampleSet.from_int_counts(int_counts, num_bits)

    def transpile_hamiltonian(self, operator: qm_o.Hamiltonian) -> qp_o.Operator:
        """
        Convert a Qamomile Hamiltonian to a QuriParts Operator.

        Args:
            operator (qm_o.Hamiltonian): The Qamomile Hamiltonian to convert.

        Returns:
            qp_o.Operator: The converted QuriParts Operator.

        Raises:
            NotImplementedError: If an unsupported Pauli operator is encountered.
        """
        qp_pauli_terms = {}
        for term, coeff in operator.terms.items():
            pauli_list = []
            for pauli in term:
                match pauli.pauli:
                    case qm_o.Pauli.X:
                        pauli_list.append((pauli.index, qp_o.SinglePauli.X))
                    case qm_o.Pauli.Y:
                        pauli_list.append((pauli.index, qp_o.SinglePauli.Y))
                    case qm_o.Pauli.Z:
                        pauli_list.append((pauli.index, qp_o.SinglePauli.Z))
                    case _:
                        raise NotImplementedError(
                            "Only Pauli X, Y, and Z are supported"
                        )

            qp_pauli_terms[qp_o.pauli_label(pauli_list)] = coeff
        h = qp_o.Operator(qp_pauli_terms)
        if operator.constant != 0:
            h.constant = operator.constant
        return h


def _single_qubit_gate(
    gate: qm_c.SingleQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a single qubit gate to the quri-parts circuit."""
    gate_map = {
        qm_c.SingleQubitGateType.H: qp_circuit.add_H_gate,
        qm_c.SingleQubitGateType.X: qp_circuit.add_X_gate,
        qm_c.SingleQubitGateType.Y: qp_circuit.add_Y_gate,
        qm_c.SingleQubitGateType.Z: qp_circuit.add_Z_gate,
        qm_c.SingleQubitGateType.S: qp_circuit.add_S_gate,
        qm_c.SingleQubitGateType.T: qp_circuit.add_T_gate,
    }
    gate_map[gate.gate](gate.qubit)
    return qp_circuit


def _parametric_single_qubit_gate(
    gate: qm_c.ParametricSingleQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
    parameters: dict[qm_c.Parameter, qp_c.Parameter],
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a parametric single qubit gate to the quri-parts circuit."""
    angle = convert_parameter(gate.parameter, parameters=parameters)
    gate_map = {
        qm_c.ParametricSingleQubitGateType.RX: qp_circuit.add_ParametricRX_gate,
        qm_c.ParametricSingleQubitGateType.RY: qp_circuit.add_ParametricRY_gate,
        qm_c.ParametricSingleQubitGateType.RZ: qp_circuit.add_ParametricRZ_gate,
    }
    gate_map[gate.gate](gate.qubit, angle)
    return qp_circuit


def _two_qubit_gate(
    gate: qm_c.TwoQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a two qubit gate to the quri-parts circuit."""
    gate_map = {
        qm_c.TwoQubitGateType.CNOT: qp_circuit.add_CNOT_gate,
        qm_c.TwoQubitGateType.CZ: qp_circuit.add_CZ_gate,
    }
    gate_map[gate.gate](gate.control, gate.target)
    return qp_circuit


def _parametric_two_qubit_gate(
    gate: qm_c.ParametricTwoQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
    parameters: dict[qm_c.Parameter, qp_c.Parameter],
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a parametric two qubit gate to the Qiskit circuit."""
    angle = convert_parameter(gate.parameter, parameters=parameters)
    match gate.gate:
        # case qm_c.ParametricTwoQubitGateType.CRX:
        #     qk_circuit.crx(angle, gate.control, gate.target)
        # case qm_c.ParametricTwoQubitGateType.CRY:
        #     qk_circuit.cry(angle, gate.control, gate.target)
        # case qm_c.ParametricTwoQubitGateType.CRZ:
        #     qk_circuit.crz(angle, gate.control, gate.target)
        case qm_c.ParametricTwoQubitGateType.RXX:
            qp_circuit.add_ParametricPauliRotation_gate(
                [gate.control, gate.target], pauli_ids=[1, 1], angle=angle
            )
        case qm_c.ParametricTwoQubitGateType.RYY:
            qp_circuit.add_ParametricPauliRotation_gate(
                [gate.control, gate.target], pauli_ids=[2, 2], angle=angle
            )
        case qm_c.ParametricTwoQubitGateType.RZZ:
            qp_circuit.add_ParametricPauliRotation_gate(
                [gate.control, gate.target], pauli_ids=[3, 3], angle=angle
            )
        case _:
            raise QamomileQuriPartsTranspileError(
                f"Unsupported parametric two qubit gate: {gate.gate}"
            )
    return qp_circuit


def _three_qubit_gate(
    gate: qm_c.ThreeQubitGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply a three qubit gate to the quri-parts circuit."""
    if gate.gate == qm_c.ThreeQubitGateType.CCX:
        qp_circuit.add_TOFFOLI_gate(gate.control1, gate.control2, gate.target)
    return qp_circuit


def _parametric_exp_gate(
    gate: qm_c.ParametricExpGate,
    qp_circuit: qp_c.LinearMappedUnboundParametricQuantumCircuit,
    parameters: dict[qm_c.Parameter, qp_c.Parameter],
    qp_operator: qp_o.Operator,
) -> qp_c.LinearMappedUnboundParametricQuantumCircuit:
    """Apply an exponential pauli rotation gate to the quri-parts circuit."""
    param_fn = convert_parameter(gate.parameter, parameters=parameters)
    for key in param_fn:
        param_fn[key] = -param_fn[key]
    add_parametric_commuting_paulis_exp_gate(qp_circuit, param_fn, qp_operator)
    return qp_circuit


# =============================================================================
# New qamomile.circuit API (QKernel-based)
# =============================================================================


class QuriPartsEmitPass(
    StandardEmitPass["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """QURI Parts-specific emission pass.

    Uses StandardEmitPass with QuriPartsGateEmitter for gate emission.
    QURI Parts does not support native control flow, so all loops are unrolled.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ):
        """Initialize the QURI Parts emit pass.

        Args:
            bindings: Parameter bindings for the circuit
            parameters: List of parameter names to preserve as backend parameters
        """
        emitter = QuriPartsGateEmitter()
        # QURI Parts has no native composite gate emitters
        composite_emitters: list[Any] = []
        super().__init__(emitter, bindings, parameters, composite_emitters)


class QuriPartsExecutor(
    QuantumExecutor["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """QURI Parts quantum executor.

    Supports both sampling and expectation value estimation.
    Uses Qulacs backend by default for efficient simulation.

    Example:
        executor = QuriPartsExecutor()  # Uses Qulacs by default
        counts = executor.execute(circuit, shots=1000)
        # counts: {"00": 512, "11": 512}
    """

    def __init__(
        self,
        sampler: Any = None,
        estimator: Any = None,
    ):
        """Initialize executor with optional sampler and estimator.

        Args:
            sampler: QURI Parts sampler (defaults to qulacs vector sampler)
            estimator: QURI Parts parametric estimator (defaults to qulacs)
        """
        self._sampler = sampler
        self._estimator = estimator

    @property
    def sampler(self) -> Any:
        """Lazy initialization of sampler."""
        if self._sampler is None:
            try:
                from quri_parts.qulacs.sampler import create_qulacs_vector_sampler

                self._sampler = create_qulacs_vector_sampler()
            except ImportError:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                )
        return self._sampler

    @property
    def parametric_estimator(self) -> Any:
        """Lazy initialization of parametric estimator for optimization."""
        if self._estimator is None:
            try:
                from quri_parts.qulacs.estimator import (
                    create_qulacs_vector_parametric_estimator,
                )

                self._estimator = create_qulacs_vector_parametric_estimator()
            except ImportError:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                )
        return self._estimator

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        Args:
            circuit: The quantum circuit to execute (bound or unbound)
            shots: Number of measurement shots

        Returns:
            Dictionary mapping bitstrings to counts (e.g., {"00": 512, "11": 512})
        """
        # Execute sampling
        counter = self.sampler(circuit, shots)

        # Convert Counter[int] to dict[str, int] with bitstring keys
        num_qubits = circuit.qubit_count
        return {format(k, f"0{num_qubits}b"): v for k, v in counter.items()}

    def bind_parameters(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "ImmutableBoundParametricQuantumCircuit":
        """Bind parameter values to the QURI Parts circuit.

        QURI Parts requires parameter values as a sequence in the order
        parameters were added to the circuit.

        Args:
            circuit: The unbound parametric circuit
            bindings: Dictionary of parameter name to value
            parameter_metadata: Metadata about the parameters

        Returns:
            Bound parametric circuit
        """
        # Build parameter values list in the order stored in metadata
        param_values = []
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                param_values.append(float(bindings[param_info.name]))
            else:
                raise ValueError(f"Missing binding for parameter: {param_info.name}")

        return circuit.bind_parameters(param_values)

    def estimate_expectation(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        hamiltonian: "qp_o.Operator",
        param_values: Sequence[float],
    ) -> float:
        """Estimate expectation value of hamiltonian for parametric circuit.

        Used during optimization (e.g., QAOA).

        Args:
            circuit: The unbound parametric circuit
            hamiltonian: QURI Parts Operator representing the Hamiltonian
            param_values: Sequence of parameter values in order

        Returns:
            Real part of the expectation value
        """
        from quri_parts.core.state import quantum_state, apply_circuit

        cb_state = quantum_state(circuit.qubit_count, bits=0)
        parametric_state = apply_circuit(circuit, cb_state)

        estimate = self.parametric_estimator(
            hamiltonian, parametric_state, param_values
        )
        return estimate.value.real


class QuriPartsCircuitTranspiler(
    Transpiler["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """New QURI Parts transpiler for qamomile.circuit module.

    Converts Qamomile QKernels into QURI Parts quantum circuits.

    Example:
        from qamomile.quri_parts import QuriPartsCircuitTranspiler
        import qamomile.circuit as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = QuriPartsCircuitTranspiler()
        circuit = transpiler.to_circuit(bell_state)
    """

    def _create_separate_pass(self) -> SeparatePass:
        return SeparatePass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["qp_c.LinearMappedUnboundParametricQuantumCircuit"]:
        return QuriPartsEmitPass(bindings, parameters)

    def executor(
        self,
        sampler: Any = None,
        estimator: Any = None,
    ) -> QuriPartsExecutor:
        """Create a QURI Parts executor.

        Args:
            sampler: Optional custom sampler (defaults to qulacs vector sampler)
            estimator: Optional custom estimator (defaults to qulacs parametric estimator)

        Returns:
            QuriPartsExecutor configured for this backend
        """
        return QuriPartsExecutor(sampler, estimator)
