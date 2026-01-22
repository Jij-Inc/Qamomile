"""Qiskit backend transpiler implementation.

This module provides QiskitTranspiler for converting Qamomile QKernels
into Qiskit QuantumCircuits.
"""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qamomile.circuit.observable import Observable

from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SeparatePass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.executable import (
    QuantumExecutor,
    ParameterMetadata,
)

from qamomile.qiskit.emitter import QiskitGateEmitter


class QiskitEmitPass(StandardEmitPass["QuantumCircuit"]):
    """Qiskit-specific emission pass.

    Extends StandardEmitPass with Qiskit-specific control flow handling
    using context managers.
    """

    def __init__(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
        use_native_composite: bool = True,
    ):
        """Initialize the Qiskit emit pass.

        Args:
            bindings: Parameter bindings for the circuit
            parameters: List of parameter names to preserve as backend parameters
            use_native_composite: If True, use native Qiskit implementations
                                  for QFT/IQFT. If False, use manual decomposition.
        """
        emitter = QiskitGateEmitter()
        composite_emitters = self._init_emitters() if use_native_composite else []
        super().__init__(emitter, bindings, parameters, composite_emitters)
        self._use_native_composite = use_native_composite

    def _init_emitters(self) -> list:
        """Initialize native CompositeGate emitters."""
        from qamomile.qiskit.emitters import QiskitQFTEmitter

        return [QiskitQFTEmitter()]

    def _emit_for(
        self,
        circuit: "QuantumCircuit",
        op: ForOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        """Emit a for loop using Qiskit's native for_loop context manager."""
        start = (
            self._resolver.resolve_int_value(op.operands[0], bindings)
            if len(op.operands) > 0
            else 0
        )
        stop = (
            self._resolver.resolve_int_value(op.operands[1], bindings)
            if len(op.operands) > 1
            else 1
        )
        step = (
            self._resolver.resolve_int_value(op.operands[2], bindings)
            if len(op.operands) > 2
            else 1
        )

        if start is None or stop is None or step is None:
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        indexset = range(start, stop, step)
        if len(indexset) == 0:
            return

        if force_unroll:
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        if self._loop_analyzer.should_unroll(op, bindings):
            self._emit_for_unrolled(circuit, op, qubit_map, clbit_map, bindings)
            return

        # Use Qiskit's native for_loop context manager
        with circuit.for_loop(indexset) as loop_param:
            loop_bindings = bindings.copy()
            loop_bindings[op.loop_var] = loop_param
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, loop_bindings
            )

    def _emit_if(
        self,
        circuit: "QuantumCircuit",
        op: IfOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit if/else using Qiskit's if_test context manager."""
        condition_uuid = op.condition.uuid

        if condition_uuid not in clbit_map:
            return

        clbit_idx = clbit_map[condition_uuid]

        with circuit.if_test((circuit.clbits[clbit_idx], 1)) as else_:
            self._emit_operations(
                circuit, op.true_operations, qubit_map, clbit_map, bindings
            )
        with else_:
            self._emit_operations(
                circuit, op.false_operations, qubit_map, clbit_map, bindings
            )

    def _emit_while(
        self,
        circuit: "QuantumCircuit",
        op: WhileOperation,
        qubit_map: dict[str, int],
        clbit_map: dict[str, int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit while loop using Qiskit's while_loop context manager."""
        if not op.operands:
            raise ValueError("WhileOperation requires a condition operand")

        condition = op.operands[0]
        condition_value = condition.value if hasattr(condition, "value") else condition
        condition_uuid = (
            condition_value.uuid
            if hasattr(condition_value, "uuid")
            else str(condition_value)
        )

        if condition_uuid not in clbit_map:
            raise ValueError("While loop condition not found in classical bit map.")

        clbit_idx = clbit_map[condition_uuid]

        with circuit.while_loop((circuit.clbits[clbit_idx], 1)):
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, bindings
            )


class QiskitExecutor(QuantumExecutor["QuantumCircuit"]):
    """Qiskit quantum executor using AerSimulator or other backends.

    Example:
        executor = QiskitExecutor()  # Uses AerSimulator by default
        counts = executor.execute(circuit, shots=1000)
        # counts: {"00": 512, "11": 512}

        # With expectation value estimation
        from qamomile.qiskit.observable import QiskitExpectationEstimator
        executor = QiskitExecutor(estimator=QiskitExpectationEstimator())
        exp_val = executor.estimate(circuit, observable)
    """

    def __init__(self, backend=None, estimator=None):
        """Initialize executor with backend and optional estimator.

        Args:
            backend: Qiskit backend (defaults to AerSimulator if available)
            estimator: Optional QiskitExpectationEstimator for expectation values
        """
        self.backend = backend
        self._estimator = estimator

        if self.backend is None:
            try:
                from qiskit_aer import AerSimulator

                self.backend = AerSimulator()
            except ImportError:
                pass

    def execute(self, circuit: "QuantumCircuit", shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        Args:
            circuit: The quantum circuit to execute
            shots: Number of measurement shots

        Returns:
            Dictionary mapping bitstrings to counts (e.g., {"00": 512, "11": 512})
        """
        from qiskit import transpile

        if self.backend is None:
            raise RuntimeError("No backend available for execution")

        circuit_with_meas = self._ensure_measurements(circuit)
        transpiled = transpile(circuit_with_meas, self.backend)
        job = self.backend.run(transpiled, shots=shots)
        return job.result().get_counts()

    def bind_parameters(
        self,
        circuit: "QuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "QuantumCircuit":
        """Bind parameter values to the Qiskit circuit.

        Args:
            circuit: The parameterized circuit
            bindings: Dict mapping parameter names (indexed format) to values
            parameter_metadata: Metadata about circuit parameters

        Returns:
            New circuit with parameters bound
        """
        qiskit_bindings = {}
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                qiskit_bindings[param_info.backend_param] = bindings[param_info.name]

        return circuit.assign_parameters(qiskit_bindings)

    def estimate(
        self,
        circuit: "QuantumCircuit",
        observable: "Observable",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of an observable.

        Args:
            circuit: Qiskit QuantumCircuit (state preparation ansatz)
            observable: The observable to measure
            params: Optional parameter values for parametric circuits

        Returns:
            The estimated expectation value

        Raises:
            RuntimeError: If no estimator is configured
        """
        if self._estimator is None:
            # Try to create default estimator
            from qamomile.qiskit.observable import QiskitExpectationEstimator
            self._estimator = QiskitExpectationEstimator()

        return self._estimator.estimate(circuit, observable, params)

    def _ensure_measurements(self, circuit: "QuantumCircuit") -> "QuantumCircuit":
        """Ensure circuit has measurements, adding measure_all if needed."""
        if circuit.num_clbits > 0:
            return circuit

        circuit_copy = circuit.copy()
        circuit_copy.measure_all()
        return circuit_copy


class QiskitTranspiler(Transpiler["QuantumCircuit"]):
    """Qiskit backend transpiler.

    Converts Qamomile QKernels into Qiskit QuantumCircuits.

    Args:
        use_native_composite: If True (default), use native Qiskit library
                              implementations for QFT/IQFT. If False, use
                              manual decomposition for all composite gates.

    Example:
        from qamomile.qiskit import QiskitTranspiler
        import qamomile as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = QiskitTranspiler()
        circuit = transpiler.to_circuit(bell_state)
        print(circuit.draw())
    """

    def __init__(self, use_native_composite: bool = True):
        """Initialize the Qiskit transpiler.

        Args:
            use_native_composite: If True, use native Qiskit implementations
                                  for QFT/IQFT. If False, use manual decomposition.
        """
        self._use_native_composite = use_native_composite

    def _create_separate_pass(self) -> SeparatePass:
        return SeparatePass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["QuantumCircuit"]:
        return QiskitEmitPass(
            bindings, parameters, use_native_composite=self._use_native_composite
        )

    def executor(
        self,
        backend=None,
    ) -> QiskitExecutor:
        """Create a Qiskit executor.

        Args:
            backend: Qiskit backend (defaults to AerSimulator)

        Returns:
            QiskitExecutor configured with the backend
        """
        return QiskitExecutor(backend)
