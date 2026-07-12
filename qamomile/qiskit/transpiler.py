"""Qiskit backend transpiler implementation.

This module provides QiskitTranspiler for converting Qamomile QKernels
into Qiskit QuantumCircuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    from qiskit import QuantumCircuit

from qamomile.circuit.transpiler.circuit_ir import (
    CircuitBackendEmitPass,
    CompilationPolicy,
)
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.qiskit.materializer import QiskitMaterializer


class QiskitExecutor(QuantumExecutor["QuantumCircuit"]):
    """Qiskit quantum executor using a safe local simulator or other backends.

    Example:
        executor = QiskitExecutor()  # Uses AerSimulator when available
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

                self.backend = AerSimulator(max_parallel_threads=1)
            except ImportError:
                try:
                    from qiskit.providers.basic_provider import BasicSimulator

                    self.backend = BasicSimulator()
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
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate the expectation value of a Hamiltonian.

        Args:
            circuit: Qiskit QuantumCircuit (state preparation ansatz)
            hamiltonian: The qamomile.observable.Hamiltonian to measure
            params: Optional parameter values for parametric circuits

        Returns:
            The estimated expectation value

        Raises:
            RuntimeError: If no estimator is configured
        """
        if self._estimator is None:
            # Create default Qiskit Estimator
            try:
                from qiskit.primitives import StatevectorEstimator

                self._estimator = StatevectorEstimator()
            except ImportError:
                try:
                    # Fallback for older Qiskit versions
                    from qiskit.primitives import (
                        Estimator,  # type: ignore[attr-defined]
                    )

                    self._estimator = Estimator()
                except ImportError:
                    from qiskit_aer.primitives import Estimator

                    self._estimator = Estimator()

        # Convert Hamiltonian to SparsePauliOp
        from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op

        sparse_pauli_op = hamiltonian_to_sparse_pauli_op(hamiltonian)

        # Run estimation
        if params is not None:
            param_values = list(params)
        else:
            param_values = []

        # Check if this is V1 or V2 interface
        # V2 interface (new): run([(circuit, observable, params)])
        # V1 interface (old): run(circuits, observables, parameter_values)
        estimator_run = cast(Any, self._estimator).run
        try:
            # Try V2 interface first
            job = estimator_run([(circuit, sparse_pauli_op, param_values)])
            result = job.result()
            return float(result[0].data.evs)
        except (TypeError, AttributeError):
            # Fall back to V1 interface
            job = estimator_run(
                [circuit],
                [sparse_pauli_op],
                [param_values] if param_values else None,
            )
            result = job.result()
            return float(result.values[0])

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
        use_native_composite (bool): Whether to prefer native Qiskit library
            realizations for semantic composites such as QFT/IQFT. Defaults
            to ``True``.
        use_native_pauli_evolution (bool): Whether to prefer
            ``PauliEvolutionGate`` over gate gadgets. Defaults to ``True``.

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

    def __init__(
        self,
        use_native_composite: bool = True,
        use_native_pauli_evolution: bool = True,
    ) -> None:
        """Initialize the Qiskit transpiler.

        Args:
            use_native_composite (bool): Whether to prefer native semantic operation
                composites such as QFT/IQFT. Defaults to ``True``.
            use_native_pauli_evolution (bool): Whether to prefer native Pauli
                evolution over gate gadgets. Defaults to ``True``.
        """
        self._use_native_composite = use_native_composite
        self._use_native_pauli_evolution = use_native_pauli_evolution

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the host-orchestrated circuit segmentation pass.

        Returns:
            SegmentationPass: Standard single-quantum-segment planner.
        """
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["QuantumCircuit"]:
        """Create the capability-driven Qiskit materialization pipeline.

        Args:
            bindings (dict[str, Any] | None): Compile-time bindings. Defaults
                to ``None``.
            parameters (list[str] | None): Runtime parameter names. Defaults
                to ``None``.

        Returns:
            EmitPass[QuantumCircuit]: Circuit lowering, legalization, and
                Qiskit materialization pass.
        """
        return CircuitBackendEmitPass(
            QiskitMaterializer(),
            bindings,
            parameters,
            policy=CompilationPolicy(
                prefer_native_semantic_ops=self._use_native_composite,
                prefer_native_pauli_evolution=self._use_native_pauli_evolution,
            ),
        )

    def executor(  # type: ignore[override]
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
