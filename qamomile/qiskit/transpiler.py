"""Qiskit backend transpiler implementation.

This module provides QiskitTranspiler for converting Qamomile QKernels
into Qiskit QuantumCircuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
    from qiskit import QuantumCircuit

from qamomile.circuit.ir.operation.arithmetic_operations import (
    CompOp,
    CompOpKind,
    CondOp,
    CondOpKind,
    NotOp,
    RuntimeClassicalExpr,
    RuntimeOpKind,
)
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support import (
    ClbitMap,
    QubitAddress,
    QubitMap,
    resolve_if_condition,
)
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.transpiler import Transpiler
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
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
        force_unroll: bool = False,
    ) -> None:
        """Emit a for loop using Qiskit's native for_loop context manager."""
        from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
            resolve_loop_bounds,
        )

        start, stop, step = resolve_loop_bounds(self._resolver, op, bindings)

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
        with circuit.for_loop(indexset) as loop_param:  # type: ignore[call-overload]
            loop_bindings = bindings.copy()
            loop_bindings[op.loop_var] = loop_param
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, loop_bindings
            )

    def _emit_if(
        self,
        circuit: "QuantumCircuit",
        op: IfOperation,
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit if/else using Qiskit's if_test context manager."""
        condition = op.condition

        # Compile-time constant conditions are handled by the base class.
        if resolve_if_condition(condition, bindings) is not None:
            super()._emit_if(circuit, op, qubit_map, clbit_map, bindings)
            return

        if_test_condition = self._resolve_runtime_condition(
            circuit, condition, clbit_map, bindings
        )

        with circuit.if_test(if_test_condition) as else_:
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
        qubit_map: QubitMap,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit while loop using Qiskit's while_loop context manager."""
        if not op.operands:
            raise ValueError("WhileOperation requires a condition operand")

        condition = op.operands[0]
        condition_value = condition.value if hasattr(condition, "value") else condition
        while_condition = self._resolve_runtime_condition(
            circuit, condition_value, clbit_map, bindings
        )

        with circuit.while_loop(while_condition):  # type: ignore[call-overload]
            self._emit_operations(
                circuit, op.operations, qubit_map, clbit_map, bindings
            )

    def _resolve_runtime_condition(
        self,
        circuit: "QuantumCircuit",
        condition: Any,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> Any:
        """Resolve a runtime if/while condition to a Qiskit ``if_test`` argument.

        The returned value is suitable for passing to
        ``QuantumCircuit.if_test`` / ``while_loop``: either a
        ``(clbit, value)`` tuple for a single-bit measurement condition, or
        a ``qiskit.circuit.classical.expr.Expr`` for a compound predicate
        (``CondOp`` / ``NotOp`` / ``CompOp`` over measurement bits)
        previously stored in ``bindings`` by ``_build_runtime_predicate_expr``.

        Args:
            circuit: The Qiskit circuit being emitted (for clbit lookup).
            condition: The IR condition Value.
            clbit_map: Map from ``QubitAddress`` to physical clbit index.
            bindings: Current bindings; may hold a backend ``Expr`` keyed by
                the condition's UUID.

        Returns:
            A condition object accepted by ``if_test`` / ``while_loop``.

        Raises:
            EmitError: If the condition is neither a measurement clbit nor a
                stored runtime expression.
        """
        condition_uuid = (
            condition.uuid if hasattr(condition, "uuid") else str(condition)
        )

        # Compound predicate built earlier by _build_runtime_predicate_expr.
        stored = bindings.get(condition_uuid)
        if stored is not None and not isinstance(stored, (bool, int, float)):
            return stored

        condition_addr = QubitAddress(condition_uuid)
        if condition_addr in clbit_map:
            clbit_idx = clbit_map[condition_addr]
            return (circuit.clbits[clbit_idx], 1)

        raise EmitError(
            "Runtime if-conditions must come from measurement results "
            "or be bound before transpilation. The condition value was "
            "neither resolved at compile time nor backed by a "
            "measurement result."
        )

    def _emit_runtime_classical_expr(
        self,
        circuit: "QuantumCircuit",
        op: RuntimeClassicalExpr,
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Lower ``RuntimeClassicalExpr`` to a Qiskit ``expr.Expr``.

        Counterpart of (and supersedes for the runtime path) the legacy
        ``_build_runtime_predicate_expr``. The IR has already declared
        this op runtime via ``ClassicalLoweringPass``, so we go straight
        to expression construction without a fold attempt.

        Args:
            circuit: The Qiskit circuit being emitted.
            op: The runtime classical expression to lower.
            clbit_map: Map from ``QubitAddress`` → physical clbit index.
            bindings: Current bindings; result is stored here.

        Raises:
            EmitError: If an operand cannot be resolved to a clbit /
                sub-expression / constant — meaning the IR is malformed
                or the lowering pass missed a case.
        """
        from qamomile.circuit.transpiler.errors import EmitError
        from qiskit.circuit.classical import expr

        def resolve_operand(v: Any) -> Any:
            """Resolve an operand to a Qiskit ``Expr`` / ``Clbit`` / Python literal."""
            if hasattr(v, "uuid"):
                stored = bindings.get(v.uuid)
                if stored is not None and not isinstance(stored, (bool, int)):
                    return stored  # already-built backend expr
                if hasattr(v, "is_constant") and v.is_constant():
                    return bool(v.get_const())
                addr = QubitAddress(v.uuid)
                if addr in clbit_map:
                    return circuit.clbits[clbit_map[addr]]
                if isinstance(stored, (bool, int)):
                    return bool(stored)
            return None

        kind = op.kind
        if kind is RuntimeOpKind.NOT:
            inner = resolve_operand(op.operands[0])
            if inner is None:
                raise EmitError(
                    f"Cannot resolve operand for RuntimeClassicalExpr(NOT): "
                    f"{op.operands[0]!r}"
                )
            result = expr.logic_not(inner)
        else:
            lhs = resolve_operand(op.operands[0])
            rhs = resolve_operand(op.operands[1])
            if lhs is None or rhs is None:
                raise EmitError(
                    f"Cannot resolve operands for RuntimeClassicalExpr({kind!r})"
                )
            result = self._build_qiskit_binary_expr(kind, lhs, rhs)
            if result is None:
                raise EmitError(
                    f"Unsupported RuntimeClassicalExpr kind for Qiskit backend: {kind}"
                )

        # Store result so downstream ``_emit_if`` / ``_emit_while`` /
        # nested ``RuntimeClassicalExpr`` can consume it.
        set_runtime_expr = getattr(bindings, "set_runtime_expr", None)
        if callable(set_runtime_expr):
            set_runtime_expr(op.results[0].uuid, result)
        else:
            bindings[op.results[0].uuid] = result

    @staticmethod
    def _build_qiskit_binary_expr(kind: RuntimeOpKind, lhs: Any, rhs: Any) -> Any:
        """Map a binary ``RuntimeOpKind`` to its Qiskit ``expr`` constructor."""
        from qiskit.circuit.classical import expr

        match kind:
            # Logical
            case RuntimeOpKind.AND:
                return expr.logic_and(lhs, rhs)
            case RuntimeOpKind.OR:
                return expr.logic_or(lhs, rhs)
            # Comparison
            case RuntimeOpKind.EQ:
                return expr.equal(lhs, rhs)
            case RuntimeOpKind.NEQ:
                return expr.not_equal(lhs, rhs)
            case RuntimeOpKind.LT:
                return expr.less(lhs, rhs)
            case RuntimeOpKind.LE:
                return expr.less_equal(lhs, rhs)
            case RuntimeOpKind.GT:
                return expr.greater(lhs, rhs)
            case RuntimeOpKind.GE:
                return expr.greater_equal(lhs, rhs)
            # Arithmetic on classical bits — Qiskit ``expr`` supports a
            # subset; map what's available, return None for the rest.
            case RuntimeOpKind.ADD:
                return expr.add(lhs, rhs)
            case RuntimeOpKind.SUB:
                return expr.sub(lhs, rhs)
            case RuntimeOpKind.MUL:
                return expr.mul(lhs, rhs)
            case RuntimeOpKind.DIV:
                return expr.div(lhs, rhs)
            case _:
                return None

    def _build_runtime_predicate_expr(
        self,
        circuit: "QuantumCircuit",
        op: "CompOp | CondOp | NotOp",
        clbit_map: ClbitMap,
        bindings: dict[str, Any],
    ) -> Any:
        """Build a Qiskit ``expr.Expr`` for a runtime classical predicate.

        Recursively resolves operands to either Qiskit ``Clbit`` references
        (via ``clbit_map``), constants, or already-built sub-expressions
        stored in ``bindings``. Returns ``None`` when any operand cannot be
        resolved — the caller leaves the predicate unbound, and the
        downstream ``if`` / ``while`` falls back to its (failing) clbit
        lookup, which raises a clear ``EmitError``.

        Supported ops:
            * ``CondOp(AND/OR)`` → ``expr.logic_and`` / ``expr.logic_or``
            * ``NotOp`` → ``expr.logic_not``
            * ``CompOp(EQ/NEQ)`` over Bit operands → ``expr.equal`` /
              ``expr.not_equal``

        Args:
            circuit: The Qiskit circuit being emitted (for clbit lookup).
            op: The unresolved classical predicate.
            clbit_map: Map from ``QubitAddress`` → physical clbit index.
            bindings: Current bindings; may hold sub-expression results
                keyed by Value UUIDs.

        Returns:
            A ``qiskit.circuit.classical.expr.Expr`` object, or ``None`` if
            the predicate cannot be expressed.
        """
        from qiskit.circuit.classical import expr

        def resolve_operand(v: Any) -> Any:
            """Operand → Qiskit ``Expr`` / ``Clbit`` / Python literal, or None."""
            if hasattr(v, "uuid"):
                stored_v = bindings.get(v.uuid)
                if stored_v is not None and not isinstance(stored_v, (bool, int)):
                    return stored_v
                if hasattr(v, "is_constant") and v.is_constant():
                    return bool(v.get_const())
                addr = QubitAddress(v.uuid)
                if addr in clbit_map:
                    return circuit.clbits[clbit_map[addr]]
                if isinstance(stored_v, (bool, int)):
                    return bool(stored_v)
            return None

        if isinstance(op, NotOp):
            inner = resolve_operand(op.operands[0])
            if inner is None:
                return None
            return expr.logic_not(inner)

        lhs = resolve_operand(op.operands[0])
        rhs = resolve_operand(op.operands[1])
        if lhs is None or rhs is None:
            return None

        if isinstance(op, CondOp):
            if op.kind is CondOpKind.AND:
                return expr.logic_and(lhs, rhs)
            if op.kind is CondOpKind.OR:
                return expr.logic_or(lhs, rhs)
            return None

        if isinstance(op, CompOp):
            if op.kind is CompOpKind.EQ:
                return expr.equal(lhs, rhs)
            if op.kind is CompOpKind.NEQ:
                return expr.not_equal(lhs, rhs)
            return None

        return None

    def _emit_pauli_evolve(
        self,
        circuit: "QuantumCircuit",
        op: "PauliEvolveOp",
        qubit_map: QubitMap,
        bindings: dict[str, Any],
    ) -> None:
        """Emit Pauli evolution using Qiskit's PauliEvolutionGate."""
        import qamomile.observable as qm_o

        if not self._use_native_composite:
            super()._emit_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Resolve Hamiltonian from bindings
        obs_value = op.observable
        hamiltonian = None
        if hasattr(obs_value, "name") and obs_value.name in bindings:
            hamiltonian = bindings[obs_value.name]
        if hamiltonian is None and hasattr(obs_value, "uuid"):
            hamiltonian = bindings.get(obs_value.uuid)
        if not isinstance(hamiltonian, qm_o.Hamiltonian):
            super()._emit_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Resolve gamma: concrete float OR backend Parameter for parametric
        # gamma (scalar / array element). Qiskit's PauliEvolutionGate accepts
        # a Parameter (or ParameterExpression) as ``time``.
        from qamomile.circuit.transpiler.passes.emit_support.pauli_evolve_emission import (
            _resolve_gamma,
        )

        gamma = _resolve_gamma(self, op, bindings)
        if gamma is None:
            super()._emit_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Validate qubit count: logical array size vs Hamiltonian
        input_array = op.qubits
        num_h_qubits = hamiltonian.num_qubits
        from qamomile.circuit.ir.value import ArrayValue

        if isinstance(input_array, ArrayValue) and input_array.shape:
            n_resolved = self._resolver.resolve_int_value(
                input_array.shape[0], bindings
            )
            if n_resolved is not None and n_resolved != num_h_qubits:
                raise EmitError(
                    f"PauliEvolveOp qubit count mismatch: "
                    f"qubit register has {n_resolved} qubits but "
                    f"Hamiltonian acts on {num_h_qubits} qubits.",
                )

        # Validate Hermitian (real coefficients)
        for operators, coeff in hamiltonian:
            if abs(coeff.imag) > 1e-10:
                raise EmitError(
                    f"PauliEvolveOp requires a Hermitian Hamiltonian "
                    f"(real coefficients), but found complex coefficient "
                    f"{coeff} on term {operators}.",
                )

        qubit_indices: list[int] = []
        for i in range(num_h_qubits):
            addr = QubitAddress(input_array.uuid, i)
            if addr in qubit_map:
                qubit_indices.append(qubit_map[addr])
            else:
                super()._emit_pauli_evolve(circuit, op, qubit_map, bindings)
                return

        try:
            from qamomile.qiskit.observable import hamiltonian_to_sparse_pauli_op
            from qiskit.circuit.library import PauliEvolutionGate

            sparse_op = hamiltonian_to_sparse_pauli_op(hamiltonian)
            # ``time`` accepts both ``float`` and Qiskit ``Parameter`` /
            # ``ParameterExpression``. For parametric gamma we pass the
            # backend parameter through so it can be bound at run-time.
            time_arg = float(gamma) if isinstance(gamma, (int, float)) else gamma
            evo_gate = PauliEvolutionGate(sparse_op, time=time_arg)
            circuit.append(evo_gate, qubit_indices)
        except ImportError:
            # Fallback to manual decomposition when Qiskit library unavailable
            super()._emit_pauli_evolve(circuit, op, qubit_map, bindings)
            return

        # Map result array to same physical qubits
        result_array = op.evolved_qubits
        for i, phys_idx in enumerate(qubit_indices):
            result_addr = QubitAddress(result_array.uuid, i)
            if result_addr not in qubit_map:
                qubit_map[result_addr] = phys_idx


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
        try:
            # Try V2 interface first
            job = self._estimator.run([(circuit, sparse_pauli_op, param_values)])  # type: ignore[arg-type,call-arg,list-item]
            result = job.result()
            return float(result[0].data.evs)  # type: ignore[union-attr]
        except (TypeError, AttributeError):
            # Fall back to V1 interface
            job = self._estimator.run(  # type: ignore[call-arg,arg-type,list-item]
                [circuit],
                [sparse_pauli_op],
                [param_values] if param_values else None,  # type: ignore[arg-type,list-item]
            )
            result = job.result()
            return float(result.values[0])  # type: ignore[union-attr]

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

    def _create_segmentation_pass(self) -> SegmentationPass:
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["QuantumCircuit"]:
        return QiskitEmitPass(
            bindings, parameters, use_native_composite=self._use_native_composite
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
