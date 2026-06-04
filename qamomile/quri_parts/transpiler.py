"""QURI Parts backend transpiler implementation.

This module provides QuriPartsTranspiler for converting Qamomile QKernels
into QURI Parts quantum circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
)
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support import (
    QubitAddress,
    QubitMap,
)
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_quantum_input_shapes,
    _expand_quantum_operands_to_phys,
    _map_controlled_u_results,
    _populate_input_qubit_map,
    emit_controlled_gate,
    resolve_power,
)
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.passes.standard_emit import StandardEmitPass
from qamomile.circuit.transpiler.transpiler import Transpiler

from .emitter import QuriPartsGateEmitter
from .exceptions import QamomileQuriPartsTranspileError

if TYPE_CHECKING:
    import qamomile.observable as qm_o
    import quri_parts.circuit as qp_c  # type: ignore[import-not-found]
    import quri_parts.core.operator as qp_o  # type: ignore[import-not-found]
    from quri_parts.circuit import (  # type: ignore[import-not-found]
        ImmutableBoundParametricQuantumCircuit,
    )


def _build_quri_controlled_qubit_map(
    emit_pass: StandardEmitPass[Any],
    block_value: Any,
    target_indices: list[int],
    bindings: dict[str, Any],
) -> QubitMap:
    """Build a block-local qubit map backed by actual QURI target slots.

    Args:
        emit_pass (StandardEmitPass[Any]): Emit pass used to resolve
            symbolic vector shapes.
        block_value (Any): Inner block whose quantum inputs are mapped.
        target_indices (list[int]): Physical QURI Parts qubit indices
            supplied as the controlled-U target operands.
        bindings (dict[str, Any]): Bindings used while resolving vector
            input shapes.

    Returns:
        QubitMap: Mapping from the inner block's formal quantum inputs to
            the parent circuit's physical QURI Parts qubit indices.

    Raises:
        EmitError: If the shared input-mapping helper cannot resolve or
            fit the block's quantum input footprint.
    """
    local_map: QubitMap = {}
    _populate_input_qubit_map(
        emit_pass,
        getattr(block_value, "input_values", []),
        len(target_indices),
        bindings,
        local_map,
    )
    return {address: target_indices[slot] for address, slot in local_map.items()}


def _resolve_quri_gate_targets(
    emit_pass: StandardEmitPass[Any],
    op: GateOperation,
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> list[int]:
    """Resolve a gate operation's quantum operands to physical qubits.

    Args:
        emit_pass (StandardEmitPass[Any]): Emit pass whose resolver maps
            IR values to QURI Parts physical qubit indices.
        op (GateOperation): Gate operation being emitted under controls.
        qubit_map (QubitMap): Current block-local qubit map.
        bindings (dict[str, Any]): Bindings used for array/slice
            resolution.

    Returns:
        list[int]: Physical target qubit indices in operand order.

    Raises:
        EmitError: If any gate operand cannot be resolved.
    """
    target_indices: list[int] = []
    for operand in op.qubit_operands:
        index = emit_pass._resolver.resolve_qubit_index(operand, qubit_map, bindings)
        if index is None:
            raise EmitError(
                f"QURI Parts controlled fallback cannot resolve gate "
                f"operand {operand.name!r} to a physical qubit.",
                operation="ControlledUOperation",
            )
        target_indices.append(index)
    return target_indices


def _propagate_quri_gate_results(
    op: GateOperation,
    target_indices: list[int],
    qubit_map: QubitMap,
) -> None:
    """Propagate gate result values to their unchanged physical slots.

    Args:
        op (GateOperation): Gate operation whose results were just
            emitted.
        target_indices (list[int]): Physical qubit indices resolved from
            ``op.qubit_operands``.
        qubit_map (QubitMap): Mutable block-local qubit map to update.
    """
    quantum_results = [result for result in op.results if result.type.is_quantum()]
    for result, index in zip(quantum_results, target_indices):
        qubit_map[QubitAddress(result.uuid)] = index


def _emit_quri_controlled_gate(
    emit_pass: StandardEmitPass[Any],
    circuit: Any,
    op: GateOperation,
    control_indices: list[int],
    target_indices: list[int],
    bindings: dict[str, Any],
) -> None:
    """Emit one QURI Parts gate under accumulated controls.

    Args:
        emit_pass (StandardEmitPass[Any]): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        op (GateOperation): Primitive gate operation to emit.
        control_indices (list[int]): Accumulated physical control qubits.
        target_indices (list[int]): Physical target qubits for ``op``.
        bindings (dict[str, Any]): Bindings used to resolve gate angles.

    Raises:
        EmitError: If the accumulated controls require decomposition that
            this QURI Parts fallback intentionally leaves to a follow-up PR.
    """
    if not control_indices:
        raise EmitError(
            "QURI Parts controlled fallback requires at least one control.",
            operation="ControlledUOperation",
        )

    if len(control_indices) == 1:
        if op.gate_type == GateOperationType.CX:
            if len(target_indices) < 2:
                raise EmitError(
                    "Controlled-CX requires two target qubits.",
                    operation="ControlledUOperation",
                )
            emit_pass._emitter.emit_toffoli(
                circuit,
                control_indices[0],
                target_indices[0],
                target_indices[1],
            )
            return
        emit_controlled_gate(
            emit_pass,
            circuit,
            op,
            control_indices[0],
            target_indices,
            bindings,
        )
        return

    if (
        len(control_indices) == 2
        and op.gate_type == GateOperationType.X
        and len(target_indices) == 1
    ):
        emit_pass._emitter.emit_toffoli(
            circuit,
            control_indices[0],
            control_indices[1],
            target_indices[0],
        )
        return

    raise EmitError(
        "Cannot emit QURI Parts controlled-U: recursive controlled fallback "
        "reached a multi-controlled operation that requires decomposition "
        "not implemented in this PR "
        f"(controls={control_indices}, gate={op.gate_type!r}).",
        operation="ControlledUOperation",
    )


def _emit_quri_nested_controlled_u(
    emit_pass: StandardEmitPass[Any],
    circuit: Any,
    op: ControlledUOperation,
    outer_control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Emit a nested controlled-U by flattening controls at emit time.

    Args:
        emit_pass (StandardEmitPass[Any]): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        op (ControlledUOperation): Nested controlled-U operation from the
            block currently being walked.
        outer_control_indices (list[int]): Controls accumulated from
            enclosing controlled-U operations.
        qubit_map (QubitMap): Current block-local qubit map.
        bindings (dict[str, Any]): Bindings visible inside the current
            block.

    Raises:
        EmitError: If the nested op is symbolic, lacks a block, or cannot
            be lowered by the guarded recursive fallback.
    """
    if not isinstance(op, ConcreteControlledU):
        raise EmitError(
            "QURI Parts recursive controlled fallback only supports "
            "concrete nested ControlledUOperation values.",
            operation="ControlledUOperation",
        )
    if op.block is None:
        raise EmitError(
            "QURI Parts recursive controlled fallback cannot emit a nested "
            "ControlledUOperation without an inner block.",
            operation="ControlledUOperation",
        )

    nested_controls = [
        index
        for operand in op.control_operands
        for index in _expand_quantum_operands_to_phys(
            emit_pass, operand, qubit_map, bindings
        )
    ]
    remaining_operands = op.operands[op.num_controls :]
    target_qubit_operands = [
        operand for operand in remaining_operands if operand.type.is_quantum()
    ]
    param_operands = [
        operand
        for operand in remaining_operands
        if operand.type.is_classical() or operand.type.is_object()
    ]
    target_index_groups = [
        _expand_quantum_operands_to_phys(emit_pass, operand, qubit_map, bindings)
        for operand in target_qubit_operands
    ]
    target_indices = [index for group in target_index_groups for index in group]
    local_bindings = emit_pass._resolver.bind_block_params(
        op.block, param_operands, bindings
    )
    _bind_quantum_input_shapes(
        emit_pass, op.block, target_qubit_operands, bindings, local_bindings
    )
    inner_qubit_map = _build_quri_controlled_qubit_map(
        emit_pass, op.block, target_indices, local_bindings
    )
    effective_controls = outer_control_indices + nested_controls
    power = resolve_power(emit_pass, op, bindings)
    for _ in range(power):
        _emit_quri_controlled_operations(
            emit_pass,
            circuit,
            op.block.operations,
            effective_controls,
            inner_qubit_map,
            local_bindings,
        )
    _map_controlled_u_results(
        op,
        op.num_controls,
        nested_controls,
        target_qubit_operands,
        target_index_groups,
        qubit_map,
    )


def _emit_quri_controlled_operations(
    emit_pass: StandardEmitPass[Any],
    circuit: Any,
    operations: Sequence[Any],
    control_indices: list[int],
    qubit_map: QubitMap,
    bindings: dict[str, Any],
) -> None:
    """Recursively emit QURI Parts operations under accumulated controls.

    Args:
        emit_pass (StandardEmitPass[Any]): QURI Parts emit pass.
        circuit (Any): QURI Parts circuit being built.
        operations (Sequence[Any]): Block operations to walk.
        control_indices (list[int]): Accumulated physical controls.
        qubit_map (QubitMap): Mutable block-local qubit map.
        bindings (dict[str, Any]): Bindings visible in this block.

    Raises:
        EmitError: If an operation cannot be emitted by the guarded QURI
            Parts recursive fallback.
    """
    for op in operations:
        if isinstance(op, ReturnOperation):
            continue
        if isinstance(op, GateOperation):
            target_indices = _resolve_quri_gate_targets(
                emit_pass, op, qubit_map, bindings
            )
            _emit_quri_controlled_gate(
                emit_pass, circuit, op, control_indices, target_indices, bindings
            )
            _propagate_quri_gate_results(op, target_indices, qubit_map)
            continue
        if isinstance(op, ControlledUOperation):
            _emit_quri_nested_controlled_u(
                emit_pass, circuit, op, control_indices, qubit_map, bindings
            )
            continue
        if isinstance(op, ForOperation):
            from qamomile.circuit.transpiler.passes.emit_support.control_flow_emission import (
                _bind_loop_var,
                resolve_loop_bounds,
            )

            start, stop, step = resolve_loop_bounds(emit_pass._resolver, op, bindings)
            if start is None or stop is None or step is None:
                raise EmitError(
                    "Cannot resolve ForOperation bounds in QURI Parts "
                    "recursive controlled fallback.",
                    operation="ControlledUOperation",
                )
            for i in range(start, stop, step):
                loop_bindings = bindings.copy()
                _bind_loop_var(loop_bindings, op, i)
                _emit_quri_controlled_operations(
                    emit_pass,
                    circuit,
                    op.operations,
                    control_indices,
                    qubit_map,
                    loop_bindings,
                )
            continue
        raise EmitError(
            "QURI Parts recursive controlled fallback only supports "
            "primitive gates, nested ControlledUOperation values, "
            "ReturnOperation, and statically resolved ForOperation bodies. "
            f"Unsupported operation: {type(op).__name__}.",
            operation="ControlledUOperation",
        )


def _create_seeded_qulacs_vector_sampler(seed: int) -> Any:
    """Create a qulacs vector sampler that seeds its measurement RNG.

    The high-level ``create_qulacs_vector_sampler`` exposed by QURI Parts
    does not thread a random seed down to qulacs, so this helper reproduces
    its qulacs state-vector sampling path while forwarding ``seed`` to
    ``qulacs.QuantumState.sampling``. Sampling the same circuit with the
    same seed therefore yields identical measurement counts.

    Unlike the default QURI Parts sampler, this path does not switch to the
    multinomial state-vector fast-path at very large shot counts (that
    branch is unseedable upstream); it always uses ``QuantumState.sampling``.
    The resulting distribution is statistically identical, only potentially
    slower for very large shot counts.

    Args:
        seed (int): Random seed forwarded to ``QuantumState.sampling`` on
            every call, making sampling deterministic.

    Returns:
        Any: A sampler callable taking ``(circuit, shots)`` and returning a
            ``collections.Counter`` mapping basis-state integers to counts.

    Raises:
        ImportError: If quri-parts-qulacs (or qulacs) is not installed.
    """
    from collections import Counter

    import qulacs  # type: ignore[import-not-found]

    from quri_parts.qulacs.circuit import (  # type: ignore[import-not-found]
        convert_circuit,
    )

    def sampler(circuit: Any, shots: int) -> Any:
        """Sample ``circuit`` for ``shots`` shots using the fixed seed.

        Args:
            circuit (Any): The QURI Parts circuit to sample.
            shots (int): Number of measurement shots.

        Returns:
            Any: A ``collections.Counter`` mapping basis-state integers to
                their observed counts.

        Raises:
            Exception: Propagates any qulacs / QURI Parts circuit-conversion
                or sampling error raised for a malformed circuit.
        """
        state = qulacs.QuantumState(circuit.qubit_count)
        convert_circuit(circuit).update_quantum_state(state)
        return Counter(state.sampling(shots, seed))

    return sampler


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
        super().__init__(emitter, bindings, parameters, composite_emitters)  # type: ignore[arg-type]

    def _blockvalue_to_gate(
        self,
        block_value: Any,
        num_qubits: int,
        bindings: dict[str, Any],
    ) -> None:
        """Return no native gate object for QURI Parts controlled blocks.

        QURI Parts' emitter cannot turn a temporary circuit into a gate
        object that can be appended and controlled later.  The shared
        probe would still build a sub-circuit before discovering that
        ``circuit_to_gate()`` returns ``None``, which can pollute the
        parent emitter's current-circuit and parameter state.  Skip that
        probe and let ``_emit_controlled_fallback`` try the shared
        gate-by-gate decomposition directly.

        Args:
            block_value (Any): Ignored inner block value.
            num_qubits (int): Ignored target qubit count.
            bindings (dict[str, Any]): Ignored local bindings.

        Returns:
            None: Always signals that the backend-specific fallback must
            handle the controlled block.
        """
        del block_value, num_qubits, bindings
        return None

    def _emit_controlled_fallback(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        block_value: Any,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        power: int,
        bindings: dict[str, Any],
    ) -> None:
        """Emit controlled-U fallback through gate-by-gate decomposition.

        QURI Parts does not expose a custom-gate object that can be
        returned by ``circuit_to_gate`` and then controlled.  Avoid
        replacing that missing primitive with a dense matrix fallback:
        full-unitary extraction scales exponentially and can hide shapes
        that the backend cannot route gate-by-gate.  Instead, delegate to
        the shared controlled decomposition and surface its ``EmitError``
        if the block shape is unsupported.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit being emitted.
            block_value (Any): Inner controlled-U block.
            num_controls (int): Number of physical control qubits.
            control_indices (list[int]): Physical control qubits.
            target_indices (list[int]): Physical target qubits.
            power (int): Positive power to apply to the inner unitary
                before controlling it.
            bindings (dict[str, Any]): Local block bindings.

        Raises:
            EmitError: If safe gate-by-gate controlled decomposition
                cannot emit the block.
        """
        if not target_indices:
            return
        if not hasattr(block_value, "operations"):
            raise EmitError(
                "Cannot emit QURI Parts controlled fallback: block has no operations.",
                operation="ControlledUOperation",
            )
        if num_controls != len(control_indices):
            raise EmitError(
                "QURI Parts controlled fallback received inconsistent "
                f"control metadata: num_controls={num_controls}, "
                f"control_indices={control_indices!r}.",
                operation="ControlledUOperation",
            )
        qubit_map = _build_quri_controlled_qubit_map(
            self, block_value, target_indices, bindings
        )
        for _ in range(power):
            _emit_quri_controlled_operations(
                self,
                circuit,
                block_value.operations,
                control_indices,
                qubit_map,
                bindings,
            )


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
        seed: int | None = None,
    ):
        """Initialize executor with optional sampler, estimator, and seed.

        Args:
            sampler (Any): QURI Parts sampler. Defaults to None, meaning a
                qulacs vector sampler is created lazily on first use.
            estimator (Any): QURI Parts parametric estimator. Defaults to
                None, meaning a qulacs parametric estimator is created
                lazily on first use.
            seed (int | None): Optional random seed forwarded to the qulacs
                vector sampler so that sampling is reproducible. When set,
                two ``execute`` calls with the same seed and circuit return
                identical shot counts, which unblocks reproducible
                tutorials and benchmarks. Defaults to None, meaning
                sampling is non-deterministic. The seed is only applied to
                the default qulacs sampler; it is ignored when a custom
                ``sampler`` is supplied, since an arbitrary sampler has no
                standard seed interface.
        """
        self._sampler = sampler
        self._estimator = estimator
        self._non_parametric_estimator: Any = None
        self._seed = seed

    @property
    def sampler(self) -> Any:
        """Lazy initialization of sampler.

        When a ``seed`` was supplied to the constructor (and no custom
        sampler was given), a seeded qulacs vector sampler is created so
        that sampling is reproducible.

        Returns:
            Any: A QURI Parts sampler callable taking ``(circuit, shots)``
                and returning measurement counts.

        Raises:
            ImportError: If quri-parts-qulacs is not installed.
        """
        if self._sampler is None:
            try:
                if self._seed is None:
                    from quri_parts.qulacs.sampler import (  # type: ignore[import-not-found]
                        create_qulacs_vector_sampler,
                    )

                    self._sampler = create_qulacs_vector_sampler()
                else:
                    self._sampler = _create_seeded_qulacs_vector_sampler(self._seed)
            except ImportError as e:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from e
        return self._sampler

    @property
    def parametric_estimator(self) -> Any:
        """Lazy initialization of parametric estimator for optimization."""
        if self._estimator is None:
            try:
                from quri_parts.qulacs.estimator import (  # type: ignore[import-not-found]
                    create_qulacs_vector_parametric_estimator,
                )

                self._estimator = create_qulacs_vector_parametric_estimator()
            except ImportError as e:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from e
        return self._estimator

    @property
    def non_parametric_estimator(self) -> Any:
        """Lazy initialization of non-parametric estimator.

        Used when the circuit has already been bound (parameters resolved)
        or is a non-parametric circuit. Unlike the parametric estimator,
        this takes (operator, state) without parameter values.
        """
        if self._non_parametric_estimator is None:
            try:
                from quri_parts.qulacs.estimator import (  # type: ignore[import-not-found]
                    create_qulacs_vector_estimator,
                )

                self._non_parametric_estimator = create_qulacs_vector_estimator()
            except ImportError as e:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from e
        return self._non_parametric_estimator

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Execute circuit and return bitstring counts.

        Args:
            circuit: The quantum circuit to execute (bound or unbound)
            shots: Number of measurement shots

        Returns:
            Dictionary mapping bitstrings to counts (e.g., {"00": 512, "11": 512})
        """
        counter = self.sampler(circuit, shots)

        num_qubits = circuit.qubit_count
        return {format(k, f"0{num_qubits}b"): v for k, v in counter.items()}

    def bind_parameters(  # type: ignore[override]
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

        Raises:
            QamomileQuriPartsTranspileError: If a required parameter binding
                is missing from ``bindings``.
        """
        param_values = []
        for param_info in parameter_metadata.parameters:
            if param_info.name in bindings:
                param_values.append(float(bindings[param_info.name]))
            else:
                raise QamomileQuriPartsTranspileError(
                    f"Missing binding for parameter '{param_info.name}'. "
                    f"Provided bindings: {list(bindings.keys())}. "
                    f"Required parameters: "
                    f"{[p.name for p in parameter_metadata.parameters]}"
                )

        return circuit.bind_parameters(param_values)

    def estimate(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate expectation value <psi|H|psi>.

        Also accepts a native ``quri_parts.core.operator.Operator`` at
        runtime for convenience (auto-conversion is skipped in that case).

        Args:
            circuit: The unbound parametric circuit (state preparation ansatz)
            hamiltonian: Hamiltonian to measure (qamomile Hamiltonian or
                native QURI Parts Operator at runtime)
            params: Parameter values for the parametric circuit

        Returns:
            Real part of the expectation value
        """
        import qamomile.observable as qm_o

        if isinstance(hamiltonian, qm_o.Hamiltonian):
            from qamomile.quri_parts.observable import hamiltonian_to_quri_operator

            hamiltonian = hamiltonian_to_quri_operator(hamiltonian)  # type: ignore[assignment]

        return self.estimate_expectation(circuit, hamiltonian, params or [])  # type: ignore[arg-type]

    def estimate_expectation(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        hamiltonian: "qp_o.Operator",
        param_values: Sequence[float],
    ) -> float:
        """Estimate expectation value of hamiltonian for a circuit.

        Handles both unbound parametric circuits (used during optimization)
        and already-bound or non-parametric circuits. When the circuit has
        already been bound (e.g., by ``_run_expval`` in ``ExecutableProgram``),
        ``apply_circuit`` produces a ``GeneralCircuitQuantumState`` instead
        of a ``ParametricCircuitQuantumState``, so we dispatch to the
        non-parametric estimator automatically.

        Args:
            circuit: The quantum circuit (unbound parametric or bound/concrete)
            hamiltonian: QURI Parts Operator representing the Hamiltonian
            param_values: Sequence of parameter values in order (ignored for
                bound/non-parametric circuits)

        Returns:
            Real part of the expectation value
        """
        from quri_parts.core.state import (  # type: ignore[import-not-found]
            apply_circuit,
            quantum_state,
        )

        cb_state = quantum_state(circuit.qubit_count, bits=0)
        circuit_state = apply_circuit(circuit, cb_state)

        # Dispatch based on whether the state is parametric or not.
        # apply_circuit creates ParametricCircuitQuantumState for unbound
        # parametric circuits, and GeneralCircuitQuantumState for bound
        # or non-parametric circuits.
        if hasattr(circuit_state, "parametric_circuit"):
            # Unbound parametric circuit → use parametric estimator
            estimate = self.parametric_estimator(
                hamiltonian, circuit_state, param_values
            )
        else:
            # Bound or non-parametric circuit → use non-parametric estimator
            estimate = self.non_parametric_estimator(hamiltonian, circuit_state)

        return estimate.value.real


class QuriPartsTranspiler(
    Transpiler["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """QURI Parts transpiler for qamomile.circuit module.

    Converts Qamomile QKernels into QURI Parts quantum circuits.

    Example:
        from qamomile.quri_parts import QuriPartsTranspiler
        import qamomile.circuit as qm

        @qm.qkernel
        def bell_state(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Bit, qm.Bit]:
            q0 = qm.h(q0)
            q0, q1 = qm.cx(q0, q1)
            return qm.measure(q0), qm.measure(q1)

        transpiler = QuriPartsTranspiler()
        circuit = transpiler.to_circuit(bell_state)
    """

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create default segmentation pass (no backend-specific overrides)."""
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["qp_c.LinearMappedUnboundParametricQuantumCircuit"]:
        """Create QURI Parts emission pass with gate emitter."""
        return QuriPartsEmitPass(bindings, parameters)

    def executor(  # type: ignore[override]
        self,
        sampler: Any = None,
        estimator: Any = None,
        seed: int | None = None,
    ) -> QuriPartsExecutor:
        """Create a QURI Parts executor.

        Args:
            sampler (Any): Optional custom sampler. Defaults to None,
                meaning the qulacs vector sampler is used.
            estimator (Any): Optional custom estimator. Defaults to None,
                meaning the qulacs parametric estimator is used.
            seed (int | None): Optional random seed forwarded to the qulacs
                vector sampler for reproducible sampling. Defaults to None,
                meaning sampling is non-deterministic. The seed is ignored
                when a custom ``sampler`` is supplied, since an arbitrary
                sampler has no standard seed interface.

        Returns:
            QuriPartsExecutor: Executor configured for this backend, bound
                to the given seed.
        """
        return QuriPartsExecutor(sampler, estimator, seed=seed)
