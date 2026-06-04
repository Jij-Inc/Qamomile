"""QURI Parts backend transpiler implementation.

This module provides QuriPartsTranspiler for converting Qamomile QKernels
into QURI Parts quantum circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np

from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.circuit.ir.operation.gate import GateOperation
from qamomile.circuit.ir.operation.return_operation import ReturnOperation
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _bind_block_inputs,
    _populate_input_qubit_map,
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


def _ensure_quri_controlled_block_gate_by_gate_supported(block_value: Any) -> None:
    """Reject controlled block bodies QURI Parts cannot decompose gate-by-gate.

    QURI Parts has no backend gate object for controlled sub-circuits.
    The shared gate-by-gate fallback can only emit primitive gates and
    statically resolved ``ForOperation`` bodies. Everything else must fail
    before emission so unsupported operations are not silently dropped.

    Args:
        block_value (Any): Controlled-U block whose body will be emitted
            through the shared gate-by-gate fallback.

    Raises:
        EmitError: If the block body contains an operation that QURI Parts
            cannot safely emit under outer controls without dense unitary
            synthesis.
    """
    _ensure_quri_controlled_operations_gate_by_gate_supported(
        getattr(block_value, "operations", ())
    )


def _ensure_quri_controlled_operations_gate_by_gate_supported(
    operations: Sequence[Any],
) -> None:
    """Reject operations outside QURI Parts controlled gate-by-gate support.

    Args:
        operations (Sequence[Any]): Operations from a controlled block body
            or from a statically unrolled ``ForOperation`` body.

    Raises:
        EmitError: If any operation cannot be emitted by the shared
            primitive gate-by-gate controlled fallback.
    """
    for op in operations:
        if isinstance(op, (GateOperation, ReturnOperation)):
            continue
        if isinstance(op, ForOperation):
            _ensure_quri_controlled_operations_gate_by_gate_supported(op.operations)
            continue
        raise EmitError(
            "QURI Parts controlled fallback only supports primitive gate "
            "bodies and statically resolved for-loops. "
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
        input_operands: list[Any] | None = None,
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
            input_operands (list[Any] | None): Ignored call-site
                operands. Defaults to None.

        Returns:
            None: Always signals that the backend-specific fallback must
            handle the controlled block.
        """
        del block_value, num_qubits, bindings, input_operands
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
        """Emit controlled-U fallback through QURI Parts-specific paths.

        QURI Parts does not expose a custom-gate object that can be
        returned by ``circuit_to_gate`` and then controlled.  Single-target
        blocks use the shared primitive gate-by-gate decomposition.  A
        single-control, multi-target block cannot be routed by that shared
        path, so this backend materializes the small local block as a dense
        unitary, wraps it with one control, and appends it as a
        ``UnitaryMatrix`` gate.

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
                cannot emit the block and dense fallback is unavailable.
        """
        if not target_indices:
            return
        if len(target_indices) > 1 and self._try_emit_controlled_dense_matrix(
            circuit,
            block_value,
            num_controls,
            control_indices,
            target_indices,
            power,
            bindings,
        ):
            return
        _ensure_quri_controlled_block_gate_by_gate_supported(block_value)
        super()._emit_controlled_fallback(
            circuit,
            block_value,
            num_controls,
            control_indices,
            target_indices,
            power,
            bindings,
        )

    def _try_emit_controlled_dense_matrix(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        block_value: Any,
        num_controls: int,
        control_indices: list[int],
        target_indices: list[int],
        power: int,
        bindings: dict[str, Any],
    ) -> bool:
        """Try emitting a single-control block as a dense matrix gate.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            block_value (Any): Controlled-U body to materialize.
            num_controls (int): Number of control qubits requested by the
                IR operation.
            control_indices (list[int]): Parent-circuit control qubits.
            target_indices (list[int]): Parent-circuit target qubits.
            power (int): Positive power applied to the body before adding
                the control.
            bindings (dict[str, Any]): Local block bindings.

        Returns:
            bool: True if a dense controlled matrix was appended. False if
                the caller should fall back to the shared gate-by-gate path.
        """
        if num_controls != 1 or len(control_indices) != 1:
            return False
        if not hasattr(block_value, "operations"):
            return False

        emitter = cast(QuriPartsGateEmitter, self._emitter)
        saved_circuit = emitter._current_circuit
        saved_param_map = dict(emitter._param_map)
        try:
            local_qubit_map, local_clbit_map, local_bindings = (
                self._prepare_local_block_maps(
                    block_value,
                    None,
                    len(target_indices),
                    bindings,
                )
            )
            local_qubit_map, local_clbit_map = self._allocator.allocate(
                block_value.operations,
                local_bindings,
                initial_qubit_map=local_qubit_map,
                initial_clbit_map=local_clbit_map,
            )
            qubit_count = (
                max(local_qubit_map.values()) + 1
                if local_qubit_map
                else len(target_indices)
            )
            if qubit_count != len(target_indices):
                return False

            sub_circuit = self._emitter.create_circuit(qubit_count, 0)
            self._emit_operations(
                sub_circuit,
                block_value.operations,
                local_qubit_map,
                local_clbit_map,
                local_bindings,
                force_unroll=True,
            )
            unitary = self._circuit_unitary(sub_circuit)
            if power > 1:
                unitary = np.linalg.matrix_power(unitary, power)
            controlled = self._controlled_unitary_matrix(unitary, num_controls)

            from quri_parts.circuit import (
                UnitaryMatrix,  # type: ignore[import-not-found]
            )

            circuit.add_gate(
                UnitaryMatrix(
                    [*control_indices, *target_indices],
                    controlled.tolist(),
                )
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
            ImportError,
        ):
            return False
        finally:
            emitter._current_circuit = saved_circuit
            emitter._param_map = saved_param_map

        return True

    def _circuit_unitary(self, circuit: Any) -> np.ndarray:
        """Materialize a non-parametric QURI Parts circuit as a unitary.

        Args:
            circuit (Any): QURI Parts circuit whose gates contain no
                unresolved runtime parameters.

        Returns:
            np.ndarray: Dense unitary matrix in QURI Parts' local
                little-endian basis order.

        Raises:
            ImportError: If quri-parts-qulacs or qulacs is unavailable.
            RuntimeError: If QURI Parts or qulacs rejects the circuit.
        """
        import qulacs  # type: ignore[import-not-found]

        from quri_parts.qulacs.circuit import (  # type: ignore[import-not-found]
            convert_circuit,
        )

        converted = convert_circuit(circuit)
        dimension = 2**circuit.qubit_count
        unitary = np.empty((dimension, dimension), dtype=np.complex128)
        for basis in range(dimension):
            state = qulacs.QuantumState(circuit.qubit_count)
            state.set_computational_basis(basis)
            converted.update_quantum_state(state)
            unitary[:, basis] = state.get_vector()
        return unitary

    def _controlled_unitary_matrix(
        self,
        unitary: np.ndarray,
        num_controls: int,
    ) -> np.ndarray:
        """Build a dense controlled-unitary matrix.

        Args:
            unitary (np.ndarray): Target-only unitary matrix in little-endian
                basis order.
            num_controls (int): Number of leading local control qubits.

        Returns:
            np.ndarray: Matrix that applies ``unitary`` when all controls
                are one and identity otherwise.
        """
        target_dimension = unitary.shape[0]
        control_dimension = 2**num_controls
        controlled_dimension = control_dimension * target_dimension
        controlled = np.eye(controlled_dimension, dtype=np.complex128)
        active_control = control_dimension - 1

        for target_in in range(target_dimension):
            in_index = target_in * control_dimension + active_control
            controlled[:, in_index] = 0.0
            for target_out in range(target_dimension):
                out_index = target_out * control_dimension + active_control
                controlled[out_index, in_index] = unitary[target_out, target_in]
        return controlled

    def _emit_custom_composite(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        op: Any,
        impl: Any,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a custom composite operation into a QURI Parts circuit.

        QURI Parts has no reusable gate object with call-site qubit
        remapping, so normal custom composites are emitted inline. For
        ``inverse(qkernel)`` composites, a non-parametric source block can
        still use QURI Parts' own ``inverse_circuit`` helper before being
        remapped and appended to the parent circuit.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            op (Any): Composite operation being emitted.
            impl (Any): Qamomile fallback implementation block.
            qubit_indices (list[int]): Parent-circuit qubit indices for
                the composite operation.
            bindings (dict[str, Any]): Active compile-time and runtime
                parameter bindings.
        """
        inverse_source_block = getattr(op, "inverse_source_block", None)
        if inverse_source_block is not None and self._try_emit_backend_inverse(
            circuit,
            inverse_source_block,
            getattr(op, "operands", None),
            qubit_indices,
            bindings,
        ):
            return

        self._emit_block_inline(
            circuit,
            impl,
            getattr(op, "operands", None),
            qubit_indices,
            bindings,
        )

    def _try_emit_backend_inverse(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        block_value: Any,
        input_operands: list[Any] | None,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> bool:
        """Try emitting ``block_value`` via QURI Parts' circuit inverse.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit to append into on success.
            block_value (Any): Source block whose inverse should be emitted.
            input_operands (list[Any] | None): Call-site operands used to
                bind the source block inputs.
            qubit_indices (list[int]): Parent-circuit qubits occupied by
                the source block.
            bindings (dict[str, Any]): Active emit bindings.

        Returns:
            bool: True when backend-native inversion was emitted, False
                when the caller should fall back to the Qamomile inverse
                implementation block.
        """
        if not hasattr(block_value, "operations"):
            return False

        try:
            local_qubit_map, local_clbit_map, local_bindings = (
                self._prepare_local_block_maps(
                    block_value,
                    input_operands,
                    len(qubit_indices),
                    bindings,
                )
            )
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
        ):
            return False

        emitter = cast(QuriPartsGateEmitter, self._emitter)
        saved_circuit = emitter._current_circuit
        saved_param_map = dict(emitter._param_map)
        try:
            local_qubit_map, local_clbit_map = self._allocator.allocate(
                block_value.operations,
                local_bindings,
                initial_qubit_map=local_qubit_map,
                initial_clbit_map=local_clbit_map,
            )
            qubit_count = (
                max(local_qubit_map.values()) + 1
                if local_qubit_map
                else len(qubit_indices)
            )
            if qubit_count > len(qubit_indices):
                return False

            sub_circuit = self._emitter.create_circuit(qubit_count, 0)
            self._emit_operations(
                sub_circuit,
                block_value.operations,
                local_qubit_map,
                local_clbit_map,
                local_bindings,
                force_unroll=True,
            )
            inverse_circuit = self._emitter.gate_inverse(sub_circuit)
        except (
            AttributeError,
            TypeError,
            ValueError,
            KeyError,
            IndexError,
            RuntimeError,
        ):
            return False
        finally:
            emitter._current_circuit = saved_circuit
            emitter._param_map = saved_param_map

        if inverse_circuit is None:
            return False

        try:
            self._append_remapped_circuit(circuit, inverse_circuit, qubit_indices)
        except (AttributeError, TypeError, ValueError, IndexError, RuntimeError):
            return False
        return True

    def _emit_block_inline(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        block_value: Any,
        input_operands: list[Any] | None,
        qubit_indices: list[int],
        bindings: dict[str, Any],
    ) -> None:
        """Emit a nested implementation block directly into ``circuit``.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            block_value (Any): Implementation block to emit.
            input_operands (list[Any] | None): Call-site operands used to
                bind the block inputs.
            qubit_indices (list[int]): Parent-circuit qubits occupied by
                the nested block.
            bindings (dict[str, Any]): Active emit bindings.
        """
        if not hasattr(block_value, "operations"):
            return

        local_qubit_map, local_clbit_map, local_bindings = (
            self._prepare_local_block_maps(
                block_value,
                input_operands,
                len(qubit_indices),
                bindings,
                parent_qubits=qubit_indices,
            )
        )
        self._emit_operations(
            circuit,
            block_value.operations,
            local_qubit_map,
            local_clbit_map,
            local_bindings,
            force_unroll=True,
        )

    def _prepare_local_block_maps(
        self,
        block_value: Any,
        input_operands: list[Any] | None,
        num_qubits: int,
        bindings: dict[str, Any],
        parent_qubits: list[int] | None = None,
    ) -> tuple[dict[Any, int], dict[Any, int], dict[str, Any]]:
        """Prepare local value maps for nested QURI Parts block emission.

        Args:
            block_value (Any): Nested block whose inputs should be mapped.
            input_operands (list[Any] | None): Call-site operands used to
                bind quantum and classical block inputs.
            num_qubits (int): Local qubit width of the nested operation.
            bindings (dict[str, Any]): Active emit bindings.
            parent_qubits (list[int] | None): Optional parent-circuit
                qubit indices used to remap local addresses. Defaults to
                None, leaving addresses in local ``0..num_qubits-1`` form.

        Returns:
            tuple[dict[Any, int], dict[Any, int], dict[str, Any]]: Local
                qubit map, local classical-bit map, and nested bindings.
        """
        local_qubit_map: dict[Any, int] = {}
        local_clbit_map: dict[Any, int] = {}
        local_bindings = _bind_block_inputs(
            self,
            block_value,
            input_operands,
            num_qubits,
            bindings,
            local_qubit_map,
        )
        if hasattr(block_value, "input_values"):
            _populate_input_qubit_map(
                self,
                block_value.input_values,
                num_qubits,
                local_bindings,
                local_qubit_map,
            )

        if parent_qubits is not None:
            local_qubit_map = {
                address: parent_qubits[local_index]
                for address, local_index in local_qubit_map.items()
            }

        return local_qubit_map, local_clbit_map, local_bindings

    def _append_remapped_circuit(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        source_circuit: Any,
        qubit_indices: list[int],
    ) -> None:
        """Append ``source_circuit`` gates after remapping their qubits.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parent QURI Parts circuit.
            source_circuit (Any): Non-parametric QURI Parts circuit whose
                gates should be appended.
            qubit_indices (list[int]): Parent-circuit qubits corresponding
                to local qubit indices in ``source_circuit``.

        Raises:
            AttributeError: If ``source_circuit`` does not expose gates.
            IndexError: If a gate references a local qubit outside
                ``qubit_indices``.
            TypeError: If QURI Parts rejects a rebuilt gate.
            ValueError: If QURI Parts rejects a rebuilt gate.
        """
        remapped_gates = [
            self._remap_gate(gate, qubit_indices) for gate in source_circuit.gates
        ]
        for gate in remapped_gates:
            circuit.add_gate(gate)

    def _remap_gate(self, gate: Any, qubit_indices: list[int]) -> Any:
        """Rebuild a QURI Parts gate with parent-circuit qubit indices.

        Args:
            gate (Any): QURI Parts ``QuantumGate`` from a local circuit.
            qubit_indices (list[int]): Parent-circuit qubits corresponding
                to local qubit indices.

        Returns:
            Any: Rebuilt QURI Parts ``QuantumGate`` with remapped target and
                control indices.

        Raises:
            IndexError: If ``gate`` references a local qubit outside
                ``qubit_indices``.
            TypeError: If QURI Parts rejects the rebuilt gate.
            ValueError: If QURI Parts rejects the rebuilt gate.
        """
        from quri_parts.circuit import QuantumGate  # type: ignore[import-not-found]

        return QuantumGate(
            name=gate.name,
            target_indices=tuple(qubit_indices[index] for index in gate.target_indices),
            control_indices=tuple(
                qubit_indices[index] for index in gate.control_indices
            ),
            classical_indices=gate.classical_indices,
            params=gate.params,
            pauli_ids=gate.pauli_ids,
            unitary_matrix=gate.unitary_matrix,
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
