"""QURI Parts backend transpiler implementation.

This module provides QuriPartsTranspiler for converting Qamomile QKernels
into QURI Parts quantum circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from qamomile.circuit.ir.parameter import ParamKind
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.emit_support.controlled_emission import (
    _populate_input_qubit_map,
)
from qamomile.circuit.transpiler.passes.emit_support.qubit_address import (
    ClbitMap,
    QubitMap,
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


def _block_has_unbound_runtime_parameters(
    block_value: Any,
    bindings: dict[str, Any],
) -> bool:
    """Return whether a block still has unbound runtime parameters.

    Args:
        block_value (Any): Block-like object to inspect.
        bindings (dict[str, Any]): Local block bindings. A runtime slot
            with a concrete binding can still be matrix-emitted.

    Returns:
        bool: True when a runtime parameter slot has no concrete binding
        or is bound to another symbolic IR value.
    """
    for slot in getattr(block_value, "param_slots", ()):
        if slot.kind != ParamKind.RUNTIME_PARAMETER:
            continue
        if slot.name not in bindings:
            return True
        bound = bindings[slot.name]
        if hasattr(bound, "uuid") and hasattr(bound, "type"):
            return True
    return False


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
        probe and let ``_emit_controlled_fallback`` choose between dense
        matrix emission and per-gate decomposition.

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
        """Emit controlled-U fallback through matrix or per-gate decomposition.

        QURI Parts does not expose a custom-gate object that can be
        returned by ``circuit_to_gate`` and then controlled.  For
        concrete blocks, build the wrapped block as a small QURI circuit,
        extract its unitary with Qulacs, extend that unitary with the
        requested control qubits, and append it as one dense matrix gate.
        If matrix extraction is impossible because the block still
        carries runtime parameters, fall back to the shared per-gate
        controlled decomposition; that path preserves parametric gates
        but still rejects multi-target shapes it cannot route safely.

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
            QamomileQuriPartsTranspileError: If both dense matrix
                extraction and safe per-gate controlled decomposition
                fail.
        """
        if not target_indices:
            return
        if _block_has_unbound_runtime_parameters(block_value, bindings):
            try:
                super()._emit_controlled_fallback(
                    circuit,
                    block_value,
                    num_controls,
                    control_indices,
                    target_indices,
                    power,
                    bindings,
                )
            except EmitError as fallback_error:
                raise QamomileQuriPartsTranspileError(
                    "QURI Parts controlled fallback cannot safely "
                    "decompose the runtime-parameter block gate-by-gate."
                ) from fallback_error
            return
        try:
            unitary = self._block_to_unitary(block_value, len(target_indices), bindings)
        except QamomileQuriPartsTranspileError:
            try:
                super()._emit_controlled_fallback(
                    circuit,
                    block_value,
                    num_controls,
                    control_indices,
                    target_indices,
                    power,
                    bindings,
                )
            except EmitError as fallback_error:
                raise QamomileQuriPartsTranspileError(
                    "QURI Parts controlled fallback could neither "
                    "matrix-emit the block nor decompose it gate-by-gate."
                ) from fallback_error
            return
        else:
            if power > 1:
                unitary = np.linalg.matrix_power(unitary, power)
            controlled = _controlled_unitary(unitary, num_controls)
            circuit.add_UnitaryMatrix_gate(
                [*control_indices, *target_indices],
                controlled.tolist(),
            )

    def _block_to_unitary(
        self,
        block_value: Any,
        num_qubits: int,
        bindings: dict[str, Any],
    ) -> np.ndarray:
        """Convert a concrete inner block to a unitary matrix.

        Args:
            block_value (Any): Block-like object with ``operations`` and
                optional ``input_values``.
            num_qubits (int): Number of local target qubits the block
                occupies.
            bindings (dict[str, Any]): Bindings used while emitting the
                local block.

        Returns:
            np.ndarray: Dense unitary matrix for the block.

        Raises:
            QamomileQuriPartsTranspileError: If the block is malformed,
                still parameterized, or cannot be converted by Qulacs.
        """
        if not hasattr(block_value, "operations"):
            raise QamomileQuriPartsTranspileError(
                "QURI Parts controlled fallback requires a block with operations."
            )

        try:
            local_qubit_map: QubitMap = {}
            local_clbit_map: ClbitMap = {}
            if hasattr(block_value, "input_values"):
                _populate_input_qubit_map(
                    self,
                    block_value.input_values,
                    num_qubits,
                    bindings,
                    local_qubit_map,
                )

            local_qubit_map, local_clbit_map = self._allocator.allocate(
                block_value.operations,
                bindings,
                initial_qubit_map=local_qubit_map,
                initial_clbit_map=local_clbit_map,
            )

            sub_circuit = None
            previous_circuit = self._emitter._current_circuit
            previous_param_map = dict(self._emitter._param_map)
            previous_parameter_map = dict(self._parameter_map)
            previous_parameter_sources = dict(self._parameter_sources)
            try:
                sub_circuit = self._emitter.create_circuit(num_qubits, 0)
                self._emit_operations(
                    sub_circuit,
                    block_value.operations,
                    local_qubit_map,
                    local_clbit_map,
                    bindings,
                    force_unroll=True,
                )
            finally:
                self._emitter._current_circuit = previous_circuit
                self._emitter._param_map = previous_param_map
                self._parameter_map = previous_parameter_map
                self._parameter_sources = previous_parameter_sources
            if sub_circuit is None:
                raise QamomileQuriPartsTranspileError(
                    "QURI Parts controlled fallback failed to create a sub-circuit."
                )
            if sub_circuit.parameter_count:
                raise QamomileQuriPartsTranspileError(
                    "QURI Parts controlled fallback cannot matrix-emit a "
                    "controlled-U block with runtime parameters. Bind the "
                    "parameters at transpile time."
                )
            frozen = sub_circuit.freeze()
            return _quri_circuit_unitary(frozen)
        except QamomileQuriPartsTranspileError:
            raise
        except Exception as exc:
            raise QamomileQuriPartsTranspileError(
                "Failed to matrix-emit a controlled-U block for QURI Parts."
            ) from exc


def _quri_circuit_unitary(circuit: Any) -> np.ndarray:
    """Return the dense unitary matrix for a concrete QURI circuit.

    Args:
        circuit (Any): Immutable concrete QURI Parts circuit.

    Returns:
        np.ndarray: Matrix whose columns are the evolved computational
        basis states.
    """
    from qulacs import QuantumState

    from quri_parts.qulacs.circuit import convert_circuit

    qulacs_circuit = convert_circuit(circuit)
    num_qubits = qulacs_circuit.get_qubit_count()
    dim = 1 << num_qubits
    unitary = np.zeros((dim, dim), dtype=np.complex128)
    for basis in range(dim):
        state = QuantumState(num_qubits)
        state.set_computational_basis(basis)
        qulacs_circuit.update_quantum_state(state)
        unitary[:, basis] = state.get_vector()
    return unitary


def _controlled_unitary(unitary: np.ndarray, num_controls: int) -> np.ndarray:
    """Build a controlled version of ``unitary`` in Qulacs bit order.

    Args:
        unitary (np.ndarray): Target-only unitary matrix.
        num_controls (int): Number of leading control qubits.

    Returns:
        np.ndarray: Dense matrix over ``controls + targets`` that applies
        ``unitary`` exactly when all controls are ``|1>``.
    """
    target_dim = unitary.shape[0]
    total_dim = target_dim << num_controls
    controlled = np.eye(total_dim, dtype=np.complex128)
    target_mask = target_dim - 1
    active_control_bits = (1 << num_controls) - 1
    for in_basis in range(total_dim):
        control_bits = in_basis & active_control_bits
        if control_bits != active_control_bits:
            continue
        in_target = (in_basis >> num_controls) & target_mask
        controlled[in_basis, in_basis] = 0.0
        for out_target in range(target_dim):
            out_basis = control_bits | (out_target << num_controls)
            controlled[out_basis, in_basis] = unitary[out_target, in_target]
    return controlled


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
