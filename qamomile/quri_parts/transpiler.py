"""QURI Parts backend transpiler implementation.

This module provides QuriPartsTranspiler for converting Qamomile QKernels
into QURI Parts quantum circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
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
