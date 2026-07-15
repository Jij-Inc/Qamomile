"""QURI Parts backend transpiler implementation.

This module provides QuriPartsTranspiler for converting Qamomile QKernels
into QURI Parts quantum circuits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.transpiler.circuit_ir import CircuitBackendEmitPass
from qamomile.circuit.transpiler.executable import (
    ParameterMetadata,
    QuantumExecutor,
)
from qamomile.circuit.transpiler.passes.emit import EmitPass
from qamomile.circuit.transpiler.passes.separate import SegmentationPass
from qamomile.circuit.transpiler.transpiler import Transpiler
from qamomile.quri_parts.exceptions import QamomileQuriPartsTranspileError
from qamomile.quri_parts.materializer import QuriPartsMaterializer

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


class QuriPartsExecutor(
    QuantumExecutor["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """Execute QURI Parts circuits with sampling and expectation estimation."""

    def __init__(
        self,
        sampler: Any = None,
        estimator: Any = None,
        seed: int | None = None,
    ):
        """Initialize the executor.

        Args:
            sampler: Optional QURI Parts sampler.
            estimator: Optional QURI Parts parametric estimator.
            seed: Optional seed for the default Qulacs sampler.
        """
        self._sampler = sampler
        self._estimator = estimator
        self._non_parametric_estimator: Any = None
        self._seed = seed

    @property
    def sampler(self) -> Any:
        """Return the configured sampler, creating the default lazily."""
        if self._sampler is None:
            try:
                if self._seed is None:
                    from quri_parts.qulacs.sampler import (  # type: ignore[import-not-found]
                        create_qulacs_vector_sampler,
                    )

                    self._sampler = create_qulacs_vector_sampler()
                else:
                    self._sampler = _create_seeded_qulacs_vector_sampler(self._seed)
            except ImportError as error:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from error
        return self._sampler

    @property
    def parametric_estimator(self) -> Any:
        """Return the parametric estimator, creating the default lazily."""
        if self._estimator is None:
            try:
                from quri_parts.qulacs.estimator import (  # type: ignore[import-not-found]
                    create_qulacs_vector_parametric_estimator,
                )

                self._estimator = create_qulacs_vector_parametric_estimator()
            except ImportError as error:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from error
        return self._estimator

    @property
    def non_parametric_estimator(self) -> Any:
        """Return the non-parametric estimator, creating it lazily."""
        if self._non_parametric_estimator is None:
            try:
                from quri_parts.qulacs.estimator import (  # type: ignore[import-not-found]
                    create_qulacs_vector_estimator,
                )

                self._non_parametric_estimator = create_qulacs_vector_estimator()
            except ImportError as error:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from error
        return self._non_parametric_estimator

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Sample a circuit and return bitstring counts.

        Args:
            circuit: Bound or unbound QURI Parts circuit.
            shots: Number of measurement shots.

        Returns:
            Counts keyed by zero-padded bitstrings. A zero-qubit circuit
            returns ``{"": shots}`` without invoking the sampler.
        """
        if circuit.qubit_count == 0:
            return {"": shots}

        counter = self.sampler(circuit, shots)
        return {
            format(value, f"0{circuit.qubit_count}b"): count
            for value, count in counter.items()
        }

    def bind_parameters(  # type: ignore[override]
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "ImmutableBoundParametricQuantumCircuit":
        """Bind named parameters in backend order.

        Args:
            circuit: Unbound parametric circuit.
            bindings: Parameter values by Qamomile name.
            parameter_metadata: Ordered backend parameter metadata.

        Returns:
            Bound QURI Parts circuit.

        Raises:
            QamomileQuriPartsTranspileError: If a required value is absent.
        """
        values = []
        for parameter in parameter_metadata.parameters:
            if parameter.name not in bindings:
                raise QamomileQuriPartsTranspileError(
                    f"Missing binding for parameter '{parameter.name}'. "
                    f"Provided bindings: {list(bindings)}. Required parameters: "
                    f"{[item.name for item in parameter_metadata.parameters]}"
                )
            values.append(float(bindings[parameter.name]))
        return circuit.bind_parameters(values)

    def estimate(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        hamiltonian: "qm_o.Hamiltonian",
        params: Sequence[float] | None = None,
    ) -> float:
        """Estimate a Hamiltonian expectation value.

        Args:
            circuit: State-preparation circuit.
            hamiltonian: Qamomile Hamiltonian or native QURI Parts operator.
            params: Optional circuit parameter values.

        Returns:
            Real expectation value.
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
        """Estimate a native QURI Parts operator expectation value.

        Args:
            circuit: Parametric or concrete QURI Parts circuit.
            hamiltonian: Native QURI Parts operator.
            param_values: Values for an unbound parametric circuit.

        Returns:
            Real expectation value.
        """
        from quri_parts.core.state import (  # type: ignore[import-not-found]
            apply_circuit,
            quantum_state,
        )

        state = apply_circuit(
            circuit,
            quantum_state(circuit.qubit_count, bits=0),
        )
        if hasattr(state, "parametric_circuit"):
            estimate = self.parametric_estimator(hamiltonian, state, param_values)
        else:
            estimate = self.non_parametric_estimator(hamiltonian, state)
        return estimate.value.real


class QuriPartsTranspiler(
    Transpiler["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """Transpile Qamomile programs to QURI Parts circuits."""

    def _create_segmentation_pass(self) -> SegmentationPass:
        """Create the default segmentation pass."""
        return SegmentationPass()

    def _create_emit_pass(
        self,
        bindings: dict[str, Any] | None = None,
        parameters: list[str] | None = None,
    ) -> EmitPass["qp_c.LinearMappedUnboundParametricQuantumCircuit"]:
        """Create the common circuit-backend emission pass.

        Args:
            bindings: Compile-time argument values.
            parameters: Arguments preserved as runtime parameters.

        Returns:
            Emit pass backed by the QURI Parts materializer.
        """
        return CircuitBackendEmitPass(
            QuriPartsMaterializer(),
            bindings,
            parameters,
        )

    def executor(  # type: ignore[override]
        self,
        sampler: Any = None,
        estimator: Any = None,
        seed: int | None = None,
    ) -> QuriPartsExecutor:
        """Create a QURI Parts executor.

        Args:
            sampler: Optional custom sampler.
            estimator: Optional custom estimator.
            seed: Optional seed for the default sampler.

        Returns:
            Configured QURI Parts executor.
        """
        return QuriPartsExecutor(sampler, estimator, seed=seed)
