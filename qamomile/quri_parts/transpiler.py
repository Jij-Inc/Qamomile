"""QURI Parts engine transpiler implementation.

This module provides QuriPartsTranspiler for converting Qamomile QKernels
into QURI Parts quantum circuits.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, Sequence

from qamomile.circuit.transpiler.circuit_ir import CircuitEngineEmitPass
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


def _normalize_sample_count(count: object) -> int:
    """Convert an SDK count to a nonnegative Python integer without rounding.

    Args:
        count (object): One count returned by a QURI Parts sampler.

    Returns:
        int: The exact nonnegative count, including integral real scalars.

    Raises:
        ValueError: If the count is boolean, nonnumeric, nonfinite, negative,
            or fractional.
    """
    message = (
        "QURI Parts sampling requires nonnegative whole-number counts; use "
        "an ordinary sampler such as create_qulacs_vector_sampler() instead "
        "of fractional ideal-sampler weights."
    )
    if isinstance(count, bool):
        raise ValueError(message)
    if isinstance(count, Integral):
        normalized = int(count)
    elif isinstance(count, Real):
        try:
            normalized = int(math.floor(count))
            ceiling = int(math.ceil(count))
        except (OverflowError, ValueError) as error:
            raise ValueError(message) from error
        # Compare integers so fractional weights are never rounded or tolerated.
        if normalized != ceiling:
            raise ValueError(message)
    else:
        raise ValueError(message)
    if normalized < 0:
        raise ValueError(message)
    return normalized


class QuriPartsExecutor(
    QuantumExecutor["qp_c.LinearMappedUnboundParametricQuantumCircuit"]
):
    """Execute QURI Parts circuits with sampling and expectation estimation."""

    def __init__(
        self,
        sampler: Any = None,
        estimator: Any = None,
        seed: int | None = None,
        *,
        bound_estimator: Any = None,
    ):
        """Initialize the executor.

        Args:
            sampler: Optional QURI Parts sampler.
            estimator: Optional QURI Parts parametric estimator, used only
                with unbound parametric circuit states.
            seed: Optional seed for the default Qulacs sampler.
            bound_estimator: Optional non-parametric estimator for circuits
                that have already been bound by ``ExecutableProgram``.
        """
        self._sampler = sampler
        self._estimator = estimator
        self._estimator_was_supplied = estimator is not None
        self._bound_estimator = bound_estimator
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
        if self._bound_estimator is None:
            try:
                from quri_parts.qulacs.estimator import (  # type: ignore[import-not-found]
                    create_qulacs_vector_estimator,
                )

                self._bound_estimator = create_qulacs_vector_estimator()
            except ImportError as error:
                raise ImportError(
                    "quri-parts-qulacs is required for QuriPartsExecutor. "
                    "Install with: pip install quri-parts-qulacs"
                ) from error
        return self._bound_estimator

    def execute(self, circuit: Any, shots: int) -> dict[str, int]:
        """Sample a circuit and return bitstring counts.

        Nonnegative whole-number SDK scalars become Python integers so
        completed jobs contain portable counts. Fractional ideal-sampler
        weights, including floating-point roundoff, are rejected rather
        than rounded to fabricated observations.

        Args:
            circuit (Any): Bound or unbound QURI Parts circuit.
            shots (int): Number of measurement shots.

        Returns:
            dict[str, int]: Counts keyed by zero-padded bitstrings. A
                zero-qubit circuit returns ``{"": shots}`` without invoking
                the sampler. Outcomes with zero counts are omitted.

        Raises:
            ImportError: If the default sampler's SDK is unavailable.
            ValueError: If a sampler count is boolean, nonnumeric, nonfinite,
                negative, or fractional. Use an ordinary sampler that draws
                observations, such as ``create_qulacs_vector_sampler``.
            Exception: Propagates SDK circuit-conversion or sampling errors.
        """
        if circuit.qubit_count == 0:
            return {"": shots}

        counter = self.sampler(circuit, shots)
        counts = {}
        for value, count in counter.items():
            normalized = _normalize_sample_count(count)
            # Zero counts are not observations and make one-shot outcomes ambiguous.
            if normalized > 0:
                counts[format(value, f"0{circuit.qubit_count}b")] = normalized
        return counts

    def bind_parameters(  # type: ignore[override]
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        bindings: dict[str, Any],
        parameter_metadata: ParameterMetadata,
    ) -> "ImmutableBoundParametricQuantumCircuit":
        """Bind named parameters in engine order.

        Args:
            circuit: Unbound parametric circuit.
            bindings: Parameter values by Qamomile name.
            parameter_metadata: Ordered engine parameter metadata.

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
        param_values = [] if params is None else params
        return self.estimate_expectation(circuit, hamiltonian, param_values)  # type: ignore[arg-type]

    def estimate_expectation(
        self,
        circuit: "qp_c.LinearMappedUnboundParametricQuantumCircuit",
        hamiltonian: "qp_o.Operator",
        param_values: Sequence[float],
    ) -> float:
        """Estimate a native QURI Parts operator expectation value.

        Args:
            circuit (qp_c.LinearMappedUnboundParametricQuantumCircuit):
                Parametric or concrete QURI Parts circuit.
            hamiltonian (qp_o.Operator): Native QURI Parts operator.
            param_values (Sequence[float]): Values for an unbound circuit.

        Returns:
            float: Real expectation value as a native Python scalar.

        Raises:
            ImportError: If a required QURI Parts estimator SDK is unavailable.
            QamomileQuriPartsTranspileError: If a bound circuit requires a
                non-parametric estimator but only a parametric one was supplied.
            Exception: Propagates SDK state-preparation or estimation errors.
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
            if self._estimator_was_supplied and self._bound_estimator is None:
                raise QamomileQuriPartsTranspileError(
                    "The configured 'estimator' follows QURI Parts' "
                    "three-argument parametric-estimator protocol, but this "
                    "circuit has already been bound and requires a two-argument "
                    "non-parametric estimator. Pass bound_estimator=... as well, "
                    "or omit estimator to use the default Qulacs estimators."
                )
            estimate = self.non_parametric_estimator(hamiltonian, state)
        return float(estimate.value.real)


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
        """Create the common circuit-engine emission pass.

        Args:
            bindings: Compile-time argument values.
            parameters: Arguments preserved as runtime parameters.

        Returns:
            Emit pass backed by the QURI Parts materializer.
        """
        return CircuitEngineEmitPass(
            QuriPartsMaterializer(),
            bindings,
            parameters,
        )

    def executor(  # type: ignore[override]
        self,
        sampler: Any = None,
        estimator: Any = None,
        seed: int | None = None,
        *,
        bound_estimator: Any = None,
    ) -> QuriPartsExecutor:
        """Create a QURI Parts executor.

        Args:
            sampler: Optional custom sampler.
            estimator: Optional custom parametric estimator for unbound states.
            seed: Optional seed for the default sampler.
            bound_estimator: Optional custom non-parametric estimator for
                circuits already bound by ``ExecutableProgram``.

        Returns:
            Configured QURI Parts executor.
        """
        return QuriPartsExecutor(
            sampler,
            estimator,
            seed=seed,
            bound_estimator=bound_estimator,
        )
