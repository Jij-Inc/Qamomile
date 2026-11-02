"""Delegate compatible finite-shot requests to Qiskit's backend estimator.

The adapter accepts only singleton measurement groups, for which Qiskit's
uncertainty can be normalized without missing within-shot covariance. Rust
continues to own source contracts, output wire order, and measurement budgets.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from qamomile import statistics
from qamomile._measurement_plan import _MeasurementGroupPlan
from qamomile.circuit import lower_to_circuit_at
from qamomile.effects import Effect
from qamomile.errors import ExecutionError
from qamomile.expectation import (
    ExpectationResult,
    PauliEstimate,
    PauliMeasurementGroup,
)
from qamomile.ir import Program
from qamomile.observable import Hamiltonian

if TYPE_CHECKING:
    from ._integration import Aer


def _shot_precision(shots: int) -> float | None:
    """Translate an integer budget only when Qiskit's ceil rule preserves it.

    Args:
        shots (int): Positive shot allocation for one measured Pauli word.

    Returns:
        float | None: Precision selecting exactly this allocation, or decline.
    """
    if shots > 1 << 52:
        return None
    precision = 1.0 / math.sqrt(shots)
    # Move into the interior of ceil(1 / precision**2) == shots. This also
    # avoids spending one extra shot due to binary64 rounding at its boundary.
    for _ in range(4):
        count = math.ceil(1.0 / precision**2)
        if count == shots:
            return precision
        precision = math.nextafter(precision, math.inf if count > shots else 0.0)
    return None


def prepare_estimation(
    endpoint: Aer,
    source: Program,
    observable: Hamiltonian,
    plan: tuple[_MeasurementGroupPlan, ...],
    parameters: Mapping[str, object],
) -> Callable[[], ExpectationResult] | None:
    """Prepare supported SDK work without executing or consuming any shots.

    Args:
        endpoint (Aer): Local simulator configuration.
        source (Program): Validated state-preparation component.
        observable (Hamiltonian): Validated Hermitian Pauli sum.
        plan (tuple[_MeasurementGroupPlan, ...]): Native grouping and allocation.
        parameters (Mapping[str, object]): Complete normalized runtime bindings.

    Returns:
        Callable[[], ExpectationResult] | None: Deferred native execution, or
            None when the common measured-program route must handle the request.

    Raises:
        ExecutionError: If SDK preparation or subsequent execution fails.
    """
    if any(
        len(group.term_indices) != 1 or group.strategy == "commuting" for group in plan
    ):
        return None
    entry = next(
        function
        for function in source.functions
        if function.symbol == source.entrypoint
    )
    if entry.effects & {
        Effect.MEASUREMENT,
        Effect.DISCARD,
        Effect.RESET,
        Effect.SHOT_CONTROL,
    }:
        return None
    precisions = tuple(_shot_precision(group.shots) for group in plan)
    if any(precision is None for precision in precisions):
        return None

    from qiskit import transpile
    from qiskit.primitives import BackendEstimatorV2
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_aer import AerSimulator

    from ._integration import _bind_circuit_parameters, _emit_for_target

    target = endpoint.target
    logical = lower_to_circuit_at(
        source,
        parameters,
        realization_capabilities=target.capabilities.realization_capabilities,
        realizations=target.capabilities.realizations,
    )
    if logical.output_wires is None:
        return None
    circuit = _bind_circuit_parameters(
        _emit_for_target(logical, target)[0],
        logical.parameter_abi,
        {slot.name: parameters[slot.name] for slot in logical.parameter_abi.slots},
    )
    backend = AerSimulator(
        method=endpoint.method,
        max_parallel_threads=1,
        max_parallel_experiments=1,
        max_parallel_shots=1,
    )
    circuit = transpile(
        circuit,
        backend=backend,
        optimization_level=endpoint.optimization_level,
        seed_transpiler=endpoint.seed_transpiler,
    )
    jobs = []
    for group, precision in zip(plan, precisions, strict=True):
        word, _ = observable.terms[group.term_indices[0]]
        label = ["I"] * logical.width
        for factor in word.factors:
            label[logical.width - 1 - logical.output_wires[factor.index]] = (
                factor.pauli.value
            )
        operator = SparsePauliOp("".join(label))
        if circuit.layout is not None:
            operator = operator.apply_layout(circuit.layout)
        primitive = BackendEstimatorV2(
            backend=backend,
            options={"abelian_grouping": False, "seed_simulator": group.seed or None},
        )
        jobs.append((primitive, operator, precision))

    def execute() -> ExpectationResult:
        """Execute the chosen route once; never retry through sampling on failure.

        Returns:
            ExpectationResult: Unbiased uncertainty and exact total device shots.

        Raises:
            ExecutionError: If SDK execution or result normalization fails.
        """
        terms = {}
        groups = []
        estimates = []
        coefficients = []
        try:
            for group, (primitive, operator, precision) in zip(plan, jobs, strict=True):
                result = primitive.run(
                    [(circuit, operator)], precision=precision
                ).result()[0]
                if result.metadata.get("shots") != group.shots:
                    raise ExecutionError(
                        "backend estimator returned an incorrect shot count"
                    )
                mean = float(result.data["evs"])
                sdk_error = float(result.data["stds"])
                if (
                    not math.isfinite(mean)
                    or abs(mean) > 1
                    or not math.isfinite(sdk_error)
                    or sdk_error < 0
                ):
                    raise ExecutionError(
                        "backend estimator returned invalid Pauli statistics"
                    )
                # Qiskit reports sqrt((1 - mean**2) / N) for one Pauli.
                # Normalize its population-moment estimate to the N-1 convention.
                error = (
                    None
                    if group.shots == 1
                    else sdk_error * math.sqrt(group.shots / (group.shots - 1))
                )
                variance = None if error is None else error * error * group.shots
                index = group.term_indices[0]
                word, coefficient = observable.terms[index]
                estimates.append(
                    statistics.Estimate._from_summary(mean, error, group.shots)
                )
                coefficients.append(coefficient.real)
                terms[index] = PauliEstimate(
                    word, coefficient.real, mean, variance, group.shots
                )
                groups.append(
                    PauliMeasurementGroup(
                        group.basis,
                        group.term_indices,
                        None if variance is None else coefficient.real**2 * variance,
                        group.shots,
                        group.seed,
                        strategy=group.strategy,
                    )
                )
            total = statistics.combine_independent(
                estimates, coefficients=coefficients, offset=observable.constant.real
            )
        except ExecutionError:
            raise
        except Exception as error:
            raise ExecutionError(f"Q8222: Qiskit estimator failed: {error}") from error
        return ExpectationResult(
            total.value,
            total.standard_error,
            endpoint.name,
            total.shots,
            tuple(terms[i] for i in range(len(terms))),
            tuple(groups),
            method="qiskit.backend_estimator_v2",
        )

    return execute
