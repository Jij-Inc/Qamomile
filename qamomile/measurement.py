"""Host-side measurement scheduling around classical-to-classical executions.

Estimator composes a quantum-returning state-preparation definition with ordinary
measurements. Rust validates signatures, derives measurement programs and schedules,
and computes statistics. Python coordinates execution and retains a bounded cache.
No live quantum state or cross-shot operation enters the qkernel language.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from . import statistics
from ._measurement_plan import (
    _group_measurement_program,
    _MeasurementGroupPlan,
    _native_commuting_expectation_plan,
    _native_qwc_expectation_plan,
)
from ._native import compiler as native
from .compilation import (
    CompiledProgram,
    ExecutionEndpoint,
    ParameterABI,
    SampleResult,
    compile,
    run,
)
from .errors import CapabilityError, ExecutionError, TypeCheckError
from .expectation import (
    ExpectationResult,
    PauliEstimate,
    PauliMeasurementGroup,
)
from .ir import Program
from .observable import Hamiltonian
from .source import ProgramFamily, QKernel, _close_program


@runtime_checkable
class _EstimationEndpoint(Protocol):
    """Optional adapter hook that selects a route before any device execution."""

    def _prepare_estimation(
        self,
        source: Program,
        observable: Hamiltonian,
        plan: tuple[_MeasurementGroupPlan, ...],
        parameters: Mapping[str, object],
    ) -> Callable[[], ExpectationResult] | None:
        """Prepare an exact-contract route, or decline without executing.

        Args:
            source (Program): Verified classical-input state preparation.
            observable (Hamiltonian): Validated Hermitian observable.
            plan (tuple[_MeasurementGroupPlan, ...]): Rust-owned shots and grouping.
            parameters (Mapping[str, object]): ABI-normalized runtime bindings.

        Returns:
            Callable[[], ExpectationResult] | None: Deferred execution, or decline.
                Errors from deferred execution must propagate without retry.
        """
        ...


@dataclass(frozen=True, slots=True, init=False)
class Estimator:
    """Prepare and reuse measurements of a Hermitian Pauli Hamiltonian.

    Construction validates source and observable without generating circuits.
    Execution first selects an optional backend estimator with matching semantics;
    otherwise ordinary measured programs are generated and compiled lazily.
    Each estimate prepares fresh states. The common route retains the latest
    target's compiled programs across changes to parameters, shots, or seeds.
    Configuration is immutable. This host coordinator is not a persistent
    artifact; its ordinary programs and execution results retain existing ABIs.

    Args:
        source (Program | ProgramFamily | QKernel): Classical-input preparation
            returning exactly one fixed-width Q[Bits[N]] after specialization.
        observable (Hamiltonian): Hermitian observable indexed by returned bits.
        grouping (str): "qwc" shares local bases, "independent" measures each
            term, and "commuting" uses joint Clifford diagonalization.
        allocation (str): "uniform" distributes the total budget equally among
            groups, giving the first groups any integer remainder.
        **bindings (object): Const arguments of the preparation.

    Raises:
        TypeError: If the observable or policy has an invalid Python shape.
        TypeCheckError: If the source signature or observable width is invalid.
        ExecutionError: If native grouping exceeds its work or size budget.
        CapabilityError: If the required native compiler is unavailable.
    """

    source: Program
    observable: Hamiltonian
    grouping: Literal["independent", "qwc", "commuting"]
    allocation: Literal["uniform"]
    _cache: list[tuple[tuple[str, str], tuple[CompiledProgram, ...]]] = field(
        repr=False, compare=False
    )

    def __init__(
        self,
        source: Program | ProgramFamily | QKernel,
        observable: Hamiltonian,
        *,
        grouping: Literal["independent", "qwc", "commuting"] = "qwc",
        allocation: Literal["uniform"] = "uniform",
        **bindings: object,
    ) -> None:
        """Validate configuration without generating measured programs.

        Args:
            source (Program | ProgramFamily | QKernel): State-preparation definition.
            observable (Hamiltonian): Hermitian Pauli sum on the returned bits.
            grouping (str): Independent, QWC, or commuting measurement grouping.
            allocation (str): Uniform total-shot allocation.
            **bindings (object): Const arguments used to specialize the source.

        Raises:
            TypeError: If a policy or observable has an invalid Python shape.
            TypeCheckError: If Rust rejects the source or observable contract.
            CapabilityError: If native validation or transformation is unavailable.
            ExecutionError: If grouping or its native projection is invalid.
        """
        if not isinstance(observable, Hamiltonian):
            raise TypeError("observable must be a Hamiltonian")
        if grouping not in {"independent", "qwc", "commuting"}:
            raise TypeError("grouping must be 'independent', 'qwc', or 'commuting'")
        if allocation != "uniform":
            raise TypeError("allocation must be 'uniform'")
        validator = getattr(native, "validate_estimator_source", None)
        if not callable(validator):
            raise CapabilityError("Q7300: native Estimator validation is unavailable")
        program = _close_program(source, bindings)
        try:
            validator(program.to_bytes(), observable._native_literal)
        except ValueError as error:
            raise TypeCheckError(str(error)) from error
        object.__setattr__(self, "source", program)
        object.__setattr__(self, "observable", observable)
        object.__setattr__(self, "grouping", grouping)
        object.__setattr__(self, "allocation", allocation)
        object.__setattr__(self, "_cache", [])

    def _plan(self, shots: int, seed: int) -> tuple[_MeasurementGroupPlan, ...]:
        """Project a bounded Rust-owned grouping and uniform total allocation.

        Args:
            shots (int): Positive total shot budget.
            seed (int): Unsigned portable base seed.

        Returns:
            tuple: Checked measurement groups in canonical order.
        """
        if self.grouping == "commuting":
            return _native_commuting_expectation_plan(
                self.observable, shots=shots, seed=seed
            )
        return _native_qwc_expectation_plan(
            self.observable,
            shots=shots,
            seed=seed,
            independent=self.grouping == "independent",
        )

    def estimate(
        self,
        *,
        on: ExecutionEndpoint,
        parameters: Mapping[str, object] | None = None,
        shots: int = 1_000,
        seed: int = 0,
    ) -> ExpectationResult:
        """Execute measurements and combine their covariance-aware estimates.

        Args:
            on (ExecutionEndpoint): Local or explicitly selected remote endpoint.
            parameters (Mapping[str, object] | None): Runtime Param arguments.
            shots (int): Positive total budget, at least the number of groups.
                Identity-only observables use zero actual device shots.
            seed (int): Unsigned 64-bit seed; zero delegates randomness to the
                endpoint. Reusing a nonzero seed reuses random streams.

        Returns:
            ExpectationResult: Mean, standard error, actual shots and group data.
                One-shot groups have unknown uncertainty, represented by None.

        Raises:
            ValueError: If shots is not a positive integer.
            ExecutionError: If the budget is insufficient or execution fails.
            CapabilityError: If native planning or statistics are unavailable.
        """
        if type(shots) is not int or shots <= 0:
            raise ValueError("shots must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 1 << 64:
            raise ExecutionError("Q8138: seed must be an unsigned 64-bit integer")
        if not isinstance(on, ExecutionEndpoint):
            raise TypeError("on= must implement ExecutionEndpoint")
        normalized = dict(ParameterABI.from_program(self.source).bind(parameters or {}))
        plan = self._plan(shots, seed)
        if plan and isinstance(on, _EstimationEndpoint):
            execute_native = on._prepare_estimation(
                self.source, self.observable, plan, normalized
            )
            if execute_native is not None:
                result = execute_native()
                if not isinstance(result, ExpectationResult) or result.shots != shots:
                    raise ExecutionError(
                        "backend estimator returned an invalid shot budget"
                    )
                return result
        artifacts: tuple[CompiledProgram, ...] = ()
        if plan:
            target = on.target
            key = (target.name, target.fingerprint)
            if self._cache and self._cache[0][0] == key:
                artifacts = self._cache[0][1]
            else:
                artifacts = tuple(
                    compile(
                        _group_measurement_program(self.source, self.observable, group),
                        to=target,
                    )
                    for group in plan
                )
                self._cache[:] = [(key, artifacts)]
        terms: dict[int, PauliEstimate] = {}
        groups = []
        estimates = []
        backend = on.name
        for group, artifact in zip(plan, artifacts, strict=True):
            samples = run(
                artifact,
                on=on,
                parameters=normalized,
                shots=group.shots,
                seed=group.seed,
            ).result
            if not isinstance(samples, SampleResult):
                raise ExecutionError(
                    "expectation endpoint did not return sample counts"
                )
            if samples.shots != group.shots:
                raise ExecutionError(
                    "expectation endpoint returned an incorrect shot count"
                )
            readouts = []
            for position, index in enumerate(group.term_indices):
                word, coefficient = self.observable.terms[index]
                if group.strategy == "commuting":
                    readout = group.readouts[position]
                    factors = readout.word.factors
                    sign = -1.0 if readout.negative else 1.0
                else:
                    factors, sign = word.factors, 1.0
                parity = statistics.Parity(tuple(factor.index for factor in factors))
                term = statistics.linear_combination(samples, [(sign, parity)])
                terms[index] = PauliEstimate(
                    word, coefficient.real, term.value, term.sample_variance, term.shots
                )
                readouts.append((sign * coefficient.real, parity))
            estimate = statistics.linear_combination(samples, readouts)
            estimates.append(estimate)
            groups.append(
                PauliMeasurementGroup(
                    group.basis,
                    group.term_indices,
                    estimate.sample_variance,
                    estimate.shots,
                    group.seed,
                    group.readouts,
                    group.diagonalizing_gates,
                    group.strategy,
                )
            )
            backend = samples.backend
        total = statistics.combine_independent(
            estimates, offset=self.observable.constant.real
        )
        return ExpectationResult(
            total.value,
            total.standard_error,
            backend,
            total.shots,
            tuple(terms[i] for i in range(len(terms))),
            tuple(groups),
        )


__all__ = ["Estimator"]
