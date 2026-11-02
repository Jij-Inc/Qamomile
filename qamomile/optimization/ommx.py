"""Immutable OMMX problem normalization and original-objective evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import ommx.v1

from .model import BinaryDomain, BinaryModel, DecodedSamples, VarType

_MAX_OMMX_EVALUATION_BYTES = 64 * 1024 * 1024
_MAX_OMMX_EVALUATION_ROWS = 1_000_000


@dataclass(frozen=True, slots=True, init=False)
class OMMXEvaluation:
    """Immutable unique OMMX solutions paired with sample multiplicities."""

    _sample_set_bytes: bytes
    _occurrences: tuple[tuple[int, int], ...]
    _sample_set: ommx.v1.SampleSet = field(repr=False, compare=False)

    @classmethod
    def _from_sample_set(
        cls,
        sample_set: ommx.v1.SampleSet,
        occurrences: tuple[int, ...],
    ) -> OMMXEvaluation:
        """Freeze one externally evaluated unique-row result."""

        return _restore_ommx_evaluation(
            sample_set.to_bytes(),
            tuple(enumerate(occurrences)),
        )

    @property
    def sample_set(self) -> ommx.v1.SampleSet:
        """Return a fresh OMMX SampleSet containing one row per unique solution."""

        return ommx.v1.SampleSet.from_bytes(self._sample_set_bytes)

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Return canonical unique-row identifiers in decoded distribution order."""

        return tuple(sample_id for sample_id, _ in self._occurrences)

    @property
    def occurrences(self) -> Mapping[int, int]:
        """Return the positive occurrence count of every unique solution."""

        return MappingProxyType(dict(self._occurrences))

    @property
    def total_occurrences(self) -> int:
        """Return the original distribution's total observation count."""

        return sum(count for _, count in self._occurrences)

    @property
    def feasible(self) -> Mapping[int, bool]:
        """Return original-constraint feasibility by unique solution ID."""

        return MappingProxyType(dict(self._sample_set.feasible))

    @property
    def objectives(self) -> Mapping[int, float]:
        """Return original objective values by unique solution ID."""

        return MappingProxyType(dict(self._sample_set.objectives))

    @property
    def best_feasible(self) -> ommx.v1.Solution:
        """Return the best feasible unique OMMX solution."""

        return self._sample_set.best_feasible

    def get(self, sample_id: int) -> ommx.v1.Solution:
        """Return one unique OMMX solution by canonical sample ID."""

        return self._sample_set.get(sample_id)

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Persist immutable external bytes and multiplicities, never a live handle."""

        return _restore_ommx_evaluation, (
            self._sample_set_bytes,
            self._occurrences,
        )


def _restore_ommx_evaluation(
    sample_set_bytes: bytes,
    occurrences: tuple[tuple[int, int], ...],
) -> OMMXEvaluation:
    """Validate and restore one bounded multiplicity-preserving OMMX result."""

    if not isinstance(sample_set_bytes, bytes):
        raise TypeError("Q7672: OMMX evaluation payload must be immutable bytes")
    if len(sample_set_bytes) > _MAX_OMMX_EVALUATION_BYTES:
        raise ValueError("Q7673: OMMX evaluation payload exceeds 64 MiB")
    if not isinstance(occurrences, tuple):
        raise TypeError("Q7672: OMMX evaluation occurrences must be a tuple")
    if len(occurrences) > _MAX_OMMX_EVALUATION_ROWS:
        raise ValueError("Q7673: OMMX evaluation exceeds one million unique rows")
    if any(
        not isinstance(row, tuple)
        or len(row) != 2
        or type(row[0]) is not int
        or row[0] != expected_id
        or type(row[1]) is not int
        or row[1] <= 0
        for expected_id, row in enumerate(occurrences)
    ):
        raise ValueError("Q7672: OMMX evaluation occurrences are not canonical")
    try:
        sample_set = ommx.v1.SampleSet.from_bytes(sample_set_bytes)
        observed_ids = sample_set.raw.sample_ids()
    except (MemoryError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Q7672: malformed OMMX evaluation payload") from error
    expected_ids = set(range(len(occurrences)))
    if observed_ids != expected_ids:
        raise ValueError("Q7672: OMMX evaluation rows disagree with occurrences")
    result = object.__new__(OMMXEvaluation)
    object.__setattr__(result, "_sample_set_bytes", sample_set_bytes)
    object.__setattr__(result, "_occurrences", occurrences)
    object.__setattr__(result, "_sample_set", sample_set)
    return result


@dataclass(frozen=True, slots=True, init=False)
class OMMXProblem:
    """An OMMX instance paired with its penalty-domain binary model.

    Both OMMX instances are retained as immutable serialized values. The
    transformed copy owns binary substitutions, slack variables, and penalty
    metadata; the original copy remains authoritative for objective and
    constraint evaluation. No mutable caller-owned instance is retained.
    """

    _original_bytes: bytes
    _transformed_bytes: bytes
    model: BinaryModel[BinaryDomain]

    @classmethod
    def from_instance(
        cls,
        instance: ommx.v1.Instance,
        *,
        uniform_penalty_weight: float | None = None,
        penalty_weights: Mapping[int, float] | None = None,
    ) -> OMMXProblem:
        """Normalize one OMMX instance into an arbitrary-order binary model.

        Args:
            instance: User-owned OMMX optimization instance.
            uniform_penalty_weight: Optional common constraint penalty.
            penalty_weights: Optional per-constraint penalty overrides.

        Returns:
            An immutable original/transformed problem pair and its model.

        Raises:
            TypeError: If ``instance`` is not an OMMX instance.
        """

        if not isinstance(instance, ommx.v1.Instance):
            raise TypeError("instance must be an ommx.v1.Instance")
        original_bytes = instance.to_bytes()
        transformed = ommx.v1.Instance.from_bytes(original_bytes)
        coefficients, constant = transformed.to_hubo(
            uniform_penalty_weight=uniform_penalty_weight,
            penalty_weights=dict(penalty_weights or {}),
        )
        problem = object.__new__(cls)
        object.__setattr__(problem, "_original_bytes", original_bytes)
        object.__setattr__(problem, "_transformed_bytes", transformed.to_bytes())
        object.__setattr__(
            problem,
            "model",
            BinaryModel.from_hubo(
                coefficients,
                constant,
                variables=tuple(
                    variable.id
                    for variable in transformed.decision_variables
                    if variable.kind == variable.BINARY
                ),
            ),
        )
        return problem

    @property
    def original_instance(self) -> ommx.v1.Instance:
        """Return a fresh copy of the authoritative original instance."""

        return ommx.v1.Instance.from_bytes(self._original_bytes)

    @property
    def transformed_instance(self) -> ommx.v1.Instance:
        """Return a fresh copy owning substitutions and penalty metadata."""

        return ommx.v1.Instance.from_bytes(self._transformed_bytes)

    def evaluate(self, samples: DecodedSamples[BinaryDomain]) -> OMMXEvaluation:
        """Evaluate decoded penalty-domain samples on the original problem.

        The transformed instance first reconstructs non-binary decision
        variables from its binary substitutions. Only original decision
        variables are then evaluated by the untouched original instance, so
        objectives and feasibility never include absorbed penalty terms.

        Args:
            samples: BINARY samples decoded by :attr:`model`.

        Returns:
            Unique OMMX solutions with original objectives, constraint status,
            and exact occurrence counts.

        Raises:
            TypeError: If ``samples`` is not a decoded sample distribution.
            ValueError: If its domain or variable layout differs from the model.
        """

        if not isinstance(samples, DecodedSamples):
            raise TypeError("samples must be qamomile.optimization.DecodedSamples")
        if samples.vartype is not VarType.BINARY:
            raise ValueError("OMMX evaluation requires BINARY decoded samples")
        expected_variables = self.model.variables
        if any(
            tuple(index for index, _ in sample.assignment) != expected_variables
            for sample in samples.samples
        ):
            raise ValueError("decoded sample variables do not match the OMMX model")

        binary_samples = ommx.v1.Samples({})
        for sample_id, sample in enumerate(samples.samples):
            binary_samples.append(
                [sample_id],
                ommx.v1.State(
                    {index: float(value) for index, value in sample.assignment}
                ),
            )

        transformed = self.transformed_instance
        reconstructed = transformed.evaluate_samples(binary_samples)
        original = self.original_instance
        original_ids = tuple(variable.id for variable in original.decision_variables)
        original_samples = ommx.v1.Samples({})
        for sample_id in range(len(samples.samples)):
            values = reconstructed.get(sample_id).decision_variables_df["value"]
            original_samples.append(
                [sample_id],
                ommx.v1.State(
                    {
                        variable_id: float(values.loc[variable_id])
                        for variable_id in original_ids
                    }
                ),
            )
        return OMMXEvaluation._from_sample_set(
            original.evaluate_samples(original_samples),
            tuple(sample.occurrences for sample in samples.samples),
        )


__all__ = ["OMMXEvaluation", "OMMXProblem"]
