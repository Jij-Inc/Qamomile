"""Deterministic single-variable local search for polynomial models."""

from __future__ import annotations

import enum
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

from .model import BinaryModel, DecodedSample, DecodedSamples, VarType

_Domain = TypeVar("_Domain", bound=VarType)
_InitialState: TypeAlias = Mapping[int, int] | Sequence[int] | DecodedSample


class LocalSearchMethod(enum.StrEnum):
    """Rule used to select one strictly improving variable flip."""

    BEST = "best"
    FIRST = "first"


@dataclass(frozen=True, slots=True, init=False)
class LocalSearch(Generic[_Domain]):
    """Immutable single-variable descent over one polynomial model.

    Rust searches the model's native BINARY or SPIN algebra directly. Its
    sparse affected-term index supports arbitrary polynomial order without
    converting or expanding the objective into another domain.
    """

    model: BinaryModel[_Domain]
    method: LocalSearchMethod

    def __init__(
        self,
        model: BinaryModel[_Domain],
        *,
        method: LocalSearchMethod | str = LocalSearchMethod.BEST,
    ) -> None:
        """Build a deterministic search index for one immutable model.

        Args:
            model: Arbitrary-order BINARY or SPIN objective to minimize.
            method: First-improvement or best-improvement flip selection.

        Raises:
            TypeError: If ``model`` is not a binary model.
            ValueError: If ``method`` is not a supported selection rule.
        """

        if not isinstance(model, BinaryModel):
            raise TypeError("model must be qamomile.optimization.BinaryModel")
        try:
            selected_method = LocalSearchMethod(method)
        except (TypeError, ValueError) as error:
            raise ValueError("method must be 'best' or 'first'") from error

        object.__setattr__(self, "model", model)
        object.__setattr__(self, "method", selected_method)

    def run(
        self,
        initial_state: _InitialState,
        *,
        max_steps: int | None = None,
    ) -> DecodedSample:
        """Descend from one assignment until no selected flip improves it.

        Args:
            initial_state: Original-ID mapping, dense model-order sequence, or
                decoded sample in the model's domain.
            max_steps: Optional maximum number of accepted flips. ``None``
                searches to a one-flip local minimum.

        Returns:
            A canonical decoded sample with recomputed energy. A decoded input
            retains its occurrence count.

        Raises:
            TypeError: If ``max_steps`` is not an integer or ``initial_state``
                has an unsupported representation.
            ValueError: If the bound, variable layout, or values are invalid.
        """

        if max_steps is not None and type(max_steps) is not int:
            raise TypeError("max_steps must be a non-negative integer or None")
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer or None")
        search = getattr(self.model._native_model, "minimize_assignment", None)
        if not callable(search):
            raise TypeError("Q7550: native local search is unavailable")
        variables, values, occurrences = self._initial_values(initial_state)
        try:
            projection = search(
                values,
                self.method.value,
                max_steps,
                variables,
            )
        except OverflowError as error:
            raise ValueError("Q7556: local search exceeds its work budget") from error
        values, energy = self._decode_native_result(projection, max_steps=max_steps)
        assignment = tuple(zip(self.model.variables, values, strict=True))
        return DecodedSample._from_native(
            assignment,
            energy,
            occurrences,
        )

    def refine(
        self,
        samples: DecodedSamples[_Domain],
        *,
        max_steps: int | None = None,
    ) -> DecodedSamples[_Domain]:
        """Improve every sample and merge rows reaching the same assignment.

        Args:
            samples: Typed decoded distribution for this model.
            max_steps: Optional accepted-flip bound applied independently to
                each unique input row.

        Returns:
            A canonical distribution retaining total occurrences.

        Raises:
            TypeError: If ``samples`` is not a decoded distribution.
            ValueError: If its domain or variable layout differs from the
                search model, or if ``max_steps`` is invalid.
        """

        if not isinstance(samples, DecodedSamples):
            raise TypeError("samples must be qamomile.optimization.DecodedSamples")
        if samples.vartype is not self.model.vartype:
            raise ValueError("decoded sample vartype does not match the model")

        occurrences: defaultdict[tuple[tuple[int, int], ...], int] = defaultdict(int)
        energies: dict[tuple[tuple[int, int], ...], float] = {}
        for sample in samples.samples:
            refined = self.run(sample, max_steps=max_steps)
            occurrences[refined.assignment] += refined.occurrences
            energies[refined.assignment] = refined.energy
        rows = tuple(
            DecodedSample._from_native(assignment, energies[assignment], count)
            for assignment, count in sorted(occurrences.items())
        )
        return DecodedSamples(self.model.vartype, rows)

    def _initial_values(
        self,
        initial_state: _InitialState,
    ) -> tuple[list[str] | None, list[str], int]:
        """Encode one supported host input while retaining multiplicity."""

        occurrences = 1
        state: Mapping[int, int] | Sequence[int]
        if isinstance(initial_state, DecodedSample):
            state = dict(initial_state.assignment)
            occurrences = initial_state.occurrences
        else:
            state = initial_state
        variables, values = self.model._assignment_input(state)
        return variables, values, occurrences

    def _decode_native_result(
        self,
        projection: object,
        *,
        max_steps: int | None,
    ) -> tuple[list[int], float]:
        """Validate one transient native local-search result."""

        if (
            not isinstance(projection, tuple)
            or len(projection) != 3
            or not isinstance(projection[0], list)
            or len(projection[0]) != self.model.num_variables
            or not isinstance(projection[1], float)
            or not math.isfinite(projection[1])
            or type(projection[2]) is not int
            or projection[2] < 0
            or (max_steps is not None and projection[2] > max_steps)
        ):
            raise TypeError("Q7558: malformed native local-search result")
        allowed = {0, 1} if self.model.vartype is VarType.BINARY else {-1, 1}
        if any(
            type(value) is not int or value not in allowed for value in projection[0]
        ):
            raise TypeError("Q7558: malformed native local-search result")
        return projection[0], projection[1]


__all__ = ["LocalSearch", "LocalSearchMethod"]
