"""Pauli Correlation Encoding over the shared optimization model."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar

from qamomile._native import compiler as native
from qamomile.observable import Hamiltonian, Pauli, PauliFactor, PauliWord

from .model import BinaryModel, DecodedSample, DecodedSamples, VarType

_Domain = TypeVar("_Domain", bound=VarType)


@dataclass(frozen=True, slots=True)
class PauliCorrelationEncoding(Generic[_Domain]):
    """Deterministically encode variables as distinct k-body Pauli words.

    PCE does not prescribe an ansatz and has no single encoded cost
    Hamiltonian. The returned observables are intended for ordinary
    :func:`qamomile.expectation` calls; :meth:`relaxed_cost` combines those
    expectation values classically and :meth:`decode` sign-rounds them back to
    the original model domain.
    """

    model: BinaryModel[_Domain]
    correlator_order: int
    num_qubits: int | None = None
    observables: tuple[Hamiltonian, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _native_plan: object = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the PCE domain and construct the immutable enumeration."""

        if not isinstance(self.model, BinaryModel):
            raise TypeError("model must be a qamomile.optimization.BinaryModel")
        _validate_capacity_inputs(self.model.num_variables, self.correlator_order)
        if self.num_qubits is not None and type(self.num_qubits) is not int:
            raise TypeError("num_qubits must be an integer")
        if self.num_qubits is not None and self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        derive = getattr(self.model._native_model, "pauli_correlation_plan", None)
        if not callable(derive):
            raise TypeError("Q7570: native Pauli correlation planning is unavailable")
        try:
            plan = derive(self.correlator_order, self.num_qubits)
        except OverflowError as error:
            raise ValueError(
                "Q7571: PCE request exceeds its capacity budget"
            ) from error
        width, words = _decode_native_plan(
            plan,
            self.model.num_variables,
            self.correlator_order,
        )
        if self.num_qubits is not None and width != self.num_qubits:
            raise TypeError("Q7575: malformed native Pauli correlation plan")
        object.__setattr__(self, "num_qubits", width)
        object.__setattr__(
            self,
            "observables",
            tuple(
                Hamiltonian(
                    (
                        (
                            PauliWord(
                                tuple(
                                    PauliFactor(index, pauli) for index, pauli in word
                                )
                            ),
                            1.0 + 0.0j,
                        ),
                    ),
                    num_qubits=width,
                )
                for word in words
            ),
        )
        object.__setattr__(self, "_native_plan", plan)

    @staticmethod
    def min_num_qubits(num_variables: int, correlator_order: int) -> int:
        """Return the smallest ``n >= k`` with enough k-body Pauli words."""

        _validate_capacity_inputs(num_variables, correlator_order)
        derive = getattr(native, "minimum_pce_qubits", None)
        if not callable(derive):
            raise TypeError("Q7570: native Pauli correlation planning is unavailable")
        try:
            width = derive(num_variables, correlator_order)
        except OverflowError as error:
            raise ValueError(
                "Q7571: PCE request exceeds its capacity budget"
            ) from error
        if type(width) is not int or width < correlator_order:
            raise TypeError("Q7575: malformed native Pauli correlation plan")
        return width

    @property
    def correlations(self) -> Mapping[int, Hamiltonian]:
        """Return original variable IDs mapped to their observables."""

        return MappingProxyType(
            dict(zip(self.model.variables, self.observables, strict=True))
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Reconstruct the private native plan from stable public inputs."""

        return type(self), (self.model, self.correlator_order, self.num_qubits)

    def relaxed_cost(self, expectations: Sequence[float]) -> float:
        """Evaluate the spin polynomial on continuous Pauli expectations."""

        values = self._expectations(expectations)
        evaluate = getattr(self._native_plan, "relaxed_cost", None)
        if not callable(evaluate):
            raise TypeError("Q7570: native Pauli correlation planning is unavailable")
        result = evaluate(values)
        if not isinstance(result, float) or not math.isfinite(result):
            raise TypeError("Q7575: malformed native Pauli correlation plan")
        return result

    def decode(self, expectations: Sequence[float]) -> DecodedSamples[_Domain]:
        """Sign-round expectations into one sample in the original domain."""

        values = self._expectations(expectations)
        decode = getattr(self._native_plan, "decode", None)
        if not callable(decode):
            raise TypeError("Q7570: native Pauli correlation planning is unavailable")
        projection = decode(values)
        decoded_values, energy = _decode_native_result(
            projection,
            self.model.num_variables,
        )
        assignment = tuple(zip(self.model.variables, decoded_values, strict=True))
        sample = DecodedSample._from_native(assignment, energy, 1)
        try:
            return DecodedSamples(self.model.vartype, (sample,))
        except (TypeError, ValueError) as error:
            raise TypeError("Q7575: malformed native Pauli correlation plan") from error

    def _expectations(self, values: object) -> tuple[float, ...]:
        """Validate one complete finite Pauli-expectation coordinate."""

        if isinstance(values, (str, bytes, bytearray)) or not isinstance(
            values, Sequence
        ):
            raise TypeError("expectations must be a real numeric sequence")
        if len(values) != self.model.num_variables:
            raise ValueError(
                f"expected {self.model.num_variables} expectation values, "
                f"got {len(values)}"
            )
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("expectations must contain only real numbers")
            expectation = float(value)
            if not math.isfinite(expectation):
                raise ValueError("expectations must be finite")
            if not -1.0 <= expectation <= 1.0:
                raise ValueError("expectations must lie in [-1, 1]")
            normalized.append(expectation)
        return tuple(normalized)


def _validate_capacity_inputs(num_variables: int, correlator_order: int) -> None:
    """Validate discrete PCE capacity inputs without Boolean coercion."""

    if type(num_variables) is not int:
        raise TypeError("num_variables must be an integer")
    if num_variables < 0:
        raise ValueError("num_variables must be non-negative")
    if type(correlator_order) is not int:
        raise TypeError("correlator_order must be an integer")
    if correlator_order <= 0:
        raise ValueError("correlator_order must be positive")


def _decode_native_plan(
    plan: object,
    count: int,
    order: int,
) -> tuple[int, tuple[tuple[tuple[int, Pauli], ...], ...]]:
    """Validate one transient native PCE word projection."""

    width = getattr(plan, "width", None)
    raw_words = getattr(plan, "words", None)
    if (
        type(width) is not int
        or width < order
        or not isinstance(raw_words, list)
        or len(raw_words) != count
    ):
        raise TypeError("Q7575: malformed native Pauli correlation plan")
    words: list[tuple[tuple[int, Pauli], ...]] = []
    for raw_word in raw_words:
        if not isinstance(raw_word, list):
            raise TypeError("Q7575: malformed native Pauli correlation plan")
        try:
            word = tuple((index, Pauli(pauli)) for index, pauli in raw_word)
        except (TypeError, ValueError) as error:
            raise TypeError("Q7575: malformed native Pauli correlation plan") from error
        indices = tuple(index for index, _ in word)
        if (
            len(word) != order
            or any(
                type(index) is not int or not 0 <= index < width for index in indices
            )
            or tuple(sorted(set(indices))) != indices
        ):
            raise TypeError("Q7575: malformed native Pauli correlation plan")
        words.append(word)
    if len(set(words)) != len(words):
        raise TypeError("Q7575: malformed native Pauli correlation plan")
    return width, tuple(words)


def _decode_native_result(
    projection: object,
    width: int,
) -> tuple[tuple[int, ...], float]:
    """Validate one native PCE sign-decoding result."""

    if (
        not isinstance(projection, tuple)
        or len(projection) != 2
        or not isinstance(projection[0], list)
        or len(projection[0]) != width
        or any(type(value) is not int for value in projection[0])
        or not isinstance(projection[1], float)
    ):
        raise TypeError("Q7575: malformed native Pauli correlation plan")
    return tuple(projection[0]), projection[1]


__all__ = ["PauliCorrelationEncoding"]
