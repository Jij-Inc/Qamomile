"""Quantum random-access encodings over one immutable model."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar

from qamomile.observable import Hamiltonian, _hamiltonian_from_native_header

from .model import BinaryModel, DecodedSample, DecodedSamples, VarType

_Domain = TypeVar("_Domain", bound=VarType)


class QRACVariant(enum.StrEnum):
    """Supported random-access encoding geometry."""

    QRAC21 = "2,1"
    QRAC31 = "3,1"
    QRAC32 = "3,2"
    SPACE_EFFICIENT = "space-efficient"


@dataclass(frozen=True, slots=True)
class QRACEncoding(Generic[_Domain]):
    """One deterministic QRAC relaxation and its decoding coordinates."""

    model: BinaryModel[_Domain]
    variant: QRACVariant = QRACVariant.QRAC31
    groups: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    observables: tuple[Hamiltonian, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    hamiltonian: Hamiltonian = field(init=False, repr=False, compare=False)
    num_qubits: int = field(init=False)
    _native_plan: object = field(init=False, repr=False, compare=False)
    _decode_row_limit: int = field(init=False, repr=False, compare=False)
    _decode_cell_limit: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the quadratic domain and derive all encoding facts."""

        if not isinstance(self.model, BinaryModel):
            raise TypeError("model must be a qamomile.optimization.BinaryModel")
        if not isinstance(self.variant, QRACVariant):
            raise TypeError("variant must be a QRACVariant")
        derive = getattr(self.model._native_model, "qrac_plan", None)
        if not callable(derive):
            raise TypeError("Q7580: native QRAC planning is unavailable")
        plan = derive(self.variant.value)
        groups, observables, hamiltonian, width, row_limit, cell_limit = (
            _decode_native_plan(
                plan,
                self.model.variables,
                self.variant,
            )
        )
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "observables", observables)
        object.__setattr__(self, "num_qubits", width)
        object.__setattr__(self, "hamiltonian", hamiltonian)
        object.__setattr__(self, "_native_plan", plan)
        object.__setattr__(self, "_decode_row_limit", row_limit)
        object.__setattr__(self, "_decode_cell_limit", cell_limit)

    @property
    def correlations(self) -> Mapping[int, Hamiltonian]:
        """Return original variable IDs mapped to their decode observables."""

        return MappingProxyType(
            dict(zip(self.model.variables, self.observables, strict=True))
        )

    def sign_decode(self, expectations: Sequence[float]) -> DecodedSamples[_Domain]:
        """Round one complete expectation vector by sign."""

        values = _expectations(expectations, self.model.num_variables)
        decode = getattr(self._native_plan, "sign_decode", None)
        if not callable(decode):
            raise TypeError("Q7580: native QRAC planning is unavailable")
        return self._decoded_samples(decode(values))

    def decode(self, spins: Sequence[int]) -> DecodedSamples[_Domain]:
        """Decode one rounded spin vector into the original model domain."""

        return self.decode_many((spins,))

    def decode_many(
        self,
        spin_rows: Sequence[Sequence[int]],
    ) -> DecodedSamples[_Domain]:
        """Aggregate repeated rounded spin vectors into typed samples."""

        return self._decode_many(spin_rows)

    def _decode_many(self, spin_rows: object) -> DecodedSamples[_Domain]:
        """Validate and aggregate a dynamically supplied spin collection."""

        if isinstance(spin_rows, (str, bytes, bytearray)) or not isinstance(
            spin_rows, Sequence
        ):
            raise TypeError("spin rows must be a sequence of sequences")
        if not spin_rows:
            raise ValueError("spin rows cannot be empty")
        if (
            len(spin_rows) > self._decode_row_limit
            or len(spin_rows) * self.model.num_variables > self._decode_cell_limit
        ):
            raise ValueError("Q7585: QRAC decoding exceeds its work budget")
        rows = tuple(_spins(row, self.model.num_variables) for row in spin_rows)
        decode = getattr(self._native_plan, "decode", None)
        if not callable(decode):
            raise TypeError("Q7580: native QRAC planning is unavailable")
        return self._decoded_samples(decode(rows))

    def _decoded_samples(self, projection: object) -> DecodedSamples[_Domain]:
        """Validate and expose one native decoded-sample projection."""

        samples = _decode_native_samples(
            projection,
            self.model.variables,
        )
        try:
            return DecodedSamples(self.model.vartype, samples)
        except (TypeError, ValueError) as error:
            raise TypeError("Q7587: malformed native QRAC plan") from error

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Reconstruct the private native plan from stable public inputs."""

        return type(self), (self.model, self.variant)


def _decode_native_plan(
    plan: object,
    variables: tuple[int, ...],
    variant: QRACVariant,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[Hamiltonian, ...],
    Hamiltonian,
    int,
    int,
    int,
]:
    """Validate one transient native QRAC plan projection."""

    width = getattr(plan, "width", None)
    raw_groups = getattr(plan, "groups", None)
    raw_observables = getattr(plan, "observables", None)
    raw_hamiltonian = getattr(plan, "hamiltonian", None)
    row_limit = getattr(plan, "max_decode_rows", None)
    cell_limit = getattr(plan, "max_decode_cells", None)
    if (
        type(width) is not int
        or width < 0
        or not isinstance(raw_groups, list)
        or not isinstance(raw_observables, list)
        or len(raw_observables) != len(variables)
        or type(row_limit) is not int
        or row_limit <= 0
        or type(cell_limit) is not int
        or cell_limit <= 0
    ):
        raise TypeError("Q7587: malformed native QRAC plan")
    positions: list[int] = []
    groups: list[tuple[int, ...]] = []
    capacity = 2 if variant in {QRACVariant.QRAC21, QRACVariant.SPACE_EFFICIENT} else 3
    for raw_group in raw_groups:
        if (
            not isinstance(raw_group, list)
            or not 1 <= len(raw_group) <= capacity
            or any(type(position) is not int for position in raw_group)
        ):
            raise TypeError("Q7587: malformed native QRAC plan")
        positions.extend(raw_group)
        try:
            groups.append(tuple(variables[position] for position in raw_group))
        except IndexError as error:
            raise TypeError("Q7587: malformed native QRAC plan") from error
    if sorted(positions) != list(range(len(variables))):
        raise TypeError("Q7587: malformed native QRAC plan")
    if variant is QRACVariant.SPACE_EFFICIENT and positions != list(
        range(len(variables))
    ):
        raise TypeError("Q7587: malformed native QRAC plan")
    expected_width = (
        sum(1 if len(group) == 1 else 2 for group in groups)
        if variant is QRACVariant.QRAC32
        else len(groups)
    )
    if width != expected_width:
        raise TypeError("Q7587: malformed native QRAC plan")
    try:
        observables = tuple(
            _hamiltonian_from_native_header(value) for value in raw_observables
        )
        hamiltonian = _hamiltonian_from_native_header(raw_hamiltonian)
    except TypeError as error:
        raise TypeError("Q7587: malformed native QRAC plan") from error
    if any(value.num_qubits != width for value in observables) or (
        hamiltonian.num_qubits != width
    ):
        raise TypeError("Q7587: malformed native QRAC plan")
    return (
        tuple(groups),
        observables,
        hamiltonian,
        width,
        row_limit,
        cell_limit,
    )


def _decode_native_samples(
    projection: object,
    variables: tuple[int, ...],
) -> tuple[DecodedSample, ...]:
    """Validate one aggregated native QRAC decode projection."""

    if not isinstance(projection, list):
        raise TypeError("Q7587: malformed native QRAC plan")
    samples: list[DecodedSample] = []
    for row in projection:
        if (
            not isinstance(row, tuple)
            or len(row) != 3
            or not isinstance(row[0], list)
            or len(row[0]) != len(variables)
            or any(type(value) is not int for value in row[0])
            or not isinstance(row[1], float)
            or type(row[2]) is not int
        ):
            raise TypeError("Q7587: malformed native QRAC plan")
        samples.append(
            DecodedSample._from_native(
                tuple(zip(variables, row[0], strict=True)),
                row[1],
                row[2],
            )
        )
    return tuple(samples)


def _expectations(values: object, width: int) -> tuple[float, ...]:
    """Validate one QRAC observable coordinate."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("expectations must be a real numeric sequence")
    if len(values) != width:
        raise ValueError(f"expected {width} expectation values, got {len(values)}")
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


def _spins(values: object, width: int) -> tuple[int, ...]:
    """Validate one complete rounded spin vector."""

    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("spins must be a sequence")
    if len(values) != width:
        raise ValueError(f"expected {width} spin values, got {len(values)}")
    if any(type(value) is not int or value not in {-1, 1} for value in values):
        raise ValueError("spin values must be -1 or +1")
    return tuple(values)


__all__ = ["QRACEncoding", "QRACVariant"]
