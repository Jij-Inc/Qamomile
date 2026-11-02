"""Immutable arbitrary-order binary and spin polynomial models."""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, Literal, TypeAlias, TypeVar, cast, overload

from qamomile._native import compiler as native
from qamomile.compilation import SampleResult
from qamomile.observable import Hamiltonian, _hamiltonian_from_native_header


class VarType(enum.StrEnum):
    """Classical value domain of one polynomial model."""

    BINARY = "BINARY"
    SPIN = "SPIN"


BinaryDomain: TypeAlias = Literal[VarType.BINARY]
SpinDomain: TypeAlias = Literal[VarType.SPIN]
_Domain = TypeVar("_Domain", bound=VarType)
_Term = tuple[tuple[int, ...], float]
_RealInput: TypeAlias = int | float


@dataclass(frozen=True, slots=True)
class DecodedSample:
    """One unique decoded assignment and its observed multiplicity."""

    assignment: tuple[tuple[int, int], ...]
    energy: float
    occurrences: int

    def __post_init__(self) -> None:
        """Construct one canonical row validated by the native compiler."""

        projected = _validate_native_decoded_sample(
            self.assignment,
            self.energy,
            self.occurrences,
        )
        object.__setattr__(self, "assignment", projected.assignment)
        object.__setattr__(self, "energy", projected.energy)
        object.__setattr__(self, "occurrences", projected.occurrences)

    @classmethod
    def _from_native(
        cls,
        assignment: tuple[tuple[int, int], ...],
        energy: float,
        occurrences: int,
    ) -> DecodedSample:
        """Build one row already checked by the native distribution owner."""

        result = object.__new__(cls)
        object.__setattr__(result, "assignment", assignment)
        object.__setattr__(result, "energy", energy)
        object.__setattr__(result, "occurrences", occurrences)
        return result

    @property
    def values(self) -> Mapping[int, int]:
        """Return a read-only variable-value mapping."""

        return MappingProxyType(dict(self.assignment))


@dataclass(frozen=True, slots=True, init=False)
class DecodedSamples(Generic[_Domain]):
    """Typed immutable sample distribution for one polynomial domain."""

    vartype: _Domain
    samples: tuple[DecodedSample, ...]
    _total_occurrences: int = field(init=False, repr=False, compare=False)
    _mean_energy: float = field(init=False, repr=False, compare=False)
    _lowest_index: int = field(init=False, repr=False, compare=False)

    def __init__(self, vartype: _Domain, samples: tuple[DecodedSample, ...]) -> None:
        """Construct one distribution validated and summarized in Rust."""

        if not isinstance(vartype, VarType):
            raise TypeError("decoded sample vartype must be VarType")
        projected = _validate_native_decoded_samples(vartype, samples)
        object.__setattr__(self, "vartype", projected.vartype)
        object.__setattr__(self, "samples", projected.samples)
        object.__setattr__(self, "_total_occurrences", projected.total_occurrences)
        object.__setattr__(self, "_mean_energy", projected.mean_energy)
        object.__setattr__(self, "_lowest_index", projected._lowest_index)

    @classmethod
    def _from_native_projection(
        cls,
        vartype: _Domain,
        projection: object,
        *,
        diagnostic: str,
    ) -> DecodedSamples[_Domain]:
        """Build one distribution from a checked native projection."""

        samples, total, mean, lowest = _decode_native_sample_projection(
            projection,
            diagnostic=diagnostic,
        )
        result = object.__new__(cls)
        object.__setattr__(result, "vartype", vartype)
        object.__setattr__(result, "samples", samples)
        object.__setattr__(result, "_total_occurrences", total)
        object.__setattr__(result, "_mean_energy", mean)
        object.__setattr__(result, "_lowest_index", lowest)
        return result

    @property
    def total_occurrences(self) -> int:
        """Return the total number of represented observations."""

        return self._total_occurrences

    @property
    def mean_energy(self) -> float:
        """Return the occurrence-weighted mean objective value."""

        return self._mean_energy

    @property
    def lowest(self) -> DecodedSample:
        """Return the minimum-energy assignment with deterministic tie-breaking."""

        return self.samples[self._lowest_index]


@dataclass(frozen=True, slots=True, init=False)
class BinaryModel(Generic[_Domain]):
    """Canonical polynomial over BINARY ``0/1`` or SPIN ``-1/+1`` values.

    Variable IDs are arbitrary non-negative integers. Terms retain those IDs,
    while :attr:`hamiltonian` assigns qubits densely in sorted variable order.
    All constructors canonicalize duplicate indices using the selected domain
    identity and accumulate like terms exactly; no tolerance silently removes
    a scientific coefficient.
    """

    vartype: _Domain
    _terms: tuple[_Term, ...]
    constant: float
    variables: tuple[int, ...]
    _native_model: object = field(repr=False, compare=False)

    def __init__(
        self,
        vartype: _Domain,
        coefficients: Mapping[tuple[int, ...], _RealInput],
        constant: _RealInput = 0.0,
        *,
        variables: Sequence[int] | None = None,
    ) -> None:
        """Construct and canonicalize one polynomial model.

        Args:
            vartype: Algebra used to reduce repeated variable indices.
            coefficients: Product terms keyed by original variable IDs.
            constant: Scalar objective offset.
            variables: Optional complete variable universe. This retains
                zero-cost or disconnected variables that do not occur in a
                nonzero coefficient.

        Raises:
            TypeError: If the domain, indices, or coefficients are ill-typed.
            ValueError: If an index is negative or a coefficient is non-finite.
        """

        if not isinstance(vartype, VarType):
            raise TypeError("vartype must be VarType.BINARY or VarType.SPIN")
        if not isinstance(coefficients, Mapping):
            raise TypeError("coefficients must be a mapping")
        raw_terms: list[tuple[str, ...]] = []
        raw_coefficients: list[float] = []
        for raw_indices, raw_coefficient in coefficients.items():
            if not isinstance(raw_indices, tuple):
                raise TypeError("coefficient keys must be tuples of variable IDs")
            raw_terms.append(tuple(_variable_hex(index) for index in raw_indices))
            raw_coefficients.append(_finite_real("coefficient", raw_coefficient))
        canonical_constant = _finite_real("constant", constant)
        explicit_variables = _variable_universe_input(variables)
        model_type = getattr(native, "SparsePolynomial", None)
        if model_type is None:
            raise TypeError("Q7560: native sparse polynomial support is unavailable")
        projection = model_type(
            vartype.value,
            raw_terms,
            raw_coefficients,
            canonical_constant,
            explicit_variables,
        )
        terms, canonical_constant, canonical_variables = _decode_native_polynomial(
            projection,
            vartype,
        )
        object.__setattr__(self, "vartype", vartype)
        object.__setattr__(self, "_terms", terms)
        object.__setattr__(self, "constant", canonical_constant)
        object.__setattr__(self, "variables", canonical_variables)
        object.__setattr__(self, "_native_model", projection)

    @staticmethod
    def from_hubo(
        coefficients: Mapping[tuple[int, ...], _RealInput],
        constant: _RealInput = 0.0,
        *,
        variables: Sequence[int] | None = None,
    ) -> BinaryModel[BinaryDomain]:
        """Construct an arbitrary-order BINARY polynomial."""

        return BinaryModel(
            VarType.BINARY,
            coefficients,
            constant,
            variables=variables,
        )

    @staticmethod
    def from_ising(
        coefficients: Mapping[tuple[int, ...], _RealInput],
        constant: _RealInput = 0.0,
        *,
        variables: Sequence[int] | None = None,
    ) -> BinaryModel[SpinDomain]:
        """Construct an arbitrary-order SPIN polynomial."""

        return BinaryModel(
            VarType.SPIN,
            coefficients,
            constant,
            variables=variables,
        )

    @property
    def coefficients(self) -> Mapping[tuple[int, ...], float]:
        """Return the canonical nonconstant coefficient mapping."""

        return MappingProxyType(dict(self._terms))

    @property
    def num_variables(self) -> int:
        """Return the number of densely encoded variables."""

        return len(self.variables)

    @property
    def order(self) -> int:
        """Return the maximum polynomial degree, or zero for a constant."""

        return max((len(indices) for indices, _ in self._terms), default=0)

    def to_bytes(self) -> bytes:
        """Serialize this model as a canonical Rust-sealed problem artifact."""

        encode = getattr(self._native_model, "to_bytes", None)
        if not callable(encode):
            raise TypeError(
                "Q7560: native sparse polynomial persistence is unavailable"
            )
        payload = encode()
        if not isinstance(payload, bytes):
            raise TypeError("Q7576: malformed native optimization problem artifact")
        return payload

    @classmethod
    def from_bytes(
        cls,
        payload: bytes | bytearray | memoryview,
    ) -> BinaryModel[VarType]:
        """Decode and verify a canonical polynomial problem artifact."""

        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("optimization problem artifact must be bytes-like")
        model_type = getattr(native, "SparsePolynomial", None)
        decode = getattr(model_type, "from_bytes", None)
        if not callable(decode):
            raise TypeError(
                "Q7560: native sparse polynomial persistence is unavailable"
            )
        projection = decode(bytes(payload))
        try:
            vartype = VarType(getattr(projection, "domain", None))
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Q7576: malformed native optimization problem projection"
            ) from error
        return cast(BinaryModel[VarType], cls._from_native(vartype, projection))

    @property
    def semantic_digest(self) -> str:
        """Return the Rust-owned SHA-256 identity of this canonical problem."""

        digest = getattr(self._native_model, "semantic_digest", None)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise TypeError("Q7576: malformed native optimization problem digest")
        return digest

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Serialize through canonical problem bytes, not a native handle."""

        return (_restore_binary_model, (self.to_bytes(),))

    @overload
    def _as_vartype(self, vartype: BinaryDomain) -> BinaryModel[BinaryDomain]:
        """Convert this model to the binary variable domain.

        Args:
            vartype (BinaryDomain): Binary-domain selector.

        Returns:
            BinaryModel[BinaryDomain]: Exactly equivalent binary model.
        """

        ...

    @overload
    def _as_vartype(self, vartype: SpinDomain) -> BinaryModel[SpinDomain]:
        """Convert this model to the spin variable domain.

        Args:
            vartype (SpinDomain): Spin-domain selector.

        Returns:
            BinaryModel[SpinDomain]: Exactly equivalent spin model.
        """

        ...

    def _as_vartype(
        self,
        vartype: VarType,
    ) -> BinaryModel[BinaryDomain] | BinaryModel[SpinDomain]:
        """Convert this model through the native polynomial domain relation.

        Args:
            vartype (VarType): Binary or spin destination domain.

        Returns:
            BinaryModel[BinaryDomain] | BinaryModel[SpinDomain]: Equivalent
                immutable model in the requested domain.

        Raises:
            TypeError: If the domain is invalid or native conversion is missing.
        """

        if not isinstance(vartype, VarType):
            raise TypeError("vartype must be VarType.BINARY or VarType.SPIN")
        if vartype is self.vartype:
            return self._from_native(vartype, self._native_model)
        convert = getattr(self._native_model, "convert", None)
        if convert is None:
            raise TypeError("Q7560: native sparse polynomial support is unavailable")
        return self._from_native(vartype, convert(vartype.value))

    @staticmethod
    def _from_native(
        vartype: VarType,
        projection: object,
    ) -> BinaryModel[BinaryDomain] | BinaryModel[SpinDomain]:
        """Construct one public immutable value from a checked native normal form."""

        terms, constant, variables = _decode_native_polynomial(projection, vartype)
        model = object.__new__(BinaryModel)
        object.__setattr__(model, "vartype", vartype)
        object.__setattr__(model, "_terms", terms)
        object.__setattr__(model, "constant", constant)
        object.__setattr__(model, "variables", variables)
        object.__setattr__(model, "_native_model", projection)
        return model

    def to_binary(self) -> BinaryModel[BinaryDomain]:
        """Convert this model to the binary variable domain.

        Returns:
            BinaryModel[BinaryDomain]: Exactly equivalent binary model.

        Raises:
            TypeError: If native polynomial conversion is unavailable.
        """

        return self._as_vartype(VarType.BINARY)

    def to_ising(self) -> BinaryModel[SpinDomain]:
        """Convert this model to the spin variable domain.

        Returns:
            BinaryModel[SpinDomain]: Exactly equivalent spin model.

        Raises:
            TypeError: If native polynomial conversion is unavailable.
        """

        return self._as_vartype(VarType.SPIN)

    def normalize_by_abs_max(self) -> BinaryModel[_Domain]:
        """Return a model scaled by its largest absolute coefficient.

        The constant is divided by the same factor, while the variable domain,
        variable universe, and source model remain unchanged. A model without
        nonconstant terms is returned unchanged as a new immutable value.

        Returns:
            BinaryModel[_Domain]: Canonical normalized model in the same domain.

        Raises:
            TypeError: If native polynomial normalization is unavailable or
                returns a malformed projection.
            ValueError: If normalization exceeds the finite binary64 range.
        """

        normalize = getattr(self._native_model, "normalized_by_abs_max", None)
        if not callable(normalize):
            raise TypeError("Q7560: native sparse polynomial support is unavailable")
        return cast(BinaryModel[_Domain], self._from_native(self.vartype, normalize()))

    def normalize_by_rms(self) -> BinaryModel[_Domain]:
        """Return a model scaled by its degree-wise coefficient RMS.

        For each represented degree, the mean squared coefficient is computed;
        their sum's square root scales every coefficient and the constant. The
        variable domain, variable universe, and source model remain unchanged.
        A model without nonconstant terms is returned unchanged as a new value.

        Returns:
            BinaryModel[_Domain]: Canonical normalized model in the same domain.

        Raises:
            TypeError: If native polynomial normalization is unavailable or
                returns a malformed projection.
            ValueError: If normalization exceeds the finite binary64 range.
        """

        normalize = getattr(self._native_model, "normalized_by_rms", None)
        if not callable(normalize):
            raise TypeError("Q7560: native sparse polynomial support is unavailable")
        return cast(BinaryModel[_Domain], self._from_native(self.vartype, normalize()))

    @property
    def hamiltonian(self) -> Hamiltonian:
        """Return the dense commuting-Z Hamiltonian for this objective."""

        project = getattr(self._native_model, "hamiltonian", None)
        if not callable(project):
            raise TypeError("Q7560: native sparse polynomial support is unavailable")
        return _hamiltonian_from_native_header(project())

    def energy(self, assignment: Mapping[int, int] | Sequence[int]) -> float:
        """Evaluate one assignment in original-ID or dense variable order."""

        variables, values = self._assignment_input(assignment)
        evaluate = getattr(self._native_model, "energy_assignment", None)
        if not callable(evaluate):
            raise TypeError("Q7560: native sparse polynomial support is unavailable")
        energy = evaluate(values, variables)
        if not isinstance(energy, float) or not math.isfinite(energy):
            raise TypeError("Q7568: malformed native sparse polynomial result")
        return energy

    def decode(self, result: SampleResult) -> DecodedSamples[_Domain]:
        """Decode logical measurement counts into this model's value domain."""

        if not isinstance(result, SampleResult):
            raise TypeError("result must be a qamomile.SampleResult")
        if result.width != self.num_variables:
            raise ValueError(
                f"sample width {result.width} does not match model width "
                f"{self.num_variables}"
            )
        decode = getattr(self._native_model, "decode_samples", None)
        if not callable(decode):
            raise TypeError("Q7560: native sparse polynomial decoding is unavailable")
        return DecodedSamples._from_native_projection(
            self.vartype,
            decode(result._native_summary),
            diagnostic="Q7568",
        )

    def _assignment_input(
        self,
        assignment: Mapping[int, int] | Sequence[int],
    ) -> tuple[list[str] | None, list[str]]:
        """Encode one host assignment without reproducing polynomial semantics."""

        if isinstance(assignment, Mapping):
            items = list(assignment.items())
            variables = [_variable_hex(index) for index, _ in items]
            values = [_assignment_value(value) for _, value in items]
        elif isinstance(assignment, Sequence) and not isinstance(
            assignment, (str, bytes, bytearray)
        ):
            variables = None
            values = [_assignment_value(value) for value in assignment]
        else:
            raise TypeError("assignment must be a mapping or dense sequence")
        return variables, values


def _validate_native_decoded_samples(
    vartype: VarType,
    samples: tuple[DecodedSample, ...],
) -> DecodedSamples[VarType]:
    """Validate one public decoded distribution through the Rust compiler."""

    if not isinstance(samples, tuple) or any(
        not isinstance(sample, DecodedSample) for sample in samples
    ):
        raise TypeError("decoded samples must be a tuple of DecodedSample values")
    validate = getattr(native, "validate_decoded_samples", None)
    if not callable(validate):
        raise TypeError("Q7560: native decoded-sample validation is unavailable")
    rows = [
        _decoded_sample_row(sample.assignment, sample.energy, sample.occurrences)
        for sample in samples
    ]
    return DecodedSamples._from_native_projection(
        vartype,
        validate(vartype.value, rows),
        diagnostic="Q7568",
    )


def _validate_native_decoded_sample(
    assignment: tuple[tuple[int, int], ...],
    energy: float,
    occurrences: int,
) -> DecodedSample:
    """Validate one standalone decoded row through the Rust compiler."""

    validate = getattr(native, "validate_decoded_samples", None)
    if not callable(validate):
        raise TypeError("Q7560: native decoded-sample validation is unavailable")
    samples, _, _, _ = _decode_native_sample_projection(
        validate(None, [_decoded_sample_row(assignment, energy, occurrences)]),
        diagnostic="Q7568",
    )
    if len(samples) != 1:
        raise TypeError("Q7568: malformed native decoded samples")
    return samples[0]


def _decoded_sample_row(
    assignment: tuple[tuple[int, int], ...],
    energy: float,
    occurrences: int,
) -> tuple[list[str], list[str], float, str]:
    """Encode one public decoded row without reproducing its invariants."""

    if not isinstance(assignment, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in assignment
    ):
        raise TypeError("decoded assignment must be a tuple of variable-value pairs")
    variables: list[str] = []
    values: list[str] = []
    for variable, value in assignment:
        if type(variable) is not int:
            raise TypeError("decoded variable IDs must be integers")
        if type(value) is not int:
            raise TypeError("decoded assignment values must be integers")
        variables.append(format(variable, "x"))
        values.append(str(value))
    if isinstance(energy, bool) or not isinstance(energy, (int, float)):
        raise TypeError("decoded sample energy must be a real number")
    if type(occurrences) is not int:
        raise TypeError("decoded sample occurrences must be an integer")
    return variables, values, float(energy), str(occurrences)


def _decode_native_sample_projection(
    projection: object,
    *,
    diagnostic: str,
) -> tuple[tuple[DecodedSample, ...], int, float, int]:
    """Validate one transient native decoded-distribution projection."""

    if (
        not isinstance(projection, tuple)
        or len(projection) != 4
        or not isinstance(projection[0], list)
        or not isinstance(projection[1], str)
        or not isinstance(projection[2], float)
        or not math.isfinite(projection[2])
        or type(projection[3]) is not int
    ):
        raise TypeError(f"{diagnostic}: malformed native decoded samples")
    samples: list[DecodedSample] = []
    try:
        for row in projection[0]:
            if (
                not isinstance(row, tuple)
                or len(row) != 4
                or not isinstance(row[0], list)
                or any(not isinstance(value, str) for value in row[0])
                or not isinstance(row[1], list)
                or any(not isinstance(value, str) for value in row[1])
                or not isinstance(row[2], float)
                or not math.isfinite(row[2])
                or not isinstance(row[3], str)
            ):
                raise TypeError
            variables = tuple(int(value, 16) for value in row[0])
            values = tuple(int(value, 10) for value in row[1])
            occurrences = int(row[3], 10)
            if len(variables) != len(values):
                raise TypeError
            samples.append(
                DecodedSample._from_native(
                    tuple(zip(variables, values, strict=True)),
                    row[2],
                    occurrences,
                )
            )
        total = int(projection[1], 10)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{diagnostic}: malformed native decoded samples") from error
    lowest = projection[3]
    if not samples or total <= 0 or lowest < 0 or lowest >= len(samples):
        raise TypeError(f"{diagnostic}: malformed native decoded samples")
    return tuple(samples), total, projection[2], lowest


def _variable_universe_input(variables: object) -> list[str] | None:
    """Validate the Python container and encode optional arbitrary-size IDs."""

    if variables is None:
        return None
    if isinstance(variables, (str, bytes, bytearray)) or not isinstance(
        variables, Sequence
    ):
        raise TypeError("variables must be a sequence of variable IDs")
    return [_variable_hex(index) for index in variables]


def _variable_hex(index: object) -> str:
    """Encode one non-negative arbitrary-precision Python variable ID."""

    if type(index) is not int:
        raise TypeError("variable IDs must be integers")
    if index < 0:
        raise ValueError("variable IDs must be non-negative")
    return format(index, "x")


def _assignment_value(value: object) -> str:
    """Encode one exact Python integer for native assignment validation."""

    if type(value) is not int:
        raise TypeError("assignment values must be integers")
    return str(value)


def _decode_native_polynomial(
    projection: object,
    vartype: VarType,
) -> tuple[tuple[_Term, ...], float, tuple[int, ...]]:
    """Validate one transient native polynomial projection."""

    domain = getattr(projection, "domain", None)
    raw_terms = getattr(projection, "terms", None)
    constant = getattr(projection, "constant", None)
    raw_variables = getattr(projection, "variables", None)
    if (
        domain != vartype.value
        or not isinstance(raw_terms, list)
        or not isinstance(constant, float)
        or not math.isfinite(constant)
        or not isinstance(raw_variables, list)
    ):
        raise TypeError("Q7568: malformed native sparse polynomial result")
    try:
        variables = tuple(int(value, 16) for value in raw_variables)
    except (TypeError, ValueError) as error:
        raise TypeError("Q7568: malformed native sparse polynomial result") from error
    if tuple(sorted(set(variables))) != variables or any(
        value < 0 for value in variables
    ):
        raise TypeError("Q7568: malformed native sparse polynomial result")
    terms: list[_Term] = []
    previous: tuple[int, ...] | None = None
    for row in raw_terms:
        if (
            not isinstance(row, tuple)
            or len(row) != 2
            or not isinstance(row[0], list)
            or not isinstance(row[1], float)
            or not math.isfinite(row[1])
            or row[1] == 0.0
        ):
            raise TypeError("Q7568: malformed native sparse polynomial result")
        try:
            indices = tuple(int(value, 16) for value in row[0])
        except (TypeError, ValueError) as error:
            raise TypeError(
                "Q7568: malformed native sparse polynomial result"
            ) from error
        if (
            not indices
            or tuple(sorted(set(indices))) != indices
            or not set(indices).issubset(variables)
            or (previous is not None and indices <= previous)
        ):
            raise TypeError("Q7568: malformed native sparse polynomial result")
        terms.append((indices, row[1]))
        previous = indices
    return tuple(terms), constant, variables


def _finite_real(name: str, value: object) -> float:
    """Normalize one finite non-Boolean real coefficient."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _restore_binary_model(
    payload: bytes,
) -> BinaryModel[VarType]:
    """Rebuild a pickled model through canonical native persistence."""

    return BinaryModel.from_bytes(payload)


__all__ = [
    "BinaryDomain",
    "BinaryModel",
    "DecodedSample",
    "DecodedSamples",
    "SpinDomain",
    "VarType",
]
