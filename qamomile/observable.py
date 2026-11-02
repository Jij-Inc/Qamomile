"""Immutable backend-neutral Pauli observables for Qamomile."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Mapping, TypeAlias

from ._native import compiler as native

Scalar: TypeAlias = int | float | complex


class Pauli(enum.StrEnum):
    """One single-qubit Pauli operator."""

    X = "X"
    Y = "Y"
    Z = "Z"


@dataclass(frozen=True, order=True, slots=True)
class PauliFactor:
    """One non-identity Pauli acting on a logical qubit index."""

    index: int
    pauli: Pauli

    def __post_init__(self) -> None:
        """Validate the logical index and normalize the Pauli spelling."""

        if isinstance(self.index, bool) or not isinstance(self.index, int):
            raise TypeError("Pauli index must be an integer")
        if self.index < 0:
            raise ValueError("Pauli index must be non-negative")


@dataclass(frozen=True, order=True, slots=True)
class PauliWord:
    """A canonical tensor product of non-identity Pauli factors."""

    factors: tuple[PauliFactor, ...] = ()

    def __post_init__(self) -> None:
        """Require sorted unique logical indices."""

        canonical = tuple(sorted(self.factors))
        if len({factor.index for factor in canonical}) != len(canonical):
            raise ValueError("PauliWord contains duplicate logical indices")
        object.__setattr__(self, "factors", canonical)

    @property
    def width(self) -> int:
        """Return the minimum register width containing this word."""

        return 0 if not self.factors else self.factors[-1].index + 1

    def commutes_with(self, other: PauliWord) -> bool:
        """Return whether two Pauli words commute.

        Args:
            other (PauliWord): Canonical word to compare with this word.

        Returns:
            bool: Whether the two operators commute.

        Raises:
            TypeError: If the installed native compiler lacks the Pauli
                algebra boundary or violates its result ABI.
            ValueError: If Rust rejects either Pauli word.
        """

        commute = getattr(native, "pauli_words_commute", None)
        if not callable(commute):
            raise TypeError("Q7400: native Pauli algebra is unavailable")
        result = commute(
            tuple((factor.index, factor.pauli.value) for factor in self.factors),
            tuple((factor.index, factor.pauli.value) for factor in other.factors),
        )
        if type(result) is not bool:
            raise TypeError("native Pauli algebra violates the Python ABI")
        return result


@dataclass(frozen=True, slots=True)
class Hamiltonian:
    """A canonical finite sum of Pauli words with complex coefficients."""

    terms: tuple[tuple[PauliWord, complex], ...] = ()
    constant: complex = 0j
    num_qubits: int = 0
    _is_hermitian: bool = field(init=False, repr=False, compare=False)
    _is_pairwise_commuting: bool | None = field(init=False, repr=False, compare=False)
    _native_literal: bytes = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Merge duplicate words, remove exact zeros, and freeze canonical order."""

        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int):
            raise TypeError("Hamiltonian num_qubits must be an integer")
        if self.num_qubits < 0:
            raise ValueError("Hamiltonian num_qubits must be non-negative")
        constant = complex(self.constant)
        terms: list[tuple[PauliWord, complex]] = []
        for word, coefficient in self.terms:
            if not isinstance(word, PauliWord):
                raise TypeError("Hamiltonian terms must contain PauliWord values")
            terms.append((word, complex(coefficient)))
        _install_native_observable(
            self,
            _normalize_observable(
                tuple(terms),
                constant,
                self.num_qubits,
            ),
        )

    @classmethod
    def zero(cls, *, num_qubits: int = 0) -> Hamiltonian:
        """Construct the zero observable."""

        return cls(num_qubits=num_qubits)

    @classmethod
    def identity(
        cls,
        coefficient: Scalar = 1,
        *,
        num_qubits: int = 0,
    ) -> Hamiltonian:
        """Construct a scalar multiple of identity."""

        return cls(constant=complex(coefficient), num_qubits=num_qubits)

    @classmethod
    def pauli(
        cls,
        pauli: Pauli,
        index: int,
        coefficient: Scalar = 1,
    ) -> Hamiltonian:
        """Construct one single-qubit Pauli term."""

        word = PauliWord((PauliFactor(index, pauli),))
        return cls(((word, complex(coefficient)),), num_qubits=index + 1)

    @property
    def is_hermitian(self) -> bool:
        """Return whether every coefficient is exactly real."""

        return self._is_hermitian

    @property
    def is_pairwise_commuting(self) -> bool:
        """Return whether all non-identity terms commute pairwise."""

        cached = self._is_pairwise_commuting
        if cached is not None:
            return cached
        inspect = getattr(native, "observable_is_pairwise_commuting", None)
        if not callable(inspect):
            raise TypeError("Q7400: native observable algebra is unavailable")
        value = inspect(self._native_literal)
        if type(value) is not bool:
            raise TypeError("native observable algebra violates the Python ABI")
        object.__setattr__(self, "_is_pairwise_commuting", value)
        return value

    def remap(self, indices: Mapping[int, int]) -> Hamiltonian:
        """Return this observable with logical qubit indices remapped."""

        if not isinstance(indices, Mapping):
            raise TypeError("observable remap must be a mapping")
        rows: list[tuple[int, int]] = []
        for source, target in indices.items():
            if (
                isinstance(source, bool)
                or not isinstance(source, int)
                or isinstance(target, bool)
                or not isinstance(target, int)
            ):
                raise TypeError("observable remap indices must be integers")
            if source < 0 or target < 0:
                raise ValueError("observable remap indices must be non-negative")
            rows.append((source, target))
        return _remap_observable(self, tuple(sorted(rows)))

    def to_portable(self) -> dict[str, object]:
        """Encode this value in QCore's closed literal domain."""

        return {
            "kind": "qamomile.pauli_sum.v1",
            "num_qubits": self.num_qubits,
            "constant": self.constant,
            "terms": tuple(
                (
                    coefficient,
                    tuple(
                        (factor.pauli.value, factor.index) for factor in word.factors
                    ),
                )
                for word, coefficient in self.terms
            ),
        }

    @classmethod
    def from_portable(cls, value: object) -> Hamiltonian:
        """Decode a validated QCore literal representation."""

        if (
            not isinstance(value, Mapping)
            or value.get("kind") != "qamomile.pauli_sum.v1"
        ):
            raise TypeError("value is not a portable Hamiltonian")
        raw_terms = value.get("terms")
        if not isinstance(raw_terms, (tuple, list)):
            raise TypeError("portable Hamiltonian terms are malformed")
        terms: list[tuple[PauliWord, complex]] = []
        for raw_term in raw_terms:
            if not isinstance(raw_term, (tuple, list)) or len(raw_term) != 2:
                raise TypeError("portable Hamiltonian term is malformed")
            coefficient, raw_factors = raw_term
            if not isinstance(raw_factors, (tuple, list)):
                raise TypeError("portable Pauli word is malformed")
            factors = tuple(
                PauliFactor(int(raw_index), Pauli(str(raw_pauli)))
                for raw_pauli, raw_index in raw_factors
            )
            terms.append((PauliWord(factors), complex(coefficient)))
        return cls(
            tuple(terms),
            complex(value.get("constant", 0)),
            int(value.get("num_qubits", 0)),
        )

    def __add__(self, other: object) -> Hamiltonian:
        """Add another Hamiltonian or scalar identity term."""

        if isinstance(other, (int, float, complex)):
            return _affine_observable(self, scale=1, shift=complex(other))
        if isinstance(other, Hamiltonian):
            return _add_scaled_observable(self, other, scale=1)
        return NotImplemented

    def __radd__(self, other: object) -> Hamiltonian:
        """Add a scalar identity term from the left."""

        return self + other

    def __neg__(self) -> Hamiltonian:
        """Negate every coefficient."""

        return _affine_observable(self, scale=-1)

    def __sub__(self, other: object) -> Hamiltonian:
        """Subtract another Hamiltonian or scalar identity term."""

        if isinstance(other, Hamiltonian):
            return _add_scaled_observable(self, other, scale=-1)
        if isinstance(other, (int, float, complex)):
            return _affine_observable(self, scale=1, shift=-complex(other))
        return NotImplemented

    def __rsub__(self, other: object) -> Hamiltonian:
        """Subtract this Hamiltonian from a scalar or Hamiltonian."""

        if isinstance(other, Hamiltonian):
            return _add_scaled_observable(other, self, scale=-1)
        if isinstance(other, (int, float, complex)):
            return _affine_observable(self, scale=-1, shift=complex(other))
        return NotImplemented

    def __mul__(self, other: object) -> Hamiltonian:
        """Multiply observables using the exact Pauli algebra."""

        if isinstance(other, (int, float, complex)):
            return _affine_observable(self, scale=complex(other))
        if not isinstance(other, Hamiltonian):
            return NotImplemented
        return _multiply_observables(self, other)

    def __rmul__(self, other: object) -> Hamiltonian:
        """Multiply by a scalar from the left."""

        if isinstance(other, (int, float, complex)):
            return self * other
        if isinstance(other, Hamiltonian):
            return other * self
        return NotImplemented


def X(index: int) -> Hamiltonian:
    """Return Pauli X on one logical qubit."""

    return Hamiltonian.pauli(Pauli.X, index)


def Y(index: int) -> Hamiltonian:
    """Return Pauli Y on one logical qubit."""

    return Hamiltonian.pauli(Pauli.Y, index)


def Z(index: int) -> Hamiltonian:
    """Return Pauli Z on one logical qubit."""

    return Hamiltonian.pauli(Pauli.Z, index)


def zero_observable(num_qubits: int) -> Hamiltonian:
    """Construct an additive observable identity with an explicit width.

    Inside a Qamomile function this call becomes a symbolic ``Const``
    observable and can be carried through ordinary loops. On the host it
    returns the equivalent realized Hamiltonian.

    Args:
        num_qubits (int): Non-negative logical width retained by the zero
            observable.

    Returns:
        Hamiltonian: Canonical zero Hamiltonian with the requested width.

    Raises:
        TypeError: If ``num_qubits`` is not an integer.
        ValueError: If ``num_qubits`` is negative.
    """

    return Hamiltonian.zero(num_qubits=num_qubits)


def commutator(left: Hamiltonian, right: Hamiltonian) -> Hamiltonian:
    """Return the exact commutator ``left * right - right * left``."""

    return left * right - right * left


def _normalize_observable(
    terms: tuple[tuple[PauliWord, complex], ...],
    constant: complex,
    num_qubits: int,
) -> tuple[bytes, bool]:
    """Normalize one Hamiltonian through the required Rust algebra.

    Args:
        terms (tuple[tuple[PauliWord, complex], ...]): Raw Pauli terms.
        constant (complex): Raw identity coefficient.
        num_qubits (int): Explicit minimum logical width.

    Returns:
        tuple[bytes, bool]: Canonical literal bytes and its Hermiticity fact.

    Raises:
        TypeError: If the installed native compiler lacks the algebra boundary.
        ValueError: If Rust rejects the portable observable domain.
    """

    normalize = getattr(native, "normalize_observable", None)
    if not callable(normalize):
        raise TypeError("Q7400: native observable algebra is unavailable")
    return _validate_native_header(
        normalize(_observable_payload(terms, constant, num_qubits))
    )


def _multiply_observables(left: Hamiltonian, right: Hamiltonian) -> Hamiltonian:
    """Multiply two Hamiltonians through the required Rust algebra.

    Args:
        left (Hamiltonian): Left operand.
        right (Hamiltonian): Right operand.

    Returns:
        Hamiltonian: Canonical Pauli product with compiler-derived facts.

    Raises:
        TypeError: If the installed native compiler lacks the algebra boundary
            or violates its result ABI.
        ValueError: If Rust rejects either portable observable.
    """

    multiply = getattr(native, "multiply_observables", None)
    if not callable(multiply):
        raise TypeError("Q7400: native observable algebra is unavailable")
    header = _validate_native_header(
        multiply(
            left._native_literal,
            right._native_literal,
        )
    )
    result = object.__new__(Hamiltonian)
    _install_native_observable(result, header)
    return result


def _affine_observable(
    value: Hamiltonian,
    *,
    scale: complex,
    shift: complex = 0j,
) -> Hamiltonian:
    """Apply one compiler-owned complex scale and identity shift.

    Args:
        value (Hamiltonian): Canonical operand.
        scale (complex): Coefficient applied to every observable term.
        shift (complex): Identity coefficient added after scaling.

    Returns:
        Hamiltonian: Canonical affine result.

    Raises:
        TypeError: If the installed native compiler lacks the algebra boundary
            or violates its result ABI.
        ValueError: If Rust rejects the operand, scalars, or result range.
    """

    affine = getattr(native, "affine_observable", None)
    if not callable(affine):
        raise TypeError("Q7400: native observable algebra is unavailable")
    return _hamiltonian_from_native_header(
        affine(
            value._native_literal,
            (scale.real, scale.imag),
            (shift.real, shift.imag),
        )
    )


def _add_scaled_observable(
    left: Hamiltonian,
    right: Hamiltonian,
    *,
    scale: complex,
) -> Hamiltonian:
    """Add one compiler-owned scaled observable to another.

    Args:
        left (Hamiltonian): Unscaled left operand.
        right (Hamiltonian): Right operand to scale and add.
        scale (complex): Coefficient applied to the right operand.

    Returns:
        Hamiltonian: Canonical linear-combination result.

    Raises:
        TypeError: If the installed native compiler lacks the algebra boundary
            or violates its result ABI.
        ValueError: If Rust rejects either operand, the scale, or result range.
    """

    combine = getattr(native, "add_scaled_observable", None)
    if not callable(combine):
        raise TypeError("Q7400: native observable algebra is unavailable")
    return _hamiltonian_from_native_header(
        combine(
            left._native_literal,
            right._native_literal,
            (scale.real, scale.imag),
        )
    )


def _remap_observable(
    value: Hamiltonian,
    indices: tuple[tuple[int, int], ...],
) -> Hamiltonian:
    """Apply one compiler-owned logical-index substitution.

    Args:
        value (Hamiltonian): Canonical observable to relabel.
        indices (tuple[tuple[int, int], ...]): Sorted source/destination rows.

    Returns:
        Hamiltonian: Canonical remapped result.

    Raises:
        TypeError: If the installed native compiler lacks the algebra boundary
            or violates its result ABI.
        ValueError: If Rust rejects an index or a non-injective word mapping.
    """

    remap = getattr(native, "remap_observable", None)
    if not callable(remap):
        raise TypeError("Q7400: native observable algebra is unavailable")
    return _hamiltonian_from_native_header(remap(value._native_literal, indices))


def _hamiltonian_from_native_header(value: object) -> Hamiltonian:
    """Construct a Hamiltonian from one compiler-owned observable result.

    Args:
        value (object): Native canonical-observable header.

    Returns:
        Hamiltonian: Immutable Python projection of the native normal form.

    Raises:
        TypeError: If the native extension violates its result ABI.
    """

    result = object.__new__(Hamiltonian)
    _install_native_observable(result, _validate_native_header(value))
    return result


def _observable_payload(
    terms: tuple[tuple[PauliWord, complex], ...],
    constant: complex,
    num_qubits: int,
) -> bytes:
    """Encode one raw observable in the existing portable literal schema.

    Args:
        terms (tuple[tuple[PauliWord, complex], ...]): Pauli terms to encode.
        constant (complex): Identity coefficient to encode.
        num_qubits (int): Explicit minimum logical width.

    Returns:
        bytes: Deterministic standalone ``LiteralProto`` bytes.
    """

    from .ir import FrozenMapping
    from .proto import qcore_pb2 as wire
    from .serialization import encode_literal

    portable = FrozenMapping.from_mapping(
        {
            "kind": "qamomile.pauli_sum.v1",
            "num_qubits": num_qubits,
            "constant": constant,
            "terms": tuple(
                (
                    coefficient,
                    tuple(
                        (factor.pauli.value, factor.index) for factor in word.factors
                    ),
                )
                for word, coefficient in terms
            ),
        }
    )
    literal = wire.LiteralProto()
    encode_literal(portable, literal)
    return literal.SerializeToString(deterministic=True)


def _validate_native_header(value: object) -> tuple[bytes, bool]:
    """Validate the thin native observable result ABI.

    Args:
        value (object): Result returned by a native observable operation.

    Returns:
        tuple[bytes, bool]: Canonical literal bytes and Hermiticity.

    Raises:
        TypeError: If the native extension returns a malformed projection.
    """

    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or not isinstance(value[0], bytes)
        or type(value[1]) is not bool
    ):
        raise TypeError("native observable algebra violates the Python ABI")
    return value


def _install_native_observable(
    target: Hamiltonian,
    header: tuple[bytes, bool],
) -> None:
    """Install one compiler-normalized observable on a frozen value object.

    Args:
        target (Hamiltonian): Uninitialized or raw frozen value to populate.
        header (tuple[bytes, bool]): Native literal and Hermiticity fact.

    Raises:
        TypeError: If the native literal violates the closed bridge ABI.
    """

    terms, constant, num_qubits = _decode_observable_payload(header[0])
    object.__setattr__(target, "terms", terms)
    object.__setattr__(target, "constant", constant)
    object.__setattr__(target, "num_qubits", num_qubits)
    object.__setattr__(target, "_is_hermitian", header[1])
    object.__setattr__(target, "_is_pairwise_commuting", None)
    object.__setattr__(target, "_native_literal", header[0])


def _decode_observable_payload(
    payload: bytes,
) -> tuple[tuple[tuple[PauliWord, complex], ...], complex, int]:
    """Decode one trusted native observable projection defensively.

    Args:
        payload (bytes): Canonical standalone ``LiteralProto`` bytes.

    Returns:
        tuple[tuple[tuple[PauliWord, complex], ...], complex, int]: Canonical
        term row, identity coefficient, and logical width.

    Raises:
        TypeError: If the native projection is malformed or noncanonical.
    """

    from .ir import FrozenMapping, thaw_value
    from .proto import qcore_pb2 as wire
    from .serialization import decode_literal

    literal = wire.LiteralProto()
    try:
        literal.ParseFromString(payload)
    except ValueError as error:
        raise TypeError("native observable literal is malformed") from error
    if literal.SerializeToString(deterministic=True) != payload:
        raise TypeError("native observable literal is not canonical")
    portable = decode_literal(literal)
    if not isinstance(portable, FrozenMapping):
        raise TypeError("native observable literal is not a mapping")
    value = thaw_value(portable)
    if not isinstance(value, dict) or set(value) != {
        "constant",
        "kind",
        "num_qubits",
        "terms",
    }:
        raise TypeError("native observable literal has invalid fields")
    if value["kind"] != "qamomile.pauli_sum.v1":
        raise TypeError("native observable literal has an invalid kind")
    raw_terms = value["terms"]
    if not isinstance(raw_terms, tuple):
        raise TypeError("native observable literal has invalid terms")
    terms: list[tuple[PauliWord, complex]] = []
    for raw_term in raw_terms:
        if not isinstance(raw_term, tuple) or len(raw_term) != 2:
            raise TypeError("native observable literal has an invalid term")
        coefficient, raw_factors = raw_term
        if not isinstance(coefficient, complex) or not isinstance(raw_factors, tuple):
            raise TypeError("native observable literal has an invalid term payload")
        factors: list[PauliFactor] = []
        for raw_factor in raw_factors:
            if not isinstance(raw_factor, tuple) or len(raw_factor) != 2:
                raise TypeError("native observable literal has an invalid factor")
            raw_pauli, raw_index = raw_factor
            if not isinstance(raw_pauli, str) or (
                isinstance(raw_index, bool) or not isinstance(raw_index, int)
            ):
                raise TypeError(
                    "native observable literal has an invalid factor payload"
                )
            factors.append(PauliFactor(raw_index, Pauli(raw_pauli)))
        terms.append((PauliWord(tuple(factors)), coefficient))
    constant = value["constant"]
    num_qubits = value["num_qubits"]
    if not isinstance(constant, complex) or (
        isinstance(num_qubits, bool) or not isinstance(num_qubits, int)
    ):
        raise TypeError("native observable literal has invalid scalar fields")
    return tuple(terms), constant, num_qubits


__all__ = [
    "Hamiltonian",
    "Pauli",
    "PauliFactor",
    "PauliWord",
    "X",
    "Y",
    "Z",
    "commutator",
    "zero_observable",
]
