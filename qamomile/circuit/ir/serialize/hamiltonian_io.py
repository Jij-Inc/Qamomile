"""Hamiltonian wrapper used by both JSON and msgpack pipelines.

``ParamSlot.bound_value`` and ``ArrayRuntimeMetadata.const_array`` may
carry :class:`qamomile.observable.Hamiltonian` objects — the bound
values of ``Observable`` kernel parameters (e.g. the documented Trotter
pattern ``kernel.build(Hs=[1.2 * Z(0), 0.8 * X(0)], ...)``). This
module defines the tagged-dict wire representation for those payloads:
a sum of Pauli products plus a constant offset and the declared qubit
register width.

The wrapper is JSON-native (numbers, strings, lists, dicts — no raw
bytes), so both wire formats carry it unchanged.

Two fidelity properties are load-bearing:

- **Term order is preserved.** ``Hamiltonian``'s term dict iteration
  order is observable through ``repr`` and therefore feeds
  ``content_hash`` (canonical bytes stringify opaque payloads via
  ``repr``). The encoder emits terms in iteration order and the
  decoder re-adds them in wire order.
- **Coefficient types are preserved.** ``repr(1.2)`` differs from
  ``repr((1.2+0j))``, so int / float coefficients are written as plain
  numbers while complex ones get an explicit ``$complex`` sub-wrapper.

Security: decoding never resolves classes dynamically. Pauli names are
mapped through an explicit allow-map, and only ``Pauli`` /
``PauliOperator`` / ``Hamiltonian`` instances are constructed.
"""

from __future__ import annotations

from typing import Any

from qamomile.observable.hamiltonian import Hamiltonian, Pauli, PauliOperator

_HAMILTONIAN_TAG = "$hamiltonian"
_COMPLEX_TAG = "$complex"

# Closed allow-map from wire Pauli names to enum members. The decoder
# never does enum lookup by attribute access on user data.
_PAULI_BY_NAME: dict[str, Pauli] = {
    "I": Pauli.I,
    "X": Pauli.X,
    "Y": Pauli.Y,
    "Z": Pauli.Z,
}


def is_hamiltonian_wrapper(d: Any) -> bool:
    """Return True if ``d`` is a Hamiltonian wrapper dict.

    Args:
        d (Any): A value to check. Typically the result of a recursive
            dict walk from ``decode``.

    Returns:
        bool: ``True`` when ``d`` is a dict carrying the
            ``$hamiltonian`` tag with a ``True`` value.
    """
    return isinstance(d, dict) and d.get(_HAMILTONIAN_TAG) is True


def hamiltonian_to_dict(h: Hamiltonian) -> dict[str, Any]:
    """Encode a ``Hamiltonian`` into the wrapper dict.

    Terms are emitted in the Hamiltonian's own term-dict iteration
    order; each term is a ``[operators, coefficient]`` pair where
    ``operators`` is a list of ``[pauli_name, qubit_index]`` entries.

    Args:
        h (Hamiltonian): The Hamiltonian to encode. Term coefficients
            and the constant must be int, float, or complex.

    Returns:
        dict[str, Any]: A wrapper dict with ``$hamiltonian``,
            ``terms``, ``constant``, and ``num_qubits`` (the declared
            register width passed to the constructor, or ``None``).

    Raises:
        TypeError: If ``h`` is not a ``Hamiltonian``, or if a
            coefficient / the constant is not int, float, or complex.
    """
    if not isinstance(h, Hamiltonian):
        raise TypeError(
            f"hamiltonian_to_dict() expected Hamiltonian, got {type(h).__name__}"
        )
    terms: list[list[Any]] = []
    for operators, coeff in h.terms.items():
        ops_wire = [[op.pauli.name, int(op.index)] for op in operators]
        terms.append([ops_wire, _coeff_to_wire(coeff)])
    return {
        _HAMILTONIAN_TAG: True,
        "terms": terms,
        "constant": _coeff_to_wire(h.constant),
        # The DECLARED register width, not the ``num_qubits`` property:
        # the property merges the declared width with the term-derived
        # width, so persisting it would freeze the derived part and
        # change ``remap_qubits`` / ``num_qubits`` behavior after decode.
        "num_qubits": h._num_qubits,
    }


def dict_to_hamiltonian(d: dict[str, Any]) -> Hamiltonian:
    """Decode a wrapper dict back into a ``Hamiltonian``.

    Terms are re-added in wire order through the public ``add_term``
    API, which is the identity on the canonical term form the encoder
    emits (operators sorted per term, no identities) and preserves the
    term-dict insertion order.

    Args:
        d (dict[str, Any]): A wrapper dict previously produced by
            :func:`hamiltonian_to_dict` (possibly after a JSON /
            msgpack round-trip).

    Returns:
        Hamiltonian: The reconstructed Hamiltonian, equal to the
            original (same terms in the same order, same coefficient
            types, same constant, same declared register width).

    Raises:
        ValueError: If ``d`` is not a valid wrapper dict — missing or
            malformed ``terms``, a Pauli name outside the allow-map, a
            non-int qubit index, a malformed coefficient, or a
            ``num_qubits`` that is neither ``None`` nor an int.
    """
    if not is_hamiltonian_wrapper(d):
        raise ValueError("dict_to_hamiltonian() called with a non-wrapper dict")
    num_qubits = d.get("num_qubits")
    if num_qubits is not None and not _is_plain_int(num_qubits):
        raise ValueError(
            f"Hamiltonian wrapper 'num_qubits' must be an int or None, got "
            f"{type(num_qubits).__name__}"
        )
    raw_terms = d.get("terms")
    if not isinstance(raw_terms, list):
        raise ValueError("Hamiltonian wrapper 'terms' must be a list")

    h = Hamiltonian(num_qubits=num_qubits)
    h.constant = _coeff_from_wire(d.get("constant"))
    for entry in raw_terms:
        if not isinstance(entry, list) or len(entry) != 2:
            raise ValueError(
                "Hamiltonian wrapper term entries must be "
                "[operators, coefficient] pairs"
            )
        raw_ops, raw_coeff = entry
        if not isinstance(raw_ops, list):
            raise ValueError("Hamiltonian wrapper term operators must be a list")
        operators = tuple(_operator_from_wire(raw_op) for raw_op in raw_ops)
        h.add_term(operators, _coeff_from_wire(raw_coeff))
    return h


def _operator_from_wire(raw_op: Any) -> PauliOperator:
    """Decode one ``[pauli_name, qubit_index]`` wire entry.

    Args:
        raw_op (Any): The wire entry. Must be a two-element list whose
            first element is a Pauli name in the allow-map and whose
            second element is an int qubit index.

    Returns:
        PauliOperator: The reconstructed operator.

    Raises:
        ValueError: If the entry shape, Pauli name, or index type is
            invalid.
    """
    if not isinstance(raw_op, list) or len(raw_op) != 2:
        raise ValueError(
            "Hamiltonian wrapper operators must be [pauli_name, index] pairs"
        )
    name, index = raw_op
    pauli = _PAULI_BY_NAME.get(name) if isinstance(name, str) else None
    if pauli is None:
        raise ValueError(
            f"Pauli name {name!r} is not in the serialization allow-map "
            f"{sorted(_PAULI_BY_NAME)}"
        )
    if not _is_plain_int(index):
        raise ValueError(
            f"Pauli operator qubit index must be an int, got {type(index).__name__}"
        )
    return PauliOperator(pauli, index)


def _coeff_to_wire(coeff: Any) -> Any:
    """Encode a term coefficient (or the constant) for the wire.

    Args:
        coeff (Any): The coefficient. Must be int, float, or complex
            (bool is rejected even though it subclasses int).

    Returns:
        Any: A plain int / float for real coefficients, or
            ``{"$complex": True, "real": <float>, "imag": <float>}``
            for complex ones, so the float-vs-complex distinction
            survives the round-trip.

    Raises:
        TypeError: If ``coeff`` is not int, float, or complex.
    """
    if isinstance(coeff, bool):
        raise TypeError("Hamiltonian coefficients must not be bool")
    if isinstance(coeff, int):
        return int(coeff)
    if isinstance(coeff, float):
        return float(coeff)
    if isinstance(coeff, complex):
        return {
            _COMPLEX_TAG: True,
            "real": float(coeff.real),
            "imag": float(coeff.imag),
        }
    raise TypeError(
        f"Cannot encode Hamiltonian coefficient of type {type(coeff).__name__!r}; "
        f"supported types are int, float, complex."
    )


def _coeff_from_wire(value: Any) -> int | float | complex:
    """Decode a wire coefficient back into a Python number.

    Args:
        value (Any): A plain int / float, or a ``$complex`` wrapper
            dict with float ``real`` / ``imag`` fields.

    Returns:
        int | float | complex: The coefficient with its original
            numeric type.

    Raises:
        ValueError: If ``value`` is neither a plain number nor a
            well-formed ``$complex`` wrapper.
    """
    if isinstance(value, bool):
        raise ValueError("Hamiltonian wire coefficient must not be bool")
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict) and value.get(_COMPLEX_TAG) is True:
        real = value.get("real")
        imag = value.get("imag")
        if (
            isinstance(real, bool)
            or isinstance(imag, bool)
            or not isinstance(real, (int, float))
            or not isinstance(imag, (int, float))
        ):
            raise ValueError(
                f"$complex wrapper 'real' / 'imag' must be numbers, got "
                f"{type(real).__name__} / {type(imag).__name__}"
            )
        return complex(real, imag)
    raise ValueError(
        f"Cannot decode Hamiltonian coefficient from {type(value).__name__!r}; "
        f"expected a number or a $complex wrapper dict."
    )


def _is_plain_int(value: Any) -> bool:
    """Return True for ints that are not bools.

    Args:
        value (Any): The value to check.

    Returns:
        bool: ``True`` when ``value`` is an ``int`` and not a ``bool``.
    """
    return isinstance(value, int) and not isinstance(value, bool)
