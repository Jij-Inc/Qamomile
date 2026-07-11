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
  order is observable through ``repr`` and ``__iter__``, so a faithful
  round-trip must not reorder terms. The encoder emits terms in
  iteration order and the decoder re-adds them in wire order.
  (``content_hash`` is deliberately order-*independent*: canonical
  bytes sort the term tokens, matching ``Hamiltonian.__eq__``'s
  dict-based term comparison — see ``canonical._hamiltonian_token``.)
- **Coefficient types are preserved.** ``repr(1.2)`` differs from
  ``repr((1.2+0j))``, so int / float coefficients are written as plain
  numbers while complex ones get an explicit ``$complex`` sub-wrapper.

Security: decoding never resolves classes dynamically. Pauli names are
mapped through an explicit allow-map, and only ``Pauli`` /
``PauliOperator`` / ``Hamiltonian`` instances are constructed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qamomile._utils import is_plain_int
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
            and the constant must be int, float, or complex, or a
            ``numpy`` scalar of one of those kinds (coerced via
            ``.item()``).

    Returns:
        dict[str, Any]: A wrapper dict with ``$hamiltonian``,
            ``terms``, ``constant``, and ``num_qubits`` (the declared
            register width passed to the constructor, or ``None``).

    Raises:
        TypeError: If ``h`` is not a ``Hamiltonian``, if a coefficient /
            the constant is not int, float, or complex, or if the declared
            ``num_qubits`` is not an int — all after coercing any ``numpy``
            scalar to its Python equivalent.
        ValueError: If the declared ``num_qubits`` is negative.
    """
    if not isinstance(h, Hamiltonian):
        raise TypeError(
            f"hamiltonian_to_dict() expected Hamiltonian, got {type(h).__name__}"
        )
    terms: list[list[Any]] = []
    for operators, coeff in h.terms.items():
        ops_wire = [_operator_to_wire(op) for op in operators]
        terms.append([ops_wire, _coeff_to_wire(coeff)])
    return {
        _HAMILTONIAN_TAG: True,
        "terms": terms,
        "constant": _coeff_to_wire(h.constant),
        # The DECLARED register width, not the ``num_qubits`` property:
        # the property merges the declared width with the term-derived
        # width, so persisting it would freeze the derived part and
        # change ``remap_qubits`` / ``num_qubits`` behavior after decode.
        "num_qubits": _num_qubits_to_wire(h._num_qubits),
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
            malformed ``terms``, a term with an empty operator list (the
            constant is carried by the dedicated ``constant`` field, so an
            empty list would double-encode it), a Pauli name outside the
            allow-map, a negative or non-int qubit index, a malformed
            coefficient, or a ``num_qubits`` that is neither ``None`` nor a
            non-negative int.
    """
    if not is_hamiltonian_wrapper(d):
        raise ValueError("dict_to_hamiltonian() called with a non-wrapper dict")
    num_qubits = _num_qubits_from_wire(d.get("num_qubits"))
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
        if not isinstance(raw_ops, list) or not raw_ops:
            raise ValueError(
                "Hamiltonian wrapper term operators must be a non-empty list; "
                "the constant term is carried by the dedicated 'constant' "
                "field, so an empty operator list would double-encode the "
                "constant via add_term"
            )
        operators = tuple(_operator_from_wire(raw_op) for raw_op in raw_ops)
        h.add_term(operators, _coeff_from_wire(raw_coeff))
    return h


def _operator_to_wire(op: PauliOperator) -> list[Any]:
    """Encode one ``PauliOperator`` into a ``[pauli_name, qubit_index]`` entry.

    The inverse of :func:`_operator_from_wire`.

    Args:
        op (PauliOperator): The operator to encode.

    Returns:
        list[Any]: A two-element list ``[pauli_name, qubit_index]`` whose
            first element is the ``Pauli`` member name (``"X"`` / ``"Y"`` /
            ``"Z"`` / ``"I"``) and second element is a plain Python int.
    """
    return [op.pauli.name, int(op.index)]


def _operator_from_wire(raw_op: Any) -> PauliOperator:
    """Decode one ``[pauli_name, qubit_index]`` wire entry.

    Args:
        raw_op (Any): The wire entry. Must be a two-element list whose
            first element is a Pauli name in the allow-map and whose
            second element is a non-negative int qubit index.

    Returns:
        PauliOperator: The reconstructed operator.

    Raises:
        ValueError: If the entry shape or Pauli name is invalid, or the
            qubit index is not a non-negative int.
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
    if not is_plain_int(index) or index < 0:
        raise ValueError(
            f"Pauli operator qubit index must be a non-negative int, got {index!r}"
        )
    return PauliOperator(pauli, index)


def _coeff_to_wire(coeff: Any) -> Any:
    """Encode a term coefficient (or the constant) for the wire.

    Args:
        coeff (Any): The coefficient. Must be int, float, or complex, or
            a ``numpy`` scalar of one of those kinds. ``bool`` (and
            ``numpy.bool_``) is rejected even though it subclasses int.

    Returns:
        Any: A plain int / float for real coefficients, or
            ``{"$complex": True, "real": <float>, "imag": <float>}``
            for complex ones, so the float-vs-complex distinction
            survives the round-trip.

    Raises:
        TypeError: If ``coeff`` is not int, float, or complex (after
            coercing any ``numpy`` scalar to its Python equivalent).
    """
    # numpy scalar coefficients (e.g. np.float64 from np.sqrt(...) used in
    # several Hamiltonian builders) behave like plain numbers but are not
    # instances of the Python int / float / complex checked below. Coerce
    # them to their Python equivalent up front so they serialize like any
    # other real / complex coefficient instead of being rejected; a
    # numpy.bool_ collapses to a Python bool and is still caught by the
    # bool guard.
    if isinstance(coeff, np.generic):
        coeff = coeff.item()
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


def _num_qubits_to_wire(num_qubits: Any) -> int | None:
    """Encode a Hamiltonian's declared register width for the wire.

    Args:
        num_qubits (Any): The declared width (``Hamiltonian._num_qubits``):
            ``None``, a Python ``int``, or a ``numpy`` integer scalar. A
            ``numpy`` scalar is coerced to its Python equivalent via
            ``.item()`` so the wrapper stays JSON / msgpack-serializable.

    Returns:
        int | None: ``None`` unchanged, or a non-negative Python ``int``.

    Raises:
        TypeError: If ``num_qubits`` is neither ``None`` nor an ``int``
            after coercing any ``numpy`` scalar (``bool`` is rejected even
            though it subclasses int).
        ValueError: If the coerced width is negative.
    """
    if num_qubits is None:
        return None
    if isinstance(num_qubits, np.generic):
        num_qubits = num_qubits.item()
    if not is_plain_int(num_qubits):
        raise TypeError(
            f"Hamiltonian declared num_qubits must be an int or None, got "
            f"{type(num_qubits).__name__}"
        )
    if num_qubits < 0:
        raise ValueError(
            f"Hamiltonian declared num_qubits must be non-negative, got {num_qubits!r}"
        )
    return num_qubits


def _num_qubits_from_wire(raw: Any) -> int | None:
    """Decode and validate a wrapper's declared register width.

    The inverse of :func:`_num_qubits_to_wire`. ``None`` (an undeclared
    width) is a legitimate value and passes through unchanged; any other
    value must be a non-negative, non-bool int.

    Args:
        raw (Any): The wire value of the ``num_qubits`` field — ``None`` or
            an int.

    Returns:
        int | None: ``None`` unchanged, or the validated non-negative int.

    Raises:
        ValueError: If ``raw`` is neither ``None`` nor a non-negative int
            (``bool`` is rejected even though it subclasses int).
    """
    if raw is not None and (not is_plain_int(raw) or raw < 0):
        raise ValueError(
            f"Hamiltonian wrapper 'num_qubits' must be a non-negative int or "
            f"None, got {raw!r}"
        )
    return raw
