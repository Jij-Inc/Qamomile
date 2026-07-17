"""Serialize factory-produced Pauli LCU block-encoding descriptors."""

from __future__ import annotations

from google.protobuf.message import DecodeError

from qamomile.circuit.frontend.qkernel_callable import qkernel_callable_attrs
from qamomile.circuit.frontend.qkernel_like import QKernelLike
from qamomile.circuit.stdlib.pauli_lcu_block_encoding import (
    PauliLCUBlockEncoding,
    pauli_lcu_block_encoding,
)
from qamomile.linalg import PauliLCU, PauliLCUTerm
from qamomile.observable import Pauli, PauliOperator

from .graph_protobuf import (
    _float_bits,
    _float_from_bits,
    _integer_from_proto,
    _integer_to_proto,
)
from .proto import qamomile_ir_pb2 as pb
from .protobuf import serialize
from .schema import QAMOMILE_VERSION

_ARTIFACT_KIND = "qamomile.pauli_lcu_block_encoding"
_PAULI_BY_NAME = {
    "X": Pauli.X,
    "Y": Pauli.Y,
    "Z": Pauli.Z,
}


def serialize_pauli_lcu_block_encoding(
    encoding: PauliLCUBlockEncoding,
) -> bytes:
    """Serialize a factory-produced Pauli LCU block encoding.

    The canonical payload stores only the retained complex Pauli recipe. The
    unitary body, normalization, and register widths are deterministic factory
    outputs and are regenerated after loading. Before emitting bytes, this
    function rebuilds the factory output and requires its complete qkernel and
    descriptor fields to match the supplied object. Hand-built descriptors
    without canonical factory metadata are therefore rejected.

    Args:
        encoding (PauliLCUBlockEncoding): Concrete descriptor returned by
            :func:`qamomile.circuit.pauli_lcu_block_encoding`.

    Returns:
        bytes: Canonical protobuf recipe accepted by
            :func:`deserialize_pauli_lcu_block_encoding`.

    Raises:
        TypeError: If ``encoding`` is not a ``PauliLCUBlockEncoding``.
        ValueError: If its metadata, descriptor fields, or unitary body does
            not match the canonical factory output.
    """
    if not isinstance(encoding, PauliLCUBlockEncoding):
        raise TypeError(
            "serialize_pauli_lcu_block_encoding() expected a "
            f"PauliLCUBlockEncoding, got {type(encoding).__name__}"
        )

    lcu = _pauli_lcu_from_unitary_metadata(encoding.unitary)
    expected = pauli_lcu_block_encoding(lcu)
    _require_matching_factory_output(encoding, expected)
    return _pauli_lcu_to_proto(lcu).SerializeToString(deterministic=True)


def deserialize_pauli_lcu_block_encoding(
    payload: bytes,
) -> PauliLCUBlockEncoding:
    """Restore a concrete Pauli LCU block encoding from protobuf bytes.

    Loading validates the typed retained-Pauli recipe and regenerates a fresh
    factory ``QKernel``. No serialized unitary body is trusted or executed.
    The restored descriptor therefore retains the normal call-time register
    checks as well as inverse, control, and nested SELECT behavior.

    Args:
        payload (bytes): Bytes produced by
            :func:`serialize_pauli_lcu_block_encoding`.

    Returns:
        PauliLCUBlockEncoding: Fresh factory-produced descriptor with all four
            public fields restored.

    Raises:
        TypeError: If ``payload`` is not bytes-like.
        ValueError: If parsing, version validation, canonicality validation,
            or Pauli LCU reconstruction fails.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(
            "deserialize_pauli_lcu_block_encoding() expected bytes, got "
            f"{type(payload).__name__}"
        )
    raw_payload = bytes(payload)
    message = pb.PauliLCUBlockEncodingArtifact()
    try:
        message.ParseFromString(raw_payload)
    except DecodeError as exc:
        raise ValueError(
            "payload is not a valid Pauli LCU block-encoding protobuf"
        ) from exc

    try:
        lcu = _pauli_lcu_from_proto(message)
        canonical = _pauli_lcu_to_proto(lcu).SerializeToString(deterministic=True)
        if canonical != raw_payload:
            raise ValueError(
                "Pauli LCU block-encoding protobuf bytes are not in canonical form"
            )
        return pauli_lcu_block_encoding(lcu)
    except (TypeError, OverflowError, RuntimeError, KeyError, IndexError) as exc:
        raise ValueError("Pauli LCU block-encoding protobuf is malformed") from exc


def _pauli_lcu_to_proto(lcu: PauliLCU) -> pb.PauliLCUBlockEncodingArtifact:
    """Encode one retained Pauli LCU as a typed protobuf recipe.

    Args:
        lcu (PauliLCU): Canonical retained decomposition.

    Returns:
        pb.PauliLCUBlockEncodingArtifact: Generated protobuf message.
    """
    message = pb.PauliLCUBlockEncodingArtifact(
        artifact_kind=_ARTIFACT_KIND,
        qamomile_version=QAMOMILE_VERSION,
    )
    message.num_qubits.CopyFrom(_integer_to_proto(lcu.num_qubits))
    for term in lcu.terms:
        term_message = message.terms.add()
        term_message.coefficient.real_bits = _float_bits(term.coefficient.real)
        term_message.coefficient.imag_bits = _float_bits(term.coefficient.imag)
        for operator in term.operators:
            operator_message = term_message.operators.add(pauli=operator.pauli.name)
            operator_message.index.CopyFrom(_integer_to_proto(int(operator.index)))
    return message


def _pauli_lcu_from_proto(message: pb.PauliLCUBlockEncodingArtifact) -> PauliLCU:
    """Decode and validate one typed retained-Pauli recipe.

    Args:
        message (pb.PauliLCUBlockEncodingArtifact): Parsed protobuf message.

    Returns:
        PauliLCU: Validated retained decomposition.

    Raises:
        TypeError: If ``message`` has the wrong generated message type.
        ValueError: If its kind, version, required fields, coefficients, or
            Pauli words are invalid.
    """
    if not isinstance(message, pb.PauliLCUBlockEncodingArtifact):
        raise TypeError(
            "_pauli_lcu_from_proto() expected PauliLCUBlockEncodingArtifact, "
            f"got {type(message).__name__}"
        )
    if message.artifact_kind != _ARTIFACT_KIND:
        raise ValueError(
            "artifact_kind mismatch: payload is not a Pauli LCU block encoding"
        )
    if message.qamomile_version != QAMOMILE_VERSION:
        raise ValueError(
            "qamomile_version mismatch: payload reports "
            f"{message.qamomile_version!r}, this loader supports "
            f"{QAMOMILE_VERSION!r}. Cross-version migration is not provided."
        )
    if not message.HasField("num_qubits"):
        raise ValueError("Pauli LCU block-encoding protobuf is missing num_qubits")
    num_qubits = _integer_from_proto(message.num_qubits)
    if num_qubits <= 0:
        raise ValueError("Pauli LCU block encoding requires positive num_qubits")

    terms: list[PauliLCUTerm] = []
    for term_index, term_message in enumerate(message.terms):
        if not term_message.HasField("coefficient"):
            raise ValueError(
                f"Pauli LCU block-encoding term {term_index} has no coefficient"
            )
        coefficient = complex(
            _float_from_bits(term_message.coefficient.real_bits),
            _float_from_bits(term_message.coefficient.imag_bits),
        )
        operators: list[PauliOperator] = []
        for operator_index, operator_message in enumerate(term_message.operators):
            pauli = _PAULI_BY_NAME.get(operator_message.pauli)
            if pauli is None:
                raise ValueError(
                    "Pauli LCU block-encoding operator "
                    f"{term_index}:{operator_index} has invalid Pauli "
                    f"{operator_message.pauli!r}"
                )
            if not operator_message.HasField("index"):
                raise ValueError(
                    "Pauli LCU block-encoding operator "
                    f"{term_index}:{operator_index} has no index"
                )
            operators.append(
                PauliOperator(pauli, _integer_from_proto(operator_message.index))
            )
        try:
            terms.append(PauliLCUTerm(coefficient, tuple(operators)))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Pauli LCU block-encoding term {term_index} is invalid"
            ) from exc
    try:
        return PauliLCU(num_qubits=num_qubits, terms=tuple(terms))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Pauli LCU block-encoding recipe is invalid") from exc


def _pauli_lcu_from_unitary_metadata(unitary: QKernelLike) -> PauliLCU:
    """Decode the canonical LCU metadata attached by the public factory.

    Args:
        unitary (QKernelLike): Candidate factory unitary.

    Returns:
        PauliLCU: Retained decomposition declared by the unitary.

    Raises:
        ValueError: If the semantic argument record is malformed.
    """
    value = qkernel_callable_attrs(unitary).get("semantic_arguments")
    try:
        if type(value) is not dict or set(value) != {
            "num_qubits",
            "signal_qubits",
            "terms",
        }:
            raise ValueError("semantic arguments have unexpected fields")
        num_qubits = value["num_qubits"]
        signal_qubits = value["signal_qubits"]
        raw_terms = value["terms"]
        if (
            type(num_qubits) is not int
            or type(signal_qubits) is not int
            or type(raw_terms) is not list
        ):
            raise ValueError("semantic argument field types are invalid")

        terms: list[PauliLCUTerm] = []
        for raw_term in raw_terms:
            if type(raw_term) is not dict or set(raw_term) != {
                "coefficient",
                "operators",
            }:
                raise ValueError("term metadata has unexpected fields")
            coefficient_parts = raw_term["coefficient"]
            raw_operators = raw_term["operators"]
            if (
                type(coefficient_parts) is not list
                or len(coefficient_parts) != 2
                or not all(type(part) is float for part in coefficient_parts)
                or type(raw_operators) is not list
            ):
                raise ValueError("term metadata field types are invalid")
            operators: list[PauliOperator] = []
            for raw_operator in raw_operators:
                if type(raw_operator) is not list or len(raw_operator) != 2:
                    raise ValueError("operator metadata must be a pair")
                pauli_name, index = raw_operator
                if (
                    type(pauli_name) is not str
                    or pauli_name not in _PAULI_BY_NAME
                    or type(index) is not int
                ):
                    raise ValueError("operator metadata is invalid")
                operators.append(PauliOperator(_PAULI_BY_NAME[pauli_name], index))
            terms.append(
                PauliLCUTerm(
                    complex(coefficient_parts[0], coefficient_parts[1]),
                    tuple(operators),
                )
            )
        lcu = PauliLCU(num_qubits=num_qubits, terms=tuple(terms))
        expected_signal = 1 if len(terms) <= 1 else (len(terms) - 1).bit_length()
        if signal_qubits != expected_signal:
            raise ValueError("signal width does not match the term count")
        return lcu
    except (TypeError, ValueError, OverflowError, KeyError, IndexError) as exc:
        raise ValueError(
            "unitary does not contain canonical Pauli LCU factory metadata"
        ) from exc


def _require_matching_factory_output(
    actual: PauliLCUBlockEncoding,
    expected: PauliLCUBlockEncoding,
) -> None:
    """Require descriptor metadata and unitary to equal factory output.

    Args:
        actual (PauliLCUBlockEncoding): Descriptor supplied for serialization.
        expected (PauliLCUBlockEncoding): Descriptor rebuilt from its metadata.

    Raises:
        ValueError: If the descriptor fields or complete unitary differ.
    """
    if (
        _float_bits(actual.normalization) != _float_bits(expected.normalization)
        or actual.num_signal_qubits != expected.num_signal_qubits
        or actual.num_system_qubits != expected.num_system_qubits
    ):
        raise ValueError(
            "Pauli LCU block-encoding descriptor fields do not match its unitary"
        )
    if serialize(actual.unitary) != serialize(expected.unitary):
        raise ValueError(
            "Pauli LCU block-encoding unitary does not match canonical factory output"
        )


__all__ = [
    "deserialize_pauli_lcu_block_encoding",
    "serialize_pauli_lcu_block_encoding",
]
