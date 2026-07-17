"""Serialize and deserialize one static qkernel with protobuf."""

from __future__ import annotations

from google.protobuf.message import DecodeError

from qamomile.circuit.frontend.qkernel_like import QKernelLike
from qamomile.circuit.serialization.graph_protobuf import (
    graph_dict_from_qkernel,
    qkernel_from_graph_dict,
)

from .decode import from_dict as _from_dict
from .encode import to_dict as _to_dict
from .kernel import SerializedQKernel
from .proto import qamomile_ir_pb2 as pb


def serialize(kernel: QKernelLike) -> bytes:
    """Serialize one unbound qkernel to canonical protobuf bytes.

    Independently traced equivalent qkernels produce identical bytes because
    process-local value identities are normalized before protobuf encoding.

    Args:
        kernel (QKernelLike): Static qkernel-like object to serialize.

    Returns:
        bytes: Deterministic protobuf encoding.

    Raises:
        TypeError: If the qkernel contains an unsupported type or payload.
        ValueError: If the qkernel is bound, lowered, or malformed.
    """
    try:
        return _to_proto(kernel).SerializeToString(deterministic=True)
    except (DecodeError, RecursionError) as exc:
        raise ValueError(
            "qkernel structure exceeds the supported protobuf nesting depth"
        ) from exc


def deserialize(payload: bytes) -> SerializedQKernel:
    """Deserialize protobuf bytes into a static qkernel-like object.

    Args:
        payload (bytes): Bytes produced by :func:`serialize`.

    Returns:
        SerializedQKernel: Reconstructed qkernel accepted by transpilers.

    Raises:
        TypeError: If ``payload`` is not bytes-like.
        ValueError: If parsing, version validation, or reconstruction fails.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"deserialize() expected bytes, got {type(payload).__name__}")
    raw_payload = bytes(payload)
    message = pb.QKernel()
    try:
        message.ParseFromString(raw_payload)
    except DecodeError as exc:
        raise ValueError("payload is not a valid Qamomile qkernel protobuf") from exc
    try:
        restored = _from_proto(message)
        if serialize(restored) != raw_payload:
            raise ValueError("protobuf qkernel bytes are not in canonical form")
    except ValueError:
        raise
    except (TypeError, RuntimeError, KeyError, IndexError, AssertionError) as exc:
        raise ValueError("protobuf qkernel payload is malformed") from exc
    return restored


def _to_proto(kernel: QKernelLike) -> pb.QKernel:
    """Convert one unbound qkernel into its generated protobuf message.

    Args:
        kernel (QKernelLike): Static qkernel-like object to convert.

    Returns:
        pb.QKernel: Generated qkernel message.

    Raises:
        TypeError: If a frontend type or payload is unsupported.
        ValueError: If the qkernel is bound, lowered, or malformed.
    """
    return qkernel_from_graph_dict(_to_dict(kernel))


def _from_proto(message: pb.QKernel) -> SerializedQKernel:
    """Convert a generated protobuf message into a static qkernel.

    Args:
        message (pb.QKernel): Generated qkernel message.

    Returns:
        SerializedQKernel: Reconstructed qkernel-like object.

    Raises:
        TypeError: If ``message`` is not a qkernel protobuf message.
        ValueError: If version, graph, or interface data is malformed.
    """
    if not isinstance(message, pb.QKernel):
        raise TypeError(f"_from_proto() expected QKernel, got {type(message).__name__}")
    restored = _from_dict(graph_dict_from_qkernel(message))
    canonical = _to_proto(restored)
    if canonical.SerializeToString(deterministic=True) != message.SerializeToString(
        deterministic=True
    ):
        raise ValueError("protobuf qkernel is not in canonical form")
    return restored
