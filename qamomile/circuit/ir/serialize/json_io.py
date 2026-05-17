"""JSON wire format for the IR serialization pipeline.

JSON has no native ``bytes`` type, so this module is responsible for
converting raw bytes (used by numpy wrapper ``data`` fields) to / from
base64 strings at the wire boundary. Apart from that, the JSON output
is a faithful rendering of the intermediate dict produced by
:mod:`qamomile.circuit.ir.serialize.encode`.

A small sentinel field (``$bytes_b64``) distinguishes a base64-string
representation of bytes from a plain JSON string, so the decoder can
restore the original ``bytes`` value losslessly.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from qamomile.circuit.ir.block import Block

from .decode import from_dict
from .encode import to_dict

_BYTES_TAG = "$bytes_b64"


def dump_json(block: Block) -> bytes:
    """Encode ``block`` into a UTF-8 JSON byte string.

    The serialized form contains the ``schema_version`` envelope from
    :func:`qamomile.circuit.ir.serialize.encode.to_dict` plus
    base64-wrapped bytes payloads (numpy array data, raw ``bytes``
    bound values) so the output stays JSON-text-safe.

    Args:
        block (Block): The block to encode. Must be AFFINE or
            ANALYZED.

    Returns:
        bytes: A UTF-8 encoded JSON document.

    Raises:
        ValueError: Forwarded from
            :func:`qamomile.circuit.ir.serialize.encode.to_dict`
            (e.g., unsupported BlockKind).
        TypeError: Forwarded from the encoder when a payload type is
            not serializable.
    """
    return json.dumps(_bytes_to_b64(to_dict(block))).encode("utf-8")


def load_json(payload: bytes | str) -> Block:
    """Decode a JSON document back into a ``Block``.

    Args:
        payload (bytes | str): JSON bytes (UTF-8) or a JSON string.

    Returns:
        Block: The reconstructed Block.

    Raises:
        ValueError: Forwarded from the dict decoder for malformed
            envelopes or unknown ``$type`` tags.
        TypeError: For numeric overflow or other JSON-side conversion
            issues.
    """
    if isinstance(payload, (bytes, bytearray)):
        text = payload.decode("utf-8")
    else:
        text = payload
    raw = json.loads(text)
    return from_dict(_b64_to_bytes(raw))


def _bytes_to_b64(obj: Any) -> Any:
    """Recursively convert ``bytes`` payloads into base64-wrapped dicts.

    Args:
        obj (Any): Any JSON-compatible Python object that may contain
            raw ``bytes`` somewhere in its structure.

    Returns:
        Any: A copy of ``obj`` with every ``bytes`` value replaced by
            ``{"$bytes_b64": "<base64 str>"}``.
    """
    if isinstance(obj, (bytes, bytearray)):
        return {_BYTES_TAG: base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, dict):
        return {k: _bytes_to_b64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_bytes_to_b64(x) for x in obj]
    if isinstance(obj, tuple):
        return [_bytes_to_b64(x) for x in obj]
    return obj


def _b64_to_bytes(obj: Any) -> Any:
    """Reverse :func:`_bytes_to_b64`.

    Walks the dict structure and replaces ``$bytes_b64`` wrappers
    with the decoded ``bytes`` object.

    Args:
        obj (Any): The JSON-loaded Python structure.

    Returns:
        Any: A copy with ``$bytes_b64`` wrappers expanded into bytes.

    Raises:
        ValueError: If a ``$bytes_b64`` wrapper has the wrong shape
            (non-string value) or contains invalid base64.
    """
    if isinstance(obj, dict):
        if _BYTES_TAG in obj and len(obj) == 1:
            raw = obj[_BYTES_TAG]
            if not isinstance(raw, str):
                raise ValueError(
                    f"$bytes_b64 wrapper value must be a string, got "
                    f"{type(raw).__name__}"
                )
            return base64.b64decode(raw, validate=True)
        return {k: _b64_to_bytes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_b64_to_bytes(x) for x in obj]
    return obj
