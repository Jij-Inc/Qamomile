"""msgpack wire format for the IR serialization pipeline.

msgpack carries ``bytes`` natively (``bin`` type), so the numpy
wrapper's ``data`` field passes through unchanged. This module is
mostly a thin wrapper around
:func:`qamomile.circuit.ir.serialize.encode.to_dict` /
:func:`qamomile.circuit.ir.serialize.decode.from_dict`, with safe
unpack defaults.
"""

from __future__ import annotations

import msgpack

from qamomile.circuit.ir.block import Block

from .decode import from_dict
from .encode import to_dict


def dump_msgpack(block: Block) -> bytes:
    """Encode ``block`` into a msgpack byte string.

    Args:
        block (Block): The block to encode. Must be AFFINE or
            ANALYZED.

    Returns:
        bytes: A msgpack-encoded payload. Compatible with the standard
            ``msgpack`` package; no extension types are used.

    Raises:
        ValueError: Forwarded from
            :func:`qamomile.circuit.ir.serialize.encode.to_dict`
            (e.g., unsupported BlockKind).
        TypeError: Forwarded from the encoder when a payload type is
            not serializable.
    """
    payload = to_dict(block)
    encoded = msgpack.packb(payload, use_bin_type=True)
    if encoded is None:
        raise RuntimeError(
            "msgpack.packb unexpectedly returned None; this should not happen "
            "for a non-streaming packer."
        )
    return encoded


def load_msgpack(payload: bytes) -> Block:
    """Decode a msgpack payload back into a ``Block``.

    The unpacker is configured with ``raw=False`` so strings come back
    as Python ``str``, and ``strict_map_key=False`` so non-string map
    keys (numeric, etc.) are accepted (qamomile's schema only uses
    string keys today, but this keeps the loader robust to future
    schema extensions).

    Args:
        payload (bytes): A msgpack-encoded payload produced by
            :func:`dump_msgpack` or an equivalent encoder.

    Returns:
        Block: The reconstructed Block.

    Raises:
        ValueError: Forwarded from the dict decoder for malformed
            envelopes or unknown ``$type`` tags.
    """
    raw = msgpack.unpackb(payload, raw=False, strict_map_key=False)
    if not isinstance(raw, dict):
        raise ValueError(
            f"msgpack payload did not decode to a dict envelope; got "
            f"{type(raw).__name__}"
        )
    return from_dict(raw)
