"""Adapt resource-estimation costs at the qkernel serialization boundary."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.estimator.wire import (
    ResourceEstimateWireDecoder,
    ResourceEstimateWireEncoder,
)


class OpaqueCostEncoder:
    """Encode every opaque cost in one qkernel with shared symbol identity."""

    def __init__(self) -> None:
        """Initialize one payload-wide resource-estimate encoder."""
        self._encoder = ResourceEstimateWireEncoder()

    def __call__(self, cost: Any) -> dict[str, Any]:
        """Encode one fixed opaque callable cost.

        Args:
            cost (Any): Cost attached to a bodyless callable definition.

        Returns:
            dict[str, Any]: Serializer-friendly fixed resource-estimate
                payload.

        Raises:
            TypeError: If ``cost`` is a process-local callback or is not a
                fixed ``ResourceEstimate``.
            ValueError: If the fixed estimate exceeds serialization limits.
            RuntimeError: If public resource metrics or metadata disagree with
                retained canonical provenance.
        """
        return encode_opaque_cost(cost, encoder=self._encoder)


class OpaqueCostDecoder:
    """Decode every opaque cost in one qkernel with shared symbol identity."""

    def __init__(self) -> None:
        """Initialize one payload-wide resource-estimate decoder."""
        self._decoder = ResourceEstimateWireDecoder()

    def __call__(self, payload: Any) -> Any:
        """Decode one fixed opaque callable cost.

        Args:
            payload (Any): Serializer-friendly resource-estimate payload.

        Returns:
            Any: Reconstructed fixed ``ResourceEstimate``.

        Raises:
            ValueError: If the fixed resource-estimate payload is malformed.
        """
        return decode_opaque_cost(payload, decoder=self._decoder)


def encode_opaque_cost(
    cost: Any,
    *,
    encoder: ResourceEstimateWireEncoder | None = None,
) -> dict[str, Any]:
    """Encode one fixed opaque callable cost for semantic serialization.

    Args:
        cost (Any): Cost attached to a bodyless callable definition.
        encoder (ResourceEstimateWireEncoder | None): Optional payload-wide
            stateful encoder. Defaults to ``None``.

    Returns:
        dict[str, Any]: Serializer-friendly fixed resource-estimate payload.

    Raises:
        TypeError: If ``cost`` is a process-local callback or is not a fixed
            ``ResourceEstimate``.
        ValueError: If the fixed estimate exceeds serialization limits.
        RuntimeError: If public resource metrics or metadata disagree with
            retained canonical provenance.
    """
    if callable(cost):
        raise TypeError(
            "CallableDef.opaque_cost callback objects cannot be serialized. "
            "Replace the callback with a fixed ResourceEstimate before "
            "serializing the qkernel."
        )
    active_encoder = ResourceEstimateWireEncoder() if encoder is None else encoder
    return active_encoder(cost)


def decode_opaque_cost(
    payload: Any,
    *,
    decoder: ResourceEstimateWireDecoder | None = None,
) -> Any:
    """Decode one fixed opaque callable cost at the serialization boundary.

    Args:
        payload (Any): Serializer-friendly resource-estimate payload.
        decoder (ResourceEstimateWireDecoder | None): Optional payload-wide
            stateful decoder. Defaults to ``None``.

    Returns:
        Any: Reconstructed fixed ``ResourceEstimate``.

    Raises:
        ValueError: If the fixed resource-estimate payload is malformed.
    """
    active_decoder = ResourceEstimateWireDecoder() if decoder is None else decoder
    return active_decoder(payload)
