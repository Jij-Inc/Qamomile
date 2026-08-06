"""Adapt resource-estimation costs at the qkernel serialization boundary."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.estimator._wire import (
    resource_estimate_from_wire,
    resource_estimate_to_wire,
)
from qamomile.circuit.estimator._wire_expression import _WireExpressionDecoder


class OpaqueCostEncoder:
    """Encode every opaque cost in one qkernel with shared symbol identity."""

    def __init__(self) -> None:
        """Initialize an empty payload-wide Dummy slot registry."""
        self._dummy_slots: dict[Any, int] = {}

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
        """
        return encode_opaque_cost(cost, dummy_slots=self._dummy_slots)


class OpaqueCostDecoder:
    """Decode every opaque cost in one qkernel with shared symbol identity."""

    def __init__(self) -> None:
        """Initialize a lazily constructed payload-wide expression decoder."""
        self._decoder: _WireExpressionDecoder | None = None

    def __call__(self, payload: Any) -> Any:
        """Decode one fixed opaque callable cost.

        Args:
            payload (Any): Serializer-friendly resource-estimate payload.

        Returns:
            Any: Reconstructed fixed ``ResourceEstimate``.

        Raises:
            ValueError: If the fixed resource-estimate payload is malformed.
        """
        if self._decoder is None:
            self._decoder = _WireExpressionDecoder()
        return decode_opaque_cost(payload, decoder=self._decoder)


def encode_opaque_cost(
    cost: Any,
    *,
    dummy_slots: dict[Any, int] | None = None,
) -> dict[str, Any]:
    """Encode one fixed opaque callable cost for semantic serialization.

    Args:
        cost (Any): Cost attached to a bodyless callable definition.
        dummy_slots (dict[Any, int] | None): Optional payload-wide mapping
            from Dummy identities to deterministic slots. Defaults to
            ``None``.

    Returns:
        dict[str, Any]: Serializer-friendly fixed resource-estimate payload.

    Raises:
        TypeError: If ``cost`` is a process-local callback or is not a fixed
            ``ResourceEstimate``.
        ValueError: If the fixed estimate exceeds serialization limits.
    """
    if callable(cost):
        raise TypeError(
            "CallableDef.opaque_cost callback objects cannot be serialized. "
            "Replace the callback with a fixed ResourceEstimate before "
            "serializing the qkernel."
        )
    return resource_estimate_to_wire(cost, dummy_slots=dummy_slots)


def decode_opaque_cost(
    payload: Any,
    *,
    decoder: _WireExpressionDecoder | None = None,
) -> Any:
    """Decode one fixed opaque callable cost at the serialization boundary.

    Args:
        payload (Any): Serializer-friendly resource-estimate payload.
        decoder (_WireExpressionDecoder | None): Optional payload-wide
            expression decoder. Defaults to ``None``.

    Returns:
        Any: Reconstructed fixed ``ResourceEstimate``.

    Raises:
        ValueError: If the fixed resource-estimate payload is malformed.
    """
    return resource_estimate_from_wire(payload, decoder=decoder)
