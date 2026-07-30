"""Adapt resource-estimation costs at the qkernel serialization boundary."""

from __future__ import annotations

from typing import Any


def encode_opaque_cost(cost: Any) -> dict[str, Any]:
    """Encode one fixed opaque callable cost for semantic serialization.

    Args:
        cost (Any): Cost attached to a bodyless callable definition.

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
    from qamomile.circuit.estimator._wire import resource_estimate_to_wire

    return resource_estimate_to_wire(cost)


def decode_opaque_cost(payload: Any) -> Any:
    """Decode one fixed opaque callable cost at the serialization boundary.

    Args:
        payload (Any): Serializer-friendly resource-estimate payload.

    Returns:
        Any: Reconstructed fixed ``ResourceEstimate``.

    Raises:
        ValueError: If the fixed resource-estimate payload is malformed.
    """
    from qamomile.circuit.estimator._wire import resource_estimate_from_wire

    return resource_estimate_from_wire(payload)
