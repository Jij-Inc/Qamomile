"""Stable stateful codecs for fixed resource-estimate wire payloads."""

from __future__ import annotations

from typing import Any

import sympy as sp

from qamomile.circuit.estimator._estimate import ResourceEstimate
from qamomile.circuit.estimator._wire import (
    resource_estimate_from_wire,
    resource_estimate_to_wire,
)
from qamomile.circuit.estimator._wire_expression import _WireExpressionDecoder


class ResourceEstimateWireEncoder:
    """Encode one payload stream while preserving shared Dummy identities."""

    def __init__(self) -> None:
        """Initialize an empty payload-wide Dummy slot mapping."""
        self._dummy_slots: dict[sp.Dummy, int] = {}

    def __call__(self, estimate: ResourceEstimate) -> dict[str, Any]:
        """Encode one fixed resource estimate.

        Args:
            estimate (ResourceEstimate): Fixed resource estimate to encode.

        Returns:
            dict[str, Any]: Serializer-friendly resource-estimate payload.

        Raises:
            TypeError: If ``estimate`` is not a ``ResourceEstimate``.
            ValueError: If the estimate exceeds the supported wire contract.
        """
        return resource_estimate_to_wire(
            estimate,
            dummy_slots=self._dummy_slots,
        )


class ResourceEstimateWireDecoder:
    """Decode one payload stream while preserving shared Dummy identities."""

    def __init__(self) -> None:
        """Initialize one payload-wide symbolic-expression decoder."""
        self._expression_decoder = _WireExpressionDecoder()

    def __call__(self, payload: Any) -> ResourceEstimate:
        """Decode one fixed resource estimate.

        Args:
            payload (Any): Serializer-friendly resource-estimate payload.

        Returns:
            ResourceEstimate: Reconstructed fixed resource estimate.

        Raises:
            ValueError: If the resource-estimate payload is malformed.
        """
        return resource_estimate_from_wire(
            payload,
            decoder=self._expression_decoder,
        )


__all__ = [
    "ResourceEstimateWireDecoder",
    "ResourceEstimateWireEncoder",
]
