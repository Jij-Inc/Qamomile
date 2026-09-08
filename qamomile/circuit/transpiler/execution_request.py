"""Describe backend-neutral quantum execution requests."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Generic, TypeAlias, TypeVar

from qamomile.circuit.transpiler.parameter_binding import ParameterMetadata

if TYPE_CHECKING:
    import qamomile.observable as qm_o

CircuitT = TypeVar("CircuitT")


@dataclasses.dataclass(frozen=True)
class Exact:
    """Request an analytic expectation value without shot noise."""


@dataclasses.dataclass(frozen=True)
class ShotBased:
    """Request a shot-based expectation value.

    Args:
        shots (int): Positive number of measurement shots.

    Raises:
        ValueError: If ``shots`` is not positive.
    """

    shots: int

    def __post_init__(self) -> None:
        """Validate the requested shot count.

        Raises:
            ValueError: If ``shots`` is not positive.
        """
        if isinstance(self.shots, bool) or self.shots <= 0:
            raise ValueError("ShotBased.shots must be positive")
        object.__setattr__(self, "shots", int(self.shots))


@dataclasses.dataclass(frozen=True)
class TargetPrecision:
    """Request an expectation value at a provider target precision.

    Args:
        precision (float): Positive absolute target precision.

    Raises:
        ValueError: If ``precision`` is not positive.
    """

    precision: float

    def __post_init__(self) -> None:
        """Validate the requested precision.

        Raises:
            ValueError: If ``precision`` is not positive.
        """
        if (
            isinstance(self.precision, bool)
            or not math.isfinite(self.precision)
            or self.precision <= 0
        ):
            raise ValueError("TargetPrecision.precision must be positive")
        object.__setattr__(self, "precision", float(self.precision))


EstimationAccuracy: TypeAlias = Exact | ShotBased | TargetPrecision


@dataclasses.dataclass(frozen=True)
class CircuitInvocation(Generic[CircuitT]):
    """Keep an emitted circuit and runtime parameter values together.

    Backends may bind the values into a new circuit or submit them through a
    native parameter-input API. Keeping both forms available preserves native
    parameter sweeps and provider-side compilation caches.

    Args:
        circuit (CircuitT): Emitted backend circuit or kernel artifact.
        bindings (Mapping[str, Any]): Flattened Qamomile runtime bindings.
        parameter_metadata (ParameterMetadata): Mapping from public parameter
            names to backend parameter objects.
    """

    circuit: CircuitT
    bindings: Mapping[str, Any]
    parameter_metadata: ParameterMetadata

    def __post_init__(self) -> None:
        """Detach runtime bindings from caller-owned mutable mappings."""
        object.__setattr__(self, "bindings", dict(self.bindings))


@dataclasses.dataclass(frozen=True)
class SampleRequest(Generic[CircuitT]):
    """Describe one sampling execution.

    Args:
        invocation (CircuitInvocation[CircuitT]): Circuit and runtime inputs.
        shots (int): Number of requested samples.

    Raises:
        ValueError: If ``shots`` is not positive.
    """

    invocation: CircuitInvocation[CircuitT]
    shots: int

    def __post_init__(self) -> None:
        """Validate the requested sampling shot count.

        Raises:
            ValueError: If ``shots`` is not positive.
        """
        if isinstance(self.shots, bool) or self.shots <= 0:
            raise ValueError("SampleRequest.shots must be positive")
        object.__setattr__(self, "shots", int(self.shots))


@dataclasses.dataclass(frozen=True)
class EstimateRequest(Generic[CircuitT]):
    """Describe one Hamiltonian expectation execution.

    Args:
        invocation (CircuitInvocation[CircuitT]): Circuit and runtime inputs.
        hamiltonian (qm_o.Hamiltonian): Observable to evaluate.
        accuracy (EstimationAccuracy | None): Explicit accuracy policy.
            ``None`` uses the executor's configured default.
    """

    invocation: CircuitInvocation[CircuitT]
    hamiltonian: qm_o.Hamiltonian
    accuracy: EstimationAccuracy | None = None


__all__ = [
    "CircuitInvocation",
    "EstimateRequest",
    "EstimationAccuracy",
    "Exact",
    "SampleRequest",
    "ShotBased",
    "TargetPrecision",
]
