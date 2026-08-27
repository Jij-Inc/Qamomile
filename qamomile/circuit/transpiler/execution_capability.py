"""Describe backend execution features without exposing provider SDK types."""

from __future__ import annotations

import dataclasses

from qamomile.circuit.transpiler.execution_request import (
    Exact,
    ShotBased,
    TargetPrecision,
)

EstimationPolicyType = type[Exact] | type[ShotBased] | type[TargetPrecision]
_ESTIMATION_POLICIES = frozenset({Exact, ShotBased, TargetPrecision})


@dataclasses.dataclass(frozen=True)
class ExecutionCapabilities:
    """Declare the execution features implemented by one executor.

    Args:
        supports_async_sampling (bool): Whether sampling submission returns
            before provider execution completes. Defaults to ``False``.
        supports_async_estimation (bool): Whether expectation submission
            returns before provider execution completes. Defaults to ``False``.
        supports_estimation (bool): Whether expectation-value execution is
            implemented. Defaults to ``False``.
        supports_cancellation (bool): Whether provider-backed handles can
            request cancellation. Defaults to ``False``.
        supports_restoration (bool): Whether execution references can recreate
            provider-backed handles. Defaults to ``False``.
        supports_native_batch (bool): Whether multiple logical requests can be
            submitted through one provider-native batch or job. Defaults to
            ``False``.
        supports_native_parameter_inputs (bool): Whether runtime values remain
            separate from emitted circuits during provider submission.
            Defaults to ``False``.
        estimation_accuracy (frozenset[EstimationPolicyType]): Explicit
            per-request accuracy policies accepted by the executor. An empty
            set means only executor-configured estimation behavior is
            available.

    Raises:
        ValueError: If an unknown estimation policy type is declared or
            estimation features are declared without estimation support.
    """

    supports_async_sampling: bool = False
    supports_async_estimation: bool = False
    supports_estimation: bool = False
    supports_cancellation: bool = False
    supports_restoration: bool = False
    supports_native_batch: bool = False
    supports_native_parameter_inputs: bool = False
    estimation_accuracy: frozenset[EstimationPolicyType] = dataclasses.field(
        default_factory=frozenset
    )

    def __post_init__(self) -> None:
        """Validate internally consistent capability declarations.

        Raises:
            ValueError: If estimation declarations are inconsistent or contain
                an unsupported policy type.
        """
        policies = frozenset(self.estimation_accuracy)
        unknown = policies - _ESTIMATION_POLICIES
        if unknown:
            names = sorted(
                getattr(policy, "__name__", repr(policy)) for policy in unknown
            )
            raise ValueError(f"Unknown estimation accuracy policies: {names}")
        if (
            self.supports_async_estimation or policies
        ) and not self.supports_estimation:
            raise ValueError("Estimation capabilities require supports_estimation=True")
        object.__setattr__(self, "estimation_accuracy", policies)


__all__ = ["EstimationPolicyType", "ExecutionCapabilities"]
