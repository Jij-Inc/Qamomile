"""Define resource-estimator configuration and unknown-call policy."""

from __future__ import annotations

import dataclasses
import enum

from qamomile.circuit.estimator._resource_base import ControlDecomposition

_DEFAULT_CONTROL_DECOMPOSITION = ControlDecomposition.CLEAN_ANCILLA_TOFFOLI


class UnknownResourcePolicy(enum.StrEnum):
    """Control how the estimator handles bodyless unknown callables.

    Values:
        ERROR: Raise when a callable has neither a body nor an opaque cost.
        OPAQUE_CALL: Count one opaque call/query and continue.
        ZERO_WITH_WARNING: Record an assumption and continue with zero cost.
    """

    ERROR = "error"
    OPAQUE_CALL = "opaque_call"
    ZERO_WITH_WARNING = "zero_with_warning"


@dataclasses.dataclass
class _ResourceEstimatorConfig:
    """Configure ``ResourceEstimator`` behavior.

    Args:
        strategies (dict[str, str]): Strategy overrides by callable name.
        trace (bool): Whether estimates should carry trace nodes.
        simplify (bool): Whether to simplify the final estimate, including
            simplification over valid qkernel input conditions.
        unknown_policy (UnknownResourcePolicy): Handling for unknown opaque
            callables.
        control_decomposition (ControlDecomposition): Coherent-control
            decomposition. Defaults to ``CLEAN_ANCILLA_TOFFOLI``.
    """

    strategies: dict[str, str] = dataclasses.field(default_factory=dict)
    trace: bool = False
    simplify: bool = True
    unknown_policy: UnknownResourcePolicy = UnknownResourcePolicy.ERROR
    control_decomposition: ControlDecomposition = _DEFAULT_CONTROL_DECOMPOSITION
