"""Immutable HUGR target profile shared by compilation and execution."""

from qamomile.effects import Requirement

FORMAT = "hugr.package.v1"
CAPABILITIES = frozenset(
    {
        Requirement.CLASSICAL_COMPUTE,
        Requirement.COHERENT_CONTROL,
        Requirement.QUANTUM,
        Requirement.MEASUREMENT,
        Requirement.MID_CIRCUIT_MEASUREMENT,
        Requirement.DYNAMIC_BRANCH,
        Requirement.DYNAMIC_LOOP,
        Requirement.DISCARD,
        Requirement.RESET,
        Requirement.ROTATION,
    }
)
