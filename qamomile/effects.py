"""Orthogonal semantic effects, callable kinds, and target requirements."""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Effect(enum.StrEnum):
    """Observable behavior retained in QCore."""

    QUANTUM = "quantum"
    ALLOCATION = "allocation"
    RELEASE = "release"
    MEASUREMENT = "measurement"
    DISCARD = "discard"
    RESET = "reset"
    SHOT_CONTROL = "shot_control"
    HOST_CONTROL = "host_control"
    PROBABILISTIC = "probabilistic"


class CallableKind(enum.StrEnum):
    """The denotational map implemented by a callable."""

    CLASSICAL = "classical"
    UNITARY = "unitary"
    ISOMETRY = "isometry"
    CHANNEL = "channel"
    INSTRUMENT = "instrument"


class Capability(enum.StrEnum):
    """Independent callable transformation proofs."""

    ADJOINTABLE = "adjointable"
    CONTROLLABLE = "controllable"


class Requirement(enum.StrEnum):
    """Target capabilities inferred from a closed program."""

    CLASSICAL_COMPUTE = "classical_compute"
    PARAMETERIZED_EXECUTION = "parameterized_execution"
    QUANTUM = "quantum"
    MEASUREMENT = "measurement"
    MID_CIRCUIT_MEASUREMENT = "mid_circuit_measurement"
    DYNAMIC_BRANCH = "dynamic_branch"
    DYNAMIC_LOOP = "dynamic_loop"
    DYNAMIC_LOOP_HISTORY = "dynamic_loop_history"
    COHERENT_CONTROL = "coherent_control"
    HOST_DISPATCH = "host_dispatch"
    RESET = "reset"
    DISCARD = "discard"
    ROTATION = "rotation"
    DYNAMIC_ROTATION = "dynamic_rotation"
    DYNAMIC_CLASSICAL_INDEX = "dynamic_classical_index"
    DYNAMIC_QUBIT_SELECTION = "dynamic_qubit_selection"
    EXTERNAL_CALL = "external_call"


@dataclass(frozen=True, slots=True)
class CallableContract:
    """The single derived semantic summary of a QCore callable.

    This value is a cache of facts reconstructed from the callable body and
    the closed call graph. It is never serialized and is never accepted as an
    assertion from a producer.
    """

    kind: CallableKind
    effects: frozenset[Effect]
    capabilities: frozenset[Capability]
    requirements: frozenset[Requirement]
