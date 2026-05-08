"""Shared helpers for backend emission."""

from .composite_decomposer import CompositeDecomposer
from .condition_resolution import (
    map_phi_outputs,
    remap_static_phi_outputs,
    resolve_if_condition,
)
from .loop_analyzer import LoopAnalyzer
from .qubit_address import ClbitMap, QubitAddress, QubitMap
from .resource_allocator import ResourceAllocator
from .value_resolver import QubitResolutionResult, ValueResolver, resolve_qubit_key

__all__ = [
    "ClbitMap",
    "CompositeDecomposer",
    "LoopAnalyzer",
    "QubitAddress",
    "QubitMap",
    "QubitResolutionResult",
    "ResourceAllocator",
    "ValueResolver",
    "map_phi_outputs",
    "remap_static_phi_outputs",
    "resolve_if_condition",
    "resolve_qubit_key",
]
