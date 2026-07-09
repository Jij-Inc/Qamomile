"""Shared helpers for backend emission."""

from .composite_decomposer import CompositeDecomposer
from .condition_resolution import (
    map_merge_outputs,
    remap_static_merge_outputs,
    resolve_if_condition,
)
from .control_flow_emission import resolve_condition_address
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
    "map_merge_outputs",
    "remap_static_merge_outputs",
    "resolve_condition_address",
    "resolve_if_condition",
    "resolve_qubit_key",
]
