"""Compatibility exports for shared IR value-mapping utilities."""

from qamomile.circuit.ir.uuid_remapper import UUIDRemapper
from qamomile.circuit.ir.value_mapping import ValueSubstitutor

__all__ = ["UUIDRemapper", "ValueSubstitutor"]
