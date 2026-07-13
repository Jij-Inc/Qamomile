"""Target-neutral containers for compiled artifacts and diagnostics."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Mapping
from copy import deepcopy
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from qamomile.circuit.transpiler.segments import ProgramABI

ArtifactT = TypeVar("ArtifactT")


def _freeze_metadata_value(value: Any) -> Any:
    """Copy metadata values into immutable container shapes.

    Args:
        value (Any): Serializer-friendly metadata value.

    Returns:
        Any: Deep-copied scalar or recursively immutable container.
    """
    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_metadata_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    return deepcopy(value)


class DiagnosticSeverity(enum.Enum):
    """Classify the severity of a compilation diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclasses.dataclass(frozen=True)
class CompilationDiagnostic:
    """Describe one target-independent or target-specific diagnostic.

    Args:
        severity (DiagnosticSeverity): Diagnostic severity.
        message (str): Human-readable explanation.
        code (str | None): Stable machine-readable diagnostic code.
        source (str | None): Optional source location or IR provenance label.
    """

    severity: DiagnosticSeverity
    message: str
    code: str | None = None
    source: str | None = None


@dataclasses.dataclass(frozen=True)
class CompilationMetadata:
    """Record how a target artifact was produced.

    Args:
        target (str): Stable compilation target name.
        pipeline (str): Lowering-family or pipeline name.
        properties (Mapping[str, Any]): Additional immutable-by-contract
            target metadata. Defaults to an empty mapping.
    """

    target: str
    pipeline: str
    properties: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Snapshot properties into a read-only metadata mapping."""
        object.__setattr__(
            self,
            "properties",
            _freeze_metadata_value(self.properties),
        )


@dataclasses.dataclass(frozen=True)
class CompiledProgram(Generic[ArtifactT]):
    """Package an artifact with its ABI, diagnostics, and provenance.

    Args:
        artifact (ArtifactT): Target-native circuit, graph, module, or package.
        abi (ProgramABI): Runtime-visible input and output contract.
        metadata (CompilationMetadata): Target and pipeline provenance.
        diagnostics (tuple[CompilationDiagnostic, ...]): Non-fatal compilation
            diagnostics. Defaults to an empty tuple.
    """

    artifact: ArtifactT
    abi: ProgramABI
    metadata: CompilationMetadata
    diagnostics: tuple[CompilationDiagnostic, ...] = ()
