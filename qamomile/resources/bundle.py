"""Internal canonical QCore programs and source-addressable analysis facts."""

from __future__ import annotations

from dataclasses import dataclass, field

from qamomile.ir import Module, Program
from qamomile.serialization import decode_image

from .values import ResourceValue


@dataclass(frozen=True, slots=True)
class AnalysisFact:
    """One source-addressable fact retained from QCore analysis."""

    kind: str
    owner: str
    source: str
    operation: str
    contributions: tuple[tuple[str, ResourceValue], ...] = ()

    def __post_init__(self) -> None:
        """Require one canonical typed operation-contribution table."""

        names = tuple(name for name, _ in self.contributions)
        if self.kind != "operation" or not self.owner or not self.source:
            raise ValueError("analysis operation fact identity is invalid")
        if not self.operation or not self.contributions:
            raise ValueError("analysis operation fact requires typed contributions")
        if names != tuple(sorted(names)) or len(names) != len(set(names)):
            raise ValueError("analysis fact contributions are not canonical")
        if any(name != value.metric.identifier for name, value in self.contributions):
            raise ValueError("analysis fact contribution metric disagrees")


@dataclass(frozen=True, slots=True)
class AnalysisBundle:
    """Retain one canonical Module or Program and its explanation facts."""

    entrypoint: str
    analysis_version: str
    qcore_payload: bytes
    facts: tuple[AnalysisFact, ...] = ()
    _source: Module | Program | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Validate identities required for a reproducible wire value."""

        if not self.entrypoint or not self.analysis_version:
            raise ValueError("analysis bundle identities must be nonempty")
        if not self.qcore_payload:
            raise ValueError("analysis bundle requires its canonical QCore payload")

    @classmethod
    def from_source(
        cls,
        source: Module | Program,
        *,
        entrypoint: str,
        analysis_version: str,
        facts: tuple[AnalysisFact, ...] = (),
    ) -> AnalysisBundle:
        """Retain the exact source whose canonical bytes accompany the study.

        Args:
            source (Module | Program): Verified immutable analysis source.
            entrypoint (str): Selected source entrypoint.
            analysis_version (str): Resource-analysis version.
            facts (tuple[AnalysisFact, ...]): Source-addressable facts.

        Returns:
            AnalysisBundle: Bundle sharing its source without restoring bytes.
        """
        bundle = cls(entrypoint, analysis_version, source.to_bytes(), facts)
        object.__setattr__(bundle, "_source", source)
        bundle.semantic_source()
        return bundle

    def semantic_source(self) -> Module | Program:
        """Decode and identity-check the retained Module or Program.

        Returns:
            Module | Program: Verified source retained by this analysis bundle.

        Raises:
            ValueError: If the decoded Program has a different entrypoint or
                the decoded Module lacks the selected entrypoint.
            SerializationError: If the retained payload is invalid.
        """

        source = self._source
        if source is None:
            source = decode_image(self.qcore_payload)
            object.__setattr__(self, "_source", source)
        if isinstance(source, Program):
            if source.entrypoint != self.entrypoint:
                raise ValueError("analysis bundle entrypoint disagrees with QCore")
            return source
        source.function(self.entrypoint)
        return source

    @property
    def qcore_digest(self) -> str:
        """Derive the selected semantic identity from the retained source."""

        source = self.semantic_source()
        if isinstance(source, Program):
            return source.semantic_digest
        return source.selected_digest(self.entrypoint)


__all__ = ["AnalysisBundle", "AnalysisFact"]
