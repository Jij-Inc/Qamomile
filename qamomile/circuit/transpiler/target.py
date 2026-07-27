"""Protocols for explicit target planning and lowering pipelines."""

from __future__ import annotations

from typing import Protocol, TypeVar

from qamomile.circuit.transpiler.artifact import CompiledProgram
from qamomile.circuit.transpiler.prepared import PreparedModule

PlanT = TypeVar("PlanT")
ArtifactT = TypeVar("ArtifactT")


class CompilationTarget(Protocol[PlanT, ArtifactT]):
    """Define the contract implemented by every compilation target."""

    @property
    def name(self) -> str:
        """Return the stable target name.

        Returns:
            str: Stable name used in diagnostics and metadata.
        """
        ...

    def plan(self, program: PreparedModule) -> PlanT:
        """Choose target-specific lowering decisions for a program.

        Args:
            program (PreparedModule): Prepared semantic program.

        Returns:
            PlanT: Immutable target-specific compilation plan.
        """
        ...

    def compile(
        self,
        program: PreparedModule,
        plan: PlanT,
    ) -> CompiledProgram[ArtifactT]:
        """Lower and materialize a prepared program for this target.

        Args:
            program (PreparedModule): Prepared semantic program.
            plan (PlanT): Decisions returned by :meth:`plan`.

        Returns:
            CompiledProgram[ArtifactT]: Target-native artifact and metadata.
        """
        ...

    def validate(self, artifact: ArtifactT) -> None:
        """Validate a materialized artifact with target-native rules.

        Args:
            artifact (ArtifactT): Target-native artifact to validate.

        Raises:
            Exception: If target-native validation rejects the artifact.
        """
        ...
