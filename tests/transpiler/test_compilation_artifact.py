"""Tests for target-neutral compilation artifact ownership."""

from __future__ import annotations

import pytest

from qamomile.circuit.transpiler.artifact import CompilationMetadata


def test_compilation_metadata_snapshots_and_freezes_properties() -> None:
    """Metadata cannot be changed through its source or returned containers."""
    source = {"features": ["control-flow"]}
    metadata = CompilationMetadata("hugr", "program_graph", source)
    source["features"].append("mutated")

    assert metadata.properties["features"] == ("control-flow",)
    with pytest.raises(TypeError):
        metadata.properties["extra"] = True  # type: ignore[index]
