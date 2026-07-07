"""Tests for the public export surface of `qamomile.circuit`.

These tests pin down which symbols `qamomile.circuit` re-exports and make
sure that re-exported types are identical to their canonical
implementations under the deep paths. This guards against accidental
shadowing (e.g., someone redefining ``SampleResult`` in
``qamomile/circuit/__init__.py`` with a different class) and against
silently removing a public re-export.

The deep paths (e.g. ``qamomile.circuit.transpiler.job.SampleResult``)
are kept importable for backward compatibility and are intentionally
re-imported here, so the test also documents the back-compat contract.
"""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.frontend.callable_signature import CallableSignature
from qamomile.circuit.frontend.composite_gate import (
    CompositeGate,
    composite,
    composite_gate,
)
from qamomile.circuit.frontend.oracle import Oracle, opaque
from qamomile.circuit.ir.operation.callable import ResourceMetadata
from qamomile.circuit.transpiler import job as _job_module


def test_job_types_are_publicly_reexported():
    """Job/result types from ``transpiler.job`` are reachable as ``qmc.X``.

    Each public attribute must be the same Python object as its source
    in ``qamomile.circuit.transpiler.job`` — not a re-defined copy.
    """
    for name in (
        "Job",
        "JobStatus",
        "SampleResult",
        "SampleJob",
        "RunJob",
        "ExpvalJob",
    ):
        public = getattr(qmc, name)
        canonical = getattr(_job_module, name)
        assert public is canonical, (
            f"qamomile.circuit.{name} must be the same object as "
            f"qamomile.circuit.transpiler.job.{name}"
        )


def test_job_types_listed_in_all():
    """Each publicly re-exported job type appears in ``qmc.__all__``.

    ``__all__`` controls ``from qamomile.circuit import *`` and is the
    documented public surface; re-exporting without listing leaves the
    name out of star-imports and out of doc tooling.
    """
    for name in (
        "Job",
        "JobStatus",
        "SampleResult",
        "SampleJob",
        "RunJob",
        "ExpvalJob",
    ):
        assert name in qmc.__all__, (
            f"{name!r} should be listed in qamomile.circuit.__all__"
        )


def test_callable_helpers_are_publicly_reexported():
    """Primary callable helper API is reachable from ``qamomile.circuit``."""
    assert qmc.composite is composite
    assert qmc.composite_gate is composite_gate
    assert qmc.opaque is opaque
    assert qmc.Oracle is Oracle
    assert qmc.CallableSignature is CallableSignature
    assert qmc.ResourceMetadata is ResourceMetadata

    for name in (
        "composite_gate",
        "opaque",
        "Oracle",
        "CallableSignature",
        "ResourceMetadata",
    ):
        assert name in qmc.__all__, (
            f"{name!r} should be listed in qamomile.circuit.__all__"
        )


def test_compat_callable_aliases_are_direct_but_not_star_exported():
    """Compatibility callable aliases stay reachable but not primary."""
    assert qmc.composite is composite
    assert qmc.CompositeGate is CompositeGate
    assert "composite" not in qmc.__all__
    assert "CompositeGate" not in qmc.__all__


def test_compiler_callable_descriptors_are_not_top_level_api():
    """Compiler-facing callable descriptors stay out of ``qamomile.circuit``.

    The intended frontend surface is expressed through qkernel,
    composite_gate, and opaque helpers. ``CallableDef`` and related IR
    descriptors are available from deep compiler paths only.
    """
    for name in (
        "CallableDef",
        "CallableRef",
        "CallableImplementation",
        "InvokeOperation",
        "CallPolicy",
        "CallTransform",
    ):
        assert name not in qmc.__all__
        assert not hasattr(qmc, name)


def test_deep_path_still_importable():
    """The internal deep path remains importable for back-compat.

    Existing user code that imports from
    ``qamomile.circuit.transpiler.job`` must keep working after the
    public re-export is added.
    """
    from qamomile.circuit.transpiler.job import (  # noqa: F401
        ExpvalJob,
        Job,
        JobStatus,
        RunJob,
        SampleJob,
        SampleResult,
    )
