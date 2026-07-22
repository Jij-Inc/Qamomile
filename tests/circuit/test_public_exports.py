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
import qamomile.circuit.stdlib as stdlib
import qamomile.circuit.stdlib.block_encoding as block_encoding
from qamomile.circuit.estimator.resource_estimator import (
    OpaqueCallContext,
    ResourceEstimator,
    UnknownResourcePolicy,
)
from qamomile.circuit.frontend.callable_signature import CallableSignature
from qamomile.circuit.frontend.composite_gate import composite_gate
from qamomile.circuit.frontend.operation.global_phase import global_phase
from qamomile.circuit.frontend.operation.measurement import (
    measure_reset,
    project_x,
    project_y,
    project_z,
    reset,
)
from qamomile.circuit.frontend.oracle import Oracle, opaque
from qamomile.circuit.frontend.struct import struct
from qamomile.circuit.stdlib.block_encoding.ising_z import (
    IsingZBlockEncoding,
    ising_z_block_encoding,
)
from qamomile.circuit.stdlib.block_encoding.lcu import (
    LCUBlockEncoding,
    LCUBlockEncodingTerm,
    identity_block_encoding,
    lcu_block_encoding,
)
from qamomile.circuit.stdlib.block_encoding.pauli import (
    PauliLCUBlockEncoding,
    pauli_lcu_block_encoding,
)
from qamomile.circuit.stdlib.block_encoding.periodic_shift import (
    PeriodicShiftLCUBlockEncoding,
    periodic_shift_lcu_block_encoding,
)
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
    assert qmc.composite_gate is composite_gate
    assert qmc.opaque is opaque
    assert qmc.Oracle is Oracle
    assert qmc.CallableSignature is CallableSignature
    assert qmc.ResourceEstimator is ResourceEstimator
    assert qmc.UnknownResourcePolicy is UnknownResourcePolicy
    assert qmc.OpaqueCallContext is OpaqueCallContext

    for name in (
        "composite_gate",
        "opaque",
        "Oracle",
        "CallableSignature",
        "ResourceEstimator",
        "UnknownResourcePolicy",
        "OpaqueCallContext",
    ):
        assert name in qmc.__all__, (
            f"{name!r} should be listed in qamomile.circuit.__all__"
        )


def test_struct_is_publicly_reexported() -> None:
    """The trace-time record decorator is part of the circuit API."""
    assert qmc.struct is struct
    assert "struct" in qmc.__all__


def test_global_phase_is_publicly_reexported() -> None:
    """The global-phase combinator is part of the curated circuit API."""
    assert qmc.global_phase is global_phase
    assert "global_phase" in qmc.__all__


def test_pauli_lcu_block_encoding_api_is_publicly_reexported() -> None:
    """The common descriptor, Pauli subtype, and factory are public."""
    assert qmc.LCUBlockEncoding is LCUBlockEncoding
    assert stdlib.LCUBlockEncoding is LCUBlockEncoding
    assert qmc.PauliLCUBlockEncoding is PauliLCUBlockEncoding
    assert stdlib.PauliLCUBlockEncoding is PauliLCUBlockEncoding
    assert qmc.pauli_lcu_block_encoding is pauli_lcu_block_encoding
    assert stdlib.pauli_lcu_block_encoding is pauli_lcu_block_encoding

    for namespace in (qmc, stdlib):
        assert "LCUBlockEncoding" in namespace.__all__
        assert "PauliLCUBlockEncoding" in namespace.__all__
        assert "pauli_lcu_block_encoding" in namespace.__all__
        assert not hasattr(namespace, "pauli_lcu_num_selection_qubits")


def test_recursive_lcu_block_encoding_api_is_publicly_reexported() -> None:
    """Recursive LCU and Ising-Z construction APIs are public."""
    exports = {
        "LCUBlockEncoding": LCUBlockEncoding,
        "LCUBlockEncodingTerm": LCUBlockEncodingTerm,
        "identity_block_encoding": identity_block_encoding,
        "lcu_block_encoding": lcu_block_encoding,
        "IsingZBlockEncoding": IsingZBlockEncoding,
        "ising_z_block_encoding": ising_z_block_encoding,
    }
    for namespace in (qmc, stdlib):
        for name, value in exports.items():
            assert getattr(namespace, name) is value
            assert name in namespace.__all__


def test_block_encoding_subpackage_groups_every_public_producer() -> None:
    """The organized namespace contains every public producer."""
    exports = {
        "LCUBlockEncoding": LCUBlockEncoding,
        "LCUBlockEncodingTerm": LCUBlockEncodingTerm,
        "identity_block_encoding": identity_block_encoding,
        "lcu_block_encoding": lcu_block_encoding,
        "IsingZBlockEncoding": IsingZBlockEncoding,
        "ising_z_block_encoding": ising_z_block_encoding,
        "PauliLCUBlockEncoding": PauliLCUBlockEncoding,
        "pauli_lcu_block_encoding": pauli_lcu_block_encoding,
        "PeriodicShiftLCUBlockEncoding": PeriodicShiftLCUBlockEncoding,
        "periodic_shift_lcu_block_encoding": periodic_shift_lcu_block_encoding,
    }
    for name, value in exports.items():
        assert getattr(block_encoding, name) is value
        assert getattr(stdlib, name) is value
        assert getattr(qmc, name) is value
        assert name in block_encoding.__all__


def test_measurement_helpers_are_publicly_reexported():
    """Measurement/projection helper API is reachable from ``qamomile.circuit``."""
    assert qmc.project_z is project_z
    assert qmc.project_x is project_x
    assert qmc.project_y is project_y
    assert qmc.reset is reset
    assert qmc.measure_reset is measure_reset

    for name in (
        "project_z",
        "project_x",
        "project_y",
        "reset",
        "measure_reset",
    ):
        assert name in qmc.__all__, (
            f"{name!r} should be listed in qamomile.circuit.__all__"
        )


def test_removed_parallel_composite_api_is_not_exposed() -> None:
    """Only the QKernel-returning composite_gate decorator is public."""
    assert not hasattr(qmc, "composite")
    assert not hasattr(qmc, "CompositeGate")


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
