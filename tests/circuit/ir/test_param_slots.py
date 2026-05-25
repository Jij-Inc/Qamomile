"""Tests for the ParamSlot manifest on ``Block``.

These tests verify that every classical kernel argument shows up as a
``ParamSlot`` with the correct kind (``RUNTIME_PARAMETER`` /
``COMPILE_TIME_BOUND``), type, default, and bound_value, both at trace
time (``QKernel.block`` via ``func_to_block``) and at the explicit
``build()`` entry point. The manifest must also survive the early
pipeline passes that re-construct the Block (inline, substitute,
parameter_shape_resolution).
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.parameter import ParamKind, ParamSlot
from qamomile.circuit.ir.types.primitives import (
    FloatType,
    UIntType,
)
from qamomile.circuit.transpiler.passes.inline import InlinePass
from qamomile.circuit.transpiler.passes.parameter_shape_resolution import (
    ParameterShapeResolutionPass,
)
from qamomile.circuit.transpiler.passes.substitution import (
    SubstitutionConfig,
    SubstitutionPass,
    SubstitutionRule,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slot_by_name(block: Block, name: str) -> ParamSlot:
    """Return the ``ParamSlot`` whose ``name`` matches ``name``.

    Args:
        block (Block): A Block whose ``param_slots`` will be searched.
        name (str): The parameter name to look up.

    Returns:
        ParamSlot: The matching slot.

    Raises:
        AssertionError: If no slot with that name exists, with the full
            list of available slot names for debugging.
    """
    for slot in block.param_slots:
        if slot.name == name:
            return slot
    raise AssertionError(
        f"No ParamSlot named {name!r}; available: {[s.name for s in block.param_slots]}"
    )


# ---------------------------------------------------------------------------
# QKernel.block (func_to_block path): no bindings, all classical → PARAM
# ---------------------------------------------------------------------------


@qmc.qkernel
def _scalar_param(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single Float parameter, no default."""
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


@qmc.qkernel
def _two_scalars(q: qmc.Qubit, alpha: qmc.Float, beta: qmc.Float) -> qmc.Qubit:
    """Two Float parameters."""
    q = qmc.rx(q, alpha)
    q = qmc.rz(q, beta)
    return q


@qmc.qkernel
def _vector_param(
    qs: qmc.Vector[qmc.Qubit], thetas: qmc.Vector[qmc.Float]
) -> qmc.Vector[qmc.Qubit]:
    """Array-typed parameter."""
    n = qs.shape[0]
    for i in qmc.range(n):
        qs[i] = qmc.rx(qs[i], thetas[i])
    return qs


@qmc.qkernel
def _no_classical_args(q: qmc.Qubit) -> qmc.Qubit:
    """Only a Qubit input; no classical arguments at all."""
    return qmc.h(q)


class TestFuncToBlockSlots:
    """``QKernel.block`` (cached via func_to_block) marks every classical arg as PARAM."""

    def test_scalar_param_slot_present(self):
        """A single Float parameter shows up as one RUNTIME_PARAMETER slot."""
        block = _scalar_param.block
        assert len(block.param_slots) == 1
        slot = block.param_slots[0]
        assert slot.name == "theta"
        assert slot.kind is ParamKind.RUNTIME_PARAMETER
        assert isinstance(slot.type, FloatType)
        assert slot.ndim == 0
        assert slot.default is None
        assert slot.bound_value is None

    def test_two_scalar_params_in_signature_order(self):
        """Multiple parameters preserve ``signature.parameters`` order."""
        slots = _two_scalars.block.param_slots
        assert [s.name for s in slots] == ["alpha", "beta"]
        for s in slots:
            assert s.kind is ParamKind.RUNTIME_PARAMETER

    def test_array_param_records_ndim(self):
        """``Vector[Float]`` parameter has ``ndim=1``."""
        slot = _slot_by_name(_vector_param.block, "thetas")
        assert slot.kind is ParamKind.RUNTIME_PARAMETER
        assert isinstance(slot.type, FloatType)
        assert slot.ndim == 1

    def test_qubit_only_kernel_has_empty_slots(self):
        """Kernels with only quantum arguments produce an empty manifest."""
        assert _no_classical_args.block.param_slots == ()


# ---------------------------------------------------------------------------
# QKernel.build(): bindings → COMPILE_TIME_BOUND, parameters=[...] → PARAM
# ---------------------------------------------------------------------------


class TestBuildSlotKinds:
    """Verify kind assignment under explicit bindings / parameters."""

    def test_kwarg_binding_is_compile_time_bound(self):
        """``build(theta=0.5)`` records theta as COMPILE_TIME_BOUND with the bound value."""
        block = _scalar_param.build(theta=0.5)
        slot = _slot_by_name(block, "theta")
        assert slot.kind is ParamKind.COMPILE_TIME_BOUND
        assert slot.bound_value == 0.5

    def test_explicit_parameter_stays_runtime(self):
        """``build(parameters=['theta'])`` keeps theta as RUNTIME_PARAMETER."""
        block = _scalar_param.build(parameters=["theta"])
        slot = _slot_by_name(block, "theta")
        assert slot.kind is ParamKind.RUNTIME_PARAMETER
        assert slot.bound_value is None

    def test_mixed_param_and_binding(self):
        """One arg can be runtime parameter while another is bound."""
        block = _two_scalars.build(parameters=["alpha"], beta=1.5)
        alpha = _slot_by_name(block, "alpha")
        beta = _slot_by_name(block, "beta")
        assert alpha.kind is ParamKind.RUNTIME_PARAMETER
        assert beta.kind is ParamKind.COMPILE_TIME_BOUND
        assert beta.bound_value == 1.5


# ---------------------------------------------------------------------------
# Python defaults: bind_defaults=True (build) vs False (kernel.block)
# ---------------------------------------------------------------------------


@qmc.qkernel
def _with_default(q: qmc.Qubit, theta: qmc.Float = 0.25) -> qmc.Qubit:  # type: ignore[assignment]
    """Float parameter with a Python default.

    The ``# type: ignore`` exists because ``qmc.Float`` is the
    frontend Handle wrapper; ``0.25`` is a plain Python float. The
    qkernel decorator coerces appropriately at trace time, but static
    type checkers see the mismatch.
    """
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return q


class TestPythonDefaults:
    """Python ``=default`` semantics differ between the two trace paths."""

    def test_kernel_block_does_not_bind_default(self):
        """``kernel.block`` leaves defaults symbolic; default appears only in ParamSlot.default."""
        slot = _slot_by_name(_with_default.block, "theta")
        assert slot.kind is ParamKind.RUNTIME_PARAMETER
        assert slot.bound_value is None
        assert slot.default == 0.25

    def test_build_binds_default(self):
        """``kernel.build()`` (no kwargs) binds Python defaults as COMPILE_TIME_BOUND."""
        block = _with_default.build()
        slot = _slot_by_name(block, "theta")
        assert slot.kind is ParamKind.COMPILE_TIME_BOUND
        assert slot.bound_value == 0.25
        assert slot.default == 0.25

    def test_build_override_takes_precedence_over_default(self):
        """Explicit kwarg overrides the default value."""
        block = _with_default.build(theta=1.0)
        slot = _slot_by_name(block, "theta")
        assert slot.kind is ParamKind.COMPILE_TIME_BOUND
        assert slot.bound_value == 1.0


# ---------------------------------------------------------------------------
# Pipeline preservation: param_slots survives inline()
# ---------------------------------------------------------------------------


class TestPipelinePreservation:
    """Pipeline passes that re-construct Block must preserve ``param_slots``."""

    def test_inline_preserves_param_slots(self):
        """``InlinePass`` does not drop ``param_slots`` when producing an AFFINE block."""
        affine = InlinePass().run(_scalar_param.block)
        assert affine.kind in (BlockKind.AFFINE, BlockKind.HIERARCHICAL)
        assert len(affine.param_slots) == 1
        assert affine.param_slots[0].name == "theta"

    def test_substitution_preserves_param_slots(self):
        """``SubstitutionPass`` carries ``param_slots`` through Block reconstruction.

        Constructs a config with a single rule whose ``source_name`` does
        not match any operation in the block. The pass still reconstructs
        the Block (it short-circuits only when ``rules`` is empty) but
        applies no operational transformation, isolating the
        param_slots-forwarding behavior.
        """
        config = SubstitutionConfig(
            rules=[SubstitutionRule(source_name="__nonexistent__", strategy="default")]
        )
        out = SubstitutionPass(config).run(_scalar_param.block)
        assert out.param_slots == _scalar_param.block.param_slots

    def test_parameter_shape_resolution_preserves_param_slots(self):
        """``ParameterShapeResolutionPass`` keeps ``param_slots`` when reconstructing.

        Feeds the cached HIERARCHICAL ``_vector_param.block`` (whose
        ``thetas`` ArrayValue has a symbolic shape dim) together with a
        concrete ``thetas`` binding. That combination produces a
        non-empty substitution map and forces the pass to rebuild the
        Block, exercising the param_slots-forwarding path.
        """
        out = ParameterShapeResolutionPass(
            bindings={"thetas": np.array([0.1, 0.2, 0.3])}
        ).run(_vector_param.block)
        assert out.param_slots == _vector_param.block.param_slots


# ---------------------------------------------------------------------------
# Block-construction invariants
# ---------------------------------------------------------------------------


class TestBlockInvariants:
    """Block.__post_init__ enforces ParamSlot name uniqueness."""

    def test_duplicate_slot_names_rejected(self):
        """Two slots with the same name raise at Block construction."""
        slot_a = ParamSlot(
            name="theta",
            type=FloatType(),
            kind=ParamKind.RUNTIME_PARAMETER,
        )
        slot_b = ParamSlot(
            name="theta",
            type=UIntType(),
            kind=ParamKind.COMPILE_TIME_BOUND,
            bound_value=3,
        )
        with pytest.raises(ValueError, match="Duplicate ParamSlot"):
            Block(
                name="bad",
                kind=BlockKind.AFFINE,
                param_slots=(slot_a, slot_b),
            )
