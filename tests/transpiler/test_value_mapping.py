"""Unit tests for value_mapping.py — UUIDRemapper and ValueSubstitutor.

Covers fixes for:
- Bug #5 (ArrayValue.shape cloning/substitution):
    UUIDRemapper.clone_value() now clones ArrayValue.shape dimension Values
    with fresh UUIDs so sub-kernel dimension parameters survive inlining.
    ValueSubstitutor.substitute_value() now substitutes shape dimension UUIDs
    through the value_map so callers' concrete sizes propagate correctly.

- Bug #6 (IfOperation merge-slot cloning/substitution):
    UUIDRemapper.clone_operation() now recurses into IfOperation merge
    slots, giving each merged output fresh UUIDs.
    ValueSubstitutor.substitute_operation() now substitutes values inside
    IfOperation merge slots so inlined merge operands are correctly
    remapped.
"""

from __future__ import annotations

import numpy as np
import pytest

from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import IfOperation
from qamomile.circuit.ir.types.primitives import (
    BitType,
    QubitType,
    UIntType,
)
from qamomile.circuit.ir.types.q_register import QFixedType
from qamomile.circuit.ir.value import (
    ArrayRuntimeMetadata,
    ArrayValue,
    CastMetadata,
    QFixedMetadata,
    Value,
    ValueMetadata,
)
from qamomile.circuit.transpiler.passes.value_mapping import (
    UUIDRemapper,
    ValueSubstitutor,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_value(name: str, type_cls: type = UIntType) -> Value:
    """Create a simple Value with the given name and type."""
    return Value(type=type_cls(), name=name)


def _make_const_value(
    name: str, const: int | float, type_cls: type = UIntType
) -> Value:
    """Create a constant Value."""
    return Value(type=type_cls(), name=name).with_const(const)


def _make_array_value(
    name: str,
    shape_vals: tuple[Value, ...] = (),
    type_cls: type = QubitType,
) -> ArrayValue:
    """Create an ArrayValue with the given shape dimension Values."""
    return ArrayValue(type=type_cls(), name=name, shape=shape_vals)


# ===========================================================================
# Metadata UUID cloning
# ===========================================================================


class TestUUIDRemapperMetadataCloning:
    """Tests that UUIDRemapper clones UUID references in ValueMetadata."""

    def test_cast_metadata_uuid_references_are_cloned(self) -> None:
        """Cast source and qubit references follow cloned Values."""
        source = _make_value("source")
        qubit = _make_value("q", QubitType)
        carrier = Value(
            type=UIntType(),
            name="cast",
            metadata=ValueMetadata(
                cast=CastMetadata(
                    source_uuid=source.uuid,
                    source_logical_id=source.logical_id,
                    qubit_uuids=(qubit.uuid,),
                    qubit_logical_ids=(qubit.logical_id,),
                ),
            ),
        )

        remapper = UUIDRemapper()
        cloned_source = remapper.clone_value(source)
        cloned_qubit = remapper.clone_value(qubit)
        cloned_carrier = remapper.clone_value(carrier)

        assert isinstance(cloned_carrier, Value)
        assert cloned_carrier.metadata.cast is not None
        assert cloned_carrier.metadata.cast.source_uuid == cloned_source.uuid
        assert cloned_carrier.metadata.cast.source_logical_id == (
            cloned_source.logical_id
        )
        assert cloned_carrier.metadata.cast.qubit_uuids == (cloned_qubit.uuid,)
        assert cloned_carrier.metadata.cast.qubit_logical_ids == (
            cloned_qubit.logical_id,
        )

    def test_qfixed_metadata_qubit_references_are_cloned(self) -> None:
        """QFixed carrier qubit references follow cloned Values."""
        q0 = _make_value("q0", QubitType)
        q1 = _make_value("q1", QubitType)
        qfixed = Value(
            type=UIntType(),
            name="qfixed",
            metadata=ValueMetadata(
                qfixed=QFixedMetadata(
                    qubit_uuids=(q0.uuid, q1.uuid),
                    num_bits=2,
                    int_bits=0,
                ),
            ),
        )

        remapper = UUIDRemapper()
        cloned_q0 = remapper.clone_value(q0)
        cloned_q1 = remapper.clone_value(q1)
        cloned_qfixed = remapper.clone_value(qfixed)

        assert isinstance(cloned_qfixed, Value)
        assert cloned_qfixed.metadata.qfixed is not None
        assert cloned_qfixed.metadata.qfixed.qubit_uuids == (
            cloned_q0.uuid,
            cloned_q1.uuid,
        )
        assert cloned_qfixed.metadata.qfixed.num_bits == 2
        assert cloned_qfixed.metadata.qfixed.int_bits == 0

    def test_cast_metadata_uuid_references_are_substituted(self) -> None:
        """Cast metadata references follow substituted Values."""
        old_source = _make_value("old_source")
        new_source = _make_value("new_source")
        old_qubit = _make_value("old_q", QubitType)
        new_qubit = _make_value("new_q", QubitType)
        carrier = Value(
            type=UIntType(),
            name="cast",
            metadata=ValueMetadata(
                cast=CastMetadata(
                    source_uuid=old_source.uuid,
                    source_logical_id=old_source.logical_id,
                    qubit_uuids=(old_qubit.uuid,),
                    qubit_logical_ids=(old_qubit.logical_id,),
                ),
            ),
        )

        sub = ValueSubstitutor({old_source.uuid: new_source, old_qubit.uuid: new_qubit})
        result = sub.substitute_value(carrier)

        assert isinstance(result, Value)
        assert result.metadata.cast is not None
        assert result.metadata.cast.source_uuid == new_source.uuid
        assert result.metadata.cast.source_logical_id == new_source.logical_id
        assert result.metadata.cast.qubit_uuids == (new_qubit.uuid,)
        assert result.metadata.cast.qubit_logical_ids == (new_qubit.logical_id,)

    def test_qfixed_metadata_qubit_references_are_substituted(self) -> None:
        """QFixed metadata references follow substituted Values."""
        old_q0 = _make_value("old_q0", QubitType)
        new_q0 = _make_value("new_q0", QubitType)
        old_q1 = _make_value("old_q1", QubitType)
        new_q1 = _make_value("new_q1", QubitType)
        qfixed = Value(
            type=UIntType(),
            name="qfixed",
            metadata=ValueMetadata(
                qfixed=QFixedMetadata(
                    qubit_uuids=(old_q0.uuid, old_q1.uuid),
                    num_bits=2,
                    int_bits=0,
                ),
            ),
        )

        sub = ValueSubstitutor({old_q0.uuid: new_q0, old_q1.uuid: new_q1})
        result = sub.substitute_value(qfixed)

        assert isinstance(result, Value)
        assert result.metadata.qfixed is not None
        assert result.metadata.qfixed.qubit_uuids == (new_q0.uuid, new_q1.uuid)
        assert result.metadata.qfixed.num_bits == 2
        assert result.metadata.qfixed.int_bits == 0


class TestArrayRuntimeMetadataSymbolicRootLimitations:
    """Known limitations around symbolic root-address metadata."""

    @pytest.mark.xfail(
        strict=True,
        reason=("ArrayRuntimeMetadata cannot encode symbolic affine root indices yet."),
    )
    def test_scalar_inline_metadata_cannot_promote_symbolic_slice_parent(
        self,
    ) -> None:
        """Scalar tuple metadata cannot promote ``q[j:j+1][0]`` to root ``q[j]``.

        A scalar helper traces ``expval((q,), obs)`` with standalone-qubit
        metadata, encoded as the ``("", -1)`` sentinel.  After inlining, that
        scalar may map to an element of a caller-side symbolic slice such as
        ``q[j:j+1][0]``.  The current metadata can only store integer root
        indices, so it cannot represent the desired root address ``(q.uuid,
        j)`` and leaves the sentinel in place.
        """
        root = _make_array_value("q")
        loop_index = _make_value("j")
        one = _make_const_value("one", 1)
        zero = _make_const_value("zero", 0)
        symbolic_view = ArrayValue(
            type=QubitType(),
            name="q[j:j+1]",
            slice_of=root,
            slice_start=loop_index,
            slice_step=one,
        )
        callee_scalar = _make_value("callee_q", QubitType)
        caller_element = Value(
            type=QubitType(),
            name="q[j]",
            parent_array=symbolic_view,
            element_indices=(zero,),
        )
        tuple_operand = ArrayValue(
            type=QubitType(),
            name="expval_qubits",
            metadata=ValueMetadata(
                array_runtime=ArrayRuntimeMetadata(
                    element_uuids=(callee_scalar.uuid,),
                    element_logical_ids=(callee_scalar.logical_id,),
                    element_parent_uuids=("",),
                    element_parent_indices=(-1,),
                ),
            ),
        )

        result = ValueSubstitutor(
            {callee_scalar.uuid: caller_element}
        ).substitute_value(tuple_operand)

        assert isinstance(result, ArrayValue)
        assert result.metadata.array_runtime is not None
        assert result.metadata.array_runtime.element_uuids == (caller_element.uuid,)
        assert result.metadata.array_runtime.element_parent_uuids == (root.uuid,)
        assert result.metadata.array_runtime.element_parent_indices == (loop_index,)


class TestArrayRuntimeMetadataSymbolicParentSubstitution:
    """Tests symbolic slice parent metadata substitution."""

    def test_symbolic_slice_parent_uuid_is_substituted(self) -> None:
        """Symbolic slice parents keep a valid substituted parent UUID."""
        old_parent = _make_array_value("callee_q")
        root = _make_array_value("caller_q")
        start = _make_value("i")
        one = _make_const_value("one", 1)
        replacement_parent = ArrayValue(
            type=QubitType(),
            name="caller_q[i:]",
            slice_of=root,
            slice_start=start,
            slice_step=one,
        )
        old_child = _make_value("callee_q_1", QubitType)
        new_child = _make_value("caller_q_i_plus_1", QubitType)
        carrier = ArrayValue(
            type=QubitType(),
            name="tuple_qubits",
            metadata=ValueMetadata(
                array_runtime=ArrayRuntimeMetadata(
                    element_uuids=(old_child.uuid,),
                    element_logical_ids=(old_child.logical_id,),
                    element_parent_uuids=(old_parent.uuid,),
                    element_parent_indices=(1,),
                ),
            ),
        )

        result = ValueSubstitutor(
            {
                old_parent.uuid: replacement_parent,
                old_child.uuid: new_child,
            }
        ).substitute_value(carrier)

        assert isinstance(result, ArrayValue)
        assert result.metadata.array_runtime is not None
        assert result.metadata.array_runtime.element_uuids == (new_child.uuid,)
        assert result.metadata.array_runtime.element_logical_ids == (
            new_child.logical_id,
        )
        assert result.metadata.array_runtime.element_parent_uuids == (
            replacement_parent.uuid,
        )
        assert result.metadata.array_runtime.element_parent_indices == (1,)

    def test_parent_metadata_without_element_uuids_is_preserved(self) -> None:
        """Parent provenance can be substituted without element UUIDs."""
        old_parent = _make_array_value("old_parent")
        new_parent = _make_array_value("new_parent")
        carrier = ArrayValue(
            type=QubitType(),
            name="parent_only",
            metadata=ValueMetadata(
                array_runtime=ArrayRuntimeMetadata(
                    element_parent_uuids=(old_parent.uuid,),
                    element_parent_indices=(2,),
                ),
            ),
        )

        result = ValueSubstitutor({old_parent.uuid: new_parent}).substitute_value(
            carrier
        )

        assert isinstance(result, ArrayValue)
        assert result.metadata.array_runtime is not None
        assert result.metadata.array_runtime.element_uuids == ()
        assert result.metadata.array_runtime.element_parent_uuids == (new_parent.uuid,)
        assert result.metadata.array_runtime.element_parent_indices == (2,)


# ===========================================================================
# Bug #5: ArrayValue.shape cloning and substitution
# ===========================================================================


class TestArrayValueShapeCloning:
    """Tests that UUIDRemapper clones ArrayValue.shape values."""

    @pytest.mark.parametrize("ndims", [1, 2, 3])
    def test_shape_values_get_fresh_uuids(self, ndims: int) -> None:
        """Cloned ArrayValue shape dims get new UUIDs for any dimensionality."""
        dims = tuple(_make_value(f"dim{i}") for i in range(ndims))
        arr = _make_array_value("q", shape_vals=dims)

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert cloned.uuid != arr.uuid
        assert len(cloned.shape) == ndims
        for i in range(ndims):
            assert cloned.shape[i].uuid != dims[i].uuid
            assert cloned.shape[i].name == dims[i].name

    def test_shape_uuids_appear_in_remap_table(self) -> None:
        """Cloned shape dimension UUIDs should be recorded in uuid_remap."""
        dim = _make_value("n")
        arr = _make_array_value("q", shape_vals=(dim,))

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert dim.uuid in remapper.uuid_remap
        assert isinstance(cloned, ArrayValue)
        assert remapper.uuid_remap[dim.uuid] == cloned.shape[0].uuid

    def test_empty_shape_is_preserved(self) -> None:
        """ArrayValue with empty shape is cloned without error."""
        arr = _make_array_value("q", shape_vals=())

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert cloned.shape == ()

    def test_random_dimensionality_shape_cloning(self) -> None:
        """ArrayValue with random-count dimensions clones all of them."""
        rng = np.random.default_rng(seed=42)
        ndims = int(rng.integers(2, 6))
        dims = tuple(_make_value(f"d{i}") for i in range(ndims))
        arr = _make_array_value("tensor", shape_vals=dims)

        remapper = UUIDRemapper()
        cloned = remapper.clone_value(arr)

        assert isinstance(cloned, ArrayValue)
        assert len(cloned.shape) == ndims
        for orig, clone in zip(dims, cloned.shape):
            assert clone.uuid != orig.uuid
            assert orig.uuid in remapper.uuid_remap


class TestCarrierMetadataMapping:
    """Tests that cast/QFixed carrier metadata follows value remapping."""

    def test_uuid_remapper_clones_indexed_cast_carriers(self) -> None:
        """Cast carrier metadata and operation mapping use cloned array UUIDs."""
        source = _make_array_value("q")
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=source.uuid,
                source_logical_id=source.logical_id,
                qubit_uuids=[f"{source.uuid}_0", f"{source.uuid}_1"],
                qubit_logical_ids=[
                    f"{source.logical_id}_0",
                    f"{source.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{source.uuid}_0", f"{source.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        op = CastOperation(
            operands=[source],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{source.uuid}_0", f"{source.uuid}_1"],
        )

        cloned = UUIDRemapper().clone_operation(op)

        assert isinstance(cloned, CastOperation)
        cloned_source = cloned.operands[0]
        cloned_result = cloned.results[0]
        assert cloned_result.get_cast_source_uuid() == cloned_source.uuid
        assert cloned_result.get_cast_qubit_uuids() == (
            f"{cloned_source.uuid}_0",
            f"{cloned_source.uuid}_1",
        )
        assert cloned_result.get_qfixed_qubit_uuids() == (
            f"{cloned_source.uuid}_0",
            f"{cloned_source.uuid}_1",
        )
        assert cloned.qubit_mapping == [
            f"{cloned_source.uuid}_0",
            f"{cloned_source.uuid}_1",
        ]

    def test_value_substitutor_rewrites_indexed_cast_carriers(self) -> None:
        """Cast carrier metadata and operation mapping follow substitution."""
        old_source = _make_array_value("old_q")
        new_source = _make_array_value("new_q")
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=old_source.uuid,
                source_logical_id=old_source.logical_id,
                qubit_uuids=[f"{old_source.uuid}_0", f"{old_source.uuid}_1"],
                qubit_logical_ids=[
                    f"{old_source.logical_id}_0",
                    f"{old_source.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{old_source.uuid}_0", f"{old_source.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        op = CastOperation(
            operands=[old_source],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{old_source.uuid}_0", f"{old_source.uuid}_1"],
        )

        substituted = ValueSubstitutor(
            {old_source.uuid: new_source}
        ).substitute_operation(op)

        assert isinstance(substituted, CastOperation)
        result = substituted.results[0]
        assert substituted.operands[0] is new_source
        assert result.get_cast_source_uuid() == new_source.uuid
        assert result.get_cast_source_logical_id() == new_source.logical_id
        assert result.get_cast_qubit_uuids() == (
            f"{new_source.uuid}_0",
            f"{new_source.uuid}_1",
        )
        assert result.get_cast_qubit_logical_ids() == (
            f"{new_source.logical_id}_0",
            f"{new_source.logical_id}_1",
        )
        assert result.get_qfixed_qubit_uuids() == (
            f"{new_source.uuid}_0",
            f"{new_source.uuid}_1",
        )
        assert substituted.qubit_mapping == [
            f"{new_source.uuid}_0",
            f"{new_source.uuid}_1",
        ]

    def test_value_substitutor_folds_view_carriers_to_root_space(self) -> None:
        """Carrier indices fold through slice views into root index space.

        Inlining a sub-kernel called with ``q[1::2]`` maps the callee formal
        to a strided view; the carrier index must be re-based to the root
        array (``view_i -> root_{start + step * i}``), not kept verbatim.
        """
        formal = _make_array_value("formal")
        root = _make_array_value("root", shape_vals=(_make_const_value("len", 4),))
        view = ArrayValue(
            type=QubitType(),
            name="view",
            shape=(_make_const_value("view_len", 2),),
            slice_of=root,
            slice_start=_make_const_value("start", 1),
            slice_step=_make_const_value("step", 2),
        )
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=formal.uuid,
                source_logical_id=formal.logical_id,
                qubit_uuids=[f"{formal.uuid}_0", f"{formal.uuid}_1"],
                qubit_logical_ids=[
                    f"{formal.logical_id}_0",
                    f"{formal.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{formal.uuid}_0", f"{formal.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        op = CastOperation(
            operands=[formal],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{formal.uuid}_0", f"{formal.uuid}_1"],
        )

        substituted = ValueSubstitutor({formal.uuid: view}).substitute_operation(op)

        assert isinstance(substituted, CastOperation)
        result = substituted.results[0]
        assert result.get_cast_source_uuid() == view.uuid
        assert result.get_cast_qubit_uuids() == (
            f"{root.uuid}_1",
            f"{root.uuid}_3",
        )
        assert result.get_cast_qubit_logical_ids() == (
            f"{root.logical_id}_1",
            f"{root.logical_id}_3",
        )
        assert result.get_qfixed_qubit_uuids() == (
            f"{root.uuid}_1",
            f"{root.uuid}_3",
        )
        assert substituted.qubit_mapping == [
            f"{root.uuid}_1",
            f"{root.uuid}_3",
        ]

    def test_value_substitutor_symbolic_view_carrier_raises(self) -> None:
        """Substituting a carrier onto a symbolic-bound slice view fails fast.

        When the mapped view's ``slice_start`` / ``slice_step`` are not
        compile-time constants, the carrier index cannot be folded into
        root-array space, so emitting a view-local key would silently drop
        the measurement. The substitutor must raise instead.
        """
        formal = _make_array_value("formal")
        root = _make_array_value("root", shape_vals=(_make_const_value("len", 4),))
        # Symbolic slice_start (plain Value, no const) blocks root-index folding.
        view = ArrayValue(
            type=QubitType(),
            name="view",
            shape=(_make_value("view_len"),),
            slice_of=root,
            slice_start=_make_value("start"),
            slice_step=_make_const_value("step", 2),
        )
        result_type = QFixedType(integer_bits=0, fractional_bits=2)
        cast_result = (
            Value(type=result_type, name="qf")
            .with_cast_metadata(
                source_uuid=formal.uuid,
                source_logical_id=formal.logical_id,
                qubit_uuids=[f"{formal.uuid}_0", f"{formal.uuid}_1"],
                qubit_logical_ids=[
                    f"{formal.logical_id}_0",
                    f"{formal.logical_id}_1",
                ],
            )
            .with_qfixed_metadata(
                qubit_uuids=[f"{formal.uuid}_0", f"{formal.uuid}_1"],
                num_bits=2,
                int_bits=0,
            )
        )
        op = CastOperation(
            operands=[formal],
            results=[cast_result],
            source_type=QubitType(),
            target_type=result_type,
            qubit_mapping=[f"{formal.uuid}_0", f"{formal.uuid}_1"],
        )

        with pytest.raises(ValueError, match="symbolic slice bounds"):
            ValueSubstitutor({formal.uuid: view}).substitute_operation(op)

    def test_uuid_remapper_clones_merge_output_carriers_from_branch_body(
        self,
    ) -> None:
        """If-merge output carriers track arrays first seen inside branch bodies.

        IfOperation results are merge outputs whose carrier metadata references
        the array cast inside the branches. The array's first appearance is
        inside the nested bodies, so cloning must fill the remap tables from
        the bodies before remapping the result metadata.
        """
        source = _make_array_value("q")
        condition = _make_value("b", type_cls=BitType)
        result_type = QFixedType(integer_bits=0, fractional_bits=2)

        def branch_cast(label: str) -> CastOperation:
            """Build a CastOperation over ``source`` with carrier metadata.

            Args:
                label (str): Name for the cast result value.

            Returns:
                CastOperation: Cast of ``source`` to QFixed carrying the
                    two composite carrier keys for ``source``'s elements.
            """
            result = (
                Value(type=result_type, name=label)
                .with_cast_metadata(
                    source_uuid=source.uuid,
                    source_logical_id=source.logical_id,
                    qubit_uuids=[f"{source.uuid}_0", f"{source.uuid}_1"],
                    qubit_logical_ids=[
                        f"{source.logical_id}_0",
                        f"{source.logical_id}_1",
                    ],
                )
                .with_qfixed_metadata(
                    qubit_uuids=[f"{source.uuid}_0", f"{source.uuid}_1"],
                    num_bits=2,
                    int_bits=0,
                )
            )
            return CastOperation(
                operands=[source],
                results=[result],
                source_type=QubitType(),
                target_type=result_type,
                qubit_mapping=[f"{source.uuid}_0", f"{source.uuid}_1"],
            )

        cast_true = branch_cast("qf_true")
        cast_false = branch_cast("qf_false")
        merge_out = Value(type=result_type, name="qf_merge").with_qfixed_metadata(
            qubit_uuids=[f"{source.uuid}_0", f"{source.uuid}_1"],
            num_bits=2,
            int_bits=0,
        )
        if_op = IfOperation(
            operands=[condition],
            true_operations=[cast_true],
            false_operations=[cast_false],
        )
        if_op.add_merge(cast_true.results[0], cast_false.results[0], merge_out)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(if_op)

        assert isinstance(cloned, IfOperation)
        cloned_source_uuid = remapper.uuid_remap[source.uuid]
        cloned_out = cloned.results[0]
        assert cloned_out.get_qfixed_qubit_uuids() == (
            f"{cloned_source_uuid}_0",
            f"{cloned_source_uuid}_1",
        )
        # The merged output and the operation result stay the same clone.
        merges = list(cloned.iter_merges())
        assert merges[0].result is cloned_out


class TestArrayValueShapeSubstitution:
    """Tests that ValueSubstitutor substitutes ArrayValue.shape values."""

    def test_shape_dimension_is_substituted(self) -> None:
        """Shape dimension referenced in value_map is substituted."""
        old_dim = _make_value("num_qubits")
        new_dim = _make_const_value("n", const=4)
        arr = _make_array_value("q", shape_vals=(old_dim,))

        sub = ValueSubstitutor({old_dim.uuid: new_dim})
        result = sub.substitute_value(arr)

        assert isinstance(result, ArrayValue)
        assert len(result.shape) == 1
        assert result.shape[0].uuid == new_dim.uuid
        assert result.shape[0].get_const() == 4

    def test_shape_not_in_map_is_unchanged(self) -> None:
        """Shape dimension not in value_map is preserved."""
        dim = _make_value("n")
        arr = _make_array_value("q", shape_vals=(dim,))

        sub = ValueSubstitutor({})
        result = sub.substitute_value(arr)

        assert isinstance(result, ArrayValue)
        assert result.shape[0].uuid == dim.uuid

    def test_partial_shape_substitution(self) -> None:
        """Only matched shape dimensions are substituted."""
        dim0 = _make_value("rows")
        dim1 = _make_value("cols")
        new_dim0 = _make_const_value("r", const=3)
        arr = _make_array_value("m", shape_vals=(dim0, dim1))

        sub = ValueSubstitutor({dim0.uuid: new_dim0})
        result = sub.substitute_value(arr)

        assert isinstance(result, ArrayValue)
        assert result.shape[0].uuid == new_dim0.uuid
        assert result.shape[1].uuid == dim1.uuid


class TestNestedArrayIndexSubstitution:
    """Test recursive substitution through nested array element indices."""

    def test_loop_index_inside_nested_array_element_is_substituted(self) -> None:
        """A mapped loop index reaches an array element used as another index."""
        loop_index = _make_value("k")
        zero = _make_const_value("zero", const=0)
        one = _make_const_value("one", const=1)
        matrix = _make_array_value(
            "indices",
            shape_vals=(
                _make_const_value("rows", 1),
                _make_const_value("cols", 2),
            ),
            type_cls=UIntType,
        )
        nested_index = Value(
            type=UIntType(),
            name="indices[k,1]",
            parent_array=matrix,
            element_indices=(loop_index, one),
        )
        qubits = _make_array_value("q")
        operand = Value(
            type=QubitType(),
            name="q[indices[k,1]]",
            parent_array=qubits,
            element_indices=(nested_index,),
        )

        substituted = ValueSubstitutor(
            {loop_index.uuid: zero}, transitive=True
        ).substitute_value(operand)

        assert isinstance(substituted, Value)
        resolved_nested = substituted.element_indices[0]
        assert resolved_nested.element_indices[0].uuid == zero.uuid
        assert resolved_nested.element_indices[1].uuid == one.uuid


# ===========================================================================
# Bug #6: IfOperation merge-slot cloning and substitution
# ===========================================================================


class TestIfOperationMergeCloning:
    """Tests that UUIDRemapper clones IfOperation merge slots."""

    def test_merges_are_cloned(self) -> None:
        """Cloning IfOperation should also clone its merge slots."""
        cond = _make_value("cond", BitType)
        true_q = _make_value("q_true", QubitType)
        false_q = _make_value("q_false", QubitType)
        merge_output = _make_value("q_merge", QubitType)

        if_op = IfOperation(
            operands=[cond],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(true_q, false_q, merge_output)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(if_op)

        assert isinstance(cloned, IfOperation)
        merges = list(cloned.iter_merges())
        assert len(merges) == 1
        # Merged output should have a fresh UUID
        assert merges[0].result.uuid != merge_output.uuid

    def test_empty_merges_is_preserved(self) -> None:
        """IfOperation with no merge slots is cloned without error."""
        cond = _make_value("cond", BitType)

        if_op = IfOperation(
            operands=[cond],
            true_operations=[],
            false_operations=[],
        )

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(if_op)

        assert isinstance(cloned, IfOperation)
        assert len(list(cloned.iter_merges())) == 0

    def test_multiple_merges_are_all_cloned(self) -> None:
        """IfOperation with multiple merge slots clones each one."""
        cond = _make_value("cond", BitType)

        if_op = IfOperation(
            operands=[cond],
            true_operations=[],
            false_operations=[],
        )
        merge_outputs = []
        for i in range(3):
            t = _make_value(f"t{i}", QubitType)
            f = _make_value(f"f{i}", QubitType)
            out = _make_value(f"merge{i}", QubitType)
            merge_outputs.append(out)
            if_op.add_merge(t, f, out)

        remapper = UUIDRemapper()
        cloned = remapper.clone_operation(if_op)

        assert isinstance(cloned, IfOperation)
        merges = list(cloned.iter_merges())
        assert len(merges) == 3
        for orig_out, cloned_merge in zip(merge_outputs, merges):
            assert cloned_merge.result.uuid != orig_out.uuid


class TestIfOperationMergeSubstitution:
    """Tests that ValueSubstitutor substitutes merge slots inside IfOperation."""

    def test_merge_operands_are_substituted(self) -> None:
        """ValueSubstitutor should substitute values inside merge slots."""
        cond = _make_value("cond", BitType)
        old_q = _make_value("q_old", QubitType)
        new_q = _make_value("q_new", QubitType)
        false_q = _make_value("q_false", QubitType)
        merge_output = _make_value("q_merge", QubitType)

        if_op = IfOperation(
            operands=[cond],
            true_operations=[],
            false_operations=[],
        )
        if_op.add_merge(old_q, false_q, merge_output)

        sub = ValueSubstitutor({old_q.uuid: new_q})
        result = sub.substitute_operation(if_op)

        assert isinstance(result, IfOperation)
        merges = list(result.iter_merges())
        assert len(merges) == 1
        # The true-branch value should be substituted
        assert merges[0].true_value.uuid == new_q.uuid
