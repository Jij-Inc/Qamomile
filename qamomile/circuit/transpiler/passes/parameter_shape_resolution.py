"""Compile-time resolution of symbolic Vector parameter shape dimensions.

Qiskit / CUDA-Q / QuriParts circuits are fixed-structure at compile time:
loop bounds and array lengths must be concrete to emit a valid circuit.
Top-level ``Vector[Float]`` / ``Vector[UInt]`` parameters are created with
symbolic ``{name}_dim{i}`` Values (see
``qamomile.circuit.frontend.func_to_block.create_dummy_input``), so when a
kernel queries ``arr.shape[i]`` the IR references those symbolic Values in
loop operands, array allocations, etc.

This pass walks the HIERARCHICAL block before inlining and, for every
input array whose name is bound to a concrete array-like value in
``bindings``, replaces its symbolic shape dim Values with constant Values
holding the runtime length. After this pass the downstream pipeline can
unroll ``for i in qmc.range(arr.shape[0])`` loops normally.

Parameters without a concrete binding (e.g. the library QAOA pattern where
only ``p`` is bound and ``gammas.shape`` is never queried) are left
untouched — their symbolic dims simply do not flow into any compile-time
structure decision, so they are harmless.
"""

from __future__ import annotations

from typing import Any, Sequence, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.value_mapping import ValueSubstitutor


def _extract_concrete_shape(binding: Any) -> tuple[int, ...] | None:
    """Return the shape tuple of a concrete array-like binding, or ``None``.

    Supports numpy arrays (``.shape``) and Python sequences (``len()``).
    Returns ``None`` for scalars, dicts, ``None``, etc. so the caller can
    leave the symbolic shape untouched.
    """
    if binding is None:
        return None
    if hasattr(binding, "shape"):
        try:
            return tuple(int(d) for d in binding.shape)
        except (TypeError, ValueError):
            return None
    if isinstance(binding, (list, tuple)):
        return (len(binding),)
    return None


class ParameterShapeResolutionPass(Pass[Block, Block]):
    """Substitute symbolic parameter array shape dims with concrete constants.

    Input: ``BlockKind.HIERARCHICAL`` (the ``transpile()`` path, before
    ``InlinePass``) or ``BlockKind.AFFINE`` / ``BlockKind.ANALYZED``
    (the ``transpile_block()`` path, where a deserialized block receives
    its array bindings after the fact). The substitution itself is
    kind-agnostic — it only rewrites shape-dim Values by UUID.
    Output: same block kind, with matching shape dim Values constant-folded.
    """

    _SUPPORTED_KINDS = (
        BlockKind.HIERARCHICAL,
        BlockKind.AFFINE,
        BlockKind.ANALYZED,
    )

    def __init__(self, bindings: dict[str, Any] | None = None) -> None:
        """Initialize the pass with the compile-time bindings.

        Args:
            bindings (dict[str, Any] | None): Compile-time parameter
                bindings keyed by kernel argument name. Defaults to
                None, meaning no substitutions are performed.
        """
        self._bindings = bindings or {}

    @property
    def name(self) -> str:
        """Return the pass name used in diagnostics.

        Returns:
            str: The literal ``"parameter_shape_resolution"``.
        """
        return "parameter_shape_resolution"

    def run(self, input: Block) -> Block:
        """Fold bound parameter-array shape dims to constants.

        Args:
            input (Block): The block to rewrite. Must be
                ``HIERARCHICAL``, ``AFFINE``, or ``ANALYZED``.

        Returns:
            Block: The block with every resolvable symbolic shape dim
                replaced by a constant, preserving ``input.kind``.

        Raises:
            ValidationError: If ``input.kind`` is ``TRACED`` (the block
                must at least be a fully traced kernel body).
        """
        if input.kind not in self._SUPPORTED_KINDS:
            raise ValidationError(
                f"ParameterShapeResolutionPass expects HIERARCHICAL, AFFINE, "
                f"or ANALYZED block, got {input.kind}"
            )

        if not self._bindings:
            return input

        substitutions = self._build_substitutions(input.input_values)
        if not substitutions:
            return input

        substitutor = ValueSubstitutor(substitutions)

        new_operations = self._walk_and_substitute(input.operations, substitutor)
        # Input/output values of a Block are always Value (or ArrayValue, which
        # is-a Value); ValueSubstitutor.substitute_value returns ValueBase for
        # generality, so narrow here for the Block constructor.
        new_input_values = [
            cast(Value, substitutor.substitute_value(v)) for v in input.input_values
        ]
        new_output_values = [
            cast(Value, substitutor.substitute_value(v)) for v in input.output_values
        ]

        return Block(
            name=input.name,
            label_args=input.label_args,
            input_values=new_input_values,
            output_values=new_output_values,
            operations=new_operations,
            kind=input.kind,
            parameters=input.parameters,
            param_slots=input.param_slots,
        )

    def _build_substitutions(
        self, input_values: Sequence[ValueBase]
    ) -> dict[str, ValueBase]:
        """Return uuid -> concrete Value map for every resolvable shape dim."""
        substitutions: dict[str, ValueBase] = {}
        for v in input_values:
            if not isinstance(v, ArrayValue):
                continue
            if v.name not in self._bindings:
                continue
            if not v.shape:
                continue

            concrete_shape = _extract_concrete_shape(self._bindings[v.name])
            if concrete_shape is None:
                continue

            for dim_index, dim_value in enumerate(v.shape):
                if dim_index >= len(concrete_shape):
                    break
                if dim_value.is_constant():
                    continue
                const_dim = Value(
                    type=UIntType(),
                    name=dim_value.name,
                ).with_const(int(concrete_shape[dim_index]))
                substitutions[dim_value.uuid] = const_dim
        return substitutions

    def _walk_and_substitute(
        self,
        operations: list[Operation],
        substitutor: ValueSubstitutor,
    ) -> list[Operation]:
        """Apply substitution to each op and recurse into nested op lists."""
        result: list[Operation] = []
        for op in operations:
            substituted = substitutor.substitute_operation(op)
            if isinstance(substituted, HasNestedOps):
                new_lists = [
                    self._walk_and_substitute(op_list, substitutor)
                    for op_list in substituted.nested_op_lists()
                ]
                substituted = substituted.rebuild_nested(new_lists)
            result.append(substituted)
        return result
