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

import dataclasses
from typing import Any, Sequence, cast

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.types.primitives import UIntType
from qamomile.circuit.ir.value import ArrayValue, Value, ValueBase, ValueLike
from qamomile.circuit.transpiler.errors import ValidationError
from qamomile.circuit.transpiler.passes import Pass
from qamomile.circuit.transpiler.passes.value_mapping import ValueSubstitutor


def _extract_concrete_shape(binding: Any) -> tuple[int, ...] | None:
    """Return the shape tuple of a concrete array-like binding, or ``None``.

    Supports numpy arrays (``.shape``) and Python sequences (``len()``).
    Returns ``None`` for scalars, dicts, ``None``, etc. so the caller can
    leave the symbolic shape untouched.

    Args:
        binding (Any): Candidate compile-time array binding.

    Returns:
        tuple[int, ...] | None: Concrete dimensions, or ``None`` when the
            binding is not array-like.
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

    Input: ``BlockKind.HIERARCHICAL`` (runs before ``InlinePass``).
    Output: same block kind, with matching shape dim Values constant-folded.
    """

    def __init__(self, bindings: dict[str, Any] | None = None) -> None:
        """Initialize parameter-shape resolution.

        Args:
            bindings (dict[str, Any] | None): Compile-time bindings keyed by
                entrypoint argument name. Defaults to ``None``.
        """
        self._bindings = bindings or {}

    @property
    def name(self) -> str:
        """Return the pass name.

        Returns:
            str: Stable diagnostic pass name.
        """
        return "parameter_shape_resolution"

    def run(self, input: Block) -> Block:
        """Replace resolvable symbolic array dimensions with constants.

        Args:
            input (Block): Hierarchical semantic block to rewrite.

        Returns:
            Block: Rewritten block preserving all display, ABI, parameter, and
                stage metadata.

        Raises:
            ValidationError: If ``input`` is not hierarchical.
        """
        if input.kind != BlockKind.HIERARCHICAL:
            raise ValidationError(
                f"ParameterShapeResolutionPass expects HIERARCHICAL block, "
                f"got {input.kind}"
            )

        if not self._bindings:
            return input

        substitutions = self._build_substitutions(input.input_values)
        if not substitutions:
            return input

        substitutor = ValueSubstitutor(substitutions)

        new_operations = self._walk_and_substitute(input.operations, substitutor)
        # Input/output values of a Block may be structural ValueLike objects;
        # ValueSubstitutor returns ValueBase for generality, so narrow here for
        # the Block constructor.
        new_input_values = [
            cast(ValueLike, substitutor.substitute_value(v)) for v in input.input_values
        ]
        new_output_values = [
            cast(ValueLike, substitutor.substitute_value(v))
            for v in input.output_values
        ]
        new_parameters = cast(
            "dict[str, Value]",
            {
                name: substitutor.substitute_value(value)
                for name, value in input.parameters.items()
            },
        )

        return dataclasses.replace(
            input,
            input_values=new_input_values,
            output_values=new_output_values,
            operations=new_operations,
            parameters=new_parameters,
        )

    def _build_substitutions(
        self, input_values: Sequence[ValueBase]
    ) -> dict[str, ValueBase]:
        """Build replacements for every resolvable shape dimension.

        Args:
            input_values (Sequence[ValueBase]): Entrypoint inputs whose array
                shapes may depend on bindings.

        Returns:
            dict[str, ValueBase]: Symbolic UUID to concrete dimension Value.
        """
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
        """Apply substitutions recursively to an operation region.

        Args:
            operations (list[Operation]): Operations in the current region.
            substitutor (ValueSubstitutor): Value-rewrite engine.

        Returns:
            list[Operation]: Rewritten operations preserving region structure.
        """
        result: list[Operation] = []
        for op in operations:
            substituted = substitutor.substitute_operation(op)
            if isinstance(substituted, HasNestedOps):
                new_regions = tuple(
                    dataclasses.replace(
                        region,
                        operations=tuple(
                            self._walk_and_substitute(
                                list(region.operations), substitutor
                            )
                        ),
                    )
                    for region in substituted.nested_regions()
                )
                substituted = substituted.rebuild_regions(new_regions)
            result.append(substituted)
        return result
