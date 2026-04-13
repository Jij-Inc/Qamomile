"""Validation pass: reject unresolved parameter shape dims in loop bounds.

Qamomile circuits are fixed-structure at compile time. Any loop whose
bound is a symbolic parameter array shape dimension (e.g.
``gamma_dim0``) must have that dimension resolved before emit — either
by ``ParameterShapeResolutionPass`` folding it from a concrete binding,
or by the user binding the dimension explicitly.

When an unresolved symbolic shape dim reaches a ``ForOperation`` operand,
this pass raises a ``CompileError`` with an actionable message that
points to the offending parameter and suggests concrete fixes. This is
the user-facing counterpart to the defensive hard fail in
``emit_for_unrolled``: it runs earlier (after ``analyze``) and can
provide clean diagnostics instead of a cryptic emit-time error.

The library QAOA pattern (``p`` bound to an int, ``for layer in
qmc.range(p)``, ``gammas.shape`` never queried) is unaffected — the
pass only trips when an unresolved symbolic shape dim is actually used
as loop bound / allocation size.
"""

from __future__ import annotations

import re

from qamomile.circuit.ir.block import Block, BlockKind
from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.operation.control_flow import ForOperation, HasNestedOps
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.errors import QamomileCompileError
from qamomile.circuit.transpiler.passes import Pass

# Shape dims created by ``func_to_block.create_dummy_input`` follow the
# ``{name}_dim{i}`` naming convention. Matching this name lets us produce
# an actionable diagnostic identifying the parameter array whose shape
# failed to resolve.
_SHAPE_DIM_NAME_PATTERN = re.compile(r"^(?P<array>.+)_dim(?P<index>\d+)$")


def _looks_like_parameter_shape_dim(v: Value) -> tuple[str, int] | None:
    """If *v* looks like a parameter array shape dim, return (array, index)."""
    if v.is_constant():
        return None
    if not v.name:
        return None
    match = _SHAPE_DIM_NAME_PATTERN.match(v.name)
    if match is None:
        return None
    return match.group("array"), int(match.group("index"))


def _format_actionable_error(
    array_name: str,
    dim_index: int,
    location_hint: str,
) -> str:
    return (
        f"Parameter array '{array_name}' has unresolved shape dimension "
        f"{dim_index}: {location_hint} depends on its length at compile "
        f"time, but no concrete value is available.\n\n"
        f"Qamomile circuits are fixed-structure at compile time — loop "
        f"bounds and array lengths must be concrete before emit. "
        f"Pick one of the following fixes:\n"
        f"  1. Bind a concrete array at transpile time so its shape is "
        f"known:\n"
        f"       transpiler.transpile(..., bindings={{'{array_name}': "
        f"[...]}})\n"
        f"     This also lets you keep it in ``parameters=[...]``, in "
        f"which case the values become backend parameters but the "
        f"length is fixed at compile time.\n"
        f"  2. Use a separate compile-time loop counter instead of "
        f"querying the shape:\n"
        f"       def kernel(p: qm.UInt, {array_name}: qm.Vector[qm.Float], ...):\n"
        f"           for layer in qm.range(p):\n"
        f"               ... {array_name}[layer] ...\n"
        f"       transpiler.transpile(..., bindings={{'p': 2}}, "
        f"parameters=['{array_name}'])"
    )


class SymbolicShapeValidationPass(Pass[Block, Block]):
    """Reject unresolved parameter shape dims in compile-time structure ops.

    Input:  ``BlockKind.ANALYZED`` (runs after ``AnalyzePass``).
    Output: same block unchanged, or raises ``QamomileCompileError``.
    """

    @property
    def name(self) -> str:
        return "symbolic_shape_validation"

    def run(self, input: Block) -> Block:
        if input.kind != BlockKind.ANALYZED:
            # Pass is defensive — only runs on analyzed blocks. Skip
            # otherwise to avoid false positives on partially-built IR.
            return input

        self._walk(input.operations)
        return input

    def _walk(self, operations: list[Operation]) -> None:
        for op in operations:
            self._check_op(op)
            if isinstance(op, HasNestedOps):
                for nested in op.nested_op_lists():
                    self._walk(nested)

    def _check_op(self, op: Operation) -> None:
        if isinstance(op, ForOperation):
            self._check_for_operation(op)

    def _check_for_operation(self, op: ForOperation) -> None:
        labels = ("start", "stop", "step")
        for label, operand in zip(labels, op.operands):
            if not isinstance(operand, Value):
                continue
            shape_info = _looks_like_parameter_shape_dim(operand)
            if shape_info is None:
                continue
            array_name, dim_index = shape_info
            location = f"a for-loop '{label}' bound (loop variable '{op.loop_var}')"
            raise QamomileCompileError(
                _format_actionable_error(array_name, dim_index, location)
            )
