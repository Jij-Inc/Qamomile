"""Validate shot context in reachable circuit-family bodies."""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.callable import InvokeOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import ControlledUOperation
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.circuit.ir.operation.select import SelectOperation
from qamomile.circuit.ir.operation.shot import GetCurrentShotOperation
from qamomile.circuit.ir.value import ValueBase
from qamomile.circuit.transpiler.block_parameter_binding import (
    block_parameter_binding_keys,
    pair_block_parameter_operands,
)
from qamomile.circuit.transpiler.errors import TargetCapabilityError
from qamomile.circuit.transpiler.passes.compile_time_if_lowering import (
    lower_compile_time_ifs_preserving_loop_conditions,
)
from qamomile.circuit.transpiler.passes.control_flow_reachability import (
    reachable_nested_regions,
)
from qamomile.circuit.transpiler.value_resolver import ValueResolver


def validate_circuit_shot_context(
    block: Block, bindings: dict[str, Any] | None, *, target: str
) -> None:
    """Reject shot context only in reachable circuit-family bodies.

    Circuit lowering selects engine-generic callable decompositions using
    the ``circuit_ir`` selector. Backend-specific alternatives are metadata
    for other consumers and must not impose their capabilities here.
    Specialize operation-owned bodies in their own parameter scope before
    inspecting them, just as the entrypoint is inspected after pruning.

    Args:
        block (Block): Partially evaluated circuit entrypoint.
        bindings (dict[str, Any] | None): Compile-time entrypoint bindings.
        target (str): Target name used in capability errors.

    Raises:
        TargetCapabilityError: If a reachable selected body reads the shot.
        ValidationError: If nested compile-time specialization is invalid.
        ValueError: If a selected callable body violates its call contract.
    """
    bodies = [(block, bindings or {}, frozenset[int]())]
    while bodies:
        body, scope, ancestors = bodies.pop()
        if id(body) in ancestors:
            continue
        ancestors = ancestors | {id(body)}
        specialized = lower_compile_time_ifs_preserving_loop_conditions(body, scope)
        resolver = ValueResolver(context=scope, bindings=scope)
        operations = list(reversed(specialized.operations))
        while operations:
            op = operations.pop()
            if isinstance(op, HasNestedOps):
                for region in reversed(reachable_nested_regions(op)):
                    operations.extend(reversed(region.operations))
            if isinstance(op, GetCurrentShotOperation):
                raise TargetCapabilityError(
                    f"{target} does not support get_current_shot(); use the HUGR target",
                    target=target,
                    operation="GetCurrentShotOperation",
                )
            nested: list[tuple[Block, tuple[ValueBase, ...]]] = []
            if isinstance(op, InvokeOperation):
                selection = op.select_body(engine="circuit_ir")
                if selection.body is not None:
                    nested.append((selection.body, selection.operands))
            elif isinstance(op, SelectOperation):
                nested.extend(
                    (case, tuple(op.param_operands)) for case in op.case_blocks
                )
            elif isinstance(op, ControlledUOperation) and op.block is not None:
                nested.append((op.block, tuple(op.operands)))
            elif isinstance(op, InverseBlockOperation):
                selected = op.implementation_block
                if selected is not None:
                    nested.append((selected, tuple(op.parameters)))
            for nested_body, operands in nested:
                parameters = [
                    value for value in operands if not value.type.is_quantum()
                ]
                nested_scope = {}
                for formal, actual in pair_block_parameter_operands(
                    nested_body, parameters
                ):
                    concrete = resolver.resolve(actual)
                    if concrete is not None:
                        for key in block_parameter_binding_keys(formal):
                            nested_scope[key] = concrete
                bodies.append((nested_body, nested_scope, ancestors))
