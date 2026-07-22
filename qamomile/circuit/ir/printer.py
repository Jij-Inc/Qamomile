"""Text pretty-printer for the ``Block`` IR.

This module provides a contributor-facing textual dump of the intermediate
representation, similar in spirit to MLIR's textual IR format.  Useful for
debugging the transpiler pipeline by inspecting the block at each stage
(``HIERARCHICAL`` / ``AFFINE`` / ``ANALYZED``).

Example:
    >>> from qamomile.circuit.ir import pretty_print_block
    >>> block = transpiler.to_block(my_kernel, bindings={"n": 3})
    >>> print(pretty_print_block(block))
    block my_kernel [HIERARCHICAL] (n: UIntType) -> Vector[BitType]
      ...

The output is intended for human inspection, not for machine parsing; its
format may change between Qamomile releases.
"""

from __future__ import annotations

from typing import Any

from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation import (
    CallTransform,
    CastOperation,
    ConcreteControlledU,
    ControlledUOperation,
    ForItemsOperation,
    GateOperation,
    GlobalPhaseOperation,
    InverseBlockOperation,
    InvokeOperation,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
    Operation,
    ProjectOperation,
    ResetOperation,
    ReturnOperation,
    SelectOperation,
)
from qamomile.circuit.ir.operation.arithmetic_operations import (
    BinOp,
    CompOp,
    CondOp,
    NotOp,
    UnaryMathOp,
)
from qamomile.circuit.ir.operation.classical_ops import DecodeQFixedOperation
from qamomile.circuit.ir.operation.control_flow import (
    ForOperation,
    IfMerge,
    IfOperation,
    WhileOperation,
)
from qamomile.circuit.ir.operation.expval import ExpvalOp
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp
from qamomile.circuit.ir.value import (
    ArrayValue,
    DictValue,
    TupleValue,
    Value,
    ValueLike,
)

__all__ = ["pretty_print_block", "format_value"]


_INDENT = "  "


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pretty_print_block(block: Block, *, depth: int = 0) -> str:
    """Return a MLIR-style textual dump of *block*.

    Args:
        block: The ``Block`` to format.  Works on any ``BlockKind``.
        depth: How many levels of callable bodies to expand inline.
            ``0`` (default) shows only the callable name and I/O. Positive
            values expand ``InvokeOperation`` bodies recursively, decrementing
            the allowance at each step. Useful for seeing what ``inline`` will
            produce without actually running the pass.

    Returns:
        A newline-separated string.  The format is for human debugging and
        is not guaranteed to be stable across releases.
    """
    printer = _BlockPrinter(depth=depth)
    printer.emit_block(block, indent=0)
    return "\n".join(printer.lines)


def format_value(value: Any) -> str:
    """Format an IR value reference as ``%name@vN``.

    Handles ``Value``, ``ArrayValue``, ``TupleValue``, ``DictValue``, and
    array-element ``Value``s (rendered as ``%parent[i]@vN``).  Constants
    and parameters are shown with their tagged metadata when available.
    Falls back to ``repr()`` for unrecognised inputs so callers can use
    this helper for any operand-like field without a type switch.
    """
    return _format_value(value)


# ---------------------------------------------------------------------------
# Internal printer
# ---------------------------------------------------------------------------


class _BlockPrinter:
    """Builds the textual output line by line."""

    def __init__(self, *, depth: int) -> None:
        self.depth = depth
        self.lines: list[str] = []

    # ------------------------------------------------------------------
    # Block-level
    # ------------------------------------------------------------------

    def emit_block(self, block: Block, *, indent: int) -> None:
        self._emit_header(block, indent=indent)
        pad = _INDENT * indent
        if block.parameters:
            self.lines.append(
                f"{pad}{_INDENT}parameters: [{', '.join(block.parameters)}]"
            )
        for slot in block.static_bindings:
            fields = ", ".join(
                f"{field.name}={_format_value(field.value)}" for field in slot.fields
            )
            suffix = f" ({fields})" if fields else ""
            self.lines.append(
                f"{pad}{_INDENT}static binding {slot.name}: {slot.type_key}{suffix}"
            )
        body_indent = indent + 1
        self._emit_ops(block.operations, indent=body_indent)
        self.lines.append(f"{pad}}}")

    def _emit_header(self, block: Block, *, indent: int) -> None:
        pad = _INDENT * indent
        name = block.name or "<anonymous>"
        kind = block.kind.name
        inputs = ", ".join(_format_param(v) for v in block.input_values)
        outputs = _format_outputs(block.output_values)
        self.lines.append(f"{pad}block {name} [{kind}] ({inputs}){outputs} {{")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    def _emit_ops(self, ops: list[Operation], *, indent: int) -> None:
        for op in ops:
            self._emit_op(op, indent=indent)

    def _emit_op(self, op: Operation, *, indent: int) -> None:
        pad = _INDENT * indent
        if isinstance(op, ForOperation):
            self._emit_for(op, indent=indent, pad=pad)
        elif isinstance(op, ForItemsOperation):
            self._emit_for_items(op, indent=indent, pad=pad)
        elif isinstance(op, IfOperation):
            self._emit_if(op, indent=indent, pad=pad)
        elif isinstance(op, WhileOperation):
            self._emit_while(op, indent=indent, pad=pad)
        elif isinstance(op, InvokeOperation):
            self._emit_invoke(op, indent=indent, pad=pad)
        elif isinstance(op, SelectOperation):
            self._emit_select(op, indent=indent, pad=pad)
        else:
            self.lines.append(pad + _format_flat_op(op))

    # -- control flow ----------------------------------------------------

    def _region_args_suffix(self, op) -> str:
        """Format a loop's region arguments as a header suffix.

        Args:
            op: A loop operation (``ForOperation`` / ``ForItemsOperation``
                / ``WhileOperation``) carrying ``region_args``.

        Returns:
            str: `` iter_args(name = %init -> %result, ...)`` when the
                loop carries region arguments, or ``""`` otherwise.
        """
        if not getattr(op, "region_args", ()):
            return ""
        parts = [
            f"{a.var_name or '_'} = {_format_value(a.init)} -> {_format_value(a.result)}"
            for a in op.region_args
        ]
        return " iter_args(" + ", ".join(parts) + ")"

    def _emit_for(self, op: ForOperation, *, indent: int, pad: str) -> None:
        start = _format_value(op.operands[0]) if len(op.operands) > 0 else "?"
        stop = _format_value(op.operands[1]) if len(op.operands) > 1 else "?"
        step = _format_value(op.operands[2]) if len(op.operands) > 2 else "?"
        lv = op.loop_var or "_"
        iter_args = self._region_args_suffix(op)
        self.lines.append(
            f"{pad}for %{lv} in range({start}, {stop}, {step}){iter_args} {{"
        )
        self._emit_ops(op.operations, indent=indent + 1)
        self.lines.append(f"{pad}}}{_format_region_yields(op)}")

    def _emit_for_items(self, op: ForItemsOperation, *, indent: int, pad: str) -> None:
        keys = ", ".join(f"%{k}" for k in op.key_vars) if op.key_vars else "%k"
        val = f"%{op.value_var}" if op.value_var else "%v"
        iterable = _format_value(op.operands[0]) if op.operands else "<iterable>"
        iter_args = self._region_args_suffix(op)
        header = f"{pad}for ({keys}), {val} in items({iterable}){iter_args} {{"
        self.lines.append(header)
        self._emit_ops(op.operations, indent=indent + 1)
        self.lines.append(f"{pad}}}{_format_region_yields(op)}")

    def _emit_if(self, op: IfOperation, *, indent: int, pad: str) -> None:
        cond = _format_value(op.operands[0]) if op.operands else "<cond>"
        self.lines.append(f"{pad}if {cond} {{")
        self._emit_ops(op.true_operations, indent=indent + 1)
        if op.false_operations:
            self.lines.append(f"{pad}}} else {{")
            self._emit_ops(op.false_operations, indent=indent + 1)
        self.lines.append(f"{pad}}}")
        # Branch merges live at the same indent as the if-operation.
        for merge in op.iter_merges():
            self.lines.append(pad + _format_merge(op, merge))

    def _emit_while(self, op: WhileOperation, *, indent: int, pad: str) -> None:
        cond = _format_value(op.operands[0]) if op.operands else "<cond>"
        max_iter = (
            f" [max_iterations={op.max_iterations}]"
            if op.max_iterations is not None
            else ""
        )
        iter_args = self._region_args_suffix(op)
        self.lines.append(f"{pad}while {cond}{max_iter}{iter_args} {{")
        self._emit_ops(op.operations, indent=indent + 1)
        self.lines.append(f"{pad}}}{_format_region_yields(op)}")

    def _emit_invoke(self, op: InvokeOperation, *, indent: int, pad: str) -> None:
        """Emit one invocation with transform and activation metadata.

        Args:
            op (InvokeOperation): Invocation to print.
            indent (int): Current indentation depth for an expanded body.
            pad (str): Precomputed indentation prefix for the invocation line.
        """
        target = op.body
        name = op.name
        args = ", ".join(_format_value(v) for v in op.operands)
        rets = ", ".join(_format_value(v) for v in op.results)
        arrow = f" -> ({rets})" if rets else ""
        metadata = _format_invoke_metadata(op)
        if self.depth > 0 and target is not None:
            self.lines.append(f"{pad}invoke {name}({args}){arrow}{metadata} {{")
            child = _BlockPrinter(depth=self.depth - 1)
            child._emit_ops(target.operations, indent=indent + 1)
            self.lines.extend(child.lines)
            self.lines.append(f"{pad}}}")
        else:
            self.lines.append(f"{pad}invoke {name}({args}){arrow}{metadata}")

    def _emit_select(self, op: SelectOperation, *, indent: int, pad: str) -> None:
        """Emit a SELECT with structural metadata and optional case bodies.

        Args:
            op (SelectOperation): SELECT operation to print.
            indent (int): Current indentation depth for expanded case bodies.
            pad (str): Precomputed indentation prefix for the SELECT line.

        Returns:
            None: Appends the formatted SELECT to ``self.lines``.
        """
        width = op.num_index_qubits
        width_text = _format_value(width) if isinstance(width, Value) else str(width)
        cases = ", ".join(
            f"{index}:{block.name or '<anonymous>'}"
            for index, block in enumerate(op.case_blocks)
        )
        args = ", ".join(_format_value(value) for value in op.operands)
        header = (
            f"{_format_results(op.results)} = select({args}) "
            f"[index_width={width_text}, index_args={op.num_index_args}, "
            f"cases=[{cases}]]"
        )
        if self.depth <= 0:
            self.lines.append(pad + header)
            return

        self.lines.append(pad + header + " {")
        case_pad = _INDENT * (indent + 1)
        for index, block in enumerate(op.case_blocks):
            name = block.name or "<anonymous>"
            self.lines.append(f"{case_pad}case {index} {name} {{")
            child = _BlockPrinter(depth=self.depth - 1)
            child._emit_ops(block.operations, indent=indent + 2)
            self.lines.extend(child.lines)
            self.lines.append(f"{case_pad}}}")
        self.lines.append(f"{pad}}}")


# ---------------------------------------------------------------------------
# Single-operation formatting (no indentation, no recursion)
# ---------------------------------------------------------------------------


def _format_flat_op(op: Operation) -> str:
    """Format a non-control-flow operation as a single line.

    Args:
        op (Operation): Operation to render.

    Returns:
        str: Single-line textual representation of the operation.
    """
    if isinstance(op, GateOperation):
        return _format_gate(op)
    if isinstance(op, ControlledUOperation):
        return _format_controlled(op)
    if isinstance(op, InvokeOperation):
        return _format_invoke(op)
    if isinstance(op, InverseBlockOperation):
        return _format_inverse_block(op)
    if isinstance(op, GlobalPhaseOperation):
        return _format_global_phase(op)
    if isinstance(op, MeasureOperation):
        return _format_measure(op, "measure")
    if isinstance(op, ProjectOperation):
        return _format_measure(op, f"project_{op.axis}")
    if isinstance(op, ResetOperation):
        return _format_measure(op, "reset")
    if isinstance(op, MeasureVectorOperation):
        return _format_measure(op, "measure_vector")
    if isinstance(op, MeasureQFixedOperation):
        return _format_measure(op, "measure_qfixed")
    if isinstance(op, DecodeQFixedOperation):
        return _format_simple(op, "decode_qfixed")
    if isinstance(op, CastOperation):
        return _format_cast(op)
    if isinstance(op, PauliEvolveOp):
        return _format_pauli_evolve(op)
    if isinstance(op, ExpvalOp):
        return _format_simple(op, "expval")
    if isinstance(op, BinOp):
        return _format_binary(op, _BINOP_SYMBOLS)
    if isinstance(op, UnaryMathOp):
        assert op.kind is not None
        return (
            f"{_format_results(op.results)} = "
            f"{op.kind.name.lower()}({_format_value(op.input)})"
        )
    if isinstance(op, CompOp):
        return _format_binary(op, _COMPOP_SYMBOLS)
    if isinstance(op, CondOp):
        return _format_binary(op, _CONDOP_SYMBOLS)
    if isinstance(op, NotOp):
        return f"{_format_results(op.results)} = not {_format_value(op.operands[0])}"
    if isinstance(op, ReturnOperation):
        args = ", ".join(_format_value(v) for v in op.operands)
        return f"return {args}" if args else "return"
    return _format_simple(op, type(op).__name__)


_BINOP_SYMBOLS: dict[Any, str] = {}
_COMPOP_SYMBOLS: dict[Any, str] = {}
_CONDOP_SYMBOLS: dict[Any, str] = {}


def _init_op_symbols() -> None:
    """Lazily populate the symbol tables after imports settle."""
    if _BINOP_SYMBOLS:
        return
    from qamomile.circuit.ir.operation.arithmetic_operations import (
        BinOpKind,
        CompOpKind,
        CondOpKind,
    )

    _BINOP_SYMBOLS.update(
        {
            BinOpKind.ADD: "+",
            BinOpKind.SUB: "-",
            BinOpKind.MUL: "*",
            BinOpKind.DIV: "/",
            BinOpKind.FLOORDIV: "//",
            BinOpKind.MOD: "%",
            BinOpKind.POW: "**",
        }
    )
    _COMPOP_SYMBOLS.update(
        {
            CompOpKind.EQ: "==",
            CompOpKind.NEQ: "!=",
            CompOpKind.LT: "<",
            CompOpKind.LE: "<=",
            CompOpKind.GT: ">",
            CompOpKind.GE: ">=",
        }
    )
    _CONDOP_SYMBOLS.update(
        {
            CondOpKind.AND: "&&",
            CondOpKind.OR: "||",
        }
    )


def _format_gate(op: GateOperation) -> str:
    name = op.gate_type.name.lower() if op.gate_type is not None else "<gate>"
    qubits = ", ".join(_format_value(q) for q in op.qubit_operands)
    theta = op.theta
    if theta is not None:
        params = f"{qubits}, θ={_format_value(theta)}"
    else:
        params = qubits
    return f"{_format_results(op.results)} = {name}({params})"


def _format_controlled(op: ControlledUOperation) -> str:
    """Format a structural controlled-unitary operation.

    Args:
        op (ControlledUOperation): Controlled operation to format.

    Returns:
        str: Single-line controlled-unitary representation including a
            non-default activation value when present.
    """
    block_name = op.block.name if op.block is not None and op.block.name else "<block>"
    power = op.power
    power_str = (
        f", power={_format_value(power)}"
        if isinstance(power, Value)
        else f", power={power}"
    )
    control_value = (
        f", control_value={op.control_value}"
        if isinstance(op, ConcreteControlledU) and op.control_value is not None
        else ""
    )
    args = ", ".join(_format_value(v) for v in op.operands)
    return (
        f"{_format_results(op.results)} = controlled "
        f"{block_name}({args}{power_str}{control_value})"
    )


def _format_invoke(op: InvokeOperation) -> str:
    """Format a flat callable invocation.

    Args:
        op (InvokeOperation): Invocation to format.

    Returns:
        str: Single-line invocation including transform metadata.
    """
    args = ", ".join(_format_value(v) for v in op.operands)
    return (
        f"{_format_results(op.results)} = invoke {op.name}({args})"
        f"{_format_invoke_metadata(op)}"
    )


def _format_invoke_metadata(op: InvokeOperation) -> str:
    """Format transform metadata that changes invocation semantics.

    Args:
        op (InvokeOperation): Invocation whose transform metadata should be
            rendered.

    Returns:
        str: Bracketed transform metadata, or an empty string for a direct
            invocation.
    """
    if op.transform is CallTransform.DIRECT:
        return ""
    fields = [f"transform={op.transform.name}"]
    if op.transform is CallTransform.CONTROLLED:
        fields.append(f"controls={op.num_control_qubits}")
        if op.control_value is not None:
            fields.append(f"control_value={op.control_value}")
    return " [" + ", ".join(fields) + "]"


def _format_inverse_block(op: InverseBlockOperation) -> str:
    """Format a first-class inverse block operation.

    Args:
        op (InverseBlockOperation): Inverse operation to format.

    Returns:
        str: Single-line inverse representation including coherent-control
            metadata when present.
    """
    args = ", ".join(_format_value(v) for v in op.operands)
    fields: list[str] = []
    if op.num_control_qubits:
        fields.append(f"controls={op.num_control_qubits}")
        if op.control_value is not None:
            fields.append(f"control_value={op.control_value}")
    metadata = " [" + ", ".join(fields) + "]" if fields else ""
    return f"{_format_results(op.results)} = inverse {op.name}({args}){metadata}"


def _format_global_phase(op: GlobalPhaseOperation) -> str:
    """Format a zero-qubit global-phase operation.

    Args:
        op (GlobalPhaseOperation): Operation to format.

    Returns:
        str: ``global_phase(<phase>)`` textual representation.
    """
    return f"global_phase({_format_value(op.phase)})"


def _format_measure(op: Operation, mnemonic: str) -> str:
    args = ", ".join(_format_value(v) for v in op.operands)
    return f"{_format_results(op.results)} = {mnemonic}({args})"


def _format_cast(op: CastOperation) -> str:
    src = _format_value(op.operands[0]) if op.operands else "<src>"
    target = op.target_type.label() if op.target_type is not None else "<type>"
    return f"{_format_results(op.results)} = cast {src} to {target}"


def _format_pauli_evolve(op: PauliEvolveOp) -> str:
    parts = [_format_value(v) for v in op.operands]
    return f"{_format_results(op.results)} = pauli_evolve({', '.join(parts)})"


def _format_merge(if_op: IfOperation, merge: IfMerge) -> str:
    """Format one if branch-merge slot as a merge line.

    Args:
        if_op (IfOperation): The if-else owning the merge (condition
            source).
        merge (IfMerge): The merge slot to format.

    Returns:
        str: One ``result = merge(cond ? true : false)`` line.
    """
    cond = _format_value(if_op.condition) if if_op.operands else "<cond>"
    tv = _format_value(merge.true_value)
    fv = _format_value(merge.false_value)
    return f"{_format_results([merge.result])} = merge({cond} ? {tv} : {fv})"


def _format_region_yields(
    op: "ForOperation | ForItemsOperation | WhileOperation",
) -> str:
    """Format a loop's region arguments as a yield/result footer suffix.

    Args:
        op (ForOperation | ForItemsOperation | WhileOperation): The loop
            whose region arguments are rendered.

    Returns:
        str: `` yield(%yielded, ...) -> (%result, ...)`` (leading space
            included), or an empty string for a carry-free loop.
    """
    if not op.region_args:
        return ""
    yields = ", ".join(_format_value(arg.yielded) for arg in op.region_args)
    results = ", ".join(_format_value(arg.result) for arg in op.region_args)
    return f" yield({yields}) -> ({results})"


def _format_binary(op: Operation, table: dict[Any, str]) -> str:
    _init_op_symbols()
    kind = getattr(op, "kind", None)
    sym = table.get(kind, str(kind))
    lhs = _format_value(op.operands[0]) if op.operands else "<lhs>"
    rhs = _format_value(op.operands[1]) if len(op.operands) > 1 else "<rhs>"
    return f"{_format_results(op.results)} = {lhs} {sym} {rhs}"


def _format_simple(op: Operation, mnemonic: str) -> str:
    args = ", ".join(_format_value(v) for v in op.operands)
    rhs = f"{mnemonic}({args})"
    if op.results:
        return f"{_format_results(op.results)} = {rhs}"
    return rhs


# ---------------------------------------------------------------------------
# Value formatting
# ---------------------------------------------------------------------------


def _format_value(value: Any) -> str:
    if value is None:
        return "<none>"
    # Constant / parameter tagging takes priority for readability.
    if isinstance(value, Value) and value.is_constant():
        return f"const({value.get_const()})"
    if isinstance(value, Value) and value.is_parameter():
        return f"param({value.parameter_name()})"
    if isinstance(value, ArrayValue):
        return _format_version(value.name, value.version)
    if isinstance(value, Value):
        if value.parent_array is not None and value.element_indices:
            idx = ",".join(_format_value(i) for i in value.element_indices)
            return _format_version(f"{value.parent_array.name}[{idx}]", value.version)
        return _format_version(value.name, value.version)
    if isinstance(value, TupleValue):
        return "(" + ", ".join(_format_value(e) for e in value.elements) + ")"
    if isinstance(value, DictValue):
        return f"%{value.name}@dict"
    return repr(value)


def _format_version(name: str, version: int) -> str:
    safe_name = name or "_"
    return f"%{safe_name}@v{version}"


def _format_results(results: list[Value]) -> str:
    if not results:
        return "()"
    return ", ".join(_format_value(v) for v in results)


def _format_param(value: ValueLike) -> str:
    t = value.type.label() if value.type is not None else "?"
    return f"{value.name or '_'}: {t}"


def _format_outputs(outputs: list[ValueLike]) -> str:
    if not outputs:
        return ""
    if len(outputs) == 1:
        t = outputs[0].type.label() if outputs[0].type is not None else "?"
        return f" -> {t}"
    parts = [v.type.label() if v.type is not None else "?" for v in outputs]
    return " -> (" + ", ".join(parts) + ")"
