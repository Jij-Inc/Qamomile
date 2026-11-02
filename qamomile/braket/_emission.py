"""Emit target-neutral circuit projections as OpenQASM 3 source."""

from __future__ import annotations

from typing import cast

from qamomile.circuit import (
    CircuitBlock,
    CircuitInstruction,
    CircuitProgram,
    CircuitValue,
    ClassicalExpr,
    MeasurementBit,
    ParameterValue,
    scalar_parameters,
)
from qamomile.errors import CapabilityError


def emit_openqasm(program: CircuitProgram) -> str:
    """Render a logical circuit family as deterministic OpenQASM 3."""

    # Braket's OpenQASM dialect exposes its native gate vocabulary directly;
    # an external ``stdgates.inc`` file would make the artifact non-portable.
    lines = ["OPENQASM 3.0;"]
    leaves = scalar_parameters(program.parameter_abi)
    parameter_symbols = {leaf.value: f"p{index}" for index, leaf in enumerate(leaves)}
    for leaf in leaves:
        value_type = leaf.type
        qasm_type = {"Real": "float[64]", "Int": "int[64]", "Bool": "bool"}.get(
            value_type.name
        )
        if qasm_type is None:
            raise CapabilityError(
                f"Q8406: unsupported OpenQASM input type {value_type}"
            )
        lines.append(f"input {qasm_type} {parameter_symbols[leaf.value]};")
    if program.measurement_count:
        lines.append(f"bit[{program.measurement_count}] c;")
    lines.append(f"qubit[{program.width}] q;")
    _emit_block(
        program.body,
        lines,
        indent="",
        parameter_symbols=parameter_symbols,
    )
    return "\n".join(lines) + "\n"


def _emit_block(
    block: CircuitBlock,
    lines: list[str],
    *,
    indent: str,
    parameter_symbols: dict[ParameterValue, str],
) -> None:
    """Render one structured block as OpenQASM statements."""

    for instruction in block.instructions:
        _emit_instruction(
            instruction,
            lines,
            indent=indent,
            parameter_symbols=parameter_symbols,
        )


def _emit_instruction(
    instruction: CircuitInstruction,
    lines: list[str],
    *,
    indent: str,
    parameter_symbols: dict[ParameterValue, str],
) -> None:
    """Render one logical circuit instruction as OpenQASM."""

    op = instruction.opcode
    qubits = (*instruction.controls, *instruction.qubits)
    modifier = "ctrl @ " * len(instruction.controls)
    if op == "global_phase":
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        if not instruction.controls:
            lines.append(f"{indent}gphase({angle});")
        elif len(instruction.controls) == 1:
            lines.append(f"{indent}phaseshift({angle}) q[{qubits[0]}];")
        else:
            prefix = "ctrl @ " * (len(instruction.controls) - 1)
            operands = ", ".join(f"q[{index}]" for index in qubits)
            lines.append(f"{indent}{prefix}phaseshift({angle}) {operands};")
        return
    if op == "x":
        controls = len(instruction.controls)
        if controls == 0:
            lines.append(f"{indent}x q[{qubits[0]}];")
        elif controls == 1:
            lines.append(f"{indent}cnot q[{qubits[0]}], q[{qubits[1]}];")
        else:
            prefix = "ctrl @ " * (controls - 2)
            operands = ", ".join(f"q[{index}]" for index in qubits)
            lines.append(f"{indent}{prefix}ccnot {operands};")
        return
    if op in {"h", "y", "z", "s", "t", "sdg", "tdg", "reset"}:
        native_op = {"sdg": "si", "tdg": "ti"}.get(op, op)
        operands = ", ".join(f"q[{index}]" for index in qubits)
        lines.append(f"{indent}{modifier}{native_op} {operands};")
        return
    if op in {"p", "rx", "ry", "rz"}:
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        if op == "p" and instruction.controls:
            prefix = "ctrl @ " * (len(instruction.controls) - 1)
            operands = ", ".join(f"q[{index}]" for index in qubits)
            lines.append(f"{indent}{prefix}cphaseshift({angle}) {operands};")
            return
        native_op = "phaseshift" if op == "p" else op
        operands = ", ".join(f"q[{index}]" for index in qubits)
        lines.append(f"{indent}{modifier}{native_op}({angle}) {operands};")
        return
    if op in {"cp", "rzz"}:
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        native_op = "cphaseshift" if op == "cp" else "zz"
        operands = ", ".join(f"q[{index}]" for index in qubits)
        lines.append(f"{indent}{modifier}{native_op}({angle}) {operands};")
        return
    if op == "ccx":
        operands = ", ".join(f"q[{index}]" for index in qubits)
        lines.append(f"{indent}{modifier}ccnot {operands};")
        return
    if op in {"cx", "cz", "swap"}:
        native_op = "cnot" if op == "cx" else op
        operands = ", ".join(f"q[{index}]" for index in qubits)
        lines.append(f"{indent}{modifier}{native_op} {operands};")
        return
    if op == "measure":
        index = instruction.parameters[0]
        if not isinstance(index, int):
            raise CapabilityError("Q8407: measurement index must be static")
        lines.append(f"{indent}c[{index}] = measure q[{qubits[0]}];")
        return
    if op == "if":
        condition = instruction.condition
        if condition is None or len(instruction.regions) != 2:
            raise CapabilityError("Q8408: malformed OpenQASM device if")
        predicate = (
            f"c[{condition.index}]"
            if isinstance(condition, MeasurementBit)
            else _emit_value(condition, parameter_symbols)
        )
        lines.append(f"{indent}if ({predicate}) {{")
        _emit_block(
            instruction.regions[0],
            lines,
            indent=indent + "  ",
            parameter_symbols=parameter_symbols,
        )
        lines.append(f"{indent}}} else {{")
        _emit_block(
            instruction.regions[1],
            lines,
            indent=indent + "  ",
            parameter_symbols=parameter_symbols,
        )
        lines.append(f"{indent}}}")
        return
    raise CapabilityError(f"Q8409: OpenQASM cannot emit instruction {op!r}")


def _emit_value(
    value: CircuitValue,
    parameter_symbols: dict[ParameterValue, str],
) -> str:
    """Render a target-neutral scalar expression as OpenQASM."""

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, ParameterValue):
        try:
            return parameter_symbols[value]
        except KeyError as error:
            raise CapabilityError(
                f"Q8411: unknown OpenQASM parameter {value!r}"
            ) from error
    if isinstance(value, MeasurementBit):
        return f"(c[{value.index}] == 1)"
    if isinstance(value, ClassicalExpr):
        values = tuple(_emit_value(item, parameter_symbols) for item in value.operands)
        op = value.opcode.removeprefix("classical.")
        if op == "neg":
            return f"(-{values[0]})"
        if op == "not":
            return f"(!{values[0]})"
        if op == "select":
            return f"({values[0]} ? {values[1]} : {values[2]})"
        if op in {"and", "or"}:
            separator = " && " if op == "and" else " || "
            return "(" + separator.join(values) + ")"
        if op == "compare_chain":
            operators = cast(
                tuple[str, ...],
                dict(value.attributes.entries).get("operators", ()),
            )
            symbols = {
                "eq": "==",
                "ne": "!=",
                "lt": "<",
                "le": "<=",
                "gt": ">",
                "ge": ">=",
            }
            parts = tuple(
                f"({left} {symbols[name]} {right})"
                for name, left, right in zip(
                    operators, values[:-1], values[1:], strict=True
                )
            )
            return "(" + " && ".join(parts) + ")"
        symbol = {
            "add": "+",
            "sub": "-",
            "mul": "*",
            "div": "/",
            "truediv": "/",
            "pow": "**",
        }.get(op)
        if symbol is not None:
            return f"({values[0]} {symbol} {values[1]})"
    raise CapabilityError(f"Q8410: unsupported OpenQASM expression {value!r}")
