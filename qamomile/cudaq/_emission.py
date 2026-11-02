"""Emit target-neutral circuit projections as CUDA-Q source."""

from __future__ import annotations

import hashlib
import math
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
from qamomile.observable import Hamiltonian


def emit_source(program: CircuitProgram) -> str:
    """Render a verified circuit as a self-contained CUDA-Q Python module.

    Args:
        program (CircuitProgram): Native circuit projection and execution ABI.

    Returns:
        str: Source defining helper kernels and the sampling entry kernel.

    Raises:
        CapabilityError: If the circuit or ABI cannot be emitted by CUDA-Q.
    """

    parameters: list[str] = []
    leaves = scalar_parameters(program.parameter_abi)
    parameter_symbols = {leaf.value: f"p{index}" for index, leaf in enumerate(leaves)}
    for leaf in leaves:
        value_type = leaf.type
        python_type = {"Real": "float", "Int": "int", "Bool": "bool"}.get(
            value_type.name
        )
        if python_type is None:
            raise CapabilityError(f"Q8507: unsupported CUDA-Q input type {value_type}")
        parameters.append(f"{parameter_symbols[leaf.value]}: {python_type}")
    logical_width = program.result_abi.width
    if logical_width <= 0:
        raise CapabilityError("Q8513: CUDA-Q sampling target requires a bit result")
    return_width = (
        program.measurement_count
        if program.requires_result_postprocessing
        else logical_width
    )
    if return_width <= 0:
        raise CapabilityError("Q8515: CUDA-Q result expression has no measurements")
    return_type = (
        "bool"
        if return_width == 1
        else "tuple[" + ", ".join("bool" for _ in range(return_width)) + "]"
    )
    lines = [
        "@cudaq.kernel",
        f"def _qamomile_entry({', '.join(parameters)}) -> {return_type}:",
        f"    q = cudaq.qvector({program.width})",
    ]
    definitions: dict[str, str] = {}
    _emit_block(
        program.body,
        lines,
        indent="    ",
        parameter_symbols=parameter_symbols,
        definitions=definitions,
    )
    returned = (
        [f"m{index}" for index in range(program.measurement_count)]
        if program.requires_result_postprocessing
        else [f"m{index}" for index in program.result_abi.source_order]
    )
    if len(returned) != return_width:
        raise CapabilityError("Q8515: CUDA-Q result expression width mismatch")
    expression = returned[0] if len(returned) == 1 else f"({', '.join(returned)})"
    lines.append(f"    return {expression}")
    return "\n".join(["import cudaq", "", *definitions.values(), *lines]) + "\n"


def _emit_block(
    block: CircuitBlock,
    lines: list[str],
    *,
    indent: str,
    parameter_symbols: dict[ParameterValue, str],
    definitions: dict[str, str],
) -> None:
    """Render a block and collect shared SDK helper definitions.

    Args:
        block (CircuitBlock): Verified structured instruction block.
        lines (list[str]): Destination source lines.
        indent (str): Current source indentation.
        parameter_symbols (dict[ParameterValue, str]): SDK parameter names.
        definitions (dict[str, str]): Helpers indexed by their Pauli label.
    """

    if not block.instructions:
        lines.append(indent + "pass")
        return
    for instruction in block.instructions:
        _emit_instruction(
            instruction,
            lines,
            indent=indent,
            parameter_symbols=parameter_symbols,
            definitions=definitions,
        )


def _emit_instruction(
    instruction: CircuitInstruction,
    lines: list[str],
    *,
    indent: str,
    parameter_symbols: dict[ParameterValue, str],
    definitions: dict[str, str],
) -> None:
    """Render one instruction, retaining accepted composite operations.

    Args:
        instruction (CircuitInstruction): Verified circuit instruction.
        lines (list[str]): Destination source lines.
        indent (str): Current source indentation.
        parameter_symbols (dict[ParameterValue, str]): SDK parameter names.
        definitions (dict[str, str]): Shared Pauli kernel definitions.

    Raises:
        CapabilityError: If the instruction cannot be emitted.
    """

    op = instruction.opcode
    qubits = instruction.qubits
    if op == "pauli_evolution":
        _emit_pauli_evolution(
            instruction,
            lines,
            indent=indent,
            parameter_symbols=parameter_symbols,
            definitions=definitions,
        )
        return
    if instruction.controls:
        _emit_controlled_instruction(
            instruction,
            lines,
            indent=indent,
            parameter_symbols=parameter_symbols,
        )
        return
    if op == "global_phase":
        lines.append(f"{indent}# closed-program global phase")
        return
    if op in {"h", "x", "y", "z", "s", "t", "reset"}:
        lines.append(f"{indent}{op}(q[{qubits[0]}])")
        return
    if op in {"sdg", "tdg"}:
        base = op[0]
        lines.append(f"{indent}{base}.adj(q[{qubits[0]}])")
        return
    if op in {"p", "rx", "ry", "rz"}:
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        native_op = "r1" if op == "p" else op
        lines.append(f"{indent}{native_op}({angle}, q[{qubits[0]}])")
        return
    if op == "cp":
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        lines.append(f"{indent}r1.ctrl({angle}, q[{qubits[0]}], q[{qubits[1]}])")
        return
    if op == "rzz":
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        lines.append(f"{indent}x.ctrl(q[{qubits[0]}], q[{qubits[1]}])")
        lines.append(f"{indent}rz({angle}, q[{qubits[1]}])")
        lines.append(f"{indent}x.ctrl(q[{qubits[0]}], q[{qubits[1]}])")
        return
    if op == "ccx":
        lines.append(f"{indent}x.ctrl(q[{qubits[0]}], q[{qubits[1]}], q[{qubits[2]}])")
        return
    if op == "cx":
        lines.append(f"{indent}x.ctrl(q[{qubits[0]}], q[{qubits[1]}])")
        return
    if op == "cz":
        lines.append(f"{indent}z.ctrl(q[{qubits[0]}], q[{qubits[1]}])")
        return
    if op == "swap":
        lines.append(f"{indent}swap(q[{qubits[0]}], q[{qubits[1]}])")
        return
    if op == "measure":
        index = instruction.parameters[0]
        if not isinstance(index, int):
            raise CapabilityError("Q8508: measurement index must be static")
        lines.append(f"{indent}m{index} = mz(q[{qubits[0]}])")
        return
    if op == "if":
        condition = instruction.condition
        if condition is None or len(instruction.regions) != 2:
            raise CapabilityError("Q8509: malformed CUDA-Q device if")
        lines.append(f"{indent}if {_emit_value(condition, parameter_symbols)}:")
        _emit_block(
            instruction.regions[0],
            lines,
            indent=indent + "    ",
            parameter_symbols=parameter_symbols,
            definitions=definitions,
        )
        lines.append(f"{indent}else:")
        _emit_block(
            instruction.regions[1],
            lines,
            indent=indent + "    ",
            parameter_symbols=parameter_symbols,
            definitions=definitions,
        )
        return
    raise CapabilityError(f"Q8510: CUDA-Q cannot emit instruction {op!r}")


def _emit_controlled_instruction(
    instruction: CircuitInstruction,
    lines: list[str],
    *,
    indent: str,
    parameter_symbols: dict[ParameterValue, str],
) -> None:
    """Render an exact controlled operation in CUDA-Q's kernel dialect.

    Args:
        instruction (CircuitInstruction): Verified controlled instruction.
        lines (list[str]): Destination source lines.
        indent (str): Current source indentation.
        parameter_symbols (dict[ParameterValue, str]): SDK parameter names.

    Raises:
        CapabilityError: If CUDA-Q cannot emit the controlled operation.
    """

    op = instruction.opcode
    controls = tuple(f"q[{index}]" for index in instruction.controls)
    targets = tuple(f"q[{index}]" for index in instruction.qubits)
    control_args = ", ".join(controls)
    all_args = ", ".join((*controls, *targets))
    if op == "global_phase":
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        if len(controls) == 1:
            lines.append(f"{indent}r1({angle}, {controls[0]})")
        else:
            phase_args = ", ".join((*controls[:-1], controls[-1]))
            lines.append(f"{indent}r1.ctrl({angle}, {phase_args})")
        return
    if op in {"h", "x", "y", "z", "s", "t"}:
        lines.append(f"{indent}{op}.ctrl({all_args})")
        return
    if op in {"sdg", "tdg"}:
        angle = -math.pi / (2 if op == "sdg" else 4)
        lines.append(f"{indent}r1.ctrl({angle!r}, {all_args})")
        return
    if op in {"p", "rx", "ry", "rz"}:
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        native = "r1" if op == "p" else op
        lines.append(f"{indent}{native}.ctrl({angle}, {all_args})")
        return
    if op in {"cx", "ccx"}:
        lines.append(f"{indent}x.ctrl({all_args})")
        return
    if op == "cz":
        lines.append(f"{indent}z.ctrl({all_args})")
        return
    if op == "cp":
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        lines.append(f"{indent}r1.ctrl({angle}, {all_args})")
        return
    if op == "swap":
        lines.append(f"{indent}swap.ctrl({all_args})")
        return
    if op == "rzz":
        angle = _emit_value(instruction.parameters[0], parameter_symbols)
        left, right = targets
        lines.append(f"{indent}x.ctrl({left}, {right})")
        lines.append(f"{indent}rz.ctrl({angle}, {control_args}, {right})")
        lines.append(f"{indent}x.ctrl({left}, {right})")
        return
    raise CapabilityError(f"Q8513: CUDA-Q cannot coherently control {op!r}")


def _emit_pauli_evolution(
    instruction: CircuitInstruction,
    lines: list[str],
    *,
    indent: str,
    parameter_symbols: dict[ParameterValue, str],
    definitions: dict[str, str],
) -> None:
    """Emit Rust-validated commuting evolution with SDK Pauli exponentials.

    CUDA-Q uses exp(i angle P), whereas QCore evolution is exp(-i time H).
    Sparse factors select only their corresponding physical wires. A helper
    with a literal word lets CUDA-Q control the exponential as one operation.

    Args:
        instruction (CircuitInstruction): Verified Hamiltonian, time and wires.
        lines (list[str]): Destination source lines.
        indent (str): Current source indentation.
        parameter_symbols (dict[ParameterValue, str]): SDK parameter names.
        definitions (dict[str, str]): Shared kernels indexed by Pauli label.

    Raises:
        CapabilityError: If the native instruction projection is malformed.
    """
    if (
        instruction.condition is not None
        or instruction.regions
        or len(instruction.parameters) != 2
    ):
        raise CapabilityError("Q8510: malformed CUDA-Q Pauli evolution")
    hamiltonian, raw_time = instruction.parameters
    if not isinstance(hamiltonian, Hamiltonian) or hamiltonian.num_qubits > len(
        instruction.qubits
    ):
        raise CapabilityError("Q8510: malformed CUDA-Q Pauli Hamiltonian")
    time = _emit_value(raw_time, parameter_symbols)
    controls = ", ".join(f"q[{index}]" for index in instruction.controls)
    if hamiltonian.constant.real and instruction.controls:
        angle = f"({-hamiltonian.constant.real!r} * ({time}))"
        native = "r1" if len(instruction.controls) == 1 else "r1.ctrl"
        lines.append(f"{indent}{native}({angle}, {controls})")
    for word, coefficient in hamiltonian.terms:
        label = "".join(factor.pauli.value for factor in word.factors)
        targets = ", ".join(
            f"q[{instruction.qubits[factor.index]}]" for factor in word.factors
        )
        angle = f"({-coefficient.real!r} * ({time}))"
        if instruction.controls:
            name = "_qamomile_pauli_" + hashlib.sha256(label.encode()).hexdigest()
            definitions.setdefault(
                label,
                f"@cudaq.kernel\ndef {name}(angle: float, targets: cudaq.qview):\n"
                f"    exp_pauli(angle, targets, {label!r})\n",
            )
            lines.append(
                f"{indent}cudaq.control({name}, [{controls}], {angle}, [{targets}])"
            )
        else:
            lines.append(f"{indent}exp_pauli({angle}, [{targets}], {label!r})")
    if not hamiltonian.terms and not (
        hamiltonian.constant.real and instruction.controls
    ):
        lines.append(f"{indent}pass  # closed-program global phase")


def _emit_value(
    value: CircuitValue,
    parameter_symbols: dict[ParameterValue, str],
) -> str:
    """Render a target-neutral scalar expression as CUDA-Q Python."""

    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, ParameterValue):
        try:
            return parameter_symbols[value]
        except KeyError as error:
            raise CapabilityError(
                f"Q8514: unknown CUDA-Q parameter {value!r}"
            ) from error
    if isinstance(value, MeasurementBit):
        return f"m{value.index}"
    if isinstance(value, ClassicalExpr):
        values = tuple(_emit_value(item, parameter_symbols) for item in value.operands)
        op = value.opcode.removeprefix("classical.")
        if op == "neg":
            return f"(-{values[0]})"
        if op == "not":
            return f"(not {values[0]})"
        if op == "select":
            return f"({values[1]} if {values[0]} else {values[2]})"
        if op in {"and", "or"}:
            return "(" + f" {op} ".join(values) + ")"
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
            return "(" + " and ".join(parts) + ")"
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
    raise CapabilityError(f"Q8511: unsupported CUDA-Q expression {value!r}")
