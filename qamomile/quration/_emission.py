"""Emit the checked static circuit projection through PyQret intrinsics."""

from __future__ import annotations

import math
from typing import Any

from qamomile.circuit import CircuitInstruction, CircuitProgram, ClassicalExpr
from qamomile.errors import CapabilityError
from qamomile.operations import evaluate_classical

from ._payload import QurationExecutable

_UNARY = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
_GATES = _UNARY | {
    "rx",
    "ry",
    "rz",
    "p",
    "cx",
    "cz",
    "ccx",
    "cp",
    "swap",
    "rzz",
    "global_phase",
    "measure",
}


def emit_module(
    logical: CircuitProgram, precision: float
) -> tuple[QurationExecutable, None]:
    """Emit and retain a concrete circuit with its complete SDK owner graph.

    Args:
        logical (CircuitProgram): Rust-projected circuit and ABI.
        precision (float): Per-rotation synthesis tolerance for FTQC compilation.

    Returns:
        tuple[QurationExecutable, None]: SDK owner and unmeasured native instruction count.

    Raises:
        CapabilityError: If a parameter, control, or gate is unsupported.
        ImportError: If the optional PyQret SDK is unavailable.
    """
    from pyqret import frontend

    if logical.parameter_abi.slots:
        raise CapabilityError(
            "Q8901: Quration requires Const parameters specialized before compilation"
        )
    for instruction in logical.body.instructions:
        _validate_instruction(instruction)
    context = frontend.Context()
    module = frontend.Module("qamomile", context)
    builder = frontend.CircuitBuilder(module)

    class GeneratedCircuit(frontend.CircuitGenerator):
        """Define the already checked static circuit in one SDK context."""

        def name(self) -> str:
            """Return the fixed entry name used when restoring the module.

            Returns:
                str: The native entry name.
            """
            return "main"

        def arg(self) -> Any:
            """Return quantum inputs and chronological measurement outputs.

            Returns:
                Any: PyQret Argument declaration using compiler-derived widths.
            """
            argument = frontend.Argument()
            if logical.width:
                argument.add_operates("q", logical.width)
            if logical.measurement_count:
                argument.add_outputs("c", logical.measurement_count)
            return argument

        def logic(self, arg: Any) -> None:
            """Emit all instructions inside the SDK definition context.

            Args:
                arg (Any): SDK argument view for this definition.
            """
            from pyqret.frontend.gate import intrinsic

            qubits = arg["q"] if logical.width else ()
            registers = arg["c"] if logical.measurement_count else ()
            for instruction in logical.body.instructions:
                if instruction.opcode == "measure":
                    intrinsic.measure(
                        qubits[instruction.qubits[0]],
                        registers[int(instruction.parameters[0])],
                    )
                else:
                    _emit_gate(instruction, qubits, builder, precision)

    generator = GeneratedCircuit(builder)
    circuit = generator.generate()
    return QurationExecutable(circuit, module, (context, builder, generator)), None


def _validate_instruction(instruction: CircuitInstruction) -> None:
    """Reject unsupported forms before allocating SDK circuit objects.

    Args:
        instruction (CircuitInstruction): One compiler-projected instruction.

    Raises:
        CapabilityError: If the operation is outside the static target profile.
    """
    if instruction.condition is not None or instruction.regions:
        raise CapabilityError("Q8902: Quration does not support shot-dependent control")
    if instruction.opcode not in _GATES:
        raise CapabilityError(
            f"Q8903: unsupported Quration instruction {instruction.opcode!r}"
        )
    for value in instruction.parameters:
        _constant(value)
    controls = len(instruction.controls)
    if controls and not (
        (
            controls == 1
            and instruction.opcode
            in _UNARY | {"rx", "ry", "rz", "p", "global_phase", "cx", "cz"}
        )
        or (controls == 2 and instruction.opcode in {"x", "y", "z"})
    ):
        raise CapabilityError(
            "Q8904: Quration supports at most two controls on Pauli gates and one on unary gates"
        )


def _constant(value: object) -> int | float:
    """Resolve a closed emitted scalar using the shared native arithmetic.

    Args:
        value (object): Literal or compiler-generated constant expression.

    Returns:
        int | float: Finite concrete scalar, preserving integer arithmetic.

    Raises:
        CapabilityError: If a value requires runtime input or is nonfinite.
    """
    if isinstance(value, ClassicalExpr):
        value = evaluate_classical(
            value.opcode,
            tuple(_constant(item) for item in value.operands),
            attributes=value.attributes.entries,
        )
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise CapabilityError(
            "Q8901: Quration requires finite concrete gate parameters"
        )
    return value


def _phase(qubit: Any, angle: float, builder: Any, precision: float) -> None:
    """Emit diag(1, exp(i angle)) with its exact global phase.

    Args:
        qubit (Any): SDK target qubit.
        angle (float): Phase in radians.
        builder (Any): SDK builder for global phase emission.
        precision (float): FTQC synthesis tolerance.
    """
    from pyqret.frontend.gate import intrinsic as gate

    gate.global_phase(builder, angle / 2, precision)
    gate.rz(qubit, angle, precision)


def _controlled_rotation(
    axis: str, control: Any, target: Any, angle: float, precision: float
) -> None:
    """Emit a singly controlled Pauli rotation through SDK intrinsics.

    Args:
        axis (str): Rotation axis x, y, or z.
        control (Any): SDK control qubit.
        target (Any): SDK target qubit.
        angle (float): Rotation angle in radians.
        precision (float): FTQC synthesis tolerance.
    """
    from pyqret.frontend.gate import intrinsic as gate

    if axis == "x":
        gate.h(target)
        _controlled_rotation("z", control, target, angle, precision)
        gate.h(target)
    else:
        rotation = gate.ry if axis == "y" else gate.rz
        rotation(target, angle / 2, precision)
        gate.cx(target, control)
        rotation(target, -angle / 2, precision)
        gate.cx(target, control)


def _emit_gate(
    instruction: CircuitInstruction, wires: Any, builder: Any, precision: float
) -> None:
    """Translate one supported gate while preserving operand order and phase.

    Args:
        instruction (CircuitInstruction): Prevalidated static gate.
        wires (Any): SDK register indexed by logical qubit number.
        builder (Any): Owning SDK definition builder.
        precision (float): Per-rotation FTQC synthesis tolerance.
    """
    from pyqret.frontend.gate import intrinsic as gate

    opcode = instruction.opcode
    qubits = tuple(wires[index] for index in instruction.qubits)
    controls = tuple(wires[index] for index in instruction.controls)
    angle = (
        float(_constant(instruction.parameters[0])) if instruction.parameters else 0.0
    )
    if opcode in {"cx", "cz", "ccx"}:
        controls += qubits[:-1]
        qubits = qubits[-1:]
        opcode = "z" if opcode == "cz" else "x"
    elif opcode == "cp":
        controls += qubits[:1]
        qubits = qubits[1:]
        opcode = "p"
    if controls:
        if opcode == "global_phase":
            _phase(controls[0], angle, builder, precision)
        elif opcode in {"x", "y", "z"}:
            getattr(gate, "c" * len(controls) + opcode)(qubits[0], *controls)
        elif opcode in {"rx", "ry", "rz"}:
            _controlled_rotation(opcode[1], controls[0], qubits[0], angle, precision)
        elif opcode == "h":
            gate.s(qubits[0])
            gate.h(qubits[0])
            gate.t(qubits[0])
            gate.cx(qubits[0], controls[0])
            gate.tdag(qubits[0])
            gate.h(qubits[0])
            gate.sdag(qubits[0])
        else:
            phases = {
                "s": math.pi / 2,
                "sdg": -math.pi / 2,
                "t": math.pi / 4,
                "tdg": -math.pi / 4,
            }
            phase = angle if opcode == "p" else phases[opcode]
            _phase(controls[0], phase / 2, builder, precision)
            _controlled_rotation("z", controls[0], qubits[0], phase, precision)
    elif opcode in _UNARY:
        getattr(gate, {"sdg": "sdag", "tdg": "tdag"}.get(opcode, opcode))(qubits[0])
    elif opcode in {"rx", "ry", "rz"}:
        getattr(gate, opcode)(qubits[0], angle, precision)
    elif opcode == "p":
        _phase(qubits[0], angle, builder, precision)
    elif opcode == "global_phase":
        gate.global_phase(builder, angle, precision)
    elif opcode == "swap":
        gate.cx(qubits[1], qubits[0])
        gate.cx(qubits[0], qubits[1])
        gate.cx(qubits[1], qubits[0])
    elif opcode == "rzz":
        gate.cx(qubits[1], qubits[0])
        gate.rz(qubits[1], angle, precision)
        gate.cx(qubits[1], qubits[0])
