"""Emit bounded static circuits through the PyQIR basic quantum instruction set.

Gate decompositions preserve the full controlled operator. Overall global
phase is unobservable for these classical-output entries and is omitted;
controlled global phase remains a relative phase and is emitted.
"""

from __future__ import annotations

import math
from typing import TypeAlias

from qamomile.circuit import CircuitInstruction, CircuitProgram, ClassicalExpr
from qamomile.errors import CapabilityError
from qamomile.operations import evaluate_classical
from qamomile.types import Unit, strip_stage

from ._payload import QIRExecutable

MAX_QIR_INSTRUCTIONS = 1_000_000
OUTPUT_LABELING_SCHEMA = "qamomile.source-order.v1"
_UNARY = {"h", "x", "y", "z", "s", "sdg", "t", "tdg"}
_ROTATIONS = {"rx", "ry", "rz", "p"}
_GATES = _UNARY | _ROTATIONS | {"cx", "cz", "ccx", "cp", "swap", "rzz", "global_phase"}
_QISOperation: TypeAlias = tuple[str, tuple[int, ...], float | int | None]


def emit_module(logical: CircuitProgram) -> tuple[QIRExecutable, int]:
    """Build a verified QIR 1.0 Base Profile module from a native projection.

    Args:
        logical (CircuitProgram): Rust-projected static circuit and result ABI.

    Returns:
        tuple[QIRExecutable, int]: Immutable bitcode and emitted LLVM instruction count.

    Raises:
        CapabilityError: If the circuit exceeds the profile or emission budget,
            or generated LLVM fails verification.
        ImportError: If the optional PyQIR dependency is unavailable.
    """
    operations = _plan_operations(logical)
    import pyqir
    import pyqir.qis as qis
    import pyqir.rt as rt

    context = pyqir.Context()
    module = pyqir.qir_module(
        context,
        "qamomile",
        qir_major_version=1,
        qir_minor_version=0,
        dynamic_qubit_management=False,
        dynamic_result_management=False,
    )
    entry = pyqir.Function(
        pyqir.FunctionType(pyqir.IntType(context, 64), []),
        pyqir.Linkage.EXTERNAL,
        "main",
        module,
    )
    for name, value in (
        ("entry_point", ""),
        ("required_num_qubits", str(logical.width)),
        ("required_num_results", str(logical.measurement_count)),
        ("qir_profiles", "base_profile"),
        ("output_labeling_schema", OUTPUT_LABELING_SCHEMA),
    ):
        pyqir.add_string_attribute(entry, name, value)
    builder = pyqir.Builder(context)
    blocks = tuple(
        pyqir.BasicBlock(context, name, entry)
        for name in ("entry", "body", "measurements", "output")
    )
    initialization, body, measurements, output = blocks
    builder.insert_at_end(initialization)
    rt.initialize(
        builder, pyqir.Constant.null(pyqir.PointerType(pyqir.IntType(context, 8)))
    )
    builder.br(body)
    builder.insert_at_end(body)
    for opcode, wires, parameter in operations:
        if opcode == "mz":
            continue
        qubits = tuple(pyqir.qubit(context, wire) for wire in wires)
        if parameter is not None:
            getattr(qis, opcode)(builder, parameter, *qubits)
        else:
            getattr(qis, opcode)(builder, *qubits)
    builder.br(measurements)
    builder.insert_at_end(measurements)
    for opcode, wires, parameter in operations:
        if opcode == "mz":
            assert isinstance(parameter, int)
            qis.mz(
                builder,
                pyqir.qubit(context, wires[0]),
                pyqir.result(context, parameter),
            )
    builder.br(output)
    builder.insert_at_end(output)
    rt.array_record_output(
        builder,
        pyqir.const(pyqir.IntType(context, 64), logical.result_abi.width),
        pyqir.global_byte_string(module, b"output"),
    )
    for position, result in enumerate(logical.result_abi.source_order):
        rt.result_record_output(
            builder,
            pyqir.result(context, result),
            pyqir.global_byte_string(module, f"output.{position}".encode()),
        )
    builder.ret(pyqir.const(pyqir.IntType(context, 64), 0))
    error = module.verify()
    if error is not None:
        raise CapabilityError(f"Q9004: generated QIR failed LLVM verification: {error}")
    return QIRExecutable(module.bitcode), sum(
        len(block.instructions) for block in blocks
    )


def _plan_operations(logical: CircuitProgram) -> tuple[_QISOperation, ...]:
    """Preflight profile constraints and bound expansion before allocating LLVM.

    Args:
        logical (CircuitProgram): Compiler-owned circuit projection.

    Returns:
        tuple[_QISOperation, ...]: Concrete calls in program order.

    Raises:
        CapabilityError: If parameters, results, gates, measurement placement,
            or instruction counts are outside the supported profile.
    """
    if logical.parameter_abi.slots:
        raise CapabilityError(
            "Q9001: QIR Base Profile requires specialized Const inputs"
        )
    if logical.requires_result_postprocessing or (
        logical.result_abi.width == 0 and strip_stage(logical.result_abi.type) != Unit
    ):
        raise CapabilityError(
            "Q9002: QIR Base Profile requires direct measurement results or Unit"
        )
    overhead = (
        logical.result_abi.width + 6
    )  # initialize, array output, three branches, return
    if overhead + logical.measurement_count > MAX_QIR_INSTRUCTIONS:
        raise CapabilityError("Q9003: QIR instruction budget exceeded")
    operations: list[_QISOperation] = []
    measured = False
    measured_wires: set[int] = set()
    for instruction in logical.body.instructions:
        if instruction.condition is not None or instruction.regions:
            raise CapabilityError(
                "Q9002: QIR Base Profile does not support structured or conditional instructions"
            )
        if instruction.opcode == "measure":
            if instruction.controls or instruction.qubits[0] in measured_wires:
                raise CapabilityError(
                    "Q9002: QIR Base Profile cannot remeasure a qubit"
                )
            measured = True
            measured_wires.add(instruction.qubits[0])
            result = instruction.parameters[0]
            assert isinstance(result, int)
            expanded: tuple[_QISOperation, ...] = (("mz", instruction.qubits, result),)
        else:
            if measured:
                raise CapabilityError(
                    "Q9002: QIR Base Profile requires all gates before measurements"
                )
            expanded = _gate_operations(instruction)
        if overhead + len(operations) + len(expanded) > MAX_QIR_INSTRUCTIONS:
            raise CapabilityError("Q9003: QIR instruction budget exceeded")
        operations.extend(expanded)
    return tuple(operations)


def _constant(value: object) -> int | float:
    """Resolve a closed rotation angle through the shared native arithmetic.

    Args:
        value (object): Literal or compiler-generated closed scalar expression.

    Returns:
        int | float: Finite scalar preserving native integer arithmetic.

    Raises:
        CapabilityError: If the value requires runtime input or is nonfinite.
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
        raise CapabilityError("Q9001: QIR requires finite concrete gate parameters")
    return value


def _controlled_rotation(
    axis: str, control: int, target: int, angle: float
) -> tuple[_QISOperation, ...]:
    """Decompose a controlled Pauli rotation into basic QIS calls.

    Args:
        axis (str): Rotation axis x, y or z.
        control (int): Logical control wire.
        target (int): Logical target wire.
        angle (float): Rotation angle in radians.

    Returns:
        tuple[_QISOperation, ...]: Phase-correct controlled rotation.
    """
    if axis == "x":
        return (
            ("h", (target,), None),
            *_controlled_rotation("z", control, target, angle),
            ("h", (target,), None),
        )
    return (
        (f"r{axis}", (target,), angle / 2),
        ("cx", (control, target), None),
        (f"r{axis}", (target,), -angle / 2),
        ("cx", (control, target), None),
    )


def _gate_operations(instruction: CircuitInstruction) -> tuple[_QISOperation, ...]:
    """Translate one gate to the supported basic QIS, preserving relative phase.

    Args:
        instruction (CircuitInstruction): Static projected gate.

    Returns:
        tuple[_QISOperation, ...]: Constant-size basic QIS decomposition.

    Raises:
        CapabilityError: If the gate, controls or parameters are unsupported.
    """
    opcode = instruction.opcode
    if opcode not in _GATES:
        raise CapabilityError(f"Q9002: unsupported QIR instruction {opcode!r}")
    wires, controls = instruction.qubits, instruction.controls
    angle = (
        float(_constant(instruction.parameters[0])) if instruction.parameters else 0.0
    )
    if opcode in {"cx", "cz", "ccx", "cp"}:
        controls += wires[:-1]
        wires = wires[-1:]
        opcode = {"cx": "x", "cz": "z", "ccx": "x", "cp": "p"}[opcode]
    if controls:
        if len(controls) == 1 and opcode == "global_phase":
            return (("rz", controls, angle),)
        if opcode in {"x", "y", "z"} and len(controls) <= 2:
            target = wires[0]
            flip: _QISOperation = (
                "cx" if len(controls) == 1 else "ccx",
                (*controls, target),
                None,
            )
            if opcode == "x":
                return (flip,)
            if opcode == "z":
                return (("h", wires, None), flip, ("h", wires, None))
            return (("s_adj", wires, None), flip, ("s", wires, None))
        if len(controls) != 1 or opcode not in _UNARY | _ROTATIONS:
            raise CapabilityError(
                "Q9002: QIR supports up to two controls on Pauli gates and one on unary gates"
            )
        control, target = controls[0], wires[0]
        if opcode in {"rx", "ry", "rz"}:
            return _controlled_rotation(opcode[1], control, target, angle)
        if opcode == "h":
            return (
                ("s", wires, None),
                ("h", wires, None),
                ("t", wires, None),
                ("cx", (control, target), None),
                ("t_adj", wires, None),
                ("h", wires, None),
                ("s_adj", wires, None),
            )
        phases = {
            "s": math.pi / 2,
            "sdg": -math.pi / 2,
            "t": math.pi / 4,
            "tdg": -math.pi / 4,
        }
        phase = angle if opcode == "p" else phases[opcode]
        return (
            ("rz", (control,), phase / 2),
            *_controlled_rotation("z", control, target, phase),
        )
    if opcode == "global_phase":
        return ()
    if opcode == "rzz":
        return (("cx", wires, None), ("rz", wires[1:], angle), ("cx", wires, None))
    if opcode in _ROTATIONS:
        return (("rz" if opcode == "p" else opcode, wires, angle),)
    return (({"sdg": "s_adj", "tdg": "t_adj"}.get(opcode, opcode), wires, None),)
