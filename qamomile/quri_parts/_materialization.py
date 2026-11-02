"""Materialize and execute portable QURI Parts circuit payloads."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from qamomile.circuit import (
    RESULT_POSTPROCESSING_METADATA,
    CircuitBlock,
    CircuitInstruction,
    CircuitValue,
    ClassicalExpr,
    ParameterValue,
    postprocess_circuit_counts,
)
from qamomile.compilation import CompiledProgram, Invocation, SampleResult
from qamomile.errors import CapabilityError, ExecutionError

from ._profile import FORMAT


def load_logical_circuit(artifact: CompiledProgram) -> dict[str, object]:
    """Decode and minimally validate an inspected portable circuit payload."""

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8701: expected {FORMAT!r}, got {artifact.format!r}")
    from copy import deepcopy

    from ._payload import QuriExecutable

    return deepcopy(artifact.execution_payload(QuriExecutable).data())


def load_circuit(artifact: CompiledProgram) -> Any:
    """Project a portable artifact to an immutable QURI Parts circuit.

    Runtime scalar coordinates remain unbound QURI Parts parameters. Aggregate
    Qamomile parameters are flattened by first gate use and retain descriptive
    names such as ``gammas[0]``. Callers may inspect the returned circuit, bind
    it with :meth:`bind_parameters`, or pass it to a QURI Parts parametric
    estimator. The SDK circuit is an adapter-owned projection and is never
    stored in the portable artifact.

    Args:
        artifact (CompiledProgram): Portable artifact emitted by :class:`QuriParts`.

    Returns:
        Any: Immutable QURI Parts linear-mapped parametric circuit.

    Raises:
        ExecutionError: If the artifact is malformed, contains nonlinear or
            controlled runtime angles unavailable in the parametric SDK, or
            requires multi-control gates absent from the installed SDK.
    """

    from ._payload import QuriExecutable

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8701: expected {FORMAT!r}, got {artifact.format!r}")
    return artifact.execution_payload(QuriExecutable).public_circuit()


def _validate_block(block: CircuitBlock) -> None:
    """Reject circuit constructs unavailable in the Qulacs sampling profile."""

    for instruction in block.instructions:
        if instruction.condition is not None or instruction.regions:
            raise CapabilityError("Q8704: QURI Parts target requires static control")
        if instruction.opcode == "reset":
            raise CapabilityError("Q8705: QURI Parts target does not support reset")


def _execute_local(
    artifact: CompiledProgram,
    invocation: Invocation,
) -> SampleResult:
    """Materialize and sample a QURI Parts circuit in this process.

    Args:
        artifact (CompiledProgram): Sealed logical circuit and result ABI.
        invocation (Invocation): Validated parameters, shots, and seed.

    Returns:
        SampleResult: Source-ordered sample counts.

    Raises:
        ExecutionError: If circuit materialization or decoding fails.
    """

    import qulacs

    from ._payload import QuriExecutable

    executable = artifact.execution_payload(QuriExecutable)
    circuit = executable.qulacs_circuit(invocation.values())
    measurements = executable.measurements
    state = qulacs.QuantumState(executable.width)
    circuit.update_quantum_state(state)
    native: Counter[int] = Counter(
        state.sampling(invocation.shots, _qulacs_seed(invocation.seed))
    )
    logical = _decode_native_counts(
        artifact,
        invocation,
        measurements,
        native,
    )
    return SampleResult.from_counts(
        backend="quri-parts.qulacs",
        counts=logical,
        expected_width=artifact.result_abi.width,
    )


def _materialize_circuit(
    artifact: CompiledProgram,
    invocation: Invocation,
) -> tuple[Any, dict[int, int]]:
    """Build one bound QURI Parts circuit from portable artifact data."""

    from ._payload import QuriExecutable

    if artifact.format != FORMAT:
        raise ExecutionError(f"Q8701: expected {FORMAT!r}, got {artifact.format!r}")
    executable = artifact.execution_payload(QuriExecutable)
    return executable.bind(invocation.values()), dict(executable.measurements)


def _decode_native_counts(
    artifact: CompiledProgram,
    invocation: Invocation,
    measurements: Mapping[int, int],
    native: object,
) -> dict[tuple[int, ...], int]:
    """Validate native basis counts and decode Qamomile result order."""

    if not isinstance(native, Mapping):
        raise ExecutionError("Q8719: QURI Parts sampler must return a count mapping")
    from ._payload import QuriExecutable

    width = artifact.execution_payload(QuriExecutable).width
    if not isinstance(width, int):
        raise ExecutionError("Q8703: invalid QURI Parts logical circuit width")
    counts: dict[int, int] = {}
    for basis, count in native.items():
        if (
            type(basis) is not int
            or basis < 0
            or basis >= 1 << width
            or type(count) is not int
            or count < 0
        ):
            raise ExecutionError(
                "Q8720: QURI Parts sampler counts must map valid basis states "
                "to nonnegative integers"
            )
        if count:
            counts[basis] = counts.get(basis, 0) + count
    plan = dict(artifact.metadata.entries).get(RESULT_POSTPROCESSING_METADATA)
    if plan is None:
        logical: dict[tuple[int, ...], int] = {}
        for basis, count in counts.items():
            try:
                bits = tuple(
                    (basis >> measurements[index]) & 1
                    for index in artifact.result_abi.source_order
                )
            except KeyError as error:
                raise ExecutionError(
                    "Q8706: result ABI names an absent measurement"
                ) from error
            logical[bits] = logical.get(bits, 0) + int(count)
    else:
        if not isinstance(plan, bytes):
            raise ExecutionError("Q8715: invalid QURI Parts result metadata")
        try:
            raw: dict[tuple[int, ...], int] = {}
            for basis, count in counts.items():
                bits = tuple(
                    (basis >> measurements[index]) & 1
                    for index in range(len(measurements))
                )
                raw[bits] = raw.get(bits, 0) + int(count)
        except KeyError as error:
            raise ExecutionError("Q8706: result names an absent measurement") from error
        logical = postprocess_circuit_counts(
            plan,
            raw,
            invocation.values(),
            artifact.result_abi,
        )
    return logical


def _qulacs_seed(seed: int) -> int:
    """Fold one unsigned 64-bit seed into Qulacs' unsigned 32-bit domain."""

    return (seed ^ (seed >> 32)) & ((1 << 32) - 1)


def _materialize_parametric_instructions(
    circuit: Any,
    instructions: tuple[CircuitInstruction, ...],
    parameters: dict[tuple[str, tuple[int | str, ...]], Any],
) -> None:
    """Materialize static instructions and linear runtime gate angles.

    Args:
        circuit (Any): Mutable QURI Parts linear-mapped circuit.
        instructions (tuple[CircuitInstruction, ...]): Checked logical instruction rows.
        parameters (dict): Previously registered flattened QURI parameters.

    Raises:
        ExecutionError: If an instruction is malformed or not parametrically
            representable.
    """

    for instruction in instructions:
        opcode = instruction.opcode
        qubits = list(instruction.qubits)
        controls = list(instruction.controls)
        raw_parameters = instruction.parameters
        if opcode == "measure" or (opcode == "global_phase" and not controls):
            continue
        affine = tuple(
            _linear_angle_expression(item, circuit, parameters)
            for item in raw_parameters
        )
        if controls:
            if any(coefficients for _, coefficients in affine):
                raise ExecutionError(
                    "Q8721: controlled runtime gates cannot be projected to a "
                    "QURI Parts parametric circuit"
                )
            _add_controlled_instruction(
                circuit,
                opcode,
                controls,
                qubits,
                tuple(constant for constant, _ in affine),
            )
            continue
        _add_parametric_instruction(circuit, opcode, qubits, affine)


def _linear_angle_expression(
    value: CircuitValue,
    circuit: Any,
    parameters: dict[tuple[str, tuple[int | str, ...]], Any],
) -> tuple[float, dict[Any, float]]:
    """Convert one circuit expression to an affine QURI parameter mapping.

    Args:
        value (CircuitValue): Checked scalar expression.
        circuit: Mutable QURI Parts linear-mapped circuit.
        parameters: Previously registered flattened QURI parameters.

    Returns:
        Constant offset and QURI parameter coefficients.

    Raises:
        ExecutionError: If the expression is malformed or nonlinear.
    """

    if isinstance(value, bool) or value is None:
        raise ExecutionError("Q8722: parametric circuit angle must be real")
    if isinstance(value, (int, float)):
        return float(value), {}
    if isinstance(value, ParameterValue):
        name, path = value.name, value.path
        key = (name, path)
        parameter = parameters.get(key)
        if parameter is None:
            parameter = circuit.add_parameter(_parameter_projection_name(name, path))
            parameters[key] = parameter
        return 0.0, {parameter: 1.0}
    if not isinstance(value, ClassicalExpr):
        raise ExecutionError("Q8722: malformed parametric circuit expression")
    opcode, operands = value.opcode, value.operands
    resolved = tuple(
        _linear_angle_expression(item, circuit, parameters) for item in operands
    )
    if opcode in {"classical.pos", "classical.neg"} and len(resolved) == 1:
        scale = -1.0 if opcode == "classical.neg" else 1.0
        return _scale_linear_expression(resolved[0], scale)
    if opcode in {"classical.add", "classical.sub"} and len(resolved) == 2:
        right_scale = -1.0 if opcode == "classical.sub" else 1.0
        return _add_linear_expressions(
            resolved[0],
            _scale_linear_expression(resolved[1], right_scale),
        )
    if opcode == "classical.mul" and len(resolved) == 2:
        left, right = resolved
        if not left[1]:
            return _scale_linear_expression(right, left[0])
        if not right[1]:
            return _scale_linear_expression(left, right[0])
    if opcode == "classical.truediv" and len(resolved) == 2 and not resolved[1][1]:
        divisor = resolved[1][0]
        if divisor == 0.0:
            raise ExecutionError("Q8722: zero divisor in parametric circuit angle")
        return _scale_linear_expression(resolved[0], 1.0 / divisor)
    raise ExecutionError(
        f"Q8722: nonlinear circuit expression {opcode!r} cannot be represented "
        "by a QURI Parts linear parameter mapping"
    )


def _parameter_projection_name(name: str, path: tuple[int | str, ...]) -> str:
    """Return a stable readable name for one flattened parameter coordinate.

    Args:
        name: Aggregate Qamomile parameter name.
        path: Integer indices or record fields selecting one scalar leaf.

    Returns:
        Readable flattened name such as ``gammas[0]``.
    """

    projected = name
    for component in path:
        if isinstance(component, int):
            projected += f"[{component}]"
        else:
            projected += f".{component}"
    return projected


def _scale_linear_expression(
    expression: tuple[float, dict[Any, float]],
    scale: float,
) -> tuple[float, dict[Any, float]]:
    """Multiply an affine expression by one closed scalar.

    Args:
        expression: Constant and parameter coefficients to scale.
        scale: Closed real multiplier.

    Returns:
        Scaled constant and nonzero parameter coefficients.
    """

    constant, coefficients = expression
    return constant * scale, {
        parameter: coefficient * scale
        for parameter, coefficient in coefficients.items()
        if coefficient * scale != 0.0
    }


def _add_linear_expressions(
    left: tuple[float, dict[Any, float]],
    right: tuple[float, dict[Any, float]],
) -> tuple[float, dict[Any, float]]:
    """Add two affine expressions while preserving parameter identities.

    Args:
        left: Left constant and parameter coefficients.
        right: Right constant and parameter coefficients.

    Returns:
        Combined constant and nonzero parameter coefficients.
    """

    coefficients = dict(left[1])
    for parameter, coefficient in right[1].items():
        combined = coefficients.get(parameter, 0.0) + coefficient
        if combined == 0.0:
            coefficients.pop(parameter, None)
        else:
            coefficients[parameter] = combined
    return left[0] + right[0], coefficients


def _add_parametric_instruction(
    circuit: Any,
    opcode: str,
    qubits: list[int],
    parameters: tuple[tuple[float, dict[Any, float]], ...],
) -> None:
    """Emit one uncontrolled instruction with affine runtime angles.

    Args:
        circuit: Mutable QURI Parts linear-mapped circuit.
        opcode: Target-neutral logical gate name.
        qubits: Ordered logical target indices.
        parameters: Affine forms for every gate parameter.

    Raises:
        ExecutionError: If QURI Parts has no supported realization.
    """

    fixed = {
        "h": "add_H_gate",
        "x": "add_X_gate",
        "y": "add_Y_gate",
        "z": "add_Z_gate",
        "s": "add_S_gate",
        "sdg": "add_Sdag_gate",
        "t": "add_T_gate",
        "tdg": "add_Tdag_gate",
        "cx": "add_CNOT_gate",
        "cz": "add_CZ_gate",
        "ccx": "add_TOFFOLI_gate",
        "swap": "add_SWAP_gate",
    }
    if opcode in fixed:
        getattr(circuit, fixed[opcode])(*qubits)
        return
    if opcode in {"p", "rx", "ry", "rz"}:
        constant, coefficients = parameters[0]
        fixed_method = {
            "p": "add_U1_gate",
            "rx": "add_RX_gate",
            "ry": "add_RY_gate",
            "rz": "add_RZ_gate",
        }[opcode]
        parametric_method = {
            "p": "add_ParametricRZ_gate",
            "rx": "add_ParametricRX_gate",
            "ry": "add_ParametricRY_gate",
            "rz": "add_ParametricRZ_gate",
        }[opcode]
        if constant != 0.0:
            getattr(circuit, fixed_method)(qubits[0], constant)
        if coefficients:
            getattr(circuit, parametric_method)(qubits[0], coefficients)
        return
    if opcode == "rzz":
        constant, coefficients = parameters[0]
        if constant != 0.0:
            circuit.add_PauliRotation_gate(qubits, [3, 3], constant)
        if coefficients:
            circuit.add_ParametricPauliRotation_gate(qubits, [3, 3], coefficients)
        return
    if opcode == "cp":
        constant, coefficients = parameters[0]
        _add_affine_rz(circuit, qubits[1], constant, coefficients, 0.5)
        circuit.add_CNOT_gate(qubits[0], qubits[1])
        _add_affine_rz(circuit, qubits[1], constant, coefficients, -0.5)
        circuit.add_CNOT_gate(qubits[0], qubits[1])
        _add_affine_rz(circuit, qubits[0], constant, coefficients, 0.5)
        return
    raise ExecutionError(f"Q8712: unsupported QURI Parts gate {opcode!r}")


def _add_affine_rz(
    circuit: Any,
    qubit: int,
    constant: float,
    coefficients: dict[Any, float],
    scale: float,
) -> None:
    """Emit one affine RZ rotation, omitting its irrelevant global phase.

    Args:
        circuit: Mutable QURI Parts linear-mapped circuit.
        qubit: Logical target index.
        constant: Closed angle offset.
        coefficients: QURI parameter coefficients.
        scale: Multiplier applied to the complete affine angle.
    """

    if constant * scale != 0.0:
        circuit.add_RZ_gate(qubit, constant * scale)
    scaled = {
        parameter: coefficient * scale
        for parameter, coefficient in coefficients.items()
        if coefficient * scale != 0.0
    }
    if scaled:
        circuit.add_ParametricRZ_gate(qubit, scaled)


def _require_real(value: object) -> float:
    """Validate and normalize a runtime circuit angle."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExecutionError("Q8714: circuit angle must be a real number")
    return float(value)


def _add_native_instruction(
    circuit: Any,
    opcode: str,
    qubits: list[int],
    parameters: tuple[object, ...],
) -> None:
    """Emit one uncontrolled logical instruction with native QURI gates."""

    if opcode == "global_phase":
        return
    fixed = {
        "h": "add_H_gate",
        "x": "add_X_gate",
        "y": "add_Y_gate",
        "z": "add_Z_gate",
        "s": "add_S_gate",
        "sdg": "add_Sdag_gate",
        "t": "add_T_gate",
        "tdg": "add_Tdag_gate",
        "cx": "add_CNOT_gate",
        "cz": "add_CZ_gate",
        "ccx": "add_TOFFOLI_gate",
        "swap": "add_SWAP_gate",
    }
    if opcode in fixed:
        getattr(circuit, fixed[opcode])(*qubits)
        return
    if opcode in {"p", "rx", "ry", "rz"}:
        method = {
            "p": "add_U1_gate",
            "rx": "add_RX_gate",
            "ry": "add_RY_gate",
            "rz": "add_RZ_gate",
        }[opcode]
        getattr(circuit, method)(qubits[0], _require_real(parameters[0]))
        return
    if opcode == "cp":
        angle = _require_real(parameters[0])
        control, target = qubits
        circuit.add_RZ_gate(target, angle / 2)
        circuit.add_CNOT_gate(control, target)
        circuit.add_RZ_gate(target, -angle / 2)
        circuit.add_CNOT_gate(control, target)
        circuit.add_U1_gate(control, angle / 2)
        return
    if opcode == "rzz":
        circuit.add_PauliRotation_gate(qubits, [3, 3], _require_real(parameters[0]))
        return
    raise ExecutionError(f"Q8712: unsupported QURI Parts gate {opcode!r}")


def _add_controlled_instruction(
    circuit: Any,
    opcode: str,
    controls: list[int],
    qubits: list[int],
    parameters: tuple[object, ...],
) -> None:
    """Emit controlled SDK gates without a matrix over the control register.

    Both bound execution and public circuit projection use this path. Composite
    gates require at most three SDK gates, each with a linear-size control list.
    No work or allocation is exponential in the number of controls.

    Args:
        circuit (Any): Mutable QURI Parts circuit, optionally linear-mapped.
        opcode (str): Verified target-neutral gate name.
        controls (list[int]): Ordered positive control indices.
        qubits (list[int]): Ordered logical operands, disjoint from controls.
        parameters (tuple[object, ...]): Concrete gate angles in radians.

    Raises:
        ExecutionError: If the opcode or arity is unsupported, or the installed
            QURI Parts SDK lacks the required multi-control gate factory.
    """

    from quri_parts.circuit import gates

    widths = {
        "global_phase": 0,
        "cx": 2,
        "cz": 2,
        "ccx": 3,
        "cp": 2,
        "swap": 2,
        "rzz": 2,
    }
    if len(qubits) != widths.get(opcode, 1):
        raise ExecutionError("Q8713: controlled gate width mismatch")
    if opcode == "global_phase":
        # A phase conditioned on every control is a phase gate on the last one.
        opcode, qubits, controls = "p", controls[-1:], controls[:-1]
    elif opcode in {"cx", "cz", "ccx", "cp"}:
        controls = [*controls, *qubits[:-1]]
        qubits = qubits[-1:]
        opcode = {"cx": "x", "cz": "z", "ccx": "x", "cp": "p"}[opcode]
    elif opcode == "swap":
        left, right = qubits
        circuit.add_CNOT_gate(right, left)
        _add_controlled_instruction(circuit, "x", [*controls, left], [right], ())
        circuit.add_CNOT_gate(right, left)
        return
    elif opcode == "rzz":
        left, right = qubits
        circuit.add_CNOT_gate(left, right)
        _add_controlled_instruction(circuit, "rz", controls, [right], parameters)
        circuit.add_CNOT_gate(left, right)
        return

    if not controls:
        _add_native_instruction(circuit, opcode, qubits, parameters)
        return
    if opcode == "x" and len(controls) <= 2:
        method = (
            circuit.add_CNOT_gate if len(controls) == 1 else circuit.add_TOFFOLI_gate
        )
        method(*controls, qubits[0])
        return
    if opcode == "z" and len(controls) == 1:
        circuit.add_CZ_gate(controls[0], qubits[0])
        return
    names = {
        "x": "MCX",
        "y": "MCY",
        "z": "MCZ",
        "h": "MCH",
        "s": "MCS",
        "sdg": "MCSdag",
        "t": "MCT",
        "tdg": "MCTdag",
        "p": "MCU1",
        "rx": "MCRX",
        "ry": "MCRY",
        "rz": "MCRZ",
    }
    name = names.get(opcode)
    if name is None:
        raise ExecutionError(f"Q8712: unsupported controlled gate {opcode!r}")
    factory = getattr(gates, name, None)
    if not callable(factory):
        raise ExecutionError(
            f"Q8722: installed QURI Parts lacks {name}; upgrade quri-parts-circuit "
            "and quri-parts-qulacs for native multi-control gates"
        )
    if opcode in {"p", "rx", "ry", "rz"}:
        circuit.add_gate(
            factory(qubits[0], _require_real(parameters[0]), control_indices=controls)
        )
    else:
        circuit.add_gate(factory(qubits[0], control_indices=controls))
