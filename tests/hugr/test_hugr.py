"""Tests for direct Qamomile semantic IR to HUGR lowering."""

from __future__ import annotations

import math

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.stdlib import amplitude_encoding

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr.package import Package

from qamomile.circuit.transpiler.errors import EmitError, ValidationError
from qamomile.hugr import HugrTranspiler


@qmc.qkernel
def _hugr_bell(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
    """Build a parameterized Bell-like program for HUGR tests."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = qmc.ry(left, theta)
    left, right = qmc.cx(left, right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_helper(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply a reusable Hadamard operation."""
    return qmc.h(qubit)


@qmc.qkernel
def _hugr_helper_entrypoint() -> qmc.Bit:
    """Call a body-backed helper without forcing circuit-style inlining."""
    qubit = qmc.qubit("qubit")
    qubit = _hugr_helper(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_inline_feedback_helper(
    qubit: qmc.Qubit,
    trigger: qmc.Bit,
) -> tuple[qmc.Qubit, qmc.Bit]:
    """Update a callable while condition from a fresh measurement."""
    while trigger:
        qubit = qmc.reset(qubit)
        qubit, trigger = qmc.project_z(qubit)
    return qubit, trigger


@qmc.qkernel
def _hugr_measured_inline_feedback_call() -> qmc.Bit:
    """Pass a measurement-backed condition into an inline-policy helper."""
    qubit = qmc.x(qmc.qubit("qubit"))
    qubit, trigger = qmc.project_z(qubit)
    qubit, _ = _hugr_inline_feedback_helper(qubit, trigger)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_constant_inline_feedback_call() -> qmc.Bit:
    """Pass an invalid constant condition into an inline-policy helper."""
    qubit = qmc.qubit("qubit")
    qubit, _ = _hugr_inline_feedback_helper(qubit, True)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_array_helper(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply a reusable broadcast Hadamard operation."""
    return qmc.h(qubits)


@qmc.qkernel
def _hugr_static_for() -> qmc.Bit:
    """Exercise a statically bounded loop with a carried qubit."""
    qubit = qmc.qubit("qubit")
    for _ in qmc.range(3):
        qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_measurement_if() -> qmc.Bit:
    """Exercise native HUGR conditional dataflow with a carried qubit."""
    qubit = qmc.qubit("qubit")
    qubit, bit = qmc.project_z(qubit)
    if bit:
        qubit = qmc.x(qubit)
    else:
        qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_destructive_branch_array_merge() -> qmc.Bit:
    """Measure different elements of one captured array across branches."""
    qubits = qmc.qubit_array(3, "qubits")
    condition = qmc.measure(qubits[0])
    qubits[2] = qmc.x(qubits[2])
    if condition:
        result = qmc.measure(qubits[1])
    else:
        result = qmc.measure(qubits[2])
    return result


@qmc.qkernel
def _hugr_vector_program() -> qmc.Vector[qmc.Bit]:
    """Exercise fixed-size vector allocation, broadcast, and measurement."""
    qubits = qmc.qubit_array(3, "qubits")
    qubits = qmc.h(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_phase_helper(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply a phase gate used by inverse-call legalization tests."""
    return qmc.s(qubit)


@qmc.qkernel
def _hugr_inverse_program() -> qmc.Bit:
    """Invoke an inverse helper at a HUGR call site."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.inverse(_hugr_phase_helper)(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_x_helper(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply X for controlled-call legalization tests."""
    return qmc.x(qubit)


@qmc.composite_gate(name="shared_display_name")
def _hugr_same_name_h(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply H through the first of two equally named callables."""
    return qmc.h(qubit)


@qmc.composite_gate(name="shared_display_name")
def _hugr_same_name_x(qubit: qmc.Qubit) -> qmc.Qubit:
    """Apply X through the second of two equally named callables."""
    return qmc.x(qubit)


@qmc.qkernel
def _hugr_same_name_program() -> tuple[qmc.Bit, qmc.Bit]:
    """Invoke distinct callable bodies that share one display name."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left = _hugr_same_name_h(left)
    right = _hugr_same_name_x(right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_controlled_program() -> tuple[qmc.Bit, qmc.Bit]:
    """Invoke a one-control helper at a HUGR call site."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_hugr_x_helper)(control, target)
    return qmc.measure(control), qmc.measure(target)


@qmc.qkernel
def _hugr_state_preparation() -> qmc.Vector[qmc.Bit]:
    """Exercise a semantic state-preparation callable in HUGR.

    Returns:
        qmc.Vector[qmc.Bit]: Prepared-state measurements.
    """
    qubits = qmc.qubit_array(2, "qubits")
    qubits = amplitude_encoding(qubits, [1.0, 2.0, 3.0, 4.0])
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_ripple_carry() -> tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]:
    """Exercise a semantic ripple-carry callable in HUGR.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Bit, qmc.Bit]: Accumulator, carry,
            and overflow measurements.
    """
    left = qmc.qubit_array(2, "left")
    right = qmc.qubit_array(2, "right")
    carry = qmc.qubit("carry")
    overflow = qmc.qubit("overflow")
    _, right, carry, overflow = qmc.ripple_carry_add(left, right, carry, overflow)
    return qmc.measure(right), qmc.measure(carry), qmc.measure(overflow)


@qmc.qkernel
def _hugr_multi_controlled_x() -> qmc.Bit:
    """Exercise semantic MCX at HUGR's current two-control boundary.

    Returns:
        qmc.Bit: Target measurement.
    """
    controls = qmc.x(qmc.qubit_array(2, "controls"))
    target = qmc.qubit("target")
    _, target = qmc.mcx(controls, target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_measurement_while() -> qmc.Bit:
    """Exercise measurement-controlled while with carried quantum state."""
    qubit = qmc.qubit("qubit")
    qubit = qmc.x(qubit)
    qubit, bit = qmc.project_z(qubit)
    while bit:
        if bit:
            qubit = qmc.x(qubit)
        else:
            qubit = qmc.h(qubit)
        qubit = qmc.reset(qubit)
        qubit, bit = qmc.project_z(qubit)
    return bit


@qmc.qkernel
def _hugr_sequential_feedback() -> qmc.Bit:
    """Exercise two sequential measurement-dependent corrections."""
    source = qmc.qubit("source")
    ancilla = qmc.qubit("ancilla")
    target = qmc.qubit("target")
    source = qmc.h(source)
    source, ancilla = qmc.cx(source, ancilla)
    ancilla, target = qmc.cx(ancilla, target)
    first = qmc.measure(source)
    if first:
        target = qmc.z(target)
    second = qmc.measure(ancilla)
    if second:
        target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_loop_with_feedback() -> qmc.Bit:
    """Exercise a carried qubit through nested loop and conditional regions."""
    qubit = qmc.qubit("qubit")
    for _ in qmc.range(3):
        qubit, bit = qmc.project_z(qubit)
        if bit:
            qubit = qmc.x(qubit)
        else:
            qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_runtime_bounded_for(repetitions: qmc.UInt) -> qmc.Bit:
    """Exercise a runtime-bounded loop without circuit-style unrolling."""
    qubit = qmc.qubit("qubit")
    for _ in qmc.range(repetitions):
        qubit = qmc.h(qubit)
    return qmc.measure(qubit)


@qmc.qkernel
def _hugr_runtime_scalar_carry(repetitions: qmc.UInt) -> qmc.Bit:
    """Thread a scalar RegionArg through a runtime loop and nested if."""
    selector = qmc.measure(qmc.qubit("selector"))
    total = 0.0
    for _ in qmc.range(repetitions):
        if selector:
            total = total + math.pi
        else:
            total = total + 2.0 * math.pi
    target = qmc.qubit("target")
    target = qmc.rx(target, total)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_runtime_uint_carry(repetitions: qmc.UInt) -> qmc.UInt:
    """Accumulate a UInt RegionArg using the runtime loop index."""
    total = qmc.uint(0)
    for index in qmc.range(repetitions):
        total = total + index
    return total


@qmc.qkernel
def _hugr_runtime_uint_conditional_carry(repetitions: qmc.UInt) -> qmc.UInt:
    """Use a UInt loop comparison to guard a carried update."""
    total = qmc.uint(0)
    for index in qmc.range(repetitions):
        if index > 0:
            total = total + index
        else:
            total = total + 0
    return total


@qmc.qkernel
def _hugr_runtime_uint_descending(
    start: qmc.UInt,
    stop: qmc.UInt,
) -> qmc.UInt:
    """Use an unsigned descending runtime range comparison."""
    total = qmc.uint(0)
    for index in qmc.range(start, stop, -1):
        total = total + index
    return total


@qmc.qkernel
def _hugr_runtime_array_carry(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Carry a qubit array through a runtime loop and nested conditional."""
    selector = qmc.measure(qmc.qubit("selector"))
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(repetitions):
        if selector:
            qubits = qmc.h(qubits)
        else:
            qubits = qmc.x(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_runtime_array_broadcast(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Carry a broadcast-updated qubit array through a runtime loop."""
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(repetitions):
        qubits = qmc.h(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_runtime_array_element(repetitions: qmc.UInt) -> qmc.Bit:
    """Carry one array element through a runtime loop."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qubits[0]
    for _ in qmc.range(repetitions):
        target = qmc.h(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_runtime_scalar_call(repetitions: qmc.UInt) -> qmc.Bit:
    """Carry a scalar qubit returned by a direct call through a runtime loop."""
    target = qmc.qubit("target")
    for _ in qmc.range(repetitions):
        target = _hugr_helper(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_runtime_array_call(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Carry a qubit array returned by a direct call through a runtime loop."""
    qubits = qmc.qubit_array(2, "qubits")
    for _ in qmc.range(repetitions):
        qubits = _hugr_array_helper(qubits)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_runtime_quantum_swap(
    repetitions: qmc.UInt,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Expose quantum handle rebinding that needs explicit linear slots."""
    left = qmc.x(qmc.qubit("left"))
    right = qmc.qubit("right")
    for _ in qmc.range(repetitions):
        left, right = right, left
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_runtime_if_qubit(repetitions: qmc.UInt) -> qmc.Bit:
    """Carry a scalar qubit through a runtime loop and nested conditional."""
    selector = qmc.measure(qmc.qubit("selector"))
    target = qmc.qubit("target")
    for _ in qmc.range(repetitions):
        if selector:
            target = qmc.x(target)
        else:
            target = qmc.h(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_while_array_broadcast() -> qmc.Vector[qmc.Bit]:
    """Carry a broadcast-updated qubit array through a measurement while."""
    control = qmc.x(qmc.qubit("control"))
    control, trigger = qmc.project_z(control)
    qubits = qmc.qubit_array(2, "qubits")
    while trigger:
        qubits = qmc.h(qubits)
        control = qmc.reset(control)
        control, trigger = qmc.project_z(control)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_vector_element_if() -> qmc.Bit:
    """Use a measured vector element as a native conditional predicate."""
    bits = qmc.measure(qmc.qubit_array(2, "condition"))
    target = qmc.qubit("target")
    if bits[0]:
        target = qmc.x(target)
    else:
        target = qmc.h(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_vector_element_merge() -> qmc.Bit:
    """Select between two measured array elements in a native conditional."""
    data = qmc.measure(qmc.qubit_array(2, "data"))
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        result = data[0]
    else:
        result = data[1]
    return result


@qmc.qkernel
def _hugr_runtime_quantum_index(
    repetitions: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:
    """Expose dynamic indexing of a flattened linear quantum array."""
    qubits = qmc.qubit_array(2, "qubits")
    for index in qmc.range(repetitions):
        qubits[index] = qmc.x(qubits[index])
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_quantum_element_merge() -> qmc.Bit:
    """Expose data-dependent selection between different linear qubits."""
    qubits = qmc.qubit_array(2, "qubits")
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        target = qubits[0]
    else:
        target = qubits[1]
    return qmc.measure(target)


@qmc.qkernel
def _hugr_duplicate_quantum_footprint() -> qmc.Bit:
    """Expose whole-array and element aliases as duplicate merge outputs."""
    qubits = qmc.qubit_array(2, "qubits")
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        qubits[0] = qmc.x(qubits[0])
        target = qubits[0]
    else:
        qubits[0] = qmc.h(qubits[0])
        target = qubits[0]
    return qmc.measure(target)


@qmc.qkernel
def _hugr_local_duplicate_quantum_footprint() -> tuple[qmc.Bit, qmc.Bit]:
    """Expose duplicate aliases of branch-local quantum resources."""
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        qubits = qmc.qubit_array(2, "true_qubits")
        qubits[0] = qmc.x(qubits[0])
        target = qubits[0]
    else:
        qubits = qmc.qubit_array(2, "false_qubits")
        qubits[0] = qmc.h(qubits[0])
        target = qubits[0]
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_conditional_quantum_swap() -> tuple[qmc.Bit, qmc.Bit]:
    """Permute two complete linear resources through a conditional."""
    left = qmc.x(qmc.qubit("left"))
    right = qmc.qubit("right")
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        left, right = right, left
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_branch_local_unused_qubit() -> qmc.Bit:
    """Allocate an unused companion qubit inside each conditional branch."""
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        qubits = qmc.qubit_array(2, "true_qubits")
        target = qubits[0]
    else:
        qubits = qmc.qubit_array(2, "false_qubits")
        target = qubits[0]
    return qmc.measure(target)


@qmc.qkernel
def _hugr_loop_local_unused_qubit(repetitions: qmc.UInt) -> qmc.Bit:
    """Allocate a loop-local scratch qubit that does not escape the body."""
    for _ in qmc.range(repetitions):
        qmc.qubit("scratch")
    return qmc.measure(qmc.qubit("output"))


@qmc.qkernel
def _hugr_disjoint_array_alias_if() -> tuple[qmc.Bit, qmc.Bit]:
    """Carry an element alias and a disjoint sibling through an if."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qmc.h(qubits[0])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        target = qmc.x(target)
        qubits[1] = qmc.h(qubits[1])
    else:
        target = qmc.z(target)
        qubits[1] = qmc.x(qubits[1])
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_single_array_element_alias_if() -> tuple[qmc.Bit, qmc.Bit]:
    """Carry one element alias while its sibling stays outside the if."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qmc.h(qubits[0])
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        target = qmc.x(target)
    else:
        target = qmc.z(target)
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_disjoint_array_alias_for(
    repetitions: qmc.UInt,
) -> tuple[qmc.Bit, qmc.Bit]:
    """Carry an element alias and a disjoint sibling through a loop."""
    qubits = qmc.qubit_array(2, "qubits")
    target = qmc.h(qubits[0])
    for _ in qmc.range(repetitions):
        target = qmc.x(target)
        qubits[1] = qmc.h(qubits[1])
    return qmc.measure(target), qmc.measure(qubits[1])


@qmc.qkernel
def _hugr_while_scalar_carry() -> qmc.UInt:
    """Expose the unsupported non-condition scalar carry in a while loop."""
    trigger = qmc.measure(qmc.qubit("trigger"))
    total = qmc.uint(0)
    while trigger:
        total = total + 1
        trigger = qmc.measure(qmc.qubit("next_trigger"))
    return total


@qmc.qkernel
def _hugr_register_feedback() -> qmc.Vector[qmc.Bit]:
    """Correct an indexed register element from a measurement predicate."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits[0], syndrome = qmc.project_z(qubits[0])
    if syndrome:
        qubits[1] = qmc.x(qubits[1])
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_independent_loop_carriers() -> tuple[qmc.Bit, qmc.Bit]:
    """Carry two independent linear qubits through one loop."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    for _ in qmc.range(2):
        left = qmc.h(left)
        right = qmc.x(right)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_gate_coverage() -> tuple[qmc.Bit, qmc.Bit]:
    """Exercise composite two-qubit gates and destructive reset lowering."""
    left = qmc.qubit("left")
    right = qmc.qubit("right")
    left, right = qmc.swap(left, right)
    left, right = qmc.rzz(left, right, 0.25)
    left, right = qmc.cp(left, right, -0.5)
    left = qmc.reset(left)
    return qmc.measure(left), qmc.measure(right)


@qmc.qkernel
def _hugr_two_control_program() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Legalize a two-control reusable X call."""
    controls = qmc.qubit_array(2, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(_hugr_x_helper, num_controls=2)(
        controls,
        target,
    )
    return (
        qmc.measure(controls[0]),
        qmc.measure(controls[1]),
        qmc.measure(target),
    )


@qmc.qkernel
def _hugr_pauli_evolution(
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercise semantic Pauli evolution at the HUGR target boundary."""
    qubits = qmc.qubit_array(2, "qubits")
    qubits = qmc.h(qubits)
    qubits = qmc.pauli_evolve(qubits, hamiltonian, gamma)
    return qmc.measure(qubits)


@qmc.qkernel
def _hugr_pauli_helper(
    qubits: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Apply Pauli evolution through a transformable helper body."""
    return qmc.pauli_evolve(qubits, hamiltonian, gamma)


@qmc.qkernel
def _hugr_controlled_pauli_evolution(
    hamiltonian: qmc.Observable,
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Bit]:
    """Exercise a controlled semantic Pauli evolution at the HUGR edge."""
    control = qmc.qubit("control")
    targets = qmc.qubit_array(2, "targets")
    control = qmc.h(control)
    control, targets = qmc.control(_hugr_pauli_helper)(
        control,
        targets,
        hamiltonian=hamiltonian,
        gamma=gamma,
    )
    return qmc.measure(targets)


@pytest.mark.hugr
def test_hugr_compiles_bound_quantum_program_and_validates() -> None:
    """A bound quantum program produces a validator-clean HUGR package."""
    compiled = HugrTranspiler().transpile(
        _hugr_bell,
        bindings={"theta": math.pi / 2},
    )

    assert isinstance(compiled.artifact, Package)
    assert compiled.metadata.target == "hugr"
    assert compiled.metadata.pipeline == "program_graph"


@pytest.mark.hugr
def test_hugr_preserves_runtime_parameter_as_public_input() -> None:
    """A runtime angle remains a Float input of the public HUGR function."""
    package = HugrTranspiler().to_hugr(_hugr_bell, parameters=["theta"])
    operations = [data.op for _, data in package.modules[0].nodes()]
    [main] = [op for op in operations if getattr(op, "f_name", None) == "main"]

    assert len(main.inputs) == 1
    assert "float64" in str(main.inputs[0])


@pytest.mark.hugr
def test_hugr_preserves_body_backed_helper_as_function_call() -> None:
    """Hierarchical helper semantics lower to HUGR definition and call nodes."""
    package = HugrTranspiler().to_hugr(_hugr_helper_entrypoint)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(
        getattr(op, "f_name", "").endswith("___hugr_helper__v1") for op in operations
    )
    assert any(type(op).__name__ == "Call" for op in operations)


@pytest.mark.hugr
def test_hugr_validates_inline_callable_while_at_its_call_site() -> None:
    """Contextual while validation keeps hierarchical HUGR call emission."""
    package = HugrTranspiler().to_hugr(_hugr_measured_inline_feedback_call)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(type(op).__name__ == "Call" for op in operations)
    assert any(type(op).__name__ == "TailLoop" for op in operations)


@pytest.mark.hugr
def test_hugr_rejects_nonmeasurement_inline_callable_while_call() -> None:
    """Inlining for validation still rejects a constant while condition."""
    with pytest.raises(ValidationError, match="measurement result"):
        HugrTranspiler().to_hugr(_hugr_constant_inline_feedback_call)


@pytest.mark.hugr
def test_hugr_distinguishes_callables_with_the_same_display_name() -> None:
    """Origin-qualified symbols prevent same-name callable miscompilation."""
    package = HugrTranspiler().to_hugr(_hugr_same_name_program)
    operations = [data.op for _, data in package.modules[0].nodes()]
    definitions = [
        op
        for op in operations
        if getattr(op, "f_name", "").endswith("__shared_display_name__v1")
    ]

    assert len(definitions) == 2
    assert any("name='H'" in str(op) for op in operations)
    assert any("name='X'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_package_serialization_round_trip_validates() -> None:
    """Serialized HUGR remains native-validator clean after deserialization."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_helper_entrypoint)
    restored = Package.from_bytes(package.to_bytes())

    transpiler.target.validate(restored)


@pytest.mark.hugr
def test_hugr_lowers_static_for_with_carried_quantum_ssa() -> None:
    """Static loops lower deterministically and remain validator-clean."""
    package = HugrTranspiler().to_hugr(_hugr_static_for)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum("name='H'" in str(op) for op in operations) == 3


@pytest.mark.hugr
def test_hugr_preserves_measurement_if_as_native_conditional() -> None:
    """Measurement-backed branching remains a HUGR Conditional region."""
    package = HugrTranspiler().to_hugr(_hugr_measurement_if)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(type(op).__name__ == "Conditional" for op in operations)


@pytest.mark.hugr
def test_hugr_captures_indexed_register_feedback_as_linear_region_input() -> None:
    """Indexed qubits cross Conditional boundaries without parent-graph edges."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_register_feedback)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert any(type(op).__name__ == "Conditional" for op in operations)


@pytest.mark.hugr
def test_hugr_keeps_independent_loop_carriers_distinct() -> None:
    """Linear-wire alias advancement cannot cross two independent qubits."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_independent_loop_carriers)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='H'" in str(op) for op in operations) == 2
    assert sum("name='X'" in str(op) for op in operations) == 2


@pytest.mark.hugr
def test_hugr_validates_composite_gate_and_reset_lowering() -> None:
    """SWAP, RZZ, CP, and reset produce native-validator-clean HUGR."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_gate_coverage)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert any("name='Reset'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_validates_two_control_call_legalization() -> None:
    """A two-control X call lowers without duplicating linear controls."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_two_control_program)

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert any("name='Toffoli'" in str(op) for op in operations)


@pytest.mark.hugr
@pytest.mark.parametrize(
    "kernel",
    [_hugr_state_preparation, _hugr_ripple_carry, _hugr_multi_controlled_x],
)
def test_hugr_validates_semantic_composite_fallbacks(kernel: qmc.QKernel) -> None:
    """New semantic composites remain validator-clean HUGR callables."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(kernel)

    transpiler.target.validate(package)


@pytest.mark.hugr
def test_hugr_lowers_pauli_evolution_only_at_target_boundary() -> None:
    """A semantic Hamiltonian evolution becomes validator-clean TKET ops."""
    hamiltonian = 0.25 * qm_o.X(0) + 0.5 * qm_o.Y(0) * qm_o.Z(1)
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_pauli_evolution,
        bindings={"hamiltonian": hamiltonian},
        parameters=["gamma"],
    )

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='Rz'" in str(op) for op in operations) == 2
    assert any("name='CX'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_lowers_controlled_pauli_evolution_at_target_boundary() -> None:
    """A one-control Pauli evolution uses TKET controlled rotations."""
    hamiltonian = 0.25 * qm_o.X(0) + 0.5 * qm_o.Y(0) * qm_o.Z(1)
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_controlled_pauli_evolution,
        bindings={"hamiltonian": hamiltonian, "gamma": 0.125},
    )

    transpiler.target.validate(package)
    operations = [data.op for _, data in package.modules[0].nodes()]
    assert sum("name='CRz'" in str(op) for op in operations) == 2


@pytest.mark.hugr
def test_hugr_materializes_fixed_vectors_as_tuple_carriers() -> None:
    """Fixed vectors use HUGR tuples at boundaries and validate natively."""
    package = HugrTranspiler().to_hugr(_hugr_vector_program)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum("name='QAlloc'" in str(op) for op in operations) == 3
    assert any(type(op).__name__ == "MakeTuple" for op in operations)


@pytest.mark.hugr
def test_hugr_legalizes_inverse_calls_at_the_call_site() -> None:
    """Inverse calls inline as adjoint primitives while direct calls remain calls."""
    package = HugrTranspiler().to_hugr(_hugr_inverse_program)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any("name='Sdg'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_legalizes_controlled_x_calls_to_tket_cx() -> None:
    """One-control X helpers lower to the TKET CX extension operation."""
    package = HugrTranspiler().to_hugr(_hugr_controlled_program)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any("name='CX'" in str(op) for op in operations)


@pytest.mark.hugr
def test_hugr_preserves_measurement_while_as_tail_loop() -> None:
    """Measurement feedback lowers to a validator-clean native TailLoop."""
    package = HugrTranspiler().to_hugr(_hugr_measurement_while)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(type(op).__name__ == "TailLoop" for op in operations)
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_preserves_sequential_measurement_feedback_regions() -> None:
    """Sequential feedback corrections remain separate conditional regions."""
    package = HugrTranspiler().to_hugr(_hugr_sequential_feedback)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_lowers_nested_loop_feedback_with_linear_qubit_carry() -> None:
    """Loop-unrolled conditionals preserve the carried qubit linearly."""
    package = HugrTranspiler().to_hugr(_hugr_loop_with_feedback)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 3


@pytest.mark.hugr
def test_hugr_preserves_runtime_bounded_for_as_tail_loop() -> None:
    """A runtime range bound remains a compact native HUGR loop."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_bounded_for,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum("name='H'" in str(op) for op in operations) == 1


@pytest.mark.hugr
def test_hugr_threads_region_args_through_runtime_for() -> None:
    """Runtime TailLoop state includes scalar RegionArgs and nested merges."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_scalar_carry,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_threads_uint_region_arg_through_runtime_for() -> None:
    """Runtime UInt arithmetic accepts RegionArg and loop-index wires."""
    HugrTranspiler().to_hugr(
        _hugr_runtime_uint_carry,
        parameters=["repetitions"],
    )


@pytest.mark.hugr
def test_hugr_lowers_uint_comparison_inside_runtime_for() -> None:
    """UInt comparisons compose with RegionArgs and nested conditionals."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_uint_conditional_carry,
        parameters=["repetitions"],
    )

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
@pytest.mark.parametrize(
    ("kernel", "parameters", "comparison_name"),
    [
        (_hugr_runtime_uint_carry, ["repetitions"], "ilt_u"),
        (_hugr_runtime_uint_descending, ["start", "stop"], "igt_u"),
    ],
)
def test_hugr_runtime_uint_ranges_use_unsigned_comparisons(
    kernel,
    parameters: list[str],
    comparison_name: str,
) -> None:
    """Runtime UInt bounds retain high-bit unsigned ordering semantics."""
    package = HugrTranspiler().to_hugr(kernel, parameters=parameters)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert any(comparison_name in str(operation) for operation in operations)


@pytest.mark.hugr
def test_hugr_flattens_array_captures_through_nested_runtime_control() -> None:
    """Runtime TailLoops flatten arrays and advance nested branch merges."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_array_carry,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
@pytest.mark.parametrize(
    "kernel",
    [
        _hugr_runtime_array_broadcast,
        _hugr_runtime_array_element,
        _hugr_runtime_array_call,
    ],
)
def test_hugr_preserves_array_capture_shapes_through_runtime_for(kernel) -> None:
    """Runtime TailLoops preserve whole-array and array-element carriers."""
    package = HugrTranspiler().to_hugr(kernel, parameters=["repetitions"])
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_advances_direct_call_result_through_runtime_for() -> None:
    """A same-resource direct call advances the captured scalar wire."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_scalar_call,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_rejects_quantum_resource_rebinding_in_runtime_for() -> None:
    """Quantum handle swaps fail before producing an invalid TailLoop."""
    with pytest.raises(EmitError, match="quantum resource rebinding"):
        HugrTranspiler().to_hugr(
            _hugr_runtime_quantum_swap,
            parameters=["repetitions"],
        )


@pytest.mark.hugr
def test_hugr_publishes_nested_scalar_quantum_merge_after_runtime_for() -> None:
    """A nested scalar quantum merge remains visible after a runtime loop."""
    package = HugrTranspiler().to_hugr(
        _hugr_runtime_if_qubit,
        parameters=["repetitions"],
    )
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1
    assert sum(type(op).__name__ == "Conditional" for op in operations) == 2


@pytest.mark.hugr
def test_hugr_preserves_array_capture_shape_through_while() -> None:
    """A measurement while carries every wire of a broadcast array."""
    package = HugrTranspiler().to_hugr(_hugr_while_array_broadcast)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "TailLoop" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_resolves_measured_vector_element_if_condition() -> None:
    """Structural measured elements resolve as native HUGR predicates."""
    package = HugrTranspiler().to_hugr(_hugr_vector_element_if)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_resolves_classical_array_element_branch_yields() -> None:
    """Copyable array elements can cross native conditional outputs."""
    package = HugrTranspiler().to_hugr(_hugr_vector_element_merge)
    operations = [data.op for _, data in package.modules[0].nodes()]

    assert sum(type(op).__name__ == "Conditional" for op in operations) == 1


@pytest.mark.hugr
def test_hugr_rejects_dynamic_quantum_array_index_with_targeted_error() -> None:
    """Dynamic tuple indexing fails at the explicit HUGR support boundary."""
    with pytest.raises(EmitError, match="dynamic quantum array indexing"):
        HugrTranspiler().to_hugr(
            _hugr_runtime_quantum_index,
            parameters=["repetitions"],
        )


@pytest.mark.hugr
def test_hugr_rejects_data_dependent_quantum_element_selection() -> None:
    """Different captured qubits cannot share one semantic merge port."""
    with pytest.raises(EmitError, match="quantum resource selection"):
        HugrTranspiler().to_hugr(_hugr_quantum_element_merge)


@pytest.mark.hugr
def test_hugr_rejects_duplicate_quantum_merge_footprints() -> None:
    """Overlapping semantic aliases fail before native HUGR validation."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_duplicate_quantum_footprint)


@pytest.mark.hugr
def test_hugr_rejects_duplicate_branch_local_quantum_footprints() -> None:
    """Branch-local root and element aliases cannot duplicate one wire."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_local_duplicate_quantum_footprint)


@pytest.mark.hugr
def test_hugr_allows_complete_conditional_quantum_permutations() -> None:
    """A conditional may permute a complete set of linear merge ports."""
    package = HugrTranspiler().to_hugr(_hugr_conditional_quantum_swap)

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_frees_unyielded_branch_local_qubits() -> None:
    """Conditional regions free local linear resources that do not escape."""
    package = HugrTranspiler().to_hugr(_hugr_branch_local_unused_qubit)

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_rejects_consumed_qubit_in_implicit_array_merge() -> None:
    """A stale implicit root merge fails before native HUGR validation."""
    with pytest.raises(EmitError, match="destructively consumed"):
        HugrTranspiler().to_hugr(_hugr_destructive_branch_array_merge)


@pytest.mark.hugr
@pytest.mark.parametrize("runtime", [False, True])
def test_hugr_frees_unyielded_loop_local_qubits(runtime: bool) -> None:
    """Static and runtime loop bodies clean up local linear allocations."""
    kwargs = (
        {"parameters": ["repetitions"]} if runtime else {"bindings": {"repetitions": 2}}
    )
    package = HugrTranspiler().to_hugr(
        _hugr_loop_local_unused_qubit,
        **kwargs,
    )

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_rejects_overlapping_array_alias_merges_through_if() -> None:
    """Whole-array and element aliases fail before duplicating a HUGR port."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_disjoint_array_alias_if)


@pytest.mark.hugr
def test_hugr_rejects_element_alias_with_implicit_whole_array_merge() -> None:
    """An element alias cannot duplicate its implicit whole-array merge."""
    with pytest.raises(EmitError, match="same linear quantum resource"):
        HugrTranspiler().to_hugr(_hugr_single_array_element_alias_if)


@pytest.mark.hugr
def test_hugr_captures_disjoint_array_aliases_once_through_runtime_for() -> None:
    """A TailLoop carries one root tuple instead of duplicating an element."""
    package = HugrTranspiler().to_hugr(
        _hugr_disjoint_array_alias_for,
        parameters=["repetitions"],
    )

    HugrTranspiler().target.validate(package)


@pytest.mark.hugr
def test_hugr_static_zero_trip_publishes_region_arg_initializer() -> None:
    """A zero-trip loop publishes its constant RegionArg initializer."""
    HugrTranspiler().to_hugr(
        _hugr_runtime_scalar_carry,
        bindings={"repetitions": 0},
    )


@pytest.mark.hugr
def test_hugr_rejects_while_scalar_carry_before_lowering() -> None:
    """Residual while scalar carries fail with a targeted diagnostic."""
    with pytest.raises(EmitError, match="loop-carried classical values.*while"):
        HugrTranspiler().to_hugr(_hugr_while_scalar_carry)


@pytest.mark.hugr
def test_hugr_nested_measurement_control_executes_on_selene(tmp_path) -> None:
    """The nested while/if package compiles and terminates on Selene."""
    selene = pytest.importorskip("selene_sim")
    package = HugrTranspiler().to_hugr(_hugr_measurement_while)
    runner = selene.build(
        package,
        name="qamomile_hugr_control_flow",
        build_dir=tmp_path,
    )

    results = list(
        runner.run(
            selene.Quest(),
            n_qubits=1,
            random_seed=1,
            timeout=15.0,
        )
    )

    assert results == []
