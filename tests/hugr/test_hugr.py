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


@qmc.qkernel
def _hugr_bit_comparisons() -> tuple[qmc.Bit, qmc.Bit]:
    """Compare two measurement-derived bits.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Equality and inequality results.
    """
    left = qmc.measure(qmc.qubit("left"))
    right = qmc.measure(qmc.qubit("right"))
    return left == right, left != right


@qmc.qkernel
def _hugr_bit_comparison_if() -> qmc.Bit:
    """Use measurement-derived Bit equality as a conditional predicate.

    Returns:
        qmc.Bit: Measurement of the conditionally updated target qubit.
    """
    left = qmc.measure(qmc.qubit("left"))
    right = qmc.measure(qmc.qubit("right"))
    target = qmc.qubit("target")
    if left == right:
        target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _hugr_uint_comparisons(
    left: qmc.UInt,
    right: qmc.UInt,
) -> tuple[qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit]:
    """Compare two runtime UInt values with every supported relation.

    Args:
        left (qmc.UInt): Left comparison operand.
        right (qmc.UInt): Right comparison operand.

    Returns:
        tuple[qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit, qmc.Bit]:
            Equality, inequality, less-than, less-than-or-equal, greater-than,
            and greater-than-or-equal results.
    """
    return (
        left == right,
        left != right,
        left < right,
        left <= right,
        left > right,
        left >= right,
    )


@qmc.qkernel
def _hugr_mixed_numeric_comparisons(
    integer: qmc.UInt,
    real: qmc.Float,
) -> tuple[
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
    qmc.Bit,
]:
    """Compare runtime UInt and Float values in both operand orders.

    Args:
        integer (qmc.UInt): Unsigned-integer comparison operand.
        real (qmc.Float): Floating-point comparison operand.

    Returns:
        tuple[qmc.Bit, ...]: All six comparison results in each operand order.
    """
    return (
        integer == real,
        integer != real,
        integer < real,
        integer <= real,
        integer > real,
        integer >= real,
        real == integer,
        real != integer,
        real < integer,
        real <= integer,
        real > integer,
        real >= integer,
    )


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
def test_hugr_lowers_bit_comparisons() -> None:
    """Measurement-derived Bit equality uses validator-clean Bool operations."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(_hugr_bit_comparisons)

    transpiler.target.validate(package)
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert "name='eq'" in operations
    assert "name='xor'" in operations

    conditional_package = transpiler.to_hugr(_hugr_bit_comparison_if)
    transpiler.target.validate(conditional_package)
    conditional_operations = [
        data.op for _, data in conditional_package.modules[0].nodes()
    ]
    assert any(
        type(operation).__name__ == "Conditional"
        for operation in conditional_operations
    )


@pytest.mark.hugr
def test_hugr_lowers_uint_comparisons() -> None:
    """Runtime UInt relations use unsigned integer comparison operations."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_uint_comparisons,
        parameters=["left", "right"],
    )

    transpiler.target.validate(package)
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    for name in ("ieq", "ine", "ilt_u", "ile_u", "igt_u", "ige_u"):
        assert f"name='{name}'" in operations


@pytest.mark.hugr
def test_hugr_lowers_mixed_numeric_comparisons() -> None:
    """The HUGR target converts UInt wires immediately before Float comparison."""
    transpiler = HugrTranspiler()
    package = transpiler.to_hugr(
        _hugr_mixed_numeric_comparisons,
        parameters=["integer", "real"],
    )

    transpiler.target.validate(package)
    operations = "\n".join(str(data.op) for _, data in package.modules[0].nodes())
    assert operations.count("name='convert_u'") == 12
    for name in ("feq", "fne", "flt", "fle", "fgt", "fge"):
        assert operations.count(f"name='{name}'") == 2


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
