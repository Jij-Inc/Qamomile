"""Const-bound slices and immutable updates preserve typed classical arrays."""

import math

import pytest

import qamomile as qm
from qamomile.persistence import deserialize, serialize

SLICE_SOURCE = """
import qamomile as qm
@qm.qkernel
def first(window: qm.Const[qm.Vector[qm.Int, n]]) -> qm.Int:
    return window[0]
@qm.qkernel
def sliced(values: qm.Const[qm.Vector[qm.Int, 5]], start: qm.Const[qm.Int], stop: qm.Const[qm.Int], step: qm.Const[qm.Int]) -> qm.Int:
    window = values[start:stop:step]
    return len(window) * 100 + first(window)
"""

UPDATE_SOURCE = """
import qamomile as qm
@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bits[2]], qm.Shot[qm.Bits[2]]]:
    bits = qm.measure(qm.x(qm.qubits(2), target=0))
    original = bits
    bits[1] = bits[0]
    return bits, original
"""

LAST_ANGLE_SOURCE = """
import qamomile as qm
@qm.qkernel
def rotate(n: qm.Const[qm.Int], row: qm.Param[qm.Vector[qm.Real, n]]) -> qm.Q[qm.Bits[2]]:
    return qm.ry(qm.qubits(2), {angle}, target=0)
@qm.qkernel
def nested(n: qm.Const[qm.Int], row: qm.Param[qm.Vector[qm.Real, n]]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(rotate(n, row))
@qm.qkernel
def inline(n: qm.Const[qm.Int], row: qm.Param[qm.Vector[qm.Real, n]]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(qm.ry(qm.qubits(2), {angle}, target=0))
"""


@pytest.mark.parametrize(
    "start,stop,step",
    [
        (1, 5, 1),
        (-3, 5, 1),
        (4, -6, -2),
        (-100, 100, 2),
        (0, 4, 1),
        (-(2**63), 2**63 - 1, 1),
    ],
)
def test_const_slice_matches_python_and_passes_complete_array(start, stop, step):
    """Specialization resolves lengths and full-array helper arguments, including zero and negative bounds."""
    values = (10, 11, 12, 13, 14)
    selected = values[start:stop:step]
    program = qm.build_source(SLICE_SOURCE).specialize(
        "sliced", values=values, start=start, stop=stop, step=step
    )
    restored = qm.Program.from_bytes(program.to_bytes())
    assert qm.evaluate(restored) == len(selected) * 100 + selected[0]


@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_original_const_slice_regression(endpoint):
    """The reported angles[start:] witness runs through the shared circuit projection."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(angles: qm.Const[qm.Vector[qm.Real, 2]], start: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    window = angles[start:]
    return qm.measure(qm.rx(qm.qubits(1), window[0]))
"""
    program = qm.build_source(source).specialize(
        "sample", angles=(0.0, math.pi), start=1
    )
    restored = qm.Program.from_bytes(program.to_bytes())
    assert qm.run(restored, on=endpoint, shots=8, seed=7).result.counts == {(1,): 8}


@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_measured_update_preserves_alias_and_result_shape(endpoint):
    """Updating a measured Bits value leaves aliases of its previous value unchanged."""
    program = qm.build_source(UPDATE_SOURCE).specialize("sample")
    program = qm.Program.from_bytes(program.to_bytes())
    assert qm.run(program, on=endpoint, shots=8, seed=7).result.values == {
        ((1, 1), (1, 0)): 8
    }


@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_measured_updates_inside_branch_and_loop(endpoint):
    """Negative indices and loop-carried measured updates keep their Shot stage."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    bits = qm.measure(qm.x(qm.qubits(3), target=0))
    if bits[0]:
        for i in range(1, 3):
            bits[i] = bits[0]
        bits[-1] = False
    return bits
"""
    program = qm.build_source(source).specialize("sample")
    assert qm.run(program, on=endpoint, shots=8, seed=7).result.counts == {(1, 1, 0): 8}


def test_param_array_update_is_functional_and_checked():
    """The native evaluator handles runtime indices without mutating an old value."""
    source = """
import qamomile as qm
@qm.qkernel
def update(values: qm.Param[qm.Vector[qm.Int, 3]], index: qm.Param[qm.Int], value: qm.Param[qm.Int]) -> tuple[qm.Param[qm.Vector[qm.Int, 3]], qm.Param[qm.Vector[qm.Int, 3]]]:
    original = values
    values[index] = value
    return values, original
"""
    program = qm.build_source(source).specialize("update")
    assert qm.evaluate(program, values=(1, 2, 3), index=-1, value=9) == (
        (1, 2, 9),
        (1, 2, 3),
    )
    with pytest.raises(qm.ExecutionError, match="out of bounds"):
        qm.evaluate(program, values=(1, 2, 3), index=3, value=9)


@pytest.mark.parametrize(
    "start,stop,step,diagnostic",
    [(1, 1, 1, "empty slice"), (0, 5, 0, "step cannot be zero")],
)
def test_invalid_const_slice_shape_fails_at_specialization(
    start, stop, step, diagnostic
):
    """Unrepresentable Const slice shapes fail before any backend execution."""
    module = qm.build_source(SLICE_SOURCE)
    with pytest.raises(qm.SpecializationError, match=diagnostic):
        module.specialize(
            "sliced", values=(0, 1, 2, 3, 4), start=start, stop=stop, step=step
        )


@pytest.mark.parametrize("target", ["state[0]", "values[:1]", "values[0, 0]"])
def test_element_assignment_rejects_quantum_or_unsupported_targets(target):
    """Only one local classical aggregate element can be rebound."""
    source = f"""
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    values = [1, 2]
    {target} = 1
    return qm.measure(state)
"""
    with pytest.raises((qm.FrontendError, qm.TypeCheckError)):
        qm.build_source(source)


def test_const_slice_loop_and_nested_slice():
    """Derived slice shapes survive another slice and ordinary aggregate iteration."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(values: qm.Const[qm.Vector[qm.Int, 5]], start: qm.Const[qm.Int]) -> qm.Int:
    window = values[start:][::-1]
    total = 0
    for item in window:
        total += item
    return total
"""
    program = qm.build_source(source).specialize(
        "sample", values=(1, 2, 3, 4, 5), start=-3
    )
    assert qm.evaluate(qm.Program.from_bytes(program.to_bytes())) == 12


@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_const_slice_of_param_array_compiles_symbolic_angles(endpoint):
    """A Const slice selects runtime coordinates through their existing parameter ABI."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real, 3]], start: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    window = angles[start:]
    return qm.measure(qm.rx(qm.qubits(1), window[0]))
"""
    program = qm.build_source(source).specialize("sample", start=1)
    assert qm.run(
        program,
        on=endpoint,
        parameters={"angles": (0.0, math.pi, 0.0)},
        shots=8,
        seed=7,
    ).result.counts == {(1,): 8}


@pytest.mark.parametrize("backend", ["reference", "qiskit"], ids=["reference", "aer"])
@pytest.mark.parametrize("entry", ["nested", "inline"])
@pytest.mark.parametrize("index", ["len(row) - 1", "-1"], ids=["length", "negative"])
def test_param_vector_last_angle_in_callee_and_inline(backend, entry, index):
    """Both forms use the last runtime angle for each specialized vector width."""
    endpoint = qm.Reference() if backend == "reference" else qm.qiskit.Aer()
    module = qm.build_source(LAST_ANGLE_SOURCE.format(angle=f"row[{index}]"))
    for n in (2, 3):
        artifact = qm.compile(module.specialize(entry, n=n), to=endpoint.target)
        assert artifact.format == endpoint.target.name
        if backend == "qiskit":
            assert qm.qiskit.load_circuit(artifact).num_parameters == 1
        for prefix, last, expected in (
            (0.0, math.pi, (1, 0)),
            (math.pi, 0.0, (0, 0)),
        ):
            row = (prefix,) * (n - 1) + (last,)
            counts = qm.run(
                artifact, on=endpoint, parameters={"row": row}, shots=8, seed=7
            ).result.counts
            assert counts == {expected: 8}, row


def test_reusable_angle_keeps_its_launch_recipe_after_emission_is_refused() -> None:
    """Compilation can defer a rejected static circuit and still execute the recipe."""
    from unittest.mock import patch

    from qamomile.qiskit import Aer, Qiskit

    program = qm.build_source(
        LAST_ANGLE_SOURCE.format(angle="row[len(row) - 1]")
    ).specialize("nested", n=2)
    endpoint = Aer()
    with patch.object(
        Qiskit, "lower_circuit", side_effect=qm.CapabilityError("emission declined")
    ) as emit:
        artifact = qm.compile(program, to=endpoint.target)
    emit.assert_called_once()
    assert artifact.format == "qamomile.launch-materializer.v1"
    for row, expected in (((0.0, math.pi), (1, 0)), ((math.pi, 0.0), (0, 0))):
        result = qm.execute(
            artifact, on=endpoint, parameters={"row": row}, shots=4, seed=2
        )
        assert result.counts == {expected: 4}


def test_nested_modulo_angle_is_bound_before_aer_emission():
    """The nested gate can use modulo once its runtime angle is known."""
    source = LAST_ANGLE_SOURCE.format(angle="row[len(row) - 1] % 10.0")
    program = qm.build_source(source).specialize("nested", n=2)
    endpoint = qm.qiskit.Aer()
    artifact = qm.compile(program, to=endpoint.target)
    assert artifact.format == "qamomile.launch-materializer.v1"
    for row, expected in (((0.0, 10.0 + math.pi), (1, 0)), ((math.pi, 10.0), (0, 0))):
        counts = qm.run(
            artifact, on=endpoint, parameters={"row": row}, shots=8, seed=7
        ).result.counts
        assert counts == {expected: 8}


def test_vector_length_condition_selects_the_specialized_branch():
    """A branch decided by the vector width needs no runtime selector."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(row: qm.Param[qm.Vector[qm.Real, n]], n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.ry(qm.qubits(1), row[0])
    if len(row) == 2:
        state = qm.x(state)
    return qm.measure(state)
"""
    module = qm.build_source(source)
    endpoint = qm.qiskit.Aer()
    for n, expected in ((2, (1,)), (3, (0,))):
        artifact = qm.compile(module.specialize("sample", n=n), to=endpoint.target)
        assert artifact.format == endpoint.target.name
        result = qm.run(
            artifact,
            on=endpoint,
            parameters={"row": (0.0,) * n},
            shots=8,
            seed=7,
        ).result
        assert result.counts == {expected: 8}


def test_vector_last_angle_preserves_runtime_branch():
    """A genuine runtime predicate still selects either circuit with indexed angles."""
    source = """
import qamomile as qm
@qm.qkernel
def rotate(row: qm.Param[qm.Vector[qm.Real, 2]], flip: qm.Param[qm.Bool]) -> qm.Q[qm.Bits[2]]:
    state = qm.ry(qm.qubits(2), row[len(row) - 1], target=0)
    if flip:
        state = qm.x(state, target=1)
    return state
@qm.qkernel
def sample(row: qm.Param[qm.Vector[qm.Real, 2]], flip: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(rotate(row, flip))
"""
    endpoint = qm.qiskit.Aer()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    assert artifact.format == "qamomile.launch-materializer.v1"
    for flip in (False, True):
        result = qm.run(
            artifact,
            on=endpoint,
            parameters={"row": (0.0, math.pi), "flip": flip},
            shots=8,
            seed=7,
        ).result
        assert result.counts == {(1, int(flip)): 8}


@pytest.mark.parametrize(
    "legacy_recipe", [False, True], ids=["direct", "legacy-recipe"]
)
def test_restored_last_angle_program_executes_terminal_measurement(legacy_recipe):
    """Native circuits and older launch recipes retain their terminal Boolean result."""
    source = (
        LAST_ANGLE_SOURCE.format(angle="row[len(row) - 1]")
        + """
@qm.qkernel
def sample(n: qm.Const[qm.Int], row: qm.Param[qm.Vector[qm.Real, n]]) -> qm.Shot[qm.Bool]:
    bit, rest = qm.measure_z(rotate(n, row), 0)
    qm.discard(rest)
    return bool(bit)
"""
    )
    endpoint = qm.qiskit.Aer()
    program = qm.build_source(source).specialize("sample", n=2)
    if legacy_recipe:
        from qamomile.compilation import _compile_launch_materializer
        from qamomile.effects import Requirement

        # Older recipes omitted the mid-circuit requirement for a terminal measurement.
        compiled = _compile_launch_materializer(
            program,
            to=endpoint.target,
            required=program.entry.requirements - {Requirement.MID_CIRCUIT_MEASUREMENT},
            planning_stage="native-lowering-planning",
            study=qm.resources.logical(program),
        )
    else:
        compiled = qm.compile(program, to=endpoint.target)
        assert compiled.format == endpoint.target.name
    artifact = deserialize(serialize(compiled))
    for row, expected in (((0.0, math.pi), True), ((math.pi, 0.0), False)):
        result = qm.run(
            artifact, on=endpoint, parameters={"row": row}, shots=8, seed=7
        ).result
        assert result.values == {expected: 8}


@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_nested_helper_projects_runtime_record_fields_in_declaration_order(endpoint):
    """Field names and tuple positions select the same launch values inside a helper."""
    source = """
from typing import NamedTuple
import qamomile as qm

class Angles(NamedTuple):
    z: tuple[qm.Real, qm.Real]
    a: tuple[qm.Real, qm.Real]

@qm.qkernel
def prepare(angles: qm.Param[Angles]) -> qm.Q[qm.Bits[2]]:
    state = qm.ry(qm.qubits(2), angles.z[1], target=0)
    return qm.ry(state, angles.a[1], target=1)

@qm.qkernel
def sample(angles: qm.Param[Angles]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(prepare(angles))
"""
    program = qm.build_source(source).specialize("sample")
    angles = {"a": (0.0, 0.0), "z": (0.0, math.pi)}
    result = qm.run(
        program, on=endpoint, parameters={"angles": angles}, shots=8, seed=7
    )
    assert result.result.counts == {(1, 0): 8}


@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_nested_helper_updates_and_slices_runtime_matrix_row(endpoint):
    """Each update and slice uses the selected row's specialized width."""
    source = """
import qamomile as qm

@qm.qkernel
def prepare(n: qm.Const[qm.Int], angles: qm.Param[qm.Matrix[qm.Real, 1, n]]) -> qm.Q[qm.Bits[2]]:
    row = angles[0]
    row[0] = row[-1]
    window = row[1:]
    state = qm.ry(qm.qubits(2), row[0], target=0)
    for i in range(n - 1):
        state = qm.ry(state, window[i], target=1)
    return state

@qm.qkernel
def sample(n: qm.Const[qm.Int], angles: qm.Param[qm.Matrix[qm.Real, 1, n]]) -> qm.Shot[qm.Bits[2]]:
    return qm.measure(prepare(n, angles))
"""
    program = qm.build_source(source).specialize("sample", n=3)
    parameters = {"angles": ((0.0, math.pi, math.pi),)}
    result = qm.run(program, on=endpoint, parameters=parameters, shots=8, seed=7)
    assert result.result.counts == {(1, 0): 8}


def test_reference_classical_array_compatibility():
    """The reference endpoint executes both shared aggregate regressions."""
    exercise_classical_array_backend(qm.Reference())


@pytest.mark.parametrize("endpoint", [qm.Reference(), qm.qiskit.Aer()])
def test_update_index_can_share_its_replacement_value(endpoint):
    """Resource propagation keeps the index when the replacement uses the same SSA value."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(index: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    values = [0, 0]
    values[index] = index
    state = qm.rx(qm.qubits(1), float(values[1]) * 3.141592653589793)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("sample", index=1)
    assert qm.run(program, on=endpoint, shots=8, seed=7).result.counts == {(1,): 8}


def exercise_classical_array_backend(endpoint, *, seed=7):
    """Run the two shared source regressions on a capable SDK endpoint."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(angles: qm.Const[qm.Vector[qm.Real, 2]], start: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    window = angles[start:]
    state = qm.rx(qm.qubits(2), window[0], target=0)
    bits = qm.measure(state)
    bits[-1] = bits[0]
    return bits
"""
    program = qm.build_source(source).specialize(
        "sample", angles=(0.0, math.pi), start=1
    )
    program = qm.Program.from_bytes(program.to_bytes())
    result = qm.run(program, on=endpoint, shots=8, seed=seed).result
    assert result.counts == {(1, 1): 8}

    source = """
import qamomile as qm
@qm.qkernel
def sample(start: qm.Const[qm.Int]) -> qm.Shot[qm.Bool]:
    bits = qm.measure(qm.x(qm.qubits(2), target=1))
    values = [bool(bits[0]), bool(bits[1])]
    window = values[start:]
    return window[0]
"""
    program = qm.build_source(source).specialize("sample", start=-1)
    result = qm.run(
        qm.Program.from_bytes(program.to_bytes()), on=endpoint, shots=8, seed=seed
    ).result
    assert result.values == {True: 8}


def test_native_rejects_forged_slice_bound_provenance():
    """Equal bound types cannot conceal a substituted Const operand in persisted QCore."""
    import hashlib

    from qamomile._native import compiler as native
    from qamomile.proto import qcore_pb2 as wire

    source = """
import qamomile as qm
@qm.qkernel
def sample(values: qm.Const[qm.Vector[qm.Int, 5]], start: qm.Const[qm.Int], other: qm.Const[qm.Int]) -> qm.Int:
    window = values[start:]
    return window[0] + other
"""
    module = qm.build_source(source)
    package = wire.QCorePackageProto.FromString(module.to_bytes())
    function = package.functions[0]
    operation = next(
        op for op in function.body.operations if op.opcode == "classical.slice"
    )
    operation.operands[1] = next(
        p.value.id for p in function.parameters if p.name == "other"
    )
    package.package_digest = b""
    package.package_digest = hashlib.sha256(
        package.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(ValueError, match="slice bound provenance"):
        native.inspect_module(package.SerializeToString(deterministic=True))


def test_const_slice_rejects_excessive_selected_items_before_execution():
    """Slice output stays within the existing selected-item budget."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(values: qm.Const[qm.Vector[qm.Int, 1025]], start: qm.Const[qm.Int]) -> qm.Int:
    window = values[start:]
    return window[0]
"""
    module = qm.build_source(source)
    with pytest.raises(qm.SpecializationError, match="slice exceeds.*selected items"):
        module.specialize("sample", values=(0,) * 1025, start=0)


def test_const_slice_keeps_bound_stage_and_ordinary_parameter_names():
    """Const boundaries propagate their stage even when the array is a literal."""
    source = """
import qamomile as qm
@qm.qkernel
def first(values: qm.Const[qm.Vector[qm.Int, n]]) -> qm.Int:
    return values[0]
@qm.qkernel
def sample(none: qm.Const[qm.Int]) -> qm.Int:
    window = [1, 2, 3][none:]
    return first(window)
"""
    from qamomile.proto import qcore_pb2 as wire

    module = qm.build_source(source)
    package = wire.QCorePackageProto.FromString(module.to_bytes())
    function = next(f for f in package.functions if f.symbol.endswith(".sample"))
    sliced = next(
        op for op in function.body.operations if op.opcode == "classical.slice"
    )
    assert sliced.results[0].type.name == "Const"
    assert qm.evaluate(module.specialize("sample", none=1)) == 2


def test_array_update_rejects_excessive_materialization():
    """The native evaluator rejects the copy before allocating an oversized result."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(values: qm.Param[qm.Vector[qm.Int, 1025]]) -> qm.Param[qm.Vector[qm.Int, 1025]]:
    values[0] = 1
    return values
"""
    program = qm.build_source(source).specialize("sample")
    with pytest.raises(qm.ExecutionError, match="item budget"):
        qm.evaluate(program, values=(0,) * 1025)
