"""Const-bound slices and immutable updates preserve typed classical arrays."""

import math

import pytest

import qamomile as qm

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
