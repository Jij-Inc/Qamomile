"""Tests for ownership, dynamic control, and automatic uncomputation."""

from __future__ import annotations

from dataclasses import replace

import pytest

import qamomile as qm
from qamomile.effects import Requirement
from qamomile.types import as_type

from .conftest import (
    CALLABLE_TRANSFORMS_SOURCE,
    COHERENT_ADJOINT_SOURCE,
    CONTROLLED_PHASE_SOURCE,
    FEEDBACK_SOURCE,
    INDEXED_CONTROL_SOURCE,
    MULTI_INPUT_CLEANUP_SOURCE,
    lower_adjoint_calls,
)


def test_reference_executes_callable_control_select_and_relative_phase() -> None:
    """Callable transforms share one coherent-control semantics in QCore."""

    selected = qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize(
        "callable_transforms"
    )
    phased = qm.build_source(CONTROLLED_PHASE_SOURCE).specialize("controlled_phase")

    assert "qcore.select_call" in selected.inspect()
    assert qm.run(selected, on=qm.Reference(), shots=8, seed=2).result.values == {
        (1, (1,)): 8
    }
    assert qm.run(phased, on=qm.Reference(), shots=8, seed=2).result.values == {
        (1, (0,)): 8
    }


def test_adjoint_of_coherent_control_is_derived_without_a_new_opcode() -> None:
    """The common dagger derives C(U) adjoint as C(U-adjoint)."""

    source = """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def controlled_flip(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    control, target = qm.split_at(state, 1)
    control, target = qm.control(flip, control, target)
    return qm.concat(control, target)

@qm.qkernel
def roundtrip() -> qm.Shot[qm.Bits[2]]:
    state = qm.prepare((1, 0))
    state = controlled_flip(state)
    state = qm.adjoint(controlled_flip, state)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("roundtrip")
    lowered = lower_adjoint_calls(program)

    assert qm.evaluate(program) == (1, 0)
    assert "qcore.adjoint_call" not in lowered.inspect()
    assert "qcore.control_call" in lowered.inspect()
    assert ".__qamomile_adjoint" in lowered.inspect()


def test_adjoint_of_coherent_selection_maps_every_branch_to_its_adjoint() -> None:
    """The common dagger maps multiplex(U_i) to multiplex(U_i-adjoint)."""

    program = qm.build_source(COHERENT_ADJOINT_SOURCE).specialize(
        "selected_adjoint_roundtrip"
    )
    lowered = lower_adjoint_calls(program)

    assert qm.evaluate(program) == (1, 0)
    assert "qcore.adjoint_call" not in lowered.inspect()
    assert "qcore.select_call" in lowered.inspect()
    assert ".__qamomile_adjoint" in lowered.inspect()


def test_aggregate_control_selector_executes_without_qubit_handles() -> None:
    """A loop index selects a coherent component while retaining its aggregate."""

    module = qm.build_source(INDEXED_CONTROL_SOURCE)
    sampled = module.specialize("indexed_control_sample")
    state = module["indexed_control_state"]
    restored = qm.Program.from_bytes(sampled.to_bytes())

    assert restored.inspect().count("qcore.control_call") == 1
    assert qm.run(restored, on=qm.Reference(), shots=8, seed=2).result.counts == {
        (1, 0, 1): 8
    }
    assert (
        qm.expectation(
            state,
            qm.Z(2),
            on=qm.Reference(),
            shots=8,
            seed=2,
        ).value
        == -1.0
    )


def test_dynamic_aggregate_control_selector_is_an_explicit_requirement() -> None:
    """A launch-time selector is typed, negotiated, and interpreted exactly."""

    source = """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def choose(at: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[3]]:
    controls = qm.prepare((1, 0))
    target = qm.qubits(1)
    system = qm.concat(controls, target)
    controls, target = qm.split_at(system, 2)
    controls, target = qm.control(flip, controls, target, at=at)
    return qm.measure(qm.concat(controls, target))
"""
    program = qm.build_source(source).specialize("choose")

    assert Requirement.DYNAMIC_QUBIT_SELECTION in program.entry.requirements
    first = qm.run(
        program,
        on=qm.Reference(),
        parameters={"at": 0},
        shots=4,
        seed=2,
    )
    second = qm.run(
        program,
        on=qm.Reference(),
        parameters={"at": 1},
        shots=4,
        seed=2,
    )

    assert (
        Requirement.DYNAMIC_QUBIT_SELECTION
        in first.artifact.execution_contract.requirements
    )
    assert first.result.counts == {(1, 0, 1): 4}
    assert second.result.counts == {(1, 0, 0): 4}


def test_dependent_control_width_specializes_one_all_active_semantics() -> None:
    """A Const width stays symbolic without constructing a Python-side mask."""

    source = """
from __future__ import annotations

import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def controlled(
    width: qm.Const[qm.Int],
) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[1]]]:
    controls = qm.qubits(width)
    for target in range(width):
        controls = qm.x(controls, target=target)
    target = qm.qubits(1)
    controls, target = qm.control(flip, controls, target)
    return qm.measure(controls), qm.measure(target)
"""
    function = qm.build_source(source)["controlled"]
    symbolic = qm.resources.logical(function)
    control_calls = qm.resources.logical_operation_count("qcore.control_call")

    assert symbolic.metric(control_calls).numeric_enclosure() == (1, 1)
    assert symbolic.metric(qm.resources.LOGICAL_PEAK_QUBITS).evaluate(
        {"width": 3}
    ).numeric_enclosure() == (4, 4)

    program = qm.specialize(function, width=3)
    restored = qm.Program.from_bytes(program.to_bytes())
    artifact = qm.compile(restored, to=qm.Reference())

    assert "control_state=None" in restored.inspect()
    assert restored.to_bytes() == program.to_bytes()
    assert qm.run(artifact, on=qm.Reference(), shots=4, seed=2).result.values == {
        ((1, 1, 1), (1,)): 4
    }


def test_dependent_explicit_control_state_is_checked_after_specialization() -> None:
    """An explicit mask remains valid only for specialized widths that contain it."""

    source = """
from __future__ import annotations

import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def controlled(
    width: qm.Const[qm.Int],
) -> tuple[qm.Q[qm.Bits[width]], qm.Q[qm.Bits[1]]]:
    controls = qm.qubits(width)
    target = qm.qubits(1)
    return qm.control(flip, controls, target, control_state=3)

@qm.qkernel
def selected(
    width: qm.Const[qm.Int],
) -> tuple[qm.Q[qm.Bits[width]], qm.Q[qm.Bits[1]]]:
    controls = qm.qubits(width)
    target = qm.qubits(1)
    return qm.control(flip, controls, target, at=2)
"""
    module = qm.build_source(source)
    function = module["controlled"]

    assert qm.specialize(function, width=2).entry.result_type == as_type(
        tuple[qm.Q[qm.Bits[2]], qm.Q[qm.Bits[1]]]
    )
    with pytest.raises(qm.SpecializationError, match="Q1052"):
        qm.specialize(function, width=1)

    assert qm.specialize(module["selected"], width=3).entry.result_type == as_type(
        tuple[qm.Q[qm.Bits[3]], qm.Q[qm.Bits[1]]]
    )
    with pytest.raises(qm.SpecializationError, match="Q1052"):
        qm.specialize(module["selected"], width=2)


def test_aggregate_control_selector_rejects_a_static_out_of_range_index() -> None:
    """A known invalid component fails at its Python source span."""

    source = """
import qamomile as qm

@qm.qkernel
def flip(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def invalid() -> tuple[qm.Q[qm.Bits[2]], qm.Q[qm.Bits[1]]]:
    controls = qm.qubits(2)
    target = qm.qubits(1)
    return qm.control(flip, controls, target, at=2)
"""
    with pytest.raises(qm.FrontendError, match="outside width 2"):
        qm.build_source(source)


def test_measurement_control_is_structured_and_preserves_result_order() -> None:
    """A Shot condition remains a region op and control bits stay ABI-private."""

    program = qm.build_source(FEEDBACK_SOURCE).specialize("feedback")
    result = qm.run(program, on=qm.Reference(), shots=64, seed=2).result

    assert "qcore.if" in program.inspect()
    assert "stage='Shot'" in program.inspect()
    assert set(result.counts) == {(0, 0), (1, 1)}


def test_dead_owned_state_is_reversed_then_certifiably_released() -> None:
    """Forget obligations become adjoints and a proof-carrying release."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(1)
    state = qm.h(state)
    state = qm.t(state)
    return None
"""
    module = qm.build_source(source)
    program = module.specialize("cleanup")

    assert "qcore.forget" in module.inspect()
    assert "qcore.forget" not in program.inspect()
    assert program.inspect().count("qcore.adjoint") == 2
    assert "quantum.release" in program.inspect()
    assert "proof='modular-adjoint'" in program.inspect()
    assert qm.evaluate(program) is None


def test_multi_input_call_cleanup_uses_the_existing_tensor_adjoint() -> None:
    """Independent owned inputs are restored without an atom or cleanup opcode."""

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )

    assert qm.evaluate(program) == (1,)
    assert program.inspect().count("quantum.release") == 2
    assert "qcore.adjoint_call" in program.inspect()
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()


def test_multi_input_cleanup_rejects_a_forged_tensor_reassociation() -> None:
    """A proof label cannot swap independently restored ownership roots."""

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    entry = program.entry
    unpack = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.unpack"
        and operation.attribute("proof") == "modular-adjoint"
    )
    first, second = unpack.results
    forged_unpack = replace(
        unpack,
        results=(
            replace(first, region=second.region),
            replace(second, region=first.region),
        ),
    )
    forged_entry = replace(
        entry,
        body=replace(
            entry.body,
            operations=tuple(
                forged_unpack if operation is unpack else operation
                for operation in entry.body.operations
            ),
        ),
    )
    forged = replace(
        program,
        functions=tuple(
            forged_entry if function.symbol == entry.symbol else function
            for function in program.functions
        ),
    )

    with pytest.raises(qm.VerificationError, match="Q1113"):
        qm.compile(forged, to=qm.Reference())


def test_explicit_discard_is_not_rewritten_as_unitary_cleanup() -> None:
    """Irreversible discard remains an explicit semantic channel effect."""

    source = """
import qamomile as qm

@qm.qkernel
def discard_state() -> qm.Unit:
    state = qm.qubits(1)
    qm.discard(state)
    return None
"""
    program = qm.build_source(source).specialize("discard_state")

    assert "quantum.discard" in program.inspect()
    assert "qcore.adjoint" not in program.inspect()


def test_cleanup_proof_string_cannot_authorize_a_forged_release() -> None:
    """Restoration verification reconstructs the chain instead of trusting text."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(1)
    state = qm.h(state)
    return None
"""
    program = qm.build_source(source).specialize("cleanup")
    entry = program.entry
    first_adjoint = next(
        operation
        for operation in entry.body.operations
        if operation.opcode == "qcore.adjoint"
    )
    forged_operations = tuple(
        replace(operation, operands=(first_adjoint.operands[0],))
        if operation.opcode == "quantum.release"
        else operation
        for operation in entry.body.operations
        if operation.opcode != "qcore.adjoint"
    )
    forged_entry = replace(
        entry, body=replace(entry.body, operations=forged_operations)
    )
    forged = replace(
        program,
        functions=tuple(
            forged_entry if function.symbol == entry.symbol else function
            for function in program.functions
        ),
    )

    with pytest.raises(qm.VerificationError, match="not the inverse"):
        qm.compile(forged, to=qm.Reference())
