"""Exact Boolean predicate-to-phase lifting tests."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.effects import CallableKind

from .conftest import BOOLEAN_LIFT_SOURCE, exercise_boolean_lift_backend


def test_reference_executes_boolean_lifting() -> None:
    """Reference executes a helper-backed two-bit predicate exactly."""

    exercise_boolean_lift_backend(endpoint=qm.Reference(), seed=2)


def test_boolean_lifts_persist_and_have_strict_resource_cost() -> None:
    """QCore retains one predicate edge for every exact Boolean lift form."""

    module = qm.build_source(
        BOOLEAN_LIFT_SOURCE,
        module_name="phase_persistence",
    )
    program = module.specialize("phase_sample")
    predicate = module.specialize("marked")
    mapping = module.specialize("marked_pair")
    permutation = module.specialize("cyclic")
    embedding = module.function("coherent_embedding")
    assert qm.evaluate(predicate, bits=(1, 1)) is True
    assert qm.evaluate(predicate, bits=(1, 0)) is False
    assert qm.evaluate(mapping, bits=(1, 1)) == (True, False)
    assert qm.evaluate(permutation, bits=(1, 1)) == (False, True)
    assert embedding.kind is CallableKind.ISOMETRY
    restored = qm.Program.from_bytes(program.to_bytes())
    operation = next(
        operation
        for function in restored.functions
        for operation in function.body.operations
        if operation.opcode == "qcore.phase_by"
    )
    assert operation.attribute("predicate") == "phase_persistence.marked"
    xor_operation = next(
        operation
        for function in restored.functions
        for operation in function.body.operations
        if operation.opcode == "qcore.xor_by"
    )
    assert xor_operation.attribute("predicate") == "phase_persistence.marked_pair"
    permutation_operation = next(
        operation
        for function in restored.functions
        for operation in function.body.operations
        if operation.opcode == "qcore.permute_by"
    )
    assert permutation_operation.attribute("predicate") == "phase_persistence.cyclic"
    assert {function.symbol for function in restored.functions} >= {
        "phase_persistence.first",
        "phase_persistence.marked",
        "phase_persistence.marked_pair",
        "phase_persistence.cyclic",
    }
    lift_plan = next(
        row
        for row in native.plan_boolean_lifts(program.to_bytes())
        if row[0] == "phase_persistence.marked_pair"
    )
    assert lift_plan == (
        "phase_persistence.marked_pair",
        2,
        [[[0, 1]], [[0], [1]]],
        None,
        None,
    )
    permutation_plan = next(
        row
        for row in native.plan_boolean_lifts(program.to_bytes())
        if row[0] == "phase_persistence.cyclic"
    )
    assert permutation_plan == (
        "phase_persistence.cyclic",
        2,
        [[[], [1]], [[0]]],
        None,
        [(0, 1), (0, 3), (0, 2)],
    )
    study = qm.resources.logical(module.specialize("phase_oracle"), strict=True)
    assert study.metric(
        qm.resources.logical_operation_count("qcore.phase_by")
    ).numeric_enclosure() == (1, 1)
    coherent_study = qm.resources.logical(
        module.specialize("coherent_search"), strict=True
    )
    assert coherent_study.metric(
        qm.resources.logical_operation_count("qcore.xor_by")
    ).numeric_enclosure() == (1, 1)
    assert coherent_study.metric(
        qm.resources.logical_operation_count("qcore.permute_by")
    ).numeric_enclosure() == (1, 1)


def test_boolean_xor_embedding_handles_a_dirty_target_exactly() -> None:
    """The exact XOR map does not smuggle in a clean-target assumption."""

    source = """
import qamomile as qm

@qm.qkernel
def marked(bits: qm.Bits[2]) -> qm.Bool:
    return bool(bits[0]) and bool(bits[1])

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    source = qm.x(qm.x(qm.qubits(2), target=0), target=1)
    target = qm.x(qm.qubits(1), target=0)
    source, target = qm.xor_by(marked, source, target)
    return qm.measure(qm.concat(source, target))
"""
    result = qm.run(
        qm.build_source(source).specialize("sample"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert isinstance(result, qm.SampleResult)
    assert result.counts == {(1, 1, 0): 4}


def test_boolean_xor_embedding_maps_multiple_outputs_in_source_order() -> None:
    """A fixed Boolean product maps to the matching dirty target lanes."""

    source = """
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (bool(bits[0]) and bool(bits[1]), bool(bits[0]) ^ bool(bits[1]))

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[4]]:
    source = qm.x(qm.qubits(2), target=0)
    target = qm.x(qm.qubits(2), target=0)
    source, target = qm.xor_by(mapping, source, target)
    return qm.measure(qm.concat(source, target))
"""
    result = qm.run(
        qm.build_source(source).specialize("sample"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert isinstance(result, qm.SampleResult)
    assert result.counts == {(1, 0, 1, 1): 4}


def test_boolean_xor_embedding_selects_a_product_result_exactly() -> None:
    """Structured classical control is normalized component by component."""

    source = """
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (bool(bits[1]), False) if bool(bits[0]) else (False, True)

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[4]]:
    source = qm.x(qm.x(qm.qubits(2), target=0), target=1)
    source, target = qm.xor_by(mapping, source, qm.qubits(2))
    return qm.measure(qm.concat(source, target))
"""
    result = qm.run(
        qm.build_source(source).specialize("sample"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert result.counts == {(1, 1, 1, 0): 4}


def test_boolean_xor_embedding_is_controllable_and_self_adjoint() -> None:
    """One source body supplies direct, controlled, and inverse semantics."""

    source = """
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (bool(bits[0]) and bool(bits[1]), bool(bits[0]) ^ bool(bits[1]))

@qm.qkernel
def oracle(state: qm.Q[qm.Bits[4]]) -> qm.Q[qm.Bits[4]]:
    source, target = qm.split_at(state, 2)
    source, target = qm.xor_by(mapping, source, target)
    return qm.concat(source, target)

@qm.qkernel
def sample_controlled() -> qm.Shot[qm.Bits[5]]:
    state = qm.x(qm.x(qm.x(qm.qubits(5), target=0), target=1), target=2)
    control, target = qm.split_at(state, 1)
    control, target = qm.control(oracle, control, target)
    return qm.measure(qm.concat(control, target))

@qm.qkernel
def sample_adjoint() -> qm.Shot[qm.Bits[4]]:
    state = qm.x(qm.x(qm.qubits(4), target=0), target=1)
    state = oracle(state)
    state = qm.adjoint(oracle, state)
    return qm.measure(state)
"""
    module = qm.build_source(source)
    controlled = qm.run(
        module.specialize("sample_controlled"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    adjoint = qm.run(
        module.specialize("sample_adjoint"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert controlled.counts == {(1, 1, 1, 1, 0): 4}
    assert adjoint.counts == {(1, 1, 0, 0): 4}


def test_boolean_permutation_executes_a_nonlinear_triangular_map() -> None:
    """Acyclic target dependencies permit exact nonlinear in-place synthesis."""

    source = """
import qamomile as qm

@qm.qkernel
def triangular(bits: qm.Bits[3]) -> tuple[qm.Bool, qm.Bool, qm.Bool]:
    return (
        bool(bits[0]) ^ (bool(bits[1]) and bool(bits[2])),
        bool(bits[1]) ^ bool(bits[2]),
        bool(bits[2]),
    )

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[3]]:
    state = qm.x(qm.x(qm.x(qm.qubits(3), target=0), target=1), target=2)
    return qm.measure(qm.permute_by(triangular, state))
"""
    module = qm.build_source(source)
    assert qm.evaluate(module.specialize("triangular"), bits=(1, 1, 1)) == (
        False,
        False,
        True,
    )
    result = qm.run(
        module.specialize("sample"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert result.counts == {(0, 0, 1): 4}


def test_boolean_permutation_accepts_a_single_bit_bool_result() -> None:
    """The one-bit boundary uses Bool without a one-element product wrapper."""

    source = """
import qamomile as qm

@qm.qkernel
def flip(bits: qm.Bits[1]) -> qm.Bool:
    return not bool(bits[0])

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.permute_by(flip, qm.qubits(1)))
"""
    result = qm.run(
        qm.build_source(source).specialize("sample"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert result.counts == {(1,): 4}


def test_general_boolean_permutation_adjoint_and_control_replay_the_proof() -> None:
    """General inverse order and outer control preserve one source meaning."""

    source = """
import qamomile as qm

@qm.qkernel
def cyclic(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (not bool(bits[1]), bool(bits[0]))

@qm.qkernel
def oracle(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    return qm.permute_by(cyclic, state)

@qm.qkernel
def controlled_sample() -> qm.Shot[qm.Bits[3]]:
    state = qm.x(qm.x(qm.x(qm.qubits(3), target=0), target=1), target=2)
    control, target = qm.split_at(state, 1)
    control, target = qm.control(oracle, control, target)
    return qm.measure(qm.concat(control, target))

@qm.qkernel
def inverse_sample() -> qm.Shot[qm.Bits[2]]:
    state = qm.x(qm.x(qm.qubits(2), target=0), target=1)
    state = oracle(state)
    return qm.measure(qm.adjoint(oracle, state))
"""
    module = qm.build_source(source)
    controlled = qm.run(
        module.specialize("controlled_sample"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    inverse = qm.run(
        module.specialize("inverse_sample"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert controlled.counts == {(1, 0, 1): 4}
    assert inverse.counts == {(1, 1): 4}


def test_constant_phase_is_preserved_when_lift_is_controlled() -> None:
    """A constant predicate becomes an observable phase under outer control."""

    source = """
import qamomile as qm

@qm.qkernel
def always(_bits: qm.Bits[1]) -> qm.Bool:
    return True

@qm.qkernel
def phased(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.phase_by(always, state)

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.h(state, target=0)
    control, target = qm.split_at(state, 1)
    control, target = qm.control(phased, control, target)
    state = qm.concat(control, target)
    state = qm.h(state, target=0)
    return qm.measure(state)
    """
    program = qm.build_source(source).specialize("sample")
    result = qm.run(program, on=qm.Reference(), shots=4, seed=2).result
    assert isinstance(result, qm.SampleResult)
    assert result.counts == {(1, 0): 4}


def test_boolean_phase_algebra_supports_negation_and_xor() -> None:
    """Canonical ANF retains complement constants and parity cancellation."""

    source = """
import qamomile as qm

@qm.qkernel
def negated(bits: qm.Bits[2]) -> qm.Bool:
    return not bool(bits[0])

@qm.qkernel
def parity(bits: qm.Bits[2]) -> qm.Bool:
    return bool(bits[0]) ^ bool(bits[1])

@qm.qkernel
def sample_negated() -> qm.Shot[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    state = qm.h(state, target=1)
    state = qm.phase_by(negated, state)
    state = qm.h(state, target=0)
    return qm.measure(qm.h(state, target=1))

@qm.qkernel
def sample_parity() -> qm.Shot[qm.Bits[2]]:
    state = qm.h(qm.qubits(2), target=0)
    state = qm.h(state, target=1)
    state = qm.phase_by(parity, state)
    state = qm.h(state, target=0)
    return qm.measure(qm.h(state, target=1))
"""
    module = qm.build_source(source)
    negated = qm.run(
        module.specialize("sample_negated"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    parity = qm.run(
        module.specialize("sample_parity"),
        on=qm.Reference(),
        shots=4,
        seed=2,
    ).result
    assert isinstance(negated, qm.SampleResult)
    assert isinstance(parity, qm.SampleResult)
    assert negated.counts == {(1, 0): 4}
    assert parity.counts == {(1, 1): 4}


def test_phase_lift_rejects_non_boolean_graphs_before_execution() -> None:
    """The native proof rejects valid classical operations outside its subset."""

    source = """
import qamomile as qm

@qm.qkernel
def structural(bits: qm.Bits[2]) -> qm.Bool:
    return len(bits) == 2

@qm.qkernel
def phase() -> qm.Q[qm.Bits[2]]:
    return qm.phase_by(structural, qm.qubits(2))
"""
    with pytest.raises(qm.VerificationError, match="Q7681:.*classical.len"):
        qm.build_source(source)


def test_phase_lift_rejects_an_external_predicate_at_the_source() -> None:
    """A backend callback cannot become a coherent predicate implicitly."""

    source = """
import qamomile as qm

@qm.qkernel(kind="classical")
def external(bits: qm.Bits[1]) -> qm.Bool: ...

@qm.qkernel
def phase() -> qm.Q[qm.Bits[1]]:
    return qm.phase_by(external, qm.qubits(1))
"""
    with pytest.raises(qm.FrontendError, match="Q5108"):
        qm.build_source(source)


def test_phase_lift_rejects_a_mismatched_state_shape_at_the_source() -> None:
    """Predicate and coherent state widths cannot drift across backends."""

    source = """
import qamomile as qm

@qm.qkernel
def marked(bits: qm.Bits[2]) -> qm.Bool:
    return bool(bits[0])

@qm.qkernel
def phase() -> qm.Q[qm.Bits[1]]:
    return qm.phase_by(marked, qm.qubits(1))
"""
    with pytest.raises(qm.TypeCheckError, match=r"state must be Q\[Bits\[2\]\]"):
        qm.build_source(source)


def test_boolean_xor_rejects_an_external_predicate_at_the_source() -> None:
    """A bodyless declaration cannot authorize a coherent XOR implementation."""

    source = """
import qamomile as qm

@qm.qkernel(kind="classical")
def external(bits: qm.Bits[1]) -> qm.Bool: ...

@qm.qkernel
def oracle() -> qm.Q[qm.Bits[2]]:
    source, target = qm.xor_by(external, qm.qubits(1), qm.qubits(1))
    return qm.concat(source, target)
"""
    with pytest.raises(qm.FrontendError, match="Q5112"):
        qm.build_source(source)


@pytest.mark.parametrize(
    ("source_expression", "target_expression", "message"),
    (
        ("qm.qubits(1)", "qm.qubits(1)", r"source must be Q\[Bits\[2\]\]"),
        ("qm.qubits(2)", "qm.qubits(2)", r"target must be Q\[Bits\[1\]\]"),
    ),
)
def test_boolean_xor_rejects_mismatched_owner_shapes_at_the_source(
    source_expression: str,
    target_expression: str,
    message: str,
) -> None:
    """Source and target widths are checked before native lowering."""

    source = f"""
import qamomile as qm

@qm.qkernel
def predicate(bits: qm.Bits[2]) -> qm.Bool:
    return bool(bits[0])

@qm.qkernel
def oracle() -> tuple[qm.Q[qm.Bits[2]], qm.Q[qm.Bits[1]]]:
    return qm.xor_by(predicate, {source_expression}, {target_expression})
"""
    with pytest.raises(qm.TypeCheckError, match=message):
        qm.build_source(source)


def test_boolean_xor_requires_target_width_to_match_all_outputs() -> None:
    """A multi-output function cannot be paired with a shorter target word."""

    source = """
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (bool(bits[0]), bool(bits[1]))

@qm.qkernel
def oracle() -> tuple[qm.Q[qm.Bits[2]], qm.Q[qm.Bits[1]]]:
    return qm.xor_by(mapping, qm.qubits(2), qm.qubits(1))
"""
    with pytest.raises(qm.TypeCheckError, match=r"target must be Q\[Bits\[2\]\]"):
        qm.build_source(source)


def test_boolean_xor_rejects_non_boolean_product_outputs() -> None:
    """Every multi-output coordinate must have the exact Bool type."""

    source = """
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[1]) -> tuple[qm.Bool, qm.Int]:
    return (bool(bits[0]), 0)

@qm.qkernel
def oracle() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[2]]]:
    return qm.xor_by(mapping, qm.qubits(1), qm.qubits(2))
"""
    with pytest.raises(qm.FrontendError, match="Q5114"):
        qm.build_source(source)


def test_boolean_xor_rejects_an_over_budget_output_without_simulation() -> None:
    """The native proof rejects an excessive fixed output before emission."""

    width = 4_097
    result_type = ", ".join("qm.Bool" for _ in range(width))
    result = ", ".join("False" for _ in range(width))
    source = f"""
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[1]) -> tuple[{result_type}]:
    return ({result})

@qm.qkernel
def oracle() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[{width}]]]:
    return qm.xor_by(mapping, qm.qubits(1), qm.qubits({width}))
"""
    with pytest.raises(qm.VerificationError, match="Q7688"):
        qm.build_source(source)


def test_boolean_xor_rejects_an_unproved_classical_graph() -> None:
    """A valid host computation is not implicitly treated as reversible."""

    source = """
import qamomile as qm

@qm.qkernel
def structural(bits: qm.Bits[2]) -> qm.Bool:
    return len(bits) == 2

@qm.qkernel
def oracle() -> qm.Q[qm.Bits[3]]:
    source, target = qm.xor_by(structural, qm.qubits(2), qm.qubits(1))
    return qm.concat(source, target)
"""
    with pytest.raises(qm.VerificationError, match="Q7681:.*classical.len"):
        qm.build_source(source)


def test_boolean_xor_rejects_aliased_owned_inputs() -> None:
    """Packing the same owner twice cannot forge a reversible product state."""

    source = """
import qamomile as qm

@qm.qkernel
def identity(bits: qm.Bits[1]) -> qm.Bool:
    return bool(bits[0])

@qm.qkernel
def oracle() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    state = qm.qubits(1)
    return qm.xor_by(identity, state, state)
"""
    with pytest.raises(qm.VerificationError, match="Q2001:.*observed 2 uses"):
        qm.build_source(source)


@pytest.mark.parametrize(
    "result_expression",
    (
        "(bool(bits[0]), bool(bits[0]))",
        "(bool(bits[0]) ^ (bool(bits[0]) and bool(bits[1])), bool(bits[1]))",
    ),
)
def test_boolean_permutation_rejects_non_bijective_maps(
    result_expression: str,
) -> None:
    """Non-surjective and nonlinear self-dependent maps fail closed."""

    source = f"""
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return {result_expression}

@qm.qkernel
def oracle() -> qm.Q[qm.Bits[2]]:
    return qm.permute_by(mapping, qm.qubits(2))
"""
    with pytest.raises(qm.VerificationError, match="Q7690"):
        qm.build_source(source)


def test_general_boolean_permutation_has_an_explicit_ten_bit_proof_boundary() -> None:
    """Exhaustive proof accepts ten bits and rejects eleven without simulation."""

    def swap_source(width: int) -> str:
        results = ["bool(bits[1])", "bool(bits[0])"] + [
            f"bool(bits[{index}])" for index in range(2, width)
        ]
        result_type = ", ".join("qm.Bool" for _ in range(width))
        return f"""
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[{width}]) -> tuple[{result_type}]:
    return ({", ".join(results)})

@qm.qkernel
def oracle(state: qm.Q[qm.Bits[{width}]]) -> qm.Q[qm.Bits[{width}]]:
    return qm.permute_by(mapping, state)
"""

    program = qm.build_source(swap_source(10)).specialize("oracle")
    [plan] = native.plan_boolean_lifts(program.to_bytes())
    assert plan[0].endswith(".mapping")
    assert plan[1] == 10
    assert plan[3] is None
    assert plan[4] is not None
    assert len(plan[4]) == 256

    with pytest.raises(qm.VerificationError, match="Q7691"):
        qm.build_source(swap_source(11))


def test_boolean_permutation_rejects_an_external_function_at_the_source() -> None:
    """A bodyless declaration cannot authorize an in-place permutation."""

    source = """
import qamomile as qm

@qm.qkernel(kind="classical")
def external(bits: qm.Bits[1]) -> qm.Bool: ...

@qm.qkernel
def oracle() -> qm.Q[qm.Bits[1]]:
    return qm.permute_by(external, qm.qubits(1))
"""
    with pytest.raises(qm.FrontendError, match="Q5116"):
        qm.build_source(source)


def test_boolean_permutation_requires_one_output_per_input_bit() -> None:
    """The source diagnostic rejects a width-changing in-place function."""

    source = """
import qamomile as qm

@qm.qkernel
def mapping(bits: qm.Bits[2]) -> qm.Bool:
    return bool(bits[0])

@qm.qkernel
def oracle() -> qm.Q[qm.Bits[2]]:
    return qm.permute_by(mapping, qm.qubits(2))
"""
    with pytest.raises(qm.FrontendError, match="Q5118"):
        qm.build_source(source)


def test_boolean_permutation_requires_a_matching_owned_state() -> None:
    """The transformed owner width must match the classical function input."""

    source = """
import qamomile as qm

@qm.qkernel
def identity(bits: qm.Bits[2]) -> tuple[qm.Bool, qm.Bool]:
    return (bool(bits[0]), bool(bits[1]))

@qm.qkernel
def oracle() -> qm.Q[qm.Bits[1]]:
    return qm.permute_by(identity, qm.qubits(1))
"""
    with pytest.raises(qm.TypeCheckError, match=r"state must be Q\[Bits\[2\]\]"):
        qm.build_source(source)
