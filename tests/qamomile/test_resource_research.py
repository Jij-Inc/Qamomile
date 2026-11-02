"""Progressive research views use native symbolic bodies, not priced Python tables."""

from dataclasses import replace

import numpy as np
import pytest

import qamomile as qm
from qamomile import resources as qr

CONTROLLED_SOURCE = """
import qamomile as qm
@qm.qkernel
def oracle(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(q)
@qm.qkernel
def step(s: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    c, q = qm.split_at(s, 1)
    c, q = qm.control(oracle, c, q)
    return qm.concat(c, q)
@qm.qkernel
def sample(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2]]:
    s = qm.qubits(2)
    for i in range(n):
        s = step(s)
    return qm.measure(s)
"""


@pytest.fixture
def study():
    """One repeated controlled-H body exposes the original soundness gap."""
    return qr.logical(qm.build_source(CONTROLLED_SOURCE)["sample"])


def test_gate_cost_model_public_names():
    """Expose a cost model and gate basis without keeping obsolete API aliases."""
    model = qr.GateCostModel(basis=qr.GateBasis.CLIFFORD_TOFFOLI_ROTATIONS)
    assert isinstance(model, qr.ResourceModel)
    assert model.basis is qr.GateBasis.CLIFFORD_TOFFOLI_ROTATIONS
    assert "GateCostModel" in qr.__all__ and "GateBasis" in qr.__all__
    assert not hasattr(qr, "PortableCircuit")
    assert not hasattr(qr, "PortableBasis")


def test_frontier_expansion_preserves_context_and_source(study):
    """Opening a row replaces parent occurrences without changing source costs."""
    original = study.to_bytes()
    coarse = study.breakdown()
    middle = coarse.expand("step")
    fine = middle.expand("oracle")
    assert "call.direct.c0.module.step" in str(coarse)
    assert "call.controlled.c1.module.oracle" in str(middle)
    assert "operation.controlled.c1.quantum.h" in str(fine)
    assert "call." not in str(fine)
    assert fine.evaluate(n=10**12).call_count("oracle").numeric_enclosure() == (
        10**12,
        10**12,
    )
    assert coarse.call_count("oracle") == fine.call_count("oracle")
    assert study.to_bytes() == original
    assert study.metric(qr.LOGICAL_GATES).finite_enclosure()[1] is None


def test_portable_controlled_h_and_serial_depth(study):
    """CH is CZ plus two rotations, not one source Clifford."""
    priced = qr.project(study, qr.GateCostModel())
    concrete = priced.evaluate(n=7)
    assert concrete.metric(qr.portable_metric("gates")).numeric_enclosure() == (21, 21)
    assert concrete.metric(
        qr.portable_metric("clifford_gates")
    ).numeric_enclosure() == (7, 7)
    assert concrete.metric(
        qr.portable_metric("rotation_gates")
    ).numeric_enclosure() == (14, 14)
    assert concrete.metric(qr.portable_metric("gate_depth")).numeric_enclosure() == (
        None,
        21,
    )
    assert concrete.require(qr.portable_metric("gate_depth")) is concrete
    with pytest.raises(ValueError, match="lacks exact"):
        concrete.require(qr.portable_metric("gate_depth"), coverage="exact")
    assert priced.metric(qr.LOGICAL_GATES) == study.metric(qr.LOGICAL_GATES)
    restored = qr.ResourceStudy.from_bytes(priced.to_bytes())
    assert restored == priced
    assert restored.breakdown(expand=None).rows == study.breakdown(expand=None).rows


def test_breakdown_selection_is_separate_from_gate_realization(study):
    """Use view expansion only for rows while gate projection opens bodies."""

    opaque = study.breakdown(expand=())
    assert any("module.step" in row.metric.identifier for row in opaque.rows)
    expanded_view = opaque.expand("step", "oracle")
    assert expanded_view.rows
    expanded = qr.project(study, qr.GateCostModel())
    expanded.require(qr.portable_metric("gates"), coverage="exact")
    with pytest.raises(ValueError, match="overwrite"):
        qr.project(expanded, qr.GateCostModel())


def test_selection_and_evaluation_are_not_hidden_specialization(study):
    """Selections fail closed, and display substitutions cannot rerun semantics."""
    with pytest.raises(Exception, match="missing|unknown|no function|not found|Q3001"):
        study.breakdown(expand=("absent",))
    with pytest.raises(ValueError, match="duplicate"):
        study.breakdown(expand=("oracle", "oracle"))
    with pytest.raises(ValueError, match="before evaluate"):
        study.evaluate(n=3).breakdown()
    with pytest.raises(ValueError, match="before evaluate"):
        qr.project(study.evaluate(n=3), qr.GateCostModel())
    with pytest.raises(ValueError, match="before evaluate"):
        study.breakdown().evaluate(n=3).expand()
    closed = qr.logical(qm.build_source(CONTROLLED_SOURCE).specialize("sample", n=2))
    assert closed.breakdown(expand=("step", "oracle")).call_count(
        "oracle"
    ).numeric_enclosure() == (2, 2)


@pytest.mark.parametrize(
    "opcode,controls",
    [
        ("h", 1),
        ("x", 1),
        ("x", 2),
        ("y", 1),
        ("z", 1),
        ("ry", 1),
        ("rz", 1),
        ("p", 1),
        ("cp", 0),
        ("ccx", 0),
        ("cx", 1),
        ("swap", 0),
        ("rzz", 0),
    ],
)
@pytest.mark.parametrize("basis", list(qr.GateBasis))
def test_recipe_materialization_preserves_the_full_unitary(opcode, controls, basis):
    """Independent SDK matrices detect even a lost controlled global phase."""
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    model = qr.GateCostModel(basis)
    recipe = model.recipe(f"quantum.{opcode}", controls=controls)
    assert recipe is not None
    width = max(max(lanes) for _, lanes, _, _, _ in recipe) + 1
    theta = 0.731
    generated = qiskit.QuantumCircuit(width)
    for name, lanes, kind, numerator, denominator in recipe:
        gate = getattr(generated, name.removeprefix("quantum."))
        if kind == "none":
            gate(*lanes)
        else:
            gate(
                (theta if kind == "input" else np.pi) * numerator / denominator, *lanes
            )
    original = qiskit.QuantumCircuit(width - controls)
    operation = getattr(original, opcode)
    if opcode in {"ry", "rz", "p", "cp", "rzz"}:
        operation(theta, *range(width - controls))
    else:
        operation(*range(width - controls))
    expected = original.to_gate()
    if controls:
        expected = expected.control(controls)
    np.testing.assert_allclose(
        Operator(generated).data, Operator(expected).data, atol=1e-14
    )


@pytest.mark.parametrize("controls,expected", [(1, 1), (2, 1), (3, 3)])
def test_control_arity_never_saturates_to_single_control(controls, expected):
    """A full control register selects the clean-MCX profile without saturation."""
    source = CONTROLLED_SOURCE.replace("qm.h(q)", "qm.x(q)")
    source = source.replace("qm.Bits[2]", f"qm.Bits[{controls + 1}]")
    source = source.replace("split_at(s, 1)", f"split_at(s, {controls})")
    source = source.replace("qm.qubits(2)", f"qm.qubits({controls + 1})")
    study = qr.logical(qm.build_source(source)["sample"])
    priced = qr.project(study, qr.GateCostModel()).evaluate(n=1)
    assert priced.metric(qr.portable_metric("gates")).numeric_enclosure()[1] == expected


def test_negative_control_charges_conjugating_x_gates():
    """Control on zero contributes two X operations around the controlled body."""
    source = CONTROLLED_SOURCE.replace("qm.h(q)", "qm.x(q)").replace(
        "qm.control(oracle, c, q)", "qm.control(oracle, c, q, control_state=0)"
    )
    priced = qr.project(
        qr.logical(qm.build_source(source)["sample"]), qr.GateCostModel()
    ).evaluate(n=1)
    assert priced.metric(qr.portable_metric("gates")).numeric_enclosure() == (3, 3)


def test_comparison_keeps_coordinate_and_missing_costs(study):
    """A comparison never treats absent values as zero or asserts task equivalence."""
    fine = qr.project(study, qr.GateCostModel())
    text = qr.compare(
        {"source": study, "portable": fine}, metrics=(qr.portable_metric("gates"),)
    )
    assert "not reported" in text and "circuit; instructions" in text
    assert "task/accuracy equivalence is not established" in text
    with pytest.raises(ValueError, match="definitions disagree"):
        qr.compare(
            {"portable": fine},
            metrics=(replace(qr.portable_metric("gates"), version="999"),),
        )


def test_missing_native_observer_fails_closed(study, monkeypatch):
    """There is no Python counting fallback."""
    import qamomile.resources.analysis as analysis

    monkeypatch.setattr(analysis.native, "inspect_symbolic_resource_observation", None)
    with pytest.raises(qm.CapabilityError, match="Q4214"):
        study.breakdown()


@pytest.fixture(scope="module")
def research_studies():
    """Analyze the five research examples once, before substituting dimensions."""
    from tests.qamomile.resource_programs import studies

    return studies()


def test_five_research_algorithms_match_independent_finite_identities(research_studies):
    """Independent formulas test accounting, not a fit to a few output samples."""
    from tests.qamomile.resource_programs import amplitude_step, evolution, mark_good

    model = qr.GateCostModel()
    priced = {
        name: qr.project(study, model) for name, study in research_studies.items()
    }
    gates = qr.portable_metric("gates")
    for n in (1, 2, 5, 16):
        qft_gates = n + 5 * n * (n - 1) // 2 + 3 * (n // 2)
        assert priced["QFT"].evaluate(width=n).metric(gates).numeric_enclosure() == (
            qft_gates,
            qft_gates,
        )
    qpe = research_studies["QPE"].breakdown(expand=None)
    ae = research_studies["amplitude_estimation"].breakdown(expand=None)
    for p in (1, 3, 7, 64):
        queries = 2**p - 1
        overhead = 1 + 2 * p + 5 * p * (p - 1) // 2 + 6 * (p // 2)
        assert qpe.evaluate(precision=p).call_count(evolution).numeric_enclosure() == (
            queries,
            queries,
        )
        assert ae.evaluate(precision=p).call_count(
            amplitude_step
        ).numeric_enclosure() == (queries, queries)
        for name, price in (("QPE", 5), ("amplitude_estimation", 10)):
            expected = overhead + price * queries
            assert priced[name].evaluate(precision=p).metric(
                gates
            ).numeric_enclosure() == (expected, expected)
    for k in (0, 1, 2, 10**12):
        expected = 3 + 18 * k
        assert priced["amplification"].evaluate(iterations=k).metric(
            gates
        ).numeric_enclosure() == (expected, expected)
        calls = (
            research_studies["amplification"]
            .breakdown(expand=None)
            .evaluate(iterations=k)
            .call_count(mark_good)
        )
        assert calls.numeric_enclosure() == (2 * k, 2 * k)
    for r in (1, 2, 10**12):
        assert priced["product_formula"].evaluate(steps=r).metric(
            gates
        ).numeric_enclosure() == (3 * r, 3 * r)


def test_toffoli_choice_changes_only_the_model(research_studies):
    """Changing the basis revises T demand without mutating source query counts."""
    source = research_studies["amplification"]
    priced = qr.project(source, qr.GateCostModel(qr.GateBasis.CLIFFORD_T_ROTATIONS))
    for k in (0, 2, 10**12):
        values = priced.evaluate(iterations=k)
        assert values.metric(qr.portable_metric("gates")).numeric_enclosure() == (
            3 + 46 * k,
            3 + 46 * k,
        )
        assert values.metric(qr.portable_metric("t_gates")).numeric_enclosure() == (
            14 * k,
            14 * k,
        )
    assert qr.ResourceStudy.from_bytes(priced.to_bytes()) == priced


def test_fused_product_formula_is_a_new_program_with_lower_cost(research_studies):
    """Replacing source is distinct from changing a model or a display frontier."""
    from tests.qamomile.resource_programs import fused_product_formula

    original = research_studies["product_formula"]
    fused = qr.logical(fused_product_formula)
    assert original.semantic_digest != fused.semantic_digest
    priced = qr.project(fused, qr.GateCostModel())
    for r in (1, 2, 10**12):
        assert priced.evaluate(steps=r).metric(
            qr.portable_metric("gates")
        ).numeric_enclosure() == (2 * r + 1, 2 * r + 1)


def test_research_programs_compute_the_claimed_quantities():
    """Small exact statevectors check answers and fused-body equivalence."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector
    from qiskit.transpiler.passes import UnrollForLoops

    from qamomile.qiskit import Qiskit
    from qamomile.qiskit._integration import load_circuit
    from tests.qamomile.resource_programs import (
        amplified_search,
        amplitude_sample,
        fused_product_formula,
        phase_sample,
        product_formula,
    )

    # The numeric phase is exact for this eigenstate, so four shots suffice.
    phase = qm.run(
        qm.specialize(phase_sample, precision=3), on=qm.Reference(), shots=4, seed=2
    )
    assert phase.result.values == {0.25: 4}
    estimated = qm.run(
        qm.specialize(amplitude_sample, precision=3), on=qm.Reference(), shots=4, seed=2
    )
    for observed in estimated.result.values:
        assert np.sin(np.pi * observed) ** 2 == pytest.approx(np.sin(np.pi / 8) ** 2)

    # Use the SDK's exact statevector only for this bounded three-qubit witness.
    def statevector(kernel, *, time=None, **parameters):
        artifact = qm.compile(qm.specialize(kernel, **parameters), to=Qiskit())
        circuit = load_circuit(artifact).remove_final_measurements(inplace=False)
        if time is not None:
            circuit = circuit.assign_parameters([time])
        # Statevector requires a concrete circuit; production compilation keeps
        # these small fixed loops structured until an execution target needs them.
        return Statevector.from_instruction(UnrollForLoops()(circuit)).data

    for k in (0, 1, 2):
        result = statevector(amplified_search, iterations=k)
        assert abs(result[7]) ** 2 == pytest.approx(
            np.sin((2 * k + 1) * np.arcsin(1 / np.sqrt(8))) ** 2
        )
    # Bind the sole runtime angle in the independently inspected SDK circuit.
    for r in (1, 2, 5):
        left = statevector(product_formula, steps=r, time=0.731)
        right = statevector(fused_product_formula, steps=r, time=0.731)
        np.testing.assert_allclose(left, right, atol=1e-14)


@pytest.mark.parametrize(
    "backend",
    [
        "reference",
        "qiskit",
        "cirq",
        "quri_parts",
        pytest.param(
            "braket",
            marks=pytest.mark.xfail(
                strict=True,
                reason="Braket LocalSimulator generic nonadjacent ctrl modifiers disagree with the exact AE unitary",
            ),
        ),
        "cudaq",
        "hugr",
    ],
)
def test_research_source_executes_across_targets(backend):
    """The same four-qubit QPE/AE sources preserve numeric answers across targets.

    Both programs return samples, so an expectation ABI is inapplicable.
    """
    from importlib import import_module

    from tests.qamomile.resource_programs import amplitude_sample, phase_sample

    dependencies = {
        "qiskit": "qiskit_aer",
        "cirq": "cirq",
        "quri_parts": "qulacs",
        "braket": "braket.circuits",
        "cudaq": "cudaq",
        "hugr": "selene_sim",
    }
    endpoints = {
        "qiskit": "Aer",
        "cirq": "CirqSimulator",
        "quri_parts": "Qulacs",
        "braket": "BraketLocal",
        "cudaq": "CudaqLocal",
        "hugr": "Selene",
    }
    if backend == "reference":
        endpoint = qm.Reference()
    else:
        pytest.importorskip(dependencies[backend])
        integration = import_module(f"qamomile.{backend}")
        endpoint = getattr(integration, endpoints[backend])()
    for kernel in (phase_sample, amplitude_sample):
        artifact = qm.compile(qm.specialize(kernel, precision=3), to=endpoint.target)
        result = qm.execute(
            artifact, on=endpoint, shots=4, seed=0 if backend == "braket" else 2
        )
        if backend == "braket" and kernel is amplitude_sample:
            # Exact SDK probabilities make the known failure deterministic;
            # four random samples alone could happen to hide it.
            from braket.devices import LocalSimulator
            from braket.ir.openqasm import Program

            from qamomile.braket import load_openqasm

            source = load_openqasm(artifact).split("c[0] = measure", 1)[0]
            native = (
                LocalSimulator()
                .run(
                    Program(source=source + "\n#pragma braket result state_vector"),
                    shots=0,
                )
                .result()
            )
            probabilities = np.abs(native.values[0]) ** 2
            # Braket statevector order is q0-first. q0 is the target;
            # q1:q3 must be 001 or 111, independently of its final value.
            assert sum(probabilities[i] for i in (1, 7, 9, 15)) == pytest.approx(1.0)
        if kernel is phase_sample:
            assert result.values == {0.25: 4}
        else:
            assert sum(result.values.values()) == 4
            for observed in result.values:
                assert np.sin(np.pi * observed) ** 2 == pytest.approx(
                    np.sin(np.pi / 8) ** 2
                )


def test_nested_control_and_multiplex_preserve_context():
    """Nested control is CCX, while a two-case selector includes its zero mask."""
    source = """
import qamomile as qm
@qm.qkernel
def flip(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(q)
@qm.qkernel
def pair(s: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    c, q = qm.split_at(s, 1)
    c, q = qm.control(flip, c, q)
    return qm.concat(c, q)
@qm.qkernel
def nested() -> qm.Shot[qm.Bits[3]]:
    c, q = qm.control(pair, qm.qubits(1), qm.qubits(2))
    return qm.measure(qm.concat(c, q))
@qm.qkernel
def selected() -> qm.Shot[qm.Bits[2]]:
    c, q = qm.multiplex(qm.qubits(1), (flip, flip), qm.qubits(1))
    return qm.measure(qm.concat(c, q))
"""
    module = qm.build_source(source)
    nested = qr.project(qr.logical(module["nested"]), qr.GateCostModel())
    assert nested.metric(qr.portable_metric("toffoli_gates")).numeric_enclosure() == (
        1,
        1,
    )
    assert nested.metric(qr.portable_metric("gates")).numeric_enclosure() == (1, 1)
    selected = qr.project(qr.logical(module["selected"]), qr.GateCostModel())
    assert selected.metric(qr.portable_metric("gates")).numeric_enclosure() == (4, 4)


def test_external_source_rule_does_not_claim_a_portable_implementation():
    """A logical axiom cannot be reused as a circuit realization."""
    source = """
import qamomile as qm
@qm.qkernel(kind="unitary")
def oracle(q: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(oracle(qm.qubits(1)))
"""
    module = qm.build_source(source)
    rule = qr.ResourceRule(
        identity="opaque.logical.v1",
        callable=module.function("oracle"),
        metric=qr.LOGICAL_GATES,
        relation=qr.Relation.EQUALITY,
        expression=qr.literal(17),
        provenance="test logical contract",
    )
    original = qr.logical(module["sample"], rules=rule)
    projected = qr.project(original, qr.GateCostModel())
    assert projected.metric(qr.LOGICAL_GATES) == original.metric(qr.LOGICAL_GATES)
    assert projected.metric(qr.portable_metric("gates")).finite_enclosure()[1] is None
    assert projected.breakdown().call_count("oracle").numeric_enclosure() == (1, 1)


def test_basis_preparation_and_selection_budgets_fail_closed(study):
    """Static basis preparation closes exactly, while selections remain bounded."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.prepare((1,)))
"""
    projected = qr.project(
        qr.logical(qm.build_source(source)["sample"]), qr.GateCostModel()
    )
    assert projected.metric(qr.portable_metric("gates")).numeric_enclosure() == (1, 1)
    assert projected.metric(qr.portable_metric("gate_depth")).numeric_enclosure() == (
        1,
        1,
    )
    with pytest.raises(ValueError, match="4096"):
        study.breakdown(expand=("oracle",) * 4097)


def test_imported_boundary_opens_without_repeating_generated_names(research_studies):
    """A visible imported boundary can be opened without naming its specialization."""
    for name in ("QFT", "QPE", "amplitude_estimation"):
        coarse = research_studies[name].breakdown()
        opened = coarse.expand()
        assert opened.rows != coarse.rows


@pytest.mark.parametrize("precision", [1, 3])
def test_target_adjoint_verification_retains_entry_dimensions(precision):
    """Transient lowering preserves the Const binding used by a dependent return ABI."""
    from qamomile._native import compiler as native
    from tests.qamomile.resource_programs import amplitude_sample, phase_sample

    for kernel in (phase_sample, amplitude_sample):
        payload = qm.specialize(kernel, precision=precision).to_bytes()
        assert native.lower_adjoint_calls_for_target(payload)
        assert native.lower_optimized_adjoint_calls_for_target(payload)
