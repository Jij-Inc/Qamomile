"""Conformance tests for Qamomile symbolic resources and final lifecycle."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

import qamomile as qm
from qamomile.circuit import lower_to_circuit
from qamomile.effects import CallableKind, Capability, Effect, Requirement
from qamomile.persistence import deserialize, export_payload, serialize
from qamomile.proto import resource_analysis_pb2 as resource_wire
from qamomile.stdlib import phase_estimation

from .conftest import (
    BELL_SOURCE,
    CALLABLE_TRANSFORMS_SOURCE,
    COLLECTION_LOOPS_SOURCE,
    CONTROLLED_PHASE_SOURCE,
    EXPLICIT_ADJOINT_SOURCE,
    MEASURE_AXES_SOURCE,
    MODULAR_CLEANUP_SOURCE,
    MULTI_INPUT_CLEANUP_SOURCE,
    PAULI_EVOLUTION_SOURCE,
    STRUCTURAL_MATH_SOURCE,
    lower_adjoint_calls,
)

SYMBOLIC_LOOP_SOURCE = """
import qamomile as qm

@qm.qkernel
def repeat(n: qm.Const[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(n):
        state = qm.h(state)
    return qm.measure(state)
"""


CONST_BRANCH_SOURCE = """
import qamomile as qm

@qm.qkernel
def choose(flag: qm.Const[qm.Bool]) -> qm.Bits[1]:
    state = qm.qubits(1)
    if flag:
        state = qm.h(state)
    return qm.measure(state)
"""


@pytest.fixture(scope="module")
def bell_resource_study() -> qm.resources.ResourceStudy:
    """Build one immutable Bell resource study shared by report-only tests."""

    return qm.resources.logical(qm.build_source(BELL_SOURCE).specialize("bell"))


def test_resource_values_render_relations_without_dataclass_internals() -> None:
    """Every public relation has one compact deterministic text form."""

    metric = qm.resources.LOGICAL_GATES
    values = (
        qm.resources.ResourceValue.equality(metric, 2),
        qm.resources.ResourceValue.lower_bound(metric, 2),
        qm.resources.ResourceValue.upper_bound(metric, 2),
        qm.resources.ResourceValue.enclosure(metric, 1, 3),
        qm.resources.ResourceValue.partial(metric, lower=1, unknown=("open loop",)),
        qm.resources.ResourceValue.unknown(metric, "opaque callable"),
    )

    assert tuple(map(str, values)) == (
        "= 2 semantic_operations",
        ">= 2 semantic_operations",
        "<= 2 semantic_operations",
        "in [1, 3] semantic_operations",
        "in [1, ?] semantic_operations; unknown: open loop",
        "unknown semantic_operations; unknown: opaque callable",
    )


def test_program_family_uses_the_final_functional_lifecycle() -> None:
    """Module selection and specialization expose the normative value stages."""

    module = qm.build_source(SYMBOLIC_LOOP_SOURCE)
    family = module["repeat"]
    program = qm.specialize(family, n=4)

    assert isinstance(module, qm.Module)
    assert isinstance(family, qm.ProgramFamily)
    assert isinstance(program, qm.Program)
    assert family.function.symbol.endswith(".repeat")


def test_qpe_resources_preserve_arbitrary_precision_loop_bounds() -> None:
    """QPE query counts retain powers that cross the signed 64-bit boundary."""

    @qm.qkernel(kind="unitary")
    def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]: ...

    @qm.qkernel
    def qpe(
        precision: qm.Const[qm.Int],
    ) -> qm.Shot[qm.UFixed[0, precision]]:  # type: ignore[valid-type]
        target = qm.prepare((1,))
        phase, target = phase_estimation(oracle, target, precision)
        result = qm.measure(phase)
        _target_result = qm.measure(target)
        return result

    rule = qm.resources.ResourceRule(
        identity="test.qpe.arbitrary-precision.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(1),
        modes=frozenset({qm.resources.TransformMode.CONTROLLED}),
        provenance="test.qpe.arbitrary-precision-loop-bound.v1",
    )

    study = qm.resources.logical(qpe, rules=rule)
    for precision in (63, 64, 65):
        evaluated = study.evaluate(precision=precision)
        queries = evaluated.metric(qm.resources.algorithmic_query_count(oracle.symbol))
        assert queries.numeric_enclosure() == (
            2**precision - 1,
            2**precision - 1,
        )
        assert evaluated.metric(
            qm.resources.LOGICAL_PEAK_QUBITS
        ).numeric_enclosure() == (precision + 1, precision + 1)
    assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study


def test_collection_loop_resources_use_the_same_normalized_program() -> None:
    """Resource analysis counts enumerate/zip work after shared specialization."""

    program = qm.build_source(COLLECTION_LOOPS_SOURCE).specialize("collection_loops")
    study = qm.resources.logical(program)

    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (3, 3)
    assert study.metric(
        qm.resources.logical_operation_count("quantum.x")
    ).numeric_enclosure() == (3, 3)


def test_pauli_measurement_is_one_logical_measurement_per_application() -> None:
    """Logical resources do not expose backend-specific basis rotations."""

    program = qm.build_source(MEASURE_AXES_SOURCE).specialize("measure_axes")
    study = qm.resources.logical(program)

    assert study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        9,
        9,
    )
    for opcode in ("quantum.measure_x", "quantum.measure_y", "quantum.measure_z"):
        assert study.metric(
            qm.resources.logical_operation_count(opcode)
        ).numeric_enclosure() == (2, 2)


def test_callable_transforms_are_analyzed_from_their_function_bodies() -> None:
    """Body-backed transforms retain operations without inventing gate families."""

    selected = qm.resources.logical(
        qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize("callable_transforms")
    )
    phased = qm.resources.logical(
        qm.build_source(CONTROLLED_PHASE_SOURCE).specialize("controlled_phase")
    )

    assert selected.metric(
        qm.resources.logical_operation_count("qcore.select_call")
    ).numeric_enclosure() == (1, 1)
    assert selected.metric(
        qm.resources.logical_operation_count("quantum.x")
    ).numeric_enclosure() == (2, 2)
    assert phased.metric(
        qm.resources.logical_operation_count("quantum.global_phase")
    ).numeric_enclosure() == (1, 1)
    gate_metrics = (
        qm.resources.LOGICAL_SPAN,
        qm.resources.LOGICAL_GATES,
        qm.resources.LOGICAL_SINGLE_QUBIT_GATES,
        qm.resources.LOGICAL_TWO_QUBIT_GATES,
        qm.resources.LOGICAL_MULTI_QUBIT_GATES,
        qm.resources.LOGICAL_CLIFFORD_GATES,
        qm.resources.LOGICAL_ROTATION_GATES,
        qm.resources.LOGICAL_T_GATES,
        qm.resources.LOGICAL_TOFFOLI_GATES,
        qm.resources.LOGICAL_NON_CLIFFORD_GATES,
        qm.resources.LOGICAL_GATE_DEPTH,
    )
    for study in (selected, phased):
        for metric in gate_metrics:
            value = study.metric(metric)
            assert value.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
            assert value.lower is not None
            assert value.upper is None
            assert any(
                "body-backed controlled transform" in reason
                for reason in value.unknown_contributors
            )

    with pytest.raises(ValueError, match="body-backed controlled transform"):
        qm.resources.logical(
            qm.build_source(CALLABLE_TRANSFORMS_SOURCE).specialize(
                "callable_transforms"
            ),
            strict=True,
        )


def test_clean_ancilla_control_projection_restores_the_selected_circuit_model() -> None:
    """A three-control call projects to the documented Toffoli ladder."""

    source = """
import qamomile as qm

@qm.qkernel
def controlled_hadamard(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def three_controlled_hadamard() -> qm.Q[qm.Bits[4]]:
    controls = qm.qubits(3)
    target = qm.qubits(1)
    controls, target = qm.control(controlled_hadamard, controls, target)
    return qm.concat(controls, target)
"""
    program = qm.build_source(
        source,
        module_name="clean_ancilla_control_projection",
    ).specialize("three_controlled_hadamard")
    logical = qm.resources.logical(program)
    projected = qm.resources.clean_ancilla_controls(logical)

    assert projected.metric(qm.resources.CIRCUIT_VIRTUAL_GATES).numeric_enclosure() == (
        5,
        5,
    )
    assert projected.metric(qm.resources.CIRCUIT_TOFFOLI_GATES).numeric_enclosure() == (
        4,
        4,
    )
    assert projected.metric(qm.resources.CIRCUIT_GATE_DEPTH).numeric_enclosure() == (
        5,
        5,
    )
    assert projected.metric(
        qm.resources.CIRCUIT_CLEAN_ANCILLA_QUBITS
    ).numeric_enclosure() == (2, 2)
    assert projected.metric(
        qm.resources.CIRCUIT_VIRTUAL_QUBITS
    ).numeric_enclosure() == (6, 6)

    gates = projected.metric(qm.resources.CIRCUIT_VIRTUAL_GATES)
    assert gates.evidence[0].authority is qm.resources.EvidenceAuthority.TRUSTED_AXIOM
    assert gates.assumptions
    assert projected.semantic_digest == logical.semantic_digest
    assert logical.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (1, None)


def test_pauli_evolution_remains_one_abstract_logical_operation() -> None:
    """Resource analysis does not confuse one semantic evolution with its gadget."""

    program = qm.build_source(PAULI_EVOLUTION_SOURCE).specialize(
        "pauli_evolution",
        **{"hamiltonian": 0.5 * qm.X(0) * qm.X(1)},
    )
    study = qm.resources.logical(program)

    assert study.metric(
        qm.resources.logical_operation_count("quantum.pauli_evolution")
    ).numeric_enclosure() == (1, 1)
    assert (
        qm.resources.logical_operation_count("quantum.cx").identifier
        not in study.values
    )


def test_symbolic_loop_is_not_unrolled_and_evaluates_portably() -> None:
    """A Const loop remains a closed expression until specialization or evaluation."""

    family = qm.build_source(SYMBOLIC_LOOP_SOURCE)["repeat"]
    study = qm.resources.logical(family)
    work = study.values[qm.resources.LOGICAL_WORK.identifier]

    assert work.expression is not None
    assert "range_trip_count" in str(work.expression)
    assert (
        study.evaluate(n=4)
        .values[qm.resources.LOGICAL_WORK.identifier]
        .expression.numeric()
        == 6
    )
    assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study
    assert study.semantic_digest == family.semantic_digest
    assert len(study.semantic_digest) == 64


def test_native_symbolic_dag_decodes_without_per_node_renormalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified native projection crosses Python in one linear typed decode."""

    from qamomile._native import compiler

    family = qm.build_source(SYMBOLIC_LOOP_SOURCE)["repeat"]

    def reject(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("native symbolic nodes were re-normalized in Python")

    monkeypatch.setattr(compiler, "normalize_resource_expression", reject)
    study = qm.resources.logical(family)

    assert study.evaluate(n=4).metric(
        qm.resources.LOGICAL_WORK
    ).numeric_enclosure() == (6, 6)


def test_resource_metric_units_are_attached_by_the_native_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public symbolic studies cross one Rust unit-projection boundary."""

    from qamomile._native import compiler

    original = compiler.retag_resource_expression_unit
    calls: list[tuple[bytes, str]] = []

    def observed(payload: bytes, unit: str) -> bytes:
        calls.append((payload, unit))
        return original(payload, unit)

    monkeypatch.setattr(compiler, "retag_resource_expression_unit", observed)
    study = qm.resources.logical(qm.build_source(SYMBOLIC_LOOP_SOURCE)["repeat"])
    work = study.metric(qm.resources.LOGICAL_WORK)

    assert calls
    assert work.expression is not None
    assert work.expression.unit == qm.resources.SEMANTIC_OPS
    assert {unit for _, unit in calls} >= {"semantic_operations"}


def test_resource_metric_unit_projection_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Metric unit projection cannot fall back to Python tree rewriting."""

    import qamomile.resources.analysis as resource_analysis

    family = qm.build_source(SYMBOLIC_LOOP_SOURCE)["repeat"]
    original = resource_analysis.native.retag_resource_expression_unit

    monkeypatch.setattr(
        resource_analysis.native,
        "retag_resource_expression_unit",
        None,
    )
    with pytest.raises(RuntimeError, match="Q4243:.*projection is unavailable"):
        qm.resources.logical(family)

    monkeypatch.setattr(
        resource_analysis.native,
        "retag_resource_expression_unit",
        lambda *_: object(),
    )
    with pytest.raises(TypeError, match="Q4243:.*returned malformed data"):
        qm.resources.logical(family)

    monkeypatch.setattr(
        resource_analysis.native,
        "retag_resource_expression_unit",
        original,
    )


def test_resource_metric_unit_projection_has_a_nesting_budget() -> None:
    """Deep piecewise data fails natively without Python recursion."""

    import qamomile.resources.analysis as resource_analysis

    expression = qm.resources.symbol("n", scope="unit-depth")
    condition = qm.resources.Predicate("true")
    for _ in range(1_025):
        expression = qm.resources.Expr(
            "piecewise",
            qm.resources.Domain.NATURAL,
            arguments=(expression, expression),
            predicates=(condition,),
        )

    with pytest.raises(ValueError, match="Q4236.*nesting-depth"):
        resource_analysis._with_unit(expression, qm.resources.LOGICAL_WORK)


def test_resource_expression_wire_hash_consing_uses_shallow_dag_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared symbolic DAGs encode without recursively hashing Python nodes."""

    from qamomile.proto import resource_analysis_pb2 as wire
    from qamomile.resources._rule_wire import encode_resource_expression_graph
    from qamomile.resources.expr import ONE, Domain, Expr, literal

    expression = literal(1)
    for _ in range(2_048):
        expression = Expr(
            "add",
            Domain.NATURAL,
            ONE,
            (expression, expression),
        )

    def reject(_expression: Expr) -> int:
        raise AssertionError("the expression encoder recursively hashed a DAG node")

    monkeypatch.setattr(Expr, "__hash__", reject)
    graph = wire.ResourceExpressionGraphProto.FromString(
        encode_resource_expression_graph(expression)
    )

    assert len(graph.nodes) == 2_049


def test_structural_math_remains_exact_in_symbolic_resources() -> None:
    """Ceil/log2 loop counts use the shared resource algebra, not unknown bounds."""

    family = qm.build_source(STRUCTURAL_MATH_SOURCE)["structural_math"]
    study = qm.resources.logical(family)
    x_gates = study.metric(qm.resources.logical_operation_count("quantum.x"))

    assert "native-symbolic-logical-counts" in study.derivations
    assert x_gates.expression is not None
    assert "ceiling" in str(x_gates.expression)
    assert "log2" in str(x_gates.expression)
    assert (
        study.evaluate(size=8)
        .metric(qm.resources.logical_operation_count("quantum.x"))
        .expression.numeric()
        == 3
    )


def test_public_resource_analysis_preserves_native_rejections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified native rejection remains visible with its stable code."""

    from qamomile import _qamomile_native as native

    family = qm.build_source(STRUCTURAL_MATH_SOURCE)["structural_math"]

    def reject(_payload: bytes) -> object:
        """Simulate one native coverage failure."""

        raise ValueError("Q4211: symbolic test coverage gap")

    monkeypatch.setattr(native, "inspect_symbolic_program_family_counts", reject)

    with pytest.raises(
        qm.CapabilityError,
        match="Q4211: symbolic test coverage gap",
    ):
        qm.resources.logical(family)


def test_public_resource_analysis_rejects_an_incomplete_native_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed but stale extension is not a license to change engines."""

    from qamomile import _qamomile_native as native

    family = qm.build_source(STRUCTURAL_MATH_SOURCE)["structural_math"]
    monkeypatch.delattr(native, "inspect_symbolic_program_family_counts")

    with pytest.raises(
        qm.CapabilityError,
        match="Q4214: the native compiler does not provide",
    ):
        qm.resources.logical(family)


def test_public_resource_analysis_rejects_missing_native_relation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python never reconstructs a relation omitted by the native compiler."""

    from types import SimpleNamespace

    from qamomile import _qamomile_native as native

    inspect = native.inspect_symbolic_program_family_counts

    class BrokenProjection:
        """Delegate every field except the deliberately stale work count."""

        def __init__(self, projection: object) -> None:
            """Wrap one otherwise valid native projection."""

            self._projection = projection
            work = getattr(projection, "work")
            self.work = SimpleNamespace(
                lower=work.lower,
                upper=work.upper,
                exact=work.exact,
                unknown=work.unknown,
                rule_identities=work.rule_identities,
            )

        def __getattr__(self, name: str) -> object:
            """Forward untouched native fields."""

            return getattr(self._projection, name)

    def omit_relation(payload: bytes) -> object:
        """Return one stale projection without compiler relation metadata."""

        return BrokenProjection(inspect(payload))

    monkeypatch.setattr(
        native,
        "inspect_symbolic_program_family_counts",
        omit_relation,
    )
    family = qm.build_source(BELL_SOURCE)["bell"]

    with pytest.raises(TypeError, match="native symbolic count violates"):
        qm.resources.logical(family)


def test_strict_resource_analysis_requires_canonical_native_unknown_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict coverage consumes one canonical compiler-owned aggregate."""

    from qamomile import _qamomile_native as native

    inspect = native.inspect_symbolic_program_family_counts
    projected_unknown: list[object] = [["native aggregate sentinel"]]

    class ProjectionWithUnknown:
        """Override only the compiler-owned unknown contributor union."""

        def __init__(self, projection: object) -> None:
            """Wrap a valid projection with the current aggregate fixture."""

            self._projection = projection
            self.unknown = projected_unknown[0]

        def __getattr__(self, name: str) -> object:
            """Forward all ordinary projection fields."""

            return getattr(self._projection, name)

    def add_unknown(payload: bytes) -> object:
        """Return a valid projection with controlled aggregate metadata."""

        return ProjectionWithUnknown(inspect(payload))

    monkeypatch.setattr(
        native,
        "inspect_symbolic_program_family_counts",
        add_unknown,
    )
    family = qm.build_source(BELL_SOURCE)["bell"]

    qm.resources.logical(family)
    with pytest.raises(ValueError, match="native aggregate sentinel"):
        qm.resources.logical(family, strict=True)

    projected_unknown[0] = ["duplicate", "duplicate"]
    with pytest.raises(TypeError, match="unknown contributors violate"):
        qm.resources.logical(family)


def test_native_resource_projection_does_not_reenter_python_analyzer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A complete Rust projection is the sole public analysis traversal."""

    import qamomile.resources.analysis as resource_analysis

    native = resource_analysis._native_symbolic_logical_counts
    events: list[str] = []

    def native_first(*args: object, **kwargs: object) -> object:
        """Record the native aggregate boundary before forwarding the request."""

        events.append("native")
        return native(*args, **kwargs)

    monkeypatch.setattr(
        resource_analysis,
        "_native_symbolic_logical_counts",
        native_first,
    )

    study = qm.resources.logical(qm.build_source(SYMBOLIC_LOOP_SOURCE)["repeat"])

    assert not hasattr(resource_analysis, "_LogicalAnalyzer")
    assert events == ["native"]
    assert "native-symbolic-logical-counts" in study.derivations


def test_display_evaluation_cannot_authorize_compilation() -> None:
    """Formula substitution must not stand in for program specialization."""

    family = qm.build_source(SYMBOLIC_LOOP_SOURCE)["repeat"]
    study = qm.resources.logical(family.specialize(n=4))
    evaluated = study.evaluate(n=4)

    with pytest.raises(TypeError, match="specialize the source program"):
        qm.compile(evaluated, to=qm.Reference())


def test_evaluate_does_not_claim_to_respecialize_control_structure() -> None:
    """Formula substitution preserves a branch enclosure while specialization folds it."""

    family = qm.build_source(CONST_BRANCH_SOURCE)["choose"]
    symbolic = qm.resources.logical(family)
    evaluated = symbolic.evaluate(flag=True)
    specialized = qm.resources.logical(qm.specialize(family, flag=True))
    metric = qm.resources.LOGICAL_WORK.identifier

    assert evaluated.values[metric].relation is qm.resources.Relation.ENCLOSURE
    assert specialized.values[metric].relation is qm.resources.Relation.EQUALITY


def test_break_and_continue_are_part_of_resource_control_flow() -> None:
    """Loop exits determine body multiplicity instead of being flattened away."""

    source = """
import qamomile as qm

@qm.qkernel
def break_once() -> qm.Unit:
    state = qm.qubits(1)
    for index in range(5):
        state = qm.h(state)
        break
    qm.discard(state)
    return None

@qm.qkernel
def skip_first() -> qm.Unit:
    state = qm.qubits(1)
    for index in range(5):
        if index == 0:
            continue
        state = qm.h(state)
    qm.discard(state)
    return None
"""
    module = qm.build_source(source)
    broken = qm.resources.logical(module["break_once"])
    continued = qm.resources.logical(module["skip_first"])

    assert broken.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (1, 1)
    assert continued.metric(
        qm.resources.logical_operation_count("quantum.h")
    ).numeric_enclosure() == (4, 4)
    assert "native-symbolic-logical-counts" in broken.derivations
    assert "native-symbolic-logical-counts" in continued.derivations


def test_consuming_call_ends_a_quantum_lifetime() -> None:
    """Peak liveness follows linear consumption across callable boundaries."""

    source = """
import qamomile as qm

@qm.qkernel
def consume(state: qm.Q[qm.Bits[1]]) -> qm.Unit:
    qm.discard(state)
    return None

@qm.qkernel
def allocate_twice() -> qm.Unit:
    first = qm.qubits(1)
    consume(first)
    second = qm.qubits(1)
    qm.discard(second)
    return None
"""
    study = qm.resources.logical(qm.build_source(source)["allocate_twice"])

    assert study.metric(qm.resources.LOGICAL_PEAK_QUBITS).numeric_enclosure() == (1, 1)


def test_dependency_span_uses_logical_lane_footprints() -> None:
    """Independent allocations and disjoint gates form parallel dependency paths."""

    source = """
import qamomile as qm

@qm.qkernel
def parallel() -> tuple[qm.Shot[qm.Bits[2]], qm.Shot[qm.Bits[2]]]:
    left = qm.qubits(2)
    left = qm.h(left, target=0)
    right = qm.qubits(2)
    right = qm.h(right, target=1)
    return (qm.measure(left), qm.measure(right))

@qm.qkernel
def disjoint_lanes() -> qm.Shot[qm.Bits[3]]:
    state = qm.qubits(3)
    state = qm.h(state, target=0)
    state = qm.h(state, target=2)
    return qm.measure(state)

@qm.qkernel
def synchronized_t_layers() -> qm.Shot[qm.Bits[2]]:
    state = qm.qubits(2)
    state = qm.t(state, target=0)
    state = qm.cx(state, control=0, target=1)
    state = qm.t(state, target=1)
    return qm.measure(state)
"""
    module = qm.build_source(source)
    study = qm.resources.logical(module.specialize("parallel"))
    disjoint = qm.resources.logical(module.specialize("disjoint_lanes"))
    synchronized = qm.resources.logical(module.specialize("synchronized_t_layers"))
    span = study.metric(qm.resources.LOGICAL_SPAN)
    work = study.metric(qm.resources.LOGICAL_WORK)

    assert span.relation is qm.resources.Relation.EQUALITY
    assert span.numeric_enclosure() == (3, 3)
    assert work.numeric_enclosure() == (6, 6)
    assert disjoint.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (3, 3)
    assert disjoint.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (4, 4)
    assert disjoint.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
        1,
        1,
    )
    assert synchronized.metric(qm.resources.LOGICAL_T_DEPTH).numeric_enclosure() == (
        2,
        2,
    )


def test_primitive_gate_omission_shares_one_circuit_and_resource_contract() -> None:
    """Unary omission broadcasts while multi-lane gates keep ordered defaults."""

    source = """
import qamomile as qm

@qm.qkernel
def defaults() -> qm.Shot[qm.Bits[3]]:
    state = qm.qubits(3)
    state = qm.h(state)
    state = qm.x(state)
    state = qm.y(state)
    state = qm.z(state)
    state = qm.s(state)
    state = qm.sdg(state)
    state = qm.t(state)
    state = qm.tdg(state)
    state = qm.p(state, 0.0)
    state = qm.rx(state, 0.0)
    state = qm.ry(state, 0.0)
    state = qm.rz(state, 0.0)
    state = qm.cx(state)
    state = qm.cz(state)
    state = qm.ccx(state)
    state = qm.cp(state, 0.0)
    state = qm.rzz(state, 0.0)
    state = qm.swap(state)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="primitive_gate_defaults")
    program = module.specialize("defaults")
    circuit = lower_to_circuit(program)
    broadcast_names = ("h", "x", "y", "z")
    expected = [
        *((name, (target,)) for name in broadcast_names for target in range(3)),
        ("cx", (0, 1)),
        ("cz", (0, 1)),
        ("ccx", (0, 1, 2)),
        ("swap", (0, 1)),
    ]
    gate_names = {opcode for opcode, _ in expected}
    gates = [
        instruction
        for instruction in circuit.body.instructions
        if instruction.opcode in gate_names
    ]

    assert [(gate.opcode, gate.qubits) for gate in gates] == expected
    for subject in (module["defaults"], program):
        study = qm.resources.logical(subject)
        assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (42, 42)
        assert study.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
            18,
            18,
        )


def test_specialized_indexed_loop_parallelizes_disjoint_lanes() -> None:
    """A bounded proof distinguishes data-parallel from shared-lane loops."""

    source = """
import qamomile as qm

@qm.qkernel
def indexed(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.qubits(n)
    for index in range(n):
        state = qm.h(state, target=index)
    return qm.measure(state)

@qm.qkernel
def shared(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.qubits(n)
    for _ in range(n):
        state = qm.h(state, target=0)
    return qm.measure(state)
"""
    module = qm.build_source(source)
    symbolic_indexed = qm.resources.logical(module["indexed"])
    symbolic_shared = qm.resources.logical(module["shared"])
    indexed = qm.resources.logical(module.specialize("indexed", n=5))
    shared = qm.resources.logical(module.specialize("shared", n=5))

    symbolic_depth = symbolic_indexed.metric(qm.resources.LOGICAL_GATE_DEPTH)
    assert symbolic_depth.relation is qm.resources.Relation.EQUALITY
    assert symbolic_depth.evaluate({"n": 100_000}).numeric_enclosure() == (1, 1)
    assert (
        symbolic_shared.metric(qm.resources.LOGICAL_GATE_DEPTH).relation
        is qm.resources.Relation.UPPER_BOUND
    )
    assert indexed.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (5, 5)
    assert indexed.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
        1,
        1,
    )
    assert shared.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
        5,
        5,
    )


def test_symbolic_dependency_algebra_composes_calls_inside_lane_maps() -> None:
    """Callable bodies and affine loops share one compact dependency algebra."""

    source = """
import qamomile as qm

@qm.qkernel
def layer(state: qm.Q[qm.Bits[n]], target: qm.Int) -> qm.Q[qm.Bits[n]]:
    state = qm.h(state, target=target)
    return qm.t(state, target=target)

@qm.qkernel
def mapped(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    state = qm.qubits(n)
    for index in range(n):
        state = layer(state, index)
    return qm.measure(state)
"""
    study = qm.resources.logical(qm.build_source(source)["mapped"])

    gate_depth = study.metric(qm.resources.LOGICAL_GATE_DEPTH)
    assert gate_depth.relation is qm.resources.Relation.EQUALITY
    assert gate_depth.evaluate({"n": 100_000}).numeric_enclosure() == (2, 2)
    assert study.metric(qm.resources.LOGICAL_CLIFFORD_DEPTH).evaluate(
        {"n": 100_000}
    ).numeric_enclosure() == (1, 1)
    assert study.metric(qm.resources.LOGICAL_T_DEPTH).evaluate(
        {"n": 100_000}
    ).numeric_enclosure() == (1, 1)

    specialized = qm.resources.logical(
        qm.build_source(source).specialize("mapped", n=7)
    )
    assert specialized.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
        2,
        2,
    )


def test_dynamic_selectors_on_independent_owners_retain_exact_depth() -> None:
    """Param lane choices cannot alias when their ownership roots are disjoint."""

    source = """
import qamomile as qm

@qm.qkernel
def dynamic_parallel(
    width: qm.Const[qm.Int],
    left_target: qm.Param[qm.Int],
    right_target: qm.Param[qm.Int],
) -> tuple[qm.Shot[qm.Bits[width]], qm.Shot[qm.Bits[width]]]:
    left = qm.qubits(width)
    right = qm.qubits(width)
    left = qm.h(left, target=left_target)
    right = qm.t(right, target=right_target)
    return qm.measure(left), qm.measure(right)
"""
    study = qm.resources.logical(qm.build_source(source)["dynamic_parallel"])

    assert "native-symbolic-logical-counts" in study.derivations
    assert study.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (1, 1)
    assert study.metric(qm.resources.LOGICAL_CLIFFORD_DEPTH).numeric_enclosure() == (
        1,
        1,
    )
    assert study.metric(qm.resources.LOGICAL_T_DEPTH).numeric_enclosure() == (1, 1)


def test_dependency_algebra_instantiates_calls_and_adjoint_without_reanalysis() -> None:
    """Ordinary and reversed calls retain the callee's lane-level schedule."""

    source = """
import qamomile as qm

@qm.qkernel
def disjoint(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    state = qm.h(state, target=0)
    return qm.h(state, target=1)

@qm.qkernel
def called() -> qm.Shot[qm.Bits[2]]:
    state = disjoint(qm.qubits(2))
    return qm.measure(state)

@qm.qkernel
def reversed_call() -> qm.Shot[qm.Bits[2]]:
    state = qm.adjoint(disjoint, qm.qubits(2))
    return qm.measure(state)
"""
    module = qm.build_source(source)

    for name in ("called", "reversed_call"):
        study = qm.resources.logical(module.specialize(name))
        assert study.metric(qm.resources.LOGICAL_SPAN).numeric_enclosure() == (3, 3)
        assert study.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
            1,
            1,
        )


def test_resources_bound_an_optional_post_return_continuation() -> None:
    """A continuation after a runtime return contributes only to upper work."""

    source = """
import qamomile as qm

@qm.qkernel
def choose() -> qm.Shot[qm.Bool]:
    candidate = qm.measure(qm.prepare(False))
    if candidate:
        return candidate
    return qm.measure(qm.prepare(False))
"""
    study = qm.resources.logical(
        qm.build_source(source, module_name="resource_optional_return_tail").specialize(
            "choose"
        )
    )

    assert study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        1,
        2,
    )
    assert study.metric(qm.resources.LOGICAL_WORK).numeric_enclosure() == (2, 4)


def test_resources_stop_a_finite_loop_at_a_statically_selected_return() -> None:
    """Exact loop interpretation propagates return instead of counting its tail."""

    source = """
import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    for index in range(3):
        if index == 1:
            state = qm.x(state)
            return qm.measure(state)
    return qm.measure(state)
"""
    study = qm.resources.logical(
        qm.build_source(source, module_name="resource_static_for_return").specialize(
            "sample"
        )
    )

    assert study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        1,
        1,
    )
    assert study.metric(
        qm.resources.logical_operation_count("quantum.x")
    ).numeric_enclosure() == (1, 1)


def test_late_adjoint_lowering_cannot_reenter_semantic_analysis() -> None:
    """A proof-erasing target lowering cannot masquerade as verified QCore."""

    program = qm.build_source(MODULAR_CLEANUP_SOURCE).specialize("modular_cleanup")
    lowered = lower_adjoint_calls(program)
    study = qm.resources.logical(program)

    assert study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure() == (
        5,
        5,
    )
    with pytest.raises(TypeError, match="logical expects"):
        qm.resources.logical(lowered)


def test_explicit_adjoint_is_algorithmic_work_not_cleanup_work() -> None:
    """Only compiler-synthesized modular adjoints discharge cleanup obligations."""

    program = qm.build_source(EXPLICIT_ADJOINT_SOURCE).specialize("explicit_adjoint")
    study = qm.resources.logical(program)

    assert study.metric(
        qm.resources.logical_operation_count("qcore.adjoint")
    ).numeric_enclosure() == (2, 2)
    assert study.metric(qm.resources.LOGICAL_CLEANUP_WORK).numeric_enclosure() == (0, 0)


def test_strict_analysis_composes_coherent_product_width() -> None:
    """A linear tensor width is derived from its ordinary component types."""

    source = """
import qamomile as qm

@qm.qkernel
def opaque_width(state: qm.Q[tuple[qm.Bit, qm.Bit]]) -> qm.Q[tuple[qm.Bit, qm.Bit]]:
    return state
"""
    family = qm.build_source(source)["opaque_width"]

    peak = qm.resources.logical(family, strict=True).metric(
        qm.resources.LOGICAL_PEAK_QUBITS
    )

    assert peak.relation is qm.resources.Relation.EQUALITY
    assert peak.numeric_enclosure() == (2, 2)


def test_invalid_static_loop_and_allocation_domains_are_rejected() -> None:
    """Known-invalid resource domains fail validation instead of estimating zero."""

    zero_step = """
import qamomile as qm

@qm.qkernel
def invalid_loop() -> qm.Unit:
    for index in range(0, 4, 0):
        pass
    return None
"""
    invalid_width = """
import qamomile as qm

@qm.qkernel
def invalid_width(n: qm.Const[qm.Int]) -> qm.Unit:
    state = qm.qubits(n + 1)
    qm.discard(state)
    return None
"""

    with pytest.raises(qm.VerificationError, match="Q1045"):
        qm.build_source(zero_step)
    # The allocation width is the type dimension ``n+1``, so specialization
    # rejects the nonpositive binding before the allocation value is checked.
    with pytest.raises(qm.SpecializationError, match='Q3007: shape dimension "n\\+1"'):
        qm.specialize(qm.build_source(invalid_width)["invalid_width"], n=-1)


def test_closed_cleanup_is_independently_checked_and_counted() -> None:
    """The proof kernel checks the realized body before resource analysis."""

    source = """
import qamomile as qm

@qm.qkernel
def cleanup() -> qm.Unit:
    state = qm.qubits(1)
    state = qm.h(state)
    return None
"""
    module = qm.build_source(source)
    program = module.specialize("cleanup")
    cleanup = qm.resources.logical(program).values[
        qm.resources.LOGICAL_CLEANUP_WORK.identifier
    ]

    assert "qcore.forget" not in program.inspect()
    assert "quantum.release" in program.inspect()
    assert cleanup.relation is qm.resources.Relation.EQUALITY
    assert cleanup.expression is not None and cleanup.expression.numeric() == 2


def test_multi_input_cleanup_preserves_exact_product_liveness() -> None:
    """Packed call results replace their inputs without duplicating ownership."""

    program = qm.build_source(MULTI_INPUT_CLEANUP_SOURCE).specialize(
        "multi_input_cleanup"
    )
    peak = qm.resources.logical(program).metric(qm.resources.LOGICAL_PEAK_QUBITS)

    assert peak.relation is qm.resources.Relation.EQUALITY
    assert peak.numeric_enclosure() == (2, 2)


def test_compilation_report_uses_coordinates_and_round_trips_in_artifact() -> None:
    """Artifacts retain a full study instead of ambiguous flat operation counters."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=qm.Reference())
    restored = deserialize(serialize(artifact))

    with pytest.raises(KeyError):
        restored.report.metric(qm.resources.CIRCUIT_VIRTUAL_QUBITS)
    # Compilation retains a live owner; explicit export does not retroactively
    # add a size measurement to its immutable compilation report.
    with pytest.raises(KeyError):
        restored.report.metric(qm.resources.EMITTED_ARTIFACT_BYTES)
    assert len(export_payload(restored)) > 0
    assert not hasattr(restored.report, "logical_operations")


def test_logical_resource_families_preserve_core_metrics() -> None:
    """Gate families and per-qubit events remain available without a flat report."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    study = qm.resources.logical(program)

    expected = {
        qm.resources.LOGICAL_INPUT_QUBITS: (0, 0),
        qm.resources.LOGICAL_ALLOCATED_QUBITS: (2, 2),
        qm.resources.LOGICAL_CIRCUIT_QUBITS: (2, 2),
        qm.resources.LOGICAL_GATES: (2, 2),
        qm.resources.LOGICAL_SINGLE_QUBIT_GATES: (1, 1),
        qm.resources.LOGICAL_TWO_QUBIT_GATES: (1, 1),
        qm.resources.LOGICAL_MULTI_QUBIT_GATES: (0, 0),
        qm.resources.LOGICAL_CLIFFORD_GATES: (2, 2),
        qm.resources.LOGICAL_ROTATION_GATES: (0, 0),
        qm.resources.LOGICAL_T_GATES: (0, 0),
        qm.resources.LOGICAL_TOFFOLI_GATES: (0, 0),
        qm.resources.LOGICAL_NON_CLIFFORD_GATES: (0, 0),
        qm.resources.LOGICAL_MEASUREMENTS: (2, 2),
        qm.resources.LOGICAL_RESETS: (0, 0),
    }
    for metric, enclosure in expected.items():
        assert study.metric(metric).numeric_enclosure() == enclosure
    for metric, expected in {
        qm.resources.LOGICAL_GATE_DEPTH: (2, 2),
        qm.resources.LOGICAL_CLIFFORD_DEPTH: (2, 2),
        qm.resources.LOGICAL_ROTATION_DEPTH: (0, 0),
        qm.resources.LOGICAL_T_DEPTH: (0, 0),
        qm.resources.LOGICAL_TOFFOLI_DEPTH: (0, 0),
        qm.resources.LOGICAL_NON_CLIFFORD_DEPTH: (0, 0),
        qm.resources.LOGICAL_MEASUREMENT_DEPTH: (1, 1),
        qm.resources.LOGICAL_RESET_DEPTH: (0, 0),
    }.items():
        value = study.metric(metric)
        assert value.relation is qm.resources.Relation.EQUALITY
        assert value.numeric_enclosure() == expected


def test_resource_trace_and_hotspots_retain_source_local_contributions(
    bell_resource_study: qm.resources.ResourceStudy,
) -> None:
    """Full studies explain structural contributors without re-running analysis."""

    restored = qm.resources.ResourceStudy.from_bytes(bell_resource_study.to_bytes())
    encoded = resource_wire.ResourceStudyProto.FromString(restored.to_bytes())
    contributions = restored.contributions(qm.resources.LOGICAL_GATES)
    hotspots = restored.hotspots(qm.resources.LOGICAL_GATES)

    assert [item.operation for item in contributions] == [
        "quantum.h",
        "quantum.cx",
    ]
    assert all(item.value.numeric_enclosure() == (1, 1) for item in contributions)
    assert encoded.schema_minor == 0
    assert all(fact.WhichOneof("value") == "operation" for fact in encoded.bundle.facts)
    assert hotspots == contributions
    rendered = restored.trace.render()
    assert "resource-study" in rendered
    assert "quantum.h" in rendered
    explanation = restored.explain(qm.resources.LOGICAL_GATES)
    assert explanation.splitlines()[0] == (
        "qamomile.logical.gates: = 2 semantic_operations"
    )
    assert "quantum.h" in explanation
    assert "quantum.cx" in explanation
    assert "qcore.constant" not in explanation
    assert "quantum.measure" not in explanation


def test_unitary_resource_facts_reference_only_declared_metrics() -> None:
    """A transformed unitary study persists only declared metric references."""

    source = """
import qamomile as qm

@qm.qkernel
def evolve(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.h(state)

@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    state = qm.adjoint(evolve, qm.qubits(1))
    return qm.measure(state)
"""
    program = qm.build_source(
        source,
        module_name="unitary_resource_fact_metrics",
    ).specialize("sample")
    study = qm.resources.logical(program)
    restored = qm.resources.ResourceStudy.from_bytes(study.to_bytes())
    metric_names = {name for name, _ in restored.metrics}

    assert all(
        identifier in metric_names
        for fact in restored._bundle.facts
        for identifier, _ in fact.contributions
    )


def test_resource_contributions_preserve_typed_uncertainty_without_reanalysis(
    bell_resource_study: qm.resources.ResourceStudy,
) -> None:
    """Typed source facts preserve unknown work without a second analysis."""

    facts = list(bell_resource_study._bundle.facts)
    fact_index = next(
        index
        for index, fact in enumerate(facts)
        if qm.resources.LOGICAL_GATES.identifier in dict(fact.contributions)
    )
    target = facts[fact_index]
    contributions = dict(target.contributions)
    exact = contributions[qm.resources.LOGICAL_GATES.identifier]
    assert exact.expression is not None
    contributions[qm.resources.LOGICAL_GATES.identifier] = (
        qm.resources.ResourceValue.partial(
            qm.resources.LOGICAL_GATES,
            lower=exact.expression,
            unknown=("local upper endpoint unavailable",),
        )
    )
    facts[fact_index] = replace(
        target,
        contributions=tuple(sorted(contributions.items())),
    )
    partial = replace(
        bell_resource_study,
        _bundle=replace(bell_resource_study._bundle, facts=tuple(facts)),
    )

    contribution = next(
        item
        for item in partial.contributions(qm.resources.LOGICAL_GATES)
        if (item.owner, item.source, item.operation)
        == (target.owner, target.source, target.operation)
    )
    assert contribution.value.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
    assert contribution.value.unknown_contributors == (
        "local upper endpoint unavailable",
    )

    encoded = resource_wire.ResourceStudyProto.FromString(partial.to_bytes())
    encoded.ClearField("study_digest")
    contribution = encoded.bundle.facts[fact_index].operation.contributions[
        next(
            index
            for index, item in enumerate(
                encoded.bundle.facts[fact_index].operation.contributions
            )
            if item.identifier == qm.resources.LOGICAL_GATES.identifier
        )
    ]
    contribution.relation = resource_wire.RESOURCE_EQUALITY
    contribution.expression.CopyFrom(contribution.lower)
    contribution.unknown_contributors.append("opaque work")
    contribution.ClearField("lower")
    with pytest.raises(ValueError, match="Q4256"):
        from qamomile._native import compiler

        compiler.seal_resource_study(encoded.SerializeToString(deterministic=True))


def test_resource_hotspots_rank_arbitrary_precision_without_float_projection(
    monkeypatch: pytest.MonkeyPatch,
    bell_resource_study: qm.resources.ResourceStudy,
) -> None:
    """Hotspot ranking retains exact numeric order and symbolic entries."""

    metric = qm.resources.LOGICAL_GATES
    enormous = 10**400
    symbolic = qm.resources.symbol(
        "n",
        scope="hotspot",
        unit=metric.unit,
    )
    contributions = (
        qm.resources.ResourceContribution(
            "symbolic",
            "f.py:1",
            "unknown",
            qm.resources.ResourceValue.equality(metric, symbolic),
        ),
        qm.resources.ResourceContribution(
            "z",
            "f.py:2",
            "largest",
            qm.resources.ResourceValue.equality(metric, enormous),
        ),
        qm.resources.ResourceContribution(
            "a",
            "f.py:3",
            "next",
            qm.resources.ResourceValue.equality(metric, enormous - 1),
        ),
        qm.resources.ResourceContribution(
            "b", "f.py:4", "tie-b", qm.resources.ResourceValue.equality(metric, 7)
        ),
        qm.resources.ResourceContribution(
            "a", "f.py:5", "tie-a", qm.resources.ResourceValue.equality(metric, 7)
        ),
    )

    def projected_contributions(
        _study: qm.resources.ResourceStudy,
        _metric: str | qm.resources.MetricDefinition,
    ) -> tuple[qm.resources.ResourceContribution, ...]:
        """Return a focused contribution table without another analysis run."""

        return contributions

    monkeypatch.setattr(
        qm.resources.ResourceStudy,
        "contributions",
        projected_contributions,
    )
    assert [item.operation for item in bell_resource_study.hotspots(metric)] == [
        "largest",
        "next",
        "tie-a",
        "tie-b",
        "unknown",
    ]
    assert [
        item.operation for item in bell_resource_study.hotspots(metric, limit=2)
    ] == [
        "largest",
        "next",
    ]
    with pytest.raises(ValueError, match="positive integer"):
        bell_resource_study.hotspots(metric, limit=0)


def test_resource_hotspots_fail_closed_on_a_stale_native_abi(
    monkeypatch: pytest.MonkeyPatch,
    bell_resource_study: qm.resources.ResourceStudy,
) -> None:
    """Hotspot ranking cannot restore the removed Python numeric ordering."""

    from qamomile._native import compiler

    native_rank = compiler.rank_resource_hotspots

    monkeypatch.delattr(compiler, "rank_resource_hotspots")
    with pytest.raises(RuntimeError, match="Q4243"):
        bell_resource_study.hotspots(qm.resources.LOGICAL_GATES)

    monkeypatch.setattr(
        compiler,
        "rank_resource_hotspots",
        lambda _candidates, _limit: [True, 0],
        raising=False,
    )
    with pytest.raises(TypeError, match="Q4243"):
        bell_resource_study.hotspots(qm.resources.LOGICAL_GATES)

    monkeypatch.setattr(compiler, "rank_resource_hotspots", native_rank)


def test_measurement_and_reset_counts_are_per_logical_qubit() -> None:
    """Aggregate instrument operations count every affected logical lane."""

    source = """
import qamomile as qm

@qm.qkernel
def reset_then_measure() -> qm.Shot[qm.Bits[3]]:
    state = qm.qubits(3)
    state = qm.reset(state)
    return qm.measure(state)
"""
    study = qm.resources.logical(
        qm.build_source(source).specialize("reset_then_measure")
    )

    assert study.metric(qm.resources.LOGICAL_RESETS).numeric_enclosure() == (3, 3)
    assert study.metric(qm.resources.LOGICAL_MEASUREMENTS).numeric_enclosure() == (
        3,
        3,
    )


def test_compile_accepts_a_resource_study_directly() -> None:
    """Compilation consumes the authoritative study without a selection copy."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    study = qm.resources.logical(program)
    artifact = qm.compile(study, to=qm.Reference())

    assert artifact.report.resources.semantic_digest == program.semantic_digest


def test_resource_study_has_one_authoritative_metric_map() -> None:
    """A single realized body introduces no scenario or selection copies."""

    study = qm.resources.logical(qm.build_source(BELL_SOURCE)["bell"])
    encoded = resource_wire.ResourceStudyProto.FromString(study.to_bytes())

    assert encoded.schema_major == 1
    assert encoded.HasField("bundle")
    assert len(encoded.study_digest) == 32
    assert [item.identifier for item in encoded.metrics] == sorted(study.values)
    for obsolete in (
        "common",
        "envelope",
        "scenarios",
        "select",
        "bundle",
    ):
        assert not hasattr(study, obsolete)
    for obsolete in (
        "AnalysisBundle",
        "ErrorBudget",
        "MetricRegistry",
        "ResourceScenario",
        "ScenarioSelection",
    ):
        assert not hasattr(qm.resources, obsolete)


def test_surface_code_projection_is_a_pure_executable_study() -> None:
    """A physical model adds coordinates without introducing planner candidates."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    logical = qm.resources.logical(program)
    physical = qm.resources.surface_code(logical)

    assert physical.metric(qm.resources.PHYSICAL_CODE_DISTANCE).numeric_enclosure() == (
        3,
        3,
    )
    assert physical.metric(qm.resources.PHYSICAL_QUBIT_COUNT).numeric_enclosure() == (
        72,
        72,
    )
    assert physical.metric(qm.resources.PHYSICAL_DURATION).numeric_enclosure() == (
        0.0,
        0.0,
    )
    assert physical.metric(qm.resources.PHYSICAL_QUBIT_TIME).numeric_enclosure() == (
        0.0,
        0.0,
    )
    evidence = physical.metric(qm.resources.PHYSICAL_QUBIT_COUNT).evidence[0]
    assert evidence.authority is qm.resources.EvidenceAuthority.HEURISTIC_MODEL
    assert not evidence.verified

    restored = qm.resources.ResourceStudy.from_bytes(physical.to_bytes())
    artifact = qm.compile(restored, to=qm.Reference())

    assert restored == physical
    assert artifact.report.metric(
        qm.resources.PHYSICAL_QUBIT_COUNT.identifier
    ).numeric_enclosure() == (72, 72)


def test_surface_code_matches_symbolic_model_reference() -> None:
    """The RSA-scale model remains available without a planner API."""

    logical = qm.resources.logical(qm.build_source(BELL_SOURCE).specialize("bell"))
    n = 2_048
    physical = qm.resources.surface_code(
        logical,
        logical_qubits=3 * n,
        non_clifford_gates=0.3 * n**3,
    )

    assert physical.metric(qm.resources.PHYSICAL_CODE_DISTANCE).numeric_enclosure() == (
        25,
        25,
    )
    assert physical.metric(qm.resources.PHYSICAL_QUBIT_COUNT).numeric_enclosure() == (
        15_360_000,
        15_360_000,
    )
    assert physical.metric(qm.resources.PHYSICAL_DURATION).numeric_enclosure() == (
        pytest.approx(64_424.50944),
        pytest.approx(64_424.50944),
    )


def test_surface_code_formula_executes_once_in_native_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exact projection has one Rust formula authority and no fallback."""

    from qamomile._native import compiler

    logical = qm.resources.logical(qm.build_source(BELL_SOURCE).specialize("bell"))
    original = compiler.project_surface_code_resources
    calls: list[tuple[bytes, bytes]] = []

    def observed(
        logical_qubits: bytes,
        non_clifford_gates: bytes,
        alpha: float,
        threshold: float,
        physical_error_rate: float,
        syndrome_cycle_seconds: float,
    ) -> object:
        calls.append((logical_qubits, non_clifford_gates))
        return original(
            logical_qubits,
            non_clifford_gates,
            alpha,
            threshold,
            physical_error_rate,
            syndrome_cycle_seconds,
        )

    with monkeypatch.context() as context:
        context.setattr(compiler, "project_surface_code_resources", observed)
        physical = qm.resources.surface_code(logical)

    assert len(calls) == 1
    assert physical.metric(qm.resources.PHYSICAL_QUBIT_COUNT).numeric_enclosure() == (
        72,
        72,
    )
    with monkeypatch.context() as context:
        context.delattr(compiler, "project_surface_code_resources")
        with pytest.raises(RuntimeError, match="Q4239"):
            qm.resources.surface_code(logical)


def test_surface_code_projection_retains_symbolic_logical_inputs() -> None:
    """Physical roots remain evaluable expressions over the logical study inputs."""

    logical = qm.resources.logical(qm.build_source(BELL_SOURCE).specialize("bell"))
    logical_qubits = qm.resources.symbol(
        "n",
        scope="surface-code",
        unit=qm.resources.LOGICAL_QUBITS,
    )
    non_clifford = qm.resources.symbol(
        "t",
        scope="surface-code",
        unit=qm.resources.SEMANTIC_OPS,
    )
    physical = qm.resources.surface_code(
        logical,
        logical_qubits=logical_qubits,
        non_clifford_gates=non_clifford,
    )
    evaluated = physical.evaluate(n=2, t=0)

    assert physical.metric(qm.resources.PHYSICAL_CODE_DISTANCE).expression is not None
    assert (
        physical.metric(qm.resources.PHYSICAL_CODE_DISTANCE).expression.numeric()
        is None
    )
    assert evaluated.metric(
        qm.resources.PHYSICAL_CODE_DISTANCE
    ).numeric_enclosure() == (
        3,
        3,
    )
    assert evaluated.metric(qm.resources.PHYSICAL_QUBIT_COUNT).numeric_enclosure() == (
        72,
        72,
    )


def test_resource_model_chain_is_one_canonical_study_transformation() -> None:
    """Custom model composition threads only the evolving immutable study."""

    inputs: list[qm.resources.ResourceStudy] = []

    class WidthModel:
        """Add one model-conditional physical width coordinate."""

        identity = "test.width-model.v1"

        def project(
            self,
            study: qm.resources.ResourceStudy,
        ) -> dict[str, qm.resources.ResourceValue]:
            """Project width and retain its assumption in the value itself."""

            inputs.append(study)
            value = qm.resources.ResourceValue.equality(
                qm.resources.PHYSICAL_QUBIT_COUNT,
                2_000,
                assumptions=("width model assumption",),
            )
            return {value.metric.identifier: value}

    class DurationModel:
        """Add duration after observing the preceding projection."""

        identity = "test.duration-model.v1"

        def project(
            self,
            study: qm.resources.ResourceStudy,
        ) -> dict[str, qm.resources.ResourceValue]:
            """Require the preceding width to be part of the input study."""

            assert qm.resources.PHYSICAL_QUBIT_COUNT.identifier in study.values
            inputs.append(study)
            value = qm.resources.ResourceValue.equality(
                qm.resources.PHYSICAL_DURATION,
                0.01,
                assumptions=("duration model assumption",),
            )
            return {value.metric.identifier: value}

    logical = qm.resources.logical(qm.build_source(BELL_SOURCE).specialize("bell"))
    width = qm.resources.project(logical, WidthModel())
    result = qm.resources.project(width, DurationModel())

    assert inputs[0].digest == logical.digest
    assert inputs[1].digest == width.digest
    assert inputs[1].semantic_digest == width.semantic_digest
    assert result.metric(qm.resources.PHYSICAL_QUBIT_COUNT).assumptions == (
        "width model assumption",
    )
    assert result.metric(qm.resources.PHYSICAL_DURATION).assumptions == (
        "duration model assumption",
    )
    assert result.derivations[-2:] == (
        "model:test.width-model.v1",
        "model:test.duration-model.v1",
    )


def test_user_model_evidence_remains_an_unverified_claim_on_wire() -> None:
    """Projection metadata cannot mint proof authority locally or during decoding."""

    class ClaimedModel:
        """Return a deliberately optimistic user-authored proof-shaped claim."""

        identity = "test.claimed-model.v1"

        def project(
            self,
            study: qm.resources.ResourceStudy,
        ) -> dict[str, qm.resources.ResourceValue]:
            """Attach a public claim without gaining a verification bit."""

            del study
            claim = qm.resources.Evidence(
                qm.resources.EvidenceAuthority.STRUCTURALLY_DERIVED,
                "user-supplied-text-is-not-a-certificate",
            )
            value = qm.resources.ResourceValue.equality(
                qm.resources.PHYSICAL_QUBIT_COUNT,
                1,
                evidence=(claim,),
            )
            return {value.metric.identifier: value}

    logical = qm.resources.logical(qm.build_source(BELL_SOURCE).specialize("bell"))
    projected = qm.resources.project(logical, ClaimedModel())
    restored = qm.resources.ResourceStudy.from_bytes(projected.to_bytes())

    assert not projected.metric(qm.resources.PHYSICAL_QUBIT_COUNT).evidence[0].verified
    assert not restored.metric(qm.resources.PHYSICAL_QUBIT_COUNT).evidence[0].verified


def test_resource_wire_rejects_tampering_and_unknown_is_not_zero(
    bell_resource_study: qm.resources.ResourceStudy,
) -> None:
    """Integrity and mathematical ignorance remain distinct explicit states."""

    payload = bytearray(bell_resource_study.to_bytes())
    payload[len(payload) // 2] ^= 1
    unknown = qm.resources.ResourceValue.unknown(
        qm.resources.LOGICAL_WORK,
        "opaque oracle has no registered rule",
    )

    with pytest.raises(ValueError):
        qm.resources.ResourceStudy.from_bytes(payload)
    assert unknown.numeric_enclosure() == (None, None)
    assert unknown.relation is qm.resources.Relation.UNKNOWN


def test_resource_study_wire_is_deterministic_and_rejects_schema_drift(
    bell_resource_study: qm.resources.ResourceStudy,
) -> None:
    """Native study persistence is deterministic and versioned fail-closed."""

    from qamomile.resources import _wire as study_wire

    first = bell_resource_study.to_bytes()
    encoded = resource_wire.ResourceStudyProto.FromString(first)

    assert first == bell_resource_study.to_bytes()
    assert len(bytes(encoded.study_digest)) == 32
    assert encoded.schema_minor == 0

    encoded.schema_major += 1
    with pytest.raises(ValueError, match="Q4252:.*schema major"):
        qm.resources.ResourceStudy.from_bytes(
            encoded.SerializeToString(deterministic=True)
        )

    encoded = resource_wire.ResourceStudyProto.FromString(first)
    encoded.schema_minor += 1
    with pytest.raises(ValueError, match="Q4253:.*schema minor"):
        qm.resources.ResourceStudy.from_bytes(
            encoded.SerializeToString(deterministic=True)
        )

    unsealed = study_wire.encode_study(
        bell_resource_study,
        include_digest=False,
    )
    with pytest.raises(ValueError, match="Q4255:.*canonical"):
        study_wire.native.seal_resource_study(unsealed + b"\x08\x01")


def test_resource_study_persistence_requires_a_well_formed_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
    bell_resource_study: qm.resources.ResourceStudy,
) -> None:
    """Study persistence never falls back to Python hashing or validation."""

    from qamomile.resources import _wire as study_wire

    payload = bell_resource_study.to_bytes()
    seal = study_wire.native.seal_resource_study
    inspect = study_wire.native.inspect_resource_study

    monkeypatch.setattr(study_wire.native, "seal_resource_study", None)
    with pytest.raises(qm.CapabilityError, match="Q4214:.*seal_resource_study"):
        bell_resource_study.to_bytes()
    monkeypatch.setattr(study_wire.native, "seal_resource_study", lambda _payload: None)
    with pytest.raises(TypeError, match="Q4257:.*sealer returned malformed"):
        bell_resource_study.to_bytes()

    monkeypatch.setattr(study_wire.native, "seal_resource_study", seal)
    monkeypatch.setattr(study_wire.native, "inspect_resource_study", None)
    with pytest.raises(qm.CapabilityError, match="Q4214:.*inspect_resource_study"):
        qm.resources.ResourceStudy.from_bytes(payload)
    monkeypatch.setattr(
        study_wire.native, "inspect_resource_study", lambda _payload: b"short"
    )
    with pytest.raises(TypeError, match="Q4257:.*inspector returned malformed"):
        qm.resources.ResourceStudy.from_bytes(payload)
    monkeypatch.setattr(study_wire.native, "inspect_resource_study", inspect)


def test_resource_values_enforce_relation_shapes_at_construction() -> None:
    """Every relation admits only its mathematically meaningful payload."""

    one = qm.resources.literal(
        1,
        qm.resources.SEMANTIC_OPS,
        qm.resources.Domain.NATURAL,
    )

    with pytest.raises(ValueError, match="requires an expression"):
        qm.resources.ResourceValue(
            qm.resources.LOGICAL_WORK,
            qm.resources.Relation.EQUALITY,
        )
    with pytest.raises(ValueError, match="cannot carry enclosure endpoints"):
        qm.resources.ResourceValue(
            qm.resources.LOGICAL_WORK,
            qm.resources.Relation.LOWER_BOUND,
            expression=one,
            lower=one,
        )
    with pytest.raises(ValueError, match="requires lower and upper endpoints only"):
        qm.resources.ResourceValue(
            qm.resources.LOGICAL_WORK,
            qm.resources.Relation.ENCLOSURE,
            lower=one,
        )
    with pytest.raises(ValueError, match="requires a finite endpoint"):
        qm.resources.ResourceValue(
            qm.resources.LOGICAL_WORK,
            qm.resources.Relation.PARTIAL_ENCLOSURE,
            unknown_contributors=("opaque contribution",),
        )
    with pytest.raises(ValueError, match="must name unknown contributors"):
        qm.resources.ResourceValue(
            qm.resources.LOGICAL_WORK,
            qm.resources.Relation.PARTIAL_ENCLOSURE,
            lower=one,
        )
    with pytest.raises(ValueError, match="cannot carry endpoints"):
        qm.resources.ResourceValue(
            qm.resources.LOGICAL_WORK,
            qm.resources.Relation.UNKNOWN,
            expression=one,
            unknown_contributors=("opaque contribution",),
        )


def test_resource_values_enforce_metric_domains_and_concrete_bounds() -> None:
    """Metric values reject invalid carriers, negatives, and reversed intervals."""

    with pytest.raises(ValueError, match="is not contained in metric domain 'natural'"):
        qm.resources.ResourceValue.equality(
            qm.resources.LOGICAL_WORK,
            qm.resources.symbol(
                "fractional_cost",
                scope="resource_value_boundary",
                unit=qm.resources.SEMANTIC_OPS,
                domain=qm.resources.Domain.PROBABILITY,
            ),
        )
    with pytest.raises(ValueError, match="must be nonnegative"):
        qm.resources.ResourceValue.equality(
            qm.resources.PHYSICAL_DURATION,
            qm.resources.literal(
                -0.5,
                qm.resources.SECONDS,
                qm.resources.Domain.REAL,
            ),
        )
    with pytest.raises(ValueError, match="lower endpoint exceeds upper"):
        qm.resources.ResourceValue.enclosure(qm.resources.LOGICAL_WORK, 2, 1)


def test_resource_value_validation_has_one_fail_closed_native_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One native query validates a relation and stale bridges never fall back."""

    from qamomile import _qamomile_native as native

    original = native.validate_resource_value
    calls: list[tuple[object, ...]] = []

    def recording_validator(*arguments: object) -> None:
        """Record and delegate one complete relation validation."""

        calls.append(arguments)
        original(*arguments)

    monkeypatch.setattr(native, "validate_resource_value", recording_validator)
    qm.resources.ResourceValue.enclosure(qm.resources.LOGICAL_WORK, 1, 2)
    assert len(calls) == 1
    qm.resources.ResourceRule(
        identity="example.native-owner.rule.v1",
        callable="example.native_owner.oracle",
        metric=qm.resources.LOGICAL_WORK,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(1),
        provenance="example.native-owner.contract.v1",
    )
    assert len(calls) == 2
    assert calls[-1][1] == "1"

    monkeypatch.setattr(native, "validate_resource_value", None)
    with pytest.raises(RuntimeError, match="Q4242:.*validation is unavailable"):
        qm.resources.ResourceValue.equality(qm.resources.LOGICAL_WORK, 1)

    monkeypatch.setattr(native, "validate_resource_value", lambda *_: object())
    with pytest.raises(TypeError, match="Q4242:.*returned malformed data"):
        qm.resources.ResourceValue.equality(qm.resources.LOGICAL_WORK, 1)


def test_resource_values_revalidate_after_symbolic_evaluation() -> None:
    """Substitution cannot turn a valid symbolic fact into an invalid value."""

    duration = qm.resources.symbol(
        "duration",
        scope="resource_value_boundary",
        domain=qm.resources.Domain.REAL,
        unit=qm.resources.SECONDS,
    )
    exact = qm.resources.ResourceValue.equality(
        qm.resources.PHYSICAL_DURATION,
        duration,
    )
    lower = qm.resources.symbol(
        "lower",
        scope="resource_value_boundary",
        unit=qm.resources.SEMANTIC_OPS,
    )
    upper = qm.resources.symbol(
        "upper",
        scope="resource_value_boundary",
        unit=qm.resources.SEMANTIC_OPS,
    )
    enclosure = qm.resources.ResourceValue.enclosure(
        qm.resources.LOGICAL_WORK,
        lower,
        upper,
    )

    with pytest.raises(ValueError, match="must be nonnegative"):
        exact.evaluate({"duration": -0.5})
    with pytest.raises(ValueError, match="lower endpoint exceeds upper"):
        enclosure.evaluate({"lower": 2, "upper": 1})


def test_short_symbol_evaluation_rejects_cross_scope_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A short substitution cannot silently bind same-named scoped symbols."""

    from qamomile._native import compiler

    native_resolve = compiler.resolve_resource_symbol_bindings
    resolutions: list[tuple[int, tuple[str, ...]]] = []

    def observed_resolution(payloads: list[bytes], names: list[str]) -> object:
        """Record the aggregate native resolution boundary."""

        resolutions.append((len(payloads), tuple(names)))
        return native_resolve(payloads, names)

    monkeypatch.setattr(
        compiler,
        "resolve_resource_symbol_bindings",
        observed_resolution,
    )

    left = qm.resources.symbol(
        "n",
        scope="left",
        unit=qm.resources.SEMANTIC_OPS,
    )
    right = qm.resources.symbol(
        "n",
        scope="right",
        unit=qm.resources.SEMANTIC_OPS,
    )
    value = qm.resources.ResourceValue.equality(
        qm.resources.LOGICAL_WORK,
        left + right,
    )

    with pytest.raises(ValueError, match="ambiguous resource symbol 'n'"):
        value.evaluate({"n": 4})

    assert value.evaluate({"left::n": 2, "right::n": 3}).numeric_enclosure() == (5, 5)
    assert resolutions == [
        (1, ("n",)),
        (1, ("left::n", "right::n")),
    ]


def test_resource_symbol_resolution_fails_closed_on_a_stale_native_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluation cannot restore Python name resolution for a stale extension."""

    from qamomile._native import compiler

    expression = qm.resources.symbol("n", scope="native-resolution")
    native_resolve = compiler.resolve_resource_symbol_bindings

    monkeypatch.delattr(compiler, "resolve_resource_symbol_bindings")
    with pytest.raises(RuntimeError, match="Q4239"):
        expression.evaluate({"n": 1})

    monkeypatch.setattr(
        compiler,
        "resolve_resource_symbol_bindings",
        lambda _payloads, _names: [("malformed", "n")],
        raising=False,
    )
    with pytest.raises(TypeError, match="Q4239"):
        expression.evaluate({"n": 1})

    monkeypatch.setattr(
        compiler,
        "resolve_resource_symbol_bindings",
        native_resolve,
    )


@pytest.mark.parametrize(
    ("scope", "name"),
    (("left::right", "n"), ("left", "right::n")),
)
def test_symbol_identity_rejects_the_qualification_delimiter(
    scope: str,
    name: str,
) -> None:
    """Scope and name cannot alias another symbol's qualified wire identity."""

    with pytest.raises(ValueError, match="reserved for qualified symbol identities"):
        qm.resources.Symbol(scope, name)


def test_closed_expression_language_round_trips_predicates_and_binders() -> None:
    """Piecewise predicates and bounded binders remain portable typed data."""

    n = qm.resources.symbol("n", scope="example")
    expression = qm.resources.piecewise(
        (
            (
                qm.resources.comparison("less_equal", n, qm.resources.literal(4)),
                n + 1,
            ),
        ),
        otherwise=qm.resources.power(n, 2),
    )
    binder = qm.resources.Symbol("example", "i")
    bounded = qm.resources.bounded_sum(
        binder,
        qm.resources.literal(0),
        n,
        qm.resources.scoped_symbol(binder),
    )

    restored_expression = qm.resources.Expr.from_data(expression.to_data())
    restored_bounded = qm.resources.Expr.from_data(bounded.to_data())

    assert restored_expression == expression
    assert restored_expression.evaluate({"n": 3}).numeric() == 4
    assert restored_expression.evaluate({"n": 5}).numeric() == 25
    assert restored_bounded == bounded
    assert restored_bounded.evaluate({"n": 3, "i": 100}).numeric() == 6
    with pytest.raises(ValueError):
        qm.resources.Expr.from_data(expression.to_data(), max_nodes=2)


def test_resource_expression_numeric_semantics_execute_in_native_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact public expression evaluation crosses the required Rust boundary."""

    from qamomile._native import compiler

    original = compiler.evaluate_resource_expression
    payloads: list[bytes] = []

    def observed(payload: bytes) -> object:
        payloads.append(payload)
        return original(payload)

    monkeypatch.setattr(compiler, "evaluate_resource_expression", observed)
    expression = qm.resources.Expr(
        "add",
        qm.resources.Domain.RATIONAL,
        qm.resources.ONE,
        (
            qm.resources.literal(Fraction(1, 3)),
            qm.resources.literal(Fraction(1, 6)),
        ),
    )

    assert expression.numeric() == Fraction(1, 2)
    assert payloads


def test_native_resource_expression_evaluation_preserves_edge_semantics() -> None:
    """Rust owns large exact values, tri-state predicates, and bounded work."""

    huge = 10**200
    expression = qm.resources.Expr(
        "multiply",
        qm.resources.Domain.NATURAL,
        qm.resources.ONE,
        (qm.resources.literal(huge), qm.resources.literal(huge)),
    )
    unknown = qm.resources.symbol("n", scope="native-evaluation")
    predicate = qm.resources.conjunction(
        qm.resources.comparison("greater", unknown, qm.resources.literal(0)),
        qm.resources.comparison(
            "less", qm.resources.literal(1), qm.resources.literal(2)
        ),
    )
    binder = qm.resources.Symbol("native-evaluation", "i")
    oversized = qm.resources.bounded_sum(
        binder,
        qm.resources.literal(0),
        qm.resources.literal(100_000),
        qm.resources.literal(1),
    )

    assert expression.numeric() == huge * huge
    assert predicate.truth() is None
    assert oversized.numeric() is None


def test_native_resource_expression_division_by_zero_is_explicit() -> None:
    """Undefined native arithmetic keeps Python's familiar exception category."""

    expression = qm.resources.Expr(
        "floor_divide",
        qm.resources.Domain.INTEGER,
        qm.resources.ONE,
        (qm.resources.literal(1), qm.resources.literal(0)),
    )

    with pytest.raises(ZeroDivisionError, match="Q4232"):
        expression.numeric()


def test_native_resource_expression_power_has_a_numeric_complexity_budget() -> None:
    """A compact power node cannot request unbounded native big-integer work."""

    expression = qm.resources.Expr(
        "power",
        qm.resources.Domain.NATURAL,
        qm.resources.ONE,
        (qm.resources.literal(2), qm.resources.literal(1_000_001)),
    )

    with pytest.raises(ValueError, match="Q4233.*power-exponent"):
        expression.numeric()


def test_resource_expression_evaluation_fails_closed_without_native_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public evaluator cannot silently fall back to Python semantics."""

    from qamomile._native import compiler

    monkeypatch.delattr(compiler, "evaluate_resource_expression")

    with pytest.raises(RuntimeError, match="Q4234"):
        qm.resources.literal(1).numeric()


def test_resource_expression_substitution_executes_in_native_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symbol discovery and substitution cross their required Rust boundaries."""

    from qamomile._native import compiler

    original_substitute = compiler.substitute_resource_expression
    original_inspect = compiler.inspect_resource_expression_symbols
    substituted_payloads: list[bytes] = []
    inspected_payloads: list[bytes] = []

    def observed_substitution(payload: bytes, bindings: object) -> bytes:
        substituted_payloads.append(payload)
        return original_substitute(payload, bindings)

    def observed_inspection(payload: bytes) -> object:
        inspected_payloads.append(payload)
        return original_inspect(payload)

    monkeypatch.setattr(
        compiler,
        "substitute_resource_expression",
        observed_substitution,
    )
    monkeypatch.setattr(
        compiler,
        "inspect_resource_expression_symbols",
        observed_inspection,
    )
    left = qm.resources.symbol("left", scope="native-substitution")
    right = qm.resources.symbol("right", scope="native-substitution")
    result = left.substitute(
        {
            "native-substitution::left": right,
            "native-substitution::right": 7,
        }
    )

    assert result == right
    assert substituted_payloads
    assert inspected_payloads


def test_resource_expression_substitution_fails_closed_without_native_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoped substitution cannot silently return to Python tree traversal."""

    from qamomile._native import compiler

    expression = qm.resources.symbol("n", scope="native-substitution")
    inspect = compiler.inspect_resource_expression_symbols

    monkeypatch.delattr(compiler, "inspect_resource_expression_symbols")
    with pytest.raises(RuntimeError, match="Q4239"):
        expression.substitute({"native-substitution::n": 1})

    monkeypatch.setattr(
        compiler,
        "inspect_resource_expression_symbols",
        inspect,
        raising=False,
    )
    monkeypatch.delattr(compiler, "substitute_resource_expression")
    with pytest.raises(RuntimeError, match="Q4237"):
        expression.substitute({"native-substitution::n": 1})


def test_resource_symbol_inspection_has_a_nesting_budget() -> None:
    """Deep symbolic trees fail deterministically without Python recursion."""

    expression = qm.resources.symbol("n", scope="native-depth")
    for _ in range(1_025):
        expression = qm.resources.Expr(
            "absolute",
            qm.resources.Domain.NATURAL,
            arguments=(expression,),
        )

    with pytest.raises(ValueError, match="Q4236.*nesting-depth"):
        expression.substitute({})


def test_resource_expression_constructor_folding_executes_in_native_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unary and exact rounding constructors use the required Rust normalizer."""

    from qamomile._native import compiler

    original = compiler.construct_resource_expression
    constructors: list[str] = []

    def observed(
        kind: str,
        arguments: object,
        predicates: object,
        attributes: object,
        binder_unit: object,
        result_contract: object,
    ) -> bytes:
        constructors.append(kind)
        return original(
            kind,
            arguments,
            predicates,
            attributes,
            binder_unit,
            result_contract,
        )

    monkeypatch.setattr(compiler, "construct_resource_expression", observed)
    magnitude = 10**100

    assert (
        qm.resources.absolute(qm.resources.literal(-magnitude)).numeric() == magnitude
    )
    assert (
        qm.resources.floor(
            qm.resources.literal(Fraction(magnitude * 3 + 2, 3))
        ).numeric()
        == magnitude
    )
    assert constructors == ["absolute", "floor"]


def test_resource_literal_predicate_and_binder_contracts_are_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Literal, Boolean, and bounded introduction share the Rust owner."""

    from qamomile._native import compiler

    native_literal = compiler.construct_resource_literal
    native_predicate = compiler.construct_resource_predicate
    native_expression = compiler.construct_resource_expression
    calls: list[str] = []

    def observed_literal(payload: bytes, unit: str, domain: object) -> bytes:
        calls.append("literal")
        return native_literal(payload, unit, domain)

    def observed_predicate(
        kind: str,
        arguments: object,
        predicates: object,
    ) -> bytes:
        calls.append(f"predicate:{kind}")
        return native_predicate(kind, arguments, predicates)

    def observed_expression(
        kind: str,
        arguments: object,
        predicates: object,
        attributes: object,
        binder_unit: object,
        result_contract: object,
    ) -> bytes:
        calls.append(f"expression:{kind}")
        return native_expression(
            kind,
            arguments,
            predicates,
            attributes,
            binder_unit,
            result_contract,
        )

    monkeypatch.setattr(compiler, "construct_resource_literal", observed_literal)
    monkeypatch.setattr(compiler, "construct_resource_predicate", observed_predicate)
    monkeypatch.setattr(compiler, "construct_resource_expression", observed_expression)

    natural = qm.resources.literal(2)
    negative = qm.resources.literal(-1)
    rational = qm.resources.literal(2, domain=qm.resources.Domain.RATIONAL)
    predicate = qm.resources.comparison("greater", natural, negative)
    binder = qm.resources.Symbol("native-construction", "i")
    bounded = qm.resources.bounded_sum(
        binder,
        qm.resources.literal(0),
        natural,
        qm.resources.scoped_symbol(binder),
    )
    recurrence = qm.resources.recurrence(
        "example.native.recurrence",
        (natural,),
        domain=qm.resources.Domain.NATURAL,
        unit=qm.resources.SEMANTIC_OPS,
    )
    contracted = qm.resources.contracted_function(
        "example.native.contract",
        (natural,),
        domain=qm.resources.Domain.REAL,
        unit=qm.resources.SECONDS,
        version="2",
    )

    assert natural.domain is qm.resources.Domain.NATURAL
    assert negative.domain is qm.resources.Domain.INTEGER
    assert rational.value == Fraction(2)
    assert predicate.truth() is True
    assert bounded.numeric() == 3
    assert recurrence.attributes == (
        ("name", "example.native.recurrence"),
        ("version", "1"),
    )
    assert recurrence.unit == qm.resources.SEMANTIC_OPS
    assert contracted.attributes == (
        ("name", "example.native.contract"),
        ("version", "2"),
    )
    assert contracted.domain is qm.resources.Domain.REAL
    assert "predicate:greater" in calls
    assert "expression:bounded_sum" in calls
    assert "expression:recurrence" in calls
    assert "expression:contracted_function" in calls
    assert calls.count("literal") == 4

    with pytest.raises(ValueError, match="outside its declared domain"):
        qm.resources.literal(2, domain=qm.resources.Domain.PROBABILITY)
    with pytest.raises(ValueError, match="units must agree"):
        qm.resources.comparison(
            "equal",
            qm.resources.literal(1, qm.resources.SECONDS),
            natural,
        )
    with pytest.raises(ValueError, match="constructor shape"):
        qm.resources.conjunction()
    with pytest.raises(ValueError, match="attributes have an invalid shape"):
        qm.resources.recurrence(
            "",
            (),
            domain=qm.resources.Domain.NATURAL,
            unit=qm.resources.ONE,
        )
    with pytest.raises(ValueError, match="attributes have an invalid shape"):
        qm.resources.contracted_function(
            "example.native.contract",
            (),
            domain=qm.resources.Domain.NATURAL,
            unit=qm.resources.ONE,
            version="",
        )
    with pytest.raises(ValueError, match="binder must be integral"):
        qm.resources.bounded_sum(
            qm.resources.Symbol(
                "native-construction",
                "r",
                domain=qm.resources.Domain.REAL,
            ),
            qm.resources.literal(0),
            qm.resources.literal(1),
            natural,
        )


def test_resource_unit_and_symbol_introduction_are_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public unit and symbol declarations use the Rust contract owner."""

    from qamomile._native import compiler

    native_unit = compiler.validate_resource_unit
    native_symbol = compiler.construct_resource_symbol
    calls: list[str] = []

    def observed_unit(name: str) -> None:
        calls.append(f"unit:{name}")
        native_unit(name)

    def observed_symbol(
        scope: str,
        name: str,
        domain: int,
        unit: str,
        origin: str,
        constraints: object,
    ) -> bytes:
        calls.append(f"symbol:{scope}::{name}")
        return native_symbol(scope, name, domain, unit, origin, constraints)

    monkeypatch.setattr(compiler, "validate_resource_unit", observed_unit)
    monkeypatch.setattr(compiler, "construct_resource_symbol", observed_symbol)

    unit = qm.resources.Unit("native_unit")
    declaration = qm.resources.Symbol(
        "native-symbol",
        "n",
        qm.resources.Domain.INTEGER,
        unit,
        constraints=("n >= 0",),
    )
    expression = qm.resources.symbol(
        "m",
        scope="native-symbol",
        domain=qm.resources.Domain.INTEGER,
        unit=unit,
    )

    assert declaration.identity == "native-symbol::n"
    assert expression.symbol is not None
    assert expression.symbol.identity == "native-symbol::m"
    assert calls == [
        "unit:native_unit",
        "symbol:native-symbol::n",
        "symbol:native-symbol::m",
    ]
    with pytest.raises(ValueError, match="nonempty and whitespace-free"):
        qm.resources.Unit("bad unit")
    with pytest.raises(ValueError, match="reserved for qualified symbol identities"):
        qm.resources.Symbol("bad::scope", "n")


def test_native_symbolic_expression_projection_does_not_reconstruct_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Rust-owned symbolic DAG crosses Python as one linear projection."""

    from qamomile._native import compiler
    from qamomile.resources._rule_wire import (
        decode_resource_expression_graph,
        encode_resource_expression_graph,
    )
    from qamomile.resources.analysis import _decode_native_resource_expressions
    from qamomile.serialization import encode_analysis_request

    module = qm.build_source(STRUCTURAL_MATH_SOURCE)
    projection = compiler.inspect_symbolic_program_family_counts(
        encode_analysis_request(module, "structural_math")
    )
    graph_payload = encode_resource_expression_graph(
        qm.resources.add(
            qm.resources.symbol("n", scope="native-projection"),
            qm.resources.literal(2),
        )
    )

    def reject(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("native expression projection re-entered construction")

    monkeypatch.setattr(compiler, "construct_resource_expression", reject)
    monkeypatch.setattr(compiler, "construct_resource_literal", reject)
    monkeypatch.setattr(compiler, "construct_resource_predicate", reject)
    monkeypatch.setattr(compiler, "construct_resource_symbol", reject)
    monkeypatch.setattr(compiler, "validate_resource_unit", reject)
    monkeypatch.setattr(compiler, "normalize_resource_expression", reject)
    monkeypatch.setattr(qm.resources.Expr, "__post_init__", reject)
    monkeypatch.setattr(qm.resources.Unit, "__init__", reject)
    monkeypatch.setattr(qm.resources.Symbol, "__init__", reject)

    expressions = _decode_native_resource_expressions(projection.expressions)
    graph = decode_resource_expression_graph(graph_payload)

    assert len(expressions) == len(projection.expressions)
    assert isinstance(graph, qm.resources.Expr)
    assert any(
        argument.symbol is not None
        and argument.symbol.identity == "native-projection::n"
        for argument in graph.arguments
    )
    assert any(
        isinstance(expression, qm.resources.Expr)
        and expression.kind == "contracted_function"
        for expression in expressions
    )


def test_resource_expression_constructor_folding_fails_closed_without_native_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource constructors cannot silently restore Python semantics."""

    from qamomile._native import compiler

    native_construct = compiler.construct_resource_expression
    monkeypatch.delattr(compiler, "construct_resource_expression")

    with pytest.raises(RuntimeError, match="Q4238"):
        qm.resources.square_root(qm.resources.literal(4))

    monkeypatch.setattr(
        compiler,
        "construct_resource_expression",
        lambda _kind, _arguments, _predicates, _attributes, _binder_unit, _result_contract: (
            object()
        ),
        raising=False,
    )
    with pytest.raises(TypeError, match="Q4238"):
        qm.resources.square_root(qm.resources.literal(4))

    monkeypatch.setattr(compiler, "construct_resource_expression", native_construct)

    native_literal = compiler.construct_resource_literal
    monkeypatch.delattr(compiler, "construct_resource_literal")
    with pytest.raises(RuntimeError, match="Q4238"):
        qm.resources.literal(4)
    monkeypatch.setattr(
        compiler,
        "construct_resource_literal",
        lambda _payload, _unit, _domain: object(),
        raising=False,
    )
    with pytest.raises(TypeError, match="Q4238"):
        qm.resources.literal(4)
    monkeypatch.setattr(compiler, "construct_resource_literal", native_literal)

    native_predicate = compiler.construct_resource_predicate
    monkeypatch.delattr(compiler, "construct_resource_predicate")
    with pytest.raises(RuntimeError, match="Q4238"):
        qm.resources.Predicate("true")
    monkeypatch.setattr(
        compiler,
        "construct_resource_predicate",
        lambda _kind, _arguments, _predicates: object(),
        raising=False,
    )
    with pytest.raises(TypeError, match="Q4238"):
        qm.resources.Predicate("true")
    monkeypatch.setattr(compiler, "construct_resource_predicate", native_predicate)

    native_unit = compiler.validate_resource_unit
    monkeypatch.delattr(compiler, "validate_resource_unit")
    with pytest.raises(RuntimeError, match="Q4238"):
        qm.resources.Unit("missing_native")
    monkeypatch.setattr(
        compiler,
        "validate_resource_unit",
        lambda _unit: object(),
        raising=False,
    )
    with pytest.raises(TypeError, match="Q4238"):
        qm.resources.Unit("invalid_abi")
    monkeypatch.setattr(compiler, "validate_resource_unit", native_unit)

    native_symbol = compiler.construct_resource_symbol
    monkeypatch.delattr(compiler, "construct_resource_symbol")
    with pytest.raises(RuntimeError, match="Q4238"):
        qm.resources.Symbol("missing-native", "n")
    monkeypatch.setattr(
        compiler,
        "construct_resource_symbol",
        lambda _scope, _name, _domain, _unit, _origin, _constraints: object(),
        raising=False,
    )
    with pytest.raises(TypeError, match="Q4238"):
        qm.resources.symbol("n", scope="invalid-abi")
    monkeypatch.setattr(compiler, "construct_resource_symbol", native_symbol)


def test_resource_expression_associative_normalization_is_native_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial associative expressions flatten and fold in the Rust DAG engine."""

    from qamomile._native import compiler

    original = compiler.construct_resource_expression
    normalized_kinds: list[str] = []

    def observed(
        kind: str,
        arguments: object,
        predicates: object,
        attributes: object,
        binder_unit: object,
        result_contract: object,
    ) -> bytes:
        normalized_kinds.append(kind)
        return original(
            kind,
            arguments,
            predicates,
            attributes,
            binder_unit,
            result_contract,
        )

    monkeypatch.setattr(compiler, "construct_resource_expression", observed)
    operations = qm.resources.SEMANTIC_OPS
    n = qm.resources.symbol(
        "n",
        scope="native-normalization",
        domain=qm.resources.Domain.INTEGER,
        unit=operations,
    )
    total = qm.resources.add(
        qm.resources.add(n, qm.resources.literal(2, operations)),
        qm.resources.literal(3, operations),
        qm.resources.literal(0, operations),
    )
    product = qm.resources.multiply(
        qm.resources.multiply(n, qm.resources.literal(2)),
        qm.resources.literal(3),
        qm.resources.literal(1),
    )
    zero = qm.resources.multiply(n, qm.resources.literal(0))

    assert total.kind == "add"
    assert len(total.arguments) == 2
    assert total.evaluate({"native-normalization::n": 4}).numeric() == 9
    assert product.kind == "multiply"
    assert len(product.arguments) == 2
    assert product.evaluate({"native-normalization::n": 4}).numeric() == 24
    assert zero.numeric() == 0
    assert zero.unit == operations
    assert total.domain is qm.resources.Domain.INTEGER
    assert product.domain is qm.resources.Domain.INTEGER
    with pytest.raises(ValueError, match="different units"):
        qm.resources.add(n, qm.resources.literal(1, qm.resources.SECONDS))
    assert normalized_kinds == [
        "add",
        "add",
        "multiply",
        "multiply",
        "multiply",
        "add",
    ]


def test_resource_expression_choice_and_power_normalization_is_native() -> None:
    """Extrema, powers, and literal predicate pruning share native semantics."""

    n = qm.resources.symbol(
        "n",
        scope="native-choice",
        domain=qm.resources.Domain.INTEGER,
    )
    maximum = qm.resources.maximum(n, n, qm.resources.literal(2))
    reciprocal = qm.resources.power(qm.resources.literal(2), -3)
    selected = qm.resources.piecewise(
        (
            (qm.resources.Predicate("false"), qm.resources.literal(99)),
            (
                qm.resources.comparison("greater", n, qm.resources.literal(0)),
                qm.resources.literal(1),
            ),
            (qm.resources.Predicate("true"), qm.resources.literal(2)),
            (qm.resources.Predicate("true"), qm.resources.literal(3)),
        ),
        otherwise=qm.resources.literal(4),
    )

    assert maximum.kind == "maximum"
    assert len(maximum.arguments) == 2
    assert maximum.evaluate({"native-choice::n": 5}).numeric() == 5
    assert reciprocal.numeric() == Fraction(1, 8)
    assert reciprocal.domain is qm.resources.Domain.RATIONAL
    dimensioned = qm.resources.symbol(
        "work",
        scope="native-choice",
        unit=qm.resources.SEMANTIC_OPS,
    )
    with pytest.raises(ValueError, match="dimensioned powers"):
        qm.resources.power(dimensioned, Fraction(1, 2))
    assert qm.resources.power(n, 0).numeric() == 1
    assert qm.resources.power(n, 1) == n
    assert selected.kind == "piecewise"
    assert len(selected.predicates) == 1
    assert selected.evaluate({"native-choice::n": 1}).numeric() == 1
    assert selected.evaluate({"native-choice::n": -1}).numeric() == 2


def test_resource_expression_from_data_validates_once_in_rust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each portable tree validates once without reconstructing its nodes."""

    from qamomile._native import compiler

    original = compiler.normalize_resource_expression
    constructors = {
        name: getattr(compiler, name)
        for name in (
            "construct_resource_expression",
            "construct_resource_literal",
            "construct_resource_predicate",
        )
    }
    calls = 0

    def observed(payload: bytes) -> bytes:
        nonlocal calls
        calls += 1
        return original(payload)

    monkeypatch.setattr(compiler, "normalize_resource_expression", observed)

    def reject(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("from_data reconstructed a partial native graph")

    for name in constructors:
        monkeypatch.setattr(compiler, name, reject)
    data = {
        "kind": "add",
        "domain": "integer",
        "unit": "1",
        "arguments": [
            {"kind": "literal", "domain": "natural", "unit": "1", "value": 3},
            {
                "kind": "add",
                "domain": "integer",
                "unit": "1",
                "arguments": [
                    {
                        "kind": "symbol",
                        "domain": "integer",
                        "unit": "1",
                        "symbol": {
                            "scope": "native-from-data",
                            "name": "n",
                            "domain": "integer",
                            "unit": "1",
                            "origin": "program_parameter",
                            "constraints": [],
                        },
                    },
                    {
                        "kind": "literal",
                        "domain": "natural",
                        "unit": "1",
                        "value": 2,
                    },
                ],
            },
        ],
    }

    expression = qm.resources.Expr.from_data(data)
    predicate = qm.resources.Predicate.from_data(
        {
            "kind": "greater",
            "arguments": [
                {"kind": "literal", "domain": "natural", "unit": "1", "value": 2},
                {"kind": "literal", "domain": "natural", "unit": "1", "value": 1},
            ],
        }
    )

    assert calls == 2
    assert expression.kind == "add"
    assert len(expression.arguments) == 2
    assert expression.arguments[1].kind == "add"
    for name, constructor in constructors.items():
        monkeypatch.setattr(compiler, name, constructor)
    assert expression.evaluate({"native-from-data::n": 4}).numeric() == 9
    assert predicate.truth() is True
    with pytest.raises(ValueError, match="Q4235"):
        qm.resources.Expr.from_data({"kind": "add", "domain": "integer", "unit": "1"})
    monkeypatch.delattr(compiler, "normalize_resource_expression")
    with pytest.raises(RuntimeError, match="Q4238"):
        qm.resources.Predicate.from_data({"kind": "true"})


def test_resource_expression_from_data_is_iterative_and_bounded() -> None:
    """Host records reject cycles and cross the native nesting limit safely."""

    cyclic: dict[str, object] = {
        "kind": "absolute",
        "domain": "natural",
        "unit": "1",
    }
    cyclic["arguments"] = [cyclic]
    with pytest.raises(ValueError, match="cyclic resource expression"):
        qm.resources.Expr.from_data(cyclic)

    data: dict[str, object] = {
        "kind": "literal",
        "domain": "natural",
        "unit": "1",
        "value": 1,
    }
    for _ in range(1_025):
        data = {
            "kind": "absolute",
            "domain": "natural",
            "unit": "1",
            "arguments": [data],
        }
    with pytest.raises(ValueError, match="Q4236.*nesting-depth"):
        qm.resources.Expr.from_data(data, max_nodes=2_000)


def test_endpoint_submit_returns_a_typed_job() -> None:
    """Submission is explicit while execute remains a visible wait convenience."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    artifact = qm.compile(program, to=qm.Reference())
    job = qm.submit(artifact, on=qm.Reference(), shots=8, seed=3)

    assert job.status is qm.JobStatus.COMPLETED
    assert isinstance(job.result(), qm.SampleResult)


def test_callable_contract_is_derived_directly_from_the_body() -> None:
    """Callable facts need no second property-judgment representation."""

    function = qm.build_source(BELL_SOURCE).function("bell")

    assert function.kind is CallableKind.INSTRUMENT
    assert Effect.MEASUREMENT in function.effects
    assert Capability.ADJOINTABLE not in function.capabilities


def test_bodyless_function_rules_compose_from_function_values() -> None:
    """Opaque call costs remain symbolic, safe, and callable-oriented."""

    @qm.qkernel(kind="unitary")
    def oracle(state: qm.Q[qm.Bits[3]], rounds: qm.Int) -> qm.Q[qm.Bits[3]]: ...

    @qm.qkernel
    def algorithm(rounds: qm.Const[qm.Int]) -> qm.Bits[3]:
        state = qm.qubits(3)
        for _ in range(rounds):
            state = oracle(state, rounds)
        return qm.measure(state)

    rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    gate_rule = qm.resources.ResourceRule(
        identity="example.oracle.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.multiply(qm.resources.literal(2), rounds),
        assumptions=("rounds >= 0",),
        validity_domain=("rounds is an integer",),
        provenance="example.oracle.contract.v1",
    )
    program = qm.specialize(algorithm, rounds=3)
    study = qm.resources.logical(program, rules=gate_rule, strict=True)

    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (18, 18)
    assert study.metric(
        qm.resources.algorithmic_call_count(oracle.symbol)
    ).numeric_enclosure() == (3, 3)
    assert study.metric(
        qm.resources.algorithmic_query_count(oracle.symbol)
    ).numeric_enclosure() == (3, 3)
    assert gate_rule.provenance in {
        item.reference for item in study.metric(qm.resources.LOGICAL_GATES).evidence
    }
    assert (
        qm.resources.ResourceStudy.from_bytes(study.to_bytes()).digest == study.digest
    )


def test_scheduling_footprints_require_distinct_quantum_parameters() -> None:
    """Opaque timing contracts reject ambiguous or classical parameter names."""

    with pytest.raises(ValueError, match="distinct nonempty parameter names"):
        qm.resources.SchedulingFootprint(("state", "state"))

    @qm.qkernel(kind="unitary")
    def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]: ...

    rule = qm.resources.ResourceRule(
        identity="example.invalid-footprint.span.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_SPAN,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(1),
        provenance="example.invalid-footprint.contract.v1",
        scheduling_footprint=qm.resources.SchedulingFootprint(("rounds",)),
    )

    with pytest.raises(ValueError, match="non-quantum parameter"):
        qm.resources.logical(oracle, rules=rule)


def test_bodyless_callable_is_never_accidentally_executable() -> None:
    """An external semantic declaration requires realization before execution."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = oracle(state)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="external_example")
    program = module.specialize("algorithm")

    assert module.function("oracle").external
    assert Requirement.EXTERNAL_CALL in module.function("algorithm").requirements
    with pytest.raises(qm.ArtifactError, match="external_call"):
        qm.compile(program, to=qm.Reference())
    with pytest.raises(ValueError, match="external direct callable"):
        qm.resources.logical(program, strict=True)


def test_resource_rules_reject_unbound_formula_symbols() -> None:
    """A misspelled callable parameter cannot silently poison an estimate."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...
"""
    family = qm.build_source(source, module_name="rule_scope")["oracle"]
    invalid = qm.resources.ResourceRule(
        identity="example.invalid.v1",
        callable=family.function,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.symbol(
            "typo", scope=family.function.symbol, domain=qm.resources.Domain.NATURAL
        ),
        provenance="example.invalid.contract.v1",
    )

    with pytest.raises(ValueError, match="unbound symbol"):
        qm.resources.logical(family, rules=invalid)


def test_resource_rules_enforce_metric_domains_and_specialized_values() -> None:
    """Rule formulas and their call-site substitutions stay nonnegative naturals."""

    @qm.qkernel(kind="unitary")
    def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]: ...

    @qm.qkernel
    def algorithm(rounds: qm.Const[qm.Int]) -> qm.Bits[1]:
        state = qm.qubits(1)
        state = oracle(state, rounds)
        return qm.measure(state)

    probability_rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.PROBABILITY,
    )
    with pytest.raises(ValueError, match="is not contained in metric domain 'natural'"):
        qm.resources.ResourceRule(
            identity="example.oracle.invalid-domain.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=probability_rounds,
            provenance="example.oracle.invalid-domain.contract.v1",
        )
    with pytest.raises(ValueError, match="must be nonnegative"):
        qm.resources.ResourceRule(
            identity="example.oracle.negative.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(
                -1,
                domain=qm.resources.Domain.INTEGER,
            ),
            provenance="example.oracle.negative.contract.v1",
        )

    integer_rounds = qm.resources.rule_parameter(oracle, "rounds")
    with pytest.raises(ValueError, match="is not contained in metric domain 'natural'"):
        qm.resources.ResourceRule(
            identity="example.oracle.integer-domain.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=integer_rounds,
            provenance="example.oracle.integer-domain.contract.v1",
        )

    natural_rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    rule = qm.resources.ResourceRule(
        identity="example.oracle.natural-domain.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=natural_rounds,
        provenance="example.oracle.natural-domain.contract.v1",
    )
    program = qm.specialize(algorithm, rounds=-1)

    with pytest.raises(ValueError, match="must be nonnegative"):
        qm.resources.logical(program, rules=rule)


def test_resource_rule_request_is_a_canonical_typed_expression_dag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native rule requests preserve predicates, binders, and shared nodes."""

    import hashlib

    from qamomile.proto import resource_analysis_pb2 as wire
    from qamomile.resources import _rule_wire as rule_wire
    from qamomile.serialization import encode_analysis_request

    native_seal = rule_wire.native.seal_resource_analysis_request
    seal_calls = 0

    def record_native_seal(unsealed: bytes) -> bytes:
        """Record one authoritative native resource-request seal."""

        nonlocal seal_calls
        seal_calls += 1
        return native_seal(unsealed)

    monkeypatch.setattr(
        rule_wire.native,
        "seal_resource_analysis_request",
        record_native_seal,
    )

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm(rounds: qm.Const[qm.Int]) -> qm.Bits[1]:
    return qm.measure(oracle(qm.qubits(1), rounds))
"""
    module = qm.build_source(source, module_name="typed_rule_request")
    oracle = module.function("oracle")
    rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    binder = qm.resources.Symbol(
        "example.rule",
        "index",
        qm.resources.Domain.NATURAL,
        origin="bounded_rule",
    )
    repeated = qm.resources.multiply(qm.resources.literal(2), rounds)
    expression = qm.resources.piecewise(
        (
            (
                qm.resources.comparison(
                    "greater",
                    rounds,
                    qm.resources.literal(0),
                ),
                qm.resources.add(
                    repeated,
                    qm.resources.bounded_sum(
                        binder,
                        qm.resources.literal(0),
                        rounds,
                        qm.resources.literal(1),
                    ),
                ),
            ),
        ),
        otherwise=qm.resources.literal(0),
    )
    rule = qm.resources.ResourceRule(
        identity="example.typed-rule.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=expression,
        modes=frozenset(
            {
                qm.resources.TransformMode.ADJOINT,
                qm.resources.TransformMode.CONTROLLED_ADJOINT,
                qm.resources.TransformMode.DIRECT,
            }
        ),
        provenance="example.typed-rule.contract.v1",
    )
    payload = rule_wire.encode_resource_analysis_request(
        encode_analysis_request(module, "algorithm"),
        (rule,),
    )
    request = wire.ResourceAnalysisRequestProto.FromString(payload)
    observed = bytes(request.request_digest)
    request.request_digest = b""

    assert (
        observed
        == hashlib.sha256(request.SerializeToString(deterministic=True)).digest()
    )
    assert seal_calls == 1
    assert list(request.rules[0].modes) == [
        wire.RESOURCE_DIRECT,
        wire.RESOURCE_ADJOINT,
        wire.RESOURCE_CONTROLLED_ADJOINT,
    ]
    graph = request.rules[0].expression
    assert graph.nodes[graph.root].expression.kind == "piecewise"
    assert any(node.WhichOneof("node") == "predicate" for node in graph.nodes)
    assert any(
        node.WhichOneof("node") == "expression"
        and node.expression.kind == "bounded_sum"
        for node in graph.nodes
    )
    for identifier, node in enumerate(graph.nodes):
        selected = node.WhichOneof("node")
        value = getattr(node, selected)
        assert all(reference < identifier for reference in value.arguments)
        assert all(reference < identifier for reference in value.predicates)


def test_resource_request_sealing_fails_closed_without_native_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resource contracts cannot fall back to a Python persistence policy."""

    from qamomile.errors import CapabilityError
    from qamomile.resources import _rule_wire as rule_wire
    from qamomile.serialization import encode_analysis_request

    module = qm.build_source(SYMBOLIC_LOOP_SOURCE)
    family = encode_analysis_request(module, "repeat")
    monkeypatch.setattr(rule_wire.native, "seal_resource_analysis_request", None)

    with pytest.raises(CapabilityError, match="Q4214:.*seal_resource_analysis_request"):
        rule_wire.encode_resource_analysis_request(family, ())


def test_resource_request_sealing_rejects_malformed_native_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Python bridge checks the native sealer's result ABI."""

    from qamomile.resources import _rule_wire as rule_wire
    from qamomile.serialization import encode_analysis_request

    module = qm.build_source(SYMBOLIC_LOOP_SOURCE)
    family = encode_analysis_request(module, "repeat")
    monkeypatch.setattr(
        rule_wire.native,
        "seal_resource_analysis_request",
        lambda _payload: object(),
    )

    with pytest.raises(TypeError, match="Q4237:.*malformed data"):
        rule_wire.encode_resource_analysis_request(family, ())


def test_invalid_substituted_rule_contribution_cannot_be_masked_by_work() -> None:
    """Each call-site rule is checked before later positive work is aggregated."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]], rounds: qm.Int) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm(rounds: qm.Const[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    state = oracle(state, rounds)
    state = qm.h(state)
    state = qm.h(state)
    state = qm.h(state)
    state = qm.h(state)
    state = qm.h(state)
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="masked_rule_contribution")
    oracle = module.function("oracle")
    rounds = qm.resources.rule_parameter(
        oracle,
        "rounds",
        domain=qm.resources.Domain.NATURAL,
        constraints=("rounds >= 0",),
    )
    work_rule = qm.resources.ResourceRule(
        identity="example.masked-rule.work.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_WORK,
        relation=qm.resources.Relation.EQUALITY,
        expression=rounds,
        provenance="example.masked-rule.work.contract.v1",
    )
    gate_rule = qm.resources.ResourceRule(
        identity="example.masked-rule.gates.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(0),
        provenance="example.masked-rule.gates.contract.v1",
    )
    program = module.specialize("algorithm", rounds=-1)

    with pytest.raises(ValueError, match="must be nonnegative"):
        qm.resources.logical(program, rules=(work_rule, gate_rule))


def test_external_adjoint_requires_explicit_transform_coverage() -> None:
    """Automatic uncomputation cannot assume an opaque inverse has equal cost."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def clean() -> qm.Unit:
    state = qm.qubits(1)
    state = oracle(state)
    return None
"""
    module = qm.build_source(source, module_name="external_cleanup")
    program = module.specialize("clean")
    oracle = module.function("oracle")
    direct_only = qm.resources.ResourceRule(
        identity="example.oracle.direct.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        provenance="example.oracle.direct.contract.v1",
    )
    with pytest.raises(ValueError, match="external adjoint callable"):
        qm.resources.logical(program, rules=direct_only, strict=True)

    both = qm.resources.ResourceRule(
        identity="example.oracle.all-transforms.v1",
        callable=oracle,
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.literal(4),
        modes=frozenset(
            {
                qm.resources.TransformMode.DIRECT,
                qm.resources.TransformMode.ADJOINT,
            }
        ),
        provenance="example.oracle.all-transforms.contract.v1",
    )
    study = qm.resources.logical(program, rules=both, strict=True)

    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (8, 8)
    assert study.metric(
        qm.resources.algorithmic_query_count(oracle.symbol)
    ).numeric_enclosure() == (2, 2)


def test_body_backed_transforms_select_nested_external_modes() -> None:
    """Concrete wrappers preserve controlled and adjoint rule distinctions."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def wrapper(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return oracle(state)

@qm.qkernel(kind="unitary")
def outer(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return wrapper(state)

@qm.qkernel
def controlled() -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[1]]]:
    control = qm.qubits(1)
    target = qm.qubits(1)
    control, target = qm.control(wrapper, control, target)
    return qm.measure(control), qm.measure(target)

@qm.qkernel
def selected() -> tuple[qm.Shot[qm.Bits[1]], qm.Shot[qm.Bits[1]]]:
    index = qm.qubits(1)
    target = qm.qubits(1)
    index, target = qm.multiplex(index, (wrapper, outer), target)
    return qm.measure(index), qm.measure(target)

@qm.qkernel
def reversed_call() -> qm.Shot[qm.Bits[1]]:
    state = qm.adjoint(wrapper, qm.qubits(1))
    return qm.measure(state)
"""
    module = qm.build_source(source, module_name="nested_external_transforms")
    oracle = module.function("oracle")
    rules = tuple(
        qm.resources.ResourceRule(
            identity=f"example.nested-transform.{mode.value}.v1",
            callable=oracle,
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(gates),
            modes=frozenset({mode}),
            provenance=f"example.nested-transform.{mode.value}.contract.v1",
        )
        for mode, gates in (
            (qm.resources.TransformMode.DIRECT, 2),
            (qm.resources.TransformMode.CONTROLLED, 10),
            (qm.resources.TransformMode.ADJOINT, 20),
        )
    )

    controlled = qm.resources.logical(
        module["controlled"],
        rules=rules,
        strict=True,
    )
    assert controlled.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
        10,
        10,
    )
    assert "native-transform-mode-composition" in controlled.derivations

    selected = qm.resources.logical(
        module["selected"],
        rules=rules,
        strict=True,
    )
    assert selected.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
        20,
        20,
    )
    assert "native-transform-mode-composition" in selected.derivations

    reversed_call = qm.resources.logical(
        module["reversed_call"],
        rules=rules,
        strict=True,
    )
    assert reversed_call.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (
        20,
        20,
    )


def test_controlled_structural_wrapper_preserves_exact_external_modes() -> None:
    """Factored owner paths compose exact controlled rules without flattening."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def left_oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def right_oracle(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:
    ...

@qm.qkernel
def wrapper(state: qm.Q[qm.Bits[3]]) -> qm.Q[qm.Bits[3]]:
    left, right = qm.split_at(state, 1)
    left = left_oracle(left)
    right = right_oracle(right)
    return qm.concat(left, right)

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[3]]]:
    control = qm.qubits(1)
    target = qm.qubits(3)
    return qm.control(wrapper, control, target)
"""
    module = qm.build_source(source, module_name="structural_control")
    rules = tuple(
        qm.resources.ResourceRule(
            identity=f"example.structural-control.{name}.v1",
            callable=module.function(name),
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(gates),
            modes=frozenset({qm.resources.TransformMode.CONTROLLED}),
            provenance=f"example.structural-control.{name}.contract.v1",
            scheduling_footprint=qm.resources.SchedulingFootprint(("state",)),
        )
        for name, gates in (("left_oracle", 2), ("right_oracle", 3))
    )

    study = qm.resources.logical(module["algorithm"], rules=rules, strict=True)

    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (5, 5)
    assert study.metric(qm.resources.LOGICAL_GATE_DEPTH).numeric_enclosure() == (
        None,
        5,
    )


def test_controlled_wrapper_with_primitive_work_remains_unknown() -> None:
    """A non-forwarding controlled body cannot borrow nested external rules."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def first(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def second(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel(kind="unitary")
def wrapper(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    state = first(state)
    state = qm.x(state)
    return second(state)

@qm.qkernel
def algorithm() -> tuple[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]]:
    control = qm.qubits(1)
    target = qm.qubits(1)
    return qm.control(wrapper, control, target)
"""
    module = qm.build_source(source, module_name="non_forwarding_control")
    rules = tuple(
        qm.resources.ResourceRule(
            identity=f"example.non-forwarding-control.{name}.v1",
            callable=module.function(name),
            metric=qm.resources.LOGICAL_GATES,
            relation=qm.resources.Relation.EQUALITY,
            expression=qm.resources.literal(2),
            modes=frozenset({qm.resources.TransformMode.CONTROLLED}),
            provenance=f"example.non-forwarding-control.{name}.contract.v1",
        )
        for name in ("first", "second")
    )

    study = qm.resources.logical(module["algorithm"], rules=rules)
    gates = study.metric(qm.resources.LOGICAL_GATES)

    assert gates.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
    assert gates.numeric_enclosure() == (5, None)

    with pytest.raises(
        ValueError,
        match="body-backed controlled transform.*has no exact gate-family",
    ):
        qm.resources.logical(module["algorithm"], rules=rules, strict=True)


def test_count_lower_bound_is_not_reused_as_a_depth_lower_bound() -> None:
    """Gate count only bounds depth from above unless timing proves more."""

    source = """
import qamomile as qm

@qm.qkernel(kind="unitary")
def oracle(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    ...

@qm.qkernel
def algorithm() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(oracle(qm.qubits(1)))
"""
    module = qm.build_source(source, module_name="count_lower_depth")
    rule = qm.resources.ResourceRule(
        identity="example.count-lower.gates.v1",
        callable=module.function("oracle"),
        metric=qm.resources.LOGICAL_GATES,
        relation=qm.resources.Relation.LOWER_BOUND,
        expression=qm.resources.literal(100),
        provenance="example.count-lower.gates.contract.v1",
    )

    study = qm.resources.logical(module["algorithm"], rules=rule)
    depth = study.metric(qm.resources.LOGICAL_GATE_DEPTH)

    assert depth.relation is qm.resources.Relation.PARTIAL_ENCLOSURE
    assert depth.numeric_enclosure() == (0, None)
    assert any(
        "without a finite upper endpoint" in reason
        for reason in depth.unknown_contributors
    )
