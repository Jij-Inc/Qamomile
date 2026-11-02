"""Deterministic non-simulator size ladders for the Qamomile compiler."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import qamomile as qm
from qamomile._compiler.frontend import SourceUnit, parse_source
from qamomile._compiler.hir_wire import encode_hir
from qamomile._compiler.lowering import lower_hir
from qamomile._native import compiler as native
from qamomile.ir import Module, freeze_value
from qamomile.serialization import encode_analysis_request, encode_selection_request


@dataclass(frozen=True, slots=True)
class _LexicalScale:
    """Retain deterministic sizes observed at each compiler boundary."""

    repeated_gates: int
    source_bytes: int
    ast_nodes: int
    hir_bytes: int
    hir_expressions: int
    hir_statements: int
    hir_preflight_nodes: int
    hir_validation_visits: int
    hir_normalization_visits: int
    hir_ssa_lowering_visits: int
    module_bytes: int
    module_operations: int
    program_bytes: int
    program_operations: int
    persistence_input_bytes: int
    persistence_nodes: int
    persistence_encoded_bytes: int
    persistence_encode_passes: int
    qcore_verification_visits: int
    specialization_visits: int
    exact_analysis_visits: int
    resource_work: int
    standard_optimization_visits: int
    circuit_lowering_visits: int
    circuit_operations: int
    circuit_scalars: int


@dataclass(frozen=True, slots=True)
class _CallScale:
    """Retain sizes for one progressively deeper static call graph."""

    depth: int
    source_bytes: int
    module_bytes: int
    program_bytes: int
    analysis_bytes: int
    hir_preflight_nodes: int
    hir_validation_visits: int
    hir_normalization_visits: int
    hir_ssa_lowering_visits: int
    persistence_input_bytes: int
    persistence_nodes: int
    persistence_encoded_bytes: int
    persistence_encode_passes: int
    qcore_verification_visits: int
    control_dependence_steps: int
    contract_derivation_steps: int
    function_count: int
    operation_count: int
    call_sites: int
    specialization_visits: int
    exact_analysis_visits: int
    symbolic_analysis_visits: int
    circuit_lowering_visits: int
    resource_expressions: int
    resource_facts: int
    circuit_operations: int


def _lexical_source(repeated_gates: int) -> str:
    """Build one source program whose only growing dimension is gate count."""

    gates = "".join("    state = qm.h(state)\n" for _ in range(repeated_gates))
    return (
        "import qamomile as qm\n\n"
        "@qm.qkernel\n"
        "def sample() -> qm.Bits[1]:\n"
        "    state = qm.qubits(1)\n"
        f"{gates}"
        "    return qm.measure(state)\n"
    )


def _call_chain_source(depth: int) -> str:
    """Build one static call chain with constant quantum denotation."""

    declarations = [
        "@qm.qkernel\n"
        "def leaf(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:\n"
        "    return qm.h(state)\n"
    ]
    callee = "leaf"
    for index in range(depth):
        name = f"level_{index}"
        declarations.append(
            "\n@qm.qkernel\n"
            f"def {name}(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:\n"
            f"    return {callee}(state)\n"
        )
        callee = name
    declarations.append(
        "\n@qm.qkernel\n"
        "def sample() -> qm.Bits[1]:\n"
        "    state = qm.qubits(1)\n"
        f"    state = {callee}(state)\n"
        "    return qm.measure(state)\n"
    )
    return "import qamomile as qm\n\n" + "".join(declarations)


def _const_parameter_source(parameter_count: int) -> str:
    """Build a pure-classical entrypoint with many independent bindings."""

    parameters = ", ".join(
        f"p{index}: qm.Const[qm.Int]" for index in range(parameter_count)
    )
    return (
        "import qamomile as qm\n\n"
        "@qm.qkernel\n"
        f"def sample({parameters}) -> qm.Int:\n"
        "    return p0\n"
    )


def _phase_lifting_source(repetitions: int, *, helper_backed: bool) -> str:
    """Build repeated uses of one simple or helper-backed phase predicate."""

    helper = (
        "@qm.qkernel\n"
        "def first(bits: qm.Bits[1]) -> qm.Bool:\n"
        "    return bool(bits[0])\n\n"
        "@qm.qkernel\n"
        "def marked(bits: qm.Bits[1]) -> qm.Bool:\n"
        "    return not (not first(bits))\n\n"
        if helper_backed
        else "@qm.qkernel\n"
        "def marked(bits: qm.Bits[1]) -> qm.Bool:\n"
        "    return bool(bits[0])\n\n"
    )
    applications = "".join(
        "    state = qm.phase_by(marked, state)\n" for _ in range(repetitions)
    )
    return (
        "import qamomile as qm\n\n"
        f"{helper}"
        "@qm.qkernel\n"
        "def sample() -> qm.Q[qm.Bits[1]]:\n"
        "    state = qm.qubits(1)\n"
        f"{applications}"
        "    return state\n"
    )


def _specialization_visits(
    module: Module,
    **bindings: object,
) -> int:
    """Measure native specialization without changing the public lifecycle."""

    frozen = tuple(
        (name, freeze_value(value)) for name, value in sorted(bindings.items())
    )
    request = encode_selection_request(module, "sample", frozen)
    return native.inspect_program_preparation_operation_visits(request)


def _observe_lexical_scale(repeated_gates: int) -> _LexicalScale:
    """Run the production boundaries without launching a simulator."""

    source = _lexical_source(repeated_gates)
    unit = SourceUnit.from_text(
        source,
        display_name="compiler_scalability.py",
        module_name="compiler_scalability",
    )
    hir = parse_source(unit)
    hir_payload = encode_hir(hir)
    hir_header = native.inspect_hir(hir_payload)
    hir_validation = native.inspect_hir_validation_work(hir_payload)
    hir_normalization = native.inspect_hir_normalization_work(hir_payload)
    hir_ssa = native.inspect_hir_ssa_lowering_work(hir_payload)
    module = lower_hir(hir)
    module_payload = module.to_bytes()
    module_header = native.inspect_module(module_payload)
    module_persistence = native.inspect_qcore_persistence_work(module_payload)
    module_verification = native.inspect_module_verification_work(module_payload)
    specialization_visits = _specialization_visits(module)
    program = module.specialize("sample")
    program_payload = program.to_bytes()
    program_header = native.inspect_program(program_payload)
    program_persistence = native.inspect_qcore_persistence_work(program_payload)
    program_verification = native.inspect_program_verification_work(program_payload)
    structure = native.inspect_structure(program_payload)
    resources = native.inspect_exact_logical_counts(program_payload)
    circuit = native.inspect_static_circuit(program_payload)
    standard_optimization_visits = (
        native.inspect_standard_optimization_operation_visits(program_payload)
    )
    circuit_lowering_visits = native.inspect_static_circuit_operation_visits(
        program_payload
    )

    assert qm.Module.from_bytes(module_payload).to_bytes() == module_payload
    assert qm.Program.from_bytes(program_payload).to_bytes() == program_payload
    assert structure.operation_count == program_header.operation_count
    assert resources.opcodes.get("quantum.h", 0) == repeated_gates
    assert circuit.width == 1
    assert circuit.measurement_count == 1
    assert [scalar.kind for scalar in circuit.scalars] == ["measurement"]

    return _LexicalScale(
        repeated_gates=repeated_gates,
        source_bytes=len(source.encode()),
        ast_nodes=sum(1 for _ in ast.walk(ast.parse(source))),
        hir_bytes=len(hir_payload),
        hir_expressions=hir_header.expression_count,
        hir_statements=hir_header.statement_count,
        hir_preflight_nodes=hir_validation.preflight_message_nodes,
        hir_validation_visits=hir_validation.semantic_node_visits,
        hir_normalization_visits=hir_normalization.source_nodes,
        hir_ssa_lowering_visits=hir_ssa.source_nodes,
        module_bytes=len(module_payload),
        module_operations=module_header.operation_count,
        program_bytes=len(program_payload),
        program_operations=program_header.operation_count,
        persistence_input_bytes=(
            module_persistence.input_bytes + program_persistence.input_bytes
        ),
        persistence_nodes=(
            module_persistence.preflight_message_nodes
            + program_persistence.preflight_message_nodes
        ),
        persistence_encoded_bytes=(
            module_persistence.canonical_encoded_bytes
            + program_persistence.canonical_encoded_bytes
        ),
        persistence_encode_passes=(
            module_persistence.canonical_encode_passes
            + program_persistence.canonical_encode_passes
        ),
        qcore_verification_visits=(
            module_verification.verification_steps
            + program_verification.verification_steps
        ),
        specialization_visits=specialization_visits,
        exact_analysis_visits=resources.analysis_operation_visits,
        resource_work=resources.work,
        standard_optimization_visits=standard_optimization_visits,
        circuit_lowering_visits=circuit_lowering_visits,
        circuit_operations=circuit.operation_count,
        circuit_scalars=len(circuit.scalars),
    )


def _assert_affine(
    rows: tuple[_LexicalScale, ...],
    attribute: str,
) -> None:
    """Require one exact constant increment per repeated source operation."""

    values = [getattr(row, attribute) for row in rows]
    baseline = values[0]
    increments = {
        (value - baseline) // row.repeated_gates
        for row, value in zip(rows[1:], values[1:], strict=True)
        if (value - baseline) % row.repeated_gates == 0
    }
    assert len(increments) == 1, f"{attribute}: {values}"
    increment = next(iter(increments))
    assert increment > 0
    assert all(
        value == baseline + increment * row.repeated_gates
        for row, value in zip(rows, values, strict=True)
    ), f"{attribute}: {values}"


def _assert_bounded_wire_growth(
    rows: tuple[_LexicalScale, ...],
    attribute: str,
) -> None:
    """Reject superlinear retained-byte growth while allowing varint steps."""

    values = [getattr(row, attribute) for row in rows]
    baseline = values[0]
    per_gate = [
        (value - baseline) / row.repeated_gates
        for row, value in zip(rows[1:], values[1:], strict=True)
    ]
    assert min(per_gate) > 0
    assert max(per_gate) <= 2 * min(per_gate), f"{attribute}: {values}"


def test_lexical_compiler_size_ladder_is_linear_without_execution() -> None:
    """Capture, verify, analyze, lower, and persist one affine source ladder."""

    rows = tuple(_observe_lexical_scale(size) for size in (0, 1, 2, 4, 8, 16, 32))

    for attribute in (
        "source_bytes",
        "ast_nodes",
        "hir_expressions",
        "hir_statements",
        "hir_preflight_nodes",
        "hir_validation_visits",
        "hir_normalization_visits",
        "hir_ssa_lowering_visits",
        "module_operations",
        "program_operations",
        "persistence_nodes",
        "qcore_verification_visits",
        "specialization_visits",
        "exact_analysis_visits",
        "resource_work",
        "standard_optimization_visits",
    ):
        _assert_affine(rows, attribute)

    assert [row.circuit_lowering_visits for row in rows] == [3, 4, 3, 3, 3, 3, 3]
    assert [row.circuit_operations for row in rows] == [1, 2, 1, 1, 1, 1, 1]

    for attribute in (
        "hir_bytes",
        "module_bytes",
        "program_bytes",
        "persistence_input_bytes",
        "persistence_encoded_bytes",
    ):
        _assert_bounded_wire_growth(rows, attribute)

    assert {row.circuit_scalars for row in rows} == {1}
    assert {row.persistence_encode_passes for row in rows} == {4}


def test_const_loop_specialization_stays_compact() -> None:
    """Keep repetition compact in QCore and expand it linearly only at lowering."""

    module = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def sample(layers: qm.Const[qm.Int]) -> qm.Bits[1]:
    state = qm.qubits(1)
    for _ in range(layers):
        state = qm.h(state)
    return qm.measure(state)
""",
        module_name="compiler_specialization_scalability",
    )
    rows: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for layers in (1, 2, 4, 8, 16, 32):
        specialization_visits = _specialization_visits(module, layers=layers)
        program = module.specialize("sample", layers=layers)
        payload = program.to_bytes()
        header = native.inspect_program(payload)
        persistence = native.inspect_qcore_persistence_work(payload)
        verification = native.inspect_program_verification_work(payload)
        resources = native.inspect_exact_logical_counts(payload)
        circuit = native.inspect_static_circuit(payload)
        circuit_lowering_visits = native.inspect_static_circuit_operation_visits(
            payload
        )
        assert qm.Program.from_bytes(payload).to_bytes() == payload
        assert resources.opcodes["quantum.h"] == layers
        assert resources.work == layers + 2
        assert circuit.operation_count == layers + 1
        assert persistence.input_bytes == len(payload)
        assert persistence.canonical_encode_passes == 2
        assert verification.blocks > 1
        assert verification.literal_nodes > 0
        assert verification.type_components > 0
        assert verification.operation_kernel_nodes > 0
        assert verification.attribute_nodes > 0
        assert verification.call_graph_steps > 0
        assert verification.declaration_nodes > 0
        assert verification.static_domain_steps > 0
        assert verification.equation_steps > 0
        assert verification.call_abi_steps > 0
        assert verification.phase_lifting_steps > 0
        assert verification.control_dependence_steps > 0
        assert verification.contract_derivation_steps > 0
        assert verification.control_flow_steps > 0
        assert verification.operations == header.operation_count
        assert verification.verification_steps == (
            verification.literal_nodes
            + verification.type_components
            + verification.operation_kernel_nodes
            + verification.attribute_nodes
            + verification.call_graph_steps
            + verification.declaration_nodes
            + verification.static_domain_steps
            + verification.equation_steps
            + verification.call_abi_steps
            + verification.phase_lifting_steps
            + verification.control_dependence_steps
            + verification.contract_derivation_steps
            + verification.functions
            + verification.blocks
            + verification.operations
            + verification.values
            + verification.operands
            + verification.topology_comparisons
            + verification.ownership_path_steps
            + verification.control_flow_steps
        )
        rows.append(
            (
                layers,
                len(payload),
                header.operation_count,
                header.type_count,
                specialization_visits,
                verification.verification_steps,
                resources.analysis_operation_visits,
                circuit_lowering_visits,
                persistence.preflight_message_nodes,
                persistence.canonical_encoded_bytes,
            )
        )

    assert len({row[2] for row in rows}) == 1
    assert len({row[3] for row in rows}) == 1
    assert len({row[4] for row in rows}) == 1
    assert len({row[5] for row in rows}) == 1
    assert len({row[8] for row in rows}) == 1
    assert max(row[1] for row in rows) - min(row[1] for row in rows) <= 8
    assert max(row[9] for row in rows) - min(row[9] for row in rows) <= 16
    baseline_layers = rows[0][0]
    for column in (6, 7):
        baseline_visits = rows[0][column]
        per_layer = [
            (row[column] - baseline_visits) / (row[0] - baseline_layers)
            for row in rows[1:]
        ]
        assert min(per_layer) > 0
        assert max(per_layer) == min(per_layer), per_layer


def test_const_binding_verification_uses_one_parameter_index() -> None:
    """Keep declaration and domain work linear in independent Const bindings."""

    rows: list[tuple[int, int, int, int, int, int]] = []
    for parameter_count in (1, 2, 4, 8, 16, 32):
        module = qm.build_source(
            _const_parameter_source(parameter_count),
            module_name=f"compiler_binding_scalability_{parameter_count}",
        )
        program = module.specialize(
            "sample",
            **{f"p{index}": index + 1 for index in range(parameter_count)},
        )
        work = native.inspect_program_verification_work(program.to_bytes())
        rows.append(
            (
                parameter_count,
                work.declaration_nodes,
                work.static_domain_steps,
                work.equation_steps,
                work.call_abi_steps,
                work.verification_steps,
            )
        )

    baseline = rows[0]
    for column in (1, 2, 3, 4, 5):
        numerator = rows[1][column] - baseline[column]
        denominator = rows[1][0] - baseline[0]
        assert numerator % denominator == 0
        increment = numerator // denominator
        assert increment > 0
        assert all(
            row[column] == baseline[column] + increment * (row[0] - baseline[0])
            for row in rows
        ), rows


def test_static_call_graph_and_symbolic_resources_scale_linearly() -> None:
    """Grow static call depth without multiplying work or lowered output."""

    rows: list[_CallScale] = []
    for depth in (0, 1, 2, 4, 8, 16, 32):
        source = _call_chain_source(depth)
        unit = SourceUnit.from_text(
            source,
            display_name="compiler_call_scalability.py",
            module_name="compiler_call_scalability",
        )
        hir = parse_source(unit)
        hir_payload = encode_hir(hir)
        hir_validation = native.inspect_hir_validation_work(hir_payload)
        hir_normalization = native.inspect_hir_normalization_work(hir_payload)
        hir_ssa = native.inspect_hir_ssa_lowering_work(hir_payload)
        module = lower_hir(hir)
        module_payload = module.to_bytes()
        module_persistence = native.inspect_qcore_persistence_work(module_payload)
        module_verification = native.inspect_module_verification_work(module_payload)
        specialization_visits = _specialization_visits(module)
        program = module.specialize("sample")
        program_payload = program.to_bytes()
        program_persistence = native.inspect_qcore_persistence_work(program_payload)
        program_verification = native.inspect_program_verification_work(program_payload)
        structure = native.inspect_structure(program_payload)
        exact = native.inspect_exact_logical_counts(program_payload)
        circuit = native.inspect_static_circuit(program_payload)
        circuit_lowering_visits = native.inspect_static_circuit_operation_visits(
            program_payload
        )
        analysis_payload = encode_analysis_request(module, "sample")
        symbolic = native.inspect_symbolic_program_family_counts(analysis_payload)

        assert exact.opcodes["quantum.h"] == 1
        assert exact.work == 3
        assert circuit.operation_count == 2
        rows.append(
            _CallScale(
                depth=depth,
                source_bytes=len(source.encode()),
                module_bytes=len(module_payload),
                program_bytes=len(program_payload),
                analysis_bytes=len(analysis_payload),
                hir_preflight_nodes=hir_validation.preflight_message_nodes,
                hir_validation_visits=hir_validation.semantic_node_visits,
                hir_normalization_visits=hir_normalization.source_nodes,
                hir_ssa_lowering_visits=hir_ssa.source_nodes,
                persistence_input_bytes=(
                    module_persistence.input_bytes + program_persistence.input_bytes
                ),
                persistence_nodes=(
                    module_persistence.preflight_message_nodes
                    + program_persistence.preflight_message_nodes
                ),
                persistence_encoded_bytes=(
                    module_persistence.canonical_encoded_bytes
                    + program_persistence.canonical_encoded_bytes
                ),
                persistence_encode_passes=(
                    module_persistence.canonical_encode_passes
                    + program_persistence.canonical_encode_passes
                ),
                qcore_verification_visits=(
                    module_verification.verification_steps
                    + program_verification.verification_steps
                ),
                control_dependence_steps=(
                    module_verification.control_dependence_steps
                    + program_verification.control_dependence_steps
                ),
                contract_derivation_steps=(
                    module_verification.contract_derivation_steps
                    + program_verification.contract_derivation_steps
                ),
                function_count=structure.function_count,
                operation_count=structure.operation_count,
                call_sites=structure.call_site_count,
                specialization_visits=specialization_visits,
                exact_analysis_visits=exact.analysis_operation_visits,
                symbolic_analysis_visits=symbolic.analysis_operation_visits,
                circuit_lowering_visits=circuit_lowering_visits,
                resource_expressions=len(symbolic.expressions),
                resource_facts=len(symbolic.facts),
                circuit_operations=circuit.operation_count,
            )
        )

    assert all(row.function_count == row.depth + 2 for row in rows)
    assert all(row.call_sites == row.depth + 1 for row in rows)
    assert all(row.operation_count == row.depth + 5 for row in rows), [
        (row.depth, row.operation_count) for row in rows
    ]
    assert {row.circuit_operations for row in rows} == {2}
    assert len({row.resource_expressions for row in rows}) == 1
    assert {row.persistence_encode_passes for row in rows} == {4}

    for attribute in ("control_dependence_steps", "contract_derivation_steps"):
        base = getattr(rows[0], attribute)
        increment = getattr(rows[1], attribute) - base
        assert increment > 0
        assert all(
            getattr(row, attribute) == base + increment * row.depth for row in rows
        ), [(row.depth, getattr(row, attribute)) for row in rows]

    baseline = rows[0]
    for attribute in (
        "source_bytes",
        "module_bytes",
        "program_bytes",
        "analysis_bytes",
        "hir_preflight_nodes",
        "hir_validation_visits",
        "hir_normalization_visits",
        "hir_ssa_lowering_visits",
        "persistence_input_bytes",
        "persistence_nodes",
        "persistence_encoded_bytes",
        "qcore_verification_visits",
        "specialization_visits",
        "exact_analysis_visits",
        "symbolic_analysis_visits",
        "circuit_lowering_visits",
        "resource_facts",
    ):
        base = getattr(baseline, attribute)
        per_level = [(getattr(row, attribute) - base) / row.depth for row in rows[1:]]
        assert min(per_level) > 0, f"{attribute}: {per_level}"
        assert max(per_level) <= 2 * min(per_level), f"{attribute}: {per_level}"


def test_phase_lifting_proves_each_referenced_predicate_once() -> None:
    """Keep repeated phase sites linear and independent of predicate proof size."""

    slopes: list[int] = []
    for helper_backed in (False, True):
        rows: list[tuple[int, int]] = []
        for repetitions in (1, 2, 4, 8, 16, 32):
            module = qm.build_source(
                _phase_lifting_source(repetitions, helper_backed=helper_backed),
                module_name=f"phase_scalability_{helper_backed}_{repetitions}",
            )
            program = module.specialize("sample")
            payload = program.to_bytes()
            work = native.inspect_program_verification_work(payload)
            circuit = native.inspect_static_circuit(payload)
            phase_plans = native.plan_boolean_lifts(payload)
            assert work.phase_lifting_steps > 0
            assert circuit.width == 1
            assert circuit.operation_count == repetitions
            assert len(phase_plans) == 1
            assert phase_plans[0][0].endswith(".marked")
            rows.append((repetitions, work.phase_lifting_steps))

        baseline = rows[0]
        numerator = rows[1][1] - baseline[1]
        denominator = rows[1][0] - baseline[0]
        assert numerator % denominator == 0
        increment = numerator // denominator
        assert increment > 0
        assert all(
            work == baseline[1] + increment * (repetitions - baseline[0])
            for repetitions, work in rows
        ), rows
        slopes.append(increment)

    assert slopes[0] == slopes[1]


def test_higher_order_normalization_reports_every_pass() -> None:
    """Exercise callable arguments, aliases, and specialization once."""

    source = """
import qamomile as qm

@qm.qkernel
def unchanged(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return state

@qm.qkernel
def flipped(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state)

@qm.qkernel
def apply(
    operation: qm.QKernel[qm.Q[qm.Bits[1]], qm.Q[qm.Bits[1]]],
    state: qm.Q[qm.Bits[1]],
) -> qm.Q[qm.Bits[1]]:
    return operation(state)

@qm.qkernel
def sample(flag: qm.Param[qm.Bool]) -> qm.Shot[qm.Bits[1]]:
    operation = flipped if flag else unchanged
    state = apply(operation, qm.qubits(1))
    return qm.measure(state)
"""
    unit = SourceUnit.from_text(
        source,
        display_name="compiler_higher_order_scalability.py",
        module_name="compiler_higher_order_scalability",
    )
    work = native.inspect_hir_normalization_work(encode_hir(parse_source(unit)))

    assert work.function_results > 0
    assert work.return_continuations > 0
    assert work.higher_order > 0
    assert work.function_aliases > 0
    assert work.source_nodes == (
        work.function_results
        + work.return_continuations
        + work.higher_order
        + work.function_aliases
    )
