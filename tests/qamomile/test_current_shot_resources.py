"""Keep one symbolic shot context through per-trial resource analysis."""

from __future__ import annotations

import pytest

import qamomile as qm


def _study(body: str, *, helpers: str = "") -> qm.resources.ResourceStudy:
    """Analyze a small source kernel without compiling it for execution."""
    source = (
        "import qamomile as qm\n"
        + helpers
        + "\n@qm.qkernel\ndef sample() -> qm.Shot[qm.Bit]:\n"
        + "    state = qm.prepare(False)\n"
        + body
        + "    return qm.measure(state)\n"
    )
    return qm.resources.logical(qm.build_source(source).specialize("sample"))


@pytest.mark.parametrize(
    "bound", ["qm.get_current_shot()", "qm.get_current_shot() % 2"]
)
def test_shot_dependent_range_remains_an_exact_expression(bound: str) -> None:
    """A known relation to shot is not replaced by an unknown iteration count."""
    study = _study(f"    for _ in range({bound}):\n        state = qm.x(state)\n")
    gates = study.metric(qm.resources.LOGICAL_GATES)
    assert gates.relation is qm.resources.Relation.EQUALITY
    assert gates.expression is not None
    assert "shot" in str(gates.expression)
    if bound == "qm.get_current_shot()":
        assert str(gates.expression) == "shot"
    for shot in (0, 1, 2, 5, 21, 10**12 + 1):
        expected = shot if bound == "qm.get_current_shot()" else shot % 2
        assert gates.evaluate({"shot": shot}).numeric_enclosure() == (
            expected,
            expected,
        )
    assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study
    with pytest.raises(ValueError):
        gates.evaluate({"shot": -1})


def test_repeated_reads_and_helper_calls_share_the_resource_shot() -> None:
    """A helper read and a loop-body read identify the same logical trial."""
    helpers = """
@qm.qkernel
def index() -> qm.Shot[qm.Int]:
    return qm.get_current_shot()
"""
    study = _study(
        "    for _ in range(index() % 3):\n"
        "        if qm.get_current_shot() == index():\n"
        "            state = qm.x(state)\n",
        helpers=helpers,
    )
    restored = qm.resources.ResourceStudy.from_bytes(study.to_bytes())
    for candidate in (study, restored):
        gates = candidate.metric(qm.resources.LOGICAL_GATES)
        assert gates.relation is qm.resources.Relation.EQUALITY
        for shot in range(7):
            assert gates.evaluate({"shot": shot}).numeric_enclosure() == (
                shot % 3,
                shot % 3,
            )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("index % 2 == 0", "index % 2 != 0"),
        ("index % 2 == 0", "not (index % 2 == 0)"),
        ("index < 4", "index >= 4"),
        ("index <= 4", "index > 4"),
        ("0 == index % 2", "index % 2 != 0"),
        ("index % 2 != 0", "index % 2 == 0"),
        ("0 <= index < 4", "not (0 <= index < 4)"),
    ],
)
def test_complementary_shot_conditions_have_one_total_gate(
    first: str, second: str
) -> None:
    """Sequential complementary branches retain their total-cost relationship."""
    study = _study(
        "    index = qm.get_current_shot()\n"
        f"    if {first}:\n        state = qm.x(state)\n"
        f"    if {second}:\n        state = qm.z(state)\n"
    )
    for candidate in (study, qm.resources.ResourceStudy.from_bytes(study.to_bytes())):
        total = candidate.metric(qm.resources.LOGICAL_GATES)
        assert total.relation is qm.resources.Relation.EQUALITY
        assert total.numeric_enclosure() == (1, 1)
        for opcode in ("quantum.x", "quantum.z"):
            count = candidate.metric(qm.resources.logical_operation_count(opcode))
            lower, upper = count.numeric_enclosure()
            if lower is not None and upper is not None:
                assert (lower, upper) == (0, 1)
            else:
                for shot in (0, 1, 4, 5):
                    value = count.evaluate({"shot": shot}).numeric_enclosure()
                    assert value in ((0, 0), (1, 1))


def test_unrelated_shot_conditions_are_not_assumed_complementary() -> None:
    """Different modular tests can both select or both skip a gate."""
    study = _study(
        "    index = qm.get_current_shot()\n"
        "    if index % 2 == 0:\n        state = qm.x(state)\n"
        "    if index % 3 == 0:\n        state = qm.z(state)\n"
    )
    gates = study.metric(qm.resources.LOGICAL_GATES)
    assert gates.numeric_enclosure() != (1, 1)
    for shot in range(8):
        expected = int(shot % 2 == 0) + int(shot % 3 == 0)
        lower, upper = gates.evaluate({"shot": shot}).numeric_enclosure()
        assert lower is not None and lower <= expected
        assert upper is None or expected <= upper


def test_helper_predicates_keep_their_complementary_relationship() -> None:
    """Separate helpers return conditions on the same analysis context."""
    helpers = """
@qm.qkernel
def even() -> qm.Shot[qm.Bool]:
    return qm.get_current_shot() % 2 == 0
@qm.qkernel
def odd() -> qm.Shot[qm.Bool]:
    return qm.get_current_shot() % 2 != 0
"""
    study = _study(
        "    if even():\n        state = qm.x(state)\n"
        "    if odd():\n        state = qm.z(state)\n",
        helpers=helpers,
    )
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (1, 1)


def test_repeating_the_same_condition_does_not_cancel_its_cost() -> None:
    """Repeated selection can execute both gates instead of exactly one."""
    study = _study(
        "    index = qm.get_current_shot()\n"
        "    if index % 2 == 0:\n        state = qm.x(state)\n"
        "    if index % 2 == 0:\n        state = qm.z(state)\n"
    )
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (0, 2)


def test_real_cast_roundtrip_is_not_treated_as_the_integer_shot() -> None:
    """Real conversion can round large integers and cannot prove a global identity."""
    study = _study(
        "    index = qm.get_current_shot()\n"
        "    if int(float(index)) == index:\n        state = qm.x(state)\n"
    )
    gates = study.metric(qm.resources.LOGICAL_GATES)
    assert gates.numeric_enclosure() != (1, 1)
    for shot in (0, 1, 2, 2**53 + 1):
        expected = int(int(float(shot)) == shot)
        lower, upper = gates.evaluate({"shot": shot}).numeric_enclosure()
        assert lower is None or lower <= expected
        assert upper is None or expected <= upper


def test_measurement_conditions_keep_conservative_bounds() -> None:
    """Measurements are not turned into deterministic shot predicates."""
    study = _study(
        "    flag = qm.measure(qm.h(qm.prepare(False)))\n"
        "    if flag:\n        state = qm.x(state)\n"
        "    if not flag:\n        state = qm.z(state)\n"
    )
    assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (1, 3)


def test_analysis_shot_is_not_a_program_or_execution_parameter() -> None:
    """Resource estimation leaves the program signature and runtime contract intact."""
    source = """
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bit]:
    state = qm.prepare(False)
    for _ in range(qm.get_current_shot()):
        state = qm.x(state)
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("sample")
    before = program.to_bytes()
    study = qm.resources.logical(program)
    assert program.to_bytes() == before
    assert study.semantic_digest == program.semantic_digest
    assert not program.entry.parameters


def _symbols(expression: qm.resources.Expr) -> set[qm.resources.Symbol]:
    """Collect declarations exposed by a small public resource expression."""
    pending = [expression]
    seen = set()
    result = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        declaration = getattr(node, "symbol", None)
        if declaration is not None:
            result.add(declaration)
        pending.extend(node.arguments)
        pending.extend(node.predicates)
    return result


def test_analysis_shot_does_not_capture_a_parameter_named_shot() -> None:
    """Qualified identities keep source inputs separate from the ambient index."""
    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def sample(shot: qm.Param[qm.Int]) -> qm.Shot[qm.Bit]:
    state = qm.prepare(False)
    for _ in range(shot + qm.get_current_shot()):
        state = qm.x(state)
    return qm.measure(state)
""").specialize("sample")
    study = qm.resources.logical(program)
    for candidate in (study, qm.resources.ResourceStudy.from_bytes(study.to_bytes())):
        gates = candidate.metric(qm.resources.LOGICAL_GATES)
        assert gates.expression is not None
        declarations = _symbols(gates.expression)
        assert len(declarations) == 2
        assert {symbol.name for symbol in declarations} == {"shot"}
        parameter = next(
            symbol for symbol in declarations if symbol.scope == program.entry.symbol
        )
        context = next(symbol for symbol in declarations if symbol is not parameter)
        assert context.domain is qm.resources.Domain.NATURAL
        assert context.identity != parameter.identity
        assert gates.evaluate(
            {parameter.identity: 2, context.identity: 5}
        ).numeric_enclosure() == (7, 7)
        with pytest.raises(ValueError, match="ambiguous"):
            gates.evaluate({"shot": 5})


def test_while_iteration_symbol_remains_independent_of_shot() -> None:
    """An unproved while count can still retain an exact shot-dependent body."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(rounds: qm.Param[qm.Int]) -> qm.Shot[qm.Bit]:
    state = qm.prepare(False)
    i = 0
    while i < rounds:
        for _ in range(qm.get_current_shot()):
            state = qm.x(state)
        i += 1
    return qm.measure(state)
"""
    study = qm.resources.logical(qm.build_source(source).specialize("sample"))
    assert len(study.iteration_symbols) == 1
    iterations = study.iteration_symbols[0]
    for candidate in (study, qm.resources.ResourceStudy.from_bytes(study.to_bytes())):
        scenario = candidate.assume_iterations({iterations: 3})
        gates = scenario.metric(qm.resources.LOGICAL_GATES)
        assert gates.evaluate({"shot": 5}).numeric_enclosure() == (15, 15)


def test_symbolic_outer_loop_does_not_rename_the_shot_context() -> None:
    """The induction variable and the ambient shot remain independent symbols."""
    source = """
import qamomile as qm
@qm.qkernel
def sample(rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bit]:
    state = qm.prepare(False)
    for _ in range(rounds):
        for _ in range(qm.get_current_shot() % 2):
            state = qm.x(state)
    return qm.measure(state)
"""
    study = qm.resources.logical(qm.build_source(source)["sample"])
    gates = study.metric(qm.resources.LOGICAL_GATES)
    for shot in (0, 1, 8, 9):
        expected = 7 * (shot % 2)
        assert gates.evaluate({"rounds": 7, "shot": shot}).numeric_enclosure() == (
            expected,
            expected,
        )


def test_branch_projection_does_not_expose_undefined_arithmetic() -> None:
    """Removing a nonzero guard must not introduce division by zero in a bound."""
    study = _study(
        "    index = qm.get_current_shot()\n"
        "    if index != 0:\n"
        "        for _ in range(10 // index):\n"
        "            state = qm.x(state)\n"
    )
    for candidate in (study, qm.resources.ResourceStudy.from_bytes(study.to_bytes())):
        gates = candidate.metric(qm.resources.LOGICAL_GATES)
        for shot in (0, 1, 2, 11):
            expected = 0 if shot == 0 else 10 // shot
            lower, upper = gates.evaluate({"shot": shot}).numeric_enclosure()
            assert lower is None or lower <= expected
            assert upper is None or expected <= upper


def test_many_independent_conditions_keep_a_sound_bounded_analysis() -> None:
    """A large condition set yields safe results instead of false exactness."""
    divisors = tuple(range(2, 22))
    body = "    index = qm.get_current_shot()\n" + "".join(
        f"    if index % {divisor} == 0:\n        state = qm.x(state)\n"
        for divisor in divisors
    )
    study = _study(body)
    for candidate in (study, qm.resources.ResourceStudy.from_bytes(study.to_bytes())):
        gates = candidate.metric(qm.resources.LOGICAL_GATES)
        for shot in (0, 1, 12, 23):
            expected = sum(shot % divisor == 0 for divisor in divisors)
            lower, upper = gates.evaluate({"shot": shot}).numeric_enclosure()
            assert lower is None or lower <= expected
            assert upper is None or expected <= upper


@pytest.mark.parametrize("cost", ["direct", "complementary"])
@pytest.mark.parametrize("register_condition", [False, True])
def test_custom_peak_does_not_acquire_an_additive_shot_equality(
    cost: str, register_condition: bool
) -> None:
    """An unsupported peak composition cannot publish the sum as an exact peak."""
    condition = "    if index < 1:\n        state = qm.x(state)\n"
    source = (
        """
import qamomile as qm
@qm.qkernel(kind="unitary")
def first(state: qm.Q[qm.Bits[1]], n: qm.Shot[qm.Int]) -> qm.Q[qm.Bits[1]]:
    ...
@qm.qkernel(kind="unitary")
def second(state: qm.Q[qm.Bits[1]], n: qm.Shot[qm.Int]) -> qm.Q[qm.Bits[1]]:
    ...
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    index = qm.get_current_shot()
    state = qm.qubits(1)
"""
        + (condition if register_condition else "")
        + """
    state = first(state, index)
    state = second(state, index)
    return qm.measure(state)
"""
    )
    module = qm.build_source(source, module_name="shot_custom_peak")
    metric = qm.resources.MetricDefinition(
        "example.shot.custom_peak",
        qm.resources.Coordinate.ALGORITHMIC,
        qm.resources.Domain.NATURAL,
        qm.resources.ONE,
        composition=qm.resources.Composition.PEAK_LIVENESS,
    )
    rules = []
    for name, selected in (("first", 2), ("second", 1)):
        function = module.function(name)
        parameter = qm.resources.rule_parameter(
            function, "n", domain=qm.resources.Domain.NATURAL
        )
        expression = parameter
        if cost == "complementary":
            predicate = qm.resources.comparison(
                "less", parameter, qm.resources.literal(1)
            )
            expression = qm.resources.piecewise(
                ((predicate, qm.resources.literal(selected)),),
                otherwise=qm.resources.literal(3 - selected),
            )
        rules.append(
            qm.resources.ResourceRule(
                identity=f"example.shot.peak.{name}.v1",
                callable=function,
                metric=metric,
                relation=qm.resources.Relation.EQUALITY,
                expression=expression,
                provenance="example.shot.custom-peak-contract.v1",
            )
        )
    study = qm.resources.logical(module.specialize("sample"), rules=tuple(rules))
    for candidate in (study, qm.resources.ResourceStudy.from_bytes(study.to_bytes())):
        value = candidate.metric(metric)
        for shot in (0, 1, 2, 13):
            expected = shot if cost == "direct" else 2
            lower, upper = value.evaluate({"shot": shot}).numeric_enclosure()
            assert lower is None or lower <= expected
            assert upper is None or expected <= upper


@pytest.mark.parametrize(
    "domain",
    [
        qm.resources.Domain.INTEGER,
        qm.resources.Domain.RATIONAL,
        qm.resources.Domain.REAL,
    ],
)
def test_signed_custom_peak_is_rejected_without_a_proved_lower_bound(
    domain: qm.resources.Domain,
) -> None:
    """The declared metric domain controls whether zero is a valid fallback."""

    @qm.qkernel(kind="unitary")
    def opaque(state: qm.Q[qm.Bool], index: qm.Shot[qm.Int]) -> qm.Q[qm.Bool]: ...

    @qm.qkernel
    def sample() -> qm.Shot[qm.Bit]:
        state = qm.prepare(False)
        state = opaque(state, qm.get_current_shot())
        return qm.measure(state)

    metric = qm.resources.MetricDefinition(
        "example.shot.signed_peak",
        qm.resources.Coordinate.ALGORITHMIC,
        domain,
        qm.resources.ONE,
        composition=qm.resources.Composition.PEAK_LIVENESS,
    )
    rule = qm.resources.ResourceRule(
        identity="example.shot.signed-peak.v1",
        callable=opaque,
        metric=metric,
        relation=qm.resources.Relation.EQUALITY,
        expression=qm.resources.rule_parameter(
            opaque, "index", domain=qm.resources.Domain.NATURAL
        ),
        provenance="example.shot.signed-peak-contract.v1",
    )
    with pytest.raises(qm.CapabilityError, match="Q4211"):
        qm.resources.logical(sample, rules=rule)
