"""Invocation-local logical indices retain their Shot stage and identity."""

import hashlib

import pytest

import qamomile as qm
from qamomile._compiler.frontend import SourceUnit, parse_source
from qamomile.effects import Effect, Requirement
from qamomile.persistence import deserialize, serialize
from qamomile.proto import qcore_pb2 as wire

SOURCE = """
import qamomile as qm

@qm.qkernel
def is_first() -> qm.Shot[qm.Bool]:
    return qm.get_current_shot() == 0

@qm.qkernel
def stable() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    index = qm.get_current_shot()
    same = index == qm.get_current_shot()
    for _ in range(3):
        same = same and index == qm.get_current_shot()
    return same, is_first(), index < 2

@qm.qkernel
def alternating() -> qm.Shot[qm.Bit]:
    state = qm.prepare(False)
    if qm.get_current_shot() % 2 == 0:
        state = qm.x(state)
    return qm.measure(state)

@qm.qkernel
def selected() -> qm.Shot[qm.Bool]:
    values = [False, True]
    return values[qm.get_current_shot() % 2]

@qm.qkernel
def optional(enabled: qm.Const[qm.Bool]) -> qm.Shot[qm.Bool]:
    flag = False
    if enabled:
        flag = is_first()
    return flag

@qm.qkernel
def raw() -> qm.Shot[qm.Int]:
    return qm.get_current_shot()
"""


def test_current_shot_source_shape_and_public_spelling():
    """The public source spelling produces a nullary Shot[Int] context read."""
    source = """
from qamomile import qkernel, Shot, Int, get_current_shot as current
@qkernel
def read() -> Shot[Int]:
    return current()
"""
    hir = parse_source(SourceUnit.from_text(source))
    expression = hir.functions[0].body.statements[0].expression
    assert expression.opcode == "qcore.current_shot"
    assert expression.type == qm.Shot[qm.Int]
    assert expression.operands == ()
    assert expression.attributes == ()
    assert expression.regions == ()
    assert "get_current_shot" in qm.__all__
    with pytest.raises(RuntimeError, match="source intrinsic"):
        qm.get_current_shot()


@pytest.mark.parametrize("arguments", ["1", "index=0", "1, index=0"])
def test_current_shot_rejects_arguments(arguments):
    """Context reads accept neither positional nor named source arguments."""
    with pytest.raises(qm.FrontendError, match="Q5105.*expects no arguments"):
        parse_source(
            SourceUnit.from_text(
                "import qamomile as qm\n@qm.qkernel\n"
                "def invalid() -> qm.Shot[qm.Int]:\n"
                f"    return qm.get_current_shot({arguments})\n"
            )
        )


@pytest.mark.parametrize(
    "corruption", ["operand", "result", "attribute", "stage", "type"]
)
def test_native_current_shot_rejects_forged_operation_contract(corruption):
    """Resealing transport bytes cannot change the nullary Shot[Int] operation."""
    source = """import qamomile as qm
@qm.qkernel
def read(unused: qm.Param[qm.Int]) -> qm.Shot[qm.Int]:
    return qm.get_current_shot()
"""
    program = qm.build_source(source).specialize("read")
    package = wire.QCorePackageProto.FromString(program.to_bytes())
    function = next(
        item for item in package.functions if item.symbol == package.entrypoint
    )
    operation = next(
        item for item in function.body.operations if item.opcode == "qcore.current_shot"
    )
    if corruption == "operand":
        operation.operands.append(function.parameters[0].value.id)
    elif corruption == "result":
        extra = operation.results.add()
        extra.CopyFrom(operation.results[0])
        extra.id += 1_000_000
    elif corruption == "attribute":
        operation.attributes.add(name="origin").value.int_value = 1
    elif corruption == "stage":
        operation.results[0].type.name = "Param"
        function.result_type.CopyFrom(operation.results[0].type)
    else:
        operation.results[0].type.arguments[0].name = "Bool"
        function.result_type.CopyFrom(operation.results[0].type)
    package.package_digest = b""
    package.package_digest = hashlib.sha256(
        package.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(qm.VerificationError, match="Q1034.*current_shot"):
        qm.Program.from_bytes(package.SerializeToString(deterministic=True))


@pytest.mark.parametrize("stage", ["Const", "Param"])
def test_current_shot_cannot_be_demoted(stage):
    """A logical shot index cannot become a specialization or launch value."""
    with pytest.raises(qm.TypeCheckError, match="expected.*got Shot"):
        parse_source(
            SourceUnit.from_text(
                "import qamomile as qm\n@qm.qkernel\n"
                f"def invalid() -> qm.{stage}[qm.Int]:\n"
                "    return qm.get_current_shot()\n"
            )
        )


def test_current_shot_cannot_enter_param_helper():
    """Ordinary helper calls retain the shot index's availability stage."""
    with pytest.raises(qm.TypeCheckError):
        parse_source(
            SourceUnit.from_text(
                "import qamomile as qm\n"
                "@qm.qkernel\n"
                "def before_shots(index: qm.Param[qm.Int]) -> qm.Param[qm.Int]:\n"
                "    return index\n"
                "@qm.qkernel\n"
                "def invalid() -> qm.Shot[qm.Int]:\n"
                "    return before_shots(qm.get_current_shot())\n"
            )
        )


def test_current_shot_measurement_free_counts_reset_and_persistence():
    """Every invocation runs its own zero-based index domain without measurement."""
    program = qm.build_source(SOURCE).specialize("is_first")
    assert Requirement.LOGICAL_SHOT_CONTEXT in program.entry.requirements
    assert Effect.MEASUREMENT not in program.entry.effects
    assert Effect.PROBABILISTIC not in program.entry.effects
    restored = qm.Program.from_bytes(program.to_bytes())
    assert restored.to_bytes() == program.to_bytes()
    artifact = deserialize(serialize(qm.compile(restored, to=qm.Reference())))
    for shots in (1, 5, 9, 5):
        result = qm.execute(artifact, on=qm.Reference(), shots=shots)
        expected = {True: 1} if shots == 1 else {True: 1, False: shots - 1}
        assert isinstance(result, qm.SampleResult)
        assert result.values == expected
        assert result.shots == shots
        snapshot = qm.ExecutionSnapshot.completed(result)
        saved = qm.ExecutionSnapshot.from_bytes(snapshot.to_bytes())
        assert saved.result.values == expected


def test_current_shot_is_stable_through_helpers_and_loops():
    """Repeated reads and transitive helper calls observe one immutable index."""
    program = qm.build_source(SOURCE).specialize("stable")
    assert Requirement.LOGICAL_SHOT_CONTEXT in program.entry.requirements
    result = qm.run(program, on=qm.Reference(), shots=7).result
    assert result.values == {
        (True, True, True): 1,
        (True, False, True): 1,
        (True, False, False): 5,
    }


@pytest.mark.parametrize("entry", ["alternating", "selected"])
def test_current_shot_controls_quantum_work_and_classical_selection(entry):
    """The logical index changes algorithm results before they reach the host."""
    result = qm.run(
        qm.build_source(SOURCE).specialize(entry), on=qm.Reference(), shots=7
    ).result
    expected = {True: 4, False: 3} if entry == "alternating" else {False: 4, True: 3}
    assert result.values == expected


def test_unreachable_current_shot_loses_context_requirement():
    """Specialization removes a context read behind an inactive Const branch."""
    module = qm.build_source(SOURCE)
    inactive = module.specialize("optional", enabled=False)
    active = module.specialize("optional", enabled=True)
    assert Requirement.LOGICAL_SHOT_CONTEXT not in inactive.entry.requirements
    assert "qcore.current_shot" not in inactive.entry.inspect()
    assert Requirement.LOGICAL_SHOT_CONTEXT in active.entry.requirements


def test_current_shot_raw_integer_sample_has_explicit_codec_limit():
    """Unbounded source Int is not silently truncated into a sampled word."""
    program = qm.build_source(SOURCE).specialize("raw")
    with pytest.raises(qm.ArtifactError, match="Q7011.*sampled results"):
        qm.compile(program, to=qm.Reference())


def test_current_shot_single_value_evaluation_preserves_unbounded_int():
    """One value evaluation needs no fixed-width histogram representation."""
    assert qm.evaluate(qm.build_source(SOURCE).specialize("raw")) == 0


@pytest.mark.parametrize(
    "annotation, body",
    [
        ("qm.Shot[qm.Unit]", "    return\n"),
        ("qm.Unit", "    ignored = qm.measure(qm.prepare(False))\n    return\n"),
    ],
)
def test_zero_width_native_shot_mode_survives_without_context(annotation, body):
    """Execution cardinality comes from native mode even when no bits survive."""
    source = (
        f"import qamomile as qm\n@qm.qkernel\ndef sample() -> {annotation}:\n" + body
    )
    artifact = deserialize(
        serialize(
            qm.compile(qm.build_source(source).specialize("sample"), to=qm.Reference())
        )
    )
    assert artifact.result_abi.per_shot
    assert (
        Requirement.LOGICAL_SHOT_CONTEXT not in artifact.execution_contract.requirements
    )
    result = qm.execute(artifact, on=qm.Reference(), shots=4)
    assert result.shots == 4
    assert result.values == {None: 4}
    snapshot = qm.ExecutionSnapshot.from_bytes(
        qm.ExecutionSnapshot.completed(result).to_bytes()
    )
    assert snapshot.result.result_abi.per_shot
    assert snapshot.result.values == {None: 4}


@pytest.mark.parametrize("seed", [0, 3, 7, 41])
def test_current_shot_evaluation_matches_seeded_one_shot_run(seed):
    """Evaluation and one-shot Reference sampling share the index-zero stream."""
    program = qm.build_source(
        """import qamomile as qm
@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bit]]:
    index = qm.get_current_shot()
    measured = qm.measure(qm.h(qm.prepare(False)))
    return index == 0, measured
"""
    ).specialize("sample")
    restored = qm.Program.from_bytes(program.to_bytes())
    evaluated = qm.evaluate(restored, seed=seed)
    sampled = qm.run(restored, on=qm.Reference(), shots=1, seed=seed).result
    assert sampled.values == {evaluated: 1}


@pytest.mark.parametrize("seed", [1, 7, 41])
def test_current_shot_reference_random_streams_do_not_overlap(seed):
    """Index seeds cannot advance the same random stream by just one draw."""
    source = """import qamomile as qm
@qm.qkernel
def sample() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool], qm.Shot[qm.Bool], qm.Shot[qm.Bool], qm.Shot[qm.Bit], qm.Shot[qm.Bit]]:
    i = qm.get_current_shot()
    a = qm.measure(qm.h(qm.prepare(False)))
    b = qm.measure(qm.h(qm.prepare(False)))
    return i % 2 == 1, (i // 2) % 2 == 1, (i // 4) % 2 == 1, (i // 8) % 2 == 1, a, b
"""
    endpoint = qm.Reference()
    artifact = qm.compile(qm.build_source(source).specialize("sample"), to=endpoint)
    result = qm.execute(artifact, on=endpoint, shots=16, seed=seed)
    assert (
        qm.execute(artifact, on=endpoint, shots=16, seed=seed).values == result.values
    )
    rows = {
        sum(int(bit) * 2**position for position, bit in enumerate(row[:4])): row[4:]
        for row in result.values
    }
    assert set(rows) == set(range(16))
    assert any(rows[index][1] != rows[index + 1][0] for index in range(15))


@pytest.mark.parametrize("endpoint_name", ["reference", "aer"])
@pytest.mark.parametrize(
    "result_type, returned, expected",
    [
        ("qm.Bool", "True", {True: 4}),
        ("qm.Shot[qm.Bool]", "True", {True: 4}),
        ("qm.Unit", "", {None: 4}),
    ],
)
def test_unused_read_preserves_constant_and_empty_completions(
    endpoint_name, result_type, returned, expected
):
    """Removing an unused read cannot change the invocation's result count."""
    source = f"""import qamomile as qm
@qm.qkernel
def sample() -> {result_type}:
    index = qm.get_current_shot()
    return {returned}
"""
    if endpoint_name == "reference":
        endpoint = qm.Reference()
    else:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    artifact = qm.compile(
        qm.build_source(source).specialize("sample"), to=endpoint.target
    )
    result = qm.execute(artifact, on=endpoint, shots=4)
    assert isinstance(result, qm.SampleResult)
    assert result.shots == 4
    assert result.values == expected
    restored = qm.ExecutionSnapshot.from_bytes(
        qm.ExecutionSnapshot.completed(result).to_bytes()
    )
    assert restored.result.values == expected


@pytest.mark.parametrize("lift", ["phase_by", "xor_by"])
def test_current_shot_capture_is_rejected_in_boolean_lifts(lift):
    """Boolean lifts cannot hide an ambient logical-index dependency."""
    body = (
        "    state = qm.phase_by(marked, qm.qubits(1))\n    return qm.measure(state)\n"
        if lift == "phase_by"
        else "    source, target = qm.xor_by(marked, qm.qubits(1), qm.qubits(1))\n"
        "    qm.discard(source)\n"
        "    return qm.measure(target)\n"
    )
    source = (
        "import qamomile as qm\n"
        "@qm.qkernel\n"
        "def first() -> qm.Shot[qm.Bool]:\n"
        "    return qm.get_current_shot() == 0\n"
        "@qm.qkernel\n"
        "def marked(bits: qm.Bits[1]) -> qm.Shot[qm.Bool]:\n"
        "    return bool(bits[0]) and first()\n"
        "@qm.qkernel\n"
        "def invalid() -> qm.Shot[qm.Bits[1]]:\n" + body
    )
    with pytest.raises(qm.QamomileError, match="(?i)context|logical.shot|predicate"):
        qm.build_source(source).specialize("invalid")
