"""Binary observations have one exact scalar type across the public lifecycle."""

from pathlib import Path

import pytest

import qamomile as qm
from qamomile.compilation import ResultABI
from qamomile.types import as_type


def program(body, result="qm.Shot[qm.Bit]", parameters="", prefix=""):
    """Build a source witness with an explicit observation contract."""
    return qm.build_source(
        f"import qamomile as qm\n{prefix}\n@qm.qkernel\ndef sample({parameters}) -> {result}:\n"
        + "\n".join(f"    {line}" for line in body.splitlines())
        + "\n"
    ).specialize("sample")


@pytest.mark.parametrize("literal", ["0", "1", "False", "True"])
def test_binary_preparation_observations_and_restored_metadata(literal):
    """Boolean payloads survive while every measured scalar decodes to int."""
    source = program(f"return qm.measure(qm.prepare({literal}))")
    expected = int(literal in {"1", "True"})
    payload = qm.Bool if literal in {"True", "False"} else qm.Bit
    preparation = next(
        op for op in source.entry.body.operations if op.opcode == "quantum.prepare"
    )
    assert preparation.results[0].type == qm.Q[payload]
    for current in (source, qm.Program.from_bytes(source.to_bytes())):
        assert current.entry.result_type == qm.Shot[qm.Bit]
        assert ResultABI.from_program(current).type == qm.Shot[qm.Bit]
        value = qm.evaluate(current)
        assert type(value) is int and value == expected
        record = qm.run(current, on=qm.Reference(), shots=4, seed=3)
        assert record.artifact.result_abi.type == qm.Shot[qm.Bit]
        snapshot = qm.ExecutionSnapshot.completed(record.result)
        for result in (
            record.result,
            qm.ExecutionSnapshot.from_bytes(snapshot.to_bytes()).result,
        ):
            assert result.values == {expected: 4}
            assert all(type(key) is int for key in result.values)


@pytest.mark.parametrize("axis,gate", [("z", "qm.x"), ("x", "qm.h"), ("y", "qm.h")])
def test_basis_observation_preserves_boolean_successor(axis, gate):
    """All basis measurements yield Bit while keeping the original Q[Bool]."""
    lines = [f"state = {gate}(qm.prepare(False))"]
    if axis == "y":
        lines.append("state = qm.s(state)")
    lines += [
        f"bit, state = qm.measure_{axis}(state)",
        "qm.discard(state)",
        "return bit",
    ]
    source = program("\n".join(lines))
    measurement = next(
        op
        for op in source.entry.body.operations
        if op.opcode == f"quantum.measure_{axis}"
    )
    assert measurement.results[0].type == as_type(tuple[qm.Shot[qm.Bit], qm.Q[qm.Bool]])
    value = qm.evaluate(source)
    assert type(value) is int
    assert value == (1 if axis == "z" else 0)
    result = qm.run(source, on=qm.Reference(), shots=4, seed=3).result
    assert result.values == {value: 4}
    assert all(type(key) is int for key in result.values)


def test_reset_and_boolean_conditional_flip():
    """Reset retains its payload type and a measured flag drives control."""
    source = program(
        "first, state = qm.measure_reset(qm.prepare(True))\n"
        "if first:\n    state = qm.x(state)\n"
        "return qm.measure(state)"
    )
    reset = next(
        op
        for op in source.entry.body.operations
        if op.opcode == "quantum.measure_reset"
    )
    assert reset.results[0].type == as_type(tuple[qm.Shot[qm.Bit], qm.Q[qm.Bool]])
    assert type(qm.evaluate(source)) is int
    assert qm.evaluate(source) == 1


@pytest.mark.parametrize(
    "expression,result,expected",
    [
        ("qm.prepare((1,))", "qm.Shot[qm.Bits[1]]", (1,)),
        ("qm.prepare((1, 0, 1))", "qm.Shot[qm.Bits[3]]", (1, 0, 1)),
        ("qm.as_uint(qm.prepare((1, 0, 1)))", "qm.Shot[qm.UInt[3]]", 5),
        (
            "qm.as_fixed(qm.prepare((1, 0, 1)), integer_bits=1)",
            "qm.Shot[qm.UFixed[1, 2]]",
            1.25,
        ),
    ],
)
def test_register_and_numeric_meanings_remain_distinct(expression, result, expected):
    """One-element registers and asymmetric numeric words retain their ABI."""
    source = program(f"return qm.measure({expression})", result)
    actual = qm.evaluate(source)
    assert type(actual) is type(expected) and actual == expected
    result = qm.run(source, on=qm.Reference(), shots=4, seed=3).result
    assert result.values == {expected: 4}
    assert type(next(iter(result.values))) is type(expected)


def test_boolean_expressions_and_boolean_preparation_calls_remain_supported():
    """Explicit conversion and a called Boolean producer keep ordinary Bool."""
    prefix = "@qm.qkernel\ndef flag() -> qm.Bool:\n    return True\n"
    source = program(
        "bit = qm.measure(qm.prepare(flag()))\nreturn bit, bool(bit), bool(bit) == True",
        "tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bool], qm.Shot[qm.Bool]]",
        prefix=prefix,
    )
    values = qm.evaluate(source)
    assert values == (1, True, True)
    assert tuple(type(value) for value in values) == (int, bool, bool)


@pytest.mark.parametrize("operand", ["2", "-1", "1.0", "0.0", "(1.0,)"])
def test_preparation_rejects_nonbinary_integers_and_real_values(operand):
    """Checked integer preparation does not accept truthiness or real values."""
    with pytest.raises(qm.QamomileError):
        program(f"return qm.measure(qm.prepare({operand}))")


def test_correlated_scalar_observations_are_exact_ints():
    """Measurement feedback preserves joint rows without Boolean keys."""
    source = program(
        "first = qm.measure(qm.h(qm.prepare(False)))\nsecond = qm.prepare(first)\nreturn first, qm.measure(second)",
        "tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bit]]",
    )
    result = qm.run(source, on=qm.Reference(), shots=64, seed=4).result
    assert set(result.values) == {(0, 0), (1, 1)}
    assert all(type(bit) is int for row in result.values for bit in row)


def test_old_program_equations_reject_but_stored_results_retain_bool():
    """Program verification changes without reinterpreting historical result ABI."""
    fixtures = Path(__file__).with_name("fixtures")
    with pytest.raises(qm.QamomileError, match="quantum.measure"):
        qm.Program.from_bytes(
            (fixtures / "boolean_measurement_program.bin").read_bytes()
        )
    restored = qm.ExecutionSnapshot.from_bytes(
        (fixtures / "boolean_measurement_result.bin").read_bytes()
    )
    assert restored.result.values == {True: 2}
    assert type(next(iter(restored.result.values))) is bool


_CAPTURED_ONE = 1


@qm.qkernel
def _prepare_named_global() -> qm.Shot[qm.Bit]:
    """Retain the name of a file-backed integer capture."""
    return qm.measure(qm.prepare(_CAPTURED_ONE))


def test_decorated_integer_captures_prepare_binary_values():
    """File-backed globals and closures prepare the same binary integer value."""
    captured = 1

    @qm.qkernel
    def local() -> qm.Shot[qm.Bit]:
        """Use a named closure capture rather than a directly written literal."""
        return qm.measure(qm.prepare(captured))

    for kernel in (_prepare_named_global, local):
        source = kernel.specialize()
        assert qm.evaluate(source) == 1
        assert type(qm.evaluate(source)) is int


@pytest.mark.parametrize("stage", ["Const", "Param"])
@pytest.mark.parametrize("value", [0, 1, -1, 2, 1 << 100])
def test_integer_preparation_checks_values_at_specialization_or_launch(stage, value):
    """Staged integer variables obey the same exact binary value domain."""
    family = qm.build_source(
        f"import qamomile as qm\n@qm.qkernel\ndef sample(value: qm.{stage}[qm.Int]) -> qm.Shot[qm.Bit]:\n    return qm.measure(qm.prepare(value))\n"
    )
    if value not in (0, 1):
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            source = family.specialize(
                "sample", **({"value": value} if stage == "Const" else {})
            )
            qm.evaluate(source, **({"value": value} if stage == "Param" else {}))
        return
    source = family.specialize(
        "sample", **({"value": value} if stage == "Const" else {})
    )
    result = qm.evaluate(source, **({"value": value} if stage == "Param" else {}))
    assert type(result) is int and result == value


@pytest.mark.parametrize("offset", [0, 2])
def test_shot_integer_preparation_checks_before_reusing_observations(offset):
    """Computed per-shot integers are checked before conditional preparation."""
    source = program(
        f"bit = qm.measure(qm.prepare(True))\nvalue = (1 if bit else 0) + {offset}\nreturn qm.measure(qm.prepare(value))"
    )
    if offset:
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.evaluate(source)
    else:
        assert qm.evaluate(source) == 1
        assert type(qm.evaluate(source)) is int


def test_untaken_preparation_does_not_check_invalid_integer():
    """A checked cast stays inside its source branch."""
    source = program(
        "bit = qm.measure(qm.prepare(False))\nif bit:\n    state = qm.prepare(value)\n    qm.discard(state)\nreturn bit",
        parameters="value: qm.Param[qm.Int]",
    )
    assert qm.evaluate(source, value=2) == 0


@pytest.mark.parametrize("backend", ["reference", "qiskit"])
def test_unused_integer_preparation_still_checks_after_optimization(backend):
    """Removing a vacuous quantum owner cannot remove its checked conversion."""
    from qamomile._native import compiler as native

    source = program(
        "unused = qm.prepare(value)\nreturn qm.measure(qm.prepare(False))",
        parameters="value: qm.Param[qm.Int]",
    )
    optimized = qm.Program.from_bytes(
        native.optimize_standard_program(source.to_bytes())
    )
    assert any(
        op.opcode == "classical.cast" and op.attribute("target") == "Bit"
        for op in optimized.entry.body.operations
    )
    if backend == "reference":
        endpoint = qm.Reference()
    else:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    artifact = qm.compile(optimized, to=endpoint.target)
    with pytest.raises(qm.QamomileError, match="0 or 1"):
        qm.execute(artifact, on=endpoint, parameters={"value": 2}, shots=4, seed=3)
    assert qm.execute(
        artifact, on=endpoint, parameters={"value": 1}, shots=4, seed=3
    ).values == {0: 4}


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("value", [0, 1, 2])
@pytest.mark.parametrize("local_measurement", [False, True])
@pytest.mark.parametrize("optimized", [False, True])
def test_branch_preparation_checks_follow_nested_instructions(
    initial, value, local_measurement, optimized
):
    """Nested checks follow their measurements and precede an early return."""
    operand = "(1 if local else 0) + value" if local_measurement else "value"
    body = (
        f"flag = qm.measure(qm.prepare({initial}))\n"
        "if flag:\n"
        "    state = qm.x(qm.prepare(False))\n"
        "    local = qm.measure(state)\n"
        f"    unused = qm.prepare({operand})\n"
        "    qm.discard(unused)\n"
        "    return flag\n"
        "return flag"
    )
    source = program(body, parameters="value: qm.Param[qm.Int]")
    if optimized:
        from qamomile._native import compiler as native

        source = qm.Program.from_bytes(
            native.optimize_standard_program(source.to_bytes())
        )
    if initial and value + int(local_measurement) > 1:
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.evaluate(source, value=value)
    else:
        result = qm.evaluate(source, value=value)
        assert type(result) is int and result == int(initial)


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("value", [0, 1, 2])
def test_sibling_branch_checks_keep_local_positions(initial, value):
    """Branches with unequal gate counts preserve checked preparation order."""
    from qamomile._native import compiler as native

    source = program(
        f"flag = qm.measure(qm.prepare({initial}))\n"
        "if flag:\n"
        "    state = qm.h(qm.x(qm.h(qm.prepare(False))))\n"
        "    qm.discard(state)\n"
        "    unused = qm.prepare(value)\n"
        "    qm.discard(unused)\n"
        "else:\n"
        "    unused = qm.prepare(value)\n"
        "    qm.discard(unused)\n"
        "return flag",
        parameters="value: qm.Param[qm.Int]",
    )
    source = qm.Program.from_bytes(native.optimize_standard_program(source.to_bytes()))
    if value == 2:
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.evaluate(source, value=value)
    else:
        assert qm.evaluate(source, value=value) == int(initial)


@pytest.mark.parametrize("indexed", [False, True])
def test_sdk_projection_rejects_unrepresentable_nested_checks(indexed):
    """SDK projections cannot silently discard checks owned by nested blocks."""
    pytest.importorskip("qiskit_aer")
    from qamomile.qiskit import Aer

    source = program(
        ("index = qm.get_current_shot()\n" if indexed else "")
        + "flag = qm.measure(qm.prepare(True))\n"
        "if flag:\n"
        "    local = qm.measure(qm.x(qm.prepare(False)))\n"
        f"    unused = qm.prepare(value{' + index' if indexed else ''})\n"
        "    qm.discard(unused)\n"
        "return flag",
        parameters="value: qm.Param[qm.Int]",
    )
    with pytest.raises(qm.QamomileError, match="ordered in-job execution"):
        qm.compile(source, to=Aer().target)


@pytest.mark.parametrize("backend", ["reference", "qiskit"])
def test_integer_preparation_inside_loop_keeps_its_domain_check(backend):
    """Loop lowering never erases a binary check as ordinary gate arithmetic."""
    from qamomile._native import compiler as native

    source = program(
        "state = qm.prepare(False)\n"
        "for index in range(3):\n"
        "    unused = qm.prepare(value)\n"
        "    state = qm.rx(state, 0.1 * index)\n"
        "return qm.measure(state)",
        parameters="value: qm.Param[qm.Int]",
    )
    source = qm.Program.from_bytes(native.optimize_standard_program(source.to_bytes()))
    if backend == "reference":
        endpoint = qm.Reference()
    else:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    artifact = qm.compile(source, to=endpoint.target)
    with pytest.raises(qm.QamomileError, match="0 or 1"):
        qm.execute(artifact, on=endpoint, parameters={"value": 2}, shots=4)
    result = qm.execute(artifact, on=endpoint, parameters={"value": 1}, shots=4, seed=3)
    assert sum(result.values.values()) == 4


@pytest.mark.parametrize("backend", ["reference", "qiskit"])
@pytest.mark.parametrize("indexed", [False, True])
@pytest.mark.parametrize("survives_measurement", [False, True])
def test_unused_shot_preparation_cannot_bypass_its_check(
    backend, indexed, survives_measurement
):
    """A live quantum owner cannot hide an invalid unused per-shot preparation."""
    from qamomile._native import compiler as native

    source = program(
        ("index = qm.get_current_shot()\n" if indexed else "")
        + ("state = qm.prepare(False)\n" if survives_measurement else "")
        + "bit = qm.measure(qm.prepare(True))\n"
        + ("" if survives_measurement else "state = qm.prepare(False)\n")
        + f"value = {'index + 2' if indexed else '2'} if bit else 0\n"
        "unused = qm.prepare(value)\n"
        "state = qm.x(state)\n"
        "return qm.measure(state)"
    )
    source = qm.Program.from_bytes(native.optimize_standard_program(source.to_bytes()))
    if backend == "reference":
        endpoint = qm.Reference()
    else:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    with pytest.raises(qm.QamomileError, match="0 or 1|ordered in-job execution"):
        artifact = qm.compile(source, to=endpoint.target)
        qm.execute(artifact, on=endpoint, shots=4, seed=3)


@pytest.mark.parametrize("direct_result", [False, True])
def test_bound_sdk_export_preserves_shot_preparation_checks(direct_result):
    """Binding a valid initial integer cannot erase a later per-shot check."""
    pytest.importorskip("qiskit_aer")
    from qamomile._native import compiler as native
    from qamomile.qiskit import Aer

    source = program(
        "state = qm.prepare(initial)\n"
        "bit = qm.measure(qm.prepare(1))\n"
        "unused = qm.prepare(2 if bit else 0)\n"
        "state = qm.x(state)\n"
        "observed = qm.measure(state)\n"
        + ("return observed" if direct_result else "return bool(observed)"),
        result="qm.Shot[qm.Bit]" if direct_result else "qm.Shot[qm.Bool]",
        parameters="initial: qm.Param[qm.Int]",
    )
    source = qm.Program.from_bytes(native.optimize_standard_program(source.to_bytes()))
    endpoint = Aer()
    artifact = qm.compile(source, to=endpoint.target)
    with pytest.raises(qm.QamomileError, match="0 or 1|ordered in-job execution"):
        qm.execute(artifact, on=endpoint, parameters={"initial": 0}, shots=4, seed=3)


@pytest.mark.parametrize("stores", [False, True])
@pytest.mark.parametrize("value", [0, 1, 2])
def test_boolean_stores_preserve_terminal_preparation_checks(stores, value):
    """Inserted stores preserve terminal validation and the binary domain."""
    pytest.importorskip("qiskit_aer")
    from qamomile._native import compiler as native
    from qamomile.qiskit import Aer, to_circuit

    source = program(
        "c = bool(qm.measure(qm.prepare(1)))\n"
        "a = bool(qm.measure(qm.prepare(1)))\n"
        "b = bool(qm.measure(qm.prepare(0)))\n"
        "selected = a if c else b\n"
        "left = qm.prepare(0)\n"
        "right = qm.prepare(0)\n"
        "if selected:\n"
        "    left = qm.x(left)\n"
        "if selected:\n"
        "    right = qm.x(right)\n"
        "first = qm.measure(left)\n"
        "second = qm.measure(right)\n"
        "unused = qm.prepare(value if first else 0)\n"
        "return first, second",
        result="tuple[qm.Shot[qm.Bit], qm.Shot[qm.Bit]]",
        parameters="value: qm.Param[qm.Int]",
    )
    source = qm.Program.from_bytes(native.optimize_standard_program(source.to_bytes()))
    endpoint = Aer(use_classical_stores=stores)
    artifact = qm.compile(source, to=endpoint.target)
    circuit = to_circuit(source, target=endpoint.target, parameters={"value": value})
    assert circuit.count_ops().get("store", 0) == int(stores)
    if value == 2:
        with pytest.raises(qm.QamomileError, match="0 or 1"):
            qm.execute(artifact, on=endpoint, parameters={"value": value}, shots=4)
    else:
        result = qm.execute(
            artifact, on=endpoint, parameters={"value": value}, shots=4, seed=3
        )
        assert result.values == {(1, 1): 4}
        assert all(type(bit) is int for row in result.values for bit in row)


@pytest.mark.parametrize("backend", ["reference", "qiskit"])
def test_current_shot_integer_preparation_checks_each_logical_index(backend):
    """The first two indices prepare bits; the third fails the binary domain."""
    source = program("return qm.measure(qm.prepare(qm.get_current_shot()))")
    if backend == "reference":
        endpoint = qm.Reference()
    else:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    artifact = qm.compile(source, to=endpoint.target)
    result = qm.execute(artifact, on=endpoint, shots=2, seed=3)
    assert result.values == {0: 1, 1: 1}
    assert all(type(value) is int for value in result.values)
    with pytest.raises(qm.QamomileError, match="0 or 1"):
        qm.execute(artifact, on=endpoint, shots=3, seed=3)


@pytest.mark.parametrize("backend", ["reference", "qiskit"])
def test_unused_current_shot_preparation_checks_before_quantum_execution(backend):
    """Launch rows retain unused checks that depend only on the logical index."""
    source = program(
        "state = qm.prepare(False)\n"
        "unused = qm.prepare(qm.get_current_shot() + 2)\n"
        "state = qm.x(state)\n"
        "return qm.measure(state)"
    )
    if backend == "reference":
        endpoint = qm.Reference()
    else:
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    artifact = qm.compile(source, to=endpoint.target)
    with pytest.raises(qm.QamomileError, match="0 or 1"):
        qm.execute(artifact, on=endpoint, shots=2, seed=3)


def test_single_evaluation_preserves_mixed_binary_and_unbounded_values():
    """Value evaluation applies the declared Bit type without packing an Int."""
    source = program(
        "bit = qm.measure(qm.prepare(True))\n"
        "return bit, qm.get_current_shot() + 2 ** 80, True",
        result="tuple[qm.Shot[qm.Bit], qm.Shot[qm.Int], qm.Shot[qm.Bool]]",
    )
    result = qm.evaluate(source)
    assert result == (1, 2**80, True)
    assert tuple(type(value) for value in result) == (int, int, bool)
