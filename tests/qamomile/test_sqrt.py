"""Source square roots use native typing, evaluation, and resource algebra."""

import math

import pytest

import qamomile as qm

SOURCE = """
import qamomile as qm
@qm.qkernel
def root(value: qm.Param[qm.Real]) -> qm.Real:
    return qm.sqrt(value)
@qm.qkernel
def integer_root(value: qm.Param[qm.Int]) -> qm.Real:
    return qm.sqrt(value)
@qm.qkernel
def constant_root(value: qm.Const[qm.Real]) -> qm.Real:
    return qm.sqrt(value)
@qm.qkernel
def sample(value: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    angle = root(value) * 3.141592653589793 / 3.0
    return qm.measure(qm.ry(qm.qubits(1), angle))
@qm.qkernel
def constant_sample() -> qm.Shot[qm.Bits[1]]:
    angle = qm.sqrt(9.0) * 3.141592653589793 / 3.0
    return qm.measure(qm.ry(qm.qubits(1), angle))
@qm.qkernel
def shot_root() -> qm.Shot[qm.Real]:
    bit = qm.measure(qm.prepare(True))
    value = 9.0 if bit else 0.0
    return qm.sqrt(value)
@qm.qkernel
def iterations(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    for _ in range(qm.ceil(qm.sqrt(n))):
        state = qm.x(state)
    return qm.measure(state)
"""


def exercise_sqrt_backend(endpoint, *, seed=7, parameterized=True):
    """Reuse one persisted Param square-root artifact and fold the Const case."""
    module = qm.build_source(SOURCE)
    program = qm.Program.from_bytes(module.specialize("sample").to_bytes())
    from qamomile.persistence import deserialize, serialize

    if parameterized:
        artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
        for value, expected in [(0.0, (0,)), (9.0, (1,))]:
            assert qm.execute(
                artifact, on=endpoint, parameters={"value": value}, shots=8, seed=seed
            ).counts == {expected: 8}
        with pytest.raises(qm.ExecutionError, match="sqrt.*finite and nonnegative"):
            qm.execute(
                artifact, on=endpoint, parameters={"value": -1.0}, shots=8, seed=seed
            )
    else:
        with pytest.raises(qm.ArtifactError, match="Q7015.*parameterized_execution"):
            qm.compile(program, to=endpoint.target)
    assert qm.run(
        module.specialize("constant_sample"), on=endpoint, shots=8, seed=seed
    ).result.counts == {(1,): 8}


def test_sqrt_evaluation_and_persistence():
    """Const, Param, and Shot roots retain their stages and native semantics."""
    module = qm.build_source(SOURCE)
    root = qm.Program.from_bytes(module.specialize("root").to_bytes())
    assert "classical.sqrt" in root.inspect()
    for value in [0.0, -0.0, 2.0, float.fromhex("0x1.fffffffffffffp+1023")]:
        actual = qm.evaluate(root, value=value)
        assert actual == math.sqrt(value)
        assert math.copysign(1, actual) == math.copysign(1, math.sqrt(value))
    assert qm.evaluate(module.specialize("integer_root"), value=1 << 100) == 2.0**50
    with pytest.raises(qm.ExecutionError, match="binary64"):
        qm.evaluate(module.specialize("integer_root"), value=1 << 1024)
    assert qm.evaluate(module.specialize("constant_root", value=9.0)) == 3.0
    assert (
        "classical.sqrt" not in module.specialize("constant_root", value=9.0).inspect()
    )
    assert qm.run(
        module.specialize("shot_root"), on=qm.Reference(), shots=8
    ).result.values == {3.0: 8}
    exercise_sqrt_backend(qm.Reference())


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("-inf"), float("nan")])
def test_sqrt_domain_rejections(value):
    """Native domain failures are visible at Const and Param availability."""
    module = qm.build_source(SOURCE)
    with pytest.raises(qm.SpecializationError, match="Q3008.*finite and nonnegative"):
        module.specialize("constant_root", value=value)
    with pytest.raises(qm.ExecutionError):
        qm.evaluate(module.specialize("root"), value=value)


@pytest.mark.parametrize("expression", ["True", "1j", "[1.0]", "qm.qubits(1)"])
def test_sqrt_rejects_non_numeric_scalar_types(expression):
    """Only Int and Real scalar operands are accepted by source and native types."""
    with pytest.raises(qm.TypeCheckError, match="qm.sqrt expects Int or Real"):
        qm.build_source(
            f"import qamomile as qm\n@qm.qkernel\ndef invalid() -> qm.Real:\n    return qm.sqrt({expression})\n"
        )


def test_sqrt_rejects_host_calls_and_arbitrary_python_math():
    """The public spelling is source-only and does not execute Python libraries."""
    with pytest.raises(RuntimeError, match="source intrinsic"):
        qm.sqrt(9.0)
    with pytest.raises(qm.FrontendError, match="Q5031"):
        qm.build_source(
            "import qamomile as qm\nimport math\n@qm.qkernel\ndef invalid() -> qm.Real:\n    return math.sqrt(9.0)\n"
        )
    with pytest.raises(qm.FrontendError, match="Q5105"):
        qm.build_source(
            "import qamomile as qm\n@qm.qkernel\ndef invalid() -> qm.Real:\n    return qm.sqrt(1, 2)\n"
        )


def test_sqrt_resource_expression_is_retained_and_restored():
    """Const-dependent iteration counts retain the existing square-root algebra."""
    study = qm.resources.logical(qm.build_source(SOURCE)["iterations"])
    metric = qm.resources.logical_operation_count("quantum.x")
    count = study.metric(metric)
    assert count.expression is not None
    assert "square_root" in str(count.expression)
    restored = qm.resources.ResourceStudy.from_bytes(study.to_bytes())
    assert restored.to_bytes() == study.to_bytes()
    for n, expected in [(0, 0), (9, 3), (10, 4), (1_000_000, 1000)]:
        assert restored.evaluate(n=n).metric(metric).expression.numeric() == expected
