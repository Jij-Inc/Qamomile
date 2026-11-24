"""Shared exact measurement decoding on every available local target."""

import importlib

import pytest

import qamomile as qm


@pytest.fixture(
    params=[
        ("reference", "qamomile", "Reference"),
        ("qiskit_aer", "qamomile.qiskit", "Aer"),
        pytest.param(
            ("cirq", "qamomile.cirq", "CirqSimulator"), marks=pytest.mark.cirq
        ),
        pytest.param(
            ("qulacs", "qamomile.quri_parts", "Qulacs"), marks=pytest.mark.quri_parts
        ),
        pytest.param(
            ("braket", "qamomile.braket", "BraketLocal"), marks=pytest.mark.braket
        ),
        pytest.param(
            ("cudaq", "qamomile.cudaq", "CudaqLocal"), marks=pytest.mark.cudaq
        ),
        pytest.param(("selene_sim", "qamomile.hugr", "Selene"), marks=pytest.mark.hugr),
        pytest.param(
            ("pyqret", "qamomile.quration", "QurationLocal"), marks=pytest.mark.quration
        ),
    ]
)
def endpoint(request):
    """Load each optional SDK only in its explicitly selected test process."""
    dependency, module, constructor = request.param
    if dependency != "reference":
        pytest.importorskip(dependency)
    return getattr(importlib.import_module(module), constructor)()


@pytest.mark.parametrize(
    "expression,annotation,expected",
    [
        ("qm.measure(qm.prepare(True))", "qm.Bit", 1),
        ("qm.measure(qm.prepare(0))", "qm.Bit", 0),
        ("qm.measure(qm.prepare(1))", "qm.Bit", 1),
        ("qm.measure(qm.prepare((1,)))", "qm.Bits[1]", (1,)),
        ("qm.measure(qm.as_uint(qm.prepare((1, 0, 1))))", "qm.UInt[3]", 5),
        (
            "qm.measure(qm.as_fixed(qm.prepare((1, 0, 1)), integer_bits=1))",
            "qm.UFixed[1, 2]",
            1.25,
        ),
        ("bool(qm.measure(qm.prepare(True)))", "qm.Bool", True),
    ],
)
def test_binary_observations_preserve_container_and_numeric_types(
    endpoint, expression, annotation, expected
):
    """The same sources decode scalar Bit, Bits, UInt, UFixed and Bool distinctly."""
    program = qm.build_source(
        f"import qamomile as qm\n@qm.qkernel\ndef sample() -> qm.Shot[{annotation}]:\n    return {expression}\n"
    ).specialize("sample")
    artifact = qm.compile(program, to=endpoint.target)
    result = qm.execute(
        artifact,
        on=endpoint,
        shots=4,
        seed=0 if endpoint.name == "quration.local" else None,
    )
    assert result.values == {expected: 4}
    observed = next(iter(result.values))
    assert type(observed) is type(expected)
    assert artifact.result_abi.type == program.entry.result_type
