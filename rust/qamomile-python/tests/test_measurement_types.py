"""Native observation and source-preparation queries fail closed."""

import pytest

import qamomile as qm
from qamomile import types
from qamomile._compiler import hir_wire
from qamomile._compiler.hir import HIRExpression
from qamomile._native import compiler as native
from qamomile.ir import SourceSpan, attributes
from qamomile.proto import hir_pb2, qcore_pb2
from qamomile.serialization import decode_type, encode_type


def encoded_type(value):
    """Serialize a canonical native type query."""
    encoded = qcore_pb2.TypeProto()
    encode_type(types.as_type(value), encoded)
    return encoded.SerializeToString(deterministic=True)


def operand(opcode="literal", type_spec=qm.Int, value=1):
    """Create a source operand with the same canonical HIR transport."""
    expression = HIRExpression(
        opcode,
        type_spec,
        attributes=attributes(value=value),
        span=SourceSpan("test.py", 1, 0, 1, 1),
    )
    encoded = hir_pb2.HIRExpressionProto()
    hir_wire._encode_expression(expression, encoded)
    return encoded


def query_preparation(expression):
    """Decode the checked native preparation type."""
    encoded = qcore_pb2.TypeProto()
    encoded.ParseFromString(
        native.infer_preparation_result_type(
            expression.SerializeToString(deterministic=True)
        )
    )
    return decode_type(encoded)


@pytest.mark.parametrize(
    "payload,expected",
    [
        (qm.Bool, qm.Bit),
        (qm.Bit, qm.Bit),
        (qm.Bits[1], qm.Bits[1]),
        (qm.Bits["n"], qm.Bits["n"]),
        (qm.UInt[3], qm.UInt[3]),
        (qm.UFixed[1, 2], qm.UFixed[1, 2]),
        (
            tuple[qm.Bool, tuple[qm.UInt[3], qm.Bool], qm.Bits[1]],
            tuple[qm.Bit, tuple[qm.UInt[3], qm.Bit], qm.Bits[1]],
        ),
    ],
)
def test_native_observation_preserves_shapes_and_normalizes_only_bool(
    payload, expected
):
    """Nested admitted products and unresolved widths retain their meaning."""
    assert native.infer_measurement_observation_type(
        encoded_type(payload)
    ) == encoded_type(expected)


@pytest.mark.parametrize(
    "payload", [qm.Int, qm.Real, qm.Vector[qm.Bool, 2], qm.Q[qm.Bool], qm.Shot[qm.Bool]]
)
def test_native_observation_does_not_broaden_coherent_admissibility(payload):
    """Noncoherent payloads remain outside measurement projection."""
    with pytest.raises(ValueError):
        native.infer_measurement_observation_type(encoded_type(payload))


def test_native_preparation_preserves_boolean_and_integer_logical_types():
    """Scalar integers become Bit while Boolean state payloads stay Boolean."""
    for stage in (qm.Int, qm.Const[qm.Int], qm.Param[qm.Int], qm.Shot[qm.Int]):
        assert query_preparation(operand(type_spec=stage)) == qm.Q[qm.Bit]
    assert query_preparation(operand(type_spec=qm.Bool, value=True)) == qm.Q[qm.Bool]
    assert query_preparation(operand("module_constant")) == qm.Q[qm.Bit]


@pytest.mark.parametrize(
    "type_spec,value",
    [
        (qm.Int, True),
        (qm.Bool, 1),
        (qm.Real, 1.0),
    ],
)
def test_native_preparation_rejects_nonbinary_or_misdeclared_literals(type_spec, value):
    """A caller cannot turn arbitrary or staged input into a binary literal."""
    with pytest.raises(ValueError):
        query_preparation(operand(type_spec=type_spec, value=value))


@pytest.mark.parametrize(
    "query",
    [native.infer_measurement_observation_type, native.infer_preparation_result_type],
)
def test_native_queries_reject_malformed_wire(query):
    """Malformed or empty encoded requests never produce a type."""
    for value in (b"", b"\xff", b"\x0a\x80"):
        with pytest.raises(ValueError):
            query(value)


@pytest.mark.parametrize(
    "name,invoke",
    [
        (
            "infer_measurement_observation_type",
            lambda: types.infer_measurement_observation_type(qm.Bool),
        ),
        (
            "infer_preparation_result_type",
            lambda: hir_wire.infer_preparation_result_type(
                HIRExpression(
                    "literal",
                    qm.Int,
                    attributes=attributes(value=1),
                    span=SourceSpan("test.py", 1, 0, 1, 1),
                )
            ),
        ),
    ],
)
def test_python_query_adapters_fail_closed(monkeypatch, name, invoke):
    """Missing or malformed native capabilities have no Python fallback."""
    monkeypatch.setattr(native, name, None)
    with pytest.raises(TypeError, match="unavailable"):
        invoke()
    monkeypatch.setattr(native, name, lambda *_: b"bad")
    with pytest.raises(RuntimeError, match="malformed"):
        invoke()


@pytest.mark.parametrize("tamper", ["result", "literal_type"])
def test_native_hir_rechecks_preparation_types(monkeypatch, tamper):
    """Resealed source HIR cannot forge the preparation payload type."""
    import hashlib

    captured = []
    lower = native.lower_hir

    def capture(payload):
        """Retain the original canonical source request before lowering."""
        captured.append(payload)
        return lower(payload)

    monkeypatch.setattr(native, "lower_hir", capture)
    qm.build_source(
        "import qamomile as qm\n@qm.qkernel\ndef sample() -> qm.Shot[qm.Bit]:\n    return qm.measure(qm.prepare(1))\n"
    )
    package = hir_pb2.HIRPackageProto.FromString(captured[0])
    preparation = package.functions[0].body.statements[0].expression.operands[0]
    if tamper == "result":
        preparation.type.arguments[0].name = "Bool"
    else:
        preparation.operands[0].type.name = "Bool"
    package.package_digest = b""
    package.package_digest = hashlib.sha256(
        package.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(ValueError):
        native.inspect_hir(package.SerializeToString(deterministic=True))
