"""Single-invocation quantum execution and typed per-shot results."""

import math

import pytest

import qamomile as qm
from qamomile.compilation import ResultABI
from qamomile.persistence import deserialize, serialize

SOURCE = """
import qamomile as qm

@qm.qkernel
def score(bit: qm.Shot[qm.Bool]) -> qm.Shot[qm.Real]:
    return -1.0 if bit else 1.0

@qm.qkernel
def sample(theta: qm.Param[qm.Real]) -> tuple[qm.Shot[qm.Real], tuple[qm.Shot[qm.Real], qm.Shot[qm.Bool]]]:
    angle = theta / 4.0
    q = qm.prepare(False)
    for outer in range(2):
        for inner in range(2):
            q = qm.ry(q, angle)
    bit = qm.measure(q)
    e = score(bit)
    return e, (e * e, bit)

@qm.qkernel
def feedback() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    q = qm.h(qm.prepare(False))
    first, q = qm.measure_z(q, 0)
    if first:
        q = qm.x(q)
    return first, qm.measure(q)
"""


@pytest.fixture(params=["reference", "aer"])
def endpoint(request):
    """Use the native interpreter and the SDK emission path."""
    if request.param == "reference":
        return qm.Reference()
    pytest.importorskip("qiskit_aer")
    from qamomile.qiskit import Aer

    return Aer()


def test_nested_per_shot_results_round_trip_and_rebind(endpoint):
    """One-shot arithmetic survives helpers, nested loops and persistence."""
    program = qm.build_source(SOURCE).specialize("sample")
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
    assert artifact.result_abi.width == 129
    for theta, expected in [(0.0, (1.0, (1.0, False))), (math.pi, (-1.0, (1.0, True)))]:
        result = qm.execute(
            artifact, on=endpoint, parameters={"theta": theta}, shots=32, seed=7
        )
        assert isinstance(result, qm.SampleResult)
        assert result.values == {expected: 32}
        assert result.shots == 32
        snapshot = qm.ExecutionSnapshot.completed(result)
        restored = qm.ExecutionSnapshot.from_bytes(snapshot.to_bytes())
        assert restored.result.values == result.values

    result = qm.execute(
        artifact, on=endpoint, parameters={"theta": math.pi / 2}, shots=128, seed=7
    )
    assert set(result.values) == {
        (1.0, (1.0, False)),
        (-1.0, (1.0, True)),
    }
    assert sum(result.values.values()) == 128


def test_measurement_feedback_remains_inside_one_execution(endpoint):
    """A measured state can continue in the same shot without ensemble replay."""
    program = qm.build_source(SOURCE).specialize("feedback")
    result = qm.run(program, on=endpoint, shots=128, seed=11).result
    assert isinstance(result, qm.SampleResult)
    assert set(result.values) == {(False, False), (True, False)}


@pytest.mark.parametrize("value", [0.0, -0.0, -3.5, 5e-324, 1.7976931348623157e308])
def test_real_sample_codec_preserves_binary64(value):
    """Finite sample values retain sign, subnormals and the full finite range."""
    abi = ResultABI(qm.Shot[qm.Real], tuple(range(64)), 64)
    bits = abi.encode_value(value)
    decoded = abi.decode_value(bits)
    assert decoded == value
    assert math.copysign(1.0, decoded) == math.copysign(1.0, value)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_real_sample_codec_rejects_nonfinite_values(value):
    """Nonfinite values fail both encoding and externally supplied decoding."""
    import struct

    abi = ResultABI(qm.Shot[qm.Real], tuple(range(64)), 64)
    with pytest.raises(qm.ExecutionError, match="Q8143"):
        abi.encode_value(value)
    raw = int.from_bytes(struct.pack("<d", value), "little")
    with pytest.raises(qm.ExecutionError, match="Q8143"):
        abi.decode_value(tuple((raw >> index) & 1 for index in range(64)))


def test_expval_is_not_a_source_operation():
    """Removed ensemble syntax fails during source compilation."""
    with pytest.raises(qm.FrontendError):
        qm.build_source("""
import qamomile as qm
@qm.qkernel
def removed(H: qm.Const[qm.Observable]) -> qm.Real:
    return qm.expval(qm.qubits(1), H)
""")


def test_persistent_qcore_rejects_removed_expval_opcode():
    """Old ensemble operations cannot bypass source checks through persistence."""
    import hashlib

    from qamomile.proto import qcore_pb2 as wire

    program = qm.build_source("""
import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.qubits(1))
""").specialize("sample")
    package = wire.QCorePackageProto.FromString(program.to_bytes())
    measurement = next(
        operation
        for operation in package.functions[0].body.operations
        if operation.opcode == "quantum.measure"
    )
    measurement.opcode = "quantum.expval"
    package.package_digest = b""
    package.package_digest = hashlib.sha256(
        package.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(qm.VerificationError, match="Q1032.*quantum.expval"):
        qm.Program.from_bytes(package.SerializeToString(deterministic=True))
