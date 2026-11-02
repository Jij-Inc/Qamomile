"""Persistent Module, Program, and artifact tests."""

from __future__ import annotations

import hashlib
import subprocess
import sys

import pytest

import qamomile as qm
from qamomile.persistence import deserialize, serialize
from qamomile.types import FunctionType

from .conftest import BELL_SOURCE, MODULAR_CLEANUP_SOURCE


def test_module_and_program_bytes_are_canonical() -> None:
    """Repeated builds and decode/encode cycles must produce identical bytes."""

    first = qm.build_source(BELL_SOURCE, module_name="stable")
    second = qm.build_source(BELL_SOURCE, module_name="stable")
    assert first.to_bytes() == second.to_bytes()

    program = first.specialize("bell")
    assert qm.Module.from_bytes(first.to_bytes()).to_bytes() == first.to_bytes()
    assert qm.Program.from_bytes(program.to_bytes()).to_bytes() == program.to_bytes()


def test_qcore_persistence_requires_native_sealing_and_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python never hashes or parses a package before the Rust wire policy."""

    import importlib

    serialization = importlib.import_module("qamomile.serialization")
    module = qm.build_source(BELL_SOURCE, module_name="native_persistence")
    original_seal = serialization.native.seal_qcore_package
    original_inspect = serialization.native.inspect_qcore_envelope
    sealed_calls = 0
    inspected_calls = 0

    def track_seal(payload: bytes) -> object:
        """Record and forward one native package seal."""

        nonlocal sealed_calls
        sealed_calls += 1
        return original_seal(payload)

    def track_inspection(payload: bytes) -> object:
        """Record and forward one native preflight inspection."""

        nonlocal inspected_calls
        inspected_calls += 1
        return original_inspect(payload)

    monkeypatch.setattr(serialization.native, "seal_qcore_package", track_seal)
    monkeypatch.setattr(
        serialization.native,
        "inspect_qcore_envelope",
        track_inspection,
    )
    encoded = module.to_bytes()
    assert qm.Module.from_bytes(encoded).to_bytes() == encoded
    assert sealed_calls > 0
    assert inspected_calls == 1

    monkeypatch.setattr(serialization.native, "seal_qcore_package", None)
    with pytest.raises(qm.SerializationError, match="Q4002"):
        module.to_bytes()

    monkeypatch.setattr(serialization.native, "seal_qcore_package", original_seal)
    monkeypatch.setattr(serialization.native, "inspect_qcore_envelope", None)
    with pytest.raises(qm.SerializationError, match="Q4002"):
        qm.Module.from_bytes(encoded)

    monkeypatch.setattr(
        serialization.native,
        "inspect_qcore_envelope",
        lambda _payload: (5, 0, "qamomile.module", b"short"),
    )
    with pytest.raises(qm.SerializationError, match="Q4008"):
        qm.Module.from_bytes(encoded)


def test_semantic_digest_erases_debug_only_source_changes() -> None:
    """Comments and unused imports change debug bytes but not program meaning."""

    source = """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
"""
    with_debug_change = "import math\n# Debug-only context.\n" + source
    first = qm.build_source(source, module_name="semantic")
    second = qm.build_source(with_debug_change, module_name="semantic")

    assert first.to_bytes() != second.to_bytes()
    assert first.semantic_digest == second.semantic_digest
    assert (
        first.semantic_digest
        == "13daffa35a02ff9ba070a69d2176aac480deceb311e291c65a0cde65aa69330b"
    )
    assert (
        first["answer"].semantic_digest
        == "17db7c219c31c31fa022c20ba38e5e2563cafdfe5a3c9c6d3b6221ec3136c036"
    )
    assert (
        first.specialize("answer").semantic_digest
        == "978575ae21c49b2ee53b46e73e9b69f2330440923c607789125147a0e2867b03"
    )


def test_semantic_digest_requires_the_native_identity_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Semantic identity has no Python projection or hash fallback."""

    import importlib

    ir = importlib.import_module("qamomile.ir")
    module = qm.build_source(BELL_SOURCE, module_name="native_identity")
    native_digest = ir.native.module_semantic_digest
    calls = 0

    def record_native_digest(payload: bytes) -> bytes:
        """Record and forward one native semantic digest request."""

        nonlocal calls
        calls += 1
        return native_digest(payload)

    monkeypatch.setattr(ir.native, "module_semantic_digest", record_native_digest)
    digest = module.semantic_digest
    assert len(digest) == 64
    assert calls == 1

    monkeypatch.setattr(ir.native, "module_semantic_digest", None)
    with pytest.raises(qm.VerificationError, match="Q4002"):
        _ = module.semantic_digest

    monkeypatch.setattr(ir.native, "module_semantic_digest", lambda _payload: b"short")
    with pytest.raises(qm.VerificationError, match="Q4008"):
        _ = module.semantic_digest


def test_qcore_wire_omits_body_derived_contract_caches() -> None:
    """Persistent QCore stores assertions and bodies, not synchronized summaries."""

    from qamomile.proto import qcore_pb2 as wire

    function_fields = wire.FunctionProto.DESCRIPTOR.fields_by_name
    operation_fields = wire.OperationProto.DESCRIPTOR.fields_by_name

    assert {"kind", "effects", "capabilities", "requirements"}.isdisjoint(
        function_fields
    )
    assert "effects" not in operation_fields


def test_type_decoder_enforces_the_closed_semantic_universe() -> None:
    """Unknown scalar names and empty array dimensions fail at the wire boundary."""

    from qamomile.proto import qcore_pb2 as wire
    from qamomile.serialization import decode_type

    unknown = wire.TypeProto(kind=wire.TypeProto.SCALAR, name="Atom")
    empty_array = wire.TypeProto(kind=wire.TypeProto.ARRAY, name="Array")
    empty_array.arguments.add(kind=wire.TypeProto.SCALAR, name="Int")

    with pytest.raises(qm.SerializationError, match="invalid semantic type"):
        decode_type(unknown)
    with pytest.raises(qm.SerializationError, match="invalid semantic type"):
        decode_type(empty_array)


def test_function_type_has_one_canonical_transient_wire_form() -> None:
    """N-ary QKernel signatures round-trip while malformed arity is rejected."""

    from qamomile.proto import qcore_pb2 as wire
    from qamomile.serialization import decode_type, encode_type

    expected = qm.QKernel[qm.Q[qm.Bits[2]], qm.Real, qm.Q[qm.Bits[2]]]
    encoded = wire.TypeProto()
    encode_type(expected, encoded)

    assert encoded.kind == wire.TypeProto.FUNCTION
    assert decode_type(encoded) == expected
    assert encoded.SerializeToString(deterministic=True) == encoded.SerializeToString(
        deterministic=True
    )

    nullary = qm.QKernel[qm.Real]
    encoded_nullary = wire.TypeProto()
    encode_type(nullary, encoded_nullary)
    assert len(encoded_nullary.arguments) == 1
    assert decode_type(encoded_nullary) == nullary

    labeled = FunctionType((qm.Int, qm.Real), qm.Bool, ("left", "right"))
    encoded_labeled = wire.TypeProto()
    encode_type(labeled, encoded_labeled)
    assert tuple(encoded_labeled.field_names) == ("left", "right")
    assert decode_type(encoded_labeled) == labeled

    malformed_labels = wire.TypeProto(kind=wire.TypeProto.FUNCTION)
    malformed_labels.arguments.add(kind=wire.TypeProto.SCALAR, name="Int")
    malformed_labels.arguments.add(kind=wire.TypeProto.SCALAR, name="Bool")
    malformed_labels.field_names.extend(("left", "extra"))
    with pytest.raises(qm.SerializationError, match="parameter names"):
        decode_type(malformed_labels)

    malformed = wire.TypeProto(kind=wire.TypeProto.FUNCTION)
    with pytest.raises(qm.SerializationError, match="expects an output"):
        decode_type(malformed)


def test_function_types_cannot_leak_into_persistent_qcore() -> None:
    """Native lowering rejects an uninstantiated higher-order declaration."""

    source = """
import qamomile as qm

@qm.qkernel
def unresolved(operation: qm.QKernel[qm.Int, qm.Int]) -> qm.Int:
    return 1
"""
    with pytest.raises(qm.CapabilityError, match="without a closed call site"):
        qm.build_source(source)


def test_program_deserializes_and_executes_in_fresh_process() -> None:
    """Program bytes must need neither source text nor a Python callable."""

    payload = qm.build_source(BELL_SOURCE, module_name="isolated").specialize("bell")
    script = """
import sys
import qamomile as qm

program = qm.Program.from_bytes(sys.stdin.buffer.read())
result = qm.run(program, on=qm.Reference(), shots=16, seed=9).result
print(sorted(result.counts))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=payload.to_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert completed.stdout.decode().strip() == "[(0, 0), (1, 1)]"


def test_modular_cleanup_round_trips_and_executes_in_fresh_process() -> None:
    """Empty gate attributes and callable adjoints survive the wire boundary."""

    payload = qm.build_source(
        MODULAR_CLEANUP_SOURCE,
        module_name="modular_isolated",
    ).specialize("modular_cleanup")
    restored = qm.Program.from_bytes(payload.to_bytes())
    assert restored.to_bytes() == payload.to_bytes()
    script = """
import sys
import qamomile as qm

program = qm.Program.from_bytes(sys.stdin.buffer.read())
result = qm.run(program, on=qm.Reference(), shots=4, seed=11).result
print(dict(result.counts))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=payload.to_bytes(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    assert completed.stdout.decode().strip() == "{(1, 0, 0): 4}"


def test_artifact_detects_tampering() -> None:
    """Artifacts integrity-check payload and ABI metadata together."""

    program = qm.build_source(BELL_SOURCE).specialize("bell")
    encoded = bytearray(serialize(qm.compile(program, to=qm.Reference())))
    encoded[-1] ^= 1
    with pytest.raises(qm.ArtifactError):
        deserialize(encoded)


def test_artifact_persistence_requires_native_sealing_and_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python never hashes or parses an artifact before the Rust wire policy."""

    import importlib

    compilation = importlib.import_module("qamomile.compilation")
    artifact = qm.compile(
        qm.build_source(BELL_SOURCE, module_name="native_artifact").specialize("bell"),
        to=qm.Reference(),
    )
    original_seal = compilation.native.seal_artifact
    original_inspect = compilation.native.inspect_artifact
    sealed_calls = 0
    inspected_calls = 0

    def track_seal(payload: bytes) -> object:
        """Record and forward one native artifact seal."""

        nonlocal sealed_calls
        sealed_calls += 1
        return original_seal(payload)

    def track_inspection(payload: bytes) -> object:
        """Record and forward one native artifact inspection."""

        nonlocal inspected_calls
        inspected_calls += 1
        return original_inspect(payload)

    monkeypatch.setattr(compilation.native, "seal_artifact", track_seal)
    monkeypatch.setattr(
        compilation.native,
        "inspect_artifact",
        track_inspection,
    )
    encoded = serialize(artifact)
    assert serialize(deserialize(encoded)) == encoded
    assert sealed_calls == 2
    assert inspected_calls == 1

    monkeypatch.setattr(compilation.native, "seal_artifact", None)
    with pytest.raises(qm.ArtifactError, match="Q7004"):
        serialize(artifact)

    monkeypatch.setattr(compilation.native, "seal_artifact", original_seal)
    monkeypatch.setattr(compilation.native, "inspect_artifact", None)
    with pytest.raises(qm.ArtifactError, match="Q7004"):
        deserialize(encoded)

    monkeypatch.setattr(
        compilation.native,
        "inspect_artifact",
        lambda _payload: (5, 0, "qamomile.qcore.program", "reference-v1", b"short"),
    )
    with pytest.raises(qm.ArtifactError, match="Q7004"):
        deserialize(encoded)


def test_artifact_decoder_rejects_noncanonical_and_unknown_contracts() -> None:
    """A valid digest cannot authorize alternate metadata or requirements."""

    from qamomile import _qamomile_native as native
    from qamomile.proto import qcore_pb2 as wire

    artifact = qm.compile(
        qm.build_source(BELL_SOURCE, module_name="artifact_contract").specialize(
            "bell"
        ),
        to=qm.Reference(),
    )
    encoded = wire.ArtifactProto.FromString(serialize(artifact))
    del encoded.metadata[:]
    for name in ("z", "a"):
        encoded.metadata.add(name=name).value.int_value = 1
    encoded.artifact_digest = b""
    encoded.artifact_digest = hashlib.sha256(
        encoded.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(qm.ArtifactError, match="Q7023"):
        deserialize(encoded.SerializeToString(deterministic=True))

    encoded = wire.ArtifactProto.FromString(serialize(artifact))
    encoded.execution_requirements.append("future_runtime")
    encoded.execution_requirements.sort()
    encoded.artifact_digest = b""
    encoded.artifact_digest = hashlib.sha256(
        encoded.SerializeToString(deterministic=True)
    ).digest()
    with pytest.raises(qm.ArtifactError, match="Q7013"):
        deserialize(encoded.SerializeToString(deterministic=True))

    encoded = wire.ArtifactProto.FromString(serialize(artifact))
    encoded.target_fingerprint = "x" * (native.MAX_QCORE_STRING_BYTES + 1)
    encoded.artifact_digest = b""
    with pytest.raises(qm.ArtifactError, match="string exceeds the size limit"):
        deserialize(encoded.SerializeToString(deterministic=True))


def test_decoder_rejects_resealed_noncanonical_function_order() -> None:
    """A recomputed checksum cannot turn alternate wire ordering into an image."""

    from qamomile.proto import qcore_pb2 as wire

    encoded = qm.build_source(
        MODULAR_CLEANUP_SOURCE,
        module_name="noncanonical",
    ).to_bytes()
    package = wire.QCorePackageProto.FromString(encoded)
    functions = list(package.functions)
    del package.functions[:]
    package.functions.extend(reversed(functions))
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()
    forged = package.SerializeToString(deterministic=True)

    with pytest.raises(qm.SerializationError, match="canonically encoded"):
        qm.Module.from_bytes(forged)


def test_program_decoder_requires_an_exact_qualified_entrypoint() -> None:
    """A local suffix cannot alias the exact closed Program entrypoint."""

    from qamomile.proto import qcore_pb2 as wire

    encoded = (
        qm.build_source(
            BELL_SOURCE,
            module_name="exact_entrypoint",
        )
        .specialize("bell")
        .to_bytes()
    )
    package = wire.QCorePackageProto.FromString(encoded)
    package.entrypoint = "bell"
    package.package_digest = b""
    canonical = package.SerializeToString(deterministic=True)
    package.package_digest = hashlib.sha256(canonical).digest()

    with pytest.raises(qm.VerificationError, match="Q1015"):
        qm.Program.from_bytes(package.SerializeToString(deterministic=True))


def test_decoder_rejects_oversized_qcore_strings_before_semantic_decode() -> None:
    """Persistent QCore enforces structural budgets on untrusted payloads."""

    from qamomile import _qamomile_native as native
    from qamomile.proto import qcore_pb2 as wire

    encoded = qm.build_source(BELL_SOURCE, module_name="bounded").to_bytes()
    package = wire.QCorePackageProto.FromString(encoded)
    package.module_name = "x" * (native.MAX_QCORE_STRING_BYTES + 1)
    forged = package.SerializeToString(deterministic=True)

    with pytest.raises(qm.SerializationError, match="string exceeds the size limit"):
        qm.Module.from_bytes(forged)


@pytest.mark.parametrize("value", [2**130 + 17, -(2**130) - 17])
def test_arbitrary_precision_integer_round_trips(value: int) -> None:
    """The portable Int contract must not silently narrow at the wire boundary."""

    source = f"""
import qamomile as qm

@qm.qkernel
def huge() -> qm.Int:
    return {value}
"""
    program = qm.build_source(source, module_name="bigint").specialize("huge")
    restored = qm.Program.from_bytes(program.to_bytes())

    assert restored.to_bytes() == program.to_bytes()
    assert qm.evaluate(restored) == value


@pytest.mark.parametrize("encoded", [b"", b"\x00\x80", b"\xff\x7f", b"\x01"])
def test_decoder_rejects_noncanonical_bigint_literals(encoded: bytes) -> None:
    """Big integers have one minimal encoding and cannot duplicate sint64."""

    from qamomile.proto import qcore_pb2 as wire
    from qamomile.serialization import decode_literal

    with pytest.raises(qm.SerializationError, match="bigint literal"):
        decode_literal(wire.LiteralProto(bigint_value=encoded))


def _annotated_module(
    rows: tuple[tuple[str, object], ...],
) -> tuple[qm.Module, qm.Module]:
    """Return the historical ``answer`` module and a copy carrying intent rows."""

    from qamomile.serialization import (
        _parse_sealed_package,
        _seal_package,
        encode_literal,
    )

    source = """
import qamomile as qm

@qm.qkernel
def answer() -> qm.Int:
    return 42
"""
    plain = qm.build_source(source, module_name="semantic")
    package = _parse_sealed_package(plain.to_bytes())
    function = package.functions[0]
    for name, value in rows:
        encode_literal(value, function.intent.add(name=name).value)
    operation = function.body.operations[0]
    encode_literal(("a", 1), operation.intent.add(name="note").value)
    return plain, qm.Module.from_bytes(_seal_package(package))


def test_intent_rows_persist_print_and_erase_from_identity() -> None:
    """Intent is retained and visible but never part of semantic identity."""

    rows = (("origin", "qamomile.stdlib.example"), ("doc", "Answer everything."))
    plain, annotated = _annotated_module(rows)

    assert annotated.functions[0].intent == rows
    assert annotated.functions[0].body.operations[0].intent == (("note", ("a", 1)),)
    assert "intent{origin='qamomile.stdlib.example', doc='Answer everything.'}" in (
        annotated.inspect()
    )
    assert "qcore.constant() {value=42} intent{note=('a', 1)}" in annotated.inspect()
    assert "intent{" not in plain.inspect()
    assert annotated.to_bytes() != plain.to_bytes()
    assert qm.Module.from_bytes(annotated.to_bytes()) == annotated
    assert (
        annotated.semantic_digest
        == plain.semantic_digest
        == "13daffa35a02ff9ba070a69d2176aac480deceb311e291c65a0cde65aa69330b"
    )

    program = annotated.specialize("answer")
    assert program.entry.intent == rows
    assert program.semantic_digest == plain.specialize("answer").semantic_digest
    artifact = qm.compile(program, to=qm.Reference())
    assert qm.execute(artifact, on=qm.Reference()).value == 42


def test_hir_function_intent_is_carried_into_qcore() -> None:
    """Rust lowering copies HIR function intent onto the QCore function."""

    from dataclasses import replace

    from qamomile._compiler.frontend import SourceUnit, parse_source
    from qamomile._compiler.lowering import lower_hir

    hir = parse_source(
        SourceUnit.from_text(
            "import qamomile as qm\n\n@qm.qkernel\ndef answer() -> qm.Int:\n    return 7\n",
            module_name="carried",
        )
    )
    rows = (("origin", "carried.answer"), ("doc", "Seven."))
    annotated = replace(
        hir,
        functions=tuple(replace(function, intent=rows) for function in hir.functions),
    )

    assert lower_hir(annotated).function("answer").intent == rows
    assert lower_hir(hir).function("answer").intent == ()
    assert lower_hir(annotated).semantic_digest == lower_hir(hir).semantic_digest
