"""Independent compiler extensions reuse defaults and preserve source authority."""

import importlib.util
from dataclasses import replace

import pytest

import qamomile as qm
from qamomile.compilation import (
    OptimizationContext,
    OptimizationPipeline,
    OptimizationProgram,
    StandardSimplification,
)
from qamomile.ir import FrozenMapping
from qamomile.persistence import deserialize, export_payload, serialize

SOURCE = """import qamomile as qm

@qm.qkernel
def sample() -> qm.Bits[1]:
    state = qm.qubits(1)
    state = qm.h(state)
    state = qm.z(state)
    state = qm.h(state)
    return qm.measure(state)
"""


class Observer:
    """Observe a pipeline without modifying its input or owning compilation."""

    name = "example.observer"
    version = "1"
    configuration = b"observe"

    def __init__(self):
        self.calls = []

    def run(self, program, context):
        self.calls.append((program.to_bytes(), context))
        return program


class Conjugation(Observer):
    """Reuse native traversal to contract a phase-sensitive unary rule."""

    name = "example.conjugation"
    configuration = b"HZH=X:v1"

    def run(self, program, context):
        self.calls.append((program.to_bytes(), context))
        return program.rewrite(
            ("quantum.h", "quantum.z", "quantum.h"), ("quantum.x",), context=context
        )


def _program():
    """Build the small phase-sensitive rewrite witness."""
    return qm.build_source(SOURCE, module_name="compiler_extension").specialize(
        "sample"
    )


def test_external_extension_adds_one_rule_without_a_backend_or_protobuf(tmp_path):
    """An independently imported package contracts HZH to X using native edits."""
    module_file = tmp_path / "extension.py"
    module_file.write_text("""
class HadamardConjugation:
    name = "example.hadamard_conjugation"
    version = "1"
    configuration = b"HZH=X:v1"

    def run(self, program, context):
        return program.rewrite(
            ("quantum.h", "quantum.z", "quantum.h"),
            ("quantum.x",), context=context,
        )
""")
    spec = importlib.util.spec_from_file_location("independent_extension", module_file)
    assert spec is not None and spec.loader is not None
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    optimizer = OptimizationPipeline.standard().append(extension.HadamardConjugation())
    source = _program()
    artifact = qm.compile(source, to=qm.Reference(), optimizer=optimizer)
    optimized = qm.Program.from_bytes(export_payload(artifact))

    assert optimized.inspect().count("quantum.x") == 1
    assert "quantum.h" not in optimized.inspect()
    assert source.inspect().count("quantum.h") == 2
    assert artifact.source_digest.hex() == source.semantic_digest
    assert artifact.report.resources._semantic_source().to_bytes() == source.to_bytes()
    assert artifact.report.stages[:2] == (
        "rust-qcore-standard-optimization",
        "optimization:example.hadamard_conjugation@1",
    )
    assert qm.execute(artifact, on=qm.Reference(), shots=4).counts == {(1,): 4}
    restored = deserialize(serialize(artifact))
    assert restored.execution_identity == artifact.execution_identity
    assert qm.execute(restored, on=qm.Reference(), shots=4).counts == {(1,): 4}


def test_partial_replacement_and_empty_pipeline_do_not_rerun_standard():
    """Explicit pipelines replace only optional optimization, never verification."""
    program = qm.build_source(
        SOURCE.replace("qm.z(state)", "qm.h(state)"), module_name="compiler_extension"
    ).specialize("sample")
    base = OptimizationPipeline.standard()
    observer = Observer()
    replacement = base.replace("qcore.simplify", observer)
    selected = qm.compile(program, to=qm.Reference(), optimizer=replacement)
    empty = qm.compile(program, to=qm.Reference(), optimizer=OptimizationPipeline())
    ordinary = qm.compile(program, to=qm.Reference())

    assert len(base.passes) == 1 and isinstance(base.passes[0], StandardSimplification)
    assert len(observer.calls) == 1
    assert export_payload(selected) == export_payload(empty) == program.to_bytes()
    assert export_payload(ordinary) != program.to_bytes()
    assert "rust-qcore-standard-optimization" not in selected.report.stages
    assert "rust-qcore-standard-optimization" not in empty.report.stages
    assert qm.execute(selected, on=qm.Reference(), shots=4).shots == 4
    assert len(observer.calls) == 1


def test_composition_preflight_and_portable_context():
    """Public slots, immutable defaults, duplicate rejection and context are usable."""
    observer = Observer()
    base = OptimizationPipeline.standard(seed=42)
    pipeline = base.insert_before("qcore.simplify", observer)
    artifact = qm.compile(_program(), to=qm.Reference(), optimizer=pipeline)
    assert artifact.report.stages[0] == "optimization:example.observer@1"
    context = observer.calls[0][1]
    assert isinstance(context, OptimizationContext)
    assert (
        context.seed == 42 and context.target_fingerprint == qm.Reference().fingerprint
    )
    with pytest.raises(ValueError, match="unknown optimization pass"):
        base.replace("missing.pass", observer)
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.append(observer)
    with pytest.raises(ValueError, match="64"):
        OptimizationPipeline([observer] * 65)
    assert base.insert_after("qcore.simplify", observer).passes[-1] is observer


def test_missing_native_pipeline_support_fails_closed(monkeypatch):
    """An old extension cannot silently substitute standard or Python semantics."""
    from qamomile._native import compiler as native

    class OldHandle:
        pass

    monkeypatch.setattr(native, "VerifiedProgramHandle", OldHandle)
    with pytest.raises(qm.ArtifactError, match="Q7802.*lacks"):
        OptimizationPipeline()


def test_identity_includes_pass_configuration_policy_and_result():
    """Equal source and target cannot alias different selected pipelines."""
    source = _program()
    first = qm.compile(
        source, to=qm.Reference(), optimizer=OptimizationPipeline.standard(seed=1)
    )
    second = qm.compile(
        source, to=qm.Reference(), optimizer=OptimizationPipeline.standard(seed=2)
    )
    repeated = qm.compile(
        source, to=qm.Reference(), optimizer=OptimizationPipeline.standard(seed=1)
    )
    assert first.execution_identity != second.execution_identity
    assert first.execution_identity == repeated.execution_identity
    assert serialize(first) == serialize(repeated)


def test_rewrite_and_receipt_fail_closed():
    """Native rule, work, corruption and source checks cannot be skipped."""
    program = OptimizationProgram.from_bytes(_program().to_bytes())
    context = OptimizationContext("test", frozenset(), max_work=1)
    with pytest.raises(qm.ArtifactError, match="budget"):
        program.rewrite(("quantum.h",), ("quantum.h",), context=context)
    with pytest.raises(qm.ArtifactError, match="rewrite requires"):
        program.rewrite(("quantum.cx",), ("quantum.x",), context=context)
    with pytest.raises(qm.ArtifactError):
        OptimizationProgram.from_bytes(b"invalid")
    artifact = qm.compile(
        _program(), to=qm.Reference(), optimizer=OptimizationPipeline.standard()
    )
    metadata = dict(artifact.metadata.entries)
    receipt = metadata["qamomile.optimization.v1"]
    metadata["qamomile.optimization.v1"] = receipt[:-1] + bytes([receipt[-1] ^ 1])
    with pytest.raises(qm.ArtifactError, match="digest"):
        replace(artifact, metadata=FrozenMapping.from_mapping(metadata))


def test_failed_and_invalid_pass_results_never_fall_back():
    """A plugin failure or ABI change cannot become an OSS compilation."""

    class Failing(Observer):
        def run(self, program, context):
            raise RuntimeError("service unavailable")

    with pytest.raises(RuntimeError, match="service unavailable"):
        qm.compile(
            _program(), to=qm.Reference(), optimizer=OptimizationPipeline((Failing(),))
        )

    class WrongAbi(Observer):
        def run(self, program, context):
            different = qm.build_source(
                SOURCE.replace("qm.Bits[1]", "qm.Bits[2]").replace(
                    "qubits(1)", "qubits(2)"
                ),
                module_name="compiler_extension",
            ).specialize("sample")
            return OptimizationProgram.from_bytes(different.to_bytes())

    with pytest.raises(qm.ArtifactError, match="ABI"):
        qm.compile(
            _program(), to=qm.Reference(), optimizer=OptimizationPipeline((WrongAbi(),))
        )


def test_run_accepts_optimizer_but_existing_artifact_rejects_it():
    """The convenience lifecycle selects preparation only when compiling source."""
    observer = Observer()
    pipeline = OptimizationPipeline.standard().append(observer)
    record = qm.run(_program(), on=qm.Reference(), optimizer=pipeline, shots=4)
    assert len(observer.calls) == 1
    assert record.result.counts == {(1,): 4}
    with pytest.raises(TypeError):
        qm.run(record.artifact, on=qm.Reference(), optimizer=pipeline)


def test_reuse_standard_options_without_copying_or_subclassing_the_simplifier():
    """A built-in pass can be configured and reused inside a custom pass."""
    source = qm.build_source(
        SOURCE.replace("qm.z(state)", "qm.h(state)"), module_name="compiler_extension"
    ).specialize("sample")
    configured = StandardSimplification(inverses=False)
    pipeline = OptimizationPipeline.standard().replace("qcore.simplify", configured)
    artifact = qm.compile(source, to=qm.Reference(), optimizer=pipeline)
    assert (
        qm.Program.from_bytes(export_payload(artifact)).inspect().count("quantum.h")
        == 3
    )
    default = qm.compile(
        source, to=qm.Reference(), optimizer=OptimizationPipeline.standard()
    )
    assert artifact.execution_identity != default.execution_identity
    with pytest.raises(qm.ArtifactError, match="work budget"):
        qm.compile(
            source,
            to=qm.Reference(),
            optimizer=OptimizationPipeline((configured,), max_work=1),
        )


@pytest.mark.parametrize("multilevel", [False, True])
@pytest.mark.parametrize("route", ["family", "launch", "shot"])
def test_deferred_routes_preserve_selection_through_persistence_without_reinvocation(
    route,
    multilevel,
):
    """Deferred families, launch recipes and shot cohorts reuse selected QCore."""
    pytest.importorskip("qiskit_aer")
    from qamomile.qiskit import Aer

    from .test_compilation_api import _HOST_DISPATCH_SOURCE
    from .test_shot_execution import SOURCE as SHOT_SOURCE

    if route == "family":
        source = _HOST_DISPATCH_SOURCE.replace(
            "    return qm.measure(state)",
            "    state = qm.h(state)\n    state = qm.z(state)\n    state = qm.h(state)\n    return qm.measure(state)",
        )
        entry = "sample"
        parameters = {"first": True, "theta": 0.25, "limit": 0}
        expected = {(0,): 4}
        format_name = "qamomile.artifact-family.v3"
    elif route == "launch":
        source = SOURCE.replace(
            "def sample()", "def sample(rounds: qm.Param[qm.Int])"
        ).replace(
            "    state = qm.h(state)",
            "    for _ in range(rounds):\n        state = qm.x(state)\n    state = qm.h(state)",
            1,
        )
        entry, parameters, expected = "sample", {"rounds": 2}, {(1,): 4}
        format_name = "qamomile.launch-materializer.v1"
    else:
        source = SHOT_SOURCE.replace(
            "a = qm.measure(qm.h(qm.prepare(False)))",
            "a = qm.measure(qm.h(qm.z(qm.h(qm.prepare(False)))))",
        )
        entry, parameters = "independent", {}
        expected = {(1, 1): 4}
        format_name = "qamomile.shot-execution.v1"
    program = qm.build_source(source).specialize(entry)
    extension = Observer() if multilevel else Conjugation()
    pipeline = (
        OptimizationPipeline.multilevel(family=False)
        if multilevel
        else OptimizationPipeline.standard()
    ).append(extension)
    endpoint = Aer()
    artifact = qm.compile(program, to=endpoint.target, optimizer=pipeline)
    assert artifact.format == format_name
    assert len(extension.calls) == 1
    for compiled in (artifact, deserialize(serialize(artifact))):
        result = qm.execute(
            compiled, on=endpoint, parameters=parameters, shots=4, seed=7
        )
        assert result.counts == expected
        assert compiled.source_digest.hex() == program.semantic_digest
    assert len(extension.calls) == 1


@pytest.mark.parametrize(
    "backend",
    [
        "qiskit",
        pytest.param("hugr", marks=pytest.mark.hugr),
        pytest.param("qir", marks=pytest.mark.qir),
    ],
)
def test_selected_rewrite_reaches_direct_target_adapters(backend):
    """Different emitters consume the chosen preparation and retain source identity."""
    if backend == "qiskit":
        pytest.importorskip("qiskit_aer")
        from qamomile.qiskit import Aer

        endpoint = Aer()
    elif backend == "hugr":
        pytest.importorskip("selene_sim")
        from qamomile.hugr import Selene

        endpoint = Selene(timeout=30.0)
    else:
        pytest.importorskip("pyqir")
        from qamomile.qir import QIR

        endpoint = None
    extension = Conjugation()
    program = _program()
    artifact = qm.compile(
        program,
        to=QIR() if endpoint is None else endpoint.target,
        optimizer=OptimizationPipeline.standard().append(extension),
    )
    assert artifact.report.stages.count("optimization:example.conjugation@1") == 1
    assert artifact.report.stages.count("rust-qcore-standard-optimization") == 1
    assert artifact.source_digest.hex() == program.semantic_digest
    restored = deserialize(serialize(artifact))
    if endpoint is None:
        from .test_qir_backend import _counts

        assert _counts(restored, shots=4) == {(1,): 4}
    else:
        assert qm.execute(restored, on=endpoint, shots=4).counts == {(1,): 4}
    assert len(extension.calls) == 1


def test_optimization_results_may_change_intent_but_not_attributes():
    """Verifier blindness to intent is exact: attributes stay schema-checked."""
    from qamomile.serialization import (
        _parse_sealed_package,
        _seal_package,
        encode_literal,
    )

    class Annotate(Observer):
        name = "example.annotate"

        def run(self, program, context):
            package = _parse_sealed_package(program.to_bytes())
            function = package.functions[0]
            encode_literal("example.agent", function.intent.add(name="origin").value)
            operation = function.body.operations[0]
            encode_literal("allocation", operation.intent.add(name="doc").value)
            return OptimizationProgram.from_bytes(_seal_package(package))

    class Attribute(Observer):
        name = "example.attribute"

        def run(self, program, context):
            package = _parse_sealed_package(program.to_bytes())
            operation = package.functions[0].body.operations[0]
            encode_literal("x", operation.attributes.add(name="intent.origin").value)
            return OptimizationProgram.from_bytes(_seal_package(package))

    source = _program()
    annotated = qm.compile(
        source, to=qm.Reference(), optimizer=OptimizationPipeline((Annotate(),))
    )
    plain = qm.compile(source, to=qm.Reference(), optimizer=OptimizationPipeline())
    result = qm.Program.from_bytes(export_payload(annotated))
    assert dict(result.entry.intent) == {"origin": "example.agent"}
    assert result.entry.body.operations[0].intent == (("doc", "allocation"),)
    assert result.semantic_digest == source.semantic_digest
    assert annotated.execution_identity != plain.execution_identity
    assert qm.execute(annotated, on=qm.Reference(), shots=4).counts == {(1,): 4}

    with pytest.raises(qm.ArtifactError, match="Q7802"):
        qm.compile(
            source, to=qm.Reference(), optimizer=OptimizationPipeline((Attribute(),))
        )


def test_with_intent_is_restored_by_artifacts_without_changing_meaning():
    """Entry intent persists through compilation while identity stays semantic."""
    source = _program()
    tagged = source.with_intent(problem="sha256:0", **{"tolerance.objective": 0.05})
    assert dict(tagged.entry.intent) == {
        "problem": "sha256:0",
        "tolerance.objective": 0.05,
    }
    assert tagged.semantic_digest == source.semantic_digest
    assert tagged.to_bytes() != source.to_bytes()
    assert dict(tagged.with_intent(problem="sha256:1").entry.intent) == {
        "problem": "sha256:1",
        "tolerance.objective": 0.05,
    }

    pipeline = OptimizationPipeline.standard(seed=3)
    plain = qm.compile(source, to=qm.Reference(), optimizer=pipeline)
    annotated = qm.compile(tagged, to=qm.Reference(), optimizer=pipeline)
    restored = qm.Program.from_bytes(export_payload(annotated))
    assert annotated.source_digest == plain.source_digest
    assert annotated.report.resources.metrics == plain.report.resources.metrics
    assert dict(restored.entry.intent)["problem"] == "sha256:0"
    assert restored.semantic_digest == source.semantic_digest
    assert qm.execute(annotated, on=qm.Reference(), shots=4).counts == {(1,): 4}

    with pytest.raises(TypeError, match="portable"):
        source.with_intent(problem=object())
    with pytest.raises(qm.VerificationError, match="Q1060"):
        source.with_intent(**{"Bad Name": 1})


def test_intent_is_carried_only_where_an_operation_survives_a_rewrite():
    """Replacement operations synthesize no intent; untouched ones keep theirs."""
    from qamomile.serialization import (
        _parse_sealed_package,
        _seal_package,
        encode_literal,
    )

    package = _parse_sealed_package(_program().to_bytes())
    for operation in package.functions[0].body.operations:
        encode_literal(operation.opcode, operation.intent.add(name="doc").value)
    annotated = OptimizationProgram.from_bytes(_seal_package(package))
    rewritten = annotated.rewrite(
        ("quantum.h", "quantum.z", "quantum.h"),
        ("quantum.x",),
        context=OptimizationContext("", frozenset()),
    )
    operations = qm.Program.from_bytes(rewritten.to_bytes()).entry.body.operations

    assert [operation.opcode for operation in operations].count("quantum.x") == 1
    for operation in operations:
        expected = (
            () if operation.opcode == "quantum.x" else (("doc", operation.opcode),)
        )
        assert operation.intent == expected


def test_context_profile_follows_the_target_capability_snapshot():
    """Reference supplies no profile; a Qiskit backend with a coupling map does."""
    observer = Observer()
    pipeline = OptimizationPipeline((observer,))
    qm.compile(_program(), to=qm.Reference(), optimizer=pipeline)
    assert observer.calls[0][1].profile is None

    pytest.importorskip("qiskit")
    from qiskit.providers.fake_provider import GenericBackendV2

    from qamomile.qiskit import Qiskit, profile_from_backend

    backend = GenericBackendV2(num_qubits=3, coupling_map=[[0, 1], [1, 2]], seed=1)
    target = Qiskit(backend=backend)
    qm.compile(_program(), to=target, optimizer=OptimizationPipeline((observer,)))
    context = observer.calls[1][1]
    assert context.profile == profile_from_backend(backend)
    assert context.target_fingerprint == target.fingerprint
    assert [edge.target for edge in context.profile.topology.edges] == [1, 2]
