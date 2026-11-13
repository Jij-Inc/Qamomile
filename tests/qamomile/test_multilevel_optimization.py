"""Executable witnesses for symbolic, hybrid, realization and circuit stages."""

import pytest

import qamomile as qm
from qamomile.compilation import (
    CommonExpressions,
    OptimizationContext,
    OptimizationPipeline,
    OptimizationProgram,
    RealizationSelection,
)
from qamomile.persistence import deserialize, serialize
from qamomile.source import ProgramFamily

SOURCE = """from __future__ import annotations
import qamomile as qm

@qm.qkernel
def sample(n: qm.Const[qm.Int], theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[n]]:
    state = qm.qubits(n)
    a = theta * 2.0
    b = theta * 2.0
    state = qm.rx(state, a, target=0)
    state = qm.rz(state, b, target=0)
    state = qm.h(state, target=0)
    state = qm.z(state, target=0)
    state = qm.h(state, target=0)
    return qm.measure(state)
"""


def family(source=SOURCE):
    """Retain a verified family before binding its dimension."""
    return ProgramFamily(qm.build_source(source, module_name="multilevel"), "sample")


def test_symbolic_family_and_hybrid_share_native_cse_without_binding():
    """The same driver shares Param expressions before and after Const binding."""
    source = family()
    assert source.module.inspect().count("classical.mul") == 2
    pipeline = OptimizationPipeline.multilevel()
    symbolic = pipeline.optimize_family(source)
    assert symbolic.module.inspect().count("classical.mul") == 1
    assert source.module.inspect().count("classical.mul") == 2
    program = source.specialize(n=1)
    result = CommonExpressions().run(
        OptimizationProgram.from_bytes(program.to_bytes()),
        OptimizationContext("", frozenset()),
    )
    assert (
        qm.Program.from_bytes(result.to_bytes()).inspect().count("classical.mul") == 1
    )


def test_symbolic_loop_work_does_not_depend_on_unbound_extent():
    """A loop stays structured and retains an unbound dimension after CSE."""
    source = SOURCE.replace(
        "    a = theta * 2.0\n    b = theta * 2.0\n    state = qm.rx(state, a, target=0)\n    state = qm.rz(state, b, target=0)",
        "    for i in range(n):\n        a = theta * 2.0\n        b = theta * 2.0\n        state = qm.rx(state, a, target=i)\n        state = qm.rz(state, b, target=i)",
    )
    result = OptimizationPipeline.multilevel(max_work=100).optimize_family(
        family(source)
    )
    text = result.module.inspect()
    assert text.count("classical.mul") == 1
    assert "qcore.for" in text
    assert len(result.module.to_bytes()) < 20_000


def test_complete_flow_preserves_source_and_restores_without_callbacks():
    """All stages reach SDK emission, retain ABI, and survive serialization."""
    pytest.importorskip("qiskit_aer")
    from qiskit_aer import AerSimulator

    from qamomile.qiskit import QiskitBackend

    source = family()
    pipeline = OptimizationPipeline.multilevel(realization=RealizationSelection())
    endpoint = QiskitBackend(AerSimulator(), optimization_level=3, seed_transpiler=7)
    target = endpoint.target
    artifact = qm.compile(source, n=1, to=target, optimizer=pipeline)
    stages = artifact.report.stages
    assert "optimization:family.common-expressions@1" in stages
    assert "optimization:qcore.common-expressions@1" in stages
    assert "optimization:realization.minimum-selectors@1" in stages
    assert "optimization:circuit.exact-local@2" in stages
    assert "qiskit-routing" in stages
    assert artifact.source_digest.hex() == source.specialize(n=1).semantic_digest
    for candidate in (artifact, deserialize(serialize(artifact))):
        assert qm.execute(
            candidate, on=endpoint, parameters={"theta": 0.0}, shots=8, seed=0
        ).counts == {(1,): 8}


@pytest.mark.parametrize(
    "gate",
    [
        "qm.x(state, target=i)",
        "qm.rx(state, 0.0, target=i)",
        "qm.cx(state, control=i, target=i)",
    ],
)
def test_standard_simplification_diagnoses_known_invalid_selectors(gate):
    """Known source-domain errors are diagnosed before identity or inverse removal."""
    source = f"""import qamomile as qm
@qm.qkernel
def sample(i: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.qubits(1)
    state = {gate}
    state = {gate}
    return qm.measure(state)
"""
    program = qm.build_source(source).specialize("sample", i=2)
    with pytest.raises(qm.QamomileError, match="Q7804"):
        qm.compile(
            program,
            to=qm.Reference(),
            optimizer=OptimizationPipeline.multilevel(family=False),
        )


def test_explicit_stage_availability_and_immutable_composition():
    """Unavailable family/circuit stages fail instead of silently disappearing."""
    source = family()
    pipeline = OptimizationPipeline.multilevel()
    with pytest.raises(ValueError, match="duplicate"):
        pipeline.replace("qcore.simplify", CommonExpressions())
    with pytest.raises(TypeError, match="family"):
        qm.compile(source.specialize(n=1), to=qm.Reference(), optimizer=pipeline)
    from .test_compiler_optimization import Observer

    extended = pipeline.append(Observer())
    assert len(extended.passes) == len(pipeline.passes) + 1
    assert extended.family_passes == pipeline.family_passes
    assert extended.circuit == pipeline.circuit
    assert extended.realization == pipeline.realization


def test_classical_cse_keeps_first_failure_and_never_speculates():
    """Identical partial operations may share only an earlier dominating result."""
    source = qm.build_source("""import qamomile as qm
@qm.qkernel
def sample(x: qm.Param[qm.Int], enabled: qm.Param[qm.Bool]) -> qm.Param[qm.Int]:
    if enabled:
        a = 10 // x
        b = 10 // x
        return a + b
    return 3
""").specialize("sample")
    artifact = qm.compile(
        source,
        to=qm.Reference(),
        optimizer=OptimizationPipeline((CommonExpressions(),)),
    )
    assert (
        qm.execute(
            artifact, on=qm.Reference(), parameters={"x": 0, "enabled": False}
        ).value
        == 3
    )
    assert (
        qm.execute(
            artifact, on=qm.Reference(), parameters={"x": 2, "enabled": True}
        ).value
        == 10
    )
    with pytest.raises(qm.QamomileError):
        qm.execute(artifact, on=qm.Reference(), parameters={"x": 0, "enabled": True})


@pytest.mark.parametrize(
    "gates",
    [("h", "z", "h"), ("h", "x", "h"), ("t", "t"), ("sdg", "sdg"), ("s", "sdg")],
)
@pytest.mark.parametrize("controlled", [False, True])
def test_late_circuit_rewrites_preserve_full_operator(gates, controlled):
    """Compare matrices including global phase, also under coherent control."""
    import numpy as np
    from qiskit.quantum_info import Operator

    from qamomile.compilation import CircuitSimplification
    from qamomile.qiskit import Qiskit, load_circuit

    body = "\n".join(f"    state = qm.{gate}(state)" for gate in gates)
    entry = (
        "    control, state = qm.split_at(qm.qubits(2), 1)\n"
        "    control, state = qm.control(action, control, state)\n"
        "    return qm.measure(qm.concat(control, state))"
        if controlled
        else "    return qm.measure(action(qm.qubits(1)))"
    )
    source = qm.build_source(f"""import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
{body}
    return state
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[{2 if controlled else 1}]]:
{entry}
""").specialize("sample")
    baseline = load_circuit(
        qm.compile(source, to=Qiskit(), optimizer=OptimizationPipeline())
    )
    optimized = load_circuit(
        qm.compile(
            source,
            to=Qiskit(),
            optimizer=OptimizationPipeline(circuit=CircuitSimplification()),
        )
    )
    np.testing.assert_allclose(
        Operator(optimized.remove_final_measurements(inplace=False)).data,
        Operator(baseline.remove_final_measurements(inplace=False)).data,
        rtol=0,
        atol=1e-14,
    )

    def body_size(circuit):
        operation = circuit.data[0].operation
        if controlled:
            operation = operation.base_op
        return len(operation.definition.data)

    assert body_size(optimized) < body_size(baseline)


def test_family_extension_reuses_driver_and_rejects_changed_interface():
    """A third-party symbolic pass needs neither a backend nor a compiler owner."""
    from qamomile.compilation import OptimizationFamily

    class FamilyCse:
        name, version, configuration = "example.family-cse", "1", b""

        def run(self, source, context):
            return source.eliminate_common_expressions(context=context)

    source = family()
    extended = OptimizationPipeline.multilevel().replace_family(
        "family.common-expressions", FamilyCse()
    )
    assert extended.optimize_family(source).module.inspect().count("classical.mul") == 1

    class ChangedSignature(FamilyCse):
        def run(self, source, context):
            other = family(
                SOURCE.replace("theta: qm.Param[qm.Real]", "theta: qm.Const[qm.Real]")
            )
            return OptimizationFamily.from_bytes(
                other.module.to_bytes(), other.entrypoint
            )

    with pytest.raises(qm.ArtifactError, match="interface"):
        OptimizationPipeline(family_passes=(ChangedSignature(),)).optimize_family(
            source
        )
    with pytest.raises(qm.ArtifactError, match="budget"):
        OptimizationPipeline.multilevel(max_work=1).optimize_family(source)


@pytest.mark.qir
def test_late_circuit_optimization_reaches_qir():
    """Circuit-consuming QIR exposes late stages without claiming a family emitter."""
    from qamomile.qir import QIR

    from .test_compiler_optimization import _program
    from .test_qir_backend import _counts

    artifact = qm.compile(
        _program(), to=QIR(), optimizer=OptimizationPipeline.multilevel(family=False)
    )
    assert "optimization:circuit.exact-local@2" in artifact.report.stages
    assert _counts(deserialize(serialize(artifact)), shots=4) == {(1,): 4}


@pytest.mark.hugr
def test_structured_target_accepts_high_levels_and_rejects_late_projection():
    """A structured target keeps symbolic/hybrid passes without forced flattening."""
    from qamomile.hugr import Hugr

    source = family()
    with pytest.raises(qm.ArtifactError, match="circuit"):
        qm.compile(source, n=1, to=Hugr(), optimizer=OptimizationPipeline.multilevel())
    artifact = qm.compile(
        source,
        n=1,
        to=Hugr(),
        optimizer=OptimizationPipeline.multilevel(realization=None, circuit=None),
    )
    assert "optimization:family.common-expressions@1" in artifact.report.stages
    assert not any("circuit.exact-local" in s for s in artifact.report.stages)


@pytest.mark.parametrize(
    "optimizer",
    [
        None,
        OptimizationPipeline(),
        OptimizationPipeline.standard(),
        OptimizationPipeline.multilevel(family=False),
    ],
    ids=["default", "empty", "standard", "multilevel"],
)
def test_inverse_callable_cancellation_uses_defined_execution_domain(optimizer):
    """Cancellation does not require totality; disabling it retains runtime checks."""
    source = qm.build_source("""import qamomile as qm
@qm.qkernel
def action(state: qm.Q[qm.Bits[1]], i: qm.Param[qm.Int]) -> qm.Q[qm.Bits[1]]:
    return qm.x(state, target=i)
@qm.qkernel
def sample(i: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = action(qm.qubits(1), i)
    state = qm.adjoint(action, state, i)
    return qm.measure(state)
""").specialize("sample")
    artifact = qm.compile(
        source,
        to=qm.Reference(),
        optimizer=optimizer,
    )
    assert qm.execute(
        artifact, on=qm.Reference(), parameters={"i": 0}, shots=1
    ).counts == {(0,): 1}
    if optimizer == OptimizationPipeline():
        with pytest.raises(qm.QamomileError, match="outside width"):
            qm.execute(artifact, on=qm.Reference(), parameters={"i": 2}, shots=1)
    else:
        assert qm.execute(
            artifact, on=qm.Reference(), parameters={"i": 2}, shots=1
        ).counts == {(0,): 1}


def test_symbolic_pipeline_reuses_source_descriptor_binding():
    """Library descriptor composition precedes native symbolic optimization."""
    from .test_stdlib import deferred_block_sample, x_block_encoding

    artifact = qm.compile(
        deferred_block_sample,
        block=x_block_encoding,
        to=qm.Reference(),
        optimizer=OptimizationPipeline.multilevel(),
    )
    assert qm.execute(artifact, on=qm.Reference(), shots=4).counts == {(0, 1): 4}
    assert (
        artifact.source_digest.hex()
        == deferred_block_sample.specialize(block=x_block_encoding).semantic_digest
    )


def test_late_realization_policy_survives_native_preparation_and_overrides():
    """Automatic selection reaches projection while explicit choices remain final."""
    from qamomile.circuit import lower_to_circuit
    from qamomile.lowering import RealizationOptions

    source = qm.build_source("""import qamomile as qm
@qm.qkernel
def sample() -> qm.Shot[qm.Bits[32]]:
    return qm.measure(qm.add_into(qm.qubits(32)))
""").specialize("sample")
    pipeline = OptimizationPipeline.multilevel(
        family=False, realization=RealizationSelection()
    )
    prepared = pipeline._prepare(source, OptimizationContext("", frozenset()))
    circuit = lower_to_circuit(prepared)
    # Sixteen-bit ripple carry has fewer selectors (666) than Fourier addition.
    assert "add.ripple" in circuit.realizations
    explicit = lower_to_circuit(
        prepared, realizations=RealizationOptions({qm.add_into: "add.increments"})
    )
    assert "add.increments" in explicit.realizations
    assert "optimization:realization.minimum-selectors@1" in circuit.realization_stages
    assert "optimization:circuit.exact-local@2" in explicit.realization_stages


@pytest.mark.parametrize("middle, expected", [("x", 2.0), ("z", 0.5)])
def test_late_rewrite_preserves_lazy_measured_error_checks(middle, expected):
    """Shortened quantum prefixes preserve guarded host arithmetic after measurement."""
    from qamomile.qiskit import Aer

    from .test_shot_execution import SOURCE as SHOT_SOURCE

    source = SHOT_SOURCE.replace(
        "a = qm.measure(qm.prepare(False))",
        f"a = qm.measure(qm.h(qm.{middle}(qm.h(qm.prepare(False)))))",
        1,
    )
    program = qm.build_source(source).specialize("terminal")
    endpoint = Aer()
    artifact = deserialize(
        serialize(
            qm.compile(
                program,
                to=endpoint.target,
                optimizer=OptimizationPipeline.multilevel(family=False),
            )
        )
    )
    assert qm.execute(
        artifact, on=endpoint, parameters={"value": 2.0}, shots=4
    ).values == {expected: 4}
    if middle == "x":
        assert qm.execute(
            artifact, on=endpoint, parameters={"value": 0.0}, shots=4
        ).values == {2.0: 4}
    else:
        with pytest.raises(qm.QamomileError):
            qm.execute(artifact, on=endpoint, parameters={"value": 0.0}, shots=4)


COMMUTING_RULES_SOURCE = """import qamomile as qm
@qm.qkernel
def state(theta: qm.Param[qm.Real]) -> qm.Q[qm.Bits[4]]:
    s = qm.qubits(4)
    for i in range(4):
        s = qm.h(s, target=i)
    s = qm.swap(s, left=0, right=3)
    s = qm.cx(s, control=0, target=1)
    s = qm.rz(s, theta, target=0)
    s = qm.t(s, target=2)
    s = qm.cx(s, control=0, target=1)
    s = qm.cp(s, 0.25, control=1, target=2)
    s = qm.rz(s, theta, target=0)
    s = qm.cp(s, -0.25, control=2, target=1)
    s = qm.swap(s, left=1, right=2)
    s = qm.t(s, target=1)
    return s
"""


@pytest.mark.parametrize("theta", [0.0, 0.7])
def test_commuting_circuit_rules_preserve_full_operator_with_relabeled_swaps(theta):
    """Lookback cancellation, angle merging and swap relabeling are phase exact."""
    import numpy as np
    from qiskit.quantum_info import Operator

    from qamomile.compilation import CircuitSimplification
    from qamomile.qiskit import Qiskit, to_circuit

    program = qm.build_source(COMMUTING_RULES_SOURCE).specialize("state")
    target = Qiskit(use_native_pauli_evolution=False)
    context = OptimizationContext("", frozenset())
    baseline = to_circuit(
        OptimizationPipeline()._prepare(program, context),
        target=target,
        parameters={"theta": theta},
    )
    assert baseline.count_ops()["cx"] == 2
    optimized_program = OptimizationPipeline(circuit=CircuitSimplification())._prepare(
        program, context
    )
    optimized = to_circuit(
        optimized_program, target=target, parameters={"theta": theta}
    )
    np.testing.assert_allclose(
        Operator(optimized).data, Operator(baseline).data, rtol=0, atol=1e-12
    )
    ops = optimized.count_ops()
    # Both CX cancel across the commuting RZ; the two RZ merge; the CP pair
    # cancels; live result wires restore the net permutation with swaps only.
    assert "cx" not in ops
    assert ops.get("rz", 0) == 1
    assert "cp" not in ops


def test_circuit_rules_accept_checks_recorded_inside_retained_loops():
    """Numeric checks emitted inside a nested loop body do not break remapping."""
    import numpy as np
    from qiskit.quantum_info import Operator
    from qiskit.transpiler.passes import UnrollForLoops

    from qamomile.compilation import CircuitSimplification
    from qamomile.qiskit import to_circuit

    program = qm.build_source("""import qamomile as qm
@qm.qkernel
def repeated(state: qm.Q[qm.Bits[1]]) -> qm.Q[qm.Bits[1]]:
    for index in range(1, 6, 2):
        state = qm.rx(state, 0.13 * index)
        state = qm.rz(state, 0.21 * index)
        state = qm.global_phase(state, 0.07 * index)
    return state

@qm.qkernel
def state() -> qm.Q[qm.Bits[2]]:
    control, target = qm.split_at(qm.qubits(2), 1)
    control, target = qm.control(repeated, control, target)
    return qm.concat(control, target)
""").specialize("state")
    context = OptimizationContext("", frozenset())
    circuits = [
        to_circuit(pipeline._prepare(program, context))
        for pipeline in (
            OptimizationPipeline(),
            OptimizationPipeline(circuit=CircuitSimplification()),
        )
    ]
    baseline, optimized = (Operator(UnrollForLoops()(item)).data for item in circuits)
    np.testing.assert_allclose(optimized, baseline, rtol=0, atol=1e-12)
