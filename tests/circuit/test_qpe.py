"""Tests for QPE: naive implementation vs built-in qmc.qpe()."""

import math
import random
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit.frontend.handle import QFixed, Qubit
from qamomile.circuit.frontend.tracer import get_current_tracer
from qamomile.circuit.ir.operation.callable import (
    CallableDef,
    CallableRef,
    CompositeGateType,
    InvokeOperation,
)
from qamomile.circuit.ir.operation.cast import CastOperation
from qamomile.circuit.ir.operation.control_flow import HasNestedOps
from qamomile.circuit.ir.operation.gate import (
    ControlledUOperation,
    MeasureQFixedOperation,
    MeasureVectorOperation,
)
from qamomile.circuit.ir.types import QFixedType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.transpiler.passes.separate import lower_operations


def _decode_phase(bits: list) -> float:
    """Decode measurement bits into a phase estimate."""
    return sum(bit * (1 / (2 ** (i + 1))) for i, bit in enumerate(reversed(bits)))


def _collect_controlled_u_ops(operations: list[Any]) -> list[ControlledUOperation]:
    """Collect controlled-U operations from a nested operation list.

    Args:
        operations (list[Any]): Operations to traverse.

    Returns:
        list[ControlledUOperation]: Controlled-U operations found at any
        nesting depth.
    """
    collected: list[ControlledUOperation] = []
    for op in operations:
        if isinstance(op, ControlledUOperation):
            collected.append(op)
        if isinstance(op, HasNestedOps):
            for body in op.nested_op_lists():
                collected.extend(_collect_controlled_u_ops(list(body)))
    return collected


@pytest.fixture
def qiskit_transpiler():
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    return QiskitTranspiler()


# -- Helper: naive QPE -------------------------------------------------------


@qmc.qkernel
def _iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    n = qubits.shape[0]
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits


@qmc.qkernel
def _phase_gate(q: qmc.Qubit, theta: float, iter: int) -> qmc.Qubit:
    for _ in qmc.range(iter):
        q = qmc.p(q, theta)
    return q


@qmc.qkernel
def naive_qpe(n: int, phase: float) -> qmc.Vector[qmc.Bit]:
    phase_register = qmc.qubit_array(n, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    controlled_phase_gate = qmc.control(_phase_gate)
    num = phase_register.shape[0]
    for i in qmc.range(num):
        phase_register[i] = qmc.h(phase_register[i])
    for i in qmc.range(num):
        phase_register[i], target = controlled_phase_gate(
            phase_register[i], target, theta=phase, iter=2**i
        )
    phase_register = _iqft(phase_register)
    bits = qmc.measure(phase_register)
    return bits


# -- Helper: built-in QPE ----------------------------------------------------


@qmc.qkernel
def _p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    return qmc.p(q, theta)


@qmc.composite_gate(name="boxed_p_gate")
def _boxed_p_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    return qmc.p(q, theta)


@qmc.qkernel
def builtin_qpe(n: int, phase: float) -> qmc.Float:
    q_phase = qmc.qubit_array(n, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    phase_q: qmc.QFixed = qmc.qpe(target, q_phase, _p_gate, theta=phase)
    return qmc.measure(phase_q)


@qmc.qkernel
def _mul_two_mod_three(work: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply multiplication by two modulo three using private workspace.

    Args:
        work (qmc.Vector[qmc.Qubit]): Two-qubit modular value register.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated modular value register.
    """
    return qmc.modmul_const(work, multiplier=2, modulus=3)


@qmc.qkernel
def qpe_with_private_workspace_unitary() -> qmc.Float:
    """Estimate the order-two phase of a workspace-allocating unitary.

    Returns:
        qmc.Float: Two-bit phase estimate, either zero or one half.
    """
    counting = qmc.qubit_array(2, name="counting")
    work = qmc.qubit_array(2, name="work")
    work[0] = qmc.x(work[0])
    phase = qmc.qpe(work, counting, _mul_two_mod_three)
    return qmc.measure(phase)


@qmc.qkernel
def builtin_qpe_with_composite_unitary(n: int, phase: float) -> qmc.Float:
    """Run QPE with a qkernel-backed composite gate callable as U."""
    q_phase = qmc.qubit_array(n, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    phase_q: qmc.QFixed = qmc.qpe(target, q_phase, _boxed_p_gate, theta=phase)
    return qmc.measure(phase_q)


@qmc.qkernel
def builtin_qpe_vector_view_phase(gammas: qmc.Vector[qmc.Float]) -> qmc.Float:
    """Estimate a phase selected from a sliced Vector through public QPE.

    Args:
        gammas (qmc.Vector[qmc.Float]): Compile-time-bound phase angle vector.

    Returns:
        qmc.Float: Decoded QPE phase estimate.
    """
    gammas_view = gammas[1:3]
    q_phase = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    phase_q: qmc.QFixed = qmc.qpe(target, q_phase, _p_gate, theta=gammas_view[0])
    return qmc.measure(phase_q)


def _fallback_qpe_phase(
    target: qmc.Qubit,
    counting: qmc.Vector[qmc.Qubit],
    theta: qmc.Float,
) -> QFixed:
    """Emit a block-free QPE composite operation for fallback emission tests.

    Args:
        target (qmc.Qubit): Eigenstate qubit consumed as the QPE target.
        counting (qmc.Vector[qmc.Qubit]): Three-qubit counting register.
        theta (qmc.Float): Phase angle operand passed through composite
            fallback phase extraction.

    Returns:
        QFixed: Fixed-point phase register produced from the counting qubits.
    """
    counting_handles = [counting[i] for i in range(3)]
    counting_operands = [handle.value for handle in counting_handles]
    target_operand = target.value
    counting_results = [value.next_version() for value in counting_operands]
    target_result = target_operand.next_version()

    for i, handle in enumerate(counting_handles):
        counting[i] = Qubit(
            value=counting_results[i],
            parent=handle.parent,
            indices=handle.indices,
            name=handle.name,
        )

    ref = CallableRef(namespace="qamomile.stdlib", name="qpe")
    attrs = {
        "kind": "composite",
        "gate_type": CompositeGateType.QPE.name,
        "num_control_qubits": 3,
        "num_target_qubits": 1,
        "custom_name": "qpe",
    }
    op = InvokeOperation(
        operands=[*counting_operands, target_operand, theta.value],
        results=[*counting_results, target_result],
        target=ref,
        attrs=attrs,
        definition=CallableDef(ref=ref, attrs=attrs),
    )
    get_current_tracer().add_operation(op)

    qubit_uuids = [result.uuid for result in counting_results]
    qubit_logical_ids = [result.logical_id for result in counting_results]
    result_value = (
        Value(
            type=QFixedType(),
            name=f"{counting.value.name}_as_qfixed",
        )
        .with_cast_metadata(
            source_uuid=counting.value.uuid,
            source_logical_id=counting.value.logical_id,
            qubit_uuids=qubit_uuids,
            qubit_logical_ids=qubit_logical_ids,
        )
        .with_qfixed_metadata(
            qubit_uuids=qubit_uuids,
            num_bits=3,
            int_bits=0,
        )
    )
    get_current_tracer().add_operation(
        CastOperation(
            operands=[counting.value],
            results=[result_value],
            source_type=counting.value.type,
            target_type=result_value.type,
            qubit_mapping=qubit_uuids,
        )
    )
    return QFixed(value=result_value)


@qmc.qkernel
def fallback_qpe_vector_view_phase(gammas: qmc.Vector[qmc.Float]) -> qmc.Float:
    """Estimate a phase selected from a sliced Vector parameter.

    Args:
        gammas (qmc.Vector[qmc.Float]): Compile-time-bound phase angle vector.

    Returns:
        qmc.Float: Decoded QPE phase estimate.
    """
    gammas_view = gammas[1:3]
    q_phase = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)
    phase_q = _fallback_qpe_phase(target, q_phase, gammas_view[0])
    return qmc.measure(phase_q)


def _assert_fallback_qpe_vector_view_phase(transpiler: Any) -> None:
    """Assert fallback QPE resolves and executes a VectorView phase operand.

    Args:
        transpiler (Any): Backend transpiler exposing ``transpile`` and
            ``executor`` methods.
    """
    executable = transpiler.transpile(
        fallback_qpe_vector_view_phase,
        bindings={"gammas": np.array([math.pi / 4, math.pi / 2, math.pi])},
    )
    result = executable.sample(transpiler.executor(), shots=256).result()

    assert len(result.results) == 1
    value, count = result.results[0]
    assert value == pytest.approx(0.25)
    assert count == 256


def _assert_builtin_qpe_vector_view_phase(transpiler: Any) -> None:
    """Assert public QPE resolves and executes a VectorView phase operand.

    Args:
        transpiler (Any): Backend transpiler exposing ``transpile`` and
            ``executor`` methods.
    """
    executable = transpiler.transpile(
        builtin_qpe_vector_view_phase,
        bindings={"gammas": np.array([math.pi / 4, math.pi / 2, math.pi])},
    )
    result = executable.sample(transpiler.executor(), shots=256).result()

    assert len(result.results) == 1
    value, count = result.results[0]
    assert value == pytest.approx(0.25)
    assert count == 256


# -- Tests --------------------------------------------------------------------


class TestQPENaive:
    """Naive QPE (manual IQFT + iter parameter) tests."""

    def test_naive_qpe_pi_over_2(self, qiskit_transpiler):
        """theta=pi/2 -> phase=0.25."""
        executable = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": 3, "phase": math.pi / 2}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for bits, count in result.results:
            phase_estimate = _decode_phase(bits)
            assert phase_estimate == pytest.approx(0.25), (
                f"Naive QPE: expected 0.25, got {phase_estimate} "
                f"(bits={bits}, count={count})"
            )

    def test_naive_qpe_pi_over_4(self, qiskit_transpiler):
        """theta=pi/4 -> phase=0.125."""
        executable = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": 3, "phase": math.pi / 4}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for bits, count in result.results:
            phase_estimate = _decode_phase(bits)
            assert phase_estimate == pytest.approx(0.125), (
                f"Naive QPE: expected 0.125, got {phase_estimate}"
            )


class TestQPEBuiltin:
    """Built-in QPE (qmc.qpe()) tests."""

    def test_symbolic_counting_size_keeps_deferred_iqft_and_cast(self):
        """Symbolic-size QPE keeps IQFT and QFixed cast as deferred aliases."""
        block = builtin_qpe.block

        iqft_ops = [
            op
            for op in block.operations
            if isinstance(op, InvokeOperation)
            and op.gate_type == CompositeGateType.IQFT
        ]
        assert len(iqft_ops) == 1
        assert iqft_ops[0].num_target_qubits == 0
        assert len(iqft_ops[0].target_qubits) == 1
        assert iqft_ops[0].target_qubits[0].shape

        cast_ops = [op for op in block.operations if isinstance(op, CastOperation)]
        assert len(cast_ops) == 1
        assert cast_ops[0].qubit_mapping == []

        assert any(isinstance(op, MeasureQFixedOperation) for op in block.operations)

        lowered = lower_operations(block)
        measure_vector_ops = [
            op for op in lowered.operations if isinstance(op, MeasureVectorOperation)
        ]
        assert len(measure_vector_ops) == 1
        assert measure_vector_ops[0].operands[0].uuid == cast_ops[0].operands[0].uuid

    def test_qpe_controlled_unitary_carries_callable_ref_and_attrs(self):
        """Built-in QPE records the target unitary identity and attrs in IR.

        QPE accepts a qkernel as its unitary argument. The IR must not only
        embed the unitary body; it should also retain the callable reference
        and attrs so compiler-side lowering and resource logic can identify the
        target.
        """
        block = builtin_qpe.build(n=3, phase=math.pi / 2)

        controlled_ops = _collect_controlled_u_ops(block.operations)

        assert controlled_ops
        assert all(op.callable_ref is not None for op in controlled_ops)
        assert {op.callable_ref.name for op in controlled_ops if op.callable_ref} == {
            "_p_gate"
        }
        assert all(
            op.callable_ref.namespace.startswith("user.qkernel.")
            for op in controlled_ops
            if op.callable_ref
        )
        assert {op.callable_attrs["kind"] for op in controlled_ops} == {"qkernel"}
        assert {op.callable_attrs["default_policy"] for op in controlled_ops} == {
            "INLINE"
        }

    def test_qpe_accepts_composite_gate_unitary_and_preserves_identity(self):
        """QPE accepts qkernel-like composite gate callables as U.

        Higher-order algorithms should not be restricted to concrete QKernel
        instances. Passing a qkernel-backed composite gate callable must retain
        the composite callable identity, not collapse it to the wrapped qkernel
        name.
        """
        block = builtin_qpe_with_composite_unitary.build(n=3, phase=math.pi / 2)

        controlled_ops = _collect_controlled_u_ops(block.operations)

        assert controlled_ops
        assert all(op.callable_ref is not None for op in controlled_ops)
        assert all(
            op.callable_ref.namespace.startswith("user.composite.")
            for op in controlled_ops
            if op.callable_ref
        )
        assert {op.callable_ref.name for op in controlled_ops if op.callable_ref} == {
            "boxed_p_gate"
        }
        assert {op.callable_attrs["kind"] for op in controlled_ops} == {"composite"}
        assert {op.callable_attrs["custom_name"] for op in controlled_ops} == {
            "boxed_p_gate"
        }
        assert {op.callable_attrs["default_policy"] for op in controlled_ops} == {
            "PRESERVE_BOX"
        }

    def test_builtin_qpe_pi_over_2(self, qiskit_transpiler):
        """theta=pi/2 -> phase=0.25."""
        executable = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": 3, "phase": math.pi / 2}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for value, count in result.results:
            assert value == pytest.approx(0.25), (
                f"Built-in QPE: expected 0.25, got {value} (count={count})"
            )

    def test_builtin_qpe_pi_over_4(self, qiskit_transpiler):
        """theta=pi/4 -> phase=0.125."""
        executable = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": 3, "phase": math.pi / 4}
        )
        job = executable.sample(qiskit_transpiler.executor(), shots=1024)
        result = job.result()

        for value, count in result.results:
            assert value == pytest.approx(0.125), (
                f"Built-in QPE: expected 0.125, got {value} (count={count})"
            )

    def test_qpe_controls_unitary_with_private_workspace(self, qiskit_transpiler):
        """QPE reserves and controls a unitary's internal ancilla wires."""
        executable = qiskit_transpiler.transpile(qpe_with_private_workspace_unitary)
        result = executable.sample(qiskit_transpiler.executor(), shots=64).result()

        phases = {value for value, _ in result.results}
        assert phases <= {0.0, 0.5}
        assert phases == {0.0, 0.5}


class TestQPEFallbackVectorViewPhase:
    """Fallback QPE composite emission with VectorView phase operands."""

    def test_qiskit_executes_vector_view_phase(self, qiskit_transpiler):
        """Qiskit executes fallback QPE with a VectorView phase element."""
        _assert_fallback_qpe_vector_view_phase(qiskit_transpiler)

    def test_quri_parts_executes_vector_view_phase(self):
        """QURI Parts executes fallback QPE with a VectorView phase element."""
        pytest.importorskip("quri_parts")
        from qamomile.quri_parts import QuriPartsTranspiler

        _assert_fallback_qpe_vector_view_phase(QuriPartsTranspiler())

    @pytest.mark.cudaq
    def test_cudaq_executes_vector_view_phase(self):
        """CUDA-Q executes fallback QPE with a VectorView phase element.

        Runs in ``-m cudaq`` sessions only: loading cudaq into a default
        session is unsafe — see tests/_cudaq_isolation.py.
        """
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        _assert_fallback_qpe_vector_view_phase(CudaqTranspiler())


class TestQPEBuiltinVectorViewPhase:
    """Public QPE with VectorView phase operands."""

    def test_qiskit_executes_vector_view_phase(self, qiskit_transpiler):
        """Qiskit executes public QPE with a VectorView phase element."""
        _assert_builtin_qpe_vector_view_phase(qiskit_transpiler)

    def test_quri_parts_executes_vector_view_phase(self):
        """QURI Parts executes public QPE with a VectorView phase element."""
        pytest.importorskip("quri_parts")
        from qamomile.quri_parts import QuriPartsTranspiler

        _assert_builtin_qpe_vector_view_phase(QuriPartsTranspiler())

    @pytest.mark.cudaq
    def test_cudaq_executes_vector_view_phase(self):
        """CUDA-Q executes public QPE with a VectorView phase element.

        Runs in ``-m cudaq`` sessions only: loading cudaq into a default
        session is unsafe — see tests/_cudaq_isolation.py.
        """
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        _assert_builtin_qpe_vector_view_phase(CudaqTranspiler())


class TestQPEConsistency:
    """Verify naive QPE and built-in QPE return the same phase estimates."""

    @pytest.mark.parametrize(
        "theta,expected_phase",
        [
            (math.pi / 2, 0.25),
            (math.pi / 4, 0.125),
            (math.pi, 0.5),
        ],
    )
    def test_naive_and_builtin_agree(self, qiskit_transpiler, theta, expected_phase):
        """Both implementations estimate the same phase."""
        # Naive QPE
        naive_exec = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": 3, "phase": theta}
        )
        naive_job = naive_exec.sample(qiskit_transpiler.executor(), shots=1024)
        naive_result = naive_job.result()
        naive_phases = set()
        for bits, count in naive_result.results:
            phase_est = _decode_phase(bits)
            naive_phases.add(round(phase_est, 6))

        # Built-in QPE
        builtin_exec = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": 3, "phase": theta}
        )
        builtin_job = builtin_exec.sample(qiskit_transpiler.executor(), shots=1024)
        builtin_result = builtin_job.result()
        builtin_phases = set()
        for value, count in builtin_result.results:
            builtin_phases.add(round(value, 6))

        # Both return the same phase set
        assert naive_phases == builtin_phases, (
            f"theta={theta}: naive got {naive_phases}, built-in got {builtin_phases}"
        )

        # Expected phase is in the results
        assert expected_phase in naive_phases, (
            f"Expected phase {expected_phase} not in naive results"
        )

    @pytest.mark.parametrize("n_qubits", [3, 5, 7])
    @pytest.mark.parametrize("seed", [901 + i for i in range(5)])
    def test_random_angle_consistency(self, qiskit_transpiler, seed, n_qubits):
        r"""Both QPE implementations return a theoretically valid phase for random angles.

        Mathematical background
        -----------------------
        For a unitary U with eigenvalue e^{2\pi i \phi}, an n-qubit QPE measures
        an integer m \in \{0, ..., 2^n - 1\} with probability:

            P(m) = \frac{\sin^2(2^n \pi (\phi - m/2^n))}
                        {2^n \sin^2(\pi (\phi - m/2^n))}

        The two highest-probability outcomes are the two integers nearest to
        \phi \cdot 2^n, i.e., m_0 = floor(\phi \cdot 2^n) and
        m_1 = ceil(\phi \cdot 2^n). Their combined probability satisfies:

            P(m_0) + P(m_1) >= 8 / \pi^2 \approx 0.81

        even in the worst case (equidistant: \phi \cdot 2^n is a half-integer).
        For non-equidistant phases the dominant outcome is even more concentrated.

        Two-tier verification
        ---------------------
        We define EQUIDISTANT_THRESHOLD = 0.1. Let frac = (\phi \cdot 2^n) mod 1.

        - Non-equidistant (|frac - 0.5| >= 0.1):
          The dominant phase m_0/2^n has probability significantly higher than
          the runner-up m_1/2^n. Specifically, when |frac - 0.5| >= 0.1 the
          gap P(m_0) - P(m_1) is large enough that with 4096 shots the top
          result is deterministic. We assert:
            (a) naive_top == builtin_top  (exact match)
            (b) naive_top in valid_phases (accuracy)

        - Equidistant (|frac - 0.5| < 0.1):
          P(m_0) and P(m_1) are nearly equal, so sampling noise can flip the
          top result between the two implementations. We only assert:
            (a) naive_top in valid_phases
            (b) builtin_top in valid_phases
          Both m_0/2^n and m_1/2^n are correct QPE outputs.

        Bug detection
        -------------
        When the bug is present (power not resolved), the built-in QPE applies
        U^1 on every counting qubit instead of U^{2^k}, producing a phase
        unrelated to the correct \phi. This fails both tier checks for the
        vast majority of random angles.
        """
        EQUIDISTANT_THRESHOLD = 0.1

        rng = random.Random(seed)
        theta = rng.uniform(0, 2 * math.pi)
        shots = 4096

        # True phase and the two nearest representable phases
        phi = (theta / (2 * math.pi)) % 1.0
        phi_scaled = phi * (2**n_qubits)
        lower = (math.floor(phi_scaled) % (2**n_qubits)) / (2**n_qubits)
        upper = (math.ceil(phi_scaled) % (2**n_qubits)) / (2**n_qubits)
        valid_phases = {round(lower, 6), round(upper, 6)}

        # Equidistant detection
        frac = phi_scaled % 1.0
        is_equidistant = abs(frac - 0.5) < EQUIDISTANT_THRESHOLD

        # Naive QPE
        naive_exec = qiskit_transpiler.transpile(
            naive_qpe, bindings={"n": n_qubits, "phase": theta}
        )
        naive_job = naive_exec.sample(qiskit_transpiler.executor(), shots=shots)
        naive_result = naive_job.result()
        naive_phase_counts = {}
        for bits, count in naive_result.results:
            phase_est = _decode_phase(bits)
            naive_phase_counts[round(phase_est, 6)] = count
        naive_top = max(naive_phase_counts, key=naive_phase_counts.get)

        # Built-in QPE
        builtin_exec = qiskit_transpiler.transpile(
            builtin_qpe, bindings={"n": n_qubits, "phase": theta}
        )
        builtin_job = builtin_exec.sample(qiskit_transpiler.executor(), shots=shots)
        builtin_result = builtin_job.result()
        builtin_phase_counts = {}
        for value, count in builtin_result.results:
            builtin_phase_counts[round(value, 6)] = count
        builtin_top = max(builtin_phase_counts, key=builtin_phase_counts.get)

        if is_equidistant:
            # Equidistant: both must be valid QPE results, may differ
            assert round(naive_top, 6) in valid_phases, (
                f"[equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"naive top={naive_top} not in valid {valid_phases}"
            )
            assert round(builtin_top, 6) in valid_phases, (
                f"[equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"builtin top={builtin_top} not in valid {valid_phases}"
            )
        else:
            # Non-equidistant: exact match required
            assert naive_top == pytest.approx(builtin_top, abs=1e-6), (
                f"[non-equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"naive top={naive_top} != builtin top={builtin_top}"
            )
            assert round(naive_top, 6) in valid_phases, (
                f"[non-equidistant] seed={seed}, n={n_qubits}, "
                f"theta={theta:.6f}, phi={phi:.6f}: "
                f"top={naive_top} not in valid {valid_phases}"
            )
