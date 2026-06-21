"""Cross-backend execution tests for ``qmc.select`` and zero-control mode.

Covers the quantum multiplexer (``qmc.select``) and the zero-control
(anti-control) extension of ``qmc.control``, exercising both the sampling
and expectation-value paths on every quantum SDK backend Qamomile ships
with (parametrized via the ``sdk_transpiler`` fixture so each backend leg
runs in its matching ``-m`` session). Inputs are randomized over seeds,
rotation angles, and index basis states, and every supported index /
target handle pattern (scalar ``Qubit``, ``Vector[Qubit]``,
``VectorView[Qubit]``) is exercised.

QURI Parts only ships single-control plus the ``C^nX`` family in its
controlled fallback, so the universally-portable legs here keep their
inner controlled-U to single-control or ``X``-cases; deeper general
multi-control coverage runs Qiskit-only with exact-statevector checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import qamomile.circuit as qm
import qamomile.observable as qm_o
from qamomile.circuit import qkernel
from qamomile.circuit.frontend.handle import Bit, Float, Observable, Qubit, Vector

# ---------------------------------------------------------------------------
# Module-scope case kernels (qkernel needs file-backed source).
# ---------------------------------------------------------------------------


@qkernel
def _id_gate(q: Qubit) -> Qubit:
    """Identity case unitary."""
    return q


@qkernel
def _x_gate(q: Qubit) -> Qubit:
    """Pauli-X case unitary."""
    return qm.x(q)


@qkernel
def _ry_case(q: Qubit, theta: Float) -> Qubit:
    """RY(theta) case unitary."""
    return qm.ry(q, theta)


@qkernel
def _rz_case(q: Qubit, theta: Float) -> Qubit:
    """RZ(theta) case unitary."""
    return qm.rz(q, theta)


@qkernel
def _id_vec(qs: Vector[Qubit]) -> Vector[Qubit]:
    """Identity case unitary over a vector target."""
    return qs


@qkernel
def _apply_select_ry(idx: Qubit, t: Qubit, theta: Float) -> tuple[Qubit, Qubit]:
    """Apply ``select([ry, rz])`` — used to test ``qmc.inverse(select)``."""
    idx, t = qm.select([_ry_case, _rz_case])(idx, t, theta=theta)
    return idx, t


@qkernel
def _apply_anti_cx(c: Qubit, t: Qubit) -> tuple[Qubit, Qubit]:
    """Apply an anti-control X — used to test ``qmc.inverse`` of a zero-control."""
    c, t = qm.control(qm.x, num_controls=1, control_values=(0,))(c, t)
    return c, t


def _executor(case, seed: int):
    """Return a (seeded, where supported) executor for a backend case.

    Args:
        case: The ``SdkTranspilerCase`` from the ``sdk_transpiler`` fixture.
        seed (int): Simulator seed (honoured by Qiskit's BasicSimulator).

    Returns:
        Any: A backend executor.
    """
    transpiler = case.transpiler
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        backend = BasicSimulator()
        backend.set_options(seed_simulator=seed)
        return transpiler.executor(backend=backend)
    return transpiler.executor()


def _expval_atol(case) -> float:
    """Return the expectation-value tolerance for a backend case."""
    return 1e-6 if case.backend_name == "cudaq" else 1e-8


# ---------------------------------------------------------------------------
# Zero-control (anti-control) — cross-backend.
# ---------------------------------------------------------------------------


class TestZeroControlCrossBackend:
    """Anti-control semantics across every supported backend."""

    @pytest.mark.parametrize("ctrl_is_one", [False, True])
    @pytest.mark.parametrize("seed", [0, 7])
    def test_single_anti_control_sampling(self, sdk_transpiler, ctrl_is_one, seed):
        """Single anti-control-X fires iff the control reads ``|0>``."""

        def build(ctrl_one: bool):
            @qkernel
            def circ() -> Bit:
                c = qm.qubit(name="c")
                t = qm.qubit(name="t")
                if ctrl_one:
                    c = qm.x(c)
                c, t = qm.control(qm.x, num_controls=1, control_values=(0,))(c, t)
                return qm.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build(ctrl_is_one))
        job = exe.sample(_executor(sdk_transpiler, seed), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        # Target flips exactly when the control reads |0>.
        expected_target = 0 if ctrl_is_one else 1
        assert set(counts) == {expected_target}, (
            f"{sdk_transpiler.backend_name}: ctrl_one={ctrl_is_one} got {counts}"
        )

    @pytest.mark.parametrize("seed", [1, 5])
    def test_single_anti_control_expval(self, sdk_transpiler, seed):
        """``<Z>`` on the target after anti-control-X with control ``|0>``.

        Control stays ``|0>`` so the anti-control fires: target becomes
        ``|1>`` and ``<Z> = -1``.
        """

        @qkernel
        def circ(obs: Observable) -> Float:
            c = qm.qubit(name="c")
            t = qm.qubit(name="t")
            c, t = qm.control(qm.x, num_controls=1, control_values=(0,))(c, t)
            return qm.expval(t, obs)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ, bindings={"obs": qm_o.Z(0)})
        val = exe.run(transpiler.executor()).result()
        assert np.isclose(val, -1.0, atol=_expval_atol(sdk_transpiler)), (
            f"{sdk_transpiler.backend_name}: expected <Z>=-1, got {val}"
        )

    @pytest.mark.parametrize("pattern", [(0, 0), (0, 1), (1, 0), (1, 1)])
    @pytest.mark.parametrize("prep", [(0, 0), (0, 1), (1, 0), (1, 1)])
    def test_two_control_pattern_sampling(self, sdk_transpiler, pattern, prep):
        """A 2-control-X with ``control_values=pattern`` fires iff prep matches.

        Uses ``X`` as the controlled gate so the inner multi-control reduces
        to ``C^2X`` (a Toffoli), which every backend's fallback supports.
        """

        def build():
            @qkernel
            def circ() -> Bit:
                c0 = qm.qubit(name="c0")
                c1 = qm.qubit(name="c1")
                t = qm.qubit(name="t")
                if prep[0]:
                    c0 = qm.x(c0)
                if prep[1]:
                    c1 = qm.x(c1)
                c0, c1, t = qm.control(qm.x, num_controls=2, control_values=pattern)(
                    c0, c1, t
                )
                return qm.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build())
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expected = 1 if tuple(prep) == tuple(pattern) else 0
        assert set(counts) == {expected}, (
            f"{sdk_transpiler.backend_name}: pattern={pattern} prep={prep} got {counts}"
        )

    def test_inverse_of_anti_control_is_identity(self, sdk_transpiler):
        """``qmc.inverse`` of an anti-control round-trips to identity.

        Regression for ``_invert_controlled_u`` dropping ``control_values``:
        the inverse of an anti-controlled X must itself be an anti-controlled
        X (not a standard control). Control stays ``|0>`` so forward fires
        (target -> |1>) and inverse must undo it (target -> |0>).
        """

        @qkernel
        def circ() -> Bit:
            c = qm.qubit(name="c")
            t = qm.qubit(name="t")
            c, t = _apply_anti_cx(c, t)
            c, t = qm.inverse(_apply_anti_cx)(c, t)
            return qm.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {0}, f"{sdk_transpiler.backend_name}: got {counts}"

    def test_anti_control_vector_handle(self, sdk_transpiler):
        """Anti-control accepts a whole ``Vector[Qubit]`` control register.

        Two controls supplied as one length-2 vector, pattern ``(0, 0)``;
        both controls left ``|0>`` so the gate fires.
        """

        @qkernel
        def circ() -> Bit:
            ctrls = qm.qubit_array(2, name="ctrls")
            t = qm.qubit(name="t")
            ctrls, t = qm.control(qm.x, num_controls=2, control_values=(0, 0))(ctrls, t)
            return qm.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"


# ---------------------------------------------------------------------------
# qmc.select — cross-backend.
# ---------------------------------------------------------------------------


class TestSelectCrossBackend:
    """SELECT (quantum multiplexer) across every supported backend."""

    @pytest.mark.parametrize("index_one", [False, True])
    @pytest.mark.parametrize("seed", [0, 3])
    def test_two_case_select_sampling(self, sdk_transpiler, index_one, seed):
        """``select([_id, X])`` flips the target iff the index reads ``|1>``."""

        def build(idx_one: bool):
            @qkernel
            def circ() -> Bit:
                idx = qm.qubit(name="idx")
                t = qm.qubit(name="t")
                if idx_one:
                    idx = qm.x(idx)
                idx, t = qm.select([_id_gate, _x_gate])(idx, t)
                return qm.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build(index_one))
        job = exe.sample(_executor(sdk_transpiler, seed), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expected = 1 if index_one else 0
        assert set(counts) == {expected}, (
            f"{sdk_transpiler.backend_name}: index_one={index_one} got {counts}"
        )

    @pytest.mark.parametrize("seed", [0, 11, 42])
    def test_two_case_select_expval(self, sdk_transpiler, seed):
        """Parameterized two-case select drives the expectation value.

        ``select([ry, rz])`` with the index in ``|0>`` applies ``ry(theta)``
        to ``|0>``; ``<Z> = cos(theta)``.
        """
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(0.0, 2.0 * math.pi))

        @qkernel
        def circ(theta: Float, obs: Observable) -> Float:
            idx = qm.qubit(name="idx")
            t = qm.qubit(name="t")
            idx, t = qm.select([_ry_case, _rz_case])(idx, t, theta=theta)
            return qm.expval(t, obs)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ, bindings={"theta": theta, "obs": qm_o.Z(0)})
        val = exe.run(transpiler.executor()).result()
        assert np.isclose(val, math.cos(theta), atol=_expval_atol(sdk_transpiler)), (
            f"{sdk_transpiler.backend_name}: theta={theta} expected "
            f"<Z>={math.cos(theta)}, got {val}"
        )

    @pytest.mark.parametrize("index_value", [0, 1, 2, 3])
    def test_four_case_select_sampling(self, sdk_transpiler, index_value):
        """``select([_id, _id, X, _id])`` flips target iff index reads 2.

        Cases are ``X`` so the inner multi-control reduces to ``C^2X``
        (Toffoli), supported on every backend. The index is prepared in a
        big-endian basis state (first index qubit = MSB).
        """

        def build(value: int):
            idx0 = (value >> 1) & 1  # MSB
            idx1 = value & 1

            @qkernel
            def circ() -> Bit:
                idx = qm.qubit_array(2, name="idx")
                if idx0:
                    idx[0] = qm.x(idx[0])
                if idx1:
                    idx[1] = qm.x(idx[1])
                t = qm.qubit(name="t")
                idx, t = qm.select([_id_gate, _id_gate, _x_gate, _id_gate])(idx, t)
                return qm.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build(index_value))
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expected = 1 if index_value == 2 else 0
        assert set(counts) == {expected}, (
            f"{sdk_transpiler.backend_name}: index={index_value} got {counts}"
        )

    def test_inverse_of_select_is_identity(self, sdk_transpiler):
        """``select`` then its inverse returns the target to ``|0>``.

        Cross-backend regression: index ``|0>`` selects ``ry(theta)``; the
        inverse selects ``ry(theta)^dagger``. On CUDA-Q this also guards the
        previous ``cudaq.adjoint``-of-select process abort (now falls back to
        the Qamomile inverse decomposition).
        """

        @qkernel
        def circ() -> Bit:
            idx = qm.qubit(name="idx")  # |0> -> ry case
            t = qm.qubit(name="t")
            idx, t = _apply_select_ry(idx, t, 0.7)
            idx, t = qm.inverse(_apply_select_ry)(idx, t, 0.7)
            return qm.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {0}, f"{sdk_transpiler.backend_name}: got {counts}"

    def test_scalar_case_broadcast_over_vector_target(self, sdk_transpiler):
        """A scalar single-qubit case broadcasts over a ``Vector[Qubit]`` target.

        Regression for ``emit_select`` lacking the vector-broadcast branch:
        ``select([x, h])(idx, reg)`` with a 2-qubit ``reg`` and index ``|0>``
        applies X to every element of ``reg`` (mirroring ``qmc.control``'s
        broadcast convenience), giving ``reg == |11>``.
        """

        @qkernel
        def circ() -> Vector[Bit]:
            idx = qm.qubit(name="idx")  # |0> -> case 0 = X
            reg = qm.qubit_array(2, name="reg")
            idx, reg = qm.select([qm.x, qm.h])(idx, reg)
            return qm.measure(reg)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {(1, 1)}, f"{sdk_transpiler.backend_name}: got {counts}"

    def test_select_index_vectorview_handle(self, sdk_transpiler):
        """SELECT accepts a ``VectorView`` slice as the index register.

        A length-3 register's ``[0:2]`` slice carries the 2-qubit index.
        Prepared to read 2 (big-endian ``reg[0]=1``), selecting the ``X``
        case.
        """

        @qkernel
        def circ() -> Bit:
            reg = qm.qubit_array(3, name="reg")
            reg[0] = qm.x(reg[0])  # MSB high -> index reads 2
            t = qm.qubit(name="t")
            view = reg[0:2]
            view, t = qm.select([_id_gate, _id_gate, _x_gate, _id_gate])(view, t)
            return qm.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"


# ---------------------------------------------------------------------------
# Qiskit-only deep checks: general multi-control + exact statevector.
# ---------------------------------------------------------------------------


def _run_statevector(transpiler, kernel, bindings=None):
    """Transpile to Qiskit and return the pre-measurement statevector."""
    from qiskit.quantum_info import Statevector

    exe = transpiler.transpile(kernel, bindings=bindings or {})
    qc = exe.quantum_circuit.remove_final_measurements(inplace=False)
    return np.asarray(Statevector.from_instruction(qc).data)


class TestSelectQiskitExact:
    """General multi-control SELECT verified against exact statevectors."""

    @pytest.mark.parametrize("index_value", [0, 1, 2, 3])
    def test_four_case_general_gates(self, qiskit_transpiler, index_value):
        """``select([X, Y, Z, H])`` applies the right single-qubit unitary.

        Verifies the post-select statevector on the (index, target) system
        equals applying only the selected case to the target, for an index
        prepared in basis state ``index_value`` (big-endian).
        """
        gates = {
            0: np.array([[0, 1], [1, 0]], dtype=complex),  # X
            1: np.array([[0, -1j], [1j, 0]], dtype=complex),  # Y
            2: np.array([[1, 0], [0, -1]], dtype=complex),  # Z
            3: np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2),  # H
        }

        def build(value: int):
            idx0 = (value >> 1) & 1
            idx1 = value & 1

            @qkernel
            def circ() -> Vector[Bit]:
                idx = qm.qubit_array(2, name="idx")
                if idx0:
                    idx[0] = qm.x(idx[0])
                if idx1:
                    idx[1] = qm.x(idx[1])
                t = qm.qubit(name="t")
                idx, t = qm.select([qm.x, qm.y, qm.z, qm.h])(idx, t)
                return qm.measure(idx)

            return circ

        sv = _run_statevector(qiskit_transpiler, build(index_value))
        # Allocation order idx[0], idx[1], t maps to Qiskit little-endian
        # qubits 0, 1, 2 -> statevector bit0=idx[0] (index MSB),
        # bit1=idx[1] (index LSB), bit2=target.
        target_state = gates[index_value] @ np.array([1, 0], dtype=complex)
        idx0 = (index_value >> 1) & 1  # idx[0], index MSB
        idx1 = index_value & 1  # idx[1], index LSB
        expected = np.zeros(8, dtype=complex)
        for tbit in (0, 1):
            amp = target_state[tbit]
            if amp == 0:
                continue
            full_index = idx0 | (idx1 << 1) | (tbit << 2)
            expected[full_index] = amp
        # Compare up to global phase.
        overlap = np.abs(np.vdot(expected, sv))
        assert np.isclose(overlap, 1.0, atol=1e-8), (
            f"index={index_value}: overlap={overlap}, sv={sv}"
        )

    def test_multi_qubit_target_case(self, qiskit_transpiler, seeded_executor):
        """A two-qubit-target case unitary is selected and applied.

        ``select`` over two 2-qubit unitaries acting on a length-2 target
        register: case 0 entangles (Bell), case 1 is identity. Index
        ``|1>`` selects identity, leaving the targets ``|00>``.
        """

        @qkernel
        def _bell(qs: Vector[Qubit]) -> Vector[Qubit]:
            qs[0] = qm.h(qs[0])
            qs[0], qs[1] = qm.cx(qs[0], qs[1])
            return qs

        @qkernel
        def _idid(qs: Vector[Qubit]) -> Vector[Qubit]:
            return qs

        @qkernel
        def circ() -> Vector[Bit]:
            idx = qm.qubit(name="idx")
            idx = qm.x(idx)  # index reads 1 -> identity case
            reg = qm.qubit_array(2, name="reg")
            idx, reg = qm.select([_bell, _idid])(idx, reg)
            return qm.measure(reg)

        exe = qiskit_transpiler.transpile(circ)
        job = exe.sample(seeded_executor, shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {(0, 0)}, f"got {counts}"


# ---------------------------------------------------------------------------
# Frontend validation (backend-independent).
# ---------------------------------------------------------------------------


class TestSelectValidation:
    """Compose-time validation of ``qmc.select`` and zero-control args."""

    def test_single_case_rejected(self):
        """A select with fewer than two cases raises ``ValueError``."""
        with pytest.raises(ValueError, match="at least 2 cases"):
            qm.select([_x_gate])

    def test_mismatched_signatures_rejected(self):
        """Cases with differing signatures raise ``ValueError``."""
        with pytest.raises(ValueError, match="same parameter signature"):
            qm.select([_x_gate, _ry_case])

    @pytest.mark.parametrize("num_cases", [2, 3, 4, 5, 8])
    def test_num_index_qubits(self, num_cases):
        """Index-qubit count is ``ceil(log2(num_cases))``."""
        gate = qm.select([_id_gate] * num_cases)
        assert gate.num_index_qubits == math.ceil(math.log2(num_cases))
        assert gate.num_cases == num_cases

    def test_control_values_width_mismatch(self):
        """A control_values width that disagrees with num_controls raises."""
        with pytest.raises(ValueError, match="must match"):
            qm.control(qm.x, num_controls=2, control_values=(0, 1, 0))

    def test_control_values_out_of_range_mask(self):
        """An int bit-mask wider than num_controls raises ``ValueError``."""
        with pytest.raises(ValueError, match="does not fit"):
            qm.control(qm.x, num_controls=1, control_values=0b10)

    def test_control_values_bad_entry(self):
        """A non-0/1 sequence entry raises ``ValueError``."""
        with pytest.raises(ValueError, match="must be 0 or 1"):
            qm.control(qm.x, num_controls=2, control_values=(0, 2))

    def test_control_values_symbolic_num_controls_rejected(self):
        """``control_values`` with a symbolic num_controls raises."""
        from qamomile.circuit.frontend.handle import UInt
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        n = UInt(value=Value(type=UIntType(), name="n"))
        with pytest.raises(ValueError, match="concrete int num_controls"):
            qm.control(qm.x, num_controls=n, control_values=(0,))

    def test_mismatched_case_defaults_rejected(self):
        """Cases with the same signature but different defaults are rejected.

        ``qmc.select`` binds the call against the first case (applying its
        defaults) and forwards the values to every case, so differing
        defaults would silently apply the first case's default everywhere.
        """

        @qkernel
        def case_a(q: Qubit, theta: Float = 0.1) -> Qubit:
            return qm.rx(q, theta)

        @qkernel
        def case_b(q: Qubit, theta: Float = 0.9) -> Qubit:
            return qm.rx(q, theta)

        with pytest.raises(ValueError, match="same parameter signature"):
            qm.select([case_a, case_b])


# ---------------------------------------------------------------------------
# Pipeline-integration regressions (Qiskit): allocation, loop, inverse, est.
# ---------------------------------------------------------------------------


class TestSelectPipelineIntegration:
    """Regression coverage for select through the full transpiler pipeline."""

    def test_element_access_of_select_result(self, qiskit_transpiler, seeded_executor):
        """A gate on an element of a select's Vector result resolves.

        Exercises the ResourceAllocator threading select operand->result for
        ``Vector[Qubit]`` targets: ``idx, reg = sel(idx, reg); reg[1] = X``.
        Index ``|0>`` selects the identity case, then X flips ``reg[1]``.
        """

        @qkernel
        def circ() -> Vector[Bit]:
            idx = qm.qubit(name="idx")  # |0> -> identity case
            reg = qm.qubit_array(2, name="reg")
            idx, reg = qm.select([_id_vec, _id_vec])(idx, reg)
            reg[1] = qm.x(reg[1])  # element access of the SELECT result
            return qm.measure(reg)

        exe = qiskit_transpiler.transpile(circ)
        job = exe.sample(seeded_executor, shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {(0, 1)}, f"got {counts}"

    def test_no_dead_wire_after_scalar_select(self, qiskit_transpiler):
        """A gate after a scalar-target select does not over-allocate a wire.

        The ResourceAllocator must thread the select's scalar result so the
        following gate reuses the target qubit instead of allocating a fresh
        (dead) wire. Circuit should use exactly 2 qubits (index + target).
        """

        @qkernel
        def circ() -> Bit:
            idx = qm.qubit(name="idx")
            t = qm.qubit(name="t")
            idx, t = qm.select([_id_gate, _x_gate])(idx, t)
            t = qm.x(t)  # gate on the SELECT result
            return qm.measure(t)

        exe = qiskit_transpiler.transpile(circ)
        assert exe.quantum_circuit.num_qubits == 2, (
            f"expected 2 qubits, got {exe.quantum_circuit.num_qubits}"
        )

    def test_select_inside_loop(self, qiskit_transpiler, seeded_executor):
        """A select inside ``qmc.range`` with ``reg[i]`` targets unrolls.

        Exercises the LoopAnalyzer treating a loop-var-indexed select operand
        as forcing unrolling (Qiskit would otherwise emit a native runtime
        loop that cannot resolve ``reg[i]``). Index ``|1>`` selects X for
        every element, flipping all of ``reg``.
        """

        @qkernel
        def circ() -> Vector[Bit]:
            idx = qm.qubit(name="idx")
            idx = qm.x(idx)  # |1> -> X case for every iteration
            reg = qm.qubit_array(3, name="reg")
            for i in qm.range(3):
                idx, reg[i] = qm.select([_id_gate, _x_gate])(idx, reg[i])
            return qm.measure(reg)

        exe = qiskit_transpiler.transpile(circ)
        job = exe.sample(seeded_executor, shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {(1, 1, 1)}, f"got {counts}"

    def test_estimator_counts_select(self):
        """Resource estimation counts a select (does not silently skip it)."""
        from qamomile.circuit.estimator import count_gates, qubits_counter

        @qkernel
        def circ() -> Bit:
            idx = qm.qubit_array(2, name="idx")
            t = qm.qubit(name="t")
            idx, t = qm.select([_id_gate, _x_gate, _id_gate, _x_gate])(idx, t)
            return qm.measure(t)

        gate_count = count_gates(circ.block)
        # 4 cases -> 4 opaque controlled gates.
        assert int(gate_count.total) == 4, f"got {gate_count.total}"
        # idx (2) + target (1); ancilla-free cases add nothing.
        assert int(qubits_counter(circ.block)) == 3
