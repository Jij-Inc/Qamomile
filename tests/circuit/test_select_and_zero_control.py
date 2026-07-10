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
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit import qkernel
from qamomile.circuit.frontend.handle import Bit, Float, Observable, Qubit, Vector

# ---------------------------------------------------------------------------
# Module-scope case kernels (qkernel needs file-backed source).
# ---------------------------------------------------------------------------


@qkernel
def _id_gate(q: Qubit) -> Qubit:
    """Return a qubit unchanged for an identity SELECT case.

    Args:
        q (Qubit): Target qubit.

    Returns:
        Qubit: The unchanged target qubit.
    """
    return q


@qkernel
def _x_gate(q: Qubit) -> Qubit:
    """Apply Pauli-X for a SELECT case.

    Args:
        q (Qubit): Target qubit.

    Returns:
        Qubit: The target after Pauli-X.
    """
    return qmc.x(q)


@qkernel
def _delegating_x_gate(q: Qubit) -> Qubit:
    """Delegate to another qkernel so the case initially contains a call op.

    Args:
        q (Qubit): Target qubit.

    Returns:
        Qubit: The target after the delegated Pauli-X.
    """
    return _x_gate(q)


@qkernel
def _ry_case(q: Qubit, theta: Float) -> Qubit:
    """Apply an RY rotation for a parameterized SELECT case.

    Args:
        q (Qubit): Target qubit.
        theta (Float): Rotation angle in radians.

    Returns:
        Qubit: The rotated target qubit.
    """
    return qmc.ry(q, theta)


@qkernel
def _rz_case(q: Qubit, theta: Float) -> Qubit:
    """Apply an RZ rotation for a parameterized SELECT case.

    Args:
        q (Qubit): Target qubit.
        theta (Float): Rotation angle in radians.

    Returns:
        Qubit: The rotated target qubit.
    """
    return qmc.rz(q, theta)


@qkernel
def _id_vec(qs: Vector[Qubit]) -> Vector[Qubit]:
    """Return a vector target unchanged for an identity SELECT case.

    Args:
        qs (Vector[Qubit]): Target register.

    Returns:
        Vector[Qubit]: The unchanged target register.
    """
    return qs


@qkernel
def _id_pair(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
    """Return two target qubits in their input positions.

    Args:
        q0 (Qubit): First target qubit.
        q1 (Qubit): Second target qubit.

    Returns:
        tuple[Qubit, Qubit]: Targets in their original order.
    """
    return q0, q1


@qkernel
def _reverse_pair_handles(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
    """Return two target handles in reversed order without a physical gate.

    Args:
        q0 (Qubit): First target qubit.
        q1 (Qubit): Second target qubit.

    Returns:
        tuple[Qubit, Qubit]: Targets in reversed handle order.
    """
    return q1, q0


@qkernel
def _swap_pair(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
    """Apply an explicit SWAP while preserving output positions.

    Args:
        q0 (Qubit): First target qubit.
        q1 (Qubit): Second target qubit.

    Returns:
        tuple[Qubit, Qubit]: Positional outputs after the physical SWAP.
    """
    q0, q1 = qmc.swap(q0, q1)
    return q0, q1


@qkernel
def _apply_select_ry(idx: Qubit, t: Qubit, theta: Float) -> tuple[Qubit, Qubit]:
    """Apply ``select([ry, rz])`` for inverse tests.

    Args:
        idx (Qubit): SELECT index qubit.
        t (Qubit): Target qubit.
        theta (Float): Shared rotation angle in radians.

    Returns:
        tuple[Qubit, Qubit]: Updated index and target qubits.
    """
    idx, t = qmc.select([_ry_case, _rz_case])(idx, t, theta=theta)
    return idx, t


@qkernel
def _apply_anti_cx(c: Qubit, t: Qubit) -> tuple[Qubit, Qubit]:
    """Apply an anti-controlled X for inverse tests.

    Args:
        c (Qubit): Anti-control qubit.
        t (Qubit): Target qubit.

    Returns:
        tuple[Qubit, Qubit]: Updated control and target qubits.
    """
    c, t = qmc.control(qmc.x, num_controls=1, control_values=(0,))(c, t)
    return c, t


@qkernel
def _select_x_body(idx: Qubit, t: Qubit) -> tuple[Qubit, Qubit]:
    """Apply X only for index ``|1>`` under an enclosing control.

    Args:
        idx (Qubit): SELECT index qubit.
        t (Qubit): Target qubit.

    Returns:
        tuple[Qubit, Qubit]: Updated index and target qubits.
    """
    idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
    return idx, t


@qkernel
def _select_zero_x_body(idx: Qubit, t: Qubit) -> tuple[Qubit, Qubit]:
    """Apply X only for index ``|0>`` under an enclosing control.

    Args:
        idx (Qubit): SELECT index qubit.
        t (Qubit): Target qubit.

    Returns:
        tuple[Qubit, Qubit]: Updated index and target qubits.
    """
    idx, t = qmc.select([_x_gate, _id_gate])(idx, t)
    return idx, t


def _executor(case: Any, seed: int) -> Any:
    """Return a (seeded, where supported) executor for a backend case.

    Args:
        case (SdkTranspilerCase): The backend case from the
            ``sdk_transpiler`` fixture (exposes ``transpiler`` and
            ``backend_name``).
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


def _runtime_control_executor(case: Any) -> Any:
    """Return an executor that supports measurement-backed control flow.

    Args:
        case (SdkTranspilerCase): The backend case from the
            ``sdk_transpiler`` fixture.

    Returns:
        Any: A backend executor capable of running runtime if/while circuits.
    """
    transpiler = case.transpiler
    if case.backend_name == "qiskit":
        from qiskit_aer import AerSimulator

        return transpiler.executor(backend=AerSimulator(method="statevector"))
    return transpiler.executor()


def _expval_atol(case: Any) -> float:
    """Return the expectation-value tolerance for a backend case.

    Args:
        case (SdkTranspilerCase): The backend case from the
            ``sdk_transpiler`` fixture (exposes ``backend_name``).

    Returns:
        float: The absolute tolerance to use for ``<Z>`` comparisons —
            looser for CUDA-Q's single-precision statevector, tight for the
            double-precision Qiskit / QURI Parts simulators.
    """
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

        def build(ctrl_one: bool) -> Any:
            """Build the sampling kernel for one prepared control state.

            Args:
                ctrl_one (bool): Whether to prepare the control in ``|1>``.

            Returns:
                Any: The constructed QKernel.
            """

            @qkernel
            def circ() -> Bit:
                c = qmc.qubit(name="c")
                t = qmc.qubit(name="t")
                if ctrl_one:
                    c = qmc.x(c)
                c, t = qmc.control(qmc.x, num_controls=1, control_values=(0,))(c, t)
                return qmc.measure(t)

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
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            c, t = qmc.control(qmc.x, num_controls=1, control_values=(0,))(c, t)
            return qmc.expval(t, obs)

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

        def build() -> Any:
            """Build the kernel for one control-pattern/preparation pair.

            Returns:
                Any: The constructed QKernel.
            """

            @qkernel
            def circ() -> Bit:
                c0 = qmc.qubit(name="c0")
                c1 = qmc.qubit(name="c1")
                t = qmc.qubit(name="t")
                if prep[0]:
                    c0 = qmc.x(c0)
                if prep[1]:
                    c1 = qmc.x(c1)
                c0, c1, t = qmc.control(qmc.x, num_controls=2, control_values=pattern)(
                    c0, c1, t
                )
                return qmc.measure(t)

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
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            c, t = _apply_anti_cx(c, t)
            c, t = qmc.inverse(_apply_anti_cx)(c, t)
            return qmc.measure(t)

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
            ctrls = qmc.qubit_array(2, name="ctrls")
            t = qmc.qubit(name="t")
            ctrls, t = qmc.control(qmc.x, num_controls=2, control_values=(0, 0))(
                ctrls, t
            )
            return qmc.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"

    @pytest.mark.parametrize("spec", [0b01, (1, 0)])
    @pytest.mark.parametrize(
        "c0_one,c1_one",
        [(False, False), (True, False), (False, True), (True, True)],
    )
    def test_int_mask_matches_sequence(self, sdk_transpiler, spec, c0_one, c1_one):
        """An int ``ctrl_state`` mask and the equal 0/1 sequence agree.

        ``control_values=0b01`` over two controls means control 0 activates
        on ``|1>`` (bit 0 set) and control 1 activates on ``|0>`` (bit 1
        clear), i.e. the sequence ``(1, 0)`` — Qiskit ``ctrl_state``
        convention where bit ``j`` is control ``j``. Both spellings must
        fire the target exactly when ``c0`` reads ``|1>`` and ``c1`` reads
        ``|0>``. Parametrizing over both spellings and all four control
        preparations pins the mask interpretation and their equivalence.
        """

        def build() -> Any:
            """Build the kernel for one mask spelling and preparation.

            Returns:
                Any: The constructed QKernel.
            """

            @qkernel
            def circ() -> Bit:
                c0 = qmc.qubit(name="c0")
                c1 = qmc.qubit(name="c1")
                t = qmc.qubit(name="t")
                if c0_one:
                    c0 = qmc.x(c0)
                if c1_one:
                    c1 = qmc.x(c1)
                c0, c1, t = qmc.control(qmc.x, num_controls=2, control_values=spec)(
                    c0, c1, t
                )
                return qmc.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build())
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expect_flip = c0_one and not c1_one
        expected = {1} if expect_flip else {0}
        assert set(counts) == expected, (
            f"{sdk_transpiler.backend_name}: spec={spec!r} "
            f"c0_one={c0_one} c1_one={c1_one} got {counts}"
        )

    @pytest.mark.parametrize("outer_one", [False, True])
    @pytest.mark.parametrize("inner_zero", [False, True])
    def test_nested_anti_control_composes(self, sdk_transpiler, outer_one, inner_zero):
        """Controlling an anti-controlled kernel keeps the inner anti-control.

        End-to-end correctness guard for composing controls over a
        zero-control: ``qmc.control(_apply_anti_cx)`` adds an outer control
        ``o`` over the inner anti-controlled X on ``(c, t)``, so ``t`` flips
        iff ``o`` reads ``|1>`` (outer active) AND ``c`` reads ``|0>`` (inner
        anti-control active). Each backend realizes this through a different
        path (Qiskit/QuriParts bake the inner block into a reusable
        controlled gate; CUDA-Q emits a controlled source kernel), and this
        checks every one produces the same, correct pattern — so a
        regression in any nested-control lowering that dropped the inner
        anti-control would surface here.
        """

        def build() -> Any:
            """Build the nested-control kernel for the parametrized states.

            Returns:
                Any: The constructed QKernel.
            """

            @qkernel
            def circ() -> Bit:
                o = qmc.qubit(name="o")
                c = qmc.qubit(name="c")
                t = qmc.qubit(name="t")
                if outer_one:
                    o = qmc.x(o)
                if not inner_zero:
                    c = qmc.x(c)  # inner_zero=False -> c reads |1> (anti inactive)
                o, c, t = qmc.control(_apply_anti_cx)(o, c, t)
                return qmc.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build())
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expect_flip = outer_one and inner_zero
        expected = {1} if expect_flip else {0}
        assert set(counts) == expected, (
            f"{sdk_transpiler.backend_name}: outer_one={outer_one} "
            f"inner_zero={inner_zero} got {counts}"
        )


# ---------------------------------------------------------------------------
# qmc.select — cross-backend.
# ---------------------------------------------------------------------------


class TestSelectCrossBackend:
    """SELECT (quantum multiplexer) across every supported backend."""

    @pytest.mark.parametrize("index_one", [False, True])
    @pytest.mark.parametrize("seed", [0, 3])
    def test_two_case_select_sampling(self, sdk_transpiler, index_one, seed):
        """``select([_id, X])`` flips the target iff the index reads ``|1>``."""

        def build(idx_one: bool) -> Any:
            """Build the two-case SELECT kernel for one index state.

            Args:
                idx_one (bool): Whether to prepare the index in ``|1>``.

            Returns:
                Any: The constructed QKernel.
            """

            @qkernel
            def circ() -> Bit:
                idx = qmc.qubit(name="idx")
                t = qmc.qubit(name="t")
                if idx_one:
                    idx = qmc.x(idx)
                idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
                return qmc.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build(index_one))
        job = exe.sample(_executor(sdk_transpiler, seed), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expected = 1 if index_one else 0
        assert set(counts) == {expected}, (
            f"{sdk_transpiler.backend_name}: index_one={index_one} got {counts}"
        )

    def test_case_qkernel_calls_are_inlined(self, sdk_transpiler):
        """A case composed from another qkernel executes instead of disappearing.

        SELECT owns its case blocks outside ``HasNestedOps``. The inline pass
        must therefore descend into every case explicitly; otherwise the
        delegated X remains a residual ``CallBlockOperation`` and is rejected
        by fallback backends or silently omitted by reusable-gate conversion.
        """

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            idx = qmc.x(idx)
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _delegating_x_gate])(idx, t)
            return qmc.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        counts = {
            bits: cnt
            for bits, cnt in exe.sample(_executor(sdk_transpiler, 0), shots=256)
            .result()
            .results
        }
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"

    def test_select_nested_under_control(self, sdk_transpiler):
        """An outer controlled-U composes its control with SELECT's index.

        With both the outer control and index in ``|1>``, the selected X case
        must flip the target. This pins the gate-by-gate fallback used by
        backends that cannot convert the enclosing block to a reusable gate.
        """

        @qkernel
        def circ() -> Bit:
            outer = qmc.qubit(name="outer")
            outer = qmc.x(outer)
            idx = qmc.qubit(name="idx")
            idx = qmc.x(idx)
            t = qmc.qubit(name="t")
            outer, idx, t = qmc.control(_select_x_body)(outer, idx, t)
            return qmc.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        counts = {
            bits: cnt
            for bits, cnt in exe.sample(_executor(sdk_transpiler, 0), shots=256)
            .result()
            .results
        }
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"

    @pytest.mark.parametrize("outer_one", [False, True])
    @pytest.mark.parametrize("index_one", [False, True])
    def test_zero_index_case_nested_under_control(
        self, sdk_transpiler, outer_one, index_one
    ):
        """Outer control composes with a SELECT case using an index anti-control."""

        @qkernel
        def circ() -> Bit:
            outer = qmc.qubit(name="outer")
            if outer_one:
                outer = qmc.x(outer)
            idx = qmc.qubit(name="idx")
            if index_one:
                idx = qmc.x(idx)
            t = qmc.qubit(name="t")
            outer, idx, t = qmc.control(_select_zero_x_body)(outer, idx, t)
            return qmc.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        counts = {
            bits: cnt
            for bits, cnt in exe.sample(_executor(sdk_transpiler, 0), shots=256)
            .result()
            .results
        }
        expected = 1 if outer_one and not index_one else 0
        assert set(counts) == {expected}, (
            f"{sdk_transpiler.backend_name}: outer={outer_one} "
            f"index={index_one} got {counts}"
        )

    def test_returned_index_vector_element_remains_mapped(self, sdk_transpiler):
        """A whole-vector SELECT index result can be indexed and measured.

        The returned index vector wraps the same next-versioned array that
        parents the SELECT's scalar index results. Measuring its first element
        after selecting a case must therefore preserve the prepared ``|1>``.
        """

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit_array(2, name="idx")
            idx[0] = qmc.x(idx[0])
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _id_gate, _x_gate, _id_gate])(idx, t)
            return qmc.measure(idx[0])

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        counts = {
            bits: cnt
            for bits, cnt in exe.sample(_executor(sdk_transpiler, 0), shots=256)
            .result()
            .results
        }
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"

    def test_select_inside_for_loop(self, sdk_transpiler):
        """SELECT threads index and target values through a nested for loop.

        The index is ``|1>`` and each iteration selects X for one loop-indexed
        target element. The symbolic element access forces backend-independent
        unrolling and guards loop-carried index mapping across iterations.
        """

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            idx = qmc.x(idx)
            reg = qmc.qubit_array(2, name="reg")
            for i in qmc.range(2):
                idx, reg[i] = qmc.select([_id_gate, _x_gate])(idx, reg[i])
            return qmc.measure(reg)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        counts = {
            bits: cnt
            for bits, cnt in exe.sample(_executor(sdk_transpiler, 0), shots=256)
            .result()
            .results
        }
        assert set(counts) == {(1, 1)}, f"{sdk_transpiler.backend_name}: got {counts}"

    def test_select_inside_runtime_if(self, sdk_transpiler):
        """SELECT results merge correctly out of a measurement-backed if.

        A deterministically true measured condition executes the selected X
        case. The index and target are both live after the branch, exercising
        merge construction and physical result aliasing for ``SelectOperation``.
        """
        if sdk_transpiler.backend_name == "quri_parts":
            pytest.skip("QURI Parts has no runtime if/else circuit primitive")

        @qkernel
        def circ() -> Bit:
            guard = qmc.qubit(name="guard")
            guard = qmc.x(guard)
            condition = qmc.measure(guard)
            idx = qmc.qubit(name="idx")
            idx = qmc.x(idx)
            t = qmc.qubit(name="t")
            if condition:
                idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
            return qmc.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        counts = {
            bits: cnt
            for bits, cnt in exe.sample(
                _runtime_control_executor(sdk_transpiler), shots=256
            )
            .result()
            .results
        }
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"

    def test_select_inside_runtime_while(self, sdk_transpiler):
        """SELECT threads its quantum results through a one-iteration while.

        The initial measured condition is deterministically true. The body
        applies the selected X case and replaces the loop condition with a
        deterministic zero measurement, so the target must flip exactly once.
        """
        if sdk_transpiler.backend_name == "quri_parts":
            pytest.skip("QURI Parts has no runtime while-loop circuit primitive")

        @qkernel
        def circ() -> Bit:
            guard = qmc.qubit(name="guard")
            guard = qmc.x(guard)
            condition = qmc.measure(guard)
            idx = qmc.qubit(name="idx")
            idx = qmc.x(idx)
            t = qmc.qubit(name="t")
            while condition:
                idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
                stop = qmc.qubit(name="stop")
                condition = qmc.measure(stop)
            return qmc.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        counts = {
            bits: cnt
            for bits, cnt in exe.sample(
                _runtime_control_executor(sdk_transpiler), shots=256
            )
            .result()
            .results
        }
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"

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
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_ry_case, _rz_case])(idx, t, theta=theta)
            return qmc.expval(t, obs)

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

        def build(value: int) -> Any:
            """Build a four-case SELECT for one big-endian index value.

            Args:
                value (int): Index basis-state value in ``[0, 3]``.

            Returns:
                Any: The constructed QKernel.
            """

            idx0 = (value >> 1) & 1  # MSB
            idx1 = value & 1

            @qkernel
            def circ() -> Bit:
                idx = qmc.qubit_array(2, name="idx")
                if idx0:
                    idx[0] = qmc.x(idx[0])
                if idx1:
                    idx[1] = qmc.x(idx[1])
                t = qmc.qubit(name="t")
                idx, t = qmc.select([_id_gate, _id_gate, _x_gate, _id_gate])(idx, t)
                return qmc.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build(index_value))
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expected = 1 if index_value == 2 else 0
        assert set(counts) == {expected}, (
            f"{sdk_transpiler.backend_name}: index={index_value} got {counts}"
        )

    @pytest.mark.parametrize("index_value", [0, 1, 2, 3])
    def test_non_power_of_two_select_out_of_range_identity(
        self, sdk_transpiler, index_value
    ):
        """A 3-case select leaves the unaddressed index value 3 as identity.

        Three cases need two index qubits (four basis states), so index
        value 3 is out of range and must act as identity — no case block
        applies. With ``cases = [_id, X, X]`` the target flips only for
        index 1 and 2; index 0 (identity case) and index 3 (unaddressed)
        both leave the target ``|0>``. This exercises the non-power-of-two
        case count and the out-of-range identity semantics on every backend.
        """

        def build(value: int) -> Any:
            """Build a three-case SELECT for one index basis state.

            Args:
                value (int): Index basis-state value in ``[0, 3]``.

            Returns:
                Any: The constructed QKernel.
            """

            idx0 = (value >> 1) & 1  # MSB
            idx1 = value & 1

            @qkernel
            def circ() -> Bit:
                idx = qmc.qubit_array(2, name="idx")
                if idx0:
                    idx[0] = qmc.x(idx[0])
                if idx1:
                    idx[1] = qmc.x(idx[1])
                t = qmc.qubit(name="t")
                idx, t = qmc.select([_id_gate, _x_gate, _x_gate])(idx, t)
                return qmc.measure(t)

            return circ

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(build(index_value))
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        expected = 1 if index_value in (1, 2) else 0
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
            idx = qmc.qubit(name="idx")  # |0> -> ry case
            t = qmc.qubit(name="t")
            idx, t = _apply_select_ry(idx, t, 0.7)
            idx, t = qmc.inverse(_apply_select_ry)(idx, t, 0.7)
            return qmc.measure(t)

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
            idx = qmc.qubit(name="idx")  # |0> -> case 0 = X
            reg = qmc.qubit_array(2, name="reg")
            idx, reg = qmc.select([qmc.x, qmc.h])(idx, reg)
            return qmc.measure(reg)

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
            reg = qmc.qubit_array(3, name="reg")
            reg[0] = qmc.x(reg[0])  # MSB high -> index reads 2
            t = qmc.qubit(name="t")
            view = reg[0:2]
            view, t = qmc.select([_id_gate, _id_gate, _x_gate, _id_gate])(view, t)
            return qmc.measure(t)

        transpiler = sdk_transpiler.transpiler
        exe = transpiler.transpile(circ)
        job = exe.sample(_executor(sdk_transpiler, 0), shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: got {counts}"


# ---------------------------------------------------------------------------
# Qiskit-only deep checks: general multi-control + exact statevector.
# ---------------------------------------------------------------------------


def _run_statevector(
    transpiler: Any,
    kernel: Any,
    bindings: dict[str, Any] | None = None,
) -> np.ndarray:
    """Transpile to Qiskit and return the pre-measurement statevector.

    Args:
        transpiler (QiskitTranspiler): The Qiskit transpiler fixture.
        kernel (QKernel): The kernel to transpile and simulate.
        bindings (dict[str, Any] | None): Compile-time bindings, or ``None``
            for none.

    Returns:
        np.ndarray: The statevector amplitudes with final measurements
            stripped, in Qiskit little-endian qubit order.
    """
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

        def build(value: int) -> Any:
            """Build the exact-statevector case for one index value.

            Args:
                value (int): Index basis-state value in ``[0, 3]``.

            Returns:
                Any: The constructed QKernel.
            """

            idx0 = (value >> 1) & 1
            idx1 = value & 1

            @qkernel
            def circ() -> Vector[Bit]:
                idx = qmc.qubit_array(2, name="idx")
                if idx0:
                    idx[0] = qmc.x(idx[0])
                if idx1:
                    idx[1] = qmc.x(idx[1])
                t = qmc.qubit(name="t")
                idx, t = qmc.select([qmc.x, qmc.y, qmc.z, qmc.h])(idx, t)
                return qmc.measure(idx)

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
            if np.isclose(amp, 0.0, atol=1e-12):
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
            qs[0] = qmc.h(qs[0])
            qs[0], qs[1] = qmc.cx(qs[0], qs[1])
            return qs

        @qkernel
        def _idid(qs: Vector[Qubit]) -> Vector[Qubit]:
            return qs

        @qkernel
        def circ() -> Vector[Bit]:
            idx = qmc.qubit(name="idx")
            idx = qmc.x(idx)  # index reads 1 -> identity case
            reg = qmc.qubit_array(2, name="reg")
            idx, reg = qmc.select([_bell, _idid])(idx, reg)
            return qmc.measure(reg)

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
            qmc.select([_x_gate])

    def test_mismatched_signatures_rejected(self):
        """Cases with differing signatures raise ``ValueError``."""
        with pytest.raises(ValueError, match="same parameter signature"):
            qmc.select([_x_gate, _ry_case])

    @pytest.mark.parametrize("num_cases", [2, 3, 4, 5, 8])
    def test_num_index_qubits(self, num_cases):
        """Index-qubit count is ``ceil(log2(num_cases))``."""
        gate = qmc.select([_id_gate] * num_cases)
        assert gate.num_index_qubits == math.ceil(math.log2(num_cases))
        assert gate.num_cases == num_cases

    def test_control_values_width_mismatch(self):
        """A control_values width that disagrees with num_controls raises."""
        with pytest.raises(ValueError, match="must match"):
            qmc.control(qmc.x, num_controls=2, control_values=(0, 1, 0))

    def test_control_values_out_of_range_mask(self):
        """An int bit-mask wider than num_controls raises ``ValueError``."""
        with pytest.raises(ValueError, match="does not fit"):
            qmc.control(qmc.x, num_controls=1, control_values=0b10)

    def test_control_values_bad_entry(self):
        """A non-0/1 sequence entry raises ``ValueError``."""
        with pytest.raises(ValueError, match="must be 0 or 1"):
            qmc.control(qmc.x, num_controls=2, control_values=(0, 2))

    def test_control_values_symbolic_num_controls_rejected(self):
        """``control_values`` with a symbolic num_controls raises."""
        from qamomile.circuit.frontend.handle import UInt
        from qamomile.circuit.ir.types.primitives import UIntType
        from qamomile.circuit.ir.value import Value

        n = UInt(value=Value(type=UIntType(), name="n"))
        with pytest.raises(ValueError, match="concrete int num_controls"):
            qmc.control(qmc.x, num_controls=n, control_values=(0,))

    def test_mismatched_case_defaults_rejected(self):
        """Cases with the same signature but different defaults are rejected.

        ``qmc.select`` binds the call against the first case (applying its
        defaults) and forwards the values to every case, so differing
        defaults would silently apply the first case's default everywhere.
        """

        @qkernel
        def case_a(q: Qubit, theta: Float = 0.1) -> Qubit:
            return qmc.rx(q, theta)

        @qkernel
        def case_b(q: Qubit, theta: Float = 0.9) -> Qubit:
            return qmc.rx(q, theta)

        with pytest.raises(ValueError, match="same parameter signature"):
            qmc.select([case_a, case_b])

    def test_mismatched_case_output_arity_rejected(self):
        """A case that returns extra quantum state (higher arity) is rejected.

        ``_signature_key`` only checks inputs, so a case that allocates and
        returns an extra qubit shares the input signature yet is not a
        unitary on the target register. Building the call must fail loudly
        rather than silently miswire against the shared result list.
        """

        @qkernel
        def _returns_extra(q: Qubit) -> tuple[Qubit, Qubit]:
            anc = qmc.qubit(name="anc")
            anc = qmc.x(anc)
            return q, anc

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_x_gate, _returns_extra])(idx, t)
            return qmc.measure(t)

        with pytest.raises(ValueError, match="not a unitary on the target"):
            _ = circ.block

    def test_duplicate_case_output_rejected(self):
        """A case cannot repeat one input wire to fake a larger output."""

        @qkernel
        def _returns_duplicate(q: Qubit) -> tuple[Qubit, Qubit]:
            return q, q

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_x_gate, _returns_duplicate])(idx, t)
            return qmc.measure(t)

        with pytest.raises(ValueError, match="not a unitary on the target"):
            _ = circ.block

    def test_bare_case_output_permutation_rejected(self):
        """A quantum-index-dependent handle relabel must use an explicit gate."""

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            idx, q0, q1 = qmc.select([_id_pair, _reverse_pair_handles])(idx, q0, q1)
            return qmc.measure(q0)

        with pytest.raises(ValueError, match="bare return-value permutations"):
            _ = circ.block

    def test_explicit_case_swap_is_accepted(self):
        """An explicit SWAP gate preserves positional outputs and is valid."""

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            q0 = qmc.qubit(name="q0")
            q1 = qmc.qubit(name="q1")
            idx, q0, q1 = qmc.select([_id_pair, _swap_pair])(idx, q0, q1)
            return qmc.measure(q0)

        _ = circ.block

    def test_case_dropping_target_for_fresh_qubit_rejected(self):
        """A same-arity case that drops the target and returns a fresh qubit fails.

        The footprint check must go beyond output *count*: a case with the
        same quantum input/output arity can still drop the target register
        and return a freshly-allocated qubit instead. Comparing quantum
        logical-id sets (input wires vs output wires) catches this where a
        count-only check would not.
        """

        @qkernel
        def _drop_and_fresh(q: Qubit) -> Qubit:
            anc = qmc.qubit(name="anc")
            anc = qmc.x(anc)
            return anc  # same arity (1) but drops the input target q

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_x_gate, _drop_and_fresh])(idx, t)
            return qmc.measure(t)

        with pytest.raises(ValueError, match="not a unitary on the target"):
            _ = circ.block


class TestSelectAndControlIrValidation:
    """IR-boundary validation for hand-built / decoded select and control_values."""

    def test_concrete_controlled_rejects_non_binary_control_value(self):
        """A ``ConcreteControlledU`` with a non-0/1 activation entry raises."""
        from qamomile.circuit.ir.operation.gate import ConcreteControlledU

        with pytest.raises(ValueError, match="must be 0 or 1"):
            ConcreteControlledU(
                operands=[], results=[], num_controls=1, control_values=(2,)
            )

    @pytest.mark.parametrize("entry", [False, 0.0])
    def test_concrete_controlled_rejects_non_int_control_value(self, entry):
        """IR activation entries reject bool/float values equal to 0 or 1."""
        from qamomile.circuit.ir.operation.gate import ConcreteControlledU

        with pytest.raises(ValueError, match="Python ints"):
            ConcreteControlledU(
                operands=[], results=[], num_controls=1, control_values=(entry,)
            )

    def test_concrete_controlled_rejects_width_mismatch(self):
        """A ``control_values`` length that disagrees with num_controls raises."""
        from qamomile.circuit.ir.operation.gate import ConcreteControlledU

        with pytest.raises(ValueError, match="one 0/1 value per control"):
            ConcreteControlledU(
                operands=[], results=[], num_controls=2, control_values=(0,)
            )

    def test_concrete_controlled_accepts_empty_and_valid(self):
        """The empty pattern and a valid per-control tuple both construct."""
        from qamomile.circuit.ir.operation.gate import ConcreteControlledU

        # Empty (all-standard) and a matching 0/1 tuple must not raise.
        ConcreteControlledU(operands=[], results=[], num_controls=2)
        ConcreteControlledU(
            operands=[], results=[], num_controls=2, control_values=(0, 1)
        )

    def test_concrete_controlled_normalizes_all_ones(self):
        """An all-ones ``control_values`` collapses to the canonical ``()``.

        ``(1, 1)`` means every control is a standard ``1``-control, which is
        semantically identical to ``()``. ``__post_init__`` must normalize it
        so a hand-built or decoded all-ones pattern hashes and serializes
        identically to the empty marker rather than as a distinct anti-control
        pattern.
        """
        from qamomile.circuit.ir.operation.gate import ConcreteControlledU

        op = ConcreteControlledU(
            operands=[], results=[], num_controls=2, control_values=(1, 1)
        )
        assert op.control_values == ()

    def test_select_rejects_empty_case_blocks(self):
        """A ``SelectOperation`` with no case blocks raises ``ValueError``."""
        from qamomile.circuit.ir.operation.select import SelectOperation

        with pytest.raises(ValueError, match="at least one case block"):
            SelectOperation(operands=[], results=[], num_index_qubits=1, case_blocks=[])

    @pytest.mark.parametrize("num_index_qubits", [True, 1.0])
    def test_select_rejects_non_int_index_width(self, num_index_qubits):
        """IR index width rejects bool/float values before bit arithmetic."""
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.select import SelectOperation

        with pytest.raises(ValueError, match="must be a Python int"):
            SelectOperation(
                operands=[],
                results=[],
                num_index_qubits=num_index_qubits,
                case_blocks=[Block()],
            )


# ---------------------------------------------------------------------------
# Pipeline-integration regressions (Qiskit): allocation, loop, inverse, est.
# ---------------------------------------------------------------------------


class TestSelectPipelineIntegration:
    """Regression coverage for select through the full transpiler pipeline."""

    def test_visualization_renders_select_box(self):
        """Circuit visualization shows SELECT instead of silently omitting it."""
        from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
        from qamomile.circuit.visualization.style import DEFAULT_STYLE
        from qamomile.circuit.visualization.visual_ir import VGate, VGateKind

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
            return qmc.measure(t)

        graph = circ._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        visual = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
        nodes = [
            node
            for node in visual.children
            if isinstance(node, VGate) and node.label == "SELECT"
        ]
        assert len(nodes) == 1
        assert nodes[0].kind == VGateKind.COMPOSITE_BOX
        assert nodes[0].qubit_indices == [0, 1]

    def test_folded_loop_visualization_keeps_select_summary(self):
        """A folded for-loop body includes its SELECT expression summary."""
        from qamomile.circuit.visualization.analyzer import CircuitAnalyzer
        from qamomile.circuit.visualization.style import DEFAULT_STYLE
        from qamomile.circuit.visualization.visual_ir import VFoldedBlock

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            for _ in qmc.range(2):
                idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
            return qmc.measure(t)

        graph = circ._build_graph_for_visualization()
        analyzer = CircuitAnalyzer(graph, DEFAULT_STYLE)
        qubit_map, qubit_names, num_qubits = analyzer.build_qubit_map(graph)
        visual = analyzer.build_visual_ir(graph, qubit_map, qubit_names, num_qubits)
        folded = [node for node in visual.children if isinstance(node, VFoldedBlock)]
        assert len(folded) == 1
        assert any("select[2]" in line for line in folded[0].body_lines)

    def test_element_access_of_select_result(self, qiskit_transpiler, seeded_executor):
        """A gate on an element of a select's Vector result resolves.

        Exercises the ResourceAllocator threading select operand->result for
        ``Vector[Qubit]`` targets: ``idx, reg = sel(idx, reg); reg[1] = X``.
        Index ``|0>`` selects the identity case, then X flips ``reg[1]``.
        """

        @qkernel
        def circ() -> Vector[Bit]:
            idx = qmc.qubit(name="idx")  # |0> -> identity case
            reg = qmc.qubit_array(2, name="reg")
            idx, reg = qmc.select([_id_vec, _id_vec])(idx, reg)
            reg[1] = qmc.x(reg[1])  # element access of the SELECT result
            return qmc.measure(reg)

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
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
            t = qmc.x(t)  # gate on the SELECT result
            return qmc.measure(t)

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
            idx = qmc.qubit(name="idx")
            idx = qmc.x(idx)  # |1> -> X case for every iteration
            reg = qmc.qubit_array(3, name="reg")
            for i in qmc.range(3):
                idx, reg[i] = qmc.select([_id_gate, _x_gate])(idx, reg[i])
            return qmc.measure(reg)

        exe = qiskit_transpiler.transpile(circ)
        job = exe.sample(seeded_executor, shots=256)
        counts = {bits: cnt for bits, cnt in job.result().results}
        assert set(counts) == {(1, 1, 1)}, f"got {counts}"

    def test_estimator_counts_select(self):
        """Resource estimation counts a select (does not silently skip it)."""
        from qamomile.circuit.estimator import count_gates, qubits_counter

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit_array(2, name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _x_gate, _id_gate, _x_gate])(idx, t)
            return qmc.measure(t)

        gate_count = count_gates(circ.block)
        # 4 cases -> 4 opaque controlled gates.
        assert int(gate_count.total) == 4, f"got {gate_count.total}"
        # idx (2) + target (1); ancilla-free cases add nothing.
        assert int(qubits_counter(circ.block)) == 3


# ---------------------------------------------------------------------------
# Serialization + content-hash (backend-independent IR properties).
# ---------------------------------------------------------------------------


def _affine(kernel: Any) -> Any:
    """Inline a kernel to an AFFINE block for serialize / content-hash tests.

    Args:
        kernel (QKernel): A ``@qkernel``-decorated function.

    Returns:
        Block: The kernel's block after ``InlinePass`` removes every
            ``CallBlockOperation``.
    """
    from qamomile.circuit.transpiler.passes.inline import InlinePass

    return InlinePass().run(kernel.block)


def _first_select(block: Any) -> Any:
    """Return the first ``SelectOperation`` in an inlined block.

    Args:
        block (Block): An AFFINE block containing a select.

    Returns:
        SelectOperation: The first select operation found.
    """
    from qamomile.circuit.ir.operation.select import SelectOperation

    return next(op for op in block.operations if isinstance(op, SelectOperation))


def _first_controlled(block: Any) -> Any:
    """Return the first ``ConcreteControlledU`` in an inlined block.

    Args:
        block (Block): An AFFINE block containing a controlled-U.

    Returns:
        ConcreteControlledU: The first concrete controlled-U found.
    """
    from qamomile.circuit.ir.operation.gate import ConcreteControlledU

    return next(op for op in block.operations if isinstance(op, ConcreteControlledU))


class TestSelectSerializationAndHash:
    """Round-trip and content-hash coverage for select / control_values IR."""

    @pytest.mark.parametrize("encode_decode", ["json", "msgpack"])
    def test_select_roundtrips(self, encode_decode):
        """A SelectOperation survives JSON and msgpack round-trips intact.

        Uses a non-power-of-two (3-case) select so both the case count and
        the ``num_index_qubits`` are non-trivial after the round-trip.
        """
        from qamomile.circuit.ir.serialize import (
            dump_json,
            dump_msgpack,
            load_json,
            load_msgpack,
        )

        @qkernel
        def circ() -> Vector[Bit]:
            idx = qmc.qubit_array(2, name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _x_gate, _id_gate])(idx, t)
            return qmc.measure(idx)

        block = _affine(circ)
        original = _first_select(block)
        if encode_decode == "json":
            loaded = load_json(dump_json(block))
        else:
            loaded = load_msgpack(dump_msgpack(block))
        restored = _first_select(loaded)
        assert restored.num_index_qubits == original.num_index_qubits == 2
        assert len(restored.case_blocks) == len(original.case_blocks) == 3

    @pytest.mark.parametrize("encode_decode", ["json", "msgpack"])
    def test_control_values_roundtrips(self, encode_decode):
        """An anti-control's ``control_values`` survives serialization."""
        from qamomile.circuit.ir.serialize import (
            dump_json,
            dump_msgpack,
            load_json,
            load_msgpack,
        )

        @qkernel
        def circ() -> Bit:
            c0 = qmc.qubit(name="c0")
            c1 = qmc.qubit(name="c1")
            t = qmc.qubit(name="t")
            c0, c1, t = qmc.control(qmc.x, num_controls=2, control_values=(0, 1))(
                c0, c1, t
            )
            return qmc.measure(t)

        block = _affine(circ)
        assert _first_controlled(block).control_values == (0, 1)
        if encode_decode == "json":
            loaded = load_json(dump_json(block))
        else:
            loaded = load_msgpack(dump_msgpack(block))
        assert _first_controlled(loaded).control_values == (0, 1)

    def test_decoder_rejects_non_int_control_values(self):
        """Wire decoding does not coerce bool activation values to integers."""
        from qamomile.circuit.ir.serialize import from_dict, to_dict

        @qkernel
        def circ() -> Bit:
            c = qmc.qubit(name="c")
            t = qmc.qubit(name="t")
            c, t = qmc.control(qmc.x, control_values=(0,))(c, t)
            return qmc.measure(t)

        payload = to_dict(_affine(circ))
        controlled = next(
            op
            for op in payload["block"]["operations"]
            if op["$type"] == "ConcreteControlledU"
        )
        controlled["control_values"] = [False]
        with pytest.raises(ValueError, match="list of Python int"):
            from_dict(payload)

    def test_decoder_rejects_non_int_select_width(self):
        """Wire decoding does not coerce a string SELECT width to an integer."""
        from qamomile.circuit.ir.serialize import from_dict, to_dict

        @qkernel
        def circ() -> Bit:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
            return qmc.measure(t)

        payload = to_dict(_affine(circ))
        select_op = next(
            op
            for op in payload["block"]["operations"]
            if op["$type"] == "SelectOperation"
        )
        select_op["num_index_qubits"] = "1"
        with pytest.raises(ValueError, match="must be a Python int"):
            from_dict(payload)

    def test_content_hash_distinguishes_case_order(self):
        """``content_hash`` differs when only the case order differs.

        Two selects over the same two distinct unitaries in swapped order
        are different programs, so their canonical content hashes must
        differ; the identical spelling must hash equally.
        """
        from qamomile.circuit.ir.canonical import content_hash

        @qkernel
        def circ_a() -> Vector[Bit]:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
            return qmc.measure(idx)

        @qkernel
        def circ_b() -> Vector[Bit]:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_x_gate, _id_gate])(idx, t)
            return qmc.measure(idx)

        @qkernel
        def circ_a2() -> Vector[Bit]:
            idx = qmc.qubit(name="idx")
            t = qmc.qubit(name="t")
            idx, t = qmc.select([_id_gate, _x_gate])(idx, t)
            return qmc.measure(idx)

        assert content_hash(_affine(circ_a)) != content_hash(_affine(circ_b))
        assert content_hash(_affine(circ_a)) == content_hash(_affine(circ_a2))

    def test_content_hash_distinguishes_control_values(self):
        """``content_hash`` differs when only ``control_values`` differs.

        The anti-control activation pattern is functional (it changes the
        emitted circuit), so it must participate in the canonical hash; an
        identical pattern must hash equally.
        """
        from qamomile.circuit.ir.canonical import content_hash

        def build(pattern: tuple[int, ...]) -> Any:
            """Build a controlled-X kernel for one activation pattern.

            Args:
                pattern (tuple[int, ...]): Per-control activation values.

            Returns:
                Any: The constructed QKernel.
            """

            @qkernel
            def circ() -> Bit:
                c0 = qmc.qubit(name="c0")
                c1 = qmc.qubit(name="c1")
                t = qmc.qubit(name="t")
                c0, c1, t = qmc.control(qmc.x, num_controls=2, control_values=pattern)(
                    c0, c1, t
                )
                return qmc.measure(t)

            return circ

        assert content_hash(_affine(build((0, 1)))) != content_hash(
            _affine(build((1, 0)))
        )
        assert content_hash(_affine(build((0, 1)))) == content_hash(
            _affine(build((0, 1)))
        )
