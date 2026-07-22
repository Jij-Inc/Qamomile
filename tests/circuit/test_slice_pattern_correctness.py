"""Cross-backend correctness for representative Vector slicing patterns.

This file groups slice tests into two complementary halves so any
regression — silent miscompilation on the legal side, swallowed error
on the illegal side — surfaces loudly under one ``pytest`` invocation.

**Legal patterns** (transpile + execute + resource estimate).  Each
kernel is a small, self-contained example of a way users are expected
to write slicing code — evens / odds Bell-pair loop, ``qft`` on a
slice, ``cast`` to ``QFixed``, in-line ``q[a:b] = h(q[a:b])`` broadcast,
top-level slice with a body loop, Python-style auto-truncation, and
passing a view as a sub-kernel argument.  Each kernel is transpiled
on every supported SDK (Qiskit / QuriParts / CUDA-Q, parametrized via
the ``sv_backend`` fixture whose quri_parts / cudaq params carry their
markers), the result statevector compared to an analytical Qiskit
reference (or ``run()`` for the QFixed Float-return kernel), and
``estimate_resources`` pinned to the expected qubit / gate counts.
A backend-agnostic IR-level resource estimator is intentionally
re-asserted here so the per-pattern claim stays auditable.

**Illegal patterns** (error-raising), grouped by concept:

- ``TestStrictReturnViolations`` — every transfer-style view consume
  (element loop, broadcast, sub-kernel call, ``pauli_evolve``) must
  be followed by a slice-assign return; otherwise ``measure(q)``
  raises ``UnreturnedBorrowError``.
- ``TestMultiViewOverlap`` — two live views that share parent slots
  are rejected at trace time by ``VectorView._wrap``.
- ``TestSliceAssignmentConsistency`` — left- and right-hand sides of
  ``q[a:b] = ...`` must cover the same slot set.
- ``TestControlFlowBodyViolations`` — releasing an outer-registered
  view from inside a loop / branch body is post-fold-rejected by
  ``SliceBorrowCheckPass``'s outer-snapshot guard.
- ``TestNestedSliceReturnOrder`` — nested slices must come back
  inner→outer→root; inner-direct-to-root, outer-while-inner-live,
  and outer-element-at-inner-slot all raise.

Each failure case asserts both the error type and the detection point
(frontend trace-time vs. post-fold IR pass) so the regression
surfaces are explicit.

Helper convention used throughout the file: kernels appear next to
the test that exercises them.  When a kernel is shared across the
``test_*`` methods of a single class, it lives inside that class as
a class-level ``@qmc.qkernel`` attribute (Python looks ``self._kern``
up via the class's ``__dict__``; ``QKernel`` is not a descriptor so
no spurious ``self``-binding happens).  When a kernel is used by
exactly one test method it is defined inline at the top of that
method.  Only the cross-backend statevector helpers and the
analytical reference builder are module-level — they are genuinely
shared by every legal-pattern test class.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import qamomile.circuit as qmc

qiskit = pytest.importorskip("qiskit")
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402

from qamomile.circuit.estimator import estimate_resources  # noqa: E402
from qamomile.qiskit import QiskitTranspiler  # noqa: E402
from tests.transpiler.gate_test_specs import statevectors_equal  # noqa: E402

# ---------------------------------------------------------------------------
# Cross-backend helpers (used by every legal-pattern test class below).
# ---------------------------------------------------------------------------


def _qiskit_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile with Qiskit and return the unitary statevector.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings forwarded to
            ``transpile``.

    Returns:
        np.ndarray: Complex amplitudes after stripping the final
            measurement instructions from the compiled circuit.
    """
    transpiler = QiskitTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    qc = exe.compiled_quantum[0].circuit
    return np.array(Statevector(qc.remove_final_measurements(inplace=False)).data)


def _quri_parts_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile with QuriParts and read the unitary statevector via Qulacs.

    Skips if ``quri_parts`` / ``quri_parts.qulacs`` is unavailable.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings.

    Returns:
        np.ndarray: Complex amplitudes from the Qulacs simulator.
    """
    pytest.importorskip("quri_parts")
    pytest.importorskip("quri_parts.qulacs")
    from quri_parts.core.state import GeneralCircuitQuantumState
    from quri_parts.qulacs.simulator import evaluate_state_to_vector

    from qamomile.quri_parts import QuriPartsTranspiler

    transpiler = QuriPartsTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    qp_circuit = exe.compiled_quantum[0].circuit
    if hasattr(qp_circuit, "parameter_count") and qp_circuit.parameter_count > 0:
        bound = qp_circuit.bind_parameters([0.0] * qp_circuit.parameter_count)
    elif hasattr(qp_circuit, "bind_parameters"):
        bound = qp_circuit.bind_parameters([])
    else:
        bound = qp_circuit
    state = GeneralCircuitQuantumState(bound.qubit_count, bound)
    return np.array(evaluate_state_to_vector(state).vector)


def _cudaq_statevector(kern, bindings: dict) -> np.ndarray:
    """Transpile with CUDA-Q and read the unitary statevector.

    Skips if ``cudaq`` is unavailable.

    Args:
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings.

    Returns:
        np.ndarray: Complex amplitudes from the CUDA-Q state simulator.
    """
    pytest.importorskip("cudaq")
    import cudaq

    from qamomile.cudaq import CudaqTranspiler

    transpiler = CudaqTranspiler()
    exe = transpiler.transpile(kern, bindings=bindings)
    cudaq_circuit = exe.compiled_quantum[0].circuit
    return np.array(cudaq.get_state(cudaq_circuit.kernel_func))


_SV_BACKEND_HELPERS = {
    "qiskit": _qiskit_statevector,
    "quri_parts": _quri_parts_statevector,
    "cudaq": _cudaq_statevector,
}


@pytest.fixture(
    params=[
        pytest.param("qiskit", id="qiskit"),
        pytest.param("quri_parts", marks=pytest.mark.quri_parts, id="quri_parts"),
        pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
    ]
)
def sv_backend(request):
    """Backend name for single-backend statevector assertions.

    The quri_parts / cudaq params carry their markers so each leg only
    runs in the matching ``-m`` session; in particular the cudaq leg
    must never load cudaq into a default session (see
    tests/_cudaq_isolation.py). The SDK import itself happens lazily
    inside the per-backend statevector helper.

    Args:
        request (pytest.FixtureRequest): Parametrization carrier.

    Returns:
        str: One of ``"qiskit"``, ``"quri_parts"``, ``"cudaq"``.
    """
    return request.param


def _assert_backend_matches(
    backend: str, kern, bindings: dict, expected_sv: np.ndarray
):
    """Assert that one backend's statevector matches the reference.

    Args:
        backend (str): Backend name supplied by the ``sv_backend``
            fixture.
        kern: Compiled qkernel.
        bindings (dict): Compile-time bindings.
        expected_sv (np.ndarray): Reference unitary statevector.
    """
    actual = _SV_BACKEND_HELPERS[backend](kern, bindings)
    assert statevectors_equal(actual, expected_sv), (
        f"[{backend}] statevector mismatch.\n"
        f"  actual:   {actual}\n"
        f"  expected: {expected_sv}"
    )


def _reference_statevector(n_qubits: int, h_qubits: list[int]) -> np.ndarray:
    """Build a reference statevector: apply H to each of ``h_qubits`` on |0>.

    Args:
        n_qubits (int): Total qubits in the reference circuit.
        h_qubits (list[int]): Qubit indices the kernel should apply
            ``H`` to.

    Returns:
        np.ndarray: Complex amplitudes after applying H to those qubits.
    """
    qc = QuantumCircuit(n_qubits)
    for q in h_qubits:
        qc.h(q)
    return np.array(Statevector(qc).data)


# ---------------------------------------------------------------------------
# Legal slicing patterns — each class defines its own kernel inline.
# ---------------------------------------------------------------------------


class TestEvensOddsCxPattern:
    """H on evens then CX(evens[i], odds[i]) produces Bell pairs."""

    @qmc.qkernel
    def _kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Evens / odds + H + CX, explicit slice-assign return."""
        q = qmc.qubit_array(n, "q")
        evens = q[0::2]
        odds = q[1::2]
        for i in qmc.range(odds.shape[0]):
            evens[i] = qmc.h(evens[i])
            evens[i], odds[i] = qmc.cx(evens[i], odds[i])
        q[0::2] = evens
        q[1::2] = odds
        return qmc.measure(q)

    @pytest.mark.parametrize("n", [4, 6])
    def test_transpile_and_statevector_match(self, sv_backend, n: int):
        """Cross-backend statevector matches ``H_evens then CX(evens, odds)``."""
        bindings = {"n": n}
        ref_qc = QuantumCircuit(n)
        n_pairs = n // 2
        for i in range(n_pairs):
            ref_qc.h(2 * i)
        for i in range(n_pairs):
            ref_qc.cx(2 * i, 2 * i + 1)
        expected_sv = np.array(Statevector(ref_qc).data)
        _assert_backend_matches(sv_backend, self._kern, bindings, expected_sv)

    @pytest.mark.parametrize(
        "n, exp_qubits, exp_single, exp_two",
        [(4, 4, 2, 2), (6, 6, 3, 3)],
    )
    def test_resource_estimate(
        self, n: int, exp_qubits: int, exp_single: int, exp_two: int
    ):
        """Resource estimate matches H + CX counts inside the loop."""
        est = estimate_resources(self._kern.block, bindings={"n": n})
        assert int(est.qubits) == exp_qubits
        assert int(est.gates.single_qubit) == exp_single
        assert int(est.gates.two_qubit) == exp_two
        assert int(est.gates.total) == exp_single + exp_two


class TestQftOnView:
    """``qft(q[lo:hi])`` applies QFT to the parent's slice."""

    @qmc.qkernel
    def _kern(lo: qmc.UInt, hi: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """``qft`` on a slice view, explicit slice-assign return."""
        q = qmc.qubit_array(8, "q")
        view = q[lo:hi]
        view = qmc.qft(view)
        q[lo:hi] = view
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self, sv_backend):
        """``qft(|000>) = |+++>``; the rest of the register stays |0>."""
        bindings = {"lo": 1, "hi": 4}
        # ``qft(|000>) = (1/sqrt(8)) * sum_k |k>`` for a 3-qubit register,
        # so on the full 8-qubit |00000000> input the slice ``q[1:4]``
        # becomes a uniform superposition while q[0] / q[4..7] stay |0>.
        # The expected statevector has amplitude 1/sqrt(8) at every
        # basis state with q[0]=q[4]=q[5]=q[6]=q[7]=0 (any q[1..3]).
        n_full = 8
        expected_sv = np.zeros(1 << n_full, dtype=complex)
        amp = 1.0 / math.sqrt(8)
        for k in range(8):
            # Qiskit indexes qubit 0 as the least-significant bit.
            idx = (k & 0b1) << 1 | ((k >> 1) & 0b1) << 2 | ((k >> 2) & 0b1) << 3
            expected_sv[idx] = amp
        _assert_backend_matches(sv_backend, self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        """``qft`` is wrapped as a composite gate; the top-level block reports 0 leaf gates.

        ``estimate_resources`` walks the IR ``Block`` at the level the
        kernel produced it.  ``CompositeGateOperation`` is one logical
        op, not a tree of H / CP, so leaf gate counters return zero.
        The qubit count (8 — the full ``qubit_array`` size) is still
        reported correctly.  This is intentional behaviour, not a slice
        regression; we pin it so any future estimator change that
        starts unpacking composite gates is noticed.
        """
        est = estimate_resources(self._kern.block, bindings={"lo": 1, "hi": 4})
        assert int(est.qubits) == 8
        assert int(est.gates.total) == 0


class TestCastSliceToQFixed:
    """``cast(q[1::2], QFixed)`` then ``measure(qf)``.

    The kernel applies no gates, so every qubit stays |0>.  The QFixed
    decode of two |0> qubits with ``int_bits=0`` is the value 0.0.
    """

    @qmc.qkernel
    def _kern() -> qmc.Float:
        """Cast a slice view to ``QFixed`` and measure."""
        q = qmc.qubit_array(4, "q")
        qf = qmc.cast(q[1::2], qmc.QFixed, int_bits=0)
        return qmc.measure(qf)

    def test_transpile_and_run_returns_zero(self):
        """The Float result of measuring |0>^2 as QFixed is 0.0."""
        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(self._kern)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_quri_parts_run_returns_zero(self):
        """Cross-backend: QuriParts ``run`` also returns 0.0."""
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        transpiler = QuriPartsTranspiler()
        exe = transpiler.transpile(self._kern)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.cudaq
    def test_cudaq_run_returns_zero(self):
        """Cross-backend: CUDA-Q ``run`` also returns 0.0.

        Runs in ``-m cudaq`` sessions only: loading cudaq into a default
        session is unsafe — see tests/_cudaq_isolation.py.
        """
        pytest.importorskip("cudaq")
        from qamomile.cudaq import CudaqTranspiler

        transpiler = CudaqTranspiler()
        exe = transpiler.transpile(self._kern)
        got = exe.run(transpiler.executor()).result()
        assert float(got) == pytest.approx(0.0, abs=1e-9)

    def test_resource_estimate(self):
        """4 qubits allocated, no gates."""
        est = estimate_resources(self._kern.block)
        assert int(est.qubits) == 4
        assert int(est.gates.total) == 0


class TestInlineSliceAssignBroadcast:
    """In-place ``q[1:n-2] = qmc.h(q[1:n-2])``.

    With ``n=6`` the slice covers ``q[1:4] = {1, 2, 3}``, so the
    expected unitary is H on those three qubits.
    """

    @qmc.qkernel
    def _kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """In-place ``q[1:n-2] = qmc.h(q[1:n-2])`` slice assignment."""
        q = qmc.qubit_array(n, "q")
        q[1 : n - 2] = qmc.h(q[1 : n - 2])
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self, sv_backend):
        bindings = {"n": 6}
        expected_sv = _reference_statevector(6, [1, 2, 3])
        _assert_backend_matches(sv_backend, self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block, bindings={"n": 6})
        assert int(est.qubits) == 6
        assert int(est.gates.single_qubit) == 3
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 3


# Module-scope sub-kernels used by ``TestInlineCallSliceBroadcast``.
# They live at module scope (rather than as class attributes) for the
# same name-resolution reason as ``_h_all``: the entry kernels reference
# them by bare name, which is looked up via the function's
# ``__globals__``.


@qmc.qkernel
def _slice_broadcast_inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply ``qs[0:2] = qmc.h(qs[0:2])`` inside a sub-kernel.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register whose first two
            qubits are broadcast-H'd in-place.

    Returns:
        qmc.Vector[qmc.Qubit]: The same register with the slice
            assignment applied.
    """
    qs[0:2] = qmc.h(qs[0:2])
    return qs


@qmc.qkernel
def _symbolic_slice_inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply ``qs[0:n-1] = qmc.h(qs[0:n-1])`` with symbolic bound.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register; the slice covers
            every element except the last.

    Returns:
        qmc.Vector[qmc.Qubit]: The same register with the symbolic
            slice broadcast applied.
    """
    n = qs.shape[0]
    qs[0 : n - 1] = qmc.h(qs[0 : n - 1])
    return qs


@qmc.qkernel
def _nested_slice_inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply a slice-of-slice broadcast inside a sub-kernel.

    Builds ``s1 = qs[0:4]`` (a view) and then broadcasts H on
    ``s1[0:2]`` (a view of a view).  The two-level slice chain
    stresses recursive ``slice_of`` rewriting through ``InlinePass``.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 4.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with H applied to
            ``qs[0]`` and ``qs[1]``.
    """
    s1 = qs[0:4]
    s1[0:2] = qmc.h(s1[0:2])
    qs[0:4] = s1
    return qs


@qmc.qkernel
def _multi_slice_inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply two independent slice broadcasts in the same sub-kernel.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 4.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with H applied to every
            element via two non-overlapping slice broadcasts.
    """
    qs[0:2] = qmc.h(qs[0:2])
    qs[2:4] = qmc.h(qs[2:4])
    return qs


@qmc.qkernel
def _slice_and_scalar_inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Mix a slice broadcast with a scalar element gate.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 3.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with H applied to
            ``qs[0]`` / ``qs[1]`` via slice broadcast and to ``qs[2]``
            via a scalar element gate.
    """
    qs[0:2] = qmc.h(qs[0:2])
    qs[2] = qmc.h(qs[2])
    return qs


@qmc.qkernel
def _slice_inner_for_two_level(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Innermost slice broadcast for the two-level inline chain.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 2.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with H applied to
            ``qs[0]`` and ``qs[1]``.
    """
    qs[0:2] = qmc.h(qs[0:2])
    return qs


@qmc.qkernel
def _slice_mid_for_two_level(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Mid layer that forwards to ``_slice_inner_for_two_level``.

    Used as the intermediate level in the entry → mid → innermost
    chain so the slice broadcast is reached after two levels of
    inlining.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 2.

    Returns:
        qmc.Vector[qmc.Qubit]: The same register, with the innermost
            slice broadcast applied.
    """
    qs = _slice_inner_for_two_level(qs)
    return qs


# Deep-chain helpers for the four-level inline test.  Each layer simply
# forwards to the next, so the entire chain only does one slice
# broadcast at the innermost level.  The chain stresses that
# ``InlinePass`` rewrites slice metadata consistently across an
# arbitrary number of inline levels — the recursive ``substitute_value``
# in ``ValueSubstitutor`` is bounded only by the depth of the call /
# slice graph, so a chain that compiles at depth 4 also compiles at
# any greater depth.


@qmc.qkernel
def _deep_chain_l0(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Innermost layer of the deep-inline chain — performs the slice broadcast.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 2.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with H applied to
            ``qs[0]`` and ``qs[1]``.
    """
    qs[0:2] = qmc.h(qs[0:2])
    return qs


@qmc.qkernel
def _deep_chain_l1(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Second-to-innermost forwarder for the deep-inline chain.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 2.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with the innermost slice
            broadcast applied.
    """
    qs = _deep_chain_l0(qs)
    return qs


@qmc.qkernel
def _deep_chain_l2(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Mid forwarder for the deep-inline chain.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 2.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with the innermost slice
            broadcast applied.
    """
    qs = _deep_chain_l1(qs)
    return qs


@qmc.qkernel
def _deep_chain_l3(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Outermost forwarder for the deep-inline chain (entry sees this).

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 2.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with the innermost slice
            broadcast applied.
    """
    qs = _deep_chain_l2(qs)
    return qs


# Companion helpers that wrap a slice-of-slice inside a multi-level
# inline chain — the slice broadcast lives two levels of inlining deep
# AND inside a slice view of a slice, so the recursion through
# ``substitute_value`` must walk both the inline chain and the slice
# chain at the same time.


@qmc.qkernel
def _nested_slice_innermost(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Innermost layer for the nested-slice deep-inline chain.

    Builds ``s1 = qs[0:4]`` and broadcasts H on ``s1[0:2]``.  The two
    slice levels are walked by emit-time root resolution; the outer
    inline call must have rewritten ``s1.slice_of`` to the caller's
    actual register for that walk to terminate at a real qubit.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 4.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with H applied to
            ``qs[0]`` and ``qs[1]``.
    """
    s1 = qs[0:4]
    s1[0:2] = qmc.h(s1[0:2])
    qs[0:4] = s1
    return qs


@qmc.qkernel
def _nested_slice_mid(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Mid layer that forwards to ``_nested_slice_innermost``.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Quantum register of length >= 4.

    Returns:
        qmc.Vector[qmc.Qubit]: The register with the slice-of-slice
            broadcast applied at the innermost level.
    """
    qs = _nested_slice_innermost(qs)
    return qs


@qmc.qkernel
def _view_arg_slicer(view: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Re-slice a ``Vector[Qubit]`` parameter and apply a slice broadcast.

    The caller is expected to pass a ``VectorView`` (e.g.
    ``q[0::2]``); the sub-kernel further slices it with ``view[0:2]``
    and applies H.  The two slice levels chain through ``InlinePass``.

    Args:
        view (qmc.Vector[qmc.Qubit]): Vector or view of length >= 2.

    Returns:
        qmc.Vector[qmc.Qubit]: The same handle with H applied to
            ``view[0]`` and ``view[1]``.
    """
    view[0:2] = qmc.h(view[0:2])
    return view


class TestInlineCallSliceBroadcast:
    """Slice broadcast lives in a sub-kernel and is reached via inline.

    Companion to :class:`TestInlineSliceAssignBroadcast`, which covers
    the same ``qs[a:b] = qmc.h(qs[a:b])`` pattern written directly in
    the entry kernel.  This class is a regression net for the bug
    where ``InlinePass`` left a sliced ``ArrayValue``'s ``slice_of``
    pointing at the callee's cloned ``qs`` parameter even after the
    parameter itself was substituted to the caller's register.  The
    emit pass then chased the stale UUID on the element values inside
    the broadcast loop and aborted with ``QubitIndexResolutionError``.

    Each test transpiles on every supported backend and compares the
    resulting statevector to an analytical Qiskit reference, so a
    structural regression that compiles but emits the wrong qubit
    indices is also caught.
    """

    @qmc.qkernel
    def _kern_concrete() -> qmc.Vector[qmc.Bit]:
        """Concrete slice ``qs[0:2]`` broadcast reached through inline."""
        qs = qmc.qubit_array(3, "qs")
        qs = _slice_broadcast_inner(qs)
        return qmc.measure(qs)

    def test_concrete_transpile_and_statevector_match(self, sv_backend):
        """The inlined ``qs[0:2] = qmc.h(qs[0:2])`` applies H to qs[0]/qs[1] only."""
        expected_sv = _reference_statevector(3, [0, 1])
        _assert_backend_matches(sv_backend, self._kern_concrete, {}, expected_sv)

    @qmc.qkernel
    def _kern_symbolic(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Symbolic slice ``qs[0:n-1]`` broadcast reached through inline."""
        qs = qmc.qubit_array(n, "qs")
        qs = _symbolic_slice_inner(qs)
        return qmc.measure(qs)

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_symbolic_transpile_and_statevector_match(self, sv_backend, n: int):
        """``qs[0:n-1]`` covers every qubit except the last across multiple sizes."""
        expected_sv = _reference_statevector(n, list(range(n - 1)))
        _assert_backend_matches(sv_backend, self._kern_symbolic, {"n": n}, expected_sv)

    @qmc.qkernel
    def _kern_nested_slice() -> qmc.Vector[qmc.Bit]:
        """Slice-of-slice broadcast reached through inline."""
        qs = qmc.qubit_array(6, "qs")
        qs = _nested_slice_inner(qs)
        return qmc.measure(qs)

    def test_nested_slice_transpile_and_statevector_match(self, sv_backend):
        """``s1 = qs[0:4]; s1[0:2] = h(s1[0:2])`` applies H to qs[0]/qs[1]."""
        expected_sv = _reference_statevector(6, [0, 1])
        _assert_backend_matches(sv_backend, self._kern_nested_slice, {}, expected_sv)

    @qmc.qkernel
    def _kern_multi_slice() -> qmc.Vector[qmc.Bit]:
        """Two independent slice broadcasts inside the inlined sub-kernel."""
        qs = qmc.qubit_array(4, "qs")
        qs = _multi_slice_inner(qs)
        return qmc.measure(qs)

    def test_multi_slice_transpile_and_statevector_match(self, sv_backend):
        """Both ``qs[0:2]`` and ``qs[2:4]`` broadcasts survive inlining."""
        expected_sv = _reference_statevector(4, [0, 1, 2, 3])
        _assert_backend_matches(sv_backend, self._kern_multi_slice, {}, expected_sv)

    @qmc.qkernel
    def _kern_slice_and_scalar() -> qmc.Vector[qmc.Bit]:
        """Slice broadcast plus a scalar element gate inside the sub-kernel."""
        qs = qmc.qubit_array(3, "qs")
        qs = _slice_and_scalar_inner(qs)
        return qmc.measure(qs)

    def test_slice_and_scalar_transpile_and_statevector_match(self, sv_backend):
        """Mixing slice and scalar element gates in one inlined sub-kernel works."""
        expected_sv = _reference_statevector(3, [0, 1, 2])
        _assert_backend_matches(
            sv_backend, self._kern_slice_and_scalar, {}, expected_sv
        )

    @qmc.qkernel
    def _kern_two_level_inline() -> qmc.Vector[qmc.Bit]:
        """Entry → mid → innermost chain; slice broadcast lives in the innermost."""
        qs = qmc.qubit_array(3, "qs")
        qs = _slice_mid_for_two_level(qs)
        return qmc.measure(qs)

    def test_two_level_inline_transpile_and_statevector_match(self, sv_backend):
        """Slice metadata survives two levels of inlining."""
        expected_sv = _reference_statevector(3, [0, 1])
        _assert_backend_matches(
            sv_backend, self._kern_two_level_inline, {}, expected_sv
        )

    @qmc.qkernel
    def _kern_view_arg_inner_slice(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Pass ``q[0::2]`` to a sub-kernel that further slices it."""
        q = qmc.qubit_array(num, "q")
        evens = q[0::2]
        evens = _view_arg_slicer(evens)
        q[0::2] = evens
        return qmc.measure(q)

    def test_view_arg_inner_slice_transpile_and_statevector_match(self, sv_backend):
        """Slice chain ``q[0::2][0:2]`` resolves to root qubits q[0] / q[2]."""
        # num=6: evens = q[0::2] = {q[0], q[2], q[4]}.  The sub-kernel
        # applies H to view[0:2] = {evens[0], evens[1]} = {q[0], q[2]}.
        expected_sv = _reference_statevector(6, [0, 2])
        _assert_backend_matches(
            sv_backend, self._kern_view_arg_inner_slice, {"num": 6}, expected_sv
        )

    @qmc.qkernel
    def _kern_four_level_inline() -> qmc.Vector[qmc.Bit]:
        """Four-level inline chain (entry → l3 → l2 → l1 → l0) with slice broadcast at the innermost.

        Exercises that ``InlinePass`` rewrites slice metadata
        consistently across an arbitrary depth of inline calls; the
        recursive ``substitute_value`` in ``ValueSubstitutor`` is
        bounded only by the depth of the call / slice graph, so a
        depth-4 chain that compiles also compiles at any greater
        depth.
        """
        qs = qmc.qubit_array(3, "qs")
        qs = _deep_chain_l3(qs)
        return qmc.measure(qs)

    def test_four_level_inline_transpile_and_statevector_match(self, sv_backend):
        """Slice metadata survives four levels of inlining."""
        expected_sv = _reference_statevector(3, [0, 1])
        _assert_backend_matches(
            sv_backend, self._kern_four_level_inline, {}, expected_sv
        )

    @qmc.qkernel
    def _kern_nested_slice_two_level_inline() -> qmc.Vector[qmc.Bit]:
        """Nested slice (slice-of-slice) reached through two levels of inline."""
        qs = qmc.qubit_array(6, "qs")
        qs = _nested_slice_mid(qs)
        return qmc.measure(qs)

    def test_nested_slice_two_level_inline_transpile_and_statevector_match(
        self, sv_backend
    ):
        """Slice-of-slice broadcast through two levels of inlining."""
        expected_sv = _reference_statevector(6, [0, 1])
        _assert_backend_matches(
            sv_backend, self._kern_nested_slice_two_level_inline, {}, expected_sv
        )


class TestTopLevelSliceWithBodyLoop:
    """``evens = q[0::2]; for i ...; q[0::2] = evens``."""

    @qmc.qkernel
    def _kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Top-level slice + per-element loop + top-level release."""
        q = qmc.qubit_array(n, "q")
        evens = q[0::2]
        for i in qmc.range(evens.shape[0]):
            evens[i] = qmc.h(evens[i])
        q[0::2] = evens
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self, sv_backend):
        bindings = {"n": 4}
        expected_sv = _reference_statevector(4, [0, 2])
        _assert_backend_matches(sv_backend, self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block, bindings={"n": 4})
        assert int(est.qubits) == 4
        assert int(est.gates.single_qubit) == 2
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 2


class TestAutoTruncatedSlice:
    """Out-of-range stop is truncated to the parent length."""

    @qmc.qkernel
    def _kern() -> qmc.Vector[qmc.Bit]:
        """Python-style auto-truncation (``q[3:10]`` → ``q[3:4]``)."""
        q = qmc.qubit_array(4, "q")
        q[3:10] = qmc.h(q[3:10])
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self, sv_backend):
        # q[3:10] on a length-4 array is truncated to q[3:4]; only q[3]
        # gets H.
        expected_sv = _reference_statevector(4, [3])
        _assert_backend_matches(sv_backend, self._kern, {}, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block)
        assert int(est.qubits) == 4
        assert int(est.gates.single_qubit) == 1
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 1


# ``_h_all`` is the sub-kernel that ``TestViewPassedToSubKernel._kern``
# passes its slice view to.  It is *only* used by that test class, but
# it must live at module scope because the body of ``_kern`` references
# it as a bare name and Python's name resolution looks free names up
# via the function's ``__globals__`` (the enclosing module's globals),
# not via the surrounding class body.  Keeping ``_h_all`` immediately
# above the class preserves visual proximity.
@qmc.qkernel
def _h_all(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Broadcast H over every element of a ``Vector[Qubit]`` via a per-element loop."""
    n = q.shape[0]
    for i in qmc.range(n):
        q[i] = qmc.h(q[i])
    return q


class TestViewPassedToSubKernel:
    """Passing a slice view as a ``Vector[Qubit]`` argument to another kernel."""

    @qmc.qkernel
    def _kern(num: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Pass a slice view as a ``Vector[Qubit]`` argument."""
        q = qmc.qubit_array(num, "q")
        evens = q[0::2]
        evens = _h_all(evens)
        q[0::2] = evens
        return qmc.measure(q)

    def test_transpile_and_statevector_match(self, sv_backend):
        bindings = {"num": 6}
        expected_sv = _reference_statevector(6, [0, 2, 4])
        _assert_backend_matches(sv_backend, self._kern, bindings, expected_sv)

    def test_resource_estimate(self):
        est = estimate_resources(self._kern.block, bindings={"num": 6})
        assert int(est.qubits) == 6
        assert int(est.gates.single_qubit) == 3
        assert int(est.gates.two_qubit) == 0
        assert int(est.gates.total) == 3


# ``TestScratchViewLeak._oracle`` lives at module scope for the same
# reason as ``_h_all`` above: ``TestScratchViewLeak._kern`` references
# it by bare name inside the kernel body.
@qmc.qkernel
def _scratch_oracle(
    qs1: qmc.Vector[qmc.Qubit],
    qs2: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Trivial Simon-style oracle: CX from each qs1[i] into qs2[i]."""
    n = qs1.shape[0]
    for i in qmc.range(n):
        qs1[i], qs2[i] = qmc.cx(qs1[i], qs2[i])
    return qs1, qs2


class TestScratchViewLeak:
    """A slice view used as a Simon-style scratch register can be discarded.

    Slice views are affine at the kernel boundary: a view left live at
    block end (no slice-assign back to the parent, no destructive
    consume) is OK as long as no other rule is violated.  The classic
    use case is a scratch register that the oracle writes into and
    that the caller never measures — exactly the structure of Simon's
    algorithm.  This regression pins that the pattern compiles
    cleanly and emits a circuit that measures only the queried half.

    Hazards that *would* fire even in this affine regime:

    - touching a slot a live view owns → ``QubitConsumedError`` (see
      ``TestMultiViewOverlap``);
    - consuming or returning the parent while a view is live →
      ``UnreturnedBorrowError`` (see ``TestStrictReturnViolations``).
    """

    @qmc.qkernel
    def _kern(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Simon-style: oracle scratch ``qs2`` discarded after the H-fold."""
        qs = qmc.qubit_array(2 * n, name="qs")
        qs1 = qs[0:n]
        qs2 = qs[n : 2 * n]
        qs1 = qmc.h(qs1)
        qs1, qs2 = _scratch_oracle(qs1, qs2)
        qs1 = qmc.h(qs1)
        return qmc.measure(qs1)  # qs2 left live — affine view discard

    def test_transpile_compiles_cleanly(self):
        """Simon-style kernel transpiles to a circuit that measures only ``qs1``."""
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        transpiler = QiskitTranspiler()
        exe = transpiler.transpile(self._kern, bindings={"n": 2})
        qc = exe.compiled_quantum[0].circuit
        # 2n = 4 physical qubits, but only the first n=2 are measured.
        assert qc.num_qubits == 4
        assert qc.num_clbits == 2
        measured_indices = sorted(
            qc.qubits.index(inst.qubits[0])
            for inst in qc.data
            if inst.operation.name == "measure"
        )
        assert measured_indices == [0, 1], (
            f"Only qs1 (root slots 0..n-1) should be measured; got {measured_indices}"
        )


# ---------------------------------------------------------------------------
# Illegal slicing patterns — each test method inlines its own kernel
# ---------------------------------------------------------------------------


class TestStrictReturnViolations:
    """Every view-consuming op that's not destructive demands a slice-assign return.

    Skipping that return leaves the parent's bulk-borrow live and the
    next parent consume (``measure(q)`` etc.) raises
    ``UnreturnedBorrowError``.  Four entry points exercise the four
    transfer-style ops the policy calls out: element loop, broadcast
    gate, sub-kernel call, and ``pauli_evolve``.
    """

    def test_loop_pattern_without_slice_assign_raises(self):
        """Element loop + no slice-assign on the loop's view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            evens = q[0:4:2]
            for i in qmc.range(evens.shape[0]):
                evens[i] = qmc.h(evens[i])
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_broadcast_without_slice_assign_raises(self):
        """``view = qmc.h(view)`` + no slice-assign on the returned view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[0::2]
            view = qmc.h(view)
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_qkernel_call_without_slice_assign_raises(self):
        """``view = sub_kernel(view)`` + no slice-assign on the returned view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def callee(q: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            return q

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[0:2]
            view = callee(view)
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block

    def test_pauli_evolve_without_slice_assign_raises(self):
        """``view = qmc.pauli_evolve(view, ...)`` + no slice-assign on the returned view."""
        from qamomile.circuit.transpiler.errors import UnreturnedBorrowError

        @qmc.qkernel
        def kern(H: qmc.Observable, gamma: qmc.Float) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            view = q[0::2]
            view = qmc.pauli_evolve(view, H, gamma)
            return qmc.measure(q)

        with pytest.raises(UnreturnedBorrowError, match="unreturned slice-view"):
            _ = kern.block


class TestMultiViewOverlap:
    """Two live views on the same root with overlapping coverage are rejected.

    Strict no-multi-view: at most one live ``VectorView`` may own any
    given parent slot.  Concrete-bounds overlap is caught at trace
    time by ``VectorView._wrap``; the same pattern inside a
    control-flow body is rejected for the same reason (the body's view
    construction sees the outer view's claim on the parent slot).
    """

    def test_top_level_overlap_raises_at_trace(self):
        """``a = q[0:3]; b = q[0:2]`` — two live views with overlapping coverage."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            a = q[0:3]
            b = q[0:2]
            _ = a
            _ = b
            return qmc.measure(q)

        with pytest.raises(
            QubitBorrowConflictError, match="already owned by another slice view"
        ):
            _ = kern.block

    def test_body_internal_overlap_raises_at_trace(self):
        """Outer view live across the body, body creates an overlapping inner view."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]  # noqa: F841 — kept live to force the overlap conflict
            for _ in qmc.range(1):
                v = q[0:3]
                v = qmc.h(v)
                q[0:3] = v
            return qmc.measure(q)

        with pytest.raises(
            QubitBorrowConflictError, match="already owned by another slice view"
        ):
            _ = kern.block


class TestSliceAssignmentConsistency:
    """Slice assignment requires LHS and RHS to describe the same slot set.

    Coverage mismatch is the canonical case: the right-hand side
    covers a different number of parent slots than the left-hand
    slice.  Caught at trace time inside ``_return_slice_view`` even
    when the right-hand side has been broadcast-consumed (the helper
    reconstructs coverage from the IR ``ArrayValue``).
    """

    def test_coverage_mismatch_raises_at_trace(self):
        """``q[0:2] = qmc.h(q[0:3])`` — LHS and RHS coverages differ."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            q[0:2] = qmc.h(q[0:3])
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="coverage mismatch"):
            _ = kern.block


class TestControlFlowBodyViolations:
    """Releasing an outer-registered view from inside a control-flow body is rejected.

    The IR pass's ``outer_snapshot_stack`` guard catches this: the
    body cannot delete a borrow-table entry that the enclosing block
    is still relying on, because the loop / branch merge cannot
    propagate the deletion outward.  Caught post-fold.
    """

    def test_release_of_outer_view_in_body_raises(self):
        """Slice-assign that releases an outer view from inside a for body."""
        from qamomile.circuit.transpiler.errors import SliceBorrowViolationError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]
            for _ in qmc.range(1):
                q[0::2] = even
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        with pytest.raises(SliceBorrowViolationError):
            transpiler.transpile(kern)

    def test_release_of_outer_view_in_while_body_raises(self):
        """Slice-assign that releases an outer view from inside a while body."""
        from qamomile.circuit.transpiler.errors import SliceBorrowViolationError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]
            t = qmc.qubit("t")
            t = qmc.h(t)
            bit = qmc.measure(t)
            while bit:
                q[0::2] = even
                t2 = qmc.qubit("t2")
                bit = qmc.measure(t2)
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        with pytest.raises(SliceBorrowViolationError):
            transpiler.transpile(kern)

    def test_release_of_outer_view_in_if_branch_raises(self):
        """Slice-assign that releases an outer view from inside an if branch.

        The runtime-``if`` variant is rejected before the IR guard ever
        sees it: the frontend's branch-scope slice-assignment validation
        raises at trace time (a broader ``AffineTypeError``, not the IR
        pass's ``SliceBorrowViolationError``). Characterized here so the
        cross-body-release contract is pinned for all three control-flow
        kinds regardless of which layer enforces it.
        """
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            even = q[0::2]
            t = qmc.qubit("t")
            t = qmc.h(t)
            bit = qmc.measure(t)
            if bit:
                q[0::2] = even
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        with pytest.raises(AffineTypeError):
            transpiler.transpile(kern)

    def test_view_created_and_released_inside_body_allowed(self):
        """Per-iteration borrow/return inside the body is the legal rewrite.

        The view is created inside the loop body, so no enclosing
        snapshot owns its entries and the in-body release is legal —
        this is the rewrite the cross-body-release error message and
        the vector-slicing tutorial point to.
        """

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(4, "q")
            for _ in qmc.range(2):
                even = q[0::2]
                even = qmc.x(even)
                q[0::2] = even
            return qmc.measure(q)

        transpiler = QiskitTranspiler()
        assert transpiler.transpile(kern) is not None


class TestNestedSliceReturnOrder:
    """Nested slices must be returned inner→outer→root.

    The slice assignment in :meth:`VectorView._return_slice_view`
    refuses to release an inner view directly to the root parent,
    refuses to release the outer view while the inner still owns
    parent slots, and the element accessor refuses to touch an
    outer-view slot that is currently owned by a nested inner.
    """

    def test_inner_returned_to_root_directly_raises(self):
        """``a = q[1:9]; b = a[1:5]; q[2:6] = b`` skips the outer-view return."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(10, "q")
            a = q[1:9]  # noqa: F841
            b = a[1:5]
            q[2:6] = b
            return qmc.measure(q)

        with pytest.raises(
            AffineTypeError, match="returned through its immediate outer view"
        ):
            _ = kern.block

    def test_outer_returned_while_inner_live_raises(self):
        """``q[1:9] = a`` with ``b`` still owning a's overlap slots."""
        from qamomile.circuit.transpiler.errors import AffineTypeError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(10, "q")
            a = q[1:9]
            b = a[1:5]  # noqa: F841
            q[1:9] = a
            return qmc.measure(q)

        with pytest.raises(AffineTypeError, match="no longer owns"):
            _ = kern.block

    def test_outer_access_at_inner_slot_raises(self):
        """Touching ``a[overlap_idx]`` while inner ``b`` owns that slot."""
        from qamomile.circuit.transpiler.errors import QubitBorrowConflictError

        @qmc.qkernel
        def kern() -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(10, "q")
            a = q[1:9]
            b = a[1:5]  # noqa: F841
            a[1] = qmc.h(a[1])  # a[1] is root q[2], owned by b
            return qmc.measure(q)

        with pytest.raises(
            QubitBorrowConflictError, match="currently held by another slice view"
        ):
            _ = kern.block
