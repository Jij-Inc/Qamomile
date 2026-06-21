"""Cross-backend tests for the ``qmc.global_phase`` combinator.

``qmc.global_phase(kernel, theta)`` multiplies the wrapped kernel's unitary
by ``e^{i*theta}``. Standalone the phase is physically unobservable, so the
tests assert it is a no-op for sampling / expectation values while being
present in the emitted unitary (Qiskit, where the unitary is exact).
Controlling the combinator turns the global phase into an observable
relative phase on the control qubit -- the projector-controlled-phase / QSVT
building block -- which the Hadamard-test legs verify on every SDK backend.

Backends are exercised through the shared ``sdk_transpiler`` fixture
(``importorskip``-guarded per backend), so the QURI Parts and CUDA-Q legs
skip automatically when those SDKs are absent. The Qiskit-only unitary
checks use ``qiskit_transpiler`` because only Qiskit exposes the exact
global phase in ``QuantumCircuit.global_phase``.
"""

from __future__ import annotations

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit import qkernel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unitary(qc):
    """Return the unitary of a transpiled Qiskit circuit (measurements stripped).

    Args:
        qc: Qiskit ``QuantumCircuit`` to convert.

    Returns:
        numpy.ndarray: Dense unitary matrix.
    """
    from qiskit.quantum_info import Operator

    return Operator(qc.remove_final_measurements(inplace=False)).data


def _counts(result) -> dict:
    """Aggregate a SampleResult's ``(value, count)`` list into a dict.

    Args:
        result: A ``SampleResult`` from ``exe.sample(...).result()``.

    Returns:
        dict: Mapping from measured value to summed shot count.
    """
    out: dict = {}
    for value, count in result.results:
        key = int(value) if not isinstance(value, tuple) else value
        out[key] = out.get(key, 0) + int(count)
    return out


def _executor(case, seed: int = 901):
    """Build a (seeded, for Qiskit) executor for an ``SdkTranspilerCase``.

    Args:
        case (SdkTranspilerCase): Backend label plus transpiler instance.
        seed (int): Simulator seed for the Qiskit leg.

    Returns:
        Any: A backend executor suitable for ``exe.sample`` / ``exe.run``.
    """
    if case.backend_name == "qiskit":
        from qiskit.providers.basic_provider import BasicSimulator

        backend = BasicSimulator()
        backend.set_options(seed_simulator=seed)
        return case.transpiler.executor(backend=backend)
    return case.transpiler.executor()


# --------------------------------------------------------------------------- #
# Module-level body kernels (qkernel needs retrievable source)
# --------------------------------------------------------------------------- #
@qkernel
def _ident(q: qmc.Qubit) -> qmc.Qubit:
    """Identity body: return the qubit unchanged."""
    return q


@qkernel
def _x_body(q: qmc.Qubit) -> qmc.Qubit:
    """Single-X body."""
    return qmc.x(q)


@qkernel
def _hz_body(q: qmc.Qubit) -> qmc.Qubit:
    """Multi-gate body: H then Z."""
    q = qmc.h(q)
    q = qmc.z(q)
    return q


@qkernel
def _rz_body(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Body with a classical Float parameter."""
    return qmc.rz(q, angle)


@qkernel
def _inner_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Inner kernel used to exercise nested calls inside a phased body."""
    q = qmc.h(q)
    return q


@qkernel
def _nested_body(q: qmc.Qubit) -> qmc.Qubit:
    """Body that calls another kernel (nested-call coverage)."""
    q = _inner_layer(q)
    q = qmc.x(q)
    return q


@qkernel
def _vec_body(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Vector[Qubit] body: an X on every element."""
    for i in qmc.range(2):
        qs[i] = qmc.x(qs[i])
    return qs


# --------------------------------------------------------------------------- #
# Standalone: phase appears in the unitary (Qiskit) but is unobservable
# --------------------------------------------------------------------------- #
class TestGlobalPhaseStandalone:
    """Standalone ``global_phase`` folds e^{iθ} into the unitary only."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    @pytest.mark.parametrize(
        "body", [_ident, _x_body, _hz_body], ids=["ident", "x", "hz"]
    )
    def test_unitary_is_phase_times_body(self, qiskit_transpiler, body, seed):
        """``U(global_phase(body, θ)) == e^{iθ} · U(body)`` exactly (Qiskit)."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            @qkernel
            def circ() -> qmc.Bit:
                q = qmc.qubit("q")
                if use_phase:
                    q = qmc.global_phase(body, theta)(q)
                else:
                    q = body(q)
                return qmc.measure(q)

            return circ

        u_phase = _unitary(
            qiskit_transpiler.transpile(make(True), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        u_plain = _unitary(
            qiskit_transpiler.transpile(make(False), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, atol=1e-9)

    @pytest.mark.parametrize("theta", [0.0, np.pi, 2 * np.pi, -1.3])
    def test_pure_phase_is_scalar_identity(self, qiskit_transpiler, theta):
        """``global_phase(identity, θ)`` standalone is exactly ``e^{iθ} I``."""

        @qkernel
        def circ() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.global_phase(_ident, theta)(q)
            return qmc.measure(q)

        u = _unitary(
            qiskit_transpiler.transpile(circ, bindings={}).compiled_quantum[0].circuit
        )
        assert np.allclose(u, np.exp(1j * theta) * np.eye(2), atol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_sample_unaffected_by_phase(self, sdk_transpiler, seed):
        """Standalone phase leaves the sampled distribution unchanged."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def circ() -> qmc.Bit:
            q = qmc.qubit("q")
            q = qmc.global_phase(_x_body, theta)(q)
            return qmc.measure(q)

        exe = sdk_transpiler.transpiler.transpile(circ, bindings={})
        counts = _counts(
            exe.sample(_executor(sdk_transpiler, seed), shots=512).result()
        )
        # X|0> = |1>; the global phase must not change the outcome.
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_expval_unaffected_by_phase(self, sdk_transpiler, seed):
        """Standalone phase leaves ``<Z>`` unchanged on every backend."""
        import qamomile.observable as qm_o

        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def phased(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.global_phase(_hz_body, theta)(q)
            return qmc.expval(q, obs)

        @qkernel
        def plain(obs: qmc.Observable) -> qmc.Float:
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = _hz_body(q)
            return qmc.expval(q, obs)

        obs = qm_o.Z(0)
        tr = sdk_transpiler.transpiler
        v_phase = (
            tr.transpile(phased, bindings={"obs": obs}).run(tr.executor()).result()
        )
        v_plain = tr.transpile(plain, bindings={"obs": obs}).run(tr.executor()).result()
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(v_phase, v_plain, atol=atol), (
            f"{sdk_transpiler.backend_name}: {v_phase} vs {v_plain}"
        )


# --------------------------------------------------------------------------- #
# Controlled: the global phase becomes an observable relative phase
# --------------------------------------------------------------------------- #
class TestGlobalPhaseControlled:
    """``control(global_phase(...))`` realizes a projector-controlled phase."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_hadamard_test_phase_kickback(self, sdk_transpiler, seed):
        """A Hadamard test reads back ``cos^2(θ/2)`` on every backend.

        Controlling a pure global phase puts ``P(θ)`` on the control qubit,
        so the standard Hadamard test (H, controlled-phase, H, measure)
        yields ``P(outcome=0) = cos^2(θ/2)`` -- the observable proof that the
        otherwise-invisible global phase became a relative phase.
        """
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(0.2, np.pi - 0.2))

        @qkernel
        def phased_ident(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def htest(angle: qmc.Float) -> qmc.Bit:
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(phased_ident)(ctrl, q, angle)
            ctrl = qmc.h(ctrl)
            return qmc.measure(ctrl)

        exe = sdk_transpiler.transpiler.transpile(htest, parameters=["angle"])
        shots = 20000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"angle": theta}
            ).result()
        )
        p0 = counts.get(0, 0) / shots
        assert np.isclose(p0, np.cos(theta / 2) ** 2, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(0)={p0}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    @pytest.mark.parametrize(
        "body", [_ident, _x_body, _hz_body], ids=["ident", "x", "hz"]
    )
    def test_controlled_unitary_relative_phase(self, qiskit_transpiler, body, seed):
        """``control(global_phase(body, θ))`` adds e^{iθ} on the ctrl=1 block.

        The ``θ=0`` build is the structurally-identical no-phase reference
        (same emitted controlled body, same qubit layout), so
        ``U(θ) · U(0)^†`` isolates exactly the controlled global phase: it
        must be diagonal with half its eigenvalues ``1`` (ctrl=0) and half
        ``e^{iθ}`` (ctrl=1). A separate assertion checks the body is genuinely
        applied so the comparison is not vacuous.
        """
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def phased_body(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            return qmc.global_phase(body, angle)(q)

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Bit:
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl, q = qmc.control(phased_body)(ctrl, q, angle)
            return qmc.measure(ctrl)

        u_theta = _unitary(
            qiskit_transpiler.transpile(circ, bindings={"angle": theta})
            .compiled_quantum[0]
            .circuit
        )
        u_zero = _unitary(
            qiskit_transpiler.transpile(circ, bindings={"angle": 0.0})
            .compiled_quantum[0]
            .circuit
        )
        # The controlled body must actually be present for non-trivial bodies.
        if body is not _ident:
            assert not np.allclose(u_zero, np.eye(u_zero.shape[0]), atol=1e-8)

        m = u_theta @ u_zero.conj().T
        assert np.allclose(m, np.diag(np.diagonal(m)), atol=1e-8)
        diag = np.diagonal(m)
        ones = int(np.isclose(diag, 1.0, atol=1e-8).sum())
        phases = int(np.isclose(diag, np.exp(1j * theta), atol=1e-8).sum())
        assert ones == phases == len(diag) // 2, (
            f"θ={theta}: diag spectrum {np.round(diag, 4)}"
        )


# --------------------------------------------------------------------------- #
# Handle-type coverage: nested calls, Vector, slice/VectorView, classical
# --------------------------------------------------------------------------- #
class TestGlobalPhaseHandleTypes:
    """Phased bodies over nested calls, vectors, slices, and classical args."""

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_nested_call_body(self, qiskit_transpiler, seed):
        """A phased body that itself calls another kernel keeps e^{iθ}·U."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            @qkernel
            def circ() -> qmc.Bit:
                q = qmc.qubit("q")
                q = (
                    qmc.global_phase(_nested_body, theta)(q)
                    if use_phase
                    else _nested_body(q)
                )
                return qmc.measure(q)

            return circ

        u_phase = _unitary(
            qiskit_transpiler.transpile(make(True), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        u_plain = _unitary(
            qiskit_transpiler.transpile(make(False), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, atol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_classical_param_body(self, qiskit_transpiler, seed):
        """A phased body carrying a classical Float arg keeps e^{iθ}·U(angle)."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))
        inner = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            @qkernel
            def circ(angle: qmc.Float) -> qmc.Bit:
                q = qmc.qubit("q")
                q = qmc.h(q)
                q = (
                    qmc.global_phase(_rz_body, theta)(q, angle)
                    if use_phase
                    else _rz_body(q, angle)
                )
                return qmc.measure(q)

            return circ

        u_phase = _unitary(
            qiskit_transpiler.transpile(make(True), bindings={"angle": inner})
            .compiled_quantum[0]
            .circuit
        )
        u_plain = _unitary(
            qiskit_transpiler.transpile(make(False), bindings={"angle": inner})
            .compiled_quantum[0]
            .circuit
        )
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, atol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_vector_body(self, qiskit_transpiler, seed):
        """A phased ``Vector[Qubit]`` body keeps e^{iθ}·U over the register."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            @qkernel
            def circ() -> qmc.Vector[qmc.Bit]:
                qs = qmc.qubit_array(2, "qs")
                qs = (
                    qmc.global_phase(_vec_body, theta)(qs)
                    if use_phase
                    else _vec_body(qs)
                )
                return qmc.measure(qs)

            return circ

        u_phase = _unitary(
            qiskit_transpiler.transpile(make(True), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        u_plain = _unitary(
            qiskit_transpiler.transpile(make(False), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, atol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_slice_view_body(self, qiskit_transpiler, seed):
        """A phased body applied to a ``VectorView`` slice keeps e^{iθ}·U."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            @qkernel
            def circ() -> qmc.Vector[qmc.Bit]:
                qs = qmc.qubit_array(4, "qs")
                view = qs[0:2]
                view = (
                    qmc.global_phase(_vec_body, theta)(view)
                    if use_phase
                    else _vec_body(view)
                )
                qs[0:2] = view
                return qmc.measure(qs)

            return circ

        u_phase = _unitary(
            qiskit_transpiler.transpile(make(True), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        u_plain = _unitary(
            qiskit_transpiler.transpile(make(False), bindings={})
            .compiled_quantum[0]
            .circuit
        )
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, atol=1e-9)


# --------------------------------------------------------------------------- #
# IR plumbing: serialization round-trip and content hashing
# --------------------------------------------------------------------------- #
class TestGlobalPhaseSerialize:
    """The op survives canonicalization, serialization, and content hashing."""

    def test_serialize_roundtrip_and_content_hash(self, qiskit_transpiler):
        """JSON round-trip preserves structure and a stable content hash."""
        from qamomile.circuit.ir.canonical import content_hash
        from qamomile.circuit.ir.serialize import dump_json, load_json

        @qkernel
        def phased_body(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            return qmc.global_phase(_x_body, angle)(q)

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Bit:
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl, q = qmc.control(phased_body)(ctrl, q, angle)
            return qmc.measure(ctrl)

        block = qiskit_transpiler.inline(
            qiskit_transpiler.to_block(circ, {}, ["angle"])
        )
        restored = load_json(dump_json(block))
        assert [type(o).__name__ for o in block.operations] == [
            type(o).__name__ for o in restored.operations
        ]
        assert content_hash(block) == content_hash(restored)
