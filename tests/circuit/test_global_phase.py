"""Cross-backend tests for the ``qmc.global_phase`` combinator.

``qmc.global_phase(qkernel, theta)`` applies an ordinary qkernel call and then
records ``e^{i*theta}``, without imposing a separate reversibility contract.
Standalone phase is physically unobservable, so sampling and expectation
values do not change, while every target must still preserve it exactly or
reject it explicitly. Coherently controlling a reversible qkernel containing
the phase turns it into an observable relative phase on the control subspace.

Backends are exercised through the shared ``sdk_transpiler`` fixture
(``importorskip``-guarded per backend), so the QURI Parts and CUDA-Q legs
skip automatically when those SDKs are absent. The Qiskit-only unitary
checks use ``qiskit_transpiler`` because only Qiskit exposes the exact
global phase in ``QuantumCircuit.global_phase``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.circuit import qkernel


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _unitary(qc: Any) -> np.ndarray:
    """Return the unitary of a transpiled Qiskit circuit (measurements stripped).

    Args:
        qc (Any): Qiskit ``QuantumCircuit`` to convert.

    Returns:
        numpy.ndarray: Dense unitary matrix.
    """
    from qiskit.quantum_info import Operator

    return Operator(qc.remove_final_measurements(inplace=False)).data


def _assert_runtime_angle_unitary_preserved(
    original: Any,
    restored: Any,
    qiskit_transpiler: Any,
) -> None:
    """Assert runtime-angle semantics survive a serialization round-trip.

    Args:
        original (Any): Source qkernel.
        restored (Any): Deserialized qkernel.
        qiskit_transpiler (Any): Qiskit transpiler used to emit both qkernels.

    Raises:
        AssertionError: If the bound unitaries differ.
        ValueError: If either circuit does not expose exactly one parameter.
    """
    original_circuit = (
        qiskit_transpiler.transpile(
            original,
            parameters=["angle"],
        )
        .compiled_quantum[0]
        .circuit
    )
    restored_circuit = (
        qiskit_transpiler.transpile(
            restored,
            parameters=["angle"],
        )
        .compiled_quantum[0]
        .circuit
    )
    (original_parameter,) = original_circuit.parameters
    (restored_parameter,) = restored_circuit.parameters

    np.testing.assert_allclose(
        _unitary(restored_circuit.assign_parameters({restored_parameter: 0.731})),
        _unitary(original_circuit.assign_parameters({original_parameter: 0.731})),
        rtol=0.0,
        atol=1e-12,
    )


def _counts(result: Any) -> dict[Any, int]:
    """Aggregate a SampleResult's ``(value, count)`` list into a dict.

    Args:
        result (Any): A ``SampleResult`` from ``exe.sample(...).result()``.

    Returns:
        dict[Any, int]: Mapping from measured value to summed shot count.
    """
    out: dict = {}
    for value, count in result.results:
        key = int(value) if not isinstance(value, tuple) else value
        out[key] = out.get(key, 0) + int(count)
    return out


def _assert_invoke_followed_by_phase(block: Any) -> None:
    """Assert that an ordinary qkernel invocation retains its following phase."""
    from qamomile.circuit.ir.operation import GlobalPhaseOperation, InvokeOperation

    assert any(
        isinstance(current, InvokeOperation)
        and isinstance(following, GlobalPhaseOperation)
        for current, following in zip(block.operations, block.operations[1:])
    )


def _executor(case: Any, seed: int = 901) -> Any:
    """Build a (seeded, for Qiskit) executor for an ``SdkTranspilerCase``.

    Args:
        case (Any): Backend label plus transpiler instance.
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
    """Return the identity body output.

    Args:
        q (qmc.Qubit): Qubit to preserve.

    Returns:
        qmc.Qubit: Unchanged qubit.
    """
    return q


@qkernel
def _x_body(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a single X gate.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    return qmc.x(q)


@qkernel
def _hz_body(q: qmc.Qubit) -> qmc.Qubit:
    """Apply H followed by Z.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    q = qmc.h(q)
    q = qmc.z(q)
    return q


@qkernel
def _rz_body(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
    """Apply an RZ gate with a classical parameter.

    Args:
        q (qmc.Qubit): Target qubit.
        angle (qmc.Float): Rotation angle in radians.

    Returns:
        qmc.Qubit: Rotated target qubit.
    """
    return qmc.rz(q, angle)


@qkernel
def _inner_layer(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the inner nested-call layer.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    q = qmc.h(q)
    return q


@qkernel
def _nested_body(q: qmc.Qubit) -> qmc.Qubit:
    """Apply a nested helper call followed by X.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    q = _inner_layer(q)
    q = qmc.x(q)
    return q


def _controlled_phased_x(theta: float) -> Any:
    """Create a controlled composite whose target body is phased X.

    Args:
        theta (float): Constant global phase in radians.

    Returns:
        Any: Controlled custom composite callable.
    """

    @qmc.composite_gate(name="controlled_phased_x")
    def phased_x(target: qmc.Qubit) -> qmc.Qubit:
        """Apply X with a captured global phase.

        Args:
            target (qmc.Qubit): Target qubit.

        Returns:
            qmc.Qubit: Updated target qubit.
        """
        return qmc.global_phase(_x_body, theta)(target)

    return qmc.control(phased_x)


@qkernel
def _vec_body(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Apply X to every element of a two-qubit vector.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Two-qubit target vector.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target vector.
    """
    for i in qmc.range(2):
        qs[i] = qmc.x(qs[i])
    return qs


@qkernel
def _for_items_phase_body(
    q: qmc.Qubit,
    phases: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Qubit:
    """Apply one global phase for every bound dictionary item.

    Args:
        q (qmc.Qubit): Qubit passed through the phase-only loop.
        phases (qmc.Dict[qmc.UInt, qmc.Float]): Phase angles keyed by an
            arbitrary unsigned integer.

    Returns:
        qmc.Qubit: Unchanged qubit with the accumulated global phase.
    """
    for _index, phase in qmc.items(phases):
        q = qmc.global_phase(_ident, phase)(q)
    return q


@qkernel
def _loop_carried_phase_body(
    q: qmc.Qubit,
    step: qmc.Float,
) -> qmc.Qubit:
    """Accumulate a phase through a loop-carried scalar.

    Args:
        q (qmc.Qubit): Qubit passed through the phase-only loop.
        step (qmc.Float): Per-iteration increment of the carried phase.

    Returns:
        qmc.Qubit: Unchanged qubit with phase ``step + 2*step + 3*step``.
    """
    phase = 0.0
    for _index in qmc.range(3):
        phase = phase + step
        q = qmc.global_phase(_ident, phase)(q)
    return q


@qkernel
def _loop_carried_pauli_phase_body(
    q: qmc.Vector[qmc.Qubit],
    hamiltonian: qmc.Observable,
    step: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    """Accumulate identity-Pauli phase through a loop-carried scalar.

    Args:
        q (qmc.Vector[qmc.Qubit]): One-qubit target register.
        hamiltonian (qmc.Observable): Identity Hamiltonian used for phase.
        step (qmc.Float): Per-iteration evolution-time increment.

    Returns:
        qmc.Vector[qmc.Qubit]: Register with total phase ``-6*step``.
    """
    evolution_time = 0.0
    for _index in qmc.range(3):
        evolution_time = evolution_time + step
        q = qmc.pauli_evolve(q, hamiltonian, evolution_time)
    return q


@qkernel
def _phase_call_from_loop_index(q: qmc.Qubit, index: qmc.UInt) -> qmc.Qubit:
    """Apply a phase whose angle is supplied by a caller loop index.

    Args:
        q (qmc.Qubit): Target qubit.
        index (qmc.UInt): Caller-owned loop index.

    Returns:
        qmc.Qubit: Unchanged qubit with phase ``0.2 * index``.
    """
    return qmc.global_phase(_ident, 0.2 * index)(q)


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
            """Build the phased or reference kernel used by this test."""

            @qkernel
            def circ() -> qmc.Bit:
                """Build the local circuit exercised by this test."""
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
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("theta", [0.0, np.pi, 2 * np.pi, -1.3])
    def test_pure_phase_is_scalar_identity(self, qiskit_transpiler, theta):
        """``global_phase(identity, θ)`` standalone is exactly ``e^{iθ} I``."""

        @qkernel
        def circ() -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            q = qmc.qubit("q")
            q = qmc.global_phase(_ident, theta)(q)
            return qmc.measure(q)

        u = _unitary(
            qiskit_transpiler.transpile(circ, bindings={}).compiled_quantum[0].circuit
        )
        assert np.allclose(u, np.exp(1j * theta) * np.eye(2), rtol=0.0, atol=1e-9)

    def test_direct_phase_call_from_loop_index_is_unrolled(self, qiskit_transpiler):
        """A boxed phase call cannot retain a caller-owned loop expression."""

        @qkernel
        def circ() -> qmc.Bit:
            """Apply indexed direct calls whose phases sum to 0.6 radians."""
            q = qmc.qubit("q")
            for index in qmc.range(3):
                q = _phase_call_from_loop_index(q, index)
            return qmc.measure(q)

        circuit = qiskit_transpiler.transpile(circ).compiled_quantum[0].circuit
        assert np.allclose(
            _unitary(circuit),
            np.exp(0.6j) * np.eye(2),
            rtol=0.0,
            atol=1e-9,
        )

    def test_measuring_qkernel_retains_standalone_phase(self, qiskit_transpiler):
        """Measurement does not prevent the following phase from materializing."""

        @qkernel
        def measure_body(q: qmc.Qubit) -> qmc.Bit:
            """Measure the supplied qubit."""
            return qmc.measure(q)

        @qkernel
        def circuit() -> qmc.Bit:
            """Apply a phase after the measuring qkernel call."""
            q = qmc.qubit("q")
            return qmc.global_phase(measure_body, 0.4)(q)

        artifact = qiskit_transpiler.transpile(circuit).get_first_circuit()

        assert float(artifact.global_phase) == pytest.approx(0.4)
        assert [instruction.operation.name for instruction in artifact.data] == [
            "measure"
        ]

    def test_runtime_if_retains_each_branch_phase(self, qiskit_transpiler):
        """Phases remain on the measured branches where users placed them."""
        from qiskit.circuit import IfElseOp

        @qkernel
        def circuit() -> qmc.Bit:
            """Apply a distinct phase in each measured branch."""
            selector = qmc.measure(qmc.qubit("selector"))
            target = qmc.qubit("target")
            if selector:
                target = qmc.global_phase(_ident, 0.25)(target)
            else:
                target = qmc.global_phase(_ident, 0.75)(target)
            return qmc.measure(target)

        artifact = qiskit_transpiler.transpile(circuit).get_first_circuit()
        [branch] = [
            instruction.operation
            for instruction in artifact.data
            if isinstance(instruction.operation, IfElseOp)
        ]
        true_block, false_block = branch.blocks

        assert float(true_block.global_phase) == pytest.approx(0.25)
        assert float(false_block.global_phase) == pytest.approx(0.75)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_sample_unaffected_by_phase(self, sdk_transpiler, seed):
        """Standalone phase leaves the sampled distribution unchanged."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def circ() -> qmc.Bit:
            """Build the local circuit exercised by this test."""
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
            """Build the globally phased circuit path."""
            q = qmc.qubit("q")
            q = qmc.h(q)
            q = qmc.global_phase(_hz_body, theta)(q)
            return qmc.expval(q, obs)

        @qkernel
        def plain(obs: qmc.Observable) -> qmc.Float:
            """Build the unphased reference circuit."""
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
        assert np.isclose(v_phase, v_plain, rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name}: {v_phase} vs {v_plain}"
        )

    @pytest.mark.parametrize("seed", [0, 42])
    def test_for_items_phases_execute_sample_and_expval(self, sdk_transpiler, seed):
        """Bound ForItems phases remain harmless on both execution paths."""
        import qamomile.observable as qm_o

        rng = np.random.default_rng(seed)
        phases = {
            0: float(rng.uniform(-np.pi, np.pi)),
            3: float(rng.uniform(-np.pi, np.pi)),
        }

        @qkernel
        def sample_circuit(
            values: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            """Sample a phased dictionary loop from ``|1>``."""
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = _for_items_phase_body(q, values)
            return qmc.measure(q)

        @qkernel
        def expval_circuit(
            obs: qmc.Observable,
            values: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Float:
            """Estimate an observable after a phased dictionary loop."""
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = _for_items_phase_body(q, values)
            return qmc.expval(q, obs)

        transpiler = sdk_transpiler.transpiler
        sample = transpiler.transpile(sample_circuit, bindings={"values": phases})
        counts = _counts(
            sample.sample(
                _executor(sdk_transpiler, seed),
                shots=256,
            ).result()
        )
        expectation = (
            transpiler.transpile(
                expval_circuit,
                bindings={"obs": qm_o.Z(0), "values": phases},
            )
            .run(transpiler.executor())
            .result()
        )

        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: {counts}"
        assert np.isclose(expectation, -1.0, rtol=0.0, atol=1e-8)


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
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def htest(angle: qmc.Float) -> qmc.Bit:
            """Build the Hadamard-test circuit for phase kickback."""
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
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(0)={p0}"
        )

    @pytest.mark.parametrize("power", [1, 3])
    @pytest.mark.parametrize("seed", [0, 1])
    def test_control_call_global_phase_keyword_kickback(
        self,
        sdk_transpiler,
        power,
        seed,
    ):
        """The control-call modifier means ``C((exp(iθ) U) ** power)``."""
        theta = float(np.random.default_rng(seed).uniform(0.3, 0.8))

        @qkernel
        def htest(angle: qmc.Float) -> qmc.Bit:
            """Observe a call-site target phase through one coherent control."""
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control = qmc.h(control)
            control, target = qmc.control(_ident)(
                control,
                target,
                power=power,
                global_phase=angle,
            )
            control = qmc.h(control)
            return qmc.measure(control)

        executable = sdk_transpiler.transpiler.transpile(
            htest,
            parameters=["angle"],
        )
        shots = 12000
        counts = _counts(
            executable.sample(
                _executor(sdk_transpiler, seed),
                shots=shots,
                bindings={"angle": theta},
            ).result()
        )
        probability_zero = counts.get(0, 0) / shots
        expected = np.cos(power * theta / 2) ** 2
        assert np.isclose(probability_zero, expected, rtol=0.0, atol=0.035), (
            f"{sdk_transpiler.backend_name} power={power}: "
            f"P(0)={probability_zero} vs {expected}"
        )

    @pytest.mark.parametrize("seed", [0, 1])
    def test_control_call_global_phase_inverse_cancels(
        self,
        sdk_transpiler,
        seed,
    ):
        """Inverse negates a call-site phase together with the target unitary."""
        theta = float(np.random.default_rng(seed).uniform(0.3, 1.2))

        @qkernel
        def phased_control(
            control: qmc.Qubit,
            target: qmc.Qubit,
            angle: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply identity with a phase on the active control branch."""
            return qmc.control(_ident)(
                control,
                target,
                global_phase=angle,
            )

        @qkernel
        def circuit(angle: qmc.Float) -> qmc.Bit:
            """Apply the phased call and its inverse before interference."""
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control = qmc.h(control)
            control, target = phased_control(control, target, angle)
            control, target = qmc.inverse(phased_control)(control, target, angle)
            control = qmc.h(control)
            return qmc.measure(control)

        executable = sdk_transpiler.transpiler.transpile(
            circuit,
            parameters=["angle"],
        )
        counts = _counts(
            executable.sample(
                _executor(sdk_transpiler, seed),
                shots=512,
                bindings={"angle": theta},
            ).result()
        )
        assert set(counts) == {0}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1])
    def test_nested_control_call_global_phase_kickback(
        self,
        sdk_transpiler,
        seed,
    ):
        """A call-site phase survives an additional coherent control layer."""
        theta = float(np.random.default_rng(seed).uniform(0.3, 1.2))

        @qkernel
        def inner_control(
            control: qmc.Qubit,
            target: qmc.Qubit,
            angle: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply a relative phase when the inner control is active."""
            return qmc.control(_ident)(
                control,
                target,
                global_phase=angle,
            )

        @qkernel
        def circuit(angle: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
            """Interfere two controls around the nested phased call."""
            outer = qmc.qubit("outer")
            inner = qmc.qubit("inner")
            target = qmc.qubit("target")
            outer = qmc.h(outer)
            inner = qmc.h(inner)
            outer, inner, target = qmc.control(inner_control)(
                outer,
                inner,
                target,
                angle,
            )
            outer = qmc.h(outer)
            inner = qmc.h(inner)
            return qmc.measure(outer), qmc.measure(inner)

        executable = sdk_transpiler.transpiler.transpile(
            circuit,
            parameters=["angle"],
        )
        shots = 16000
        counts = _counts(
            executable.sample(
                _executor(sdk_transpiler, seed),
                shots=shots,
                bindings={"angle": theta},
            ).result()
        )
        probability_zero_zero = counts.get((0, 0), 0) / shots
        expected = (5 + 3 * np.cos(theta)) / 8
        assert np.isclose(probability_zero_zero, expected, rtol=0.0, atol=0.035), (
            f"{sdk_transpiler.backend_name}: "
            f"P(00)={probability_zero_zero} vs {expected}"
        )

    def test_control_call_global_phase_binds_after_target_parameters(
        self,
        qiskit_transpiler,
    ):
        """The private phase formal follows existing target formals exactly."""
        angle = 0.41
        phase = -0.73

        @qkernel
        def circuit(theta: qmc.Float, phi: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
            """Apply a phase-augmented controlled RX with two runtime scalars."""
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control, target = qmc.control(qmc.rx)(
                control,
                target,
                angle=theta,
                global_phase=phi,
            )
            return qmc.measure(control), qmc.measure(target)

        transpiled = qiskit_transpiler.transpile(
            circuit,
            bindings={"theta": angle, "phi": phase},
        )
        actual = _unitary(transpiled.compiled_quantum[0].circuit)

        cosine = np.cos(angle / 2)
        sine = np.sin(angle / 2)
        target_unitary = np.array(
            [[cosine, -1j * sine], [-1j * sine, cosine]],
            dtype=complex,
        )
        expected = np.eye(4, dtype=complex)
        expected[np.ix_([1, 3], [1, 3])] = np.exp(1j * phase) * target_unitary
        assert np.allclose(actual, expected, rtol=0.0, atol=1e-10)

    def test_control_call_global_phase_survives_runtime_if(
        self,
        qiskit_transpiler,
    ):
        """A measured branch keeps the controlled call's coherent phase."""
        pytest.importorskip("qiskit_aer")

        @qkernel
        def circuit() -> qmc.Bit:
            """Take one measured branch containing a controlled pi phase."""
            predicate = qmc.x(qmc.qubit("predicate"))
            take_branch = qmc.measure(predicate)
            control = qmc.h(qmc.qubit("control"))
            target = qmc.qubit("target")
            if take_branch:
                control, target = qmc.control(_ident)(
                    control,
                    target,
                    global_phase=np.pi,
                )
            control = qmc.h(control)
            return qmc.measure(control)

        executable = qiskit_transpiler.transpile(circuit)
        counts = _counts(
            executable.sample(
                qiskit_transpiler.executor(),
                shots=64,
            ).result()
        )
        assert set(counts) == {1}

    def test_control_call_global_phase_survives_runtime_while(
        self,
        qiskit_transpiler,
    ):
        """One measured loop iteration keeps the controlled relative phase."""
        pytest.importorskip("qiskit_aer")

        @qkernel
        def circuit() -> qmc.Bit:
            """Run exactly one while iteration containing a controlled pi phase."""
            predicate = qmc.x(qmc.qubit("predicate"))
            run = qmc.measure(predicate)
            control = qmc.h(qmc.qubit("control"))
            target = qmc.qubit("target")
            while run:
                control, target = qmc.control(_ident)(
                    control,
                    target,
                    global_phase=np.pi,
                )
                run = qmc.measure(qmc.qubit("stopper"))
            control = qmc.h(control)
            return qmc.measure(control)

        executable = qiskit_transpiler.transpile(circuit)
        counts = _counts(
            executable.sample(
                qiskit_transpiler.executor(),
                shots=64,
            ).result()
        )
        assert set(counts) == {1}

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
            """Apply the test body with a global phase."""
            return qmc.global_phase(body, angle)(q)

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Bit:
            """Build the local circuit exercised by this test."""
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
            assert not np.allclose(u_zero, np.eye(u_zero.shape[0]), rtol=0.0, atol=1e-8)

        m = u_theta @ u_zero.conj().T
        assert np.allclose(m, np.diag(np.diagonal(m)), rtol=0.0, atol=1e-8)
        diag = np.diagonal(m)
        ones = int(np.isclose(diag, 1.0, rtol=0.0, atol=1e-8).sum())
        phases = int(np.isclose(diag, np.exp(1j * theta), rtol=0.0, atol=1e-8).sum())
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
            """Build the phased or reference kernel used by this test."""

            @qkernel
            def circ() -> qmc.Bit:
                """Build the local circuit exercised by this test."""
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
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_classical_param_body(self, qiskit_transpiler, seed):
        """A phased body carrying a classical Float arg keeps e^{iθ}·U(angle)."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))
        inner = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            """Build the phased or reference kernel used by this test."""

            @qkernel
            def circ(angle: qmc.Float) -> qmc.Bit:
                """Build the local circuit exercised by this test."""
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
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_vector_body(self, qiskit_transpiler, seed):
        """A phased ``Vector[Qubit]`` body keeps e^{iθ}·U over the register."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            """Build the phased or reference kernel used by this test."""

            @qkernel
            def circ() -> qmc.Vector[qmc.Bit]:
                """Build the local circuit exercised by this test."""
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
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, rtol=0.0, atol=1e-9)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_slice_view_body(self, qiskit_transpiler, seed):
        """A phased body applied to a ``VectorView`` slice keeps e^{iθ}·U."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        def make(use_phase: bool):
            """Build the phased or reference kernel used by this test."""

            @qkernel
            def circ() -> qmc.Vector[qmc.Bit]:
                """Build the local circuit exercised by this test."""
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
        assert np.allclose(u_phase, np.exp(1j * theta) * u_plain, rtol=0.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# IR plumbing: qkernel serialization round-trip
# --------------------------------------------------------------------------- #
class TestGlobalPhaseSerialize:
    """The op survives a qkernel protobuf round-trip.

    ``param_slots`` are rebuilt from the qkernel interface rather than carried
    on the wire, so a reloaded body is compared through its encoded static IR
    instead of ``content_hash``. Canonical content hashing of the op itself is
    covered by ``tests/transpiler/test_global_phase_pipeline.py``.
    """

    def test_serialize_roundtrip_preserves_static_body(self, qiskit_transpiler):
        """A phase inside a controlled body survives the protobuf round-trip."""
        from qamomile.circuit.serialization import deserialize, serialize
        from qamomile.circuit.serialization.encode import to_dict as kernel_to_dict

        @qkernel
        def phased_body(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply the test body with a global phase."""
            return qmc.global_phase(_x_body, angle)(q)

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl, q = qmc.control(phased_body)(ctrl, q, angle)
            return qmc.measure(ctrl)

        restored = deserialize(serialize(circ))
        block = qiskit_transpiler.inline(circ.block)
        restored_block = qiskit_transpiler.inline(restored.block)

        assert [type(o).__name__ for o in block.operations] == [
            type(o).__name__ for o in restored_block.operations
        ]
        assert (
            kernel_to_dict(restored)["artifact"]["body"]
            == kernel_to_dict(circ)["artifact"]["body"]
        )
        _assert_runtime_angle_unitary_preserved(circ, restored, qiskit_transpiler)

    def test_control_call_phase_roundtrip_preserves_static_body(
        self, qiskit_transpiler
    ):
        """A call-site phase formal and actual survive semantic IR round-trip."""
        from qamomile.circuit.serialization import deserialize, serialize
        from qamomile.circuit.serialization.encode import to_dict as kernel_to_dict

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Bit:
            """Build a directly phase-augmented controlled identity."""
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control, target = qmc.control(_ident)(
                control,
                target,
                global_phase=angle,
            )
            return qmc.measure(control)

        restored = deserialize(serialize(circ))
        block = qiskit_transpiler.inline(circ.block)
        restored_block = qiskit_transpiler.inline(restored.block)

        assert [type(operation).__name__ for operation in block.operations] == [
            type(operation).__name__ for operation in restored_block.operations
        ]
        assert (
            kernel_to_dict(restored)["artifact"]["body"]
            == kernel_to_dict(circ)["artifact"]["body"]
        )
        _assert_runtime_angle_unitary_preserved(circ, restored, qiskit_transpiler)

    def test_roundtrip_preserves_zero_result_phase_operand(self, qiskit_transpiler):
        """A constant phase keeps its angle and zero-result layout on reload."""
        from qamomile.circuit.ir.operation import GlobalPhaseOperation
        from qamomile.circuit.serialization import deserialize, serialize

        @qkernel
        def circ() -> qmc.Bit:
            """Apply a constant global phase to an otherwise plain body."""
            q = qmc.qubit("q")
            q = qmc.global_phase(_x_body, 0.375)(q)
            return qmc.measure(q)

        restored = qiskit_transpiler.inline(deserialize(serialize(circ)).block)

        phase_ops = [
            op for op in restored.operations if isinstance(op, GlobalPhaseOperation)
        ]
        assert len(phase_ops) == 1
        assert phase_ops[0].results == []
        assert phase_ops[0].phase.get_const() == pytest.approx(
            0.375,
            rel=0.0,
            abs=0.0,
        )

    def test_serialize_rejects_array_phase_operand(self):
        """The serialization boundary rejects an array-valued phase angle."""
        from qamomile.circuit.ir.operation import GlobalPhaseOperation
        from qamomile.circuit.ir.types.primitives import FloatType
        from qamomile.circuit.ir.value import ArrayValue
        from qamomile.circuit.serialization import deserialize, serialize

        restored = deserialize(serialize(_phase_call_from_loop_index))
        phase = next(
            operation
            for operation in restored.block.operations
            if isinstance(operation, GlobalPhaseOperation)
        )
        phase.operands[0] = ArrayValue(
            type=FloatType(),
            name="phase_array",
        )

        with pytest.raises(
            ValueError,
            match=r"GlobalPhaseOperation.*phase operand must be a scalar Value",
        ):
            serialize(restored)


# --------------------------------------------------------------------------- #
# Argument-shape body kernels (deterministic Z-basis outcomes)
# --------------------------------------------------------------------------- #
@qkernel
def _two_qubit_body(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Map two scalar qubits from ``|00>`` to ``|11>``.

    Args:
        a (qmc.Qubit): Control qubit.
        b (qmc.Qubit): Target qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Updated control and target qubits.
    """
    a = qmc.x(a)
    a, b = qmc.cx(a, b)
    return a, b


@qkernel
def _two_vec_body(
    xs: qmc.Vector[qmc.Qubit], ys: qmc.Vector[qmc.Qubit]
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Apply X to every element of two two-qubit vectors.

    Args:
        xs (qmc.Vector[qmc.Qubit]): First two-qubit vector.
        ys (qmc.Vector[qmc.Qubit]): Second two-qubit vector.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Updated vectors.
    """
    for i in qmc.range(2):
        xs[i] = qmc.x(xs[i])
    for i in qmc.range(2):
        ys[i] = qmc.x(ys[i])
    return xs, ys


@qkernel
def _interleaved_body(
    a: qmc.Qubit, ang: qmc.Float, b: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Exercise interleaved arguments while mapping ``|00>`` to ``|11>``.

    Args:
        a (qmc.Qubit): First target qubit.
        ang (qmc.Float): Z-rotation angle in radians.
        b (qmc.Qubit): Second target qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Updated target qubits.
    """
    a = qmc.x(a)
    a = qmc.rz(a, ang)
    b = qmc.x(b)
    return a, b


@qkernel
def _vecfloat_body(
    qs: qmc.Vector[qmc.Qubit], angs: qmc.Vector[qmc.Float]
) -> qmc.Vector[qmc.Qubit]:
    """Apply X and per-element RZ gates to a qubit vector.

    Args:
        qs (qmc.Vector[qmc.Qubit]): Two-qubit target vector.
        angs (qmc.Vector[qmc.Float]): Per-qubit rotation angles in radians.

    Returns:
        qmc.Vector[qmc.Qubit]: Updated target vector.
    """
    for i in qmc.range(2):
        qs[i] = qmc.x(qs[i])
        qs[i] = qmc.rz(qs[i], angs[i])
    return qs


class TestGlobalPhaseArgShapes:
    """Standalone global phase is invisible across diverse argument shapes.

    Each body deterministically maps |0...0> to a known Z-basis bitstring, so
    a correct (phase-invisible, body-applied) emission yields exactly that
    bitstring on every backend. Executed on Qiskit + QURI Parts (+ CUDA-Q via
    importorskip) through ``sdk_transpiler``; theta is a runtime parameter, so
    the symbolic-phase path is exercised too.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_two_scalar_qubits(self, sdk_transpiler, seed):
        """global_phase over a 2-qubit entangling body; |00> -> |11>."""
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs[0], qs[1] = qmc.global_phase(_two_qubit_body, angle)(qs[0], qs[1])
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["angle"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {(1, 1)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_two_vector_qubit_args(self, sdk_transpiler, seed):
        """global_phase over a body taking TWO Vector[Qubit] args."""
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(4, "qs")
            v0 = qs[0:2]
            v1 = qs[2:4]
            v0, v1 = qmc.global_phase(_two_vec_body, angle)(v0, v1)
            qs[0:2] = v0
            qs[2:4] = v1
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["angle"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {(1, 1, 1, 1)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_vector_plus_sliceview_mixed(self, sdk_transpiler, seed):
        """global_phase call mixing a whole Vector[Qubit] and a VectorView slice."""
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            whole = qmc.qubit_array(2, "whole")
            big = qmc.qubit_array(4, "big")
            view = big[1:3]  # a VectorView slice
            whole, view = qmc.global_phase(_two_vec_body, angle)(whole, view)
            big[1:3] = view
            # whole -> (1,1); big elements 1,2 -> 1, elements 0,3 stay 0
            return qmc.measure(big)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["angle"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {(0, 1, 1, 0)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_strided_sliceview(self, sdk_transpiler, seed):
        """global_phase over a strided VectorView ``qs[0::2]``."""
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(4, "qs")
            view = qs[0::2]  # elements 0 and 2
            view = qmc.global_phase(_vec_body, angle)(view)
            qs[0::2] = view
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["angle"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {(1, 0, 1, 0)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_interleaved_quantum_classical_args(self, sdk_transpiler, seed):
        """global_phase over a body with interleaved Qubit / Float / Qubit args."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))
        inner = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def circ(angle: qmc.Float, inner_ang: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs[0], qs[1] = qmc.global_phase(_interleaved_body, angle)(
                qs[0], inner_ang, qs[1]
            )
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(
            circ, parameters=["angle", "inner_ang"]
        )
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed),
                shots=512,
                bindings={"angle": theta, "inner_ang": inner},
            ).result()
        )
        assert set(counts) == {(1, 1)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_vector_float_param_body(self, sdk_transpiler, seed):
        """global_phase over a body carrying a Vector[Float] classical param."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))
        angs = rng.uniform(-np.pi, np.pi, size=2)

        @qkernel
        def circ(angle: qmc.Float, ar: qmc.Vector[qmc.Float]) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs = qmc.global_phase(_vecfloat_body, angle)(qs, ar)
            return qmc.measure(qs)

        # ``ar`` is an array parameter -> compile-time binding; ``angle`` is a
        # scalar runtime parameter.
        exe = sdk_transpiler.transpiler.transpile(
            circ, bindings={"ar": angs}, parameters=["angle"]
        )
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed),
                shots=512,
                bindings={"angle": theta},
            ).result()
        )
        assert set(counts) == {(1, 1)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_expval_invariant_vector_body(self, sdk_transpiler, seed):
        """Standalone phase leaves a multi-qubit expectation value unchanged."""
        import qamomile.observable as qm_o

        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def phased(obs: qmc.Observable) -> qmc.Float:
            """Build the globally phased circuit path."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
            qs = qmc.global_phase(_vec_body, theta)(qs)
            return qmc.expval(qs, obs)

        @qkernel
        def plain(obs: qmc.Observable) -> qmc.Float:
            """Build the unphased reference circuit."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
            qs = _vec_body(qs)
            return qmc.expval(qs, obs)

        obs = qm_o.Z(0) * qm_o.Z(1)
        tr = sdk_transpiler.transpiler
        v_phase = (
            tr.transpile(phased, bindings={"obs": obs}).run(tr.executor()).result()
        )
        v_plain = tr.transpile(plain, bindings={"obs": obs}).run(tr.executor()).result()
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(v_phase, v_plain, rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name}: {v_phase} vs {v_plain}"
        )


class TestGlobalPhaseSpecialCases:
    """Loop bodies, multi-control, inverse, phase-from-vector, composition."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_global_phase_on_loop_variable_element(self, sdk_transpiler, n, seed):
        """``global_phase(body)(qs[i])`` inside ``qmc.range`` transpiles + runs.

        Regression for the loop-unroll analysis: when the wrapped call carries
        the only loop-variable element access, the loop must still be unrolled,
        otherwise Qiskit's native for-loop path cannot resolve ``qs[i]``.
        """
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def circ(m: qmc.UInt, angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(m, "qs")
            for i in qmc.range(m):
                qs[i] = qmc.global_phase(_x_body, angle)(qs[i])
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(
            circ, bindings={"m": n}, parameters=["angle"]
        )
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {tuple(1 for _ in range(n))}, (
            f"{sdk_transpiler.backend_name} n={n}: {counts}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_two_control_phase_kickback(self, sdk_transpiler, seed):
        """Controlling a global phase by TWO controls is a 2-controlled phase.

        With both controls prepared in |+>, the kickback applies ``e^{iθ}``
        only on the |11> control branch, so a Hadamard test on the joint
        controls measures ``P(both 0) = (3 + 2cos θ + ...)``; instead of the
        closed form, compare with/without the phase having the SAME structure
        is not possible here, so assert the |11>-branch interference matches
        the analytic 2-control kickback probability.
        """
        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased_ident(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def htest(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the Hadamard-test circuit for phase kickback."""
            cs = qmc.qubit_array(2, "cs")
            q = qmc.qubit("q")
            cs[0] = qmc.h(cs[0])
            cs[1] = qmc.h(cs[1])
            cs[0], cs[1], q = qmc.control(phased_ident, num_controls=2)(
                cs[0], cs[1], q, angle
            )
            cs[0] = qmc.h(cs[0])
            cs[1] = qmc.h(cs[1])
            return qmc.measure(cs)

        exe = sdk_transpiler.transpiler.transpile(htest, parameters=["angle"])
        shots = 20000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"angle": theta}
            ).result()
        )
        # Controls start in |++>; CCphase(θ) applies e^{iθ} to the |11> branch
        # only. After H⊗H, P(00) = |3 + e^{iθ}|^2 / 16.
        p00 = counts.get((0, 0), 0) / shots
        expected = abs(3 + np.exp(1j * theta)) ** 2 / 16
        assert np.isclose(p00, expected, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(00)={p00} vs {expected}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_inverse_roundtrip_is_identity(self, sdk_transpiler, seed):
        """``global_phase`` followed by its ``inverse`` is the identity."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def phased(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Build the globally phased circuit path."""
            return qmc.global_phase(_hz_body, angle)(q)

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            q = qmc.qubit("q")
            q = qmc.x(q)  # start from |1> so identity is observable as outcome 1
            q = phased(q, angle)
            q = qmc.inverse(phased)(q, angle)
            return qmc.measure(q)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["angle"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_inverse_unitary_negates_phase(self, qiskit_transpiler, seed):
        """``inverse(global_phase(body, θ)) == e^{-iθ} · U(body)†`` (Qiskit)."""
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def phased(q: qmc.Qubit) -> qmc.Qubit:
            """Build the globally phased circuit path."""
            return qmc.global_phase(_hz_body, theta)(q)

        @qkernel
        def inv_circ() -> qmc.Bit:
            """Build the circuit for the inverse phased body."""
            q = qmc.qubit("q")
            q = qmc.inverse(phased)(q)
            return qmc.measure(q)

        @qkernel
        def body_circ() -> qmc.Bit:
            """Build the circuit for the non-inverted phased body."""
            q = qmc.qubit("q")
            q = _hz_body(q)
            return qmc.measure(q)

        u_inv = _unitary(
            qiskit_transpiler.transpile(inv_circ, bindings={})
            .compiled_quantum[0]
            .circuit
        )
        u_body = _unitary(
            qiskit_transpiler.transpile(body_circ, bindings={})
            .compiled_quantum[0]
            .circuit
        )
        assert np.allclose(
            u_inv, np.exp(-1j * theta) * u_body.conj().T, rtol=0.0, atol=1e-9
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_phase_from_vector_float_element_kickback(self, sdk_transpiler, seed):
        """theta drawn from a Vector[Float] element survives into the kickback."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.3, np.pi - 0.3, size=3)
        idx = int(rng.integers(0, 3))

        @qkernel
        def phased_ident(q: qmc.Qubit, angles: qmc.Vector[qmc.Float]) -> qmc.Qubit:
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, angles[idx])(q)

        @qkernel
        def htest(ph: qmc.Vector[qmc.Float]) -> qmc.Bit:
            """Build the Hadamard-test circuit for phase kickback."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(phased_ident)(ctrl, q, ph)
            ctrl = qmc.h(ctrl)
            return qmc.measure(ctrl)

        exe = sdk_transpiler.transpiler.transpile(htest, parameters=["ph"])
        shots = 20000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"ph": phases}
            ).result()
        )
        p0 = counts.get(0, 0) / shots
        assert np.isclose(p0, np.cos(phases[idx] / 2) ** 2, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name}: P(0)={p0}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_phase_composition_adds(self, qiskit_transpiler, seed):
        """Nesting global_phase adds the phases: e^{iα}·e^{iβ}·U(body) (Qiskit)."""
        rng = np.random.default_rng(seed)
        alpha = float(rng.uniform(-np.pi, np.pi))
        beta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            """Apply the inner globally phased body."""
            return qmc.global_phase(_x_body, beta)(q)

        @qkernel
        def outer() -> qmc.Bit:
            """Build the outer kernel that invokes the test body."""
            q = qmc.qubit("q")
            q = qmc.global_phase(inner, alpha)(q)
            return qmc.measure(q)

        @qkernel
        def plain() -> qmc.Bit:
            """Build the unphased reference circuit."""
            q = qmc.qubit("q")
            q = _x_body(q)
            return qmc.measure(q)

        u_nested = _unitary(
            qiskit_transpiler.transpile(outer, bindings={}).compiled_quantum[0].circuit
        )
        u_plain = _unitary(
            qiskit_transpiler.transpile(plain, bindings={}).compiled_quantum[0].circuit
        )
        assert np.allclose(
            u_nested, np.exp(1j * (alpha + beta)) * u_plain, rtol=0.0, atol=1e-9
        )


class TestGlobalPhaseControlledCompositions:
    """Controlled compositions that earlier slipped through (bug regressions).

    These cover the cells a second adversarial pass flagged: a controlled
    ``inverse(global_phase)`` (was dropped on CUDA-Q), a loop whose only
    loop-variable dependency is the phase angle (was un-unrolled on Qiskit),
    controlled expectation values, a gate-bearing controlled body, the
    three-control contract, and a register-size sweep -- all executed on every
    SDK backend.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_controlled_inverse_phase_kickback(self, sdk_transpiler, seed):
        """``control(inverse(global_phase(ident, θ)))`` kicks back ``-θ``.

        CUDA-Q's helper omits an abstract whole-circuit phase, so its
        CircuitProgram materializer must negate the explicit control-phase
        correction. `cos^2` is even, so the kickback magnitude is
        `cos^2(θ/2)` either sign; ``test_controlled_phase_cancellation`` pins
        the sign itself.
        """
        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Build the globally phased circuit path."""
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def inv_phased(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply the inverse of the globally phased body."""
            return qmc.inverse(phased)(q, angle)

        @qkernel
        def htest(angle: qmc.Float) -> qmc.Bit:
            """Build the Hadamard-test circuit for phase kickback."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(inv_phased)(ctrl, q, angle)
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
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(0)={p0}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_controlled_phase_cancellation(self, sdk_transpiler, seed):
        """``control(gp)`` then ``control(inverse(gp))`` cancel: ``P(0)=1``.

        Pins the *sign* of the controlled inverse phase: the forward ``+θ`` and
        inverse ``-θ`` kickbacks must cancel exactly, so the Hadamard test
        returns the control to ``|0>`` on every backend.
        """
        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Build the globally phased circuit path."""
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def inv_phased(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply the inverse of the globally phased body."""
            return qmc.inverse(phased)(q, angle)

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(phased)(ctrl, q, angle)
            ctrl, q = qmc.control(inv_phased)(ctrl, q, angle)
            ctrl = qmc.h(ctrl)
            return qmc.measure(ctrl)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["angle"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=4000, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {0}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_loop_phase_only_depends_on_loop_var(self, sdk_transpiler, seed):
        """A loop whose only loop-var dependency is the phase angle runs.

        Regression for the unroll analysis: ``global_phase(body, angs[i])``
        with a fixed target must still force the loop to unroll (Qiskit's
        native for-loop cannot bind ``angs[i]`` per iteration). X applied
        twice returns the qubit to ``|0>``; the phases are invisible.
        """
        rng = np.random.default_rng(seed)
        angs = rng.uniform(-np.pi, np.pi, size=2)

        @qkernel
        def circ(ar: qmc.Vector[qmc.Float]) -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(1, "qs")
            for i in qmc.range(2):
                qs[0] = qmc.global_phase(_x_body, ar[i])(qs[0])
            return qmc.measure(qs[0])

        exe = sdk_transpiler.transpiler.transpile(circ, bindings={"ar": angs})
        counts = _counts(
            exe.sample(_executor(sdk_transpiler, seed), shots=512).result()
        )
        assert set(counts) == {0}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1])
    def test_control_call_phase_from_loop_element_kickback(
        self,
        sdk_transpiler,
        seed,
    ):
        """A loop-indexed call-site phase unrolls and adds coherently."""
        angles = np.random.default_rng(seed).uniform(0.2, 0.7, size=2)

        @qkernel
        def circuit(phases: qmc.Vector[qmc.Float]) -> qmc.Bit:
            """Apply two indexed phases directly at controlled call sites."""
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control = qmc.h(control)
            for index in qmc.range(2):
                control, target = qmc.control(_ident)(
                    control,
                    target,
                    global_phase=phases[index],
                )
            control = qmc.h(control)
            return qmc.measure(control)

        executable = sdk_transpiler.transpiler.transpile(
            circuit,
            bindings={"phases": angles},
        )
        shots = 12000
        counts = _counts(
            executable.sample(
                _executor(sdk_transpiler, seed),
                shots=shots,
            ).result()
        )
        probability_zero = counts.get(0, 0) / shots
        expected = np.cos(float(np.sum(angles)) / 2) ** 2
        assert np.isclose(probability_zero, expected, rtol=0.0, atol=0.035), (
            f"{sdk_transpiler.backend_name}: P(0)={probability_zero} vs {expected}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_controlled_loop_carried_phase_kickback(self, sdk_transpiler, seed):
        """A carried scalar contributes each iteration's coherent phase.

        The three iterations apply ``step``, ``2*step``, and ``3*step``.
        Controlling that body therefore kicks back total phase ``6*step`` and
        the Hadamard test measures ``P(0) = cos²(3*step)``.
        """
        step = float(np.random.default_rng(seed).uniform(0.2, 0.8))

        @qkernel
        def htest(angle: qmc.Float) -> qmc.Bit:
            """Expose the carried phase as interference on one control."""
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control = qmc.h(control)
            control, target = qmc.control(_loop_carried_phase_body)(
                control,
                target,
                angle,
            )
            control = qmc.h(control)
            return qmc.measure(control)

        executable = sdk_transpiler.transpiler.transpile(
            htest,
            parameters=["angle"],
        )
        shots = 20000
        counts = _counts(
            executable.sample(
                _executor(sdk_transpiler, seed),
                shots=shots,
                bindings={"angle": step},
            ).result()
        )
        probability_zero = counts.get(0, 0) / shots
        assert np.isclose(
            probability_zero, np.cos(3.0 * step) ** 2, rtol=0.0, atol=0.03
        ), f"{sdk_transpiler.backend_name} step={step}: P(0)={probability_zero}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_controlled_loop_carried_pauli_identity_phase(self, sdk_transpiler, seed):
        """A carried Pauli time preserves the identity term as kickback."""
        import qamomile.observable as qm_o

        step = float(np.random.default_rng(seed).uniform(0.2, 0.8))

        @qkernel
        def htest(hamiltonian: qmc.Observable, angle: qmc.Float) -> qmc.Bit:
            """Expose the carried identity-evolution phase on one control."""
            control = qmc.h(qmc.qubit("control"))
            target = qmc.qubit_array(1, "target")
            control, target = qmc.control(_loop_carried_pauli_phase_body)(
                control,
                target,
                hamiltonian,
                angle,
            )
            control = qmc.h(control)
            return qmc.measure(control)

        executable = sdk_transpiler.transpiler.transpile(
            htest,
            bindings={"hamiltonian": qm_o.Hamiltonian.identity(1.0)},
            parameters=["angle"],
        )
        shots = 12000
        counts = _counts(
            executable.sample(
                _executor(sdk_transpiler, seed),
                shots=shots,
                bindings={"angle": step},
            ).result()
        )
        probability_zero = counts.get(0, 0) / shots
        assert np.isclose(
            probability_zero, np.cos(3.0 * step) ** 2, rtol=0.0, atol=0.04
        ), f"{sdk_transpiler.backend_name} step={step}: P(0)={probability_zero}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_controlled_phase_expectation_value(self, sdk_transpiler, seed):
        """A controlled global phase moves ``<Z>`` on the control: ``cos θ``.

        Fills the controlled-expectation-value gap. ``H · cP(θ) · H`` on a
        control prepared in ``|0>`` gives ``<Z> = cos θ`` (the ancilla qubit
        stays ``|0>`` and is covered by the identity observable factor).
        """
        import qamomile.observable as qm_o

        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def phased_ident(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def htest(obs: qmc.Observable, angle: qmc.Float) -> qmc.Float:
            """Build the Hadamard-test circuit for phase kickback."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.h(qs[0])
            qs[0], qs[1] = qmc.control(phased_ident)(qs[0], qs[1], angle)
            qs[0] = qmc.h(qs[0])
            return qmc.expval(qs, obs)

        tr = sdk_transpiler.transpiler
        val = (
            tr.transpile(htest, bindings={"obs": qm_o.Z(0), "angle": theta})
            .run(tr.executor())
            .result()
        )
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(val, np.cos(theta), rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name} θ={theta}: <Z>={val} vs {np.cos(theta)}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_controlled_gate_bearing_body_is_applied(self, sdk_transpiler, seed):
        """A controlled gate-bearing global-phase body applies its gates.

        Every other controlled test uses an identity body; this one wraps an
        X so a dropped body would be observable. With the control prepared in
        ``|1>``, the controlled ``e^{iθ}X`` flips the target to ``|1>`` (the
        phase is a global factor of the fired branch, hence invisible here).
        """
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def phased_x(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply a globally phased X body."""
            return qmc.global_phase(_x_body, angle)(q)

        @qkernel
        def circ(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.x(qs[0])  # control = |1> so the controlled op fires
            qs[0], qs[1] = qmc.control(phased_x)(qs[0], qs[1], angle)
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["angle"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {(1, 1)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_three_control_phase_cross_backend(self, sdk_transpiler, seed):
        """Three-control global phase executes on every supported backend.

        Qiskit and CUDA-Q use native control support. QURI Parts uses the
        shared clean-ancilla Toffoli-cascade decomposition. Every backend must
        match the analytic 3-control kickback ``|7 + e^{iθ}|² / 64``.
        """
        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased_ident(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def htest(angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the Hadamard-test circuit for phase kickback."""
            cs = qmc.qubit_array(3, "cs")
            q = qmc.qubit("q")
            cs[0] = qmc.h(cs[0])
            cs[1] = qmc.h(cs[1])
            cs[2] = qmc.h(cs[2])
            cs[0], cs[1], cs[2], q = qmc.control(phased_ident, num_controls=3)(
                cs[0], cs[1], cs[2], q, angle
            )
            cs[0] = qmc.h(cs[0])
            cs[1] = qmc.h(cs[1])
            cs[2] = qmc.h(cs[2])
            return qmc.measure(cs)

        tr = sdk_transpiler.transpiler
        exe = tr.transpile(htest, parameters=["angle"])
        shots = 30000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"angle": theta}
            ).result()
        )
        p000 = counts.get((0, 0, 0), 0) / shots
        expected = abs(7 + np.exp(1j * theta)) ** 2 / 64
        assert np.isclose(p000, expected, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(000)={p000} vs {expected}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_vector_body_size_sweep(self, sdk_transpiler, n, seed):
        """A phased Vector[Qubit] body works across register sizes (X all)."""
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def vec_x(qs: qmc.Vector[qmc.Qubit], m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            """Apply X gates across the requested register prefix."""
            for i in qmc.range(m):
                qs[i] = qmc.x(qs[i])
            return qs

        @qkernel
        def circ(m: qmc.UInt, angle: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(m, "qs")
            qs = qmc.global_phase(vec_x, angle)(qs, m)
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(
            circ, bindings={"m": n}, parameters=["angle"]
        )
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"angle": theta}
            ).result()
        )
        assert set(counts) == {tuple(1 for _ in range(n))}, (
            f"{sdk_transpiler.backend_name} n={n}: {counts}"
        )


@qkernel
def _control_sub(q: qmc.Qubit) -> qmc.Qubit:
    """Apply the leaf X gate used by a controlled body.

    Args:
        q (qmc.Qubit): Target qubit.

    Returns:
        qmc.Qubit: Updated target qubit.
    """
    return qmc.x(q)


@qkernel
def _control_bearing_body(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply a controlled X through the control combinator.

    Args:
        a (qmc.Qubit): Control qubit.
        b (qmc.Qubit): Target qubit.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Updated control and target qubits.
    """
    a, b = qmc.control(_control_sub)(a, b)
    return a, b


class TestGlobalPhaseRound3Coverage:
    """Round-3 regression / gap-filler tests, executed on every SDK backend.

    Covers a found bug (a control-bearing body inside ``inverse(global_phase)``
    crashed CUDA-Q's adjoint autogeneration) plus deeper compositions and the
    controlled-expectation-value column the coverage audit flagged.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_inverse_of_control_bearing_body(self, sdk_transpiler, seed):
        """``inverse(global_phase(body-with-control))`` runs (no CUDA-Q crash).

        Regression: CUDA-Q tried to autogenerate ``cudaq.adjoint`` of a kernel
        containing controlled-kernel synthesis (the adjoint-helper validator
        did not inspect the wrapped body) and
        aborted the interpreter. The forward then the inverse of a
        control-bearing body must cancel: with the control prepared in ``|1>``
        the inner CX flips ``b`` to ``|1>`` and the inverse flips it back.
        """
        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def gp_body(
            a: qmc.Qubit, b: qmc.Qubit, ang: qmc.Float
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply a global phase to the control-bearing body."""
            return qmc.global_phase(_control_bearing_body, ang)(a, b)

        @qkernel
        def inv_gp(
            a: qmc.Qubit, b: qmc.Qubit, ang: qmc.Float
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply the inverse globally phased control-bearing body."""
            return qmc.inverse(gp_body)(a, b, ang)

        @qkernel
        def circ(ang: qmc.Float) -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.x(qs[0])  # control = |1> so the inner CX fires
            qs[0], qs[1] = gp_body(qs[0], qs[1], ang)
            qs[0], qs[1] = inv_gp(qs[0], qs[1], ang)
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(circ, parameters=["ang"])
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=512, bindings={"ang": theta}
            ).result()
        )
        assert set(counts) == {(1, 0)}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_double_inverse_phase_kickback(self, sdk_transpiler, seed):
        """``control(inverse(inverse(global_phase(ident, θ))))`` kicks back ``+θ``."""
        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def gp1(q: qmc.Qubit, ang: qmc.Float) -> qmc.Qubit:
            """Apply the first global-phase wrapper."""
            return qmc.global_phase(_ident, ang)(q)

        @qkernel
        def gp2(q: qmc.Qubit, ang: qmc.Float) -> qmc.Qubit:
            """Apply the inverse global-phase wrapper."""
            return qmc.inverse(gp1)(q, ang)

        @qkernel
        def gp3(q: qmc.Qubit, ang: qmc.Float) -> qmc.Qubit:
            """Apply the second inverse global-phase wrapper."""
            return qmc.inverse(gp2)(q, ang)

        @qkernel
        def htest(ang: qmc.Float) -> qmc.Bit:
            """Build the Hadamard-test circuit for phase kickback."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(gp3)(ctrl, q, ang)
            ctrl = qmc.h(ctrl)
            return qmc.measure(ctrl)

        exe = sdk_transpiler.transpiler.transpile(htest, parameters=["ang"])
        shots = 20000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"ang": theta}
            ).result()
        )
        p0 = counts.get(0, 0) / shots
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(0)={p0}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_repeated_controlled_phase_accumulates(self, sdk_transpiler, seed):
        """Applying ``control(global_phase)`` twice accumulates ``2θ`` on control."""
        theta = float(np.random.default_rng(seed).uniform(0.2, np.pi / 2 - 0.2))

        @qkernel
        def phased_ident(q: qmc.Qubit, ang: qmc.Float) -> qmc.Qubit:
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, ang)(q)

        @qkernel
        def htest(ang: qmc.Float) -> qmc.Bit:
            """Build the Hadamard-test circuit for phase kickback."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(phased_ident)(ctrl, q, ang)
            ctrl, q = qmc.control(phased_ident)(ctrl, q, ang)
            ctrl = qmc.h(ctrl)
            return qmc.measure(ctrl)

        exe = sdk_transpiler.transpiler.transpile(htest, parameters=["ang"])
        shots = 20000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"ang": theta}
            ).result()
        )
        # Two P(θ) kickbacks compose to P(2θ): P(0) = cos^2(θ).
        p0 = counts.get(0, 0) / shots
        assert np.isclose(p0, np.cos(theta) ** 2, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(0)={p0} vs {np.cos(theta) ** 2}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_two_control_expectation_value(self, sdk_transpiler, seed):
        """A two-control global phase moves a two-control expectation value.

        With both controls in ``|+>`` and a ``|11>``-only phase ``e^{iθ}``, the
        final ``H^2`` maps ``<Z0 Z1>`` to ``<X0 X1>`` of the phased state, which
        evaluates to ``(1 + cos θ) / 2``.
        """
        import qamomile.observable as qm_o

        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))

        @qkernel
        def phased_ident(q: qmc.Qubit, ang: qmc.Float) -> qmc.Qubit:
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, ang)(q)

        @qkernel
        def htest(obs: qmc.Observable, ang: qmc.Float) -> qmc.Float:
            """Build the Hadamard-test circuit for phase kickback."""
            qs = qmc.qubit_array(3, "qs")
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
            qs[0], qs[1], qs[2] = qmc.control(phased_ident, num_controls=2)(
                qs[0], qs[1], qs[2], ang
            )
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
            return qmc.expval(qs, obs)

        obs = qm_o.Z(0) * qm_o.Z(1)
        tr = sdk_transpiler.transpiler
        val = (
            tr.transpile(htest, bindings={"obs": obs, "ang": theta})
            .run(tr.executor())
            .result()
        )
        expected = (1 + np.cos(theta)) / 2
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(val, expected, rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name} θ={theta}: <Z0 Z1>={val} vs {expected}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_classical_param_body_executes_cross_backend(self, sdk_transpiler, seed):
        """A classical-Float-param phased body samples on every backend.

        Promotes the formerly Qiskit-unitary-only classical-param body to a
        cross-backend execution check. ``rz`` is invisible in the Z basis, so
        ``X`` then a phased ``rz`` deterministically yields ``|1>``.
        """
        rng = np.random.default_rng(seed)
        theta = float(rng.uniform(-np.pi, np.pi))
        inner = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def circ(angle: qmc.Float, inner_ang: qmc.Float) -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.global_phase(_rz_body, angle)(q, inner_ang)
            return qmc.measure(q)

        exe = sdk_transpiler.transpiler.transpile(
            circ, parameters=["angle", "inner_ang"]
        )
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed),
                shots=512,
                bindings={"angle": theta, "inner_ang": inner},
            ).result()
        )
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: {counts}"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_phase_composition_executes_cross_backend(self, sdk_transpiler, seed):
        """A nested ``global_phase`` composition samples on every backend.

        Promotes the formerly Qiskit-unitary-only composition body to a
        cross-backend execution check; the inner X is the observable effect.
        """
        rng = np.random.default_rng(seed)
        alpha = float(rng.uniform(-np.pi, np.pi))
        beta = float(rng.uniform(-np.pi, np.pi))

        @qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            """Apply the inner globally phased body."""
            return qmc.global_phase(_x_body, beta)(q)

        @qkernel
        def circ() -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            q = qmc.qubit("q")
            q = qmc.global_phase(inner, alpha)(q)
            return qmc.measure(q)

        exe = sdk_transpiler.transpiler.transpile(circ, bindings={})
        counts = _counts(
            exe.sample(_executor(sdk_transpiler, seed), shots=512).result()
        )
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: {counts}"


class TestGlobalPhaseRound5Coverage:
    """Round-5 regression: a controlled global phase nested in a compile-time ``if``.

    The compile-time branch must be selected before CircuitProgram phase
    aggregation, including when it is nested in a controlled callable. The
    selected phase becomes relative under the outer control; the dead branch
    must contribute neither gates nor phase.
    """

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_controlled_phase_in_compile_time_if_true_branch(
        self, sdk_transpiler, seed
    ):
        """``control(if True: global_phase)`` kicks back ``θ`` on the control."""
        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased_in_if(q: qmc.Qubit, ang: qmc.Float) -> qmc.Qubit:
            """Apply a global phase in the selected true branch."""
            if True:
                q = qmc.global_phase(_ident, ang)(q)
            return q

        @qkernel
        def htest(ang: qmc.Float) -> qmc.Bit:
            """Build the Hadamard-test circuit for phase kickback."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(phased_in_if)(ctrl, q, ang)
            ctrl = qmc.h(ctrl)
            return qmc.measure(ctrl)

        exe = sdk_transpiler.transpiler.transpile(htest, parameters=["ang"])
        shots = 20000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"ang": theta}
            ).result()
        )
        p0 = counts.get(0, 0) / shots
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(0)={p0}"
        )

    @pytest.mark.parametrize("seed", [0, 1, 2, 42])
    def test_controlled_phase_in_compile_time_if_else_branch(
        self, sdk_transpiler, seed
    ):
        """``control(if False: x else: global_phase)`` kicks back ``θ`` only.

        The dead ``if False`` branch's ``x`` must not be emitted; only the
        ``else`` branch's global phase contributes, so the kickback is the
        same ``cos^2(θ/2)`` as the true-branch case.
        """
        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased_in_else(q: qmc.Qubit, ang: qmc.Float) -> qmc.Qubit:
            """Apply a global phase in the selected false branch."""
            if False:
                q = qmc.x(q)
            else:
                q = qmc.global_phase(_ident, ang)(q)
            return q

        @qkernel
        def htest(ang: qmc.Float) -> qmc.Bit:
            """Build the Hadamard-test circuit for phase kickback."""
            ctrl = qmc.qubit("ctrl")
            q = qmc.qubit("q")
            ctrl = qmc.h(ctrl)
            ctrl, q = qmc.control(phased_in_else)(ctrl, q, ang)
            ctrl = qmc.h(ctrl)
            return qmc.measure(ctrl)

        exe = sdk_transpiler.transpiler.transpile(htest, parameters=["ang"])
        shots = 20000
        counts = _counts(
            exe.sample(
                _executor(sdk_transpiler, seed), shots=shots, bindings={"ang": theta}
            ).result()
        )
        p0 = counts.get(0, 0) / shots
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.03), (
            f"{sdk_transpiler.backend_name} θ={theta}: P(0)={p0}"
        )

    @pytest.mark.parametrize("take_true", [True, False], ids=["true", "else"])
    def test_controlled_compile_time_if_phi_output_consumed(
        self, sdk_transpiler, take_true
    ):
        """A gate after a compile-time ``if`` in a controlled body resolves its phi.

        ``control(inner)`` where ``inner`` conditionally rewrites ``q`` and then
        applies a further gate to it (``if <const>: q = x(q)`` then
        ``q = h(q)``) must alias the ``if``'s phi output to the selected
        branch's value before resolving the trailing gate. The QURI Parts
        controlled fallback used to leave the phi output unmapped and raise
        ``cannot resolve gate operand 'q_phi_0'``. Both the taken-branch and
        the dead-branch (``else``) selections are covered. For both bodies the
        controlled unitary on ``|+>_0 |0>_1`` yields the analytic distribution
        ``(0,0):1/2, (1,0):1/4, (1,1):1/4``.
        """

        @qkernel
        def inner_true(q: qmc.Qubit) -> qmc.Qubit:
            """Apply the true-branch body and preserve its output."""
            if True:
                q = qmc.x(q)
            q = qmc.h(q)
            return q

        @qkernel
        def inner_else(q: qmc.Qubit) -> qmc.Qubit:
            """Apply the false-branch body and preserve its output."""
            if False:
                q = qmc.x(q)
            q = qmc.h(q)
            return q

        body = inner_true if take_true else inner_else

        @qkernel
        def circ() -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs[0] = qmc.h(qs[0])
            qs[0], qs[1] = qmc.control(body)(qs[0], qs[1])
            return qmc.measure(qs)

        exe = sdk_transpiler.transpiler.transpile(circ)
        shots = 20000
        counts = _counts(exe.sample(_executor(sdk_transpiler), shots=shots).result())
        probs = {key: value / shots for key, value in counts.items()}
        expected = {(0, 0): 0.5, (1, 0): 0.25, (1, 1): 0.25}
        for outcome, want in expected.items():
            assert np.isclose(probs.get(outcome, 0.0), want, rtol=0.0, atol=0.03), (
                f"{sdk_transpiler.backend_name} take_true={take_true}: "
                f"{outcome}={probs.get(outcome, 0.0)} (want {want}); full={probs}"
            )


class TestGlobalPhaseDeepControlRegressions:
    """Exercise global phase through composite, nested-control, and loop paths."""

    @pytest.mark.parametrize("seed", [0, 42])
    def test_controlled_composite_preserves_phase(self, sdk_transpiler, seed):
        """The callable transform controls a composite's phased target body.

        Preparing the target in ``|+>`` turns controlled
        ``exp(i*theta) X`` into pure phase kickback on the callable transform's
        control.
        """
        import qamomile.observable as qm_o

        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))
        gate = _controlled_phased_x(theta)

        @qkernel
        def prepare(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            """Prepare the controlled-phase test register."""
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.h(qs[1])
            qs[0], qs[1] = gate(qs[0], qs[1])
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def sample_circuit() -> qmc.Bit:
            """Build the sampling circuit used by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs = prepare(qs)
            return qmc.measure(qs[0])

        @qkernel
        def expval_circuit(obs: qmc.Observable) -> qmc.Float:
            """Build the expectation-value circuit used by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs = prepare(qs)
            return qmc.expval(qs, obs)

        tr = sdk_transpiler.transpiler
        shots = 12000
        counts = _counts(
            tr.transpile(sample_circuit)
            .sample(_executor(sdk_transpiler, seed), shots=shots)
            .result()
        )
        p0 = counts.get(0, 0) / shots
        value = (
            tr.transpile(expval_circuit, bindings={"obs": qm_o.Z(0)})
            .run(tr.executor())
            .result()
        )
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.035), (
            f"{sdk_transpiler.backend_name} theta={theta}: P(0)={p0}"
        )
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(value, np.cos(theta), rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name} theta={theta}: <Z>={value}"
        )

    @pytest.mark.parametrize("seed", [0, 42])
    def test_outer_control_composes_with_controlled_composite(
        self, sdk_transpiler, seed
    ):
        """Nested callable transforms both gate a target-only phase.

        The inner control is fixed to ``|1>`` while an outer control is prepared
        in ``|+>``. The target is an X eigenstate, so the two-level controlled
        composite contributes exactly one ``theta`` kickback to the outer
        control.
        """
        import qamomile.observable as qm_o

        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))
        gate = _controlled_phased_x(theta)

        @qkernel
        def composite_body(
            inner_control: qmc.Qubit,
            target: qmc.Qubit,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply the composite body with its inner control."""
            return gate(inner_control, target)

        @qkernel
        def prepare(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            """Prepare the controlled-phase test register."""
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[2] = qmc.h(qs[2])
            qs[0], qs[1], qs[2] = qmc.control(composite_body)(qs[0], qs[1], qs[2])
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def sample_circuit() -> qmc.Bit:
            """Build the sampling circuit used by this test."""
            qs = qmc.qubit_array(3, "qs")
            qs = prepare(qs)
            return qmc.measure(qs[0])

        @qkernel
        def expval_circuit(obs: qmc.Observable) -> qmc.Float:
            """Build the expectation-value circuit used by this test."""
            qs = qmc.qubit_array(3, "qs")
            qs = prepare(qs)
            return qmc.expval(qs, obs)

        tr = sdk_transpiler.transpiler
        shots = 12000
        counts = _counts(
            tr.transpile(sample_circuit)
            .sample(_executor(sdk_transpiler, seed), shots=shots)
            .result()
        )
        p0 = counts.get(0, 0) / shots
        value = (
            tr.transpile(expval_circuit, bindings={"obs": qm_o.Z(0)})
            .run(tr.executor())
            .result()
        )
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.035), (
            f"{sdk_transpiler.backend_name} theta={theta}: P(0)={p0}"
        )
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(value, np.cos(theta), rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name} theta={theta}: <Z>={value}"
        )

    @pytest.mark.parametrize("seed", [0, 42])
    def test_nested_controlled_phase_is_encoded_once(self, sdk_transpiler, seed):
        """An outer control correctly controls an inner controlled phase once.

        The inner control is fixed to ``|1>``. The nested controlled-U therefore
        contributes one ``theta`` kickback to the outer control. Recursive call
        materialization must propagate both controls while accounting for the
        nested body phase exactly once.
        """
        import qamomile.observable as qm_o

        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased_identity(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply a global phase to an identity body."""
            return qmc.global_phase(_ident, angle)(q)

        @qkernel
        def inner_control(
            control: qmc.Qubit,
            target: qmc.Qubit,
            angle: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Apply the inner controlled phased identity."""
            return qmc.control(phased_identity)(control, target, angle)

        @qkernel
        def prepare(
            qs: qmc.Vector[qmc.Qubit], angle: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            """Prepare the controlled-phase test register."""
            qs[0] = qmc.h(qs[0])
            qs[1] = qmc.x(qs[1])
            qs[0], qs[1], qs[2] = qmc.control(inner_control)(qs[0], qs[1], qs[2], angle)
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def sample_circuit(angle: qmc.Float) -> qmc.Bit:
            """Build the sampling circuit used by this test."""
            qs = qmc.qubit_array(3, "qs")
            qs = prepare(qs, angle)
            return qmc.measure(qs[0])

        @qkernel
        def expval_circuit(obs: qmc.Observable, angle: qmc.Float) -> qmc.Float:
            """Build the expectation-value circuit used by this test."""
            qs = qmc.qubit_array(3, "qs")
            qs = prepare(qs, angle)
            return qmc.expval(qs, obs)

        tr = sdk_transpiler.transpiler
        shots = 12000
        counts = _counts(
            tr.transpile(sample_circuit, parameters=["angle"])
            .sample(
                _executor(sdk_transpiler, seed),
                shots=shots,
                bindings={"angle": theta},
            )
            .result()
        )
        p0 = counts.get(0, 0) / shots
        value = (
            tr.transpile(
                expval_circuit,
                bindings={"obs": qm_o.Z(0), "angle": theta},
            )
            .run(tr.executor())
            .result()
        )
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.035), (
            f"{sdk_transpiler.backend_name} theta={theta}: P(0)={p0}"
        )
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(value, np.cos(theta), rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name} theta={theta}: <Z>={value}"
        )

    @pytest.mark.parametrize("seed", [0, 42])
    def test_loop_index_comparison_selects_one_controlled_phase(
        self, sdk_transpiler, seed
    ):
        """A controlled loop folds ``i == 0`` with each iteration binding.

        The controlled body retains ``ForOperation -> CompOp -> IfOperation``
        until backend emission. Exactly the first iteration contributes the
        phase, so sampling and expectation values both read back ``theta``.
        """
        import qamomile.observable as qm_o

        theta = float(np.random.default_rng(seed).uniform(0.3, np.pi - 0.3))

        @qkernel
        def phased_loop(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply the phase selected by the loop condition."""
            for i in qmc.range(2):
                if i == 0:
                    q = qmc.global_phase(_ident, angle)(q)
            return q

        @qkernel
        def prepare(
            qs: qmc.Vector[qmc.Qubit], angle: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            """Prepare the controlled-phase test register."""
            qs[0] = qmc.h(qs[0])
            qs[0], qs[1] = qmc.control(phased_loop)(qs[0], qs[1], angle)
            qs[0] = qmc.h(qs[0])
            return qs

        @qkernel
        def sample_circuit(angle: qmc.Float) -> qmc.Bit:
            """Build the sampling circuit used by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs = prepare(qs, angle)
            return qmc.measure(qs[0])

        @qkernel
        def expval_circuit(obs: qmc.Observable, angle: qmc.Float) -> qmc.Float:
            """Build the expectation-value circuit used by this test."""
            qs = qmc.qubit_array(2, "qs")
            qs = prepare(qs, angle)
            return qmc.expval(qs, obs)

        tr = sdk_transpiler.transpiler
        shots = 12000
        counts = _counts(
            tr.transpile(sample_circuit, parameters=["angle"])
            .sample(
                _executor(sdk_transpiler, seed),
                shots=shots,
                bindings={"angle": theta},
            )
            .result()
        )
        p0 = counts.get(0, 0) / shots
        value = (
            tr.transpile(
                expval_circuit,
                bindings={"obs": qm_o.Z(0), "angle": theta},
            )
            .run(tr.executor())
            .result()
        )
        assert np.isclose(p0, np.cos(theta / 2) ** 2, rtol=0.0, atol=0.035), (
            f"{sdk_transpiler.backend_name} theta={theta}: P(0)={p0}"
        )
        atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
        assert np.isclose(value, np.cos(theta), rtol=0.0, atol=atol), (
            f"{sdk_transpiler.backend_name} theta={theta}: <Z>={value}"
        )


class TestGlobalPhaseVisualization:
    """Drawing a global-phase kernel renders its wrapped body (not a blank wire)."""

    def test_box_mode_labels_the_wrapped_body(self):
        """In box mode the wrapped body remains visible as an ordinary call box."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qamomile.circuit.visualization.drawer import MatplotlibDrawer

        @qkernel
        def circ() -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            q = qmc.qubit("q")
            q = qmc.global_phase(_hz_body, 0.7)(q)
            return qmc.measure(q)

        fig = MatplotlibDrawer.draw_kernel(circ, expand_composite=False)
        labels = [t.get_text() for t in fig.axes[0].texts]
        assert "_HZ_BODY" in labels

    def test_expand_mode_retains_body_call(self):
        """Drawing the flat phase form retains the wrapped body's call box."""
        pytest.importorskip("matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        from qamomile.circuit.visualization.drawer import MatplotlibDrawer

        @qkernel
        def circ() -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            q = qmc.qubit("q")
            q = qmc.global_phase(_hz_body, 0.7)(q)
            return qmc.measure(q)

        fig = MatplotlibDrawer.draw_kernel(circ, expand_composite=True)
        labels = [t.get_text() for t in fig.axes[0].texts]
        assert "_HZ_BODY" in labels


class TestGlobalPhaseArgumentValidation:
    """global_phase shares qkernel argument validation without requiring unitarity."""

    def test_quantum_handle_into_classical_slot_rejected(self):
        """A Qubit bound to a Float parameter fails fast with a clear message."""

        @qkernel
        def bad(q: qmc.Qubit, r: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Build the intentionally invalid kernel used by this test."""
            q2 = qmc.global_phase(_rz_body, 0.1)(q, r)
            return q2, r

        with pytest.raises(
            TypeError, match="declared as a classical parameter but received quantum"
        ):
            _ = bad.block

    def test_swapped_arguments_rejected(self):
        """Swapped positional args (Float into the Qubit slot) fail fast."""

        @qkernel
        def swapped(q: qmc.Qubit, a: qmc.Float) -> qmc.Qubit:
            """Bind intentionally swapped arguments to the phased body."""
            return qmc.global_phase(_rz_body, 0.2)(a, q)

        with pytest.raises(
            TypeError, match="declared as a quantum parameter but received non-quantum"
        ):
            _ = swapped.block

    def test_scalar_into_vector_slot_rejected(self):
        """A scalar Qubit bound to a Vector[Qubit] parameter fails fast."""

        @qkernel
        def bad(q: qmc.Qubit) -> qmc.Qubit:
            """Build the intentionally invalid kernel used by this test."""
            return qmc.global_phase(_vec_body, 0.3)(q)

        with pytest.raises(TypeError, match="declared as a quantum array"):
            _ = bad.block

    def test_invalid_phase_does_not_consume_quantum_input(self):
        """Phase validation runs before the ordinary qkernel invocation."""
        from qamomile.circuit.frontend.tracer import trace

        with trace():
            q = qmc.qubit("q")
            with pytest.raises(TypeError, match="not bool"):
                qmc.global_phase(_ident, True)(q)
            assert not q._consumed

    def test_measuring_body_is_accepted(self):
        """A standalone phase can follow a measuring qkernel call."""

        @qkernel
        def measure_body(q: qmc.Qubit) -> qmc.Bit:
            """Measure the target inside the intentionally non-unitary body."""
            return qmc.measure(q)

        @qkernel
        def outer(q: qmc.Qubit) -> qmc.Bit:
            """Build the outer kernel that invokes the measuring body."""
            return qmc.global_phase(measure_body, 0.4)(q)

        assert outer.block is not None
        _assert_invoke_followed_by_phase(outer.block)

    def test_nested_reset_helper_is_accepted(self):
        """A standalone phase can follow a qkernel containing reset."""
        from qamomile.circuit.frontend.tracer import trace

        @qkernel
        def reset_helper(q: qmc.Qubit) -> qmc.Qubit:
            """Reset the helper input."""
            return qmc.reset(q)

        @qkernel
        def nested_reset(q: qmc.Qubit) -> qmc.Qubit:
            """Hide a reset behind an ordinary invoke."""
            return reset_helper(q)

        with trace() as tracer:
            q = qmc.qubit("q")
            result = qmc.global_phase(nested_reset, 0.4)(q)
            assert q._consumed
            assert not result._consumed
        _assert_invoke_followed_by_phase(tracer)

    def test_classical_output_body_is_preserved(self):
        """The wrapper preserves mixed quantum and classical qkernel outputs."""

        @qkernel
        def classical_out(q: qmc.Qubit, a: qmc.Float) -> tuple[qmc.Qubit, qmc.Float]:
            """Return a classical value from the intentionally invalid body."""
            return qmc.rz(q, a), a

        @qkernel
        def outer(
            q: qmc.Qubit,
            a: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Float]:
            """Build the outer kernel that invokes the test body."""
            return qmc.global_phase(classical_out, 0.5)(q, a)

        assert outer.block is not None
        _assert_invoke_followed_by_phase(outer.block)

    def test_classical_only_body_is_accepted(self):
        """A standalone phase does not impose quantum I/O on the wrapped call."""

        @qkernel
        def classical_only(a: qmc.Float) -> qmc.Float:
            """Build the intentionally classical-only body."""
            return a

        @qkernel
        def outer(q: qmc.Qubit, a: qmc.Float) -> qmc.Qubit:
            """Build the outer kernel that invokes the test body."""
            qmc.global_phase(classical_only, 0.6)(a)
            return qmc.h(q)

        assert outer.block is not None
        _assert_invoke_followed_by_phase(outer.block)

    @pytest.mark.parametrize("seed", [0, 42])
    def test_classical_only_helper_inside_quantum_body_executes(
        self, sdk_transpiler, seed
    ):
        """A classical helper call inside a quantum body remains valid.

        Ordinary invocation preserves helper calls used to compute gate angles.
        """

        @qkernel
        def shift(angle: qmc.Float) -> qmc.Float:
            """Shift the body angle with a purely classical helper."""
            return angle + 0.125

        @qkernel
        def body(q: qmc.Qubit, angle: qmc.Float) -> qmc.Qubit:
            """Apply the local quantum body used by this test."""
            return qmc.rz(q, shift(angle))

        @qkernel
        def circuit(angle: qmc.Float) -> qmc.Bit:
            """Build the local circuit exercised by this test."""
            q = qmc.qubit("q")
            q = qmc.x(q)
            q = qmc.global_phase(body, 0.37)(q, angle)
            return qmc.measure(q)

        theta = float(np.random.default_rng(seed).uniform(-np.pi, np.pi))
        tr = sdk_transpiler.transpiler
        executable = tr.transpile(circuit, parameters=["angle"])
        counts = _counts(
            executable.sample(
                _executor(sdk_transpiler, seed),
                shots=256,
                bindings={"angle": theta},
            ).result()
        )
        assert set(counts) == {1}, f"{sdk_transpiler.backend_name}: {counts}"

    def test_specialized_hidden_allocation_is_accepted(self):
        """The selected specialization may allocate internal qubits.

        The cached symbolic block returns before allocating anything because
        its vector width is unknown. At a concrete call site the body is
        retraced, enters the non-empty branch, and allocates an internal qubit.
        The phase wrapper records that exact ordinary qkernel call without
        imposing a separate reversibility contract.
        """
        from qamomile.circuit.frontend.handle.utils import get_size
        from qamomile.circuit.frontend.tracer import trace
        from qamomile.circuit.ir.operation.callable import InvokeOperation
        from qamomile.circuit.ir.operation.control_flow import HasNestedOps
        from qamomile.circuit.ir.operation.operation import QInitOperation

        @qkernel
        def shape_sensitive(
            qs: qmc.Vector[qmc.Qubit],
        ) -> qmc.Vector[qmc.Qubit]:
            """Build the shape-specialized body under validation."""
            try:
                size = get_size(qs)
            except ValueError:
                return qs
            if size > 0:
                ancilla = qmc.qubit("hidden")
                ancilla = qmc.h(ancilla)
            return qs

        assert not any(
            isinstance(op, QInitOperation) for op in shape_sensitive.block.operations
        )

        with trace() as tracer:
            source = qmc.qubit_array(1, "qs")
            result = qmc.global_phase(shape_sensitive, 0.2)(source)

        calls = [op for op in tracer.operations if isinstance(op, InvokeOperation)]
        assert len(calls) == 1
        assert calls[0].body is not None
        pending = [calls[0].body.operations]
        found_allocation = False
        while pending:
            for operation in pending.pop():
                found_allocation |= isinstance(operation, QInitOperation)
                if isinstance(operation, HasNestedOps):
                    pending.extend(operation.nested_op_lists())
        assert found_allocation
        assert source._consumed
        assert not result._consumed

    def test_specialized_dead_if_branch_is_not_traced(self):
        """The ordinary call follows the exact statically selected branch."""

        @qkernel
        def candidate(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
            """Allocate only on the specialization's unreachable branch."""
            if flag:
                hidden = qmc.qubit("hidden")
                hidden = qmc.h(hidden)
            return q

        @qkernel
        def outer(q: qmc.Qubit) -> qmc.Qubit:
            """Phase the allocation-free ``flag=False`` specialization."""
            return qmc.global_phase(candidate, 0.2)(q, False)

        assert outer.block is not None
        _assert_invoke_followed_by_phase(outer.block)

    def test_stateful_specialization_is_traced_once(self):
        """The wrapper does not retrace the ordinary qkernel specialization."""
        from qamomile.circuit.frontend.tracer import trace
        from qamomile.circuit.ir.operation.callable import InvokeOperation
        from qamomile.circuit.ir.operation.operation import QInitOperation

        trace_count = 0

        def allocate_only_on_second_trace() -> None:
            """Allocate a hidden qubit only if the body is retraced."""
            nonlocal trace_count
            trace_count += 1
            if trace_count == 2:
                qmc.qubit("hidden")

        @qkernel
        def stateful_body(
            qs: qmc.Vector[qmc.Qubit],
        ) -> qmc.Vector[qmc.Qubit]:
            """Build the stateful body used to detect retracing."""
            allocate_only_on_second_trace()
            return qs

        with trace() as tracer:
            qs = qmc.qubit_array(1, "qs")
            qs = qmc.global_phase(stateful_body, 0.2)(qs)

        calls = [op for op in tracer.operations if isinstance(op, InvokeOperation)]
        assert trace_count == 1
        assert len(calls) == 1
        assert calls[0].body is not None
        assert not any(
            isinstance(op, QInitOperation) for op in calls[0].body.operations
        )
        assert not qs._consumed

    def test_qubit_reordering_body_is_accepted(self):
        """The wrapper preserves the ordinary qkernel's output permutation."""

        @qkernel
        def swap_body(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Return the body qubits in intentionally reversed order."""
            a = qmc.x(a)
            return b, a

        @qkernel
        def outer(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Build the outer kernel that invokes the test body."""
            return qmc.global_phase(swap_body, 0.1)(a, b)

        assert outer.block is not None
        _assert_invoke_followed_by_phase(outer.block)

    def test_order_preserving_body_accepted(self):
        """A body returning its qubits in declaration order traces cleanly."""

        @qkernel
        def keep_order(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Return the body qubits in their declaration order."""
            a = qmc.x(a)
            return a, b

        @qkernel
        def outer(a: qmc.Qubit, b: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            """Build the outer kernel that invokes the test body."""
            return qmc.global_phase(keep_order, 0.1)(a, b)

        assert outer.block is not None

    def test_runtime_if_preserving_resource_is_accepted(self):
        """A Bit-parameterized unitary family preserves its quantum resource."""

        @qkernel
        def conditional_body(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
            """Apply one of two unitary gates selected by a classical input."""
            if flag:
                q = qmc.x(q)
            else:
                q = qmc.h(q)
            return qmc.z(q)

        @qkernel
        def outer(
            q: qmc.Qubit,
            flag: qmc.Bit,
        ) -> qmc.Qubit:
            """Wrap the conditional unitary family in a global phase."""
            return qmc.global_phase(conditional_body, 0.1)(q, flag)

        assert outer.block is not None

    def test_measurement_selected_unitary_family_is_standalone_only(
        self,
        qiskit_transpiler,
    ):
        """A measured selector is valid standalone but fails under control."""
        from qiskit.circuit import IfElseOp

        from qamomile.circuit.transpiler.errors import DependencyError

        @qkernel
        def conditional(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
            """Select one resource-preserving unitary branch."""
            if flag:
                q = qmc.x(q)
            else:
                q = qmc.h(q)
            return q

        @qkernel
        def standalone() -> qmc.Bit:
            """Apply the family after measuring its classical selector."""
            selector = qmc.qubit("selector")
            target = qmc.qubit("target")
            flag = qmc.measure(selector)
            target = qmc.global_phase(conditional, 0.2)(target, flag)
            return qmc.measure(target)

        circuit = qiskit_transpiler.transpile(standalone).get_first_circuit()

        assert any(
            isinstance(instruction.operation, IfElseOp) for instruction in circuit.data
        )
        assert float(circuit.global_phase) == pytest.approx(0.2)

        @qkernel
        def phased(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
            """Apply the same conditional family with a common phase."""
            return qmc.global_phase(conditional, 0.2)(q, flag)

        controlled = qmc.control(phased)

        @qkernel
        def controlled_circuit() -> tuple[qmc.Bit, qmc.Bit]:
            """Attempt coherent control with an unresolved measured selector."""
            selector = qmc.qubit("selector")
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            flag = qmc.measure(selector)
            control, target = controlled(control, target, flag)
            return qmc.measure(control), qmc.measure(target)

        with pytest.raises(
            DependencyError,
            match=r"ConcreteControlledU.*depends on measurement result",
        ) as exc_info:
            qiskit_transpiler.transpile(controlled_circuit)

        assert exc_info.value.quantum_op == "ConcreteControlledU"

    @pytest.mark.parametrize("condition", [True, False])
    def test_compile_time_if_preserving_resource_is_accepted(self, condition):
        """A Python-bool branch selects one resource-preserving unitary."""

        @qkernel
        def conditional_body(q: qmc.Qubit) -> qmc.Qubit:
            """Apply a statically selected gate while preserving the qubit."""
            if condition:
                q = qmc.x(q)
            else:
                q = qmc.z(q)
            return q

        @qkernel
        def outer(q: qmc.Qubit) -> qmc.Qubit:
            """Wrap the compile-time conditional body in a global phase."""
            return qmc.global_phase(conditional_body, 0.1)(q)

        assert outer.block is not None

    def test_nested_for_if_preserving_resource_is_accepted(self):
        """Nested static loops and runtime branches preserve one resource."""

        @qkernel
        def nested_body(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
            """Apply a conditional gate in each static-loop iteration."""
            for _index in qmc.range(2):
                if flag:
                    q = qmc.x(q)
                else:
                    q = qmc.z(q)
            return q

        @qkernel
        def outer(
            q: qmc.Qubit,
            flag: qmc.Bit,
        ) -> qmc.Qubit:
            """Wrap nested control flow in a global phase."""
            return qmc.global_phase(nested_body, 0.2)(q, flag)

        assert outer.block is not None

    def test_invoked_runtime_if_preserving_resource_is_accepted(self):
        """Provenance follows a conditional unitary through an ordinary call."""

        @qkernel
        def conditional_helper(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
            """Apply a conditional X while preserving the input resource."""
            if flag:
                q = qmc.x(q)
            return q

        @qkernel
        def nested_body(q: qmc.Qubit, flag: qmc.Bit) -> qmc.Qubit:
            """Delegate the conditional operation to a helper qkernel."""
            q = conditional_helper(q, flag)
            return qmc.z(q)

        @qkernel
        def outer(
            q: qmc.Qubit,
            flag: qmc.Bit,
        ) -> qmc.Qubit:
            """Wrap the delegated conditional unitary in a global phase."""
            return qmc.global_phase(nested_body, 0.3)(q, flag)

        assert outer.block is not None

    def test_runtime_if_full_reslice_preserving_resource_is_accepted(self):
        """A full view after a conditional register merge preserves its source."""

        @qkernel
        def conditional_body(
            qs: qmc.Vector[qmc.Qubit],
            flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            """Apply one conditional register gate and return a full view."""
            if flag:
                qs = qmc.x(qs)
            else:
                qs = qmc.h(qs)
            return qs[:]

        @qkernel
        def outer(
            qs: qmc.Vector[qmc.Qubit],
            flag: qmc.Bit,
        ) -> qmc.Vector[qmc.Qubit]:
            """Wrap the conditional register body in a global phase."""
            return qmc.global_phase(conditional_body, 0.4)(qs, flag)

        assert outer.block is not None

    def test_whole_vector_control_preserves_register_provenance(self):
        """Expanded scalar control ports reconstruct their returned Vector."""

        @qkernel
        def x_target(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        @qkernel
        def controlled_body(
            controls: qmc.Vector[qmc.Qubit],
            target: qmc.Qubit,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls, target = qmc.control(x_target, num_controls=2)(
                controls,
                target,
            )
            return controls, target

        @qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls = qmc.qubit_array(2, "controls")
            target = qmc.qubit("target")
            return qmc.global_phase(controlled_body, 0.2)(controls, target)

        assert circuit.block is not None

    def test_individual_controls_from_one_vector_preserve_provenance(self):
        """Scalar control elements do not create a second parent producer."""

        @qkernel
        def x_target(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        @qkernel
        def controlled_body(
            controls: qmc.Vector[qmc.Qubit],
            target: qmc.Qubit,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls[0], controls[1], target = qmc.control(
                x_target,
                num_controls=2,
            )(controls[0], controls[1], target)
            return controls, target

        @qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls = qmc.qubit_array(2, "controls")
            target = qmc.qubit("target")
            return qmc.global_phase(controlled_body, 0.2)(controls, target)

        assert circuit.block is not None

    def test_whole_vector_controlled_composite_preserves_provenance(self):
        """Controlled composite Invokes reconstruct a returned control Vector."""

        @qmc.composite_gate(name="global_phase_boxed_x")
        def boxed_x(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        @qkernel
        def controlled_body(
            controls: qmc.Vector[qmc.Qubit],
            target: qmc.Qubit,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls, target = qmc.control(boxed_x, num_controls=2)(
                controls,
                target,
            )
            return controls, target

        @qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls = qmc.qubit_array(2, "controls")
            target = qmc.qubit("target")
            return qmc.global_phase(controlled_body, 0.2)(controls, target)

        assert circuit.block is not None

    def test_symbolic_vector_control_preserves_register_provenance(self):
        """Symbolic control pools retain their array-shaped control ports."""

        @qkernel
        def x_target(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.x(q)

        @qkernel
        def controlled_body(
            controls: qmc.Vector[qmc.Qubit],
            target: qmc.Qubit,
            count: qmc.UInt,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls, target = qmc.control(x_target, num_controls=count)(
                controls,
                target,
            )
            return controls, target

        @qkernel
        def circuit(
            count: qmc.UInt,
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Qubit]:
            controls = qmc.qubit_array(count, "controls")
            target = qmc.qubit("target")
            return qmc.global_phase(controlled_body, 0.2)(
                controls,
                target,
                count,
            )

        assert circuit.block is not None

    def test_controlled_composite_interleaved_ports_preserve_provenance(self):
        """Controlled composite ports are realigned to declaration order."""

        @qmc.composite_gate(name="global_phase_interleaved_box")
        def boxed(
            first: qmc.Qubit,
            angle: qmc.Float,
            second: qmc.Qubit,
        ) -> tuple[qmc.Qubit, qmc.Qubit]:
            return qmc.rx(first, angle), qmc.x(second)

        @qkernel
        def controlled_body(
            control: qmc.Qubit,
            first: qmc.Qubit,
            angle: qmc.Float,
            second: qmc.Qubit,
        ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
            control, first, second = qmc.control(boxed)(
                control,
                first,
                angle,
                second,
            )
            return control, first, second

        @qkernel
        def circuit(
            angle: qmc.Float,
        ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
            control = qmc.qubit("control")
            first = qmc.qubit("first")
            second = qmc.qubit("second")
            return qmc.global_phase(controlled_body, 0.2)(
                control,
                first,
                angle,
                second,
            )

        assert circuit.block is not None

    def test_concrete_full_reslice_body_preserves_register(self, sdk_transpiler):
        """Treat ``qs[:]`` as the same ordered register on every backend."""

        @qkernel
        def full_reslice(
            qs: qmc.Vector[qmc.Qubit],
        ) -> qmc.Vector[qmc.Qubit]:
            """Return a full ordered view of the input register."""
            return qs[:]

        @qkernel
        def circuit() -> qmc.Vector[qmc.Bit]:
            """Build the local circuit exercised by this test."""
            qs = qmc.qubit_array(3, "qs")
            qs = qmc.global_phase(full_reslice, 0.2)(qs)
            qs[1] = qmc.x(qs[1])
            return qmc.measure(qs)

        executable = sdk_transpiler.transpiler.transpile(circuit)
        counts = _counts(
            executable.sample(_executor(sdk_transpiler), shots=128).result()
        )
        assert set(counts) == {(0, 1, 0)}, f"{sdk_transpiler.backend_name}: {counts}"


class TestGlobalPhaseOperationInvariants:
    """``GlobalPhaseOperation`` enforces its zero-qubit IR contract."""

    @staticmethod
    def _float_phase(value: float = 0.5) -> Any:
        """Create a constant scalar phase value.

        Args:
            value (float): Phase angle in radians. Defaults to 0.5.

        Returns:
            Any: Scalar ``FloatType`` IR value.
        """
        from qamomile.circuit.ir.types.primitives import FloatType
        from qamomile.circuit.ir.value import Value

        return Value(type=FloatType(), name="theta").with_const(value)

    def test_missing_phase_operand_rejected(self):
        """An operation without its single phase operand is rejected."""
        from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation

        with pytest.raises(ValueError, match="exactly one phase operand"):
            GlobalPhaseOperation(operands=[], results=[])

    def test_multiple_phase_operands_rejected(self):
        """More than one phase operand is rejected."""
        from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation

        phase = self._float_phase()
        with pytest.raises(ValueError, match="exactly one phase operand"):
            GlobalPhaseOperation(operands=[phase, phase], results=[])

    def test_non_float_phase_rejected(self):
        """A ``BitType`` phase (a ``True`` read as ``1.0`` rad) is rejected."""
        from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
        from qamomile.circuit.ir.types.primitives import BitType
        from qamomile.circuit.ir.value import Value

        bit_phase = Value(type=BitType(), name="b").with_const(True)
        with pytest.raises(ValueError, match="must be a FloatType angle"):
            GlobalPhaseOperation(operands=[bit_phase], results=[])

    def test_valid_construction_accepted(self):
        """One scalar FloatType operand and no results construct cleanly."""
        from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation

        phase = self._float_phase(0.3)
        op = GlobalPhaseOperation(operands=[phase], results=[])
        assert op.phase is phase

    def test_quantum_result_rejected(self):
        """A zero-qubit phase cannot produce a quantum result."""
        from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
        from qamomile.circuit.ir.types.primitives import QubitType
        from qamomile.circuit.ir.value import Value

        result = Value(type=QubitType(), name="q")
        with pytest.raises(ValueError, match="must not produce results"):
            GlobalPhaseOperation(
                operands=[self._float_phase()],
                results=[result],
            )


class TestGlobalPhaseEstimatorControlled:
    """The estimator counts the phase gate a controlled global phase emits."""

    def test_controlled_global_phase_counts_phase_gate(self):
        """A global phase under an accumulated control counts one phase gate.

        Reached via a controlled construct the counter descends (here a
        one-control ``InverseBlockOperation`` whose body is a global phase):
        the controlled global phase is a real ``P`` on the control, so the
        estimate must include it, not just the (empty) wrapped body.
        """
        from qamomile.circuit.estimator import estimate_resources
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
        from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
        from qamomile.circuit.ir.types.primitives import FloatType, QubitType
        from qamomile.circuit.ir.value import Value

        body_q = Value(type=QubitType(), name="bq")
        gp = GlobalPhaseOperation(
            operands=[Value(type=FloatType(), name="a").with_const(0.7)],
            results=[],
        )
        gp_block = Block(operations=[gp], input_values=[body_q])
        control = Value(type=QubitType(), name="c")
        target = Value(type=QubitType(), name="q")
        inv = InverseBlockOperation(
            operands=[control, target],
            results=[control.next_version(), target.next_version()],
            num_control_qubits=1,
            num_target_qubits=1,
            source_block=gp_block,
            implementation_block=gp_block,
        )
        estimate = estimate_resources(Block(operations=[inv]))
        # The global phase's body is empty, so the only gate is the controlled
        # phase P on the single control: 1 single-qubit rotation gate.
        assert estimate.gates.total == 1
        assert estimate.gates.single_qubit == 1
        assert estimate.gates.rotation == 1

    def test_controlled_composite_global_phase_counts_phase_gate(self):
        """A global phase in a controlled composite's body counts its phase gate.

        The body-derived callable estimator must propagate the controlled
        transform into the composite implementation; otherwise the phase would
        be counted as an unobservable standalone operation.
        """
        from qamomile.circuit.estimator import estimate_resources

        @qmc.composite_gate(name="controlled_gp_composite")
        def phase_box(target: qmc.Qubit) -> qmc.Qubit:
            """Apply a constant global phase."""
            return qmc.global_phase(_ident, 0.7)(target)

        @qkernel
        def controlled_phase_box() -> tuple[qmc.Qubit, qmc.Qubit]:
            """Control the composite global phase."""
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            return qmc.control(phase_box)(control, target)

        estimate = estimate_resources(controlled_phase_box.block)
        assert estimate.gates.total == 1
        assert estimate.gates.single_qubit == 1
        assert estimate.gates.rotation == 1

    def test_standalone_global_phase_counts_no_gate(self):
        """A standalone (uncontrolled) global phase contributes no gate."""
        from qamomile.circuit.estimator import estimate_resources

        @qkernel
        def standalone(angle: qmc.Float) -> qmc.Qubit:
            """Apply a standalone global phase to one qubit."""
            q = qmc.qubit("q")
            return qmc.global_phase(_ident, angle)(q)

        est = estimate_resources(standalone.block)
        assert est.gates.total == 0


class TestGlobalPhaseOperandVisibility:
    """The flat phase is an ordinary input seen by every generic pass."""

    def test_phase_is_the_only_input_value(self):
        """``all_input_values`` exposes the sole phase operand directly."""
        from qamomile.circuit.ir.operation.global_phase import GlobalPhaseOperation
        from qamomile.circuit.ir.types.primitives import FloatType
        from qamomile.circuit.ir.value import Value

        phase = Value(type=FloatType(), name="phase")
        op = GlobalPhaseOperation(operands=[phase], results=[])
        assert op.all_input_values() == [phase]
