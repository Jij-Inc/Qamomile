"""Execute shared SELECT semantics through native HUGR and Selene."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.transpiler import EmitError, ShotBased
from qamomile.hugr import HugrTranspiler, SeleneExecutionOptions
from tests.circuit import test_select_global_phase as phase_suite
from tests.circuit.test_select import (
    TestSelectCrossEngine as _SelectCrossEngine,
    _identity,
    _parameterized_identity,
    _ry,
)
from tests.hugr.test_shape_transforms import _forward_angles, _last_angle

pytestmark = pytest.mark.hugr


class _SeleneSelectTranspiler:
    """Adapt shared SDK tests to the native HUGR executable facade.

    Args:
        build_dir (Path): Native Selene build directory for this test.
    """

    def __init__(self, build_dir: Path):
        """Initialize the native compiler and simulator configuration.

        Args:
            build_dir (Path): Directory containing native test artifacts.
        """
        self.native = HugrTranspiler()
        self.options = SeleneExecutionOptions(seed=31, build_dir=build_dir)

    def transpile(self, kernel, bindings=None, parameters=None):
        """Compile a shared-suite kernel through the native executable facade.

        Args:
            kernel (qmc.QKernel): Kernel supplied by the shared suite.
            bindings (dict[str, Any] | None): Compile-time input values.
            parameters (list[str] | None): Retained runtime argument names.

        Returns:
            HugrExecutable: Validator-clean native executable.
        """
        return self.native.transpile(kernel, bindings, parameters)

    def executor(self):
        """Create the real Selene executor used by shared semantic tests.

        Returns:
            HugrExecutor: Seeded local native simulator executor.
        """
        return self.native.executor(options=self.options)


@pytest.fixture
def sdk_transpiler(tmp_path):
    """Provide native compilation and execution to inherited semantic tests.

    Args:
        tmp_path (Path): Isolated directory supplied by pytest.

    Returns:
        SimpleNamespace: HUGR engine identity and executable adapter.
    """
    return SimpleNamespace(
        engine_name="hugr", transpiler=_SeleneSelectTranspiler(tmp_path)
    )


class TestSelectOnSelene(_SelectCrossEngine):
    """Reuse basis, shape, transform, broadcast, and runtime-control coverage."""

    def test_runtime_parameter_reaches_expectation_value(self, sdk_transpiler):
        """Use explicit shot accuracy for the shared parameterized RY observable."""

        @qmc.qkernel
        def circuit(angle: qmc.Float, observable: qmc.Observable) -> qmc.Float:
            """Expose a runtime SELECT rotation through a target observable.

            Args:
                angle (qmc.Float): Runtime angle in radians.
                observable (qmc.Observable): Compile-time measured Hamiltonian.

            Returns:
                qmc.Float: Expectation of the selected target state.
            """
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit("target")
            index, target = qmc.select([_parameterized_identity, _ry])(
                index, target, theta=angle
            )
            return qmc.expval(target, observable)

        executable = sdk_transpiler.transpiler.transpile(
            circuit, bindings={"observable": qm_o.Z(0)}, parameters=["angle"]
        )
        result = executable.run(
            sdk_transpiler.transpiler.executor(),
            bindings={"angle": 0.73},
            estimation=ShotBased(8192),
        ).result()
        assert result == pytest.approx(math.cos(0.73), abs=0.04)


@pytest.mark.parametrize(
    "shared_test",
    [
        phase_suite.test_identity_case_phase_becomes_relative_phase,
        phase_suite.test_symbolic_wide_index_preserves_identity_case_phase,
        phase_suite.test_eight_case_phase_select_uses_lsb_zero,
        phase_suite.test_outer_control_preserves_select_relative_phase,
        phase_suite.test_inverse_and_nested_select_preserve_phase,
        phase_suite.test_inverse_cancels_phased_pauli_case,
        phase_suite.test_runtime_phase_parameter_is_forwarded,
        phase_suite.test_eight_case_runtime_phase_uses_lsb_zero,
        phase_suite.test_loop_variable_phase_forces_safe_select_unrolling,
    ],
    ids=lambda test: test.__name__,
)
def test_shared_coherent_phase_semantics(sdk_transpiler, shared_test):
    """Run the existing deterministic phase-sensitive SELECT regressions."""
    shared_test(sdk_transpiler)


@pytest.mark.parametrize("target_size", [1, 2, 3])
@pytest.mark.parametrize("vector_case", [False, True])
def test_broadcast_and_vector_phase_multiplicity(
    sdk_transpiler, target_size, vector_case
):
    """A scalar case contributes one phase per element, a vector case only one."""
    cases = (
        [phase_suite._identity_vector, phase_suite._phase_half_pi_identity_vector]
        if vector_case
        else [phase_suite._identity, phase_suite._phase_half_pi_identity]
    )
    multiplicity = 1 if vector_case else target_size

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Interfere the selected phase in an exact X or Y eigenbasis.

        Returns:
            qmc.Bit: Deterministic signed phase measurement.
        """
        index = qmc.h(qmc.qubit("index"))
        targets = qmc.qubit_array(target_size, "targets")
        index, targets = qmc.select(cases)(index, targets)
        if multiplicity % 2:
            index = qmc.sdg(index)
        return qmc.measure(qmc.h(index))

    executable = sdk_transpiler.transpiler.transpile(circuit)
    result = executable.sample(sdk_transpiler.transpiler.executor(), shots=16).result()
    assert result.results == [(multiplicity in (2, 3), 16)]


@qmc.qkernel
def _runtime_array_select(values: qmc.Vector[qmc.Float]) -> tuple[qmc.Bit, qmc.Bit]:
    """Interfere a SELECT with transitive and direct runtime-array cases.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime angles with a compile-time shape.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Index and interfered target measurements.
    """
    index = qmc.qubit("index")
    target = qmc.h(qmc.qubit("target"))
    index, target = qmc.select([_forward_angles, _last_angle])(index, target, values)
    return qmc.measure(index), qmc.measure(qmc.h(target))


def test_runtime_array_shape_and_values_remain_independent(sdk_transpiler):
    """SELECT case specialization retains a shared runtime-array parameter."""
    executable = sdk_transpiler.transpiler.native.transpile(
        _runtime_array_select, parameters=["values"], parameter_shapes={"values": (2,)}
    )
    original = executable.artifact.to_bytes()
    executor = sdk_transpiler.transpiler.executor()
    for angle, expected in [(0.0, False), (math.pi, True)]:
        result = executable.sample(
            executor, shots=16, bindings={"values": [0.0, angle]}
        ).result()
        assert result.results == [((False, expected), 16)]
        assert executable.artifact.to_bytes() == original


@qmc.qkernel
def _hadamard(q: qmc.Qubit) -> qmc.Qubit:
    """Apply H with the shared scalar-case signature.

    Args:
        q (qmc.Qubit): Scalar target qubit.

    Returns:
        qmc.Qubit: Target after the Hadamard gate.
    """
    return qmc.h(q)


def test_selected_hadamard_and_rotation_are_coherent(sdk_transpiler):
    """Controlled H and runtime RY retain amplitudes and relative phase."""

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
        """Uncompute a selected Hadamard using a runtime Y rotation.

        Args:
            theta (qmc.Float): Runtime uncomputation angle in radians.

        Returns:
            tuple[qmc.Bit, qmc.Bit]: Interfered index and target measurements.
        """
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        index, target = qmc.select([_identity, _hadamard])(index, target)
        index, target = qmc.select([_parameterized_identity, _ry])(
            index, target, theta=theta
        )
        return qmc.measure(qmc.h(index)), qmc.measure(target)

    executable = sdk_transpiler.transpiler.transpile(circuit, parameters=["theta"])
    result = executable.sample(
        sdk_transpiler.transpiler.executor(), shots=16, bindings={"theta": -math.pi / 2}
    ).result()
    assert result.results == [((False, False), 16)]


@pytest.mark.parametrize("captured_stop", [False, True])
def test_phase_in_nested_runtime_regions(sdk_transpiler, captured_stop):
    """Nested SELECT preserves phase; a consumed loop capture is rejected."""
    if captured_stop:
        with pytest.raises(EmitError, match="destructively consumed"):
            phase_suite.test_phase_select_survives_mixed_runtime_control_flow(
                sdk_transpiler
            )
        return

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Allocate the stopping measurement locally in each native iteration.

        Returns:
            qmc.Bit: Deterministic interference from the selected phase.
        """
        condition = qmc.measure(qmc.x(qmc.qubit("condition")))
        branch = qmc.measure(qmc.x(qmc.qubit("branch")))
        index = qmc.h(qmc.qubit("index"))
        target = qmc.qubit("target")
        while condition:
            if branch:
                for _iteration in qmc.range(1):
                    index, target = qmc.select(
                        [_identity, phase_suite._phase_pi_identity]
                    )(index, target)
            condition = qmc.measure(qmc.qubit("stop"))
        return qmc.measure(qmc.h(index))

    executable = sdk_transpiler.transpiler.transpile(circuit)
    assert executable.sample(
        sdk_transpiler.transpiler.executor(), shots=16
    ).result().results == [(True, 16)]


@qmc.qkernel
def _controlled_phase_case(
    index: qmc.Qubit, target: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Apply a positive relative phase through a nested controlled call.

    Args:
        index (qmc.Qubit): Local control for the case's phase.
        target (qmc.Qubit): Unchanged target of the phased identity.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Controlled phase operands after execution.
    """
    return qmc.control(phase_suite._phase_half_pi_identity)(index, target)


@qmc.qkernel
def _inverse_controlled_phase_case(
    index: qmc.Qubit, target: qmc.Qubit
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Invert a nested controlled case while preserving its local controls.

    Args:
        index (qmc.Qubit): Local coherent control.
        target (qmc.Qubit): Unchanged target of the phased identity.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Operands after the adjoint controlled phase.
    """
    return qmc.inverse(_controlled_phase_case)(index, target)


@pytest.mark.parametrize("inverse_case", [False, True])
def test_select_composes_with_nested_control_and_inverse(sdk_transpiler, inverse_case):
    """SELECT retains signed phase in nested controlled and adjoint call cases."""
    selected = (
        _inverse_controlled_phase_case if inverse_case else _controlled_phase_case
    )

    @qmc.qkernel
    def circuit() -> qmc.Bit:
        """Read the signed phase of the selected nested-control unitary.

        Returns:
            qmc.Bit: Zero after the correct signed phase correction.
        """
        outer = qmc.h(qmc.qubit("outer"))
        inner = qmc.x(qmc.qubit("inner"))
        target = qmc.qubit("target")
        outer, inner, target = qmc.select([phase_suite._identity_pair, selected])(
            outer, inner, target
        )
        if inverse_case:
            outer = qmc.s(outer)
        else:
            outer = qmc.sdg(outer)
        return qmc.measure(qmc.h(outer))

    executable = sdk_transpiler.transpiler.transpile(circuit)
    assert executable.sample(
        sdk_transpiler.transpiler.executor(), shots=16
    ).result().results == [(False, 16)]


@qmc.qkernel
def _mixed_loop_case(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Alternate two gates while reusing the traced scalar loop input.

    Args:
        q (qmc.Qubit): Target state.
        theta (qmc.Float): Runtime base rotation in radians.

    Returns:
        qmc.Qubit: State after three distinct rotations and basis changes.
    """
    for i in qmc.range(3):
        q = qmc.h(q)
        q = qmc.rx(q, theta * (i + 1))
    return q


@qmc.qkernel
def _mixed_unrolled_case(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Express the reference unitary without reusing loop-body identities.

    Args:
        q (qmc.Qubit): Target state.
        theta (qmc.Float): Runtime base rotation in radians.

    Returns:
        qmc.Qubit: State after the explicitly ordered reference gates.
    """
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    q = qmc.h(q)
    q = qmc.rx(q, theta * 2.0)
    q = qmc.h(q)
    return qmc.rx(q, theta * 3.0)


@qmc.qkernel
def _mixed_loop_select(
    index: qmc.Qubit, target: qmc.Qubit, theta: qmc.Float
) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Select a scalar loop containing multiple primitive gates.

    Args:
        index (qmc.Qubit): Coherent selector.
        target (qmc.Qubit): Case target state.
        theta (qmc.Float): Runtime base rotation in radians.

    Returns:
        tuple[qmc.Qubit, qmc.Qubit]: Selector and updated target.
    """
    return qmc.select([_parameterized_identity, _mixed_loop_case])(index, target, theta)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("select", [False, True], ids=["control", "select"])
def test_mixed_gate_loop_matches_unrolled_reference(sdk_transpiler, inverse, select):
    """Forward and inverse coherent loops match independently expanded gates."""
    if select:
        loop = qmc.inverse(_mixed_loop_select) if inverse else _mixed_loop_select
    else:
        loop = qmc.control(
            qmc.inverse(_mixed_loop_case) if inverse else _mixed_loop_case
        )
    reference = qmc.control(
        _mixed_unrolled_case if inverse else qmc.inverse(_mixed_unrolled_case)
    )

    @qmc.qkernel
    def circuit(theta: qmc.Float) -> tuple[qmc.Bit, qmc.Bit]:
        """Cancel a loop against an explicit reference and interfere both qubits.

        Args:
            theta (qmc.Float): Runtime base rotation in radians.

        Returns:
            tuple[qmc.Bit, qmc.Bit]: Deterministic zeros for the matching unitary.
        """
        index = qmc.h(qmc.qubit("index"))
        target = qmc.h(qmc.qubit("target"))
        index, target = loop(index, target, theta)
        index, target = reference(index, target, theta)
        return qmc.measure(qmc.h(index)), qmc.measure(qmc.h(target))

    executable = sdk_transpiler.transpiler.transpile(circuit, parameters=["theta"])
    executor = sdk_transpiler.transpiler.executor()
    for theta in (0.73, -1.37):
        assert executable.sample(
            executor, shots=32, bindings={"theta": theta}
        ).result().results == [((False, False), 32)]
