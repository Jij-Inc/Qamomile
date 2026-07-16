"""Execution and validation coverage for ``qmc.select``."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit import qkernel
from qamomile.circuit.frontend.handle import (
    Bit,
    Float,
    Observable,
    Qubit,
    UInt,
    Vector,
)
from qamomile.circuit.frontend.tracer import trace
from qamomile.circuit.ir.types.primitives import QubitType
from qamomile.circuit.ir.value import ArrayValue, Value
from qamomile.circuit.transpiler.errors import QubitConsumedError


def _make_qubit(name: str) -> Qubit:
    """Create a standalone qubit handle for frontend transaction tests.

    Args:
        name (str): Diagnostic value name.

    Returns:
        Qubit: Fresh unconsumed qubit handle.
    """
    return Qubit(value=Value(type=QubitType(), name=name))


@qkernel
def _identity(q: Qubit) -> Qubit:
    """Return one target unchanged."""
    return q


@qkernel
def _x(q: Qubit) -> Qubit:
    """Apply X to one target."""
    return qmc.x(q)


@qkernel
def _parameterized_identity(q: Qubit, theta: Float) -> Qubit:
    """Accept a shared parameter without changing the target."""
    _ = theta
    return q


@qkernel
def _ry(q: Qubit, theta: Float) -> Qubit:
    """Apply a shared RY parameter."""
    return qmc.ry(q, theta)


@qkernel
def _conditional_x(q: Qubit, flag: UInt) -> Qubit:
    """Apply X when a compile-time flag is true."""
    if flag:
        q = qmc.x(q)
    return q


@qkernel
def _conditional_z(q: Qubit, flag: UInt) -> Qubit:
    """Apply Z when a compile-time flag is true."""
    if flag:
        q = qmc.z(q)
    return q


@qkernel
def _loop_identity(q: Qubit, repetitions: UInt) -> Qubit:
    """Keep a target unchanged through case-local loop and branch nodes."""
    for iteration in qmc.range(repetitions):
        if iteration < 0:
            q = qmc.x(q)
    return q


@qkernel
def _loop_x_on_second_iteration(q: Qubit, repetitions: UInt) -> Qubit:
    """Apply X on the second iteration of a case-local loop."""
    for iteration in qmc.range(repetitions):
        if iteration == 1:
            q = qmc.x(q)
    return q


@qmc.composite_gate(name="select_test_x")
def _composite_x(q: Qubit) -> Qubit:
    """Apply X through a boxed composite callable."""
    return qmc.x(q)


@qkernel
def _delegating_composite_x(q: Qubit) -> Qubit:
    """Invoke a boxed composite from a SELECT case."""
    return _composite_x(q)


@qkernel
def _identity_vector(qs: Vector[Qubit]) -> Vector[Qubit]:
    """Return a vector target unchanged."""
    return qs


@qkernel
def _identity_pair(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
    """Return two targets in their original positions."""
    return q0, q1


@qkernel
def _reverse_pair(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
    """Return target handles in the wrong positional order."""
    return q1, q0


@qkernel
def _swap_pair(q0: Qubit, q1: Qubit) -> tuple[Qubit, Qubit]:
    """Apply an explicit physical SWAP to two targets."""
    return qmc.swap(q0, q1)


@qkernel
def _identity_scalar_vector(
    scalar: Qubit,
    vector: Vector[Qubit],
) -> tuple[Qubit, Vector[Qubit]]:
    """Return mixed scalar and vector targets unchanged."""
    return scalar, vector


@qkernel
def _x_scalar_vector(
    scalar: Qubit,
    vector: Vector[Qubit],
) -> tuple[Qubit, Vector[Qubit]]:
    """Apply X to the scalar target and preserve the vector target."""
    return qmc.x(scalar), vector


@qkernel
def _select_x_on_one(index: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """Apply X only when a one-qubit index reads one."""
    return qmc.select([_identity, _x])(index, target)


@qkernel
def _select_x_on_zero(index: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """Apply X only when a one-qubit index reads zero."""
    return qmc.select([_x, _identity])(index, target)


@qkernel
def _symbolic_select_x(
    index: Vector[Qubit],
    target: Qubit,
    width: UInt,
) -> tuple[Vector[Qubit], Qubit]:
    """Apply identity-or-X SELECT with a symbolic index width."""
    return qmc.select(
        [_identity, _x],
        num_index_qubits=width,
    )(index, target)


def _executor(case: Any, *, runtime_control: bool = False) -> Any:
    """Return a simulator executor for a cross-backend test case.

    Args:
        case (Any): Backend fixture containing a transpiler and backend name.
        runtime_control (bool): Whether Qiskit needs a dynamic-control capable
            simulator. Defaults to ``False``.

    Returns:
        Any: Executor for the selected SDK backend.
    """
    if case.backend_name == "qiskit":
        if runtime_control:
            from qiskit_aer import AerSimulator

            return case.transpiler.executor(backend=AerSimulator(method="statevector"))
        from qiskit.providers.basic_provider import BasicSimulator

        return case.transpiler.executor(backend=BasicSimulator())
    return case.transpiler.executor()


def _sample_outcomes(
    case: Any,
    kernel: Any,
    *,
    runtime_control: bool = False,
) -> set[Any]:
    """Transpile a deterministic kernel and return its sampled outcomes.

    Args:
        case (Any): Backend fixture containing a transpiler and backend name.
        kernel (Any): Deterministic qkernel to transpile and execute.
        runtime_control (bool): Whether the kernel uses dynamic control flow.
            Defaults to ``False``.

    Returns:
        set[Any]: Distinct classical outcomes observed in the sample.
    """
    executable = case.transpiler.transpile(kernel)
    job = executable.sample(
        _executor(case, runtime_control=runtime_control),
        shots=128,
    )
    return {bits for bits, _ in job.result().results}


class TestSelectCrossBackend:
    """Execute SELECT through every supported SDK backend."""

    @pytest.mark.parametrize("index_value", [0, 1])
    def test_two_case_basis_selection(
        self,
        sdk_transpiler: Any,
        index_value: int,
    ) -> None:
        """A two-case SELECT applies the case matching its index."""

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit("index")
            if index_value:
                index = qmc.x(index)
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _x])(index, target)
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {index_value}

    @pytest.mark.parametrize("index_value", [0, 1, 2, 3])
    def test_four_case_index_is_lsb_first(
        self,
        sdk_transpiler: Any,
        index_value: int,
    ) -> None:
        """Index qubit zero represents bit zero of a four-case index."""
        bit_0 = index_value & 1
        bit_1 = (index_value >> 1) & 1

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit_array(2, "index")
            if bit_0:
                index[0] = qmc.x(index[0])
            if bit_1:
                index[1] = qmc.x(index[1])
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _identity, _x, _identity])(
                index,
                target,
            )
            return qmc.measure(target)

        expected = 1 if index_value == 2 else 0
        assert _sample_outcomes(sdk_transpiler, circuit) == {expected}

    @pytest.mark.parametrize("index_value", [0, 1, 2, 3])
    def test_unassigned_index_value_is_identity(
        self,
        sdk_transpiler: Any,
        index_value: int,
    ) -> None:
        """A non-power-of-two case list leaves unused basis states unchanged."""
        bit_0 = index_value & 1
        bit_1 = (index_value >> 1) & 1

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit_array(2, "index")
            if bit_0:
                index[0] = qmc.x(index[0])
            if bit_1:
                index[1] = qmc.x(index[1])
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _identity, _x])(
                index,
                target,
            )
            return qmc.measure(target)

        expected = 1 if index_value == 2 else 0
        assert _sample_outcomes(sdk_transpiler, circuit) == {expected}

    @pytest.mark.parametrize(
        ("index_value", "expected"),
        [(1, 1), (7, 0)],
    )
    def test_explicit_wide_index_leaves_extra_states_as_identity(
        self,
        sdk_transpiler: Any,
        index_value: int,
        expected: int,
    ) -> None:
        """An explicit over-wide register maps only supplied case addresses."""
        bits = tuple((index_value >> offset) & 1 for offset in range(3))

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit_array(3, "index")
            if bits[0]:
                index[0] = qmc.x(index[0])
            if bits[1]:
                index[1] = qmc.x(index[1])
            if bits[2]:
                index[2] = qmc.x(index[2])
            target = qmc.qubit("target")
            index, target = qmc.select(
                [_identity, _x],
                num_index_qubits=3,
            )(index, target)
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {expected}

    @pytest.mark.parametrize(
        ("index_value", "expected"),
        [
            (1, (1, (0, 0))),
            (5, (0, (0, 0))),
        ],
    )
    def test_symbolic_width_flattens_mixed_index_and_target_groups(
        self,
        sdk_transpiler: Any,
        index_value: int,
        expected: tuple[int, tuple[int, int]],
    ) -> None:
        """A bound UInt width preserves grouped operands and LSB-first order."""
        bits = tuple((index_value >> offset) & 1 for offset in range(3))

        @qkernel
        def circuit(width: UInt) -> tuple[Bit, Vector[Bit]]:
            index_scalar = qmc.qubit("index_scalar")
            index_array = qmc.qubit_array(2, "index_array")
            if bits[0]:
                index_scalar = qmc.x(index_scalar)
            if bits[1]:
                index_array[0] = qmc.x(index_array[0])
            if bits[2]:
                index_array[1] = qmc.x(index_array[1])
            target_scalar = qmc.qubit("target_scalar")
            target_array = qmc.qubit_array(2, "target_array")
            index_scalar, index_array, target_scalar, target_array = qmc.select(
                [_identity_scalar_vector, _x_scalar_vector],
                num_index_qubits=width,
            )(
                index_scalar,
                index_array,
                target_scalar,
                target_array,
            )
            return qmc.measure(target_scalar), qmc.measure(target_array)

        executable = sdk_transpiler.transpiler.transpile(
            circuit,
            bindings={"width": 3},
        )
        result = executable.sample(_executor(sdk_transpiler), shots=128).result()
        assert {bits for bits, _ in result.results} == {expected}

    def test_symbolic_width_requires_exact_flattened_index_size(
        self,
        sdk_transpiler: Any,
    ) -> None:
        """Lowering rejects a bound width that would split an index array."""
        from qamomile.circuit.transpiler.errors import EmitError

        @qkernel
        def circuit(width: UInt) -> Bit:
            index_scalar = qmc.qubit("index_scalar")
            index_array = qmc.qubit_array(2, "index_array")
            target = qmc.qubit("target")
            index_scalar, index_array, target = qmc.select(
                [_identity, _x],
                num_index_qubits=width,
            )(index_scalar, index_array, target)
            return qmc.measure(target)

        with pytest.raises(
            EmitError,
            match=r"expanded to 3 qubit\(s\).*resolves to 2",
        ):
            sdk_transpiler.transpiler.transpile(
                circuit,
                bindings={"width": 2},
            )

    def test_inverse_preserves_symbolic_select_width(
        self,
        sdk_transpiler: Any,
    ) -> None:
        """A symbolic-width SELECT followed by its inverse cancels exactly."""

        @qkernel
        def circuit(width: UInt) -> Bit:
            index = qmc.qubit_array(width, "index")
            index[0] = qmc.x(index[0])
            target = qmc.qubit("target")
            index, target = _symbolic_select_x(index, target, width)
            index, target = qmc.inverse(_symbolic_select_x)(index, target, width)
            return qmc.measure(target)

        executable = sdk_transpiler.transpiler.transpile(
            circuit,
            bindings={"width": 3},
        )
        result = executable.sample(_executor(sdk_transpiler), shots=128).result()
        assert {bits for bits, _ in result.results} == {0}

    @pytest.mark.parametrize("outer_value", [0, 1])
    @pytest.mark.parametrize("index_value", [0, 1])
    def test_outer_control_composes_with_zero_index_case(
        self,
        sdk_transpiler: Any,
        outer_value: int,
        index_value: int,
    ) -> None:
        """An outer control composes with an internal zero-pattern bracket."""

        @qkernel
        def circuit() -> Bit:
            outer = qmc.qubit("outer")
            if outer_value:
                outer = qmc.x(outer)
            index = qmc.qubit("index")
            if index_value:
                index = qmc.x(index)
            target = qmc.qubit("target")
            outer, index, target = qmc.control(_select_x_on_zero)(
                outer,
                index,
                target,
            )
            return qmc.measure(target)

        expected = 1 if outer_value and not index_value else 0
        assert _sample_outcomes(sdk_transpiler, circuit) == {expected}

    def test_scalar_case_broadcasts_over_vector_target(
        self,
        sdk_transpiler: Any,
    ) -> None:
        """Built-in QKernelLike cases broadcast over a vector target."""

        @qkernel
        def circuit() -> Vector[Bit]:
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit_array(3, "target")
            index, target = qmc.select([qmc.z, qmc.x])(index, target)
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {(1, 1, 1)}

    def test_multi_qubit_target_case(self, sdk_transpiler: Any) -> None:
        """A selected two-target unitary preserves every returned mapping."""

        @qkernel
        def circuit() -> tuple[Bit, Bit]:
            index = qmc.x(qmc.qubit("index"))
            first = qmc.x(qmc.qubit("first"))
            second = qmc.qubit("second")
            index, first, second = qmc.select([_identity_pair, _swap_pair])(
                index,
                first,
                second,
            )
            return qmc.measure(first), qmc.measure(second)

        assert _sample_outcomes(sdk_transpiler, circuit) == {(0, 1)}

    def test_vector_view_index_result_elements_remain_mapped(
        self,
        sdk_transpiler: Any,
    ) -> None:
        """A returned VectorView index keeps its LSB-first element mapping."""

        @qkernel
        def circuit() -> tuple[Bit, Bit, Bit]:
            register = qmc.qubit_array(4, "register")
            register[2] = qmc.x(register[2])
            index = register[1:3]
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _identity, _x, _identity])(
                index,
                target,
            )
            index[0] = qmc.x(index[0])
            return (
                qmc.measure(index[0]),
                qmc.measure(index[1]),
                qmc.measure(target),
            )

        assert _sample_outcomes(sdk_transpiler, circuit) == {(1, 1, 1)}

    def test_case_compile_time_if_is_lowered(self, sdk_transpiler: Any) -> None:
        """Compile-time control flow inside a case is lowered in its own scope."""

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_conditional_x, _conditional_z])(
                index,
                target,
                flag=1,
            )
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {1}

    def test_case_local_for_and_if_are_lowered(
        self,
        sdk_transpiler: Any,
    ) -> None:
        """Case-local loops and branches lower before index control is added."""

        @qkernel
        def circuit() -> Bit:
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit("target")
            index, target = qmc.select([_loop_identity, _loop_x_on_second_iteration])(
                index,
                target,
                repetitions=2,
            )
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {1}

    @pytest.mark.parametrize("index_value", [0, 1])
    def test_boxed_composite_remains_index_controlled(
        self,
        sdk_transpiler: Any,
        index_value: int,
    ) -> None:
        """A composite invocation in a case never escapes the index control."""

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit("index")
            if index_value:
                index = qmc.x(index)
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _delegating_composite_x])(
                index,
                target,
            )
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {index_value}

    def test_runtime_parameter_reaches_case(self, sdk_transpiler: Any) -> None:
        """A shared case argument remains a backend runtime parameter."""

        @qkernel
        def circuit(theta: Float) -> Bit:
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit("target")
            index, target = qmc.select([_parameterized_identity, _ry])(
                index,
                target,
                theta=theta,
            )
            return qmc.measure(target)

        executable = sdk_transpiler.transpiler.transpile(
            circuit,
            parameters=["theta"],
        )
        result = executable.sample(
            _executor(sdk_transpiler),
            shots=128,
            bindings={"theta": np.pi},
        ).result()
        assert {bits for bits, _ in result.results} == {1}

    def test_runtime_parameter_reaches_expectation_value(
        self,
        sdk_transpiler: Any,
    ) -> None:
        """SELECT runtime parameters execute through the expval path."""
        theta = 0.73

        @qkernel
        def circuit(angle: Float, observable: Observable) -> Float:
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit("target")
            index, target = qmc.select([_parameterized_identity, _ry])(
                index,
                target,
                theta=angle,
            )
            return qmc.expval(target, observable)

        executable = sdk_transpiler.transpiler.transpile(
            circuit,
            bindings={"observable": qm_o.Z(0)},
            parameters=["angle"],
        )
        value = executable.run(
            _executor(sdk_transpiler),
            bindings={"angle": theta},
        ).result()
        assert value == pytest.approx(np.cos(theta), abs=1e-6)

    def test_select_inside_for(self, sdk_transpiler: Any) -> None:
        """SELECT values remain mapped through a statically unrolled loop."""

        @qkernel
        def circuit() -> Vector[Bit]:
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit_array(2, "target")
            for i in qmc.range(2):
                index, target[i] = qmc.select([_identity, _x])(
                    index,
                    target[i],
                )
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {(1, 1)}

    def test_symbolic_width_uses_one_trip_loop_value(
        self,
        sdk_transpiler: Any,
    ) -> None:
        """One-trip loop lowering substitutes SELECT's structural width."""

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit_array(3, "index")
            index[0] = qmc.x(index[0])
            target = qmc.qubit("target")
            for width in qmc.range(3, 4):
                index, target = qmc.select(
                    [_identity, _x],
                    num_index_qubits=width,
                )(index, target)
            return qmc.measure(target)

        assert _sample_outcomes(sdk_transpiler, circuit) == {1}

    def test_select_inside_runtime_if(self, sdk_transpiler: Any) -> None:
        """SELECT results merge from a measurement-backed conditional."""
        if sdk_transpiler.backend_name == "quri_parts":
            pytest.skip("QURI Parts has no dynamic if primitive")

        @qkernel
        def circuit() -> Bit:
            guard = qmc.measure(qmc.x(qmc.qubit("guard")))
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit("target")
            if guard:
                index, target = qmc.select([_identity, _x])(index, target)
            return qmc.measure(target)

        assert _sample_outcomes(
            sdk_transpiler,
            circuit,
            runtime_control=True,
        ) == {1}

    def test_select_inside_runtime_while(self, sdk_transpiler: Any) -> None:
        """SELECT results remain loop-carried through a dynamic while."""
        if sdk_transpiler.backend_name == "quri_parts":
            pytest.skip("QURI Parts has no dynamic while primitive")

        @qkernel
        def circuit() -> Bit:
            condition = qmc.measure(qmc.x(qmc.qubit("guard")))
            index = qmc.x(qmc.qubit("index"))
            target = qmc.qubit("target")
            while condition:
                index, target = qmc.select([_identity, _x])(index, target)
                condition = qmc.measure(qmc.qubit("stop"))
            return qmc.measure(target)

        assert _sample_outcomes(
            sdk_transpiler,
            circuit,
            runtime_control=True,
        ) == {1}


class TestSelectValidation:
    """Reject SELECT programs that do not describe compatible unitaries."""

    def test_single_case_is_rejected(self) -> None:
        """A one-case operation is not a multiplexer."""
        with pytest.raises(ValueError, match="at least 2 cases"):
            qmc.select([_identity])

    def test_mismatched_signatures_are_rejected(self) -> None:
        """Every case receives exactly the same argument list."""
        with pytest.raises(ValueError, match="same parameter signature"):
            qmc.select([_identity, _ry])

    def test_mismatched_defaults_are_rejected(self) -> None:
        """Case defaults cannot silently inherit the first case's value."""

        @qkernel
        def first(q: Qubit, theta: Float = 0.1) -> Qubit:
            return qmc.rx(q, theta)

        @qkernel
        def second(q: Qubit, theta: Float = 0.9) -> Qubit:
            return qmc.rx(q, theta)

        with pytest.raises(ValueError, match="same parameter signature"):
            qmc.select([first, second])

    @pytest.mark.parametrize(("case_count", "width"), [(2, 1), (3, 2), (8, 3)])
    def test_num_index_qubits(self, case_count: int, width: int) -> None:
        """Index width is the ceiling of the case-count base-two logarithm."""
        assert qmc.select([_identity] * case_count).num_index_qubits == width

    def test_explicit_wider_index_width_is_preserved(self) -> None:
        """A concrete width may exceed the minimum required by the cases."""
        selector = qmc.select([_identity, _x], num_index_qubits=3)
        assert selector.num_index_qubits == 3

    @pytest.mark.parametrize("width", [True, 2.0, "2"])
    def test_invalid_frontend_index_width_type_is_rejected(self, width: Any) -> None:
        """The public API accepts only int, UInt, or None index widths."""
        with pytest.raises(TypeError, match="int, UInt, or None"):
            qmc.select([_identity, _x], num_index_qubits=width)

    @pytest.mark.parametrize("width", [0, 1])
    def test_too_small_frontend_index_width_is_rejected(self, width: int) -> None:
        """A concrete width must address every supplied SELECT case."""
        with pytest.raises(ValueError, match="at least 2 index qubit"):
            qmc.select([_identity] * 4, num_index_qubits=width)

    def test_symbolic_width_is_preserved_by_public_api(self) -> None:
        """A UInt width remains symbolic until the transpiler resolves it."""
        width = qmc.uint("width")
        selector = qmc.select([_identity, _x], num_index_qubits=width)
        assert selector.num_index_qubits is width

    @pytest.mark.parametrize("width", [True, 3.0])
    def test_symbolic_width_binding_requires_a_plain_int(
        self,
        qiskit_transpiler: Any,
        width: Any,
    ) -> None:
        """Compile-time bindings cannot coerce bool or float into a width."""

        @qkernel
        def circuit(num_index_qubits: UInt) -> Bit:
            index = qmc.qubit_array(3, "index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [_identity, _x],
                num_index_qubits=num_index_qubits,
            )(index, target)
            return qmc.measure(target)

        with pytest.raises(TypeError) as exc_info:
            qiskit_transpiler.transpile(
                circuit,
                bindings={"num_index_qubits": width},
            )
        message = str(exc_info.value)
        assert "num_index_qubits" in message
        assert "integer" in message

    def test_symbolic_mixed_prefix_and_targets_keep_grouped_ir_layout(self) -> None:
        """Symbolic SELECT groups each scalar/array index and target argument."""
        from qamomile.circuit.ir.operation.select import SelectOperation

        @qkernel
        def pair_case(
            scalar: Qubit,
            vector: Vector[Qubit],
            theta: Float = 0.5,
        ) -> tuple[Qubit, Vector[Qubit]]:
            _ = theta
            return scalar, vector

        @qkernel
        def circuit(width: UInt) -> tuple[Bit, Vector[Bit]]:
            index_scalar = qmc.qubit("index_scalar")
            index_array = qmc.qubit_array(width - 1, "index_array")
            target_scalar = qmc.qubit("target_scalar")
            target_array = qmc.qubit_array(2, "target_array")
            index_scalar, index_array, target_scalar, target_array = qmc.select(
                [pair_case, pair_case],
                num_index_qubits=width,
            )(
                index_scalar,
                index_array,
                target_scalar,
                target_array,
            )
            return qmc.measure(target_scalar), qmc.measure(target_array)

        select_op = next(
            operation
            for operation in circuit.block.operations
            if isinstance(operation, SelectOperation)
        )
        assert select_op.num_index_args == 2
        assert select_op.num_index_qubits.name == "width"
        assert not isinstance(select_op.operands[0], ArrayValue)
        assert isinstance(select_op.operands[1], ArrayValue)
        assert not isinstance(select_op.operands[2], ArrayValue)
        assert isinstance(select_op.operands[3], ArrayValue)
        assert len(select_op.results) == 4

    def test_symbolic_width_accepts_a_lone_scalar_index(self) -> None:
        """SELECT permits one scalar index when a UInt width resolves to one."""
        from qamomile.circuit.ir.operation.select import SelectOperation

        @qkernel
        def circuit(width: UInt) -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [_identity, _x],
                num_index_qubits=width,
            )(index, target)
            return qmc.measure(target)

        select_op = next(
            operation
            for operation in circuit.block.operations
            if isinstance(operation, SelectOperation)
        )
        assert select_op.num_index_args == 1
        assert select_op.num_index_qubits.name == "width"

    def test_symbolic_case_failure_does_not_consume_inputs(self) -> None:
        """Symbolic SELECT validates every case before ownership transfer."""

        @qkernel
        def reset_case(q: Qubit) -> Qubit:
            return qmc.reset(q)

        width = qmc.uint("width")
        index = _make_qubit("index")
        target = _make_qubit("target")

        with trace():
            with pytest.raises(ValueError, match="non-unitary ResetOperation"):
                qmc.select(
                    [_identity, reset_case],
                    num_index_qubits=width,
                )(index, target)

        assert not index._consumed
        assert not target._consumed

    def test_concrete_width_does_not_split_an_index_array(self) -> None:
        """A concrete index boundary must not truncate an array argument."""
        with trace():
            index = qmc.qubit_array(3, "index")
            target = qmc.qubit("target")
            with pytest.raises(ValueError, match="boundary mid-argument"):
                qmc.select(
                    [_identity] * 4,
                    num_index_qubits=2,
                )(index, target)

            assert not index._consumed
            assert not target._consumed

    def test_aliased_index_and_target_are_rejected(self) -> None:
        """An aliased SELECT fails before committing index ownership."""
        qubit = _make_qubit("qubit")

        with trace():
            with pytest.raises(QubitConsumedError, match="overlapping physical"):
                qmc.select([_identity, _x])(qubit, qubit)

        assert not qubit._consumed

    def test_missing_tracer_does_not_consume_inputs(self) -> None:
        """SELECT validates tracer availability before ownership transfer."""
        index = _make_qubit("index")
        target = _make_qubit("target")

        with pytest.raises(RuntimeError, match="No active tracer"):
            qmc.select([_identity, _x])(index, target)

        assert not index._consumed
        assert not target._consumed

    def test_case_validation_failure_does_not_consume_inputs(self) -> None:
        """SELECT validates every case before committing input ownership."""

        @qkernel
        def reset_case(q: Qubit) -> Qubit:
            return qmc.reset(q)

        index = _make_qubit("index")
        target = _make_qubit("target")

        with trace():
            with pytest.raises(ValueError, match="non-unitary ResetOperation"):
                qmc.select([_identity, reset_case])(index, target)

        assert not index._consumed
        assert not target._consumed

    def test_output_permutation_requires_explicit_swap(self) -> None:
        """A case cannot conditionally relabel handles without a gate."""

        @qkernel
        def invalid() -> Bit:
            index = qmc.qubit("index")
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            index, q0, q1 = qmc.select([_identity_pair, _reverse_pair])(
                index,
                q0,
                q1,
            )
            return qmc.measure(q0)

        @qkernel
        def valid() -> Bit:
            index = qmc.qubit("index")
            q0 = qmc.qubit("q0")
            q1 = qmc.qubit("q1")
            index, q0, q1 = qmc.select([_identity_pair, _swap_pair])(
                index,
                q0,
                q1,
            )
            return qmc.measure(q0)

        with pytest.raises(ValueError, match="bare return-value permutations"):
            _ = invalid.block
        _ = valid.block

    def test_reset_and_internal_ancilla_are_rejected(self) -> None:
        """Cases are unitary on exactly the shared target footprint."""

        @qkernel
        def reset_case(q: Qubit) -> Qubit:
            return qmc.reset(q)

        @qkernel
        def ancilla_case(q: Qubit) -> Qubit:
            _ = qmc.qubit("ancilla")
            return q

        @qkernel
        def with_reset() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, reset_case])(index, target)
            return qmc.measure(target)

        @qkernel
        def with_ancilla() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, ancilla_case])(index, target)
            return qmc.measure(target)

        with pytest.raises(ValueError, match="non-unitary ResetOperation"):
            _ = with_reset.block
        with pytest.raises(ValueError, match="internal QInitOperation"):
            _ = with_ancilla.block

    def test_global_phase_gate_uses_explicit_qkernel_wrapper(self) -> None:
        """Combinators accept qkernels, not a SELECT-specific phase exception."""
        phase_gate = qmc.global_phase(_identity, 0.25)
        with pytest.raises(TypeError):
            qmc.select([_identity, phase_gate])

        @qkernel
        def phased_identity(q: Qubit) -> Qubit:
            return phase_gate(q)

        assert qmc.select([_identity, phased_identity]).num_cases == 2

    @pytest.mark.parametrize("num_index_qubits", [True, 1.0, 0, -1])
    def test_ir_rejects_invalid_index_width(self, num_index_qubits: Any) -> None:
        """Hand-built SELECT IR validates a positive plain-integer width."""
        from qamomile.circuit.ir.block import Block
        from qamomile.circuit.ir.operation.select import SelectOperation

        with pytest.raises(ValueError, match="Python int|positive"):
            SelectOperation(
                operands=[],
                results=[],
                num_index_qubits=num_index_qubits,
                case_blocks=[Block(), Block()],
            )


class TestSelectSerialization:
    """Preserve SELECT-owned case blocks across qkernel persistence."""

    @staticmethod
    def _affine(kernel: Any) -> Any:
        """Inline a kernel into the affine form accepted by serializers.

        Args:
            kernel (Any): QKernel whose traced block should be inlined.

        Returns:
            Any: Affine IR block accepted by persistent serializers.
        """
        from qamomile.circuit.transpiler.passes.inline import InlinePass

        return InlinePass().run(kernel.block)

    def test_round_trip_preserves_case_order(self) -> None:
        """QKernel protobuf persistence retains the case list and LSB ordering."""
        from qamomile.circuit.ir.canonical import content_hash
        from qamomile.circuit.ir.operation.select import SelectOperation
        from qamomile.circuit.serialization import deserialize, serialize

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit_array(2, "index")
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _x, _identity, _x])(
                index,
                target,
            )
            return qmc.measure(target)

        block = self._affine(circuit)
        restored = self._affine(deserialize(serialize(circuit)))
        [select] = [
            operation
            for operation in restored.operations
            if isinstance(operation, SelectOperation)
        ]
        assert select.num_index_qubits == 2
        assert len(select.case_blocks) == 4
        assert content_hash(restored) == content_hash(block)

    def test_round_trip_keeps_case_values_independent(self) -> None:
        """QKernel persistence does not alias values between SELECT cases."""
        from qamomile.circuit.ir.operation.select import SelectOperation
        from qamomile.circuit.serialization import deserialize, serialize

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _x])(index, target)
            return qmc.measure(target)

        restored = deserialize(serialize(circuit)).block
        select = next(
            operation
            for operation in restored.operations
            if isinstance(operation, SelectOperation)
        )
        left_case, right_case = select.case_blocks
        assert left_case is not right_case
        assert left_case.input_values[0].uuid != right_case.input_values[0].uuid
        assert left_case.output_values[0].uuid != right_case.output_values[0].uuid

    def test_content_hash_distinguishes_case_order(self) -> None:
        """Changing case order changes the semantic content hash."""
        from qamomile.circuit.ir.canonical import content_hash

        @qkernel
        def left() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _x])(index, target)
            return qmc.measure(target)

        @qkernel
        def right() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_x, _identity])(index, target)
            return qmc.measure(target)

        assert content_hash(self._affine(left)) != content_hash(self._affine(right))
