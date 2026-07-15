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
from qamomile.circuit.transpiler.errors import QubitConsumedError


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
def _select_x_on_one(index: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """Apply X only when a one-qubit index reads one."""
    return qmc.select([_identity, _x])(index, target)


@qkernel
def _select_x_on_zero(index: Qubit, target: Qubit) -> tuple[Qubit, Qubit]:
    """Apply X only when a one-qubit index reads zero."""
    return qmc.select([_x, _identity])(index, target)


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

    def test_aliased_index_and_target_are_rejected(self) -> None:
        """The same physical qubit cannot be both index and target."""

        @qkernel
        def circuit() -> Bit:
            qubit = qmc.qubit("qubit")
            qubit, qubit = qmc.select([_identity, _x])(qubit, qubit)
            return qmc.measure(qubit)

        with pytest.raises(QubitConsumedError, match="already consumed"):
            _ = circuit.block

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
    """Preserve SELECT-owned case blocks across persistent IR boundaries."""

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

    @pytest.mark.parametrize("encode_decode", ["json", "msgpack"])
    def test_round_trip_preserves_case_order(self, encode_decode: str) -> None:
        """JSON and msgpack retain the case list and its LSB ordering."""
        from qamomile.circuit.ir.canonical import content_hash
        from qamomile.circuit.ir.operation.select import SelectOperation
        from qamomile.circuit.ir.serialize import (
            dump_json,
            dump_msgpack,
            load_json,
            load_msgpack,
        )

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
        if encode_decode == "json":
            restored = load_json(dump_json(block))
        else:
            restored = load_msgpack(dump_msgpack(block))
        [select] = [
            operation
            for operation in restored.operations
            if isinstance(operation, SelectOperation)
        ]
        assert select.num_index_qubits == 2
        assert len(select.case_blocks) == 4
        assert content_hash(restored) == content_hash(block)

    def test_case_blocks_use_independent_value_tables(self) -> None:
        """Serialized case-local UUID tables stay disjoint and linear-sized."""
        from qamomile.circuit.ir.serialize import to_dict

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _x])(index, target)
            return qmc.measure(target)

        payload = to_dict(self._affine(circuit))["block"]
        select_payload = next(
            operation
            for operation in payload["operations"]
            if operation["$type"] == "SelectOperation"
        )
        tables = [payload["value_table"]] + [
            case["value_table"] for case in select_payload["case_blocks"]
        ]
        uuid_sets = [{entry["uuid"] for entry in table} for table in tables]

        for position, left in enumerate(uuid_sets):
            for right in uuid_sets[position + 1 :]:
                assert left.isdisjoint(right)

    def test_decoder_rejects_non_integer_index_width(self) -> None:
        """Wire decoding rejects a coercible string SELECT width."""
        from qamomile.circuit.ir.serialize import from_dict, to_dict

        @qkernel
        def circuit() -> Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_identity, _x])(index, target)
            return qmc.measure(target)

        payload = to_dict(self._affine(circuit))
        select_payload = next(
            operation
            for operation in payload["block"]["operations"]
            if operation["$type"] == "SelectOperation"
        )
        select_payload["num_index_qubits"] = "1"

        with pytest.raises(ValueError, match="must be a Python int"):
            from_dict(payload)

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
