"""Tests for Layer 3: ``SymbolicShapeValidationPass``.

When a top-level Vector parameter's symbolic shape dimension reaches a
``ForOperation`` loop bound without being folded, transpile must raise a
``QamomileCompileError`` with an actionable message — not silently elide
the loop, not fail cryptically at emit time. The same applies to loop
bounds left as runtime parameters (``parameters=["n"]`` with
``qmc.range(n)``): they must fail here, before segmentation, with the
"Cannot unroll loop" message — not as a misleading
``MultipleQuantumSegmentsError`` at plan or a late emit-time
``ValueError``. The library QAOA pattern (``p`` bound in bindings,
``gammas.shape`` never queried) must keep working unchanged.
"""

import pytest

pytest.importorskip("qiskit")

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.algorithm.basic import superposition_vector
from qamomile.circuit.algorithm.qaoa import qaoa_layers, x_mixer
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import (
    QamomileCompileError,
    ValidationError,
)
from qamomile.qiskit.transpiler import QiskitTranspiler


def _make_h() -> qm_o.Hamiltonian:
    H = qm_o.Hamiltonian()
    H.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Z, 1),
        ),
        1.0,
    )
    return H


@qmc.qkernel
def _owned_expression_identity(
    target: qmc.Qubit,
    repetitions: qmc.UInt,
) -> qmc.Qubit:
    """Keep a target unchanged through a case-local expression-bound loop."""
    for iteration in qmc.range(repetitions + 1):
        if iteration < 0:
            target = qmc.x(target)
    return target


@qmc.qkernel
def _owned_direct_x(
    target: qmc.Qubit,
    repetitions: qmc.UInt,
) -> qmc.Qubit:
    """Apply X through a directly parameter-bound operation-owned loop."""
    for iteration in qmc.range(repetitions):
        if iteration >= 0:
            target = qmc.x(target)
    return target


@qmc.qkernel
def _owned_array_shape_identity(
    target: qmc.Qubit,
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Qubit:
    """Keep a target unchanged through an owned array-shape loop."""
    for index in qmc.range(sizes.shape[0]):
        if index < 0:
            target = qmc.x(target)
    return target


@qmc.qkernel
def _owned_array_shape_x(
    target: qmc.Qubit,
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Qubit:
    """Apply X through an owned array-shape loop."""
    for index in qmc.range(sizes.shape[0]):
        if index >= 0:
            target = qmc.x(target)
    return target


@qmc.qkernel
def _owned_derived_array_shape_identity(
    target: qmc.Qubit,
    sizes: qmc.Vector[qmc.UInt],
) -> qmc.Qubit:
    """Keep a target unchanged through an owned derived-shape loop.

    Args:
        target (qmc.Qubit): Target qubit to preserve.
        sizes (qmc.Vector[qmc.UInt]): Values whose symbolic length controls the
            loop structure.

    Returns:
        qmc.Qubit: Unchanged target qubit.
    """
    pair_count = (sizes.shape[0] - 1) // 2
    for index in qmc.range(pair_count):
        if index < 0:
            target = qmc.x(target)
    return target


@qmc.qkernel
def _structural_identity(target: qmc.Qubit) -> qmc.Qubit:
    """Return a target unchanged for symbolic-structure tests."""
    return target


@qmc.qkernel
def _structural_x(target: qmc.Qubit) -> qmc.Qubit:
    """Apply X to a target for symbolic-structure tests."""
    return qmc.x(target)


@qmc.qkernel
def _runtime_array_select_width(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    """Use a runtime array element as a SELECT width."""
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_structural_identity, _structural_x],
        num_index_qubits=values[0],
    )(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _runtime_array_num_controls(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    """Use a runtime array element as a controlled-U width."""
    controls = qmc.qubit_array(2, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(qmc.x, num_controls=values[0])(
        controls,
        target,
    )
    return qmc.measure(target)


@qmc.qkernel
def _runtime_array_control_power(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    """Use a runtime array element as a controlled-U power."""
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(qmc.x)(control, target, power=values[0])
    return qmc.measure(target)


@qmc.qkernel
def _runtime_array_control_index(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    """Use a runtime array element as a control-pool index."""
    controls = qmc.qubit_array(2, "controls")
    target = qmc.qubit("target")
    controls, target = qmc.control(qmc.x, num_controls=qmc.uint(1))(
        controls,
        target,
        control_indices=[values[0]],
    )
    return qmc.measure(target)


@qmc.qkernel
def _runtime_array_element_output(
    values: qmc.Vector[qmc.Float],
) -> qmc.Float:
    """Return the first runtime array element without an operation consumer.

    Args:
        values (qmc.Vector[qmc.Float]): Runtime output values.

    Returns:
        qmc.Float: First array element.
    """
    return values[0]


@qmc.qkernel
def _owned_array_element_rx(
    target: qmc.Qubit,
    values: qmc.Vector[qmc.Float],
) -> qmc.Qubit:
    """Rotate a target by the first operation-owned array element.

    Args:
        target (qmc.Qubit): Target qubit to rotate.
        values (qmc.Vector[qmc.Float]): Compile-time angle values.

    Returns:
        qmc.Qubit: Rotated target qubit.
    """
    return qmc.rx(target, values[0])


@qmc.qkernel
def _owned_array_element_identity(
    target: qmc.Qubit,
    values: qmc.Vector[qmc.Float],
) -> qmc.Qubit:
    """Preserve a target through a signature-compatible SELECT case.

    Args:
        target (qmc.Qubit): Target qubit to preserve.
        values (qmc.Vector[qmc.Float]): Unused angle values retained for the
            shared SELECT signature.

    Returns:
        qmc.Qubit: Unchanged target qubit.
    """
    return target


@qmc.qkernel
def _owned_select_array_element(values: qmc.Vector[qmc.Float]) -> qmc.Bit:
    """Use an array element inside a SELECT-owned case block.

    Args:
        values (qmc.Vector[qmc.Float]): Compile-time angle values.

    Returns:
        qmc.Bit: Measured target qubit.
    """
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_owned_array_element_rx, _owned_array_element_identity]
    )(index, target, values=values)
    return qmc.measure(target)


@qmc.qkernel
def _owned_control_array_element(values: qmc.Vector[qmc.Float]) -> qmc.Bit:
    """Use an array element inside a controlled-unitary block.

    Args:
        values (qmc.Vector[qmc.Float]): Compile-time angle values.

    Returns:
        qmc.Bit: Measured target qubit.
    """
    control = qmc.qubit("control")
    target = qmc.qubit("target")
    control, target = qmc.control(_owned_array_element_rx)(
        control,
        target,
        values=values,
    )
    return qmc.measure(target)


@qmc.qkernel
def _serialized_zero_trip_shape_body(
    values: qmc.Vector[qmc.UInt],
    repetitions: qmc.UInt,
) -> qmc.Bit:
    """Use a runtime shape only inside a possibly empty counted loop.

    Args:
        values (qmc.Vector[qmc.UInt]): Runtime array whose shape is structural.
        repetitions (qmc.UInt): Compile-time outer-loop trip count.

    Returns:
        qmc.Bit: Measurement of the loop target.
    """
    target = qmc.qubit("target")
    for _outer in qmc.range(repetitions):
        for _inner in qmc.range(values.shape[0]):
            target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _serialized_for_items_shape_body(
    values: qmc.Vector[qmc.UInt],
    items: qmc.Dict[qmc.UInt, qmc.UInt],
) -> qmc.Bit:
    """Use a runtime shape only inside a compile-time items loop.

    Args:
        values (qmc.Vector[qmc.UInt]): Runtime array whose shape is structural.
        items (qmc.Dict[qmc.UInt, qmc.UInt]): Compile-time loop entries.

    Returns:
        qmc.Bit: Measurement of the loop target.
    """
    target = qmc.qubit("target")
    for _key, _value in qmc.items(items):
        for _inner in qmc.range(values.shape[0]):
            target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _serialized_unknown_items_region_result(
    initial: qmc.UInt,
    items: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Bit:
    """Use a possibly zero-trip items-loop result as later structure.

    Args:
        initial (qmc.UInt): Runtime initializer selected when ``items`` is empty.
        items (qmc.Dict[qmc.UInt, qmc.Float]): Runtime mapping with unresolved
            compile-time cardinality.

    Returns:
        qmc.Bit: Measurement of the post-loop target.
    """
    count = initial
    for _key, _value in qmc.items(items):
        count = qmc.uint(1)
    target = qmc.qubit("target")
    for _index in qmc.range(count):
        target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _serialized_vector_key_shape_range(
    items: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Bit:
    """Use each bound vector-key length as a nested loop bound.

    Args:
        items (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Compile-time mapping
            with vector keys whose lengths are resolved per item.

    Returns:
        qmc.Bit: Measurement of the loop target.
    """
    target = qmc.qubit("target")
    for key, _value in qmc.items(items):
        for _index in qmc.range(key.shape[0]):
            target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _serialized_vector_key_shape_select(
    items: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Bit:
    """Use a bound vector-key length as a SELECT index width.

    Args:
        items (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Compile-time mapping
            with two-element vector keys.

    Returns:
        qmc.Bit: Measurement of the selected target.
    """
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    for key, _value in qmc.items(items):
        index, target = qmc.select(
            [_structural_identity, _structural_x],
            num_index_qubits=key.shape[0],
        )(index, target)
    return qmc.measure(target)


@qmc.qkernel
def _serialized_vector_key_shape_control(
    items: qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float],
) -> qmc.Bit:
    """Use a bound vector-key length as a controlled-unitary width.

    Args:
        items (qmc.Dict[qmc.Vector[qmc.UInt], qmc.Float]): Compile-time mapping
            with two-element vector keys.

    Returns:
        qmc.Bit: Measurement of the controlled target.
    """
    controls = qmc.qubit_array(2, "controls")
    target = qmc.qubit("target")
    for key, _value in qmc.items(items):
        controls, target = qmc.control(
            qmc.x,
            num_controls=key.shape[0],
        )(controls, target)
    return qmc.measure(target)


@qmc.qkernel
def _runtime_if_distinct_structural_bound(flag: qmc.UInt) -> qmc.Bit:
    """Select two distinct loop bounds with a runtime condition.

    Args:
        flag (qmc.UInt): Runtime branch selector.

    Returns:
        qmc.Bit: Measurement of the loop target.
    """
    bound = qmc.uint(1)
    if flag == 1:
        bound = qmc.uint(1)
    else:
        bound = qmc.uint(2)
    target = qmc.qubit("target")
    for _index in qmc.range(bound):
        target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _runtime_if_equal_structural_bound(flag: qmc.UInt) -> qmc.Bit:
    """Select the same loop bound from both runtime branches.

    Args:
        flag (qmc.UInt): Runtime branch selector.

    Returns:
        qmc.Bit: Measurement of the loop target.
    """
    bound = qmc.uint(0)
    if flag == 1:
        bound = qmc.uint(1)
    else:
        bound = qmc.uint(1)
    target = qmc.qubit("target")
    for _index in qmc.range(bound):
        target = qmc.x(target)
    return qmc.measure(target)


@qmc.qkernel
def _runtime_if_identity_structural_bound(flag: qmc.UInt) -> qmc.Bit:
    """Preserve one loop-bound handle through both runtime branches.

    Args:
        flag (qmc.UInt): Runtime branch selector.

    Returns:
        qmc.Bit: Measurement of the loop target.
    """
    bound = qmc.uint(1)
    if flag == 1:
        bound = bound
    else:
        bound = bound
    target = qmc.qubit("target")
    for _index in qmc.range(bound):
        target = qmc.x(target)
    return qmc.measure(target)


class TestRejection:
    """Patterns that Layer 3 should catch."""

    def test_flat_kernel_unresolved_shape_raises(self):
        """Flat kernel using ``gamma.shape[0]`` with no binding is rejected."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(gamma.shape[0]):
                q = x_mixer(q, gamma[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"n": H.num_qubits, "hamiltonian": H},
                parameters=["gamma"],
            )
        msg = str(exc_info.value)
        assert "gamma" in msg
        assert "shape dimension 0" in msg

    def test_error_suggests_concrete_binding(self):
        """Error message guides users to bind the array concretely."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            betas: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(betas.shape[0]):
                q = x_mixer(q, betas[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"n": H.num_qubits, "hamiltonian": H},
                parameters=["betas"],
            )
        msg = str(exc_info.value)
        assert "bindings" in msg
        assert "betas" in msg

    def test_error_suggests_loop_counter(self):
        """Error message also shows a parameter-specific counter pattern."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(gamma.shape[0]):
                q = x_mixer(q, gamma[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"n": H.num_qubits, "hamiltonian": H},
                parameters=["gamma"],
            )
        msg = str(exc_info.value)
        assert "qm.range" in msg
        assert "gamma_count" in msg

    def test_derived_array_shape_bound_names_array_and_counter(self):
        """Arithmetic derived from a runtime array shape keeps its provenance."""

        @qmc.qkernel
        def kernel(phases: qmc.Vector[qmc.Float]) -> qmc.Bit:
            """Use an arithmetic expression derived from a phase-vector shape.

            Args:
                phases (qmc.Vector[qmc.Float]): Runtime phase parameters.

            Returns:
                qmc.Bit: Measured output qubit.
            """
            q = qmc.qubit("q")
            pair_count = (phases.shape[0] - 1) // 2
            for pair in qmc.range(pair_count):
                q = qmc.rx(q, phases[pair])
            return qmc.measure(q)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(kernel, parameters=["phases"])

        msg = str(exc_info.value)
        assert "Parameter array 'phases' has unresolved shape dimension 0" in msg
        assert "bindings={'phases': [...]}" in msg
        assert "phase_count" in msg
        assert "phases: qm.Vector[qm.Float]" in msg

    def test_derived_array_shape_bound_accepts_compile_time_array(self):
        """The derived bound becomes valid when the array shape is bound."""

        @qmc.qkernel
        def kernel(phases: qmc.Vector[qmc.Float]) -> qmc.Bit:
            """Use a bound phase-vector shape as a compile-time loop bound.

            Args:
                phases (qmc.Vector[qmc.Float]): Compile-time phase values.

            Returns:
                qmc.Bit: Measured output qubit.
            """
            q = qmc.qubit("q")
            pair_count = (phases.shape[0] - 1) // 2
            for pair in qmc.range(pair_count):
                q = qmc.rx(q, phases[pair])
            return qmc.measure(q)

        executable = QiskitTranspiler().transpile(
            kernel,
            bindings={"phases": [0.1, 0.2, 0.3]},
        )

        assert executable.get_first_circuit() is not None


class TestRegionBoundaryStructure:
    """Region boundaries preserve structural provenance and reachability."""

    def test_two_trip_runtime_scalar_region_arg_fails_before_emit(self) -> None:
        """A runtime scalar carried into a nested bound keeps its source."""

        @qmc.qkernel
        def kernel(n: qmc.UInt) -> qmc.Bit:
            """Use a runtime scalar through a two-trip RegionArg bound.

            Args:
                n (qmc.UInt): Runtime value carried as the nested loop bound.

            Returns:
                qmc.Bit: Measurement of the loop target.
            """
            target = qmc.qubit("target")
            count = n
            for _outer in qmc.range(2):
                for _inner in qmc.range(count):
                    target = qmc.x(target)
                count = count + 0
            return qmc.measure(target)

        restored = deserialize(serialize(kernel))

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(restored, parameters=["n"])

        assert type(exc_info.value) is QamomileCompileError
        message = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved" in message
        assert "runtime parameter 'n'" in message

    def test_two_trip_runtime_shape_region_arg_fails_before_emit(self) -> None:
        """A runtime shape carried into a nested bound keeps its source."""

        @qmc.qkernel
        def kernel(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
            """Use a runtime shape through a two-trip RegionArg bound.

            Args:
                values (qmc.Vector[qmc.UInt]): Runtime array whose shape is
                    carried.

            Returns:
                qmc.Bit: Measurement of the loop target.
            """
            target = qmc.qubit("target")
            count = values.shape[0]
            for _outer in qmc.range(2):
                for _inner in qmc.range(count):
                    target = qmc.x(target)
                count = count + 0
            return qmc.measure(target)

        restored = deserialize(serialize(kernel))

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(restored, parameters=["values"])

        assert type(exc_info.value) is QamomileCompileError
        message = str(exc_info.value)
        assert "Parameter array 'values' has unresolved shape dimension 0" in message
        assert "value_count" in message

    def test_single_trip_region_arg_does_not_follow_unused_yield(self) -> None:
        """A sole iteration reads its constant init, not its runtime yield."""

        @qmc.qkernel
        def kernel(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
            """Yield a runtime shape after the sole constant-bound use.

            Args:
                values (qmc.Vector[qmc.UInt]): Runtime array yielded after the
                    only loop iteration.

            Returns:
                qmc.Bit: Measurement of the loop target.
            """
            target = qmc.qubit("target")
            count = qmc.uint(1)
            for _outer in qmc.range(1):
                for _inner in qmc.range(count):
                    target = qmc.x(target)
                count = values.shape[0]
            return qmc.measure(target)

        restored = deserialize(serialize(kernel))

        executable = QiskitTranspiler().transpile(
            restored,
            parameters=["values"],
        )

        assert executable.get_first_circuit() is not None

    def test_post_loop_region_result_keeps_runtime_shape_source(self) -> None:
        """A structural use after the loop follows the final yielded value."""

        @qmc.qkernel
        def kernel(values: qmc.Vector[qmc.UInt]) -> qmc.Bit:
            """Use a runtime shape after it exits a RegionArg loop.

            Args:
                values (qmc.Vector[qmc.UInt]): Runtime array whose shape is
                    yielded.

            Returns:
                qmc.Bit: Measurement of the post-loop target.
            """
            target = qmc.qubit("target")
            count = qmc.uint(1)
            for _outer in qmc.range(2):
                count = values.shape[0]
            for _inner in qmc.range(count):
                target = qmc.x(target)
            return qmc.measure(target)

        restored = deserialize(serialize(kernel))

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(restored, parameters=["values"])

        assert type(exc_info.value) is QamomileCompileError
        assert "Parameter array 'values' has unresolved shape dimension 0" in str(
            exc_info.value
        )

    def test_serialized_zero_trip_for_skips_runtime_shape_body(self) -> None:
        """A runtime shape inside a statically empty range loop is unreachable."""
        restored = deserialize(serialize(_serialized_zero_trip_shape_body))

        executable = QiskitTranspiler().transpile(
            restored,
            bindings={"repetitions": 0},
            parameters=["values"],
        )

        assert executable.get_first_circuit() is not None

    def test_serialized_nonempty_for_validates_runtime_shape_body(self) -> None:
        """The same runtime shape is structural when the outer loop executes."""
        restored = deserialize(serialize(_serialized_zero_trip_shape_body))

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(
                restored,
                bindings={"repetitions": 1},
                parameters=["values"],
            )

        assert type(exc_info.value) is QamomileCompileError
        assert "Parameter array 'values' has unresolved shape dimension 0" in str(
            exc_info.value
        )

    def test_serialized_empty_for_items_skips_runtime_shape_body(self) -> None:
        """A runtime shape inside an empty bound items loop is unreachable."""
        restored = deserialize(serialize(_serialized_for_items_shape_body))

        executable = QiskitTranspiler().transpile(
            restored,
            bindings={"items": {}},
            parameters=["values"],
        )

        assert executable.get_first_circuit() is not None

    def test_serialized_nonempty_for_items_validates_runtime_shape_body(self) -> None:
        """The same items-body shape is structural for a nonempty Dict."""
        restored = deserialize(serialize(_serialized_for_items_shape_body))

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(
                restored,
                bindings={"items": {0: 1}},
                parameters=["values"],
            )

        assert type(exc_info.value) is QamomileCompileError
        assert "Parameter array 'values' has unresolved shape dimension 0" in str(
            exc_info.value
        )


class TestRuntimeParameterLoopBound:
    """Loop bounds left as runtime parameters fail early and actionably.

    Regression suite for the diagnostic inconsistency where the same user
    mistake (a ``qmc.range`` bound depending on a runtime parameter)
    surfaced either as a misleading ``MultipleQuantumSegmentsError``
    blaming measurement-dependent control flow, or as a late emit-time
    ``ValueError`` — depending on which pass tripped first.
    """

    def test_direct_runtime_parameter_bound_raises_actionable_error(self):
        """``qmc.range(n)`` with ``parameters=["n"]`` fails at validation.

        Pre-fix this case passed segmentation and failed at emit with a
        ``ValueError``; it must now raise ``QamomileCompileError`` with the
        canonical "Cannot unroll loop" wording and a bindings fix.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(n):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(kernel, bindings={"theta": 0.5}, parameters=["n"])
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'n'" in msg
        assert "bindings" in msg

    def test_bound_expression_no_longer_multiple_segments_error(self):
        """An arithmetic bound expression gets the same clear error.

        Pre-fix ``qmc.range(num_pairs + num_pairs)`` stranded the bound's
        ``BinOp`` between quantum ops, so segmentation raised
        ``MultipleQuantumSegmentsError`` blaming measurement-dependent
        control flow — the wrong diagnosis. The dataflow walk must trace
        the bound back to ``num_pairs`` and name it. (The raised
        ``QamomileCompileError`` is not a ``MultipleQuantumSegmentsError``
        — the two classes are unrelated in the exception hierarchy.)
        """

        @qmc.qkernel
        def kernel(
            theta: qmc.Vector[qmc.Float], num_pairs: qmc.UInt
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for pair in qmc.range(num_pairs + num_pairs):
                q0 = q[0]
                q1 = q[1]
                q0, q1 = qmc.cp(q0, q1, theta[pair])
                q[0] = q0
                q[1] = q1
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"theta": [0.1, 0.2, 0.3, 0.4]},
                parameters=["num_pairs"],
            )
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'num_pairs'" in msg

    def test_helper_kernel_repro_raises_actionable_error(self):
        """The reported repro shape (helper + indexed angle) is caught.

        A helper qkernel applying a controlled phase, called in a loop
        whose bound is a runtime parameter, must get the loop-bound
        diagnostic naming ``num_pairs`` — not an emit-time ``ValueError``
        or a segmentation error.
        """

        @qmc.qkernel
        def cphase_helper(
            q: qmc.Vector[qmc.Qubit], angle: qmc.Float
        ) -> qmc.Vector[qmc.Qubit]:
            q0 = q[0]
            q1 = q[1]
            q0, q1 = qmc.cp(q0, q1, angle)
            q[0] = q0
            q[1] = q1
            return q

        @qmc.qkernel
        def kernel(
            theta: qmc.Vector[qmc.Float], num_pairs: qmc.UInt
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for pair in qmc.range(num_pairs):
                q = cphase_helper(q, theta[pair + pair])
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"theta": [0.1, 0.2, 0.3, 0.4]},
                parameters=["num_pairs"],
            )
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'num_pairs'" in msg

    def test_auto_detected_runtime_parameter_bound_raises(self):
        """A bound auto-detected as a runtime parameter is caught too.

        With no ``parameters`` list, an unbound classical argument without
        a Python default becomes a runtime parameter via auto-detect; a
        loop bound depending on it must get the same diagnostic.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(n):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(kernel, bindings={"theta": 0.5})
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'n'" in msg

    def test_runtime_parameter_select_width_raises_actionable_error(self):
        """A runtime SELECT width fails before semantic lowering."""

        @qmc.qkernel
        def identity(target: qmc.Qubit) -> qmc.Qubit:
            """Return a SELECT target unchanged."""
            return target

        @qmc.qkernel
        def flipped(target: qmc.Qubit) -> qmc.Qubit:
            """Apply X to a SELECT target."""
            return qmc.x(target)

        @qmc.qkernel
        def kernel(width: qmc.UInt) -> qmc.Bit:
            """Use a runtime parameter as SELECT's structural width."""
            index = qmc.qubit_array(2, "index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [identity, flipped],
                num_index_qubits=width,
            )(index, target)
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(kernel, parameters=["width"])

        msg = str(exc_info.value)
        assert "Cannot resolve SELECT index width at compile time" in msg
        assert "runtime parameter 'width'" in msg
        assert "bindings={'width': <int>}" in msg

    def test_runtime_parameter_array_element_bound_raises(self):
        """A bound indexing a runtime parameter array names the array.

        ``qmc.range(idxs[0])`` with ``parameters=["idxs"]`` reaches the
        array through the element's parent-array dataflow edge; the
        diagnostic must name ``idxs``.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, idxs: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(idxs[0]):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(kernel, bindings={"theta": 0.5}, parameters=["idxs"])
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'idxs'" in msg

    def test_runtime_parameter_index_into_bound_array_raises(self):
        """A bound using a runtime index into a bound array names the index.

        ``qmc.range(idxs[start])`` with ``idxs`` bound but ``start`` left as
        a runtime parameter must fail during validation, not later at emit.
        """

        @qmc.qkernel
        def kernel(
            theta: qmc.Float,
            idxs: qmc.Vector[qmc.UInt],
            start: qmc.UInt,
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(idxs[start]):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"theta": 0.5, "idxs": [1, 2]},
                parameters=["start"],
            )
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'start'" in msg

    def test_runtime_parameter_array_element_index_into_bound_array_raises(self):
        """A bound using a runtime array element as an index names the array.

        ``qmc.range(sizes[idxs[0]])`` must trace through the index value's
        own parent array metadata and report that ``idxs`` is runtime.
        """

        @qmc.qkernel
        def kernel(
            theta: qmc.Float,
            sizes: qmc.Vector[qmc.UInt],
            idxs: qmc.Vector[qmc.UInt],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(sizes[idxs[0]]):
                q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        with pytest.raises(QamomileCompileError) as exc_info:
            tr.transpile(
                kernel,
                bindings={"theta": 0.5, "sizes": [1, 2]},
                parameters=["idxs"],
            )
        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved at compile time" in msg
        assert "'idxs'" in msg


class TestOperationOwnedStructure:
    """Operation-owned blocks and controlled fields fail before emit."""

    def test_runtime_parameter_select_case_bound_raises_actionable_error(self):
        """A SELECT case expression inherits runtime provenance from its actual."""

        @qmc.qkernel
        def kernel(repetitions: qmc.UInt) -> qmc.Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_owned_expression_identity, _owned_direct_x])(
                index, target, repetitions=repetitions
            )
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(
                kernel,
                parameters=["repetitions"],
            )

        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved" in msg
        assert "runtime parameter 'repetitions'" in msg
        assert "report this as a compiler bug" not in msg

    def test_runtime_parameter_controlled_body_bound_raises_actionable_error(self):
        """A controlled body inherits runtime provenance from its call actual."""

        @qmc.qkernel
        def kernel(repetitions: qmc.UInt) -> qmc.Bit:
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control, target = qmc.control(_owned_direct_x)(
                control,
                target,
                repetitions=repetitions,
            )
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(
                kernel,
                parameters=["repetitions"],
            )

        msg = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved" in msg
        assert "runtime parameter 'repetitions'" in msg

    def test_runtime_array_actual_shape_raises_in_select_case(self):
        """An owned formal shape traces through its runtime array actual."""

        @qmc.qkernel
        def kernel(sizes: qmc.Vector[qmc.UInt]) -> qmc.Bit:
            """Select an owned callable whose loop uses an array shape.

            Args:
                sizes (qmc.Vector[qmc.UInt]): Runtime values with symbolic
                    vector length.

            Returns:
                qmc.Bit: Measured target qubit.
            """
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [_owned_array_shape_identity, _owned_array_shape_x]
            )(index, target, sizes=sizes)
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(kernel, parameters=["sizes"])

        msg = str(exc_info.value)
        assert "Parameter array 'sizes' has unresolved shape dimension 0" in msg
        assert "bindings={'sizes': [...]}" in msg
        assert "size_count" in msg

    def test_derived_runtime_array_shape_traces_through_owned_formal(self):
        """An owned derived shape follows its formal-to-actual dimension edge."""

        @qmc.qkernel
        def kernel(sizes: qmc.Vector[qmc.UInt]) -> qmc.Bit:
            """Route a runtime shape into an operation-owned derived loop.

            Args:
                sizes (qmc.Vector[qmc.UInt]): Runtime values with symbolic
                    vector length.

            Returns:
                qmc.Bit: Measured target qubit.
            """
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [_owned_derived_array_shape_identity, _owned_array_shape_x]
            )(index, target, sizes=sizes)
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(kernel, parameters=["sizes"])

        msg = str(exc_info.value)
        assert "Parameter array 'sizes' has unresolved shape dimension 0" in msg
        assert "size_count" in msg
        assert "sizes: qm.Vector[qm.UInt]" in msg

    def test_concrete_array_actual_shape_passes_in_select_case(self):
        """A concrete array actual resolves its owned formal shape."""

        @qmc.qkernel
        def kernel(sizes: qmc.Vector[qmc.UInt]) -> qmc.Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [_owned_array_shape_identity, _owned_array_shape_x]
            )(index, target, sizes=sizes)
            return qmc.measure(target)

        executable = QiskitTranspiler().transpile(
            kernel,
            bindings={"sizes": [1, 2]},
        )
        assert executable.get_first_circuit() is not None

    def test_runtime_parameter_num_controls_raises_before_emit(self):
        """A runtime symbolic control width gets the structural diagnostic."""

        @qmc.qkernel
        def kernel(num_controls: qmc.UInt) -> qmc.Bit:
            controls = qmc.qubit_array(2, "controls")
            target = qmc.qubit("target")
            controls, target = qmc.control(
                qmc.x,
                num_controls=num_controls,
            )(controls, target)
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(
                kernel,
                parameters=["num_controls"],
            )

        msg = str(exc_info.value)
        assert "controlled-unitary num_controls" in msg
        assert "runtime parameter 'num_controls'" in msg

    def test_runtime_parameter_control_power_raises_before_emit(self):
        """A runtime controlled-U power gets the structural diagnostic."""

        @qmc.qkernel
        def kernel(power: qmc.UInt) -> qmc.Bit:
            control = qmc.qubit("control")
            target = qmc.qubit("target")
            control, target = qmc.control(qmc.x)(
                control,
                target,
                power=power,
            )
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(kernel, parameters=["power"])

        msg = str(exc_info.value)
        assert "controlled-unitary power" in msg
        assert "runtime parameter 'power'" in msg

    def test_runtime_parameter_control_index_raises_before_emit(self):
        """A runtime control-pool index gets the structural diagnostic."""

        @qmc.qkernel
        def kernel(num_controls: qmc.UInt, selected: qmc.UInt) -> qmc.Bit:
            controls = qmc.qubit_array(2, "controls")
            target = qmc.qubit("target")
            controls, target = qmc.control(
                qmc.x,
                num_controls=num_controls,
            )(
                controls,
                target,
                control_indices=[selected],
            )
            return qmc.measure(target)

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(
                kernel,
                bindings={"num_controls": 1},
                parameters=["selected"],
            )

        msg = str(exc_info.value)
        assert "controlled-unitary control_indices[0]" in msg
        assert "runtime parameter 'selected'" in msg

    @pytest.mark.parametrize(
        ("kernel", "diagnostic"),
        [
            (_runtime_array_select_width, "SELECT index width"),
            (_runtime_array_num_controls, "controlled-unitary num_controls"),
            (_runtime_array_control_power, "controlled-unitary power"),
            (
                _runtime_array_control_index,
                "controlled-unitary control_indices[0]",
            ),
        ],
        ids=["select-width", "num-controls", "power", "control-index"],
    )
    def test_runtime_array_element_structure_raises_before_emit(
        self,
        kernel,
        diagnostic: str,
    ) -> None:
        """Subclass-specific structural inputs retain runtime provenance."""
        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(kernel, parameters=["values"])

        message = str(exc_info.value)
        assert diagnostic in message
        assert "runtime parameter 'values'" in message
        assert "report this as a compiler bug" not in message

    @pytest.mark.parametrize(
        "kernel",
        [
            _runtime_array_select_width,
            _runtime_array_num_controls,
            _runtime_array_control_power,
            _runtime_array_control_index,
        ],
        ids=["select-width", "num-controls", "power", "control-index"],
    )
    def test_bound_empty_structural_array_fails_before_emit(self, kernel) -> None:
        """Deserialized structural fields use the early array-bounds check."""
        restored = deserialize(serialize(kernel))

        with pytest.raises(ValidationError) as exc_info:
            QiskitTranspiler().transpile(restored, bindings={"values": []})

        message = str(exc_info.value)
        assert "Index 0 is out of range" in message
        assert "values" in message

    def test_bound_empty_public_output_array_fails_before_emit(self) -> None:
        """A deserialized output-only element access is checked before emit."""
        restored = deserialize(serialize(_runtime_array_element_output))

        with pytest.raises(ValidationError) as exc_info:
            QiskitTranspiler().transpile(restored, bindings={"values": []})

        message = str(exc_info.value)
        assert "Index 0 is out of range" in message
        assert "values" in message

    @pytest.mark.parametrize(
        "kernel",
        [_owned_select_array_element, _owned_control_array_element],
        ids=["select-case", "controlled-body"],
    )
    def test_bound_empty_operation_owned_array_fails_before_emit(
        self,
        kernel,
    ) -> None:
        """Deserialized owned blocks inherit array bounds from call actuals."""
        restored = deserialize(serialize(kernel))

        with pytest.raises(ValidationError) as exc_info:
            QiskitTranspiler().transpile(restored, bindings={"values": []})

        message = str(exc_info.value)
        assert "Index 0 is out of range" in message
        assert "values" in message

    @pytest.mark.parametrize(
        "kernel",
        [_owned_select_array_element, _owned_control_array_element],
        ids=["select-case", "controlled-body"],
    )
    def test_bound_nonempty_operation_owned_array_compiles(self, kernel) -> None:
        """Owned blocks accept an actual array covering every used index."""
        restored = deserialize(serialize(kernel))

        executable = QiskitTranspiler().transpile(
            restored,
            bindings={"values": [0.25]},
        )

        assert executable.get_first_circuit() is not None

    def test_compile_time_owned_and_control_structure_values_pass(self):
        """Compile-time bindings still resolve every newly checked structure."""

        @qmc.qkernel
        def select_kernel(repetitions: qmc.UInt) -> qmc.Bit:
            index = qmc.qubit("index")
            target = qmc.qubit("target")
            index, target = qmc.select([_owned_expression_identity, _owned_direct_x])(
                index, target, repetitions=repetitions
            )
            return qmc.measure(target)

        @qmc.qkernel
        def control_kernel(
            num_controls: qmc.UInt,
            power: qmc.UInt,
            selected: qmc.UInt,
        ) -> qmc.Bit:
            controls = qmc.qubit_array(2, "controls")
            target = qmc.qubit("target")
            controls, target = qmc.control(
                qmc.x,
                num_controls=num_controls,
            )(
                controls,
                target,
                power=power,
                control_indices=[selected],
            )
            return qmc.measure(target)

        transpiler = QiskitTranspiler()
        assert (
            transpiler.transpile(
                select_kernel,
                bindings={"repetitions": 1},
            ).get_first_circuit()
            is not None
        )
        assert (
            transpiler.transpile(
                control_kernel,
                bindings={"num_controls": 1, "power": 2, "selected": 0},
            ).get_first_circuit()
            is not None
        )


class TestAcceptance:
    """Patterns that Layer 3 should leave alone."""

    def test_nested_bound_on_outer_loop_var_passes(self):
        """A nested ``qmc.range(i + 1)`` bound resolves during unrolling.

        Bounds derived from an enclosing loop variable are not runtime
        parameters — the dataflow walk must stop at the loop variable and
        let emit-time unrolling supply the concrete value.
        """

        @qmc.qkernel
        def kernel(theta: qmc.Float, p: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(2, "q")
            for i in qmc.range(p):
                for j in qmc.range(i + 1):
                    q[0] = qmc.rx(q[0], theta)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(kernel, bindings={"p": 3}, parameters=["theta"])
        assert exe.get_first_circuit() is not None

    def test_library_qaoa_layers_pattern_passes(self):
        """``qaoa_layers`` with ``p`` bound is the blessed pattern."""

        @qmc.qkernel
        def kernel(
            p: qmc.UInt,
            quad: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            linear: qmc.Dict[qmc.UInt, qmc.Float],
            n: qmc.UInt,
            gammas: qmc.Vector[qmc.Float],
            betas: qmc.Vector[qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = superposition_vector(n)
            q = qaoa_layers(p, quad, linear, q, gammas, betas)
            return qmc.measure(q)

        tr = QiskitTranspiler()
        exe = tr.transpile(
            kernel,
            bindings={
                "p": 2,
                "quad": {(0, 1): 0.5},
                "linear": {0: 0.1},
                "n": 2,
            },
            parameters=["gammas", "betas"],
        )
        assert exe.compiled_quantum[0].circuit.num_parameters >= 2

    def test_concrete_array_binding_passes(self):
        """When ``gamma`` is bound concretely, shape is folded → no error."""

        @qmc.qkernel
        def kernel(
            n: qmc.UInt,
            gamma: qmc.Vector[qmc.Float],
            hamiltonian: qmc.Observable,
        ) -> qmc.Float:
            q = superposition_vector(n)
            for i in qmc.range(gamma.shape[0]):
                q = x_mixer(q, gamma[i])
            return qmc.expval(q, hamiltonian)

        H = _make_h()
        tr = QiskitTranspiler()
        exe = tr.transpile(
            kernel,
            bindings={
                "n": H.num_qubits,
                "hamiltonian": H,
                "gamma": [0.3, 0.5],
            },
        )
        circuit = exe.compiled_quantum[0].circuit
        # 2 H (init) + 2 * 2 = 4 Rx from x_mixer(2*beta) unrolled for 2 layers
        assert circuit.size() >= 2
        assert circuit.num_qubits == H.num_qubits

    def test_compile_time_bound_select_width_passes(self):
        """A UInt SELECT width supplied in bindings reaches lowering."""

        @qmc.qkernel
        def identity(target: qmc.Qubit) -> qmc.Qubit:
            """Return a SELECT target unchanged."""
            return target

        @qmc.qkernel
        def flipped(target: qmc.Qubit) -> qmc.Qubit:
            """Apply X to a SELECT target."""
            return qmc.x(target)

        @qmc.qkernel
        def kernel(width: qmc.UInt) -> qmc.Bit:
            """Apply SELECT with a compile-time-bound structural width."""
            index = qmc.qubit_array(2, "index")
            target = qmc.qubit("target")
            index, target = qmc.select(
                [identity, flipped],
                num_index_qubits=width,
            )(index, target)
            return qmc.measure(target)

        executable = QiskitTranspiler().transpile(
            kernel,
            bindings={"width": 2},
        )
        assert executable.get_first_circuit() is not None


class TestIfMergeStructure:
    """Unknown classical branches preserve only genuine bound dependencies."""

    def test_distinct_yields_track_runtime_condition_source(self) -> None:
        """A structural merge of different constants depends on its condition."""
        restored = deserialize(serialize(_runtime_if_distinct_structural_bound))

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(restored, parameters=["flag"])

        assert type(exc_info.value) is QamomileCompileError
        message = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved" in message
        assert "runtime parameter 'flag'" in message

    def test_equal_constant_yields_are_not_structural(self) -> None:
        """Equal branch constants make the merged loop bound unconditional."""
        restored = deserialize(serialize(_runtime_if_equal_structural_bound))

        executable = QiskitTranspiler().transpile(restored, parameters=["flag"])

        assert executable.get_first_circuit() is not None

    def test_identity_yields_are_not_structural(self) -> None:
        """Identity branch yields do not make a constant bound conditional."""
        restored = deserialize(serialize(_runtime_if_identity_structural_bound))

        executable = QiskitTranspiler().transpile(restored, parameters=["flag"])

        assert executable.get_first_circuit() is not None


class TestVectorKeyStructuralShape:
    """Bound vector-key dimensions are item-loop structural variables."""

    def test_variable_key_lengths_drive_nested_loop_bounds(self) -> None:
        """Each vector key supplies its own concrete nested-loop trip count."""
        restored = deserialize(serialize(_serialized_vector_key_shape_range))

        executable = QiskitTranspiler().transpile(
            restored,
            bindings={"items": {(0, 1): 1.0, (2, 3, 4): 2.0}},
        )

        assert executable.get_first_circuit() is not None

    def test_vector_key_length_drives_select_width(self) -> None:
        """A bound vector-key dimension remains valid SELECT structure."""
        restored = deserialize(serialize(_serialized_vector_key_shape_select))

        executable = QiskitTranspiler().transpile(
            restored,
            bindings={"items": {(0, 1): 1.0, (2, 3): 2.0}},
        )

        assert executable.get_first_circuit() is not None

    def test_vector_key_length_drives_control_width(self) -> None:
        """A bound vector-key dimension remains valid controlled structure."""
        restored = deserialize(serialize(_serialized_vector_key_shape_control))

        executable = QiskitTranspiler().transpile(
            restored,
            bindings={"items": {(0, 1): 1.0, (2, 3): 2.0}},
        )

        assert executable.get_first_circuit() is not None


class TestUnknownTripRegionResult:
    """Unknown loop cardinality keeps zero-trip initializer provenance."""

    def test_items_result_retains_runtime_initializer_source(self) -> None:
        """A post-loop bound includes the possible zero-trip init value."""
        restored = deserialize(serialize(_serialized_unknown_items_region_result))

        with pytest.raises(QamomileCompileError) as exc_info:
            QiskitTranspiler().transpile(
                restored,
                parameters=["initial", "items"],
            )

        assert type(exc_info.value) is QamomileCompileError
        message = str(exc_info.value)
        assert "Cannot unroll loop: bounds could not be resolved" in message
        assert "runtime parameter 'initial'" in message
