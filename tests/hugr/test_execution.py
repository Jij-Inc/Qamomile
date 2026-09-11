"""Exercise the public HUGR executor with real Selene and deferred providers."""

from __future__ import annotations

import math
from typing import Any

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.transpiler import CompiledProgram
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.hugr import (
    HugrExecutable,
    HugrExecutor,
    HugrTranspiler,
    NexusExecutionOptions,
    SeleneExecutionOptions,
)

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _rotation(theta: qmc.Float) -> qmc.Bit:
    """Measure a single rotation with a retained runtime angle.

    Args:
        theta (qmc.Float): Runtime rotation angle in radians.

    Returns:
        qmc.Bit: Measured rotated qubit.
    """
    q = qmc.qubit("q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


@qmc.qkernel
def _complex(
    theta: qmc.Float, label: qmc.UInt
) -> tuple[qmc.Vector[qmc.Bit], qmc.Float, qmc.UInt]:
    """Return an ordered bit array and two computed classical outputs.

    Args:
        theta (qmc.Float): Runtime rotation angle in radians.
        label (qmc.UInt): Unsigned value used to verify native arithmetic.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Float, qmc.UInt]: Measured bits, half the rotation angle, and incremented label.
    """
    q = qmc.qubit_array(3, "q")
    q[0] = qmc.x(q[0])
    q[2] = qmc.ry(q[2], theta)
    bits = qmc.measure(q)
    return bits, theta * 0.5, label + 1


@qmc.qkernel
def _fixed() -> qmc.Float:
    """Measure little-endian fixed-point bits without host-side decoding.

    Returns:
        qmc.Float: Measured value with binary weights 2**(-2) and 2**1, totaling 2.25.
    """
    q = qmc.qubit_array(4, "q")
    q[0] = qmc.x(q[0])
    q[3] = qmc.x(q[3])
    number = qmc.cast(q, qmc.QFixed, int_bits=2)
    return qmc.measure(number)


@qmc.qkernel
def _integer() -> qmc.UInt:
    """Measure an asymmetric QInt pattern through the native integer result ABI.

    Returns:
        qmc.UInt: Integer three encoded with the first two of three bits set.
    """
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _empty_integer() -> qmc.UInt:
    """Return the integer zero from an empty quantum register.

    Returns:
        qmc.UInt: Zero as an integer rather than an empty result.
    """
    register = qmc.qubit_array(0, "register")
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _structured_integer(
    label: qmc.UInt,
) -> tuple[qmc.UInt, qmc.Vector[qmc.Bit], qmc.UInt]:
    """Return a measured QInt beside a bit vector and a full-width integer.

    Args:
        label (qmc.UInt): Unsigned label whose bits must survive unchanged.

    Returns:
        tuple[qmc.UInt, qmc.Vector[qmc.Bit], qmc.UInt]: Measured integer three,
        asymmetric bit vector, and the unchanged label.
    """
    number = _integer()
    register = qmc.qubit_array(2, "register")
    register[0] = qmc.x(register[0])
    return number, qmc.measure(register), label


@pytest.mark.ci_smoke
def test_public_sampling_reuses_parameterized_artifact(tmp_path) -> None:
    """Changing angles changes actual simulator results without recompilation."""
    executable = HugrTranspiler().transpile(_rotation, parameters=["theta"])
    assert isinstance(executable, HugrExecutable)
    original = executable.artifact.to_bytes()
    executor = HugrExecutor(options=SeleneExecutionOptions(seed=19, build_dir=tmp_path))
    for angle, expected in [(0.0, False), (math.pi, True), (2 * math.pi, False)]:
        result = executable.sample(
            executor, shots=12, bindings={"theta": angle}
        ).result()
        assert result.results == [(expected, 12)]
        assert result.shots == 12
        assert executable.artifact.to_bytes() == original
    assert not executor.capabilities.supports_native_parameter_inputs


def test_run_and_structured_sample_preserve_types_order_and_uint_width(
    tmp_path,
) -> None:
    """Multiple outputs retain bit order and 64-bit unsigned arithmetic."""
    executable = HugrTranspiler().transpile(_complex, parameters=["theta", "label"])
    assert isinstance(executable, HugrExecutable)
    executor = HugrExecutor(options=SeleneExecutionOptions(seed=17, build_dir=tmp_path))
    for label in (0, 2**32, 2**63 + 2, 2**64 - 1):
        value = executable.run(
            executor, bindings={"theta": math.pi, "label": label}
        ).result()
        assert value[0] == (True, False, True)
        assert isinstance(value[0][0], bool)
        assert value[1] == pytest.approx(math.pi / 2)
        assert value[2] == (label + 1) % (2**64)
    sampled = executable.sample(
        executor, shots=5, bindings={"theta": 0.0, "label": 7}
    ).result()
    assert sampled.results == [(((True, False, False), 0.0, 8), 5)]


def test_qfixed_runs_in_hugr_on_selene(tmp_path) -> None:
    """QFixed weights decode the low and high bit within the HUGR graph."""
    executable = HugrTranspiler().transpile(_fixed)
    executor = HugrExecutor(options=SeleneExecutionOptions(seed=13, build_dir=tmp_path))
    assert executable.run(executor).result() == pytest.approx(2.25)


@pytest.mark.parametrize(
    "kernel,expected",
    [
        (_integer, 3),
        (_empty_integer, 0),
        (_structured_integer, (3, (True, False), 2**64 - 1)),
    ],
)
def test_qint_run_and_sample_preserve_integer_results(
    tmp_path, kernel, expected
) -> None:
    """Native QInt measurements retain scalar and structured types on Selene."""
    structured = kernel is _structured_integer
    parameters = ["label"] if structured else []
    bindings = {"label": 2**64 - 1} if structured else {}
    executable = HugrTranspiler().transpile(kernel, parameters=parameters)
    executor = HugrExecutor(options=SeleneExecutionOptions(seed=29, build_dir=tmp_path))
    actual = executable.run(executor, bindings=bindings).result()
    assert actual == expected
    assert type(actual[0] if structured else actual) is int
    if structured:
        assert type(actual[1][0]) is bool
        assert type(actual[2]) is int

    sampled = executable.sample(executor, shots=4, bindings=bindings).result()
    assert sampled.results == [(expected, 4)]
    assert sampled.shots == 4
    sampled_value = sampled.results[0][0]
    assert type(sampled_value[0] if structured else sampled_value) is int


def test_compiled_program_can_be_wrapped_without_retracing() -> None:
    """The compile result remains accepted by an explicit execution facade."""
    compiled = HugrTranspiler().compile(_rotation, parameters=["theta"])
    assert isinstance(compiled, CompiledProgram)
    executable = HugrExecutable(compiled)
    assert executable.abi.public_inputs.keys() == {"theta"}
    assert executable.metadata.target == "hugr"


@pytest.mark.parametrize("shots", [True, 0, -1, 1.5])
def test_invalid_shots_fail_before_submission(shots) -> None:
    """Invalid counts are rejected before any provider action."""
    executable = HugrTranspiler().transpile(_rotation, parameters=["theta"])
    with pytest.raises(ValueError, match="shots"):
        executable.sample(HugrExecutor(), shots=shots, bindings={"theta": 0.0})


@pytest.mark.parametrize(
    "bindings",
    [{}, {"theta": 0.0, "extra": 1}, {"theta": float("nan")}, {"theta": True}],
)
def test_invalid_runtime_values_fail_before_submission(bindings) -> None:
    """Names and scalar domain are validated independently from compilation."""
    executable = HugrTranspiler().transpile(_rotation, parameters=["theta"])
    with pytest.raises((ValueError, TypeError)):
        executable.run(HugrExecutor(), bindings=bindings)


@pytest.mark.parametrize(
    "target,options",
    [("selene", NexusExecutionOptions()), ("helios", SeleneExecutionOptions())],
)
def test_wrong_destination_options_are_rejected(target, options) -> None:
    """Options cannot silently select another provider."""
    with pytest.raises(TypeError):
        HugrExecutor(target, options=options)


def test_bindings_parameters_overlap_is_rejected() -> None:
    """The public HUGR entrypoint preserves disjoint binding contracts."""
    with pytest.raises(ValueError):
        HugrTranspiler().transpile(
            _rotation, bindings={"theta": 0.1}, parameters=["theta"]
        )


@qmc.qkernel
def _bound_tuple(
    pair: qmc.Tuple[qmc.UInt, qmc.Float],
) -> qmc.Tuple[qmc.UInt, qmc.Float]:
    """Return the existing structured ABI from a compile-time tuple input.

    Args:
        pair (qmc.Tuple[qmc.UInt, qmc.Float]): Typed pair carried through the callable ABI.

    Returns:
        qmc.Tuple[qmc.UInt, qmc.Float]: Unchanged tuple value.
    """
    return pair


@qmc.qkernel
def _bound_dict(
    mapping: qmc.Dict[qmc.UInt, qmc.Float],
) -> qmc.Dict[qmc.UInt, qmc.Float]:
    """Return the existing bound-data dictionary ABI.

    Args:
        mapping (qmc.Dict[qmc.UInt, qmc.Float]): Dictionary with compile-time keys and values.

    Returns:
        qmc.Dict[qmc.UInt, qmc.Float]: Unchanged dictionary value.
    """
    return mapping


@qmc.qkernel
def _allocate() -> qmc.Qubit:
    """Allocate a helper-owned qubit returned to its caller.

    Returns:
        qmc.Qubit: Newly allocated qubit in the zero state.
    """
    return qmc.qubit("allocated")


@qmc.qkernel
def _two_allocations() -> tuple[qmc.Bit, qmc.Bit]:
    """Keep the outputs of two calls to one allocator alive together.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Measurements of zero and one from independent allocations.
    """
    left = _allocate()
    right = _allocate()
    right = qmc.x(right)
    return qmc.measure(left), qmc.measure(right)


@pytest.mark.parametrize(
    "kernel,bindings,expected",
    [
        (_bound_tuple, {"pair": (7, -0.25)}, (7, -0.25)),
        (_bound_dict, {"mapping": {0: 1.25, 3: -2.0}}, {0: 1.25, 3: -2.0}),
        (_bound_dict, {"mapping": {}}, {}),
    ],
)
def test_structural_returns_use_the_existing_public_abi(
    tmp_path, kernel, bindings, expected
) -> None:
    """Bound tuple and dictionary outputs survive actual remote-style recording."""
    executable = HugrTranspiler().transpile(kernel, bindings=bindings)
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    result = executable.sample(executor, shots=3).result()
    assert result.results == [(expected, 3)]


def test_default_capacity_counts_repeated_helper_calls(tmp_path) -> None:
    """One allocation site called twice needs two simultaneous qubit slots."""
    executable = HugrTranspiler().transpile(_two_allocations)
    assert executable.run(
        HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    ).result() == (False, True)


@pytest.mark.parametrize("label", [-1, 2**64])
def test_out_of_range_compile_time_uint_is_rejected(label) -> None:
    """Public UInt bindings cannot silently wrap outside the backend domain."""
    with pytest.raises((EmitError, ValueError, TypeError)):
        HugrTranspiler().transpile(_complex, bindings={"theta": 0.0, "label": label})


@qmc.qkernel
def _matrix_return(values: qmc.Matrix[qmc.Float]) -> qmc.Matrix[qmc.Float]:
    """Return runtime matrix contents without changing shape or element values.

    Args:
        values (qmc.Matrix[qmc.Float]): Runtime matrix with fixed structural dimensions.

    Returns:
        qmc.Matrix[qmc.Float]: Unchanged matrix contents and shape.
    """
    return values


def test_public_parameter_shapes_preserve_runtime_matrix_contents(tmp_path) -> None:
    """The public facade specializes dimensions while retaining native arguments."""
    executable = HugrTranspiler().transpile(
        _matrix_return, parameters=["values"], parameter_shapes={"values": (2, 3)}
    )
    original = executable.artifact.to_bytes()
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    for values in (
        ((0.0, 1.0, 2.0), (3.0, 4.0, 5.0)),
        ((6.0, 7.0, 8.0), (9.0, 10.0, 11.0)),
    ):
        assert executable.run(executor, bindings={"values": values}).result() == values
        assert executable.artifact.to_bytes() == original


def _remote_executor(
    monkeypatch: pytest.MonkeyPatch, records: list[list[tuple[str, Any]]]
) -> tuple[HugrExecutor, Any]:
    """Create a SDK-shaped fake that passes tagged results through the real transport.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture replacing SDK construction.
        records (list[list[tuple[str, object]]]): Result records for each shot.

    Returns:
        tuple[HugrExecutor, SimpleNamespace]: Executor and observable fake client.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from hugr.qsystem.result import QsysResult

    from qamomile.hugr._nexus import NexusTransport

    native = SimpleNamespace(id="sample-job", job_type="execute")
    client = SimpleNamespace(
        HeliosConfig=Mock(
            side_effect=lambda **kwargs: SimpleNamespace(type="HeliosConfig", **kwargs)
        ),
        hugr=SimpleNamespace(upload=Mock(return_value="uploaded-hugr")),
        start_execute_job=Mock(return_value=native),
        jobs=SimpleNamespace(
            get=Mock(return_value=native),
            cancel=Mock(),
            status=Mock(return_value=SimpleNamespace(status="COMPLETED")),
            results=Mock(
                return_value=[
                    SimpleNamespace(
                        download_result=Mock(return_value=QsysResult(records))
                    )
                ]
            ),
        ),
    )
    monkeypatch.setattr(
        "qamomile.hugr.execution._NexusTransport",
        lambda options: NexusTransport(options, client=client),
    )
    return HugrExecutor("helios"), client


@pytest.mark.parametrize("operation", ["run", "sample"])
def test_helios_typed_jobs_restore_without_resubmitting(monkeypatch, operation) -> None:
    """Both public job kinds use real ABI decoding and saved Nexus references."""
    executable = HugrTranspiler().transpile(_complex, parameters=["theta", "label"])
    record = [
        ("qamomile.output.0.0", True),
        ("qamomile.output.0.1", False),
        ("qamomile.output.0.2", True),
        ("qamomile.output.1.0", 0.5),
        ("qamomile.output.2.0", 9),
    ]
    count = 1 if operation == "run" else 3
    executor, client = _remote_executor(monkeypatch, [record] * count)
    bindings = {"theta": 1.0, "label": 8}
    job = (
        executable.run(executor, bindings=bindings)
        if operation == "run"
        else executable.sample(executor, shots=count, bindings=bindings)
    )
    client.jobs.results.assert_not_called()
    restored = executable.restore(executor, job.snapshot(), bindings=bindings)
    expected = ((True, False, True), 0.5, 9)
    if operation == "run":
        assert restored.result() == expected
    else:
        assert restored.result().results == [(expected, count)]
    client.start_execute_job.assert_called_once()
    restored.cancel()
    client.jobs.cancel.assert_called_once()
    with pytest.raises(ValueError, match="program and runtime bindings"):
        executable.restore(
            executor, job.snapshot(), bindings={"theta": 2.0, "label": 8}
        )


@pytest.mark.parametrize("operation", ["run", "sample"])
@pytest.mark.parametrize("label", [None, 2**63, 2**64 - 1])
def test_helios_qint_results_use_the_native_integer_abi(
    monkeypatch, operation, label
) -> None:
    """Local Nexus records preserve measured QInt and full-width structured UInts."""
    structured = label is not None
    kernel = _structured_integer if structured else _integer
    executable = HugrTranspiler().transpile(
        kernel, parameters=["label"] if structured else []
    )
    bindings = {"label": label} if structured else {}
    record = [("qamomile.output.0.0", 3)]
    if structured:
        record.extend(
            [
                ("qamomile.output.1.0", True),
                ("qamomile.output.1.1", False),
                ("qamomile.output.2.0", label),
            ]
        )
    count = 1 if operation == "run" else 3
    executor, client = _remote_executor(monkeypatch, [record] * count)
    job = (
        executable.run(executor, bindings=bindings)
        if operation == "run"
        else executable.sample(executor, shots=count, bindings=bindings)
    )
    expected = (3, (True, False), label) if structured else 3
    result = job.result()
    if operation == "sample":
        assert result.results == [(expected, count)]
        actual = result.results[0][0]
    else:
        actual = result
    assert actual == expected
    assert type(actual[0] if structured else actual) is int
    if structured:
        assert type(actual[2]) is int
    client.hugr.upload.assert_called_once()
    client.start_execute_job.assert_called_once()


def test_restore_rejects_equal_native_carriers_with_different_public_shapes(
    monkeypatch,
) -> None:
    """A flat tuple cannot be restored with a different matrix shape."""
    first = HugrTranspiler().transpile(
        _matrix_return, parameters=["values"], parameter_shapes={"values": (2, 3)}
    )
    second = HugrTranspiler().transpile(
        _matrix_return, parameters=["values"], parameter_shapes={"values": (3, 2)}
    )
    executor, client = _remote_executor(
        monkeypatch, [[(f"qamomile.output.0.{i}", float(i)) for i in range(6)]]
    )
    job = first.run(executor, bindings={"values": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]})
    with pytest.raises(ValueError, match="public ABI"):
        second.restore(
            executor,
            job.snapshot(),
            bindings={"values": [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]},
        )
    client.jobs.get.assert_not_called()


@qmc.qkernel
def _tuple_call(pair: qmc.Tuple[qmc.UInt, qmc.Float]) -> qmc.Tuple[qmc.UInt, qmc.Float]:
    """Carry a structural value through a helper call boundary.

    Args:
        pair (qmc.Tuple[qmc.UInt, qmc.Float]): Typed pair carried through the callable ABI.

    Returns:
        qmc.Tuple[qmc.UInt, qmc.Float]: Tuple returned by the helper.
    """
    return _bound_tuple(pair)


@qmc.qkernel
def _dict_call(mapping: qmc.Dict[qmc.UInt, qmc.Float]) -> qmc.Dict[qmc.UInt, qmc.Float]:
    """Carry a bound dictionary through a helper call boundary.

    Args:
        mapping (qmc.Dict[qmc.UInt, qmc.Float]): Dictionary with compile-time keys and values.

    Returns:
        qmc.Dict[qmc.UInt, qmc.Float]: Dictionary returned by the helper.
    """
    return _bound_dict(mapping)


@pytest.mark.parametrize(
    "kernel,bindings,expected",
    [
        (_tuple_call, {"pair": (7, 0.25)}, (7, 0.25)),
        (_dict_call, {"mapping": {1: 0.25, 3: -0.5}}, {1: 0.25, 3: -0.5}),
    ],
)
def test_compound_values_survive_helper_call_abis(
    tmp_path, kernel, bindings, expected
) -> None:
    """Both caller operands and callee results retain recursive carriers."""
    executable = HugrTranspiler().transpile(kernel, bindings=bindings)
    assert (
        executable.run(
            HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
        ).result()
        == expected
    )
