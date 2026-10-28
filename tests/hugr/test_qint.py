"""Verify direct HUGR QInt measurement and quantum callable ownership."""

from __future__ import annotations

from collections import Counter

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.ir.operation.gate import MeasureQIntOperation
from qamomile.circuit.ir.types import QUIntType, UIntType
from qamomile.circuit.ir.value import Value
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler.errors import EmitError
from qamomile.hugr import (
    HugrExecutable,
    HugrExecutor,
    HugrTranspiler,
    SeleneExecutionOptions,
)
from qamomile.hugr.lowerer import _lower_measure_qint

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

from hugr import ops
from hugr.package import Package
from hugr.std.float import FloatVal

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _bound_qint(width: qmc.UInt) -> qmc.UInt:
    """Measure an all-ones register whose width is fixed by a binding.

    Args:
        width (qmc.UInt): Compile-time non-negative register width.

    Returns:
        qmc.UInt: Integer ``2**width - 1``, including zero for an empty register.
    """
    register = qmc.qubit_array(width, "register")
    for index in qmc.range(width):
        register[index] = qmc.x(register[index])
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _ordered_qint() -> qmc.UInt:
    """Set the first two of three carriers to distinguish bit order.

    Returns:
        qmc.UInt: Integer three from the weights ``2**0 + 2**1``.
    """
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    return qmc.measure(qmc.cast(register, qmc.QInt))


@qmc.qkernel
def _pack_qint(register: qmc.Vector[qmc.Qubit]) -> qmc.QInt:
    """Return a caller-owned vector through the QInt callable ABI.

    Args:
        register (qmc.Vector[qmc.Qubit]): Quantum register in carrier order.

    Returns:
        qmc.QInt: Packed register with its first carrier as the least significant bit.
    """
    return qmc.cast(register, qmc.QInt)


@qmc.qkernel
def _identity_qint(value: qmc.QInt) -> qmc.QInt:
    """Forward a QInt without consuming its individual carriers.

    Args:
        value (qmc.QInt): Caller-owned packed quantum register.

    Returns:
        qmc.QInt: Unmodified input register.
    """
    return value


@qmc.qkernel
def _measure_qint(value: qmc.QInt) -> qmc.UInt:
    """Consume every carrier received through a QInt function input.

    Args:
        value (qmc.QInt): Packed quantum register to measure.

    Returns:
        qmc.UInt: Unsigned integer decoded from the measured carriers.
    """
    return qmc.measure(value)


@qmc.qkernel
def _qint_with_tag(value: qmc.QInt, tag: qmc.UInt) -> tuple[qmc.QInt, qmc.UInt]:
    """Forward a symbolic packed width alongside a runtime classical argument.

    Args:
        value (qmc.QInt): Caller-owned register with a potentially symbolic width.
        tag (qmc.UInt): Runtime unsigned label to retain.

    Returns:
        tuple[qmc.QInt, qmc.UInt]: Unmodified register and unsigned label.
    """
    return value, tag


@qmc.qkernel
def _serialized_bound_qint(width: qmc.UInt, tag: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
    """Keep width bindings connected through serialized quantum call boundaries.

    Args:
        width (qmc.UInt): Compile-time non-negative register width.
        tag (qmc.UInt): Runtime unsigned label forwarded through the helper.

    Returns:
        tuple[qmc.UInt, qmc.UInt]: Measured integer ``2**width - 1`` and the label.
    """
    register = qmc.qubit_array(width, "register")
    for index in qmc.range(width):
        register[index] = qmc.x(register[index])
    number, forwarded_tag = _qint_with_tag(qmc.cast(register, qmc.QInt), tag)
    return _measure_qint(number), forwarded_tag


@qmc.qkernel
def _called_qint(width: qmc.UInt) -> qmc.UInt:
    """Pass a register through vector, QInt, and measurement call boundaries.

    Args:
        width (qmc.UInt): Compile-time non-negative register width.

    Returns:
        qmc.UInt: Integer ``2**width - 1`` decoded after the three helper calls.
    """
    register = qmc.qubit_array(width, "register")
    for index in qmc.range(width):
        register[index] = qmc.x(register[index])
    return _measure_qint(_identity_qint(_pack_qint(register)))


@qmc.qkernel
def _allocate_qint() -> qmc.QInt:
    """Return newly allocated qubits as a packed integer.

    Returns:
        qmc.QInt: Newly allocated three-qubit register initialized to zero.
    """
    return qmc.cast(qmc.qubit_array(3, "unused"), qmc.QInt)


@qmc.qkernel
def _discarded_qint() -> qmc.UInt:
    """Leave a returned QInt unused so the caller must release its carriers.

    Returns:
        qmc.UInt: Constant seven, independent of the unused quantum register.
    """
    _unused = _allocate_qint()
    return qmc.uint(7)


@qmc.qkernel
def _slice_qint(register: qmc.Vector[qmc.Qubit]) -> qmc.QInt:
    """Return odd-indexed carriers and release the other input qubits.

    Args:
        register (qmc.Vector[qmc.Qubit]): Caller-owned quantum register.

    Returns:
        qmc.QInt: Packed register containing indices 1, 3, and subsequent odd slots.
    """
    return qmc.cast(register[1::2], qmc.QInt)


@qmc.qkernel
def _called_slice_qint() -> qmc.UInt:
    """Decode a strided QInt returned across a function boundary.

    Returns:
        qmc.UInt: Integer three from set slice positions zero and one.
    """
    register = qmc.qubit_array(6, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    register[3] = qmc.x(register[3])
    return _measure_qint(_identity_qint(_slice_qint(register)))


@qmc.qkernel
def _measured_slice_qint() -> tuple[qmc.UInt, qmc.Bit]:
    """Measure a packed slice followed by a disjoint remaining root qubit.

    Returns:
        tuple[qmc.UInt, qmc.Bit]: Integer three from set slice positions zero and
        one, and true from the separately measured root position zero.
    """
    register = qmc.qubit_array(6, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    register[3] = qmc.x(register[3])
    value = qmc.measure(qmc.cast(register[1::2], qmc.QInt))
    return value, qmc.measure(register[0])


@qmc.qkernel
def _runtime_merged_qint(width: qmc.UInt) -> qmc.UInt:
    """Attempt a packed-register merge across a measurement-driven branch.

    Args:
        width (qmc.UInt): Compile-time non-negative register width.

    Returns:
        qmc.UInt: Measured integer zero on engines supporting this register merge.
    """
    register = qmc.qubit_array(width, "register")
    selector = qmc.measure(qmc.qubit("selector"))
    if selector:
        value = qmc.cast(register, qmc.QInt)
    else:
        value = qmc.cast(register, qmc.QInt)
    return qmc.measure(value)


@qmc.qkernel
def _bound_branch_qint(selector: qmc.UInt) -> qmc.UInt:
    """Select a packed carrier alias using a compile-time branch binding.

    Args:
        selector (qmc.UInt): Compile-time branch selector, zero or one.

    Returns:
        qmc.UInt: Integer three from the weights ``2**0 + 2**1`` in either branch.
    """
    register = qmc.qubit_array(3, "register")
    register[0] = qmc.x(register[0])
    register[1] = qmc.x(register[1])
    if selector:
        value = qmc.cast(register, qmc.QInt)
    else:
        value = qmc.cast(register, qmc.QInt)
    return qmc.measure(value)


def _operation_counts(package: Package) -> Counter[str]:
    """Count extension operations across every HUGR function.

    Args:
        package (Package): HUGR package whose modules are inspected.

    Returns:
        Counter[str]: Node counts keyed by canonical extension operation name.
    """
    return Counter(
        str(name())
        for module in package.modules
        for _, data in module.nodes()
        if callable(name := getattr(data.op, "name", None))
    )


@pytest.mark.parametrize("width", [0, 1, 3, 64])
@pytest.mark.parametrize("method", ["compile", "transpile"])
def test_qint_supported_widths_validate_without_quantum_simulation(
    width: int, method: str
) -> None:
    """Full-width graphs validate with one destructive measurement per qubit."""
    compiled = getattr(HugrTranspiler(), method)(_bound_qint, bindings={"width": width})
    counts = _operation_counts(compiled.artifact)
    assert counts["tket.quantum.QAlloc"] == width
    assert counts["tket.quantum.Measure"] == width
    assert counts["tket.quantum.QFree"] == width
    assert isinstance(compiled.abi.output_values[0].type, UIntType)
    assert not any("float" in name for name in counts)
    assert not any(
        isinstance(data.op, ops.Const) and isinstance(data.op.val, FloatVal)
        for _, data in compiled.artifact.modules[0].nodes()
    )


@pytest.mark.parametrize("method", ["compile", "transpile"])
def test_qint_width_65_is_rejected_by_both_public_entrypoints(method: str) -> None:
    """The diagnostic identifies the target-specific integer carrier limit."""
    with pytest.raises(EmitError, match="HUGR QInt.*0 to 64.*65"):
        getattr(HugrTranspiler(), method)(_bound_qint, bindings={"width": 65})


@pytest.mark.parametrize("method", ["compile", "transpile"])
def test_qint_runtime_width_is_rejected_by_both_public_entrypoints(
    method: str,
) -> None:
    """A dynamic register width cannot silently turn into an empty QInt."""
    with pytest.raises(EmitError, match="static dimensions|resolved bit width"):
        getattr(HugrTranspiler(), method)(_bound_qint, parameters=["width"])


def test_unresolved_qint_width_is_not_an_empty_carrier() -> None:
    """Missing structural information stays distinct from a known zero width."""
    width = Value(type=UIntType(), name="width")
    operand = Value(type=QUIntType(width=width), name="register")
    result = Value(type=UIntType(), name="result")
    operation = MeasureQIntOperation(operands=[operand], results=[result])
    with pytest.raises(EmitError, match="resolved bit width.*bindings"):
        _lower_measure_qint(operation, object(), {operand.uuid: []}, {})


@pytest.mark.parametrize("width,carrier_size", [(0, 1), (1, 0), (3, 2)])
def test_qint_measurement_rejects_a_carrier_width_mismatch(
    width: int, carrier_size: int
) -> None:
    """Declared widths cannot truncate carriers or reinterpret missing qubits as zero."""
    operand = Value(type=QUIntType(width=width), name="register")
    result = Value(type=UIntType(), name="result")
    operation = MeasureQIntOperation(operands=[operand], results=[result])
    carrier = [object() for _ in range(carrier_size)]
    with pytest.raises(EmitError, match="width disagrees with its qubit carrier"):
        _lower_measure_qint(operation, object(), {operand.uuid: carrier}, {})


@pytest.mark.parametrize("width", [0, 3])
def test_runtime_qint_merges_report_the_unsupported_carrier_layout(width: int) -> None:
    """Unsupported packed conditional ports produce a direct capability diagnostic."""
    with pytest.raises(
        EmitError, match="runtime conditionals do not support QInt register merges"
    ):
        HugrTranspiler().compile(_runtime_merged_qint, bindings={"width": width})


@pytest.mark.parametrize("width", [0, 1, 3])
def test_bound_qint_widths_execute_on_selene(tmp_path, width: int) -> None:
    """Concrete empty and nonempty register widths execute as Python integers."""
    executable = HugrTranspiler().transpile(_bound_qint, bindings={"width": width})
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=11))
    actual = executable.run(executor).result()
    assert actual == (1 << width) - 1
    assert type(actual) is int


@pytest.mark.parametrize("width", [0, 1, 3])
def test_qint_callable_abi_executes_on_selene(tmp_path, width: int) -> None:
    """Packing, forwarding, and consuming QInts retain every linear carrier."""
    executable = HugrTranspiler().transpile(_called_qint, bindings={"width": width})
    counts = _operation_counts(executable.artifact)
    assert counts["tket.quantum.QAlloc"] == width
    assert counts["tket.quantum.Measure"] == width
    assert counts["tket.quantum.QFree"] == width
    assert (
        sum(
            isinstance(data.op, ops.Call)
            for _, data in executable.artifact.modules[0].nodes()
        )
        == 3
    )
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=13))
    assert executable.run(executor).result() == (1 << width) - 1


@pytest.mark.parametrize("selector", [0, 1])
def test_compile_time_qint_branches_execute_on_selene(tmp_path, selector: int) -> None:
    """Both bound branches resolve before unsupported runtime packed merging."""
    executable = HugrTranspiler().transpile(
        _bound_branch_qint, bindings={"selector": selector}
    )
    assert not any(
        isinstance(data.op, ops.Conditional)
        for _, data in executable.artifact.modules[0].nodes()
    )
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=23))
    assert executable.run(executor).result() == 3


@pytest.mark.parametrize(
    "kernel,expected,allocations,measurements",
    [
        (_ordered_qint, 3, 3, 3),
        (_discarded_qint, 7, 3, 0),
        (_called_slice_qint, 3, 6, 3),
        (_measured_slice_qint, (3, True), 6, 4),
    ],
)
def test_qint_bit_order_and_residual_carriers_execute_on_selene(
    tmp_path, kernel, expected, allocations: int, measurements: int
) -> None:
    """Ordered slices, returned carriers, and untouched roots release exactly once."""
    executable = HugrTranspiler().transpile(kernel)
    counts = _operation_counts(executable.artifact)
    assert counts["tket.quantum.QAlloc"] == allocations
    assert counts["tket.quantum.Measure"] == measurements
    assert counts["tket.quantum.QFree"] == allocations
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=17))
    assert executable.run(executor).result() == expected


def test_qint_serialization_roundtrips_preserve_callable_integer_execution(
    tmp_path,
) -> None:
    """Qamomile and HUGR serialization preserve QInt types, calls, and bit order."""
    from dataclasses import replace

    restored_kernel = deserialize(serialize(_called_slice_qint))
    transpiler = HugrTranspiler()
    compiled = transpiler.compile(restored_kernel)
    restored_package = Package.from_bytes(compiled.artifact.to_bytes())
    transpiler.target.validate(restored_package)
    executable = HugrExecutable(replace(compiled, artifact=restored_package))
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path, seed=19))
    assert executable.run(executor).result() == 3
    assert executable.sample(executor, shots=3).result().results == [(3, 3)]


@pytest.mark.parametrize("method", ["compile", "transpile"])
@pytest.mark.parametrize("width", [0, 1, 3])
def test_serialized_qint_calls_resolve_bindings_and_keep_runtime_parameters(
    tmp_path, method, width
):
    """Post-deserialization width bindings preserve both quantum and classical ABIs."""
    restored = deserialize(serialize(_serialized_bound_qint))
    compiled = getattr(HugrTranspiler(), method)(
        restored, bindings={"width": width}, parameters=["tag"]
    )
    executable = HugrExecutable(compiled) if method == "compile" else compiled
    executor = HugrExecutor(options=SeleneExecutionOptions(build_dir=tmp_path))
    for tag in [2**63, 2**64 - 1]:
        expected = ((1 << width) - 1, tag)
        assert executable.run(executor, bindings={"tag": tag}).result() == expected
        assert executable.sample(
            executor, shots=2, bindings={"tag": tag}
        ).result().results == [(expected, 2)]


@pytest.mark.parametrize("width", [64, 65])
def test_serialized_qint_calls_keep_the_hugr_width_boundary(width):
    """Serialized bound call graphs preserve full width and reject overflow."""
    restored = deserialize(serialize(_serialized_bound_qint))
    if width == 65:
        with pytest.raises(EmitError, match="HUGR QInt.*0 to 64.*65"):
            HugrTranspiler().compile(restored, bindings={"width": width, "tag": 0})
    else:
        compiled = HugrTranspiler().compile(
            restored, bindings={"width": width, "tag": 0}
        )
        assert _operation_counts(compiled.artifact)["tket.quantum.Measure"] == width
