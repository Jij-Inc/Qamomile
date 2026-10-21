"""Exercise device shot context, parameter selection, and typed classical results."""

from __future__ import annotations

import math
from decimal import Decimal

import numpy as np
import pytest

pytest.importorskip("hugr")
pytest.importorskip("tket_exts")

import qamomile.circuit as qmc
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.transpiler import ExecutionError, JobSnapshot
from qamomile.hugr import (
    HugrExecutor,
    HugrTranspiler,
    NexusExecutionOptions,
    SeleneExecutionOptions,
)
from qamomile.hugr._qir import to_h2_qir
from qamomile.hugr._runtime import prepare_execution

pytestmark = pytest.mark.hugr


@qmc.qkernel
def _indices() -> tuple[qmc.UInt, qmc.UInt, qmc.UInt]:
    """Return stable repeated reads and an arithmetic expression."""
    a = qmc.get_current_shot()
    b = qmc.get_current_shot()
    return a, b, a * 2 + 3


@qmc.qkernel
def _conditional_shot() -> qmc.Bit:
    """Flip exactly the second repetition."""
    q = qmc.qubit("q")
    if qmc.get_current_shot() == 1:
        q = qmc.x(q)
    return qmc.measure(q)


@qmc.qkernel
def _read_table(values: qmc.Vector[qmc.UInt]) -> qmc.UInt:
    """Return a UInt selected from one runtime table."""
    return values[qmc.get_current_shot()]


@qmc.qkernel
def _read_bits(values: qmc.Vector[qmc.Bit]) -> qmc.Bit:
    """Return a dynamically selected classical Bit."""
    return values[qmc.get_current_shot()]


@qmc.qkernel
def _angle_table(angles: qmc.Vector[qmc.Float]) -> qmc.Bit:
    """Select an angle with an explicit bounds guard for QIR inspection."""
    shot = qmc.get_current_shot()
    theta = qmc.float_(0.0)
    if shot < angles.shape[0]:
        theta = angles[shot]
    q = qmc.qubit("q")
    q = qmc.rx(q, theta)
    return qmc.measure(q)


@qmc.qkernel
def _unguarded_mask(masks: qmc.Vector[qmc.UInt]) -> qmc.Bit:
    """Apply and decode a mask while retaining a runtime bounds failure."""
    m = masks[qmc.get_current_shot()]
    q = qmc.qubit("q")
    if m == 1:
        q = qmc.x(q)
    return qmc.measure(q) != m


@qmc.qkernel
def _masked_proposal(n: qmc.UInt, masks: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
    """Apply application masks and decode each measured bit in the kernel."""
    shot = qmc.get_current_shot()
    qs = qmc.qubit_array(n, "qs")
    qs[0] = qmc.x(qs[0])
    for i in qmc.range(n):
        index = shot * n + i
        if index < masks.shape[0]:
            m = masks[index]
            if m == 1:
                qs[i] = qmc.x(qs[i])
    bits = qmc.measure(qs)
    for i in qmc.range(n):
        index = shot * n + i
        mask = qmc.uint(0)
        if index < masks.shape[0]:
            mask = masks[index]
        bits[i] = bits[i] != mask
    return bits


@qmc.qkernel
def _nested_shot(q: qmc.Qubit) -> qmc.Qubit:
    """Read device context in an ordinary called qkernel."""
    if qmc.get_current_shot() == 1:
        q = qmc.x(q)
    return q


@qmc.qkernel
def _nested_entry() -> qmc.Bit:
    """Preserve the shot index across a qkernel call."""
    q = qmc.qubit("q")
    q = _nested_shot(q)
    return qmc.measure(q)


@qmc.qkernel
def _guarded_bound_table(values: qmc.Vector[qmc.UInt]) -> qmc.UInt:
    """Select a compile-time table inside a runtime branch."""
    shot = qmc.get_current_shot()
    result = qmc.uint(7)
    if shot < values.shape[0]:
        result = values[shot]
    return result


@qmc.qkernel
def _decode_loop(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Carry updates to a measured vector, including a zero-trip loop."""
    qs = qmc.qubit_array(3, "qs")
    qs[0] = qmc.x(qs[0])
    bits = qmc.measure(qs)
    for i in qmc.range(repetitions):
        bits[1] = bits[0] != bits[1]
        bits[2] = bits[1]
    return bits


@qmc.qkernel
def _saved_before_loop(repetitions: qmc.UInt) -> tuple[qmc.Bit, qmc.Vector[qmc.Bit]]:
    """Keep a scalar snapshot while repeated stores update its source vector."""
    qs = qmc.qubit_array(1, "qs")
    bits = qmc.measure(qs)
    old = bits[0]
    for i in qmc.range(repetitions):
        bits[0] = ~bits[0]
    return old, bits


@qmc.qkernel
def _snapshot_used_in_loop(repetitions: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """Read a captured element alongside the current mutable vector."""
    qs = qmc.qubit_array(2, "qs")
    bits = qmc.measure(qs)
    old = bits[0]
    for i in qmc.range(repetitions):
        bits[0] = ~old
        bits[1] = bits[0]
    return bits


@qmc.qkernel
def _nested_snapshots(repetitions: qmc.UInt) -> tuple[qmc.Bit, qmc.Vector[qmc.Bit]]:
    """Capture a fresh element before each inner loop and retain an outer copy."""
    qs = qmc.qubit_array(2, "qs")
    bits = qmc.measure(qs)
    old = bits[0]
    for i in qmc.range(repetitions):
        inner_old = bits[0]
        for j in qmc.range(2):
            bits[0] = ~inner_old
        bits[1] = old
    return old, bits


@qmc.qkernel
def _read_float_table(values: qmc.Vector[qmc.Float]) -> qmc.Float:
    """Return a Float selected from a bound table."""
    return values[qmc.get_current_shot()]


def _executor():
    """Construct a deterministic local simulator."""
    pytest.importorskip("selene_sim")
    return HugrExecutor(options=SeleneExecutionOptions(seed=5))


def _table_program(kernel, name, size, **bindings):
    """Compile fixed structure while leaving table contents as a runtime argument."""
    return HugrTranspiler().transpile(
        kernel, bindings=bindings, parameters=[name], parameter_shapes={name: (size,)}
    )


def test_shot_sequence_restarts_with_each_sample_and_run():
    """A cached local runner restarts at zero and repeated reads remain stable."""
    p = HugrTranspiler().transpile(_indices)
    executor = _executor()
    for shots in [4, 2, 1]:
        assert p.sample(executor, shots=shots).result().results == [
            ((i, i, 2 * i + 3), 1) for i in range(shots)
        ]
    assert p.run(executor).result() == (0, 0, 3)


@pytest.mark.parametrize("kernel", [_conditional_shot, _nested_entry])
def test_shot_controls_gates_through_called_kernels(kernel):
    """Only shot one flips the qubit, including inside a called kernel."""
    p = HugrTranspiler().transpile(kernel)
    assert dict(p.sample(_executor(), shots=5).result().results) == {False: 4, True: 1}


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("n,shots", [(1, 1), (2, 4), (3, 3)])
def test_masked_vector_returns_decoded_counts(seed, n, shots):
    """Independent application masks preserve the proposal's decoded bit vector."""
    masks = np.random.default_rng(seed).integers(0, 2, size=(shots, n)).ravel().tolist()
    p = _table_program(_masked_proposal, "masks", len(masks), n=n)
    before = p._compiled.artifact.to_bytes()
    result = p.sample(_executor(), shots=shots, bindings={"masks": masks}).result()
    assert result.results == [((True, *(False for _ in range(n - 1))), shots)]
    assert p._compiled.artifact.to_bytes() == before


@pytest.mark.parametrize(
    "kernel,values",
    [(_read_table, [9, 2, 9, 7])],
)
def test_runtime_table_contents_are_not_replaced_by_a_sweep(kernel, values):
    """One parameter table is bound once, with runtime selection across repetitions."""
    p = _table_program(kernel, "values", len(values))
    executor = _executor()
    for table in [values, list(reversed(values))]:
        result = p.sample(executor, shots=4, bindings={"values": table}).result()
        expected = {}
        for value in table:
            expected[value] = expected.get(value, 0) + 1
        assert dict(result.results) == expected


def test_local_runtime_float_selection_and_fixed_bindings():
    """Helios-compatible HUGR supports shot-selected angles and bound tables."""
    p = _table_program(_angle_table, "angles", 2)
    assert p.sample(
        _executor(), shots=2, bindings={"angles": [0.0, math.pi]}
    ).result().results == [(False, 1), (True, 1)]
    fixed = HugrTranspiler().transpile(_conditional_shot)
    assert dict(fixed.sample(_executor(), shots=3).result().results) == {
        False: 2,
        True: 1,
    }


def test_unguarded_lookup_fails_out_of_range_without_partial_counts():
    """Selene reports a bounds failure instead of wrapping or returning partial counts."""
    p = _table_program(_unguarded_mask, "masks", 2)
    assert p.sample(
        _executor(), shots=2, bindings={"masks": [0, 1]}
    ).result().results == [(False, 2)]
    with pytest.raises(Exception, match="(?i)(panic|range|shot|exit|error|return)"):
        p.sample(_executor(), shots=3, bindings={"masks": [0, 1]}).result()


def test_serialized_kernel_keeps_device_context():
    """Static persistence roundtrips the abstract read without baking in a shot."""
    restored = deserialize(serialize(_conditional_shot))
    p = HugrTranspiler().transpile(restored)
    assert dict(p.sample(_executor(), shots=3).result().results) == {False: 2, True: 1}


@pytest.mark.parametrize("kernel", [_conditional_shot, _nested_entry])
def test_h2_shot_conditional_produces_validated_qir(kernel):
    """The device call survives conversion through the tagged void entrypoint."""
    pyqir = pytest.importorskip("pyqir")
    pytest.importorskip("hugr_qir")
    prepared = prepare_execution(HugrTranspiler().compile(kernel))
    bitcode, outputs = to_h2_qir(prepared.package)
    llvm = str(pyqir.Module.from_bitcode(pyqir.Context(), bitcode))
    assert "call i64 @___get_current_shot()" in llvm
    assert "insertvalue" not in llvm
    assert "freeze" not in llvm
    assert set(outputs) == set(prepared.output_tags)


@pytest.mark.parametrize("n,shots", [(1, 2), (2, 4), (3, 3)])
def test_h2_masked_proposal_has_validated_qir(n, shots):
    """Explicit bounds guards let integer mask selection target validated H2 QIR."""
    pytest.importorskip("hugr_qir")
    p = _table_program(_masked_proposal, "masks", shots * n, n=n)
    masks = np.random.default_rng(13).integers(0, 2, shots * n).tolist()
    prepared = prepare_execution(p._compiled, {"masks": masks})
    bitcode, outputs = to_h2_qir(prepared.package)
    assert bitcode.startswith(b"BC\xc0\xde")
    assert len(outputs) == n


def test_h2_unguarded_lookup_has_actionable_converter_error():
    """The pinned converter cannot lower bounds panic to valid H2 QIR."""
    pytest.importorskip("hugr_qir")
    p = _table_program(_unguarded_mask, "masks", 2)
    with pytest.raises(
        ExecutionError, match=r"Guard table access.*index < table.shape"
    ):
        to_h2_qir(prepare_execution(p._compiled, {"masks": [0, 1]}).package)


def test_h2_float_table_folds_into_constant_angle_branches():
    """Selected constants may target H2 when conversion eliminates runtime Floats."""
    pytest.importorskip("hugr_qir")
    p = _table_program(_angle_table, "angles", 2)
    bitcode, _ = to_h2_qir(
        prepare_execution(p._compiled, {"angles": [0.3, 1.7]}).package
    )
    assert bitcode.startswith(b"BC\xc0\xde")


@pytest.mark.parametrize(
    "instruction",
    [
        "%x = fadd double 1.0, 2.0",
        "%x = udiv i64 7, 3",
        "%x = urem i64 7, 3",
    ],
)
def test_h2_additional_checks_reject_helios_only_instructions(instruction):
    """The shared validator accepts instructions outside the narrower H2 profile."""
    pyqir = pytest.importorskip("pyqir")
    from qamomile.hugr._qir import _validate_h2_classical_instructions

    module = pyqir.Module.from_ir(
        pyqir.Context(), f"define void @main() {{ {instruction}\n ret void }}"
    )
    with pytest.raises(ExecutionError, match="H2 does not support runtime"):
        _validate_h2_classical_instructions(module.bitcode)


@pytest.mark.parametrize(
    "kernel,values", [(_read_bits, [False, True]), (_read_table, [5, 9])]
)
def test_bound_classical_tables_keep_contents_with_runtime_indices(kernel, values):
    """Compile-time table bindings are preserved when only the index is dynamic."""
    p = HugrTranspiler().transpile(kernel, bindings={"values": values})
    assert p.sample(_executor(), shots=2).result().results == [
        (values[0], 1),
        (values[1], 1),
    ]


def test_h2_typed_snapshot_restores_decoded_results_without_submission(monkeypatch):
    """Decoded Bit vectors use existing labels and restoration, with one submission."""
    pytest.importorskip("hugr_qir")
    pytest.importorskip("pytket")
    from tests.hugr.test_h2_qir import _install_h2_client

    tags = ["qamomile.output.0.0", "qamomile.output.0.1"]
    client = _install_h2_client(monkeypatch, tags, [[1, 0]] * 3)
    p = _table_program(_masked_proposal, "masks", 6, n=2)
    bindings = {"masks": [0, 1, 1, 0, 1, 1]}
    executor = HugrExecutor("nexus", options=NexusExecutionOptions(system_name="H2-1E"))
    job = p.sample(executor, shots=3, bindings=bindings)
    assert job.result().results == [((True, False), 3)]
    saved = JobSnapshot.from_dict(job.snapshot().to_dict())
    assert p.restore(executor, saved, bindings).result().results == [((True, False), 3)]
    client.qir.upload.assert_called_once()
    client.start_execute_job.assert_called_once()
    assert client.start_execute_job.call_args.kwargs["n_shots"] == [3]


@pytest.mark.parametrize("repetitions", [0, 1, 2, 3])
def test_classical_decode_loop_threads_latest_array_version(repetitions):
    """Persistent Bit updates retain earlier stores and the zero-trip initializer."""
    p = HugrTranspiler().transpile(_decode_loop, bindings={"repetitions": repetitions})
    expected = (True, bool(repetitions % 2), bool(repetitions % 2))
    assert p.sample(_executor(), shots=2).result().results == [(expected, 2)]


def test_bound_table_is_available_inside_runtime_branch():
    """A bound table is a region-local constant, while its index remains a capture."""
    p = HugrTranspiler().transpile(_guarded_bound_table, bindings={"values": [3, 5]})
    assert p.sample(_executor(), shots=3).result().results == [(3, 1), (5, 1), (7, 1)]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("repetitions", [0, 1, 2, 3])
def test_loop_stores_preserve_pre_loop_element_snapshot(repetitions, serialized):
    """A saved element retains its initial value after every loop trip count."""
    kernel = (
        deserialize(serialize(_saved_before_loop)) if serialized else _saved_before_loop
    )
    p = HugrTranspiler().transpile(kernel, bindings={"repetitions": repetitions})
    expected = (False, (bool(repetitions % 2),))
    assert p.sample(_executor(), shots=2).result().results == [(expected, 2)]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("repetitions", [0, 1, 2, 3])
def test_loop_body_reads_snapshot_and_latest_store_independently(
    repetitions, serialized
):
    """A captured scalar stays fixed while body element reads see earlier stores."""
    kernel = (
        deserialize(serialize(_snapshot_used_in_loop))
        if serialized
        else _snapshot_used_in_loop
    )
    p = HugrTranspiler().transpile(kernel, bindings={"repetitions": repetitions})
    expected = (repetitions > 0, repetitions > 0)
    assert p.sample(_executor(), shots=2).result().results == [(expected, 2)]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("runtime", [False, True])
@pytest.mark.parametrize(
    "kernel,values",
    [
        (_read_table, [0, 2**63 + 1, 2**64 - 1]),
        (_read_table, np.array([0, 2**63 + 1, 2**64 - 1], dtype=np.uint64)),
        (_read_float_table, [Decimal("0.125"), Decimal("0.875")]),
    ],
)
def test_classical_table_bindings_preserve_scalar_values(
    kernel, values, runtime, serialized
):
    """Flattening preserves UInt precision and accepts frontend-valid Float values."""
    if serialized:
        kernel = deserialize(serialize(kernel))
    if runtime:
        p = _table_program(kernel, "values", len(values))
        result = p.sample(_executor(), shots=len(values), bindings={"values": values})
    else:
        p = HugrTranspiler().transpile(kernel, bindings={"values": values})
        result = p.sample(_executor(), shots=len(values))
    expected = [
        float(value) if isinstance(value, Decimal) else int(value) for value in values
    ]
    assert result.result().results == [(value, 1) for value in expected]


@pytest.mark.parametrize("repetitions", [0, 1, 2, 3])
def test_nested_loop_snapshots_are_refreshed_for_each_outer_iteration(repetitions):
    """Inner captures see each iteration's input, while outer captures stay fixed."""
    p = HugrTranspiler().transpile(
        _nested_snapshots, bindings={"repetitions": repetitions}
    )
    expected = (False, (bool(repetitions % 2), False))
    assert p.sample(_executor(), shots=2).result().results == [(expected, 2)]
