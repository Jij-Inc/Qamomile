"""Cross-backend execution tests for loop-carried classical values.

Loop-carry slots are resolved by two different executors depending on
where the loop lands: the classical segment interpreter (block-output
reductions) and emit-time unrolling (gate parameters). Both paths must
produce Python semantics on every supported SDK backend, so each case
transpiles AND executes, verifying the classical outcome value.

The quri_parts / cudaq variants are marker-gated (deselected by the
default ``uv run pytest`` run) and skip when the SDK is missing.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import qamomile.circuit as qmc

BACKENDS = [
    pytest.param("qiskit", id="qiskit"),
    pytest.param("quri_parts", marks=pytest.mark.quri_parts, id="quri_parts"),
    pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
]
RUNTIME_BACKENDS = [
    pytest.param("qiskit", id="qiskit"),
    pytest.param("cudaq", marks=pytest.mark.cudaq, id="cudaq"),
]


def _make_transpiler(backend: str) -> Any:
    """Build the requested backend transpiler, skipping when unavailable.

    Args:
        backend (str): One of ``"qiskit"`` / ``"quri_parts"`` / ``"cudaq"``.

    Returns:
        Any: The backend transpiler instance.
    """
    if backend == "qiskit":
        pytest.importorskip("qiskit")
        from qamomile.qiskit import QiskitTranspiler

        return QiskitTranspiler()
    if backend == "quri_parts":
        pytest.importorskip("quri_parts")
        pytest.importorskip("quri_parts.qulacs")
        from qamomile.quri_parts import QuriPartsTranspiler

        return QuriPartsTranspiler()
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


def _sample_single(backend: str, kernel: Any, bindings: dict[str, Any]) -> Any:
    """Transpile and sample a kernel, returning the deterministic outcome.

    Args:
        backend (str): Backend name accepted by :func:`_make_transpiler`.
        kernel (Any): The qkernel to run.
        bindings (dict[str, Any]): Compile-time bindings.

    Returns:
        Any: The single deterministic sampled outcome.
    """
    transpiler = _make_transpiler(backend)
    executable = transpiler.transpile(kernel, bindings=bindings)
    result = executable.sample(transpiler.executor(), shots=100).result()
    assert len(result.results) == 1, f"expected deterministic result: {result}"
    return result.results[0][0]


@qmc.qkernel
def sum_kernel(n: qmc.UInt) -> qmc.UInt:
    """Accumulate all indices below ``n``.

    Args:
        n (qmc.UInt): Exclusive loop bound.

    Returns:
        qmc.UInt: Sum of ``range(n)``.
    """
    q = qmc.qubit("q")
    qmc.measure(q)
    total = qmc.uint(0)
    for i in qmc.range(n):
        total = total + i
    return total


@qmc.qkernel
def swap_kernel(n: qmc.UInt) -> tuple[qmc.UInt, qmc.UInt]:
    """Swap two carried integers ``n`` times.

    Args:
        n (qmc.UInt): Number of swaps.

    Returns:
        tuple[qmc.UInt, qmc.UInt]: Final ordered pair.
    """
    q = qmc.qubit("q")
    qmc.measure(q)
    a = qmc.uint(1)
    b = qmc.uint(2)
    for _i in qmc.range(n):
        a, b = b, a
    return a, b


@qmc.qkernel
def angle_kernel(n: qmc.UInt) -> qmc.Bit:
    """Rotate by a carried sum of ``n`` pi increments.

    Args:
        n (qmc.UInt): Number of pi increments.

    Returns:
        qmc.Bit: Measurement of the rotated qubit.
    """
    total = 0.0
    for _i in qmc.range(n):
        total = total + math.pi
    q = qmc.qubit("q")
    q = qmc.rx(q, total)
    return qmc.measure(q)


@qmc.qkernel
def indexed_while_kernel() -> qmc.Bit:
    """Update two distinct measured-vector conditions inside a static loop.

    Returns:
        qmc.Bit: Deterministic zero after both runtime loops terminate.
    """
    ancilla = qmc.qubit_array(2, "ancilla")
    ancilla[0] = qmc.x(ancilla[0])
    measured = qmc.measure(ancilla)
    for index in qmc.range(2):
        condition = measured[index]
        while condition:
            fresh = qmc.qubit("fresh")
            condition = qmc.measure(fresh)
    return qmc.measure(qmc.qubit("output"))


@qmc.qkernel
def items_indexed_while_kernel(
    items: qmc.Dict[qmc.UInt, qmc.UInt],
) -> qmc.Bit:
    """Update measured-vector conditions selected by bound dictionary keys.

    Args:
        items (qmc.Dict[qmc.UInt, qmc.UInt]): Ordered keys selecting measured
            vector elements; values are ignored.

    Returns:
        qmc.Bit: Deterministic zero after every runtime loop terminates.
    """
    ancilla = qmc.qubit_array(2, "ancilla")
    ancilla[0] = qmc.x(ancilla[0])
    measured = qmc.measure(ancilla)
    for index, _value in qmc.items(items):
        condition = measured[index]
        while condition:
            fresh = qmc.qubit("fresh")
            condition = qmc.measure(fresh)
    return qmc.measure(qmc.qubit("output"))


@qmc.qkernel
def repeated_indexed_while_kernel() -> qmc.Bit:
    """Reread one overwritten measurement snapshot through nested loops.

    Returns:
        qmc.Bit: Unreachable output because emission must reject the stale
            snapshot read.
    """
    measured = qmc.measure(qmc.qubit_array(3, "ancilla"))
    for outer in qmc.range(2):
        for inner in qmc.range(2):
            condition = measured[outer + inner]
            while condition:
                fresh = qmc.qubit("fresh")
                condition = qmc.measure(fresh)
    return qmc.measure(qmc.qubit("output"))


@qmc.qkernel
def indexed_while_original_vector_output_kernel() -> qmc.Vector[qmc.Bit]:
    """Keep the original measured vector live after indexed while updates.

    Returns:
        qmc.Vector[qmc.Bit]: Unrepresentable immutable pre-update snapshot.
    """
    ancilla = qmc.qubit_array(2, "ancilla")
    ancilla[0] = qmc.x(ancilla[0])
    measured = qmc.measure(ancilla)
    for index in qmc.range(2):
        condition = measured[index]
        while condition:
            condition = qmc.measure(qmc.qubit("fresh"))
    return measured


@qmc.qkernel
def while_disjoint_element_output_kernel() -> qmc.Bit:
    """Update element zero while exposing the disjoint element one.

    Returns:
        qmc.Bit: Original measurement snapshot for untouched element one.
    """
    ancilla = qmc.qubit_array(2, "ancilla")
    ancilla[1] = qmc.x(ancilla[1])
    measured = qmc.measure(ancilla)
    condition = measured[0]
    while condition:
        condition = qmc.measure(qmc.qubit("fresh"))
    return measured[1]


@qmc.qkernel
def while_same_element_output_kernel() -> qmc.Bit:
    """Update and then expose the same immutable measured element.

    Returns:
        qmc.Bit: Unrepresentable original element-zero snapshot.
    """
    measured = qmc.measure(qmc.qubit_array(2, "ancilla"))
    condition = measured[0]
    while condition:
        condition = qmc.measure(qmc.qubit("fresh"))
    return measured[0]


@qmc.qkernel
def crossed_mixed_merge_kernel(selector: qmc.UInt) -> tuple[qmc.Bit, qmc.Bit]:
    """Cross one old measurement through two runtime Bit merge outputs.

    Args:
        selector (qmc.UInt): Compile-time flag selecting the measured runtime
            branch outcome.

    Returns:
        tuple[qmc.Bit, qmc.Bit]: Independently selected merge outputs.
    """
    old_qubit = qmc.x(qmc.qubit("old"))
    old = qmc.measure(old_qubit)
    condition_qubit = qmc.qubit("condition")
    if selector:
        condition_qubit = qmc.x(condition_qubit)
    condition = qmc.measure(condition_qubit)
    if condition:
        first = qmc.measure(qmc.qubit("true_fresh"))
        second = old
    else:
        first = old
        second = qmc.measure(qmc.qubit("false_fresh"))
    return first, second


@pytest.mark.parametrize("backend", BACKENDS)
class TestLoopCarriedValuesAcrossBackends:
    """Carried loops execute with Python semantics on every backend."""

    def test_classical_segment_accumulation(self, backend):
        """sum(range(4)) == 6 through the classical segment interpreter."""
        assert _sample_single(backend, sum_kernel, {"n": 4}) == 6

    def test_classical_segment_swap(self, backend):
        """Three swaps of (1, 2) land on (2, 1)."""
        assert _sample_single(backend, swap_kernel, {"n": 3}) == (2, 1)

    def test_emit_absorbed_angle_accumulation(self, backend):
        """An accumulated rx angle of 3*pi flips |0> deterministically.

        The carried loop feeds a gate parameter, so it is absorbed into
        the quantum segment and evaluated by emit-time unrolling.
        """
        assert _sample_single(backend, angle_kernel, {"n": 3}) == 1


@pytest.mark.parametrize("backend", RUNTIME_BACKENDS)
class TestNestedRuntimeLoopConditionsAcrossBackends:
    """Static replay preserves runtime condition clbits on capable backends."""

    def test_range_indexed_while_updates_the_selected_clbit(self, backend):
        """Each range iteration remeasures its own condition clbit and exits."""
        assert _sample_single(backend, indexed_while_kernel, {}) == 0

    def test_items_indexed_while_updates_the_selected_clbit(self, backend):
        """ForItems keys resolve condition aliases independently per entry."""
        assert (
            _sample_single(
                backend,
                items_indexed_while_kernel,
                {"items": {0: 0, 1: 0}},
            )
            == 0
        )

    def test_repeated_indexed_snapshot_is_rejected(self, backend):
        """A later replay cannot reread a snapshot whose clbit was overwritten."""
        from qamomile.circuit.transpiler.errors import EmitError

        with pytest.raises(EmitError, match="snapshot is read after"):
            _make_transpiler(backend).transpile(repeated_indexed_while_kernel)

    def test_original_vector_live_out_is_rejected(self, backend):
        """The immutable measured vector cannot escape after clbit reuse."""
        from qamomile.circuit.transpiler.errors import EmitError

        with pytest.raises(EmitError, match="snapshot remains live"):
            _make_transpiler(backend).transpile(
                indexed_while_original_vector_output_kernel
            )

    def test_disjoint_vector_element_live_out_is_allowed(self, backend):
        """An untouched sibling snapshot remains a valid public output."""
        assert _sample_single(backend, while_disjoint_element_output_kernel, {}) == 1

    def test_same_vector_element_live_out_is_rejected(self, backend):
        """The exact overwritten element cannot remain a public output."""
        from qamomile.circuit.transpiler.errors import EmitError

        with pytest.raises(EmitError, match="snapshot remains live"):
            _make_transpiler(backend).transpile(while_same_element_output_kernel)

    @pytest.mark.parametrize(
        ("selector", "expected"),
        [(0, (1, 0)), (1, (0, 1))],
    )
    def test_crossed_mixed_bit_merges_execute(self, backend, selector, expected):
        """Post-quantum SELECTs preserve two crossed Bit merge outputs.

        Classical lowering keeps all branch measurements on independent
        physical clbits and evaluates each merge as a host-side SELECT. No
        physical clbit mux or aliasing is needed for this output-only shape.
        """
        assert (
            _sample_single(
                backend,
                crossed_mixed_merge_kernel,
                {"selector": selector},
            )
            == expected
        )
