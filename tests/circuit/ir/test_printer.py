"""Tests for the Block IR text pretty-printer.

Note: do not add ``from __future__ import annotations`` here — the @qkernel
frontend introspects runtime annotation classes (``qmc.Float`` etc.) and
comparing them to strings breaks parameter binding.
"""

import qamomile.circuit as qmc
from qamomile.circuit.ir import format_value, pretty_print_block
from qamomile.circuit.ir.block import BlockKind
from qamomile.qiskit import QiskitTranspiler


# ---------------------------------------------------------------------------
# Fixture kernels — small and focused on one feature each.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _bell(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Helper used to exercise CallBlockOperation."""
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.qkernel
def _gate_only(theta: qmc.Float) -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    q = qmc.rx(q, theta)
    return qmc.measure(q)


@qmc.qkernel
def _with_for(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = _bell(q[i], q[i + 1])
        q[i + 1] = qmc.rz(q[i + 1], theta)
    return qmc.measure(q)


@qmc.qkernel
def _with_runtime_if() -> qmc.Bit:
    q = qmc.qubit(name="q")
    q = qmc.h(q)
    bit = qmc.measure(q)
    if bit:
        q = qmc.x(q)
    return qmc.measure(q)


# ---------------------------------------------------------------------------
# Basic output shape & header
# ---------------------------------------------------------------------------


def test_gate_only_hierarchical_header_and_body():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_gate_only, parameters=["theta"])
    out = pretty_print_block(block)
    lines = out.splitlines()
    # Header must mention the kernel name, kind tag, and closing brace.
    assert lines[0].startswith("block _gate_only [HIERARCHICAL]")
    assert lines[0].rstrip().endswith("{")
    assert lines[-1].rstrip() == "}"
    # Body should list both gates.
    assert any(" = h(" in ln for ln in lines)
    assert any(" = rx(" in ln for ln in lines)
    # Parameters section appears when parameters are declared.
    assert any("parameters:" in ln and "theta" in ln for ln in lines)


def test_header_shows_signature_types():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_gate_only, parameters=["theta"])
    out = pretty_print_block(block)
    header = out.splitlines()[0]
    # Inputs appear as ``name: TypeLabel``; output type after ``->``.
    assert "theta: FloatType" in header
    assert "->" in header and "BitType" in header


# ---------------------------------------------------------------------------
# Control flow: for / if / while
# ---------------------------------------------------------------------------


def test_for_body_is_indented():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_with_for, bindings={"n": 3}, parameters=["theta"])
    out = pretty_print_block(block)
    lines = out.splitlines()
    # There must be a ``for`` header at some indent.
    for_lines = [ln for ln in lines if ln.lstrip().startswith("for ")]
    assert for_lines, f"no for-loop line found in:\n{out}"
    header = for_lines[0]
    # Body operations inside the for-loop live at a strictly deeper indent.
    header_indent = len(header) - len(header.lstrip())
    idx = lines.index(header)
    body = lines[idx + 1 : ]
    # The first non-close-brace body line should be deeper than the header.
    for ln in body:
        stripped = ln.strip()
        if stripped == "}":
            break
        if stripped:
            assert len(ln) - len(ln.lstrip()) > header_indent, (
                f"body line not indented deeper than header:\n{out}"
            )
            break


def test_if_renders_both_branches_and_phi():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_with_runtime_if)
    out = pretty_print_block(block)
    assert "if " in out
    # The runtime-if kernel has no explicit else, so only the true branch is
    # required — but a phi is always emitted for values that survive the if.
    assert "phi(" in out, f"missing phi in:\n{out}"


# ---------------------------------------------------------------------------
# CallBlockOperation depth expansion
# ---------------------------------------------------------------------------


def test_call_block_depth_zero_shows_name_only():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_with_for, bindings={"n": 3}, parameters=["theta"])
    out = pretty_print_block(block, depth=0)
    # The callee name must appear in a "call" line.
    call_lines = [ln for ln in out.splitlines() if "call " in ln]
    assert any("_bell" in ln for ln in call_lines), (
        f"expected callee name in call line, got:\n{out}"
    )
    # Depth 0 must not open a nested block after the call.
    for ln in call_lines:
        assert not ln.rstrip().endswith("{"), (
            f"depth=0 should not open a nested call block:\n{out}"
        )


def test_call_block_depth_one_expands_body():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_with_for, bindings={"n": 3}, parameters=["theta"])
    out = pretty_print_block(block, depth=1)
    # At depth=1 we expect to see the callee body (h + cx) expanded inline.
    assert " = h(" in out
    assert " = cx(" in out
    # A nested open brace should appear on a call line.
    assert any(
        "call " in ln and ln.rstrip().endswith("{") for ln in out.splitlines()
    ), f"depth=1 should open a nested call block:\n{out}"


# ---------------------------------------------------------------------------
# BlockKind coverage: works at each stage of the pipeline
# ---------------------------------------------------------------------------


def test_runs_on_all_block_kinds_without_error():
    transpiler = QiskitTranspiler()
    bindings = {"n": 3}
    parameters = ["theta"]

    block = transpiler.to_block(_with_for, bindings=bindings, parameters=parameters)
    assert block.kind is BlockKind.HIERARCHICAL
    hier = pretty_print_block(block)
    assert "[HIERARCHICAL]" in hier.splitlines()[0]

    block = transpiler.inline(block)
    assert block.kind is BlockKind.AFFINE
    affine = pretty_print_block(block)
    assert "[AFFINE]" in affine.splitlines()[0]

    block = transpiler.partial_eval(block, bindings=bindings)
    analysed = transpiler.analyze(block)
    assert analysed.kind is BlockKind.ANALYZED
    out = pretty_print_block(analysed)
    assert "[ANALYZED]" in out.splitlines()[0]


# ---------------------------------------------------------------------------
# format_value helper
# ---------------------------------------------------------------------------


def test_format_value_shapes():
    from qamomile.circuit.ir.types.primitives import FloatType, QubitType
    from qamomile.circuit.ir.value import Value

    q = Value(type=QubitType(), name="q", version=0)
    assert format_value(q) == "%q@v0"
    q1 = q.next_version()
    assert format_value(q1) == "%q@v1"

    # Parameter-tagged value shows the parameter name.
    theta = Value(type=FloatType(), name="theta", version=0).with_parameter("theta")
    assert format_value(theta) == "param(theta)"

    # Constant-tagged value shows the literal.
    c = Value(type=FloatType(), name="c", version=0).with_const(2.5)
    assert format_value(c) == "const(2.5)"

    # Unrecognised input falls back to repr rather than crashing.
    assert format_value(42) == "42"
