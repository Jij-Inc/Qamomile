"""Tests for the Block IR text pretty-printer.

Note: do not add ``from __future__ import annotations`` here — the @qkernel
frontend introspects runtime annotation classes (``qmc.Float`` etc.) and
comparing them to strings breaks parameter binding.
"""

import qamomile.circuit as qmc
from qamomile.circuit.ir import format_value, pretty_print_block
from qamomile.circuit.ir.block import BlockKind
from qamomile.circuit.ir.operation.inverse_block import InverseBlockOperation
from qamomile.qiskit import QiskitTranspiler

# ---------------------------------------------------------------------------
# Fixture kernels — small and focused on one feature each.
# ---------------------------------------------------------------------------


@qmc.qkernel
def _bell(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """Helper used to exercise inline InvokeOperation."""
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.composite_gate(name="printer_composite_x")
def _printer_composite_x(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X inside a preserved callable."""
    return qmc.x(q)


@qmc.qkernel
def _printer_patterned_concrete(
    c0: qmc.Qubit,
    c1: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply a concrete control on the LSB-first value two."""
    return qmc.control(qmc.x, num_controls=2, control_value=2)(c0, c1, target)


@qmc.qkernel
def _printer_patterned_composite(
    c0: qmc.Qubit,
    c1: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Apply a boxed control on the LSB-first value two."""
    return qmc.control(
        _printer_composite_x,
        num_controls=2,
        control_value=2,
    )(c0, c1, target)


@qmc.qkernel
def _printer_patterned_inverse(
    c0: qmc.Qubit,
    c1: qmc.Qubit,
    target: qmc.Qubit,
) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
    """Invert a qkernel containing a patterned boxed control."""
    return qmc.inverse(_printer_patterned_composite)(c0, c1, target)


@qmc.qkernel
def _printer_select_identity(target: qmc.Qubit) -> qmc.Qubit:
    """Return a SELECT target unchanged."""
    return target


@qmc.qkernel
def _printer_select_x(target: qmc.Qubit) -> qmc.Qubit:
    """Apply X in a SELECT case."""
    return qmc.x(target)


@qmc.qkernel
def _printer_select_h(target: qmc.Qubit) -> qmc.Qubit:
    """Apply H in a SELECT case."""
    return qmc.h(target)


@qmc.qkernel
def _printer_select_z(target: qmc.Qubit) -> qmc.Qubit:
    """Apply Z in a SELECT case."""
    return qmc.z(target)


@qmc.qkernel
def _printer_select_identity_x() -> qmc.Bit:
    """Apply an identity/X SELECT for printer tests."""
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    index, target = qmc.select([_printer_select_identity, _printer_select_x])(
        index, target
    )
    return qmc.measure(target)


@qmc.qkernel
def _printer_select_h_z() -> qmc.Bit:
    """Apply an H/Z SELECT for printer tests."""
    index = qmc.qubit("index")
    target = qmc.qubit("target")
    index, target = qmc.select([_printer_select_h, _printer_select_z])(
        index,
        target,
    )
    return qmc.measure(target)


@qmc.qkernel
def _printer_symbolic_select(width: qmc.UInt) -> qmc.Bit:
    """Apply a SELECT whose index width remains symbolic while printing."""
    index = qmc.qubit_array(2, "index")
    target = qmc.qubit("target")
    index, target = qmc.select(
        [_printer_select_identity, _printer_select_x],
        num_index_qubits=width,
    )(index, target)
    return qmc.measure(target)


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
    c = qmc.qubit(name="c")
    c = qmc.h(c)
    bit = qmc.measure(c)
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
    body = lines[idx + 1 :]
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


def test_if_renders_both_branches_and_merge():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_with_runtime_if)
    out = pretty_print_block(block)
    assert "if " in out
    # The runtime-if kernel has no explicit else, so only the true branch is
    # required — but a merge is always emitted for values that survive the if.
    assert "merge(" in out, f"missing merge in:\n{out}"


# ---------------------------------------------------------------------------
# InvokeOperation depth expansion
# ---------------------------------------------------------------------------


def test_invoke_depth_zero_shows_name_only():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_with_for, bindings={"n": 3}, parameters=["theta"])
    out = pretty_print_block(block, depth=0)
    # The callee name must appear in an "invoke" line.
    invoke_lines = [ln for ln in out.splitlines() if "invoke " in ln]
    assert any("_bell" in ln for ln in invoke_lines), (
        f"expected callee name in invoke line, got:\n{out}"
    )
    # Depth 0 must not open a nested block after the invocation.
    for ln in invoke_lines:
        assert not ln.rstrip().endswith("{"), (
            f"depth=0 should not open a nested invoke block:\n{out}"
        )


def test_invoke_depth_one_expands_body():
    transpiler = QiskitTranspiler()
    block = transpiler.to_block(_with_for, bindings={"n": 3}, parameters=["theta"])
    out = pretty_print_block(block, depth=1)
    # At depth=1 we expect to see the callee body (h + cx) expanded inline.
    assert " = h(" in out
    assert " = cx(" in out
    # A nested open brace should appear on an invoke line.
    assert any(
        "invoke " in ln and ln.rstrip().endswith("{") for ln in out.splitlines()
    ), f"depth=1 should open a nested invoke block:\n{out}"


def test_select_depth_zero_shows_metadata_and_case_names():
    """Depth zero distinguishes SELECTs by width and named case list."""
    transpiler = QiskitTranspiler()
    identity_x = pretty_print_block(
        transpiler.to_block(_printer_select_identity_x),
        depth=0,
    )
    h_z = pretty_print_block(
        transpiler.to_block(_printer_select_h_z),
        depth=0,
    )

    assert identity_x != h_z
    assert "index_width=1" in identity_x
    assert "index_args=1" in identity_x
    assert "0:_printer_select_identity" in identity_x
    assert "1:_printer_select_x" in identity_x
    assert not any(
        line.lstrip().startswith("case ") for line in identity_x.splitlines()
    )


def test_select_depth_one_expands_each_case_body():
    """Positive depth opens named SELECT cases and prints their operations."""
    out = pretty_print_block(
        QiskitTranspiler().to_block(_printer_select_h_z),
        depth=1,
    )

    assert "case 0 _printer_select_h {" in out
    assert "case 1 _printer_select_z {" in out
    assert " = h(" in out
    assert " = z(" in out


def test_select_symbolic_width_is_visible():
    """A symbolic SELECT width is rendered as its runtime parameter."""
    block = QiskitTranspiler().to_block(
        _printer_symbolic_select,
        parameters=["width"],
    )
    out = pretty_print_block(block)

    assert "index_width=param(width)" in out
    assert "index_args=1" in out


def test_select_metadata_survives_every_block_kind():
    """SELECT metadata remains visible through inline, partial eval, and analyze."""
    transpiler = QiskitTranspiler()
    hierarchical = transpiler.to_block(_printer_select_identity_x)
    affine = transpiler.inline(hierarchical)
    partially_evaluated = transpiler.partial_eval(affine, bindings={})
    analyzed = transpiler.analyze(partially_evaluated)

    for block in (hierarchical, affine, partially_evaluated, analyzed):
        out = pretty_print_block(block)
        assert "index_width=1" in out
        assert "0:_printer_select_identity" in out
        assert "1:_printer_select_x" in out


def test_patterned_control_metadata_is_visible_for_every_call_representation():
    """Pretty printing distinguishes zero/one activation patterns."""
    transpiler = QiskitTranspiler()

    concrete = pretty_print_block(transpiler.to_block(_printer_patterned_concrete))
    assert "controlled x(" in concrete
    assert "control_value=2" in concrete

    invoke = pretty_print_block(transpiler.to_block(_printer_patterned_composite))
    assert "transform=CONTROLLED" in invoke
    assert "controls=2" in invoke
    assert "control_value=2" in invoke

    inverse_entry = transpiler.to_block(_printer_patterned_inverse)
    [outer_inverse] = [
        op for op in inverse_entry.operations if isinstance(op, InverseBlockOperation)
    ]
    assert outer_inverse.implementation_block is not None
    inverse = pretty_print_block(outer_inverse.implementation_block)
    assert "inverse printer_composite_x_inverse(" in inverse
    assert "controls=2" in inverse
    assert "control_value=2" in inverse


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
