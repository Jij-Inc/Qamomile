"""Regression tests for the controlled-U box label in the drawer.

Asserts that a ControlledUOperation rendered as ``VGateKind.CONTROLLED_U_BOX``
carries the wrapped callable's name plus a parameter suffix that lists
each classical parameter at the call site. Covers four label paths:

- a user ``@qmc.qkernel`` whose Python default is auto-filled by
  ``inspect.Signature.bind + apply_defaults``;
- a user ``@qmc.qkernel`` whose classical kwargs are passed explicitly;
- a built-in gate function wrapped via ``qmc.control(qmc.rx)``, where
  the lowercase block name (``"rx"``) is mapped to the same TeX label
  (``$R_x$``) that the inline-gate path uses.
- a parameterized ``@qmc.composite_gate`` invocation preserved as a semantic
  reusable call.

Each test inspects ``VisualCircuit.children`` produced from the verified
target-neutral ``CircuitProgram`` so the public drawing path, rather than the
legacy analyzer, owns the regression contract.
"""

from __future__ import annotations

import math
from typing import Any

import qamomile.circuit as qmc
from qamomile.circuit.transpiler.circuit_ir import CallInstruction, ParameterExpr
from qamomile.circuit.visualization.circuit_adapter import (
    circuit_program_to_visual_ir,
)
from qamomile.circuit.visualization.drawing_compiler import (
    compile_qkernel_for_drawing,
)
from qamomile.circuit.visualization.style import DEFAULT_STYLE
from qamomile.circuit.visualization.visual_ir import VGate, VGateKind, VInlineBlock


def _controlled_u_box(kernel: Any) -> VGate:
    """Return the single CONTROLLED_U_BOX ``VGate`` produced by *kernel*.

    Args:
        kernel (Any): A ``@qmc.qkernel`` whose top-level Block contains
            exactly one ``ControlledUOperation``.

    Returns:
        VGate: The ``VGate`` (with ``label``, ``power``, etc.) that
        the exact circuit adapter emitted for the controlled-U operation.

    Raises:
        AssertionError: If the kernel does not produce exactly one
            ``CONTROLLED_U_BOX`` node.
    """
    drawing = compile_qkernel_for_drawing(kernel)
    vc = circuit_program_to_visual_ir(drawing.circuit)
    boxes = [
        n
        for n in vc.children
        if isinstance(n, VGate) and n.kind is VGateKind.CONTROLLED_U_BOX
    ]
    assert len(boxes) == 1, (
        f"expected exactly one CONTROLLED_U_BOX VGate, found {len(boxes)}"
    )
    return boxes[0]


def _controlled_u_label(kernel: Any) -> str:
    """Return only the controlled-U box label for a kernel.

    Args:
        kernel (Any): QKernel-like object containing one controlled call.

    Returns:
        str: Label resolved for the controlled-U box.
    """
    return _controlled_u_box(kernel).label


def test_controlled_box_label_includes_default_arg_value() -> None:
    """Auto-filled default arg surfaces on the controlled-U box label."""

    @qmc.qkernel
    def _phase(q: qmc.Qubit, theta: qmc.Float = math.pi / 2) -> qmc.Qubit:
        return qmc.rx(q, theta)

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        c = qmc.x(c)
        cg = qmc.control(_phase)
        c, t = cg(c, t)
        return qmc.measure(t)

    label = _controlled_u_label(kernel)
    # Format: ``<block_name>(<param>=<formatted_value>)``.  The
    # parameter name passes through ``_format_call_parameter_name`` so
    # Greek-letter names render in TeX (``$\theta$``); the value
    # passes through ``_format_call_argument`` which prints two decimals
    # in the (0.01, 10) range, so math.pi / 2 ≈ 1.57.
    assert label == r"_phase($\theta$=1.57)", label


def test_controlled_box_label_includes_multiple_kwargs() -> None:
    """All classical kwargs appear on the controlled-U box label, signature order."""

    @qmc.qkernel
    def _two_param(q: qmc.Qubit, alpha: qmc.Float, beta: qmc.Float) -> qmc.Qubit:
        q = qmc.rx(q, alpha)
        q = qmc.rz(q, beta)
        return q

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        c = qmc.x(c)
        cg = qmc.control(_two_param)
        c, t = cg(c, t, alpha=0.7, beta=1.3)
        return qmc.measure(t)

    label = _controlled_u_label(kernel)
    assert label == r"_two_param($\alpha$=0.70, $\beta$=1.30)", label


def test_controlled_box_label_uses_tex_name_for_builtin_gate() -> None:
    """``qmc.control(qmc.rx)`` renders with the same ``$R_x$`` TeX label as inline rx."""

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        c = qmc.x(c)
        cg = qmc.control(qmc.rx)
        c, t = cg(c, t, angle=math.pi / 4)
        return qmc.measure(t)

    label = _controlled_u_label(kernel)
    assert label == "$R_x$(angle=0.79)", label


def test_composite_box_label_includes_call_site_parameter() -> None:
    """Parameterized composite calls retain exact scalar call metadata."""

    @qmc.composite_gate(name="boxed_phase")
    def boxed_phase(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
        """Apply a phase rotation through a preserved composite boundary.

        Args:
            q (qmc.Qubit): Target qubit.
            theta (qmc.Float): Rotation angle.

        Returns:
            qmc.Qubit: Rotated target qubit.
        """
        return qmc.rz(q, theta)

    @qmc.qkernel
    def kernel(angle: qmc.Float) -> qmc.Bit:
        """Call the parameterized composite and measure its target.

        Args:
            angle (qmc.Float): Composite rotation angle.

        Returns:
            qmc.Bit: Target measurement result.
        """
        q = qmc.qubit(name="q")
        q = boxed_phase(q, angle)
        return qmc.measure(q)

    drawing = compile_qkernel_for_drawing(kernel)
    calls = [
        operation
        for operation in drawing.circuit.operations
        if isinstance(operation, CallInstruction)
    ]
    assert len(calls) == 1
    assert calls[0].callee.call_arguments == (("theta", ParameterExpr("angle")),)

    visual = circuit_program_to_visual_ir(drawing.circuit)
    boxes = [node for node in visual.children if isinstance(node, VGate)]
    assert boxes[0].label == r"boxed_phase($\theta$=angle)"


def test_controlled_box_power_lives_on_vgate_not_label() -> None:
    """``power=k`` keeps label, inner width, and outer width separate.

    Verifies the split between in-label text (just the wrapped
    callable + classical kwargs) and the ``power`` field that the
    renderer turns into an outer ``pow=N`` wrapper box. The layout-facing
    ``estimated_width`` includes both wrapper margins, while ``box_width``
    remains the width of the inner target box consumed by the renderer.
    """

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        c = qmc.x(c)
        cg = qmc.control(qmc.rx)
        c, t = cg(c, t, angle=math.pi / 4, power=3)
        return qmc.measure(t)

    box = _controlled_u_box(kernel)
    assert box.label == "$R_x$(angle=0.79)", box.label
    assert box.power == 3, box.power
    assert box.box_width is not None
    assert math.isclose(
        box.estimated_width,
        box.box_width + 2 * DEFAULT_STYLE.power_wrapper_margin,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_controlled_box_power_defaults_to_one() -> None:
    """An unpowered box uses the same width for layout and rendering."""

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        c = qmc.qubit(name="c")
        t = qmc.qubit(name="t")
        c = qmc.x(c)
        cg = qmc.control(qmc.rx)
        c, t = cg(c, t, angle=math.pi / 4)
        return qmc.measure(t)

    box = _controlled_u_box(kernel)
    assert box.power == 1, box.power
    assert box.box_width is not None
    assert math.isclose(
        box.estimated_width,
        box.box_width,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _controlled_u_box_with_bindings(kernel: Any, **bindings: Any) -> VGate:
    """Same as ``_controlled_u_box`` but threads concrete bindings through.

    Args:
        kernel (Any): A ``@qmc.qkernel`` whose top-level Block contains
            exactly one ``ControlledUOperation``.
        **bindings (Any): Concrete values for kernel parameters, used so
            symbolic ``UInt`` indices can be resolved.

    Returns:
        VGate: The CONTROLLED_U_BOX node produced for the kernel.
    """
    drawing = compile_qkernel_for_drawing(kernel, bindings)
    vc = circuit_program_to_visual_ir(drawing.circuit)
    boxes = [
        n
        for n in vc.children
        if isinstance(n, VGate) and n.kind is VGateKind.CONTROLLED_U_BOX
    ]
    assert len(boxes) == 1, (
        f"expected exactly one CONTROLLED_U_BOX VGate, found {len(boxes)}"
    )
    return boxes[0]


def test_controlled_box_subset_indices_filter_inactive_pool_slots() -> None:
    """Non-contiguous ``control_indices`` drop the skipped slot from VGate.

    The diagram for a ``SymbolicControlledU`` with
    ``control_indices=[0, 1, 3]`` over a 4-qubit pool should
    keep three control dots (the active slots, on the pool's
    wire indices) and omit pool[2] entirely.  ``control_count``
    is the active count and the inactive slot's wire index does
    not appear in the control prefix of ``qubit_indices``.
    """

    @qmc.qkernel
    def kernel(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        pool = qmc.qubit_array(n, "pool")
        tgt = qmc.qubit(name="tgt")
        cg = qmc.control(qmc.x, num_controls=k_ctrls)
        pool, tgt = cg(pool, tgt, control_indices=[0, 1, 3])
        return qmc.measure(pool)

    box = _controlled_u_box_with_bindings(kernel, n=4, k_ctrls=3)
    assert box.control_count == 3, box.control_count
    controls = box.qubit_indices[: box.control_count]
    # The pool sits on wires 0..3; control_indices=[0, 1, 3]
    # should keep wires 0, 1, 3 and drop wire 2.
    assert controls == [0, 1, 3], controls


def test_exact_controlled_subset_indices_filter_inactive_pool_slots() -> None:
    """Exact controlled calls omit inactive control-pool slots."""

    @qmc.qkernel
    def kernel(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Apply a selected symbolic control subset to one target.

        Args:
            n (qmc.UInt): Width of the control pool.
            k_ctrls (qmc.UInt): Number of selected active controls.

        Returns:
            qmc.Vector[qmc.Bit]: Measurements of the control pool.
        """
        pool = qmc.qubit_array(n, "pool")
        target = qmc.qubit(name="target")
        controlled_x = qmc.control(qmc.x, num_controls=k_ctrls)
        pool, target = controlled_x(
            pool,
            target,
            control_indices=[0, 1, 3],
        )
        return qmc.measure(pool)

    drawing = compile_qkernel_for_drawing(kernel, {"n": 4, "k_ctrls": 3})
    circuit = circuit_program_to_visual_ir(
        drawing.circuit,
        expand_calls=True,
    )

    [block] = [node for node in circuit.children if isinstance(node, VInlineBlock)]
    assert block.control_qubit_indices == [0, 1, 3]
    assert block.affected_qubits == [4]


def test_controlled_box_subset_indices_with_uint_entry_resolves_via_bindings() -> None:
    """``control_indices`` entries that involve ``UInt`` arithmetic resolve from bindings.

    With ``[0, 1, n - 1]`` and ``n=4`` the third entry resolves
    to 3, leaving wire 2 inactive — matching the literal
    ``[0, 1, 3]`` shape of the previous test.
    """

    @qmc.qkernel
    def kernel(n: qmc.UInt, k_ctrls: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        pool = qmc.qubit_array(n, "pool")
        tgt = qmc.qubit(name="tgt")
        cg = qmc.control(qmc.x, num_controls=k_ctrls)
        pool, tgt = cg(pool, tgt, control_indices=[0, 1, n - 1])
        return qmc.measure(pool)

    box = _controlled_u_box_with_bindings(kernel, n=4, k_ctrls=3)
    assert box.control_count == 3, box.control_count
    controls = box.qubit_indices[: box.control_count]
    assert controls == [0, 1, 3], controls
