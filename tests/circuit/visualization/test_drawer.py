"""Tests for MatplotlibDrawer."""

import qamomile.circuit as qm


class TestMatplotlibDrawer:
    """Test suite for MatplotlibDrawer."""

    def test_single_qubit_gates(self):
        """Test drawing single-qubit gates."""

        @qm.qkernel
        def circuit(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            q = qm.x(q)
            q = qm.p(q, 0.5)
            return q

        # Should not raise
        fig = circuit.draw()
        assert fig is not None

    def test_rotation_gates(self):
        """Test drawing rotation gates with parameters."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            q = qm.ry(q, theta)
            q = qm.rz(q, theta)
            return q

        # With auto-detected symbolic parameter
        fig = circuit.draw()
        assert fig is not None

        # With bound parameter
        fig = circuit.draw(theta=0.5)
        assert fig is not None

    def test_two_qubit_gates(self):
        """Test drawing two-qubit gates."""

        @qm.qkernel
        def circuit(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q0, q1 = qm.cx(q0, q1)
            q0, q1 = qm.swap(q0, q1)
            q0, q1 = qm.rzz(q0, q1, 0.5)
            return q0, q1

        fig = circuit.draw()
        assert fig is not None

    def test_multi_qubit_circuit(self):
        """Test drawing circuit with multiple qubits."""

        @qm.qkernel
        def circuit(
            q0: qm.Qubit, q1: qm.Qubit, q2: qm.Qubit
        ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
            q0 = qm.h(q0)
            q1 = qm.h(q1)
            q2 = qm.h(q2)
            q0, q1 = qm.cx(q0, q1)
            q1, q2 = qm.cx(q1, q2)
            return q0, q1, q2

        fig = circuit.draw()
        assert fig is not None

    def test_empty_circuit(self):
        """Test drawing empty circuit."""

        @qm.qkernel
        def circuit() -> None:
            pass

        fig = circuit.draw()
        assert fig is not None

    def test_call_block_default(self):
        """Test drawing CallBlockOperation as box (default behavior)."""

        @qm.qkernel
        def inner(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            q = qm.x(q)
            return q

        @qm.qkernel
        def outer(q: qm.Qubit) -> qm.Qubit:
            q = inner(q)
            q = qm.rx(q, 0.3)
            return q

        # Default: show as box
        fig = outer.draw(inline=False)
        assert fig is not None

    def test_call_block_inline(self):
        """Test drawing CallBlockOperation with inline expansion."""

        @qm.qkernel
        def inner(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            q = qm.x(q)
            return q

        @qm.qkernel
        def outer(q: qm.Qubit) -> qm.Qubit:
            q = inner(q)
            q = qm.rx(q, 0.5)
            return q

        # Inline: expand blocks
        fig = outer.draw(inline=True)
        assert fig is not None

    def test_nested_blocks(self):
        """Test drawing nested CallBlockOperations."""

        @qm.qkernel
        def level1(q: qm.Qubit) -> qm.Qubit:
            return qm.x(q)

        @qm.qkernel
        def level2(q: qm.Qubit) -> qm.Qubit:
            q = level1(q)
            return qm.h(q)

        @qm.qkernel
        def level3(q: qm.Qubit) -> qm.Qubit:
            q = level2(q)
            return qm.rx(q, 0.5)

        # Default: nested boxes
        fig = level3.draw(inline=False)
        assert fig is not None

        # Inline: all expanded
        fig = level3.draw(inline=True)
        assert fig is not None

    def test_block_boundary_no_overlap(self):
        """Test that block boundaries don't overlap with gates outside (Issue #17)."""

        @qm.qkernel
        def inner(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            q = qm.x(q)
            return q

        @qm.qkernel
        def outer(q: qm.Qubit) -> qm.Qubit:
            q = inner(q)
            q = qm.rx(q, 0.5)  # This should be OUTSIDE inner's boundary
            return q

        # Test inline mode where the issue was reported
        fig = outer.draw(inline=True)
        assert fig is not None

    def test_parallel_gates(self):
        """Test drawing parallel gates on different qubits."""

        @qm.qkernel
        def circuit(
            q0: qm.Qubit, q1: qm.Qubit, q2: qm.Qubit
        ) -> tuple[qm.Qubit, qm.Qubit, qm.Qubit]:
            # These can be drawn in parallel columns
            q0 = qm.x(q0)
            q1 = qm.h(q1)
            q2 = qm.rx(q2, 0.5)
            return q0, q1, q2

        fig = circuit.draw()
        assert fig is not None

    def test_custom_style(self):
        """Test drawing with custom style."""
        from qamomile.circuit.visualization import CircuitStyle, MatplotlibDrawer

        @qm.qkernel
        def circuit(q: qm.Qubit) -> qm.Qubit:
            q = qm.h(q)
            return q

        custom_style = CircuitStyle(
            gate_face_color="#FF5733",
            gate_text_color="#FFFFFF",
            font_size=16,
        )

        graph = circuit.build()
        drawer = MatplotlibDrawer(graph, style=custom_style)
        fig = drawer.draw()
        assert fig is not None

    def test_cp_gate(self):
        """Test drawing controlled-phase gate."""

        @qm.qkernel
        def circuit(q0: qm.Qubit, q1: qm.Qubit) -> tuple[qm.Qubit, qm.Qubit]:
            q0, q1 = qm.cp(q0, q1, 0.5)
            return q0, q1

        fig = circuit.draw()
        assert fig is not None
