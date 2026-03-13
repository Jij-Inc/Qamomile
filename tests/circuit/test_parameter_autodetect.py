"""Tests for parameter auto-detection and extended parameter types."""

import pytest

import qamomile.circuit as qm


class TestParameterAutoDetection:
    """Test auto-detection of parameters in build() and draw()."""

    def test_auto_detect_float_parameter(self):
        """Test that float parameters are auto-detected."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # Auto-detect (parameters=None, default)
        graph = circuit.build()
        assert "theta" in graph.parameters
        assert graph.parameters["theta"].params.get("parameter") == "theta"

    def test_auto_detect_uint_parameter(self):
        """Test that UInt parameters are auto-detected."""

        @qm.qkernel
        def circuit(q: qm.Qubit, n: qm.UInt) -> qm.Qubit:
            q = qm.rx(q, n)
            return q

        # Auto-detect
        graph = circuit.build()
        assert "n" in graph.parameters
        assert graph.parameters["n"].params.get("parameter") == "n"

    def test_auto_detect_int_parameter(self):
        """Test that int parameters are auto-detected."""

        @qm.qkernel
        def circuit(q: qm.Qubit, n: int) -> qm.Qubit:
            q = qm.rx(q, n)
            return q

        # Auto-detect
        graph = circuit.build()
        assert "n" in graph.parameters
        assert graph.parameters["n"].params.get("parameter") == "n"

    def test_auto_detect_multiple_parameters(self):
        """Test that multiple parameters are auto-detected."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float, phi: float, n: int) -> qm.Qubit:
            q = qm.rx(q, theta)
            q = qm.ry(q, phi)
            q = qm.rz(q, n)
            return q

        # Auto-detect all
        graph = circuit.build()
        assert "theta" in graph.parameters
        assert "phi" in graph.parameters
        assert "n" in graph.parameters

    def test_auto_detect_skips_provided_kwargs(self):
        """Test that parameters with values in kwargs are not auto-detected."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float, phi: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            q = qm.ry(q, phi)
            return q

        # Provide theta, auto-detect phi
        graph = circuit.build(theta=0.5)
        assert "theta" not in graph.parameters
        assert "phi" in graph.parameters

    def test_auto_detect_skips_defaults(self):
        """Test that parameters with defaults are not auto-detected."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float = 0.5, phi: float = 1.0) -> qm.Qubit:
            q = qm.rx(q, theta)
            q = qm.ry(q, phi)
            return q

        # No parameters should be auto-detected (all have defaults)
        graph = circuit.build()
        assert len(graph.parameters) == 0

    def test_auto_detect_skips_qubits(self):
        """Test that Qubit arguments are never auto-detected."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # Only theta should be detected, not q
        graph = circuit.build()
        assert "theta" in graph.parameters
        assert "q" not in graph.parameters

    def test_explicit_empty_list_requires_all_values(self):
        """Test that parameters=[] means no parameters allowed."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # Explicit empty list should require theta value
        with pytest.raises(ValueError, match="must be provided"):
            circuit.build(parameters=[])

    def test_explicit_empty_list_with_kwargs_works(self):
        """Test that parameters=[] works when all values provided."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # Explicit empty list + theta value should work
        graph = circuit.build(parameters=[], theta=0.5)
        assert len(graph.parameters) == 0

    def test_explicit_parameter_list_still_works(self):
        """Test that explicit parameter list still works as before."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # Explicit parameter list
        graph = circuit.build(parameters=["theta"])
        assert "theta" in graph.parameters


class TestExtendedParameterTypes:
    """Test that UInt and int types work as parameters."""

    def test_uint_scalar_parameter(self):
        """Test UInt scalar as parameter."""

        @qm.qkernel
        def circuit(q: qm.Qubit, n: qm.UInt) -> qm.Qubit:
            q = qm.rx(q, n)
            return q

        graph = circuit.build(parameters=["n"])
        assert "n" in graph.parameters
        # Check that the parameter has UIntType
        from qamomile.circuit.ir.types import UIntType

        assert isinstance(graph.parameters["n"].type, UIntType)

    def test_int_scalar_parameter(self):
        """Test int scalar as parameter."""

        @qm.qkernel
        def circuit(q: qm.Qubit, n: int) -> qm.Qubit:
            q = qm.rx(q, n)
            return q

        graph = circuit.build(parameters=["n"])
        assert "n" in graph.parameters
        # Check that the parameter has UIntType
        from qamomile.circuit.ir.types import UIntType

        assert isinstance(graph.parameters["n"].type, UIntType)

    def test_uint_bound_value(self):
        """Test UInt with bound value."""

        @qm.qkernel
        def circuit(q: qm.Qubit, n: qm.UInt) -> qm.Qubit:
            q = qm.rx(q, n)
            return q

        graph = circuit.build(n=5)
        assert "n" not in graph.parameters
        # The value should be bound as a constant

    def test_int_bound_value(self):
        """Test int with bound value."""

        @qm.qkernel
        def circuit(q: qm.Qubit, n: int) -> qm.Qubit:
            q = qm.rx(q, n)
            return q

        graph = circuit.build(n=5)
        assert "n" not in graph.parameters


class TestDrawAutoDetection:
    """Test that draw() auto-detects parameters."""

    def test_draw_auto_detects_parameter(self):
        """Test that draw() auto-detects parameters."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # Should auto-detect theta without needing parameters argument
        fig = circuit.draw()
        assert fig is not None

    def test_draw_with_bound_parameter(self):
        """Test that draw() works with bound parameters."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # Bind theta
        fig = circuit.draw(theta=0.5)
        assert fig is not None

    def test_draw_no_longer_accepts_parameters_argument(self):
        """Test that draw() no longer accepts parameters argument."""

        @qm.qkernel
        def circuit(q: qm.Qubit, theta: float) -> qm.Qubit:
            q = qm.rx(q, theta)
            return q

        # parameters argument should raise TypeError
        with pytest.raises(TypeError):
            circuit.draw(parameters=["theta"])
