"""Tests for partial/selective measurement correctness.

Verifies that backends correctly decode measurement results when only a subset
of qubits are measured.  This is a regression test for a bug where QURI Parts
(whose emit_measure is a no-op) returned bits indexed by qubit position rather
than by the actually-measured qubit.

Covers:
  - Single qmc.Qubit: measure one qubit out of many
  - qmc.Vector[qmc.Qubit]: measure an array while other qubits are unmeasured
  - Mixed qmc.Qubit + qmc.Vector[qmc.Qubit] scenarios
  - Parametric (symbolic) circuits with later binding
  - Multiple selective measurements in varied order

Note: Do NOT use ``from __future__ import annotations`` in this file.
The @qkernel AST transformer relies on resolved type annotations.
"""

import math

import pytest

import qamomile.circuit as qmc

# ---------------------------------------------------------------------------
# Skip if backends are not installed
# ---------------------------------------------------------------------------
pytest.importorskip("quri_parts")
pytest.importorskip("quri_parts.qulacs")

from qamomile.quri_parts import QuriPartsTranspiler
from qamomile.qiskit import QiskitTranspiler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sample_values(
    transpiler, kernel, shots=256,
    transpile_kwargs=None, sample_kwargs=None,
):
    """Transpile, sample, and return the list of (value, count) results."""
    executable = transpiler.transpile(kernel, **(transpile_kwargs or {}))
    job = executable.sample(
        transpiler.executor(), shots=shots, **(sample_kwargs or {})
    )
    return job.result().results


def _assert_all_equal(results, expected_value, label=""):
    """Assert every sampled value matches expected_value."""
    for value, count in results:
        assert value == expected_value, (
            f"{label}Expected {expected_value}, got {value} (count={count})"
        )


# ---------------------------------------------------------------------------
# 1. Single qmc.Qubit – measure one qubit out of several
# ---------------------------------------------------------------------------


class TestSingleQubitPartialMeasurement:
    """Measure a single qmc.Qubit while other qubits exist."""

    @staticmethod
    @qmc.qkernel
    def measure_q1_only() -> qmc.Bit:
        """X on q0, measure q1 (should be 0)."""
        qs = qmc.qubit_array(2, "qs")
        qs[0] = qmc.x(qs[0])
        return qmc.measure(qs[1])

    @staticmethod
    @qmc.qkernel
    def measure_q0_only() -> qmc.Bit:
        """X on q0, measure q0 (should be 1)."""
        qs = qmc.qubit_array(2, "qs")
        qs[0] = qmc.x(qs[0])
        return qmc.measure(qs[0])

    @staticmethod
    @qmc.qkernel
    def measure_middle_of_three() -> qmc.Bit:
        """3 qubits, X on q0 and q2, measure q1 (should be 0)."""
        q0 = qmc.qubit("q0")
        q1 = qmc.qubit("q1")
        q2 = qmc.qubit("q2")
        q0 = qmc.x(q0)
        q2 = qmc.x(q2)
        return qmc.measure(q1)

    @staticmethod
    @qmc.qkernel
    def measure_last_of_three() -> qmc.Bit:
        """3 qubits, X on q2 only, measure q2 (should be 1)."""
        q0 = qmc.qubit("q0")
        q1 = qmc.qubit("q1")
        q2 = qmc.qubit("q2")
        q2 = qmc.x(q2)
        return qmc.measure(q2)

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_measure_untouched_qubit_returns_zero(self, transpiler_factory):
        """X on q0, measure q1 → all 0."""
        t = transpiler_factory()
        results = _sample_values(t, self.measure_q1_only)
        _assert_all_equal(results, 0, label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_measure_flipped_qubit_returns_one(self, transpiler_factory):
        """X on q0, measure q0 → all 1."""
        t = transpiler_factory()
        results = _sample_values(t, self.measure_q0_only)
        _assert_all_equal(results, 1, label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_measure_middle_qubit(self, transpiler_factory):
        """X on q0 and q2, measure q1 → 0."""
        t = transpiler_factory()
        results = _sample_values(t, self.measure_middle_of_three)
        _assert_all_equal(results, 0, label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_measure_last_qubit(self, transpiler_factory):
        """X on q2, measure q2 → 1."""
        t = transpiler_factory()
        results = _sample_values(t, self.measure_last_of_three)
        _assert_all_equal(results, 1, label=f"[{transpiler_factory.__name__}] ")


# ---------------------------------------------------------------------------
# 2. qmc.Vector[qmc.Qubit] – measure a vector subset
# ---------------------------------------------------------------------------


class TestVectorPartialMeasurement:
    """Measure a qmc.Vector[qmc.Qubit] while other qubits go unmeasured."""

    @staticmethod
    @qmc.qkernel
    def measure_vector_subset() -> qmc.Vector[qmc.Bit]:
        """4 qubits (2 measured, 2 unmeasured). X on unmeasured[0] and measured[0].
        Measure only measured. Expect (1, 0)."""
        measured = qmc.qubit_array(2, "measured")
        unmeasured = qmc.qubit_array(2, "unmeasured")
        unmeasured[0] = qmc.x(unmeasured[0])
        measured[0] = qmc.x(measured[0])
        return qmc.measure(measured)

    @staticmethod
    @qmc.qkernel
    def measure_second_vector() -> qmc.Vector[qmc.Bit]:
        """Two arrays: anchor (2q, X on both) and target (2q, untouched).
        Measure only target. Expect (0, 0)."""
        anchor = qmc.qubit_array(2, "anchor")
        target = qmc.qubit_array(2, "target")
        anchor[0] = qmc.x(anchor[0])
        anchor[1] = qmc.x(anchor[1])
        return qmc.measure(target)

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_measure_vector_subset(self, transpiler_factory):
        t = transpiler_factory()
        results = _sample_values(t, self.measure_vector_subset)
        _assert_all_equal(results, (1, 0), label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_measure_second_vector_all_zero(self, transpiler_factory):
        t = transpiler_factory()
        results = _sample_values(t, self.measure_second_vector)
        _assert_all_equal(results, (0, 0), label=f"[{transpiler_factory.__name__}] ")


# ---------------------------------------------------------------------------
# 3. Mixed qmc.Qubit + qmc.Vector[qmc.Qubit]
# ---------------------------------------------------------------------------


class TestMixedQubitVectorMeasurement:
    """Measure a mix of scalar qubits and vector qubits."""

    @staticmethod
    @qmc.qkernel
    def mixed_measure_scalar_from_vector_circuit() -> qmc.Bit:
        """Vector of 3 + scalar qubit. X on vec[0], measure scalar (should be 0)."""
        vec = qmc.qubit_array(3, "vec")
        scalar = qmc.qubit("scalar")
        vec[0] = qmc.x(vec[0])
        return qmc.measure(scalar)

    @staticmethod
    @qmc.qkernel
    def mixed_measure_vector_with_scalar_excited() -> qmc.Vector[qmc.Bit]:
        """Scalar + vector of 2. X on scalar, measure vector (should be (0, 0))."""
        scalar = qmc.qubit("scalar")
        vec = qmc.qubit_array(2, "vec")
        scalar = qmc.x(scalar)
        return qmc.measure(vec)

    @staticmethod
    @qmc.qkernel
    def mixed_both_measured() -> tuple[qmc.Bit, qmc.Vector[qmc.Bit]]:
        """Scalar + vector, X on scalar only, measure both.
        Expect (1, (0, 0))."""
        scalar = qmc.qubit("scalar")
        vec = qmc.qubit_array(2, "vec")
        scalar = qmc.x(scalar)
        return qmc.measure(scalar), qmc.measure(vec)

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_scalar_from_vector_circuit(self, transpiler_factory):
        """X on vec[0], measure scalar → 0."""
        t = transpiler_factory()
        results = _sample_values(t, self.mixed_measure_scalar_from_vector_circuit)
        _assert_all_equal(results, 0, label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_vector_with_scalar_excited(self, transpiler_factory):
        """X on scalar, measure vector → (0, 0)."""
        t = transpiler_factory()
        results = _sample_values(t, self.mixed_measure_vector_with_scalar_excited)
        _assert_all_equal(results, (0, 0), label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_mixed_both_measured(self, transpiler_factory):
        """X on scalar, measure scalar + vector → (1, (0, 0))."""
        t = transpiler_factory()
        results = _sample_values(t, self.mixed_both_measured)
        _assert_all_equal(results, (1, (0, 0)), label=f"[{transpiler_factory.__name__}] ")


# ---------------------------------------------------------------------------
# 4. Parametric circuits with later binding
# ---------------------------------------------------------------------------


class TestParametricPartialMeasurement:
    """Parametric circuits where angles are bound at sample time."""

    @staticmethod
    @qmc.qkernel
    def parametric_measure_second(theta: qmc.Float) -> qmc.Bit:
        """RX(theta) on q0, measure q1. Regardless of theta, q1 = 0."""
        q0 = qmc.qubit("q0")
        q1 = qmc.qubit("q1")
        q0 = qmc.rx(q0, theta)
        return qmc.measure(q1)

    @staticmethod
    @qmc.qkernel
    def parametric_measure_first(theta: qmc.Float) -> qmc.Bit:
        """RX(pi) on q0, RX(theta) on q1, measure q0. q0 should be 1."""
        q0 = qmc.qubit("q0")
        q1 = qmc.qubit("q1")
        q0 = qmc.rx(q0, math.pi)
        q1 = qmc.rx(q1, theta)
        return qmc.measure(q0)

    @pytest.mark.parametrize("theta_val", [0.0, math.pi / 4, math.pi / 2, math.pi])
    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_parametric_measure_untouched(self, transpiler_factory, theta_val):
        """RX(theta) on q0, measure q1 → 0."""
        t = transpiler_factory()
        results = _sample_values(
            t,
            self.parametric_measure_second,
            transpile_kwargs={"parameters": ["theta"]},
            sample_kwargs={"bindings": {"theta": theta_val}},
        )
        _assert_all_equal(
            results, 0,
            label=f"[{transpiler_factory.__name__}, theta={theta_val}] ",
        )

    @pytest.mark.parametrize("theta_val", [0.0, math.pi / 2, math.pi])
    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_parametric_measure_flipped(self, transpiler_factory, theta_val):
        """RX(pi) on q0, RX(theta) on q1, measure q0 → 1."""
        t = transpiler_factory()
        results = _sample_values(
            t,
            self.parametric_measure_first,
            transpile_kwargs={"parameters": ["theta"]},
            sample_kwargs={"bindings": {"theta": theta_val}},
        )
        _assert_all_equal(
            results, 1,
            label=f"[{transpiler_factory.__name__}, theta={theta_val}] ",
        )


# ---------------------------------------------------------------------------
# 5. Multiple selective measurements in varied order
# ---------------------------------------------------------------------------


class TestMultipleSelectiveMeasurements:
    """Measure individual qubits from a larger register in non-sequential order."""

    @staticmethod
    @qmc.qkernel
    def measure_first_and_last() -> tuple[qmc.Bit, qmc.Bit]:
        """3 qubits, X on q0 and q2, measure q0 and q2. Expect (1, 1)."""
        qs = qmc.qubit_array(3, "qs")
        qs[0] = qmc.x(qs[0])
        qs[2] = qmc.x(qs[2])
        return qmc.measure(qs[0]), qmc.measure(qs[2])

    @staticmethod
    @qmc.qkernel
    def measure_reverse_order() -> tuple[qmc.Bit, qmc.Bit]:
        """3 qubits, X on q0. Measure q2 first, then q0.
        Expect (0, 1) — q2 is 0, q0 is 1."""
        qs = qmc.qubit_array(3, "qs")
        qs[0] = qmc.x(qs[0])
        return qmc.measure(qs[2]), qmc.measure(qs[0])

    @staticmethod
    @qmc.qkernel
    def measure_all_individually() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
        """3 qubits, X on q1 only. Measure all individually.
        Expect (0, 1, 0)."""
        qs = qmc.qubit_array(3, "qs")
        qs[1] = qmc.x(qs[1])
        return qmc.measure(qs[0]), qmc.measure(qs[1]), qmc.measure(qs[2])

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_first_and_last(self, transpiler_factory):
        t = transpiler_factory()
        results = _sample_values(t, self.measure_first_and_last)
        _assert_all_equal(results, (1, 1), label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_reverse_order(self, transpiler_factory):
        t = transpiler_factory()
        results = _sample_values(t, self.measure_reverse_order)
        _assert_all_equal(results, (0, 1), label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_all_individually(self, transpiler_factory):
        t = transpiler_factory()
        results = _sample_values(t, self.measure_all_individually)
        _assert_all_equal(results, (0, 1, 0), label=f"[{transpiler_factory.__name__}] ")


# ---------------------------------------------------------------------------
# 6. Measurement of all qubits (regression: ensure this still works)
# ---------------------------------------------------------------------------


class TestFullMeasurement:
    """Measure all qubits — should work identically to before the fix."""

    @staticmethod
    @qmc.qkernel
    def measure_all_vector() -> qmc.Vector[qmc.Bit]:
        """2 qubits, X on q0, measure all. Expect (1, 0)."""
        qs = qmc.qubit_array(2, "qs")
        qs[0] = qmc.x(qs[0])
        return qmc.measure(qs)

    @staticmethod
    @qmc.qkernel
    def bell_measure_all() -> qmc.Vector[qmc.Bit]:
        """Bell state, measure all — expect either (0,0) or (1,1)."""
        qs = qmc.qubit_array(2, "qs")
        qs[0] = qmc.h(qs[0])
        qs[0], qs[1] = qmc.cx(qs[0], qs[1])
        return qmc.measure(qs)

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_full_vector_measurement(self, transpiler_factory):
        t = transpiler_factory()
        results = _sample_values(t, self.measure_all_vector)
        _assert_all_equal(results, (1, 0), label=f"[{transpiler_factory.__name__}] ")

    @pytest.mark.parametrize(
        "transpiler_factory",
        [QuriPartsTranspiler, QiskitTranspiler],
        ids=["quri_parts", "qiskit"],
    )
    def test_bell_state_measurement(self, transpiler_factory):
        t = transpiler_factory()
        results = _sample_values(t, self.bell_measure_all, shots=1024)
        for value, count in results:
            assert value in ((0, 0), (1, 1)), (
                f"[{transpiler_factory.__name__}] Bell state produced {value}"
            )
