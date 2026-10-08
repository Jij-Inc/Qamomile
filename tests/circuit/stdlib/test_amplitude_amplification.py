"""Execution and serialization tests for amplitude amplification."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.serialization import deserialize, serialize
from qamomile.circuit.stdlib.amplitude_amplification.schedule import _fpaa_gamma


def _ising_encoding(identity: float, single: float) -> qmc.IsingZBlockEncoding:
    """Build a two-term diagonal encoding with a known success probability.

    The encoded operator is ``A = identity * I + single * Z_0`` with
    subnormalization ``alpha = |identity| + |single|``. Applied to the all-zero
    system state it acts as multiplication by ``identity + single``, so the
    all-zero-signal outcome has probability
    ``|identity + single|**2 / alpha**2``.

    Args:
        identity (float): Coefficient of the identity word.
        single (float): Coefficient of the ``Z_0`` word.

    Returns:
        qmc.IsingZBlockEncoding: Two-term diagonal block encoding.
    """
    return qmc.ising_z_block_encoding(
        {(): complex(identity), (0,): complex(single)},
        num_system_qubits=1,
    )


def _analytic_success_probability(identity: float, single: float) -> float:
    """Return the unamplified all-zero-signal probability of ``_ising_encoding``.

    Args:
        identity (float): Coefficient of the identity word.
        single (float): Coefficient of the ``Z_0`` word.

    Returns:
        float: Initial success probability ``a``.
    """
    normalization = abs(identity) + abs(single)
    return abs(identity + single) ** 2 / normalization**2


def _zero_projector_observable(num_qubits: int) -> qm_o.Hamiltonian:
    """Return the projector onto an all-zero register as a Hamiltonian.

    Args:
        num_qubits (int): Positive register width.

    Returns:
        qm_o.Hamiltonian: Product of ``(I + Z_i) / 2`` over every qubit.
    """
    projector = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    identity = qm_o.Hamiltonian.identity(num_qubits=num_qubits)
    for index in range(num_qubits):
        projector = projector * (0.5 * (identity + qm_o.Z(index)))
    return projector


def _standard_template(iterations: int) -> qmc.QKernel:
    """Build a template amplifying a bound encoding a fixed number of rounds.

    Args:
        iterations (int): Number of standard amplification rounds.

    Returns:
        qmc.QKernel: Template with a static ``block_encoding`` slot.
    """

    @qmc.qkernel
    def template(
        block_encoding: qmc.LCUBlockEncoding,
        observable: qmc.Observable,
    ) -> qmc.Float:
        """Estimate the all-zero-signal probability after amplification.

        Args:
            block_encoding (qmc.LCUBlockEncoding): Static block encoding whose
                post-selection success is amplified.
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Expected value of the all-zero signal projector.
        """
        signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            block_encoding.unitary,
            qmc.standard_amplification_schedule(iterations),
        )
        return qmc.expval(signal, observable)

    return template


def _encoding_template(encoding: Any, iterations: int) -> qmc.QKernel:
    """Build a closed-over template amplifying one concrete encoding.

    Args:
        encoding (Any): Block encoding baked into the kernel body.
        iterations (int): Number of standard amplification rounds.

    Returns:
        qmc.QKernel: Kernel estimating the all-zero-signal probability.
    """

    @qmc.qkernel
    def template(observable: qmc.Observable) -> qmc.Float:
        """Estimate the all-zero-signal probability after amplification.

        Args:
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Expected value of the all-zero signal projector.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            encoding.unitary,
            qmc.standard_amplification_schedule(iterations),
        )
        return qmc.expval(signal, observable)

    return template


def _scheduled_template(
    encoding: Any,
    schedule: Any,
    preparation: Any | None = None,
) -> qmc.QKernel:
    """Build a template applying one explicit schedule.

    Args:
        encoding (Any): Block encoding supplying the register widths.
        schedule (Any): Amplification schedule to apply.
        preparation (Any | None): Preparation to amplify. Defaults to the
            encoding unitary.

    Returns:
        qmc.QKernel: Kernel estimating the all-zero-signal probability.
    """
    target = encoding.unitary if preparation is None else preparation

    @qmc.qkernel
    def template(observable: qmc.Observable) -> qmc.Float:
        """Estimate the all-zero-signal probability after amplification.

        Args:
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Expected value of the all-zero signal projector.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(signal, system, target, schedule)
        return qmc.expval(signal, observable)

    return template


def _preparation_template(
    encoding: Any,
    preparation: Any,
    iterations: int,
) -> qmc.QKernel:
    """Build a template amplifying a composed state preparation.

    Args:
        encoding (Any): Block encoding supplying the register widths.
        preparation (Any): State preparation with the block-encoding ABI.
        iterations (int): Number of standard amplification rounds.

    Returns:
        qmc.QKernel: Kernel estimating the all-zero-signal probability.
    """

    @qmc.qkernel
    def template(observable: qmc.Observable) -> qmc.Float:
        """Estimate the all-zero-signal probability after amplification.

        Args:
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Expected value of the all-zero signal projector.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            preparation,
            qmc.standard_amplification_schedule(iterations),
        )
        return qmc.expval(signal, observable)

    return template


def _fixed_point_template(
    success_probability_bound: float,
    failure_tolerance: float,
    query_count: int | None = None,
) -> qmc.QKernel:
    """Build a template applying the fixed-point schedule to a bound encoding.

    Args:
        success_probability_bound (float): Lower bound ``lambda``.
        failure_tolerance (float): Target failure tolerance ``delta``.
        query_count (int | None): Explicit odd query count. Defaults to
            ``None``.

    Returns:
        qmc.QKernel: Template with a static ``block_encoding`` slot.
    """

    @qmc.qkernel
    def template(
        block_encoding: qmc.LCUBlockEncoding,
        observable: qmc.Observable,
    ) -> qmc.Float:
        """Estimate the all-zero-signal probability after amplification.

        Args:
            block_encoding (qmc.LCUBlockEncoding): Static block encoding whose
                post-selection success is amplified.
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Expected value of the all-zero signal projector.
        """
        signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            block_encoding.unitary,
            qmc.fixed_point_amplification_schedule(
                success_probability_bound,
                failure_tolerance,
                query_count,
            ),
        )
        return qmc.expval(signal, observable)

    return template


@qmc.qkernel
def _single_round_template(
    block_encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Amplify a bound encoding for exactly one standard round.

    Args:
        block_encoding (qmc.LCUBlockEncoding): Static block encoding to
            amplify.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal and
            system registers.
    """
    signal = qmc.qubit_array(block_encoding.num_signal_qubits, "signal")
    system = qmc.qubit_array(block_encoding.num_system_qubits, "system")
    signal, system = qmc.amplitude_amplification(
        signal,
        system,
        block_encoding.unitary,
        qmc.standard_amplification_schedule(1),
    )
    return qmc.measure(signal), qmc.measure(system)


@qmc.qkernel
def _two_encoding_template(
    first_encoding: qmc.LCUBlockEncoding,
    second_encoding: qmc.LCUBlockEncoding,
) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
    """Amplify two independently bound encodings in one template.

    Args:
        first_encoding (qmc.LCUBlockEncoding): First static block encoding.
        second_encoding (qmc.LCUBlockEncoding): Second static block encoding.

    Returns:
        tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measured signal
            registers of both amplifications.
    """
    first_signal = qmc.qubit_array(first_encoding.num_signal_qubits, "first_signal")
    first_system = qmc.qubit_array(first_encoding.num_system_qubits, "first_system")
    first_signal, first_system = qmc.amplitude_amplification(
        first_signal,
        first_system,
        first_encoding.unitary,
        qmc.standard_amplification_schedule(1),
    )
    second_signal = qmc.qubit_array(second_encoding.num_signal_qubits, "second_signal")
    second_system = qmc.qubit_array(second_encoding.num_system_qubits, "second_system")
    second_signal, second_system = qmc.amplitude_amplification(
        second_signal,
        second_system,
        second_encoding.unitary,
        qmc.standard_amplification_schedule(1),
    )
    _ = qmc.measure(first_system)
    _ = qmc.measure(second_system)
    return qmc.measure(first_signal), qmc.measure(second_signal)


@qmc.qkernel
def _uniform_system_preparation(
    system: qmc.Vector[qmc.Qubit],
) -> qmc.Vector[qmc.Qubit]:
    """Prepare a uniform superposition on the system register.

    Args:
        system (qmc.Vector[qmc.Qubit]): All-zero system register.

    Returns:
        qmc.Vector[qmc.Qubit]: System register in a uniform superposition.
    """
    return qmc.h(system)


@qmc.qkernel
def _allocating_preparation(
    signal: qmc.Vector[qmc.Qubit],
    system: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return a preparation that allocates internally and cannot be inverted.

    Args:
        signal (qmc.Vector[qmc.Qubit]): Signal register.
        system (qmc.Vector[qmc.Qubit]): System register.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Unchanged
            registers after an internal allocation.
    """
    scratch = qmc.qubit("scratch")
    scratch = qmc.h(scratch)
    system[0], scratch = qmc.cx(system[0], scratch)
    return signal, system


@qmc.qkernel
def _reversed_preparation(
    system: qmc.Vector[qmc.Qubit],
    signal: qmc.Vector[qmc.Qubit],
) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
    """Return registers through an intentionally reversed preparation ABI.

    Args:
        system (qmc.Vector[qmc.Qubit]): System register.
        signal (qmc.Vector[qmc.Qubit]): Signal register.

    Returns:
        tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]: Unchanged
            registers.
    """
    return system, signal


def _bare_block_kernel(encoding: Any) -> qmc.QKernel:
    """Build an allocation-only kernel applying a block encoding once.

    Args:
        encoding (Any): Block encoding to materialize densely.

    Returns:
        qmc.QKernel: Kernel applying the bare block-encoding unitary.
    """

    @qmc.qkernel
    def kernel() -> qmc.Bit:
        """Apply the bare block encoding for dense inspection.

        Returns:
            qmc.Bit: Unused measurement keeping the kernel well formed.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = encoding.unitary(signal, system)
        return qmc.measure(signal[0])

    return kernel


def _run_closed_over(transpiler: Any, template: qmc.QKernel, width: int) -> float:
    """Transpile and run a template whose encoding is closed over.

    Args:
        transpiler (Any): Qiskit transpiler fixture.
        template (qmc.QKernel): Kernel taking only the projector observable.
        width (int): Signal-register width.

    Returns:
        float: Estimated all-zero-signal probability.
    """
    executable = transpiler.transpile(
        template,
        bindings={"observable": _zero_projector_observable(width)},
    )
    return float(executable.run(transpiler.executor()).result())


def _run_expectation(transpiler: Any, template: qmc.QKernel, encoding: Any) -> float:
    """Transpile and run an expectation-value template on one encoding.

    Args:
        transpiler (Any): Qiskit transpiler fixture.
        template (qmc.QKernel): Template with a ``block_encoding`` slot.
        encoding (Any): Block encoding to bind.

    Returns:
        float: Estimated all-zero-signal probability.
    """
    executable = transpiler.transpile(
        template,
        bindings={
            "block_encoding": encoding,
            "observable": _zero_projector_observable(encoding.num_signal_qubits),
        },
    )
    return float(executable.run(transpiler.executor()).result())


# --------------------------------------------------------------------------
# Classical schedule helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("success_probability", "expected"),
    [(1.0, 0), (0.5, 0), (0.25, 1), (0.05, 3), (0.01, 7)],
)
def test_iteration_count_matches_optimal_rotation(
    success_probability: float,
    expected: int,
) -> None:
    """The optimal iteration count follows ``round(pi / (4 theta) - 1/2)``."""
    assert qmc.amplitude_amplification_iteration_count(success_probability) == expected


def test_iteration_count_maximizes_the_sine_schedule() -> None:
    """The reported count beats every other count below twice its value."""
    for probability in (0.02, 0.09, 0.3, 0.77):
        angle = math.asin(math.sqrt(probability))
        best = qmc.amplitude_amplification_iteration_count(probability)
        best_value = math.sin((2 * best + 1) * angle) ** 2
        for candidate in range(0, 2 * best + 3):
            value = math.sin((2 * candidate + 1) * angle) ** 2
            assert value <= best_value + 1e-12


@pytest.mark.parametrize("value", [0.0, -0.1, 1.5])
def test_iteration_count_rejects_out_of_range(value: float) -> None:
    """Probabilities outside ``(0, 1]`` are rejected."""
    with pytest.raises(ValueError):
        qmc.amplitude_amplification_iteration_count(value)


def test_iteration_count_rejects_boolean() -> None:
    """Booleans are rejected rather than silently coerced."""
    with pytest.raises(TypeError):
        qmc.amplitude_amplification_iteration_count(True)


@pytest.mark.parametrize("bound", [0.01, 0.05, 0.19, 0.5, 0.9])
@pytest.mark.parametrize("tolerance", [0.05, 0.1, 0.3])
def test_query_count_is_the_smallest_admissible_odd_length(
    bound: float,
    tolerance: float,
) -> None:
    """The schedule derives the minimal admissible odd ``L``."""
    count = qmc.fixed_point_amplification_schedule(bound, tolerance).query_count

    assert count >= 1
    assert count % 2 == 1
    assert 1.0 - _fpaa_gamma(count, tolerance) ** 2 <= bound + 1e-12
    if count > 1:
        smaller = count - 2
        assert 1.0 - _fpaa_gamma(smaller, tolerance) ** 2 > bound


def test_query_count_is_one_for_certain_success() -> None:
    """A bound of one needs no amplification rounds."""
    assert qmc.fixed_point_amplification_schedule(1.0, 0.1).query_count == 1


@pytest.mark.parametrize("tolerance", [0.0, 1.0, 1.5])
def test_query_count_rejects_invalid_tolerance(tolerance: float) -> None:
    """Failure tolerances outside ``(0, 1)`` are rejected."""
    with pytest.raises(ValueError):
        qmc.fixed_point_amplification_schedule(0.1, tolerance)


def test_phase_schedule_length_and_pairing() -> None:
    """The schedule holds ``L - 1`` phases mirrored between the two lists."""
    tolerance = 0.3
    count = 7
    phases = qmc.fixed_point_amplification_schedule(
        failure_tolerance=tolerance, query_count=count
    ).phases

    assert len(phases) == count - 1
    good = phases[0::2]
    zero = phases[1::2]
    assert good == tuple(reversed(zero))


def test_phase_schedule_is_empty_for_a_single_query() -> None:
    """A single query is the bare preparation, so no phases are emitted."""
    assert (
        qmc.fixed_point_amplification_schedule(
            failure_tolerance=0.1, query_count=1
        ).phases
        == ()
    )


def test_phase_schedule_matches_reference_values() -> None:
    """The schedule reproduces the Yoder-Low-Chuang closed form."""
    tolerance = 0.3
    count = 5
    gamma = _fpaa_gamma(count, tolerance)
    scale = math.sqrt(1.0 - gamma * gamma)
    kappa = [
        2.0 * math.atan2(1.0, math.tan(2.0 * math.pi * j / count) * scale)
        for j in (1, 2)
    ]
    expected = (-kappa[1], -kappa[0], -kappa[0], -kappa[1])

    phases = qmc.fixed_point_amplification_schedule(
        failure_tolerance=tolerance, query_count=count
    ).phases

    assert phases == pytest.approx(expected, abs=1e-12)


def test_phase_schedule_requires_a_bound_or_a_count() -> None:
    """Omitting both the bound and the count is an error."""
    with pytest.raises(ValueError):
        qmc.fixed_point_amplification_schedule(failure_tolerance=0.1)


def test_phase_schedule_rejects_even_query_count() -> None:
    """The fixed-point sequence length must be odd."""
    with pytest.raises(ValueError):
        qmc.fixed_point_amplification_schedule(failure_tolerance=0.1, query_count=4)


def test_fixed_point_success_probability_is_identity_for_one_query() -> None:
    """A single query leaves the success probability unchanged."""
    for probability in (0.05, 0.4, 0.95):
        observed = qmc.fixed_point_amplification_schedule(
            failure_tolerance=0.1, query_count=1
        ).success_probability(probability)
        assert observed == pytest.approx(probability, abs=1e-12)


@pytest.mark.parametrize("bound", [0.02, 0.1, 0.4])
@pytest.mark.parametrize("tolerance", [0.05, 0.2, 0.5])
def test_fixed_point_success_probability_meets_the_guarantee(
    bound: float,
    tolerance: float,
) -> None:
    """Above the bound the closed form never drops below ``1 - delta**2``."""
    schedule = qmc.fixed_point_amplification_schedule(bound, tolerance)
    for probability in np.linspace(bound, 0.999, 40):
        observed = schedule.success_probability(float(probability))
        assert observed >= 1.0 - tolerance**2 - 1e-12
        assert observed <= 1.0 + 1e-12


def test_fixed_point_success_probability_uses_the_hyperbolic_branch() -> None:
    """Below the admissible bound the Chebyshev argument exceeds one."""
    count = qmc.fixed_point_amplification_schedule(0.5, 0.1).query_count
    gamma = _fpaa_gamma(count, 0.1)
    probability = 0.001

    assert math.sqrt(1.0 - probability) / gamma > 1.0
    observed = qmc.fixed_point_amplification_schedule(
        failure_tolerance=0.1, query_count=count
    ).success_probability(probability)
    assert observed < 1.0 - 0.1**2


def test_schedule_success_probability_agrees_with_the_generic_sequence() -> None:
    """Each closed form matches the generic two-level evaluation exactly."""
    generic = qmc.AmplificationSchedule.success_probability

    for iterations in range(5):
        schedule = qmc.standard_amplification_schedule(iterations)
        for probability in (0.05, 0.1, 0.5, 0.9, 1.0):
            assert schedule.success_probability(probability) == pytest.approx(
                generic(schedule, probability), abs=1e-12
            )

    for bound, tolerance in ((0.05, 0.1), (0.05, 0.3), (0.19, 0.2), (0.5, 0.05)):
        schedule = qmc.fixed_point_amplification_schedule(bound, tolerance)
        for probability in (bound, min(1.0, 2 * bound), 0.5, 0.9, 1.0):
            assert schedule.success_probability(probability) == pytest.approx(
                generic(schedule, probability), abs=1e-12
            )


# --------------------------------------------------------------------------
# Schedule and preparation descriptors
# --------------------------------------------------------------------------


def _field_names(descriptor: Any) -> tuple[str, ...]:
    """Return the declared dataclass field names of a descriptor.

    Args:
        descriptor (Any): Descriptor instance or class.

    Returns:
        tuple[str, ...]: Declared field names in order.
    """
    return tuple(field.name for field in dataclasses.fields(descriptor))


def test_schedules_are_frozen_noncallable_identity_objects() -> None:
    """Schedules follow the block-encoding descriptor contract."""
    base = qmc.AmplificationSchedule((0.4, -0.7))
    standard = qmc.standard_amplification_schedule(2)
    fixed_point = qmc.fixed_point_amplification_schedule(0.05, 0.3)

    assert _field_names(qmc.AmplificationSchedule) == ("phases",)
    assert _field_names(qmc.StandardAmplificationSchedule) == ("phases",)
    assert _field_names(fixed_point) == ("phases", "failure_tolerance")
    for schedule in (base, standard, fixed_point):
        assert not callable(schedule)
        assert not hasattr(schedule, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            schedule.phases = ()  # type: ignore[misc]

    repeated = qmc.fixed_point_amplification_schedule(0.05, 0.3)
    assert fixed_point is not repeated
    assert fixed_point != repeated


def test_schedule_derives_its_round_and_query_counts() -> None:
    """Round and query counts follow from the phases alone."""
    standard = qmc.standard_amplification_schedule(3)

    assert standard.iterations == 3
    assert standard.num_rounds == 3
    assert standard.query_count == 7
    assert len(standard.phases) == 6
    assert qmc.standard_amplification_schedule(0).query_count == 1


def test_standard_schedule_rejects_non_hard_reflections() -> None:
    """The standard schedule is exactly a list of pi phases."""
    with pytest.raises(ValueError, match="math.pi"):
        qmc.StandardAmplificationSchedule((0.3, 0.3))


def test_fixed_point_schedule_rejects_mismatched_phases() -> None:
    """A fixed-point schedule must carry the phases its formula describes."""
    with pytest.raises(ValueError, match="Yoder-Low-Chuang"):
        qmc.FixedPointAmplificationSchedule((0.1, 0.2), 0.3)


@pytest.mark.parametrize("tolerance", [0.0, 1.0, 1.5])
def test_fixed_point_schedule_rejects_invalid_tolerance(tolerance: float) -> None:
    """Failure tolerances outside ``(0, 1)`` are rejected by the descriptor."""
    phases = qmc.fixed_point_amplification_schedule(0.05, 0.3).phases
    with pytest.raises(ValueError):
        qmc.FixedPointAmplificationSchedule(phases, tolerance)


def test_schedule_replace_revalidates() -> None:
    """``dataclasses.replace`` re-runs the descriptor's own validation."""
    fixed_point = qmc.fixed_point_amplification_schedule(0.05, 0.3)
    standard = qmc.standard_amplification_schedule(1)

    with pytest.raises(ValueError, match="Yoder-Low-Chuang"):
        dataclasses.replace(fixed_point, phases=(0.1, 0.2))

    widened = dataclasses.replace(standard, phases=(math.pi,) * 4)
    assert widened.iterations == 2


def test_fixed_point_bound_is_at_least_as_tight_as_requested() -> None:
    """Rounding the query count up covers more than the requested bound."""
    for bound, tolerance in ((0.05, 0.1), (0.05, 0.3), (0.19, 0.1)):
        schedule = qmc.fixed_point_amplification_schedule(bound, tolerance)
        admissible = schedule.admissible_success_probability_bound
        assert admissible <= bound + 1e-12
        assert schedule.guaranteed_success_probability == pytest.approx(
            1.0 - tolerance**2
        )
        assert (
            schedule.success_probability(admissible)
            >= schedule.guaranteed_success_probability - 1e-9
        )


def test_preparation_descriptor_is_frozen_noncallable_and_identity_based() -> None:
    """The preparation descriptor follows the same contract as a block encoding."""
    encoding = _ising_encoding(1.0, -0.4)
    prepared = qmc.amplification_preparation(encoding)

    assert _field_names(prepared) == (
        "preparation",
        "num_signal_qubits",
        "num_system_qubits",
    )
    assert not callable(prepared)
    assert not hasattr(prepared, "__dict__")
    assert tuple(prepared.preparation.signature.parameters) == ("signal", "system")
    with pytest.raises(dataclasses.FrozenInstanceError):
        prepared.num_signal_qubits = 3  # type: ignore[misc]

    repeated = qmc.amplification_preparation(encoding)
    assert prepared is not repeated
    assert prepared != repeated


def test_preparation_descriptor_rejects_invalid_fields() -> None:
    """Widths and the preparation ABI are validated at construction."""
    encoding = _ising_encoding(1.0, -0.4)
    unitary = encoding.unitary

    with pytest.raises(TypeError, match="QKernel"):
        qmc.AmplificationPreparation(object(), 1, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="signature"):
        qmc.AmplificationPreparation(_reversed_preparation, 1, 1)
    with pytest.raises(TypeError):
        qmc.AmplificationPreparation(unitary, True, 1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        qmc.AmplificationPreparation(unitary, 0, 1)


# --------------------------------------------------------------------------
# Circuit execution
# --------------------------------------------------------------------------


@pytest.mark.parametrize("single", [-0.8, -0.5, -0.2, 0.3])
@pytest.mark.parametrize("iterations", [0, 1, 2, 3])
def test_standard_amplification_matches_the_sine_schedule(
    qiskit_transpiler: Any,
    single: float,
    iterations: int,
) -> None:
    """Standard amplification reproduces ``sin**2((2k + 1) theta)``."""
    encoding = _ising_encoding(1.0, single)
    initial = _analytic_success_probability(1.0, single)
    angle = math.asin(math.sqrt(initial))
    expected = math.sin((2 * iterations + 1) * angle) ** 2

    observed = _run_expectation(
        qiskit_transpiler,
        _standard_template(iterations),
        encoding,
    )

    assert observed == pytest.approx(expected, abs=1e-8)


def test_zero_iterations_leave_the_encoding_unamplified(
    qiskit_transpiler: Any,
) -> None:
    """Zero rounds emit the bare preparation."""
    encoding = _ising_encoding(1.0, -0.6)
    expected = _analytic_success_probability(1.0, -0.6)

    observed = _run_expectation(qiskit_transpiler, _standard_template(0), encoding)

    assert observed == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("query_count", [1, 3, 5, 7])
@pytest.mark.parametrize("single", [-0.8, -0.4, 0.2])
def test_fixed_point_matches_the_closed_form(
    qiskit_transpiler: Any,
    query_count: int,
    single: float,
) -> None:
    """The emitted fixed-point sequence matches its analytic probability."""
    tolerance = 0.3
    encoding = _ising_encoding(1.0, single)
    initial = _analytic_success_probability(1.0, single)
    expected = qmc.fixed_point_amplification_schedule(
        initial, tolerance, query_count
    ).success_probability(initial)

    observed = _run_expectation(
        qiskit_transpiler,
        _fixed_point_template(initial, tolerance, query_count),
        encoding,
    )

    assert observed == pytest.approx(expected, abs=1e-8)


@pytest.mark.parametrize("single", [-0.6, -0.45, -0.2, 0.0, 0.5])
def test_fixed_point_meets_tolerance_across_the_admissible_range(
    qiskit_transpiler: Any,
    single: float,
) -> None:
    """One schedule clears ``1 - delta**2`` for every amplitude above the bound."""
    bound = 0.05
    tolerance = 0.3
    encoding = _ising_encoding(1.0, single)
    initial = _analytic_success_probability(1.0, single)
    assert initial >= bound

    observed = _run_expectation(
        qiskit_transpiler,
        _fixed_point_template(bound, tolerance),
        encoding,
    )

    assert observed >= 1.0 - tolerance**2 - 1e-8


def test_fixed_point_beats_a_mistuned_standard_schedule(
    qiskit_transpiler: Any,
) -> None:
    """Standard amplification collapses where the fixed-point schedule holds."""
    encoding = _ising_encoding(1.0, -0.6)
    initial = _analytic_success_probability(1.0, -0.6)
    angle = math.asin(math.sqrt(initial))
    worst = min(
        range(13),
        key=lambda rounds: math.sin((2 * rounds + 1) * angle) ** 2,
    )
    worst_value = math.sin((2 * worst + 1) * angle) ** 2
    assert worst_value < 0.1

    overshot = _run_expectation(
        qiskit_transpiler,
        _standard_template(worst),
        encoding,
    )
    fixed_point = _run_expectation(
        qiskit_transpiler,
        _fixed_point_template(0.05, 0.3),
        encoding,
    )

    assert overshot == pytest.approx(worst_value, abs=1e-8)
    assert fixed_point >= 1.0 - 0.3**2 - 1e-8


def test_amplification_of_a_dense_block_encoding_matches_the_top_left_block(
    qiskit_transpiler: Any,
) -> None:
    """The amplified probability follows the dense block's action on ``|0>``."""
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import Operator

    encoding = qmc.ising_z_block_encoding(
        {(): 0.6 + 0.0j, (0,): -0.25 + 0.0j, (1,): 0.15 + 0.0j},
        num_system_qubits=2,
    )
    circuit = qiskit_transpiler.transpile(_bare_block_kernel(encoding)).quantum_circuit
    unitary = np.asarray(
        Operator(circuit.remove_final_measurements(inplace=False)).data
    )
    system_indices = [
        index << encoding.num_signal_qubits
        for index in range(1 << encoding.num_system_qubits)
    ]
    block = unitary[np.ix_(system_indices, system_indices)]
    initial = float(np.sum(np.abs(block[:, 0]) ** 2))
    angle = math.asin(math.sqrt(initial))

    for iterations in (0, 1, 2):
        observed = _run_closed_over(
            qiskit_transpiler,
            _encoding_template(encoding, iterations),
            encoding.num_signal_qubits,
        )
        expected = math.sin((2 * iterations + 1) * angle) ** 2
        assert observed == pytest.approx(expected, abs=1e-8)


def test_generic_schedule_matches_a_simulated_circuit(
    qiskit_transpiler: Any,
) -> None:
    """A hand-written schedule's prediction matches the emitted circuit.

    This pins the sign and ordering convention of the generic evaluation in
    ``AmplificationSchedule.success_probability`` against real gates, which no
    purely classical comparison can do.
    """
    encoding = _ising_encoding(1.0, -0.5)
    initial = _analytic_success_probability(1.0, -0.5)
    schedule = qmc.AmplificationSchedule((0.7, -1.9, 2.4, 0.35))

    observed = _run_closed_over(
        qiskit_transpiler,
        _scheduled_template(encoding, schedule),
        encoding.num_signal_qubits,
    )

    assert observed == pytest.approx(schedule.success_probability(initial), abs=1e-8)


def test_amplification_accepts_a_preparation_descriptor(
    qiskit_transpiler: Any,
) -> None:
    """Passing the descriptor and passing its kernel emit the same circuit."""
    encoding = _ising_encoding(1.0, -0.5)
    prepared = qmc.amplification_preparation(encoding)
    schedule = qmc.standard_amplification_schedule(1)

    through_descriptor = _run_closed_over(
        qiskit_transpiler,
        _scheduled_template(encoding, schedule, preparation=prepared),
        encoding.num_signal_qubits,
    )
    through_kernel = _run_closed_over(
        qiskit_transpiler,
        _scheduled_template(encoding, schedule, preparation=prepared.preparation),
        encoding.num_signal_qubits,
    )

    assert through_descriptor == pytest.approx(through_kernel, abs=1e-12)


def test_single_qubit_signal_register_amplifies(qiskit_transpiler: Any) -> None:
    """A one-qubit signal register uses a plain CX and stays exact."""
    encoding = qmc.identity_block_encoding(2)
    assert encoding.num_signal_qubits == 1

    observed = _run_expectation(qiskit_transpiler, _standard_template(2), encoding)

    assert observed == pytest.approx(1.0, abs=1e-8)


def test_system_preparation_is_folded_into_the_amplified_state(
    qiskit_transpiler: Any,
) -> None:
    """A non-zero system input is amplified when it is part of ``A``."""
    encoding = qmc.ising_z_block_encoding(
        {(): 0.7 + 0.0j, (0,): -0.3 + 0.0j},
        num_system_qubits=1,
    )
    preparation = qmc.amplification_preparation(
        encoding,
        _uniform_system_preparation,
    )
    width = encoding.num_signal_qubits

    initial = _run_closed_over(
        qiskit_transpiler,
        _preparation_template(encoding, preparation, 0),
        width,
    )
    observed = _run_closed_over(
        qiskit_transpiler,
        _preparation_template(encoding, preparation, 1),
        width,
    )
    angle = math.asin(math.sqrt(initial))

    assert observed == pytest.approx(math.sin(3 * angle) ** 2, abs=1e-8)


def test_preparation_passes_the_encoding_unitary_through() -> None:
    """Omitting the system preparation wraps the encoding unitary directly."""
    encoding = _ising_encoding(1.0, -0.4)

    prepared = qmc.amplification_preparation(encoding)

    assert prepared.num_signal_qubits == encoding.num_signal_qubits
    assert prepared.num_system_qubits == encoding.num_system_qubits
    assert prepared.preparation.name == encoding.unitary.name
    assert prepared.preparation.input_types == encoding.unitary.input_types
    assert prepared.preparation.output_types == encoding.unitary.output_types


def test_sampled_amplification_matches_the_statevector(
    qiskit_transpiler: Any,
    seeded_executor: Any,
) -> None:
    """Sampling the amplified template agrees with the analytic probability."""
    encoding = _ising_encoding(1.0, -0.5)
    initial = _analytic_success_probability(1.0, -0.5)
    angle = math.asin(math.sqrt(initial))
    expected = math.sin(3 * angle) ** 2

    executable = qiskit_transpiler.transpile(
        _single_round_template,
        bindings={"block_encoding": encoding},
    )
    result = executable.sample(seeded_executor, shots=4096).result()
    shots = sum(count for _, count in result.results)
    successes = sum(
        count
        for value, count in result.results
        if int(value[0][0]) == 0  # noqa: E501
    )

    assert successes / shots == pytest.approx(expected, abs=0.05)


# --------------------------------------------------------------------------
# Serialization, structure and resources
# --------------------------------------------------------------------------


def test_serialized_template_binds_the_encoding_slot(qiskit_transpiler: Any) -> None:
    """A round-tripped template still resolves its static encoding slot."""
    payload = serialize(_single_round_template)
    restored = deserialize(payload)
    encoding = _ising_encoding(1.0, -0.5)

    assert restored.block.static_bindings[0].name == "block_encoding"
    assert restored.block.static_bindings[0].type_key == (
        "qamomile.stdlib.lcu_block_encoding"
    )

    executable = qiskit_transpiler.transpile(
        restored,
        bindings={"block_encoding": encoding},
    )

    assert executable.quantum_circuit.num_qubits == 3
    assert serialize(restored) == payload


def test_two_independent_encodings_share_the_reflection_composites(
    qiskit_transpiler: Any,
) -> None:
    """Differently sized encodings coexist without a composite name clash."""
    first = qmc.ising_z_block_encoding({(): 0.6 + 0j, (0,): -0.3 + 0j}, 1)
    second = qmc.ising_z_block_encoding(
        {(): 0.5 + 0j, (0,): -0.2 + 0j, (1,): 0.1 + 0j},
        num_system_qubits=2,
    )

    executable = qiskit_transpiler.transpile(
        _two_encoding_template,
        bindings={"first_encoding": first, "second_encoding": second},
    )

    assert executable.quantum_circuit.num_qubits > 0


def test_amplification_emits_named_reflection_composites(
    qiskit_transpiler: Any,
) -> None:
    """One round emits exactly one good and one zero phase rotation."""
    encoding = _ising_encoding(1.0, -0.5)
    executable = qiskit_transpiler.transpile(
        _single_round_template,
        bindings={"block_encoding": encoding},
    )
    names = [
        instruction.operation.name
        for instruction in executable.quantum_circuit.data
        if instruction.operation.name != "measure"
    ]

    assert names.count("amplitude_amplification_good_phase_rotation") == 1
    assert names.count("amplitude_amplification_zero_phase_rotation") == 1


def test_amplification_uses_one_reusable_auxiliary() -> None:
    """One amplification allocates the registers plus a single auxiliary."""
    encoding = _ising_encoding(1.0, -0.5)
    width = encoding.num_signal_qubits + encoding.num_system_qubits
    estimate = _standard_template(2).estimate_resources(
        inputs={
            "block_encoding": encoding,
            "observable": _zero_projector_observable(encoding.num_signal_qubits),
        }
    )

    assert estimate.width.allocated_qubits == width + 1
    assert estimate.width.peak_qubits == width + 1


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------


def test_schedule_rejects_odd_phase_counts() -> None:
    """Each round consumes exactly two phases."""
    with pytest.raises(ValueError, match="even length"):
        qmc.AmplificationSchedule((math.pi, math.pi, math.pi))


def test_schedule_rejects_a_multidimensional_phase_array() -> None:
    """A phase array must be one-dimensional."""
    with pytest.raises(ValueError, match="one-dimensional"):
        qmc.AmplificationSchedule(np.full((2, 2), np.pi))


def test_schedule_rejects_non_numeric_phases() -> None:
    """Phase entries must be real numbers."""
    with pytest.raises(TypeError):
        qmc.AmplificationSchedule((math.pi, True))
    with pytest.raises(TypeError):
        qmc.AmplificationSchedule("pi")


def test_amplification_accepts_a_numpy_backed_schedule(
    qiskit_transpiler: Any,
) -> None:
    """A schedule built from a NumPy array behaves like a Python sequence."""
    encoding = _ising_encoding(1.0, -0.5)
    initial = _analytic_success_probability(1.0, -0.5)
    angle = math.asin(math.sqrt(initial))
    schedule = qmc.AmplificationSchedule(np.full(2, np.pi))

    @qmc.qkernel
    def template(observable: qmc.Observable) -> qmc.Float:
        """Amplify with a NumPy-backed phase schedule.

        Args:
            observable (qmc.Observable): All-zero signal projector.

        Returns:
            qmc.Float: Success probability after amplification.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            encoding.unitary,
            schedule,
        )
        return qmc.expval(signal, observable)

    observed = _run_closed_over(
        qiskit_transpiler,
        template,
        encoding.num_signal_qubits,
    )

    assert observed == pytest.approx(math.sin(3 * angle) ** 2, abs=1e-8)


def test_amplification_rejects_a_raw_phase_sequence() -> None:
    """The emitter takes a schedule descriptor, not a bare phase list."""
    encoding = _ising_encoding(1.0, -0.5)

    @qmc.qkernel
    def template() -> qmc.Vector[qmc.Bit]:
        """Apply an intentionally bare phase list.

        Returns:
            qmc.Vector[qmc.Bit]: Measured signal register.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            encoding.unitary,
            [math.pi, math.pi],
        )
        return qmc.measure(signal)

    with pytest.raises(TypeError, match="AmplificationSchedule"):
        serialize(template)


def test_amplification_rejects_a_preparation_with_the_wrong_abi() -> None:
    """A reversed preparation signature is rejected before tracing proceeds."""
    encoding = _ising_encoding(1.0, -0.5)

    @qmc.qkernel
    def template() -> qmc.Vector[qmc.Bit]:
        """Apply an intentionally mismatched preparation.

        Returns:
            qmc.Vector[qmc.Bit]: Measured signal register.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            _reversed_preparation,
            qmc.standard_amplification_schedule(1),
        )
        return qmc.measure(signal)

    with pytest.raises(TypeError, match="state_preparation"):
        serialize(template)


def test_standard_schedule_rejects_negative_iterations() -> None:
    """A negative round count is rejected when the schedule is built."""
    with pytest.raises(ValueError, match="non-negative"):
        qmc.standard_amplification_schedule(-1)


def test_standard_schedule_rejects_boolean_iterations() -> None:
    """Booleans are rejected rather than silently coerced."""
    with pytest.raises(TypeError):
        qmc.standard_amplification_schedule(True)


def test_amplification_rejects_a_preparation_that_allocates() -> None:
    """A preparation allocating internal qubits cannot be inverted."""
    encoding = _ising_encoding(1.0, -0.5)

    @qmc.qkernel
    def template() -> qmc.Vector[qmc.Bit]:
        """Apply a preparation that allocates internally.

        Returns:
            qmc.Vector[qmc.Bit]: Measured signal register.
        """
        signal = qmc.qubit_array(encoding.num_signal_qubits, "signal")
        system = qmc.qubit_array(encoding.num_system_qubits, "system")
        signal, _ = qmc.amplitude_amplification(
            signal,
            system,
            _allocating_preparation,
            qmc.standard_amplification_schedule(1),
        )
        return qmc.measure(signal)

    with pytest.raises(NotImplementedError):
        serialize(template)


def test_amplification_preparation_rejects_a_non_encoding() -> None:
    """Only block-encoding descriptors are accepted."""
    with pytest.raises(TypeError, match="LCUBlockEncoding"):
        qmc.amplification_preparation(object())
