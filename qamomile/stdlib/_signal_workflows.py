"""Explicit preparation, success readout, and editable amplification workflows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import qamomile as qm
from qamomile._native import compiler as native
from qamomile.types import is_type_assignable

from ._structure import adjoint_unary, lift_slice
from .signals import (
    Region,
    Signal,
    _closed,
    _embedding,
    _validate_process,
    sequence,
    turn,
)


def _input_preparation(signal: Signal, source: qm.QKernel | None) -> qm.QKernel:
    """Prepare an input embedding from a reusable unary logical preparation.

    Args:
        signal (Signal): Selected map with a concrete input embedding.
        source (qm.QKernel | None): Logical preparation applied to zeros, or None.

    Returns:
        qm.QKernel: Ambient unitary preparation with a usable adjoint.

    Raises:
        TypeError: If source is not a compatible closed unary preparation.
        ValueError: If a preparation is supplied for a rank-one input space.
    """
    embedding = _embedding(signal.input_region)
    if source is None:
        return embedding
    payload = len(signal.input_region.payload_indices)
    if payload == 0:
        raise ValueError("a rank-one input requires no logical preparation")
    _validate_process(source, payload)
    if len(source.module().function(source.function_name).parameters) != 1:
        raise TypeError("source must be a closed unary preparation")
    lifted = lift_slice(
        total_width=signal.ambient_width,
        start=signal.ambient_width - payload,
        width=payload,
        transform=source,
    )
    return sequence(lifted, embedding, width=signal.ambient_width)


def sample(
    signal: Signal,
    *,
    source: qm.QKernel | None = None,
    readout: qm.QKernel | None = None,
) -> qm.QKernel:
    """Prepare fresh state and return success plus logical payload measurement.

    Every ambient lane is measured before result projection. No retry or
    postselection renormalization occurs; failures remain in the distribution.

    Args:
        signal (Signal): Closed selected-map recipe.
        source (qm.QKernel | None): Unary preparation on logical input zero bits;
            None uses the logical zero state.
        readout (qm.QKernel | None): Optional unary output-payload basis change.

    Returns:
        qm.QKernel: Nullary sampler returning ``(Bool, Bits[payload])``. Rank-one
        outputs return just Bool, never a zero-width bit register.

    Raises:
        TypeError: If a process, preparation, or readout ABI is incompatible.
        ValueError: If source/readout is supplied for a rank-one space.
    """
    _closed(signal)
    width = signal.ambient_width
    payload = len(signal.output_region.payload_indices)
    fixed = width - payload
    steps = [
        _input_preparation(signal, source),
        signal.process,
        adjoint_unary(width=width, transform=_embedding(signal.output_region)),
    ]
    if readout is not None:
        if payload == 0:
            raise ValueError("a rank-one output requires no logical readout")
        _validate_process(readout, payload)
        steps.append(
            lift_slice(total_width=width, start=fixed, width=payload, transform=readout)
        )
    process = sequence(*steps, width=width)
    if payload == 0:

        @qm.qkernel
        def point() -> qm.Bool:  # type: ignore[valid-type]
            """Measure the complete rank-one success condition.

            Returns:
                qm.Bool: Whether every success lane was zero.
            """
            bits = qm.measure(process(qm.qubits(width)))
            success = True
            for i in range(width):
                success = success & (not bits[i])
            return success

        return point

    if fixed == 0:

        @qm.qkernel
        def full_space() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bits[payload]]]:  # type: ignore[valid-type]
            """Measure a full-space payload with an always-true success flag.

            Returns:
                tuple[qm.Bool, qm.Bits[payload]]: True and measured payload.
            """
            return True, qm.measure(process(qm.qubits(width)))

        return full_space

    @qm.qkernel
    def sampled() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bits[payload]]]:  # type: ignore[valid-type]
        """Measure success and payload coordinates in source order.

        Returns:
            tuple[qm.Bool, qm.Bits[payload]]: Success flag and payload bits.
        """
        flags, data = qm.split_at(process(qm.qubits(width)), fixed)
        bits = qm.measure(flags)
        payload_bits = qm.measure(data)
        success = True
        for i in range(fixed):
            success = success & (not bits[i])
        return success, payload_bits

    return sampled


@dataclass(frozen=True, slots=True)
class SuccessEstimate:
    """Observed success and conditional parity with simultaneous intervals.

    Statements assume independent identically distributed shots. They are not
    operator certificates or device-noise bounds. With zero successes, the
    conditional value is None and its interval is uninformative [-1, 1].

    Args:
        shots (int): All observed shots, including failures.
        successes (int): Observed successful shots.
        probability (float): Observed success fraction.
        probability_interval (tuple[float, float]): Success confidence interval.
        conditional (float | None): Success-conditioned Z parity, or None.
        conditional_interval (tuple[float, float]): Conditional parity interval.
        confidence (float): Joint confidence for both intervals.
    """

    shots: int
    successes: int
    probability: float
    probability_interval: tuple[float, float]
    conditional: float | None
    conditional_interval: tuple[float, float]
    confidence: float


def estimate_success(
    result: qm.SampleResult, *, z_indices: Sequence[int] = (), confidence: float = 0.95
) -> SuccessEstimate:
    """Aggregate success-first samples without hiding the success denominator.

    Args:
        result (qm.SampleResult): Result of ``sample`` or an equivalent typed
            success-first sampler; raw unstructured bits are rejected.
        z_indices (Sequence[int]): Distinct payload coordinates for conditional
            Z-product estimation. Empty means the identity observable.
        confidence (float): Joint confidence in (0, 1), default 0.95.

    Returns:
        SuccessEstimate: Native-derived estimates and simultaneous Hoeffding
        intervals, conditional on independent identically distributed shots.

    Raises:
        TypeError: If result ABI or host argument types are invalid.
        ValueError: If samples, indices, confidence, or work budgets fail.
    """
    if not isinstance(result, qm.SampleResult) or result.result_abi is None:
        raise TypeError("expected a typed success-first SampleResult")
    expected = (
        qm.Bool if result.width == 1 else tuple[qm.Bool, qm.Bits[result.width - 1]]  # type: ignore[valid-type]
    )
    if not is_type_assignable(qm.Shot[expected], result.result_abi.type):
        raise TypeError("result must be Bool or (Bool, Bits[payload])")
    if len(z_indices) > result.width or any(type(i) is not int for i in z_indices):
        raise TypeError("z_indices must be distinct integer payload coordinates")
    rows = [(list(bits), count) for bits, count in result.counts.items()]
    value = native.signal_estimate(rows, list(z_indices), confidence)
    if not isinstance(value, native.SignalEstimate):
        raise TypeError("Q7734: malformed native success estimate")
    return SuccessEstimate(
        value.shots,
        value.successes,
        value.probability,
        value.probability_interval,
        value.conditional,
        value.conditional_interval,
        confidence,
    )


def amplify(
    signal: Signal,
    *,
    phases: Sequence[tuple[float, float]],
    source: qm.QKernel | None = None,
) -> Signal:
    """Apply explicit amplitude-amplification rounds to a reusable preparation.

    This does not amplify an unknown state supplied once. The target is newly
    defined by this preparation; old operator and polynomial claims are not
    inherited. Overshooting remains the user's choice of phases.

    Args:
        signal (Signal): Closed source process and its success condition.
        phases (Sequence[tuple[float, float]]): At most 256 input/output phase
            pairs; an empty schedule performs only the original preparation.
        source (qm.QKernel | None): Unary logical preparation on zero bits.

    Returns:
        Signal: Rank-one input selection of the amplified preparation.

    Raises:
        TypeError: If the process, preparation, or phase-row types are invalid.
        ValueError: If phase values or construction budgets are invalid.
    """
    _closed(signal)
    if not isinstance(phases, Sequence) or len(phases) > native.MAX_SIGNAL_BRANCHES:
        raise ValueError("Q7742: at most 256 phase rounds are supported")
    schedule = tuple(native.signal_schedule(list(phases)))
    width = signal.ambient_width
    prepared = sequence(_input_preparation(signal, source), signal.process, width=width)
    input_region = Region("0" * width)
    contract = signal._contract.prepared()
    if not schedule:
        return Signal._from_contract(prepared, contract, "fresh prepared-state target")
    initial = turn(input_region)
    successful = turn(signal.output_region)

    @qm.qkernel
    def amplified(state: qm.Q[qm.Bits[width]]) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Run the explicitly chosen coherent amplification rounds.

        Args:
            state (qm.Q[qm.Bits[width]]): Fresh zero ambient register.

        Returns:
            qm.Q[qm.Bits[width]]: Complete output, including failure amplitudes.
        """
        state = prepared(state)
        for pair in schedule:
            state = successful(state, pair[1])
            state = qm.adjoint(prepared, state)
            state = initial(state, pair[0])
            state = prepared(state)
        return state

    return Signal._from_contract(
        amplified,
        contract,
        "explicit amplified preparation; prior matrix claims discarded",
    )


def inspect_schedule(
    phases: Sequence[tuple[float, float]], amplitude: float
) -> complex:
    """Inspect a canonical two-dimensional response at one success amplitude.

    Args:
        phases (Sequence[tuple[float, float]]): Input/output phase pairs as in
            ``amplify``; at most 256 rounds.
        amplitude (float): Real initial success amplitude in [-1, 1].

    Returns:
        complex: Numerical selected amplitude retaining scalar phase; pointwise
        evidence, not a uniform polynomial or success-probability certificate.

    Raises:
        ValueError: If phases, amplitude, or inspection budgets are invalid.
    """
    if not isinstance(phases, Sequence) or len(phases) > native.MAX_SIGNAL_BRANCHES:
        raise ValueError("Q7742: at most 256 phase rounds are supported")
    real, imaginary = native.signal_schedule_response(list(phases), amplitude)
    return complex(real, imaginary)
