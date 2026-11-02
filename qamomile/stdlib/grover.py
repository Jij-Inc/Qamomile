# ruff: noqa: F821 -- ``width`` is closed into a concrete dependent shape.

"""Functional Grover search composed from existing QCore transformations."""

from __future__ import annotations

import qamomile as qm
from qamomile._native import compiler as native

from ._primitives import flip


def grover_iteration_count(num_qubits: int, num_marked: int = 1) -> int:
    """Return ``floor((pi / 4) * sqrt(2**num_qubits / num_marked))``.

    Native planning uses bounded arbitrary-precision integer arithmetic, so
    large search spaces return ordinary Python integers without overflowing a
    binary64 intermediate.

    Args:
        num_qubits: Positive number of search-register qubits.
        num_marked: Positive number of marked basis states. Defaults to one.

    Returns:
        The conventional Grover iteration count.

    Raises:
        TypeError: If either argument is not an integer or is a Boolean.
        ValueError: If either argument is nonpositive or the marked-state count
            exceeds the search space.
    """

    for name, value in (("num_qubits", num_qubits), ("num_marked", num_marked)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    derive = getattr(native, "grover_iteration_count", None)
    if derive is None:
        raise TypeError("Q7594: native Grover iteration planning is unavailable")
    result = derive(str(num_qubits), str(num_marked))
    if not isinstance(result, str) or not result.isdecimal():
        raise TypeError("Q7595: malformed native Grover iteration plan")
    return int(result)


def grover_search(oracle: qm.QKernel, *, width: int) -> qm.QKernel:
    """Construct Grover amplitude amplification for a static phase oracle.

    The host boundary fixes the register split used by the diffusion operator.
    Qamomile dimensions currently admit names and literals, but not the
    derived ``width - 1`` shape required by ``qm.split_at``. The returned source
    transform keeps the iteration count as an ordinary ``Const[Int]`` input.

    Args:
        oracle (qm.QKernel): Exact unary phase oracle over
            ``Q[Bits[width]]``.
        width (int): Positive search-register width.

    Returns:
        qm.QKernel: A source-backed transform
        ``(state, iterations) -> state``.

    Raises:
        TypeError: If an argument has the wrong host type.
        ValueError: If ``width`` is not positive.
    """

    if not isinstance(oracle, qm.QKernel):
        raise TypeError("oracle must be a Qamomile QKernel")
    if isinstance(width, bool) or not isinstance(width, int):
        raise TypeError("width must be an integer")
    if width <= 0:
        raise ValueError("width must be positive")

    if width == 1:

        @qm.qkernel
        def single_search(
            state: qm.Q[qm.Bits[1]],  # type: ignore[valid-type]
            iterations: qm.Const[qm.Int],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[1]]:  # type: ignore[valid-type]
            """Apply Grover search to a one-component state.

            Args:
                state (qm.Q[qm.Bits[1]]): Owned search state.
                iterations (qm.Const[qm.Int]): Compile-time iteration count.

            Returns:
                qm.Q[qm.Bits[1]]: State after the configured search iterations.
            """

            state = qm.h(state)
            for _ in range(iterations):
                state = oracle(state)
                state = qm.h(state)
                state = qm.x(state)
                state = qm.z(state)
                state = qm.x(state)
                state = qm.h(state)
            return state

        return single_search

    control_width = width - 1

    @qm.qkernel
    def search(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        iterations: qm.Const[qm.Int],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply Grover search to the configured register.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned search state.
            iterations (qm.Const[qm.Int]): Compile-time iteration count.

        Returns:
            qm.Q[qm.Bits[width]]: State after the configured search iterations.
        """

        for index in range(width):
            state = qm.h(state, target=index)
        for _ in range(iterations):
            state = oracle(state)
            for index in range(width):
                state = qm.h(state, target=index)
                state = qm.x(state, target=index)
            controls, target = qm.split_at(state, control_width)
            target = qm.h(target)
            controls, target = qm.control(flip, controls, target)
            target = qm.h(target)
            state = qm.concat(controls, target)
            for index in range(width):
                state = qm.x(state, target=index)
                state = qm.h(state, target=index)
        return state

    return search


__all__ = ["grover_iteration_count", "grover_search"]
