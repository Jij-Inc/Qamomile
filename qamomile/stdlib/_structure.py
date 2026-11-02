"""Internal structural composition shared by standard-library factories."""

from __future__ import annotations

from collections.abc import Sequence

import qamomile as qm


@qm.qkernel
def identity(
    state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]  # noqa: F821
) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]  # noqa: F821
    """Return an owned state unchanged.

    Args:
        state (qm.Q[qm.Bits[width]]): State whose ownership is transferred through
            the identity transform.

    Returns:
        qm.Q[qm.Bits[width]]: The same owned state.

    Raises:
        ValueError: If the dependent state width is invalid.
    """

    return state


def compose_unary(
    *,
    width: int,
    steps: Sequence[qm.QKernel],
) -> qm.QKernel:
    """Compose fixed-width unary transforms without a runtime registry."""

    frozen = tuple(steps)
    if not frozen:
        return identity

    current = frozen[0]
    for step in frozen[1:]:
        previous = current

        @qm.qkernel
        def composed(
            state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
            """Apply the accumulated transforms in sequence.

            Args:
                state (qm.Q[qm.Bits[width]]): Owned state to transform.

            Returns:
                qm.Q[qm.Bits[width]]: State after both configured transforms.
            """

            return step(previous(state))

        current = composed
    return current


def adjoint_unary(*, width: int, transform: qm.QKernel) -> qm.QKernel:
    """Construct the exact inverse of one configured unary transform.

    Args:
        width (int): Positive concrete state width.
        transform (qm.QKernel): Closed unary transform over that width.

    Returns:
        qm.QKernel: Compiler-derived inverse with the same signature.

    Raises:
        ValueError: If ``width`` is not positive.
    """

    if width <= 0:
        raise ValueError("adjoint transform width must be positive")

    @qm.qkernel
    def inverse(
        state: qm.Q[qm.Bits[width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[width]]:  # type: ignore[valid-type]
        """Apply the compiler-derived inverse of the closed transform.

        Args:
            state (qm.Q[qm.Bits[width]]): Owned state to transform.

        Returns:
            qm.Q[qm.Bits[width]]: Inverse-transformed state.
        """

        return qm.adjoint(transform, state)

    return inverse


def lift_slice(
    *,
    total_width: int,
    start: int,
    width: int,
    transform: qm.QKernel,
) -> qm.QKernel:
    """Lift one unary transform onto a fixed contiguous state slice."""

    if start < 0 or width <= 0 or start + width > total_width:
        raise ValueError("slice must be nonempty and contained in the state")
    if start == 0 and width == total_width:
        return transform
    if start == 0:

        @qm.qkernel
        def leading(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Apply the configured transform to the leading slice.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Owned complete state.

            Returns:
                qm.Q[qm.Bits[total_width]]: State with its leading slice transformed.
            """

            selected, suffix = qm.split_at(state, width)
            return qm.concat(transform(selected), suffix)

        return leading
    if start + width == total_width:

        @qm.qkernel
        def trailing(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Apply the configured transform to the trailing slice.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Owned complete state.

            Returns:
                qm.Q[qm.Bits[total_width]]: State with its trailing slice transformed.
            """

            prefix, selected = qm.split_at(state, start)
            return qm.concat(prefix, transform(selected))

        return trailing

    @qm.qkernel
    def interior(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the configured transform to an interior slice.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Owned complete state.

        Returns:
            qm.Q[qm.Bits[total_width]]: State with its interior slice transformed.
        """

        prefix, remainder = qm.split_at(state, start)
        selected, suffix = qm.split_at(remainder, width)
        return qm.concat(prefix, transform(selected), suffix)

    return interior


def lift_indices(
    *,
    total_width: int,
    indices: tuple[int, ...],
    transform: qm.QKernel,
) -> qm.QKernel:
    """Lift a closed transform through zero-cost owned tensor regrouping.

    Args:
        total_width (int): Positive ambient register width.
        indices (tuple[int, ...]): Unique in-range positions in helper order.
        transform (qm.QKernel): Unary transform over the selected register.

    Returns:
        qm.QKernel: Unary ambient transform that restores the original order.

    Raises:
        ValueError: If selection is empty, duplicated, or outside the register.
    """

    if not indices:
        raise ValueError("lifted indices must be nonempty")
    if len(set(indices)) != len(indices):
        raise ValueError("lifted indices must be unique")
    if any(index < 0 or index >= total_width for index in indices):
        raise ValueError("lifted index is outside the state")
    if indices == tuple(range(total_width)):
        return transform
    remaining = tuple(index for index in range(total_width) if index not in indices)
    permutation = _tensor_permutation(
        width=total_width,
        order=(*indices, *remaining),
    )
    order = (*indices, *remaining)
    inverse_order = [0] * total_width
    for position, original in enumerate(order):
        inverse_order[original] = position
    inverse_permutation = _tensor_permutation(
        width=total_width, order=tuple(inverse_order)
    )
    selected_width = len(indices)
    if selected_width == total_width:

        @qm.qkernel
        def complete(
            state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
            """Apply a reordered transform spanning the complete state.

            Args:
                state (qm.Q[qm.Bits[total_width]]): Owned complete register.

            Returns:
                qm.Q[qm.Bits[total_width]]: Transformed register in original order.
            """

            state = permutation(state)
            state = transform(state)
            return inverse_permutation(state)

        return complete

    @qm.qkernel
    def lifted(
        state: qm.Q[qm.Bits[total_width]],  # type: ignore[valid-type]
    ) -> qm.Q[qm.Bits[total_width]]:  # type: ignore[valid-type]
        """Apply the configured transform to selected components.

        Args:
            state (qm.Q[qm.Bits[total_width]]): Owned complete register.

        Returns:
            qm.Q[qm.Bits[total_width]]: Transformed register in original order.
        """

        state = permutation(state)
        selected, rest = qm.split_at(state, selected_width)
        selected = transform(selected)
        state = qm.concat(selected, rest)
        return inverse_permutation(state)

    return lifted


def _tensor_permutation(*, width: int, order: tuple[int, ...]) -> qm.QKernel:
    """Reorder tensor factors through owned splits and concatenations.

    Args:
        width (int): Positive ambient register width.
        order (tuple[int, ...]): Complete permutation of the register positions.

    Returns:
        qm.QKernel: Zero-gate tensor-coordinate permutation.

    Raises:
        ValueError: If the order is not a complete permutation.
    """

    if len(order) != width or set(order) != set(range(width)):
        raise ValueError("order must be a permutation of the complete state")
    current = list(range(width))
    positions = list(range(width))
    steps: list[qm.QKernel] = []
    for destination, original in enumerate(order):
        source = positions[original]
        if source == destination:
            continue
        steps.append(_tensor_exchange(width=width, left=destination, right=source))
        displaced = current[destination]
        current[destination], current[source] = original, displaced
        positions[original], positions[displaced] = destination, source
    return compose_unary(width=width, steps=steps)


def _tensor_exchange(*, width: int, left: int, right: int) -> qm.QKernel:
    """Exchange two static tensor coordinates while transferring ownership.

    Args:
        width (int): Ambient register width.
        left (int): Lower coordinate of the exchange.
        right (int): Higher coordinate of the exchange.

    Returns:
        qm.QKernel: Structural permutation with no physical SWAP gate.
    """

    selected_width = right - left + 1
    if selected_width == 2:

        @qm.qkernel
        def adjacent(state: qm.Q[qm.Bits[2]]) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
            """Exchange adjacent owned tensor factors.

            Args:
                state (qm.Q[qm.Bits[2]]): Owned pair of qubits.

            Returns:
                qm.Q[qm.Bits[2]]: Pair in reversed coordinate order.
            """

            first, last = qm.split_at(state, 1)
            return qm.concat(last, first)

        transform = adjacent
    else:
        middle_width = selected_width - 2

        @qm.qkernel
        def endpoints(
            state: qm.Q[qm.Bits[selected_width]],  # type: ignore[valid-type]
        ) -> qm.Q[qm.Bits[selected_width]]:  # type: ignore[valid-type]
            """Exchange the endpoints of an owned contiguous block.

            Args:
                state (qm.Q[qm.Bits[selected_width]]): Owned block to regroup.

            Returns:
                qm.Q[qm.Bits[selected_width]]: Block with endpoints exchanged.
            """

            first, rest = qm.split_at(state, 1)
            middle, last = qm.split_at(rest, middle_width)
            return qm.concat(last, middle, first)

        transform = endpoints
    return lift_slice(
        total_width=width, start=left, width=selected_width, transform=transform
    )


__all__: list[str] = []
