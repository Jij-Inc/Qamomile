"""Clean-ancilla planning for the shared multi-controlled decomposition.

Backends without a native multi-controlled gate primitive lower an
irreducible ``n``-controlled single-qubit gate through the standard
Toffoli-cascade construction (arXiv:2307.07478, Appendix A.3): the
logical AND of all ``n`` controls is accumulated onto ``n - 1`` clean
ancilla qubits with a cascade of Toffoli gates, the gate is applied
once under a single control (the last ancilla), and the cascade is
uncomputed in reverse. Every ancilla therefore returns to ``|0>`` and
the same pool can be reused by every multi-controlled gate in the
segment.

Backend circuits are created with a fixed qubit count before emission
starts, so the pool must be sized up front. Its size is measured by a
count-only dry-run of the real emission (see
``StandardEmitPass._count_multi_control_ancilla_demand`` and
``counting_emitter.CountingEmitter``): the same walk the real emission
runs is executed against a no-op emitter and a counting pool that records
peak concurrent usage, so the size can never drift from what the emitter
actually does. This module provides only :class:`MultiControlAncillaPool`
— the reserved (or counting) block of physical qubit indices appended
after the segment's data qubits.

The pool has stack discipline (a moving offset) so leaf cascades and the
AND ladders that batched controlled-block emission holds via
:meth:`MultiControlAncillaPool.try_hold` draw from disjoint ranges. Under
counting the pool is unbounded and only records the peak, which is an
upper bound of real usage because the dry run biases capability answers
toward the cascade path; a shortfall in a real (bounded) pool would mean
the count under-measured and is reported as a compiler bug at emit time by
``StandardEmitPass._emit_irreducible_multi_controlled_gate``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator


class MultiControlAncillaPool:
    """A reserved block of clean ancilla qubits for multi-control lowering.

    The pool occupies a contiguous range of physical qubit indices
    appended after a quantum segment's data qubits. A single moving
    ``_offset`` gives it stack discipline so nested users never collide:

    * A leaf Toffoli cascade uncomputes every ancilla back to ``|0>``
      before it returns and holds nothing across gates, so it simply
      ``take``s from the current offset (atomic acquire-and-release).
    * Batched controlled-block emission ``try_hold``s the block's AND
      ladder for the duration of the body walk, advancing the offset so
      the leaf cascades emitted inside the body draw from the qubits
      *after* the held ladder. The offset is rewound when the ``with``
      block exits, so sibling batches reuse the same range.
    """

    def __init__(self, first_index: int, count: int, *, counting: bool = False) -> None:
        """Initialize the pool.

        Args:
            first_index (int): Physical index of the first reserved
                ancilla qubit.
            count (int): Number of reserved ancilla qubits. Must be
                non-negative. Ignored in counting mode.
            counting (bool): When True the pool is unbounded and only
                records the peak concurrent usage (``take`` / ``try_hold``
                never fail), so a dry-run emission can measure how many
                ancillas a real emission would need. Defaults to False.

        Raises:
            ValueError: If ``count`` is negative.
        """
        if count < 0:
            raise ValueError(f"Ancilla pool count must be non-negative, got {count}.")
        self._first_index = first_index
        self._count = count
        self._offset = 0
        self._counting = counting
        self._peak = 0

    @property
    def count(self) -> int:
        """Return the number of reserved ancilla qubits.

        Returns:
            int: Pool size.
        """
        return self._count

    @property
    def peak(self) -> int:
        """Return the peak concurrent usage recorded in counting mode.

        Returns:
            int: The largest ``offset + count`` any ``take`` / ``try_hold``
                reached; zero for a non-counting pool.
        """
        return self._peak

    def take(self, count: int) -> list[int] | None:
        """Return ``count`` clean ancilla indices from the current offset.

        Args:
            count (int): Number of ancilla qubits requested.

        Returns:
            list[int] | None: The next ``count`` reserved physical
                indices starting at the current offset, or None when the
                unheld remainder of the pool is smaller than the request
                (an estimation bug surfaced by the caller). In counting
                mode the request always succeeds and the peak usage is
                recorded instead.

        Raises:
            ValueError: If ``count`` is negative — a caller bug rather than
                a shortfall, which a plain ``> count`` check would silently
                turn into an empty list (and, via ``try_hold``, a negative
                offset).
        """
        if count < 0:
            raise ValueError(f"Ancilla request count must be non-negative, got {count}.")
        if self._counting:
            self._peak = max(self._peak, self._offset + count)
        elif self._offset + count > self._count:
            return None
        start = self._first_index + self._offset
        return list(range(start, start + count))

    @contextlib.contextmanager
    def try_hold(self, count: int) -> Iterator[list[int] | None]:
        """Reserve ``count`` ancillas for the duration of a ``with`` block.

        Used by batched controlled-block emission to keep an AND ladder
        live while the block body is walked under a single control. The
        offset advances so that ``take``/``try_hold`` calls made inside
        the body draw from the qubits after the held range, and it is
        rewound on exit (including on exception) so sibling batches reuse
        the same range. When the unheld remainder cannot satisfy the
        request the caller receives ``None`` and must fall back to
        per-gate emission rather than treating it as an error — the
        demand estimate legitimately reserves fewer ancillas than a
        naive batch would want (e.g. a statically empty loop body).

        Args:
            count (int): Number of contiguous ancilla qubits to hold.

        Yields:
            list[int] | None: The held physical indices, or None when the
                unheld remainder is too small to satisfy the request.
        """
        indices = self.take(count)
        if indices is None:
            yield None
            return
        self._offset += count
        try:
            yield indices
        finally:
            self._offset -= count
