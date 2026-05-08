"""Composite gate decomposition helpers."""

from __future__ import annotations


class CompositeDecomposer:
    """Decomposes composite gates into primitive operations."""

    @staticmethod
    def qft_structure(n: int) -> list[tuple[str, tuple[int, ...], float | None]]:
        import math

        gates: list[tuple[str, tuple[int, ...], float | None]] = []
        for i in range(n):
            gates.append(("h", (i,), None))
            for j in range(i + 1, n):
                k = j - i
                angle = math.pi / (2**k)
                gates.append(("cp", (j, i), angle))

        for i in range(n // 2):
            gates.append(("swap", (i, n - 1 - i), None))

        return gates

    @staticmethod
    def iqft_structure(n: int) -> list[tuple[str, tuple[int, ...], float | None]]:
        import math

        gates: list[tuple[str, tuple[int, ...], float | None]] = []
        for i in range(n - 1, -1, -1):
            gates.append(("h", (i,), None))
            for j in range(i - 1, -1, -1):
                k = i - j
                angle = -math.pi / (2**k)
                gates.append(("cp", (j, i), angle))

        for i in range(n // 2):
            gates.append(("swap", (i, n - 1 - i), None))

        return gates
