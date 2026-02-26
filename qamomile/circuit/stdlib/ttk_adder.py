"""TTK Inplace Adder from arXiv:0910.2530v1, Section 2.2.

Adds two n-bit binary numbers a[n-1]...a[0] and b[n-1]...b[0],
storing the (n+1)-bit sum in-place: B_i <- s_i, z <- z XOR s_n.

Available:
    Classes:
        - TTKInplaceAdder: CompositeGate implementing the TTK adder

    Functions:
        - ttk_adder: Apply TTK inplace adder to qubit registers

Qubit layout (2n+1 qubits, grouped by register):
    (a_0, a_1, ..., a_{n-1}, b_0, b_1, ..., b_{n-1}, z)

Resources (n >= 2):
    - Gates: 7n - 6  (5n-5 CNOTs + 2n-1 Toffolis)
    - Depth: 5n - 3
    - Ancilla: 0
"""

from __future__ import annotations

import qamomile.circuit as qmc
from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.frontend.handle import Qubit, Vector
from qamomile.circuit.frontend.handle.utils import _get_size
from qamomile.circuit.ir.operation.composite_gate import (
    CompositeGateType,
    ResourceMetadata,
)



class TTKInplaceAdder(CompositeGate):
    """TTK inplace adder (arXiv:0910.2530v1, Section 2.2).

    Adds two n-bit binary numbers using only CNOT and Toffoli gates,
    requiring no ancilla qubits. The exact gate count is 7n - 6.

    Qubit ordering (2n+1 qubits total, grouped by register):
        (a_0, a_1, ..., a_{n-1}, b_0, b_1, ..., b_{n-1}, z)

    After execution:
        - A_i is restored to a_i (unchanged)
        - B_i contains s_i (sum bit i), for 0 <= i <= n-1
        - z contains z XOR s_n (carry-out / MSB of sum)

    Example::

        adder = TTKInplaceAdder(n=3)
        result = adder(a0, a1, a2, b0, b1, b2, z)
    """

    gate_type = CompositeGateType.CUSTOM
    custom_name = "ttk_adder"

    _strategies: dict = {}  # type: ignore[type-arg]
    _default_strategy = "standard"

    def __init__(self, n: int):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self._n = n

    @property
    def num_target_qubits(self) -> int:
        return 2 * self._n + 1

    def _decompose(
        self,
        qubits: tuple[Qubit, ...],
    ) -> tuple[Qubit, ...]:
        """Decompose TTK adder into CNOT and Toffoli gates.

        Implements the 6-step algorithm from Section 2.2 of the paper.
        """
        n = self._n
        q = list(qubits)

        # Index helpers for grouped layout: (a_0..a_{n-1}, b_0..b_{n-1}, z)
        def A(i: int) -> int:
            return i

        def B(i: int) -> int:
            return n + i

        Z = 2 * n  # carry-out qubit

        # --- Special case n=1: only Step 3 (i=0) and Step 6 (i=0) ---
        if n == 1:
            q[B(0)], q[A(0)], q[Z] = qmc.ccx(q[B(0)], q[A(0)], q[Z])
            q[A(0)], q[B(0)] = qmc.cx(q[A(0)], q[B(0)])
            return tuple(q)

        # --- Full algorithm for n >= 2 ---

        # Step 1: For i = 1, ..., n-1: CNOT(ctrl=A_i, tgt=B_i)
        for i in range(1, n):
            q[A(i)], q[B(i)] = qmc.cx(q[A(i)], q[B(i)])

        # Step 2: For i = n-1, ..., 1: CNOT(ctrl=A_i, tgt=A_{i+1})
        #   When i = n-1, A_{i+1} = A_n = z qubit
        for i in range(n - 1, 0, -1):
            tgt = Z if i == n - 1 else A(i + 1)
            q[A(i)], q[tgt] = qmc.cx(q[A(i)], q[tgt])

        # Step 3: For i = 0, ..., n-1: Toffoli(ctrl=B_i,A_i, tgt=A_{i+1})
        for i in range(n):
            tgt = Z if i == n - 1 else A(i + 1)
            q[B(i)], q[A(i)], q[tgt] = qmc.ccx(q[B(i)], q[A(i)], q[tgt])

        # Step 4: For i = n-1, ..., 1:
        #   CNOT(ctrl=A_i, tgt=B_i), then Toffoli(ctrl=B_{i-1},A_{i-1}, tgt=A_i)
        for i in range(n - 1, 0, -1):
            q[A(i)], q[B(i)] = qmc.cx(q[A(i)], q[B(i)])
            q[B(i - 1)], q[A(i - 1)], q[A(i)] = qmc.ccx(
                q[B(i - 1)], q[A(i - 1)], q[A(i)]
            )

        # Step 5: For i = 1, ..., n-2: CNOT(ctrl=A_i, tgt=A_{i+1})
        for i in range(1, n - 1):
            q[A(i)], q[A(i + 1)] = qmc.cx(q[A(i)], q[A(i + 1)])

        # Step 6: For i = 0, ..., n-1: CNOT(ctrl=A_i, tgt=B_i)
        for i in range(n):
            q[A(i)], q[B(i)] = qmc.cx(q[A(i)], q[B(i)])

        return tuple(q)

    def _resources(self) -> ResourceMetadata:
        n = self._n
        if n == 1:
            return ResourceMetadata(
                t_gate_count=7,
                custom_metadata={
                    "num_cnot_gates": 1,
                    "num_toffoli_gates": 1,
                    "total_gates": 2,
                    "depth": 2,
                    "operand_bits": 1,
                },
            )
        else:
            toffoli_count = 2 * n - 1
            return ResourceMetadata(
                t_gate_count=7 * toffoli_count,
                custom_metadata={
                    "num_cnot_gates": 5 * n - 5,
                    "num_toffoli_gates": toffoli_count,
                    "total_gates": 7 * n - 6,
                    "depth": 5 * n - 3,
                    "operand_bits": n,
                },
            )


def ttk_adder(
    b: Vector[Qubit],
    a: Vector[Qubit],
    z: Qubit,
) -> tuple[Vector[Qubit], Vector[Qubit], Qubit]:
    """Apply TTK inplace adder to two qubit registers and a carry-out qubit.

    Computes a + b in-place: after execution, b contains the n low-order
    sum bits and z contains z XOR carry-out. Register a is restored.

    Args:
        b: Vector of n qubits holding b[n-1]...b[0] (becomes sum bits).
        a: Vector of n qubits holding a[n-1]...a[0] (restored at end).
        z: Single qubit for carry-out (becomes z XOR s_n).

    Returns:
        Tuple of (b, a, z) after addition.
    """
    n_b = _get_size(b)
    n_a = _get_size(a)
    if n_b != n_a:
        raise ValueError(
            f"Registers a and b must have the same size, got a={n_a}, b={n_b}"
        )
    n = n_b
    adder = TTKInplaceAdder(n)

    # Build grouped qubit list: (a_0..a_{n-1}, b_0..b_{n-1}, z)
    qubit_list: list[Qubit] = []
    for i in range(n):
        qubit_list.append(a[i])
    for i in range(n):
        qubit_list.append(b[i])
    qubit_list.append(z)

    result = adder(*qubit_list)

    # Write results back
    for i in range(n):
        a[i] = result[i]
        b[i] = result[n + i]
    z_out = result[2 * n]

    return b, a, z_out
