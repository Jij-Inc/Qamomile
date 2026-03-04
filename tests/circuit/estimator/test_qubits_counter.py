import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator import qubits_counter
from qamomile.circuit.frontend.composite_gate import CompositeGate, _StubCompositeGate
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata


class TestQInitOp:
    """Verify that resource estimation returns correct sympy expressions for qubit counts via UInt arithmetic."""

    def test_naive_one_qubit(self):
        """Single qubit() call → qubits == 1."""

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_naive_two_qubits(self):
        """Two qubit() calls → qubits == 2."""

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            return q1, q2

        resource = qubits_counter(circuit.block)
        assert resource == 2

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_naive_int_qubits(self, n):
        """qubit_array(int) → qubits == n."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qubits = qmc.qubit_array(n, name="qs")
            return qubits

        resource = qubits_counter(circuit.block)
        assert resource == n

    def test_naive_symboic_qubits_n(self):
        """qubit_array(UInt) → symbolic qubit count."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == sp.Symbol("n", integer=True, positive=True)

    def test_naive_symbolic_qubits_symbol(self):
        """Parameter name is reflected as the symbol name."""

        @qmc.qkernel
        def circuit(symbol: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(symbol, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == sp.Symbol("symbol", integer=True, positive=True)

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_naive_symbolic_and_int_qubits(self, m):
        """Mixed symbolic and fixed qubits → correct total expression."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, symbol: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit], qmc.Qubit]:
            qs_symbol = qmc.qubit_array(n, name="qs_symbol")
            qs_fixed = qmc.qubit_array(m, name="qs_fixed")
            q = qmc.qubit(name="q")
            return qs_symbol, qs_fixed, q

        resource = qubits_counter(circuit.block)
        expected = (1 + m) + sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_add_int_qubits(self, a):
        """__add__: n + a (symbol + int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n + a
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = a + sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    def test_naive_symbolic_add_symbol_qubits(self):
        """__add__: n + m (symbol + symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n + m
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) + sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_radd_int_qubits(self, a):
        """__radd__: a + n (int + symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = a + n
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = a + sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_sub_int_qubits(self, a):
        """__sub__: n - a (symbol - int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n - a
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) - a
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_rsub_int_qubits(self, a):
        """__rsub__: a - n (int - symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = a - n
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = a - sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    def test_naive_symbolic_sub_symbol_qubits(self):
        """__sub__: n - m (symbol - symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n - m
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) - sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_mul_int_qubits(self, a):
        """__mul__: n * a (symbol * int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n * a
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) * a
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_rmul_int_qubits(self, a):
        """__rmul__: a * n (int * symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = a * n
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = a * sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    def test_naive_symbolic_mul_symbol_qubits(self):
        """__mul__: n * m (symbol * symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n * m
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) * sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_floordiv_int_qubits(self, a):
        """__floordiv__: n // a (symbol // int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n // a
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.floor(sp.Symbol("n", integer=True, positive=True) / a)
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_rfloordiv_int_qubits(self, a):
        """__rfloordiv__: a // n (int // symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = a // n
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.floor(a / sp.Symbol("n", integer=True, positive=True))
        assert resource == expected

    def test_naive_symbolic_floordiv_symbol_qubits(self):
        """__floordiv__: n // m (symbol // symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n // m
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.floor(
            sp.Symbol("n", integer=True, positive=True)
            / sp.Symbol("m", integer=True, positive=True)  # type: ignore
        )
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_pow_int_qubits(self, a):
        """__pow__: n ** a (symbol ** int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n**a
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) ** a
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_naive_symbolic_rpow_int_qubits(self, a):
        """__rpow__: a ** n (int ** symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = a**n
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = a ** sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    def test_naive_symbolic_pow_symbol_qubits(self):
        """__pow__: n ** m (symbol ** symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            num = n**m
            qs = qmc.qubit_array(num, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) ** sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_argument_int_qubits(self, n):
        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            qs = qmc.qubit_array(n, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == n

    def test_argument_symbolical_qubits(self):
        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            qs = qmc.qubit_array(n, name="qs")
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == sp.Symbol("n", integer=True, positive=True)


class TestForOp:
    """Verify that resource estimation correctly handles for loops with UInt iteration counts."""

    def test_int_qubits_outside_loop(self):
        """Qubit allocation outside loop → qubits == allocated qubits, not multiplied by iterations."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            q = qmc.qubit(name="q")
            for _ in qmc.range(2):
                q = qmc.h(q)
            return qmc.measure(q)

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_symbolic_qubits_outside_loop(self):
        """Qubit allocation outside loop → qubits == allocated qubits, not multiplied by iterations."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            for i in qmc.range(2):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        resource = qubits_counter(circuit.block)
        assert resource == sp.Symbol("n", integer=True, positive=True)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_int_for_loop_qubits(self, n):
        """For loop with int iteration count → qubits == loop body qubits * iterations."""

        @qmc.qkernel
        def circuit() -> qmc.Bit:
            for i in qmc.range(n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        assert resource == n

    def test_symbolic_for_loop_qubits(self):
        """For loop with UInt iteration count → qubits == loop body qubits * iterations."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True)  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("num", [1, 2, 5, 10, 100])
    def test_symbolic_and_int_for_loop_qubits(self, num):
        """For loop with symbolic + int iteration count → qubits == loop body qubits * iterations."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)
            qmc.qubit_array(num, name="extra_qubits")  # Extra qubits outside loop
            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) + num  # type: ignore
        assert resource == expected

    def test_nested_for_loop_qubits(self):
        """Nested for loops with UInt iteration counts → qubits == product of loop body qubits and iterations."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n):
                for j in qmc.range(m):
                    q = qmc.qubit(name=f"q_{i}_{j}")
                    bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) * sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_add_int_for_loop_qubits(self, a):
        """__add__: range(n + a) (symbol + int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n + a):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = a + sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_radd_int_for_loop_qubits(self, a):
        """__radd__: range(a + n) (int + symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(a + n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = a + sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    def test_symbolic_add_symbol_for_loop_qubits(self):
        """__add__: range(n + m) (symbol + symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n + m):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) + sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_sub_int_for_loop_qubits(self, a):
        """__sub__: range(n - a) (symbol - int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n - a):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        n = sp.Symbol("n", integer=True, positive=True)
        expected = sp.Max(0, n - a)
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_rsub_int_for_loop_qubits(self, a):
        """__rsub__: range(a - n) (int - symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(a - n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        n = sp.Symbol("n", integer=True, positive=True)
        expected = sp.Max(0, a - n)
        assert resource == expected

    def test_symbolic_sub_symbol_for_loop_qubits(self):
        """__sub__: range(n - m) (symbol - symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n - m):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        n = sp.Symbol("n", integer=True, positive=True)
        m = sp.Symbol("m", integer=True, positive=True)
        expected = sp.Max(0, n - m)
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_mul_int_for_loop_qubits(self, a):
        """__mul__: range(n * a) (symbol * int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n * a):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) * a
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_rmul_int_for_loop_qubits(self, a):
        """__rmul__: range(a * n) (int * symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(a * n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = a * sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    def test_symbolic_mul_symbol_for_loop_qubits(self):
        """__mul__: range(n * m) (symbol * symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n * m):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) * sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_floordiv_int_for_loop_qubits(self, a):
        """__floordiv__: range(n // a) (symbol // int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n // a):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.floor(sp.Symbol("n", integer=True, positive=True) / a)
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_rfloordiv_int_for_loop_qubits(self, a):
        """__rfloordiv__: range(a // n) (int // symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(a // n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.floor(a / sp.Symbol("n", integer=True, positive=True))
        assert resource == expected

    def test_symbolic_floordiv_symbol_for_loop_qubits(self):
        """__floordiv__: range(n // m) (symbol // symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n // m):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.floor(
            sp.Symbol("n", integer=True, positive=True)
            / sp.Symbol("m", integer=True, positive=True)  # type: ignore
        )
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_pow_int_for_loop_qubits(self, a):
        """__pow__: range(n ** a) (symbol ** int)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n**a):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) ** a
        assert resource == expected

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_symbolic_rpow_int_for_loop_qubits(self, a):
        """__rpow__: range(a ** n) (int ** symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(a**n):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = a ** sp.Symbol("n", integer=True, positive=True)
        assert resource == expected

    def test_symbolic_pow_symbol_for_loop_qubits(self):
        """__pow__: range(n ** m) (symbol ** symbol)."""

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            for i in qmc.range(n**m):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) ** sp.Symbol(
            "m", integer=True, positive=True
        )  # type: ignore
        assert resource == expected


class TestForItemsOp:
    """Verify that resource estimation correctly handles for-items loops over Dict."""

    def test_int_qubits_outside_loop(self):
        """Qubit allocation outside items loop → qubits == allocated qubits, not multiplied by iterations."""

        @qmc.qkernel
        def circuit(
            angles: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            q = qmc.qubit(name="q")
            for _, _ in qmc.items(angles):
                q = qmc.h(q)
            return qmc.measure(q)

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_symbolic_qubits_outside_loop(self):
        """Qubit allocation outside items loop → qubits == allocated qubits, not multiplied by iterations."""

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            for (i, j), _ in qmc.items(ising):
                q[i] = qmc.h(q[i])
            return qmc.measure(q)

        resource = qubits_counter(circuit.block)
        assert resource == sp.Symbol("n", integer=True, positive=True)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_int_for_items_loop_qubits(self, n):
        """For-items loop with qubit_array(n) in body → qubits == n * dict cardinality."""

        @qmc.qkernel
        def circuit(
            angles: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            for _, _ in qmc.items(angles):
                qs = qmc.qubit_array(n, name="qs")
                bit = qmc.measure(qs[0])

            return bit

        resource = qubits_counter(circuit.block)
        expected = n * sp.Symbol("|angles|", integer=True, positive=True)
        assert resource == expected

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_int_for_items_loop_tuple_key_qubits(self, n):
        """For-items loop with tuple-key Dict and qubit_array(n) in body → qubits == n * dict cardinality."""

        @qmc.qkernel
        def circuit(
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Bit:
            for _, _ in qmc.items(ising):
                qs = qmc.qubit_array(n, name="qs")
                bit = qmc.measure(qs[0])

            return bit

        resource = qubits_counter(circuit.block)
        expected = n * sp.Symbol("|ising|", integer=True, positive=True)
        assert resource == expected

    def test_symbolic_for_items_loop_single_key_qubits(self):
        """For-items loop with single-key Dict and qubit allocation in body → qubits == dict cardinality."""

        @qmc.qkernel
        def circuit(
            angles: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            for i, _ in qmc.items(angles):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("|angles|", integer=True, positive=True)  # type: ignore
        assert resource == expected

    def test_symbolic_for_items_loop_qubits(self):
        """For-items loop with qubit allocation in body → qubits == dict cardinality."""

        @qmc.qkernel
        def circuit(
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Bit:
            for (i, _), _ in qmc.items(ising):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("|ising|", integer=True, positive=True)  # type: ignore
        assert resource == expected

    @pytest.mark.parametrize("num", [1, 2, 5, 10, 100])
    def test_symbolic_and_int_for_items_loop_qubits(self, num):
        """For-items loop + extra fixed qubits → qubits == dict cardinality + extra."""

        @qmc.qkernel
        def circuit(
            ising: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Bit:
            for (i, _), _ in qmc.items(ising):
                q = qmc.qubit(name=f"q_{i}")
                bit = qmc.measure(q)
            qmc.qubit_array(num, name="extra_qubits")  # Extra qubits outside loop
            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("|ising|", integer=True, positive=True) + num  # type: ignore
        assert resource == expected

    def test_nested_for_items_loop_single_single_qubits(self):
        """Nested for-items loops (single × single) → qubits == product of dict cardinalities."""

        @qmc.qkernel
        def circuit(
            outer: qmc.Dict[qmc.UInt, qmc.Float],
            inner: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            for i, _ in qmc.items(outer):
                for j, _ in qmc.items(inner):
                    q = qmc.qubit(name=f"q_{i}_{j}")
                    bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("|outer|", integer=True, positive=True) * sp.Symbol(
            "|inner|", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    def test_nested_for_items_loop_single_tuple_qubits(self):
        """Nested for-items loops (single × tuple) → qubits == product of dict cardinalities."""

        @qmc.qkernel
        def circuit(
            outer: qmc.Dict[qmc.UInt, qmc.Float],
            inner: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Bit:
            for i, _ in qmc.items(outer):
                for (j, _), _ in qmc.items(inner):
                    q = qmc.qubit(name=f"q_{i}_{j}")
                    bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("|outer|", integer=True, positive=True) * sp.Symbol(
            "|inner|", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    def test_nested_for_items_loop_tuple_single_qubits(self):
        """Nested for-items loops (tuple × single) → qubits == product of dict cardinalities."""

        @qmc.qkernel
        def circuit(
            outer: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            inner: qmc.Dict[qmc.UInt, qmc.Float],
        ) -> qmc.Bit:
            for (i, _), _ in qmc.items(outer):
                for j, _ in qmc.items(inner):
                    q = qmc.qubit(name=f"q_{i}_{j}")
                    bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("|outer|", integer=True, positive=True) * sp.Symbol(
            "|inner|", integer=True, positive=True
        )  # type: ignore
        assert resource == expected

    def test_nested_for_items_loop_tuple_tuple_qubits(self):
        """Nested for-items loops (tuple × tuple) → qubits == product of dict cardinalities."""

        @qmc.qkernel
        def circuit(
            outer: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
            inner: qmc.Dict[qmc.Tuple[qmc.UInt, qmc.UInt], qmc.Float],
        ) -> qmc.Bit:
            for (i, _), _ in qmc.items(outer):
                for (j, _), _ in qmc.items(inner):
                    q = qmc.qubit(name=f"q_{i}_{j}")
                    bit = qmc.measure(q)

            return bit

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("|outer|", integer=True, positive=True) * sp.Symbol(
            "|inner|", integer=True, positive=True
        )  # type: ignore
        assert resource == expected


class TestIfOp:
    """Verify that resource estimation takes the maximum qubits across branches for if statements."""

    def test_if_single_qubit_in_one_branch(self):
        """If with single qubit in one branch → qubits == 1."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Bit:
            if cond:
                q = qmc.qubit(name="q")
                return qmc.measure(q)
            else:
                return qmc.bit(0)

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_if_different_qubits_in_branches(self):
        """If with different qubits in each branch → qubits == max of both branches."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Bit:
            if cond:
                q = qmc.qubit(name="q")
                return qmc.measure(q)
            else:
                q = qmc.qubit(name="q")
                return qmc.measure(q)

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_if_symbolic_qubits_in_branches_false_larger(self):
        """If with symbolic qubits in each branch → qubits == max of both branches."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            if cond:
                qs = qmc.qubit_array(n, name="qs")
                return qmc.measure(qs[0])
            else:
                qs = qmc.qubit_array(n + 1, name="qs")
                return qmc.measure(qs[0])

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) + 1
        assert resource == expected

    def test_if_symbolic_qubits_in_branches_true_larger(self):
        """If with symbolic qubits in each branch → qubits == max of both branches."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            if cond:
                qs = qmc.qubit_array(n + 1, name="qs")
                return qmc.measure(qs[0])
            else:
                qs = qmc.qubit_array(n, name="qs")
                return qmc.measure(qs[0])

        resource = qubits_counter(circuit.block)
        expected = sp.Symbol("n", integer=True, positive=True) + 1
        assert resource == expected

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_if_int_qubits_in_branches_false_larger(self, n):
        """If with int qubits in each branch → qubits == max of both branches."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Bit:
            if cond:
                qs1 = qmc.qubit_array(n, name="qs1")
                return qmc.measure(qs1[0])
            else:
                qs2 = qmc.qubit_array(n + 1, name="qs2")
                return qmc.measure(qs2[0])

        resource = qubits_counter(circuit.block)
        assert resource == n + 1

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 100])
    def test_if_int_qubits_in_branches_true_larger(self, n):
        """If with int qubits in each branch → qubits == max of both branches."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Bit:
            if cond:
                qs1 = qmc.qubit_array(n + 1, name="qs1")
                return qmc.measure(qs1[0])
            else:
                qs2 = qmc.qubit_array(n, name="qs2")
                return qmc.measure(qs2[0])

        resource = qubits_counter(circuit.block)
        assert resource == n + 1


class _CustomGate(CompositeGate):
    """Custom composite gate with implementation (no new qubits)."""

    def __init__(self, num_qubits: int, num_controls: int = 0):
        self._n = num_qubits
        self._nc = num_controls

    @property
    def num_target_qubits(self) -> int:
        return self._n

    @property
    def num_control_qubits(self) -> int:
        return self._nc

    def _decompose(self, qubits):
        return tuple(qmc.h(q) for q in qubits)


class _CustomGateWithAncilla(CompositeGate):
    """Custom composite gate that allocates one ancilla qubit in decomposition."""

    def __init__(self, num_qubits: int, num_controls: int = 0):
        self._n = num_qubits
        self._nc = num_controls

    @property
    def num_target_qubits(self) -> int:
        return self._n

    @property
    def num_control_qubits(self) -> int:
        return self._nc

    def _decompose(self, qubits):
        ancilla = qmc.qubit(name="ancilla")
        ancilla = qmc.h(ancilla)
        return qubits


def _apply_gate_to_vector(gate, qs, n):
    """Apply composite gate to vector elements (not @qkernel — avoids UInt indexing issue)."""
    result = gate(*[qs[i] for i in range(n)])
    for i in range(n):
        qs[i] = result[i]


def _apply_controlled_gate(gate, qs, n, ctrl):
    """Apply controlled gate, returning updated ctrl."""
    result = gate(*[qs[i] for i in range(n)], controls=[ctrl])
    ctrl_out = result[0]
    for i in range(n):
        qs[i] = result[1 + i]
    return ctrl_out


def _apply_multi_controlled_gate(gate, qs, n_targets, ctrls, n_controls):
    """Apply multi-controlled gate, updating ctrls and qs in place."""
    result = gate(
        *[qs[i] for i in range(n_targets)],
        controls=[ctrls[i] for i in range(n_controls)],
    )
    for i in range(n_controls):
        ctrls[i] = result[i]
    for i in range(n_targets):
        qs[i] = result[n_controls + i]


class TestCompositeGateOperation:
    """Verify that resource estimation correctly counts qubits from composite gate operations."""

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_stub_int_qubits(self, n):
        """Stub gate on n qubits → qubits == n."""
        gate = _StubCompositeGate(_num_targets=n, _custom_name="stub")

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            _apply_gate_to_vector(gate, qs, n)
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_stub_ancilla_int_qubits(self, n):
        """Stub gate with ancilla on n qubits → qubits == n + 2."""
        gate = _StubCompositeGate(
            _num_targets=n,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            _apply_gate_to_vector(gate, qs, n)
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == n + 2

    @pytest.mark.parametrize("k", [1, 2, 5, 10])
    def test_stub_ancilla_parametrize(self, k):
        """Stub gate with variable ancilla count → qubits == 3 + k."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=k),
        )

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(3, name="qs")
            qs[0], qs[1], qs[2] = gate(qs[0], qs[1], qs[2])
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == 3 + k

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_custom_int_qubits(self, n):
        """Custom gate (no new qubits) on n qubits → qubits == n."""
        gate = _CustomGate(n)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            _apply_gate_to_vector(gate, qs, n)
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_custom_ancilla_int_qubits(self, n):
        """Custom gate with ancilla on n qubits → qubits == n + 1."""
        gate = _CustomGateWithAncilla(n)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            _apply_gate_to_vector(gate, qs, n)
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == n + 1

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_stub_and_extra_int_qubits(self, m):
        """Stub gate on 3 qubits + extra m qubits → qubits == 3 + m."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        assert resource == 3 + m

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_stub_ancilla_and_extra_int_qubits(self, m):
        """Stub gate with ancilla + extra m qubits → qubits == 5 + m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        assert resource == 5 + m

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_custom_and_extra_int_qubits(self, m):
        """Custom gate + extra m qubits → qubits == 3 + m."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        assert resource == 3 + m

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_custom_ancilla_and_extra_int_qubits(self, m):
        """Custom gate with ancilla + extra m qubits → qubits == 4 + m."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        assert resource == 4 + m

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_add_int_qubits(self, a):
        """Stub: n + a."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_add_int_qubits(self, a):
        """Stub with ancilla: n + a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_add_int_qubits(self, a):
        """Custom: n + a."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_add_int_qubits(self, a):
        """Custom with ancilla: n + a."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_radd_int_qubits(self, a):
        """Stub: a + n."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a + n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a + n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_radd_int_qubits(self, a):
        """Stub with ancilla: a + n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a + n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a + n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_radd_int_qubits(self, a):
        """Custom: a + n."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a + n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a + n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_radd_int_qubits(self, a):
        """Custom with ancilla: a + n."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a + n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a + n_sym

    def test_stub_symbolic_add_symbol_qubits(self):
        """Stub: n + m."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym + m_sym

    def test_stub_ancilla_symbolic_add_symbol_qubits(self):
        """Stub with ancilla: n + m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym + m_sym

    def test_custom_symbolic_add_symbol_qubits(self):
        """Custom: n + m."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym + m_sym

    def test_custom_ancilla_symbolic_add_symbol_qubits(self):
        """Custom with ancilla: n + m."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n + m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym + m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_sub_int_qubits(self, a):
        """Stub: n - a."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_sub_int_qubits(self, a):
        """Stub with ancilla: n - a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_sub_int_qubits(self, a):
        """Custom: n - a."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_sub_int_qubits(self, a):
        """Custom with ancilla: n - a."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_rsub_int_qubits(self, a):
        """Stub: a - n."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a - n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a - n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_rsub_int_qubits(self, a):
        """Stub with ancilla: a - n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a - n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a - n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_rsub_int_qubits(self, a):
        """Custom: a - n."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a - n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a - n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_rsub_int_qubits(self, a):
        """Custom with ancilla: a - n."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a - n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a - n_sym

    def test_stub_symbolic_sub_symbol_qubits(self):
        """Stub: n - m."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym - m_sym

    def test_stub_ancilla_symbolic_sub_symbol_qubits(self):
        """Stub with ancilla: n - m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym - m_sym

    def test_custom_symbolic_sub_symbol_qubits(self):
        """Custom: n - m."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym - m_sym

    def test_custom_ancilla_symbolic_sub_symbol_qubits(self):
        """Custom with ancilla: n - m."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n - m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym - m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_mul_int_qubits(self, a):
        """Stub: n * a."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_mul_int_qubits(self, a):
        """Stub with ancilla: n * a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_mul_int_qubits(self, a):
        """Custom: n * a."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_mul_int_qubits(self, a):
        """Custom with ancilla: n * a."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_rmul_int_qubits(self, a):
        """Stub: a * n."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a * n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a * n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_rmul_int_qubits(self, a):
        """Stub with ancilla: a * n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a * n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a * n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_rmul_int_qubits(self, a):
        """Custom: a * n."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a * n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a * n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_rmul_int_qubits(self, a):
        """Custom with ancilla: a * n."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a * n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a * n_sym

    def test_stub_symbolic_mul_symbol_qubits(self):
        """Stub: n * m."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym * m_sym

    def test_stub_ancilla_symbolic_mul_symbol_qubits(self):
        """Stub with ancilla: n * m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym * m_sym

    def test_custom_symbolic_mul_symbol_qubits(self):
        """Custom: n * m."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym * m_sym

    def test_custom_ancilla_symbolic_mul_symbol_qubits(self):
        """Custom with ancilla: n * m."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n * m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym * m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_floordiv_int_qubits(self, a):
        """Stub: n // a."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_floordiv_int_qubits(self, a):
        """Stub with ancilla: n // a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_floordiv_int_qubits(self, a):
        """Custom: n // a."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_floordiv_int_qubits(self, a):
        """Custom with ancilla: n // a."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_rfloordiv_int_qubits(self, a):
        """Stub: a // n."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a // n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + sp.floor(a / n_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_rfloordiv_int_qubits(self, a):
        """Stub with ancilla: a // n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a // n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + sp.floor(a / n_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_rfloordiv_int_qubits(self, a):
        """Custom: a // n."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a // n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + sp.floor(a / n_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_rfloordiv_int_qubits(self, a):
        """Custom with ancilla: a // n."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a // n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + sp.floor(a / n_sym)

    def test_stub_symbolic_floordiv_symbol_qubits(self):
        """Stub: n // m."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + sp.floor(n_sym / m_sym)

    def test_stub_ancilla_symbolic_floordiv_symbol_qubits(self):
        """Stub with ancilla: n // m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + sp.floor(n_sym / m_sym)

    def test_custom_symbolic_floordiv_symbol_qubits(self):
        """Custom: n // m."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + sp.floor(n_sym / m_sym)

    def test_custom_ancilla_symbolic_floordiv_symbol_qubits(self):
        """Custom with ancilla: n // m."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n // m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + sp.floor(n_sym / m_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_pow_int_qubits(self, a):
        """Stub: n ** a."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_pow_int_qubits(self, a):
        """Stub with ancilla: n ** a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_pow_int_qubits(self, a):
        """Custom: n ** a."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_pow_int_qubits(self, a):
        """Custom with ancilla: n ** a."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**a, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_symbolic_rpow_int_qubits(self, a):
        """Stub: a ** n."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a**n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a**n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_symbolic_rpow_int_qubits(self, a):
        """Stub with ancilla: a ** n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a**n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a**n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_symbolic_rpow_int_qubits(self, a):
        """Custom: a ** n."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a**n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 3 + a**n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_symbolic_rpow_int_qubits(self, a):
        """Custom with ancilla: a ** n."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(a**n, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a**n_sym

    def test_stub_symbolic_pow_symbol_qubits(self):
        """Stub: n ** m."""
        gate = _StubCompositeGate(_num_targets=3, _custom_name="stub")

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym**m_sym

    def test_stub_ancilla_symbolic_pow_symbol_qubits(self):
        """Stub with ancilla: n ** m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _custom_name="stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym**m_sym

    def test_custom_symbolic_pow_symbol_qubits(self):
        """Custom: n ** m."""
        gate = _CustomGate(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 3 + n_sym**m_sym

    def test_custom_ancilla_symbolic_pow_symbol_qubits(self):
        """Custom with ancilla: n ** m."""
        gate = _CustomGateWithAncilla(3)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            fixed = qmc.qubit_array(3, name="main")
            fixed[0], fixed[1], fixed[2] = gate(fixed[0], fixed[1], fixed[2])
            extra = qmc.qubit_array(n**m, name="extra")
            return fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym**m_sym

    # --- Control qubit tests ---

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_stub_control_int_qubits(self, n):
        """Stub gate with 1 control on n targets → qubits == n + 1."""
        gate = _StubCompositeGate(
            _num_targets=n, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(n, name="qs")
            ctrl = _apply_controlled_gate(gate, qs, n, ctrl)
            return ctrl, qs

        resource = qubits_counter(circuit.block)
        assert resource == n + 1

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_stub_ancilla_control_int_qubits(self, n):
        """Stub gate with ancilla + 1 control on n targets → qubits == n + 3."""
        gate = _StubCompositeGate(
            _num_targets=n,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(n, name="qs")
            ctrl = _apply_controlled_gate(gate, qs, n, ctrl)
            return ctrl, qs

        resource = qubits_counter(circuit.block)
        assert resource == n + 3

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_custom_control_int_qubits(self, n):
        """Custom gate with 1 control on n targets → qubits == n + 1."""
        gate = _CustomGate(n, num_controls=1)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(n, name="qs")
            ctrl = _apply_controlled_gate(gate, qs, n, ctrl)
            return ctrl, qs

        resource = qubits_counter(circuit.block)
        assert resource == n + 1

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_custom_ancilla_control_int_qubits(self, n):
        """Custom gate with ancilla + 1 control on n targets → qubits == n + 2."""
        gate = _CustomGateWithAncilla(n, num_controls=1)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(n, name="qs")
            ctrl = _apply_controlled_gate(gate, qs, n, ctrl)
            return ctrl, qs

        resource = qubits_counter(circuit.block)
        assert resource == n + 2

    # --- Multi-control tests ---

    @pytest.mark.parametrize("c", [1, 2, 3])
    def test_stub_multi_control_int_qubits(self, c):
        """Stub gate with c controls on 3 targets → qubits == 3 + c."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=c, _custom_name="mctrl_stub"
        )

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrls = qmc.qubit_array(c, name="ctrls")
            qs = qmc.qubit_array(3, name="qs")
            _apply_multi_controlled_gate(gate, qs, 3, ctrls, c)
            return ctrls, qs

        resource = qubits_counter(circuit.block)
        assert resource == 3 + c

    @pytest.mark.parametrize("c", [1, 2, 3])
    def test_stub_ancilla_multi_control_int_qubits(self, c):
        """Stub gate with ancilla + c controls on 3 targets → qubits == 5 + c."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=c,
            _custom_name="mctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrls = qmc.qubit_array(c, name="ctrls")
            qs = qmc.qubit_array(3, name="qs")
            _apply_multi_controlled_gate(gate, qs, 3, ctrls, c)
            return ctrls, qs

        resource = qubits_counter(circuit.block)
        assert resource == 5 + c

    @pytest.mark.parametrize("c", [1, 2, 3])
    def test_custom_multi_control_int_qubits(self, c):
        """Custom gate with c controls on 3 targets → qubits == 3 + c."""
        gate = _CustomGate(3, num_controls=c)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrls = qmc.qubit_array(c, name="ctrls")
            qs = qmc.qubit_array(3, name="qs")
            _apply_multi_controlled_gate(gate, qs, 3, ctrls, c)
            return ctrls, qs

        resource = qubits_counter(circuit.block)
        assert resource == 3 + c

    @pytest.mark.parametrize("c", [1, 2, 3])
    def test_custom_ancilla_multi_control_int_qubits(self, c):
        """Custom gate with ancilla + c controls on 3 targets → qubits == 4 + c."""
        gate = _CustomGateWithAncilla(3, num_controls=c)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrls = qmc.qubit_array(c, name="ctrls")
            qs = qmc.qubit_array(3, name="qs")
            _apply_multi_controlled_gate(gate, qs, 3, ctrls, c)
            return ctrls, qs

        resource = qubits_counter(circuit.block)
        assert resource == 4 + c

    # --- Mixed control + extra qubit tests ---

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_stub_control_and_extra_int_qubits(self, m):
        """Stub gate with 1 control on 3 targets + extra m → qubits == 4 + m."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(3, name="qs")
            result = gate(qs[0], qs[1], qs[2], controls=[ctrl])
            ctrl = result[0]
            qs[0], qs[1], qs[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(m, name="extra")
            return ctrl, qs, extra

        resource = qubits_counter(circuit.block)
        assert resource == 4 + m

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_and_extra_int_qubits(self, m):
        """Stub gate with ancilla + 1 control on 3 targets + extra m → qubits == 6 + m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(3, name="qs")
            result = gate(qs[0], qs[1], qs[2], controls=[ctrl])
            ctrl = result[0]
            qs[0], qs[1], qs[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(m, name="extra")
            return ctrl, qs, extra

        resource = qubits_counter(circuit.block)
        assert resource == 6 + m

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_custom_control_and_extra_int_qubits(self, m):
        """Custom gate with 1 control on 3 targets + extra m → qubits == 4 + m."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(3, name="qs")
            result = gate(qs[0], qs[1], qs[2], controls=[ctrl])
            ctrl = result[0]
            qs[0], qs[1], qs[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(m, name="extra")
            return ctrl, qs, extra

        resource = qubits_counter(circuit.block)
        assert resource == 4 + m

    @pytest.mark.parametrize("m", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_and_extra_int_qubits(self, m):
        """Custom gate with ancilla + 1 control on 3 targets + extra m → qubits == 5 + m."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            qs = qmc.qubit_array(3, name="qs")
            result = gate(qs[0], qs[1], qs[2], controls=[ctrl])
            ctrl = result[0]
            qs[0], qs[1], qs[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(m, name="extra")
            return ctrl, qs, extra

        resource = qubits_counter(circuit.block)
        assert resource == 5 + m

    # --- Control qubit symbolic arithmetic tests ---

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_add_int_qubits(self, a):
        """Stub with control: n + a."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_add_int_qubits(self, a):
        """Stub with ancilla + control: n + a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_add_int_qubits(self, a):
        """Custom with control: n + a."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_add_int_qubits(self, a):
        """Custom with ancilla + control: n + a."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_radd_int_qubits(self, a):
        """Stub with control: a + n."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a + n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a + n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_radd_int_qubits(self, a):
        """Stub with ancilla + control: a + n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a + n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + a + n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_radd_int_qubits(self, a):
        """Custom with control: a + n."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a + n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a + n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_radd_int_qubits(self, a):
        """Custom with ancilla + control: a + n."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a + n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a + n_sym

    def test_stub_control_symbolic_add_symbol_qubits(self):
        """Stub with control: n + m."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym + m_sym

    def test_stub_ancilla_control_symbolic_add_symbol_qubits(self):
        """Stub with ancilla + control: n + m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 6 + n_sym + m_sym

    def test_custom_control_symbolic_add_symbol_qubits(self):
        """Custom with control: n + m."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym + m_sym

    def test_custom_ancilla_control_symbolic_add_symbol_qubits(self):
        """Custom with ancilla + control: n + m."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n + m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym + m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_sub_int_qubits(self, a):
        """Stub with control: n - a."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_sub_int_qubits(self, a):
        """Stub with ancilla + control: n - a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_sub_int_qubits(self, a):
        """Custom with control: n - a."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_sub_int_qubits(self, a):
        """Custom with ancilla + control: n - a."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_rsub_int_qubits(self, a):
        """Stub with control: a - n."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a - n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a - n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_rsub_int_qubits(self, a):
        """Stub with ancilla + control: a - n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a - n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + a - n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_rsub_int_qubits(self, a):
        """Custom with control: a - n."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a - n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a - n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_rsub_int_qubits(self, a):
        """Custom with ancilla + control: a - n."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a - n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a - n_sym

    def test_stub_control_symbolic_sub_symbol_qubits(self):
        """Stub with control: n - m."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym - m_sym

    def test_stub_ancilla_control_symbolic_sub_symbol_qubits(self):
        """Stub with ancilla + control: n - m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 6 + n_sym - m_sym

    def test_custom_control_symbolic_sub_symbol_qubits(self):
        """Custom with control: n - m."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym - m_sym

    def test_custom_ancilla_control_symbolic_sub_symbol_qubits(self):
        """Custom with ancilla + control: n - m."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n - m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym - m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_mul_int_qubits(self, a):
        """Stub with control: n * a."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_mul_int_qubits(self, a):
        """Stub with ancilla + control: n * a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_mul_int_qubits(self, a):
        """Custom with control: n * a."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_mul_int_qubits(self, a):
        """Custom with ancilla + control: n * a."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_rmul_int_qubits(self, a):
        """Stub with control: a * n."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a * n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a * n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_rmul_int_qubits(self, a):
        """Stub with ancilla + control: a * n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a * n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + a * n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_rmul_int_qubits(self, a):
        """Custom with control: a * n."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a * n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a * n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_rmul_int_qubits(self, a):
        """Custom with ancilla + control: a * n."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a * n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a * n_sym

    def test_stub_control_symbolic_mul_symbol_qubits(self):
        """Stub with control: n * m."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym * m_sym

    def test_stub_ancilla_control_symbolic_mul_symbol_qubits(self):
        """Stub with ancilla + control: n * m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 6 + n_sym * m_sym

    def test_custom_control_symbolic_mul_symbol_qubits(self):
        """Custom with control: n * m."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym * m_sym

    def test_custom_ancilla_control_symbolic_mul_symbol_qubits(self):
        """Custom with ancilla + control: n * m."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n * m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym * m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_floordiv_int_qubits(self, a):
        """Stub with control: n // a."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_floordiv_int_qubits(self, a):
        """Stub with ancilla + control: n // a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_floordiv_int_qubits(self, a):
        """Custom with control: n // a."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_floordiv_int_qubits(self, a):
        """Custom with ancilla + control: n // a."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_rfloordiv_int_qubits(self, a):
        """Stub with control: a // n."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a // n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + sp.floor(a / n_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_rfloordiv_int_qubits(self, a):
        """Stub with ancilla + control: a // n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a // n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + sp.floor(a / n_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_rfloordiv_int_qubits(self, a):
        """Custom with control: a // n."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a // n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + sp.floor(a / n_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_rfloordiv_int_qubits(self, a):
        """Custom with ancilla + control: a // n."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a // n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + sp.floor(a / n_sym)

    def test_stub_control_symbolic_floordiv_symbol_qubits(self):
        """Stub with control: n // m."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + sp.floor(n_sym / m_sym)

    def test_stub_ancilla_control_symbolic_floordiv_symbol_qubits(self):
        """Stub with ancilla + control: n // m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 6 + sp.floor(n_sym / m_sym)

    def test_custom_control_symbolic_floordiv_symbol_qubits(self):
        """Custom with control: n // m."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + sp.floor(n_sym / m_sym)

    def test_custom_ancilla_control_symbolic_floordiv_symbol_qubits(self):
        """Custom with ancilla + control: n // m."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n // m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + sp.floor(n_sym / m_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_pow_int_qubits(self, a):
        """Stub with control: n ** a."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_pow_int_qubits(self, a):
        """Stub with ancilla + control: n ** a."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_pow_int_qubits(self, a):
        """Custom with control: n ** a."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_pow_int_qubits(self, a):
        """Custom with ancilla + control: n ** a."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**a, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_control_symbolic_rpow_int_qubits(self, a):
        """Stub with control: a ** n."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a**n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a**n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_stub_ancilla_control_symbolic_rpow_int_qubits(self, a):
        """Stub with ancilla + control: a ** n."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a**n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 6 + a**n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_control_symbolic_rpow_int_qubits(self, a):
        """Custom with control: a ** n."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a**n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 4 + a**n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_custom_ancilla_control_symbolic_rpow_int_qubits(self, a):
        """Custom with ancilla + control: a ** n."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt,
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(a**n, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 5 + a**n_sym

    def test_stub_control_symbolic_pow_symbol_qubits(self):
        """Stub with control: n ** m."""
        gate = _StubCompositeGate(
            _num_targets=3, _num_controls=1, _custom_name="ctrl_stub"
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym**m_sym

    def test_stub_ancilla_control_symbolic_pow_symbol_qubits(self):
        """Stub with ancilla + control: n ** m."""
        gate = _StubCompositeGate(
            _num_targets=3,
            _num_controls=1,
            _custom_name="ctrl_stub",
            _resource_metadata=ResourceMetadata(ancilla_qubits=2),
        )

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 6 + n_sym**m_sym

    def test_custom_control_symbolic_pow_symbol_qubits(self):
        """Custom with control: n ** m."""
        gate = _CustomGate(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 4 + n_sym**m_sym

    def test_custom_ancilla_control_symbolic_pow_symbol_qubits(self):
        """Custom with ancilla + control: n ** m."""
        gate = _CustomGateWithAncilla(3, num_controls=1)

        @qmc.qkernel
        def circuit(
            n: qmc.UInt, m: qmc.UInt
        ) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit], qmc.Vector[qmc.Qubit]]:
            ctrl = qmc.qubit(name="ctrl")
            fixed = qmc.qubit_array(3, name="main")
            result = gate(fixed[0], fixed[1], fixed[2], controls=[ctrl])
            ctrl = result[0]
            fixed[0], fixed[1], fixed[2] = result[1], result[2], result[3]
            extra = qmc.qubit_array(n**m, name="extra")
            return ctrl, fixed, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 5 + n_sym**m_sym


class TestCallBlockOperation:
    """Verify that resource estimation correctly counts qubits from CallBlockOperation."""

    def test_no_inner_alloc_single_qubit(self):
        """Inner kernel applies gate to passed qubit, no new allocation → total == 1."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_no_inner_alloc_vector_qubit(self, n):
        """Inner kernel applies gate to passed Vector[Qubit], no new allocation → total == n."""

        @qmc.qkernel
        def inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            qs = inner(qs)
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_inner_alloc_int_qubits(self, n):
        """Inner kernel allocates n qubits internally → total == 1 + n."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1 + n

    @pytest.mark.parametrize("m", [1, 2, 5, 10])
    def test_outer_and_inner_alloc_int_qubits(self, m):
        """Outer allocates 3, inner allocates m → total == 3 + m."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            extra = qmc.qubit_array(m, name="extra")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(3, name="qs")
            qs[0] = inner(qs[0])
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == 3 + m

    def test_inner_alloc_symbolic_qubits(self):
        """Inner kernel allocates n qubits symbolically → total == 1 + n."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + n_sym

    def test_outer_symbolic_inner_no_alloc(self):
        """Outer allocates symbolic qubits, inner just applies gates → total == n."""

        @qmc.qkernel
        def inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            qs = inner(qs)
            return qs

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == n_sym

    def test_outer_and_inner_symbolic_alloc(self):
        """Outer allocates n, inner allocates m → total == n + m."""

        @qmc.qkernel
        def inner(qs: qmc.Vector[qmc.Qubit], m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            extra = qmc.qubit_array(m, name="extra")
            qs[0] = qmc.h(qs[0])
            return qs

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            qs = inner(qs, m)
            return qs

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == n_sym + m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_add_int(self, a):
        """__add__: inner allocates qubit_array(n + a) → total == 1 + n + a."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n + a, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + n_sym + a

    def test_inner_alloc_add_symbol(self):
        """__add__: inner allocates qubit_array(n + m) → total == 1 + n + m."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n + m, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n, m)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 1 + n_sym + m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_radd_int(self, a):
        """__radd__: inner allocates qubit_array(a + n) → total == 1 + a + n."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(a + n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + a + n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_sub_int(self, a):
        """__sub__: inner allocates qubit_array(n - a) → total == 1 + n - a."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n - a, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + n_sym - a

    def test_inner_alloc_sub_symbol(self):
        """__sub__: inner allocates qubit_array(n - m) → total == 1 + n - m."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n - m, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n, m)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 1 + n_sym - m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_rsub_int(self, a):
        """__rsub__: inner allocates qubit_array(a - n) → total == 1 + a - n."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(a - n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + a - n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_mul_int(self, a):
        """__mul__: inner allocates qubit_array(n * a) → total == 1 + n * a."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n * a, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + n_sym * a

    def test_inner_alloc_mul_symbol(self):
        """__mul__: inner allocates qubit_array(n * m) → total == 1 + n * m."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n * m, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n, m)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 1 + n_sym * m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_rmul_int(self, a):
        """__rmul__: inner allocates qubit_array(a * n) → total == 1 + a * n."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(a * n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + a * n_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_floordiv_int(self, a):
        """__floordiv__: inner allocates qubit_array(n // a) → total == 1 + floor(n / a)."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n // a, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + sp.floor(n_sym / a)

    def test_inner_alloc_floordiv_symbol(self):
        """__floordiv__: inner allocates qubit_array(n // m) → total == 1 + floor(n / m)."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n // m, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n, m)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 1 + sp.floor(n_sym / m_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_rfloordiv_int(self, a):
        """__rfloordiv__: inner allocates qubit_array(a // n) → total == 1 + floor(a / n)."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(a // n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + sp.floor(a / n_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_pow_int(self, a):
        """__pow__: inner allocates qubit_array(n ** a) → total == 1 + n ** a."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n**a, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + n_sym**a

    def test_inner_alloc_pow_symbol(self):
        """__pow__: inner allocates qubit_array(n ** m) → total == 1 + n ** m."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(n**m, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n, m)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 1 + n_sym**m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_inner_alloc_rpow_int(self, a):
        """__rpow__: inner allocates qubit_array(a ** n) → total == 1 + a ** n."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            ancilla = qmc.qubit_array(a**n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q, n)
            return q

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + a**n_sym

    def test_vector_pass_no_inner_alloc(self):
        """Pass Vector[Qubit] to inner, no inner allocation → total == n."""

        @qmc.qkernel
        def inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            qs[0] = qmc.h(qs[0])
            return qs

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            qs = inner(qs)
            return qs

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == n_sym

    @pytest.mark.parametrize("m", [1, 2, 5, 10])
    def test_vector_pass_inner_alloc_int(self, m):
        """Pass Vector[Qubit] to inner, inner allocates m extra → total == n + m."""

        @qmc.qkernel
        def inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            extra = qmc.qubit_array(m, name="extra")
            qs[0] = qmc.h(qs[0])
            return qs

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            qs = inner(qs)
            return qs

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == n_sym + m

    def test_multiple_calls_no_inner_alloc(self):
        """Inner kernel called twice, no inner allocation → total == 2."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q1 = inner(q1)
            q2 = inner(q2)
            return q1, q2

        resource = qubits_counter(circuit.block)
        assert resource == 2

    @pytest.mark.parametrize("m", [1, 2, 5, 10])
    def test_multiple_calls_inner_alloc_int(self, m):
        """Inner kernel called twice, each allocates m → total == 2 + 2*m."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            extra = qmc.qubit_array(m, name="extra")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q1 = inner(q1)
            q2 = inner(q2)
            return q1, q2

        resource = qubits_counter(circuit.block)
        assert resource == 2 + 2 * m

    def test_multiple_calls_inner_alloc_symbolic(self):
        """Inner kernel called twice, each allocates n → total == 2 + 2*n."""

        @qmc.qkernel
        def inner(q: qmc.Qubit, n: qmc.UInt) -> qmc.Qubit:
            extra = qmc.qubit_array(n, name="extra")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Qubit, qmc.Qubit]:
            q1 = qmc.qubit(name="q1")
            q2 = qmc.qubit(name="q2")
            q1 = inner(q1, n)
            q2 = inner(q2, n)
            return q1, q2

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 2 + 2 * n_sym

    def test_nested_calls_no_alloc(self):
        """Three-level nesting, no internal allocation in mid/inner → total == 1."""

        @qmc.qkernel
        def inner_inner(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            q = inner_inner(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1

    @pytest.mark.parametrize("m", [1, 2, 5, 10])
    def test_nested_calls_innermost_alloc_int(self, m):
        """Three-level nesting, innermost allocates m → total == 1 + m."""

        @qmc.qkernel
        def inner_inner(q: qmc.Qubit) -> qmc.Qubit:
            extra = qmc.qubit_array(m, name="deep_extra")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            q = inner_inner(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1 + m

    def test_nested_calls_all_levels_alloc(self):
        """Three-level nesting, each level allocates 1 qubit → total == 3."""

        @qmc.qkernel
        def inner_inner(q: qmc.Qubit) -> qmc.Qubit:
            extra = qmc.qubit(name="deep_extra")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            extra = qmc.qubit(name="mid_extra")
            q = inner_inner(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 3

    def test_nested_calls_all_levels_symbolic_alloc(self):
        """Three-level nesting, outer: n, mid: m, inner: k → total == n + m + k."""

        @qmc.qkernel
        def inner_inner(q: qmc.Qubit, k: qmc.UInt) -> qmc.Qubit:
            extra = qmc.qubit_array(k, name="deep_extra")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def inner(q: qmc.Qubit, m: qmc.UInt, k: qmc.UInt) -> qmc.Qubit:
            extra = qmc.qubit_array(m, name="mid_extra")
            q = inner_inner(q, k)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt, k: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            qs[0] = inner(qs[0], m, k)
            return qs

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        k_sym = sp.Symbol("k", integer=True, positive=True)
        assert resource == n_sym + m_sym + k_sym


class TestWhileOperation:
    """Verify that resource estimation correctly counts qubits from while loops."""

    W = sp.Symbol("|while|", integer=True, positive=True)

    def test_int_qubits_outside_while(self):
        """Qubit allocation outside while → qubits == allocated qubits, not affected by loop."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Bit:
            q = qmc.qubit(name="q")
            while cond:
                q = qmc.h(q)
            return qmc.measure(q)

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_symbolic_qubits_outside_while(self):
        """Symbolic qubit array outside while → qubits == n."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Vector[qmc.Bit]:
            q = qmc.qubit_array(n, name="q")
            while cond:
                q[0] = qmc.h(q[0])
            return qmc.measure(q)

        resource = qubits_counter(circuit.block)
        assert resource == sp.Symbol("n", integer=True, positive=True)

    def test_while_loop_qubits(self):
        """While body allocates 1 qubit → qubits == |while|."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            return bit

        resource = qubits_counter(circuit.block)
        assert resource == self.W

    @pytest.mark.parametrize("num", [1, 2, 5, 10, 100])
    def test_while_and_int_extra_qubits(self, num):
        """While body (1 qubit) + extra int qubits → |while| + num."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(num, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        assert resource == self.W + num

    def test_while_symbolic_extra_qubits(self):
        """While body (1 qubit) + extra symbolic qubits → |while| + n."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + n_sym

    def test_nested_while_loop_qubits(self):
        """Nested while loops → |while|²."""

        @qmc.qkernel
        def circuit(cond1: qmc.Bit, cond2: qmc.Bit) -> qmc.Bit:
            while cond1:
                while cond2:
                    q = qmc.qubit(name="wq")
                    bit = qmc.measure(q)
            return bit

        resource = qubits_counter(circuit.block)
        assert resource == self.W**2

    # --- Arithmetic tests: while body (1 qubit) + extra array ---

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_add_int_extra_qubits(self, a):
        """__add__: while + extra(n + a)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n + a, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + n_sym + a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_radd_int_extra_qubits(self, a):
        """__radd__: while + extra(a + n)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(a + n, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + a + n_sym

    def test_while_add_symbol_extra_qubits(self):
        """__add__: while + extra(n + m)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n + m, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == self.W + n_sym + m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_sub_int_extra_qubits(self, a):
        """__sub__: while + extra(n - a)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n - a, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + n_sym - a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_rsub_int_extra_qubits(self, a):
        """__rsub__: while + extra(a - n)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(a - n, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + a - n_sym

    def test_while_sub_symbol_extra_qubits(self):
        """__sub__: while + extra(n - m)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n - m, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == self.W + n_sym - m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_mul_int_extra_qubits(self, a):
        """__mul__: while + extra(n * a)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n * a, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + n_sym * a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_rmul_int_extra_qubits(self, a):
        """__rmul__: while + extra(a * n)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(a * n, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + a * n_sym

    def test_while_mul_symbol_extra_qubits(self):
        """__mul__: while + extra(n * m)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n * m, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == self.W + n_sym * m_sym

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_floordiv_int_extra_qubits(self, a):
        """__floordiv__: while + extra(n // a)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n // a, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + sp.floor(n_sym / a)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_rfloordiv_int_extra_qubits(self, a):
        """__rfloordiv__: while + extra(a // n)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(a // n, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + sp.floor(a / n_sym)

    def test_while_floordiv_symbol_extra_qubits(self):
        """__floordiv__: while + extra(n // m)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n // m, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == self.W + sp.floor(n_sym / m_sym)

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_pow_int_extra_qubits(self, a):
        """__pow__: while + extra(n ** a)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n**a, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + n_sym**a

    @pytest.mark.parametrize("a", [1, 2, 5, 10, 100])
    def test_while_rpow_int_extra_qubits(self, a):
        """__rpow__: while + extra(a ** n)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(a**n, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == self.W + a**n_sym

    def test_while_pow_symbol_extra_qubits(self):
        """__pow__: while + extra(n ** m)."""

        @qmc.qkernel
        def circuit(cond: qmc.Bit, n: qmc.UInt, m: qmc.UInt) -> qmc.Bit:
            while cond:
                q = qmc.qubit(name="wq")
                bit = qmc.measure(q)
            qmc.qubit_array(n**m, name="extra")
            return bit

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == self.W + n_sym**m_sym


class TestControlledUOperation:
    """Verify that resource estimation correctly counts qubits from controlled-U operations."""

    def test_no_inner_alloc_single_target(self):
        """controlled(h)(ctrl, target) with no inner allocation → total == 2."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
            ctrl = qmc.qubit(name="ctrl")
            target = qmc.qubit(name="target")
            cgate = qmc.controlled(gate)
            ctrl, target = cgate(ctrl, target)
            return ctrl, target

        resource = qubits_counter(circuit.block)
        assert resource == 2

    def test_no_inner_alloc_parametric(self):
        """controlled(rx)(ctrl, target, theta=0.5) with no inner allocation → total == 2."""

        @qmc.qkernel
        def gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, theta)
            return q

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
            ctrl = qmc.qubit(name="ctrl")
            target = qmc.qubit(name="target")
            cgate = qmc.controlled(gate)
            ctrl, target = cgate(ctrl, target, theta=0.5)
            return ctrl, target

        resource = qubits_counter(circuit.block)
        assert resource == 2

    def test_no_inner_alloc_symbolic_qubits(self):
        """Outer allocates qubit_array(n) + qubit(target), controlled in loop → total == n + 1."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            counting = qmc.qubit_array(n, name="counting")
            target = qmc.qubit(name="target")
            cgate = qmc.controlled(gate)
            for i in qmc.range(n):
                counting[i], target = cgate(counting[i], target)
            return counting

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == n_sym + 1

    def test_no_inner_alloc_power(self):
        """controlled(gate, power=4) → qubit count same as power=1 (total == 2)."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
            ctrl = qmc.qubit(name="ctrl")
            target = qmc.qubit(name="target")
            cgate = qmc.controlled(gate)
            ctrl, target = cgate(ctrl, target, power=4)
            return ctrl, target

        resource = qubits_counter(circuit.block)
        assert resource == 2

    def test_no_inner_alloc_in_loop(self):
        """controlled in qm.range(m) loop → qubit count unchanged."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(m: qmc.UInt) -> tuple[qmc.Qubit, qmc.Qubit]:
            ctrl = qmc.qubit(name="ctrl")
            target = qmc.qubit(name="target")
            cgate = qmc.controlled(gate)
            for _ in qmc.range(m):
                ctrl, target = cgate(ctrl, target)
            return ctrl, target

        resource = qubits_counter(circuit.block)
        assert resource == 2

    def test_no_inner_alloc_multiple_calls(self):
        """Same controlled gate applied to different qubit pairs → only outer allocation."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit, qmc.Qubit]:
            ctrl1 = qmc.qubit(name="ctrl1")
            target1 = qmc.qubit(name="target1")
            ctrl2 = qmc.qubit(name="ctrl2")
            target2 = qmc.qubit(name="target2")
            cgate = qmc.controlled(gate)
            ctrl1, target1 = cgate(ctrl1, target1)
            ctrl2, target2 = cgate(ctrl2, target2)
            return ctrl1, target1, ctrl2, target2

        resource = qubits_counter(circuit.block)
        assert resource == 4

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_inner_alloc_int_qubits(self, n):
        """Inner block allocates n qubits → total == 2 + n."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            _ancilla = qmc.qubit_array(n, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Qubit]:
            ctrl = qmc.qubit(name="ctrl")
            target = qmc.qubit(name="target")
            cgate = qmc.controlled(gate)
            ctrl, target = cgate(ctrl, target)
            return ctrl, target

        resource = qubits_counter(circuit.block)
        assert resource == 2 + n

    def test_inner_alloc_symbolic_qubits(self):
        """Inner block allocates symbolic m qubits → total == 2 + m."""

        @qmc.qkernel
        def gate(q: qmc.Qubit, m: qmc.UInt) -> qmc.Qubit:
            _ancilla = qmc.qubit_array(m, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(m: qmc.UInt) -> tuple[qmc.Qubit, qmc.Qubit]:
            ctrl = qmc.qubit(name="ctrl")
            target = qmc.qubit(name="target")
            cgate = qmc.controlled(gate)
            ctrl, target = cgate(ctrl, target, m=m)
            return ctrl, target

        resource = qubits_counter(circuit.block)
        m_sym = sp.Symbol("m", integer=True, positive=True)
        assert resource == 2 + m_sym


class TestControlledUIndexSpec:
    """Verify qubit counting for ControlledUOperation with target_indices/controlled_indices."""

    def test_target_indices_no_inner_alloc(self):
        """controlled(z, num_controls=3) with target_indices=[3] on 4-qubit array → 4."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.z(q)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            cg = qmc.controlled(gate, num_controls=3)
            qs = cg(qs, target_indices=[3])
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == 4

    def test_controlled_indices_no_inner_alloc(self):
        """controlled(z, num_controls=3) with controlled_indices=[0,1,2] on 4-qubit array → 4."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.z(q)

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            cg = qmc.controlled(gate, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, 2])
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == 4

    def test_controlled_indices_inner_alloc(self):
        """Inner block allocates ancilla → total == array_size + ancilla."""

        @qmc.qkernel
        def gate_with_ancilla(q: qmc.Qubit) -> qmc.Qubit:
            _anc = qmc.qubit_array(3, name="anc")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            cg = qmc.controlled(gate_with_ancilla, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, 2])
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == 4 + 3

    def test_controlled_indices_with_params(self):
        """Inner block with float param, controlled_indices → qubits == 4."""

        @qmc.qkernel
        def param_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
            q = qmc.rx(q, theta)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            cg = qmc.controlled(param_gate, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, 2], theta=0.5)
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == 4

    def test_controlled_indices_in_loop(self):
        """controlled_indices inside loop → no new qubits per iteration."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.z(q)

        @qmc.qkernel
        def circuit(m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            cg = qmc.controlled(gate, num_controls=3)
            for _ in qmc.range(m):
                qs = cg(qs, controlled_indices=[0, 1, 2])
            return qs

        resource = qubits_counter(circuit.block)
        assert resource == 4

    def test_controlled_vs_target_indices_equivalence(self):
        """Same partition: controlled_indices=[0,1,2] vs target_indices=[3] → same qubit count."""

        @qmc.qkernel
        def gate(q: qmc.Qubit) -> qmc.Qubit:
            return qmc.z(q)

        @qmc.qkernel
        def circuit_ci() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            cg = qmc.controlled(gate, num_controls=3)
            qs = cg(qs, controlled_indices=[0, 1, 2])
            return qs

        @qmc.qkernel
        def circuit_ti() -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(4, name="qs")
            cg = qmc.controlled(gate, num_controls=3)
            qs = cg(qs, target_indices=[3])
            return qs

        resource_ci = qubits_counter(circuit_ci.block)
        resource_ti = qubits_counter(circuit_ti.block)
        assert resource_ci == resource_ti


class TestInputQubits:
    """Verify that resource estimation correctly counts qubits from input qubits."""

    def test_int_qubit(self):
        """Input qubits should be counted."""

        @qmc.qkernel
        def circuit(q1: qmc.Qubit) -> qmc.Qubit:
            return q1

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_int_two_qubits(self):
        """Input qubits should be counted."""

        @qmc.qkernel
        def circuit(q1: qmc.Qubit, q2: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
            return q1, q2

        resource = qubits_counter(circuit.block)
        assert resource == 2

    def test_int_two_qubits_and_allocated_qubits(self):
        """Input qubits + allocated qubits should be counted."""

        @qmc.qkernel
        def circuit(
            q1: qmc.Qubit, q2: qmc.Qubit
        ) -> tuple[qmc.Qubit, qmc.Qubit, qmc.Qubit]:
            q3 = qmc.qubit(name="q3")
            return q1, q2, q3

        resource = qubits_counter(circuit.block)
        assert resource == 3

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_int_qubits(self, n):
        """Int qubits should be counted."""

        @qmc.qkernel
        def circuit() -> qmc.Vector[qmc.Qubit]:
            extra = qmc.qubit_array(n, name="extra")
            return extra

        resource = qubits_counter(circuit.block)
        assert resource == n

    @pytest.mark.parametrize("n", [1, 2, 5, 10])
    def test_int_qubits_and_allocated_qubits(self, n):
        """Int qubits + allocated qubits should be counted."""

        @qmc.qkernel
        def circuit() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
            q1 = qmc.qubit(name="q1")
            extra = qmc.qubit_array(n, name="extra")
            return q1, extra

        resource = qubits_counter(circuit.block)
        assert resource == 1 + n

    def test_symbolic_qubits(self):
        """Symbolic qubits should be counted."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            extra = qmc.qubit_array(n, name="extra")
            return extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == n_sym

    def test_symbolic_qubits_and_allocated_qubits(self):
        """Symbolic qubits + allocated qubits should be counted."""

        @qmc.qkernel
        def circuit(n: qmc.UInt) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
            q1 = qmc.qubit(name="q1")
            extra = qmc.qubit_array(n, name="extra")
            return q1, extra

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 1 + n_sym


class TestCallBlockInForOperation:
    """Verify ancilla qubit reuse for clean CallBlockOperation inside loops."""

    def test_clean_call_no_alloc_in_for(self):
        """Clean call with no inner alloc inside for loop -> no extra qubits."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            for _ in qmc.range(10):
                q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1

    def test_clean_call_with_ancilla_in_for(self):
        """Clean call with internal ancilla inside for loop -> ancilla counted once."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            ancilla = qmc.qubit_array(3, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            for _ in qmc.range(10):
                q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1 + 3  # q + ancilla (reused)

    def test_clean_call_with_symbolic_ancilla_in_for(self):
        """Clean call with symbolic ancilla inside for loop -> ancilla counted once."""

        @qmc.qkernel
        def inner(qs: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
            n = qs.shape[0]
            ancilla = qmc.qubit_array(n - 2, name="ancilla")
            qs[0] = qmc.h(qs[0])
            return qs

        @qmc.qkernel
        def circuit(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
            qs = qmc.qubit_array(n, name="qs")
            for _ in qmc.range(m):
                qs = inner(qs)
            return qs

        resource = qubits_counter(circuit.block)
        n_sym = sp.Symbol("n", integer=True, positive=True)
        assert resource == 2 * n_sym - 2

    def test_non_clean_call_in_for_multiplicative(self):
        """Non-clean call (returns new qubits) inside for loop -> multiplied."""

        @qmc.qkernel
        def inner() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            for _ in qmc.range(5):
                q = inner()
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 5

    def test_mixed_clean_and_persistent_in_for(self):
        """Mix of clean call (ancilla reusable) and QInit (persistent) in loop."""

        @qmc.qkernel
        def clean_inner(q: qmc.Qubit) -> qmc.Qubit:
            ancilla = qmc.qubit_array(2, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            for _ in qmc.range(4):
                q = clean_inner(q)
                extra = qmc.qubit(name="extra")
            return q

        resource = qubits_counter(circuit.block)
        # q(1) + 4 * extra(1) persistent + 2 reusable ancilla = 1 + 4 + 2 = 7
        assert resource == 7

    def test_clean_call_in_while_loop(self):
        """Clean call with ancilla in while loop -> ancilla counted once."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            ancilla = qmc.qubit_array(3, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit(cond: qmc.Bit) -> qmc.Qubit:
            q = qmc.qubit(name="q")
            while cond:
                q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1 + 3  # q + ancilla (reused)

    def test_multiple_clean_calls_max_watermark(self):
        """Multiple clean calls in loop body -> reusable is max of all ancilla."""

        @qmc.qkernel
        def small_inner(q: qmc.Qubit) -> qmc.Qubit:
            ancilla = qmc.qubit_array(2, name="small_anc")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def large_inner(q: qmc.Qubit) -> qmc.Qubit:
            ancilla = qmc.qubit_array(5, name="large_anc")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            for _ in qmc.range(10):
                q = small_inner(q)
                q = large_inner(q)
            return q

        resource = qubits_counter(circuit.block)
        # q(1) + max(2, 5) reusable = 1 + 5 = 6
        assert resource == 6

    def test_zero_iteration_for_no_reusable(self):
        """For loop with 0 iterations -> no qubits from body."""

        @qmc.qkernel
        def inner(q: qmc.Qubit) -> qmc.Qubit:
            ancilla = qmc.qubit_array(3, name="ancilla")
            q = qmc.h(q)
            return q

        @qmc.qkernel
        def circuit() -> qmc.Qubit:
            q = qmc.qubit(name="q")
            for _ in qmc.range(0, 0):
                q = inner(q)
            return q

        resource = qubits_counter(circuit.block)
        assert resource == 1  # Just q, no ancilla since 0 iterations
