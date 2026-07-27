"""Tests for the constant modular multiplication stdlib primitive."""

from __future__ import annotations

import importlib.util

import pytest

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.arithmetic import (
    _xor_constant,
    modmul_const,
)


def _value_of(bits: tuple[int, ...]) -> int:
    """Decode a little-endian measured bitstring into an integer.

    Args:
        bits (tuple[int, ...]): Measured bits, ``bits[0]`` least significant.

    Returns:
        int: Encoded register value.
    """
    return sum(bit * (2**i) for i, bit in enumerate(bits))


def _load_probe(a: int, x: int, n: int, modulus: int, path: str):
    """Generate and import a kernel preparing ``|x>`` then multiplying by ``a``.

    Args:
        a (int): Constant multiplier.
        x (int): Basis state to prepare.
        n (int): Register width.
        modulus (int): Modulus.
        path (str): File path to write the generated kernel to (AST transform
            requires a real source file).

    Returns:
        QKernel: Generated classical-I/O kernel returning the measured register.
    """
    set_bits = [i for i in range(n) if (x >> i) & 1]
    prep = "\n".join(f"    reg[{i}] = qmc.x(reg[{i}])" for i in set_bits) or "    pass"
    src = (
        "import qamomile.circuit as qmc\n"
        "from qamomile.circuit.stdlib.arithmetic import modmul_const\n"
        "@qmc.qkernel\n"
        "def probe() -> qmc.Vector[qmc.Bit]:\n"
        f"    reg = qmc.qubit_array({n}, name='reg')\n"
        f"{prep}\n"
        f"    reg = modmul_const(reg, multiplier={a}, modulus={modulus})\n"
        "    return qmc.measure(reg)\n"
    )
    with open(path, "w") as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location("modmul_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.probe


def test_modmul_const_non_cyclic_executes(sdk_transpiler, tmp_path) -> None:
    """A small non-rotation instance executes the polynomial reversible body."""
    if sdk_transpiler.backend_name == "quri_parts":
        pytest.skip("QURI Parts cannot represent modmul's mid-circuit reset")

    probe = _load_probe(3, 2, 3, 7, str(tmp_path / "noncyclic.py"))
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(probe)
    result = executable.sample(transpiler.executor(), shots=1).result()

    assert _value_of(result.most_common(1)[0][0]) == 6


def test_modmul_const_non_cyclic_estimates_its_executable_body() -> None:
    """A non-rotation estimate traverses the reversible implementation."""

    @qmc.qkernel
    def mul() -> qmc.Vector[qmc.Qubit]:
        """Multiply a four-bit register by a non-rotation constant once."""
        reg = qmc.qubit_array(4, name="reg")
        reg = modmul_const(reg, multiplier=7, modulus=15)
        return reg

    est = mul.estimate_resources()
    assert est.gates.total > 0
    assert "modmul_const" not in est.calls.calls_by_name


def test_modmul_const_body_growth_is_quadratic_at_fixed_window() -> None:
    """Specialized body estimates follow quadratic modular-multiply growth."""
    cases = [(2, 2, 3), (3, 2, 5), (4, 2, 15)]
    normalized = []
    for width, multiplier, modulus in cases:

        @qmc.qkernel
        def mul() -> qmc.Vector[qmc.Qubit]:
            """Build one specialized modular multiplication."""
            reg = qmc.qubit_array(width, name="reg")
            return modmul_const(
                reg,
                multiplier=multiplier,
                modulus=modulus,
                window_size=2,
            )

        estimate = mul.estimate_resources()
        assert estimate.qubits == 3 * width + 2 + 7
        normalized.append(float(estimate.gates.total) / (width**2))

    assert max(normalized) / min(normalized) < 1.5


def test_modmul_const_rejects_non_coprime_multiplier() -> None:
    """A multiplier sharing a factor with the modulus is rejected."""

    @qmc.qkernel
    def bad() -> qmc.Vector[qmc.Qubit]:
        """Attempt a non-invertible modular multiplication."""
        reg = qmc.qubit_array(4, name="reg")
        return modmul_const(reg, multiplier=3, modulus=15)

    with pytest.raises(ValueError, match="coprime"):
        bad.build()


def test_modmul_const_rejects_modulus_wider_than_register() -> None:
    """A modulus that cannot be represented by the register is rejected."""

    @qmc.qkernel
    def bad() -> qmc.Vector[qmc.Qubit]:
        """Attempt multiplication modulo five in a two-qubit register."""
        reg = qmc.qubit_array(2, name="reg")
        return modmul_const(reg, multiplier=3, modulus=5)

    with pytest.raises(ValueError, match="does not fit"):
        bad.build()


@qmc.qkernel
def _controlled_xor_phase_probe() -> qmc.Bit:
    """Kick an exact controlled XOR against an X-eigenstate target."""
    control = qmc.qubit("control")
    target = qmc.qubit_array(1, name="target")
    control = qmc.h(control)
    target[0] = qmc.h(target[0])
    controlled_xor = qmc.control(_xor_constant)
    control, target = controlled_xor(control, target, 1)
    control = qmc.h(control)
    return qmc.measure(control)


def test_xor_constant_preserves_controlled_phase(sdk_transpiler) -> None:
    """Controlled XOR uses X, not the relative-phase ``RX(pi)`` surrogate."""
    transpiler = sdk_transpiler.transpiler
    result = (
        transpiler.transpile(_controlled_xor_phase_probe)
        .sample(transpiler.executor(), shots=1)
        .result()
    )

    assert result.results == [(0, 1)]


def test_modmul_const_requires_a_concrete_register_width() -> None:
    """FTQC modular multiplication is specialized to a concrete width."""

    @qmc.qkernel
    def mul(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Multiply an n-bit register by a constant once."""
        reg = qmc.qubit_array(n, name="reg")
        reg = modmul_const(reg, multiplier=2, modulus=15)
        return reg

    with pytest.raises(ValueError, match="requires a concrete register width"):
        mul.build()


def test_modmul_const_rejects_invalid_window_size() -> None:
    """The standard lookup multiplier rejects an empty address window."""

    @qmc.qkernel
    def bad() -> qmc.Vector[qmc.Qubit]:
        """Attempt modular multiplication with a zero-width lookup."""
        reg = qmc.qubit_array(2, name="reg")
        return modmul_const(
            reg,
            multiplier=2,
            modulus=3,
            window_size=0,
        )

    with pytest.raises(ValueError, match="window_size must be positive"):
        bad.build()


def test_modmul_const_requires_a_real_problem_instance() -> None:
    """Omitting arithmetic constants is no longer an estimation-only mode."""

    @qmc.qkernel
    def abstract(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Apply an abstract constant modular multiplication once."""
        reg = qmc.qubit_array(n, name="reg")
        reg = modmul_const(reg)
        return reg

    with pytest.raises(TypeError, match="multiplier.*modulus"):
        abstract.build()


def test_modmul_const_control_is_derived_from_the_controlled_body() -> None:
    """Controlled multiplication reclassifies its actual controlled swaps."""

    @qmc.qkernel
    def uncontrolled() -> qmc.Vector[qmc.Qubit]:
        """Multiply a four-bit register by a constant, no control."""
        reg = qmc.qubit_array(4, name="reg")
        reg = modmul_const(reg, multiplier=2, modulus=15)
        return reg

    @qmc.qkernel
    def controlled() -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
        """Multiply a four-bit register, conditioned on a control."""
        ctrl = qmc.qubit("c")
        reg = qmc.qubit_array(4, name="reg")
        ctrl, reg = modmul_const(reg, multiplier=2, modulus=15, control=ctrl)
        return ctrl, reg

    plain = uncontrolled.estimate_resources()
    ctrl = controlled.estimate_resources()

    assert plain.assumptions == ()
    assert ctrl.assumptions == ()
    assert plain.gates.two_qubit > 0
    assert ctrl.gates.multi_qubit > 0


def test_modmul_const_requires_both_constants() -> None:
    """Supplying only one arithmetic constant is a normal signature error."""

    @qmc.qkernel
    def half() -> qmc.Vector[qmc.Qubit]:
        """Attempt modmul with a multiplier but no modulus."""
        reg = qmc.qubit_array(4, name="reg")
        return modmul_const(reg, multiplier=2)

    with pytest.raises(TypeError, match="modulus"):
        half.build()


def test_modmul_const_has_no_abstract_execution_mode(sdk_transpiler) -> None:
    """The removed no-argument form fails before backend lowering."""

    @qmc.qkernel
    def abstract() -> qmc.Vector[qmc.Bit]:
        """Apply an abstract modular multiplication then measure."""
        reg = qmc.qubit_array(4, name="reg")
        reg = modmul_const(reg)
        return qmc.measure(reg)

    with pytest.raises(TypeError, match="multiplier.*modulus"):
        sdk_transpiler.transpiler.transpile(abstract)


@pytest.mark.parametrize("x", [0, 1, 2, 3])
def test_modmul_const_small_instance_is_correct(
    qiskit_transpiler, x: int, tmp_path
) -> None:
    """Execute the smallest nontrivial FTQC modular multiplication.

    The n=2 instance exercises the same polynomial reversible body used by
    larger problems without allocating the n=4 workspace statevector. The
    21-qubit Shor benchmark separately verifies large-instance transpilation.
    """
    a, n, modulus = 2, 2, 3
    path = str(tmp_path / f"probe_{a}_{x}.py")
    probe = _load_probe(a, x, n, modulus, path)
    exe = qiskit_transpiler.transpile(probe)
    measured = exe.sample(qiskit_transpiler.executor(), shots=1).result().most_common(1)
    got = _value_of(measured[0][0])
    expected = (a * x) % modulus if x < modulus else x
    assert got == expected


@qmc.qkernel
def _modmul_phase_kickback() -> tuple[qmc.Bit, qmc.Vector[qmc.Bit]]:
    """Kick the multiplier's minus eigenphase into a control qubit."""
    control = qmc.qubit("control")
    reg = qmc.qubit_array(2, name="reg")
    control = qmc.h(control)
    reg[0] = qmc.x(reg[0])
    reg[1] = qmc.h(reg[1])
    reg[1], reg[0] = qmc.cx(reg[1], reg[0])
    reg[1] = qmc.z(reg[1])
    control, reg = modmul_const(
        reg,
        multiplier=2,
        modulus=3,
        control=control,
    )
    control = qmc.h(control)
    return qmc.measure(control), qmc.measure(reg)


def test_modmul_const_preserves_coherent_phase(qiskit_transpiler) -> None:
    """Measurement-assisted arithmetic preserves multiplier eigenphases."""
    result = (
        qiskit_transpiler.transpile(_modmul_phase_kickback)
        .sample(qiskit_transpiler.executor(), shots=8)
        .result()
    )
    assert sum(count for (control, _), count in result.results if control == 1) == 8
    assert {bits for (_, bits), _ in result.results} <= {(1, 0), (0, 1)}
