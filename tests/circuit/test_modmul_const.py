"""Tests for the constant modular multiplication stdlib primitive."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.stdlib.arithmetic import modmul_const


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
    """A non-rotation constant uses the reversible permutation body."""
    probe = _load_probe(7, 2, 4, 15, str(tmp_path / "noncyclic.py"))
    transpiler = sdk_transpiler.transpiler
    executable = transpiler.transpile(probe)
    result = executable.sample(transpiler.executor(), shots=64).result()

    assert _value_of(result.most_common(1)[0][0]) == 14


def test_modmul_const_non_cyclic_uses_attached_model() -> None:
    """A non-rotation body and its scalable resource model coexist."""

    @qmc.qkernel
    def mul() -> qmc.Vector[qmc.Qubit]:
        """Multiply a four-bit register by a non-rotation constant once."""
        reg = qmc.qubit_array(4, name="reg")
        reg = modmul_const(reg, multiplier=7, modulus=15)
        return reg

    est = mul.estimate_resources()
    assert est.calls.calls_by_name.get("modmul_const") == 1


def test_modmul_const_rejects_non_coprime_multiplier() -> None:
    """A multiplier sharing a factor with the modulus is rejected."""

    @qmc.qkernel
    def bad() -> qmc.Vector[qmc.Qubit]:
        """Attempt a non-invertible modular multiplication."""
        reg = qmc.qubit_array(4, name="reg")
        return modmul_const(reg, multiplier=3, modulus=15)

    with pytest.raises(ValueError, match="coprime"):
        bad.build()


def test_modmul_const_symbolic_resource_model() -> None:
    """A symbolic width asks for a concrete synthesis width explicitly."""

    @qmc.qkernel
    def mul(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Multiply an n-bit register by a constant once."""
        reg = qmc.qubit_array(n, name="reg")
        reg = modmul_const(reg, multiplier=2, modulus=15)
        return reg

    with pytest.raises(ValueError, match="compile-time-known register width"):
        mul.estimate_resources()


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


def test_modmul_const_control_coverage_assumption_matches_call() -> None:
    """The control-coverage assumption is honest about whether the call is controlled.

    An uncontrolled ``modmul_const`` must not claim its cost "includes the single
    control", while a controlled one must; the gate counts are identical either
    way (the control is lower-order against the O(n^2) leading term).
    """

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

    def messages(est):
        return [a.message for a in est.assumptions if a.source == "modmul_const"]

    plain_msgs = " ".join(messages(plain))
    ctrl_msgs = " ".join(messages(ctrl))
    assert "includes the single control" not in plain_msgs
    assert "overestimate" in plain_msgs
    assert "includes the single control" in ctrl_msgs
    # Same cost regardless of control (control is lower-order).
    assert sp.simplify(plain.gates.non_clifford - ctrl.gates.non_clifford) == 0


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


@pytest.mark.parametrize("a", [2, 4, 8])
@pytest.mark.parametrize("x", [0, 1, 2, 5, 7, 14, 15])
def test_modmul_const_cyclic_shift_is_correct(
    sdk_transpiler, a: int, x: int, tmp_path
) -> None:
    """Executing modmul on a cyclic-shift constant computes ``a*x mod 15``.

    Multiplication by a power of two modulo ``2**4 - 1 = 15`` is a cyclic bit
    rotation with an executable SWAP-network body, verified across every SDK
    backend for a spread of basis states.
    """
    n, modulus = 4, 15
    path = str(tmp_path / f"probe_{a}_{x}.py")
    probe = _load_probe(a, x, n, modulus, path)
    transpiler = sdk_transpiler.transpiler
    exe = transpiler.transpile(probe)
    measured = exe.sample(transpiler.executor(), shots=64).result().most_common(1)
    got = _value_of(measured[0][0])
    expected = (a * x) % modulus if x < modulus else x
    assert got == expected, (
        f"{sdk_transpiler.backend_name}: a={a} x={x} got {got}, expected {expected}"
    )


@qmc.qkernel
def _modmul_expval(obs: qmc.Observable) -> qmc.Float:
    """Estimate ``<obs>`` after multiplying ``|1>`` by 2 modulo 15 (n=4)."""
    reg = qmc.qubit_array(4, name="reg")
    reg[0] = qmc.x(reg[0])  # prepare |1>
    reg = modmul_const(reg, multiplier=2, modulus=15)  # -> |2> (rotate by one)
    return qmc.expval(reg, obs)


@pytest.mark.parametrize("qubit", [0, 1, 2, 3])
def test_modmul_const_cyclic_shift_expval_cross_backend(sdk_transpiler, qubit) -> None:
    """modmul_const's expectation-value path agrees across SDK backends.

    Exercises the estimator (expval) primitive, which regresses independently
    from sampling, on the concrete cyclic-shift modmul at n=4. ``|1> -> |2>``, a
    computational-basis state, so ``<Z_q>`` is analytically +/-1: qubit 1 (set in
    ``|2> = 0b0010``) gives -1, the others +1. Also cross-checked against a
    Qiskit statevector reference.
    """
    pytest.importorskip("qiskit")
    import qamomile.observable as qm_o
    from qamomile.qiskit import QiskitTranspiler

    observable = qm_o.Z(qubit)
    reference_transpiler = QiskitTranspiler()
    reference = (
        reference_transpiler.transpile(_modmul_expval, bindings={"obs": observable})
        .run(reference_transpiler.executor())
        .result()
    )
    analytic = -1.0 if qubit == 1 else 1.0
    assert np.isclose(reference, analytic, atol=1e-8)

    transpiler = sdk_transpiler.transpiler
    value = (
        transpiler.transpile(_modmul_expval, bindings={"obs": observable})
        .run(transpiler.executor())
        .result()
    )
    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(value, reference, atol=atol), (
        f"{sdk_transpiler.backend_name} qubit={qubit}: expected {reference}, got {value}"
    )
