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


def test_modmul_const_non_cyclic_raises_at_transpile(sdk_transpiler, tmp_path) -> None:
    """A non-rotation modular multiplication has no body and must not emit.

    ``multiplier=7 mod 15`` is not a cyclic bit rotation, so ``modmul_const``
    produces an estimation-only box with resource models but no executable body.
    Transpiling it to an executable circuit must raise rather than silently emit
    a barrier (which would run as identity and give a wrong quantum result).
    """
    from qamomile.circuit.transpiler.errors import EmitError

    path = str(tmp_path / "noncyclic.py")
    src = (
        "import qamomile.circuit as qmc\n"
        "from qamomile.circuit.stdlib.arithmetic import modmul_const\n"
        "@qmc.qkernel\n"
        "def noncyclic() -> qmc.Vector[qmc.Bit]:\n"
        "    reg = qmc.qubit_array(4, name='reg')\n"
        "    reg = modmul_const(reg, multiplier=7, modulus=15)\n"
        "    return qmc.measure(reg)\n"
    )
    with open(path, "w") as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location("noncyclic_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    transpiler = sdk_transpiler.transpiler
    with pytest.raises(EmitError, match="no executable body"):
        transpiler.transpile(module.noncyclic)


def test_modmul_const_non_cyclic_still_estimates() -> None:
    """A non-rotation modular multiplication still produces a resource estimate.

    Only *execution* is unavailable for general constants; the resource model
    path (which never reaches emit) must remain intact.
    """

    @qmc.qkernel
    def mul(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Multiply an n-bit register by a non-rotation constant once."""
        reg = qmc.qubit_array(n, name="reg")
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
    """A symbolic-width modmul reports ~0.15 n^2 non-Clifford via its model."""

    @qmc.qkernel
    def mul(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Multiply an n-bit register by a constant once."""
        reg = qmc.qubit_array(n, name="reg")
        reg = modmul_const(reg, multiplier=2, modulus=15)
        return reg

    est = mul.estimate_resources()
    n = sp.Symbol("n", integer=True, positive=True)
    # Leading non-Clifford term is 0.15 n^2 (calibrated modular multiplication).
    assert sp.limit(est.gates.non_clifford / n**2, n, sp.oo) == sp.Rational(15, 100)
    assert est.calls.calls_by_name.get("modmul_const") == 1


def test_modmul_const_abstract_mode_estimates_same_as_concrete() -> None:
    """Abstract modmul (no constant) reports the same cost as a concrete one."""

    @qmc.qkernel
    def abstract(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Apply an abstract constant modular multiplication once."""
        reg = qmc.qubit_array(n, name="reg")
        reg = modmul_const(reg)
        return reg

    est = abstract.estimate_resources()
    n = sp.Symbol("n", integer=True, positive=True)
    assert sp.limit(est.gates.non_clifford / n**2, n, sp.oo) == sp.Rational(15, 100)
    assert est.calls.calls_by_name.get("modmul_const") == 1


def test_modmul_const_control_coverage_assumption_matches_call() -> None:
    """The control-coverage assumption is honest about whether the call is controlled.

    An uncontrolled ``modmul_const`` must not claim its cost "includes the single
    control", while a controlled one must; the gate counts are identical either
    way (the control is lower-order against the O(n^2) leading term).
    """

    @qmc.qkernel
    def uncontrolled(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
        """Multiply an n-bit register by a constant, no control."""
        reg = qmc.qubit_array(n, name="reg")
        reg = modmul_const(reg, multiplier=2, modulus=15)
        return reg

    @qmc.qkernel
    def controlled(n: qmc.UInt) -> tuple[qmc.Qubit, qmc.Vector[qmc.Qubit]]:
        """Multiply an n-bit register by a constant, conditioned on a control."""
        ctrl = qmc.qubit("c")
        reg = qmc.qubit_array(n, name="reg")
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


def test_modmul_const_requires_both_or_neither_constant() -> None:
    """Supplying only one of multiplier/modulus is rejected."""

    @qmc.qkernel
    def half() -> qmc.Vector[qmc.Qubit]:
        """Attempt modmul with a multiplier but no modulus."""
        reg = qmc.qubit_array(4, name="reg")
        return modmul_const(reg, multiplier=2)

    with pytest.raises(ValueError, match="both given"):
        half.build()


def test_modmul_const_abstract_mode_is_estimation_only(sdk_transpiler) -> None:
    """Abstract modmul has no executable body and must not transpile."""
    from qamomile.circuit.transpiler.errors import EmitError

    @qmc.qkernel
    def abstract() -> qmc.Vector[qmc.Bit]:
        """Apply an abstract modular multiplication then measure."""
        reg = qmc.qubit_array(4, name="reg")
        reg = modmul_const(reg)
        return qmc.measure(reg)

    transpiler = sdk_transpiler.transpiler
    with pytest.raises(EmitError, match="no executable body"):
        transpiler.transpile(abstract)


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
