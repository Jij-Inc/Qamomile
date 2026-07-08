"""Tests for the Shor order-finding stdlib kernel and its resource estimate."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import sympy as sp

import qamomile.circuit as qmc
from qamomile.circuit.estimator import ResourcePolicy
from qamomile.circuit.estimator.physical import estimate_physical_resources
from qamomile.circuit.stdlib.shor import shor_order_finding


def test_shor_symbolic_estimate_matches_book_scaling() -> None:
    """The symbolic estimate reproduces the book's RSA order-finding scaling.

    Verifies ~3n algorithmic logical qubits, ~0.3 n^3 non-Clifford gates, and
    O(n) (== 2n) modular multiplications — the figures from Section 6.1.
    """
    est = shor_order_finding.estimate_resources()
    n = sp.Symbol("n", integer=True, positive=True)

    assert sp.simplify(est.qubits - 3 * n) == 0
    # Leading non-Clifford term is 0.3 n^3.
    assert sp.limit(est.gates.non_clifford / n**3, n, sp.oo) == sp.Rational(3, 10)
    # 2n modular multiplications, one per exponent bit (O(n) multiplications).
    assert sp.simplify(est.calls.calls_by_name["modmul_const"] - 2 * n) == 0


def test_shor_estimate_reproduces_rsa2048_numbers() -> None:
    """``substitutions={"n": 2048}`` yields the book figures in one call.

    The book numbers come from estimating symbolically and substituting the
    bit-width via ``substitutions`` — never by binding ``n=2048`` at build time,
    which would attempt to construct a 4096-qubit circuit.
    """
    est = shor_order_finding.estimate_resources(substitutions={"n": 2048})

    assert int(est.qubits) == 6144  # ~3n = ~6000 logical qubits
    non_clifford = float(est.gates.non_clifford)
    # Book: ~3e9 non-Clifford gates; worksheet: 2.577e9. Allow margin.
    assert 2.3e9 < non_clifford < 3.3e9


def test_shor_substitutions_rejects_unknown_and_overlap() -> None:
    """``substitutions`` fails fast on typos and on bindings overlap."""
    with pytest.raises(ValueError, match="neither free symbols"):
        shor_order_finding.estimate_resources(substitutions={"m": 10})
    with pytest.raises(ValueError, match="both bindings and substitutions"):
        shor_order_finding.estimate_resources(bindings={"n": 4}, substitutions={"n": 4})


def test_shor_physical_estimate_matches_worksheet() -> None:
    """Feeding the logical estimate into the surface-code model matches the sheet."""
    est = shor_order_finding.estimate_resources()
    phys = estimate_physical_resources(est)
    n = sp.Symbol("n", integer=True, positive=True)

    assert int(phys.code_distance.subs(n, 2048)) == 24
    physical_qubits = float(phys.physical_qubits.subs(n, 2048))
    assert abs(physical_qubits - 1.4156e7) / 1.4156e7 < 0.02
    assert abs(float(phys.runtime_hours.subs(n, 2048)) - 17.2) < 0.2


def test_shor_literature_and_asymptotic_policies_differ() -> None:
    """Literature gives the calibrated 0.3 n^3; asymptotic leaves the prefactor open.

    The default and ``LITERATURE`` policies select the calibrated modular
    multiplication model (0.3 n^3 non-Clifford), while ``ASYMPTOTIC`` selects the
    open-prefactor model whose leading term carries the free symbol
    ``c_modmul`` — confirming the policy machinery has an observable effect.
    """
    n = sp.Symbol("n", integer=True, positive=True)
    c_modmul = sp.Symbol("c_modmul", positive=True)

    literature = shor_order_finding.estimate_resources(policy=ResourcePolicy.LITERATURE)
    assert sp.limit(literature.gates.non_clifford / n**3, n, sp.oo) == sp.Rational(
        3, 10
    )

    asymptotic = shor_order_finding.estimate_resources(policy=ResourcePolicy.ASYMPTOTIC)
    assert c_modmul in asymptotic.gates.non_clifford.free_symbols
    # 2n multiplications x c_modmul n^2 -> leading 2 c_modmul n^3.
    assert sp.limit(asymptotic.gates.non_clifford / n**3, n, sp.oo) == 2 * c_modmul


def test_shor_estimate_surfaces_borrowed_scratch() -> None:
    """The order-finding estimate reports the O(n) modular-arithmetic scratch."""
    est = shor_order_finding.estimate_resources()
    n = sp.Symbol("n", integer=True, positive=True)

    # Peak width stays ~3n (borrowable scratch does not inflate it) ...
    assert sp.simplify(est.qubits - 3 * n) == 0
    # ... but the scratch requirement is carried as machine-readable data.
    assert sp.simplify(est.width.dirty_ancilla_qubits - n) == 0


def _build_true_schedule_shor(
    tmp_path, n_bits: int = 4, base: int = 2, modulus: int = 15
):
    """Generate a period-correct order-finding kernel with the true schedule.

    Unrolls the classically-precomputed ``base**(2**k) mod N`` multiplier
    schedule at a concrete width. For modulus ``2**n - 1`` and base 2 every such
    multiplier is a power of two, hence a cyclic bit rotation with an executable
    body, so the resulting circuit is a genuine (small) order finder.

    Args:
        tmp_path: pytest temporary directory for the generated source file.
        n_bits (int): Work-register width. Defaults to 4.
        base (int): Multiplication base. Defaults to 2.
        modulus (int): Modulus. Defaults to 15.

    Returns:
        QKernel: The period-correct order-finding kernel.
    """
    count_bits = 2 * n_bits
    lines = [
        "import qamomile.circuit as qmc",
        "from qamomile.circuit.stdlib.arithmetic import modmul_const",
        "@qmc.qkernel",
        "def true_shor() -> qmc.Vector[qmc.Bit]:",
        f"    counting = qmc.qubit_array({count_bits}, name='counting')",
        f"    work = qmc.qubit_array({n_bits}, name='work')",
        "    work[0] = qmc.x(work[0])",
        f"    for k in qmc.range({count_bits}):",
        "        counting[k] = qmc.h(counting[k])",
    ]
    for k in range(count_bits):
        multiplier = pow(base, 2**k, modulus)
        if multiplier == 1:
            continue  # multiply-by-1 is the identity; no operation needed
        lines.append(
            f"    counting[{k}], work = modmul_const(work, multiplier={multiplier}, "
            f"modulus={modulus}, control=counting[{k}])"
        )
    lines.append("    counting = qmc.iqft(counting)")
    lines.append("    return qmc.measure(counting)")
    path = str(tmp_path / "true_shor.py")
    with open(path, "w") as handle:
        handle.write("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location("true_shor_mod", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.true_shor


def test_shor_kernel_is_estimation_only(sdk_transpiler) -> None:
    """``shor_order_finding`` is estimation-only and refuses to transpile.

    Its modular multiplications are abstract (no concrete constant), so it has
    no executable body and must raise on every backend rather than emit a
    silently-wrong circuit. Genuine executable order finding is covered by
    :func:`test_shor_true_schedule_finds_period`, which uses concrete
    cyclic-shift constants.
    """
    from qamomile.circuit.transpiler.errors import EmitError

    transpiler = sdk_transpiler.transpiler
    with pytest.raises(EmitError, match="no executable body"):
        transpiler.transpile(shor_order_finding, bindings={"n": 4})


def test_shor_true_schedule_finds_period(sdk_transpiler, tmp_path) -> None:
    """A period-correct order finder recovers order r=4 for base 2 mod 15.

    Runs the true ``base**(2**k) mod N`` schedule at n=4 (8 counting qubits) and
    checks the four dominant phase outcomes are exactly the multiples of
    ``2**8 / r = 64`` — i.e. ``{0, 64, 128, 192}`` — which encode order r=4 (the
    multiplicative order of 2 modulo 15). A modular multiplication that silently
    executed as identity would leave only the ``0`` peak, so this also guards
    the emit path against the estimation-only box being dropped.
    """
    true_shor = _build_true_schedule_shor(tmp_path)
    transpiler = sdk_transpiler.transpiler
    exe = transpiler.transpile(true_shor)
    result = exe.sample(transpiler.executor(), shots=4096).result()

    def value(bits: tuple[int, ...]) -> int:
        return sum(bit * (2**i) for i, bit in enumerate(bits))

    counts: dict[int, int] = {}
    for bits, count in result.results:
        counts[value(bits)] = counts.get(value(bits), 0) + count

    targets = {0, 64, 128, 192}
    shots = result.shots
    # The four period-4 phase peaks together carry the overwhelming majority of
    # shots, and each is individually a substantial fraction (~25% ideal). Off-
    # peak leakage bins stay small. This is robust to shot noise while still
    # failing hard if a modular multiplication were dropped to identity (which
    # would collapse everything onto the single 0 peak).
    on_peak = sum(counts.get(v, 0) for v in targets)
    assert on_peak / shots > 0.7, (
        f"{sdk_transpiler.backend_name}: period-4 peaks {targets} carry only "
        f"{on_peak}/{shots}"
    )
    for v in targets:
        assert counts.get(v, 0) / shots > 0.1, (
            f"{sdk_transpiler.backend_name}: peak {v} too small "
            f"({counts.get(v, 0)}/{shots})"
        )
    off_peak = max(
        (c for v, c in counts.items() if v not in targets),
        default=0,
    )
    assert off_peak / shots < 0.1, (
        f"{sdk_transpiler.backend_name}: unexpected off-peak bin with "
        f"{off_peak}/{shots}"
    )


@qmc.qkernel
def _shor_expval(n: qmc.UInt, obs: qmc.Observable) -> qmc.Float:
    """Estimate ``<obs>`` on the Shor counting register (n-bit modulus)."""
    counting = qmc.qubit_array(2 * n, name="counting")
    work = qmc.qubit_array(n, name="work")
    work[0] = qmc.x(work[0])
    for k in qmc.range(2 * n):
        counting[k] = qmc.h(counting[k])
    for k in qmc.range(2 * n):
        counting[k], work = qmc.modmul_const(
            work, multiplier=2, modulus=15, control=counting[k]
        )
    counting = qmc.iqft(counting)
    return qmc.expval(counting, obs)


@pytest.mark.parametrize("qubit", [0, 1, 3])
def test_shor_cross_backend_expval(sdk_transpiler, qubit: int) -> None:
    """Shor's counting-register expectation value agrees across SDK backends.

    Exercises the estimator (expectation-value) primitive of each backend,
    which regresses independently from sampling, on a real controlled
    modular-multiplication circuit at n=4 (concrete cyclic-shift constants). The
    value is checked against a Qiskit statevector reference (a cross-backend
    equivalence check).
    """
    pytest.importorskip("qiskit")
    import qamomile.observable as qm_o
    from qamomile.qiskit import QiskitTranspiler

    observable = qm_o.Z(qubit)
    reference_transpiler = QiskitTranspiler()
    reference_exe = reference_transpiler.transpile(
        _shor_expval, bindings={"n": 4, "obs": observable}
    )
    reference = reference_exe.run(reference_transpiler.executor()).result()

    transpiler = sdk_transpiler.transpiler
    exe = transpiler.transpile(_shor_expval, bindings={"n": 4, "obs": observable})
    value = exe.run(transpiler.executor()).result()

    atol = 1e-6 if sdk_transpiler.backend_name == "cudaq" else 1e-8
    assert np.isclose(value, reference, atol=atol), (
        f"{sdk_transpiler.backend_name} qubit={qubit}: "
        f"expected {reference}, got {value}"
    )
