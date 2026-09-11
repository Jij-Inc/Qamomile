"""Tests for the Grover Adaptive Search converter (qamomile/optimization/gas.py).

Covers both the QUBO (build-in kernel) and HUBO (qkernel-factory) paths.

GAS is an oracle-based algorithm: ``GASConverter.get_cost_hamiltonian()``
raises ``NotImplementedError`` by design and the circuit terminates in
``qmc.measure(q_input)`` to read out the optimal bitstring. There is no
expectation-value semantics, so the cross-backend coverage required by
CLAUDE.md exercises the *sampling* primitive only — the expval path does
not apply to this algorithm.
"""

import numpy as np
import pytest

import qamomile.circuit as qmc
from qamomile.optimization.binary_model import BinaryModel, VarType
from qamomile.optimization.gas import GASConverter

# ---------------------------------------------------------------------------
# Backend transpiler factories (cross-backend execution matrix)
# ---------------------------------------------------------------------------


def _qiskit_transpiler():
    """Return a QiskitTranspiler, skipping the test if qiskit is unavailable.

    Returns:
        QiskitTranspiler: A fresh Qiskit backend transpiler.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit.transpiler import QiskitTranspiler

    return QiskitTranspiler()


def _quri_parts_transpiler():
    """Return a QuriPartsTranspiler, skipping the test if quri_parts is unavailable.

    Returns:
        QuriPartsTranspiler: A fresh QuriParts backend transpiler.
    """
    pytest.importorskip("quri_parts")
    from qamomile.quri_parts import QuriPartsTranspiler

    return QuriPartsTranspiler()


def _cudaq_transpiler():
    """Return a CudaqTranspiler, skipping the test if cudaq is unavailable.

    Returns:
        CudaqTranspiler: A fresh CUDA-Q backend transpiler.
    """
    pytest.importorskip("cudaq")
    from qamomile.cudaq import CudaqTranspiler

    return CudaqTranspiler()


# Each entry is a zero-arg factory that lazily imports + instantiates a
# transpiler, calling pytest.importorskip so the case skips (not errors)
# when the SDK is absent. Parametrizing over this list gives every
# execution test coverage on the full supported backend matrix.
_BACKENDS = [
    pytest.param(_qiskit_transpiler, id="qiskit"),
    pytest.param(_quri_parts_transpiler, id="quri_parts"),
    pytest.param(_cudaq_transpiler, id="cudaq"),
]


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


def _make_qubo_model() -> BinaryModel:
    """Build a small quadratic model used by GAS QUBO-path tests.

    Returns:
        BinaryModel: A purely quadratic (no higher-order) BINARY model.
    """
    return BinaryModel.from_hubo(
        {
            (0,): 1.0,
            (1,): 2.0,
            (0, 1): -1.0,
        }
    )


def _make_hubo_model() -> BinaryModel:
    """Build a small higher-order model used by GAS HUBO-path tests.

    Returns:
        BinaryModel: A BINARY model with a degree-3 term.
    """
    return BinaryModel.from_hubo(
        {
            (0,): 1.0,
            (1,): -1.0,
            (0, 1, 2): 1.0,
        }
    )


class _CaptureTranspiler:
    """Minimal transpiler stand-in that records the bindings instead of compiling."""

    def transpile(self, kernel, bindings=None):
        """Return the bindings instead of compiling.

        Args:
            kernel: Kernel that would be compiled (unused).
            bindings (dict | None): Bindings under test.

        Returns:
            dict | None: The bindings, verbatim.
        """
        return bindings


# ---------------------------------------------------------------------------
# Dispatch (backend-independent)
# ---------------------------------------------------------------------------


def test_transpile_dispatches_to_qubo_path(monkeypatch: pytest.MonkeyPatch):
    """Verify transpile dispatches to the quadratic implementation for QUBO."""
    model = _make_qubo_model()
    converter = GASConverter(model)
    calls = {"qubo": 0, "hubo": 0}

    def _fake_qubo(*args, **kwargs):
        """Record a call to the quadratic path instead of compiling.

        Args:
            *args: Positional arguments the real method would receive.
            **kwargs: Keyword arguments the real method would receive.

        Returns:
            str: The sentinel ``"qubo"``.
        """
        calls["qubo"] += 1
        return "qubo"

    def _fake_hubo(*args, **kwargs):
        """Record a call to the HUBO path instead of compiling.

        Args:
            *args: Positional arguments the real method would receive.
            **kwargs: Keyword arguments the real method would receive.

        Returns:
            str: The sentinel ``"hubo"``.
        """
        calls["hubo"] += 1
        return "hubo"

    monkeypatch.setattr(converter, "_transpile_quadratic", _fake_qubo)
    monkeypatch.setattr(converter, "_transpile_hubo", _fake_hubo)

    out = converter.transpile(object(), y=0, num_iterations=1, output_bits=3)

    assert out == "qubo"
    assert calls == {"qubo": 1, "hubo": 0}


def test_transpile_dispatches_to_hubo_path(monkeypatch: pytest.MonkeyPatch):
    """Verify transpile dispatches to the HUBO implementation when needed."""
    model = _make_hubo_model()
    converter = GASConverter(model)
    calls = {"qubo": 0, "hubo": 0}

    def _fake_qubo(*args, **kwargs):
        """Record a call to the quadratic path instead of compiling.

        Args:
            *args: Positional arguments the real method would receive.
            **kwargs: Keyword arguments the real method would receive.

        Returns:
            str: The sentinel ``"qubo"``.
        """
        calls["qubo"] += 1
        return "qubo"

    def _fake_hubo(*args, **kwargs):
        """Record a call to the HUBO path instead of compiling.

        Args:
            *args: Positional arguments the real method would receive.
            **kwargs: Keyword arguments the real method would receive.

        Returns:
            str: The sentinel ``"hubo"``.
        """
        calls["hubo"] += 1
        return "hubo"

    monkeypatch.setattr(converter, "_transpile_quadratic", _fake_qubo)
    monkeypatch.setattr(converter, "_transpile_hubo", _fake_hubo)

    out = converter.transpile(object(), y=0, num_iterations=1, output_bits=3)

    assert out == "hubo"
    assert calls == {"qubo": 0, "hubo": 1}


# ---------------------------------------------------------------------------
# Cross-backend smoke: QUBO and HUBO paths transpile and sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_qubo_transpile_and_sample_smoke(make_transpiler):
    """The QUBO GAS path transpiles and samples on each supported backend."""
    transpiler = make_transpiler()
    model = _make_qubo_model()
    converter = GASConverter(model)

    executable = converter.transpile(
        transpiler,
        output_bits=4,
        y=0,
        num_iterations=1,
    )
    result = executable.sample(transpiler.executor(), shots=16).result()

    assert len(result.results) > 0
    for sample, count in result.results:
        assert len(sample) == model.num_bits
        assert count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_hubo_transpile_and_sample_smoke(make_transpiler):
    """The HUBO GAS path transpiles and samples on each supported backend."""
    transpiler = make_transpiler()
    model = _make_hubo_model()
    converter = GASConverter(model)

    executable = converter.transpile(
        transpiler,
        output_bits=4,
        y=0,
        num_iterations=1,
    )
    result = executable.sample(transpiler.executor(), shots=16).result()

    assert len(result.results) > 0
    for sample, count in result.results:
        assert len(sample) == model.num_bits
        assert count > 0


# ---------------------------------------------------------------------------
# _required_output_bits (backend-independent)
# ---------------------------------------------------------------------------


def test_required_output_bits_qubo_models():
    """_required_output_bits computes the minimum sufficient register size for QUBO models."""
    # f = x0 + 2*x1 : f_max=3, f_min=0, half_range=3 -> floor(log2(3))+2 = 3
    conv = GASConverter(BinaryModel.from_hubo({(0,): 1.0, (1,): 2.0}))
    assert conv._required_output_bits(conv.binary_model) == 3

    # f = 4*x0 : f_max=4, f_min=0, half_range=4 -> floor(log2(4))+2 = 4
    conv4 = GASConverter(BinaryModel.from_hubo({(0,): 4.0}))
    assert conv4._required_output_bits(conv4.binary_model) == 4

    # mixed signs: f = x0 - x1 + x0*x1; coeffs=[1,-1,1]; f_max=1, f_min=-1, half_range=2 -> 3
    conv_mix = GASConverter(BinaryModel.from_hubo({(0,): 1.0, (1,): -1.0, (0, 1): 1.0}))
    assert conv_mix._required_output_bits(conv_mix.binary_model) == 3


def test_required_output_bits_hubo_only_model():
    """_required_output_bits accounts for higher-order terms in HUBO-only models."""
    # Only a degree-3 term with coefficient 8; range is [0, 8], requiring 5 bits:
    # floor(log2(8)) + 2 = 3 + 2 = 5.
    conv = GASConverter(BinaryModel.from_hubo({(0, 1, 2): 8.0}))
    assert conv._required_output_bits(conv.binary_model) == 5

    # Mixed: linear and higher-order; all positive coefficients 1+1+1=3 -> 3 bits.
    conv2 = GASConverter(BinaryModel.from_hubo({(0,): 1.0, (1,): 1.0, (0, 1, 2): 1.0}))
    assert conv2._required_output_bits(conv2.binary_model) == 3


def test_required_output_bits_accounts_for_the_threshold():
    """Sizing follows f(x) - y, not the bare objective span.

    f = 10 + x0 spans only [10, 11], but with y = 0 the register has to hold
    values up to 11, so three bits are not enough.
    """
    conv = GASConverter(BinaryModel.from_hubo({(0,): 1.0}, constant=10.0))

    # bound = max(|10 - 0|, |11 - 0|) = 11 -> floor(log2(11)) + 2 = 5
    assert conv._required_output_bits(conv.binary_model, 0.0) == 5
    # Centring the threshold inside the range shrinks the register again.
    assert conv._required_output_bits(conv.binary_model, 10.5) == 2
    # A threshold far below the range widens it.
    assert conv._required_output_bits(conv.binary_model, -100.0) == 8


def test_required_output_bits_public_wrapper_matches_transpile():
    """``required_output_bits`` reports exactly what ``transpile`` would pick."""
    conv = GASConverter(BinaryModel.from_hubo({(0,): 1.0, (1,): 2.0}))

    for y in (-4.0, 0.0, 1.0, 3.0):
        bindings = conv.transpile(_CaptureTranspiler(), y=y, num_iterations=1)
        assert bindings["m"] == conv.required_output_bits(y)


@pytest.mark.parametrize("output_bits", [0, -1, 1])
def test_transpile_rejects_non_positive_output_bits(output_bits: int):
    """Zero or negative widths cannot represent the objective and are rejected."""
    conv = GASConverter(_make_qubo_model())
    with pytest.raises(ValueError, match="too small to hold"):
        conv.transpile(
            _CaptureTranspiler(), output_bits=output_bits, y=0.0, num_iterations=1
        )


def test_transpile_rejects_insufficient_output_bits():
    """A width one bit short of the requirement is rejected, not silently wrapped.

    f = 2 - x0 - x1 - x0*x1 ranges over [-1, 2] and needs three bits at y = 0.
    Two bits wrap the encoding around and invert the sign the oracle tests, so
    the marked set would be wrong rather than merely imprecise.
    """
    model = BinaryModel.from_hubo({(0,): -1.0, (1,): -1.0, (0, 1): -1.0}, constant=2.0)
    conv = GASConverter(model)
    assert conv.required_output_bits(0.0) == 3

    with pytest.raises(ValueError, match="at least 3 qubits"):
        conv.transpile(_CaptureTranspiler(), output_bits=2, y=0.0, num_iterations=1)

    # The sufficient width is accepted.
    bindings = conv.transpile(
        _CaptureTranspiler(), output_bits=3, y=0.0, num_iterations=1
    )
    assert bindings["m"] == 3


@pytest.mark.parametrize("y", [-50.0, 50.0])
def test_transpile_widens_the_register_for_out_of_range_thresholds(y: float):
    """A threshold outside the objective range still gets a register that fits it."""
    model = BinaryModel.from_hubo({(0,): 1.0, (1,): 2.0})
    conv = GASConverter(model)

    bindings = conv.transpile(_CaptureTranspiler(), y=y, num_iterations=1)
    m = bindings["m"]
    f_min, f_max = conv._objective_bounds(conv.effective_model)
    assert -(2 ** (m - 1)) <= f_min - y
    assert f_max - y <= 2 ** (m - 1) - 1
    # An out-of-range threshold must cost more than a centred one.
    assert m > conv._required_output_bits(conv.effective_model, 1.5)


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
@pytest.mark.parametrize("num_iterations", [0, 1])
def test_single_variable_qubo_transpiles_and_samples(make_transpiler, num_iterations):
    """A one-variable QUBO transpiles and samples the analytic distribution.

    Regression for the diffusion operator: with n = 1 the X^n C^{n-1}Z X^n
    identity would need a controlled-Z with zero controls, so transpilation
    used to fail outright on a perfectly valid single-variable problem.

    f = 1 - 2*x0 gives f(1) = -1 < 0 = y and f(0) = 1, so exactly one of the
    two states is marked. With one marked state out of two, the Grover angle is
    arcsin(1/sqrt(2)) = pi/4 and the success probability after k iterations is
    sin^2((2k+1) * pi/4) = 1/2 for every k -- two states are too few to
    amplify. The point here is that the circuit builds and behaves as the
    theory says, not that it concentrates.
    """
    transpiler = make_transpiler()
    model = BinaryModel.from_hubo({(0,): -2.0}, constant=1.0)
    conv = GASConverter(model)

    shots = 4096
    exe = conv.transpile(
        transpiler, output_bits=3, y=0.0, num_iterations=num_iterations
    )
    result = exe.sample(transpiler.executor(), shots=shots).result()

    total = sum(count for _, count in result.results)
    assert total == shots
    for bits, _ in result.results:
        assert len(bits) == 1
    marked = sum(
        count for bits, count in result.results if tuple(int(b) for b in bits) == (1,)
    )
    # Three sigma of the binomial shot noise around the analytic 1/2.
    tolerance = 3.0 * (0.25 / shots) ** 0.5
    assert abs(marked / shots - 0.5) < tolerance, (
        f"Expected ~50% shots at (1,); got {marked}/{shots}"
    )


def test_post_init_keeps_all_zero_real_model_unchanged():
    """All-zero real model remains valid and does not trigger divide-by-zero."""
    model = BinaryModel.from_hubo({(0,): 0.0, (1,): 0.0}, constant=0.0)
    conv = GASConverter(model)

    assert conv.binary_model.constant == 0.0
    assert all(c == 0.0 for c in conv.binary_model.coefficients.values())


# ---------------------------------------------------------------------------
# _align_precision (backend-independent)
# ---------------------------------------------------------------------------


def test_align_precision_auto_detects_four_digits():
    """_align_precision infers 4 decimals from the finest meaningful value."""
    aligned = GASConverter._align_precision(
        [0.0, 0.1234, 2.1000000000000005, 0.30000000000000004]
    )
    assert aligned == [0.0, 0.1234, 2.1, 0.3]


def test_align_precision_auto_detects_finest_meaningful_digits():
    """_align_precision infers 4 decimals while suppressing floating-point tails."""
    aligned = GASConverter._align_precision(
        [0.0, 1.2345, 2.1000000000000005, 0.30000000000000004]
    )
    assert aligned == [0.0, 1.2345, 2.1, 0.3]


def test_align_precision_snaps_integer_like_float_noise():
    """_align_precision snaps near-integer float artifacts to exact integers."""
    aligned = GASConverter._align_precision(
        [4.000000000000005, -0.9999999999999998, -1.0000000000000002]
    )
    assert aligned == [4.0, -1.0, -1.0]


def test_align_precision_preserves_large_fractional_values():
    """A large fractional coefficient survives alignment untouched.

    ``np.isclose``'s default relative tolerance scales with the compared
    magnitude: at 1e5 it accepts a difference of about 1.0, so ``100000.4``
    would compare equal to ``100000`` and be snapped to an integer. Every
    comparison in ``_align_precision`` is therefore absolute-only.
    """
    aligned = GASConverter._align_precision([100000.4, 2.5, 3.0])
    assert aligned == [100000.4, 2.5, 3.0]


def test_large_fractional_coefficient_survives_construction():
    """The converter stores a large fractional coefficient as written.

    Regression for the same tolerance bug reaching the model: the coefficient
    must still be non-integer after ``__post_init__`` so that ``transpile()``
    recognizes it and warns before quantizing.
    """
    model = BinaryModel.from_hubo({(0,): 100000.4, (1,): 2.0})
    conv = GASConverter(model)

    assert np.isclose(
        conv.binary_model.coefficients[(0,)], 100000.4, atol=1e-9, rtol=0.0
    )

    with pytest.warns(UserWarning, match="non-integer coefficients"):
        conv.transpile(_CaptureTranspiler(), y=0.0, num_iterations=1)
    assert conv.quantization_scale > 1.0


# ---------------------------------------------------------------------------
# BINARY-domain intake (backend-independent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [3, 4, 5, 6])
def test_high_degree_binary_model_is_not_round_tripped(degree: int):
    """A high-degree BINARY model reaches the circuit with its own coefficients.

    Converting BINARY to SPIN and back expands a degree-``d`` monomial into
    ``2**d`` terms in each direction and leaves float noise behind. The
    converter now takes the BINARY model the base class already normalized, so
    the coefficients must come through bit-for-bit and the variable count must
    still agree with ``spin_model`` (which decoding length-checks against).
    """
    coeffs = {tuple(range(degree)): 1.0, (0,): -2.0, (0, 1): 3.0}
    model = BinaryModel.from_hubo(coeffs, constant=5.0)
    conv = GASConverter(model)

    assert conv.binary_model.constant == 5.0
    assert conv.binary_model.coefficients == model.coefficients
    assert conv.binary_model.num_bits == conv.spin_model.num_bits


def test_spin_input_is_converted_to_binary_once():
    """A genuine SPIN input is still converted into the BINARY domain."""
    spin = BinaryModel.from_ising({0: 1.0}, {(0, 1): -1.0}, constant=0.5)
    conv = GASConverter(spin)

    assert conv.binary_model.vartype is VarType.BINARY
    for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        spins = [1 - 2 * b for b in bits]
        assert np.isclose(
            conv.binary_model.calc_energy(list(bits)),
            spin.calc_energy(spins),
            atol=1e-9,
            rtol=0.0,
        )


# ---------------------------------------------------------------------------
# _detect_eps (backend-independent)
# ---------------------------------------------------------------------------


def test_detect_eps_integer_floats():
    """_detect_eps returns 0.1 for integer-valued floats (str repr always has '.0')."""
    # str(1.0)='1.0', str(2.0)='2.0' → exp=-1 → 10^-1=0.1 for each
    eps = GASConverter._detect_eps([1.0, 2.0, 3.0])
    assert np.isclose(eps, 0.1, atol=1e-12, rtol=0.0)


def test_detect_eps_one_decimal_place():
    """_detect_eps returns 0.1 for values with exactly one decimal place."""
    eps = GASConverter._detect_eps([0.3, 0.7])
    assert np.isclose(eps, 0.1, atol=1e-12, rtol=0.0)


def test_detect_eps_two_decimal_places():
    """_detect_eps returns 0.01 for values with two decimal places."""
    eps = GASConverter._detect_eps([0.25, 0.50])
    assert np.isclose(eps, 0.01, atol=1e-12, rtol=0.0)


def test_detect_eps_uses_finest_precision():
    """_detect_eps returns the minimum (finest) epsilon across all input values."""
    # 0.3 → 0.1; 0.25 → 0.01 → min = 0.01
    eps = GASConverter._detect_eps([0.3, 0.25])
    assert np.isclose(eps, 0.01, atol=1e-12, rtol=0.0)


def test_detect_eps_irrational_fraction_is_tiny():
    """_detect_eps returns a very small epsilon for full-precision float fractions."""
    # str(1/3) has 16+ significant decimal digits
    eps = GASConverter._detect_eps([1 / 3])
    assert eps <= 1e-10


def test_detect_eps_empty_list():
    """_detect_eps returns 1.0 when given an empty list."""
    eps = GASConverter._detect_eps([])
    assert np.isclose(eps, 1.0, atol=1e-12, rtol=0.0)


# ---------------------------------------------------------------------------
# _greedy_quantization_parameter (backend-independent)
# ---------------------------------------------------------------------------


def test_greedy_quantization_parameter_returns_int():
    """_greedy_quantization_parameter always returns a plain int."""
    model = BinaryModel.from_hubo({(0,): 1.0, (1,): 2.0})
    result = GASConverter._greedy_quantization_parameter(model)
    assert isinstance(result, int)


def test_greedy_quantization_parameter_minimum_safety_bits():
    """_greedy_quantization_parameter returns at least the safety offset (2 bits)."""
    # Even a trivial single-term model must have the 2-bit safety margin.
    model = BinaryModel.from_hubo({(0,): 1.0})
    result = GASConverter._greedy_quantization_parameter(model)
    assert result >= 2


def test_greedy_quantization_parameter_known_value():
    """_greedy_quantization_parameter matches the hand-computed value for a simple model.

    For BinaryModel.from_hubo({(0,): 2.0}):
      coeffs = [0.0, 2.0], max_coeff=2.0
      eps = _detect_eps([0.0, 2.0]) = 0.1  (both str-repr have one decimal)
      p = ceil(log2(2.0/0.1)) = ceil(log2(20)) = 5
      N_terms=2  →  log_N = ceil(log2(2)) = 1
      result = p + log_N + safety = 5 + 1 + 2 = 8
    """
    model = BinaryModel.from_hubo({(0,): 2.0})
    result = GASConverter._greedy_quantization_parameter(model)
    assert result == 8


def test_greedy_quantization_parameter_increases_with_precision():
    """A finer-precision coefficient alongside a large one raises the parameter.

    m1 = {(0,): 3.0}:            max=3.0, eps=0.1,  ratio=30,  p=5
    m2 = {(0,): 3.0, (1,): 0.01}: max=3.0, eps=0.01, ratio=300, p=9

    Adding a more precise coefficient tightens eps and forces a higher p.
    """
    m1 = BinaryModel.from_hubo({(0,): 3.0})
    m2 = BinaryModel.from_hubo({(0,): 3.0, (1,): 0.01})
    q1 = GASConverter._greedy_quantization_parameter(m1)
    q2 = GASConverter._greedy_quantization_parameter(m2)
    assert q2 > q1


def test_greedy_quantization_parameter_more_terms_increases_log_n():
    """Adding more terms increases log_N, raising the quantization parameter."""
    m_small = BinaryModel.from_hubo({(0,): 1.0})
    m_large = BinaryModel.from_hubo(
        {
            (0,): 1.0,
            (1,): 1.0,
            (2,): 1.0,
            (3,): 1.0,
            (4,): 1.0,
            (5,): 1.0,
            (6,): 1.0,
            (7,): 1.0,
        }
    )
    q_small = GASConverter._greedy_quantization_parameter(m_small)
    q_large = GASConverter._greedy_quantization_parameter(m_large)
    assert q_large >= q_small


# ---------------------------------------------------------------------------
# approximate_real_valued_model (backend-independent)
# ---------------------------------------------------------------------------


def _all_integer(model: BinaryModel) -> bool:
    """Return True when every coefficient in ``model`` is integer-valued.

    Args:
        model (BinaryModel): Model whose constant and interaction coefficients
            are inspected.

    Returns:
        bool: ``True`` when every value is an integer within ``atol=1e-12``.
    """
    vals = [model.constant] + list(model.coefficients.values())
    return all(np.isclose(v, round(v), atol=1e-12, rtol=0.0) for v in vals)


def test_approximate_real_valued_model_fractional_constant():
    """A fractional constant alone is rounded to the nearest integer."""
    model = BinaryModel.from_hubo({(0,): 1.0}, constant=0.5)
    approx, _ = GASConverter.approximate_real_valued_model(model)
    assert _all_integer(approx)


def test_approximate_real_valued_model_fractional_coefficients():
    """Real-valued coefficients are all mapped to integers."""
    model = BinaryModel.from_hubo({(0,): 0.3, (1,): -1.2, (0, 1): 0.25}, constant=0.7)
    approx, _ = GASConverter.approximate_real_valued_model(model)
    assert _all_integer(approx)


def test_approximate_real_valued_model_zero_model_unchanged():
    """All-zero model is returned as-is without raising divide-by-zero."""
    model = BinaryModel.from_hubo({(0,): 0.0, (1,): 0.0}, constant=0.0)
    approx, scale = GASConverter.approximate_real_valued_model(model)
    assert np.isclose(approx.constant, 0.0, atol=1e-12, rtol=0.0)
    assert all(
        np.isclose(c, 0.0, atol=1e-12, rtol=0.0) for c in approx.coefficients.values()
    )
    # Returned unchanged, so the model is already in the caller's scale.
    assert np.isclose(scale, 1.0, atol=1e-12, rtol=0.0)


def test_approximate_real_valued_model_preserves_signs():
    """Positive and negative coefficients retain their signs after approximation."""
    model = BinaryModel.from_hubo({(0,): 1.0, (1,): -2.0}, constant=0.0)
    approx, _ = GASConverter.approximate_real_valued_model(
        model, quantization_parameter=4
    )
    assert approx.coefficients[(0,)] > 0
    assert approx.coefficients[(1,)] < 0


def test_approximate_real_valued_model_explicit_quantization_parameter():
    """Explicit quantization_parameter controls the integer scale factor.

    With quantization_parameter=4 and max_coeff=4.0:
      round(4.0/4.0 * 2^3) = 8   (coefficient (0,))
      round(2.0/4.0 * 2^3) = 4   (coefficient (1,))
    and the reported scale is 2^3 / 4.0 = 2.0.
    """
    model = BinaryModel.from_hubo({(0,): 4.0, (1,): 2.0}, constant=0.0)
    approx, scale = GASConverter.approximate_real_valued_model(
        model, quantization_parameter=4
    )
    assert np.isclose(approx.coefficients[(0,)], 8.0, atol=1e-12, rtol=0.0)
    assert np.isclose(approx.coefficients[(1,)], 4.0, atol=1e-12, rtol=0.0)
    assert np.isclose(scale, 2.0, atol=1e-12, rtol=0.0)


def test_approximate_real_valued_model_auto_vs_explicit_both_integer():
    """Auto (None) and explicit quantization_parameter both produce integer models."""
    model = BinaryModel.from_hubo({(0,): 0.3, (1,): 0.7}, constant=0.1)
    approx_auto, _ = GASConverter.approximate_real_valued_model(
        model, quantization_parameter=None
    )
    approx_explicit, _ = GASConverter.approximate_real_valued_model(
        model, quantization_parameter=8
    )
    assert _all_integer(approx_auto)
    assert _all_integer(approx_explicit)


def test_approximate_real_valued_model_auto_quantization_stays_float_exact_int():
    """Auto quantization keeps integer-valued coefficients exactly representable in float.

    Repeating-decimal coefficients can otherwise push the greedy precision very high,
    which may exceed exact integer representation in float-backed storage.
    """
    model = BinaryModel.from_hubo({(0,): 1 / 3, (1,): -2 / 3}, constant=0.1)
    approx, _ = GASConverter.approximate_real_valued_model(model)

    vals = [float(approx.constant)] + [float(v) for v in approx.coefficients.values()]
    assert all(v.is_integer() for v in vals)
    assert all(abs(v) <= 2**53 for v in vals)


# ---------------------------------------------------------------------------
# _make_term_encoding factory (cross-backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_make_term_encoding_degree0_transpiles_and_samples(make_transpiler):
    """_make_term_encoding([]) (degree-0) produces a transpilable constant-phase encoder."""
    transpiler = make_transpiler()
    conv = GASConverter(_make_hubo_model())
    encoder = conv._make_term_encoding([])  # degree = 0

    @qmc.qkernel
    def wrap_const(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Apply the degree-0 encoder on fresh registers and measure the input.

        Args:
            n (qmc.UInt): Number of input (decision-variable) qubits.
            m (qmc.UInt): Number of output (objective-value) qubits.

        Returns:
            qmc.Vector[qmc.Bit]: Measurement outcomes of the input register.
        """
        q_output = qmc.qubit_array(m, name="q_output")
        q_input = qmc.qubit_array(n, name="q_input")
        q_output, q_input = encoder(q_output, q_input, 0.0)
        return qmc.measure(q_input)

    exe = transpiler.transpile(wrap_const, bindings={"n": 2, "m": 3})
    results = exe.sample(transpiler.executor(), shots=16).result().results

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == 2 and count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
@pytest.mark.parametrize(
    "ctrl_indices,n_input",
    [
        ([0], 2),
        ([0, 1], 3),
        ([0, 1, 2], 4),
    ],
    ids=["degree-1", "degree-2", "degree-3"],
)
def test_make_term_encoding_higher_degrees_transpile(
    make_transpiler, ctrl_indices: list, n_input: int
):
    """_make_term_encoding for degrees 1/2/3 transpiles and produces valid samples."""
    transpiler = make_transpiler()
    conv = GASConverter(BinaryModel.from_hubo({(0, 1, 2): 1.0, (0,): 0.5}))
    encoder = conv._make_term_encoding(ctrl_indices)

    @qmc.qkernel
    def wrap_term(n: qmc.UInt, m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
        """Apply a higher-degree encoder on fresh registers and measure the input.

        Args:
            n (qmc.UInt): Number of input (decision-variable) qubits.
            m (qmc.UInt): Number of output (objective-value) qubits.

        Returns:
            qmc.Vector[qmc.Bit]: Measurement outcomes of the input register.
        """
        q_output = qmc.qubit_array(m, name="q_output")
        q_input = qmc.qubit_array(n, name="q_input")
        q_output, q_input = encoder(q_output, q_input, 1.0)
        return qmc.measure(q_input)

    exe = transpiler.transpile(wrap_term, bindings={"n": n_input, "m": 3})
    results = exe.sample(transpiler.executor(), shots=16).result().results

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == n_input and count > 0


# ---------------------------------------------------------------------------
# HUBO state-preparation correctness (cross-backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_hubo_prep_dagger_restores_state(make_transpiler):
    """Applying HUBO prep then its dagger returns the input register to |0...0>.

    This is the HUBO analogue of test_apply_then_dagger_restores_input_register
    in tests/circuit/algorithm/test_gas.py. A perfect inversion guarantees that
    the forward and dagger kernels are exact conjugates of each other, which is
    a load-bearing invariant for the Grover operator.
    """
    transpiler = make_transpiler()
    model = _make_hubo_model()
    conv = GASConverter(model)
    output_bits = 4

    forward = conv._make_apply_function_preparation_hubo(output_bits=output_bits)
    dagger = conv._make_apply_function_preparation_hubo_dagger(output_bits=output_bits)

    @qmc.qkernel
    def wrap_prep_then_dagger(
        n: qmc.UInt, m: qmc.UInt, y: qmc.Float
    ) -> tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]:
        """Apply forward preparation then its dagger and measure both registers.

        Args:
            n (qmc.UInt): Number of input (decision-variable) qubits.
            m (qmc.UInt): Number of output (objective-value) qubits.
            y (qmc.Float): Threshold offset handed to both kernels.

        Returns:
            tuple[qmc.Vector[qmc.Bit], qmc.Vector[qmc.Bit]]: Measurement
                outcomes of the output and input registers.
        """
        q_output = qmc.qubit_array(m, name="q_output")
        q_input = qmc.qubit_array(n, name="q_input")
        q_output, q_input = forward(q_output, q_input, y)
        q_output, q_input = dagger(q_output, q_input, y)
        return qmc.measure(q_output), qmc.measure(q_input)

    exe = transpiler.transpile(
        wrap_prep_then_dagger,
        # y must be Float: both prep kernels declare `y: qmc.Float`.
        bindings={"n": model.num_bits, "m": output_bits, "y": 0.0},
    )
    results = exe.sample(transpiler.executor(), shots=32).result().results

    # A^dagger A = I, so *both* registers are deterministically |0...0>.
    assert len(results) == 1, f"Expected only |0...0>, got {results}"
    (((output_bits_measured, input_bits), count),) = results
    assert len(output_bits_measured) == output_bits
    assert len(input_bits) == model.num_bits
    assert all(b == 0 for b in output_bits_measured), (
        f"Output register not restored: {output_bits_measured}"
    )
    assert all(b == 0 for b in input_bits), f"Input register not restored: {input_bits}"
    assert count == 32


# ---------------------------------------------------------------------------
# Edge cases (cross-backend)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_hubo_multiple_iterations_transpile_and_sample(make_transpiler):
    """HUBO GAS with num_iterations=2 transpiles and produces valid samples."""
    transpiler = make_transpiler()
    model = _make_hubo_model()
    conv = GASConverter(model)

    exe = conv.transpile(transpiler, output_bits=4, y=0, num_iterations=2)
    results = exe.sample(transpiler.executor(), shots=16).result().results

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == model.num_bits and count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_hubo_only_higher_order_terms(make_transpiler):
    """HUBO path handles a model consisting entirely of degree-3 terms."""
    transpiler = make_transpiler()
    # Degree-3 term only; no linear or quadratic part.
    model = BinaryModel.from_hubo({(0, 1, 2): 1.0})
    conv = GASConverter(model)

    exe = conv.transpile(transpiler, output_bits=4, y=0, num_iterations=1)
    results = exe.sample(transpiler.executor(), shots=16).result().results

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == model.num_bits and count > 0


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_hubo_degree4_term_transpiles_and_samples(make_transpiler):
    """HUBO path handles a degree-4 higher-order term alongside a linear term."""
    transpiler = make_transpiler()
    model = BinaryModel.from_hubo({(0, 1, 2, 3): 1.0, (0,): -0.5})
    conv = GASConverter(model)

    # Encode the real coefficients directly: quantizing them would rescale the
    # objective by 2**(q-1) and demand a register far too wide to simulate,
    # which is beside the point for this structural check.
    exe = conv.transpile(
        transpiler,
        output_bits=5,
        y=0,
        num_iterations=1,
        approximate_real_coefficients=False,
    )
    results = exe.sample(transpiler.executor(), shots=16).result().results

    assert len(results) > 0
    for bits, count in results:
        assert len(bits) == model.num_bits and count > 0


# ---------------------------------------------------------------------------
# Randomized validity (cross-backend, seeded)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
@pytest.mark.parametrize("seed", [0, 1, 2, 42])
def test_hubo_random_coefficients_sample_valid_bitstrings(make_transpiler, seed):
    """Randomized HUBO models with a degree-3 term sample valid input bitstrings.

    Exercises the qkernel-factory path with random (positive and negative)
    coefficients on every backend, asserting only structural validity since
    the per-shot distribution is not fixed for arbitrary coefficients.
    """
    transpiler = make_transpiler()
    rng = np.random.default_rng(seed)

    # 3 variables: random linear/quad coefficients plus a random degree-3 term.
    coeffs = {
        (0,): float(rng.uniform(-2.0, 2.0)),
        (1,): float(rng.uniform(-2.0, 2.0)),
        (2,): float(rng.uniform(-2.0, 2.0)),
        (0, 1): float(rng.uniform(-2.0, 2.0)),
        (0, 1, 2): float(rng.uniform(-2.0, 2.0)),
    }
    model = BinaryModel.from_hubo(coeffs)
    conv = GASConverter(model)

    # As above: the raw real coefficients keep the register narrow enough to
    # simulate, and this test only asserts structural validity.
    exe = conv.transpile(
        transpiler,
        output_bits=4,
        y=0,
        num_iterations=1,
        approximate_real_coefficients=False,
    )
    results = exe.sample(transpiler.executor(), shots=32).result().results

    assert len(results) > 0
    seen = 0
    for bits, count in results:
        assert len(bits) == model.num_bits
        assert all(b in (0, 1) for b in bits)
        assert count > 0
        seen += count
    assert seen == 32


# ---------------------------------------------------------------------------
# Statistical correctness: Grover amplifies the unique optimal state
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_qubo_grover_finds_unique_optimal_state(make_transpiler):
    """QUBO GAS reaches P=1 for the unique optimal state after 1 Grover iteration.

    Model: f(x0,x1) = 2 - x0 - x1 - x0*x1
      f(0,0) = 2,  f(1,0) = 1,  f(0,1) = 1,  f(1,1) = -1  <- unique minimum

    With y=0 the oracle marks only (1,1) (f=-1 < 0); y_circuit = 2-0 = 2 > 0.
    For 1 marked state out of 4: k* = floor(pi/4 * sqrt(4/1)) = 1 iteration
    gives P = sin^2(3 * arcsin(1/2)) = sin^2(pi/2) = 1.0 exactly.
    All 64 shots must therefore be (1,1) on a noise-free simulator.

    This test also exercises the QUBO diffusion operator's linear-type
    correctness: a dropped controlled-Z gate would break amplitude
    amplification and scatter shots across all 4 states.
    """
    transpiler = make_transpiler()
    model = BinaryModel.from_hubo(
        {(0,): -1.0, (1,): -1.0, (0, 1): -1.0},
        constant=2.0,
    )
    conv = GASConverter(model)

    exe = conv.transpile(transpiler, output_bits=4, y=0, num_iterations=1)
    result = exe.sample(transpiler.executor(), shots=64).result()

    optimal_bits = (1, 1)
    total_shots = sum(count for _, count in result.results)
    optimal_count = sum(
        count
        for bits, count in result.results
        if tuple(int(b) for b in bits) == optimal_bits
    )
    # P = 1.0 exactly in theory; allow a small tolerance for backend gate
    # decomposition and float rounding in phase angles.
    assert optimal_count / total_shots >= 0.99, (
        f"Expected >= 99% shots at (1,1); got {optimal_count}/{total_shots}"
    )


@pytest.mark.parametrize("make_transpiler", _BACKENDS)
def test_hubo_grover_biases_toward_optimal_state(make_transpiler):
    """HUBO GAS amplifies the unique optimal state to near-certainty.

    Model: f(x0,x1,x2) = 5 - x0 - x1 - x2 - x0*x1*x2
      f(1,1,1) = 1   <- unique minimum
      f(1,1,0) = 3
      f(0,0,0) = 5

    With y=3, the oracle marks only (1,1,1) (since f=1 < 3); y_circuit = 5-3 = 2 > 0.
    Optimal Grover iterations for 1 marked / 8 total: floor(pi/4 * sqrt(8)) = 2.
    After 2 iterations: P(optimal) = sin^2(5 * arcsin(1/(2*sqrt(2)))) ~= 0.945,
    so >= 70% guards against regressions while tolerating shot noise.
    """
    transpiler = make_transpiler()
    model = BinaryModel.from_hubo(
        {(0,): -1.0, (1,): -1.0, (2,): -1.0, (0, 1, 2): -1.0},
        constant=5.0,
    )
    conv = GASConverter(model)

    exe = conv.transpile(transpiler, output_bits=4, y=3, num_iterations=2)
    result = exe.sample(transpiler.executor(), shots=128).result()

    optimal_bits = (1, 1, 1)
    total_shots = sum(count for _, count in result.results)
    optimal_count = sum(
        count
        for bits, count in result.results
        if tuple(int(b) for b in bits) == optimal_bits
    )
    assert optimal_count / total_shots >= 0.70, (
        f"Expected >= 70% optimal samples; got {optimal_count}/{total_shots}"
    )


# ---------------------------------------------------------------------------
# Negative tests (backend-independent)
# ---------------------------------------------------------------------------


def test_get_cost_hamiltonian_raises_not_implemented():
    """GAS is oracle-based, so requesting a cost Hamiltonian must fail loudly."""
    conv = GASConverter(_make_qubo_model())
    with pytest.raises(NotImplementedError, match="does not expose a cost Hamiltonian"):
        conv.get_cost_hamiltonian()


def test_compose_encoders_baked_rejects_length_mismatch():
    """Mismatched encoder/coefficient lists raise instead of silently truncating."""
    conv = GASConverter(_make_hubo_model())
    encoders = [conv._make_term_encoding([0]), conv._make_term_encoding([1])]
    with pytest.raises(ValueError, match="must have the same length"):
        conv._compose_encoders_baked(encoders, [1.0])


# ---------------------------------------------------------------------------
# Converter state invariants (backend-independent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("make_model", [_make_qubo_model, _make_hubo_model])
def test_effective_model_available_before_transpile(make_model):
    """effective_model/quantization_scale are set at construction, not by transpile().

    The HUBO kernel factories read `effective_model`, so a freshly built
    converter must already expose it; otherwise those factories raise
    AttributeError when used standalone.
    """
    conv = GASConverter(make_model())
    assert conv.effective_model is conv.binary_model
    assert np.isclose(conv.quantization_scale, 1.0, atol=1e-12, rtol=0.0)
    # The HUBO factories must be usable without transpile() having run.
    assert conv._make_apply_function_preparation_hubo(output_bits=3) is not None
    assert conv._make_apply_function_preparation_hubo_dagger(output_bits=3) is not None


def test_integer_model_keeps_unit_quantization_scale():
    """An already-integer model is not quantized, so the scale stays exactly 1."""
    conv = GASConverter(_make_qubo_model())

    bindings = conv.transpile(_CaptureTranspiler(), y=2.0, num_iterations=1)
    assert np.isclose(conv.quantization_scale, 1.0, atol=1e-12, rtol=0.0)
    # y_circuit = constant - y * scale = 0.0 - 2.0 * 1.0
    assert np.isclose(bindings["y"], -2.0, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("y", [-2.0, -1.0, -0.5, 0.0, 0.5])
def test_quantized_threshold_is_scaled_into_the_effective_domain(y):
    """The oracle threshold tracks the coefficient quantization scale.

    The circuit register holds ``y_circuit + (f_eff(x) - eff.constant)``. With
    coefficients scaled by ``s``, that equals ``s * (f(x) - y)`` only if the
    threshold is scaled too. Since ``s > 0``, the oracle's sign test then agrees
    with ``f(x) < y`` for every x -- which is what this asserts. Leaving y
    un-scaled makes the comparison mix the original and quantized domains and
    mark the wrong states for any nonzero y.
    """
    # Non-integer coefficients, so the quantization path is taken.
    model = BinaryModel.from_hubo({(0,): 0.5, (1,): -1.5, (0, 1): 0.25}, constant=0.0)

    def f_orig(bits):
        """Evaluate the original objective on a bit tuple.

        Args:
            bits (tuple[int, ...]): Assignment of the two decision variables.

        Returns:
            float: Objective value in the original scale.
        """
        return 0.5 * bits[0] - 1.5 * bits[1] + 0.25 * bits[0] * bits[1]

    conv = GASConverter(model)
    with pytest.warns(UserWarning, match="non-integer coefficients"):
        bindings = conv.transpile(_CaptureTranspiler(), y=y, num_iterations=1)

    eff = conv.effective_model
    scale = conv.quantization_scale
    assert scale > 1.0, "expected the non-integer model to be rescaled"

    for bits in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        f_eff = (
            eff.constant
            + sum(eff.linear.get(i, 0.0) * bits[i] for i in range(2))
            + sum(c * np.prod([bits[i] for i in key]) for key, c in eff.quad.items())
        )
        register = bindings["y"] + (f_eff - eff.constant)
        assert (register < 0) == (f_orig(bits) < y), (
            f"oracle disagrees with f(x) < y at x={bits}, y={y}: "
            f"register={register}, f(x)={f_orig(bits)}, scale={scale}"
        )
