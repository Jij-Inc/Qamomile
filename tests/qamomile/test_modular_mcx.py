"""Checked modular controls survive native planning and CUDA-Q serialization."""

from __future__ import annotations

import runpy

import numpy as np
import pytest

from qamomile._native import compiler as native
from qamomile.circuit import CircuitInstruction


def test_native_modular_plan_retains_controls_only_when_accepted() -> None:
    """Retained MCX saves chain scratch without changing the selected arithmetic."""
    options = ["mod_mul.schoolbook_ripple"]
    fallback = native.plan_mod_pow(2, 1, 2, 3, False, 2, [], options)
    retained = native.plan_mod_pow(2, 1, 2, 3, False, 2, ["mcx"], options)
    assert fallback[:2] == retained[:2]
    assert fallback[4] == retained[4]
    assert fallback[2] > retained[2]
    assert len(fallback[3]) > len(retained[3])
    assert all(not controls for _, _, controls in fallback[3])
    assert any(gate == "x" and len(controls) >= 3 for gate, _, controls in retained[3])
    for basis in range(32):
        target = basis & 3
        exponent = (basis >> 2) & 1
        expected_target = (
            2 * target % 3 if basis >> 3 == 3 and exponent and target < 3 else target
        )
        expected = (basis & ~3) | expected_target
        for plan in (fallback, retained):
            state = basis
            for gate, wires, controls in plan[3]:
                intrinsic = wires[:-1]
                assert gate in {"x", "cx", "ccx"}
                if all(state & (1 << control) for control in [*controls, *intrinsic]):
                    state ^= 1 << wires[-1]
            assert (
                state == expected
            )  # Includes clean final scratch and unchanged controls.


def test_native_direct_mod_mul_selects_shared_schoolbook_only_for_override() -> None:
    """HUGR's bridge preserves the portable default and honors schoolbook options."""
    assert native.plan_mod_mul(2, 2, 3, False, 0) is None
    plan = native.plan_mod_mul(2, 2, 3, False, 0, ["mod_mul.schoolbook_ripple"])
    assert plan is not None
    assert plan[:2] == ("gates", "mod_mul.schoolbook_ripple")
    assert plan[2] > 0 and plan[3]
    assert all(not controls for _, _, controls in plan[3])
    with pytest.raises(ValueError, match="Q"):
        native.plan_mod_mul(2, 0, 3, False, 0, ["mod_mul.schoolbook_ripple"])


@pytest.mark.cudaq
@pytest.mark.parametrize("control_count", [3, 4, 5])
def test_cudaq_mcx_emission_preserves_basis_and_complex_state_after_roundtrip(
    tmp_path, control_count: int
) -> None:
    """The advertised SDK operation is phase exact and keeps every control."""
    cudaq = pytest.importorskip("cudaq")
    from qamomile.compilation import RealizationCapability
    from qamomile.cudaq import Cudaq, _integration
    from qamomile.cudaq._emission import _emit_controlled_instruction

    target = Cudaq()
    assert RealizationCapability.MCX in target.capabilities.realization_capabilities
    fingerprint = target.fingerprint
    previous_capabilities = _integration._REALIZATIONS
    try:
        _integration._REALIZATIONS = previous_capabilities - {RealizationCapability.MCX}
        assert target.fingerprint != fingerprint
    finally:
        _integration._REALIZATIONS = previous_capabilities

    width = control_count + 1
    body = [f"    q = cudaq.qvector({width})"]
    for index in range(width):
        body.extend([f"    if basis & {1 << index}:", f"        x(q[{index}])"])
    body.append("    if phase:")
    for index in range(width):
        body.extend(
            [
                f"        ry({0.37 + index * 0.13}, q[{index}])",
                f"        rz({0.29 + index * 0.17}, q[{index}])",
            ]
        )
    reference = [*body]
    _emit_controlled_instruction(
        CircuitInstruction("x", (control_count,), controls=tuple(range(control_count))),
        body,
        indent="    ",
        parameter_symbols={},
    )
    source = "import cudaq\n@cudaq.kernel\ndef probe(basis: int, phase: bool):\n"
    source += "\n".join(body) + "\n\n"
    source += "@cudaq.kernel\ndef reference(basis: int, phase: bool):\n"
    source += "\n".join(reference) + "\n"
    path = tmp_path / "mcx_witness.py"
    path.write_text(source)
    previous_target = cudaq.get_target()
    try:
        cudaq.set_target("qpp-cpu")
        scope = runpy.run_path(str(path))
        restored = cudaq.PyKernelDecorator.from_json(scope["probe"].to_json())
        mask = (1 << control_count) - 1
        for basis in range(1 << width):
            actual = np.asarray(cudaq.get_state(restored, basis, False))
            expected = np.zeros(1 << width, dtype=complex)
            index = basis ^ (1 << control_count) if basis & mask == mask else basis
            expected[index] = 1
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
        expected = np.asarray(cudaq.get_state(scope["reference"], 0, True)).copy()
        expected[mask], expected[-1] = expected[-1], expected[mask]
        actual = np.asarray(cudaq.get_state(restored, 0, True))
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    finally:
        cudaq.set_target(previous_target)


@pytest.mark.hugr
def test_hugr_direct_mod_mul_honors_shared_schoolbook_override() -> None:
    """Direct HUGR emits and reports the chosen multiplication implementation."""
    pytest.importorskip("hugr")
    import qamomile as qm
    from qamomile.hugr import Hugr

    source = """
import qamomile as qm
@qm.qkernel
def witness() -> qm.Bits[2]:
    target = qm.as_uint(qm.prepare((1, 0)))
    return qm.measure(qm.as_bits(qm.mod_mul(target, factor=2, modulus=3)))
"""
    program = qm.build_source(source).specialize("witness")
    default = qm.compile(program, to=Hugr())
    schoolbook = qm.compile(
        program,
        to=Hugr(
            realizations=qm.RealizationOptions(
                {qm.mod_mul: "mod_mul.schoolbook_ripple"}
            )
        ),
    )
    assert "realization:arithmetic.portable" in default.report.stages
    assert "realization:mod_mul.schoolbook_ripple" in schoolbook.report.stages
    assert "realization:arithmetic.portable" not in schoolbook.report.stages
