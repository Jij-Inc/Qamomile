"""Const slice bounds preserve iteration over runtime array values."""

import math

import pytest

import qamomile as qm
from qamomile.qiskit import Aer

SOURCE = """import qamomile as qm
@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real,2]], start: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    window = angles[start:]
    state = qm.qubits(1)
    for angle in window:
        state = qm.rx(state, angle)
    return qm.measure(state)
"""


@pytest.mark.parametrize("endpoint", [qm.Reference(), Aer()], ids=["reference", "aer"])
@pytest.mark.parametrize(
    "loop",
    ["for angle in window:", "for index, angle in enumerate(window):"],
    ids=["for", "enumerate"],
)
def test_const_bound_param_slice_iteration(endpoint, loop):
    """Both loops execute the selected runtime angle after artifact round trips."""
    module = qm.build_source(SOURCE.replace("for angle in window:", loop))
    module = qm.Module.from_bytes(module.to_bytes())
    program = module.specialize("sample", start=1)
    program = qm.Program.from_bytes(program.to_bytes())
    result = qm.run(
        program,
        on=endpoint,
        parameters={"angles": (0.0, math.pi)},
        shots=16,
        seed=29,
    ).result.values
    assert result == {(1,): 16}


@pytest.mark.parametrize("endpoint", [qm.Reference(), Aer()], ids=["reference", "aer"])
@pytest.mark.parametrize(
    "control",
    ["literal-for", "literal-enumerate", "unsliced-enumerate", "const-enumerate"],
)
def test_slice_iteration_controls(endpoint, control):
    """Literal bounds, unsliced arrays, and Const arrays establish working controls."""
    source = SOURCE
    constants = {"start": 1}
    parameters = {"angles": (0.0, math.pi)}
    if control.startswith("literal"):
        source = source.replace("angles[start:]", "angles[1:]")
    elif control == "unsliced-enumerate":
        source = source.replace("angles[start:]", "angles")
    elif control == "const-enumerate":
        source = source.replace("qm.Param[qm.Vector", "qm.Const[qm.Vector")
        constants.update(parameters)
        parameters = {}
    if control.endswith("enumerate"):
        source = source.replace(
            "for angle in window:", "for index, angle in enumerate(window):"
        )
    module = qm.build_source(source)
    module = qm.Module.from_bytes(module.to_bytes())
    program = module.specialize("sample", **constants)
    program = qm.Program.from_bytes(program.to_bytes())
    assert qm.run(
        program, on=endpoint, parameters=parameters, shots=16, seed=29
    ).result.values == {(1,): 16}


@pytest.mark.parametrize("endpoint", [qm.Reference(), Aer()], ids=["reference", "aer"])
@pytest.mark.parametrize("enumerated", [False, True], ids=["for", "enumerate"])
@pytest.mark.parametrize(
    "start,stop,step,nested",
    [
        (1, 5, 1, False),
        (-3, 5, 1, False),
        (-100, 100, 2, False),
        (4, -6, -2, False),
        (1, 5, 2, False),
        (1, 5, 1, True),
    ],
    ids=["multiple", "negative", "clipped", "reverse", "stride", "nested"],
)
def test_const_slice_iteration_resolves_general_shapes(
    endpoint, enumerated, start, stop, step, nested
):
    """Different slice shapes and used enumerate indices match Python selection."""
    source = SOURCE.replace("qm.Vector[qm.Real,2]", "qm.Vector[qm.Real,5]")
    source = source.replace(
        "start: qm.Const[qm.Int]",
        "start: qm.Const[qm.Int], stop: qm.Const[qm.Int], step: qm.Const[qm.Int]",
    )
    source = source.replace(
        "angles[start:]", "angles[start:stop:step]" + ("[::-1]" if nested else "")
    )
    if enumerated:
        source = source.replace(
            "for angle in window:", "for index, angle in enumerate(window, start=3):"
        )
        source = source.replace(
            "qm.rx(state, angle)", "qm.rx(state, angle * float(index + 1))"
        )
    angles = (math.pi, 0.0, math.pi, math.pi, 0.0)
    selected = angles[start:stop:step]
    if nested:
        selected = selected[::-1]
    rotations = sum(
        round(angle / math.pi) * (index + 4 if enumerated else 1)
        for index, angle in enumerate(selected)
    )
    module = qm.Module.from_bytes(qm.build_source(source).to_bytes())
    program = module.specialize("sample", start=start, stop=stop, step=step)
    program = qm.Program.from_bytes(program.to_bytes())
    assert qm.run(
        program,
        on=endpoint,
        parameters={"angles": angles},
        shots=16,
        seed=29,
    ).result.values == {(rotations % 2,): 16}


@pytest.mark.parametrize("endpoint", [qm.Reference(), Aer()], ids=["reference", "aer"])
@pytest.mark.parametrize("enumerated", [False, True], ids=["for", "enumerate"])
def test_const_slice_iteration_inside_dependent_helper(endpoint, enumerated):
    """A retained helper derives its slice length from the call's Const bindings."""
    source = SOURCE.replace("def sample(", "def apply_angles(")
    source = source.replace("qm.Vector[qm.Real,2]", "qm.Vector[qm.Real,n]")
    if enumerated:
        source = source.replace(
            "for angle in window:", "for index, angle in enumerate(window):"
        )
    source += """
@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real,2]]) -> qm.Shot[qm.Bits[1]]:
    return apply_angles(angles, 1)
"""
    module = qm.Module.from_bytes(qm.build_source(source).to_bytes())
    program = module.specialize("sample")
    program = qm.Program.from_bytes(program.to_bytes())
    assert qm.run(
        program,
        on=endpoint,
        parameters={"angles": (0.0, math.pi)},
        shots=16,
        seed=29,
    ).result.values == {(1,): 16}


@pytest.mark.parametrize("endpoint", [qm.Reference(), Aer()], ids=["reference", "aer"])
def test_slice_enumerate_index_remains_available_to_const_helper(endpoint):
    """Element availability does not delay a shape-derived index used as Const."""
    source = """import qamomile as qm
@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]], angle: qm.Param[qm.Real], index: qm.Const[qm.Int]) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, angle * float(index + 1))
""" + SOURCE.replace("import qamomile as qm\n", "").replace(
        "for angle in window:", "for index, angle in enumerate(window):"
    ).replace("qm.rx(state, angle)", "rotate(state, angle, index)")
    module = qm.Module.from_bytes(qm.build_source(source).to_bytes())
    program = qm.Program.from_bytes(module.specialize("sample", start=1).to_bytes())
    assert qm.run(
        program,
        on=endpoint,
        parameters={"angles": (0.0, math.pi)},
        shots=16,
        seed=29,
    ).result.values == {(1,): 16}


@pytest.mark.parametrize("offset", [0, 1])
def test_slice_enumerate_preserves_runtime_start(offset):
    """A Param enumerate offset remains runtime data even with Const loop bounds."""
    source = """import qamomile as qm
@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real,2]], start: qm.Const[qm.Int], offset: qm.Param[qm.Int]) -> qm.Param[qm.Int]:
    window = angles[start:]
    total = 0
    for index, angle in enumerate(window, start=offset):
        total += index
    return total
"""
    module = qm.Module.from_bytes(qm.build_source(source).to_bytes())
    program = qm.Program.from_bytes(module.specialize("sample", start=1).to_bytes())
    assert qm.evaluate(program, angles=(0.0, math.pi), offset=offset) == offset
