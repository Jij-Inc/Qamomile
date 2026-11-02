"""Measured control promotes functional classical array updates."""

import pytest

import qamomile as qm
from qamomile.qiskit import Aer

SOURCE = """import qamomile as qm
@qm.qkernel
def sample(flag: qm.Const[qm.Bool]) -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]:
    condition = qm.measure(qm.prepare(flag))
    values = [False, False]
    if condition:
        values[0] = True
    return values[0], values[1]
"""


@pytest.mark.parametrize("endpoint", [qm.Reference(), Aer()], ids=["reference", "aer"])
@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize(
    "source",
    [SOURCE, SOURCE.replace("values[0] = True", "values = [True, False]")],
    ids=["element-update", "whole-array-control"],
)
def test_measured_branch_updates_const_array(source, flag, endpoint):
    """Both branch outcomes execute after module and program persistence."""
    module = qm.build_source(source)
    module = qm.Module.from_bytes(module.to_bytes())
    program = module.specialize("sample", flag=flag)
    program = qm.Program.from_bytes(program.to_bytes())
    assert qm.run(program, on=endpoint, shots=16, seed=29).result.values == {
        (flag, False): 16
    }


@pytest.mark.parametrize("endpoint", [qm.Reference(), Aer()], ids=["reference", "aer"])
@pytest.mark.parametrize("flag", [False, True])
def test_measured_branch_update_preserves_const_alias(flag, endpoint):
    """A branch update rebinds its local array while preserving the old alias."""
    source = (
        SOURCE.replace(
            "tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool]]",
            "tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bool], qm.Shot[qm.Bool]]",
        )
        .replace("    if condition:", "    original = values\n    if condition:")
        .replace(
            "return values[0], values[1]",
            "return values[0], values[1], unchanged(original[0])",
        )
    )
    source += """
@qm.qkernel
def unchanged(value: qm.Const[qm.Bool]) -> qm.Const[qm.Bool]:
    return value
"""
    module = qm.build_source(source)
    module = qm.Module.from_bytes(module.to_bytes())
    program = module.specialize("sample", flag=flag)
    program = qm.Program.from_bytes(program.to_bytes())
    assert qm.run(program, on=endpoint, shots=16, seed=29).result.values == {
        (flag, False, False): 16
    }
