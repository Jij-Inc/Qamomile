"""Const-bound Param slices retain their meaning across local SDK endpoints."""

import math
from importlib import import_module

import pytest

import qamomile as qm

from .test_const_slice_iteration_regressions import SOURCE

_LOOPS = ["for angle in window:", "for index, angle in enumerate(window):"]


def _restored_slice_program(source, loop, **constants):
    """Restore both public artifacts before selecting a local execution target."""
    module = qm.build_source(source.replace("for angle in window:", loop))
    module = qm.Module.from_bytes(module.to_bytes())
    program = module.specialize("sample", **constants)
    return qm.Program.from_bytes(program.to_bytes())


@pytest.mark.parametrize("loop", _LOOPS, ids=["for", "enumerate"])
@pytest.mark.parametrize(
    ("backend", "module_name", "endpoint_name", "dependency"),
    [
        pytest.param(
            "cirq",
            "qamomile.cirq",
            "CirqSimulator",
            "cirq",
            marks=pytest.mark.cirq,
            id="cirq",
        ),
        pytest.param(
            "quri_parts",
            "qamomile.quri_parts",
            "Qulacs",
            "qulacs",
            marks=pytest.mark.quri_parts,
            id="qulacs",
        ),
        pytest.param(
            "braket",
            "qamomile.braket",
            "BraketLocal",
            "braket.circuits",
            marks=pytest.mark.braket,
            id="braket",
        ),
        pytest.param(
            "cudaq",
            "qamomile.cudaq",
            "CudaqLocal",
            "cudaq",
            marks=pytest.mark.cudaq,
            id="cudaq",
        ),
        pytest.param(
            "hugr",
            "qamomile.hugr",
            "Selene",
            "selene_sim",
            marks=pytest.mark.hugr,
            id="selene",
        ),
    ],
)
def test_const_bound_param_slice_iteration_on_local_sdk(
    backend, module_name, endpoint_name, dependency, loop
):
    """Each capable SDK executes the original Param-array witness unchanged."""
    pytest.importorskip(dependency)
    endpoint = getattr(import_module(module_name), endpoint_name)()
    program = _restored_slice_program(SOURCE, loop, start=1)
    assert qm.run(
        program,
        on=endpoint,
        parameters={"angles": (0.0, math.pi)},
        shots=16,
        seed=0 if backend == "braket" else 29,
    ).result.values == {(1,): 16}


@pytest.mark.quration
@pytest.mark.parametrize("backend", ["quration"])
@pytest.mark.parametrize("loop", _LOOPS, ids=["for", "enumerate"])
def test_const_bound_param_slice_quration_rejects_runtime_parameters(backend, loop):
    """Quration rejects the original runtime inputs with its stable capability error."""
    pytest.importorskip("pyqret")
    integration = import_module(f"qamomile.{backend}")
    program = _restored_slice_program(SOURCE, loop, start=1)
    with pytest.raises(qm.ArtifactError, match="parameterized_execution"):
        qm.compile(program, to=integration.Quration())


@pytest.mark.quration
@pytest.mark.parametrize("backend", ["quration"])
@pytest.mark.parametrize("loop", _LOOPS, ids=["for", "enumerate"])
def test_const_bound_const_slice_executes_on_quration(backend, loop):
    """Quration executes the same slice when its angle array is specialized."""
    pytest.importorskip("pyqret")
    integration = import_module(f"qamomile.{backend}")
    source = SOURCE.replace("qm.Param[qm.Vector", "qm.Const[qm.Vector")
    program = _restored_slice_program(source, loop, start=1, angles=(0.0, math.pi))
    assert qm.run(
        program, on=integration.QurationLocal(), shots=16, seed=29
    ).result.values == {(1,): 16}
