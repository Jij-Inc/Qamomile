"""Runtime integer casts preserve native arithmetic at circuit launch."""

import math
from importlib import import_module

import pytest

import qamomile as qm
from qamomile.persistence import deserialize, serialize
from qamomile.qiskit import Aer, to_circuit

SLICE_SOURCE = """import qamomile as qm
@qm.qkernel
def total_angle(angles: qm.Param[qm.Vector[qm.Real,2]], start: qm.Const[qm.Int], offset: qm.Param[qm.Int]) -> qm.Real:
    window = angles[start:]
    total = 0.0
    for index, angle in enumerate(window, start=offset):
        total += angle * float(index + 1)
    return total
@qm.qkernel
def sample(angles: qm.Param[qm.Vector[qm.Real,2]], start: qm.Const[qm.Int], offset: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    window = angles[start:]
    state = qm.qubits(1)
    for index, angle in enumerate(window, start=offset):
        state = qm.rx(state, angle * float(index + 1))
    return qm.measure(state)
"""


def _restored_module(source):
    """Restore source through the public module persistence boundary."""
    return qm.Module.from_bytes(qm.build_source(source).to_bytes())


def _restored_program(module, entry="sample", **constants):
    """Restore a specialized program before compiling it for an endpoint."""
    return qm.Program.from_bytes(module.specialize(entry, **constants).to_bytes())


def exercise_param_real_cast_backend(endpoint, *, seed=29, source=SLICE_SOURCE):
    """Reuse one persisted artifact with three runtime enumerate offsets."""
    module = _restored_module(source)
    program = _restored_program(module, start=1)
    oracle = _restored_program(module, "total_angle", start=1)
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
    for offset in [0, 1, 2]:
        parameters = {"angles": (0.0, math.pi), "offset": offset}
        rotation = qm.evaluate(oracle, **parameters)
        expected = (round(rotation / math.pi) % 2,)
        assert qm.execute(
            artifact, on=endpoint, parameters=parameters, shots=16, seed=seed
        ).counts == {expected: 16}


@pytest.mark.parametrize("backend", ["reference", "qiskit"], ids=["reference", "aer"])
@pytest.mark.parametrize("control", ["const-bound", "literal-bound", "unsliced"])
def test_param_enumerate_cast_reuses_persisted_artifact(backend, control):
    """Const and literal slices and full arrays preserve runtime index casts."""
    source = SLICE_SOURCE
    if control == "literal-bound":
        source = source.replace("angles[start:]", "angles[1:]")
    elif control == "unsliced":
        source = source.replace("angles[start:]", "angles")
    endpoint = qm.Reference() if backend == "reference" else Aer()
    exercise_param_real_cast_backend(endpoint, source=source)


@pytest.mark.parametrize("backend", ["reference", "qiskit"], ids=["reference", "aer"])
def test_param_real_cast_inside_retained_helper(backend):
    """Quantum helper calls preserve a Param integer's conversion to Real."""
    helper = """@qm.qkernel
def rotate(state: qm.Q[qm.Bits[1]], angle: qm.Param[qm.Real], index: qm.Param[qm.Int]) -> qm.Q[qm.Bits[1]]:
    return qm.rx(state, angle * float(index + 1))
"""
    source = SLICE_SOURCE.replace(
        "state = qm.rx(state, angle * float(index + 1))",
        "state = rotate(state, angle, index)",
    ).replace("import qamomile as qm\n", "import qamomile as qm\n" + helper)
    endpoint = qm.Reference() if backend == "reference" else Aer()
    exercise_param_real_cast_backend(endpoint, source=source)


@pytest.mark.parametrize("backend", ["reference", "qiskit"], ids=["reference", "aer"])
def test_param_real_cast_from_integer_aggregate(backend):
    """An integer projected from a Param vector retains exact arithmetic."""
    source = """import qamomile as qm
@qm.qkernel
def angle(values: qm.Param[qm.Vector[qm.Int,2]]) -> qm.Real:
    return float(values[0] + values[1])
@qm.qkernel
def sample(values: qm.Param[qm.Vector[qm.Int,2]]) -> qm.Shot[qm.Bits[1]]:
    state = qm.rx(qm.qubits(1), angle(values) * 3.141592653589793)
    return qm.measure(state)
"""
    module = _restored_module(source)
    endpoint = qm.Reference() if backend == "reference" else Aer()
    artifact = deserialize(
        serialize(qm.compile(_restored_program(module), to=endpoint.target))
    )
    oracle = _restored_program(module, "angle")
    for values in [(0, 1), (1, 1), (10**400, 1 - 10**400)]:
        expected = (round(qm.evaluate(oracle, values=values)) % 2,)
        assert qm.execute(
            artifact, on=endpoint, parameters={"values": values}, shots=16, seed=29
        ).counts == {expected: 16}


def _numeric_source(expression):
    """Use the same native expression for the classical oracle and rotation."""
    return f"""import qamomile as qm
@qm.qkernel
def angle(n: qm.Param[qm.Int]) -> qm.Real:
    return {expression}
@qm.qkernel
def sample(n: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    state = qm.rx(qm.qubits(1), angle(n) * 3.141592653589793)
    return qm.measure(state)
"""


def _rotation_angles(circuit):
    """Read emitted Rx angles, including rotations inside retained helpers."""
    angles = []
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "rx":
            angles.append(float(operation.params[0]))
        elif operation.definition is not None:
            angles.extend(_rotation_angles(operation.definition))
    return angles


@pytest.mark.parametrize("backend", ["reference", "qiskit"], ids=["reference", "aer"])
@pytest.mark.parametrize(
    "expression,n,expected",
    [
        ("float(n + 1) - float(n)", 2**53, 0.0),
        ("float((n + 1) - n)", 10**400, 1.0),
    ],
    ids=["round-each-cast", "cancel-before-cast"],
)
def test_param_real_cast_preserves_numeric_order(backend, expression, n, expected):
    """Binary64 rounding follows exact integer work at each explicit cast."""
    module = _restored_module(_numeric_source(expression))
    program = _restored_program(module)
    oracle = _restored_program(module, "angle")
    assert qm.evaluate(oracle, n=n) == expected
    endpoint = qm.Reference() if backend == "reference" else Aer()
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
    assert qm.execute(
        artifact, on=endpoint, parameters={"n": n}, shots=16, seed=29
    ).counts == {(int(expected),): 16}
    if backend == "qiskit":
        circuit = to_circuit(program, parameters={"n": n})
        assert _rotation_angles(circuit) == [expected * math.pi]


def test_param_real_cast_rejects_overflow_before_sdk_execution(monkeypatch):
    """An unrepresentable integer raises a binary64 error before Aer runs."""
    from qiskit_aer import AerSimulator

    module = _restored_module(_numeric_source("float(n)"))
    program = _restored_program(module)
    with pytest.raises(qm.ExecutionError, match="binary64"):
        qm.evaluate(_restored_program(module, "angle"), n=10**400)
    endpoint = Aer()
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))

    def forbidden(*args, **kwargs):
        raise AssertionError("overflow must be rejected before SDK execution")

    monkeypatch.setattr(AerSimulator, "run", forbidden)
    with pytest.raises(qm.ExecutionError, match="binary64"):
        qm.execute(artifact, on=endpoint, parameters={"n": 10**400}, shots=16, seed=29)


def test_param_real_cast_in_expectation():
    """Concrete expectation preparation uses the runtime integer cast."""
    source = """import qamomile as qm
@qm.qkernel
def state(offset: qm.Param[qm.Int]) -> qm.Q[qm.Bits[1]]:
    return qm.rx(qm.qubits(1), float(offset) * 3.141592653589793)
"""
    program = _restored_program(_restored_module(source), "state")
    for offset, expected in [(0, 1.0), (1, -1.0)]:
        result = qm.expectation(
            program, qm.Z(0), on=Aer(), parameters={"offset": offset}, shots=8, seed=29
        )
        assert result.value == expected


@pytest.mark.parametrize("backend", ["reference", "qiskit"], ids=["reference", "aer"])
def test_param_real_cast_in_unselected_shot_branch(backend):
    """An untaken measured branch never converts its overflowing integer."""
    source = """import qamomile as qm
@qm.qkernel
def sample(n: qm.Param[qm.Int]) -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Real], qm.Shot[qm.Bool]]:
    selected = qm.measure(qm.prepare(False))
    if selected:
        result = float(n)
    else:
        result = 2.0
    return (selected, result, qm.measure(qm.prepare(selected)))
"""
    endpoint = qm.Reference() if backend == "reference" else Aer()
    artifact = deserialize(
        serialize(
            qm.compile(_restored_program(_restored_module(source)), to=endpoint.target)
        )
    )
    assert qm.execute(
        artifact, on=endpoint, parameters={"n": 10**400}, shots=8, seed=29
    ).values == {(False, 2.0, False): 8}


BASIC_SOURCE = """import qamomile as qm
@qm.qkernel
def angle(offset: qm.Param[qm.Int]) -> qm.Real:
    return float(offset + 1)
@qm.qkernel
def sample(offset: qm.Param[qm.Int]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.rx(qm.qubits(1), angle(offset) * 3.141592653589793))
"""


def exercise_basic_param_real_cast_backend(endpoint, *, seed=29):
    """Persist one scalar cast artifact and reuse three runtime inputs."""
    module = _restored_module(BASIC_SOURCE)
    oracle = _restored_program(module, "angle")
    artifact = deserialize(
        serialize(qm.compile(_restored_program(module), to=endpoint.target))
    )
    for offset in [0, 1, 2]:
        expected = (round(qm.evaluate(oracle, offset=offset)) % 2,)
        assert qm.execute(
            artifact, on=endpoint, parameters={"offset": offset}, shots=8, seed=seed
        ).counts == {expected: 8}


@pytest.mark.parametrize(
    "backend,module_name,endpoint_name,dependency,seed",
    [
        pytest.param(
            "cirq",
            "qamomile.cirq",
            "CirqSimulator",
            "cirq",
            29,
            marks=pytest.mark.cirq,
            id="cirq",
        ),
        pytest.param(
            "quri_parts",
            "qamomile.quri_parts",
            "Qulacs",
            "qulacs",
            29,
            marks=pytest.mark.quri_parts,
            id="qulacs",
        ),
        pytest.param(
            "braket",
            "qamomile.braket",
            "BraketLocal",
            "braket.circuits",
            0,
            marks=pytest.mark.braket,
            id="braket",
        ),
        pytest.param(
            "cudaq",
            "qamomile.cudaq",
            "CudaqLocal",
            "cudaq",
            29,
            marks=pytest.mark.cudaq,
            id="cudaq",
        ),
        pytest.param(
            "hugr",
            "qamomile.hugr",
            "Selene",
            "selene_sim",
            29,
            marks=pytest.mark.hugr,
            id="selene",
        ),
    ],
)
def test_param_real_cast_on_local_sdk(
    backend, module_name, endpoint_name, dependency, seed
):
    """Capable local SDKs execute a restored artifact with Param integer casts."""
    pytest.importorskip(dependency)
    endpoint_type = getattr(import_module(module_name), endpoint_name)
    endpoint = endpoint_type(timeout=30.0) if backend == "hugr" else endpoint_type()
    exercise_basic_param_real_cast_backend(endpoint, seed=seed)


@pytest.mark.parametrize(
    "target_module,target_name,dependency",
    [
        pytest.param(
            "qamomile.quration",
            "Quration",
            "pyqret",
            marks=pytest.mark.quration,
            id="quration",
        ),
        pytest.param(
            "qamomile.qir",
            "QIR",
            "pyqir",
            marks=pytest.mark.qir,
            id="qir",
        ),
    ],
)
def test_param_real_cast_preserves_static_target_rejection(
    target_module, target_name, dependency
):
    """Static targets reject Param integer casts with the stable capability error."""
    pytest.importorskip(dependency)
    target = getattr(import_module(target_module), target_name)()
    program = _restored_program(_restored_module(BASIC_SOURCE))
    with pytest.raises(qm.ArtifactError, match="Q7015.*parameterized_execution"):
        qm.compile(program, to=target)


def test_param_real_cast_keeps_unrelated_real_parameter_symbolic():
    """Native cast materialization consumes only its integer input."""
    from qamomile.circuit import lower_to_circuit_at

    source = """import qamomile as qm
@qm.qkernel
def sample(n: qm.Param[qm.Int], theta: qm.Param[qm.Real]) -> qm.Shot[qm.Bits[1]]:
    return qm.measure(qm.rx(qm.qubits(1), theta * float(n)))
"""
    program = _restored_program(_restored_module(source))
    with pytest.raises(qm.CapabilityError, match="Q4306"):
        to_circuit(program)
    logical = lower_to_circuit_at(program, {"n": 1})
    assert tuple(slot.name for slot in logical.parameter_abi.slots) == ("theta",)
    endpoint = Aer()
    artifact = deserialize(serialize(qm.compile(program, to=endpoint.target)))
    for theta, expected in [(0.0, (0,)), (math.pi, (1,))]:
        assert qm.execute(
            artifact,
            on=endpoint,
            parameters={"n": 1, "theta": theta},
            shots=8,
            seed=29,
        ).counts == {expected: 8}
