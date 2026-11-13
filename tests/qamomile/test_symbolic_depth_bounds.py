"""Symbolic dependency-depth upper bounds stay sound and get tighter.

The scheduler keeps exact results where its exact rules apply. Elsewhere it
no longer gives up on the whole program: unprovable lane selections occupy
their whole owner, loops whose iterations are proved disjoint (lanes
``i + c`` with offsets separated by the trip count) run in parallel, other
loops compose their locally scheduled bodies sequentially, and branches with
different schedules become one region lasting the longer branch. Such results
are reported as upper bounds no larger than the algebraic bound.
"""

from __future__ import annotations

import random

import pytest
from qiskit import QuantumCircuit

import qamomile as qm
import qamomile.resources as qr
from qamomile.compilation import OptimizationPipeline
from qamomile.qiskit import Qiskit

SOURCE = """
from __future__ import annotations

import qamomile as qm

@qm.qkernel
def pairs(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2 * n]]:
    q = qm.qubits(2 * n)
    for i in range(n):
        q = qm.cx(q, control=i, target=i + n)
    return qm.measure(q)

@qm.qkernel
def doubling_tree(n: qm.Const[qm.Int], levels: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    q = qm.h(qm.qubits(n), target=0)
    for level in range(levels):
        stride = 2 ** level
        for i in range(stride):
            q = qm.cx(q, control=i, target=i + stride)
    return qm.measure(q)

@qm.qkernel
def brick(n: qm.Const[qm.Int], rounds: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[2 * n]]:
    q = qm.qubits(2 * n)
    for _round in range(rounds):
        for i in range(n):
            q = qm.h(q, target=i)
            q = qm.cx(q, control=i, target=i + n)
    return qm.measure(q)

@qm.qkernel
def layered(
    n: qm.Const[qm.Int],
    m: qm.Const[qm.Int],
    layers: qm.Const[qm.Int],
    edges: qm.Const[qm.Map[tuple[qm.Int, qm.Int], qm.Real, m]],
) -> qm.Shot[qm.Bits[n]]:
    q = qm.qubits(n)
    for i in range(n):
        q = qm.h(q, target=i)
    for _layer in range(layers):
        for edge, weight in edges.items():
            q = qm.rzz(q, weight, left=edge[0], right=edge[1])
        for i in range(n):
            q = qm.rx(q, 0.3, target=i)
    return qm.measure(q)

@qm.qkernel
def guarded(n: qm.Const[qm.Int], steps: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    q = qm.qubits(n)
    if steps > 0:
        for i in range(n):
            q = qm.h(q, target=i)
    else:
        q = qm.x(q, target=0)
    for i in range(n - 1):
        q = qm.cx(q, control=i, target=i + 1)
    return qm.measure(q)

@qm.qkernel
def while_bound(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[n]]:
    q = qm.h(qm.qubits(n), target=0)
    levels = 0
    while (2 ** levels) < n:
        levels += 1
    for level in range(levels):
        q = qm.x(q, target=0)
    return qm.measure(q)
"""


def _depth(entry: str, binding: dict[str, int]) -> tuple[object, object, str]:
    study = qr.logical(qm.build_source(SOURCE)[entry])
    depth = study.metric(qr.LOGICAL_GATE_DEPTH)
    low, high = depth.evaluate(binding).numeric_enclosure()
    return low, high, depth.relation.value


@pytest.mark.parametrize(
    ("entry", "binding", "bound"),
    [
        ("pairs", {"n": 16}, 1),
        ("doubling_tree", {"n": 32, "levels": 5}, 6),
        ("brick", {"n": 8, "rounds": 3}, 6),
        ("layered", {"n": 64, "m": 96, "layers": 2}, 195),
    ],
)
def test_parallel_structure_tightens_the_depth_upper_bound(
    entry: str, binding: dict[str, int], bound: int
) -> None:
    """Disjoint iterations and parallel inner loops no longer cost their work."""

    _low, high, relation = _depth(entry, binding)
    assert high == bound
    assert relation in {"upper_bound", "enclosure"}


def test_branches_with_different_schedules_keep_the_rest_of_the_schedule() -> None:
    """The branch becomes one region; the following chain is still scheduled."""

    _low, high, _relation = _depth("guarded", {"n": 10, "steps": 1})
    assert high is not None and high <= 1 + 9


def test_loop_bound_from_a_const_while_is_an_unknown_not_a_failure() -> None:
    """An underivable trip count leaves named unknowns instead of an error."""

    study = qr.logical(qm.build_source(SOURCE)["while_bound"])
    depth = study.metric(qr.LOGICAL_GATE_DEPTH)
    assert depth.unknown_contributors
    assert study.metric(qr.LOGICAL_PEAK_QUBITS).evaluate(
        {"n": 8}
    ).numeric_enclosure() == (
        8,
        8,
    )


def _circuit_depth(circuit: QuantumCircuit) -> int:
    """Depth of the non-measurement gates with structured loops unrolled."""

    body = QuantumCircuit(*circuit.qregs)

    def append(source: QuantumCircuit, qubits: list) -> None:
        for instruction in source.data:
            name = instruction.operation.name
            mapped = [
                qubits[source.find_bit(qubit).index] for qubit in instruction.qubits
            ]
            if name in {"measure", "barrier"}:
                continue
            if name == "for_loop":
                indices, parameter, loop_body = instruction.operation.params
                for value in indices:
                    bound = (
                        loop_body.assign_parameters({parameter: value})
                        if parameter is not None
                        else loop_body
                    )
                    append(bound, mapped)
                continue
            body.append(instruction.operation, mapped)

    append(circuit, list(body.qubits))
    return body.depth()


def _unoptimized_circuit(module: qm.Module, n: int) -> QuantumCircuit:
    """Project the source program without optional optimization passes.

    The logical depth describes the source, so cancellations performed by the
    standard pipeline must not shrink the reference depth.
    """

    artifact = qm.compile(
        module.specialize("f", n=n),
        to=Qiskit(optimization_level=0),
        optimizer=OptimizationPipeline(),
    )
    return qm.qiskit.load_circuit(artifact)


def _random_kernel(rng: random.Random) -> str:
    lines = []
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["fixed", "lane", "affine", "nested", "chain"])
        if kind == "fixed":
            control, target = rng.sample(range(3), 2)
            lines.append(f"    q = qm.cx(q, control={control}, target={target})")
        elif kind == "lane":
            lines.append("    for i in range(n):")
            lines.append(
                f"        q = qm.h(q, target=i + {rng.choice(['0', '1', 'n'])})"
            )
        elif kind == "affine":
            lines.append("    for i in range(n):")
            lines.append(
                f"        q = qm.cx(q, control=i, target=i + {rng.choice(['1', '2', 'n'])})"
            )
            if rng.random() < 0.5:
                lines.append("        q = qm.rz(q, 0.1, target=i)")
        elif kind == "nested":
            lines.append(f"    for j in range({rng.choice(['2', 'n'])}):")
            lines.append("        for i in range(n):")
            lines.append("            q = qm.cx(q, control=i, target=i + n)")
            if rng.random() < 0.5:
                lines.append("        q = qm.x(q, target=j)")
        else:
            lines.append("    for i in range(n - 1):")
            lines.append("        q = qm.cx(q, control=i, target=i + 1)")
    return (
        "from __future__ import annotations\n\nimport qamomile as qm\n\n"
        "@qm.qkernel\n"
        "def f(n: qm.Const[qm.Int]) -> qm.Shot[qm.Bits[3 * n + 3]]:\n"
        "    q = qm.qubits(3 * n + 3)\n"
        + "\n".join(lines)
        + "\n    return qm.measure(q)\n"
    )


@pytest.mark.parametrize("seed", range(6))
def test_depth_bounds_enclose_the_emitted_circuit_depth(seed: int) -> None:
    """Every reported bound contains the depth of the projected circuit."""

    rng = random.Random(seed)
    for _ in range(8):
        module = qm.build_source(_random_kernel(rng))
        depth = qr.logical(module["f"]).metric(qr.LOGICAL_GATE_DEPTH)
        for n in (2, 3, 5):
            low, high = depth.evaluate({"n": n}).numeric_enclosure()
            actual = _circuit_depth(_unoptimized_circuit(module, n))
            assert high is None or actual <= high
            assert low is None or low <= actual
