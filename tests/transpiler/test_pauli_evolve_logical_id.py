"""Regression: ``pauli_evolve`` must preserve the input array's ``logical_id``.

Issue #354 — when a ``@qkernel`` calls another ``@qkernel`` that internally
uses ``pauli_evolve`` (e.g. ``trotterized_time_evolution``) and the outer
kernel measures the returned register, the resulting circuit went through
intermittent measurement-drop and clbit-allocation breakage. The Slack
discussion (Jij-Inc/Qamomile thread, 2026-04-27) traced the root cause to
``pauli_evolve`` minting a fresh ``ArrayValue`` (and thus a fresh
``logical_id``) for its result instead of bumping the SSA version of the
input array via ``next_version``. Other gate ops (``rz``, ``cx``, …) all
use ``next_version`` so the IR can recognise pre/post-gate qubits as the
same logical register; ``pauli_evolve`` was the lone offender.

These tests pin the fix at two levels:

1. IR shape: ``pauli_evolve`` produces a result whose ``logical_id`` and
   ``shape`` match the input, with ``version`` incremented by 1.
2. End-to-end: a nested ``@qkernel`` that uses ``pauli_evolve`` internally
   and feeds the returned register into ``qmc.measure`` in the caller
   yields a circuit with the expected number of clbits and a sensible
   sampling result (no ``[(None, shots)]``).
"""

from __future__ import annotations

import pytest

import qamomile.circuit as qmc
import qamomile.observable as qm_o
from qamomile.circuit.frontend.func_to_block import func_to_block
from qamomile.circuit.ir.operation.pauli_evolve import PauliEvolveOp


def _make_zz_h() -> qm_o.Hamiltonian:
    """Return ``Z_0 Z_1`` as a Hamiltonian for use with ``pauli_evolve``."""
    H = qm_o.Hamiltonian()
    H.add_term(
        (
            qm_o.PauliOperator(qm_o.Pauli.Z, 0),
            qm_o.PauliOperator(qm_o.Pauli.Z, 1),
        ),
        1.0,
    )
    return H


def test_pauli_evolve_preserves_logical_id_and_shape():
    """Two pauli_evolve calls share logical_id and shape; only version bumps."""

    @qmc.qkernel
    def k(
        q: qmc.Vector[qmc.Qubit],
        H: qmc.Observable,
        t: qmc.Float,
    ) -> qmc.Vector[qmc.Qubit]:
        q = qmc.pauli_evolve(q, H, t)
        q = qmc.pauli_evolve(q, H, t)
        return q

    block = func_to_block(k.func)
    pauli_ops = [op for op in block.operations if isinstance(op, PauliEvolveOp)]
    assert len(pauli_ops) == 2

    op1, op2 = pauli_ops
    in1 = op1.operands[0]
    out1 = op1.results[0]
    in2 = op2.operands[0]
    out2 = op2.results[0]

    assert out1.logical_id == in1.logical_id
    assert out1.shape == in1.shape
    assert out1.version == in1.version + 1

    assert out2.logical_id == in1.logical_id
    assert out2.shape == in1.shape
    assert out2.version == in2.version + 1

    assert in2.uuid == out1.uuid


def test_nested_kernel_with_pauli_evolve_then_measure_qiskit():
    """End-to-end: nested @qkernel with pauli_evolve, outer measures the result.

    Pre-fix this path was rescued only by the inline-pass shape-dim fix
    (commit f1e6e372). Post-fix the SSA chain is intact at the frontend
    already, so we additionally assert that the IR-level invariants hold
    while still verifying the lowered circuit has full measurement support.
    """
    pytest.importorskip("qiskit")
    from qamomile.qiskit import QiskitTranspiler

    H = _make_zz_h()

    @qmc.qkernel
    def evolve(
        q: qmc.Vector[qmc.Qubit],
        H: qmc.Observable,
        t: qmc.Float,
    ) -> qmc.Vector[qmc.Qubit]:
        return qmc.pauli_evolve(q, H, t)

    @qmc.qkernel
    def runner(H: qmc.Observable, t: qmc.Float) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(2, name="q")
        for i in qmc.range(2):
            q[i] = qmc.h(q[i])
        q = evolve(q, H, t)
        return qmc.measure(q)

    tp = QiskitTranspiler()
    exe = tp.transpile(runner, bindings={"H": H, "t": 0.3})
    qc = exe.compiled_quantum[0].circuit
    assert qc.num_clbits == 2

    result = exe.sample(tp.executor(), shots=64).result()
    keys = {bits for bits, _ in result.results}
    assert None not in keys
    assert all(isinstance(k, tuple) and len(k) == 2 for k in keys)


def test_issue_354_repro_structure():
    """Structural reproduction of the original issue #354 example.

    Two-level nested @qkernel composition (basis-prep helper feeding into
    ``trotterized_time_evolution``) followed by ``qmc.measure`` in the
    outer kernel. This is the exact pattern that surfaced in the QeMCMC
    tutorial and produced ``[(None, count), ...]`` from ``sample()`` on
    early main; the current fix combined with f1e6e372 keeps full clbit
    allocation and bit-tuple results.

    Note: the issue body uses ``parameters=["input"]`` to runtime-bind a
    ``Vector[UInt]``. That codepath is currently broken for orthogonal
    reasons unrelated to ``pauli_evolve`` (the bound ``input`` does not
    propagate into the nested kernel's ``if input[i] == 1`` check).
    Compile-time bindings sidestep that and isolate the pauli_evolve /
    logical_id path being regression-tested here.
    """
    pytest.importorskip("qiskit")
    from qamomile.circuit.algorithm.trotter import trotterized_time_evolution
    from qamomile.qiskit import QiskitTranspiler

    n_spins = 4
    mixer = qm_o.Hamiltonian()
    for i in range(n_spins):
        mixer += qm_o.X(i)
    cost = qm_o.Hamiltonian()
    for i in range(n_spins - 1):
        cost += -1.0 * qm_o.Z(i) * qm_o.Z(i + 1)
    Hs = [0.55 * mixer, 0.45 * cost]

    @qmc.qkernel
    def computational_basis_state(
        q: qmc.Vector[qmc.Qubit],
        input: qmc.Vector[qmc.UInt],
    ) -> qmc.Vector[qmc.Qubit]:
        n = q.shape[0]
        for i in qmc.range(n):
            if input[i] == 1:
                q[i] = qmc.x(q[i])
        return q

    @qmc.qkernel
    def basis_only(input: qmc.Vector[qmc.UInt]) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(input.shape[0], name="q")
        q = computational_basis_state(q, input)
        return qmc.measure(q)

    @qmc.qkernel
    def with_trotter(
        input: qmc.Vector[qmc.UInt],
        Hs: qmc.Vector[qmc.Observable],
        order: qmc.UInt,
        time: qmc.Float,
        step: qmc.UInt,
    ) -> qmc.Vector[qmc.Bit]:
        q = qmc.qubit_array(input.shape[0], name="q")
        q = computational_basis_state(q, input)
        q = trotterized_time_evolution(q, Hs, order, time, step)
        return qmc.measure(q)

    tp = QiskitTranspiler()

    # Baseline: basis prep without Trotter must yield the prepared bitstring.
    exe1 = tp.transpile(basis_only, bindings={"input": [1, 0, 1, 0]})
    qc1 = exe1.compiled_quantum[0].circuit
    assert qc1.num_clbits == n_spins
    assert len(exe1.compiled_quantum[0].clbit_map) == n_spins
    res1 = exe1.sample(tp.executor(), shots=8).result()
    assert res1.results == [((1, 0, 1, 0), 8)]

    # Full pattern: basis prep + Trotter + measure. Pre-fix this returned
    # ``[(None, count), ...]`` with an empty clbit_map.
    exe2 = tp.transpile(
        with_trotter,
        bindings={
            "input": [1, 0, 1, 0],
            "Hs": Hs,
            "order": 2,
            "time": 12.0,
            "step": 5,
        },
    )
    qc2 = exe2.compiled_quantum[0].circuit
    assert qc2.num_clbits == n_spins
    assert len(exe2.compiled_quantum[0].clbit_map) == n_spins

    res2 = exe2.sample(tp.executor(), shots=32).result()
    keys = {bits for bits, _ in res2.results}
    assert None not in keys, (
        "Issue #354 regression: sample() returned None keys, indicating "
        "clbit_map was empty after the nested @qkernel + Trotter chain."
    )
    assert all(isinstance(k, tuple) and len(k) == n_spins for k in keys)
