"""Tests for zero-cost coherent product reassociation."""

from __future__ import annotations

import pytest

import qamomile as qm
from qamomile.circuit import lower_to_circuit

from .conftest import COHERENT_RESHAPE_SOURCE, lower_adjoint_calls


def test_split_at_and_concat_are_one_zero_cost_core_isomorphism() -> None:
    """Surface duals share one structural IR op and emit no circuit instruction."""

    module = qm.build_source(COHERENT_RESHAPE_SOURCE)
    program = module.specialize("reshaped_basis")
    direct = qm.build_source(
        """
import qamomile as qm

@qm.qkernel
def direct() -> qm.Bits[3]:
    state = qm.prepare((1, 0, 0))
    state = qm.x(state, target=0)
    state = qm.x(state, target=2)
    return qm.measure(state)
"""
    ).specialize("direct")

    assert program.inspect().count("qcore.reshape") == 2
    assert qm.evaluate(program) == (0, 0, 1)
    assert (
        lower_to_circuit(program).operation_count
        == lower_to_circuit(direct).operation_count
    )
    assert (
        qm.resources.logical(program)
        .metric(qm.resources.LOGICAL_WORK)
        .numeric_enclosure()
        == qm.resources.logical(direct)
        .metric(qm.resources.LOGICAL_WORK)
        .numeric_enclosure()
    )


def test_split_concat_preserves_entanglement_and_serialization() -> None:
    """Views retain shared amplitudes across a canonical program round-trip."""

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("reshaped_bell")
    restored = qm.Program.from_bytes(program.to_bytes())
    counts = qm.run(restored, on=qm.Reference(), shots=64, seed=4).result.counts

    assert set(counts) == {(0, 0), (1, 1)}
    assert restored.to_bytes() == program.to_bytes()


def test_adjoint_reverses_a_branched_linear_string_diagram() -> None:
    """Dagger lowering exchanges tuple/unpack instead of requiring one chain."""

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("branched_adjoint")
    lowered = lower_adjoint_calls(program)

    assert qm.evaluate(program) == (1, 0, 1)
    assert "qcore.tuple" in lowered.inspect()
    assert "proof='modular-adjoint'" in lowered.inspect()


def test_uncomputation_reuses_the_same_branched_dagger() -> None:
    """A dead split temporary is restored without a cleanup-only interpreter."""

    program = qm.build_source(COHERENT_RESHAPE_SOURCE).specialize("branched_temporary")

    assert qm.evaluate(program) is None
    assert "quantum.release" in program.inspect()
    assert program.inspect().count("proof='modular-adjoint'") >= 6


@pytest.mark.parametrize("boundary", [0, 3, -1])
def test_split_at_rejects_nonproper_boundaries(boundary: int) -> None:
    """A split boundary must produce two nonempty coherent components."""

    source = f"""
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Bits[3]:
    state = qm.qubits(3)
    left, right = qm.split_at(state, {boundary})
    return qm.measure(qm.concat(left, right))
"""
    with pytest.raises(qm.FrontendError, match="boundary must satisfy"):
        qm.build_source(source)


def test_concat_accepts_three_owners_and_replaces_the_old_public_names() -> None:
    """Variadic concatenation preserves order and obsolete aliases stay absent."""

    source = """
import qamomile as qm

@qm.qkernel
def combine() -> qm.Bits[3]:
    first = qm.prepare(True)
    second = qm.prepare(False)
    third = qm.prepare(True)
    return qm.measure(qm.concat(first, second, third))
"""

    assert qm.evaluate(qm.build_source(source).specialize("combine")) == (1, 0, 1)
    assert not hasattr(qm, "factor")
    assert not hasattr(qm, "tensor")


def test_concat_rejects_fewer_than_two_owners() -> None:
    """Concatenation requires a meaningful multi-owner join."""

    source = """
import qamomile as qm

@qm.qkernel
def invalid() -> qm.Bits[1]:
    state = qm.qubits(1)
    return qm.measure(qm.concat(state))
"""

    with pytest.raises(qm.FrontendError, match="expects at least two"):
        qm.build_source(source)
