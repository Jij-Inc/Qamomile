"""Checked fixed-basis region projections at the native boundary."""

import pytest

import qamomile as qm
from qamomile import _qamomile_native as native


def test_signal_region_layout_is_checked_and_read_only() -> None:
    region = native.signal_region("1*00*")
    assert isinstance(region, native.SignalRegion)
    assert region.pattern == "1*00*"
    assert region.width == 5
    assert region.payload_indices == [1, 4]
    assert region.target == 3
    assert region.target_is_zero
    assert region.controls == [0, 2]
    assert region.control_state == "1"
    assert region.phase_width == 3
    order = list(range(region.width))
    for left, right in region.routing_swaps:
        order[left], order[right] = order[right], order[left]
    assert order[: region.phase_width] == [0, 2, 3]
    with pytest.raises(AttributeError):
        region.pattern = "*****"
    with pytest.raises(TypeError):
        native.SignalRegion()
    controls = region.controls
    controls.clear()
    assert region.controls == [0, 2]


def test_signal_region_budget_and_arbitrary_precision_controls() -> None:
    assert native.MAX_SIGNAL_REGION_WIDTH == 4096
    with pytest.raises(ValueError, match="Q7730"):
        native.signal_region("*" * 4097)
    with pytest.raises(ValueError, match="Q7731"):
        native.signal_region("0x1")
    region = native.signal_region("1" * 4096)
    assert int(region.control_state) == (1 << 4095) - 1
    assert region.payload_indices == []
    full = native.signal_region("***")
    assert full.target is None
    assert full.controls == []
    assert full.routing_swaps == []
    assert full.phase_width == 0


def test_derived_flag_retains_the_measured_word_sibling() -> None:
    source = """
import qamomile as qm
@qm.qkernel
def readout() -> tuple[qm.Shot[qm.Bool], qm.Shot[qm.Bits[2]]]:
    flags = qm.measure(qm.qubits(1))
    payload = qm.measure(qm.x(qm.qubits(2), target=1))
    return not flags[0], payload
"""
    program = qm.build_source(source)["readout"].specialize()
    result = qm.run(program, on=qm.Reference(), shots=4, seed=0).result
    assert result.values == {(True, (0, 1)): 4}
