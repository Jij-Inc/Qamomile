"""Frontend validation for qkernel return annotation shapes."""

import pytest

import qamomile.circuit as qmc


def test_scalar_annotation_rejects_array_return() -> None:
    """An array return is rejected before backend emission for a scalar API."""

    @qmc.qkernel
    def invalid() -> qmc.Qubit:
        return qmc.qubit_array(2, "q")  # type: ignore[return-value]

    with pytest.raises(TypeError, match="declares a scalar.*returned Vector"):
        _ = invalid.block


def test_array_annotation_rejects_scalar_return() -> None:
    """A scalar return is rejected before backend emission for an array API."""

    @qmc.qkernel
    def invalid() -> qmc.Vector[qmc.Qubit]:
        return qmc.qubit("q")  # type: ignore[return-value]

    with pytest.raises(TypeError, match="declares an array.*returned Qubit"):
        _ = invalid.block
