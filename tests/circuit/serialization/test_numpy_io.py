"""Tests for rank- and dtype-preserving NumPy payload serialization."""

import numpy as np
import pytest

from qamomile.circuit.ir.serialize.numpy_io import array_to_dict, dict_to_array
from qamomile.circuit.serialization.graph_protobuf import _payload_to_proto


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_zero_dimensional_array_round_trip_preserves_shape(dtype: type) -> None:
    """Zero-dimensional arrays remain rank zero for every float width."""
    source = np.array(1.25, dtype=dtype)

    restored = dict_to_array(array_to_dict(source))

    assert restored.shape == ()
    assert restored.dtype == source.dtype
    assert restored.item() == source.item()


def test_c_contiguous_array_skips_recontiguation(mocker) -> None:
    """C-contiguous inputs avoid a redundant full-array conversion."""
    source = np.arange(12, dtype=np.float64).reshape(3, 4)
    ascontiguousarray = mocker.patch.object(
        np,
        "ascontiguousarray",
        side_effect=AssertionError("C-contiguous input must not be copied"),
    )

    restored = dict_to_array(array_to_dict(source))

    ascontiguousarray.assert_not_called()
    np.testing.assert_array_equal(restored, source)


def test_payload_nesting_is_rejected_before_protobuf_parser_limit() -> None:
    """Encoding rejects excessive nesting with the documented ValueError."""
    nested: object = 0
    for _ in range(42):
        nested = {"$tuple": [nested]}

    with pytest.raises(ValueError, match="nesting exceeds"):
        _payload_to_proto(nested)
