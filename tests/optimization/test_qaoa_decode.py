"""Tests for QAOA converter decode functionality."""

import pytest
import numpy as np
import ommx.v1
from qamomile.optimization.qaoa import QAOAConverter, _convert_sampleresult_to_bitssampleset
from qamomile.circuit.transpiler.job import SampleResult
import qamomile.core.bitssample as qm_bs


@pytest.fixture
def simple_qubo_instance():
    """Create a simple QUBO instance for testing."""
    # Create a simple 2-variable QUBO problem
    # min: x0 + 2*x1 + 3*x0*x1

    # Create decision variables
    x = [ommx.v1.DecisionVariable.binary(i) for i in range(2)]

    # Create objective: x0 + 2*x1 + 3*x0*x1
    objective = x[0] + 2 * x[1] + 3 * x[0] * x[1]

    instance = ommx.v1.Instance.from_components(
        decision_variables=x,
        objective=objective,
        constraints=[],
        sense=ommx.v1.Instance.MINIMIZE,
    )

    return instance


@pytest.fixture
def qaoa_converter(simple_qubo_instance):
    """Create QAOAConverter for testing."""
    return QAOAConverter(simple_qubo_instance)


def test_convert_sampleresult_to_bitssampleset():
    """Test conversion from SampleResult to BitsSampleSet."""
    # Create a mock SampleResult
    result = SampleResult(
        results=[([0, 1, 0], 10), ([1, 1, 1], 5), ([0, 0, 0], 3)],
        shots=18
    )

    # Convert to BitsSampleSet
    bitssampleset = _convert_sampleresult_to_bitssampleset(result)

    # Verify structure
    assert isinstance(bitssampleset, qm_bs.BitsSampleSet)
    assert len(bitssampleset.bitarrays) == 3

    # Verify first sample
    assert bitssampleset.bitarrays[0].bits == [0, 1, 0]
    assert bitssampleset.bitarrays[0].num_occurrences == 10

    # Verify second sample
    assert bitssampleset.bitarrays[1].bits == [1, 1, 1]
    assert bitssampleset.bitarrays[1].num_occurrences == 5

    # Verify third sample
    assert bitssampleset.bitarrays[2].bits == [0, 0, 0]
    assert bitssampleset.bitarrays[2].num_occurrences == 3


def test_convert_empty_sampleresult():
    """Test conversion with empty SampleResult."""
    result = SampleResult(results=[], shots=0)
    bitssampleset = _convert_sampleresult_to_bitssampleset(result)

    assert isinstance(bitssampleset, qm_bs.BitsSampleSet)
    assert len(bitssampleset.bitarrays) == 0


def test_qaoa_converter_decode_simple(qaoa_converter):
    """Test decode method with simple SampleResult."""
    # Create a mock SampleResult with 2-bit strings
    sample_result = SampleResult(
        results=[([0, 0], 512), ([1, 1], 512)],
        shots=1024
    )

    # Decode to SampleSet
    sampleset = qaoa_converter.decode(sample_result)

    # Verify result type
    assert isinstance(sampleset, ommx.v1.SampleSet)

    # Verify total number of samples
    assert len(sampleset.sample_ids) == 1024

    # Verify we have objectives computed
    assert hasattr(sampleset, 'objectives')


def test_qaoa_converter_decode_single_sample(qaoa_converter):
    """Test decode with single sample."""
    sample_result = SampleResult(
        results=[([0, 1], 1)],
        shots=1
    )

    sampleset = qaoa_converter.decode(sample_result)

    assert isinstance(sampleset, ommx.v1.SampleSet)
    assert len(sampleset.sample_ids) == 1


def test_qaoa_converter_decode_multiple_occurrences(qaoa_converter):
    """Test that num_occurrences is properly handled."""
    sample_result = SampleResult(
        results=[([0, 0], 100), ([0, 1], 50), ([1, 0], 30), ([1, 1], 20)],
        shots=200
    )

    sampleset = qaoa_converter.decode(sample_result)

    # Total samples should equal total occurrences
    assert len(sampleset.sample_ids) == 200


def test_decode_preserves_ising_mapping(qaoa_converter):
    """Test that Ising index mapping is correctly applied."""
    sample_result = SampleResult(
        results=[([0, 0], 1), ([1, 1], 1)],
        shots=2
    )

    sampleset = qaoa_converter.decode(sample_result)

    # Verify that samples have valid OMMX structure
    assert isinstance(sampleset, ommx.v1.SampleSet)

    # The converter should map Ising indices to original problem variable IDs
    # This is tested by verifying the decode doesn't raise errors
    assert len(sampleset.sample_ids) == 2
