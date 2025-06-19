import pytest

from qamomile.core.bitssample import BitsSample, BitsSampleSet


# >>> BitSample >>>
@pytest.mark.parametrize(
    "num_occurrences, bits",
    [
        # Standard cases with different bit lengths
        (3, [1, 0, 1]),
        (2, [0, 0]),
        (1, [1]),
        (0, []),
    ],
)
def test_bits_sample_creation(num_occurrences, bits):
    """Create a BitsSample instance and verify its attributes.

    Check if
    1. num_occurrences is set correctly,
    2. bits is set correctly,
    3. num_bits returns the correct length of the bits list.
    """
    sample = BitsSample(num_occurrences=num_occurrences, bits=bits)
    # 1. num_occurrences is set correctly,
    assert sample.num_occurrences == num_occurrences
    # 2. bits is set correctly,
    assert sample.bits == bits
    # 3. num_bits returns the correct length of the bits list.
    num_bits = len(bits)
    assert sample.num_bits == num_bits


# <<< Bit Sample <<<


# >>> BitsSampleSet >>>
@pytest.mark.parametrize(
    "samples",
    [
        # Standard case
        [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])],
        # Case with different lengths of bits
        [BitsSample(4, [1, 1, 0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1])],
        # Case with single sample
        [BitsSample(1, [1])],
    ],
)
def test_bits_sample_set_creation(samples):
    """Create a BitsSampleSet instance and verify its length.

    Check if
    1. The number of bitarrays in the set matches the number of samples provided.
    """
    sample_set = BitsSampleSet(samples)
    # 1. The number of bitarrays in the set matches the number of samples provided.
    num_bitarrays = len(samples)
    assert len(sample_set.bitarrays) == num_bitarrays


@pytest.mark.parametrize(
    "samples, expected_counts",
    [
        # Standard cases with different bit lengths
        (
            [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])],
            {0: 3, 1: 2, 2: 1},
        ),
        (
            [BitsSample(1, [1]), BitsSample(4, [0])],
            {0: 4, 1: 1},
        ),
        (
            [BitsSample(5, [1, 0, 1]), BitsSample(2, [0, 0, 1])],
            {5: 5, 1: 2},
        ),
    ],
)
def test_get_int_counts(samples, expected_counts):
    """Run the get_int_counts method of BitsSampleSet.

    Check if
    1. The integer counts are correctly computed from the samples.
    """
    sample_set = BitsSampleSet(samples)
    int_counts = sample_set.get_int_counts()
    # 1. The integer counts are correctly computed from the samples.
    assert int_counts == expected_counts


@pytest.mark.parametrize(
    "int_counts, bit_length, expected_samples",
    [
        # Standard cases with different bit lengths
        (
            {0: 3, 1: 2},
            1,
            [
                {"bits": [0], "num_occurrences": 3},
                {"bits": [1], "num_occurrences": 2},
            ],
        ),
        (
            {3: 4, 2: 2},
            2,
            [
                {"bits": [1, 1], "num_occurrences": 4},
                {"bits": [0, 1], "num_occurrences": 2},
            ],
        ),
        (
            {7: 1, 0: 5},
            3,
            [
                {"bits": [1, 1, 1], "num_occurrences": 1},
                {"bits": [0, 0, 0], "num_occurrences": 5},
            ],
        ),
    ],
)
def test_from_int_counts(int_counts, bit_length, expected_samples):
    """Run BitsSampleSet.from_int_counts class method.

    Check if
    1. The sample set is correctly created from a dictionary of integer counts,
    2. Each sample has the correct bits and occurrence count.
    """
    sample_set = BitsSampleSet.from_int_counts(int_counts, bit_length=bit_length)
    # 1. The sample set is correctly created from a dictionary of integer counts,
    num_samples = len(expected_samples)
    assert len(sample_set.bitarrays) == num_samples
    for expected_sample in expected_samples:
        # 2. Each sample has the correct bits and occurrence count.
        assert any(
            (
                sample.bits == expected_sample["bits"]
                and sample.num_occurrences == expected_sample["num_occurrences"]
            )
            for sample in sample_set.bitarrays
        )


@pytest.mark.parametrize(
    "samples, n, expected_bits",
    [
        # Standard case: 3 samples, get top 2
        (
            [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])],
            2,
            [[0, 0], [0, 1]],
        ),
        # n greater than number of samples
        ([BitsSample(1, [1, 1, 1])], 3, [[1, 1, 1]]),
        # n is zero
        ([BitsSample(2, [1, 0])], 0, []),
        # Empty samples
        ([], 2, []),
        # Tie in num_occurrences, should preserve order
        (
            [BitsSample(2, [1, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 1])],
            2,
            [[1, 0], [0, 1]],
        ),
    ],
)
def test_get_most_common(samples, n, expected_bits):
    """Run the get_most_common method of BitsSampleSet.

    Check if
    1. The most common samples are returned in the correct order,
    2. The number of returned samples matches the requested count.
    """
    sample_set = BitsSampleSet(samples)
    most_common = sample_set.get_most_common(n)
    # 1. The most common samples are returned in the correct order,
    assert [s.bits for s in most_common] == expected_bits
    # 2. The number of returned samples matches the requested count.
    assert len(most_common) == min(n, len(samples))


@pytest.mark.parametrize(
    "samples, expected_total",
    [
        # Standard case
        ([BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])], 6),
        # Single sample
        ([BitsSample(5, [1, 1, 1])], 5),
        # No samples
        ([], 0),
        # Multiple samples with zero occurrences
        ([BitsSample(0, [1, 0]), BitsSample(0, [0, 1])], 0),
        # Mixed zero and nonzero occurrences
        ([BitsSample(0, [1, 0]), BitsSample(2, [0, 1])], 2),
    ],
)
def test_total_samples(samples, expected_total):
    """Test the total_samples method of BitsSampleSet.

    Check if
    1. The total number of samples is computed correctly.
    """
    sample_set = BitsSampleSet(samples)
    # 1. The total number of samples is computed correctly.
    assert sample_set.total_samples() == expected_total


def test_empty_sample_set():
    """Create BitsSampleSet with an empty samples and run its methods.
    Also, compare the instance created by BitsSampleSet.from_int_counts with an empty int_counts.

    Check if
    1. get_int_counts returns an empty dict,
    2. get_most_common returns an empty list,
    3. total_samples returns 0,
    4. from_int_counts with an empty int_counts returns an empty BitsSampleSet being the same as the first creation in terms of their bitarrays.
    """
    sample_set = BitsSampleSet([])
    # 1. get_int_counts returns an empty dict,
    assert sample_set.get_int_counts() == {}
    # 2. get_most_common returns an empty list,
    assert sample_set.get_most_common() == []
    # 3. total_samples returns 0.
    assert sample_set.total_samples() == 0
    # 4. from_int_counts with an empty int_counts returns an empty BitsSampleSet being the same as the first creation in terms of their bitarrays.
    empty_int_counts = {}
    sample_set_from_int_counts = BitsSampleSet.from_int_counts(
        int_counts=empty_int_counts, bit_length=0
    )
    assert sample_set_from_int_counts.bitarrays == sample_set.bitarrays


def test_from_int_counts_with_larger_bit_length():
    """Test BitsSampleSet.from_int_counts with a larger bit length.

    Check if
    1. All samples have the correct bit length,
    2. The expected bit patterns are present.
    """
    int_counts = {0: 1, 15: 1}  # 15 is 1111 in binary
    sample_set = BitsSampleSet.from_int_counts(int_counts, bit_length=5)
    # 1. All samples should have bit length 5
    assert all(len(sample.bits) == 5 for sample in sample_set.bitarrays)
    # 2. Check for expected bit patterns
    assert any(sample.bits == [0, 0, 0, 0, 0] for sample in sample_set.bitarrays)
    assert any(sample.bits == [1, 1, 1, 1, 0] for sample in sample_set.bitarrays)


def test_get_most_common_with_ties():
    """Test get_most_common when there are ties in occurrence counts.

    Check if
    1. Samples with the same number of occurrences are handled correctly,
    2. The total number of returned samples matches the requested count.
    """
    samples = [BitsSample(2, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])]
    sample_set = BitsSampleSet(samples)
    most_common = sample_set.get_most_common(3)
    # 1. There should be three samples, with the first two having the same occurrence count
    assert len(most_common) == 3
    assert most_common[0].num_occurrences == most_common[1].num_occurrences == 2
    assert most_common[2].num_occurrences == 1


# <<< BitsSampleSet <<<
