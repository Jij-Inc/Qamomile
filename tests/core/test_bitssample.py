import pytest

from qamomile.core.bitssample import BitsSample, BitsSampleSet


# >>> BitSample >>>
@pytest.mark.parametrize(
    "num_occurrences, bits",
    [
        (5, [0, 1, 1, 0]),
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
        [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])],
        # TODO: Check if this is a valid test case.
        [BitsSample(4, [1, 1, 0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1])],
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


def test_bits_sample_set_creation_with_empty_sample_set():
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


@pytest.mark.parametrize(
    "samples",
    [
        ([BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])]),
        ([BitsSample(1, [1]), BitsSample(4, [0])]),
        ([BitsSample(5, [1, 0, 1]), BitsSample(2, [0, 0, 1])]),
    ],
)
def test_get_int_counts(samples):
    """Run the get_int_counts method of BitsSampleSet.

    Check if
    1. The integer counts are correctly computed from the samples.
    """
    sample_set = BitsSampleSet(samples)
    int_counts = sample_set.get_int_counts()

    # Create expected counts by converting bits to integers.
    expected_counts = {
        # applying str to bits converts them into a string of '0's and '1's,
        # then joining them into a single string,
        # and finally converting that string to an integer with base 2.
        int("".join(map(str, bits_sample.bits)), 2): bits_sample.num_occurences
        for bits_sample in samples
    }
    # 1. The integer counts are correctly computed from the samples.
    assert int_counts == expected_counts


def test_from_int_counts():
    int_counts = {0: 3, 1: 2, 2: 1}
    sample_set = BitsSampleSet.from_int_counts(int_counts, bit_length=2)
    assert len(sample_set.bitarrays) == 3
    assert any(
        sample.bits == [0, 0] and sample.num_occurrences == 3
        for sample in sample_set.bitarrays
    )
    assert any(
        sample.bits == [1, 0] and sample.num_occurrences == 2
        for sample in sample_set.bitarrays
    )
    assert any(
        sample.bits == [0, 1] and sample.num_occurrences == 1
        for sample in sample_set.bitarrays
    )


def test_get_most_common():
    samples = [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])]
    sample_set = BitsSampleSet(samples)
    most_common = sample_set.get_most_common(2)
    assert len(most_common) == 2
    assert most_common[0].bits == [0, 0]
    assert most_common[1].bits == [0, 1]


def test_total_samples():
    samples = [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])]
    sample_set = BitsSampleSet(samples)
    assert sample_set.total_samples() == 6


def test_from_int_counts_with_larger_bit_length():
    int_counts = {0: 1, 15: 1}  # 15 is 1111 in binary
    sample_set = BitsSampleSet.from_int_counts(int_counts, bit_length=5)
    assert all(len(sample.bits) == 5 for sample in sample_set.bitarrays)
    assert any(sample.bits == [0, 0, 0, 0, 0] for sample in sample_set.bitarrays)
    assert any(sample.bits == [1, 1, 1, 1, 0] for sample in sample_set.bitarrays)


def test_get_most_common_with_ties():
    samples = [BitsSample(2, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])]
    sample_set = BitsSampleSet(samples)
    most_common = sample_set.get_most_common(3)
    assert len(most_common) == 3
    assert most_common[0].num_occurrences == most_common[1].num_occurrences == 2
    assert most_common[2].num_occurrences == 1


# <<< BitsSampleSet <<<
