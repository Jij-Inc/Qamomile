from qamomile.core.bitssample import BitsSample, BitsSampleSet


def test_bits_sample_creation():
    sample = BitsSample(num_occurrences=5, bits=[0, 1, 1, 0])
    assert sample.num_occurrences == 5
    assert sample.bits == [0, 1, 1, 0]
    assert sample.num_bits == 4


def test_bits_sample_set_creation():
    samples = [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])]
    sample_set = BitsSampleSet(samples)
    assert len(sample_set.bitarrays) == 3


def test_get_int_counts():
    samples = [BitsSample(3, [0, 0]), BitsSample(2, [0, 1]), BitsSample(1, [1, 0])]
    sample_set = BitsSampleSet(samples)
    int_counts = sample_set.get_int_counts()
    assert int_counts == {0: 3, 1: 2, 2: 1}


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


def test_empty_sample_set():
    sample_set = BitsSampleSet([])
    assert sample_set.get_int_counts() == {}
    assert sample_set.get_most_common() == []
    assert sample_set.total_samples() == 0


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

