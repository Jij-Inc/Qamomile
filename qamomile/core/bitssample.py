"""
This module provides classes for representing and manipulating bit samples
from quantum computations in Qamomile.
"""

from __future__ import annotations
import dataclasses


@dataclasses.dataclass
class BitsSample:
    """
    Represents a single bit array sample with its occurrence count.

    Attributes:
        num_occurrences (int): The number of times this bit array occurred in the sample set.
        bits (List[int]): The bit array represented as a list of integers (0 or 1).
    """

    num_occurrences: int
    bits: list[int]

    @property
    def num_bits(self) -> int:
        """
        Returns the number of bits in the sample.

        Returns:
            int: The length of the bit array.
        """
        return len(self.bits)


@dataclasses.dataclass
class BitsSampleSet:
    """
    Represents a set of bit array samples from quantum computations.

    This class provides methods for converting between different representations
    of the sample set and analyzing the results.

    Attributes:
        bitarrays (List[BitsSample]): A list of BitsSample objects representing the sample set.
    """

    bitarrays: list[BitsSample]

    def __post_init__(self):
        """
        Validates that all BitsSample instances have the same bit length.

        Raises:
            ValueError: If BitsSample instances have different bit lengths.
        """
        if not self.bitarrays:
            return

        expected_length = self.bitarrays[0].num_bits
        for i, sample in enumerate(self.bitarrays):
            if sample.num_bits != expected_length:
                raise ValueError(
                    f"All BitsSample instances must have the same bit length. "
                    f"Expected {expected_length}, but sample at index {i} has {sample.num_bits} bits."
                )

    def get_int_counts(self) -> dict[int, int]:
        """
        Converts the bit array samples to integer counts.

        This method interprets each bit array as a binary number and counts
        the occurrences of each unique integer value.

        Returns:
            dict[int, int]: A dictionary mapping integer values to their occurrence counts.
        """
        int_counts = {}
        for bitarray in self.bitarrays:
            # Convert the bit array to an integer
            int_value = int("".join(map(str, bitarray.bits)), 2)
            # Add the occurrence count to the dictionary
            int_counts[int_value] = bitarray.num_occurrences
        return int_counts

    @classmethod
    def from_int_counts(
        cls, int_counts: dict[int, int], bit_length: int
    ) -> BitsSampleSet:
        """
        Creates a BitsSampleSet from a dictionary of integer counts.

        This class method converts integer-based sample counts to bit array samples.

        Args:
            int_counts (dict[int, int]): A dictionary mapping integer values to their occurrence counts.
            bit_length (int): The length of the bit arrays to be created.

        Returns:
            BitsSampleSet: A new BitsSampleSet object containing the converted samples.
            
        Raises:
            ValueError: If any integer value requires more bits than the specified bit_length.
        """
        # Validate that all integer values can be represented with the given bit_length
        max_representable = (1 << bit_length) - 1  # 2^bit_length - 1
        for int_value in int_counts.keys():
            if int_value < 0:
                raise ValueError(f"Negative integer values are not supported: {int_value}")
            if int_value > max_representable:
                required_bits = int_value.bit_length()
                raise ValueError(
                    f"Integer value {int_value} requires {required_bits} bits, "
                    f"but bit_length is only {bit_length}. "
                    f"Maximum representable value is {max_representable}."
                )
        
        bitarrays = []
        for int_value, count in int_counts.items():
            # Convert the integer to a bit array of the specified length
            bitarray = list(map(int, bin(int_value)[2:].zfill(bit_length)[::-1]))
            bitarrays.append(BitsSample(count, bitarray))

        return cls(bitarrays)

    def get_most_common(self, n: int = 1) -> list[BitsSample]:
        """
        Returns the n most common bit samples in the set.

        Args:
            n (int, optional): The number of most common samples to return. Defaults to 1.

        Returns:
            List[BitsSample]: A list of the n most common BitsSample objects,
                              sorted by occurrence in descending order.
        """
        sorted_samples = sorted(
            self.bitarrays, key=lambda x: x.num_occurrences, reverse=True
        )
        return sorted_samples[:n]

    def total_samples(self) -> int:
        """
        Calculates the total number of samples in the set.

        Returns:
            int: The sum of occurrence counts across all samples.
        """
        return sum(sample.num_occurrences for sample in self.bitarrays)
