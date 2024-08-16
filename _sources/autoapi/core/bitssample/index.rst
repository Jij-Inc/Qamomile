core.bitssample
===============

.. py:module:: core.bitssample

.. autoapi-nested-parse::

   This module provides classes for representing and manipulating bit samples
   from quantum computations in Qamomile.



Classes
-------

.. autoapisummary::

   core.bitssample.BitsSample
   core.bitssample.BitsSampleSet


Module Contents
---------------

.. py:class:: BitsSample

   Represents a single bit array sample with its occurrence count.

   .. attribute:: num_occurrences

      The number of times this bit array occurred in the sample set.

      :type: int

   .. attribute:: bits

      The bit array represented as a list of integers (0 or 1).

      :type: List[int]


   .. py:attribute:: num_occurrences
      :type:  int


   .. py:attribute:: bits
      :type:  list[int]


   .. py:property:: num_bits
      :type: int

      Returns the number of bits in the sample.

      :returns: The length of the bit array.
      :rtype: int


.. py:class:: BitsSampleSet

   Represents a set of bit array samples from quantum computations.

   This class provides methods for converting between different representations
   of the sample set and analyzing the results.

   .. attribute:: bitarrays

      A list of BitsSample objects representing the sample set.

      :type: List[BitsSample]


   .. py:attribute:: bitarrays
      :type:  list[BitsSample]


   .. py:method:: get_int_counts() -> dict[int, int]

      Converts the bit array samples to integer counts.

      This method interprets each bit array as a binary number and counts
      the occurrences of each unique integer value.

      :returns: A dictionary mapping integer values to their occurrence counts.
      :rtype: dict[int, int]



   .. py:method:: from_int_counts(int_counts: dict[int, int], bit_length: int) -> BitsSampleSet
      :classmethod:


      Creates a BitsSampleSet from a dictionary of integer counts.

      This class method converts integer-based sample counts to bit array samples.

      :param int_counts: A dictionary mapping integer values to their occurrence counts.
      :type int_counts: dict[int, int]
      :param bit_length: The length of the bit arrays to be created.
      :type bit_length: int

      :returns: A new BitsSampleSet object containing the converted samples.
      :rtype: BitsSampleSet



   .. py:method:: get_most_common(n: int = 1) -> list[BitsSample]

      Returns the n most common bit samples in the set.

      :param n: The number of most common samples to return. Defaults to 1.
      :type n: int, optional

      :returns:

                A list of the n most common BitsSample objects,
                                  sorted by occurrence in descending order.
      :rtype: List[BitsSample]



   .. py:method:: total_samples() -> int

      Calculates the total number of samples in the set.

      :returns: The sum of occurrence counts across all samples.
      :rtype: int



