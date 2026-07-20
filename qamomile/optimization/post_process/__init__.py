"""Classical post-processing of decoded quantum optimization samples.

Design intent: everything here runs *after* quantum execution and
decoding — it consumes ``BinaryModel`` / ``BinarySampleSet`` (and OMMX
samples) and refines them purely classically. ``LocalSearch``
(``local_search.py``) implements strictly-improving bit-flip descent,
with the step strategy selected by ``LocalSearchMethod``.

Constraints and extension points:
- No quantum or backend imports; this package must stay importable and
  runnable without any quantum SDK installed.
- New refinement strategies (tabu, simulated annealing, ...) belong here
  as sibling modules operating on the same ``BinaryModel``/sample-set
  contract, keeping converters free of post-processing logic.
"""

from .local_search import LocalSearch, LocalSearchMethod

__all__ = ["LocalSearch", "LocalSearchMethod"]
