# Limitations

## Dict Subscript Lookup Does Not Yet Support Container Values

When `Dict[K, V]` uses a container value type such as `qmc.Tuple` or `qmc.Vector`, `Dict.__getitem__` raises `NotImplementedError`.

The current lookup path represents the result of `d[key]` as a single scalar `Value`. It does not yet rebuild frontend handles for structured lookup results or represent multi-value results in `DictGetItemOperation`, serialization, emission, and the classical executor.

A future implementation should allow `DictGetItemOperation` to produce `TupleValue` or `ArrayValue` results, then extend frontend handle reconstruction plus serialize, emit, and classical executor support for those structured results.
