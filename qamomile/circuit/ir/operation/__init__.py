"""Operation hierarchy of the qamomile IR.

Design center
-------------

Every IR node is an ``Operation`` (``operation.py``): a dataclass with
``operands`` / ``results`` lists of SSA-like ``Value``s, a declared
``Signature``, and an ``OperationKind`` (QUANTUM / CLASSICAL / HYBRID /
CONTROL) that drives classical/quantum segmentation. Operations are
data, not behavior — passes rewrite them; engines interpret them at
emit time.

Design principles
-----------------

- **Stay as abstract as the program semantics allow.** An operation
  expresses *what the program means*, never how an engine realizes it.
  ``MeasureVectorOperation`` is one op for a whole Vector — never N
  per-qubit ``MeasureOperation``s at IR level; per-qubit expansion is an
  emit-time engine concern. ``MeasureQFixedOperation`` and
  ``MeasureQIntOperation`` sit even higher (HYBRID measure + classical
  decode) and are split only when ``plan``'s segmentation forces it.
  Pre-expanding an abstract concept into
  per-element / per-qubit ops here is a design regression.
- **Generic value-access protocol.** Passes reach Values only through
  ``Operation.all_input_values()`` / ``replace_values()``. Subclasses
  that carry extra Value fields outside ``operands`` (e.g.
  ``ControlledUOperation.power``, loop bounds) override both, so generic
  passes need no per-subclass special cases. A new operation with extra
  Value fields MUST override these or passes will silently miss them.
- **Explicit ``Region`` protocol for control flow.** For / ForItems / If /
  While (``control_flow.py``) expose operations, block arguments, captures,
  and yields through ``nested_regions()`` / ``rebuild_regions()``; passes
  recurse through this protocol instead of isinstance chains, so new
  control-flow ops cannot be missed.
- **Loop-carried classical scalars are explicit ``RegionArg``s**
  (``init`` / ``block_arg`` / ``yielded`` / ``result``, in the style of
  MLIR ``scf.for`` iter_args/yield) on For / ForItems / While, making
  the carried dependency visible to dependency analysis.
- **Composite gates and callables stay boxed.** QFT / QPE / user
  kernels are ``InvokeOperation``s referencing a ``CallableDef``
  (``callable.py``) with an optional ``body``, alternative
  ``implementations``, and an optional bodyless ``opaque_cost``; whether an engine emits
  a native gate, the embedded body, or a shared decomposition is
  decided at emit time, not here.
- **Typed construction over raw operand lists.** ``GateOperation``
  offers ``rotation()`` / ``fixed()`` factories that keep theta as the
  last operand, plus ``theta`` / ``qubit_operands`` accessors for typed
  read access.
"""

from .callable import (
    CallableBodyRef,
    CallableDef,
    CallableImplementation,
    CallableRef,
    CallPolicy,
    CallTransform,
    CompositeGateType,
    InvokeOperation,
)
from .cast import CastOperation
from .classical_ops import (
    DecodeQFixedOperation,
    DecodeQIntOperation,
    DictGetItemOperation,
    ReturnQuantumArrayElementOperation,
    StoreArrayElementOperation,
)
from .control_flow import (
    BranchRebind,
    ForItemsOperation,
    HasNestedOps,
    LoopCarriedRebind,
    Region,
    RegionArg,
    validate_region_args,
)
from .expval import ExpvalOp
from .gate import (
    ConcreteControlledU,
    ControlledUOperation,
    GateOperation,
    GateOperationType,
    MeasureOperation,
    MeasureQFixedOperation,
    MeasureQIntOperation,
    MeasureVectorOperation,
    ProjectOperation,
    ResetOperation,
    SymbolicControlledU,
)
from .global_phase import GlobalPhaseOperation
from .inverse_block import InverseBlockOperation
from .operation import Operation
from .return_operation import ReturnOperation
from .select import SelectOperation
from .slice_array import ReleaseSliceViewOperation, SliceArrayOperation

__all__ = [
    "Operation",
    "ReturnOperation",
    "CallableBodyRef",
    "CallableDef",
    "CallableImplementation",
    "CallableRef",
    "CallPolicy",
    "CallTransform",
    "InvokeOperation",
    "InverseBlockOperation",
    "GlobalPhaseOperation",
    "CompositeGateType",
    "GateOperation",
    "GateOperationType",
    "MeasureOperation",
    "MeasureVectorOperation",
    "MeasureQFixedOperation",
    "MeasureQIntOperation",
    "ProjectOperation",
    "ResetOperation",
    "ControlledUOperation",
    "ConcreteControlledU",
    "SymbolicControlledU",
    "SelectOperation",
    "DecodeQFixedOperation",
    "DecodeQIntOperation",
    "DictGetItemOperation",
    "ReturnQuantumArrayElementOperation",
    "StoreArrayElementOperation",
    "CastOperation",
    # Control flow operations
    "BranchRebind",
    "ForItemsOperation",
    "HasNestedOps",
    "LoopCarriedRebind",
    "Region",
    "RegionArg",
    "validate_region_args",
    # Expectation value operation
    "ExpvalOp",
    # Slice operation
    "SliceArrayOperation",
    "ReleaseSliceViewOperation",
]
