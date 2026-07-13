"""Backend-neutral circuit code-generation IR.

This module is intentionally lower-level than Qamomile's semantic IR and
higher-level than any SDK object. It contains virtual quantum wires,
target-neutral scalar expressions, structured control flow, and reusable
circuit calls. Circuit-family targets legalize and materialize this IR;
program-graph targets such as HUGR do not pass through it.

The semantic-to-circuit lowering currently reuses the established emit walker
to preserve frontend behavior; the immutable ``CircuitProgram`` boundary is
where that mutable traversal ends.

Two rules govern what survives into this IR and how targets consume it:

* **Lowering may erase how a program was written, never what it means.**
  Semantic values, slices, bindings, and call machinery are erased; intent a
  target could exploit — semantic identity and immutable semantic arguments
  (:class:`CallableIdentity`), deferred call transforms, structured control
  flow, Pauli-evolution semantics — is preserved until a target explicitly
  decides otherwise.
  Erasure is irreversible while preservation costs one tag, so the default
  is to preserve.
* **Capabilities declare, legalization decides, materializers execute.**
  Each target owns an immutable :class:`CircuitCapabilities` declaration;
  :func:`legalize_program` rewrites a program under that declaration and the
  user's :class:`CompilationPolicy`; :func:`verify_target_legal` proves the
  result before materialization, so a materializer only ever converts a
  program it has already declared it accepts.
"""

from qamomile.circuit.transpiler.circuit_ir.capability import (
    ALL_BINARY_OPERATORS,
    ALL_PRIMITIVE_GATES,
    ALL_UNARY_OPERATORS,
    ARITHMETIC_BINARY_OPERATORS,
    DEFAULT_POLICY,
    CallControlMode,
    CallPhaseMode,
    CallTransformCapabilities,
    CircuitCapabilities,
    CompilationPolicy,
    GlobalPhaseCapabilities,
    NativeSemanticOpCapabilities,
    ScalarAtom,
    ScalarCapabilities,
    ScalarExpressionForm,
    StandalonePhaseMode,
)
from qamomile.circuit.transpiler.circuit_ir.emitter import CircuitGateEmitter
from qamomile.circuit.transpiler.circuit_ir.legalize import (
    legalize_program,
    verify_target_legal,
)
from qamomile.circuit.transpiler.circuit_ir.lowering import (
    CircuitLoweringPass,
    lower_circuit_plan,
)
from qamomile.circuit.transpiler.circuit_ir.materialize import (
    CircuitBackendEmitPass,
    CircuitMaterializer,
    MaterializedCircuit,
    materialize_executable,
)
from qamomile.circuit.transpiler.circuit_ir.model import (
    IQFT_SEMANTIC_KEY,
    MULTI_CONTROLLED_X_SEMANTIC_KEY,
    QFT_SEMANTIC_KEY,
    RIPPLE_CARRY_ADD_SEMANTIC_KEY,
    STATE_PREPARATION_SEMANTIC_KEY,
    BarrierInstruction,
    BinaryExpr,
    BinaryOperator,
    CallableIdentity,
    CallInstruction,
    CircuitBuilder,
    CircuitInstruction,
    CircuitProgram,
    ClassicalBitExpr,
    ForInstruction,
    GateInstruction,
    IfInstruction,
    LiteralExpr,
    LoopVariableExpr,
    MeasureInstruction,
    MeasureVectorInstruction,
    ParameterExpr,
    PauliEvolutionInstruction,
    PauliEvolutionRealization,
    ResetInstruction,
    ReusableCircuit,
    ScalarExpr,
    SemanticArguments,
    SemanticOpKey,
    UnaryExpr,
    UnaryOperator,
    WhileInstruction,
    WireId,
)
from qamomile.circuit.transpiler.circuit_ir.verify import verify_circuit

__all__ = [
    "ALL_BINARY_OPERATORS",
    "ALL_PRIMITIVE_GATES",
    "ALL_UNARY_OPERATORS",
    "ARITHMETIC_BINARY_OPERATORS",
    "BarrierInstruction",
    "BinaryExpr",
    "BinaryOperator",
    "CallableIdentity",
    "CallInstruction",
    "CallControlMode",
    "CallPhaseMode",
    "CallTransformCapabilities",
    "CircuitBuilder",
    "CircuitBackendEmitPass",
    "CircuitCapabilities",
    "CircuitGateEmitter",
    "CircuitLoweringPass",
    "CircuitInstruction",
    "CircuitMaterializer",
    "CircuitProgram",
    "ClassicalBitExpr",
    "CompilationPolicy",
    "DEFAULT_POLICY",
    "ForInstruction",
    "GateInstruction",
    "GlobalPhaseCapabilities",
    "IfInstruction",
    "LiteralExpr",
    "LoopVariableExpr",
    "MeasureInstruction",
    "MeasureVectorInstruction",
    "MaterializedCircuit",
    "NativeSemanticOpCapabilities",
    "ParameterExpr",
    "PauliEvolutionInstruction",
    "PauliEvolutionRealization",
    "QFT_SEMANTIC_KEY",
    "IQFT_SEMANTIC_KEY",
    "MULTI_CONTROLLED_X_SEMANTIC_KEY",
    "RIPPLE_CARRY_ADD_SEMANTIC_KEY",
    "STATE_PREPARATION_SEMANTIC_KEY",
    "ResetInstruction",
    "ReusableCircuit",
    "ScalarExpr",
    "SemanticOpKey",
    "SemanticArguments",
    "ScalarAtom",
    "ScalarCapabilities",
    "ScalarExpressionForm",
    "StandalonePhaseMode",
    "UnaryExpr",
    "UnaryOperator",
    "WhileInstruction",
    "WireId",
    "legalize_program",
    "verify_circuit",
    "verify_target_legal",
    "lower_circuit_plan",
    "materialize_executable",
]
