"""Preserve compiler imports for the shared classical IR evaluator.

Evaluation semantics live in ``ir.classical_eval`` so frontend tracing and
compiler passes use the same pure operations without reversing dependencies.
"""

from qamomile.circuit.ir.classical_eval import (
    _RUNTIME_TO_BINOP_KIND as _RUNTIME_TO_BINOP_KIND,
    _RUNTIME_TO_COMPOP_KIND as _RUNTIME_TO_COMPOP_KIND,
    _RUNTIME_TO_CONDOP_KIND as _RUNTIME_TO_CONDOP_KIND,
    FoldPolicy as FoldPolicy,
    _is_runtime_parameter_operand as _is_runtime_parameter_operand,
    evaluate_binop_values as evaluate_binop_values,
    evaluate_compop_values as evaluate_compop_values,
    evaluate_condop_values as evaluate_condop_values,
    evaluate_notop_value as evaluate_notop_value,
    evaluate_runtime_op_values as evaluate_runtime_op_values,
    evaluate_unary_math_value as evaluate_unary_math_value,
    fold_classical_op as fold_classical_op,
)
