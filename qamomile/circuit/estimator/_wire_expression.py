"""Encode and decode the closed symbolic resource-expression language."""

from __future__ import annotations

import ast
import re
from typing import Any, cast

import sympy as sp
from sympy.core.function import AppliedUndef, FunctionClass
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.logic.boolalg import Boolean

from qamomile.circuit.estimator._clifford_t_decomposition import (
    _CanonicalPhaseClass,
)
from qamomile.circuit.estimator._resource_expressions import (
    _ConditionIndicator,
    _RangeAny,
    _RangeAtLeastTwo,
)
from qamomile.circuit.estimator._serialization import SymbolRegistry
from qamomile.circuit.estimator._symbolic import _CappedRangeSum

_MAX_EXPRESSION_NODES = 100_000
_MAX_NUMERIC_BITS = 4096
# ceil(4096 / log2(10)); bounds decimal mantissa parsing independently from a
# Float's compact decimal exponent.
_MAX_DECIMAL_FLOAT_DIGITS = 1234
_MAX_DECIMAL_FLOAT_EXPONENT_DIGITS = 4
_SAFE_SYMPY_NAMES = frozenset(
    {
        "Abs",
        "Add",
        "And",
        "BooleanFalse",
        "BooleanTrue",
        "Dummy",
        "E",
        "Equality",
        "ExprCondPair",
        "Float",
        "Function",
        "GreaterThan",
        "Integer",
        "Lambda",
        "LessThan",
        "Max",
        "Min",
        "Mod",
        "Mul",
        "Not",
        "Or",
        "Piecewise",
        "Pow",
        "Rational",
        "StrictGreaterThan",
        "StrictLessThan",
        "Sum",
        "Symbol",
        "Tuple",
        "Unequality",
        "Xor",
        "ceiling",
        "cos",
        "exp",
        "floor",
        "log",
        "nan",
        "oo",
        "pi",
        "sign",
        "sin",
        "tan",
        "true",
        "false",
        "zoo",
    }
)


class _WireExpressionEncoder:
    """Canonicalize every expression in one resource wire payload.

    Ordinary symbols retain their public names. Identity-only ``Dummy``
    symbols instead receive payload-local slots in deterministic encounter
    order, removing SymPy's process-random ``dummy_index`` while preserving
    identity across every metric, requirement, and trace expression.

    Args:
        registry (SymbolRegistry): Symbol registry for one resource estimate.
        dummy_slots (dict[sp.Dummy, int] | None): Optional payload-wide mapping
            from source Dummy identities to deterministic slots. Defaults to a
            mapping local to this resource estimate.
    """

    def __init__(
        self,
        registry: SymbolRegistry,
        dummy_slots: dict[sp.Dummy, int] | None = None,
    ) -> None:
        """Build canonical replacements for every registered symbol.

        Args:
            registry (SymbolRegistry): Symbol registry for one resource
                estimate.
            dummy_slots (dict[sp.Dummy, int] | None): Optional payload-wide
                Dummy slot mapping. Defaults to ``None``.
        """
        replacements: dict[sp.Symbol, sp.Symbol] = {}
        resolved_dummy_slots = {} if dummy_slots is None else dummy_slots
        for symbol, public_name in registry.aliases().items():
            if isinstance(symbol, sp.Dummy):
                dummy_slot = resolved_dummy_slots.get(symbol)
                if dummy_slot is None:
                    dummy_slot = len(resolved_dummy_slots)
                    resolved_dummy_slots[symbol] = dummy_slot
                replacement = sp.Dummy(
                    public_name,
                    dummy_index=dummy_slot,
                    **symbol.assumptions0,
                )
            else:
                replacement = sp.Symbol(public_name, **symbol.assumptions0)
            replacements[symbol] = replacement
        self._replacements = replacements

    def encode(self, expression: Any) -> str:
        """Encode one expression with payload-canonical symbol identities.

        Args:
            expression (Any): SymPy-compatible expression.

        Returns:
            str: Deterministic SymPy structural representation.

        Raises:
            ValueError: If the expression cannot be decoded by the matching
                closed wire-expression language.
        """
        normalized = sp.sympify(expression).xreplace(self._replacements)
        _validate_sympy_expression_for_wire(normalized)
        return sp.srepr(normalized)


class _WireExpressionDecoder:
    """Decode Dummy identities from one paired encoder stream.

    A decoder may span multiple resource records only when their encoders
    shared the same ``dummy_slots`` mapping. Independent wire payloads require
    independent decoder instances because their integer slots are local.
    """

    def __init__(self) -> None:
        """Initialize one encoder-stream-local Dummy mapping."""
        self._dummies: dict[int, sp.Dummy] = {}

    def decode(self, payload: Any, label: str) -> sp.Basic:
        """Decode one expression and freshen its canonical Dummy symbols.

        Args:
            payload (Any): Structural SymPy representation string.
            label (str): Diagnostic label.

        Returns:
            sp.Basic: Decoded expression sharing fresh Dummies only within this
                resource payload.

        Raises:
            ValueError: If the expression is outside the supported language.
        """
        expression = _expression_from_wire(payload, label)
        replacements: dict[sp.Dummy, sp.Dummy] = {}
        for symbol in expression.atoms(sp.Dummy):
            slot = cast(int, getattr(symbol, "dummy_index"))
            replacement = self._dummies.get(slot)
            if replacement is None:
                replacement = sp.Dummy(symbol.name, **symbol.assumptions0)
                self._dummies[slot] = replacement
            elif replacement.assumptions0 != symbol.assumptions0:
                raise ValueError(
                    f"{label} assigns conflicting assumptions to Dummy slot {slot}"
                )
            replacements[symbol] = replacement
        return expression.xreplace(replacements)


def _expression_from_wire(payload: Any, label: str) -> sp.Basic:
    """Decode one symbolic expression without evaluating arbitrary Python.

    Args:
        payload (Any): Structural SymPy representation string.
        label (str): Diagnostic label.

    Returns:
        sp.Basic: Reconstructed SymPy expression or Boolean.

    Raises:
        ValueError: If the syntax, constructor, node count, or result type is
            outside the supported closed expression language.
    """
    if not isinstance(payload, str):
        raise ValueError(f"{label} must be a symbolic-expression string")
    try:
        tree = ast.parse(payload, mode="eval")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{label} has invalid symbolic-expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > _MAX_EXPRESSION_NODES:
        raise ValueError(f"{label} exceeds the symbolic-expression node limit")
    result = _evaluate_sympy_ast(tree.body)
    if not isinstance(result, sp.Basic):
        raise ValueError(f"{label} did not decode to a SymPy expression")
    return result


def _validate_sympy_expression_for_wire(expression: sp.Basic) -> None:
    """Validate an existing SymPy tree before persisting its representation.

    Args:
        expression (sp.Basic): Trusted, already-constructed SymPy expression.

    Raises:
        ValueError: If the expression contains an unsupported constructor or
            exceeds a numeric or structural wire budget.
    """
    for count, node in enumerate(sp.preorder_traversal(expression), start=1):
        if count > _MAX_EXPRESSION_NODES:
            raise ValueError(
                "resource expression exceeds the symbolic-expression node limit"
            )
        constructor_name = _wire_constructor_name(node)
        if constructor_name not in _SAFE_SYMPY_NAMES and constructor_name not in {
            "_CanonicalPhaseClass",
            "_ConditionIndicator",
            "_RangeAny",
            "_RangeAtLeastTwo",
            "_CappedRangeSum",
        }:
            raise ValueError(f"unsupported symbolic constructor {constructor_name!r}")
        if isinstance(node, AppliedUndef):
            _validate_undefined_function_constructor([node.func.__name__], {})
        if isinstance(node, sp.Float):
            _validate_float_constructor(
                [str(node)],
                {"precision": node._prec},
            )
        if isinstance(node, sp.Pow):
            _validate_sympy_constructor_call("Pow", list(node.args), {})
        _validate_sympy_numeric_size(node)


def _wire_constructor_name(expression: sp.Basic) -> str:
    """Return the closed-wire constructor name for one SymPy node.

    Args:
        expression (sp.Basic): One node from an expression tree.

    Returns:
        str: Constructor or singleton name emitted by ``sympy.srepr``.
    """
    if isinstance(expression, AppliedUndef):
        return "Function"
    if isinstance(expression, sp.Integer):
        return "Integer"
    if isinstance(expression, sp.Rational):
        return "Rational"
    if isinstance(expression, sp.Float):
        return "Float"
    special_names = (
        (sp.E, "E"),
        (sp.pi, "pi"),
        (sp.nan, "nan"),
        (sp.oo, "oo"),
        (-sp.oo, "oo"),
        (sp.zoo, "zoo"),
    )
    for singleton, name in special_names:
        if expression is singleton:
            return name
    return type(expression).__name__


def _resource_expression_from_wire(
    payload: Any,
    label: str,
    decoder: _WireExpressionDecoder,
) -> sp.Expr:
    """Decode one numeric resource expression.

    Args:
        payload (Any): Structural SymPy representation string.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        sp.Expr: Reconstructed numeric expression.

    Raises:
        ValueError: If the payload decodes to a Boolean or non-expression.
    """
    expression = decoder.decode(payload, label)
    if not isinstance(expression, sp.Expr):
        raise ValueError(f"{label} must decode to a numeric SymPy expression")
    return expression


def _boolean_expression_from_wire(
    payload: Any,
    label: str,
    decoder: _WireExpressionDecoder,
) -> Boolean:
    """Decode one symbolic Boolean guard.

    Args:
        payload (Any): Structural SymPy representation string.
        label (str): Diagnostic label.
        decoder (_WireExpressionDecoder): Shared payload-local expression
            decoder.

    Returns:
        Boolean: Reconstructed Boolean condition.

    Raises:
        ValueError: If the payload does not decode to a SymPy Boolean.
    """
    expression = decoder.decode(payload, label)
    if not isinstance(expression, Boolean):
        raise ValueError(f"{label} must decode to a SymPy Boolean")
    return expression


def _evaluate_sympy_ast(node: ast.AST) -> Any:
    """Evaluate one validated SymPy-constructor AST node.

    Args:
        node (ast.AST): Expression node parsed from ``sympy.srepr`` output.

    Returns:
        Any: Primitive constructor argument, SymPy constructor, or constructed
            SymPy expression.

    Raises:
        ValueError: If the node uses executable Python syntax, an unsupported
            constructor, or invalid constructor arguments.
    """
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(
            node.value,
            (bool, int, float, str),
        ):
            return node.value
        raise ValueError("symbolic expression contains an unsupported literal")
    if isinstance(node, ast.Name):
        return _sympy_name(node.id)
    if isinstance(node, ast.Tuple):
        return tuple(_evaluate_sympy_ast(item) for item in node.elts)
    if isinstance(node, ast.List):
        return [_evaluate_sympy_ast(item) for item in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op,
        (ast.UAdd, ast.USub),
    ):
        operand = _evaluate_sympy_ast(node.operand)
        if not isinstance(operand, (int, float, sp.Expr)):
            raise ValueError("symbolic unary signs require a numeric operand")
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if not isinstance(node, ast.Call):
        raise ValueError(
            "symbolic expression contains syntax outside safe constructors"
        )
    constructor = _evaluate_sympy_ast(node.func)
    if not _is_sympy_constructor(constructor):
        raise ValueError("symbolic expression tried to call a non-constructor")
    if any(keyword.arg is None for keyword in node.keywords):
        raise ValueError("symbolic expression cannot expand keyword mappings")
    args = [_evaluate_sympy_ast(item) for item in node.args]
    kwargs = {
        cast(str, keyword.arg): _evaluate_sympy_ast(keyword.value)
        for keyword in node.keywords
    }
    constructor_name = (
        node.func.id if isinstance(node.func, ast.Name) else "_AppliedUndefinedFunction"
    )
    _validate_sympy_constructor_call(constructor_name, args, kwargs)
    if constructor_name == "_CappedRangeSum":
        kwargs = {**kwargs, "evaluate": False}
    try:
        result = constructor(*args, **kwargs)
    except (TypeError, ValueError, sp.SympifyError) as exc:
        raise ValueError(
            "symbolic expression constructor arguments are invalid"
        ) from exc
    if not isinstance(result, (sp.Basic, FunctionClass)):
        raise ValueError("symbolic constructor produced an unsupported result")
    if isinstance(result, sp.Basic):
        _validate_sympy_numeric_size(result)
    return result


def _validate_sympy_constructor_call(
    name: str,
    args: list[Any],
    kwargs: dict[str, Any],
) -> None:
    """Reject constructor inputs that can trigger unbounded eager arithmetic.

    Args:
        name (str): Validated SymPy constructor name.
        args (list[Any]): Recursively decoded positional arguments.
        kwargs (dict[str, Any]): Recursively decoded keyword arguments.

    Raises:
        ValueError: If a numeric constructor request exceeds the wire budget.
    """
    if name == "Float":
        _validate_float_constructor(args, kwargs)
    if name == "Function":
        _validate_undefined_function_constructor(args, kwargs)
    if name == "_AppliedUndefinedFunction":
        if kwargs or not all(isinstance(arg, sp.Basic) for arg in args):
            raise ValueError(
                "symbolic applied functions require SymPy positional arguments"
            )
    if name == "_CappedRangeSum" and (len(args) != 4 or kwargs):
        raise ValueError("symbolic _CappedRangeSum requires four positional arguments")
    if name != "Pow" or len(args) < 2:
        return
    base, exponent = args[:2]
    if not isinstance(exponent, sp.Integer):
        return
    exponent_value = int(exponent)
    if not isinstance(base, (sp.Integer, sp.Rational)):
        return
    if base in (sp.Integer(-1), sp.Integer(0), sp.Integer(1)):
        return
    magnitude = max(
        abs(int(base.p)).bit_length(),
        abs(int(base.q)).bit_length(),
    )
    if magnitude * max(1, abs(exponent_value)) > _MAX_NUMERIC_BITS:
        raise ValueError("symbolic numeric power exceeds the wire budget")


def _validate_undefined_function_constructor(
    args: list[Any],
    kwargs: dict[str, Any],
) -> None:
    """Restrict undefined functions to canonical bounded identifiers.

    Args:
        args (list[Any]): Function-factory positional arguments.
        kwargs (dict[str, Any]): Function-factory keyword arguments.

    Raises:
        ValueError: If the factory request is not ``Function(identifier)``.
    """
    if len(args) != 1 or kwargs or not isinstance(args[0], str):
        raise ValueError("symbolic Function requires one identifier string")
    if re.fullmatch(r"[A-Za-z_]\w{0,255}", args[0], flags=re.ASCII) is None:
        raise ValueError("symbolic Function name must be a bounded identifier")


def _validate_float_constructor(args: list[Any], kwargs: dict[str, Any]) -> None:
    """Reject Float payloads whose parsing work exceeds the wire budget.

    The encoder's structural representation always uses one decimal string
    plus an optional ``precision`` keyword. Restricting the decoder to that
    canonical shape prevents positional ``dps`` or enormous exponent text from
    triggering expensive arbitrary-precision construction. A compact exponent
    may describe an arbitrarily small or large finite value without requiring
    a proportionally large mantissa, so its magnitude is not charged as
    decimal digits.

    Args:
        args (list[Any]): Recursively decoded Float positional arguments.
        kwargs (dict[str, Any]): Recursively decoded Float keyword arguments.

    Raises:
        ValueError: If the Float is noncanonical or its representation exceeds
            the wire budget.
    """
    if len(args) != 1 or not isinstance(args[0], str):
        raise ValueError("symbolic Float requires one decimal string")
    if set(kwargs) - {"precision"}:
        raise ValueError("symbolic Float contains unsupported keyword arguments")
    precision = kwargs.get("precision", 53)
    if not isinstance(precision, int) or isinstance(precision, bool):
        raise ValueError("symbolic Float precision must be an integer")
    if precision < 1 or precision > _MAX_NUMERIC_BITS:
        raise ValueError("symbolic Float precision exceeds the wire budget")

    literal = args[0]
    if len(literal) > _MAX_DECIMAL_FLOAT_DIGITS + 16:
        raise ValueError("symbolic Float literal exceeds the wire budget")
    match = re.fullmatch(
        r"[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE]([+-]?\d+))?",
        literal,
    )
    if match is None:
        raise ValueError("symbolic Float literal is malformed")
    digits = sum(
        character.isdigit() for character in literal.split("e")[0].split("E")[0]
    )
    exponent_text = match.group(1)
    if exponent_text is not None:
        exponent_digits = exponent_text.lstrip("+-")
        if len(exponent_digits) > _MAX_DECIMAL_FLOAT_EXPONENT_DIGITS:
            raise ValueError("symbolic Float exponent exceeds the wire budget")
    if digits > _MAX_DECIMAL_FLOAT_DIGITS:
        raise ValueError("symbolic Float mantissa exceeds the wire budget")


def _validate_sympy_numeric_size(expression: sp.Basic) -> None:
    """Reject an exact numeric result that exceeds the wire arithmetic budget.

    Args:
        expression (sp.Basic): Newly constructed SymPy expression or number.

    Raises:
        ValueError: If an exact rational result exceeds the bit limit.
    """
    if not isinstance(expression, sp.Rational):
        return
    if (
        max(
            abs(int(expression.p)).bit_length(),
            abs(int(expression.q)).bit_length(),
        )
        > _MAX_NUMERIC_BITS
    ):
        raise ValueError("symbolic numeric value exceeds the wire budget")


def _sympy_name(name: str) -> Any:
    """Resolve one safe constructor or symbolic constant by name.

    Args:
        name (str): Name emitted by ``sympy.srepr``.

    Returns:
        Any: SymPy ``Basic`` constant or expression constructor.

    Raises:
        ValueError: If the name is not a safe SymPy expression constructor.
    """
    if name == "ExprCondPair":
        return ExprCondPair
    if name == "_CanonicalPhaseClass":
        return _CanonicalPhaseClass
    if name == "_ConditionIndicator":
        return _ConditionIndicator
    if name == "_RangeAny":
        return _RangeAny
    if name == "_RangeAtLeastTwo":
        return _RangeAtLeastTwo
    if name == "_CappedRangeSum":
        return _CappedRangeSum
    if name not in _SAFE_SYMPY_NAMES:
        raise ValueError(f"unsupported symbolic constructor {name!r}")
    candidate = getattr(sp, name, None)
    if isinstance(candidate, sp.Basic) or _is_sympy_constructor(candidate):
        return candidate
    raise ValueError(f"unsupported symbolic constructor {name!r}")


def _is_sympy_constructor(value: Any) -> bool:
    """Return whether a value is a safe SymPy expression constructor.

    Args:
        value (Any): Candidate object.

    Returns:
        bool: Whether the object can only construct ``sympy.Basic`` values.
    """
    return isinstance(value, FunctionClass) or (
        isinstance(value, type) and issubclass(value, sp.Basic)
    )
