"""Estimate logical resources for block-encoding QPE workloads."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import sympy as sp

from qamomile.circuit.estimator import GateCount, ResourceEstimate
from qamomile.resource_estimation._common import (
    _as_expr,
    _convert_fields,
    _SympyLike,
    _validate_nonnegative,
    _validate_positive,
)


@dataclass(frozen=True)
class BlockEncodingResource:
    """Describe a block-encoding resource contract for QPE estimates.

    The model records the logical footprint and the reusable subroutine costs
    for a block encoding. It does not lower PREPARE, SELECT, or reflection
    into Qamomile IR; those implementation details remain outside this
    abstract resource contract.

    Args:
        system_qubits (sp.Expr | int | float): Logical system-register qubits.
        normalization (sp.Expr | int | float): Block-encoding normalization
            alpha such that the encoded operator represents ``H / alpha``.
        select_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            SELECT or oracle application.
        prepare_cost_toffoli (sp.Expr | int | float): Toffoli cost for one
            PREPARE or inverse-PREPARE call. Defaults to 0.
        reflection_cost_toffoli (sp.Expr | int | float): Toffoli cost for the
            reflection in one qubitized walk. Defaults to 0.
        ancilla_qubits (sp.Expr | int | float): Logical ancilla and workspace
            qubits used by the block encoding, excluding QPE readout qubits.
            Defaults to 0.
        name (str): Reader-facing label for this block-encoding model.

    Raises:
        ValueError: If a positive-valued quantity is non-positive, or if a
            nonnegative-valued quantity is negative.

    Example:
        >>> block = BlockEncodingResource(
        ...     system_qubits=10,
        ...     normalization=100,
        ...     prepare_cost_toffoli=20,
        ...     select_cost_toffoli=50,
        ... )
        >>> block.walk_cost_toffoli
        90
    """

    system_qubits: sp.Expr
    normalization: sp.Expr
    select_cost_toffoli: sp.Expr
    prepare_cost_toffoli: sp.Expr = sp.Integer(0)
    reflection_cost_toffoli: sp.Expr = sp.Integer(0)
    ancilla_qubits: sp.Expr = sp.Integer(0)
    name: str = "block_encoding"

    def __post_init__(self) -> None:
        """Convert and validate block-encoding quantities after construction.

        Numeric fields are sympified once here, so later accesses see
        ``sp.Expr`` values without per-access conversion.

        Raises:
            TypeError: If a field cannot be converted to a SymPy expression.
            ValueError: If a positive-valued quantity is non-positive, or if a
                nonnegative-valued quantity is negative.
        """
        _convert_fields(
            self,
            (
                "system_qubits",
                "normalization",
                "select_cost_toffoli",
                "prepare_cost_toffoli",
                "reflection_cost_toffoli",
                "ancilla_qubits",
            ),
        )
        _validate_positive(self.system_qubits, "system_qubits")
        _validate_positive(self.normalization, "normalization")
        for name, expr in [
            ("select_cost_toffoli", self.select_cost_toffoli),
            ("prepare_cost_toffoli", self.prepare_cost_toffoli),
            ("reflection_cost_toffoli", self.reflection_cost_toffoli),
            ("ancilla_qubits", self.ancilla_qubits),
        ]:
            _validate_nonnegative(expr, name)

    @cached_property
    def logical_qubits(self) -> sp.Expr:
        """Return block-encoding logical qubits before QPE readout.

        Returns:
            sp.Expr: ``system_qubits + ancilla_qubits``.
        """
        return sp.simplify(self.system_qubits + self.ancilla_qubits)

    @cached_property
    def walk_cost_toffoli(self) -> sp.Expr:
        """Return the Toffoli cost of one qubitized walk.

        Returns:
            sp.Expr: ``2 * prepare_cost_toffoli + select_cost_toffoli +
            reflection_cost_toffoli``.
        """
        return sp.simplify(
            2 * self.prepare_cost_toffoli
            + self.select_cost_toffoli
            + self.reflection_cost_toffoli
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the block-encoding contract.

        Returns:
            dict[str, Any]: JSON-friendly block-encoding metadata.
        """
        return {
            "name": self.name,
            "system_qubits": str(self.system_qubits),
            "normalization": str(self.normalization),
            "select_cost_toffoli": str(self.select_cost_toffoli),
            "prepare_cost_toffoli": str(self.prepare_cost_toffoli),
            "reflection_cost_toffoli": str(self.reflection_cost_toffoli),
            "ancilla_qubits": str(self.ancilla_qubits),
            "logical_qubits": str(self.logical_qubits),
            "walk_cost_toffoli": str(self.walk_cost_toffoli),
        }

    def resource_values(self) -> dict[str, sp.Expr]:
        """Return canonical resource values exposed by the block encoding.

        Returns:
            dict[str, sp.Expr]: Resource values keyed by stable quantity names.
        """
        return {
            "system_qubits": self.system_qubits,
            "lambda_norm": self.normalization,
            "select_cost_toffoli": self.select_cost_toffoli,
            "prepare_cost_toffoli": self.prepare_cost_toffoli,
            "reflection_cost_toffoli": self.reflection_cost_toffoli,
            "block_encoding_ancilla_qubits": self.ancilla_qubits,
            "logical_qubits": self.logical_qubits,
            "walk_cost_toffoli": self.walk_cost_toffoli,
        }


def estimate_qubitized_qpe_resources_from_block_encoding(
    block_encoding: BlockEncodingResource,
    precision: _SympyLike,
    *,
    qpe_register_qubits: _SympyLike = 0,
) -> ResourceEstimate:
    """Estimate logical qubitized-QPE resources from a block encoding.

    Args:
        block_encoding (BlockEncodingResource): Block-encoding contract with
            logical footprint, normalization, and walk subroutine costs.
        precision (sp.Expr | int | float): Target precision in the same units
            as ``block_encoding.normalization``.
        qpe_register_qubits (sp.Expr | int | float): Optional QPE readout
            register qubits added to the block-encoding footprint. Defaults
            to 0.

    Returns:
        ResourceEstimate: Architecture-independent logical resource estimate.

    Raises:
        TypeError: If ``block_encoding`` is not a ``BlockEncodingResource``.
        ValueError: If ``precision`` is non-positive or if
            ``qpe_register_qubits`` is negative.
    """
    if not isinstance(block_encoding, BlockEncodingResource):
        raise TypeError("block_encoding must be a BlockEncodingResource.")

    precision_expr = _as_expr(precision, "precision")
    qpe_qubits = _as_expr(qpe_register_qubits, "qpe_register_qubits")
    _validate_positive(precision_expr, "precision")
    _validate_nonnegative(qpe_qubits, "qpe_register_qubits")

    qpe_iterations = sp.simplify(block_encoding.normalization / precision_expr)
    toffoli_gates = sp.simplify(qpe_iterations * block_encoding.walk_cost_toffoli)
    logical_qubits = sp.simplify(block_encoding.logical_qubits + qpe_qubits)

    return ResourceEstimate(
        qubits=logical_qubits,
        gates=GateCount(
            total=toffoli_gates,
            single_qubit=sp.Integer(0),
            two_qubit=sp.Integer(0),
            multi_qubit=toffoli_gates,
            t_gates=sp.Integer(0),
            clifford_gates=sp.Integer(0),
            rotation_gates=sp.Integer(0),
            oracle_calls={"qpe_iterations": qpe_iterations},
            oracle_queries={},
        ),
    ).simplify()
