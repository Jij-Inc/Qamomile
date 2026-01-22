import dataclasses
from typing import TYPE_CHECKING

from qamomile.circuit.ir.operation import Operation
from qamomile.circuit.ir.value import Value

if TYPE_CHECKING:
    pass

    # from qamomile.estimator.cost import SynthesisConfig
    # from qamomile.estimator.resource_estimator import (
    #     HierarchicalResourceCost,
    #     ResourceCost,
    #     SubstitutedResourceCost,
    # )


@dataclasses.dataclass
class Graph:
    """Represents a traced computation graph.

    A Graph is the result of building a QKernel with concrete parameters.
    It contains the list of operations and can be used for:
    - Resource estimation
    - Transpilation to various backends
    - Visualization
    """

    operations: list[Operation]
    input_values: list[Value] = dataclasses.field(default_factory=list)
    output_values: list[Value] = dataclasses.field(default_factory=list)
    name: str = ""
    parameters: dict[str, Value] = dataclasses.field(default_factory=dict)

    def unbound_parameters(self) -> list[str]:
        """Return list of unbound parameter names."""
        return list(self.parameters.keys())

    # def estimate(
    #     self, binding: dict[str, int | float | bool] | None = None
    # ) -> "ResourceCost | SubstitutedResourceCost":
    #     """Estimate resources required by this graph (basic qubit counting).

    #     If binding is provided and resolves all symbolic dimensions,
    #     returns SubstitutedResourceCost with concrete int values.
    #     Otherwise returns ResourceCost with symbolic SymPy expressions.

    #     For detailed gate counting and T-count estimation, use estimate_hierarchical().

    #     Args:
    #         binding: Optional dictionary mapping symbolic dimension names to values.

    #     Returns:
    #         ResourceCost (symbolic) or SubstitutedResourceCost (concrete).

    #     Example:
    #         ```python
    #         # Symbolic estimation
    #         cost = graph.estimate()
    #         print(cost.num_qubits)  # n (SymPy symbol)

    #         # Concrete estimation
    #         cost = graph.estimate({"n": 10})
    #         print(cost.num_qubits)  # 10
    #         ```
    #     """
    #     from qamomile.estimator.resource_estimator import (
    #         estimate_from_ops,
    #         subs_resources,
    #     )

    #     if binding is None:
    #         binding = {}
    #     raw_cost = estimate_from_ops(self.operations, self.input_values)
    #     return subs_resources(raw_cost, binding)

    # def estimate_hierarchical(
    #     self,
    #     decomposition_level: str = "none",
    #     binding: Dict[str, int | float] | None = None,
    # ) -> "HierarchicalResourceCost":
    #     """Perform hierarchical resource estimation.

    #     This method estimates resources while preserving loop structure
    #     and symbolic expressions. It provides detailed gate counts,
    #     T-count, and other metrics for fault-tolerant resource estimation.

    #     Args:
    #         decomposition_level: How deep to decompose composite gates.
    #             - "none": Keep composite gates as-is (QPE, QFT counted directly)
    #             - "standard": Decompose to standard gates (H, CX, CP, etc.)
    #             - "clifford_t": Decompose to Clifford+T gate set
    #             - "primitive": Full decomposition to primitives
    #         binding: Parameter bindings for symbolic values.

    #     Returns:
    #         HierarchicalResourceCost with symbolic gate counts, T-count, etc.

    #     Example:
    #         ```python
    #         graph = my_kernel.build()

    #         # Get symbolic costs
    #         cost = graph.estimate_hierarchical()
    #         print(f"T-count (symbolic): {cost.t_count}")

    #         # Substitute concrete values
    #         concrete_cost = cost.substitute({"n": 10})
    #         print(f"T-count (n=10): {concrete_cost.t_count}")

    #         # Or include binding directly
    #         cost = graph.estimate_hierarchical(binding={"n": 10})
    #         ```
    #     """
    #     from qamomile.estimator.resource_estimator import HierarchicalEstimator

    #     estimator = HierarchicalEstimator(decomposition_level)
    #     cost = estimator.estimate(self.operations)

    #     if binding:
    #         cost = cost.substitute(binding)

    #     return cost

    # def call_graph(self) -> Dict[str, Dict[str, "sp.Expr"]]:
    #     """Build the call graph of operations.

    #     The call graph shows which operations call which sub-operations
    #     and their counts. This is useful for understanding algorithm
    #     structure without full decomposition.

    #     Returns:
    #         Dict mapping operation names to their callee counts.

    #     Example:
    #         ```python
    #         graph = qpe_kernel.build()
    #         cg = graph.call_graph()
    #         # cg might be:
    #         # {
    #         #     "QPE": {"H": n, "Controlled-U": 2**n - 1, "iQFT": 1},
    #         #     "iQFT": {"H": n, "CP": n*(n-1)/2}
    #         # }
    #         ```
    #     """
    #     from qamomile.estimator.resource_estimator import HierarchicalEstimator

    #     estimator = HierarchicalEstimator()
    #     cost = estimator.estimate(self.operations)
    #     return cost.call_graph
