# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [usage, chemistry, resource-estimation]
# ---
#
# # FTQC Research Map
#
# This page maps representative fault-tolerant quantum chemistry resource
# estimation papers to Qamomile's canonical resource quantities.
# It is a design guide for reading papers and deciding which quantities to
# model first.
# It is not a live bibliography or a reproduction of the linked estimates.

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import qamomile.resource_estimation as qre

# %% [markdown]
# ## Scope
#
# Recent chemistry resource estimates do not all improve the same layer.
# Some reduce the Hamiltonian normalization that drives phase-estimation
# iterations.
# Others change the time-evolution primitive, the logical operation accounting,
# or the physical architecture lift.
# Qamomile keeps these as separate symbolic quantities so that a comparison can
# say which layer moved.
#
# The following table uses primary arXiv links for a compact research map:
#
# | Research signal | Representative source | Qamomile quantities |
# | --- | --- | --- |
# | Symmetry-compressed factorization lowers the qubitized-QPE work signal | [SCDF](https://arxiv.org/abs/2403.03502) | `lambda_norm`, `walk_cost_toffoli`, `qpe_iterations`, `non_clifford_count` |
# | Improved tensor factorization plus active-volume compilation changes both Hamiltonian and architecture costs | [BLISS-THC and active volume](https://arxiv.org/abs/2501.06165) | `effective_lambda_norm`, `logical_operations`, `active_volume`, `active_volume_runtime_seconds` |
# | Adaptive real-space grids change the representation before QPE or QEVE | [Adaptive grids](https://arxiv.org/abs/2507.20583) | `system_qubits`, `lambda_norm`, `target_precision`, `representation_error` |
# | Unitary weight concentration targets early-FTQC Trotter-QPE costs | [Unitary weight concentration](https://arxiv.org/abs/2603.22778) | `effective_lambda_norm`, `unitary_weight_factor`, `pauli_rotations`, `runtime_seconds` |
#
# :::{note}
# The links are intentionally paper-level sources.
# The executable checks below verify only the Qamomile quantity map, not the
# numerical claims inside those papers.
# :::

# %%
research_signals = [
    {
        "signal": "symmetry-compressed factorization",
        "source": "https://arxiv.org/abs/2403.03502",
        "layer": "hamiltonian compression",
        "quantities": [
            "lambda_norm",
            "walk_cost_toffoli",
            "qpe_iterations",
            "non_clifford_count",
        ],
    },
    {
        "signal": "BLISS-THC with active-volume compilation",
        "source": "https://arxiv.org/abs/2501.06165",
        "layer": "factorization plus architecture lift",
        "quantities": [
            "effective_lambda_norm",
            "logical_operations",
            "active_volume",
            "active_volume_runtime_seconds",
        ],
    },
    {
        "signal": "adaptive real-space grids",
        "source": "https://arxiv.org/abs/2507.20583",
        "layer": "representation choice",
        "quantities": [
            "system_qubits",
            "lambda_norm",
            "target_precision",
            "representation_error",
        ],
    },
    {
        "signal": "unitary weight concentration",
        "source": "https://arxiv.org/abs/2603.22778",
        "layer": "early-FTQC Trotter QPE",
        "quantities": [
            "effective_lambda_norm",
            "unitary_weight_factor",
            "pauli_rotations",
            "runtime_seconds",
        ],
    },
]

quantity_names = {quantity.value for quantity in qre.ResourceQuantity}

for signal in research_signals:
    unsupported = set(signal["quantities"]) - quantity_names
    assert unsupported == set()
    assert signal["source"].startswith("https://arxiv.org/abs/")

print(
    [
        {
            "signal": signal["signal"],
            "layer": signal["layer"],
            "quantity_count": len(signal["quantities"]),
        }
        for signal in research_signals
    ]
)

# %% [markdown]
# ## Quantity Layers
#
# A paper claim should first be placed in the layer it changes.
# The same phrase, such as "runtime reduction", can mean fewer QPE iterations,
# fewer logical non-Clifford operations, or a different physical scheduling
# bottleneck.
# Qamomile's canonical quantity metadata keeps those cases separate.

# %%
quantity_layers = {
    quantity_name: qre.describe_resource_quantity(quantity_name).category.value
    for signal in research_signals
    for quantity_name in signal["quantities"]
}

for quantity_name, category in sorted(quantity_layers.items()):
    print({"quantity": quantity_name, "category": category})

assert quantity_layers["lambda_norm"] == qre.ResourceCategory.PROBLEM
assert quantity_layers["qpe_iterations"] == qre.ResourceCategory.ALGORITHM
assert quantity_layers["non_clifford_count"] == qre.ResourceCategory.LOGICAL
assert quantity_layers["runtime_seconds"] == qre.ResourceCategory.PHYSICAL

# %% [markdown]
# ## Design Boundary
#
# The foundation layer should expose quantities, workloads, and rough physical
# lifts.
# It should not freeze a report format while the modeling surface is still
# settling.
# For now, a reader can keep the paper map as ordinary data and feed the
# selected quantities into `compare_resource_values()` or the workflow in
# [FTQC resource workflow](ftqc_resource_workflow).

# %%
foundation_surfaces = {
    "hamiltonian summaries": [
        "n_qubits",
        "n_pauli_terms",
        "lambda_norm",
        "max_locality",
    ],
    "qubitized QPE workloads": [
        "qpe_iterations",
        "walk_cost_toffoli",
        "logical_qubits",
        "non_clifford_count",
    ],
    "Trotter QPE workloads": [
        "effective_lambda_norm",
        "trotter_steps_per_sample",
        "pauli_rotations",
        "rotation_synthesis_t_gates",
    ],
    "architecture lifts": [
        "physical_qubits",
        "runtime_seconds",
        "physical_qubit_seconds",
        "active_volume",
    ],
}

report_layer_terms = {"profile", "report", "manifest", "pareto", "scenario"}

for surface, quantities in foundation_surfaces.items():
    assert not (report_layer_terms & set(surface.split()))
    assert set(quantities) <= quantity_names

print(foundation_surfaces)

# %% [markdown]
# ## Summary
#
# In this notebook, we learned:
#
# - Read FTQC chemistry papers as changes to a specific modeling layer.
# - Keep paper signals mapped to canonical quantities before adding any report
#   abstraction.
# - Use the workflow page to turn this map into executable symbolic resource
#   comparisons.
