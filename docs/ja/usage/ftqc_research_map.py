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
# # FTQC研究マップ
#
# このページでは、fault-tolerantな量子化学リソース推定の代表的な論文を、Qamomileのcanonical resource quantityへ対応づけます。
# 論文を読み、最初にどのquantityをmodel化するかを決めるための設計ガイドです。
# リンク先の推定値を再現するページでも、網羅的な文献リストでもありません。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile

# %%
import qamomile.resource_estimation as qre

# %% [markdown]
# ## 対象範囲
#
# 近年の量子化学リソース推定は、同じ層を改善しているわけではありません。
# phase-estimation iterationを決めるHamiltonian normalizationを下げる研究もあります。
# time-evolution primitive、論理operation accounting、物理architecture liftを変える研究もあります。
# Qamomileでは、比較でどの層が動いたかを言えるように、これらを別々のsymbolic quantityとして扱います。
#
# 次の表は、primary sourceであるarXivへのリンクを使った小さな研究マップです。
#
# | Research signal | Representative source | Qamomile quantity |
# | --- | --- | --- |
# | Symmetry-compressed factorizationがqubitized QPEのwork signalを下げる | [SCDF](https://arxiv.org/abs/2403.03502) | `lambda_norm`, `walk_cost_toffoli`, `qpe_iterations`, `non_clifford_count` |
# | 改良されたtensor factorizationとactive-volume compilationがHamiltonian costとarchitecture costを変える | [BLISS-THC and active volume](https://arxiv.org/abs/2501.06165) | `effective_lambda_norm`, `logical_operations`, `active_volume`, `active_volume_runtime_seconds` |
# | Adaptive real-space gridがQPEまたはQEVEの前のrepresentationを変える | [Adaptive grids](https://arxiv.org/abs/2507.20583) | `system_qubits`, `lambda_norm`, `target_precision`, `representation_error` |
# | Unitary weight concentrationがearly-FTQC Trotter QPEのcostを対象にする | [Unitary weight concentration](https://arxiv.org/abs/2603.22778) | `effective_lambda_norm`, `unitary_weight_factor`, `pauli_rotations`, `runtime_seconds` |
#
# :::{note}
# リンクは論文単位のsourceを指します。
# 下の実行可能なcheckはQamomile quantity mapだけを検証し、論文中の数値主張は検証しません。
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
# ## Quantity layer
#
# 論文中の主張は、最初に「どの層を変える主張か」に配置します。
# たとえば「runtime reduction」という同じ言い方でも、QPE iterationが減る場合、論理non-Clifford operationが減る場合、物理scheduling bottleneckが変わる場合があります。
# Qamomileのcanonical quantity metadataは、これらを分けて扱います。

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
# ## 設計上の境界
#
# foundation layerは、quantity、workload、roughなphysical liftを公開します。
# modeling surfaceがまだ固まっていない段階で、report formatまで固定する必要はありません。
# 今はpaper mapを通常のdataとして保ち、選んだquantityを`compare_resource_values()`または[FTQCリソースワークフロー](ftqc_resource_workflow)へ渡せば十分です。

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
# ## まとめ
#
# このnotebookでは、次のことを確認しました。
#
# - FTQC chemistry paperを、特定のmodeling layerへの変更として読みます。
# - report abstractionを足す前に、paper signalをcanonical quantityへ対応づけます。
# - このmapを実行可能なsymbolic resource comparisonにするには、workflow pageを使います。
