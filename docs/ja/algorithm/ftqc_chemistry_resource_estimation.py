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
# tags: [algorithm, chemistry, resource-estimation, simulation]
# ---
#
# # FTQC化学計算のリソース推定
#
# このノートブックでは、Qamomileでfault-tolerantな量子化学計算のリソースモデルを比較する方法を示します。完全な論理回路がまだない段階で重要になる、Hamiltonian normalization、phase-estimation iterations、ToffoliまたはT counts、論理量子ビット、物理量子ビット、runtime proxiesに注目します。

# %%
# 最新のQamomileをpipでインストールします！
# # !pip install qamomile

# %%
import sympy as sp

from qamomile.circuit.estimator.algorithmic import (
    ChemistryQPEMethod,
    FTQCCostModel,
    estimate_qubitized_chemistry_qpe,
    estimate_single_ancilla_trotter_qpe,
)

# %% [markdown]
# ## Background
#
# Fault-tolerantな化学計算アルゴリズムは、NISQのvariationalな例とは異なるリソース量で比較されることが多くあります。qubitized QPEでは、Hamiltonian block-encoding normalizationがwalk callsの回数を決め、per-walk implementationがToffoli countを決めます。近年の化学計算向け提案では、backend-levelのgate decompositionだけを変えるのではなく、Hamiltonian representationを変えることでこれらのコストを下げようとします。
#
# 例として、symmetry-compressed double factorization([arXiv:2403.03502](https://arxiv.org/abs/2403.03502))、simultaneous symmetry shifts and tensor factorizations([arXiv:2412.01338](https://arxiv.org/abs/2412.01338))、unitary weight concentrationを使うearly-FTQC single-ancilla QPE([arXiv:2603.22778](https://arxiv.org/abs/2603.22778))があります。以下のEstimatorは意図的にsymbolicです。具体的なHamiltonian-loading circuitに進む前に、提案したrepresentationがコストを支配する量をどう変えるかを確認できます。

# %% [markdown]
# ## Problem Settings
#
# 同じactive-space scaleについて、次の3つのシナリオを比較します。
#
# 1. tensor-hypercontraction-likeなqubitized QPE baseline、
# 2. symmetry-compressed double-factorization-styleなqubitized QPE estimate、
# 3. unitary-weight concentration factorを持つearly-FTQC single-ancilla Trotter QPE estimate。
#
# 数値は小さく、人工的に選んでいます。このノートブックを高速でレビューしやすくするためです。特定の分子についての主張ではなく、workflowのデモとして読んでください。

# %%
n_spin_orbitals = 40
precision = sp.Float("0.0016")  # Hartree単位のおおよそのchemical accuracy

cost_model = FTQCCostModel(
    physical_qubits_per_logical=800,
    logical_cycle_time_seconds=sp.Float("1e-6"),
    factory_qubits=20000,
    toffoli_throughput_per_second=sp.Float("2e6"),
)

# %% [markdown]
# ## Qubitized QPE Comparison
#
# Qamomileはchemistry factorization costをIRの外側に保ちます。このEstimatorは、representation-dependent normalizationとone-walk Toffoli costを入力として受け取ります。
#
# ```text
# qpe_iterations = lambda_norm / precision
# toffoli_gates = qpe_iterations * walk_cost_toffoli
# ```
#
# これによりモデルを明確に保てます。Hamiltonian normalizationを下げること、walk circuitを下げること、physical architectureを変えることは、それぞれ別の設計選択です。

# %%
thc = estimate_qubitized_chemistry_qpe(
    n_spin_orbitals=n_spin_orbitals,
    lambda_norm=sp.Float("2.0e5"),
    precision=precision,
    walk_cost_toffoli=sp.Integer(4_000),
    method=ChemistryQPEMethod.TENSOR_HYPERCONTRACTION,
    cost_model=cost_model,
)

scdf = estimate_qubitized_chemistry_qpe(
    n_spin_orbitals=n_spin_orbitals,
    lambda_norm=sp.Float("1.0e5"),
    precision=precision,
    walk_cost_toffoli=sp.Integer(4_400),
    method=ChemistryQPEMethod.SYMMETRY_COMPRESSED_DF,
    second_factor_rank=9,
    cost_model=cost_model,
)

assert scdf.qpe_iterations < thc.qpe_iterations
assert scdf.toffoli_gates < thc.toffoli_gates

print("THC Toffoli gates:", sp.N(thc.toffoli_gates, 4))
print("SCDF-style Toffoli gates:", sp.N(scdf.toffoli_gates, 4))
print("SCDF-style logical qubits:", scdf.logical_qubits)

# %% [markdown]
# ## Early-FTQC Trotter QPE
#
# Early-FTQCの提案では、量子ビット数とdepthが強く制限される場合、qubitized walksよりも浅いsingle-ancilla QPEとPauli rotationsが好まれることがあります。下のunitary-weight concentration factorは、コストを支配するeffective weightを下げるspectrally invariantなHamiltonian transformationを表します。

# %%
plain_trotter = estimate_single_ancilla_trotter_qpe(
    n_spin_orbitals=n_spin_orbitals,
    n_pauli_terms=20_000,
    lambda_norm=sp.Float("2.0e5"),
    precision=precision,
    trotter_steps_per_sample=8,
    samples=128,
    cost_model=cost_model,
)

uwc_trotter = estimate_single_ancilla_trotter_qpe(
    n_spin_orbitals=n_spin_orbitals,
    n_pauli_terms=20_000,
    lambda_norm=sp.Float("2.0e5"),
    precision=precision,
    trotter_steps_per_sample=8,
    samples=128,
    unitary_weight_factor=sp.Float("0.1"),
    randomized_compilation_factor=sp.Float("0.5"),
    rotation_synthesis_t_gates=3,
    cost_model=cost_model,
)

assert uwc_trotter.qpe_iterations < plain_trotter.qpe_iterations
assert uwc_trotter.logical_depth < plain_trotter.logical_depth

print("Plain Trotter QPE depth proxy:", sp.N(plain_trotter.logical_depth, 4))
print("UWC-style Trotter QPE depth proxy:", sp.N(uwc_trotter.logical_depth, 4))
print("UWC-style T gates:", sp.N(uwc_trotter.t_gates, 4))

# %% [markdown]
# ## Result
#
# 推定結果を小さな表にまとめます。重要なのは、各列が別々の設計上の意味を持つことです。Hamiltonian representationの変更は`qpe_iterations`とper-step costに効くべきで、hardware modelの変更は`physical_qubits`とruntimeに効くべきです。

# %%
rows = [
    ("THC qubitized QPE", thc),
    ("SCDF-style qubitized QPE", scdf),
    ("Plain Trotter QPE", plain_trotter),
    ("UWC-style Trotter QPE", uwc_trotter),
]

for name, estimate in rows:
    print(
        name,
        {
            "logical_qubits": sp.N(estimate.logical_qubits, 4),
            "physical_qubits": sp.N(estimate.physical_qubits, 4),
            "qpe_iterations": sp.N(estimate.qpe_iterations, 4),
            "toffoli_gates": sp.N(estimate.toffoli_gates, 4),
            "t_gates": sp.N(estimate.t_gates, 4),
            "runtime_seconds": sp.N(estimate.runtime_seconds, 4),
        },
    )

assert scdf.physical_qubits > thc.physical_qubits
assert uwc_trotter.physical_qubits == plain_trotter.physical_qubits

# %% [markdown]
# ## Summary
#
# このノートブックでは、次のことを確認しました。
#
# - FTQC化学計算のリソース量を、Hamiltonian normalization、QPE iterations、non-Clifford counts、論理量子ビット、物理量子ビット、runtime proxiesに分けて扱いました。
# - Qamomile IRをbackend-specificなchemistry loading circuitsへloweringせずに、qubitized QPE representationsを比較しました。
# - unitary-weight concentration factorを、early-FTQC single-ancilla Trotter QPEのcost-driver reductionとしてモデル化する方法を示しました。
