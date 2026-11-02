# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [integration, optimization, variational]
# ---
#
# # qBraidサポート
#
# このpageではQamomileのqBraid supportを紹介し、`QBraidDevice`を使って
# Qamomile workflowを実行する方法を説明します。QamomileはQiskit artifact形式を
# 介してqBraidと連携するため、基本的な流れは
# `qkernel` → `Qiskit` target → `QBraidDevice`です。
#
# ## このノートで扱う内容
#
# このnotebookは、元のMaxCut workflowを題材にしたqBraid tutorialです。
# 主な目的はqBraid deviceを設定し、qkernelを1つのQPY artifactへcompileし、
# 同じartifactをoptimization loop全体で再利用しながら、最後にqBraidから返る
# sampleを確認する流れを示すことです。
#
# documentationではuser credentialなしにremote jobをsubmitできません。
# そこで、コンパイルしたQPY artifactをQiskit Aerで実行し、
# 元の8-node、5-layer workflow全体を確認してから、credential-dependentな
# device置換を示します。graph構築、continuous parameter optimization、
# 最終energy distribution、partition visualization、SCIP比較も維持します。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install "qamomile[qbraid]" jijmodeling ommx-pyscipopt-adapter
# ```

# %%
# Install the latest Qamomile and its qBraid integration through pip!
# # !pip install "qamomile[qbraid]" jijmodeling ommx-pyscipopt-adapter

# %%
import os
from collections import defaultdict

import jijmodeling as jm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter
from scipy.optimize import minimize

from qamomile.optimization import BinaryModel, QAOAExperiment
from qamomile.persistence import deserialize, serialize
from qamomile.qiskit import Aer, load_circuit

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
SEED = 901
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## MaxCut問題を構築する
#
# 元のチュートリアルと同じ再現可能な8-node 3-regular graphを使います。
# spin $s_i\in\{-1,+1\}$について
#
# $$
# E(\mathbf{s})=\frac12\sum_{(i,j)\in E}s_i s_j-\frac{|E|}{2}
# $$
#
# を最小化することはcut edge数の最大化と等価で、$E=-\mathrm{cut}$です。

# %%
graph = nx.random_regular_graph(d=3, n=8, seed=SEED)
positions = nx.spring_layout(graph, seed=SEED)
EDGES = tuple(sorted(tuple(sorted(edge)) for edge in graph.edges()))

assert graph.number_of_nodes() == 8
assert graph.number_of_edges() == 12

# %%
figure, axis = plt.subplots(figsize=(6, 5))
axis.set_title("Graph")
nx.draw_networkx(
    graph,
    positions,
    ax=axis,
    node_size=500,
    width=2,
    with_labels=True,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5層QAOAを構築・Compileする
#
# `QAOAExperiment`はspin Hamiltonianと5層topologyを1つのtyped programへ閉じます。
# `gammas`と`betas`だけがruntime ABIに残ります。Qiskit targetはdeterministicなQPY
# bytesをemitし、local AerとqBraidのどちらもproblemやansatzの意味を変えません。
# SDK回路の準備は各エンドポイントが担当します。実行先のtargetに対してコンパイルします。

# %%
model = BinaryModel.from_ising(
    {edge: 0.5 for edge in EDGES},
    constant=-len(EDGES) / 2.0,
    variables=range(graph.number_of_nodes()),
)
LAYERS = 5
experiment = QAOAExperiment(model, layers=LAYERS)
endpoint = Aer()
compiled = experiment.compile(to=endpoint.target)
restored_artifact = deserialize(serialize(compiled.artifact))
circuit = load_circuit(restored_artifact)

assert restored_artifact.execution_identity == compiled.artifact.execution_identity
assert circuit.num_qubits == graph.number_of_nodes()
assert circuit.num_parameters == 2 * LAYERS
assert tuple(slot.name for slot in compiled.artifact.parameter_abi.slots) == (
    "gammas",
    "betas",
)

print(circuit.draw(output="text", fold=120))

# %% [markdown]
# ## QAOAパラメータの最適化
#
# 目的関数はコンパイル済みプログラムに`gammas`と`betas`を名前で渡し、Aerエンドポイントでサンプリングします。SDK回路の準備とビット順の正規化はアダプターが担当し、Rustが管理するモデルでスピンのエネルギーを復号・平均します。リモートの例では、同じexperimentをqBraidのtargetに対してコンパイルします。

# %%
OPTIMIZATION_SHOTS = 16 if docs_test_mode else 2_000
MAX_ITERATIONS = 12 if docs_test_mode else 100
energy_history = []


def sample_coordinate(parameters, *, shots: int, seed: int):
    """Sample one aggregate QAOA coordinate and decode its spin energies."""

    record = compiled.run(
        on=endpoint,
        gammas=tuple(float(value) for value in parameters[:LAYERS]),
        betas=tuple(float(value) for value in parameters[LAYERS:]),
        shots=shots,
        seed=seed,
    )
    return record.samples


def objective_function(parameters) -> float:
    """Return and record the sampled mean MaxCut energy."""

    decoded = sample_coordinate(
        parameters,
        shots=OPTIMIZATION_SHOTS,
        seed=SEED + len(energy_history),
    )
    energy_history.append(decoded.mean_energy)
    return decoded.mean_energy


initial_parameters = rng.uniform(-np.pi / 4.0, np.pi / 4.0, 2 * LAYERS)
optimization = minimize(
    objective_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": MAX_ITERATIONS, "rhobeg": 0.5},
)

print(f"Optimized gammas: {optimization.x[:LAYERS]}")
print(f"Optimized betas: {optimization.x[LAYERS:]}")
print(f"Final mean energy: {optimization.fun:+.4f}")

assert len(optimization.x) == 2 * LAYERS
assert len(energy_history) == optimization.nfev

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Energy evaluation")
plt.ylabel("Mean energy")
plt.title("QAOA optimization convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 最適化したDistributionを評価する
#
# 最後の2,000-shot runでenergy分布全体を得て、観測された最低energyのpartitionを
# 選びます。

# %%
FINAL_SHOTS = 128 if docs_test_mode else 2_000
final_samples = sample_coordinate(
    optimization.x,
    shots=FINAL_SHOTS,
    seed=2 * SEED,
)
energy_counts = defaultdict(int)
for sample in final_samples.samples:
    energy_counts[sample.energy] += sample.occurrences

best_sample = final_samples.lowest
best_spins = dict(best_sample.assignment)
best_cut = sum(best_spins[left] != best_spins[right] for left, right in EDGES)

print(f"Best QAOA energy: {best_sample.energy:+.1f}")
print(f"Best QAOA cut: {best_cut}")
print(f"Best QAOA spins: {best_spins}")

assert final_samples.total_occurrences == FINAL_SHOTS
assert best_sample.energy == -float(best_cut)

# %%
energies = sorted(energy_counts, reverse=True)
plt.figure(figsize=(12, 5))
plt.bar(
    np.arange(len(energies)),
    [energy_counts[energy] for energy in energies],
)
plt.xticks(np.arange(len(energies)), energies, rotation=90)
plt.xlabel("Energy")
plt.ylabel("Counts")
plt.title("QAOA measurement frequency distribution")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 最良解を可視化する


# %%
def show_solution(assignment, title: str) -> None:
    """Draw one binary or spin assignment on the MaxCut graph."""

    first_group = {node for node, value in assignment.items() if value in {1, 1.0}}
    edge_colors = [
        "r" if (left in first_group) != (right in first_group) else "b"
        for left, right in EDGES
    ]
    node_colors = [
        "#2696EB" if node in first_group else "#EA9B26" for node in graph.nodes()
    ]
    cut_edges = sum(color == "r" for color in edge_colors)
    _, axis = plt.subplots(figsize=(6, 5))
    axis.set_title(f"{title} (cut edges: {cut_edges})")
    nx.draw_networkx(
        graph,
        positions,
        ax=axis,
        node_size=500,
        width=3,
        with_labels=True,
        edge_color=edge_colors,
        node_color=node_colors,
    )
    plt.tight_layout()
    plt.show()


show_solution(best_spins, "QAOA")

# %% [markdown]
# ## 同じExperimentをqBraidで実行する
#
# remote submissionにはqBraid accountが必要でexternal workを作るため、documentation
# buildでは実行しません。実行先を`QBraidDevice`へ替え、そのtargetに対してコンパイルします。最適化済みpointを次のように
# 1回送信できます。
#
# ```python
# from qbraid import QbraidProvider
# from qamomile.qbraid import QBraidDevice
#
# provider = QbraidProvider()
# endpoint = QBraidDevice(
#     provider.get_device("qbraid:qbraid:sim:qir-sv"),
#     wait_timeout_seconds=120.0,
# )
# remote_compiled = experiment.compile(to=endpoint.target)
# remote_run = remote_compiled.run(
#     on=endpoint,
#     gammas=tuple(float(value) for value in optimization.x[:LAYERS]),
#     betas=tuple(float(value) for value in optimization.x[LAYERS:]),
#     shots=2_000,
#     seed=0,
# )
# print(remote_run.samples.lowest)
# ```
#
# `seed=0`はportable seedを要求しないことを意味します。qBraid deviceごとにseed controlが
# 異なるため、endpointは偽の再現性を約束せずnonzero seedを拒否します。

# %% [markdown]
# ## 古典SCIP比較
#
# JijModelingで同じgraphをMAXIMIZE問題として表します。SCIPのexact partitionをQAOAと
# 並べて表示します。

# %%
classical_problem = jm.Problem("Maxcut", sense=jm.ProblemSense.MAXIMIZE)


@classical_problem.update
def _(problem: jm.DecoratedProblem):
    vertices = problem.Dim()
    edges = problem.Graph()
    x = problem.BinaryVar(shape=(vertices,))
    problem += (
        edges.rows()
        .map(lambda edge: 0.5 * (1 - (2 * x[edge[0]] - 1) * (2 * x[edge[1]] - 1)))
        .sum()
    )


classical_instance = classical_problem.eval(
    {
        "vertices": graph.number_of_nodes(),
        "edges": graph.edges(),
    }
)
scip_solution = OMMXPySCIPOptAdapter.solve(classical_instance)
scip_values = scip_solution.decision_variables_df["value"]
scip_assignment = {
    index: int(round(float(scip_values.loc[index])))
    for index in range(graph.number_of_nodes())
}
scip_cut = int(round(scip_solution.objective))

print(f"SCIP cut: {scip_cut}")
print(f"QAOA cut: {best_cut}")

assert scip_solution.feasible
assert scip_cut >= best_cut

# %%
show_solution(scip_assignment, "SCIP")
show_solution(best_spins, "QAOA")

# %% [markdown]
# ## まとめ
#
# - 元の8-node graphと5-layer QAOAを実行できます。
# - 1つのdeterministic QPY artifactを連続COBYLA最適化と最終2,000-shot評価で再bindします。
# - energy分布、best partition、graph可視化、SCIP比較まで元の解析を再現します。
# - 同じexperimentを`QBraidDevice`のtargetに対してコンパイルして実行できます。認証とリモート送信は明示的に行います。
