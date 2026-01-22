# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: qamomile
# ---


# %% [markdown]
# # Quantum Approximate Optimization Algorithm (QAOA)
# このチュートリアルでは、Qamomileを使用して量子近似最適化アルゴリズム（QAOA）を実装する方法を説明します。
# QAOAは量子アニーリングからインスパイアされたハイブリッド量子古典アルゴリズムであり、組合せ最適化問題のためのヒューリスティクスです。
# またQAOAではパラメータ化された量子回路を使用し、古典的な最適化手法でパラメータを調整します。
# このチュートリアルではQAOAを通してどのようにQamomileでパラメーター化された量子回路を構築し、最適化するかを学びます。

# %% [markdown]
# ## QAOAの基本概念
# QAOAは以下の主要なステップで構成されます：
# 1. **問題の定式化**: 最適化したい組合せ最適化問題を定式化します。例えば、最大カット問題などです。
# 2. **量子回路の構築**: 問題に基づいてパラメータ化された量子回路を構築します。これにはコストハミルトニアンとミキサーハミルトニアンの適用が含まれます。
# 3. **測定**: 量子回路を実行し、結果を測定します。
# 4. **古典的最適化**: 測定結果に基づいてパラメータを更新し、量子回路を再構築します。
# 5. **反復**: 最適な解が得られるまでステップ2から4を繰り返します。

# %% [markdown]
# ## スクラッチでのQAOA実装
# まずはQamomileの基本的な量子ゲートを使用してQAOAを実装してみましょう。
# 適当なランダムイジングモデルのエネルギー最小化問題を解く例を示します。

# %%
import numpy as np

import qamomile.circuit as qmc


@qmc.qkernel
def qaoa_cost_operator(
    qubits: qmc.Vector[qmc.Qubit],
    edges: qmc.Matrix[qmc.UInt],
    weights: qmc.Vector[qmc.Float],
    bias: qmc.Vector[qmc.Float],
    gamma: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    e = edges.shape[0]
    for _e in qmc.range(e):
        i = edges[_e, 0]
        j = edges[_e, 1]
        wij = weights[_e]
        qubits[i], qubits[j] = qmc.rzz(qubits[i], qubits[j], angle=gamma * wij)

    n = qubits.shape[0]
    for i in qmc.range(n):
        bi = bias[i]
        qubits[i] = qmc.rz(qubits[i], angle=gamma * bi)
    return qubits


# %% [markdown]
# 次にQAOAのMixerオペレーターを定義します。


# %%
@qmc.qkernel
def qaoa_mixer_operator(
    qubits: qmc.Vector[qmc.Qubit],
    beta: qmc.Float,
) -> qmc.Vector[qmc.Qubit]:
    n = qubits.shape[0]
    for i in qmc.range(n):
        qubits[i] = qmc.rx(qubits[i], angle=2 * beta)
    return qubits


# %% [markdown]
# 最後にQAOA回路全体を定義します。


# %%
@qmc.qkernel
def qaoa_circuit(
    edges: qmc.Matrix[qmc.UInt],
    weights: qmc.Vector[qmc.Float],
    bias: qmc.Vector[qmc.Float],
    p: int,
    gammas: qmc.Vector[qmc.Float],
    betas: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    n = bias.shape[0]
    qubits = qmc.qubit_array(n, name="qaoa_qubits")

    # 初期状態の準備（均一重ね合わせ状態）
    for i in qmc.range(n):
        qubits[i] = qmc.h(qubits[i])

    # QAOAレイヤーの適用
    for layer in qmc.range(p):
        qubits = qaoa_cost_operator(qubits, edges, weights, bias, gammas[layer])
        qubits = qaoa_mixer_operator(qubits, betas[layer])

    return qmc.measure(qubits)


# %% [markdown]
# ## 異なる量子SDKでのQAOA実行
#
# Qamomileは複数の量子SDKをサポートしています。同じ回路定義がすべてのバックエンドで動作します。
# お好みのSDKを選択してください:
#
# ::::{tab-set}
# :::{tab-item} Qiskit
# :sync: sdk
#
# ```python
# from qamomile.qiskit import QiskitTranspiler
#
# transpiler = QiskitTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# :::{tab-item} Quri-Parts
# :sync: sdk
#
# ```python
# from qamomile.quri_parts import QuriPartsCircuitTranspiler
#
# transpiler = QuriPartsCircuitTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# # シミュレーションにはquri-parts-qulacsが必要
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# :::{tab-item} PennyLane
# :sync: sdk
#
# ```python
# from qamomile.pennylane import PennylaneTranspiler
#
# transpiler = PennylaneTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# :::{tab-item} CUDA-Q
# :sync: sdk
#
# ```{note}
# CUDA-QはNVIDIA GPUを搭載したLinuxシステムでのみ利用可能です。
# ```
#
# ```python
# from qamomile.cudaq import CudaqTranspiler
#
# transpiler = CudaqTranspiler()
# executable = transpiler.transpile(
#     qaoa_circuit,
#     bindings={"edges": edges, "weights": weights, "bias": bias, "p": 2},
#     parameters=["gammas", "betas"],
# )
#
# job = executable.sample(
#     transpiler.executor(),
#     bindings={"gammas": init_gammas, "betas": init_betas},
#     shots=1024,
# )
# result = job.result()
# ```
#
# :::
# ::::
#
# 以下のコードはQiskitを使用してQAOA回路を実行します（メインの例）:

# %% [markdown]
# ## Qiskitを用いたQAOAの最適化
# QAOA回路が定義できたので、Qiskitを用いてパラメータの最適化を行います。

# %%
import random

from qamomile.qiskit import QiskitTranspiler


def random_ising(n: int, sparsity: float = 0.5):
    edges = []
    weights = []
    bias = []
    for i in range(n):
        bi = round(random.uniform(-1.0, 1.0), 2)
        bias.append(bi)
        for j in range(i + 1, n):
            if random.random() < sparsity:
                wij = round(random.uniform(-1.0, 1.0), 2)
                edges.append([i, j])
                weights.append(wij)

    return (
        edges,
        weights,
        bias,
    )


n = 5
edges, weights, bias = random_ising(n=n, sparsity=0.7)


# %%
transpiler = QiskitTranspiler()
executable = transpiler.transpile(
    qaoa_circuit,
    bindings={
        "edges": edges,
        "weights": weights,
        "bias": bias,
        "p": 2,
    },
    parameters=["gammas", "betas"],
)


init_gammas = np.random.uniform(0, np.pi, size=2)
init_betas = np.random.uniform(0, np.pi / 2, size=2)

job = executable.sample(
    transpiler.executor(),
    bindings={
        "gammas": init_gammas,
        "betas": init_betas,
    },
    shots=1024,
)


# %%
result = job.result()
print(result)

# %% [markdown]
# どういう量子回路が生成されたか確認してみましょう。


# %%
qiskit_circuit = executable.get_first_circuit()
print(qiskit_circuit.draw(output="text"))

# %% [markdown]
# ## エネルギー計算と古典的最適化
# QAOAでは測定結果からエネルギー（コスト関数の期待値）を計算し、
# それを最小化するようにパラメータを最適化します。
# まず、イジングモデルのエネルギーを計算する関数を定義します。


# %%
def calculate_ising_energy(
    bitstring: list[int],
    edges: list[list[int]],
    weights: list[float],
    bias: list[float],
) -> float:
    """
    イジングモデルのエネルギーを計算します。

    ビット列 z_i ∈ {0, 1} を スピン s_i ∈ {-1, +1} に変換して計算します。
    s_i = 1 - 2*z_i (z_i=0 → s_i=1, z_i=1 → s_i=-1)

    E = Σ_{(i,j)} w_ij * s_i * s_j + Σ_i b_i * s_i
    """
    spins = [1 - 2 * b for b in bitstring]

    energy = 0.0
    # 相互作用項
    for (i, j), wij in zip(edges, weights):
        energy += wij * spins[i] * spins[j]
    # バイアス項
    for i, bi in enumerate(bias):
        energy += bi * spins[i]

    return energy


def calculate_expectation_value(
    sample_result,
    edges: list[list[int]],
    weights: list[float],
    bias: list[float],
) -> float:
    """
    測定結果からエネルギーの期待値を計算します。
    """
    total_energy = 0.0
    total_counts = 0

    for bitstring, count in sample_result.results:
        energy = calculate_ising_energy(bitstring, edges, weights, bias)
        total_energy += energy * count
        total_counts += count

    return total_energy / total_counts


# %% [markdown]
# 次に、scipy.optimizeを使用してパラメータを最適化します。

# %%
from scipy.optimize import minimize

# 最適化の履歴を保存するリスト
energy_history = []


def objective_function(
    params, transpiler, executable, edges, weights, bias, shots=1024
):
    """
    最適化する目的関数。
    パラメータを受け取り、QAOA回路を実行してエネルギー期待値を返します。
    """
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    job = executable.sample(
        transpiler.executor(),
        bindings={
            "gammas": gammas,
            "betas": betas,
        },
        shots=shots,
    )
    result = job.result()

    energy = calculate_expectation_value(result, edges, weights, bias)
    energy_history.append(energy)

    return energy


# %%
# 最適化の実行
p = 2  # QAOAのレイヤー数

# 初期パラメータ
np.random.seed(42)
init_params = np.concatenate(
    [
        np.random.uniform(0, np.pi, size=p),  # gammas
        np.random.uniform(0, np.pi / 2, size=p),  # betas
    ]
)

# 履歴をクリア
energy_history = []

# NELDERーMEAD法で最適化
result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, edges, weights, bias),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print(f"\n最適化されたパラメータ:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas: {result_opt.x[p:]}")
print(f"最終エネルギー: {result_opt.fun:.4f}")

# %% [markdown]
# ## 最適化結果の可視化
# 最適化の収束の様子を可視化してみましょう。

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("QAOA Optimization Convergence")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 最適化されたパラメータでの解の確認
# 最適化されたパラメータを使用して、最終的な解の分布を確認します。

# %%
# 最適化されたパラメータでサンプリング
optimal_gammas = result_opt.x[:p]
optimal_betas = result_opt.x[p:]

job_final = executable.sample(
    transpiler.executor(),
    bindings={
        "gammas": optimal_gammas,
        "betas": optimal_betas,
    },
    shots=4096,
)
result_final = job_final.result()

# 結果をエネルギーでソート
results_with_energy = []
for bitstring, count in result_final.results:
    energy = calculate_ising_energy(bitstring, edges, weights, bias)
    results_with_energy.append((bitstring, count, energy))

results_with_energy.sort(key=lambda x: x[2])

print("測定結果（エネルギー順）:")
print("-" * 50)
for bitstring, count, energy in results_with_energy[:10]:
    bitstring_str = "".join(map(str, bitstring))
    print(f"  {bitstring_str}: count={count:4d}, energy={energy:.4f}")

# %% [markdown]
# ## 結果の解釈
# QAOAの最終結果から、最も低いエネルギーを持つビット列が最適解の候補となります。
# 厳密解との比較を行って、QAOAがどの程度良い解を見つけたか確認してみましょう。

# %%
from itertools import product


def find_exact_ground_state(
    n: int, edges: list[list[int]], weights: list[float], bias: list[float]
) -> tuple[tuple[int, ...], float]:
    """
    全探索で厳密な基底状態を見つけます（小規模問題のみ）。
    """
    min_energy = float("inf")
    best_bitstring: tuple[int, ...] = tuple([0] * n)

    for bitstring in product([0, 1], repeat=n):
        energy = calculate_ising_energy(list(bitstring), edges, weights, bias)
        if energy < min_energy:
            min_energy = energy
            best_bitstring = bitstring

    return best_bitstring, min_energy


exact_solution, exact_energy = find_exact_ground_state(n, edges, weights, bias)
qaoa_best = results_with_energy[0]

print("厳密解との比較:")
print("-" * 50)
print(f"厳密解:      {''.join(map(str, exact_solution))}, energy={exact_energy:.4f}")
print(f"QAOA最良解:  {''.join(map(str, qaoa_best[0]))}, energy={qaoa_best[2]:.4f}")
print(f"エネルギー差: {qaoa_best[2] - exact_energy:.4f}")

# 近似率の計算（エネルギーが負の場合を考慮）
if exact_energy != 0:
    approx_ratio = qaoa_best[2] / exact_energy
    print(f"近似率:      {approx_ratio:.4f}")

# %% [markdown]
# ## まとめ
# このチュートリアルでは、Qamomileを使用してQAOAを実装する方法を学びました。
#
# 主なポイント:
# 1. **qkernelデコレータ**: 量子回路を関数として定義でき、Pythonライクな記法で量子ゲートを適用できます
# 2. **パラメータ化された回路**: `qmc.Float`型を使用することで、最適化可能なパラメータを持つ量子回路を作成できます
# 3. **Transpiler**: `QiskitTranspiler`を使用して、定義した回路をQiskitの量子回路に変換し、実行できます
# 4. **古典的最適化**: `scipy.optimize`などの古典的最適化ライブラリと組み合わせて、変分量子アルゴリズムを実装できます
#
# QAOAはNISQ（ノイズあり中規模量子）デバイスで実行可能な有望なアルゴリズムであり、
# Qamomileを使用することで、簡潔かつ直感的にQAOAを実装できます。
# %%
