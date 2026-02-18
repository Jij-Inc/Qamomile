# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: qamomile
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 制約付き最適化のためのFQAOA
#
# このチュートリアルでは、JijModelingとQamomileを使い、
# **Fermionic QAOA (FQAOA)** で制約付きバイナリ最適化問題を解きます。
#
# FQAOAはバイナリ変数をフェルミオンの占有数にエンコードします。
# $\sum_i x_i = M$ の形式の等式制約は、フェルミオン数 $M$ の
# 保存によって*厳密に*満たされるため、標準QAOAで必要となる
# ペナルティ項が不要になります。
#
# **参考文献**: Yoshioka et al., *Fermionic Quantum Approximate Optimization Algorithm* (2023).

# %%
import jijmodeling as jm
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 最適化問題の定義
#
# 等式制約を持つ二次バイナリ最適化問題を考えます:
#
# $$
#   \min \quad \sum_{i,j} J_{i,j}
#       \sum_{d, d'} x_{i,d}\, x_{j,d'}
#   \qquad \text{s.t.} \quad \sum_{i,d} x_{i,d} = M
# $$
#
# ここで $x_{i,d} \in \{0, 1\}$、$M$ はフェルミオン数です。
#
# 標準QAOAでは制約をペナルティ項
# $\lambda \bigl(\sum_{i,d} x_{i,d} - M\bigr)^2$ として追加する必要があり、
# $\lambda$ の調整は容易ではありません。FQAOAはこれを完全に回避します。


# %%
problem = jm.Problem("qubo")


@problem.update
def _(problem: jm.DecoratedProblem):
    J = problem.Float(ndim=2)
    n = J.len_at(0, latex="n")
    D = problem.Dim()
    x = problem.BinaryVar(shape=(n, D))

    # Quadratic objective
    problem += J.ndenumerate().map(
        lambda ij_v: ij_v[1] * x[ij_v[0][0]].sum() * x[ij_v[0][1]].sum()
    ).sum()

    # Equality constraint: total number of selected bits equals M
    problem += problem.Constraint("constraint", x.sum() == 4)


problem

# %% [markdown]
# ## インスタンスデータの準備
#
# $4 \times 4$ の係数行列 $J$ と $D = 2$ ビット/整数の小規模な
# インスタンスを用意します。等式制約により、ちょうど $M = 4$ 個の
# ビットが1になることが要求されます。

# %%
instance_data = {
    "J": [
        [0.0, 0.4, 0.0, 0.0],
        [0.0, 0.0, 0.8, 0.0],
        [0.0, 0.0, 0.0, 0.3],
        [0.0, 0.0, 0.0, 0.0],
    ],
    "D": 2,
}

num_fermions = 4  # must match the constraint sum

# %% [markdown]
# ## コンパイル済みインスタンスの作成
#
# `problem.eval()` を使って、数学モデルとインスタンスデータを
# まとめてコンパイルします。

# %%
instance = problem.eval(instance_data)

# %% [markdown]
# ## FQAOA回路とハミルトニアンへの変換
#
# `FQAOAConverter` はコンパイル済みインスタンス**と**フェルミオン数
# $M$ を受け取ります。等式制約はフェルミオンエンコーディング自体に
# よって強制されるため、ペナルティ項は不要です。
#
# 以下の操作が可能です:
# - `transpile()` でFQAOA量子回路を生成
# - `get_cost_hamiltonian()` でコストハミルトニアンを確認
#
# FQAOAの層数 $p$ はここでは $2$ に設定しています。

# %%
from qamomile.optimization.fqaoa import FQAOAConverter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

p = 2  # Number of FQAOA layers
converter = FQAOAConverter(instance, num_fermions=num_fermions)
executable = converter.transpile(transpiler, p=p)

# %% [markdown]
# コストハミルトニアンを確認しましょう。ハミルトニアンはQUBO目的関数のイジング表現から構築されます。

# %%
cost_hamiltonian = converter.get_cost_hamiltonian()
cost_hamiltonian

# %% [markdown]
# 生成された量子回路を見てみましょう。標準QAOAとは異なり、FQAOA回路にはGivens回転による初期状態準備とフェルミオンホッピングミキサーが含まれています。

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw()

# %% [markdown]
# ## VQE最適化
#
# scipyのCOBYLA最適化器を使って、変分パラメータ `gammas` と `betas` に
# 対するエネルギー期待値を最小化します。

# %%
from scipy.optimize import minimize

energy_history = []


def objective_function(params, transpiler, executable, converter, shots=1024):
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    job = executable.sample(
        transpiler.executor(),
        bindings={"gammas": gammas, "betas": betas},
        shots=shots,
    )
    result = job.result()

    sampleset = converter.decode(result)
    energy = sampleset.energy_mean()
    energy_history.append(energy)
    return energy


# %%
np.random.seed(901)

init_params = np.concatenate(
    [
        np.random.uniform(0, 2 * np.pi, size=p),  # gammas
        np.random.uniform(0, np.pi, size=p),  # betas
    ]
)

energy_history = []

print(f"Starting FQAOA optimization with p={p} layers...")

result_opt = minimize(
    objective_function,
    init_params,
    args=(transpiler, executable, converter),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\nOptimized parameters:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas:  {result_opt.x[p:]}")
print(f"Final energy: {result_opt.fun:.4f}")

# %% [markdown]
# ## 最適化結果の可視化
#
# 最適化プロセスの収束の様子を可視化しましょう。

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("FQAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 最終解の分析
#
# 最適化された回路からサンプリングし、結果を分析しましょう。

# %%
optimal_gammas = result_opt.x[:p]
optimal_betas = result_opt.x[p:]

job_final = executable.sample(
    transpiler.executor(),
    bindings={"gammas": optimal_gammas, "betas": optimal_betas},
    shots=4096,
)
result_final = job_final.result()

# Decode results using the converter
sampleset = converter.decode(result_final)

num_vars = converter.num_qubits

# Build frequency distribution over all sampled bitstrings
bitstrings = []
counts = []
energies = []
for i in range(len(sampleset.samples)):
    sample = sampleset.samples[i]
    bitstring_str = "".join(str(sample[j]) for j in range(num_vars))
    bitstrings.append(bitstring_str)
    counts.append(sampleset.num_occurrences[i])
    energies.append(sampleset.energy[i])

# Sort by bitstring for consistent display
sorted_order = np.argsort(bitstrings)
bitstrings = [bitstrings[i] for i in sorted_order]
counts = [counts[i] for i in sorted_order]
energies = [energies[i] for i in sorted_order]

# Determine optimal energy
best_sample, best_energy, best_count = sampleset.lowest()

# Plot frequency distribution
fig, ax = plt.subplots(figsize=(12, 5))
x_pos = np.arange(len(bitstrings))
bars = ax.bar(x_pos, counts)

# Highlight optimal solutions with red bars
for i, e in enumerate(energies):
    if np.isclose(e, best_energy):
        bars[i].set_color("red")

ax.set_xticks(x_pos)
ax.set_xticklabels(bitstrings, rotation=90)
ax.set_xlabel("Bitstring")
ax.set_ylabel("Counts")
ax.set_title(f"FQAOA Measurement Frequency Distribution (red = optimal, energy = {best_energy:.2f})")
plt.tight_layout()
plt.show()

# %% [markdown]
# **すべてのビット列がちょうど $M = 4$ 個のビットが1になっている**ことに注目してください。これは、フェルミオン数保存がペナルティ項なしで等式制約を強制していることを示しています。赤いバーは最適解を示しており、FQAOAが測定確率を最適解に集中させることに成功していることがわかります。

# %%
print("Best solution found:")
print(f"  Variable assignment: {best_sample}")
print(f"  Energy: {best_energy:.4f}")
print(f"  Occurrences: {best_count}")

# %% [markdown]
# ## 厳密解との比較
#
# Qamomileのコンバータは `ommx.v1.Instance` を受け付けるため、
# 古典ソルバーとの比較が容易です。同じインスタンスをSCIPで厳密に解いて、
# FQAOAの解と比較してみましょう。

# %%
from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

solution = OMMXPySCIPOptAdapter.solve(instance)

print(f"Exact optimal value: {solution.objective:.4f}")
print(f"FQAOA best energy:   {best_energy:.4f}")

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、QamomileのFQAOAを使って制約付き最適化問題を
# 解く方法を実演しました:
#
# 1. **問題の定式化**: JijModelingを使って等式制約付きQUBOを定義
# 2. **ハミルトニアンと回路の生成**: `FQAOAConverter` がコストハミルトニアンと、Givens回転初期状態・フェルミオンミキサーを含むFQAOA回路を自動生成
# 3. **VQE最適化**: scipyのCOBYLA最適化器で最適なFQAOAパラメータを探索
# 4. **解の分析**: 測定頻度分布により、FQAOAが測定確率を最適解に集中させ、すべてのビット列が制約を厳密に満たしていることを確認
#
# FQAOAの標準QAOAに対する主な利点:
# - 等式制約が構造的に**厳密に**満たされる
# - ペナルティ重み $\lambda$ の調整が不要
# - 探索空間が実行可能解に制限されるため、収束が改善される
