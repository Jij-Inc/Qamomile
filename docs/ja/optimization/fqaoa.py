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
# このチュートリアルでは、JijModelingとQamomileを使って
# **Fermionic QAOA (FQAOA)** で制約付きバイナリ最適化問題を解きます。
#
# FQAOAはバイナリ変数をフェルミオン占有数にエンコードします。
# $\sum_i x_i = M$ の形式の等式制約は、フェルミオン数 $M$ を
# 保存することで*厳密に*強制されるため、標準QAOAで必要な
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
# ここで $x_{i,d} \in \{0, 1\}$ であり、$M$ はフェルミオン数です。
#
# 標準QAOAでは制約をペナルティ項
# $\lambda \bigl(\sum_{i,d} x_{i,d} - M\bigr)^2$ として追加する必要があり、
# $\lambda$ の調整は簡単ではありません。FQAOAはこれを完全に回避します。

# %%
def constrained_qubo_problem() -> jm.Problem:
    J = jm.Placeholder("J", ndim=2)
    n = J.len_at(0, latex="n")
    D = jm.Placeholder("D")
    x = jm.BinaryVar("x", shape=(n, D))

    problem = jm.Problem("qubo")
    i, j = jm.Element("i", n), jm.Element("j", n)
    d, d_dash = jm.Element("d", D), jm.Element("d'", D)

    # 二次目的関数
    problem += jm.sum([i, j], J[i, j] * jm.sum([d, d_dash], x[i, d] * x[j, d_dash]))

    # 等式制約: 選択されたビットの合計がMに等しい
    problem += jm.Constraint("constraint", jm.sum([i, d], x[i, d]) == 4)

    return problem


problem = constrained_qubo_problem()
problem

# %% [markdown]
# ## インスタンスデータの準備
#
# $4 \times 4$ の係数行列 $J$ と $D = 2$ ビット/整数の
# 小規模なインスタンスを準備します。等式制約は
# ちょうど $M = 4$ ビットが1になることを要求しています。

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

num_fermions = 4  # 制約の合計値と一致する必要がある

# %% [markdown]
# ## コンパイル済みインスタンスの作成
#
# `jm.Interpreter` を使用して数学モデルとインスタンスデータを
# コンパイルします。

# %%
interpreter = jm.Interpreter(instance_data)
instance = interpreter.eval_problem(problem)

# %% [markdown]
# ## FQAOA回路への変換
#
# `FQAOAConverter` はコンパイル済みインスタンス**と**フェルミオン数
# $M$ を受け取ります。内部的には、制約がフェルミオンエンコーディング
# 自体で強制されるため、`uniform_penalty_weight=0.0` でQUBOが
# 生成されます。
#
# `transpile` メソッドは、変分角度 `gammas` と `betas` のみが
# 自由パラメータの実行可能プログラムを生成します。

# %%
from qamomile.optimization.fqaoa import FQAOAConverter
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

p = 2  # FQAOAの層数
converter = FQAOAConverter(instance, num_fermions=num_fermions)
executable = converter.transpile(transpiler, p=p)

# %%
qiskit_circuit = executable.get_first_circuit()
if qiskit_circuit is not None:
    print(f"量子ビット数: {qiskit_circuit.num_qubits}")
    print(f"変分パラメータ数: {len(qiskit_circuit.parameters)}")
    print(f"回路の深さ: {qiskit_circuit.depth()}")

# %% [markdown]
# ## エネルギー計算
#
# VQEループを実行するには、測定結果をイジングエネルギーに
# 変換する関数が必要です。コンバータは `converter.spin_model` に
# スピンモデルを格納しています。

# %%
def calculate_ising_energy(bitstring: list[int], spin_model) -> float:
    """測定ビット列をイジングエネルギーに変換する。

    変換規則: z_i ∈ {0, 1} -> s_i = 1 - 2*z_i ∈ {+1, -1}。
    """
    spins = [1 - 2 * b for b in bitstring]
    return spin_model.calc_energy(spins)


def calculate_expectation_value(sample_result, spin_model) -> float:
    """全測定結果の加重平均エネルギー。"""
    total_energy = 0.0
    total_counts = 0
    for bitstring, count in sample_result.results:
        total_energy += calculate_ising_energy(bitstring, spin_model) * count
        total_counts += count
    return total_energy / total_counts


# %% [markdown]
# ## VQE最適化
#
# scipyのCOBYLA最適化器を使用して、変分パラメータ `gammas` と `betas` に
# 対するエネルギー期待値を最小化します。

# %%
from scipy.optimize import minimize

energy_history: list[float] = []


def objective_function(params, spin_model, shots=1024):
    """VQEループの目的関数。"""
    p = len(params) // 2
    gammas = params[:p]
    betas = params[p:]

    job = executable.sample(
        transpiler.executor(),
        bindings={"gammas": gammas, "betas": betas},
        shots=shots,
    )
    result = job.result()

    energy = calculate_expectation_value(result, spin_model)
    energy_history.append(energy)
    return energy


# %%
np.random.seed(42)

init_params = np.concatenate([
    np.random.uniform(0, 2 * np.pi, size=p),  # gammas
    np.random.uniform(0, np.pi, size=p),       # betas
])

energy_history = []

print(f"FQAOA最適化を開始します (p={p}層)...")
print(f"量子ビット数: {converter.num_qubits}")
print(f"フェルミオン数: {converter.num_fermions}")

result_opt = minimize(
    objective_function,
    init_params,
    args=(converter.spin_model,),
    method="COBYLA",
    options={"maxiter": 100, "disp": True},
)

print("\n最適化されたパラメータ:")
print(f"  gammas: {result_opt.x[:p]}")
print(f"  betas:  {result_opt.x[p:]}")
print(f"最終エネルギー: {result_opt.fun:.4f}")

# %% [markdown]
# ## 最適化の収束可視化

# %%
plt.figure(figsize=(10, 5))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("FQAOA Optimization Convergence")
plt.grid(True)
plt.tight_layout()
# plt.show()

# %% [markdown]
# ## 最終解の分析
#
# 最適化された回路からより多くのショットでサンプリングし、
# 測定結果を元のバイナリ変数の領域にデコードします。

# %%
optimal_gammas = result_opt.x[:p]
optimal_betas = result_opt.x[p:]

job_final = executable.sample(
    transpiler.executor(),
    bindings={"gammas": optimal_gammas, "betas": optimal_betas},
    shots=4096,
)
result_final = job_final.result()

# 測定結果をバイナリ変数の割り当てにデコード
sampleset = converter.decode(result_final)

# 最良解を表示
best_sample, best_energy, best_count = sampleset.lowest()
print("見つかった最良解:")
print(f"  変数の割り当て: {best_sample}")
print(f"  エネルギー: {best_energy:.4f}")
print(f"  出現回数: {best_count}")

# %% [markdown]
# エネルギー順にソートした上位の測定結果も確認できます。

# %%
results_with_energy = []
for bitstring, count in result_final.results:
    energy = calculate_ising_energy(bitstring, converter.spin_model)
    results_with_energy.append((bitstring, count, energy))

results_with_energy.sort(key=lambda x: x[2])

print("上位の測定結果（エネルギー順）:")
print("-" * 60)
for bitstring, count, energy in results_with_energy[:10]:
    bitstring_str = "".join(map(str, bitstring))
    probability = count / 4096
    # フェルミオン数保存の検証: すべてのビット列は
    # ちょうど num_fermions 個のビットが1になるはず
    num_ones = sum(bitstring)
    print(
        f"  {bitstring_str}: count={count:4d}, "
        f"prob={probability:.3f}, energy={energy:.4f}, "
        f"ones={num_ones}"
    )

# %% [markdown]
# **すべてのビット列がちょうど $M = 4$ 個のビットが1** であることに
# 注目してください。これは、フェルミオン数保存が等式制約を
# ペナルティ項なしで強制していることを確認するものです。

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは、QamomileでFQAOAを使って制約付き
# 最適化問題を解く方法を実演しました:
#
# 1. **問題の定式化**: JijModelingを使って等式制約付きQUBOを定義
# 2. **制約の処理**: `FQAOAConverter`がフェルミオン数保存で制約をエンコード ── ペナルティ重みの調整が不要
# 3. **回路の生成**: `converter.transpile()`がGivens回転初期状態とフェルミオンミキサーを含む完全なFQAOAアンサッツを生成
# 4. **VQE最適化**: scipyのCOBYLA最適化器で最適な変分パラメータを探索
# 5. **解のデコード**: `converter.decode()`が測定結果を変数の割り当てにマッピング
#
# FQAOAの標準QAOAに対する主な利点:
# - 等式制約が構造的に**厳密に**充足される
# - ペナルティ重み $\lambda$ の調整が不要
# - 探索空間が実行可能解に制限されるため、収束が改善される
