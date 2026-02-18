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
#     name: qamomile
# ---

# %% [markdown]
# # カスタム最適化コンバーターの作成
#
# このチュートリアルでは、`MathematicalProblemConverter` を継承して
# 独自の最適化コンバーターを作成する方法を解説します。コンバーターは
# 数理最適化問題と、それを解く量子回路との橋渡しをする役割を持ちます。
# 具体的には、問題をコストハミルトニアンに変換し、測定結果を古典的な解に
# デコードします。
#
# このチュートリアルでは `MathematicalProblemConverter`
# (`qamomile.optimization.converter`) に焦点を当てます。このクラスは
# `BinaryModel`（または `ommx.v1.Instance`）を内部的にスピンモデルに
# 変換する定型処理を担い、組み込みの `decode()` メソッドを提供します。
# 実装が必要な抽象メソッドは `get_cost_hamiltonian()` の1つだけです。

# %% [markdown]
# ## 基底クラスの理解
#
# `MathematicalProblemConverter` の主要な構造は以下の通りです:
#
# ```
# MathematicalProblemConverter
#     __init__(instance: ommx.v1.Instance | BinaryModel)
#         -> 内部的にスピンモデルに変換 (self.spin_model)
#         -> __post_init__() を呼び出し
#
#     get_cost_hamiltonian() -> Hamiltonian    [抽象メソッド、実装が必要]
#     decode(samples: SampleResult) -> BinarySampleSet   [組み込み]
#     __post_init__()                          [オーバーライド可能]
# ```
#
# コンバーターのインスタンスを作成すると、以下の処理が行われます:
#
# 1. コンストラクタが `ommx.v1.Instance` または `BinaryModel` を受け取ります。
# 2. 内部的に問題をスピン（イジング）モデルに変換し、`self.spin_model` に格納します。
# 3. `__post_init__()` が呼び出されます。カスタム初期化処理のためにオーバーライドできます。
# 4. `get_cost_hamiltonian()` が `self.spin_model` を読み取り、`Hamiltonian` を返します。
# 5. 量子計算の実行後、`decode()` が測定ビット列を元の変数型の `BinarySampleSet` に変換します。
#
# ステップ2の挙動は入力型によって異なります:
#
# | 入力型 | スピンモデルへの変換 | 目的関数の方向の処理 |
# |---|---|---|
# | `ommx.v1.Instance` | 内部で `instance.to_qubo()` を呼び出し、QUBO → SPIN に変換 | 最大化問題は自動的に符号反転され最小化形式になります。最大化問題のデコード後のエネルギーは負の値になります。 |
# | `BinaryModel` | 直接 `change_vartype(SPIN)` を呼び出し | 目的関数の方向の概念はありません。係数はそのまま使用されます。符号の扱いはユーザーの責任です。 |
#
# 主要なクラスをインポートしましょう。

# %%
import qamomile.observable as qm_o
from qamomile.circuit.transpiler.job import SampleResult
from qamomile.optimization.binary_model import (
    BinaryModel,
)
from qamomile.optimization.converter import MathematicalProblemConverter

# %% [markdown]
# ## Observable モジュール
#
# コンバーターを構築する前に、ハミルトニアンの構成に使う
# `qamomile.observable` モジュールを確認しましょう。
#
# このモジュールは以下を提供します:
#
# - `Hamiltonian` -- 重み付きパウリ演算子の積の和に定数項を加えたもの
# - `PauliOperator(pauli, index)` -- 特定の量子ビット上の単一パウリゲート
# - `Pauli` -- `X`, `Y`, `Z`, `I` の値を持つ列挙型
# - 簡易ファクトリ関数: `X(i)`, `Y(i)`, `Z(i)` -- それぞれ単一項の `Hamiltonian` を返す
#
# ハミルトニアンは算術演算（`+`, `-`, `*`, スカラー乗算）をサポートしているため、
# 自然な形で組み立てることができます。

# %%
# Create single Pauli-Z operators on qubits 0 and 1
Z0 = qm_o.Z(0)
Z1 = qm_o.Z(1)

# Build a Hamiltonian: -1.0 * Z0*Z1 + 0.5 * Z0
H_example = -1.0 * Z0 * Z1 + 0.5 * Z0
print("Example Hamiltonian:", H_example)
print("Number of qubits:", H_example.num_qubits)
print("Constant term:", H_example.constant)

# %% [markdown]
# `add_term()` を使って項ごとにハミルトニアンを構築することもできます:

# %%
H_manual = qm_o.Hamiltonian()
H_manual.add_term(
    (qm_o.PauliOperator(qm_o.Pauli.Z, 0), qm_o.PauliOperator(qm_o.Pauli.Z, 1)),
    -1.0,
)
H_manual.add_term(
    (qm_o.PauliOperator(qm_o.Pauli.Z, 0),),
    0.5,
)
print("Manual Hamiltonian:", H_manual)
print("Are they equal?", H_example == H_manual)

# %% [markdown]
# ## 例: シンプルなイジングコンバーター
#
# それでは、カスタムコンバーターを作成しましょう。最も単純なケースは
# 標準的なイジングエンコーディングです。各スピン変数をパウリ Z 演算子に
# マッピングします。これは本質的に `QAOAConverter` が内部的に行っていること
# と同じですが、ゼロから書くことでパターンを明確に理解できます。
#
# スピンモデル (`self.spin_model`) は SPIN 変数型の `BinaryModel` です。
# 以下のプロパティを持ちます:
#
# - `linear` -- `dict[int, float]`: 量子ビットインデックスから線形係数へのマッピング
# - `quad` -- `dict[tuple[int, int], float]`: 量子ビットペアから二次係数へのマッピング
# - `higher` -- `dict[tuple[int, ...], float]`: 高次項
# - `constant` -- `float`: 定数オフセット
# - `num_bits` -- `int`: スピン変数の数


# %%
class SimpleIsingConverter(MathematicalProblemConverter):
    """A simple converter that creates a Z-only Hamiltonian from the spin model.

    This converter maps each spin variable s_i to a Pauli Z_i operator.
    The resulting Hamiltonian is:

        H = sum_{(i,j)} J_ij Z_i Z_j  +  sum_i h_i Z_i  +  constant
    """

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        hamiltonian = qm_o.Hamiltonian()

        # Add Z-Z interaction terms (quadratic)
        for (i, j), Jij in self.spin_model.quad.items():
            hamiltonian += Jij * qm_o.Z(i) * qm_o.Z(j)

        # Add single-Z field terms (linear)
        for i, hi in self.spin_model.linear.items():
            hamiltonian += hi * qm_o.Z(i)

        # Add constant energy offset
        hamiltonian += self.spin_model.constant

        return hamiltonian


# %% [markdown]
# これがコンバーターの全体です。`decode()` メソッドは
# `MathematicalProblemConverter` から継承されており、測定ビット列から
# 元の変数型への変換を自動的に処理します。

# %% [markdown]
# ## BinaryModel による問題の定義
#
# コンバーターをテストするには問題インスタンスが必要です。最も簡単な方法は
# `BinaryModel` を直接使うことで、JijModeling を使う必要がありません。
#
# `BinaryModel` は、変数型（BINARY または SPIN）、定数オフセット、
# インデックスのタプルをキーとする係数辞書を保持する `BinaryExpr` から
# 構築されます。`from_ising()` や `from_qubo()` などの便利なコンストラクタも
# 用意されています。
#
# 3スピンのイジング問題を作成しましょう:
#
# $$
# E(s) = -1.0 \, s_0 s_1  -0.5 \, s_1 s_2  + 0.3 \, s_0
# $$
#
# ここで $s_i \in \{+1, -1\}$ です。

# %%
model = BinaryModel.from_ising(
    linear={0: 0.3},
    quad={(0, 1): -1.0, (1, 2): -0.5},
    constant=0.0,
)

print("Variable type:", model.vartype)
print("Number of spins:", model.num_bits)
print("Linear terms:", model.linear)
print("Quadratic terms:", model.quad)
print("Constant:", model.constant)

# %% [markdown]
# ## コンバーターの作成とハミルトニアンの確認
#
# `BinaryModel` を `SimpleIsingConverter` に渡します。基底クラスは
# 内部的にスピンモデルに変換します（ここでは既に SPIN 形式です）。

# %%
converter = SimpleIsingConverter(model)
cost_hamiltonian = converter.get_cost_hamiltonian()

print("Cost Hamiltonian:", cost_hamiltonian)
print("Number of qubits:", cost_hamiltonian.num_qubits)
print("Constant:", cost_hamiltonian.constant)
print()
print("Terms:")
for ops, coeff in cost_hamiltonian:
    print(f"  {ops} -> {coeff}")

# %% [markdown]
# ## 変分回路の構築
#
# 次に、作成したハミルトニアンを使う簡単な変分量子固有値ソルバー（VQE）回路を
# 構築します。この回路はハミルトニアンを `Observable` パラメータとして受け取り、
# `qmc.expval()` を使って期待値を計算します。

# %%
import qamomile.circuit as qmc


@qmc.qkernel
def variational_circuit(
    n_qubits: qmc.UInt,
    theta: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    """A simple variational ansatz: Ry rotations + CNOT entangling layer."""
    q = qmc.qubit_array(n_qubits, name="q")

    # Apply parameterized Ry rotations
    for i in qmc.range(n_qubits):
        q[i] = qmc.ry(q[i], theta[i])

    # Apply CNOT entangling layer
    for i in qmc.range(n_qubits - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])

    # Compute and return the expectation value
    return qmc.expval(q, H)


# %% [markdown]
# ## エンドツーエンドのワークフロー
#
# すべてをまとめましょう: 問題の定義、コンバーターの作成、ハミルトニアンの構築、
# 回路のトランスパイル、実行、そして結果のデコードまでを行います。

# %%
import numpy as np

from qamomile.qiskit import QiskitTranspiler

# Step 1: Define the problem (reuse the model from above)
print("Step 1: Problem defined")
print(f"  {model.num_bits} spins, linear={model.linear}, quad={model.quad}")

# Step 2: Create the converter
converter = SimpleIsingConverter(model)
print("\nStep 2: Converter created")

# Step 3: Get the cost Hamiltonian
cost_hamiltonian = converter.get_cost_hamiltonian()
print(f"\nStep 3: Cost Hamiltonian ({cost_hamiltonian.num_qubits} qubits)")
for ops, coeff in cost_hamiltonian:
    print(f"  {ops} : {coeff}")

# Step 4: Transpile the variational circuit with the Hamiltonian
transpiler = QiskitTranspiler()
n_qubits = cost_hamiltonian.num_qubits

executable = transpiler.transpile(
    variational_circuit,
    bindings={
        "n_qubits": n_qubits,
        "H": cost_hamiltonian,
    },
    parameters=["theta"],
)
print("\nStep 4: Circuit transpiled")

# Step 5: Execute with some initial parameters
np.random.seed(901)
theta_init = np.random.uniform(0, np.pi, size=n_qubits)
job = executable.run(
    transpiler.executor(),
    bindings={"theta": theta_init},
)
expval_result = job.result()
print(f"\nStep 5: Expectation value = {expval_result:.4f}")

# %% [markdown]
# ### パラメータの最適化
#
# 古典オプティマイザーを使って、コストハミルトニアンの期待値を
# 最小化するパラメータを探索できます。

# %%
from scipy.optimize import minimize

energy_history = []


def objective(params, transpiler, executable):
    job = executable.run(
        transpiler.executor(),
        bindings={"theta": params},
    )
    energy = job.result()
    energy_history.append(energy)
    return energy


init_params = np.random.uniform(0, np.pi, size=n_qubits)

energy_history = []
result_opt = minimize(
    objective,
    init_params,
    args=(transpiler, executable),
    method="COBYLA",
    options={"maxiter": 100},
)

print(f"Optimized energy: {result_opt.fun:.4f}")
print(f"Optimal theta: {result_opt.x}")

# %% [markdown]
# ### 収束の可視化

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(energy_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Energy")
plt.title("VQE Optimization Convergence")
plt.grid(True)
plt.tight_layout()

# %% [markdown]
# ### サンプリングによる結果のデコード
#
# 古典的な解を抽出するにはサンプリングモードに切り替えます。測定回路を構築し、
# ビット列をサンプリングして、コンバーターの `decode()` メソッドで
# デコードします。


# %%
@qmc.qkernel
def sampling_circuit(
    n_qubits: qmc.UInt,
    theta: qmc.Vector[qmc.Float],
) -> qmc.Vector[qmc.Bit]:
    """Same ansatz as above, but ending with measurement."""
    q = qmc.qubit_array(n_qubits, name="q")
    for i in qmc.range(n_qubits):
        q[i] = qmc.ry(q[i], theta[i])
    for i in qmc.range(n_qubits - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
    return qmc.measure(q)


sample_executable = transpiler.transpile(
    sampling_circuit,
    bindings={"n_qubits": n_qubits},
    parameters=["theta"],
)

# Sample with the optimized parameters
sample_job = sample_executable.sample(
    transpiler.executor(),
    bindings={"theta": result_opt.x},
    shots=1024,
)
sample_result = sample_job.result()

print("Top measurement results:")
for bitstring, count in sample_result.most_common(5):
    print(f"  {bitstring} : {count} counts")

# %% [markdown]
# 次に、`converter.decode()` を使って測定結果を元の変数ドメインに
# デコードします。入力モデルが SPIN 変数を使用していたため、
# デコードされたサンプルも SPIN 形式（$+1$ / $-1$）になります。

# %%
decoded = converter.decode(sample_result)

print(f"Variable type: {decoded.vartype}")
print(f"Number of samples: {len(decoded.samples)}")

# Show the lowest-energy solution
best_sample, best_energy, best_count = decoded.lowest()
print("\nBest solution:")
print(f"  Sample: {best_sample}")
print(f"  Energy: {best_energy:.4f}")
print(f"  Occurrences: {best_count}")

print(f"\nMean energy: {decoded.energy_mean():.4f}")

# %% [markdown]
# ## BINARY 変数の扱い
#
# コンバーターは変数型の変換を自動的に処理します。BINARY モデルを渡すと、
# 基底クラスが内部的に SPIN に変換し、`decode()` が結果を BINARY (0/1) に
# 戻します。

# %%
binary_model = BinaryModel.from_qubo(
    qubo={
        (0, 1): -2.0,
        (1, 2): -1.0,
        (0, 0): 1.0,  # diagonal = linear term in QUBO
        (1, 1): -0.5,
    },
    constant=0.0,
)

print("Input vartype:", binary_model.vartype)

binary_converter = SimpleIsingConverter(binary_model)
H_binary = binary_converter.get_cost_hamiltonian()
print("Cost Hamiltonian:", H_binary)
print("Num qubits:", H_binary.num_qubits)

# %% [markdown]
# デコード時には、結果は BINARY 形式で返されます:

# %%
# Simulate a fake sample result for demonstration
fake_result = SampleResult(
    results=[
        ([0, 0, 0], 100),  # all |0>
        ([1, 1, 0], 200),  # |110>
        ([1, 1, 1], 150),  # |111>
    ],
    shots=450,
)

decoded_binary = binary_converter.decode(fake_result)
print(f"Decoded vartype: {decoded_binary.vartype}")
for sample, energy in zip(decoded_binary.samples, decoded_binary.energy):
    print(f"  {sample} -> energy = {energy:.4f}")

# %% [markdown]
# ## JijModeling と `ommx.v1.Instance` の利用
#
# 実際には、最適化問題は JijModeling を使って記号的に定義し、
# `ommx.v1.Instance` にコンパイルするのが一般的です。基底クラス
# `MathematicalProblemConverter` は `BinaryModel` と `ommx.v1.Instance` の
# どちらも受け付けるため、`SimpleIsingConverter` はコードの変更なしに
# 両方で動作します。
#
# 4ノードグラフの小さな Max-Cut 問題で実演しましょう。

# %%
import jijmodeling as jm
import networkx as nx

# Define the Max-Cut problem symbolically
problem = jm.Problem("Maxcut", sense=jm.ProblemSense.MAXIMIZE)


@problem.update
def _(problem: jm.DecoratedProblem):
    V = problem.Dim()
    E = problem.Graph()
    x = problem.BinaryVar(shape=(V,))
    obj = (
        E.rows()
        .map(lambda e: 1 / 2 * (1 - (2 * x[e[0]] - 1) * (2 * x[e[1]] - 1)))
        .sum()
    )
    problem += obj


# Create a small graph instance
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (0, 3)])

data = {"V": G.number_of_nodes(), "E": list(G.edges())}
instance = problem.eval(data)

# %% [markdown]
# `ommx.v1.Instance` をカスタムコンバーターに直接渡します:

# %%
ommx_converter = SimpleIsingConverter(instance)
H_ommx = ommx_converter.get_cost_hamiltonian()

print("Cost Hamiltonian from OMMX instance:")
print(H_ommx)
print(f"Number of qubits: {H_ommx.num_qubits}")

# %% [markdown]
# コンバーターは同じように動作します。基底クラスが内部的に
# `ommx.v1.Instance` を QUBO に変換し、さらにスピンモデルに変換します。
# サンプリングと結果のデコードも以前と全く同じ方法で行えます:

# %%
ommx_sample_exec = transpiler.transpile(
    sampling_circuit,
    bindings={"n_qubits": H_ommx.num_qubits},
    parameters=["theta"],
)

ommx_sample_job = ommx_sample_exec.sample(
    transpiler.executor(),
    bindings={"theta": np.random.uniform(0, np.pi, size=H_ommx.num_qubits)},
    shots=512,
)
ommx_decoded = ommx_converter.decode(ommx_sample_job.result())

print(f"Decoded vartype: {ommx_decoded.vartype}")
best_sample, best_energy, best_count = ommx_decoded.lowest()
print(f"Best solution: {best_sample}")
print(f"Best energy: {best_energy:.4f}")

# %% [markdown]
# ## 応用: `__post_init__` のオーバーライド
#
# 基底クラス `MathematicalProblemConverter` は、`__init__()` の最後に
# `self.spin_model` が利用可能になった後で `__post_init__()` を呼び出します。
# このフックをオーバーライドすることで、スピンモデルに依存するカスタム
# 初期化処理を実行できます。
#
# 例えば、組み込みの `QRAC31Converter` は `__post_init__` を使って
# 相互作用グラフのグラフ彩色を行い、スピンを量子ビットにどのように
# パッキングするかを決定します:
#
# ```python
# class QRAC31Converter(MathematicalProblemConverter):
#     def __post_init__(self) -> None:
#         _, color_group = greedy_graph_coloring(
#             graph=self.spin_model.quad.keys(),
#             max_color_group_size=3,
#         )
#         self.color_group = color_group
#         self.encoded_ope = color_group_to_qrac_encode(color_group)
#         # ...
# ```
#
# 以下は実践的な例です: スピンモデルから**相互作用グラフ**を事前計算します。
# これは、コンバーターが変数間の相互作用の構造情報を必要とする場合
# （例: 回路レイアウトの最適化や変数順序のヒューリスティクス）に有用です。


# %%
class GraphAwareIsingConverter(MathematicalProblemConverter):
    """Ising converter that precomputes the interaction graph structure."""

    def __post_init__(self) -> None:
        # Build adjacency list and degree information from the spin model
        self.adjacency: dict[int, list[int]] = {
            i: [] for i in range(self.spin_model.num_bits)
        }
        for i, j in self.spin_model.quad:
            self.adjacency[i].append(j)
            self.adjacency[j].append(i)

        self.degree = {
            node: len(neighbors) for node, neighbors in self.adjacency.items()
        }

        # Identify isolated variables (no interactions) and hub variables
        self.isolated = [i for i, d in self.degree.items() if d == 0]
        self.max_degree_node = (
            max(self.degree, key=self.degree.get) if self.degree else None
        )

    def get_cost_hamiltonian(self) -> qm_o.Hamiltonian:
        hamiltonian = qm_o.Hamiltonian()
        for (i, j), Jij in self.spin_model.quad.items():
            hamiltonian += Jij * qm_o.Z(i) * qm_o.Z(j)
        for i, hi in self.spin_model.linear.items():
            hamiltonian += hi * qm_o.Z(i)
        hamiltonian += self.spin_model.constant
        return hamiltonian


# Demonstrate the precomputed structure
graph_converter = GraphAwareIsingConverter(model)

print("Adjacency list:", graph_converter.adjacency)
print("Degree:", graph_converter.degree)
print("Isolated nodes:", graph_converter.isolated)
print("Highest-degree node:", graph_converter.max_degree_node)

# The Hamiltonian itself is the same as SimpleIsingConverter
H_graph = graph_converter.get_cost_hamiltonian()
print("\nCost Hamiltonian:", H_graph)

# %% [markdown]
# `__post_init__` フックにより、コンストラクタ実行時に問題の構造を
# 一度だけ解析できます。この情報は以下のような用途に活用できます:
#
# - **回路レイアウト**: 次数の高い変数を接続性の良い量子ビットにマッピング
# - **変数の順序付け**: カスタムアンザッツ設計でハブノードを優先的に処理
# - **問題の診断**: 非連結な部分問題の検出
#
# 組み込みコンバーターはこのパターンを広く活用しています:
#
# - `QRAC31Converter`: QRAC エンコーディングを決定するためのグラフ彩色
# - `FQAOAConverter`: フェルミオンエンコーディングのための巡回変数マッピング

# %% [markdown]
# ## まとめ
#
# このチュートリアルでは以下の内容を扱いました:
#
# 1. **`MathematicalProblemConverter` 基底クラス** -- `BinaryModel` または
#    `ommx.v1.Instance` を受け取り、内部的にスピンモデルに変換し、
#    組み込みの `decode()` メソッドを提供します。
#
# 2. **`get_cost_hamiltonian()` の実装** -- 実装が必要な唯一の抽象メソッドです。
#    `self.spin_model`（linear, quad, higher, constant）を読み取り、
#    `qm_o.Hamiltonian` を返します。
#
# 3. **エンドツーエンドのワークフロー** -- 問題の定義、コンバーターの作成、
#    `qmc.expval()` を使った変分回路の構築、パラメータの最適化、
#    サンプリング、結果のデコード。
#
# 4. **`ommx.v1.Instance` の利用** -- JijModeling で記号的に問題を定義し、
#    インスタンスにコンパイルして、コンバーターに直接渡すことができます。
#    コンバーターの変更は不要です。
#
# 5. **`__post_init__()` のオーバーライド** -- スピンモデルが利用可能に
#    なった後に実行されるカスタム初期化ロジック（例: 相互作用グラフの
#    構造の事前計算、グラフ彩色）。
#
# ### 参考: 組み込みコンバーター
#
# - `QAOAConverter` (`qamomile.optimization.qaoa`) -- Z のみのハミルトニアンを
#   用いた標準的な QAOA。QAOA アンザッツ用の `transpile()` メソッドを含みます。
#
# - `QRAC31Converter` (`qamomile.optimization.qrao`) -- 量子ランダムアクセス
#   符号エンコーディング。X/Y/Z 演算子を使って最大3スピンを1量子ビットに
#   パッキングします。
#
# - `FQAOAConverter` (`qamomile.optimization.fqaoa`) -- 粒子数保存を持つ
#   フェルミオン QAOA。制約付き最適化に使用されます。
