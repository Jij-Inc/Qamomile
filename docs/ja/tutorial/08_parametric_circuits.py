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
# # パラメトリック回路と変分量子アルゴリズム
#
# このチュートリアルでは、Qamomile における**パラメトリック回路**の使い方を学び、
# それを応用して**変分量子分類器**をゼロから構築します。
# パラメトリック回路は、量子回路と古典的最適化を組み合わせた変分量子アルゴリズム（VQA）の基盤となるものです。
#
# ## 学習内容
# - パラメトリック回路が変分量子アルゴリズムにとって重要な理由
# - トランスパイル時の `bindings=` と `parameters=` の違い
# - `Observable` と `expval()` を用いた期待値の計算方法
# - 回転ゲートによるデータエンコーディング
# - 変分量子分類器の段階的な構築
# - 量子・古典ハイブリッド最適化ループの実行

# %%
import math

import numpy as np

import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 1. Qamomile におけるパラメトリック回路
#
# ### パラメトリック回路が必要な理由
#
# 多くの量子アルゴリズムは**変分的**なアプローチを取ります：
# パラメータ化された量子状態を準備し、コスト関数を測定し、古典的なオプティマイザでパラメータを調整します。
# これが **VQE**、**QAOA**、**量子機械学習** などの VQA の核となる考え方です。
#
# これらのアルゴリズムでは、**回路全体を再構築することなく**、
# 実行間でパラメータを効率的に変更できる回路が必要です。

# %% [markdown]
# ### トランスパイル時の `bindings=` と `parameters=`
#
# `qmc.Float` パラメータは、2つの異なるタイミングで解決できます：
#
# | 仕組み | 解決タイミング | ユースケース |
# |-----------|---------------|----------|
# | `bindings=` | **トランスパイル時** | 回路構造に影響する値（配列サイズ、ループ回数、ハミルトニアン） |
# | `parameters=` | **実行時** | 実行ごとに変更可能な値（最適化のための回転角度） |
#
# パラメータ名を `parameters=` に指定すると、Qamomile はトランスパイル後の回路にそのパラメータを
# シンボリック変数として保持します。再トランスパイルなしに、実行時に異なる値を渡すことができます。

# %% [markdown]
# ### 例：シンプルなパラメータ付き回転
#
# 1量子ビットの回路で実際に動作を確認しましょう。


# %%
@qmc.qkernel
def param_rotation(theta: qmc.Float) -> qmc.Bit:
    """A single RY rotation with a tunable angle."""
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


param_rotation.draw()

# %%
# Transpile with theta as a free parameter (not a fixed binding)
executable_rot = transpiler.transpile(param_rotation, parameters=["theta"])

# Execute with different parameter values — no retranspilation needed
print("=== Parameterized Rotation: theta sweep ===\n")

for theta_val in [0.0, math.pi / 4, math.pi / 2, math.pi]:
    result = executable_rot.sample(
        transpiler.executor(), bindings={"theta": theta_val}, shots=1000
    ).result()

    counts = {str(v): c for v, c in result.results}
    p1 = counts.get("1", 0) / 1000
    print(f"  theta = {theta_val:.4f}  ->  P(1) = {p1:.3f}")

# %% [markdown]
# `theta` が 0 から $\pi$ に増加するにつれて、`1` を測定する確率は
# $P(1) = \sin^2(\theta/2)$ に従って 0 から 1 へと変化します。
#
# 重要なのは、**`transpile()` の呼び出しは1回だけ**ということです。
# `executable_rot.sample(...)` を異なる `bindings` で呼び出すたびに、
# シンボリックパラメータに新しい値が代入されるだけです。

# %% [markdown]
# ### 使い分けの指針
#
# - **`bindings=`**：回路の**構造**に影響する値（量子ビット数、ループ回数、
#   辺のリスト、ハミルトニアン）。変更するには新たに `transpile()` を呼び出す必要があります。
# - **`parameters=`**：**最適化**や**スイープ**の対象となる値（回転角度）。
#   実行時に自由に変更できます。

# %% [markdown]
# ## 2. オブザーバブルと期待値
#
# 変分アルゴリズムは量子オブザーバブルの**期待値**を計算します。
# Qamomile は `Observable` 型と `expval()` 演算によりこれをサポートしています。
#
# ### ハミルトニアンの構築

# %%
import qamomile.observable as qmo

# Single Pauli operators on specific qubits
Z0 = qmo.Z(0)  # Z operator on qubit 0
Z1 = qmo.Z(1)  # Z operator on qubit 1

# Combine with arithmetic
hamiltonian_simple = Z0 + 0.5 * Z0 * Z1

print("Hamiltonian:", hamiltonian_simple)

# %% [markdown]
# ### QKernel 内での `qmc.expval()` の使用
#
# `qmc.expval()` 関数は量子ビット配列と `Observable` パラメータを受け取り、
# $\langle \psi | H | \psi \rangle$ を表す `Float` を返します。
#
# `Observable` 型は特別なパラメータ型で、常に `bindings` を通じて提供されます
# （ハミルトニアンが測定の構造を決定するため）。


# %%
@qmc.qkernel
def simple_vqe(theta: qmc.Float, H: qmc.Observable) -> qmc.Float:
    """Prepare a parameterized state and compute <psi|H|psi>."""
    q = qmc.qubit_array(1, name="q")
    q[0] = qmc.ry(q[0], theta)
    return qmc.expval(q, H)


simple_vqe.draw()

# %% [markdown]
# トランスパイルして theta をスイープし、期待値がどのように変化するか見てみましょう。
#
# **`run()` と `sample()` の違い**：このカーネルは `qmc.expval()` を通じて `Float` を返す
# （測定結果ではない）ため、`executable.sample()` ではなく `executable.run()` を使用します。
# `run()` は期待値を計算して `Float` を直接返しますが、`sample()` はショットベースの測定を行い、
# ビット文字列のカウントを含む `SampleResult` を返します。
# 概要については [01_introduction](01_introduction.ipynb) を参照してください。

# %%
# Build Hamiltonian: H = Z_0
H_z = qmo.Z(0)

# Transpile: H is bound (structure), theta is a free parameter
executable_vqe = transpiler.transpile(
    simple_vqe,
    bindings={"H": H_z},
    parameters=["theta"],
)

# Sweep theta and compute expectation values
thetas = np.linspace(0, 2 * np.pi, 21)
energies = []

for theta_val in thetas:
    job = executable_vqe.run(
        transpiler.executor(),
        bindings={"theta": float(theta_val)},
    )
    energy = job.result()
    energies.append(energy)

# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(thetas, energies, "o-", markersize=4)
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\langle Z \rangle$")
plt.title(r"Expectation value $\langle \psi(\theta) | Z | \psi(\theta) \rangle$")
plt.axhline(y=-1, color="r", linestyle="--", alpha=0.5, label="Ground state energy")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# $RY(\theta)|0\rangle$ に対する期待値は $\langle Z \rangle = \cos(\theta)$ です。
# $\theta = \pi$ での最小値 $-1$ は基底状態 $|1\rangle$ に対応します。
# この変分的アプローチは、量子最適化（QAOA）と量子機械学習の両方の基盤となっています。

# %% [markdown]
# ## 3. シンプルな変分量子分類器
#
# ここまでに学んだことを活用して、**変分量子分類器**を構築しましょう。
# これは、データ点を2つのカテゴリに分類することを学習する量子回路です。
#
# アイデアはシンプルです：
# 1. 古典データを量子ビットの回転角度として**エンコード**する
# 2. 学習可能な変分レイヤーを**適用**し、各レイヤーでデータを**再エンコード**する
# 3. オブザーバブルを**測定**する ── その期待値が予測となる
# 4. 分類誤差を最小化するように回路パラメータを**最適化**する
#
# 深い機械学習の知識は不要です。量子回路による関数フィッティングにすぎません。

# %% [markdown]
# ### データセット
#
# シンプルな2次元の二値分類問題を生成します：2つのクラスタからなる点群です。

# %%
np.random.seed(901)
n_samples = 15

# Class 0: cluster centered at (-0.5, 0)
X0 = np.random.randn(n_samples, 2) * 0.3 + np.array([-0.5, 0.0])
# Class 1: cluster centered at (+0.5, 0)
X1 = np.random.randn(n_samples, 2) * 0.3 + np.array([+0.5, 0.0])

X_data = np.vstack([X0, X1]) * np.pi  # Scale for angle encoding
y_data = np.array([0] * n_samples + [1] * n_samples)

plt.figure(figsize=(6, 4))
plt.scatter(
    X_data[:n_samples, 0], X_data[:n_samples, 1], c="blue", label="Class 0", marker="o"
)
plt.scatter(
    X_data[n_samples:, 0], X_data[n_samples:, 1], c="red", label="Class 1", marker="x"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Training Data")
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# ### データエンコーディング
#
# **角度エンコーディング**は、各特徴量を量子ビットの回転角度にマッピングします。
# 2つの特徴量と2つの量子ビットで、量子ビット $i$ に $RY(x_i)$ を適用します。
#
# algorithm モジュールの `ry_layer` 関数がまさにこれを行います。
# パラメータベクトルから指定されたオフセット以降の各量子ビットに $RY$ 回転を適用します。
# データベクトル `x` をオフセット 0 で渡すことで、`ry_layer` が角度エンコーダとして機能します。
#
# また、生の特徴量を $\pi$ でスケーリングすることで、$\pm 0.5$ のクラスタ中心が
# 回転角度 $\pm \pi/2$ にマッピングされ、2つのクラスに対してほぼ直交した量子状態が得られます。


# %% [markdown]
# ### 分類器回路
#
# Qamomile の algorithm モジュールにある `ry_layer` と `cz_entangling_layer` を使って
# 分類器を構築します（[05_stdlib](05_stdlib.ipynb) を参照）。
#
# ここでの重要なテクニックは**データ再アップロード**です：入力データを最初だけでなく、
# *すべての*変分レイヤーでエンコードします。これにより、データと学習可能な回転、
# エンタングルメントが交互に配置され、回路の関数近似能力が大幅に向上します
# （ニューラルネットワークが各層で重みを適用するのと同様です）。
#
# 回路の構造（各レイヤーで繰り返し）：
# 1. $RY$ データエンコーディング ── `ry_layer(qubits, x, 0)`
# 2. $RY$ 変分回転 ── `ry_layer(qubits, params, offset)`
# 3. $CZ$ エンタングルメント ── `cz_entangling_layer(qubits)`
#
# 2量子ビット、2レイヤーで、**4つの学習可能なパラメータ**があります。

# %%
from qamomile.circuit.algorithm import cz_entangling_layer, ry_layer


@qmc.qkernel
def classifier(
    x: qmc.Vector[qmc.Float],
    params: qmc.Vector[qmc.Float],
    H: qmc.Observable,
) -> qmc.Float:
    """Variational quantum classifier with data re-uploading.

    Args:
        x: Input features (2D data point, pre-scaled)
        params: Trainable parameters (4 values for 2 layers × 2 qubits)
        H: Observable to measure (Z on qubit 0)
    """
    qubits = qmc.qubit_array(2, name="q")
    n = qubits.shape[0]

    for layer in qmc.range(2):
        # Data encoding (re-uploaded each layer)
        qubits = ry_layer(qubits, x, 0)
        # Variational rotations + entanglement
        qubits = ry_layer(qubits, params, layer * n)
        qubits = cz_entangling_layer(qubits)

    return qmc.expval(qubits, H)


classifier.draw(fold_loops=False, inline=True)

# %% [markdown]
# ## 4. トランスパイル：bindings と parameters の実践
#
# これは `bindings=` と `parameters=` の違いを示す好例です：
#
# - **`H`（Observable）** は `binding` です ── 測定の構造を決定します
# - **`x`（データ）と `params`（学習可能な重み）** は `parameters` です ──
#   再トランスパイルなしに実行ごとに変更できます
#
# 一度トランスパイルすれば、任意のデータ点と任意のパラメータ値で評価できます。

# %%
# Build the measurement observable: Z on qubit 0.
# The num_qubits parameter ensures the observable matches the 2-qubit circuit.
H_label = qmo.Hamiltonian(num_qubits=2)
H_label += qmo.Z(0)

executable = transpiler.transpile(
    classifier,
    bindings={"H": H_label},
    parameters=["x", "params"],
)

# Quick test: evaluate with a sample data point
test_expval = executable.run(
    transpiler.executor(),
    bindings={"x": [0.5, -0.3], "params": [0.1, 0.2, 0.3, 0.4]},
).result()
print(f"Test expectation value: {test_expval:.4f}")

# %% [markdown]
# ## 5. 分類器の学習
#
# 学習ループは、すべての変分アルゴリズムで使われる量子・古典ハイブリッドのパターンと同じです：
# 量子回路が予測を計算し、古典的なオプティマイザがパラメータを調整して損失を最小化します。
#
# ### ラベルの対応
#
# クラスラベルを $Z$ の固有値にマッピングします：
# - クラス 0 → 目標 $\langle Z \rangle = +1$
# - クラス 1 → 目標 $\langle Z \rangle = -1$
#
# 損失関数は、予測値と目標値の平均二乗誤差です。

# %%
loss_history = []


def classification_loss(params_flat, executable, transpiler, X_train, y_train):
    """Compute MSE between predicted <Z> and target labels."""
    total_loss = 0.0
    for xi, yi in zip(X_train, y_train):
        target = 1.0 - 2.0 * yi  # 0 → +1, 1 → −1
        pred = executable.run(
            transpiler.executor(),
            bindings={"x": xi.tolist(), "params": params_flat.tolist()},
        ).result()
        total_loss += (pred - target) ** 2
    mse = total_loss / len(X_train)
    loss_history.append(mse)
    return mse


# %% [markdown]
# ### 最適化の実行
#
# `scipy.optimize.minimize` の COBYLA 法を使用します。
# COBYLA はノイズのある量子目的関数に適した勾配フリーのオプティマイザです。

# %%
from scipy.optimize import minimize

# Initial random parameters (4 values for 2 layers × 2 qubits)
n_params = 4
init_params = np.random.uniform(-np.pi, np.pi, size=n_params)

print(f"Initial parameters: {init_params}")

# Clear history
loss_history = []

# Run COBYLA optimization
result_opt = minimize(
    classification_loss,
    init_params,
    args=(executable, transpiler, X_data, y_data),
    method="COBYLA",
    options={"maxiter": 80, "disp": True},
)

print(f"\nOptimized parameters: {result_opt.x}")
print(f"Final loss: {result_opt.fun:.4f}")

# %% [markdown]
# ### 収束の可視化

# %%
plt.figure(figsize=(8, 4))
plt.plot(loss_history, marker="o", markersize=3)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Classifier Training Convergence")
plt.grid(True)
plt.show()

# %% [markdown]
# ## 6. 分類器の評価
#
# 学習済みの分類器がデータセットに対してどの程度正しく分類できるか確認しましょう。
# $\langle Z \rangle > 0$ ならクラス 0、それ以外ならクラス 1 と予測します。

# %%
optimal_params = result_opt.x

# Predict on training data
predictions = []
for xi in X_data:
    pred = executable.run(
        transpiler.executor(),
        bindings={"x": xi.tolist(), "params": optimal_params.tolist()},
    ).result()
    predictions.append(pred)

predictions = np.array(predictions)
predicted_labels = (predictions < 0).astype(int)  # <Z> < 0 → class 1

accuracy = np.mean(predicted_labels == y_data)
print(f"Classification accuracy: {accuracy:.1%}")
print("\nPer-sample predictions:")
for i, (xi, yi, pred, label) in enumerate(
    zip(X_data, y_data, predictions, predicted_labels)
):
    status = "correct" if label == yi else "WRONG"
    print(f"  [{i:2d}] true={yi}, pred_label={label}, <Z>={pred:+.3f}  {status}")

# %% [markdown]
# ### 決定境界の可視化
#
# グリッド上の各点で $\langle Z \rangle$ を評価し、決定境界を可視化します。

# %%
# Create a grid (matching the π-scaled feature range)
grid_range = np.linspace(-3.0, 3.0, 25)
xx, yy = np.meshgrid(grid_range, grid_range)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

# Evaluate classifier on grid
grid_predictions = []
for point in grid_points:
    pred = executable.run(
        transpiler.executor(),
        bindings={"x": point.tolist(), "params": optimal_params.tolist()},
    ).result()
    grid_predictions.append(pred)

grid_predictions = np.array(grid_predictions).reshape(xx.shape)

# Plot
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, grid_predictions, levels=20, cmap="RdBu", alpha=0.7)
plt.colorbar(label=r"$\langle Z \rangle$")
plt.contour(xx, yy, grid_predictions, levels=[0.0], colors="black", linewidths=2)
plt.scatter(
    X_data[:n_samples, 0],
    X_data[:n_samples, 1],
    c="blue",
    edgecolors="k",
    label="Class 0",
    marker="o",
)
plt.scatter(
    X_data[n_samples:, 0], X_data[n_samples:, 1], c="red", label="Class 1", marker="x"
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Quantum Classifier Decision Boundary")
plt.legend()
plt.show()

# %% [markdown]
# 黒い線が決定境界（$\langle Z \rangle = 0$）を示しています。
# 青い領域はクラス 0、赤い領域はクラス 1 と予測されます。

# %% [markdown]
# ## 7. まとめ
#
# このチュートリアルでは、パラメトリック量子回路の基本概念を学び、
# それを応用して変分量子分類器を構築しました。
#
# ### 重要なポイント
#
# 1. **パラメトリック回路**は変分量子アルゴリズムの基盤です。
#    Qamomile では `parameters=` で自由パラメータを宣言でき、
#    再トランスパイルなしに実行時にパラメータ値を変更できます。
#
# 2. **`bindings=` と `parameters=` の違い**：
#    - `bindings`：トランスパイル時に固定される（問題の構造、ハミルトニアン）
#    - `parameters`：実行時まで自由な値（回転角度、入力データ）
#
# 3. **Observable と `expval()`**：Qamomile は `Observable` 型と `qmc.expval()` を用いて、
#    qkernel 内で直接期待値 $\langle \psi | H | \psi \rangle$ を計算できます。
#
# 4. **変分量子分類器**：
#    - データエンコーディング：特徴量をスケーリングし、量子ビットの回転にマッピング
#    - データ再アップロード：各レイヤーでデータをエンコードし、表現力を向上
#    - 変分アンザッツ：学習可能な回転 + エンタングルメントレイヤー
#    - 予測：パウリオブザーバブルの期待値
#    - 学習：古典オプティマイザが損失関数を最小化
#
# 5. **ハイブリッド最適化ループ**：量子最適化（QAOA）と量子機械学習の両方で、
#    同じ量子・古典ハイブリッドパターンが適用されます。
#
# ### 次のステップ
#
# - [QAOA](../optimization/qaoa.ipynb)：Qamomile の組み込みコンバータが組合せ最適化問題をどのように扱うかを確認
# - [リソース見積もり](09_resource_estimation.ipynb)：回路の深さやゲート数の見積もり
# - [カスタムエグゼキュータ](11_custom_executor.ipynb)：クラウド量子ハードウェアでの回路実行

# %% [markdown]
# ## 学習した内容
#
# - **パラメトリック回路が変分量子アルゴリズムにとって重要な理由** ── 同じ回路構造を異なるパラメータ値で再利用でき、古典・量子の最適化ループが可能になります。
# - **トランスパイル時の `bindings=` と `parameters=` の違い** ── `bindings` はトランスパイル時に値を固定し（問題構造）、`parameters` は実行時まで自由なままにします（学習可能な角度）。
# - **`Observable` と `expval()` を用いた期待値の計算方法** ── `qmc.expval(qubits, H)` は $\langle\psi|H|\psi\rangle$ を直接計算し、`run()` を通じて `Float` を返します。
# - **回転ゲートによるデータエンコーディング** ── 古典的な特徴量をスケーリングし、`ry_layer` で量子ビットの回転にマッピングします。各変分レイヤーでデータを再アップロードする（データ再アップロード）ことで、回路の表現力が大幅に向上します。
# - **変分量子分類器の段階的な構築** ── データエンコーディングレイヤー、学習可能な回転 + エンタングルメントアンザッツレイヤー、およびパウリ Z オブザーバブルを組み合わせた二値分類器です。
# - **量子・古典ハイブリッド最適化ループの実行** ── 古典オプティマイザ（例：scipy）が、量子期待値から計算された損失関数を最小化するように回路パラメータを更新します。
