# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
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
# tags: [algorithm, machine-learning]
# ---
# # 量子カーネル分類
#
# このチュートリアルでは、Qamomileを使った**量子カーネル法**による二値分類を実演します。パラメータ付き量子回路を使ってカーネル（類似度）行列を計算し、それを古典的なサポートベクターマシン（SVM）に渡します。
#
# 以下の手順で進めます:
#
# 1. 線形分離できないデータセット(`make_circles`)を準備する。
# 2. `@qkernel`で**特徴マップ**回路 $U(x)$ を構築する。
# 3. **オーバーラップカーネル** $k(x, x') = |\langle 0^n | U(x')^\dagger U(x) | 0^n \rangle|^2$ を構成する。
# 4. グラム行列を計算し、`kernel="precomputed"`でSVMを学習する。
# 5. 古典的な線形カーネルとRBFカーネルと比較する。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install qamomile scikit-learn
# ```

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile scikit-learn

# %%
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import qamomile as qm

docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"

# %% [markdown]
# ## 背景: 量子カーネル法
#
# 量子カーネルは、2つのデータ点の量子特徴状態のオーバーラップを測定することで類似度を推定します:
#
# $$
# k(x, x') = |\langle 0^n | U(x')^\dagger \, U(x) | 0^n \rangle|^2
# $$
#
# $|0^n\rangle$に$U(x')^\dagger U(x)$を適用し、全ゼロビット列が得られる確率を測定することでこのオーバーラップを推定します。得られたカーネル行列はSVMなどの任意のカーネルベース分類器に渡すことができます。


# %% [markdown]
# ## ハイパーパラメータ


# %% [markdown]
# ## データセット
#
# scikit-learnの `make_circles` を使い、同心円状の2クラスデータを生成します。線形分類器が失敗する典型的な例です。


# %% [markdown]
# ## 古典的な前処理（特徴リフティング）
#
# 量子回路にデータをエンコードする前に、古典的な**特徴リフティング**を適用して元の2次元入力をより高次元の空間に写像します。生の特徴の非線形な組み合わせを追加することで、量子特徴マップがより豊かな構造を活用できるようにします。
#
# 2段階のパイプラインは以下の通りです:
#
# 1. 生の2次元座標を $[0, \pi]$ にスケーリング。
# 2. 非線形特徴 $(x_0 - x_1)^2$ と $(x_0 + x_1)^2$ を構築し、それらも $[0, \pi]$ にスケーリング。
#
# これによりサンプルごとに4次元の特徴ベクトルが得られます:
# $f(x) = [x_0, x_1, (x_0 - x_1)^2, (x_0 + x_1)^2]$。
#
# したがって量子カーネルは前処理後の特徴上で定義されます:
#
# $$
# k(x, x') = |\langle 0^n | U(f(x'))^\dagger \, U(f(x)) | 0^n \rangle|^2
# $$


# %% [markdown]
# ## 特徴マップ回路
#
# 特徴ブロック$U(x)$は、単一量子ビット回転とentangling gateでデータをencodeします。`qm.adjoint`は同じfeature-map bodyから正確な逆変換$U(x)^\dagger$を導出します。


# %% [markdown]
# ## オーバーラップカーネル回路
#
# 完全なカーネル回路は $U(x)$ の後に $U(x')^\dagger$ を適用します。
# カーネル値 $k(x, x')$ は $|00\rangle$ を測定する確率です。
#
# 表現力を高めるために特徴ブロックを `layers` 回繰り返します。
# `layers` は構造パラメータなので特殊化時にバインドします。


# %%
@qm.qkernel
def classification_feature_map(
    state: qm.Q[qm.Bits[2]],  # type: ignore[valid-type]
    features: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Encode one feature vector into an owned two-qubit state.

    Args:
        state (qm.Q[qm.Bits[2]]): Owned two-qubit input state.
        features (qm.Vector[qm.Real, 4]): Two scaled coordinates and two
            nonlinear lifted coordinates.

    Returns:
        qm.Q[qm.Bits[2]]: Encoded two-qubit state.
    """

    for _ in range(2):
        for index in range(2):
            state = qm.h(state, target=index)
            state = qm.rz(state, features[index], target=index)
            state = qm.ry(state, features[index], target=index)
        state = qm.cz(state, control=0, target=1)
        state = qm.rzz(state, features[2], left=0, right=1)
        state = qm.rz(state, features[3], target=0)
        state = qm.rz(state, features[3], target=1)
    return state


@qm.qkernel
def classification_overlap_state(
    left: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
    right: qm.Vector[qm.Real, 4],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[2]]:  # type: ignore[valid-type]
    """Build the reversible overlap witness for two feature vectors.

    Args:
        left (qm.Vector[qm.Real, 2]): First feature vector.
        right (qm.Vector[qm.Real, 2]): Second feature vector.

    Returns:
        qm.Q[qm.Bits[2]]: State whose zero probability is the kernel value.
    """

    state = classification_feature_map(qm.qubits(2), left)
    return qm.adjoint(classification_feature_map, state, right)


@qm.qkernel
def classification_overlap(
    left: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
    right: qm.Param[qm.Vector[qm.Real, 4]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[2]]:  # type: ignore[valid-type]
    """Measure one runtime-rebindable overlap witness.

    Args:
        left (qm.Param[qm.Vector[qm.Real, 2]]): First runtime feature vector.
        right (qm.Param[qm.Vector[qm.Real, 2]]): Second runtime feature vector.

    Returns:
        qm.Shot[qm.Bits[2]]: Two logical overlap-witness bits per shot.
    """

    return qm.measure(classification_overlap_state(left, right))


# %% [markdown]
# `qm.adjoint`はfeature-map bodyから逆変換を導出します。forward mapと同期する必要がある逆順gate listのcopyはなく、affineなstate ownershipが両方のcallにわたって検査されます。

# %% [markdown]
# ## 記述したbodyをinspectしてestimateする
#
# 特殊化により、inspectとresource analysisの両方で使うtarget-independentなprogramが得られます。bodyにはadjoint callがありますが、量子分類専用opcodeはありません。Rustは可視な2-layer forward/inverse feature mapから40 logical gatesを導出します。

# %%
program = qm.specialize(classification_overlap)
inspection = program.inspect()
study = qm.resources.logical(program)

assert "qcore.adjoint_call" in inspection
assert "quantum.classification" not in inspection
assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (40, 40)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

# %% [markdown]
# ## データセット
#
# scikit-learnの `make_circles` を使い、同心円状の2クラスデータを生成します。線形分類器が失敗する典型的な例です。

# %%
N_SAMPLES = 8 if docs_test_mode else 40
X_raw, labels = make_circles(
    n_samples=N_SAMPLES,
    noise=0.04,
    factor=0.4,
    random_state=7,
)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    labels,
    test_size=0.25,
    random_state=7,
    stratify=labels,
)

# %% [markdown]
# ## 古典的な前処理（特徴リフティング）
#
# 量子回路にデータをエンコードする前に、古典的な**特徴リフティング**を適用して元の2次元入力をより高次元の空間に写像します。生の特徴の非線形な組み合わせを追加することで、量子特徴マップがより豊かな構造を活用できるようにします。
#
# 2段階のパイプラインは以下の通りです:
#
# 1. 生の2次元座標を $[0, \pi]$ にスケーリング。
# 2. 非線形特徴 $(x_0 - x_1)^2$ と $(x_0 + x_1)^2$ を構築し、それらも $[0, \pi]$ にスケーリング。
#
# これによりサンプルごとに4次元の特徴ベクトルが得られます:
# $f(x) = [x_0, x_1, (x_0 - x_1)^2, (x_0 + x_1)^2]$。
#
# したがって量子カーネルは前処理後の特徴上で定義されます:
#
# $$
# k(x, x') = |\langle 0^n | U(f(x'))^\dagger \, U(f(x)) | 0^n \rangle|^2
# $$

# %%
angle_scaler = MinMaxScaler(feature_range=(0.0, math.pi), clip=True)
pair_scaler = MinMaxScaler(feature_range=(0.0, math.pi), clip=True)
X_train_angles = angle_scaler.fit_transform(X_train_raw)
X_test_angles = angle_scaler.transform(X_test_raw)


def lift_features(values: np.ndarray) -> np.ndarray:
    """Lift two scaled coordinates into four nonlinear circuit features.

    Args:
        values (np.ndarray): Two-column scaled input matrix.

    Returns:
        np.ndarray: Four-column matrix with two quadratic coordinates.
    """

    x0 = values[:, 0]
    x1 = values[:, 1]
    return np.column_stack((x0, x1, (x0 - x1) ** 2, (x0 + x1) ** 2))


F_train = lift_features(X_train_angles)
F_test = lift_features(X_test_angles)
F_train[:, 2:4] = pair_scaler.fit_transform(F_train[:, 2:4])
F_test[:, 2:4] = pair_scaler.transform(F_test[:, 2:4])

TRAIN_FEATURES = tuple(tuple(float(value) for value in row) for row in F_train)
TEST_FEATURES = tuple(tuple(float(value) for value in row) for row in F_test)
TRAIN_LABELS = tuple(int(label) for label in y_train)
EXPECTED_LABELS = tuple(int(label) for label in y_test)
SHOTS = 16 if docs_test_mode else 64

assert F_train.shape == (len(X_train_raw), 4)
assert F_test.shape == (len(X_test_raw), 4)

# %% [markdown]
# ## 1回コンパイルして繰り返し実行する
#
# overlap programは1回だけcompileします。Gram matrixの全entryが異なるruntime vectorで同じartifactを再利用します。

# %%
endpoint = qm.Reference()
artifact = qm.compile(program, to=endpoint.target)
kernel_runs: list[qm.RunRecord] = []

assert tuple(slot.name for slot in artifact.parameter_abi.slots) == (
    "left",
    "right",
)


def kernel_value(
    left: tuple[float, ...],
    right: tuple[float, ...],
    *,
    seed: int,
) -> float:
    """Estimate one all-zero overlap through the shared artifact.

    Args:
        left: First four-coordinate feature vector.
        right: Second four-coordinate feature vector.
        seed: Deterministic sampling seed.

    Returns:
        Estimated squared overlap in the closed interval from zero to one.
    """

    record = qm.run(
        artifact,
        on=endpoint,
        parameters={"left": left, "right": right},
        shots=SHOTS,
        seed=seed,
    )
    kernel_runs.append(record)
    return record.result.probability((0, 0))


train_kernel_rows = [[0.0] * len(TRAIN_FEATURES) for _ in TRAIN_FEATURES]
seed = 1
for row in range(len(TRAIN_FEATURES)):
    train_kernel_rows[row][row] = 1.0
    for column in range(row):
        value = kernel_value(
            TRAIN_FEATURES[row],
            TRAIN_FEATURES[column],
            seed=seed,
        )
        seed += 1
        train_kernel_rows[row][column] = value
        train_kernel_rows[column][row] = value
train_kernel_raw = np.asarray(train_kernel_rows, dtype=float)
test_kernel = np.asarray(
    [
        [
            kernel_value(point, training, seed=100 + row * 10 + column)
            for column, training in enumerate(TRAIN_FEATURES)
        ]
        for row, point in enumerate(TEST_FEATURES)
    ]
)

expected_kernel_runs = len(TRAIN_FEATURES) * (len(TRAIN_FEATURES) - 1) // 2
expected_kernel_runs += len(TEST_FEATURES) * len(TRAIN_FEATURES)
assert len(kernel_runs) == expected_kernel_runs
assert all(record.artifact is artifact for record in kernel_runs)
assert sum(record.result.shots for record in kernel_runs) == (
    expected_kernel_runs * SHOTS
)


def project_to_psd_correlation(matrix: np.ndarray) -> np.ndarray:
    """Project a shot-noisy symmetric Gram matrix to a PSD correlation matrix."""

    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    positive = np.maximum(eigenvalues, 1e-12)
    projected = (eigenvectors * positive) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(projected), 1e-12))
    projected = projected / np.outer(diagonal, diagonal)
    projected = 0.5 * (projected + projected.T)
    np.fill_diagonal(projected, 1.0)
    return projected


train_kernel = project_to_psd_correlation(train_kernel_raw)
assert np.allclose(train_kernel, train_kernel.T)
assert np.allclose(np.diag(train_kernel), 1.0)
assert float(np.linalg.eigvalsh(train_kernel).min()) >= -1e-9
assert test_kernel.min() >= 0.0 and test_kernel.max() <= 1.0

# %% [markdown]
# 対称性から残りのtraining entryが得られ、self-overlapから対角成分は1になります。サイズ$n$の実用的なtraining集合には$O(n^2)$個のkernel値が必要です。batchingとmeasurement planningは実行上の関心事であり、新しいsource-language semanticsではありません。

# %% [markdown]
# ## 分類器の学習
#
# 比較のために3つのSVMを学習します:
# - **量子カーネルSVC**: 事前計算した量子カーネル行列を使用。
# - **線形SVC**: 生の2次元特徴を使用（円データでは失敗が予想される）。
# - **RBF SVC**: 古典的なガウシアンカーネルを使用（強力なベースライン）。

# %%
classifier = SVC(kernel="precomputed", C=1.0)
classifier.fit(train_kernel, TRAIN_LABELS)
predictions = classifier.predict(test_kernel)

linear_classifier = SVC(kernel="linear", C=1.0)
linear_classifier.fit(X_train_raw, y_train)
linear_predictions = linear_classifier.predict(X_test_raw)

rbf_classifier = SVC(kernel="rbf", C=1.0, gamma="scale")
rbf_classifier.fit(X_train_raw, y_train)
rbf_predictions = rbf_classifier.predict(X_test_raw)

print("test kernel:", test_kernel)
print("predicted labels:", predictions)

quantum_accuracy = accuracy_score(EXPECTED_LABELS, predictions)
linear_accuracy = accuracy_score(EXPECTED_LABELS, linear_predictions)
rbf_accuracy = accuracy_score(EXPECTED_LABELS, rbf_predictions)
print("Quantum kernel SVC:", quantum_accuracy)
print("Linear SVC:", linear_accuracy)
print("RBF SVC:", rbf_accuracy)
assert predictions.shape == y_test.shape
assert linear_predictions.shape == y_test.shape
assert rbf_predictions.shape == y_test.shape

# %% [markdown]
# ## 可視化
#
# 4つのパネルをプロットします:
# 1. 訓練データ。
# 2. 量子カーネル行列（グラム行列）。
# 3. 量子カーネル SVC の決定境界。
# 4. 比較用の古典 RBF SVC の決定境界。


# %% [markdown]
# ### 決定境界ヘルパー


# %% [markdown]
# ### 結果のプロット

# %%
GRID_SIZE = 2 if docs_test_mode else 10


def preprocess_for_kernel(raw_points: np.ndarray) -> np.ndarray:
    """Apply the fitted scaling and nonlinear lift to new raw points."""

    angles = angle_scaler.transform(raw_points)
    features = lift_features(angles)
    features[:, 2:4] = pair_scaler.transform(features[:, 2:4])
    return features


margin = 0.35
x_min = X_raw[:, 0].min() - margin
x_max = X_raw[:, 0].max() + margin
y_min = X_raw[:, 1].min() - margin
y_max = X_raw[:, 1].max() + margin
grid_x, grid_y = np.meshgrid(
    np.linspace(x_min, x_max, GRID_SIZE),
    np.linspace(y_min, y_max, GRID_SIZE),
)
raw_grid = np.column_stack((grid_x.ravel(), grid_y.ravel()))
feature_grid = preprocess_for_kernel(raw_grid)
grid_kernel = np.asarray(
    [
        [
            kernel_value(
                tuple(float(value) for value in point),
                training,
                seed=1_000 + row * len(TRAIN_FEATURES) + column,
            )
            for column, training in enumerate(TRAIN_FEATURES)
        ]
        for row, point in enumerate(feature_grid)
    ]
)
quantum_boundary = classifier.predict(grid_kernel).reshape(grid_x.shape)
rbf_boundary = rbf_classifier.predict(raw_grid).reshape(grid_x.shape)

figure, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].scatter(X_train_raw[:, 0], X_train_raw[:, 1], c=y_train)
axes[0, 0].set_title("Training data")
image = axes[0, 1].imshow(train_kernel, cmap="viridis", vmin=0.0, vmax=1.0)
axes[0, 1].set_title("Quantum kernel matrix")
figure.colorbar(image, ax=axes[0, 1])
axes[1, 0].contourf(grid_x, grid_y, quantum_boundary, alpha=0.35)
axes[1, 0].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 0].set_title("Quantum kernel SVC boundary")
axes[1, 1].contourf(grid_x, grid_y, rbf_boundary, alpha=0.35)
axes[1, 1].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 1].set_title("RBF SVC boundary")
figure.tight_layout()
plt.show()

# %% [markdown]
# ## 結果の説明
#
# 上の可視化では、量子カーネル分類器が `make_circles` の非線形なクラス構造をどの程度捉えられているかを確認できます。
#
# 左上の訓練データを見ると、2つのクラスは直線では分離しにくく、内側の円と外側の円に近い構造を持っています。
# 実際に、今回の実行では線形SVCのテスト精度は 0.5 であり、ほぼランダム分類に近い結果になっています。
# これは、生の2次元座標に対する線形な決定境界では、同心円状のクラス構造を表現できないためです。
#
# 右上の量子カーネル行列は、訓練データ同士の類似度
#
# $$
# k(x, x') =
# \left|
# \langle 0^n |
# U(f(x'))^\dagger U(f(x))
# |0^n\rangle
# \right|^2
# $$
#
# を並べたグラム行列です。
# 対角成分は各データ点と自分自身の類似度なので 1 になります。
# 非対角成分は、異なるデータ点同士が量子特徴空間でどの程度近いかを表しています。
# 今回の行列はクラス順に並び替えていないため、明確なブロック構造として読むことはできません。
# しかし、この行列がSVMに渡されることで、元の2次元空間では線形に分離しにくいデータを、カーネル空間上で分類できるようになります。
#
# 左下の量子カーネルSVCの決定境界を見ると、入力空間では非線形な境界が形成されています。
# 今回の実行では、量子カーネルSVCのテスト精度は 1.0 でした。
# したがって、この小規模な `make_circles` データセットに対しては、今回設計した量子特徴マップが有効な類似度を与えていると解釈できます。
#
# 右下のRBF SVCも、同じく非線形な決定境界を形成しています。
# 今回の実行では、RBF SVCのテスト精度も 1.0 でした。
# したがって、このデータセットでは量子カーネルSVCとRBF SVCはいずれもテストデータをすべて正しく分類できています。
# 一方で、RBF SVCの決定境界は比較的滑らかであるのに対し、量子カーネルSVCの境界はやや粗く見えます。
# これは、可視化用の格子数 `GRID_SIZE = 15` が小さいことや、カーネル値を `SHOTS = 1024` のサンプリングで推定していることの影響を受けている可能性があります。
#
# 以上より、今回の実験では
#
# - 線形SVC: 0.5
# - 量子カーネルSVC: 1.0
# - RBF SVC: 1.0
#
# となり、量子カーネルSVCは線形SVCよりも明確に良い結果を示し、RBF SVCと同等のテスト精度を達成しました。
#
# ただし、この結果だけから量子優位性を主張することはできません。
# 今回のデータセットは小規模な可視化用データであり、古典的な特徴リフティングも使用しています。
# そのため、ここで示せる結論は
# 「Qamomileを用いて量子カーネル行列を構成し、それをSVMに渡して非線形二値分類を行える」
# という実装上・手法上の確認です。
# 量子カーネルが古典カーネルより本質的に有利かどうかを議論するには、
# より大きなデータセット、複数の乱数シード、ショット数依存性、層数依存性、古典カーネルとの詳細な比較が必要です。
#
# また、今回の量子カーネル評価では、各データ点ペアに対して量子回路を実行してカーネル値を推定します。
# そのため、訓練データ数を $n$ とすると、訓練カーネル行列の計算には概ね $O(n^2)$ 個のカーネル評価が必要になります。
# 今回のような小規模データでは問題ありませんが、データ数を増やす場合には計算コストが急速に増える点に注意が必要です。
