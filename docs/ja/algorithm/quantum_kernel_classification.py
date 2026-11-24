# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# tags: [algorithm, machine-learning]
# ---
#
# # 量子カーネル分類
#
# このチュートリアルでは、Qamomileを用いて量子カーネル法による二値分類器を実装します。2量子ビットの特徴マップ回路でデータ点どうしの類似度を推定します。この類似度を使って、scikit-learnの`SVC`（サポートベクトルマシンを使った分類器）を学習します。
#
# ここでは、次の順序で実装を進めます。
#
# 1. 同心円状の2クラスデータ（`make_circles`）を準備する。
# 2. 古典的な前処理で、2次元の座標から4個の回路入力値を作る。
# 3. `@qkernel`で特徴マップ回路とオーバーラップ測定回路を定義する。
# 4. カーネル行列を計算し、`kernel="precomputed"`のSVCで分類する。
# 5. 生の2次元座標に対する線形SVC・RBF SVCと結果を比較する。
#
# この例は、量子カーネルの使い方を学ぶための小規模な実演です。古典的な非線形特徴を前処理で追加しているため、この分類結果だけでは量子カーネルの優位性は示せません。

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
# ## 背景
#
# ### 量子カーネル法
#
# 生のデータ点を$x$とし、前処理後の4次元の特徴ベクトルを$f(x)$と書きます。2量子ビットの量子特徴状態は
#
# $$
# |\phi(x)\rangle = U(f(x))|00\rangle
# $$
#
# です。今回使うフィデリティカーネル（オーバーラップカーネル）は
#
# $$
# k(x,x') = |\langle\phi(x')|\phi(x)\rangle|^2
# = \left|\langle00|U(f(x'))^{\dagger}U(f(x))|00\rangle\right|^2
# $$
#
# のように定義します。$U(f(x))$に続けて$U(f(x'))^{\dagger}$を作用させ、$|00\rangle$が観測される確率から$k(x,x')$を推定できます {cite:p}`10.1038/s41586-019-0980-2`。
#
# 理想的なカーネル値は$0\leq k(x,x')\leq1$で、$k(x,x)=1$です。このような理想カーネル行列は、半正定値となります。実際、$\rho(x)=|\phi(x)\rangle\langle\phi(x)|$と置けば、$k(x,x')=\operatorname{Tr}[\rho(x)\rho(x')]$のように書けます。
#
# 一方、有限ショットで各行列要素を別々に推定した行列は、統計誤差のため半正定値とは限りません。後で学習用カーネル行列に補正を加える理由はここにあります。

# %% [markdown]
# ## 問題設定
#
# ### データセット
#
# scikit-learnの`make_circles`を用いて、同心円状の2クラスデータを生成します。生の2次元座標では直線による分離が難しい例です。
#
# この例では40件のデータを生成し、学習用30件・テスト用10件に分割します。

# %%
N_SAMPLES = 4 if docs_test_mode else 40
X_raw, labels = make_circles(
    n_samples=N_SAMPLES,
    noise=0.04,
    factor=0.4,
    random_state=7,
)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw,
    labels,
    test_size=0.5 if docs_test_mode else 0.25,
    random_state=7,
    stratify=labels,
)

# %% [markdown]
# ### 古典的な前処理：特徴リフティング
#
# 今回の量子回路には4個の実数値を渡します。元の座標$x=(x_0,x_1)$に対し、まず学習データに合わせたスケーラー$s_1$を用いて
#
# $$
# a=s_1(x)=(a_0,a_1), \qquad a_i\in[0,\pi]
# $$
#
# のように変換を施します。次に2種類の非線形特徴を
#
# $$
# q(a)=\big((a_0-a_1)^2,\ (a_0+a_1)^2\big)
# $$
#
# と定義し、この2列にも別のスケーラー$s_2$を適用します。最終的な回路入力は
#
# $$
# f(x)=\big(a_0,a_1,s_2(q(a))_0,s_2(q(a))_1\big)
# $$
#
# となります。したがって、単に二乗項を追加するだけでなく、追加した2列も$[0,\pi]$に再スケーリングしています。
#
# 2つの`MinMaxScaler`は学習データだけで`fit`し、テストデータには`transform`だけを適用します。
#
# `clip=True`により学習範囲から外れる座標を端に切り詰めるため、可視化の外側領域では異なる入力が同じ回路特徴量に写る可能性があります。

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
SHOTS = 1 if docs_test_mode else 64

assert F_train.shape == (len(X_train_raw), 4)
assert F_test.shape == (len(X_test_raw), 4)

# %% [markdown]
# ## アルゴリズム
#
# ### 特徴マップ回路
#
# 2量子ビットに4個の実数特徴量を符号化します。今回の`classification_feature_map`は、各量子ビットのH・RZ・RYゲートに加え、CZ・RZZゲートなどを使ったブロックを2回繰り返す固定回路です。
#
# この回路の入力値はデータから決まるため、ここでは量子回路のパラメータを最適化しません。学習するのは分類器の`SVC`です。
#
# ### オーバーラップ測定回路
#
# 最初に$U(f(x))$を作用させ、その後で`qm.adjoint`を用いて$U(f(x'))^{\dagger}$を作用させます。測定結果が`(0, 0)`になる確率が、カーネル値です。
#
# `classification_overlap`の`left`と`right`は、それぞれ長さ4の実数ベクトルを実行時に受け取ります。特徴ブロックの繰り返し回数はコード上で2回に固定されており、`layers`という特殊化パラメータはありません。


# %% [markdown]
# ## 実装

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
        left (qm.Vector[qm.Real, 4]): First feature vector.
        right (qm.Vector[qm.Real, 4]): Second feature vector.

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
        left (qm.Param[qm.Vector[qm.Real, 4]]): First runtime feature vector.
        right (qm.Param[qm.Vector[qm.Real, 4]]): Second runtime feature vector.

    Returns:
        qm.Shot[qm.Bits[2]]: Two logical overlap-witness bits per shot.
    """

    return qm.measure(classification_overlap_state(left, right))


# %% [markdown]
# `qm.adjoint`を用いることで、共通の特徴マップ回路から逆演算を表現することができます。逆向きゲート列を手作業で定義する必要がないため、特徴マップを変更したときにも順方向と逆方向の対応を保ちやすくなります。
#
# ### 回路を特殊化して論理リソースを確認する
#
# `qm.specialize`で実行時パラメータを持つ回路をプログラムとして確定し、`qm.resources.logical`で論理ゲート数の見積もりを調べます。
#
# 現在の回路では、特徴ブロック1回分が10ゲート、これを2回繰り返す特徴マップが20ゲートです。順方向と逆方向の特徴マップを合わせたオーバーラップ回路では、測定前の論理ゲート数は40になります。リソース数は回路定義や集計の方式によって変わり得ます。

# %%
program = qm.specialize(classification_overlap)
study = qm.resources.logical(program)
logical_gate_range = study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure()
print("Logical gates (lower, upper):", logical_gate_range)

# %% [markdown]
# ## 結果
#
# ### 1回コンパイルし、異なる入力のカーネルを求める
#
# オーバーラップ回路は1回だけコンパイルし、データ点の組み合わせごとに`left`と`right`に各点の特徴ベクトルを渡して繰り返し実行します。実行先はQamomileの`Reference()`です。
#
# 学習カーネル行列については対称性$k(x_i,x_j)=k(x_j,x_i)$を用い、対角線より下の成分だけを測定し、対角線より上の成分には対応する同じ値を使います。理想的には$k(x_i,x_i)=1$なので、対角成分は1に設定します。
#
# この例では1組あたり64ショットです。`(0, 0)`が観測された比率は$1/64$刻みの推定値になり、標準誤差は最大$1/(2\sqrt{64})=0.0625$です。これは有限ショットの統計誤差であり、実機ノイズを模擬しているわけではありません。

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
next_train_seed = 1
for row in range(len(TRAIN_FEATURES)):
    train_kernel_rows[row][row] = 1.0
    for column in range(row):
        value = kernel_value(
            TRAIN_FEATURES[row],
            TRAIN_FEATURES[column],
            seed=next_train_seed,
        )
        next_train_seed += 1
        train_kernel_rows[row][column] = value
        train_kernel_rows[column][row] = value
train_kernel_raw = np.asarray(train_kernel_rows, dtype=float)
test_seed_start = next_train_seed
test_kernel = np.asarray(
    [
        [
            kernel_value(point, training, seed=test_seed_start + row * len(TRAIN_FEATURES) + column)
            for column, training in enumerate(TRAIN_FEATURES)
        ]
        for row, point in enumerate(TEST_FEATURES)
    ]
)

grid_seed_start = test_seed_start + len(TEST_FEATURES) * len(TRAIN_FEATURES)

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
print("Raw train Gram minimum eigenvalue:", float(np.linalg.eigvalsh(train_kernel_raw)[0]))
print("Maximum absolute PSD correction:", float(np.max(np.abs(train_kernel - train_kernel_raw))))
assert np.allclose(train_kernel, train_kernel.T)
assert np.allclose(np.diag(train_kernel), 1.0)
assert float(np.linalg.eigvalsh(train_kernel).min()) >= -1e-9
assert np.all(np.isfinite(train_kernel)) and np.all(np.isfinite(test_kernel))
assert test_kernel.min() >= 0.0 and test_kernel.max() <= 1.0

# %% [markdown]
# ### 有限ショットのカーネル行列と半正定値補正
#
# 理想的なフィデリティカーネル行列は半正定値ですが、非対角要素をそれぞれ64ショットで推定した`train_kernel_raw`は、負の固有値を持つことがあります。そこでこのノートブックでは、固有値を小さな正数以上に置き換えて行列を再構成し、対角を1に規格化した`train_kernel`をSVCに渡します。これは学習用行列を半正定値にするための数値的な補正であり、最も近い行列への厳密な最適化を行っているわけではありません。
#
# 補正によって各行列要素の値も変化します。補正後には、個々の値が元の測定確率と一致する保証はなく、わずかに負の値になる場合もあります。
#
# テスト用の`test_kernel`は補正前の測定値のままなので、両者が一つの厳密な量子カーネル行列に由来することまで保証する手順ではありません。この近似の影響を厳密に評価するには、ショット数の変更や別の補正方法との比較が必要です。
#
# この例では学習データが30件なので、学習カーネルの非対角成分について重複を除いた435組を評価します。テストデータ10件と学習データ30件の比較は300組です。後で決定境界を描くための10×10点の格子ではさらに3000組を評価するため、合計3735組の回路評価を行います。
#
# ### 分類器の学習
#
# ここでは、3種類のSVCを比較します。
#
# - 量子カーネルSVC：前処理を含む量子特徴マップから推定した行列を使う。
# - 線形SVC：元の2次元座標をそのまま使う。
# - RBF SVC：元の2次元座標にガウシアンカーネルを使う。
#
# 量子カーネル側には古典的な非線形特徴リフティングを適用していますが、比較用の古典SVCには適用していません。そのため、これら3つのモデルの精度差は量子計算の有無だけによるものではありません。公平な性能比較には、同じ前処理を使った古典モデルなどとの追加比較が必要です。

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

print("Test kernel shape:", test_kernel.shape)
print("First test kernel row (up to 8 values):", np.round(test_kernel[0, :8], 3))
print("True labels:", EXPECTED_LABELS)
print("Predicted labels:", predictions)

quantum_accuracy = accuracy_score(EXPECTED_LABELS, predictions)
linear_accuracy = accuracy_score(EXPECTED_LABELS, linear_predictions)
rbf_accuracy = accuracy_score(EXPECTED_LABELS, rbf_predictions)
print("Test sample count:", len(EXPECTED_LABELS))
print("Quantum kernel SVC:", quantum_accuracy)
print("Linear SVC:", linear_accuracy)
print("RBF SVC:", rbf_accuracy)
assert predictions.shape == y_test.shape
assert linear_predictions.shape == y_test.shape
assert rbf_predictions.shape == y_test.shape

# %% [markdown]
# ### 可視化
#
# ここでは、4つの図を示します。
#
# 1. 学習データの配置。
# 2. SVCに渡した、半正定値補正後の学習カーネル行列。
# 3. 量子カーネルSVCの決定境界。
# 4. RBF SVCの決定境界。
#
# 格子点は10×10点と少なく、表示される境界は粗い近似です。また、学習範囲外の格子点では`MinMaxScaler(clip=True)`による切り詰めも発生します。


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
                seed=grid_seed_start + row * len(TRAIN_FEATURES) + column,
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
image = axes[0, 1].imshow(
    train_kernel, cmap="viridis", vmin=min(0.0, float(train_kernel.min())), vmax=1.0
)
axes[0, 1].set_title("PSD-adjusted training kernel")
figure.colorbar(image, ax=axes[0, 1])
axes[1, 0].contourf(grid_x, grid_y, quantum_boundary, levels=[-0.5, 0.5, 1.5], alpha=0.35)
axes[1, 0].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 0].set_title("Quantum kernel SVC boundary")
axes[1, 1].contourf(grid_x, grid_y, rbf_boundary, levels=[-0.5, 0.5, 1.5], alpha=0.35)
axes[1, 1].scatter(X_test_raw[:, 0], X_test_raw[:, 1], c=y_test)
axes[1, 1].set_title("RBF SVC boundary")
figure.tight_layout()
plt.show()

# %% [markdown]
# ### 結果の読み方と限界
#
# 学習データは同心円状であり、生の2次元座標上の直線では分けにくい構造を持っています。量子カーネルSVCとRBF SVCは非線形の決定境界を作れます。今回のデータ分割での結果を、表示されたテスト精度で比較してください。精度はショット数や乱数シードに左右されます。
#
# 学習カーネル行列の対角成分は1です。非対角成分はデータ点間の類似度に対応しますが、今回表示するのは半正定値補正後の行列です。補正の影響もあるため、各成分を未補正の測定確率と同一視しないでください。また、学習データはクラス順に並べていないため、色のまとまりを単純にクラス間・クラス内の類似度と解釈することはできません。
#
# この例では学習データが30件、テストデータが10件です。テスト精度が0.9なら10件中9件が正解という意味です。この規模では1件の正誤で精度が0.1変動するため、単一のデータ分割の結果を一般化しないようにしましょう。
#
# 今回の比較から言えるのは、Qamomileで量子カーネルを推定し、scikit-learnのSVCに渡して二値分類を行えることです。量子優位性や古典カーネルより本質的に優れた性能を示すものではありません。性能を比較するには、データ分割・シードを変えた反復実験、ショット数への依存性、同じ古典前処理を使った対照実験などが必要です。
#
# 計算コストの面では、学習データ数を$n$とした場合、学習カーネルの構築にはおおむね$O(n^2)$組の評価が必要です。決定境界を細かく表示すると、格子点ごとに全学習データとのカーネル評価も増えます。

# %% [markdown]
# ## まとめ
#
# - 量子カーネル法では、古典データを量子状態にエンコーディングし、量子状態間のフィデリティをデータ点どうしの類似度として利用します。本チュートリアルでは、2量子ビットの特徴マップを用いてカーネル行列を構築しました。
# - Qamomileの`@qkernel`と`qm.adjoint`を利用することで、特徴マップとその逆演算を簡潔に実装することが可能です。また、コンパイル済みの量子回路を再利用し、入力パラメータを変更しながら異なるデータ点間のカーネル値を計算できます。
# - 有限ショットによる測定誤差のため、推定したカーネル行列は理論上の半正定値性を満たさない場合があります。本チュートリアルでは、学習用カーネル行列に数値的な補正を施し、古典的なサポートベクトル分類器（SVC）による分類に利用しました。
# - 量子カーネルと古典的な機械学習の組合せにより、非線形な二値分類を実装できることを確認しました。ただし、今回の結果だけで量子カーネルの優位性を示すことはできず、性能評価にはデータ分割やショット数を変えた検証、公平な古典モデルとの比較が必要です。
