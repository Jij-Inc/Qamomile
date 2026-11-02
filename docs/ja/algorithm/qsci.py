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
# tags: [algorithm, sample-based]
# ---
#
# ---
# tags: [algorithm, sample-based]
# ---
#
# # Quantum Selected Configuration Interaction (QSCI)
#
# **QSCI**は量子状態からサンプリングしたビット列を使って小さな有効ハミルトニアンを構築し、それを古典コンピュータで厳密に対角化するハイブリッド量子・古典アルゴリズムです。素のVQEと比べた利点は、結果が厳密な**変分原理の保証**を引き継ぐ点にあります。ノイズのあるハードウェア上でも、
#
# $$
# E_{\mathrm{QSCI}} \;\geq\; E_{\mathrm{exact}}.
# $$
#
# Kanno *et al.*{cite:p}`10.48550/arXiv.2302.11320`が導入した4ステップのレシピは次のとおりです:
#
# 1. 量子コンピュータ上で入力状態$|\psi_{\mathrm{in}}\rangle$を準備する（典型的には大まかに最適化されたVQE状態）。
# 2. $|\psi_{\mathrm{in}}\rangle$を計算基底で多数回測定する。
# 3. 出現頻度が高い上位$K$個のビット列を離散部分空間$\{|s_i\rangle\}_{i=1}^{K}$として選ぶ。
# 4. 有効ハミルトニアン$H^{\mathrm{sub}}_{ij} = \langle s_i | H | s_j \rangle$を構築し、古典的に対角化する。
#
# このチュートリアルでは4量子ビットの横磁場Isingモデルでフローを通しで歩きます。量子状態の準備とサンプリングは**QURI Parts**連携（Qulacsシミュレータ）で実行し、部分空間の構築と対角化は`qamomile.linalg.solve_subspace`を使います。内部ではベクトル化されたZ基底ルーチン`subspace_hamiltonian`（XOR・パリティ計算のみで行列積不要）を呼びます。

# %% [markdown]
# ## インストール
#
# ```bash
# pip install qamomile
# ```

# %%
# Install the latest Qamomile through pip!
# # !pip install qamomile

# %%
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import qamomile as qm
from qamomile.measurement import Estimator
from qamomile.quri_parts import Qulacs, QuriParts

# %% [markdown]
# ## 小さなテスト用ハミルトニアン
#
# オープンチェーン上の4量子ビット横磁場Isingモデルを使います:
#
# $$
# H \;=\; -J \sum_{i=0}^{n-2} Z_i Z_{i+1} \;-\; h \sum_{i=0}^{n-1} X_i,
# \quad J = 1,\; h = 0.7.
# $$
#
# 16次元のヒルベルト空間は十分小さいので、NumPyで厳密な基底状態エネルギーを直接計算してQSCIのリファレンスとして使えます。

# %%
N_QUBITS = 4
FIELD = 0.7
HAMILTONIAN = qm.Hamiltonian.zero(num_qubits=N_QUBITS)
for index in range(N_QUBITS - 1):
    HAMILTONIAN = HAMILTONIAN - qm.Z(index) * qm.Z(index + 1)
for index in range(N_QUBITS):
    HAMILTONIAN = HAMILTONIAN - FIELD * qm.X(index)

ALL_CONFIGURATIONS = tuple(itertools.product((0, 1), repeat=N_QUBITS))
exact_matrix = qm.linalg.subspace_hamiltonian(ALL_CONFIGURATIONS, HAMILTONIAN)
EXACT_ENERGY = float(np.linalg.eigvalsh(exact_matrix)[0])

assert HAMILTONIAN.num_qubits == N_QUBITS
assert HAMILTONIAN.is_hermitian
assert exact_matrix.shape == (2**N_QUBITS, 2**N_QUBITS)
print(f"Exact ground-state energy: {EXACT_ENERGY:+.6f}")

# %% [markdown]
# ## Hardware-efficient ansatz
#
# 単純なalternating-layer ansatzとして、全量子bitへのRYとlinear CX chainを2回
# 適用し、最後にもう1つRY layerを置きます。3 layerで合計12個のruntime parameterを
# 使います。VQE用にstateを返すprogram、QSCI sampling用にmeasurement programを
# 定義します。


# %%
@qm.qkernel
def qsci_state(
    thetas: qm.Param[qm.Vector[qm.Real, 12]],  # type: ignore[valid-type]
) -> qm.Q[qm.Bits[4]]:  # type: ignore[valid-type]
    """Prepare one runtime-rebindable four-qubit QSCI trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 12]]): Runtime RY angles.

    Returns:
        qm.Q[qm.Bits[4]]: Prepared trial state.
    """

    state = qm.qubits(4)
    for repetition in range(2):
        for target in range(4):
            state = qm.ry(state, thetas[repetition * 4 + target], target=target)
        for control in range(3):
            state = qm.cx(state, control=control, target=control + 1)
    for target in range(4):
        state = qm.ry(state, thetas[8 + target], target=target)
    return state


@qm.qkernel
def qsci_samples(
    thetas: qm.Param[qm.Vector[qm.Real, 12]],  # type: ignore[valid-type]
) -> qm.Shot[qm.Bits[4]]:  # type: ignore[valid-type]
    """Sample one runtime-rebindable QSCI trial state.

    Args:
        thetas (qm.Param[qm.Vector[qm.Real, 12]]): Runtime RY angles.

    Returns:
        qm.Shot[qm.Bits[4]]: Computational-basis configuration per shot.
    """

    return qm.measure(qsci_state(thetas))


# %% [markdown]
# ## 両方のprogramをQURI Partsでcompileする
#
# specializeしたprogramには通常のcallがあり、QSCI固有のoperationはありません。
# Rustは見えているbodyから12個のrotationと6個のCX gateを厳密に導出します。
# QURI Parts artifactのruntime ABIには`thetas`だけが残ります。

# %%
program = qm.specialize(qsci_samples)
inspection = program.inspect()
study = qm.resources.logical(program)

assert "qcore.call" in inspection
assert "quantum.qsci" not in inspection
assert study.metric(qm.resources.LOGICAL_GATES).numeric_enclosure() == (18, 18)
assert qm.resources.ResourceStudy.from_bytes(study.to_bytes()) == study

sample_endpoint = Qulacs()
artifact = qm.compile(program, to=QuriParts())
assert tuple(slot.name for slot in artifact.parameter_abi.slots) == ("thetas",)
assert artifact.format == "quri-parts.logical-circuit.v2"

# %% [markdown]
# ## ステップ1: 短いVQEで$|\psi_{\mathrm{in}}\rangle$を準備
#
# QSCIは入力状態の最適化が甘くても頑健で、ランダムパラメータでも意味のある部分空間が得られますが、短いVQEを走らせるとサンプリング分布が真の基底状態を支配するビット列に集中するため、同じ$K$でも部分空間の情報量が大きくなります。ここでは数回のCOBYLA反復だけ走らせます。

# %%
docs_test_mode = os.environ.get("QAMOMILE_DOCS_TEST") == "1"
expectation_endpoint = qm.Reference()
vqe_shots = 4 if docs_test_mode else 128
energy_estimator = Estimator(qsci_state, HAMILTONIAN, grouping="qwc")
cost_history = []


def cost_function(parameters):
    """Estimate the trial-state energy at one variational point."""

    estimate = energy_estimator.estimate(
        on=expectation_endpoint,
        parameters={"thetas": tuple(float(value) for value in parameters)},
        shots=vqe_shots,
        seed=10 + len(cost_history),
    )
    cost_history.append(estimate.value)
    return estimate.value


rng = np.random.default_rng(0)
initial_parameters = rng.uniform(0.0, 2.0 * np.pi, 12)
max_iterations = 14 if docs_test_mode else 80
optimization = minimize(
    cost_function,
    initial_parameters,
    method="COBYLA",
    options={"maxiter": max_iterations, "rhobeg": 0.5},
)
optimized_parameters = tuple(float(value) for value in optimization.x)

print(
    f"VQE energy = {optimization.fun:+.6f} "
    f"(gap to E_exact: {optimization.fun - EXACT_ENERGY:+.4e})"
)

assert np.isfinite(optimization.fun)
assert len(optimized_parameters) == 12
assert len(cost_history) == optimization.nfev

# %% [markdown]
# 有限ショットのエネルギー推定値は、サンプリングノイズによって厳密な基底エネルギーを下回ることがあります。変分原理の下限は真の期待値に対して成立し、個々の推定値や最適化で選んだ最小値に対する保証ではありません。後で行う部分空間の厳密対角化では、誤差の原因が異なります。

# %% [markdown]
# ## ステップ2: Z基底でビット列をサンプリング
#
# 各サンプルはタプル`(b_0, ..., b_{n-1})`で、$q$番目の要素は量子ビット$q$のZ固有値インデックスです。これは`subspace_hamiltonian`が期待する形式そのものです。

# %%
SHOTS = 128 if docs_test_mode else 4_000

record = qm.run(
    artifact,
    on=sample_endpoint,
    parameters={"thetas": optimized_parameters},
    shots=SHOTS,
    seed=2,
)
counts = record.result.counts
configurations = tuple(
    bits
    for bits, _ in sorted(
        counts.items(),
        key=lambda item: item[1],
        reverse=True,
    )
)

print("sample counts:", counts)

assert record.artifact is artifact
assert sum(counts.values()) == SHOTS
assert len(configurations) <= 2**N_QUBITS
assert all(len(bits) == N_QUBITS for bits in configurations)

# %% [markdown]
# ## ステップ3 + 4: QSCI部分空間の構築と対角化
#
# 各部分空間サイズ$K$について、出現頻度上位$K$個のビット列を`solve_subspace`に渡します。`solve_subspace`はベクトル化されたXOR・パリティルーチンで$H^{\mathrm{sub}}_{ij} = \langle s_i|H|s_j\rangle$を構築し、`numpy.linalg.eigh`を実行します。返ってくる最小固有値がQSCIエネルギーの推定値で、変分原理によって任意の$K$に対し$E_{\mathrm{QSCI}}(K) \geq E_{\mathrm{exact}}$が保証されます。

# %%
SUBSPACE_SIZES = tuple(
    size
    for size in (1, 2, 4, 8, 16, len(configurations))
    if size <= len(configurations)
)
energies = tuple(
    float(qm.linalg.solve_subspace(configurations[:size], HAMILTONIAN)[0][0])
    for size in SUBSPACE_SIZES
)
for size, energy in zip(SUBSPACE_SIZES, energies):
    print(f"K={size}: E_QSCI={energy:+.12f}")
print(f"E_exact={EXACT_ENERGY:+.12f}")

assert all(left >= right - 1e-12 for left, right in zip(energies, energies[1:]))
assert all(energy >= EXACT_ENERGY - 1e-12 for energy in energies)

# %%
plt.figure(figsize=(6, 4))
plt.plot(SUBSPACE_SIZES, energies, "-o", label=r"$E_{\mathrm{QSCI}}$")
plt.axhline(EXACT_ENERGY, color="black", linestyle="--", label=r"$E_{\mathrm{exact}}$")
plt.xlabel("Subspace size $K$")
plt.ylabel("Energy")
plt.title("QSCI convergence — 4-qubit TFIM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## まとめ
#
# - `Q[Bits[4]]`、`Param[Vector[Real, 12]]`、`Shot[Bits[4]]`により、ownership、
#   stage、result widthが明示されます。
# - 1つのsourceをinspect、厳密なlogical resource、compile、samplingに使います。QSCI固有のcompiler operationやbackend分岐は追加しません。
# - host eigensolveの前に、同じimmutable Hamiltonianをsampled stateへRustで射影します。
# - continuous COBYLA VQEでstateを選び、QURI Parts/Qulacs samplingから最大16状態の
#   完全Hilbert空間以下のsubspaceを構築します。

# %% [markdown]
# ## 補足
#
# - `solve_subspace`内部で使われるZ基底の高速経路（`subspace_hamiltonian`）は行列積を必要としません。各Pauli項は1つのXORマスクとパリティ符号として寄与し、$K^2$個のサンプルペア全体にわたってベクトル化されます。
# - 重複したサンプルビット列は上記のunique-bitstringリストから除外されます。結果として得られる部分空間は条件数が良く、`solve_subspace`は通常のHermitian固有値分解を返します。
