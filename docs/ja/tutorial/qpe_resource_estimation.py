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
# # 量子位相推定のリソース見積もり
#
# このチュートリアルでは、量子位相推定（QPE）を例にして、Qamomileの代数的リソース見積もり機能を実演します。SymPyを使ったシンボリック式による量子リソース見積もりの方法を示します。
#
# **前提知識:** 量子回路の基礎（Hadamard、CNOT、制御演算）
#

# %%
import math

import qamomile.circuit as qmc
from qamomile.circuit.estimator import estimate_resources

# %% [markdown]
# ## セクション1: 代数的リソース見積もりの紹介
#
# ### なぜ代数的リソース見積もりなのか？
#
# 従来のリソース見積もりでは、特定の回路を実装して解析する必要がありました。Qamomileの代数的アプローチでは以下が可能になります：
#
# 1. **シンボリックなリソース見積もり** - パラメータ値を固定せずに見積もり
# 2. **設計空間の探索** - パラメータを代数的に変化させて探索
# 3. **アルゴリズムの比較** - 理論的複雑度の式を使った比較
# 4. **ハードウェア要件の計画** - パラメトリック解析による計画
#
# ### Qamomileでの2つのアプローチ
#
# 1. **回路ベース**: `@qkernel`関数を`estimate_resources()`に渡す - 実際の回路IRを解析
# 2. **アルゴリズム的**: `estimate_qpe()`のような理論的な式を使用 - 研究論文からの複雑度解析に基づく
#
# まず、APIを理解するために簡単な例から始めましょう。

# %% [markdown]
# ### ウォームアップ: ベル状態の見積もり
#
# まず、`estimate_resources()`の動作を理解するためにベル状態のリソースを見積もってみましょう。

# %%
@qmc.qkernel
def bell_state() -> qmc.Vector[qmc.Qubit]:
    """ベル状態 |Φ+⟩ = (|00⟩ + |11⟩)/√2 を作成"""
    q = qmc.qubit_array(2, name="q")
    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return q


bell_state.draw()

# %%
# qkernelのblockをestimate_resourcesに渡すだけ
est = estimate_resources(bell_state.block)

print("ベル状態のリソース見積もり:")
print(f"  量子ビット数: {est.qubits}")
print(f"  総ゲート数: {est.gates.total}")
print(f"  単一量子ビットゲート: {est.gates.single_qubit}")
print(f"  2量子ビットゲート: {est.gates.two_qubit}")
print(f"  Cliffordゲート: {est.gates.clifford_gates}")

# %% [markdown]
# 期待通り: 2量子ビット、2ゲート（H + CX）、両方ともClifford、深さ2。
#
# ### シンボリック変数を使ったパラメトリック見積もり
#
# 次に、可変サイズのGHZ状態というパラメトリック回路を試してみましょう。

# %%
@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """n量子ビットのGHZ状態 |0...0⟩ + |1...1⟩ を作成"""
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i+1] = qmc.cx(q[i], q[i+1])
    return q


ghz_state.draw(n=4, fold_loops=False)

# %%
# シンボリックパラメータnで見積もり
est_ghz = estimate_resources(ghz_state.block)

print("\nGHZ状態のリソース見積もり（シンボリック）:")
print(f"  量子ビット数: {est_ghz.qubits}")
print(f"  総ゲート数: {est_ghz.gates.total}")
print(f"  2量子ビットゲート: {est_ghz.gates.two_qubit}")

# %% [markdown]
# 結果にシンボル`n`が含まれていることに注目してください！これが代数的見積もりの力です。
#
# 具体的な値を代入して特定の見積もりを得ることができます：

# %%
# n=10を代入
est_ghz_10 = est_ghz.substitute(n=10)

print("\nn=10のGHZ状態:")
print(f"  量子ビット数: {est_ghz_10.qubits}")
print(f"  総ゲート数: {est_ghz_10.gates.total}")
print(f"  2量子ビットゲート: {est_ghz_10.gates.two_qubit}")

# n=100を代入
est_ghz_100 = est_ghz.substitute(n=100)

print("\nn=100のGHZ状態:")
print(f"  量子ビット数: {est_ghz_100.qubits}")
print(f"  総ゲート数: {est_ghz_100.gates.total}")
print(f"  2量子ビットゲート: {est_ghz_100.gates.two_qubit}")

# %% [markdown]
# それでは、より複雑なアルゴリズムである量子位相推定に適用してみましょう。

# %% [markdown]
# ## セクション2: 基本コンポーネントからのQPE実装
#
# 量子位相推定（QPE）は、ユニタリ演算子の位相（固有値）を推定します。固有状態$|\psi\rangle$を持つユニタリ$U$が次のようになるとき：
#
# $$U|\psi\rangle = e^{2\pi i \theta}|\psi\rangle$$
#
# QPEは位相$\theta$を$m$ビットの精度で推定します。
#
# ### アルゴリズムの概要
#
# 1. $m$個のカウント量子ビットを重ね合わせ状態に準備
# 2. 制御された$U^{2^k}$演算を適用
# 3. 逆量子フーリエ変換（IQFT）を適用
# 4. 測定して位相推定値を取得
#
# 各コンポーネントをゼロから実装してみましょう。

# %% [markdown]
# ### ステップ1: 逆量子フーリエ変換（IQFT）
#
# IQFTはQPEの最終ステップで、位相情報を測定可能な基底状態に変換します。

# %%
@qmc.qkernel
def iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """逆量子フーリエ変換"""
    n = qubits.shape[0]
    # 量子ビットの交換（順序を逆にする）
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    # 逆QFTゲートを適用
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], theta=angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits


iqft.draw(qubits=4, fold_loops=False)

# %%
# シンボリックnでIQFTのリソースを見積もる
@qmc.qkernel
def iqft_n(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """n量子ビットでのIQFT"""
    qubits = qmc.qubit_array(n, name="q")
    return iqft(qubits)

est_iqft = estimate_resources(iqft_n.block)
print("IQFTのリソース見積もり（シンボリックn）:")
print(f"  量子ビット数: {est_iqft.qubits}")
print(f"  総ゲート数: {est_iqft.gates.total}")
print(f"  2量子ビットゲート: {est_iqft.gates.two_qubit}")

# %% [markdown]
# IQFTはネストされたループのため$O(n^2)$のゲートが必要です。

# %% [markdown]
# ### ステップ2: 対象ユニタリの定義
#
# このチュートリアルでは、対象ユニタリとして単純な位相ゲートを使用します：
#
# $$P(\theta)|1\rangle = e^{i\theta}|1\rangle$$
#
# これは固有状態$|1\rangle$に対して固有値$e^{i\theta}$を持ちます。

# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: float, iter: int) -> qmc.Qubit:
    """位相ゲート: P(θ)|1⟩ = e^{iθ}|1⟩"""
    for i in qmc.range(iter):
        q = qmc.p(q, theta)
    return q


phase_gate.draw(iter=4, fold_loops=False)

# %% [markdown]
# ### ステップ3: QPEをゼロから実装
#
# 基本ゲートを使用して完全なQPEアルゴリズムを実装します。

# %%
@qmc.qkernel
def qpe_manual(theta: float, m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """mビット精度のQPE、基本ゲートから実装"""
    # 量子ビットを割り当て
    counting = qmc.qubit_array(m, name="counting")
    target = qmc.qubit(name="target")

    # 対象を固有状態|1⟩に準備
    target = qmc.x(target)

    # ステップ1: カウント量子ビットを重ね合わせ状態に準備
    for i in qmc.range(m):
        counting[i] = qmc.h(counting[i])

    # ステップ2: 制御されたU^(2^k)演算を適用
    # 位相ゲートの場合、U^kは単に位相をk倍する
    controlled_phase = qmc.controlled(phase_gate)
    for i in qmc.range(m):
        # U^(2^i) = P(2^i * theta)を適用
        iterations = 2 ** i
        counting[i], target = controlled_phase(counting[i], target, theta=theta, iter=iterations)

    # ステップ3: IQFTを適用
    counting = iqft(counting)

    # ステップ4: 測定
    bits = qmc.measure(counting)
    return bits


qpe_manual.draw(theta=math.pi / 2, m=3, fold_loops=False, inline=True)

# %%
est_qpe_manual = estimate_resources(qpe_manual.block)
print("\n手動QPEのリソース見積もり（シンボリックm）:")
print(f"  量子ビット数: {est_qpe_manual.qubits}")
print(f"  総ゲート数: {est_qpe_manual.gates.total}")
print(f"  2量子ビットゲート: {est_qpe_manual.gates.two_qubit}")


# %% [markdown]
# ### 観察結果
#
# - **量子ビット数**: $m + 1$ (m個のカウント量子ビット + 1個の対象量子ビット) - 線形にスケール
# - **ゲート数**: $2^m$の制御演算により精度とともに急速に増加
# - **深さ**: 現行APIの`estimate_resources()`では返されません

# %% [markdown]
# ## セクション3: Qamomileの組み込みQPEを使用
#
# Qamomileは便利な組み込みの`qmc.qpe()`関数を提供しています。手動実装と比較してみましょう。

# %%

@qmc.qkernel
def simple_phase_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    """qmc.qpe()用のシンプルな位相ゲート: P(θ)|1⟩ = e^{iθ}|1⟩

    注意: このバージョンは位相ゲートを1回だけ適用します。
    qmc.qpe()と一緒に使用すると、繰り返し（2^k回）はpowerパラメータで処理されます。
    """
    return qmc.p(q, theta)

@qmc.qkernel
def qpe_builtin(theta: float, n: qmc.UInt) -> qmc.Float:
    """Qamomileの組み込みqpe関数を使用したQPE（8ビット精度）"""
    counting = qmc.qubit_array(n, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # 固有状態を準備

    # qmc.qpe()は制御演算とIQFTを内部で処理
    phase = qmc.qpe(target, counting, simple_phase_gate, theta=theta)
    return qmc.measure(phase)


qpe_builtin.draw(theta=math.pi / 2, n=3, fold_loops=False, inline=True)

# %%
est_builtin = estimate_resources(qpe_builtin.block)
est_builtin = est_builtin.simplify()

print("\n組み込みQPE（m=8）のリソース見積もり:")
print(f"  量子ビット数: {est_builtin.qubits}")
print(f"  総ゲート数: {est_builtin.gates.total}")
print(f"  2量子ビットゲート: {est_builtin.gates.two_qubit}")

# %%
