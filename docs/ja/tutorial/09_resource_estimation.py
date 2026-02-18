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
# # 代数的リソース推定
# **重要な注意事項**:
# リソース推定機能は現在バグ修正中であり、一部の回路で正しくない結果を返す可能性があります。
# このチュートリアルは近い将来大幅に変更される可能性があります。
#
# アルゴリズムに必要な量子ビット数は？ゲート数は？問題サイズに対してコストはどうスケールする？
# これらの疑問に答えるのが**リソース推定**です。Qamomileでは、具体的なパラメータ値を
# 決める前に、*シンボリック*にリソースを推定できます。
#
# ## このチュートリアルで学ぶこと
# - `estimate_resources()` を使って量子ビット数、ゲート数、回路深さを SymPy 式として取得する方法
# - `.to_dict()` で利用可能な全メトリクスを反復処理する方法
# - シンボリック（パラメトリック）回路での `.substitute()` と `.simplify()` の使い方
# - 量子位相推定（QPE）のリソースプロファイル分析
# - 複合ゲートの分解戦略の比較
# - 戦略付きカスタム複合ゲートの定義と、ブラックボックスリソース推定のためのスタブゲートの使い方
#
# **前提知識:** チュートリアル 01--03（基本的な回路構成）。
# QPE の知識があると役立ちますが、必須ではありません。

# %%
import math

import qamomile.circuit as qmc
from qamomile.circuit.estimator import estimate_resources

# %% [markdown]
# ## 1. 代数的リソース推定とは
#
# 従来のリソース推定は*固定された*回路に対して行われます。特定のサイズの回路を
# 構築し、ゲートを数えます。小規模な実験には十分ですが、アルゴリズムが
# どのように*スケール*するかはわかりません。
#
# Qamomile の代数的推定器は異なるアプローチを取ります：
#
# 1. シンボリックなサイズ（例: `n: qmc.UInt`）をパラメータに含む
#    `@qkernel` を記述します。
# 2. `estimate_resources(kernel.block)` を呼び出します。
# 3. 推定器が回路 IR をたどり（ループ、関数呼び出し、制御操作を含む）、
#    量子ビット数、ゲート数、回路深さをシンボリックパラメータを含む
#    **SymPy 式**として返します。
#
# これにより、サイズごとに個別の回路を構築することなく、
# 設計空間全体を探索できます。

# %% [markdown]
# ## 2. リソースメトリクスのリファレンス
#
# `estimate_resources()` が返すオブジェクトは `ResourceEstimate` であり、
# 以下のフィールドを持ちます：
#
# | 属性                         | 説明                                      |
# |-----------------------------|-------------------------------------------|
# | `est.qubits`               | 論理量子ビット数                            |
# | `est.gates.total`          | 総ゲート数                                  |
# | `est.gates.single_qubit`   | 単一量子ビットゲート（H, X, RZ, P, ...）    |
# | `est.gates.two_qubit`      | 2量子ビットゲート（CX, CZ, CP, SWAP, ...）  |
# | `est.gates.t_gates`        | T ゲートと Tdg ゲート（フォールトトレランスに重要） |
# | `est.gates.clifford_gates` | Clifford ゲート（H, S, CX, CZ, SWAP, ...） |
# | `est.depth.total_depth`    | 回路深さ（逐次実行の上界）                    |
# | `est.depth.t_depth`        | T 深さ                                     |
# | `est.depth.two_qubit_depth`| 2量子ビットゲート深さ                        |
# | `est.parameters`           | シンボル名から SymPy シンボルへの辞書         |
#
# すべての値は **SymPy 式** です。3つの便利なメソッドが用意されています：
#
# - **`est.simplify()`** -- すべての式に `sympy.simplify()` を適用した
#   新しい `ResourceEstimate` を返します。
# - **`est.substitute(**kwargs)`** -- 指定したシンボルを具体的な値に
#   置換した新しい `ResourceEstimate` を返します。
# - **`est.to_dict()`** -- すべてのフィールドを文字列として含むネスト辞書を
#   返します。シリアライズや反復処理に適しています。
#
# フィールドセットは将来のリリースで拡張される可能性があります（例: 回転ゲート数）。
# 前方互換性のあるコードを書くには、フィールド名をハードコードせず
# `to_dict()` で反復処理してください。
#
# ### Qamomile レベル vs バックエンドレベルの推定
#
# **重要:** これらの推定値は **Qamomile IR レベル** で定義されたゲートを
# 反映しています。バックエンド（Qiskit、CUDA-Q など）にトランスパイルすると、
# 以下の理由でゲート数が変わる可能性があります：
#
# - バックエンド固有の分解（例: SWAP $\rightarrow$ 3 CX）
# - ゲート数や深さを削減する最適化パス
# - ネイティブゲートセットへの変換
#
# これらの推定値は、**アルゴリズムレベルの分析**や**設計比較**に使用してください。
# 正確なハードウェアコストとしては使用しないでください。

# %% [markdown]
# ## 3. 簡単な例
#
# ### 3.1 Bell 状態 -- 具体的なリソースカウント
#
# 最も単純なケースから始めましょう。シンボリックパラメータを持たない
# Bell 状態回路です。推定器はプレーンな整数を返すはずです。


# %%
@qmc.qkernel
def bell_state() -> qmc.Vector[qmc.Qubit]:
    """Create a Bell state |Phi+> = (|00> + |11>) / sqrt(2)."""
    q = qmc.qubit_array(2, name="q")
    q[0] = qmc.h(q[0])
    q[0], q[1] = qmc.cx(q[0], q[1])
    return q


bell_state.draw()

# %%
est = estimate_resources(bell_state.block)

# Iterate over all fields using to_dict()
data = est.to_dict()
print("Bell State Resource Estimate:")
for section, values in data.items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# `to_dict()` を使うと、すべてのメトリクスを一目で確認できます。
# また前方互換性があり、将来のリリースで新しいフィールドが追加された場合も
# 自動的に表示されます。
#
# フィールドには直接アクセスすることもできます：
# ```python
# est.qubits          # 2
# est.gates.total      # 2
# est.gates.two_qubit  # 1
# ```

# %% [markdown]
# ### 3.2 GHZ 状態 -- パラメトリック推定
#
# 次に、サイズ `n` をシンボリックに残した GHZ 状態を考えます。


# %%
@qmc.qkernel
def ghz_state(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """Create an n-qubit GHZ state (|0...0> + |1...1>) / sqrt(2)."""
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
    return q


ghz_state.draw(n=4, fold_loops=False)

# %%
est_ghz = estimate_resources(ghz_state.block)

print("GHZ State Resource Estimate (symbolic):")
print(f"  Qubits:            {est_ghz.qubits}")
print(f"  Total gates:       {est_ghz.gates.total}")
print(f"  Single-qubit gates:{est_ghz.gates.single_qubit}")
print(f"  Two-qubit gates:   {est_ghz.gates.two_qubit}")
print(f"  T gates:           {est_ghz.gates.t_gates}")
print(f"  Clifford gates:    {est_ghz.gates.clifford_gates}")
print(f"  Total depth:       {est_ghz.depth.total_depth}")
print(f"  T-depth:           {est_ghz.depth.t_depth}")
print(f"  Two-qubit depth:   {est_ghz.depth.two_qubit_depth}")
print(f"  Parameters:        {est_ghz.parameters}")

# %% [markdown]
# 結果にはシンボル **n** が含まれています。これが代数的推定の核心です：
# リソースが問題サイズに対してどのように増加するかを表す閉じた形の式が得られます。
#
# ### 3.3 具体的な値の代入
#
# `.substitute()` を使って、シンボリックパラメータに具体的な値を代入できます。

# %%
for size in [10, 50, 100]:
    concrete = est_ghz.substitute(n=size)
    print(
        f"  n={size:>3}:  qubits={concrete.qubits}, "
        f"gates={concrete.gates.total}, "
        f"two_qubit={concrete.gates.two_qubit}"
    )

# %% [markdown]
# `to_dict()` を使って推定結果を JSON としてシリアライズすることもできます：

# %%
import json

data = est_ghz.to_dict()
print(json.dumps(data, indent=2))

# %% [markdown]
# ## 4. QPE のリソース分析
#
# 量子位相推定（QPE）は、量子コンピューティングにおいて最も重要な
# サブルーチンの一つです。ユニタリ演算子の位相 $\theta$ を推定します：
#
# $$U|\psi\rangle = e^{2\pi i\theta}|\psi\rangle$$
#
# QPE は4つのステージで構成されます：
#
# 1. $m$ 個のカウンティング量子ビットを重ね合わせ状態に準備
# 2. 制御 $U^{2^k}$ 操作を適用
# 3. 逆量子フーリエ変換（IQFT）を適用
# 4. カウンティングレジスタを測定
#
# 各コンポーネントをゼロから実装し、リソースプロファイルを分析しましょう。
#
# ### 4.1 逆 QFT（IQFT）
#
# IQFT は QPE の最終ステップです。カウンティングレジスタに
# エンコードされた位相情報を、測定可能な基底状態に変換します。


# %%
@qmc.qkernel
def iqft(qubits: qmc.Vector[qmc.Qubit]) -> qmc.Vector[qmc.Qubit]:
    """Inverse Quantum Fourier Transform."""
    n = qubits.shape[0]
    # Swap qubits (reverse order)
    for j in qmc.range(n // 2):
        qubits[j], qubits[n - j - 1] = qmc.swap(qubits[j], qubits[n - j - 1])
    # Apply inverse QFT gates
    for j in qmc.range(n):
        for k in qmc.range(j):
            angle = -math.pi / (2 ** (j - k))
            qubits[j], qubits[k] = qmc.cp(qubits[j], qubits[k], theta=angle)
        qubits[j] = qmc.h(qubits[j])
    return qubits


iqft.draw(qubits=4, fold_loops=False)

# %% [markdown]
# IQFT を独自の量子ビットを確保するカーネルでラップすることで、
# 推定器が量子ビット数を追跡し、すべてをシンボリックサイズ `n` で
# 表現できるようにします。


# %%
@qmc.qkernel
def iqft_n(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """IQFT on n freshly-allocated qubits."""
    qubits = qmc.qubit_array(n, name="q")
    return iqft(qubits)


est_iqft = estimate_resources(iqft_n.block)

print("IQFT Resource Estimate (symbolic n):")
for section, values in est_iqft.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# IQFT はネストしたループにより $O(n^2)$ 個のゲートを必要とします：
# 外側のループが $n$ 回、内側のループが最大 $j$ 回実行されるため、
# $\sum_{j=0}^{n-1} j = n(n-1)/2$ 個の制御位相ゲートに加え、
# $n$ 個のアダマールゲートと $\lfloor n/2 \rfloor$ 個の SWAP ゲートが必要です。
#
# いくつかの具体的な値を代入して確認してみましょう：

# %%
for size in [4, 8, 16]:
    concrete = est_iqft.substitute(n=size)
    print(
        f"  n={size:>2}:  qubits={concrete.qubits}, "
        f"gates={concrete.gates.total}, depth={concrete.depth.total_depth}"
    )

# %% [markdown]
# ### 4.2 ターゲットユニタリ
#
# このチュートリアルでは、ターゲットユニタリとして単純な位相ゲート
# $P(\theta)$ を使用します。その固有状態は $|1\rangle$ で、
# 固有値は $e^{i\theta}$ です。


# %%
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float, iter: qmc.UInt) -> qmc.Qubit:
    """Apply P(theta) a total of `iter` times."""
    for i in qmc.range(iter):
        q = qmc.p(q, theta)
    return q


phase_gate.draw(iter=4, fold_loops=False)

# %% [markdown]
# ### 4.3 完全な手動 QPE
#
# 上で定義したビルディングブロックを使って、完全な QPE アルゴリズムを
# 組み立てます。


# %%
@qmc.qkernel
def qpe_manual(theta: qmc.Float, m: qmc.UInt) -> qmc.Vector[qmc.Bit]:
    """QPE with m-bit precision, implemented from basic gates."""
    # Allocate qubits
    counting = qmc.qubit_array(m, name="counting")
    target = qmc.qubit(name="target")

    # Prepare target in eigenstate |1>
    target = qmc.x(target)

    # Step 1: Put counting qubits in superposition
    for i in qmc.range(m):
        counting[i] = qmc.h(counting[i])

    # Step 2: Apply controlled-U^(2^k) operations
    controlled_phase = qmc.controlled(phase_gate)
    for i in qmc.range(m):
        iterations = 2**i
        counting[i], target = controlled_phase(
            counting[i], target, theta=theta, iter=iterations
        )

    # Step 3: Apply IQFT
    counting = iqft(counting)

    # Step 4: Measure
    bits = qmc.measure(counting)
    return bits


qpe_manual.draw(theta=math.pi / 2, m=3, fold_loops=False, inline=True)

# %% [markdown]
# ### 4.4 簡約化前と簡約化後の式
#
# ネストしたループや関数呼び出しを含む回路では、生の SymPy 式が
# 複雑になることがあります。`.simplify()` メソッドは推定結果の
# すべてのフィールドに対して SymPy の簡約化エンジンを実行します。
# その違いを見てみましょう：

# %%
est_qpe_raw = estimate_resources(qpe_manual.block)
est_qpe_simplified = est_qpe_raw.simplify()

print("Manual QPE -- Before simplify():")
print(f"  Total gates: {est_qpe_raw.gates.total}")
print(f"  Two-qubit:   {est_qpe_raw.gates.two_qubit}")
print(f"  Depth:       {est_qpe_raw.depth.total_depth}")
print()
print("Manual QPE -- After simplify():")
print(f"  Total gates: {est_qpe_simplified.gates.total}")
print(f"  Two-qubit:   {est_qpe_simplified.gates.two_qubit}")
print(f"  Depth:       {est_qpe_simplified.depth.total_depth}")

# %% [markdown]
# **考察：**
#
# - **量子ビット数**: $m + 1$ -- $m$ 個のカウンティング量子ビットと1個のターゲット量子ビット。
#   線形にスケールします。
# - **ゲート数**: $2^m$ 個の制御位相操作により、精度に対して急速に増加します。
# - **深さ**: 制御ユニタリが逐次的に適用されるため、同様に大きく増加します。
#
# いくつかの値を代入して具体的な数値を確認しましょう：

# %%
for precision in [4, 8, 12]:
    concrete = est_qpe_raw.substitute(m=precision).simplify()
    print(
        f"  m={precision:>2}:  qubits={concrete.qubits}, "
        f"gates={concrete.gates.total}, depth={concrete.depth.total_depth}"
    )

# %% [markdown]
# ### 4.5 組み込み QPE
#
# Qamomile には、制御操作と IQFT を内部で処理する組み込みの `qmc.qpe()`
# プリミティブが用意されています。手動実装とリソースプロファイルを
# 比較してみましょう。


# %%
@qmc.qkernel
def simple_phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Single application of the phase gate.

    When used with qmc.qpe(), the repetitions (2^k) are handled
    internally by the power parameter.
    """
    return qmc.p(q, theta)


@qmc.qkernel
def qpe_builtin(theta: qmc.Float, n: qmc.UInt) -> qmc.Float:
    """QPE using Qamomile's built-in qpe function."""
    counting = qmc.qubit_array(n, name="counting")
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # Prepare eigenstate

    # qmc.qpe() handles controlled operations and IQFT internally
    phase = qmc.qpe(target, counting, simple_phase_gate, theta=theta)
    return qmc.measure(phase)


qpe_builtin.draw(theta=math.pi / 2, n=3, fold_loops=False, inline=True)

# %%
est_builtin = estimate_resources(qpe_builtin.block)
est_builtin = est_builtin.simplify()

print("Built-in QPE Resource Estimate (symbolic n):")
for section, values in est_builtin.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# 組み込み QPE は手動実装と同じ漸近的スケーリングを示します。
# 組み込み版の主な利点は利便性と、バックエンドが最適化された複合ゲート
# 実装に置き換えられる点です。

# %% [markdown]
# ## 5. QFT 戦略の比較
#
# Qamomile の `QFT` 複合ゲートは、複数の分解**戦略**をサポートしています。
# 各戦略は、ゲート数（コスト）と近似誤差の間の異なるトレードオフを提供します。
#
# 標準 QFT は $O(n^2)$ 個のゲートを使用しますが、近似 QFT は
# 小角度回転を切り捨てることでゲート数を $O(nk)$ に削減します。
# ここで $k$ は切り捨て深さです。
#
# ### 5.1 利用可能な戦略の一覧

# %%
from qamomile.circuit.stdlib.qft import QFT

print("Available QFT strategies:")
for name in QFT.list_strategies():
    print(f"  - {name}")

# %% [markdown]
# ### 5.2 ゲート数の比較
#
# 8量子ビット QFT について、標準戦略と近似戦略を比較してみましょう。

# %%
qft_gate = QFT(8)

standard = qft_gate.get_resources_for_strategy("standard")
approx = qft_gate.get_resources_for_strategy("approximate")  # k=3

print("8-qubit QFT -- Standard Strategy:")
print(f"  H gates:    {standard.custom_metadata['num_h_gates']}")
print(f"  CP gates:   {standard.custom_metadata['num_cp_gates']}")
print(f"  SWAP gates: {standard.custom_metadata['num_swap_gates']}")
print(f"  Total:      {standard.custom_metadata['total_gates']}")

print()
print("8-qubit QFT -- Approximate Strategy (k=3):")
print(f"  H gates:    {approx.custom_metadata['num_h_gates']}")
print(f"  CP gates:   {approx.custom_metadata['num_cp_gates']}")
print(f"  SWAP gates: {approx.custom_metadata['num_swap_gates']}")
print(f"  Total:      {approx.custom_metadata['total_gates']}")

# %% [markdown]
# 近似戦略は、すべてのアダマールゲートと SWAP ゲートを維持しつつ、
# 制御位相ゲートの数を大幅に削減します。切り捨てによって導入される
# 誤差は $O(n / 2^k)$ でスケールするため、$k$ を増やすとゲート数が
# 増える代わりに精度が向上します。

# %%
# Compare across several sizes
print(f"{'n':>4}  {'Standard':>10}  {'Approx k=3':>12}  {'Savings':>8}")
print("-" * 42)
for n in [4, 8, 16, 32, 64]:
    qft_n = QFT(n)
    std = qft_n.get_resources_for_strategy("standard")
    apx = qft_n.get_resources_for_strategy("approximate")
    std_total = std.custom_metadata["total_gates"]
    apx_total = apx.custom_metadata["total_gates"]
    savings = 1 - apx_total / std_total if std_total > 0 else 0
    print(f"{n:>4}  {std_total:>10}  {apx_total:>12}  {savings:>7.1%}")

# %% [markdown]
# $n$ が大きくなるにつれ、近似戦略による削減効果はますます顕著になります。
# $n = 64$ の場合、近似 QFT は標準分解のおよそ半分のゲート数で済みます。
#
# ### 5.3 回路レベルでの戦略選択
#
# `@qkernel` 内で複合ゲートを適用する際に戦略を選択することもできます。
# 推定器は対応するリソースメタデータを自動的に使用します：
#
# ```python
# @qmc.qkernel
# def my_qft_circuit() -> qmc.Vector[qmc.Qubit]:
#     q = qmc.qubit_array(8, name="q")
#     qft_gate = QFT(8)
#     result = qft_gate(*[q[i] for i in range(8)], strategy="approximate")
#     for i in range(8):
#         q[i] = result[i]
#     return q
# ```
#
# これにより、アルゴリズムの他の部分を変更することなく、
# コストと精度のトレードオフをきめ細かく制御できます。

# %% [markdown]
# ## 6. カスタム複合ゲートと戦略
#
# Qamomile では、複数の分解戦略を持つ独自の複合ゲートを定義できます。
# これは以下の場面で役立ちます：
#
# - 実装のトレードオフの比較（精度 vs. ゲート数）
# - 内部構造が未知または無関係なオラクルやサブルーチンに対する
#   リソースメタデータの提供（スタブゲート）
#
# ### 6.1 カスタム複合ゲートの定義
#
# マーキングされた状態に位相を適用する単純な「マーキングオラクル」を
# 作成しましょう。2つの戦略を用意します：$P(\pi/4)$ 回転（T ゲート相当、
# 高精度）を使うものと、$P(\pi/2)$ 回転（S ゲート相当、Clifford のみ）
# を使うものです。

# %%
from dataclasses import dataclass

from qamomile.circuit.frontend.composite_gate import CompositeGate
from qamomile.circuit.ir.operation.composite_gate import ResourceMetadata


class MarkingOracle(CompositeGate):
    """A custom composite gate demonstrating strategy selection.

    The gate applies a marking pattern to each qubit using
    H-P-H rotations, followed by entangling CZ gates.
    """

    custom_name = "marking_oracle"

    def __init__(self, num_qubits: int):
        self._num_qubits = num_qubits

    @property
    def num_target_qubits(self) -> int:
        return self._num_qubits

    def _decompose(self, qubits):
        """Standard decomposition using P(pi/4) (T-equivalent) rotations."""
        n = self._num_qubits
        qubits_list = list(qubits)
        for i in range(n):
            qubits_list[i] = qmc.h(qubits_list[i])
            qubits_list[i] = qmc.p(qubits_list[i], math.pi / 4)  # T-equivalent
            qubits_list[i] = qmc.h(qubits_list[i])
        for i in range(n - 1):
            qubits_list[i], qubits_list[i + 1] = qmc.cz(
                qubits_list[i], qubits_list[i + 1]
            )
        return tuple(qubits_list)

    def _resources(self) -> ResourceMetadata:
        n = self._num_qubits
        return ResourceMetadata(
            t_gate_count=n,
            custom_metadata={
                "num_h_gates": 2 * n,
                "num_p_gates": n,
                "num_cz_gates": n - 1,
                "total_gates": 3 * n + (n - 1),
            },
        )


# %% [markdown]
# ### 6.2 戦略の追加
#
# 次に、$P(\pi/4)$ を $P(\pi/2)$（S ゲート相当）に置き換える Clifford
# のみの戦略を定義します。これにより T ゲートのコストが完全になくなります。
# T ゲートが高コストなフォールトトレラント環境で有用です。


# %%
@dataclass
class CliffordOracleStrategy:
    """Clifford-only strategy: replace P(pi/4) with P(pi/2)."""

    @property
    def name(self) -> str:
        return "clifford_only"

    def decompose(self, qubits):
        n = len(qubits)
        qubits_list = list(qubits)
        for i in range(n):
            qubits_list[i] = qmc.h(qubits_list[i])
            qubits_list[i] = qmc.p(qubits_list[i], math.pi / 2)  # S-equivalent
            qubits_list[i] = qmc.h(qubits_list[i])
        for i in range(n - 1):
            qubits_list[i], qubits_list[i + 1] = qmc.cz(
                qubits_list[i], qubits_list[i + 1]
            )
        return tuple(qubits_list)

    def resources(self, num_qubits):
        n = num_qubits
        return ResourceMetadata(
            t_gate_count=0,  # No T gates!
            custom_metadata={
                "num_h_gates": 2 * n,
                "num_p_gates": n,
                "num_cz_gates": n - 1,
                "total_gates": 3 * n + (n - 1),
            },
        )


# Register the strategy
MarkingOracle.register_strategy("clifford_only", CliffordOracleStrategy())

# %% [markdown]
# ### 6.3 戦略間のリソース比較

# %%
oracle = MarkingOracle(5)

std_res = oracle.get_resources_for_strategy()  # default: from _resources()
cliff_res = oracle.get_resources_for_strategy("clifford_only")

print("MarkingOracle (5 qubits) -- Strategy comparison:")
print()
print(f"  {'Metric':<20} {'Standard':>10} {'Clifford':>10}")
print(f"  {'-' * 40}")
print(f"  {'T gates':<20} {std_res.t_gate_count:>10} {cliff_res.t_gate_count:>10}")
print(
    f"  {'Total gates':<20} {std_res.custom_metadata['total_gates']:>10}"
    f" {cliff_res.custom_metadata['total_gates']:>10}"
)

# %% [markdown]
# 総ゲート数は同じですが、Clifford のみの戦略ではすべての T ゲートが
# 排除されます。これはフォールトトレラント量子コンピューティングにおいて
# 大きな利点です。T ゲートにはコストの高いマジック状態蒸留が必要だからです。

# %% [markdown]
# ### 6.4 カスタムゲートの回路での使用


# %%
@qmc.qkernel
def algorithm_with_oracle() -> qmc.Vector[qmc.Qubit]:
    """A simple circuit using our custom composite gate."""
    q = qmc.qubit_array(4, name="q")
    # Apply Hadamard layer
    for i in qmc.range(4):
        q[i] = qmc.h(q[i])
    # Apply our custom oracle
    oracle_gate = MarkingOracle(4)
    q[0], q[1], q[2], q[3] = oracle_gate(q[0], q[1], q[2], q[3])
    return q


est_algo = estimate_resources(algorithm_with_oracle.block)
print("Algorithm with MarkingOracle:")
for section, values in est_algo.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# ### 6.5 ブラックボックス推定のためのスタブゲート
#
# 内部構造が未知または無関係なサブルーチンを使うアルゴリズムのリソースを
# 推定する必要がある場合があります。Qamomile は**スタブゲート**をサポート
# しています。これはリソースメタデータを持つがゲートレベルの実装を持たない
# 複合ゲートです。

# %%
from qamomile.circuit.frontend.composite_gate import composite_gate


@composite_gate(
    stub=True,
    name="black_box_oracle",
    num_qubits=3,
    t_gate_count=10,
    query_complexity=1,
)
def black_box_oracle():
    """A stub gate: resource metadata only, no implementation."""
    pass


# Inspect metadata
meta = black_box_oracle.get_resource_metadata()
print("Stub gate metadata:")
print(f"  T-gate count:      {meta.t_gate_count}")
print(f"  Query complexity:  {meta.query_complexity}")


# %%
# Use the stub gate in a circuit
@qmc.qkernel
def grover_iteration() -> qmc.Vector[qmc.Qubit]:
    """Simplified Grover iteration using a black-box oracle."""
    q = qmc.qubit_array(3, name="q")
    for i in qmc.range(3):
        q[i] = qmc.h(q[i])
    # Apply the black-box oracle (no implementation needed)
    q[0], q[1], q[2] = black_box_oracle(q[0], q[1], q[2])
    # Diffuser
    for i in qmc.range(3):
        q[i] = qmc.h(q[i])
        q[i] = qmc.x(q[i])
    q[0], q[1] = qmc.cz(q[0], q[1])
    q[1], q[2] = qmc.cz(q[1], q[2])
    for i in qmc.range(3):
        q[i] = qmc.x(q[i])
        q[i] = qmc.h(q[i])
    return q


est_grover = estimate_resources(grover_iteration.block)
print("Grover iteration with stub oracle:")
for section, values in est_grover.to_dict().items():
    if isinstance(values, dict):
        print(f"  {section}:")
        for key, val in values.items():
            print(f"    {key}: {val}")
    else:
        print(f"  {section}: {values}")

# %% [markdown]
# 推定器はスタブの T ゲート数（10）を検出し、周囲のゲートに加算します。
# これにより、オラクルの内部が利用できない場合でも、アルゴリズム全体の
# コストを推定できます。
#
# **クエリ複雑性**は `estimate_resources()` の出力には表示されません。
# これは複合ゲート自体のメタデータであり、回路レベルのメトリクスでは
# ないためです。アクセスするには、ゲートに対して直接
# `get_resource_metadata()` を使用してください：

# %%
meta = black_box_oracle.get_resource_metadata()
print(f"Oracle query complexity: {meta.query_complexity}")

# %% [markdown]
# **次のステップ：**
#
# - [トランスパイル](10_transpile.ipynb) -- トランスパイルパイプラインについて学ぶ
# - [カスタムエグゼキュータ](11_custom_executor.ipynb) -- クラウド量子ハードウェアで回路を実行する
# - [QAOA](../optimization/qaoa.ipynb) -- 変分回路による最適化

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **`estimate_resources()` を使って量子ビット数、ゲート数、回路深さを SymPy 式として取得する方法** -- `estimate_resources(kernel.block)` が回路 IR をたどり、すべてのメトリクスをシンボリック式として含む `ResourceEstimate` を返します。
# - **`.to_dict()` で利用可能な全メトリクスを反復処理する方法** -- ネスト辞書は前方互換性があり、シリアライズに適しています。将来のリリースで追加された新しいフィールドも自動的に表示されます。
# - **シンボリック（パラメトリック）回路での `.substitute()` と `.simplify()` の使い方** -- `.substitute()` で具体的な値を代入し、`.simplify()` で SymPy により式の複雑さを軽減します。
# - **量子位相推定（QPE）のリソースプロファイル分析** -- 手動実装（IQFT + 制御ユニタリ）と組み込みの `qmc.qpe()` プリミティブはどちらも同じ漸近的スケーリングを示します。
# - **複合ゲートの分解戦略の比較** -- QFT の標準戦略と近似戦略が、小角度回転の切り捨てによって精度とゲート数削減をトレードオフする様子を示しました。
# - **戦略付きカスタム複合ゲートの定義と、ブラックボックスリソース推定のためのスタブゲートの使い方** -- カスタム `CompositeGate` クラスは複数の戦略をサポートし、スタブゲートはゲートレベルの実装なしにリソースメタデータを提供することで、アルゴリズムレベルのコスト分析を可能にします。
