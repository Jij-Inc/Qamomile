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
# # Qamomile入門
#
# Qamomileへようこそ。Qamomileは、Pythonで量子回路を構築・解析・実行するための
# 量子コンピューティングSDKです。
#
# ## このチュートリアルで学ぶこと
# - Qamomileとは何か、量子エコシステムにおける位置づけ
# - 最初の量子回路の作成と実行
# - 線形型システム（量子複製不可能定理の適用）
# - QiskitTranspilerを用いた実行
# - Qamomileのトレースとトランスパイルの仕組み
# - パラメトリック回路の基礎
# - リソース見積もりの基礎
# - 標準ライブラリとアルゴリズムライブラリの基礎
# - 最適化機能の概要

# %% [markdown]
# ## 1. Qamomileとは
#
# **Qamomile**は、量子コンピューティング向けのPython SDKです。
# ノイズのある中規模量子コンピューティング（NISQ）から
# 誤り耐性量子コンピューティング（FTQC）への橋渡しとなるように設計されており、
# 両方のパラダイムに対応した量子プログラムを単一のフレームワークで記述できます。
#
# ### 位置づけ
#
# Qamomileは量子コンピューティングの両方のパラダイムをサポートしています：
#
# - **NISQアルゴリズム**: QAOAやVQEなどの変分アルゴリズム
# - **誤り耐性アルゴリズム**: 標準ライブラリを通じたQPEなどの厳密アルゴリズム、
#   および計画のための代数的リソース見積もり
#
# ### 主な特徴
#
# | 特徴 | 説明 |
# |------|------|
# | **Pythonライクな構文** | `@qkernel`デコレータで量子プログラムを定義 |
# | **型安全性** | すべてのパラメータと戻り値に型アノテーションが必要 |
# | **線形型** | 実行前に量子複製不可能定理を適用 |
# | **マルチバックエンド** | 現在Qiskitに対応、CUDA-QおよびQURI Partsは近日対応予定、さらに拡大予定 |
# | **標準ライブラリ** | QFT、IQFT、QPEを組み込みで提供（分解戦略あり、さらに追加予定） |
# | **リソース見積もり** | シンボリックなゲート数と深さの解析 |
# | **最適化** | [ommx](https://jij-inc.github.io/ommx/en/introduction.html)統合によるQAOA、FQAOA、QRAOコンバータ（さらに追加予定） |

# %% [markdown]
# ## 2. 最初の量子回路
#
# ### `@qkernel`関数としての量子プログラム
#
# Qamomileでは、**すべての量子プログラム（量子回路）は
# `@qmc.qkernel`デコレータを付けたPython関数として記述します**。
# これが量子計算を定義する唯一の方法であり、回路を作成する他の方法はありません。
#
# `@qkernel`関数には最低限以下の要件があります：
#
# 1. **`@qmc.qkernel`デコレータ** — 関数を量子カーネルとしてマークし、
#    Qamomileがトレース、可視化、トランスパイルできるようにします。
# 2. **すべてのパラメータと戻り値の型アノテーション** — Qamomileはこれらを使って
#    量子ビットの割り当て、パラメータの処理、測定結果のデコードを行います。
#    省略するとトレース時にエラーが発生します。
# 3. **戻り値** — 関数は何かを返す必要があります（測定結果、量子ビット、
#    または期待値）。戻り値の型アノテーションにより、
#    Qamomileが出力の解釈方法を決定します。
#
# ```python
# @qmc.qkernel
# def my_circuit(param: qmc.Float) -> qmc.Bit:   # annotations required
#     q = qmc.qubit(name="q")      # allocate qubits inside the function
#     q = qmc.ry(q, param)         # apply gates (reassign to respect affine types)
#     return qmc.measure(q)        # return measurement result
# ```
#
# 利用可能な型の全カタログについては[02_type_system](02_type_system.ipynb)を参照してください。
#
# ### 量子ビットとゲート
#
# **量子ビット**（qubit）は量子情報の基本単位です。古典ビット（常に0か1）とは異なり、
# 量子ビットは測定されるまで$|0\rangle$と$|1\rangle$の**重ね合わせ状態**に
# 存在することができます。
#
# 量子**ゲート**は量子ビットの状態を変換します。最も単純なゲートは
# **Xゲート**（NOTゲート）で、$|0\rangle \to |1\rangle$を反転し、その逆も同様です。
#
# 最初の回路を作成してみましょう。

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def x_gate_circuit() -> qmc.Bit:
    """Apply X gate to flip |0> to |1>."""
    q = qmc.qubit(name="q")
    q = qmc.x(q)
    return qmc.measure(q)


x_gate_circuit.draw()

# %% [markdown]
# ### コードの解説
#
# 1. **`@qmc.qkernel`**: この関数を量子カーネルとしてマーク
# 2. **`-> qmc.Bit`**: 戻り値の型アノテーション（測定結果）
# 3. **`qmc.qubit(name="q")`**: 量子ビットを1つ作成、初期状態は$|0\rangle$
# 4. **`q = qmc.x(q)`**: Xゲートを適用。再代入に注目してください！
# 5. **`qmc.measure(q)`**: 量子ビットを測定し、古典`Bit`を返す
# 6. **`x_gate_circuit.draw()`**: 回路図を可視化。すべての`@qkernel`には`.draw()`メソッドがあり、Matplotlibを使って回路を描画します。

# %% [markdown]
# ## 3. 線形型システム
#
# `q = qmc.x(q)`というパターンに気づいたかもしれません。なぜ再代入が必要なのでしょうか？
#
# 量子力学では、量子ビットはコピーできません（**複製不可能定理**）。
# Qamomileはこれを**線形型システム**で強制します。量子ビットがゲートに入ると、
# 古いハンドルは消費され、新しいハンドルが返されます。
# 常に戻り値を受け取る必要があります。
#
# ```python
# # Correct
# q = qmc.h(q)      # captures the new handle
# q = qmc.x(q)      # uses the updated handle
#
# # Wrong — will cause an error
# qmc.h(q)           # ignores return value
# qmc.x(q)           # tries to use consumed handle
# ```


# %%
# Error example: using a qubit twice
@qmc.qkernel
def bad_example() -> tuple[qmc.Bit, qmc.Bit]:
    q = qmc.qubit(name="q")
    q1 = qmc.h(q)  # consumes q
    q2 = qmc.x(q)  # ERROR: q was already consumed
    return qmc.measure(q1), qmc.measure(q2)


try:
    bad_example.draw()
except Exception as e:
    print(f"Error (expected): {type(e).__name__}: {e}")

# %% [markdown]
# ### 線形型のルール
#
# | コード | 有効？ | 理由 |
# |--------|--------|------|
# | `q = qmc.h(q)` | OK | 戻り値を再代入 |
# | `qmc.h(q)` | NG | 戻り値を無視 |
# | `q1 = qmc.h(q); q2 = qmc.x(q)` | NG | qを2回使用 |
# | `q = qmc.h(q); q = qmc.x(q)` | OK | 順番に更新 |
#
# 多量子ビットゲートの場合、両方の量子ビットが返されます：
# ```python
# q0, q1 = qmc.cx(q0, q1)   # CNOT returns both qubits
# ```

# %% [markdown]
# ## 4. Qiskitでの実行
#
# Qamomileの回路はバックエンドに依存しません。実行するには、
# 回路を特定のバックエンド形式に変換する**トランスパイラ**を使用します。
#
# 現在、`QiskitTranspiler`がサポートされているバックエンドです。
# CUDA-QおよびQURI Partsのサポートは現在開発中で、
# さらに多くのバックエンドが計画されています。回路はバックエンドに依存しない
# `@qkernel`関数として定義されるため、新しいバックエンドが利用可能になっても
# コードを変更する必要はありません。
#
# 実行パイプラインには、カーネルの戻り値に応じて2つのモードがあります：
# ```
# @qkernel (returns Bit/Vector[Bit]/Float via measure)
#   → transpile() → ExecutableProgram → sample(executor, shots=N) → SampleResult
#
# @qkernel (returns Float via expval)
#   → transpile() → ExecutableProgram → run(executor) → Float
# ```
#
# - **`sample()`**: ショットベースの測定。`(value, count)`ペアを持つ`SampleResult`を
#   返します。個別の結果ごとに1つのエントリが含まれます。
# - **`run()`**: 期待値計算。`Float`値を直接返します
#   （期待値$\langle\psi|H|\psi\rangle$）。変分アルゴリズムで`qmc.expval()`と
#   一緒に使用します（[08_parametric_circuits](08_parametric_circuits.ipynb)を参照）。
#
# このチュートリアルでは`sample()`のみ使用します。
# オブザーバブルや変分回路を扱う際に`run()`を紹介します。

# %%
# Create transpiler
transpiler = QiskitTranspiler()

# Compile the circuit
executable = transpiler.transpile(x_gate_circuit)

# Execute on simulator (1000 shots)
job = executable.sample(transpiler.executor(), shots=1000)
result = job.result()

print("=== X Gate Circuit Results ===")
for value, count in result.results:
    print(f"  {value}: {count}")

# %% [markdown]
# Xゲートは$|0\rangle$を$|1\rangle$に反転させるため、1000回の測定すべてで
# `1`が得られるはずです。
#
# トランスパイル後のQiskit回路も確認できます：

# %%
qiskit_circuit = executable.get_first_circuit()
qiskit_circuit.draw(output="mpl")

# %% [markdown]
# ### 量子ビットの順序規則
#
# 複数の量子ビットを扱う場合、Qamomileは以下の規則を使用します：
#
# - **ケット表記**はビッグエンディアンです：最も左のビットが
#   **最も大きいインデックス**の量子ビットです。例えば、3量子ビットの$|110\rangle$は
#   `q[2]=1, q[1]=1, q[0]=0`を意味します。
# - **タプル結果**は配列順に従います：`(q[0], q[1], ..., q[n-1])`。
#   したがって、状態$|110\rangle$は測定結果では`(0, 1, 1)`として現れます。
#
# 3量子ビットの回路で実際に確認してみましょう。


# %%
@qmc.qkernel
def ordering_demo() -> tuple[qmc.Bit, qmc.Bit, qmc.Bit]:
    """Demonstrate qubit ordering: q0=0, q1=1, q2=1 → ket |110>."""
    q0 = qmc.qubit(name="q0")
    q1 = qmc.qubit(name="q1")
    q2 = qmc.qubit(name="q2")
    # q0 stays |0>
    q1 = qmc.x(q1)  # q1 → |1>
    q2 = qmc.x(q2)  # q2 → |1>
    return qmc.measure(q0), qmc.measure(q1), qmc.measure(q2)


ordering_demo.draw()

# %%
exec_ord = transpiler.transpile(ordering_demo)
result_ord = exec_ord.sample(transpiler.executor(), shots=100).result()

print("=== Qubit Ordering Demo ===")
for value, count in result_ord.results:
    print(f"  {value}: {count}")

# %% [markdown]
# 状態は`q0=0, q1=1, q2=1`です。ケット表記（ビッグエンディアン：q2 q1 q0）では
# $|110\rangle$ですが、タプル結果は配列順`(q0, q1, q2)`に従い`(0, 1, 1)`となります。

# %% [markdown]
# ## 5. トレースとコンパイル
#
# `@qmc.qkernel`関数を定義しても、Qamomileは量子操作を即座に実行しません。
# 代わりに、2段階のアプローチを使用します：
#
# 1. **トレース**: `.draw()`や`transpile()`を呼び出すと、Qamomileは
#    関数本体を**トレース**して中間表現（IR）を構築します。
#    これは操作とデータ依存関係の有向グラフです。
# 2. **トランスパイル**: IRグラフはマルチパスのパイプラインを通じて処理され、
#    最適化されてバックエンド固有の回路に変換されます。
#
# ```
# @qkernel function
#     ↓  trace
# IR Graph (operations + dependencies)
#     ↓  inline → constant_fold → analyze → separate → emit
# Backend Circuit (e.g., Qiskit QuantumCircuit)
# ```
#
# このアーキテクチャには2つの重要な利点があります：
#
# - **バックエンド非依存性**: 回路を`@qkernel`として一度定義すれば、
#   コードを変更せずに任意のサポートバックエンドにトランスパイルできます。
# - **最適化の機会**: マルチパスパイプラインにより、最終的な回路を生成する前に
#   サブルーチンのインライン化、定数畳み込み、依存関係解析が可能です。
#
# トランスパイラパイプラインの詳細は[10_transpile](10_transpile.ipynb)を参照してください。

# %% [markdown]
# ## 6. パラメトリック回路
#
# 多くの量子アルゴリズムは、調整可能なパラメータを持つ回路を使用します。
# Qamomileでは、パラメータの型として`qmc.Float`を使用します。

# %%
import math


@qmc.qkernel
def rotation_circuit(theta: qmc.Float) -> qmc.Bit:
    """Parameterized rotation around the Y-axis."""
    q = qmc.qubit(name="q")
    q = qmc.ry(q, theta)
    return qmc.measure(q)


rotation_circuit.draw()

# %% [markdown]
# `draw()`にパラメータ値を直接渡して、具体的な値が埋め込まれた
# 回路を確認することもできます：

# %%
rotation_circuit.draw(theta=math.pi / 4)

# %% [markdown]
# ### bindingsとparameters
#
# トランスパイル時に、2つの方法で値を提供できます：
#
# - **`bindings`**: トランスパイル時に固定される値（回路構造がこれに依存する場合があります）
# - **`parameters`**: 自由なまま保持され、再トランスパイルなしに実行間で変更できる値

# %%
# Fix theta at transpile time
exec_fixed = transpiler.transpile(rotation_circuit, bindings={"theta": math.pi / 2})
result_fixed = exec_fixed.sample(transpiler.executor(), shots=1000).result()

print("=== RY(pi/2) — fixed at transpile time ===")
for value, count in result_fixed.results:
    percentage = count / 1000 * 100
    print(f"  {value}: {count} ({percentage:.1f}%)")

# %%
# Keep theta as a free parameter
exec_param = transpiler.transpile(rotation_circuit, parameters=["theta"])

# Execute multiple times with different values without retranspiling
for angle, name in [(0, "0"), (math.pi / 4, "pi/4"), (math.pi, "pi")]:
    res = exec_param.sample(
        transpiler.executor(), bindings={"theta": angle}, shots=1000
    ).result()
    counts = {str(v): c for v, c in res.results}
    print(f"RY({name}): {counts}")

# %% [markdown]
# これはVQEやQAOAなどの変分量子アルゴリズムの基盤であり、
# 古典-量子ループの中でパラメータが最適化されます。

# %% [markdown]
# ## 7. リソース見積もり
#
# 実機で実行する前に、回路に必要な量子ビット数やゲート数を見積もりたい場合が
# あります。Qamomileはシンボリックパラメータに対応した
# **代数的リソース見積もり**を提供します。
#
# 回路がシンボリックな`UInt`の上限を持つ`qmc.range()`を使用する場合、
# 見積もり結果はリソースが問題サイズに対してどのようにスケールするかを
# 記述するSymPy式になります。


# %%
from qamomile.circuit.estimator import estimate_resources


@qmc.qkernel
def ghz_circuit(n: qmc.UInt) -> qmc.Vector[qmc.Qubit]:
    """GHZ state with symbolic size n."""
    q = qmc.qubit_array(n, name="q")
    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = qmc.cx(q[i], q[i + 1])
    return q


est_ghz = estimate_resources(ghz_circuit.block)

print("=== GHZ State Resources (symbolic) ===")
print(f"  Qubits:        {est_ghz.qubits}")
print(f"  Total gates:   {est_ghz.gates.total}")
print(f"  Two-qubit:     {est_ghz.gates.two_qubit}")
print(f"  Circuit depth: {est_ghz.depth.total_depth}")

# %% [markdown]
# 結果にはシンボル**n**が含まれており、リソースが問題サイズに対してどのように
# 増加するかを表す閉形式の式です。リソース見積もりの完全なチュートリアルは
# [09_resource_estimation](09_resource_estimation.ipynb)を参照してください。

# %% [markdown]
# ## 8. 標準ライブラリとアルゴリズムライブラリ
#
# Qamomileは、ビルド済みの回路コンポーネントとして2つのライブラリを提供しています。
#
# **標準ライブラリ**（`qamomile.circuit.stdlib`）：量子アルゴリズムで
# よく使われる基本的なビルディングブロックです。
#
# | コンポーネント | 説明 |
# |----------------|------|
# | `qft` / `iqft` | 量子フーリエ変換（複数の分解戦略あり） |
# | `qpe` | 量子位相推定 |
#
# **アルゴリズムライブラリ**（`qamomile.circuit.algorithm`）：変分アルゴリズムや
# 最適化アルゴリズム向けのより具体的な回路パターンです。
#
# | コンポーネント | 説明 |
# |----------------|------|
# | `qaoa_circuit`, `qaoa_state` | QAOA回路の構築 |
# | `fqaoa_layers`, `fqaoa_state` | フェルミオンQAOAのコンポーネント（Givens回転、ホッピングゲート） |
# | `rx_layer`, `ry_layer`, `rz_layer` | パラメータ付き回転レイヤー |
# | `cz_entangling_layer` | CZエンタングリングレイヤー |
#
# どちらのライブラリも活発に開発中です。次にどの回路パターンやアルゴリズムを
# 追加すべきかについてのフィードバックを歓迎します。
# [GitHub](https://github.com/Jij-Inc/Qamomile)でissueを作成してください。
#
# 標準ライブラリの詳細なチュートリアルは[05_stdlib](05_stdlib.ipynb)を参照してください。

# %% [markdown]
# ## 9. 最適化機能
#
# Qamomileには、量子コンピュータで組合せ最適化問題を解くための
# コンバータが含まれています。
#
# 最適化パイプライン：
# ```
# Mathematical Model (JijModeling)
#      ↓  Interpreter + ommx.v1.Instance
# Converter (QAOAConverter, FQAOAConverter, QRAO31Converter)
#      ↓  get_cost_hamiltonian() / transpile()
# Quantum Circuit + Classical Optimization Loop
#      ↓  decode()
# Solution
# ```
#
# 利用可能なコンバータ：
#
# | コンバータ | アルゴリズム | 用途 |
# |------------|--------------|------|
# | `QAOAConverter` | QAOA | 汎用的な組合せ最適化 |
# | `FQAOAConverter` | Fermionic QAOA | 制約付き最適化（制約の厳密な適用） |
# | `QRAO31Converter` | QRAO 3-to-1 | 量子ビット効率の良いエンコーディング（1量子ビットに3変数） |
#
# 新しい量子最適化アルゴリズムの開発に伴い、さらに多くのコンバータが計画されています。
# 次にどのアルゴリズムやコンバータを優先すべきかについて、
# コミュニティからのフィードバックを積極的に歓迎しています。
# 新しいコンバータが有用なユースケースがある場合は、
# [GitHub](https://github.com/Jij-Inc/Qamomile)でissueを作成してください。
#
# 数理モデリング層の詳細については、
# [ommxドキュメント](https://jij-inc.github.io/ommx/en/introduction.html)
# および[JijModelingチュートリアル](https://jij-inc-jijmodeling-tutorials-en.readthedocs-hosted.com/en/latest/introduction.html)を参照してください。
#
# 最適化チュートリアル（[QAOA](../optimization/qaoa.ipynb)、[FQAOA](../optimization/fqaoa.ipynb)、[QRAO](../optimization/qrao31.ipynb)、[カスタムコンバータ](../optimization/custom_converter.ipynb)）で詳細を確認できます。

# %% [markdown]
# ## 10. まとめ
#
# このチュートリアルでは、Qamomileの基本的な概念を扱いました：
#
# 1. **`@qmc.qkernel`**: 量子回路をPython関数として定義
# 2. **線形型**: ゲート適用後は常に再代入（`q = qmc.h(q)`）
# 3. **実行**: `QiskitTranspiler` → `transpile()` → `sample()`
# 4. **トレースとトランスパイル**: `@qkernel`はIRグラフにトレースされ、マルチパスパイプラインを通じてトランスパイルされる
# 5. **パラメトリック回路**: `bindings` / `parameters`を使った`qmc.Float`パラメータ
# 6. **リソース見積もり**: `estimate_resources(kernel.block)`によるシンボリックなゲート数
# 7. **標準ライブラリとアルゴリズムライブラリ**: ビルド済みのQFT、QPE、QAOA、FQAOAコンポーネント
# 8. **最適化**: QAOA、FQAOA、QRAO用コンバータ（さらに追加予定）
#
# ### 次のステップ
#
# | チュートリアル | トピック |
# |----------------|---------|
# | `02_type_system.ipynb` | 完全な型システム：Qubit、Float、UInt、Bit、Vector、Dict |
# | `03_gates.ipynb` | 完全なゲートリファレンス（全11ゲート） |
# | `04_superposition_entanglement.ipynb` | 重ね合わせ、干渉、Bell/GHZ状態 |
# | `05_stdlib.ipynb` | QFT、QPE、アルゴリズムモジュール |
# | `06_composite_gate.ipynb` | CompositeGate、`@composite_gate`、スタブゲート |
# | `07_first_algorithm.ipynb` | Deutsch-Jozsaアルゴリズム |
# | `08_parametric_circuits.ipynb` | パラメトリック回路とQAOAのスクラッチ実装 |
# | `09_resource_estimation.ipynb` | 代数的リソース見積もり |
# | `10_transpile.ipynb` | トランスパイラパイプラインの内部構造 |
# | `11_custom_executor.ipynb` | カスタムバックエンドの統合 |
# | `optimization/qaoa.ipynb` | 組合せ最適化のためのQAOA |
# | `optimization/fqaoa.ipynb` | 制約適用付きフェルミオンQAOA |
# | `optimization/qrao31.ipynb` | 量子ランダムアクセス最適化 |
# | `optimization/custom_converter.ipynb` | 独自コンバータの構築 |

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **Qamomileとは何か、量子エコシステムにおける位置づけ** — QamomileはNISQからFTQCへの橋渡しとなり、両方のパラダイムに対応した量子プログラムを単一のフレームワークで記述できます。
# - **最初の量子回路の作成と実行** — `@qmc.qkernel`と`qmc.qubit()`、`qmc.x()`などのゲート、`qmc.measure()`を使って回路を構築・可視化しました。
# - **線形型システム（量子複製不可能定理の適用）** — ゲートは量子ビットを消費して返します。量子複製不可能定理をトランスパイル時に適用するために、常に再代入（`q = qmc.h(q)`）してください。
# - **QiskitTranspilerを用いた実行** — `QiskitTranspiler`は`transpile()`でカーネルをトランスパイルし、`sample()`で実行して測定結果を得ます。
# - **Qamomileのトレースとトランスパイルの仕組み** — `@qkernel`関数はIRグラフにトレースされ、マルチパスパイプライン（inline、constant fold、analyze、separate、emit）を通じてバックエンド固有の回路が生成されます。
# - **パラメトリック回路の基礎** — `qmc.Float`パラメータは`bindings=`でトランスパイル時に固定するか、`parameters=`で自由なまま保持して変分アルゴリズムに使用できます。
# - **リソース見積もりの基礎** — `estimate_resources(kernel.block)`はシンボリックなゲート数と回路深さを生成し、SymPy式を通じて問題サイズに応じたスケーリングを表現します。
# - **標準ライブラリとアルゴリズムライブラリの基礎** — QFT、QPEなどのビルド済みコンポーネント（さらに追加予定）。
# - **最適化機能の概要** — QAOA、FQAOA、QRAOコンバータがJijModelingの問題をモデル→コンバータ→回路のパイプラインで実行可能な量子回路に変換します。
