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
# # Qamomile v0.10.0
#
# Qamomile v0.10.0は、量子プログラミング層をゼロから再構築したリリースです。これにより従来のスコープである量子最適化アルゴリズム開発のみならず、より一般的に任意の量子プログラム/アルゴリズムの開発をカバーできるようになりました。v0.9.0からの変更点は非常に多岐にわたるため、差分の一覧ではなく、**v0.10.0で何ができるようになったか**をご紹介します。
#
# より詳しい使い方については[チュートリアル](../tutorial)をご覧ください。
# ```
# pip install qamomile==0.10.0
# ```

# %% [markdown]
# ## フロントエンド: `@qkernel`
#
# Qamomile v0.10.0で量子プログラムを記述する中心となるAPIは`@qkernel`デコレータです。**型アノテーション**付きの通常のPython関数を書くだけで、Qamomileがそれを中間表現（IR）にトレースします。このIRは解析・可視化・トランスパイルに利用されますが、ユーザーが直接操作する必要はありません。また、`for`/`while`ループや`if`文による古典的な制御フローもサポートされており、これらもIRにトレースされます。
#
# `@qkernel`を書くための必ず守る必要があるルールは以下の通りです:
#
# - **型ヒントは必須です。**引数と戻り値には主にQamomileで用意された型（`Qubit`、`Bit`、`Float`、`UInt`、`Vector[T]`、`Dict[K, V]`、`Tuple[...]`）を使用します。
# - **量子ビットはアフィン型です。**`Qubit`ハンドルはゲート適用のたびに再代入が必要です（`q = qmc.h(q)`）。
#
# 利用可能なゲートにはH、X、Y、Z、S、T、RX、RY、RZ、RZZ、CX、CZ、CCX、CP、SWAPなどがあります。測定には`qmc.measure()`を使用します。

# %%
import qamomile.circuit as qmc


@qmc.qkernel
def ghz_state(
    n: qmc.UInt,
) -> qmc.Vector[qmc.Bit]:  # 引数と戻り値に型ヒントを指定
    q = qmc.qubit_array(n, name="q")  # 指定したサイズと名前で量子ビット配列を確保

    q[0] = qmc.h(q[0])
    for i in qmc.range(1, n):  # qmc.rangeで量子ビット1からn-1をイテレーション
        q[0], q[i] = qmc.cx(q[0], q[i])

    return qmc.measure(q)


# %%
ghz_state.draw(n=4, fold_loops=False)

# %% [markdown]
# 関連の深いチュートリアル：[はじめての量子カーネル](../tutorial/your-first-quantum-kernel)、[パラメータ付き量子カーネル](../tutorial/parameterized-kernels)、[実行モデル: sample() vs run()](../tutorial/execution-models)、[再利用パターン: QKernelの合成とコンポジットゲート](../tutorial/reuse-patterns)

# %% [markdown]
# ## リソース推定
#
# Qamomile v0.10.0ではシンボリックなリソース推定機能が提供されます。任意のqkernelに対して`estimate_resources()`を呼び出すと、qkernelを**実行することなく**量子ビット数やゲートの内訳を取得できます。これらは入力パラメータのシンボリック式として表現されるため、問題サイズに対するリソースのスケーリングの解析が容易です。具体的な値を代入して、特定の入力サイズに対する具体的な推定値を得ることも可能です。さらに、`@composite_gate`デコレータにて`stub=True`を指定して定義できる実装が不要なブラックボックスオラクルを定義することができ、これを用いたqkernelに対するリソースの推定も可能です。

# %%
est = ghz_state.estimate_resources()
print("qubits:", est.qubits)
print("total two-qubit gates:", est.gates.two_qubit)

# %%
# 特定のサイズで評価
print("two-qubit gates at n=100:", est.substitute(n=100).gates.two_qubit)

# %% [markdown]
# 関連の深いチュートリアル：[リソース推定](../tutorial/resource-estimation)、[再利用パターン: QKernelの合成とコンポジットゲート](../tutorial/reuse-patterns)

# %% [markdown]
# ## マルチ量子SDKトランスパイル
#
# `@qkernel`で回路を一度定義すれば、対応するサポート済みの量子SDKの形式へとトランスパイルが可能です。トランスパイルされたqkernelは対応する変換先の量子SDK毎の`ExecutableProgram`となります。Qamomile v0.10.0には各量子SDKにプリセットのExecutorが用意されているため，変換先の量子SDKのコードを書かずとも実行まで行うことができます。このプリセットのExecutorは原則シミュレータ実行を仮定しています。より細かい制御や実機へのアクセスのために、カスタムExecutorを作成することもできます。
#
# | バックエンド | モジュール | トランスパイラ | デフォルト実行 |
# |---------|--------|-----------|-----------|
# | Qiskit | `qamomile.qiskit` | `QiskitTranspiler` | ローカルシミュレータ |
# | QURI Parts | `qamomile.quri_parts` | `QuriPartsTranspiler` | ローカルシミュレータ |
# | CUDA-Q | `qamomile.cudaq` | `CudaqTranspiler` | ローカルシミュレータ |
#
# さらに、Qiskitについては、[qBraid](https://docs.qbraid.com/)実行をサポートとしており、クラウド量子デバイス上で実行することも可能です。
#
# Qiskitは`pip install qamomile`にデフォルトで含まれていますが、その他の量子SDKやqBraid環境実行のためにはオプションの追加パッケージが必要です:
#
# ```bash
# pip install "qamomile[cudaq-cu12]"   # CUDA-Q with CUDA 12.x (Linux)
# pip install "qamomile[cudaq-cu13]"   # CUDA-Q with CUDA 13.x (Linux / macOS ARM64)
# pip install "qamomile[quri_parts]"   # QURI Parts
# pip install "qamomile[qbraid]"       # qBraid integration for Qiskit
# ```

# %%
from qamomile.qiskit import QiskitTranspiler

qiskit_transpiler = QiskitTranspiler()
qiskit_executable = qiskit_transpiler.transpile(ghz_state, bindings={"n": 4})
samples = qiskit_executable.sample(
    qiskit_transpiler.executor(),  # 実行するエグゼキュータを指定
    shots=1024,
).result()

for outcome, count in samples.results:
    print(f"  outcome={outcome}, count={count}")

# %% [markdown]
# 関連の深いチュートリアル：[実行モデル: sample() vs run()](../tutorial/execution-models)、[qBraidサポート - QBraidExecutor](../collaboration/qbraid-executor)

# %% [markdown]
# ## 標準ライブラリ
#
# 量子アルゴリズムを手軽に書けるよう、Qamomile v0.10.0では`qmc.stdlib`と`qmc.algorithm`モジュールに組み込みのアルゴリズムを用意しています。現在は量子フーリエ変換（QFT）や量子位相推定（QPE）、Quantum Approximation Optimization Algorithm（QAOA）などが含まれます。`qmc.stdlib`は広く使われる基本的なアルゴリズムブロック、`qmc.algorithm`はより特化したアルゴリズムとして収録していますが、この分類は厳密なものではなく、リリース毎に変更される可能性があります。
#
# これらのモジュールにはアルゴリズムを随時追加していく予定です。ぜひご期待ください！

# %%
from qamomile.circuit.stdlib import qft


@qmc.qkernel
def qft_example() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(4, name="q")
    q[0] = qmc.x(q[0])
    q = qft(q)
    return qmc.measure(q)


qft_example.draw(expand_composite=True)

# %% [markdown]
# ## 最適化コンバータ
#
# v0.9.0で提供されていたコンバータは`qamomile.optimization`以下で引き続き利用可能で、新しい`@qkernel`ベースの回路層を使って書き直されています。
# [JijModeling](https://www.documentation.jijzept.com/docs/jijmodeling/)やOMMXの数理モデルから、すぐに実行可能な回路を生成します。

# %% [markdown]
# ## さらに詳しく
# - [チュートリアル](../tutorial)
# - [GitHubリポジトリ](https://github.com/Jij-Inc/Qamomile)
