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
# Qamomile v0.10.0は、量子プログラミング層をゼロから再構築したリリースです。v0.9.0からの変更点は非常に多岐にわたるため、差分の一覧ではなく、**v0.10.0で何ができるようになったか**をご紹介します。
#
# v0.10.0では、型付きのPython関数として量子プログラムを記述でき、古典的な制御フロー（ループ、条件分岐、スパースなイテレーション）にも対応しています。さらに、ブラックボックスのオラクルを含むプログラムでもシンボリックなリソース推定が可能です。同一のプログラムを複数の量子SDK（現時点ではQiskit、QURI Parts、CUDA-Q）にトランスパイルできます。
#
# 実践的な使い方は[チュートリアル](../tutorial)をご覧ください。
# ```
# pip install qamomile==0.10.0
# ````

# %% [markdown]
# ## フロントエンド: `@qkernel`
#
# 中心となるAPIは`@qkernel`デコレータです。**型アノテーション**付きの通常のPython関数を書くだけで、Qamomileがそれを中間表現（IR）にトレースします。このIRは解析・可視化・トランスパイルに利用されますが、ユーザーが直接操作する必要はありません。`@qkernel`でPythonコードを書けば、あとはトランスパイラが処理をします。

# %% [markdown]
# ### `@qkernel`を書くときのルール:
#
# - **型ヒントは必須です。** 引数と戻り値にはQamomileのシンボリック型（`Qubit`、`Bit`、`Float`、`UInt`、`Vector[T]`、`Dict[K, V]`、`Tuple[...]`）を使用します。
# - **量子ビットはアフィン型です。** `Qubit`ハンドルはゲート適用のたびに再代入が必要です（`q = qmc.h(q)`）。これにより、コンパイラが量子ビットのライフタイムを正確に追跡できます。
# - **戻り値の型が実行モードを決定します。** `Bit` / `Vector[Bit]`を返す場合は`sample()`（ショットベース）、`Float`（`expval`経由）を返す場合は`run()`（期待値計算）になります。
#
# 利用可能なゲートにはH、X、Y、Z、S、T、RX、RY、RZ、RZZ、CX、CZ、CCX、CP、SWAPなどがあります。測定には`qmc.measure()`を使用します。
#
# 詳しくは[はじめての量子カーネル](../tutorial/your-first-quantum-kernel)、[パラメータ付き量子カーネル](../tutorial/parameterized-kernels)、[実行モデル: sample() vs run()](../tutorial/execution-models)、[再利用パターン: QKernelの合成とコンポジットゲート](../tutorial/reuse-patterns)をご参照ください。

# %% [markdown]
# ### 古典制御フロー
#
# Qamomileはqkernel**内部**での古典制御フローをサポートしています:
#
# - **`qmc.range(start, stop, step)`**（または **`qmc.range(stop)`**）— 量子ビットに対するパラメータ付き`for`ループ。
# - **`qmc.items(dict)`** — グラフのエッジや相互作用マップなどのスパースデータに対するイテレーション。QAOAスタイルの回路に便利です。
# - **`if` / `else`**（`Bit`条件）— 回路中間での測定結果に基づく条件分岐。
# - **`while`**（`Bit`条件）— 測定結果を条件とするランタイムループ。
#
# これらは通常のPythonループではなく、コンパイラがIRノードにトレースし、各バックエンドが適切に処理します（展開、ネイティブ制御フローなど）。
#
# 詳しくは[古典制御フローパターン](../tutorial/classical-flow-patterns)をご参照ください。

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
# ## リソース推定
#
# Qamomileはシンボリックなリソース推定を提供します。任意のqkernelに対して`estimate_resources()`を呼び出すと、qkernelを**実行することなく**量子ビット数やゲートの内訳を取得できます。これらは入力パラメータのシンボリック式として表現されるため、問題サイズに対するリソースのスケーリングの解析が容易です。具体的な値を代入して、特定の入力サイズに対する具体的な推定値を得ることも可能です。
#
# さらに、`@composite_gate`に`stub=True`を指定して定義したブラックボックスオラクルに対しても推定が可能です。オラクルの実装がなくても、指定したリソースコストとクエリ回数を含むリソース推定を行えます。
#
# 詳しくは[リソース推定](../tutorial/resource-estimation)および[再利用パターン: QKernelの合成とコンポジットゲート](../tutorial/reuse-patterns)をご参照ください。

# %%
est = ghz_state.estimate_resources()
print("qubits:", est.qubits)
print("total two-qubit gates:", est.gates.two_qubit)

# %%
# 特定のサイズで評価
print("two-qubit gates at n=100:", est.substitute(n=100).gates.two_qubit)

# %% [markdown]
# ## マルチ量子SDKトランスパイル
#
# `@qkernel`で回路を一度定義すれば、対応する任意の量子SDKにトランスパイルできます。Qamomile v0.10.0では、各バックエンドにプリセットのExecutorが用意されています。トランスパイルしたqkernelに対してプリセットのExecutorを指定して`sample()` / `run()`を呼び出すだけで実行ができます。もちろん、より細かい制御や実機へのアクセスのために、カスタムExecutorを作成することもできます。
#
# | バックエンド | モジュール | トランスパイラ | デフォルト実行環境 |
# |---------|--------|-----------|-----------|
# | Qiskit | `qamomile.qiskit` | `QiskitTranspiler` | ローカルシミュレータ; クラウドデバイスはqBraid経由 |
# | QURI Parts | `qamomile.quri_parts` | `QuriPartsTranspiler` | ローカルシミュレータ |
# | CUDA-Q | `qamomile.cudaq` | `CudaqTranspiler` | ローカルシミュレータ |
#
# Qiskitについては、[qBraid](https://docs.qbraid.com/)との連携により`qamomile.qbraid.QBraidExecutor`を使ってクラウド量子デバイス上で実行することも可能です。
#
# Qiskitは`pip install qamomile`にデフォルトで含まれています。その他のバックエンドはオプションの追加パッケージです:
#
# ```bash
# pip install "qamomile[cudaq-cu12]"   # CUDA-Q with CUDA 12.x (Linux)
# pip install "qamomile[cudaq-cu13]"   # CUDA-Q with CUDA 13.x (Linux / macOS ARM64)
# pip install "qamomile[quri_parts]"   # QURI Parts
# pip install "qamomile[qbraid]"       # qBraid integration for Qiskit
# ```
#
# 詳しくは[実行モデル: sample() vs run()](../tutorial/execution-models)および[qBraidサポート - QBraidExecutor](../collaboration/qbraid-executor)をご参照ください。

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
# ## 標準ライブラリ
#
# アルゴリズムを手軽に書けるよう、Qamomile v0.10.0では`qmc.stdlib`と`qmc.algorithm`モジュールに組み込みアルゴリズムを用意しています。現在量子フーリエ変換（QFT）や量子位相推定（QPE）などが含まれます。`qmc.stdlib`は広く使われる基本的なビルディングブロック、`qmc.algorithm`はより特化したアルゴリズムを収録していますが、この分類は厳密なものではなく、今後変更される可能性があります。
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
# - [チュートリアル](../tutorial) — 実践的なサンプルでQamomileを学べます。
# - [GitHubリポジトリ](https://github.com/Jij-Inc/Qamomile)
