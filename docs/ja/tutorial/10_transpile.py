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
# # トランスパイラの内部：@qkernelから実行可能プログラムへ
#
# このチュートリアルでは、Qamomileのトランスパイラパイプラインを順を追って解説します。
# これまでのチュートリアルでは `transpiler.transpile()` を一括で呼び出していましたが、
# ここでは内部で何が起きているのかを詳しく見ていきます。各パスが `@qkernel` の
# Python関数をバックエンド固有の実行可能プログラムへどのように変換するかを追います。
#
# ## このチュートリアルで学ぶこと
# - トランスパイラパイプラインの全体像と各パスの役割
# - `draw()` による回路の異なる詳細度での可視化
# - カーネル呼び出しがフラットな命令列にインライン展開される仕組み
# - 検証、定数畳み込み、依存関係解析の動作
# - プログラムが量子セグメントと古典セグメントに分離される仕組み
# - emitパスによるバックエンド固有コードの生成
# - `TranspilerConfig` による合成ゲートの分解戦略の制御

# %% [markdown]
# ## 1. 概要
#
# `@qkernel` を書いて `transpiler.transpile()` を呼び出すと、Qamomileは
# 以下のパスからなるパイプラインを実行します：
#
# ```
# @qkernel
#     |  to_block
#     v
#   Block (HIERARCHICAL)
#     |  substitute (オプション — TranspilerConfigの戦略ルールを適用)
#     v
#   Block (HIERARCHICAL, 戦略適用済み)
#     |  inline
#     v
#   Block (AFFINE)
#     |  affine_validate
#     v
#   Block (検証済み)
#     |  constant_fold
#     v
#   Block (定数評価済み)
#     |  analyze
#     v
#   Block (ANALYZED)
#     |  separate
#     v
#   SeparatedProgram (classical_prep -> quantum -> classical_post)
#     |  emit
#     v
#   ExecutableProgram (バックエンド固有の回路 + 後処理)
# ```
#
# 各パスは中間表現（IR）をより洗練された形式に変換します。`substitute` パスは
# オプションであり、`TranspilerConfig` が合成ゲートの分解戦略ルールを指定した
# 場合にのみ実行されます。重要なポイントは、量子操作と古典操作が最初は単一の
# ブロック内に混在しており、パイプラインが段階的に検証・簡略化し、最終的に
# 分離することで、量子部分を実デバイスに送り、古典的な後処理をCPU上で
# 実行できるようにするということです。
#
# セクション2〜6では**1つの回路**を一貫して使い、最初から最後まで各変換を
# 追跡します。その後セクション7でQPEを完全なエンドツーエンドの例として示します。

# %% [markdown]
# ## 2. Blockの構築と描画
#
# 最初のステップは `@qkernel` を `Block`（Qamomileの中間表現（IR））に
# 変換することです。このチュートリアル全体で使用する回路を定義しましょう。

# %%
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler


@qmc.qkernel
def flip(q: qmc.Qubit) -> qmc.Qubit:
    """Apply X gate."""
    return qmc.x(q)


@qmc.qkernel
def prepare(q: qmc.Qubit) -> qmc.Qubit:
    """Flip then apply H."""
    q = flip(q)
    q = qmc.h(q)
    return q


@qmc.qkernel
def my_circuit(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    """Prepare, rotate, entangle, and measure."""
    doubled_angle = theta * 2
    qs = qmc.qubit_array(n, name="qs")
    qs[0] = prepare(qs[0])
    qs[0] = qmc.rz(qs[0], doubled_angle)
    qs[0], qs[1] = qmc.cx(qs[0], qs[1])
    return qmc.measure(qs)


# %% [markdown]
# この回路は3層のネスト構造を持っています：`my_circuit` が `prepare` を呼び、
# `prepare` が `flip` を呼びます。また、量子ビットの割り当て前に `theta * 2` を
# 計算します。この古典的な算術演算は `BinOp` 操作として現れ、後の定数畳み込みや
# セグメント分離パスで重要な役割を果たします。
#
# `draw()` メソッドを使うと、回路を異なる詳細度で可視化できます。

# %%
# Default: sub-kernel calls shown as boxes
my_circuit.draw(n=2)

# %%
# inline=True with inline_depth=1: expand one level only.
# prepare is expanded (showing flip as a box + H), but flip stays as a box.
my_circuit.draw(n=2, inline=True, inline_depth=1)

# %%
# inline=True (unlimited depth): fully expanded to primitive gates.
my_circuit.draw(n=2, inline=True)

# %% [markdown]
# `inline_depth` パラメータは、何階層のネストを展開するかを制御します：
# - `None`（`inline=True` 時のデフォルト）— 無制限、すべてを展開
# - `0` — インライン展開なし（`inline=False` と同じ）
# - `1` — トップレベルの呼び出しのみ展開

# %% [markdown]
# ### Blockへの変換
#
# パイプラインの最初のパスは `to_block()` です。`@qkernel` 関数をトレースして
# `Block`（後続のすべてのパスが操作するIR）を生成します。`bindings` を渡して
# `n`（量子ビット配列のサイズ）などのパラメータに具体的な値を与えます。

# %%
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.gate import GateOperation


def print_block_operations(block: Block):
    """Print all operations in a block."""
    for op in block.operations:
        print(op.__class__.__name__ + ":", end="")
        if isinstance(op, CallBlockOperation):
            print(op.operands[0].name)
        elif isinstance(op, GateOperation):
            print(op.gate_type)
        else:
            print("")


# %%
transpiler = QiskitTranspiler()
block = transpiler.to_block(my_circuit, bindings={"n": 2})
print_block_operations(block)

# %% [markdown]
# このブロックは**階層的**な形式です。`prepare` の `CallBlockOperation` は
# まだ展開されていません（ネストされた呼び出しを含んでいる可能性があります）。
# `QInitOperation` は量子ビットの割り当てステップであり、トランスパイラに
# 物理量子ビットをいくつ作成するかを指示します。

# %% [markdown]
# ## 3. インラインパス
#
# **inline**パスは、階層的なブロックを受け取り、すべての `CallBlockOperation`
# （他のqkernelへの呼び出し）をフラットで線形なプリミティブ操作の列に展開します。
# インライン展開後は、ネストされた呼び出しは残りません。

# %%
print("=== Before inlining ===")
print_block_operations(block)

# %%
inlined = transpiler.inline(block)
print("=== After inlining ===")
print_block_operations(inlined)

# %% [markdown]
# `prepare` の `CallBlockOperation` が完全に展開されました：
# `prepare` は `flip`（Xゲート）を呼び、次にHゲートを適用していました。
# どちらもフラットなブロック内で `GateOperation` として見えるようになりました。
# `theta * 2` の `BinOp` も見えています。これは古典的な操作であり、
# 次のステップの `constant_fold` で処理されます。

# %% [markdown]
# ## 4. 検証、畳み込み、解析 — ステップバイステップ
#
# インライン展開後、トランスパイラは正確性の検証とセグメント分離の準備のために
# 3つのパスを実行します。それぞれを見ていきましょう。
#
# ### 4a. `affine_validate` — 複製不可能性のセーフティネット
#
# `affine_validate` パスは、フロントエンドのチェックをすり抜けた可能性のある
# 線形型違反を検出する**セーフティネット**です。実際には、ほとんどの違反
# （消費済み量子ビットの再利用やエイリアシングなど）は、フロントエンドの
# ハンドルシステムによって**トレース時**に検出されます。
# [チュートリアル02](02_type_system.ipynb)の線形型エラーを参照してください。
# このパスはIRレベルでの追加的な検証レイヤーを提供します。

# %%
validated = transpiler.affine_validate(inlined)
print("=== After affine_validate ===")
print_block_operations(validated)

# %% [markdown]
# 操作は変更されていません。`my_circuit` の各量子ビットが操作ごとに正確に
# 1回だけ使用されているため、検証に合格しました。同じ量子ビットを2回使おうと
# した場合（例：`qmc.cx(q, q)`）、フロントエンドがトレース時（`draw()` や
# `to_block()` の実行中）に `QubitAliasError` を発生させ、回路がこのパスに
# 到達する前にエラーになります。
#
# ### 4b. `constant_fold` — 既知の定数の評価
#
# `constant_fold` パスは、すべてのオペランドが既知の定数または束縛済み
# パラメータである場合に算術式を評価します。この回路には
# `doubled_angle = theta * 2` が含まれており、`BinOp` 操作として現れます。
# `theta = 0.5` を束縛すると、パスは `0.5 * 2 = 1.0` を評価し、`BinOp` を
# 除去します。

# %%
print("=== Before constant_fold ===")
print_block_operations(validated)

# %%
folded = transpiler.constant_fold(validated, bindings={"theta": 0.5})
print("=== After constant_fold ===")
print_block_operations(folded)

# %% [markdown]
# `theta * 2` の `BinOp` が評価されて除去されました。Rzゲートは具体的な値
# `1.0` を直接使用するようになりました。`theta` が未束縛のままだった場合、
# `BinOp` は残存し、分離時に `classical_prep` セグメントに配置されます。
# これはセクション5で実演します。
#
# ### 4c. `analyze` — 依存関係グラフとI/O検証
#
# `analyze` パスは依存関係グラフを構築し、以下を検証します：
# - ブロックのすべての入力と出力が古典型であること（トップレベルでの量子I/Oは
#   不可）
# - 量子操作が測定結果に依存していないこと（ミッドサーキット測定のサポートが
#   必要になるため、まだ利用不可）

# %%
analyzed = transpiler.analyze(folded)
print("=== After analyze ===")
print_block_operations(analyzed)

# %% [markdown]
# 操作は再び変更されていません。解析により以下が確認されました：
# - 入力 `n`（UInt）と `theta`（Float）は古典型
# - 出力 `Vector[Bit]` は古典型
# - 量子ゲートが測定結果に依存していない
#
# ブロックは依存関係グラフが付与された **ANALYZED** 状態になりました。
# 検証に失敗した場合は、問題のある操作を指し示す明確なエラーメッセージが
# 表示されます。

# %% [markdown]
# ## 5. セグメント分離
#
# **separate** パスは、解析済みのブロックを個別のセグメントに分割します：
#
# - **`classical_prep`**: 量子回路の*前*に実行される古典的な操作
#   （例：トランスパイル時に畳み込めなかったパラメータの事前計算）。
#   すべてのパラメータが束縛されている場合、`constant_fold` がそれらを解決
#   するため、このセグメントは通常 `None` になります。
#
# - **`quantum`**: 量子操作（ゲート、測定）。
#
# - **`classical_post`**: 測定*後*に実行される古典的な操作
#   （例：`QFixed` のビット列を `Float` 値に変換する `DecodeQFixedOperation`）。
#   このセグメントは、ユーザーが生のビット列ではなく `Float` のような
#   高レベルの結果を受け取れるようにする型変換を処理します。
#
# この3部構成は、現在の量子ハードウェアの現実を反映しています：
# 古典的な前処理はCPU上で行われ、量子回路はQPU上で実行され、
# 古典的な後処理は再びCPU上で行われます。
#
# ### 5a. すべてのパラメータが束縛されている場合
#
# すべてのパラメータが束縛されている場合、`constant_fold` がすべての算術を
# 解決するため、古典的な前処理は不要です。

# %%
separated = transpiler.separate(analyzed)

print(f"classical_prep:  {separated.classical_prep}")
print(f"quantum:         {separated.quantum.kind.name}")
print(f"classical_post:  {separated.classical_post}")
print(f"boundaries:      {len(separated.boundaries)}")
print()
print("=== Quantum segment operations ===")
for op in separated.quantum.operations:
    print(f"  {op.__class__.__name__}")

# %% [markdown]
# 期待通りの結果です：
# - `classical_prep` なし — `theta * 2` は `constant_fold` によって `1.0` に畳み込まれました。
# - ゲートと測定操作を含む1つの **QUANTUM** セグメント。
# - `classical_post` なし — `Bit` の測定にはデコードが不要です。
#
# ### 5b. ランタイムパラメータ — `classical_prep` の動作
#
# `theta` がトランスパイル時に束縛*されていない*場合はどうなるでしょうか？
# `theta * 2` の `BinOp` は `constant_fold` で解決できず、量子回路をQPUに
# 送る*前*にランタイムで計算する必要があります。`separate` パスはこれを
# 検出し、`BinOp` を `classical_prep` セグメントに配置します。

# %%
# Re-run the pipeline without binding theta
block_param = transpiler.to_block(my_circuit, bindings={"n": 2})
inlined_param = transpiler.inline(block_param)
validated_param = transpiler.affine_validate(inlined_param)
folded_param = transpiler.constant_fold(validated_param, bindings={})  # theta unbound
analyzed_param = transpiler.analyze(folded_param)
separated_param = transpiler.separate(analyzed_param)

# Show all segments
segments_param = []
if separated_param.classical_prep:
    segments_param.append(("classical_prep", separated_param.classical_prep))
segments_param.append(("quantum", separated_param.quantum))
if separated_param.classical_post:
    segments_param.append(("classical_post", separated_param.classical_post))

for name, segment in segments_param:
    print(f"=== {name} ({segment.kind.name}) ===")
    for op in segment.operations:
        print(f"  {op.__class__.__name__}")

# %% [markdown]
# `classical_prep` が出現し、`theta * 2` の `BinOp` を含んでいます。
# この計算は実行時にCPU上で、量子回路がQPUに送信される直前に行われます。
#
# セクション7では、`QFixed → Float` 変換のための `DecodeQFixedOperation` を
# 含む `classical_post` を持つQPEを見ていきます。

# %% [markdown]
# ## 6. Emitパス
#
# 最後のパスは **emit** です。`SeparatedProgram` を受け取り、各セグメントを
# バックエンド固有のコードに変換します：
#
# - **量子セグメント**はバックエンド固有の回路（例：Qiskitの `QuantumCircuit`）
#   になります。
# - **古典セグメント**は、生の測定結果を高レベルの型に変換する後処理関数に
#   なります。
#
# 回路をemitして実行してみましょう。

# %%
executable = transpiler.emit(separated, bindings={"n": 2, "theta": 0.5})

circuit = executable.get_first_circuit()
print("=== Qiskit circuit ===")
circuit.draw(output="mpl")

# %%
# Execute and see measurement results
executor = transpiler.executor()
job = executable.sample(executor)
result = job.result()

print("\n=== Results ===")
for value, count in result.results:
    print(f"Measured: {value}, Count: {count}")

# %% [markdown]
# ゲートはQiskitの `QuantumCircuit` になります：X, H, Rz(1.0), CX、
# そして測定が続きます。`theta * 2` は `constant_fold` によってすでに
# 評価されているため、Rzゲートは元の `0.5` ではなく事前計算された値 `1.0` を
# 使用しています。結果は `Bit` 値として返されます — 後処理は不要です。
#
# `transpiler.transpile()` を使えば、パイプライン全体を一括で実行することも
# できます：
# ```python
# executable = transpiler.transpile(my_circuit, bindings={"n": 2, "theta": 0.5})
# ```
# 上記のステップバイステップのアプローチは同等ですが、各中間段階を
# 検査できるという利点があります。

# %% [markdown]
# ## 7. パイプライン全体 — QPE
#
# ここからは、より複雑な例である**量子位相推定（QPE）**でパイプライン全体の
# 動作を見てみましょう。`my_circuit` と異なり、QPEは `QFixed`（量子固定小数点数）
# を返し、測定後に自動的に `Float` にデコードされます。このデコードは
# `classical_post` セグメントで行われます。
#
# 標準ライブラリの `qpe()` は**通常の関数**であり、`CompositeGate` ではないことに
# 注意してください。ブロックに操作（IQFTの `CompositeGateOperation` を含む）を
# 出力します。標準ライブラリの `CompositeGate` はQFTとIQFTです。

# %%
import math


@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: qmc.Float) -> qmc.Qubit:
    """Phase gate: P(theta)|1> = e^{i*theta}|1>"""
    return qmc.p(q, theta)


@qmc.qkernel
def qpe_3bit(phase: qmc.Float) -> qmc.Float:
    """3-bit Quantum Phase Estimation."""
    phase_register = qmc.qubit_array(3, name="phase_reg")
    target = qmc.qubit(name="target")
    target = qmc.x(target)

    phase_q: qmc.QFixed = qmc.qpe(target, phase_register, phase_gate, theta=phase)

    return qmc.measure(phase_q)


# %%
# Transpile the entire QPE pipeline in one call
test_phase = math.pi / 2
executable_qpe = transpiler.transpile(qpe_3bit, bindings={"phase": test_phase})

# Show the Qiskit circuit
circuit_qpe = executable_qpe.get_first_circuit()
print("=== QPE circuit ===")
circuit_qpe.draw(output="mpl")

# %%
# Execute and see decoded Float results
job_qpe = executable_qpe.sample(executor)
result_qpe = job_qpe.result()

print("\n=== QPE Results ===")
for value, count in result_qpe.results:
    print(f"Measured value: {value}, Count: {count}")

# %% [markdown]
# 測定結果は生のビット列ではなく `Float` として返されます。
# `theta = pi/2` の場合、期待される位相は `theta / (2*pi) = 0.25` であり、
# QPEアルゴリズムはこの値を正しく推定しています。
#
# ### QPEセグメントの内部構造
#
# これを可能にしているセグメント構造を覗いてみましょう。

# %%
# Run the pipeline step-by-step to inspect segments
block_qpe = transpiler.to_block(qpe_3bit, bindings={"phase": test_phase})
inlined_qpe = transpiler.inline(block_qpe)
validated_qpe = transpiler.affine_validate(inlined_qpe)
folded_qpe = transpiler.constant_fold(validated_qpe, bindings={"phase": test_phase})
analyzed_qpe = transpiler.analyze(folded_qpe)
separated_qpe = transpiler.separate(analyzed_qpe)

# Show the segment structure
segments = []
if separated_qpe.classical_prep:
    segments.append(("classical_prep", separated_qpe.classical_prep))
segments.append(("quantum", separated_qpe.quantum))
if separated_qpe.classical_post:
    segments.append(("classical_post", separated_qpe.classical_post))

for name, segment in segments:
    print(f"=== {name} ({segment.kind.name}) ===")
    for op in segment.operations:
        print(f"  {op.__class__.__name__}")

# %% [markdown]
# QPEは2つのセグメントを生成します：
#
# 1. **quantum** — すべてのゲート（H、制御P、逆QFT）と測定。
#    QPEは位相レジスタ全体を1つの `MeasureVectorOperation` として測定します。
#
# 2. **classical_post** — 測定されたビット列を `Float` 値に変換する
#    `DecodeQFixedOperation`。これにより、QPEは生のビット列ではなく
#    `Float` を返すことができます。
#
# セクション2〜6の `my_circuit` と比較してみてください。`Bit` の結果には
# デコードが不要なため、`classical_post` がありませんでした。

# %% [markdown]
# ## 8. TranspilerConfigと戦略
#
# トランスパイラパイプラインは固定されたものではなく、**戦略**を設定することで
# 合成ゲートの分解方法を制御できます。
#
# 標準ライブラリの `CompositeGate`（QFTやIQFTなど）には、複数の分解戦略を
# 登録できます。例えば、QFTゲートは標準戦略（完全精度、O(n^2)ゲート）と
# 近似戦略（回転の打ち切り、O(n*k)ゲート）の両方をサポートしています。
#
# emitパスでどの戦略を使用するかは `TranspilerConfig` で制御します。

# %%
from qamomile.circuit.stdlib.qft import QFT
from qamomile.circuit.stdlib.qft_strategies import (
    ApproximateQFTStrategy,
    StandardQFTStrategy,
)

# Register strategies on the QFT class
QFT.register_strategy("standard", StandardQFTStrategy())
QFT.register_strategy("approximate_k2", ApproximateQFTStrategy(truncation_depth=2))


# Define a kernel that uses QFT and measures the result
@qmc.qkernel
def qft_and_measure() -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(4, name="q")
    for i in range(4):
        q[i] = qmc.h(q[i])
    q = qmc.qft(q)
    return qmc.measure(q)


# %%
# Transpile with the default (standard) strategy.
# use_native_composite=False forces Qamomile's own decomposition
# so that strategy selection takes effect.
transpiler_standard = QiskitTranspiler(use_native_composite=False)
circuit_standard = transpiler_standard.to_circuit(qft_and_measure)

print("=== Standard QFT (full precision) ===")
circuit_standard.draw(output="mpl")

# %%
# Transpile with the approximate strategy (truncation_depth=2).
# For 4 qubits, this skips the CP(pi/8) rotation whose exponent (3)
# exceeds the truncation depth (2).
from qamomile.circuit.transpiler.transpiler import TranspilerConfig

config = TranspilerConfig.with_strategies({"qft": "approximate_k2"})

transpiler_approx = QiskitTranspiler(use_native_composite=False)
transpiler_approx.set_config(config)

circuit_approx = transpiler_approx.to_circuit(qft_and_measure)

print("=== Approximate QFT (truncation_depth=2) ===")
circuit_approx.draw(output="mpl")

# %%
# Compare gate counts
print("=== Gate Count Comparison ===")
print(f"Standard QFT gates:    {circuit_standard.size()}")
print(f"Approximate QFT gates: {circuit_approx.size()}")

# %% [markdown]
# 近似戦略は、小角度の制御位相回転を打ち切ることでゲート数を削減します。
# 4量子ビットで `truncation_depth=2` の場合、指数（3）が深度を超える
# CP(pi/8) 回転がスキップされます。これは精度と効率のトレードオフであり、
# 厳密なQFTの精度が不要な場合に有用です。
#
# `TranspilerConfig.with_strategies()` はゲート名を戦略名にマッピングする
# 設定を作成します。トランスパイラはemitパス中にこのマッピングを使用して、
# 各合成ゲートに適切な分解を選択します。

# %% [markdown]
# ## 9. まとめ
#
# このチュートリアルでは、`@qkernel` 関数から `ExecutableProgram` までの
# 完全なパスを追跡しました。各パスの要約を以下に示します：
#
# | パス | 入力 | 出力 | 目的 |
# |------|------|------|------|
# | `to_block()` | `QKernel` | Block (HIERARCHICAL) | Python関数をIRに変換 |
# | `substitute()` | Block | Block | `TranspilerConfig` の戦略ルールを適用（オプション） |
# | `inline()` | Block | Block (AFFINE) | すべてのカーネル呼び出しを展開 |
# | `affine_validate()` | Block | Block (検証済み) | 複製不可能性違反のセーフティネット |
# | `constant_fold()` | Block | Block (畳み込み済み) | 既知の定数を評価 |
# | `analyze()` | Block | Block (ANALYZED) | 依存関係グラフの構築、I/O検証 |
# | `separate()` | Block | SeparatedProgram | 古典/量子セグメントに分割 |
# | `emit()` | SeparatedProgram | ExecutableProgram | バックエンド固有コードを生成 |
#
# 主なポイント：
#
# - **`transpile()` は魔法ではない** — 明確に定義されたパスの列であり、
#   デバッグのために個別に呼び出すことができます。
# - **`draw()` の `inline` と `inline_depth`** — トランスパイラパイプラインに
#   入る前に、回路を異なる詳細度で可視化できます。
# - **セグメント分離**はプログラムを `classical_prep`（`theta` が未束縛の場合の
#   ランタイムパラメータ計算）、`quantum`（ゲートと測定）、`classical_post`
#   （QPEで見たQFixed → Floatデコードなど）に分割します。
# - **`TranspilerConfig`** はemitパス中にQFTやIQFTなどの合成ゲートに
#   使用する分解戦略を制御します。
# - **QPE** は `qpe()` を使用します。これは通常の関数であり、CompositeGate
#   ではありません。標準ライブラリのCompositeGateはQFTとIQFTであり、
#   プラグ可能な戦略をサポートしています。
#
# ### 次のステップ
#
# - [カスタムExecutor](11_custom_executor.ipynb): クラウド量子ハードウェアでの回路実行
# - [リソース推定](09_resource_estimation.ipynb): ゲート数と回路深度の分析
# - [QAOA](../optimization/qaoa.ipynb): 変分回路による最適化

# %% [markdown]
# ## このチュートリアルで学んだこと
#
# - **トランスパイラパイプラインの全体像と各パスの役割** — 8つのパス（`to_block` → `substitute` → `inline` → `affine_validate` → `constant_fold` → `analyze` → `separate` → `emit`）が `@qkernel` をバックエンド固有のコードに変換します。`substitute` パスはオプションであり、`TranspilerConfig` の戦略ルールを適用します。
# - **`draw()` による回路の可視化** — `inline=True` でサブカーネルの呼び出しを展開し、`inline_depth` で展開するネスト階層数を制御します。
# - **カーネル呼び出しがフラットな命令列にインライン展開される仕組み** — `inline()` パスがすべての `CallBlockOperation` を再帰的に展開し、単一の線形ブロックを生成します。
# - **検証、定数畳み込み、依存関係解析の動作** — `affine_validate` は複製不可能性違反のセーフティネットです（ほとんどはフロントエンドによりトレース時に検出されます）。`constant_fold` は `theta * 2` のような `BinOp` 式を具体的な値に評価し、`analyze` はI/O検証のための依存関係グラフを構築します。
# - **プログラムが量子セグメントと古典セグメントに分離される仕組み** — `separate()` はブロックを `classical_prep`（値が未束縛の場合のランタイムパラメータ計算）、`quantum`、`classical_post` セグメントに分割します。QPEは `DecodeQFixedOperation` によるQFixed → Float変換で `classical_post` を実演しています。
# - **emitパスによるバックエンド固有コードの生成** — `emit()` は量子セグメントを走査し、対象バックエンド向けのネイティブな回路オブジェクト（例：Qiskitの `QuantumCircuit`）を生成します。
# - **`TranspilerConfig` による合成ゲート分解戦略の制御** — `TranspilerConfig.with_strategies()` がゲート名を戦略名にマッピングし、emitパス中に分解を選択します。
