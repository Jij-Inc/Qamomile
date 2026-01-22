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
# # Transpile & Execute
#
# このチュートリアルでは、QamomilenおTranspileの流れを説明します。
#
# ## 基本的な使い方
#
# 量子位相推定を例に、Qamomileのトランスパイルと実行の流れを説明します。
# まずは基本的な使い方を確認しましょう。

# %%
import qamomile.circuit as qmc


# %% [markdown]
# ### QPEの概要
#
# 量子位相推定は、ユニタリ演算子Uの固有値e^{2πiφ}の位相φを推定するアルゴリズムです。
#
# Qamomileでは、`qpe()`関数を使用してQPEを簡単に実装できます：
# - 入力: ターゲット状態、位相レジスタ、ユニタリ演算
# - 出力: `QFixed`（量子固定小数点数）
#
# `QFixed`を`measure()`で測定すると、自動的に`Float`にデコードされます。

# %%
import math

# 位相ゲートをユニタリとして定義
# P(θ)|1⟩ = e^{iθ}|1⟩ なので、|1⟩は固有値e^{iθ}を持つ固有状態
@qmc.qkernel
def phase_gate(q: qmc.Qubit, theta: float) -> qmc.Qubit:
    """Phase gate: P(θ)|1⟩ = e^{iθ}|1⟩"""
    return qmc.p(q, theta)


# 3ビット精度のQPE
@qmc.qkernel
def qpe_3bit(phase: float) -> qmc.Float:
    """3-bit Quantum Phase Estimation.

    Args:
        phase: The phase angle θ (the algorithm estimates θ/(2π))

    Returns:
        Float: Estimated phase as a fraction (0 to 1)
    """
    # 位相レジスタを作成（3ビット精度）
    phase_register = qmc.qubit_array(3, name="phase_reg")

    # ターゲット状態を|1⟩に初期化（P(θ)の固有状態）
    target = qmc.qubit(name="target")
    target = qmc.x(target)  # |0⟩ → |1⟩

    # QPEを適用
    phase_q: qmc.QFixed = qmc.qpe(target, phase_register, phase_gate, theta=phase)

    # QFixedを測定してFloatに変換
    return qmc.measure(phase_q)

# %%
# Transpile and Execute
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

test_phase = math.pi / 2  # θ = π/2, expected output ≈ 0.25 (since θ/(2π) = 0.25)
executable = transpiler.transpile(qpe_3bit, bindings={"phase": test_phase})

executor = transpiler.executor()
job = executable.sample(executor)
result = job.result()

for value, count in result.results:
    print(f"Measured value: {value}, Count: {count}")

# %% [markdown]
# bitstringではなくてFloatとして測定されていることがわかります。
# 
# ## Inline Pass
# トランスパイルの各ステップを詳しく見ていきましょう。
# まずは`Inline`パスです。これは、すべての`CallBlockOperation`をインライン展開します。
# 先ほどのQPEの例だとインライン展開するものがないので、別の例を見てみましょう。

# %%
@qmc.qkernel
def add_one(q: qmc.Qubit) -> qmc.Qubit:
    """Add one to a qubit (|0⟩ → |1⟩, |1⟩ → |0⟩)"""
    return qmc.x(q)

@qmc.qkernel
def add_two(q: qmc.Qubit) -> qmc.Qubit:
    """Add two to a qubit by calling add_one twice"""
    q = add_one(q)
    q = add_one(q)
    return q

@qmc.qkernel
def add_three(q: qmc.Qubit) -> qmc.Qubit:
    """Add three to a qubit by calling add_two and add_one"""
    q = add_two(q)
    q = add_one(q)
    return q

# %# [markdown]
# これらのカーネルをインライン展開してみましょう。

# %%
from qamomile.circuit.ir.block import Block
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.gate import GateOperation

transpiler = QiskitTranspiler()

def print_block_operations(block: Block):
    for op in block.operations:
        print(op.__class__.__name__ + ":", end="")
        if isinstance(op, CallBlockOperation):
            print(op.operands[0].name)
        elif isinstance(op, GateOperation):
            print(op.gate_type)
        else:
            print("")

# インライン展開前
block = transpiler.to_block(add_three)
print_block_operations(block)

# %% [markdown]
# `CallBlockOperation`がただ2回の呼ばれただけであることがわかります。
# では、インライン展開を実行してみましょう。

# %%
inlined_block = transpiler.inline(block)
print_block_operations(inlined_block)

# %% [markdown]
# `add_three`が`add_two`と`add_one`の中身、つまり`X`ゲート3回に展開されていることがわかります。

# %% [markdown]
# ## Analyze Pass and Separate Pass
# 次に`Analyze`パスです。これは、依存関係の解析と検証を行います。特に計算パスに対して変更は行われません。
# その次に`Separate`パスです。これは、量子セグメントと古典セグメントに分離します。
# これらのパスをQPEの例で見てみましょう。

# %%
block = transpiler.to_block(qpe_3bit)
inlined_block = transpiler.inline(block)
analyzed_block = transpiler.analyze(inlined_block)
separated_program = transpiler.separate(analyzed_block)


# %%
for i, segment in enumerate(separated_program.segments):
    print(f"Segment {i}: {segment.kind.name}")
    for op in segment.operations:
        print(" ", op.__class__.__name__)

# %% [markdown]
# 量子操作（ゲート、測定など）は`QUANTUM`セグメントに、古典操作（デコードなど）は`CLASSICAL`セグメントに分離されます。
#
# `boundaries`は量子と古典の境界（主に測定）を追跡します：

# %%
print(f"Boundaries: {len(separated_program.boundaries)}")
for boundary in separated_program.boundaries:
    print(f"  {boundary.operation.__class__.__name__}: segment {boundary.source_segment_index} → {boundary.target_segment_index}")

# %% [markdown]
# ## Emit Pass
#
# 最後に`Emit`パスです。このパスでは、分離されたプログラムをバックエンド固有のコードに変換します。
#
# ### 量子セグメントのEmit
# QUANTUMセグメントの操作は、バックエンド固有の量子回路にemitされます。
# Qiskitの場合、`QuantumCircuit`オブジェクトが生成されます。
#
# ### 古典セグメントの後処理
# CLASSICALセグメントは、測定結果に対する後処理として追加されます。
# 例えば、QPEの`QFixed`を測定すると：
# 1. QUANTUMセグメント: 各qubitの測定 → 生のビット列
# 2. CLASSICALセグメント: ビット列をFloatにデコード
#
# これにより、ユーザーは生のビット列ではなく`Float`を直接受け取ることができます。

# %%
executable = transpiler.emit(separated_program, bindings={"phase": test_phase})

# 量子回路の確認
print("=== 量子回路 ===")
circuit = executable.get_first_circuit()
print(circuit.draw(output="text"))

# 古典処理の確認
print("\n=== 古典後処理 ===")
print(f"Total segments: {len(separated_program.segments)}")
for i, segment in enumerate(separated_program.segments):
    print(f"Segment {i}: {segment.kind.name}")
    if segment.kind.name == "CLASSICAL":
        for op in segment.operations:
            print(f"  {op.__class__.__name__}")

# %% [markdown]
# 上記のように、QPEの実行では：
#
# 1. **QUANTUMセグメント** → Qiskit `QuantumCircuit`にemit
#    - Hゲート、制御位相ゲート、逆QFT
#    - 3つの`MeasureOperation`（各qubitの測定）
#
# 2. **CLASSICALセグメント** → 測定後の古典処理
#    - `DecodeQFixedOperation`がビット列をFloatにデコード
#
# これにより、最初の例で見たように`Measured value: 0.25`という`Float`値が得られます。
# ユーザーは生のビット列を意識せずに、高レベルな型（`QFixed` → `Float`）で結果を受け取れます。


