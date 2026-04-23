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
# # コンパイルとトランスパイル: 内部の仕組み
#
# このチュートリアルではQamomileの`@qkernel`がどのような処理フローを経て、Python関数から量子回路へと変換されるのかを、コンパイラの内部の視点から見ていきます。ユーザーが見るのは`@qkernel`を書き、`transpiler.transpile(...)`を呼び、executableを受け取る、という流れです。この章ではそのブラックボックスを開きます。
#
# 対象読者は**コントリビュータ**、つまり以下のようなニーズを持つ方です:
#
# - トレーシングからemissionの間のどこかで失敗するカーネルをデバッグしたい
# - カスタムコンパイラパスを書きたい
# - 新しいバックエンド（別の量子SDKなど）を追加したい
# - `transpile()`が実際に何をしているのかを単純に理解したい
#
# 小さな`@qkernel`を`Transpiler`のステップ実行用公開APIでパイプラインを1段ずつ通し、各ステップで中間表現を確認します。そして同じプランを2つのバックエンド（QiskitとQURI Parts）がどのように異なる回路へ変換するかを比較します。

# %%
# 最新のQamomileをpipからインストールします！
# # !pip install qamomile
# # or
# # !uv add qamomile

# %% [markdown]
# ## 1. パイプラインの全体像
#
# `Transpiler.transpile()`は`qamomile/circuit/transpiler/transpiler.py`に、10個のパスの合成として記述されています。各ステージは**フロントエンド → インライン化 → 解析 → emission**という4つの帯に分かれ、**`BlockKind`**の遷移で区切られます:
#
# ```
# QKernel
#    │  to_block                    (トレーシング: Python AST → IR)
#    ▼
# Block [HIERARCHICAL]
#    │  substitute                  (ルールベースの置換、オプション)
#    │  resolve_parameter_shapes    (Vectorのshape次元を具体化)
#    │  inline                      (CallBlockOperationsを展開)
#    ▼
# Block [AFFINE]
#    │  unroll_recursion            (inline ↔ partial_evalの反復)
#    │  affine_validate             (アフィン型のセーフティネット)
#    │  partial_eval                (定数畳み込み + コンパイル時if)
#    │  analyze                     (依存グラフ + I/O検証)
#    ▼
# Block [ANALYZED]
#    │  validate_symbolic_shapes    (未解決のVector次元を拒否)
#    │  plan                        (C→Q→Cにセグメント化)
#    ▼
# ProgramPlan
#    │  emit                        (バックエンド固有のコード生成)
#    ▼
# ExecutableProgram[T]
# ```
#
# どのパスも冪等で、`Transpiler`の公開メソッドとして公開されています。そのため1つずつ実行して、間で`Block`を出力できます。これがQamomileで最も役立つデバッグ手法です。

# %% [markdown]
# ## 2. IRの用語
#
# パスを実行する前に、出力することになる対象に名前を付けておきましょう。
#
# ### `Block`
# 
# `Block` (`qamomile.circuit.ir.block`) はパイプラインを流れるコンテナです。以下を保持します:
#
# - `operations`: `Operation`インスタンスの順序付きリスト
# - `input_values` / `output_values`: カーネルのシグネチャに対応するSSAの`Value`
# - `parameters`: 未バインドのパラメータ名から`Value`への辞書
# - `kind`: どの不変条件が現在成立しているかを示す`BlockKind`タグ（`TRACED`、`HIERARCHICAL`、`AFFINE`、`ANALYZED`）
#
#
# ### `BlockKind`
#
# `BlockKind`はパイプラインのステートマシンです。各パスは`kind`に対する事前条件を持ち、成功時に`kind`を進めます。進行は単調です:
#
# ```
# TRACED  →  HIERARCHICAL  →  AFFINE  →  ANALYZED
# ```
#
# #### 余談: なぜ「AFFINE」と呼ぶのか
#
# プログラミング言語の型理論では、ある値を何回使ってよいかで型を3種類に分けます:
#
# | 区分 | 使用回数 | 例 |
# |------|---------|------|
# | Unrestricted（通常の型） | 0回以上、何度でも | Pythonの`int`、古典ビット。コピー可能な値。 |
# | **Affine** | **高々1回（使わなくてもよい）** | **量子ビット** |
# | Linear | ちょうど1回（捨てるのも禁止） | 「必ず消費せよ」と強制したい値 |
#
# 量子の場合、**no-cloning theorem（量子状態は複製できない）**によって「同じ値を2回使う」のは物理的に不可能です。つまり`q`を`qmc.h(q)`で消費したら、その古い`q`はもう使えず、新しい版`q'`が生まれる — これがまさに**affine**（高々1回）の定義に一致します。
#
# ちなみに「使わずに捨てる」ことは許容されます（`q = qmc.h(q)`のあと`q`を返さず捨ててもエラーにはならない。最終的には測定で消費されるのが健全ですが、型制約としては"linear"ほど厳しくない）。そのためQamomileは"linear"ではなく"affine"という呼び方を選んでいます。
#
# `BlockKind.AFFINE`は、このaffine型不変条件（「各量子値は高々1回だけ使われている」）が検証可能な状態でブロックが仕上がっていることを意味します。実際の検証は`AffineValidationPass`が担当し、違反すると`AffineTypeError`が送出されます。
#
# ### `Value` と`Operation`
# 
# **`Value`** (`qamomile.circuit.ir.value`) はSSAスタイルの型付き値です。`Qubit`に限らず`Float`、`UInt`、`Bit`などすべての値が`Value`として表現されます。ゲート適用や古典演算でその値が更新されるたびに、`Value.next_version()`が新しい`Value`を生成します。このとき`version`と`uuid`は新しくなりますが、`logical_id`と型・メタデータは保たれます。
#
# `logical_id`は「SSAのバージョンをまたいで**同じ論理的な変数**を指す」ための安定した識別子です。たとえば`q = qmc.h(q)`で新しい`Value`が作られても、元の`q`と同じ`logical_id`を持ちます。これは物理量子ビットへのマッピングではなく、IR上で「同じ変数の別バージョン」を結びつけるためのもので、`Float`パラメータや`Bit`などにも同じ仕組みが使われます（バックエンドの物理量子ビット割り当ては後段の`emit`で`ResourceAllocator`が決めます）。
#
# メタデータで値をパラメータ（`with_parameter("theta")`）や定数（`with_const(2.0)`）としてタグ付けできます。
#
# `Operation`はオペレーション階層の基底クラスです。サブクラスには以下があります:
#
# | サブクラス | 用途 | ファイル |
# |----------|---------|------|
# | `GateOperation` | `H`、`RX`、`CX`、… | `ir/operation/gate.py` |
# | `MeasureOperation` | 測定 | `ir/operation/measurement.py` |
# | `ForOperation`、`IfOperation`、`WhileOperation` | 制御フロー | `ir/operation/control_flow.py` |
# | `CallBlockOperation` | 別の`Block`の呼び出し（`inline`で除去） | `ir/operation/call_block_ops.py` |
#
# 制御フロー系のOperationはすべて`HasNestedOps`プロトコル（`nested_op_lists()` / `rebuild_nested()`）を実装しているので、パスは各Operationの型を特別扱いせず、ループや分岐の本体へ統一的に踏み込めます。
#
# どの`Operation`も`operation_kind`（`QUANTUM`、`CLASSICAL`、`HYBRID`、`CONTROL`）を報告します。これは`plan`ステージがブロックを古典・量子・期待値のステップへセグメント化する際に使います。

# %%
import qamomile.circuit as qmc
from qamomile.circuit.ir import pretty_print_block
from qamomile.circuit.ir.block import BlockKind
from qamomile.circuit.ir.operation.call_block_ops import CallBlockOperation
from qamomile.circuit.ir.operation.control_flow import ForOperation
from qamomile.qiskit import QiskitTranspiler

transpiler = QiskitTranspiler()

# %% [markdown]
# ## 3. 題材となるカーネル
#
# 出力できる程度に小さく、複数のステージを動かせる程度に豊かなカーネルが必要です:
#
# - ヘルパーとなる`@qkernel`（**inline**を動かすため）
# - `qmc.range(n)`を駆動する`UInt`パラメータ（**partial_eval**を動かすため）
# - 未バインドのまま残す`Float`パラメータ（**emit**のパラメータ処理を動かすため）


# %%
@qmc.qkernel
def entangle_pair(q0: qmc.Qubit, q1: qmc.Qubit) -> tuple[qmc.Qubit, qmc.Qubit]:
    """ヘルパーサブルーチン。呼び出し元にインライン展開されます。"""
    q0 = qmc.h(q0)
    q0, q1 = qmc.cx(q0, q1)
    return q0, q1


@qmc.qkernel
def demo_kernel(n: qmc.UInt, theta: qmc.Float) -> qmc.Vector[qmc.Bit]:
    q = qmc.qubit_array(n, name="q")

    q[0] = qmc.h(q[0])
    for i in qmc.range(n - 1):
        q[i], q[i + 1] = entangle_pair(q[i], q[i + 1])
        q[i + 1] = qmc.rz(q[i + 1], theta)

    return qmc.measure(q)


# %% [markdown]
# `n=3`をコンパイル時にバインドし、`theta`はバックエンドパラメータとして保持してトランスパイルします。


# %%
def summarise(block):
    """Blockのコンパクトなサマリー。各パスの後で呼び出します。"""
    by_kind = {}
    for op in block.operations:
        by_kind[type(op).__name__] = by_kind.get(type(op).__name__, 0) + 1
    return (
        f"kind={block.kind.name:13s} "
        f"ops={len(block.operations):>2d} "
        f"breakdown={by_kind}"
    )


# %% [markdown]
# 1行サマリーでは十分でないときのために、`qamomile.circuit.ir.pretty_print_block`が`Block`のMLIR風テキストダンプを返します。各パスの前後で**何がどう変わったか**を目で確認するには、こちらが最速です。`depth`引数で`CallBlockOperation`の展開深さを制御できるので、たとえば`depth=1`なら「`inline`が実行したら何が起こるか」を先取りして眺められます。

# %% [markdown]
# ## 4. ステージごとのウォークスルー
#
# 各パスを手動で実行していきます。上の`summarise`ヘルパーがステージごとに1行ずつ出力するので、`BlockKind`とOperationの内訳を比較しやすくなります。`pretty_print_block`はステージの詳細を追いたいタイミングで差し込んでください。
#
# ### 4.1 `to_block` — Python関数のトレーシング
#
# `to_block`はデコレート済み関数をトレーサコンテキスト下で実行します。`qmc.h(...)`、`qmc.range(...)`、`entangle_pair(...)`の各呼び出しは`Operation`としてBlockに記録されます。他の`@qkernel`への呼び出しは`CallBlockOperation`になり、本体は**まだ**インライン展開されません。

# %%
bindings = {"n": 3}
parameters = ["theta"]

block = transpiler.to_block(demo_kernel, bindings=bindings, parameters=parameters)
pretty_print_block(block)
print("after to_block:   ", summarise(block))
print("parameters:       ", list(block.parameters))
print("CallBlockOps:     ", sum(1 for op in block.operations if isinstance(op, CallBlockOperation)))
# 注意: `CallBlockOperation`は`ForOperation`の本体内部にも存在しうるので、
# 必ずしもトップレベルのリストにあるとは限りません。

# %% [markdown]
# ブロックの中身を実際にテキストで眺めてみましょう。`pretty_print_block`は`Block`をMLIR風のインデント付きテキストへ整形します。`for`本体に`call entangle_pair(...)`がまだ生きていることが確認できます。

# %%
print(pretty_print_block(block))

# %% [markdown]
# `depth=1`を付けると、`CallBlockOperation`の先が呼び出し行のブレース内にインライン展開された形で表示されます。これは「`inline`を1回通したらどうなるか」を先読みしているのと同じ見た目で、次の節の内容を予習できます。

# %%
print(pretty_print_block(block, depth=1))

# %% [markdown]
# ブロックは`HIERARCHICAL`です。他のブロックへの呼び出しや複合ゲートをまだ含む可能性があります。`block.parameters`は渡した引数`parameters=["theta"]`を反映しています。`parameters`に**ない**入力は、`bindings`でバインドする（`n`のように）か、トレース時のPythonコードで消費されなければなりません。
#
# ### 4.2 `inline` — ネストしたブロック呼び出しの平坦化
#
# `inline`はすべての`CallBlockOperation`を対象ブロックのOperationで置き換え、結果がwell-formedであり続けるようSSA値を置換します。`CallBlockOperation`が残らなくなるとブロックは`AFFINE`へ遷移します。


# %%
def count_calls(ops):
    total = 0
    for op in ops:
        if isinstance(op, CallBlockOperation):
            total += 1
        # ループ内の呼び出しも数えるため、ネストした制御フロー本体を再帰的に辿ります。
        for child in getattr(op, "nested_op_lists", lambda: [])():
            total += count_calls(child)
    return total


block = transpiler.inline(block)
print("after inline:     ", summarise(block))
print("CallBlockOps (deep):", count_calls(block.operations))
print("is_affine:        ", block.is_affine())

# %% [markdown]
# 再度`pretty_print_block`で眺めると、`call entangle_pair(...)`が消え、`h`/`cx`が直接`for`本体に並んでいることが確認できます。ブロックの`kind`は`AFFINE`へ進みました。

# %%
print(pretty_print_block(block))

# %% [markdown]
# `inline`は**ループをアンロールしない**ことに注目してください。`ForOperation`の本体には、元の`entangle_pair`本体内にあった`for`の各反復に対応する`GateOperation`が1つずつ入っています。インライン化は制御フローを保ちます。アンロールは次です。
#
# ### 4.3 `partial_eval` — 定数畳み込みとコンパイル時`if`の除去
#
# `partial_eval`は2つのサブパスから構成されます:
#
# 1. **`ConstantFoldingPass`** — オペランドがすべて定数（またはバインド済みパラメータ）の`BinOp`/`CompOp`を具体値へ畳み込みます。`n=3`をバインドしたので、`qmc.range(n - 1)`の`n - 1`は`2`に畳み込まれ、`ForOperation`の境界はリテラル値になります。
# 2. **`CompileTimeIfLoweringPass`** — 条件がコンパイル時に解決可能な`IfOperation`を、選ばれた分岐のOperation列で置き換えます。測定結果に依存する`IfOperation`はここでは触りません。
#
# なお、`ForOperation`自体は**このパスではアンロールされません**。ループ展開は必要であれば後段の`emit`で`LoopAnalyzer`が判定します（詳しくは5章）。そのためここで`ForOperations`の個数は減りません。

# %%
block = transpiler.partial_eval(block, bindings=bindings)
print("after partial_eval:", summarise(block))
print("ForOperations:    ", sum(1 for op in block.operations if isinstance(op, ForOperation)))

# %% [markdown]
# `UInt`を未バインドのまま残してループ境界に使うと、下流の`validate_symbolic_shapes`パスが該当する値の名前とともに`QamomileCompileError`を送出します。これは「このカーネルは実はコンパイル時に構造化されていない」という状況を、後段での分かりにくいクラッシュではなく読みやすいエラーへ変換することを担当するパスです。
#
# ### 4.4 `analyze` — 依存グラフとI/O検証
#
# `analyze`は値の依存グラフを構築し、2つの不変条件をチェックします:
#
# 1. ブロックの入出力が古典であること（量子I/Oはエントリポイントではなく*サブルーチン*ブロックでのみ許可されます）。
# 2. どの量子Operationも、測定から計算された古典値に依存しないこと。これを許すとJITコンパイルが必要になりますが、Qamomileは現時点でサポートしていません。`plan`ステージが量子セグメントを1つに制限することでこれを強制します。
#
# 成功するとブロックは`ANALYZED`へ遷移します。

# %%
block = transpiler.analyze(block)
print("after analyze:    ", summarise(block))

# %% [markdown]
# ### 4.5 `plan` — `ProgramPlan`へのセグメント化
#
# `plan`は解析済みブロックを辿り、`OperationKind`ごとにOperationをグループ化し、`ClassicalStep` / `QuantumStep` / `ExpvalStep`のエントリから`ProgramPlan`を組み立てます。デフォルトのトランスパイラが使う`NisqSegmentationStrategy`は**`QuantumStep`を多くても1つ**に制限します。これが典型的なC→Q→Cパターンです。

# %%
plan = transpiler.plan(block)
for i, step in enumerate(plan.steps):
    seg = step.segment
    print(f"  step {i}: {type(step).__name__} ({type(seg).__name__}, {len(seg.operations)} ops)")
print("total unbound parameters:", list(plan.parameters))

# %% [markdown]
# 量子セグメントは`qubit_values`と`num_qubits`も持ちます。これにより`emit`はゲートを配置する前に、バックエンド回路が必要とする量子ビット本数を把握できます。
#
# ### 4.6 `emit` — バックエンド固有のコード生成
#
# `emit`はプランを対象バックエンドの`EmitPass`に渡します。emitパスは具体的な量子ビットインデックスを割り当て、量子セグメントを辿ってバックエンドの`GateEmitter`プロトコルのメソッド（`emit_h`、`emit_rx`、…）を呼び出してネイティブ回路を構築します。

# %%
executable = transpiler.emit(plan, bindings=bindings, parameters=parameters)
print("parameter_names:  ", executable.parameter_names)
print()
print(executable.quantum_circuit)

# %% [markdown]
# 残ったパラメータはまさに保持したもの（`theta`）です。量子ビット数、ループのアンロール、どのCXがどこに入るかといった構造的な決定はすべてコンパイル時に解決されました。
#
# ### 4.7 スキップしたパス
#
# `transpile()`の一部でありながら明示的に呼ばなかったパスが5つあります:
#
# - **`substitute`** — ユーザーが設定した`SubstitutionRule`を適用してブロックのターゲットを置換したり、複合ゲートの戦略を上書きします。`TranspilerConfig`にルールがない場合はno-opです。
# - **`resolve_parameter_shapes`** — `bindings`が具体的な`Vector`や`Matrix`値を提供する場合、`{name}_dim{i}`のshape次元を埋めます。これにより下流で`arr.shape[0]`が具体的な`UInt`として解決されます。
# - **`unroll_recursion`** — 自己再帰の`@qkernel`（例: Suzuki–Trotter、チュートリアル07参照）に対する`inline ↔ partial_eval`の固定点ループです。再帰が底まで展開されると終了し、bindingsでベースケースに到達できない場合はエラーになります。
# - **`affine_validate`** — フロントエンドのチェックをすり抜けたアフィン型違反を捕まえるセーフティネットです。
# - **`validate_symbolic_shapes`** — 未解決の`Vector`shape次元が`ForOperation`の境界に到達した場合、実行可能なエラーメッセージで拒否します。
#
# いずれも冪等かつ安価なので、`transpile()`は常にこれらを実行します。パスを書く側としては順序に注意するだけで十分です: `substitute`と`resolve_parameter_shapes`は`inline`の**前**、`affine_validate`は`inline`の**後**、`validate_symbolic_shapes`は`analyze`の**後**（依存グラフが使えるように）に実行されます。

# %% [markdown]
# ## 5. 制御フロー (`if` / `for` / `while`) の取り扱い
#
# パイプラインが制御フローをどう扱うかは、フロントエンドで何を受け付けるかから、各パスがそれをどう変形するか、そしてバックエンドが実行時分岐をサポートするかまで、複数のレイヤーに関わります。ここではその全体像を整理します。ユーザー向けの書き方は[チュートリアル05](05_classical_flow_patterns)にあり、本章はコンパイラ側の視点に絞ります。
#
# ### 5.1 フロントエンドで受け付ける形
#
# `@qkernel`はトレース前にASTを書き換えます（`qamomile/circuit/frontend/ast_transform.py`の`ControlFlowTransformer`）。ここで以下のように変換されます:
#
# - Pythonの`if`文 → `emit_if(cond, true_branch, false_branch, ...)`呼び出し
# - `for`文 → `for_loop(start, stop, step)`または`for_items(dict)`コンテキストマネージャ
#
# このため、実行時に決まる量子レジスタ・測定結果・未バインドの値を素のPythonの`if`/`for`で使ってもトレース時に両分岐・各反復を実行する、という直感的な動作になります。対応するループ記法は:
#
# - **`qmc.range(n)`** — シンボリックなループ境界。`n`がコンパイル時に定まらなくても`ForOperation`としてIRに残せます。
# - **`qmc.items(d)`** — 辞書・スパースデータ用。**常にコンパイル時にアンロール**されます（`ForItemsOperation`）。
# - 直接の`for i in <runtime_value>:` — 拒否されます。必ず`qmc.range(...)`か`qmc.items(...)`を経由してください。
#
# `while`は`while_loop`コンテキストマネージャで書きます。条件は**測定結果 (`Bit`)** でなければならず、古典変数や定数を条件にすると後段の`ValidateWhileContractPass`で拒否されます。
#
# ### 5.2 IR表現
#
# `qamomile/circuit/ir/operation/control_flow.py`に以下が定義されています:
#
# | Operation | ネストリスト | 条件・境界 | 特記事項 |
# |-----------|------------|----------|--------|
# | `ForOperation` | `operations`（本体） | `operands = [start, stop, step]`（いずれも`UInt`） | `loop_var`名を持つ |
# | `ForItemsOperation` | `operations`（本体） | `operands[0]`が`DictValue` | 常にコンパイル時アンロール |
# | `IfOperation` | `true_operations`, `false_operations` | `operands[0]`が`Bit` | `phi_ops`で分岐後の値マージ |
# | `WhileOperation` | `operations`（本体） | `operands[0]`（初期条件）, `operands[1]`（ループキャリー条件） | 測定結果`Bit`必須、`max_iterations`ヒント可 |
#
# 4つとも`HasNestedOps`を実装しているので、パスは`nested_op_lists()` / `rebuild_nested()`経由で本体へ再帰的に入れます。`isinstance`のチェーンは書かないのが流儀です。
#
# `IfOperation`には値をマージする**Phiノード** (`PhiOp`) が付きます。両分岐で同じ論理量子ビット・古典変数を異なるバージョンで更新した場合、分岐後に使う側はPhi経由でどちらのバージョンなのかを参照します。
#
# ### 5.3 パスごとの挙動
#
# 制御フローは各パスで次のように扱われます:
#
# | パス | `IfOperation` | `ForOperation` | `WhileOperation` |
# |------|-------------|---------------|-----------------|
# | `inline` | 両分岐の本体へ再帰 | 本体へ再帰 | 本体へ再帰 |
# | `partial_eval` | 条件が定数なら**選ばれた分岐で置換**（`CompileTimeIfLoweringPass`）。測定結果条件なら保持 | 境界の`BinOp`は畳み込まれる。**アンロールはしない** | 何もしない（ここでは変形対象外） |
# | `analyze` | Phiが依存グラフに反映される | `loop_var`が本体の依存に入る | 測定結果条件を量子オペランドと同様に扱う |
# | `validate_symbolic_shapes` | — | 未解決の`Vector`shape次元が境界にあると拒否 | — |
# | `plan` | `OperationKind.CONTROL`としてセグメント境界を作る | 同左 | 同左 |
# | `emit` | 実行時`if`として出力（バックエンドが対応していれば） | `LoopAnalyzer.should_unroll()`で判定し、必要ならアンロール | 実行時`while`として出力 |
#
# **`LoopAnalyzer.should_unroll()`** （`transpiler/passes/emit_support/loop_analyzer.py`）の判定基準は:
#
# 1. ループ境界が外側のループ変数に依存する（動的入れ子）
# 2. 本体で配列を`loop_var`でインデックスしている（例: `q[i]`）
# 3. `loop_var`が`BinOp`に現れる（例: `i + 1`、`2 * i`）
#
# 本チュートリアルの`demo_kernel`は`q[i]`と`q[i + 1]`の両方を使うので、条件1, 2, 3に該当して`emit`時にアンロールされます。これが`executable.quantum_circuit`がフラットな2量子ビット分のCX列になっている理由です。上記のどれにも該当しないループは、バックエンドが対応している限り実行時ループとして回路に残ります。
#
# ### 5.4 量子と古典の依存関係ルール (`analyze`)
#
# `analyze`パスは、**量子Operationは測定から派生した古典値に依存してはならない**という不変条件を強制します。
#
# ```python
# # OK: 測定結果Bitを条件に量子ゲートを実行
# b = qmc.measure(q)
# if b:
#     q = qmc.x(q)
#
# # NG: 測定結果から計算した古典値を量子ゲートの引数に使う
# b = qmc.measure(q)
# x = some_classical(b)
# q = qmc.rx(q, x)   # analyzeで拒否
# ```
#
# 前者は測定結果`Bit`を`IfOperation`の条件として直接使うだけで、量子オペランドの型は変わりません（位相キックバック的な制御は行いません）。後者はJITコンパイルが必要になり、現時点ではサポートしていません。`plan`ステージが量子セグメントを1つに制限することがこの保証の裏返しです。
#
# ### 5.5 バックエンドの実行時分岐サポート
#
# 実行時の`if`/`while`（=測定結果に依存する分岐）が回路まで落ちてくるかはバックエンドの`MeasurementMode`に依存します（`qamomile/circuit/transpiler/gate_emitter.py`）:
#
# | モード | 実行時if/while | 用例 |
# |------|--------------|------|
# | `NATIVE` | サポート。条件付きゲートを明示的にemit | Qiskit（`QuantumCircuit.if_test` 等） |
# | `STATIC` | 非サポート。測定前の状態ベクトル・演算子を返す | QURI Parts |
# | `RUNNABLE` | フルサポート。実行時ループ/分岐も含む | CUDA-Q (`cudaq.run()`経由) |
#
# 非対応モードのバックエンドで`IfOperation`/`WhileOperation`を含むカーネルをtranspileしようとすると、emitパスがエラーを送出します。モードを意識してカーネル側で実行時分岐を書くかどうか決めるのがコントリビュータの責任です。
#
# ### 5.6 よくあるエラー
#
# - **`ValidationError` (analyze)** — 測定から派生した古典値を量子ゲートの引数に使った。パターンを書き換えるか、測定の代わりに状態を保つように設計を見直してください。
# - **`ValidateWhileContractPass`エラー** — `while`の条件が測定結果`Bit`でない。Pythonの古典変数や定数条件でのループは未サポートです。
# - **`QamomileCompileError` (validate_symbolic_shapes)** — `ForOperation`の境界に未解決の`Vector` shape次元が届いた。該当する`Vector`を`bindings`で具体化するか、`qmc.items`を使う設計に変えてください。
# - **emit時エラー** — `MeasurementMode.STATIC`のバックエンドに実行時`if`が到達した。バックエンドを変えるか、カーネルを別の等価表現で書き直します。

# %% [markdown]
# ## 6. バックエンドemission: Qiskit vs QURI Parts
#
# どのバックエンドも、`qamomile/circuit/transpiler/`で定義された2つのプロトコルを実装することでパイプラインに接続します:
#
# - **`GateEmitter[T]`** (`gate_emitter.py`): 「ゲートをどう描くか」のAPIです。`create_circuit(num_qubits, num_clbits) -> T`、`create_parameter(name) -> Any`、ゲートごとの約40個のエントリポイント（`emit_h`、`emit_rx`、`emit_cx`、…）を持ちます。加えて`measurement_mode: MeasurementMode`を告知します:
#
#   | モード | 意味 | 利用バックエンド |
#   |------|---------|---------|
#   | `NATIVE` | emitパスが呼ぶ明示的な測定命令をバックエンドが持つ。 | Qiskit |
#   | `STATIC` | バックエンドは測定前の状態ベクトル・演算子を受け取り、samplerが測定を外部で処理する。 | QURI Parts |
#   | `RUNNABLE` | バックエンドがランタイム制御フロー付きのmid-circuit測定をサポートする。 | CUDA-Q (`cudaq.run()`経由) |
#
# - **`CompositeGateEmitter[C]`** (`passes/emit.py`): オプションです。バックエンドが複合ゲート（QFT、QPE、…）をネイティブ実装でショートカットできるようにします。`can_emit(gate_type) -> bool` / `emit(...) -> bool`のコントラクトで、オプトアウトするには`False`を返します。その場合emitパスはライブラリレベルの分解にフォールバックします。
#
# `Transpiler`のサブクラスは`_create_segmentation_pass`と`_create_emit_pass`をオーバーライドし、ランタイム側のために`executor()`も実装することでこれらを接続します。`qamomile/qiskit/transpiler.py`は約50行の標準的なリファレンス実装です。
#
# では同じカーネルをQURI Parts経由でトランスパイルして比較してみましょう。QURI Partsはオプション依存です。ローカルで再現するには`pip install 'qamomile[quri_parts]'`でインストールしてください。

# %%
try:
    from qamomile.quri_parts import QuriPartsTranspiler

    quri_transpiler = QuriPartsTranspiler()
    quri_exe = quri_transpiler.transpile(
        demo_kernel, bindings=bindings, parameters=parameters
    )

    print("backend circuit type: ", type(quri_exe.quantum_circuit).__name__)
    print("parameter_names:      ", quri_exe.parameter_names)
    print()
    for gate in quri_exe.quantum_circuit.gates:
        print(" ", gate)
except ModuleNotFoundError:
    # ``qamomile[quri_parts]``はオプション依存のグループなので、インストールされていない
    # 場合は並列比較をスキップし、このnotebookが引き続き実行できるようにします。
    print("QURI Parts is not installed; skipping the side-by-side output.")

# %% [markdown]
# 指摘すべき違いは3つあります:
#
# 1. **回路の型。** Qiskitは`Parameter`オブジェクトを埋め込んだ`QuantumCircuit`をemitします。QURI PartsはパラメータがQURI Partsの`Parameter`インスタンスである`LinearMappedUnboundParametricQuantumCircuit`をemitします。どちらもQamomileの`parameter_names`を同じ形で往復します。
# 2. **測定。** Qiskitの回路は`measure`命令で終わります（`measurement_mode=NATIVE`）。QURI Partsの回路は測定ゲートを持ちません。サンプリングは実行時にexecutorが処理します（`measurement_mode=STATIC`）。
# 3. **複合ゲート。** カーネルが`qmc.qft(...)`を使う場合、Qiskitの`QiskitQFTEmitter`は`QFTGate`ボックスを配置しますが、QURI Partsバックエンドはライブラリパス経由で分解します。IRは同じですが、実現される回路は異なります。カーネルごとに`TranspilerConfig.with_strategies({"qft": "approximate"})`で上書きできます。

# %% [markdown]
# ## 7. コントリビュータ向けのポインタ
#
# **カスタムパスの記述。** `qamomile/circuit/transpiler/passes/`に配置し、`Block`を受け取り`Block`を返すようにして、入力`kind`の事前条件を冒頭でアサートしてください。Operationを辿るときは`isinstance(op, ForOperation)`のチェーンではなく`HasNestedOps`を使ってください。そうすれば将来の制御フローOperationも自動的に扱えます:
#
# ```python
# def rewrite(ops):
#     new_ops = []
#     for op in ops:
#         if hasattr(op, "nested_op_lists"):
#             op = op.rebuild_nested([rewrite(child) for child in op.nested_op_lists()])
#         new_ops.append(transform(op))
#     return new_ops
# ```
#
# **新しいバックエンドの追加。** 最低限のチェックリスト:
#
# 1. 対象SDK向けに`GateEmitter[T]`を実装します（`T`はSDKの回路型）。`qamomile/qiskit/emitter.py`から始めるとよいでしょう。
# 2. `Transpiler[T]`をサブクラス化し、`_create_segmentation_pass`（他に必要がなければ`NisqSegmentationStrategy`を使用）と、`StandardEmitPass(your_emitter)`を返す`_create_emit_pass`を実装します。
# 3. ユーザーが`executor()`を呼べるように`QuantumExecutor[T]`のサブクラスを実装します。
# 4. オプション: emitされた回路で高レベル構造を保つため、QFT/QPEなどの`CompositeGateEmitter`を追加します。
#
# **transpileエラーのデバッグ。** パスを1つずつ実行し、その間に`summarise(block)`で件数の変化を追い、気になるところは`pretty_print_block(block)`で中身を覗きます。`BlockKind`が進まない、Operation数が爆発する、例外が送出される、というステージが最初に見るべき場所です。`pretty_print_block(block, depth=N)`で`CallBlockOperation`の展開深さを変えながら`inline`前後を比較すると、どこで値が切れたか・どのPhiが漏れたかが読み取りやすくなります。

# %% [markdown]
# ## 8. まとめ
#
# パイプラインは4つのkindを遷移していくSSAスタイルのIRです:
#
# - `HIERARCHICAL` — 生のトレース、ブロック呼び出しが未展開
# - `AFFINE` — フラットなOperationと制御フロー、ブロック呼び出しなし
# - `ANALYZED` — 検証済み、依存グラフ化済み、セグメント化可能
# - `ProgramPlan` → `ExecutableProgram[T]` — セグメント化されemit済み
#
# 各パスは限定的な仕事を持ち、`BlockKind`に対する事前条件を持ちます。`Transpiler`のステップ実行用APIはすべてのパスを公開しています。カーネルが期待通りに動かないときの主たるデバッグツールとして、またパスやバックエンドを追加するときの拡張点として活用してください。
#
# 制御フローの要点:
#
# - `if`/`for`はトレース時にASTが書き換えられ、`IfOperation` / `ForOperation` / `ForItemsOperation` / `WhileOperation`というIRに変換される
# - `partial_eval`はコンパイル時`if`を除去するが、`for`のアンロールは`emit`の`LoopAnalyzer`が判定する
# - `analyze`は「量子Operationが測定由来の古典値に依存しないこと」を保証する
# - 実行時分岐を回路まで落とせるかはバックエンドの`MeasurementMode`次第（`NATIVE`か`RUNNABLE`が必要）
