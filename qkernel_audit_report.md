# Qamomile qkernelシステム 実装調査レポート

- **調査日**: 2026-07-03
- **対象**: qkernelシステムのコンパイルパイプラインとバックエンド実行(`qamomile/circuit/frontend/`、`qamomile/circuit/ir/`、`qamomile/circuit/transpiler/`、`qamomile/{qiskit,quri_parts,cudaq,qbraid}/`)— 約4万行
- **範囲外**: `visualization/`、`estimator/`、`optimization/`(コンパイル・実行の中核ではないため)
- **方法**: 5領域(Frontend / IR / Transpilerパス / Emit・実行ランタイム / バックエンド)の並列監査。全ファイル精読+Operationサブクラスのプロトコル網羅チェック+encode/decode対称性検査+3バックエンドのゲート規約の数理検証。最重要指摘は再現スクリプトによるエンドツーエンド検証、またはコード再読による裏取りを実施済み(各項目に「再現済み」と明記)。
- **観点**: (1) バグの有無 (2) 非効率な処理の有無 (3) リファクタリング可能性
- **深刻度**: P1 = トリガー可能な正確性バグ(サイレント誤コンパイル等)/ P2 = エッジケースバグまたは重大な非効率 / P3 = エッジリスク・保守性・軽微な非効率

**発見総数: 約100件(P1 = 5件、P2 = 約20件、P3 = 約75件)**

---

## 1. アーキテクチャ概観

### 1.1 コンパイルパイプライン(実装ベース)

`Transpiler.transpile()`([transpiler.py:525-611](qamomile/circuit/transpiler/transpiler.py))の実際の実行順:

```
to_block → validate_entrypoint → substitute → resolve_parameter_shapes
→ inline → unroll_recursion → affine_validate
→ partial_eval(内部: reject_self_referential_loop_stores → ConstantFolding
               → CompileTimeIfLowering → ConstantFolding 2回目)
→ slice_borrow_check → strip_slice_ops → analyze → classical_lowering
→ validate_symbolic_shapes
→ plan(内部: ValidateWhileContract → materialize_return
        → lower_measure_qfixed → セグメント分割)
→ emit
```

CLAUDE.md 記載のパス列には `slice_borrow_check` / `strip_slice_ops` / `validate_while` が現れず、`transpiler.py` の docstring(:129-152, :558-568)も実装と乖離している。BlockKind 前提条件の検査は各パスが個別に手書きしており、「raiseする / 黙って素通しする / チェック自体が no-op」の3様式が混在する。

### 1.2 Emit・実行ランタイム

emit はバックエンド固有 `EmitPass`(`StandardEmitPass` サブクラス)が `ProgramPlan` の C→Q→C ステップを走査する。量子セグメントは2相処理: `ResourceAllocator.allocate()` が qubit/clbit マップを構築し、`_emit_operations()` が isinstance ディスパッチで `GateEmitter` プロトコルへ流す。`LoopAnalyzer` が unroll かネイティブループかを決定し、controlled-U は `blockvalue_to_gate`(再利用可能ゲート化)→ 失敗時 `emit_controlled_operations`(ゲート単位walker)にフォールバックする。

実行側は `ExecutableProgram.sample()/run()` → `ProgramOrchestrator` がユーザー束縛を indexed 展開して `ExecutionContext` に積み、プレ量子古典セグメントを `ClassicalExecutor`(Pythonインタープリタ)で実行、`QuantumExecutor.execute()` で counts を取得し、ポスト量子古典セグメントをユニークビット文字列ごとに再実行して `output_refs` から最終出力を復元する。

### 1.3 バックエンド横断比較

| 機能 | Qiskit | QURI Parts | CUDA-Q |
|---|---|---|---|
| circuit 型 | `QuantumCircuit` | `LinearMappedUnboundParametricQuantumCircuit` | 生成Pythonソース→`@cudaq.kernel` |
| P ゲート | ネイティブ(厳密) | 具体値=U1 / parametric=RZ代用(global phase差、文書化済) | `r1`(厳密) |
| controlled-U | `to_gate()`+`Gate.control()` | transpiler内の再帰walker(~740行) | ヘルパーカーネル+`cudaq.control` |
| 測定モード | NATIVE | STATIC(全qubit測定+map復号) | STATIC / RUNNABLE(事前スキャンで切替) |
| runtime if/while | ネイティブ(`if_test`/`while_loop`) | 非対応 | RUNNABLEモードで生成Pythonの`if`/`while` |
| パラメータ束縛 | dict(**欠落は無言スキップ**) | 順序リスト(欠落は例外) | 順序リスト(欠落は例外) |
| seed 再現性 | なし | あり(qulacs) | なし |
| observable 変換 | `SparsePauliOp`(`abs()>1e-15`) | `Operator`(`abs()>1e-15`) | `SpinOperator`(**`math.isclose`→complexでTypeError**) |

ゲート規約(RZ/RZZ の 1/2 因子、CH/CY/CRX 分解、CCP/CRZZ 構成、`exp_pauli` の符号、counts のエンディアン)は数理検証の結果 **3バックエンドで意味論的に一致**。既知差分は QURI parametric P→RZ の global phase のみ。

---

## 2. 総合評価と横断的バグパターン

アーキテクチャの骨格(IR抽象度原則、BlockKind状態機械、HasNestedOps / all_input_values プロトコル、StandardEmitPass への emit ループ集約)は健全で、設計思想と実装は概ね一致している。全28種の Operation について `all_input_values`/`replace_values` の拡張フィールド被覆と `HasNestedOps` 実装は欠落なし、serialize の閉じたディスパッチテーブル(セキュリティ不変条件)も正しく守られている。

一方、発見されたバグには明確な**反復パターン**があり、個別修正よりパターンを閉じる構造的リファクタリングが有効:

| パターン | 該当バグ |
|---|---|
| ① 手書きネスト走査の経路漏れ | TP-2(substitutionネスト未到達)、TP-3(lower_measure_qfixedトップレベル限定)、TP-13(results非置換)ほか7箇所以上 |
| ② 「Value参照を持つ場所」のリストの手書き複製(canonical / serialize / clone / substituteの4実装) | IR-1(content_hash非決定性)、IR-2(serialize未登録)、IR-6 |
| ③ 演算子オーバーロード欠落によるサイレントPython定数化 | FE-3(Bit.__eq__)、FE-5(UInt==Float) |
| ④ 出力ガード・エラー昇格の非対称 | TP-1(BinOp畳み込みのみガードなし)、EM-2(無音0.0)、EM-7(warning止まり) |
| ⑤ 3バックエンドの並行実装ドリフト | BE-1(complex係数)、BE-18(CUDA-Qデッドコード化)、BE-20(walker3重実装) |

---

## 3. 最優先修正事項(P1: サイレント誤コンパイル)

エラーなく誤った結果を返す。リファクタリングを待たず単独PRで修正すべき。

### P1-1. 同一サブカーネル複数呼び出しで結果Valueが共有され、測定結果が1つのclbitに潰れる【再現済み】
- **場所**: [block.py:95-100](qamomile/circuit/ir/block.py)(`Block.call`)、発火点 [qkernel.py:866](qamomile/circuit/frontend/qkernel.py)
- **内容**: `Block.call` は出力が入力由来でない場合、キャッシュされた callee ブロック自身の出力 Value **インスタンスそのもの**を `results` に返す。`QKernel.__call__` は `self.block` を全呼び出しサイトで共有するため、Bit を返すサブカーネルを2回呼ぶと2つの `CallBlockOperation.results` が同一 UUID を持つ。再現: `b0 = flip(q0); b1 = flip(q1)` → 2つの MeasureOperation が同一 Bit Value に書き込み、`clbit_map` は1エントリ、b0 と b1 が常に同一値になる。
- **提案**: 非パススルー出力に `dummy_return.next_version()`(または fresh Value + logical_id 継承)を発行。回帰テスト(同一サブカーネル2回呼び出し→clbit 2本)を追加。

### P1-2. モジュールグローバルと同名のローカル変数がif分岐スレッディングから漏れ、分岐内の代入が消える【再現済み】
- **場所**: [ast_transform.py:1281](qamomile/circuit/frontend/ast_transform.py)(`global_names = set(func.__globals__.keys())`)、除外箇所 :105-118
- **内容**: `VariableCollector` はモジュールグローバル名を変数収集から一律除外するが、判定が `__globals__` の全キーのため、モジュールトップレベル(ノートブックでは過去の全変数)と同名のローカル変数が `visit_If` の input/output 集合から漏れ、**分岐内の代入がサイレントに破棄される**。再現: モジュールに `count = 123` がある状態で kernel 内 `count = 0.0; if flag==1: count = 1.0` → 出力は phi ではなく定数 `0.0` に固定。
- **提案**: `symtable` / `co_varnames` で関数スコープの Store 名を先に収集し `global_names` から差し引く。

### P1-3. `Bit` に `__eq__`/`__ne__` の DSL オーバーロードがなく、実行時比較がコンパイル時定数 False になる【再現済み】
- **場所**: [primitives.py:506-571](qamomile/circuit/frontend/handle/primitives.py)(`Bit`)、[control_flow.py:689-694](qamomile/circuit/frontend/operation/control_flow.py)
- **内容**: `UInt`/`Float` は `__eq__` で CompOp を発行するが `Bit` は dataclass 生成の `__eq__` のまま。測定結果同士の `m0 == m1` はフィールド比較でほぼ常に Python `False` を返し、`emit_if` がそれを `IfOperation.operands[0]` に格納 → 実行時分岐のつもりがコンパイル時定数 False として畳まれ else 側のみ残る。副作用として `Bit`/`Qubit`/`Vector` は `__hash__ = None`(unhashable)。
- **提案**: `Bit.__eq__`/`__ne__` を CondOp/CompOp 発行(XNOR/XOR 相当)として実装し `__hash__` を復元。`emit_if` で生 bool 条件を検出したら警告または明示化。

### P1-4. ブロック出力を生成する BinOp が定数畳み込みで削除され、実行結果が None になる【再現済み】
- **場所**: [constant_fold.py:150-156](qamomile/circuit/transpiler/passes/constant_fold.py)(対比: Store側のガード :162-186)
- **内容**: `StoreArrayElementOperation` の畳み込みには「結果がブロック出力なら op を残す」`block_output_uuids` ガードがあるのに、BinOp の畳み込みにはなく無条件削除。再現: `y = x * 2.0; return y, bits` を `bindings={"x": 3.0}` で transpile→sample すると結果が `None`(`parameters=["x"]` なら正しく `6.0`)。束縛済み入力をそのまま返す `return x, bits` も同根で None になる。
- **提案**: BinOp 畳み込みにも `block_output_uuids` ガードを追加。オーケストレータ側でも `output_refs` 解決時に bindings / 定数メタデータへフォールバックし、解決不能は None ではなく明示エラーに。

### P1-5. シンボリック幅 `QUIntType`/`QFixedType` で `content_hash` が非決定的になる【再現済み】
- **場所**: [canonical.py:452-570](qamomile/circuit/ir/canonical.py)(`canonical_value` が `type` 内の Value をリマップしない)、[q_register.py:30-50](qamomile/circuit/ir/types/q_register.py)(`label()` が Value の repr を埋め込む)
- **内容**: `QUIntType.width` 等はシンボリック `Value` を取り得るが、canonicalize は型を素通しし、`label()` の f-string が Value の dataclass repr(生 uuid4 + `<UIntType object at 0x...>` の**メモリアドレス**)を canonical bytes に展開する。再現: 構造的に同一なブロック2つの `content_hash` が不一致、同一プロセス内の再実行でも揺れる。content-addressable キャッシュ・IR diff の根幹保証が破れる。
- **提案**: `canonical_value` で型内 Value に再帰適用。`label()` はシンボリック幅を UUID のみの表現に落とす(serialize の `$value_ref` 方式に揃える)。

---

## 4. 領域別詳細発見事項

### 4.1 Frontend(`qamomile/circuit/frontend/`)

P1-1〜P1-3 は上記参照(元番号 F-1〜F-3)。

#### バグ(P2)

- **FE-4** [ast_transform.py:1318](qamomile/circuit/frontend/ast_transform.py) — `name_space = func.__globals__.copy()` によりデコレーション後にモジュールへ追加された名前(後方定義のヘルパーカーネル)がトレース時に見えず `NameError`【再現済み】。Python の遅延束縛と乖離。→ `__missing__` で実グローバルへフォールバックする dict サブクラスに変更。
- **FE-5** [primitives.py:104-113, 485-492](qamomile/circuit/frontend/handle/primitives.py) — `UInt == Float` が両者 NotImplemented → 同一性フォールバックで Python `False` になり、if 条件でコンパイル時定数化【再現済み】。→ 型昇格して CompOp を発行するか TypeError に。
- **FE-6** [qkernel.py:837-848](qamomile/circuit/frontend/qkernel.py)、[control.py:476-514](qamomile/circuit/frontend/operation/control.py)、[inverse.py:1482-1507](qamomile/circuit/frontend/operation/inverse.py) — 定数引数・具象形状のサブカーネル呼び出しが**呼び出しサイトごとに毎回 full re-trace**。さらに `_extract_return_names()`(qkernel.py:1143-1179)が毎回 `inspect.getsource`+`ast.parse` を実行。→ `(parameters, bindings, qubit_sizes)` キーの per-signature キャッシュ+return 名抽出の cached_property 化。

#### バグ(P3)

- **FE-7** [ast_transform.py:976-990](qamomile/circuit/frontend/ast_transform.py) — `qmc.range(n-1)` の境界式がガードと `for_loop` で二重評価され、シンボリック境界ごとに死んだ BinOp が IR に残る【再現済み】。
- **FE-8** [control_flow.py:92-104](qamomile/circuit/frontend/operation/control_flow.py) — while 条件の2回目評価が「定義オペが IR に存在しない Value」を `while_op.operands[1]` に載せうる(現在は ValidateWhileContractPass が守っている)。
- **FE-9** [control_flow.py:271-293](qamomile/circuit/frontend/operation/control_flow.py)、[array.py:2455-2571](qamomile/circuit/frontend/handle/array.py) — if 分岐トレースで `VectorView` の浅いコピーが共有 `_slice_parent` の `_borrowed_indices` を両分岐から変異させ、分岐内での view 破壊的消費で所有権整合が弱まる。
- **FE-10** [ast_transform.py:380-403](qamomile/circuit/frontend/ast_transform.py) — while 条件で禁止する量子オペ一覧 `_QUANTUM_OPS` が手動同期で `pauli_evolve`/`cast` が漏れ(fail-open)。
- **FE-13** [pauli_evolve.py:95-137](qamomile/circuit/frontend/operation/pauli_evolve.py) — 他ゲートと逆に「emit → consume」の順で、失敗時にゴミオペが tracer に残る。`gamma` に生 float のリテラル昇格もない。
- **FE-14** [qkernel.py:108-120](qamomile/circuit/frontend/qkernel.py) — `_promote_literal_to_handle` が `int`/`float`/`bool` の生アノテーションを昇格せず、`def sub(q: Qubit, n: int)` のサブカーネル呼び出しが失敗(build()/transpile() 経路とは契約が食い違う)。
- **FE-15** [constructors.py:52-54, 70-72](qamomile/circuit/frontend/constructors.py) — `uint("name")`/`float_("name")` が `.with_parameter(name)` を付けず、bindings の名前解決経路に乗らない疑い。
- **FE-16** [qkernel.py:435-445](qamomile/circuit/frontend/qkernel.py) — `_block_building`/`_specializing`/`_pending_self_calls` がスレッド間共有(Tracer は contextvars 分離済みなのと対照的)。
- **FE-17** [ast_transform.py:37-49, 105-118](qamomile/circuit/frontend/ast_transform.py) — `VariableCollector._exclude` が訪問順依存(呼び出し名の除外集合が visit 中に成長)。

#### 非効率(P3)

- **FE-11** [array.py:633](qamomile/circuit/frontend/handle/array.py) — 古典配列の読み取りアクセスでも `_borrowed_indices` に永続エントリが積まれ続ける(解放経路なし)。
- **FE-12** [handle.py:57-133](qamomile/circuit/frontend/handle/handle.py) — `_emit_binop` は定数を eager fold するが `_emit_compop`/`_emit_condop`/`_emit_notop` は畳まない非対称。
- **FE-18** ast_transform.py / qkernel.py — 1カーネルにつき `inspect.getsource`+`ast.parse` が最大3回(transform / rebind解析 / return名抽出)。
- **FE-19** [composite_gate.py:418, 324-363](qamomile/circuit/frontend/composite_gate.py) — 呼び出しごとにデコンポジション再トレース+Vector/tuple 分岐のほぼ同一コード重複。
- **FE-23** [measurement.py:148-151](qamomile/circuit/frontend/operation/measurement.py) — `validate_all_returned` の二重呼び出し(consume() 内でも実行される)。

#### リファクタリング(P3)

- **FE-20** — `object.__new__` + 手動属性設定による Handle 生成が7箇所以上に重複(qkernel.py:1278、func_to_block.py:355-450、array.py:254/2166/2273、control_flow.py:909)。`id` 初期化も `uuid4()` と `str(id(instance))` で不統一。→ 型別ファクトリに集約。
- **FE-21** — 配列要素型抽出ヘルパーの三重定義(qkernel.py:60、func_to_block.py:62、param_validation.py:58)+同パターンのインライン展開約6箇所。`is_array_type` はクラス名文字列照合で誤判定リスク。
- **FE-22** [qubit_gates.py](qamomile/circuit/frontend/operation/qubit_gates.py) — ブロードキャストヘルパー3種(:60-151, :476-508)がほぼ同一、`p()`/`cp`/`rzz` が共通ヘルパーを再実装。
- **FE-24** ast_transform.py:1331-1334 — 現挙動と乖離した古いコメント。
- **FE-25** inverse.py:263-275 — 逆ループ境界が負値を `UIntType` 定数で表現(型整合性の smell、コメントで自認)。

### 4.2 IR(`qamomile/circuit/ir/`)

P1-5 は上記参照(元番号 1-1)。

#### バグ(P2)

- **IR-2** [encode.py:627-651](qamomile/circuit/ir/serialize/encode.py) / [decode.py:729-750](qamomile/circuit/ir/serialize/decode.py) — シンボリック幅 Value が value_table に登録されず(`$value_ref` を書くだけ)、幅 Value が他所に現れないブロックは encode 成功 → decode 失敗【再現済み】。
- **IR-3** [decode.py:823-837, 1065-1076](qamomile/circuit/ir/serialize/decode.py) — コンテナ operand(`TupleValue`/`DictValue`)を持つ `ReturnOperation` は encode できるが decode で失敗【再現済み】。許容版デコーダは `ForItems`/`InverseBlock` にだけ場当たり的に生えている。
- **IR-4** [q_register.py:17-50](qamomile/circuit/ir/types/q_register.py)、[hamiltonian.py:15-16](qamomile/circuit/ir/types/hamiltonian.py) — `QUIntType`/`QFixedType`/`ObservableType` が dataclass の仕様により unhashable(基底クラスの「dict キーとして使える」契約に違反、`TupleType`/`DictType` は `__hash__` 定義済みで非対称)【検証済み】。
- **IR-5** [canonical.py:1010-1036](qamomile/circuit/ir/canonical.py) — display-only と宣言された `loop_var`/`key_vars`/`value_var` が canonical bytes に混入し、ループ変数名の変更で `content_hash` が変わる【再現済み】。

#### バグ(P3)

- **IR-6** [value.py:620-632](qamomile/circuit/ir/value.py) — `ArrayValue.next_version()` が `parent_array`/`element_indices` を落とす(`Value.next_version` は保持、value.py:734-737 の前提と矛盾)。
- **IR-7** [encode.py:547-567](qamomile/circuit/ir/serialize/encode.py) / decode.py:591-598 — `ParamSlot.bound_value`/`default` の tuple→list、int キー→str キーが round-trip で非可逆。
- **IR-8** [value.py:64-81, 528-552](qamomile/circuit/ir/value.py) — `const_array` に ndarray が入ると frozen dataclass の hash/eq が TypeError/ValueError になり得る(現在は identity fast-path で偶然回避)。
- **IR-9** [canonical.py:406](qamomile/circuit/ir/canonical.py) — 値を持たない Operation を複製せず共有するため、正規化後 Block が入力と mutable オブジェクトを共有。
- **IR-10** [value_mapping.py:805-813](qamomile/circuit/ir/value_mapping.py) — マップ先が `TupleValue`/`DictValue` の場合に要素の再置換が行われない非対称。
- **IR-11** value.py:19-26 — 存在しない `thaw_data` への docstring 参照。

#### 非効率

- **IR-12(P2)** [encode.py:210](qamomile/circuit/ir/serialize/encode.py) — ネスト Block の block dict が**トップレベルと同一の value_table list を共有**し、JSON/msgpack 化時に Block 数×全 Value 数でペイロード膨張【再現済み】。decode 側もネスト Block ごとに全値を再マテリアライズ。→ 到達可能値のみのスコープ分割(スキーマバンプ不要)。
- **IR-13(P3)** canonical.py:812-849 + :994-1007 — ネスト Block の値宣言が親と自身の2箇所に二重出力。
- **IR-14(P3)** [value_mapping.py:729](qamomile/circuit/ir/value_mapping.py) — 非マップ値にもメタデータをフル再構築してから等値比較で捨てる(高速判定 `_metadata_has_referenced_field` は既存なのに未使用)。
- **IR-15(P3)** [operation.py:70-91](qamomile/circuit/ir/operation/operation.py) ほか21箇所 — ホットパスで `@runtime_checkable` Protocol への isinstance(具象タプル判定に置換可能)。

#### リファクタリング

- **IR-16(P2)** — encode/decode の 28 op × 2 のボイラープレートをテーブル駆動化(`$type → OpSpec(cls, fields)`)。「対で書くべきものの片側漏れ」(IR-3 の類)を構造的に防止。閉じたディスパッチのセキュリティ不変条件はテーブルが静的なら維持される。
- **IR-17(P2)** — Value 深部 walk の統一: `canonical_value` / `UUIDRemapper.clone_value` / `_resubstitute_fields` / `_substitute_value_fields` の4実装(うち後2者は約150行がほぼ重複)を `map_value_tree(value, fn)` に集約。P1-5・IR-2 型のバグを構造的に再発不能にする。
- **IR-18(P3)** — `CastOperation.qubit_mapping` の帯域外 UUID 書き換えが3箇所に散在(value_mapping.py:86、canonical.py:415、passes/value_mapping.py:117)。`Operation.remap_uuid_refs(fn)` フックに統一。
- **IR-19(P3)** — `IfOperation.phi_ops` の扱いが `ValueSubstitutor`(内部で置換)と `HasNestedOps`(呼び手が再帰)で非対称。
- **IR-20(P3)** — 無意味な遅延初期化辞書(arithmetic_operations.py:255-284、printer.py:264-306)。
- **IR-21(P3)** — 小粒: `_chase_transitive` の重複実装、`_replace_power` の未使用 bool 戻り値、`operation/__init__.py` の `__all__` 不足、`expval.py:49` の deprecated プロパティ、docstring typo「operants」。

### 4.3 Transpilerパス(`qamomile/circuit/transpiler/passes/` ほか)

P1-4 は上記参照(元番号 #1)。

#### バグ(P2)

- **TP-2** [substitution.py:237-295](qamomile/circuit/transpiler/passes/substitution.py) — SubstitutionPass が callee ブロック内部の呼び出しに再帰せず、入れ子(`entry → layer → oracle`)のルールが黙って無視される【再現済み】。`ControlledUOperation.block` 等も未到達。→ 訪問済み Block をメモしつつ再帰変換。

#### バグ(P3)

- **TP-3** [separate.py:118-131](qamomile/circuit/transpiler/passes/separate.py) — `lower_measure_qfixed` がトップレベルのみ走査し、For/If 内の `MeasureQFixedOperation` を分割しない(CLAUDE.md の契約がネスト時に不適用)。
- **TP-4** [analyze.py:431-434](qamomile/circuit/transpiler/passes/analyze.py) — AnalyzePass が冪等でない(ANALYZED 再入力で例外)。「全パス冪等」という文書化済み契約と矛盾。
- **TP-5** [classical_lowering.py:86-90](qamomile/circuit/transpiler/passes/classical_lowering.py) — 前提条件チェックが `if ...: pass` の no-op。
- **TP-6** [separate.py:223-228](qamomile/circuit/transpiler/passes/separate.py) — SegmentationPass.run に BlockKind 前提条件が一切ない。
- **TP-7** [affine_validate.py:21-26](qamomile/circuit/transpiler/passes/affine_validate.py) — docstring が実装にない検査(silent discard)を謳う。ループ内消費→ループ後再消費も検出されない(コメントで自認)。
- **TP-8** constant_fold.py:756-793 vs analyze.py:659-691 — ControlledU power 検証が2箇所に重複し、素の `ValueError` と `ValidationError` で例外型が不一致。
- **TP-9** [value_resolver.py:103-105](qamomile/circuit/transpiler/value_resolver.py) — スカラ名前解決に SSA バージョンガードがない(配列側 :164-169 にはある)。#354 系 stale-binding の再発リスク。
- **TP-10** [compile_time_if_lowering.py:680-694](qamomile/circuit/transpiler/passes/compile_time_if_lowering.py) — `_eliminate_dead_ops` に副作用(純粋性)検査がなく、uuid を持たない results だけの op は空生成器→True で削除される穴。
- **TP-11** [inline.py:96 vs :57](qamomile/circuit/transpiler/passes/inline.py) — `count_call_blocks` と `_has_any_call_block` の block=None の扱いが不一致で、誤解を招くエラーに帰着。
- **TP-12** [inline.py:195, 555](qamomile/circuit/transpiler/passes/inline.py) — インライン時のブロック出力マッピングが単段 dict 参照(オペランド側の `ValueSubstitutor(transitive=True)` と非対称)で、stale な shape dim UUID が残る余地。
- **TP-13** [compile_time_if_lowering.py:437-446](qamomile/circuit/transpiler/passes/compile_time_if_lowering.py) — `_apply_substitution` が一般 op の results を置換せず、`SliceArray`/`Cast` だけ個別パッチの積み重ね。
- **TP-14** [symbolic_shape_validation.py:37-49, 93-96](qamomile/circuit/transpiler/passes/symbolic_shape_validation.py) — shape dim 判定が `Value.name` の正規表現(`^(.+)_dim(\d+)$`)依存で、ユーザ命名 `x_dim0` を誤検出し得る。kind≠ANALYZED では黙って検証放棄。
- **TP-15** [validate_while.py:13](qamomile/circuit/transpiler/passes/validate_while.py) — docstring の実行位置(analyze前)と実態(plan内部)の乖離。

#### 非効率

- **TP-16(P2)** analyze.py:446 / classical_lowering.py:92-94 / separate.py:251-254 — 依存グラフ+測定テイントを3パスが独立に再計算(+`reject_self_referential_loop_stores` 2回で実質**5回**のグラフ構築)。→ `AnalysisResult` をパス間で受け渡す。
- **TP-17(P2)** [analyze.py:744-764](qamomile/circuit/transpiler/passes/analyze.py) — `_depends_on_measurement` の DFS は先行集合チェックの部分集合しか探索できず**論理的に常に False を返すデッドコード**なのに、量子opの古典オペランドごとに後方依存錐を全探索。→ 削除。
- **TP-18(P3)** partial_eval.py:75-82 — if-lowering が無変更でも ConstantFolding を無条件2回実行。
- **TP-19(P3)** [transpiler.py:330-336](qamomile/circuit/transpiler/transpiler.py) — `unroll_recursion` が kind を進めるためだけに inline をフル再実行(`dataclasses.replace(kind=AFFINE)` で足りる)。
- **TP-20(P3)** compile_time_if_lowering.py:671-699 — DCE が1 op 削除ごとに used 集合を全再構築+自己再帰(O(N²)+再帰深度リスク)。
- **TP-21(P3)** separate.py:683-796 — absorbable 固定点の全再走査と `_effective_kind` のメモ化なし再帰(最悪 O(N²×深さ))。
- **TP-22(P3)** constant_fold.py:341-349 / compile_time_if_lowering.py:116-123 — オペランド解決のたびに ValueResolver を新規生成。
- **TP-23(P3)** separate.py:855-869 — セグメント output_refs が全 results の過大近似(docstring と乖離)。
- **TP-24(P3)** parameter_shape_resolution.py:144-152 — IfOperation の phi_ops を二重置換(IR-19 と同根)。

#### リファクタリング

- **TP-25(P2)** [slice_borrow_check.py](qamomile/circuit/transpiler/passes/slice_borrow_check.py)(2121行)の分割 — BoundToken 記号区間証明器 / スナップショットガード / 所有権ステートマシン / エラー整形の4責務が1クラスに同居。200行の7分岐メソッドあり。`is_destructive = True` の残骸変数と到達不能 `elif`(:667-668)も削除。→ `slice_borrow/` サブパッケージへ。
- **TP-26(P2)** — 走査ボイラープレートの手書き重複(inline.py / compile_time_if_lowering.py / analyze.py / substitution.py / validate_while.py / symbolic_shape_validation.py / separate.py の7箇所以上)。`control_flow_visitor.py` の基盤があるのに未使用で、IfOperation の isinstance 特別扱いはプロジェクト規約とも摩擦。→ `OperationTransformer` にスコープ付き状態と phi 認識を追加して移設。パターン①のバグ族を閉じる。
- **TP-27(P3)** — デッドモジュール: `compile_check.py`(`is_block_compilable`)、`capabilities.py`(参照ゼロ)。
- **TP-28(P3)** [transpiler.py:129-158, 558-568](qamomile/circuit/transpiler/transpiler.py) — docstring パイプラインの乖離+パスを可変**クラス属性**で共有(状態を持つパスを置いた瞬間に競合バグになる構造)。
- **TP-29(P3)** — 値解決・畳み込みロジックの残存重複(constant_fold.py:303-318 vs value_resolver.py:160-171、constant_fold.py:408-509 vs value_mapping.py:681-793)。

### 4.4 Emit・実行ランタイム(`qamomile/circuit/transpiler/` 実行系 + `passes/emit_support/`)

#### バグ

- **EM-1(P2)** [program_orchestrator.py:74-88](qamomile/circuit/transpiler/program_orchestrator.py)、[job.py:64-82](qamomile/circuit/transpiler/job.py) — `convert_counts` が変換後の同一出力値をマージせず、値が縮退するカーネル(`return s[0]` 等)で `most_common()`/`probabilities()` が誤った分布を返す。→ Counter で集約。
- **EM-2(P2)** [gate_emission.py:223](qamomile/circuit/transpiler/passes/emit_support/gate_emission.py) — 回転角が全解決経路で失敗した場合に **0.0 を無音で焼き込む**(docstring 自身が「silent 0.0 hazard」と呼ぶ事象の残存経路)。→ theta 非 None なら EmitError に昇格。
- **EM-3(P3)** [controlled_emission.py:1951-1957 ほか](qamomile/circuit/transpiler/passes/emit_support/controlled_emission.py) — `gate_controlled`/`gate_power` の None 戻り値を未チェックで `append_gate` する経路が4箇所(composite 経路 :2231-2247 はチェック済み)。新バックエンド追加時に無音ゲート欠落になる構造。
- **EM-4(P3)** [program_orchestrator.py:102-108](qamomile/circuit/transpiler/program_orchestrator.py) — `run()` の Job 型判定がプレ量子古典ステップも数え、「古典前処理+expval」が `ExpvalJob` にならない。
- **EM-5(P3)** program_orchestrator.py:278-291 — `array_name` フォールバックで非展開型のユーザー束縛が配列要素パラメータに丸ごと渡り得る。
- **EM-6(P3)** [classical_executor.py:546-571](qamomile/circuit/transpiler/classical_executor.py) — 値解決順序が emit 側リゾルバ(constant → param名 → UUID → 名前、「歴史的ドリフト防止のため一元化」と明記)と逆(名前が定数より先)。
- **EM-7(P3)** measurement_emission.py:66-78, 128-134 / cast_binop_emission.py:57-66 — 測定・cast の解決失敗が warning のみで実行継続し、counts から当該ビットが静かに欠落。
- **EM-8(P3)** controlled_emission.py:2374-2381 — `blockvalue_to_gate` が `RuntimeError`(「compiler bug; please report it」と自称する内部不変条件違反を含む)まで握りつぶして無音フォールバック。
- **EM-9(P3)** [job.py:148-160](qamomile/circuit/transpiler/job.py) — `RunJob.result()` が None 結果をキャッシュできず converter を再実行。
- **EM-10(P3)** standard_emit.py:131-137 / passes/emit.py:127-137 — emit パスインスタンスが再利用不可(`_parameter_map`・bindings への書き込み蓄積)。冪等契約に反する暗黙の使い捨て。

#### 非効率

- **EM-11(P2)** [controlled_emission.py:2283-2381](qamomile/circuit/transpiler/passes/emit_support/controlled_emission.py) — reusable-gate 非対応バックエンド(QURI/CUDA-Q)で捨て回路をフル emit してから `circuit_to_gate`=None が判明し、ゲート単位で再 emit(二重 emit)。inverse_emission は事前チェック済みで非対称。QURI のオーバーライド回避策は「probe が親 emitter を汚染し得る」と docstring 自身が述べる。→ `supports_reusable_gates()` の事前チェック。
- **EM-12(P3)** controlled_emission.py:83-88, 733, 2124 — QPE の controlled-U^(2^k) フォールバックが counting n 本で 2^n−1 回ブロック再 emit(単一回転ゲートの角度スケーリング検出すらない)。
- **EM-13(P3)** [control_flow_emission.py:245-250, 288](qamomile/circuit/transpiler/passes/emit_support/control_flow_emission.py)、emit_context.py:322-342 — unroll 反復ごとに EmitContext の計8 dict を全コピー(二重ループで実質 O(N²))+スロットとフラット dict への二重書き込み。→ push/pop 方式。
- **EM-14(P3)** [loop_analyzer.py:57-71](qamomile/circuit/transpiler/passes/emit_support/loop_analyzer.py) — `should_unroll` が本体を4回フル走査し、外側 unroll の反復ごとに再実行。→ 1回走査に融合+`id(op)` メモ化。
- **EM-15(P3)** pauli_evolve_emission.py:158-187 — 非 controlled 版の Hamiltonian 二重走査(controlled 版は融合済みと明記)。
- **EM-16(P3)** program_orchestrator.py:76-85 — ポスト量子古典ステップ不在時もユニークビット文字列ごとに `context.copy()`+`ClassicalExecutor()` 生成+全ステップ走査。

#### リファクタリング

- **EM-17(P1・最重要)** [controlled_emission.py](qamomile/circuit/transpiler/passes/emit_support/controlled_emission.py)(3211行)の分割と controlled-U 3経路の統合 — トップレベル3経路(:1826-1971 / :1446-1667 / :1670-1823)が同一のテールをコピペ変奏し、結果マッピングも4箇所に重複。**ネスト経路用 `resolve_controlled_u_call`(:479-637)が既に3レイアウトを統一済み**でトップレベルだけ未移行。EM-3 の null チェック漏れはこの構造の直接の産物。→ 統合後、`controlled_call_resolution` / `controlled_walker` / `multi_control_gates` / `nested_block_binding` の4〜5モジュール(各400〜700行)に分割。
- **EM-18(P2)** — controlled walker の3重実装: 共有版(:182-354)/ QURI Parts(transpiler.py:73-810、約700行)/ CUDA-Q(transpiler.py:1484-2714、約1200行)。差分は葉のゲート lowering のみで、修正を常に3回要求する構造。→ 共有 walker 1本+`ControlledGateLowerer` ストラテジ。
- **EM-19(P2)** — 同名2つの `ValueResolver` の重複: [transpiler/value_resolver.py:109-273](qamomile/circuit/transpiler/value_resolver.py)(寛容: 失敗=None)vs [passes/emit_support/value_resolver.py:518-707](qamomile/circuit/transpiler/passes/emit_support/value_resolver.py)(厳格: 失敗=EmitError)。配列要素解決・slice アフィン合成・数値正規化がほぼ二重実装。`resolve_angle` は回転ゲート1個ごとに両方を生成・呼び出し。→ FailurePolicy パラメータ付き単一実装へ。
- **EM-20(P3)** — デッドコード群(参照ゼロを grep 確認済み): `transpiler/result.py:1-85`(EmitResult 一式)、`transpiler/capabilities.py` 全体、`gate_emitter.py:37-129`(GateKind/GateSpec/GATE_SPECS)、`standard_emit.py:128`(未使用 CompositeDecomposer → `composite_decomposer.py` ごと削除可)、`program_orchestrator.py:161-179`(`_validate_bindings`)、`execution_context.py:22-23`(`get_many`)、`executable.py:78`(`num_output_bits`)、`compile_check.py`、`segments.py:50-53`。
- **EM-21(P3)** program_orchestrator.py:27-31 — `if __builtins__:` の無意味なインポートガード(通常の `if TYPE_CHECKING:` に)。
- **EM-22(P3)** resource_allocator.py:829-858 vs cast_binop_emission.py:31-66 — cast エイリアス処理の二重実装。
- **EM-23(P3)** controlled_emission.py:2384-2429 vs emit_support/value_resolver.py:460-481 — ネストブロック入力束縛が UUID キー方式と名前キー方式の2系統(後者は文書化された規約に違反)。
- **EM-24(P3)** resource_allocator.py:487-501 / standard_emit.py:147 / orchestrator↔classical_executor — `f"{uuid}_{i}"` / `_dim0` の文字列規約依存の残存(`QubitAddress` の導入意図に反する)。`re.match` は `re.fullmatch` へ。
- **EM-25(P3)** program_orchestrator.py:357-364 — `hamiltonian._num_qubits` への私有属性直接代入(公開 API `with_num_qubits(n)` を observable 側に)。

### 4.5 バックエンド(`qamomile/{qiskit,quri_parts,cudaq,qbraid}/`)

#### バグ

- **BE-1(P2)** [cudaq/observable.py:44, 49](qamomile/cudaq/observable.py) — `math.isclose` が complex 係数で `TypeError`。`Hamiltonian.constant` は `float | complex` で、Pauli 積経由では値が実数でも complex 型になる。Qiskit/QURI は `abs()` で正常 → バックエンド間乖離(`# type: ignore` が既に警告していた)。→ `abs() > 1e-15` に統一+complex 係数のクロスバックエンドテスト。
- **BE-2(P2)** [quri_parts/transpiler.py:1093-1174](qamomile/quri_parts/transpiler.py) — inverse プローブ失敗時に emitter 側は復元するが**パス側 `_parameter_map` を復元せず**、破棄回路の Parameter がキャッシュ残留 → フォールバック時に別回路の Parameter が渡る/プローブが常時失敗。→ finally で復元、根本的には EM-11 のプロトコル解決。
- **BE-3(P2)** [qiskit/emitter.py:211-339](qamomile/qiskit/emitter.py) — 制御フローコンテキストヘルパー3クラスが共有パスの契約(eager 型)と不整合(`__enter__` されないコンテキストマネージャを返す)。現在は Pass 上書きで到達不能だが、「emitter が True と申告する能力を共有パスが使うと壊れる」矛盾状態。→ 契約を明文化しヘルパーを修正または削除。
- **BE-4(P3)** [cudaq/emitter.py:854-924](qamomile/cudaq/emitter.py) — `emit_multi_controlled_p/rx/ry/rz` が `_qref` を使わず `q[i]` 直書き(ヘルパーカーネル生成文脈で存在しない参照を emit)。ネスト controlled+定数項 pauli_evolve で発火。→ 4メソッドを `self._qref(...)` へ(1行×4)。
- **BE-5(P3)** [qiskit/transpiler.py:859-864](qamomile/qiskit/transpiler.py) — `bind_parameters` が欠落パラメータを無言スキップ(QURI/CUDA-Q は例外)。qbraid/executor.py:288-292 も同型。
- **BE-6(P3)** qiskit/transpiler.py:918-932 — estimator V2/V1 判別の try/except が実行成功後の結果アクセス例外まで飲み込み二重実行し得る。
- **BE-7(P3)** [quri_parts/transpiler.py:1511](qamomile/quri_parts/transpiler.py) — `params or []` が長さ2以上の ndarray で ValueError。
- **BE-8(P3)** cudaq/transpiler.py:2836-2837 — STATIC counts の `zfill` が逆側パディング(`ljust` が正)。
- **BE-9(P3)** qiskit/transpiler.py:556-560 — `_emit_pauli_evolve` の Hamiltonian 解決だけが name 優先(同ファイルの「UUID 優先」方針と不一致)。
- **BE-10(P3)** [quri_parts/emitter.py:635-644](qamomile/quri_parts/emitter.py) — `append_gate` が全例外を warning で握りつぶし「フォールバックする」と言いながらフォールバックせず続行(ゲート無音欠落)。
- **BE-11(P3)** [qbraid/executor.py:201-206](qamomile/qbraid/executor.py) — タイムアウト時にリモートジョブが未キャンセルで残る(課金・キュー占有)。

#### 非効率

- **BE-12(P2)** [qiskit/transpiler.py:838-840](qamomile/qiskit/transpiler.py) — `execute` が毎回 `qiskit.transpile` をフル実行(+`_ensure_measurements` の毎回コピー)。QAOA 等の最適化ループで支配的コスト。→ 未束縛回路で1回だけ transpile し、`assign_parameters` を transpile 後回路に。
- **BE-13(P2)** qiskit:905 / quri:1506 / cudaq:2990 + program_orchestrator.py:337-366 — expval 経路で Hamiltonian→backend observable 変換を最適化イテレーションごとに再実行。→ `CompiledExpvalSegment` に変換済みオペレータをキャッシュ。
- **BE-14(P3)** cudaq/transpiler.py:2784-2796 — `_ensure_target` が毎実行 `cudaq.set_target`(現 target 照会なし)。
- **BE-15(P3)** qiskit/transpiler.py:814 — `AerSimulator(max_parallel_threads=1)` 固定で解除手段なし。
- **BE-16(P3)** quri_parts/emitter.py:57-102 — ゲート角度演算のホットパスで毎回 import 文。
- **BE-17(P3)** qbraid/executor.py:397-411 — expval が基底グループごとに直列サブミット(バッチ submit 可能)。

#### リファクタリング

- **BE-18(P2)** [cudaq/transpiler.py:96-591, 1804-2751](qamomile/cudaq/transpiler.py) — **約1,440行がデッドコード**(旧 controlled walker 一族。唯一の外部参照はテスト1本の直呼び)。本番経路は `_emit_controlled_fallback` に移行済み。デッドコード内には「現役なら P1 級」のバグ(identity 因子を CX ラダーに混入、許容誤差のハードコード不一致)が眠る。→ 一括削除でファイルが 3047→約1,600行に半減。
- **BE-19(P3)** transpiler/capabilities.py — `BackendCapability`/`CapableBackend` が完全未使用(実際の能力機構は `supports_*()`)。二重の能力表現は新規バックエンド実装者を迷わせる。
- **BE-20(P2)** — controlled fallback walker の重複(EM-18 と同一指摘のバックエンド側)。QURI の約740行は共有 walker+葉フック化で数十行に縮む。
- **BE-21(P3)** — CP/CH/CY/CRX 等の分解が decompositions.py のレシピと各バックエンドのインライン実装で三重メンテナンス(一致は目視のみ)。→ レシピを解釈実行してユニタリ比較する conformance テストを1本追加。デッド import(quri emitter.py:16-23)と幻の `emit_decomposition` 参照(cudaq emitter.py:781, 795)を削除。
- **BE-22(P3)** — executor API の非対称: seed は QURI のみ、bound artifact + `params` 併用時に CUDA-Q だけ無言無視。→ `executor(seed=...)` を共通シグネチャに、併用は ValueError に。
- **BE-23(P3)** — `QBraidExecutor` と `QiskitExecutor` の `_ensure_measurements`/`bind_parameters` が逐語コピー。→ 共通 mixin へ。
- **BE-24(P3)** qiskit/transpiler.py:146-153 — `_emit_if` が空の else ブロックを常に生成。

---

## 5. リファクタリングロードマップ(推奨)

### Phase 0: P1 修正(即時・各独立PR)
P1-1〜P1-5 の単独修正+回帰テスト化。EM-1(convert_counts 集約)、BE-1(complex 係数)も同時期に。
回帰テスト例: 同一サブカーネル2回呼び出し→clbit 2本 / 束縛済み出力の sample 結果 / `content_hash` の2回ビルド一致 / グローバル同名変数の if 分岐 / `Bit == Bit` の実行時分岐。

### Phase 1: デッドコード一括削除(約2,000行、挙動不変)
BE-18(cudaq 約1,440行)、EM-20(capabilities / compile_check / result.py / GateSpec / CompositeDecomposer ほか)、BE-3 の到達不能ヘルパー。公開 API 面は `__init__.py` 再エクスポートを確認の上、1〜2コミットで。

### Phase 2: 性能改善
TP-16/TP-17(AnalysisResult 共有+デッド DFS 削除)→ BE-12(Qiskit transpile キャッシュ)→ BE-13(observable 変換キャッシュ)→ FE-6(特殊化再トレースキャッシュ)→ IR-12(serialize の value_table スコープ分割)→ EM-13/EM-14(EmitContext push/pop、LoopAnalyzer メモ化)。

### Phase 3: 構造リファクタリング(依存順)
1. TP-26: 走査基盤の統一(`OperationTransformer` 拡張)— パターン①を閉じる
2. IR-17: Value 深部 walk の単一化(`map_value_tree`)— パターン②を閉じる
3. EM-19: ValueResolver の統合(FailurePolicy パラメータ化)— EM-6 の解決順序ドリフトも同時に解消
4. EM-18/BE-20: controlled walker のコアへの吊り上げ(`ControlledBodyWalker` + 葉ゲート lowering フック)— EM-11 の probe 問題もプロトコルレベルで解消
5. EM-17: controlled_emission.py の3経路統合と4〜5モジュール分割
6. TP-25: slice_borrow_check.py のサブパッケージ分割

### Phase 4: 契約・API 整流
TP-4/5/6/8(Pass 基底への `expected_kinds` 一元化+例外階層統一+冪等スモークテスト)、BE-5/BE-22(executor API 対称化: seed / 欠落パラメータ / bound+params)、IR-16(serialize テーブル駆動化)、FE-3 残件(Handle 比較プロトコル統一)、TP-28/CLAUDE.md(docstring・文書の実装同期)。

### 検証方法
- 各 Phase で `uv run pytest tests/` 全ユニット+対象領域のクロスバックエンド実行テスト(sampling + expval 両経路)
- Phase 3 の等価性確認には canonical `content_hash`(P1-5 修正後)と3バックエンド期待値クロスチェック(`np.allclose`, atol=1e-8)を利用
- Phase 1 の削除は grep による参照ゼロ再確認+全テストグリーンを条件とする

---

# 追補1: 共通化観点の追加調査(2026-07-03)

**観点**: 「同じ種類のモジュールは可能な限り共通化されたコンポーネント、あるいは共通化された処理フローで記述されるべき」という設計原則に照らした精査。本編の既出項目(controlled walker 3重実装、ValueResolver 2重実装、Value walk 4重実装、走査ボイラープレート、serialize テーブル駆動化、executor API 非対称など)は除外リストとして調査エージェントに渡し、**新規事項のみ**を収集した。5領域(バックエンド / コアemit・実行 / フロントエンド / IR消費側=estimator・visualization / IR層)の並列調査で**新規46件**。全件 file:line をコードで検証済み。CE-1(最重要)は本編作成者が再度コードで裏取り済み。

## 追補の横断的結論

1. **estimator/ と visualization/ はトランスパイラ共有基盤をゼロimport**(grep確認: `eval_utils`/`control_flow_visitor`/両`ValueResolver`/`loop_analyzer`/`resolve_root_qubit_address` いずれも参照0件。唯一の再利用は `CompileTimeIfLoweringPass` のみ)。結果、値解決チェーンは**リポジトリ全体で5系統**、スライスアフィン合成は4系統、ループアンロール実行は3系統に増殖している。本編の範囲外だったこの2パッケージが、共通化欠如の最大の未計上領域だった。
2. **「resolverコールバックを注入して評価ロジックを共有する」パターン(`fold_classical_op` の `FoldPolicy`)は既に存在するのに水平展開されていない**。emit時とruntime時で同じIR opの解釈が独立実装され、既に意味論が乖離した実例がある(CE-1)。
3. **知識テーブルの多重定義**が系統的に存在する: 回転ゲート集合×3、ゲート逆元表×2、演算子表示シンボル×2、Handle↔IR型対応×7、controls/targets/paramsレイアウト解析×4。「新ゲート/新opを1箇所追加すれば全層に行き渡る」構造になっていない。
4. **ドリフトは理論上のリスクではなく観測事実**: CE-1(stopデフォルト1 vs 0)、XC-2(スライス合成の契約ガードがvisualization版のみ欠落)、XC-4(transpilerが根絶した名前ベースのループ変数同一性をestimator/visualizationが再導入)、XC-7(zip整列バグが3サブシステムで独立に3回修正されている)、BE2-3(PauliEvolve前処理がQiskitだけslice view非対応)、FE2-1(特殊化ガードがcontrol側だけ追加条件を持つ)。

## 新規発見一覧

### A. バックエンド間(BE2-1〜BE2-7)

- **BE2-1(P2)** `hamiltonian_to_*` 変換の反復スケルトン三重実装 — [qiskit/observable.py:40-67](qamomile/qiskit/observable.py) / [quri_parts/observable.py:46-70](qamomile/quri_parts/observable.py) / [cudaq/observable.py:35-75](qamomile/cudaq/observable.py)。「定数項閾値判定 → 項ループ → Pauli変換 → 恒等項特別扱い」の同一骨格が独立実装され、既に5軸で乖離(ゼロ係数スキップはCUDA-Qのみ、恒等因子スキップはquri/cudaqのみ、重複ラベルマージはQURIのみ、定数項加算位置が不一致、閾値が生リテラル×2+定数×1に分裂しコメントで「qiskit側と合わせている」と結合)。本編BE-1のcomplex係数バグはこの三重化の産物。→ core に `convert_hamiltonian(h, sink: PauliTermSink)` +バックエンドは~10行のsink実装のみ。
- **BE2-2(P3)** GateEmitterの「ゲート1個=メソッド1個」×3バックエンド+core側のmatchテーブル分裂 — プロトコル約25メソッド([gate_emitter.py:189-337](qamomile/circuit/transpiler/gate_emitter.py))を各バックエンドが1行写経で実装し、core側は [gate_emission.py:105-155](qamomile/circuit/transpiler/passes/emit_support/gate_emission.py) と [controlled_emission.py:761-790, 1113-1120](qamomile/circuit/transpiler/passes/emit_support/controlled_emission.py) に**同じ写像の別テーブルが2系統**。QURIの回転6メソッドは規則的フォークの反復、CUDA-Qの `emit_multi_controlled_{p,rx,ry,rz}` はゲート名以外同一の4連。→ 放棄されたGATE_SPECSの形(kind+num_qubits+has_angle)を唯一のゲート形状ソースとして復活させ、`emit_gate(circuit, kind, qubits, angle=None)` 1本+バックエンド宣言テーブルへ。新ゲート追加が「enum+spec+各テーブル1行」になる。
- **BE2-3(P2)** PauliEvolveOp emitの前処理・検証・結果マッピングの三重実装 — core [pauli_evolve_emission.py:102-259](qamomile/circuit/transpiler/passes/emit_support/pauli_evolve_emission.py) / Qiskit native [qiskit/transpiler.py:518-680](qamomile/qiskit/transpiler.py) / CUDA-Q native [cudaq/transpiler.py:1201-1338](qamomile/cudaq/transpiler.py)。ガジェット分解自体はcore一本化済みだが、Hamiltonian解決→gamma解決→幅検証→エルミート検証→slice解決→結果再登録の前後処理が3複製(エルミート検証のエラー文言まで逐語一致、cudaq側には「mirroring the shared bookkeeping」と写経自認コメント)。観測済みドリフト4軸: Hamiltonian解決手段が同一ファイル内でも分裂(qiskit nativeだけ手書きbindings参照)、qiskit nativeだけslice view非対応(必ずgadgetフォールバックに落ちる)、結果登録がdirectのみ(core/cudaqはdirect+slice-root)、定数項検証文言の分岐。→ `resolve_pauli_evolve()` + `map_pauli_evolve_results()` の2関数抽出で3サイトは項ループ本体のみに。
- **BE2-4(P3)** `_blockvalue_to_gate` no-opオーバーライドの二重化 — [quri_parts/transpiler.py:894-926](qamomile/quri_parts/transpiler.py) と [cudaq/transpiler.py:1155-1179](qamomile/cudaq/transpiler.py) が同じ理由(probe汚染防止)・同じ `return None` を並存。判定ヘルパ `_emitter_supports_reusable_gates` は既存(controlled_emission.py:2270)で、inverse側は使用済み。本編EM-11のpre-gate提案の具体化: 1行追加で両オーバーライド約60行が削除でき、`supports_reusable_gates` の定義有無がQURI/CUDA-Q間で非対称な点も同時に是正。純粋なsuper()委譲の `CudaqEmitPass._emit_custom_composite`(cudaq/transpiler.py:1181-1199)も削除可。
- **BE2-5(P3)** `_emit_if` のコンパイル時定数ガードの二重化 — [qiskit/transpiler.py:134-140](qamomile/qiskit/transpiler.py) と [cudaq/transpiler.py:1133-1138](qamomile/cudaq/transpiler.py) の冒頭がコメントまで同一。→ `StandardEmitPass._emit_if` をtemplate method化(定数条件はbaseが畳み、`_emit_runtime_if` フックのみバックエンド実装)。
- **BE2-6(P3)** Transpiler/EmitPassサブクラスの生成ボイラープレート三重化 — `_create_segmentation_pass` は3つとも本体 `return SegmentationPass()` のみ(abstractのせいで強制コピー)、`_create_emit_pass` と EmitPass `__init__`(emitter生成→composite準備→super)も平行構造。→ abstractをやめてデフォルト実装化+`emit_pass_cls` クラス属性化。
- **BE2-7(P2)** countsのcanonical big-endian契約を5通りの手書きで実装 — 契約([quantum_executor.py:70-72](qamomile/circuit/transpiler/quantum_executor.py))に対し、Qiskit=**無加工素通し**(pad/検証なし)、QURI=intキー→固定幅二進、CUDA-Q sample=zfill+反転、CUDA-Q run=bool行→反転連結、qBraid=空白除去+zfill+検証、と正規化が5系統。qBraidが実地で必要としたガードが他に無い。→ `transpiler/counts.py` に `counts_from_int_keys` / `counts_from_bit_rows` / `canonicalize_bitstring_counts` を新設し5サイトを各1行に。エンディアン規約のテストを1モジュールに集約。

### B. コアemit/実行(CE-1〜CE-10)

- **CE-1(P2、実質バグ)** emit時とruntime時の制御フロー評価の二重実装と**既に発生している意味論乖離** — `resolve_loop_bounds`([control_flow_emission.py:107-110](qamomile/circuit/transpiler/passes/emit_support/control_flow_emission.py): 欠損stopのデフォルト**1**)vs `_execute_for`([classical_executor.py:467-470](qamomile/circuit/transpiler/classical_executor.py): デフォルト**0**)【本編作成者がコードで再検証済み】。`step==0` ガードもruntime側のみ(emit側は生の `ValueError`)。phi選択も condition_resolution.py:149 と classical_executor.py:197 で二重実装。同じ `ForOperation` がunroll経路と解釈経路で挙動が変わり得る。→ `eval_utils` に `resolve_for_bounds(op, resolve)` / `select_phi_input(phi, condition)` を追加し、emit/runtime双方がresolver注入で共用(`fold_classical_op` パターンの水平展開)。
- **CE-2(P2)** 例外階層からの逸脱 — `MultipleQuantumSegmentsError`([segments.py:117](qamomile/circuit/transpiler/segments.py))と `SignatureCompatibilityError`(substitution.py:37)が素の `Exception` 直系で `except QamomileCompileError` に掛からない。`Job.result()` のdocstringは `ExecutionError` を宣言するのに実装は `RuntimeError`(job.py:158)— **契約と実装の矛盾が顕在化**。emit/実行経路のビルトインraiseが計8箇所(emit.py:244,250,290 / gate_emission.py:155 / control_flow_emission.py:235,279 / program_orchestrator.py:296)。→ 2クラスを errors.py 階層へ移設+ビルトインraiseをEmitError/ExecutionErrorに統一。
- **CE-3(P3)** plan.stepsのisinstanceチェーン散在(計11箇所)と**未使用のSegmentKind** — EmitPass.run と orchestrator の4メソッドが `isinstance(step, ...)` チェーンと「量子ステップ前/後」分割を各自手書きする一方、ディスパッチ用に用意された `SegmentKind` enum は消費者ゼロ。→ `ProgramPlan` にクエリAPI(`pre_quantum_steps()`/`post_quantum_steps()`/`has_expval()`)を集約。
- **CE-4(P3)** 測定emit+STATIC記録の3連逐語コピー(measurement_emission.py:63-65, 159-161, 200-202)とゲート結果登録エピローグの重複(gate_emission.py:158-160 ≡ composite_gate_emission.py:426-428)。→ `emit_one_measurement()` / `map_results_positional()` に抽出。
- **CE-5(P3)** 量子オペランド解決プロローグの診断品質分岐 — `emit_gate` はリッチ診断(`QubitIndexResolutionError`+修正提案)だがその構築ループ自体が**同一関数内で2回コピペ**(gate_emission.py:48-67 / 73-96)、composite と pauli_evolve は素朴な `EmitError`。同じ失敗にop種別で診断品質が変わる。→ `resolve_qubit_operands(..., op_label)` に一元化し全プロローグ置換。
- **CE-6(P3)** Orchestrator 3メソッドの同一3行プロローグ、shot変換クロージャの5手順二重実装(sample vs run)、Job 3クラスの status/キャッシュ手書き。→ `_begin_execution()` / `_convert_one_shot()` / `CompletedJob` 基底に集約。
- **CE-7(P3)** op型ディスパッチチェーンの二重手書き — `_emit_operations`(20分岐)と `_execute_operation`(13分岐)で**9つのop型が両方に登場**し、catch-allの例外型が不一致(`NotImplementedError` vs `ExecutionError`)。バックエンドのオーバーライドシーム7個は全てop型と1:1対応でレジストリ化に素直に載る。→ `OpHandlerRegistry`(op型→ハンドラ辞書)を両者で共用。
- **CE-8(P3)** EmitContext typed-writeのduck-typing分岐(`getattr(bindings, "set_value", None)` パターン)が6箇所に散在 — ヘルパ `_set_emit_value` は既存だがcast_binop_emissionローカルに閉じている。→ emit_context.py にモジュール関数として公開し6箇所+バックエンドを差し替え。
- **CE-9(P3)** ClassicalExecutorが共有 `fold_classical_op` を使わず4兄弟ラッパー(_execute_binop/compop/notop/condop)を手書き — emit側は既にresolver注入型を使用しており、runtime側だけ旧様式。→ 単一の `_execute_classical_op` に統合し `fold_classical_op` へ委譲。
- **CE-10(P3)** EmitPass/StandardEmitPass二層の形骸化 — 具象派生は全てStandardEmitPass系のみで、基底が派生専用属性を `getattr(self, "_measurement_qubit_map", {})`(emit.py:190)で覗く(層境界の崩壊)。Hamiltonian束縛解決+型検査も emit.py:284-294 と pauli_evolve_emission.py:124-130 で二重実装(例外型も `RuntimeError`/`TypeError` vs `EmitError` に分裂)。→ `resolve_hamiltonian_binding()` 新設+層統合または正式フック化。

### C. フロントエンド(FE2-1〜FE2-11)

- **FE2-1(P2)** 特殊化ブロック選択フローの3重実装 — 「`_specializing` ガード→`_extract_calltime_specialization`→try/finallyで `_build_specialized`→失敗時 `self.block`」の同一振り付けが [qkernel.py:836-850](qamomile/circuit/frontend/qkernel.py) / [control.py:496-514](qamomile/circuit/frontend/operation/control.py) / [inverse.py:1491-1507](qamomile/circuit/frontend/operation/inverse.py)。可変状態を伴うコピーで、control側だけ追加ガード(all-Handleチェック)を持ち既に挙動分岐(仕様かコピー漏れか判別不能)。→ `QKernel.specialized_block_for(arguments)` に一本化。
- **FE2-2(P2)** 呼び出しプロローグ(bind→apply_defaults→リテラル昇格→Handle検証)の3重実装 — qkernel.py:691-716 / inverse.py:1455-1480 / control.py:746-772(+unknown-kwargs検査が control.py 内で同文2回)。→ `param_validation.bind_call_arguments()` に集約。
- **FE2-3(P2)** 結果Handle再ラップ+VectorView借用移譲の**5重実装** — `VectorView._wrap_unregistered(...)` + `_transfer_borrow_to(...)` の対(引数リストまで同一)が qkernel.py:943-950 / control.py:1167-1174 / inverse.py:1591-1598 / control_flow.py:202-211 / pauli_evolve.py:123-130。affine型システムの正しさを担うコードの並走であり、仕様変更時に1箇所漏れると「静かな借用リーク/二重消費」に直結。→ `handle/utils.py` に `rewrap_result(template, value, operation_name)` を新設し5箇所を置換。
- **FE2-4(P3)** ゲート逆元知識の2表並立 — 同一ファイル内(inverse.py)で `GateOperationType` キーの表(:61-90)とフロント関数キーの表(:1777-1808)が二重管理。新ゲート追加時に3箇所同期が必要。→ `GATE_INVERSE_SPECS` 単一表から導出生成。
- **FE2-5(P2)** Handle型↔IR型対応のスカラーディスパッチ**7重定義** — func_to_block.py(TYPE_MAPPING / handle_type_map / create_dummy_handle)、param_validation.py、qkernel.py(_create_parameter_input / _create_bound_input)、control_flow.py(_create_handle_from_value / _value_to_ir_value)に同型対応が分散。本編FE-15(constructorsの `with_parameter` 欠落)はこの表の分裂が根因。→ `handle/registry.py` に `HANDLE_SPECS` 単一表+4関数(`make_parameter_handle`/`make_const_handle`/`ir_type_of`/`handle_class_for`)。
- **FE2-6(P2)** シグネチャ・型ヒント解決フローの3重実装 — qkernel.py:395-429 / func_to_block.py:501-540 / control.py:1817-1840。しかも `func_to_block` は `QKernel` が解析した直後の同じ関数を**丸ごと再解析**し、例外捕捉方針も3者で異なる(controlのコメントが乖離を自認)。→ `frontend/signature.py` の `resolve_kernel_signature()` に一本化し、解析済み結果を引き回す。
- **FE2-7(P3)** 「量子パラメータ判定」の手書き再実装5箇所 — 既存の共有述語 `param_validation._is_quantum_param_decl` があるのに qkernel.py×4 + func_to_block.py×1 が同じ2段判定をインライン展開。→ 既存部品への収斂のみ(新規コンポーネント不要)。
- **FE2-8(P2)** stdlib QFT: 標準分解ボディとリソース式の逐語重複 — `QFT._decompose`(qft.py:111-144)≡ `StandardQFTStrategy.decompose`(qft_strategies.py:56-86)、IQFT同様、リソース式(`n*(n-1)//2` 等)は qft.py×2 / qft_strategies.py×2 / qpe.py×1 の**5コピー**。ストラテジ登録済みなのにフォールバックが同アルゴリズムを別実装。→ `_decompose` をストラテジ委譲化+`standard_qft_resources(n)` 1関数に。
- **FE2-9(P3)** 戦略レジストリ機構の並立 — `CompositeGate._strategies` クラスレジストリ(composite_gate.py:74-137)と `decomposition.StrategyRegistry`+グローバルレジストリ(decomposition.py:137-255)の2系統。後者の生産コード利用は**ゼロ**(grep確認: 利用者はテスト1本のみ)。なお `frontend/decomposition.py` と `transpiler/decompositions.py` はレシピ重複なし(役割が別、紛らわしい命名のみ)。→ どちらかを唯一の登録機構と宣言し他方を委譲化または削除。
- **FE2-10(P3)** 制御フロービルダー4種(while_loop/for_loop/emit_if/for_items)の「親トレーサ取得→子Tracer→trace()→op構築→operand詰め→親へadd」骨格反復。→ `_nested_trace()` コンテキストマネージャ+`_emit_to_parent()` に共通化。
- **FE2-11(P3)** inverse.py内の古典演算クローンヘルパ同文二重定義 — `_clone_classical_operation`(:586-604)と `_clone_forward_operation`(:1079-1098)の本体5行が完全同一。→ 1関数に統合。

### D. IR消費側: estimator / visualization(XC-1〜XC-10)— 本編の範囲外だった領域

- **XC-1(P2)** 値解決チェーンの再実装(リポジトリ計**5系統**)— transpilerの2つ(既出)に加え、[estimator/_resolver.py:223-295](qamomile/circuit/estimator/_resolver.py)(sympy出力)と [visualization/analyzer.py:2918-3023](qamomile/circuit/visualization/analyzer.py)が「定数→パラメータ名→コンテキスト→名前→配列要素」の同一解決順序を独立実装。キーの流儀も三分裂(transpiler=UUID/名前、estimator=UUID+名前+Symbol、visualization=**logical_id**+`_loop_名前`)。→ `ir/value_resolution.py` に解決コアを置き、出力ドメイン(Python数値/sympy/表示)をアダプタ化。
- **XC-2(P2)** スライスチェーンaffine合成の4実装目+**契約ガード欠落のドリフト実証** — analyzer.py:3561-3629 の `_resolve_view_chain_to_root` は `root = start + step*i` 合成の4実装目だが、transpiler側3実装が全て持つ「`start<0 or step<=0` 拒否」ガードが**ない**。→ `ir/value.py` の共有実装にresolverコールバック引数を足して集約、analyzer版を削除。
- **XC-3(P2)** ループ境界解決+アンロール実行の3系統実装 — estimator(_loop_executor.py:33-76+_engine.py:278-320)/ visualization(analyzer.py:4080-4138, 1738-1761, 845-876 — analyzer内だけでも2本)/ emit(control_flow_emission.py)。反復回数の計算式も三重。→ 共有 `LoopIterationPlan`(bounds解決+iterations+for_each)に畳み、3者は「1反復に何をするか」だけ渡す。
- **XC-4(P2)** ループ変数同一性の方針ドリフト — transpilerは loop_analyzer.py:8-15 で「**名前フォールバック禁止**(名前比較はネストループ衝突バグの温床だった)」と明記しUUIDキーに統一済みだが、estimator(`v.name in self._loop_var_names`)と visualization(`_loop_{name}` キー)は名前ベースを独自実装し、**根絶済みのバグ土壌を再導入**している。→ XC-3の共有部品にUUIDキー束縛を内蔵。
- **XC-5(P2)** 制御フロー走査ディスパッチ骨格の5重実装 — gate_counter.py:104-196 / qubits_counter.py:272-480 / analyzer.py:979-1145 / analyzer.py:252-937 が「For→スコープ×反復 / While→係数 / If→両分岐結合(max/和)」の同一骨格を持つ。既存 `ControlFlowVisitor` は盲目的再帰のみでスコープ生成・分岐結合ができず、誰も使えずに代替が乱立。→ 集約型走査エンジン `fold_operations(ops, handlers, combine_if, scale_for, scale_while)` を追加(gate_counter/qubits_counterだけで約250行削減見込み)。
- **XC-6(P2)** 古典演算評価ディスパッチの3重実装 — eval_utils は自ら「single source of truth」を宣言するのに、analyzer.py:2997-3021 が同じ具象評価(ゼロ除算→None、MIN含む)をelifチェーンで丸ごと再実装。estimatorの sympy 対応表も Kind→演算子の3つ目の平行定義。→ analyzer は `evaluate_binop_values` 呼び出しに即置換可(挙動同一)。
- **XC-7(P2)** callee実引数バインディング(formal→actual zip+const/shape伝播)の3系統実装と**同一バグの独立3回修正** — emit `bind_block_params` / estimator `call_child_scope` / visualization `_build_block_value_mappings`+3分岐。「量子/古典が交互のシグネチャでzipがずれる」同一バグの修正コメントが qubits_counter.py:148-153、analyzer.py:4155-4160、analyzer.py:672-682 に3回出現。→ `bind_call_scope(block, actuals, resolve)` を ir/block.py 近傍に1定義。
- **XC-8(P3)** 回転ゲート集合の三重定義 — gate.py:41(`_ROTATION_GATES` frozenset)/ _catalog.py:44(小文字文字列set)/ analyzer.py:4622(ローカル再定義)。文字列版はtypoが静かに素通り。→ gate.py に `GATE_PROPERTIES`(is_rotation/base_qubits等)を一元定義。
- **XC-9(P3)** 演算子表示シンボル表の重複 — analyzer.py:168-188 と printer.py:264-305 でBinOp/CompOp表が完全一致(CondOpのみ意図的方言差)。→ arithmetic_operations.py に正準表を1つ。
- **XC-10(P3)** QFT/IQFTゲート数公式(_catalog.py:282-304)と実分解構造(composite_decomposer.py:10-41)の乖離リスク — 見積り式は分解系列の閉形式を手写ししたもので、結合テストなし。→ 整合テスト追加、本筋は `resource_metadata` 経由で estimator の QFT 特例分岐ごと削除。

### E. IR層(IR2-1〜IR2-8)

- **IR2-1(P2)** controls→targets→paramsオペランドレイアウトの4重実装 — ConcreteControlledU(gate.py:249-261)/ SymbolicControlledU(gate.py:342-355)/ CompositeGateOperation(composite_gate.py:121-136)/ InverseBlockOperation(inverse_block.py:117-152)が同一レイアウト概念を**異なるプロパティ名・異なる終端規則**(カウント分割 vs is_classical()フィルタ vs 「量子でなくなるまで走査」)で実装。inverse_block のdocstringは「CompositeGateのレイアウト規約を共有する」と言いながらコードは非共有。`signature` の `ParamHint` ループも3重手書き。→ 共有 `OperandLayout` 記述子を operation.py に導入し、各opは `layout()` のみ実装。
- **IR2-2(P2)** `all_input_values`/`replace_values` 追加フィールド処理の手書き4連 — ControlledUOperation.power / SymbolicControlledU.num_controls・control_indices / ForOperation.loop_var_value / ForItemsOperation.key_var_values・value_var_value が同一定型(「uuidがmappingにあればisinstanceチェックしてreplace」)を各自実装。ForOperationのdocstring自身が「override忘れはUUIDRemapperのフィールド取り残し」ハザードを自認。→ `Operation.EXTRA_VALUE_FIELDS: ClassVar[tuple[str, ...]]` 宣言レジストリを基底に追加し、4形状(Value / Value|None / tuple / int|Value)を基底が一括処理。override忘れが構造的に不可能になる。
- **IR2-3(P2)** printerのper-op構造知識がserialize/encodeと二重管理 — printer.py:141-261 のisinstanceチェーン+15分岐が、encode.py:1032-1112 と同じ知識(どのopがネスト本体を持つか、Forのoperands[0..2]の意味)を別引き出しで保持。printerは `HasNestedOps` を使っておらず、新制御フローop追加時は黙ってフラット表示に落ちる。→ 本編IR-16のOpSpecテーブルを `ir/op_spec.py` に置き、**serializeとprinterの両方が同一テーブルを参照**する構成へ拡張。
- **IR2-4(P3)** `next_version()` 4実装のフィールド手書き列挙 — Value/ArrayValue/TupleValue/DictValueが全フィールドを手で列挙コピー。本編IR-6のフィールド脱落バグはこの構造の帰結。→ `dataclasses.replace(self, uuid=..., version=...)` ベースの単一実装をミックスインに(未指定フィールドが全保存されるためコピー漏れバグが構造的に消える)。
- **IR2-5(P3)** パラメトリック型の `__eq__`/`__hash__`/`label` 手書き並行実装 — TupleType/DictTypeは同形を各自手書き、QUIntType/QFixedType/ObservableTypeは未定義(→本編IR-4のunhashableバグ)。5クラスの非一様性。→ `ParametricValueType` 基底(`_key()` のみサブクラス実装)に統一。
- **IR2-6(P3)** パスでのBlock全フィールド手書き再構築 — substitution.py:197-206 / parameter_shape_resolution.py:97-106 / inline.py:203-212 が9フィールド列挙で再構築し、**3サイトとも `output_names` を渡していない**(canonical.pyは保存しており非一様)。`param_slots` 追加時に全サイト修正が必要だった構造が残存。→ `dataclasses.replace` への置換(separate.py:59が正しい先例)または `Block.with_updates()`。
- **IR2-7(P3)** メタデータ `with_*` 5メソッドの同一二段ラップ。→ `_with_slot(**slots)` に縮退。
- **IR2-8(P3)** `HasNestedOps` 単一ボディ実装の三重複 — While/For/ForItemsが文字通り同一の `nested_op_lists`/`rebuild_nested` を持つ。→ `SingleBodyNestedOps` ミックスイン。

## ロードマップへの反映

本編のPhase 3(構造リファクタリング)を以下のとおり拡張・優先度調整することを推奨する:

1. **Phase 0への追加**: CE-1(ループ境界デフォルト乖離)は共通化以前に**意味論バグ**として即時修正(`eval_utils.resolve_for_bounds` 抽出と同時に)。CE-2の `Job.result()` docstring/実装矛盾も同時期に。
2. **Phase 3の中核に「統一Value解決コア」(XC-1)を昇格**: 本編のValueResolver統合(EM-19)は「transpilerの2実装の統合」だったが、実際は5実装(+スライス合成4実装、ループ実行3実装、callee束縛3実装)であり、`ir/value_resolution.py` コア+ドメインアダプタ方式に拡張することで estimator/visualization のドリフト系欠陥(XC-2/XC-4)まで構造的に解消できる。
3. **知識テーブルの一元化を独立ワークストリームに**: GATE_PROPERTIES(XC-8+BE2-2)、GATE_INVERSE_SPECS(FE2-4)、HANDLE_SPECS(FE2-5)、KIND_SYMBOLS(XC-9)、OpSpec(IR2-3、serialize+printer共用)、OperandLayout(IR2-1)。いずれも「新ゲート/新opの追加が1箇所で済む」状態を作る同型の作業であり、まとめて実施すると効率が良い。
4. **宣言的プロトコル化**: IR2-2(EXTRA_VALUE_FIELDS)はIR-17(Value walk単一化)と同時に実施すると、「新しいValue保持フィールドの追加」が完全に宣言的になる。
5. **バックエンド共通化の追加項目**: BE2-1(observable変換sink化)・BE2-3(PauliEvolve前処理抽出)・BE2-7(counts正規化モジュール)は、本編のcontrolled walker吊り上げ(EM-18)と同じ「バックエンドは差分のみ実装する」方向の仕上げとして Phase 3 末尾に追加。BE2-4はEM-11の修正に含めて即時実施可能。
6. **estimator/visualizationの位置づけ**: 本編で範囲外としていたが、共通化観点では最大の負債源(XC-1〜XC-7)。Phase 3 の共有部品(値解決コア、LoopIterationPlan、fold_operations、bind_call_scope)を設計する際は、**最初からこの2パッケージを消費者として設計に含める**こと。transpilerだけを見て設計すると同じ再実装が再発する。

---

# 追補2: テスト監査(2026-07-03)

**対象**: `tests/` 全体 — 162ファイル、102,151行、テスト関数約3,500個(pytest収集: ユニット9,231 / docs込み11,491)。3観点(構造・重複 / カバレッジギャップ / 品質・衛生)の並列監査。重要主張のうち「QAOAテストが`.sample()`を呼んでいない」「デッドwalkerをテストが直呼びしている」は本編作成者がコードで再検証済み。

## 総合評価

**土台は健全、問題は「重複」「無音化」「P1領域の空白」に集中している。**

明るい材料(裏取り済み): 未シード乱数**ゼロ**、wall-clock依存**ゼロ**、広域`except Exception`実質ゼロ(3箇所・全て正当)、統計的にフレークするサンプリング検証は検出されず(疑義箇所は再計算の結果9σ以上の余裕)、**既知のP1バグを「仕様」として固定するテストも存在しない**(=P1修正PRはテスト側の抵抗なく着地できる)。xfailは2箇所のみで両方reason付き(うち1つはstrict=True)。

一方で3つの構造問題がある:

1. **P1バグ5件はすべて「テストが存在しない領域」で生きていた** — 各P1の発火シナリオを個別にgrep+精読した結果、5件とも該当テストなし(下表)。テストがバグを固定していないのは幸運だが、裏返せば修正PRに付ける回帰テストは全て新規作成になる。
2. **テストスイートの約20%(約20,000行)がバックエンド別コピペ3ファイルに集中** — 本番コードの横断パターン⑤(3バックエンド並行実装ドリフト)のテスト版。
3. **退行の無音化構造** — `except EmitError → pytest.skip` が16箇所、必須依存qiskitへの`importorskip`が52ファイル。リファクタリング中に最も退行が起きやすい経路で、最も退行が見えにくい。

## P1/P2バグ × テストカバレッジ突合表(全件検索根拠付きで確認)

| 本編の指摘 | カバレッジ | 最接近のテストと不足 |
|---|---|---|
| P1-1 サブカーネル2回呼び出しの結果Value共有 | **ゼロ** | [test_composite_gate.py:1123](tests/circuit/test_composite_gate.py) `test_same_qkernel_called_twice` はパススルー出力(バグ経路に乗らない)+qubit数の構造チェックのみ。Bitを返すサブカーネルの2回呼び出し+実行は全テストに不在 |
| P1-2 グローバル同名ローカルのif分岐消失 | **ゼロ** | shadow系テストは「ループ変数が関数パラメータをshadow」のSyntaxError系のみ。[test_control_flow_if.py:36](tests/circuit/test_control_flow_if.py) は `global_names=set()` を明示的に渡してバグの前提を構造的に回避 |
| P1-3 `Bit.__eq__` 欠落 | **ゼロ** | [test_bit_logical_ops.py](tests/circuit/test_bit_logical_ops.py) は `&` `\|` `~` とbool混在まで手厚いのに **`==`/`!=` だけ1件もない**(隣接演算子は良カバレッジなのに当該演算子だけ穴、という指摘そのままの形) |
| P1-4 束縛済み古典出力がNone | **ゼロ** | constant_foldテストはゲート角検証のみ。実行系の該当テストはProgramPlanを手組み(constant_foldを通らない)+runtime bindings経路(=正しく動く側)のみ |
| P1-5 シンボリック幅content_hash非決定 | **ゼロ** | [test_canonical.py](tests/circuit/ir/test_canonical.py)(39本)は決定性を正しく検証するが具象幅のみ。`grep -rln "QUInt" tests` → **0ファイル**(シンボリック幅レジスタ型は全スイートで一度も生成されていない) |
| EM-1 convert_counts集約漏れ | **ゼロ** | `grep convert_counts` → 0件。sample実行テストの出力は常に相異なる値に変換されるカーネルのみ |
| BE-1 cudaq complex係数 | **ゼロ+非対称** | QURIには [test_observable.py:43](tests/quri_parts/test_observable.py) で `1j` までparametrize済み、**CUDA-Q用observable単体テストファイル自体が不在** |
| TP-2 SubstitutionPassネスト | **ゼロ** | [test_substitution_pass.py](tests/circuit/test_substitution_pass.py)(18本)はトップレベル置換のみ |
| IR-2/IR-3 serialize round-trip失敗2形状 | **ゼロ** | [test_serialize.py](tests/circuit/ir/test_serialize.py)(67本)は充実だが当該2形状は不在(再現済みバグなので通るテストは存在し得ない=未カバー確定) |
| CE-1 ForOperation境界デフォルト乖離 | **ゼロ** | `_execute_for` のテストは存在するが `ForOperation` は常に3オペランド完備で構築。オペランド欠損・`step==0` は emit/runtime とも未テスト |

## 主要指摘(抜粋)

### カバレッジ(T-COV)

- **T-COV-11(P2)QAOAテストがCLAUDE.md必須要件に全面非準拠**【再検証済み】 — [test_qaoa.py](tests/circuit/algorithm/test_qaoa.py) は `importorskip("qiskit")` のみで**Qiskit単独**。致命的なのは `test_qaoa_state_sample`(:285)/`test_hubo_qaoa_state_sample`(:336)が docstring で「transpile, sample, verify」と言いながら**実際は `.sample()` を一度も呼ばず**ゲート数比較で終わっている点。expval経路ゼロ、seed乱数化ゼロ、QURI/CUDA-Qゼロ。FQAOAもqiskit transpile+quri sampling 1本のみ。**Trotterはほぼ準拠**(3バックエンドstatevector一致+seed付きランダムHamiltonianまで実施、expvalのみ欠け)、**QFT/IQFTは準拠**(`test_392_cross_backend_sampling/expval` が3バックエンド×seed×サイズ)、QPEはsamplingのみ準拠でexpvalゼロ。
- **サブシステム別の空白**: `analyze` パスの専用テストなし — **`pytest.raises(DependencyError)` は全スイート0件**(measurement-taint検証が一度も発火していない)。`InliningError` も0件。`ProgramOrchestrator` 直接テスト0、`convert_counts` 0、`LoopAnalyzer` 専用テストなし(しかも [test_theta_handling.py:8](tests/transpiler/test_theta_handling.py) のdocstringが参照する `test_emit_support.py` は**存在しないファイル**)。inlineパスも専用ファイルなし(間接カバーのみ)。厚いのは: affine/borrow系(216本+119本、例外型まで検証)、compile_time_if_lowering(51本)、canonical(39本)、serialize(67本)、バックエンドfrontend実行(約1,530本)。
- **エラーパスの偏り**: 例外型assertは77+あるがAffineTypeErrorファミリーに集中(QubitConsumedError 55 / QubitRebindError 49)。DependencyError/InliningError/EntrypointValidationErrorは未発火。

### 構造・重複(T-STR)

- **T-STR-1(P2)バックエンド別frontendテスト3ファイルの大規模コピペ(約20,000行)** — test_qiskit_frontend.py(8,614行・316本)/ test_quri_parts_frontend.py(6,990行・264本)/ test_cudaq_frontend.py(4,580行・187本)。同名テストは qiskit∩quri **184本**、3ファイル共通 **59本**。`test_bell_state` はカーネル定義・期待値とも完全同一、シードリスト10連まで一致するコピペ変奏あり。差分は先頭のヘルパ(`_transpile_and_get_circuit`/`_run_statevector`)だけで、まさにアダプタに閉じ込められる構造。cudaqはシナリオ写経が追いつかず共通テスト数が少ない(=コピペ方式の帰結としてのカバレッジ非対称)。→ 共通シナリオ+`FrontendBackendAdapter` 方式へ(エミッタ層の `TranspilerTestSuite`(base_test.py:25)が既に完成している良い先例で、その設計思想がfrontend層に届いていないだけ)。
- **T-STR-2(P2)3バックエンドparametrized fixtureの8箇所再実装** — 正規版 `sdk_transpiler`([tests/circuit/conftest.py:35](tests/circuit/conftest.py))があるのに、`backend`×4 / `sv_backend`×2 / `sdk_backend`×1 の別名ローカル再定義が7箇所(cudaq隔離コメントまで複製)。正規版の採用は4ファイルのみ。→ root conftestへ昇格+executor付き `SdkTranspilerCase` に統一。
- **T-STR-3(P2)algorithm/stdlibテスト内のバックエンド別コピペ関数/クラス** — `test_qiskit`/`test_quri_parts`/`test_cudaq` の3連(transpiler getter 1語違い)が computational_basis_state×3クラス、mottonen×3クラス(**同一ファイル内で `sdk_transpiler` を正しく使う新パターンと混在**)、trotter のゲート数3連。
- **T-STR-4(P2)ボイラープレートの手書き重複** — `_run_statevector` 系10定義(正規版conftest:73があるのに)、`_counts` 6定義+生ループ28箇所、`qiskit_transpiler` fixture 9箇所再定義(**test_shape_propagation.py は同一ファイル内に3回**)、`QiskitTranspiler()` 直接生成403箇所。
- **T-STR-6(P3)許容誤差・シード・shots規約の散在** — atolマジックナンバー8種(1e-10×133〜0.15×2)、`_TOLERANCE` が1e-10と1e-8で別ファイル定義、統計的に正しい `_shot_noise_tolerance(p, shots)`(mottonen)と根拠不明の生定数(0.05/0.1/0.15)が併存、シードリスト6流派。→ `tests/_tolerances.py` に集約。
- **T-STR-7(P3)** バックエンドテスト配置が3方式併存(QURIは tests/quri_parts/ と tests/transpiler/backends/ に分裂し、同じモジュールを別階層で二重テスト)。**T-STR-5(P3)** SDK可用性チェックブロックの5ファイル複製(「mirrors test_gate_broadcast.py」と自認コメント付き)。**T-STR-8(P3)** カーネル/ゲート行列カタログ3系統のうち2つがほぼ死蔵(tests/utils.py の行列群は利用0、qkernel_catalog は利用2ファイルに対し bell手書きが10箇所以上)。**T-STR-9(P3)** テストdocstring欠落389本(約11%、CLAUDE.md規約違反。ワースト: test_affine_types.py 71本、test_printer.py/test_runtime_execution.py は全滅)。

### 品質・衛生(T-QUAL)

- **T-QUAL-1(P2)デッドwalker固定テスト3本**【再検証済み】 — [test_cudaq.py:94-137](tests/transpiler/backends/test_cudaq.py) のヘルパが `emit_pass._emit_cudaq_controlled_ops(...)` を直呼び+私設メソッドをモンキーパッチし、`TestCudaqControlledSliceElementFallback` の3本が使用。**本編BE-18のデッドコード約1,440行削除を直接ブロックする唯一のテスト群**(テスト名は "controlled_fallback" だが実際は旧walkerをテストしており名前も誤解を招く)。しかも `pytest.mark.cudaq` 配下なのでデフォルトCIでは走らず、削除時に「ローカル緑・cudaqレーンだけ赤」の形で顕在化する。→ 3本が検証するslice/ループ意味論を生存経路(`_emit_controlled_fallback` 経由end-to-end)のテストに書き換えてから削除。
- **T-QUAL-2(P2)`StrategyRegistry`(本番参照ゼロ)固定テスト1クラス** — [test_decomposition_strategy.py:47-78](tests/circuit/test_decomposition_strategy.py)。追補1 FE2-9の削除をブロック。
- **朗報**: 上記2件以外、**本編EM-20/BE-19が挙げた全デッドモジュール(capabilities / compile_check / EmitResult / GateSpec / CompositeDecomposer / _validate_bindings / get_many / num_output_bits)はテスト参照ゼロ**をgrepで全数確認 — Phase 1の削除はテスト修正ほぼなしで通る。ただし [gate_test_specs.py:3](tests/transpiler/gate_test_specs.py) のdocstringが本番の GateKind を文言参照しており、削除PRで文言更新が必要。
- **T-QUAL-4(P2)`except EmitError: pytest.skip` 16箇所(全て test_controlled.py)** — 「今日通っているバックエンドが明日EmitErrorを投げるようになっても失敗ではなくskipになる」構造。controlled-U経路はまさにEM-17/EM-18の大規模リファクタリング対象であり、**リファクタリング中の退行がこの16箇所で無音化される**。→ 期待マトリクス方式(`XFAIL_MATRIX = {("cudaq", "CSWAP"), ...}`)に改め、マトリクス外のEmitErrorは失敗させる。
- **T-QUAL-8(P2)必須依存qiskitへのモジュールレベル`importorskip`が52ファイル** — qiskitはpyproject.tomlの必須依存であり、環境破損時にスイートの半分近くが**エラーではなく無音skip**になりCIが緑のまま検証ゼロになり得る。→ 素の `import qiskit` に置換(collection errorとして顕在化)。
- **T-QUAL-6(P3)** [_cudaq_source_assertions.py](tests/transpiler/backends/_cudaq_source_assertions.py)(607行)がCUDA-Q emitterの生成ソースをテスト側で**丸ごと再実装**して文字列一致検証(「Mirror ``CudaqKernelEmitter._angle_expr``」と自認)— 本番の並行実装ドリフトパターン⑤のテスト版。emitterを1行変えるたびにミラーも要変更。→ 実行意味論検証+要所パターンマッチへ縮小。
- **T-QUAL-3(P3)** 私設ヘルパー直叩きのホワイトボックステスト約11群(生存コードなのでデッドコード固定ではないが、EM-17/EM-18/TP-25のリファクタリング時に一斉改修が必要になる一覧をレポート化済み)。**T-QUAL-13(P3)** `slow` マーカー不在のまま重量級テスト(shots=16384×90回超のSDK実行、50シードグリッド等)が常時実行。**T-QUAL-5(P3)** 自明アサーションのみ181本(約5%、多くはdocstringでスモーク宣言済みのため許容範囲)。
- **設定の注意点**: pyproject.toml の `addopts = "-m 'not docs and not quri_parts and not cudaq'"` に対し、(1) CLAUDE.mdの「デフォルト=docsスキップ」という記述は不正確(quri/cudaqも除外)、(2) マーカー運用が不均一で、cudaqインストール環境ではskipifガード方式のテスト経由でデフォルト実行にもcudaqケースが走る(「デフォルト=cudaqなし」は実は不成立)。

## ロードマップへの反映(テスト観点)

1. **Phase 0(P1修正)**: 回帰テストは全て新規作成が必要。追補2のT-COV-1〜10の「提案」欄が受け入れテスト仕様としてそのまま使える(例: `test_subkernel_double_call_distinct_results`、`test_bound_classical_output_survives_fold`、`test_bit_equality_emits_runtime_comparison`、`test_for_bounds_defaults_agree_between_emit_and_runtime`)。
2. **Phase 1(デッドコード削除)の前提タスク**: T-QUAL-1の3本を生存経路end-to-endテストへ移植(cudaq CIレーンで確認)+T-QUAL-2の1クラス削除。それ以外の削除対象はテスト参照ゼロ確認済みで即日削除可能。
3. **Phase 3(controlled walker共通化)の前工程**: (a) T-STR-2のfixture一本化 →(b) frontend 3ファイルの共通59本+2ファイル共通分を共通スイートへ吊り上げ(T-STR-1/T-QUAL-12)→(c) T-QUAL-4のskipパターンを期待マトリクス化。この順で行わないと、walker共通化の修正を3重適用しながら退行が無音化される最悪の組み合わせになる。
4. **CLAUDE.md準拠の是正**: QAOA(+FQAOA)に `sdk_transpiler` 版のsampling+expvalテストを追加(T-COV-11)、Trotter/QPEにexpval経路を追加。`analyze`(DependencyError発火)とProgramOrchestrator/convert_countsの直接テストを新設。
5. **衛生の一括是正(小PR向き)**: qiskit importorskip 52ファイルの置換、docstring欠落ワースト5ファイルの追記、`slow` マーカー導入、`tests/_tolerances.py` 新設。
