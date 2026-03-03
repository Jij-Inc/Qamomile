# Resource Estimation 設計仕様書

Qamomile で構築した量子回路の代数的リソース推定を提供するモジュールである。
全推定値は SymPy シンボリック式で表され、問題サイズパラメータに依存する式として扱える。

本仕様書は二つのパートで構成される。

- **Part 1 — カウント規則**: 量子ビットカウント・ゲートカウント・デプスカウントそれぞれの規則を、全 Operation 種別・制御構造・シンボリックケースについて網羅的に記述する。
- **Part 2 — 関数リファレンス**: 各ファイル・関数が何をしているかを簡潔に記述する。

---

# Part 1 — カウント規則

## 1. データ構造

### 1.1 GateCount（gate_counter.py）

ゲート数の内訳を保持する。全フィールドは `sp.Expr`。

| フィールド | 意味 |
|---|---|
| `total` | 全ゲート数 |
| `single_qubit` | 1量子ビットゲート数 |
| `two_qubit` | 2量子ビットゲート数 |
| `multi_qubit` | 3量子ビット以上のゲート数 |
| `t_gates` | T/Tdg ゲート数（フォールトトレラント計算のコスト指標） |
| `clifford_gates` | Clifford ゲート数 |
| `rotation_gates` | 回転ゲート数（パラメータ付きゲート） |
| `oracle_calls` | `dict[str, sp.Expr]`: オラクル呼び出し回数（スタブゲート追跡用） |
| `oracle_queries` | `dict[str, sp.Expr]`: オラクルクエリ数（query_complexity による重み付け） |

演算: `+`（加算）、`*`（スカラー倍）、`.max()`（要素ごと最大）、`.simplify()`、`.zero()`

### 1.2 CircuitDepth（depth_estimator.py）

回路深さの内訳を保持する。全フィールドは `sp.Expr`。

| フィールド | 意味 |
|---|---|
| `total_depth` | 全ゲートの回路深さ（クリティカルパス長） |
| `t_depth` | T ゲートのみの深さ |
| `two_qubit_depth` | 2量子ビットゲートのみの深さ |
| `multi_qubit_depth` | 3量子ビット以上のゲートのみの深さ |
| `rotation_depth` | 回転ゲートのみの深さ |

演算: `+`（直列合成）、`*`（スカラー倍）、`.max()`（並列合成）、`.simplify()`、`.substitute()`、`.zero()`

**`apply_gate_to_qubits` のフィールド別選択的伝播**: ゲート適用時、各深さフィールドは独立に処理される:
- ゲートが当該フィールドに寄与する（値が非ゼロ）場合: 関与する全量子ビットの当該フィールドの `max` + ゲート寄与 を全量子ビットに設定
- ゲートが当該フィールドに寄与しない（値がゼロ）場合: 各量子ビットは自身の現在値を保持（他の量子ビットに伝播しない）

### 1.3 ResourceEstimate（resource_estimator.py）

3つのメトリクスを統合する。

| フィールド | 型 | 意味 |
|---|---|---|
| `qubits` | `sp.Expr` | 論理量子ビット数 |
| `gates` | `GateCount` | ゲート数内訳 |
| `depth` | `CircuitDepth` | 回路深さ内訳 |
| `parameters` | `dict[str, sp.Symbol]` | 式中のパラメータシンボル |
| `_worst_case_depth_dicts` | `frozenset[str]` | 逐次上界を使用した dict 名の集合（内部用） |

### 1.4 ResourceMetadata（composite_gate.py）

CompositeGateOperation に付与するメタデータ。

各推定器が読み取るフィールド:

```
ResourceMetadata
├── ゲートカウント型付きフィールド → gate_counter
│   total_gates, single_qubit_gates, two_qubit_gates,
│   multi_qubit_gates, t_gates, clifford_gates, rotation_gates
│
├── 深さ型付きフィールド → depth_estimator
│   total_depth, t_depth, two_qubit_depth,
│   multi_qubit_depth, rotation_depth
│
├── ancilla_qubits → qubits_counter
│
├── query_complexity → gate_counter が oracle_queries の重みとして使用
└── custom_metadata → 推定器は直接使用しない（補助情報用）
```

**`None` の扱い**: 型付きフィールドが `None` の場合、推定器は **0 として扱う**。

**整合性チェック（ゲートカウントのみ）**: `total_gates` が設定済みで、`single_qubit_gates`/`two_qubit_gates`/`multi_qubit_gates` のいずれかが `None` かつ既知合計が `total_gates` 未満の場合、`UserWarning` を発行する。

---

## 2. ゲート分類定数

gate_counter.py と depth_estimator.py の両方で定義される定数と、gate_counter.py のみで定義される定数がある。

**共通定義（gate_counter.py・depth_estimator.py の両方）**:

| 定数名 | ゲート |
|---|---|
| `T_GATES` | t, tdg |
| `SINGLE_QUBIT_GATES` | h, x, y, z, s, sdg, t, tdg, rx, ry, rz, p, u, u1, u2, u3 |
| `TWO_QUBIT_GATES` | cx, cy, cz, swap, cp, crx, cry, crz, rzz |
| `ROTATION_GATES` | rx, ry, rz, p, cp, crx, cry, crz, rzz |
| `MULTI_QUBIT_GATES` | toffoli |
| `_GATE_BASE_QUBITS` | toffoli → 3 |

**gate_counter.py のみで定義**:

| 定数名 | ゲート |
|---|---|
| `CLIFFORD_GATES` | h, x, y, z, s, sdg, cx, cy, cz, swap |
| `_CONTROLLED_CLIFFORD_GATES` | x, y, z（1制御の CX/CY/CZ のみ Clifford） |

注意: CP, CRX, CRY, CRZ, RZZ は `TWO_QUBIT_GATES` と `ROTATION_GATES` の両方に属する。

---

## 3. 量子ビットカウント規則（qubits_counter.py）

### 3.1 基本ルール

- **`QInitOperation`**:
  - 結果が `ArrayValue` かつ `QubitType` → 全次元の積を加算
  - 結果が単一 `QubitType` → 1 を加算
  - 結果が非量子型 → 0
- **入力 qubit（`BlockValue.input_values`）**: トップレベル呼び出し（`qubits_counter(block)`）時にのみカウント。再帰的な内部呼び出し（`_count_from_operations`）ではスキップし、二重カウントを防ぐ。

### 3.2 クリーンコール判定

`_is_clean_call(block)` は BlockValue が「クリーンコール」であるかを判定する。

**判定条件**: BlockValue の全ての量子型 `return_values` の `logical_id` が `input_values` の `logical_id` 集合に含まれていること。

**意味**: 内部で確保されたアンシラ qubit は関数終了時に解放され、ループ反復間で再利用可能。

**安全側の判定**: キャスト/エイリアスにより新しい `logical_id` が生成された場合は非クリーン（過大カウントにはなるが過小カウントにはならない）。

### 3.3 ループ本体の persistent/reusable 分割

ループ本体の qubit を「persistent（反復ごとに蓄積）」と「reusable（最大ウォーターマーク、一度だけカウント）」に分割する。

| Operation | persistent への寄与 | reusable への寄与 |
|---|---|---|
| `QInitOperation` | += count | — |
| `CallBlockOperation`（clean） | — | = Max(reusable, inner_alloc) |
| `CallBlockOperation`（non-clean） | += inner_alloc | — |
| `ControlledUOperation`（clean） | — | = Max(reusable, inner_alloc) |
| `ControlledUOperation`（non-clean） | += inner_alloc | — |
| `CompositeGateOperation`（impl + clean） | — | = Max(reusable, impl_alloc + ancilla) |
| `CompositeGateOperation`（stub + metadata） | — | = Max(reusable, ancilla) |
| `CompositeGateOperation`（non-clean） | += alloc | — |
| `ForOperation`（ネスト） | += inner_p × iterations | = Max(reusable, inner_r × [iterations > 0]) |
| `WhileOperation`（ネスト） | += inner_p × `\|while\|` | = Max(reusable, inner_r) |
| `IfOperation`（ネスト） | += Max(true_p, false_p) | = Max(reusable, true_r, false_r) |
| `ForItemsOperation`（ネスト） | += inner_p × `\|dict_name\|` | = Max(reusable, inner_r) |
| `BinOp` | — | — （symbol_map に結果を登録） |

### 3.4 トップレベル Operation ごとの qubit カウント規則

`_count_from_operations` が処理する規則:

| Operation | カウント規則 |
|---|---|
| `QInitOperation` | += `_count_qinit(op)` |
| `ForOperation` | 本体を persistent/reusable に分割 → `persistent × iterations + reusable × [iterations > 0]` |
| `WhileOperation` | 本体を persistent/reusable に分割 → `persistent × \|while\| + reusable`（`\|while\|` は常に正なので reusable は1回カウント） |
| `IfOperation` | `Max(true_count, false_count)` |
| `CallBlockOperation` | 呼び出し先の `operations` を再帰カウント（`input_values` はスキップ） |
| `ControlledUOperation` | 制御ブロックの `operations` を再帰カウント（`input_values` はスキップ） |
| `ForItemsOperation` | 本体を persistent/reusable に分割 → `persistent × \|dict_name\| + reusable` |
| `CompositeGateOperation` | implementation 内の `operations` を再帰カウント + `ancilla_qubits` を加算 |
| `BinOp` | symbol_map に結果を登録（qubit カウントには寄与しない） |

### 3.5 シンボリック値の解決（qubits_counter 固有）

`_resolve_value_to_sympy` の優先順位:

1. 定数（`is_constant()`） → `sp.Integer(int(const_val))`
2. パラメータ（`is_parameter()`） → `sp.Symbol(param_name, integer=True, positive=True)`
3. `symbol_map` に UUID で登録済み → キャッシュ済み式（BinOp の結果等）
4. フォールバック → `sp.Symbol(value.name, integer=True, positive=True)`

**BinOp 結果の登録**: `_register_binop` が BinOp 演算結果を `symbol_map[output.uuid]` に登録。演算マッピングは `BINOP_TO_SYMPY`（`_utils.py` で定義）に基づく。

### 3.6 特殊シンボル

| シンボル | 用途 | assumptions |
|---|---|---|
| `\|while\|` | WhileOperation の反復回数（未知） | `integer=True, positive=True` |
| `\|dict_name\|` | ForItemsOperation の辞書の濃度（未知） | `integer=True, positive=True` |

全モジュール（qubits_counter, gate_counter, depth_estimator）で同じ assumptions を使用している。

**worst-case 推定への影響**: これらのシンボルが推定結果に出現するということは、反復回数や辞書サイズが推定時に未知であることを意味する。特に **深さ推定** においては、`|while|` や `|dict_name|` が現れるとき、内部深さは `_compute_sequential_depth`（全ゲートを直列に並べた保守的上界、§5.11 参照）で算出されるため、実際の並列深さではなく **逐次上界 × シンボル** という worst-case 推定になる。`|dict_name|` が深さ推定に出現した場合は `_worst_case_depth_dicts` に記録され、`substitute()` 時に警告が再発行される（§6.2, §6.3 参照）。量子ビットカウント・ゲートカウントについては、シンボル値が与えられれば正確な式であり、worst-case 性は深さ推定に固有である。

---

## 4. ゲートカウント規則（gate_counter.py）

### 4.1 単一ゲートの分類

#### 制御なし（`num_controls == 0`）

| ゲート | total | single | two | multi | t_gates | clifford | rotation |
|---|---|---|---|---|---|---|---|
| H, X, Y, Z, S, Sdg | 1 | 1 | 0 | 0 | 0 | 1 | 0 |
| T, Tdg | 1 | 1 | 0 | 0 | 1 | 0 | 0 |
| RX, RY, RZ, P | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| U, U1, U2, U3 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| CX, CY, CZ, SWAP | 1 | 0 | 1 | 0 | 0 | 1 | 0 |
| CP, CRX, CRY, CRZ, RZZ | 1 | 0 | 1 | 0 | 0 | 0 | 1 |
| Toffoli | 1 | 0 | 0 | 1 | 0 | 0 | 0 |

#### 制御あり（`num_controls > 0`）

1. **single_qubit**: 常に **0**
2. **t_gates**: 常に **0**（制御付き T/Tdg は T ゲートとして扱わない）
3. **total_qubits** = `num_controls` + ベースゲートの量子ビット数（SINGLE_QUBIT → 1, TWO_QUBIT → 2, toffoli → 3, その他 → 1）
4. `total_qubits == 2` → `two_qubit=1, multi_qubit=0`
5. `total_qubits > 2` → `two_qubit=0, multi_qubit=1`
6. **clifford**: ベースが `{x, y, z}` に属し `num_controls == 1` の場合のみ 1（CX/CY/CZ のみ Clifford）
7. **rotation**: ベースが `ROTATION_GATES` に属する場合 1
8. **symbolic num_controls**: `sp.Piecewise` で条件分岐式を生成（two_qubit, multi_qubit, clifford の各フィールド）

### 4.2 制御構造でのゲートカウント規則

| Operation | 処理 |
|---|---|
| `GateOperation` | `_count_gate_operation(op, num_controls)` で1ゲート分加算 |
| `ForOperation` | §4.3 参照 |
| `WhileOperation` | 内部カウント × `\|while\|` シンボル |
| `IfOperation` | `true_count.max(false_count)` |
| `CallBlockOperation` | 仮引数→実引数マッピング後（配列次元含む）、呼び出し先を再帰カウント |
| `ControlledUOperation` | 不透明ゲートとして1ゲート分加算（§4.5 参照） |
| `ForItemsOperation` | 内部カウント × `\|dict_name\|` シンボル |
| `CompositeGateOperation` | §4.4 の優先順位で処理 |
| `BinOp` / `CompOp` | ゲートカウントには寄与しないが、値解決のトレースに使用される |

### 4.3 ForOperation のゲートカウント

1. ループ変数シンボル `sp.Symbol(loop_var, integer=True, positive=True)` を作成
2. ループ本体用のローカルブロック（スコープ）を作成し、`parent_blocks` にカレントブロックを追加
3. 本体の内部カウントを再帰的に算出
4. **ループ変数に依存する場合**: 内部カウントの `total`, `two_qubit`, `multi_qubit`, `oracle_calls`, `oracle_queries` のいずれかの free_symbols にループ変数が含まれる → `Sum(inner_count, (loop_var, start, stop-1))` 式（逆ループ時は `(loop_var, stop+1, start)`）
5. **ループ変数に依存しない場合**: `inner_count × ((stop - start) / step)`

### 4.4 CompositeGateOperation のゲートカウント

優先順位:

1. **ResourceMetadata**: 型付きフィールドから GateCount を抽出。`total_gates` 未設定時は `single_qubit + two_qubit + multi_qubit` で算出。スタブ（implementation なし）は `oracle_calls` にゲート名と回数（= 1）を自動記録。`query_complexity` が設定されていれば `oracle_queries` にも記録
2. **Implementation**: BlockValue の分解回路を再帰的にカウント（仮引数→実引数のマッピング適用）
3. **既知型（QFT/IQFT）**: 公式を使用
4. **エラー**: いずれもない場合は ValueError

### 4.5 ControlledUOperation（不透明ゲート）

`qmc.controlled(qkernel)` で生成される ControlledUOperation は、内部ブロックを展開せず **1つの不透明ゲート** としてカウントする。

```
GateCount:
  total     = 1
  single    = 0
  two_qubit = 1 if (num_controls + num_targets == 2) else 0
  multi_qubit = 1 if (num_controls + num_targets > 2) else 0
  t_gates   = 0
  clifford  = 0
  rotation  = 0
```

`num_targets` の決定: `controlled_block` が BlockValue の場合は量子型 `input_values` の数。それ以外は `target_operands` の長さ（なければ 1）。

シンボリック `num_controls` の場合は `sp.Piecewise` で `two_qubit`, `multi_qubit` を条件分岐式にする。

### 4.6 スタブゲートの設計

`stub=True` で定義された CompositeGate（implementation なし）はブラックボックスとして扱う。

- 型付きフィールドが `None` → **0 として扱う**
- `oracle_calls` にゲート名と回数（= 1）を自動記録
- `query_complexity` が設定されていれば `oracle_queries` にも記録

### 4.7 値解決（gate_counter 固有）

`value_to_expr` の優先順位:

1. SymPy 式 → そのまま返す
2. 非 Value 型（int, float, bool） → 対応する SymPy 型に変換
3. `call_context` に UUID で登録済み → 再帰的に解決
4. 定数（`is_constant()`） → `sp.Integer` / `sp.Float`
5. パラメータ（`is_parameter()`） → `sp.Symbol(param_name, integer=True, positive=True)`
6. ループ変数 → `loop_var_symbols` から解決
7. 計算値 → `_trace_value_operation()` でカレントブロックの BinOp/CompOp を再帰的にトレース
8. **parent_blocks スコープチェーン** → 祖先ブロックを逆順に辿ってトレース
9. フォールバック → `sp.Symbol(v.name, integer=True, positive=True)`

**BinOp トレースの詳細**（`_trace_value_operation`）:

| BinOpKind | SymPy 変換 |
|---|---|
| ADD | `left + right` |
| SUB | `left - right` |
| MUL | `left * right` |
| DIV | `left / right` |
| FLOORDIV | `sp.simplify(left / right)` → Integer/Symbol/Pow なら `floor()` 省略、それ以外は `sp.floor(left / right)` |
| POW | `left ** right` |

---

## 5. 回路深さ推定規則（depth_estimator.py）

### 5.1 DAG クリティカルパス方式

各量子ビットごとに独立に CircuitDepth を追跡する（`QubitDepthMap = dict[str, CircuitDepth]`）。

ゲート適用時:

```
gate_depth = _estimate_gate_depth(op, num_controls)
qubit_ids = [operand の名前 for operand in op.operands]
# フィールド別選択的伝播（§1.2 参照）
CircuitDepth.apply_gate_to_qubits(qubit_depths, qubit_ids, gate_depth)
```

最終結果: 全量子ビットの深さの `max` → `.simplify()`

### 5.2 エントリポイント

`estimate_depth(block, *, bindings=None)` は2つのパスを持つ:

1. **具体パス**: `bindings` が指定されている場合、`_prepare_concrete_env` で `var_env`, `dict_bindings`, `call_context` を構築し、`_simulate_parallel_depth_concrete` で全操作を具体値でシミュレート。`_UnresolvableForOpError` が発生した場合はシンボリックパスにフォールバック
2. **シンボリックパス**: `_estimate_parallel_depth()` で per-qubit depth tracking を行う

### 5.3 単一ゲートの深さ分類

#### 制御なし

| ゲート | total | t_depth | two_qubit | multi_qubit | rotation |
|---|---|---|---|---|---|
| T, Tdg | 1 | 1 | 0 | 0 | 0 |
| CX, CY, CZ, SWAP | 1 | 0 | 1 | 0 | 0 |
| CP, CRX, CRY, CRZ, RZZ | 1 | 0 | 1 | 0 | 1 |
| Toffoli | 1 | 0 | 0 | 1 | 0 |
| RX, RY, RZ, P | 1 | 0 | 0 | 0 | 1 |
| H, X, Y, Z, S, Sdg, U 等 | 1 | 0 | 0 | 0 | 0 |

#### 制御あり

ゲートカウント（§4.1）と同じ `total_qubits` 分類を使用。`t_depth` は常に 0。

### 5.4 シンボリックパスでの各 Operation の処理

`_estimate_parallel_depth` が処理する規則:

**前処理 1 — BinOp 登録**: 全 BinOp の結果を `symbol_map[output.uuid]` に事前登録する。これにより後続の `_qubit_key()` や `_resolve_value_expr()` で BinOp 結果を SymPy 式として解決できるようになる。

**前処理 2 — フルブロックサンプリング判定**: 以下のいずれかを検出した場合、`_estimate_by_full_block_sampling` でブロック全体を具体サンプリングベースで推定する（§5.6 参照）。

- **BinOp 名前衝突**: 同名の結果を持つ BinOp が複数存在（`_has_binop_name_collisions`）。ForItemsOperation/WhileOperation ガードなし（常にチェック）
- **パラメトリックループと周辺操作の qubit 重複**: パラメトリック ForOp の本体が前後の操作と qubit を共有（`_has_parametric_loop_with_surrounding_overlap`）。**ただし**、前後の操作・ループ本体のいずれかに ForItemsOperation/WhileOperation が含まれる場合はそのループについてスキップ（ガードは `_has_parametric_loop_with_surrounding_overlap` 内部に組み込まれている）

**qubit 名の解決**: GateOperation, MeasureOperation, CompositeGateOperation 等の qubit 名は `_qubit_key(v, symbol_map, ...)` で解決される。`_qubit_key` は配列要素の場合 `element_indices` を `_resolve_value_expr` で SymPy 式に変換し `parent_array.name[expr1,expr2,...]` 形式の正規化された名前を返す。これにより BinOp エイリアシングを回避する。

| Operation | 処理 |
|---|---|
| **GateOperation** | qubit 名を `_qubit_key()` で解決 → `apply_gate_to_qubits(qubit_depths, qubit_ids, gate_depth)` |
| **MeasureOperation** | qubit 名を `_qubit_key()` で解決 → `total_depth += 1` を加算。結果を `value_depths[result.uuid]` に記録 |
| **MeasureVectorOperation** | 配列名に一致する全 qubit に `apply_gate_to_qubits` で `total_depth += 1`。結果を `value_depths` に記録 |
| **MeasureQFixedOperation** | MeasureVectorOperation と同じセマンティクス |
| **NotOp** | `value_depths[output.uuid] = value_depths[input.uuid]`（深さをそのまま伝播） |
| **CondOp** | `value_depths[result.uuid] = max(lhs_depth, rhs_depth)` |
| **CompOp** | `value_depths[result.uuid] = max(lhs_depth, rhs_depth)` |
| **IfOperation** | §5.5 参照 |
| **ForOperation** | §5.6 参照 |
| **WhileOperation** | `_compute_sequential_depth` による内部深さ × `\|while\|` シンボル（`integer=True, positive=True`）。全関与 qubit に `current_max + increase` を設定 |
| **ForItemsOperation** | `_compute_sequential_depth` による内部深さ × `\|dict_name\|` シンボル。全関与 qubit に `current_max + total_depth` を設定 |
| **CallBlockOperation** | §5.8 参照 |
| **ControlledUOperation** | §5.9 参照 |
| **CompositeGateOperation** | qubit 名を `_qubit_key()` で解決し、配列 operand はプレフィックスマッチで要素レベルに展開 → `apply_gate_to_qubits` |

注意: NotOp, CondOp, CompOp の `value_depths` 伝播は IfOperation の condition 深さ追跡に不可欠。

### 5.5 IfOperation の深さ推定

1. `value_depths` から condition の深さを取得（未登録なら zero）
2. 両ブランチの `qubit_depths` をコピー
3. ブランチ内で使用される qubit の深さを `condition_depth` まで bump（`current.max(condition_depth)`）
4. 両ブランチを独立に `_estimate_parallel_depth` で推定
5. 全 qubit について `max(true_depth, false_depth)` を取る
6. `phi_ops` の深さ伝播: `_qubit_key(output)` に `max(true_value_depth, false_value_depth)` を設定

### 5.6 ForOperation の深さ推定

#### 前処理: フルブロックサンプリング

§5.4 の前処理 2 で説明したとおり、BinOp 名前衝突またはパラメトリックループ周辺 qubit 重複を検出した場合にブロック全体をサンプリングベースで推定する。

フルブロックサンプリング（`_estimate_by_full_block_sampling`）:
1. パラメトリックシンボルを特定（`_scan_parametric_for_loops`）
2. n=2〜29 の範囲から有効サンプル点を選択（`_compute_valid_sample_points_for_block`）
3. 各サンプル点でブロック全体を `_simulate_parallel_depth_concrete` で具体シミュレート
4. 最後のサンプル点を検証用に分離
5. `_interpolate_depth` で補間し、検証点で確認
6. 検証失敗時は検証点を含めて再補間

#### 具体ループ（境界が全て定数）

全イテレーションを `_simulate_parallel_depth_concrete` で直接シミュレーションし、per-qubit の深さを正確に追跡する。`_UnresolvableForOpError`（内部 ForOp の境界が解決不能）発生時は、1イテレーション分を `_estimate_parallel_depth` でシンボリック推定し、`iterations` 倍にフォールバック。

#### パラメトリックループ（境界にシンボルを含む）

サンプリング＋補間方式:

1. **サンプル点の選択**: n=2〜20 の範囲からループ反復回数 > 0 となる点を最大 7 点選択（6 点トレーニング + 1 点検証）
2. **具体シミュレーション**: 各サンプル点でパラメータを代入し全イテレーションを `_simulate_parallel_depth_concrete` でシミュレート。`_UnresolvableForOpError` 発生時は1イテレーション分をシンボリック推定し iterations 倍
3. **補間** (`_interpolate_depth`): CircuitDepth フィールドごとに3段階で補間:
   - Leave-one-out 多項式補間（最後の点で検証、次数は自動）
   - 指数+線形形式 `a*2^n + b*n + c`（3点で連立方程式を解き、全点で検証）
   - フル多項式補間（フォールバック）
4. **検証**: 保留した検証点で確認。不一致なら検証点を含めて再補間
5. **フォールバック**: 有効サンプル点が 3 点未満の場合は `_compute_sequential_depth × iterations`

### 5.7 具体パスでの各 Operation の処理

`_simulate_parallel_depth_concrete` が処理する規則（`bindings` 指定時に使用）:

| Operation | 処理 |
|---|---|
| **GateOperation** | qubit 名を `_concretize_qubit_operand` で具体化 → `apply_gate_to_qubits` |
| **BinOp** | 両オペランドを具体評価し、結果を `var_env` に name と uuid の両方で登録 |
| **MeasureOperation** | qubit 名を具体化 → `total_depth += 1` |
| **MeasureVectorOperation** | 配列名に一致する qubit を検索 → `apply_gate_to_qubits` |
| **MeasureQFixedOperation** | MeasureVectorOperation と同じ |
| **NotOp / CondOp / CompOp** | シンボリックパスと同じ `value_depths` 伝播 |
| **IfOperation** | シンボリックパスと同じロジック（condition bump → ブランチ独立推定 → per-qubit max → PhiOp伝播） |
| **ForItemsOperation** | `bindings` で dict エントリを解決できる場合: 具体的に展開し per-qubit 追跡。解決できない場合: `sequential_depth × \|dict_name\|` シンボル（警告を発行） |
| **WhileOperation** | `sequential_depth × \|while\|` シンボル（警告を発行） |
| **ForOperation** | 全境界を具体評価 → 全イテレーションをシミュレート。境界が解決不能なら `_UnresolvableForOpError` |
| **CallBlockOperation** | 仮引数→実引数マッピング（具体化）→ qubit name mapping → local depth map → 再帰 → write back |
| **ControlledUOperation** | 不透明ゲートとして `total_depth=1` を全関与 qubit に加算 |
| **CompositeGateOperation** | `_estimate_composite_gate_depth` で深さ取得 → `var_env` で代入 → 配列 operand を展開 → `apply_gate_to_qubits` |

### 5.8 CallBlockOperation の深さ推定

1. 呼び出し先の `call_context` を構築（仮引数 UUID → 実引数の `value_to_expr` 結果。配列次元も同様にマッピング）
2. `_build_qubit_name_map` で仮 qubit 名 → 実 qubit 名のマッピングを構築
3. `_map_depths_to_local` でローカル深さマップを作成（ベース名フォールバックと配列プレフィックスマッチングを含む）
4. 呼び出し先の `operations` を `_estimate_parallel_depth` で再帰推定
5. `_write_back_depths` でローカル深さを呼び出し元に書き戻し（新規確保された qubit も伝播）

### 5.9 ControlledUOperation の深さ

不透明ゲートとして `total_depth=1` を加算する。

`num_targets` の決定: `controlled_block` が BlockValue の場合は量子型 `input_values` の数。それ以外は **常に 1**（gate_counter と異なり、`target_operands` は参照しない）。

```
CircuitDepth:
  total_depth       = 1
  t_depth           = 0
  two_qubit_depth   = 1 if (num_controls + num_targets == 2) else 0
  multi_qubit_depth = 1 if (num_controls + num_targets > 2) else 0
  rotation_depth    = 0
```

全関与量子ビット（制御 + ターゲット）の現在の深さの `max` に加算。

`has_index_spec` モードの場合: vector operand のシェイプを解決し、全要素の qubit 名を列挙。

### 5.10 CompositeGateOperation の深さ推定

優先順位:

1. **ResourceMetadata**: 型付きフィールドから CircuitDepth を抽出。**制御付きスタブの最小深さルール**: `num_control_qubits > 0` かつ `total_depth == 0` かつ `two_qubit_depth == 0` の場合、`total_depth=1, two_qubit_depth=1` に強制
2. **Implementation**: `_compute_sequential_depth` で分解回路の深さを再帰推定（仮引数→実引数のマッピング適用）
3. **既知型（QFT/IQFT）**: 公式を使用
4. **エラー**: いずれもない場合は ValueError

### 5.11 逐次深さ（`_compute_sequential_depth`）

全ゲートの深さを単純加算する保守的な推定。並列深さ推定のフォールバック、および WhileOperation・ForItemsOperation のシンボリック推定に使用される。

| 操作 | 処理 |
|---|---|
| **GateOperation** | `+= _estimate_gate_depth(op, num_controls)` |
| **Measure 系** | `+= 1`（total_depth のみ） |
| **ForOperation** | 内部深さがループ変数に依存する場合は `Sum(inner, (i, start, stop-1))` 式。依存しない場合は `inner × ((stop − start) / step)` |
| **ForItemsOperation** | 内部深さ × `\|dict_name\|` |
| **IfOperation** | `max(true_depth, false_depth)` |
| **WhileOperation** | 処理しない（呼び出し元で `\|while\|` 倍する） |
| **CallBlockOperation** | 呼び出し先を再帰的に `_compute_sequential_depth` で推定 |
| **ControlledUOperation** | 不透明ゲートとして depth=1 加算（two_qubit/multi_qubit はゲートカウントと同じ分類） |
| **CompositeGateOperation** | §5.10 の優先順位で深さ取得し加算 |

### 5.12 値解決（depth_estimator 固有）

`value_to_expr` の優先順位:

1. SymPy 式 → そのまま返す
2. 非 Value 型（int, float, bool） → 対応する SymPy 型に変換
3. `call_context` に UUID で登録済み → 再帰的に解決
4. 定数（`is_constant()`） → `sp.Integer` / `sp.Float`
5. パラメータ（`is_parameter()`） → `sp.Symbol(param_name, integer=True, positive=True)`
6. ループ変数 → `loop_var_symbols` から解決
7. 計算値 → `_trace_value_operation()` でカレントブロックの BinOp/CompOp を再帰的にトレース
8. フォールバック → `sp.Symbol(v.name, integer=True, positive=True)`

**gate_counter との差異**:

| 項目 | gate_counter | depth_estimator |
|---|---|---|
| `parent_blocks` スコープ | 祖先ブロックを逆順に辿って値を解決 | なし（カレントブロックのみ） |
| FLOORDIV 処理 | `sp.simplify(left / right)` → Integer/Symbol/Pow なら `floor()` 省略 | 常に `sp.floor(left / right)` |
| ForOperation パラメトリック | `Sum` 式またはイテレーション数倍 | サンプリング＋多項式補間 |
| ControlledUOperation `num_targets` | BlockValue でない場合は `target_operands` の長さ（なければ 1） | BlockValue でない場合は常に 1 |

---

## 6. 統合インターフェース（resource_estimator.py）

### 6.1 estimate_resources

3つの推定器を統合する:

1. `qubits_counter(block)` → qubit 数
2. `count_gates(block)` → ゲートカウント
3. `estimate_depth(block, bindings=bindings)` → 回路深さ

### 6.2 bindings による具体実行

`bindings` が指定された場合の追加処理:

1. **深さ推定**: `estimate_depth` が具体パス（`_simulate_parallel_depth_concrete`）を先に試行
2. **worst-case 追跡**: 深さ推定結果に出現した `|dict_name|` シンボルを `_worst_case_depth_dicts` に記録
3. **シンボル代入**: dict は `|dict_name|` → `len(dict_value)` に、スカラーは対応する Symbol に代入。gate_count, circuit_depth, qubit_count の全てに適用
4. **パラメータ収集**: 全式の free_symbols を集めて `parameters` 辞書に格納

### 6.3 substitute の注意点

`substitute()` はシンボルに具体値を代入するが、並列深さ情報を回復できない。正確な深さが必要な場合は推定時に `bindings` を渡す必要がある。`_worst_case_depth_dicts` に含まれる dict の警告は `substitute()` 呼び出し時に再発行される。

---

# Part 2 — 関数リファレンス

## _utils.py

| 関数/定数 | 概要 |
|---|---|
| `BINOP_TO_SYMPY` | BinOpKind → SymPy 演算のマッピング辞書。qubits_counter の `_register_binop` が使用 |
| `_strip_nonneg_max(expr)` | `Max(0, x)` → `x` の正規化。リソース推定値は物理的に非負であるため冗長な Max を除去 |

## qubits_counter.py

| 関数 | 概要 |
|---|---|
| `qubits_counter(block)` | エントリポイント。BlockValue/list[Operation] の qubit 数を返す。BlockValue の場合は input_values + operations 両方をカウント |
| `_count_from_operations(operations, symbol_map)` | Operation リストを走査し qubit 数を算出。ループ/分岐/呼び出しを再帰処理 |
| `_count_loop_body_split(operations, symbol_map)` | ループ本体の qubit を (persistent, reusable) に分割 |
| `_count_qinit(op, symbol_map)` | QInitOperation から qubit 数を抽出（配列なら次元の積、単一なら 1） |
| `_count_input_qubits(input_values, symbol_map)` | BlockValue.input_values の qubit 型をカウント。トップレベルのみ使用 |
| `_is_clean_call(block)` | BlockValue がクリーンコール（入力 qubit のみ返す）かを判定 |
| `_register_binop(op, symbol_map)` | BinOp の結果を symbol_map に登録 |
| `_build_call_block_inner_map(op, symbol_map)` | CallBlockOperation の仮引数→実引数の配列次元マッピングを構築 |
| `_build_controlled_u_inner_map(op, symbol_map)` | ControlledUOperation の仮引数→実引数マッピングを構築（index_spec/default の両モード対応） |
| `_compute_for_iterations(op, symbol_map)` | ForOperation の反復回数を `(stop - start) / step` として算出 |
| `_resolve_for_items_cardinality(op)` | ForItemsOperation の辞書の濃度シンボル `\|dict_name\|` を生成 |
| `_resolve_value_to_sympy(value, symbol_map)` | Value を SymPy 式に変換（定数 → パラメータ → symbol_map → フォールバック） |

## gate_counter.py

| 関数 | 概要 |
|---|---|
| `count_gates(block)` | エントリポイント。BlockValue/Block/list[Operation] のゲート数を返す |
| `_count_from_operations(operations, block, parent_blocks, ...)` | Operation リストを走査しゲート数を算出。ループ/分岐/呼び出し/CompositeGate を再帰処理 |
| `_count_gate_operation(op, num_controls)` | 単一の GateOperation を分類してカウント |
| `_count_composite_gate(op, block, ...)` | CompositeGateOperation のゲート数を優先順位に従って算出 |
| `_extract_gate_count_from_metadata(meta)` | ResourceMetadata の型付きフィールドから GateCount を抽出 |
| `value_to_expr(v, block, call_context, loop_var_symbols, parent_blocks)` | Value を SymPy 式に変換。parent_blocks によるスコープチェーン解決を含む |
| `_trace_value_operation(v, block, visited, ...)` | ブロック内の操作を後方トレースして Value の算出元を特定 |
| `_find_loop_variable_values(operations, loop_var_name)` | ループ変数名と完全一致する Value を検出（計算式は含まない） |
| `_apply_sum_to_count(count, loop_var, start, stop, step)` | GateCount の全フィールドに SymPy `Sum` を適用（逆ループ対応） |

## depth_estimator.py

| 関数 | 概要 |
|---|---|
| `estimate_depth(block, *, bindings)` | エントリポイント。具体パスとシンボリックパスの2段構え |
| `_estimate_parallel_depth(operations, qubit_depths, ...)` | シンボリックパスの中核。per-qubit depth tracking で DAG クリティカルパスを推定 |
| `_simulate_parallel_depth_concrete(operations, qubit_depths, ...)` | 具体パスの中核。全変数を具体値で解決し per-qubit 追跡 |
| `_compute_sequential_depth(operations, ...)` | 全ゲートの深さを単純加算する保守的推定。フォールバック用 |
| `_estimate_gate_depth(op, num_controls)` | 単一の GateOperation の深さを分類 |
| `_estimate_composite_gate_depth(op, block, ...)` | CompositeGateOperation の深さを優先順位に従って算出 |
| `_extract_depth_from_metadata(meta)` | ResourceMetadata の型付きフィールドから CircuitDepth を抽出 |
| `_handle_for_parallel(op, qubit_depths, ...)` | ForOperation の並列深さ推定（具体ループ/パラメトリック/フォールバック） |
| `_handle_call_block_parallel(op, qubit_depths, ...)` | CallBlockOperation の並列深さ推定（name mapping + local depth + write back） |
| `_handle_controlled_u_parallel(op, qubit_depths, ...)` | ControlledUOperation の並列深さ推定（不透明ゲート） |
| `_prepare_for_loop(op, block, ...)` | ForOperation のループ変数・境界・コンテキストを準備 |
| `_interpolate_depth(samples, sym)` | 具体サンプルから CircuitDepth を多項式/指数+線形で補間 |
| `_estimate_by_full_block_sampling(operations, ...)` | ブロック全体を具体サンプリングし補間で深さ推定 |
| `_has_binop_name_collisions(operations)` | BinOp 結果名の衝突を検出 |
| `_has_parametric_loop_with_surrounding_overlap(operations, ...)` | パラメトリックループと周辺操作の qubit 重複を検出 |
| `_scan_parametric_for_loops(operations, ...)` | トップレベルのパラメトリック ForOperation をスキャンしシンボルと境界を返す |
| `_compute_valid_sample_points_for_block(param_sym, loop_bounds, ...)` | 全パラメトリックループで iterations > 0 となるサンプル点を計算 |
| `value_to_expr(v, block, call_context, loop_var_symbols)` | Value を SymPy 式に変換（parent_blocks スコープなし、FLOORDIV は常に `sp.floor()`） |
| `_trace_value_operation(v, block, visited, ...)` | ブロック内の操作を後方トレースして Value の算出元を特定 |
| `_find_loop_variable_values(operations, loop_var_name)` | ループ変数名と完全一致する Value を検出 |
| `_apply_sum_to_depth(depth, loop_var, start, stop, step)` | CircuitDepth の全フィールドに SymPy `Sum` を適用 |
| `_resolve_value_expr(v, symbol_map, ...)` | symbol_map 優先の Value→SymPy 解決。symbol_map に UUID があればそれを返し、なければ `value_to_expr` にフォールバック |
| `_qubit_key(v, symbol_map, ...)` | 正規化された qubit キーを返す。配列要素の場合 `element_indices` を SymPy 式に変換し `parent_array[expr,...]` 形式にする。BinOp エイリアシング回避用 |
| `_get_max_depth(qubit_depths)` | 全 qubit の深さの max を返す |
| `_collect_all_qubit_names(operations, symbol_map)` | 全 Operation から qubit 名を収集。`symbol_map` があれば `_qubit_key` を使用 |
| `_qubit_base_name(name)` | qubit 名を配列ベース名に正規化（`a[0]` → `a`、`a[i]` → `a`、`ancilla` → `ancilla`） |
| `_to_base_qubit_names(names)` | qubit 名集合に `_qubit_base_name` を適用してベース名集合を返す |
| `_build_qubit_name_map(called_block, op_operands, offset)` | 仮 qubit 名 → 実 qubit 名のマッピングを構築 |
| `_map_depths_to_local(qubit_name_map, qubit_depths)` | 実名ベースの深さマップからローカル（仮名ベース）の深さマップを作成 |
| `_write_back_depths(qubit_name_map, local_depths, qubit_depths)` | ローカル深さを呼び出し元のマップに書き戻す |
| `_prepare_concrete_env(block, bindings)` | bindings から var_env, dict_bindings, call_context を構築 |
| `_resolve_dict_entries_for_depth(dict_value, dict_bindings)` | DictValue を具体的な (key, value) ペアに解決 |
| `_concretize_qubit_name(name, var_env)` | qubit 名のシンボリックインデックスを具体値に置換 |
| `_concretize_qubit_operand(v, block, call_context, var_env)` | qubit operand を具体名に解決（element_indices による UUID ベーストレース優先） |
| `_eval_value_concrete(v, block, call_context, var_env)` | Value を具体的な int に評価（解決不能なら None） |
| `_resolve_qubit_base_name(v)` | Value のベース qubit 名を取得（parent_array があればその名前） |
| `_body_has_for_items_or_while(operations)` | 操作リスト内に ForItemsOperation/WhileOperation が含まれるか再帰チェック |

## resource_estimator.py

| 関数/クラス | 概要 |
|---|---|
| `estimate_resources(block, *, bindings)` | エントリポイント。3推定器を統合し ResourceEstimate を返す |
| `ResourceEstimate` | 統合結果のデータクラス。`substitute()`, `simplify()`, `to_dict()`, `__str__()` メソッドを持つ |

---

## ファイル構成

```
qamomile/circuit/estimator/
├── __init__.py               # 公開 API
├── DESIGN.md                 # 本ファイル
├── _utils.py                 # 共有ユーティリティ（BINOP_TO_SYMPY, _strip_nonneg_max）
├── qubits_counter.py         # qubit 数推定
├── gate_counter.py           # ゲート数推定
├── depth_estimator.py        # 回路深さ推定
├── resource_estimator.py     # 統合インターフェース
└── algorithmic/              # アルゴリズム固有の理論推定
    ├── __init__.py
    ├── qaoa.py               # estimate_qaoa, estimate_qaoa_ising
    ├── qpe.py                # estimate_qpe, estimate_eigenvalue_filtering
    └── hamiltonian_simulation.py  # estimate_trotter, estimate_qsvt, estimate_qdrift
```
