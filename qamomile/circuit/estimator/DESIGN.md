# Resource Estimation 設計仕様書

Qamomile で構築した量子回路の代数的リソース推定を提供するモジュールである。
全推定値は SymPy シンボリック式で表され、問題サイズパラメータに依存する式として扱える。

本仕様書は二つのパートで構成される。

- **Part 1 — カウント規則**: 量子ビットカウント・ゲートカウントそれぞれの規則を、全 Operation 種別・制御構造・シンボリックケースについて網羅的に記述する。
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

### 1.2 ResourceEstimate（resource_estimator.py）

2つのメトリクスを統合する。

| フィールド | 型 | 意味 |
|---|---|---|
| `qubits` | `sp.Expr` | 論理量子ビット数 |
| `gates` | `GateCount` | ゲート数内訳 |
| `parameters` | `dict[str, sp.Symbol]` | 式中のパラメータシンボル |

### 1.3 ResourceMetadata（composite_gate.py）

CompositeGateOperation に付与するメタデータ。

各推定器が読み取るフィールド:

```
ResourceMetadata
├── ゲートカウント型付きフィールド → gate_counter
│   total_gates, single_qubit_gates, two_qubit_gates,
│   multi_qubit_gates, t_gates, clifford_gates, rotation_gates
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

全定数は `_catalog.py` に一元化されている。

| 定数名 | ゲート |
|---|---|
| `CLIFFORD_GATES` | h, x, y, z, s, sdg, cx, cy, cz, swap |
| `T_GATES` | t, tdg |
| `SINGLE_QUBIT_GATES` | h, x, y, z, s, sdg, t, tdg, rx, ry, rz, p, u, u1, u2, u3 |
| `TWO_QUBIT_GATES` | cx, cy, cz, swap, cp, crx, cry, crz, rzz |
| `ROTATION_GATES` | rx, ry, rz, p, cp, crx, cry, crz, rzz |
| `MULTI_QUBIT_GATES` | toffoli, ccx（ccx は toffoli に正規化） |
| `_GATE_BASE_QUBITS` | toffoli → 3, ccx → 3 |
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

### 3.5 シンボリック値の解決

全推定器で共有される `ExprResolver`（`_resolver.py`）を使用する。qubits_counter は `resolver.resolve(v)` でシンボリック解決を行う。

`ExprResolver.resolve()` の優先順位:

1. SymPy 式 → そのまま返す
2. 非 Value 型（int, float, bool） → 対応する SymPy 型に変換
3. `context` に UUID で登録済み → 再帰的に解決
4. 定数（`is_constant()`） → `sp.Integer` / `sp.Float`
5. パラメータ（`is_parameter()`） → `sp.Symbol(param_name, integer=True, positive=True)`
6. ループ変数（名前ベース） → `loop_var_names` から解決
7. BinOp/CompOp 結果 → カレントブロックの操作をオンデマンドでトレース
8. `parent_blocks` スコープチェーン → 祖先ブロックを逆順に辿ってトレース
9. フォールバック → `sp.Symbol(v.name, integer=True, positive=True)`

**BinOp のトレース**: ExprResolver は BinOp 結果を事前登録せず、値が必要になった時点でブロック内の操作を順方向走査して算出元を特定する（`_trace` メソッド）。演算マッピングは `_apply_binop`（`_resolver.py`）に基づく。

### 3.6 特殊シンボル

| シンボル | 用途 | assumptions |
|---|---|---|
| `\|while\|` | WhileOperation の反復回数（未知） | `integer=True, positive=True` |
| `\|dict_name\|` | ForItemsOperation の辞書の濃度（未知） | `integer=True, positive=True` |

全モジュール（qubits_counter, gate_counter）で同じ assumptions を使用している。

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

### 4.7 値解決

gate_counter は全推定器で共有される `ExprResolver`（`_resolver.py`）を使用する。解決優先順位は §3.5 と同一。

**BinOp 演算のマッピング**（`_apply_binop` in `_resolver.py`）:

| BinOpKind | SymPy 変換 |
|---|---|
| ADD | `left + right` |
| SUB | `left - right` |
| MUL | `left * right` |
| DIV | `left / right` |
| FLOORDIV | スマート FLOORDIV: `sp.simplify(left / right)` → Integer/Symbol/非負指数Pow なら `floor()` 省略、それ以外は `sp.floor(left / right)` |
| POW | `left ** right` |

**スコープ**: `ExprResolver.child_scope()` により、ループ本体などの内部スコープから親ブロックの値を自動的にトレース可能。`call_child_scope()` は CallBlockOperation 用で、仮引数→実引数マッピング（配列次元含む）を構築し、parent_blocks はリセットされる。

---

## 5. 統合インターフェース（resource_estimator.py）

### 5.1 estimate_resources

2つの推定器を統合する:

1. `qubits_counter(block)` → qubit 数
2. `count_gates(block)` → ゲートカウント

### 5.2 bindings による具体実行

`bindings` が指定された場合の追加処理:

1. **シンボル代入**: dict は `|dict_name|` → `len(dict_value)` に、スカラーは対応する Symbol に代入。gate_count, qubit_count の全てに適用
2. **パラメータ収集**: 全式の free_symbols を集めて `parameters` 辞書に格納

---

# Part 2 — 関数リファレンス

## _utils.py

| 関数/定数 | 概要 |
|---|---|
| `BINOP_TO_SYMPY` | BinOpKind → SymPy 演算のマッピング辞書。`_resolver.py` の `_apply_binop` が参照 |
| `_strip_nonneg_max(expr)` | `Max(0, x)` → `x` の正規化。リソース推定値は物理的に非負であるため冗長な Max を除去 |

## _resolver.py

| 関数/クラス | 概要 |
|---|---|
| `ExprResolver` | 全推定器で共有される値解決クラス。`resolve()` でシンボリック解決、`resolve_concrete()` で具体値解決 |
| `ExprResolver.child_scope(inner_block, ...)` | ループ/分岐用の子スコープを作成。parent_blocks を伝播し外側スコープの値をトレース可能にする |
| `ExprResolver.call_child_scope(call_op)` | CallBlockOperation 用の子スコープ。仮引数→実引数マッピング（配列次元含む）を構築し、parent_blocks はリセット |
| `UnresolvedValueError` | 具体値解決が不可能な場合の例外 |
| `_apply_binop(kind, left, right)` | BinOp 演算適用。FLOORDIV はスマート処理（Integer/Symbol/非負指数 Pow なら `floor()` 省略） |
| `_apply_compop(kind, left, right)` | CompOp 演算適用（EQ, NEQ, LT, LE, GT, GE） |

## _catalog.py

| 関数/定数 | 概要 |
|---|---|
| ゲート集合定数 | `CLIFFORD_GATES`, `T_GATES`, `SINGLE_QUBIT_GATES`, `TWO_QUBIT_GATES`, `ROTATION_GATES`, `MULTI_QUBIT_GATES`, `_GATE_BASE_QUBITS`, `_CONTROLLED_CLIFFORD_GATES` |
| `classify_gate(op, num_controls)` | GateOperation を GateCount に分類（制御あり/なし両対応、symbolic Piecewise 対応） |
| `classify_controlled_u(nc, num_targets)` | ControlledUOperation を不透明ゲートとして GateCount に分類（total=1） |
| `extract_gate_count_from_metadata(meta)` | ResourceMetadata から GateCount を抽出（None→0、整合性警告） |
| `qft_iqft_gate_count(n)` | QFT/IQFT のゲート数公式 |

## _engine.py

| 関数/クラス | 概要 |
|---|---|
| `CompositeGateResolution` | CompositeGate 解決結果のデータクラス（metadata/implementation/qft_iqft/error の4分岐） |
| `resolve_composite_gate(op, resolver)` | CompositeGateOperation のリソースソースを優先順位に従って解決 |
| `resolve_controlled_u(op, resolver)` | ControlledUOperation を `(num_controls, num_targets)` に解決。全推定器で共有 |
| `resolve_for_items_cardinality(op)` | ForItemsOperation の辞書の濃度シンボル `\|dict_name\|` を生成 |
| `build_for_loop_scope(op, resolver)` | ForOperation の子スコープ・境界・ループシンボルを構築 |
| `build_while_scope(op, resolver)` | WhileOperation の子スコープ・トリップカウントシンボルを構築 |
| `build_if_scopes(op, resolver)` | IfOperation の true/false 両ブランチの子スコープを構築 |
| `build_for_items_scope(op, resolver)` | ForItemsOperation の子スコープを構築 |
| `_LocalBlock` | ループ本体用の軽量ブロック stand-in |

## _loop_executor.py

| 関数 | 概要 |
|---|---|
| `try_resolve_range(resolver, start, stop, step)` | ループ境界を具体 int に解決。シンボリックなら None |
| `concrete_range(start, stop, step)` | Python `range()` セマンティクスでリスト生成 |
| `find_parametric_symbols(start, stop, step)` | ループ境界の free_symbols を返す |
| `collect_sample_points(start, stop, step, param_sym, sample_fn, ...)` | パラメトリックループのサンプル点を収集（n=2〜20、iterations > 0 のみ） |
| `interpolate_scalar(sample_points, symbol)` | サンプル点から多項式/指数+線形で補間（holdout 検証付き） |
| `interpolate_fields(field_samples, symbol)` | フィールドごとに独立に `interpolate_scalar` を適用 |
| `symbolic_iterations(start, stop, step)` | `(stop − start) / step` としてシンボリック反復回数を算出 |

## qubits_counter.py

| 関数 | 概要 |
|---|---|
| `qubits_counter(block)` | エントリポイント。BlockValue/list[Operation] の qubit 数を返す。BlockValue の場合は input_values + operations 両方をカウント |
| `_count_from_operations(operations, resolver)` | Operation リストを走査し qubit 数を算出。ループ/分岐/呼び出しを再帰処理 |
| `_count_loop_body_split(operations, resolver)` | ループ本体の qubit を (persistent, reusable) に分割 |
| `_count_qinit(op, resolver)` | QInitOperation から qubit 数を抽出（配列なら次元の積、単一なら 1） |
| `_count_input_qubits(input_values, resolver)` | BlockValue.input_values の qubit 型をカウント。トップレベルのみ使用 |
| `_is_clean_call(block)` | BlockValue がクリーンコール（入力 qubit のみ返す）かを判定 |
| `_build_controlled_u_child_resolver(op, resolver)` | ControlledUOperation の仮引数→実引数マッピングを構築（index_spec/default の両モード対応） |
| `_count_composite_split(op, resolver)` | CompositeGateOperation の qubit 数を (alloc, is_reusable) で返す |
| `_count_composite_total(op, resolver)` | CompositeGateOperation の qubit 数を非分割で返す |

## gate_counter.py

| 関数 | 概要 |
|---|---|
| `count_gates(block)` | エントリポイント。BlockValue/Block/list[Operation] のゲート数を返す |
| `_count_from_operations(operations, resolver, num_controls)` | Operation リストを走査しゲート数を算出。ループ/分岐/呼び出し/CompositeGate を再帰処理 |
| `_handle_for(op, resolver, num_controls)` | ForOperation のゲートカウント（ループ変数依存→Sum / 非依存→iterations倍） |
| `_handle_call(op, resolver, num_controls)` | CallBlockOperation のゲートカウント（呼び出し先を再帰カウント） |
| `_handle_composite(op, resolver, num_controls)` | CompositeGateOperation のゲートカウント（metadata > impl > formula > error） |
| `_apply_sum_to_count(count, loop_var, start, stop, step)` | GateCount の全フィールドに SymPy `Sum` を適用（逆ループ対応） |

## resource_estimator.py

| 関数/クラス | 概要 |
|---|---|
| `estimate_resources(block, *, bindings)` | エントリポイント。2推定器を統合し ResourceEstimate を返す |
| `ResourceEstimate` | 統合結果のデータクラス。`substitute()`, `simplify()`, `to_dict()`, `__str__()` メソッドを持つ |

---

## ファイル構成

```
qamomile/circuit/estimator/
├── __init__.py               # 公開 API
├── DESIGN.md                 # 本ファイル
├── _utils.py                 # 共有ユーティリティ（BINOP_TO_SYMPY, _strip_nonneg_max）
├── _resolver.py              # 統一値解決（ExprResolver）
├── _catalog.py               # ゲート分類定数・メタデータ抽出・QFT公式
├── _engine.py                # 共有操作処理ヘルパー（CompositeGate/ControlledU/ループスコープ）
├── _loop_executor.py         # ループ実行戦略（具体範囲解決・補間）
├── qubits_counter.py         # qubit 数推定
├── gate_counter.py           # ゲート数推定
├── resource_estimator.py     # 統合インターフェース
└── algorithmic/              # アルゴリズム固有の理論推定
    ├── __init__.py
    ├── qaoa.py               # estimate_qaoa, estimate_qaoa_ising
    ├── qpe.py                # estimate_qpe, estimate_eigenvalue_filtering
    └── hamiltonian_simulation.py  # estimate_trotter, estimate_qsvt, estimate_qdrift
```
