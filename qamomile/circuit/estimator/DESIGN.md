# Resource Estimation 設計・実装仕様書

本モジュールは Qamomile で構築した量子回路の代数的リソース推定を提供する。
全推定値は SymPy のシンボリック式で表され、問題サイズパラメータに依存する式として扱える。

参考文献: arXiv:2310.03011v2 "Quantum algorithms: A survey of applications and end-to-end complexities"

---

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

演算: `+`（加算＝直列合成）、`*`（スカラー倍）、`.max()`（要素ごと最大＝並列合成）、`.simplify()`、`.substitute()`、`.zero()`

### 1.3 ResourceEstimate（resource_estimator.py）

3つのメトリクスを統合する。

```python
@dataclass
class ResourceEstimate:
    qubits: sp.Expr          # 論理量子ビット数
    gates: GateCount          # ゲート数内訳
    depth: CircuitDepth       # 回路深さ内訳
    parameters: dict[str, sp.Symbol]  # 式中のパラメータシンボル
```

メソッド:
- `substitute(**values)`: パラメータを具体値に代入
- `simplify()`: 全式を簡約化
- `to_dict()`: JSON/YAML 用辞書に変換
- `__str__()`: 可読なフォーマット出力

---

## 2. ゲート分類定数

以下の定数で各ゲートを分類する。gate_counter.py と depth_estimator.py の両方で共通の定義を使用。

| 定数名 | ゲート |
|---|---|
| `CLIFFORD_GATES` | h, x, y, z, s, sdg, cx, cy, cz, swap |
| `T_GATES` | t, tdg |
| `SINGLE_QUBIT_GATES` | h, x, y, z, s, sdg, t, tdg, rx, ry, rz, p, u, u1, u2, u3 |
| `TWO_QUBIT_GATES` | cx, cy, cz, swap, cp, crx, cry, crz, rzz |
| `ROTATION_GATES` | rx, ry, rz, p, cp, crx, cry, crz, rzz |
| `MULTI_QUBIT_GATES` | toffoli |
| `_GATE_BASE_QUBITS` | toffoli → 3 |
| `_CONTROLLED_CLIFFORD_GATES` | x, y, z（1制御の CX/CY/CZ のみ Clifford） |

注意:
- CP, CRX, CRY, CRZ, RZZ は `TWO_QUBIT_GATES` と `ROTATION_GATES` の両方に属する
- Toffoli は `MULTI_QUBIT_GATES` に属し、`TWO_QUBIT_GATES` には属さない

---

## 3. Qubit カウント（qubits_counter.py）

### 3.1 基本ルール

`qubits_counter(block)` は回路内で確保される量子ビットの総数を返す。

- `QInitOperation`: 量子ビット配列のサイズ（次元の積）を加算。単一qubitなら1。
- 入力 qubit（関数引数として渡される qubit）はトップレベルでのみカウントし、再帰呼び出し時にはカウントしない（二重カウント回避）。

### 3.2 制御構造での処理

| Operation | 処理 |
|---|---|
| `ForOperation` | 内部 qubit 数 × ループ回数 `(stop - start) / step` |
| `WhileOperation` | 内部 qubit 数 × `\|while\|` シンボル |
| `IfOperation` | `max(true側, false側)` |
| `CallBlockOperation` | 呼び出し先ブロックの operations を再帰カウント（input_values はスキップ） |
| `ControlledUOperation` | 制御ブロックの operations を再帰カウント |
| `ForItemsOperation` | 内部 qubit 数 × `\|dict_name\|` シンボル |
| `CompositeGateOperation` | implementation の内部確保 + resource_metadata の ancilla_qubits |

### 3.3 例：Bell 状態

```python
@qm.qkernel
def bell_state() -> qm.Vector[qm.Qubit]:
    q = qm.qubit_array(2)   # QInitOperation: 2 qubits
    q[0] = qm.h(q[0])
    q[0], q[1] = qm.cx(q[0], q[1])
    return q
```

`qubits_counter(bell_state.block)` → `2`

### 3.4 例：GHZ 状態（パラメトリック）

```python
@qm.qkernel
def ghz_state(n: qm.UInt) -> qm.Vector[qm.Qubit]:
    q = qm.qubit_array(n)   # QInitOperation: n qubits
    q[0] = qm.h(q[0])
    for i in qm.range(n - 1):
        q[i], q[i+1] = qm.cx(q[i], q[i+1])
    return q
```

`qubits_counter(ghz_state.block)` → `n`（SymPy シンボル）

---

## 4. ゲートカウント（gate_counter.py）

### 4.1 単一ゲートの分類ルール

`_count_gate_operation(op, num_controls=0)` で1つのゲートを分類する。

#### 制御なし（`num_controls == 0`）

`total` は常に 1。他フィールドは所属する集合に応じて 0 or 1。

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

制御ゲートの分類ルール:

1. **t_gates**: 常に **0**。制御付き T/Tdg は T ゲートとして扱わない。
2. **total_qubits** = `num_controls` + ベースゲートの量子ビット数で分類:
   - `SINGLE_QUBIT_GATES` → +1
   - `TWO_QUBIT_GATES` → +2
   - `MULTI_QUBIT_GATES` → `_GATE_BASE_QUBITS` から取得（toffoli → +3）
3. `total_qubits == 2` → `two_qubit=1, multi_qubit=0`
4. `total_qubits > 2` → `two_qubit=0, multi_qubit=1`
5. **clifford**: `_CONTROLLED_CLIFFORD_GATES`（x, y, z）に属し `num_controls == 1` の場合のみ 1
   - CX, CY, CZ は Clifford。CCX, CCZ は Clifford ではない。
6. **rotation**: ベースが `ROTATION_GATES` に属する場合 1
7. **symbolic num_controls**: `sp.Piecewise` で条件分岐式を生成し、`subs()` で正しく評価可能にする

**例：制御付き T（C-T、1制御）**

```
gate_name = "t", num_controls = 1
total_qubits = 1 + 1 = 2
→ total=1, single=0, two=1, multi=0, t_gates=0, clifford=0, rotation=0
```

**例：Toffoli（CCX、2制御の X）**

```
gate_name = "x", num_controls = 2
total_qubits = 2 + 1 = 3
→ total=1, single=0, two=0, multi=1, t_gates=0, clifford=0, rotation=0
```

### 4.2 制御構造でのカウントルール

| Operation | 処理 |
|---|---|
| `GateOperation` | `_count_gate_operation(op, num_controls)` で1ゲート分加算 |
| `ForOperation` | 内部カウントがループ変数に依存 → `Sum` 式、非依存 → 回数倍 |
| `WhileOperation` | 内部カウント × `\|while\|` シンボル |
| `IfOperation` | `true_count.max(false_count)` |
| `CallBlockOperation` | 仮引数→実引数マッピング後、呼び出し先を再帰カウント |
| `ControlledUOperation` | 不透明ゲートとして1ゲート分加算（§4.7 参照） |
| `ForItemsOperation` | 内部カウント × `\|dict_name\|` シンボル |
| `CompositeGateOperation` | 下記の優先順位で処理 |

### 4.3 CompositeGateOperation の処理優先順位

1. **ResourceMetadata**: `_extract_gate_count_from_metadata` で型付きフィールド（`meta.single_qubit_gates`, `meta.two_qubit_gates`, `meta.multi_qubit_gates`, `meta.t_gates`, `meta.clifford_gates`, `meta.rotation_gates`, `meta.total_gates`）から直接 GateCount を取得。`total_gates` が未設定の場合は `single_qubit + two_qubit + multi_qubit` で算出。implementation がないスタブは `oracle_calls` に記録
2. **Implementation**: 分解回路を再帰的にカウント
3. **既知型（QFT/IQFT）**: 以下の公式を使用（シンボリック QPE のフォールバック用として維持）
4. **エラー**: いずれもない場合は ValueError

注意: depth_estimator.py では異なる方式で ResourceMetadata を読み取る（§5.6 参照）。

### 4.4 QFT/IQFT のゲートカウント公式

#### n の決定方法（ゲートカウント）

`_count_composite_gate` は以下の優先順位で n を決定する:

1. **target_qubits が非空の場合**: 最初の qubit の `parent_array` を辿り、配列のシンボリックな次元を取得。これにより、具体的な qubit 数ではなくパラメトリックな式（例: `m`）が保持される。`parent_array` が辿れない場合は `len(target_qubits)` を使用
2. **target_qubits が空の場合**: ブロック内の `QInitOperation` から `name == "counting"` の配列（QPE 用）を探索
3. **フォールバック**: `op.num_target_qubits` フィールド → シンボリック `n`

注意: depth_estimator.py では `parent_array` による逆引きを行わず、target_qubits が非空なら常に `len(target_qubits)` を使用する（§5.7 参照）。

n 量子ビットの QFT は以下のゲートで構成される:

| ゲート種別 | 個数 | 分類 |
|---|---|---|
| H | n | single_qubit, clifford |
| CP (controlled-phase) | n(n-1)/2 | two_qubit, rotation |
| SWAP | n//2 | two_qubit, clifford |

```
GateCount:
  total        = n + n(n-1)/2 + n//2
  single_qubit = n
  two_qubit    = n(n-1)/2 + n//2
  multi_qubit  = 0
  t_gates      = 0
  clifford     = n + n//2      (H + SWAP)
  rotation     = n(n-1)/2      (CP)
```

### 4.5 例：Bell 状態のゲートカウント

```
Bell 状態: H(q0), CX(q0, q1)

H:  total=1, single=1, two=0, multi=0, t=0, clifford=1, rotation=0
CX: total=1, single=0, two=1, multi=0, t=0, clifford=1, rotation=0
合計: total=2, single=1, two=1, multi=0, t=0, clifford=2, rotation=0
```

### 4.6 例：GHZ 状態のゲートカウント（パラメトリック）

```
GHZ(n): H(q0), CX(q0,q1), CX(q1,q2), ..., CX(qn-2,qn-1)

H:  1 gate (single, clifford)
CX: n-1 gates (ForOperation で n-1 回) → two_qubit, clifford

合計: total=n, single=1, two=n-1, multi=0, t=0, clifford=n, rotation=0
```

### 4.7 ControlledUOperation の扱い（不透明ゲート）

`qmc.controlled(qkernel)` で生成される `ControlledUOperation` は、内部ブロックを展開せず、
**1つの不透明（opaque）ゲート**としてカウントする。

#### 設計原則

ControlledUOperation はユーザーが定義した量子カーネルを制御付きで呼び出すものであり、
内部実装の詳細に依存しないリソース推定を提供する。
これにより、オラクル呼び出しや複合サブルーチンを含む QPE 等のアルゴリズムで、
「何回オラクルを呼んだか」を正確に追跡できる。

#### プリミティブゲートとの区別

| 種別 | IR Operation | 例 | rotation/clifford 分類 |
|------|-------------|----|-----------------------|
| プリミティブゲート | `GateOperation` | `qmc.cp()`, `qmc.cx()`, `qmc.h()` | セクション4.1に従い分類する |
| 制御付きカーネル | `ControlledUOperation` | `qmc.controlled(my_kernel)` | **分類しない**（rotation=0, clifford=0） |

#### カウントルール

```
GateCount:
  total        = 1
  single_qubit = 0
  two_qubit    = 1 if (num_controls + num_targets == 2) else 0
  multi_qubit  = 1 if (num_controls + num_targets > 2) else 0
  t_gates      = 0
  clifford     = 0   ← 内部ゲートに依存しない
  rotation     = 0   ← 内部ゲートに依存しない
```

- `num_controls` が symbolic の場合は `sp.Piecewise` で two/multi を条件分岐
- `num_targets` は内部ブロックの量子ビット入力数から決定

#### 例：Controlled-P（1制御、1ターゲット）

```python
_phase = qmc.qkernel(lambda q, theta: qmc.p(q, theta))
controlled_u = qmc.controlled(_phase)
```

```
num_controls=1, num_targets=1 → total_qubits=2
→ total=1, two_qubit=1, multi_qubit=0, rotation=0, clifford=0
```

プリミティブの `qmc.cp()` とは異なることに注意:
- `qmc.cp()` → GateOperation → rotation=1
- `qmc.controlled(_phase)` → ControlledUOperation → rotation=0

#### 不透明ゲート設計の一般原則

ControlledUOperation の不透明ゲート扱いは、以下の一般原則に基づく:

**「ユーザーが意味的に1つの操作として定義したものは、リソース推定でも1つのゲートとしてカウントする」**

この原則は ControlledUOperation に限らず、`power(qkernel, k)` のような冪乗演算子にも適用される。
`power(my_kernel, k)` は `my_kernel` を k 回適用する操作だが、リソース推定ではこれを k 個の
独立したゲートとしてカウントするのではなく、1つの不透明ゲートとして扱う。
内部実装の詳細なリソースが必要な場合は、`CompositeGateOperation` + `ResourceMetadata` で
明示的にアノテーションする。

この設計により:
- アルゴリズムの論理的構造（オラクル呼び出し回数等）を正確に追跡できる
- 内部実装に依存しない抽象的なリソース推定が可能になる
- ユーザーは必要に応じて `ResourceMetadata` で詳細なコスト情報を提供できる

---

## 4.8 CompositeGateOperation と ResourceMetadata

CompositeGateOperation は複合ゲート（QFT, IQFT, QPE, カスタムゲート）を IR 上の単一操作として表現する。
リソース推定においては、3つの推定器がそれぞれ異なるフィールドを読み取る。

### 4.8.1 ResourceMetadata 全フィールド

`ResourceMetadata`（`composite_gate.py`）は CompositeGateOperation に付与するメタデータである。

```python
@dataclass
class ResourceMetadata:
    # --- Gate count fields ---
    query_complexity: int | None = None
    t_gates: int | None = None
    ancilla_qubits: int = 0
    total_gates: int | None = None
    single_qubit_gates: int | None = None
    two_qubit_gates: int | None = None
    multi_qubit_gates: int | None = None
    clifford_gates: int | None = None
    rotation_gates: int | None = None
    # --- Depth fields ---
    total_depth: int | None = None
    t_depth: int | None = None
    two_qubit_depth: int | None = None
    multi_qubit_depth: int | None = None
    rotation_depth: int | None = None
    # --- Free-form metadata ---
    custom_metadata: dict[str, Any] = field(default_factory=dict)
```

| フィールド | 型 | デフォルト | 読み取る推定器 | 説明 |
|---|---|---|---|---|
| `query_complexity` | `int \| None` | `None` | なし（情報提供のみ） | オラクル/ユニタリクエリの回数。推定器は直接使用しないが、アルゴリズム解析で参照可能 |
| `t_gates` | `int \| None` | `None` | gate_counter | T/Tdg ゲート数 |
| `ancilla_qubits` | `int` | `0` | qubits_counter | 内部で確保する補助量子ビット数。implementation 内の QInitOperation とは別にカウントされる |
| `total_gates` | `int \| None` | `None` | gate_counter | 全ゲート数。未設定（`None`）の場合は `single_qubit_gates + two_qubit_gates + multi_qubit_gates` で自動算出 |
| `single_qubit_gates` | `int \| None` | `None` | gate_counter | 1量子ビットゲート数 |
| `two_qubit_gates` | `int \| None` | `None` | gate_counter | 2量子ビットゲート数 |
| `multi_qubit_gates` | `int \| None` | `None` | gate_counter | 3量子ビット以上のゲート数 |
| `clifford_gates` | `int \| None` | `None` | gate_counter | Clifford ゲート数 |
| `rotation_gates` | `int \| None` | `None` | gate_counter | 回転ゲート数 |
| `total_depth` | `int \| None` | `None` | depth_estimator | 全回路深さ |
| `t_depth` | `int \| None` | `None` | depth_estimator | T ゲート深さ |
| `two_qubit_depth` | `int \| None` | `None` | depth_estimator | 2量子ビットゲート深さ |
| `multi_qubit_depth` | `int \| None` | `None` | depth_estimator | 3量子ビット以上ゲート深さ |
| `rotation_depth` | `int \| None` | `None` | depth_estimator | 回転ゲート深さ |
| `custom_metadata` | `dict[str, Any]` | `{}` | なし（情報提供のみ） | 戦略固有の補助情報（`precision`, `strategy`, `num_h_gates` 等） |

**`None` の扱い**: 型付きフィールド（`total_gates`, `single_qubit_gates`, `total_depth` 等）が `None` の場合、各推定器は **0 として扱う**。これは「不明」を意味し、真の値がゼロであることを保証しない。カウントしたい場合は明示的に値を設定する必要がある。

**整合性チェック（ゲートカウントのみ）**: `_extract_gate_count_from_metadata` は以下の条件がすべて満たされる場合に `UserWarning` を発行する:

1. `total_gates` が設定されている（`None` でも `0` でもない）
2. `single_qubit_gates`, `two_qubit_gates`, `multi_qubit_gates` のいずれかが `None`
3. `None` でないサブカテゴリの合計が `total_gates` 未満

この警告は情報提供のみであり、動作は変更しない（`None` → `0` のフォールバックは維持）。

**推奨プラクティス**:
- `total_gates` を設定する場合は、`single_qubit_gates`, `two_qubit_gates`, `multi_qubit_gates` も設定する
- スタブゲートで内部ゲート構成が不明な場合は、`total_gates` のみ設定し、サブカテゴリは `None` のままにしてよい（警告は情報提供として受容する）
- 深さフィールドのサブカテゴリ（`t_depth`, `two_qubit_depth` 等）は並列性仮定が異なるため `total_depth` と合計が一致する必要はない。深さに対する整合性チェックは行わない

### 4.8.2 型付きフィールドと custom_metadata の役割分担

ResourceMetadata には2種類のデータ格納方式がある:

1. **型付きフィールド** — ゲートカウント情報と深さ情報（推定器が消費するコアメトリクス）
2. **`custom_metadata` 辞書** — 戦略固有の補助情報（`precision`, `strategy`, `num_h_gates` 等）

各推定器が読み取るフィールドは以下の通り:

```
ResourceMetadata
│
├── ゲートカウント型付きフィールド ──→ gate_counter.py (_extract_gate_count_from_metadata)
│   total_gates, single_qubit_gates, two_qubit_gates,
│   multi_qubit_gates, t_gates, clifford_gates, rotation_gates
│
├── 深さ型付きフィールド ──→ depth_estimator.py (_extract_depth_from_metadata)
│   total_depth, t_depth, two_qubit_depth,
│   multi_qubit_depth, rotation_depth
│
├── ancilla_qubits ──→ qubits_counter.py
│
├── custom_metadata ──→ 推定器は直接使用しない（補助情報用）
│
└── query_complexity ──→ 推定器は直接使用しない（情報提供用）
```

ゲートカウントと深さの両方が統一された型付きフィールドで指定される。`custom_metadata` はコアメトリクスではない補助情報（`precision`, `strategy`, `num_h_gates`, `truncation_depth` 等）のために残されている。

### 4.8.3 スタブゲートの設計原則

`stub=True` で定義された CompositeGate（implementation なし）は、ブラックボックスとして扱われる。

#### ゲートカウント

- 型付きフィールドが `None` → **0 として扱う**。ゲートとしてカウントしたい場合は `total_gates`, `two_qubit_gates` 等を明示的に設定する
- `oracle_calls` にゲート名と回数（= 1）を自動記録する
- 例: `ResourceMetadata(query_complexity=1)` のみ → `GateCount(total=0, oracle_calls={"name": 1})`
- 例: `ResourceMetadata(total_gates=1, two_qubit_gates=1)` → `GateCount(total=1, two_qubit=1, oracle_calls={"name": 1})`

#### 深さ

- `custom_metadata` から深さを取得（未設定なら 0）
- **制御付きスタブの最小深さルール**: `num_control_qubits > 0` かつ `total_depth == 0` かつ `two_qubit_depth == 0` の場合、`total_depth=1, two_qubit_depth=1` に自動補正する。制御ゲートは最低1層の2量子ビットゲートが必要なため

#### qubit 数

- implementation がないため内部 qubit はカウントしない
- `ancilla_qubits` のみ加算（デフォルト 0）

#### 例：Hadamard テスト用 controlled_oracle

```python
@qmc.composite_gate(
    stub=True,
    name="controlled_oracle",
    num_qubits=1,
    num_controls=1,
    resource_metadata=ResourceMetadata(
        query_complexity=1,
        total_gates=1,        # ← 1ゲートとしてカウント
        two_qubit_gates=1,    # ← 2量子ビットゲートとして分類
        custom_metadata={"depth": 1},
    ),
)
def _controlled_oracle():
    pass
```

推定結果:
- ゲート: `GateCount(total=1, two_qubit=1, oracle_calls={"controlled_oracle": 1})`
- 深さ: `CircuitDepth(total_depth=1, two_qubit_depth=1)` （`custom_metadata` の `"depth": 1` + 制御付き補正で `two_qubit_depth=1`）
- qubit: 0（スタブ自身は qubit を確保しない。入力 qubit は呼び出し元でカウント）

### 4.8.4 3推定器の CompositeGateOperation 処理優先順位

| 優先順位 | gate_counter | depth_estimator | qubits_counter |
|---|---|---|---|
| **1. ResourceMetadata** | 型付きフィールドから GateCount 抽出。スタブは `oracle_calls` 記録 | `custom_metadata` から CircuitDepth 抽出。制御付きスタブは最低深さ補正 | `ancilla_qubits` を加算 |
| **2. Implementation** | BlockValue を再帰カウント | `_compute_sequential_depth` で再帰推定 | implementation 内の operations を再帰カウント |
| **3. 既知型** | QFT/IQFT 公式（§4.4） | QFT/IQFT 深さ公式（§5.7） | — |
| **4. エラー** | ValueError | ValueError | — |

注意:
- qubits_counter は優先順位1と2を両方適用する（ancilla_qubits + implementation 内部 qubit）
- gate_counter と depth_estimator は優先順位1が存在すれば2以降はスキップ
- ResourceMetadata が設定されている場合、implementation があっても展開されない（metadata が優先）

---

## 5. 回路深さ推定（depth_estimator.py）

### 5.1 基本原理：DAG クリティカルパス方式

回路深さは各量子ビットごとに独立に追跡する。

```
QubitDepthMap = dict[str, CircuitDepth]
```

**各ゲートの深さ決定**:

```
gate_depth = _estimate_gate_depth(op, num_controls)
involved_qubits = [operand.name for operand in op.operands]
max_current = max(qubit_depths[q] for q in involved_qubits)
new_depth = max_current + gate_depth
for q in involved_qubits:
    qubit_depths[q] = new_depth
```

**最終結果**: 全量子ビットの深さの `max` → `.simplify()`

### 5.2 単一ゲートの深さ分類

`_estimate_gate_depth(op, num_controls)` で1ゲートの深さ寄与を決定する。

#### 制御なし（`num_controls == 0`）

全ゲートで `total_depth = 1`。他フィールドは集合に応じて 0 or 1。

| ゲート | total | t_depth | two_qubit | multi_qubit | rotation |
|---|---|---|---|---|---|
| T, Tdg | 1 | 1 | 0 | 0 | 0 |
| CX, CY, CZ, SWAP | 1 | 0 | 1 | 0 | 0 |
| CP, CRX, CRY, CRZ, RZZ | 1 | 0 | 1 | 0 | 1 |
| Toffoli | 1 | 0 | 0 | 1 | 0 |
| RX, RY, RZ, P | 1 | 0 | 0 | 0 | 1 |
| H, X, Y, Z, S, Sdg, U 等 | 1 | 0 | 0 | 0 | 0 |

#### 制御あり（`num_controls > 0`）

ゲートカウントと同じ `total_qubits` 分類を使用:

- `t_depth`: 常に **0**（制御付き T/Tdg は T ゲートとして扱わない）
- `total_depth`: 常に **1**
- `total_qubits == 2` → `two_qubit_depth=1, multi_qubit_depth=0`
- `total_qubits > 2` → `two_qubit_depth=0, multi_qubit_depth=1`
- `rotation_depth`: ベースが `ROTATION_GATES` に属する場合 1
- symbolic `num_controls`: `sp.Piecewise` で条件分岐

### 5.3 各 Operation 種別の処理

エントリポイント `estimate_depth(block)` は `_estimate_parallel_depth()` を呼び出し、per-qubit depth tracking を行う。

| Operation | 処理 |
|---|---|
| **GateOperation** | `max(関与qubitの深さ) + gate_depth` を全関与qubitに設定 |
| **MeasureOperation** | `total_depth += 1`（他フィールドは 0）を対象 qubit に加算。結果を `value_depths` に記録 |
| **MeasureVectorOperation** | 配列の全要素の深さの `max + 1` を全要素に設定 |
| **MeasureQFixedOperation** | MeasureVectorOperation と同じセマンティクス。QFixed 型の全要素の深さの `max + 1` を全要素に設定 |
| **NotOp** | `value_depths` に入力の深さをそのまま伝播（`value_depths[output] = value_depths[input]`） |
| **CondOp** | `value_depths` に両オペランドの深さの max を記録（`max(lhs_depth, rhs_depth)`） |
| **CompOp** | CondOp と同様、`value_depths` に両オペランドの深さの max を記録 |
| **ForOperation** | §5.4 参照 |
| **IfOperation** | condition の深さまで各ブランチの qubit 深さを bump → 両ブランチ独立推定 → per-qubit `max(true, false)` → `phi_ops` の深さ伝播（true_value と false_value の深さの max を output に記録） |
| **WhileOperation** | `_compute_sequential_depth` による内部深さ × `\|while\|` シンボル。body 内全 qubit を同一深さに更新 |
| **CallBlockOperation** | 仮引数→実引数の qubit 名マッピング構築 → ローカル深さマップで再帰推定 → 結果を呼び出し元に書き戻し |
| **ControlledUOperation** | 不透明ゲートとして `total_depth=1` を加算（§5.10 参照）。全関与 qubit（制御+ターゲット）を `max(current) + 1` に更新 |
| **CompositeGateOperation** | `_estimate_composite_gate_depth()` で深さ取得 → `max(関与qubit) + composite_depth` を全関与 qubit に設定 |
| **ForItemsOperation** | パラレル推定では sequential_depth × `\|dict_name\|` シンボル。具体シミュレーションでも sequential_depth × `\|dict_name\|` シンボルによる上界を使用（warning付き） |

注意: NotOp, CondOp, CompOp の `value_depths` 伝播は IfOperation の condition 深さ追跡に不可欠。condition が measurement → CompOp → CondOp と構成される場合、これらの深さが正しく伝播されないと IfOperation のブランチ開始深さが不正確になる。

### 5.4 ForOperation の深さ推定

ForOperation は具体ループとパラメトリックループで異なる方式を使う。

#### 具体ループ（境界が全て定数）

全イテレーションを `_simulate_parallel_depth_concrete` で直接シミュレーションする。
各イテレーションで qubit 名を具体値に解決し、アクセスパターンを正確に追跡する。

#### パラメトリックループ（境界にシンボルを含む）

サンプリング＋補間方式:

1. **サンプル点の選択**: `_compute_valid_sample_points_for_block` で n=2〜29 の範囲からループ反復回数 > 0 となる点のみ選択（最大7点: 6訓練 + 1検証）。`_handle_for_parallel` 内の通常パスでは n=2〜20 を使用
2. **具体シミュレーション**: 各サンプル点でパラメータを代入し、全イテレーションを具体的にシミュレート
3. **補間** (`_interpolate_depth`): 各 CircuitDepth フィールドごとに以下の3段階で補間:
   - Step 1: Leave-one-out 多項式補間（最後の点で検証）
   - Step 2: 指数+線形形式 `a·2^n + b·n + c`（3点で連立方程式を解き、全点で検証）
   - Step 3: フル多項式補間（フォールバック）
4. **検証**: 保留した検証点で補間結果を確認。不一致なら検証点を含めて再補間
5. **適用**: `new_depth = entry_depth + interpolated` を全 qubit に設定

最小3サンプル点が必要。不足する場合は `_compute_sequential_depth` にフォールバック。

#### パラメトリックループの内部パイプライン

1. **`_scan_parametric_for_loops`**: トップレベルの ForOperation をスキャンし、パラメトリックなシンボルとループ境界を収集
2. **`_has_binop_name_collisions`**: 同じ結果名を持つ BinOp が複数存在するか検出（例: 同名の `uint_tmp`）
3. 衝突がない場合: 通常のサンプリング+補間パスを使用
4. 衝突がある場合: `_estimate_by_full_block_sampling` でブロック全体をサンプリングベースで推定

#### BinOp 名前衝突時

複数の BinOp が同じ結果名を持つ場合（例: 同名の `uint_tmp`）、qubit 名の解決が曖昧になる。
`_has_binop_name_collisions` がこれを検出し、`_estimate_by_full_block_sampling` にフォールバックする。

`_estimate_by_full_block_sampling` は以下の手順で動作する:

1. `_compute_valid_sample_points_for_block` でサンプル点を選択（n=2〜29）
2. 各サンプル点でブロック全体のパラメータを代入し、全イテレーションを具体的にシミュレート
3. `_interpolate_depth` でシンボリック式に補間

### 5.5 フォールバック：逐次深さ（`_compute_sequential_depth`）

全ゲートの深さを単純加算する（並列性を考慮しない）保守的な推定。

- ForOperation: `Sum` 式またはイテレーション数倍
- IfOperation: `max(true, false)`
- WhileOperation: 処理しない（スキップ）

WhileOperation は `_estimate_parallel_depth` 側でこの関数を呼んで内部深さを取得し、`|while|` シンボル倍する。

### 5.6 CompositeGateOperation の深さ推定

優先順位:

1. **ResourceMetadata** → `_extract_depth_from_metadata` で `meta.custom_metadata` 辞書から取得。キーは `"depth"` or `"total_depth"`, `"t_depth"`, `"two_qubit_depth"`, `"rotation_depth"`。未設定のフィールドは 0。**制御付きスタブの最小深さルール**: `op.num_control_qubits > 0` かつ `total_depth == 0` かつ `two_qubit_depth == 0` の場合、`total_depth=1, two_qubit_depth=1` に強制する（制御ゲートは最低1層の2量子ビットゲートが必要）
2. **Implementation** → `_compute_sequential_depth` で分解回路の深さを再帰推定
3. **既知型（QFT/IQFT のみ）**: 公式を使用（下記）（シンボリック QPE のフォールバック用として維持）
4. **エラー**: いずれもない場合は ValueError

gate_counter.py と depth_estimator.py の両方が ResourceMetadata の型付きフィールドから値を抽出する（§4.8.2 参照）。

### 5.7 QFT/IQFT の深さ公式

#### n の決定方法（深さ推定）

depth_estimator.py では gate_counter.py とは異なり、`parent_array` による逆引きを行わない:

1. **target_qubits が非空の場合**: `len(target_qubits)` を使用（具体値）
2. **target_qubits が空の場合**: gate_counter.py と同じフォールバック（`QInitOperation` の `"counting"` 配列 → `num_target_qubits` → シンボリック `n`）

このため、パラメトリックな配列サイズを持つ QFT/IQFT では、gate_counter がシンボリック式を返す一方で depth_estimator は具体値を返す可能性がある。

```
CircuitDepth:
  total_depth       = 2*n        （パラレルスケジューリング前提、IQFT._resources() と一致）
  t_depth           = 0
  two_qubit_depth   = n(n-1)/2   （2量子ビットゲートのみの逐次推定）
  multi_qubit_depth = 0
  rotation_depth    = n(n-1)/2   （CP ゲートのみの逐次推定）
```

注意: `total_depth` はゲート並列配置を考慮した深さ、`two_qubit_depth` と `rotation_depth` は対象ゲートのみの逐次推定であるため、n ≥ 3 のとき `two_qubit_depth > total_depth` となる。これらは異なる並列性仮定に基づく独立した指標である。

### 5.8 例：Bell 状態の深さ

```
回路: H(q0), CX(q0, q1)

Step 1: H(q0)
  qubit_depths = {q0: CircuitDepth(1,0,0,0,0)}

Step 2: CX(q0, q1)
  involved = [q0, q1]
  max_current = max(q0の深さ, q1の深さ) = CircuitDepth(1,0,0,0,0)
  gate_depth = CircuitDepth(1,0,1,0,0)
  new_depth = CircuitDepth(2,0,1,0,0)
  qubit_depths = {q0: (2,0,1,0,0), q1: (2,0,1,0,0)}

最終結果: max(全qubit) = CircuitDepth(total=2, t=0, two_qubit=1, multi=0, rotation=0)
```

### 5.9 例：3量子ビット GHZ 状態の深さ

```
回路: H(q0), CX(q0,q1), CX(q1,q2)

Step 1: H(q0)
  q0=(1,0,0,0,0)

Step 2: CX(q0, q1)
  max(q0, q1) = (1,0,0,0,0)  // q0=1, q1=0 → max=1
  + gate(1,0,1,0,0)
  → (2,0,1,0,0)
  q0=(2,0,1,0,0), q1=(2,0,1,0,0)

Step 3: CX(q1, q2)
  max(q1, q2) = (2,0,1,0,0)  // q1=2, q2=0 → max=2
  + gate(1,0,1,0,0)
  → (3,0,2,0,0)
  q1=(3,0,2,0,0), q2=(3,0,2,0,0)

最終結果: max(q0, q1, q2) = CircuitDepth(total=3, t=0, two_qubit=2, multi=0, rotation=0)

DAG 図:
  q0: ─ H ─ CX ─────
  q1: ───── CX ─ CX ─
  q2: ─────────── CX ─
  深さ: 1    2    3
```

### 5.10 ControlledUOperation の深さ

セクション4.7 と同様、`ControlledUOperation` は不透明ゲートとして `total_depth=1` を加算する。

```
CircuitDepth:
  total_depth       = 1
  t_depth           = 0
  two_qubit_depth   = 1 if (num_controls + num_targets == 2) else 0
  multi_qubit_depth = 1 if (num_controls + num_targets > 2) else 0
  rotation_depth    = 0   ← 内部ゲートに依存しない
```

全関与量子ビット（制御 + ターゲット）の現在の深さの `max` に上記を加算し、
全関与量子ビットの深さを更新する。

逐次深さ推定（`_compute_sequential_depth`）でも同じルールを適用する。

---

## 6. 値解決（Value Resolution）

gate_counter.py と depth_estimator.py はそれぞれ独立した `value_to_expr` / `_trace_value_operation` 関数を持つ。
基本的なロジックは共通だが、以下の差異がある。

### 6.1 共通ロジック

`value_to_expr(v, block, call_context, loop_var_symbols)` は Value を SymPy 式に変換する:

1. **定数** → `sp.Integer()` / `sp.Float()`
2. **パラメータ** → `sp.Symbol(name, integer=True, positive=True)`
3. **ループ変数** → `loop_var_symbols` から解決
4. **計算値** → `_trace_value_operation()` で BinOp / CompOp を再帰的に辿る
5. **フォールバック** → `sp.Symbol(v.name, integer=True, positive=True)`

### 6.2 gate_counter.py 固有: `parent_blocks` スコープチェーン

gate_counter.py の `value_to_expr` と `_count_from_operations` は `parent_blocks` パラメータを持つ。
ネストされたループで値のトレースに失敗した場合、祖先ブロックを逆順に辿って値を解決する。

```python
def value_to_expr(v, block, call_context, loop_var_symbols, parent_blocks=None):
    ...
    if parent_blocks:
        for pb in reversed(parent_blocks):
            traced = _trace_value_operation(v, pb, ...)
            if traced is not None:
                return traced
```

depth_estimator.py はこのパラメータを持たず、現在のブロックのみで値を解決する。

### 6.3 FLOORDIV 処理の差異

`_trace_value_operation` で `BinOpKind.FLOORDIV` を処理する際:

- **gate_counter.py**: `sp.simplify(left / right)` で商を簡約化し、結果が `sp.Integer`, `sp.Symbol`, `sp.Pow` のいずれかなら `floor()` でラップしない（例: `2**m / 2**i = 2**(m-i)` のような正確なシンボリック除算を保持）
- **depth_estimator.py**: 常に `sp.floor(left / right)` を使用

この差異により、同じ回路に対して gate_counter と depth_estimator が異なるシンボリック式を生成する場合がある。

### 6.4 補助関数

- **`_find_loop_variable_values(operations, loop_var_name)`**: ループ変数を直接表す Value のみを検出（`2**i` のような派生式は含まない）
- **`_apply_sum_to_count(count, loop_var, start, stop, step)`** / **`_apply_sum_to_depth(...)`**: SymPy `Sum` を GateCount/CircuitDepth の全フィールドに適用。逆ループ（step < 0）の場合は境界を反転

---

## 7. 統合インターフェース（resource_estimator.py）

### 7.1 estimate_resources

```python
def estimate_resources(block: BlockValue | Block | list[Operation]) -> ResourceEstimate:
```

3つの推定器を統合する:

1. `qubits_counter(block)` → qubit 数
2. `count_gates(block)` → ゲートカウント
3. `estimate_depth(block)` → 回路深さ

全式中のシンボルを収集し `parameters` 辞書に格納する。

### 7.2 使用例

```python
import qamomile.circuit as qm
from qamomile.circuit.estimator import estimate_resources

@qm.qkernel
def ghz_state(n: qm.UInt) -> qm.Vector[qm.Qubit]:
    q = qm.qubit_array(n)
    q[0] = qm.h(q[0])
    for i in qm.range(n - 1):
        q[i], q[i+1] = qm.cx(q[i], q[i+1])
    return q

est = estimate_resources(ghz_state.block)
print(est.qubits)             # n
print(est.gates.total)         # n
print(est.gates.two_qubit)     # n - 1

concrete = est.substitute(n=100)
print(concrete.qubits)         # 100
print(concrete.gates.total)    # 100
```

---

## 8. アルゴリズム固有 Estimator（algorithmic/）

回路の IR を解析するのではなく、量子アルゴリズムのパラメータから理論的な漸近計算量に基づいてリソースを推定する。

### 8.1 QAOA（qaoa.py）

#### estimate_qaoa

```python
def estimate_qaoa(
    n: sp.Expr | int,           # 量子ビット数
    p: sp.Expr | int,           # QAOA レイヤー数
    num_edges: sp.Expr | int,   # 問題グラフの辺数
    mixer_type: str = "x",      # ミキサー種別（現在 "x" のみ）
) -> ResourceEstimate:
```

p 層の標準 QAOA（X ミキサー）の公式:

| メトリクス | 式 |
|---|---|
| qubits | n |
| gates.total | n + p·(edges + n) |
| gates.single_qubit | n + p·n（H 初期化 + 各層 RX ミキサー） |
| gates.two_qubit | p·edges（各層 RZZ コスト） |
| gates.t_gates | 0 |
| gates.clifford | n（H ゲートのみ） |
| gates.rotation | p·(edges + n)（RZZ + RX） |
| depth.total_depth | n + p·(edges + n)（逐次推定） |
| depth.two_qubit_depth | p·edges |
| depth.rotation_depth | p·(edges + n) |

#### estimate_qaoa_ising

```python
def estimate_qaoa_ising(
    n: sp.Expr | int,
    p: sp.Expr | int,
    quadratic_terms: sp.Expr | int,    # J_ij 項数
    linear_terms: sp.Expr | int | None = None,  # h_i 項数（default: n）
) -> ResourceEstimate:
```

Ising ハミルトニアン用の便利ラッパー。`estimate_qaoa` をベースに、線形項の RZ ゲートを追加する。
線形項はミキサーと並列化可能なため、depth は増加しない。

### 8.2 QPE（qpe.py）

#### estimate_qpe

```python
def estimate_qpe(
    n_system: sp.Expr | int,              # システム量子ビット数
    precision: sp.Expr | int,             # 精度ビット数（ε = 2^(-precision)）
    hamiltonian_norm: sp.Expr | float | None = None,  # ||H||（qubitization に必要）
    method: str = "qubitization",         # "qubitization" or "trotter"
) -> ResourceEstimate:
```

**Qubitization 方式**:

| メトリクス | 式 |
|---|---|
| qubits | n + m + ⌈log₂(n)⌉ |
| gates.total | α·2^m·n（α = hamiltonian_norm） |
| depth.total_depth | α·2^m·n |
| single/two 分割 | total / 2（概算） |

**Trotter 方式**:

| メトリクス | 式 |
|---|---|
| qubits | n + m |
| gates.total | 2^m·n |
| depth.total_depth | 2^m·n |

#### estimate_eigenvalue_filtering

```python
def estimate_eigenvalue_filtering(
    n_system: sp.Expr | int,
    target_overlap: sp.Expr | float,  # 目標オーバーラップ γ
    gap: sp.Expr | float | None = None,  # スペクトルギャップ Δ
) -> ResourceEstimate:
```

- ギャップなし: O(1/γ) 回のブロックエンコーディング呼び出し
- ギャップあり: O(1/√(γΔ)) 回

### 8.3 Hamiltonian Simulation（hamiltonian_simulation.py）

#### estimate_trotter

```python
def estimate_trotter(
    n: sp.Expr | int,           # 量子ビット数
    L: sp.Expr | int,           # ハミルトニアン項数
    time: sp.Expr | float,      # 時間発展 t
    error: sp.Expr | float,     # 目標誤差 ε
    order: int = 2,             # Trotter 次数（2, 4, 6, ...）
    hamiltonian_1norm: sp.Expr | float | None = None,  # ||H||₁（default: L）
) -> ResourceEstimate:
```

p 次の Trotter-Suzuki 公式:

```
ステップ数 r = (||H||₁·t)^(1+1/p) / ε^(1/p)
gates.total = r · L · n
depth.total_depth = r · L
```

#### estimate_qsvt

```python
def estimate_qsvt(
    n: sp.Expr | int,
    hamiltonian_norm: sp.Expr | float,  # α（ブロックエンコーディング正規化）
    time: sp.Expr | float,
    error: sp.Expr | float,
) -> ResourceEstimate:
```

QSVT ベースの近似最適シミュレーション:

```
ブロックエンコーディング呼び出し数 = α·t + log(1/ε) / log(log(1/ε))
gates.total = 呼び出し数 × n
```

#### estimate_qdrift

```python
def estimate_qdrift(
    L: sp.Expr | int,
    hamiltonian_1norm: sp.Expr | float,  # ||H||₁
    time: sp.Expr | float,
    error: sp.Expr | float,
) -> ResourceEstimate:
```

ランダムサンプリング方式:

```
サンプル数 = ||H||₁² · t² / ε
gates.total = サンプル数
qubits = n（シンボル、公式からは決定不能）
```

注: single/two-qubit の内訳は不明（項の構造に依存するため全て 0）。

---

## 9. ファイル構成

```
qamomile/circuit/estimator/
├── __init__.py               # 公開 API（ResourceEstimate, GateCount, CircuitDepth, 各推定関数）
├── DESIGN.md                 # 本ファイル
├── qubits_counter.py         # qubit 数推定
├── gate_counter.py           # ゲート数推定
├── depth_estimator.py        # 回路深さ推定
├── resource_estimator.py     # 統合インターフェース
└── algorithmic/              # アルゴリズム固有の理論推定
    ├── __init__.py            # estimate_qaoa, estimate_qpe, estimate_trotter, estimate_qsvt, estimate_qdrift
    ├── qaoa.py
    ├── qpe.py
    └── hamiltonian_simulation.py
```

エクスポートされていないが利用可能な関数:
- `estimate_qaoa_ising`（qaoa.py）
- `estimate_eigenvalue_filtering`（qpe.py）

---

## 10. 参考文献

- arXiv:2310.03011v2 — "Quantum algorithms: A survey of applications and end-to-end complexities"
- arXiv:1411.4028 — Farhi et al. "A Quantum Approximate Optimization Algorithm"
- arXiv:1610.06546 — Low & Chuang, QSVT/Qubitization framework
- arXiv:1806.01838 — Gilyen et al., QSP/QSVT
- arXiv:1912.08854 — Childs et al., improved Trotter bounds
- arXiv:1811.08017 — Campbell, qDRIFT algorithm
- arXiv:1910.14596 — Lin & Tong, eigenstate filtering via QSVT
