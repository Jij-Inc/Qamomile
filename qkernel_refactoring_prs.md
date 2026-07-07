# qkernelリファクタリング PR分割計画(集約版)

- **作成日**: 2026-07-03(同日、初版27 PRから集約改訂)
- **元資料**: [qkernel_audit_report.md](qkernel_audit_report.md)(本編+追補1: 共通化観点+追補2: テスト監査)。本文中のID(P1-x / FE-x / IR-x / TP-x / EM-x / BE-x / 追補1のBE2-x / CE-x / FE2-x / XC-x / IR2-x / 追補2のT-STR-x / T-COV-x / T-QUAL-x)は同レポートの指摘番号を指す。
- **全17 PR** = フェーズ0(バグ修正7本、相互独立)+フェーズ1以降(集約10本)。
- **集約の方針**: 初版でフェーズ1以降に20本あったPRを、(1) 触るファイル群が重なる、(2) レビュー観点が同質(削除のみ/テストのみ/純リファクタ)、(3) 直列依存で分割する意味が薄い、の3基準で10本に統合した。各統合PRの内部は**Stage(コミット単位のマイルストーン)**に分けてあり、スタックコミットでレビューする。レビュー負荷が想定を超えた場合の再分割点もDetailsに明記した。

## 共通ルール(全PRに適用)

- PRタイトル・本文・コミットメッセージは英語(CLAUDE.md)。PR作成前に `/local-review` をクリーンになるまで反復。
- 各PRで `uv run pytest tests/`(必要に応じ `-m ""` でquri_parts/cudaqレーンも)を実行。
- フェーズ0の7本は相互に独立で並行作業可能。フェーズ1以降は下表の依存に従う。

## 優先度ランク(3段階)

判断軸は3つ: **(1) 計算結果の正しさへの直結度**(サイレント誤りは最優先)、**(2) 後続PRのブロッカー性**、**(3) 対象機能の現在の利用度と失敗の顕在性**(静かに誤るものは、大声で落ちるものより上)。

- **高** = ユーザーの計算結果が今日サイレントに(または一般的なワークフローで)壊れる。他の何よりも先に修正。
- **中** = 構造負債の中核。バグの族の再発を止める・後続をブロック解除する・ユーザー体感性能に効く。高の完了後、依存順に着手。
- **低** = 重要だが緊急でない保守性・整合性の改善。中の進行と並行して隙間で消化可能(低≠不要。着手順が後というだけ)。

| 優先度 | PR |
|---|---|
| **高**(6本) | PR-01, PR-02, PR-03, PR-04, PR-06, PR-07 |
| **中**(7本) | PR-05, PR-08, PR-09, PR-10, PR-11, PR-13, PR-15 |
| **低**(4本) | PR-12, PR-14, PR-16, PR-17 |

## 依存関係と優先度の概観

| PR | 内容 | 優先度 | 依存 | 優先度の根拠 |
|---|---|---|---|---|
| PR-01 | Block.call結果Value共有の修正(issue #563) | **高** | なし | サイレント誤コンパイル(測定結果が常に相関)。最悪級 |
| PR-02 | if分岐グローバルshadow修正(issue #564) | **高** | なし | サイレント誤コンパイル。ノートブック利用で高頻度に発火 |
| PR-03 | Handle比較演算子の追加(issue #565) | **高** | なし | 実行時分岐のつもりがコンパイル時定数化(サイレント) |
| PR-04 | 定数畳み込みの出力ガード(issue #566) | **高** | なし | ユーザーが最初に書く形のカーネルで結果がNone |
| PR-05 | シンボリック幅のcanonical/serialize修正 | **中** | なし | P1分類だが、対象機能(シンボリック幅QUInt・content_hashキャッシュ)の現利用は限定的(テストでも生成ゼロ)。serialize失敗はloud。PR-11の等価検証に使うため中の先頭で |
| PR-06 | counts集約・ループ境界セマンティクス(issue #567) | **高** | なし | most_common()が誤答(サイレント)+emit/runtimeの意味論乖離 |
| PR-07 | バックエンド挙動差の修正(issue #568) | **高** | なし | Pauli積由来Hamiltonianを使う一般的な最適化ワークフローがCUDA-Qで壊れる(BE-1)+無音のゲート欠落・誤束縛経路 |
| PR-08 | デッドコード一括削除 | **中** | なし | 挙動不変だが安価・低リスクで、PR-13のブロッカー解除と監査面積の半減に直結 |
| PR-09 | 再計算・オーバーヘッド削減 | **中** | PR-08推奨 | QAOA最適化ループの体感性能に直結(毎iterのtranspile・observable変換・再トレース) |
| PR-10 | テストスイート全面整備 | **中** | なし | PR-13の**必須**前提(退行の無音化解消)+CLAUDE.md必須要件(QAOA非準拠)の是正。以降の全リファクタの安全網 |
| PR-11 | IR走査・Value木プロトコル統一 | **中** | PR-05推奨 | 横断バグパターン①②(最多産の2族)の恒久停止+PR-12の土台 |
| PR-12 | 値解決・反復エンジン統一 | **低** | PR-11 | 影響は主にestimator/visualizationの整合性(実行結果ではなく見積り・描画のエッジケース誤り)。ドリフト再発防止の価値は大きいが緊急でない |
| PR-13 | controlled emission大改修 | **中** | PR-08・PR-10**必須** | 現役最大の重複(QURI約740行の並行walker)+経路間非一貫性バグの温床。危険度が高いため前提2本の完了後すみやかに |
| PR-14 | 知識テーブル導入 | **低** | PR-08推奨 | 純粋な保守性投資(新ゲート/新op追加コストの恒久削減)。現行バグなし |
| PR-15 | バックエンドemit・executor統一 | **中** | PR-07**必須**、PR-14推奨 | Stage Bのサイレント失敗昇格(回転角の無音0.0焼き込み=EM-2、測定欠落のwarning継続=EM-7)は正しさに準じる。BE-1を実際に生んだ三重化構造の解消 |
| PR-16 | フロントエンド高階API・stdlib統一 | **低** | PR-14推奨 | 主に保守性。FE2-3(借用移譲5重)は正しさ直結コードだが現状は整合しており、将来ドリフトの予防 |
| PR-17 | パス契約・例外階層・ドキュメント同期 | **低** | PR-11後推奨 | 開発体験と契約の一貫性。ユーザーの計算結果には影響しない |

## openなissue/PRとの整合(2026-07-03 調査)

open issue 16件・open PR 21件をこの計画と突合した結果(調査時点のスナップショット):

- **PR #546**(measurement-derived classical outputs修正)が **#567の前半(EM-1: convert_counts集約)を実装済み**。マージ後、本計画のPR-06のスコープは **CE-1(for境界乖離)+EM-4/EM-9 に縮小**する。また同PRは `_resolve_outputs` を全面書き換え(定数メタデータのフォールバック追加)しており、#566(PR-04)の症状を緩和する可能性がある — **PR-04は #546マージ後のコードベース上で、constant_fold.py の `block_output_uuids` ガード(根本修正)を主軸に**実装する。
- **PR #551**(IRシリアライゼーション)が **PR-05のうちIR-3(コンテナoperandのdecode失敗)と、PR-09 Stage BのうちIR-12/13(ネストBlock value_table複製)を実装済み**。マージ後、PR-05のスコープは **P1-5/IR-2/IR-4/IR-5(canonical+シンボリック幅)に縮小**、PR-09からIR-12/13を除外する。
- **ファイル競合によるマージ順の注意**: PR #555(controlled_emission.py大改変)→ 計画PR-07/13、PR #561(visit_If改変)→ 計画PR-02/03、PR #562(orchestrator/transpiler)→ 計画PR-04/06、PR #485(estimatorのresource_estimationへの移動)→ 計画PR-12/14。これらのopen PRの帰趨を確認してから該当計画PRに着手する。
- **既存issueとの重複**: #313(SDKテストテンプレート)は **PR-10と同一スコープ**(トラッキングissueとして流用可)。#253/#251は参照ファイルが現行コードに存在せずstale疑い(クローズ候補)。#136(未対応ゲート追加)はPR-14が実装コストを下げる関係。
- #563/#564/#565/#568 の欠陥箇所に触れるopen PRは存在しない(計画PR-01/02/03/07はフルスコープのまま有効)。

---

## PR-01: Give each sub-kernel call site fresh result values

### Background

`Block.call`(ir/block.py:100)が非パススルー出力にcalleeブロックの出力Valueインスタンスをそのまま返すため、同一サブカーネルを2回呼ぶと2つの`CallBlockOperation.results`が同一UUIDを共有し、Bitを返すサブカーネルではclbitが1本に潰れて測定結果が常に相関する(P1-1、再現済みのサイレント誤コンパイル)。テストカバレッジはゼロ(T-COV-1: 既存の`test_same_qkernel_called_twice`はパススルー出力でバグ経路に乗らない)。

### Goal

同一サブカーネルの複数回呼び出しで、各呼び出しサイトが固有の結果Valueを持ち、測定結果が独立になる。

### Details

- 対象: P1-1。修正箇所は `Block.call`(qamomile/circuit/ir/block.py:95-100)の`else`分岐 — `dummy_return`をそのまま返す代わりに `next_version()`(または fresh Value + logical_id継承)を発行する。
- 備考: `_create_traced_block` のコメント(qkernel.py:1742-1752)は特殊化パスでのみReturnOperationで回避済みと述べており、通常パスだけが無防備。SSA不変条件「call結果は常に呼び出しサイト固有」をIR構築時点で確立する`Block.call`修正が本筋(将来の特殊化キャッシュFE-6の前提も単純化される)。
- 影響範囲: inline パスの値マッピングが新しいUUIDを正しく追うことを確認(既存の`ValueSubstitutor(transitive=True)`経路で吸収されるはず)。

### ToDo List

- [ ] `Block.call` の非パススルー出力に fresh Value を発行する
- [ ] `logical_id` 継続性の要否を判断し docstring に契約を明記する
- [ ] 回帰テスト `test_subkernel_double_call_distinct_results` を追加する
- [ ] 既存の `test_same_qkernel_called_twice` に UUID 非共有のアサーションを追記する

### Test Plan

- 新規: Bitを返す`flip(q)`サブカーネルを2回呼び、(1) 2つの`CallBlockOperation.results`のUUIDが異なる、(2) transpile後にclbitが2本、(3) 片方だけ`x`を挿入してsampleし b0≠b1 の分布になる、を検証(3バックエンド)。
- `uv run pytest tests/circuit/ tests/transpiler/` 全グリーン。

---

## PR-02: Thread module-global-shadowed locals through if branches

### Background

AST変換の`VariableCollector`が`func.__globals__`の全キーを変数収集から除外するため(ast_transform.py:1281)、モジュールグローバルと同名のカーネルローカル変数がif分岐のinput/output集合から漏れ、分岐内の代入がサイレントに破棄される(P1-2、再現済み)。ノートブック(過去の全変数がグローバル)で特に踏みやすい。カバレッジゼロ(T-COV-2)。

### Goal

関数スコープで代入される名前は、同名のモジュールグローバルが存在しても正しくif分岐スレッディング(phi生成)の対象になる。

### Details

- 対象: P1-2。`symtable`/`co_varnames`で関数スコープのStore名を先に収集し、`global_names`から差し引く(最小修正)。
- 同時修正: FE-17(`VariableCollector._exclude`の訪問順依存 — 呼び出し名収集を2パス化)。
- 備考: FE-4(`__globals__.copy()`の遅延束縛乖離)は同ファイルの別修正でありリスク分離のため含めない(PR-16で対応)。

### ToDo List

- [ ] 関数ローカル束縛名の事前収集を実装し `global_names` から除外する
- [ ] `VariableCollector` の呼び出し名除外を2パス化する(FE-17)
- [ ] 回帰テスト `test_module_global_shadowing_local_threads_through_if` を追加する
- [ ] 訪問順依存の回帰テストを追加する

### Test Plan

- 新規: モジュールに`count = 123`を置き、カーネル内`count = 0.0; if flag == 1: count = 1.0`を`bindings={"flag": 1}`でtranspile→実行し出力1.0を検証。
- 既存のif分岐スイート(test_control_flow_if.py 71本、test_ast_transform.py 48本)全グリーン。

---

## PR-03: Add missing DSL comparison operators on handles

### Background

`Bit`に`__eq__`/`__ne__`のDSLオーバーロードがなく、`if m0 == m1:`がdataclass等値比較→Python bool定数として畳まれ、実行時分岐のつもりがコンパイル時Falseに固定される(P1-3、再現済み)。`UInt == Float`もNotImplemented→同一性フォールバックで同族のサイレント定数化(FE-5)。dataclass eqの副作用で`Bit`/`Qubit`/`Vector`はunhashable。カバレッジゼロ(T-COV-3: `&` `|` `~`は手厚いのに`==`だけ1件もない)。

### Goal

Handle同士の比較演算子は常にIR演算(CompOp/CondOp)を発行するか明示的にTypeErrorになり、Python bool定数へのサイレント縮退が起きない。

### Details

- 対象: P1-3、FE-5。`Bit.__eq__`/`__ne__`をXNOR/XOR相当のCondOp/CompOp発行として実装し、`__hash__ = object.__hash__`を復元(UInt/Floatと同じパターン)。`UInt`↔`Float`は型昇格してCompOp発行。
- 防御層: `emit_if`(control_flow.py:689-694)で生bool条件を検出し警告(同族バグの将来退行を安く検出)。
- 備考: `Qubit`/`Vector`のhash復元は借用台帳等でのdictキー利用への影響を確認。

### ToDo List

- [ ] `Bit.__eq__`/`__ne__`/`__hash__` を実装する
- [ ] `UInt`↔`Float` 比較の型昇格を実装する
- [ ] `emit_if` に生bool条件の検出を追加する
- [ ] `test_bit_logical_ops.py` に `==`/`!=` セクションを追加する

### Test Plan

- 新規: `test_bit_equality_emits_runtime_comparison` — `IfOperation.operands[0]`がIR Valueであること+実行分布の検証。`UInt == Float`がCompOpを発行するテスト。
- 既存Bit論理演算スイート全グリーン。

---

## PR-04: Preserve block outputs through constant folding

### Background

`ConstantFoldingPass`のBinOp畳み込みに`block_output_uuids`ガードがなく(constant_fold.py:150-156。Store側にはある)、ブロック出力を生成するBinOpがIRから消えて`bindings={"x":3.0}`で`return x*2.0`の結果が実行時にNoneになる(P1-4、エンドツーエンド再現済み)。素通し`return x, bits`も同根。カバレッジゼロ(T-COV-4)。

### Goal

コンパイル時束縛された古典値をブロック出力として返すカーネルが、sample/runで正しい値を返す。

### Details

- 対象: P1-4。(1) BinOp畳み込みにStore側と同じ出力ガードを追加。(2) `ProgramOrchestrator._resolve_outputs`にbindings/定数フォールバック+解決不能時の明示`ExecutionError`を追加(素通しケースの救済+診断改善)。
- 備考: (2)は出力解決経路に触れるため既存sampleテストへの影響を確認しながら段階的に。

### ToDo List

- [ ] BinOp畳み込みに `block_output_uuids` ガードを追加する
- [ ] `_resolve_outputs` にフォールバック+明示エラーを追加する
- [ ] 回帰テスト `test_bound_classical_output_survives_fold`(演算あり/素通しの2形)を追加する

### Test Plan

- 新規: `y = x * 2.0; return y, measure(qs)` を `bindings={"x":3.0}` でtranspile→sampleし第1要素6.0を検証(+`parameters=["x"]`版の不変も並記)。
- constant_fold系既存テスト全グリーン。

---

## PR-05: Fix canonical hashing and serialization of symbolic-width and container values

### Background

シンボリック幅`QUIntType.width`等の型内Valueをcanonicalizeがリマップせず、`label()`がメモリアドレス入りreprを埋め込むため`content_hash`が同一プロセス内でも非決定(P1-5、再現済み)。serialize側も同根で、シンボリック幅Valueのdecode失敗(IR-2)、コンテナoperandのReturnOperationのdecode失敗(IR-3)が再現済み。`loop_var`名のhash混入(IR-5)、`QUIntType`等のunhashable(IR-4)も同領域。カバレッジゼロ(T-COV-5/T-COV-8: `QUInt`はテスト全体で一度も生成されていない)。

### Goal

シンボリック幅型・コンテナ値を含むBlockで、`content_hash`がビルド非依存に決定的であり、JSON/msgpackのround-tripが成功する。

### Details

- 対象: P1-5、IR-2、IR-3、IR-4、IR-5。
  - canonical: `canonical_value`で型内Valueに再帰適用。`label()`はシンボリック幅をUUIDベース表現に。`_OP_FIELD_EXCLUDES`に`loop_var`等を追加(ハッシュ値が変わるためリリースノートに明記)。
  - serialize: `_encode_qreg_width`系に`_EncodeContext`を引き回し幅Valueを登録。`ReturnOperation`のoperandデコードを許容版へ。
  - types: 3型に`__hash__`を最小定義(構造統一はPR-14で)。
- 備考: 恒久対策(Value参照箇所リストの手書き複製解消)はPR-11の`map_value_tree`。本PRは正しさの回復を優先。

### ToDo List

- [ ] `canonical_value` の型内Value再帰リマップを実装する
- [ ] シンボリック幅の `label()` 表現をUUIDベースに変更する
- [ ] `loop_var` 等をcanonical bytesから除外する
- [ ] serializeのシンボリック幅登録とコンテナ許容を実装する
- [ ] 3型の `__hash__` を定義する
- [ ] 回帰テスト4本(hash決定性/round-trip×2/hashability)を追加する

### Test Plan

- 新規: `test_content_hash_symbolic_width_deterministic` / `test_symbolic_qreg_width_round_trip` / `test_container_operand_return_round_trip` / `hash(QUIntType(3))`成立。
- 既存 canonical(39本)/serialize(67本)全グリーン(loop_var除外による期待値更新は明示)。

---

## PR-06: Fix runtime counts aggregation and for-loop bounds semantics

### Background

(1) `convert_counts`が変換後の同一出力値をマージしないため、値が縮退するカーネルで`most_common()`/`probabilities()`が誤った分布を返す(EM-1)。(2) `ForOperation`の境界解決がemit側とruntime側で二重実装され、欠損stopのデフォルトが1と0に**既に乖離**、`step==0`ガードもruntime側のみ(CE-1、検証済み)。カバレッジゼロ(T-COV-6/T-COV-10)。

### Goal

sample結果は出力値ごとに正しく集約され、ForOperationの境界セマンティクスがemit/runtimeで単一の定義を共有する。

### Details

- 対象: EM-1、CE-1。同梱の小修正: EM-4(`run()`のJob型判定)、EM-9(`RunJob.result()`のNoneキャッシュ)、CE-6の一部(orchestrator共通プロローグ抽出 — 本修正で触る箇所のため)。
- 実装: `convert_counts`のCounter集約。`eval_utils`に`resolve_for_bounds(op, resolve)`(step==0検査込み)と`select_phi_input`を新設し両経路をresolver注入で委譲(`fold_classical_op`パターンの水平展開)。
- 備考: デフォルトは「stop欠損=0(空ループ)」に統一が安全側。決定をdocstringに明記。

### ToDo List

- [ ] `convert_counts` のCounter集約を実装する
- [ ] `eval_utils.resolve_for_bounds` / `select_phi_input` を新設し両経路を委譲する
- [ ] EM-4/EM-9 を修正する
- [ ] 回帰テスト(縮退出力の集約/欠損operandの両経路一致/step==0)を追加する

### Test Plan

- 新規: `test_convert_counts_merges_collapsed_outputs`、`test_for_bounds_defaults_agree_between_emit_and_runtime`。
- `tests/transpiler/test_runtime_execution.py` 全グリーン。

---

## PR-07: Fix backend-specific correctness divergences

### Background

3バックエンドの並行実装ドリフトに起因する正確性バグ群: CUDA-Q observableのcomplex係数TypeError(BE-1)、`emit_multi_controlled_p/rx/ry/rz`の`_qref`不使用(BE-4)、QURI inverseプローブのパス側`_parameter_map`未復元(BE-2)、`params or []`のndarray例外(BE-7)、STATIC countsのzfill逆側パディング(BE-8)、Qiskit pauli_evolveのname優先解決(BE-9)、QURI `append_gate`の例外握りつぶし(BE-10)。CUDA-Q用observableテストファイル自体が不在(T-COV-7)。

### Goal

同一のHamiltonian・同一のカーネルに対し、3バックエンドが型エラーや無音のゲート欠落なく一貫した結果を返す。

### Details

- 対象: BE-1、BE-2、BE-4、BE-7、BE-8、BE-9、BE-10。いずれも局所的な1〜10行修正の集合。
- 備考: 変換スケルトン自体の共通化(BE2-1)はPR-15で実施。本PRは挙動差の解消のみ(バグ修正と純リファクタを混ぜない)。

### ToDo List

- [ ] BE-1/BE-4/BE-7/BE-8/BE-9/BE-10 の各修正を適用する
- [ ] BE-2 のプローブ状態復元を実装する
- [ ] CUDA-Q用observableテスト(complex係数含む)を新設し3バックエンド横並びで検証する
- [ ] ネストcontrolled+定数項pauli_evolveの回帰テスト(BE-4)を追加する

### Test Plan

- 新規: complex型係数Hamiltonianで3バックエンドのexpval一致(atol=1e-8)。BE-4再現構成でCUDA-Qコンパイル成功。
- `uv run pytest -m "" tests/transpiler/backends/ tests/quri_parts/` 全グリーン。

---

## PR-08: Remove dead code across the compiler, backends, and pinned tests

*(初版のPR-08+PR-09を統合。統合理由: どちらも「削除のみ」でレビュー観点が同質。差分は大きいが削除は行単位で機械的に確認できる)*

### Background

参照ゼロを確認済みのデッドコードが計約2,000行ある。最大はcudaq/transpiler.pyの旧controlled walker一族約1,440行(BE-18 — 本番経路は`_emit_controlled_fallback`に移行済み。唯一の外部参照はテスト3本の私設メソッド直呼び=T-QUAL-1)。ほかに`capabilities.py`全体(BE-19)、`compile_check.py`、`result.py`のEmitResult一式、GateSpecレジストリ、CompositeDecomposer等(EM-20、TP-27)、`StrategyRegistry`+固定テスト1クラス(FE2-9/T-QUAL-2)、Qiskitの到達不能制御フローヘルパー3クラス(BE-3)。デッドコード内には「現役ならP1級」のバグが現役に見える形で残っており監査・レビューコストの温床。

### Goal

デッドコード約2,000行が一掃され、cudaq/transpiler.pyが約1,600行に半減し、能力表現が`supports_*()`一系統・戦略レジストリが`CompositeGate._strategies`一系統に統一される。削除された挙動保証は生存経路のテストに移植されている。

### Details

- **Stage A(cudaq旧walker)**: (1) `TestCudaqControlledSliceElementFallback`の3本が検証する意味論(固定slice要素解決/ループ依存sliceスロット再計算/シンボリックvector要素再シード)を`_emit_controlled_fallback`経由のend-to-endテストに書き直す。(2) 旧walker一族(:96-591, :1804-2751)+ヘルパ+旧テスト3本を削除。
- **Stage B(その他の削除)**: EM-20/BE-19/TP-27/BE-3/FE2-9/T-QUAL-2の各削除+小掃除(EM-21の`__builtins__`ハック、IR-20の遅延初期化辞書、IR-21の小粒、BE-24の空elseガード)。
- 備考: cudaq系テストは`pytest.mark.cudaq`配下でデフォルトCIでは走らない — **ローカル緑でもcudaqレーンで壊れる**失敗形に注意し`uv run pytest -m cudaq`を必ず実行。テスト側`gate_test_specs.py:3`のdocstring文言更新。BE-3削除時はQiskitの制御フロー対応がPass上書きで正であることをdocstringに明記。公開API面(`__init__.py`再エクスポート)を削除前に確認。

### ToDo List

- [ ] Stage A: 3本の意味論を生存経路end-to-endテストへ移植する(削除前にグリーン確認)
- [ ] Stage A: 旧walker一族・ヘルパ・旧テストを削除する
- [ ] Stage B: EM-20/BE-19/TP-27/BE-3/FE2-9/T-QUAL-2 を削除する
- [ ] Stage B: 小掃除(EM-21/IR-20/IR-21/BE-24)を適用する
- [ ] 全削除シンボルの残存参照ゼロをgrepで確認する

### Test Plan

- 移植テストが削除**前**のコードでもグリーンであることを先に確認(挙動等価の証明)。
- `uv run pytest tests/` + `-m ""` フルレーン全グリーン(追補2の裏取りどおりテスト修正はStage Aの3本+T-QUAL-2の1クラスのみのはず)。`uv run zuban check qamomile/` で未使用importなし。

---

## PR-09: Reduce recomputation across the pipeline, emit, and execution

*(初版のPR-10+PR-11+PR-12を統合。統合理由: すべて「挙動不変の性能改善」でレビュー観点が同一。Stage単位で独立に検証可能)*

### Background

性能上の無駄が3層に分布している。パイプライン層: 依存グラフ+測定テイントが1回のtranspileで実質5回構築され(TP-16)、常にFalseを返すデッドDFSが全探索コストを払う(TP-17)。emit層: reusable-gate非対応バックエンドでの捨て回路フルemit(EM-11/BE2-4)、unroll反復ごとの8-dict全コピー(EM-13)、LoopAnalyzerの4回走査×反復ごと再実行(EM-14)、serializeのネストBlock value_table共有によるペイロード膨張(IR-12)。実行・ビルド層: Qiskitの毎sample()フルtranspile(BE-12)、expvalごとのobservable変換再実行(BE-13)、サブカーネル特殊化の毎回full re-trace+ソース再パース(FE-6/FE-18)。

### Goal

同一カーネルの反復実行・大規模カーネルのコンパイルで、解析・transpile・変換・トレースの再計算が初回のみになり、計測可能な高速化とペイロード削減が得られる(挙動は完全不変)。

### Details

- **Stage A(パイプライン)**: `AnalysisResult`(依存グラフ+taint集合)をanalyze→classical_lowering→planで受け渡し(TP-16)、デッドDFS削除(TP-17)、TP-18/19/20/21/23の各効率化。
- **Stage B(emit)**: `blockvalue_to_gate`冒頭の`supports_reusable_gates()`事前チェック+QURI/CUDA-Qの防御オーバーライド2件削除(EM-11/BE2-4)、EmitContextのpush/pop化+typed-writeヘルパ公開(EM-13/CE-8)、LoopAnalyzer走査融合+`id(op)`メモ化(EM-14)、EM-15/16、TP-22、IR-14/15、serialize value_tableスコープ分割(IR-12/13 — スキーマバンプ不要のadditive変更)。
- **Stage C(キャッシュ)**: Qiskit transpileキャッシュ(BE-12: 未束縛回路で1回transpile→`assign_parameters`)、observable変換キャッシュ(BE-13: `CompiledExpvalSegment`に格納)、特殊化キャッシュ+`resolve_kernel_signature`統合(FE-6/FE-18/FE2-6)、BE-14/16/17。
- 備考: PR-08の先行を推奨(cudaqファイルの衝突回避)。キャッシュの無効化条件(bindings/shape変更、Hamiltonianのemit時スナップショット方針との整合)をdocstringに明記。**レビュー負荷が高い場合はStage Cを別PRに切り出す**(自然な分割点)。

### ToDo List

- [ ] Stage A: `AnalysisResult` 共有とデッドDFS削除、TP-18〜21/23
- [ ] Stage B: EM-11/BE2-4、EM-13/CE-8、EM-14、IR-12/13、残りの効率化
- [ ] Stage C: transpile/observable/特殊化キャッシュ、BE-14/16/17
- [ ] QAOA最適化ループと深いunrollカーネルのbefore/afterベンチをPRに記載する

### Test Plan

- 既存全テストグリーン(キャッシュ有無・スコープ変更で結果不変)。キャッシュ無効化テスト(異なるbindings/shapeで再計算されること)。
- serialize: ネストBlockカーネルのペイロード削減確認+round-trip不変。

---

## PR-10: Overhaul the test suite (fixtures, consolidation, and mandated coverage)

*(初版のPR-13+PR-14+PR-15を統合。統合理由: 全変更がtests/配下のみで本番挙動に影響しない。Stageの直列依存が強く分割の意味が薄い)*

### Background

テスト側に3つの構造問題がある。(1) 「唯一の正しいバックエンド列挙点」がなく、正規fixture `sdk_transpiler`と同等物が8箇所に別名再実装、ヘルパ・許容誤差・シードが多流派に散在、必須依存qiskitへの`importorskip`52ファイルが環境破損を無音化(T-STR-2/4/5/6、T-QUAL-8/13)。(2) スイートの約20%(約20,000行)がバックエンド別コピペ3ファイルに集中し同名テスト184本が3重メンテナンス、`except EmitError: pytest.skip`16箇所がemit退行を無音化(T-STR-1/3/7、T-QUAL-4/12)。(3) QAOAがCLAUDE.md必須要件に全面非準拠(docstringに反し`.sample()`を呼ばない)、analyze/orchestrator/LoopAnalyzer等の空白サブシステム(T-COV-11ほか)。**PR-13(controlled emission大改修)の前提**。

### Goal

バックエンド列挙・共有ヘルパ・許容誤差規約が各1箇所に定義され、共通シナリオが共有スイート1箇所+バックエンドアダプタ方式になり、emit退行がskipではなく失敗として検出され、algorithm/stdlibの全対象がsampling+expvalの両経路で3バックエンド実行される。

### Details

- **Stage A(基盤)**: `sdk_transpiler`のroot conftest昇格+ローカルfixture 7箇所削除、`tests/_helpers.py`/`tests/_tolerances.py`/`tests/_sdk_availability.py`新設、qiskit importorskip 52ファイル置換、`slow`マーカー導入。
- **Stage B(統合)**: `FrontendBackendAdapter`+共通シナリオモジュール新設、3ファイル共通59本→2ファイル共通分(184本ベース)の段階移行、固有テストの意図棚卸し(意図的固有か写経漏れか)、EmitError→skip 16箇所の期待マトリクス化、QURIテスト配置整理(T-STR-7)。
- **Stage C(カバレッジ)**: QAOA/FQAOAの`sdk_transpiler`版sampling+expvalテスト、Trotter/QPEのexpval経路、analyze(`DependencyError`発火)/inline/LoopAnalyzer/orchestratorの直接テスト、docstring欠落ワースト5ファイルの追記(T-STR-9)。
- 備考: エミッタ層で完成済みの`TranspilerTestSuite`パターンの水平展開が設計方針。バックエンド固有検証(cudaqソース断片アサーション等)は各ファイルに残す。**Stage Bの移行量が大きい場合、2ファイル共通分の移行をfollow-up PRに残す再分割点あり**(共通59本+期待マトリクス化までは本PRで必須 — PR-13の前提のため)。
- **既存issue #313(SDKテストテンプレート)は本PRと同一スコープ**(「各バックエンドの最小テストケースを定義する小さなテンプレートを用意し、既存および今後のquantum SDKで共通パターンを抽出する」)。**このPRを作成する際、本文に `Closes #313` を含めて #313 をリンク・クローズすること。**

### ToDo List

- [ ] Stage A: fixture一本化・ヘルパ/許容誤差モジュール新設・importorskip置換・slowマーカー
- [ ] Stage B: アダプタ+共通シナリオ化、共通テスト移行、期待マトリクス化、配置整理
- [ ] Stage C: QAOA/FQAOA/Trotter/QPEの必須カバレッジ、空白サブシステムの直接テスト、docstring追記
- [ ] 移行前後の収集数比較(テスト数が減っていないこと+cudaqで新たに走る共通シナリオ数)をPRに記載

### Test Plan

- `uv run pytest tests/`(デフォルト)と`-m ""`(フル)が置換前後で同等以上の収集数で全グリーン。
- 意図的にemitterを壊した状態で期待マトリクス外がskipでなく失敗になることをローカル確認。qiskit import不能環境でcollection errorになること。

---

## PR-11: Unify IR traversal and value-walking protocols

*(初版のPR-16+PR-17+PR-24を統合。統合理由: いずれも「IRの歩き方の一本化」で、Operation走査とValue木走査は同一の設計判断を共有する。slice_borrow分割は同じパス層の整理として同梱)*

### Background

横断バグパターン①②の恒久対策。7箇所以上のパスが手書きのネスト再帰を持ち、経路漏れバグの族を生んでいる(TP-26: SubstitutionPassのネスト未到達=TP-2、`lower_measure_qfixed`のトップレベル限定=TP-3、results非置換=TP-13等)。「Value参照を持つ場所のリスト」もcanonical/UUIDRemapper/ValueSubstitutor/constant_foldの4実装に手書き複製され(IR-17)、P1-5型バグの温床。Operationの追加Valueフィールドoverride 4連(IR2-2)、`next_version()`4実装のフィールド手書き列挙(IR2-4 — `ArrayValue`のフィールド脱落バグIR-6を既に生んだ)も同族。`slice_borrow_check.py`(2,121行、4責務同居)の分割(TP-25)も同じパス層の整理。

### Goal

Operation走査は「スコープ状態・phi認識付きの共通Transformer」1本、Value木走査は`map_value_tree`1実装、追加Valueフィールドは`EXTRA_VALUE_FIELDS`宣言1行、`next_version`は`dataclasses.replace`ベース1実装に統一され、経路漏れ・コピー漏れ型のバグが構造的に発生しなくなる。

### Details

- **Stage A(Operation走査)**: `OperationTransformer`拡張(スコープ状態フック・phiリスト識別・演算所有ブロック到達)→ substitution/lower_measure_qfixed/compile_time_if_lowering/inlineの手書き再帰を移設。バグ修正が自然に随伴: TP-2(訪問済みBlockのid()メモ付きネスト再帰)、TP-3、TP-13、TP-24/IR-19(phi二重置換)、TP-10(DCE純粋性ホワイトリスト)、IR2-8(`SingleBodyNestedOps`)。
- **Stage B(Value木)**: `map_value_tree`を`ir/value.py`に置きcanonical/UUIDRemapper/ValueSubstitutorをポリシー注入の薄いラッパに(IR-17)→`EXTRA_VALUE_FIELDS`導入で4クラスのoverrideを宣言化(IR2-2)→`next_version`統一(IR2-4、IR-6解消)→`remap_uuid_refs`フック(IR-18)、IR-9/IR-10/IR2-6(Block再構築の`dataclasses.replace`化、`output_names`欠落3サイト修正)/IR2-7、TP-12/TP-29。
- **Stage C(slice_borrow分割)**: `slice_borrow/`サブパッケージ(bound_tokens/state/guards/pass)へ分割+残骸変数削除(TP-25、挙動不変)。
- 備考: PR-05先行でリファクタ等価性検証に修正済み`content_hash`が使える。CE-7(op型ハンドラレジストリ)の設計はここで確定し、実装はPR-13/15と分担。**Stage Cは完全に独立なので、サイズ調整用の切り出し候補**。

### ToDo List

- [ ] Stage A: Transformer拡張と7箇所の移設、随伴バグ修正(TP-2/3/13/24等)
- [ ] Stage B: `map_value_tree`/`EXTRA_VALUE_FIELDS`/`next_version`統一、IR-18ほか
- [ ] Stage C: slice_borrow分割
- [ ] 全Operationサブクラス列挙の置換往復テスト(EXTRA_VALUE_FIELDS網羅)を追加する

### Test Plan

- リファクタ前後で代表カーネル群(qkernel_catalog)の`content_hash`一致+serialize round-trip一致(構造等価の機械検証)。
- 新規: `test_substitution_reaches_nested_callee`、ネストfor内MeasureQFixed分割、`ArrayValue.next_version`フィールド保存。slice系スイート(216本+119本)完全不変。

---

## PR-12: Unify value resolution and iteration engines across all IR consumers

*(初版のPR-18+PR-19を統合。統合理由: 反復エンジン(XC-3〜7)は値解決コア(XC-1)の上に載る従属実装で、直列依存が強い。消費者(estimator/visualization)の移行も一体で行う方が中間状態を作らない)*

### Background

「定数→パラメータ→コンテキスト→名前→配列要素」の値解決チェーンがリポジトリに**5系統**(transpiler×2=EM-19、estimator、visualization=XC-1)、スライスaffine合成が4実装(visualization版のみ契約ガード欠落=XC-2)、ループ境界解決+アンロール実行が3系統(XC-3)、制御フロー畳み込み走査骨格が5箇所(XC-5)、callee実引数バインディングが3系統(zip整列バグが独立に3回修正=XC-7)。transpilerが根絶した名前ベースのループ変数同一性をestimator/visualizationが再導入(XC-4)、analyzerはeval_utilsの「single source of truth」宣言を無視して古典演算評価を再実装(XC-6)。ClassicalExecutorの解決順序逆行(EM-6)、CE-9/CE-10/CE-5も同族。

### Goal

値解決・スライス合成・ループ反復・制御フロー畳み込み・callee束縛が共有部品になり、transpiler/estimator/visualizationが同一セマンティクスで動く。「物理qubitを静かに取り違える」ロジックの多重実装が解消される。

### Details

- **Stage A(解決コア)**: `ir/value_resolution.py`に解決コア(FailurePolicy: PERMISSIVE/STRICT)+出力ドメインアダプタ(Python数値/sympy/表示)。スライス合成は`resolve_root_array_index`(ガード込み)にresolverコールバック引数を追加して集約(XC-2解消)。transpiler 2クラス統合(EM-19)、EM-6/TP-9/CE-9/CE-10/CE-5適用。
- **Stage B(反復・畳み込みエンジン)**: `LoopIterationPlan`(PR-06の`resolve_for_bounds`利用、UUIDキー束縛内蔵=XC-4解消)、`fold_operations`集約走査エンジン(gate_counter/qubits_counter統合で約250行削減見込み=XC-5)、`bind_call_scope`1定義(XC-7)、analyzerのeval_utils置換(XC-6)、QFT見積り整合テスト(XC-10)。
- 備考: **estimator/visualizationを最初から消費者として設計する**(追補1の結論6 — transpilerだけ見て設計すると再実装が再発する)。PR-11(map_value_tree等)の後に着手。**再分割点: Stage AとBは別PRに切れる**が、Bの移行対象がAのAPI利用者そのものなので一体を推奨。

### ToDo List

- [ ] Stage A: 解決コア+ドメインアダプタ、transpiler 2クラス統合、estimator/visualization移行
- [ ] Stage B: `LoopIterationPlan`/`fold_operations`/`bind_call_scope`、analyzerのeval_utils置換
- [ ] 境界テスト(負index/シンボリック境界/スライスチェーン)をコアに集中的に追加
- [ ] ネスト・同名ループ変数カーネルで3サブシステムの反復回数一致テスト(XC-4再発防止)

### Test Plan

- estimator: 既存見積り値が全て不変(tests/circuit/estimator/ 全グリーン)。visualization: 代表カーネルの描画IRスナップショット不変。emit: 物理qubit割付不変(代表カーネル群で機械比較)。

---

## PR-13: Hoist the controlled-body walker and split controlled emission

*(初版PR-20のまま独立維持。理由: 意味論的に最も危険な変更であり、他と混ぜるべきでない)*

### Background

controlled fallback walkerが3重実装されており(共有版/QURI約740行/CUDA-Qデッドコピー=PR-08で削除済み)(EM-18/BE-20)、`controlled_emission.py`(3,211行)はトップレベルcontrolled-U 3経路が統一済みの`resolve_controlled_u_call`を使わずコピペ変奏、結果マッピング4重複(EM-17)。経路間の非一貫性バグ(EM-3)はこの構造の直接の産物。**PR-08とPR-10(期待マトリクス化済み)の完了が前提** — この経路は退行が最も起きやすく、最も見えにくい(追補2)。

### Goal

controlled-Uの走査・解決・SSA伝播がコアの`ControlledBodyWalker`1本になり、バックエンドは「多重制御プリミティブ1個のemit」フックのみを実装。`controlled_emission.py`は4〜5モジュールに分割される。

### Details

- **Stage A**: トップレベル3経路を`resolve_controlled_u_call`+単一lowering関数に統合、結果マッピング1本化(EM-17前半)、EM-3のnullチェック統一。
- **Stage B**: `ControlledBodyWalker`+`ControlledGateLowerer`ストラテジ(QURI=密行列/CUDA-Q=`.ctrl`/Qiskit=`Gate.control`)へ吊り上げ、QURI約740行を数十行に(EM-18)。
- **Stage C**: ファイル分割(`controlled_call_resolution`/`controlled_walker`/`multi_control_gates`/`nested_block_binding`)+EM-12/EM-22/EM-23/CE-4。
- 備考: T-QUAL-3で列挙したホワイトボックステスト(私設メソッド直叩き)は事前にend-to-end化。**Stage A/B/Cはそれぞれ独立にマージ可能な自然な再分割点**(レビュー負荷次第で2〜3PRへ)。

### ToDo List

- [ ] 影響するホワイトボックステストをend-to-end化する
- [ ] Stage A: 3経路統合+結果マッピング1本化+EM-3
- [ ] Stage B: walker吊り上げ+loweringストラテジ、QURI walker置換
- [ ] Stage C: ファイル分割+EM-12/22/23/CE-4

### Test Plan

- controlled系スイート(test_controlled.py 140本ほか)全グリーン — **PR-10の期待マトリクス化済みの状態で実行**(skip無音化なし)。
- QPE(controlled-U^2^k)の3バックエンド統計比較、ネストcontrolled-U結果マッピング回帰テスト。

---

## PR-14: Introduce table-driven operation, gate, and handle knowledge

*(初版PR-21のまま独立維持。理由: 6テーブルは相互独立で段階マージできる中規模PRであり、他へ混ぜると依存が複雑化する)*

### Background

「新ゲート/新opを追加したら1箇所で済む」構造がなく、知識が多重定義されている: serialize 28op×2の定型(IR-16)とprinterの同一構造知識の別実装(IR2-3)、GateEmitter 25メソッド写経×3+matchテーブル2系統(BE2-2)、回転ゲート集合×3(XC-8)、演算子シンボル表×2(XC-9)、ゲート逆元表×2(FE2-4)、Handle↔IR型対応×7(FE2-5 — FE-15の根因)、controls/targets/paramsレイアウト解析×4(IR2-1)、パラメトリック型のeq/hash手書き並行(IR2-5)。

### Goal

OpSpec(serialize+printer共用)、GATE_PROPERTIES、GATE_INVERSE_SPECS、HANDLE_SPECS、KIND_SYMBOLS、OperandLayoutの6テーブルが各知識の単一ソースになる。

### Details

- 実装は独立性の高い順: (1) GATE_PROPERTIES(XC-8解消、estimator/visualizationの表を置換)→(2) KIND_SYMBOLS/GATE_INVERSE_SPECS →(3) HANDLE_SPECS+HandleFactory(FE-15/FE-20/FE-21解消)→(4) OperandLayout(4クラスのプロパティ名統一)→(5) OpSpecテーブル(serializeテーブル駆動化+printerの同一テーブル参照+`HasNestedOps`使用)→(6) `ParametricValueType`基底(IR2-5)。
- 備考: GateEmitterの`emit_gate(kind, qubits, angle)`一本化(BE2-2の後半)は影響が広いためオプション — 実施時はPR-15と調整。閉じたディスパッチのセキュリティ不変条件はテーブルが静的なら維持。IR-7(ParamSlot round-trip忠実性)の方針決定を同時に文書化。

### ToDo List

- [ ] GATE_PROPERTIES / KIND_SYMBOLS / GATE_INVERSE_SPECS を導入し参照側を置換する
- [ ] HANDLE_SPECS + HandleFactory を導入する
- [ ] OperandLayout を導入し4クラスを移行する
- [ ] OpSpec で serialize と printer を駆動する
- [ ] `ParametricValueType` 基底へ5型を統一する

### Test Plan

- serialize既存67本+スキーマv1ピン止めで round-trip 完全一致。printerは代表カーネルのスナップショット比較(意図的変更は明記)。
- 「ダミーゲートをテーブルに1行追加すると全層に行き渡る」ことのテスト。

---

## PR-15: Unify backend emission and executor behavior

*(初版のPR-22+PR-26を統合。統合理由: どちらも「バックエンド層の共通化と対称化」で、変更ファイル群がほぼ同一(3バックエンド+コアexecutor周辺)。挙動変更を含むStageを明示的に分離する)*

### Background

バックエンド間で同一骨格が独立実装されている: `hamiltonian_to_*`スケルトン3重(BE2-1 — complex係数バグを実際に生んだ構造)、PauliEvolve前処理の3複製とドリフト4軸(BE2-3)、`_emit_if`定数ガード二重化(BE2-5)、生成ボイラープレート3重(BE2-6)、counts正規化5系統(BE2-7)、plan.stepsのisinstanceチェーン11箇所(CE-3)。またexecutor APIの非対称とサイレント失敗: Qiskitのみ欠落パラメータ無言スキップ(BE-5)、seed対応QURIのみ(BE-22)、回転角の無音0.0焼き込み(EM-2)、測定/cast解決失敗のwarning継続(EM-7)、`RuntimeError`握りつぶし(EM-8)ほか。

### Goal

observable変換・PauliEvolve前処理・counts正規化・EmitPass生成が共有部品になり、3バックエンドのexecutorが同一シグネチャ・同一失敗挙動(欠落=例外、seed対応)を持ち、emit時の解決失敗が例外として顕在化する。新バックエンド追加時に書くコードが「差分のみ」になる。

### Details

- **Stage A(純リファクタ、挙動不変)**: `convert_hamiltonian(h, sink)`+バックエンドsink(BE2-1)、`resolve_pauli_evolve()`/`map_pauli_evolve_results()`抽出(BE2-3)、`transpiler/counts.py`(BE2-7)、BE2-5/BE2-6/BE-23/CE-3、decompositionsのconformanceテスト(BE-21)。
- **Stage B(挙動変更を含む対称化)**: `executor(seed=...)`共通化(BE-22)、欠落パラメータ例外化(BE-5)、bound+params併用のValueError、BE-6/BE-11/BE-15、EM-2(theta非None全経路失敗→EmitError)、EM-7(MeasureOperationのEmitError昇格 — 必要ならopt-in緩和フラグ)、EM-8(`RuntimeError`を捕捉集合から除外)、EM-5/EM-10/EM-24/EM-25、CE-6残り(`CompletedJob`基底)。
- 備考: **PR-07(挙動差のバグ修正)マージ後に着手**(修正とリファクタを混ぜない)。warning→error昇格は既存コードを壊し得るためリリースノート必須+deprecation期間の要否をレビューで確定。**再分割点: Stage A(挙動不変)とStage B(挙動変更)は性質が違うため、レビュー方針次第で2PRに分離可**。

### ToDo List

- [ ] Stage A: 変換sink/PauliEvolve前処理/counts.py/BE2-5/6/BE-23/CE-3/BE-21
- [ ] Stage B: seed共通化・欠落例外化・EM-2/5/7/8/10・BE-6/11/15・CompletedJob
- [ ] 3バックエンドのseed付き再現性テストと欠落パラメータ同一例外テストを追加
- [ ] 挙動変更点のリリースノート原案をPRに添付

### Test Plan

- Stage A: 既存observable/expval/samplingテスト全グリーン+クロスバックエンド一致(atol=1e-8)不変。counts正規化の単体テスト(intキー/bool行/桁落ち/空白入り)を新モジュールに集約。
- Stage B: 同一seedで3バックエンドのsample再現、欠落パラメータで同一例外。warning→error昇格の影響テストを明示的に更新。

---

## PR-16: Unify frontend higher-order gate machinery and stdlib

*(初版PR-23のまま独立維持。理由: フロントエンドに閉じた変更で他PRとファイルが重ならない)*

### Background

control.py(2,148行)/inverse.py(1,862行)/composite_gate.pyが「既存カーネルを高階ゲート化する」同一機構を並行実装: 特殊化ブロック選択3重(FE2-1 — 可変状態コピーで既に挙動分岐)、呼び出しプロローグ3重(FE2-2)、**affine正しさ直結のVectorView借用移譲5重**(FE2-3)、量子パラメータ判定5箇所(FE2-7)、クローンヘルパ同文二重(FE2-11)。stdlibはQFT分解ボディ+リソース式が逐語5コピー(FE2-8)。本編FE-4/FE-7〜FE-23の残件も消化する。

### Goal

高階ゲートの共通機構(特殊化取得・引数束縛・結果再ラップ)が共有ヘルパに集約され、stdlib QFTの真実がStrategy1箇所になる。

### Details

- **Stage A(共有ヘルパ)**: `rewrap_result(template, value, operation_name)`新設と5箇所置換(FE2-3 — 最優先)、`QKernel.specialized_block_for`/`bind_call_arguments`への3重実装集約(FE2-1/2)、FE2-7/FE2-11、制御フロービルダー骨格共通化(FE2-10)。
- **Stage B(stdlib+残件)**: QFTのStrategy委譲化+リソース式1関数化(FE2-8)、FE-4(遅延束縛`__missing__`dict)、FE-7〜FE-14/FE-16/FE-19/FE-22/FE-23(HandleFactory参照への置換はPR-14導入後)。
- 備考: FE2-1のcontrol側追加ガード(all-Handleチェック)が仕様かコピー漏れかをこの機会に確定しdocstring化。**再分割点: Stage A(高階API)とStage B(stdlib+細部)**。

### ToDo List

- [ ] Stage A: `rewrap_result`/`specialized_block_for`/`bind_call_arguments`、FE2-7/10/11
- [ ] Stage B: stdlib QFT統合、FE-4、FE残件一式
- [ ] FE-4向けに後方定義ヘルパーカーネルの新規テストを追加する

### Test Plan

- affine/borrow系スイート(216本+119本)全グリーン — FE2-3置換で借用移譲セマンティクス不変であることの最重要検証。
- QFT/QPEのクロスバックエンドテスト不変。

---

## PR-17: Unify pass contracts, the error hierarchy, and documentation

*(初版のPR-25+PR-27を統合。統合理由: 契約の一元化と文書同期は「宣言と実装を一致させる」同一の仕事で、docstring修正が両方に跨る)*

### Background

BlockKind前提条件の検査が「raise/黙認/no-op」の3様式に分裂(TP-5/TP-6)、AnalyzePassは文書化された冪等契約に違反(TP-4)、power検証の例外型不一致(TP-8)。`MultipleQuantumSegmentsError`等が`QamomileCompileError`階層外、`Job.result()`のdocstringと実装が矛盾、ビルトインraise 8箇所(CE-2)。shape dim判定の名前正規表現依存(TP-14)。ドキュメント側も`transpiler.py`のdocstringパイプラインが実装と乖離(TP-28)、CLAUDE.mdの「全パス冪等」「デフォルト=docsスキップ」が不正確、stale参照(IR-11ほか)。

### Goal

`Pass`基底の`expected_kinds`一元検査、`except QamomileCompileError`での全捕捉、全パスの機械的な冪等スモークテスト、そして一次資料(docstring・CLAUDE.md)と実装の一致。

### Details

- **Stage A(契約)**: `expected_kinds`一元検査(TP-5/6統一)、AnalyzePassの冪等化(ANALYZED入力=no-op、TP-4)、例外2クラスのerrors.py移設+後方互換alias、ビルトインraiseのEmitError/ExecutionError置換、`Job.result()`契約一致(CE-2)、power検証統合(TP-8)、TP-11、TP-14(メタデータフラグ化)、`run(run(x))`冪等スモークのparametrize機械生成。
- **Stage B(文書)**: transpiler.pyの2 docstring同期(TP-28)、TP-7/TP-15、CLAUDE.mdの2記述修正、IR-11/FE-24/FE-25/staleテスト参照の一掃。
- 備考: PR-11(走査統一)後に実施すると冪等スモークが新基盤で書ける。例外型変更は既存の例外assert(77+箇所)の期待更新を伴う — リリースノート項目。

### ToDo List

- [ ] Stage A: expected_kinds/冪等化/例外階層整理/ビルトインraise置換/TP-14
- [ ] Stage A: 冪等スモークテストの機械生成
- [ ] Stage B: docstring・CLAUDE.md・stale参照の同期
- [ ] 例外型変更点のリリースノート原案を添付

### Test Plan

- 新規: 全パス×代表Blockの`run(run(x))`一致テスト。`except QamomileCompileError`で分割エラー/シグネチャ非互換が捕捉できるテスト。
- `uv run pytest -m docs -v` グリーン。既存例外assertの期待型更新を確認。

---

## 付録1: 初版(27 PR)→ 集約版(17 PR)の対応

| 集約版 | 統合した初版PR | 統合理由 |
|---|---|---|
| PR-01〜07 | PR-01〜07(不変) | バグ修正は独立・並行可能なまま維持 |
| PR-08 | 旧PR-08+09 | 削除のみ・レビュー観点同質 |
| PR-09 | 旧PR-10+11+12 | 挙動不変の性能改善で同質。Stage Cが再分割点 |
| PR-10 | 旧PR-13+14+15 | tests/のみで本番挙動不変。直列依存が強い |
| PR-11 | 旧PR-16+17+24 | 「IRの歩き方の一本化」で設計判断を共有 |
| PR-12 | 旧PR-18+19 | 反復エンジンは解決コアの従属実装 |
| PR-13 | 旧PR-20 | 最も危険な変更のため独立維持 |
| PR-14 | 旧PR-21 | 独立維持(6テーブルは内部で段階マージ可) |
| PR-15 | 旧PR-22+26 | バックエンド層の共通化+対称化で変更ファイルがほぼ同一 |
| PR-16 | 旧PR-23 | フロントエンドに閉じるため独立維持 |
| PR-17 | 旧PR-25+27 | 「宣言と実装の一致」という同一の仕事 |

## 付録2: 指摘ID → PR 対応表(逆引き)

| PR | 収容する指摘ID |
|---|---|
| PR-01 | P1-1, T-COV-1 |
| PR-02 | P1-2, FE-17, T-COV-2 |
| PR-03 | P1-3, FE-5, T-COV-3 |
| PR-04 | P1-4, T-COV-4 |
| PR-05 | P1-5, IR-2, IR-3, IR-4, IR-5, T-COV-5, T-COV-8 |
| PR-06 | EM-1, EM-4, EM-9, CE-1, CE-6(一部), T-COV-6, T-COV-10 |
| PR-07 | BE-1, BE-2, BE-4, BE-7, BE-8, BE-9, BE-10, T-COV-7 |
| PR-08 | BE-18, BE-19, BE-3, BE-24, EM-20, EM-21, TP-27, FE2-9, IR-20, IR-21, T-QUAL-1, T-QUAL-2 |
| PR-09 | TP-16〜21, TP-22, TP-23, EM-11, EM-13〜16, IR-12〜15, BE-12〜14, BE-16, BE-17, FE-6, FE-18, FE2-6, BE2-4, CE-8 |
| PR-10 | T-STR-1〜9, T-QUAL-4, T-QUAL-8, T-QUAL-12, T-QUAL-13, T-COV-11, 追補2カバレッジマップの空白 |
| PR-11 | TP-2, TP-3, TP-10, TP-12, TP-13, TP-24〜26, TP-29, IR-6, IR-8〜10, IR-17〜19, IR2-2, IR2-4, IR2-6〜8, T-COV-9, CE-7(設計) |
| PR-12 | XC-1〜7, XC-10, EM-6, EM-19, TP-9, CE-5, CE-9, CE-10 |
| PR-13 | EM-3, EM-12, EM-17, EM-18, EM-22, EM-23, BE-20, CE-4, T-QUAL-3(一部) |
| PR-14 | IR-7, IR-16, IR2-1, IR2-3, IR2-5, BE2-2, XC-8, XC-9, FE2-4, FE2-5, FE-15, FE-20, FE-21 |
| PR-15 | BE2-1, BE2-3, BE2-5〜7, BE-5, BE-6, BE-11, BE-15, BE-21〜23, EM-2, EM-5, EM-7, EM-8, EM-10, EM-24, EM-25, CE-3, CE-6(残り) |
| PR-16 | FE-4, FE-7〜FE-14, FE-16, FE-19, FE-22, FE-23, FE2-1〜3, FE2-7, FE2-8, FE2-10, FE2-11 |
| PR-17 | TP-4〜8, TP-11, TP-14, TP-15, TP-28, CE-2, IR-11, FE-24, FE-25, stale参照, CLAUDE.md記述 |
