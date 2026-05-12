---
name: generate_release_note
description: 前バージョンタグから現在のmainまでの変更をもとに，QamomileのEN/JAリリースノート(`docs/{en,ja}/release_notes/v<X_Y_Z>.md`)を作成する。構成順・スニペット検証・リンク規約・関連ファイル更新を定義。
---

# リリースノート生成スキル

`docs/en/release_notes/v<X_Y_Z>.md`と`docs/ja/release_notes/v<X_Y_Z>.md`を作成し，関連する目次ファイルも更新する。引数で前バージョンタグと新バージョンを受け取る（例: `/generate_release_note v0.11.1 v0.12.0`）。新バージョンが省略された場合は`pyproject.toml`等から推測して確認する。

## ワークフロー

### Phase 1: 変更の収集

1. `git tag --list | sort -V`で既存タグを確認し，前バージョンタグを特定
2. `git log <prev_tag>..HEAD --oneline --no-merges | wc -l`でコミット規模を把握
3. `git log <prev_tag>..HEAD --oneline --merges`でマージされたPR一覧を取得
4. `git diff <prev_tag>..HEAD --stat | tail -50`で大きく変わったファイル群を確認
5. `git log <prev_tag>..HEAD --diff-filter=A --name-only --format= | sort -u`で新規追加ファイルを抽出
6. 既存リリースノート(`docs/en/release_notes/v<最新>.md`)を読み，スタイル(言い回し，1行ブラブ，`pip install`，セクション構成)を確認

### Phase 2: 変更の分類

各変更をユーザー視点で以下のカテゴリに分類する。**コミットメッセージのprefix(feat/fix等)に依存せず**，実際にユーザーが触る面が変わるかで判断する。

| カテゴリ | 判定基準 |
|---|---|
| **破壊的変更** | 公開importパスの削除/移動，シグネチャ変更，挙動の互換性破壊。privateにムーブされた場合(`qamomile.optimization.utils` → `qamomile._utils`)も含む |
| **新機能** | ユーザーが直接呼び出す新しい公開API・新モジュール |
| **内部的な変更** | コンパイラ・フロントエンドの新しい能力。それ自体が公開APIの拡張(新パラメータ型，新パス等) |
| **バグ修正** | ユーザー視点で挙動が変わる修正(可視化の見た目，性能など) |
| **ドキュメント** | 新チュートリアル，新VQA，indexページの変更 |
| **DX/Tooling** | CI，lint設定，AGENTS.md等。**書かない**ことも多い(ユーザーに見えないため) |

### Phase 3: 構成順(必須)

セクションは**この順番**で並べる。

````markdown
# Qamomile vX.Y.Z

<1段落の概要 — このリリースの「大きな話」を1〜2文で>

```
pip install qamomile==X.Y.Z
```

## Breaking Changes        # 1. 破壊的変更(あれば)
## New Features            # 2. 新機能(目玉から順)
## Internal Changes        # 3. 内部的な変更
## Bug Fixes               # 4. バグ修正
## Documentation           # 5. ドキュメント
## Learn More              # 6. リンク
````

破壊的変更がない場合は`## Breaking Changes`セクション自体を省略する。`## DX / Tooling`を出すかは判断 — 大抵は出さない。

### Phase 4: セクション執筆ルール

#### `## New Features`

各機能ごとに `### <機能名>` で見出しを付け，以下を含める:

1. **1〜2段落の説明** — 何ができるか，どう動くかをユーザー視点で。最後にPRリンクをまとめる（必ず`[#NNN](https://github.com/Jij-Inc/Qamomile/pull/NNN)`形式でリンク化する。プレーンな`(#NNN)`は不可 — 詳細はPhase 6参照）
2. **使用例コード** — Phase 5で検証済みのもののみ
3. **期待される出力** — `print`系の出力がある場合は ` ```text` ブロックで，実際の出力に**完全一致**するもの
4. **チュートリアルへの導線** — 該当チュートリアルがあれば末尾に「See [Tutorial NN](...)」

#### `## Internal Changes`

セクション冒頭に，これらの内部的な変更が**何のためのものか**を1段落で書く。例:

> The Trotter feature above is built on three new compiler/frontend capabilities. They are independently usable, but their direct motivation is making `trotterized_time_evolution` expressible as natural Python.

各変更ごとに `### <変更名>` で見出しを付け，以下を含める:

1. **仕様の説明** — 何が変わったか，何ができるようになったか
2. **使用例コード** — Phase 5で検証済みのもの
3. **`**Why**:` 段落** — なぜこの変更を行ったか
   - メイン機能(目玉)を支えるための変更なら，その機能を引用してWhyに書く
   - **独立した変更ならそれと明記する**(無理に主機能と結びつけない — 例: 「**Unlike the items above, this is not part of the X feature**」)
   - 「ifサポートを強化したから boolも自然な完成形になった」のように，**設計思想**としてWhyを書くこともある(機能Aのために機能Bを「必要としている」とは限らない)

#### `## Bug Fixes`

箇条書き。1つ1つは1〜2文。**個別の可視化修正を全部列挙しない** — 「visualization polish:」のように束ねる。

#### `## Documentation`

新チュートリアル，新VQA，index更新。各エントリは1行で，**チュートリアル名はリンクテキスト**にする。EN版では「in EN and JA」と書かない(EN文書がJAに言及する必要はない)。

### Phase 5: スニペット検証(必須)

リリースノート内の**全ての**コード例は以下を満たすこと:

1. `QiskitTranspiler().transpile(kernel, bindings=...)` が成功する
2. `executable.sample(transpiler.executor(), shots=...).result()` が値を返す

検証手順:

```bash
# /tmp配下に検証スクリプトを書く(リポジトリには残さない)
cat > /tmp/verify_release_snippets.py <<'EOF'
import qamomile.circuit as qmc
from qamomile.qiskit import QiskitTranspiler
# ... 各スニペットをそのままコピーして実行
EOF
uv run python /tmp/verify_release_snippets.py
```

エラーが出たら**スニペット側を修正**する(機能側のバグでない限り)。修正後にmdへ反映。

#### よくある落とし穴

| 症状 | 原因 | 対処 |
|---|---|---|
| `EntrypointValidationError: ... quantum inputs/outputs` | quantum I/Oカーネルを直接`transpile()`に渡している | 外側にclassical I/Oのエントリポイントを書く(`qmc.qubit_array(...)`で確保→`qmc.measure()`で返す) |
| `AffineTypeError: Cannot return a value to 'q[0]' that was not borrowed` | `q[0] = my_kernel(q[0], ...)`という配列代入。アフィン型システムがユーザーカーネル経由のborrowを追えない | 単一qubitなら`qmc.qubit(name="q")` + `q = my_kernel(q, ...)`，配列なら`q = my_kernel(q, ...)`で全体を渡す |
| `MultipleQuantumSegmentsError: Found N quantum segments` | `parameters=[...]`で量子値の経路が分裂 | 全スカラーを`bindings`にまとめて渡す。**ランタイムパラメータを使わない** |
| `Line N: only the 'if' branch has a 'return' statement` | `@qkernel`内で`if`に`return`，後続コードに別の`return` | 全分岐に`return`を置くか，**末尾に1つだけ`return`を置く**(`if/else`構造に書き換え) |
| 出力が文書と食い違う | `Hamiltonian.terms`のreprは`(Z0, Z1)`であり`(Z(0), Z(1))`ではない等，実際のreprを確認していない | 検証スクリプトの出力をコピペでmdに貼る |

#### mdに含めるもの・含めないもの

- ✅ 含める: カーネル定義，`transpile()`呼び出し
- ❌ 含めない: `executable.sample(...)`等の実行行(検証用なので非掲載)

### Phase 6: リンク規約

| 種類 | 形式 |
|---|---|
| PR参照 | `[#NNN](https://github.com/Jij-Inc/Qamomile/pull/NNN)` — **必ずリンク化**。プレーンな`(#NNN)`は使わない |
| チュートリアル/最適化/VQAノートリンク | `https://github.com/Jij-Inc/Qamomile/blob/v<X.Y.Z>/docs/en/<section>/<file>.ipynb` 形式 — **リリースタグ付きGitHub blob URL**。`<section>`は`tutorial` / `optimization` / `vqa`など。ReadTheDocsホスト型URLや相対パスは使わない |
| GitHubリポジトリ | `https://github.com/Jij-Inc/Qamomile` |

`Learn More` / `さらに詳しく` セクションに Tutorials トップへのリンクは入れない — RTD ホストのサイドバー目次に常時表示されるため冗長になる。

タグ`v<X.Y.Z>`はリリース前なので一時的に404するが，リリース時に解決する旨を理解しておく。

### Phase 7: 関連ファイル更新(EN)

1. `docs/en/myst.yml`の`release_notes`セクションの`children`の**先頭**に追加:
   ```yaml
       - title: Release Notes
         file: release_notes/index.md
         children:
           - file: release_notes/v<X_Y_Z>.md   # 新規追加
           - file: release_notes/v<前>.md
           ...
   ```
2. `docs/en/release_notes/index.md`の**先頭**に1行追加:
   ```markdown
   - [v<X.Y.Z>](v<X_Y_Z>) — <1行サマリー: 主要機能を3つ程度，バッククォート使用OK>
   ```

### Phase 8: 日本語版作成

`/translate`スキルのルール(`.claude/skills/translate/SKILL.md`)に従って訳す。リリースノート固有の追加ルール:

#### 翻訳ルール(リリースノート版)

- **見出し**:
  - `Breaking Changes` → 破壊的変更
  - `New Features` → 新機能
  - `Internal Changes` → 内部的な変更
  - `Bug Fixes` → バグ修正
  - `Documentation` → ドキュメント
  - `Learn More` → さらに詳しく
- **`**Why**:`** はそのまま英語ラベルで残す(構造的なマーカーのため)
- **チュートリアル名**は日本語に訳す:
  - `Tutorial 07 — Hamiltonian Simulation` → `チュートリアル07 — ハミルトニアンシミュレーション`
- **チュートリアルリンク**は`docs/en/...`を`docs/ja/...`に変更:
  - `.../blob/v0.12.0/docs/en/tutorial/07_xxx.ipynb` → `.../blob/v0.12.0/docs/ja/tutorial/07_xxx.ipynb`
- **コード内コメント**は日本語に訳す。**コード本体は変更しない**
- **日本語と英数字の間にスペースを入れない**(translate skill ルール2)
- **広く認知されている技術用語は英語のまま**(`@qkernel`, `Hamiltonian`, `Vector[Observable]`, `pauli_evolve`, `Suzuki–Trotter`等)

#### JAファイル更新

1. `docs/ja/release_notes/v<X_Y_Z>.md`を作成
2. `docs/ja/myst.yml`の`release_notes` `children`先頭に追加
3. `docs/ja/release_notes/index.md`先頭に1行追加(リンクサマリーも日本語化)

## チェックリスト

最終確認:

- [ ] EN/JA両方の`v<X_Y_Z>.md`を作成
- [ ] **全コード例**が`transpile()` + `sample()`で成功(`/tmp/verify_*.py`で確認済み)
- [ ] mdには`transpile()`まで掲載，`sample()`は非掲載
- [ ] PR番号が**全て**リンク化されている
- [ ] チュートリアルリンクがGitHub blob URL + 正しいリリースタグ
- [ ] EN本文に「in EN and JA」のような表現なし
- [ ] 内部的な変更ごとに`**Why**:`段落あり
- [ ] 独立した内部変更には「not part of <主機能> feature」と明記
- [ ] EN/JA myst.ymlのtoc更新
- [ ] EN/JA release_notes/index.md先頭に1行追加
- [ ] JAは日英間にスペースなし
- [ ] JAのチュートリアル名が日本語に訳されている
- [ ] JAのチュートリアルリンクが`docs/ja/...`を指している

## 構成例(参考)

直近の例として[docs/en/release_notes/v0_12_0.md](../../../docs/en/release_notes/v0_12_0.md)を参照。Trotter機能を主軸に，それを支えるinternal changes(self-recursive `@qkernel`，`Vector[Observable]`，`bool`)，それと独立した変更(MLIR pretty-printer)を区別して書いている。
