# 作業ガイド（Codex / Claude Code / Gemini CLI 共通）

このリポジトリは JRA 中央競馬のデータ収集・可視化・予想ツールです。持ち主は Mac / iOS / Chrome 中心で、ときどき Windows を使います。説明やコメント、レポートの文言は日本語で書いてください。

## 構成

- `keiba/scraper/` netkeiba の取得（間隔制御・gzip キャッシュ）とパース。パーサーは純粋関数で、見出しの文字列で列を特定する。
  netkeiba で確定成績・開催日・レース一覧が取れないときは、JRA 公式サイト（`jra.py` がページをたどり、`jra_parsers.py` が同じ形に直す）から取る。JRA のページ指定（CNAME）の検査値は計算せず、前のページのリンクから読む。
- `keiba/db.py` SQLite。`races` `results` `entries` `payouts` `odds` `horses` `news` `analyses` `predictions`。
- `keiba/features.py` 特徴量。過去成績と出馬表を 1 つの表にして同じ処理を通す。
- `keiba/model.py` HistGradientBoosting → レース内ソフトマックス（温度 τ）→ オッズとの対数線形ブレンド。
- `keiba/betting.py` Harville による券種別確率、期待値、分数ケリー。
- `keiba/backtest.py` 戦略別の回収率。
- `keiba/report/` Jinja2 テンプレートの HTML。`stats.py` が馬柱・着別度数・コース傾向を集計する。
- `keiba/llm/` 作業者（`claude_api` `gemini_api` `openai_api` `claude_cli` `gemini_cli` `codex_cli`）と、`tasks.py` のプロンプト・出力検査。
- `keiba/pipeline.py` 収集 → 予想 → AI → HTML のつなぎ。`keiba/cli.py` がコマンド。モデルは DB ごとに `models/<DBファイル名>/` に置く。
- `keiba/demo.py` 架空データ。ネットにつながらない環境のテストはこれで動かす。

## 必ず守ること

1. **未来の情報を特徴量に混ぜない。** あるレースの特徴量はそのレースより前の日付の情報だけで作る。新しい特徴量を足したら `tests/test_features.py::test_no_future_leakage` が通ることを確かめ、必要ならその特徴量もテストの対象に入る（`FEATURES` に入れれば自動で対象になる）。
2. **netkeiba への負荷を増やさない。** `http.min_interval` の下限（1 秒）を下げない、並列取得しない、キャッシュを使う。
3. **レポートで中国語フォントを使わない。** `font-family` は日本語フォント（Hiragino / Yu Gothic / Meiryo / Noto Sans JP / IPA）だけ。`<html lang="ja">` を必ず付ける。`tests/test_pipeline.py` が禁止フォント名を検査している。
4. **AI の出力は検査してから使う。** 馬番・組番・金額は `llm/tasks.py::validate_prediction` を通す。Web 由来の URL は http(s) だけ許可する（`dedupe_sources` / `safe_url`）。
5. **CLI 作業者に危険な権限を与えない。** プロンプトに Web 由来の文章が入るので、書き込みやコマンド実行を自動承認するフラグ（`--yolo`、`--dangerously-*`、`danger-full-access` など）を既定にしない。既定のコマンドではプロンプトを標準入力で渡し（`{prompt}` を引数にしない）、Claude Code のツールは Web 検索だけに絞り、MCP サーバーも読み込ませない（`--strict-mcp-config`）。Codex の `--sandbox read-only` はファイルの読み取りを止めないことを忘れない。Web 由来の文章はプロンプトに JSON の文字列として入れ、データとして扱うよう指示する。
6. **回収率を誇張しない。** バックテストは学習・検証期間より後で行い、確定オッズを使っている限界を明記する。デモデータの結果を実力として扱わない。
7. 変更したら `pytest` を通す。パーサーを直すときは、実ページの HTML（`data/raw/` に gzip で保存されている。パースに失敗したページは `data/raw/_failed/` に移される）から必要部分を `tests/fixtures/` に入れてテストを足す。

## 改善タスク候補

番号で依頼されたらここから選んで実装する。終わったら、このリストから消すか状態を書き換える。

1. **実ページでのパーサー確認**: `keiba collect race <race_id>` を数レース分実行し、`data/raw/` の実 HTML で `parsers.py` を検証・修正する。結果ページ・出馬表・オッズ API・血統ページのそれぞれについて fixture を実データ由来のものに差し替える。
2. **オッズ API の券種追加**: `api_get_jra_odds.html` の type=4（馬連）、type=5（ワイド）、type=7（三連複）などを取得し、`betting.combo_probs` と組み合わせて組み合わせ馬券の期待値を出す。
3. **馬ページからの過去成績取り込み**: DB の収集期間より前の近走を、馬ページの成績表から（レースページを全部取らずに）取り込む。
4. **ペース・展開の特徴量**: 出走馬の脚質（`h_style`）からレースの想定ペース（逃げ・先行馬の数）を作り、各馬の脚質との相性を特徴量にする。
5. **調教データ**: 調教タイムを取得できるデータ源を調べて特徴量にする（netkeiba の調教ページはプレミアム会員限定の項目が多い点に注意）。
6. **定期実行**: 開催日の朝と締切 15 分前に `keiba day` を回す仕組み（macOS の launchd 例など）を `docs/` に書く。
7. **JRA-VAN Data Lab 対応**: Windows で JV-Link から取得したデータを同じ DB スキーマに入れる取り込み処理（`keiba/sources/jravan.py` など）。
8. **ウォークフォワード学習**: バックテストで月ごとに学習し直す方式を追加し、単一分割との差を比べる。
9. **複勝・ワイドのモデル**: Harville は人気薄の2・3着確率を過小評価しがちなので、3着内を直接学習するモデルを追加し、キャリブレーションを比べる。
