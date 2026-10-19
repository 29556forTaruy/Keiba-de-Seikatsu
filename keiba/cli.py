"""コマンドライン: keiba <コマンド>

例:
  keiba demo                                   # 架空データでお試し
  keiba collect results --start 2021-01-01 --end 2025-12-31
  keiba train --train-end 2024-12-31 --valid-end 2025-06-30
  keiba backtest --start 2025-07-01 --end 2025-12-31
  keiba day --date 2026-10-25 --llm --news     # 当日の予想一式
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from keiba import __version__, db
from keiba.config import Settings, load_settings
from keiba.scraper.http import FetchError


def _date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"日付は YYYY-MM-DD で指定してください: {s}") from exc


def _roles(args: argparse.Namespace) -> set[str]:
    roles = set()
    if getattr(args, "ai", False):
        roles = {"news", "predict", "review"}
    if getattr(args, "news", False):
        roles.add("news")
    if getattr(args, "llm", False):
        roles.add("predict")
    if getattr(args, "review", False):
        roles.add("review")  # 予想を作らない場合は保存済みのAI予想を点検する
    return roles


def _print_pages(pages: list[Path]) -> None:
    if not pages:
        print("レポートは作成されませんでした")
        return
    index = [p for p in pages if p.name == "index.html"]
    print(f"レポートを {len(pages)} ファイル作成しました")
    for p in index or pages[:5]:
        print(f"  {p}")
    if index:
        opener = "start" if sys.platform.startswith("win") else "open"
        print(f"ブラウザで開く: {opener} {index[0]}")


# ---------------------------------------------------------------- 各コマンド

def cmd_init(settings: Settings, args: argparse.Namespace) -> None:
    db.connect(settings.db_path).close()
    print(f"DBを用意しました: {settings.db_path}")


def cmd_status(settings: Settings, args: argparse.Namespace) -> None:
    from keiba.pipeline import latest_model_path, models_dir

    conn = db.connect(settings.db_path)
    q = lambda sql: conn.execute(sql).fetchone()  # noqa: E731
    races = q("SELECT COUNT(*), MIN(date), MAX(date) FROM races WHERE race_id IN (SELECT race_id FROM results)")
    print(f"DB: {settings.db_path}")
    print(f"  確定レース: {races[0]}件 ({races[1]} 〜 {races[2]})")
    print(f"  成績行: {q('SELECT COUNT(*) FROM results')[0]}  馬: {q('SELECT COUNT(DISTINCT horse_id) FROM results')[0]}頭")
    print(f"  出馬表: {q('SELECT COUNT(DISTINCT race_id) FROM entries')[0]}レース  血統: {q('SELECT COUNT(*) FROM horses')[0]}頭")
    print(f"  直前情報: {q('SELECT COUNT(*) FROM news')[0]}件  AI予想: {q('SELECT COUNT(*) FROM analyses')[0]}件")
    upcoming = conn.execute(
        "SELECT date, COUNT(*) FROM races WHERE race_id NOT IN (SELECT race_id FROM results) GROUP BY date ORDER BY date"
    ).fetchall()
    for d, n in upcoming[-5:]:
        print(f"  未確定: {d} {n}レース")
    path = latest_model_path(settings)
    print(f"  最新モデル: {path if path else f'なし (keiba train で {models_dir(settings)} に作成)'}")


def cmd_demo(settings: Settings, args: argparse.Namespace) -> None:
    from keiba import demo

    path = settings.db_path
    if path.exists() and not args.force:
        print(f"{path} は既にあります。作り直すなら --force を付けてください")
        return
    if path.exists():
        for suffix in ("", "-wal", "-shm"):
            Path(str(path) + suffix).unlink(missing_ok=True)
    conn = db.connect(path)
    upcoming = demo.generate(conn, seed=args.seed, weeks=args.weeks)
    conn.close()
    flag = f" --db {args.db or 'data/demo.sqlite'}"
    # 生成した期間を 学習6割 / 検証2割 / バックテスト2割 に分けた例を示す
    first = date.fromisoformat(db.connect(path).execute("SELECT MIN(date) FROM races").fetchone()[0])
    span = (upcoming - first).days
    train_end = first + timedelta(days=int(span * 0.6))
    valid_end = first + timedelta(days=int(span * 0.8))
    print()
    print("次の手順で一通り試せます:")
    print(f"  keiba{flag} train --train-end {train_end} --valid-end {valid_end}")
    print(f"  keiba{flag} backtest --start {valid_end + timedelta(days=1)} --end {upcoming - timedelta(days=1)}")
    print(f"  keiba{flag} day --date {upcoming} --offline")


def _fetcher(settings: Settings, args: argparse.Namespace):
    from keiba.scraper.http import Fetcher

    return Fetcher.from_settings(settings, offline=getattr(args, "offline", False))


def cmd_collect(settings: Settings, args: argparse.Namespace) -> None:
    from keiba.scraper import collect

    conn = db.connect(settings.db_path)
    fetcher = _fetcher(settings, args)
    if args.what == "results":
        n = collect.collect_results(conn, fetcher, args.start, args.end, refresh=args.refresh)
        print(f"新たに {n} レースを保存しました")
    elif args.what == "race":
        for race_id in args.race_ids:
            saved = collect.collect_race(conn, fetcher, race_id, refresh=args.refresh)
            print(f"{race_id}: {'保存' if saved else '取得済み'}")
    elif args.what == "card":
        if args.date:
            collect.collect_cards_for_date(conn, fetcher, args.date)
        for race_id in args.race_ids or []:
            parsed = collect.collect_card(conn, fetcher, race_id)
            print(f"{race_id}: {parsed['race'].get('race_name')} {len(parsed['entries'])}頭")
    elif args.what == "odds":
        ids = list(args.race_ids or [])
        if args.date:
            ids += db.race_ids_on(conn, args.date.isoformat())
        for race_id in ids:
            parsed = collect.collect_odds(conn, fetcher, race_id)
            print(f"{race_id}: {len(parsed['odds'])}頭 (状態 {parsed['status']}, {parsed['official_datetime']})")
    elif args.what == "pedigree":
        n = collect.collect_pedigrees(conn, fetcher, limit=args.limit)
        print(f"血統 {n}頭を保存しました")
    elif args.what == "history":
        n = collect.collect_horse_histories(conn, fetcher, args.horse_ids)
        print(f"過去レース {n}件を保存しました")


def cmd_train(settings: Settings, args: argparse.Namespace) -> None:
    from keiba import features, model
    from keiba.pipeline import save_model

    conn = db.connect(settings.db_path)
    runs = features.load_runs(conn)
    if runs.empty:
        raise SystemExit("DBにデータがありません")
    print(f"学習中… (データ {runs['race_id'].nunique()}レース)")
    bundle = model.train(runs, train_end=args.train_end.isoformat(), valid_end=args.valid_end.isoformat(), version=args.name)
    path = save_model(settings, bundle)
    m = bundle.metrics
    v = m.get("valid", {})
    print(f"保存しました: {path}")
    print(f"  学習 {m['train_races']}レース / 検証 {m['valid_races']}レース / 反復 {m['iterations']}")
    if v:
        print("  検証期間の精度 (対数損失は小さいほど良い):")
        print(f"    モデル単独 対数損失 {v['nll_model']:.3f}  本命的中 {v['top1_model']:.1%}")
        print(f"    オッズ     対数損失 {v['nll_market']:.3f}  本命的中 {v['top1_market']:.1%}")
        print(f"    ブレンド   対数損失 {v['nll_blend']:.3f}  本命的中 {v['top1_blend']:.1%}")
    print(f"  ブレンド係数 (モデル, 市場) = {m['blend']}")


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> None:
    from keiba import backtest, features, model
    from keiba.pipeline import load_model
    from keiba.report import render

    conn = db.connect(settings.db_path)
    bundle = load_model(settings, args.model)
    if bundle is None:
        raise SystemExit("モデルがありません。先に keiba train を実行してください")
    runs = features.load_runs(conn)
    res = backtest.run_backtest(conn, bundle, args.start.isoformat(), args.end.isoformat(), settings.betting, runs=runs)
    for w in res["warnings"]:
        print(f"注意: {w}")
    print(f"{res['start']}〜{res['end']} {res['races']}レース (モデル {res['model_version']})")
    print(f"{'戦略':<14}{'購入':>6}{'的中率':>8}{'回収率':>8}{'収支':>14}")
    for s in res["strategies"]:
        print(f"{s['name']:<14}{s['bets']:>6}{s['hit_rate']:>8.1%}{s['roi']:>8.1%}{s['profit']:>+14,.0f}")
    print(f"ケリー資金管理: {res['start_bankroll']:,.0f}円 → {res['final_bankroll']:,.0f}円")
    feat = features.build_features(runs, bundle.baseline)
    period = feat[(feat["date"] >= str(args.start)) & (feat["date"] <= str(args.end))]
    test_metrics = model.evaluate(bundle, period)
    is_demo = conn.execute("SELECT 1 FROM races WHERE source = 'demo' LIMIT 1").fetchone() is not None
    out = settings.path("reports") / f"backtest_{args.start:%Y%m%d}_{args.end:%Y%m%d}.html"
    render.render_backtest(res, out, test_metrics=test_metrics, is_demo=is_demo)
    print(f"レポート: {out}")
    if args.json:
        Path(args.json).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def cmd_predict(settings: Settings, args: argparse.Namespace) -> None:
    from keiba.pipeline import build_reports, load_model

    conn = db.connect(settings.db_path)
    bundle = load_model(settings, args.model)  # --model の指定ミスは収集の前に知らせる
    if args.collect:
        from keiba.scraper import collect

        fetcher = _fetcher(settings, args)
        try:
            for race_id in args.race_ids:
                if not db.has_results(conn, race_id):
                    collect.collect_card(conn, fetcher, race_id)
                    try:
                        collect.collect_odds(conn, fetcher, race_id)
                    except FetchError:
                        raise
                    except Exception as exc:
                        print(f"{race_id}: オッズ取得スキップ ({exc})")
        except FetchError as exc:  # アクセス拒否・通信断. DBにある分だけで作る
            print(f"注意: 取得に失敗したので、DBにあるデータだけでレポートを作ります ({exc})")
    pages = build_reports(conn, settings, args.race_ids, bundle=bundle,
                          roles=_roles(args), llm_filter="all", budget=args.budget)
    _print_pages(pages)


def cmd_day(settings: Settings, args: argparse.Namespace) -> None:
    from keiba.pipeline import run_day

    conn = db.connect(settings.db_path)
    pages = run_day(
        conn, settings, args.date, offline=args.offline, odds=not args.no_odds, history=args.history,
        pedigree=not args.no_pedigree, roles=_roles(args), llm_filter=args.llm_races, venue=args.venue,
        model_path=args.model,
    )
    _print_pages(pages)


def cmd_report(settings: Settings, args: argparse.Namespace) -> None:
    from keiba import features
    from keiba.report import render

    conn = db.connect(settings.db_path)
    runs = features.load_runs(conn)
    if runs.empty:
        raise SystemExit("DBにデータがありません。先に collect か demo を実行してください")
    feat = features.build_features(runs, features.SpeedBaseline().fit(runs[runs["is_future"] == 0]))
    is_demo = conn.execute("SELECT 1 FROM races WHERE source = 'demo' LIMIT 1").fetchone() is not None
    surface = {"ダート": "ダ", "障害": "障"}.get(args.surface, args.surface)
    race = {"venue": args.venue, "surface": surface, "distance": args.distance}
    out = settings.path("reports") / f"course_{args.venue}{surface}{args.distance}.html"
    render.render_course(race, feat, out, is_demo=is_demo)
    print(f"レポート: {out}")


# ---------------------------------------------------------------- 引数

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="keiba", description="中央競馬のデータ収集・可視化・予想")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--db", help="SQLiteファイル (既定: data/keiba.sqlite、demo は data/demo.sqlite)")
    p.add_argument("--config", help="設定ファイル (既定: ./config.toml)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="DBを作る")
    s.set_defaults(func=cmd_init)
    s = sub.add_parser("status", help="DBとモデルの状態")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("demo", help="架空データのDBを作る (ネット不要)")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--weeks", type=int, default=104)
    s.add_argument("--force", action="store_true", help="既存のデモDBを作り直す")
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("collect", help="netkeibaからデータを集める")
    cs = s.add_subparsers(dest="what", required=True)
    c = cs.add_parser("results", help="期間内の確定成績")
    c.add_argument("--start", type=_date, required=True)
    c.add_argument("--end", type=_date, required=True)
    c.add_argument("--refresh", action="store_true", help="取得済みも取り直す")
    c = cs.add_parser("race", help="指定レースの確定成績")
    c.add_argument("race_ids", nargs="+")
    c.add_argument("--refresh", action="store_true")
    c = cs.add_parser("card", help="出馬表")
    c.add_argument("--date", type=_date)
    c.add_argument("race_ids", nargs="*")
    c = cs.add_parser("odds", help="単勝・複勝オッズ")
    c.add_argument("--date", type=_date)
    c.add_argument("race_ids", nargs="*")
    c = cs.add_parser("pedigree", help="血統 (未取得の馬)")
    c.add_argument("--limit", type=int)
    c = cs.add_parser("history", help="指定馬の過去レースを補完")
    c.add_argument("horse_ids", nargs="+")
    for c in cs.choices.values():
        c.add_argument("--offline", action="store_true", help="通信せずキャッシュだけ使う")
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("train", help="勝率モデルを学習")
    s.add_argument("--train-end", type=_date, required=True, help="この日までで学習")
    s.add_argument("--valid-end", type=_date, required=True, help="この日までで確率の補正を調整")
    s.add_argument("--name", help="モデル名 (既定: 日時)")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("backtest", help="過去期間で回収率を検証")
    s.add_argument("--start", type=_date, required=True)
    s.add_argument("--end", type=_date, required=True)
    s.add_argument("--model", help="モデルファイル (既定: 最新)")
    s.add_argument("--json", help="結果をJSONでも保存")
    s.set_defaults(func=cmd_backtest)

    def ai_flags(c: argparse.ArgumentParser) -> None:
        c.add_argument("--news", action="store_true", help="作業者に直前情報を集めさせる (Web検索)")
        c.add_argument("--llm", action="store_true", help="作業者に予想(印・買い目・見解)を書かせる")
        c.add_argument("--review", action="store_true", help="別の作業者にセカンドオピニオンを書かせる")
        c.add_argument("--ai", action="store_true", help="上の3つをすべて")
        c.add_argument("--model", help="モデルファイル (既定: 最新)")

    s = sub.add_parser("predict", help="指定レースの予想レポート")
    s.add_argument("race_ids", nargs="+")
    s.add_argument("--collect", action="store_true", help="出馬表とオッズを先に取得する")
    s.add_argument("--budget", type=int, help="AI予想の予算 (円)")
    ai_flags(s)
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("day", help="開催日1日分: 収集→予想→レポート")
    s.add_argument("--date", type=_date, required=True)
    s.add_argument("--venue", help="競馬場で絞る (例: 東京)")
    s.add_argument("--offline", action="store_true", help="収集せずDBにあるデータだけで作る")
    s.add_argument("--no-odds", action="store_true", help="オッズを取得しない")
    s.add_argument("--no-pedigree", action="store_true", help="血統を取得しない")
    s.add_argument("--history", action="store_true", help="出走馬の過去レースを補完する (遅い)")
    s.add_argument("--llm-races", default="main", help="AIを使うレース: main(11Rと重賞) / all / 10,11,12")
    ai_flags(s)
    s.set_defaults(func=cmd_day)

    s = sub.add_parser("report", help="コース傾向のレポート")
    s.add_argument("--venue", required=True, help="例: 東京")
    s.add_argument("--surface", required=True, help="芝 / ダ")
    s.add_argument("--distance", type=int, required=True)
    s.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    db_arg = args.db
    if args.command == "demo" and not db_arg:
        db_arg = "data/demo.sqlite"
    settings = load_settings(args.config, db=db_arg)
    try:
        args.func(settings, args)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"エラー: {exc}") from None
    except FetchError as exc:  # アクセス拒否・通信断 (http.py が利用者向けの説明を書いている)
        raise SystemExit(
            f"エラー: {exc}\n"
            "通信できないときは、day は --offline、collect は --offline (キャッシュだけ使う)、"
            "predict は --collect なしで、DBにあるデータだけで実行できます"
        ) from None


if __name__ == "__main__":
    main()
