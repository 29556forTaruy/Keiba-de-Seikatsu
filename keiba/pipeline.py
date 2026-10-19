"""収集 → 特徴量 → 予想 → (LLM作業者) → HTML をつなぐ処理."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from keiba import db
from keiba import features as F
from keiba.betting import finish_probs, place_slots
from keiba.config import Settings
from keiba.llm import WorkerError, make_worker
from keiba.llm.base import dedupe_sources
from keiba.llm import tasks as llm_tasks
from keiba.model import ModelBundle
from keiba.report import render, stats

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


# ---------------------------------------------------------------- モデルの保存場所

def models_dir(settings: Settings) -> Path:
    """モデルの置き場所. DBごとに分ける (デモDBで学習したモデルが実データの予想に使われないように)."""
    return settings.path("models") / settings.db_path.stem


def latest_model_path(settings: Settings) -> Path | None:
    models = models_dir(settings)
    pointer = models / "LATEST"
    if pointer.exists():
        path = models / pointer.read_text().strip()
        if path.exists():
            return path
    found = sorted(models.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def save_model(settings: Settings, bundle: ModelBundle) -> Path:
    models = models_dir(settings)
    bundle.metrics["db"] = str(settings.db_path.resolve())  # どのDBで学習したか
    path = bundle.save(models / f"{bundle.version}.joblib")
    (models / "LATEST").write_text(path.name)
    return path


def load_model(
    settings: Settings, path: str | Path | None = None, *, progress: Progress = print
) -> ModelBundle | None:
    """モデルを読む. path 未指定なら このDBの最新、なければ None (市場の勝率だけで予想する).

    path を指定したのに見つからないときは、黙って市場だけにせず FileNotFoundError にする。
    ファイル名だけ (keiba train --name の名前) でも、このDBのモデル置き場から探す。
    """
    if path:
        given, models = Path(path), models_dir(settings)
        for cand in (given, models / given, models / f"{given}.joblib"):
            if cand.is_file():
                return ModelBundle.load(cand)
        raise FileNotFoundError(f"モデルファイルが見つかりません: {path}")
    found = latest_model_path(settings)
    if found is None:
        if any(settings.path("models").glob("*.joblib")):
            progress(f"注意: {settings.path('models')} 直下の古いモデルは使いません (モデルはDBごとに "
                     f"{models_dir(settings)} に置きます)。keiba train で学習し直すか --model で指定してください")
        return None
    bundle = ModelBundle.load(found)
    trained_on = bundle.metrics.get("db")
    if trained_on and trained_on != str(settings.db_path.resolve()):
        progress(f"注意: モデル {found.name} は別のDB ({trained_on}) で学習したものです")
    return bundle


# ---------------------------------------------------------------- 予想

def market_only_predict(rows: pd.DataFrame) -> pd.DataFrame:
    """モデル未学習のとき: オッズがあれば市場の勝率、なければ均等."""
    out = rows[["race_id", "horse_id", "horse_name", "number", "win_odds", "p_market"]].copy()
    n = out.groupby("race_id")["horse_id"].transform("size")
    out["p_model"] = np.nan
    out["p_win"] = out["p_market"].fillna(1.0 / n)
    out["p_win"] = out["p_win"] / out.groupby("race_id")["p_win"].transform("sum")
    for col in ("p_top2", "p_top3", "p_place"):
        out[col] = np.nan
    for _, idx in out.groupby("race_id").groups.items():
        probs = finish_probs(out.loc[idx, "p_win"].to_numpy())
        out.loc[idx, "p_top2"] = probs["top2"]
        out.loc[idx, "p_top3"] = probs["top3"]
        out.loc[idx, "p_place"] = probs["top2" if place_slots(len(idx)) == 2 else "top3"]
    return out


def is_target(race: dict[str, Any], llm_filter: str) -> bool:
    """LLMを使うレースか. 'all' / 'main'(11Rと重賞) / '10,11,12'(レース番号)"""
    if llm_filter == "all":
        return True
    if llm_filter == "main":
        return race.get("race_num") == 11 or race.get("grade") in ("G1", "G2", "G3")
    wanted = {int(x) for x in llm_filter.replace(" ", "").split(",") if x.isdigit()}
    return race.get("race_num") in wanted


def _news_dict(conn: sqlite3.Connection, race_id: str) -> dict[str, Any] | None:
    items = db.latest_news(conn, race_id)
    return items[0] if items else None


def _attempt(label: str, progress: Progress, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """AI作業者の1役割を実行する. 失敗しても (APIキー未設定・通信断・出力不正など) None を返して先へ進む."""
    try:
        return fn(*args, **kwargs)
    except WorkerError as exc:
        progress(f"{label}: 作業者エラー ({exc})")
    except Exception as exc:  # SDKの想定外の例外でも、統計の予想ページと一覧は必ず作る
        log.info("作業者の想定外のエラー", exc_info=True)
        progress(f"{label}: 作業者エラー ({exc.__class__.__name__}: {exc})")
    return None


def _drop_stale_cards(runs: pd.DataFrame, race_ids: list[str], progress: Progress) -> pd.DataFrame:
    """対象レース以外の未確定レース (出馬表だけの行) を除く.

    成績を取り込んでいない過去の出馬表が残っていると、その馬の前走が「着順なしの1走」として数えられ、
    近走の特徴量が欠けた (学習時には無い) 組み合わせになる。対象より後の出馬表は特徴量に関係しない。
    """
    targeted = runs["race_id"].isin(race_ids)
    keep = (runs["is_future"] == 0) | targeted
    if targeted.any():
        stale = runs[~keep & (runs["date"] < runs.loc[targeted, "date"].min())]
        if not stale.empty:
            first, last = stale["date"].min().date(), stale["date"].max().date()
            progress(f"注意: 成績が未取得の過去レースが {stale['race_id'].nunique()}件あります ({first}〜{last})。"
                     f"前走の情報が欠けるので keiba collect results --start {first} --end {last} で取り込んでください")
    return runs[keep].reset_index(drop=True)


# 一覧用の要約に残す本命馬の項目
INDEX_TOP_KEYS = ("number", "horse_name", "p_win")


def _save_summary(path: Path, item: dict[str, Any]) -> None:
    """一覧用の要約をレースページの隣に保存する (あとで一覧をその日の全レースから作り直すため)."""
    top = item.get("top")
    item = {**item, "top": {k: top.get(k) for k in INDEX_TOP_KEYS} if top else None}
    path.write_text(json.dumps(render.np_safe(item), ensure_ascii=False, default=str), encoding="utf-8")


def _load_summaries(day_dir: Path) -> list[dict[str, Any]]:
    """その日のレースページ (HTMLが残っているもの) の要約をすべて読む."""
    items = []
    for p in sorted(day_dir.glob("*.summary.json")):
        if not p.with_name(p.name.removesuffix(".summary.json") + ".html").exists():
            continue
        try:
            items.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            log.warning("一覧用の要約を読めません: %s", p)
    return items


def build_reports(
    conn: sqlite3.Connection,
    settings: Settings,
    race_ids: Iterable[str],
    *,
    bundle: ModelBundle | None,
    roles: set[str] = frozenset(),
    llm_filter: str = "main",
    budget: int | None = None,
    progress: Progress = print,
) -> list[Path]:
    """指定レースの予想HTMLと、日付ごとの一覧HTMLを作る."""
    race_ids = list(dict.fromkeys(race_ids))
    if not race_ids:
        return []
    runs = F.load_runs(conn)
    if runs.empty:
        raise ValueError("DBにデータがありません。先に collect か demo を実行してください")
    runs = _drop_stale_cards(runs, race_ids, progress)
    baseline = bundle.baseline if bundle else F.SpeedBaseline().fit(runs[runs["is_future"] == 0])
    feat = F.build_features(runs, baseline)
    target = feat[feat["race_id"].isin(race_ids)]
    missing = set(race_ids) - set(target["race_id"])
    for rid in sorted(missing):
        progress(f"{rid}: 出馬表・成績がDBにないためスキップ")
    if target.empty:
        return []
    pred_all = bundle.predict(target) if bundle else market_only_predict(target)
    if bundle:
        db.save_predictions(conn, [
            {"race_id": r.race_id, "model_version": bundle.version, "horse_id": r.horse_id, "number": r.number,
             "score": float(r.score), "p_win": float(r.p_win), "p_top3": float(r.p_top3),
             "p_market": None if pd.isna(r.p_market) else float(r.p_market)}
            for r in pred_all.itertuples()
        ])
        conn.commit()

    by_horse = stats.index_by_horse(feat, target["horse_id"].unique())
    top2 = stats.top2_by_race(feat)
    betting = dict(settings.betting)
    budget = budget or int(betting["bankroll"] * betting["race_budget_ratio"])
    model_info = None
    if bundle:
        model_info = {"version": bundle.version, "metrics": bundle.metrics,
                      "train_end": bundle.train_end, "valid_end": bundle.valid_end}
    else:
        progress("学習済みモデルがないため、オッズ(市場)の勝率だけで表示します (keiba train で学習できます)")

    workers: dict[str, Any] = {}

    def worker_for(role: str):
        name = settings.workers.get(role) or ""
        if not name:
            return None
        if name not in workers:
            try:
                workers[name] = make_worker(name, settings)
            except Exception as exc:  # APIキー未設定・未インストールなど. 1回だけ知らせ、AIなしで続ける
                workers[name] = None
                progress(f"作業者 {name} を使えません ({exc})")
        return workers[name]

    out_root = settings.path("reports")
    pages: list[Path] = []
    day_dirs: dict[str, Path] = {}
    for race_id in [r for r in race_ids if r not in missing]:
        race = db.get_race(conn, race_id)
        pred = pred_all[pred_all["race_id"] == race_id]
        day_dir = out_root / race["date"].replace("-", "")

        def context(analysis=None, review=None, news=None):
            return render.race_context(
                race, feat, pred, by_horse, top2, betting=betting, model=model_info,
                analysis=analysis, review=review, news=news, index_href="index.html",
            )

        news = _news_dict(conn, race_id)
        ctx = context(news=news)
        stale = False  # 今回AI予想を頼んだのに作れなかった
        if roles and is_target(race, llm_filter):
            label = f"{race['venue']}{race['race_num']}R"
            # 役割ごとに失敗を分ける (直前情報が取れなくても予想は作る)
            if "news" in roles and (w := worker_for("news")):
                progress(f"{label}: {w.name} が直前情報を収集中…")
                res = _attempt(label, progress, llm_tasks.gather_news, w, ctx)
                if res is not None:
                    db.save_news(conn, race_id, res.worker, res.text, dedupe_sources(res.sources))
                    conn.commit()
                    news = _news_dict(conn, race_id)
            if "predict" in roles:
                res = None
                if w := worker_for("predict"):
                    progress(f"{label}: {w.name} が予想を作成中…")
                    res = _attempt(label, progress, llm_tasks.predict, w, ctx, budget=budget,
                                   news=news["content"] if news else None)
                if res is not None:
                    db.save_analysis(conn, race_id, "predict", res.worker, res.data)
                    conn.commit()
                stale = res is None
            analysis = db.latest_analysis(conn, race_id, "predict")
            if stale and analysis:
                # 古い予想を点検しても意味が薄いので、セカンドオピニオンは作らない
                progress(f"{label}: 予想を作れなかったので、前回のAI予想 ({analysis['created_at']}) を表示します")
            elif "review" in roles and analysis and (w := worker_for("review")):
                progress(f"{label}: {w.name} がセカンドオピニオンを作成中…")
                res = _attempt(label, progress, llm_tasks.review, w, ctx, analysis["content"])
                if res is not None:
                    db.save_analysis(conn, race_id, "review", res.worker, res.text, ref_id=analysis["id"])
                    conn.commit()
        analysis = db.latest_analysis(conn, race_id, "predict")
        # セカンドオピニオンは、表示する予想を点検したものだけ出す
        review = db.review_for(conn, race_id, analysis["id"]) if analysis else None
        if analysis and stale:
            analysis["stale"] = True
        if analysis or review or news:
            ctx = context(analysis=analysis, review=review, news=news)
        path = render.render_race(ctx, day_dir / f"{race_id}.html")
        pages.append(path)
        item = render.summarize_for_index(ctx, f"{race_id}.html")
        item["is_demo"] = race.get("source") == "demo"
        _save_summary(day_dir / f"{race_id}.summary.json", item)
        day_dirs[race["date"]] = day_dir

    # 一覧は今回作ったレースだけでなく、その日のレースページすべてから作り直す (1レースだけ更新しても減らない)
    for day, day_dir in day_dirs.items():
        items = _load_summaries(day_dir)
        pages.append(render.render_index(day, items, day_dir / "index.html",
                                         is_demo=any(it.get("is_demo") for it in items)))
    return pages


# ---------------------------------------------------------------- 開催日まとめて

def upcoming_race_ids(conn: sqlite3.Connection, day: str) -> list[str]:
    return [rid for rid in db.race_ids_on(conn, day)
            if conn.execute("SELECT 1 FROM entries WHERE race_id = ? LIMIT 1", (rid,)).fetchone()
            or db.has_results(conn, rid)]


def unsettled_race_ids(conn: sqlite3.Connection, before: str) -> list[str]:
    """出馬表だけあって成績が未取得の、指定日より前のレース (デモデータは除く)."""
    rows = conn.execute("""
        SELECT DISTINCT ra.race_id, ra.date FROM races ra JOIN entries e USING (race_id)
        WHERE ra.date < ? AND COALESCE(ra.source, '') != 'demo'
          AND ra.race_id NOT IN (SELECT race_id FROM results)
        ORDER BY ra.date, ra.race_id
    """, (before,)).fetchall()
    return [r[0] for r in rows]


# 1件ずつ取りに行く処理で、取得エラーがこの回数続いたら (通信できない・アクセス拒否) 収集をやめる
MAX_FETCH_FAILURES = 3


def _fetch_each(race_ids: list[str], fetch: Callable[[str], Any], what: str, progress: Progress) -> None:
    from keiba.scraper.http import AccessDenied, FetchError

    failures = 0
    for race_id in race_ids:
        try:
            fetch(race_id)
            failures = 0
        except AccessDenied:  # アクセス拒否のあとは続けても無駄なので止める
            raise
        except FetchError as exc:
            failures += 1
            progress(f"  {race_id}: {what}スキップ ({exc})")
            if failures >= MAX_FETCH_FAILURES:
                raise
        except Exception as exc:  # オッズ未発売・成績未確定などは続行
            progress(f"  {race_id}: {what}スキップ ({exc})")


def _collect_day(
    conn: sqlite3.Connection,
    settings: Settings,
    day: date,
    *,
    odds: bool,
    history: bool,
    pedigree: bool,
    progress: Progress,
) -> None:
    from keiba.scraper import collect
    from keiba.scraper.http import Fetcher

    fetcher = Fetcher.from_settings(settings)
    collect.collect_cards_for_date(conn, fetcher, day, progress=progress)
    # 前の開催日の出馬表だけ残っているレースは、成績を取り込んでから予想する (前走の情報が欠けないように)
    unsettled = unsettled_race_ids(conn, min(day, date.today()).isoformat())
    if unsettled:
        progress(f"成績が未取得の過去レース {len(unsettled)}件の成績を取得します")
        _fetch_each(unsettled, lambda rid: collect.collect_race(conn, fetcher, rid), "成績取得", progress)
    ids = db.race_ids_on(conn, day.isoformat())
    if odds:
        def fetch_odds(rid: str) -> None:
            if not collect.collect_odds(conn, fetcher, rid).get("market", True):
                progress(f"  {rid}: オッズ未発売 (予想オッズは使いません)")

        _fetch_each(ids, fetch_odds, "オッズ取得", progress)
    horse_ids = [r[0] for r in conn.execute(
        f"SELECT DISTINCT horse_id FROM entries WHERE race_id IN ({','.join('?' * len(ids))})", ids
    ).fetchall()] if ids else []
    if history and horse_ids:
        progress(f"出走馬 {len(horse_ids)}頭の過去レースを補完します (時間がかかります)")
        collect.collect_horse_histories(conn, fetcher, horse_ids, progress=progress)
    if pedigree and horse_ids:
        known = {r[0] for r in conn.execute("SELECT horse_id FROM horses").fetchall()}
        todo = [h for h in horse_ids if h not in known]
        if todo:
            progress(f"血統 {len(todo)}頭を取得します")
            collect.collect_pedigrees(conn, fetcher, todo, progress=progress)


def run_day(
    conn: sqlite3.Connection,
    settings: Settings,
    day: date,
    *,
    offline: bool = False,
    odds: bool = True,
    history: bool = False,
    pedigree: bool = True,
    roles: set[str] = frozenset(),
    llm_filter: str = "main",
    venue: str | None = None,
    model_path: str | None = None,
    progress: Progress = print,
) -> list[Path]:
    """開催日1日分: 出馬表・オッズ収集 → 予想 → レポート."""
    day_str = day.isoformat()
    bundle = load_model(settings, model_path, progress=progress)  # --model の指定ミスは収集の前に知らせる
    if not offline:
        from keiba.scraper.http import FetchError

        try:
            _collect_day(conn, settings, day, odds=odds, history=history, pedigree=pedigree, progress=progress)
        except FetchError as exc:  # アクセス拒否・通信断. --offline と同じく DB にある分だけで作る
            progress(f"注意: 取得に失敗したので収集をやめ、DBにあるデータだけでレポートを作ります。"
                     f"出馬表・オッズは前回取得した時点のものです ({exc})")

    race_ids = upcoming_race_ids(conn, day_str)
    if venue:
        race_ids = [rid for rid in race_ids if (db.get_race(conn, rid) or {}).get("venue") == venue]
    if not race_ids:
        raise ValueError(f"{day_str} のレースがDBにありません")
    return build_reports(conn, settings, race_ids, bundle=bundle, roles=roles, llm_filter=llm_filter, progress=progress)
