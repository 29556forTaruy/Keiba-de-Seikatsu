"""収集 → 特徴量 → 予想 → (LLM作業者) → HTML をつなぐ処理."""

from __future__ import annotations

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

def latest_model_path(settings: Settings) -> Path | None:
    models = settings.path("models")
    pointer = models / "LATEST"
    if pointer.exists():
        path = models / pointer.read_text().strip()
        if path.exists():
            return path
    found = sorted(models.glob("*.joblib"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def save_model(settings: Settings, bundle: ModelBundle) -> Path:
    models = settings.path("models")
    path = bundle.save(models / f"{bundle.version}.joblib")
    (models / "LATEST").write_text(path.name)
    return path


def load_model(settings: Settings, path: str | Path | None = None) -> ModelBundle | None:
    path = Path(path) if path else latest_model_path(settings)
    return ModelBundle.load(path) if path and path.exists() else None


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
            workers[name] = make_worker(name, settings)
        return workers[name]

    out_root = settings.path("reports")
    pages: list[Path] = []
    index_items: dict[str, list[dict[str, Any]]] = {}
    demo_dates: set[str] = set()
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
        if roles and is_target(race, llm_filter):
            label = f"{race['venue']}{race['race_num']}R"
            try:
                if "news" in roles and (w := worker_for("news")):
                    progress(f"{label}: {w.name} が直前情報を収集中…")
                    res = llm_tasks.gather_news(w, ctx)
                    db.save_news(conn, race_id, res.worker, res.text, dedupe_sources(res.sources))
                    conn.commit()
                    news = _news_dict(conn, race_id)
                if "predict" in roles and (w := worker_for("predict")):
                    progress(f"{label}: {w.name} が予想を作成中…")
                    res = llm_tasks.predict(w, ctx, budget=budget, news=news["content"] if news else None)
                    db.save_analysis(conn, race_id, "predict", res.worker, res.data)
                    conn.commit()
                analysis = db.latest_analysis(conn, race_id, "predict")
                if "review" in roles and analysis and (w := worker_for("review")):
                    progress(f"{label}: {w.name} がセカンドオピニオンを作成中…")
                    res = llm_tasks.review(w, ctx, analysis["content"])
                    db.save_analysis(conn, race_id, "review", res.worker, res.text)
                    conn.commit()
            except WorkerError as exc:
                progress(f"{label}: 作業者エラー ({exc})")
        analysis = db.latest_analysis(conn, race_id, "predict")
        review = db.latest_analysis(conn, race_id, "review")
        if analysis or review or news:
            ctx = context(analysis=analysis, review=review, news=news)
        path = render.render_race(ctx, day_dir / f"{race_id}.html")
        pages.append(path)
        index_items.setdefault(race["date"], []).append(render.summarize_for_index(ctx, f"{race_id}.html"))
        if race.get("source") == "demo":
            demo_dates.add(race["date"])

    for day, items in index_items.items():
        pages.append(render.render_index(day, items, out_root / day.replace("-", "") / "index.html",
                                         is_demo=day in demo_dates))
    return pages


# ---------------------------------------------------------------- 開催日まとめて

def upcoming_race_ids(conn: sqlite3.Connection, day: str) -> list[str]:
    return [rid for rid in db.race_ids_on(conn, day)
            if conn.execute("SELECT 1 FROM entries WHERE race_id = ? LIMIT 1", (rid,)).fetchone()
            or db.has_results(conn, rid)]


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
    if not offline:
        from keiba.scraper import collect
        from keiba.scraper.http import Fetcher, FetchError

        fetcher = Fetcher.from_settings(settings)
        collect.collect_cards_for_date(conn, fetcher, day, progress=progress)
        ids = db.race_ids_on(conn, day_str)
        if odds:
            for race_id in ids:
                try:
                    collect.collect_odds(conn, fetcher, race_id)
                except Exception as exc:  # オッズ未発売などは続行
                    progress(f"  {race_id}: オッズ取得スキップ ({exc})")
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
                try:
                    collect.collect_pedigrees(conn, fetcher, todo, progress=progress)
                except FetchError as exc:
                    progress(f"血統取得を中断 ({exc})")

    race_ids = upcoming_race_ids(conn, day_str)
    if venue:
        race_ids = [rid for rid in race_ids if (db.get_race(conn, rid) or {}).get("venue") == venue]
    if not race_ids:
        raise ValueError(f"{day_str} のレースがDBにありません")
    bundle = load_model(settings, model_path)
    return build_reports(conn, settings, race_ids, bundle=bundle, roles=roles, llm_filter=llm_filter, progress=progress)
