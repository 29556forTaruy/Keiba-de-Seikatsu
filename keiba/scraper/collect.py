"""取得 → パース → DB保存 をつなぐ収集処理."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable
from datetime import date, timedelta

from keiba import db
from keiba.scraper import parsers, urls
from keiba.scraper.http import Fetcher, FetchError

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


def _parse(fetcher: Fetcher, url: str, parse, *args, live: bool = False, refresh: bool = False, **kwargs):
    """取得してパース。パースに失敗したらキャッシュを捨てて、次回は取り直す."""
    html = fetcher.get(url, live=live, refresh=refresh)
    try:
        return parse(html, *args, **kwargs)
    except parsers.ParseError:
        fetcher.invalidate(url)
        raise


def _months(start: date, end: date) -> Iterable[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def kaisai_dates(fetcher: Fetcher, start: date, end: date) -> list[date]:
    """期間内の開催日 (カレンダーページから)."""
    out: list[date] = []
    for y, m in _months(start, end):
        html = fetcher.get(urls.calendar(y, m), live=(y, m) >= (date.today().year, date.today().month))
        found = [date(int(s[:4]), int(s[4:6]), int(s[6:])) for s in parsers.extract_kaisai_dates(html)]
        if not found:
            # カレンダーが読めなければ土日を総当たりする
            log.warning("%d年%d月のカレンダーが読めないため土日を総当たりします", y, m)
            d = date(y, m, 1)
            while d.month == m:
                if d.weekday() >= 5:
                    found.append(d)
                d += timedelta(days=1)
        out += [d for d in found if start <= d <= end]
    return sorted(set(out))


def race_ids_for_date(fetcher: Fetcher, day: date) -> list[str]:
    key = day.strftime("%Y%m%d")
    live = day >= date.today() - timedelta(days=1)
    ids = parsers.extract_race_ids(fetcher.get(urls.race_list(key), live=live))
    if not ids:
        ids = parsers.extract_race_ids(fetcher.get(urls.db_race_list(key), live=live))
    return ids


def collect_race(conn: sqlite3.Connection, fetcher: Fetcher, race_id: str, *, refresh: bool = False) -> bool:
    """確定成績1レース分を保存する。保存できたら True."""
    if not refresh and db.has_results(conn, race_id):
        return False
    parsed = _parse(fetcher, urls.race_result(race_id), parsers.parse_race_result, race_id, refresh=refresh)
    db.save_race(conn, parsed["race"])
    db.save_results(conn, race_id, parsed["results"])
    db.save_payouts(conn, race_id, parsed["payouts"])
    conn.commit()
    return True


def collect_results(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    start: date,
    end: date,
    *,
    refresh: bool = False,
    progress: Progress = print,
) -> int:
    """期間内の全レースの確定成績を集める (取得済みはスキップ)."""
    saved = 0
    for day in kaisai_dates(fetcher, start, end):
        try:
            ids = race_ids_for_date(fetcher, day)
        except FetchError as exc:
            progress(f"{day}: レース一覧の取得に失敗 ({exc})")
            continue
        new = 0
        for race_id in ids:
            try:
                new += collect_race(conn, fetcher, race_id, refresh=refresh)
            except (FetchError, parsers.ParseError) as exc:
                progress(f"  {race_id}: スキップ ({exc})")
        saved += new
        progress(f"{day}: {len(ids)}レース中 {new}件を新規保存 (通信 {fetcher.network_requests}回)")
    return saved


def collect_card(conn: sqlite3.Connection, fetcher: Fetcher, race_id: str, *, race_date: str | None = None) -> dict:
    """出馬表を保存する."""
    parsed = _parse(fetcher, urls.shutuba(race_id), parsers.parse_shutuba, race_id, date=race_date, live=True)
    existing = db.get_race(conn, race_id)
    race = parsed["race"]
    if existing and existing.get("source") != "netkeiba":
        race["source"] = existing.get("source")
    db.save_race(conn, race)
    db.save_entries(conn, race_id, parsed["entries"])
    conn.commit()
    return parsed


def collect_cards_for_date(
    conn: sqlite3.Connection, fetcher: Fetcher, day: date, *, progress: Progress = print
) -> list[str]:
    ids = race_ids_for_date(fetcher, day)
    done = []
    for race_id in ids:
        try:
            collect_card(conn, fetcher, race_id, race_date=day.isoformat())
            done.append(race_id)
        except (FetchError, parsers.ParseError) as exc:
            progress(f"  {race_id}: 出馬表スキップ ({exc})")
    progress(f"{day}: 出馬表 {len(done)}/{len(ids)}レース")
    return done


def collect_odds(conn: sqlite3.Connection, fetcher: Fetcher, race_id: str) -> dict:
    parsed = _parse(fetcher, urls.odds_api(race_id), parsers.parse_odds_json, race_id, live=True)
    rows = [r for r in parsed["odds"] if r["win_odds"] is not None]
    if rows:
        db.save_odds(conn, race_id, rows)
        conn.commit()
    return parsed


def collect_pedigrees(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    horse_ids: list[str] | None = None,
    *,
    limit: int | None = None,
    progress: Progress = print,
) -> int:
    ids = horse_ids if horse_ids is not None else db.horses_missing_pedigree(conn, limit)
    saved = 0
    for i, horse_id in enumerate(ids, 1):
        try:
            db.save_horse(conn, _parse(fetcher, urls.pedigree(horse_id), parsers.parse_pedigree, horse_id))
            saved += 1
        except (FetchError, parsers.ParseError) as exc:
            progress(f"  {horse_id}: 血統スキップ ({exc})")
        if i % 50 == 0:
            conn.commit()
            progress(f"血統 {i}/{len(ids)}")
    conn.commit()
    return saved


def collect_horse_histories(
    conn: sqlite3.Connection,
    fetcher: Fetcher,
    horse_ids: list[str],
    *,
    max_races_per_horse: int = 10,
    progress: Progress = print,
) -> int:
    """出走馬の過去レースのうちDBに無いものを集める (馬柱を埋めるため)."""
    saved = 0
    for horse_id in horse_ids:
        try:
            race_ids = parsers.extract_race_ids(fetcher.get(urls.horse(horse_id), live=True))
        except FetchError as exc:
            progress(f"  {horse_id}: 馬ページ取得失敗 ({exc})")
            continue
        for race_id in race_ids[:max_races_per_horse]:
            try:
                saved += collect_race(conn, fetcher, race_id)
            except (FetchError, parsers.ParseError) as exc:
                progress(f"  {race_id}: スキップ ({exc})")
    return saved
