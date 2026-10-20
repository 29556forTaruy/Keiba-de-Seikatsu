"""取得 → パース → DB保存 をつなぐ収集処理."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable, Iterable
from datetime import date, timedelta

from keiba import db
from keiba.scraper import jra, parsers, urls
from keiba.scraper.http import AccessDenied, Fetcher, FetchError
from keiba.scraper.pages import jst_midnight as _jst_midnight
from keiba.scraper.pages import parse_page as _parse

log = logging.getLogger(__name__)
Progress = Callable[[str], None]


def _months(start: date, end: date) -> Iterable[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _from_jra(fetcher: Fetcher, what: str, reason: Exception | str, fetch: Callable, *args, **kwargs):
    """netkeiba で取れなかったもの (理由は reason) を JRA 公式サイトから取る.

    JRA にもアクセスを拒否されたら AccessDenied をそのまま投げる (収集を止める)。
    それ以外で JRA からも取れなければ、両方の理由をつけて同じ種類の例外を投げる。
    """
    try:
        result = fetch(fetcher, *args, **kwargs)
    except AccessDenied:
        raise
    except (FetchError, parsers.ParseError) as exc:
        raise type(exc)(f"{exc} (netkeiba: {reason})") from exc
    if result:
        log.warning("%s: netkeiba で取れないため JRA から取りました (%s)", what, reason)
    return result


def kaisai_dates(fetcher: Fetcher, start: date, end: date, *, refresh: bool = False) -> list[date]:
    """期間内の開催日 (カレンダーページから). 読めなければ JRA の開催一覧、それもだめなら土日を総当たり."""
    out: list[date] = []
    today = date.today()
    for y, m in _months(start, end):
        url = urls.calendar(y, m)
        next_month = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        reason: Exception | None = None
        try:
            # 月が終わる前に取ったカレンダーは、代替開催の追加などに備えて月末後に一度取り直す
            html = fetcher.get(url, live=(y, m) >= (today.year, today.month), refresh=refresh,
                               stale_before=_jst_midnight(next_month + timedelta(days=7)))
        except FetchError as exc:
            reason, found = exc, []
        else:
            found = [date(int(s[:4]), int(s[4:6]), int(s[6:])) for s in parsers.extract_kaisai_dates(html)]
        if not found and reason is None:
            # 読めないページは _failed/ に移して次回また読む
            kept = fetcher.keep_failed(url)
            reason = parsers.ParseError(f"{y}年{m}月のカレンダーが読めません (取得したページ: {kept})")
        if not found and fetcher.jra_fallback:
            try:
                found = _from_jra(fetcher, f"{y}年{m}月の開催日", reason, jra.kaisai_dates, y, m, refresh=refresh)
            except AccessDenied:
                raise
            except (FetchError, parsers.ParseError) as exc:
                log.warning("%s", exc)
        if not found:
            if not isinstance(reason, parsers.ParseError):
                raise reason  # 通信できない (netkeiba も JRA も) ときは、これまでどおり止める
            # カレンダーが読めなければ土日を総当たりする
            log.warning("%s。土日を総当たりします", reason)
            d = date(y, m, 1)
            while d.month == m:
                if d.weekday() >= 5:
                    found.append(d)
                d += timedelta(days=1)
        out += [d for d in found if start <= d <= end]
    return sorted(set(out))


def race_ids_for_date(fetcher: Fetcher, day: date, *, refresh: bool = False) -> list[str]:
    """開催日のレース一覧. race.netkeiba で取れなければ db.netkeiba、それもだめなら JRA を見る.

    開催日が終わる前に取った一覧 (未発表で空・一部だけなど) は確定後に一度取り直し、
    レースが1つも無い一覧はキャッシュに残さない。
    """
    key = day.strftime("%Y%m%d")
    live = day >= date.today() - timedelta(days=1)
    stale_before = _jst_midnight(day + timedelta(days=1))
    last_exc: FetchError | None = None
    for url in (urls.race_list(key), urls.db_race_list(key)):
        try:
            ids = parsers.extract_race_ids(fetcher.get(url, live=live, refresh=refresh, stale_before=stale_before))
        except FetchError as exc:  # アクセス拒否も含む. 同じサイトの2つ目は通信せずに拒否になる
            last_exc = exc
            continue
        if ids:
            return ids
        fetcher.invalidate(url)
    # JRA の結果一覧には終わった開催日しか無いので、先の日は見に行かない
    if fetcher.jra_fallback and day <= date.today():
        reason = last_exc or "レース一覧が空です"
        try:
            ids = _from_jra(fetcher, f"{day} のレース一覧", reason, jra.race_ids_for_date, day, refresh=refresh)
        except AccessDenied:
            raise
        except (FetchError, parsers.ParseError) as exc:
            if last_exc is None:  # netkeiba では空だっただけなら、これまでどおり空で返す
                log.warning("%s", exc)
                return []
            raise FetchError(str(exc)) from exc
        if ids:
            return ids
    if last_exc is not None:
        raise last_exc
    return []


def _race_from_jra(
    conn: sqlite3.Connection, fetcher: Fetcher, race_id: str, reason: Exception, race_date: date | None, refresh: bool
) -> dict:
    """netkeiba で取れなかった確定成績を JRA から取る. 予備を使わない場合は reason を投げ直す."""
    if not fetcher.jra_fallback:
        raise reason
    if race_date is None:
        known = db.get_race(conn, race_id)
        race_date = date.fromisoformat(known["date"]) if known and known.get("date") else None
    if isinstance(reason, parsers.RaceNotFound) and (race_date is None or race_date >= date.today()):
        # race_id の誤りか、結果がまだのレース。日付の分からないレースのために JRA の月一覧を総なめにしない
        raise reason
    return _from_jra(fetcher, race_id, reason, jra.fetch_result, race_id, race_date=race_date, refresh=refresh)


def collect_race(
    conn: sqlite3.Connection, fetcher: Fetcher, race_id: str, *, refresh: bool = False, race_date: date | None = None
) -> bool:
    """確定成績1レース分を保存する。保存できたら True.

    netkeiba で取れなければ (アクセス拒否・ページの変更など)、JRA 公式サイトから取る (設定 sources.jra_fallback)。
    race_date が分かっていれば JRA で探すページが減る。
    """
    if not refresh and db.has_results(conn, race_id):
        return False
    try:
        parsed = _parse(fetcher, urls.race_result(race_id), parsers.parse_race_result, race_id, refresh=refresh)
    except (FetchError, parsers.ParseError) as exc:
        parsed = _race_from_jra(conn, fetcher, race_id, exc, race_date, refresh)
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
    for day in kaisai_dates(fetcher, start, end, refresh=refresh):
        try:
            ids = race_ids_for_date(fetcher, day, refresh=refresh)
        except AccessDenied:
            raise
        except FetchError as exc:
            progress(f"{day}: レース一覧の取得に失敗 ({exc})")
            continue
        new = 0
        for race_id in ids:
            try:
                new += collect_race(conn, fetcher, race_id, refresh=refresh, race_date=day)
            except AccessDenied:
                raise
            except (FetchError, parsers.ParseError) as exc:
                progress(f"  {race_id}: スキップ ({exc})")
        saved += new
        progress(f"{day}: {len(ids)}レース中 {new}件を新規保存 (通信 {fetcher.network_requests}回)")
    return saved


def collect_card(conn: sqlite3.Connection, fetcher: Fetcher, race_id: str, *, race_date: str | None = None) -> dict:
    """出馬表を保存する.

    確定成績があるレースは結果ページの値 (馬場・コース・条件など) を正とし、
    出馬表では空欄だけを埋める。
    """
    parsed = _parse(fetcher, urls.shutuba(race_id), parsers.parse_shutuba, race_id, date=race_date, live=True)
    existing = db.get_race(conn, race_id)
    race = parsed["race"]
    if existing and db.has_results(conn, race_id):
        race = {**race, **{k: v for k, v in existing.items() if v is not None}}
        parsed["race"] = race
    elif existing and existing.get("source") != "netkeiba":
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
        except AccessDenied:
            raise
        except (FetchError, parsers.ParseError) as exc:
            progress(f"  {race_id}: 出馬表スキップ ({exc})")
    progress(f"{day}: 出馬表 {len(done)}/{len(ids)}レース")
    return done


def collect_odds(conn: sqlite3.Connection, fetcher: Fetcher, race_id: str) -> dict:
    """単勝・複勝オッズを保存する. 発売前の予想オッズ (market=False) は保存しない."""
    parsed = _parse(fetcher, urls.odds_api(race_id), parsers.parse_odds_json, race_id, live=True)
    if not parsed["market"]:
        # 以前に保存された予想オッズも、実オッズとして使われないよう消しておく
        db.save_odds(conn, race_id, [])
        conn.commit()
        return parsed
    rows = [r for r in parsed["odds"] if r["win_odds"] is not None]
    parsed["scratched"] = []
    if rows:
        db.save_odds(conn, race_id, rows)
        # 発売中に単勝オッズの無い馬は取消・除外。出馬表 (キャッシュ) に未反映でも取消扱いにする
        priced = {r["number"] for r in rows}
        gone = [n for (n,) in conn.execute(
            "SELECT number FROM entries WHERE race_id = ? AND scratched = 0 AND number IS NOT NULL", (race_id,)
        ).fetchall() if n not in priced]
        if gone:
            conn.executemany("UPDATE entries SET scratched = 1 WHERE race_id = ? AND number = ?",
                             [(race_id, n) for n in gone])
            parsed["scratched"] = gone
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
        except AccessDenied:
            conn.commit()  # 中断しても、それまでに取れた分は残す
            raise
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
        except AccessDenied:
            raise
        except FetchError as exc:
            progress(f"  {horse_id}: 馬ページ取得失敗 ({exc})")
            continue
        for race_id in race_ids[:max_races_per_horse]:
            try:
                saved += collect_race(conn, fetcher, race_id)
            except AccessDenied:
                raise
            except (FetchError, parsers.ParseError) as exc:
                progress(f"  {race_id}: スキップ ({exc})")
    return saved
