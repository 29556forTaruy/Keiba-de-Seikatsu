"""JRA 公式サイトからの取得 (netkeiba で取れないときの予備).

たどり方: 過去レース結果検索 (月ごとの検査値) → 月の開催一覧 → 開催日のレース一覧 → レース結果 → 確定オッズ。
ページの指定 (CNAME) はどれもひとつ前のページのリンクから読むので、検査値を自分で計算しない。
すべて Fetcher を通すので、間隔・キャッシュ・アクセス拒否での停止は netkeiba と同じに働く。
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

from keiba.scraper import jra_parsers as J
from keiba.scraper.http import AccessDenied, Fetcher, FetchError
from keiba.scraper.pages import jst_midnight, parse_page
from keiba.scraper.parsers import ParseError, RaceNotFound

log = logging.getLogger(__name__)

BASE = "https://www.jra.go.jp/JRADB/"
RESULT_URL = BASE + "accessS.html"
ODDS_URL = BASE + "accessO.html"


def _form(cname: str) -> dict[str, str]:
    return {"cname": cname}


def month_index(fetcher: Fetcher, *, refresh: bool = False) -> dict[str, Any]:
    """月ごとの検査値の表. 月が替わると中身が変わるので、出馬表と同じく短い間だけキャッシュを使う."""
    return parse_page(fetcher, RESULT_URL, J.parse_month_index, data=_form(J.SEARCH_CNAME), live=True, refresh=refresh)


def days_in_month(fetcher: Fetcher, year: int, month: int, *, refresh: bool = False) -> list[dict[str, Any]]:
    """その月の開催日 (場・回・日目ごと). 検索ページに無い月 (古すぎる・先の月) は空."""
    cname = J.month_cname(month_index(fetcher, refresh=refresh), year, month)
    if cname is None:
        return []
    today = date.today()
    next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
    data = _form(cname)
    # 月が終わる前に取った一覧は、開催が増えるので月末後に一度取り直す (netkeiba のカレンダーと同じ)
    days = parse_page(fetcher, RESULT_URL, J.extract_days, data=data, refresh=refresh,
                      live=(year, month) >= (today.year, today.month),
                      stale_before=jst_midnight(next_month + timedelta(days=7)))
    prefix = f"{year:04d}-{month:02d}"
    days = [d for d in days if d["date"].startswith(prefix)]
    if not days:
        fetcher.invalidate(RESULT_URL, data)
    return days


def kaisai_dates(fetcher: Fetcher, year: int, month: int, *, refresh: bool = False) -> list[date]:
    return sorted({date.fromisoformat(d["date"]) for d in days_in_month(fetcher, year, month, refresh=refresh)})


def races_on_day(fetcher: Fetcher, day: dict[str, Any], *, refresh: bool = False) -> list[dict[str, Any]]:
    """開催日 (場・回・日目) のレース一覧 → [{'race_id', 'cname', 'date'}]."""
    d = date.fromisoformat(day["date"])
    data = _form(day["cname"])
    races = parse_page(fetcher, RESULT_URL, J.extract_races, data=data, refresh=refresh,
                       live=d >= date.today() - timedelta(days=1), stale_before=jst_midnight(d + timedelta(days=1)))
    prefix = day["year"] + day["venue_code"] + day["kai"] + day["nichi"]
    races = [r for r in races if r["race_id"].startswith(prefix)]
    if not races:
        fetcher.invalidate(RESULT_URL, data)
    return races


def race_ids_for_date(fetcher: Fetcher, day: date, *, refresh: bool = False) -> list[str]:
    """その日の全場のレース. 一部の場が取れなくても残りは返す (全部だめなら最後の失敗を投げる)."""
    ids: list[str] = []
    last_exc: Exception | None = None
    for entry in days_in_month(fetcher, day.year, day.month, refresh=refresh):
        if entry["date"] != day.isoformat():
            continue
        try:
            ids += [r["race_id"] for r in races_on_day(fetcher, entry, refresh=refresh)]
        except AccessDenied:
            raise
        except (FetchError, ParseError) as exc:
            last_exc = exc
            log.warning("JRAの %s %s回%s日目のレース一覧が取れません (%s)", day, entry["kai"], entry["nichi"], exc)
    if not ids and last_exc is not None:
        raise last_exc
    return ids


def find_race(fetcher: Fetcher, race_id: str, race_date: date | None = None) -> dict[str, Any]:
    """race_id のレース結果ページの CNAME を探す.

    race_id に日付は入っていないので、日付が分からなければその年の月を順に見る (月の一覧はキャッシュされる)。
    """
    year, venue, kai, nichi = race_id[:4], race_id[4:6], race_id[6:8], race_id[8:10]
    today = date.today()
    if race_date is not None:
        months = [(race_date.year, race_date.month)]
    else:
        months = [(int(year), m) for m in range(1, 13) if (int(year), m) <= (today.year, today.month)]
    for y, m in months:
        for entry in days_in_month(fetcher, y, m):
            if (entry["year"], entry["venue_code"], entry["kai"], entry["nichi"]) != (year, venue, kai, nichi):
                continue
            for race in races_on_day(fetcher, entry):
                if race["race_id"] == race_id:
                    return race
            raise RaceNotFound(f"JRAの {entry['date']} のレース一覧に {race_id} がありません")
    raise RaceNotFound(f"JRAの過去レース結果に {race_id} が見つかりません")


def fetch_result(
    fetcher: Fetcher, race_id: str, *, race_date: date | None = None, refresh: bool = False
) -> dict[str, Any]:
    """確定成績1レース分 (parsers.parse_race_result と同じ形). 単勝オッズは確定オッズページから埋める."""
    race = find_race(fetcher, race_id, race_date)
    # 結果ページはリンクと同じ GET で取れる (月・開催日の一覧は POST でないと取れない)
    parsed = parse_page(fetcher, f"{RESULT_URL}?CNAME={race['cname']}", J.parse_result, race_id, refresh=refresh)
    odds_cname = parsed.pop("odds_cname")
    if odds_cname is None:
        log.warning("%s: JRAの結果ページに確定オッズへのリンクが無いため、単勝オッズなしで保存します", race_id)
        return parsed
    try:
        odds = parse_page(fetcher, ODDS_URL, J.parse_win_odds, data=_form(odds_cname), refresh=refresh)
    except AccessDenied:
        raise
    except (FetchError, ParseError) as exc:
        log.warning("%s: JRAの確定オッズが取れないため、単勝オッズなしで保存します (%s)", race_id, exc)
        return parsed
    for row in parsed["results"]:
        row["odds"] = odds.get(row["number"])
    return parsed
