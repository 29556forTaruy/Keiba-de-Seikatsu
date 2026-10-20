"""取得とパースのつなぎ (netkeiba・JRA 共通)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Callable

from keiba.scraper import parsers
from keiba.scraper.http import Fetcher

JST = timezone(timedelta(hours=9))


def jst_midnight(day: date) -> float:
    """その日の 0:00 (日本時間) の UNIX 時刻."""
    return datetime.combine(day, time(0), tzinfo=JST).timestamp()


def parse_page(
    fetcher: Fetcher,
    url: str,
    parse: Callable[..., Any],
    *args: Any,
    data: dict[str, str] | None = None,
    live: bool = False,
    refresh: bool = False,
    stale_before: float | None = None,
    **kwargs: Any,
) -> Any:
    """取得してパース。パースに失敗したら次回は取り直す.

    中身の無いページ (RaceNotFound) は捨て、それ以外はパーサーを直す材料として data/raw/_failed/ に残す。
    """
    html = fetcher.get(url, data=data, live=live, refresh=refresh, stale_before=stale_before)
    try:
        return parse(html, *args, **kwargs)
    except parsers.RaceNotFound:
        fetcher.invalidate(url, data)
        raise
    except parsers.ParseError as exc:
        kept = fetcher.keep_failed(url, data)
        if kept is None:
            raise
        raise type(exc)(f"{exc} (取得したページ: {kept})") from exc
