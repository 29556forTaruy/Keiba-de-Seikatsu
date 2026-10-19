"""収集処理 (通信はモック)."""

import gzip
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from keiba import db
from keiba.scraper import collect, parsers, urls
from keiba.scraper.http import AccessDenied, Fetcher, FetchError

FIX = Path(__file__).parent / "fixtures"
JST = timezone(timedelta(hours=9))


class FakeResponse:
    """pages の値が int なら、その HTTP ステータスを返す."""

    def __init__(self, page: str | int):
        self.status_code = page if isinstance(page, int) else 200
        self.content = b"" if isinstance(page, int) else page.encode("utf-8")
        self.encoding = "utf-8"
        self.headers = {"content-type": "text/html; charset=utf-8"}


def fix(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    f = Fetcher(tmp_path / "raw")
    monkeypatch.setattr(f, "_wait", lambda: None)
    pages: dict[str, str] = {}
    monkeypatch.setattr(f.session, "get", lambda url, timeout: FakeResponse(pages[url]))
    f.pages = pages
    return f


def test_unparsable_result_page_is_not_cached(fetcher, tmp_path):
    """開催前に取った結果ページはキャッシュに残さず、レース後に取り直す."""
    conn = db.connect(tmp_path / "k.sqlite")
    race_id = "202405040911"
    url = urls.race_result(race_id)
    fetcher.pages[url] = "<html><body>まだ結果がありません</body></html>"
    with pytest.raises(parsers.ParseError):
        collect.collect_race(conn, fetcher, race_id)
    assert not fetcher.cache_path(url).exists()

    fetcher.pages[url] = (FIX / "race_result.html").read_text(encoding="utf-8")
    assert collect.collect_race(conn, fetcher, race_id) is True
    assert db.has_results(conn, race_id)
    assert fetcher.network_requests == 2
    # 2回目以降はキャッシュも DB もあるので通信しない
    assert collect.collect_race(conn, fetcher, race_id) is False
    assert fetcher.network_requests == 2


def test_card_only_db_builds_report(fetcher, tmp_path):
    """成績が1件もないDB (出馬表だけ) でもレポートが作れる."""
    from keiba import pipeline
    from keiba.config import load_settings

    conn = db.connect(tmp_path / "k.sqlite")
    race_id = "202608040711"
    fetcher.pages[urls.shutuba(race_id)] = (FIX / "shutuba.html").read_text(encoding="utf-8")
    fetcher.pages[urls.odds_api(race_id)] = (FIX / "odds.json").read_text(encoding="utf-8")
    collect.collect_card(conn, fetcher, race_id)
    collect.collect_odds(conn, fetcher, race_id)
    settings = load_settings(db=str(tmp_path / "k.sqlite"))
    settings.raw["paths"]["reports"] = str(tmp_path / "reports")
    pages = pipeline.build_reports(conn, settings, [race_id], bundle=None, progress=lambda *_: None)
    html = pages[0].read_text(encoding="utf-8")
    assert "テストホースアルファ" in html
    assert "テストホースガンマ" not in html  # 取消馬は出さない


def test_forecast_odds_are_not_saved(fetcher, tmp_path):
    """発売前の予想オッズ (status=yoso) は保存せず、以前の行も消す."""
    conn = db.connect(tmp_path / "k.sqlite")
    race_id = "202608040711"
    fetcher.pages[urls.shutuba(race_id)] = fix("shutuba.html")
    collect.collect_card(conn, fetcher, race_id)
    count = lambda: conn.execute("SELECT COUNT(*) FROM odds WHERE race_id = ?", (race_id,)).fetchone()[0]  # noqa: E731

    fetcher.pages[urls.odds_api(race_id)] = fix("odds.json")
    parsed = collect.collect_odds(conn, fetcher, race_id)
    assert parsed["market"] is True and count() == 3

    fetcher.live_ttl = 0  # 次の取得はキャッシュを使わない
    fetcher.pages[urls.odds_api(race_id)] = fix("odds.json").replace('"status":"middle"', '"status":"yoso"')
    parsed = collect.collect_odds(conn, fetcher, race_id)
    assert parsed["market"] is False and parsed["odds"]
    assert count() == 0


def test_access_denied_stops_all_network(fetcher, tmp_path):
    """403 を受けたら残りの馬を取りに行かず、以後の通信もしない (キャッシュは使える)."""
    conn = db.connect(tmp_path / "k.sqlite")
    a, b, c = "2020100001", "2020100002", "2020100003"
    fetcher.pages[urls.pedigree(a)] = fix("pedigree.html")
    fetcher.pages[urls.pedigree(b)] = 403
    fetcher.pages[urls.pedigree(c)] = fix("pedigree.html")
    with pytest.raises(AccessDenied):
        collect.collect_pedigrees(conn, fetcher, [a, b, c], progress=lambda *_: None)
    assert fetcher.network_requests == 2
    assert conn.execute("SELECT COUNT(*) FROM horses").fetchone()[0] == 1  # 取れた分は保存済み
    # キャッシュにあるページは読めるが、無いページは通信せずに止まる
    assert "テストサイアー" in fetcher.get(urls.pedigree(a))
    with pytest.raises(AccessDenied):
        fetcher.get(urls.pedigree(c))
    assert fetcher.network_requests == 2


def test_access_denied_aborts_collect_results(fetcher, tmp_path):
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.calendar(2024, 10)] = fix("calendar.html")
    fetcher.pages[urls.race_list("20241005")] = fix("race_list.html")
    fetcher.pages[urls.race_result("202405040901")] = 403
    fetcher.pages[urls.race_result("202405040911")] = fix("race_result.html")
    with pytest.raises(AccessDenied):
        collect.collect_results(conn, fetcher, date(2024, 10, 5), date(2024, 10, 6), progress=lambda *_: None)
    assert fetcher.network_requests == 3  # カレンダー・一覧・最初の1レースだけ
    assert not db.has_results(conn, "202405040911")


def test_other_fetch_errors_still_skip(fetcher, tmp_path):
    """404 などは従来どおりその1件だけ飛ばして続ける."""
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.calendar(2024, 10)] = fix("calendar.html")
    fetcher.pages[urls.race_list("20241005")] = fix("race_list.html")
    fetcher.pages[urls.race_list("20241006")] = fix("race_list.html").replace("20240504", "20240505")
    fetcher.pages[urls.race_result("202405040901")] = 404
    fetcher.pages[urls.race_result("202405040911")] = fix("race_result.html")
    for rid in ("202405050901", "202405050911"):
        fetcher.pages[urls.race_result(rid)] = 404
    n = collect.collect_results(conn, fetcher, date(2024, 10, 5), date(2024, 10, 6), progress=lambda *_: None)
    assert n == 1 and db.has_results(conn, "202405040911")


def _set_mtime(fetcher, url: str, when: datetime) -> None:
    ts = when.timestamp()
    os.utime(fetcher.cache_path(url), (ts, ts))


def test_race_list_fetched_before_race_day_is_refetched(fetcher):
    """開催前に取った一覧 (未発表や一部だけ) を、開催日が過ぎたあとも使い続けない."""
    day = date(2024, 10, 5)
    url = urls.race_list("20241005")
    partial = '<a href="../race/result.html?race_id=202405040911">11R</a>'
    fetcher.pages[url] = partial
    assert collect.race_ids_for_date(fetcher, day) == ["202405040911"]
    _set_mtime(fetcher, url, datetime(2024, 10, 3, 12, tzinfo=JST))  # 開催2日前に取った扱い

    fetcher.pages[url] = fix("race_list.html")
    assert collect.race_ids_for_date(fetcher, day) == ["202405040901", "202405040911"]
    assert fetcher.network_requests == 2
    # 確定後に取り直したものは以後キャッシュを使う
    assert collect.race_ids_for_date(fetcher, day) == ["202405040901", "202405040911"]
    assert fetcher.network_requests == 2


def test_empty_race_list_is_not_cached(fetcher):
    day = date(2024, 10, 5)
    fetcher.pages[urls.race_list("20241005")] = "<html><body></body></html>"
    fetcher.pages[urls.db_race_list("20241005")] = "<html><body></body></html>"
    assert collect.race_ids_for_date(fetcher, day) == []
    assert not fetcher.cache_path(urls.race_list("20241005")).exists()
    assert not fetcher.cache_path(urls.db_race_list("20241005")).exists()
    # 次に呼んだときは取り直すので、発表後の一覧が読める
    fetcher.pages[urls.race_list("20241005")] = fix("race_list.html")
    assert collect.race_ids_for_date(fetcher, day) == ["202405040901", "202405040911"]


def test_race_list_falls_back_to_db_on_fetch_error(fetcher):
    fetcher.pages[urls.race_list("20241005")] = 404
    fetcher.pages[urls.db_race_list("20241005")] = fix("race_list.html")
    assert collect.race_ids_for_date(fetcher, date(2024, 10, 5)) == ["202405040901", "202405040911"]


def test_refresh_reaches_calendar_and_race_list(fetcher):
    fetcher.pages[urls.calendar(2024, 10)] = fix("calendar.html")
    fetcher.pages[urls.race_list("20241005")] = fix("race_list.html")
    collect.kaisai_dates(fetcher, date(2024, 10, 1), date(2024, 10, 31))
    collect.race_ids_for_date(fetcher, date(2024, 10, 5))
    assert fetcher.network_requests == 2
    collect.kaisai_dates(fetcher, date(2024, 10, 1), date(2024, 10, 31))
    collect.race_ids_for_date(fetcher, date(2024, 10, 5))
    assert fetcher.network_requests == 2
    collect.kaisai_dates(fetcher, date(2024, 10, 1), date(2024, 10, 31), refresh=True)
    collect.race_ids_for_date(fetcher, date(2024, 10, 5), refresh=True)
    assert fetcher.network_requests == 4


def test_unreadable_calendar_is_not_cached(fetcher):
    url = urls.calendar(2024, 10)
    fetcher.pages[url] = "<html><body>メンテナンス中</body></html>"
    days = collect.kaisai_dates(fetcher, date(2024, 10, 1), date(2024, 10, 31))
    assert date(2024, 10, 5) in days  # 土日の総当たり
    assert not fetcher.cache_path(url).exists()


def test_broken_cache_is_refetched(fetcher):
    """書き込み途中で止まった gzip キャッシュで止まらず、取り直す."""
    url = urls.race_result("202405040911")
    fetcher.pages[url] = fix("race_result.html")
    fetcher.get(url)
    path = fetcher.cache_path(url)
    data = path.read_bytes()
    assert gzip.decompress(data).decode("utf-8") == fix("race_result.html")
    assert not list(path.parent.glob("*.tmp"))  # 一時ファイルは残らない

    path.write_bytes(data[: len(data) // 2])  # 途中で切れたファイル
    assert fetcher.get(url) == fix("race_result.html")
    assert fetcher.network_requests == 2
    corrupted = bytearray(path.read_bytes())
    corrupted[20:40] = b"\xff" * 20  # 中身が壊れたファイル
    path.write_bytes(bytes(corrupted))
    assert fetcher.get(url) == fix("race_result.html")
    assert fetcher.network_requests == 3
    assert gzip.decompress(path.read_bytes()).decode("utf-8") == fix("race_result.html")

    path.write_bytes(b"\x1f\x8b\x08")
    fetcher.offline = True
    with pytest.raises(FetchError):
        fetcher.get(url)


def test_card_does_not_overwrite_finished_race(fetcher, tmp_path):
    """確定成績のあるレースは、出馬表を取り直しても結果ページの値を残す."""
    conn = db.connect(tmp_path / "k.sqlite")
    race_id = "202405040911"
    fetcher.pages[urls.race_result(race_id)] = fix("race_result.html")
    collect.collect_race(conn, fetcher, race_id)
    before = db.get_race(conn, race_id)

    card = fix("shutuba.html").replace("馬場:重", "馬場:稍").replace("(右)", "(右 外)")
    fetcher.pages[urls.shutuba(race_id)] = card
    collect.collect_card(conn, fetcher, race_id, race_date="2024-10-27")
    after = db.get_race(conn, race_id)
    for col in ("going", "surface", "distance", "direction", "conditions", "field_size", "weather", "start_time"):
        assert after[col] == before[col], col
    assert after["going"] == "稍重"
    assert after["course_detail"] == "外"  # 結果ページで空だった列だけ出馬表で埋める
