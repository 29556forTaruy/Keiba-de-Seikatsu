"""収集処理 (通信はモック)."""

from pathlib import Path

import pytest

from keiba import db
from keiba.scraper import collect, parsers, urls
from keiba.scraper.http import Fetcher

FIX = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, text: str):
        self.status_code = 200
        self.content = text.encode("utf-8")
        self.encoding = "utf-8"
        self.headers = {"content-type": "text/html; charset=utf-8"}


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
