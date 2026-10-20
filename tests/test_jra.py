"""JRA 公式サイトの予備 (パーサーと、netkeiba で取れないときの切り替え. 通信はモック)."""

from datetime import date
from pathlib import Path

import pytest

from keiba import db
from keiba.scraper import collect, jra, parsers, urls
from keiba.scraper import jra_parsers as J
from keiba.scraper.http import AccessDenied, Fetcher, request_key

FIX = Path(__file__).parent / "fixtures"
RACE_ID = "202405040811"  # 2024年 天皇賞(秋)
RACE_CNAME = "pw01sde1005202404081120241027/EF"
ODDS_CNAME = "pw151ou1005202404081120241027Z/31"


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- パーサー

def test_month_index_and_cname():
    index = J.parse_month_index(read("jra_search.html"))
    assert index["year_month"] == "202610"
    assert J.month_cname(index, 2024, 10) == "pw01skl10202410/92"
    # 当月以降は検索ページと同じく接頭辞が変わる
    assert J.month_cname(index, 2026, 10) == "pw01skl00202610/C3"
    assert J.month_cname(index, 2020, 1) is None


def test_days_and_races():
    days = J.extract_days(read("jra_month.html"))
    assert len(days) == 6
    tokyo = next(d for d in days if d["venue_code"] == "05" and d["date"] == "2024-10-27")
    assert tokyo == {"cname": "pw01srl10052024040820241027/46", "venue_code": "05", "year": "2024",
                     "kai": "04", "nichi": "08", "date": "2024-10-27"}
    races = J.extract_races(read("jra_day.html"))
    assert [r["race_id"] for r in races] == [f"2024050408{n:02d}" for n in range(1, 13)]
    assert races[10] == {"cname": RACE_CNAME, "race_id": RACE_ID, "date": "2024-10-27"}


def test_result():
    parsed = J.parse_result(read("jra_result.html"), RACE_ID)
    race = parsed["race"]
    assert race["source"] == "jra"
    assert (race["date"], race["start_time"], race["race_name"], race["grade"]) == (
        "2024-10-27", "15:40", "第170回天皇賞(秋)", "G1")
    assert (race["surface"], race["direction"], race["distance"], race["going"], race["weather"]) == (
        "芝", "左", 2000, "良", "曇")
    assert (race["venue"], race["race_num"], race["race_class"]) == ("東京", 11, "オープン")
    assert race["field_size"] == 3  # 取消は数えない

    rows = {r["number"]: r for r in parsed["results"]}
    win = rows[7]
    # ID は netkeiba と同じ体系 (馬は血統登録番号、騎手・調教師は5桁)
    assert (win["horse_id"], win["jockey_id"], win["trainer_id"]) == ("2019105283", "00666", "01061")
    assert (win["horse_name"], win["jockey_name"], win["trainer_name"]) == ("ドウデュース", "武豊", "友道康夫")
    assert (win["finish_pos"], win["frame"], win["sex"], win["age"], win["weight_carried"]) == (1, 4, "牡", 5, 58.0)
    assert (win["time_sec"], win["passing"], win["last3f"], win["popularity"]) == (117.3, "14-14-13", 32.5, 2)
    assert (win["horse_weight"], win["weight_diff"], win["margin"]) == (504, -4, None)
    assert win["prize"] == 22323.4  # 本賞金 + 付加賞 (netkeiba の賞金列と同じ)
    assert win["odds"] is None  # 結果表には無い. 確定オッズページで埋める
    assert (rows[4]["margin"], rows[4]["prize"]) == ("1.1/4", 8892.4)
    assert (rows[12]["horse_weight"], rows[12]["weight_diff"], rows[12]["prize"]) == (492, 0, None)
    gone = rows[16]
    assert (gone["finish_pos"], gone["finish_status"], gone["sex"], gone["jockey_id"]) == (None, "取消", "セ", "01999")

    pays = {(p["bet_type"], p["combination"]): p["payout"] for p in parsed["payouts"]}
    assert pays[("単勝", "7")] == 380
    assert pays[("複勝", "4")] == 1020
    assert pays[("馬単", "7-4")] == 13560
    assert pays[("三連複", "4-7-9")] == 102180
    assert pays[("三連単", "7-4-9")] == 397100
    assert parsed["odds_cname"] == ODDS_CNAME


def test_win_odds():
    assert J.parse_win_odds(read("jra_odds.html")) == {4: 53.8, 7: 3.8, 12: 2.3}


def test_error_page_is_parse_error():
    html = "<html><head><title>パラメータエラー　JRA</title></head><body></body></html>"
    with pytest.raises(parsers.ParseError, match="エラーページ"):
        J.extract_days(html)
    with pytest.raises(parsers.ParseError):
        J.parse_result(html, RACE_ID)


@pytest.mark.parametrize("text, expected", [
    ("１ 1/4", "1.1/4"), ("１／２", "1/2"), ("クビ", "クビ"), ("大差", "大"), ("", None),
])
def test_normalize_margin(text, expected):
    assert J.normalize_margin(text) == expected


@pytest.mark.parametrize("detail, expected", [
    ("（芝・左）", ("芝", "左", None)),
    ("（ダート・右）", ("ダ", "右", None)),
    ("（芝・左 外）", ("芝", "左", "外")),
    ("（芝・右・外）", ("芝", "右", "外")),
    ("（芝・直）", ("芝", "直線", None)),
])
def test_course_variants(detail, expected):
    html = read("jra_result.html").replace("（芝・左）", detail)
    race = J.parse_result(html, RACE_ID)["race"]
    assert (race["surface"], race["direction"], race["course_detail"]) == expected


def test_jump_race_and_dirt_going():
    """障害はコースの欄に書かれず (「（芝 外内）」)、条件の「障害」で分かる. ダートの馬場の欄は class が durt."""
    html = (read("jra_result.html")
            .replace("（芝・左）", "（芝 外内）")
            .replace('<div class="cell category">3歳以上</div>', '<div class="cell category">障害3歳以上</div>'))
    race = J.parse_result(html, RACE_ID)["race"]
    assert (race["surface"], race["direction"], race["course_detail"], race["going"]) == ("障", None, "外-内", "良")

    html = read("jra_result.html").replace("（芝・左）", "（ダート・左）").replace(
        '<li class="turf"><span class="inner"><span class="cap">芝</span><span class="txt">良</span>',
        '<li class="durt"><span class="inner"><span class="cap">ダート</span><span class="txt">稍重</span>')
    race = J.parse_result(html, RACE_ID)["race"]
    assert (race["surface"], race["going"]) == ("ダ", "稍重")


def test_netkeiba_conventions():
    """初出走の馬体重・発走時刻・見習騎手の印・オープンのグレードを netkeiba の表記にそろえる."""
    html = (read("jra_result.html")
            .replace("15時40分", "9時50分")
            .replace('<td class="h_weight">504<span>(-4)</span>', '<td class="h_weight">504<span>(初出走)</span>')
            .replace(">武 豊<", ">☆武 豊<")
            .replace('<span class="grade_icon lg"><img alt="GⅠ" src="/JRADB/img/grade/icon_grade_g1.png"/></span>', ""))
    parsed = J.parse_result(html, RACE_ID)
    win = next(r for r in parsed["results"] if r["number"] == 7)
    assert (win["horse_weight"], win["weight_diff"], win["jockey_name"]) == (504, 0, "武豊")
    assert (parsed["race"]["start_time"], parsed["race"]["grade"]) == ("09:50", "OP")


# ---------------------------------------------------------------- netkeiba で取れないときの切り替え

class FakeResponse:
    def __init__(self, page: str | int):
        self.status_code = page if isinstance(page, int) else 200
        self.content = b"" if isinstance(page, int) else page.encode("utf-8")
        self.encoding = "utf-8"
        self.headers = {"content-type": "text/html; charset=utf-8"}


@pytest.fixture
def fetcher(tmp_path, monkeypatch):
    """pages に無いURLは 404. POST は request_key(url, data) で引く."""
    f = Fetcher(tmp_path / "raw", jra_fallback=True)
    monkeypatch.setattr(f, "_wait", lambda: None)
    pages: dict[str, str | int] = {}
    monkeypatch.setattr(f.session, "get", lambda url, timeout: FakeResponse(pages.get(url, 404)))
    monkeypatch.setattr(f.session, "post",
                        lambda url, data, timeout: FakeResponse(pages.get(request_key(url, data), 404)))
    f.pages = pages
    return f


def _post(url: str, cname: str) -> str:
    return request_key(url, {"cname": cname})


def jra_pages(fetcher, *, months=(10,)) -> None:
    p = fetcher.pages
    p[_post(jra.RESULT_URL, J.SEARCH_CNAME)] = read("jra_search.html")
    index = J.parse_month_index(read("jra_search.html"))
    for m in months:
        p[_post(jra.RESULT_URL, J.month_cname(index, 2024, m))] = read("jra_month.html")
    p[_post(jra.RESULT_URL, "pw01srl10052024040820241027/46")] = read("jra_day.html")
    p[f"{jra.RESULT_URL}?CNAME={RACE_CNAME}"] = read("jra_result.html")
    p[_post(jra.ODDS_URL, ODDS_CNAME)] = read("jra_odds.html")


def test_race_falls_back_to_jra_when_netkeiba_denies(fetcher, tmp_path, caplog):
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.race_result(RACE_ID)] = 403
    jra_pages(fetcher)
    assert collect.collect_race(conn, fetcher, RACE_ID, race_date=date(2024, 10, 27)) is True
    race = db.get_race(conn, RACE_ID)
    assert (race["source"], race["date"], race["race_name"]) == ("jra", "2024-10-27", "第170回天皇賞(秋)")
    odds = dict(conn.execute("SELECT number, odds FROM results WHERE race_id = ?", (RACE_ID,)).fetchall())
    assert odds == {7: 3.8, 4: 53.8, 12: 2.3, 16: None}
    assert conn.execute("SELECT COUNT(*) FROM payouts WHERE race_id = ?", (RACE_ID,)).fetchone()[0] == 12
    assert "netkeiba で取れないため JRA から取りました" in caplog.text
    # netkeiba に拒否されても JRA は止めない (拒否はサイトごと)
    assert set(fetcher.blocked) == {"netkeiba.com"}


def test_race_without_date_searches_the_year(fetcher, tmp_path):
    """日付が分からなければ、その年の月を順に見る (検査値の無い月は飛ばす)."""
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.race_result(RACE_ID)] = 403
    jra_pages(fetcher, months=(9, 10))
    assert collect.collect_race(conn, fetcher, RACE_ID) is True
    assert db.get_race(conn, RACE_ID)["source"] == "jra"


def test_fallback_can_be_turned_off(fetcher, tmp_path):
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.jra_fallback = False
    fetcher.pages[urls.race_result(RACE_ID)] = 403
    jra_pages(fetcher)
    with pytest.raises(AccessDenied):
        collect.collect_race(conn, fetcher, RACE_ID, race_date=date(2024, 10, 27))
    assert fetcher.network_requests == 1


def test_missing_race_without_date_does_not_search_jra(fetcher, tmp_path):
    """race_id の誤りかもしれないレースのために、JRA の月一覧を総なめにしない."""
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.race_result(RACE_ID)] = read("race_result_empty.html")
    jra_pages(fetcher)
    with pytest.raises(parsers.RaceNotFound):
        collect.collect_race(conn, fetcher, RACE_ID)
    assert fetcher.network_requests == 1


def test_unparsable_netkeiba_page_falls_back(fetcher, tmp_path):
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.race_result(RACE_ID)] = "<html><body>見出しが変わったページ</body></html>"
    jra_pages(fetcher)
    assert collect.collect_race(conn, fetcher, RACE_ID, race_date=date(2024, 10, 27)) is True
    assert db.get_race(conn, RACE_ID)["source"] == "jra"


def test_both_sources_fail_reports_both(fetcher, tmp_path):
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.race_result(RACE_ID)] = 403
    jra_pages(fetcher)
    fetcher.pages[f"{jra.RESULT_URL}?CNAME={RACE_CNAME}"] = "<html><body>メンテナンス中</body></html>"
    with pytest.raises(parsers.ParseError) as info:
        collect.collect_race(conn, fetcher, RACE_ID, race_date=date(2024, 10, 27))
    assert "JRAの成績表が見つかりません" in str(info.value) and "netkeiba: HTTP 403" in str(info.value)


def test_jra_denied_too_stops_collecting(fetcher, tmp_path):
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.race_result(RACE_ID)] = 403
    fetcher.pages[_post(jra.RESULT_URL, J.SEARCH_CNAME)] = 403
    with pytest.raises(AccessDenied):
        collect.collect_race(conn, fetcher, RACE_ID, race_date=date(2024, 10, 27))
    assert set(fetcher.blocked) == {"netkeiba.com", "jra.go.jp"}


def test_race_list_falls_back_to_jra(fetcher, caplog):
    fetcher.pages[urls.race_list("20241027")] = 403
    jra_pages(fetcher)
    ids = collect.race_ids_for_date(fetcher, date(2024, 10, 27))
    # fixture は東京だけ. 京都・新潟の一覧が取れなくても東京の分は返す
    assert ids == [f"2024050408{n:02d}" for n in range(1, 13)]
    assert caplog.text.count("のレース一覧が取れません") == 2
    # db.netkeiba も同じサイトなので、拒否のあとは通信しない (netkeiba 1 + 検索・月 2 + 3場)
    assert fetcher.network_requests == 1 + 2 + 3


def test_calendar_falls_back_to_jra(fetcher):
    fetcher.pages[urls.calendar(2024, 10)] = 403
    jra_pages(fetcher)
    assert collect.kaisai_dates(fetcher, date(2024, 10, 26), date(2024, 10, 27)) == [
        date(2024, 10, 26), date(2024, 10, 27)]


def test_collect_results_continues_through_jra(fetcher, tmp_path):
    """netkeiba に拒否されても、開催日・レース一覧・成績を JRA でたどって集め続ける."""
    conn = db.connect(tmp_path / "k.sqlite")
    fetcher.pages[urls.calendar(2024, 10)] = 403
    jra_pages(fetcher)
    lines: list[str] = []
    saved = collect.collect_results(conn, fetcher, date(2024, 10, 27), date(2024, 10, 27), progress=lines.append)
    assert saved == 1  # fixture にある 11R だけ取れ、ほかはスキップ
    assert db.has_results(conn, RACE_ID)
    assert sum("スキップ" in line for line in lines) == 11


# ---------------------------------------------------------------- Fetcher

def test_post_is_cached_per_form_value(fetcher):
    url = jra.RESULT_URL
    fetcher.pages[_post(url, "a")] = "<html>A</html>"
    fetcher.pages[_post(url, "b")] = "<html>B</html>"
    assert fetcher.get(url, data={"cname": "a"}) == "<html>A</html>"
    assert fetcher.get(url, data={"cname": "b"}) == "<html>B</html>"
    assert fetcher.get(url, data={"cname": "a"}) == "<html>A</html>"
    assert fetcher.network_requests == 2
    assert fetcher.cache_path(url, {"cname": "a"}) != fetcher.cache_path(url, {"cname": "b"})


def test_shift_jis_page_is_read_as_cp932():
    from keiba.scraper.http import decode_html

    body = '<meta charset="Shift_JIS"><p>髙橋 ①</p>'.encode("cp932")
    assert "髙橋 ①" in decode_html(body)
