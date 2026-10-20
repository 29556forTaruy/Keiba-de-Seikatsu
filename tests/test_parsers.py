from pathlib import Path

import pytest

from keiba.scraper import parsers
from keiba.scraper.http import decode_html

FIX = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def test_small_helpers():
    assert parsers.parse_time("1:57.3") == 117.3
    assert parsers.parse_time("59.8") == 59.8
    assert parsers.parse_time("") is None
    assert parsers.parse_horse_weight("500(+4)") == (500, 4)
    assert parsers.parse_horse_weight("488(0)") == (488, 0)
    assert parsers.parse_horse_weight("計不") == (None, None)
    assert parsers.parse_sex_age("セ6") == ("セ", 6)
    assert parsers.parse_finish("3(降)") == (3, "降着")
    assert parsers.parse_finish("取") == (None, "取消")
    assert parsers.parse_course("芝右 外1600m")["course_detail"] == "外"
    assert parsers.parse_course("障芝3000m")["surface"] == "障"
    assert parsers.normalize_grade("天皇賞(秋)(GⅠ)") == "G1"
    assert parsers.normalize_grade("テスト(Jpn3)") == "G3"
    assert parsers.detect_class("3歳以上1000万下", None) == ("2勝クラス", 3)
    assert parsers.id_from_href("/horse/sire/2010100100/", "horse") == "2010100100"
    assert parsers.id_from_href("https://db.netkeiba.com/jockey/result/recent/a04e2/", "jockey") == "a04e2"


def test_decode_euc_jp():
    html = '<meta charset="EUC-JP"><p>天皇賞(秋)</p>'
    assert "天皇賞" in decode_html(html.encode("euc_jp"))
    assert "天皇賞" in decode_html(html.replace("EUC-JP", "utf-8").encode("utf-8"))


def test_decode_euc_jp_extension_chars():
    """CP51932 の NEC特殊文字・NEC選定IBM拡張文字を別の漢字に化けさせない."""
    def e(text: str) -> bytes:
        return text.encode("euc_jp")

    # 髙=FCE2 (92区), 﨑=F9F5 (89区), ①=ADA1 (13区)
    body = e('<meta charset="EUC-JP"><p>') + b"\xfc\xe2" + e("杉吏麒 宮") + b"\xf9\xf5" + e("北斗 ") + b"\xad\xa1" + e("</p>")
    text = decode_html(body)
    assert "髙杉吏麒" in text and "宮﨑北斗" in text and "①" in text
    assert "郄" not in text and "粼" not in text
    # metaタグが無くてもヘッダで EUC-JP と分かれば同じ
    plain = e("<p>") + b"\xfc\xe2" + e("木登</p>")
    assert "髙木登" in decode_html(plain, "EUC-JP")
    # 読めない文字は別の漢字にせず U+FFFD にする (euc_jis_2004 に回さない)
    broken = e('<meta charset="EUC-JP"><p>天皇賞') + b"\xf5\xa1" + e("</p>")
    text = decode_html(broken)
    assert "天皇賞" in text and "\ufffd" in text


def test_race_result():
    parsed = parsers.parse_race_result(read("race_result.html"), "202405040911")
    race = parsed["race"]
    assert race["date"] == "2024-10-27"
    assert race["venue"] == "東京"
    assert race["race_num"] == 11
    assert race["race_name"] == "テスト記念"
    assert race["grade"] == "G2"
    assert race["class_level"] == 7
    assert (race["surface"], race["direction"], race["distance"]) == ("芝", "左", 2000)
    assert race["going"] == "稍重"
    assert race["weather"] == "晴"
    assert race["start_time"] == "15:40"
    assert race["field_size"] == 4  # 取消を除く

    rows = {r["number"]: r for r in parsed["results"]}
    win = rows[7]
    assert win["horse_id"] == "2020100001"
    assert win["finish_pos"] == 1
    assert win["frame"] == 4
    assert win["time_sec"] == 117.3
    assert win["last3f"] == 32.5
    assert win["odds"] == 3.5 and win["popularity"] == 2
    assert (win["horse_weight"], win["weight_diff"]) == (500, 4)
    assert win["jockey_id"] == "01126" and win["trainer_id"] == "01061"
    assert win["trainer_region"] == "栗東"
    assert win["passing"] == "10-10-9"
    assert win["prize"] == 6700.0
    assert rows[15]["finish_status"] == "降着" and rows[15]["horse_id"] == "000a012345"
    assert rows[15]["horse_weight"] is None
    assert rows[9]["finish_pos"] is None and rows[9]["finish_status"] == "中止"
    assert rows[11]["finish_status"] == "取消"

    pays = {(p["bet_type"], p["combination"]): p["payout"] for p in parsed["payouts"]}
    assert pays[("単勝", "7")] == 350
    assert pays[("複勝", "15")] == 880
    assert pays[("ワイド", "7-15")] == 2450
    assert pays[("馬単", "7-2")] == 1200
    assert pays[("三連複", "2-7-15")] == 8760
    assert pays[("三連単", "7-2-15")] == 41230
    assert len(parsed["payouts"]) == 12


def test_shutuba():
    parsed = parsers.parse_shutuba(read("shutuba.html"), "202608040711")
    race = parsed["race"]
    assert race["date"] == "2026-10-25"
    assert race["race_name"] == "テストステークス"
    assert race["grade"] == "G3"
    assert (race["surface"], race["distance"], race["direction"]) == ("ダ", 1800, "右")
    assert race["going"] == "重" and race["weather"] == "曇"
    assert race["start_time"] == "15:35"
    assert race["race_class"] == "オープン"
    assert race["field_size"] == 3
    entries = {e["number"]: e for e in parsed["entries"]}
    assert entries[1]["horse_id"] == "2020100001"
    assert entries[1]["weight_carried"] == 57.0
    assert entries[1]["trainer_region"] == "栗東"
    assert (entries[2]["horse_weight"], entries[2]["weight_diff"]) == (458, -4)
    assert entries[3]["scratched"] == 1
    assert entries[4]["sex"] == "セ" and entries[4]["weight_diff"] == 0


@pytest.mark.parametrize("going, expected", [
    ("良", "良"), ("稍", "稍重"), ("稍重", "稍重"), ("重", "重"), ("不", "不良"), ("不良", "不良"),
])
def test_shutuba_going_abbreviations(going, expected):
    """race.netkeiba の出馬表は「馬場:稍」「馬場:不」と略す. 結果ページと同じ表記にそろえる."""
    html = read("shutuba.html").replace("馬場:重", f"馬場:{going}")
    assert parsers.parse_shutuba(html, "202608040711")["race"]["going"] == expected


@pytest.mark.parametrize("course, expected", [
    ("(右)", ("右", None)),
    ("(右 内2周)", ("右", "内2周")),
    ("(右 外-内)", ("右", "外-内")),
    ("(左 外)", ("左", "外")),
    ("(直線)", ("直線", None)),
])
def test_shutuba_course_detail_matches_result_page(course, expected):
    html = read("shutuba.html").replace("(右)", course)
    race = parsers.parse_shutuba(html, "202608040711")["race"]
    assert (race["direction"], race["course_detail"]) == expected


def test_race_result_without_win_payout_is_parse_error():
    """1着馬がいるのに単勝の払戻が読めないページは保存しない (バックテストで外れ扱いになるため)."""
    html = read("race_result.html").replace("pay_table_01", "pay_table_xx")
    with pytest.raises(parsers.ParseError, match="単勝"):
        parsers.parse_race_result(html, "202405040911")


def test_odds_json():
    parsed = parsers.parse_odds_json(read("odds.json"), "202608040711")
    odds = {o["number"]: o for o in parsed["odds"]}
    assert odds[1]["win_odds"] == 3.2 and odds[1]["popularity"] == 1
    assert odds[2]["place_odds_min"] == 1.5 and odds[2]["place_odds_max"] == 2.1
    assert odds[3]["win_odds"] is None
    assert parsed["market"] is True


def test_odds_json_forecast_is_not_market():
    """発売前の予想オッズ (status=yoso) は実際のオッズとして扱わない."""
    text = read("odds.json").replace('"status":"middle"', '"status":"yoso"')
    parsed = parsers.parse_odds_json(text, "202608040711")
    assert parsed["status"] == "yoso" and parsed["market"] is False
    # 発売前の空データも読める
    empty = parsers.parse_odds_json('{"status":"middle","data":"","reason":"result odds empty"}', "202608040711")
    assert empty["odds"] == [] and empty["market"] is True


def test_pedigree():
    ped = parsers.parse_pedigree(read("pedigree.html"), "2020100001")
    assert ped["sire"] == "テストサイアー" and ped["sire_id"] == "2010100100"
    assert ped["dam"] == "テストダム"
    assert ped["damsire"] == "テストダムサイアー" and ped["damsire_id"] == "2001100400"
    assert ped["horse_name"] == "テストホースアルファ"


def test_lists():
    assert parsers.extract_kaisai_dates(read("calendar.html")) == ["20241005", "20241006"]
    assert parsers.extract_race_ids(read("race_list.html")) == ["202405040901", "202405040911"]


def test_parse_error_on_empty_page():
    with pytest.raises(parsers.ParseError) as info:
        parsers.parse_race_result("<html><body>no data</body></html>", "202405040911")
    assert not isinstance(info.value, parsers.RaceNotFound)


def test_race_result_1970_template_is_race_not_found():
    """存在しない race_id は 1970年1月1日 の空ページになる. 成績表が無いという誤解を招く表示にしない."""
    with pytest.raises(parsers.RaceNotFound, match="race_id の誤り"):
        parsers.parse_race_result(read("race_result_empty.html"), "202405040911")


@pytest.mark.parametrize("text, expected", [
    ("芝左2000m / 天候 : 晴 / 芝 : 良", ("芝", "左", None, 2000)),
    ("芝右 外1600m", ("芝", "右", "外", 1600)),
    ("芝右 内2周3600m", ("芝", "右", "内2周", 3600)),
    ("障芝 外-内4250m", ("障", None, "外-内", 4250)),
    ("ダ右1800m", ("ダ", "右", None, 1800)),
    ("芝直線1000m", ("芝", "直線", None, 1000)),
])
def test_parse_course_variants(text, expected):
    c = parsers.parse_course(text)
    assert (c["surface"], c["direction"], c["course_detail"], c["distance"]) == expected
