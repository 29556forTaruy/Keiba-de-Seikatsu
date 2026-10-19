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


def test_odds_json():
    parsed = parsers.parse_odds_json(read("odds.json"), "202608040711")
    odds = {o["number"]: o for o in parsed["odds"]}
    assert odds[1]["win_odds"] == 3.2 and odds[1]["popularity"] == 1
    assert odds[2]["place_odds_min"] == 1.5 and odds[2]["place_odds_max"] == 2.1
    assert odds[3]["win_odds"] is None


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
    with pytest.raises(parsers.ParseError):
        parsers.parse_race_result("<html><body>no data</body></html>", "202405040911")
