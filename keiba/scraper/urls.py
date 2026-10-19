"""netkeiba のURL."""

from __future__ import annotations

DB = "https://db.netkeiba.com"
RACE = "https://race.netkeiba.com"


def calendar(year: int, month: int) -> str:
    return f"{RACE}/top/calendar.html?year={year}&month={month}"


def race_list(kaisai_date: str) -> str:
    """kaisai_date: YYYYMMDD"""
    return f"{RACE}/top/race_list_sub.html?kaisai_date={kaisai_date}"


def db_race_list(kaisai_date: str) -> str:
    return f"{DB}/race/list/{kaisai_date}/"


def race_result(race_id: str) -> str:
    return f"{DB}/race/{race_id}/"


def shutuba(race_id: str) -> str:
    return f"{RACE}/race/shutuba.html?race_id={race_id}"


def odds_api(race_id: str, bet_type: int = 1) -> str:
    """bet_type=1: 単勝・複勝"""
    return f"{RACE}/api/api_get_jra_odds.html?race_id={race_id}&type={bet_type}&action=update"


def horse(horse_id: str) -> str:
    return f"{DB}/horse/{horse_id}/"


def pedigree(horse_id: str) -> str:
    return f"{DB}/horse/ped/{horse_id}/"
