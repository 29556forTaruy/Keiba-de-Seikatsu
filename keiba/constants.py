"""JRAの固定的なコード・区分."""

from __future__ import annotations

# netkeiba / JRA の競馬場コード (race_id の5-6桁目)
VENUES: dict[str, str] = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}
VENUE_CODES: dict[str, str] = {v: k for k, v in VENUES.items()}

SURFACES = ("芝", "ダ", "障")
GOINGS = ("良", "稍重", "重", "不良")

BET_TYPES = ("単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単")
# 着順を区別しない券種 (組番を昇順に正規化する)
UNORDERED_BETS = {"枠連", "馬連", "ワイド", "三連複"}

# JRAの控除率 (払戻率 = 1 - 控除率)
TAKEOUT = {
    "単勝": 0.20,
    "複勝": 0.20,
    "枠連": 0.225,
    "馬連": 0.225,
    "ワイド": 0.225,
    "馬単": 0.25,
    "三連複": 0.25,
    "三連単": 0.275,
}

# クラスの序列 (数値が大きいほど上のクラス)
CLASS_LEVELS: dict[str, int] = {
    "新馬": 0,
    "未勝利": 1,
    "1勝クラス": 2,
    "2勝クラス": 3,
    "3勝クラス": 4,
    "オープン": 5,
    "L": 5,
    "G3": 6,
    "G2": 7,
    "G1": 8,
}
# 2019年夏以前の旧クラス名
LEGACY_CLASS_NAMES = {
    "500万下": "1勝クラス",
    "1000万下": "2勝クラス",
    "1600万下": "3勝クラス",
}

MARKS = ("◎", "○", "▲", "△", "☆", "注")


def distance_band(distance: int | float | None) -> str:
    """距離をカテゴリに丸める (短距離/マイル/中距離/長距離)."""
    if distance is None or distance != distance:  # NaN
        return "不明"
    d = int(distance)
    if d <= 1400:
        return "短距離"
    if d <= 1800:
        return "マイル"
    if d <= 2200:
        return "中距離"
    return "長距離"


def venue_from_race_id(race_id: str) -> str | None:
    return VENUES.get(race_id[4:6]) if len(race_id) == 12 else None


def is_jra_race_id(race_id: str) -> bool:
    return len(race_id) == 12 and race_id.isdigit() and race_id[4:6] in VENUES
