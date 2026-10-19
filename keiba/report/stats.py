"""レポート用の集計 (馬柱・着別度数・コース傾向など).

すべて「対象レースの日付より前」のデータだけを使う。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from keiba.constants import distance_band


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (float, np.floating)) and not np.isfinite(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    return v


def _int(v: Any) -> int | None:
    v = _clean(v)
    return None if v is None else int(round(float(v)))


def _sex_age(sex: Any, age: Any) -> str | None:
    """性齢 (牡4). 片方が欠けていても「nan」「None」を混ぜない."""
    sex, age = _clean(sex), _int(age)
    return f"{sex or ''}{'' if age is None else age}" or None


def record(rows: pd.DataFrame) -> dict[str, Any]:
    """着別度数 (1着-2着-3着-着外) と勝率・複勝率."""
    pos = rows["finish_pos"]
    n = len(rows)
    first, second, third = int((pos == 1).sum()), int((pos == 2).sum()), int((pos == 3).sum())
    return {
        "n": n,
        "text": f"{first}-{second}-{third}-{n - first - second - third}",
        "win": first / n if n else None,
        "top3": (first + second + third) / n if n else None,
    }


def running_style(first_pos: float | None, passing: str | None) -> str | None:
    """1コーナー(最初の通過順)の位置から脚質を分類する."""
    if passing:
        head = str(passing).split("-")[0]
        if head.isdigit() and int(head) == 1:
            return "逃げ"
    if first_pos is None or not np.isfinite(first_pos):
        return None
    if first_pos <= 0.35:
        return "先行"
    if first_pos <= 0.7:
        return "差し"
    return "追込"


def history_before(by_horse: dict[str, pd.DataFrame], horse_id: str, date: pd.Timestamp) -> pd.DataFrame:
    """by_horse: horse_id → その馬の全成績 (index_by_horse で作る)"""
    h = by_horse.get(horse_id)
    if h is None:
        return pd.DataFrame()
    h = h[(h["date"] < date) & (h["is_future"] == 0)]
    return h.sort_values("date", ascending=False)


def index_by_horse(feat: pd.DataFrame, horse_ids) -> dict[str, pd.DataFrame]:
    sub = feat[feat["horse_id"].isin(set(horse_ids))]
    return {hid: g for hid, g in sub.groupby("horse_id", sort=False)}


def top2_by_race(feat: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    """race_id → [(1着馬ID, 馬名), (2着馬ID, 馬名)]"""
    top = feat[feat["finish_pos"].isin([1, 2])].sort_values(["race_id", "finish_pos"])
    out: dict[str, list[tuple[str, str]]] = {}
    for race_id, horse_id, name in zip(top["race_id"], top["horse_id"], top["horse_name"]):
        out.setdefault(race_id, []).append((horse_id, name))
    return out


def past_runs(hist: pd.DataFrame, top2: dict[str, list[tuple[str, str]]], n: int = 5) -> list[dict[str, Any]]:
    """馬柱用の近走リスト (新しい順). 勝った時は2着馬、負けた時は勝ち馬を添える."""
    out = []
    for _, r in hist.head(n).iterrows():
        rival = next((name for hid, name in top2.get(r["race_id"], []) if hid != r["horse_id"]), None)
        out.append({
            "date": r["date"].strftime("%Y/%m/%d"),
            "race_id": r["race_id"],
            "venue": r["venue"],
            "race_name": r["race_name"],
            "grade": _clean(r.get("grade")),
            "surface": r["surface"],
            "distance": _int(r["distance"]),
            "going": r["going"],
            "field_size": _int(r["field_size"]),
            "number": _int(r["number"]),
            "popularity": _int(r["popularity"]),
            "odds": _clean(r.get("win_odds")),
            "finish_pos": _int(r["finish_pos"]),
            "finish_status": _clean(r.get("finish_status")),
            "time": _format_time(r["time_sec"]),
            "margin": _clean(r.get("margin")),
            "passing": _clean(r.get("passing")),
            "last3f": _clean(r.get("last3f")),
            "jockey": _clean(r["jockey_name"]),
            "weight_carried": _clean(r["weight_carried"]),
            "horse_weight": _int(r.get("horse_weight")),
            "weight_diff": _int(r.get("weight_diff")),
            "speed_fig": _clean(r.get("speed_fig")),
            "rival": rival,
            "style": running_style(_clean(r.get("first_pos")), _clean(r.get("passing"))),
        })
    return out


def _format_time(sec: Any) -> str | None:
    sec = _clean(sec)
    if sec is None:
        return None
    m, s = divmod(float(sec), 60)
    return f"{int(m)}:{s:04.1f}" if m else f"{s:.1f}"


def aptitude(hist: pd.DataFrame, race: dict[str, Any]) -> list[dict[str, Any]]:
    """条件別の着別度数."""
    band = distance_band(race["distance"])
    heavy = race.get("going") in ("重", "不良")
    groups = [
        ("全成績", hist),
        (f"{race['surface']}コース", hist[hist["surface"] == race["surface"]]),
        (band, hist[hist["distance"].map(distance_band) == band]),
        (race["venue"], hist[hist["venue"] == race["venue"]]),
        (f"同コース ({race['venue']}{race['surface']}{race['distance']}m)",
         hist[(hist["venue"] == race["venue"]) & (hist["surface"] == race["surface"]) & (hist["distance"] == race["distance"])]),
        ("道悪 (重・不良)" if heavy else "良・稍重", hist[hist["going"].isin(["重", "不良"]) == heavy]),
    ]
    return [{"label": label, **record(rows)} for label, rows in groups]


def course_trends(feat: pd.DataFrame, race: dict[str, Any], date: pd.Timestamp) -> dict[str, Any]:
    """同じ競馬場・芝ダ・距離の過去レースの傾向."""
    past = feat[
        (feat["is_future"] == 0)
        & (feat["date"] < date)
        & (feat["venue"] == race["venue"])
        & (feat["surface"] == race["surface"])
        & (feat["distance"] == race["distance"])
    ].copy()
    out: dict[str, Any] = {"races": int(past["race_id"].nunique())}
    if past.empty:
        return out
    if past["date"].min() < date - pd.Timedelta(days=365 * 5):
        past = past[past["date"] >= date - pd.Timedelta(days=365 * 5)]
        out["races"] = int(past["race_id"].nunique())
    out["since"] = past["date"].min().strftime("%Y/%m/%d")

    def table(key: pd.Series, order: list[Any] | None = None) -> list[dict[str, Any]]:
        rows = []
        keys = order if order is not None else sorted(key.dropna().unique())
        for k in keys:
            g = past[key == k]
            if g.empty:
                continue
            won = g["finish_pos"] == 1
            stake = g["win_odds"].notna().sum() * 100
            ret = (g.loc[won, "win_odds"].fillna(0) * 100).sum()
            rows.append({"key": k, **record(g), "roi": ret / stake if stake else None})
        return rows

    out["frame"] = table(past["frame"], list(range(1, 9)))
    styles = [running_style(fp, ps) for fp, ps in zip(past["first_pos"], past["passing"])]
    out["style"] = table(pd.Series(styles, index=past.index), ["逃げ", "先行", "差し", "追込"])
    pop = past["popularity"].clip(upper=10)
    out["popularity"] = [
        {**r, "key": "10番人気以下" if r["key"] == 10 else f"{int(r['key'])}番人気"}
        for r in table(pop, list(range(1, 11)))
    ]

    def leaders(col: str, name_col: str, min_n: int = 10, k: int = 8) -> list[dict[str, Any]]:
        rows = []
        for key_val, g in past.groupby(col):
            if len(g) < min_n:
                continue
            rows.append({"name": g[name_col].iloc[-1], **record(g)})
        return sorted(rows, key=lambda r: (-r["top3"], -r["n"]))[:k]

    out["jockeys"] = leaders("jockey_id", "jockey_name")
    if past["sire_id"].notna().any():
        out["sires"] = leaders("sire_id", "sire", min_n=8)
    return out


def same_race_history(feat: pd.DataFrame, race: dict[str, Any], date: pd.Timestamp, k: int = 10) -> list[dict[str, Any]]:
    """同名レースの過去の勝ち馬."""
    name = race.get("race_name")
    if not name or race.get("class_level") is None or race["class_level"] < 5:
        return []
    past = feat[(feat["race_name"] == name) & (feat["date"] < date) & (feat["finish_pos"] == 1)]
    out = []
    for _, r in past.sort_values("date", ascending=False).head(k).iterrows():
        out.append({
            "date": r["date"].strftime("%Y/%m/%d"),
            "horse": _clean(r["horse_name"]),
            "sex_age": _sex_age(r["sex"], r["age"]),
            "jockey": _clean(r["jockey_name"]),
            "popularity": _int(r["popularity"]),
            "odds": _clean(r["win_odds"]),
            # 枠順前の出馬表が DB にあると frame 列全体が float (5.0) になるので整数に戻す
            "frame": _int(r["frame"]),
            "time": _format_time(r["time_sec"]),
            "going": _clean(r["going"]),
            "style": running_style(_clean(r.get("first_pos")), _clean(r.get("passing"))),
        })
    return out
