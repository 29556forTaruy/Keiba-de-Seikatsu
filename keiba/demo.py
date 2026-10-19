"""オフラインで全工程を試すための架空データ生成.

netkeiba に接続できない環境でも、収集後と同じ形のDBを作って
学習・バックテスト・レポートを動かせるようにする。
馬・騎手・調教師はすべて架空で、レース条件や配当の分布も簡略化している。
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from keiba import db
from keiba.betting import combo_probs, finish_probs, normalize, place_slots
from keiba.constants import CLASS_LEVELS, TAKEOUT, VENUES

NAME_A = [
    "アオゾラ", "ハヤテ", "ツバサ", "ミライ", "ホシゾラ", "カザミ", "ヒカリ", "ナギサ", "コハル", "ユウナギ",
    "アカツキ", "シラナミ", "ハルカゼ", "イナズマ", "サザナミ", "ツキカゲ", "ヒバリ", "カスミ", "ミズホ", "トキワ",
    "アサギリ", "ユキミ", "ソヨカゼ", "ムラサメ", "ホムラ",
]
NAME_B = [
    "スター", "ボーイ", "ガール", "ロード", "キング", "クイーン", "ドリーム", "ウイング", "ブレイブ", "ソング",
    "ランナー", "フラッシュ", "ハート", "ムーン", "サン", "エース", "ジェット", "ノヴァ", "リーフ", "ベル",
    "メロディ", "アロー", "ダンサー", "ヴォイス", "ブリーズ",
]
FAMILY = ["青木", "石川", "上田", "大野", "岡本", "川口", "北村", "小島", "斎藤", "佐野",
          "清水", "高田", "田村", "中川", "西田", "野口", "原口", "福井", "松井", "山下"]
GIVEN = ["翔", "大輝", "蓮", "陽斗", "湊", "颯", "悠真", "樹", "瑛太", "陸", "結衣", "美咲"]
NAME_PRE = ["", "", "", "ニュー", "グラン", "ミス", "ドン", "ザ", "マイ", "スーパー", "ラ", "ル", "サン", "エル"]
STAKES = ["ステークス", "記念", "カップ", "特別", "賞", "トロフィー"]

TURF_DISTANCES = [1200, 1400, 1600, 1800, 2000, 2200, 2400, 2500, 3000]
DIRT_DISTANCES = [1000, 1150, 1200, 1400, 1600, 1700, 1800, 1900, 2100]
GOINGS = ["良", "稍重", "重", "不良"]
GOING_P = [0.7, 0.15, 0.1, 0.05]
# 6週ごとに開催する競馬場の組 (週番号で決めるので同名重賞が毎年同じ時期に来る)
ROTATION = [("05", "09"), ("06", "08"), ("07", "10"), ("05", "08"), ("06", "09"), ("03", "07"),
            ("04", "02"), ("01", "10"), ("04", "07")]
CLASS_PRIZE = {0: 700, 1: 560, 2: 780, 3: 1060, 4: 1550, 5: 2400, 6: 4100, 7: 6000, 8: 15000}


@dataclass
class Horse:
    idx: int
    horse_id: str
    name: str
    sex: str
    birth_year: int
    sire: int
    trainer: int
    ability: float
    turf: float
    dirt: float
    dist_pref: float
    mud: float
    style: float
    base_weight: int
    cls: int = 0  # 0=未出走, 1=未勝利, 2..4=n勝クラス, 5=オープン
    starts: int = 0
    last_run: date | None = None
    retired: bool = False


class DemoWorld:
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.sires = [
            {"id": f"DS{i:04d}", "name": f"デモサイアー{chr(0x30A2 + i * 2)}" if i < 20 else f"デモサイアー{i}",
             "turf": r.normal(0, 0.6), "dirt": r.normal(0, 0.6), "dist": r.uniform(1300, 2500)}
            for i in range(24)
        ]
        self.jockeys = [
            {"id": f"DJ{i:03d}", "name": f"{FAMILY[i % 20]}{GIVEN[(i * 7) % 12]}",
             "skill": r.normal(0, 0.18) + (0.25 if i < 6 else 0.0), "fame": (0.35 if i < 6 else 0.0)}
            for i in range(48)
        ]
        self.trainers = [
            {"id": f"DT{i:03d}", "name": f"{FAMILY[(i * 3) % 20]}{GIVEN[(i * 5) % 12]}",
             "skill": r.normal(0, 0.15), "region": "美浦" if i % 2 else "栗東"}
            for i in range(60)
        ]
        self.horses: list[Horse] = []
        self.names: set[str] = set()

    # ------------------------------------------------------------ 馬
    def _name(self) -> str:
        r = self.rng
        for _ in range(50):
            name = NAME_PRE[r.integers(len(NAME_PRE))] + NAME_A[r.integers(len(NAME_A))] + NAME_B[r.integers(len(NAME_B))]
            if name not in self.names:
                self.names.add(name)
                return name
        name = f"{name}{len(self.names)}"
        self.names.add(name)
        return name

    def new_horse(self, birth_year: int) -> Horse:
        r = self.rng
        sire_i = int(r.integers(len(self.sires)))
        sire = self.sires[sire_i]
        h = Horse(
            idx=len(self.horses),
            horse_id=f"D{birth_year}{len(self.horses):05d}",
            name=self._name(),
            sex=str(r.choice(["牡", "牝", "セ"], p=[0.5, 0.42, 0.08])),
            birth_year=birth_year,
            sire=sire_i,
            trainer=int(r.integers(len(self.trainers))),
            ability=float(r.normal(0, 0.55)),
            turf=float(sire["turf"] + r.normal(0, 0.4)),
            dirt=float(sire["dirt"] + r.normal(0, 0.4)),
            dist_pref=float(np.clip(sire["dist"] + r.normal(0, 250), 1000, 3200)),
            mud=float(r.normal(0, 0.25)),
            style=float(r.beta(2, 2)),
            base_weight=int(r.normal(475, 22)),
        )
        self.horses.append(h)
        return h

    def strength(self, h: Horse, race: dict, today: date) -> float:
        age = today.year - h.birth_year
        growth = {2: -0.45, 3: -0.05, 4: 0.2, 5: 0.25, 6: 0.05, 7: -0.25}.get(age, -0.5)
        apt = h.turf if race["surface"] == "芝" else h.dirt
        dist_fit = -abs(race["distance"] - h.dist_pref) / 550
        mud = h.mud if race["going"] in ("重", "不良") else 0.0
        trainer = self.trainers[h.trainer]["skill"]
        return h.ability + growth + 0.55 * apt + dist_fit + mud + 0.5 * trainer

    # ------------------------------------------------------------ 番組
    def plan_race(self, venue_code: str, race_num: int, day: date, going: dict[str, str]) -> dict:
        r = self.rng
        week = day.isocalendar()[1]
        if race_num == 11:
            seed = int(venue_code) * 1000 + week * 10 + day.weekday()
            rr = np.random.default_rng(seed)
            grade = rr.choice(["OP", "L", "G3", "G2", "G1"], p=[0.35, 0.2, 0.25, 0.13, 0.07])
            surface = "芝" if rr.random() < 0.7 else "ダ"
            dist = int(rr.choice(TURF_DISTANCES if surface == "芝" else DIRT_DISTANCES))
            name = f"{VENUES[venue_code]}{NAME_A[seed % len(NAME_A)]}{STAKES[seed % len(STAKES)]}"
            level = CLASS_LEVELS.get(str(grade), 5)
            race_class = "オープン"
            grade_out = None if grade == "OP" else str(grade)
        else:
            if race_num <= 3:
                level = 1
            elif race_num == 4:
                level = 0 if day.month >= 6 or day.month <= 3 else 1
            elif race_num <= 7 or race_num == 12:
                level = 2
            elif race_num <= 9:
                level = 3
            else:
                level = 4
            race_class = {0: "新馬", 1: "未勝利", 2: "1勝クラス", 3: "2勝クラス", 4: "3勝クラス"}[level]
            surface = "芝" if r.random() < 0.52 else "ダ"
            dist = int(r.choice(TURF_DISTANCES[:7] if surface == "芝" else DIRT_DISTANCES))
            age_cond = "2歳" if level == 0 and day.month >= 6 else ("3歳" if level <= 1 else "3歳以上")
            name = f"{age_cond}{race_class}"
            grade_out = None
        return {
            "venue_code": venue_code, "venue": VENUES[venue_code], "race_num": race_num,
            "race_name": name, "grade": grade_out, "race_class": race_class, "class_level": level,
            "surface": surface, "distance": dist, "going": going[surface],
            "direction": "左" if venue_code in ("04", "05", "07") else "右",
            "start_time": "{:d}:{:02d}".format(*divmod(10 * 60 + 10 + (race_num - 1) * 30, 60)),
            "date": day.isoformat(),
        }

    def pools(self, day: date) -> dict[int, list[Horse]]:
        """その日に出走できる年齢の現役馬を、番組上のクラス別に分ける."""
        by_level: dict[int, list[Horse]] = {lv: [] for lv in range(6)}
        for h in self.horses:
            age = day.year - h.birth_year
            if h.retired or age < 2 or (age == 2 and day.month < 6):
                continue
            if h.cls == 0 and age <= 3:
                by_level[0].append(h)
            if h.cls <= 1 and h.starts > 0 and age <= 3:
                by_level[1].append(h)
            if 2 <= h.cls <= 4 and age >= 3:
                by_level[h.cls].append(h)
            if h.cls >= 5 and age >= 3:
                by_level[5].append(h)
        return by_level

    def pick_field(self, race: dict, day: date, pools: dict[int, list[Horse]]) -> list[Horse]:
        pool = [
            h for h in pools[min(race["class_level"], 5)]
            if h.last_run is None or (day - h.last_run).days >= 21
        ]
        if len(pool) < 5:
            return []
        r = self.rng
        size = int(min(len(pool), r.integers(8, 19)))
        fit = np.array([
            (h.turf if race["surface"] == "芝" else h.dirt) - abs(race["distance"] - h.dist_pref) / 400
            for h in pool
        ])
        w = np.exp(1.5 * fit)
        chosen = r.choice(len(pool), size=size, replace=False, p=w / w.sum())
        return [pool[i] for i in chosen]

    # ------------------------------------------------------------ 1レース
    def build_card(self, race: dict, field: list[Horse], day: date) -> list[dict]:
        r = self.rng
        n = len(field)
        order = r.permutation(n)
        rows = []
        for slot, i in enumerate(order, start=1):
            h = field[i]
            frame = 1 + (slot - 1) * 8 // n if n > 8 else slot
            age = day.year - h.birth_year
            wc = 57.0 if age >= 4 else (56.0 if age == 3 else 55.0)
            if h.sex == "牝":
                wc -= 2.0
            jockey_i = int(r.choice(len(self.jockeys), p=self._jockey_weights(h, race)))
            weight = int(round((h.base_weight + r.normal(0, 6)) / 2) * 2)
            rows.append({
                "horse": h, "number": slot, "frame": frame, "weight_carried": wc, "jockey": jockey_i,
                "horse_weight": weight, "weight_diff": int(round(r.normal(0, 5) / 2) * 2),
            })
        return rows

    def _jockey_weights(self, h: Horse, race: dict) -> np.ndarray:
        skill = np.array([j["skill"] for j in self.jockeys])
        w = np.exp(skill * (2.0 + 0.4 * race["class_level"]) + 0.1 * h.ability)
        return w / w.sum()

    def market(self, race: dict, card: list[dict], day: date) -> tuple[np.ndarray, np.ndarray]:
        """(真の能力値, 市場の勝率)"""
        r = self.rng
        strength = np.array([
            self.strength(c["horse"], race, day)
            + self.jockeys[c["jockey"]]["skill"]
            - (c["weight_carried"] - 55) * 0.04
            + (0.04 * (4.5 - c["frame"]) if race["surface"] == "芝" and race["distance"] <= 1600 else 0.0)
            for c in card
        ])
        fame = np.array([self.jockeys[c["jockey"]]["fame"] for c in card])
        # 実際のJRAに近い回収率 (人気馬ほど高めで8割前後) になるよう調整した市場モデル
        est = strength + r.normal(0, 0.28, len(card)) + 0.4 * fame
        p_market = normalize(np.exp(1.7 * est))
        return strength, p_market

    def run_race(self, race: dict, card: list[dict], day: date) -> tuple[list[dict], list[dict]]:
        r = self.rng
        strength, p_market = self.market(race, card, day)
        n = len(card)
        perf = strength + r.normal(0, 0.62, n)
        scratched = r.random(n) < 0.01
        dnf = (r.random(n) < 0.003) & ~scratched
        perf[scratched | dnf] = -np.inf
        finish_order = np.argsort(-perf)

        odds = np.maximum(1.0, np.floor((1 - TAKEOUT["単勝"]) / p_market * 10) / 10)
        pop = np.argsort(np.argsort(odds)) + 1

        dist = race["distance"]
        speed = 16.9 - 0.00075 * (dist - 1200)
        if race["surface"] == "ダ":
            speed *= 0.962
        going_adj = {"芝": [1.0, 0.995, 0.988, 0.98], "ダ": [1.0, 1.004, 1.008, 1.01]}[race["surface"]]
        speed *= going_adj[GOINGS.index(race["going"])]
        top = perf[finish_order[0]]
        win_time = dist / speed - race["class_level"] * 0.12 * dist / 1600 - top * 0.25 + r.normal(0, 0.35)
        corners = 2 if dist <= 1400 else (3 if dist <= 1800 else 4)
        style = np.array([c["horse"].style for c in card])
        early = np.argsort(np.argsort(style + r.normal(0, 0.12, n))) + 1

        results = []
        t_prev = win_time
        for rank, i in enumerate(finish_order, start=1):
            c = card[i]
            h: Horse = c["horse"]
            row = self._base_row(race, c, day)
            if scratched[i]:
                row.update(finish_pos=None, finish_status="取消", odds=None, popularity=None)
                results.append(row)
                continue
            row.update(odds=float(odds[i]), popularity=int(pop[i]))
            if dnf[i]:
                row.update(finish_pos=None, finish_status="中止")
            else:
                gap = (top - perf[i]) * 0.33 * dist / 1600
                t = max(t_prev, win_time + gap + abs(r.normal(0, 0.02)))
                margin_len = (t - t_prev) / 0.17
                t_prev = t
                last3f = (34.3 if race["surface"] == "芝" else 37.2) + (0.6 if dist > 2000 else 0) \
                    - perf[i] * 0.25 - (style[i] - 0.5) * 0.8 + r.normal(0, 0.3) \
                    + (0.5 if race["going"] in ("重", "不良") else 0)
                pos = [int(early[i])]
                for k in range(1, corners):
                    mix = (corners - k) / corners
                    pos.append(int(np.clip(round(mix * early[i] + (1 - mix) * rank + r.normal(0, 1)), 1, n)))
                row.update(
                    finish_pos=rank,
                    finish_status=None,
                    time_sec=round(t, 1),
                    margin=None if rank == 1 else _margin_text(margin_len),
                    passing="-".join(str(x) for x in pos),
                    last3f=round(float(last3f), 1),
                    prize=float(CLASS_PRIZE[race["class_level"]] * [1, 0.4, 0.25, 0.15, 0.1][rank - 1]) if rank <= 5 else 0.0,
                )
            results.append(row)
            h.starts += 1
            h.last_run = day

        finishers = [card[i] for i in finish_order if np.isfinite(perf[i])]
        winner: Horse = finishers[0]["horse"]
        if race["class_level"] <= 4:
            winner.cls = max(winner.cls, min(5, max(race["class_level"], 1) + 1))
        if race["class_level"] == 0:
            for c in finishers[1:]:
                c["horse"].cls = max(c["horse"].cls, 1)
        payouts = self._payouts(race, card, p_market, finishers, odds)
        return results, payouts

    def _base_row(self, race: dict, c: dict, day: date) -> dict:
        h: Horse = c["horse"]
        j = self.jockeys[c["jockey"]]
        t = self.trainers[h.trainer]
        return {
            "race_id": race["race_id"], "horse_id": h.horse_id, "horse_name": h.name,
            "frame": c["frame"], "number": c["number"], "sex": h.sex, "age": day.year - h.birth_year,
            "weight_carried": c["weight_carried"], "jockey_id": j["id"], "jockey_name": j["name"],
            "trainer_id": t["id"], "trainer_name": t["name"], "trainer_region": t["region"],
            "horse_weight": c["horse_weight"], "weight_diff": c["weight_diff"],
            "time_sec": None, "margin": None, "passing": None, "last3f": None, "prize": None,
            "owner": "デモ馬主",
        }

    def _payouts(self, race: dict, card: list[dict], p_market: np.ndarray, finishers: list[dict], odds) -> list[dict]:
        numbers = [c["number"] for c in card]
        idx = {c["number"]: i for i, c in enumerate(card)}
        top = [c["number"] for c in finishers[:3]]
        out = []

        def pay(bet_type: str, combo: str, p: float) -> None:
            amount = max(100, int(round((1 - TAKEOUT[bet_type]) / max(p, 1e-6) * 100, -1)))
            out.append({"race_id": race["race_id"], "bet_type": bet_type, "combination": combo, "payout": amount, "popularity": None})

        out.append({"race_id": race["race_id"], "bet_type": "単勝", "combination": str(top[0]),
                    "payout": int(round(odds[idx[top[0]]] * 100)), "popularity": None})
        if len(finishers) < 3:
            return out
        fin = finish_probs(p_market)
        slots = place_slots(len(finishers))
        for num in top[:slots]:
            key = "top2" if slots == 2 else "top3"
            pay("複勝", str(num), float(fin[key][idx[num]]))
        combos = combo_probs(p_market, numbers)
        a, b, c = top
        pay("馬連", "-".join(map(str, sorted((a, b)))), combos["馬連"]["-".join(map(str, sorted((a, b))))])
        pay("馬単", f"{a}-{b}", combos["馬単"][f"{a}-{b}"])
        for x, y in ((a, b), (a, c), (b, c)):
            key = "-".join(map(str, sorted((x, y))))
            pay("ワイド", key, combos["ワイド"][key])
        key3 = "-".join(map(str, sorted(top)))
        pay("三連複", key3, combos["三連複"][key3])
        pay("三連単", f"{a}-{b}-{c}", combos["三連単"][f"{a}-{b}-{c}"])
        return out

    def age_out(self, day: date) -> None:
        for h in self.horses:
            age = day.year - h.birth_year
            if age >= 8 or (age >= 4 and h.cls <= 1) or h.starts >= 35:
                h.retired = True


def _margin_text(lengths: float) -> str:
    table = [(0.05, "ハナ"), (0.1, "アタマ"), (0.25, "クビ"), (0.6, "1/2"), (0.85, "3/4"), (1.1, "1"),
             (1.35, "1.1/4"), (1.6, "1.1/2"), (1.85, "1.3/4"), (2.25, "2"), (2.75, "2.1/2"), (3.25, "3"),
             (3.75, "3.1/2"), (4.5, "4"), (5.5, "5"), (6.5, "6"), (7.5, "7"), (8.5, "8"), (9.5, "9"), (10.5, "10")]
    for limit, text in table:
        if lengths < limit:
            return text
    return "大差"


def _going_for_day(rng: np.random.Generator) -> dict[str, str]:
    turf = str(rng.choice(GOINGS, p=GOING_P))
    dirt = str(rng.choice(GOINGS, p=GOING_P))
    return {"芝": turf, "ダ": dirt}


def generate(
    conn: sqlite3.Connection,
    *,
    seed: int = 42,
    start: date = date(2023, 1, 7),
    weeks: int = 104,
    venues_per_day: int = 2,
    races_per_venue: int = 12,
    progress=print,
) -> date:
    """架空の過去レースと、最後に「次の開催日」の出馬表・オッズを作る。次の開催日を返す."""
    world = DemoWorld(seed)
    rng = world.rng
    for birth in range(start.year - 6, start.year - 1):
        for _ in range(700):
            h = world.new_horse(birth)
            age = start.year - birth
            h.starts = int(rng.integers(3, 15)) if age >= 3 else 0
            h.cls = int(np.clip(round(1 + h.ability * 2 + rng.normal(0, 0.8) + (age - 3) * 0.6), 1, 5)) if age >= 3 else 0
    crop_year = start.year - 2
    day_counter: dict[tuple[int, str], int] = {}

    def race_day(day: date, *, upcoming: bool) -> int:
        nonlocal crop_year
        if day.year - 2 > crop_year:
            crop_year = day.year - 2
            for _ in range(900):
                world.new_horse(crop_year)
            world.age_out(day)
        week = day.isocalendar()[1]
        venue_codes = ROTATION[((week - 1) // 6) % len(ROTATION)][:venues_per_day]
        made = 0
        pools = world.pools(day)
        for venue_code in venue_codes:
            # 同じ年・競馬場の開催日数から 回・日目 を振る (8日で1開催)
            held = day_counter.get((day.year, venue_code), 0)
            day_counter[(day.year, venue_code)] = held + 1
            kai, nichi = 1 + held // 8, 1 + held % 8
            going = _going_for_day(rng)
            weather = "晴" if going["芝"] == "良" else ("曇" if going["芝"] == "稍重" else "雨")
            for race_num in range(1, races_per_venue + 1):
                race = world.plan_race(venue_code, race_num, day, going)
                race_id = f"{day.year}{venue_code}{kai:02d}{nichi:02d}{race_num:02d}"
                race.update(race_id=race_id, kai=kai, nichi=nichi, weather=weather,
                            conditions=race["race_name"], source="demo", course_detail=None)
                field = world.pick_field(race, day, pools)
                if not field:
                    continue
                card = world.build_card(race, field, day)
                race["field_size"] = len(card)
                if upcoming:
                    _, p_market = world.market(race, card, day)
                    db.save_race(conn, race)
                    db.save_entries(conn, race_id, [_entry_row(world, race, c, day) for c in card])
                    db.save_odds(conn, race_id, _demo_odds(race_id, card, p_market))
                    for c in card:
                        c["horse"].last_run = day
                else:
                    results, payouts = world.run_race(race, card, day)
                    race["field_size"] = sum(1 for x in results if x["finish_status"] != "取消")
                    db.save_race(conn, race)
                    db.save_results(conn, race_id, results)
                    db.save_payouts(conn, race_id, payouts)
                made += 1
        return made

    day = start
    total = 0
    for w in range(weeks):
        for d in (day, day + timedelta(days=1)):
            total += race_day(d, upcoming=False)
        conn.commit()
        if (w + 1) % 13 == 0:
            progress(f"デモデータ: {w + 1}/{weeks}週 ({total}レース)")
        day += timedelta(days=7)

    for h in world.horses:
        sire = world.sires[h.sire]
        db.save_horse(conn, {"horse_id": h.horse_id, "horse_name": h.name, "sire": sire["name"], "sire_id": sire["id"],
                             "dam": None, "dam_id": None, "damsire": None, "damsire_id": None})
    upcoming = day
    race_day(upcoming, upcoming=True)
    conn.commit()
    progress(f"デモデータ: 過去{total}レース + {upcoming} の出馬表を作成しました")
    return upcoming


def _demo_odds(race_id: str, card: list[dict], p_market: np.ndarray) -> list[dict]:
    top = finish_probs(p_market)["top2" if place_slots(len(card)) == 2 else "top3"]
    win_odds = np.maximum(1.0, np.floor((1 - TAKEOUT["単勝"]) / p_market * 10) / 10)
    pop = np.argsort(np.argsort(win_odds)) + 1
    rows = []
    for i, c in enumerate(card):
        fair = (1 - TAKEOUT["複勝"]) / top[i]
        rows.append({
            "race_id": race_id, "number": c["number"], "win_odds": float(win_odds[i]),
            "place_odds_min": float(max(1.0, math.floor(fair * 0.75 * 10) / 10)),
            "place_odds_max": float(max(1.1, math.floor(fair * 1.3 * 10) / 10)),
            "popularity": int(pop[i]),
        })
    return rows


def _entry_row(world: DemoWorld, race: dict, c: dict, day: date) -> dict:
    row = world._base_row(race, c, day)
    return {k: row.get(k) for k in db.ENTRY_COLS if k != "scratched"} | {"scratched": 0}
