"""netkeiba のHTMLを辞書に変換する純粋関数群.

HTMLの列順が変わっても壊れにくいよう、表は見出しセルの文字列で列を特定する。
パースに失敗したら ParseError を投げ、保存済みHTML (data/raw) をテスト用
フィクスチャにして直せるようにしている。
"""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from bs4 import BeautifulSoup, Tag

from keiba.constants import (
    CLASS_LEVELS,
    LEGACY_CLASS_NAMES,
    UNORDERED_BETS,
    VENUES,
    is_jra_race_id,
)


class ParseError(ValueError):
    pass


# ---------------------------------------------------------------- 文字列ユーティリティ

def norm(text: str | None) -> str:
    """全角英数・半角カナを正規化し、空白を詰める."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def cell_text(el: Tag | None) -> str:
    return norm(el.get_text(" ", strip=True)) if el is not None else ""


def to_int(text: str | None) -> int | None:
    m = re.search(r"-?\d+", (text or "").replace(",", ""))
    return int(m.group()) if m else None


def to_float(text: str | None) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", (text or "").replace(",", ""))
    return float(m.group()) if m else None


def parse_time(text: str | None) -> float | None:
    """'1:57.3' → 117.3, '57.3' → 57.3"""
    text = norm(text)
    m = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)", text)
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    return round(minutes * 60 + float(m.group(2)), 1)


def parse_horse_weight(text: str | None) -> tuple[int | None, int | None]:
    """'500(+4)' → (500, 4), '計不' → (None, None)"""
    text = norm(text)
    m = re.search(r"(\d{3})\s*(?:\(\s*([+-]?\d+)\s*\))?", text)
    if not m:
        return None, None
    diff = int(m.group(2)) if m.group(2) is not None else None
    return int(m.group(1)), diff


def parse_sex_age(text: str | None) -> tuple[str | None, int | None]:
    m = re.search(r"(牡|牝|セ)\s*(\d+)", norm(text))
    if not m:
        return None, None
    return m.group(1), int(m.group(2))


_STATUS = {"取": "取消", "除": "除外", "中": "中止", "失": "失格"}


def parse_finish(text: str | None) -> tuple[int | None, str | None]:
    """着順セル → (着順, 異常区分). '3(降)' は (3, '降着')."""
    text = norm(text)
    m = re.match(r"(\d+)", text)
    if m:
        return int(m.group(1)), ("降着" if "降" in text else None)
    for key, status in _STATUS.items():
        if key in text:
            return None, status
    return None, text or None


def id_from_href(href: str | None, kind: str) -> str | None:
    """'/horse/2021105898/' → '2021105898'. kind: horse / jockey / trainer / owner"""
    if not href:
        return None
    m = re.search(rf"/{kind}/(?:(?:result/recent|result|sire|ped)/)?([0-9A-Za-z]{{5,}})", href)
    return m.group(1) if m else None


def parse_course(text: str) -> dict[str, Any]:
    """'芝左2000m' / 'ダ右1800m' / '芝右 外1600m' / '障芝3000m' / '芝右 内2周3600m' / '障芝 外-内4250m' を分解する."""
    text = norm(text)
    out: dict[str, Any] = {"surface": None, "direction": None, "distance": None, "course_detail": None}
    head = text.split("/")[0]
    surface = re.search(r"(障|芝|ダ)", head)
    distance = re.search(r"(\d{3,4})\s*m", head)
    if not surface or not distance:
        return out
    out["surface"] = surface.group(1)
    out["distance"] = int(distance.group(1))
    between = head[surface.end():distance.start()]
    direction = re.search(r"(右|左|直線|直)", between)
    if direction:
        out["direction"] = "直線" if direction.group(1).startswith("直") else direction.group(1)
    detail = re.search(r"(外-内|内-外|外内|内外|内2周|外2周|2周|外|内)", between)
    out["course_detail"] = detail.group(1) if detail else None
    return out


def normalize_grade(text: str | None) -> str | None:
    """'(G1)' '(GI)' '(Jpn1)' '(L)' などからグレードを取り出す."""
    text = norm(text)
    m = re.search(r"\(\s*(J\.?\s*)?(G|Jpn)\s*(I{1,3}|[1-3])\s*\)", text)
    if m:
        num = {"I": "1", "II": "2", "III": "3"}.get(m.group(3), m.group(3))
        return f"G{num}"
    if re.search(r"\(\s*L\s*\)", text):
        return "L"
    if re.search(r"\(\s*OP\s*\)", text):
        return "OP"
    return None


def detect_class(conditions: str, grade: str | None) -> tuple[str | None, int | None]:
    """条件文字列とグレードからクラス名と序列を求める."""
    text = norm(conditions)
    for legacy, new in LEGACY_CLASS_NAMES.items():
        text = text.replace(legacy, new)
    race_class = None
    for name in ("新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン"):
        if name in text:
            race_class = name
            break
    if grade in ("G1", "G2", "G3", "L"):
        race_class = race_class or "オープン"
        return race_class, CLASS_LEVELS[grade]
    if grade == "OP":
        race_class = "オープン"
    if race_class is None:
        return None, None
    return race_class, CLASS_LEVELS[race_class]


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _parse_date(text: str) -> str | None:
    m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text)
    if not m:
        m = re.search(r"(\d{4})(\d{2})(\d{2})", text)
        if not m:
            return None
    y, mo, d = (int(g) for g in m.groups())
    return f"{y:04d}-{mo:02d}-{d:02d}"


def _race_id_parts(race_id: str) -> dict[str, Any]:
    return {
        "venue_code": race_id[4:6],
        "venue": VENUES.get(race_id[4:6]),
        "kai": int(race_id[6:8]),
        "nichi": int(race_id[8:10]),
        "race_num": int(race_id[10:12]),
    }


# ---------------------------------------------------------------- 開催日・レース一覧

def extract_kaisai_dates(html: str) -> list[str]:
    """カレンダーページから開催日 (YYYYMMDD) を抜き出す."""
    return sorted(set(re.findall(r"kaisai_date=(\d{8})", html)))


def extract_race_ids(html: str) -> list[str]:
    """レース一覧・馬ページなどから中央競馬の race_id を出現順に抜き出す."""
    seen: dict[str, None] = {}
    for rid in re.findall(r"(?:race_id=|/race/)(\d{12})", html):
        if is_jra_race_id(rid):
            seen.setdefault(rid, None)
    return list(seen)


# ---------------------------------------------------------------- レース結果 (db.netkeiba.com/race/{id}/)

_RESULT_HEADERS = {
    "着順": "finish",
    "枠番": "frame",
    "枠": "frame",
    "馬番": "number",
    "馬名": "horse",
    "性齢": "sex_age",
    "斤量": "weight_carried",
    "騎手": "jockey",
    "タイム": "time",
    "着差": "margin",
    "通過": "passing",
    "上り": "last3f",
    "上がり": "last3f",
    "単勝": "odds",
    "人気": "popularity",
    "馬体重": "horse_weight",
    "調教師": "trainer",
    "馬主": "owner",
    "賞金(万円)": "prize",
    "賞金": "prize",
}

_REGION = {"東": "美浦", "西": "栗東", "地": "地方", "外": "海外"}


def _find_table(soup: BeautifulSoup, css: str, required: tuple[str, ...]) -> Tag | None:
    table = soup.select_one(css)
    if table is not None:
        return table
    for t in soup.find_all("table"):
        heads = {norm(th.get_text()) for th in t.find_all("th")}
        if all(r in heads for r in required):
            return t
    return None


def _header_map(table: Tag, mapping: dict[str, str]) -> dict[int, str]:
    header_row = table.find("tr")
    cols: dict[int, str] = {}
    for i, th in enumerate(header_row.find_all(["th", "td"])):
        key = norm(th.get_text()).replace(" ", "")
        if key in mapping and mapping[key] not in cols.values():
            cols[i] = mapping[key]
    return cols


def parse_race_result(html: str, race_id: str) -> dict[str, Any]:
    """確定成績ページ → {'race': {...}, 'results': [...], 'payouts': [...]}"""
    soup = _soup(html)
    intro = soup.select_one("div.data_intro") or soup
    race: dict[str, Any] = {"race_id": race_id, **_race_id_parts(race_id), "source": "netkeiba"}

    h1 = intro.find("h1")
    race_name = cell_text(h1)
    race["grade"] = normalize_grade(race_name)
    for img in (h1.find_all("img") if h1 else []):
        alt = norm(img.get("alt") or img.get("src") or "")
        race["grade"] = race["grade"] or normalize_grade(f"({alt})")
    race["race_name"] = re.sub(r"\s*\((?:J\.?)?(?:G|Jpn)\s*(?:I{1,3}|[1-3])\)|\s*\((?:L|OP)\)", "", race_name).strip() or None

    racedata = intro.select_one("dl.racedata p") or intro.select_one("diary_snap_cut") or intro
    info = cell_text(racedata)
    race.update(parse_course(info))
    m = re.search(r"天候\s*:\s*(\S+)", info)
    race["weather"] = m.group(1) if m else None
    m = re.search(r"(?:芝|ダート|ダ)\s*:\s*(良|稍重|重|不良)", info)
    race["going"] = m.group(1) if m else None
    m = re.search(r"発走\s*:\s*(\d{1,2}:\d{2})", info)
    race["start_time"] = m.group(1) if m else None

    small = cell_text(intro.select_one("p.smalltxt"))
    race["date"] = _parse_date(small) or _parse_date(cell_text(soup.find("title")))
    if race["date"] is None:
        raise ParseError(f"開催日が読み取れません: {race_id}")
    m = re.search(r"\d+回\S+?\d+日目\s*(.*)", small)
    race["conditions"] = m.group(1).strip() if m else small
    race["race_class"], race["class_level"] = detect_class(race["conditions"], race["grade"])

    table = _find_table(soup, "table.race_table_01", ("着順", "馬名"))
    if table is None:
        raise ParseError(f"成績表が見つかりません: {race_id}")
    cols = _header_map(table, _RESULT_HEADERS)
    if "horse" not in cols.values():
        raise ParseError(f"成績表の見出しが想定外です: {race_id}")

    results = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if not tds:
            continue
        cell = {name: tds[i] for i, name in cols.items() if i < len(tds)}
        horse_a = cell["horse"].find("a")
        horse_id = id_from_href(horse_a.get("href") if horse_a else None, "horse")
        if horse_id is None:
            continue
        row: dict[str, Any] = {"race_id": race_id, "horse_id": horse_id}
        row["horse_name"] = cell_text(horse_a)
        row["finish_pos"], row["finish_status"] = parse_finish(cell_text(cell.get("finish")))
        row["frame"] = to_int(cell_text(cell.get("frame")))
        row["number"] = to_int(cell_text(cell.get("number")))
        row["sex"], row["age"] = parse_sex_age(cell_text(cell.get("sex_age")))
        row["weight_carried"] = to_float(cell_text(cell.get("weight_carried")))
        jockey_a = cell["jockey"].find("a") if "jockey" in cell else None
        row["jockey_id"] = id_from_href(jockey_a.get("href") if jockey_a else None, "jockey")
        row["jockey_name"] = cell_text(jockey_a) if jockey_a else cell_text(cell.get("jockey")) or None
        row["time_sec"] = parse_time(cell_text(cell.get("time")))
        row["margin"] = cell_text(cell.get("margin")) or None
        row["passing"] = cell_text(cell.get("passing")) or None
        row["last3f"] = to_float(cell_text(cell.get("last3f")))
        row["odds"] = to_float(cell_text(cell.get("odds")))
        row["popularity"] = to_int(cell_text(cell.get("popularity")))
        row["horse_weight"], row["weight_diff"] = parse_horse_weight(cell_text(cell.get("horse_weight")))
        trainer_cell = cell.get("trainer")
        trainer_a = trainer_cell.find("a") if trainer_cell else None
        row["trainer_id"] = id_from_href(trainer_a.get("href") if trainer_a else None, "trainer")
        row["trainer_name"] = cell_text(trainer_a) if trainer_a else None
        m = re.search(r"\[(東|西|地|外)\]", cell_text(trainer_cell))
        row["trainer_region"] = _REGION.get(m.group(1)) if m else None
        row["owner"] = cell_text(cell.get("owner")) or None
        row["prize"] = to_float(cell_text(cell.get("prize")))
        results.append(row)

    if not results:
        raise ParseError(f"出走馬が0頭です: {race_id}")
    race["field_size"] = sum(1 for r in results if r["finish_status"] not in ("取消", "除外"))
    return {"race": race, "results": results, "payouts": parse_payouts(soup, race_id)}


def normalize_bet_type(text: str) -> str | None:
    text = norm(text).replace("3連", "三連").replace(" ", "")
    return text if text in ("単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単") else None


def normalize_combination(bet_type: str, text: str) -> str:
    nums = [int(n) for n in re.findall(r"\d+", text)]
    if bet_type in UNORDERED_BETS:
        nums.sort()
    return "-".join(str(n) for n in nums)


def _split_cell(td: Tag) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    for node in td.descendants:
        if isinstance(node, Tag) and node.name == "br":
            parts.append(norm("".join(buf)))
            buf = []
        elif isinstance(node, str):
            buf.append(node)
    parts.append(norm("".join(buf)))
    return [p for p in parts if p]


def parse_payouts(soup: BeautifulSoup, race_id: str) -> list[dict[str, Any]]:
    tables = soup.select("table.pay_table_01") or soup.select("table.Payout_Detail_Table")
    rows = []
    for table in tables:
        for tr in table.find_all("tr"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if th is None or len(tds) < 2:
                continue
            bet_type = normalize_bet_type(th.get_text())
            if bet_type is None:
                continue
            combos = _split_cell(tds[0])
            pays = _split_cell(tds[1])
            pops = _split_cell(tds[2]) if len(tds) > 2 else []
            if bet_type in ("単勝", "複勝") and len(combos) == 1 and len(pays) > 1:
                combos = re.findall(r"\d+", combos[0])
            for i, combo in enumerate(combos):
                if i >= len(pays):
                    break
                rows.append({
                    "race_id": race_id,
                    "bet_type": bet_type,
                    "combination": normalize_combination(bet_type, combo),
                    "payout": to_int(pays[i]),
                    "popularity": to_int(pops[i]) if i < len(pops) else None,
                })
    return rows


# ---------------------------------------------------------------- 出馬表 (race.netkeiba.com/race/shutuba.html)

_GRADE_ICON = {"1": "G1", "2": "G2", "3": "G3", "5": "OP", "10": "G1", "11": "G2", "12": "G3", "15": "L"}


def parse_shutuba(html: str, race_id: str, date: str | None = None) -> dict[str, Any]:
    """出馬表ページ → {'race': {...}, 'entries': [...]}"""
    soup = _soup(html)
    race: dict[str, Any] = {"race_id": race_id, **_race_id_parts(race_id), "source": "netkeiba"}

    name_el = soup.select_one(".RaceName")
    race["race_name"] = cell_text(name_el) or None
    grade = None
    icon = name_el.select_one("[class*=Icon_GradeType]") if name_el else None
    if icon is not None:
        for cls in icon.get("class", []):
            m = re.fullmatch(r"Icon_GradeType(\d+)", cls)
            if m:
                grade = _GRADE_ICON.get(m.group(1))
    race["grade"] = grade or normalize_grade(race["race_name"])

    data1 = cell_text(soup.select_one(".RaceData01"))
    m = re.search(r"(障|芝|ダ)\D{0,3}?(\d{3,4})\s*m", data1)
    race["surface"] = m.group(1) if m else None
    race["distance"] = int(m.group(2)) if m else None
    m = re.search(r"\(\s*(右|左|直線|直)\s*(外|内)?", data1)
    race["direction"] = ("直線" if m.group(1).startswith("直") else m.group(1)) if m else None
    race["course_detail"] = m.group(2) if m else None
    m = re.search(r"天候\s*:\s*([^\s/]+)", data1)
    race["weather"] = m.group(1) if m else None
    m = re.search(r"馬場\s*:\s*(良|稍重|重|不良)", data1)
    race["going"] = m.group(1) if m else None
    m = re.search(r"(\d{1,2}:\d{2})\s*発走", data1)
    race["start_time"] = m.group(1) if m else None

    data2_el = soup.select_one(".RaceData02")
    spans = [cell_text(s) for s in data2_el.find_all("span")] if data2_el else []
    data2 = " ".join(spans)
    race["conditions"] = " ".join(s for s in spans[3:7]) if len(spans) >= 7 else data2
    race["race_class"], race["class_level"] = detect_class(data2, race["grade"])

    race["date"] = date or _parse_date(cell_text(soup.find("title")))
    if race["date"] is None:
        active = soup.select_one("dd.Active a[href*=kaisai_date], .RaceList_DateList .Active a")
        race["date"] = _parse_date(active.get("href", "")) if active else None
    if race["date"] is None:
        raise ParseError(f"開催日が読み取れません: {race_id} (date を指定してください)")

    entries = []
    for tr in soup.select("tr.HorseList"):
        info = tr.select_one("td.HorseInfo")
        horse_a = info.find("a") if info else None
        horse_id = id_from_href(horse_a.get("href") if horse_a else None, "horse")
        if horse_id is None:
            continue
        row: dict[str, Any] = {"race_id": race_id, "horse_id": horse_id}
        row["horse_name"] = norm(horse_a.get("title")) or cell_text(horse_a)
        waku = tr.select_one("td[class*=Waku]")
        umaban = tr.select_one("td[class*=Umaban]")
        row["frame"] = to_int(cell_text(waku))
        row["number"] = to_int(cell_text(umaban))
        barei = tr.select_one("td.Barei")
        row["sex"], row["age"] = parse_sex_age(cell_text(barei))
        kinryo = barei.find_next_sibling("td") if barei else None
        row["weight_carried"] = to_float(cell_text(kinryo))
        jockey_a = tr.select_one("td.Jockey a")
        row["jockey_id"] = id_from_href(jockey_a.get("href") if jockey_a else None, "jockey")
        row["jockey_name"] = (norm(jockey_a.get("title")) or cell_text(jockey_a)) if jockey_a else None
        trainer_td = tr.select_one("td.Trainer")
        trainer_a = trainer_td.find("a") if trainer_td else None
        row["trainer_id"] = id_from_href(trainer_a.get("href") if trainer_a else None, "trainer")
        row["trainer_name"] = (norm(trainer_a.get("title")) or cell_text(trainer_a)) if trainer_a else None
        region = cell_text(trainer_td.find("span")) if trainer_td else ""
        row["trainer_region"] = region if region in ("美浦", "栗東", "地方", "海外") else None
        row["horse_weight"], row["weight_diff"] = parse_horse_weight(cell_text(tr.select_one("td.Weight")))
        tr_text = cell_text(tr)
        row["scratched"] = int("Cancel" in (tr.get("class") or []) or "取消" in tr_text or "除外" in tr_text)
        entries.append(row)

    if not entries:
        raise ParseError(f"出馬表に馬がいません: {race_id}")
    race["field_size"] = sum(1 for e in entries if not e["scratched"])
    return {"race": race, "entries": entries}


# ---------------------------------------------------------------- オッズAPI

def _odds_value(value: Any) -> float | None:
    v = to_float(str(value)) if value not in (None, "") else None
    return v if v and v > 0 else None


def parse_odds_json(text: str, race_id: str) -> dict[str, Any]:
    """api_get_jra_odds.html (type=1) のJSON → 単勝・複勝オッズ."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"オッズJSONが読めません: {race_id}") from exc
    data = payload.get("data") or {}
    odds = data.get("odds") or {}
    win = odds.get("1") or {}
    place = odds.get("2") or {}
    rows = []
    for key in sorted(set(win) | set(place), key=lambda k: int(k)):
        w = win.get(key) or []
        p = place.get(key) or []
        rows.append({
            "race_id": race_id,
            "number": int(key),
            "win_odds": _odds_value(w[0]) if len(w) > 0 else None,
            "place_odds_min": _odds_value(p[0]) if len(p) > 0 else None,
            "place_odds_max": _odds_value(p[1]) if len(p) > 1 else None,
            "popularity": to_int(str(w[2])) if len(w) > 2 and w[2] else None,
        })
    return {"status": payload.get("status"), "official_datetime": data.get("official_datetime"), "odds": rows}


# ---------------------------------------------------------------- 血統 (db.netkeiba.com/horse/ped/{id}/)

def parse_pedigree(html: str, horse_id: str) -> dict[str, Any]:
    """血統表から父・母・母父を取り出す (rowspan の大きいセルが父・母)."""
    soup = _soup(html)
    table = soup.select_one("table.blood_table")
    if table is None:
        raise ParseError(f"血統表が見つかりません: {horse_id}")
    cells = table.find_all("td")
    spans = [int(td.get("rowspan", 1)) for td in cells]
    top = max(spans)
    parents = [td for td, s in zip(cells, spans) if s == top]
    if len(parents) < 2:
        raise ParseError(f"血統表の形式が想定外です: {horse_id}")

    def name_id(td: Tag | None) -> tuple[str | None, str | None]:
        if td is None:
            return None, None
        a = td.find("a", href=re.compile(r"/horse/"))
        if a is None:
            return (cell_text(td).split(" ")[0] or None), None
        return cell_text(a) or None, id_from_href(a.get("href"), "horse")

    sire, sire_id = name_id(parents[0])
    dam, dam_id = name_id(parents[1])
    damsire, damsire_id = name_id(parents[1].find_next_sibling("td"))
    title = soup.select_one(".horse_title h1") or soup.find("h1")
    horse_name = cell_text(title).split(" ")[0] if title else None
    return {
        "horse_id": horse_id,
        "horse_name": horse_name or None,
        "sire": sire,
        "sire_id": sire_id,
        "dam": dam,
        "dam_id": dam_id,
        "damsire": damsire,
        "damsire_id": damsire_id,
    }
