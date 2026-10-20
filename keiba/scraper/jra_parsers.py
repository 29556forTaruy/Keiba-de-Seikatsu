"""JRA 公式サイト (www.jra.go.jp/JRADB) のHTMLを辞書に変換する純粋関数群.

netkeiba で取れないときの予備。parse_result は parsers.parse_race_result と同じ形を返す。
ID は netkeiba と同じ体系に直す:
  - race_id: 年4桁 + 場2桁 + 回2桁 + 日目2桁 + R2桁 (JRA のページ指定 CNAME から組み立てる)
  - 馬: 血統登録番号 (netkeiba の horse_id と同じ)
  - 騎手・調教師: JRA の4桁を5桁にゼロ埋め (netkeiba の jockey_id / trainer_id と同じ)

JRA のページは CNAME (例: pw01sde1005202404081120241027/EF) で指定し、末尾の2桁は
サイトが付ける検査値。自分で計算せず、ページ内のリンクに書かれた CNAME をたどって使う。
"""

from __future__ import annotations

import re
from typing import Any

from bs4 import BeautifulSoup, Tag

from keiba.scraper.parsers import (
    ParseError,
    _parse_date,
    _race_id_parts,
    cell_text,
    detect_class,
    norm,
    normalize_bet_type,
    normalize_combination,
    normalize_grade,
    parse_finish,
    parse_horse_weight,
    parse_sex_age,
    parse_time,
    to_float,
    to_int,
)

# 過去レース結果検索ページ (月ごとの検査値の表を持つ). トップページのリンクと同じ固定の CNAME
SEARCH_CNAME = "pw01skl00999999/B3"

_CS = r"/[0-9A-F]{2}"
_OBJ_PARAM = re.compile(r'objParam\["(\d{4})"\]\s*=\s*"([0-9A-F]{2})"')
_YEAR_MONTH = re.compile(r'var\s+yearMonth\s*=\s*"(\d{6})"')
# 開催日 (場・回・日目) のレース一覧: pw01srl10 + 場 + 年 + 回 + 日目 + 日付
_DAY_CNAME = re.compile(r"pw01srl10(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})" + _CS)
# 1レースの結果: pw01sde10 + 場 + 年 + 回 + 日目 + R + 日付
_RACE_CNAME = re.compile(r"pw01sde10(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})" + _CS)
# 結果ページから飛べる確定オッズ (単勝・複勝)
_ODDS_CNAME = re.compile(r"pw151ou10\d{20}Z" + _CS)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


def _check_error_page(html: str) -> None:
    title = re.search(r"<title>([^<]*)", html)
    if title and "エラー" in title.group(1):
        raise ParseError(f"JRAのエラーページです: {norm(title.group(1))}")


def _iso(yyyymmdd: str) -> str:
    return f"{yyyymmdd[:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:]}"


def make_race_id(venue_code: str, year: str, kai: str, nichi: str, race_num: str) -> str:
    return f"{year}{venue_code}{kai}{nichi}{race_num}"


# ---------------------------------------------------------------- 検索ページ・開催日・レース一覧

def parse_month_index(html: str) -> dict[str, Any]:
    """過去レース結果検索ページ → {'checksums': {'2410': '92', ...}, 'year_month': '202610'}."""
    _check_error_page(html)
    checksums = dict(_OBJ_PARAM.findall(html))
    if not checksums:
        raise ParseError("JRAの過去レース結果検索ページから月の一覧が読めません")
    m = _YEAR_MONTH.search(html)
    return {"checksums": checksums, "year_month": m.group(1) if m else None}


def month_cname(index: dict[str, Any], year: int, month: int) -> str | None:
    """月の開催一覧の CNAME. 検索ページの setParameter() と同じ組み立て (当月以降は接頭辞が変わる)."""
    ym = f"{year:04d}{month:02d}"
    cs = index["checksums"].get(ym[2:])
    if cs is None:
        return None
    current = index.get("year_month")
    prefix = "pw01skl00" if current and ym >= current else "pw01skl10"
    return f"{prefix}{ym}/{cs}"


def extract_days(html: str) -> list[dict[str, Any]]:
    """月の開催一覧 → 開催日 (場・回・日目) ごとの CNAME. 出現順、重複なし."""
    _check_error_page(html)
    out: dict[str, dict[str, Any]] = {}
    for m in _DAY_CNAME.finditer(html):
        venue, year, kai, nichi, ymd = m.groups()
        out.setdefault(m.group(0), {
            "cname": m.group(0), "venue_code": venue, "year": year, "kai": kai, "nichi": nichi, "date": _iso(ymd),
        })
    return list(out.values())


def extract_races(html: str) -> list[dict[str, Any]]:
    """開催日のレース一覧 (など) → レースごとの CNAME と netkeiba 形式の race_id."""
    _check_error_page(html)
    out: dict[str, dict[str, Any]] = {}
    for m in _RACE_CNAME.finditer(html):
        venue, year, kai, nichi, rnum, ymd = m.groups()
        race_id = make_race_id(venue, year, kai, nichi, rnum)
        out.setdefault(race_id, {"cname": m.group(0), "race_id": race_id, "date": _iso(ymd)})
    return list(out.values())


# ---------------------------------------------------------------- レース結果

_SEX = {"せん": "セ", "騸": "セ"}
# 馬場の欄の class 名 (JRA のページはダートを durt と書く)
_BABA_CLASS = {"turf": "芝", "dirt": "ダ", "durt": "ダ"}
# 見習騎手などの印 (netkeiba は付けない)
_JOCKEY_MARKS = re.compile(r"^[☆▲△◇★]+")


def _ids_in(el: Tag | None, pattern: str) -> str | None:
    if el is None:
        return None
    for a in el.find_all("a"):
        m = re.search(pattern, (a.get("href") or "") + (a.get("onclick") or ""))
        if m:
            return m.group(1)
    return None


_PERSON_CNAME = {"jockey": r"pw04kmk\d\d(\d{4})/", "trainer": r"pw05cmk\d\d(\d{4})/"}


def _person_id(el: Tag | None, kind: str) -> str | None:
    """騎手 pw04kmk00XXXX / 調教師 pw05cmk00XXXX → netkeiba と同じ5桁."""
    code = _ids_in(el, _PERSON_CNAME[kind])
    return code.zfill(5) if code else None


def _person_name(el: Tag | None) -> str | None:
    return _JOCKEY_MARKS.sub("", cell_text(el).replace(" ", "")) or None


def normalize_margin(text: str | None) -> str | None:
    """JRA の着差を netkeiba の表記にそろえる. '１ 1/4' → '1.1/4', '大差' → '大'."""
    text = norm(text)
    if not text:
        return None
    text = re.sub(r"^(\d+)\s+(\d/\d)$", r"\1.\2", text)
    return "大" if text == "大差" else text


# netkeiba の表記にそろえる
_DETAIL = {"外内": "外-内", "内外": "内-外", "外→内": "外-内", "内→外": "内-外"}


def _course(el: Tag | None, jump: bool) -> dict[str, Any]:
    """'コース： 2,000 メートル （芝・左）' / '（ダート・右）' / '（芝・左 外）' / '（芝・直）' / 障害の '（芝 外内）'.

    障害はコースの欄に書かれないので、条件 (「障害3歳以上」など) から jump で受け取る。
    """
    text = cell_text(el)
    out: dict[str, Any] = {"surface": None, "direction": None, "distance": None, "course_detail": None}
    m = re.search(r"([\d,]{3,6})\s*メートル", text)
    out["distance"] = to_int(m.group(1)) if m else None
    detail = re.search(r"\(([^)]*)\)", text)
    parts = [p for p in re.split(r"[・･\s]+", detail.group(1)) if p] if detail else []
    if jump or any(p.startswith("障害") for p in parts):
        out["surface"] = "障"
    elif any(p.startswith("ダート") for p in parts):
        out["surface"] = "ダ"
    elif "芝" in parts:
        out["surface"] = "芝"
    for p in parts:
        p = _DETAIL.get(p, p)
        if p in ("右", "左", "直線", "直"):
            # 障害は netkeiba でも回りを持たない
            out["direction"] = None if out["surface"] == "障" else ("直線" if p == "直" else p)
        elif re.fullmatch(r"(外-内|内-外|内2周|外2周|2周|外|内)", p):
            out["course_detail"] = p
    return out


def _prizes(soup: BeautifulSoup) -> list[float]:
    """本賞金 + 付加賞 (万円) を着順ごとに足したもの. netkeiba の賞金列と同じ値になる."""
    total: list[float] = []
    for unit in soup.select("ul.prize div.prize_unit"):
        for i, li in enumerate(unit.select("div.data li")):
            value = to_float(cell_text(li.select_one("span.num")))
            if value is None:
                continue
            while len(total) <= i:
                total.append(0.0)
            total[i] = round(total[i] + value, 1)
    return total


def _payouts(soup: BeautifulSoup, race_id: str) -> list[dict[str, Any]]:
    rows = []
    for dl in soup.select("div.refund_area dl"):
        bet_type = normalize_bet_type(cell_text(dl.find("dt")))
        if bet_type is None:
            continue
        for line in dl.select("div.line"):
            combo = cell_text(line.select_one("div.num"))
            pay = to_int(cell_text(line.select_one("div.yen")))
            if not combo or pay is None:
                continue
            rows.append({
                "race_id": race_id,
                "bet_type": bet_type,
                "combination": normalize_combination(bet_type, combo),
                "payout": pay,
                "popularity": to_int(cell_text(line.select_one("div.pop"))),
            })
    return rows


def parse_result(html: str, race_id: str) -> dict[str, Any]:
    """JRA のレース結果ページ → {'race', 'results', 'payouts', 'odds_cname'}.

    結果表に単勝オッズは無いので odds は None のまま。odds_cname の確定オッズページで埋める。
    馬主・調教師の所属 (美浦/栗東) も JRA のページには無いので None。
    短期免許の外国人騎手は JRA のページにリンクが無いので jockey_id が None になる。
    """
    _check_error_page(html)
    soup = _soup(html)
    table = None
    for t in soup.select("table.basic"):
        heads = {norm(th.get_text()).replace(" ", "") for th in t.find_all("th")}
        if {"着順", "馬名"} <= heads:
            table = t
            break
    if table is None:
        raise ParseError(f"JRAの成績表が見つかりません: {race_id}")

    race: dict[str, Any] = {"race_id": race_id, **_race_id_parts(race_id), "source": "jra"}
    race["date"] = _parse_date(cell_text(soup.select_one("div.date_line div.cell.date")))
    if race["date"] is None:
        raise ParseError(f"JRAの結果ページの開催日が読み取れません: {race_id}")
    m = re.search(r"(\d{1,2})時(\d{2})分", cell_text(soup.select_one("div.date_line div.cell.time")))
    race["start_time"] = f"{int(m.group(1)):02d}:{m.group(2)}" if m else None

    title = soup.select_one("div.race_title")
    name_el = title.select_one("span.race_name") if title else None
    grade_img = name_el.select_one(".grade_icon img") if name_el else None
    alt = norm(grade_img.get("alt")) if grade_img else ""
    race["grade"] = "L" if "リステッド" in alt else normalize_grade(f"({alt})") if alt else None
    if grade_img is not None:
        grade_img.parent.extract()
    cap = cell_text(title.select_one("span.main span.cap")) if title else ""
    race["race_name"] = (cap + cell_text(name_el)).replace(" ", "") or None

    cells = [cell_text(title.select_one(f"div.cell.{c}")) for c in ("category", "class", "rule", "weight")] if title else []
    race["conditions"] = "".join(cells)
    race.update(_course(title.select_one("div.cell.course") if title else None, jump="障害" in race["conditions"]))
    race["race_class"], race["class_level"] = detect_class(race["conditions"], race["grade"])
    if race["grade"] is None and race["race_class"] == "オープン":
        # JRA は重賞・リステッド以外のオープンに印を付けないが、netkeiba は (OP) とする
        race["grade"] = "OP"

    baba = soup.select_one("div.date_line div.cell.baba")
    weather = baba.select_one("li.weather span.txt") if baba else None
    race["weather"] = cell_text(weather) or None
    goings = {}
    for li in (baba.select("li") if baba else []):
        surf = next((_BABA_CLASS[c] for c in li.get("class", []) if c in _BABA_CLASS), None)
        if surf:
            goings[surf] = cell_text(li.select_one("span.txt")) or None
    # 障害は netkeiba と同じく芝の馬場を使う
    race["going"] = goings.get("芝" if race["surface"] in ("芝", "障") else "ダ") or next(iter(goings.values()), None)

    prizes = _prizes(soup)
    results = []
    for tr in table.select("tbody tr"):
        horse = tr.select_one("td.horse")
        horse_id = _ids_in(horse, r"pw01dud\d\d(\w{10})/")
        if horse_id is None:
            continue
        row: dict[str, Any] = {"race_id": race_id, "horse_id": horse_id, "horse_name": cell_text(horse) or None}
        row["finish_pos"], row["finish_status"] = parse_finish(cell_text(tr.select_one("td.place")))
        waku = tr.select_one("td.waku img")
        row["frame"] = to_int(norm(waku.get("alt"))) if waku else to_int(cell_text(tr.select_one("td.waku")))
        row["number"] = to_int(cell_text(tr.select_one("td.num")))
        sex_age = cell_text(tr.select_one("td.age"))
        for k, v in _SEX.items():
            sex_age = sex_age.replace(k, v)
        row["sex"], row["age"] = parse_sex_age(sex_age)
        row["weight_carried"] = to_float(cell_text(tr.select_one("td.weight")))
        jockey = tr.select_one("td.jockey")
        row["jockey_id"], row["jockey_name"] = _person_id(jockey, "jockey"), _person_name(jockey)
        row["time_sec"] = parse_time(cell_text(tr.select_one("td.time")))
        row["margin"] = normalize_margin(cell_text(tr.select_one("td.margin")))
        corners = re.findall(r"\d+", cell_text(tr.select_one("td.corner")))
        row["passing"] = "-".join(corners) or None
        row["last3f"] = to_float(cell_text(tr.select_one("td.f_time")))
        row["odds"] = None
        row["popularity"] = to_int(cell_text(tr.select_one("td.pop")))
        weight_text = cell_text(tr.select_one("td.h_weight")).replace(" ", "")
        row["horse_weight"], row["weight_diff"] = parse_horse_weight(weight_text)
        if row["horse_weight"] is not None and row["weight_diff"] is None and (
                "(" not in weight_text or "初出走" in weight_text):
            # JRA は増減0を書かず、初出走は '(初出走)' と書く (netkeiba はどちらも '492(0)')
            row["weight_diff"] = 0
        trainer = tr.select_one("td.trainer")
        row["trainer_id"], row["trainer_name"] = _person_id(trainer, "trainer"), _person_name(trainer)
        row["trainer_region"] = None
        row["owner"] = None
        pos = row["finish_pos"]
        row["prize"] = prizes[pos - 1] if pos and pos <= len(prizes) and row["finish_status"] != "降着" else None
        results.append(row)

    if not results:
        raise ParseError(f"JRAの成績表に馬がいません: {race_id}")
    race["field_size"] = sum(1 for r in results if r["finish_status"] not in ("取消", "除外"))
    payouts = _payouts(soup, race_id)
    if any(r["finish_pos"] == 1 for r in results) and not any(p["bet_type"] == "単勝" for p in payouts):
        raise ParseError(f"JRAの結果ページから単勝の払戻が読み取れません: {race_id}")
    m = _ODDS_CNAME.search(html)
    return {"race": race, "results": results, "payouts": payouts, "odds_cname": m.group(0) if m else None}


def parse_win_odds(html: str) -> dict[int, float]:
    """単勝・複勝オッズ (馬番順) ページ → {馬番: 単勝オッズ}. 取消などオッズの無い馬は含めない."""
    _check_error_page(html)
    soup = _soup(html)
    table = soup.select_one("table.tanpuku")
    if table is None:
        raise ParseError("JRAの単勝オッズ表が見つかりません")
    odds = {}
    for tr in table.select("tbody tr"):
        number = to_int(cell_text(tr.select_one("td.num")))
        value = cell_text(tr.select_one("td.odds_tan"))
        if number is not None and re.fullmatch(r"\d+(\.\d+)?", value):
            odds[number] = float(value)
    if not odds:
        raise ParseError("JRAの単勝オッズが読めません")
    return odds
