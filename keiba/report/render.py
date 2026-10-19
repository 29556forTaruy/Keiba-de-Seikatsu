"""HTMLレポートの組み立てと書き出し."""

from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader, select_autoescape

from keiba import __version__
from keiba.betting import suggest_bets, top_combos
from keiba.constants import MARKS
from keiba.report import stats
from keiba.report.svg import CHART_JS, line_chart, sparkline

WEEKDAYS = "月火水木金土日"
SURFACE_LABEL = {"芝": "芝", "ダ": "ダート", "障": "障害"}


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and not math.isfinite(v))


def _pct(v: Any) -> str:
    return "-" if _missing(v) else f"{float(v) * 100:.1f}%"


def _pct0(v: Any) -> str:
    return "-" if _missing(v) else f"{float(v) * 100:.0f}%"


def _odds(v: Any) -> str:
    if _missing(v):
        return "-"
    v = float(v)
    return f"{v:,.0f}" if v >= 1000 else f"{v:.1f}"


def _yen(v: Any) -> str:
    return "-" if _missing(v) else f"{float(v):,.0f}円"


def _signed_yen(v: Any) -> str:
    return "-" if _missing(v) else f"{float(v):+,.0f}円"


def _num(fmt: str):
    return lambda v: "" if _missing(v) else format(float(v), fmt)


def _safe_url(v: Any) -> str:
    s = str(v or "")
    return s if s.startswith(("https://", "http://")) else "#"


def _stamp(v: Any) -> str:
    """DBの作成日時 (2024-03-02T09:15:30) → 2024-03-02 09:15"""
    if _missing(v) or not str(v):
        return ""
    try:
        return pd.Timestamp(str(v)).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(v)


def _blank(v: Any) -> Any:
    # 値のない欄 (None / NaN) を「None」「nan」と出さない
    return "" if _missing(v) else v


def environment() -> Environment:
    env = Environment(
        loader=PackageLoader("keiba.report", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
        finalize=_blank,
    )
    env.filters.update(
        pct=_pct, pct0=_pct0, odds=_odds, yen=_yen, signed_yen=_signed_yen,
        num0=_num(".0f"), num1=_num(".1f"), safe_url=_safe_url, stamp=_stamp,
    )
    env.globals.update(app_version=__version__, chart_js=CHART_JS)
    return env


def date_label(date: str | pd.Timestamp) -> str:
    d = pd.Timestamp(date)
    return f"{d.year}年{d.month}月{d.day}日（{WEEKDAYS[d.weekday()]}）"


def _write(path: Path, html: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


def informative(pred: pd.DataFrame) -> bool:
    """勝率に意味があるか. モデル未学習かつオッズ未取得だと全馬均等 (1/頭数) で、
    その並びは馬番順にすぎないので印・本命・組み合わせに使わない."""
    p = pred["p_win"].dropna()
    if p.empty or p.max() - p.min() < 1e-12:
        return False
    sources = [c for c in ("p_model", "p_market") if c in pred.columns]
    return not sources or any(pred[c].notna().any() for c in sources)


def default_marks(pred: pd.DataFrame, min_ev: float) -> dict[int, str]:
    """予想勝率の順に ◎○▲△△、それ以外で期待値が高い馬に ☆ (馬番確定前・勝率が均等なときは印なし)."""
    if not informative(pred):
        return {}
    pred = pred[pred["number"].notna() & (pred["p_win"] > 0)]  # 勝率0 (取消扱い) の馬に印は付けない
    order = pred.sort_values("p_win", ascending=False)["number"].astype(int).tolist()
    marks = {}
    for i, num in enumerate(order[:5]):
        marks[num] = ["◎", "○", "▲", "△", "△"][i]
    ev = (pred["p_win"] * pred["win_odds"]).fillna(0)
    rest = pred[~pred["number"].astype(int).isin(marks) & (ev >= min_ev)]
    if not rest.empty:
        marks[int(rest.loc[(rest["p_win"] * rest["win_odds"]).idxmax(), "number"])] = "☆"
    return marks


def race_context(
    race: dict[str, Any],
    feat: pd.DataFrame,
    pred: pd.DataFrame,
    by_horse: dict[str, pd.DataFrame],
    top2: dict[str, list[tuple[str, str]]],
    *,
    betting: dict[str, Any],
    model: Any = None,
    analysis: dict[str, Any] | None = None,
    review: dict[str, Any] | None = None,
    news: dict[str, Any] | None = None,
    index_href: str | None = None,
) -> dict[str, Any]:
    race_id = race["race_id"]
    rows = feat[feat["race_id"] == race_id].sort_values("number")
    date = pd.Timestamp(race["date"])
    race = {**race, "date_label": date_label(date), "surface_label": SURFACE_LABEL.get(race.get("surface"), race.get("surface") or "")}
    pred = pred.set_index("horse_id")
    min_ev = betting.get("min_ev", 1.15)

    marks = default_marks(pred.reset_index(), min_ev)
    if analysis and analysis.get("content", {}).get("marks"):
        marks = {int(m["number"]): m["mark"] for m in analysis["content"]["marks"] if m.get("mark") in MARKS}

    horses = []
    for _, r in rows.iterrows():
        p = pred.loc[r["horse_id"]] if r["horse_id"] in pred.index else None
        hist = stats.history_before(by_horse, r["horse_id"], date)
        runs = stats.past_runs(hist, top2) if not hist.empty else []
        styles = [x["style"] for x in runs if x["style"]]
        figs = [x["speed_fig"] for x in reversed(runs)]
        win_odds = stats._clean(r.get("win_odds"))
        p_win = stats._clean(p["p_win"]) if p is not None else None
        horses.append({
            "number": stats._int(r["number"]),
            "frame": stats._int(r["frame"]),
            "horse_id": r["horse_id"],
            "anchor": f"h{stats._int(r['number'])}" if not _missing(r["number"]) else f"id{r['horse_id']}",
            "horse_name": stats._clean(r["horse_name"]),
            "sex": stats._clean(r["sex"]),
            "age": stats._int(r["age"]),
            "sex_age": stats._sex_age(r["sex"], r["age"]),
            "weight_carried": stats._clean(r["weight_carried"]),
            "jockey_name": stats._clean(r["jockey_name"]),
            "trainer_name": stats._clean(r.get("trainer_name")),
            "trainer_region": stats._clean(r.get("trainer_region")),
            "horse_weight": stats._int(r.get("horse_weight")),
            "weight_diff": stats._int(r.get("weight_diff")),
            "sire": stats._clean(r.get("sire")),
            "damsire": stats._clean(r.get("damsire")),
            "win_odds": win_odds,
            "place_odds_min": stats._clean(r.get("place_odds_min")),
            "popularity": stats._int(r.get("popularity")),
            "p_win": p_win,
            "p_place": stats._clean(p["p_place"]) if p is not None else None,
            "p_market": stats._clean(r.get("p_market")),
            "ev": p_win * win_odds if p_win is not None and win_odds else None,
            "mark": marks.get(int(r["number"]), "") if not _missing(r["number"]) else "",
            "past_runs": runs,
            "aptitude": stats.aptitude(hist, race) if not hist.empty else [],
            "style": Counter(styles).most_common(1)[0][0] if styles else None,
            "days_since": stats._int(r.get("h_days_since")),
            "j_n": stats._int(r.get("j_n")),
            "j_win": stats._clean(r.get("j_win")),
            "j_top3": stats._clean(r.get("j_top3")),
            "spark": sparkline(figs),
        })

    numbered = all(h["number"] is not None for h in horses)
    has_signal = informative(pred)
    has_odds = any(h["win_odds"] for h in horses)
    # 一部の馬だけオッズがない (出走取消を出馬表に反映する前など). その馬は推奨馬券の対象外
    no_odds = [h["number"] or h["horse_name"] for h in horses if not h["win_odds"]] if has_odds else []
    # 馬番が確定する前 (枠順発表前) は馬券の組番を作れない
    bets = suggest_bets(horses, **betting) if numbered else []
    probs = [h["p_win"] or 0 for h in horses]
    numbers = [h["number"] for h in horses]
    combos = {}
    if numbered and has_signal and len(horses) >= 3 and sum(probs) > 0:
        combos = {bt: top_combos(probs, numbers, bt, k=5, min_ev=min_ev) for bt in ("馬連", "ワイド", "三連複")}
    return {
        "race": race,
        "horses": horses,
        # 勝率が均等なときは並べ替えず馬番順のまま
        "horses_by_prob": sorted(horses, key=lambda h: -(h["p_win"] or 0)) if has_signal else horses,
        "max_p": max([h["p_win"] or 0 for h in horses] + [1e-9]),
        "names": {h["number"]: h["horse_name"] for h in horses},
        "has_odds": has_odds,
        "no_odds": no_odds,
        "informative": has_signal,
        "numbered": numbered,
        "bets": bets,
        "combos": combos,
        "min_ev": min_ev,
        "bankroll": betting.get("bankroll"),
        "trends": stats.course_trends(feat, race, date),
        "same_race": stats.same_race_history(feat, race, date),
        "analysis": analysis,
        "review": review,
        "news": news,
        "model": model,
        "index_href": index_href,
    }


def render_race(ctx: dict[str, Any], path: Path) -> Path:
    html = environment().get_template("race.html.j2").render(**ctx, generated_at=_now())
    return _write(path, html)


def render_index(date: str, items: list[dict[str, Any]], path: Path, *, is_demo: bool = False) -> Path:
    by_venue: dict[str, list[dict[str, Any]]] = {}
    for it in sorted(items, key=lambda x: (x["venue"], x["race_num"])):
        by_venue.setdefault(it["venue"], []).append(it)
    html = environment().get_template("index.html.j2").render(
        date_label=date_label(date), races=items, by_venue=by_venue, is_demo=is_demo,
        total_stake=sum(it.get("stake", 0) for it in items), generated_at=_now(),
    )
    return _write(path, html)


def render_course(race: dict[str, Any], feat: pd.DataFrame, path: Path, *, is_demo: bool = False) -> Path:
    race = {**race, "surface_label": SURFACE_LABEL.get(race["surface"], race["surface"])}
    trends = stats.course_trends(feat, race, pd.Timestamp.today() + pd.Timedelta(days=1))
    html = environment().get_template("course.html.j2").render(race=race, trends=trends, is_demo=is_demo, generated_at=_now())
    return _write(path, html)


def render_backtest(bt: dict[str, Any], path: Path, *, test_metrics: dict[str, Any] | None = None, is_demo: bool = False) -> Path:
    flat = [s for s in bt["strategies"] if s["name"] != "ケリー資金管理"]
    flat_series = [{"name": s["name"], "values": s["curve"], "slot": i + 1} for i, s in enumerate(flat[:4])]
    monthly = {s["name"]: {m["month"]: m for m in s["monthly"]} for s in bt["strategies"]}
    months = sorted({m for s in monthly.values() for m in s})
    html = environment().get_template("backtest.html.j2").render(
        bt=bt,
        flat_series=flat_series,
        flat_chart=line_chart(flat_series, chart_id="flat", aria_label="累積収支の推移（100円均一の戦略）"),
        # equity[0] は最初のレース前の初期資金. 資金は損益ではないので符号を付けず、縦軸に 0 も入れない
        equity_chart=line_chart(
            [{"name": "資金", "values": bt["equity"], "slot": 1}], chart_id="equity", y_format="yen_abs",
            include_zero=False, aria_label="ケリー資金管理の資金推移",
            tip_labels=["開始"] + [f"{i}レース後" for i in range(1, len(bt["equity"]))],
        ),
        monthly=monthly,
        months=months,
        test_metrics=test_metrics,
        is_demo=is_demo,
        generated_at=_now(),
    )
    return _write(path, html)


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def summarize_for_index(ctx: dict[str, Any], href: str) -> dict[str, Any]:
    race = ctx["race"]
    # 勝率が均等 (モデルもオッズもない) なら本命は空欄
    top = ctx["horses_by_prob"][0] if ctx["horses_by_prob"] and ctx.get("informative", True) else None
    analysis = ctx.get("analysis")
    conf = analysis["content"].get("confidence") if analysis and isinstance(analysis.get("content"), dict) else None
    label = f"自信度{conf}" if conf else ("あり" if analysis else "")
    if analysis and analysis.get("stale"):
        label += "（前回）"
    return {
        "venue": race["venue"], "race_num": race["race_num"], "race_name": race.get("race_name"),
        "race_class": race.get("race_class"), "grade": race.get("grade"), "surface": race.get("surface"),
        "distance": race.get("distance"), "field_size": len(ctx["horses"]), "start_time": race.get("start_time"),
        "href": href, "top": top, "bets": len(ctx["bets"]), "stake": sum(b["stake"] for b in ctx["bets"]),
        # 推奨が 0 点のとき、見送り (期待値が基準未満) か、そもそも計算できないのかを区別する
        "numbered": ctx.get("numbered", True), "has_odds": ctx.get("has_odds", True),
        "analysis": label,
    }


def np_safe(obj: Any) -> Any:
    """LLMに渡すJSON用に numpy / NaN を素の値にする."""
    if isinstance(obj, dict):
        return {k: np_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [np_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
