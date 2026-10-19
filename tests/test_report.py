"""レポート (HTML・グラフ・集計) の表示まわり."""

import re
import shutil

import numpy as np
import pandas as pd
import pytest

from keiba import db, features, pipeline
from keiba.config import load_settings
from keiba.report import render, stats, svg


def _copy_db(demo_db, tmp_path):
    src, upcoming = demo_db
    dst = tmp_path / "copy.sqlite"
    shutil.copy(src, dst)
    return db.connect(dst), upcoming


def _settings(tmp_path):
    settings = load_settings(db=str(tmp_path / "unused.sqlite"))
    settings.raw["paths"]["reports"] = str(tmp_path / "reports")
    return settings


def _context(conn, settings, race_id, **kw):
    """build_reports と同じ手順で1レース分の表示用データを作る (モデルなし)."""
    runs = features.load_runs(conn)
    feat = features.build_features(runs, features.SpeedBaseline().fit(runs[runs["is_future"] == 0]))
    target = feat[feat["race_id"] == race_id]
    pred = pipeline.market_only_predict(target)
    by_horse = stats.index_by_horse(feat, target["horse_id"].unique())
    return render.race_context(db.get_race(conn, race_id), feat, pred, by_horse, stats.top2_by_race(feat),
                               betting=dict(settings.betting), index_href="index.html", **kw)


def _ticks(chart: str) -> list[str]:
    return re.findall(r'class="tick tick-y"[^>]*>([^<]*)<', chart)


# ---------------------------------------------------------------- グラフ

@pytest.mark.parametrize("end, expected", [
    (-11_000, ["-1万", "-7.5千", "-5千", "-2.5千", "0"]),
    (2_200, ["0", "500", "1千", "1.5千", "2千"]),
    (21_000, ["0", "5千", "1万", "1.5万", "2万"]),
    (150_000_000, ["0", "5000万", "1億", "1.5億"]),
])
def test_tick_labels_keep_fractions(end, expected):
    """目盛りを丸めて 2千 が2つ並んだり 7500 が 8千 になったりしない."""
    chart = svg.line_chart([{"name": "x", "values": list(np.linspace(0, end, 50))}])
    labels = _ticks(chart)
    assert labels == expected
    assert len(set(labels)) == len(labels)


def test_equity_chart_is_bankroll_not_profit(tmp_path):
    equity = [100_000, 98_000, 95_000, 97_000, 103_000, 108_000, 112_000, 110_000]
    bt = {
        "start": "2024-01-01", "end": "2024-02-01", "model_version": "t", "races": len(equity) - 1,
        "warnings": [], "strategies": [], "equity": equity, "calibration": [],
        "start_bankroll": equity[0], "final_bankroll": equity[-1], "model_metrics": None,
    }
    html = render.render_backtest(bt, tmp_path / "bt.html").read_text(encoding="utf-8")
    chart = html[html.index('id="equity"') - 200:]
    assert 'aria-label="ケリー資金管理の資金推移"' in chart
    labels = _ticks(chart)
    assert "0" not in labels, "資金の推移で縦軸に 0 を入れると線が上端に潰れる"
    assert len(labels) >= 3
    data = re.search(r'data-chart="([^"]*)"[^>]*>\s*<svg class="chart" id="equity"', html).group(1)
    import html as html_lib
    import json

    cfg = json.loads(html_lib.unescape(data))
    assert cfg["format"] == "yen_abs"
    assert cfg["labels"][0] == "開始" and cfg["labels"][-1] == f"{len(equity) - 1}レース後"
    assert "cfg.format === 'yen_abs'" in svg.CHART_JS


def test_chart_supports_touch():
    """スマホ: タップで表示し、指を離しても消さない. 目盛りは画面幅に合わせて拡大する."""
    js = svg.CHART_JS
    assert "addEventListener('pointerdown', show)" in js
    assert "e.pointerType === 'mouse'" in js
    assert "getComputedTextLength" in js and "addEventListener('resize', fit)" in js
    base = render.environment().get_template("base.html.j2").render()
    assert re.search(r"\.chart \{[^}]*touch-action: pan-y", base)


def test_form_controls_inherit_japanese_font():
    base = render.environment().get_template("base.html.j2").render()
    assert re.search(r"button[^{]*\{[^}]*font: inherit", base)
    assert '<html lang="ja">' in base


# ---------------------------------------------------------------- 欠けた値の表示

def test_missing_jockey_trainer_sex_not_shown_as_nan(demo_db, tmp_path):
    conn, upcoming = _copy_db(demo_db, tmp_path)
    race_id = pipeline.upcoming_race_ids(conn, upcoming.isoformat())[0]
    conn.execute(
        "UPDATE entries SET jockey_id = NULL, jockey_name = NULL, trainer_name = NULL, trainer_region = NULL,"
        " sex = NULL, age = NULL WHERE race_id = ? AND number = 1", (race_id,))
    conn.execute("UPDATE entries SET trainer_region = NULL WHERE race_id = ? AND number = 2", (race_id,))
    conn.commit()
    pages = pipeline.build_reports(conn, _settings(tmp_path), [race_id], bundle=None, progress=lambda *_: None)
    html = pages[0].read_text(encoding="utf-8")
    for bad in (">nan<", "nan厩舎", "nanNone", "（nan）", "None"):
        assert bad not in html, bad
    assert "騎手未定" in html


def test_same_race_history_frame_is_int():
    """枠順前の出馬表があると frame 列が float になる. 過去の勝ち馬の枠が 5.0 / f5.0 にならない."""
    feat = pd.DataFrame({
        "race_name": ["テスト記念", "テスト記念", "別のレース"],
        "date": pd.to_datetime(["2023-03-04", "2023-03-04", "2024-03-02"]),
        "finish_pos": [1, 2, np.nan],
        "horse_name": ["勝ち馬", "2着馬", "出走予定"],
        "sex": ["牡", "牝", None], "age": [4.0, 5.0, np.nan],
        "jockey_name": [np.nan, "騎手", "騎手"],
        "popularity": [2.0, 1.0, np.nan], "win_odds": [4.5, 2.0, np.nan],
        "frame": [5.0, 3.0, np.nan], "time_sec": [94.5, 94.6, np.nan], "going": ["良", "良", None],
        "first_pos": [0.2, 0.5, np.nan], "passing": ["2-2", "5-4", None],
    })
    rows = stats.same_race_history(feat, {"race_name": "テスト記念", "class_level": 6}, pd.Timestamp("2024-03-02"))
    assert len(rows) == 1
    assert rows[0]["frame"] == 5 and isinstance(rows[0]["frame"], int)
    assert rows[0]["sex_age"] == "牡4"
    assert rows[0]["jockey"] is None


def test_same_race_frame_fallback_in_template(demo_db, tmp_path):
    conn, upcoming = _copy_db(demo_db, tmp_path)
    race_id = pipeline.upcoming_race_ids(conn, upcoming.isoformat())[0]
    ctx = _context(conn, _settings(tmp_path), race_id)
    ctx["race"]["race_name"] = ctx["race"].get("race_name") or "テスト記念"
    ctx["same_race"] = [
        {"date": "2023/03/04", "horse": "勝ち馬", "sex_age": "牡4", "jockey": "騎手", "popularity": 2, "odds": 4.5,
         "frame": 5, "time": "1:34.5", "going": "良", "style": "先行"},
        {"date": "2022/03/05", "horse": "古い勝ち馬", "sex_age": None, "jockey": None, "popularity": None, "odds": None,
         "frame": None, "time": None, "going": None, "style": None},
    ]
    html = render.render_race(ctx, tmp_path / "r.html").read_text(encoding="utf-8")
    assert '<span class="frame f5">5</span>' in html
    assert "fNone" not in html and ">None<" not in html


# ---------------------------------------------------------------- 勝率に情報がない・計算できないとき

def test_no_model_no_odds_has_no_marks_or_top(demo_db, tmp_path):
    """モデルもオッズもないと全馬均等. 馬番順に ◎○▲ を付けたり、1番を本命にしたりしない."""
    conn, upcoming = _copy_db(demo_db, tmp_path)
    conn.execute("DELETE FROM odds")
    # 他のテストが保存した AI予想 の印は (均等でも) そのまま出す仕様なので消しておく
    conn.execute("DELETE FROM analyses")
    conn.commit()
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    ctx = _context(conn, _settings(tmp_path), ids[0])
    assert ctx["informative"] is False
    assert all(h["mark"] == "" for h in ctx["horses"])
    assert ctx["combos"] == {}
    assert render.summarize_for_index(ctx, "x.html")["top"] is None

    pages = pipeline.build_reports(conn, _settings(tmp_path), ids, bundle=None, progress=lambda *_: None)
    race_html = pages[0].read_text(encoding="utf-8")
    assert "全馬同じ勝率" in race_html
    assert not re.search(r'<span class="mark">[◎○▲△☆]</span>', race_html)
    assert "見送り" not in race_html and "期待値を計算できません" in race_html
    index = next(p for p in pages if p.name == "index.html").read_text(encoding="utf-8")
    assert "オッズ未取得" in index and "見送り" not in index
    assert not re.search(r'<td class="num">\d+\.\d%</td>', index), "本命の勝率は空欄"


def test_marks_with_odds_still_shown(demo_db, tmp_path):
    conn, upcoming = _copy_db(demo_db, tmp_path)
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    ctx = _context(conn, _settings(tmp_path), ids[0])
    assert ctx["informative"] is True
    assert "◎" in [h["mark"] for h in ctx["horses"]]
    top = render.summarize_for_index(ctx, "x.html")
    assert top["top"] is not None and top["has_odds"] and top["numbered"]


def test_before_draw_index_says_undrawn(demo_db, tmp_path):
    conn, upcoming = _copy_db(demo_db, tmp_path)
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    conn.execute("UPDATE entries SET number = NULL, frame = NULL WHERE race_id = ?", (ids[0],))
    conn.execute("DELETE FROM odds WHERE race_id = ?", (ids[0],))
    conn.commit()
    pages = pipeline.build_reports(conn, _settings(tmp_path), ids[:1], bundle=None, progress=lambda *_: None)
    race_html = pages[0].read_text(encoding="utf-8")
    assert "枠順(馬番)の確定後に表示します" in race_html and "見送り" not in race_html
    index = next(p for p in pages if p.name == "index.html").read_text(encoding="utf-8")
    assert "枠順未確定" in index and "見送り" not in index


def test_partial_odds_note(demo_db, tmp_path):
    """一部の馬だけオッズがない (出走取消の反映前など) ときは注記を出す."""
    conn, upcoming = _copy_db(demo_db, tmp_path)
    race_id = pipeline.upcoming_race_ids(conn, upcoming.isoformat())[0]
    conn.execute("DELETE FROM odds WHERE race_id = ? AND number = 3", (race_id,))
    conn.commit()
    ctx = _context(conn, _settings(tmp_path), race_id)
    assert ctx["has_odds"] and ctx["no_odds"] == [3]
    assert all(b["combination"] != "3" for b in ctx["bets"])
    html = render.render_race(ctx, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "馬番 3 のオッズがありません" in html
    assert f"keiba collect card {race_id}" in html


# ---------------------------------------------------------------- AI予想の作成日時

def test_analysis_shows_created_at_and_stale_label(demo_db, tmp_path):
    conn, upcoming = _copy_db(demo_db, tmp_path)
    race_id = pipeline.upcoming_race_ids(conn, upcoming.isoformat())[0]
    analysis = {
        "id": 1, "worker": "fake_predict", "created_at": "2024-03-02T09:15:30", "stale": True,
        "content": {"summary": "朝の見解", "marks": [{"number": 2, "mark": "◎", "reason": ""}], "bets": [],
                    "risks": [], "skip_reason": "", "confidence": "B", "pace": "ミドル"},
    }
    review = {"id": 2, "worker": "fake_review", "created_at": "2024-03-02T09:16:05", "content": "総合判断: 賛成"}
    ctx = _context(conn, _settings(tmp_path), race_id, analysis=analysis, review=review)
    html = render.render_race(ctx, tmp_path / "r.html").read_text(encoding="utf-8")
    assert "fake_predict ・ 2024-03-02 09:15作成" in html
    assert "fake_review ・ 2024-03-02 09:16作成" in html
    assert "前回の予想（今回の作成に失敗）" in html
    assert render.summarize_for_index(ctx, "x.html")["analysis"] == "自信度B（前回）"

    fresh = {**analysis, "stale": False}
    html = render.render_race(_context(conn, _settings(tmp_path), race_id, analysis=fresh),
                              tmp_path / "r2.html").read_text(encoding="utf-8")
    assert "2024-03-02 09:15作成" in html and "前回の予想" not in html
