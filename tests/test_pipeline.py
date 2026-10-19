"""架空データで 学習 → バックテスト → レポート(LLM作業者はダミー) を通す."""

import re
from datetime import timedelta

import pytest

from keiba import backtest, db, features, model, pipeline
from keiba.config import load_settings
from keiba.llm import WorkerResult
from keiba.llm import tasks as llm_tasks

# 中国語フォント (簡体・繁体) の名前。レポートに含めてはいけない
CHINESE_FONTS = [
    "PingFang", "Microsoft YaHei", "SimSun", "SimHei", "NSimSun", "Heiti SC", "Heiti TC", "STHeiti",
    "Songti", "Kaiti", "STSong", "Noto Sans SC", "Noto Sans TC", "Noto Sans CJK SC", "Noto Sans CJK TC",
    "Source Han Sans CN", "Source Han Sans TW", "Microsoft JhengHei", "WenQuanYi", "Hiragino Sans GB",
]


class FakeWorker:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def run(self, prompt, *, system=None, schema=None, web_search=False):
        self.calls.append({"prompt": prompt, "schema": schema, "web_search": web_search})
        if web_search:
            return WorkerResult(text="1 テスト: 追い切り良好（出典 https://example.com/a）", worker=self.name,
                                sources=[{"url": "https://example.com/a", "title": "例"}, {"url": "javascript:alert(1)"}])
        if schema:
            nums = [int(n) for n in re.findall(r'"馬番": (\d+)', prompt)]
            return WorkerResult(text="{}", worker=self.name, data={
                "summary": "テスト用の見解", "pace": "ミドル", "confidence": "B",
                "marks": [{"number": nums[0], "mark": "◎", "reason": "確率最上位"},
                          {"number": 999, "mark": "○", "reason": "存在しない馬番"}],
                "bets": [{"type": "単勝", "combination": str(nums[0]), "amount": 50_000, "reason": "予算超過"},
                         {"type": "馬連", "combination": f"{nums[0]}-{nums[0]}", "amount": 100, "reason": "不正"}],
                "risks": ["テスト"], "skip_reason": "",
            })
        return WorkerResult(text="総合判断: 賛成", worker=self.name)


@pytest.fixture(scope="module")
def trained(demo_db, tmp_path_factory):
    path, upcoming = demo_db
    conn = db.connect(path)
    runs = features.load_runs(conn)
    dates = sorted(runs.loc[runs["is_future"] == 0, "date"].unique())
    train_end, valid_end = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.85)]
    bundle = model.train(runs, train_end=str(train_end.date()), valid_end=str(valid_end.date()))
    return conn, runs, bundle, valid_end, upcoming


def test_model_probabilities(trained):
    conn, runs, bundle, _, _ = trained
    feat = features.build_features(runs, bundle.baseline)
    fut = feat[feat["is_future"] == 1]
    pred = bundle.predict(fut)
    sums = pred.groupby("race_id")["p_win"].sum()
    assert sums.round(6).eq(1.0).all()
    assert (pred["p_place"] >= pred["p_win"] - 1e-9).all()
    assert bundle.blend is not None
    assert bundle.metrics["valid"]["races"] > 0


def test_backtest_runs(trained):
    conn, runs, bundle, valid_end, upcoming = trained
    start = (valid_end + timedelta(days=1)).date()
    res = backtest.run_backtest(conn, bundle, str(start), str(upcoming), {"bankroll": 100_000}, runs=runs)
    assert res["races"] > 0
    assert not res["warnings"]
    names = [s["name"] for s in res["strategies"]]
    assert "1番人気の単勝" in names and "ケリー資金管理" in names
    overlap = backtest.run_backtest(conn, bundle, str(runs["date"].min().date()), str(upcoming), {}, runs=runs)
    assert overlap["warnings"], "学習期間と重なるときは警告を出す"


def test_reports_with_workers(trained, tmp_path, monkeypatch):
    conn, runs, bundle, _, upcoming = trained
    settings = load_settings(db=str(tmp_path / "unused.sqlite"))
    settings.raw["paths"]["reports"] = str(tmp_path / "reports")
    settings.raw["workers"] = {"news": "fake_news", "predict": "fake_predict", "review": "fake_review"}
    workers = {}

    def fake_make_worker(name, _settings):
        return workers.setdefault(name, FakeWorker(name))

    monkeypatch.setattr(pipeline, "make_worker", fake_make_worker)
    race_ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    pages = pipeline.build_reports(conn, settings, race_ids, bundle=bundle,
                                   roles={"news", "predict", "review"}, llm_filter="all", budget=3_000,
                                   progress=lambda *_: None)
    index = [p for p in pages if p.name == "index.html"]
    assert len(index) == 1 and len(pages) == len(race_ids) + 1
    assert len(workers["fake_predict"].calls) == len(race_ids)
    assert workers["fake_news"].calls[0]["web_search"] is True

    analysis = db.latest_analysis(conn, race_ids[0], "predict")["content"]
    assert [m["number"] for m in analysis["marks"]] != [] and all(m["number"] != 999 for m in analysis["marks"])
    assert sum(b["amount"] for b in analysis["bets"]) <= 3_000
    assert all(b["type"] != "馬連" for b in analysis["bets"])

    html = pages[0].read_text(encoding="utf-8")
    assert '<html lang="ja">' in html
    assert "Hiragino Sans" in html
    assert "AI予想" in html and "テスト用の見解" in html and "総合判断: 賛成" in html
    assert "javascript:alert" not in html
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for font in CHINESE_FONTS:
            assert font not in text, f"{page.name} に中国語フォント {font} が含まれています"


def test_validate_prediction_drops_invalid():
    data = {
        "summary": "x", "pace": "スロー", "confidence": "C", "risks": [], "skip_reason": "",
        "marks": [{"number": 1, "mark": "◎", "reason": ""}, {"number": 1, "mark": "○", "reason": "重複"}],
        "bets": [{"type": "三連単", "combination": "1→2→3", "amount": 250, "reason": ""},
                 {"type": "ワイド", "combination": "1-9", "amount": 100, "reason": "出走していない馬"}],
    }
    out = llm_tasks.validate_prediction(data, {1, 2, 3}, budget=1_000)
    assert out["marks"] == [{"number": 1, "mark": "◎", "reason": ""}]
    assert out["bets"] == [{"type": "三連単", "combination": "1-2-3", "amount": 200, "reason": ""}]


def _copy_db(demo_db, tmp_path):
    import shutil

    src, upcoming = demo_db
    dst = tmp_path / "copy.sqlite"
    shutil.copy(src, dst)
    return db.connect(dst), upcoming


def _settings(tmp_path):
    settings = load_settings(db=str(tmp_path / "unused.sqlite"))
    settings.raw["paths"]["reports"] = str(tmp_path / "reports")
    return settings


@pytest.mark.parametrize("use_model", [True, False])
def test_reports_without_odds(trained, demo_db, tmp_path, use_model):
    """オッズ未発売 (odds テーブルが空) でもレポートが作れる."""
    _, _, bundle, _, _ = trained
    conn, upcoming = _copy_db(demo_db, tmp_path)
    conn.execute("DELETE FROM odds")
    conn.commit()
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    pages = pipeline.build_reports(conn, _settings(tmp_path), ids, bundle=bundle if use_model else None,
                                   progress=lambda *_: None)
    assert len(pages) == len(ids) + 1
    assert "オッズ未取得" in pages[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("use_model", [True, False])
def test_reports_before_draw(trained, demo_db, tmp_path, use_model):
    """枠順発表前 (馬番が空) の出馬表でもレポートが作れる."""
    _, _, bundle, _, _ = trained
    conn, upcoming = _copy_db(demo_db, tmp_path)
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    conn.execute("UPDATE entries SET number = NULL, frame = NULL WHERE race_id = ?", (ids[0],))
    conn.commit()
    pages = pipeline.build_reports(conn, _settings(tmp_path), ids[:1], bundle=bundle if use_model else None,
                                   progress=lambda *_: None)
    html = pages[0].read_text(encoding="utf-8")
    assert "枠順(馬番)が未確定" in html
    assert ">None<" not in html


def test_last_popularity_uses_previous_field(demo_db):
    path, _ = demo_db
    conn = db.connect(path)
    runs = features.load_runs(conn)
    feat = features.build_features(runs, features.SpeedBaseline().fit(runs[runs["is_future"] == 0]))
    h = feat.groupby("horse_id", sort=False)
    expected = h["popularity"].shift() / h["field_size"].shift()
    ok = expected.notna()
    assert ok.any()
    assert (feat.loc[ok, "h_last_pop"] - expected[ok]).abs().max() < 1e-9
