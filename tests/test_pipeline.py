"""架空データで 学習 → バックテスト → レポート(LLM作業者はダミー) を通す."""

import dataclasses
import json
import re
from datetime import timedelta

import pytest

from keiba import backtest, db, features, model, pipeline
from keiba.config import load_settings
from keiba.llm import WorkerError, WorkerResult
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


def _settings(tmp_path, db_name="unused.sqlite"):
    settings = load_settings(db=str(tmp_path / db_name))
    settings.raw["paths"]["reports"] = str(tmp_path / "reports")
    settings.raw["paths"]["models"] = str(tmp_path / "models")
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


# ---------------------------------------------------------------- 成績未取得の過去の出馬表

def _unsettle(conn, upcoming, n_days):
    """直前 n_days 開催日のレースを「keiba day で出馬表だけ取り、成績は取り込んでいない」状態にする."""
    days = [r[0] for r in conn.execute(
        "SELECT DISTINCT date FROM races WHERE date < ? ORDER BY date DESC LIMIT ?", (upcoming.isoformat(), n_days))]
    ids = [r[0] for r in conn.execute(
        f"SELECT race_id FROM races WHERE date IN ({','.join('?' * len(days))}) ORDER BY date, race_id", days)]
    marks = ",".join("?" * len(ids))
    conn.execute(f"""
        INSERT OR REPLACE INTO entries (race_id, horse_id, horse_name, frame, number, sex, age, weight_carried,
            jockey_id, jockey_name, trainer_id, trainer_name, trainer_region, horse_weight, weight_diff, scratched)
        SELECT race_id, horse_id, horse_name, frame, number, sex, age, weight_carried, jockey_id, jockey_name,
            trainer_id, trainer_name, trainer_region, horse_weight, weight_diff,
            CASE WHEN finish_status IN ('取消', '除外') THEN 1 ELSE 0 END
        FROM results WHERE race_id IN ({marks})""", ids)
    conn.execute(f"DELETE FROM results WHERE race_id IN ({marks})", ids)
    conn.execute(f"DELETE FROM payouts WHERE race_id IN ({marks})", ids)
    conn.commit()
    return ids


def _blank_last_run(feat, race_ids):
    t = feat[feat["race_id"].isin(race_ids)]
    return int((t["h_last_fin"].isna() & (t["h_n_runs"] > 0)).sum())


def test_unsettled_past_cards_do_not_blank_last_run(demo_db, tmp_path):
    conn, upcoming = _copy_db(demo_db, tmp_path)
    _unsettle(conn, upcoming, 8)
    targets = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    runs = features.load_runs(conn)
    baseline = features.SpeedBaseline().fit(runs[runs["is_future"] == 0])
    # 直さないと、前走が「着順なしの1走」になって近走の特徴量が欠ける (学習時には無い組み合わせ)
    assert _blank_last_run(features.build_features(runs, baseline), targets) > 0

    msgs = []
    kept = pipeline._drop_stale_cards(runs, targets, msgs.append)
    assert set(kept.loc[kept["is_future"] == 1, "race_id"]) == set(targets)
    assert any("成績が未取得の過去レース" in m and "keiba collect results" in m for m in msgs)
    assert _blank_last_run(features.build_features(kept, baseline), targets) == 0


def test_run_day_collects_unsettled_results_and_falls_back_on_fetch_error(demo_db, tmp_path, monkeypatch):
    from keiba.scraper import collect, http

    conn, upcoming = _copy_db(demo_db, tmp_path)
    stale = _unsettle(conn, upcoming, 1)
    marks = ",".join("?" * len(stale))
    conn.execute(f"UPDATE races SET source = 'netkeiba' WHERE race_id IN ({marks})", stale)
    conn.commit()
    assert pipeline.unsettled_race_ids(conn, upcoming.isoformat()) == stale

    calls = {"race": [], "odds": [], "pedigree": 0}
    monkeypatch.setattr(http.Fetcher, "from_settings", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(collect, "collect_cards_for_date", lambda *a, **k: [])
    monkeypatch.setattr(collect, "collect_race", lambda conn, f, rid, **k: calls["race"].append(rid))
    monkeypatch.setattr(collect, "collect_pedigrees", lambda *a, **k: calls.__setitem__("pedigree", 1))

    # オッズが取れれば、そのあと前の開催日の成績を取り込んでから予想する
    monkeypatch.setattr(collect, "collect_odds", lambda conn, f, rid: {"market": True, "scratched": []})
    pipeline.run_day(conn, _settings(tmp_path), upcoming, progress=lambda *_: None)
    assert calls["race"] == stale, "前の開催日の成績を取り込む"

    def blocked(conn, f, rid):
        calls["odds"].append(rid)
        raise http.FetchError("HTTP 403: アクセスが拒否されました")

    calls["race"].clear()
    calls["pedigree"] = 0
    monkeypatch.setattr(collect, "collect_odds", blocked)
    msgs = []
    pages = pipeline.run_day(conn, _settings(tmp_path), upcoming, progress=msgs.append)
    # 取得エラーが続いたら以降の通信はせず、DBにある分だけでレポートを作る
    assert len(calls["odds"]) == pipeline.MAX_FETCH_FAILURES and calls["pedigree"] == 0 and not calls["race"]
    assert any("DBにあるデータだけ" in m for m in msgs)
    assert any(p.name == "index.html" for p in pages)


def test_run_day_falls_back_when_race_list_fails(demo_db, tmp_path, monkeypatch):
    from keiba.scraper import collect, http

    conn, upcoming = _copy_db(demo_db, tmp_path)
    monkeypatch.setattr(http.Fetcher, "from_settings", staticmethod(lambda *a, **k: None))

    def down(*a, **k):
        raise http.FetchError("取得に失敗しました: race_list_sub")

    monkeypatch.setattr(collect, "collect_cards_for_date", down)
    msgs = []
    pages = pipeline.run_day(conn, _settings(tmp_path), upcoming, progress=msgs.append)
    assert any("DBにあるデータだけ" in m for m in msgs)
    assert len(pages) == len(pipeline.upcoming_race_ids(conn, upcoming.isoformat())) + 1


# ---------------------------------------------------------------- モデルの置き場所

def test_models_are_scoped_per_db(trained, tmp_path):
    _, _, bundle, _, _ = trained
    bundle = dataclasses.replace(bundle, metrics=dict(bundle.metrics))
    demo = _settings(tmp_path, "demo.sqlite")
    path = pipeline.save_model(demo, bundle)
    assert path.parent == tmp_path / "models" / "demo"
    assert pipeline.latest_model_path(demo) == path
    assert pipeline.load_model(demo).metrics["db"] == str((tmp_path / "demo.sqlite").resolve())
    # 別のDBでは、デモDBで学習したモデルを使わない (市場の勝率だけになる)
    real = _settings(tmp_path, "keiba.sqlite")
    assert pipeline.latest_model_path(real) is None
    assert pipeline.load_model(real, progress=lambda *_: None) is None
    # 以前の版で models/ 直下に保存したモデルは使わず、そのことを知らせる
    (tmp_path / "models" / "old.joblib").write_bytes(b"")
    msgs = []
    assert pipeline.load_model(real, progress=msgs.append) is None
    assert any("直下の古いモデルは使いません" in m for m in msgs)
    # 名前だけでも、そのDBのモデル置き場から探せる
    assert pipeline.load_model(demo, bundle.version).version == bundle.version


def test_explicit_missing_model_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="モデルファイルが見つかりません"):
        pipeline.load_model(_settings(tmp_path), tmp_path / "models" / "v2025.joblb")


# ---------------------------------------------------------------- AI作業者の失敗

class BrokenWorker:
    def __init__(self, name, exc):
        self.name, self.exc, self.calls = name, exc, 0

    def run(self, prompt, **kwargs):
        self.calls += 1
        raise self.exc


def _capture_contexts(monkeypatch):
    seen = {}
    original = pipeline.render.render_race

    def spy(ctx, path):
        seen[ctx["race"]["race_id"]] = ctx
        return original(ctx, path)

    monkeypatch.setattr(pipeline.render, "render_race", spy)
    return seen


def test_news_failure_does_not_skip_prediction(trained, demo_db, tmp_path, monkeypatch):
    _, _, bundle, _, _ = trained
    conn, upcoming = _copy_db(demo_db, tmp_path)
    settings = _settings(tmp_path)
    settings.raw["workers"] = {"news": "bad_news", "predict": "good", "review": ""}
    workers = {"bad_news": BrokenWorker("bad_news", WorkerError("quota")), "good": FakeWorker("good")}
    monkeypatch.setattr(pipeline, "make_worker", lambda name, _s: workers[name])
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())[:2]
    msgs = []
    pipeline.build_reports(conn, settings, ids, bundle=bundle, roles={"news", "predict"}, llm_filter="all",
                           progress=msgs.append)
    assert workers["bad_news"].calls == 2
    assert len(workers["good"].calls) == 2, "直前情報が失敗しても予想は作る"
    assert db.latest_analysis(conn, ids[0], "predict")["worker"] == "good"


@pytest.mark.parametrize("failure", ["run", "init"])
def test_unexpected_worker_errors_keep_pages_and_mark_stale(trained, demo_db, tmp_path, monkeypatch, failure):
    """APIキー未設定 (SDKの TypeError) や初期化失敗でも、全レースのページと一覧を作る."""
    _, _, bundle, _, _ = trained
    conn, upcoming = _copy_db(demo_db, tmp_path)
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    db.save_analysis(conn, ids[0], "predict", "old", {"summary": "前回の見解", "marks": [], "bets": []})
    conn.commit()
    settings = _settings(tmp_path)
    settings.raw["workers"] = {"news": "", "predict": "p", "review": ""}
    broken = BrokenWorker("p", TypeError("Could not resolve authentication method"))
    built = []

    def make(name, _s):
        built.append(name)
        if failure == "init":
            raise RuntimeError("SDKの初期化に失敗")
        return broken

    monkeypatch.setattr(pipeline, "make_worker", make)
    seen = _capture_contexts(monkeypatch)
    msgs = []
    pages = pipeline.build_reports(conn, settings, ids, bundle=bundle, roles={"predict"}, llm_filter="all",
                                   progress=msgs.append)
    assert len(pages) == len(ids) + 1 and pages[-1].name == "index.html"
    assert built == ["p"], "作れなかった作業者を毎レース作り直さない"
    if failure == "run":
        assert broken.calls == len(ids)
        assert any("作業者エラー" in m and "TypeError" in m for m in msgs)
    # 今回の予想に失敗したので、前回の予想は古いものとして渡す
    assert seen[ids[0]]["analysis"]["stale"] is True
    assert seen[ids[0]]["analysis"]["created_at"]


def test_review_is_shown_only_with_the_prediction_it_reviewed(trained, demo_db, tmp_path, monkeypatch):
    _, _, bundle, _, _ = trained
    conn, upcoming = _copy_db(demo_db, tmp_path)
    settings = _settings(tmp_path)
    settings.raw["workers"] = {"news": "", "predict": "p", "review": "r"}
    workers = {"p": FakeWorker("p"), "r": FakeWorker("r")}
    monkeypatch.setattr(pipeline, "make_worker", lambda name, _s: workers[name])
    seen = _capture_contexts(monkeypatch)
    rid = pipeline.upcoming_race_ids(conn, upcoming.isoformat())[0]
    run = lambda roles: pipeline.build_reports(conn, settings, [rid], bundle=bundle, roles=roles,  # noqa: E731
                                               llm_filter="all", progress=lambda *_: None)
    run({"predict", "review"})
    first = seen[rid]["analysis"]
    assert seen[rid]["review"]["ref_id"] == first["id"]
    assert "stale" not in first
    run({"predict"})  # 予想だけ作り直した: 古い予想への点検は出さない
    assert seen[rid]["analysis"]["id"] > first["id"]
    assert seen[rid]["review"] is None
    run(set())  # AIを使わない再実行でも同じ
    assert seen[rid]["review"] is None


def test_review_for_legacy_rows(tmp_path):
    conn = db.connect(tmp_path / "x.sqlite")
    a1 = db.save_analysis(conn, "R1", "predict", "w", {"summary": "1"})
    conn.execute("INSERT INTO analyses (race_id, role, worker, content) VALUES ('R1', 'review', 'w', '古い点検')")
    a2 = db.save_analysis(conn, "R1", "predict", "w", {"summary": "2"})
    assert db.review_for(conn, "R1", a1)["content"] == "古い点検"
    assert db.review_for(conn, "R1", a2) is None
    r = db.save_analysis(conn, "R1", "review", "w", "新しい点検", ref_id=a2)
    assert db.review_for(conn, "R1", a2)["id"] == r


def test_old_db_gets_ref_id_column(tmp_path):
    import sqlite3

    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE analyses (id INTEGER PRIMARY KEY AUTOINCREMENT, race_id TEXT NOT NULL, "
                "role TEXT NOT NULL, worker TEXT, content TEXT, created_at TEXT)")
    old.commit()
    old.close()
    conn = db.connect(path)
    assert "ref_id" in {r[1] for r in conn.execute("PRAGMA table_info(analyses)")}


# ---------------------------------------------------------------- 日付の一覧

def test_single_race_run_keeps_the_day_index(trained, demo_db, tmp_path):
    _, _, bundle, _, _ = trained
    conn, upcoming = _copy_db(demo_db, tmp_path)
    settings = _settings(tmp_path)
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    pipeline.build_reports(conn, settings, ids, bundle=bundle, progress=lambda *_: None)
    pages = pipeline.build_reports(conn, settings, ids[:1], bundle=bundle, progress=lambda *_: None)
    index = pages[-1]
    assert index.name == "index.html"
    html = index.read_text(encoding="utf-8")
    assert all(f'href="{rid}.html"' in html for rid in ids), "1レースだけ作り直しても一覧は減らない"
    assert f"{len(ids)}レース" in html
    summary = json.loads((index.parent / f"{ids[0]}.summary.json").read_text(encoding="utf-8"))
    assert set(summary["top"]) == set(pipeline.INDEX_TOP_KEYS)
    assert summary["is_demo"] is True
    # ページを消したレースは一覧から外す
    (index.parent / f"{ids[1]}.html").unlink()
    html = pipeline.build_reports(conn, settings, ids[:1], bundle=bundle, progress=lambda *_: None)[-1].read_text(
        encoding="utf-8")
    assert f'href="{ids[1]}.html"' not in html and f"{len(ids) - 1}レース" in html


def test_market_only_predict_zeroes_horses_without_odds():
    """モデル未学習でも、オッズのあるレースでオッズの無い馬は取消扱い (勝率0)."""
    import pandas as pd

    rows = pd.DataFrame({
        "race_id": ["R1"] * 4 + ["R2"] * 3, "horse_id": list("abcdefg"), "horse_name": list("abcdefg"),
        "number": [1, 2, 3, 4, 1, 2, 3], "win_odds": [2.0, 4.0, None, 8.0, None, None, None],
        "p_market": [0.57, 0.29, None, 0.14, None, None, None],
    })
    out = pipeline.market_only_predict(rows).set_index("horse_id")
    assert out.loc["c", "p_win"] == 0.0
    assert out.loc[list("abd"), "p_win"].sum() == pytest.approx(1.0)
    assert out.loc[list("efg"), "p_win"].tolist() == pytest.approx([1 / 3] * 3)  # オッズが無いレースは均等


def test_day_index_ignores_pages_from_another_db(trained, demo_db, tmp_path):
    """reports/ を共有していても、別のDBで作ったページは一覧に混ぜない."""
    import shutil

    _, _, bundle, _, _ = trained
    conn, upcoming = _copy_db(demo_db, tmp_path)
    ids = pipeline.upcoming_race_ids(conn, upcoming.isoformat())
    pipeline.build_reports(conn, _settings(tmp_path, "copy.sqlite"), ids, bundle=bundle, progress=lambda *_: None)
    other = tmp_path / "other.sqlite"
    shutil.copy(tmp_path / "copy.sqlite", other)
    conn2 = db.connect(other)
    index = pipeline.build_reports(conn2, _settings(tmp_path, "other.sqlite"), ids[:1], bundle=bundle,
                                   progress=lambda *_: None)[-1]
    html = index.read_text(encoding="utf-8")
    assert f'href="{ids[0]}.html"' in html
    assert f'href="{ids[1]}.html"' not in html and "1レース" in html


def test_backtest_metrics_use_the_same_races_as_the_simulation(trained):
    conn, runs, bundle, valid_end, upcoming = trained
    start = (valid_end + timedelta(days=1)).date()
    res = backtest.run_backtest(conn, bundle, str(start), str(upcoming), {"bankroll": 100_000}, runs=runs)
    assert res["test_metrics"]["races"] == res["races"]
    assert {"nll_model", "nll_market", "nll_blend"} <= set(res["test_metrics"])


def test_catch_up_only_fetches_recent_unsettled_races(demo_db, tmp_path, monkeypatch):
    """day のついでに取り込むのは直近の未取得レースだけ。古いものは collect results を案内する."""
    from keiba.scraper import collect

    conn, upcoming = _copy_db(demo_db, tmp_path)
    stale = _unsettle(conn, upcoming, 8)
    conn.execute(f"UPDATE races SET source = 'netkeiba' WHERE race_id IN ({','.join('?' * len(stale))})", stale)
    conn.commit()
    fetched, msgs = [], []
    monkeypatch.setattr(collect, "collect_race", lambda conn, f, rid, **k: fetched.append(rid))
    pipeline._catch_up_results(conn, None, upcoming, msgs.append)
    since = (upcoming - timedelta(days=pipeline.CATCH_UP_DAYS)).isoformat()
    assert fetched and len(fetched) < len(stale)
    assert all(db.get_race(conn, rid)["date"] >= since for rid in fetched)
    assert len(fetched) <= pipeline.CATCH_UP_MAX_RACES
    assert any("keiba collect results" in m for m in msgs)
