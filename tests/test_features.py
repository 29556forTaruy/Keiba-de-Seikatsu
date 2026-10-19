import numpy as np
import pandas as pd
import pytest

from keiba import db, features


def _build(runs):
    baseline = features.SpeedBaseline().fit(runs[runs["is_future"] == 0])
    return features.build_features(runs, baseline)


def test_no_future_leakage(demo_db):
    """ある日より後の結果を書き換えても、それ以前のレースの特徴量は変わらない."""
    path, _ = demo_db
    conn = db.connect(path)
    runs = features.load_runs(conn)
    baseline = features.SpeedBaseline().fit(runs[runs["is_future"] == 0])
    cutoff = runs["date"].quantile(0.6)

    original = features.build_features(runs, baseline)
    tampered = runs.copy()
    later = tampered["date"] > cutoff
    rng = np.random.default_rng(0)
    tampered.loc[later, "finish_pos"] = rng.permutation(tampered.loc[later, "finish_pos"].to_numpy())
    tampered.loc[later, "time_sec"] = tampered.loc[later, "time_sec"] + rng.normal(0, 2, later.sum())
    tampered.loc[later, "prize"] = 0.0
    tampered.loc[later, "horse_weight"] = tampered.loc[later, "horse_weight"] + rng.integers(-20, 21, later.sum())
    changed = features.build_features(tampered, baseline)

    early = original["date"] <= cutoff
    for col in features.FEATURES:
        a = original.loc[early, col]
        b = changed.loc[early, col]
        if pd.api.types.is_numeric_dtype(a):
            assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True), col
        else:
            assert (a.astype(str) == b.astype(str)).all(), col


def test_future_rows_have_features(demo_db):
    path, upcoming = demo_db
    conn = db.connect(path)
    feat = _build(features.load_runs(conn))
    fut = feat[feat["is_future"] == 1]
    assert not fut.empty
    assert (fut["date"] == pd.Timestamp(upcoming)).all()
    assert fut["won"].isna().all()
    # 過去に走った馬は近走の特徴量が埋まっている
    ran_before = fut["h_n_runs"] > 0
    assert fut.loc[ran_before, "h_last_fin"].notna().all()
    assert fut["j_win"].notna().all()
    assert fut["p_market"].notna().all()
    # 市場勝率はレースごとに合計1
    assert np.allclose(fut.groupby("race_id")["p_market"].sum(), 1.0)


def test_features_do_not_need_race_day_weight(demo_db):
    """馬体重の発表前 (前日・朝の予想) と発表後で、モデルの入力が変わらない."""
    path, _ = demo_db
    conn = db.connect(path)
    runs = features.load_runs(conn)
    baseline = features.SpeedBaseline().fit(runs[runs["is_future"] == 0])
    fut = runs["is_future"] == 1
    assert runs.loc[fut, "horse_weight"].notna().all()
    blank = runs.copy()
    blank.loc[fut, ["horse_weight", "weight_diff"]] = np.nan
    a = features.build_features(runs, baseline)
    b = features.build_features(blank, baseline)
    assert "horse_weight" not in features.FEATURES and "weight_diff" not in features.FEATURES
    pd.testing.assert_frame_equal(features.model_matrix(a[fut]), features.model_matrix(b[fut]))
    # 前走の馬体重は、その馬の1走前の行の値
    prev = a[a["is_future"] == 0].groupby("horse_id").tail(1).set_index("horse_id")["horse_weight"]
    ran = a[fut & (a["h_n_runs"] > 0)]
    assert not ran.empty
    assert np.allclose(ran["h_last_weight"].to_numpy(float), prev.reindex(ran["horse_id"]).to_numpy(float),
                       equal_nan=True)


def _course(rng, venue, surface, distance, going, n, mean):
    return pd.DataFrame({
        "venue": venue, "surface": surface, "distance": float(distance), "going": going,
        "time_sec": rng.normal(mean, 1.0, n), "finish_pos": 1.0,
    })


def _going_world(factor, n_heavy, surface="芝", seed=0):
    """東京は不良馬場の件数が少なく、ほかのコースから馬場の影響を学ぶ架空データ."""
    rng = np.random.default_rng(seed)
    parts = [
        _course(rng, "東京", surface, 1600, "良", 200, 94.5),
        _course(rng, "東京", surface, 1600, "不良", n_heavy, 94.5 * factor),
        _course(rng, "中山", surface, 1600, "良", 150, 95.0),
        _course(rng, "中山", surface, 1600, "不良", 60, 95.0 * factor),
        _course(rng, "阪神", surface, 1800, "良", 150, 107.0),
        _course(rng, "阪神", surface, 1800, "不良", 60, 107.0 * factor),
    ]
    return pd.concat(parts, ignore_index=True)


def _fig(baseline, surface, going, time_sec):
    row = pd.DataFrame({"venue": ["東京"], "surface": [surface], "distance": [1600.0], "going": [going],
                        "time_sec": [time_sec], "finish_pos": [np.nan]})
    return float(baseline.transform(row).iloc[0])


@pytest.mark.parametrize("surface, factor", [("芝", 1.03), ("ダ", 0.99)])
def test_speed_figure_comparable_across_goings(surface, factor):
    """件数の少ない馬場でも、その馬場の平均的なタイムは指数50前後になる.

    (以前は良馬場が大半のコース平均と比べていたので、芝の不良は低すぎ、ダートの不良は高すぎた)
    """
    base = features.SpeedBaseline().fit(_going_world(factor, n_heavy=5, surface=surface))
    assert abs(_fig(base, surface, "良", 94.5) - 50) < 3
    assert abs(_fig(base, surface, "不良", 94.5 * factor) - 50) < 3


def test_speed_figure_does_not_jump_at_cell_size():
    """(競馬場, 芝ダ, 距離, 馬場) の件数がしきい値をまたいでも、同じタイムの指数がほぼ同じ."""
    figs = [_fig(features.SpeedBaseline().fit(_going_world(1.03, n_heavy=n)), "芝", "不良", 97.3)
            for n in (28, 30)]
    assert abs(figs[0] - figs[1]) < 2
    assert all(abs(f - 50) < 4 for f in figs)


def test_speed_figure_by_going_on_demo(demo_db):
    """デモDBでも、馬場ごとの指数の平均が50前後にそろう."""
    path, _ = demo_db
    conn = db.connect(path)
    runs = features.load_runs(conn)
    past = runs[runs["is_future"] == 0]
    fig = features.SpeedBaseline().fit(past).transform(past)
    done = past.assign(fig=fig)[past["finish_pos"].notna()]
    stats = done.groupby(["surface", "going"])["fig"].agg(["mean", "size"])
    stats = stats[stats["size"] >= 100]
    assert len(stats) >= 6
    assert ((stats["mean"] - 50).abs() < 3).all(), stats
