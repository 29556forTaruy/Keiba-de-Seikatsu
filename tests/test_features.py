import numpy as np
import pandas as pd

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
