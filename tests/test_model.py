"""勝率モデル: オッズが一部の馬だけ無いレースの扱いと、検証期間の精度の測り方."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from keiba import db, features, model

METRIC_KEYS = {"races", "nll_model", "top1_model", "nll_market", "top1_market", "nll_blend", "top1_blend"}


@pytest.fixture(scope="module")
def trained(demo_db):
    path, _ = demo_db
    conn = db.connect(path)
    runs = features.load_runs(conn)
    conn.close()
    dates = sorted(runs.loc[runs["is_future"] == 0, "date"].unique())
    train_end, valid_end = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.85)]
    bundle = model.train(runs, train_end=str(train_end.date()), valid_end=str(valid_end.date()))
    return runs, bundle


def _predict_future(bundle, runs):
    feat = features.build_features(runs, bundle.baseline)
    return bundle.predict(feat[feat["is_future"] == 1])


def test_predict_blends_when_some_odds_missing(trained):
    """1頭だけオッズが無いとき (当日取消が出馬表に未反映など)、その馬は勝率0、ほかの馬はオッズとのブレンドを使う."""
    runs, bundle = trained
    bundle = dataclasses.replace(bundle, blend=(0.6, 0.8))
    full = _predict_future(bundle, runs)
    race_ids = sorted(full["race_id"].unique())
    partial_race, no_odds_race = race_ids[0], race_ids[1]

    tampered = runs.copy()
    in_race = tampered.index[tampered["race_id"] == partial_race]
    gone = in_race[len(in_race) // 2]
    tampered.loc[gone, "win_odds"] = np.nan
    tampered.loc[tampered["race_id"] == no_odds_race, "win_odds"] = np.nan
    part = _predict_future(bundle, tampered)

    assert np.allclose(part.groupby("race_id")["p_win"].sum(), 1.0)
    assert part.loc[part["race_id"] == partial_race, "partial_market"].all()
    assert not part.loc[part["race_id"] != partial_race, "partial_market"].any()
    # オッズの無い馬は取消とみなして0、残りはオッズのある馬どうしのブレンドで配分する
    assert part.loc[gone, "p_win"] == 0.0
    assert part.loc[gone, "p_place"] < 1e-6
    others = in_race.drop(gone)
    assert part.loc[others, "p_win"].sum() == pytest.approx(1.0)
    rel_part = part.loc[others, "p_win"] / part.loc[others, "p_win"].sum()
    rel_full = full.loc[others, "p_win"] / full.loc[others, "p_win"].sum()
    assert np.allclose(rel_part, rel_full)
    assert not np.allclose(part.loc[others, "p_win"], part.loc[others, "p_model"])
    # オッズが1頭も無いレースはモデル単独
    no_odds = part["race_id"] == no_odds_race
    assert np.allclose(part.loc[no_odds, "p_win"], part.loc[no_odds, "p_model"])
    # ほかのレースは変わらない
    rest = ~part["race_id"].isin([partial_race, no_odds_race])
    assert np.allclose(part.loc[rest, "p_win"], full.loc[rest, "p_win"])
    assert (part["p_place"] >= part["p_win"] - 1e-9).all()


def _noise_races(seed, n_races=300, size=12):
    """市場の勝率どおりに勝ち馬が決まり、モデルのスコアはただの乱数という検証データ."""
    rng = np.random.default_rng(seed)
    race_ids = np.repeat([f"R{i:04d}" for i in range(n_races)], size)
    p_market = np.concatenate([rng.dirichlet(np.full(size, 1.5)) for _ in range(n_races)])
    won = np.zeros(n_races * size)
    for i in range(n_races):
        won[i * size + rng.choice(size, p=p_market[i * size:(i + 1) * size])] = 1
    va = pd.DataFrame({"race_id": race_ids, "won": won, "p_market": p_market})
    return va, rng.normal(0, 1, len(va))


def test_cross_fit_metrics_do_not_flatter_useless_model():
    """同じ期間で当てはめて測るとブレンドは必ず市場以上に見えるが、交差検証ではそうならない."""
    gaps = []
    for seed in range(4):
        va, score = _noise_races(seed)
        race_ids, won, p_market = va["race_id"].to_numpy(), va["won"].to_numpy(), va["p_market"].to_numpy()
        tau = model.fit_tau(score, won, race_ids)
        p_model = model.race_softmax(score / tau, race_ids)
        a, b = model.fit_blend(p_model, p_market, won, race_ids)
        in_sample = model.winner_nll(model.race_softmax(a * np.log(p_model) + b * np.log(p_market), race_ids),
                                     won, race_ids)
        assert in_sample <= model.winner_nll(p_market, won, race_ids) + 1e-9

        cv = model.cross_fit_metrics(score, va, blend=True)
        assert set(cv) == METRIC_KEYS
        assert cv["races"] == 300
        assert cv["nll_market"] == round(model.winner_nll(p_market, won, race_ids), 4)
        assert cv["nll_blend"] > in_sample
        gaps.append(cv["nll_blend"] - cv["nll_market"])
    assert np.mean(gaps) > 0


def test_valid_metrics_are_cross_fitted(trained):
    runs, bundle = trained
    valid = bundle.metrics["valid"]
    assert set(valid) == METRIC_KEYS
    feat = features.build_features(runs, bundle.baseline)
    rows = model.training_rows(feat)
    va = rows[(rows["date"] > pd.Timestamp(bundle.train_end)) & (rows["date"] <= pd.Timestamp(bundle.valid_end))]
    score = bundle.score(va)
    # τ・ブレンドは検証期間全体で当てはめ、報告する精度は交差検証の値
    assert bundle.tau == pytest.approx(model.fit_tau(score, va["won"].to_numpy(), va["race_id"].to_numpy()))
    assert valid == model.cross_fit_metrics(score, va, blend=bundle.blend is not None)
    in_sample = model.evaluate(bundle, va)
    assert valid["races"] == in_sample["races"]
    assert valid["nll_market"] == in_sample["nll_market"]  # 市場単独は当てはめる値が無いので同じ
