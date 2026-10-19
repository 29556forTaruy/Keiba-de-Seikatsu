"""バックテスト: 払戻データが欠けたレースの扱い."""

import sqlite3
from datetime import timedelta

import pandas as pd
import pytest

from keiba import backtest, db, features, model


@pytest.fixture(scope="module")
def copied(demo_db, tmp_path_factory):
    """払戻を消すので、共有のデモDBではなくコピーを使う."""
    path, upcoming = demo_db
    copy = tmp_path_factory.mktemp("backtest") / "demo.sqlite"
    src, dst = db.connect(path), sqlite3.connect(copy)
    src.backup(dst)
    src.close()
    dst.close()
    conn = db.connect(copy)
    runs = features.load_runs(conn)
    dates = sorted(runs.loc[runs["is_future"] == 0, "date"].unique())
    train_end, valid_end = dates[int(len(dates) * 0.6)], dates[int(len(dates) * 0.85)]
    bundle = model.train(runs, train_end=str(train_end.date()), valid_end=str(valid_end.date()))
    yield conn, runs, bundle, valid_end, upcoming
    conn.close()


def _by_name(res):
    return {s["name"]: s for s in res["strategies"]}


def test_races_without_win_payout_are_skipped(copied):
    """単勝の払戻が無いレースは外れとして数えず、全戦略から除いて警告する."""
    conn, runs, bundle, valid_end, upcoming = copied
    start, end = (valid_end + timedelta(days=1)).date(), upcoming
    before = backtest.run_backtest(conn, bundle, str(start), str(end), {"bankroll": 100_000}, runs=runs)
    assert not before["warnings"]

    period = runs[(runs["is_future"] == 0) & (runs["field_size"] >= 5)
                  & (runs["date"] >= pd.Timestamp(start)) & (runs["date"] <= pd.Timestamp(end))]
    race_ids = sorted(period["race_id"].unique())
    dropped = race_ids[::3]
    conn.executemany("DELETE FROM payouts WHERE race_id = ?", [(r,) for r in dropped])
    conn.commit()
    after = backtest.run_backtest(conn, bundle, str(start), str(end), {"bankroll": 100_000}, runs=runs)

    assert after["races"] == before["races"] - len(dropped)
    assert any(f"{len(dropped)} レース" in w for w in after["warnings"])
    # 1番人気の単勝: 除いたレースの分だけ購入・的中が減る (勝っていたのに外れとして数えない)
    fav = period[(period["popularity"] == 1) & period["race_id"].isin(dropped)]
    fav_wins = int((fav["finish_pos"] == 1).sum())
    assert fav_wins > 0
    b, a = _by_name(before)["1番人気の単勝"], _by_name(after)["1番人気の単勝"]
    assert a["bets"] == b["bets"] - fav["race_id"].nunique()
    assert a["hits"] == b["hits"] - fav_wins
    # ケリーの資金推移も除いたレースを飛ばす
    assert len(after["equity"]) == after["races"] + 1

    conn.executemany("DELETE FROM payouts WHERE race_id = ?", [(r,) for r in race_ids])
    conn.commit()
    with pytest.raises(ValueError, match="払戻"):
        backtest.run_backtest(conn, bundle, str(start), str(end), {"bankroll": 100_000}, runs=runs)

