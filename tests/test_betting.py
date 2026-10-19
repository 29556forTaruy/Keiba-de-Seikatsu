import math

import numpy as np
import pytest

from keiba import betting


P = [0.4, 0.25, 0.15, 0.1, 0.06, 0.04]
NUMS = [1, 2, 3, 4, 5, 6]


def test_harville_distributions_sum_to_one():
    ex = betting.exacta_matrix(P)
    tri = betting.trifecta_tensor(P)
    assert ex.sum() == pytest.approx(1.0)
    assert tri.sum() == pytest.approx(1.0)
    # 1着確率は元の勝率に一致する
    assert np.allclose(ex.sum(axis=1), P)
    probs = betting.finish_probs(P)
    assert probs["top3"].sum() == pytest.approx(3.0)
    assert probs["top2"].sum() == pytest.approx(2.0)
    assert all(probs["top3"] >= probs["top2"])


def test_combo_probs_consistent():
    combos = betting.combo_probs(P, NUMS)
    assert sum(combos["馬連"].values()) == pytest.approx(1.0)
    assert sum(combos["三連複"].values()) == pytest.approx(1.0)
    assert sum(combos["三連単"].values()) == pytest.approx(1.0)
    # ワイドは3頭から2頭を選ぶので合計は3
    assert sum(combos["ワイド"].values()) == pytest.approx(3.0)
    assert combos["馬連"]["1-2"] == pytest.approx(combos["馬単"]["1-2"] + combos["馬単"]["2-1"])


def test_market_probs_and_kelly():
    p = betting.market_probs([2.0, 4.0, None])
    assert p[2] == 0 and p.sum() == pytest.approx(1.0)
    assert betting.kelly(0.5, 3.0) == pytest.approx(0.25)
    assert betting.kelly(0.2, 3.0) == 0.0
    assert math.isinf(betting.fair_odds(0.0))


def test_suggest_bets_respects_budget_and_skips_missing_odds():
    horses = [
        {"number": 1, "p_win": 0.5, "win_odds": 3.0, "p_place": 0.8, "place_odds_min": 1.6},
        {"number": 2, "p_win": 0.3, "win_odds": 5.0, "p_place": None, "place_odds_min": None},
        {"number": 3, "p_win": 0.2, "win_odds": float("nan"), "p_place": 0.5, "place_odds_min": None},
    ]
    bets = betting.suggest_bets(horses, bankroll=100_000, race_budget_ratio=0.05, max_bet_ratio=0.05)
    assert bets, "期待値の高い馬券が提案されるはず"
    assert sum(b["stake"] for b in bets) <= 5_000
    assert all(b["stake"] % 100 == 0 for b in bets)
    assert all(b["combination"] != "3" for b in bets)
    capped = betting.suggest_bets(horses, bankroll=10_000_000, max_stake=1_000)
    assert all(b["stake"] <= 1_000 for b in capped)
