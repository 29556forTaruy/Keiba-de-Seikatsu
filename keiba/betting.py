"""確率から馬券の期待値・購入額を計算する.

単勝確率から他の券種の確率は Harville モデルで近似する:
  P(i が1着, j が2着) = p_i * p_j / (1 - p_i)
人気薄の2・3着確率をやや過小評価する癖があるので、目安として使う。
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from keiba.constants import TAKEOUT


def normalize(p: Sequence[float]) -> np.ndarray:
    arr = np.clip(np.asarray(p, dtype=float), 1e-9, None)
    return arr / arr.sum()


def market_probs(odds: Sequence[float | None]) -> np.ndarray:
    """単勝オッズ → 控除率を取り除いた市場の勝率."""
    inv = np.array([1.0 / o if o and o > 0 else 0.0 for o in odds], dtype=float)
    if inv.sum() == 0:
        return np.full(len(inv), np.nan)
    return inv / inv.sum()


def exacta_matrix(p: Sequence[float]) -> np.ndarray:
    """M[i, j] = P(i が1着, j が2着)"""
    p = normalize(p)
    m = np.outer(p, p) / (1.0 - p)[:, None]
    np.fill_diagonal(m, 0.0)
    return m


def trifecta_tensor(p: Sequence[float]) -> np.ndarray:
    """T[i, j, k] = P(i, j, k の順で入線)"""
    p = normalize(p)
    n = len(p)
    ex = exacta_matrix(p)
    rest = 1.0 - p[:, None] - p[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = ex[:, :, None] * p[None, None, :] / rest[:, :, None]
    idx = np.arange(n)
    t[idx, idx, :] = 0
    t[idx, :, idx] = 0
    t[:, idx, idx] = 0
    return np.nan_to_num(t, nan=0.0, posinf=0.0)


def finish_probs(p: Sequence[float]) -> dict[str, np.ndarray]:
    """各馬の 1着・2着以内・3着以内 の確率."""
    p = normalize(p)
    ex = exacta_matrix(p)
    tri = trifecta_tensor(p)
    second = ex.sum(axis=0)
    third = tri.sum(axis=(0, 1))
    return {"win": p, "top2": p + second, "top3": p + second + third}


def place_slots(field_size: int) -> int:
    """複勝の対象着順 (JRA: 7頭以下は2着まで)."""
    return 2 if field_size <= 7 else 3


def combo_probs(p: Sequence[float], numbers: Sequence[int]) -> dict[str, dict[str, float]]:
    """券種ごとの組番 → 的中確率."""
    p = normalize(p)
    n = len(p)
    ex = exacta_matrix(p)
    tri = trifecta_tensor(p)
    nums = list(numbers)
    out: dict[str, dict[str, float]] = {"馬連": {}, "馬単": {}, "ワイド": {}, "三連複": {}, "三連単": {}}
    for i, j in itertools.permutations(range(n), 2):
        out["馬単"][f"{nums[i]}-{nums[j]}"] = ex[i, j]
    for i, j in itertools.combinations(range(n), 2):
        key = "-".join(str(x) for x in sorted((nums[i], nums[j])))
        out["馬連"][key] = ex[i, j] + ex[j, i]
        # 2頭とも3着以内: 3頭の並びのうち i, j を含むもの
        both = tri[i, j, :].sum() + tri[j, i, :].sum() + tri[i, :, j].sum() + tri[j, :, i].sum()
        both += tri[:, i, j].sum() + tri[:, j, i].sum()
        out["ワイド"][key] = both
    for i, j, k in itertools.combinations(range(n), 3):
        key = "-".join(str(x) for x in sorted((nums[i], nums[j], nums[k])))
        out["三連複"][key] = sum(tri[a, b, c] for a, b, c in itertools.permutations((i, j, k)))
    for i, j, k in itertools.permutations(range(n), 3):
        out["三連単"][f"{nums[i]}-{nums[j]}-{nums[k]}"] = tri[i, j, k]
    return out


def kelly(p: float, odds: float) -> float:
    """ケリー基準の賭け比率 (オッズは払戻倍率)."""
    if odds <= 1 or p <= 0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


def fair_odds(p: float, takeout: float = 0.0) -> float:
    return math.inf if p <= 0 else (1.0 - takeout) / p


def suggest_bets(
    horses: list[dict[str, Any]],
    *,
    bankroll: float,
    kelly_fraction: float = 0.25,
    min_ev: float = 1.15,
    min_prob: float = 0.02,
    max_bet_ratio: float = 0.05,
    race_budget_ratio: float = 0.10,
    unit: int = 100,
    max_stake: float | None = None,
    **_: Any,
) -> list[dict[str, Any]]:
    """単勝・複勝の期待値が閾値を超える馬に、分数ケリーで購入額を割り当てる.

    horses: number, horse_name, p_win, p_place(複勝圏に入る確率), win_odds, place_odds_min を持つ辞書
    """
    candidates = []
    for h in horses:
        for bet_type, p, odds in (
            ("単勝", h.get("p_win"), h.get("win_odds")),
            ("複勝", h.get("p_place"), h.get("place_odds_min")),
        ):
            if p is None or odds is None or not np.isfinite(p) or not np.isfinite(odds) or odds <= 1:
                continue
            ev = p * odds
            if ev < min_ev or p < min_prob:
                continue
            f = kelly(p, odds) * kelly_fraction
            stake = min(f, max_bet_ratio) * bankroll
            if max_stake:
                stake = min(stake, max_stake)
            candidates.append({
                "type": bet_type,
                "combination": str(h["number"]),
                "horse_name": h.get("horse_name"),
                "prob": float(p),
                "odds": float(odds),
                "ev": float(ev),
                "stake": stake,
            })

    budget = race_budget_ratio * bankroll
    total = sum(c["stake"] for c in candidates)
    scale = min(1.0, budget / total) if total > 0 else 1.0
    bets = []
    for c in sorted(candidates, key=lambda c: -c["ev"]):
        stake = int(c["stake"] * scale // unit * unit)
        if stake >= unit:
            bets.append({**c, "stake": stake})
    return bets


def top_combos(
    p: Sequence[float], numbers: Sequence[int], bet_type: str, k: int = 5, min_ev: float = 1.0
) -> list[dict[str, Any]]:
    """確率上位の組番と「このオッズ以上なら期待値プラス」の目安."""
    probs = combo_probs(p, numbers)[bet_type]
    best = sorted(probs.items(), key=lambda kv: -kv[1])[:k]
    return [
        {
            "combination": combo,
            "prob": float(pr),
            "min_odds": float(min_ev / pr) if pr > 0 else math.inf,
            "fair_market_odds": fair_odds(pr, TAKEOUT[bet_type]),
        }
        for combo, pr in best
    ]
