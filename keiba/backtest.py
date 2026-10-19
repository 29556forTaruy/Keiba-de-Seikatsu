"""過去期間での回収率シミュレーション.

注意: 過去データのオッズは「確定オッズ」で、実際に馬券を買う時点のオッズとは
ずれる。締切直前に買えば近いが、バックテストの回収率は実戦より少し甘く出る。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from keiba import db
from keiba import features as F
from keiba.betting import suggest_bets
from keiba.model import ModelBundle, training_rows


@dataclass
class Strategy:
    name: str
    description: str
    bets: int = 0
    hits: int = 0
    stake: float = 0.0
    payout: float = 0.0
    curve: list[float] = field(default_factory=list)
    monthly: dict[str, list[float]] = field(default_factory=dict)

    def add(self, month: str, stake: float, payout: float) -> None:
        if stake <= 0:
            return
        self.bets += 1
        self.hits += int(payout > 0)
        self.stake += stake
        self.payout += payout
        self.curve.append(self.payout - self.stake)
        m = self.monthly.setdefault(month, [0.0, 0.0])
        m[0] += stake
        m[1] += payout

    def summary(self) -> dict[str, Any]:
        curve = np.array(self.curve) if self.curve else np.zeros(1)
        peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))
        drawdown = float((peak[1:] - curve).max()) if self.curve else 0.0
        return {
            "name": self.name,
            "description": self.description,
            "bets": self.bets,
            "hits": self.hits,
            "hit_rate": self.hits / self.bets if self.bets else 0.0,
            "stake": self.stake,
            "payout": self.payout,
            "profit": self.payout - self.stake,
            "roi": self.payout / self.stake if self.stake else 0.0,
            "max_drawdown": drawdown,
            "curve": [round(x) for x in self.curve],
            "monthly": [
                {"month": k, "stake": v[0], "payout": v[1], "roi": v[1] / v[0] if v[0] else 0.0}
                for k, v in sorted(self.monthly.items())
            ],
        }


def _payout_lookup(conn: sqlite3.Connection, race_ids: list[str]) -> dict[tuple[str, str, str], int]:
    out: dict[tuple[str, str, str], int] = {}
    for chunk in range(0, len(race_ids), 500):
        ids = race_ids[chunk:chunk + 500]
        q = f"SELECT race_id, bet_type, combination, payout FROM payouts WHERE race_id IN ({','.join('?' * len(ids))})"
        for r in conn.execute(q, ids).fetchall():
            out[(r[0], r[1], r[2])] = r[3]
    return out


def calibration_table(p: np.ndarray, won: np.ndarray) -> list[dict[str, Any]]:
    edges = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{lo:.0%}–{hi:.0%}",
            "count": int(m.sum()),
            "predicted": float(p[m].mean()),
            "actual": float(won[m].mean()),
        })
    return rows


def run_backtest(
    conn: sqlite3.Connection,
    bundle: ModelBundle,
    start: str,
    end: str,
    betting: dict[str, Any],
    runs: pd.DataFrame | None = None,
) -> dict[str, Any]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    warnings = []
    if bundle.valid_end and pd.Timestamp(bundle.valid_end) >= start_ts:
        warnings.append(
            f"モデルの学習・検証期間 (〜{bundle.valid_end}) とテスト期間が重なっています。"
            "回収率が実力以上に良く見えるので、start を検証期間より後にしてください。"
        )
    runs = F.load_runs(conn) if runs is None else runs
    feat = F.build_features(runs, bundle.baseline)
    rows = training_rows(feat)
    rows = rows[(rows["date"] >= start_ts) & (rows["date"] <= end_ts)]
    if rows.empty:
        raise ValueError("指定期間に確定レースがありません")
    pred = bundle.predict(rows)
    pred["date"] = rows["date"].to_numpy()
    pred["finish_pos"] = rows["finish_pos"].to_numpy()
    pred["popularity"] = rows["popularity"].to_numpy()
    pred["won"] = rows["won"].to_numpy()
    pays = _payout_lookup(conn, sorted(pred["race_id"].unique()))
    # 単勝の払戻が1件も無いレースは精算できない (払戻表の取得・解析漏れ)。外れ扱いにすると
    # 1番人気を含む全戦略の回収率が下がるので、ケリーの資金推移も含めて全戦略から除く
    paid = {race_id for race_id, bet_type, _ in pays if bet_type == "単勝"}
    unpaid = int(pred.loc[~pred["race_id"].isin(paid), "race_id"].nunique())
    if unpaid:
        warnings.append(
            f"単勝の払戻データが無い {unpaid} レースを集計から除きました。"
            "払戻表の取得漏れの可能性があります (keiba collect results --refresh で取り直せます)。"
        )
        pred = pred[pred["race_id"].isin(paid)]
        if pred.empty:
            raise ValueError(f"指定期間の確定レース ({unpaid}レース) に単勝の払戻データがありません")

    min_ev = betting.get("min_ev", 1.15)
    min_prob = betting.get("min_prob", 0.02)
    s_fav = Strategy("1番人気の単勝", "比較用: 毎レース1番人気を100円")
    s_top = Strategy("本命の単勝", "毎レース予想勝率1位を100円")
    s_place = Strategy("本命の複勝", "毎レース予想勝率1位の複勝を100円")
    s_ev = Strategy("期待値の単勝", f"勝率×オッズ≥{min_ev} かつ 勝率≥{min_prob:.0%} の馬を100円ずつ")
    s_kelly = Strategy("ケリー資金管理", f"期待値の単勝を分数ケリー({betting.get('kelly_fraction', 0.25)})で賭ける")
    bankroll = float(betting.get("bankroll", 100_000))
    equity = [bankroll]

    for race_id, g in pred.sort_values(["date", "race_id"]).groupby("race_id", sort=False):
        month = g["date"].iloc[0].strftime("%Y-%m")
        win_pay = lambda num: pays.get((race_id, "単勝", str(num)), 0)  # noqa: E731
        fav = g[g["popularity"] == 1]
        if not fav.empty:
            s_fav.add(month, 100, win_pay(int(fav["number"].iloc[0])))
        top = g.loc[g["p_win"].idxmax()]
        s_top.add(month, 100, win_pay(int(top["number"])))
        s_place.add(month, 100, pays.get((race_id, "複勝", str(int(top["number"]))), 0))
        for _, h in g.iterrows():
            if h["win_odds"] and h["p_win"] * h["win_odds"] >= min_ev and h["p_win"] >= min_prob:
                s_ev.add(month, 100, win_pay(int(h["number"])))
        horses = [
            {"number": int(h.number), "p_win": float(h.p_win), "win_odds": h.win_odds, "p_place": None}
            for h in g.itertuples()
        ]
        for bet in suggest_bets(horses, **{**betting, "bankroll": bankroll}):
            got = win_pay(bet["combination"]) / 100 * bet["stake"]
            s_kelly.add(month, bet["stake"], got)
            bankroll += got - bet["stake"]
        equity.append(bankroll)

    strategies = [s.summary() for s in (s_top, s_place, s_ev, s_kelly, s_fav)]
    return {
        "start": str(start_ts.date()),
        "end": str(end_ts.date()),
        "races": int(pred["race_id"].nunique()),
        "model_version": bundle.version,
        "model_metrics": bundle.metrics,
        "warnings": warnings,
        "strategies": strategies,
        "calibration": calibration_table(pred["p_win"].to_numpy(), pred["won"].to_numpy()),
        "equity": [round(x) for x in equity],
        "final_bankroll": bankroll,
        "start_bankroll": float(betting.get("bankroll", 100_000)),
    }
