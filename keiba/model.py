"""勝率モデル.

1. 勾配ブースティング (scikit-learn HistGradientBoosting) で「1着になるか」を学習
2. 検証期間で、レース内ソフトマックスの温度 τ を最尤推定して勝率に変換
3. オッズがあれば、モデル勝率と市場勝率を対数線形にブレンド (Benter 方式)
     p ∝ exp(a·log p_model + b·log p_market)
   市場はかなり賢いので、モデル単独より大抵よくなる。
4. 検証期間の精度 (metrics["valid"]) は、τ とブレンド係数をレース単位の交差検証で
   当て直して測る (同じ期間で当てはめて測ると、当てはめた分だけ良く見えるため)。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.ensemble import HistGradientBoostingClassifier

from keiba import features as F
from keiba.betting import finish_probs, place_slots

log = logging.getLogger(__name__)


def race_softmax(score: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    s = pd.Series(score)
    s = s - s.groupby(race_ids).transform("max")
    e = np.exp(s)
    return (e / e.groupby(race_ids).transform("sum")).to_numpy()


def winner_nll(p: np.ndarray, won: np.ndarray, race_ids: np.ndarray) -> float:
    """1レースあたりの勝ち馬の負の対数尤度 (小さいほどよい)."""
    mask = won == 1
    n_races = len(np.unique(race_ids[mask]))
    return float(-np.log(np.clip(p[mask], 1e-12, None)).sum() / max(n_races, 1))


def _top1_hit(p: np.ndarray, won: np.ndarray, race_ids: np.ndarray) -> float:
    df = pd.DataFrame({"p": p, "won": won, "race": race_ids})
    top = df.loc[df.groupby("race")["p"].idxmax()]
    return float(top["won"].mean())


@dataclass
class ModelBundle:
    version: str
    clf: HistGradientBoostingClassifier
    baseline: F.SpeedBaseline
    tau: float = 1.0
    blend: tuple[float, float] | None = None
    features: list[str] = field(default_factory=lambda: list(F.FEATURES))
    metrics: dict[str, Any] = field(default_factory=dict)
    train_end: str = ""
    valid_end: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    # ------------------------------------------------------------ 推論
    def score(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.clf.predict_proba(F.model_matrix(df, self.features))[:, 1]
        proba = np.clip(proba, 1e-6, 1 - 1e-6)
        return np.log(proba / (1 - proba))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """特徴量付きの出走表 → 勝率など. df は build_features の出力.

        一部の馬だけオッズが無いレース (当日の取消が出馬表に反映されていないときなど) は、
        オッズの無い馬にモデルの勝率を残し、残りの確率をオッズのある馬どうしのブレンドで配分する。
        partial_market 列でそのレースを示す。オッズが1頭も無いレースはモデルの勝率だけを使う。
        """
        out = df[["race_id", "horse_id", "horse_name", "number", "win_odds", "p_market"]].copy()
        race_ids = df["race_id"].to_numpy()
        race = out["race_id"]
        out["score"] = self.score(df)
        out["p_model"] = race_softmax(out["score"].to_numpy() / self.tau, race_ids)
        out["p_win"] = out["p_model"]
        has = out["p_market"].notna()
        n_has = has.groupby(race).transform("sum")
        any_mkt = (n_has > 0).to_numpy()
        out["partial_market"] = any_mkt & (n_has < race.groupby(race).transform("size")).to_numpy()
        if self.blend is not None:
            a, b = self.blend
            logit = a * np.log(out["p_model"].clip(1e-9)) + b * np.log(out["p_market"].clip(1e-9).fillna(1.0))
            # オッズの無い馬はブレンドから外す (オッズが1頭も無いレースは下で使わないので 0 のまま)
            logit = np.where(has, logit, np.where(any_mkt, -np.inf, 0.0))
            blended = race_softmax(logit, race_ids)
            missing = out["p_model"].where(~has, 0.0).groupby(race).transform("sum").to_numpy()
            mixed = np.where(has, blended * (1.0 - missing), out["p_model"])
            out["p_win"] = np.where(any_mkt, mixed, out["p_model"])
        out["p_top2"] = np.nan
        out["p_top3"] = np.nan
        out["p_place"] = np.nan
        for _, idx in out.groupby("race_id").groups.items():
            probs = finish_probs(out.loc[idx, "p_win"].to_numpy())
            out.loc[idx, "p_top2"] = probs["top2"]
            out.loc[idx, "p_top3"] = probs["top3"]
            out.loc[idx, "p_place"] = probs["top2" if place_slots(len(idx)) == 2 else "top3"]
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        return joblib.load(path)


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["is_future"] == 0) & (df["field_size"] >= 5)]


def fit_tau(score: np.ndarray, won: np.ndarray, race_ids: np.ndarray) -> float:
    res = minimize_scalar(
        lambda t: winner_nll(race_softmax(score / t, race_ids), won, race_ids),
        bounds=(0.2, 5.0), method="bounded",
    )
    return float(res.x)


def fit_blend(p_model: np.ndarray, p_market: np.ndarray, won: np.ndarray, race_ids: np.ndarray) -> tuple[float, float]:
    lm, lk = np.log(np.clip(p_model, 1e-9, None)), np.log(np.clip(p_market, 1e-9, None))

    def loss(ab: np.ndarray) -> float:
        return winner_nll(race_softmax(ab[0] * lm + ab[1] * lk, race_ids), won, race_ids)

    res = minimize(loss, x0=np.array([0.5, 0.5]), method="L-BFGS-B", bounds=[(0, 3), (0, 3)])
    return float(res.x[0]), float(res.x[1])


def _metrics(p_model: np.ndarray, p_market: np.ndarray, p_blend: np.ndarray, won: np.ndarray,
             race_ids: np.ndarray) -> dict[str, Any]:
    m: dict[str, Any] = {"races": int(len(np.unique(race_ids)))}
    for name, p in (("model", p_model), ("market", p_market), ("blend", p_blend)):
        m[f"nll_{name}"] = round(winner_nll(p, won, race_ids), 4)
        m[f"top1_{name}"] = round(_top1_hit(p, won, race_ids), 4)
    return m


def evaluate(bundle: ModelBundle, df: pd.DataFrame) -> dict[str, Any]:
    """確定レースでの精度. 市場(オッズ)単独との比較つき."""
    rows = training_rows(df)
    rows = rows[rows["p_market"].notna().groupby(rows["race_id"]).transform("all")]
    if rows.empty:
        return {}
    pred = bundle.predict(rows)
    return _metrics(pred["p_model"].to_numpy(), rows["p_market"].to_numpy(), pred["p_win"].to_numpy(),
                    rows["won"].to_numpy(), rows["race_id"].to_numpy())


CV_FOLDS = 5


def cross_fit_metrics(score: np.ndarray, va: pd.DataFrame, *, blend: bool, folds: int = CV_FOLDS) -> dict[str, Any]:
    """検証期間の精度を、τ とブレンド係数をレース単位の K 分割で当て直して測る.

    τ・係数を検証期間全体で決めてから同じ期間で測ると、当てはめた分だけ良く見える
    (ブレンドは市場単独を含むので、モデルに力が無くても市場より悪くならない)。
    各分割を、ほかの分割で決めた τ・係数で予測して測る。キーは evaluate と同じ。
    """
    race_ids = va["race_id"].to_numpy()
    won = va["won"].to_numpy()
    p_market = va["p_market"].to_numpy()
    mkt_ok = va["p_market"].notna().groupby(va["race_id"]).transform("all").to_numpy()
    if not mkt_ok.any():
        return {}
    races = np.unique(race_ids)
    fold = pd.Series(np.arange(len(races)) % folds, index=races).reindex(race_ids).to_numpy()
    lk = np.log(np.clip(p_market, 1e-9, None))
    p_model = np.full(len(va), np.nan)
    p_blend = np.full(len(va), np.nan)
    for k in range(folds):
        te, tr = fold == k, fold != k
        if not te.any() or not tr.any():
            continue
        tau = fit_tau(score[tr], won[tr], race_ids[tr])
        p_model[te] = race_softmax(score[te] / tau, race_ids[te])
        p_blend[te] = p_model[te]
        fm, tm = tr & mkt_ok, te & mkt_ok
        if blend and fm.any() and tm.any():
            a, b = fit_blend(race_softmax(score[fm] / tau, race_ids[fm]), p_market[fm], won[fm], race_ids[fm])
            p_blend[tm] = race_softmax(a * np.log(np.clip(p_model[tm], 1e-9, None)) + b * lk[tm], race_ids[tm])
    ok = mkt_ok & ~np.isnan(p_model)
    if not ok.any():
        return {}
    return _metrics(p_model[ok], p_market[ok], p_blend[ok], won[ok], race_ids[ok])


def train(
    df: pd.DataFrame,
    *,
    train_end: str,
    valid_end: str,
    version: str | None = None,
    random_state: int = 0,
) -> ModelBundle:
    """train_end までで学習し、(train_end, valid_end] で温度とブレンドを調整する.

    df は load_runs の出力 (特徴量はこの関数内で作る)。
    """
    train_end_ts, valid_end_ts = pd.Timestamp(train_end), pd.Timestamp(valid_end)
    if valid_end_ts <= train_end_ts:
        raise ValueError("valid_end は train_end より後にしてください")
    # 未確定の出馬表 (結果を取り込んでいない過去の出馬表を含む) は学習に要らず、近走の特徴量を乱すので除く
    df = df[df["is_future"] == 0]
    baseline = F.SpeedBaseline().fit(df[df["date"] <= train_end_ts])
    feat = F.build_features(df, baseline)
    rows = training_rows(feat)
    tr = rows[rows["date"] <= train_end_ts]
    va = rows[(rows["date"] > train_end_ts) & (rows["date"] <= valid_end_ts)]
    if tr["race_id"].nunique() < 100:
        raise ValueError(f"学習データが少なすぎます ({tr['race_id'].nunique()}レース)。もっと過去データを集めてください")
    if va["race_id"].nunique() < 30:
        raise ValueError(f"検証データが少なすぎます ({va['race_id'].nunique()}レース)。valid_end を延ばしてください")

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=600,
        max_leaf_nodes=31,
        min_samples_leaf=80,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=40,
        random_state=random_state,
    )
    clf.fit(F.model_matrix(tr), tr["won"].astype(int))

    bundle = ModelBundle(
        version=version or datetime.now().strftime("v%Y%m%d-%H%M"),
        clf=clf,
        baseline=baseline,
        train_end=str(train_end_ts.date()),
        valid_end=str(valid_end_ts.date()),
    )
    race_ids = va["race_id"].to_numpy()
    won = va["won"].to_numpy()
    score = bundle.score(va)
    bundle.tau = fit_tau(score, won, race_ids)
    mkt_ok = va["p_market"].notna().groupby(va["race_id"]).transform("all").to_numpy()
    if mkt_ok.sum() > 0 and va.loc[mkt_ok, "race_id"].nunique() >= 30:
        p_model = race_softmax(score / bundle.tau, race_ids)
        bundle.blend = fit_blend(p_model[mkt_ok], va["p_market"].to_numpy()[mkt_ok], won[mkt_ok], race_ids[mkt_ok])
    bundle.metrics = {
        "train_races": int(tr["race_id"].nunique()),
        "valid_races": int(va["race_id"].nunique()),
        "iterations": int(clf.n_iter_),
        "tau": round(bundle.tau, 4),
        "blend": None if bundle.blend is None else [round(x, 4) for x in bundle.blend],
        # τ・ブレンドは検証期間全体で決めるが、精度は交差検証で測る (同じデータで測ると甘く出る)
        "valid": cross_fit_metrics(score, va, blend=bundle.blend is not None),
    }
    return bundle
