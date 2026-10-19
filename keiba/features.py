"""特徴量の作成.

原則: あるレースの特徴量は「そのレースより前の日付」の情報だけから作る。
  - 馬の過去成績は馬ごとに日付順に並べて shift() で1走前以前だけを見る
  - 騎手・調教師・種牡馬の成績は日別集計の累積から「前日まで」を引く
過去レース(確定済み)と出馬表(未確定)を1つの表にして同じ処理を通すので、
学習時と予想時で特徴量の作り方がずれない。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from keiba import db
from keiba.constants import GOINGS, SURFACES, VENUES, distance_band

CATEGORIES = {
    "venue": list(VENUES.values()),
    "surface": list(SURFACES),
    "going": list(GOINGS),
    "sex": ["牡", "牝", "セ"],
}

RACE_JOIN = """
    ra.date, ra.venue, ra.race_num, ra.race_name, ra.grade, ra.race_class, ra.class_level,
    ra.surface, ra.distance, ra.direction, ra.going, ra.weather, ra.field_size, ra.source
"""


# 成績・オッズ由来で、データによっては存在しない列
OPTIONAL_COLS = [
    "finish_pos", "finish_status", "time_sec", "margin", "passing", "last3f", "odds", "win_odds",
    "popularity", "place_odds_min", "place_odds_max", "prize", "horse_weight", "weight_diff",
]
NUMERIC_COLS = [
    "finish_pos", "time_sec", "last3f", "odds", "win_odds", "popularity", "place_odds_min", "place_odds_max",
    "prize", "horse_weight", "weight_diff", "weight_carried", "frame", "number", "age", "distance",
    "class_level", "field_size", "race_num",
]


def load_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    """確定成績と出馬表(未確定レース)をまとめて読み込む."""
    past = db.read_df(conn, f"""
        SELECT r.*, {RACE_JOIN}, r.odds AS win_odds
        FROM results r JOIN races ra USING (race_id)
    """)
    past["is_future"] = 0
    future = db.read_df(conn, f"""
        SELECT e.*, {RACE_JOIN}, o.win_odds, o.place_odds_min, o.place_odds_max, o.popularity
        FROM entries e JOIN races ra USING (race_id)
        LEFT JOIN odds o ON o.race_id = e.race_id AND o.number = e.number
        WHERE e.scratched = 0
          AND e.race_id NOT IN (SELECT DISTINCT race_id FROM results)
    """)
    future["is_future"] = 1
    frames = [f for f in (past, future) if not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    # 成績がまだ無いDBやオッズ未発売のレースでも列がそろうようにし、NULLだけの列も数値型にする
    for col in OPTIONAL_COLS:
        if col not in df:
            df[col] = np.nan
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["finish_status"] = df["finish_status"].astype("object").where(df["finish_status"].notna(), None)
    sires = db.read_df(conn, "SELECT horse_id, sire, sire_id, damsire, damsire_id FROM horses")
    df = df.merge(sires, on="horse_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    # 取消・除外は走っていないので除く
    df = df[~df["finish_status"].isin(["取消", "除外"])].copy()
    # 未確定レースの頭数は出走予定頭数
    n = df.groupby("race_id")["horse_id"].transform("size")
    df["field_size"] = df["field_size"].fillna(n).where(df["is_future"] == 0, n)
    return df.sort_values(["date", "race_id", "number"], kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------- スピード指数

@dataclass
class SpeedBaseline:
    """走破タイムを、コースと馬場から決めた基準タイムに対する偏差値にする.

    基準タイム = コース (競馬場, 芝ダ, 距離) の基準 × 馬場係数 (芝ダ, 馬場, 距離帯)。
    馬場係数は全コースをまとめて推定するので、重・不良の少ないコースでも良馬場の平均とは比べない。
    標準偏差はコースごとの、馬場の差を除いた残差のばらつき。
    (競馬場, 芝ダ, 距離, 馬場) の実測の平均・標準偏差は、件数 n に応じて n/(n+k) の重みで
    この基準に寄せる (件数がしきい値をまたいでも指数が跳ばない)。
    学習期間のデータだけで fit し、テスト期間や予想には transform だけを使う。
    """

    min_count: int = 30
    shrink_k: float = 30.0
    course: list[tuple[list[str], pd.DataFrame]] = field(default_factory=list)
    going: list[tuple[list[str], pd.DataFrame]] = field(default_factory=list)
    cells: pd.DataFrame | None = None

    CELL = ["venue", "surface", "distance", "going"]
    # コースの基準 (件数が足りなければ競馬場をまとめる) と馬場係数 (同じく距離帯をまとめる)
    COURSE_LEVELS = (["venue", "surface", "distance"], ["surface", "distance"])
    GOING_LEVELS = (["surface", "going", "band"], ["surface", "going"])

    def fit(self, df: pd.DataFrame) -> "SpeedBaseline":
        ok = df[df["time_sec"].notna() & df["finish_pos"].notna()].dropna(subset=["venue", "surface", "distance"])
        ok = ok[self.CELL + ["time_sec"]].assign(band=ok["distance"].map(distance_band))
        t = ok["time_sec"]
        # コース基準と馬場係数を交互に当てはめる (コースごとの馬場の偏りが基準に混ざらないように)
        r = pd.Series(1.0, index=ok.index)
        for _ in range(10):
            base = (t / r).groupby([ok[k] for k in self.COURSE_LEVELS[0]]).transform("mean")
            r = (t / base).groupby([ok[k] for k in self.GOING_LEVELS[0]]).transform("mean").fillna(1.0)
        ok["_x"] = t / r  # 馬場の影響を除いたタイム
        self.course = []
        for keys in self.COURSE_LEVELS:
            # 標準偏差は馬場の差を除いた残差から取る (馬場をまぜた標準偏差は大きく出る)
            mu = ok.groupby(keys)["_x"].transform("mean")
            g = ok.assign(_e=t - mu * r).groupby(keys).agg(
                _mu=("_x", "mean"), _sd=("_e", "std"), _n=("_x", "size")).reset_index()
            g = g[(g["_n"] >= self.min_count) & (g["_sd"] > 0)]
            self.course.append((list(keys), g[keys + ["_mu", "_sd"]]))
        ok["_ratio"] = t / base
        self.going = []
        for keys in self.GOING_LEVELS:
            g = ok.groupby(keys)["_ratio"].agg(_r="mean", _n="size").reset_index()
            self.going.append((list(keys), g.loc[g["_n"] >= self.min_count, keys + ["_r"]]))
        self.cells = ok.groupby(self.CELL)["time_sec"].agg(_m="mean", _csd="std", _n="size").reset_index()
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        if self.cells is None:
            return self._transform_legacy(df)
        key = df[self.CELL].assign(band=df["distance"].map(distance_band))
        base = self._lookup(key, self.course, ["_mu", "_sd"])
        mu0 = base["_mu"] * self._lookup(key, self.going, ["_r"])["_r"].fillna(1.0)
        sd0 = base["_sd"]
        # 実測の (競馬場, 芝ダ, 距離, 馬場) は件数に応じて混ぜる
        cell = key[self.CELL].merge(self.cells, on=self.CELL, how="left")
        cell.index = df.index
        n = cell["_n"].fillna(0)
        w = n / (n + self.shrink_k)
        mu = mu0 + w * (cell["_m"] - mu0).fillna(0)
        sd = sd0 + w * (cell["_csd"] - sd0).fillna(0)
        fig = 50 + 10 * (mu - df["time_sec"]) / sd
        return fig.clip(0, 100)

    @staticmethod
    def _lookup(key: pd.DataFrame, levels: list[tuple[list[str], pd.DataFrame]], cols: list[str]) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=key.index, columns=cols)
        for keys, table in levels:
            merged = key[keys].merge(table, on=keys, how="left")
            merged.index = key.index
            out = out.fillna(merged[cols])
        return out

    def _transform_legacy(self, df: pd.DataFrame) -> pd.Series:
        """旧形式 (階層ごとの平均・標準偏差) で保存されたモデル用. 学習時と同じ指数を返す."""
        mu = pd.Series(np.nan, index=df.index)
        sd = pd.Series(np.nan, index=df.index)
        for keys, table in self.__dict__.get("tables", []):
            merged = df[keys].merge(table, on=keys, how="left")
            merged.index = df.index
            mu = mu.fillna(merged["_mu"])
            sd = sd.fillna(merged["_sd"])
        fig = 50 + 10 * (mu - df["time_sec"]) / sd
        return fig.clip(0, 100)


# ---------------------------------------------------------------- 補助

def _exclusive_cumsum(df: pd.DataFrame, keys: list[str] | str, col: str) -> pd.Series:
    """同じキーの「自分より前の行」の合計 (未確定レースの NaN は0扱い)."""
    values = df[col].fillna(0)
    by = [df[k] for k in keys] if isinstance(keys, list) else df[keys]
    return values.groupby(by, sort=False).cumsum() - values


def _prev_roll(df: pd.DataFrame, key: str, col: str, window: int, how: str = "mean") -> pd.Series:
    prev = df.groupby(key, sort=False)[col].shift()
    roll = prev.groupby(df[key], sort=False).rolling(window, min_periods=1)
    out = getattr(roll, how)()
    return out.reset_index(level=0, drop=True).reindex(df.index)


def _entity_rates(
    df: pd.DataFrame, key: pd.Series, prefix: str, *, window_days: int | None, k: float,
    prior_win: float, prior_top3: float,
) -> pd.DataFrame:
    """騎手などの「前日までの」成績 (window_days 日以内) をベイズ平滑化した率で返す."""
    missing = key.isna().to_numpy()
    tmp = pd.DataFrame({
        "_key": key.fillna("__none__").astype(str).values, "date": df["date"].values,
        "n": df["won"].notna().astype(float).values,
        "w": df["won"].fillna(0).values, "t": df["top3"].fillna(0).values,
    })
    daily = tmp.groupby(["_key", "date"], sort=True)[["n", "w", "t"]].sum().reset_index()
    cum = daily.groupby("_key", sort=False)[["n", "w", "t"]].cumsum()
    daily[["cn", "cw", "ct"]] = cum.values
    # その日を含まない累積
    daily["xn"] = daily["cn"] - daily["n"]
    daily["xw"] = daily["cw"] - daily["w"]
    daily["xt"] = daily["ct"] - daily["t"]
    rows = tmp[["_key", "date"]].merge(daily[["_key", "date", "xn", "xw", "xt"]], on=["_key", "date"], how="left")
    n, w, t = rows["xn"].to_numpy(), rows["xw"].to_numpy(), rows["xt"].to_numpy()
    if window_days is not None:
        start = tmp[["_key", "date"]].copy()
        start["date"] = start["date"] - pd.Timedelta(days=window_days)
        start["_row"] = np.arange(len(start))
        start = start.sort_values("date", kind="mergesort")
        right = daily[["_key", "date", "cn", "cw", "ct"]].sort_values("date", kind="mergesort")
        old = pd.merge_asof(start, right, on="date", by="_key", allow_exact_matches=False, direction="backward")
        old = old.sort_values("_row")
        n = n - old["cn"].fillna(0).to_numpy()
        w = w - old["cw"].fillna(0).to_numpy()
        t = t - old["ct"].fillna(0).to_numpy()
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_n"] = n
    out[f"{prefix}_win"] = (w + prior_win * k) / (n + k)
    out[f"{prefix}_top3"] = (t + prior_top3 * k) / (n + k)
    out.loc[missing, [f"{prefix}_n", f"{prefix}_win", f"{prefix}_top3"]] = np.nan
    return out


# 1頭あたりの平均的な勝率・複勝率 (平均出走頭数 約14頭)
PRIOR_WIN, PRIOR_TOP3 = 0.072, 0.215

REL_COLS = ["h_best5_fig", "h_mean3_fig", "h_last_fig", "h_mean3_fin", "h_win_rate", "j_win", "h_prize_log",
            "weight_carried", "h_style"]

# 当日の馬体重 (horse_weight / weight_diff) は発走約70分前まで出ないので使わない。
# 学習時は必ずあるのに前日の予想では欠ける、という食い違いを避けるため前走の馬体重を使う
FEATURES = [
    "distance", "field_size", "class_level", "venue", "surface", "going",
    "frame", "number_ratio", "weight_carried", "age", "sex", "h_last_weight",
    "h_n_runs", "h_days_since", "h_last_fin", "h_mean3_fin", "h_mean5_fin",
    "h_last_fig", "h_mean3_fig", "h_best5_fig", "h_last_behind", "h_mean3_behind",
    "h_mean5_l3f", "h_style", "h_win_rate", "h_top3_rate", "h_dist_change", "h_class_change",
    "h_last_pop", "h_surf_top3", "h_band_top3", "h_venue_top3", "h_course_n", "h_prize_log",
    "j_n", "j_win", "j_top3", "t_win", "t_top3", "jv_top3", "s_n", "s_top3",
] + [f"{c}_rel" for c in REL_COLS] + [f"{c}_rank" for c in ("h_best5_fig", "h_mean3_fig", "j_win")]


def build_features(df: pd.DataFrame, baseline: SpeedBaseline) -> pd.DataFrame:
    """load_runs の結果に特徴量列を足して返す (行の並びは維持)."""
    df = df.copy()
    field = df["field_size"].clip(lower=2)
    finished = df["finish_pos"].notna()
    df["won"] = np.where(df["is_future"] == 1, np.nan, (df["finish_pos"] == 1).astype(float))
    df["top3"] = np.where(df["is_future"] == 1, np.nan, (df["finish_pos"] <= 3).astype(float))
    df["fin_ratio"] = ((df["finish_pos"] - 1) / (field - 1)).where(finished)
    df.loc[(df["is_future"] == 0) & ~finished, "fin_ratio"] = 1.0  # 競走中止は最下位扱い
    df["speed_fig"] = baseline.transform(df)
    win_time = df.groupby("race_id")["time_sec"].transform("min")
    df["time_behind"] = (df["time_sec"] - win_time).clip(upper=5.0)
    df["l3f_rank"] = df.groupby("race_id")["last3f"].rank(method="min") / field
    first = pd.to_numeric(df["passing"].astype("string").str.split("-").str[0], errors="coerce")
    df["first_pos"] = first / field
    df["band"] = df["distance"].map(distance_band)

    # ---- 馬の過去成績 (1走前以前のみ)
    h = df.groupby("horse_id", sort=False)
    df["h_n_runs"] = h.cumcount()
    df["h_days_since"] = (df["date"] - h["date"].shift()).dt.days
    df["h_last_fin"] = h["fin_ratio"].shift()
    df["h_mean3_fin"] = _prev_roll(df, "horse_id", "fin_ratio", 3)
    df["h_mean5_fin"] = _prev_roll(df, "horse_id", "fin_ratio", 5)
    df["h_last_fig"] = h["speed_fig"].shift()
    df["h_mean3_fig"] = _prev_roll(df, "horse_id", "speed_fig", 3)
    df["h_best5_fig"] = _prev_roll(df, "horse_id", "speed_fig", 5, "max")
    df["h_last_behind"] = h["time_behind"].shift()
    df["h_mean3_behind"] = _prev_roll(df, "horse_id", "time_behind", 3)
    df["h_mean5_l3f"] = _prev_roll(df, "horse_id", "l3f_rank", 5)
    df["h_style"] = _prev_roll(df, "horse_id", "first_pos", 5)
    wins_before = _exclusive_cumsum(df, "horse_id", "won")
    top3_before = _exclusive_cumsum(df, "horse_id", "top3")
    runs = df["h_n_runs"]
    df["h_win_rate"] = (wins_before + 0.08 * 3) / (runs + 3)
    df["h_top3_rate"] = (top3_before + 0.25 * 3) / (runs + 3)
    df["h_dist_change"] = df["distance"] - h["distance"].shift()
    df["h_class_change"] = df["class_level"] - h["class_level"].shift()
    df["h_last_pop"] = h["popularity"].shift() / h["field_size"].shift().clip(lower=2)
    df["h_last_weight"] = h["horse_weight"].shift()
    df["h_prize_log"] = np.log1p(_exclusive_cumsum(df, "horse_id", "prize").clip(lower=0))
    for name, keys in (("surf", ["horse_id", "surface"]), ("band", ["horse_id", "band"]),
                       ("venue", ["horse_id", "venue"])):
        n = df.groupby(keys, sort=False).cumcount()
        t3 = _exclusive_cumsum(df, keys, "top3")
        df[f"h_{name}_top3"] = (t3 + 0.25 * 2) / (n + 2)
    df["h_course_n"] = df.groupby(["horse_id", "venue", "surface", "distance"], sort=False).cumcount()

    # ---- 人の成績 (前日まで)
    # 事前分布は固定値 (データ全体の平均を使うと未来の情報が混ざるため)
    prior_w, prior_t = PRIOR_WIN, PRIOR_TOP3
    jockey = df["jockey_id"].astype("string")
    trainer = df["trainer_id"].astype("string")
    sire_key = df["sire_id"].astype("string") + "|" + df["surface"].astype("string") + "|" + df["band"].astype("string")
    parts = [
        _entity_rates(df, jockey, "j", window_days=365, k=30, prior_win=prior_w, prior_top3=prior_t),
        _entity_rates(df, trainer, "t", window_days=365, k=30, prior_win=prior_w, prior_top3=prior_t),
        _entity_rates(df, jockey + "|" + df["venue"].astype("string"), "jv", window_days=730, k=30,
                      prior_win=prior_w, prior_top3=prior_t),
        _entity_rates(df, sire_key, "s", window_days=None, k=50, prior_win=prior_w, prior_top3=prior_t),
    ]
    for p in parts:
        df[p.columns] = p
    df = df.drop(columns=["jv_n", "jv_win", "t_n", "s_win"])

    # ---- レース内での相対値
    df["number_ratio"] = df["number"] / field
    race = df.groupby("race_id")
    for c in REL_COLS:
        df[f"{c}_rel"] = df[c] - race[c].transform("mean")
    for c in ("h_best5_fig", "h_mean3_fig", "j_win"):
        df[f"{c}_rank"] = race[c].rank(ascending=False, pct=True)

    # ---- 市場の評価 (モデルの入力には使わず、ブレンドに使う)
    inv = 1.0 / df["win_odds"].where(df["win_odds"] > 0)
    df["p_market"] = inv / inv.groupby(df["race_id"]).transform("sum")
    return df


def model_matrix(df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """モデルの入力. features は学習時の列 (保存済みモデルは自分の列を渡す)."""
    X = df[features or FEATURES].copy()
    for col, cats in CATEGORIES.items():
        if col in X:
            X[col] = pd.Categorical(X[col], categories=cats)
    for col in X.columns:
        if col not in CATEGORIES:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)
    return X
