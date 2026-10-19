"""SQLiteによる保存層.

テーブル:
  races       レース情報 (過去レース・これからのレースの両方)
  results     確定成績 (1頭1行)
  entries     出馬表 (これからのレース, 1頭1行)
  payouts     払戻金
  odds        単勝・複勝オッズ (取得時点のもの)
  horses      馬の血統情報
  news        LLM作業者が集めた直前情報
  analyses    LLM作業者の予想・セカンドオピニオン
  predictions モデルの予想確率
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS races (
    race_id      TEXT PRIMARY KEY,
    date         TEXT NOT NULL,
    venue_code   TEXT,
    venue        TEXT,
    kai          INTEGER,
    nichi        INTEGER,
    race_num     INTEGER,
    race_name    TEXT,
    grade        TEXT,
    race_class   TEXT,
    class_level  INTEGER,
    surface      TEXT,
    distance     INTEGER,
    direction    TEXT,
    course_detail TEXT,
    weather      TEXT,
    going        TEXT,
    start_time   TEXT,
    field_size   INTEGER,
    conditions   TEXT,
    source       TEXT,
    updated_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_races_date ON races(date);

CREATE TABLE IF NOT EXISTS results (
    race_id       TEXT NOT NULL,
    horse_id      TEXT NOT NULL,
    horse_name    TEXT,
    finish_pos    INTEGER,
    finish_status TEXT,
    frame         INTEGER,
    number        INTEGER,
    sex           TEXT,
    age           INTEGER,
    weight_carried REAL,
    jockey_id     TEXT,
    jockey_name   TEXT,
    time_sec      REAL,
    margin        TEXT,
    passing       TEXT,
    last3f        REAL,
    odds          REAL,
    popularity    INTEGER,
    horse_weight  INTEGER,
    weight_diff   INTEGER,
    trainer_id    TEXT,
    trainer_name  TEXT,
    trainer_region TEXT,
    owner         TEXT,
    prize         REAL,
    PRIMARY KEY (race_id, horse_id)
);
CREATE INDEX IF NOT EXISTS idx_results_horse ON results(horse_id);

CREATE TABLE IF NOT EXISTS entries (
    race_id       TEXT NOT NULL,
    horse_id      TEXT NOT NULL,
    horse_name    TEXT,
    frame         INTEGER,
    number        INTEGER,
    sex           TEXT,
    age           INTEGER,
    weight_carried REAL,
    jockey_id     TEXT,
    jockey_name   TEXT,
    trainer_id    TEXT,
    trainer_name  TEXT,
    trainer_region TEXT,
    horse_weight  INTEGER,
    weight_diff   INTEGER,
    scratched     INTEGER DEFAULT 0,
    updated_at    TEXT,
    PRIMARY KEY (race_id, horse_id)
);

CREATE TABLE IF NOT EXISTS payouts (
    race_id     TEXT NOT NULL,
    bet_type    TEXT NOT NULL,
    combination TEXT NOT NULL,
    payout      INTEGER,
    popularity  INTEGER,
    PRIMARY KEY (race_id, bet_type, combination)
);

CREATE TABLE IF NOT EXISTS odds (
    race_id        TEXT NOT NULL,
    number         INTEGER NOT NULL,
    win_odds       REAL,
    place_odds_min REAL,
    place_odds_max REAL,
    popularity     INTEGER,
    fetched_at     TEXT,
    PRIMARY KEY (race_id, number)
);

CREATE TABLE IF NOT EXISTS horses (
    horse_id    TEXT PRIMARY KEY,
    horse_name  TEXT,
    sire        TEXT,
    sire_id     TEXT,
    dam         TEXT,
    dam_id      TEXT,
    damsire     TEXT,
    damsire_id  TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS news (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id    TEXT NOT NULL,
    worker     TEXT,
    content    TEXT,
    sources    TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id    TEXT NOT NULL,
    role       TEXT NOT NULL,
    worker     TEXT,
    content    TEXT,
    created_at TEXT,
    ref_id     INTEGER  -- review のとき、点検した predict の id
);

CREATE TABLE IF NOT EXISTS predictions (
    race_id       TEXT NOT NULL,
    model_version TEXT NOT NULL,
    horse_id      TEXT NOT NULL,
    number        INTEGER,
    score         REAL,
    p_win         REAL,
    p_top3        REAL,
    p_market      REAL,
    created_at    TEXT,
    PRIMARY KEY (race_id, model_version, horse_id)
);
"""

RACE_COLS = [
    "race_id", "date", "venue_code", "venue", "kai", "nichi", "race_num", "race_name",
    "grade", "race_class", "class_level", "surface", "distance", "direction",
    "course_detail", "weather", "going", "start_time", "field_size", "conditions", "source",
]
RESULT_COLS = [
    "race_id", "horse_id", "horse_name", "finish_pos", "finish_status", "frame", "number",
    "sex", "age", "weight_carried", "jockey_id", "jockey_name", "time_sec", "margin",
    "passing", "last3f", "odds", "popularity", "horse_weight", "weight_diff", "trainer_id",
    "trainer_name", "trainer_region", "owner", "prize",
]
ENTRY_COLS = [
    "race_id", "horse_id", "horse_name", "frame", "number", "sex", "age", "weight_carried",
    "jockey_id", "jockey_name", "trainer_id", "trainer_name", "trainer_region",
    "horse_weight", "weight_diff", "scratched",
]
PAYOUT_COLS = ["race_id", "bet_type", "combination", "payout", "popularity"]
ODDS_COLS = ["race_id", "number", "win_odds", "place_odds_min", "place_odds_max", "popularity"]
HORSE_COLS = ["horse_id", "horse_name", "sire", "sire_id", "dam", "dam_id", "damsire", "damsire_id"]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """古いDBに後から足した列を追加する."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()}
    if "ref_id" not in cols:
        conn.execute("ALTER TABLE analyses ADD COLUMN ref_id INTEGER")
        conn.commit()


def _upsert(
    conn: sqlite3.Connection,
    table: str,
    cols: list[str],
    rows: Iterable[Mapping[str, Any]],
    *,
    stamp: str | None = None,
) -> int:
    all_cols = cols + ([stamp] if stamp else [])
    sql = (
        f"INSERT OR REPLACE INTO {table} ({', '.join(all_cols)}) "
        f"VALUES ({', '.join('?' for _ in all_cols)})"
    )
    ts = now_iso()
    values = [
        tuple(row.get(c) for c in cols) + ((ts,) if stamp else ())
        for row in rows
    ]
    if values:
        conn.executemany(sql, values)
    return len(values)


def save_race(conn: sqlite3.Connection, race: Mapping[str, Any]) -> None:
    _upsert(conn, "races", RACE_COLS, [race], stamp="updated_at")


def save_results(conn: sqlite3.Connection, race_id: str, rows: list[Mapping[str, Any]]) -> int:
    conn.execute("DELETE FROM results WHERE race_id = ?", (race_id,))
    return _upsert(conn, "results", RESULT_COLS, rows)


def save_entries(conn: sqlite3.Connection, race_id: str, rows: list[Mapping[str, Any]]) -> int:
    conn.execute("DELETE FROM entries WHERE race_id = ?", (race_id,))
    return _upsert(conn, "entries", ENTRY_COLS, rows, stamp="updated_at")


def save_payouts(conn: sqlite3.Connection, race_id: str, rows: list[Mapping[str, Any]]) -> int:
    conn.execute("DELETE FROM payouts WHERE race_id = ?", (race_id,))
    return _upsert(conn, "payouts", PAYOUT_COLS, rows)


def save_odds(conn: sqlite3.Connection, race_id: str, rows: list[Mapping[str, Any]]) -> int:
    conn.execute("DELETE FROM odds WHERE race_id = ?", (race_id,))
    return _upsert(conn, "odds", ODDS_COLS, rows, stamp="fetched_at")


def save_horse(conn: sqlite3.Connection, horse: Mapping[str, Any]) -> None:
    _upsert(conn, "horses", HORSE_COLS, [horse], stamp="updated_at")


def save_news(
    conn: sqlite3.Connection, race_id: str, worker: str, content: str, sources: list[dict]
) -> None:
    conn.execute(
        "INSERT INTO news (race_id, worker, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
        (race_id, worker, content, json.dumps(sources, ensure_ascii=False), now_iso()),
    )


def save_analysis(
    conn: sqlite3.Connection, race_id: str, role: str, worker: str, content: Any, *, ref_id: int | None = None
) -> int:
    """content は dict (JSON保存) か文字列. ref_id は review が点検した predict の id. 保存した行の id を返す."""
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    cur = conn.execute(
        "INSERT INTO analyses (race_id, role, worker, content, created_at, ref_id) VALUES (?, ?, ?, ?, ?, ?)",
        (race_id, role, worker, text, now_iso(), ref_id),
    )
    return int(cur.lastrowid)


def _analysis_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    d = dict(row)
    try:
        d["content"] = json.loads(d["content"])
    except (TypeError, json.JSONDecodeError):
        pass
    return d


def latest_analysis(conn: sqlite3.Connection, race_id: str, role: str) -> dict[str, Any] | None:
    return _analysis_dict(conn.execute(
        "SELECT * FROM analyses WHERE race_id = ? AND role = ? ORDER BY id DESC LIMIT 1", (race_id, role)
    ).fetchone())


def review_for(conn: sqlite3.Connection, race_id: str, analysis_id: int) -> dict[str, Any] | None:
    """指定の AI予想 (predict) を点検したセカンドオピニオン.

    ref_id のない古い行は、その予想より後に書かれたものだけを対象にする。
    """
    return _analysis_dict(conn.execute(
        """SELECT * FROM analyses WHERE race_id = ? AND role = 'review'
             AND (ref_id = ? OR (ref_id IS NULL AND id > ?))
           ORDER BY id DESC LIMIT 1""",
        (race_id, analysis_id, analysis_id),
    ).fetchone())


def save_predictions(conn: sqlite3.Connection, rows: list[Mapping[str, Any]]) -> int:
    cols = ["race_id", "model_version", "horse_id", "number", "score", "p_win", "p_top3", "p_market"]
    return _upsert(conn, "predictions", cols, rows, stamp="created_at")


# ---------------------------------------------------------------- 読み出し

def has_results(conn: sqlite3.Connection, race_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM results WHERE race_id = ? LIMIT 1", (race_id,)).fetchone()
    return row is not None


def get_race(conn: sqlite3.Connection, race_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM races WHERE race_id = ?", (race_id,)).fetchone()
    return dict(row) if row else None


def race_ids_on(conn: sqlite3.Connection, date: str) -> list[str]:
    rows = conn.execute(
        "SELECT race_id FROM races WHERE date = ? ORDER BY venue_code, race_num", (date,)
    ).fetchall()
    return [r["race_id"] for r in rows]


def latest_news(conn: sqlite3.Connection, race_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM news WHERE race_id = ? ORDER BY id DESC", (race_id,)
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["sources"] = json.loads(d["sources"] or "[]")
        out.append(d)
    return out


def read_df(conn: sqlite3.Connection, sql: str, params: tuple | dict = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def horses_missing_pedigree(conn: sqlite3.Connection, limit: int | None = None) -> list[str]:
    sql = """
        SELECT DISTINCT h.horse_id FROM (
            SELECT horse_id FROM results UNION SELECT horse_id FROM entries
        ) h LEFT JOIN horses p ON p.horse_id = h.horse_id
        WHERE p.horse_id IS NULL
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(sql).fetchall()]
