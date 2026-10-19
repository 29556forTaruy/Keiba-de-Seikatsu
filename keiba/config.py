"""設定の読み込み.

既定値 → config.toml (あれば) → 環境変数 の順に上書きする。
APIキーは設定ファイルに書かず、環境変数 (ANTHROPIC_API_KEY など) で渡す。
"""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULTS: dict[str, Any] = {
    "paths": {
        "db": "data/keiba.sqlite",
        "raw_cache": "data/raw",
        "models": "models",
        "reports": "reports",
    },
    "http": {
        # netkeibaへの負荷を避けるため、リクエスト間隔は最低1秒以上
        "min_interval": 1.5,
        "jitter": 0.5,
        "timeout": 20,
        "max_retries": 3,
        # 出馬表・オッズなど変化するページのキャッシュ有効期間(秒)
        "live_ttl": 600,
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
        ),
    },
    "betting": {
        "bankroll": 100_000,
        "kelly_fraction": 0.25,
        "min_ev": 1.15,
        "min_prob": 0.02,
        "max_bet_ratio": 0.05,
        # 1点あたりの上限額。大きく賭けるとオッズが下がるので、資金が増えても青天井にしない
        "max_stake": 30_000,
        "race_budget_ratio": 0.10,
        "unit": 100,
    },
    # 役割ごとにどの作業者(LLM)を使うか。空文字なら使わない。
    # 選べる作業者: claude_api / gemini_api / openai_api / claude_cli / gemini_cli / codex_cli
    "workers": {
        "news": "claude_api",
        "predict": "claude_api",
        "review": "",
    },
    "claude": {
        "model": "claude-opus-5-5",
        "effort": "high",
        "news_effort": "medium",
        "max_tokens": 16000,
        "web_search_max_uses": 6,
        # 安全分類器で断られた場合にサーバー側で別モデルに再実行する
        "fallbacks": True,
    },
    "gemini": {
        "model": "gemini-flash-latest",
    },
    "openai": {
        "model": "gpt-5",
    },
    # サブスク契約のCLIを作業者として使う場合のコマンド。プロンプトは標準入力で渡す。
    # ({prompt} を書くと引数で渡すが、Windows ではコマンドラインの長さ制限を超えやすいので使わない)
    # Claude Code は使えるツールを Web 検索だけに絞る (手元のファイルを読ませない)。
    "cli": {
        "timeout": 600,
        "claude_cli": ["claude", "-p", "--tools", "WebSearch", "--allowedTools", "WebSearch"],
        "gemini_cli": ["gemini"],
        "codex_cli": ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "-"],
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Settings:
    raw: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    base_dir: Path = field(default_factory=Path.cwd)

    def section(self, name: str) -> dict[str, Any]:
        return self.raw.get(name, {})

    def path(self, name: str) -> Path:
        p = Path(self.raw["paths"][name]).expanduser()
        return p if p.is_absolute() else self.base_dir / p

    @property
    def db_path(self) -> Path:
        return self.path("db")

    @property
    def http(self) -> dict[str, Any]:
        return self.raw["http"]

    @property
    def betting(self) -> dict[str, Any]:
        return self.raw["betting"]

    @property
    def workers(self) -> dict[str, str]:
        return self.raw["workers"]


def load_settings(config_path: str | Path | None = None, db: str | None = None) -> Settings:
    """設定を読み込む。config_path 未指定なら ./config.toml を探す。"""
    raw = copy.deepcopy(DEFAULTS)
    path = Path(config_path) if config_path else Path("config.toml")
    base_dir = Path.cwd()
    if path.exists():
        with path.open("rb") as f:
            raw = _deep_merge(raw, tomllib.load(f))
        base_dir = path.resolve().parent
    elif config_path:
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")

    env_db = os.environ.get("KEIBA_DB")
    if db or env_db:
        # コマンドラインで渡したパスはカレントディレクトリ基準
        raw["paths"]["db"] = str(Path(db or env_db).expanduser().resolve())
    return Settings(raw=raw, base_dir=base_dir)
