"""作業者の共通インターフェース."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


class WorkerError(RuntimeError):
    pass


@dataclass
class WorkerResult:
    text: str
    worker: str
    data: dict[str, Any] | None = None
    sources: list[dict[str, str]] = field(default_factory=list)


class Worker(Protocol):
    name: str

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> WorkerResult: ...


def extract_json(text: str) -> dict[str, Any]:
    """文章の中から最初のJSONオブジェクトを取り出す (CLI出力向け)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for c in candidates:
        try:
            value = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise WorkerError("出力からJSONを読み取れませんでした")


def dedupe_sources(sources: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, dict[str, str]] = {}
    for s in sources:
        url = s.get("url")
        if url and url.startswith(("http://", "https://")) and url not in seen:
            seen[url] = {"url": url, "title": s.get("title") or ""}
    return list(seen.values())


WORKER_NAMES = ("claude_api", "gemini_api", "openai_api", "claude_cli", "gemini_cli", "codex_cli")


def make_worker(name: str, settings) -> Worker:
    """設定名から作業者を作る. 作れないとき (SDKの初期化失敗など) は WorkerError."""
    try:
        return _make_worker(name, settings)
    except WorkerError:
        raise
    except Exception as exc:
        raise WorkerError(f"{name} を初期化できません: {exc}") from exc


def _make_worker(name: str, settings) -> Worker:
    if name == "claude_api":
        from keiba.llm.claude import ClaudeWorker

        return ClaudeWorker(settings.section("claude"))
    if name == "gemini_api":
        from keiba.llm.gemini import GeminiWorker

        return GeminiWorker(settings.section("gemini"))
    if name == "openai_api":
        from keiba.llm.openai_worker import OpenAIWorker

        return OpenAIWorker(settings.section("openai"))
    if name in ("claude_cli", "gemini_cli", "codex_cli"):
        from keiba.llm.cli_worker import CLIWorker

        cli = settings.section("cli")
        return CLIWorker(name, cli[name], timeout=cli.get("timeout", 600))
    raise WorkerError(f"不明な作業者です: {name} (選べるもの: {', '.join(WORKER_NAMES)})")
