"""OpenAI API の作業者 (Responses API)."""

from __future__ import annotations

import json
from typing import Any

from keiba.llm.base import WorkerError, WorkerResult, dedupe_sources


class OpenAIWorker:
    name = "openai_api"

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            import openai
        except ImportError as exc:
            raise WorkerError("openai が未インストールです: pip install -e '.[openai]'") from exc
        self._openai = openai
        # OPENAI_API_KEY を使う
        try:
            self.client = openai.OpenAI()
        except openai.OpenAIError as exc:
            raise WorkerError(f"OpenAI APIを使えません。OPENAI_API_KEY を確認してください ({exc})") from exc
        self.model = cfg.get("model", "gpt-5")

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> WorkerResult:
        kwargs: dict[str, Any] = {"model": self.model, "input": prompt}
        if system:
            kwargs["instructions"] = system
        if web_search:
            kwargs["tools"] = [{"type": "web_search"}]
        if schema:
            kwargs["text"] = {"format": {"type": "json_schema", "name": "keiba_output", "schema": schema, "strict": True}}
        try:
            resp = self.client.responses.create(**kwargs)
        except self._openai.APIError as exc:
            raise WorkerError(f"OpenAI APIエラー: {exc}") from exc
        text = (resp.output_text or "").strip()
        sources = []
        for item in resp.output or []:
            for part in getattr(item, "content", None) or []:
                for ann in getattr(part, "annotations", None) or []:
                    if getattr(ann, "type", "") == "url_citation":
                        sources.append({"url": ann.url, "title": getattr(ann, "title", "") or ""})
        data = None
        if schema:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WorkerError("OpenAIの出力がJSONとして読めませんでした") from exc
        return WorkerResult(text=text, worker=f"{self.name}:{self.model}", data=data, sources=dedupe_sources(sources))
