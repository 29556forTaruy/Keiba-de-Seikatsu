"""Gemini API の作業者 (google-genai SDK). Google検索のグラウンディングで情報集めに向く."""

from __future__ import annotations

import json
from typing import Any

from keiba.llm.base import WorkerError, WorkerResult, dedupe_sources


def _network_errors() -> tuple[type[Exception], ...]:
    """google-genai の通信エラーの型 (httpx と、その互換フォーク httpx2). APIError にならないので別に捕まえる."""
    import httpx

    errs: list[type[Exception]] = [httpx.HTTPError]
    try:
        import httpx2

        errs.append(httpx2.HTTPError)
    except (ImportError, AttributeError):
        pass
    return tuple(errs)


class GeminiWorker:
    name = "gemini_api"

    def __init__(self, cfg: dict[str, Any]) -> None:
        try:
            from google import genai
        except ImportError as exc:
            raise WorkerError("google-genai が未インストールです: pip install -e '.[gemini]'") from exc
        # GEMINI_API_KEY (または GOOGLE_API_KEY) を使う
        try:
            self.client = genai.Client()
        except ValueError as exc:  # APIキーが無い
            raise WorkerError(f"Gemini APIを使えません。GEMINI_API_KEY を確認してください ({exc})") from exc
        self.model = cfg.get("model", "gemini-flash-latest")

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> WorkerResult:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(google_search=types.GoogleSearch())] if web_search else None,
            response_mime_type="application/json" if schema else None,
            response_json_schema=schema,
        )
        try:
            resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        except errors.APIError as exc:
            raise WorkerError(f"Gemini APIエラー: {exc}") from exc
        except _network_errors() as exc:  # 接続できない・応答がないなど
            raise WorkerError(f"Gemini APIに接続できません。ネットワークを確認してください: {exc}") from exc
        text = (resp.text or "").strip()
        sources = []
        for cand in resp.candidates or []:
            meta = getattr(cand, "grounding_metadata", None)
            for chunk in (getattr(meta, "grounding_chunks", None) or []):
                web = getattr(chunk, "web", None)
                if web is not None and web.uri:
                    sources.append({"url": web.uri, "title": web.title or ""})
        data = None
        if schema:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WorkerError("Geminiの出力がJSONとして読めませんでした") from exc
        return WorkerResult(text=text, worker=f"{self.name}:{self.model}", data=data, sources=dedupe_sources(sources))
