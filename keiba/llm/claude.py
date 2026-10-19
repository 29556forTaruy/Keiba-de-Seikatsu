"""Claude API の作業者 (公式 anthropic SDK)."""

from __future__ import annotations

import json
from typing import Any

import anthropic

from keiba.llm.base import WorkerError, WorkerResult, dedupe_sources

MAX_CONTINUATIONS = 5


class ClaudeWorker:
    name = "claude_api"

    def __init__(self, cfg: dict[str, Any]) -> None:
        # ANTHROPIC_API_KEY (または `ant auth login` のプロファイル) を使う
        self.client = anthropic.Anthropic()
        self.model = cfg.get("model", "claude-opus-5-5")
        self.effort = cfg.get("effort", "high")
        self.news_effort = cfg.get("news_effort", "medium")
        self.max_tokens = int(cfg.get("max_tokens", 16000))
        self.web_search_max_uses = int(cfg.get("web_search_max_uses", 6))
        self.fallbacks = bool(cfg.get("fallbacks", True))

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> WorkerResult:
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {"effort": self.news_effort if web_search else self.effort},
        }
        if system:
            params["system"] = system
        if schema:
            params["output_config"]["format"] = {"type": "json_schema", "schema": schema}
        if web_search:
            params["tools"] = [{
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": self.web_search_max_uses,
                "user_location": {"type": "approximate", "country": "JP", "timezone": "Asia/Tokyo"},
            }]

        # max_tokens が大きいと SDK は非ストリーミングの呼び出しを拒否する (10分を超えうるため) ので、
        # ストリーミングで受けて最後のメッセージだけ使う
        if self.fallbacks:
            # 安全分類器に断られたとき、サーバー側で推奨モデルに再実行させる
            stream = self.client.beta.messages.stream
            params["betas"] = ["server-side-fallback-2026-07-01"]
            params["fallbacks"] = "default"
        else:
            stream = self.client.messages.stream

        texts: list[str] = []
        sources: list[dict[str, str]] = []
        try:
            for _ in range(MAX_CONTINUATIONS):
                with stream(**params) as s:
                    resp = s.get_final_message()
                for block in resp.content:
                    if block.type == "text":
                        texts.append(block.text)
                        for c in getattr(block, "citations", None) or []:
                            if getattr(c, "url", None):
                                sources.append({"url": c.url, "title": getattr(c, "title", "") or ""})
                    elif block.type == "web_search_tool_result" and isinstance(block.content, list):
                        for r in block.content:
                            if getattr(r, "url", None):
                                sources.append({"url": r.url, "title": getattr(r, "title", "") or ""})
                # Web検索がサーバー側の上限回数に達すると pause_turn で一旦止まるので続きを頼む
                if resp.stop_reason != "pause_turn":
                    break
                params["messages"] = params["messages"] + [{"role": "assistant", "content": resp.content}]
        except anthropic.AuthenticationError as exc:
            raise WorkerError("Claude APIの認証に失敗しました。ANTHROPIC_API_KEY を確認してください") from exc
        except anthropic.RateLimitError as exc:
            raise WorkerError("Claude APIのレート制限です。時間をおいて再実行してください") from exc
        except anthropic.APIStatusError as exc:
            raise WorkerError(f"Claude APIエラー ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise WorkerError("Claude APIに接続できません。ネットワークを確認してください") from exc
        except anthropic.APIError as exc:
            raise WorkerError(f"Claude APIエラー: {exc}") from exc
        except TypeError as exc:
            # APIキーもログイン情報もないと、SDK は送信前に TypeError を出す
            raise WorkerError("Claude APIの認証情報がありません。ANTHROPIC_API_KEY を設定するか "
                              f"ant auth login でログインしてください ({exc})") from exc
        except ValueError as exc:
            raise WorkerError(f"Claude APIの呼び出し設定が正しくありません: {exc}") from exc

        if resp.stop_reason == "refusal":
            raise WorkerError("Claudeが応答を辞退しました (refusal)")
        if resp.stop_reason == "max_tokens":
            raise WorkerError("出力が max_tokens で途中終了しました。claude.max_tokens を増やしてください")

        text = "".join(texts).strip()
        data = None
        if schema:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise WorkerError("Claudeの出力がJSONとして読めませんでした") from exc
        return WorkerResult(text=text, worker=f"{self.name}:{self.model}", data=data, sources=dedupe_sources(sources))
