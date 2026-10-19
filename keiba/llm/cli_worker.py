"""サブスク契約のCLI (claude -p / gemini -p / codex exec) を作業者として使う.

APIキーなしで、各社の定額プランの範囲で動かせる。
プロンプトにはWeb由来の文章が含まれうるので、CLIは空の一時ディレクトリで
起動し、手元のファイルに触れないようにしている。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from typing import Any

from keiba.llm.base import WorkerError, WorkerResult, dedupe_sources, extract_json


class CLIWorker:
    def __init__(self, name: str, command: list[str], *, timeout: float = 600) -> None:
        if not command:
            raise WorkerError(f"{name} のコマンドが設定されていません (config.toml の [cli])")
        self.name = name
        self.command = list(command)
        self.timeout = timeout

    def build_prompt(self, prompt: str, system: str | None, schema: dict[str, Any] | None, web_search: bool) -> str:
        parts = []
        if system:
            parts.append(system)
        if web_search:
            parts.append("必要に応じてWeb検索で最新の情報を調べ、参照したURLを本文中に書いてください。")
        parts.append(prompt)
        if schema:
            parts.append(
                "出力は次のJSONスキーマに従うJSONオブジェクトだけにしてください。前後に説明文やコードブロックを付けないこと。\n"
                + json.dumps(schema, ensure_ascii=False)
            )
        return "\n\n".join(parts)

    def run(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> WorkerResult:
        full = self.build_prompt(prompt, system, schema, web_search)
        uses_arg = any("{prompt}" in part for part in self.command)
        cmd = [part.replace("{prompt}", full) for part in self.command]
        if shutil.which(cmd[0]) is None:
            raise WorkerError(f"コマンドが見つかりません: {cmd[0]} (インストールとログインを確認してください)")
        with tempfile.TemporaryDirectory(prefix="keiba-worker-") as workdir:
            try:
                proc = subprocess.run(
                    cmd,
                    input=None if uses_arg else full,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=workdir,
                )
            except subprocess.TimeoutExpired as exc:
                raise WorkerError(f"{self.name} が {self.timeout} 秒で終わりませんでした") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip()[-500:]
            raise WorkerError(f"{self.name} が失敗しました (終了コード {proc.returncode}): {tail}")
        text = proc.stdout.strip()
        data = extract_json(text) if schema else None
        urls = re.findall(r"https?://[^\s)\]」>\"']+", text)
        return WorkerResult(text=text, worker=self.name, data=data, sources=dedupe_sources([{"url": u} for u in urls]))
