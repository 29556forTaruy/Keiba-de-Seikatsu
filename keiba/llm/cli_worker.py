"""サブスク契約のCLI (claude -p / gemini / codex exec) を作業者として使う.

APIキーなしで、各社の定額プランの範囲で動かせる。プロンプトは標準入力で渡す。
プロンプトにはWeb由来の文章が含まれうる。CLIは空の一時ディレクトリで起動するので
プロジェクトのファイルを書き換えることはないが、それだけでは手元のファイルの読み取りは防げない。
既定のコマンドでは Claude Code のツールを Web 検索だけに絞っている。Codex の read-only
サンドボックスは書き込みと通信を止めるだけで、ユーザーが読めるファイルは読める (README 参照)。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from typing import Any

from keiba.llm.base import WorkerError, WorkerResult, dedupe_sources, extract_json

# 本文中のURL. 日本語の文では「（URL）、」のように全角の括弧・句読点がすぐ後に続くので、そこで切る
URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]（）「」『』【】〔〕＜＞、。，．！？…]+")


def find_urls(text: str) -> list[str]:
    return [u.rstrip(".,;:!?*") for u in URL_RE.findall(text)]


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
        # Windows の npm 版 (gemini.CMD など) も起動できるよう、PATH で見つけた実体のパスで起動する
        exe = shutil.which(self.command[0])
        if exe is None:
            raise WorkerError(f"コマンドが見つかりません: {self.command[0]} (インストールとログインを確認してください)")
        if uses_arg and exe.lower().endswith((".cmd", ".bat")):
            # cmd.exe が引数を解釈し直すので、Web由来の文章を引数で渡すとコマンドとして実行されうる
            raise WorkerError(f"{self.name}: {exe} にはプロンプトを引数で渡せません。"
                              "config.toml の [cli] から {prompt} を外して標準入力で渡してください")
        cmd = [exe] + [part.replace("{prompt}", full) for part in self.command[1:]]
        with tempfile.TemporaryDirectory(prefix="keiba-worker-") as workdir:
            try:
                proc = subprocess.run(
                    cmd,
                    input=None if uses_arg else full,
                    capture_output=True,
                    text=True,
                    # 各CLIはUTF-8で読み書きする (日本語版 Windows の既定 cp932 で読むと文字化け・例外になる)
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout,
                    cwd=workdir,
                )
            except subprocess.TimeoutExpired as exc:
                raise WorkerError(f"{self.name} が {self.timeout} 秒で終わりませんでした") from exc
            except (OSError, UnicodeError) as exc:
                hint = " (プロンプトを引数で渡す設定ならコマンドラインが長すぎます。標準入力で渡してください)" if uses_arg else ""
                raise WorkerError(f"{self.name} を起動できません: {exc}{hint}") from exc
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        if proc.returncode != 0:
            tail = (stderr or stdout).strip()[-500:]
            raise WorkerError(f"{self.name} が失敗しました (終了コード {proc.returncode}): {tail}")
        text = stdout.strip()
        data = extract_json(text) if schema else None
        return WorkerResult(text=text, worker=self.name, data=data,
                            sources=dedupe_sources([{"url": u} for u in find_urls(text)]))
