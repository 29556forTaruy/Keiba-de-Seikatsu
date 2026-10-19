"""礼儀正しいHTTP取得 (間隔制御・ディスクキャッシュ・文字コード判定).

一度取得したページは data/raw に gzip で保存し、確定成績のような変わらない
ページは二度と取りに行かない。パーサーを直したときは保存済みHTMLを
読み直すだけで再構築できる。
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import random
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)

_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-]+)""", re.I)


class FetchError(RuntimeError):
    pass


def decode_html(body: bytes, header_charset: str | None = None) -> str:
    """netkeibaはEUC-JPとUTF-8が混在するので、ヘッダ→metaタグ→総当たりの順に判定する."""
    candidates: list[str] = []
    if header_charset:
        candidates.append(header_charset)
    m = _META_CHARSET.search(body[:4096])
    if m:
        candidates.append(m.group(1).decode("ascii", "ignore"))
    candidates += ["utf-8", "euc_jp", "euc_jis_2004", "cp932"]
    seen = set()
    for enc in candidates:
        enc = enc.lower().replace("-", "_")
        if enc in ("euc_jp", "eucjp", "x_euc_jp"):
            enc = "euc_jp"
        if enc in seen:
            continue
        seen.add(enc)
        try:
            return body.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return body.decode("euc_jp", errors="replace")


class Fetcher:
    def __init__(
        self,
        cache_dir: str | Path,
        *,
        min_interval: float = 1.5,
        jitter: float = 0.5,
        timeout: float = 20,
        max_retries: int = 3,
        live_ttl: float = 600,
        user_agent: str | None = None,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.min_interval = max(1.0, float(min_interval))
        self.jitter = jitter
        self.timeout = timeout
        self.max_retries = max_retries
        self.live_ttl = live_ttl
        self.offline = offline
        self.session = requests.Session()
        if user_agent:
            self.session.headers["User-Agent"] = user_agent
        self.session.headers["Accept-Language"] = "ja,en;q=0.8"
        self._last_request = 0.0
        self.network_requests = 0

    @classmethod
    def from_settings(cls, settings, *, offline: bool = False) -> "Fetcher":
        h = settings.http
        return cls(
            settings.path("raw_cache"),
            min_interval=h["min_interval"],
            jitter=h["jitter"],
            timeout=h["timeout"],
            max_retries=h["max_retries"],
            live_ttl=h["live_ttl"],
            user_agent=h["user_agent"],
            offline=offline,
        )

    def cache_path(self, url: str) -> Path:
        parsed = urlparse(url)
        digest = hashlib.sha1(url.encode()).hexdigest()[:16]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", parsed.path.strip("/"))[-60:] or "index"
        return self.cache_dir / parsed.netloc / f"{slug}_{digest}.html.gz"

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        delay = self.min_interval + random.uniform(0, self.jitter)
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def get(self, url: str, *, live: bool = False, refresh: bool = False) -> str:
        """URLを取得してテキストで返す.

        live=True のページ(出馬表・オッズ)は live_ttl 秒より古いキャッシュを使わない。
        """
        path = self.cache_path(url)
        if path.exists() and not refresh:
            fresh = not live or (time.time() - path.stat().st_mtime) < self.live_ttl
            if fresh or self.offline:
                return gzip.decompress(path.read_bytes()).decode("utf-8")
        if self.offline:
            raise FetchError(f"オフラインモードでキャッシュがありません: {url}")

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait()
            try:
                self._last_request = time.monotonic()
                self.network_requests += 1
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("取得失敗 (%s) %s", exc.__class__.__name__, url)
            else:
                if resp.status_code == 200:
                    text = decode_html(resp.content, resp.encoding if "charset" in resp.headers.get("content-type", "").lower() else None)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(gzip.compress(text.encode("utf-8")))
                    return text
                if resp.status_code in (403, 400):
                    raise FetchError(
                        f"HTTP {resp.status_code}: アクセスが拒否されました。"
                        f"間隔(http.min_interval)を広げて時間をおいてください: {url}"
                    )
                if resp.status_code == 404:
                    raise FetchError(f"HTTP 404: ページがありません: {url}")
                last_exc = FetchError(f"HTTP {resp.status_code}: {url}")
                log.warning("HTTP %s %s", resp.status_code, url)
            time.sleep(min(60, 2 ** (attempt + 1)))
        raise FetchError(f"取得に失敗しました: {url}") from last_exc
