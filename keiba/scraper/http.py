"""礼儀正しいHTTP取得 (間隔制御・ディスクキャッシュ・文字コード判定).

一度取得したページは data/raw に gzip で保存し、確定成績のような変わらない
ページは二度と取りに行かない。パーサーを直したときは保存済みHTMLを
読み直すだけで再構築できる。
"""

from __future__ import annotations

import codecs
import gzip
import hashlib
import logging
import os
import random
import re
import tempfile
import time
import zlib
from pathlib import Path
from urllib.parse import urlencode, urlparse

import requests

log = logging.getLogger(__name__)

# パースできなかったページの置き場 (cache_dir の下. サイト名のフォルダと重ならない名前)
FAILED_DIR = "_failed"

_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-]+)""", re.I)


class FetchError(RuntimeError):
    pass


class AccessDenied(FetchError):
    """HTTP 400/403. 拒否されたサイトには、同じ実行の中ではもう通信しない."""


# Web の EUC-JP (ブラウザが使う WHATWG 版 / Microsoft の CP51932) は、①や髙・﨑などの
# NEC特殊文字 (13区) と NEC選定IBM拡張文字 (89〜92区) を含みうる。Python の euc_jp は
# これを読めず、euc_jis_2004 は別の漢字 (髙→郄 など) として黙って読んでしまうので、
# 該当する2バイトだけ JIS の区点 → Shift_JIS に直して cp932 で読む。
_EUCJP_EXT_ROWS = {13, 89, 90, 91, 92}
_EUCJP_LABELS = {"euc_jp", "eucjp", "x_euc_jp", "cp51932", "eucjp_win", "eucjp_ms", "euc_jp_ms"}


def _eucjp_ext_char(data: bytes, pos: int) -> str | None:
    if pos + 1 >= len(data):
        return None
    lead, trail = data[pos], data[pos + 1]
    if not (0xA1 <= lead <= 0xFE and 0xA1 <= trail <= 0xFE):
        return None
    row, cell = lead - 0xA0, trail - 0xA0
    if row not in _EUCJP_EXT_ROWS:
        return None
    s1 = ((row - 1) >> 1) + (0x81 if row <= 62 else 0xC1)
    if row % 2:
        s2 = cell + (0x3F if cell <= 63 else 0x40)
    else:
        s2 = cell + 0x9E
    try:
        return bytes((s1, s2)).decode("cp932")
    except UnicodeDecodeError:
        return None


def _eucjp_ext_strict(exc: UnicodeError):
    if not isinstance(exc, UnicodeDecodeError):
        raise exc
    char = _eucjp_ext_char(exc.object, exc.start)
    if char is None:
        raise exc
    return char, exc.start + 2


def _eucjp_ext_replace(exc: UnicodeError):
    if not isinstance(exc, UnicodeDecodeError):
        raise exc
    char = _eucjp_ext_char(exc.object, exc.start)
    if char is None:
        return "\ufffd", exc.start + 1
    return char, exc.start + 2


codecs.register_error("keiba_eucjp_ext", _eucjp_ext_strict)
codecs.register_error("keiba_eucjp_ext_replace", _eucjp_ext_replace)


# JRA の Shift_JIS は Windows の拡張文字 (①・髙など) を含みうるので cp932 で読む
_SJIS_LABELS = {"shift_jis", "shiftjis", "sjis", "x_sjis", "ms_kanji", "windows_31j", "cp932"}


def _normalize_label(enc: str) -> str:
    enc = enc.lower().replace("-", "_")
    if enc in _SJIS_LABELS:
        return "cp932"
    return "euc_jp" if enc in _EUCJP_LABELS else enc


def site_of(url: str) -> str:
    """アクセス拒否を判断する単位. race.netkeiba.com と db.netkeiba.com は同じ netkeiba.com として扱う."""
    host = urlparse(url).hostname or ""
    parts = host.split(".")
    n = 3 if len(parts) >= 3 and parts[-2] in ("co", "go", "or", "ne", "ac") and len(parts[-1]) == 2 else 2
    return ".".join(parts[-n:])


def request_key(url: str, data: dict[str, str] | None = None) -> str:
    """キャッシュの鍵. POST はフォームの値も含める (JRA は同じURLに CNAME を送り分ける)."""
    return url if not data else f"{url}#POST?{urlencode(sorted(data.items()))}"


def decode_html(body: bytes, header_charset: str | None = None) -> str:
    """netkeibaはEUC-JPとUTF-8が混在するので、ヘッダ→metaタグ→総当たりの順に判定する.

    EUC-JP と書かれたページは CP51932 の拡張文字まで読み、euc_jis_2004 や cp932 には
    回さない (別の漢字に化けても気づけないため)。読めない文字は U+FFFD にして警告を出す。
    """
    labels: list[str] = []
    if header_charset:
        labels.append(header_charset)
    m = _META_CHARSET.search(body[:4096])
    if m:
        labels.append(m.group(1).decode("ascii", "ignore"))
    labels = [_normalize_label(enc) for enc in labels]
    if "euc_jp" in labels:
        # 別の文字コードで「読めてしまう」と化けに気づけないので、UTF-8 の取り違えだけ確かめる
        candidates = labels + ["utf-8"]
    else:
        candidates = labels + ["utf-8", "euc_jp", "euc_jis_2004", "cp932"]
    seen = set()
    for enc in candidates:
        enc = _normalize_label(enc)
        if enc in seen:
            continue
        seen.add(enc)
        try:
            if enc == "euc_jp":
                return body.decode("euc_jp", "keiba_eucjp_ext")
            return body.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    log.warning("文字コードを判定できないため、読めない文字を置き換えます")
    return body.decode("euc_jp", "keiba_eucjp_ext_replace")


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
        jra_fallback: bool = False,
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
        # 一度アクセスを拒否されたサイトには、この実行では以後ネットワークに出ない (キャッシュは使う)。
        # netkeiba に拒否されても、予備の JRA からは取れるようにサイトごとに持つ
        self.blocked: dict[str, AccessDenied] = {}
        # netkeiba で取れないとき JRA 公式サイトから取るか (collect.py が見る)
        self.jra_fallback = jra_fallback

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
            jra_fallback=bool(settings.section("sources").get("jra_fallback", True)),
        )

    def cache_path(self, url: str, data: dict[str, str] | None = None) -> Path:
        parsed = urlparse(url)
        digest = hashlib.sha1(request_key(url, data).encode()).hexdigest()[:16]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", parsed.path.strip("/"))[-60:] or "index"
        return self.cache_dir / parsed.netloc / f"{slug}_{digest}.html.gz"

    def invalidate(self, url: str, data: dict[str, str] | None = None) -> None:
        """中身の無いページ (開催前の結果ページなど) のキャッシュを消す."""
        self.cache_path(url, data).unlink(missing_ok=True)

    def keep_failed(self, url: str, data: dict[str, str] | None = None) -> Path | None:
        """パースできなかったページを data/raw/_failed/ に移す. 移した先を返す.

        次回は取り直し、取ったページはパーサーを直す材料として残す (同じページの前回分は上書き)。
        """
        src = self.cache_path(url, data)
        if not src.exists():
            return None
        dst = self.cache_dir / FAILED_DIR / src.relative_to(self.cache_dir)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
        except OSError as exc:
            log.warning("パースできなかったページを残せません (%s) %s", exc, dst)
            src.unlink(missing_ok=True)
            return None
        return dst

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        delay = self.min_interval + random.uniform(0, self.jitter)
        if elapsed < delay:
            time.sleep(delay - elapsed)

    def _read_cache(self, path: Path, url: str) -> str | None:
        try:
            return gzip.decompress(path.read_bytes()).decode("utf-8")
        except (EOFError, OSError, zlib.error, UnicodeDecodeError) as exc:
            # 書き込み途中で止まったなどで壊れたキャッシュは捨てて取り直す
            log.warning("壊れたキャッシュを破棄します (%s) %s", exc.__class__.__name__, url)
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, path: Path, text: str) -> None:
        """一時ファイルに書いてから置き換える (途中で止まっても壊れたファイルを残さない)."""
        tmp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                fh.write(gzip.compress(text.encode("utf-8")))
            os.replace(tmp, path)
            tmp = None
        except OSError as exc:
            log.warning("キャッシュを保存できません (%s) %s", exc, path)
        finally:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)

    def get(
        self,
        url: str,
        *,
        data: dict[str, str] | None = None,
        live: bool = False,
        refresh: bool = False,
        stale_before: float | None = None,
    ) -> str:
        """URLを取得してテキストで返す.

        live=True のページ(出馬表・オッズ)は live_ttl 秒より古いキャッシュを使わない。
        stale_before (UNIX時刻) より前に保存したキャッシュも、まだ内容が変わりうる
        時点のものとして live と同じに扱う (開催日が終わる前に取ったレース一覧などを、
        確定後に一度取り直すため)。
        data を渡すとフォームの POST で取る (JRA のページ)。
        """
        path = self.cache_path(url, data)
        if path.exists() and not refresh:
            mtime = path.stat().st_mtime
            if stale_before is not None and mtime < stale_before:
                live = True
            fresh = not live or (time.time() - mtime) < self.live_ttl
            if fresh or self.offline:
                text = self._read_cache(path, url)
                if text is not None:
                    return text
        if self.offline:
            raise FetchError(f"オフラインモードでキャッシュがありません: {url}")
        site = site_of(url)
        if site in self.blocked:
            raise AccessDenied(f"アクセスを拒否されたため、この実行では {site} への通信を止めています: {url}") from self.blocked[site]

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._wait()
            try:
                self._last_request = time.monotonic()
                self.network_requests += 1
                if data:
                    resp = self.session.post(url, data=data, timeout=self.timeout)
                else:
                    resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("取得失敗 (%s) %s", exc.__class__.__name__, url)
            else:
                if resp.status_code == 200:
                    text = decode_html(resp.content, resp.encoding if "charset" in resp.headers.get("content-type", "").lower() else None)
                    self._write_cache(path, text)
                    return text
                if resp.status_code in (403, 400):
                    self.blocked[site] = AccessDenied(
                        f"HTTP {resp.status_code}: アクセスが拒否されました。"
                        f"間隔(http.min_interval)を広げて時間をおいてください: {url}"
                    )
                    raise self.blocked[site]
                if resp.status_code == 404:
                    raise FetchError(f"HTTP 404: ページがありません: {url}")
                last_exc = FetchError(f"HTTP {resp.status_code}: {url}")
                log.warning("HTTP %s %s", resp.status_code, url)
            time.sleep(min(60, 2 ** (attempt + 1)))
        raise FetchError(f"取得に失敗しました: {url}") from last_exc
