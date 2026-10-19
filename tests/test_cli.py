"""keiba コマンドのエラー表示 (通信はしない)."""

import shutil

import pytest

from keiba import cli
from keiba.scraper.http import FetchError


def test_fetch_error_is_a_clean_message_with_offline_hint(monkeypatch, tmp_path):
    def blocked(settings, args):
        raise FetchError("HTTP 403: アクセスが拒否されました。時間をおいてください")

    monkeypatch.setattr(cli, "cmd_status", blocked)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--db", str(tmp_path / "x.sqlite"), "status"])
    message = str(exc.value)
    assert message.startswith("エラー: HTTP 403") and "--offline" in message


def test_explicit_model_typo_stops_before_collecting(monkeypatch, demo_db, tmp_path):
    src, _ = demo_db
    path = tmp_path / "copy.sqlite"
    shutil.copy(src, path)
    monkeypatch.chdir(tmp_path)

    def no_network(*a, **k):
        raise AssertionError("モデルの指定ミスは収集の前に知らせる")

    monkeypatch.setattr("keiba.scraper.http.Fetcher.from_settings", no_network)
    for argv in (["predict", "202301070101", "--collect", "--model", "models/v2025.joblb"],
                 ["day", "--date", "2023-01-07", "--model", "models/v2025.joblb"]):
        with pytest.raises(SystemExit, match="モデルファイルが見つかりません"):
            cli.main(["--db", str(path), *argv])


def test_status_shows_per_db_model_dir(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cli.main(["--db", str(tmp_path / "real.sqlite"), "status"])
    out = capsys.readouterr().out
    assert "最新モデル: なし" in out and str(tmp_path / "models" / "real") in out
