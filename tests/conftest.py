import pytest

from keiba import db, demo


@pytest.fixture(scope="session")
def demo_db(tmp_path_factory):
    """小さめの架空データDB (約14週分 + 次の開催日の出馬表)."""
    path = tmp_path_factory.mktemp("demo") / "demo.sqlite"
    conn = db.connect(path)
    upcoming = demo.generate(conn, seed=7, weeks=14, progress=lambda *_: None)
    conn.close()
    return path, upcoming
