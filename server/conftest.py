# pytest puts this file's directory (server/) on sys.path, so tests can `import jarvis`
# when run from the repo root.
import pytest

from jarvis import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point jarvis.db at a throwaway SQLite file so tests never touch ~/jarvis-data."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    return path
