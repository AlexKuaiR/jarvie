# pytest puts this file's directory (server/) on sys.path, so tests can `import jarvie`
# when run from the repo root.
import pytest

from jarvie import db


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Point jarvie.db at a throwaway SQLite file so tests never touch ~/jarvie-data."""
    path = tmp_path / "test.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


@pytest.fixture(autouse=True)
def no_real_cli_models(monkeypatch):
    """Tests must never run the real `claude` or `codex` (they'd use your plans).

    Any job that reaches a subscription backend without a fake fails clearly instead.
    """
    from jarvie import cli_models
    from jarvie.llm import LLMError

    class Blocked:
        def __init__(self, model=None, **kwargs):
            self.name = "blocked"

        def complete(self, *args, **kwargs):
            raise LLMError("real CLI models are disabled in tests; pass a fake")

    monkeypatch.setattr(cli_models, "BACKENDS", {"claude-code": Blocked, "codex": Blocked})
