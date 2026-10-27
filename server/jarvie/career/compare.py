"""Score the same saved postings with two (or more) backends and compare, side by side.

    .venv/bin/python -m jarvie.career.compare            # 10 postings, codex vs claude-code:sonnet
    .venv/bin/python -m jarvie.career.compare --n 5 --backends codex claude-code:opus
    .venv/bin/python -m jarvie.career.compare --ids 20 23

Each backend gets the exact prompt real scoring uses, but its save_posting calls are
only recorded here, never written to your postings, so nothing you've scored changes.
Results are on the dashboard's Compare page (http://localhost:7861/compare); judge
which backend's scores and reasons read truer, then set [models] research to it.
"""

import inspect
import time
import uuid
from datetime import datetime

from loguru import logger

from ..cli_models import make_agent
from ..config import TZ
from ..llm import LLMError
from ..storage import connect_with, dumps, loads
from . import postings
from .profile import CareerProfile
from .research_agent import SaveLog, make_save_posting, scoring_prompt, system_prompt

COMPARE_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,          -- one comparison run: the same postings, several backends
    backend TEXT NOT NULL,         -- a [models] spec, e.g. 'codex' or 'claude-code:sonnet'
    posting_id INTEGER NOT NULL,
    scores TEXT,                   -- JSON: the save_posting arguments, or NULL if none came back
    seconds REAL,                  -- time for this backend's whole run
    error TEXT,
    created_at TEXT NOT NULL
);
"""

DEFAULT_BACKENDS = ["codex", "claude-code:sonnet"]


def connect():
    return connect_with(COMPARE_SCHEMA)


def recording_tool(sink: dict[int, dict], resume_names: tuple[str, ...]):
    """A save_posting with the real one's signature and docs that only records."""
    real = make_save_posting(SaveLog(), None, "web", resume_names)

    def save_posting(**kwargs) -> str:
        if kwargs.get("posting_id") is None:
            return "error: include the posting_id from the prompt"
        sink[int(kwargs["posting_id"])] = kwargs
        return "recorded"

    save_posting.__signature__ = inspect.signature(real)
    save_posting.__annotations__ = real.__annotations__
    save_posting.__doc__ = real.__doc__
    return save_posting


def pick(n: int, ids: list[int] | None = None) -> list[dict]:
    """The postings to compare: the ids given, or the newest n with a description."""
    if ids:
        found = [postings.get(i) for i in ids]
        missing = [i for i, p in zip(ids, found) if p is None]
        if missing:
            raise ValueError(f"no postings {missing}")
        return found
    items = [p for p in postings.search(sort="newest", limit=500) if p["raw_text"]]
    return items[:n]


def run(
    profile: CareerProfile, backends: list[str], items: list[dict], agents: dict | None = None
) -> str:
    """Score `items` with each backend and store the results. Returns the run id.

    agents maps a backend spec to an AgentModel (tests); otherwise each is built.
    """
    run_id = uuid.uuid4().hex[:8]
    system = system_prompt(profile)
    now = datetime.now(TZ).isoformat()
    for spec in backends:
        sink: dict[int, dict] = {}
        tool = recording_tool(sink, tuple(profile.resumes))
        agent = (agents or {}).get(spec) or make_agent(spec)
        started, error = time.time(), None
        try:
            for start in range(0, len(items), 8):
                agent.run(system, scoring_prompt(items[start : start + 8]), [tool], web_search=4)
        except LLMError as e:
            error = str(e)
        seconds = round(time.time() - started, 1)
        logger.info("compare {}: {} of {} scored in {}s", spec, len(sink), len(items), seconds)
        with connect() as conn:
            conn.executemany(
                """INSERT INTO model_comparisons
                   (run_id, backend, posting_id, scores, seconds, error, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                [
                    (run_id, spec, p["id"], dumps(sink.get(p["id"])), seconds, error, now)
                    for p in items
                ],
            )
    return run_id


def runs() -> list[dict]:
    """Past comparison runs, newest first."""
    with connect() as conn:
        rows = conn.execute(
            """SELECT run_id, MIN(created_at) AS at, GROUP_CONCAT(DISTINCT backend) AS backends,
                      COUNT(DISTINCT posting_id) AS postings
               FROM model_comparisons GROUP BY run_id ORDER BY MIN(id) DESC"""
        ).fetchall()
    return [dict(r) for r in rows]


def table(run_id: str) -> dict:
    """One run as {backends, rows: [{posting, by_backend: {spec: scores}}], errors, seconds}."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM model_comparisons WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
    backends = list(dict.fromkeys(r["backend"] for r in rows))
    by_posting: dict[int, dict] = {}
    errors, seconds = {}, {}
    for r in rows:
        by_posting.setdefault(r["posting_id"], {})[r["backend"]] = loads(r["scores"])
        if r["error"]:
            errors[r["backend"]] = r["error"]
        seconds[r["backend"]] = r["seconds"]
    return {
        "backends": backends,
        "rows": [
            {"posting": postings.get(pid), "by_backend": scores}
            for pid, scores in by_posting.items()
        ],
        "errors": errors,
        "seconds": seconds,
    }


def main() -> None:
    import argparse

    from .profile import load

    parser = argparse.ArgumentParser(description="Compare backends on the same postings.")
    parser.add_argument(
        "--backends", nargs="+", default=DEFAULT_BACKENDS, help="[models] specs to compare"
    )
    parser.add_argument("--n", type=int, default=10, help="how many postings (newest first)")
    parser.add_argument("--ids", type=int, nargs="*", help="specific posting ids instead")
    args = parser.parse_args()
    items = pick(args.n, args.ids)
    if not items:
        raise SystemExit("no saved postings with a description; run jarvie.career.ingest first")
    print(f"scoring {len(items)} postings with {', '.join(args.backends)} (this takes a while)")
    run_id = run(load(), args.backends, items)
    t = table(run_id)
    for spec in t["backends"]:
        done = sum(1 for r in t["rows"] if r["by_backend"].get(spec))
        extra = f", error: {t['errors'][spec]}" if spec in t["errors"] else ""
        print(f"  {spec}: {done}/{len(items)} scored in {t['seconds'][spec]}s{extra}")
    print(f"Side by side: http://localhost:7861/compare?run={run_id}")


if __name__ == "__main__":
    main()
