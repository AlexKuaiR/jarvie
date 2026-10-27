"""The job queue between the voice bot and the background worker.

The bot only inserts rows (enqueue) and reads them (get). The worker, a separate
process, claims queued rows one at a time, runs them, and writes the result back.
Using a table as the queue means no extra service to run: both processes just share
the SQLite file. Shared by every feature with slow, model-driven work (career search,
outreach), so none of them depends on another.
"""

import time
from datetime import datetime

from .config import TZ
from .storage import connect_with, dumps, loads

# kind -> which feature runs it. Career kinds search for postings; outreach kinds find
# people and emails and draft messages, and work without any posting involved.
KINDS = {
    "discover_postings": "career",
    "score_posting": "career",
    "write_document": "career",
    "find_contacts": "outreach",
    "find_email": "outreach",
    "draft_outreach": "outreach",
}

JOBS_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,           -- JSON dict
    status TEXT NOT NULL DEFAULT 'queued',   -- queued, running, done, failed
    result TEXT,
    error TEXT,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
-- One row, written every few seconds by a running worker, so the bot and dashboard
-- can tell you when jobs are queued with nothing to run them.
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    at REAL                         -- unix time; NULL after a clean shutdown
);
"""

HEARTBEAT_EVERY = 10.0  # seconds between heartbeats
HEARTBEAT_STALE = 30.0  # no heartbeat for this long = the worker isn't running
START_WORKER = "cd server && .venv/bin/python -m jarvie.worker"


def connect():
    return connect_with(JOBS_SCHEMA)


def _now() -> str:
    return datetime.now(TZ).isoformat()


def _decode(r) -> dict:
    out = dict(r)
    out["params"] = loads(out["params"], {})
    out["result"] = loads(out["result"])
    return out


def enqueue(kind: str, params: dict | None = None) -> int:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {sorted(KINDS)}")
    with connect() as conn:
        job_id = conn.execute(
            "INSERT INTO jobs (kind, params, created_at) VALUES (?, ?, ?)",
            (kind, dumps(params or {}), _now()),
        ).lastrowid
    # Inserted first, so a worker that's about to go idle sees this job and stays.
    from .autostart import ensure_worker

    ensure_worker()
    return job_id


def get(job_id: int) -> dict | None:
    with connect() as conn:
        r = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _decode(r) if r else None


def claim_next() -> dict | None:
    """Mark the oldest queued job as running and return it, or None if the queue is empty."""
    # One UPDATE ... RETURNING statement claims the row atomically, so two workers
    # running at once can never pick up the same job.
    with connect() as conn:
        r = conn.execute(
            """UPDATE jobs SET status = 'running', started_at = ?
               WHERE id = (SELECT id FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1)
               RETURNING *""",
            (_now(),),
        ).fetchone()
    return _decode(r) if r else None


def queued() -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id").fetchall()
    return [_decode(r) for r in rows]


def claim(job_id: int) -> dict | None:
    """Claim one specific queued job (None if another worker got it first)."""
    with connect() as conn:
        r = conn.execute(
            """UPDATE jobs SET status = 'running', started_at = ?
               WHERE id = ? AND status = 'queued' RETURNING *""",
            (_now(), job_id),
        ).fetchone()
    return _decode(r) if r else None


def retry(job_id: int) -> bool:
    """Queue a failed job again with the same parameters (e.g. after fixing a key)."""
    with connect() as conn:
        cur = conn.execute(
            """UPDATE jobs SET status = 'queued', error = NULL, result = NULL,
               started_at = NULL, finished_at = NULL WHERE id = ? AND status = 'failed'""",
            (job_id,),
        )
        return cur.rowcount > 0


def requeue(job_id: int) -> None:
    """Put a running job back in the queue (e.g. a plan limit was hit before it ran)."""
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'queued', started_at = NULL WHERE id = ? AND status = 'running'",
            (job_id,),
        )


def finish(
    job_id: int,
    result: dict,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float = 0.0,
) -> None:
    with connect() as conn:
        conn.execute(
            """UPDATE jobs SET status = 'done', result = ?, input_tokens = ?,
               output_tokens = ?, cost_usd = ?, finished_at = ? WHERE id = ?""",
            (dumps(result), input_tokens, output_tokens, cost_usd, _now(), job_id),
        )


def fail(job_id: int, error: str, cost_usd: float = 0.0) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, cost_usd = ?, finished_at = ? WHERE id = ?",
            (error, cost_usd, _now(), job_id),
        )


def recent(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_decode(r) for r in rows]


def spent_today() -> float:
    """Dollars spent on jobs started today, for the daily cost cap."""
    today = datetime.now(TZ).date().isoformat()
    with connect() as conn:
        return conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM jobs WHERE created_at >= ?", (today,)
        ).fetchone()[0]


def queued_count() -> int:
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM jobs WHERE status = 'queued'").fetchone()[0]


def heartbeat(alive: bool = True) -> None:
    """Called by the worker: alive=True while running, False when it shuts down."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO worker_heartbeat (id, at) VALUES (1, ?) "
            "ON CONFLICT (id) DO UPDATE SET at = excluded.at",
            (time.time() if alive else None,),
        )


def claim_start() -> bool:
    """Atomically take the right to start a worker: True only if none is alive or
    starting. It writes a fresh heartbeat, so a second caller a moment later gets False."""
    now = time.time()
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO worker_heartbeat (id, at) VALUES (1, ?)
               ON CONFLICT (id) DO UPDATE SET at = excluded.at
               WHERE worker_heartbeat.at IS NULL OR worker_heartbeat.at < ?""",
            (now, now - HEARTBEAT_STALE),
        )
        return cur.rowcount > 0


def worker_running() -> bool:
    with connect() as conn:
        r = conn.execute("SELECT at FROM worker_heartbeat WHERE id = 1").fetchone()
    return bool(r and r["at"] and time.time() - r["at"] < HEARTBEAT_STALE)


def worker_note() -> dict:
    """Extra fields for a tool result that queued a job: a warning if nothing will run it."""
    if worker_running():
        return {}
    from .autostart import enabled

    how = (
        "it should start on its own in a moment; if it doesn't, check "
        "~/jarvie-data/logs/worker.log or start it"
        if enabled()
        else "start it"
    )
    return {
        "worker_running": False,
        "tell_user": f"The background worker isn't running, so this is queued; {how}: "
        f"{START_WORKER}",
    }
