"""Helpers for feature packages that keep their own tables in the shared database.

Each feature (career, outreach, the job queue, briefs) declares its CREATE TABLE
statements next to its code and opens the database through connect_with(), so it
stays self-contained instead of every table piling up in db.SCHEMA.
"""

import json

from . import db


def connect_with(schema: str):
    """The shared Jarvie database, with `schema`'s tables created if missing."""
    conn = db.connect()
    conn.executescript(schema)
    return conn


def dumps(value) -> str | None:
    return None if value is None else json.dumps(value)


def loads(text: str | None, default=None):
    return default if text is None else json.loads(text)
