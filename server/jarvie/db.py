import sqlite3

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY, value TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS sleep (
    date TEXT PRIMARY KEY, total_min INTEGER, hrv REAL
);
CREATE TABLE IF NOT EXISTS deadlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    course TEXT,
    name TEXT NOT NULL,
    kind TEXT DEFAULT 'exam',
    date TEXT NOT NULL,
    weight REAL,
    prep_hours_needed REAL,
    muted_until TEXT,
    topics TEXT,
    completed_at TEXT
);
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn
