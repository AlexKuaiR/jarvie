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
CREATE TABLE IF NOT EXISTS dev_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    category TEXT DEFAULT 'feature',
    created_at TEXT NOT NULL,
    done_at TEXT
);
-- Columns mirror Semantic Scholar's paper object so S2 results can land here unchanged.
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,   -- "manual:<n>" for hand-entered, S2 paperId otherwise
    title TEXT NOT NULL,
    authors TEXT,                -- JSON list of names
    year INTEGER,
    venue TEXT,
    abstract TEXT,
    doi TEXT,
    url TEXT,
    source TEXT NOT NULL DEFAULT 'manual',
    notes TEXT,
    added_at TEXT NOT NULL
);
-- Full-text index over papers. The triggers keep it in sync, so code only writes to papers.
CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, authors, content='papers', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract, authors)
    VALUES (new.rowid, new.title, new.abstract, new.authors);
END;
CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors)
    VALUES ('delete', old.rowid, old.title, old.abstract, old.authors);
END;
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract, authors)
    VALUES ('delete', old.rowid, old.title, old.abstract, old.authors);
    INSERT INTO papers_fts(rowid, title, abstract, authors)
    VALUES (new.rowid, new.title, new.abstract, new.authors);
END;
"""


def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    return conn
