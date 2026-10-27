from ..storage import connect_with

# Postings and application tracking. People and emails live in jarvie/outreach, which
# works on its own; an email may point at a posting here, but doesn't have to.
# List-valued columns (fields, skills, process) are stored as JSON text.
CAREER_SCHEMA = """
CREATE TABLE IF NOT EXISTS postings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE,
    source TEXT NOT NULL,           -- 'greenhouse', 'lever', 'ashby', 'simplify', 'handshake', 'web'
    ats TEXT,                       -- how you apply: 'greenhouse', 'workday', 'handshake', ...
    company TEXT NOT NULL,          -- company or lab
    title TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'internship',   -- 'internship' or 'research'
    location TEXT,
    deadline TEXT,                  -- 'YYYY-MM-DD'
    fields TEXT,                    -- JSON list, e.g. ["machine learning"]
    paid INTEGER,                   -- 1, 0, or NULL when unknown
    pay TEXT,
    grad_window TEXT,               -- graduation dates the posting accepts, as written
    prestige INTEGER, prestige_reason TEXT,               -- 1-5 rubric scores
    competitiveness INTEGER, competitiveness_reason TEXT,
    ownership INTEGER, ownership_reason TEXT,
    skills TEXT,                    -- JSON list
    process TEXT,                   -- JSON list of steps, e.g. ["oa", "essays"]
    fit_score REAL, fit_reasons TEXT,
    raw_text TEXT,
    found_at TEXT NOT NULL,
    found_by_job INTEGER
);
CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    posting_id INTEGER NOT NULL UNIQUE REFERENCES postings(id),
    status TEXT NOT NULL,
    has_coffee_chat INTEGER NOT NULL DEFAULT 0,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS status_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER NOT NULL REFERENCES applications(id),
    status TEXT NOT NULL,
    note TEXT,
    at TEXT NOT NULL
);
"""


def connect():
    return connect_with(CAREER_SCHEMA)
