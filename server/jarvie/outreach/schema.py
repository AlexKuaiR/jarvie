from ..storage import connect_with

OUTREACH_SCHEMA = """
CREATE TABLE IF NOT EXISTS contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    company TEXT,                   -- company, lab, school or organization; optional
    role TEXT,
    is_recruiter_or_admissions INTEGER NOT NULL DEFAULT 0,
    alumni_of TEXT,                 -- JSON list: "berkeley", "nnhs"
    source TEXT NOT NULL DEFAULT 'agent',   -- 'agent' (found by research) or 'user' (you added)
    evidence_urls TEXT NOT NULL,    -- JSON list; required for agent-found contacts
    email TEXT UNIQUE,
    email_source TEXT,              -- e.g. 'hunter', 'user'
    email_verified TEXT,            -- verifier result, e.g. 'valid', 'accept_all'
    features TEXT,                  -- JSON dict the writer personalizes from
    notes TEXT,
    do_not_contact INTEGER NOT NULL DEFAULT 0,
    found_at TEXT NOT NULL,
    found_by_job INTEGER
);
CREATE TABLE IF NOT EXISTS outreach (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    contact_id INTEGER NOT NULL REFERENCES contacts(id),
    purpose TEXT NOT NULL,          -- 'coffee_chat', 'research_inquiry', 'referral', 'advice', 'other'
    posting_id INTEGER,             -- optional link to a career posting; no foreign key on
                                    -- purpose, so outreach works without the career tables
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'drafted',
    scheduled_at TEXT,
    sent_at TEXT,
    gmail_message_id TEXT,
    gmail_thread_id TEXT,
    follow_ups INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
-- At most one live email thread per person: the database itself refuses a second one,
-- so a bug elsewhere can't double-email anyone.
CREATE UNIQUE INDEX IF NOT EXISTS outreach_one_active_per_contact
    ON outreach(contact_id) WHERE status IN ('drafted', 'approved', 'scheduled', 'sent');
"""


def connect():
    return connect_with(OUTREACH_SCHEMA)
