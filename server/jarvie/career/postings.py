from datetime import datetime

from ..config import TZ
from ..storage import dumps, loads
from .schema import connect

KINDS = ("internship", "research")
STATUSES = ("interested", "applied", "oa", "interviewing", "offer", "rejected", "withdrawn")

# Columns a source or the research agent may set. JSON_FIELDS hold lists.
POSTING_FIELDS = {
    "url",
    "source",
    "ats",
    "company",
    "title",
    "kind",
    "location",
    "deadline",
    "fields",
    "paid",
    "pay",
    "grad_window",
    "prestige",
    "prestige_reason",
    "competitiveness",
    "competitiveness_reason",
    "ownership",
    "ownership_reason",
    "skills",
    "process",
    "fit_score",
    "fit_reasons",
    "raw_text",
    "resume",
    "found_by_job",
}
JSON_FIELDS = {"fields", "skills", "process"}
SCORE_FIELDS = ("prestige", "competitiveness", "ownership")


def _now() -> str:
    return datetime.now(TZ).isoformat()


def _encode(fields: dict) -> dict:
    unknown = set(fields) - POSTING_FIELDS
    if unknown:
        raise ValueError(f"unknown posting fields: {sorted(unknown)}")
    if "kind" in fields and fields["kind"] not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    for key in SCORE_FIELDS:
        value = fields.get(key)
        if value is not None and not 1 <= value <= 5:
            raise ValueError(f"{key} must be 1-5")
    if "paid" in fields and fields["paid"] is not None:
        fields["paid"] = int(bool(fields["paid"]))
    return {k: dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()}


def _decode(r) -> dict:
    out = dict(r)
    for key in JSON_FIELDS:
        out[key] = loads(out[key], [])
    out["paid"] = None if out["paid"] is None else bool(out["paid"])
    return out


def save(company: str, title: str, source: str, **fields) -> int:
    """Insert a posting, or update the one with the same URL, and return its id.

    Updating only overwrites the fields passed in, so re-scoring a posting
    doesn't erase what a source filled in earlier.
    """
    if not company or not title:
        raise ValueError("a posting needs a company and a title")
    values = _encode({"company": company, "title": title, "source": source, **fields})
    with connect() as conn:
        url = values.get("url")
        existing = (
            conn.execute("SELECT id FROM postings WHERE url = ?", (url,)).fetchone()
            if url
            else None
        )
        if existing:
            sets = ", ".join(f"{k} = ?" for k in values)
            conn.execute(
                f"UPDATE postings SET {sets} WHERE id = ?", (*values.values(), existing["id"])
            )
            return existing["id"]
        values["found_at"] = _now()
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        cur = conn.execute(
            f"INSERT INTO postings ({cols}) VALUES ({marks})", tuple(values.values())
        )
        return cur.lastrowid


def update(posting_id: int, **fields) -> bool:
    """Change fields of a saved posting by id (works for postings without a URL)."""
    if not fields:
        return False
    values = _encode(fields)
    sets = ", ".join(f"{k} = ?" for k in values)
    with connect() as conn:
        cur = conn.execute(
            f"UPDATE postings SET {sets} WHERE id = ?", (*values.values(), posting_id)
        )
        return cur.rowcount > 0


def set_resume(posting_id: int, name: str | None, only_if_unset: bool = False) -> bool:
    """Pick the resume a posting uses (None = back to best match by fields).

    only_if_unset is for the research agent's suggestion, so it never overrides yours.
    """
    sql = "UPDATE postings SET resume = ? WHERE id = ?"
    if only_if_unset:
        sql += " AND resume IS NULL"
    with connect() as conn:
        return conn.execute(sql, (name, posting_id)).rowcount > 0


def get(posting_id: int) -> dict | None:
    with connect() as conn:
        r = conn.execute(
            """SELECT postings.*, applications.status, applications.has_coffee_chat
               FROM postings LEFT JOIN applications ON applications.posting_id = postings.id
               WHERE postings.id = ?""",
            (posting_id,),
        ).fetchone()
    if r is None:
        return None
    out = _decode(r)
    out["has_coffee_chat"] = bool(out["has_coffee_chat"])
    return out


SORTS = {
    "fit": "fit_score IS NULL, fit_score DESC",
    "deadline": "deadline IS NULL, deadline",
    "newest": "found_at DESC, postings.id DESC",
}


def search(
    kind: str | None = None,
    field: str | None = None,
    status: str | None = None,
    min_fit: float | None = None,
    sort: str = "fit",
    limit: int = 50,
) -> list[dict]:
    """Postings matching every filter given. status "none" means not acted on yet."""
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {sorted(SORTS)}")
    sql = """SELECT postings.*, applications.status, applications.has_coffee_chat
             FROM postings LEFT JOIN applications ON applications.posting_id = postings.id
             WHERE 1 = 1"""
    args: list = []
    if kind:
        sql += " AND kind = ?"
        args.append(kind)
    if field:
        # fields is a JSON list; json_each lets SQLite look inside it.
        sql += " AND EXISTS (SELECT 1 FROM json_each(postings.fields) WHERE value = ?)"
        args.append(field)
    if status == "none":
        sql += " AND applications.status IS NULL"
    elif status:
        sql += " AND applications.status = ?"
        args.append(status)
    if min_fit is not None:
        sql += " AND fit_score >= ?"
        args.append(min_fit)
    sql += f" ORDER BY {SORTS[sort]} LIMIT ?"
    args.append(limit)
    with connect() as conn:
        rows = conn.execute(sql, args).fetchall()
    out = []
    for r in rows:
        item = _decode(r)
        item["has_coffee_chat"] = bool(item["has_coffee_chat"])
        out.append(item)
    return out


def set_status(posting_id: int, status: str, note: str | None = None) -> int:
    """Record where an application stands, creating it on first use. Returns its id."""
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    now = _now()
    with connect() as conn:
        if conn.execute("SELECT 1 FROM postings WHERE id = ?", (posting_id,)).fetchone() is None:
            raise ValueError(f"no posting {posting_id}")
        app = conn.execute(
            "SELECT id FROM applications WHERE posting_id = ?", (posting_id,)
        ).fetchone()
        if app is None:
            app_id = conn.execute(
                """INSERT INTO applications (posting_id, status, notes, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (posting_id, status, note, now, now),
            ).lastrowid
        else:
            app_id = app["id"]
            conn.execute(
                "UPDATE applications SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, app_id),
            )
        conn.execute(
            "INSERT INTO status_events (application_id, status, note, at) VALUES (?, ?, ?, ?)",
            (app_id, status, note, now),
        )
        return app_id


def set_coffee_chat(posting_id: int, had_chat: bool = True) -> bool:
    """Mark whether there's been a coffee chat for this posting's company or team."""
    with connect() as conn:
        cur = conn.execute(
            "UPDATE applications SET has_coffee_chat = ?, updated_at = ? WHERE posting_id = ?",
            (int(had_chat), _now(), posting_id),
        )
        return cur.rowcount > 0


def history(posting_id: int) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT status_events.status, status_events.note, status_events.at
               FROM status_events JOIN applications ON applications.id = application_id
               WHERE applications.posting_id = ? ORDER BY status_events.id""",
            (posting_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def pipeline_summary() -> dict:
    """How many applications are at each status, plus how many have had a coffee chat."""
    with connect() as conn:
        counts = {
            r["status"]: r["n"]
            for r in conn.execute(
                "SELECT status, COUNT(*) AS n FROM applications GROUP BY status"
            ).fetchall()
        }
        chats = conn.execute(
            "SELECT COUNT(*) FROM applications WHERE has_coffee_chat = 1"
        ).fetchone()[0]
    return {"by_status": {s: counts.get(s, 0) for s in STATUSES}, "coffee_chats": chats}
