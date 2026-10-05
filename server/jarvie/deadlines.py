from datetime import datetime, timedelta

from .config import TZ
from .db import connect


def add(
    name: str,
    date: str,
    course: str | None = None,
    kind: str = "exam",
    weight: float | None = None,
    prep_hours_needed: float | None = None,
    topics: str | None = None,
) -> int:
    """
    Insert a deadline and returns a

    args:
        name: human label, e.g. Midterm 1
        date: "YYYY-MM-DD"
    """
    with connect() as conn:
        cur = conn.execute(
            """INSERT INTO deadlines
               (course, name, kind, date, weight, prep_hours_needed, topics)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (course, name, kind, date, weight, prep_hours_needed, topics),
        )
        return cur.lastrowid


# outputs upcoming events within 7 days that are not muted, muted is set as a date to mute by
# muted can be bypassed if no longer muted or if the deadline is within a day or if function
# is called in a question for my upcoming rather than on startup
def upcoming(days: int = 7, include_muted: bool = False) -> list[dict]:
    today = datetime.now(TZ).date()
    horizon = today + timedelta(days=days)
    now_iso = datetime.now(TZ).isoformat()
    sql = "SELECT * FROM deadlines WHERE completed_at is NULL AND date >= ? AND date <= ?"
    args = [today.isoformat(), horizon.isoformat()]

    if not include_muted:
        sql += """ AND (muted_until is NULL
                        OR muted_until < ?
                        OR date <= ?)"""
        args += [now_iso, (today + timedelta(days=1)).isoformat()]

    sql += " ORDER BY date"

    with connect() as conn:
        rows = conn.execute(sql, args).fetchall()
        return [
            {
                "id": r["id"],
                "course": r["course"],
                "name": r["name"],
                "kind": r["kind"],
                "date": r["date"],
                "weight": r["weight"],
                "topics": r["topics"],
                "days_away": (datetime.fromisoformat(r["date"]).date() - today).days,
            }
            for r in rows
        ]


# updates deadlines db
def update(item_id: int, **fields) -> bool:
    allowed = {"course", "name", "kind", "date", "weight", "topics", "prep_hours_needed"}

    sets, values = [], []
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"cannot update {key}")
        sets.append(f"{key} = ?")
        values.append(value)

    if not sets:
        return False

    values.append(item_id)
    sql = f"UPDATE deadlines SET {', '.join(sets)} WHERE id = ?"
    with connect() as conn:
        cur = conn.execute(sql, values)
        return cur.rowcount > 0


# i can tell the model that a certain task is complete
def complete(item_id: int) -> bool:
    now = datetime.now(TZ).isoformat()
    with connect() as conn:
        return (
            conn.execute(
                "UPDATE deadlines SET completed_at = ? WHERE id = ?", (now, item_id)
            ).rowcount
            > 0
        )


# mute a notification
def mute(item_id: int, days: int = 3) -> bool:
    mute_until = datetime.now(TZ) + timedelta(days=days)
    with connect() as conn:
        return (
            conn.execute(
                "UPDATE deadlines SET muted_until = ? WHERE id = ?",
                (mute_until.isoformat(), item_id),
            ).rowcount
            > 0
        )

# get overdue assignments
def overdue() -> list[dict]:
    # Compare dates, not datetimes: stored dates are "YYYY-MM-DD", and as strings
    # "2026-10-04" < "2026-10-04T17:00..." would make anything due today look overdue.
    today = datetime.now(TZ).date()
    sql = "SELECT * FROM deadlines WHERE completed_at is NULL AND date < ? ORDER BY date"
    with connect() as conn:
        rows = conn.execute(sql, (today.isoformat(),)).fetchall()
        return [
            {
                "id": r["id"],
                "course": r["course"],
                "name": r["name"],
                "kind": r["kind"],
                "date": r["date"],
                "weight": r["weight"],
                "topics": r["topics"],
                "days_overdue": (today - datetime.fromisoformat(r["date"]).date()).days,
            }
            for r in rows
        ]
